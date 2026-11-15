#ifndef SINKHORN_NORM_KERNELS_F13_WIDE_COMPACT_FIVE_H_
#define SINKHORN_NORM_KERNELS_F13_WIDE_COMPACT_FIVE_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f13 {

using namespace AscendC;

static __aicore__ __attribute__((always_inline)) inline void WideRoundMask(uint64_t mask) {
    SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, mask);
}

template<int Op>
static __aicore__ __attribute__((always_inline)) inline void CompactCountSpan(
    __ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b,
    uint32_t count, float scalar = 0.0f) {
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, count);
    if constexpr (Op == 0) vadd(dst, a, b, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 1) vadds(dst, a, scalar, 1, 1, 1, 8, 8);
    if constexpr (Op == 2) vdiv(dst, a, b, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst),
                                 reinterpret_cast<__ubuf__ uint32_t *>(a), 1, 1, 1, 8, 8);
    SetMaskNorm();
    WideRoundMask(~uint64_t{0});
}

struct CapacityFiveGeometry {
    uint32_t h, p, v, q, qr, rows, span, blocks, slot, p0, p1;
    uint64_t last;
    __aicore__ inline void Init(uint32_t height, uint32_t pitch, uint32_t valid,
                                uint32_t slotFloats, uint32_t phase0, uint32_t phase1) {
        h = height; p = pitch; v = valid; q = (h + 3) / 4; qr = (q * v + 7) / 8 * 8;
        rows = 4 * qr; span = p * v; blocks = p / 8; slot = slotFloats; p0 = phase0; p1 = phase1;
        last = ((uint64_t{1} << (h - (blocks - 1) * 8)) - 1) * uint64_t{0x0101010101010101};
    }
};

__aicore__ __attribute__((always_inline)) inline uint32_t QuarterNode(uint32_t s, uint32_t j) {
    return (s >> (3 * j)) & 7;
}

__aicore__ __attribute__((always_inline)) inline uint32_t QuarterPack(
    uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    return a | (b << 3) | (c << 6) | (d << 9) | (e << 12);
}

__aicore__ __attribute__((always_inline)) inline uint32_t QuarterAdvance(uint32_t s) {
    return QuarterPack(QuarterNode(s, 4), QuarterNode(s, 3), QuarterNode(s, 0),
                       QuarterNode(s, 2), QuarterNode(s, 1));
}

__aicore__ __attribute__((always_inline)) inline __ubuf__ float *CapacityFiveView(
    __ubuf__ float *m, __ubuf__ float *s, uint32_t node, uint32_t logical,
    const CapacityFiveGeometry &g) {
    if (node == 4) return s + (logical == 0 ? g.p0 : g.p1);
    return m + node * g.slot + (((node & 1) ^ (logical == 0 ? 0U : 1U)) ? 64 : 0);
}

__aicore__ __attribute__((always_inline)) inline __ubuf__ float *CapacityFiveColumnCoeff(
    __ubuf__ float *tmp, uint32_t logical, const CapacityFiveGeometry &g) {
    return logical == 0 ? tmp : tmp + (g.span + 127) / 128 * 128 + 64;
}

__aicore__ __attribute__((always_inline)) inline void CapacityFiveScaleColumns(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *coeff,
    uint32_t height, const CapacityFiveGeometry &g) {
    for (uint32_t col = 0; col < g.span; col += 64) {
        const uint32_t n = g.span - col < 64 ? g.span - col : 64;
        if (n < 64) WideRoundMask((uint64_t{1} << n) - 1);
        vmul(dst + col, src + col, coeff + col, height, 1, 1, 1, g.span / 8, g.span / 8, 0);
        if (n < 64) WideRoundMask(~uint64_t{0});
    }
    pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) inline void CapacityFiveInverse(
    __ubuf__ float *sum, __ubuf__ float *tmp, __ubuf__ float *ones,
    __ubuf__ float *denom, float eps, const CapacityFiveGeometry &g,
    __ubuf__ float *saved = nullptr) {
    CompactCountSpan<1>(denom, sum, sum, g.span, eps); pipe_barrier(PIPE_V);
    if (saved)
        CompactCountSpan<3>(saved, denom, denom, g.span);
    CompactCountSpan<2>(tmp, ones, denom, g.span); pipe_barrier(PIPE_V);
    CompactCountSpan<3>(tmp + (g.span + 127) / 128 * 128 + 64, tmp, tmp, g.span);
    pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) inline void CapacityFiveFoldColumns(
    __ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b,
    __ubuf__ float *c, __ubuf__ float *d, const CapacityFiveGeometry &g) {
    const uint32_t quarter = g.q * g.span;
    CompactCountSpan<0>(dst, a, b, quarter); pipe_barrier(PIPE_V);
    CompactCountSpan<0>(dst, dst, c, quarter); pipe_barrier(PIPE_V);
    CompactCountSpan<0>(dst, dst, d, (g.h - 3 * g.q) * g.span); pipe_barrier(PIPE_V);
    for (uint32_t height = g.q; height > 1;) {
        const uint32_t half = height / 2, upper = height - half;
        CompactCountSpan<0>(dst, dst, dst + upper * g.span, half * g.span);
        pipe_barrier(PIPE_V); height = upper;
    }
}

template<bool Maximum>
__aicore__ __attribute__((always_inline)) inline void CollectQuarterPlanes(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *scratch, const CapacityFiveGeometry &g) {
    auto pack = scratch + 8 * g.rows;
    if constexpr (Maximum) {
        vcgmax(pack, a, g.qr, 1, 1, g.blocks);
        vcgmax(pack + g.qr * 8, b, g.qr, 1, 1, g.blocks);
        vcgmax(pack + 2 * g.qr * 8, c, g.qr, 1, 1, g.blocks);
        vcgmax(pack + 3 * g.qr * 8, d, g.qr, 1, 1, g.blocks);
    } else {
        vcgadd(pack, a, g.qr, 1, 1, g.blocks);
        vcgadd(pack + g.qr * 8, b, g.qr, 1, 1, g.blocks);
        vcgadd(pack + 2 * g.qr * 8, c, g.qr, 1, 1, g.blocks);
        vcgadd(pack + 3 * g.qr * 8, d, g.qr, 1, 1, g.blocks);
    }
    for (uint32_t t = 0; t + 8 < g.blocks; ++t) {
        const uint32_t block = 8 + t;
        if (block + 1 == g.blocks) WideRoundMask(g.last);
        auto out = scratch + (t + 1) * g.rows;
        if constexpr (Maximum) {
            vcgmax(out, a + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgmax(out + g.qr, b + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgmax(out + 2 * g.qr, c + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgmax(out + 3 * g.qr, d + block * 8, g.qr / 8, 1, g.blocks, g.p);
        } else {
            vcgadd(out, a + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgadd(out + g.qr, b + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgadd(out + 2 * g.qr, c + block * 8, g.qr / 8, 1, g.blocks, g.p);
            vcgadd(out + 3 * g.qr, d + block * 8, g.qr / 8, 1, g.blocks, g.p);
        }
        if (block + 1 == g.blocks) WideRoundMask(~uint64_t{0});
    }
    pipe_barrier(PIPE_V);
    if constexpr (Maximum) {
        vcgmax(scratch, pack, g.qr / 8, 1, 1, 8);
        vcgmax(scratch + g.qr, pack + g.qr * 8, g.qr / 8, 1, 1, 8);
        vcgmax(scratch + 2 * g.qr, pack + 2 * g.qr * 8, g.qr / 8, 1, 1, 8);
        vcgmax(scratch + 3 * g.qr, pack + 3 * g.qr * 8, g.qr / 8, 1, 1, 8);
    } else {
        vcgadd(scratch, pack, g.qr / 8, 1, 1, 8);
        vcgadd(scratch + g.qr, pack + g.qr * 8, g.qr / 8, 1, 1, 8);
        vcgadd(scratch + 2 * g.qr, pack + 2 * g.qr * 8, g.qr / 8, 1, 1, 8);
        vcgadd(scratch + 3 * g.qr, pack + 3 * g.qr * 8, g.qr / 8, 1, 1, 8);
    }
    pipe_barrier(PIPE_V);
}

template<bool Maximum>
__aicore__ __attribute__((always_inline)) inline void QuarterRowReduction(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *scratch, __ubuf__ float *result, const CapacityFiveGeometry &g) {
    CollectQuarterPlanes<Maximum>(a, b, c, d, scratch, g);
    uint32_t planes = g.blocks - 7;
    while (planes > 1) {
        const uint32_t pairs = planes / 2, upper = planes - pairs;
        auto dst = planes == 2 ? result : scratch;
        const uint32_t count = planes == 2 ? g.rows : pairs * g.rows;
        if constexpr (Maximum) {
            SetMaskCount(); SetVectorMask<float, MaskMode::COUNTER>(0, count);
            vmax(dst, scratch, scratch + upper * g.rows, 1, 1, 1, 1, 8, 8, 8);
            SetMaskNorm(); WideRoundMask(~uint64_t{0});
        } else {
            CompactCountSpan<0>(dst, scratch, scratch + upper * g.rows, count);
        }
        pipe_barrier(PIPE_V); planes = upper;
    }
}

// One repeat covers eight contiguous column blocks of a single token row.
// The next token row begins g.blocks datablocks later, so a whole quarter is
// two instructions rather than one per column block.
template<bool Subtract>
__aicore__ __attribute__((always_inline)) inline void ApplyPackedRows(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *coeff,
    const CapacityFiveGeometry &g) {
    for (uint32_t base = 0; base < g.blocks; base += 8) {
        const uint32_t width = g.blocks - base > 8 ? 8 : g.blocks - base;
        if (width < 8) WideRoundMask((uint64_t{1} << (width * 8)) - 1);
        for (uint32_t row = 0; row < g.qr;) {
            const uint32_t n = g.qr - row > 255 ? 255 : g.qr - row;
            auto d = dst + row * g.p + base * 8;
            auto s = src + row * g.p + base * 8;
            auto c = coeff + row * 8;
            if constexpr (Subtract) vsub(d, s, c, n, 1, 1, 0, g.blocks, g.blocks, 1);
            else vmul(d, s, c, n, 1, 1, 0, g.blocks, g.blocks, 1);
            row += n;
        }
        if (width < 8) WideRoundMask(~uint64_t{0});
    }
}

template<bool Subtract>
__aicore__ __attribute__((always_inline)) inline void QuarterApplyFirstRows(
    __ubuf__ float *matrix, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *work, uint32_t logical, const CapacityFiveGeometry &g) {
    auto coeff = tmp + logical * g.qr * 8;
    const uint32_t sourcePhase = logical == 0 ? 0 : 256;
    if ((256 + logical * g.qr * 32) % 512 == sourcePhase) {
        coeff = work + (sourcePhase == 0 ? 64 : 0);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeff),
              reinterpret_cast<__ubuf__ uint32_t *>(result + logical * g.qr), 1, 8, g.qr / 8);
        pipe_barrier(PIPE_V);
    }
    ApplyPackedRows<Subtract>(matrix, matrix, coeff, g);
    pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) inline void QuarterFirstExp(
    __ubuf__ float *m, uint32_t count) {
    SetMaskCount(); SetVectorMask<float, MaskMode::COUNTER>(0, count);
    vexp(m, m, 1, 1, 1, 8, 8);
    SetMaskNorm(); WideRoundMask(~uint64_t{0});
}

__aicore__ __attribute__((always_inline)) inline void CapacityFiveScaleRows(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *primary,
    __ubuf__ float *result, __ubuf__ float *work, uint32_t logical, const CapacityFiveGeometry &g) {
    auto coeff = primary + logical * g.qr * 8;
    const uint32_t phase = (256 + logical * g.qr * 32) % 512;
    const uint32_t sourcePhase = logical == 0 ? 0 : 256;
    if (g.qr % 16 == 0 || phase == sourcePhase) {
        coeff = logical == 0 ? primary : work;
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeff),
              reinterpret_cast<__ubuf__ uint32_t *>(result + logical * g.qr), 1, 8, g.qr / 8);
        pipe_barrier(PIPE_V);
    }
    ApplyPackedRows<false>(dst, src, coeff, g);
    pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) inline void BroadcastQuarter(
    __ubuf__ float *dst, __ubuf__ float *result, uint32_t logical, uint32_t qr) {
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(dst),
          reinterpret_cast<__ubuf__ uint32_t *>(result + logical * qr), 1, 8, qr / 8);
}

__aicore__ __attribute__((always_inline)) inline void ScaleRowsBatched(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *empty, __ubuf__ float *flippedA, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *work, __ubuf__ float *coeffPair,
    const CapacityFiveGeometry &g) {
    auto second = coeffPair + g.qr * 8;
    BroadcastQuarter(tmp, result, 0, g.qr);
    BroadcastQuarter(work, result, 2, g.qr);
    BroadcastQuarter(coeffPair, result, 3, g.qr);
    BroadcastQuarter(second, result, 1, g.qr);
    pipe_barrier(PIPE_V);
    ApplyPackedRows<false>(empty, a, tmp, g);
    pipe_barrier(PIPE_V);
    ApplyPackedRows<false>(flippedA, c, work, g);
    pipe_barrier(PIPE_V);
    ApplyPackedRows<false>(c, d, coeffPair, g);
    pipe_barrier(PIPE_V);
    ApplyPackedRows<false>(d, b, second, g);
    pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) inline void CapacityFiveRound(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *empty, __ubuf__ float *flippedA, __ubuf__ float *flippedB,
    __ubuf__ float *flippedEmpty, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *work, __ubuf__ float *ones, __ubuf__ float *denom,
    __ubuf__ float *coeffPair, float eps, const CapacityFiveGeometry &g,
    __ubuf__ float *saved = nullptr) {
    if ASCEND_IS_AIV {
        CollectQuarterPlanes<false>(a, b, c, d, empty, g);
        uint32_t planes = g.blocks - 7;
        while (planes > 2) {
            const uint32_t pairs = planes / 2, upper = planes - pairs;
            CompactCountSpan<0>(empty, empty, empty + upper * g.rows, pairs * g.rows);
            pipe_barrier(PIPE_V); planes = upper;
        }
        CompactCountSpan<0>(result, empty, empty + g.rows, g.rows); pipe_barrier(PIPE_V);
        CompactCountSpan<1>(result, result, result, g.rows, eps); pipe_barrier(PIPE_V);
        if (saved) {
            CompactCountSpan<3>(saved, result, result, g.rows);
            pipe_barrier(PIPE_V);
        }
        CompactCountSpan<2>(result, ones, result, g.rows); pipe_barrier(PIPE_V);
        if (g.qr % 16 == 0 && coeffPair != nullptr) {
            ScaleRowsBatched(a, b, c, d, empty, flippedA, tmp, result, work, coeffPair, g);
        } else {
            if (g.qr % 16 != 0) {
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
                      reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, g.rows / 8);
                pipe_barrier(PIPE_V);
            }
            CapacityFiveScaleRows(empty, a, tmp, result, work, 0, g);
            CapacityFiveScaleRows(flippedA, c, tmp, result, work, 2, g);
            CapacityFiveScaleRows(c, d, tmp, result, work, 3, g);
            CapacityFiveScaleRows(d, b, tmp, result, work, 1, g);
        }
        CapacityFiveFoldColumns(flippedB, empty, d, flippedA, c, g);
        CapacityFiveInverse(flippedB, tmp, ones, denom, eps, g, saved ? saved + g.rows : nullptr);
        CapacityFiveScaleColumns(flippedB, empty, CapacityFiveColumnCoeff(tmp, 0, g), g.q, g);
        CapacityFiveScaleColumns(flippedEmpty, flippedA, CapacityFiveColumnCoeff(tmp, 2, g), g.q, g);
        CapacityFiveScaleColumns(flippedA, c, CapacityFiveColumnCoeff(tmp, 3, g), g.q, g);
        CapacityFiveScaleColumns(c, d, CapacityFiveColumnCoeff(tmp, 1, g), g.q, g);
    }
}

static __aicore__ __attribute__((noinline)) void RunQuarterBatchSoftmax(
    __ubuf__ float *matrix, __ubuf__ float *spill, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *work, __ubuf__ float *ones,
    float eps, uint32_t state, const CapacityFiveGeometry &g) {
    if ASCEND_IS_AIV {
        WideRoundMask(~uint64_t{0});
        auto a = CapacityFiveView(matrix, spill, QuarterNode(state, 0), 0, g);
        auto b = CapacityFiveView(matrix, spill, QuarterNode(state, 1), 1, g);
        auto c = CapacityFiveView(matrix, spill, QuarterNode(state, 2), 2, g);
        auto d = CapacityFiveView(matrix, spill, QuarterNode(state, 3), 3, g);
        auto scratch = CapacityFiveView(matrix, spill, QuarterNode(state, 4), 0, g);
        QuarterRowReduction<true>(a, b, c, d, scratch, result, g);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp), reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, g.rows / 8);
        pipe_barrier(PIPE_V);
        QuarterApplyFirstRows<true>(a, tmp, result, work, 0, g);
        QuarterApplyFirstRows<true>(b, tmp, result, work, 1, g);
        QuarterApplyFirstRows<true>(c, tmp, result, work, 2, g);
        QuarterApplyFirstRows<true>(d, tmp, result, work, 3, g);
        QuarterFirstExp(a, g.qr * g.p); QuarterFirstExp(b, g.qr * g.p);
        QuarterFirstExp(c, g.qr * g.p); QuarterFirstExp(d, g.qr * g.p); pipe_barrier(PIPE_V);
        QuarterRowReduction<false>(a, b, c, d, scratch, result, g);
        CompactCountSpan<2>(result, ones, result, g.rows); pipe_barrier(PIPE_V);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp), reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, g.rows / 8);
        pipe_barrier(PIPE_V);
        QuarterApplyFirstRows<false>(a, tmp, result, work, 0, g);
        QuarterApplyFirstRows<false>(b, tmp, result, work, 1, g);
        QuarterApplyFirstRows<false>(c, tmp, result, work, 2, g);
        QuarterApplyFirstRows<false>(d, tmp, result, work, 3, g);
        CompactCountSpan<1>(a, a, a, g.qr * g.p, eps); CompactCountSpan<1>(b, b, b, g.qr * g.p, eps);
        CompactCountSpan<1>(c, c, c, g.qr * g.p, eps); CompactCountSpan<1>(d, d, d, g.qr * g.p, eps); pipe_barrier(PIPE_V);
    }
}

constexpr uint32_t kWideForecastSteps = 6;

struct WideForecastWeights {
    float recent, older;
};

__aicore__ constexpr WideForecastWeights WideForecastWeightsAfter(uint32_t steps) {
    WideForecastWeights current{1.0f, 0.0f};
    WideForecastWeights previous{0.0f, 1.0f};
    for (uint32_t i = 0; i < steps; ++i) {
        const WideForecastWeights next{
            1.45f * current.recent - 0.51f * previous.recent,
            1.45f * current.older - 0.51f * previous.older};
        previous = current;
        current = next;
    }
    return current;
}

// Evaluate the finite denominator states independently, then reduce their
// products before applying the row and column scales to the matrix.
static __aicore__ __attribute__((noinline, section(".sinkhorn_wide_forecast"))) void ApplyWideForecast(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *empty, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *work, __ubuf__ float *ones,
    __ubuf__ float *older, uint32_t length, const CapacityFiveGeometry &g) {
    if ASCEND_IS_AIV {
        constexpr uint32_t steps = kWideForecastSteps;
        constexpr WideForecastWeights weights[] = {
            WideForecastWeightsAfter(1), WideForecastWeightsAfter(2),
            WideForecastWeightsAfter(3), WideForecastWeightsAfter(4),
            WideForecastWeightsAfter(5), WideForecastWeightsAfter(6)};
        const uint32_t repeats = length / 64;
        auto recent = older + length;
        auto product = empty;
        auto floor = empty + steps * length;
        vadds(older, older, -1.0f, repeats, 1, 1, 8, 8);
        vadds(recent, recent, -1.0f, repeats, 1, 1, 8, 8);
        vector_dup(floor, 0.25f, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        #pragma unroll
        for (uint32_t step = 0; step < steps; ++step)
            vmuls(product + step * length, older, static_cast<float>(weights[step].older),
                  repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        #pragma unroll
        for (uint32_t step = 0; step < steps; ++step)
            vaxpy(product + step * length, recent, static_cast<float>(weights[step].recent),
                  repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(product, product, 1.0f, steps * repeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(product, product, floor, steps * repeats, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        #pragma unroll
        for (uint32_t distance = 1; distance < steps; distance *= 2) {
            #pragma unroll
            for (uint32_t step = 0; step + distance < steps; step += 2 * distance) {
                auto left = product + step * length;
                vmul(left, left, left + distance * length, repeats, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
        }
        CompactCountSpan<2>(result, ones, product, g.rows);
        pipe_barrier(PIPE_V);
        CapacityFiveScaleRows(a, a, tmp, result, work, 0, g);
        CapacityFiveScaleRows(b, b, tmp, result, work, 1, g);
        CapacityFiveScaleRows(c, c, tmp, result, work, 2, g);
        CapacityFiveScaleRows(d, d, tmp, result, work, 3, g);
        auto low = CapacityFiveColumnCoeff(tmp, 0, g);
        auto high = CapacityFiveColumnCoeff(tmp, 1, g);
        CompactCountSpan<2>(low, ones, product + g.rows, g.span);
        CompactCountSpan<2>(high, ones, product + g.rows, g.span);
        pipe_barrier(PIPE_V);
        CapacityFiveScaleColumns(a, a, low, g.q, g);
        CapacityFiveScaleColumns(b, b, high, g.q, g);
        CapacityFiveScaleColumns(c, c, high, g.q, g);
        CapacityFiveScaleColumns(d, d, high, g.h - 3 * g.q, g);
    }
}

template<bool EnableForecast>
static __aicore__ __attribute__((always_inline)) inline bool RunQuarterInputFive(
    __ubuf__ float *matrix, __ubuf__ float *spill, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *work, __ubuf__ float *ones,
    __ubuf__ float *denom, __ubuf__ float *coeffPair, float eps, uint32_t rounds,
    uint32_t state, const CapacityFiveGeometry &g) {
    if ASCEND_IS_AIV {
        WideRoundMask(~uint64_t{0});
        auto a = CapacityFiveView(matrix, spill, QuarterNode(state, 0), 0, g);
        auto b = CapacityFiveView(matrix, spill, QuarterNode(state, 1), 1, g);
        auto c = CapacityFiveView(matrix, spill, QuarterNode(state, 2), 2, g);
        auto d = CapacityFiveView(matrix, spill, QuarterNode(state, 3), 3, g);
        auto empty = CapacityFiveView(matrix, spill, QuarterNode(state, 4), 0, g);
        CapacityFiveFoldColumns(empty, a, b, c, d, g);
        CapacityFiveInverse(empty, tmp, ones, denom, eps, g);
        CapacityFiveScaleColumns(empty, a, CapacityFiveColumnCoeff(tmp, 0, g), g.q, g);
        CapacityFiveScaleColumns(CapacityFiveView(matrix, spill, QuarterNode(state, 0), 1, g),
                                 c, CapacityFiveColumnCoeff(tmp, 2, g), g.q, g);
        CapacityFiveScaleColumns(c, d, CapacityFiveColumnCoeff(tmp, 3, g), g.h - 3 * g.q, g);
        CapacityFiveScaleColumns(d, b, CapacityFiveColumnCoeff(tmp, 1, g), g.q, g);
        state = QuarterAdvance(state);
        const uint32_t length = g.rows + (g.span + 63) / 64 * 64;
        const uint32_t columnTop = (g.span + 127) / 128 * 128 + 64 +
                                   (g.span + 63) / 64 * 64;
        const uint32_t historyStart = g.qr * 8 > columnTop ? g.qr * 8 : columnTop;
        const bool forecast = EnableForecast && rounds >= 30 && g.qr % 16 == 0 &&
                              eps >= 0x1p-24f && eps <= 1.0e-5f &&
                              historyStart + 2 * length <= 8 * g.rows &&
                              kWideForecastSteps * length + 64 <= g.qr * g.p;
        auto history = tmp + historyStart;
        constexpr uint32_t exactBetween = 3;
        constexpr uint32_t exactSuffix = 8;
        const uint32_t firstForecast = rounds >= 30
            ? rounds - 2 * kWideForecastSteps - exactBetween - exactSuffix : 0;
        const uint32_t secondForecast = firstForecast + kWideForecastSteps + exactBetween;
        if (forecast) {
            vector_dup(history, 1.0f, 2 * length / 64, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
        }
        for (uint32_t round = 0; round < rounds; ++round) {
            const uint32_t checkpoint = round < firstForecast ? firstForecast : secondForecast;
            if (forecast && (round == firstForecast || round == secondForecast)) {
                ApplyWideForecast(
                    CapacityFiveView(matrix, spill, QuarterNode(state, 0), 0, g),
                    CapacityFiveView(matrix, spill, QuarterNode(state, 1), 1, g),
                    CapacityFiveView(matrix, spill, QuarterNode(state, 2), 2, g),
                    CapacityFiveView(matrix, spill, QuarterNode(state, 3), 3, g),
                    CapacityFiveView(matrix, spill, QuarterNode(state, 4), 0, g),
                    tmp, result, work, ones, history, length, g);
                round += kWideForecastSteps - 1;
                continue;
            }
            __ubuf__ float *saved = nullptr;
            if (forecast && round + 2 >= checkpoint && round < checkpoint)
                saved = history + (round + 2 - checkpoint) * length;
            CapacityFiveRound(CapacityFiveView(matrix, spill, QuarterNode(state, 0), 0, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 1), 1, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 2), 2, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 3), 3, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 4), 0, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 0), 1, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 1), 0, g),
                              CapacityFiveView(matrix, spill, QuarterNode(state, 4), 1, g),
                              tmp, result, work, ones, denom, coeffPair, eps, g, saved);
            state = QuarterAdvance(QuarterAdvance(state));
        }
        return forecast;
    }
    return false;
}

class QuarterStreamSinkhorn {
public:
    __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
                                const sinkhorn_norm::TilingData &t) {
        t_ = t;
        const uint64_t core = GetBlockIdx();
        start_ = core * t.tokens_per_core + (core < t.extra_core_count ? core : t.extra_core_count);
        count_ = t.tokens_per_core + (core < t.extra_core_count ? 1 : 0);
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
        pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
        post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
        comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

        const uint32_t qr = (((t.h + 3) / 4 * t.tile_tokens + 7) / 8 * 8);
        const uint32_t rows = 4 * qr;
        const uint32_t span = t.h_pad * t.tile_tokens;
        const uint32_t tmp = 8 * rows;
        const uint32_t ones = rows > 2 * span ? rows : 2 * span;
        const uint32_t prefix = (512 - (256 + 4 * tmp + 4 * rows) % 512) % 512 / 4;
        uint32_t work = 0;
        for (uint32_t v = 1; v <= t.tile_tokens; ++v) {
            const uint32_t r = (((t.h + 3) / 4 * v + 7) / 8 * 8);
            const uint32_t n = 8 * r + (r % 16 ? 64 : 0);
            if (n > work) work = n;
        }

        pipe_.InitBuffer(matrixBuffer_, t.matrix_bytes);
        pipe_.InitBuffer(gateBuffer_, 2 * t.gate_bytes);
        pipe_.InitBuffer(partialBuffer_, t.input_bytes);
        pipe_.InitBuffer(tmpBuffer_, 4 * tmp);
        pipe_.InitBuffer(resultBuffer_, 4 * rows);
        pipe_.InitBuffer(workBuffer_, 4 * (prefix + work));
        pipe_.InitBuffer(onesBuffer_, 4 * ones);
        pipe_.InitBuffer(denomBuffer_, 4 * span);
        const uint32_t used = t.matrix_bytes + 2 * t.gate_bytes + t.input_bytes +
                              4 * (tmp + rows + prefix + work + ones + span);
        const uint32_t workPhase = (t.matrix_bytes + 2 * t.gate_bytes + t.input_bytes +
                                    4 * tmp + 4 * rows + 4 * prefix) % 512;
        const uint32_t pad = (workPhase + 512 - used % 512) % 512;
        const uint32_t pairFloats = 8 * qr;
        const uint32_t pairBytes = pad + 8 * pairFloats;
        if (qr % 16 == 0 && used + pairBytes + 1024 <= 196352) {
            pipe_.InitBuffer(coeffBuffer_, pairBytes);
            coeffPair_ = coeffBuffer_.Get<float>()[pad / 4];
            coeffPairFloats_ = pairFloats;
        }

        matrixRoot_ = matrixBuffer_.Get<float>();
        gates_ = gateBuffer_.Get<float>();
        spillRoot_ = partialBuffer_.Get<float>();
        tmp_ = tmpBuffer_.Get<float>();
        result_ = resultBuffer_.Get<float>();
        work_ = workBuffer_.Get<float>()[prefix];
        ones_ = onesBuffer_.Get<float>();
        denom_ = denomBuffer_.Get<float>();
        slot_ = t.matrix_bytes / 16;
        const uint32_t base = t.matrix_bytes + 2 * t.gate_bytes;
        p0_ = (512 - base % 512) % 512 / 4;
        p1_ = (256 + 512 - base % 512) % 512 / 4;

        Duplicate(matrixRoot_, 0.0f, t.matrix_bytes / 4);
        Duplicate(spillRoot_, 0.0f, t.input_bytes / 4);
        Duplicate(result_, 0.0f, rows);
        Duplicate(ones_, 1.0f, ones);
        PipeBarrier<PIPE_V>();

        ready_ = pipe_.AllocEventID<HardEvent::V_MTE3>();
        loaded_ = pipe_.AllocEventID<HardEvent::MTE2_V>();
        gateLoaded_ = pipe_.AllocEventID<HardEvent::MTE2_V>();
        reusable_ = pipe_.AllocEventID<HardEvent::V_MTE2>();
        outputReusable_ = pipe_.AllocEventID<HardEvent::MTE3_V>();
        gateStored_ = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        for (uint32_t j = 0; j < 4; ++j) {
            stored_[j] = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        }
    }

    template<bool EnableForecast>
    __aicore__ inline void Process() {
        uint32_t input = QuarterPack(0, 1, 2, 3, 4);
        uint64_t off = 0;
        Select(off);
        Reusable();
        LoadGates(start_);
        GateReady();
        Gates();
        inputMapHeight_ = 0;
        for (uint32_t j = 0; j < 4; ++j) LoadPackedQuarter(start_, input, j, false);

        for (;;) {
            rows_ = t_.h * valid_;
            CapacityFiveGeometry g;
            g.Init(t_.h, t_.h_pad, valid_, slot_, p0_, p1_);
            SetMaskNorm();
            PipeBarrier<PIPE_V>();

            for (uint32_t j = 0; j < 4; ++j) {
                auto dst = Node(QuarterNode(input, j), j);
                const uint32_t used = Height(j) * span_;
                if (used < g.qr * g.p) Duplicate(dst[used], 0.0f, g.qr * g.p - used);
            }
            PipeBarrier<PIPE_V>();
            RunQuarterBatchSoftmax(Raw(matrixRoot_), Raw(spillRoot_), Raw(tmp_),
                                  Raw(result_), Raw(work_), Raw(ones_),
                                  t_.eps, input, g);

            __ubuf__ float *coeffPair = coeffPairFloats_ ? Raw(coeffPair_) : nullptr;
            const bool forecast = RunQuarterInputFive<EnableForecast>(Raw(matrixRoot_), Raw(spillRoot_), Raw(tmp_),
                                Raw(result_), Raw(work_), Raw(ones_),
                                Raw(denom_), coeffPair, t_.eps, t_.iters - 1, input, g);
            PipeBarrier<PIPE_V>();
            ResetMask();

            uint32_t output = input;
            for (uint32_t k = 0; k < (2 * t_.iters - 1 - (forecast ? 4 * kWideForecastSteps : 0)) % 5; ++k) output = QuarterAdvance(output);

            SetFlag<HardEvent::V_MTE3>(ready_);
            WaitFlag<HardEvent::V_MTE3>(ready_);
            StoreGates(start_ + off);
            SetFlag<HardEvent::MTE3_MTE2>(gateStored_);

            for (uint32_t j = 0; j < 4; ++j) {
                StorePackedQuarter(start_ + off, output, j);
                SetFlag<HardEvent::MTE3_MTE2>(stored_[j]);
            }

            SetFlag<HardEvent::MTE3_V>(outputReusable_);
            WaitFlag<HardEvent::MTE3_V>(outputReusable_);

            const uint64_t next = off + valid_;
            if (next == count_) {
                Drain();
                break;
            }

            WaitFlag<HardEvent::MTE3_MTE2>(gateStored_);
            Select(next);
            LoadGates(start_ + next);
            GateReady();
            Gates();

            const uint32_t nextInput = output;
            inputMapHeight_ = 0;
            for (uint32_t j = 0; j < 4; ++j) LoadPackedQuarter(start_ + next, nextInput, j, true);

            input = nextInput;
            off = next;
        }

        pipe_.ReleaseEventID<HardEvent::V_MTE3>(ready_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(loaded_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(gateLoaded_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(reusable_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReusable_);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(gateStored_);
        for (uint32_t j = 0; j < 4; ++j) {
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(stored_[j]);
        }
    }

private:
    __aicore__ inline __ubuf__ float *Raw(LocalTensor<float> x) {
        return reinterpret_cast<__ubuf__ float *>(x.GetPhyAddr());
    }

    __aicore__ inline LocalTensor<float> Node(uint32_t node, uint32_t logical) {
        if (node == 4) return spillRoot_[logical == 0 ? p0_ : p1_];
        return matrixRoot_[node * slot_ + (((node & 1) ^ (logical == 0 ? 0U : 1U)) ? 64 : 0)];
    }

    __aicore__ inline uint32_t Height(uint32_t logical) {
        const uint32_t q = (t_.h + 3) / 4;
        return logical == 3 ? t_.h - 3 * q : q;
    }

    __aicore__ inline void LoadReady() {
        SetFlag<HardEvent::MTE2_V>(loaded_);
        WaitFlag<HardEvent::MTE2_V>(loaded_);
    }

    __aicore__ inline void GateReady() {
        SetFlag<HardEvent::MTE2_V>(gateLoaded_);
        WaitFlag<HardEvent::MTE2_V>(gateLoaded_);
    }

    __aicore__ inline void Reusable() {
        SetFlag<HardEvent::V_MTE2>(reusable_);
        WaitFlag<HardEvent::V_MTE2>(reusable_);
    }

    __aicore__ inline void Select(uint64_t off) {
        const uint64_t left = count_ - off;
        valid_ = left < t_.tile_tokens ? left : t_.tile_tokens;
        rows_ = t_.h * valid_;
        span_ = t_.h_pad * valid_;
    }

    __aicore__ inline void Gates() {
        const uint32_t n = 2 * span_;
        auto numerator = tmp_[n];
        auto denominator = tmp_[2 * n];
        // Match the main interleaved gate calculation: one bounded exponential.
        Duplicate(numerator, 1.0f, n);
        Muls(tmp_, gates_, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Mins(tmp_, tmp_, 80.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(tmp_, tmp_, n);
        PipeBarrier<PIPE_V>();
        Adds(denominator, tmp_, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(gates_, numerator, denominator, n);
        PipeBarrier<PIPE_V>();
        Adds(gates_, gates_, t_.eps, span_);
        Muls(gates_[span_], gates_[span_], 2.0f, span_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadGates(uint64_t first) {
        const DataCopyExtParams g{2, 4 * t_.h, 0, (valid_ - 1) * t_.h_pad / 8, 0};
        const DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(t_.h_pad - t_.h), 0.0f};
        for (uint32_t token = 0; token < valid_; ++token) {
            DataCopyPad(gates_[token * t_.h_pad], input_[(first + token) * t_.input_width], g, pad);
        }
    }

    __aicore__ inline void BuildPackedInputMap(uint32_t height, uint32_t packedStride) {
        if (inputMapHeight_ == height) return;
        const uint32_t capacityQr = (((t_.h + 3) / 4 * t_.tile_tokens + 7) / 8 * 8);
        inputMapRows_ = 32 * capacityQr / span_;
        if (inputMapRows_ > 8) inputMapRows_ = 8;
        auto index = tmp_.ReinterpretCast<int32_t>();
        for (uint32_t token = 0; token < valid_; ++token) {
            CreateVecIndex(index[token * t_.h_pad], static_cast<int32_t>(token * packedStride), t_.h_pad);
        }
        PipeBarrier<PIPE_V>();
        Muls(index, index, int32_t{4}, span_);
        PipeBarrier<PIPE_V>();
        const uint32_t tail = t_.h - (t_.h_pad / 8 - 1) * 8;
        uint64_t mask[2] = {uint64_t{0xff} ^ ((uint64_t{1} << tail) - 1), 0};
        Duplicate(index[t_.h_pad - 8], int32_t{0}, mask,
                  static_cast<uint8_t>(valid_), uint16_t{1}, static_cast<uint8_t>(t_.h_pad / 8));
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < inputMapRows_; ++row) {
            Adds(index[row * span_], index, static_cast<int32_t>(4 * row * t_.h), span_);
        }
        PipeBarrier<PIPE_V>();
        inputMapHeight_ = height;
    }

    __aicore__ inline void LoadPackedQuarter(uint64_t first, uint32_t state,
                                            uint32_t logical, bool waitStored) {
        const uint32_t height = Height(logical), q = (t_.h + 3) / 4;
        const uint32_t packedStride = (height * t_.h + 7) / 8 * 8;
        auto staging = Node(QuarterNode(state, 4), 0);
        auto dst = Node(QuarterNode(state, logical), logical);
        Reusable();
        const DataCopyExtParams copy{1, 4 * height * t_.h, 0, 0, 0};
        const DataCopyPadExtParams<float> pad{
            true, 0, static_cast<uint8_t>(packedStride - height * t_.h), 0.0f};
        for (uint32_t token = 0; token < valid_; ++token) {
            DataCopyPad(staging[token * packedStride],
                        input_[(first + token) * t_.input_width + (2 + logical * q) * t_.h],
                        copy, pad);
        }
        if (waitStored) WaitFlag<HardEvent::MTE3_MTE2>(stored_[logical]);
        LoadReady();
        BuildPackedInputMap(height, packedStride);
        auto index = tmp_.ReinterpretCast<uint32_t>();
        for (uint32_t row = 0; row < height; row += inputMapRows_) {
            const uint32_t n = height - row < inputMapRows_ ? height - row : inputMapRows_;
            Gather(dst[row * span_], staging, index, uint32_t{4} * row * t_.h, n * span_);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t qr = (q * valid_ + 7) / 8 * 8;
        const uint32_t tail = t_.h - (t_.h_pad / 8 - 1) * 8;
        const uint64_t validMask = ((uint64_t{1} << tail) - 1) * uint64_t{0x0101010101010101};
        uint64_t mask[2] = {~validMask, 0};
        SetMaskNorm();
        Duplicate(dst[t_.h_pad - 8], 0.0f, mask, static_cast<uint8_t>(qr / 8),
                  static_cast<uint16_t>(t_.h_pad / 8), static_cast<uint8_t>(t_.h_pad));
        WideRoundMask(~uint64_t{0});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreGates(uint64_t first) {
        const DataCopyExtParams g{1, 4 * t_.h, 0, 0, 0};
        for (uint32_t token = 0; token < valid_; ++token) {
            DataCopyPad(pre_[(first + token) * t_.h], gates_[token * t_.h_pad], g);
            DataCopyPad(post_[(first + token) * t_.h], gates_[span_ + token * t_.h_pad], g);
        }
    }

    __aicore__ inline void StorePackedQuarter(uint64_t first, uint32_t state, uint32_t logical) {
        const uint32_t height = Height(logical), q = (t_.h + 3) / 4;
        const uint32_t packedStride = (height * t_.h + 7) / 8 * 8;
        const uint32_t stageNode = QuarterNode(state, logical == 0 ? 4 : logical - 1);
        auto staging = Node(stageNode, 0);
        auto src = Node(QuarterNode(state, logical), logical);
        const GatherMaskParams gather{1, static_cast<uint16_t>(height),
                                      static_cast<uint16_t>(span_ / 8), 0};
        for (uint32_t token = 0; token < valid_; ++token) {
            uint64_t retained = 0;
            GatherMask(staging[token * packedStride], src[token * t_.h_pad],
                       uint8_t{7}, true, t_.h, gather, retained);
        }
        ResetMask();
        SetFlag<HardEvent::V_MTE3>(ready_);
        WaitFlag<HardEvent::V_MTE3>(ready_);
        const DataCopyExtParams copy{1, 4 * height * t_.h, 0, 0, 0};
        for (uint32_t token = 0; token < valid_; ++token) {
            DataCopyPad(comb_[(first + token) * t_.h * t_.h + logical * q * t_.h],
                        staging[token * packedStride], copy);
        }
    }

    __aicore__ inline void Drain() {
        WaitFlag<HardEvent::MTE3_MTE2>(gateStored_);
        for (uint32_t j = 0; j < 4; ++j) WaitFlag<HardEvent::MTE3_MTE2>(stored_[j]);
        LoadReady();
        PipeBarrier<PIPE_ALL>();
    }

    sinkhorn_norm::TilingData t_;
    uint64_t start_{0};
    uint64_t count_{0};
    uint32_t valid_{0};
    uint32_t rows_{0};
    uint32_t span_{0};
    uint32_t slot_{0};
    uint32_t p0_{0};
    uint32_t p1_{0};
    uint32_t inputMapHeight_{0};
    uint32_t inputMapRows_{0};
    uint32_t coeffPairFloats_{0};

    TPipe pipe_;
    GlobalTensor<float> input_;
    GlobalTensor<float> pre_;
    GlobalTensor<float> post_;
    GlobalTensor<float> comb_;

    TBuf<TPosition::VECCALC> matrixBuffer_;
    TBuf<TPosition::VECCALC> gateBuffer_;
    TBuf<TPosition::VECCALC> partialBuffer_;
    TBuf<TPosition::VECCALC> tmpBuffer_;
    TBuf<TPosition::VECCALC> resultBuffer_;
    TBuf<TPosition::VECCALC> workBuffer_;
    TBuf<TPosition::VECCALC> onesBuffer_;
    TBuf<TPosition::VECCALC> denomBuffer_;
    TBuf<TPosition::VECCALC> coeffBuffer_;

    LocalTensor<float> matrixRoot_;
    LocalTensor<float> spillRoot_;
    LocalTensor<float> gates_;
    LocalTensor<float> tmp_;
    LocalTensor<float> result_;
    LocalTensor<float> work_;
    LocalTensor<float> ones_;
    LocalTensor<float> denom_;
    LocalTensor<float> coeffPair_;

    TEventID ready_;
    TEventID loaded_;
    TEventID gateLoaded_;
    TEventID reusable_;
    TEventID gateStored_;
    TEventID stored_[4];
    TEventID outputReusable_;
};

}  // namespace f13
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f13_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f13::QuarterStreamSinkhorn op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process<false>();
}

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64), section(".sinkhorn_wide_forecast_entry")))
void sinkhorn_f13_forecast_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f13::QuarterStreamSinkhorn op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process<true>();
}

#endif  // SINKHORN_NORM_KERNELS_F13_WIDE_COMPACT_FIVE_H_
