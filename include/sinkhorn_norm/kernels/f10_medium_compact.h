#ifndef SINKHORN_NORM_KERNELS_F10_MEDIUM_COMPACT_H_
#define SINKHORN_NORM_KERNELS_F10_MEDIUM_COMPACT_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"
#include "sinkhorn_norm/common/normalization_forecast.h"
#include "sinkhorn_norm/kernels/f12_wide_blocked.h"

namespace sinkhorn_norm {
namespace f10 {

using namespace AscendC;

static __aicore__ __attribute__((always_inline)) inline void WideRoundMask(uint64_t mask) {
  SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, mask);
}

template <int Op>
static __aicore__ __attribute__((always_inline)) inline void WideRoundSpan(
    __ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b,
    uint32_t count, float scalar = 0.0f) {
  uint32_t first = 0;
  while (count - first >= 64) {
    const uint32_t left = (count - first) / 64;
    const uint8_t repeats = left < 255 ? left : 255;
    if constexpr (Op == 0) vadd(dst + first, a + first, b + first, repeats, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 1) vadds(dst + first, a + first, scalar, repeats, 1, 1, 8, 8);
    if constexpr (Op == 2) vdiv(dst + first, a + first, b + first, repeats, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first), reinterpret_cast<__ubuf__ uint32_t *>(a + first), repeats, 1, 1, 8, 8);
    first += 64 * repeats;
  }
  if (first < count) {
    WideRoundMask((uint64_t{1} << (count - first)) - 1);
    if constexpr (Op == 0) vadd(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 1) vadds(dst + first, a + first, scalar, 1, 1, 1, 8, 8);
    if constexpr (Op == 2) vdiv(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first), reinterpret_cast<__ubuf__ uint32_t *>(a + first), 1, 1, 1, 8, 8);
    WideRoundMask(~uint64_t{0});
  }
}

constexpr uint32_t kBlocked56ForecastSteps = 4;

static __aicore__ __attribute__((always_inline)) inline void ScaleBlocked56Columns(
    __ubuf__ float *matrix, __ubuf__ float *denom, uint32_t slabs) {
  if ASCEND_IS_AIV {
    for (uint32_t group = 0; group < 7; ++group) {
      auto m = matrix + group * 64;
      vmul(m, m, denom, slabs, 1, 1, 0, 56, 56, 1);
    }
    pipe_barrier(PIPE_V);
  }
}

static __aicore__ __attribute__((always_inline)) inline void ColumnsBlocked56(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *ones,
    __ubuf__ float *denom, __ubuf__ float *tmp, uint32_t valid, float eps,
    __ubuf__ float *capture = nullptr) {
  if ASCEND_IS_AIV {
    using namespace sinkhorn_f12;
    constexpr uint32_t h = 56;
    const uint32_t q = h / 8, slabs = h / 8 * valid;
    // All seven-row accumulation stages use the same mask and repeat strides.
    // Keep the mask live until the following 7-to-4 reduction changes it.
    WideMask((uint64_t{1} << (7 * 8)) - 1);
    vadd(partial, matrix, matrix + q * 8, slabs, 1, 1, 1, q, h, h);
    vadd(tmp, matrix + 4 * q * 8, matrix + 5 * q * 8,
         slabs, 1, 1, 1, q, h, h);
    pipe_barrier(PIPE_V);
    vadd(partial, partial, matrix + 2 * q * 8, slabs, 1, 1, 1, q, q, h);
    vadd(tmp, tmp, matrix + 6 * q * 8, slabs, 1, 1, 1, q, q, h);
    pipe_barrier(PIPE_V);
    vadd(partial, partial, matrix + 3 * q * 8, slabs, 1, 1, 1, q, q, h);
    vadd(tmp, tmp, matrix + 7 * q * 8, slabs, 1, 1, 1, q, q, h);
    pipe_barrier(PIPE_V);
    vadd(partial, partial, tmp, slabs, 1, 1, 1, q, q, q);
    pipe_barrier(PIPE_V);
    const uint32_t height = q;
    const uint32_t span = slabs * 8;
    const uint32_t half = height / 2, upper = height - half;
    WideMask((uint64_t{1} << (half * 8)) - 1);
    vadd(tmp, partial, partial + upper * 8, slabs, slabs, 1, 1, 1, q, q);
    if (upper > half) {
      WideMask(255);
      vcopy(reinterpret_cast<__ubuf__ uint32_t *>(tmp + half * span),
            reinterpret_cast<__ubuf__ uint32_t *>(partial + half * 8), slabs, 1,
            1, 1, q);
    }
    WideMask(~uint64_t{0});
    pipe_barrier(PIPE_V);
    const uint32_t pairs = upper / 2, next = upper - pairs;
    WideSpan<0>(tmp, tmp, tmp + next * span, pairs * span);
    pipe_barrier(PIPE_V);
    const uint8_t spanRepeats = static_cast<uint8_t>((span + 63) / 64);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vadd(denom, tmp, tmp + span, spanRepeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    if (capture != nullptr) {
      vsub(capture, denom, ones, spanRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    vadds(denom, denom, eps, spanRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(denom, ones, denom, spanRepeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
    // One repeat sequence covers the same rows of every column slab.
    ScaleBlocked56Columns(matrix, denom, slabs);
  }
}

// Exact normalization on [column block, token, row, eight columns].
// tmp and partial provide opposite 256-byte bank phases without new UB storage.
static __aicore__ __attribute__((always_inline)) inline void RunBlocked56(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *ones, __ubuf__ float *denom,
    uint32_t valid, float eps, uint32_t iterations, uint32_t capacity) {
  if ASCEND_IS_AIV {
    using namespace sinkhorn_f12;
    const uint32_t rows = 56 * valid, plane = rows * 8;
    const uint32_t partialPhase = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(partial) - reinterpret_cast<uint64_t>(matrix)) & 511u;
    const uint32_t tmpPhase = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(tmp) - reinterpret_cast<uint64_t>(matrix)) & 511u;
    auto opposite = partial + (((tmpPhase ^ 256u) - partialPhase) & 511u) / 4;
    auto phaseZero = tmpPhase == 0 ? tmp : opposite;
    auto phaseHalf = tmpPhase == 0 ? opposite : tmp;
    InitWide(matrix, partial, tmp, result, 56, valid);
    constexpr uint32_t tail = 8;
    const bool forecast = iterations >= 4 + tail + kBlocked56ForecastSteps &&
        capacity >= 12 * rows + 128 && eps >= 0x1p-24f && eps <= 1.0e-5f;
    const uint32_t checkpoint = forecast ? iterations - tail - kBlocked56ForecastSteps : iterations;
    auto history = forecast ? partial + capacity - 4 * rows : partial;
    for (uint32_t round = 0; round < iterations; ++round) {
      const bool predict = round == checkpoint;
      const bool measure = forecast && round < checkpoint && round + 2 >= checkpoint;
      auto snapshot = history + (round % 2) * 2 * rows;
      if (predict) {
        ForecastDenominatorProduct(result, denom, partial, tmp, ones,
                          history + ((round - 1) % 2) * 2 * rows, snapshot, rows, eps);
      } else {
        WideRows<false>(matrix, partial, result, rows, 7);
        if (measure) {
          SetMaskCount();
          SetVectorMask<float, MaskMode::COUNTER>(0, rows);
          vsub(snapshot, result, ones, (rows + 63) / 64, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          SetMaskNorm();
          WideRoundMask(~uint64_t{0});
        }
        if (round != 0) {
          WideSpan<1>(result, result, result, rows, eps);
          pipe_barrier(PIPE_V);
        }
        WideSpan<2>(result, ones, result, rows);
        pipe_barrier(PIPE_V);
      }
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(phaseHalf),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, rows / 8);
      if ((plane * sizeof(float)) & 511u)
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(phaseZero),
              reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, rows / 8);
      pipe_barrier(PIPE_V);
      for (uint32_t block = 0; block < 7; ++block) {
        auto coeff = ((block * plane * sizeof(float)) & 511u) ? phaseZero : phaseHalf;
        auto m = matrix + block * plane;
        vmul(m, m, coeff, plane / 64, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
      if (round == 0) {
        WideSpan<1>(matrix, matrix, matrix, rows * 56, eps);
        pipe_barrier(PIPE_V);
      }
      if (predict) {
        ScaleBlocked56Columns(matrix, denom, 7 * valid);
        round += kBlocked56ForecastSteps - 1;
      } else {
        ColumnsBlocked56(matrix, partial, ones, denom, tmp, valid, eps,
                         measure ? snapshot + rows : nullptr);
      }
    }
  }
}

template <bool TreeRows = false>
static __aicore__ __attribute__((always_inline)) inline void RunWideCompleteRounds(
    __ubuf__ float *__restrict matrix, __ubuf__ float *__restrict partial,
    __ubuf__ float *__restrict tmp, __ubuf__ float *__restrict result,
    __ubuf__ float *__restrict ones, __ubuf__ float *__restrict denom,
    uint32_t h, uint32_t pitch, uint32_t valid, float eps, uint32_t rounds) {
  if ASCEND_IS_AIV {
    const uint32_t rows = h * valid;
    const uint32_t rp = (rows + 7) / 8 * 8;
    const uint32_t groups = rows / 8, tail = rows % 8;
    const uint32_t blocks = pitch / 8, span = pitch * valid;
    const uint32_t quarterRows = (h + 3) / 4;
    const uint64_t lastMask = ((uint64_t{1} << (h - (blocks - 1) * 8)) - 1) * uint64_t{0x0101010101010101};
    WideRoundMask(~uint64_t{0});
    for (uint32_t round = 0; round < rounds; ++round) {
      if (blocks == 7 && tail == 0) {
        // Direct unrolled 7-block vcgadd for JP5 (H=56)
        vcgadd(result, matrix + 0 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 0 * rp, matrix + 1 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 1 * rp, matrix + 2 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 2 * rp, matrix + 3 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 3 * rp, matrix + 4 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 4 * rp, matrix + 5 * 8, groups, 1, 7, pitch);
        vcgadd(partial + 5 * rp, matrix + 6 * 8, groups, 1, 7, pitch);
      } else {
        for (uint32_t block = 0; block < blocks; ++block) {
          const uint64_t mask = block + 1 == blocks ? lastMask : ~uint64_t{0};
          if (mask != ~uint64_t{0}) WideRoundMask(mask);
          auto dst = TreeRows ? partial + block * rp : (block == 0 ? result : partial + (block - 1) * rp);
          vcgadd(dst, matrix + block * 8, groups, 1, blocks, pitch);
          if (tail) {
            WideRoundMask(mask & ((uint64_t{1} << (8 * tail)) - 1));
            vcgadd(dst + groups * 8, matrix + groups * 8 * pitch + block * 8, 1, 1, blocks, pitch);
          }
          if (tail || mask != ~uint64_t{0}) WideRoundMask(~uint64_t{0});
        }
      }
      pipe_barrier(PIPE_V);
      if constexpr (TreeRows) {
        uint32_t planes = blocks;
        while (planes > 2) {
          const uint32_t pairs = planes / 2, upper = planes - pairs;
          WideRoundSpan<0>(partial, partial, partial + upper * rp, pairs * rp);
          pipe_barrier(PIPE_V);
          planes = upper;
        }
        WideRoundSpan<0>(result, partial, partial + rp, rows);
      } else {
        for (uint32_t first = 0; first < rows; first += 64) {
          const uint32_t width = rows - first < 64 ? rows - first : 64;
          if (width < 64) WideRoundMask((uint64_t{1} << width) - 1);
          vadd(result + first, partial + first, result + first, blocks - 1, 1, 1, 1, 0, rp / 8, 0);
          if (width < 64) WideRoundMask(~uint64_t{0});
        }
      }
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(0, rows);
      vadds(result, result, eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(result, ones, result, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      WideRoundMask(~uint64_t{0});
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp), reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, rp / 8);
      pipe_barrier(PIPE_V);
      if (blocks == 7 && tail == 0) {
        // Direct unrolled 7-block vmul for JP5 (H=56)
        vmul(matrix + 0 * 8, matrix + 0 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 1 * 8, matrix + 1 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 2 * 8, matrix + 2 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 3 * 8, matrix + 3 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 4 * 8, matrix + 4 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 5 * 8, matrix + 5 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
        vmul(matrix + 6 * 8, matrix + 6 * 8, tmp, groups, 7, 7, 1, pitch, pitch, 8);
      } else {
        for (uint32_t block = 0; block < blocks; ++block) {
          const uint64_t mask = block + 1 == blocks ? lastMask : ~uint64_t{0};
          if (mask != ~uint64_t{0}) WideRoundMask(mask);
          auto m = matrix + block * 8;
          vmul(m, m, tmp, groups, blocks, blocks, 1, pitch, pitch, 8);
          if (tail) {
            WideRoundMask(mask & ((uint64_t{1} << (8 * tail)) - 1));
            auto last = m + groups * 8 * pitch;
            vmul(last, last, tmp + groups * 64, 1, blocks, blocks, 1, pitch, pitch, 8);
          }
          if (tail || mask != ~uint64_t{0}) WideRoundMask(~uint64_t{0});
        }
      }
      pipe_barrier(PIPE_V);
      if (h == 56) {
        // Direct unrolled column binary tree for H=56
        const uint32_t quarter = 14 * span;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, quarter);
        vadd(partial, matrix, matrix + quarter, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(partial, partial, matrix + 2 * quarter, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(partial, partial, matrix + 3 * quarter, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 14 -> 7
        SetVectorMask<float, MaskMode::COUNTER>(0, 7 * span);
        vadd(partial, partial, partial + 7 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 7 -> 4 (pairs=3, upper=4)
        SetVectorMask<float, MaskMode::COUNTER>(0, 3 * span);
        vadd(partial, partial, partial + 4 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 4 -> 2 (pairs=2, upper=2)
        SetVectorMask<float, MaskMode::COUNTER>(0, 2 * span);
        vadd(partial, partial, partial + 2 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 2 -> 1 (pairs=1, upper=1)
        SetVectorMask<float, MaskMode::COUNTER>(0, 1 * span);
        vadd(partial, partial, partial + 1 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // Column div:
        SetVectorMask<float, MaskMode::COUNTER>(0, span);
        vadds(denom, partial, eps, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(denom, ones, denom, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        SetMaskNorm();
        WideRoundMask(~uint64_t{0});
      } else {
        const uint32_t quarter = quarterRows * span;
        WideRoundSpan<0>(partial, matrix, matrix + quarter, quarter);
        pipe_barrier(PIPE_V);
        WideRoundSpan<0>(partial, partial, matrix + 2 * quarter, quarter);
        pipe_barrier(PIPE_V);
        WideRoundSpan<0>(partial, partial, matrix + 3 * quarter, (h - 3 * quarterRows) * span);
        pipe_barrier(PIPE_V);
        for (uint32_t height = quarterRows; height > 1;) {
          const uint32_t half = height / 2, upper = height - half;
          WideRoundSpan<0>(partial, partial, partial + upper * span, half * span);
          pipe_barrier(PIPE_V);
          height = upper;
        }
        WideRoundSpan<3>(denom, partial, partial, span);
        pipe_barrier(PIPE_V);
        WideRoundSpan<1>(denom, denom, denom, span, eps);
        pipe_barrier(PIPE_V);
        WideRoundSpan<2>(denom, ones, denom, span);
        pipe_barrier(PIPE_V);
      }
      for (uint32_t col = 0; col < span; col += 64) {
        const uint32_t width = span - col < 64 ? span - col : 64;
        if (width < 64) WideRoundMask((uint64_t{1} << width) - 1);
        vmul(matrix + col, matrix + col, denom + col, h, 1, 1, 1, span / 8, span / 8, 0);
        if (width < 64) WideRoundMask(~uint64_t{0});
      }
      pipe_barrier(PIPE_V);
    }
  }
}

class MediumCompactKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const sinkhorn_norm::MediumCompactArgs &t) {
    t_ = {};
    t_.eps = t.eps;
    t_.iters = t.iters;
    t_.h = t.h;
    t_.h_pad = (t.h + 7) / 8 * 8;
    t_.tile_tokens = t.tile;
    t_.tokens_per_core = t.owner;
    t_.extra_core_count = t.extra;
    t_.input_width = t.h * (t.h + 2);
    const uint32_t rows = t.h * t.tile, span = t_.h_pad * t.tile;
    t_.input_bytes = t.input_bytes;
    t_.matrix_bytes = 4 * rows * t_.h_pad;
    t_.gate_bytes = 4 * span;
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);

    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    const uint32_t rp = (t_.h * t_.tile_tokens + 7) / 8 * 8;
    const uint32_t gateSpan = t_.tile_tokens * t_.h_pad;
    const uint32_t ones = rp > 2 * gateSpan ? rp : 2 * gateSpan;
    const uint32_t tmp = 8 * rp > 6 * gateSpan ? 8 * rp : 6 * gateSpan;

    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes);
    pipe_.InitBuffer(partialBuffer_, t_.input_bytes);
    pipe_.InitBuffer(tmpBuffer_, tmp * sizeof(float));
    pipe_.InitBuffer(resultBuffer_, rp * sizeof(float));
    pipe_.InitBuffer(onesBuffer_, ones * sizeof(float));
    pipe_.InitBuffer(denomBuffer_, gateSpan * sizeof(float));

    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>();
    tmp_ = tmpBuffer_.Get<float>();
    result_ = resultBuffer_.Get<float>();
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();

    Duplicate(result_, 0.0f, rp);
    Duplicate(ones_, 1.0f, ones);
    constants_ready_ = false;
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t first = start_ + offset;
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left) : t_.tile_tokens;
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;

      if (t_.h == 56 && t_.h_pad == 56) {
        CopyInBlocked56(first);
        Gates();
        SetMaskNorm();
        PipeBarrier<PIPE_V>();
        RunBlocked56(
            reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
            valid_, t_.eps, t_.iters, t_.input_bytes / sizeof(float));
        CopyOutBlocked56(first);
      } else {
        CopyIn(first);
        Gates();
        Softmax();
        Columns();

        if (t_.iters > 1) {
          SetMaskNorm();
          PipeBarrier<PIPE_V>();
          RunWideCompleteRounds<false>(
              reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
              t_.h, t_.h_pad, valid_, t_.eps, t_.iters - 1);
          PipeBarrier<PIPE_V>();
          ResetMask();
        }

        CopyOut(first);
      }
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void CopyInBlocked56(uint64_t first) {
    const DataCopyExtParams gate{static_cast<uint16_t>(valid_), 4 * t_.h,
                                 4 * (t_.input_width - t_.h), 0, 0};
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    const uint64_t base = first * t_.input_width;
    DataCopyPad(gates_, input_[base], gate, noPad);
    DataCopyPad(gates_[span_], input_[base + t_.h], gate, noPad);
    const uint32_t capacity = t_.input_bytes / (4 * t_.h) / 8 * 8;
    for (uint32_t token = 0; token < valid_; ++token) {
      for (uint32_t row = 0; row < t_.h;) {
        const uint32_t count = t_.h - row < capacity ? t_.h - row : capacity;
        Sync<HardEvent::V_MTE2>();
        const DataCopyExtParams dense{1, 4 * count * t_.h, 0, 0, 0};
        DataCopyPad(partial_, input_[base + token * t_.input_width + 2 * t_.h + row * t_.h], dense, noPad);
        Sync<HardEvent::MTE2_V>();
        for (uint32_t block = 0; block < 7; ++block)
          sinkhorn_f12::WideCopy8(
              reinterpret_cast<__ubuf__ float *>(matrix_[(block * valid_ + token) * 56 * 8 + row * 8].GetPhyAddr()),
              reinterpret_cast<__ubuf__ float *>(partial_[block * 8].GetPhyAddr()), count, 1, 7);
        PipeBarrier<PIPE_V>();
        row += count;
      }
    }
  }

  __aicore__ inline void CopyOutBlocked56(uint64_t first) {
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams gate{1, 4 * span_, 0, 0, 0};
    DataCopyPad(pre_[first * 56], gates_, gate);
    DataCopyPad(post_[first * 56], gates_[span_], gate);
    const uint32_t capacity = t_.input_bytes / (4 * 56) / 8 * 8;
    for (uint32_t row = 0; row < rows_;) {
      const uint32_t count = rows_ - row < capacity ? rows_ - row : capacity;
      for (uint32_t block = 0; block < 7; ++block)
        sinkhorn_f12::WideCopy8(
            reinterpret_cast<__ubuf__ float *>(partial_[block * 8].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(matrix_[block * rows_ * 8 + row * 8].GetPhyAddr()), count, 7, 1);
      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams dense{1, 4 * count * 56, 0, 0, 0};
      DataCopyPad(comb_[first * 56 * 56 + row * 56], partial_, dense);
      Sync<HardEvent::MTE3_V>();
      row += count;
    }
    Sync<HardEvent::MTE3_MTE2>();
  }

  __aicore__ inline void CopyIn(uint64_t first) {
    if (t_.h == t_.h_pad && valid_ >= 2) {
      const DataCopyExtParams gates{static_cast<uint16_t>(valid_),
                                    4 * t_.h, 4 * (t_.input_width - t_.h), 0, 0};
      const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
      const uint64_t base = first * t_.input_width;
      DataCopyPad(gates_, input_[base], gates, noPad);
      DataCopyPad(gates_[span_], input_[base + t_.h], gates, noPad);
      const DataCopyExtParams matrix{static_cast<uint16_t>(t_.h),
                                     4 * t_.h, 0, (valid_ - 1) * t_.h_pad / 8, 0};
      for (uint32_t token = 0; token < valid_; ++token) {
        DataCopyPad(matrix_[token * t_.h_pad],
                    input_[base + token * t_.input_width + 2 * t_.h],
                    matrix, noPad);
      }
    } else {
      const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
      const DataCopyPadExtParams<float> pad{
          true, 0, static_cast<uint8_t>(t_.h_pad - t_.h), 0.0f};
      const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                                4 * t_.h, 0, stride, 0};
      const DataCopyExtParams g{2, 4 * t_.h, 0, stride, 0};
      for (uint32_t token = 0; token < valid_; ++token) {
        const uint64_t base = (first + token) * t_.input_width;
        DataCopyPad(gates_[token * t_.h_pad], input_[base], g, pad);
        DataCopyPad(matrix_[token * t_.h_pad], input_[base + 2 * t_.h], m, pad);
      }
    }
    Sync<HardEvent::MTE2_V>();
  }

  __aicore__ inline void Gates() {
    const uint32_t n = 2 * span_;
    auto numerator = tmp_[n];
    auto denominator = tmp_[2 * n];
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

  __aicore__ inline void Softmax() {
    // 1. GroupedRowMaxReduce
    {
      const uint32_t blocks = t_.h_pad / 8;
      const uint32_t rp = (rows_ + 7) / 8 * 8;
      const uint32_t groups = rows_ / 8;
      const uint32_t tail = rows_ % 8;
      if (tail != 0) {
        Duplicate(partial_[rp - 8], 0.0f, uint64_t{8},
                  static_cast<uint8_t>(blocks), uint16_t{1},
                  static_cast<uint8_t>(rp / 8));
        PipeBarrier<PIPE_V>();
      }
      for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t col = block * 8;
        const uint32_t width = t_.h - col < 8 ? t_.h - col : 8;
        uint64_t mask[2] = {
            ((uint64_t{1} << width) - 1) * uint64_t{0x0101010101010101}, 0};
        auto dst = partial_[block * rp];
        if (groups != 0) {
          BlockReduceMax(dst, matrix_[col], static_cast<int32_t>(groups),
                         mask, 1, static_cast<int32_t>(blocks),
                         static_cast<int32_t>(t_.h_pad));
        }
        if (tail != 0) {
          mask[0] &= (uint64_t{1} << (8 * tail)) - 1;
          BlockReduceMax(dst[groups * 8],
                         matrix_[groups * 8 * t_.h_pad + col], 1,
                         mask, 1, static_cast<int32_t>(blocks),
                         static_cast<int32_t>(t_.h_pad));
        }
      }
      PipeBarrier<PIPE_V>();
      uint32_t planes = blocks;
      while (planes > 2) {
        const uint32_t pairs = planes / 2;
        const uint32_t upper = planes - pairs;
        Max(partial_, partial_, partial_[upper * rp], pairs * rp);
        PipeBarrier<PIPE_V>();
        planes = upper;
      }
      Max(result_, partial_, partial_[rp], rows_);
      PipeBarrier<PIPE_V>();
    }

    // 2. BroadcastRows
    Brcb(tmp_, result_, static_cast<uint8_t>((rows_ + 7) / 8), {1, 8});
    PipeBarrier<PIPE_V>();

    // 3. Subtract max
    {
      const auto blockStride = static_cast<uint8_t>(t_.h_pad / 8);
      const auto repeatStride = static_cast<uint8_t>(t_.h_pad);
      const auto groups = static_cast<uint8_t>(rows_ / 8);
      const uint32_t tail = rows_ % 8;
      for (uint32_t col = 0; col < t_.h; col += 8) {
        const uint32_t width = t_.h - col < 8 ? t_.h - col : 8;
        uint64_t mask[2] = {
            ((uint64_t{1} << width) - 1) * uint64_t{0x0101010101010101}, 0};
        auto m = matrix_[col];
        if (groups != 0) {
          Sub(m, m, tmp_, mask, groups,
              {blockStride, blockStride, 1, repeatStride, repeatStride, 8});
        }
        if (tail != 0) {
          mask[0] &= (uint64_t{1} << (8 * tail)) - 1;
          const uint32_t first = static_cast<uint32_t>(groups) * 8;
          auto last = matrix_[first * t_.h_pad + col];
          Sub(last, last, tmp_[first * 8], mask, 1,
              {blockStride, blockStride, 1, repeatStride, repeatStride, 8});
        }
      }
      PipeBarrier<PIPE_V>();
    }

    // 4. Exp
    Exp(matrix_, matrix_, rows_ * t_.h_pad);
    PipeBarrier<PIPE_V>();

    // 5. RowSums
    {
      const uint32_t blocks = t_.h_pad / 8;
      const uint32_t rp = (rows_ + 7) / 8 * 8;
      const uint32_t groups = rows_ / 8;
      const uint32_t tail = rows_ % 8;
      for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t col = block * 8;
        const uint32_t width = t_.h - col < 8 ? t_.h - col : 8;
        uint64_t mask[2] = {
            ((uint64_t{1} << width) - 1) * uint64_t{0x0101010101010101}, 0};
        auto dst = block == 0 ? result_ : partial_[(block - 1) * rp];
        if (groups != 0) {
          BlockReduceSum(dst, matrix_[col], static_cast<int32_t>(groups),
                         mask, 1, static_cast<int32_t>(blocks),
                         static_cast<int32_t>(t_.h_pad));
        }
        if (tail != 0) {
          mask[0] &= (uint64_t{1} << (8 * tail)) - 1;
          BlockReduceSum(dst[groups * 8],
                         matrix_[groups * 8 * t_.h_pad + col], 1, mask, 1,
                         static_cast<int32_t>(blocks), static_cast<int32_t>(t_.h_pad));
        }
      }
      PipeBarrier<PIPE_V>();
      for (uint32_t first = 0; first < rows_; first += 64) {
        const uint64_t width = rows_ - first < 64 ? rows_ - first : 64;
        auto dst = result_[first];
        Add(dst, partial_[first], dst, width,
            static_cast<uint8_t>(blocks - 1),
            {1, 1, 1, 0, static_cast<uint8_t>(rp / 8), 0});
      }
      PipeBarrier<PIPE_V>();
    }

    // 6. ScaleRows: result_ = ones_ / result_; Broadcast; Multiply; Adds eps
    Div(result_, ones_, result_, rows_);
    PipeBarrier<PIPE_V>();
    Brcb(tmp_, result_, static_cast<uint8_t>((rows_ + 7) / 8), {1, 8});
    PipeBarrier<PIPE_V>();
    {
      const auto blockStride = static_cast<uint8_t>(t_.h_pad / 8);
      const auto repeatStride = static_cast<uint8_t>(t_.h_pad);
      const auto groups = static_cast<uint8_t>(rows_ / 8);
      const uint32_t tail = rows_ % 8;
      for (uint32_t col = 0; col < t_.h; col += 8) {
        const uint32_t width = t_.h - col < 8 ? t_.h - col : 8;
        uint64_t mask[2] = {
            ((uint64_t{1} << width) - 1) * uint64_t{0x0101010101010101}, 0};
        auto m = matrix_[col];
        if (groups != 0) {
          Mul(m, m, tmp_, mask, groups,
              {blockStride, blockStride, 1, repeatStride, repeatStride, 8});
        }
        if (tail != 0) {
          mask[0] &= (uint64_t{1} << (8 * tail)) - 1;
          const uint32_t first = static_cast<uint32_t>(groups) * 8;
          auto last = matrix_[first * t_.h_pad + col];
          Mul(last, last, tmp_[first * 8], mask, 1,
              {blockStride, blockStride, 1, repeatStride, repeatStride, 8});
        }
      }
      PipeBarrier<PIPE_V>();
    }
    Adds(matrix_, matrix_, t_.eps, rows_ * t_.h_pad);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Columns() {
    uint32_t rows = (t_.h + 3) / 4;
    Add(partial_, matrix_, matrix_[rows * span_], rows * span_);
    PipeBarrier<PIPE_V>();
    Add(partial_, partial_, matrix_[2 * rows * span_], rows * span_);
    PipeBarrier<PIPE_V>();
    Add(partial_, partial_, matrix_[3 * rows * span_],
        (t_.h - 3 * rows) * span_);
    PipeBarrier<PIPE_V>();
    while (rows > 1) {
      const uint32_t half = rows / 2;
      const uint32_t upper = rows - half;
      Add(partial_, partial_, partial_[upper * span_], half * span_);
      PipeBarrier<PIPE_V>();
      rows = upper;
    }
    if (span_ >= 64) {
      Copy(denom_, partial_, uint64_t{64},
           static_cast<uint8_t>(span_ / 64), {1, 1, 8, 8});
    }
    if (span_ % 64 != 0) {
      Copy(denom_[span_ / 64 * 64], partial_[span_ / 64 * 64],
           uint64_t{span_ % 64}, 1, {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();
    Adds(denom_, denom_, t_.eps, span_);
    PipeBarrier<PIPE_V>();
    Div(denom_, ones_, denom_, span_);
    PipeBarrier<PIPE_V>();
    for (uint32_t col = 0; col < span_; col += 64) {
      const uint64_t width = span_ - col < 64 ? span_ - col : 64;
      Mul(matrix_[col], matrix_[col], denom_[col], width,
          static_cast<uint8_t>(t_.h),
          {1, 1, 1, static_cast<uint8_t>(span_ / 8),
           static_cast<uint8_t>(span_ / 8), 0});
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyOut(uint64_t first) {
    Sync<HardEvent::V_MTE3>();
    if (t_.h == t_.h_pad && valid_ >= 2) {
      const DataCopyExtParams gates{1, 4 * span_, 0, 0, 0};
      DataCopyPad(pre_[first * t_.h], gates_, gates);
      DataCopyPad(post_[first * t_.h], gates_[span_], gates);
      const DataCopyExtParams matrix{static_cast<uint16_t>(t_.h),
                                     4 * t_.h, (valid_ - 1) * t_.h_pad / 8, 0, 0};
      for (uint32_t token = 0; token < valid_; ++token) {
        DataCopyPad(comb_[(first + token) * t_.h * t_.h],
                    matrix_[token * t_.h_pad], matrix);
      }
    } else {
      const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
      const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                                4 * t_.h, stride, 0, 0};
      const DataCopyExtParams g{1, 4 * t_.h, 0, 0, 0};
      for (uint32_t token = 0; token < valid_; ++token) {
        const uint64_t n = first + token;
        DataCopyPad(pre_[n * t_.h], gates_[token * t_.h_pad], g);
        DataCopyPad(post_[n * t_.h], gates_[span_ + token * t_.h_pad], g);
        DataCopyPad(comb_[n * t_.h * t_.h], matrix_[token * t_.h_pad], m);
      }
    }
    Sync<HardEvent::MTE3_MTE2>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_, gateBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_, tmpBuffer_, resultBuffer_,
      onesBuffer_, denomBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, result_, ones_, denom_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t valid_, rows_, span_;
  bool constants_ready_;
};

}  // namespace f10
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f10_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::MediumCompactArgs tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f10::MediumCompactKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F10_MEDIUM_COMPACT_H_
