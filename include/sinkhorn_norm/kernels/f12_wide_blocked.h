#ifndef SINKHORN_NORM_KERNELS_F12_WIDE_BLOCKED_H_
#define SINKHORN_NORM_KERNELS_F12_WIDE_BLOCKED_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

#define F12_FN __aicore__ __attribute__((always_inline)) inline

namespace sinkhorn_f12 {

using namespace AscendC;
using sinkhorn_norm::TilingData;
using Fp = __ubuf__ float *;
using Up = __ubuf__ uint32_t *;

F12_FN void WideMask(
    uint64_t mask) {
  if ASCEND_IS_AIV {
    SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, mask);
  }
}

// Copy eight rows per vector repeat between dense staging and blocked UB.
// Only the last part group needs a mask. Caller owns DMA/vector fences.
F12_FN void WideCopy8(
    Fp dst, Fp src, uint32_t rows,
    uint32_t dstStride, uint32_t srcStride) {
  if ASCEND_IS_AIV {
    WideMask(~uint64_t{0});
    const uint8_t repeats = static_cast<uint8_t>(rows / 8);
    vcopy(reinterpret_cast<Up>(dst),
          reinterpret_cast<Up>(src),
          repeats, dstStride, srcStride, dstStride * 8, srcStride * 8);
  }
}

// Complete aligned rows are stored as [column block, token, row, eight columns].
template <int Op>
F12_FN void WideSpan(
    Fp dst, Fp a, Fp b, uint32_t count,
    float scalar = 0.0f) {
  if ASCEND_IS_AIV {
    SetMaskCount();
    for (uint32_t offset = 0; offset < count;) {
      const uint32_t left = count - offset;
      const uint32_t current = left < 255 * 64 ? left : 255 * 64;
      const uint8_t repeats = static_cast<uint8_t>((current + 63) / 64);
      SetVectorMask<float, MaskMode::COUNTER>(0, current);
      if constexpr (Op == 0)
        vadd(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
      else if constexpr (Op == 1)
        vadds(dst + offset, a + offset, scalar, repeats, 1, 1, 8, 8);
      else if constexpr (Op == 2)
        vdiv(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
      else if constexpr (Op == 3)
        vmul(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
      else if constexpr (Op == 4)
        vmax(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
      else if constexpr (Op == 5)
        vsub(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
      else if constexpr (Op == 6)
        vexp(dst + offset, a + offset, repeats, 1, 1, 8, 8);
      offset += current;
    }
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
  }
}

template <bool Maximum>
F12_FN void WideRows(
    Fp mat, Fp part, Fp res,
    uint32_t rows, uint32_t blocks) {
  if ASCEND_IS_AIV {
    const uint8_t groups = static_cast<uint8_t>(rows / 8);
    auto dst = part;
    auto src = mat;
    for (uint32_t block = 0; block < blocks; ++block) {
      if constexpr (Maximum)
        vcgmax(dst, src, groups, 1, 1, 8);
      else
        vcgadd(dst, src, groups, 1, 1, 8);
      dst += rows;
      src += rows * 8;
    }
    pipe_barrier(PIPE_V);
    for (uint32_t planes = blocks; planes > 2;) {
      const uint32_t pairs = planes / 2, upper = planes - pairs;
      WideSpan<Maximum ? 4 : 0>(part, part, part + upper * rows,
                                       pairs * rows);
      pipe_barrier(PIPE_V);
      planes = upper;
    }
    WideSpan<Maximum ? 4 : 0>(res, part, part + rows, rows);
    pipe_barrier(PIPE_V);
  }
}

// Sum corresponding lanes of the column planes before the one 8-lane
// row reduction. Opposite accumulator phases match the residual plane reads.
F12_FN void RowFold(
    Fp mat, Fp part, Fp res,
    uint32_t rows, uint32_t blocks, uint32_t capacity) {
  if ASCEND_IS_AIV {
    const uint32_t plane = 8 * rows;
    if ((plane * sizeof(float) & 511u) == 0 ||
        blocks < 4 || capacity < 2 * plane + 128) {
      WideRows<false>(mat, part, res, rows, blocks);
      return;
    }
    const uint32_t phase = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(part) -
        reinterpret_cast<uint64_t>(mat)) & 511u;
    auto first = part + ((256u - phase) & 511u) / sizeof(float);
    auto second = first + plane;
    const uint8_t repeats = static_cast<uint8_t>(rows / 8);
    WideMask(~uint64_t{0});
    vadd(first, mat, mat + plane, repeats, 1, 1, 1, 8, 8, 8);
    vadd(second, mat + 2 * plane, mat + 3 * plane,
         repeats, 1, 1, 1, 8, 8, 8);
    for (uint32_t block = 4; block < blocks; ++block) {
      auto dst = (block & 1) ? second : first;
      vadd(dst, dst, mat + block * plane, repeats, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    vadd(first, first, second, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcgadd(res, first, repeats, 1, 1, 8);
    pipe_barrier(PIPE_V);
  }
}

F12_FN void WideColAdd(
    Fp dst, Fp a, Fp b, uint32_t count,
    uint32_t slabs, uint32_t ds, uint32_t as, uint32_t bs) {
  if ASCEND_IS_AIV {
    const uint8_t repeats = static_cast<uint8_t>(slabs);
    for (uint32_t first = 0; first < count; first += 64) {
      const uint32_t width = count - first < 64 ? count - first : 64;
      if (width < 64) WideMask((uint64_t{1} << width) - 1);
      vadd(dst + first, a + first, b + first, repeats, 1, 1, 1, ds, as, bs);
      if (width < 64) WideMask(~uint64_t{0});
    }
  }
}

F12_FN void Cols(
    Fp mat, Fp part, Fp ones,
    Fp denom, Fp tmp, uint32_t h, uint32_t valid,
    float eps) {
  if ASCEND_IS_AIV {
    if (h == 88) {
      const uint32_t slabs = 11 * valid;
      const uint32_t stride = 9;  // 9 blocks = 72 floats (8 data + 1 pad)
      auto buf0 = part;
      auto buf1 = part + slabs * 72;

      // Group 0..10 are at mat + g * 64, inter-repeat stride = 88 blocks
      WideColAdd(buf0, mat + 0 * 64, mat + 1 * 64, 64, slabs,
                           stride, 88, 88);
      WideColAdd(buf1, mat + 2 * 64, mat + 3 * 64, 64, slabs,
                           stride, 88, 88);
      WideColAdd(buf0, buf0, mat + 4 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf1, buf1, mat + 5 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf0, buf0, mat + 6 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf1, buf1, mat + 7 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf0, buf0, mat + 8 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf1, buf1, mat + 9 * 64, 64, slabs, stride,
                           stride, 88);
      WideColAdd(buf0, buf0, mat + 10 * 64, 64, slabs, stride,
                           stride, 88);
      pipe_barrier(PIPE_V);

      WideColAdd(buf0, buf0, buf1, 64, slabs, stride, stride, stride);
      pipe_barrier(PIPE_V);

      // Scatter four part rows into contiguous column planes. The next
      // two folds keep the original addition associations with full vectors.
      const uint32_t span = slabs * 8;
      const uint8_t nSpan = static_cast<uint8_t>((span + 63) / 64);
      WideMask(uint64_t{0xffffffff});
      vadd(tmp, buf0, buf0 + 32, static_cast<uint8_t>(slabs),
           static_cast<uint8_t>(slabs), 1, 1, 1, stride, stride);
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(0, 2 * span);
      vadd(tmp, tmp, tmp + 2 * span,
           static_cast<uint8_t>((2 * span + 63) / 64), 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::COUNTER>(0, span);
      vadd(denom, tmp, tmp + span, nSpan, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadds(denom, denom, eps, nSpan, 1, 1, 8, 8);
      vdiv(denom, ones, denom, nSpan, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});

      auto mCol = mat;
      auto dCol = denom;
      for (uint32_t slab = 0; slab < slabs; ++slab) {
        vmul(mCol, mCol, dCol, 11, 1, 1, 0, 8, 8, 0);
        mCol += 88 * 8;
        dCol += 8;
      }
      pipe_barrier(PIPE_V);
      return;
    }

    const uint32_t q = h / 4, slabs = h / 8 * valid;
    WideColAdd(part, mat, mat + q * 8, q * 8, slabs, q, h,
                         h);
    pipe_barrier(PIPE_V);
    WideColAdd(part, part, mat + 2 * q * 8, q * 8, slabs, q,
                         q, h);
    pipe_barrier(PIPE_V);
    WideColAdd(part, part, mat + 3 * q * 8, q * 8, slabs, q,
                         q, h);
    pipe_barrier(PIPE_V);
    uint32_t height = q;
    while (height > 8) {
      const uint32_t half = height / 2, upper = height - half;
      WideColAdd(part, part, part + upper * 8, half * 8,
                           slabs, q, q, q);
      pipe_barrier(PIPE_V);
      height = upper;
    }
    const uint32_t span = slabs * 8;
    const uint32_t half = height / 2, upper = height - half;
    WideMask((uint64_t{1} << (half * 8)) - 1);
    vadd(tmp, part, part + upper * 8, slabs, slabs, 1, 1, 1, q, q);
    if (upper > half) {
      WideMask(255);
      vcopy(reinterpret_cast<Up>(tmp + half * span),
            reinterpret_cast<Up>(part + half * 8), slabs, 1,
            1, 1, q);
    }
    WideMask(~uint64_t{0});
    pipe_barrier(PIPE_V);
    const uint32_t pairs = upper / 2, next = upper - pairs;
    WideSpan<0>(tmp, tmp, tmp + next * span, pairs * span);
    pipe_barrier(PIPE_V);
    const uint8_t nSpan = static_cast<uint8_t>((span + 63) / 64);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vadd(denom, tmp, tmp + span, nSpan, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(denom, denom, eps, nSpan, 1, 1, 8, 8);
    vdiv(denom, ones, denom, nSpan, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
    auto mCol = mat;
    auto dCol = denom;
    const uint8_t nCol = static_cast<uint8_t>(h / 8);
    for (uint32_t slab = 0; slab < slabs; ++slab) {
      vmul(mCol, mCol, dCol, nCol, 1, 1, 0, 8, 8, 0);
      mCol += h * 8;
      dCol += 8;
    }
    pipe_barrier(PIPE_V);
  }
}

F12_FN void InitWide(
    Fp mat, Fp part, Fp tmp,
    Fp res, uint32_t h, uint32_t valid) {
  if ASCEND_IS_AIV {
    const uint32_t rows = h * valid, blocks = h / 8, plane = rows * 8;
    const uint8_t nPlane = static_cast<uint8_t>(plane / 64);
    WideMask(~uint64_t{0});
    WideRows<true>(mat, part, res, rows, blocks);
    vbrcb(reinterpret_cast<Up>(tmp),
          reinterpret_cast<Up>(res), 1, 8, rows / 8);
    pipe_barrier(PIPE_V);
    auto mBlock = mat;
    for (uint32_t block = 0; block < blocks; ++block) {
      vsub(mBlock, mBlock, tmp, nPlane, 1, 1, 1, 8, 8, 8);
      mBlock += plane;
    }
    pipe_barrier(PIPE_V);
    WideSpan<6>(mat, mat, mat, rows * h);
    pipe_barrier(PIPE_V);
  }
}

F12_FN void Rounds(
    Fp mat, Fp part, Fp tmp,
    Fp res, Fp ones, Fp denom,
    uint32_t h, uint32_t valid, float eps, uint32_t iterations,
    uint32_t capacity) {
  if ASCEND_IS_AIV {
    const uint32_t rows = h * valid, blocks = h / 8, plane = rows * 8;
    const uint8_t nPlane = static_cast<uint8_t>(plane / 64);
    const uint32_t planeB = plane * sizeof(float);
    const uint32_t phase =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(part) -
                              reinterpret_cast<uint64_t>(mat)) &
        511u;
    const uint32_t offset = ((0u - phase) & 255u) / sizeof(float);
    const uint32_t planePhase = planeB & 511u;
    auto coeff0 = part + offset;
    auto coeff1 =
        coeff0 + plane + (planePhase ? 0 : 64);
    const bool firstIsZero =
        ((phase + offset * sizeof(float)) & 511u) == 0;
    auto phase0 =
        firstIsZero ? coeff0 : coeff1;
    auto phase1 =
        firstIsZero ? coeff1 : coeff0;
    InitWide(mat, part, tmp, res, h, valid);

    // Round 0: specialized (no res += eps, mat += eps at end)
    if (iterations > 0) {
      RowFold(mat, part, res, rows, blocks, capacity);
      WideSpan<2>(res, ones, res, rows);
      pipe_barrier(PIPE_V);
      vbrcb(reinterpret_cast<Up>(phase1),
            reinterpret_cast<Up>(res), 1, 8, rows / 8);
      if (planePhase)
        vbrcb(reinterpret_cast<Up>(phase0),
              reinterpret_cast<Up>(res), 1, 8, rows / 8);
      auto mBlock = mat;
      for (uint32_t block = 0; block < blocks; ++block) {
        auto coeff =
            ((block * planeB) & 511u) ? phase0 : phase1;
        vmul(mBlock, mBlock, coeff, nPlane, 1, 1, 1, 8, 8, 8);
        mBlock += plane;
      }
      pipe_barrier(PIPE_V);
      WideSpan<1>(mat, mat, mat, rows * h, eps);
      pipe_barrier(PIPE_V);
      Cols(mat, part, ones, denom, tmp, h, valid, eps);
    }

    // Rounds 1 to iterations - 1: clean branch-free loop
    for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
      RowFold(mat, part, res, rows, blocks, capacity);
      const uint8_t nRows = static_cast<uint8_t>((rows + 63) / 64);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(0, rows);
      vadds(res, res, eps, nRows, 1, 1, 8, 8);
      vdiv(res, ones, res, nRows, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      vbrcb(reinterpret_cast<Up>(phase1),
            reinterpret_cast<Up>(res), 1, 8, rows / 8);
      if (planePhase)
        vbrcb(reinterpret_cast<Up>(phase0),
              reinterpret_cast<Up>(res), 1, 8, rows / 8);
      auto mBlock = mat;
      for (uint32_t block = 0; block < blocks; ++block) {
        auto coeff =
            ((block * planeB) & 511u) ? phase0 : phase1;
        vmul(mBlock, mBlock, coeff, nPlane, 1, 1, 1, 8, 8, 8);
        mBlock += plane;
      }
      pipe_barrier(PIPE_V);
      Cols(mat, part, ones, denom, tmp, h, valid, eps);
    }
  }
}

class WideBlockedKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const TilingData &t) {
    t_ = t;
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);
    tail_start_ = count_;
    tail_first_ = t_.tile_tokens;
    if (t_.iters >= 8 && t_.tile_tokens >= 3 && count_ > t_.tile_tokens) {
      const uint32_t remainder = count_ % t_.tile_tokens;
      if (remainder != 0) {
        const uint32_t pair = t_.tile_tokens + remainder;
        const uint32_t first = (pair + 1) / 2;
        const uint32_t second = pair / 2;
        const uint32_t oldRows = (t_.h * t_.tile_tokens + 247) / 248 +
                                 (t_.h * remainder + 247) / 248;
        const uint32_t newRows =
            (t_.h * first + 247) / 248 + (t_.h * second + 247) / 248;
        const uint32_t oldColumns = (t_.h_pad * t_.tile_tokens + 63) / 64 +
                                    (t_.h_pad * remainder + 63) / 64;
        const uint32_t newColumns =
            (t_.h_pad * first + 63) / 64 + (t_.h_pad * second + 63) / 64;
        if (newRows < oldRows && newColumns < oldColumns) {
          tail_start_ = count_ - pair;
          tail_first_ = first;
        }
      }
    }
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    const uint32_t rp = (t_.h * t_.tile_tokens + 7) / 8 * 8;
    const uint32_t span = t_.tile_tokens * t_.h_pad;
    const uint32_t ones = rp > 2 * span ? rp : 2 * span;
    const uint32_t tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;
    // Host budgets the complete-token staging area when it fits. Smaller
    // capacities remain valid through the existing bounded DMA chunk loops.
    partialB_ = t_.input_bytes;
    pipe_.InitBuffer(matrixBuf_, t_.matrix_bytes);
    pipe_.InitBuffer(gateBuf_, 2 * t_.gate_bytes + 32);
    pipe_.InitBuffer(partialBuf_, partialB_);
    pipe_.InitBuffer(tmpBuf_, tmp * sizeof(float));
    pipe_.InitBuffer(resultBuf_, rp * sizeof(float));
    pipe_.InitBuffer(workBuf_, rp * sizeof(float));
    pipe_.InitBuffer(onesBuf_, ones * sizeof(float));
    pipe_.InitBuffer(denomBuf_, span * sizeof(float));
    mat_ = matrixBuf_.Get<float>();
    gates_ = gateBuf_.Get<float>();
    part_ = partialBuf_.Get<float>();
    tmp_ = tmpBuf_.Get<float>();
    res_ = resultBuf_.Get<float>();
    ones_ = onesBuf_.Get<float>();
    denom_ = denomBuf_.Get<float>();
    Duplicate(res_, 0.0f, rp);
    Duplicate(ones_, 1.0f, ones);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += valid_) {
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left)
                                     : t_.tile_tokens;
      if (offset == tail_start_) {
        valid_ = tail_first_;
      }
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;
      CopyIn(start_ + offset);
      SetMaskNorm();
      PipeBarrier<PIPE_V>();
      Rounds(
          reinterpret_cast<Fp>(mat_.GetPhyAddr()),
          reinterpret_cast<Fp>(part_.GetPhyAddr()),
          reinterpret_cast<Fp>(tmp_.GetPhyAddr()),
          reinterpret_cast<Fp>(res_.GetPhyAddr()),
          reinterpret_cast<Fp>(ones_.GetPhyAddr()),
          reinterpret_cast<Fp>(denom_.GetPhyAddr()), t_.h,
          valid_, t_.eps, t_.iters, partialB_ / sizeof(float));
      PipeBarrier<PIPE_V>();
      ResetMask();
      CopyOut(start_ + offset);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    const auto id = pipe_.FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline void CopyIn(uint64_t first) {
    const DataCopyExtParams gate{static_cast<uint16_t>(valid_), 4 * t_.h,
                                 4 * (t_.input_width - t_.h), 0, 0};
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    const uint64_t base = first * t_.input_width;
    DataCopyPad(gates_, input_[base], gate, noPad);
    DataCopyPad(gates_[span_], input_[base + t_.h], gate, noPad);
    const uint32_t area = t_.h * t_.h;
    if (partialB_ / sizeof(float) >= 2 * area && valid_ > 1) {
      const DataCopyExtParams dense{1, 4 * area, 0, 0, 0};
      Sync<HardEvent::V_MTE2>();
      DataCopyPad(part_, input_[base + 2 * t_.h], dense, noPad);
      Sync<HardEvent::MTE2_V>();
      for (uint32_t token = 0; token < valid_; ++token) {
        if (token + 1 < valid_) {
          Sync<HardEvent::V_MTE2>();
          DataCopyPad(part_[((token + 1) & 1) * area],
                      input_[base + (token + 1) * t_.input_width + 2 * t_.h],
                      dense, noPad);
        }
        if (token == 0) Gates();
        for (uint32_t block = 0; block < t_.h / 8; ++block)
          WideCopy8(
              reinterpret_cast<Fp>(
                  mat_[(block * valid_ + token) * t_.h * 8].GetPhyAddr()),
              reinterpret_cast<Fp>(
                  part_[(token & 1) * area + block * 8].GetPhyAddr()),
              t_.h, 1, t_.h / 8);
        if (token + 1 < valid_) Sync<HardEvent::MTE2_V>();
      }
      PipeBarrier<PIPE_V>();
      return;
    }
    const uint32_t rowsFit = partialB_ / (sizeof(float) * t_.h);
    for (uint32_t token = 0; token < valid_; ++token) {
      for (uint32_t row = 0; row < t_.h;) {
        const uint32_t count =
            t_.h - row < rowsFit ? t_.h - row : rowsFit;
        const DataCopyExtParams dense{1, 4 * count * t_.h, 0, 0, 0};
        Sync<HardEvent::V_MTE2>();
        DataCopyPad(
            part_,
            input_[base + token * t_.input_width + 2 * t_.h + row * t_.h],
            dense, noPad);
        Sync<HardEvent::MTE2_V>();
        for (uint32_t block = 0; block < t_.h / 8; ++block)
          WideCopy8(
              reinterpret_cast<Fp>(
                  mat_[(block * valid_ + token) * t_.h * 8 + row * 8]
                      .GetPhyAddr()),
              reinterpret_cast<Fp>(
                  part_[block * 8].GetPhyAddr()),
              count, 1, t_.h / 8);
        PipeBarrier<PIPE_V>();
        row += count;
      }
    }
    Sync<HardEvent::V_MTE2>();
    Sync<HardEvent::MTE2_V>();
    Gates();
  }

  __aicore__ inline void Gates() {
    const uint32_t n = 2 * span_;
    auto numerator = tmp_[n];
    auto denominator = tmp_[2 * n];
    Duplicate(numerator, 1.0f, n);
    Muls(tmp_, gates_, -1.0f, n);
    Mins(tmp_, tmp_, 80.0f, n);
    Exp(tmp_, tmp_, n);
    Adds(denominator, tmp_, 1.0f, n);
    Div(gates_, numerator, denominator, n);
    Adds(gates_, gates_, t_.eps, span_);
    Muls(gates_[span_], gates_[span_], 2.0f, span_);
  }

  __aicore__ inline void CopyOut(uint64_t first) {
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams gate{1, 4 * span_, 0, 0, 0};
    DataCopyPad(pre_[first * t_.h], gates_, gate);
    DataCopyPad(post_[first * t_.h], gates_[span_], gate);
    const uint32_t rowsFit = partialB_ / (sizeof(float) * t_.h);
    for (uint32_t row = 0; row < rows_;) {
      const uint32_t count =
          rows_ - row < rowsFit ? rows_ - row : rowsFit;
      for (uint32_t block = 0; block < t_.h / 8; ++block)
        WideCopy8(
            reinterpret_cast<Fp>(
                part_[block * 8].GetPhyAddr()),
            reinterpret_cast<Fp>(
                mat_[block * rows_ * 8 + row * 8].GetPhyAddr()),
            count, t_.h / 8, 1);
      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams dense{1, 4 * count * t_.h, 0, 0, 0};
      DataCopyPad(comb_[first * t_.h * t_.h + row * t_.h], part_, dense);
      Sync<HardEvent::MTE3_V>();
      row += count;
    }
    Sync<HardEvent::MTE3_MTE2>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuf_, gateBuf_;
  TBuf<TPosition::VECCALC> partialBuf_, tmpBuf_, resultBuf_,
      workBuf_, onesBuf_, denomBuf_;
  LocalTensor<float> mat_, gates_, part_, tmp_, res_, ones_, denom_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  TilingData t_;
  uint64_t start_, count_, tail_start_;
  uint32_t valid_, rows_, span_, tail_first_, partialB_;
};

}  // namespace sinkhorn_f12

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f12_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_f12::WideBlockedKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#undef F12_FN

#endif  // SINKHORN_NORM_KERNELS_F12_WIDE_BLOCKED_H_
