#ifndef SINKHORN_NORM_KERNELS_F14_WIDE_ROTATING128_H_
#define SINKHORN_NORM_KERNELS_F14_WIDE_ROTATING128_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_f14 {

using namespace AscendC;
using sinkhorn_norm::TilingData;

constexpr uint32_t kRotateNodeFloats = 8192 + 64;
struct RotateState {
  uint32_t node[5];
};

__aicore__ constexpr uint32_t RotateNodeAt(uint32_t phase, uint32_t idx) {
  constexpr uint32_t table[5][5] = {
      {4, 3, 1, 0, 2},
      {2, 0, 3, 4, 1},
      {1, 4, 0, 2, 3},
      {3, 2, 4, 1, 0},
      {0, 1, 2, 3, 4}};
  return table[phase % 5][idx];
}

__aicore__ constexpr RotateState RotateStateAt(uint32_t phase) {
  RotateState state{{4, 3, 1, 0, 2}};
  for (uint32_t stage = 0; stage < phase; ++stage) {
    state = RotateState{
        {state.node[4], state.node[3], state.node[1], state.node[0], state.node[2]}};
  }
  return state;
}

template <uint32_t Node, uint32_t Logical>
__aicore__ __attribute__((always_inline)) inline __ubuf__ float *RotateView(
    __ubuf__ float *matrix, __ubuf__ float *spill0) {
  if constexpr (Node == 4) {
    if constexpr (Logical == 0) return spill0;
    else return spill0 - 64;
  } else {
    constexpr uint32_t phase = (Logical == 0 ? 0 : 1);
    constexpr uint32_t offset = Node * kRotateNodeFloats + (((Node & 1) ^ phase) ? 64 : 0);
    return matrix + offset;
  }
}

__aicore__ __attribute__((always_inline)) inline __ubuf__ float *RotateView(
    __ubuf__ float *matrix, __ubuf__ float *spill0, uint32_t node,
    uint32_t logical) {
  if (node == 4) return spill0 - (logical == 0 ? 0 : 64);
  const uint32_t phase = logical == 0 ? 0 : 1;
  return matrix + node * kRotateNodeFloats + (((node & 1) ^ phase) ? 64 : 0);
}

template <uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void RotateRowParts(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *partial, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *ones, float eps, bool exact = true) {
  if ASCEND_IS_AIV {
    constexpr uint32_t n = 32 * Valid, rows = 128 * Valid;
    vcgadd(partial, a, n, 2, 1, 16);
    vcgadd(partial + 8, a + 64, n, 2, 1, 16);
    vcgadd(partial + n * 16, b, n, 2, 1, 16);
    vcgadd(partial + n * 16 + 8, b + 64, n, 2, 1, 16);
    vcgadd(partial + n * 32, c, n, 2, 1, 16);
    vcgadd(partial + n * 32 + 8, c + 64, n, 2, 1, 16);
    vcgadd(partial + n * 48, d, n, 2, 1, 16);
    vcgadd(partial + n * 48 + 8, d + 64, n, 2, 1, 16);
    pipe_barrier(PIPE_V);
    vcgadd(tmp, partial, rows / 4, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vcpadd(result, tmp, rows / 32, 1, 1, 8);
    pipe_barrier(PIPE_V);
    vadds(result, result, eps, rows / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    auto inverse = partial + 16 * rows;
    if (exact) {
      vdiv(inverse, ones + 64, result, rows / 64, 1, 1, 1, 8, 8, 8);
    } else {
      vrec(inverse, result, rows / 64, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
          reinterpret_cast<__ubuf__ uint32_t *>(inverse), 1, 16, rows / 8);
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp + 64),
          reinterpret_cast<__ubuf__ uint32_t *>(inverse), 1, 16, rows / 8);
    pipe_barrier(PIPE_V);
  }
}

template <uint32_t Valid, uint32_t Logical>
__aicore__ __attribute__((always_inline)) inline void RotateScaleRowQuarter(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *coeff) {
  if ASCEND_IS_AIV {
    constexpr uint32_t n = 32 * Valid;
    coeff += Logical * n * 16;
    #pragma unroll
    for (uint32_t row = 0; row < 8; ++row) {
      auto low = coeff + row * 8 + (Logical == 0 ? 64 : 0);
      auto high = coeff + row * 8 + (Logical == 0 ? 0 : 64);
      vmul(dst + row * 128, src + row * 128, low, n / 8, 1, 1, 0, 128, 128,
           16);
      vmul(dst + row * 128 + 64, src + row * 128 + 64, high, n / 8, 1, 1, 0,
           128, 128, 16);
    }
  }
}

template <uint32_t Valid, uint32_t Height>
__aicore__ __attribute__((always_inline)) inline __ubuf__ float *
RotateColumnTree(__ubuf__ float *src, __ubuf__ float *dst) {
  if constexpr (Height > 1) {
    if ASCEND_IS_AIV {
      constexpr uint32_t span = 128 * Valid;
      vadd(dst, src, src + Height / 2 * span, Height / 2 * span / 64, 1, 1, 1,
           8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    return RotateColumnTree<Valid, Height / 2>(dst, src);
  } else {
    return src;
  }
}

template <uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void RotateColumnParts(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *partial, __ubuf__ float *tmp, __ubuf__ float *ones,
    __ubuf__ float *denom, float eps, __ubuf__ float *scratchPhaseOne = nullptr,
    bool exact = true) {
  if ASCEND_IS_AIV {
    constexpr uint32_t span = 128 * Valid, quarter = 32 * span;
    __ubuf__ float *sum;
    if constexpr (Valid == 2) {
      constexpr uint32_t half = quarter / 2;
      auto phaseOne = scratchPhaseOne ? scratchPhaseOne : partial - 64;
      const bool firstUpper = phaseOne < partial;
      auto first = firstUpper ? partial + half : partial;
      auto second = firstUpper ? phaseOne : phaseOne + half;
      vadd(first, a, b, half / 64, 1, 1, 1, 8, 8, 8);
      vadd(tmp, a + half, b + half, half / 64, 1, 1, 1, 8, 8, 8);
      vadd(first, first, c, half / 64, 1, 1, 1, 8, 8, 8);
      vadd(tmp, tmp, c + half, half / 64, 1, 1, 1, 8, 8, 8);
      vadd(first, first, d, half / 64, 1, 1, 1, 8, 8, 8);
      vadd(second, tmp, d + half, half / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      constexpr uint32_t fragment = 4 * span;
      vadd(tmp, first, second, fragment / 64, 1, 1, 1, 8, 8, 8);
      vadd(tmp + fragment + 64, first + fragment, second + fragment,
           fragment / 64, 1, 1, 1, 8, 8, 8);
      vadd(tmp + 2 * fragment + 64, first + 2 * fragment,
           second + 2 * fragment, fragment / 64, 1, 1, 1, 8, 8, 8);
      vadd(tmp + 3 * fragment + 128, first + 3 * fragment,
           second + 3 * fragment, fragment / 64, 1, 1, 1, 8, 8, 8);
      vadd(partial, tmp, tmp + 2 * fragment + 64, fragment / 64, 1, 1, 1, 8, 8,
           8);
      vadd(partial + fragment + 64, tmp + fragment + 64,
           tmp + 3 * fragment + 128, fragment / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(tmp, partial, partial + fragment + 64, fragment / 64, 1, 1, 1, 8, 8,
           8);
      pipe_barrier(PIPE_V);
      sum = RotateColumnTree<Valid, 4>(tmp, partial);
    } else {
      vadd(partial, a, b, quarter / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(partial, partial, c, quarter / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(partial, partial, d, quarter / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      sum = RotateColumnTree<Valid, 32>(partial, tmp);
    }
    vadds(denom, sum, eps, span / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    if (exact) {
      vdiv(denom, ones + 64, denom, span / 64, 1, 1, 1, 8, 8, 8);
    } else {
      vrec(denom, denom, span / 64, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    vcopy(reinterpret_cast<__ubuf__ uint32_t *>(tmp + 64),
          reinterpret_cast<__ubuf__ uint32_t *>(denom), span / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }
}

template <uint32_t Valid, uint32_t Logical>
__aicore__ __attribute__((always_inline)) inline void RotateScaleColumnQuarter(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *tmp,
    __ubuf__ float *denom) {
  if ASCEND_IS_AIV {
    constexpr uint32_t span = 128 * Valid;
    auto factor = Logical == 0 ? tmp + 64 : denom;
    #pragma unroll
    for (uint32_t col = 0; col < span; col += 64)
      vmul(dst + col, src + col, factor + col, 32, 1, 1, 1, span / 8, span / 8,
           0);
  }
}

template <uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void RotateCommonRoundImpl(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *empty, __ubuf__ float *flippedA, __ubuf__ float *flippedC,
    __ubuf__ float *flippedEmpty, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *ones, __ubuf__ float *denom, float eps, bool exact = true) {
  RotateRowParts<Valid>(a, b, c, d, empty, tmp, result, ones, eps, exact);
  RotateScaleRowQuarter<Valid, 0>(empty, a, tmp);
  RotateScaleRowQuarter<Valid, 3>(flippedA, d, tmp);
  RotateScaleRowQuarter<Valid, 1>(d, b, tmp);
  RotateScaleRowQuarter<Valid, 2>(b, c, tmp);
  pipe_barrier(PIPE_V);
  RotateColumnParts<Valid>(empty, d, b, flippedA, flippedC, tmp, ones, denom,
                           eps, c, exact);
  RotateScaleColumnQuarter<Valid, 0>(flippedC, empty, tmp, denom);
  RotateScaleColumnQuarter<Valid, 3>(flippedEmpty, flippedA, tmp, denom);
  RotateScaleColumnQuarter<Valid, 1>(flippedA, d, tmp, denom);
  RotateScaleColumnQuarter<Valid, 2>(d, b, tmp, denom);
  pipe_barrier(PIPE_V);
}

template <uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void RotateCommonRound(
    __ubuf__ float *a, __ubuf__ float *b, __ubuf__ float *c, __ubuf__ float *d,
    __ubuf__ float *empty, __ubuf__ float *flippedA, __ubuf__ float *flippedC,
    __ubuf__ float *flippedEmpty, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *ones, __ubuf__ float *denom, float eps, bool exact = true) {
  RotateCommonRoundImpl<Valid>(a, b, c, d, empty, flippedA, flippedC,
                               flippedEmpty, tmp, result, ones, denom, eps, exact);
}

template <uint32_t Valid, uint32_t Phase>
__aicore__ __attribute__((always_inline)) inline void RotateCompleteRound(
    __ubuf__ float *matrix, __ubuf__ float *spill, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *ones, __ubuf__ float *denom,
    float eps, bool exact = true) {
  constexpr uint32_t n0 = RotateNodeAt(Phase, 0);
  constexpr uint32_t n1 = RotateNodeAt(Phase, 1);
  constexpr uint32_t n2 = RotateNodeAt(Phase, 2);
  constexpr uint32_t n3 = RotateNodeAt(Phase, 3);
  constexpr uint32_t n4 = RotateNodeAt(Phase, 4);
  RotateCommonRoundImpl<Valid>(
      RotateView<n0, 0>(matrix, spill),
      RotateView<n1, 1>(matrix, spill),
      RotateView<n2, 2>(matrix, spill),
      RotateView<n3, 3>(matrix, spill),
      RotateView<n4, 0>(matrix, spill),
      RotateView<n0, 1>(matrix, spill),
      RotateView<n2, 0>(matrix, spill),
      RotateView<n4, 1>(matrix, spill), tmp, result, ones, denom, eps, exact);
}

template <uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void
RunFirstColumnsAndRotatingRoundsImpl(__ubuf__ float *matrix,
                                     __ubuf__ float *spill, __ubuf__ float *tmp,
                                     __ubuf__ float *result,
                                     __ubuf__ float *ones,
                                     __ubuf__ float *denom, float eps,
                                     uint32_t rounds) {
  constexpr uint32_t quarter = 32 * 128 * Valid;
  RotateColumnParts<Valid>(matrix, matrix + quarter + 64,
                           matrix + 2 * quarter + 64, matrix + 3 * quarter + 64,
                           spill, tmp, ones, denom, eps, nullptr, false);
  RotateScaleColumnQuarter<Valid, 0>(RotateView<4, 0>(matrix, spill), matrix,
                                     tmp, denom);
  RotateScaleColumnQuarter<Valid, 3>(RotateView<0, 3>(matrix, spill),
                                     matrix + 3 * quarter + 64, tmp, denom);
  RotateScaleColumnQuarter<Valid, 1>(RotateView<3, 1>(matrix, spill),
                                     matrix + quarter + 64, tmp, denom);
  RotateScaleColumnQuarter<Valid, 2>(RotateView<1, 2>(matrix, spill),
                                     matrix + 2 * quarter + 64, tmp, denom);
  pipe_barrier(PIPE_V);
  while (rounds >= 5) {
    RotateCompleteRound<Valid, 0>(matrix, spill, tmp, result, ones, denom, eps, rounds <= 4);
    RotateCompleteRound<Valid, 2>(matrix, spill, tmp, result, ones, denom, eps, rounds - 1 <= 4);
    RotateCompleteRound<Valid, 4>(matrix, spill, tmp, result, ones, denom, eps, rounds - 2 <= 4);
    RotateCompleteRound<Valid, 1>(matrix, spill, tmp, result, ones, denom, eps, rounds - 3 <= 4);
    RotateCompleteRound<Valid, 3>(matrix, spill, tmp, result, ones, denom, eps, rounds - 4 <= 4);
    rounds -= 5;
  }
  if (rounds > 0)
    RotateCompleteRound<Valid, 0>(matrix, spill, tmp, result, ones, denom, eps, rounds <= 4);
  if (rounds > 1)
    RotateCompleteRound<Valid, 2>(matrix, spill, tmp, result, ones, denom, eps, rounds - 1 <= 4);
  if (rounds > 2)
    RotateCompleteRound<Valid, 4>(matrix, spill, tmp, result, ones, denom, eps, rounds - 2 <= 4);
  if (rounds > 3)
    RotateCompleteRound<Valid, 1>(matrix, spill, tmp, result, ones, denom, eps, rounds - 3 <= 4);
}

__aicore__ __attribute__((always_inline)) inline void
RunFirstColumnsAndRotatingRounds1(__ubuf__ float *matrix, __ubuf__ float *spill,
                                  __ubuf__ float *tmp, __ubuf__ float *result,
                                  __ubuf__ float *ones, __ubuf__ float *denom,
                                  float eps, uint32_t rounds) {
  RunFirstColumnsAndRotatingRoundsImpl<1>(matrix, spill, tmp, result, ones,
                                          denom, eps, rounds);
}

__aicore__ __attribute__((always_inline)) inline void
RunFirstColumnsAndRotatingRounds2(__ubuf__ float *matrix, __ubuf__ float *spill,
                                  __ubuf__ float *tmp, __ubuf__ float *result,
                                  __ubuf__ float *ones, __ubuf__ float *denom,
                                  float eps, uint32_t rounds) {
  RunFirstColumnsAndRotatingRoundsImpl<2>(matrix, spill, tmp, result, ones,
                                          denom, eps, rounds);
}

template <bool Subtract, uint32_t Rows>
__aicore__ __attribute__((always_inline)) inline void EmitSkewRowRectangle(
    __ubuf__ float *matrix, __ubuf__ float *coefficients) {
  if ASCEND_IS_AIV {
    if constexpr (Rows <= 64) {
      for (uint32_t row = 0; row < Rows; row += 8) {
        auto m = matrix + row * 128;
        auto c = coefficients + row * 16;
        if constexpr (Subtract)
          vsub(m, m, c, 8, 1, 1, 0, 16, 16, 1);
        else
          vmul(m, m, c, 8, 1, 1, 0, 16, 16, 1);
      }
    } else {
      for (uint32_t row = 0; row < 8; ++row) {
        auto m = matrix + row * 128;
        auto c = coefficients + row * 8;
        if constexpr (Subtract)
          vsub(m, m, c, Rows / 8, 1, 1, 0, 128, 128, 16);
        else
          vmul(m, m, c, Rows / 8, 1, 1, 0, 128, 128, 16);
      }
    }
  }
}

template <bool Subtract, uint32_t Valid>
__aicore__ __attribute__((always_inline)) inline void ApplyCompleteSkewRowsImpl(
    __ubuf__ float *__restrict matrix,
    __ubuf__ float *__restrict coefficients) {
  if ASCEND_IS_AIV {
    constexpr uint32_t prefix = 32 * Valid;
    constexpr uint32_t rest = 96 * Valid;
    EmitSkewRowRectangle<Subtract, prefix>(matrix, coefficients + 64);
    EmitSkewRowRectangle<Subtract, prefix>(matrix + 64, coefficients);
    EmitSkewRowRectangle<Subtract, rest>(matrix + prefix * 128 + 64,
                                         coefficients + prefix * 16);
    EmitSkewRowRectangle<Subtract, rest>(matrix + prefix * 128 + 128,
                                         coefficients + prefix * 16 + 64);
  }
}

SINKHORN_AIV_NOINLINE void ApplyCompleteSkewRowsSub1(
    __ubuf__ float *__restrict matrix,
    __ubuf__ float *__restrict coefficients) {
  ApplyCompleteSkewRowsImpl<true, 1>(matrix, coefficients);
}

SINKHORN_AIV_NOINLINE void ApplyCompleteSkewRowsSub2(
    __ubuf__ float *__restrict matrix,
    __ubuf__ float *__restrict coefficients) {
  ApplyCompleteSkewRowsImpl<true, 2>(matrix, coefficients);
}

SINKHORN_AIV_NOINLINE void ApplyCompleteSkewRowsMul1(
    __ubuf__ float *__restrict matrix,
    __ubuf__ float *__restrict coefficients) {
  ApplyCompleteSkewRowsImpl<false, 1>(matrix, coefficients);
}

SINKHORN_AIV_NOINLINE void ApplyCompleteSkewRowsMul2(
    __ubuf__ float *__restrict matrix,
    __ubuf__ float *__restrict coefficients) {
  ApplyCompleteSkewRowsImpl<false, 2>(matrix, coefficients);
}

class WideRotating128Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const TilingData &t) {
    t_ = t;
    t_.h = 128;
    t_.h_pad = 128;
    t_.input_width = 128 * (128 + 2);
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    const uint32_t rows = t_.h * t_.tile_tokens;
    const uint32_t span = t_.h_pad * t_.tile_tokens;
    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes);
    pipe_.InitBuffer(partialBuffer_, t_.input_bytes);
    pipe_.InitBuffer(tmpBuffer_, 17 * rows * sizeof(float));
    pipe_.InitBuffer(onesBuffer_, 2 * span * sizeof(float));
    pipe_.InitBuffer(denomBuffer_, span * sizeof(float));
    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>()[64];
    tmp_ = tmpBuffer_.Get<float>();
    result_ = tmp_[16 * rows];
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();
    Duplicate(result_, 0.0f, rows);
    Duplicate(ones_, 1.0f, 2 * span);
    const uint64_t one = (uint64_t{1} << 16) - 1;
    sumMask_[0] = 0;
    sumMask_[1] = 0;
    for (uint32_t row = 0; row < 4; ++row) sumMask_[0] |= one << (16 * row);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += valid_) {
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left)
                                     : t_.tile_tokens;
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;
      if (offset > 0) {
        Sync<HardEvent::V_MTE2>();
      }
      SkewCopyIn(start_ + offset);
      Sync<HardEvent::MTE2_V>();
      Gates();
      Softmax();
      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(64);
      PipeBarrier<PIPE_V>();
      auto matrix = reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
      auto partial = reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr());
      auto tmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
      auto result = reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr());
      auto ones = reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr());
      auto denom = reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr());
      if (valid_ == 2)
        RunFirstColumnsAndRotatingRounds2(matrix, partial, tmp, result, ones,
                                          denom, t_.eps, t_.iters - 1);
      else
        RunFirstColumnsAndRotatingRounds1(matrix, partial, tmp, result, ones,
                                          denom, t_.eps, t_.iters - 1);
      PipeBarrier<PIPE_V>();
      ResetMask();
      Sync<HardEvent::V_MTE3>();
      CopyRotatedOut(start_ + offset);
      Sync<HardEvent::MTE3_V>();
      Sync<HardEvent::MTE3_MTE2>();
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    const auto id = pipe_.FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline void SkewCopyIn(uint64_t first) {
    const uint32_t prefix = 32;
    const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
    const DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
    const DataCopyExtParams g{2, 4 * t_.h, 0, stride, 0};
    const DataCopyExtParams lo{static_cast<uint16_t>(prefix), 4 * t_.h, 0,
                               stride, 0};
    const DataCopyExtParams hi{static_cast<uint16_t>(t_.h - prefix), 4 * t_.h,
                               0, stride, 0};
    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t base = (first + token) * t_.input_width;
      DataCopyPad(gates_[token * t_.h_pad], input_[base], g, pad);
      DataCopyPad(matrix_[token * t_.h_pad], input_[base + 2 * t_.h], lo, pad);
      DataCopyPad(matrix_[prefix * span_ + 64 + token * t_.h_pad],
                  input_[base + (2 + prefix) * t_.h], hi, pad);
    }
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

  template <bool Maximum>
  __aicore__ inline void SkewCounterParts() {
    const uint32_t prefix = 32 * valid_;
    SetMaskCount();
    for (uint32_t segment = 0; segment < 2; ++segment) {
      const uint32_t first = segment == 0 ? 0 : prefix;
      const uint32_t count = segment == 0 ? prefix : rows_ - prefix;
      auto m = matrix_[first * t_.h_pad + (segment == 0 ? 0 : 64)];
      SetVectorMask<float, MaskMode::COUNTER>(0, count * t_.h);
      if (Maximum) {
        BlockReduceMax<float, false>(partial_[first * 16], m, 1,
                                     MASK_PLACEHOLDER, 1, 1, 8);
      } else {
        BlockReduceSum<float, false>(partial_[first * 16], m, 1,
                                     MASK_PLACEHOLDER, 1, 1, 8);
      }
    }
  }

  __aicore__ inline void BlockWideRowMax() {
    SkewCounterParts<true>();
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    const uint32_t groups = rows_ / 8;
    Max(tmp_, partial_, partial_[8], uint64_t{64}, static_cast<uint8_t>(groups),
        {1, 2, 2, 8, 16, 16});
    PipeBarrier<PIPE_V>();
    BlockReduceMax(result_, tmp_, static_cast<int32_t>(groups), int32_t{64}, 1,
                   1, 8);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void BroadcastRows() {
    const auto repeats = static_cast<uint8_t>(rows_ / 8);
    Brcb(partial_, result_, repeats, {1, 16});
    Brcb(partial_[64], result_, repeats, {1, 16});
  }

  template <bool Subtract>
  __aicore__ inline void SkewApplyRows() {
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(64);
    PipeBarrier<PIPE_V>();
    auto matrix = reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
    auto coefficients =
        reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr());
    if (valid_ == 2) {
      if constexpr (Subtract)
        ApplyCompleteSkewRowsSub2(matrix, coefficients);
      else
        ApplyCompleteSkewRowsMul2(matrix, coefficients);
    } else {
      if constexpr (Subtract)
        ApplyCompleteSkewRowsSub1(matrix, coefficients);
      else
        ApplyCompleteSkewRowsMul1(matrix, coefficients);
    }
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
  }

  __aicore__ inline void RowSums() {
    SkewCounterParts<false>();
    SetMaskNorm();
    ResetMask();
    const uint32_t groups = rows_ / 4;
    BlockReduceSum(tmp_, partial_, static_cast<int32_t>(groups), sumMask_, 1, 1,
                   8);
    PairReduceSum(result_, tmp_, static_cast<int32_t>(rows_ / 32), int32_t{64},
                  1, 1, 8);
  }

  __aicore__ inline void Softmax() {
    BlockWideRowMax();
    BroadcastRows();
    SkewApplyRows<true>();
    const uint32_t first = 32 * valid_ * t_.h_pad;
    Exp(matrix_, matrix_, first);
    Exp(matrix_[first + 64], matrix_[first + 64], rows_ * t_.h_pad - first);
    RowSums();
    Div(result_, ones_, result_, rows_);
    BroadcastRows();
    SkewApplyRows<false>();
    Adds(matrix_, matrix_, t_.eps, first);
    Adds(matrix_[first + 64], matrix_[first + 64], t_.eps,
         rows_ * t_.h_pad - first);
  }

  __aicore__ inline void CopyRotatedOut(uint64_t first) {
    RotateState state;
    switch ((2 * (t_.iters - 1)) % 5) {
      case 0:
        state = RotateStateAt(0);
        break;
      case 1:
        state = RotateStateAt(1);
        break;
      case 2:
        state = RotateStateAt(2);
        break;
      case 3:
        state = RotateStateAt(3);
        break;
      default:
        state = RotateStateAt(4);
        break;
    }
    const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
    const DataCopyExtParams g{1, 4 * t_.h, 0, 0, 0};
    const DataCopyExtParams q{32, 4 * t_.h, stride, 0, 0};
    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t n = first + token;
      DataCopyPad(pre_[n * t_.h], gates_[token * t_.h_pad], g);
      DataCopyPad(post_[n * t_.h], gates_[span_ + token * t_.h_pad], g);
      for (uint32_t logical = 0; logical < 4; ++logical) {
        const auto node = state.node[logical];
        LocalTensor<float> source;
        if (node == 4) {
          auto allocation = partialBuffer_.Get<float>();
          source = allocation[(logical == 0 ? 64 : 0) + token * t_.h_pad];
        } else {
          source = matrix_[node * kRotateNodeFloats +
                           (((node & 1) ^ (logical == 0 ? 0U : 1U)) ? 64 : 0) +
                           token * t_.h_pad];
        }
        DataCopyPad(comb_[n * t_.h * t_.h + logical * 32 * t_.h], source, q);
      }
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_, gateBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_, tmpBuffer_, onesBuffer_,
      denomBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, result_, ones_, denom_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  TilingData t_;
  uint64_t start_, count_, sumMask_[2];
  uint32_t valid_, rows_, span_;
};

}  // namespace sinkhorn_f14

template <int = 0>
__global__ __aicore__ __attribute__((aligned(4096))) void sinkhorn_f14_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_f14::WideRotating128Kernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F14_WIDE_ROTATING128_H_
