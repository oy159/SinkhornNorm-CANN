#ifndef SINKHORN_NORM_KERNELS_F04_NARROW_SINGLE_H_
#define SINKHORN_NORM_KERNELS_F04_NARROW_SINGLE_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_f04 {

using namespace AscendC;
using sinkhorn_norm::TilingData;

// P24 row-prefold and eps-first serial column sums for single-owner H in [17, 24]
// (including C3: H=22, N=26, iter=20). Compile-time H makes row/tail masks and
// counter lengths immediate constants with zero runtime mask shifts.
template <uint32_t H>
__aicore__ __attribute__((always_inline)) inline void RunPackedSerialRoundsP24(
    __ubuf__ float *matrix, __ubuf__ float *tmp, __ubuf__ float *result,
    __ubuf__ float *columns, __ubuf__ float *denom, __ubuf__ float *work, float eps,
    uint32_t rounds) {
  static_assert(H >= 17 && H <= 24, "Use the static P24 family");
  if ASCEND_IS_AIV {
    constexpr uint64_t rowMask = (uint64_t{1} << H) - 1;
    constexpr uint64_t tailMask =
        ((uint64_t{1} << (H - 16)) - 1) * uint64_t{0x0101010101010101};
    SetMaskNorm();
    for (uint32_t round = 0; round < rounds; ++round) {
      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, ~uint64_t{0});
      vadd(tmp, matrix, matrix + 8, 3, 1, 3, 3, 8, 24, 24);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, tailMask);
      vadd(tmp, tmp, matrix + 16, 3, 1, 1, 3, 8, 8, 24);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, ~uint64_t{0});
      vcgadd(result, tmp, 3, 1, 1, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, rowMask);
      vadds(result, result, eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);

      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, ~uint64_t{0});
      // Broadcast one 8-lane block per row; all three matrix blocks
      // consume this same compact denominator tile.
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(columns),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 3);
      SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, rowMask);
      if constexpr (H % 2 == 0) {
        // work[24] aliases denom. Seed epsilon in the second accumulator,
        // allowing the final two partials to combine with one vadd.
        SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << 24) - 1);
        vector_dup(denom, eps, 1, 1, 1, 8, 0);
        vector_dup(work, 0.0f, 1, 1, 1, 8, 0);
      } else {
        vector_dup(denom, eps, 1, 1, 1, 8, 0);
        SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << 48) - 1);
        vector_dup(tmp, 0.0f, 1, 1, 1, 8, 0);
      }
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(uint64_t{0}, uint64_t{H * 8});
      for (uint32_t block = 0; block < 3; ++block) {
        vdiv(matrix + 8 * block, matrix + 8 * block, columns,
             1, 3, 3, 1, 24, 24, 8);
      }
      pipe_barrier(PIPE_V);

      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << 48) - 1);
      // Even and odd rows accumulate in separate lanes of the same repeat.
      auto *partial = H % 2 == 0 ? work : tmp;
      vadd(partial, matrix, partial, H / 2, 1, 1, 1, 0, 6, 0);
      if constexpr (H % 2 != 0) {
        SetVectorMask<float, MaskMode::NORMAL>(0, rowMask);
        vadd(denom, matrix + (H - 1) * 24, denom, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(0, rowMask);
      if constexpr (H % 2 == 0) {
        vadd(denom, work, denom, 1, 1, 1, 1, 8, 8, 8);
      } else {
        vadd(denom, tmp, denom, 2, 1, 1, 1, 0, 3, 0);
      }
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(uint64_t{0}, uint64_t{H * 8});
      for (uint32_t block = 0; block < 3; ++block) {
        vdiv(matrix + 8 * block, matrix + 8 * block, denom + 8 * block, 1, 3, 3,
             0, 24, 24, 0);
      }
      pipe_barrier(PIPE_V);
      SetMaskNorm();
    }
    ResetMask();
  }
}

template <uint32_t StaticH>
class NarrowSingleP24Kernel {
  static_assert(StaticH >= 17 && StaticH <= 24, "Complete fixed P24 family");
  static constexpr uint32_t P = 24;
  static constexpr uint32_t kTileBytes =
      4 * StaticH * P + ((4 * (P - StaticH) * P + 511) / 512) * 512 + 8 * P;

 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, float eps, uint32_t iters) {
    eps_ = eps;
    iters_ = iters;
    constexpr uint32_t width = StaticH - 16;
    packedRowMask_[0] =
        ((uint64_t{1} << width) - 1) * uint64_t{0x0101010101010101};
    packedRowMask_[1] = 0;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    preGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    combGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    matrix_ = LocalTensor<float>(TPosition::VECOUT, 0, kTileBytes / 4);
    tmp_ = LocalTensor<float>(TPosition::VECCALC, kTileBytes, 8 * P);
    work_ = LocalTensor<float>(TPosition::VECCALC, kTileBytes + 32 * P, P);
    denom_ = LocalTensor<float>(TPosition::VECCALC, kTileBytes + 36 * P, P);
    result_ = LocalTensor<float>(TPosition::VECCALC, kTileBytes + 40 * P, P);
    columns_ =
        LocalTensor<float>(TPosition::VECCALC, kTileBytes + 44 * P, P * P);
    Duplicate(denom_, 1.0f, P);
    Duplicate(result_, 0.0f, P);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    const uint64_t token = GetBlockIdx();
    constexpr uint32_t inputWidth = StaticH * (StaticH + 2);
    const DataCopyPadExtParams<float> pad{
        true, 0, static_cast<uint8_t>(P - StaticH), 0.0f};
    const DataCopyExtParams copy{static_cast<uint16_t>(StaticH + 2),
                                 StaticH * uint32_t{sizeof(float)}, 0, 0, 0};
    DataCopyPad(matrix_, inputGm_[token * inputWidth], copy, pad);
    Sync<HardEvent::MTE2_V>();
    if constexpr (StaticH < P) {
      Duplicate(matrix_[2 * P + StaticH * P], 0.0f, (P - StaticH) * P);
    }
    GatePair();
    Softmax();
    NormalizeColumns();
    PipeBarrier<PIPE_V>();
    RunPackedSerialRoundsP24<StaticH>(
        reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()) + 2 * P,
        reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(columns_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(work_.GetPhyAddr()), eps_,
        iters_ - 1);
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams gateCopy{1, StaticH * uint32_t{sizeof(float)}, 0, 0,
                                     0};
    DataCopyPad(preGm_[token * StaticH], matrix_, gateCopy);
    DataCopyPad(postGm_[token * StaticH], matrix_[P], gateCopy);
    const DataCopyExtParams matrixCopy{
        static_cast<uint16_t>(StaticH), StaticH * uint32_t{sizeof(float)}, 0, 0,
        0};
    DataCopyPad(combGm_[token * StaticH * StaticH], matrix_[2 * P], matrixCopy);
    Sync<HardEvent::MTE3_MTE2>();
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void GatePair() {
    constexpr uint32_t count = 2 * P;
    auto absExp = tmp_;
    auto numerator = tmp_[count];
    auto denominator = tmp_[2 * count];
    Muls(absExp, matrix_, -1.0f, count);
    Duplicate(numerator, 1.0f, count);
    Mins(absExp, absExp, 80.0f, count);
    Exp(absExp, absExp, count);
    Adds(denominator, absExp, 1.0f, count);
    Div(matrix_, numerator, denominator, count);
    Adds(matrix_, matrix_, eps_, StaticH);
    Muls(matrix_[P], matrix_[P], 2.0f, StaticH);
  }

  template <bool Maximum>
  __aicore__ inline void ReduceRows() {
    if constexpr (Maximum) {
      Max(tmp_, matrix_[2 * P], matrix_[2 * P + 8], uint64_t{64}, uint8_t{3},
          {1, 3, 3, 8, 24, 24});
      Max(tmp_, tmp_, matrix_[2 * P + 16], packedRowMask_, uint8_t{3},
          {1, 1, 3, 8, 8, 24});
      BlockReduceMax(result_, tmp_, int32_t{3}, int32_t{64}, 1, 1, 8);
    } else {
      Add(tmp_, matrix_[2 * P], matrix_[2 * P + 8], uint64_t{64}, uint8_t{3},
          {1, 3, 3, 8, 24, 24});
      Add(tmp_, tmp_, matrix_[2 * P + 16], packedRowMask_, uint8_t{3},
          {1, 1, 3, 8, 8, 24});
      BlockReduceSum(result_, tmp_, int32_t{3}, int32_t{64}, 1, 1, 8);
    }
  }

  __aicore__ inline void Softmax() {
    ReduceRows<true>();
    Brcb(tmp_, result_, uint8_t{3}, {1, 8});
    Sub(matrix_[2 * P], matrix_[2 * P], tmp_, uint64_t{StaticH},
        static_cast<uint8_t>(StaticH), {1, 1, 0, 3, 3, 1});
    Exp(matrix_[2 * P], matrix_[2 * P], StaticH * P);
    ReduceRows<false>();
    for (uint32_t b = 0; b < 3; ++b) {
      Brcb(columns_[8 * b], result_, uint8_t{3}, {3, P});
    }
    Div(matrix_[2 * P], matrix_[2 * P], columns_, StaticH * P);
    Adds(matrix_[2 * P], matrix_[2 * P], eps_, StaticH * P);
  }

  __aicore__ inline void NormalizeColumns() {
    Duplicate(denom_, eps_, StaticH);
    Add(denom_, matrix_[2 * P], denom_, uint64_t{StaticH},
        static_cast<uint8_t>(StaticH), {1, 1, 1, 0, 3, 0});
    Copy(columns_, denom_, uint64_t{P}, static_cast<uint8_t>(StaticH),
         {1, 1, 3, 0});
    Div(matrix_[2 * P], matrix_[2 * P], columns_, StaticH * P);
  }

  GlobalTensor<float> inputGm_, preGm_, postGm_, combGm_;
  LocalTensor<float> matrix_, tmp_, work_, denom_, result_, columns_;
  uint64_t packedRowMask_[2];
  uint32_t iters_;
  float eps_;
};

}  // namespace sinkhorn_f04

template <uint32_t StaticH>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f04_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  sinkhorn_f04::NarrowSingleP24Kernel<StaticH> op;
  op.Init(mixes, pre, post, comb, tiling.eps, tiling.iters);
  op.Process();
  PipeBarrier<PIPE_ALL>();
}

#endif  // SINKHORN_NORM_KERNELS_F04_NARROW_SINGLE_H_
