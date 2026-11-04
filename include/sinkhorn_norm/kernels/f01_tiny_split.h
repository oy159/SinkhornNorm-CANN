#ifndef SINKHORN_NORM_KERNELS_F01_TINY_SPLIT_H_
#define SINKHORN_NORM_KERNELS_F01_TINY_SPLIT_H_
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "sinkhorn_norm/tiling_data.h"
namespace sinkhorn_f01 {
using namespace AscendC;
using sinkhorn_norm::TilingData;
// Dual-block gate/matrix split for H in [2, 8], N=1, iter=1.
template <uint32_t StaticH = 0>
class TinySplitKernel {
  static constexpr uint32_t P = 8;
  static constexpr uint32_t kTileBytes = 4 * P * P + 512 + 8 * P;
  static constexpr uint32_t kTmpOffset = kTileBytes;
  static constexpr uint32_t kDenomOffset = kTmpOffset + 8 * P * 4 + P * 4;
  static constexpr uint32_t kResultOffset = kDenomOffset + P * 4;
  static constexpr uint32_t kPatternOffset = kResultOffset + P * 4;
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const TilingData &t) {
    if constexpr (StaticH != 0) {
      h_ = StaticH;
    } else {
      h_ = t.h;
    }
    eps_ = t.eps;
  }
  __aicore__ inline void Process(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                                GM_ADDR comb) {
    constexpr uint32_t effH = StaticH != 0 ? StaticH : 0;
    const uint32_t hVal = effH != 0 ? effH : h_;
    const bool gates = (GetBlockIdx() & 1) == (hVal > 2 ? 1 : 0);
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    matrix_ = LocalTensor<float>(TPosition::VECCALC, 0, kTileBytes / 4);
    tmp_ = LocalTensor<float>(TPosition::VECCALC, kTmpOffset, 8 * P);
    if (gates) {
      // Issue MTE2 GM read immediately on the first cycles of the Gate core.
      if ASCEND_IS_AIV { set_mov_pad_val(0); }
      asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
                          reinterpret_cast<__gm__ float *>(mixes),
                          2, hVal * 4, 0, P - hVal, 0, 0);
      // Overlap GlobalTensor binding and numerator=1.0 initialization with DMA.
      preGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
      postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
      if ASCEND_IS_AIV {
        auto *gTmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << (2 * P)) - 1);
        vector_dup(gTmp + 2 * P, 1.0f, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
      }
      Sync<HardEvent::MTE2_V>();
      GatePair(hVal);
      Sync<HardEvent::V_MTE3>();
      asc_copy_ub2gm_align(reinterpret_cast<__gm__ float *>(pre),
                          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()), hVal * 4);
      asc_copy_ub2gm_align(reinterpret_cast<__gm__ float *>(post),
                          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()) + P, hVal * 4);
    } else {
      // Issue MTE2 GM read immediately on the first cycles of the Matrix core.
      if ASCEND_IS_AIV { set_mov_pad_val(0); }
      asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()) + 2 * P,
                          reinterpret_cast<__gm__ float *>(mixes) + 2 * hVal,
                          hVal, hVal * 4, 0, P - hVal, 0, 0);
      combGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
      denom_ = LocalTensor<float>(TPosition::VECCALC, kDenomOffset, P);
      result_ = LocalTensor<float>(TPosition::VECCALC, kResultOffset, P);
      pattern_ = LocalTensor<uint32_t>(TPosition::VECCALC, kPatternOffset, P);
      uint64_t mask0 = 0;
      if constexpr (StaticH != 0) {
        constexpr uint64_t columns =
            uint64_t{0x0101010101010101} * ((uint64_t{1} << StaticH) - 1);
        constexpr uint64_t rows =
            StaticH == 8 ? ~uint64_t{0} : (~uint64_t{0} >> (64 - 8 * StaticH));
        mask0 = columns & rows;
      } else {
        const uint64_t columns =
            uint64_t{0x0101010101010101} * ((uint64_t{1} << hVal) - 1);
        const uint64_t rows =
            hVal == 8 ? ~uint64_t{0} : (~uint64_t{0} >> (64 - 8 * hVal));
        mask0 = columns & rows;
      }
      const bool packOutput = (hVal < 8) && (hVal * hVal * sizeof(float) > 32);
      if (packOutput) {
        const uint32_t rowBits = ((uint32_t{1} << hVal) - 1) * 0x01010101U;
        Duplicate(pattern_, rowBits, P);
      }
      const float scale = (StaticH == 5) ? 6.0f :
                          (hVal == 5) ? 6.0f :
                          (hVal == 2) ? 3.0f :
                          (hVal == 3) ? 4.0f :
                          (hVal == 4) ? 5.0f :
                          (hVal == 6) ? 7.0f :
                          (hVal == 7) ? 8.0f :
                          (hVal == 8) ? 9.0f : 6.0f;
      const float seededEps = scale * eps_;
      Duplicate(result_, 0.0f, P);
      Duplicate(denom_, seededEps, P);
      if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(0, mask0);
        pipe_barrier(PIPE_V);
      }
      Sync<HardEvent::MTE2_V>();
      RunMatrixSingleRound(hVal, mask0);
      if (packOutput) {
        uint64_t packedCount = 0;
        GatherMask(tmp_, matrix_[2 * P], pattern_, true, hVal * P, {1, 1, 8, 0},
                   packedCount);
      }
      Sync<HardEvent::V_MTE3>();
      if (packOutput) {
          asc_copy_ub2gm_align(reinterpret_cast<__gm__ float *>(comb),
                            reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()), hVal * hVal * 4);
      } else {
          asc_copy_ub2gm_align(reinterpret_cast<__gm__ float *>(comb),
                            reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()) + 2 * P,
                            hVal, hVal * 4, 0, 0, 0, 0);
      }
    }
  }
 private:
  template <HardEvent Event>
  __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
  }
  __aicore__ inline void GatePair(uint32_t hVal) {
    if ASCEND_IS_AIV {
      auto *gates = reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
      auto *gTmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
      auto *gNum = gTmp + 2 * P;
      auto *gDenom = gTmp + 4 * P;
      // Vector mask is already preset to (1ULL << (2 * P)) - 1 during DMA.
      vmuls(gTmp, gates, -1.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vmins(gTmp, gTmp, 80.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vexp(gTmp, gTmp, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadds(gDenom, gTmp, 1.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(gates, gNum, gDenom, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << hVal) - 1);
      vadds(gates, gates, eps_, 1, 1, 1, 8, 8);
      vmuls(gates + P, gates + P, 2.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      ResetMask();
    }
  }
  __aicore__ inline void RunMatrixSingleRound(uint32_t hVal, uint64_t mask0) {
    if ASCEND_IS_AIV {
      auto *mat =
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()) + 2 * P;
      auto *tmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
      auto *denom = reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr());
      auto *result = reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr());
      // Vector mask is already preset to mask0 during MTE2 DMA transfer.
      vcgmax(result, mat, 1, 1, 1, 8);
      pipe_barrier(PIPE_V);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 1);
      pipe_barrier(PIPE_V);
      vsub(mat, mat, tmp, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vexp(mat, mat, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vcgadd(result, mat, 1, 1, 1, 8);
      pipe_barrier(PIPE_V);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 1);
      pipe_barrier(PIPE_V);
      // Div produces un-shifted softmax probabilities P_ij into tmp buffer.
      vdiv(tmp, mat, tmp, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      // Interleaved Dual-Consumer Window:
      // 1. Adds computes numerator: mat = P_ij + eps.
      // 2. vadd accumulates column sums directly from P_ij into denom pre-seeded with (hVal+1)*eps!
      // Both read tmp simultaneously with ZERO RAW conflicts, eliminating an entire pipe_barrier!
      vadds(mat, tmp, eps_, 1, 1, 1, 8, 8);
      SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << hVal) - 1);
      vadd(denom, tmp, denom, static_cast<uint8_t>(hVal), 1, 1, 1, 0, 1, 0);
      pipe_barrier(PIPE_V);
      // Final Column Div directly scales numerator into normalized comb output.
      SetVectorMask<float, MaskMode::NORMAL>(0, mask0);
      vdiv(mat, mat, denom, 1, 1, 1, 0, 8, 8, 0);
      pipe_barrier(PIPE_V);
      ResetMask();
    }
  }
  GlobalTensor<float> inputGm_, preGm_, postGm_, combGm_;
  LocalTensor<float> matrix_, tmp_, denom_, result_;
  LocalTensor<uint32_t> pattern_;
  uint32_t h_;
  float eps_;
};
}
template <uint32_t StaticH = 0>
__global__ __aicore__ void sinkhorn_f01_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  sinkhorn_f01::TinySplitKernel<StaticH> op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process(mixes, pre, post, comb);
  PipeBarrier<PIPE_ALL>();
}
#endif
