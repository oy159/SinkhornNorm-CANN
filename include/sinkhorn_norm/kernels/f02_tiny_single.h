#ifndef SINKHORN_NORM_KERNELS_F02_TINY_SINGLE_H_
#define SINKHORN_NORM_KERNELS_F02_TINY_SINGLE_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
using namespace AscendC;

template <uint32_t H>
class F02_TinySingle {
  static_assert(H >= 2 && H < 8, "Tiny single geometry family: H in [2, 7]");
  static constexpr uint32_t P = 8;
  // 2 gate rows + 8 matrix rows (rows 0..H-1 live, row H = eps, rows H+1..7 = 0).
  static constexpr uint32_t kMatrixFloats = 10 * P;
  static constexpr uint32_t TmpOffset = 512;
  static constexpr uint32_t GateTmpOffset = TmpOffset + 8 * P * 4;
  static constexpr uint32_t DenomOffset = GateTmpOffset + 8 * P * 4;
  static constexpr uint32_t ResultOffset = DenomOffset + 8 * P * 4;
  static constexpr uint32_t PatternOffset = ResultOffset + 8 * P * 4;
  // Private scratch stays inside the existing 8192-byte tiny-single reservation.
  static constexpr uint32_t TransposeOffset = 2048;
  static constexpr uint32_t IndexOffset = 2304;
  static constexpr uint32_t RowFactorOffset = 2560;
  static constexpr uint32_t ColFactorOffset = 2816;
  static constexpr uint32_t EpsFactorOffset = 3072;
  static_assert(EpsFactorOffset + 256 <= 8192, "Tiny-single UB budget");

  static constexpr uint64_t kCols =
      uint64_t{0x0101010101010101} * ((uint64_t{1} << H) - 1);
  // An independent identity block keeps padded factor lanes nonzero at eps=0.
  static constexpr uint64_t kPaddingDiagonal = uint64_t{0x8040201008040201} & ~kCols;
  static constexpr uint64_t kRows = ~uint64_t{0} >> (64 - 8 * H);
  static constexpr uint64_t kMask = kCols & kRows;
  // Columns 0..H in rows 0..H-1: includes column H (pre-populated with eps_).
  static constexpr uint64_t kRowSumCols =
      uint64_t{0x0101010101010101} * ((uint64_t{1} << (H + 1)) - 1);
  static constexpr uint64_t kRowSumMask = kRowSumCols & kRows;
  // Bitmask for pre-populating col H of rows 0..H-1 with eps_.
  static constexpr uint64_t kColHEpsMask =
      (uint64_t{0x0101010101010101} << H) & kRows;
  static constexpr uint64_t kGatePairMask = (uint64_t{1} << (2 * P)) - 1;
  static constexpr uint64_t kPreGateMask = (uint64_t{1} << H) - 1;
  static constexpr uint64_t kPostGateMask = ((uint64_t{1} << H) - 1) << P;

 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, float eps) {
    eps_ = eps;
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    preGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    combGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    matrix_ = LocalTensor<float>(TPosition::VECCALC, 0, kMatrixFloats);
    tmp_ = LocalTensor<float>(TPosition::VECCALC, TmpOffset, 8 * P);
    gateTmp_ = LocalTensor<float>(TPosition::VECCALC, GateTmpOffset, 8 * P);
    denom_ = LocalTensor<float>(TPosition::VECCALC, DenomOffset, 8 * P);
    result_ = LocalTensor<float>(TPosition::VECCALC, ResultOffset, 8 * P);
    pattern_ = LocalTensor<uint32_t>(TPosition::VECCALC, PatternOffset, P);
    transpose_ = LocalTensor<float>(TPosition::VECCALC, TransposeOffset, 64);
    transposeIndex_ = LocalTensor<uint32_t>(TPosition::VECCALC, IndexOffset, 64);
    rowFactor_ = LocalTensor<float>(TPosition::VECCALC, RowFactorOffset, 64);
    colFactor_ = LocalTensor<float>(TPosition::VECCALC, ColFactorOffset, 64);
    epsFactor_ = LocalTensor<float>(TPosition::VECCALC, EpsFactorOffset, 64);
  }

  template <bool MatrixOnly = false>
  __aicore__ inline void Process(uint32_t iters) {
    const uint64_t token = GetBlockIdx();
    const DataCopyExtParams inputCopy{
        static_cast<uint16_t>(MatrixOnly ? H : H + 2), H * uint32_t{sizeof(float)}, 0, 0, 0};
    const DataCopyPadExtParams<float> pad{
        true, 0, static_cast<uint8_t>(P - H), 0.0f};
    // Private input stage and map fit the existing 8192-byte reservation.
    auto denseInput = LocalTensor<float>(TPosition::VECCALC, 4096, 64);
    auto inputIndex = LocalTensor<int32_t>(TPosition::VECCALC, 4352, 64);
    if constexpr (MatrixOnly) {
      const DataCopyExtParams denseCopy{1, H * H * uint32_t{sizeof(float)}, 0, 0, 0};
      const DataCopyPadExtParams<float> densePad{false, 0, 0, 0.0f};
      DataCopyPad(denseInput, inputGm_[token * H * (H + 2) + 2 * H], denseCopy, densePad);
    } else {
      DataCopyPad(matrix_, inputGm_[token * H * (H + 2)], inputCopy, pad);
    }

    // Overlap UB constant setup with MTE2 input DMA transfer.
    Duplicate(result_, 0.0f, P);
    if constexpr (!MatrixOnly) Duplicate(gateTmp_[2 * P], 1.0f, 2 * P);
    if constexpr (H < P) {
      constexpr uint32_t rowBits = ((uint32_t{1} << H) - 1) * 0x01010101U;
      Duplicate(pattern_, rowBits, P);
    }
    const bool useFactors = H != 3 && eps_ >= 0x1p-40f && eps_ <= 1.0f;
    // Coefficients need a bounded positive epsilon. Keep direct normalization
    // for zero/extremely small epsilon and for large epsilon to avoid factor
    // underflow or overflow; no input tensor values are inspected on Host.
    if constexpr (MatrixOnly) {
      // Padding maps to the live first element of its row; masked math ignores it.
      for (uint32_t c=0;c<8;++c) inputIndex.SetValue(c, c<H ? c*4 : 0);
    }
    if (useFactors) {
      for (uint32_t c=0;c<8;++c) transposeIndex_.SetValue(c,c*32);
    }
    if (MatrixOnly || useFactors) {
      Sync<HardEvent::S_V>();
      SetMaskNorm();
      // Double the initialized rows at each level. Both maps share each fence.
      for (uint32_t rows = 1; rows < 8; rows *= 2) {
        SetVectorMask<int32_t, MaskMode::NORMAL>(
            0, (uint64_t{1} << (rows * 8)) - 1);
        if constexpr (MatrixOnly) {
          Adds<int32_t, false>(inputIndex[rows * 8], inputIndex,
                              static_cast<int32_t>(rows * H * 4), rows * 8);
        }
        if (useFactors) {
          auto index = transposeIndex_.ReinterpretCast<int32_t>();
          Adds<int32_t, false>(index[rows * 8], index,
                              static_cast<int32_t>(rows * 4), rows * 8);
        }
        PipeBarrier<PIPE_V>();
      }
      ResetMask();
    }
    Sync<HardEvent::MTE2_V>();
    if constexpr (MatrixOnly) {
      Gather(matrix_[2*P],denseInput,inputIndex.ReinterpretCast<uint32_t>(),uint32_t{0},H*P);
      PipeBarrier<PIPE_V>();
    }

    // Execute interleaved GatePair + Softmax + Sinkhorn rounds + GatherMask
    // in one unbroken PIPE_V stream with a single terminal V_MTE3 sync.
    RunTinySingleFused<MatrixOnly>(iters, useFactors);
    if constexpr (H < P) {
      uint64_t packedCount = 0;
      GatherMask(tmp_, matrix_[2 * P], pattern_, true, H * P, {1, 1, 8, 0},
                 packedCount);
    }

    Sync<HardEvent::V_MTE3>();
    if constexpr (!MatrixOnly) {
    const DataCopyExtParams gateCopy{1, H * uint32_t{sizeof(float)}, 0, 0, 0};
    DataCopyPad(preGm_[token * H], matrix_, gateCopy);
    DataCopyPad(postGm_[token * H], matrix_[P], gateCopy);
    }
    if constexpr (H < P) {
      const DataCopyExtParams matrixCopy{
          1, H * H * uint32_t{sizeof(float)}, 0, 0, 0};
      DataCopyPad(combGm_[token * H * H], tmp_, matrixCopy);
    } else {
      const DataCopyExtParams matrixCopy{
          static_cast<uint16_t>(H), H * uint32_t{sizeof(float)}, 0, 0, 0};
      DataCopyPad(combGm_[token * H * H], matrix_[2 * P], matrixCopy);
    }
  }

 private:
  template <HardEvent Event>
  __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
  }

  template <bool MatrixOnly>
  __aicore__ inline void RunTinySingleFused(uint32_t iters, bool useFactors) {
    if ASCEND_IS_AIV {
      auto *gates = reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
      auto *mat = gates + 2 * P;
      auto *tmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
      auto *gTmp = reinterpret_cast<__ubuf__ float *>(gateTmp_.GetPhyAddr());
      auto *denom = reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr());
      auto *result = reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr());
      auto *gNum = gTmp + 2 * P;
      auto *gDenom = gTmp + 4 * P;

      SetMaskNorm();

      // Window 1: Gate vmuls(-x) [VALU] || Softmax RowMax [VREDUCE] + clear rows H+1..7
      if constexpr (!MatrixOnly) {
      SetVectorMask<float, MaskMode::NORMAL>(0, kGatePairMask);
      vmuls(gTmp, gates, -1.0f, 1, 1, 1, 8, 8);
      }
      if constexpr (H < 7) {
        constexpr uint64_t kZeroRowsMask = ~uint64_t{0} << (8 * (H + 1));
        SetVectorMask<float, MaskMode::NORMAL>(0, kZeroRowsMask);
        vector_dup(mat, 0.0f, 1, 1, 1, 8, 0);
      }
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vcgmax(result, mat, 1, 1, 1, 8);
      pipe_barrier(PIPE_V);

      // Window 2: Gate vmins(80) [VALU] || Softmax vbrcb(row_max) [VBRCB]
      if constexpr (!MatrixOnly) {
      SetVectorMask<float, MaskMode::NORMAL>(0, kGatePairMask);
      vmins(gTmp, gTmp, 80.0f, 1, 1, 1, 8, 8);
      }
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 1);
      pipe_barrier(PIPE_V);

      // Window 3: Gate vexp [VSFU] || Softmax vsub [VALU]
      if constexpr (!MatrixOnly) {
      vexp(gTmp, gTmp, 1, 1, 1, 8, 8);
      }
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vsub(mat, mat, tmp, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);

      // Window 4: Gate vadds(+1) [VALU] || Softmax vexp [VSFU]
      if constexpr (!MatrixOnly) {
      SetVectorMask<float, MaskMode::NORMAL>(0, kGatePairMask);
      vadds(gDenom, gTmp, 1.0f, 1, 1, 1, 8, 8);
      }
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vexp(mat, mat, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);

      // Window 5: Gate vdiv [VDIV] || Softmax RowSum [VREDUCE]
      if constexpr (!MatrixOnly) {
      SetVectorMask<float, MaskMode::NORMAL>(0, kGatePairMask);
      vdiv(gates, gNum, gDenom, 1, 1, 1, 1, 8, 8, 8);
      }
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vcgadd(result, mat, 1, 1, 1, 8);
      pipe_barrier(PIPE_V);

      // Window 6: Gate pre(+eps)/post(*2) [VALU] || Softmax vbrcb(row_sum) [VBRCB]
      if constexpr (!MatrixOnly) {
      SetVectorMask<float, MaskMode::NORMAL>(0, kPreGateMask);
      vadds(gates, gates, eps_, 1, 1, 1, 8, 8);
      SetVectorMask<float, MaskMode::NORMAL>(0, kPostGateMask);
      vmuls(gates, gates, 2.0f, 1, 1, 1, 8, 8);
      }
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 1);
      pipe_barrier(PIPE_V);

      // Window 7: Softmax vdiv [VDIV] || pre-populate col H of mat with eps_ and denom with eps_ [VALU]
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vdiv(mat, mat, tmp, 1, 1, 1, 1, 8, 8, 8);
      SetVectorMask<float, MaskMode::NORMAL>(0, kColHEpsMask);
      vector_dup(mat, eps_, 1, 1, 1, 8, 0);
      SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << H) - 1);
      vector_dup(denom, eps_, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);

      // Window 8: Softmax vadds(+eps), leaving active mask at kMask!
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vadds(mat, mat, eps_, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);

      if (!useFactors) {
        RunDirectIterations(iters, mat, tmp, denom, result);
        ResetMask();
        return;
      }

      // Keep A and A^T fixed. The iterated state is M_ij = A_ij/(p_i*q_j).
      // Row update: p <- sum_j(A_ij/q_j) + eps*p; column update is symmetric.
      // The extra column carries epsilon; live entries of A and A^T stay fixed.
      // The initial Softmax already supplies the first row normalization.
      auto *at = reinterpret_cast<__ubuf__ float *>(transpose_.GetPhyAddr());
      auto *p = reinterpret_cast<__ubuf__ float *>(rowFactor_.GetPhyAddr());
      auto *q = reinterpret_cast<__ubuf__ float *>(colFactor_.GetPhyAddr());
      auto *oldEps = reinterpret_cast<__ubuf__ float *>(epsFactor_.GetPhyAddr());
      SetVectorMask<float, MaskMode::NORMAL>(0, ~kMask);
      vector_dup(mat, 0.0f, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(0, kPaddingDiagonal);
      vector_dup(mat, 1.0f, 1, 1, 1, 8, 0);
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      vector_dup(result, 0.0f, 1, 1, 1, 8, 0);
      vector_dup(p, 1.0f, 1, 1, 1, 8, 0);
      vector_dup(q, 1.0f, 1, 1, 1, 8, 0);
      vector_dup(tmp, 0.0f, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      Gather(transpose_, matrix_[2 * P], transposeIndex_, uint32_t{0}, uint32_t{64});
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      // The reserved column H divides by the padded coefficient one. Its
      // numerator contributes eps*self_old directly to the dot product.
      SetVectorMask<float, MaskMode::NORMAL>(0, kColHEpsMask);
      vector_dup(mat, eps_, 1, 1, 1, 8, 0);
      vector_dup(at, eps_, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      AugmentedHalf(at, mat, p, q, tmp, oldEps);
      for (uint32_t iter = 1; iter < iters; ++iter) {
        AugmentedHalf(mat, at, q, p, tmp, oldEps);
        AugmentedHalf(at, mat, p, q, tmp, oldEps);
      }
      // Materialize the final output exactly once; no reciprocal approximation.
      SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
      vdiv(tmp, mat, q, 1, 1, 1, 0, 8, 8, 0);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(gTmp),
            reinterpret_cast<__ubuf__ uint32_t *>(p), 1, 8, 1);
      pipe_barrier(PIPE_V);
      vdiv(mat, tmp, gTmp, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      ResetMask();
    }
  }

  __aicore__ inline void RunDirectIterations(uint32_t iters,
                                             __ubuf__ float *mat,
                                             __ubuf__ float *tmp,
                                             __ubuf__ float *denom,
                                             __ubuf__ float *result) {
      // Sinkhorn iterations: single-instruction hardware column accumulation via vadd repeat
      for (uint32_t iter = 0; iter < iters; ++iter) {
        if (iter > 0) {
          SetVectorMask<float, MaskMode::NORMAL>(0, kRowSumMask);
          vcgadd(result, mat, 1, 1, 1, 8);
          pipe_barrier(PIPE_V);
          vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
                reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, 1);
          pipe_barrier(PIPE_V);
          SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
          vdiv(mat, mat, tmp, 1, 1, 1, 1, 8, 8, 8);

          // Overlap vector_dup(denom, eps) on VALU with vdiv on VDIV pipe
          SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << H) - 1);
          vector_dup(denom, eps_, 1, 1, 1, 8, 0);
          pipe_barrier(PIPE_V);
        }

        SetVectorMask<float, MaskMode::NORMAL>(0, (uint64_t{1} << H) - 1);
        vadd(denom, mat, denom, static_cast<uint8_t>(H), 1, 1, 1, 0, 1, 0);
        pipe_barrier(PIPE_V);

        SetVectorMask<float, MaskMode::NORMAL>(0, kMask);
        vdiv(mat, mat, denom, 1, 1, 1, 0, 8, 8, 0);
        pipe_barrier(PIPE_V);
      }
  }

  __aicore__ inline void AugmentedHalf(
      __ubuf__ float *a, __ubuf__ float *nextA,
      __ubuf__ float *other, __ubuf__ float *self,
      __ubuf__ float *work, __ubuf__ float *broadcast) {
    // Stage 1: weighted matrix and independent preparation for the next half.
    vdiv(work, a, other, 1, 1, 1, 0, 8, 8, 0);
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(broadcast),
          reinterpret_cast<__ubuf__ uint32_t *>(other), 1, 8, 1);
    pipe_barrier(PIPE_V);
    // Stage 2: reduction includes epsilon; update only the opposite padding
    // column. Real matrix entries and padded identity rows remain unchanged.
    vcgadd(self, work, 1, 1, 1, 8);
    SetVectorMask<float, MaskMode::NORMAL>(0, kColHEpsMask);
    vmuls(nextA, broadcast, eps_, 1, 1, 1, 8, 8);
    SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
    pipe_barrier(PIPE_V);
  }

  GlobalTensor<float> inputGm_, preGm_, postGm_, combGm_;
  LocalTensor<float> matrix_, tmp_, gateTmp_, denom_, result_;
  LocalTensor<uint32_t> pattern_, transposeIndex_;
  LocalTensor<float> transpose_, rowFactor_, colFactor_, epsFactor_;
  float eps_;
};

class F02_ThreeRowPlanes {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR comb, uint32_t tokens, float eps) {
    tokens_ = tokens; eps_ = eps;
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    matrix_ = LocalTensor<float>(TPosition::VECCALC, 0, 128);
    row_ = LocalTensor<float>(TPosition::VECCALC, 512, 32);
    col_ = LocalTensor<float>(TPosition::VECCALC, 640, 32);
  }
  __aicore__ inline void Process(uint32_t iters) {
    const uint32_t first = 8 * GetBlockIdx(), left = tokens_ - first;
    const uint32_t valid = left < 8 ? left : 8;
    auto dense = LocalTensor<float>(TPosition::VECCALC, 4096, 128);
    auto in = LocalTensor<int32_t>(TPosition::VECCALC, 4608, 128);
    auto out = LocalTensor<int32_t>(TPosition::VECCALC, 5120, 128);
    auto ts = LocalTensor<int32_t>(TPosition::VECCALC, 768, 8);
    const DataCopyExtParams copy{1, 60 * valid, 0, 0, 0};
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(dense, input_[15 * first], copy, pad);
    Duplicate(in[72], int32_t{0}, 56);
#pragma unroll
    for (uint32_t t=0;t<8;++t) ts.SetValue(t, t < valid ? 60*t : 0);
    auto outputPairs = out.ReinterpretCast<uint64_t>();
#pragma unroll
    for (uint32_t q=0;q<72;q+=2) {
      const uint64_t lo = 4 * (8 * (q % 9) + q / 9);
      const uint64_t hi = 4 * (8 * ((q+1) % 9) + (q+1) / 9);
      outputPairs.SetValue(q/2, lo | (hi << 32));
    }
    Sync<HardEvent::S_V>(); SetMaskNorm();
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xff);
#pragma unroll
    for (uint32_t j=0;j<9;++j) Adds<int32_t,false>(in[8*j],ts,static_cast<int32_t>(24+4*j),8);
    PipeBarrier<PIPE_V>(); ResetMask();
    Sync<HardEvent::MTE2_V>();
    Gather(matrix_,dense,in.ReinterpretCast<uint32_t>(),uint32_t{0},128);
    PipeBarrier<PIPE_V>(); Compute(iters);
    Gather(dense,matrix_,out.ReinterpretCast<uint32_t>(),uint32_t{0},72);
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams result{1,36*valid,0,0,0};
    DataCopyPad(output_[9*first],dense,result);
  }
 private:
  template<HardEvent E> __aicore__ inline void Sync() {
    SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0);
  }
  __aicore__ inline void Compute(uint32_t iters) {
    if ASCEND_IS_AIV {
      auto *m=reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
      auto *r=reinterpret_cast<__ubuf__ float *>(row_.GetPhyAddr());
      auto *d=reinterpret_cast<__ubuf__ float *>(col_.GetPhyAddr());
      SetMaskNorm(); SetVectorMask<float,MaskMode::NORMAL>(0,0xffffff);
      vmax(r,m+8,m,1,1,3,3,0,0,0); pipe_barrier(PIPE_V);
      vmax(r,m+16,r,1,1,3,1,0,0,0); pipe_barrier(PIPE_V);
      vsub(m,m,r,3,1,1,0,3,3,1); pipe_barrier(PIPE_V);
      SetMaskCount(); set_vector_mask(uint64_t{0},uint64_t{72});
      vexp(m,m,1,1,1,8,8);
      SetMaskNorm(); SetVectorMask<float,MaskMode::NORMAL>(0,0xffffff);
      vector_dup(r,0.0f,1,1,1,8,0); pipe_barrier(PIPE_V);
      vadd(r,m,r,3,1,3,1,0,1,0); pipe_barrier(PIPE_V);
      vdiv(m,m,r,3,1,1,0,3,3,1);
      vector_dup(d,eps_,1,1,1,8,0); pipe_barrier(PIPE_V);
      SetMaskCount(); set_vector_mask(uint64_t{0},uint64_t{72});
      vadds(m,m,eps_,1,1,1,8,8);
      SetMaskNorm(); SetVectorMask<float,MaskMode::NORMAL>(0,0xffffff);
      pipe_barrier(PIPE_V);
      for(uint32_t i=0;i<iters;++i) {
        vadd(d,m,d,3,1,1,1,0,3,0); pipe_barrier(PIPE_V);
        vdiv(m,m,d,3,1,1,1,3,3,0);
        vector_dup(r,eps_,1,1,1,8,0); pipe_barrier(PIPE_V);
        if(i+1<iters) {
          vadd(r,m,r,3,1,3,1,0,1,0); pipe_barrier(PIPE_V);
          vdiv(m,m,r,3,1,1,0,3,3,1);
          vector_dup(d,eps_,1,1,1,8,0); pipe_barrier(PIPE_V);
        }
      }
      ResetMask();
    }
  }
  GlobalTensor<float> input_,output_;
  LocalTensor<float> matrix_,row_,col_;
  uint32_t tokens_; float eps_;
};

__aicore__ constexpr inline uint64_t GateWord(uint32_t w, uint32_t col) {
  uint64_t mask = 0;
  for (uint32_t b = 0; b < 64; ++b) {
    const uint32_t c = (64 * w + b) % 15;
    if (c >= col && c < col + 3) mask |= uint64_t{1} << b;
  }
  return mask;
}
template <uint32_t W = 0>
__aicore__ inline void ThreeGateMasks(LocalTensor<uint64_t> a, LocalTensor<uint64_t> b) {
  constexpr auto x = GateWord(W, 0), y = GateWord(W, 3);
  a.SetValue(W, x); b.SetValue(W, y);
  if constexpr (W < 15) ThreeGateMasks<W + 1>(a, b);
}
__aicore__ inline void ThreeGateRole(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                                    uint32_t tokens, float eps) {
  GlobalTensor<float> input, outPre, outPost;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
  outPre.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
  outPost.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
  const uint32_t n = 3 * tokens, p = (n + 63) / 64 * 64;
  auto data = LocalTensor<float>(TPosition::VECCALC, 0, 1024);
  auto gate = LocalTensor<float>(TPosition::VECCALC, 4096, 512);
  auto work = LocalTensor<float>(TPosition::VECCALC, 6144, 512);
  auto ones = LocalTensor<float>(TPosition::VECCALC, 8192, 512);
  auto pa = LocalTensor<uint64_t>(TPosition::VECCALC, 10240, 16);
  auto pb = LocalTensor<uint64_t>(TPosition::VECCALC, 10368, 16);
  const DataCopyExtParams in{1, 60 * tokens, 0, 0, 0};
  const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
  DataCopyPad(data, input, in, pad);
  Duplicate(gate, 0.0f, 2 * p);
  Duplicate(ones, 1.0f, 2 * p);
  ThreeGateMasks(pa, pb);
  SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  if ASCEND_IS_AIV {
    auto *g = reinterpret_cast<__ubuf__ float *>(gate.GetPhyAddr());
    auto *w = reinterpret_cast<__ubuf__ float *>(work.GetPhyAddr());
    auto *o = reinterpret_cast<__ubuf__ float *>(ones.GetPhyAddr());
    auto *d = reinterpret_cast<__ubuf__ uint32_t *>(data.GetPhyAddr());
    SetMaskCount(); set_vector_mask(uint64_t{0}, uint64_t{15 * tokens});
    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(g), d,
              reinterpret_cast<__ubuf__ uint32_t *>(pa.GetPhyAddr()), 1, 1, 0, 8, 0);
    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(g + p), d,
              reinterpret_cast<__ubuf__ uint32_t *>(pb.GetPhyAddr()), 1, 1, 0, 8, 0);
    SetMaskNorm(); SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
    pipe_barrier(PIPE_V);
    const uint8_t r = 2 * p / 64;
    vmuls(w, g, -1.0f, r, 1, 1, 8, 8); pipe_barrier(PIPE_V);
    vmins(w, w, 80.0f, r, 1, 1, 8, 8); pipe_barrier(PIPE_V);
    vexp(w, w, r, 1, 1, 8, 8); pipe_barrier(PIPE_V);
    vadds(w, w, 1.0f, r, 1, 1, 8, 8); pipe_barrier(PIPE_V);
    vdiv(g, o, w, r, 1, 1, 1, 8, 8, 8); pipe_barrier(PIPE_V);
    vadds(g, g, eps, p / 64, 1, 1, 8, 8);
    vmuls(g + p, g + p, 2.0f, p / 64, 1, 1, 8, 8);
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  const DataCopyExtParams out{1, 4 * n, 0, 0, 0};
  DataCopyPad(outPre, gate, out); DataCopyPad(outPost, gate[p], out);
}

// One role computes both gate outputs for a small token batch. Matrix roles
// keep their original arithmetic and write only comb; no inter-core exchange.
template <uint32_t H>
__aicore__ inline void RunBatchedGateRole(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                                         uint32_t tokens, float eps) {
  GlobalTensor<float> input, preOut, postOut;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
  preOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
  postOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
  TPipe pipe;
  TBuf<TPosition::VECCALC> arena;
  pipe.InitBuffer(arena, 32768);
  auto dense = arena.Get<float>();
  auto gates = dense[4096];
  auto work = dense[5120];
  auto ones = dense[6144];
  auto index = dense[7168].template ReinterpretCast<int32_t>();
  auto row = dense[7680].template ReinterpretCast<int32_t>();
  const uint32_t n = H * tokens, p = (n + 7) / 8 * 8;
  const DataCopyExtParams in{1, 4 * H * (H + 2) * tokens, 0, 0, 0};
  const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
  DataCopyPad(dense, input, in, noPad);
  CreateVecIndex(index, int32_t{0}, n);
  Duplicate(gates, 0.0f, 2 * p);
  Duplicate(ones, 1.0f, 2 * p);
  PipeBarrier<PIPE_V>();
  // Exact floor(i/H) for the bounded index domain i<448.
  Muls(row, index, int32_t{((1 << 20) + H - 1) / H}, n);
  PipeBarrier<PIPE_V>();
  ShiftRight(row, row, int32_t{20}, n);
  PipeBarrier<PIPE_V>();
  Muls(row, row, int32_t{H * (H + 1)}, n);
  PipeBarrier<PIPE_V>();
  Add(index, index, row, n);
  PipeBarrier<PIPE_V>();
  Muls(index, index, int32_t{4}, n);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  PipeBarrier<PIPE_V>();
  Gather(gates, dense, index.ReinterpretCast<uint32_t>(), uint32_t{0}, n);
  Gather(gates[p], dense, index.ReinterpretCast<uint32_t>(), uint32_t{4 * H}, n);
  PipeBarrier<PIPE_V>();
  Muls(work, gates, -1.0f, 2 * p);
  PipeBarrier<PIPE_V>();
  Mins(work, work, 80.0f, 2 * p);
  PipeBarrier<PIPE_V>();
  Exp(work, work, 2 * p);
  PipeBarrier<PIPE_V>();
  Adds(work, work, 1.0f, 2 * p);
  PipeBarrier<PIPE_V>();
  Div(gates, ones, work, 2 * p);
  PipeBarrier<PIPE_V>();
  Adds(gates, gates, eps, n);
  Muls(gates[p], gates[p], 2.0f, n);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  const DataCopyExtParams out{1, 4 * n, 0, 0, 0};
  DataCopyPad(preOut, gates, out);
  DataCopyPad(postOut, gates[p], out);
}

}  // namespace sinkhorn_norm

template <uint32_t H>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f02_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  if (GetBlockNum() == tiling.core_count + 1) {
    if (GetBlockIdx() == tiling.core_count) {
      sinkhorn_norm::RunBatchedGateRole<H>(mixes, pre, post, tiling.token_count, tiling.eps);
    } else {
      sinkhorn_norm::F02_TinySingle<H> op;
      op.Init(mixes, pre, post, comb, tiling.eps);
      op.template Process<true>(tiling.iters);
    }
  } else {
    sinkhorn_norm::F02_TinySingle<H> op;
    op.Init(mixes, pre, post, comb, tiling.eps);
    op.Process(tiling.iters);
  }
  PipeBarrier<PIPE_ALL>();
}

template <uint32_t H>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f02_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TinyThreeRowArgs args) {
  static_assert(H == 3, "Fused three-row geometry");
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  if (GetBlockIdx() + 1 == GetBlockNum()) {
    sinkhorn_norm::ThreeGateRole(mixes, pre, post,
        args.tokens, args.eps);
  } else {
    sinkhorn_norm::F02_ThreeRowPlanes op;
    op.Init(mixes, comb, args.tokens, args.eps);
    op.Process(args.iters);
  }
  PipeBarrier<PIPE_ALL>();
}

#endif  // SINKHORN_NORM_KERNELS_F02_TINY_SINGLE_H_
