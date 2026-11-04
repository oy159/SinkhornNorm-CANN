#ifndef SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_
#define SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f05 {

using namespace AscendC;

static __aicore__ __attribute__((always_inline)) inline void WideRoundMask(uint64_t mask) {
  SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, mask);
}

static __aicore__ __attribute__((always_inline)) inline void RunNarrowShortBatchGates(
    __ubuf__ float *gates, __ubuf__ float *tmp, __ubuf__ float *ones,
    uint32_t span, float eps) {
  if ASCEND_IS_AIV {
    const uint32_t n = 2 * span;
    const uint32_t gateRepeats = (n + 63) / 64;
    const uint32_t spanRepeats = (span + 63) / 64;
    const uint32_t spanFull = span / 64;
    const uint32_t spanTail = span % 64;

    SetMaskNorm();
    WideRoundMask(~uint64_t{0});
    vmuls(tmp, gates, -1.0f, gateRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmins(tmp, tmp, 80.0f, gateRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vexp(tmp, tmp, gateRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(tmp, tmp, 1.0f, gateRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(gates, ones, tmp, gateRepeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    if (spanFull != 0) {
      vadds(gates, gates, eps, spanFull, 1, 1, 8, 8);
    }
    if (spanTail != 0) {
      WideRoundMask((uint64_t{1} << spanTail) - 1);
      vadds(gates + spanFull * 64, gates + spanFull * 64, eps, 1, 1, 1, 8, 8);
      WideRoundMask(~uint64_t{0});
    }
    vmuls(gates + span, gates + span, 2.0f, spanRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    ResetMask();
  }
}

static __aicore__ __attribute__((always_inline)) inline void RunNarrowShortBatchCompleteRounds(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *ones,
    __ubuf__ float *denom, uint32_t h, uint32_t valid, float eps,
    uint32_t iters) {
  if ASCEND_IS_AIV {
    const uint32_t rows = h * valid, span = 16 * valid;
    const uint32_t groups4 = (rows + 3) / 4;
    const uint32_t pairs32 = (rows + 31) / 32;
    const uint32_t rowRepeats = (rows + 63) / 64;
    const uint32_t spanRepeats = (span + 63) / 64;
    const uint32_t matRepeats = (rows * 16 + 63) / 64;
    const uint32_t q4Repeats = (4 * span + 63) / 64;
    const uint32_t q4Stride = q4Repeats * 64;
    const uint32_t q2Repeats = (2 * span + 63) / 64;
    const uint32_t broadcasts = (rows + 7) / 8;
    const uint8_t spanBlocks = static_cast<uint8_t>(span / 8);
    const uint64_t pairMask =
        ((uint64_t{1} << h) - 1) * uint64_t{0x0001000100010001};
    const uint64_t colTailMask =
        (span & 63) ? ((uint64_t{1} << (span & 63)) - 1) : ~uint64_t{0};

    // Softmax subtract-max using 16-lane broadcast + single contiguous pairMask vsub.
    WideRoundMask(~uint64_t{0});
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial),
          reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial + 8),
          reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
    pipe_barrier(PIPE_V);
    WideRoundMask(pairMask);
    vsub(matrix, matrix, partial, groups4, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    WideRoundMask(~uint64_t{0});
    vexp(matrix, matrix, matRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // Unified loop for Round 0 (Softmax row-scale + col-norm) and Rounds 1..iters-1.
    for (uint32_t round = 0; round < iters; ++round) {
      WideRoundMask(pairMask);
      vcgadd(partial, matrix, groups4, 1, 1, 8);
      pipe_barrier(PIPE_V);
      WideRoundMask(~uint64_t{0});
      vcpadd(denom, partial, pairs32, 1, 1, 8);
      pipe_barrier(PIPE_V);
      if (round > 0) {
        vadds(denom, denom, eps, rowRepeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
      }
      vdiv(denom, ones, denom, rowRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);

      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial),
            reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial + 8),
            reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
      pipe_barrier(PIPE_V);
      vmul(matrix, matrix, partial, matRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      if (round == 0) {
        vadds(matrix, matrix, eps, matRepeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        if (h < 16) {
          vector_dup(matrix + rows * 16, 0.0f, 1, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
        }
      }

      // 4-stage parallel binary column tree
      vadd(partial, matrix, matrix + 4 * span, q4Repeats, 1, 1, 1, 8, 8, 8);
      if (h > 12) {
        vadd(partial + q4Stride, matrix + 8 * span, matrix + 12 * span,
             q4Repeats, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
      if (h > 12) {
        vadd(partial, partial, partial + q4Stride, q4Repeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      } else if (h > 8) {
        vadd(partial, partial, matrix + 8 * span, q4Repeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      }
      vadd(partial, partial, partial + 2 * span, q2Repeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(denom, partial, partial + span, spanRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadds(denom, denom, eps, spanRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(denom, ones, denom, spanRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);

      uint32_t col = 0;
      for (; col + 64 <= span; col += 64) {
        vmul(matrix + col, matrix + col, denom + col, h, 1, 1, 1, spanBlocks,
             spanBlocks, 0);
      }
      if (col < span) {
        WideRoundMask(colTailMask);
        vmul(matrix + col, matrix + col, denom + col, h, 1, 1, 1, spanBlocks,
             spanBlocks, 0);
      }
      pipe_barrier(PIPE_V);
    }
  }
}

class NarrowP16ShortBatchKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const sinkhorn_norm::TilingData &t) {
    t_ = t;
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);

    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    rp_ = (t_.h * t_.tile_tokens + 7) / 8 * 8;
    const uint32_t span = t_.tile_tokens * t_.h_pad;
    ones_cap_ = rp_ > 2 * span ? rp_ : 2 * span;
    const uint32_t tmp = 8 * rp_ > 6 * span ? 8 * rp_ : 6 * span;

    dense_gate_cap_ = (t_.h * t_.tile_tokens + 63) / 64 * 64;
    const uint32_t matrixCap = (t_.h * t_.h * t_.tile_tokens + 63) / 64 * 64;
    zero_offset_ = 2 * dense_gate_cap_ + matrixCap;
    input_gate_cap_ = (2 * span + 63) / 64 * 64;
    const uint32_t inputMatrixCap = (t_.h * span + 63) / 64 * 64;

    pipe_.InitBuffer(matrixBuffer_, 16 * span * sizeof(float) + 256);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes + 256);
    pipe_.InitBuffer(partialBuffer_,
                     t_.input_bytes + (tmp + 2 * rp_) * sizeof(float) + 256);
    pipe_.InitBuffer(onesBuffer_, ones_cap_ * sizeof(float) + 256);
    pipe_.InitBuffer(denomBuffer_, span * sizeof(float) + 256);
    pipe_.InitBuffer(denseOutputBuffer_, 4 * zero_offset_ + 32);
    pipe_.InitBuffer(inputIndexBuffer_,
                     4 * (input_gate_cap_ + inputMatrixCap));

    dense_output_ = denseOutputBuffer_.Get<float>();
    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>();
    tmp_ = partial_[t_.input_bytes / sizeof(float)];
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();
    input_index_ = inputIndexBuffer_.Get<int32_t>();
    constants_ready_ = false;
    input_indexed_valid_ = 0;
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t first = start_ + offset;
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left) : t_.tile_tokens;
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;

      CopyInAndBuildMaps(first);
      GatesAndRowMax();

      // Launch async Strided MTE3 writebacks for pre and post IMMEDIATELY
      // so they 100% overlap with CompleteRounds on PIPE_V!
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
      const uint32_t gStride = (valid_ - 1) * t_.h_pad / 8;
      const DataCopyExtParams gOut{1, 4 * t_.h, gStride, 0, 0};
      for (uint32_t token = 0; token < valid_; ++token) {
        const uint64_t base = first + token;
        DataCopyPad(pre_[base * t_.h], gates_[token * t_.h_pad], gOut);
        DataCopyPad(post_[base * t_.h], gates_[span_ + token * t_.h_pad], gOut);
      }

      SetMaskNorm();
      RunNarrowShortBatchCompleteRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          t_.h, valid_, t_.eps, t_.iters);
      ResetMask();

      // Zero-gather, zero-dynamic-map direct hardware strided comb writeback!
      CopyOutCombStrided(first, offset + t_.tile_tokens < count_);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void SetDenseInputPaddingAnchor(LocalTensor<int32_t> tensor,
                                                    uint32_t rows) {
    const uint32_t live = t_.h % 8;
    if (live == 0) return;
    const int32_t anchorByteOffset = static_cast<int32_t>(4 * zero_offset_);
    uint64_t mask[2] = {uint64_t{0xff} ^ ((uint64_t{1} << live) - 1), 0};
    for (uint32_t first = 0; first < rows; first += 248) {
      const uint32_t n = rows - first < 248 ? rows - first : 248;
      Duplicate(tensor[first * t_.h_pad + t_.h_pad - 8], anchorByteOffset, mask,
                static_cast<uint8_t>(n), uint16_t{1},
                static_cast<uint8_t>(t_.h_pad / 8));
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void BuildDenseInputMaps() {
    auto token = partial_.template ReinterpretCast<int32_t>();
    CreateVecIndex(input_index_, int32_t{0}, span_);
    PipeBarrier<PIPE_V>();
    ShiftRight(token, input_index_, int32_t{4}, span_);
    PipeBarrier<PIPE_V>();
    Muls(token, token, static_cast<int32_t>(t_.input_width - 16), span_);
    PipeBarrier<PIPE_V>();
    Add(input_index_, input_index_, token, span_);
    PipeBarrier<PIPE_V>();
    Muls(input_index_, input_index_, int32_t{4}, span_);
    PipeBarrier<PIPE_V>();
    for (uint32_t planes = 1; planes < t_.h + 2; planes *= 2) {
      const uint32_t n =
          t_.h + 2 - planes < planes ? t_.h + 2 - planes : planes;
      Adds(input_index_[planes * span_], input_index_,
           static_cast<int32_t>(4 * planes * t_.h), n * span_);
      PipeBarrier<PIPE_V>();
    }
    SetDenseInputPaddingAnchor(input_index_[2 * span_], rows_);
  }

  __aicore__ inline void CopyInAndBuildMaps(uint64_t first) {
    const DataCopyExtParams packed{1, 4 * valid_ * t_.input_width, 0, 0, 0};
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(dense_output_, input_[first * t_.input_width], packed, noPad);

    if (!constants_ready_) {
      Duplicate(denom_, 0.0f, rp_ + 64);
      Duplicate(ones_, 1.0f, ones_cap_ + 64);
      Duplicate(dense_output_[zero_offset_], 0.0f, 8);
      PipeBarrier<PIPE_V>();
      constants_ready_ = true;
    }
    if (t_.h < 16) {
      Duplicate(matrix_[rows_ * t_.h_pad], 0.0f, (16 - t_.h) * span_ + 64);
      PipeBarrier<PIPE_V>();
    }
    if (input_indexed_valid_ != valid_) {
      BuildDenseInputMaps();
      input_indexed_valid_ = valid_;
    }

    Sync<HardEvent::MTE2_V>();
    auto index = input_index_.template ReinterpretCast<uint32_t>();
    Gather(gates_, dense_output_, index, uint32_t{0}, 2 * span_);
    Gather(matrix_, dense_output_, index[2 * span_], uint32_t{0}, t_.h * span_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GatesAndRowMax() {
    // Interleave Gate Muls(-1) with RowMax on matrix_!
    Muls(tmp_, gates_, -1.0f, 2 * span_);
    for (uint32_t first = 0; first < rows_; first += 255) {
      const int32_t count =
          static_cast<int32_t>(rows_ - first < 255 ? rows_ - first : 255);
      WholeReduceMax(denom_[first], matrix_[first * t_.h_pad],
                     static_cast<int32_t>(t_.h), count, 1, 1,
                     static_cast<int32_t>(t_.h_pad / 8),
                     ReduceOrder::ORDER_ONLY_VALUE);
    }
    PipeBarrier<PIPE_V>();

    RunNarrowShortBatchGates(
        reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
        span_, t_.eps);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
  }

  __aicore__ inline void CopyOutCombStrided(uint64_t first, bool hasMoreTiles) {
    Sync<HardEvent::V_MTE3>();
    const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                              4 * t_.h, stride, 0, 0};
    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t base = first + token;
      DataCopyPad(comb_[base * t_.h * t_.h], matrix_[token * t_.h_pad], m);
    }
    if (hasMoreTiles) {
      Sync<HardEvent::MTE3_MTE2>();
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_, gateBuffer_, denseOutputBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_, onesBuffer_, denomBuffer_,
      inputIndexBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, ones_, denom_,
      dense_output_;
  LocalTensor<int32_t> input_index_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t valid_, rows_, span_, rp_, ones_cap_, dense_gate_cap_,
      input_gate_cap_, zero_offset_, input_indexed_valid_;
  bool constants_ready_;
};

}  // namespace f05
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f05_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f05::NarrowP16ShortBatchKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_
