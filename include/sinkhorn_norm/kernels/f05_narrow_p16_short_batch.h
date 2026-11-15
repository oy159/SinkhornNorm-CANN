#ifndef SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_
#define SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/common/normalization_forecast.h"
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

static __aicore__ __attribute__((always_inline)) inline void CaptureShortDenominator(
    __ubuf__ float *snapshot, __ubuf__ float *sum, __ubuf__ float *ones,
    uint32_t count, uint32_t capacity) {
  if ASCEND_IS_AIV {
    SetMaskCount();
    if (count < capacity) {
      SetVectorMask<float, MaskMode::COUNTER>(0, capacity);
      vector_dup(snapshot, 0.0f, (capacity + 63) / 64, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, count);
    vsub(snapshot, sum, ones, (count + 63) / 64, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    WideRoundMask(~uint64_t{0});
  }
}

// Keep exact updates for tiles whose measured row residuals remain large.
static __aicore__ __attribute__((always_inline)) inline bool ShortForecastAdmitted(
    __ubuf__ float *recent, __ubuf__ float *scratch, uint32_t span) {
  if ASCEND_IS_AIV {
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vabs(scratch, recent, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcgmax(scratch + span, scratch, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    const uint32_t groups = span / 8;
    WideRoundMask((uint64_t{1} << groups) - 1);
    vcgmax(scratch, scratch + span, 1, 1, 1, 8);
    WideRoundMask(~uint64_t{0});
    const auto event = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    constexpr float bound = 0x1p-10f;
    return scratch[0] <= bound && (groups <= 8 || scratch[1] <= bound);
  }
  return false;
}

template <bool Multiply>
static __aicore__ __attribute__((always_inline)) inline void ScaleShortColumns(
    __ubuf__ float *matrix, __ubuf__ float *factor, uint32_t h, uint32_t span) {
  if ASCEND_IS_AIV {
    const uint8_t stride = static_cast<uint8_t>(span / 8);
    for (uint32_t col = 0; col < span; col += 64) {
      if (span - col < 64) WideRoundMask((uint64_t{1} << (span - col)) - 1);
      if constexpr (Multiply)
        vmul(matrix + col, matrix + col, factor + col, h, 1, 1, 1, stride, stride, 0);
      else
        vdiv(matrix + col, matrix + col, factor + col, h, 1, 1, 1, stride, stride, 0);
    }
    pipe_barrier(PIPE_V);
  }
}

static __aicore__ __attribute__((always_inline)) inline void RunNarrowShortBatchCompleteRounds(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *ones,
    __ubuf__ float *denom, uint32_t h, uint32_t valid, float eps,
    uint32_t iters, __ubuf__ float *history, uint32_t historyCapacity,
    uint32_t partialCapacity) {
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
    const uint64_t pairMask =
        ((uint64_t{1} << h) - 1) * uint64_t{0x0001000100010001};
    constexpr uint32_t steps = 6, tail = 8;
    const bool forecast = iters >= 4 + steps + tail && eps >= 0x1p-24f &&
        eps <= 1.0e-5f && historyCapacity >= 7 * span && partialCapacity >= 10 * span;
    const uint32_t checkpoint = forecast ? iters - steps - tail : iters;

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
      const bool predict = round == checkpoint &&
          ShortForecastAdmitted(history + ((round - 1) % 2) * 2 * span, partial, span);
      const bool measure = forecast && round < checkpoint && round + 2 >= checkpoint;
      auto snapshot = history + (round % 2) * 2 * span;
      auto columns = history + 6 * span;
      if (predict) {
        ForecastDenominatorProduct<steps>(denom, columns, partial, history + 4 * span,
            ones, history + ((round - 1) % 2) * 2 * span, snapshot, span, eps);
      } else {
        WideRoundMask(pairMask);
        vcgadd(partial, matrix, groups4, 1, 1, 8);
        pipe_barrier(PIPE_V);
        WideRoundMask(~uint64_t{0});
        vcpadd(denom, partial, pairs32, 1, 1, 8);
        pipe_barrier(PIPE_V);
        if (measure) CaptureShortDenominator(snapshot, denom, ones, rows, span);
        if (round > 0) {
          vadds(denom, denom, eps, rowRepeats, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
        }
      }

      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial),
            reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(partial + 8),
            reinterpret_cast<__ubuf__ uint32_t *>(denom), 2, 16, broadcasts);
      pipe_barrier(PIPE_V);
      if (predict)
        vmul(matrix, matrix, partial, matRepeats, 1, 1, 1, 8, 8, 8);
      else
        vdiv(matrix, matrix, partial, matRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      if (predict) {
        ScaleShortColumns<true>(matrix, columns, h, span);
        round += steps - 1;
        continue;
      }
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
      if (measure) CaptureShortDenominator(snapshot + span, denom, ones, span, span);
      vadds(denom, denom, eps, spanRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);

      ScaleShortColumns<false>(matrix, denom, h, span);
    }
  }
}

class NarrowP16ShortBatchKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const sinkhorn_norm::ShortNarrowArgs &t) {
    t_.eps = t.eps;
    t_.iters = t.iters;
    t_.h = t.h;
    t_.h_pad = 16;
    t_.core_count = GetBlockNum();
    t_.tokens_per_core = t.owner;
    t_.extra_core_count = t.extra;
    t_.token_count = uint64_t{t.owner} * t_.core_count + t.extra;
    t_.tile_tokens = t.owner + (t.extra != 0);
    t_.input_width = t.h * (t.h + 2);
    t_.input_bytes = 32 * t.h * t_.tile_tokens;
    t_.matrix_bytes = 64 * t.h * t_.tile_tokens;
    t_.gate_bytes = 64 * t_.tile_tokens;
    t_.scratch_bytes = 0;  // Init computes the actual scratch extents below.

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
          t_.h, valid_, t_.eps, t_.iters,
          reinterpret_cast<__ubuf__ float *>(input_index_.GetPhyAddr()),
          input_index_.GetSize(), partial_.GetSize());
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
    // Reduce eight independent 16-wide rows per repeat, rather than issuing
    // a separate whole reduction for every row. Padding must not win the max.
    if ASCEND_IS_AIV {
      auto *matrix = reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr());
      auto *partial = reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr());
      auto *denom = reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr());
      const uint64_t live = ((uint64_t{1} << t_.h) - 1) *
                            uint64_t{0x0001000100010001};
      SetMaskNorm();
      if (t_.h < 16) {
        WideRoundMask(~live);
        vector_dup(matrix, -3.402823466e+38f, (rows_ * 16 + 63) / 64, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
      }
      WideRoundMask(~uint64_t{0});
      vmax(partial, matrix, matrix + 8, (rows_ + 7) / 8,
           1, 2, 2, 8, 16, 16);
      pipe_barrier(PIPE_V);
      vcgmax(denom, partial, (rows_ + 7) / 8, 1, 1, 8);
      ResetMask();
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
    const uint32_t n = valid_ * t_.h * t_.h;
    auto row = partial_.ReinterpretCast<int32_t>();
    auto token = dense_output_.ReinterpretCast<int32_t>();
    const int32_t reciprocal = static_cast<int32_t>(((1u << 20) + t_.h - 1) / t_.h);
    // For the bounded F05 index domain this integer reciprocal gives exact
    // floor division. These are layout addresses, never computed tensor values.
    CreateVecIndex(input_index_, int32_t{0}, n);
    PipeBarrier<PIPE_V>();
    Muls(row, input_index_, reciprocal, n);
    PipeBarrier<PIPE_V>();
    ShiftRight(row, row, int32_t{20}, n);
    PipeBarrier<PIPE_V>();
    Muls(token, row, reciprocal, n);
    PipeBarrier<PIPE_V>();
    ShiftRight(token, token, int32_t{20}, n);
    PipeBarrier<PIPE_V>();
    Muls(row, row, static_cast<int32_t>(span_ - t_.h), n);
    Muls(token, token, int32_t{16} - static_cast<int32_t>(t_.h * span_), n);
    PipeBarrier<PIPE_V>();
    Add(input_index_, input_index_, row, n);
    PipeBarrier<PIPE_V>();
    Add(input_index_, input_index_, token, n);
    PipeBarrier<PIPE_V>();
    Muls(input_index_, input_index_, int32_t{4}, n);
    PipeBarrier<PIPE_V>();
    Gather(dense_output_, matrix_, input_index_.ReinterpretCast<uint32_t>(), uint32_t{0}, n);
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams out{1, n * uint32_t{sizeof(float)}, 0, 0, 0};
    DataCopyPad(comb_[first * t_.h * t_.h], dense_output_, out);
    input_indexed_valid_ = 0;
    if (hasMoreTiles) Sync<HardEvent::MTE3_MTE2>();
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
    sinkhorn_norm::ShortNarrowArgs tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f05::NarrowP16ShortBatchKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F05_NARROW_P16_SHORT_BATCH_H_
