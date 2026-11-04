#ifndef SINKHORN_NORM_KERNELS_F06_NARROW_P16_PAIRED_ROWS_H_
#define SINKHORN_NORM_KERNELS_F06_NARROW_P16_PAIRED_ROWS_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f06 {

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
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first),
                                 reinterpret_cast<__ubuf__ uint32_t *>(a + first), repeats, 1, 1, 8, 8);
    if constexpr (Op == 4) vmul(dst + first, a + first, b + first, repeats, 1, 1, 1, 8, 8, 8);
    first += 64 * repeats;
  }
  if (first < count) {
    WideRoundMask((uint64_t{1} << (count - first)) - 1);
    if constexpr (Op == 0) vadd(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 1) vadds(dst + first, a + first, scalar, 1, 1, 1, 8, 8);
    if constexpr (Op == 2) vdiv(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first),
                                 reinterpret_cast<__ubuf__ uint32_t *>(a + first), 1, 1, 1, 8, 8);
    if constexpr (Op == 4) vmul(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    WideRoundMask(~uint64_t{0});
  }
}

static __aicore__ __attribute__((always_inline)) inline void WideRoundFullRepeatsAdd(
    __ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b,
    uint32_t count) {
  uint32_t repeatsLeft = (count + 63) / 64;
  uint32_t offset = 0;
  while (repeatsLeft > 0) {
    const uint8_t repeats = repeatsLeft < 255 ? repeatsLeft : 255;
    vadd(dst + offset, a + offset, b + offset, repeats, 1, 1, 1, 8, 8, 8);
    offset += 64 * repeats;
    repeatsLeft -= repeats;
  }
}

static __aicore__ __attribute__((always_inline)) inline void RunNarrowPairedGates(
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

static __aicore__ __attribute__((always_inline)) inline void RunNarrowPairedCompleteRounds(
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
    // Every RAW dependency is protected by an explicit pipe_barrier(PIPE_V).
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
      }

      if (rows >= 64) {
        const uint32_t pairs = h / 2;
        WideRoundFullRepeatsAdd(partial, matrix, matrix + pairs * span,
                                pairs * span);
        pipe_barrier(PIPE_V);
        if (h == 9) {
          // Disjoint-destination tree for H=9 (JP7): writes Plane 0+3 to denom
          // and Plane 1+Row 8 to partial+4*span, eliminating all 60 tail vadds
          // and 120 SetVectorMask switches while keeping 100% bit-for-bit order!
          vadd(denom, partial, partial + 3 * span, spanRepeats, 1, 1, 1, 8, 8,
               8);
          vadd(partial + 4 * span, partial + span, matrix + 8 * span,
               spanRepeats, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          vadd(denom, denom, partial + 2 * span, spanRepeats, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          vadd(denom, denom, partial + 4 * span, spanRepeats, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
        } else {
          uint32_t planes = pairs;
          if (h % 2) {
            ++planes;
            const uint32_t half = planes / 2, upper = planes - half;
            WideRoundSpan<0>(partial, partial, partial + upper * span,
                             (half - 1) * span);
            WideRoundSpan<0>(partial + (half - 1) * span,
                             partial + (half - 1) * span,
                             matrix + (h - 1) * span, span);
            pipe_barrier(PIPE_V);
            planes = upper;
          }
          while (planes > 2) {
            const uint32_t half = planes / 2, upper = planes - half;
            if (half == upper) {
              WideRoundFullRepeatsAdd(partial, partial, partial + upper * span,
                                      half * span);
            } else {
              WideRoundSpan<0>(partial, partial, partial + upper * span,
                               half * span);
            }
            pipe_barrier(PIPE_V);
            planes = upper;
          }
          vadd(denom, partial, partial + span, spanRepeats, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
        }
      } else {
        for (uint32_t group = 0; group < 4; ++group) {
          WideRoundSpan<3>(partial + group * span, matrix + group * span,
                           matrix, span);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t group = 0; group < 4; ++group) {
          const uint32_t repeats = (h - 1 - group) / 4;
          for (uint32_t col = 0; col < span; col += 64) {
            const uint32_t width = span - col < 64 ? span - col : 64;
            if (width < 64) WideRoundMask((uint64_t{1} << width) - 1);
            auto acc = partial + group * span + col;
            vadd(acc, matrix + (group + 4) * span + col, acc, repeats, 1, 1, 1,
                 0, 4 * span / 8, 0);
            if (width < 64) WideRoundMask(~uint64_t{0});
          }
        }
        pipe_barrier(PIPE_V);
        WideRoundFullRepeatsAdd(partial, partial, partial + 2 * span, 2 * span);
        pipe_barrier(PIPE_V);
        vadd(denom, partial, partial + span, spanRepeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      }

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

class NarrowP16PairedRowsKernel {
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

    const uint32_t baseBytes =
        (t_.matrix_bytes + 256) + (2 * t_.gate_bytes + 256) +
        (t_.input_bytes + (tmp + 2 * rp_) * sizeof(float) + 256) +
        (ones_cap_ * sizeof(float) + 256) + (span * sizeof(float) + 256);
    const uint32_t denseBytes =
        (4 * zero_offset_ + 32) + 4 * (dense_gate_cap_ + matrixCap) +
        4 * (input_gate_cap_ + inputMatrixCap);
    use_dense_io_ = (baseBytes + denseBytes <= 188416);

    // Add +256B tail padding on UB buffers so full-repeat vector instructions
    // never need tail mask splits on row/span boundaries.
    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes + 256);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes + 256);
    pipe_.InitBuffer(partialBuffer_,
                     t_.input_bytes + (tmp + 2 * rp_) * sizeof(float) + 256);
    pipe_.InitBuffer(onesBuffer_, ones_cap_ * sizeof(float) + 256);
    pipe_.InitBuffer(denomBuffer_, span * sizeof(float) + 256);

    if (use_dense_io_) {
      // Reserve 32B (8 floats) at zero_offset_ permanently holding 0.0f so Gather
      // automatically zero-pads lanes h..15 of matrix_ and gates_!
      pipe_.InitBuffer(denseOutputBuffer_, 4 * zero_offset_ + 32);
      pipe_.InitBuffer(outputIndexBuffer_, 4 * (dense_gate_cap_ + matrixCap));
      pipe_.InitBuffer(inputIndexBuffer_, 4 * (input_gate_cap_ + inputMatrixCap));

      dense_output_ = denseOutputBuffer_.Get<float>();
      output_index_ = outputIndexBuffer_.Get<int32_t>();
      input_index_ = inputIndexBuffer_.Get<int32_t>();
    }
    indexed_valid_ = 0;
    input_indexed_valid_ = 0;
    gate_output_map_ready_ = false;

    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>();
    tmp_ = partial_[t_.input_bytes / sizeof(float)];
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();
    constants_ready_ = false;
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t first = start_ + offset;
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left) : t_.tile_tokens;
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;

      if (use_dense_io_) {
        CopyInAndBuildMaps(first);
        GatesAndRowMax();
      } else {
        CopyInAndGatesStrided(first);
        RowMaxOnly();
      }

      SetMaskNorm();
      RunNarrowPairedCompleteRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          t_.h, valid_, t_.eps, t_.iters);
      ResetMask();

      if (use_dense_io_) {
        // Queue comb Gather onto PIPE_V first, then issue contiguous pre/post MTE3 DMA
        // writebacks while PIPE_V executes the 20 Sinkhorn rounds and comb Gather!
        const uint32_t gates = t_.h * valid_;
        const uint32_t matrix = t_.h * gates;
        auto gateIndex = output_index_.template ReinterpretCast<uint32_t>();
        Gather(dense_output_[2 * dense_gate_cap_], matrix_,
               gateIndex[dense_gate_cap_], uint32_t{0}, matrix);

        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        const DataCopyExtParams gOut{1, 4 * gates, 0, 0, 0};
        DataCopyPad(pre_[first * t_.h], dense_output_, gOut);
        DataCopyPad(post_[first * t_.h], dense_output_[dense_gate_cap_], gOut);

        CopyOutComb(first, matrix, offset + t_.tile_tokens < count_);
      } else {
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        const DataCopyExtParams gOut{1, 4 * t_.h, 0, 0, 0};
        for (uint32_t token = 0; token < valid_; ++token) {
          const uint64_t base = first + token;
          DataCopyPad(pre_[base * t_.h], gates_[token * t_.h_pad], gOut);
          DataCopyPad(post_[base * t_.h], gates_[span_ + token * t_.h_pad], gOut);
        }
        CopyOutCombStrided(first, offset + t_.tile_tokens < count_);
      }
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void SetDenseInputPaddingAnchor(LocalTensor<int32_t> tensor,
                                                    uint32_t count) {
    const int32_t anchorByteOffset = static_cast<int32_t>(4 * zero_offset_);
    uint64_t mask[2] = {
        (uint64_t{0xffff} ^ ((uint64_t{1} << t_.h) - 1)) *
            uint64_t{0x0001000100010001},
        0};
    const uint32_t full = count / 64;
    if (full != 0) {
      Duplicate(tensor, anchorByteOffset, mask, static_cast<uint8_t>(full), 1, 8);
    }
    const uint32_t tail = count % 64;
    if (tail != 0) {
      mask[0] &= (uint64_t{1} << tail) - 1;
      Duplicate(tensor[full * 64], anchorByteOffset, mask, uint8_t{1}, 1, 8);
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
    SetDenseInputPaddingAnchor(input_index_[2 * span_], t_.h * span_);
  }

  __aicore__ inline void CopyInAndBuildMaps(uint64_t first) {
    // Issue single contiguous bulk MTE2 DMA read first so constants and index maps build on PIPE_V in parallel!
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
    if (input_indexed_valid_ != valid_) {
      BuildDenseInputMaps();
      input_indexed_valid_ = valid_;
    }
    if (!gate_output_map_ready_) {
      BuildDenseOutputMap<false>(output_index_, t_.h * t_.tile_tokens);
      gate_output_map_ready_ = true;
    }
    if (indexed_valid_ != valid_) {
      auto matrixIndex = output_index_[dense_gate_cap_];
      uint32_t group = 1;
      while ((group * t_.h * t_.h) % 8 != 0) group *= 2;
      if (group > valid_) group = valid_;
      BuildDenseOutputMap<true>(matrixIndex, group * t_.h * t_.h);
      for (uint32_t tokens = group; tokens < valid_; tokens *= 2) {
        const uint32_t n = valid_ - tokens < tokens ? valid_ - tokens : tokens;
        Adds(matrixIndex[tokens * t_.h * t_.h], matrixIndex,
             static_cast<int32_t>(4 * tokens * t_.h_pad), n * t_.h * t_.h);
        PipeBarrier<PIPE_V>();
      }
      indexed_valid_ = valid_;
    }
    Sync<HardEvent::MTE2_V>();
    auto index = input_index_.template ReinterpretCast<uint32_t>();
    Gather(gates_, dense_output_, index, uint32_t{0}, 2 * span_);
    Gather(matrix_, dense_output_, index[2 * span_], uint32_t{0}, t_.h * span_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GatesAndRowMax() {
    const uint32_t gatesCount = t_.h * valid_;

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

    if ASCEND_IS_AIV {
      const uint32_t n = 2 * span_;
      const uint32_t gateRepeats = (n + 63) / 64;
      const uint32_t spanRepeats = (span_ + 63) / 64;
      const uint32_t spanFull = span_ / 64;
      const uint32_t spanTail = span_ % 64;
      __ubuf__ float *gates = reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr());
      __ubuf__ float *tmp = reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr());
      __ubuf__ float *ones = reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr());

      SetMaskNorm();
      WideRoundMask(~uint64_t{0});
      vmins(tmp, tmp, 80.0f, gateRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vexp(tmp, tmp, gateRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadds(tmp, tmp, 1.0f, gateRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(gates, ones, tmp, gateRepeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      if (spanFull != 0) {
        vadds(gates, gates, t_.eps, spanFull, 1, 1, 8, 8);
      }
      if (spanTail != 0) {
        WideRoundMask((uint64_t{1} << spanTail) - 1);
        vadds(gates + spanFull * 64, gates + spanFull * 64, t_.eps, 1, 1, 1, 8, 8);
        WideRoundMask(~uint64_t{0});
      }
      vmuls(gates + span_, gates + span_, 2.0f, spanRepeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      ResetMask();
    }

    auto gateIndex = output_index_.template ReinterpretCast<uint32_t>();
    Gather(dense_output_, gates_, gateIndex, uint32_t{0}, gatesCount);
    Gather(dense_output_[dense_gate_cap_], gates_[span_], gateIndex, uint32_t{0},
           gatesCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
  }

  template <bool Matrix>
  __aicore__ inline void BuildDenseOutputMap(LocalTensor<int32_t> index,
                                             uint32_t count) {
    auto row = partial_.template ReinterpretCast<int32_t>();
    auto token = tmp_.template ReinterpretCast<int32_t>();
    const int32_t reciprocal =
        static_cast<int32_t>(((1U << 20) + t_.h - 1) / t_.h);
    const uint32_t chunk = t_.input_bytes / sizeof(int32_t);
    for (uint32_t first = 0; first < count; first += chunk) {
      const uint32_t length = count - first < chunk ? count - first : chunk;
      auto dst = index[first];
      CreateVecIndex(dst, static_cast<int32_t>(first), length);
      PipeBarrier<PIPE_V>();
      Muls(row, dst, reciprocal, length);
      PipeBarrier<PIPE_V>();
      ShiftRight(row, row, int32_t{20}, length);
      PipeBarrier<PIPE_V>();
      if constexpr (Matrix) {
        Muls(token, row, reciprocal, length);
        PipeBarrier<PIPE_V>();
        ShiftRight(token, token, int32_t{20}, length);
        Muls(row, row, static_cast<int32_t>(span_ - t_.h), length);
        PipeBarrier<PIPE_V>();
        Muls(token, token,
             static_cast<int32_t>(t_.h_pad) -
                 static_cast<int32_t>(t_.h * span_),
             length);
        Add(dst, dst, row, length);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, token, length);
      } else {
        Muls(row, row, static_cast<int32_t>(t_.h_pad - t_.h), length);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, row, length);
      }
      PipeBarrier<PIPE_V>();
      Muls(dst, dst, int32_t{4}, length);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyOutComb(uint64_t first, uint32_t matrix, bool hasMoreTiles) {
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams m{1, 4 * matrix, 0, 0, 0};
    DataCopyPad(comb_[first * t_.h * t_.h], dense_output_[2 * dense_gate_cap_], m);
    if (hasMoreTiles) {
      Sync<HardEvent::MTE3_MTE2>();
    }
  }

  __aicore__ inline void CopyInAndGatesStrided(uint64_t first) {
    const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
    const DataCopyPadExtParams<float> pad{
        true, 0, static_cast<uint8_t>(t_.h_pad - t_.h), 0.0f};
    const DataCopyExtParams g{2, 4 * t_.h, 0, stride, 0};
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                              4 * t_.h, 0, stride, 0};

    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t base = (first + token) * t_.input_width;
      DataCopyPad(gates_[token * t_.h_pad], input_[base], g, pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t base = (first + token) * t_.input_width;
      DataCopyPad(matrix_[token * t_.h_pad], input_[base + 2 * t_.h], m, pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    if (!constants_ready_) {
      Duplicate(denom_, 0.0f, rp_ + 64);
      Duplicate(ones_, 1.0f, ones_cap_ + 64);
      constants_ready_ = true;
    }

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    RunNarrowPairedGates(
        reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
        span_, t_.eps);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
  }

  __aicore__ inline void RowMaxOnly() {
    for (uint32_t first = 0; first < rows_; first += 255) {
      const int32_t count =
          static_cast<int32_t>(rows_ - first < 255 ? rows_ - first : 255);
      WholeReduceMax(denom_[first], matrix_[first * t_.h_pad],
                     static_cast<int32_t>(t_.h), count, 1, 1,
                     static_cast<int32_t>(t_.h_pad / 8),
                     ReduceOrder::ORDER_ONLY_VALUE);
    }
    PipeBarrier<PIPE_V>();
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
      outputIndexBuffer_, inputIndexBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, ones_, denom_,
      dense_output_;
  LocalTensor<int32_t> output_index_, input_index_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t valid_, rows_, span_, rp_, ones_cap_, dense_gate_cap_,
      input_gate_cap_, zero_offset_, indexed_valid_, input_indexed_valid_;
  bool constants_ready_, gate_output_map_ready_, use_dense_io_;
};

}  // namespace f06
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f06_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f06::NarrowP16PairedRowsKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F06_NARROW_P16_PAIRED_ROWS_H_
