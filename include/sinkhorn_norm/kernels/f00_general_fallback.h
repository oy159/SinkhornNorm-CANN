#ifndef SINKHORN_NORM_KERNELS_F00_GENERAL_FALLBACK_H_
#define SINKHORN_NORM_KERNELS_F00_GENERAL_FALLBACK_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"
#include "sinkhorn_norm/tiling_key.h"

namespace sinkhorn_f00 {

using namespace AscendC;
using sinkhorn_norm::TilingData;
using sinkhorn_norm::primitives::ComputeGatePairStable;
using sinkhorn_norm::primitives::ComputeGateStable;
using sinkhorn_norm::primitives::SyncEvent;

template <bool MergeGates, bool TreeColumns, bool ReuseInverse>
class GeneralFallbackKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const TilingData &tiling) {
    t_ = tiling;
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    preGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    combGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    // Each token is (H + 2) rows: pre, post, then H matrix rows.
    pipe_.InitBuffer(tileBuffer_, t_.matrix_bytes + 2 * t_.gate_bytes);
    pipe_.InitBuffer(tmpBuffer_, t_.scratch_bytes / 12 * 8);
    pipe_.InitBuffer(workBuffer_, t_.scratch_bytes / 12);
    pipe_.InitBuffer(denomBuffer_, t_.scratch_bytes / 12);
    pipe_.InitBuffer(resultBuffer_, t_.scratch_bytes / 12);
    matrix_ = tileBuffer_.Get<float>();
    tmp_ = tmpBuffer_.Get<float>();
    work_ = workBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();
    result_ = resultBuffer_.Get<float>();

    if constexpr (TreeColumns) {
      pipe_.InitBuffer(columnBuffer_,
                       ((t_.h + 1) / 2) * t_.h_pad * sizeof(float));
      columns_ = columnBuffer_.Get<float>();
    }
    // Brcb reads complete groups of eight sums. Initialize padded lanes once.
    Duplicate(result_, 0.0f, t_.h_pad);
    if constexpr (ReuseInverse) {
      pipe_.InitBuffer(onesBuffer_, t_.scratch_bytes / 12);
      ones_ = onesBuffer_.Get<float>();
      Duplicate(ones_, 1.0f, t_.h_pad);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t remaining = count_ - offset;
      const uint32_t valid = remaining < t_.tile_tokens
                                 ? static_cast<uint32_t>(remaining)
                                 : t_.tile_tokens;
      CopyIn(start_ + offset, valid);
      for (uint32_t token = 0; token < valid; ++token) {
        const uint32_t tokenOffset = token * (t_.h + 2) * t_.h_pad;
        if constexpr (MergeGates) {
          ComputeGatePairStable(matrix_[tokenOffset], tmp_, t_.h, t_.h_pad,
                                t_.eps);
        } else {
          ComputeGateStable(matrix_[tokenOffset], tmp_, work_, denom_, t_.h,
                            t_.eps, true);
          ComputeGateStable(matrix_[tokenOffset + t_.h_pad], tmp_, work_,
                            denom_, t_.h, t_.eps, false);
        }
        const uint32_t matrixOffset = tokenOffset + 2 * t_.h_pad;
        Softmax(matrixOffset);
        NormalizeColumns(matrixOffset);
        for (uint32_t iter = 1; iter < t_.iters; ++iter) {
          NormalizeRows(matrixOffset);
          NormalizeColumns(matrixOffset);
        }
      }
      CopyOut(start_ + offset, valid);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint64_t first, uint32_t valid) {
    const uint32_t rows = valid * (t_.h + 2);
    const DataCopyPadExtParams<float> pad{
        true, 0, static_cast<uint8_t>(t_.h_pad - t_.h), 0.0f};
    for (uint32_t copied = 0; copied < rows;) {
      const uint32_t left = rows - copied;
      const uint32_t chunk = left < 4095 ? left : 4095;
      const DataCopyExtParams copy{static_cast<uint16_t>(chunk),
                                   t_.h * uint32_t{sizeof(float)}, 0, 0, 0};
      DataCopyPad(matrix_[copied * t_.h_pad],
                  inputGm_[first * t_.input_width + uint64_t{copied} * t_.h],
                  copy, pad);
      copied += chunk;
    }
    SyncEvent<HardEvent::MTE2_V>(pipe_);
  }

  template <bool maximum>
  __aicore__ inline void ReduceRows(uint32_t base) {
    const uint64_t width = t_.h < 64 ? t_.h : 64;
    const auto rows = static_cast<uint8_t>(t_.h);
    const auto stride = static_cast<uint8_t>(t_.h_pad / 8);
    if constexpr (maximum) {
      WholeReduceMax(result_, matrix_[base], width, rows, 1, 1, stride,
                     ReduceOrder::ORDER_ONLY_VALUE);
    } else {
      WholeReduceSum(result_, matrix_[base], width, rows, 1, 1, stride);
    }
    if (t_.h > 64) {
      if constexpr (maximum) {
        WholeReduceMax(work_, matrix_[base + 64], uint64_t{t_.h - 64}, rows, 1,
                       1, stride, ReduceOrder::ORDER_ONLY_VALUE);
      } else {
        WholeReduceSum(work_, matrix_[base + 64], uint64_t{t_.h - 64}, rows, 1,
                       1, stride);
      }
      PipeBarrier<PIPE_V>();
      if constexpr (maximum) {
        Max(result_, result_, work_, t_.h);
      } else {
        Add(result_, result_, work_, t_.h);
      }
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void BroadcastRows() {
    Brcb(tmp_, result_, static_cast<uint8_t>(t_.h_pad / 8), {1, 8});
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void NormalizeFromRowSums(uint32_t base) {
    if constexpr (ReuseInverse) {
      Div(result_, ones_, result_, t_.h);
      PipeBarrier<PIPE_V>();
    }
    BroadcastRows();
    const auto stride = static_cast<uint8_t>(t_.h_pad / 8);
    for (uint32_t col = 0; col < t_.h; col += 64) {
      const uint64_t width = t_.h - col < 64 ? t_.h - col : 64;
      if constexpr (ReuseInverse) {
        Mul(matrix_[base + col], matrix_[base + col], tmp_, width,
            static_cast<uint8_t>(t_.h), {1, 1, 0, stride, stride, 1});
      } else {
        Div(matrix_[base + col], matrix_[base + col], tmp_, width,
            static_cast<uint8_t>(t_.h), {1, 1, 0, stride, stride, 1});
      }
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Softmax(uint32_t base) {
    ReduceRows<true>(base);
    BroadcastRows();
    const auto stride = static_cast<uint8_t>(t_.h_pad / 8);
    for (uint32_t col = 0; col < t_.h; col += 64) {
      const uint64_t width = t_.h - col < 64 ? t_.h - col : 64;
      Sub(matrix_[base + col], matrix_[base + col], tmp_, width,
          static_cast<uint8_t>(t_.h), {1, 1, 0, stride, stride, 1});
    }
    PipeBarrier<PIPE_V>();
    Exp(matrix_[base], matrix_[base], t_.h * t_.h_pad);
    PipeBarrier<PIPE_V>();
    ReduceRows<false>(base);
    NormalizeFromRowSums(base);
    Adds(matrix_[base], matrix_[base], t_.eps, t_.h * t_.h_pad);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void NormalizeRows(uint32_t base) {
    ReduceRows<false>(base);
    Adds(result_, result_, t_.eps, t_.h);
    PipeBarrier<PIPE_V>();
    NormalizeFromRowSums(base);
  }

  __aicore__ inline void NormalizeColumns(uint32_t base) {
    const auto stride = static_cast<uint8_t>(t_.h_pad / 8);
    if constexpr (TreeColumns) {
      const uint32_t pairs = t_.h / 2;
      uint32_t rows = t_.h - pairs;
      Add(columns_, matrix_[base], matrix_[base + rows * t_.h_pad],
          pairs * t_.h_pad);
      if (t_.h & 1) {
        Adds(columns_[pairs * t_.h_pad], matrix_[base + pairs * t_.h_pad], 0.0f,
             t_.h_pad);
      }
      PipeBarrier<PIPE_V>();
      while (rows > 1) {
        const uint32_t count = rows / 2;
        const uint32_t upper = rows - count;
        Add(columns_, columns_, columns_[upper * t_.h_pad], count * t_.h_pad);
        PipeBarrier<PIPE_V>();
        rows = upper;
      }
      Adds(denom_, columns_, t_.eps, t_.h);
    } else {
      Duplicate(tmp_, 0.0f, t_.h);
      PipeBarrier<PIPE_V>();
      for (uint32_t col = 0; col < t_.h; col += 64) {
        const uint64_t width = t_.h - col < 64 ? t_.h - col : 64;
        Add(tmp_[col], matrix_[base + col], tmp_[col], width,
            static_cast<uint8_t>(t_.h), {1, 1, 1, 0, stride, 0});
      }
      PipeBarrier<PIPE_V>();
      Adds(denom_, tmp_, t_.eps, t_.h);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (ReuseInverse) {
      Div(denom_, ones_, denom_, t_.h);
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t col = 0; col < t_.h; col += 64) {
      const uint64_t width = t_.h - col < 64 ? t_.h - col : 64;
      if constexpr (ReuseInverse) {
        Mul(matrix_[base + col], matrix_[base + col], denom_[col], width,
            static_cast<uint8_t>(t_.h), {1, 1, 1, stride, stride, 0});
      } else {
        Div(matrix_[base + col], matrix_[base + col], denom_[col], width,
            static_cast<uint8_t>(t_.h), {1, 1, 1, stride, stride, 0});
      }
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyOut(uint64_t first, uint32_t valid) {
    SyncEvent<HardEvent::V_MTE3>(pipe_);
    for (uint32_t token = 0; token < valid; ++token) {
      const uint64_t n = first + token;
      const DataCopyExtParams gateCopy{1, t_.h * uint32_t{sizeof(float)}, 0, 0,
                                       0};
      const uint32_t tokenOffset = token * (t_.h + 2) * t_.h_pad;
      DataCopyPad(preGm_[n * t_.h], matrix_[tokenOffset], gateCopy);
      DataCopyPad(postGm_[n * t_.h], matrix_[tokenOffset + t_.h_pad], gateCopy);
      const DataCopyExtParams matrixCopy{
          static_cast<uint16_t>(t_.h), t_.h * uint32_t{sizeof(float)}, 0, 0, 0};
      DataCopyPad(combGm_[n * t_.h * t_.h],
                  matrix_[tokenOffset + 2 * t_.h_pad], matrixCopy);
    }
    SyncEvent<HardEvent::MTE3_MTE2>(pipe_);
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> tileBuffer_;
  TBuf<TPosition::VECCALC> tmpBuffer_, workBuffer_, denomBuffer_, resultBuffer_,
      onesBuffer_, columnBuffer_;
  GlobalTensor<float> inputGm_, preGm_, postGm_, combGm_;
  LocalTensor<float> matrix_, tmp_, work_, denom_, result_, ones_, columns_;
  TilingData t_;
  uint64_t start_, count_;
};

}  // namespace sinkhorn_f00

template <uint32_t Variant>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f00_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  constexpr bool MergeGates = (Variant & 1U) != 0;
  constexpr bool TreeColumns = (Variant & 2U) != 0;
  constexpr bool ReuseInverse = (Variant & 4U) != 0;
  sinkhorn_f00::GeneralFallbackKernel<MergeGates, TreeColumns, ReuseInverse> op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F00_GENERAL_FALLBACK_H_
