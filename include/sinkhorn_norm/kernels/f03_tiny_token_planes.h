#ifndef SINKHORN_NORM_KERNELS_F03_TINY_TOKEN_PLANES_H_
#define SINKHORN_NORM_KERNELS_F03_TINY_TOKEN_PLANES_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"
#include "sinkhorn_norm/kernels/f03_h2_blocked.h"

namespace sinkhorn_norm {
namespace kernels {
using namespace AscendC;

__aicore__ inline uint32_t GetCompactPlaneCapacity(const sinkhorn_norm::TilingData &t) {
  const uint64_t h = t.h;
  const uint64_t gather_limit = (255 / (h * h)) * 64;
  const uint64_t width = h * (h + 2);
  const uint64_t plane_capacity = t.input_bytes / (4 * width);
  const bool plane_budget =
      t.gate_bytes == 0 && plane_capacity % 64 == 0 &&
      uint64_t{t.input_bytes} == 4 * width * plane_capacity &&
      t.matrix_bytes == t.input_bytes &&
      (uint64_t{t.scratch_bytes} == 4 * (h * h + h + 3) * plane_capacity ||
       uint64_t{t.scratch_bytes} == 4 * (h * h + h + 4) * plane_capacity);
  if (plane_budget) {
    return static_cast<uint32_t>(
        plane_capacity < gather_limit ? plane_capacity : gather_limit);
  }
  const uint64_t bytes = uint64_t{t.input_bytes} + t.matrix_bytes +
                         2 * uint64_t{t.gate_bytes} + t.scratch_bytes;
  const uint64_t lanes = 3 * h * h + 5 * h + 4;
  const uint64_t capacity = bytes / (4 * lanes) / 64 * 64;
  const uint64_t owner =
      uint64_t{t.tokens_per_core} + (t.extra_core_count != 0);
  const uint64_t owner_limit = (owner + 63) / 64 * 64;
  const uint64_t limit =
      gather_limit < owner_limit ? gather_limit : owner_limit;
  return static_cast<uint32_t>(capacity < limit ? capacity : limit);
}

__aicore__ __attribute__((always_inline)) inline void RunCompactPlaneSoftmaxAndRoundsH2(
    __ubuf__ float *m, __ubuf__ float *d, uint32_t p, uint32_t valid,
    float eps, uint32_t iters) {
  if ASCEND_IS_AIV {
    // H2 dense staging has 8*p floats and is dead during these rounds.
    // The four temporary planes plus this 64-float skew fit even at p=64.
    // Offset denominator operands by half a 512-byte UB address phase.
    d += 64;
    pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(uint64_t{0}, uint64_t{valid});
    auto m00 = m + 0 * p;
    auto m01 = m + 1 * p;
    auto m10 = m + 2 * p;
    auto m11 = m + 3 * p;
    auto a0 = d + 0 * p;
    auto a1 = d + 1 * p;
    auto b0 = d + 2 * p;
    auto b1 = d + 3 * p;

    // --- Native Softmax H=2 (Interleaved Row 0 & 1) ---
    vmax(a0, m00, m01, 1, 1, 1, 1, 8, 8, 8);
    vmax(a1, m10, m11, 1, 1, 1, 1, 8, 8, 8);
    vsub(m00, m00, a0, 1, 1, 1, 1, 8, 8, 8);
    vsub(m01, m01, a0, 1, 1, 1, 1, 8, 8, 8);
    vsub(m10, m10, a1, 1, 1, 1, 1, 8, 8, 8);
    vsub(m11, m11, a1, 1, 1, 1, 1, 8, 8, 8);
    vexp(m00, m00, 1, 1, 1, 8, 8);
    vexp(m01, m01, 1, 1, 1, 8, 8);
    vexp(m10, m10, 1, 1, 1, 8, 8);
    vexp(m11, m11, 1, 1, 1, 8, 8);
    vadd(b0, m00, m01, 1, 1, 1, 1, 8, 8, 8);
    vadd(b1, m10, m11, 1, 1, 1, 1, 8, 8, 8);
    vdiv(m00, m00, b0, 1, 1, 1, 1, 8, 8, 8);
    vdiv(m01, m01, b0, 1, 1, 1, 1, 8, 8, 8);
    vdiv(m10, m10, b1, 1, 1, 1, 1, 8, 8, 8);
    vdiv(m11, m11, b1, 1, 1, 1, 1, 8, 8, 8);
    vadds(m00, m00, eps, 1, 1, 1, 8, 8);
    vadds(m01, m01, eps, 1, 1, 1, 8, 8);
    vadds(m10, m10, eps, 1, 1, 1, 8, 8);
    vadds(m11, m11, eps, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // --- Native Sinkhorn Rounds H=2 ---
    auto d0 = d + 0 * p;
    auto d1 = d + 1 * p;
    for (uint32_t iter = 0; iter < iters; ++iter) {
      if (iter != 0) {
        vadd(d0, m00, m01, 1, 1, 1, 1, 8, 8, 8);
        vadd(d1, m10, m11, 1, 1, 1, 1, 8, 8, 8);
        vadds(d0, d0, eps, 1, 1, 1, 8, 8);
        vadds(d1, d1, eps, 1, 1, 1, 8, 8);
        vdiv(m00, m00, d0, 1, 1, 1, 1, 8, 8, 8);
        vdiv(m01, m01, d0, 1, 1, 1, 1, 8, 8, 8);
        vdiv(m10, m10, d1, 1, 1, 1, 1, 8, 8, 8);
        vdiv(m11, m11, d1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      }

      vadds(d0, m00, eps, 1, 1, 1, 8, 8);
      vadds(d1, m01, eps, 1, 1, 1, 8, 8);
      vadd(d0, m10, d0, 1, 1, 1, 1, 8, 8, 8);
      vadd(d1, m11, d1, 1, 1, 1, 1, 8, 8, 8);
      vdiv(m00, m00, d0, 1, 1, 1, 1, 8, 8, 8);
      vdiv(m10, m10, d0, 1, 1, 1, 1, 8, 8, 8);
      vdiv(m01, m01, d1, 1, 1, 1, 1, 8, 8, 8);
      vdiv(m11, m11, d1, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    SetMaskNorm();
    ResetMask();
  }
}

__aicore__ __attribute__((always_inline)) inline void RunCompactPlaneRounds(
    __ubuf__ float *m, __ubuf__ float *d, uint32_t h, uint32_t p,
    uint32_t valid, float eps, uint32_t iters) {
  if ASCEND_IS_AIV {
    pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(uint64_t{0}, uint64_t{valid});
    for (uint32_t iter = 0; iter < iters; ++iter) {
      if (iter != 0) {
        for (uint32_t r = 0; r < h; ++r) {
          vadd(d + r * p, m + r * h * p, m + (r * h + 1) * p, 1, 1, 1, 1, 8, 8,
               8);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t c = 2; c < h; ++c) {
          for (uint32_t r = 0; r < h; ++r) {
            vadd(d + r * p, d + r * p, m + (r * h + c) * p, 1, 1, 1, 1, 8, 8,
                 8);
          }
          pipe_barrier(PIPE_V);
        }
        for (uint32_t r = 0; r < h; ++r) {
          vadds(d + r * p, d + r * p, eps, 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t r = 0; r < h; ++r) {
          for (uint32_t c = 0; c < h; ++c) {
            auto cell = m + (r * h + c) * p;
            vdiv(cell, cell, d + r * p, 1, 1, 1, 1, 8, 8, 8);
          }
        }
        pipe_barrier(PIPE_V);
      }

      for (uint32_t c = 0; c < h; ++c) {
        vadds(d + c * p, m + c * p, eps, 1, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      for (uint32_t r = 1; r < h; ++r) {
        for (uint32_t c = 0; c < h; ++c) {
          vadd(d + c * p, m + (r * h + c) * p, d + c * p, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
      }
      for (uint32_t r = 0; r < h; ++r) {
        for (uint32_t c = 0; c < h; ++c) {
          auto cell = m + (r * h + c) * p;
          vdiv(cell, cell, d + c * p, 1, 1, 1, 1, 8, 8, 8);
        }
      }
      pipe_barrier(PIPE_V);
    }
    SetMaskNorm();
    ResetMask();
  }
}

class F03_TinyTokenPlanes {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, sinkhorn_norm::TilingData t,
                              uint32_t capacity) {
    t_ = t;
    cap_ = capacity;
    h_ = t.h;
    width_ = h_ * (h_ + 2);
    const uint64_t block = GetBlockIdx();
    count_ = t.tokens_per_core + (block < t.extra_core_count ? 1 : 0);
    start_ = block * t.tokens_per_core +
             (block < t.extra_core_count ? block : t.extra_core_count);
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    pipe_.InitBuffer(denseBuffer_, 4 * width_ * cap_);
    pipe_.InitBuffer(planeBuffer_, 4 * width_ * cap_);
    pipe_.InitBuffer(indexBuffer_, 4 * (1 + h_ + h_ * h_) * cap_);
    pipe_.InitBuffer(scratchBuffer_, 4 * 2 * cap_);
    dense_ = denseBuffer_.Get<float>();
    planes_ = planeBuffer_.Get<float>();
    inIndex_ = indexBuffer_.Get<int32_t>();
    gateIndex_ = inIndex_[cap_];
    matrixIndex_ = gateIndex_[h_ * cap_];
    a_ = scratchBuffer_.Get<float>();
    b_ = a_[cap_];
    c_ = dense_[cap_];

    SeedOutputIndices(gateIndex_, h_);
    SeedOutputIndices(matrixIndex_, h_ * h_);
    Sync<HardEvent::S_V>();
    indices_ready_ = false;
  }

  __aicore__ inline void Process() {
    for (uint64_t offset = 0; offset < count_; offset += cap_) {
      const uint32_t valid = count_ - offset < cap_
                                 ? static_cast<uint32_t>(count_ - offset)
                                 : cap_;
      const uint64_t first = start_ + offset;
      const DataCopyExtParams copy{1, 4 * width_ * valid, 0, 0, 0};
      const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
      DataCopyPad(dense_, input_[first * width_], copy, padding);
      if (!indices_ready_) {
        ExpandOutputIndices(gateIndex_, h_);
        ExpandOutputIndices(matrixIndex_, h_ * h_);
        CreateVecIndex(inIndex_, int32_t{0}, cap_);
        PipeBarrier<PIPE_V>();
        Muls(inIndex_, inIndex_, static_cast<int32_t>(4 * width_), cap_);
        PipeBarrier<PIPE_V>();
        indices_ready_ = true;
      }
      Sync<HardEvent::MTE2_V>();
      for (uint32_t element = 0; element < width_; ++element) {
        Gather(planes_[element * cap_], dense_,
               inIndex_.ReinterpretCast<uint32_t>(), 4 * element, valid);
      }
      PipeBarrier<PIPE_V>();
      Gates(valid);
      if (h_ == 2) {
        RunCompactPlaneSoftmaxAndRoundsH2(
            reinterpret_cast<__ubuf__ float *>(planes_.GetPhyAddr()) +
                2 * h_ * cap_,
            reinterpret_cast<__ubuf__ float *>(dense_.GetPhyAddr()), cap_,
            valid, t_.eps, t_.iters);
      } else {
        Softmax(valid);
        RunCompactPlaneRounds(
            reinterpret_cast<__ubuf__ float *>(planes_.GetPhyAddr()) +
                2 * h_ * cap_,
            reinterpret_cast<__ubuf__ float *>(dense_.GetPhyAddr()), h_, cap_,
            valid, t_.eps, t_.iters);
      }

      Gather(dense_, planes_, gateIndex_.ReinterpretCast<uint32_t>(),
             uint32_t{0}, h_ * valid);
      Gather(dense_[h_ * cap_], planes_[h_ * cap_],
             gateIndex_.ReinterpretCast<uint32_t>(), uint32_t{0}, h_ * valid);
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

      Gather(dense_[2 * h_ * cap_], planes_[2 * h_ * cap_],
             matrixIndex_.ReinterpretCast<uint32_t>(), uint32_t{0},
             h_ * h_ * valid);

      WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
      const DataCopyExtParams gateCopy{1, 4 * h_ * valid, 0, 0, 0};
      DataCopyPad(pre_[first * h_], dense_, gateCopy);
      DataCopyPad(post_[first * h_], dense_[h_ * cap_], gateCopy);

      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams matrixCopy{1, 4 * h_ * h_ * valid, 0, 0, 0};
      DataCopyPad(comb_[first * h_ * h_], dense_[2 * h_ * cap_], matrixCopy);
      if (offset + cap_ < count_) {
        Sync<HardEvent::MTE3_MTE2>();
      }
    }
    PipeBarrier<PIPE_ALL>();
  }

 private:
  template <HardEvent Event>
  __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
  }

  __aicore__ inline void SeedOutputIndices(LocalTensor<int32_t> index,
                                           uint32_t elements) {
    for (uint32_t token = 0; token < 8; ++token) {
      for (uint32_t element = 0; element < elements; ++element) {
        index.SetValue(token * elements + element,
                       static_cast<int32_t>(4 * (element * cap_ + token)));
      }
    }
  }

  __aicore__ inline void ExpandOutputIndices(LocalTensor<int32_t> index,
                                             uint32_t elements) {
    for (uint32_t token = 8; token < cap_; token += 8) {
      Adds(index[token * elements], index, static_cast<int32_t>(4 * token),
           8 * elements);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline LocalTensor<float> Cell(uint32_t row, uint32_t col) {
    return planes_[(2 * h_ + row * h_ + col) * cap_];
  }

  __aicore__ inline void Gates(uint32_t valid) {
    const bool full = valid == cap_;
    const uint32_t n = full ? 2 * h_ * cap_ : valid;
    auto a = full ? dense_ : a_;
    auto b = full ? dense_[n] : b_;
    for (uint32_t g = 0; g < (full ? 1 : 2 * h_); ++g) {
      auto x = planes_[g * cap_];
      Duplicate(b, 1.0f, n);
      Muls(a, x, -1.0f, n);
      PipeBarrier<PIPE_V>();
      Mins(a, a, 80.0f, n);
      PipeBarrier<PIPE_V>();
      Exp(a, a, n);
      PipeBarrier<PIPE_V>();
      Adds(a, a, 1.0f, n);
      PipeBarrier<PIPE_V>();
      Div(x, b, a, n);
      PipeBarrier<PIPE_V>();
      if (full) {
        Adds(x, x, t_.eps, h_ * cap_);
        Muls(x[h_ * cap_], x[h_ * cap_], 2.0f, h_ * cap_);
      } else if (g < h_) {
        Adds(x, x, t_.eps, n);
      } else {
        Muls(x, x, 2.0f, n);
      }
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void Softmax(uint32_t valid) {
    for (uint32_t row = 0; row < h_; ++row) {
      Max(a_, Cell(row, 0), Cell(row, 1), valid);
      PipeBarrier<PIPE_V>();
      for (uint32_t col = 2; col < h_; ++col) {
        Max(a_, a_, Cell(row, col), valid);
        PipeBarrier<PIPE_V>();
      }
      for (uint32_t col = 0; col < h_; ++col) {
        Sub(Cell(row, col), Cell(row, col), a_, valid);
      }
      PipeBarrier<PIPE_V>();
      for (uint32_t col = 0; col < h_; ++col) {
        Exp(Cell(row, col), Cell(row, col), valid);
      }
      PipeBarrier<PIPE_V>();
      RowSum(row, valid);
      for (uint32_t col = 0; col < h_; ++col) {
        Div(Cell(row, col), Cell(row, col), b_, valid);
      }
      PipeBarrier<PIPE_V>();
      for (uint32_t col = 0; col < h_; ++col) {
        Adds(Cell(row, col), Cell(row, col), t_.eps, valid);
      }
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void RowSum(uint32_t row, uint32_t valid) {
    Add(b_, Cell(row, 0), Cell(row, 1), valid);
    PipeBarrier<PIPE_V>();
    for (uint32_t col = 2; col < h_; ++col) {
      Add(b_, b_, Cell(row, col), valid);
      PipeBarrier<PIPE_V>();
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> denseBuffer_, planeBuffer_, indexBuffer_,
      scratchBuffer_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  LocalTensor<float> dense_, planes_, a_, b_, c_;
  LocalTensor<int32_t> inIndex_, gateIndex_, matrixIndex_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t h_, width_, cap_;
  bool indices_ready_;
};

}  // namespace kernels
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f03_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  using namespace AscendC;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  if (tiling.h == 2 && tiling.iters >= 2 && tiling.iters <= 4 &&
      tiling.eps >= 0x1p-24f && tiling.eps <= 1.0e-3f) {
    const uint32_t capacity = sinkhorn_norm::kernels::H2BlockedCapacity(tiling);
    if (capacity >= 64) {
      sinkhorn_norm::kernels::F03H2Blocked packed;
      packed.Init(mixes, pre, post, comb, tiling, capacity);
      packed.Process();
      return;
    }
  }
  sinkhorn_norm::kernels::F03_TinyTokenPlanes op;
  op.Init(mixes, pre, post, comb, tiling,
          sinkhorn_norm::kernels::GetCompactPlaneCapacity(tiling));
  op.Process();
  PipeBarrier<PIPE_ALL>();
}

#endif  // SINKHORN_NORM_KERNELS_F03_TINY_TOKEN_PLANES_H_
