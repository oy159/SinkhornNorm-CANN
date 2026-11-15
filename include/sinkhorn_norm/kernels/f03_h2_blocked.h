#ifndef SINKHORN_NORM_KERNELS_F03_H2_BLOCKED_H_
#define SINKHORN_NORM_KERNELS_F03_H2_BLOCKED_H_
#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"
namespace sinkhorn_norm {
namespace kernels {
using namespace AscendC;

__aicore__ inline bool H2CanPipeline(const TilingData &t) {
  const uint64_t budget = uint64_t{t.input_bytes} + t.matrix_bytes +
                          2 * uint64_t{t.gate_bytes} + t.scratch_bytes;
  const uint64_t owner = uint64_t{t.tokens_per_core} + (t.extra_core_count != 0);
  const uint64_t physical = budget / 64 / 64 * 64;
  const uint64_t singleLimit = physical < 4032 ? physical : 4032;
  return owner > singleLimit && budget >= 96 * 64;
}

__aicore__ inline uint32_t H2BlockedCapacity(const TilingData &t) {
  const uint64_t budget = uint64_t{t.input_bytes} + t.matrix_bytes +
                          2 * uint64_t{t.gate_bytes} + t.scratch_bytes;
  const uint64_t owner = uint64_t{t.tokens_per_core} + (t.extra_core_count != 0);
  const uint64_t bytesPerToken = H2CanPipeline(t) ? 96 : 64;
  const uint64_t physical = budget / bytesPerToken / 64 * 64;
  const uint64_t limit = physical < 4032 ? physical : 4032;
  if (limit == 0 || owner == 0) return 0;
  const uint64_t tiles = (owner + limit - 1) / limit;
  return static_cast<uint32_t>(((owner + tiles - 1) / tiles + 63) / 64 * 64);
}

// Within every 64-token block the eight planes use an 8x8 token order.
// This lets the output transpose pack pairs/fours directly, with no indexed
// Gather or compression pass. Every requested epsilon round is retained.
class F03H2Blocked {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR pre, GM_ADDR post,
                             GM_ADDR comb, TilingData t, uint32_t capacity) {
    t_ = t;
    cap_ = capacity;
    const uint64_t block = GetBlockIdx();
    count_ = t.tokens_per_core + (block < t.extra_core_count);
    start_ = block * t.tokens_per_core +
             (block < t.extra_core_count ? block : t.extra_core_count);
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));
    pipelined_ = H2CanPipeline(t);
    pipe_.InitBuffer(denseBuffer_, 32 * cap_ * (pipelined_ ? 2 : 1));
    pipe_.InitBuffer(planeBuffer_, 32 * cap_);
    dense_ = denseBuffer_.Get<float>();
    denseBase_ = dense_;
    planes_ = planeBuffer_.Get<float>();
  }

  __aicore__ inline void Process() {
    if (pipelined_) {
      ProcessPipelined();
      return;
    }
    for (uint64_t offset = 0; offset < count_; offset += cap_) {
      const uint32_t valid = count_ - offset < cap_ ? count_ - offset : cap_;
      const uint32_t p = (valid + 63) / 64 * 64;
      const uint64_t first = start_ + offset;
      const DataCopyExtParams inputCopy{1, 32 * valid, 0, 0, 0};
      const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
      DataCopyPad(dense_, input_[first * 8], inputCopy, padding);
      Sync<HardEvent::MTE2_V>();
      if (valid != p) {
        Duplicate(dense_[8 * valid], 0.0f, 8 * (p - valid));
        PipeBarrier<PIPE_V>();
      }
      TransposeInput(p);
      PipeBarrier<PIPE_V>();
      Normalize(p);
      TransposeOutput(p);
      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams gateCopy{1, 8 * valid, 0, 0, 0};
      const DataCopyExtParams matrixCopy{1, 16 * valid, 0, 0, 0};
      DataCopyPad(pre_[first * 2], dense_, gateCopy);
      DataCopyPad(post_[first * 2], dense_[2 * p], gateCopy);
      DataCopyPad(comb_[first * 4], dense_[4 * p], matrixCopy);
      if (offset + cap_ < count_) Sync<HardEvent::MTE3_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
  }

 private:
  template <HardEvent Event> __aicore__ inline void Sync() {
    SetFlag<Event>(EVENT_ID0);
    WaitFlag<Event>(EVENT_ID0);
  }

  __aicore__ inline event_t SlotEvent(uint32_t slot) {
    return slot == 0 ? EVENT_ID0 : EVENT_ID1;
  }

  __aicore__ inline void LoadSlot(uint32_t slot, uint64_t offset,
                                 bool &storePending) {
    if (storePending) {
      WaitFlag<HardEvent::MTE3_MTE2>(SlotEvent(slot));
      storePending = false;
    }
    const uint32_t valid = count_ - offset < cap_ ? count_ - offset : cap_;
    const DataCopyExtParams copy{1, 32 * valid, 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
    DataCopyPad(denseBase_[slot * 8 * cap_], input_[(start_ + offset) * 8],
                copy, padding);
    SetFlag<HardEvent::MTE2_V>(SlotEvent(slot));
  }

  __aicore__ inline void ProcessPipelined() {
    if (count_ == 0) return;
    bool storePending[2] = {false, false};
    LoadSlot(0, 0, storePending[0]);
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < count_; offset += cap_) {
      const uint32_t valid = count_ - offset < cap_ ? count_ - offset : cap_;
      const uint32_t p = (valid + 63) / 64 * 64;
      const uint64_t first = start_ + offset;
      WaitFlag<HardEvent::MTE2_V>(SlotEvent(slot));
      dense_ = denseBase_[slot * 8 * cap_];
      if (valid != p) {
        Duplicate(dense_[8 * valid], 0.0f, 8 * (p - valid));
        PipeBarrier<PIPE_V>();
      }
      // The previous store reads the other dense slot. It can finish while
      // the current input transpose consumes this slot and fills planes_.
      TransposeInput(p);
      PipeBarrier<PIPE_V>();
      if (offset + cap_ < count_) {
        const uint32_t other = slot ^ 1;
        LoadSlot(other, offset + cap_, storePending[other]);
      }
      Normalize(p);
      TransposeOutput(p);
      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams gates{1, 8 * valid, 0, 0, 0};
      const DataCopyExtParams matrix{1, 16 * valid, 0, 0, 0};
      DataCopyPad(pre_[first * 2], dense_, gates);
      DataCopyPad(post_[first * 2], dense_[2 * p], gates);
      DataCopyPad(comb_[first * 4], dense_[4 * p], matrix);
      SetFlag<HardEvent::MTE3_MTE2>(SlotEvent(slot));
      storePending[slot] = true;
      slot ^= 1;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (storePending[i]) WaitFlag<HardEvent::MTE3_MTE2>(SlotEvent(i));
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void TransposeInput(uint32_t p) {
    uint64_t src[16], dst[16];
    TransDataTo5HDParams param{};
    param.repeatTimes = p / 64;
    param.srcRepStride = p == 64 ? 0 : 64;
    param.dstRepStride = p == 64 ? 0 : 8;
    for (uint32_t phase = 0; phase < 4; ++phase) {
      #pragma unroll
      for (uint32_t i = 0; i < 16; ++i) {
        src[i] = reinterpret_cast<uint64_t>(dense_[
            8 * ((i % 8) * 8 + i / 8 + phase * 2)].GetPhyAddr());
        dst[i] = reinterpret_cast<uint64_t>(planes_[
            (i / 2) * p + (i % 2) * 8 + phase * 16].GetPhyAddr());
      }
      TransDataTo5HD<float>(dst, src, param);
    }
  }

  __aicore__ inline void TransposeOutput(uint32_t p) {
    uint64_t src[16], dst[16];
    TransDataTo5HDParams param{};
    param.repeatTimes = p / 64;
    param.srcRepStride = p == 64 ? 0 : 8;
    param.dstRepStride = p == 64 ? 0 : 16;
    for (uint32_t part = 0; part < 2; ++part) {
      #pragma unroll
      for (uint32_t i = 0; i < 16; ++i) {
        src[i] = reinterpret_cast<uint64_t>(planes_[
            (part * 2 + i % 2) * p + (i / 2) * 8].GetPhyAddr());
        dst[i] = reinterpret_cast<uint64_t>(dense_[
            part * 2 * p + (i / 2) * 16 + (i % 2) * 8].GetPhyAddr());
      }
      TransDataTo5HD<float>(dst, src, param);
    }
    param.dstRepStride = p == 64 ? 0 : 32;
    for (uint32_t phase = 0; phase < 2; ++phase) {
      #pragma unroll
      for (uint32_t i = 0; i < 16; ++i) {
        src[i] = reinterpret_cast<uint64_t>(planes_[
            (4 + i % 4) * p + (i / 4) * 8 + phase * 32].GetPhyAddr());
        dst[i] = reinterpret_cast<uint64_t>(dense_[
            4 * p + (i / 2) * 32 + (i % 2) * 8 + phase * 16].GetPhyAddr());
      }
      TransDataTo5HD<float>(dst, src, param);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Normalize(uint32_t p) {
    if ASCEND_IS_AIV {
      auto gates = reinterpret_cast<__ubuf__ float *>(planes_.GetPhyAddr());
      auto a = gates + 4 * p;
      auto b = a + p;
      auto c = b + p;
      auto d = c + p;
      auto scratch = reinterpret_cast<__ubuf__ float *>(dense_.GetPhyAddr());
      auto ratios = scratch + 4 * p;
      auto ones = scratch + 6 * p;
      auto u = scratch + 64;
      auto v = u + p;
      auto w = v + p;
      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      vector_dup(ones, 1.0f, 1, 1, 1, 8, 0);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(0, 4 * p);
      vmuls(scratch, gates, -1.0f, 1, 1, 1, 8, 8);
      SetVectorMask<float, MaskMode::COUNTER>(0, p);
      vsub(ratios, b, a, 1, 1, 1, 1, 8, 8, 8);
      vsub(ratios + p, d, c, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      // Use one exponential per row and form its small output by
      // multiplication, avoiding cancellation from subtracting from one.
      SetVectorMask<float, MaskMode::COUNTER>(0, 6 * p);
      vmins(scratch, scratch, 80.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vexp(scratch, scratch, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::COUNTER>(0, 4 * p);
      vadds(scratch, scratch, 1.0f, 1, 1, 1, 8, 8);
      SetVectorMask<float, MaskMode::COUNTER>(0, p);
      vadds(b, ratios, 1.0f, 1, 1, 1, 8, 8);
      vadds(d, ratios + p, 1.0f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      SetVectorMask<float, MaskMode::NORMAL>(0, ~uint64_t{0});
      vdiv(gates, ones, scratch, 4 * p / 64, 1, 1, 1, 8, 0, 8);
      vdiv(a, ones, b, p / 64, 1, 1, 1, 8, 0, 8);
      vdiv(c, ones, d, p / 64, 1, 1, 1, 8, 0, 8);
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(0, 2 * p);
      vadds(gates, gates, t_.eps, 1, 1, 1, 8, 8);
      vmuls(gates + 2 * p, gates + 2 * p, 2.0f, 1, 1, 1, 8, 8);
      SetVectorMask<float, MaskMode::COUNTER>(0, p);
      vmul(b, ratios, a, 1, 1, 1, 1, 8, 8, 8);
      vmul(d, ratios + p, c, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::COUNTER>(0, 4 * p);
      vadds(a, a, t_.eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float, MaskMode::COUNTER>(0, p);
      vadds(u, a, t_.eps, 1, 1, 1, 8, 8);
      vadds(v, b, t_.eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(u, u, c, 1, 1, 1, 1, 8, 8, 8);
      vadd(v, v, d, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      Divide(a, b, c, d, u, v, true);
      for (uint32_t iter = 1; iter < t_.iters; ++iter) {
        vadd(u, a, b, 1, 1, 1, 1, 8, 8, 8);
        vadd(v, c, d, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(u, u, t_.eps, 1, 1, 1, 8, 8);
        vadds(v, v, t_.eps, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(a, a, v, 1, 1, 1, 1, 8, 8, 8);
        vmul(b, b, v, 1, 1, 1, 1, 8, 8, 8);
        vmul(c, c, u, 1, 1, 1, 1, 8, 8, 8);
        vmul(d, d, u, 1, 1, 1, 1, 8, 8, 8);
        vmul(w, u, v, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(u, a, c, 1, 1, 1, 1, 8, 8, 8);
        vadd(v, b, d, 1, 1, 1, 1, 8, 8, 8);
        vmuls(w, w, t_.eps, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(u, u, w, 1, 1, 1, 1, 8, 8, 8);
        vadd(v, v, w, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        Divide(a, b, c, d, u, v, true);
      }
      SetMaskNorm();
      ResetMask();
    }
  }

  __aicore__ inline void Divide(__ubuf__ float *a, __ubuf__ float *b,
                                __ubuf__ float *c, __ubuf__ float *d,
                                __ubuf__ float *u, __ubuf__ float *v,
                                bool column) {
    if ASCEND_IS_AIV {
      vdiv(a, a, u, 1, 1, 1, 1, 8, 8, 8);
      vdiv(b, b, column ? v : u, 1, 1, 1, 1, 8, 8, 8);
      vdiv(c, c, column ? u : v, 1, 1, 1, 1, 8, 8, 8);
      vdiv(d, d, v, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> denseBuffer_, planeBuffer_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  LocalTensor<float> dense_, denseBase_, planes_;
  TilingData t_;
  uint32_t cap_;
  uint64_t start_, count_;
  bool pipelined_;
};
}  // namespace kernels
}  // namespace sinkhorn_norm
#endif
