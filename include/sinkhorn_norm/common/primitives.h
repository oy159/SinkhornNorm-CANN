#ifndef SINKHORN_NORM_COMMON_PRIMITIVES_H_
#define SINKHORN_NORM_COMMON_PRIMITIVES_H_

#include "kernel_operator.h"

#define SINKHORN_AIV_NOINLINE static __aicore__ __attribute__((noinline))
#define SINKHORN_AIV_NOINLINE_ALIGNED static __aicore__ __attribute__((noinline))

namespace sinkhorn_norm {
namespace primitives {

using namespace AscendC;

template <HardEvent event>
__aicore__ __attribute__((always_inline)) inline void SyncEvent(TPipe &pipe) {
  const auto id = pipe.FetchEventID(event);
  SetFlag<event>(id);
  WaitFlag<event>(id);
}

// Numerically stable single-row sigmoid: exp(min(x,0)) / (1 + exp(-abs(x)))
// followed by +eps (pre) or *2.0f (post).
__aicore__ __attribute__((always_inline)) inline void ComputeGateStable(
    const LocalTensor<float> &gate, const LocalTensor<float> &tmp,
    const LocalTensor<float> &work, const LocalTensor<float> &denom,
    uint32_t h, float eps, bool is_pre) {
  Abs(tmp, gate, h);
  Mins(work, gate, 0.0f, h);
  PipeBarrier<PIPE_V>();
  Muls(tmp, tmp, -1.0f, h);
  Exp(work, work, h);
  PipeBarrier<PIPE_V>();
  Exp(tmp, tmp, h);
  PipeBarrier<PIPE_V>();
  Adds(denom, tmp, 1.0f, h);
  PipeBarrier<PIPE_V>();
  Div(gate, work, denom, h);
  PipeBarrier<PIPE_V>();
  if (is_pre) {
    Adds(gate, gate, eps, h);
  } else {
    Muls(gate, gate, 2.0f, h);
  }
  PipeBarrier<PIPE_V>();
}

// Merged two-row gate pipeline when 2 * h_pad <= 64.
// Reuses three 2*h_pad slices inside the 8*h_pad broadcast buffer.
__aicore__ __attribute__((always_inline)) inline void ComputeGatePairStable(
    const LocalTensor<float> &gates, const LocalTensor<float> &tmp,
    uint32_t h, uint32_t h_pad, float eps) {
  const uint32_t count = 2 * h_pad;
  auto abs_exp = tmp;
  auto numerator = tmp[count];
  auto denominator = tmp[2 * count];
  Abs(abs_exp, gates, count);
  Mins(numerator, gates, 0.0f, count);
  PipeBarrier<PIPE_V>();
  Muls(abs_exp, abs_exp, -1.0f, count);
  Exp(numerator, numerator, count);
  PipeBarrier<PIPE_V>();
  Exp(abs_exp, abs_exp, count);
  PipeBarrier<PIPE_V>();
  Adds(denominator, abs_exp, 1.0f, count);
  PipeBarrier<PIPE_V>();
  Div(gates, numerator, denominator, count);
  PipeBarrier<PIPE_V>();
  Adds(gates, gates, eps, h);
  Muls(gates[h_pad], gates[h_pad], 2.0f, h);
  PipeBarrier<PIPE_V>();
}

}  // namespace primitives
}  // namespace sinkhorn_norm

#endif  // SINKHORN_NORM_COMMON_PRIMITIVES_H_
