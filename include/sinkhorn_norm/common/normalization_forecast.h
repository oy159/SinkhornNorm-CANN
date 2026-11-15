#ifndef SINKHORN_NORM_COMMON_NORMALIZATION_FORECAST_H_
#define SINKHORN_NORM_COMMON_NORMALIZATION_FORECAST_H_

#include "kernel_operator.h"

namespace sinkhorn_norm {

// Project finite denominators from two measured row/column residuals.
template <uint32_t Steps = 4>
static __aicore__ __attribute__((always_inline)) inline void ForecastDenominatorProduct(
    __ubuf__ float *row, __ubuf__ float *col, __ubuf__ float *scratch,
    __ubuf__ float *next, __ubuf__ float *ones, __ubuf__ float *recent,
    __ubuf__ float *older, uint32_t rows, float eps) {
  static_assert(Steps == 4 || Steps == 6);
  using namespace AscendC;
  if ASCEND_IS_AIV {
    const uint32_t span = 2 * rows;
    const uint8_t repeats = (span + 63) / 64;
    auto product = scratch;
    auto norm = product + span;
    auto cross = norm + span;
    auto rate = cross + span;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vmul(norm, older, older, repeats, 1, 1, 1, 8, 8, 8);
    vmul(cross, older, recent, repeats, 1, 1, 1, 8, 8, 8);
    vector_dup(product, 0.0f, repeats, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vadds(norm, norm, 0x1p-28f, repeats, 1, 1, 8, 8);
    vadds(cross, cross, static_cast<float>(0.8f * 0x1p-28f), repeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(rate, cross, norm, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmax(rate, rate, product, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmins(rate, rate, 1.0f, repeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(cross, recent, rate, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    const float center = 1.0f + eps;
    vadds(product, cross, static_cast<float>(center), repeats, 1, 1, 8, 8);
    vmul(next, cross, rate, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(norm, next, static_cast<float>(center), repeats, 1, 1, 8, 8);
    vmul(cross, next, rate, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(product, product, norm, repeats, 1, 1, 1, 8, 8, 8);
    vmul(next, cross, rate, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    auto extra=scratch + 4 * span;
    if constexpr (Steps == 6)
      vmul(extra, next, rate, repeats, 1, 1, 1, 8, 8, 8);
    vadds(cross, cross, static_cast<float>(center), repeats, 1, 1, 8, 8);
    vadds(norm, next, static_cast<float>(center), repeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(cross, cross, norm, repeats, 1, 1, 1, 8, 8, 8);
    if constexpr (Steps == 6)
      vmul(next, extra, rate, repeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(product, product, cross, repeats, 1, 1, 1, 8, 8, 8);
    if constexpr (Steps == 6) {
      vadds(extra, extra, static_cast<float>(center), repeats, 1, 1, 8, 8);
      vadds(next, next, static_cast<float>(center), repeats, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vmul(norm, extra, next, repeats, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vmul(product, product, norm, repeats, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);
    const uint8_t rowRepeats = (rows + 63) / 64;
    SetVectorMask<float, MaskMode::COUNTER>(0, rows);
    vdiv(row, ones, product, rowRepeats, 1, 1, 1, 8, 8, 8);
    vdiv(col, ones, product + rows, rowRepeats, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(uint64_t{0}, ~uint64_t{0});
  }
}

}  // namespace sinkhorn_norm

#endif
