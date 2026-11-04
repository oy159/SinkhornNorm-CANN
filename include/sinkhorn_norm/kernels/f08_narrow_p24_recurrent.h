#ifndef SINKHORN_NORM_KERNELS_F08_NARROW_P24_RECURRENT_H_
#define SINKHORN_NORM_KERNELS_F08_NARROW_P24_RECURRENT_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f08 {

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
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first), reinterpret_cast<__ubuf__ uint32_t *>(a + first), repeats, 1, 1, 8, 8);
    first += 64 * repeats;
  }
  if (first < count) {
    WideRoundMask((uint64_t{1} << (count - first)) - 1);
    if constexpr (Op == 0) vadd(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 1) vadds(dst + first, a + first, scalar, 1, 1, 1, 8, 8);
    if constexpr (Op == 2) vdiv(dst + first, a + first, b + first, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (Op == 3) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst + first), reinterpret_cast<__ubuf__ uint32_t *>(a + first), 1, 1, 1, 8, 8);
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

static __aicore__ __attribute__((always_inline)) inline void RunP24CompleteRounds(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *tmp,
    __ubuf__ float *result, __ubuf__ float *ones, __ubuf__ float *denom,
    uint32_t h, uint32_t valid, float eps, uint32_t iters) {
  if ASCEND_IS_AIV {
    const uint32_t rows = h * valid, span = 24 * valid;
    const uint32_t groups = rows / 8, tail = rows % 8;
    const uint32_t rp = (rows + 7) / 8 * 8;
    const uint32_t rowRepeats = (rows + 63) / 64;
    const uint32_t spanRepeats = (span + 63) / 64;
    const uint32_t matRepeats = (rows * 24 + 63) / 64;
    const uint64_t lastMask = ((uint64_t{1} << (h - 16)) - 1) *
                              uint64_t{0x0101010101010101};
    const uint64_t tailMask = tail ? (uint64_t{1} << (8 * tail)) - 1 : 0;
    const uint64_t colTailMask =
        (span & 63) ? ((uint64_t{1} << (span & 63)) - 1) : ~uint64_t{0};
    const uint32_t halfRows = h / 2;
    const uint32_t upperRows = h - halfRows;

    // Softmax: Broadcast RowMax -> 3-block vsub -> vexp
    WideRoundMask(~uint64_t{0});
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
          reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, rp / 8);
    pipe_barrier(PIPE_V);

    for (uint32_t block = 0; block < 3; ++block) {
      const uint64_t mask = block == 2 ? lastMask : ~uint64_t{0};
      WideRoundMask(mask);
      auto m = matrix + block * 8;
      if (groups != 0) {
        vsub(m, m, tmp, groups, 3, 3, 1, 24, 24, 8);
      }
      if (tail) {
        WideRoundMask(mask & tailMask);
        auto last = m + groups * 192;
        vsub(last, last, tmp + groups * 64, 1, 3, 3, 1, 24, 24, 8);
      }
    }
    pipe_barrier(PIPE_V);

    WideRoundMask(~uint64_t{0});
    vexp(matrix, matrix, matRepeats, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // Unified loop for Round 0 (Softmax row-scale + half-row binary column tree)
    // and Rounds 1..iters-1. Every RAW dependency is protected by pipe_barrier(PIPE_V).
    for (uint32_t round = 0; round < iters; ++round) {
      for (uint32_t block = 0; block < 3; ++block) {
        const uint64_t mask = block == 2 ? lastMask : ~uint64_t{0};
        WideRoundMask(mask);
        auto dst = block == 0 ? result : partial + (block - 1) * rp;
        if (groups != 0) {
          vcgadd(dst, matrix + block * 8, groups, 1, 3, 24);
        }
        if (tail) {
          WideRoundMask(mask & tailMask);
          vcgadd(dst + groups * 8, matrix + groups * 192 + block * 8,
                 1, 1, 3, 24);
        }
      }
      pipe_barrier(PIPE_V);
      WideRoundMask(~uint64_t{0});
      for (uint32_t first = 0; first < rows; first += 64) {
        const uint32_t width = rows - first < 64 ? rows - first : 64;
        if (width < 64) WideRoundMask((uint64_t{1} << width) - 1);
        vadd(result + first, partial + first, result + first, 2,
             1, 1, 1, 0, rp / 8, 0);
        if (width < 64) WideRoundMask(~uint64_t{0});
      }
      pipe_barrier(PIPE_V);

      if (round > 0) {
        vadds(result, result, eps, rowRepeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(result, ones, result, rowRepeats, 1, 1, 1, 8, 8, 8);
      } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, rows);
        vdiv(result, ones, result, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        SetMaskNorm();
        WideRoundMask(~uint64_t{0});
      }
      pipe_barrier(PIPE_V);

      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result), 1, 8, rp / 8);
      pipe_barrier(PIPE_V);

      for (uint32_t block = 0; block < 3; ++block) {
        const uint64_t mask = block == 2 ? lastMask : ~uint64_t{0};
        WideRoundMask(mask);
        auto m = matrix + block * 8;
        if (groups != 0) {
          vmul(m, m, tmp, groups, 3, 3, 1, 24, 24, 8);
        }
        if (tail) {
          WideRoundMask(mask & tailMask);
          auto last = m + groups * 192;
          vmul(last, last, tmp + groups * 64, 1, 3, 3, 1, 24, 24, 8);
        }
      }
      pipe_barrier(PIPE_V);
      WideRoundMask(~uint64_t{0});

      if (round == 0) {
        vadds(matrix, matrix, eps, matRepeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
      }

      if (h == 23) {
        // Direct unrolled column binary tree for JP4 (H=23)
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, 11 * span);
        vadd(partial, matrix, matrix + 12 * span, 1, 1, 1, 1, 8, 8, 8);
        SetVectorMask<float, MaskMode::COUNTER>(0, span);
        vcopy(reinterpret_cast<__ubuf__ uint32_t *>(partial + 11 * span),
              reinterpret_cast<__ubuf__ uint32_t *>(matrix + 11 * span), 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // 12 -> 6
        SetVectorMask<float, MaskMode::COUNTER>(0, 6 * span);
        vadd(partial, partial, partial + 6 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 6 -> 3
        SetVectorMask<float, MaskMode::COUNTER>(0, 3 * span);
        vadd(partial, partial, partial + 3 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // 3 -> 2
        SetVectorMask<float, MaskMode::COUNTER>(0, span);
        vadd(partial, partial, partial + 2 * span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // Final denom
        vadd(denom, partial, partial + span, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(denom, denom, eps, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(denom, ones, denom, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        SetMaskNorm();
        WideRoundMask(~uint64_t{0});
      } else {
        WideRoundSpan<0>(partial, matrix, matrix + upperRows * span,
                         halfRows * span);
        if (h % 2) {
          WideRoundSpan<3>(partial + halfRows * span,
                           matrix + halfRows * span, matrix, span);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t planes = upperRows; planes > 2;) {
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
        vadds(denom, denom, eps, spanRepeats, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(denom, ones, denom, spanRepeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      }

      uint32_t col = 0;
      for (; col + 64 <= span; col += 64) {
        vmul(matrix + col, matrix + col, denom + col, h,
             1, 1, 1, span / 8, span / 8, 0);
      }
      if (col < span) {
        WideRoundMask(colTailMask);
        vmul(matrix + col, matrix + col, denom + col, h,
             1, 1, 1, span / 8, span / 8, 0);
        WideRoundMask(~uint64_t{0});
      }
      pipe_barrier(PIPE_V);
    }
  }
}

static __aicore__ __attribute__((always_inline)) inline void P24FoldRowMax(
    __ubuf__ float *matrix, __ubuf__ float *partial, __ubuf__ float *result,
    uint32_t h, uint32_t rows, bool blocked) {
  if ASCEND_IS_AIV {
    const uint32_t groups=rows/8, tail=rows%8;
    const uint64_t tailMask=tail ? (uint64_t{1}<<(8*tail))-1 : 0;
    const uint64_t lastMask=((uint64_t{1}<<(h-16))-1)*uint64_t{0x0101010101010101};
    const uint32_t stride=blocked ? 1 : 3, plane=blocked ? rows*8 : 8;
    SetMaskNorm();
    WideRoundMask(~uint64_t{0});
    if(groups) vmax(partial,matrix,matrix+plane,groups,1,stride,stride,8,8*stride,8*stride);
    if(tail) {
      WideRoundMask(tailMask);
      vmax(partial+groups*64,matrix+groups*64*stride,matrix+groups*64*stride+plane,1,1,stride,stride,8,8*stride,8*stride);
    }
    pipe_barrier(PIPE_V);
    WideRoundMask(lastMask);
    if(groups) vmax(partial,partial,matrix+2*plane,groups,1,1,stride,8,8,8*stride);
    if(tail) {
      WideRoundMask(lastMask&tailMask);
      vmax(partial+groups*64,partial+groups*64,matrix+groups*64*stride+2*plane,1,1,1,stride,8,8,8*stride);
    }
    pipe_barrier(PIPE_V);
    WideRoundMask(~uint64_t{0});
    if(groups) vcgmax(result,partial,groups,1,1,8);
    if(tail) {
      WideRoundMask(tailMask);
      vcgmax(result+groups*8,partial+groups*64,1,1,1,8);
    }
    pipe_barrier(PIPE_V);
    ResetMask();
  }
}

// Dense layout: [column block][row][token][eight columns].
// No matrix layout conversion is performed inside the iteration loop.
static __aicore__ __attribute__((always_inline)) inline void RunP24BlockedRounds(
    __ubuf__ float *m, __ubuf__ float *partial, __ubuf__ float *tmp,
    __ubuf__ float *r, __ubuf__ float *ones, __ubuf__ float *d,
    uint32_t h, uint32_t n, float eps, uint32_t iters) {
  if ASCEND_IS_AIV {
    const uint32_t rows=h*n, rp=(rows+7)/8*8, plane=rows*8;
    const uint64_t lastMask=((uint64_t{1}<<(h-16))-1)*uint64_t{0x0101010101010101};
    WideRoundMask(~uint64_t{0});
    vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),reinterpret_cast<__ubuf__ uint32_t *>(r),1,8,rp/8);
    pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(0,plane);
    for(uint32_t b=0;b<3;++b) vsub(m+b*plane,m+b*plane,tmp,1,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    SetVectorMask<float,MaskMode::COUNTER>(0,3*plane);
    vexp(m,m,1,1,1,8,8);
    pipe_barrier(PIPE_V);
    SetMaskNorm();
    for(uint32_t round=0;round<iters;++round) {
      for(uint32_t b=0;b<3;++b) {
        const uint64_t mask=b==2 ? lastMask : ~uint64_t{0};
        WideRoundMask(mask);
        // Matrix has 64 extra floats. Up to seven extra row sums stay in
        // this block's rp-sized partial range and are never consumed.
        vcgadd(partial+b*rp,m+b*plane,rp/8,1,1,8);
      }
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,rows);
      vadd(r,partial,partial+rp,1,1,1,1,8,8,8);
      if(round) vadds(partial+2*rp,partial+2*rp,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      vadd(r,r,partial+2*rp,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      vdiv(r,ones,r,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();WideRoundMask(~uint64_t{0});
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),reinterpret_cast<__ubuf__ uint32_t *>(r),1,8,rp/8);
      pipe_barrier(PIPE_V);
      SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,plane);
      for(uint32_t b=0;b<3;++b) vmul(m+b*plane,m+b*plane,tmp,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      if(!round) {
        SetVectorMask<float,MaskMode::COUNTER>(0,3*plane);
        vadds(m,m,eps,1,1,1,8,8);pipe_barrier(PIPE_V);
      }
      SetMaskNorm();WideRoundMask(~uint64_t{0});
      // Compact all three column planes with the same binary row tree.
      // partial+tmp are free after row scaling; keep each block's tree separate.
      const uint32_t span=8*n, half=h/2, pitch=h-half, work=pitch*span;
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,half*span);
      for(uint32_t b=0;b<3;++b)
        vadd(partial+b*work,m+b*plane,m+b*plane+pitch*span,1,1,1,1,8,8,8);
      if(h%2) {
        SetVectorMask<float,MaskMode::COUNTER>(0,span);
        for(uint32_t b=0;b<3;++b)
          vadds(partial+b*work+half*span,m+b*plane+half*span,eps,1,1,1,8,8);
      }
      pipe_barrier(PIPE_V);
      for(uint32_t height=pitch;height>2;) {
        const uint32_t pairs=height/2, upper=height-pairs;
        SetVectorMask<float,MaskMode::COUNTER>(0,pairs*span);
        for(uint32_t b=0;b<3;++b)
          vadd(partial+b*work,partial+b*work,partial+b*work+upper*span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);height=upper;
      }
      SetVectorMask<float,MaskMode::COUNTER>(0,span);
      for(uint32_t b=0;b<3;++b)
        vadd(d+b*span,partial+b*work,partial+b*work+span,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetVectorMask<float,MaskMode::COUNTER>(0,3*span);
      if(!(h%2)) {vadds(d,d,eps,1,1,1,8,8);pipe_barrier(PIPE_V);}
      vdiv(d,ones,d,1,1,1,1,8,8,8);pipe_barrier(PIPE_V);
      SetMaskNorm();
      WideRoundMask(~uint64_t{0});
      uint32_t off=0;
      for(;off+64<=span;off+=64)
        for(uint32_t b=0;b<3;++b)
          vmul(m+b*plane+off,m+b*plane+off,d+b*span+off,h,1,1,1,n,n,0);
      if(off<span) {
        if(n<=31 && span-off<=16) {
          for(uint32_t i=off;i<span;i+=8)
            for(uint32_t b=0;b<3;++b)
              vmul(m+b*plane+i,m+b*plane+i,d+b*span+i,h/8,n,n,0,8*n,8*n,0);
          if(h%8) {
            WideRoundMask((uint64_t{1}<<(8*(h%8)))-1);
            for(;off<span;off+=8)
              for(uint32_t b=0;b<3;++b) {
                auto q=m+b*plane+(h/8)*8*span+off;
                vmul(q,q,d+b*span+off,1,n,n,0,8*n,8*n,0);
              }
          }
        } else {
          WideRoundMask((uint64_t{1}<<(span-off))-1);
          for(uint32_t b=0;b<3;++b)
            vmul(m+b*plane+off,m+b*plane+off,d+b*span+off,h,1,1,1,n,n,0);
        }
      }
      pipe_barrier(PIPE_V);
    }
    WideRoundMask(~uint64_t{0});
  }
}

class NarrowP24RecurrentKernel {
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

    const uint32_t rp = (t_.h * t_.tile_tokens + 7) / 8 * 8;
    const uint32_t span = t_.tile_tokens * t_.h_pad;
    const uint32_t ones = rp > 2 * span ? rp : 2 * span;
    const uint32_t tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;

    dense_gate_cap_ = (t_.h * t_.tile_tokens + 63) / 64 * 64;
    const uint32_t matrixCap = (t_.h * t_.h * t_.tile_tokens + 63) / 64 * 64;
    zero_offset_ = 2 * dense_gate_cap_ + matrixCap;
    input_gate_cap_ = (2 * span + 63) / 64 * 64;
    const uint32_t inputMatrixCap = (t_.h * span + 63) / 64 * 64;

    const uint32_t baseBytes =
        (t_.matrix_bytes + 256) + (2 * t_.gate_bytes + 256) +
        (t_.input_bytes + (tmp + rp) * sizeof(float) + 256) +
        (rp * sizeof(float) + 256) + (ones * sizeof(float) + 256) +
        (span * sizeof(float) + 256);
    const uint32_t denseBytes =
        (4 * zero_offset_ + 32) + 4 * (dense_gate_cap_ + matrixCap) +
        4 * (input_gate_cap_ + inputMatrixCap);
    use_dense_io_ = (baseBytes + denseBytes <= 188416);

    // Allocate +256B (64 floats) tail padding on UB buffers to allow full-repeat
    // vector execution without SetVectorMask tail splitting.
    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes + 256);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes + 256);
    pipe_.InitBuffer(partialBuffer_,
                     t_.input_bytes + (tmp + rp) * sizeof(float) + 256);
    pipe_.InitBuffer(resultBuffer_, rp * sizeof(float) + 256);
    pipe_.InitBuffer(onesBuffer_, ones * sizeof(float) + 256);
    pipe_.InitBuffer(denomBuffer_, span * sizeof(float) + 256);

    if (use_dense_io_) {
      // Reserve 32B (8 floats) at zero_offset_ permanently holding 0.0f so Gather
      // automatically zero-pads lanes h..23 of matrix_ and gates_!
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
    rp_ = rp;
    ones_cap_ = ones;
    constants_ready_ = false;

    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>()[32];
    tmp_ = partial_[t_.input_bytes / sizeof(float)];
    result_ = resultBuffer_.Get<float>();
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();
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
      if (use_dense_io_) {
      RunP24BlockedRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          t_.h, valid_, t_.eps, t_.iters);
      } else {
      RunP24CompleteRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          t_.h, valid_, t_.eps, t_.iters);
      }
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
          const uint64_t n = first + token;
          DataCopyPad(pre_[n * t_.h], gates_[token * t_.h_pad], gOut);
          DataCopyPad(post_[n * t_.h], gates_[span_ + token * t_.h_pad], gOut);
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
    CreateVecIndex(input_index_, int32_t{0}, t_.h_pad);
    PipeBarrier<PIPE_V>();Muls(input_index_,input_index_,int32_t{4},t_.h_pad);
    PipeBarrier<PIPE_V>();
    auto dst=input_index_[2*span_];
    Adds(dst,input_index_,static_cast<int32_t>(8*t_.h),8);PipeBarrier<PIPE_V>();
    for(uint32_t tokens=1;tokens<valid_;tokens*=2) {
      const uint32_t n=valid_-tokens<tokens ? valid_-tokens : tokens;
      const int32_t offset=4*tokens*t_.input_width;
      Adds(input_index_[tokens*24],input_index_,offset,n*24);
      Adds(dst[tokens*8],dst,offset,n*8);
      PipeBarrier<PIPE_V>();
    }
    Adds(input_index_[span_],input_index_,static_cast<int32_t>(4*t_.h),span_);
    for(uint32_t rows=1;rows<t_.h;rows*=2) {
      const uint32_t n=t_.h-rows<rows ? t_.h-rows : rows;
      Adds(dst[rows*valid_*8],dst,static_cast<int32_t>(4*rows*t_.h),n*valid_*8);PipeBarrier<PIPE_V>();
    }
    Adds(dst[rows_*8],dst,int32_t{32},rows_*8);
    Adds(dst[rows_*16],dst,int32_t{64},rows_*8);PipeBarrier<PIPE_V>();
    if(t_.h<24) {
      uint64_t mask[2]={uint64_t{255}^((uint64_t{1}<<(t_.h-16))-1),0};
      for(uint32_t first=0;first<rows_;first+=248) {
        const uint32_t n=rows_-first<248?rows_-first:248;
        Duplicate(dst[rows_*16+first*8],static_cast<int32_t>(4*zero_offset_),mask,
                  static_cast<uint8_t>(n),uint16_t{1},uint8_t{1});
      }
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyInAndBuildMaps(uint64_t first) {
    // Issue single contiguous bulk MTE2 DMA read first so constants and index maps build on PIPE_V in parallel!
    const DataCopyExtParams packed{1, 4 * valid_ * t_.input_width, 0, 0, 0};
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(dense_output_, input_[first * t_.input_width], packed, noPad);

    if (!constants_ready_) {
      Duplicate(result_, 0.0f, rp_ + 64);
      Duplicate(ones_, 1.0f, ones_cap_ + 64);
      Duplicate(dense_output_[zero_offset_], 0.0f, 8);
      PipeBarrier<PIPE_V>();
      constants_ready_ = true;
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
             static_cast<int32_t>(4 * tokens * 8), n * t_.h * t_.h);
        PipeBarrier<PIPE_V>();
      }
      indexed_valid_ = valid_;
    }
    // Output maps borrow matrix/input-index storage before Gather populates it.
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
    const uint32_t gatesCount = t_.h * valid_;
    const uint32_t n = 2 * span_;

    // Interleave Gate Muls(-1) with RowMax on matrix_!
    Muls(tmp_, gates_, -1.0f, n);
    P24FoldRowMax(reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
                  t_.h, rows_, use_dense_io_);
    PipeBarrier<PIPE_V>();

    Mins(tmp_, tmp_, 80.0f, n);
    PipeBarrier<PIPE_V>();
    Exp(tmp_, tmp_, n);
    PipeBarrier<PIPE_V>();
    Adds(tmp_, tmp_, 1.0f, n);
    PipeBarrier<PIPE_V>();
    Div(gates_, ones_, tmp_, n);
    PipeBarrier<PIPE_V>();
    Adds(gates_, gates_, t_.eps, span_);
    Muls(gates_[span_], gates_[span_], 2.0f, span_);
    PipeBarrier<PIPE_V>();

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
    auto row = matrix_.template ReinterpretCast<int32_t>();
    auto token = input_index_;
    const int32_t reciprocal =
        static_cast<int32_t>(((1U << 20) + t_.h - 1) / t_.h);
    const uint32_t chunk = t_.matrix_bytes / sizeof(int32_t);
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
        Muls(token,row,static_cast<int32_t>(t_.h),length);PipeBarrier<PIPE_V>();
        Sub(token,dst,token,length);PipeBarrier<PIPE_V>();
        ShiftRight(token,token,int32_t{3},length);PipeBarrier<PIPE_V>();
        Muls(token,token,static_cast<int32_t>(8*rows_-8),length);PipeBarrier<PIPE_V>();
        Add(dst,dst,token,length);PipeBarrier<PIPE_V>();
        Muls(token,row,reciprocal,length);PipeBarrier<PIPE_V>();
        ShiftRight(token,token,int32_t{20},length);PipeBarrier<PIPE_V>();
        Muls(token,token,int32_t{8}-static_cast<int32_t>(8*rows_),length);
        Muls(row,row,static_cast<int32_t>(8*valid_)-static_cast<int32_t>(t_.h),length);
        PipeBarrier<PIPE_V>();Add(dst,dst,row,length);PipeBarrier<PIPE_V>();
        Add(dst,dst,token,length);
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
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                              4 * t_.h, 0, stride, 0};
    const DataCopyExtParams g{2, 4 * t_.h, 0, stride, 0};
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
      Duplicate(result_, 0.0f, rp_ + 64);
      Duplicate(ones_, 1.0f, ones_cap_ + 64);
      PipeBarrier<PIPE_V>();
      constants_ready_ = true;
    }

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t n = 2 * span_;
    Muls(tmp_, gates_, -1.0f, n);
    PipeBarrier<PIPE_V>();
    Mins(tmp_, tmp_, 80.0f, n);
    PipeBarrier<PIPE_V>();
    Exp(tmp_, tmp_, n);
    PipeBarrier<PIPE_V>();
    Adds(tmp_, tmp_, 1.0f, n);
    PipeBarrier<PIPE_V>();
    Div(gates_, ones_, tmp_, n);
    PipeBarrier<PIPE_V>();
    Adds(gates_, gates_, t_.eps, span_);
    Muls(gates_[span_], gates_[span_], 2.0f, span_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
  }

  __aicore__ inline void RowMaxOnly() {
    P24FoldRowMax(reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
                  t_.h, rows_, use_dense_io_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyOutCombStrided(uint64_t first, bool hasMoreTiles) {
    Sync<HardEvent::V_MTE3>();
    const uint32_t stride = (valid_ - 1) * t_.h_pad / 8;
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),
                              4 * t_.h, stride, 0, 0};
    for (uint32_t token = 0; token < valid_; ++token) {
      const uint64_t n = first + token;
      DataCopyPad(comb_[n * t_.h * t_.h], matrix_[token * t_.h_pad], m);
    }
    if (hasMoreTiles) {
      Sync<HardEvent::MTE3_MTE2>();
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_, gateBuffer_, denseOutputBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_, resultBuffer_, onesBuffer_,
      denomBuffer_, outputIndexBuffer_, inputIndexBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, result_, ones_, denom_,
      dense_output_;
  LocalTensor<int32_t> output_index_, input_index_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t valid_, rows_, span_, rp_, ones_cap_, dense_gate_cap_,
      input_gate_cap_, zero_offset_, indexed_valid_, input_indexed_valid_;
  bool constants_ready_, gate_output_map_ready_, use_dense_io_;
};

}  // namespace f08
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f08_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f08::NarrowP24RecurrentKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F08_NARROW_P24_RECURRENT_H_
