#ifndef SINKHORN_NORM_KERNELS_F11_MEDIUM_ALIGNED_64_H_
#define SINKHORN_NORM_KERNELS_F11_MEDIUM_ALIGNED_64_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f11 {

using namespace AscendC;
using U32=uint32_t;
using U64=uint64_t;
using UBFloat=__ubuf__ float;

static __aicore__ __attribute__((always_inline)) inline void Mask64(U64 mask) {
  SetVectorMask<float,MaskMode::NORMAL>(U64{0},mask);
}

template <bool P>
static __aicore__ __attribute__((always_inline)) inline void Rounds64(
    UBFloat *a,UBFloat *b,
    UBFloat *__restrict p,
    UBFloat *__restrict w,UBFloat *__restrict r,
    UBFloat *__restrict u,UBFloat *__restrict d,
    U32 tokens,U32 cap,U32 uo,
    float eps,U32 rounds) {
  if ASCEND_IS_AIV {
    constexpr U32 lanes=64,h=lanes;
    constexpr bool pk=P;
    const U32 n=P ? 4U : tokens;
    const U32 nr=h * n,s=h * n;
    const U32 qr=h / 4,q=qr * s;
    const U32 hr=h / 2,half=hr * s;
    const bool hf=P || (half <= cap && half / lanes <= 255);
    const uint8_t nv=nr / lanes,st=s / 8;

    // Round 0 Softmax (a -> b)
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(0,nr * h);
    vcgmax(p,a,1,1,1,8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    SetMaskNorm();
    Mask64(~U64{0});
    vcgmax(r,p,nr / 8,1,1,8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    vbrcb(reinterpret_cast<__ubuf__ U32 *>(w),
          reinterpret_cast<__ubuf__ U32 *>(r),1,8,nr / 8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(0,nr * h);
    vsub(a,a,w,1,1,1,0,8,8,1);
    if constexpr (!P) pipe_barrier(PIPE_V);
    vexp(a,a,1,1,1,8,8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    vcgadd(p,a,1,1,1,8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    SetMaskNorm();
    Mask64(~U64{0});
    vcgadd(r,p,nr / 8,1,1,8);
    pipe_barrier(PIPE_V);
    vdiv(r,u + uo,r,nv,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    vbrcb(reinterpret_cast<__ubuf__ U32 *>(w),
          reinterpret_cast<__ubuf__ U32 *>(r),1,8,nr / 8);
    if constexpr (!P) pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(0,nr * h);
    vmul(a,a,w,1,1,1,0,8,8,1);
    if constexpr (!P) pipe_barrier(PIPE_V);
    if constexpr (P) {
      SetMaskNorm();
      Mask64(~U64{0});
      for (U32 token=0; token < 4; ++token)
        vadds(b + token * 64,a + token * 4096,
              eps,64,1,1,32,8);
    } else {
      vadds(b,a,eps,1,1,1,8,8);
    }
    if constexpr (!P) pipe_barrier(PIPE_V);
    SetMaskNorm();
    Mask64(~U64{0});

    for (U32 round=0; round < rounds; ++round) {
      if (round > 0) {
        if constexpr (P) {
          // All 256 full nr,split only for the NORMAL repeat limit.
          vcgadd(w,a,128,1,1,8);
          vcgadd(w + 1024,a + 8192,128,1,1,8);
        } else {
          SetMaskCount();
          SetVectorMask<float,MaskMode::COUNTER>(0,nr * h);
          vcgadd(p,a,1,1,1,8);
          pipe_barrier(PIPE_V);
          SetMaskNorm();
          Mask64(~U64{0});
        }
        vcgadd(r,P ? w : p,nr / 8,1,1,8);
        pipe_barrier(PIPE_V);
        vadds(r,r,eps,nv,1,1,8,8);
        pipe_barrier(PIPE_V);
        // Use the other initialized constant phase when both row operands align.
        vdiv(r,u + uo,r,nv,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        vbrcb(reinterpret_cast<__ubuf__ U32 *>(w),
              reinterpret_cast<__ubuf__ U32 *>(r),1,8,nr / 8);
        if constexpr (!P) pipe_barrier(PIPE_V);
        if constexpr (P) {
          // The row reciprocal already runs with the full NORMAL mask.
          for (U32 token=0; token < 4; ++token)
            vmul(b + token * 64,a + token * 4096,
                 w + token * 512,64,1,1,0,32,8,1);
        } else {
          SetMaskCount();
          SetVectorMask<float,MaskMode::COUNTER>(0,nr * h);
          vmul(b,a,w,1,1,1,0,8,8,1);
          pipe_barrier(PIPE_V);
          SetMaskNorm();
          Mask64(~U64{0});
        }
      }
      auto columnSum=p;
      if (n == 4 && hf) {
        // Direct unrolled straight-line column binary tree (JP6: H=64,s=256)
        // Row scaling already wrote row/token-interleaved data.
        vadd(p,b,b + 32 * s,128,1,1,1,8,8,8);
        if constexpr (!P) pipe_barrier(PIPE_V);
        if (pk && b != a) {
          auto other=a + 64;
          vadd(other,p,p + 16 * s,64,1,1,1,8,8,8);
          vadd(p,other,other + 8 * s,32,1,1,1,8,8,8);
          vadd(other,p,p + 4 * s,16,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          vadd(p,other,other + 2 * s,8,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          vadd(other,p,p + s,4,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          columnSum=other;
        } else {
        vadd(p,p,p + 16 * s,64,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        vadd(p,p,p + 8 * s,32,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        vadd(p,p,p + 4 * s,16,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        vadd(p,p,p + 2 * s,8,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        vadd(p,p,p + 1 * s,4,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        }
      } else {
        U32 height=qr;
        if (hf) {
          vadd(p,b,b + half,half / lanes,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          height=hr;
        } else {
          vadd(p,b,b + q,q / lanes,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          vadd(p,p,b + 2 * q,q / lanes,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
          vadd(p,p,b + 3 * q,q / lanes,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
        }
        for (; height > 1; height /= 2) {
          vadd(p,p,p + height / 2 * s,
               height / 2 * s / lanes,1,1,1,8,8,8);
          pipe_barrier(PIPE_V);
        }
      }
      vadds(d,columnSum,eps,nv,1,1,8,8);
      pipe_barrier(PIPE_V);
      if constexpr (P) {
        // Contiguous tokens in the interleaved row a share one relative
        // coefficient phase. Write the division r in its final location.
        const U32 phase =
            ((reinterpret_cast<U64>(b) ^
              reinterpret_cast<U64>(p)) & 256u) ? 0 : 64;
        vdiv(p + phase,u,d,nv,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
        for (U32 token=0; token < 4; ++token)
          vmul(a + token * 4096,b + token * 64,
               p + phase + token * 64,
               64,1,1,1,8,32,0);
      } else {
        vdiv(d,u,d,nv,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
      }
      if constexpr (P) {
        // Scaling already consumed the compact coefficients above.
      } else if (n == 4) {
        // Direct unrolled column scaling vmul
        vmul(a + 0,b + 0,d + 0,64,1,1,1,32,32,0);
        vmul(a + 64,b + 64,d + 64,64,1,1,1,32,32,0);
        vmul(a + 128,b + 128,d + 128,64,1,1,1,32,32,0);
        vmul(a + 192,b + 192,d + 192,64,1,1,1,32,32,0);
      } else {
        for (U32 col=0; col < s; col += lanes) {
          vmul(a + col,b + col,d + col,h,1,1,1,st,st,0);
        }
      }
      if constexpr (!P) pipe_barrier(PIPE_V);
    }
  }
}

class Medium64 {
 public:
  __aicore__ inline void Init(GM_ADDR mixes,GM_ADDR pre,GM_ADDR post,
                              GM_ADDR comb,const sinkhorn_norm::TilingData &t) {
    t_=t;
    const U64 core=GetBlockIdx();
    start_=core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_=t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);

    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    const U32 rp=(t_.h * t_.tile_tokens + 7) / 8 * 8;
    const U32 s=t_.tile_tokens * t_.h_pad;
    const U32 u=rp > 2 * s ? rp : 2 * s;
    const U32 w=8 * rp > 6 * s ? 8 * rp : 6 * s;

    pipe_.InitBuffer(matBuf_,t_.matrix_bytes);
    pipe_.InitBuffer(gateBuf_,2 * t_.gate_bytes);
    pipe_.InitBuffer(sumBuf_,t_.input_bytes);
    pipe_.InitBuffer(tmpBuf_,w * sizeof(float));
    pipe_.InitBuffer(resBuf_,rp * sizeof(float));
    pipe_.InitBuffer(oneBuf_,u * sizeof(float));

    U32 pad=0;
    U32 reserve=0;
    const U32 ss=13 * s * sizeof(float) + 256;
    if (t_.scratch_bytes == ss + t_.matrix_bytes + 512)
      reserve=t_.matrix_bytes + 512;
    if (t_.scratch_bytes == ss || reserve)
      pad=256;

    pipe_.InitBuffer(denBuf_,s * sizeof(float) + pad +
                                       reserve);

    a_=matBuf_.Get<float>();
    g_=gateBuf_.Get<float>();
    p_=sumBuf_.Get<float>();
    w_=tmpBuf_.Get<float>();
    r_=resBuf_.Get<float>();
    u_=oneBuf_.Get<float>();
    d_=denBuf_.Get<float>();
    if (pad)
      d_=d_[pad / sizeof(float)];

    Duplicate(r_,0.0f,rp);
    Duplicate(u_,1.0f,u);

    sm_[0]=~U64{0};
    sm_[1]=0;
  }

  __aicore__ inline void Process() {
    for (U64 offset=0; offset < count_; offset += t_.tile_tokens) {
      valid_=count_ - offset < t_.tile_tokens ? count_ - offset : t_.tile_tokens;
      rows_=valid_ * t_.h;
      span_=valid_ * t_.h_pad;
      const bool prev=(offset > 0);
      const bool more=(offset + t_.tile_tokens < count_);

      pk_=valid_ == 4 && t_.input_bytes >= 32768 &&
          t_.scratch_bytes == 13 * t_.h_pad * t_.tile_tokens * sizeof(float) +
                              256 + t_.matrix_bytes + 512;
      Input(start_ + offset,prev,more);

      bool useAlignedAlternate=false;
      const U32 capacitySpan=t_.h_pad * t_.tile_tokens;
      const U32 ss=13 * capacitySpan * sizeof(float) + 256;
      if (t_.scratch_bytes == ss + t_.matrix_bytes + 512) {
        useAlignedAlternate=(span_ * sizeof(float)) % 512 == 0;
        d_=denBuf_.Get<float>();
        if (!useAlignedAlternate) d_=d_[256 / sizeof(float)];
      }

      auto b=reinterpret_cast<UBFloat *>(a_.GetPhyAddr());
      if (useAlignedAlternate) {
        const U32 end=t_.matrix_bytes + 2 * t_.gate_bytes +
                             t_.input_bytes + ss;
        const U32 padding=(256 + 512 - end % 512) % 512;
        b=reinterpret_cast<UBFloat *>(d_.GetPhyAddr()) +
                    capacitySpan + (256 + padding) / sizeof(float);
      }

      if (pk_ && b != reinterpret_cast<UBFloat *>(a_.GetPhyAddr())) {
      Rounds64<true>(
          reinterpret_cast<UBFloat *>(a_.GetPhyAddr()),b,
          reinterpret_cast<UBFloat *>(p_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(w_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(r_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(u_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(d_.GetPhyAddr()),
          valid_,t_.input_bytes / sizeof(float),
          t_.tile_tokens % 2 == 0 ? 64U : 0U,t_.eps,t_.iters);
      } else {
      Rounds64<false>(
          reinterpret_cast<UBFloat *>(a_.GetPhyAddr()),b,
          reinterpret_cast<UBFloat *>(p_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(w_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(r_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(u_.GetPhyAddr()),
          reinterpret_cast<UBFloat *>(d_.GetPhyAddr()),
          valid_,t_.input_bytes / sizeof(float),
          t_.tile_tokens % 2 == 0 ? 64U : 0U,t_.eps,t_.iters);
      }

      PipeBarrier<PIPE_V>();
      ResetMask();

      Output(start_ + offset,more);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void Input(U64 first,bool prev,bool more) {
    const U32 st=(valid_ - 1) * t_.h_pad / 8;
    const DataCopyPadExtParams<float> noPad{false,0,0,0.0f};
    const DataCopyExtParams g{static_cast<uint16_t>(valid_),
                              4 * t_.h,4 * (t_.input_width - t_.h),0,0};
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),4 * t_.h,0,st,0};
    const U64 base=first * t_.input_width;
    if (prev) {
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    DataCopyPad(g_,input_[base],g,noPad);
    DataCopyPad(g_[span_],input_[base + t_.h],g,noPad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (prev) {
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (pk_) {
      const DataCopyExtParams dense{static_cast<uint16_t>(valid_),
          4 * t_.h * t_.h,4 * (t_.input_width - t_.h * t_.h),0,0};
      DataCopyPad(a_,input_[base + 2 * t_.h],dense,noPad);
    } else {
    for (U32 token=0; token < valid_; ++token) {
      DataCopyPad(a_[token * t_.h_pad],
                  input_[base + token * t_.input_width + 2 * t_.h],m,noPad);
    }
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    // Wait only for g_ and execute Gates() on PIPE_V while a_ MTE2 DMA transfers!
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gates();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

    // Issue contiguous pre and post MTE3 DMA writebacks while a_ MTE2 and Sinkhorn rounds execute!
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    const DataCopyExtParams gOut{1,4 * span_,0,0,0};
    DataCopyPad(pre_[first * t_.h],g_,gOut);
    DataCopyPad(post_[first * t_.h],g_[span_],gOut);
    if (more) {
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
  }

  __aicore__ inline void Gates() {
    const U32 n=2 * span_;
    Muls(w_,g_,-1.0f,n);
    PipeBarrier<PIPE_V>();
    Mins(w_,w_,80.0f,n);
    PipeBarrier<PIPE_V>();
    Exp(w_,w_,n);
    PipeBarrier<PIPE_V>();
    Adds(w_,w_,1.0f,n);
    PipeBarrier<PIPE_V>();
    Div(g_,u_,w_,n);
    PipeBarrier<PIPE_V>();
    Adds(g_,g_,t_.eps,span_);
    Muls(g_[span_],g_[span_],2.0f,span_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Output(U64 first,bool more) {
    Sync<HardEvent::V_MTE3>();
    const U32 st=(valid_ - 1) * t_.h_pad / 8;
    const DataCopyExtParams m{static_cast<uint16_t>(t_.h),4 * t_.h,st,0,0};
    if (pk_) {
      const DataCopyExtParams dense{1,4 * valid_ * t_.h * t_.h,0,0,0};
      DataCopyPad(comb_[first * t_.h * t_.h],a_,dense);
    } else {
    for (U32 token=0; token < valid_; ++token) {
      const U64 n=first + token;
      DataCopyPad(comb_[n * t_.h * t_.h],a_[token * t_.h_pad],m);
    }
    }
    if (more) {
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matBuf_,gateBuf_;
  TBuf<TPosition::VECCALC> sumBuf_,tmpBuf_,resBuf_,
      oneBuf_,denBuf_;
  LocalTensor<float> a_,g_,p_,w_,r_,u_,d_;
  GlobalTensor<float> input_,pre_,post_,comb_;
  sinkhorn_norm::TilingData t_;
  U64 start_,count_,sm_[2];
  U32 valid_,rows_,span_;
  bool pk_;
};

}  // namespace f11
}  // namespace sinkhorn_norm

template <int=0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f11_kernel(
    GM_ADDR mixes,GM_ADDR pre,GM_ADDR post,GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f11::Medium64 op;
  op.Init(mixes,pre,post,comb,tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F11_MEDIUM_ALIGNED_64_H_
