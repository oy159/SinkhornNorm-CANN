#ifndef SINKHORN_NORM_KERNELS_F09_MEDIUM_DENSE_IO_H_
#define SINKHORN_NORM_KERNELS_F09_MEDIUM_DENSE_IO_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_f09 {

using namespace AscendC;
using sinkhorn_norm::TilingData;

// Keep vector mask ownership inside the complete medium round. The input
// address map is dead here and is rebuilt before output. Reuse it for a
// half-matrix column tree without changing tile capacity or UB allocation.
__aicore__ __attribute__((always_inline)) inline void RunMediumCompleteRounds(
    __ubuf__ float *matrix,__ubuf__ float *partial,__ubuf__ float *tmp,
    __ubuf__ float *result,__ubuf__ float *ones,__ubuf__ float *denom,
    __ubuf__ float *alternate,__ubuf__ float *columnScratch,uint32_t h,
    uint32_t pitch,uint32_t valid,float eps,uint32_t rounds) {
  if ASCEND_IS_AIV {
    auto original=matrix;
    auto spare=alternate;
    const uint32_t rows=(h*valid+7) / 8*8;
    const uint32_t span=pitch*valid;
    const uint32_t blocks=pitch / 8;
    const uint32_t groups=rows / 8;
    const uint32_t halfRows=h / 2;
    const uint32_t upperRows=h - halfRows;
    const uint32_t lastWidth=h - (blocks - 1)*8;
    const uint64_t lastMask =
        ((uint64_t{1} << lastWidth) - 1)*uint64_t{0x0101010101010101};
    SetMaskNorm();
    if (rounds != 0 && lastWidth != 8) {
      SetVectorMask<float,MaskMode::NORMAL>(0,~lastMask);
      vsub(matrix+(blocks - 1)*8,ones,ones,groups,blocks,0,0,pitch,
           0,0);
      pipe_barrier(PIPE_V);
    }
    for (uint32_t round=0; round < rounds; ++round) {
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      if (blocks == 5) {
        vcgadd(result,matrix+0*8,groups,1,5,pitch);
        vcgadd(partial+0*rows,matrix+1*8,groups,1,5,pitch);
        vcgadd(partial+1*rows,matrix+2*8,groups,1,5,pitch);
        vcgadd(partial+2*rows,matrix+3*8,groups,1,5,pitch);
        vcgadd(partial+3*rows,matrix+4*8,groups,1,5,pitch);
      } else {
        for (uint32_t block=0; block < blocks; ++block) {
          auto dst=block == 0 ? result : partial+(block - 1)*rows;
          vcgadd(dst,matrix+block*8,groups,1,blocks,pitch);
        }
      }
      pipe_barrier(PIPE_V);
      if (rows <= 256) {
        SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
        vadd(result+0,partial+0,result+0,blocks - 1,1,1,1,0,
             rows / 8,0);
        vadd(result+64,partial+64,result+64,blocks - 1,1,1,1,0,
             rows / 8,0);
        vadd(result+128,partial+128,result+128,blocks - 1,1,1,1,0,
             rows / 8,0);
        const uint32_t lastChunk=rows - 192;
        SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t{1} << lastChunk) - 1);
        vadd(result+192,partial+192,result+192,blocks - 1,1,1,1,0,
             rows / 8,0);
      } else {
        for (uint32_t first=0; first < rows; first += 64) {
          const uint32_t width=rows - first < 64 ? rows - first : 64;
          const uint64_t mask =
              width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
          SetVectorMask<float,MaskMode::NORMAL>(0,mask);
          vadd(result+first,partial+first,result+first,blocks - 1,1,1,
               1,0,rows / 8,0);
        }
      }
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,rows);
      vadds(result,result,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      vdiv(result,ones,result,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(tmp),
            reinterpret_cast<__ubuf__ uint32_t *>(result),1,8,groups);
      pipe_barrier(PIPE_V);
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      if (blocks == 5) {
        vmul(matrix+0*8,matrix+0*8,tmp,groups,5,5,1,pitch,pitch,8);
        vmul(matrix+1*8,matrix+1*8,tmp,groups,5,5,1,pitch,pitch,8);
        vmul(matrix+2*8,matrix+2*8,tmp,groups,5,5,1,pitch,pitch,8);
        vmul(matrix+3*8,matrix+3*8,tmp,groups,5,5,1,pitch,pitch,8);
        vmul(matrix+4*8,matrix+4*8,tmp,groups,5,5,1,pitch,pitch,8);
      } else {
        for (uint32_t block=0; block < blocks; ++block) {
          auto m=matrix+block*8;
          vmul(m,m,tmp,groups,blocks,blocks,1,pitch,pitch,8);
        }
      }
      pipe_barrier(PIPE_V);

      SetMaskCount();
      if (h == 34) {
        auto bufA=columnScratch;
        auto bufB=columnScratch+18*span;

        // Step 1: 34 -> 17 rows in bufA
        SetVectorMask<float,MaskMode::COUNTER>(0,17*span);
        vadd(bufA,matrix,matrix+17*span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);

        // Step 2: 17 -> 8 rows in bufB: bufA[0..7]+bufA[8..15]
        SetVectorMask<float,MaskMode::COUNTER>(0,8*span);
        vadd(bufB,bufA,bufA+8*span,1,1,1,1,8,8,8);

        // Step 3: Add leftover row bufA[16] into bufB[0]
        SetVectorMask<float,MaskMode::COUNTER>(0,span);
        vadd(bufB,bufB,bufA+16*span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);

        // Step 4: 8 -> 4 rows in bufA: bufB[0..3]+bufB[4..7]
        SetVectorMask<float,MaskMode::COUNTER>(0,4*span);
        vadd(bufA,bufB,bufB+4*span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);

        // Step 5: 4 -> 2 rows in bufB: bufA[0..1]+bufA[2..3]
        SetVectorMask<float,MaskMode::COUNTER>(0,2*span);
        vadd(bufB,bufA,bufA+2*span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);

        // Step 6: 2 -> 1 row in columnScratch: bufB[0]+bufB[1]
        SetVectorMask<float,MaskMode::COUNTER>(0,span);
        vadd(columnScratch,bufB,bufB+span,1,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
      } else {
        SetVectorMask<float,MaskMode::COUNTER>(0,halfRows*span);
        vadd(columnScratch,matrix,matrix+upperRows*span,1,1,1,1,8,8,
             8);
        if (h != 2*halfRows) {
          SetVectorMask<float,MaskMode::COUNTER>(0,span);
          vcopy(reinterpret_cast<__ubuf__ uint32_t *>(columnScratch +
                                                      halfRows*span),
                reinterpret_cast<__ubuf__ uint32_t *>(matrix+halfRows*span),
                1,1,1,8,8);
        }
        pipe_barrier(PIPE_V);
        for (uint32_t height=upperRows; height > 1;) {
          const uint32_t half=height / 2;
          const uint32_t upper=height - half;
          SetVectorMask<float,MaskMode::COUNTER>(0,half*span);
          vadd(columnScratch,columnScratch,columnScratch+upper*span,1,1,
               1,1,8,8,8);
          pipe_barrier(PIPE_V);
          height=upper;
        }
      }
      SetVectorMask<float,MaskMode::COUNTER>(0,span);
      vadds(denom,columnScratch,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      vdiv(denom,ones,denom,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      auto output=round+1 == rounds ? original : alternate;
      if (h == 34) {
        vmul(output+0,matrix+0,denom+0,34,1,1,1,span / 8,span / 8,0);
        vmul(output+64,matrix+64,denom+64,34,1,1,1,span / 8,span / 8,0);
        vmul(output+128,matrix+128,denom+128,34,1,1,1,span / 8,span / 8,0);
        if (span == 240) {
          SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t{1} << 48) - 1);
          vmul(output+192,matrix+192,denom+192,34,1,1,1,span / 8,span / 8,0);
        } else if (span == 280) {
          vmul(output+192,matrix+192,denom+192,34,1,1,1,span / 8,span / 8,0);
          SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t{1} << 24) - 1);
          vmul(output+256,matrix+256,denom+256,34,1,1,1,span / 8,span / 8,0);
        } else {
          for (uint32_t col=192; col < span; col += 64) {
            const uint32_t width=span - col < 64 ? span - col : 64;
            if (width != 64)
              SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t{1} << width) - 1);
            vmul(output+col,matrix+col,denom+col,34,1,1,1,span / 8,
                 span / 8,0);
          }
        }
      } else {
        for (uint32_t col=0; col < span; col += 64) {
          const uint32_t width=span - col < 64 ? span - col : 64;
          if (width != 64)
            SetVectorMask<float,MaskMode::NORMAL>(0,(uint64_t{1} << width) - 1);
          vmul(output+col,matrix+col,denom+col,h,1,1,1,span / 8,
               span / 8,0);
        }
      }
      pipe_barrier(PIPE_V);
      matrix=output;
      alternate=output == original ? spare : original;
    }
    SetMaskNorm();
    ResetMask();
  }
}

// Exact-division factor recurrence on a fixed,padded,blocked K. K is the
// already completed first round; each remaining row and column step keeps eps.
__aicore__ __attribute__((always_inline)) inline void RunBlockedFactors(
    __ubuf__ float *k,__ubuf__ float *scratch,__ubuf__ float *sum,
    __ubuf__ float *a,__ubuf__ float *b,__ubuf__ float *ones,
    __ubuf__ float *coeffZero,__ubuf__ float *coeffHalf,
    uint32_t h,uint32_t pitch,float eps,uint32_t rounds,bool seededColumn=false,uint32_t outputTokens=0) {
  if ASCEND_IS_AIV {
    const uint32_t rows=8*h,plane=64*h,blocks=pitch / 8;
    const uint32_t span=64*blocks;
    const bool uniformPhase=plane % 128 == 0;
    auto end=coeffHalf+plane;
    auto vbase=end+(((0u - static_cast<uint32_t>(reinterpret_cast<uint64_t>(end))) & 511u) / 4);
    auto acc=scratch+(((128u - static_cast<uint32_t>(reinterpret_cast<uint64_t>(scratch))) & 511u) / 4);
    SetMaskNorm();
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
    vector_dup(a,1.0f,(rows+63) / 64,1,1,8,0);
    if (seededColumn)
      vcopy(reinterpret_cast<__ubuf__ uint32_t *>(b),reinterpret_cast<__ubuf__ uint32_t *>(sum),blocks,1,1,8,8);
    else vector_dup(b,1.0f,blocks,1,1,8,0);
    vector_dup(vbase,0.0f,2*blocks,1,1,8,0);
    pipe_barrier(PIPE_V);
    const uint32_t width=h - (blocks - 1)*8;
    const uint64_t liveMask=((uint64_t{1} << width) - 1)*uint64_t{0x0101010101010101};
    for (uint32_t round=0; round < rounds; ++round) {
      if (uniformPhase) {
        auto v=vbase+((reinterpret_cast<uint64_t>(k) & 256u) ? 0 : 64);
        SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
        vdiv(v,ones,b,blocks - 1,1,1,1,16,0,8);
        SetVectorMask<float,MaskMode::NORMAL>(0,liveMask);
        vdiv(v+(blocks - 1)*128,ones,b+(blocks - 1)*64,1,1,1,1,8,8,8);
      } else {
      for (uint32_t q=0; q < blocks; ++q) {
        auto m=k+q*plane;
        auto v=vbase+q*128+((reinterpret_cast<uint64_t>(m) & 256u) ? 0 : 64);
        SetVectorMask<float,MaskMode::NORMAL>(0,q+1 == blocks ? liveMask : ~uint64_t{0});
        vdiv(v,ones,b+q*64,1,1,1,1,8,8,8);
      }
      }
      pipe_barrier(PIPE_V);
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      for (uint32_t q=0; q < blocks; ++q) {
        auto m=k+q*plane;
        auto v=vbase+q*128+((reinterpret_cast<uint64_t>(m) & 256u) ? 0 : 64);
        if (q == 0) vmul(acc,m,v,h,1,1,1,8,8,0);
        else vmla(acc,m,v,h,1,1,1,8,8,0);
      }
      pipe_barrier(PIPE_V);
      vcgadd(sum,acc,h,1,1,8);
      pipe_barrier(PIPE_V);
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,rows);
      vaxpy(sum,a,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      auto previousA=a;
      a=sum;
      sum=previousA;
      vdiv(sum,ones,a,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      if (uniformPhase) {
        auto coeff=(reinterpret_cast<uint64_t>(k) & 256u) ? coeffZero : coeffHalf;
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeff),reinterpret_cast<__ubuf__ uint32_t *>(sum),1,8,h);
      } else {
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeffZero),reinterpret_cast<__ubuf__ uint32_t *>(sum),1,8,h);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeffHalf),reinterpret_cast<__ubuf__ uint32_t *>(sum),1,8,h);
      }
      pipe_barrier(PIPE_V);
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      // Four independent row accumulators per plane leave one less
      // tree level before the packed denominator.
      for (uint32_t row=0; row < h; row += 4) {
        const uint32_t n=h - row < 4 ? h - row : 4;
        for (uint32_t q=0; q < blocks; ++q) {
          auto m=k+q*plane;
          auto coeff=(reinterpret_cast<uint64_t>(m) & 256u) ? coeffZero : coeffHalf;
          if (row == 0) vmul(acc+q*256,m,coeff,n,1,1,1,8,8,8);
          else vmla(acc+q*256,m+row*64,coeff+row*64,n,1,1,1,8,8,8);
        }
      }
      pipe_barrier(PIPE_V);
      for (uint32_t q=0; q < blocks; ++q) {
        auto c=acc+q*256;
        vadd(c,c,c+128,2,1,1,1,8,8,8);
      }
      pipe_barrier(PIPE_V);
      vadd(sum,acc,acc+64,blocks,1,1,1,8,32,32);
      pipe_barrier(PIPE_V);
      vaxpy(sum,b,eps,blocks,1,1,8,8);
      pipe_barrier(PIPE_V);
      auto previousB=b;
      b=sum;
      sum=previousB;
    }
    // coeffZero/coeffHalf still contain the final inverse row factor.
    for (uint32_t q=0; q < blocks; ++q) {
      auto m=k+q*plane;
      auto v=vbase+q*128+((reinterpret_cast<uint64_t>(m) & 256u) ? 0 : 64);
      SetVectorMask<float,MaskMode::NORMAL>(0,q+1 == blocks ? liveMask : ~uint64_t{0});
      vdiv(v,ones,b+q*64,1,1,1,1,8,8,8);
    }
    pipe_barrier(PIPE_V);
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
    if (outputTokens)
      SetVectorMask<float,MaskMode::NORMAL>(0,outputTokens==8?~uint64_t{0}:(uint64_t{1}<<(8*outputTokens))-1);
    for (uint32_t q=0; q < blocks; ++q) {
      auto m=k+q*plane;
      auto coeff=(reinterpret_cast<uint64_t>(m) & 256u) ? coeffZero : coeffHalf;
      if (outputTokens)
        vmul(scratch+q*8,m,coeff,h,blocks,1,1,outputTokens*blocks,8,8);
      else vmul(m,m,coeff,h,1,1,1,8,8,8);
    }
    pipe_barrier(PIPE_V);
    for (uint32_t q=0; q < blocks; ++q) {
      auto m=k+q*plane;
      auto v=vbase+q*128+((reinterpret_cast<uint64_t>(m) & 256u) ? 0 : 64);
      if (outputTokens)
        vmul(scratch+q*8,scratch+q*8,v,h,blocks,blocks,1,outputTokens*blocks,outputTokens*blocks,0);
      else vmul(m,m,v,h,1,1,1,8,8,0);
    }
    pipe_barrier(PIPE_V);
  }
}

// Reuse the retired dense staging/index arrays for blocked matrix storage and
// phased row coefficients. GM packing,the first round and total UB stay fixed.
template <bool FullVectors=false,bool Factors=false,bool DeferredColumn=false>
__aicore__ __attribute__((always_inline)) inline void RunMediumBlockedRounds(
    __ubuf__ float *original,__ubuf__ float *blocked,__ubuf__ float *partial,
    __ubuf__ float *result,__ubuf__ float *ones,__ubuf__ float *denom,
    __ubuf__ float *coeffStorage,uint32_t h,uint32_t pitch,uint32_t valid,
    float eps,uint32_t rounds) {
  if ASCEND_IS_AIV {
    if (rounds == 0) return;
    const uint32_t storedValid=FullVectors ? 8 : valid;
    const uint32_t originalRows=(h*valid+7) / 8*8;
    const uint32_t rows=(h*storedValid+7) / 8*8;
    const uint32_t groups=rows / 8,blocks=pitch / 8;
    const uint32_t plane=rows*8,tokenSpan=storedValid*8;
    const uint32_t halfRows=h / 2,upperRows=h - halfRows;
    const uint32_t columnPlane=upperRows*tokenSpan;
    const uint32_t span=pitch*storedValid;
    const uint32_t width=h - (blocks - 1)*8;
    SetMaskNorm();
    if (width != 8) {
      const uint64_t mask=((uint64_t{1} << width) - 1)*uint64_t{0x0101010101010101};
      SetVectorMask<float,MaskMode::NORMAL>(0,~mask);
      vsub(original+(blocks - 1)*8,ones,ones,originalRows / 8,blocks,0,0,pitch,0,0);
      pipe_barrier(PIPE_V);
    }
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
    if constexpr (FullVectors) {
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,blocks*plane);
      vector_dup(blocked,0.0f,1,1,1,8,0);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      SetVectorMask<float,MaskMode::NORMAL>(0,valid == 8 ? ~uint64_t{0} : (uint64_t{1} << (8*valid)) - 1);
      for (uint32_t b=0; b < blocks; ++b)
        vcopy(reinterpret_cast<__ubuf__ uint32_t *>(blocked+b*plane),
              reinterpret_cast<__ubuf__ uint32_t *>(original+b*8),
              h,1,blocks,storedValid,valid*blocks);
    } else {
    for (uint32_t b=0; b < blocks; ++b) {
      vcopy(reinterpret_cast<__ubuf__ uint32_t *>(blocked+b*plane),
            reinterpret_cast<__ubuf__ uint32_t *>(original+b*8),
            groups,1,blocks,8,pitch);
    }
    }
    pipe_barrier(PIPE_V);
    const uint32_t prefix=((0u - static_cast<uint32_t>(reinterpret_cast<uint64_t>(coeffStorage))) & 511u) / 4;
    auto coeffZero=coeffStorage+prefix;
    auto coeffHalf=coeffZero+plane+(plane % 128 == 0 ? 64 : 0);
    if constexpr (Factors) {
      if constexpr (DeferredColumn) {
        // The first column normalization is represented by b rather than
        // written into K. Repack its real denominators before partial reuse.
        SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
        vector_dup(partial,1.0f,blocks,1,1,8,0);
        pipe_barrier(PIPE_V);
        SetVectorMask<float,MaskMode::NORMAL>(0,valid==8?~uint64_t{0}:(uint64_t{1}<<(8*valid))-1);
        for (uint32_t q=0;q<blocks;++q)
          vcopy(reinterpret_cast<__ubuf__ uint32_t *>(partial+q*64),
                reinterpret_cast<__ubuf__ uint32_t *>(denom+q*8),1,1,blocks,8,pitch);
        pipe_barrier(PIPE_V);
      }
      RunBlockedFactors(blocked,original,partial,result,denom,ones,
                        coeffZero,coeffHalf,h,pitch,eps,rounds,DeferredColumn,FullVectors?valid:0);
    } else {
    for (uint32_t round=0; round < rounds; ++round) {
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      for (uint32_t b=0; b < blocks; ++b) {
        auto dst=b == 0 ? result : partial+(b - 1)*rows;
        vcgadd(dst,blocked+b*plane,groups,1,1,8);
      }
      pipe_barrier(PIPE_V);
      for (uint32_t first=0; first < rows; first += 64) {
        const uint32_t n=rows - first < 64 ? rows - first : 64;
        SetVectorMask<float,MaskMode::NORMAL>(0,n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1);
        vadd(result+first,partial+first,result+first,blocks - 1,
             1,1,1,0,rows / 8,0);
      }
      pipe_barrier(PIPE_V);
      if constexpr (FullVectors) {
        SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
        vadds(result,result,eps,(rows+63) / 64,1,1,8,8);
        pipe_barrier(PIPE_V);
        vdiv(result,ones,result,(rows+63) / 64,1,1,1,8,8,8);
        pipe_barrier(PIPE_V);
      } else {
      SetMaskCount();
      SetVectorMask<float,MaskMode::COUNTER>(0,rows);
      vadds(result,result,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      vdiv(result,ones,result,1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      SetMaskNorm();
      }
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeffZero),
            reinterpret_cast<__ubuf__ uint32_t *>(result),1,8,groups);
      vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(coeffHalf),
            reinterpret_cast<__ubuf__ uint32_t *>(result),1,8,groups);
      pipe_barrier(PIPE_V);
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
      for (uint32_t b=0; b < blocks; ++b) {
        auto m=blocked+b*plane;
        auto coeff=(reinterpret_cast<uint64_t>(m) & 256u) ? coeffZero : coeffHalf;
        vmul(m,m,coeff,groups,1,1,1,8,8,8);
      }
      pipe_barrier(PIPE_V);
      if constexpr (!FullVectors) {
        SetMaskCount();
        SetVectorMask<float,MaskMode::COUNTER>(0,halfRows*tokenSpan);
      }
      for (uint32_t b=0; b < blocks; ++b) {
        auto m=blocked+b*plane;
        vadd(original+b*columnPlane,m,m+upperRows*tokenSpan,FullVectors ? halfRows : 1,1,1,1,8,8,8);
      }
      if (h != 2*halfRows) {
        if constexpr (!FullVectors) SetVectorMask<float,MaskMode::COUNTER>(0,tokenSpan);
        for (uint32_t b=0; b < blocks; ++b) {
          vcopy(reinterpret_cast<__ubuf__ uint32_t *>(original+b*columnPlane+halfRows*tokenSpan),
                reinterpret_cast<__ubuf__ uint32_t *>(blocked+b*plane+halfRows*tokenSpan),1,1,1,8,8);
        }
      }
      pipe_barrier(PIPE_V);
      for (uint32_t height=upperRows; height > 1;) {
        const uint32_t half=height / 2,upper=height - half;
        if constexpr (!FullVectors) SetVectorMask<float,MaskMode::COUNTER>(0,half*tokenSpan);
        for (uint32_t b=0; b < blocks; ++b) {
          auto c=original+b*columnPlane;
          vadd(c,c,c+upper*tokenSpan,FullVectors ? half : 1,1,1,1,8,8,8);
        }
        pipe_barrier(PIPE_V);
        height=upper;
      }
      if constexpr (!FullVectors) SetVectorMask<float,MaskMode::COUNTER>(0,tokenSpan);
      for (uint32_t b=0; b < blocks; ++b)
        vadds(denom+b*tokenSpan,original+b*columnPlane,eps,1,1,1,8,8);
      pipe_barrier(PIPE_V);
      if constexpr (!FullVectors) SetVectorMask<float,MaskMode::COUNTER>(0,span);
      vdiv(denom,ones,denom,FullVectors ? span / 64 : 1,1,1,1,8,8,8);
      pipe_barrier(PIPE_V);
      if constexpr (!FullVectors) SetMaskNorm();
      for (uint32_t col=0; col < tokenSpan; col += 64) {
        const uint32_t n=tokenSpan - col < 64 ? tokenSpan - col : 64;
        SetVectorMask<float,MaskMode::NORMAL>(0,n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1);
        for (uint32_t b=0; b < blocks; ++b) {
          auto m=blocked+b*plane+col;
          vmul(m,m,denom+b*tokenSpan+col,h,1,1,1,storedValid,storedValid,0);
        }
      }
      pipe_barrier(PIPE_V);
    }
    }
    if constexpr (FullVectors && Factors) {
      SetMaskNorm();
      ResetMask();
      return;
    }
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t{0});
    if constexpr (FullVectors) {
      SetVectorMask<float,MaskMode::NORMAL>(0,valid == 8 ? ~uint64_t{0} : (uint64_t{1} << (8*valid)) - 1);
      for (uint32_t b=0; b < blocks; ++b)
        vcopy(reinterpret_cast<__ubuf__ uint32_t *>(original+b*8),
              reinterpret_cast<__ubuf__ uint32_t *>(blocked+b*plane),
              h,blocks,1,valid*blocks,storedValid);
    } else {
    for (uint32_t b=0; b < blocks; ++b) {
      vcopy(reinterpret_cast<__ubuf__ uint32_t *>(original+b*8),
            reinterpret_cast<__ubuf__ uint32_t *>(blocked+b*plane),
            groups,blocks,1,pitch,8);
    }
    }
    pipe_barrier(PIPE_V);
    ResetMask();
  }
}

class MediumDenseIoKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes,GM_ADDR pre,GM_ADDR post,
                              GM_ADDR comb,const TilingData &t) {
    t_=t;
    const uint64_t core=GetBlockIdx();
    start_=core*t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_=t_.tokens_per_core+(core < t_.extra_core_count ? 1 : 0);
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    const uint32_t rp=(t_.h*t_.tile_tokens+7) / 8*8;
    const uint32_t span=t_.tile_tokens*t_.h_pad;
    const uint32_t ones=rp > 2*span ? rp : 2*span;
    const uint32_t tmp=8*rp > 6*span ? 8*rp : 6*span;

    pipe_.InitBuffer(matrixBuffer_,t_.matrix_bytes);
    pipe_.InitBuffer(gateBuffer_,2*t_.gate_bytes);
    pipe_.InitBuffer(partialBuffer_,t_.input_bytes);
    pipe_.InitBuffer(tmpBuffer_,tmp*sizeof(float));
    const uint32_t paddedRows=(t_.h*8+63) / 64*64;
    const uint32_t paddedSpan=t_.h_pad*8;
    const uint32_t extra=paddedRows > rp && paddedSpan >= span
        ? paddedRows - rp+paddedSpan - span : rp+1;
    // At least 3/4 occupancy of one 64-lane token vector; use only existing
    // work capacity,and verify the pooled matrix/coefficients at the call.
    // Uniform 512-byte block phase avoids alternating coefficient banks.
    // Eight remaining rounds amortize packing/materialization,with a 2x
    // margin over the repeat-count break-even estimate. No data-based exit.
    paddedBlocked_=(t_.h*64*sizeof(float)) % 512 == 0 &&
                     t_.iters >= 9 && t_.eps >= 0x1p-40f && t_.eps <= 1.0f &&
                     t_.tile_tokens*8 >= 48 && t_.tile_tokens <= 8 &&
                     extra <= rp && paddedRows <= ones && paddedSpan <= ones;
    pipe_.InitBuffer(resultBuffer_,(paddedBlocked_ ? paddedRows : rp)*sizeof(float));
    if (!paddedBlocked_ || rp > extra)
      pipe_.InitBuffer(workBuffer_,(paddedBlocked_ ? rp - extra : rp)*sizeof(float));
    pipe_.InitBuffer(onesBuffer_,ones*sizeof(float));
    pipe_.InitBuffer(denomBuffer_,(paddedBlocked_ ? paddedSpan : span)*sizeof(float));

    const uint32_t physical=(t_.h+2)*t_.h_pad*t_.tile_tokens;
    const uint32_t gate=(t_.h*t_.tile_tokens+63) / 64*64;
    const uint32_t packed=2*gate+t_.h*t_.h*t_.tile_tokens;
    const uint32_t elements =
        ((physical > packed ? physical : packed)+63) / 64*64;
    medium_stage_capacity_=elements;
    pipe_.InitBuffer(mediumStageBuffer_,8*elements);
    medium_stage_=mediumStageBuffer_.Get<float>();
    medium_index_=mediumStageBuffer_.Get<int32_t>()[elements];
    medium_indexed_valid_=0;

    matrix_=matrixBuffer_.Get<float>();
    gates_=gateBuffer_.Get<float>();
    partial_=partialBuffer_.Get<float>();
    tmp_=tmpBuffer_.Get<float>();
    result_=resultBuffer_.Get<float>();
    ones_=onesBuffer_.Get<float>();
    denom_=denomBuffer_.Get<float>();
    Duplicate(result_,0.0f,paddedBlocked_ ? paddedRows : rp);
    Duplicate(ones_,1.0f,ones);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Process() {
    for (uint64_t offset=0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t left=count_ - offset;
      valid_=left < t_.tile_tokens ? static_cast<uint32_t>(left)
                                     : t_.tile_tokens;
      rows_=t_.h*valid_;
      span_=t_.h_pad*valid_;
      CopyInMedium(start_+offset);
      const uint32_t paddedRows=(rows_+7) / 8*8;
      if (paddedRows != rows_) {
        Duplicate(matrix_[rows_*t_.h_pad],0.0f,
                  (paddedRows - rows_)*t_.h_pad);
      }
      rows_=paddedRows;
      Gates();
      Softmax();
      const uint32_t paddedMatrix=t_.h*8*t_.h_pad;
      const uint32_t paddedCoefficients=16*t_.h*8+512+128*(t_.h_pad/8);
      const bool useFactors=paddedBlocked_ && valid_*8>=48 && valid_<=8 &&
          t_.eps>=0x1p-40f && t_.eps<=1.0f &&
          paddedMatrix+paddedCoefficients<=2*medium_stage_capacity_ &&
          ((t_.h+1)/2)*8*t_.h_pad*sizeof(float)<=t_.matrix_bytes;
      if (useFactors) Columns<false>();
      else Columns<true>();
      auto alternate=matrix_;
      const uint32_t rowCount=(t_.h*valid_+7) / 8*8;
      if (rowCount*t_.h_pad <= medium_stage_capacity_) {
        alternate=medium_stage_;
        if (rowCount > t_.h*valid_) {
          Duplicate(alternate[t_.h*span_],0.0f,
                    (rowCount - t_.h*valid_)*t_.h_pad);
          PipeBarrier<PIPE_V>();
        }
      }
      if (useFactors) {
        auto pool=reinterpret_cast<__ubuf__ float *>(medium_stage_.GetPhyAddr());
        RunMediumBlockedRounds<true,true,true>(
            reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),pool,
            reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
            pool+paddedMatrix,t_.h,t_.h_pad,valid_,t_.eps,t_.iters - 1);
      } else {
      RunMediumCompleteRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(partial_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(result_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(alternate.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(medium_index_.GetPhyAddr()),t_.h,
          t_.h_pad,valid_,t_.eps,t_.iters - 1);
      }
      PipeBarrier<PIPE_V>();
      CopyOutMedium(start_+offset);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    const auto id=pipe_.FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  template <typename T>
  __aicore__ inline void ZeroMediumPadding(LocalTensor<T> data,uint32_t rows) {
    const uint32_t live=t_.h % 8;
    if (live == 0) return;
    uint64_t mask[2]={uint64_t{0xff} ^ ((uint64_t{1} << live) - 1),0};
    for (uint32_t first=0; first < rows; first += 248) {
      const uint32_t n=rows - first < 248 ? rows - first : 248;
      Duplicate(data[first*t_.h_pad+t_.h_pad - 8],T{0},mask,
                static_cast<uint8_t>(n),uint16_t{1},
                static_cast<uint8_t>(t_.h_pad / 8));
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void BuildMediumInputMap() {
    auto index=medium_index_;
    CreateVecIndex(index,int32_t{0},t_.h_pad);
    PipeBarrier<PIPE_V>();
    ZeroMediumPadding(index,1);
    Muls(index,index,int32_t{4},t_.h_pad);
    PipeBarrier<PIPE_V>();
    for (uint32_t token=1; token < valid_; ++token)
      Adds(index[token*t_.h_pad],index,
           static_cast<int32_t>(4*token*t_.input_width),t_.h_pad);
    PipeBarrier<PIPE_V>();
    for (uint32_t planes=1; planes < t_.h+2; planes *= 2) {
      const uint32_t n =
          t_.h+2 - planes < planes ? t_.h+2 - planes : planes;
      Adds(index[planes*span_],index,
           static_cast<int32_t>(4*planes*t_.h),n*span_);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyInMedium(uint64_t first) {
    if (medium_indexed_valid_ != valid_) {
      BuildMediumInputMap();
      medium_indexed_valid_=valid_;
    }
    const DataCopyExtParams copy{1,4*valid_*t_.input_width,0,0,0};
    const DataCopyPadExtParams<float> noPad{false,0,0,0.0f};
    DataCopyPad(medium_stage_,input_[first*t_.input_width],copy,noPad);
    Sync<HardEvent::MTE2_V>();
    auto index=medium_index_.ReinterpretCast<uint32_t>();
    Gather(gates_,medium_stage_,index,uint32_t{0},2*span_);
    Gather(matrix_,medium_stage_,index[2*span_],uint32_t{0},
           t_.h*span_);
    PipeBarrier<PIPE_V>();
    ZeroMediumPadding(gates_,2*valid_);
    ZeroMediumPadding(matrix_,t_.h*valid_);
  }

  __aicore__ inline void Gates() {
    const uint32_t n=2*span_;
    auto numerator=ones_;
    auto denominator=tmp_[2*n];
    Muls(tmp_,gates_,-1.0f,n);
    Mins(tmp_,tmp_,80.0f,n);
    Exp(tmp_,tmp_,n);
    // Independent row maxima can execute while the gate exponential settles.
    GroupedRowMaxReduce();
    Adds(denominator,tmp_,1.0f,n);
    Div(gates_,numerator,denominator,n);
    Adds(gates_,gates_,t_.eps,span_);
    Muls(gates_[span_],gates_[span_],2.0f,span_);
  }

  __aicore__ inline void GroupedRowMaxReduce() {
    const uint32_t blocks=t_.h_pad / 8;
    const uint32_t rp=(rows_+7) / 8*8;
    const uint32_t groups=rows_ / 8;
    for (uint32_t block=0; block < blocks; ++block) {
      const uint32_t col=block*8;
      const uint32_t width=t_.h - col < 8 ? t_.h - col : 8;
      uint64_t mask[2]={
          ((uint64_t{1} << width) - 1)*uint64_t{0x0101010101010101},0};
      auto dst=partial_[block*rp];
      if (groups != 0) {
        BlockReduceMax(dst,matrix_[col],static_cast<int32_t>(groups),mask,1,
                       static_cast<int32_t>(blocks),
                       static_cast<int32_t>(t_.h_pad));
      }
    }
    PipeBarrier<PIPE_V>();
    uint32_t planes=blocks;
    while (planes > 2) {
      const uint32_t pairs=planes / 2;
      const uint32_t upper=planes - pairs;
      Max(partial_,partial_,partial_[upper*rp],pairs*rp);
      PipeBarrier<PIPE_V>();
      planes=upper;
    }
    Max(result_,partial_,partial_[rp],rows_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void RowSums() {
    const uint32_t blocks=t_.h_pad / 8;
    const uint32_t rp=(rows_+7) / 8*8;
    const uint32_t groups=rows_ / 8;
    for (uint32_t block=0; block < blocks; ++block) {
      const uint32_t col=block*8;
      const uint32_t width=t_.h - col < 8 ? t_.h - col : 8;
      uint64_t mask[2]={
          ((uint64_t{1} << width) - 1)*uint64_t{0x0101010101010101},0};
      auto dst=block == 0 ? result_ : partial_[(block - 1)*rp];
      if (groups != 0) {
        BlockReduceSum(dst,matrix_[col],static_cast<int32_t>(groups),mask,1,
                       static_cast<int32_t>(blocks),
                       static_cast<int32_t>(t_.h_pad));
      }
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t first=0; first < rows_; first += 64) {
      const uint64_t width=rows_ - first < 64 ? rows_ - first : 64;
      auto dst=result_[first];
      Add(dst,partial_[first],dst,width,static_cast<uint8_t>(blocks - 1),
          {1,1,1,0,static_cast<uint8_t>(rp / 8),0});
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void BroadcastRows() {
    Brcb(tmp_,result_,static_cast<uint8_t>((rows_+7) / 8),{1,8});
    PipeBarrier<PIPE_V>();
  }

  template <bool Subtract>
  __aicore__ inline void ApplyRows() {
    const auto blockStride=static_cast<uint8_t>(t_.h_pad / 8);
    const auto repeatStride=static_cast<uint8_t>(t_.h_pad);
    const auto groups=static_cast<uint8_t>(rows_ / 8);
    for (uint32_t col=0; col < t_.h; col += 8) {
      const uint32_t width=t_.h - col < 8 ? t_.h - col : 8;
      uint64_t mask[2]={
          ((uint64_t{1} << width) - 1)*uint64_t{0x0101010101010101},0};
      auto m=matrix_[col];
      if (groups != 0) {
        if constexpr (Subtract) {
          Sub(m,m,tmp_,mask,groups,
              {blockStride,blockStride,1,repeatStride,repeatStride,8});
        } else {
          Mul(m,m,tmp_,mask,groups,
              {blockStride,blockStride,1,repeatStride,repeatStride,8});
        }
      }
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Softmax() {
    // Gates prepared the row maxima before consuming its exponential.
    BroadcastRows();
    ApplyRows<true>();
    Exp(matrix_,matrix_,rows_*t_.h_pad);
    PipeBarrier<PIPE_V>();
    RowSums();
    Div(result_,ones_,result_,rows_);
    PipeBarrier<PIPE_V>();
    BroadcastRows();
    ApplyRows<false>();
    Adds(matrix_,matrix_,t_.eps,rows_*t_.h_pad);
    PipeBarrier<PIPE_V>();
  }

  template <bool Materialize=true>
  __aicore__ inline void Columns() {
    uint32_t rows=(t_.h+3) / 4;
    Add(partial_,matrix_,matrix_[rows*span_],rows*span_);
    PipeBarrier<PIPE_V>();
    Add(partial_,partial_,matrix_[2*rows*span_],rows*span_);
    PipeBarrier<PIPE_V>();
    Add(partial_,partial_,matrix_[3*rows*span_],
        (t_.h - 3*rows)*span_);
    PipeBarrier<PIPE_V>();
    while (rows > 1) {
      const uint32_t half=rows / 2;
      const uint32_t upper=rows - half;
      Add(partial_,partial_,partial_[upper*span_],half*span_);
      PipeBarrier<PIPE_V>();
      rows=upper;
    }
    if (span_ >= 64) {
      Copy(denom_,partial_,uint64_t{64},static_cast<uint8_t>(span_ / 64),
           {1,1,8,8});
    }
    if (span_ % 64 != 0) {
      Copy(denom_[span_ / 64*64],partial_[span_ / 64*64],
           uint64_t{span_ % 64},1,{1,1,8,8});
    }
    PipeBarrier<PIPE_V>();
    Adds(denom_,denom_,t_.eps,span_);
    PipeBarrier<PIPE_V>();
    if constexpr (!Materialize) return;
    Div(denom_,ones_,denom_,span_);
    PipeBarrier<PIPE_V>();
    for (uint32_t col=0; col < span_; col += 64) {
      const uint64_t width=span_ - col < 64 ? span_ - col : 64;
      Mul(matrix_[col],matrix_[col],denom_[col],width,
          static_cast<uint8_t>(t_.h),
          {1,1,1,static_cast<uint8_t>(span_ / 8),
           static_cast<uint8_t>(span_ / 8),0});
    }
    PipeBarrier<PIPE_V>();
  }

  template <bool Matrix>
  __aicore__ inline void BuildMediumOutputMap(LocalTensor<int32_t> index,
                                              uint32_t count) {
    auto row=medium_stage_.ReinterpretCast<int32_t>();
    auto token=partial_.ReinterpretCast<int32_t>();
    const uint32_t chunk=(t_.input_bytes / sizeof(float) / 64)*64;
    const int32_t reciprocal =
        static_cast<int32_t>(((1U << 20)+t_.h - 1) / t_.h);
    for (uint32_t first=0; first < count; first += chunk) {
      const uint32_t length=count - first < chunk ? count - first : chunk;
      auto dst=index[first];
      CreateVecIndex(dst,static_cast<int32_t>(first),length);
      PipeBarrier<PIPE_V>();
      Muls(row,dst,reciprocal,length);
      PipeBarrier<PIPE_V>();
      ShiftRight(row,row,int32_t{20},length);
      PipeBarrier<PIPE_V>();
      if constexpr (Matrix) {
        Muls(token,row,reciprocal,length);
        PipeBarrier<PIPE_V>();
        ShiftRight(token,token,int32_t{20},length);
        Muls(row,row,static_cast<int32_t>(span_ - t_.h),length);
        PipeBarrier<PIPE_V>();
        Muls(token,token,
             static_cast<int32_t>(t_.h_pad) -
                 static_cast<int32_t>(t_.h*span_),
             length);
        Add(dst,dst,row,length);
        PipeBarrier<PIPE_V>();
        Add(dst,dst,token,length);
      } else {
        Muls(row,row,static_cast<int32_t>(t_.h_pad - t_.h),length);
        PipeBarrier<PIPE_V>();
        Add(dst,dst,row,length);
      }
      PipeBarrier<PIPE_V>();
      Muls(dst,dst,int32_t{4},length);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void CopyOutMedium(uint64_t first) {
    const uint32_t packedMatrixStride=(t_.h*t_.h+7) / 8*8;
    const uint32_t directGateCap=(t_.h*t_.tile_tokens+63) / 64*64;
    if (valid_ <= 8 && 2*directGateCap+valid_*packedMatrixStride <= medium_stage_capacity_) {
      // Pack each token with the hardware all-elements GatherMask pattern.
      // Only token boundaries need UB padding; MTE3 strips that padding.
      uint64_t retained=0;
      const GatherMaskParams gates{1,static_cast<uint16_t>(valid_),
                                   static_cast<uint16_t>(t_.h_pad / 8),0};
      GatherMask(medium_stage_,gates_,uint8_t{7},true,t_.h,gates,retained);
      GatherMask(medium_stage_[directGateCap],gates_[span_],uint8_t{7},
                 true,t_.h,gates,retained);
      const GatherMaskParams matrix{1,static_cast<uint16_t>(t_.h),
                                    static_cast<uint16_t>(span_ / 8),0};
      for (uint32_t token=0; token < valid_; ++token)
        GatherMask(medium_stage_[2*directGateCap+token*packedMatrixStride],
                   matrix_[token*t_.h_pad],uint8_t{7},true,t_.h,
                   matrix,retained);
      medium_indexed_valid_=0;
      Sync<HardEvent::V_MTE3>();
      const DataCopyExtParams g{1,4*t_.h*valid_,0,0,0};
      const DataCopyExtParams m{static_cast<uint16_t>(valid_),
                               4*t_.h*t_.h,0,0,0};
      DataCopyPad(pre_[first*t_.h],medium_stage_,g);
      DataCopyPad(post_[first*t_.h],medium_stage_[directGateCap],g);
      DataCopyPad(comb_[first*t_.h*t_.h],medium_stage_[2*directGateCap],m);
      Sync<HardEvent::MTE3_MTE2>();
      return;
    }
    const uint32_t gateCap=(t_.h*t_.tile_tokens+63) / 64*64;
    const uint32_t gates=t_.h*valid_,matrix=t_.h*gates;
    auto index=medium_index_;
    BuildMediumOutputMap<false>(index,gates);
    auto matrixIndex=index[gateCap];
    uint32_t group=1;
    while ((group*t_.h*t_.h) % 8 != 0) group *= 2;
    if (group > valid_) group=valid_;
    BuildMediumOutputMap<true>(matrixIndex,group*t_.h*t_.h);
    for (uint32_t tokens=group; tokens < valid_; tokens *= 2) {
      const uint32_t n=valid_ - tokens < tokens ? valid_ - tokens : tokens;
      Adds(matrixIndex[tokens*t_.h*t_.h],matrixIndex,
           static_cast<int32_t>(4*tokens*t_.h_pad),n*t_.h*t_.h);
      PipeBarrier<PIPE_V>();
    }
    medium_indexed_valid_=0;
    auto map=index.ReinterpretCast<uint32_t>();
    Gather(medium_stage_,gates_,map,uint32_t{0},gates);
    Gather(medium_stage_[gateCap],gates_[span_],map,uint32_t{0},gates);
    Gather(medium_stage_[2*gateCap],matrix_,map[gateCap],uint32_t{0},
           matrix);
    Sync<HardEvent::V_MTE3>();
    const DataCopyExtParams g{1,4*gates,0,0,0};
    const DataCopyExtParams m{1,4*matrix,0,0,0};
    DataCopyPad(pre_[first*t_.h],medium_stage_,g);
    DataCopyPad(post_[first*t_.h],medium_stage_[gateCap],g);
    DataCopyPad(comb_[first*t_.h*t_.h],medium_stage_[2*gateCap],m);
    Sync<HardEvent::MTE3_MTE2>();
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_,gateBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_,tmpBuffer_,resultBuffer_,
      workBuffer_,onesBuffer_,denomBuffer_,mediumStageBuffer_,
      mediumIndexBuffer_;
  LocalTensor<float> matrix_,gates_,partial_,tmp_,result_,ones_,denom_,
      medium_stage_;
  LocalTensor<int32_t> medium_index_;
  GlobalTensor<float> input_,pre_,post_,comb_;
  bool paddedBlocked_;
  TilingData t_;
  uint64_t start_,count_;
  uint32_t valid_,rows_,span_,medium_indexed_valid_,medium_stage_capacity_;
};

}  // namespace sinkhorn_f09

template <int=0>
__global__ __aicore__ __attribute__((aligned(1024))) void sinkhorn_f09_kernel(
    GM_ADDR mixes,GM_ADDR pre,GM_ADDR post,GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_f09::MediumDenseIoKernel op;
  op.Init(mixes,pre,post,comb,tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F09_MEDIUM_DENSE_IO_H_
