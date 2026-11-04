#ifndef SINKHORN_NORM_KERNELS_F07_NARROW_P16_DENSE_IO_H_
#define SINKHORN_NORM_KERNELS_F07_NARROW_P16_DENSE_IO_H_

#include "kernel_operator.h"
#include "sinkhorn_norm/common/primitives.h"
#include "sinkhorn_norm/tiling_data.h"

namespace sinkhorn_norm {
namespace f07 {

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

static __aicore__ __attribute__((always_inline)) inline void RunNarrowDenseIoGatesTail(
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

// Compact column/row/token planes. All vector lanes represent independent tokens.
template <int Op, bool CounterActive=false>
static __aicore__ __attribute__((always_inline)) inline void PlaneOp(
    __ubuf__ float *dst, __ubuf__ float *a, __ubuf__ float *b, uint32_t n,
    float scalar=0.0f) {
  if ASCEND_IS_AIV {
    if constexpr(!CounterActive) SetMaskCount();
    SetVectorMask<float,MaskMode::COUNTER>(0,n);
    if constexpr(Op==0) vadd(dst,a,b,1,1,1,1,8,8,8);
    if constexpr(Op==1) vmax(dst,a,b,1,1,1,1,8,8,8);
    if constexpr(Op==2) vsub(dst,a,b,1,1,1,1,8,8,8);
    if constexpr(Op==3) vmul(dst,a,b,1,1,1,1,8,8,8);
    if constexpr(Op==4) vdiv(dst,a,b,1,1,1,1,8,8,8);
    if constexpr(Op==5) vadds(dst,a,scalar,1,1,1,8,8);
    if constexpr(Op==6) vexp(dst,a,1,1,1,8,8);
    if constexpr(Op==7) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(dst),reinterpret_cast<__ubuf__ uint32_t *>(a),1,1,1,8,8);
    if constexpr(!CounterActive) SetMaskNorm();
  }
}
template <bool Max>
static __aicore__ __attribute__((always_inline)) inline void PlaneRowReduce(
    __ubuf__ float *out,__ubuf__ float *scratch,__ubuf__ float *matrix,
    uint32_t h,uint32_t span) {
  if ASCEND_IS_AIV {
    SetMaskCount();
    const uint32_t half=h/2;
    PlaneOp<Max?1:0,true>(scratch,matrix,matrix+half*span,half*span);
    if(h%2) PlaneOp<7,true>(scratch+half*span,matrix+(h-1)*span,matrix,span);
    pipe_barrier(PIPE_V);
    uint32_t planes=(h+1)/2;
    while(planes>1) {
      const uint32_t n=planes/2, upper=planes-n;
      PlaneOp<Max?1:0,true>(planes==2?out:scratch,scratch,scratch+upper*span,n*span);
      pipe_barrier(PIPE_V); planes=upper;
    }
    SetMaskNorm();
  }
}
// Natural [row][column][token] planes: reduce the inner column dimension.
template <bool Max>
static __aicore__ __attribute__((always_inline)) inline void PlaneInnerReduce(
    __ubuf__ float *out,__ubuf__ float *scratch,__ubuf__ float *matrix,
    uint32_t h,uint32_t pitch) {
  if ASCEND_IS_AIV {
    const uint32_t span=h*pitch,pairs=h/2,partialPitch=((h+1)/2)*pitch;
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,pairs*pitch);
    for(uint32_t r=0;r<h;++r) {
      if constexpr(Max) vmax(scratch+r*partialPitch,matrix+r*span,matrix+r*span+pairs*pitch,1,1,1,1,8,8,8);
      else vadd(scratch+r*partialPitch,matrix+r*span,matrix+r*span+pairs*pitch,1,1,1,1,8,8,8);
    }
    if(h%2) {
      SetVectorMask<float,MaskMode::COUNTER>(0,pitch);
      for(uint32_t r=0;r<h;++r)
        vcopy(reinterpret_cast<__ubuf__ uint32_t *>(scratch+r*partialPitch+pairs*pitch),
              reinterpret_cast<__ubuf__ uint32_t *>(matrix+r*span+(h-1)*pitch),1,1,1,8,8);
    }
    pipe_barrier(PIPE_V);
    uint32_t planes=(h+1)/2;
    while(planes>1) {
      const uint32_t half=planes/2,upper=planes-half;
      SetVectorMask<float,MaskMode::COUNTER>(0,half*pitch);
      for(uint32_t r=0;r<h;++r) {
        auto dst=planes==2?out+r*pitch:scratch+r*partialPitch;
        if constexpr(Max) vmax(dst,scratch+r*partialPitch,scratch+r*partialPitch+upper*pitch,1,1,1,1,8,8,8);
        else vadd(dst,scratch+r*partialPitch,scratch+r*partialPitch+upper*pitch,1,1,1,1,8,8,8);
      }
      pipe_barrier(PIPE_V);planes=upper;
    }
    SetMaskNorm();
  }
}
static __aicore__ __attribute__((always_inline)) inline void RunNarrowDenseIoCompleteRounds(
    __ubuf__ float *matrix,__ubuf__ float *scratch,__ubuf__ float *ones,
    __ubuf__ float *denom,uint32_t h,uint32_t pitch,float eps,uint32_t iters) {
  if ASCEND_IS_AIV {
    const uint32_t span=h*pitch,n=h*span;
    SetMaskNorm();WideRoundMask(pitch==64?~uint64_t{0}:(uint64_t{1}<<pitch)-1);
    for(uint32_t r=0;r<h;++r)
      vsub(matrix+r*span,matrix+r*span,denom+r*pitch,h,1,1,1,pitch/8,pitch/8,0);
    pipe_barrier(PIPE_V);
    PlaneOp<6>(matrix,matrix,matrix,n);pipe_barrier(PIPE_V);
    for(uint32_t round=0;round<iters;++round) {
      PlaneInnerReduce<false>(denom,scratch,matrix,h,pitch);
      if(round) {PlaneOp<5>(denom,denom,denom,span,eps);pipe_barrier(PIPE_V);}
      PlaneOp<4>(denom,ones,denom,span);pipe_barrier(PIPE_V);
      WideRoundMask(pitch==64?~uint64_t{0}:(uint64_t{1}<<pitch)-1);
      for(uint32_t r=0;r<h;++r)
        vmul(matrix+r*span,matrix+r*span,denom+r*pitch,h,1,1,1,pitch/8,pitch/8,0);
      pipe_barrier(PIPE_V);
      if(!round) {PlaneOp<5>(matrix,matrix,matrix,n,eps);pipe_barrier(PIPE_V);}
      PlaneRowReduce<false>(denom,scratch,matrix,h,span);
      PlaneOp<5>(denom,denom,denom,span,eps);pipe_barrier(PIPE_V);
      PlaneOp<4>(denom,ones,denom,span);pipe_barrier(PIPE_V);
      SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,span);
      for(uint32_t r=0;r<h;++r) vmul(matrix+r*span,matrix+r*span,denom,1,1,1,1,8,8,8);
      SetMaskNorm();pipe_barrier(PIPE_V);ResetMask();
    }
  }
}

class NarrowP16DenseIoKernel {
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

    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes);
    pipe_.InitBuffer(partialBuffer_, 2 * span * sizeof(float) + 256);
    pipe_.InitBuffer(onesBuffer_, ones * sizeof(float));
    pipe_.InitBuffer(denomBuffer_, rp * sizeof(float));

    dense_gate_cap_ = (t_.h * t_.tile_tokens + 63) / 64 * 64;
    const uint32_t matrixCap = (t_.h * t_.h * t_.tile_tokens + 63) / 64 * 64;
    zero_offset_ = 2 * dense_gate_cap_ + matrixCap;
    // Decouple denseInputBuffer_ and denseOutputBuffer_ so MTE3 writeback of tile k
    // overlaps with MTE2 read of tile k+1 without MTE3_MTE2 synchronization!
    pipe_.InitBuffer(denseInputBuffer_, 4*((t_.input_width+7)/8*8)*t_.tile_tokens);
    const uint32_t outputCap=((t_.h*t_.h+15)/16*16)*t_.tile_tokens;
    const uint32_t packedCap=outputCap>matrixCap?outputCap:matrixCap;
    pipe_.InitBuffer(denseOutputBuffer_, 4 * (2*dense_gate_cap_+packedCap));
    pipe_.InitBuffer(outputIndexBuffer_, 4 * dense_gate_cap_);

    input_gate_cap_ = (2 * span + 63) / 64 * 64;
    const uint32_t inputMatrixCap = (t_.h * span + 63) / 64 * 64;
    pipe_.InitBuffer(inputIndexBuffer_, 4 * input_gate_cap_);

    dense_input_ = denseInputBuffer_.Get<float>();
    dense_output_ = denseOutputBuffer_.Get<float>();
    output_index_ = outputIndexBuffer_.Get<int32_t>();
    input_index_ = inputIndexBuffer_.Get<int32_t>();

    comb_write_pending_ = false;

    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>();
    tmp_ = partial_;
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();

    if(count_>0) {
      const DataCopyExtParams packed{static_cast<uint16_t>(t_.tile_tokens),4*t_.input_width,0,0,0};
      const DataCopyPadExtParams<float> noPad{false,0,0,0.0f};
      DataCopyPad(dense_input_,input_[start_*t_.input_width],packed,noPad);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    Duplicate(denom_, 0.0f, rp);
    Duplicate(ones_, 1.0f, ones);
    PipeBarrier<PIPE_V>();

    // Pre-build index maps once in Init()
    valid_ = t_.tile_tokens;
    rows_ = t_.h * valid_;
    span_ = t_.h_pad * valid_;
    BuildDenseInputMaps();
    input_indexed_valid_ = valid_;
    BuildDenseOutputMap<false>(output_index_, t_.h * t_.tile_tokens);
    gate_output_map_ready_ = true;
  }

  __aicore__ inline void Process() {
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    for (uint64_t offset = 0; offset < count_; offset += t_.tile_tokens) {
      const uint64_t first = start_ + offset;
      const uint64_t left = count_ - offset;
      valid_ = left < t_.tile_tokens ? static_cast<uint32_t>(left) : t_.tile_tokens;
      rows_ = t_.h * valid_;
      span_ = t_.h_pad * valid_;
      matrix_=matrixBuffer_.Get<float>()[((2*t_.h)%8)*((valid_+15)/16*16)];

      if (input_indexed_valid_ != valid_) {
        BuildDenseInputMaps();
        input_indexed_valid_ = valid_;
      }
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
      auto index = input_index_.template ReinterpretCast<uint32_t>();
      Gather(gates_, dense_input_, index, uint32_t{0}, 2 * span_);
      UnpackMatrixBlocks();
      PipeBarrier<PIPE_V>();

      if (offset + t_.tile_tokens < count_) {
        const uint64_t next_first = first + t_.tile_tokens;
        const uint64_t next_left = count_ - (offset + t_.tile_tokens);
        const uint32_t next_valid =
            next_left < t_.tile_tokens ? static_cast<uint32_t>(next_left) : t_.tile_tokens;
        const DataCopyExtParams nextPacked{static_cast<uint16_t>(next_valid), 4 * t_.input_width, 0, 0, 0};
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        DataCopyPad(dense_input_, input_[next_first * t_.input_width], nextPacked, noPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
      }

      GatesAndRowMax();

      const uint32_t gates = t_.h * valid_;
      const uint32_t matrix = t_.h * gates;

      WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
      const DataCopyExtParams g{1, 4 * gates, 0, 0, 0};
      DataCopyPad(pre_[first * t_.h], dense_output_, g);
      DataCopyPad(post_[first * t_.h], dense_output_[dense_gate_cap_], g);

      SetMaskNorm();
      RunNarrowDenseIoCompleteRounds(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(dense_output_[2 * dense_gate_cap_].GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          t_.h, (valid_ + 15) / 16 * 16, t_.eps, t_.iters);
      ResetMask();

      PackMatrixBlocks();

      CopyOutComb(first, matrix, offset + t_.tile_tokens < count_);
    }
  }

 private:
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void BuildDenseInputMaps() {
    auto token = partial_.template ReinterpretCast<int32_t>();
    CreateVecIndex(input_index_, int32_t{0}, span_);
    PipeBarrier<PIPE_V>();
    ShiftRight(token, input_index_, int32_t{4}, span_);
    PipeBarrier<PIPE_V>();
    Muls(token, token, static_cast<int32_t>((t_.input_width+7)/8*8 - 16), span_);
    PipeBarrier<PIPE_V>();
    Add(input_index_, input_index_, token, span_);
    PipeBarrier<PIPE_V>();
    Muls(input_index_, input_index_, int32_t{4}, span_);
    PipeBarrier<PIPE_V>();
    Adds(input_index_[span_], input_index_,static_cast<int32_t>(4*t_.h),span_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GatesAndRowMax() {
    const uint32_t gatesCount = t_.h * valid_;

    Muls(tmp_, gates_, -1.0f, 2 * span_);
    PipeBarrier<PIPE_V>();
    RunNarrowDenseIoGatesTail(
        reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(tmp_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
        span_, t_.eps);

    if (comb_write_pending_) {
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      comb_write_pending_ = false;
    }

    // Gather pre and post gates into dense_output_ immediately and signal V_MTE3 (EVENT_ID1)!
    auto gateIndex = output_index_.template ReinterpretCast<uint32_t>();
    Gather(dense_output_, gates_, gateIndex, uint32_t{0}, gatesCount);
    Gather(dense_output_[dense_gate_cap_], gates_[span_], gateIndex, uint32_t{0},
           gatesCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

    PlaneInnerReduce<true>(
        reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(dense_output_[2*dense_gate_cap_].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
        t_.h,((valid_+15)/16*16));
    ResetMask();
  }

  template <bool Matrix>
  __aicore__ inline void BuildDenseOutputMap(LocalTensor<int32_t> index,
                                             uint32_t count) {
    auto row = partial_.template ReinterpretCast<int32_t>();
    // Matrix storage is idle while rebuilding output indices. Keep the two
    // arithmetic temporaries disjoint without splitting the index seed.
    auto token = matrix_.template ReinterpretCast<int32_t>();
    const int32_t reciprocal =
        static_cast<int32_t>(((1U << 20) + t_.h - 1) / t_.h);
    const uint32_t chunk = 6 * span_;
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
        const int32_t pitch=static_cast<int32_t>((valid_+7)/8*8);
        Muls(row,row,pitch*(1-static_cast<int32_t>(t_.h*t_.h)),length);
        PipeBarrier<PIPE_V>();
        Muls(token,token,1-static_cast<int32_t>(t_.h)*pitch,length);
        Muls(dst,dst,static_cast<int32_t>(t_.h)*pitch,length);
        PipeBarrier<PIPE_V>();
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

  __aicore__ inline void UnpackMatrixBlocks() {
    const uint32_t pitch=(valid_+15)/16*16,inputPitch=(t_.input_width+7)/8*8;
    const uint32_t prefix=(2*t_.h)%8,firstFeature=2*t_.h-prefix;
    const uint8_t repeats=(prefix+t_.h*t_.h+7)/8;
    const uint64_t inBase=reinterpret_cast<uint64_t>(dense_input_.GetPhyAddr());
    const uint64_t dstBase=reinterpret_cast<uint64_t>(matrixBuffer_.Get<float>().GetPhyAddr());
    uint64_t srcList[16],dstList[16];
    for(uint32_t tokenBase=0;tokenBase<pitch;tokenBase+=16) {
      const uint32_t remaining=valid_-tokenBase;
      for(uint32_t token=0;token<16;++token) {
        const uint32_t actual=token<remaining?token:0;
        srcList[token]=inBase+4*((tokenBase+actual)*inputPitch+firstFeature);
      }
      for(uint32_t feature=0;feature<8;++feature) {
        dstList[2*feature]=dstBase+4*(feature*pitch+tokenBase);
        dstList[2*feature+1]=dstList[2*feature]+32;
      }
      const TransDataTo5HDParams params{false,false,repeats,
          static_cast<uint16_t>(pitch),uint16_t{1}};
      TransDataTo5HD<float>(dstList,srcList,params);
    }
  }

  __aicore__ inline void PackMatrixBlocks() {
    const uint32_t pitch=(valid_+15)/16*16,features=t_.h*t_.h;
    const uint32_t outPitch=(features+15)/16*16;
    if(outPitch>features) {
      Duplicate(matrix_[features*pitch],0.0f,(outPitch-features)*pitch);
      PipeBarrier<PIPE_V>();
    }
    const uint64_t matrixBase=reinterpret_cast<uint64_t>(matrix_.GetPhyAddr());
    const uint64_t outputBase=reinterpret_cast<uint64_t>(dense_output_[2*dense_gate_cap_].GetPhyAddr());
    uint64_t srcList[16],dstList[16];
    for(uint32_t tokenBase=0;tokenBase<pitch;tokenBase+=8) {
      for(uint32_t feature=0;feature<16;++feature)
        srcList[feature]=matrixBase+4*(feature*pitch+tokenBase);
      for(uint32_t token=0;token<8;++token) {
        dstList[2*token]=outputBase+4*((tokenBase+token)*outPitch);
        dstList[2*token+1]=dstList[2*token]+32;
      }
      const TransDataTo5HDParams params{false,false,static_cast<uint8_t>(outPitch/16),
          uint16_t{2},static_cast<uint16_t>(2*pitch)};
      TransDataTo5HD<float>(dstList,srcList,params);
    }
  }

  __aicore__ inline void CopyOutComb(uint64_t first, uint32_t matrix,
                                     bool hasMoreTiles) {
    Sync<HardEvent::V_MTE3>();
    const uint32_t features=t_.h*t_.h,outPitch=(features+15)/16*16;
    const DataCopyExtParams m{static_cast<uint16_t>(valid_),4*features,
                             (outPitch-(features+7)/8*8)/8,0,0};
    DataCopyPad(comb_[first * t_.h * t_.h], dense_output_[2 * dense_gate_cap_], m);
    if (hasMoreTiles) {
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      comb_write_pending_ = true;
    }
  }

  TPipe pipe_;
  TBuf<TPosition::VECOUT> matrixBuffer_, gateBuffer_;
  TBuf<TPosition::VECCALC> partialBuffer_, onesBuffer_, denomBuffer_;
  TBuf<TPosition::VECCALC> denseInputBuffer_, denseOutputBuffer_,
      outputIndexBuffer_, inputIndexBuffer_;
  LocalTensor<float> matrix_, gates_, partial_, tmp_, ones_, denom_,
      dense_input_, dense_output_;
  LocalTensor<int32_t> output_index_, input_index_;
  GlobalTensor<float> input_, pre_, post_, comb_;
  sinkhorn_norm::TilingData t_;
  uint64_t start_, count_;
  uint32_t valid_, rows_, span_;
  uint32_t dense_gate_cap_, zero_offset_, indexed_valid_, input_gate_cap_,
      input_indexed_valid_;
  bool gate_output_map_ready_, comb_write_pending_;
};

}  // namespace f07
}  // namespace sinkhorn_norm

template <int = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f07_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::TilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f07::NarrowP16DenseIoKernel op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F07_NARROW_P16_DENSE_IO_H_
