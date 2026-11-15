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

static __aicore__ __attribute__((always_inline)) inline void RunNarrowDenseIoGatesTail(
    __ubuf__ float *gates, __ubuf__ float *tmp, __ubuf__ float *ones,
    uint32_t span, float eps) {
  if ASCEND_IS_AIV {
    const uint32_t n = 2 * span;
    const uint32_t gateRepeats = (n + 63) / 64;
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
      vmuls(gates + span, gates + span, 2.0f, spanFull, 1, 1, 8, 8);
    }
    if (spanTail != 0) {
      WideRoundMask((uint64_t{1} << spanTail) - 1);
      vadds(gates + spanFull * 64, gates + spanFull * 64, eps, 1, 1, 1, 8, 8);
      vmuls(gates + span + spanFull * 64, gates + span + spanFull * 64,
            2.0f, 1, 1, 1, 8, 8);
      WideRoundMask(~uint64_t{0});
    }
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
    uint32_t h,uint32_t span,float eps=0.0f) {
  if ASCEND_IS_AIV {
    SetMaskCount();
    const uint32_t half=h/2;
    PlaneOp<Max?1:0,true>(scratch,matrix,matrix+half*span,half*span);
    if(h%2) {
      if constexpr(Max) PlaneOp<7,true>(scratch+half*span,matrix+(h-1)*span,matrix,span);
      else PlaneOp<5,true>(scratch+half*span,matrix+(h-1)*span,matrix,span,eps);
    }
    pipe_barrier(PIPE_V);
    uint32_t planes=(h+1)/2;
    while(planes>1) {
      const uint32_t n=planes/2, upper=planes-n;
      PlaneOp<Max?1:0,true>(planes==2?out:scratch,scratch,scratch+upper*span,n*span);
      pipe_barrier(PIPE_V); planes=upper;
    }
    SetMaskNorm();
    if constexpr(!Max) {
      if(h%2==0 && eps!=0.0f) {PlaneOp<5>(out,out,out,span,eps);pipe_barrier(PIPE_V);}
    }
  }
}
// Natural [row][column][token] planes: reduce the inner column dimension.
template <bool Max, bool PackedPairs>
static __aicore__ __attribute__((always_inline)) inline void PlaneInnerReduce(
    __ubuf__ float *out,__ubuf__ float *scratch,__ubuf__ float *matrix,
    uint32_t h,uint32_t pitch,float eps=0.0f) {
  if ASCEND_IS_AIV {
    const uint32_t span=h*pitch,pairs=h/2,partialPitch=((h+1)/2)*pitch;
    const uint32_t group = PackedPairs ? 64 / pitch : 1;
    const uint64_t one = pitch == 64 ? ~uint64_t{0} : (uint64_t{1} << pitch) - 1;
    // Each repeat still owns one row. Adjacent column pairs share it when
    // their token planes fit in the same vector.
    SetMaskNorm();
    WideRoundMask(one);
    for(uint32_t c=0;c<pairs;c+=group) {
      if (group > 1) {
        const uint32_t active = (pairs - c < group ? pairs - c : group) * pitch;
        WideRoundMask(active == 64 ? ~uint64_t{0} : (uint64_t{1} << active) - 1);
      }
      if constexpr(Max) vmax(scratch+c*pitch,matrix+c*pitch,matrix+(c+pairs)*pitch,
                             h,1,1,1,partialPitch/8,span/8,span/8);
      else vadd(scratch+c*pitch,matrix+c*pitch,matrix+(c+pairs)*pitch,
                h,1,1,1,partialPitch/8,span/8,span/8);
    }
    if(h%2) {
      if (group > 1) WideRoundMask(one);
      if constexpr(Max) vcopy(reinterpret_cast<__ubuf__ uint32_t *>(scratch+pairs*pitch),
                             reinterpret_cast<__ubuf__ uint32_t *>(matrix+(h-1)*pitch),
                             h,1,1,partialPitch/8,span/8);
      else vadds(scratch+pairs*pitch,matrix+(h-1)*pitch,eps,h,1,1,partialPitch/8,span/8);
    }
    pipe_barrier(PIPE_V);
    uint32_t planes=(h+1)/2;
    while(planes>1) {
      const uint32_t half=planes/2,upper=planes-half;
      for(uint32_t c=0;c<half;c+=group) {
        if (group > 1) {
          const uint32_t active = (half - c < group ? half - c : group) * pitch;
          WideRoundMask(active == 64 ? ~uint64_t{0} : (uint64_t{1} << active) - 1);
        }
        auto dst=planes==2?out+c*pitch:scratch+c*pitch;
        const uint32_t dstPitch=planes==2?pitch:partialPitch;
        if constexpr(Max) vmax(dst,scratch+c*pitch,scratch+(c+upper)*pitch,
                               h,1,1,1,dstPitch/8,partialPitch/8,partialPitch/8);
        else vadd(dst,scratch+c*pitch,scratch+(c+upper)*pitch,
                  h,1,1,1,dstPitch/8,partialPitch/8,partialPitch/8);
      }
      pipe_barrier(PIPE_V);planes=upper;
    }
    WideRoundMask(~uint64_t{0});
    if constexpr(!Max) {
      if(h%2==0 && eps!=0.0f) {PlaneOp<5>(out,out,out,span,eps);pipe_barrier(PIPE_V);}
    }
  }
}
// Adjacent column planes can share one vector repeat without mixing
// their token lanes. Fold the two lanes only after accumulating all pairs.
static __aicore__ __attribute__((always_inline)) inline void AccumulatePairedRowDots(
    __ubuf__ float *out, __ubuf__ float *parts, __ubuf__ float *matrix,
    __ubuf__ float *factors, uint32_t h, uint32_t pitch) {
  if ASCEND_IS_AIV {
    const uint32_t span = h * pitch;
    const uint32_t pairs = h / 2;
    const uint32_t pairedStride = 2 * pitch / 8;
    const uint32_t rowStride = span / 8;
    const uint32_t lanes = 2 * pitch;
    auto even = parts;
    auto odd = parts + 2 * span;
    WideRoundMask(lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1);
    vmul(even, matrix, factors, h, 1, 1, 1, pairedStride, rowStride, 0);
    vmul(odd, matrix + lanes, factors + lanes, h, 1, 1, 1, pairedStride, rowStride, 0);
    pipe_barrier(PIPE_V);
    uint32_t pair = 2;
    for (; pair + 1 < pairs; pair += 2) {
      const uint32_t offset = pair * lanes;
      vmla(even, matrix + offset, factors + offset, h,
           1, 1, 1, pairedStride, rowStride, 0);
      vmla(odd, matrix + offset + lanes, factors + offset + lanes, h,
           1, 1, 1, pairedStride, rowStride, 0);
      pipe_barrier(PIPE_V);
    }
    if (pair < pairs) {
      const uint32_t offset = pair * lanes;
      vmla(even, matrix + offset, factors + offset, h,
           1, 1, 1, pairedStride, rowStride, 0);
      pipe_barrier(PIPE_V);
    }
    PlaneOp<0>(even, even, odd, 2 * span);
    pipe_barrier(PIPE_V);
    WideRoundMask((uint64_t{1} << pitch) - 1);
    vadd(odd, even, even + pitch, h, 1, 1, 1, pitch / 8, pairedStride, pairedStride);
    if (h % 2) {
      const uint32_t last = (h - 1) * pitch;
      vmla(out, matrix + last, factors + last, h, 1, 1, 1, pitch / 8, rowStride, 0);
    }
    pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vadd(out, out, odd, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
  }
}

template <bool Columns>
static __aicore__ __attribute__((always_inline)) inline void AccumulateResidentDots(
    __ubuf__ float *out, __ubuf__ float *parts, __ubuf__ float *matrix,
    __ubuf__ float *factors, uint32_t h, uint32_t pitch) {
  if ASCEND_IS_AIV {
    const uint32_t span = h * pitch;
    const uint32_t stride = Columns ? pitch : span;
    const uint32_t step = Columns ? span : pitch;
    WideRoundMask(pitch == 64 ? ~uint64_t{0} : (uint64_t{1} << pitch) - 1);
    for (uint32_t first = 0; first < h; first += 4) {
      #pragma unroll
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t index = first + lane;
        if (index >= h) continue;
        auto dst = lane == 0 ? out : parts + (lane - 1) * span;
        auto src = matrix + index * step;
        auto factor = factors + index * pitch;
        if (first == 0 && lane != 0)
          vmul(dst, src, factor, h, 1, 1, 1, pitch / 8, stride / 8, 0);
        else
          vmla(dst, src, factor, h, 1, 1, 1, pitch / 8, stride / 8, 0);
      }
      // The next group reads the accumulators written by this group.
      pipe_barrier(PIPE_V);
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vadd(out, out, parts, 1, 1, 1, 1, 8, 8, 8);
    vadd(parts + span, parts + span, parts + 2 * span, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(out, out, parts + span, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
  }
}

// For a fixed positive matrix X, the represented state is X_ij/(a_i*b_j).
// Updating a with X/b + eps*a and then b with X^T/a + eps*b preserves
// a segment of finite rounds before writing the matrix and resetting the factors.
static __aicore__ __attribute__((always_inline)) inline void RunResidentPlaneSegment(
    __ubuf__ float *matrix, __ubuf__ float *scratch, __ubuf__ float *ones,
    uint32_t h, uint32_t pitch, float eps, uint32_t rounds) {
  if ASCEND_IS_AIV {
    const uint32_t span = h * pitch;
    const uint32_t pitchBlocks = pitch / 8;
    const uint64_t lanes = pitch == 64 ? ~uint64_t{0} : (uint64_t{1} << pitch) - 1;
    auto rowFactor = scratch;
    auto colFactor = scratch + span;
    auto inverseRow = scratch + 2 * span;
    auto inverseCol = scratch + 3 * span;
    auto partialDots = scratch + 4 * span;

    PlaneOp<7>(inverseCol, ones, ones, span);
    PlaneOp<7>(colFactor, ones, ones, span);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    vector_dup(rowFactor, eps, 1, 1, 1, 8, 0);
    SetMaskNorm();
    pipe_barrier(PIPE_V);
    for (uint32_t round = 0; round < rounds; ++round) {
      if (pitch <= 32)
        AccumulatePairedRowDots(rowFactor, partialDots, matrix, colFactor, h, pitch);
      else
        AccumulateResidentDots<false>(rowFactor, partialDots, matrix, colFactor, h, pitch);
      // The previous column factor is now dead. Seed its next sum while
      // converting the independent new row denominator to a factor.
      vdiv(inverseRow, ones, rowFactor, 1, 1, 1, 1, 8, 8, 8);
      vmuls(colFactor, inverseCol, eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      auto oldRow = inverseRow;
      inverseRow = rowFactor;
      rowFactor = oldRow;

      SetMaskNorm();
      AccumulateResidentDots<true>(colFactor, partialDots, matrix, rowFactor, h, pitch);
      vdiv(inverseCol, ones, colFactor, 1, 1, 1, 1, 8, 8, 8);
      if (round + 1 < rounds)
        vmuls(rowFactor, inverseRow, eps, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      auto oldCol = inverseCol;
      inverseCol = colFactor;
      colFactor = oldCol;
      SetMaskNorm();
    }
    WideRoundMask(lanes);
    for (uint32_t row = 0; row < h; ++row) {
      auto dst = matrix + row * span;
      vmul(dst, dst, rowFactor + row * pitch, h, 1, 1, 1, pitchBlocks, pitchBlocks, 0);
    }
    pipe_barrier(PIPE_V);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, span);
    for (uint32_t row = 0; row < h; ++row)
      vmul(matrix + row * span, matrix + row * span, colFactor, 1, 1, 1, 1, 8, 8, 8);
    SetMaskNorm();
    pipe_barrier(PIPE_V);
    ResetMask();
  }
}

template <int FixedH = 0, int FixedI = 0, bool UseFactors = false>
static __aicore__ __attribute__((always_inline)) inline void RunNarrowDenseIoCompleteRounds(
    __ubuf__ float *matrix,__ubuf__ float *scratch,__ubuf__ float *ones,
    __ubuf__ float *denom,uint32_t height,uint32_t pitch,float eps,uint32_t iterations) {
  if ASCEND_IS_AIV {
    const uint32_t h = FixedH ? FixedH : height;
    const uint32_t iters = FixedI ? FixedI : iterations;
    const uint32_t span=h*pitch,n=h*span;
    SetMaskNorm();WideRoundMask(pitch==64?~uint64_t{0}:(uint64_t{1}<<pitch)-1);
    #pragma unroll
    for(uint32_t r=0;r<h;++r)
      vsub(matrix+r*span,matrix+r*span,denom+r*pitch,h,1,1,1,pitch/8,pitch/8,0);
    pipe_barrier(PIPE_V);
    PlaneOp<6>(matrix,matrix,matrix,n);pipe_barrier(PIPE_V);
    #pragma unroll
    for(uint32_t round=0;round<iters;++round) {
      PlaneInnerReduce<false, UseFactors>(denom,scratch,matrix,h,pitch,round?eps:0.0f);
      PlaneOp<4>(denom,ones,denom,span);pipe_barrier(PIPE_V);
      WideRoundMask(pitch==64?~uint64_t{0}:(uint64_t{1}<<pitch)-1);
      #pragma unroll
      for(uint32_t r=0;r<h;++r)
        vmul(matrix+r*span,matrix+r*span,denom+r*pitch,h,1,1,1,pitch/8,pitch/8,0);
      pipe_barrier(PIPE_V);
      if(!round) {PlaneOp<5>(matrix,matrix,matrix,n,eps);pipe_barrier(PIPE_V);}
      PlaneRowReduce<false>(denom,scratch,matrix,h,span,eps);
      PlaneOp<4>(denom,ones,denom,span);pipe_barrier(PIPE_V);
      SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,span);
      #pragma unroll
      for(uint32_t r=0;r<h;++r) vmul(matrix+r*span,matrix+r*span,denom,1,1,1,1,8,8,8);
      SetMaskNorm();pipe_barrier(PIPE_V);ResetMask();
      if constexpr (UseFactors && FixedI == 0) {
        if (round == 1 && iters >= 12 && eps >= 0x1p-24f && eps <= 0x1p-8f) {
          constexpr uint32_t exactTail = 4;
          constexpr uint32_t blockLimit = 16;
          while (round + 1 < iters - exactTail) {
            const uint32_t left = iters - exactTail - round - 1;
            const uint32_t block = left < blockLimit ? left : blockLimit;
            RunResidentPlaneSegment(matrix, scratch, ones, h, pitch, eps, block);
            round += block;
          }
        }
      }
    }
  }
}

template <int FixedH = 0, int FixedI = 0, bool UseFactors = false>
class NarrowP16DenseIoKernel {
 public:
  __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                              GM_ADDR comb, const sinkhorn_norm::NarrowDenseArgs &t) {
    t_ = {};
    t_.eps = t.eps; t_.iters = FixedI ? FixedI : t.iters; t_.h = FixedH ? FixedH : t.h; t_.h_pad = 16;
    t_.tokens_per_core = t.owner; t_.extra_core_count = t.extra;
    t_.tile_tokens = t.tile; t_.input_width = Height() * (Height() + 2);
    t_.matrix_bytes = 4 * ((2 * Height()) % 8 + (Height() * Height() + 15) / 16 * 16) * t.tile;
    t_.gate_bytes = 4 * 16 * t.tile;
    const uint64_t core = GetBlockIdx();
    start_ = core * t_.tokens_per_core +
             (core < t_.extra_core_count ? core : t_.extra_core_count);
    count_ = t_.tokens_per_core + (core < t_.extra_core_count ? 1 : 0);
    // Legal B*S is at most1048576; only these local schedule divisions
    // narrow to32bits. GM offsets and ownership stay64bit.
    const uint32_t owned = static_cast<uint32_t>(count_);
    const uint32_t tiles = (owned + t_.tile_tokens - 1) / t_.tile_tokens;
    tile_count_ = tiles;
    tile_base_ = tiles ? owned / tiles / 16 * 16 : 0;
    tile_tail_ = owned % 16;
    const uint32_t extra = tiles ? (owned - tiles * tile_base_) / 16 : 0;
    tile_small_count_ = tiles - extra - (tile_tail_ ? 1 : 0);

    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mixes));
    pre_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
    post_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(post));
    comb_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(comb));

    const uint32_t rp = (Height() * t_.tile_tokens + 7) / 8 * 8;
    const uint32_t span = t_.tile_tokens * PaddedHeight();
    const uint32_t ones = rp > 2 * span ? rp : 2 * span;
    const uint32_t tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;

    pipe_.InitBuffer(matrixBuffer_, t_.matrix_bytes);
    pipe_.InitBuffer(gateBuffer_, 2 * t_.gate_bytes);
    pipe_.InitBuffer(partialBuffer_, 2 * span * sizeof(float) + 256);
    pipe_.InitBuffer(onesBuffer_, ones * sizeof(float));
    pipe_.InitBuffer(denomBuffer_, rp * sizeof(float));

    dense_gate_cap_ = (Height() * t_.tile_tokens + 63) / 64 * 64;
    const uint32_t matrixCap = (Height() * Height() * t_.tile_tokens + 63) / 64 * 64;
    zero_offset_ = 2 * dense_gate_cap_ + matrixCap;
    // Decouple denseInputBuffer_ and denseOutputBuffer_ so MTE3 writeback of tile k
    // overlaps with MTE2 read of tile k+1 without MTE3_MTE2 synchronization!
    pipe_.InitBuffer(denseInputBuffer_, 4*((InputWidth()+7)/8*8)*t_.tile_tokens);
    const uint32_t outputCap=((Height()*Height()+15)/16*16)*t_.tile_tokens;
    const uint32_t packedCap=outputCap>matrixCap?outputCap:matrixCap;
    pipe_.InitBuffer(denseOutputBuffer_, 4 * (2*dense_gate_cap_+packedCap));
    pipe_.InitBuffer(outputIndexBuffer_, 4 * dense_gate_cap_);

    input_gate_cap_ = (2 * span + 63) / 64 * 64;
    const uint32_t inputMatrixCap = (Height() * span + 63) / 64 * 64;
    pipe_.InitBuffer(inputIndexBuffer_, 4 * input_gate_cap_);

    dense_input_ = denseInputBuffer_.Get<float>();
    dense_output_ = denseOutputBuffer_.Get<float>();
    output_index_ = outputIndexBuffer_.Get<int32_t>();
    input_index_ = inputIndexBuffer_.Get<int32_t>();

    comb_write_pending_ = false;
    compact_gates_ = UseFactors || count_ % 8 == 0;
    input_map_span_ = (compact_gates_ ? Height() : PaddedHeight()) * t_.tile_tokens;

    matrix_ = matrixBuffer_.Get<float>();
    gates_ = gateBuffer_.Get<float>();
    partial_ = partialBuffer_.Get<float>();
    tmp_ = partial_;
    ones_ = onesBuffer_.Get<float>();
    denom_ = denomBuffer_.Get<float>();

    if(count_>0) {
      const DataCopyExtParams packed{static_cast<uint16_t>(TileLength(0,count_)),4*InputWidth(),0,0,0};
      const DataCopyPadExtParams<float> noPad{false,0,0,0.0f};
      DataCopyPad(dense_input_,input_[start_*InputWidth()],packed,noPad);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    Duplicate(denom_, 0.0f, rp);
    Duplicate(ones_, 1.0f, ones);
    PipeBarrier<PIPE_V>();

    // Pre-build index maps once in Init()
    valid_ = TileLength(0,count_);
    rows_ = Height() * valid_;
    span_ = GateSpan();
    BuildDenseInputMaps();
    if(!compact_gates_) BuildDenseOutputMap<false>(output_index_, Height() * t_.tile_tokens);
    gate_output_map_ready_ = true;
  }

  __aicore__ inline void Process() {
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    for (uint64_t offset = 0, tile = 0; offset < count_; offset += valid_, ++tile) {
      const uint64_t first = start_ + offset;
      const uint64_t left = count_ - offset;
      valid_ = TileLength(tile,left);
      rows_ = Height() * valid_;
      span_ = GateSpan();
      matrix_=matrixBuffer_.Get<float>()[((2*Height())%8)*((valid_+15)/16*16)];

      WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
      auto index = input_index_.template ReinterpretCast<uint32_t>();
      // Two fixed maps share the same token-prefix ordering for every batch.
      uint32_t liveGates = span_;
      if constexpr (UseFactors) {
        liveGates = Height() * valid_;
        if (liveGates != span_) {
          Duplicate(gates_, 0.0f, 2 * span_);
          PipeBarrier<PIPE_V>();
        }
      }
      Gather(gates_,dense_input_,index,uint32_t{0},liveGates);
      Gather(gates_[span_],dense_input_,index[input_map_span_],uint32_t{0},liveGates);
      UnpackMatrixBlocks();
      PipeBarrier<PIPE_V>();

      if (offset + valid_ < count_) {
        const uint64_t next_first = first + valid_;
        const uint64_t next_left = count_ - (offset + valid_);
        const uint32_t next_valid =
            TileLength(tile+1,next_left);
        const DataCopyExtParams nextPacked{static_cast<uint16_t>(next_valid), 4 * InputWidth(), 0, 0, 0};
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        DataCopyPad(dense_input_, input_[next_first * InputWidth()], nextPacked, noPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
      }

      GatesAndRowMax();

      const uint32_t gates = Height() * valid_;
      const uint32_t matrix = Height() * gates;

      WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
      const DataCopyExtParams g{1, 4 * gates, 0, 0, 0};
      DataCopyPad(pre_[first * Height()], dense_output_, g);
      DataCopyPad(post_[first * Height()], dense_output_[dense_gate_cap_], g);

      SetMaskNorm();
      RunNarrowDenseIoCompleteRounds<FixedH, FixedI, UseFactors>(
          reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(dense_output_[2 * dense_gate_cap_].GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(ones_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
          Height(), (valid_ + 15) / 16 * 16, t_.eps, Iterations());
      ResetMask();

      PackMatrixBlocks();

      CopyOutComb(first, matrix, offset + valid_ < count_);
    }
  }

 private:
  __aicore__ inline uint32_t Height() const {
    if constexpr (FixedH != 0) return FixedH;
    return t_.h;
  }
  __aicore__ inline uint32_t Iterations() const {
    if constexpr (FixedI != 0) return FixedI;
    return t_.iters;
  }
  __aicore__ inline uint32_t GateSpan() const {
    if constexpr (UseFactors)
      return (Height() * valid_ + 7) / 8 * 8;
    return (compact_gates_ ? Height() : PaddedHeight()) * valid_;
  }
  __aicore__ inline uint32_t PaddedHeight() const { return 16; }
  __aicore__ inline uint32_t InputWidth() const {
    return Height() * (Height() + 2);
  }
  __aicore__ inline uint32_t TileLength(uint64_t tile,uint64_t left) const {
    // Preserve the original number of batches and total 16-lane rounded work.
    // A too-small average retains the original max-sized batches.
    if(tile_base_==0) return left<t_.tile_tokens?static_cast<uint32_t>(left):t_.tile_tokens;
    if(tile+1==tile_count_ && tile_tail_) return tile_base_+tile_tail_;
    return tile_base_+(tile<tile_small_count_?0:16);
  }
  template <HardEvent event>
  __aicore__ inline void Sync() {
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
  }

  __aicore__ inline void BuildDenseInputMaps() {
    auto token = partial_.template ReinterpretCast<int32_t>();
    CreateVecIndex(input_index_, int32_t{0}, input_map_span_);
    PipeBarrier<PIPE_V>();
    if(compact_gates_) {
      const int32_t reciprocal=static_cast<int32_t>(((1U<<20)+Height()-1)/Height());
      Muls(token,input_index_,reciprocal,input_map_span_);
      PipeBarrier<PIPE_V>();
      ShiftRight(token,token,int32_t{20},input_map_span_);
    } else ShiftRight(token, input_index_, int32_t{4}, input_map_span_);
    PipeBarrier<PIPE_V>();
    Muls(token, token, static_cast<int32_t>((InputWidth()+7)/8*8 - (compact_gates_ ? Height() : 16)), input_map_span_);
    PipeBarrier<PIPE_V>();
    Add(input_index_, input_index_, token, input_map_span_);
    PipeBarrier<PIPE_V>();
    Muls(input_index_, input_index_, int32_t{4}, input_map_span_);
    PipeBarrier<PIPE_V>();
    Adds(input_index_[input_map_span_], input_index_,static_cast<int32_t>(4*Height()),input_map_span_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GatesAndRowMax() {
    const uint32_t gatesCount = Height() * valid_;

    Muls(tmp_, gates_, -1.0f, 2 * span_);
    // The exp/div helper executes full vector repeats, including a compact tail.
    if(compact_gates_ && 2*span_%64)
      Duplicate(tmp_[2*span_],0.0f,64-2*span_%64);
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
    if(compact_gates_) {
      PlaneOp<7>(reinterpret_cast<__ubuf__ float *>(dense_output_.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(gates_.GetPhyAddr()),gatesCount);
      PlaneOp<7>(reinterpret_cast<__ubuf__ float *>(dense_output_[dense_gate_cap_].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(gates_[span_].GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float *>(gates_[span_].GetPhyAddr()),gatesCount);
    } else {
      auto gateIndex = output_index_.template ReinterpretCast<uint32_t>();
      Gather(dense_output_, gates_, gateIndex, uint32_t{0}, gatesCount);
      Gather(dense_output_[dense_gate_cap_], gates_[span_], gateIndex, uint32_t{0},gatesCount);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

    PlaneInnerReduce<true, UseFactors>(
        reinterpret_cast<__ubuf__ float *>(denom_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(dense_output_[2*dense_gate_cap_].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(matrix_.GetPhyAddr()),
        Height(),((valid_+15)/16*16));
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
        static_cast<int32_t>(((1U << 20) + Height() - 1) / Height());
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
        Muls(row,row,pitch*(1-static_cast<int32_t>(Height()*Height())),length);
        PipeBarrier<PIPE_V>();
        Muls(token,token,1-static_cast<int32_t>(Height())*pitch,length);
        Muls(dst,dst,static_cast<int32_t>(Height())*pitch,length);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, row, length);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, token, length);
      } else {
        Muls(row, row, static_cast<int32_t>(PaddedHeight() - Height()), length);
        PipeBarrier<PIPE_V>();
        Add(dst, dst, row, length);
      }
      PipeBarrier<PIPE_V>();
      Muls(dst, dst, int32_t{4}, length);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void UnpackMatrixBlocks() {
    const uint32_t pitch=(valid_+15)/16*16,inputPitch=(InputWidth()+7)/8*8;
    const uint32_t prefix=(2*Height())%8,firstFeature=2*Height()-prefix;
    const uint8_t repeats=(prefix+Height()*Height()+7)/8;
    const uint64_t inBase=reinterpret_cast<uint64_t>(dense_input_.GetPhyAddr());
    const uint64_t dstBase=reinterpret_cast<uint64_t>(matrixBuffer_.Get<float>().GetPhyAddr());
    uint64_t srcList[16],dstList[16];
    for(uint32_t tokenBase=0;tokenBase<pitch;tokenBase+=16) {
      const uint32_t remaining=valid_-tokenBase;
      if constexpr (UseFactors) {
        #pragma unroll
        for(uint32_t token=0;token<16;++token) {
          const uint32_t actual=token<remaining?token:0;
          srcList[token]=inBase+4*((tokenBase+actual)*inputPitch+firstFeature);
        }
        #pragma unroll
        for(uint32_t feature=0;feature<8;++feature) {
          dstList[2*feature]=dstBase+4*(feature*pitch+tokenBase);
          dstList[2*feature+1]=dstList[2*feature]+32;
        }
      } else {
        for(uint32_t token=0;token<16;++token) {
          const uint32_t actual=token<remaining?token:0;
          srcList[token]=inBase+4*((tokenBase+actual)*inputPitch+firstFeature);
        }
        for(uint32_t feature=0;feature<8;++feature) {
          dstList[2*feature]=dstBase+4*(feature*pitch+tokenBase);
          dstList[2*feature+1]=dstList[2*feature]+32;
        }
      }
      const TransDataTo5HDParams params{false,false,repeats,
            static_cast<uint16_t>(pitch),uint16_t{1}};
      TransDataTo5HD<float>(dstList,srcList,params);
    }
  }

  __aicore__ inline void PackMatrixBlocks() {
    const uint32_t pitch=(valid_+15)/16*16,features=Height()*Height();
    const uint32_t outPitch=(features+15)/16*16;
    if(outPitch>features) {
      Duplicate(matrix_[features*pitch],0.0f,(outPitch-features)*pitch);
      PipeBarrier<PIPE_V>();
    }
    const uint64_t matrixBase=reinterpret_cast<uint64_t>(matrix_.GetPhyAddr());
    const uint64_t outputBase=reinterpret_cast<uint64_t>(dense_output_[2*dense_gate_cap_].GetPhyAddr());
    uint64_t srcList[16],dstList[16];
    for(uint32_t tokenBase=0;tokenBase<pitch;tokenBase+=8) {
      if constexpr (UseFactors) {
        #pragma unroll
        for(uint32_t feature=0;feature<16;++feature)
          srcList[feature]=matrixBase+4*(feature*pitch+tokenBase);
        #pragma unroll
        for(uint32_t token=0;token<8;++token) {
          dstList[2*token]=outputBase+4*((tokenBase+token)*outPitch);
          dstList[2*token+1]=dstList[2*token]+32;
        }
      } else {
        for(uint32_t feature=0;feature<16;++feature)
          srcList[feature]=matrixBase+4*(feature*pitch+tokenBase);
        for(uint32_t token=0;token<8;++token) {
          dstList[2*token]=outputBase+4*((tokenBase+token)*outPitch);
          dstList[2*token+1]=dstList[2*token]+32;
        }
      }
      const TransDataTo5HDParams params{false,false,static_cast<uint8_t>(outPitch/16),
            uint16_t{2},static_cast<uint16_t>(2*pitch)};
      TransDataTo5HD<float>(dstList,srcList,params);
    }
  }

  __aicore__ inline void CopyOutComb(uint64_t first, uint32_t matrix,
                                     bool hasMoreTiles) {
    Sync<HardEvent::V_MTE3>();
    const uint32_t features=Height()*Height(),outPitch=(features+15)/16*16;
    const DataCopyExtParams m{static_cast<uint16_t>(valid_),4*features,
                             (outPitch-(features+7)/8*8)/8,0,0};
    DataCopyPad(comb_[first * Height() * Height()], dense_output_[2 * dense_gate_cap_], m);
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
  uint64_t start_, count_, tile_count_, tile_small_count_;
  uint32_t tile_base_, tile_tail_;
  uint32_t valid_, rows_, span_;
  uint32_t dense_gate_cap_, zero_offset_, indexed_valid_, input_gate_cap_,
      input_indexed_valid_, input_map_span_;
  bool gate_output_map_ready_, comb_write_pending_, compact_gates_;
};

}  // namespace f07
}  // namespace sinkhorn_norm

template <int Key = 0>
__global__ __aicore__ __attribute__((aligned(64))) void sinkhorn_f07_kernel(
    GM_ADDR mixes, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
    sinkhorn_norm::NarrowDenseArgs tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f07::NarrowP16DenseIoKernel<Key / 4, Key % 4> op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}


template <int = 0>
__global__ __aicore__ __attribute__((aligned(64), section(".sinkhorn_f07_factor_entry")))
void sinkhorn_f07_factor_kernel(GM_ADDR mixes, GM_ADDR pre, GM_ADDR post,
                                GM_ADDR comb, sinkhorn_norm::NarrowDenseArgs tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  sinkhorn_norm::f07::NarrowP16DenseIoKernel<0, 0, true> op;
  op.Init(mixes, pre, post, comb, tiling);
  op.Process();
}

#endif  // SINKHORN_NORM_KERNELS_F07_NARROW_P16_DENSE_IO_H_
