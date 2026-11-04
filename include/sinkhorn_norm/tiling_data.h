#ifndef SINKHORN_NORM_TILING_DATA_H_
#define SINKHORN_NORM_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

namespace sinkhorn_norm {

// Template-independent launch metadata. Contains no tensor data or pointers.
// The host and the Ascend C kernel must use this same 64-byte layout.
struct TilingData {
  float eps;
  std::uint32_t iters;
  std::uint64_t token_count;
  std::uint64_t tokens_per_core;
  std::uint32_t extra_core_count;
  std::uint32_t core_count;
  std::uint32_t h;
  std::uint32_t h_pad;
  std::uint32_t input_width;
  std::uint32_t tile_tokens;
  std::uint32_t input_bytes;
  std::uint32_t matrix_bytes;
  // Capacity of each of the two gate buffers, not their combined capacity.
  std::uint32_t gate_bytes;
  // Combined scratch capacity for the metadata-selected kernel layout.
  std::uint32_t scratch_bytes;
};

static_assert(sizeof(TilingData) == 64, "Unexpected TilingData ABI layout");
static_assert(offsetof(TilingData, eps) == 0 &&
                  offsetof(TilingData, iters) == 4,
              "Keep eps and iters adjacent at the front of TilingData");

using CompactFrontArgs = TilingData;

struct TinyThreeRowArgs {
  float eps;
  std::uint32_t iters;
  std::uint32_t tokens;
};
static_assert(sizeof(TinyThreeRowArgs) == 12, "Three-row argument ABI");

// Positive normalization denominators satisfy D <= H + (H+1)*eps initially and
// D <= H*(1+roundoff) + eps afterwards. This bound leaves a factor-of-two
// margin below 2^126, keeping 1/D normal in FP32 even after rounding.
inline bool HasNormalInverseEpsilon(std::uint64_t h, float eps) {
  return h >= 2 && h <= 128 && eps >= 0x1p-126f &&
         eps <= 0x1p125f / static_cast<float>(h + 1);
}

inline bool HasBoundedReductionEpsilon(std::uint64_t h, float eps) {
  return h >= 2 && h <= 128 && eps >= 0.0f &&
         eps <= 0x1p126f / static_cast<float>(h + 1);
}

// [F01_TinySplit / C2]: single-token, single-round H in [2, 8] split across 2 blocks.
inline bool UsesCompactSplitOutputs(const TilingData &t) {
  return t.h >= 2 && t.h <= 8 && t.h_pad == 8 && t.eps >= 0.0f &&
         t.token_count == 1 && t.iters == 1 &&
         t.token_count == t.core_count && t.tokens_per_core == 1 &&
         t.extra_core_count == 0 && t.tile_tokens == 1 &&
         t.matrix_bytes == 4 * t.h * t.h_pad &&
         t.gate_bytes == 4 * t.h_pad && t.scratch_bytes == 48 * t.h_pad;
}

// [F02_TinySingle / p1, p2]: single-token per core H in [2, 7], resident in static UB.
inline bool UsesTinySingle(const TilingData &t) {
  return t.h >= 2 && t.h <= 7 && t.h_pad == 8 && t.eps >= 0.0f &&
         t.iters >= 1 && t.iters <= 32 && t.core_count > 0 &&
         t.token_count == t.core_count && t.tokens_per_core == 1 &&
         t.extra_core_count == 0 && t.tile_tokens == 1 &&
         t.input_width == t.h * (t.h + 2) &&
         t.matrix_bytes == 4 * t.h * 8 && t.gate_bytes == 32 &&
         t.scratch_bytes == 384;
}

// Eight-token H3 matrix planes; the final chunk clips live tokens.
inline bool UsesTinyThreeRow(const TilingData &t) {
  return t.h == 3 && t.h_pad == 8 && t.eps >= 0.0f && t.iters >= 1 &&
         t.iters <= 32 && t.token_count > 0 && t.token_count <= 64 &&
         t.core_count == (t.token_count + 7) / 8 &&
         t.tokens_per_core == 8 && t.extra_core_count == 0 &&
         t.tile_tokens == 8 && t.input_width == 15 &&
         t.input_bytes == 512 && t.matrix_bytes == 512 &&
         t.gate_bytes == 64 && t.scratch_bytes == 16384;
}

// Small matrices use token planes across vector lanes.
inline std::uint32_t CompactPlaneCapacity(const TilingData &t) {
  const std::uint64_t h = t.h;
  const std::uint64_t gather_limit = (255 / (h * h)) * 64;
  const std::uint64_t width = h * (h + 2);
  const std::uint64_t plane_capacity = t.input_bytes / (4 * width);
  const bool plane_budget =
      t.gate_bytes == 0 && plane_capacity % 64 == 0 &&
      std::uint64_t{t.input_bytes} == 4 * width * plane_capacity &&
      t.matrix_bytes == t.input_bytes &&
      (std::uint64_t{t.scratch_bytes} == 4 * (h * h + h + 3) * plane_capacity ||
       std::uint64_t{t.scratch_bytes} == 4 * (h * h + h + 4) * plane_capacity);
  if (plane_budget) {
    return static_cast<std::uint32_t>(
        plane_capacity < gather_limit ? plane_capacity : gather_limit);
  }
  const std::uint64_t bytes = std::uint64_t{t.input_bytes} + t.matrix_bytes +
                              2 * std::uint64_t{t.gate_bytes} + t.scratch_bytes;
  const std::uint64_t lanes = 3 * h * h + 5 * h + 4;
  const std::uint64_t capacity = bytes / (4 * lanes) / 64 * 64;
  const std::uint64_t owner =
      std::uint64_t{t.tokens_per_core} + (t.extra_core_count != 0);
  const std::uint64_t owner_limit = (owner + 63) / 64 * 64;
  const std::uint64_t limit =
      gather_limit < owner_limit ? gather_limit : owner_limit;
  return static_cast<std::uint32_t>(capacity < limit ? capacity : limit);
}

inline bool UsesCompactTokenPlanes(const TilingData &t) {
  return t.h >= 2 && t.h <= 4 && t.h_pad == 8 && t.eps >= 0.0f &&
         t.eps <= 0x1.fffffep127f && t.iters >= 1 && t.tokens_per_core >= 64 &&
         t.tile_tokens >= 64 && CompactPlaneCapacity(t) >= 64;
}

inline std::uint64_t PackedPaddedRowPaddingBytes(std::uint64_t h,
                                                 std::uint64_t pitch) {
  const auto bytes = 4 * (pitch - h) * pitch;
  return (bytes + 511) / 512 * 512;
}

// [F04_NarrowSingle / C3]: single-token per core P24 narrow matrix (H in [17, 24]).
inline bool UsesPackedPaddedRows(const TilingData &t) {
  if (t.h < 9 || t.h > 24 || t.iters < 2 ||
      !HasBoundedReductionEpsilon(t.h, t.eps) ||
      t.token_count != t.core_count || t.tokens_per_core != 1 ||
      t.extra_core_count != 0 || t.tile_tokens != 1 ||
      t.h_pad != (t.h + 7) / 8 * 8)
    return false;
  const auto matrix = std::uint64_t{4} * t.h * t.h_pad;
  return t.matrix_bytes == matrix + PackedPaddedRowPaddingBytes(t.h, t.h_pad);
}

inline bool IsEstablishedInterleavedWorkload(std::uint64_t h,
                                             std::uint64_t iters,
                                             std::uint64_t tokens_per_core,
                                             float eps) {
  return (h >= 33 || (h >= 9 && iters >= 2 && tokens_per_core * h >= 64)) &&
         tokens_per_core >= 2 && HasNormalInverseEpsilon(h, eps);
}

inline bool IsFullSpanInterleavedWorkload(
    std::uint64_t h, std::uint64_t h_pad, std::uint64_t iters,
    std::uint64_t tokens_per_core, std::uint64_t extra_core_count, float eps) {
  return h >= 9 && h <= 32 && iters >= 2 && tokens_per_core >= 4 &&
         tokens_per_core * h < 64 && tokens_per_core * h_pad == 64 &&
         extra_core_count == 0 && HasNormalInverseEpsilon(h, eps);
}

inline bool IsShortNarrowBatchWorkload(
    std::uint64_t h, std::uint64_t h_pad, std::uint64_t iters,
    std::uint64_t tokens_per_core, std::uint64_t extra_core_count, float eps) {
  return h >= 9 && h <= 16 && h_pad == 16 && iters >= 2 && iters <= 32 &&
         tokens_per_core >= 3 && tokens_per_core * h < 64 &&
         HasNormalInverseEpsilon(h, eps) &&
         !IsFullSpanInterleavedWorkload(h, h_pad, iters, tokens_per_core,
                                       extra_core_count, eps);
}

inline bool UsesInterleavedLayout(const TilingData &t) {
  if (IsEstablishedInterleavedWorkload(t.h, t.iters, t.tokens_per_core, t.eps))
    return true;
  const bool short_batch =
      IsShortNarrowBatchWorkload(t.h, t.h_pad, t.iters, t.tokens_per_core,
                                 t.extra_core_count, t.eps) &&
      t.tile_tokens >= t.tokens_per_core + (t.extra_core_count != 0 ? 1 : 0);
  if (!short_batch) return false;
  const std::uint64_t rows = (std::uint64_t{t.h} * t.tile_tokens + 7) / 8 * 8;
  const std::uint64_t span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const auto tmp = 8 * rows > 6 * span ? 8 * rows : 6 * span;
  const auto ones = rows > 2 * span ? rows : 2 * span;
  return t.scratch_bytes == 4 * (tmp + 2 * rows + ones + span);
}

// [F05_NarrowP16ShortBatch / p3]: H=15, N=192, iter=20
inline bool UsesShortNarrowBatch(const TilingData &t) {
  return t.core_count != 0 && t.token_count >= t.core_count &&
         t.tokens_per_core == t.token_count / t.core_count &&
         t.extra_core_count == t.token_count % t.core_count &&
         t.tile_tokens == t.tokens_per_core + (t.extra_core_count != 0 ? 1 : 0) &&
         t.input_width == t.h * (t.h + 2) &&
         IsShortNarrowBatchWorkload(t.h, t.h_pad, t.iters, t.tokens_per_core,
                                   t.extra_core_count, t.eps) &&
         UsesInterleavedLayout(t);
}

// [F07_NarrowP16DenseIo / p8]: H in [9, 16), iters in [2, 3], dense multi-tile IO
inline bool IsDenseNarrowIoWorkload(std::uint64_t h, std::uint64_t pitch,
                                    std::uint64_t iters, std::uint64_t owner,
                                    std::uint64_t tile, float eps) {
  return h >= 9 && h < 16 && pitch == 16 && iters >= 2 && iters <= 3 &&
         tile != 0 && tile <= 64 && h * tile >= 64 && owner >= 2 * tile &&
         IsEstablishedInterleavedWorkload(h, iters, owner, eps);
}

inline std::uint64_t DenseNarrowInputBytes(std::uint64_t h,std::uint64_t tile) {
  return 4*((h*(h+2)+7)/8*8)*tile;
}
inline std::uint64_t DenseNarrowMatrixBytes(std::uint64_t h,std::uint64_t tile) {
  return 4*((2*h)%8+(h*h+15)/16*16)*tile;
}
inline std::uint64_t DenseNarrowActualUbBytes(std::uint64_t h,std::uint64_t tile) {
  const auto span=16*tile,gate=(h*tile+63)/64*64;
  const auto matrix=(h*h*tile+63)/64*64,output=(h*h+15)/16*16*tile;
  const auto outCap=matrix>output?matrix:output;
  return DenseNarrowMatrixBytes(h,tile)+3*8*span+256+4*h*tile+
         DenseNarrowInputBytes(h,tile)+4*(2*gate+outCap)+4*gate+8*span;
}
inline std::uint64_t DenseNarrowScratchBytes(std::uint64_t h,std::uint64_t tile) {
  return DenseNarrowActualUbBytes(h,tile)-DenseNarrowInputBytes(h,tile)-
         DenseNarrowMatrixBytes(h,tile)-8*16*tile;
}

inline bool UsesDenseNarrowIo(const TilingData &t) {
  if (!IsDenseNarrowIoWorkload(t.h, t.h_pad, t.iters, t.tokens_per_core,
                              t.tile_tokens, t.eps) ||
      t.core_count == 0 || t.tokens_per_core != t.token_count / t.core_count ||
      t.extra_core_count != t.token_count % t.core_count ||
      t.tile_tokens > t.tokens_per_core + (t.extra_core_count != 0 ? 1 : 0) ||
      t.input_width != t.h * (t.h + 2))
    return false;
  return t.input_bytes==DenseNarrowInputBytes(t.h,t.tile_tokens) &&
         t.matrix_bytes==DenseNarrowMatrixBytes(t.h,t.tile_tokens) &&
         t.gate_bytes==4*16*t.tile_tokens &&
         t.scratch_bytes==DenseNarrowScratchBytes(t.h,t.tile_tokens);
}

// [F09_MediumDenseIo / C5]: medium non-aligned H in [33, 56] with padded rows & dense IO.
inline std::uint64_t NarrowGroupedRowPaddingBytes(std::uint64_t h,
                                                  std::uint64_t h_pad,
                                                  std::uint64_t tile_tokens) {
  if (h < 33 || h > 56 || h_pad >= 64 || tile_tokens == 0) return 0;
  const auto rows = h * tile_tokens;
  const auto padded_rows = (rows + 7) / 8 * 8;
  const auto tail_bytes = 4 * (padded_rows - rows) * h_pad;
  return (tail_bytes + 511) / 512 * 512;
}

inline bool UsesPaddedGroupedRows(const TilingData &t) {
  const auto extra = NarrowGroupedRowPaddingBytes(t.h, t.h_pad, t.tile_tokens);
  const auto matrix = std::uint64_t{4} * t.h * t.tile_tokens * t.h_pad;
  return extra != 0 && UsesInterleavedLayout(t) &&
         t.matrix_bytes == matrix + extra;
}

inline bool IsDenseMediumIoWorkload(std::uint64_t h, std::uint64_t pitch,
                                    std::uint64_t iters, std::uint64_t owner,
                                    std::uint64_t tile, float eps) {
  return h >= 33 && h <= 56 && pitch < 64 && tile > 0 && eps <= 1.0f &&
         h * h * h * tile < (1U << 20) &&
         NarrowGroupedRowPaddingBytes(h, pitch, tile) != 0 &&
         IsEstablishedInterleavedWorkload(h, iters, owner, eps);
}

inline std::uint64_t DenseMediumIoElements(std::uint64_t h,
                                           std::uint64_t pitch,
                                           std::uint64_t tile) {
  const auto physical = (h + 2) * pitch * tile;
  const auto gate = (h * tile + 63) / 64 * 64;
  const auto packed = 2 * gate + h * h * tile;
  return ((physical > packed ? physical : packed) + 63) / 64 * 64;
}

inline std::uint64_t DenseMediumIoExtraBytes(std::uint64_t h,
                                             std::uint64_t pitch,
                                             std::uint64_t tile) {
  return 8 * DenseMediumIoElements(h, pitch, tile);
}

inline bool UsesDenseMediumIo(const TilingData &t) {
  if (!IsDenseMediumIoWorkload(t.h, t.h_pad, t.iters, t.tokens_per_core,
                               t.tile_tokens, t.eps) ||
      !UsesPaddedGroupedRows(t))
    return false;
  const auto rp = (std::uint64_t{t.h} * t.tile_tokens + 7) / 8 * 8;
  const auto span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const auto tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;
  const auto ones = rp > 2 * span ? rp : 2 * span;
  return t.scratch_bytes ==
         4 * (tmp + 2 * rp + ones + span) +
             DenseMediumIoExtraBytes(t.h, t.h_pad, t.tile_tokens);
}

// [F12_WideBlocked / C4]: 8-aligned non-32-aligned wide H in [72, 120] blocked layout.
inline bool UsesWideBlockedRounds(const TilingData &t) {
  if (!UsesInterleavedLayout(t) || UsesPaddedGroupedRows(t) || t.h <= 64 ||
      t.h > 120 || t.h % 8 != 0 || t.h_pad != t.h || t.h_pad >= 128 ||
      (t.h_pad / 8) % 4 == 0 || t.iters < 2)
    return false;
  const auto rows = std::uint64_t{t.h} * t.tile_tokens;
  const auto rp = (rows + 7) / 8 * 8;
  const auto span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const auto col_ws = ((std::uint64_t{t.h} + 3) / 4) * span;
  const auto workspace = rows * 16 > col_ws ? rows * 16 : col_ws;
  const auto tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;
  const auto ones = rp > 2 * span ? rp : 2 * span;
  return t.matrix_bytes == 4 * rows * t.h_pad &&
         t.input_bytes >= 4 * workspace && t.gate_bytes == 4 * span &&
         t.scratch_bytes == 4 * (tmp + 2 * rp + ones + span);
}

// [F14_WideRotating128 / C1]: H=128 rotating 5-node quarter bank skew layout.
inline bool IsQuarterBankSkewWorkload(std::uint64_t h, std::uint64_t pitch,
                                      std::uint64_t iters,
                                      std::uint64_t tile_tokens, float eps) {
  return h == 128 && pitch == 128 && iters >= 8 && tile_tokens >= 1 &&
         tile_tokens <= 2 && HasNormalInverseEpsilon(h, eps);
}

inline bool UsesRotatingQuarters(const TilingData &t) {
  if (!IsQuarterBankSkewWorkload(t.h, t.h_pad, t.iters, t.tile_tokens, t.eps) ||
      !UsesInterleavedLayout(t))
    return false;
  const std::uint64_t rows = std::uint64_t{t.h} * t.tile_tokens;
  const std::uint64_t span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const auto rotating_scratch = 4 * (17 * rows + 3 * span);
  return t.matrix_bytes == 132352 && t.input_bytes == 33024 &&
         t.gate_bytes == 4 * span && t.scratch_bytes == rotating_scratch;
}

// [F11_MediumAligned64 / p6]: H=64, N=160, iter=20 aligned 64-lane ping-pong half tree
inline bool IsAlignedMediumWorkload(std::uint64_t h, std::uint64_t pitch,
                                    std::uint64_t iters, float eps) {
  return h == 64 && pitch == 64 && iters >= 8 && HasNormalInverseEpsilon(h, eps);
}

inline bool UsesAlignedMedium(const TilingData &t) {
  if (!IsAlignedMediumWorkload(t.h, t.h_pad, t.iters, t.eps) ||
      !UsesInterleavedLayout(t))
    return false;
  const uint32_t span = t.h_pad * t.tile_tokens;
  const uint32_t phased_scratch = 13 * span * sizeof(float) + 256;
  return t.scratch_bytes == phased_scratch ||
         t.scratch_bytes == phased_scratch + t.matrix_bytes + 512;
}

// [F13_WideQuarterStream / p9]: H=100, N=768, iter=32 quarter-stream capacity 5
struct CompactFiveLayout {
  std::uint64_t slot, input, gate, scratch, quarter_rows, work, prefix;
};

inline CompactFiveLayout MakeCompactFiveLayout(std::uint64_t h,
                                              std::uint64_t pitch,
                                              std::uint64_t tile) {
  const auto q = (h + 3) / 4;
  const auto qr = (q * tile + 7) / 8 * 8;
  const auto rows = 4 * qr, span = pitch * tile;
  const auto slot = (4 * qr * pitch + 511) / 512 * 512 + 256;
  const auto base = 4 * slot + 8 * span;
  const auto p0 = (512 - base % 512) % 512;
  const auto p1 = (256 + 512 - base % 512) % 512;
  auto input = 4 * qr * pitch + (p0 > p1 ? p0 : p1);
  input += (256 + 512 - (base + input) % 512) % 512;
  const auto tmp = 8 * rows;
  const auto prefix = (512 - (256 + 4 * tmp + 4 * rows) % 512) % 512 / 4;
  std::uint64_t work = 0;
  for (std::uint64_t v = 1; v <= tile; ++v) {
    const auto r = (q * v + 7) / 8 * 8;
    const auto n = 8 * r + (r % 16 != 0 ? 64 : 0);
    if (n > work) work = n;
  }
  const auto ones = rows > 2 * span ? rows : 2 * span;
  return {slot, input, 4 * span,
          4 * (tmp + rows + prefix + work + ones + span), qr, work, prefix};
}

inline bool IsCompactFiveWorkload(std::uint64_t h, std::uint64_t pitch,
                                  std::uint64_t iters, float eps) {
  return h > 64 && h <= 120 && pitch == (h + 7) / 8 * 8 && pitch < 128 &&
         (pitch / 8) % 4 != 0 && iters >= 2 &&
         HasNormalInverseEpsilon(h, eps) && h % 8 != 0;
}

inline bool UsesCompactFive(const TilingData &t) {
  if (!IsCompactFiveWorkload(t.h, t.h_pad, t.iters, t.eps) ||
      !UsesInterleavedLayout(t) || t.tile_tokens == 0)
    return false;
  const auto x = MakeCompactFiveLayout(t.h, t.h_pad, t.tile_tokens);
  return t.matrix_bytes == 4 * x.slot && t.input_bytes == x.input &&
         t.gate_bytes == x.gate && t.scratch_bytes == x.scratch;
}

// [F08_NarrowP24Recurrent / p4]: H in [17, 24] recurrent rounds
inline bool UsesGroupedNarrowRound(std::uint32_t blocks, const TilingData &t) {
  if (t.h < 9 || t.h > 24 || t.h_pad != (t.h + 7) / 8 * 8 || t.iters < 2 ||
      t.iters > 32 || !(HasNormalInverseEpsilon(t.h, t.eps)) ||
      !UsesInterleavedLayout(t) || t.core_count == 0 ||
      blocks != t.core_count || t.token_count < t.core_count ||
      t.tokens_per_core != t.token_count / t.core_count ||
      t.extra_core_count != t.token_count % t.core_count ||
      t.tile_tokens < 2 || t.tile_tokens > 510 / t.h_pad ||
      t.tile_tokens > t.tokens_per_core + (t.extra_core_count != 0 ? 1 : 0) ||
      t.input_width != t.h * (t.h + 2))
    return false;
  const std::uint64_t rows = std::uint64_t{t.h} * t.tile_tokens;
  if (rows < 64) return false;
  const std::uint64_t rp = (rows + 7) / 8 * 8;
  const std::uint64_t span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const std::uint64_t workspace = 8 * rows > 4 * span ? 8 * rows : 4 * span;
  const std::uint64_t tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;
  const std::uint64_t ones = rp > 2 * span ? rp : 2 * span;
  return t.matrix_bytes == 4 * rows * t.h_pad && t.gate_bytes == 4 * span &&
         t.input_bytes == 4 * workspace &&
         t.scratch_bytes == 4 * (tmp + 2 * rp + ones + span);
}

// [F10_MediumCompact / p5]: H in [33, 64], compact workspace tile expansion
inline bool UsesCompactMediumWorkspace(const TilingData &t) {
  if (t.h < 33 || t.h_pad >= 64 || !UsesInterleavedLayout(t)) return false;
  const std::uint64_t rp = (std::uint64_t{t.h} * t.tile_tokens + 7) / 8 * 8;
  const std::uint64_t span = std::uint64_t{t.h_pad} * t.tile_tokens;
  const auto tmp = 8 * rp > 6 * span ? 8 * rp : 6 * span;
  const auto ones = rp > 2 * span ? rp : 2 * span;
  return t.scratch_bytes == 4 * (tmp + rp + ones + span);
}

}  // namespace sinkhorn_norm

#endif  // SINKHORN_NORM_TILING_DATA_H_
