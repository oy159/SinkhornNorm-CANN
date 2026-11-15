#ifndef SINKHORN_NORM_TILING_KEY_H_
#define SINKHORN_NORM_TILING_KEY_H_

#include "sinkhorn_norm/tiling_data.h"
#include <cstdint>

namespace sinkhorn_norm {

// Upper 32 bits: algorithm family; lower 32 bits: compiled variant.
using TilingKey = std::uint64_t;

enum class KernelFamily : std::uint32_t {
  F00_GeneralFallback = 1,

  F01_TinySplit = 2,
  F02_TinySingle = 3,
  F03_TinyTokenPlanes = 4,

  F04_NarrowSingle = 5,
  F05_NarrowP16ShortBatch = 6,
  F06_NarrowP16PairedRows = 7,
  F07_NarrowP16DenseIo = 8,
  F08_NarrowP24Recurrent = 9,

  F09_MediumDenseIo = 10,
  F10_MediumCompact = 11,
  F11_MediumAligned64 = 12,

  F12_WideBlocked = 13,
  F13_WideQuarterStream = 14,
  F14_WideRotating128 = 15,
};

constexpr TilingKey KeyOf(KernelFamily family, std::uint32_t variant = 0) {
  return (static_cast<TilingKey>(family) << 32) | variant;
}

constexpr KernelFamily FamilyOf(TilingKey key) {
  return static_cast<KernelFamily>(key >> 32);
}

constexpr std::uint32_t VariantOf(TilingKey key) {
  return static_cast<std::uint32_t>(key);
}

// Variant flags for F00_GeneralFallback:
// bit 0 (1): MergeGates (2 * h_pad <= 64)
// bit 1 (2): TreeColumns (h > 64 && eps >= 0)
// bit 2 (4): ReuseInverse (h > 64 && HasNormalInverseEpsilon(h, eps))
enum GeneralFallbackVariant : std::uint32_t {
  kGeneralBase = 0,
  kGeneralMergeGates = 1,
  kGeneralWideTree = 2,
  kGeneralWideTreeInverse = 6,
};

inline constexpr std::uint32_t kNarrowDenseFactorVariant = 1;

// Reads only shape/iteration/eps metadata, ownership and allocated capacities.
// Returns one exact compiled key; zero is reserved for no matching helper.
TilingKey SelectTilingKey(std::uint32_t blocks, const TilingData &tiling,
                          bool bank_columns = false);

}  // namespace sinkhorn_norm

// All compiled production choices in deterministic .text emission order.
// Medium dense first; retain all dispatch choices.
#define SINKHORN_TILING_KEY_CASES(X) \
  X(F09_MediumDenseIo, 0)            \
  X(F12_WideBlocked, 0)              \
  X(F04_NarrowSingle, 22)            \
  X(F01_TinySplit, 5)                \
  X(F13_WideQuarterStream, 0)        \
  X(F11_MediumAligned64, 0)          \
  X(F08_NarrowP24Recurrent, 1)       \
  X(F10_MediumCompact, 0)            \
  X(F06_NarrowP16PairedRows, 0)      \
  X(F05_NarrowP16ShortBatch, 0)      \
  X(F07_NarrowP16DenseIo, 0)        \
  X(F07_NarrowP16DenseIo, 38)       \
  X(F07_NarrowP16DenseIo, 39)       \
  X(F07_NarrowP16DenseIo, 42)       \
  X(F07_NarrowP16DenseIo, 43)       \
  X(F07_NarrowP16DenseIo, 46)       \
  X(F07_NarrowP16DenseIo, 47)       \
  X(F07_NarrowP16DenseIo, 50)       \
  X(F07_NarrowP16DenseIo, 51)       \
  X(F07_NarrowP16DenseIo, 54)       \
  X(F07_NarrowP16DenseIo, 55)       \
  X(F07_NarrowP16DenseIo, 58)       \
  X(F07_NarrowP16DenseIo, 59)       \
  X(F07_NarrowP16DenseIo, 62)       \
  X(F07_NarrowP16DenseIo, 63)       \
  X(F02_TinySingle, 3)               \
  X(F02_TinySingle, 7)               \
  X(F02_TinySingle, 2)               \
  X(F02_TinySingle, 4)               \
  X(F02_TinySingle, 5)               \
  X(F02_TinySingle, 6)               \
  X(F01_TinySplit, 0)                \
  X(F04_NarrowSingle, 17)            \
  X(F04_NarrowSingle, 18)            \
  X(F04_NarrowSingle, 19)            \
  X(F04_NarrowSingle, 20)            \
  X(F04_NarrowSingle, 21)            \
  X(F04_NarrowSingle, 23)            \
  X(F04_NarrowSingle, 24)            \
  X(F00_GeneralFallback, 0)          \
  X(F00_GeneralFallback, 1)          \
  X(F00_GeneralFallback, 2)          \
  X(F00_GeneralFallback, 6)          \
  X(F03_TinyTokenPlanes, 0)         \
  X(F14_WideRotating128, 0)         \
  X(F07_NarrowP16DenseIo, sinkhorn_norm::kNarrowDenseFactorVariant)

#endif  // SINKHORN_NORM_TILING_KEY_H_
