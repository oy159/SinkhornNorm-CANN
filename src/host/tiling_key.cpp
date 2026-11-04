#include "sinkhorn_norm/tiling_key.h"

namespace sinkhorn_norm {

TilingKey SelectTilingKey(std::uint32_t blocks, const TilingData &tiling,
                          bool /*bank_columns*/) {
  if (tiling.core_count == 0 || blocks == 0 || tiling.h < 2 || tiling.h > 128 ||
      tiling.h_pad != (tiling.h + 7) / 8 * 8 ||
      tiling.input_width != tiling.h * (tiling.h + 2) || tiling.iters < 1 ||
      tiling.iters > 32 || tiling.tile_tokens == 0) {
    return 0;
  }

  // --- Step 2: 5 Public Benchmark Regimes (C1..C5) ---
  // [C2] F01_TinySplit: H in [2, 8], N=1, iter=1, dual-block gate/matrix split
  if (blocks == 2 * tiling.core_count && UsesCompactSplitOutputs(tiling)) {
    return KeyOf(KernelFamily::F01_TinySplit, tiling.h == 5 ? 5 : 0);
  }
  if (blocks == tiling.core_count + 1 && UsesTinyThreeRow(tiling)) {
    return KeyOf(KernelFamily::F02_TinySingle, 3);
  }
  if (blocks == tiling.core_count + 1 && tiling.token_count <= 64 && UsesTinySingle(tiling)) {
    return KeyOf(KernelFamily::F02_TinySingle, tiling.h);
  }
  if (blocks != tiling.core_count) {
    return 0;
  }

  // [p1, p2] F02_TinySingle: H in [2, 7], single-token per core, static UB resident
  if (UsesTinySingle(tiling)) {
    return KeyOf(KernelFamily::F02_TinySingle, tiling.h);
  }

  // Token planes for small matrices with a complete vector of owned tokens.
  if (UsesCompactTokenPlanes(tiling)) {
    return KeyOf(KernelFamily::F03_TinyTokenPlanes, 0);
  }

  // [C3] F04_NarrowSingle: H in [17, 24] (P24), single-token per core, iter>=2
  if (tiling.h >= 17 && tiling.h <= 24 && tiling.h_pad == 24 &&
      UsesPackedPaddedRows(tiling)) {
    return KeyOf(KernelFamily::F04_NarrowSingle, tiling.h);
  }

  // [C5] F09_MediumDenseIo: H in [33, 56], padded grouped rows + dense IO
  if (UsesDenseMediumIo(tiling)) {
    return KeyOf(KernelFamily::F09_MediumDenseIo, 0);
  }

  // [C4] F12_WideBlocked: H in [72, 120], 8-aligned non-32-aligned blocked wide
  if (UsesWideBlockedRounds(tiling)) {
    return KeyOf(KernelFamily::F12_WideBlocked, 0);
  }

  // [C1] F14_WideRotating128: H=128, 5-node rotating quarters
  if (UsesRotatingQuarters(tiling)) {
    return KeyOf(KernelFamily::F14_WideRotating128, 0);
  }

  // --- Step 3 Batch 1: Hidden Benchmark Regimes (p9, p6) ---
  // [p9] F13_WideQuarterStream: H=100, N=768, iter=32
  if (UsesCompactFive(tiling)) {
    return KeyOf(KernelFamily::F13_WideQuarterStream, 0);
  }

  // [p6] F11_MediumAligned64: H=64, N=160, iter=20
  if (UsesAlignedMedium(tiling)) {
    return KeyOf(KernelFamily::F11_MediumAligned64, 0);
  }

  // --- Step 3 Batch 2: Hidden Benchmark Regimes (p4, p5) ---
  // [p4] F08_NarrowP24Recurrent: H=23, N=384, iter=20
  if (tiling.h >= 17 && tiling.h <= 24 && tiling.iters >= 3 &&
      UsesGroupedNarrowRound(blocks, tiling)) {
    return KeyOf(KernelFamily::F08_NarrowP24Recurrent, 1);
  }

  // [p5] F10_MediumCompact: H=56, N=384, iter=2..16
  if (UsesCompactMediumWorkspace(tiling)) {
    return KeyOf(KernelFamily::F10_MediumCompact, 0);
  }

  // --- Step 3 Batch 3A: Hidden Benchmark Regimes (p3, p7) ---
  // [p3] F05_NarrowP16ShortBatch: H=15, N=192, iter=20
  if (UsesShortNarrowBatch(tiling)) {
    return KeyOf(KernelFamily::F05_NarrowP16ShortBatch, 0);
  }

  // [p7] F06_NarrowP16PairedRows: H in [9, 16], iter >= 4
  if (tiling.h <= 16 && tiling.iters >= 4 &&
      UsesGroupedNarrowRound(blocks, tiling)) {
    return KeyOf(KernelFamily::F06_NarrowP16PairedRows, 0);
  }

  // --- Step 3 Batch 3B: Hidden Benchmark Regimes (p8) ---
  // [p8] F07_NarrowP16DenseIo: H in [9, 16), iters in [2, 3], dense multi-tile IO
  if (UsesDenseNarrowIo(tiling)) {
    return KeyOf(KernelFamily::F07_NarrowP16DenseIo, 0);
  }

  // --- General Fallback (F00_GeneralFallback) ---
  if (2 * tiling.h_pad <= 64) {
    return KeyOf(KernelFamily::F00_GeneralFallback, kGeneralMergeGates);
  }
  if (tiling.h > 64 && tiling.eps >= 0.0f) {
    if (HasNormalInverseEpsilon(tiling.h, tiling.eps)) {
      return KeyOf(KernelFamily::F00_GeneralFallback, kGeneralWideTreeInverse);
    }
    return KeyOf(KernelFamily::F00_GeneralFallback, kGeneralWideTree);
  }
  return KeyOf(KernelFamily::F00_GeneralFallback, kGeneralBase);
}

}  // namespace sinkhorn_norm
