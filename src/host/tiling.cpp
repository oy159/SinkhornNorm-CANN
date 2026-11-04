#include "sinkhorn_norm/tiling.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sinkhorn_norm {
namespace {

constexpr std::uint64_t kAlignmentBytes = 32;
constexpr std::uint64_t kFloatBytes = 4;
constexpr std::uint64_t kReservedBytes = 8192;
constexpr std::uint64_t kScratchVectors = 12;

constexpr std::uint64_t Align32(std::uint64_t bytes) {
  return (bytes + kAlignmentBytes - 1) / kAlignmentBytes * kAlignmentBytes;
}

struct BufferSizes {
  std::uint64_t input;
  std::uint64_t matrix;
  std::uint64_t gate;
  std::uint64_t scratch;

  std::uint64_t Total() const {
    return input + matrix + 2 * gate + scratch + kReservedBytes;
  }

  bool FitsMetadata() const {
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
    return input <= limit && matrix <= limit && gate <= limit &&
           scratch <= limit;
  }
};

BufferSizes GetGeneralBufferSizes(std::uint64_t tile_tokens, std::uint64_t h,
                                  std::uint64_t h_pad,
                                  std::uint64_t input_width,
                                  bool packed_div = false) {
  auto input_reserve = Align32(kFloatBytes * tile_tokens * input_width);
  if (packed_div) {
    const auto coefficient_bytes = kFloatBytes * h_pad * h_pad;
    input_reserve = std::max(input_reserve, coefficient_bytes);
  }
  return {input_reserve,
          Align32(kFloatBytes * tile_tokens * h * h_pad),
          Align32(kFloatBytes * tile_tokens * h_pad),
          kScratchVectors * kFloatBytes * h_pad};
}

BufferSizes GetInterleavedBufferSizes(std::uint64_t tile_tokens,
                                      std::uint64_t h, std::uint64_t h_pad,
                                      bool compact_medium = false) {
  const auto rows = tile_tokens * h;
  const auto row_pad = (rows + 7) / 8 * 8;
  const auto span = tile_tokens * h_pad;
  const auto column_rows = h >= 33 ? (h + 3) / 4 : 4;
  const auto column_workspace = column_rows * span;
  const auto workspace =
      std::max(rows * (h > 64 ? 16 : 8), column_workspace);
  const auto temporary = std::max(8 * row_pad, 6 * span);
  const auto scratch = temporary + (compact_medium ? 1 : 2) * row_pad +
                       std::max(row_pad, 2 * span) + span;
  const auto matrix_bytes = 4 * rows * h_pad;
  const auto gate_bytes = 4 * span;
  return {4 * workspace, matrix_bytes, gate_bytes, 4 * scratch};
}

}  // namespace

bool MakeTiling(const Problem &problem, const PlatformResources &platform,
                TilingData &tiling, OutputShapes &output_shapes,
                std::string &error, bool *bank_columns) {
  if (bank_columns != nullptr) *bank_columns = false;
  tiling = {};
  output_shapes = {};
  error.clear();

  auto reject = [&error](const char *message) {
    error = message;
    return false;
  };

  if (problem.rank != 3) {
    return reject("mixes must have rank 3");
  }
  if (problem.dtype != DType::Float32) {
    return reject("mixes must use float32");
  }
  if (problem.shape[0] < 1 || problem.shape[1] < 1 ||
      static_cast<std::uint64_t>(problem.shape[0]) * problem.shape[1] > 1048576ULL) {
    return reject("Total tokens B * S must be in [1, 1048576]");
  }
  if (problem.h < 2 || problem.h > 128) {
    return reject("hc_mult must be in [2, 128]");
  }
  if (problem.iters < 1 || problem.iters > 32) {
    return reject("hc_sinkhorn_iters must be in [1, 32]");
  }
  if (!std::isfinite(problem.eps)) {
    return reject("eps must be finite; no positivity restriction is imposed");
  }

  const auto h = static_cast<std::uint64_t>(problem.h);
  const auto input_width = h * (h + 2);
  if (problem.shape[2] != static_cast<std::int64_t>(input_width)) {
    return reject("mixes last dimension must equal H * (H + 2)");
  }
  if (platform.aiv_cores == 0) {
    return reject("platform must provide at least one AIV core");
  }

  const auto token_count = static_cast<std::uint64_t>(problem.shape[0]) *
                           static_cast<std::uint64_t>(problem.shape[1]);
  const auto core_count =
      std::min(token_count, static_cast<std::uint64_t>(platform.aiv_cores));
  const auto tokens_per_core = token_count / core_count;
  const auto extra_core_count = token_count % core_count;
  const auto max_core_tokens = tokens_per_core + (extra_core_count != 0 ? 1 : 0);
  const auto h_pad = (h + 7) / 8 * 8;

  // Check minimum 1-token general capacity first.
  const bool narrow_single_p24 =
      h >= 17 && h <= 24 && h_pad == 24 && problem.iters >= 2 &&
      HasBoundedReductionEpsilon(h, problem.eps) && token_count == core_count;
  const auto minimum_general =
      GetGeneralBufferSizes(1, h, h_pad, input_width, narrow_single_p24);
  if (minimum_general.Total() > platform.ub_bytes) {
    return reject("UB capacity cannot hold one complete token, scratch, and "
                  "8192 reserved bytes");
  }

  // Step 2: Check whether this problem targets one of the 3 interleaved public
  // regimes (F09_MediumDenseIo / C5, F12_WideBlocked / C4, F14_WideRotating128 / C1).
  bool use_interleaved = false;
  std::uint64_t selected_tile = 1;
  BufferSizes selected_sizes = minimum_general;

  if (IsEstablishedInterleavedWorkload(h, problem.iters, tokens_per_core,
                                       problem.eps) &&
      GetInterleavedBufferSizes(1, h, h_pad).Total() <= platform.ub_bytes) {
    std::uint64_t lower = 1;
    std::uint64_t upper = std::min(max_core_tokens, 510 / h_pad);
    while (lower < upper) {
      const auto candidate = lower + (upper - lower + 1) / 2;
      const auto sizes = GetInterleavedBufferSizes(candidate, h, h_pad);
      if (sizes.Total() <= platform.ub_bytes && sizes.FitsMetadata()) {
        lower = candidate;
      } else {
        upper = candidate - 1;
      }
    }
    auto sizes = GetInterleavedBufferSizes(lower, h, h_pad);

    // Check if compact medium would supersede dense medium IO (e.g., p5 H=56).
    bool compact_medium = false;
    if (h >= 33 && h_pad < 64 && problem.iters >= 2 && problem.eps <= 1.0f) {
      auto compact_limit = std::min(max_core_tokens, 2040 / h_pad);
      auto compact_tile = lower;
      while (compact_tile < compact_limit) {
        const auto candidate =
            compact_tile + (compact_limit - compact_tile + 1) / 2;
        const auto compact_sizes =
            GetInterleavedBufferSizes(candidate, h, h_pad, true);
        if (compact_sizes.Total() <= platform.ub_bytes &&
            compact_sizes.FitsMetadata()) {
          compact_tile = candidate;
        } else {
          compact_limit = candidate - 1;
        }
      }
      if ((max_core_tokens + compact_tile - 1) / compact_tile <
          (max_core_tokens + lower - 1) / lower) {
        lower = compact_tile;
        compact_medium = true;
      }
    }

    // 1. F12_WideBlocked (C4: H=88, N=96, iter=20)
    if (h > 64 && h <= 120 && h % 8 == 0 && h_pad == h && h_pad < 128 &&
        (h_pad / 8) % 4 != 0 && problem.iters >= 2) {
      use_interleaved = true;
      selected_tile = lower;
      selected_sizes = sizes;
      // Stage a complete token only when the existing tile's UB budget permits
      // it. Otherwise the blocked kernel retains its capacity-bounded chunks.
      auto full_token_sizes = sizes;
      full_token_sizes.input = std::max(full_token_sizes.input, 4 * h * h);
      if (full_token_sizes.Total() <= platform.ub_bytes &&
          full_token_sizes.FitsMetadata()) {
        selected_sizes = full_token_sizes;
      }
      auto pingpong_sizes = selected_sizes;
      const auto missing = 8 * h * h > pingpong_sizes.input
                               ? 8 * h * h - pingpong_sizes.input : 0;
      pingpong_sizes.input += (missing + 511) / 512 * 512;
      if (pingpong_sizes.Total() <= platform.ub_bytes &&
          pingpong_sizes.FitsMetadata()) {
        selected_sizes = pingpong_sizes;
      }
    }
    // 2. F14_WideRotating128 (C1: H=128, N=896, iter=15)
    else if (IsQuarterBankSkewWorkload(h, h_pad, problem.iters, lower,
                                       problem.eps)) {
      auto skewed_sizes = sizes;
      skewed_sizes.matrix += 256;
      skewed_sizes.input += 256;
      if (skewed_sizes.Total() <= platform.ub_bytes &&
          skewed_sizes.FitsMetadata()) {
        auto rotating_sizes = skewed_sizes;
        rotating_sizes.matrix = 132352;
        rotating_sizes.input = 33024;
        const auto rows = h * lower, span = h_pad * lower;
        rotating_sizes.scratch = 4 * (17 * rows + 3 * span);
        const bool amortized =
            problem.iters >= 12 &&
            tokens_per_core >= static_cast<std::uint64_t>(
                                   (128 + problem.iters - 2) /
                                   (problem.iters - 1));
        if (amortized && rotating_sizes.Total() <= platform.ub_bytes &&
            rotating_sizes.FitsMetadata()) {
          use_interleaved = true;
          selected_tile = lower;
          selected_sizes = rotating_sizes;
        }
      }
    }
    // 3. F09_MediumDenseIo (C5: H=34, N=256, iter=20)
    else if (!compact_medium &&
             IsDenseMediumIoWorkload(h, h_pad, problem.iters, tokens_per_core,
                                     lower, problem.eps)) {
      const auto extra_pad = NarrowGroupedRowPaddingBytes(h, h_pad, lower);
      const auto extra_io = DenseMediumIoExtraBytes(h, h_pad, lower);
      auto dense_sizes = sizes;
      dense_sizes.matrix += extra_pad;
      dense_sizes.scratch += extra_io;
      if (dense_sizes.Total() <= platform.ub_bytes &&
          dense_sizes.FitsMetadata()) {
        use_interleaved = true;
        selected_tile = lower;
        selected_sizes = dense_sizes;
      }
    }
    // 4. F11_MediumAligned64 (p6: H=64, N=160, iter=20)
    else if (IsAlignedMediumWorkload(h, h_pad, problem.iters, problem.eps)) {
      auto half_sizes = sizes;
      half_sizes.input =
          std::max(half_sizes.input, 4 * (h / 2) * h_pad * lower);
      if (half_sizes.Total() <= platform.ub_bytes &&
          half_sizes.FitsMetadata()) {
        sizes = half_sizes;
      }
      auto shifted_sizes = sizes;
      shifted_sizes.scratch += 256;
      if (shifted_sizes.Total() <= platform.ub_bytes &&
          shifted_sizes.FitsMetadata()) {
        sizes = shifted_sizes;
        auto alternate_sizes = sizes;
        alternate_sizes.scratch += sizes.matrix + 512;
        if ((h_pad * lower * sizeof(float)) % 512 == 0 &&
            alternate_sizes.Total() <= platform.ub_bytes &&
            alternate_sizes.FitsMetadata()) {
          sizes = alternate_sizes;
        }
      }
      use_interleaved = true;
      selected_tile = lower;
      selected_sizes = sizes;
    }
    // 5. F13_WideQuarterStream (p9: H=100, N=768, iter=32)
    else if (IsCompactFiveWorkload(h, h_pad, problem.iters, problem.eps)) {
      const auto q = (h + 3) / 4;
      const auto bound = std::min(
          max_core_tokens, platform.ub_bytes / (5 * 4 * q * h_pad));
      for (auto candidate = bound; candidate >= 1; --candidate) {
        const auto x = MakeCompactFiveLayout(h, h_pad, candidate);
        BufferSizes five{x.input, 4 * x.slot, x.gate, x.scratch};
        if (five.Total() <= platform.ub_bytes && five.FitsMetadata() &&
            4 * x.quarter_rows * 8 <= 9 * h * candidate &&
            x.quarter_rows / 8 <= 255) {
          use_interleaved = true;
          selected_tile = candidate;
          selected_sizes = five;
          break;
        }
      }
    }
    // 6. F08_NarrowP24Recurrent (p4: H=23, N=384, iter=20)
    else if (h >= 17 && h <= 24 && problem.iters >= 3 &&
             IsEstablishedInterleavedWorkload(h, problem.iters, tokens_per_core,
                                             problem.eps)) {
      use_interleaved = true;
      selected_tile = lower;
      selected_sizes = sizes;
    }
    // 7. F10_MediumCompact (p5: H=56, N=384, iter=2..16)
    else if (compact_medium) {
      use_interleaved = true;
      selected_tile = lower;
      selected_sizes = GetInterleavedBufferSizes(lower, h, h_pad, true);
    }
    // 8. F07_NarrowP16DenseIo (p8: H in [9, 16), iters in [2, 3])
    else if (IsDenseNarrowIoWorkload(h, h_pad, problem.iters, tokens_per_core,
                                     lower, problem.eps)) {
      const auto f07_limit = std::min(max_core_tokens, std::uint64_t{64});
      for (auto candidate = f07_limit; candidate >= lower; --candidate) {
        if (candidate % 16 != 0) continue; // complete 16-token transpose groups
        if (!IsDenseNarrowIoWorkload(h,h_pad,problem.iters,tokens_per_core,candidate,problem.eps)) continue;
        const BufferSizes candidate_sizes{DenseNarrowInputBytes(h,candidate),
            DenseNarrowMatrixBytes(h,candidate),4*16*candidate,
            DenseNarrowScratchBytes(h,candidate)};
        if (candidate_sizes.Total() <= platform.ub_bytes &&
            DenseNarrowActualUbBytes(h,candidate) <= platform.ub_bytes &&
            candidate_sizes.FitsMetadata()) {
          use_interleaved = true;
          selected_tile = candidate;
          selected_sizes = candidate_sizes;
          break;
        }
      }
    }
    // 9. F06_NarrowP16PairedRows (p7: H in [9, 16], iter >= 4)
    else if (h <= 16 && problem.iters >= 4) {
      use_interleaved = true;
      selected_tile = lower;
      selected_sizes = sizes;
    }
  }

  // 9. F05_NarrowP16ShortBatch (p3: H in [9, 16], N short batch, iter in [2, 32])
  if (!use_interleaved &&
      IsShortNarrowBatchWorkload(h, h_pad, problem.iters, tokens_per_core,
                                extra_core_count, problem.eps)) {
    const auto batch_sizes =
        GetInterleavedBufferSizes(max_core_tokens, h, h_pad);
    if (batch_sizes.Total() <= platform.ub_bytes &&
        batch_sizes.FitsMetadata()) {
      use_interleaved = true;
      selected_tile = max_core_tokens;
      selected_sizes = batch_sizes;
    }
  }

  if (!use_interleaved) {
    std::uint64_t lower = 1;
    std::uint64_t upper = max_core_tokens;
    while (lower < upper) {
      const auto candidate = lower + (upper - lower + 1) / 2;
      const auto sizes = GetGeneralBufferSizes(candidate, h, h_pad, input_width,
                                               narrow_single_p24);
      if (sizes.Total() <= platform.ub_bytes && sizes.FitsMetadata()) {
        lower = candidate;
      } else {
        upper = candidate - 1;
      }
    }
    auto sizes =
        GetGeneralBufferSizes(lower, h, h_pad, input_width, narrow_single_p24);
    if (narrow_single_p24 && lower == 1) {
      auto padded_sizes = sizes;
      padded_sizes.matrix += PackedPaddedRowPaddingBytes(h, h_pad);
      if (padded_sizes.Total() <= platform.ub_bytes &&
          padded_sizes.FitsMetadata()) {
        sizes = padded_sizes;
      }
    }
    selected_tile = lower;
    selected_sizes = sizes;
  }

  tiling.token_count = token_count;
  tiling.tokens_per_core = tokens_per_core;
  tiling.extra_core_count = static_cast<std::uint32_t>(extra_core_count);
  tiling.core_count = static_cast<std::uint32_t>(core_count);
  tiling.h = static_cast<std::uint32_t>(h);
  tiling.h_pad = static_cast<std::uint32_t>(h_pad);
  tiling.input_width = static_cast<std::uint32_t>(input_width);
  tiling.iters = static_cast<std::uint32_t>(problem.iters);
  tiling.tile_tokens = static_cast<std::uint32_t>(selected_tile);
  tiling.input_bytes = static_cast<std::uint32_t>(selected_sizes.input);
  tiling.matrix_bytes = static_cast<std::uint32_t>(selected_sizes.matrix);
  tiling.gate_bytes = static_cast<std::uint32_t>(selected_sizes.gate);
  tiling.scratch_bytes = static_cast<std::uint32_t>(selected_sizes.scratch);
  tiling.eps = problem.eps;

  output_shapes.pre[0] = problem.shape[0];
  output_shapes.pre[1] = problem.shape[1];
  output_shapes.pre[2] = problem.h;
  output_shapes.post[0] = problem.shape[0];
  output_shapes.post[1] = problem.shape[1];
  output_shapes.post[2] = problem.h;
  output_shapes.comb[0] = problem.shape[0];
  output_shapes.comb[1] = problem.shape[1];
  output_shapes.comb[2] = problem.h;
  output_shapes.comb[3] = problem.h;
  return true;
}

bool MakeLaunchPlan(const Problem &problem, const PlatformResources &platform,
                    LaunchPlan &plan, OutputShapes &output_shapes,
                    std::string &error) {
  plan = {};
  bool bank_columns = false;
  if (!MakeTiling(problem, platform, plan.data, output_shapes, error,
                  &bank_columns)) {
    return false;
  }
  plan.blocks = plan.data.core_count;
  if (UsesCompactSplitOutputs(plan.data) &&
      plan.data.token_count <= platform.aiv_cores / 2) {
    plan.blocks = 2 * plan.data.core_count;
  } else if (UsesTinySingle(plan.data) && plan.data.h == 3 &&
             plan.data.token_count <= 64 && platform.aiv_cores >= 2 &&
             platform.ub_bytes >= 40960) {
    auto &t = plan.data;
    t.core_count = static_cast<uint32_t>((t.token_count + 7) / 8);
    t.tokens_per_core = 8;
    t.extra_core_count = 0;
    t.tile_tokens = 8;
    t.input_bytes = 512;
    t.matrix_bytes = 512;
    t.gate_bytes = 64;
    t.scratch_bytes = 16384;
    plan.blocks = t.core_count + 1;
  } else if (UsesTinySingle(plan.data) && plan.data.token_count <= 64 &&
             plan.data.core_count < platform.aiv_cores && platform.ub_bytes >= 40960) {
    // Reserve one otherwise idle AIV for all gates; matrix ownership is unchanged.
    plan.blocks = plan.data.core_count + 1;
  }
  plan.key = SelectTilingKey(plan.blocks, plan.data, bank_columns);
  if (plan.key == 0) {
    plan = {};
    error = "no compiled tiling key for launch metadata";
    return false;
  }
  if (FamilyOf(plan.key) == KernelFamily::F03_TinyTokenPlanes) {
    auto &t = plan.data;
    const uint64_t h = t.h, width = t.input_width;
    const uint64_t lanes = 3 * h * h + 5 * h + 4;
    const uint64_t usable = platform.ub_bytes > kReservedBytes
                                ? platform.ub_bytes - kReservedBytes
                                : 0;
    const uint64_t owner =
        uint64_t{t.tokens_per_core} + (t.extra_core_count != 0);
    const uint64_t limit = std::min({usable / (4 * lanes) / 64 * 64,
                                     (255 / (h * h)) * 64,
                                     (owner + 63) / 64 * 64});
    if (limit > t.tile_tokens) {
      t.tile_tokens = static_cast<uint32_t>(std::min(limit, owner));
      t.input_bytes = static_cast<uint32_t>(4 * width * limit);
      t.matrix_bytes = t.input_bytes;
      t.gate_bytes = 0;
      t.scratch_bytes = static_cast<uint32_t>(4 * (h * h + h + 4) * limit);
    }
    const uint64_t current = CompactPlaneCapacity(t);
    const uint64_t live_lanes = 3 * h * h + 5 * h + 3;
    const uint64_t physical_limit =
        std::min(usable / (4 * live_lanes) / 64 * 64, (255 / (h * h)) * 64);
    const uint64_t largest =
        physical_limit > current
            ? current + (physical_limit - current) / 128 * 128
            : current;
    const uint64_t best_batches = (owner + largest - 1) / largest;
    const uint64_t required = (owner + best_batches - 1) / best_batches;
    const uint64_t expanded =
        required > current ? current + (required - current + 127) / 128 * 128
                           : current;
    bool improves =
        expanded > current &&
        (owner + expanded - 1) / expanded < (owner + current - 1) / current;
    for (uint64_t kind = 0;
         improves && kind < (t.extra_core_count != 0 ? 2ULL : 1ULL); ++kind) {
      const uint64_t count = t.tokens_per_core + kind;
      improves = (count + expanded - 1) / expanded <=
                 (count + current - 1) / current;
    }
    if (improves) {
      t.tile_tokens = static_cast<uint32_t>(std::min(expanded, owner));
      t.input_bytes = static_cast<uint32_t>(4 * width * expanded);
      t.matrix_bytes = t.input_bytes;
      t.gate_bytes = 0;
      t.scratch_bytes = static_cast<uint32_t>(4 * (h * h + h + 3) * expanded);
    }
  }
  return true;
}

}  // namespace sinkhorn_norm
