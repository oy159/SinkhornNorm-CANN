#ifndef SINKHORN_NORM_TILING_H_
#define SINKHORN_NORM_TILING_H_

#include "sinkhorn_norm/tiling_data.h"
#include "sinkhorn_norm/tiling_key.h"

#include <cstdint>
#include <string>

namespace sinkhorn_norm {

enum class DType { Float32, Unsupported };

struct Problem {
  std::int64_t shape[3] = {0, 0, 0};
  std::uint32_t rank = 3;
  DType dtype = DType::Float32;
  std::int64_t h = 4;
  std::int64_t iters = 20;
  float eps = 1e-6F;
};

struct PlatformResources {
  std::uint32_t aiv_cores;
  std::uint64_t ub_bytes;
};

struct OutputShapes {
  std::int64_t pre[3];
  std::int64_t post[3];
  std::int64_t comb[4];
};

// Reads only shape, attributes, and platform resources. On failure, zeroes
// tiling/output_shapes and sets error; on success, clears error.
// eps is preserved exactly. Any finite value is accepted because the problem
// specifies no eps range; callers must supply finite eps.
// bank_columns is a Host-only dispatch result, reset to false on failure.
// Launch must request it; buffer capacities alone do not identify this mode.
bool MakeTiling(const Problem& problem, const PlatformResources& platform,
                TilingData& tiling, OutputShapes& output_shapes,
                std::string& error, bool* bank_columns = nullptr);

// Host-only launch plan; the unchanged 64-byte Device payload contains no key.
struct LaunchPlan {
  TilingData data{};
  TilingKey key = 0;
  std::uint32_t blocks = 0;
};
bool MakeLaunchPlan(const Problem &problem, const PlatformResources &platform,
                    LaunchPlan &plan, OutputShapes &output_shapes,
                    std::string &error);

}  // namespace sinkhorn_norm

#endif  // SINKHORN_NORM_TILING_H_
