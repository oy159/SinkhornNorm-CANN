#include "sinkhorn_norm/tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

std::uint64_t checks = 0;

void Check(bool condition, const char* expression, int line) {
  ++checks;
  if (!condition) {
    std::cerr << "tiling_test.cpp:" << line << ": failed: " << expression
              << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expression) Check((expression), #expression, __LINE__)

using sinkhorn_norm::DType;
using sinkhorn_norm::FamilyOf;
using sinkhorn_norm::KernelFamily;
using sinkhorn_norm::LaunchPlan;
using sinkhorn_norm::MakeLaunchPlan;
using sinkhorn_norm::MakeTiling;
using sinkhorn_norm::OutputShapes;
using sinkhorn_norm::PlatformResources;
using sinkhorn_norm::Problem;
using sinkhorn_norm::TilingData;

constexpr PlatformResources kTestResources{40, 192 * 1024};

std::uint64_t Align32(std::uint64_t bytes) { return (bytes + 31) / 32 * 32; }

std::uint64_t RequiredUb(std::uint64_t tokens, std::uint32_t h) {
  const std::uint64_t h_pad = (h + 7) / 8 * 8;
  const std::uint64_t width = static_cast<std::uint64_t>(h) * (h + 2);
  return Align32(tokens * width * 4) + tokens * h * h_pad * 4 +
         2 * tokens * h_pad * 4 + 48 * h_pad + 8192;
}

Problem MakeProblem(std::int64_t b, std::int64_t s, std::int64_t h,
                    std::int64_t iters = 20) {
  Problem problem{};
  problem.rank = 3;
  problem.dtype = DType::Float32;
  problem.shape[0] = b;
  problem.shape[1] = s;
  problem.shape[2] = h * (h + 2);
  problem.h = h;
  problem.iters = iters;
  problem.eps = 1e-6f;
  return problem;
}

LaunchPlan CheckPlan(const Problem& problem,
                     const PlatformResources& resources = kTestResources) {
  LaunchPlan plan{};
  OutputShapes output{};
  std::string error = "stale error from an earlier call";
  CHECK(MakeLaunchPlan(problem, resources, plan, output, error));
  CHECK(error.empty());
  CHECK(plan.key != 0);
  CHECK(plan.blocks >= 1);
  const auto& tiling = plan.data;

  const auto n = static_cast<std::uint64_t>(problem.shape[0]) *
                 static_cast<std::uint64_t>(problem.shape[1]);
  const auto h = static_cast<std::uint32_t>(problem.h);
  const auto h_pad = (h + 7) / 8 * 8;
  const bool eight_token_group = FamilyOf(plan.key) == KernelFamily::F02_TinySingle &&
                                h == 3 && tiling.tile_tokens == 8;
  const auto cores = eight_token_group ? (n + 7) / 8
      : std::min<std::uint64_t>(n, resources.aiv_cores);
  const auto maximum_count = (n + cores - 1) / cores;
  CHECK(tiling.token_count == n);
  CHECK(tiling.h == h);
  CHECK(tiling.h_pad == h_pad);
  CHECK(tiling.input_width == h * (h + 2));
  CHECK(tiling.iters == problem.iters);
  CHECK(tiling.eps == problem.eps);
  CHECK(std::signbit(tiling.eps) == std::signbit(problem.eps));
  CHECK(tiling.core_count == cores);
  CHECK(tiling.tokens_per_core == (eight_token_group ? 8 : n / cores));
  CHECK(tiling.extra_core_count == (eight_token_group ? 0 : n % cores));
  CHECK(tiling.tile_tokens >= 1);
  CHECK(tiling.tile_tokens <= (eight_token_group ? 8 : maximum_count));
  const std::uint64_t reserve = UsesRotatingQuarters(tiling)
      ? 8192ULL - 8ULL * tiling.h_pad * tiling.tile_tokens : 8192ULL;
  const std::uint64_t total_ub =
      static_cast<std::uint64_t>(tiling.input_bytes) + tiling.matrix_bytes +
      2ULL * tiling.gate_bytes + tiling.scratch_bytes + reserve;
  CHECK(total_ub <= resources.ub_bytes);

  if (FamilyOf(plan.key) == KernelFamily::F00_GeneralFallback) {
    CHECK(RequiredUb(tiling.tile_tokens, h) <= resources.ub_bytes);
    CHECK(tiling.tile_tokens == maximum_count ||
          RequiredUb(static_cast<std::uint64_t>(tiling.tile_tokens) + 1, h) >
              resources.ub_bytes);
    CHECK(tiling.input_bytes ==
          Align32(static_cast<std::uint64_t>(tiling.tile_tokens) *
                  tiling.input_width * 4));
    CHECK(tiling.matrix_bytes ==
          static_cast<std::uint64_t>(tiling.tile_tokens) * h * h_pad * 4);
    CHECK(tiling.gate_bytes ==
          static_cast<std::uint64_t>(tiling.tile_tokens) * h_pad * 4);
    CHECK(tiling.scratch_bytes == 48 * h_pad);
    CHECK(static_cast<std::uint64_t>((h + 1) / 2) * h_pad * 4 <=
          tiling.input_bytes);
  }

  for (int i = 0; i < 2; ++i) {
    CHECK(output.pre[i] == problem.shape[i]);
    CHECK(output.post[i] == problem.shape[i]);
    CHECK(output.comb[i] == problem.shape[i]);
  }
  CHECK(output.pre[2] == h);
  CHECK(output.post[2] == h);
  CHECK(output.comb[2] == h);
  CHECK(output.comb[3] == h);

  std::uint64_t cursor = 0;
  std::uint64_t minimum_count = n;
  std::uint64_t actual_maximum_count = 0;
  for (std::uint32_t core = 0; core < tiling.core_count; ++core) {
    const std::uint64_t start = eight_token_group ? 8ULL * core :
        static_cast<std::uint64_t>(core) * tiling.tokens_per_core +
        std::min(core, tiling.extra_core_count);
    const std::uint64_t count = eight_token_group ? std::min<std::uint64_t>(8, n - start) :
        tiling.tokens_per_core + (core < tiling.extra_core_count ? 1 : 0);
    CHECK(start == cursor);
    CHECK(count >= 1);
    CHECK(start + count <= n);
    const std::uint64_t full_tiles = count / tiling.tile_tokens;
    const std::uint64_t tail = count % tiling.tile_tokens;
    CHECK(full_tiles * tiling.tile_tokens + tail == count);
    CHECK(tail < tiling.tile_tokens);
    minimum_count = std::min(minimum_count, count);
    actual_maximum_count = std::max(actual_maximum_count, count);
    cursor += count;
  }
  CHECK(cursor == n);
  CHECK(actual_maximum_count - minimum_count <= (eight_token_group ? 7 : 1));
  return plan;
}

TilingData CheckValid(const Problem& problem,
                      const PlatformResources& resources = kTestResources) {
  return CheckPlan(problem, resources).data;
}

void CheckInvalid(const Problem& problem,
                  const PlatformResources& resources = kTestResources) {
  TilingData tiling{};
  OutputShapes output{};
  std::string error = "stale error from an earlier call";
  CHECK(!MakeTiling(problem, resources, tiling, output, error));
  CHECK(!error.empty());
  CHECK(tiling.token_count == 0);
  CHECK(tiling.core_count == 0);
  CHECK(tiling.tile_tokens == 0);
  CHECK(output.pre[0] == 0);
  CHECK(output.post[0] == 0);
  CHECK(output.comb[0] == 0);
}

void TestFifteenBenchmarkRegimes() {
  constexpr PlatformResources kJudgeResources{40, 196352};
  // C1..C5 and p1..p10 from the 15-point profile (C1 iter = 15)
  const auto c1 = CheckPlan(MakeProblem(7, 128, 128, 15), kJudgeResources);
  CHECK(FamilyOf(c1.key) == KernelFamily::F14_WideRotating128);

  const auto c2 = CheckPlan(MakeProblem(1, 1, 5, 1), kJudgeResources);
  CHECK(FamilyOf(c2.key) == KernelFamily::F01_TinySplit);
  CHECK(c2.blocks == 2);

  const auto c3 = CheckPlan(MakeProblem(2, 13, 22, 20), kJudgeResources);
  CHECK(FamilyOf(c3.key) == KernelFamily::F04_NarrowSingle);

  const auto c4 = CheckPlan(MakeProblem(3, 32, 88, 20), kJudgeResources);
  CHECK(FamilyOf(c4.key) == KernelFamily::F12_WideBlocked);

  const auto c5 = CheckPlan(MakeProblem(8, 32, 34, 20), kJudgeResources);
  CHECK(FamilyOf(c5.key) == KernelFamily::F09_MediumDenseIo);

  const auto p1 = CheckPlan(MakeProblem(1, 21, 3, 3), kJudgeResources);
  CHECK(FamilyOf(p1.key) == KernelFamily::F02_TinySingle);
  CHECK(p1.blocks == 4);
  CHECK(p1.data.tile_tokens == 8);
  CheckValid(MakeProblem(1, 35, 7, 20), kJudgeResources);       // p2
  const auto p3 = CheckPlan(MakeProblem(1, 192, 15, 20), kJudgeResources);     // p3
  CHECK(FamilyOf(p3.key) == KernelFamily::F05_NarrowP16ShortBatch);
  const auto p4 = CheckPlan(MakeProblem(1, 384, 23, 20), kJudgeResources);     // p4
  CHECK(FamilyOf(p4.key) == KernelFamily::F08_NarrowP24Recurrent);
  const auto p5 = CheckPlan(MakeProblem(1, 384, 56, 8), kJudgeResources);      // p5
  CHECK(FamilyOf(p5.key) == KernelFamily::F10_MediumCompact);
  const auto p6 = CheckPlan(MakeProblem(1, 160, 64, 20), kJudgeResources);     // p6
  CHECK(FamilyOf(p6.key) == KernelFamily::F11_MediumAligned64);
  const auto p7 = CheckPlan(MakeProblem(1, 900, 9, 20), kJudgeResources);      // p7
  CHECK(FamilyOf(p7.key) == KernelFamily::F06_NarrowP16PairedRows);
  const auto p8 = CheckPlan(MakeProblem(1, 8000, 13, 3), kJudgeResources);     // p8
  CHECK(FamilyOf(p8.key) == KernelFamily::F07_NarrowP16DenseIo);
  const auto p9 = CheckPlan(MakeProblem(1, 768, 100, 32), kJudgeResources);    // p9
  CHECK(FamilyOf(p9.key) == KernelFamily::F13_WideQuarterStream);
  CheckValid(MakeProblem(128, 8192, 2, 8), kJudgeResources);    // p10
}

void TestAllHAndPartitions() {
  for (std::int64_t h = 2; h <= 128; ++h) {
    for (const std::int64_t n : {1, 13, 39, 40, 41, 79, 80, 81, 8191}) {
      for (const std::int64_t iters : {1, 32}) {
        CheckValid(MakeProblem(1, n, h, iters));
      }
    }

    std::uint32_t capacity = 0;
    while (RequiredUb(static_cast<std::uint64_t>(capacity) + 1,
                      static_cast<std::uint32_t>(h)) <= kTestResources.ub_bytes) {
      ++capacity;
    }
    CHECK(capacity >= 1);
    const auto tail_one = CheckPlan(MakeProblem(40, 2 * capacity + 1, h, 1));
    if (FamilyOf(tail_one.key) == KernelFamily::F00_GeneralFallback) {
      CHECK(tail_one.data.tile_tokens == capacity);
      CHECK(tail_one.data.tokens_per_core % capacity == 1 % capacity);
    }
    if (capacity > 1) {
      const auto tail_last = CheckPlan(MakeProblem(40, 3 * capacity - 1, h, 1));
      if (FamilyOf(tail_last.key) == KernelFamily::F00_GeneralFallback) {
        CHECK(tail_last.data.tile_tokens == capacity);
        CHECK(tail_last.data.tokens_per_core % capacity == capacity - 1);
      }
    }
    const auto complete = CheckPlan(MakeProblem(40, 2 * capacity, h, 1));
    if (FamilyOf(complete.key) == KernelFamily::F00_GeneralFallback) {
      CHECK(complete.data.tile_tokens == capacity);
      CHECK(complete.data.tokens_per_core % capacity == 0);
    }

    const auto minimum_ub = RequiredUb(1, static_cast<std::uint32_t>(h));
    CheckInvalid(MakeProblem(1, 41, h, 1), {40, minimum_ub - 1});
    const auto exact = CheckValid(MakeProblem(1, 41, h, 1), {40, minimum_ub});
    CHECK(exact.tile_tokens == 1);
  }
}

void TestMaximumShapeAndResourceSelection() {
  const auto max = CheckValid(MakeProblem(128, 8192, 128, 32));
  CHECK(max.token_count == 1048576ULL);
  for (const auto cores : {1U, 7U, 13U, 40U, 64U, 128U, 65535U}) {
    CheckValid(MakeProblem(1, 41, 17), {cores, kTestResources.ub_bytes});
  }
  CheckValid(MakeProblem(128, 8192, 2),
             {1, std::numeric_limits<std::uint64_t>::max()});
  for (const float eps : {0.0f, -0.0f, 1e-6f, 0.001f, 0.125f, -0.125f}) {
    auto problem = MakeProblem(1, 41, 7);
    problem.eps = eps;
    CheckValid(problem);
  }
}

void TestInvalidArguments() {
  const auto valid = MakeProblem(1, 41, 4);
  for (const std::uint32_t rank : {0U, 1U, 2U, 4U, 99U}) {
    auto problem = valid;
    problem.rank = rank;
    CheckInvalid(problem);
  }
  auto problem = valid;
  problem.dtype = DType::Unsupported;
  CheckInvalid(problem);
  for (const std::int64_t b : {-1, 0, 129}) {
    problem = valid;
    problem.shape[0] = b;
    CheckInvalid(problem);
  }
  for (const std::int64_t s : {-1, 0, 8193}) {
    problem = valid;
    problem.shape[1] = s;
    CheckInvalid(problem);
  }
  for (const std::int64_t h : {-1, 0, 1, 129}) {
    problem = valid;
    problem.h = h;
    CheckInvalid(problem);
  }
  for (const std::int64_t iters : {-1, 0, 33}) {
    problem = valid;
    problem.iters = iters;
    CheckInvalid(problem);
  }
  for (const std::int64_t width : {-1, 0, 23, 25}) {
    problem = valid;
    problem.shape[2] = width;
    CheckInvalid(problem);
  }
  problem = valid;
  problem.shape[0] = std::numeric_limits<std::int64_t>::max();
  problem.shape[1] = std::numeric_limits<std::int64_t>::max();
  CheckInvalid(problem);
  problem = valid;
  problem.h = std::numeric_limits<std::int64_t>::max();
  CheckInvalid(problem);
  CheckInvalid(valid, {0, kTestResources.ub_bytes});
  CheckInvalid(valid, {40, 0});
  for (const float eps : {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()}) {
    problem = valid;
    problem.eps = eps;
    CheckInvalid(problem);
  }
}

}  // namespace

int main() {
  TestFifteenBenchmarkRegimes();
  TestAllHAndPartitions();
  TestMaximumShapeAndResourceSelection();
  TestInvalidArguments();
  std::cout << "PASS: " << checks << " host tiling checks (no NPU execution)\n";
  return EXIT_SUCCESS;
}
