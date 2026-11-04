#include "sinkhorn_norm/launch.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

extern "C" void sinkhorn_norm_enqueue(uint32_t blocks, void *stream,
                                      uint8_t *mixes, uint8_t *pre,
                                      uint8_t *post, uint8_t *comb,
                                      sinkhorn_norm::TilingData tiling,
                                      sinkhorn_norm::TilingKey key);

namespace sinkhorn_norm {

bool QueryPlatformResources(PlatformResources &resources, std::string &error)
{
    resources = {};
    error.clear();
#if defined(ASCENDC_CPU_DEBUG) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    // CANN CPU debugging has no physical device to query. The compiler's
    // dav-2201 target selects the same SDK platform description as CPU launch.
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1");
#else
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
#endif
    if (platform == nullptr) {
        error = "CANN platform is unavailable; initialize ACL and select a device first";
        return false;
    }
    resources.aiv_cores = platform->GetCoreNumAiv();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, resources.ub_bytes);
    if (resources.aiv_cores == 0 || resources.ub_bytes == 0) {
        error = "CANN returned an empty AIV/UB resource description";
        return false;
    }
    return true;
}

bool Launch(const Problem &problem, const DeviceBuffers &buffers, void *stream,
            std::string &error)
{
    return Launch(problem, buffers, stream, std::numeric_limits<uint64_t>::max(), error);
}

bool Launch(const Problem &problem, const DeviceBuffers &buffers, void *stream,
            uint64_t core_limit, std::string &error)
{
    if (core_limit == 0) {
        error = "available vector core count must be positive";
        return false;
    }
    if (!buffers.mixes || !buffers.pre || !buffers.post || !buffers.comb) {
        error = "all four device buffers are required";
        return false;
    }
    PlatformResources resources{};
    if (!QueryPlatformResources(resources, error)) {
        return false;
    }
    resources.aiv_cores = static_cast<uint32_t>(
        std::min<uint64_t>(resources.aiv_cores, core_limit));
    LaunchPlan plan{};
    OutputShapes shapes{};
    if (!MakeLaunchPlan(problem, resources, plan, shapes, error)) {
        return false;
    }
    const auto &tiling = plan.data;
    const void *pointers[] = {buffers.mixes, buffers.pre, buffers.post, buffers.comb};
    const uint64_t gateBytes = tiling.token_count * tiling.h * sizeof(float);
    const uint64_t sizes[] = {tiling.token_count * tiling.input_width * sizeof(float),
                              gateBytes, gateBytes, gateBytes * tiling.h};
    uintptr_t begin[4]{};
    uintptr_t end[4]{};
    for (unsigned i = 0; i < 4; ++i) {
        begin[i] = reinterpret_cast<uintptr_t>(pointers[i]);
        if (sizes[i] > std::numeric_limits<uintptr_t>::max() - begin[i]) {
            error = "device buffer address range overflows";
            return false;
        }
        end[i] = begin[i] + static_cast<uintptr_t>(sizes[i]);
        for (unsigned j = 0; j < i; ++j) {
            if (begin[i] < end[j] && begin[j] < end[i]) {
                error = "input and output device buffer ranges must be disjoint";
                return false;
            }
        }
    }
    sinkhorn_norm_enqueue(plan.blocks, stream,
                          static_cast<uint8_t *>(buffers.mixes),
                          static_cast<uint8_t *>(buffers.pre),
                          static_cast<uint8_t *>(buffers.post),
                          static_cast<uint8_t *>(buffers.comb), tiling,
                          plan.key);
    return true;
}

} // namespace sinkhorn_norm
