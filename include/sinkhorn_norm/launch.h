#pragma once

#include "sinkhorn_norm/tiling.h"

namespace sinkhorn_norm {

// These pointers refer to device memory. Host code never dereferences them.
struct DeviceBuffers {
    void *mixes = nullptr;
    void *pre = nullptr;
    void *post = nullptr;
    void *comb = nullptr;
};

bool QueryPlatformResources(PlatformResources &resources, std::string &error);

// Enqueues exactly one complete kernel. The caller owns the buffers and stream,
// allocates outputs using OutputShapes, and checks asynchronous runtime errors.
bool Launch(const Problem &problem, const DeviceBuffers &buffers, void *stream,
            std::string &error);

// The template supplies an upper bound on available vector cores. Keep this
// value wide until it has been intersected with actual platform resources.
bool Launch(const Problem &problem, const DeviceBuffers &buffers, void *stream,
            uint64_t core_limit, std::string &error);

} // namespace sinkhorn_norm
