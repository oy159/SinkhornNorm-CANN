// Included after the shared implementation in the exported kernel.asc.
// TensorInfo and TensorGroupInfo are supplied by the original entry source.
#include <stdexcept>

namespace {

const TensorInfo &SinkhornRequireTensor(const TensorGroupInfo &group,
                                        int64_t rank, const char *name)
{
    if (group.numTensors != 1 || group.tensors == nullptr) {
        throw std::invalid_argument(std::string(name) + ": expected one tensor");
    }
    const TensorInfo &tensor = group.tensors[0];
    if (tensor.numDims != rank || tensor.shape == nullptr || tensor.dtype != 0) {
        throw std::invalid_argument(std::string(name) + ": expected FP32 tensor with correct rank");
    }
    return tensor;
}

void SinkhornRequireOutput(const TensorGroupInfo &group, int64_t rank,
                           int64_t b, int64_t s, int64_t h, const char *name)
{
    const auto &tensor = SinkhornRequireTensor(group, rank, name);
    if (tensor.shape[0] != b || tensor.shape[1] != s || tensor.shape[2] != h ||
        (rank == 4 && tensor.shape[3] != h)) {
        throw std::invalid_argument(std::string(name) + ": output shape mismatch");
    }
}

} // namespace

extern "C" void run_kernel(
    GM_ADDR mixes,
    const TensorGroupInfo &info_mixes,
    GM_ADDR pre,
    const TensorGroupInfo &info_pre,
    GM_ADDR post,
    const TensorGroupInfo &info_post,
    GM_ADDR comb,
    const TensorGroupInfo &info_comb,
    int64_t availableCoreNum,
    aclrtStream stream,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    float eps)
{
    // Metadata is host-resident. Device tensor contents are never inspected.
    const auto &input = SinkhornRequireTensor(info_mixes, 3, "mixes");
    if (availableCoreNum <= 0) {
        throw std::invalid_argument("availableCoreNum must be positive");
    }
    sinkhorn_norm::Problem problem{};
    for (unsigned i = 0; i < 3; ++i) {
        problem.shape[i] = input.shape[i];
    }
    problem.h = hc_mult;
    problem.iters = hc_sinkhorn_iters;
    problem.eps = eps;
    SinkhornRequireOutput(info_pre, 3, problem.shape[0], problem.shape[1], hc_mult, "pre");
    SinkhornRequireOutput(info_post, 3, problem.shape[0], problem.shape[1], hc_mult, "post");
    SinkhornRequireOutput(info_comb, 4, problem.shape[0], problem.shape[1], hc_mult, "comb");
    const sinkhorn_norm::DeviceBuffers buffers{mixes, pre, post, comb};
    std::string error;
    if (!sinkhorn_norm::Launch(problem, buffers, stream,
                               static_cast<uint64_t>(availableCoreNum), error)) {
        throw std::invalid_argument(error);
    }
}
