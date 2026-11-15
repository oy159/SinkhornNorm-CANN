#!/usr/bin/env bash
# Runs inside the pinned CANN image; builds only, never launches a kernel.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source_dir="${1:-.}"
while [[ "$source_dir" == ./* ]]; do source_dir="${source_dir#./}"; done
while [[ "$source_dir" == */ ]]; do source_dir="${source_dir%/}"; done
source_dir="${source_dir:-.}"
arch="${NPU_ARCH:-dav-2201}"
build_dir="build/cann9/${arch}/${source_dir}"
cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$source_dir" == "submission" ]]; then
    cmake_args+=(-DCMAKE_ASC_ARCHITECTURES="$arch" -DCMAKE_ASC_RUN_MODE=npu)
elif [[ "$source_dir" == "testbench" ]]; then
    revision="${SINKHORN_TEST_REVISION:-$(git rev-parse HEAD 2>/dev/null || echo '0000000000000000000000000000000000000000')}"
    cmake_args+=(-DCMAKE_ASC_ARCHITECTURES="$arch" -DCMAKE_ASC_RUN_MODE=npu -DSINKHORN_TEST_REVISION="$revision")
else
    cmake_args+=(-DNPU_ARCH="$arch")
    if [[ "$source_dir" == "." ]]; then
        cmake_args+=(-DSINKHORN_BUILD_DEVICE=ON)
    fi
fi
cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-2}"
if [[ "$source_dir" == "." ]]; then
    ctest --test-dir "$build_dir" --output-on-failure
fi
