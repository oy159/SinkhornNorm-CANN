#!/usr/bin/env bash
set -eo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    for setup in /home/developer/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        if [[ -f "$setup" ]]; then source "$setup"; break; fi
    done
fi
set -u
revision="${SINKHORN_TEST_REVISION:-$(git -C "$root" rev-parse HEAD)}"
python3 "$root/scripts/export-template.py" --check
cmake -S "$root/testbench" -B "$root/build/testbench" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASC_ARCHITECTURES="${NPU_ARCH:-dav-2201}" \
    -DSINKHORN_TEST_REVISION="$revision"
cmake --build "$root/build/testbench" --parallel "${BUILD_JOBS:-2}"
