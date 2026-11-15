#!/usr/bin/env bash
# Usage: scripts/remote-build.sh [project-relative CMake source directory]
# ASCEND_BUILD_WORKSPACE_NAME selects an isolated remote workspace basename.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_host="${ASCEND_BUILD_HOST:?Set ASCEND_BUILD_HOST to an SSH host or alias}"
workspace_name="${ASCEND_BUILD_WORKSPACE_NAME-SinkhornNorm}"
source_dir="${1:-.}"
arch="${NPU_ARCH:-dav-2201}"
jobs="${BUILD_JOBS:-2}"
image='swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:0eeec44e67cd85a496d5189cf96c3430142a5d0dc71fa3d315d282daf8278729'
if [[ ! "$workspace_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$workspace_name" == . || "$workspace_name" == .. ]]; then
    echo 'ASCEND_BUILD_WORKSPACE_NAME must be a safe basename beginning with a letter or digit' >&2
    exit 2
fi
case "$source_dir" in
    /*|..|../*|*/../*|*/..) echo 'Use a project-relative source directory without ..' >&2; exit 2 ;;
esac
while [[ "$source_dir" == ./* ]]; do source_dir="${source_dir#./}"; done
while [[ "$source_dir" == */ ]]; do source_dir="${source_dir%/}"; done
source_dir="${source_dir:-.}"
if [[ ! -f "$project_root/$source_dir/CMakeLists.txt" ]]; then
    echo "Missing CMakeLists.txt: $source_dir" >&2
    exit 2
fi
if [[ "$source_dir" == "submission" || "$source_dir" == "testbench" ]]; then
    python3 "$project_root/scripts/export-template.py" --check
fi
printf -v mkdir_command \
    'mkdir -p "$HOME/ascend-dev/workspaces/%s" "$HOME/ascend-dev/cann9/logs"' \
    "$workspace_name"
printf -v extract_command 'tar -xzf - -C "$HOME/ascend-dev/workspaces/%s"' \
    "$workspace_name"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" "$mkdir_command"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude='.git' --exclude='build' \
    --exclude='build-npu' --exclude='build-cpu' --exclude='__pycache__' \
    --exclude='./scripts/.cache' --exclude='./scripts/acounts.env' \
    --exclude='./scripts/accounts.env' \
    --exclude='history' --exclude='logs' --exclude='.DS_Store' -C "$project_root" . |
    ssh -o BatchMode=yes "$remote_host" "$extract_command"
    revision="$(git -C "$project_root" rev-parse HEAD)"
    printf -v build_command \
        'docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e NPU_ARCH=%q -e BUILD_JOBS=%q -e SINKHORN_TEST_REVISION=%q -v "$HOME/ascend-dev/workspaces/%s:/workspace" -w /workspace %q bash scripts/cann9-build.sh %q' \
        "$arch" "$jobs" "$revision" "$workspace_name" "$image" "$source_dir"
ssh -o BatchMode=yes "$remote_host" "set -o pipefail; $build_command 2>&1 | tee \"\$HOME/ascend-dev/cann9/logs/build-\$(date +%Y%m%dT%H%M%S%z).log\""
