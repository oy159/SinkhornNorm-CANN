#!/usr/bin/env bash
# Isolated cloud testbench, never a CANNJudge submission.
# ASCEND_TEST_HOST=ssh-alias scripts/remote-testbench.sh build
# ASCEND_TEST_HOST=ssh-alias scripts/remote-testbench.sh run --suite smoke
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_host="${ASCEND_TEST_HOST:?Set ASCEND_TEST_HOST to an SSH host or alias}"
action="${1:-build}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in build|run) ;; *) echo 'usage: remote-testbench.sh build | run [run.py arguments]' >&2; exit 2 ;; esac
python3 "$root/scripts/export-template.py" --check
revision="$(git -C "$root" rev-parse HEAD)"
source_paths=(src include submission/kernel.asc submission/reference.py scripts/export-template.py testbench)
if [[ -n "$(git -C "$root" status --porcelain --untracked-files=all -- "${source_paths[@]}")" ]]; then
    echo 'Commit testbench and operator sources first; cloud run must identify the actual tested commit.' >&2
    exit 2
fi
# Get the home path from SSH, then quote every remote argument explicitly.
remote_home="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" 'printf "%s" "$HOME"')"
remote_root="${ASCEND_TEST_ROOT:-$remote_home/ascend-dev/workspaces/SinkhornNorm-testbench/$revision}"
printf -v quoted_root '%q' "$remote_root"
ssh -o BatchMode=yes "$remote_host" "mkdir -p $quoted_root"
git -C "$root" archive HEAD "${source_paths[@]}" | ssh -o BatchMode=yes "$remote_host" "tar -xf - -C $quoted_root && find $quoted_root -name \"*.sh\" -exec sed -i \"s/\r$//\" {} +"
if [[ "$action" == build ]]; then
    ssh -o BatchMode=yes "$remote_host" "cd $quoted_root && SINKHORN_TEST_REVISION=$revision bash testbench/build.sh"
else
    # No implicit retries or full suite: default smoke and explicit 1+5 protocol.
    stamp="$(date +%Y%m%dT%H%M%S%z)"
    remote_out="$remote_home/ascend-dev/testbench-results/SinkhornNorm/$revision/$stamp"
    args=(python3 testbench/run.py --revision "$revision" --out "$remote_out" "$@")
    printf -v quoted_args '%q ' "${args[@]}"
    setup='if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then for f in /home/developer/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do if [[ -f "$f" ]]; then source "$f"; break; fi; done; fi'
    echo "Cloud results: $remote_host:$remote_out"
    ssh -o BatchMode=yes "$remote_host" "set -eo pipefail; $setup; cd $quoted_root; $quoted_args"
fi
