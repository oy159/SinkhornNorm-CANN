#!/usr/bin/env python3
"""Cloud-only fixtures, correctness gate and official device profiling."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import struct
import subprocess
import sys

from cases import SUITES, fixture_seed, select_cases
from profiler import (COMPACT_SINGLE_EXP_ABI, COMPACT_SPLIT_OUTPUTS_ABI,
                      CUBE_FACTOR_KERNEL, compact_static_split_abi,
                      parse_tasks, profiler_command)
from protocol import GM_LAYOUT, validate_completion

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("pre", "post", "comb")


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(argv, log, timeout):
    with log.open("x") as stream:
        with subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Profiler can have a child application; stop the whole group.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
                raise
    if status:
        raise RuntimeError(f"command failed ({status}); see {log}")


def fixture(case, dest, seed, chunk_tokens):
    # Verification harness only. This module/reference is never linked into the operator.
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("official_reference", ROOT / "submission/reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    dest.mkdir()
    rng = np.random.default_rng(seed)
    width, h, n = case.shape[-1], case.h, case.b * case.s
    (dest / "case.spec").write_text(f"{case.b} {case.s} {h} {case.iters} {case.eps:.9g}\n")
    sensitivity = None
    handles = [(dest / name).open("wb") for name in ["mixes.bin"] + [f"expected_{k}.bin" for k in OUTPUTS]]
    try:
        for begin in range(0, n, chunk_tokens):
            count = min(chunk_tokens, n - begin)
            bound = 50 if case.distribution == "uniform50" else 2
            x = rng.uniform(-bound, bound, (count, width)).astype("<f4")
            if case.distribution == "mixed":
                for i in range(count):
                    kind = (begin + i) % 5
                    if kind == 1:
                        x[i].fill(-1.25)
                    elif kind == 2:
                        x[i] = rng.uniform(-40, -0.01, width)
                    elif kind == 3:
                        x[i] = rng.uniform(-1000, 1000, width)
                    elif kind == 4:
                        x[i, 2 * h:] += np.repeat(rng.uniform(-1000, 1000, h), h)
            elif case.distribution == "slow":
                # Near-triangular 2x2 subsystem inside a near-diagonal matrix.
                # Competing column with a weak reverse edge converges slowly.
                for i in range(count):
                    matrix = rng.uniform(-0.25, 0.25, (h, h)).astype("<f4") - 20
                    np.fill_diagonal(matrix, 0)
                    matrix[0, 1] = rng.uniform(-0.25, 0.25)
                    x[i, 2 * h:] = matrix.ravel()
            with torch.no_grad():
                result = reference.impl(torch.from_numpy(x.reshape(1, count, width)), h, case.iters, case.eps)
                if case.distribution == "slow":
                    earlier = reference.impl(torch.from_numpy(x.reshape(1, count, width)), h, case.iters - 1, case.eps)[2]
                    ref, old = result[2].numpy(), earlier.numpy()
                    # Stronger than both this bench's OR criterion and np.isclose.
                    signal = float(np.max(np.abs(ref - old) / (1e-4 + 1e-4 * np.abs(ref))))
                    sensitivity = max(sensitivity or 0, signal)
            x.tofile(handles[0])
            for stream, values in zip(handles[1:], result):
                values.numpy().astype("<f4", copy=False).tofile(stream)
    finally:
        for stream in handles:
            stream.close()
    if case.distribution == "slow" and (sensitivity is None or sensitivity <= 1):
        raise ValueError("slow-convergence fixture does not distinguish I from I-1")
    metadata = dict(case.metadata(), seed=seed, chunk_tokens=chunk_tokens,
                    reference="unmodified submission/reference.py; CPU FP32; token chunks",
                    iteration_sensitivity_ratio=sensitivity)
    save(dest / "fixture.json", metadata)
    return metadata


def verify(case, fixture_dir, actual_dir):
    import numpy as np
    report = {}
    for name, count in (("pre", case.b * case.s * case.h), ("post", case.b * case.s * case.h),
                        ("comb", case.b * case.s * case.h * case.h)):
        expected_path, actual_path = fixture_dir / f"expected_{name}.bin", actual_dir / f"{name}.bin"
        if expected_path.stat().st_size != count * 4 or actual_path.stat().st_size != count * 4:
            raise ValueError(f"{name}: output length mismatch")
        expected = np.memmap(expected_path, dtype="<f4", mode="r")
        actual = np.memmap(actual_path, dtype="<f4", mode="r")
        bad, maximum = 0, 0.0
        for offset in range(0, count, 262144):
            ref = expected[offset:offset + 262144].astype("float64")
            got = actual[offset:offset + 262144].astype("float64")
            delta = np.abs(ref - got)
            close = np.isfinite(ref) & np.isfinite(got) & ((delta < 1e-4) | (delta < 1e-4 * np.abs(ref)) | (got == ref))
            bad += int(np.count_nonzero(~close))
            finite = delta[np.isfinite(delta)]
            if finite.size:
                maximum = max(maximum, float(finite.max()))
        report[name] = {"elements": count, "mismatch_count": bad, "max_absolute_error": maximum, "passed": bad == 0}
    completion = json.loads((actual_dir / "completion.json").read_text())
    return {"passed": all(r["passed"] for r in report.values()) and completion["input_unchanged"] and completion["guards_passed"],
            "criterion": "finite; abs_error < 1e-4 OR relative_error < 1e-4; zero mismatch allowance",
            "outputs": report, "completion": completion}


def find_tool(explicit):
    candidates = [explicit] if explicit else []
    toolkit = os.environ.get("ASCEND_HOME_PATH", "")
    if toolkit:
        candidates.append(str(Path(toolkit) / "tools/msopprof/bin/msopprof"))
    candidates.append(shutil.which("msopprof"))
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise ValueError("msopprof not found; source CANN set_env.sh or pass --profiler")


def full_commit_sha(value):
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("expected a full lowercase 40-character Git commit SHA")
    return value


def read_completion(actual_dir, revision, warmup, tests, case):
    completion = json.loads((actual_dir / "completion.json").read_text())
    metadata = case if isinstance(case, dict) else case.metadata()
    validate_completion(completion, metadata, revision, warmup, tests)
    return completion


def expected_dispatch(case, completion, cube_aiv_per_aic=None,
                      compact_binary_contract=None):
    """Validate capacity metadata or an explicitly requested binary entry.

    Explicit ratio is never inferred from a name. completion.cores remains a
    tiling budget; the parser separately verifies the actual launch dimensions.
    Compact requests assert an entry; they do not predict a source version's
    dispatch or turn its fallback into a split-output launch.
    """
    if (cube_aiv_per_aic is not None and type(cube_aiv_per_aic) is not int or
            cube_aiv_per_aic not in (None, 1, 2)):
        raise ValueError("Cube AIV-per-AIC ratio must be 1 or 2")
    if compact_binary_contract not in (None, "single-exp", "split-outputs", "static-split-outputs"):
        raise ValueError("unknown Compact binary contract")
    if compact_binary_contract is not None and cube_aiv_per_aic is not None:
        raise ValueError("Compact and Cube binary contracts are mutually exclusive")
    budget = completion["cores"]
    if type(budget) is not int or budget < 1:
        raise ValueError("invalid application core count")
    eps = struct.unpack("<f", struct.pack("<f", case.eps))[0]
    if compact_binary_contract is not None:
        fields = ("tile_tokens", "tokens_per_core", "extra_core_count", "h_pad",
                  "queried_aiv_cores", "available_core_limit")
        if any(type(completion.get(name)) is not int for name in fields):
            raise ValueError("Compact contract requires integer geometry and queried AIV budgets")
        queried = completion["queried_aiv_cores"]
        limit = completion["available_core_limit"]
        static_split = compact_binary_contract == "static-split-outputs"
        if (not 1 <= queried <= 0xffffffff or not 1 <= limit <= 0xffffffff or
                not static_split and queried > limit):
            raise ValueError("invalid Compact queried AIV budget or available limit")
        tokens = case.b * case.s
        if (not 2 <= case.h <= 8 or not math.isfinite(eps) or eps < 0 or
                budget != tokens or not 1 <= budget <= queried or
                completion["h_pad"] != 8 or completion["tile_tokens"] != 1 or
                completion["tokens_per_core"] != 1 or completion["extra_core_count"] != 0):
            raise ValueError("Compact contract requires H2..8/P8/nonnegative FP32 eps and one token per core")
        if static_split:
            if (any(type(x) is not int for x in (case.b, case.s, case.h, case.iters)) or
                    case.b != 1 or case.s != 1 or case.iters != 1 or
                    type(completion.get("h")) is not int or completion["h"] != case.h or
                    type(completion.get("iters")) is not int or completion["iters"] != 1 or
                    completion.get("shape") != case.shape or
                    any(type(x) is not int for x in completion.get("shape", []))):
                raise ValueError("static Compact split requires exact H2..8/N1/I1 completion geometry")
            actual_eps = completion.get("eps")
            try:
                same_eps = (type(actual_eps) in (int, float) and
                            struct.pack("<f", actual_eps) == struct.pack("<f", eps))
            except (OverflowError, struct.error):
                same_eps = False
            if not same_eps:
                raise ValueError("static Compact split completion epsilon differs from fixture")
            # This is the existing Host's T1 UB admission budget, including
            # its aligned input reserve and 8192-byte runtime reserve.
            input_reserve = (4 * case.h * (case.h + 2) + 31) // 32 * 32
            required_ub = input_reserve + 32 * case.h + 64 + 384 + 8192
            ub = completion.get("ub_bytes")
            effective = min(queried, limit)
            if type(ub) is not int or not required_ub <= ub < 1 << 64 or effective < 2:
                raise ValueError("static Compact split requires effective AIV>=2 and the Host T1 UB budget")
            return dict(source="requested_binary_contract", core_budget=budget,
                        compact_binary_contract_requested=compact_binary_contract,
                        queried_aiv_cores=queried, available_core_limit=limit,
                        effective_aiv_cores=effective,
                        required_resource_budgets=dict(aiv_cores=2, ub_bytes=required_ub),
                        guard_contract="requested exact H ABI: H2..8/P8/eps>=0,N1,I1,T1; two vector blocks",
                        guard_matched=True, kernel_family=compact_static_split_abi(case.h),
                        op_type="vector", block_dim=2, mix_block_dim=None,
                        aic_blocks=0, aiv_blocks=2)
        split = compact_binary_contract == "split-outputs"
        blocks = tokens * (2 if split else 1)
        if blocks > queried:
            raise ValueError("Compact requested launch exceeds queried AIV budget")
        return dict(source="requested_binary_contract", core_budget=budget,
                    compact_binary_contract_requested=compact_binary_contract,
                    queried_aiv_cores=queried, available_core_limit=limit,
                    guard_contract="requested exact entry: H2..8/P8/eps>=0,T1,N token owners; split launches 2N<=queried AIV",
                    guard_matched=True,
                    kernel_family=COMPACT_SPLIT_OUTPUTS_ABI if split else COMPACT_SINGLE_EXP_ABI,
                    op_type="vector", block_dim=blocks, mix_block_dim=None,
                    aic_blocks=0, aiv_blocks=blocks)
    eligible = (case.h > 64 and case.h % 8 == 0 and case.iters >= 2 and
                2 ** -40 <= eps < 1.0 and budget >= 2)
    if "cube_batch_tokens" in completion:
        batch = completion["cube_batch_tokens"]
        if type(batch) is not int or batch < 0 or batch == 1 or batch > 255:
            raise ValueError("cube_batch_tokens must be 0 or an integer in [2,255]")
        resources = completion.get("queried_resource_budgets")
        if resources is not None:
            names = ("aiv_cores", "aic_cores", "ub_bytes", "l1_bytes",
                     "l0a_bytes", "l0b_bytes", "l0c_bytes")
            if (not isinstance(resources, dict) or any(
                    type(resources.get(name)) is not int or resources[name] < 0
                    for name in names)):
                raise ValueError("invalid queried resource budgets")
            if "ub_bytes" in completion and completion["ub_bytes"] != resources["ub_bytes"]:
                raise ValueError("application UB budget mismatch")
        required = None
        if batch:
            if cube_aiv_per_aic != 2:
                raise ValueError("Cube batch requires explicit --cube-aiv-per-aic 2")
            if not eligible or case.h > 128 or budget % 2:
                raise ValueError("Cube batch capacity metadata violates numeric/core guard")
            if type(completion.get("tile_tokens")) is not int or completion["tile_tokens"] != batch:
                raise ValueError("Cube batch capacity differs from tile_tokens")
            if resources is None:
                raise ValueError("Cube batch requires queried resource budgets")
            tokens = case.b * case.s
            expected_lanes = 2 * min(resources["aic_cores"],
                                     min(tokens, resources["aiv_cores"]) // 2)
            if budget != expected_lanes or batch * budget > tokens:
                raise ValueError("Cube batch token/core budget mismatch")
            r = (case.h + 15) // 16 * 16
            required = dict(ub_bytes=4 * (r * r + 8 * r + 640 + 5 * 128 * batch) + 8192,
                            l1_bytes=2 * batch * (4 * r * r + 512) + 8192,
                            l0a_bytes=512, l0b_bytes=4 * r * r,
                            l0c_bytes=2 * batch * 16 * r * 4)
            if (2 * batch * 128 > case.h * case.h or
                    any(required[name] > resources[name] for name in required)):
                raise ValueError("Cube batch exceeds memory or mailbox budget")
        aic_blocks = budget // 2 if batch else 0
        return dict(source="cube_batch_capacity_metadata", core_budget=budget,
                    cube_aiv_per_aic_requested=cube_aiv_per_aic,
                    cube_batch_tokens=batch, queried_resource_budgets=resources,
                    required_resource_budgets=required,
                    guard_contract="batch: H72..128,H%8=0,I>=2,FP32eps>=2^-40,<1,T>=2,capacity/core/mailbox budgets",
                    guard_matched=bool(batch),
                    kernel_family=CUBE_FACTOR_KERNEL if batch else None,
                    op_type="mix" if batch else "vector",
                    block_dim=aic_blocks if batch else budget,
                    mix_block_dim=budget if batch else None,
                    aic_blocks=aic_blocks, aiv_blocks=budget)
    mixed = cube_aiv_per_aic is not None and eligible
    aic_blocks = budget // 2 if mixed else 0
    tiny_split = (not mixed and case.b * case.s == 1 and 2 <= case.h <= 8 and
                  case.iters == 1 and math.isfinite(eps) and eps >= 0 and
                  completion.get("queried_aiv_cores", 0) >= 2 and
                  completion.get("available_core_limit", 0) >= 2)
    vector_blocks = 2 * budget if tiny_split else budget
    return dict(source="requested_binary_contract", core_budget=budget,
                cube_aiv_per_aic_requested=cube_aiv_per_aic,
                guard_contract="1f8/142: H>64,H%8=0,I>=2,FP32eps>=2^-40,<1,core_budget>=2",
                guard_matched=eligible if cube_aiv_per_aic is not None else None,
                kernel_family=CUBE_FACTOR_KERNEL if mixed else None,
                op_type="mix" if mixed else "vector",
                block_dim=aic_blocks if mixed else vector_blocks,
                mix_block_dim=aic_blocks * cube_aiv_per_aic if mixed else None,
                aic_blocks=aic_blocks,
                aiv_blocks=aic_blocks * cube_aiv_per_aic if mixed else vector_blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--cases", help="comma separated stable IDs; overrides suite")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/testbench/sinkhorn_tb")
    parser.add_argument("--profiler")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--revision", help="tested source's full Git commit SHA")
    parser.add_argument("--inherit-correctness-from", type=full_commit_sha, metavar="FULL40SHA",
                        help="inherit an existing correctness conclusion when computation is unchanged; skip numerical checks")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1800, help="per process timeout seconds")
    contract = parser.add_mutually_exclusive_group()
    contract.add_argument("--cube-aiv-per-aic", type=int, choices=(1, 2),
                        help="explicit Cube binary contract: AIV blocks per AIC; new batch metadata requires 2, legacy 1f8/142 accepts 1 or 2; default expects vector and rejects selected MIX batches")
    contract.add_argument("--compact-binary-contract", choices=("single-exp", "split-outputs", "static-split-outputs"),
                          help="assert the exact serial Compact entry and N/2N vector launch; requires queried AIV resource metadata")
    args = parser.parse_args(argv)
    cases = select_cases(args.suite, args.cases)
    if args.list:
        print(json.dumps([c.metadata() for c in cases], ensure_ascii=False, indent=2))
        return 0
    if not args.out or not args.revision or len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--out and full lowercase commit SHA --revision are required")
    if args.device < 0 or not 0 <= args.warmup <= 1000 or not 1 <= args.tests <= 5000 or args.chunk_tokens < 1 or args.timeout < 1:
        parser.error("invalid device, warmup, tests, chunk-tokens or timeout")
    binary, out = args.binary.resolve(), args.out.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("testbench binary is missing or not executable")
    identity = subprocess.check_output([str(binary), "--revision"], text=True, timeout=30).strip()
    if identity != args.revision:
        parser.error("binary's compiled revision differs from --revision; rebuild the tested commit")
    tool = find_tool(args.profiler)
    if out.exists():
        parser.error("--out must be a new directory; raw results are never overwritten")
    os.umask(0o077)
    out.mkdir(parents=True)
    records = []
    inherited = args.inherit_correctness_from
    correctness_policy = {"mode": "inherited" if inherited else "checked",
                          "inherited_from": inherited,
                          "note": "运算方式未修改，沿用已有正确性结论" if inherited else None}
    run = {"schema_version": 1, "code_version": args.revision, "binary": str(binary),
           "started_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
           "machine": platform.machine(), "device": args.device, "profiler": tool,
           "requested_binary_contract": {"source": "requested_binary_contract", "cube_aiv_per_aic": args.cube_aiv_per_aic},
           "protocol": {**GM_LAYOUT, "warmup": args.warmup, "tests": args.tests, "profiler_warmup": 0,
                        "replay_mode": "application", "metrics": "BasicInfo", "synchronize_each_launch": True,
                        "correctness_preflight_launches": 0 if inherited else 1,
                        "profiled_numerical_outputs_checked": not bool(inherited),
                        "intermediate_profile_outputs_checked": False,
                        "independent_profile_processes_per_case": 1},
           "correctness_policy": correctness_policy, "cases": records, "success": False}
    if args.compact_binary_contract is not None:
        run["requested_binary_contract"]["compact_binary_contract"] = args.compact_binary_contract
    save(out / "run.json", run)
    command([tool, "--help"], out / "profiler-help.log", args.timeout)
    if shutil.which("npu-smi"):
        command(["npu-smi", "info"], out / "npu-smi.log", args.timeout)
    for case in cases:
        directory = out / case.case_id
        directory.mkdir()
        row = dict(case.metadata(), status="pending", correctness=None, latency_us=None, timing=None)
        row["correctness_policy"] = correctness_policy
        records.append(row)
        save(out / "run.json", run)
        try:
            stages = "generate / profile (inherited correctness)" if inherited else "generate / preflight / profile"
            print(f"{case.case_id}: {stages}", flush=True)
            row["fixture"] = fixture(case, directory / "fixture", fixture_seed(case.case_id, args.seed), args.chunk_tokens)
            preflight, profiled = directory / "preflight", directory / "profiled-output"
            if inherited:
                row.update(preflight_command=None, preflight=None, profiled_correctness=None)
            else:
                app = [str(binary), str(directory / "fixture"), str(preflight), str(args.device), "0", "1"]
                row["preflight_command"] = app
                command(app, directory / "preflight.log", args.timeout)
                row["preflight_completion"] = read_completion(preflight, args.revision, 0, 1, case)
                row["preflight"] = verify(case, directory / "fixture", preflight)
                if not row["preflight"]["passed"]:
                    row["correctness"] = False
                    raise ValueError("correctness preflight failed; timing not attempted")
            app = [str(binary), str(directory / "fixture"), str(profiled), str(args.device), str(args.warmup), str(args.tests)]
            argv = profiler_command(tool, shlex.join(app), directory / "profile", args.warmup, args.tests)
            row["profile_command"] = argv
            save(out / "run.json", run)
            command(argv, directory / "profiler.log", args.timeout)
            completion = read_completion(profiled, args.revision, args.warmup, args.tests, case)
            row["completion"] = completion
            if not inherited:
                row["profiled_correctness"] = verify(case, directory / "fixture", profiled)
                row["correctness"] = row["profiled_correctness"]["passed"]
            dispatch = expected_dispatch(case, completion, args.cube_aiv_per_aic,
                                         args.compact_binary_contract)
            row["expected_dispatch"] = dispatch
            row["timing"] = parse_tasks(directory / "profile", args.tests,
                blocks=dispatch["block_dim"], expected_kernel=dispatch["kernel_family"],
                op_type=dispatch["op_type"], mix_blocks=dispatch["mix_block_dim"])
            row["launch"] = row["timing"].get("launch")
            if row["correctness"] is False:
                raise ValueError("profiled final output failed correctness")
            row.update(status="success", latency_us=row["timing"]["median_us"])
            print(f"{case.case_id}: median={row['latency_us']:.6g} us; raw={row['timing']['samples_us']}", flush=True)
        except Exception as error:
            row.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed", error=str(error))
            print(f"{case.case_id}: {row['status']}: {error}", file=sys.stderr, flush=True)
        finally:
            save(directory / "result.json", row)
            save(out / "run.json", run)
    run.update(success=all(r["status"] == "success" for r in records), finished_at=datetime.now(timezone.utc).isoformat())
    save(out / "run.json", run)
    with (out / "cases.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(out / "run.json")
    return 0 if run["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
