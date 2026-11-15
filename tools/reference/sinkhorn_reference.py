#!/usr/bin/env python3
"""Offline FP32 reference and raw-file comparator; never part of the operator.

Uses only the Python standard library. Arithmetic is rounded to IEEE binary32
after each elementary operation. Sequential CPU reductions and libm exp can
differ slightly from an NPU reduction tree or its exp approximation. This tool
does not replace the official judge or precision checker.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def add(a, b):
    return f32(a + b)


def div(a, b):
    if b == 0.0:
        return math.nan if a == 0.0 else math.copysign(math.inf, a * b)
    return f32(a / b)


def sum_f32(values):
    total = 0.0
    for value in values:
        total = add(total, value)
    return total


def sigmoid(value):
    # Both branches avoid exponent overflow for finite, extreme FP32 logits.
    if value >= 0.0:
        return div(1.0, add(1.0, f32(math.exp(-value))))
    exp_value = f32(math.exp(value))
    return div(exp_value, add(1.0, exp_value))


def normalize_columns(matrix, h, eps):
    denominators = [
        add(sum_f32(matrix[row * h + column] for row in range(h)), eps)
        for column in range(h)
    ]
    return [
        div(matrix[row * h + column], denominators[column])
        for row in range(h)
        for column in range(h)
    ]


def normalize_rows(matrix, h, eps):
    normalized = []
    for row in range(h):
        values = matrix[row * h : (row + 1) * h]
        denominator = add(sum_f32(values), eps)
        normalized.extend(div(value, denominator) for value in values)
    return normalized


def reference_token(mixes, h=4, iters=20, eps=1e-6):
    """Return pre/post/row-major comb for exactly one input token."""
    if not 2 <= h <= 128 or not 1 <= iters <= 32:
        raise ValueError("h must be 2..128 and iters must be 1..32")
    if len(mixes) != h * (h + 2):
        raise ValueError("token width must equal h*(h+2)")
    values = [f32(value) for value in mixes]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("the reference requires finite FP32 input logits")
    eps = f32(eps)
    if not math.isfinite(eps):
        raise ValueError("eps must be finite and representable as FP32")
    pre = [add(sigmoid(value), eps) for value in values[:h]]
    post = [f32(2.0 * sigmoid(value)) for value in values[h : 2 * h]]
    comb = []
    for row in range(h):
        logits = values[2 * h + row * h : 2 * h + (row + 1) * h]
        row_max = max(logits)
        exponentials = [f32(math.exp(f32(value - row_max))) for value in logits]
        denominator = sum_f32(exponentials)
        comb.extend(add(div(value, denominator), eps) for value in exponentials)

    # The first normalization is a column normalization, even for iters=1.
    comb = normalize_columns(comb, h, eps)
    for _ in range(iters - 1):
        comb = normalize_rows(comb, h, eps)
        comb = normalize_columns(comb, h, eps)
    return pre, post, comb


def write_floats(stream, values):
    stream.write(struct.pack("<" + "f" * len(values), *values))


def validate_problem(b, s, h, iters, eps):
    if not 1 <= b <= 128 or not 1 <= s <= 8192:
        raise ValueError("b must be 1..128 and s must be 1..8192")
    if not 2 <= h <= 128 or not 1 <= iters <= 32:
        raise ValueError("h must be 2..128 and iters must be 1..32")
    if not math.isfinite(f32(eps)):
        raise ValueError("eps must be finite and representable as FP32")


def expected_manifest(b, s, h, iters, eps, input_path, input_sha256):
    return {
        "format_version": 1,
        "offline_reference_only": True,
        "dtype": "float32",
        "byte_order": "little",
        "shape": [b, s, h * (h + 2)],
        "hc_mult": h,
        "hc_sinkhorn_iters": iters,
        "eps": f32(eps),
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "outputs": {
            "pre": {"shape": [b, s, h], "file": "expected_pre.bin"},
            "post": {"shape": [b, s, h], "file": "expected_post.bin"},
            "comb": {"shape": [b, s, h, h], "file": "expected_comb.bin"},
        },
        "arithmetic": "FP32 elementary operations, sequential reductions, libm exp",
    }


def compute_files(input_path, output_dir, b, s, h, iters, eps):
    validate_problem(b, s, h, iters, eps)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    token_bytes = h * (h + 2) * 4
    expected_bytes = b * s * token_bytes
    if input_path.stat().st_size != expected_bytes:
        raise ValueError(f"input must contain exactly {expected_bytes} bytes")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"expected_{name}.bin" for name in ("pre", "post", "comb")]
    if input_path.resolve() in [path.resolve() for path in output_paths]:
        raise ValueError("input must not be overwritten by an expected output")
    digest = hashlib.sha256()
    with input_path.open("rb") as source, output_paths[0].open("wb") as pre_out, \
            output_paths[1].open("wb") as post_out, output_paths[2].open("wb") as comb_out:
        for _ in range(b * s):
            raw = source.read(token_bytes)
            digest.update(raw)
            token = struct.unpack("<" + "f" * (token_bytes // 4), raw)
            pre, post, comb = reference_token(token, h, iters, eps)
            write_floats(pre_out, pre)
            write_floats(post_out, post)
            write_floats(comb_out, comb)
    manifest = expected_manifest(b, s, h, iters, eps, input_path, digest.hexdigest())
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest_path


def generate_fixture(args):
    validate_problem(args.b, args.s, args.h, args.iters, args.eps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "mixes.bin"
    rng = random.Random(args.seed)

    def next_value(index):
        if args.distribution == "constant":
            return args.constant
        if args.distribution == "negative":
            return rng.uniform(-20.0, -0.01)
        if args.distribution == "wide":
            return rng.uniform(-80.0, 80.0)
        if args.distribution == "extreme":
            return (-3e38, -1000.0, -80.0, 0.0, 80.0, 1000.0, 3e38)[index % 7]
        return rng.uniform(-2.0, 2.0)

    if not math.isfinite(f32(args.constant)):
        raise ValueError("constant must be finite and representable as FP32")
    width = args.h * (args.h + 2)
    with input_path.open("wb") as stream:
        for token in range(args.b * args.s):
            write_floats(stream, [f32(next_value(token * width + i)) for i in range(width)])
    manifest_path = compute_files(input_path, output_dir, args.b, args.s, args.h,
                                  args.iters, args.eps)
    manifest = json.loads(manifest_path.read_text())
    manifest["generator"] = {"seed": args.seed, "distribution": args.distribution}
    if args.distribution == "constant":
        manifest["generator"]["constant"] = f32(args.constant)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest_path


def compare_raw(expected_path, actual_path, count, atol=1e-4, rtol=1e-4):
    """Compare all elements using abs_error < atol OR relative_error < rtol."""
    expected_path, actual_path = Path(expected_path), Path(actual_path)
    expected_bytes = count * 4
    result = {
        "elements": count,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_path.stat().st_size,
        "passed": False,
        "mismatch_count": None,
        "first_mismatch_index": None,
        "max_absolute_error": None,
        "max_relative_error": None,
        "nonfinite_count": None,
    }
    if expected_path.stat().st_size != expected_bytes:
        raise ValueError(f"reference file has incorrect length: {expected_path}")
    if result["actual_bytes"] != expected_bytes:
        result["reason"] = "actual file length does not match the full output shape"
        return result
    mismatches = 0
    nonfinite = 0
    max_abs = max_rel = 0.0
    offset = 0
    with expected_path.open("rb") as expected, actual_path.open("rb") as actual:
        while offset < count:
            chunk_count = min(16384, count - offset)
            expected_values = struct.iter_unpack("<f", expected.read(chunk_count * 4))
            actual_values = struct.iter_unpack("<f", actual.read(chunk_count * 4))
            for local_index, ((ref,), (got,)) in enumerate(zip(expected_values, actual_values)):
                if not math.isfinite(ref) or not math.isfinite(got):
                    nonfinite += 1
                    passed = False
                else:
                    absolute = abs(got - ref)
                    relative = absolute / abs(ref) if ref != 0.0 else (0.0 if absolute == 0.0 else math.inf)
                    max_abs = max(max_abs, absolute)
                    max_rel = max(max_rel, relative)
                    passed = absolute < atol or relative < rtol or got == ref
                if not passed:
                    mismatches += 1
                    if result["first_mismatch_index"] is None:
                        result["first_mismatch_index"] = offset + local_index
            offset += chunk_count
    result.update({
        "passed": mismatches == 0,
        "mismatch_count": mismatches,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel if math.isfinite(max_rel) else None,
        "nonfinite_count": nonfinite,
    })
    return result


def verify_fixture(args):
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be finite and nonnegative")
    fixture = Path(args.fixture)
    manifest = json.loads((fixture / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("dtype") != "float32" or manifest.get("byte_order") != "little":
        raise ValueError("unsupported manifest format, dtype or byte order")
    results = {}
    for name in ("pre", "post", "comb"):
        spec = manifest["outputs"][name]
        count = math.prod(spec["shape"])
        results[name] = compare_raw(fixture / spec["file"], getattr(args, name),
                                    count, args.atol, args.rtol)
    return {
        "offline_reference_only": True,
        "passed": all(result["passed"] for result in results.values()),
        "criterion": "absolute_error < atol OR relative_error < rtol; exact values pass",
        "atol": args.atol,
        "rtol": args.rtol,
        "outputs": results,
    }


def add_problem_arguments(parser):
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="write input and offline expected raw FP32 outputs")
    add_problem_arguments(generate)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--seed", type=int, default=20261023)
    generate.add_argument("--distribution", choices=("random", "constant", "negative", "wide", "extreme"), default="random")
    generate.add_argument("--constant", type=float, default=0.0)
    reference = subcommands.add_parser("reference", help="compute offline expected outputs from an existing raw input")
    add_problem_arguments(reference)
    reference.add_argument("--input", required=True)
    reference.add_argument("--output-dir", required=True)
    verify = subcommands.add_parser("verify", help="compare all three device raw outputs with a fixture")
    verify.add_argument("--fixture", required=True)
    for name in ("pre", "post", "comb"):
        verify.add_argument(f"--{name}", required=True)
    verify.add_argument("--atol", type=float, default=1e-4)
    verify.add_argument("--rtol", type=float, default=1e-4)
    verify.add_argument("--report")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            print(generate_fixture(args))
        elif args.command == "reference":
            print(compute_files(args.input, args.output_dir, args.b, args.s, args.h,
                                args.iters, args.eps))
        else:
            report = verify_fixture(args)
            output = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if args.report:
                Path(args.report).write_text(output)
            print(output, end="")
            return 0 if report["passed"] else 1
    except (ValueError, OSError, KeyError, struct.error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
