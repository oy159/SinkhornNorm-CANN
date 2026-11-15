"""Validate measured GM layout; never upgrade or rewrite historical evidence."""
import math
import struct


PROTOCOL_VERSION = "sinkhorn-testbench-gm512-v1"
GM_LAYOUT = {"protocol_version": PROTOCOL_VERSION,
             "gm_data_offset_bytes": 512, "guard_bytes": 64}
CAPTURE_PROTOCOL = {"warmup": 1, "tests": 5, "profiler_warmup": 0,
                    "metrics": "BasicInfo", "replay_mode": "application",
                    "synchronize_each_launch": True}
_UINT64_LIMIT = 1 << 64
_ALIGNMENTS = (128, 256, 512, 4096)


def _require(condition, message):
    if not condition:
        raise ValueError("measurement protocol: " + message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and minimum <= value < _UINT64_LIMIT,
             name + " must be a bounded integer")
    return value


def _fp32(value):
    _require(type(value) in (int, float), "epsilon is not numeric data")
    try:
        packed = struct.pack("<f", value)
    except (OverflowError, struct.error) as error:
        raise ValueError("measurement protocol: epsilon is not finite FP32") from error
    _require(math.isfinite(struct.unpack("<f", packed)[0]), "epsilon is not finite FP32")
    return packed


def validate_layout_label(metadata):
    _require(isinstance(metadata, dict), "layout declaration is missing")
    for key, expected in GM_LAYOUT.items():
        _require(type(metadata.get(key)) is type(expected) and metadata[key] == expected,
                 "unsupported or missing " + key)


def validate_capture_protocol(metadata):
    validate_layout_label(metadata)
    for key, expected in CAPTURE_PROTOCOL.items():
        _require(type(metadata.get(key)) is type(expected) and metadata[key] == expected,
                 "unsupported capture setting " + key)
    _require(type(metadata.get("independent_profile_processes_per_case", 1)) is int and
             metadata.get("independent_profile_processes_per_case", 1) == 1,
             "capture does not represent one application process")


def validate_completion(completion, fixture, revision, warmup, tests):
    """Return actual residues after strict checks; no allocator alignment assumed.

    New runs may record a truthful unaligned allocation. Automatic reuse applies
    an additional explicit aligned-baseline policy, rather than inventing zeros.
    """
    _require(isinstance(revision, str) and len(revision) == 40 and
             all(c in "0123456789abcdef" for c in revision), "expected full lowercase code SHA")
    validate_layout_label(completion)
    _require(completion.get("code_version") == revision, "application completion version mismatch")
    for key, expected in (("launches", warmup + tests), ("warmup", warmup), ("tests", tests)):
        _require(type(completion.get(key)) is int and completion[key] == expected,
                 "application launch count mismatch: " + key)
    _require(completion.get("input_unchanged") is True and completion.get("guards_passed") is True,
             "application input or buffer guard check failed")
    _integer(completion.get("cores"), "cores", 1)
    _require(isinstance(fixture, dict), "fixture metadata missing")
    b, s, h = (_integer(fixture.get(k), "fixture." + k, 1) for k in ("b", "s", "h"))
    _require(2 <= h <= 128 and fixture.get("dtype") == "float32", "unsupported fixture type/width")
    iters = _integer(fixture.get("iters"), "fixture.iters", 1)
    _require(iters <= 32, "invalid iteration count")
    shape = [b, s, h * h + 2 * h]
    _require(fixture.get("shape", shape) == shape, "fixture shape is inconsistent")
    _require(isinstance(completion.get("shape"), list) and
             all(type(x) is int for x in completion["shape"]) and completion["shape"] == shape,
             "completion shape differs from fixture")
    for key, expected in (("h", h), ("iters", iters)):
        _require(type(completion.get(key)) is int and completion[key] == expected,
                 "completion differs from fixture: " + key)
    _require(_fp32(completion.get("eps")) == _fp32(fixture.get("eps")), "completion epsilon differs from fixture")
    n = b * s
    sizes = {"mixes": n * shape[2] * 4, "pre": n * h * 4,
             "post": n * h * 4, "comb": n * h * h * 4}
    buffers = completion.get("buffers")
    _require(isinstance(buffers, dict) and set(buffers) == set(sizes), "four buffer records are required")
    residues, allocations = {}, []
    for name, size in sizes.items():
        buf = buffers[name]
        _require(isinstance(buf, dict), name + " buffer metadata is missing")
        values = {key: _integer(buf.get(key), name + "." + key) for key in
                  ("bytes", "allocation_bytes", "base_uintptr", "data_uintptr", "data_offset",
                   "front_guard_uintptr", "back_guard_uintptr")}
        _require(values["bytes"] == size and values["allocation_bytes"] == size + 1024,
                 name + " buffer byte count differs from fixture/layout")
        base, data = values["base_uintptr"], values["data_uintptr"]
        end = base + values["allocation_bytes"]
        _require(base > 0 and end <= _UINT64_LIMIT, name + " allocation address overflows")
        _require(values["data_offset"] == 512 and data == base + 512,
                 name + " payload is not base+512")
        _require(values["front_guard_uintptr"] == data - 64 and
                 values["back_guard_uintptr"] == data + size and
                 base <= data - 64 and data + size + 64 <= end,
                 name + " guards are not adjacent/in allocation")
        residues[name] = {}
        for alignment in _ALIGNMENTS:
            for kind, address in (("base", base), ("data", data)):
                key = f"{kind}_mod{alignment}"
                actual = _integer(buf.get(key), name + "." + key)
                _require(actual == address % alignment, name + "." + key + " contradicts address")
                residues[name][key] = actual
        allocations.append((base, end))
    allocations.sort()
    _require(all(left[1] <= right[0] for left, right in zip(allocations, allocations[1:])),
             "buffer allocations overlap")
    return {"status": "current", "protocol_version": PROTOCOL_VERSION,
            "aligned_to_512": all(r["base_mod512"] == r["data_mod512"] == 0 for r in residues.values()),
            "actual_residues": residues}


def inspect_history_protocol(protocol, completion, fixture, revision):
    """Classify retained evidence without mutating/rejecting legacy records."""
    declarations = [m.get("protocol_version") if isinstance(m, dict) else None
                    for m in (protocol, completion)]
    if all(value is None for value in declarations):
        return {"status": "legacy", "reason": "no GM protocol declaration", "declared_versions": declarations}
    try:
        validate_capture_protocol(protocol)
        return validate_completion(completion, fixture, revision, 1, 5)
    except ValueError as error:
        status = "invalid" if all(value == PROTOCOL_VERSION for value in declarations) else "unknown"
        return {"status": status, "reason": str(error), "declared_versions": declarations}
