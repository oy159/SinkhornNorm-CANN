#!/usr/bin/env python3
"""Export the shared implementation as the template's self-contained kernel.asc.

The original entry, reference and verification files are not edited. This tool
only bundles operator source; it never generates tensor values or runs a judge.
"""

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
SOURCES = (
    "include/sinkhorn_norm/tiling_data.h",
    "include/sinkhorn_norm/tiling_key.h",
    "include/sinkhorn_norm/tiling.h",
    "include/sinkhorn_norm/launch.h",
    "src/host/tiling.cpp",
    "src/host/tiling_key.cpp",
    "include/sinkhorn_norm/common/primitives.h",
    "include/sinkhorn_norm/kernels/f09_medium_dense_io.h",
    "include/sinkhorn_norm/kernels/f12_wide_blocked.h",
    "include/sinkhorn_norm/kernels/f04_narrow_single.h",
    "include/sinkhorn_norm/kernels/f01_tiny_split.h",
    "include/sinkhorn_norm/kernels/f02_tiny_single.h",
    "include/sinkhorn_norm/kernels/f03_h2_blocked.h",
    "include/sinkhorn_norm/kernels/f03_tiny_token_planes.h",
    "include/sinkhorn_norm/common/normalization_forecast.h",
    "include/sinkhorn_norm/kernels/f05_narrow_p16_short_batch.h",
    "include/sinkhorn_norm/kernels/f06_narrow_p16_paired_rows.h",
    "include/sinkhorn_norm/kernels/f07_narrow_p16_dense_io.h",
    "include/sinkhorn_norm/kernels/f08_narrow_p24_recurrent.h",
    "include/sinkhorn_norm/kernels/f10_medium_compact.h",
    "include/sinkhorn_norm/kernels/f11_medium_aligned_64.h",
    "include/sinkhorn_norm/kernels/f13_wide_compact_five.h",
    "include/sinkhorn_norm/kernels/f00_general_fallback.h",
    "include/sinkhorn_norm/kernels/f14_wide_rotating128.h",
    "src/kernel/sinkhorn_norm.asc",
    "src/host/launch.cpp",
    "src/host/template_adapter.inc",
)

EXPORT_ALIASES = {
    "SN_INL": "__aicore__ __attribute__((always_inline)) inline",
    "SN_AI": "__aicore__ inline",
    "SN_UB": "__ubuf__ float",
    "SN_VB": "PipeBarrier<PIPE_V>()",
    "SN_RC": "reinterpret_cast",
    "SN_V": "pipe_barrier(PIPE_V)",
    "SN_PB": "pipe_barrier",
    "SN_VM": "SetVectorMask",
    "SN_SC": "static_cast",
    "SN_U32": "uint32_t",
    "SN_U64": "uint64_t",
}
EXPORT_ALIAS_STARTS = frozenset(spelling[0] for spelling in EXPORT_ALIASES.values())


def compact_indentation(source, tight=False, *, use_aliases=False):
    """Trim indentation and line comments, preserving lines and literals.

    The maintained sources stay readable. In particular, indentation inside a
    raw string or a backslash-continued quoted literal is data, not formatting.
    Tight export packing also trims safe punctuator boundaries outside directives.
    """
    # Line splicing precedes tokenization and can split comment delimiters.
    # Keep these entire source units verbatim rather than partially lexing them.
    if "\\\n" in source or "\\\r\n" in source or "??/" in source:
        return source
    output = []
    state, quote, raw_end = "code", "", ""
    at_line_start = True
    in_directive = False
    i = 0
    while i < len(source):
        if at_line_start and state not in ("quoted", "raw"):
            while i < len(source) and source[i] in " \t":
                i += 1
            if i == len(source):
                break
            in_directive = state == "code" and source[i] == "#"
            # A standalone // comment is not a preprocessing token. Keep
            # its newline so diagnostics and __LINE__ retain their values.
            # Raw strings, block comments and spliced units are preserved.
            if state == "code" and source.startswith("//", i):
                end = source.find("\n", i)
                if end < 0:
                    break
                i = end
        at_line_start = False
        ch = source[i]
        pair = source[i:i + 2]
        if state == "raw" and source.startswith(raw_end, i):
            output.append(raw_end)
            i += len(raw_end)
            state = "code"
            continue
        if state == "quoted" and ch == "\\" and i + 1 < len(source):
            output.append(pair)
            at_line_start = pair[1] == "\n"
            i += 2
            continue
        if state == "quoted" and ch == quote:
            state = "code"
        elif state == "block" and pair == "*/":
            output.append(pair)
            i += 2
            state = "code"
            continue
        elif state == "line" and pair == "\\\n":
            output.append(pair)
            i += 2
            at_line_start = True
            continue
        elif state == "line" and ch == "\n":
            state = "code"
        elif state == "code":
            if pair == "//":
                end = source.find("\n", i)
                if end < 0:
                    break
                i = end
                continue
            if pair in ("//", "/*"):
                state = "line" if pair == "//" else "block"
                output.append(pair)
                i += 2
                continue
            if pair == 'R"':
                raw = re.match(r'R"([^ ()\\\t\r\n]{0,16})\(', source[i:])
                if raw:
                    raw_end = ")" + raw.group(1) + '"'
                    output.append(raw.group())
                    i += len(raw.group())
                    state = "raw"
                    continue
            if use_aliases and not in_directive and ch in EXPORT_ALIAS_STARTS and (
                i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_")
            ):
                for name, spelling in EXPORT_ALIASES.items():
                    end = i + len(spelling)
                    if source.startswith(spelling, i) and (
                        end == len(source) or not (source[end].isalnum() or source[end] == "_")
                    ):
                        output.append(name)
                        i = end
                        break
                else:
                    name = None
                if name is not None:
                    continue
            digit_separator = (ch == "'" and i > 0 and i + 1 < len(source)
                               and source[i - 1].isalnum() and source[i + 1].isalnum())
            if ch in ('"', "'") and not digit_separator:
                state, quote = "quoted", ch
        # These punctuation boundaries cannot join C++ tokens. Compact only
        # horizontal formatting in code; retain every newline and literal.
        if state == "code" and (ch == ")" or
                                (tight and not in_directive and ch in "(,;{}[]")):
            while output and output[-1] in (" ", "\t"):
                output.pop()
        if state == "code" and tight and not in_directive and ch == "=":
            n = len(output)
            while n and output[n - 1] in (" ", "\t"):
                n -= 1
            if n and output[n - 1] not in "<>!=+-*/%&|^":
                del output[n:]
        # Remove horizontal space around selected punctuators only when it
        # cannot create a compound operator, comment delimiter or digraph.
        blocked_pairs = {"/*", "*/", "//", "/=", ".*", ">*", "++", "--", "->", "+=", "-=",
                         "&&", "||", "*=", "&=", "|=", "^=", "%=", "%:", "%>", "<%",
                         "<<", ">>", "<=", ">=", "!=", "::", "<:", ":>"}
        if state == "code" and tight and not in_directive and ch in "*&|^%+-/<>!:":
            n = len(output)
            while n and output[n - 1] in (" ", "\t"):
                n -= 1
            exponent = n and output[n - 1] in "eEpP" and ch in "+-"
            if n and output[n - 1] + ch not in blocked_pairs and not exponent:
                del output[n:]
        output.append(ch)
        at_line_start = ch == "\n"
        i += 1
        if state == "code" and tight and not in_directive and ch in "*&|^%+-/<>!:":
            end = i
            while end < len(source) and source[end] in " \t":
                end += 1
            if end < len(source) and ch + source[end] not in blocked_pairs:
                i = end
        if state == "code" and (ch in ",;()}" or
                                (tight and not in_directive and ch in "{[]=")):
            end = i
            while end < len(source) and source[end] in " \t":
                end += 1
            if ch != "=" or source[end:end + 1] != "=":
                i = end
    return "".join(output)


def pack_lines(source):
    # The judge request budget counts each newline as an escaped character.
    # Joining after a finished statement does not change tokens. A preprocessor
    # directive keeps its following line, so a macro that ends in ';' cannot
    # absorb the invocation that uses it.
    packed = []
    in_directive = False
    hold_next = False
    for line in source.splitlines():
        if not line.strip():
            continue
        nxt = line.lstrip()
        previous = packed[-1] if packed else ""
        if (not in_directive and not hold_next and previous
                and "//" not in previous and "/*" not in previous
                and not previous.lstrip().startswith("#")
                and not nxt.startswith("#") and not nxt.startswith("//")
                and not previous.endswith("\\")
                and previous.endswith((";", "{", "}", ","))):
            packed[-1] = previous + nxt
            text = packed[-1]
        else:
            packed.append(nxt)
            text = nxt
        hold_next = False
        stripped = text.rstrip()
        if in_directive or stripped.lstrip().startswith("#"):
            in_directive = stripped.endswith("\\")
            hold_next = not in_directive
        else:
            in_directive = False
    return "\n".join(packed)


def render():
    parts = [
        "// Generated by scripts/export-template.py; edit the shared sources instead.\n",
        "// Self-contained operator implementation for the original template entry.\n",
        '#include "acl/acl.h"\n#include "kernel_operator.h"\n',
    ]
    parts.extend(f"#define {name} {spelling}\n" for name, spelling in EXPORT_ALIASES.items())
    for relative in SOURCES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        source = re.sub(r'^\s*#include "sinkhorn_norm/[^"\n]+"\s*$', "", source,
                        flags=re.MULTILINE)
        source = re.sub(r'^\s*#pragma once\s*$', "", source, flags=re.MULTILINE)
        source = compact_indentation(source, tight=True, use_aliases=True)
        # Keep section line numbers while omitting file banners from the upload.
        parts.append(f"\n\n{pack_lines(source)}\n")
    parts.extend(f"#undef {name}\n" for name in EXPORT_ALIASES)
    return "".join(parts).rstrip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission/kernel.asc")
    parser.add_argument("--check", action="store_true", help="Fail if export is absent or stale")
    args = parser.parse_args()
    source = render()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != source:
            parser.exit(1, f"Stale export: run {Path(__file__).name} --output {args.output}\n")
        print(f"Export is current: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(source.encode("utf-8"))
    print(f"Exported self-contained kernel: {args.output}")


if __name__ == "__main__":
    main()
