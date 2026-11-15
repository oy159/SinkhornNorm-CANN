"""Strict parsing of official OpBasicInfo Task Duration(us); no host timing."""
import csv
import math
from pathlib import Path
import re
import statistics


# Match the complete known ABI signatures for the compact/legacy entry
# and the independently dispatched medium/wide interleaved entry.
MODULAR_REGIME_KERNEL_NAMES = frozenset(
    f"sinkhorn_f{i:02d}_kernel" for i in range(15)
)
MODULAR_REGIME_ABI = re.compile(
    r"_Z\d+(?P<base>sinkhorn_f\d{2}_kernel)(?:ILi\d+EE)?vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE"
)

SPECIALIZED_KERNEL = re.compile(
    r"(?:_Z20sinkhorn_norm_kernelI(?:Lb[01]E){7,8}E|"
    r"_Z27sinkhorn_column_tree_kernelI(?:Lb[01]E)E|"
    r"_Z27sinkhorn_interleaved_kernelI(?:Lb[01]E){1,2}E)"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
COMPACT_EXTERNAL_KERNEL = (
    "_Z32sinkhorn_compact_external_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
COMPACT_SINGLE_EXP_KERNEL = "sinkhorn_compact_single_exp_kernel"
COMPACT_SINGLE_EXP_ABI = (
    "_Z34sinkhorn_compact_single_exp_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
COMPACT_SPLIT_OUTPUTS_KERNEL = "sinkhorn_compact_split_outputs_kernel"
COMPACT_SPLIT_OUTPUTS_ABI = (
    "_Z37sinkhorn_compact_split_outputs_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
COMPACT_OUTPUT_ABIS = {
    COMPACT_SINGLE_EXP_KERNEL: COMPACT_SINGLE_EXP_ABI,
    COMPACT_SPLIT_OUTPUTS_KERNEL: COMPACT_SPLIT_OUTPUTS_ABI,
}
COMPACT_STATIC_SPLIT_KERNEL = "sinkhorn_compact_static_split_kernel"
COMPACT_STATIC_SPLIT_ABI = re.compile(
    r"_Z36sinkhorn_compact_static_split_kernelILj(?P<h>[2-8])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
COMPACT_STATIC_SINGLE_KERNEL = "sinkhorn_compact_static_single_kernel"
COMPACT_STATIC_SINGLE_ABI = re.compile(
    r"_Z37sinkhorn_compact_static_single_kernelILj(?P<h>[2-8])EE"
    r"vPhS0_S0_S0_(?:N13sinkhorn_norm10TilingDataE|fj)")
COMPACT_DENSE_INPUT_KERNEL = "sinkhorn_compact_dense_input_kernel"
COMPACT_DENSE_INPUT_ABI = (
    "_Z35sinkhorn_compact_dense_input_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
COMPACT_TOKEN_PACK_KERNEL = "sinkhorn_compact_token_pack_kernel"
COMPACT_TOKEN_PACK_ABI = re.compile(
    r"_Z34sinkhorn_compact_token_pack_kernelILb(?P<dense_input>[01])E"
    r"Lb(?P<aligned_input>[01])EEvPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
NARROW_DENSE_IO_KERNEL = "sinkhorn_narrow_dense_io_kernel"
NARROW_DENSE_IO_ABI = "_Z31sinkhorn_narrow_dense_io_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
NARROW_DENSE_OUTPUT_KERNEL = "sinkhorn_narrow_dense_output_kernel"
NARROW_DENSE_OUTPUT_ABI = "_Z35sinkhorn_narrow_dense_output_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
NARROW_PAIR_ROWS_KERNEL = "sinkhorn_narrow_pair_rows_kernel"
NARROW_PAIR_ROWS_ABI = "_Z32sinkhorn_narrow_pair_rows_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
SHORT_NARROW_BATCH_KERNEL = "sinkhorn_short_narrow_batch_kernel"
SHORT_NARROW_BATCH_ABI = "_Z34sinkhorn_short_narrow_batch_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
NARROW_QUARTER_KERNEL = "sinkhorn_narrow_quarter_kernel"
NARROW_QUARTER_ABI = re.compile(
    r"_Z30sinkhorn_narrow_quarter_kernelILb(?P<grouped_rows>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
COMPACT_GATHER_COLUMNS_KERNEL = "sinkhorn_compact_gather_columns_kernel"
COMPACT_GATHER_COLUMNS_ABI = (
    "_Z38sinkhorn_compact_gather_columns_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
ALIGNED_INPUT_KERNEL = "sinkhorn_aligned_input_kernel"
ALIGNED_INPUT_ABI = re.compile(
    r"_Z29sinkhorn_aligned_input_kernelI(?P<flags>(?:Lb[01]E){8})E"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
ALIGNED_NARROW_KERNEL = "sinkhorn_aligned_narrow_kernel"
ALIGNED_NARROW_ABI = re.compile(
    r"_Z30sinkhorn_aligned_narrow_kernelILj(?P<h>16|24|32)EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
ALIGNED_MEDIUM_KERNEL = "sinkhorn_aligned_medium_kernel"
ALIGNED_MEDIUM_ABI = re.compile(
    r"_Z30sinkhorn_aligned_medium_kernelILb(?P<packed>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
ALIGNED_COMPACT_KERNEL = "sinkhorn_aligned_compact_kernel"
ALIGNED_COMPACT_ABI = (
    "_Z31sinkhorn_aligned_compact_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
ALIGNED_KERNELS = (ALIGNED_INPUT_KERNEL, ALIGNED_NARROW_KERNEL,
                   ALIGNED_MEDIUM_KERNEL, ALIGNED_COMPACT_KERNEL)
CUBE_FACTOR_KERNEL = "sinkhorn_cube_factor_kernel"
CUBE_FACTOR_ABI = "_Z27sinkhorn_cube_factor_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
INTERLEAVED_ROWS_KERNEL = "sinkhorn_interleaved_rows_kernel"
INTERLEAVED_ROWS_ABI = "_Z32sinkhorn_interleaved_rows_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE"
INTERLEAVED_GROUPED_NARROW_KERNEL = "sinkhorn_interleaved_grouped_narrow_kernel"
INTERLEAVED_GROUPED_NARROW_ABI = (
    "_Z42sinkhorn_interleaved_grouped_narrow_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_HALF_COLUMNS_KERNEL = "sinkhorn_interleaved_half_columns_kernel"
INTERLEAVED_HALF_COLUMNS_ABI = (
    "_Z40sinkhorn_interleaved_half_columns_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_ROTATING_QUARTERS_KERNEL = "sinkhorn_interleaved_rotating_quarters_kernel"
INTERLEAVED_ROTATING_QUARTERS_ABI = (
    "_Z45sinkhorn_interleaved_rotating_quarters_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_PAIRED_ROWS_KERNEL = "sinkhorn_interleaved_paired_rows_kernel"
INTERLEAVED_PAIRED_ROWS_ABI = (
    "_Z39sinkhorn_interleaved_paired_rows_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_PADDED_ROWS_KERNEL = "sinkhorn_interleaved_padded_rows_kernel"
INTERLEAVED_PADDED_ROWS_ABI = (
    "_Z39sinkhorn_interleaved_padded_rows_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_GROUPED_MAX_KERNEL = "sinkhorn_interleaved_grouped_max_kernel"
INTERLEAVED_GROUPED_MAX_ABI = re.compile(
    r"_Z39sinkhorn_interleaved_grouped_max_kernelILb(?P<wide>[01])E"
    r"Lb(?P<pack_rows>[01])ELb(?P<pad_rows>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_ROW_TREE_KERNEL = "sinkhorn_interleaved_row_tree_kernel"
INTERLEAVED_ROW_TREE_ABI = re.compile(
    r"_Z36sinkhorn_interleaved_row_tree_kernelILb[01]EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_COUNTER_ROWS_KERNEL = "sinkhorn_interleaved_counter_rows_kernel"
INTERLEAVED_COUNTER_ROWS_ABI = re.compile(
    r"_Z40sinkhorn_interleaved_counter_rows_kernelILb(?P<wide>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
SINGLE_TOKEN_ROWS_KERNEL = "sinkhorn_single_token_rows_kernel"
SINGLE_TOKEN_ROWS_ABI = re.compile(
    r"_Z33sinkhorn_single_token_rows_kernelILb(?P<merge_gates>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
MEDIUM_SINGLE_KERNEL = "sinkhorn_medium_single_kernel"
MEDIUM_SINGLE_ABI = re.compile(
    r"_Z29sinkhorn_medium_single_kernelILb(?P<packed>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
MEDIUM_WORKSPACE_KERNEL = "sinkhorn_medium_compact_workspace_kernel"
MEDIUM_WORKSPACE_ABI = re.compile(
    r"_Z40sinkhorn_medium_compact_workspace_kernelILb(?P<pad_rows>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
WIDE_COMPLETE_KERNEL = "sinkhorn_wide_complete_rounds_kernel"
WIDE_COMPLETE_ABI = re.compile(
    r"_Z36sinkhorn_wide_complete_rounds_kernelILb(?P<pad_rows>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
MEDIUM_DIRECT_KERNEL = "sinkhorn_medium_direct_columns_kernel"
MEDIUM_DIRECT_ABI = re.compile(
    r"_Z37sinkhorn_medium_direct_columns_kernelILb(?P<pad_rows>[01])ELb(?P<compact>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
PACKED_PADDED_ROWS_KERNEL = "sinkhorn_packed_padded_rows_kernel"
PACKED_PADDED_ROWS_ABI = re.compile(
    r"_Z34sinkhorn_packed_padded_rows_kernelILj(?P<pitch>16|24)EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
NARROW_STATIC_KERNEL = "sinkhorn_narrow_static_kernel"
NARROW_STATIC_ABI = re.compile(
    r"_Z29sinkhorn_narrow_static_kernelILj(?P<h>9|1[0-9]|2[0-9]|3[0-2])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm(?P<args>10TilingData|16CompactFrontArgs)E")
NARROW_SMALL_TILE_KERNEL = "sinkhorn_narrow_small_tile_kernel"
NARROW_SMALL_TILE_ABI = re.compile(
    r"_Z33sinkhorn_narrow_small_tile_kernelILj(?P<h>2[5-9]|3[0-2])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
WIDE_STATIC_KERNEL = "sinkhorn_wide_static_kernel"
WIDE_STATIC_ABI = re.compile(
    r"_Z27sinkhorn_wide_static_kernelILj(?P<h>72|80|88|96|104|112|120|128)EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")
INTERLEAVED_BANK_KERNEL = "sinkhorn_interleaved_bank_kernel"
INTERLEAVED_BANK_ABI = re.compile(
    r"_Z32sinkhorn_interleaved_bank_kernelILb(?P<overlay>[01])ELb(?P<packed_rows>[01])EE"
    r"vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")


def aligned_strategy(name):
    """Return the independent entry and every argument of a complete aligned ABI."""
    if not isinstance(name, str):
        return None
    match = ALIGNED_INPUT_ABI.fullmatch(name)
    if match:
        return ALIGNED_INPUT_KERNEL, tuple(value == "1" for value in
                                           re.findall(r"Lb([01])E", match["flags"]))
    match = ALIGNED_NARROW_ABI.fullmatch(name)
    if match:
        return ALIGNED_NARROW_KERNEL, (int(match["h"]),)
    match = ALIGNED_MEDIUM_ABI.fullmatch(name)
    if match:
        return ALIGNED_MEDIUM_KERNEL, (match["packed"] == "1",)
    if name == ALIGNED_COMPACT_ABI:
        return ALIGNED_COMPACT_KERNEL, ()
    return None


def compact_token_pack_strategy(name):
    """Accept complete DenseInput/AlignedInput flags for the three real entries."""
    match = COMPACT_TOKEN_PACK_ABI.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return None
    flags = tuple(match[key] == "1" for key in ("dense_input", "aligned_input"))
    return flags if flags != (True, True) else None


def narrow_quarter_strategy(name):
    """Accept exactly one GroupedRows bool and the complete quarter ABI."""
    match = NARROW_QUARTER_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["grouped_rows"] == "1" if match else None


def bank_strategy(name):
    """Return the two known bank template flags, or None if not recorded."""
    match = INTERLEAVED_BANK_ABI.fullmatch(name) if isinstance(name, str) else None
    return (match["overlay"] == "1", match["packed_rows"] == "1") if match else None


def counter_rows_strategy(name):
    """Accept exactly one recorded Wide bool and the complete entry ABI."""
    match = INTERLEAVED_COUNTER_ROWS_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["wide"] == "1" if match else None


def grouped_max_strategy(name):
    """Accept the complete Wide/PackRows/PadRows ABI for three real entries."""
    match = INTERLEAVED_GROUPED_MAX_ABI.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return None
    strategy = tuple(match[key] == "1" for key in ("wide", "pack_rows", "pad_rows"))
    return strategy if strategy in ((True, True, False), (True, False, False),
                                    (False, False, True)) else None


def single_token_rows_strategy(name):
    """Accept exactly one MergeGates bool and the complete isolated entry ABI."""
    match = SINGLE_TOKEN_ROWS_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["merge_gates"] == "1" if match else None


def medium_single_strategy(name):
    """Accept the old medium entry's complete single Packed bool, including false."""
    match = MEDIUM_SINGLE_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["packed"] == "1" if match else None


def medium_workspace_strategy(name):
    """Return the complete compact workspace entry's padding flag."""
    match = MEDIUM_WORKSPACE_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["pad_rows"] == "1" if match else None

def wide_complete_strategy(name):
    """Return the complete wide complete-round entry's aligned-row flag."""
    match = WIDE_COMPLETE_ABI.fullmatch(name) if isinstance(name, str) else None
    return match["pad_rows"] == "1" if match else None


def medium_direct_strategy(name):
    match = MEDIUM_DIRECT_ABI.fullmatch(name) if isinstance(name, str) else None
    return (match['pad_rows'] == '1', match['compact'] == '1') if match else None


def packed_padded_rows_strategy(name):
    """Accept only the complete uint32 Pitch16/Pitch24 entry ABI."""
    match = PACKED_PADDED_ROWS_ABI.fullmatch(name) if isinstance(name, str) else None
    return int(match["pitch"]) if match else None


def narrow_static_strategy(name):
    """Accept exactly one canonical uint32 H9..32 and the full entry ABI."""
    match = NARROW_STATIC_ABI.fullmatch(name) if isinstance(name, str) else None
    if not match:
        return None
    h = int(match["h"])
    if match["args"] == "16CompactFrontArgs" and not 17 <= h <= 24:
        return None
    return h


def narrow_small_tile_strategy(name):
    """Accept one canonical uint32 H25..32 and the complete small-tile ABI."""
    match = NARROW_SMALL_TILE_ABI.fullmatch(name) if isinstance(name, str) else None
    return int(match["h"]) if match else None


def wide_static_strategy(name):
    """Accept one canonical uint32 aligned H72..128 and the complete ABI."""
    match = WIDE_STATIC_ABI.fullmatch(name) if isinstance(name, str) else None
    return int(match["h"]) if match else None


def compact_static_split_strategy(name):
    """Accept exactly one canonical uint32 H2..8 and the complete split ABI."""
    match = COMPACT_STATIC_SPLIT_ABI.fullmatch(name) if isinstance(name, str) else None
    return int(match["h"]) if match else None


def compact_static_split_abi(h):
    if type(h) is not int or not 2 <= h <= 8:
        raise ValueError("static Compact split H must be an integer in [2,8]")
    return (f"_Z36sinkhorn_compact_static_split_kernelILj{h}EE"
            "vPhS0_S0_S0_N13sinkhorn_norm10TilingDataE")


def compact_static_single_strategy(name):
    """Accept exactly one canonical uint32 H2..8 and the complete single ABI."""
    match = COMPACT_STATIC_SINGLE_ABI.fullmatch(name) if isinstance(name, str) else None
    return int(match["h"]) if match else None


def compact_static_single_abi(h, *, lean=False):
    if type(h) is not int or not 2 <= h <= 8:
        raise ValueError("static Compact single H must be an integer in [2,8]")
    suffix = "fj" if lean else "N13sinkhorn_norm10TilingDataE"
    return f"_Z37sinkhorn_compact_static_single_kernelILj{h}EEvPhS0_S0_S0_{suffix}"


def interleaved_entry(name):
    """Keep public entry identity separate from the shared reporting family."""
    public_entries = {
        "sinkhorn_interleaved_wide_counter_sync_kernel":
            "_Z45sinkhorn_interleaved_wide_counter_sync_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE",
        "sinkhorn_interleaved_quarter_bank_skew_kernel":
            "_Z45sinkhorn_interleaved_quarter_bank_skew_kernelPhS_S_S_N13sinkhorn_norm10TilingDataE",
        INTERLEAVED_GROUPED_NARROW_KERNEL: INTERLEAVED_GROUPED_NARROW_ABI,
        INTERLEAVED_HALF_COLUMNS_KERNEL: INTERLEAVED_HALF_COLUMNS_ABI,
        INTERLEAVED_PAIRED_ROWS_KERNEL: INTERLEAVED_PAIRED_ROWS_ABI,
        INTERLEAVED_ROTATING_QUARTERS_KERNEL: INTERLEAVED_ROTATING_QUARTERS_ABI,
        INTERLEAVED_PADDED_ROWS_KERNEL: INTERLEAVED_PADDED_ROWS_ABI,
    }
    for short, full in public_entries.items():
        if name in (short, full):
            return short
    if name == INTERLEAVED_BANK_KERNEL or bank_strategy(name) is not None:
        return INTERLEAVED_BANK_KERNEL
    if name in (INTERLEAVED_ROWS_KERNEL, INTERLEAVED_ROWS_ABI):
        return INTERLEAVED_ROWS_KERNEL
    if name == INTERLEAVED_ROW_TREE_KERNEL or (
            isinstance(name, str) and INTERLEAVED_ROW_TREE_ABI.fullmatch(name)):
        return INTERLEAVED_ROW_TREE_KERNEL
    if counter_rows_strategy(name) is not None:
        return INTERLEAVED_COUNTER_ROWS_KERNEL
    if name == INTERLEAVED_GROUPED_MAX_KERNEL or grouped_max_strategy(name) is not None:
        return INTERLEAVED_GROUPED_MAX_KERNEL
    if name == "sinkhorn_interleaved_kernel" or (
            isinstance(name, str) and name.startswith("_Z27sinkhorn_interleaved_kernelI")
            and SPECIALIZED_KERNEL.fullmatch(name)):
        return "sinkhorn_interleaved_kernel"
    return None


def kernel_family(name):
    """Recognize complete known names without merging vector and MIX entries."""
    if name in (NARROW_DENSE_IO_KERNEL, NARROW_DENSE_IO_ABI):
        return NARROW_DENSE_IO_KERNEL
    if name in (NARROW_DENSE_OUTPUT_KERNEL, NARROW_DENSE_OUTPUT_ABI):
        return NARROW_DENSE_OUTPUT_KERNEL
    if name in (NARROW_PAIR_ROWS_KERNEL, NARROW_PAIR_ROWS_ABI):
        return NARROW_PAIR_ROWS_KERNEL
    if name in (SHORT_NARROW_BATCH_KERNEL, SHORT_NARROW_BATCH_ABI):
        return SHORT_NARROW_BATCH_KERNEL
    if name == NARROW_QUARTER_KERNEL or narrow_quarter_strategy(name) is not None:
        return NARROW_QUARTER_KERNEL
    if name in (COMPACT_GATHER_COLUMNS_KERNEL, COMPACT_GATHER_COLUMNS_ABI):
        return COMPACT_GATHER_COLUMNS_KERNEL
    if name == COMPACT_TOKEN_PACK_KERNEL or compact_token_pack_strategy(name) is not None:
        return COMPACT_TOKEN_PACK_KERNEL
    aligned = aligned_strategy(name)
    if aligned is not None:
        return aligned[0]
    if interleaved_entry(name) is not None:
        return "sinkhorn_interleaved_kernel"
    families = ("sinkhorn_norm_kernel", "sinkhorn_interleaved_kernel",
                "sinkhorn_column_tree_kernel", "sinkhorn_compact_external_kernel",
                CUBE_FACTOR_KERNEL)
    if name in families:
        return name
    if name == COMPACT_EXTERNAL_KERNEL:
        return "sinkhorn_compact_external_kernel"
    for short, full in COMPACT_OUTPUT_ABIS.items():
        if name == full:
            return short
    if name == COMPACT_DENSE_INPUT_ABI:
        return COMPACT_DENSE_INPUT_KERNEL
    if name == CUBE_FACTOR_ABI:
        return CUBE_FACTOR_KERNEL
    if single_token_rows_strategy(name) is not None:
        return SINGLE_TOKEN_ROWS_KERNEL
    if name == MEDIUM_WORKSPACE_KERNEL or medium_workspace_strategy(name) is not None:
        return MEDIUM_WORKSPACE_KERNEL
    if name == MEDIUM_DIRECT_KERNEL or medium_direct_strategy(name) is not None:
        return MEDIUM_DIRECT_KERNEL
    if name == WIDE_COMPLETE_KERNEL or wide_complete_strategy(name) is not None:
        return WIDE_COMPLETE_KERNEL
    if medium_single_strategy(name) is not None:
        return MEDIUM_SINGLE_KERNEL
    if packed_padded_rows_strategy(name) is not None:
        return PACKED_PADDED_ROWS_KERNEL
    if narrow_static_strategy(name) is not None:
        return NARROW_STATIC_KERNEL
    if narrow_small_tile_strategy(name) is not None:
        return NARROW_SMALL_TILE_KERNEL
    if wide_static_strategy(name) is not None:
        return WIDE_STATIC_KERNEL
    if compact_static_single_strategy(name) is not None:
        return COMPACT_STATIC_SINGLE_KERNEL
    if compact_static_split_strategy(name) is not None:
        return COMPACT_STATIC_SPLIT_KERNEL
    if isinstance(name, str):
        if name in MODULAR_REGIME_KERNEL_NAMES:
            return name
        m = MODULAR_REGIME_ABI.fullmatch(name)
        if m is not None:
            return m.group("base")
    if isinstance(name, str) and SPECIALIZED_KERNEL.fullmatch(name):
        return next((family for family in families
                     if name.startswith(f"_Z{len(family)}{family}I")), None)
    return None


def resolve_modular_regime_capture_name(reported_name, path, capture_root):
    if reported_name in MODULAR_REGIME_KERNEL_NAMES:
        return reported_name, None
    m = MODULAR_REGIME_ABI.fullmatch(reported_name)
    if m is not None:
        return reported_name, None
    for directory in path.parents:
        if directory == capture_root:
            break
        dm = MODULAR_REGIME_ABI.fullmatch(directory.name)
        if dm is not None:
            if reported_name == dm.group("base") or reported_name == directory.name:
                return directory.name, dict(source_csv=str(path), abi_directory=str(directory), kernel_name=directory.name)
    return reported_name, None


def resolve_bank_capture_name(reported_name, path, capture_root):
    """Resolve a short bank name only from its unique actual capture folder."""
    if interleaved_entry(reported_name) != INTERLEAVED_BANK_KERNEL:
        return reported_name, None
    directories = []
    for directory in path.parents:
        if directory.name.startswith("_Z32sinkhorn_interleaved_bank_kernel"):
            if bank_strategy(directory.name) is None:
                raise ValueError("malformed bank ABI capture directory")
            directories.append(directory)
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous bank ABI capture directories")
    if not directories:
        return reported_name, None
    directory = directories[0]
    if reported_name != INTERLEAVED_BANK_KERNEL and reported_name != directory.name:
        raise ValueError("bank CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_counter_capture_name(reported_name, path, capture_root):
    """Resolve a short Counter name only from its unique complete capture ABI."""
    if (reported_name != INTERLEAVED_COUNTER_ROWS_KERNEL and
            counter_rows_strategy(reported_name) is None):
        return reported_name, None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_counter_rows_kernel", directory.name):
            if counter_rows_strategy(directory.name) is None:
                raise ValueError("malformed Counter ABI capture directory")
            directories.append(directory)
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous Counter ABI capture directories")
    if not directories:
        if reported_name == INTERLEAVED_COUNTER_ROWS_KERNEL:
            raise ValueError("short Counter CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name != INTERLEAVED_COUNTER_ROWS_KERNEL and reported_name != directory.name:
        raise ValueError("Counter CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_single_token_rows_capture_name(reported_name, path, capture_root):
    """Resolve the short name only from one actual complete single-token ABI."""
    if (reported_name != SINGLE_TOKEN_ROWS_KERNEL and
            single_token_rows_strategy(reported_name) is None):
        return reported_name, None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_single_token_rows_kernel", directory.name):
            if single_token_rows_strategy(directory.name) is None:
                raise ValueError("malformed single-token rows ABI capture directory")
            directories.append(directory)
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous single-token rows ABI capture directories")
    if not directories:
        if reported_name == SINGLE_TOKEN_ROWS_KERNEL:
            raise ValueError("short single-token rows CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name != SINGLE_TOKEN_ROWS_KERNEL and reported_name != directory.name:
        raise ValueError("single-token rows CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_medium_single_capture_name(reported_name, path, capture_root):
    """Resolve a medium short name only from one complete actual Packed ABI."""
    reported_medium = reported_name == MEDIUM_SINGLE_KERNEL or medium_single_strategy(reported_name) is not None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_medium_single_kernel", directory.name):
            if medium_single_strategy(directory.name) is None:
                raise ValueError("malformed medium single ABI capture directory")
            directories.append(directory)
        elif reported_medium and directory.name.startswith("_Z"):
            raise ValueError("medium single CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous medium single ABI capture directories")
    if not directories:
        if reported_name == MEDIUM_SINGLE_KERNEL:
            raise ValueError("short medium single CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (MEDIUM_SINGLE_KERNEL, directory.name):
        raise ValueError("medium single CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_medium_workspace_capture_name(reported_name, path, capture_root):
    """Resolve a medium short name only from one complete actual PadRows ABI."""
    reported_medium = reported_name == MEDIUM_WORKSPACE_KERNEL or medium_workspace_strategy(reported_name) is not None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_medium_compact_workspace_kernel", directory.name):
            if medium_workspace_strategy(directory.name) is None:
                raise ValueError("malformed medium workspace ABI capture directory")
            directories.append(directory)
        elif reported_medium and directory.name.startswith("_Z"):
            raise ValueError("medium workspace CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous medium workspace ABI capture directories")
    if not directories:
        if reported_name == MEDIUM_WORKSPACE_KERNEL:
            raise ValueError("short medium workspace CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (MEDIUM_WORKSPACE_KERNEL, directory.name):
        raise ValueError("medium workspace CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)
def resolve_medium_direct_capture_name(reported_name, path, capture_root):
    """Resolve a medium short name only from one complete actual PadRows ABI."""
    reported_medium = reported_name == MEDIUM_DIRECT_KERNEL or medium_direct_strategy(reported_name) is not None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_medium_direct_columns_kernel", directory.name):
            if medium_direct_strategy(directory.name) is None:
                raise ValueError("malformed medium direct columns ABI capture directory")
            directories.append(directory)
        elif reported_medium and directory.name.startswith("_Z"):
            raise ValueError("medium direct columns CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous medium direct columns ABI capture directories")
    if not directories:
        if reported_name == MEDIUM_DIRECT_KERNEL:
            raise ValueError("short medium direct columns CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (MEDIUM_DIRECT_KERNEL, directory.name):
        raise ValueError("medium direct columns CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)

def resolve_wide_complete_capture_name(reported_name, path, capture_root):
    """Resolve a medium short name only from one complete actual PadRows ABI."""
    reported_medium = reported_name == WIDE_COMPLETE_KERNEL or wide_complete_strategy(reported_name) is not None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_wide_complete_rounds_kernel", directory.name):
            if wide_complete_strategy(directory.name) is None:
                raise ValueError("malformed wide complete rounds ABI capture directory")
            directories.append(directory)
        elif reported_medium and directory.name.startswith("_Z"):
            raise ValueError("wide complete rounds CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous wide complete rounds ABI capture directories")
    if not directories:
        if reported_name == WIDE_COMPLETE_KERNEL:
            raise ValueError("short wide complete rounds CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (WIDE_COMPLETE_KERNEL, directory.name):
        raise ValueError("wide complete rounds CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_packed_padded_rows_capture_name(reported_name, path, capture_root):
    """Resolve a short Packed name only from one actual complete Pitch ABI."""
    if (reported_name != PACKED_PADDED_ROWS_KERNEL and
            packed_padded_rows_strategy(reported_name) is None):
        return reported_name, None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_packed_padded_rows_kernel", directory.name):
            if packed_padded_rows_strategy(directory.name) is None:
                raise ValueError("malformed padded Packed row ABI capture directory")
            directories.append(directory)
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous padded Packed row ABI capture directories")
    if not directories:
        if reported_name == PACKED_PADDED_ROWS_KERNEL:
            raise ValueError("short padded Packed row CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name != PACKED_PADDED_ROWS_KERNEL and reported_name != directory.name:
        raise ValueError("padded Packed row CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_narrow_static_capture_name(reported_name, path, capture_root):
    """Resolve short static geometry names only from one complete actual ABI."""
    if (reported_name != NARROW_STATIC_KERNEL and
            narrow_static_strategy(reported_name) is None):
        return reported_name, None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_static_kernel", directory.name):
            if narrow_static_strategy(directory.name) is None:
                raise ValueError("malformed narrow static geometry ABI capture directory")
            directories.append(directory)
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow static geometry ABI capture directories")
    if not directories:
        if reported_name == NARROW_STATIC_KERNEL:
            raise ValueError("short narrow static CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name != NARROW_STATIC_KERNEL and reported_name != directory.name:
        raise ValueError("narrow static CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_narrow_small_tile_capture_name(reported_name, path, capture_root):
    """Use only a unique complete small-tile ABI on this CSV's ancestry."""
    directories = []
    recognized = (reported_name == NARROW_SMALL_TILE_KERNEL or
                  narrow_small_tile_strategy(reported_name) is not None)
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_small_tile_kernel", directory.name):
            if narrow_small_tile_strategy(directory.name) is None:
                raise ValueError("malformed narrow small-tile ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("narrow small-tile CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow small-tile ABI capture directories")
    if not directories:
        if reported_name == NARROW_SMALL_TILE_KERNEL:
            raise ValueError("short narrow small-tile CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (NARROW_SMALL_TILE_KERNEL, directory.name):
        raise ValueError("narrow small-tile CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_wide_static_capture_name(reported_name, path, capture_root):
    """Bind a short wide name to this CSV's unique complete ABI ancestor."""
    directories = []
    recognized = reported_name == WIDE_STATIC_KERNEL or wide_static_strategy(reported_name) is not None
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_wide_static_kernel", directory.name):
            if wide_static_strategy(directory.name) is None:
                raise ValueError("malformed wide static ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("wide static CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous wide static ABI capture directories")
    if not directories:
        if reported_name == WIDE_STATIC_KERNEL:
            raise ValueError("short wide static CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (WIDE_STATIC_KERNEL, directory.name):
        raise ValueError("wide static CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_dense_input_capture_name(reported_name, path, capture_root):
    """Resolve a short dense-input name only from its exact non-template ABI."""
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_dense_input_kernel", directory.name):
            if directory.name != COMPACT_DENSE_INPUT_ABI:
                raise ValueError("malformed compact dense input ABI capture directory")
            directories.append(directory)
        elif (reported_name in (COMPACT_DENSE_INPUT_KERNEL, COMPACT_DENSE_INPUT_ABI)
              and directory.name.startswith("_Z")):
            raise ValueError("compact dense input CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous compact dense input ABI capture directories")
    if not directories:
        if reported_name == COMPACT_DENSE_INPUT_KERNEL:
            raise ValueError("short compact dense input CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name != COMPACT_DENSE_INPUT_KERNEL and reported_name != directory.name:
        raise ValueError("compact dense input CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_output_capture_name(reported_name, path, capture_root):
    """Resolve either independent Compact entry only from its actual full ABI."""
    directories = []
    recognized = any(reported_name in (short, full)
                     for short, full in COMPACT_OUTPUT_ABIS.items())
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_(?:single_exp|split_outputs)_kernel", directory.name):
            if directory.name not in COMPACT_OUTPUT_ABIS.values():
                raise ValueError("malformed Compact output ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("Compact output CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous Compact output ABI capture directories")
    if not directories:
        if reported_name in COMPACT_OUTPUT_ABIS:
            raise ValueError("short Compact output CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    short = next(short for short, full in COMPACT_OUTPUT_ABIS.items()
                 if full == directory.name)
    if reported_name not in (short, directory.name):
        raise ValueError("Compact output CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_aligned_capture_name(reported_name, path, capture_root):
    """Require one actual complete ABI before accepting any aligned short name."""
    reported_aligned = reported_name in ALIGNED_KERNELS or aligned_strategy(reported_name) is not None
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_aligned_(?:input|narrow|medium|compact)_kernel", directory.name):
            if aligned_strategy(directory.name) is None:
                raise ValueError("malformed aligned input ABI capture directory")
            directories.append(directory)
        elif reported_aligned and directory.name.startswith("_Z"):
            raise ValueError("aligned input CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous aligned input ABI capture directories")
    if not directories:
        if reported_name in ALIGNED_KERNELS:
            raise ValueError("short aligned CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    entry = aligned_strategy(directory.name)[0]
    if reported_name not in (entry, directory.name):
        raise ValueError("aligned CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_token_pack_capture_name(reported_name, path, capture_root):
    """Bind a token-pack short name to its unique actual complete ABI ancestor."""
    recognized = (reported_name == COMPACT_TOKEN_PACK_KERNEL or
                  compact_token_pack_strategy(reported_name) is not None)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_token_pack_kernel", directory.name):
            if compact_token_pack_strategy(directory.name) is None:
                raise ValueError("malformed Compact token-pack ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("Compact token-pack CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous Compact token-pack ABI capture directories")
    if not directories:
        if reported_name == COMPACT_TOKEN_PACK_KERNEL:
            raise ValueError("short Compact token-pack CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (COMPACT_TOKEN_PACK_KERNEL, directory.name):
        raise ValueError("Compact token-pack CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_narrow_quarter_capture_name(reported_name, path, capture_root):
    """Require one real full quarter ABI ancestor for either reported spelling."""
    recognized = (reported_name == NARROW_QUARTER_KERNEL or
                  narrow_quarter_strategy(reported_name) is not None)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_quarter_kernel", directory.name):
            if narrow_quarter_strategy(directory.name) is None:
                raise ValueError("malformed narrow quarter ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("narrow quarter CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow quarter ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("narrow quarter CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (NARROW_QUARTER_KERNEL, directory.name):
        raise ValueError("narrow quarter CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_gather_columns_capture_name(reported_name, path, capture_root):
    """Require one real complete non-template gather-columns ABI ancestor."""
    recognized = reported_name in (COMPACT_GATHER_COLUMNS_KERNEL,
                                   COMPACT_GATHER_COLUMNS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_gather_columns_kernel", directory.name):
            if directory.name != COMPACT_GATHER_COLUMNS_ABI:
                raise ValueError("malformed Compact gather-columns ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("Compact gather-columns CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous Compact gather-columns ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("Compact gather-columns CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (COMPACT_GATHER_COLUMNS_KERNEL, directory.name):
        raise ValueError("Compact gather-columns CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_static_split_capture_name(reported_name, path, capture_root):
    """Resolve short static split names only from one actual complete H ABI."""
    recognized = (reported_name == COMPACT_STATIC_SPLIT_KERNEL or
                  compact_static_split_strategy(reported_name) is not None)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_static_split_kernel", directory.name):
            if compact_static_split_strategy(directory.name) is None:
                raise ValueError("malformed static Compact split ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("static Compact split CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous static Compact split ABI capture directories")
    if not directories:
        if reported_name == COMPACT_STATIC_SPLIT_KERNEL:
            raise ValueError("short static Compact split CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (COMPACT_STATIC_SPLIT_KERNEL, directory.name):
        raise ValueError("static Compact split CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_compact_static_single_capture_name(reported_name, path, capture_root):
    """Resolve short static single names only from one actual complete H ABI."""
    recognized = (reported_name == COMPACT_STATIC_SINGLE_KERNEL or
                  compact_static_single_strategy(reported_name) is not None)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_compact_static_single_kernel", directory.name):
            if compact_static_single_strategy(directory.name) is None:
                raise ValueError("malformed static Compact single ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("static Compact single CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous static Compact single ABI capture directories")
    if not directories:
        if reported_name == COMPACT_STATIC_SINGLE_KERNEL:
            raise ValueError("short static Compact single CSV name requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if reported_name not in (COMPACT_STATIC_SINGLE_KERNEL, directory.name):
        raise ValueError("static Compact single CSV name conflicts with capture directory ABI")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_grouped_narrow_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the independent row engine."""
    recognized = reported_name in (INTERLEAVED_GROUPED_NARROW_KERNEL,
                                   INTERLEAVED_GROUPED_NARROW_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_grouped_narrow_kernel", directory.name):
            if directory.name != INTERLEAVED_GROUPED_NARROW_ABI:
                raise ValueError("malformed grouped narrow ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("grouped narrow CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous grouped narrow ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("grouped narrow CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("grouped narrow capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_narrow_dense_io_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the dense-io layout."""
    recognized = reported_name in (NARROW_DENSE_IO_KERNEL,
                                   NARROW_DENSE_IO_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_dense_io_kernel", directory.name):
            if directory.name != NARROW_DENSE_IO_ABI:
                raise ValueError("malformed narrow dense io ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("narrow dense io CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow dense io ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("narrow dense io CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("narrow dense io capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)

def resolve_narrow_dense_output_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the dense-output layout."""
    recognized = reported_name in (NARROW_DENSE_OUTPUT_KERNEL,
                                   NARROW_DENSE_OUTPUT_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_dense_output_kernel", directory.name):
            if directory.name != NARROW_DENSE_OUTPUT_ABI:
                raise ValueError("malformed narrow dense output ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("narrow dense output CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow dense output ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("narrow dense output CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("narrow dense output capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)

def resolve_narrow_pair_rows_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the compact row-pair layout."""
    recognized = reported_name in (NARROW_PAIR_ROWS_KERNEL,
                                   NARROW_PAIR_ROWS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_narrow_pair_rows_kernel", directory.name):
            if directory.name != NARROW_PAIR_ROWS_ABI:
                raise ValueError("malformed narrow pair rows ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("narrow pair rows CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous narrow pair rows ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("narrow pair rows CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("narrow pair rows capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)

def resolve_short_narrow_batch_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the complete short row batch."""
    recognized = reported_name in (SHORT_NARROW_BATCH_KERNEL,
                                   SHORT_NARROW_BATCH_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_short_narrow_batch_kernel", directory.name):
            if directory.name != SHORT_NARROW_BATCH_ABI:
                raise ValueError("malformed short narrow batch ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("short narrow batch CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous short narrow batch ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("short narrow batch CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("short narrow batch capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)

def resolve_half_columns_capture_name(reported_name, path, capture_root):
    """Require one actual exact ABI ancestor for the independent column tree."""
    recognized = reported_name in (INTERLEAVED_HALF_COLUMNS_KERNEL,
                                   INTERLEAVED_HALF_COLUMNS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_half_columns_kernel", directory.name):
            if directory.name != INTERLEAVED_HALF_COLUMNS_ABI:
                raise ValueError("malformed half columns ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("half columns CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous half columns ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("half columns CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("half columns capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_interleaved_padded_rows_capture_name(reported_name, path, capture_root):
    """Recover the legacy padded entry only from one exact actual ABI ancestor."""
    recognized = reported_name in (INTERLEAVED_PADDED_ROWS_KERNEL,
                                   INTERLEAVED_PADDED_ROWS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_padded_rows_kernel", directory.name):
            if directory.name != INTERLEAVED_PADDED_ROWS_ABI:
                raise ValueError("malformed interleaved padded rows ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("interleaved padded rows CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous interleaved padded rows ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("interleaved padded rows CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("interleaved padded rows capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_grouped_max_capture_name(reported_name, path, capture_root):
    """Resolve three-bool row maxima only from a unique actual full ABI."""
    recognized = (reported_name == INTERLEAVED_GROUPED_MAX_KERNEL or
                  grouped_max_strategy(reported_name) is not None)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_grouped_max_kernel", directory.name):
            if grouped_max_strategy(directory.name) is None:
                raise ValueError("malformed grouped max ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("grouped max CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous grouped max ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("grouped max CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized or reported_name not in (INTERLEAVED_GROUPED_MAX_KERNEL, directory.name):
        raise ValueError("grouped max capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_paired_rows_capture_name(reported_name, path, capture_root):
    """Bind this independent entry only to one actual complete ABI ancestor."""
    recognized = reported_name in (INTERLEAVED_PAIRED_ROWS_KERNEL,
                                   INTERLEAVED_PAIRED_ROWS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_paired_rows_kernel", directory.name):
            if directory.name != INTERLEAVED_PAIRED_ROWS_ABI:
                raise ValueError("malformed paired rows ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("paired rows CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous paired rows ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("paired rows CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("paired rows capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def resolve_rotating_quarters_capture_name(reported_name, path, capture_root):
    """Bind this independent entry only to one actual complete ABI ancestor."""
    recognized = reported_name in (INTERLEAVED_ROTATING_QUARTERS_KERNEL,
                                   INTERLEAVED_ROTATING_QUARTERS_ABI)
    directories = []
    for directory in path.parents:
        if re.match(r"_Z\d+sinkhorn_interleaved_rotating_quarters_kernel", directory.name):
            if directory.name != INTERLEAVED_ROTATING_QUARTERS_ABI:
                raise ValueError("malformed rotating quarters ABI capture directory")
            directories.append(directory)
        elif recognized and directory.name.startswith("_Z"):
            raise ValueError("rotating quarters CSV conflicts with another capture entry")
        if directory == capture_root:
            break
    if len(directories) > 1:
        raise ValueError("ambiguous rotating quarters ABI capture directories")
    if not directories:
        if recognized:
            raise ValueError("rotating quarters CSV requires a complete ABI capture directory")
        return reported_name, None
    directory = directories[0]
    if not recognized:
        raise ValueError("rotating quarters capture directory conflicts with CSV entry")
    return directory.name, dict(source_csv=str(path), abi_directory=str(directory),
                                kernel_name=directory.name)


def profiler_command(tool, application, output, warmup=1, tests=5):
    # Application itself makes warmup+tests launches. Never add profiler warmups.
    # Do not pass an exact base-name filter: templated entries have mangled
    # names. This binary only launches this operator; validate every CSV row.
    return [str(tool), "--application=" + application, "--output=" + str(output),
            "--aic-metrics=BasicInfo",
            "--replay-mode=application", "--warm-up=0",
            "--launch-skip-before-match=" + str(warmup),
            "--launch-count=" + str(tests), "--kill=off"]


def parse_tasks(root, expected=5, kernel="sinkhorn_norm_kernel", blocks=None,
                expected_kernel=None, op_type=None, mix_blocks=None):
    """Parse actual launch dimensions, optionally enforcing a binary contract.

    blocks is the launched Block Dim, not a tiling resource budget. MIX rows
    must always identify their AIC/AIV dimensions and Op Type. Older known
    vector tables may omit Op Type/Mix Block Dim.
    """
    if expected_kernel is not None and kernel_family(expected_kernel) is None:
        raise ValueError("unknown expected kernel family")
    if op_type not in (None, "vector", "mix"):
        raise ValueError("unknown expected Op Type")
    capture_root = Path(root)
    files = sorted(capture_root.rglob("*OpBasicInfo*.csv"))
    if not files:
        raise ValueError("official OpBasicInfo CSV not found")
    # CANN 9.0 exports one CSV per task: OPPROF_*/kernel/{0..N-1}/CSV.
    # Also accept a single aggregate table, but never merge unrelated captures.
    tables = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if "Task Duration(us)" not in (reader.fieldnames or []):
            raise ValueError(f"Task Duration(us) column missing: {path}")
        if rows:
            tables.append((path, rows))
    if not tables:
        raise ValueError("OpBasicInfo tables are empty")
    if len(tables) > 1:
        parents = {path.parent.parent for path, _ in tables}
        if len(parents) != 1 or any(not path.parent.name.isdigit() or len(rows) != 1 for path, rows in tables):
            raise ValueError("multiple tables do not form one numbered task capture")
        indices = sorted(int(path.parent.name) for path, _ in tables)
        if indices != list(range(expected)):
            raise ValueError("duplicate, missing or extra task indices")
        tables.sort(key=lambda item: int(item[0].parent.name))
    rows = [(path, row) for path, table in tables for row in table]
    if len(rows) != expected:
        raise ValueError(f"expected exactly {expected} tasks, found {len(rows)}")
    values, raw, selected_name, selected_launch = [], [], None, None
    reported_names, resolution_paths = [], []
    for path, row in rows:
        reported_name = row.get("Kernel Name", row.get("Op Name", ""))
        name, resolution = resolve_modular_regime_capture_name(reported_name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_medium_workspace_capture_name(reported_name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_medium_direct_capture_name(reported_name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_wide_complete_capture_name(reported_name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_quarter_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_gather_columns_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_token_pack_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_rotating_quarters_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_paired_rows_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_wide_static_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_grouped_narrow_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_dense_io_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_dense_output_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_pair_rows_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_short_narrow_batch_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_half_columns_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_interleaved_padded_rows_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_grouped_max_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_static_single_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_static_split_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_output_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_aligned_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_medium_single_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_compact_dense_input_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_small_tile_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_narrow_static_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_packed_padded_rows_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_single_token_rows_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_counter_capture_name(name, path, capture_root)
        if resolution is None:
            name, resolution = resolve_bank_capture_name(name, path, capture_root)
        reported_names.append(reported_name)
        if resolution is not None and resolution not in resolution_paths:
            resolution_paths.append(resolution)
        family = kernel_family(name)
        known_entry = kernel == "sinkhorn_norm_kernel" and family is not None
        if family is None or (name != kernel and not known_entry and family != kernel_family(kernel)):
            raise ValueError(f"unexpected kernel name: {name!r}")
        if kernel != "sinkhorn_norm_kernel" and (
                (interleaved_entry(name) == INTERLEAVED_BANK_KERNEL) !=
                (interleaved_entry(kernel) == INTERLEAVED_BANK_KERNEL)):
            raise ValueError("profiled bank public entry differs from requested kernel")
        if expected_kernel is not None and family != kernel_family(expected_kernel):
            raise ValueError("profiled kernel family differs from expected dispatch")
        for requested in (kernel, expected_kernel):
            workspace_strategy = medium_workspace_strategy(requested)
            if workspace_strategy is not None and medium_workspace_strategy(name) != workspace_strategy:
                raise ValueError("profiled medium workspace padding differs from requested dispatch")
            direct_strategy = medium_direct_strategy(requested)
            if direct_strategy is not None and medium_direct_strategy(name) != direct_strategy:
                raise ValueError("profiled medium direct layout differs from requested dispatch")
            complete_strategy = wide_complete_strategy(requested)
            if complete_strategy is not None and wide_complete_strategy(name) != complete_strategy:
                raise ValueError("profiled wide complete row layout differs from requested dispatch")
            entry = interleaved_entry(requested)
            if entry in ("sinkhorn_interleaved_wide_counter_sync_kernel",
                         "sinkhorn_interleaved_quarter_bank_skew_kernel",
                         INTERLEAVED_ROTATING_QUARTERS_KERNEL) and \
                    interleaved_entry(name) != entry:
                raise ValueError("profiled wide Counter public entry differs from requested dispatch")
            if entry == INTERLEAVED_GROUPED_NARROW_KERNEL and interleaved_entry(name) != entry:
                raise ValueError("profiled grouped narrow entry differs from requested dispatch")
            if entry == INTERLEAVED_HALF_COLUMNS_KERNEL and interleaved_entry(name) != entry:
                raise ValueError("profiled half columns entry differs from requested dispatch")
            if entry == INTERLEAVED_PAIRED_ROWS_KERNEL and interleaved_entry(name) != entry:
                raise ValueError("profiled paired rows entry differs from requested dispatch")
            if entry == INTERLEAVED_PADDED_ROWS_KERNEL and interleaved_entry(name) != entry:
                raise ValueError("profiled interleaved padded rows entry differs from requested dispatch")
            if entry == INTERLEAVED_GROUPED_MAX_KERNEL and interleaved_entry(name) != entry:
                raise ValueError("profiled grouped max entry differs from requested dispatch")
            max_strategy = grouped_max_strategy(requested)
            if max_strategy is not None and grouped_max_strategy(name) != max_strategy:
                raise ValueError("profiled grouped max bool arguments differ from requested dispatch")
        # Historical callers use the plain family for the isolated rows entry.
        # Bank is a separate policy: do not let that family shorthand accept it.
        if expected_kernel is not None and (
                (interleaved_entry(name) == INTERLEAVED_BANK_KERNEL) !=
                (interleaved_entry(expected_kernel) == INTERLEAVED_BANK_KERNEL)):
            raise ValueError("profiled bank public entry differs from expected dispatch")
        expected_strategy = bank_strategy(expected_kernel) or bank_strategy(kernel)
        if expected_strategy is not None and bank_strategy(name) != expected_strategy:
            raise ValueError("profiled bank template strategy differs from expected dispatch")
        counter_strategy = counter_rows_strategy(expected_kernel)
        if counter_strategy is None:
            counter_strategy = counter_rows_strategy(kernel)
        if counter_strategy is not None and counter_rows_strategy(name) != counter_strategy:
            raise ValueError("profiled Counter row strategy differs from expected dispatch")
        single_token_strategy = single_token_rows_strategy(expected_kernel)
        if single_token_strategy is None:
            single_token_strategy = single_token_rows_strategy(kernel)
        if (single_token_strategy is not None and
                single_token_rows_strategy(name) != single_token_strategy):
            raise ValueError("profiled single-token rows strategy differs from expected dispatch")
        for requested in (kernel, expected_kernel):
            strategy = medium_single_strategy(requested)
            if strategy is not None and medium_single_strategy(name) != strategy:
                raise ValueError("profiled medium single Packed argument differs from expected dispatch")
        padded_strategy = packed_padded_rows_strategy(expected_kernel)
        if padded_strategy is None:
            padded_strategy = packed_padded_rows_strategy(kernel)
        if padded_strategy is not None and packed_padded_rows_strategy(name) != padded_strategy:
            raise ValueError("profiled padded Packed row pitch differs from expected dispatch")
        static_strategy = narrow_static_strategy(expected_kernel)
        if static_strategy is None:
            static_strategy = narrow_static_strategy(kernel)
        if static_strategy is not None and narrow_static_strategy(name) != static_strategy:
            raise ValueError("profiled narrow static geometry differs from expected dispatch")
        for requested in (kernel, expected_kernel):
            small_tile_strategy = narrow_small_tile_strategy(requested)
            if (small_tile_strategy is not None and
                    narrow_small_tile_strategy(name) != small_tile_strategy):
                raise ValueError("profiled narrow small-tile geometry differs from expected dispatch")
            wide_strategy = wide_static_strategy(requested)
            if wide_strategy is not None and wide_static_strategy(name) != wide_strategy:
                raise ValueError("profiled wide static geometry differs from expected dispatch")
        for requested in (kernel, expected_kernel):
            quarter_strategy = narrow_quarter_strategy(requested)
            if (quarter_strategy is not None and
                    narrow_quarter_strategy(name) != quarter_strategy):
                raise ValueError("profiled narrow quarter GroupedRows argument differs from requested dispatch")
            strategy = aligned_strategy(requested)
            if strategy is not None and aligned_strategy(name) != strategy:
                raise ValueError("profiled aligned entry/template arguments differ from expected dispatch")
            token_pack_strategy = compact_token_pack_strategy(requested)
            if (token_pack_strategy is not None and
                    compact_token_pack_strategy(name) != token_pack_strategy):
                raise ValueError("profiled Compact token-pack bool arguments differ from expected dispatch")
        for requested in (kernel, expected_kernel):
            one_strategy = compact_static_single_strategy(requested)
            if one_strategy is not None and name != requested:
                raise ValueError("profiled static Compact single H/parameter ABI differs from expected dispatch")
        tiny_strategy = compact_static_split_strategy(expected_kernel)
        if tiny_strategy is None:
            tiny_strategy = compact_static_split_strategy(kernel)
        if tiny_strategy is not None and compact_static_split_strategy(name) != tiny_strategy:
            raise ValueError("profiled static Compact split H differs from expected dispatch")
        if selected_name is not None and name != selected_name:
            raise ValueError("profiled tasks contain different kernel specializations")
        if reported_name != reported_names[0]:
            raise ValueError("profiled tasks contain different reported kernel names")
        selected_name = name
        try:
            block_dim = int(row["Block Dim"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid profiled Block Dim") from error
        if block_dim < 1 or blocks is not None and block_dim != blocks:
            raise ValueError("profiled Block Dim differs from application metadata")
        actual_type = row.get("Op Type")
        if family == COMPACT_STATIC_SPLIT_KERNEL and (block_dim != 2 or actual_type != "vector"):
            raise ValueError("static Compact split requires exactly two vector blocks")
        expected_type = "mix" if family == CUBE_FACTOR_KERNEL else "vector"
        if actual_type is not None and actual_type != expected_type:
            raise ValueError("profiled Op Type differs from kernel family")
        if op_type is not None and (expected_type != op_type or
                                    actual_type is not None and actual_type != op_type):
            raise ValueError("profiled Op Type differs from expected dispatch")
        if expected_type == "mix":
            if actual_type != "mix":
                raise ValueError("MIX capture requires Op Type=mix")
            try:
                mix_dim = int(row["Mix Block Dim"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("MIX capture requires a positive Mix Block Dim") from error
            if mix_dim not in (block_dim, 2 * block_dim):
                raise ValueError("MIX dimensions do not describe a supported 1:1 or 1:2 split")
            if mix_blocks is not None and mix_dim != mix_blocks:
                raise ValueError("profiled Mix Block Dim differs from expected dispatch")
            launch = dict(op_type="mix", block_dim=block_dim, mix_block_dim=mix_dim,
                          aic_blocks=block_dim, aiv_blocks=mix_dim,
                          aiv_per_aic=mix_dim // block_dim)
        else:
            if row.get("Mix Block Dim") not in (None, "", "NA", "N/A", "0") or mix_blocks is not None:
                raise ValueError("vector capture must not declare MIX blocks")
            launch = dict(op_type="vector", block_dim=block_dim, mix_block_dim=None,
                          aic_blocks=0, aiv_blocks=block_dim, aiv_per_aic=None)
        if selected_launch is not None and launch != selected_launch:
            raise ValueError("profiled tasks contain different launch dimensions")
        selected_launch = launch
        value = float(row["Task Duration(us)"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid official task duration")
        values.append(value)
        raw.append(row["Task Duration(us)"])
    return {"source_csvs": [str(path) for path, _ in tables], "metric": "OpBasicInfo Task Duration(us)",
            "kernel_name": selected_name, "kernel_family": kernel_family(selected_name), "launch": selected_launch,
            "reported_kernel_name": reported_names[0],
            "kernel_name_resolution_source": "verified_capture_directory" if resolution_paths else "csv",
            "kernel_name_resolution_paths": resolution_paths,
            "task_duration_raw": raw, "samples_us": values,
            "median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values), "mean_us": statistics.mean(values),
            "task_rows": [row for _, row in rows], "samples": len(values), "independent_processes": 1}
