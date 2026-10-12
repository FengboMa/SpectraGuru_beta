"""Readers for instrument-vendor spectrum files.

Each reader turns one vendor file into a Raman-shift axis and a block of
spectra. Only the axis and the intensities are kept; map positions, times,
images and other metadata are ignored. Streamlit is not imported here, so the
module can be run from plain Python.

Renishaw WDF was checked (2026-10) against 36 WiRE text exports of two inVia
map files, and against Renishaw's own reader (renishaw-wdf 1.4.0) on those two
files and 14 public sample files. Renishaw WXD was checked against the WDF
files of the same two measurements: identical once rounded to 32-bit floats.
"""
import os
import re
import struct
import zlib

import numpy as np
import pandas as pd


# --- Errors ------------------------------------------------------------------
class InstrumentFileError(Exception):
    """A reason an instrument file could not be loaded, worded for the user."""

    def __init__(self, message, suggestions=None):
        super().__init__(message)
        self.suggestions = list(suggestions or [])


class UnsupportedFileError(InstrumentFileError):
    """The file is intact but is not something this module can load."""


class CorruptFileError(InstrumentFileError):
    """The file is damaged, cut short, or not what its signature claims."""


class AxisMismatchError(InstrumentFileError):
    """The files of one class cannot be put on a common x-axis."""


# --- Limits ------------------------------------------------------------------
MAX_CELLS = 20_000_000          # spectra x points accepted in one upload
MAX_BLOCKS = 4096               # blocks walked in one WDF file
MAX_STREAMS = 100_000           # rows accepted in one table of a WXD file
MAX_STREAM_BYTES = 64 << 20     # size accepted for one unpacked WXD stream
AXIS_TOLERANCE = 1e-3           # cm-1; axes closer than this are the same axis
RESERVED_LABELS = ("ramanshift", "average", "standard deviation", "all")

DAMAGED_HINTS = [
    "Check that the file opens in the instrument software.",
    "Copy the file from the instrument again; the transfer may have been cut short.",
]
TEXT_EXPORT_HINT = "Export the spectra as text from the instrument software and use a text format instead."
RAMAN_SHIFT_HINT = "SpectraGuru works on Raman-shift axes. Convert the data to Raman shift in the instrument software first."


# --- Shared helpers ----------------------------------------------------------
def _stem(name):
    """File name without its folder or final extension."""
    stem = os.path.splitext(os.path.basename(name))[0].strip()
    # Clustering drops any column whose name contains "Unnamed".
    return stem.replace("Unnamed", "unnamed") or "spectrum"


def _ascending(name, x, Y):
    """Return the axis low-to-high, with the spectra reordered to match."""
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise CorruptFileError(f"{name} does not contain a usable x-axis.", DAMAGED_HINTS)
    step = np.diff(x)
    if np.all(step < 0):
        x, Y = x[::-1], Y[:, ::-1]
    elif not np.all(step > 0):
        raise UnsupportedFileError(
            f"{name} has an x-axis that is not in increasing or decreasing order.",
            [TEXT_EXPORT_HINT],
        )
    if not np.all(np.isfinite(Y)):
        raise UnsupportedFileError(
            f"{name} contains intensity values that are not numbers.",
            [TEXT_EXPORT_HINT],
        )
    return np.ascontiguousarray(x), np.ascontiguousarray(Y)


def _too_large(name, count, points):
    return UnsupportedFileError(
        f"{name} is too large to load: {count} spectra of {points} points.",
        ["Split the measurement or export a subset from the instrument software."],
    )


def _unfinished_note(name, count, planned):
    return ("info", (
        f"{name} is an unfinished measurement: {count} of {planned} "
        f"planned spectra were collected, and those were loaded."
    ))


def _spectrum_labels(stem, indices):
    """Column names for the spectra of one file."""
    if len(indices) == 1:
        return [stem]
    return [f"{stem}_{i}" for i in indices]


def _unique_labels(labels):
    """Make labels unique and keep them off the names the app reserves."""
    out, used, changed = [], set(), 0
    for label in labels:
        new, k = label, 2
        while new in used or new.lower() in RESERVED_LABELS:
            new = f"{label} ({k})"
            k += 1
        if new != label:
            changed += 1
        used.add(new)
        out.append(new)
    return out, changed


def align_axes(names, axes, blocks):
    """Put the spectra of several files on one ascending axis."""
    first = axes[0]
    same = all(
        x.size == first.size and np.max(np.abs(x - first)) <= AXIS_TOLERANCE
        for x in axes[1:]
    )
    if same:
        return first, blocks, []

    # Same grid rule the upload page applies across classes.
    lo = int(np.ceil(max(x[0] for x in axes)))
    hi = int(np.floor(min(x[-1] for x in axes)))
    if lo >= hi:
        raise AxisMismatchError(
            "The selected files have no overlapping Raman-shift range.",
            [f"{n}: {x[0]:.1f} to {x[-1]:.1f} cm⁻¹" for n, x in zip(names, axes)],
        )
    grid = np.arange(lo, hi + 1, dtype=np.float64)
    aligned = [
        np.vstack([np.interp(grid, x, row) for row in Y])
        for x, Y in zip(axes, blocks)
    ]

    n_spectra = sum(Y.shape[0] for Y in blocks)
    widest = max(x[-1] - x[0] for x in axes)
    kept = 100 * (hi - lo) / widest
    if kept < 50:
        note = ("warning", (
            f"The {len(axes)} files in this class cover different Raman-shift ranges. "
            f"Only the shared range, {lo} to {hi} cm⁻¹ ({grid.size} points, "
            f"{kept:.0f}% of the widest file), was kept; all {n_spectra} spectra were "
            f"linearly interpolated onto a 1 cm⁻¹ grid."
        ))
    else:
        note = ("info", (
            f"The {len(axes)} files in this class do not share one Raman-shift axis, "
            f"so all {n_spectra} spectra were linearly interpolated onto a common "
            f"1 cm⁻¹ grid over the range every file covers ({lo} to {hi} cm⁻¹, "
            f"{grid.size} points). Data outside this range was dropped."
        ))
    return grid, aligned, [note]


def _to_dataframe(x, blocks, labels):
    """Build the wide frame the upload page works on."""
    n_spectra = sum(Y.shape[0] for Y in blocks)
    table = np.empty((x.size, n_spectra + 1), dtype=np.float64)
    table[:, 0] = x
    col = 1
    for Y in blocks:
        table[:, col:col + Y.shape[0]] = Y.T
        col += Y.shape[0]
    return pd.DataFrame(table, columns=["RamanShift"] + labels)


# --- Renishaw WDF ------------------------------------------------------------
# Block layout as published in Renishaw's own reader (renishaw-wdf, Apache-2.0).
WDF_HEADER_SIZE = 512
WDF_FLAG_XYXY = 1 << 0          # several x-list and data blocks
WDF_FLAG_MULTITRACK = 1 << 3    # a separate x-list for each spectrum
WDF_UNIT_RAMAN_SHIFT = 1
WDF_UNIT_NAMES = {0: "arbitrary units", 2: "nanometres", 3: "nanometres", 4: "electron volts", 5: "micrometres"}


def _wdf_sniff(data):
    return data[:4] == b"WDF1"


def _wdf_layout(name, data):
    """Check a WDF file's structure and return where its spectra are."""
    if len(data) < WDF_HEADER_SIZE:
        raise CorruptFileError(f"{name} is too short to be a WDF file.", DAMAGED_HINTS)
    flags = struct.unpack_from("<Q", data, 16)[0]
    points = struct.unpack_from("<I", data, 60)[0]
    capacity, count = struct.unpack_from("<QQ", data, 64)
    y_length, x_length = struct.unpack_from("<II", data, 84)

    blocks, pos, complete = {}, 0, False
    for _ in range(MAX_BLOCKS):
        if pos == len(data):
            complete = True
            break
        if pos + 16 > len(data):
            break
        block_id, _, size = struct.unpack_from("<4sIQ", data, pos)
        if size < 16 or pos + size > len(data):
            break
        blocks.setdefault(block_id, (pos + 16, size - 16))
        pos += size
    if b"DATA" not in blocks or b"XLST" not in blocks:
        raise CorruptFileError(f"{name} ends before its spectra; the file is incomplete.", DAMAGED_HINTS)

    if y_length > 1:
        raise UnsupportedFileError(
            f"{name} stores detector images rather than single spectra, which cannot be loaded.",
            [TEXT_EXPORT_HINT],
        )
    if flags & (WDF_FLAG_XYXY | WDF_FLAG_MULTITRACK):
        raise UnsupportedFileError(
            f"{name} is a multi-track file with more than one x-axis, which cannot be loaded.",
            [TEXT_EXPORT_HINT],
        )
    if count == 0:
        raise UnsupportedFileError(f"{name} contains no collected spectra.")
    if count > capacity or points < 2 or x_length != points:
        raise CorruptFileError(f"{name} has an inconsistent header.", DAMAGED_HINTS)
    if count * points > MAX_CELLS:
        raise _too_large(name, count, points)

    x_start, x_size = blocks[b"XLST"]
    data_start, data_size = blocks[b"DATA"]
    if x_size < 8 + 4 * points or data_size < 4 * count * points:
        raise CorruptFileError(f"{name} is missing part of its spectra.", DAMAGED_HINTS)
    unit = struct.unpack_from("<I", data, x_start + 4)[0]
    if unit != WDF_UNIT_RAMAN_SHIFT:
        raise UnsupportedFileError(
            f"{name} has its x-axis in {WDF_UNIT_NAMES.get(unit, 'an unknown unit')}, not Raman shift.",
            [RAMAN_SHIFT_HINT],
        )
    return {
        "points": points, "count": count, "capacity": capacity, "complete": complete,
        "x_offset": x_start + 8, "data_offset": data_start,
    }


def _wdf_count(name, data):
    return _wdf_layout(name, data)["count"]


def _wdf_read(name, data):
    """Return (x, Y, indices, notes) for one WDF file, in file order."""
    lay = _wdf_layout(name, data)
    points, count = lay["points"], lay["count"]
    x = np.frombuffer(data, dtype="<f4", count=points, offset=lay["x_offset"])
    Y = np.frombuffer(data, dtype="<f4", count=count * points, offset=lay["data_offset"])
    with np.errstate(invalid="ignore"):     # damaged values are refused later, not warned about
        x = x.astype(np.float64)
        Y = Y.astype(np.float64).reshape(count, points)
    notes = []
    if count < lay["capacity"]:
        notes.append(_unfinished_note(name, count, lay["capacity"]))
    if not lay["complete"]:
        notes.append(("warning", f"{name} is damaged after its spectra; the spectra themselves were read in full."))
    return x, Y, list(range(count)), notes


# --- Metakit container -------------------------------------------------------
# WXD files keep their streams in a Metakit datafile (Metakit is MIT-licensed,
# Copyright (c) 1996-2007 Jean-Claude Wippler). Only the one storage layout
# WiRE writes is read; MK_SMALL_WIDTHS is Metakit's table for short columns.
MK_SCHEMA = (
    b"dirs[name:S,parent:I,ctime:I,atime:I,mtime:I,clsid:S,state:I,"
    b"files[_B[name:S,size:I,date:I,state:I,contents:B]]]"
)
MK_SMALL_WIDTHS = (
    (8, 16, 1, 32, 2, 4),
    (4, 8, 1, 16, 2, 0),
    (2, 4, 8, 1, 0, 16),
    (2, 4, 0, 8, 1, 0),
    (1, 2, 4, 0, 8, 0),
    (1, 2, 4, 0, 0, 8),
    (1, 2, 0, 4, 0, 0),
)
MK_INT_CODES = {8: "b", 16: "h", 32: "i", 64: "q"}


def _mk_take(mk, start, size):
    """Bytes [start, start + size) of the datafile, or fail if out of range."""
    if start < 0 or size < 0 or start + size > len(mk):
        raise ValueError("reference outside the file")
    return mk[start:start + size]


def _mk_varint(buf, pos):
    """Read one variable-length number; returns (value, next position)."""
    if buf[pos] == 0:
        raise ValueError("negative number")
    value = 0
    for _ in range(6):
        byte = buf[pos]
        pos += 1
        value = (value << 7) + byte
        if byte & 0x80:
            return value - 0x80, pos
    raise ValueError("number too long")


def _mk_colref(buf, pos):
    """Read a column reference; returns ((start, size), next position)."""
    size, pos = _mk_varint(buf, pos)
    start = 0
    if size > 0:
        start, pos = _mk_varint(buf, pos)
    return (start, size), pos


def _mk_rows(buf, pos):
    """Read the start of a table: its reserved zero, then its row count."""
    reserved, pos = _mk_varint(buf, pos)
    rows, pos = _mk_varint(buf, pos)
    if reserved != 0 or rows > MAX_STREAMS:
        raise ValueError("bad table header")
    return rows, pos


def _mk_ints(mk, ref, rows):
    """Decode a column of adaptive-width integers."""
    start, size = ref
    if size == 0 or rows == 0:
        return [0] * rows
    width = (size << 3) // rows
    if rows <= 7 and size <= 6:
        width = MK_SMALL_WIDTHS[rows - 1][size - 1]
    raw = _mk_take(mk, start, size)
    if width in (1, 2, 4):
        per_byte, mask = 8 // width, (1 << width) - 1
        return [(raw[i // per_byte] >> ((i % per_byte) * width)) & mask for i in range(rows)]
    if width in MK_INT_CODES:
        return list(struct.unpack_from(f"<{rows}{MK_INT_CODES[width]}", raw))
    raise ValueError("bad integer width")


def _mk_items(mk, buf, pos, rows):
    """Decode a string or bytes column; returns (items, next position)."""
    data_ref, pos = _mk_colref(buf, pos)
    sizes_ref = (0, 0)
    if data_ref[1] > 0:
        sizes_ref, pos = _mk_colref(buf, pos)
    memo_ref, pos = _mk_colref(buf, pos)

    sizes = _mk_ints(mk, sizes_ref, rows)
    data = _mk_take(mk, *data_ref)
    if any(n < 0 for n in sizes) or sum(sizes) != len(data):
        raise ValueError("item sizes do not match the column")
    items, offset = [], 0
    for n in sizes:
        items.append(data[offset:offset + n])
        offset += n

    # Large items are stored on their own and listed here by row.
    memo = _mk_take(mk, *memo_ref)
    at, row = 0, 0
    while at < len(memo):
        skip, at = _mk_varint(memo, at)
        row += skip
        ref, at = _mk_colref(memo, at)
        items[row] = _mk_take(mk, *ref)
        row += 1
    return items, pos


def _mk_directory(name, data):
    """Streams of the storage's root folder: {name: (size, stored bytes)}."""
    skip_mark = data[-16:-12]
    if len(data) < 24 or data[-8] != 0x80 or not (skip_mark == b"\x80\0\0\0" or skip_mark[0] & 0xF0 == 0x90):
        raise ValueError("no datafile tail")
    base = len(data) - 16 - int.from_bytes(data[-12:-8], "big")
    if base < 0 or data[base:base + 3] != b"JL\x1a":
        raise ValueError("no datafile header")
    mk = memoryview(data)[base:]
    walk = _mk_take(mk, int.from_bytes(data[-4:], "big"), int.from_bytes(data[-7:-4], "big"))

    reserved, at = _mk_varint(walk, 0)
    length, at = _mk_varint(walk, at)
    if reserved != 0 or bytes(walk[at:at + length]) != MK_SCHEMA:
        raise UnsupportedFileError(
            f"{name} uses a WXD layout that is not supported yet.",
            ["Save the measurement as .wdf in WiRE, or export the spectra as text."],
        )
    rows, at = _mk_varint(walk, at + length)
    if rows != 1:
        raise ValueError("bad root table")
    dirs_ref, at = _mk_colref(walk, at)

    # dirs: name, parent, ctime, atime, mtime, clsid, state, files
    dirs = _mk_take(mk, *dirs_ref)
    n_dirs, at = _mk_rows(dirs, 0)
    _, at = _mk_items(mk, dirs, at, n_dirs)
    parent_ref, at = _mk_colref(dirs, at)
    for _ in range(3):
        _, at = _mk_colref(dirs, at)
    _, at = _mk_items(mk, dirs, at, n_dirs)
    _, at = _mk_colref(dirs, at)
    files_ref, at = _mk_colref(dirs, at)
    root = _mk_ints(mk, parent_ref, n_dirs).index(-1)

    # files is one table per folder, split into blocks; every block row,
    # including those of the trailing separator block, is a real file.
    files = _mk_take(mk, *files_ref)
    streams, at = {}, 0
    for folder in range(n_dirs):
        n_blocks, at = _mk_rows(files, at)
        if n_blocks == 0:
            continue
        blocks_ref, at = _mk_colref(files, at)
        if folder != root:
            continue
        blocks, pos = _mk_take(mk, *blocks_ref), 0
        for _ in range(n_blocks):
            n_files, pos = _mk_rows(blocks, pos)
            if n_files == 0:
                continue
            names, pos = _mk_items(mk, blocks, pos, n_files)
            sizes_ref, pos = _mk_colref(blocks, pos)
            for _ in range(2):
                _, pos = _mk_colref(blocks, pos)
            contents, pos = _mk_items(mk, blocks, pos, n_files)
            sizes = _mk_ints(mk, sizes_ref, n_files)
            for stream, size, stored in zip(names, sizes, contents):
                streams[bytes(stream).rstrip(b"\0").decode("utf-8")] = (size, stored)
    return streams


def _mk_inflate(stored, size, limit):
    """A stream's bytes; contents are zlib-packed when shorter than size."""
    if size < 0 or size > limit:
        raise ValueError("stream too large")
    if len(stored) == size:
        return bytes(stored)
    engine = zlib.decompressobj()
    out = engine.decompress(stored, size + 1)
    if len(out) != size or not engine.eof:
        raise ValueError("stream does not unpack to its stated size")
    return out


# --- Renishaw WXD ------------------------------------------------------------
# No description of this format is published. The layout was worked out from
# files saved by WiRE and checked against WDF files of the same measurements.
# A stream holds named entries (uint32 name size, UTF-16 name, COM VARIANT
# type, value); a list holds a count, tagged numbers and an end mark. Axis
# types and units use the same codes as WDF.
WXD_LIST_END = b"\xc0\xff\xee\x01"
WXD_LIST_NAME = "DataList".encode("utf-16-le")
WXD_NUMBER_TYPES = {2: "<i2", 3: "<i4", 4: "<f4", 5: "<f8"}    # VARIANT type -> dtype
WXD_INT = b"\x03\x00"          # VARIANT type 3, a 32-bit integer
WXD_OBJECT = b"\x0d\x00"       # VARIANT type 13, then a 16-byte class id
WXD_TYPE_SPECTRAL = (1, 19)


def _wxd_sniff(data):
    return data[:4] == b"#WXD"


def _wxd_stream(streams, key, limit=MAX_STREAM_BYTES):
    size, stored = streams[key]
    return _mk_inflate(stored, size, limit)


def _wxd_int(stream, key):
    """Value of a named 32-bit integer entry, or None if the stream lacks it."""
    raw = key.encode("utf-16-le") + b"\0\0"
    at = stream.find(raw)
    while at >= 0:
        end = at + len(raw)
        if at >= 4 and struct.unpack_from("<I", stream, at - 4)[0] == len(raw) and stream[end:end + 2] == WXD_INT:
            return struct.unpack_from("<i", stream, end + 2)[0]
        at = stream.find(raw, at + 2)
    return None


def _wxd_list(stream, at):
    """Decode the list of numbers that starts at offset at."""
    count = struct.unpack_from("<Q", stream, at)[0]
    if not 0 < count <= MAX_CELLS:
        raise ValueError("bad list length")
    number = WXD_NUMBER_TYPES[struct.unpack_from("<H", stream, at + 8)[0]]
    records = np.frombuffer(stream, dtype=[("type", "<u2"), ("value", number)], count=count, offset=at + 8)
    end = at + 8 + records.nbytes
    if np.any(records["type"] != records["type"][0]) or stream[end:end + 4] != WXD_LIST_END:
        raise ValueError("bad list")
    return records["value"].astype(np.float64)


def _wxd_spectrum(stream):
    """The intensity list embedded in one DataSet stream, found by its name."""
    at = stream.find(WXD_LIST_NAME)
    while at >= 0:
        size = struct.unpack_from("<I", stream, at - 4)[0] if at >= 4 else 0
        if 0 < size <= 64:
            entry = stream[at:at + size].decode("utf-16-le", "replace")
            if re.fullmatch(r"DataList\d+\0", entry) and stream[at + size:at + size + 2] == WXD_OBJECT:
                return _wxd_list(stream, at + size + 18)
        at = stream.find(WXD_LIST_NAME, at + 2)
    raise ValueError("no spectrum in dataset")


def _wxd_layout(name, data):
    """Check a WXD file's structure and return its streams and spectrum order."""
    streams = _mk_directory(name, data)
    datasets = sorted(int(n[7:]) for n in streams if re.fullmatch(r"DataSet\d+", n))
    if not datasets:
        raise UnsupportedFileError(f"{name} contains no collected spectra.")
    properties = _wxd_stream(streams, "Properties")

    axes = []
    for stream_name in streams:
        match = re.fullmatch(r"DataList(\d+)", stream_name)
        if match and _wxd_int(properties, "dataListType" + match.group(1)) in WXD_TYPE_SPECTRAL:
            axes.append((stream_name, _wxd_int(properties, "dataListUnits" + match.group(1))))
    if len(axes) != 1:
        raise UnsupportedFileError(
            f"{name} does not describe a single spectral axis, so it cannot be loaded.",
            [TEXT_EXPORT_HINT],
        )
    x_stream, unit = axes[0]
    if unit != WDF_UNIT_RAMAN_SHIFT:
        raise UnsupportedFileError(
            f"{name} has its x-axis in {WDF_UNIT_NAMES.get(unit, 'an unknown unit')}, not Raman shift.",
            [RAMAN_SHIFT_HINT],
        )

    collected = _wxd_int(properties, "nCollectedDatasets")
    if collected is None or not 0 < collected <= len(datasets):
        collected = len(datasets)
    return {"streams": streams, "x_stream": x_stream, "datasets": datasets[:collected], "planned": len(datasets)}


def _wxd_count(name, data):
    return len(_wxd_layout(name, data)["datasets"])


def _wxd_read(name, data):
    """Return (x, Y, indices, notes) for one WXD file, in file order."""
    lay = _wxd_layout(name, data)
    streams, datasets = lay["streams"], lay["datasets"]
    x = _wxd_list(_wxd_stream(streams, lay["x_stream"]), 0)
    if len(datasets) * x.size > MAX_CELLS:
        raise _too_large(name, len(datasets), x.size)
    Y = np.empty((len(datasets), x.size), dtype=np.float64)
    limit = 12 * x.size + 65536     # ten bytes a point, plus the dataset's few properties
    for row, index in enumerate(datasets):
        values = _wxd_spectrum(_wxd_stream(streams, f"DataSet{index}", limit))
        if values.size != x.size:
            raise ValueError("spectrum length differs from the axis")
        Y[row] = values
    notes = []
    if len(datasets) < lay["planned"]:
        notes.append(_unfinished_note(name, len(datasets), lay["planned"]))
    return x, Y, datasets, notes


# --- Registry and public API -------------------------------------------------
READERS = {
    "renishaw": {
        "vendor": "Renishaw WiRE",
        "label": "Renishaw WiRE (.wdf / .wxd)",
        "types": ["wdf", "wxd"],
        "formats": [
            {"name": "WDF", "ext": "wdf", "sniff": _wdf_sniff, "count": _wdf_count, "read": _wdf_read},
            {"name": "WXD", "ext": "wxd", "sniff": _wxd_sniff, "count": _wxd_count, "read": _wxd_read},
        ],
    },
}

LOW_LEVEL_ERRORS = (struct.error, zlib.error, IndexError, KeyError, ValueError, OverflowError)


def _format_for(kind, name, data):
    """Pick the format whose signature matches the file's content."""
    reader = READERS[kind]
    for fmt in reader["formats"]:
        if fmt["sniff"](data):
            return fmt
    head = data[:512]
    if head and all(32 <= b < 127 or b in (9, 10, 13) for b in head):
        raise UnsupportedFileError(
            f"{name} is a text file, not a {reader['vendor']} file.",
            ["Choose a text format such as 'Multi TXT (two-column, common x-axis)' for this file."],
        )
    raise UnsupportedFileError(
        f"{name} is not a recognised {reader['vendor']} file.",
        ["Check that the file opens in the instrument software.", TEXT_EXPORT_HINT],
    )


def _guard(name, action):
    """Run a reader step, reporting low-level failures as a damaged file."""
    try:
        return action()
    except InstrumentFileError:
        raise
    except LOW_LEVEL_ERRORS as e:
        raise CorruptFileError(
            f"{name} could not be read; the file appears to be damaged or incomplete.",
            DAMAGED_HINTS,
        ) from e


def _read_one(kind, name, data):
    def action():
        fmt = _format_for(kind, name, data)
        x, Y, indices, notes = fmt["read"](name, data)
        x, Y = _ascending(name, x, Y)
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        if ext != fmt["ext"] and ext in READERS[kind]["types"]:
            notes.insert(0, ("info", f"{name} is a {fmt['name']} file by content and was read as one."))
        return x, Y, indices, notes
    return _guard(name, action)


def count_spectra(kind, name, data):
    """Number of spectra in one file, from its structure alone."""
    return _guard(name, lambda: _format_for(kind, name, data)["count"](name, data))


def read_spectra(kind, name, data):
    """Read one file: ascending x, spectra as rows, and notices for the user."""
    x, Y, _, notes = _read_one(kind, name, data)
    return x, Y, notes


def read_files(kind, files):
    """Read the files of one class into a wide DataFrame plus user notices."""
    names, axes, blocks, labels, notes, seen = [], [], [], [], [], {}
    for name, data in files:
        x, Y, indices, file_notes = _read_one(kind, name, data)
        stem = _stem(name)
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] > 1:
            stem = f"{stem} ({seen[stem]})"
        names.append(name)
        axes.append(x)
        blocks.append(Y)
        labels.extend(_spectrum_labels(stem, indices))
        notes.extend(file_notes)
    if not axes:
        raise UnsupportedFileError("No file was provided.")
    if sum(Y.size for Y in blocks) > MAX_CELLS:
        raise UnsupportedFileError(
            "The selected files are too large to load together.",
            ["Upload fewer files at once."],
        )

    labels, renamed = _unique_labels(labels)
    x, blocks, align_notes = align_axes(names, axes, blocks)
    notes.extend(align_notes)
    if renamed:
        notes.append(("info", f"{renamed} spectrum name(s) were changed so that every column name is unique."))
    return _to_dataframe(x, blocks, labels), notes
