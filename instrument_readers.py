"""Readers for instrument-vendor spectrum files.

Each reader turns one vendor file into a Raman-shift axis and a block of
spectra. Only the axis and the intensities are kept; map positions, times,
images and other metadata are ignored. Streamlit is not imported here, so the
module can be run from plain Python.

Renishaw WDF was checked (2026-10) against 36 WiRE text exports of two inVia
map files, and against Renishaw's own reader (renishaw-wdf 1.4.0) on those two
files and 14 public sample files.
"""
import os
import struct

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
MAX_CELLS = 20_000_000      # spectra x points accepted in one upload
MAX_BLOCKS = 4096           # blocks walked in one WDF file
AXIS_TOLERANCE = 1e-3       # cm-1; axes closer than this are the same axis
RESERVED_LABELS = ("ramanshift", "average", "standard deviation", "all")

DAMAGED_HINTS = [
    "Check that the file opens in the instrument software.",
    "Copy the file from the instrument again; the transfer may have been cut short.",
]
TEXT_EXPORT_HINT = "Export the spectra as text from the instrument software and use a text format instead."


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
        raise UnsupportedFileError(
            f"{name} is too large to load: {count} spectra of {points} points.",
            ["Split the measurement or export a subset from the instrument software."],
        )

    x_start, x_size = blocks[b"XLST"]
    data_start, data_size = blocks[b"DATA"]
    if x_size < 8 + 4 * points or data_size < 4 * count * points:
        raise CorruptFileError(f"{name} is missing part of its spectra.", DAMAGED_HINTS)
    unit = struct.unpack_from("<I", data, x_start + 4)[0]
    if unit != WDF_UNIT_RAMAN_SHIFT:
        raise UnsupportedFileError(
            f"{name} has its x-axis in {WDF_UNIT_NAMES.get(unit, 'an unknown unit')}, not Raman shift.",
            ["SpectraGuru works on Raman-shift axes. Convert the data to Raman shift in the instrument software first."],
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
        notes.append(("info", (
            f"{name} is an unfinished measurement: {count} of {lay['capacity']} "
            f"planned spectra were collected, and those were loaded."
        )))
    if not lay["complete"]:
        notes.append(("warning", f"{name} is damaged after its spectra; the spectra themselves were read in full."))
    return x, Y, list(range(count)), notes


# --- Registry and public API -------------------------------------------------
READERS = {
    "renishaw": {
        "vendor": "Renishaw WiRE",
        "label": "Renishaw WiRE (.wdf)",
        "types": ["wdf"],
        "formats": [
            {"name": "WDF", "ext": "wdf", "sniff": _wdf_sniff, "count": _wdf_count, "read": _wdf_read},
        ],
    },
}

LOW_LEVEL_ERRORS = (struct.error, IndexError, KeyError, ValueError, OverflowError)


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
