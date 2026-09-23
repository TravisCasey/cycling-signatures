# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature indicator (Charney-DeVore)
=======================================

What a single window of the Charney-DeVore trajectory reads, at four window
lengths across the same stretch of time. Each cell is one window, colored by
its span: white where the span is trivial, a class color where the span is that
one class of the dominant span, light gray where it is exactly the dominant
span, and mid gray for everything else, which is any other single class and any
span of rank 2 or more that is not the dominant one.

A row is a window length. Short windows read nothing most of the time and
otherwise read one class at a time; long windows settle on the dominant span
and then start picking up the rare classes outside it. The printed output
counts each reading per row.
"""

# %%
# Load the storage for cube side 0.005 and the detection points it indexes,
# both from the published example data, fetched and cached on first use. The
# detection points contribute only their times, which place every window on
# the time axis and turn a window length in time units into a point count.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
STORAGE = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
TIMES = _support.charney_devore_detection(BOXSIZE).times

# %%
# Take the dominant span, the subspace its classes span, and the class colors
# this section uses. All of them live in this cover's generator basis, which is
# the only basis they mean anything in.

SPAN = _support.dominant_span(STORAGE)
CLASSES = STORAGE.classes()
DOMINANT_SUBSPACE = cs.Subspace([CLASSES[class_id] for class_id in SPAN.class_ids])
CLASS_COLORS = _support.signature_colors()[: len(SPAN.class_ids)]

# %%
# The grid the figure draws: window lengths and window starts, both in the
# model's own time units, over the opening stretch of the storage's extent.
# Times come per detection point, so each length is turned into a point count
# through the extent's mean time per point, and each start into the first
# detection point at or after it.

WINDOW_LENGTHS = (250.0, 500.0, 1000.0, 2000.0)
STRETCH = 20_000.0
COLUMN_STEP = 50.0

extent_start, extent_stop = STORAGE.extent()
elapsed = float(TIMES[extent_stop - 1] - TIMES[extent_start])
time_per_point = elapsed / (extent_stop - 1 - extent_start)

column_starts = np.arange(0.0, STRETCH, COLUMN_STEP)
column_points = np.searchsorted(TIMES - TIMES[extent_start], column_starts)

# %%
# Classify every window. The codes run trivial, then one per class of the
# dominant span in frequency order, then the dominant span itself, then
# everything else, and they index both the colormap and the legend below.

READINGS = ["trivial", *(f"class {position}" for position in range(1, len(SPAN.class_ids) + 1))]
READINGS += ["dominant span", "other"]
DOMINANT_CODE = len(READINGS) - 2
OTHER_CODE = len(READINGS) - 1


def window_reading(start: int, length: int) -> int:
    """Return the code the window of `length` points at `start` reads."""
    subspace = STORAGE.signature(range(start, min(start + length, extent_stop))).span()
    if subspace.rank() == 0:
        return 0
    if subspace.rank() == 1:
        for position, class_id in enumerate(SPAN.class_ids, start=1):
            if subspace.contains(CLASSES[class_id]):
                return position
    elif subspace == DOMINANT_SUBSPACE:
        return DOMINANT_CODE
    return OTHER_CODE


codes = np.empty((len(WINDOW_LENGTHS), len(column_starts)), dtype=np.int64)
for row, window_length in enumerate(WINDOW_LENGTHS):
    length_in_points = round(window_length / time_per_point)
    for column, start in enumerate(column_points):
        codes[row, column] = window_reading(int(start), length_in_points)

# %%
# Report what each row holds, as a count of windows per reading.

for row, window_length in enumerate(WINDOW_LENGTHS):
    tally = np.bincount(codes[row], minlength=len(READINGS))
    counted = ", ".join(
        f"{reading} {count}" for reading, count in zip(READINGS, tally, strict=True)
    )
    print(f"windows of {window_length:.0f} time units: {counted}")

# %%
# Draw the grid. Rows are drawn in the order the window lengths are listed, and
# the time axis carries the window's start rather than the stretch it covers.


def build_figure() -> plt.Figure:
    """Return the signature indicator grid."""
    palette = [(1.0, 1.0, 1.0), *CLASS_COLORS, "0.88", "0.6"]
    figure, axes = plt.subplots(figsize=(15, 3.6))
    axes.imshow(
        codes,
        cmap=ListedColormap(palette),
        vmin=-0.5,
        vmax=len(READINGS) - 0.5,
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, STRETCH, len(WINDOW_LENGTHS) - 0.5, -0.5),
    )
    axes.set_yticks(range(len(WINDOW_LENGTHS)))
    axes.set_yticklabels([f"{length:.0f}" for length in WINDOW_LENGTHS])
    axes.set_ylabel("window length (time units)")
    axes.set_xlabel("window start (time units)")
    axes.legend(
        handles=[
            Patch(facecolor=color, label=reading)
            for color, reading in zip(palette[1:], READINGS[1:], strict=True)
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    return figure


figure = build_figure()

# %%
# The shortest windows are mostly white, with single classes appearing where a
# recurrence happens to fit. As the window grows the single classes give way to
# the dominant span, which at two thousand time units holds most of the row.
# *Window rank against window length* takes that over every length at once, and
# reads the same thing as a share of windows against duration.
