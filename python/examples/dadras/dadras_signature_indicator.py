# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature indicator heatmap (Dadras)
=======================================

The cycling signature present at each time, shown for several window
lengths. Each (window length, time) cell is colored by its signature: white
for the trivial signature (rank 0); a rank-1 signature is the span of a single
frequent homology class and takes that class's color (the same colors the
coverage barcode uses); higher-rank signatures, spanning several independent
loop types at once, get their own colors.

Only the most frequent signatures are distinguished. The Dadras storage keeps
many rare signatures, each confined to a small part of the attractor, alongside
a frequent few; cells whose signature falls outside the frequent library render
white like the trivial ones.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. The storage's
# point indices are positions in the detection trajectory, and that
# trajectory's ``parameters()`` give the integration time of each detection
# point: they place every window on the time axis below and turn a window
# length in detection points into a duration.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.dadras_trajectory())
STORAGE = cs.CycleStorage.load(_support.dadras_storage())
PARAMETERS = TRAJECTORY.parameters()

# %%
# **Rank the classes by frequency and assign canonical colors.** Classes are
# ranked by how often they recur (their total cycle count across components),
# the same ordering the coverage barcode uses, so "class 1" names the same
# class in both plots and takes the same color via ``class_color_map``. A
# rank-1 signature is the span of a single homology class; when that class is
# among the frequent ones it takes the class's color and label.

TOP_CLASSES = 5

class_objects = STORAGE.classes()
class_keys = [
    tuple(int(value) for value in homology_class.to_array()) for homology_class in class_objects
]
nonzero_classes = [
    (key, homology_class)
    for key, homology_class in zip(class_keys, class_objects, strict=True)
    if any(key)
]

class_cycle_counts: Counter[int] = Counter()
for component in STORAGE.components():
    if any(class_keys[component.class_id()]):
        class_cycle_counts[component.class_id()] += component.cycle_count()

ordered_class_ids = [class_id for class_id, _ in class_cycle_counts.most_common(TOP_CLASSES)]
CLASS_COLORS = _support.class_color_map([class_keys[class_id] for class_id in ordered_class_ids])
CLASS_POSITIONS = {
    class_keys[class_id]: position for position, class_id in enumerate(ordered_class_ids, start=1)
}

# %%
# **Build a signature library.** Slide a window of a single representative
# length across the extent and tally the distinct non-trivial signatures.
#
# The library keeps the most frequent signatures, then orders them by rank and
# by descending frequency within a rank, so the legend lists the rank-1
# signatures first and the higher ranks after.

LIBRARY_LENGTH = 240
LIBRARY_STEP = 200
LIBRARY_SIZE = 6

extent_start, extent_stop = STORAGE.extent()

frequency: Counter[cs.Subspace] = Counter()
for window_start in range(extent_start, extent_stop - LIBRARY_LENGTH + 1, LIBRARY_STEP):
    subspace = STORAGE.signature(range(window_start, window_start + LIBRARY_LENGTH)).span()
    if subspace.rank() != 0:
        frequency[subspace] += 1

most_common = frequency.most_common(LIBRARY_SIZE)
ordered = sorted(most_common, key=lambda item: (item[0].rank(), -item[1]))
library = [subspace for subspace, _ in ordered]

# %%
# **Build the label array.** Each cell is an integer label: -1 for a signature
# outside the library (trivial or uncommon) and 0..len(library)-1 for library
# members.

WINDOW_LENGTHS = (180, 240, 360)
SAMPLE_WINDOW_START = 0
SAMPLE_WINDOW_STOP = 8000
COLUMN_STEP = 10

column_starts = np.arange(SAMPLE_WINDOW_START, SAMPLE_WINDOW_STOP, COLUMN_STEP)
num_rows = len(WINDOW_LENGTHS)
num_columns = len(column_starts)

library_index = {subspace: index for index, subspace in enumerate(library)}
labels = np.full((num_rows, num_columns), -1, dtype=np.int8)

for row_index, length in enumerate(WINDOW_LENGTHS):
    for column_index, start in enumerate(column_starts):
        if start + length > extent_stop:
            continue
        subspace = STORAGE.signature(range(int(start), int(start) + length)).span()
        labels[row_index, column_index] = library_index.get(subspace, -1)

# %%
# **Place the columns and rows in time.** Columns step a fixed number of
# detection points, which is not a fixed amount of time: the points follow the
# trajectory's geometry, so one spans more time where the flow runs slowly.
# Each column therefore runs from its window's start time to the next column's
# start time, tiling the axis with no gaps; the column marks where its window
# begins, not the time its window covers.

column_edges = np.append(
    PARAMETERS[column_starts],
    PARAMETERS[min(int(column_starts[-1]) + COLUMN_STEP, len(PARAMETERS) - 1)],
)


def median_window_duration(length: int) -> float:
    """Return the median time a window of ``length`` points spans."""
    starts = np.arange(extent_start, extent_stop - length + 1)
    return float(np.median(PARAMETERS[starts + length - 1] - PARAMETERS[starts]))


# %%
# **Assign colors and labels to the library.** A rank-1 signature spanning a
# frequent class takes that class's color and frequency label; every other
# library signature is named by its library position and rank, and takes a
# palette color no frequent-class entry in this figure uses. The class vectors
# themselves are typically wide for this system and are not printed.


def library_colors_and_labels(
    subspaces: list[cs.Subspace],
) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Return the color and legend label for each non-trivial signature."""
    matched_class_keys: list[tuple[int, ...] | None] = []
    for subspace in subspaces:
        key = None
        if subspace.rank() == 1:
            key = next(
                key for key, homology_class in nonzero_classes if subspace.contains(homology_class)
            )
            if key not in CLASS_COLORS:
                key = None
        matched_class_keys.append(key)

    used = {CLASS_COLORS[key] for key in matched_class_keys if key is not None}
    remaining = [color for color in _support.signature_colors() if color not in used]

    colors: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for position, (subspace, key) in enumerate(zip(subspaces, matched_class_keys, strict=True), 1):
        if key is not None:
            colors.append(CLASS_COLORS[key])
            labels.append(f"class {CLASS_POSITIONS[key]} (rank 1)")
        else:
            colors.append(remaining.pop(0))
            labels.append(f"signature {position} (rank {subspace.rank()})")
    return colors, labels


# %%
# **Render the heatmap.** The colormap puts trivial (white) at index 0 and
# each library signature at indices 1..len(library). Shift ``labels + 1`` so
# -1 (trivial or unlabeled) maps to 0.


def build_figure() -> plt.Figure:
    """Return the signature indicator heatmap figure."""
    library_colors, library_labels = library_colors_and_labels(library)
    colors = [(1.0, 1.0, 1.0), *library_colors]
    tick_labels = ["trivial", *library_labels]
    colormap = ListedColormap(colors)

    figure, axes = plt.subplots(figsize=(14, 5))
    image = axes.pcolormesh(
        column_edges,
        np.arange(num_rows + 1) - 0.5,
        labels + 1,
        cmap=colormap,
        vmin=-0.5,
        vmax=len(library) + 0.5,
    )
    axes.invert_yaxis()

    axes.set_yticks(range(len(WINDOW_LENGTHS)))
    axes.set_yticklabels(
        [
            f"{length} points\n({median_window_duration(length):.2f} time)"
            for length in WINDOW_LENGTHS
        ]
    )
    axes.set_xlabel("Time")
    axes.set_ylabel("Window length")
    axes.set_title("Signature indicator: Dadras attractor")

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_ticks(range(len(library) + 1))
    colorbar.set_ticklabels(tick_labels)

    figure.tight_layout()
    return figure


figure = build_figure()
