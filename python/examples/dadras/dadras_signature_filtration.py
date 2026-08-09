# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature filtration heatmap (Dadras)
========================================

The cycling signature of a sliding window at every adjacency threshold up
to the top of the stored detection band. Each column fixes one window of
the Dadras trajectory; climbing the column replays that window's
filtration: white below the window's first generator birth, then the span
of every generator born by the threshold. Colors name the frequent
signatures (shared with the signature indicator); gray marks non-trivial
signatures outside that library. A column whose color locks in well below
the band top carries a signature that is stable across the detection band;
gray growing toward the band top shows rare signatures that only appear at
the loosest recurrence scales.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. The storage's
# point indices are positions in the detection trajectory, and that
# trajectory's ``parameters()`` give the integration time of each detection
# point,
# which places every column on the time axis below. ``threshold()`` is the top
# of the stored detection band and bounds every filtration query below.

import math
from bisect import bisect_left
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.dadras_trajectory())
STORAGE = cs.CycleStorage.load(_support.dadras_storage())
PARAMETERS = TRAJECTORY.parameters()
BAND_TOP = STORAGE.threshold()
assert math.isfinite(BAND_TOP)

# %%
# **Order the classes by frequency and assign canonical colors.** Classes are
# ordered by how often they recur (their total cycle count across components),
# the same ordering the other Dadras examples use, so "class 1" names the
# same class throughout. A rank-1 signature spanning a frequent class takes
# that class's color and label; the class vectors themselves are typically
# wide for this system and are not printed.

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
# **Query one filtration per column.** Each column start yields a single
# ``signature`` query; the window's span can change only at its generators'
# births, so recording the span at each distinct birth captures the whole
# column exactly.

WINDOW_LENGTH = 240
POINT_WINDOW_START = 0
POINT_WINDOW_STOP = 8000
COLUMN_STEP = 10
ROW_COUNT = 60

extent_start, extent_stop = STORAGE.extent()
column_starts = np.arange(
    max(POINT_WINDOW_START, extent_start),
    min(POINT_WINDOW_STOP, extent_stop - WINDOW_LENGTH + 1),
    COLUMN_STEP,
)
row_thresholds = [BAND_TOP * (row + 0.5) / ROW_COUNT for row in range(ROW_COUNT)]

column_filtrations: list[tuple[list[float], list[cs.Subspace]]] = []
for window_start in column_starts:
    signature = STORAGE.signature(range(int(window_start), int(window_start) + WINDOW_LENGTH))
    births = sorted(set(signature.births()))
    column_filtrations.append((births, [signature.span_at(birth) for birth in births]))

# %%
# **Place the columns in time.** Columns step a fixed number of detection
# points, which is not a fixed amount of time: the points follow the
# trajectory's geometry, so one spans more time where the flow runs slowly.
# Each column runs from its window's start time to the next column's start
# time, tiling the axis with no gaps; the column marks where its window begins,
# not the time its window covers.

column_edges = np.append(
    PARAMETERS[column_starts],
    PARAMETERS[min(int(column_starts[-1]) + COLUMN_STEP, len(PARAMETERS) - 1)],
)
window_starts = np.arange(extent_start, extent_stop - WINDOW_LENGTH + 1)
MEDIAN_WINDOW_DURATION = float(
    np.median(PARAMETERS[window_starts + WINDOW_LENGTH - 1] - PARAMETERS[window_starts])
)

# %%
# **Build a signature library from the band-top spans.** The most frequent
# full-band signatures get the named colors; anything rarer renders gray, so
# unstable or uncommon structure stays visible without a name. The library
# is ordered by rank then descending frequency.

LIBRARY_SIZE = 6

frequency: Counter[cs.Subspace] = Counter(
    spans[-1] for births, spans in column_filtrations if births
)
most_common = frequency.most_common(LIBRARY_SIZE)
ordered = sorted(most_common, key=lambda item: (item[0].rank(), -item[1]))
library = [span for span, _ in ordered]

# %%
# **Assign colors and labels.** A rank-1 signature spanning a frequent class
# takes that class's color and frequency label; every other library entry is
# named by its position and rank, and takes a palette color no frequent-class
# entry uses.


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
# **Build the label array.** Cell values: 0 for the trivial signature
# (white), 1 through ``len(library)`` for library signatures, and one extra
# label (gray) for any non-trivial signature outside the library.

library_index = {span: index for index, span in enumerate(library, start=1)}
OTHER_LABEL = len(library) + 1

labels = np.zeros((ROW_COUNT, len(column_starts)), dtype=np.int8)
for column, (births, spans) in enumerate(column_filtrations):
    for row, threshold in enumerate(row_thresholds):
        born = bisect_left(births, threshold) - 1
        if born < 0:
            continue
        labels[row, column] = library_index.get(spans[born], OTHER_LABEL)

# %%
# **Render the heatmap.** ``pcolormesh`` draws each row at the y edges handed
# to it, in the order given; those edges run from 0 to ``BAND_TOP`` ascending,
# so threshold zero lands at the bottom of the axes and each column reads
# upward as its window's filtration, with the top edge of the image at the
# band top.


def build_figure() -> plt.Figure:
    """Return the signature filtration heatmap figure."""
    library_colors, library_labels = library_colors_and_labels(library)
    colors = [(1.0, 1.0, 1.0), *library_colors, (0.75, 0.75, 0.75)]
    tick_labels = ["trivial", *library_labels, "other non-trivial"]
    colormap = ListedColormap(colors)

    figure, axes = plt.subplots(figsize=(14, 6))
    image = axes.pcolormesh(
        column_edges,
        np.linspace(0.0, BAND_TOP, ROW_COUNT + 1),
        labels,
        cmap=colormap,
        vmin=-0.5,
        vmax=OTHER_LABEL + 0.5,
    )

    axes.set_xlabel("Time")
    axes.set_ylabel("Adjacency threshold t")
    axes.set_title(
        "Signature filtration: Dadras attractor, window length "
        f"{WINDOW_LENGTH} points ({MEDIAN_WINDOW_DURATION:.2f} time)"
    )

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_ticks(range(OTHER_LABEL + 1))
    colorbar.set_ticklabels(tick_labels)

    figure.tight_layout()
    return figure


figure = build_figure()
