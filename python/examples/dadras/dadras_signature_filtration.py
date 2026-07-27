# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature filtration heatmap
===============================

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
# Load the prebuilt ``CycleStorage`` from the published example data, fetched
# and cached on first use. ``threshold()`` is the top of the stored detection
# band and bounds every filtration query below.

import math
from bisect import bisect_right
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.dadras_storage())
BAND_TOP = STORAGE.threshold()
assert math.isfinite(BAND_TOP)

# %%
# **Rank the classes by frequency and assign canonical colors.** Classes are
# ranked by how often they recur (their total cycle count across components),
# the same ordering the other Dadras examples use, so "class 1" names the
# same class throughout. A rank-1 signature spanning a frequent class takes
# that class's color and label; the class vectors themselves are typically
# wide for this system and are not printed.

TOP_CLASSES = 5

class_objects = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in class_objects]
nonzero_classes = [
    (key, hclass) for key, hclass in zip(class_keys, class_objects, strict=True) if any(key)
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

WINDOW_LENGTH = 600
TIME_WINDOW_START = 0
TIME_WINDOW_STOP = 20000
COLUMN_STEP = 25
ROW_COUNT = 60

extent_start, extent_stop = STORAGE.extent()
column_starts = list(
    range(
        max(TIME_WINDOW_START, extent_start),
        min(TIME_WINDOW_STOP, extent_stop - WINDOW_LENGTH + 1),
        COLUMN_STEP,
    )
)
row_thresholds = [BAND_TOP * (row + 0.5) / ROW_COUNT for row in range(ROW_COUNT)]

column_filtrations: list[tuple[list[float], list[cs.Subspace]]] = []
for window_start in column_starts:
    signature = STORAGE.signature(range(window_start, window_start + WINDOW_LENGTH))
    births = sorted(set(signature.births()))
    column_filtrations.append((births, [signature.span_at(birth) for birth in births]))

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
    frequent_keys: list[tuple[int, ...] | None] = []
    for subspace in subspaces:
        key = None
        if subspace.rank() == 1:
            key = next(key for key, hclass in nonzero_classes if subspace.contains(hclass))
            if key not in CLASS_COLORS:
                key = None
        frequent_keys.append(key)

    used = {CLASS_COLORS[key] for key in frequent_keys if key is not None}
    remaining = [color for color in _support.signature_colors() if color not in used]

    colors: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for position, (subspace, key) in enumerate(zip(subspaces, frequent_keys, strict=True), 1):
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
        born = bisect_right(births, threshold) - 1
        if born < 0:
            continue
        labels[row, column] = library_index.get(spans[born], OTHER_LABEL)

# %%
# **Render the heatmap.** ``origin="lower"`` puts threshold zero at the
# bottom, so each column reads upward as its window's filtration; the top
# edge of the image is the band top.


def build_figure() -> plt.Figure:
    """Return the signature filtration heatmap figure."""
    library_colors, library_labels = library_colors_and_labels(library)
    colors = [(1.0, 1.0, 1.0), *library_colors, (0.75, 0.75, 0.75)]
    tick_labels = ["trivial", *library_labels, "other non-trivial"]
    colormap = ListedColormap(colors)

    figure, axes = plt.subplots(figsize=(14, 6))
    image = axes.imshow(
        labels,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=colormap,
        vmin=-0.5,
        vmax=OTHER_LABEL + 0.5,
        extent=(TIME_WINDOW_START, TIME_WINDOW_STOP, 0.0, BAND_TOP),
    )

    axes.set_xlabel("Time (sample index)")
    axes.set_ylabel("Adjacency threshold t")
    axes.set_title(f"Signature filtration: Dadras attractor, window length {WINDOW_LENGTH}")

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_ticks(range(OTHER_LABEL + 1))
    colorbar.set_ticklabels(tick_labels)

    figure.tight_layout()
    return figure


figure = build_figure()
