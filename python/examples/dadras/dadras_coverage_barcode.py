# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Coverage barcode
==================

Three barcodes stacked by the longest cycle each panel admits. Every row tracks
one frequent homology class of the Dadras attractor over time, colored where
some near-recurrent cycle of that class, no longer than the panel's length cap,
has a sample range covering the time. Shorter caps admit only the tightest
returns; raising the cap fills the rows in.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the published example data, fetched
# and cached on first use.
# ``extent()`` gives the half-open sample range covered by all stored
# components, and ``max_length()`` is the longest cycle the storage was built to
# detect; the per-panel caps below stay under it.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.dadras_storage())
EXTENT_START, EXTENT_STOP = STORAGE.extent()
COMPONENTS = STORAGE.components()

# %%
# **Rank the classes by frequency.** The Dadras storage keeps many rare,
# swath-dependent homology classes alongside a frequent few that carry the
# attractor's structure. Classes are ranked by how often they recur (their
# total cycle count across components) and only the most frequent are shown;
# the trivial (all-zero) class is excluded. Every class-ranked plot in the
# gallery uses this ordering, so "class 1" names the same class throughout.

TOP_CLASSES = 5

classes = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in classes]

class_cycle_counts: Counter[int] = Counter()
for component in COMPONENTS:
    if any(class_keys[component.class_id()]):
        class_cycle_counts[component.class_id()] += component.cycle_count()

ordered_class_ids = [class_id for class_id, _ in class_cycle_counts.most_common(TOP_CLASSES)]

num_rows = len(ordered_class_ids)
row_by_class_id = {class_id: row_index for row_index, class_id in enumerate(ordered_class_ids)}

# %%
# **Canonical class colors.** Each frequent class maps to a stable color via
# the shared ``class_color_map``. Rows are labeled by frequency position; the
# class vectors themselves are typically wide for this system and are not
# printed.

CLASS_COLORS = _support.class_color_map([class_keys[class_id] for class_id in ordered_class_ids])

# %%
# **Collect the cycles in the time window.** Each ``Component`` exposes its
# cycles through ``cycles()``, and every ``Cycle`` carries its sample
# ``range()`` and ``length()``. Filtering cycles by length closely approximates
# the figure a storage built at a smaller cap would produce. The window
# restricts the figure to a legible slice of the full extent.

COLUMN_STEP = 25
TIME_WINDOW_START = EXTENT_START
TIME_WINDOW_STOP = min(EXTENT_START + 30000, EXTENT_STOP)

column_times = np.arange(TIME_WINDOW_START, TIME_WINDOW_STOP, COLUMN_STEP)
num_cols = len(column_times)

windowed_cycles: list[tuple[int, int, int, int]] = []
for component in COMPONENTS:
    class_id = component.class_id()
    if class_id not in row_by_class_id:
        continue
    coverage_start, coverage_stop = component.coverage()
    if coverage_stop <= TIME_WINDOW_START or coverage_start >= TIME_WINDOW_STOP:
        continue
    row_index = row_by_class_id[class_id]
    for cycle in component.cycles():
        cycle_start, cycle_stop = cycle.range()
        if cycle_stop <= TIME_WINDOW_START or cycle_start >= TIME_WINDOW_STOP:
            continue
        windowed_cycles.append((row_index, cycle_start, cycle_stop, cycle.length()))

# %%
# **Build one label array per cap.** A cell is its class row index plus one
# where any cycle no longer than the cap covers it, and zero (white) otherwise.
# The columns a cycle covers are the sample times that fall inside its range.

LENGTHS = (220, 320, 450)


def coverage_labels(max_cycle_length: int) -> np.ndarray:
    """Return the label array for cycles no longer than ``max_cycle_length``."""
    labels = np.zeros((num_rows, num_cols), dtype=np.int8)
    for row_index, cycle_start, cycle_stop, cycle_length in windowed_cycles:
        if cycle_length > max_cycle_length:
            continue
        # Columns inside the half-open range [cycle_start, cycle_stop).
        first_column = int(np.searchsorted(column_times, cycle_start))
        last_column = int(np.searchsorted(column_times, cycle_stop))
        labels[row_index, first_column:last_column] = row_index + 1
    return labels


# %%
# **Render the stacked barcodes.** The colormap starts with white (uncovered)
# and assigns each ranked class its canonical color. The y-axis tick labels
# name classes by frequency position, and each panel is labeled with its
# cycle-length cap.

row_colors = [CLASS_COLORS[class_keys[class_id]] for class_id in ordered_class_ids]


def build_figure() -> plt.Figure:
    """Return the stacked coverage barcode figure."""
    colormap = ListedColormap([(1.0, 1.0, 1.0), *row_colors])

    figure, panels = plt.subplots(len(LENGTHS), 1, sharex=True, figsize=(14, 8))
    for panel, length in zip(panels, LENGTHS, strict=True):
        panel.imshow(
            coverage_labels(length),
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=-0.5,
            vmax=num_rows + 0.5,
            extent=(TIME_WINDOW_START, TIME_WINDOW_STOP, num_rows - 0.5, -0.5),
        )
        panel.set_yticks(range(num_rows))
        panel.set_yticklabels([f"class {row_index + 1}" for row_index in range(num_rows)])
        panel.set_title(f"cycles up to length {length}", loc="left")

    panels[-1].set_xlabel("Time (sample index)")
    figure.supylabel("Homology class (by frequency)")
    figure.suptitle("Coverage barcode: Dadras attractor")
    figure.tight_layout()
    return figure


figure = build_figure()
