# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Coverage barcode
==================

Three barcodes stacked by the longest cycle each panel admits. Every row tracks
one nonzero homology class over time, colored where some near-recurrent cycle of
that class, no longer than the panel's length cap, has a sample range covering
the time, and white otherwise. Shorter caps admit only the tightest returns;
raising the cap fills the rows in.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the published example data, fetched
# and cached on first use.
# ``extent()`` gives the half-open sample range covered by all stored
# components, and ``max_length()`` is the longest cycle the storage was built to
# detect; the per-panel caps below stay under it.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.lorenz_path("lorenz_storage.cyc"))
EXTENT_START, EXTENT_STOP = STORAGE.extent()
COMPONENTS = STORAGE.components()

# %%
# **Canonical class colors.** Each homology class maps to a stable color via
# ``class_color_map``, shared with the signature-indicator example so the same
# class gets the same color in both plots.

classes = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in classes]
CLASS_COLORS = _support.class_color_map(class_keys)

# %%
# **Order the rows by homology class.** Nonzero classes are listed in ascending
# key order, the same order ``class_color_map`` uses to assign palette colors,
# so every class-ordered plot in the gallery lists classes the same way. The
# trivial (all-zero) class is excluded: a trivial-class row would render white
# on white.

TOP_CLASSES = 5

ordered_class_ids = sorted(
    (class_id for class_id, key in enumerate(class_keys) if any(key)),
    key=lambda class_id: class_keys[class_id],
)[:TOP_CLASSES]

num_rows = len(ordered_class_ids)
row_by_class_id = {class_id: row_index for row_index, class_id in enumerate(ordered_class_ids)}

# %%
# **Collect the cycles in the time window.** Each ``Component`` exposes its
# cycles through ``cycles()``, and every ``Cycle`` carries its sample
# ``range()`` and ``length()``. Filtering cycles by length approximates the
# figure a storage built at a smaller cap would produce, so the single fixture
# feeds every panel. The window keeps the figure to a legible slice of the full
# extent.

COLUMN_STEP = 5
TIME_WINDOW_START = EXTENT_START
TIME_WINDOW_STOP = min(EXTENT_START + 10000, EXTENT_STOP)

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

LENGTHS = (160, 230, 300)


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
# and assigns each ranked class its canonical color. The y-axis tick labels show
# the class vector so the reader can relate rows to homology classes, and each
# panel is labeled with its cycle-length cap.

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
        panel.set_yticklabels(
            [f"[{' '.join(map(str, class_keys[class_id]))}]" for class_id in ordered_class_ids]
        )
        panel.set_title(f"cycles up to length {length}", loc="left")

    panels[-1].set_xlabel("Time (sample index)")
    figure.supylabel("Homology class")
    figure.suptitle("Coverage barcode: Lorenz attractor")
    figure.tight_layout()
    return figure


figure = build_figure()
