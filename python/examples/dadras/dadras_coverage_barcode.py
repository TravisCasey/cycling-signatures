# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Coverage barcode
==================

Three barcodes stacked by the birth cap each panel admits. Every row tracks
one frequent homology class of the Dadras attractor over time, colored where
some cycle of that class born by the panel's threshold spans the time. A
cycle's birth is the metric distance between its endpoint samples, so low caps
admit only the tightest recurrences; raising the cap toward the top of the
stored detection band fills the rows in. All panels admit only cycles up to
one fixed length, counted in detection samples rather than in time, so
coverage always means participation in a recurrence at that declared sampling
scale.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. ``extent()`` gives
# the half-open sample range covered by all stored components, in indices into
# the detection trajectory, and that trajectory's ``parameters()`` give the
# integration time of each sample, which turns a cycle's sample range into the
# span of time it covers. ``threshold()`` is the top of the stored detection
# band; the per-panel birth caps below sit at or under it.

import math
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
# ``range()`` and ``birth()``. A cycle's sample range becomes the closed span
# of time between its first and last sample. Filtering cycles by birth matches
# the cycles a detection capped at that threshold would admit. Cycles longer
# than ``LENGTH_CAP`` detection samples are excluded; that cap is a sample
# count, not a duration, so the time a capped cycle covers varies with the
# local speed of the flow. The window restricts the figure to a legible span
# of the full extent, and its columns are a uniform width in time.

LENGTH_CAP = 800
WINDOW_DURATION = 192.0
COLUMN_DURATION = 0.16

TIME_WINDOW_START = float(PARAMETERS[EXTENT_START])
TIME_WINDOW_STOP = min(TIME_WINDOW_START + WINDOW_DURATION, float(PARAMETERS[EXTENT_STOP - 1]))

column_times = np.arange(TIME_WINDOW_START, TIME_WINDOW_STOP, COLUMN_DURATION)
num_cols = len(column_times)

windowed_cycles: list[tuple[int, float, float, float]] = []
for component in COMPONENTS:
    class_id = component.class_id()
    if class_id not in row_by_class_id:
        continue
    coverage_start, coverage_stop = component.coverage()
    if (
        PARAMETERS[coverage_stop - 1] <= TIME_WINDOW_START
        or PARAMETERS[coverage_start] >= TIME_WINDOW_STOP
    ):
        continue
    row_index = row_by_class_id[class_id]
    for cycle in component.cycles():
        if cycle.length() > LENGTH_CAP:
            continue
        cycle_start, cycle_stop = cycle.range()
        first_time = float(PARAMETERS[cycle_start])
        last_time = float(PARAMETERS[cycle_stop - 1])
        if last_time <= TIME_WINDOW_START or first_time >= TIME_WINDOW_STOP:
            continue
        windowed_cycles.append((row_index, first_time, last_time, cycle.birth()))

# %%
# **Build one label array per birth cap.** A cell is its class row index plus
# one where any cycle born by the cap covers it, and zero (white) otherwise.
# The lower caps are quantiles of the windowed cycles' births and the last cap
# is the band top, so the panels sweep the stored band from the tightest
# recurrences up to everything the storage detected.

BIRTH_QUANTILE_LEVELS = (0.25, 0.50)

all_births = np.array([birth for _, _, _, birth in windowed_cycles])
BIRTH_CAPS = (
    *(float(np.quantile(all_births, level)) for level in BIRTH_QUANTILE_LEVELS),
    BAND_TOP,
)


def coverage_labels(max_birth: float) -> np.ndarray:
    """Return the label array for cycles born by ``max_birth``."""
    labels = np.zeros((num_rows, num_cols), dtype=np.int8)
    for row_index, first_time, last_time, cycle_birth in windowed_cycles:
        if cycle_birth > max_birth:
            continue
        # Columns whose time falls in the closed span [first_time, last_time].
        first_column = int(np.searchsorted(column_times, first_time, side="left"))
        last_column = int(np.searchsorted(column_times, last_time, side="right"))
        labels[row_index, first_column:last_column] = row_index + 1
    return labels


# %%
# **Render the stacked barcodes.** The colormap starts with white (uncovered)
# and assigns each ranked class its canonical color. The y-axis tick labels
# name classes by frequency position, and each panel is labeled with its
# birth cap.

row_colors = [CLASS_COLORS[class_keys[class_id]] for class_id in ordered_class_ids]


def build_figure() -> plt.Figure:
    """Return the stacked coverage barcode figure."""
    colormap = ListedColormap([(1.0, 1.0, 1.0), *row_colors])

    figure, panels = plt.subplots(len(BIRTH_CAPS), 1, sharex=True, figsize=(14, 8))
    for panel, cap in zip(panels, BIRTH_CAPS, strict=True):
        panel.imshow(
            coverage_labels(cap),
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=-0.5,
            vmax=num_rows + 0.5,
            extent=(TIME_WINDOW_START, TIME_WINDOW_STOP, num_rows - 0.5, -0.5),
        )
        panel.set_yticks(range(num_rows))
        panel.set_yticklabels([f"class {row_index + 1}" for row_index in range(num_rows)])
        panel.set_title(f"cycles born by t <= {cap:.3f}", loc="left")

    panels[-1].set_xlabel("Time")
    figure.supylabel("Homology class (by frequency)")
    figure.suptitle(
        f"Coverage barcode: Dadras attractor (cycles up to {LENGTH_CAP} detection samples)"
    )
    figure.tight_layout()
    return figure


figure = build_figure()
