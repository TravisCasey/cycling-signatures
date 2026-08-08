# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle birth and duration population (Dadras)
===============================================

Every stored cycle binned by its birth (the metric distance between its
endpoint detection points) and its duration (the integration time from its
first detection point to its last), colored by homology class: cell intensity
grows with the number of cycles, and a blended hue marks a cell several
classes share. The dashed
line marks the top of the stored detection band. A vertical cut at any
threshold keeps exactly the cycles a detection capped there would admit, so the
plot shows what each threshold choice trades away.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. A cycle's
# ``range()`` indexes the detection trajectory, whose ``parameters()`` give
# the integration time of each detection point and so turn that range into a
# duration. ``threshold()`` is the top of the stored detection band; every
# stored birth lies at or under it.

import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
# same class throughout. Cycles outside the frequent classes (rare classes
# and the trivial class) are drawn in gray.

TOP_CLASSES = 5

classes = STORAGE.classes()
class_keys = [
    tuple(int(value) for value in homology_class.to_array()) for homology_class in classes
]

class_cycle_counts: Counter[int] = Counter()
for component in STORAGE.components():
    if any(class_keys[component.class_id()]):
        class_cycle_counts[component.class_id()] += component.cycle_count()

ordered_class_ids = [class_id for class_id, _ in class_cycle_counts.most_common(TOP_CLASSES)]
CLASS_COLORS = _support.class_color_map([class_keys[class_id] for class_id in ordered_class_ids])
frequent_keys = [class_keys[class_id] for class_id in ordered_class_ids]

# %%
# **Collect every cycle's birth and duration per class.** Each ``Component``
# exposes its cycles through ``cycles()``, and every ``Cycle`` carries its
# ``birth()`` and its point ``range()``; the duration is the time between the
# parameters of the range's first and last detection point. Cycles are grouped
# by frequent class, with one shared bucket for everything else.

OTHER = "other"

births_by_group: dict[tuple[int, ...] | str, list[float]] = {key: [] for key in frequent_keys}
durations_by_group: dict[tuple[int, ...] | str, list[float]] = {key: [] for key in frequent_keys}
births_by_group[OTHER] = []
durations_by_group[OTHER] = []
for component in STORAGE.components():
    key = class_keys[component.class_id()]
    group: tuple[int, ...] | str = key if key in CLASS_COLORS else OTHER
    for cycle in component.cycles():
        cycle_start, cycle_stop = cycle.range()
        births_by_group[group].append(cycle.birth())
        durations_by_group[group].append(
            float(PARAMETERS[cycle_stop - 1] - PARAMETERS[cycle_start])
        )

# %%
# **Bin and composite the population.** A uniform grid of duration bins by
# birth bins, fine enough that the recurrence time scales read as distinct
# bands: cycles of one detection point count do not all last the same time.
# Each cell's color is the count-weighted mix of the class colors present, its
# intensity grows with the logarithm of the cell's weighted count, and empty
# cells stay white; a blended hue therefore means several classes share that
# cell. Cycles outside the frequent classes are drawn with reduced weight, so
# they read as background where the frequent classes overlap them.

BIRTH_BIN_COUNT = 300
DURATION_BIN_COUNT = 200


def build_figure() -> plt.Figure:
    """Return the cycle population composite."""
    groups: list[tuple[tuple[float, float, float], float, list[float], list[float], str]] = []
    if births_by_group[OTHER]:
        gray = (0.75, 0.75, 0.75)
        groups.append(
            (gray, 0.2, births_by_group[OTHER], durations_by_group[OTHER], "rare or trivial")
        )
    for position, key in enumerate(frequent_keys, start=1):
        if not births_by_group[key]:
            continue
        label = f"class {position}"
        groups.append(
            (CLASS_COLORS[key], 1.0, births_by_group[key], durations_by_group[key], label)
        )

    duration_max = max(max(durations) for _, _, _, durations, _ in groups)
    duration_edges = np.linspace(0.0, duration_max, DURATION_BIN_COUNT + 1)
    birth_edges = np.linspace(0.0, BAND_TOP, BIRTH_BIN_COUNT + 1)
    weighted_counts = [
        weight * np.histogram2d(durations, births, bins=(duration_edges, birth_edges))[0]
        for _, weight, births, durations, _ in groups
    ]

    total_counts = np.sum(weighted_counts, axis=0)
    occupied = total_counts > 0
    mixed_colors = np.zeros((*total_counts.shape, 3))
    for (color, _, _, _, _), counts in zip(groups, weighted_counts, strict=True):
        mixed_colors += counts[:, :, np.newaxis] * np.array(color)
    mixed_colors[occupied] /= total_counts[occupied, np.newaxis]
    intensity = np.zeros_like(total_counts)
    intensity[occupied] = 0.25 + 0.75 * np.log1p(total_counts[occupied]) / np.log1p(
        total_counts.max()
    )
    image = 1.0 - intensity[:, :, np.newaxis] * (1.0 - mixed_colors)

    figure, axes = plt.subplots(figsize=(10, 7))
    axes.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, BAND_TOP, 0.0, duration_edges[-1]),
    )
    axes.set_xlim(0.0, BAND_TOP * 1.04)

    band_line = axes.axvline(BAND_TOP, color="0.3", linestyle="--", linewidth=1.0, label="band top")
    swatches = [
        Line2D([], [], linestyle="none", marker="s", markersize=8, color=color, label=label)
        for color, _, _, _, label in groups
    ]
    axes.set_xlabel("Birth (metric distance)")
    axes.set_ylabel("Cycle duration (time)")
    axes.set_title("Cycle births: Dadras attractor")
    axes.legend(handles=[*swatches, band_line], loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure


figure = build_figure()
