# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle birth and duration population (Lorenz)
===============================================

Every stored cycle binned by its birth (the metric distance between its
endpoint detection points) and its duration (the integration time from its
first detection point to its last), colored by homology class: cell intensity
grows with the number of cycles, and a blended hue marks a cell shared by
several classes. The dashed line marks the cube side length, 1, the distance
a pair must be under to be admitted. A vertical cut at any birth cap keeps
exactly the cycles a
filtered query at that cap admits, so the plot shows the data each birth cap
trades away.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. A cycle's
# ``range()`` indexes the detection trajectory, whose ``parameters()`` give
# the integration time of each detection point and so turn that range into a
# duration.
# A pair is admitted only under the cube side length, 1, so every stored
# birth lies below it.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.lorenz_trajectory())
STORAGE = cs.CycleStorage.load(_support.lorenz_storage())
PARAMETERS = TRAJECTORY.parameters()

# %%
# **Canonical class colors.** Each homology class maps to a stable color via
# ``class_color_map``, shared with the other Lorenz examples. Trivial-class
# cycles are kept but drawn in gray: they close a loop in the cover without
# enclosing a hole.

classes = STORAGE.classes()
class_keys = [
    tuple(int(value) for value in homology_class.to_array()) for homology_class in classes
]
CLASS_COLORS = _support.class_color_map(class_keys)
nonzero_keys = sorted(key for key in set(class_keys) if any(key))

# %%
# **Collect every cycle's birth and duration per class.** Each ``Component``
# exposes its cycles through ``cycles()``, and every ``Cycle`` carries its
# ``birth()`` and its point ``range()``; the duration is the time between the
# parameters of the range's first and last detection point.

births_by_key: dict[tuple[int, ...], list[float]] = {key: [] for key in class_keys}
durations_by_key: dict[tuple[int, ...], list[float]] = {key: [] for key in class_keys}
for component in STORAGE.components():
    key = class_keys[component.class_id()]
    for cycle in component.cycles():
        cycle_start, cycle_stop = cycle.range()
        births_by_key[key].append(cycle.birth())
        durations_by_key[key].append(float(PARAMETERS[cycle_stop - 1] - PARAMETERS[cycle_start]))

# %%
# **Bin and composite the population.** A uniform grid of duration bins by
# birth bins, fine enough that the recurrence time scales read as distinct
# bands: cycles of one detection point count do not all last the same time.
# Each cell's color is the count-weighted mix of the class colors present, its
# intensity grows with the logarithm of the cell's weighted count, and empty
# cells stay white; a blended hue therefore means several classes share it.
# Trivial cycles are drawn with reduced weight, so they read as background
# where classes overlap them.

BIRTH_BIN_COUNT = 500
DURATION_BIN_COUNT = 400


def build_figure() -> plt.Figure:
    """Return the cycle population composite."""
    groups: list[tuple[tuple[float, float, float], float, list[float], list[float], str]] = []
    trivial_key = next((key for key in class_keys if not any(key)), None)
    if trivial_key is not None and births_by_key[trivial_key]:
        gray = (0.75, 0.75, 0.75)
        groups.append(
            (gray, 0.2, births_by_key[trivial_key], durations_by_key[trivial_key], "trivial class")
        )
    for key in nonzero_keys:
        if not births_by_key[key]:
            continue
        label = f"[{' '.join(map(str, key))}]"
        groups.append((CLASS_COLORS[key], 1.0, births_by_key[key], durations_by_key[key], label))

    duration_max = max(max(durations) for _, _, _, durations, _ in groups)
    duration_edges = np.linspace(0.0, duration_max, DURATION_BIN_COUNT + 1)
    birth_edges = np.linspace(0.0, 1.0, BIRTH_BIN_COUNT + 1)
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
        extent=(0.0, 1.0, 0.0, duration_edges[-1]),
    )
    axes.set_xlim(0.0, 1.04)

    cube_side_line = axes.axvline(
        1.0, color="0.3", linestyle="--", linewidth=1.0, label="cube side length"
    )
    swatches = [
        Line2D([], [], linestyle="none", marker="s", markersize=8, color=color, label=label)
        for color, _, _, _, label in groups
    ]
    axes.set_xlabel("Birth (metric distance)")
    axes.set_ylabel("Cycle duration (time)")
    axes.set_title("Cycle births: Lorenz attractor")
    axes.legend(handles=[*swatches, cube_side_line], loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure


figure = build_figure()
