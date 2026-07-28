# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle birth and length population
====================================

Every stored cycle binned by its birth (the metric distance between its
endpoint samples) and its length in samples, colored by homology class:
cell intensity grows with the number of cycles, and a blended hue marks a
cell several classes share. The dashed line marks the top of the stored
detection band. A vertical cut at any threshold keeps exactly the cycles a
detection capped there would admit, so the plot shows what each threshold
choice trades away.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the published example data, fetched
# and cached on first use. ``threshold()`` is the top of the stored detection
# band; every stored birth lies at or under it.

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.lorenz_storage())
BAND_TOP = STORAGE.threshold()
assert math.isfinite(BAND_TOP)

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
# **Collect every cycle's birth and length per class.** Each ``Component``
# exposes its cycles through ``cycles()``, and every ``Cycle`` carries its
# ``birth()`` and ``length()``.

births_by_key: dict[tuple[int, ...], list[float]] = {key: [] for key in class_keys}
lengths_by_key: dict[tuple[int, ...], list[int]] = {key: [] for key in class_keys}
for component in STORAGE.components():
    key = class_keys[component.class_id()]
    for cycle in component.cycles():
        births_by_key[key].append(cycle.birth())
        lengths_by_key[key].append(cycle.length())

# %%
# **Bin and composite the population.** One image row per integer cycle
# length and a fixed grid of birth bins, so the banded length structure
# renders exactly. Each cell's color is the count-weighted mix of the class
# colors present, its intensity grows with the logarithm of the cell's
# weighted count, and empty cells stay white; a blended hue therefore means
# several classes genuinely share that cell. Trivial cycles enter at half
# weight, so they read as background where classes overlap them.

BIRTH_BIN_COUNT = 500
LENGTH_BIN_SIZE = 1


def build_figure() -> plt.Figure:
    """Return the cycle population composite."""
    groups: list[tuple[tuple[float, float, float], float, list[float], list[int], str]] = []
    trivial_key = next((key for key in class_keys if not any(key)), None)
    if trivial_key is not None and births_by_key[trivial_key]:
        gray = (0.75, 0.75, 0.75)
        groups.append(
            (gray, 0.2, births_by_key[trivial_key], lengths_by_key[trivial_key], "trivial class")
        )
    for key in nonzero_keys:
        if not births_by_key[key]:
            continue
        label = f"[{' '.join(map(str, key))}]"
        groups.append((CLASS_COLORS[key], 1.0, births_by_key[key], lengths_by_key[key], label))

    length_max = max(max(lengths) for _, _, _, lengths, _ in groups)
    length_edges = np.arange(-0.5, length_max + 0.5 + LENGTH_BIN_SIZE, LENGTH_BIN_SIZE)
    birth_edges = np.linspace(0.0, BAND_TOP, BIRTH_BIN_COUNT + 1)
    weighted_counts = [
        weight * np.histogram2d(lengths, births, bins=(length_edges, birth_edges))[0]
        for _, weight, births, lengths, _ in groups
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
        extent=(0.0, BAND_TOP, -0.5, length_edges[-1]),
    )
    axes.set_xlim(0.0, BAND_TOP * 1.04)

    band_line = axes.axvline(BAND_TOP, color="0.3", linestyle="--", linewidth=1.0, label="band top")
    swatches = [
        Line2D([], [], linestyle="none", marker="s", markersize=8, color=color, label=label)
        for color, _, _, _, label in groups
    ]
    axes.set_xlabel("Birth (metric distance)")
    axes.set_ylabel("Cycle length (samples)")
    axes.set_title("Cycle births: Lorenz attractor")
    axes.legend(handles=[*swatches, band_line], loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure


figure = build_figure()
