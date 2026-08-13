# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle duration by homology class (Lorenz)
============================================

How long the cycles of each homology class last, as a share of that class's
own population. Duration is the integration time from a cycle's first detection
point to its last, so a peak marks a recurrence time the class prefers. Each
class is visibly multi-modal, and prefer different times: two of them share
every peak, while the third takes the gaps between those peaks and the trivial
class takes a time none of the others do. The curves are normalized per class,
so a tall peak means the class concentrates there rather than that the class is
frequent.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. A cycle's
# ``range()`` indexes the detection trajectory, whose ``parameters()`` give
# the integration time of each detection point and so turn that range into a
# duration.

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.lorenz_trajectory())
STORAGE = cs.CycleStorage.load(_support.lorenz_storage())
PARAMETERS = TRAJECTORY.parameters()

# %%
# **Assign canonical class colors.** Each class maps to a stable color through
# the shared ``class_color_map``, so a class keeps its color across the Lorenz
# examples. Nonzero classes are listed in ascending key order, the order that
# map assigns colors in. The trivial class is drawn in gray.

classes = STORAGE.classes()
class_keys = [
    tuple(int(value) for value in homology_class.to_array()) for homology_class in classes
]
CLASS_COLORS = _support.class_color_map(class_keys)
TRIVIAL_COLOR = (0.55, 0.55, 0.55)

nonzero_keys = sorted({key for key in class_keys if any(key)})

# %%
# **Collect every cycle's duration per class.** Each ``Component`` exposes its
# cycles through ``cycles()``, and every ``Cycle`` carries its point
# ``range()``; the duration is the time between the parameters of the range's
# first and last detection point.

durations_by_key: defaultdict[tuple[int, ...], list[float]] = defaultdict(list)
for component in STORAGE.components():
    key = class_keys[component.class_id()]
    for cycle in component.cycles():
        cycle_start, cycle_stop = cycle.range()
        durations_by_key[key].append(float(PARAMETERS[cycle_stop - 1] - PARAMETERS[cycle_start]))

# %%
# **Draw one curve per class.** A shared set of duration bins, each class's
# counts divided by its own total, so classes of different sizes are
# comparable.

BIN_COUNT = 200


def build_figure() -> plt.Figure:
    """Return the per-class duration figure."""
    duration_max = max(max(durations) for durations in durations_by_key.values())
    edges = np.linspace(0.0, duration_max, BIN_COUNT + 1)

    figure, axes = plt.subplots(figsize=(11, 6))
    for key in nonzero_keys:
        counts = np.histogram(durations_by_key[key], bins=edges)[0]
        axes.step(
            edges[:-1],
            counts / counts.sum(),
            where="post",
            color=CLASS_COLORS[key],
            linewidth=1.2,
            label=f"[{' '.join(map(str, key))}]",
        )
    trivial_key = next((key for key in class_keys if not any(key)), None)
    if trivial_key is not None:
        counts = np.histogram(durations_by_key[trivial_key], bins=edges)[0]
        axes.step(
            edges[:-1],
            counts / counts.sum(),
            where="post",
            color=TRIVIAL_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="trivial class",
        )

    axes.set_xlim(0.0, duration_max)
    axes.set_xlabel("Cycle duration (time)")
    axes.set_ylabel("Share of the class's cycles")
    axes.set_title("Cycle duration by homology class: Lorenz attractor")
    axes.legend(title="Class", loc="upper right", fontsize=9)
    figure.tight_layout()
    return figure


figure = build_figure()
