# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle duration by homology class (Dadras)
============================================

How long the cycles of each frequent homology class last, as a share of that
class's own population. Duration is the integration time from a cycle's first
detection point to its last, so a peak marks a recurrence time the class
prefers. Four of the five frequent classes trace one curve, peaking together
and holding nearly identical cycle counts; the fifth prefers a longer time and
is alone there. The rare classes, which hold three quarters of the stored cycles
between them, have no preferred duration at all: their curve is flat wherever
there are cycles to count. The curves are normalized per class, so a tall peak
means the class concentrates there rather than that the class is large.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. A cycle's
# ``range()`` indexes the detection trajectory, whose ``parameters()`` give
# the integration time of each detection point and so turn that range into a
# duration.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.dadras_trajectory())
STORAGE = cs.CycleStorage.load(_support.dadras_storage())
PARAMETERS = TRAJECTORY.parameters()

# %%
# **Order the classes by frequency and assign canonical colors.** Classes are
# ordered by how often they recur (their total cycle count across components),
# the same ordering the other Dadras examples use, so "class 1" names the same
# class throughout. The storage holds many more classes than the plot names:
# the rest are gathered into one rare-class curve, and the trivial class keeps
# its own. Both are drawn in gray, with the trivial one dashed.

TOP_CLASSES = 5
RARE_COLOR = (0.76, 0.76, 0.76)
TRIVIAL_COLOR = (0.45, 0.45, 0.45)

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
# **Collect every cycle's duration per group.** Each ``Component`` exposes its
# cycles through ``cycles()``, and every ``Cycle`` carries its point
# ``range()``; the duration is the time between the parameters of the range's
# first and last detection point.

RARE = "rare"
TRIVIAL = "trivial"

durations_by_group: dict[tuple[int, ...] | str, list[float]] = {key: [] for key in frequent_keys}
durations_by_group[RARE] = []
durations_by_group[TRIVIAL] = []
for component in STORAGE.components():
    key = class_keys[component.class_id()]
    if key in CLASS_COLORS:
        group: tuple[int, ...] | str = key
    elif any(key):
        group = RARE
    else:
        group = TRIVIAL
    for cycle in component.cycles():
        cycle_start, cycle_stop = cycle.range()
        durations_by_group[group].append(
            float(PARAMETERS[cycle_stop - 1] - PARAMETERS[cycle_start])
        )

# %%
# **Draw one curve per group.** A shared set of duration bins, each group's
# counts divided by its own total, so groups of very different sizes are
# comparable. Every group is drawn on one axis rather than in its own panel.

BIN_COUNT = 320
ZOOM_LIMIT = 4.0


def build_figure() -> plt.Figure:
    """Return the per-class duration figure."""
    duration_max = max(max(durations) for durations in durations_by_group.values())
    edges = np.linspace(0.0, duration_max, BIN_COUNT + 1)
    shares = {
        group: np.histogram(durations, bins=edges)[0] / len(durations)
        for group, durations in durations_by_group.items()
    }

    figure, (whole, zoom) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 2]}
    )
    for axes in (whole, zoom):
        axes.step(
            edges[:-1],
            shares[RARE],
            where="post",
            color=RARE_COLOR,
            linewidth=1.0,
            label="rare classes",
        )
        axes.step(
            edges[:-1],
            shares[TRIVIAL],
            where="post",
            color=TRIVIAL_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="trivial class",
        )
        for position, key in enumerate(frequent_keys, start=1):
            axes.step(
                edges[:-1],
                shares[key],
                where="post",
                color=CLASS_COLORS[key],
                linewidth=1.2,
                label=f"class {position}",
            )
        axes.set_ylabel("Share of the class's cycles")

    whole.axvspan(0.0, ZOOM_LIMIT, color="0.92", zorder=0)
    whole.set_xlim(0.0, duration_max)
    whole.set_title("Cycle duration by homology class: Dadras attractor")
    whole.legend(title="Class", loc="upper right", fontsize=9, ncols=2)

    zoom.set_xlim(0.0, ZOOM_LIMIT)
    zoom.set_xlabel("Cycle duration (time)")
    figure.tight_layout()
    return figure


figure = build_figure()
