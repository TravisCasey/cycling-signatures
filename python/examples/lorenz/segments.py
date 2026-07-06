# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle segments overlaid on the trajectory
=============================================

A detected cycle is a contiguous run of trajectory samples that nearly returns
to its start. Because the storage indexes cycles by sample range, and the raw
trajectory is sample-indexed too, a cycle's range slices the raw positions
directly: ``RAW[cycle_start:cycle_stop]`` is the loop it traces through space.

This example picks a window of the Lorenz trajectory that shows a clear
representative of each cycle class, then overlays those loops, in their class
colors (shared with the other gallery examples), on a longer stretch of
trajectory that traces out the attractor shape. The single-wing classes trace
one lobe; the both-wings class traces the full figure-eight. The second figure
shows the analysis window as coordinate-versus-sample traces, with each
representative cycle's sample range shaded, locating the loops in time.
"""

# %%
# Load the raw trajectory and the prebuilt ``CycleStorage`` from the published
# example data, fetched and cached on first use. The storage indexes samples of
# a swath of the raw trajectory that skips a leading off-attractor transient
# (the swath ``examples/data/generate_lorenz.py`` embeds); slicing the raw to
# that swath keeps the positions sample-indexed, so a cycle's sample range
# slices them directly.

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

TRANSIENT = 10_000
SWATH = 400_000

RAW = np.load(_support.lorenz_path("lorenz_raw.npy"))[TRANSIENT : TRANSIENT + SWATH]
STORAGE = cs.CycleStorage.load(_support.lorenz_path("lorenz_storage.cyc"))
COMPONENTS = STORAGE.components()

# %%
# **Canonical class colors.** Each homology class maps to a stable color via
# ``class_color_map``, shared with the coverage-barcode, signature-indicator,
# and dominance examples. Only nonzero classes are drawn, in ascending key
# order.

classes = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in classes]
CLASS_COLORS = _support.class_color_map(class_keys)
nonzero_keys = sorted(key for key in set(class_keys) if any(key))

# %%
# **Pick a window with a clear loop of each class.** Take each component's
# shortest cycle (its tightest single recurrence), as the cleanest geometric
# representative. Each ``Cycle`` reports its sample ``range()``, so a
# representative is just a sample interval.

WINDOW_LENGTH = 440
WINDOW_SCAN_STEP = 40

representative_class = []
representative_start = []
representative_stop = []
for component in COMPONENTS:
    class_key = class_keys[component.class_id()]
    if not any(class_key):
        continue
    cycle_start, cycle_stop = component.shortest_cycle().range()
    representative_class.append(class_key)
    representative_start.append(cycle_start)
    representative_stop.append(cycle_stop)

representative_start = np.array(representative_start)
representative_stop = np.array(representative_stop)
representative_length = representative_stop - representative_start
class_mask = {
    key: np.array([candidate == key for candidate in representative_class]) for key in nonzero_keys
}

# %%
# Scan window starts and choose the one whose longest per-class representative
# is as short as possible, so every class shows a tight single loop. A window
# qualifies only if it contains a representative of every class.

extent_start, extent_stop = STORAGE.extent()

best_start = extent_start
best_score = None
for window_start in range(extent_start, extent_stop - WINDOW_LENGTH + 1, WINDOW_SCAN_STEP):
    inside = (representative_start >= window_start) & (
        representative_stop <= window_start + WINDOW_LENGTH
    )
    per_class_shortest = []
    for key in nonzero_keys:
        lengths = representative_length[inside & class_mask[key]]
        if lengths.size == 0:
            per_class_shortest = None
            break
        per_class_shortest.append(int(lengths.min()))
    if per_class_shortest is not None:
        score = max(per_class_shortest)
        if best_score is None or score < best_score:
            best_score = score
            best_start = window_start

window_start = best_start
window_stop = window_start + WINDOW_LENGTH

# %%
# **Select one representative cycle per class** inside the chosen window: the
# shortest in-window loop of each class.

representatives: dict[tuple[int, ...], tuple[int, int]] = {}
inside_window = (representative_start >= window_start) & (representative_stop <= window_stop)
for key in nonzero_keys:
    candidate = inside_window & class_mask[key]
    masked_length = np.where(candidate, representative_length, representative_length.max() + 1)
    chosen = int(np.argmin(masked_length))
    representatives[key] = (int(representative_start[chosen]), int(representative_stop[chosen]))

# %%
# **Overlay the loops on the trajectory.** A longer stretch of the trajectory is
# drawn in faint gray to trace the attractor shape, and each representative
# cycle is overdrawn in its class color. ``RAW[start:stop]`` is the loop, a
# direct slice of the sample-indexed positions.

CONTEXT_LENGTH = 5000

context_center = (window_start + window_stop) // 2
context_start = max(extent_start, context_center - CONTEXT_LENGTH // 2)
context_stop = min(extent_stop, context_start + CONTEXT_LENGTH)


def build_overlay_figure() -> plt.Figure:
    """Return the 3-D trajectory overlay."""
    figure = plt.figure(figsize=(9, 8))
    axes = figure.add_subplot(projection="3d")
    context_path = RAW[context_start:context_stop]
    axes.plot(
        context_path[:, 0],
        context_path[:, 1],
        context_path[:, 2],
        color="0.7",
        linewidth=0.5,
        alpha=0.7,
    )
    for key in nonzero_keys:
        cycle_start, cycle_stop = representatives[key]
        loop = RAW[cycle_start:cycle_stop]
        axes.plot(
            loop[:, 0],
            loop[:, 1],
            loop[:, 2],
            color=CLASS_COLORS[key],
            linewidth=2.2,
            label=f"[{' '.join(map(str, key))}] cycle",
        )
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("Cycle segments on the Lorenz attractor")
    axes.view_init(elev=25, azim=-75)
    axes.legend(loc="upper left", fontsize=9)
    return figure


overlay_figure = build_overlay_figure()

# %%
# **Locate the loops in time.** The same window as coordinate-versus-sample
# traces. Each representative cycle's sample range is shaded in its class color,
# showing when along the trajectory the loop happens.


def build_timeseries_figure() -> plt.Figure:
    """Return the coordinate-versus-sample traces with cycle ranges shaded."""
    samples = np.arange(window_start, window_stop)
    window_path = RAW[window_start:window_stop]

    figure, panels = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
    for panel, coordinate, axis_label in zip(panels, range(3), "xyz", strict=True):
        panel.plot(samples, window_path[:, coordinate], color="0.4", linewidth=0.9)
        for key in nonzero_keys:
            cycle_start, cycle_stop = representatives[key]
            panel.axvspan(cycle_start, cycle_stop, color=CLASS_COLORS[key], alpha=0.25)
        panel.set_ylabel(axis_label)

    panels[-1].set_xlabel("Sample index")
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS[key], linewidth=6, alpha=0.5)
        for key in nonzero_keys
    ]
    labels = [f"[{' '.join(map(str, key))}] cycle" for key in nonzero_keys]
    panels[0].legend(handles, labels, loc="upper right", fontsize=9)
    figure.suptitle("Cycle segments in time: Lorenz attractor")
    figure.tight_layout()
    return figure


timeseries_figure = build_timeseries_figure()
