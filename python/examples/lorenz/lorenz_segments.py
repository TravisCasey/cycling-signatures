# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Cycle segments overlaid on the trajectory
=============================================

A detected cycle is a contiguous run of trajectory samples that nearly returns
to its start.

This example picks a window of the Lorenz trajectory that shows a clear
representative of each cycle class, then overlays those loops, in their class
colors (shared with the other gallery examples), on a longer stretch of
trajectory that traces out the attractor shape. The single-wing classes trace
one lobe; the both-wings class traces the full figure-eight. The second figure
shows the analysis window as coordinate-versus-time traces, with each
representative cycle's time span shaded.
"""

# %%
# Load the raw trajectory, the detection trajectory, and the prebuilt
# ``CycleStorage`` from the published example data, fetched and cached on first
# use. A cycle's sample range indexes the detection trajectory, which is
# thinned relative to the raw samples, so drawing a loop as a smooth curve
# means crossing back to the raw rows. The detection trajectory's
# ``parameters()`` are that bridge: each entry is the integration time its
# point was sampled at, and the raw rows sit a fixed ``LORENZ_DT`` apart in
# time, so recovering a row from a time is a search through the synthesized
# row times rather than a division.

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

RAW = np.load(_support.lorenz_raw())
TRAJECTORY = cs.Trajectory.load(_support.lorenz_trajectory())
STORAGE = cs.CycleStorage.load(_support.lorenz_storage())
COMPONENTS = STORAGE.components()
PARAMETERS = TRAJECTORY.parameters()
ROW_TIMES = np.arange(len(RAW)) * _support.LORENZ_DT


def raw_rows(start: int, stop: int) -> tuple[int, int]:
    """Return the raw row range the sample range ``[start, stop)`` spans.

    Searching the raw row times to the right of a sample's time gives the
    first row past it. One less is the row the start sample sits at or just
    after; taken at the last sample it is one past that sample's row, so
    slicing the raw positions with the pair covers every sample in the range.
    """
    first = int(np.searchsorted(ROW_TIMES, PARAMETERS[start], side="right")) - 1
    last = int(np.searchsorted(ROW_TIMES, PARAMETERS[stop - 1], side="right"))
    return first, last


# %%
# **Canonical class colors.** Each homology class maps to a stable color via
# ``class_color_map``, shared with the coverage-barcode, signature-indicator,
# and dominance examples. Only nonzero classes are drawn, in ascending key
# order.

classes = STORAGE.classes()
class_keys = [
    tuple(int(value) for value in homology_class.to_array()) for homology_class in classes
]
CLASS_COLORS = _support.class_color_map(class_keys)
nonzero_keys = sorted(key for key in set(class_keys) if any(key))

# %%
# **Pick a window with a clear loop of each class.** Take each component's
# shortest cycle (its tightest single recurrence), as the cleanest geometric
# representative. Each ``Cycle`` reports its sample ``range()``, so a
# representative is just a sample interval.

WINDOW_LENGTH = 630
WINDOW_SCAN_STEP = 60

representative_class = []
start_values: list[int] = []
stop_values: list[int] = []
for component in COMPONENTS:
    class_key = class_keys[component.class_id()]
    if not any(class_key):
        continue
    cycle_start, cycle_stop = component.shortest_cycle().range()
    representative_class.append(class_key)
    start_values.append(cycle_start)
    stop_values.append(cycle_stop)

representative_start = np.array(start_values)
representative_stop = np.array(stop_values)
representative_length = representative_stop - representative_start
class_mask = {
    key: np.array([candidate == key for candidate in representative_class]) for key in nonzero_keys
}

# %%
# Scan window starts and pick, class by class, the shortest in-window loop
# that does not overlap the loops already picked; the chosen window minimizes
# the longest picked loop. Disjoint loops keep the classes visually separate
# in both figures. If no window admits a fully disjoint pick, the scan repeats
# without the disjointness requirement.

extent_start, extent_stop = STORAGE.extent()


def pick_representatives(
    window_start: int, disjoint: bool
) -> dict[tuple[int, ...], tuple[int, int]] | None:
    """Return one loop per class in the window, or None if a class has none."""
    inside = (representative_start >= window_start) & (
        representative_stop <= window_start + WINDOW_LENGTH
    )
    picked: dict[tuple[int, ...], tuple[int, int]] = {}
    for key in nonzero_keys:
        candidates = np.flatnonzero(inside & class_mask[key])
        candidates = candidates[np.argsort(representative_length[candidates])]
        chosen = None
        for candidate in candidates:
            start = int(representative_start[candidate])
            stop = int(representative_stop[candidate])
            overlapping = any(
                start < other_stop and other_start < stop
                for other_start, other_stop in picked.values()
            )
            if disjoint and overlapping:
                continue
            chosen = (start, stop)
            break
        if chosen is None:
            return None
        picked[key] = chosen
    return picked


best_representatives: dict[tuple[int, ...], tuple[int, int]] | None = None
best_start = extent_start
best_score = None
for require_disjoint in (True, False):
    for window_start in range(extent_start, extent_stop - WINDOW_LENGTH + 1, WINDOW_SCAN_STEP):
        picked = pick_representatives(window_start, require_disjoint)
        if picked is None:
            continue
        score = max(stop - start for start, stop in picked.values())
        if best_score is None or score < best_score:
            best_score = score
            best_start = window_start
            best_representatives = picked
    if best_representatives is not None:
        break

assert best_representatives is not None
representatives: dict[tuple[int, ...], tuple[int, int]] = best_representatives
window_start = best_start
window_stop = window_start + WINDOW_LENGTH

# %%
# **Overlay the loops on the trajectory.** A longer stretch of the trajectory is
# drawn in faint gray to trace the attractor shape, and each representative
# cycle is overdrawn in its class color. Each sample range is carried through
# ``raw_rows`` and the raw positions sliced with the result, so the loops draw
# at the raw sampling density rather than the detection one.

CONTEXT_LENGTH = 7000

context_center = (window_start + window_stop) // 2
context_start = max(extent_start, context_center - CONTEXT_LENGTH // 2)
context_stop = min(extent_stop, context_start + CONTEXT_LENGTH)


def build_overlay_figure() -> plt.Figure:
    """Return the 3-D trajectory overlay."""
    figure = plt.figure(figsize=(9, 8))
    axes = figure.add_subplot(projection="3d")
    first_row, last_row = raw_rows(context_start, context_stop)
    context_path = RAW[first_row:last_row]
    axes.plot(
        context_path[:, 0],
        context_path[:, 1],
        context_path[:, 2],
        color="0.7",
        linewidth=0.5,
        alpha=0.7,
    )
    draw_keys = sorted(
        nonzero_keys,
        key=lambda key: representatives[key][1] - representatives[key][0],
        reverse=True,
    )
    for key in draw_keys:
        cycle_first_row, cycle_last_row = raw_rows(*representatives[key])
        loop = RAW[cycle_first_row:cycle_last_row]
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
# **Locate the loops in time.** The same window as coordinate-versus-time
# traces, drawn at the raw sampling density. Each representative cycle's time
# span is shaded in its class color, showing when along the trajectory the loop
# happens.


def build_timeseries_figure() -> plt.Figure:
    """Return the coordinate-versus-time traces with cycle spans shaded."""
    first_row, last_row = raw_rows(window_start, window_stop)
    row_times = ROW_TIMES[first_row:last_row]
    window_path = RAW[first_row:last_row]

    figure, panels = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
    for panel, coordinate, axis_label in zip(panels, range(3), "xyz", strict=True):
        panel.plot(row_times, window_path[:, coordinate], color="0.4", linewidth=0.9)
        for key in nonzero_keys:
            cycle_start, cycle_stop = representatives[key]
            panel.axvspan(
                PARAMETERS[cycle_start],
                PARAMETERS[cycle_stop - 1],
                color=CLASS_COLORS[key],
                alpha=0.25,
            )
        panel.set_ylabel(axis_label)

    panels[-1].set_xlabel("Time")
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
