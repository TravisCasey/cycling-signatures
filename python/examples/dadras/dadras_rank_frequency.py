# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Rank-frequency histogram (Dadras)
====================================

Frequency of appearance of each signature rank as a function of window length.
Longer windows capture more independent loop types of the Dadras attractor,
shifting the distribution toward higher ranks as the length grows. Windows are
swept in detection points, and each length is presented at the median time its
windows span.
"""

# %%
# Load the detection trajectory and the prebuilt ``CycleStorage`` from the
# published example data, fetched and cached on first use. The trajectory
# contributes only its ``parameters()``, the integration time of each detection
# point, which turn a window length in detection points into a duration.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

TRAJECTORY = cs.Trajectory.load(_support.dadras_trajectory())
STORAGE = cs.CycleStorage.load(_support.dadras_storage())
PARAMETERS = TRAJECTORY.parameters()

# %%
# Sweep over window lengths and tally rank occurrences. For each length the
# window slides across the full storage extent and each
# ``signature(...).rank()`` call queries the number of independent loop types
# this window contains.

LENGTH_STEP = 10
SCAN_STEP = 125

extent_start, extent_stop = STORAGE.extent()
extent_length = extent_stop - extent_start
max_length = min(STORAGE.max_length(), extent_length)

window_lengths: list[int] = list(range(LENGTH_STEP, max_length + 1, LENGTH_STEP))

rank_counts: list[Counter[int]] = []
for length in window_lengths:
    counter: Counter[int] = Counter()
    for window_start in range(extent_start, extent_stop - length + 1, SCAN_STEP):
        rank = STORAGE.signature(range(window_start, window_start + length)).rank()
        counter[rank] += 1
    rank_counts.append(counter)

all_ranks = sorted({rank for counter in rank_counts for rank in counter})

# %%
# Place each window length on a time axis. A length is a detection point count,
# and the time such a window spans varies along the trajectory, so a length is
# drawn at the median time spanned by exactly the windows tallied above.
# Consecutive medians are not evenly spaced, so each bar runs from the midpoint
# before its median to the midpoint after: the bars tile the axis and their
# widths show how much time each step in window length buys.

duration_values: list[float] = []
for length in window_lengths:
    starts = np.arange(extent_start, extent_stop - length + 1, SCAN_STEP)
    duration_values.append(float(np.median(PARAMETERS[starts + length - 1] - PARAMETERS[starts])))
median_durations = np.array(duration_values)

bar_boundaries = np.empty(len(median_durations) + 1)
bar_boundaries[1:-1] = (median_durations[:-1] + median_durations[1:]) / 2
bar_boundaries[0] = 2 * median_durations[0] - bar_boundaries[1]
bar_boundaries[-1] = 2 * median_durations[-1] - bar_boundaries[-2]
bar_widths = np.diff(bar_boundaries)

# %%
# Build a stacked bar chart. Each bar represents one window length; the stacked
# segments show how many windows produced each rank. Viridis colors the rank
# stack, dark purple at rank 0 through to yellow at the highest rank present.


def build_figure() -> plt.Figure:
    """Return the rank-frequency stacked bar figure."""
    colormap = plt.get_cmap("viridis")
    num_ranks = len(all_ranks)
    rank_colors = {
        rank: colormap(position / max(num_ranks - 1, 1)) for position, rank in enumerate(all_ranks)
    }

    figure, axes = plt.subplots(figsize=(10, 5))

    bar_bottoms = [0] * len(window_lengths)
    for rank in all_ranks:
        heights = [counter.get(rank, 0) for counter in rank_counts]
        axes.bar(
            bar_boundaries[:-1],
            heights,
            bottom=bar_bottoms,
            width=bar_widths,
            align="edge",
            color=rank_colors[rank],
            label=f"rank {rank}",
            linewidth=0,
        )
        bar_bottoms = [bottom + height for bottom, height in zip(bar_bottoms, heights, strict=True)]

    axes.set_xlabel("Median window duration")
    axes.set_ylabel("Number of windows")
    axes.set_title("Signature rank frequency vs. window length (Dadras)")
    axes.set_xlim(bar_boundaries[0], bar_boundaries[-1])
    axes.legend(title="Rank", loc="upper right")
    figure.tight_layout()
    return figure


figure = build_figure()
