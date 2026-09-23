# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Window rank against window length (Charney-DeVore)
=====================================================

How much of the dominant span a window of the Charney-DeVore trajectory sees
depends on how long the window is. The upper panel splits the windows at each
length by signature rank; the lower panel takes the share of those windows
whose span is exactly the dominant span, the rank-2 span of the three most
frequent classes.

Short windows see nothing: a window has to be long enough to contain a whole
recurrence before it carries any class at all. From a few hundred time units
on, rank 2 is the most common reading, and around fourteen hundred time units
three windows in four carry exactly the dominant span. Longer windows than that
pick up the rare classes as well, which raises the rank above 2 without adding
anything the cycles concentrate on.

Windows are swept in detection points, and each length is drawn at the median
time its windows span.
"""

# %%
# Load the storage for cube side 0.005 and the detection points it indexes,
# both from the published example data, fetched and cached on first use. The
# detection points contribute only their times, which turn a window length in
# detection points into a duration.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
STORAGE = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
TIMES = _support.charney_devore_detection(BOXSIZE).times

# %%
# Take the dominant span and build the subspace its classes span, which is
# what the lower panel compares each window's span against. Both live in this
# cover's generator basis, which is the only basis they mean anything in.

SPAN = _support.dominant_span(STORAGE)
DOMINANT_SUBSPACE = cs.Subspace([STORAGE.classes()[class_id] for class_id in SPAN.class_ids])

# %%
# Sweep over window lengths and tally what each window reads. ``LENGTH_STEP``
# and ``SCAN_STEP`` are counted in detection points, as is ``MAX_LENGTH``,
# which runs to several times the longest cycle the storage holds: the
# interesting behavior is at windows long enough to hold many recurrences, not
# just one.

LENGTH_STEP = 1_000
SCAN_STEP = 2_000
MAX_LENGTH = 100_000

extent_start, extent_stop = STORAGE.extent()
window_lengths = list(range(LENGTH_STEP, MAX_LENGTH + 1, LENGTH_STEP))

rank_shares: list[dict[int, float]] = []
dominant_shares: list[float] = []
for length in window_lengths:
    counter: Counter[int] = Counter()
    dominant_windows = 0
    window_count = 0
    for window_start in range(extent_start, extent_stop - length + 1, SCAN_STEP):
        signature = STORAGE.signature(range(window_start, window_start + length))
        counter[signature.rank()] += 1
        dominant_windows += signature.span() == DOMINANT_SUBSPACE
        window_count += 1
    rank_shares.append({rank: count / window_count for rank, count in counter.items()})
    dominant_shares.append(dominant_windows / window_count)

all_ranks = sorted({rank for shares in rank_shares for rank in shares})

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
    duration_values.append(float(np.median(TIMES[starts + length - 1] - TIMES[starts])))
median_durations = np.array(duration_values)

bar_boundaries = np.empty(len(median_durations) + 1)
bar_boundaries[1:-1] = (median_durations[:-1] + median_durations[1:]) / 2
bar_boundaries[0] = 2 * median_durations[0] - bar_boundaries[1]
bar_boundaries[-1] = 2 * median_durations[-1] - bar_boundaries[-2]
bar_widths = np.diff(bar_boundaries)

# %%
# Report the length at which the dominant span is the whole reading most often.

peak = int(np.argmax(dominant_shares))
print(
    f"the span is exactly the dominant span for {100 * dominant_shares[peak]:.1f}% of windows "
    f"at {median_durations[peak]:.0f} time units, the highest share of any length"
)

# %%
# Draw the two panels on a shared time axis. Viridis colors the rank stack,
# dark purple at rank 0 through to yellow at the highest rank present.


def build_figure() -> plt.Figure:
    """Return the rank-frequency and dominant-span figure."""
    colormap = plt.get_cmap("viridis")
    rank_colors = {
        rank: colormap(position / max(len(all_ranks) - 1, 1))
        for position, rank in enumerate(all_ranks)
    }

    figure, (rank_axes, share_axes) = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        height_ratios=(2, 1),
    )

    bar_bottoms = [0.0] * len(window_lengths)
    for rank in all_ranks:
        heights = [shares.get(rank, 0.0) for shares in rank_shares]
        rank_axes.bar(
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
    rank_axes.set_ylim(0, 1)
    rank_axes.set_ylabel("share of windows")
    # The stack fills its axes, so the rank legend sits beside them.
    rank_axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    share_axes.plot(
        median_durations,
        dominant_shares,
        color=_support.signature_colors()[0],
        linewidth=2,
    )
    share_axes.set_ylim(0, 1)
    share_axes.set_xlabel("window length (time units)")
    share_axes.set_ylabel("share with exactly the dominant span")

    rank_axes.set_xlim(bar_boundaries[0], bar_boundaries[-1])
    figure.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    return figure


figure = build_figure()
