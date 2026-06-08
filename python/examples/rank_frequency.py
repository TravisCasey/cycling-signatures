# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Rank-frequency histogram
===========================

Frequency of appearance of each signature rank as a function of window length.
For the Lorenz attractor, longer windows capture more of the two independent
cycles, so rank-2 windows become dominant as the length grows. This figure makes
that transition visible.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the committed fixture. No raw
# trajectory data is needed; the storage already encodes the homological
# information.

from collections import Counter

import matplotlib.pyplot as plt

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.lorenz_path("storage.cyc"))

# %%
# Sweep over window lengths and tally rank occurrences. For each length the
# window slides across the full storage extent and each
# ``signature(...).rank()`` call queries the number of independent cycles this
# window contains.

LENGTH_STEP = 5
SCAN_STEP = 25

extent_start, extent_stop = STORAGE.extent()
extent_length = extent_stop - extent_start
max_length = min(STORAGE.max_length(), extent_length)

window_lengths: list[int] = list(range(LENGTH_STEP, max_length + 1, LENGTH_STEP))

rank_counts: list[Counter[int]] = []
for length in window_lengths:
    counter: Counter[int] = Counter()
    for window_start in range(extent_start, extent_stop - length, SCAN_STEP):
        rank = STORAGE.signature(range(window_start, window_start + length)).rank()
        counter[rank] += 1
    rank_counts.append(counter)

all_ranks = sorted({rank for counter in rank_counts for rank in counter})

# %%
# Build a stacked bar chart. Each bar represents one window length; the stacked
# segments show how many windows produced each rank. Viridis maps low ranks
# (cold, near-zero) to high ranks (warm).


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
            window_lengths,
            heights,
            bottom=bar_bottoms,
            width=LENGTH_STEP * 0.85,
            color=rank_colors[rank],
            label=f"rank {rank}",
        )
        bar_bottoms = [bottom + height for bottom, height in zip(bar_bottoms, heights, strict=True)]

    axes.set_xlabel("Window length (samples)")
    axes.set_ylabel("Number of windows")
    axes.set_title("Signature rank frequency vs. window length (Lorenz)")
    axes.legend(title="Rank", loc="upper right")
    figure.tight_layout()
    return figure


figure = build_figure()
