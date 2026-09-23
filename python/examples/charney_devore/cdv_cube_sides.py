# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""The same classes at three cube sides (Charney-DeVore)
========================================================

Covering the trajectory at a coarser or a finer cube side is a bigger change
than moving the lattice: the cubes are different, the cover's generators are
different, and detection itself runs against a different set of cubes, so the
recurrences it finds are not the same recurrences. The dominant span survives
it anyway, and this example matches the three covers up recurrence by
recurrence to show how.

Each component is taken through its shortest cycle, reduced to the stretch of
integration time that cycle occupies. A component of the cover at cube side
0.005 is matched to the component of another cover whose stretch overlaps it
most, as a fraction of the time the two stretches cover between them, and only
when that fraction reaches half. The printed output reports how many components
find a match each way; the heatmaps send the dominant classes at 0.005, on the
rows, to the classes of the other cover, on the columns, in that cover's own
frequency order. A cell is the share of the row class's matched cycles whose
matched component carries the column class, and white cells carry no cycles.
"""

# %%
# Load the three storages and the detection times each one indexes, from the
# published example data, fetched and cached on first use. Each cube side has
# its own detection points, which is why the match runs on time rather than on
# point indices: a point index means something different at each cube side,
# while integration time is the one coordinate the three share.

from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import _support
import cycling_signatures as cs

REFERENCE_SIDE = 0.005
OTHER_SIDES = (0.004, 0.006)

# A match keeps a pair whose stretches share at least this much of the time
# they cover between them.
MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class CoverReading:
    """What one cover's storage says, reduced to arrays over its components.

    `starts` and `stops` bound each component's shortest cycle in integration
    time, `class_ids` and `cycle_counts` carry that component's class and how
    many cycles it holds, `prefixes` holds the prefixes of the frequency order
    that are closed under addition, and `span` is the cover's dominant span.
    """

    starts: NDArray[np.float64]
    stops: NDArray[np.float64]
    class_ids: NDArray[np.int64]
    cycle_counts: NDArray[np.int64]
    prefixes: list[_support.ClosedPrefix]
    span: _support.DominantSpan


def read_cover(boxsize: float) -> CoverReading:
    """Return the reading of the published cover at cube side `boxsize`."""
    storage = cs.CycleStorage.load(_support.charney_devore_storage(boxsize))
    times = _support.charney_devore_detection(boxsize).times
    components = storage.components()
    ranges = np.array([component.shortest_cycle().range() for component in components])
    order = _support.frequency_order(storage)
    return CoverReading(
        starts=times[ranges[:, 0]],
        stops=times[ranges[:, 1] - 1],
        class_ids=np.array([component.class_id() for component in components]),
        cycle_counts=np.array([component.cycle_count() for component in components]),
        prefixes=_support.closed_prefixes(storage, order),
        span=_support.dominant_span(storage),
    )


REFERENCE = read_cover(REFERENCE_SIDE)
OTHER_READINGS = {boxsize: read_cover(boxsize) for boxsize in OTHER_SIDES}

# %%
# Before matching anything, ask each cover what its own frequency order says.
# The dominant span is the closed prefix with the largest drop, so it is only
# unambiguous where one prefix closes.

for side, reading in [(REFERENCE_SIDE, REFERENCE), *OTHER_READINGS.items()]:
    closed = ", ".join(
        f"{prefix.size} classes at rank {prefix.rank}" for prefix in reading.prefixes
    )
    count = len(reading.prefixes)
    print(f"cube side {side}: {count} closed prefix{'es' if count != 1 else ''} ({closed})")

# %%
# Match each component of one cover to a component of another by the overlap of
# their stretches. Sorting the candidates by start time bounds the search: a
# candidate that can overlap a given stretch starts after that stretch's start
# less the longest candidate, and before that stretch's end.


def matched_components(reading: CoverReading, other: CoverReading) -> NDArray[np.int64]:
    """Return the component of `other` each component of `reading` matches.

    Entries index `other`'s components, or are -1 where no candidate overlaps
    enough to count as the same recurrence.
    """
    by_start = np.argsort(other.starts)
    candidate_starts = other.starts[by_start]
    candidate_stops = other.stops[by_start]
    longest = float((other.stops - other.starts).max())

    first = np.searchsorted(candidate_starts, reading.starts - longest)
    last = np.searchsorted(candidate_starts, reading.stops)
    matches = np.full(len(reading.starts), -1, dtype=np.int64)
    for index, (begin, end) in enumerate(zip(first, last, strict=True)):
        if begin >= end:
            continue
        shared = np.minimum(reading.stops[index], candidate_stops[begin:end]) - np.maximum(
            reading.starts[index], candidate_starts[begin:end]
        )
        covered = np.maximum(reading.stops[index], candidate_stops[begin:end]) - np.minimum(
            reading.starts[index], candidate_starts[begin:end]
        )
        overlap = np.where(shared > 0, shared / covered, 0.0)
        best = int(np.argmax(overlap))
        if overlap[best] >= MIN_OVERLAP:
            matches[index] = by_start[begin + best]
    return matches


# %%
# For each of the other two cube sides, match both ways and tally where the
# dominant classes at 0.005 land. Weighting by cycle count asks what became of
# the cycles, not just of the components.


def image_shares(other: CoverReading) -> tuple[np.ma.MaskedArray, list[str], float, float]:
    """Return the image matrix, its column labels and the matched fractions.

    Rows are the dominant classes at cube side 0.005 in frequency order and
    columns are `other`'s dominant classes in its own frequency order, followed
    by one column for everything else. Entries are shares of a row's matched
    cycles, masked where a cell holds none.
    """
    forward = matched_components(REFERENCE, other)
    backward = matched_components(other, REFERENCE)

    images: dict[int, Counter[int]] = {class_id: Counter() for class_id in REFERENCE.span.class_ids}
    for index in np.flatnonzero(forward >= 0):
        tally = images.get(int(REFERENCE.class_ids[index]))
        if tally is not None:
            tally[int(other.class_ids[forward[index]])] += int(REFERENCE.cycle_counts[index])

    columns = list(other.span.class_ids)
    matrix = np.zeros((len(REFERENCE.span.class_ids), len(columns) + 1))
    for row, class_id in enumerate(REFERENCE.span.class_ids):
        tally = images[class_id]
        total = sum(tally.values())
        named = 0
        for column, other_class_id in enumerate(columns):
            named += tally[other_class_id]
            matrix[row, column] = tally[other_class_id] / total
        matrix[row, -1] = (total - named) / total

    labels = [f"class {position}" for position in range(1, len(columns) + 1)] + ["other"]
    return (
        np.ma.masked_equal(matrix, 0.0),
        labels,
        float(np.count_nonzero(forward >= 0)) / len(forward),
        float(np.count_nonzero(backward >= 0)) / len(backward),
    )


RESULTS = {boxsize: image_shares(reading) for boxsize, reading in OTHER_READINGS.items()}

# %%
# Report how many components find a match each way, as a share of the cover
# they are matched from.

for boxsize in OTHER_SIDES:
    _, _, forward_share, backward_share = RESULTS[boxsize]
    print(
        f"cube side {boxsize}: {100 * forward_share:.1f}% of the components at "
        f"{REFERENCE_SIDE} matched, {100 * backward_share:.1f}% of its own matched back"
    )

# %%
# Draw one heatmap per other cube side, on the shared purity scale.


def build_figure() -> plt.Figure:
    """Return the image heatmaps of the two other cube sides."""
    figure, panels = plt.subplots(1, len(OTHER_SIDES), figsize=(14, 6), squeeze=False)
    row_labels = [f"class {position}" for position in range(1, len(REFERENCE.span.class_ids) + 1)]
    image = None
    for panel, boxsize in zip(panels[0], OTHER_SIDES, strict=True):
        matrix, column_labels, _, _ = RESULTS[boxsize]
        image = _support.purity_heatmap(panel, matrix, row_labels, column_labels)
        panel.set_xlabel(f"class at cube side {boxsize}")
        panel.set_title(f"cube side {boxsize}")
    for panel in panels[0][1:]:
        panel.tick_params(labelleft=False)
    panels[0][0].set_ylabel(f"class at cube side {REFERENCE_SIDE}")
    if image is not None:
        colorbar = figure.colorbar(image, ax=panels[0].tolist(), fraction=0.046, pad=0.04)
        colorbar.set_label("share of the row class's matched cycles")
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return figure


figure = build_figure()

# %%
# The three classes pair up one for one at both other cube sides, and the
# leftover column takes only a small part of each row. Coarsening or refining
# the cubes changes which recurrences are detected and how many of them there
# are, and it changes the cover's generators outright, so nothing about this
# correspondence is forced by construction: the same two independent loop types
# and their sum are what the trajectory recurs on, read three ways.
