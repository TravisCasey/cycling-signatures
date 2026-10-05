# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Trajectory length and the cover (Charney-DeVore)
===================================================

A cover reads the holes of the region the trajectory has swept out, so a cover
built from too little trajectory has holes the motion does not: stretches of
the attractor visited too rarely to be filled in. The four storages here hold
the same cycles over the same extent, at cube side 0.005, and differ only in
how many raw rows their cover was built from: the first 10 M, the first 20 M
(on the grid the coordinates arrive on and on the turned grid), and all 60 M.

Each panel orders one storage's classes by frequency and colors each class by
the class of the 60 M-row cover its cycles carry there, which the shared cycles
make well defined. The shaded prefixes are the closed ones. At 10 M rows each
of the three classes is split into dozens of rarer ones; at 20 M rows the
classes close into a rank-3 span, one of whose classes is trivial on the
60 M-row cover. The turned grid at 20 M rows sends that same class to zero.
The printed output gives the closed prefixes, how many classes each of the
three is split into, and the check that the two covers remove the same class.
"""

# %%
# Load the four storages from the published example data, fetched and cached
# on first use. All four index the same detection points, so no positions or
# times are needed here.

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
LEADING = 16
GRAY = (0.72, 0.72, 0.72)
WHITE = (1.0, 1.0, 1.0)

REFERENCE = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
SPAN = _support.dominant_span(REFERENCE)
COLORS = dict(zip(SPAN.class_ids, _support.signature_colors()[: len(SPAN.class_ids)], strict=True))

SHORT = {
    "the first 10 M raw rows": cs.CycleStorage.load(
        _support.charney_devore_storage(BOXSIZE, raw_rows=10_000_000)
    ),
    "the first 20 M raw rows": cs.CycleStorage.load(
        _support.charney_devore_storage(BOXSIZE, raw_rows=20_000_000)
    ),
    "the first 20 M raw rows, turned grid": cs.CycleStorage.load(
        _support.charney_devore_storage(BOXSIZE, "rotation", raw_rows=20_000_000)
    ),
}
STORAGES = {**SHORT, "all 60 M raw rows": REFERENCE}

# %%
# Two storages over the same detection points with the same cap hold the same
# components whatever cover they were built against, so a class of one is
# carried to the other component by component. Weighting by cycle count asks
# which class of the target the cycles of each source class carry there.


def images(source: cs.CycleStorage, target: cs.CycleStorage) -> dict[int, Counter[int]]:
    """Return, per class of `source`, its cycles counted by `target` class.

    Raises
    ------
    RuntimeError
        If the two storages do not hold the same components over the same
        ranges of detection points.
    """
    source_components = source.components()
    target_components = target.components()
    if len(source_components) != len(target_components) or any(
        first.shortest_cycle().range() != second.shortest_cycle().range()
        for first, second in zip(source_components, target_components, strict=True)
    ):
        raise RuntimeError("the two storages do not hold the same components")
    tallies: dict[int, Counter[int]] = defaultdict(Counter)
    for first, second in zip(source_components, target_components, strict=True):
        tallies[first.class_id()][second.class_id()] += first.cycle_count()
    return tallies


def image_of(tallies: dict[int, Counter[int]]) -> dict[int, int]:
    """Return the target class carrying most of each source class's cycles."""
    return {class_id: tally.most_common(1)[0][0] for class_id, tally in tallies.items()}


ORDERS = {label: _support.frequency_order(storage) for label, storage in STORAGES.items()}
TO_REFERENCE = {label: images(storage, REFERENCE) for label, storage in SHORT.items()}

# %%
# Report each storage's closed prefixes, how purely its classes land on the
# 60 M-row cover's, and how many of its classes land on each of the three.

for label, storage in STORAGES.items():
    print(f"{label}: {len(ORDERS[label])} nonzero classes")
    for prefix in _support.closed_prefixes(storage, ORDERS[label]):
        drop = "" if prefix.drop is None else f", drop {prefix.drop:.1f}x"
        print(
            f"  closed prefix of {prefix.size} classes at rank {prefix.rank}, "
            f"share {100 * prefix.share:.1f}%{drop}"
        )
    if label in TO_REFERENCE:
        tallies = TO_REFERENCE[label]
        purity = min(tally.most_common(1)[0][1] / tally.total() for tally in tallies.values())
        landing = Counter(image_of(tallies).values())
        split = ", ".join(str(landing[class_id]) for class_id in SPAN.class_ids)
        print(f"  {100 * purity:.1f}% of each class's cycles land on one class of the 60 M rows")
        print(f"  classes landing on the three of the 60 M rows: {split}")

# %%
# The 20 M-row cover's rank-3 span loses one direction both to the
# longer trajectory and to the turned grid. Map the span's seven classes each
# way, check the map respects their relations, and compare what goes to zero.


def kernel(source: cs.CycleStorage, target: cs.CycleStorage) -> tuple[list[int], bool]:
    """Return the span positions that go to zero, and whether relations hold.

    The span is `source`'s dominant span, mapped into `target` class by class.
    """
    span_ids = _support.dominant_span(source).class_ids
    source_classes = source.classes()
    target_classes = target.classes()
    image = image_of(images(source, target))
    mapped = {source_classes[class_id]: target_classes[image[class_id]] for class_id in span_ids}
    relations_hold = all(
        mapped[first] ^ mapped[second] == mapped[first ^ second]
        for first in mapped
        for second in mapped
        if first != second
    )
    zeros = [
        position
        for position, class_id in enumerate(span_ids, 1)
        if mapped[source_classes[class_id]].is_zero()
    ]
    return zeros, relations_hold


TWENTY = SHORT["the first 20 M raw rows"]
LONGER = kernel(TWENTY, REFERENCE)
TURNED = kernel(TWENTY, SHORT["the first 20 M raw rows, turned grid"])
for name, (zeros, relations_hold) in (("all 60 M raw rows", LONGER), ("the turned grid", TURNED)):
    verdict = "relations hold" if relations_hold else "relations fail"
    print(f"20 M-row span onto {name}: positions {zeros} go to zero, {verdict}")
print(f"the same class goes to zero both ways: {LONGER[0] == TURNED[0]}")

# %%
# Draw one panel per storage: its leading classes as shares of the cycles
# carrying a nonzero class, colored by their class on the 60 M-row cover, with
# the closed prefixes shaded.


def bar_colors(label: str) -> list[tuple[float, float, float]]:
    """Return the color of each leading class of storage `label`."""
    order = ORDERS[label][:LEADING]
    if label not in TO_REFERENCE:
        return [COLORS.get(class_id, GRAY) for class_id, _ in order]
    image = image_of(TO_REFERENCE[label])
    reference_classes = REFERENCE.classes()
    return [
        WHITE if reference_classes[image[class_id]].is_zero() else COLORS.get(image[class_id], GRAY)
        for class_id, _ in order
    ]


def build_figure() -> plt.Figure:
    """Return the four frequency panels."""
    figure, panels = plt.subplots(len(STORAGES), 1, figsize=(11, 12), sharex=True)
    for axes, (label, storage) in zip(panels, STORAGES.items(), strict=True):
        order = ORDERS[label]
        total = sum(count for _, count in order)
        shares = [100 * count / total for _, count in order[:LEADING]]
        positions = np.arange(1, len(shares) + 1)
        axes.bar(
            positions,
            shares,
            color=bar_colors(label),
            edgecolor="0.35",
            linewidth=0.5,
            width=0.6,
        )
        top = 1.15 * max(shares)
        for prefix in _support.closed_prefixes(storage, order):
            axes.axvspan(0.5, prefix.size + 0.5, color="0.93", zorder=0)
            drop = "" if prefix.drop is None else f", drop {prefix.drop:.1f}x"
            axes.text(
                prefix.size + 0.7,
                0.9 * top,
                f"closed: rank {prefix.rank}, share {100 * prefix.share:.0f}%{drop}",
                va="top",
                fontsize=9,
            )
        axes.set_ylim(0, top)
        axes.set_xlim(0.5, LEADING + 0.5)
        axes.set_title(f"cover from {label}", loc="left", fontsize=10)
        axes.set_ylabel("share of cycles (%)")
    panels[-1].set_xticks(np.arange(1, LEADING + 1))
    panels[-1].set_xlabel("frequency position")

    handles = [
        Patch(facecolor=COLORS[class_id], label=f"class {position} of the 60 M-row cover")
        for position, class_id in enumerate(SPAN.class_ids, 1)
    ]
    handles.append(Patch(facecolor=WHITE, edgecolor="0.35", label="trivial on the 60 M-row cover"))
    handles.append(Patch(facecolor=GRAY, label="any other class"))
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


figure = build_figure()

# %%
# Every class of a shorter cover lands on a single class of the 60 M-row
# cover, so more trajectory only merges classes and sends some to zero; it
# never splits one. At 10 M rows the merging is wholesale and the only closed
# prefix is a remnant. At 20 M rows the frequent classes already close, but at
# rank 3, and the extra direction is one that the turned grid and the longer
# trajectory both remove. A storage over a short extent reads its cover's
# structure, so the length that matters is the length the cover was built
# from.
