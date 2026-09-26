# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""The signature under a rigid motion of the grid (Charney-DeVore)
==================================================================

The cubical cover is built on a lattice of cubes laid over the trajectory, and
where that lattice sits is an arbitrary choice. Turning it by a rotation of the
six-dimensional state space, or shifting it half a cube along every axis,
leaves the trajectory and its detection points exactly as they were and
rebuilds the cover from scratch: different cubes, a different generator count,
a different generator basis. If the dominant span is a property of the motion
rather than of the lattice, each of its classes should come back as one class
of the moved cover.

A rigid motion leaves the metric untouched, so the components are the same
components, and this example checks that before reading anything off them: the
two moved covers must hold the same number of components, each closing over
the same range of detection points. That is what makes the classes comparable
at all, since the two covers share no generator basis and their class vectors
cannot be compared entry by entry.

Each heatmap sends the three dominant classes of the identity grid, the lattice
the coordinates arrive on, on the rows, to the classes of one moved grid, on the
columns, in that grid's own frequency order. A cell is the share of the row
class's cycles whose component carries the column class, so a row that lands in
one column is a class the motion carries over intact. White cells carry no
cycles at all.
The printed output gives how much of each row the heaviest column takes and
whether the three columns the rows land on satisfy the same relation.
"""

# %%
# Load the three storages at cube side 0.005 from the published example data,
# fetched and cached on first use: the grid the coordinates arrive on, the
# turned grid and the shifted grid. All three index the same detection points,
# so no positions or times are needed here.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
PARTNER_GRIDS = ("rotation", "shift")
PARTNER_NAMES = {"rotation": "turned grid", "shift": "half-cube shifted grid"}

IDENTITY = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
IDENTITY_COMPONENTS = IDENTITY.components()
IDENTITY_RANGES = [component.shortest_cycle().range() for component in IDENTITY_COMPONENTS]
SPAN = _support.dominant_span(IDENTITY)

# %%
# For one moved grid, check that it holds the same components and tally where
# each dominant class's cycles land. A component's cycles all carry its class,
# so weighting by cycle count asks which class of the moved cover the cycles
# of this class became.


def image_shares(grid: str) -> tuple[np.ma.MaskedArray, list[str], bool]:
    """Return the image matrix, its column labels, and the relation's verdict.

    Rows are the dominant classes of the grid the coordinates arrive on, in
    frequency order, and columns are `grid`'s dominant classes in its own
    frequency order, followed by one column for everything else. Entries are
    shares of a row's cycles, masked where a cell holds none. The verdict is
    whether the three classes the rows land on satisfy the same relation.

    Raises
    ------
    RuntimeError
        If `grid` does not detect the same components over the same ranges of
        detection points, which a rigid motion of the lattice has to.
    """
    partner = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE, grid))
    components = partner.components()
    if len(components) != len(IDENTITY_COMPONENTS):
        raise RuntimeError(
            f"the {PARTNER_NAMES[grid]} holds {len(components)} components against "
            f"{len(IDENTITY_COMPONENTS)}; moving the lattice must leave the detection "
            f"points, and so the components, untouched"
        )
    for index, (component, point_range) in enumerate(zip(components, IDENTITY_RANGES, strict=True)):
        if component.shortest_cycle().range() != point_range:
            raise RuntimeError(
                f"component {index} of the {PARTNER_NAMES[grid]} closes over a different "
                f"range of detection points than the same component on the grid the "
                f"coordinates arrive on"
            )

    images: dict[int, Counter[int]] = {class_id: Counter() for class_id in SPAN.class_ids}
    for component, partner_component in zip(IDENTITY_COMPONENTS, components, strict=True):
        tally = images.get(component.class_id())
        if tally is not None:
            tally[partner_component.class_id()] += component.cycle_count()

    partner_span = _support.dominant_span(partner)
    columns = list(partner_span.class_ids)
    matrix = np.zeros((len(SPAN.class_ids), len(columns) + 1))
    for row, class_id in enumerate(SPAN.class_ids):
        tally = images[class_id]
        total = sum(tally.values())
        named = 0
        for column, partner_class_id in enumerate(columns):
            named += tally[partner_class_id]
            matrix[row, column] = tally[partner_class_id] / total
        matrix[row, -1] = (total - named) / total

    partner_classes = partner.classes()
    leading = [partner_classes[tally.most_common(1)[0][0]] for tally in images.values()]
    present = set(leading)
    relation_holds = len(present) == len(leading) and all(
        first ^ second in present
        for index, first in enumerate(leading)
        for second in leading[index + 1 :]
    )

    labels = [f"class {position}" for position in range(1, len(columns) + 1)] + ["other"]
    return np.ma.masked_equal(matrix, 0.0), labels, relation_holds


RESULTS = {grid: image_shares(grid) for grid in PARTNER_GRIDS}

# %%
# Report, per moved grid, how much of its worst row the heaviest column takes,
# and whether the three columns the rows land on satisfy the same relation.

for grid in PARTNER_GRIDS:
    matrix, _, relation_holds = RESULTS[grid]
    worst = float(matrix[:, :-1].filled(0.0).max(axis=1).min())
    print(
        f"{PARTNER_NAMES[grid]}: {100 * worst:.1f}% of the cycles kept at worst, "
        + ("relation holds" if relation_holds else "relation fails")
    )

# %%
# Draw one heatmap per moved grid, on the shared purity scale.


def build_figure() -> plt.Figure:
    """Return the image heatmaps of the two moved grids."""
    figure, panels = plt.subplots(1, len(PARTNER_GRIDS), figsize=(14, 5.5), squeeze=False)
    row_labels = [f"class {position}" for position in range(1, len(SPAN.class_ids) + 1)]
    image = None
    for panel, grid in zip(panels[0], PARTNER_GRIDS, strict=True):
        matrix, column_labels, _ = RESULTS[grid]
        image = _support.purity_heatmap(panel, matrix, row_labels, column_labels)
        panel.set_xlabel(f"class on the {PARTNER_NAMES[grid]}")
        panel.set_title(PARTNER_NAMES[grid])
    for panel in panels[0][1:]:
        panel.tick_params(labelleft=False)
    panels[0][0].set_ylabel("class on the identity grid")
    if image is not None:
        colorbar = figure.colorbar(image, ax=panels[0].tolist(), fraction=0.046, pad=0.04)
        colorbar.set_label("share of the row class's cycles")
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return figure


figure = build_figure()

# %%
# Each row lands overwhelmingly on one column, 97.8 % at worst, with the
# remainder scattered among the turned grid's other classes, and the three
# columns it lands on are three different classes of the moved cover
# satisfying the same relation, so the dominant span maps onto the moved
# grid's dominant span class for class.
