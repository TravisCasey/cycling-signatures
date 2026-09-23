# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Representatives of the three classes (Charney-DeVore)
========================================================

The shortest cycle carrying each class of the dominant span, drawn in the two
planes the model's coordinates pair into. The zonal plane is coordinates 1 and
4, the amplitudes of the two zonal-flow modes; the wave plane is coordinates 2
and 3, the cosine and sine amplitudes of the leading wave, so a turn around it
is that wave's phase advancing once.

Each column is one class, named by its position in the frequency order, and
each representative closes on itself: its two ends lie within one cube side of
each other, which is what admits it as a cycle. The printed output gives the
time each one takes. The faint backdrop is the detection points themselves,
which is where all of this motion lives.
"""

# %%
# Load the storage for cube side 0.005 and the detection points it indexes,
# both from the published example data, fetched and cached on first use. A
# cycle's point range indexes the detection points directly, so a range picks
# out positions to draw and times to measure the cycle's duration with.

import matplotlib.pyplot as plt

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
STORAGE = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
DETECTION = _support.charney_devore_detection(BOXSIZE)
POSITIONS = DETECTION.positions
TIMES = DETECTION.times

# The two coordinate pairs, as indices into a detection point.
ZONAL = (0, 3)
WAVE = (1, 2)

# Detection points between backdrop points are a small fraction of a cube
# apart, so a coarse stride still fills the attractor's outline.
BACKDROP_STEP = 150

# %%
# Take the dominant span and, for each of its classes, the shortest cycle of
# any component carrying that class. A component's own shortest cycle is the
# tightest recurrence it holds, and the shortest of those over a class is the
# least motion this trajectory needs to wrap that class.

SPAN = _support.dominant_span(STORAGE)
COLORS = dict(zip(SPAN.class_ids, _support.signature_colors()[: len(SPAN.class_ids)], strict=True))

shortest_cycles: dict[int, cs.Cycle] = {}
for component in STORAGE.components():
    class_id = component.class_id()
    if class_id not in COLORS:
        continue
    candidate = component.shortest_cycle()
    held = shortest_cycles.get(class_id)
    if held is None or candidate.length() < held.length():
        shortest_cycles[class_id] = candidate

# %%
# Report how long each representative takes, as the integration time between
# the first and last detection point of its range.

for position, class_id in enumerate(SPAN.class_ids, start=1):
    first_point, last_point = shortest_cycles[class_id].range()
    elapsed = float(TIMES[last_point - 1] - TIMES[first_point])
    print(f"class {position}: shortest representative takes {elapsed:.1f} time units")

# %%
# Draw one column per class and one row per plane. The two panels of a column
# are the same cycle seen through two coordinate pairs, and the panels of a row
# share axes so the three representatives are directly comparable.


def build_figure() -> plt.Figure:
    """Return the representative-cycle panels."""
    planes = [(ZONAL, ("x1", "x4")), (WAVE, ("x2", "x3"))]
    figure, panels = plt.subplots(
        len(planes),
        len(SPAN.class_ids),
        figsize=(15, 10),
        sharex="row",
        sharey="row",
        squeeze=False,
    )
    backdrop = POSITIONS[::BACKDROP_STEP]
    for row, (plane, axis_names) in enumerate(planes):
        first, second = plane
        for column, class_id in enumerate(SPAN.class_ids):
            panel = panels[row][column]
            panel.scatter(
                backdrop[:, first],
                backdrop[:, second],
                color=(0.76, 0.76, 0.76),
                s=2,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
            )
            start, stop = shortest_cycles[class_id].range()
            points = POSITIONS[start:stop]
            panel.plot(
                points[:, first],
                points[:, second],
                color=COLORS[class_id],
                linewidth=1.4,
            )
            panel.plot(
                points[0, first],
                points[0, second],
                "o",
                color=COLORS[class_id],
                markersize=5,
            )
            panel.set_xlabel(axis_names[0])
            panel.set_ylabel(axis_names[1])
    for column in range(len(SPAN.class_ids)):
        panels[0][column].set_title(f"class {column + 1}")
    figure.tight_layout()
    return figure


figure = build_figure()

# %%
# All three representatives take the wave's phase once around the wave plane,
# and they differ in what else they do. Classes 1 and 2 double back through the
# wave plane's interior and rise twice in the zonal plane before running down
# to close; class 3 stays on one turn of the wave plane and makes a single
# zonal rise with a shoulder on the way down. That is the shape of the two
# independent loop types and their sum, and it is why the three are hard to
# tell apart by eye: what separates them sharply is how often they occur, which
# *Class frequency and the dominant span* shows, and what survives a rebuilt
# cover, which *The signature under a rigid motion of the grid* shows.
