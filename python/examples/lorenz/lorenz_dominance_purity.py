# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Dominance and purity maps
===========================

This example assigns every query point on the Lorenz attractor a *dominant*
signature: the library signature most common among the point's spatial
neighbors. It also measures the assigned signature's *purity*: the fraction of
neighbors that share that signature.

The first figure colors the attractor by dominant signature. The two wings of
the Lorenz butterfly are each dominated by a single-wing rank-1 signature (the
same colors as the coverage barcode and signature-indicator examples), while
the transition region near the cross-wing bridge shows mixed or rank-2
dominance. The second figure breaks the same query points into one purity map
per signature: each shades the attractor by that signature's local share, dark
red where nearly all neighbors carry it and gray where few do. Reading the
purity maps together shows how the signatures partition the attractor and where
they overlap.
"""

# %%
# Load the raw trajectory and the prebuilt ``CycleStorage`` from the published
# example data, fetched and cached on first use. The raw positions are in
# native Lorenz coordinates.

import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

import _support
import cycling_signatures as cs

RAW = np.load(_support.lorenz_raw())
STORAGE = cs.CycleStorage.load(_support.lorenz_storage())

# %%
# Constants that control the analysis. ``WINDOW_LENGTH`` is the number of
# trajectory samples in each signature query. ``EPSILON`` is the neighborhood
# radius in native Lorenz coordinates. ``MIN_NEIGHBORS`` is the minimum count
# of labeled neighbors to trust a dominance reading; points with fewer are
# skipped. ``QUERY_STEP`` and ``BACKGROUND_STEP`` downsample the scatter
# for a legible figure. ``LIBRARY_SIZE`` caps the number of distinct
# signatures shown.

WINDOW_LENGTH = 230
EPSILON = 3.0
MIN_NEIGHBORS = 20
QUERY_STEP = 1
BACKGROUND_STEP = 40
LIBRARY_SIZE = 4
LIBRARY_SCAN_STEP = 25

# %%
# **Canonical class colors.** Rank-1 signatures each span a single homology
# class and take that class's color, shared with the coverage-barcode and
# signature-indicator examples. Higher-rank signatures take distinct palette
# colors beyond the class colors.

class_objects = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in class_objects]
CLASS_COLORS = _support.class_color_map(class_keys)
nonzero_classes = [
    (key, hclass) for key, hclass in zip(class_keys, class_objects, strict=True) if any(key)
]


def signature_color_and_label(
    subspace: cs.Subspace,
    higher_rank_colors: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], str]:
    """Return the color and legend label for one non-trivial signature."""
    if subspace.rank() == 1:
        key = next(key for key, hclass in nonzero_classes if subspace.contains(hclass))
        return CLASS_COLORS[key], f"[{' '.join(map(str, key))}] (rank 1)"
    return higher_rank_colors.pop(0), f"rank {subspace.rank()}"


# %%
# **Build a signature library.** Slide a window across the storage extent in
# coarse steps and tally the distinct non-trivial signatures. Keep the
# ``LIBRARY_SIZE`` most common, ordered by descending frequency so the legend
# lists the most common signatures first.

extent_start, extent_stop = STORAGE.extent()

frequency: Counter[cs.Subspace] = Counter()
for window_start in range(extent_start, extent_stop - WINDOW_LENGTH + 1, LIBRARY_SCAN_STEP):
    subspace = STORAGE.signature(range(window_start, window_start + WINDOW_LENGTH)).span()
    if subspace.rank() != 0:
        frequency[subspace] += 1

library = [subspace for subspace, _ in frequency.most_common(LIBRARY_SIZE)]

# %%
# **Label every sample.** For each sample in the labeled prefix (the range
# where a full window fits), query ``STORAGE.signature()`` and look it up in
# the library: the value is the library position, or -1 for a signature that
# is trivial or absent from the library.

library_index = {subspace: index for index, subspace in enumerate(library)}
labeled_stop = extent_stop - WINDOW_LENGTH
labeled_samples = np.arange(extent_start, labeled_stop)
labeled_values = np.array(
    [
        library_index.get(STORAGE.signature(range(sample, sample + WINDOW_LENGTH)).span(), -1)
        for sample in labeled_samples
    ],
    dtype=np.int32,
)

# %%
# **Compute neighborhood dominance and purity.** Build one KD-tree per library
# signature, holding the labeled samples that carry it. For each query sample
# (stepped by ``QUERY_STEP`` over the labeled prefix), count each signature's
# labeled samples within ``EPSILON``. The dominant signature is the most
# common, and purity is the per-signature fraction of the labeled neighbors.
# Query points with fewer than ``MIN_NEIGHBORS`` labeled neighbors are
# dropped.

member_trees = [
    KDTree(RAW[labeled_samples[labeled_values == position]]) for position in range(len(library))
]

query_points = RAW[np.arange(extent_start, labeled_stop, QUERY_STEP)]
label_counts = np.stack(
    [
        member_tree.query_ball_point(query_points, EPSILON, return_length=True, workers=-1)
        for member_tree in member_trees
    ],
    axis=1,
)
labeled_neighbor_totals = label_counts.sum(axis=1)

kept = labeled_neighbor_totals >= MIN_NEIGHBORS
positions_array = query_points[kept]
dominant_array = label_counts[kept].argmax(axis=1)
purity_array = label_counts[kept] / labeled_neighbor_totals[kept, np.newaxis]

# %%
# **Assign colors and labels to the library.** Each library signature gets its
# color and label via the same scheme as ``lorenz_signature_indicator.py``.

higher_rank_colors = _support.signature_colors()[len(nonzero_classes) :]
library_colors: list[tuple[float, float, float]] = []
library_labels: list[str] = []
for subspace in library:
    color, label = signature_color_and_label(subspace, higher_rank_colors)
    library_colors.append(color)
    library_labels.append(label)

purity_cmap = _support.purity_colormap()


# %%
# **Dominant-signature map.** A 3-D scatter of the attractor, each query point
# colored by its dominant signature over a faint gray backdrop for spatial
# context. The legend annotates each signature's rank.


def build_dominant_figure() -> plt.Figure:
    """Return the dominant-signature scatter."""
    figure = plt.figure(figsize=(14, 11))
    axes = figure.add_subplot(projection="3d")
    background = RAW[::BACKGROUND_STEP]
    axes.scatter(
        background[:, 0],
        background[:, 1],
        background[:, 2],
        color=(0.82, 0.82, 0.82),
        s=1,
        alpha=0.2,
        linewidths=0,
        rasterized=True,
    )
    for library_index in range(len(library)):
        mask = dominant_array == library_index
        if not np.any(mask):
            continue
        axes.scatter(
            positions_array[mask, 0],
            positions_array[mask, 1],
            positions_array[mask, 2],
            color=library_colors[library_index],
            s=2,
            alpha=0.6,
            linewidths=0,
            label=library_labels[library_index],
            rasterized=True,
        )
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("Dominant signature: Lorenz attractor")
    axes.view_init(elev=25, azim=-75)
    axes.legend(
        title="Dominant signature",
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        markerscale=3,
    )
    return figure


dominant_figure = build_dominant_figure()

# %%
# **Per-signature purity.** One x-z map per library signature, arranged in a
# grid. Every query point is shaded by that signature's local share via the
# gray-to-dark-red colormap, drawn in ascending purity order so the prevalent
# regions sit on top. A shared colorbar spans the grid.


def build_purity_figure() -> plt.Figure:
    """Return the per-signature purity maps."""
    num_library = len(library)
    purity_columns = 2
    purity_rows = math.ceil(num_library / purity_columns)

    figure, panel_grid = plt.subplots(
        purity_rows,
        purity_columns,
        figsize=(11, 9),
        squeeze=False,
    )
    panels = panel_grid.ravel()
    purity_scatter = None
    for library_index in range(num_library):
        panel = panels[library_index]
        signature_purity = purity_array[:, library_index]
        draw_order = np.argsort(signature_purity)
        purity_scatter = panel.scatter(
            positions_array[draw_order, 0],
            positions_array[draw_order, 2],
            c=signature_purity[draw_order],
            cmap=purity_cmap,
            vmin=0.0,
            vmax=1.0,
            s=4,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )
        panel.set_xlabel("x")
        panel.set_ylabel("z")
        panel.set_title(library_labels[library_index], fontsize=10)
    for extra_index in range(num_library, len(panels)):
        panels[extra_index].set_axis_off()

    if purity_scatter is not None:
        colorbar = figure.colorbar(
            purity_scatter,
            ax=panels.tolist(),
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("Purity (fraction of neighbors)")
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    figure.suptitle("Neighborhood purity per signature: Lorenz attractor")
    return figure


purity_figure = build_purity_figure()
