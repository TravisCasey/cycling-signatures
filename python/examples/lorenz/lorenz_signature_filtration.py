# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature filtration heatmap
===============================

The cycling signature of a sliding window at every adjacency threshold up
to the top of the stored detection band. Each column fixes one window of
the Lorenz trajectory; climbing the column replays that window's
filtration: white below the window's first generator birth, then the span
of every generator born by the threshold. Colors name the frequent
signatures (shared with the signature indicator); gray marks non-trivial
signatures outside that library. A column whose color locks in well below
the band top carries a signature that is stable across the detection band
rather than an artifact of one threshold choice.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the published example data, fetched
# and cached on first use. ``threshold()`` is the top of the stored detection
# band and bounds every filtration query below.

import math
from bisect import bisect_right
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.lorenz_storage())
BAND_TOP = STORAGE.threshold()
assert math.isfinite(BAND_TOP)

# %%
# **Canonical class colors.** A rank-1 signature is the span of a single
# homology class, so it takes that class's color (shared with the coverage
# barcode and signature-indicator examples). Higher-rank signatures take
# distinct palette colors beyond the class colors.

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
# **Query one filtration per column.** Each column start yields a single
# ``signature`` query; the window's span can change only at its generators'
# births, so recording the span at each distinct birth captures the whole
# column exactly.

WINDOW_LENGTH = 330
TIME_WINDOW_START = 0
TIME_WINDOW_STOP = 6000
COLUMN_STEP = 5
ROW_COUNT = 60

extent_start, extent_stop = STORAGE.extent()
column_starts = list(
    range(
        max(TIME_WINDOW_START, extent_start),
        min(TIME_WINDOW_STOP, extent_stop - WINDOW_LENGTH + 1),
        COLUMN_STEP,
    )
)
row_thresholds = [BAND_TOP * (row + 0.5) / ROW_COUNT for row in range(ROW_COUNT)]

column_filtrations: list[tuple[list[float], list[cs.Subspace]]] = []
for window_start in column_starts:
    signature = STORAGE.signature(range(window_start, window_start + WINDOW_LENGTH))
    births = sorted(set(signature.births()))
    column_filtrations.append((births, [signature.span_at(birth) for birth in births]))

# %%
# **Build a signature library from the band-top spans.** The most frequent
# full-band signatures get the named colors; anything rarer renders gray, so
# unstable or uncommon structure stays visible without a name. The library
# is ordered by rank then descending frequency.

LIBRARY_SIZE = 4

frequency: Counter[cs.Subspace] = Counter(
    spans[-1] for births, spans in column_filtrations if births
)
most_common = frequency.most_common(LIBRARY_SIZE)
ordered = sorted(most_common, key=lambda item: (item[0].rank(), -item[1]))
library = [span for span, _ in ordered]

# %%
# **Build the label array.** Cell values: 0 for the trivial signature
# (white), 1 through ``len(library)`` for library signatures, and one extra
# label (gray) for any non-trivial signature outside the library.

library_index = {span: index for index, span in enumerate(library, start=1)}
OTHER_LABEL = len(library) + 1

labels = np.zeros((ROW_COUNT, len(column_starts)), dtype=np.int8)
for column, (births, spans) in enumerate(column_filtrations):
    for row, threshold in enumerate(row_thresholds):
        born = bisect_right(births, threshold) - 1
        if born < 0:
            continue
        labels[row, column] = library_index.get(spans[born], OTHER_LABEL)

# %%
# **Render the heatmap.** ``origin="lower"`` puts threshold zero at the
# bottom, so each column reads upward as its window's filtration; the top
# edge of the image is the band top.


def build_figure() -> plt.Figure:
    """Return the signature filtration heatmap figure."""
    higher_rank_colors = _support.signature_colors()[len(nonzero_classes) :]
    colors: list[tuple[float, float, float]] = [(1.0, 1.0, 1.0)]
    tick_labels = ["trivial"]
    for span in library:
        color, label = signature_color_and_label(span, higher_rank_colors)
        colors.append(color)
        tick_labels.append(label)
    colors.append((0.75, 0.75, 0.75))
    tick_labels.append("other non-trivial")
    colormap = ListedColormap(colors)

    figure, axes = plt.subplots(figsize=(14, 6))
    image = axes.imshow(
        labels,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=colormap,
        vmin=-0.5,
        vmax=OTHER_LABEL + 0.5,
        extent=(TIME_WINDOW_START, TIME_WINDOW_STOP, 0.0, BAND_TOP),
    )

    axes.set_xlabel("Time (sample index)")
    axes.set_ylabel("Adjacency threshold t")
    axes.set_title(f"Signature filtration: Lorenz attractor, window length {WINDOW_LENGTH}")

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_ticks(range(OTHER_LABEL + 1))
    colorbar.set_ticklabels(tick_labels)

    figure.tight_layout()
    return figure


figure = build_figure()
