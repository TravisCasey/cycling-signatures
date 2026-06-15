# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Signature indicator heatmap
==============================

The homological signature present at each time, shown for several window
lengths. Each (window length, time) cell is colored by its signature: white
for the trivial signature (rank 0); a rank-1 signature is the span of a single
homology class and takes that class's color (the same colors the coverage
barcode uses); the rank-2 signature, spanning both cycles, gets its own color.

The Lorenz attractor has two independent homological cycles, so a window's
signature is trivial, captures one cycle (rank 1), or captures both (rank 2)
depending on which part of the trajectory it covers.
"""

# %%
# Load the prebuilt ``CycleStorage`` from the bundled example data.

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import _support
import cycling_signatures as cs

STORAGE = cs.CycleStorage.load(_support.lorenz_path("storage.cyc"))

# %%
# **Canonical class colors.** A rank-1 signature is the span of a single
# homology class, so it takes that class's color (shared with the coverage
# barcode). The zero class is trivial; higher-rank signatures take distinct
# palette colors beyond the class colors.

class_objects = STORAGE.classes()
class_keys = [tuple(int(value) for value in hclass.to_array()) for hclass in class_objects]
CLASS_COLORS = _support.class_color_map(class_keys)
nonzero_classes = [
    (key, hclass) for key, hclass in zip(class_keys, class_objects, strict=True) if any(key)
]

# %%
# **Build a signature library.** Slide a window of a single representative
# length across the extent and tally the distinct non-trivial signatures.
#
# The library is ordered by rank, then by descending frequency within a rank,
# so the legend lists the rank-1 signatures first and the rank-2 signature
# last.

LIBRARY_LENGTH = 230
LIBRARY_STEP = 250
LIBRARY_SIZE = 6

extent_start, extent_stop = STORAGE.extent()

frequency: Counter[cs.Subspace] = Counter()
for window_start in range(extent_start, extent_stop - LIBRARY_LENGTH + 1, LIBRARY_STEP):
    subspace = STORAGE.signature(range(window_start, window_start + LIBRARY_LENGTH))
    if subspace.rank() != 0:
        frequency[subspace] += 1

ordered = sorted(frequency.items(), key=lambda item: (item[0].rank(), -item[1]))
library = [subspace for subspace, _ in ordered[:LIBRARY_SIZE]]

# %%
# **Build the label array.** Each cell is an integer label: -1 for a signature
# outside the library (trivial or uncommon) and 0..len(library)-1 for library
# members.

WINDOW_LENGTHS = (160, 230, 300)
TIME_WINDOW_START = 0
TIME_WINDOW_STOP = 6000
COLUMN_STEP = 5

column_times = list(range(TIME_WINDOW_START, TIME_WINDOW_STOP, COLUMN_STEP))
num_rows = len(WINDOW_LENGTHS)
num_cols = len(column_times)

library_index = {subspace: index for index, subspace in enumerate(library)}
labels = np.full((num_rows, num_cols), -1, dtype=np.int8)

for row_index, length in enumerate(WINDOW_LENGTHS):
    for col_index, time in enumerate(column_times):
        if time + length > extent_stop:
            continue
        subspace = STORAGE.signature(range(time, time + length))
        labels[row_index, col_index] = library_index.get(subspace, -1)

# %%
# **Render the heatmap.** The colormap puts trivial (white) at index 0 and
# each library signature at indices 1..len(library): a rank-1 signature takes
# its homology class's color, a higher-rank one a distinct palette color.
# Shift ``labels + 1`` so -1 (trivial or unlabeled) maps to 0.


def signature_color_and_label(
    subspace: cs.Subspace,
    higher_rank_colors: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], str]:
    """Return the color and legend label for one non-trivial signature."""
    if subspace.rank() == 1:
        key = next(key for key, hclass in nonzero_classes if subspace.contains(hclass))
        return CLASS_COLORS[key], f"[{' '.join(map(str, key))}] (rank 1)"
    return higher_rank_colors.pop(0), f"rank {subspace.rank()}"


def build_figure() -> plt.Figure:
    """Return the signature indicator heatmap figure."""
    higher_rank_colors = _support.signature_colors()[len(nonzero_classes) :]
    colors: list[tuple[float, float, float]] = [(1.0, 1.0, 1.0)]
    tick_labels = ["trivial"]
    for subspace in library:
        color, label = signature_color_and_label(subspace, higher_rank_colors)
        colors.append(color)
        tick_labels.append(label)
    colormap = ListedColormap(colors)

    figure, axes = plt.subplots(figsize=(14, 5))
    image = axes.imshow(
        labels + 1,
        aspect="auto",
        interpolation="nearest",
        cmap=colormap,
        vmin=-0.5,
        vmax=len(library) + 0.5,
        extent=(TIME_WINDOW_START, TIME_WINDOW_STOP, len(WINDOW_LENGTHS) - 0.5, -0.5),
    )

    axes.set_yticks(range(len(WINDOW_LENGTHS)))
    axes.set_yticklabels([str(length) for length in WINDOW_LENGTHS])
    axes.set_xlabel("Time (sample index)")
    axes.set_ylabel("Window length (samples)")
    axes.set_title("Signature indicator: Lorenz attractor")

    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_ticks(range(len(library) + 1))
    colorbar.set_ticklabels(tick_labels)

    figure.tight_layout()
    return figure


figure = build_figure()
