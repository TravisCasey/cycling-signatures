# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Class frequency and the dominant span (Charney-DeVore)
=========================================================

Three classes carry almost every cycle the Charney-DeVore storage holds, and
those three are exactly the nonzero elements of one rank-2 span. The bars count
the cycles carrying each nonzero class, ordered most frequent first, with the
three classes of the dominant span in the colors this section gives them.

Only one prefix of the frequency order closes here, at three classes of rank 2,
which is two independent loop types and their sum. The printed output gives the
share of the cycles those three carry, the drop in cycle count to the first
class outside them, and the one relation the three satisfy.
"""

# %%
# Load the prebuilt ``CycleStorage`` for cube side 0.005 from the published
# example data, fetched and cached on first use. This example needs the
# storage alone: every quantity in it is a count of cycles or a rank.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import _support
import cycling_signatures as cs

BOXSIZE = 0.005
STORAGE = cs.CycleStorage.load(_support.charney_devore_storage(BOXSIZE))
CLASSES = STORAGE.classes()

# %%
# Order the nonzero classes by how many cycles carry them, find the prefixes of
# that order which are closed under addition, and take the dominant span: the
# closed prefix with the largest drop to the class after it.

ORDER = _support.frequency_order(STORAGE)
PREFIXES = _support.closed_prefixes(STORAGE, ORDER)
SPAN = _support.dominant_span(STORAGE)

# Colors follow frequency position, so class 1, 2 and 3 keep one color each
# across this section's figures.
COLORS = dict(zip(SPAN.class_ids, _support.signature_colors()[: len(SPAN.class_ids)], strict=True))

# %%
# The three classes of a rank-2 span satisfy one relation: the sum of two of
# them is the third. Check it on the classes themselves rather than on their
# vectors, which are coordinates in a generator basis this cover fixes and no
# other run reproduces.

FIRST, SECOND, THIRD = (CLASSES[class_id] for class_id in SPAN.class_ids)
RELATION_HOLDS = (FIRST ^ SECOND) == THIRD

# %%
# Report what the frequency order says: how many classes it holds, where it
# closes, and what the closed prefix the dominant span comes from is worth.

print(f"{len(ORDER)} nonzero classes")
for prefix in PREFIXES:
    print(f"closed prefix of {prefix.size} classes at rank {prefix.rank}")
print(
    f"dominant span: rank {SPAN.rank}, carrying {100 * SPAN.share:.1f}% of the cycles "
    f"that carry a nonzero class, with a drop of {SPAN.drop:.1f}x to the next class"
)
print(f"the sum of the first two classes is the third: {RELATION_HOLDS}")

# %%
# Draw the bars. The three classes of the dominant span take their class colors
# and every other class is gray, so the span's standing above the rest is the
# figure's whole content.


def build_figure() -> plt.Figure:
    """Return the class-frequency bars."""
    figure, axes = plt.subplots(figsize=(11, 5))

    positions = np.arange(1, len(ORDER) + 1)
    cycle_counts = [count for _, count in ORDER]
    bar_colors = [COLORS.get(class_id, (0.72, 0.72, 0.72)) for class_id, _ in ORDER]
    axes.bar(positions, cycle_counts, color=bar_colors, width=0.55, linewidth=0)
    axes.set_xticks(positions)
    axes.tick_params(axis="x", labelsize=8)
    axes.yaxis.set_major_formatter(FuncFormatter(lambda count, _position: f"{count / 1e6:g}"))
    axes.set_xlabel("frequency position")
    axes.set_ylabel("cycles carrying the class (millions)")

    figure.tight_layout()
    return figure


figure = build_figure()

# %%
# The three classes of the dominant span stand clear of everything below them,
# and theirs is the only prefix of the frequency order that fills its span. The
# classes outside it are rare, and whether they or the three are what a
# differently built cover agrees on is what *The signature under a rigid motion
# of the grid* and *The same classes at three cube sides* test.
