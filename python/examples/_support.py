# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Shared helpers for the gallery examples: fixture paths and color constants.

Raw color values are written as RGB triples in the 0-255 range and normalized
to the [0, 1] floats matplotlib expects through ``_normalized``.
"""

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

_LORENZ_DATA = Path(__file__).resolve().parent / "data" / "lorenz"


def lorenz_path(name: str) -> Path:
    """Return the path to a committed Lorenz fixture file."""
    return _LORENZ_DATA / name


def _normalized(red: int, green: int, blue: int) -> tuple[float, float, float]:
    return (red / 255, green / 255, blue / 255)


# Eight-color categorical palette for distinguishing cycle signatures.
_SIGNATURE_PALETTE = [
    (68, 119, 238),
    (238, 136, 51),
    (85, 187, 85),
    (187, 102, 221),
    (238, 68, 68),
    (0, 187, 187),
    (238, 187, 51),
    (170, 102, 68),
]

# Five entry gray-to-dark-red colormap for purity values.
_PURITY_STOPS = [
    (0.00, (185, 185, 185)),
    (0.25, (255, 237, 160)),
    (0.50, (254, 178, 76)),
    (0.75, (253, 141, 60)),
    (1.00, (189, 0, 38)),
]

# Neutral gray for trivial cycle classes, and a lighter gray for backgrounds.
TRIVIAL_GRAY = _normalized(160, 160, 160)
BACKGROUND_GRAY = _normalized(180, 180, 180)


def signature_colors() -> list[tuple[float, float, float]]:
    """Return the categorical palette as normalized RGB triples."""
    return [_normalized(*rgb) for rgb in _SIGNATURE_PALETTE]


def purity_colormap() -> LinearSegmentedColormap:
    """Return a gray-to-dark-red colormap for purity values."""
    stops = [(position, _normalized(*rgb)) for position, rgb in _PURITY_STOPS]
    return LinearSegmentedColormap.from_list("purity", stops)
