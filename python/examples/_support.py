# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Shared helpers for the gallery examples: data fetching and color constants.

Raw color values are written as RGB triples in the 0-255 range and normalized
to the [0, 1] floats matplotlib expects through ``_normalized``.
"""

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class _RemoteFile:
    """A single downloadable example-data file and its expected digest."""

    url: str
    sha256: str


def _download_verified(remote: _RemoteFile, target: Path) -> None:
    """Download `remote` to `target`, verifying its SHA-256.

    On success the file appears atomically at `target`. On any failure
    (including a digest mismatch) `target` is left untouched and no partial
    file remains.

    # Raises

    ValueError if the downloaded bytes do not match the expected digest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    handle, temporary = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as sink, urllib.request.urlopen(remote.url) as response:
            while chunk := response.read(1 << 20):
                sink.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != remote.sha256:
            raise ValueError(f"downloaded data for {target.name} failed its integrity check")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


_LORENZ_CACHE = Path(__file__).resolve().parent / "lorenz" / "data"

# Published example data on Zenodo.
_LORENZ_STORAGE = _RemoteFile(
    url="https://zenodo.org/records/21229992/files/lorenz_storage.cyc?download=1",
    sha256="862012e78dcc81b7709e4959329736c3748766b5a57cd75c0181dc7c528e9c0a",
)
_LORENZ_RAW = _RemoteFile(
    url="https://zenodo.org/records/21229992/files/lorenz_raw.npy?download=1",
    sha256="06d0a0c2324347d82007fa8a4f9c561a9647fbf1e75943f8fe04262e50a4cd5e",
)


def _cached(remote: _RemoteFile, target: Path) -> Path:
    """Return the local path to a data file, downloading it if absent.

    A present cache entry is returned without touching the network, so a build
    is offline after the first fetch or if the file is placed there manually.
    """
    if not target.exists():
        _download_verified(remote, target)
    return target


def lorenz_storage() -> Path:
    """Return the local path to the Lorenz cycle storage, fetching it if absent.

    The storage is cached under the gallery's `lorenz/data/` directory.
    """
    return _cached(_LORENZ_STORAGE, _LORENZ_CACHE / "lorenz_storage.cyc")


def lorenz_raw() -> Path:
    """Return the local path to the Lorenz trajectory, fetching it if absent.

    The trajectory is cached under the gallery's `lorenz/data/` directory.
    """
    return _cached(_LORENZ_RAW, _LORENZ_CACHE / "lorenz_raw.npy")


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


def signature_colors() -> list[tuple[float, float, float]]:
    """Return the categorical palette as normalized RGB triples."""
    return [_normalized(*rgb) for rgb in _SIGNATURE_PALETTE]


def class_color_map(
    class_keys: list[tuple[int, ...]],
) -> dict[tuple[int, ...], tuple[float, float, float]]:
    """Map homology-class vectors to stable colors, shared across plots.

    Each key is a class as a tuple of ints (its ``to_array`` vector). The zero
    (trivial) class maps to white; the distinct nonzero classes take palette
    colors in ascending key order, so the same class gets the same color in
    every plot built from one storage.
    """
    palette = signature_colors()
    nonzero = sorted(key for key in set(class_keys) if any(key))
    mapping = {key: palette[index] for index, key in enumerate(nonzero)}
    for key in class_keys:
        if not any(key):
            mapping[key] = (1.0, 1.0, 1.0)
    return mapping


def purity_colormap() -> LinearSegmentedColormap:
    """Return a gray-to-dark-red colormap for purity values."""
    stops = [(position, _normalized(*rgb)) for position, rgb in _PURITY_STOPS]
    return LinearSegmentedColormap.from_list("purity", stops)
