# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Shared helpers for the gallery examples: data fetching and color constants.

Each system publishes the raw position trajectory as ``.npy``, the detection
trajectory the storage was built over, and the cycle storage itself. Dadras
adds the integration time of each raw row, since its raw rows are spaced by
distance travelled rather than by time.

A storage index is an index into the detection trajectory, and that
trajectory's ``parameters()`` carry the integration time of each detection
point, in the system's own time units.
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

    Raises
    ------
    ValueError
        If the downloaded bytes do not match the expected digest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    handle, temporary = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as sink, urllib.request.urlopen(remote.url) as response:
            while chunk := response.read(1 << 20):
                sink.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != remote.sha256:
            raise ValueError(
                f"downloaded data for {target} failed its integrity check: "
                f"expected {remote.sha256}, got {actual}"
            )
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# The Zenodo record holding the published example data:
# https://zenodo.org/records/21927190
_ZENODO_RECORD = "21927190"


def _published(name: str, sha256: str) -> _RemoteFile:
    """Return the published example-data file `name` and its expected digest."""
    return _RemoteFile(
        url=f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{name}?download=1",
        sha256=sha256,
    )


_LORENZ_CACHE = Path(__file__).resolve().parent / "lorenz" / "data"

_LORENZ_STORAGE = _published(
    "lorenz_storage.cyc",
    "6c0954b42c731f56d3c6c5fa12993211c928ec2fee90ad201aa61447832403de",
)
_LORENZ_TRAJECTORY = _published(
    "lorenz_trajectory.cyc",
    "8c729cc41dc2c03e4cb8b5bf556495e38563714a373cc69170746dd8edcfac9a",
)
_LORENZ_RAW = _published(
    "lorenz_raw.npy",
    "74103f830bfc532f91a0a999a805b835f2444ed799de73ae631b372036993101",
)

# Real position units per cube: the divisor the raw Lorenz positions were
# scaled by. Multiplying a detection trajectory's position half by it recovers
# native Lorenz coordinates.
LORENZ_BOXSIZE = 5.0

# Time units per raw row: the fixed interval the raw Lorenz trajectory was
# recorded at. Lorenz raw row `i` is time `i * LORENZ_DT`, so dividing a
# detection parameter by it gives the raw row coordinate.
LORENZ_DT = 0.007


def _file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of `path`'s contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _cached(remote: _RemoteFile, target: Path) -> Path:
    """Return the local path to a data file, re-fetching a stale cache entry.

    A present cache entry is re-hashed against the known digest before being
    returned, and a mismatch triggers a re-download. A build stays offline
    after the first fetch (or if the file is placed there manually) as long as
    its digest still matches.
    """
    if not target.exists() or _file_digest(target) != remote.sha256:
        _download_verified(remote, target)
    return target


def lorenz_storage() -> Path:
    """Return the local path to the Lorenz cycle storage, fetching it if absent.

    The storage is cached under the gallery's `lorenz/data/` directory.
    """
    return _cached(_LORENZ_STORAGE, _LORENZ_CACHE / "lorenz_storage.cyc")


def lorenz_trajectory() -> Path:
    """Return the local path to the Lorenz detection trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `lorenz/data/` directory. It is the point sequence the published storage
    indexes: storage index `i` is its detection point `i`, and its
    `parameters()` are integration times in Lorenz time units.
    """
    return _cached(_LORENZ_TRAJECTORY, _LORENZ_CACHE / "lorenz_trajectory.cyc")


def lorenz_raw() -> Path:
    """Return the local path to the raw Lorenz trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `lorenz/data/` directory. Its raw rows are taken `LORENZ_DT` time units
    apart and the storage does not index them; dividing a detection parameter
    by `LORENZ_DT` gives the raw row coordinate of that detection point.
    """
    return _cached(_LORENZ_RAW, _LORENZ_CACHE / "lorenz_raw.npy")


_DADRAS_CACHE = Path(__file__).resolve().parent / "dadras" / "data"

_DADRAS_STORAGE = _published(
    "dadras_storage.cyc",
    "e359f9dd16813a2fadfc69acd0670ab92157023afd3a64117f637c217fb4398d",
)
_DADRAS_TRAJECTORY = _published(
    "dadras_trajectory.cyc",
    "97abd65ef1f85e0a2b77f1dfb2d2cd5a58e8cf452f091e4ce789b0af62a7a72d",
)
_DADRAS_RAW = _published(
    "dadras_raw.npy",
    "d9a57917ff8e44e2a9ca4b9879af3eedaf9b0c37aeb2ea4e5facd715d806e61e",
)
_DADRAS_TIMES = _published(
    "dadras_times.npy",
    "f3449347349c9015c0d8a46a262ca48c028fc8bf91123e283b5a9a6ffcb38972",
)

# Real position units per cube: the divisor the raw Dadras positions were
# scaled by. Multiplying a detection trajectory's position half by it recovers
# native Dadras coordinates.
DADRAS_BOXSIZE = 12.0


def dadras_storage() -> Path:
    """Return the local path to the Dadras cycle storage, fetching it if absent.

    The storage is cached under the gallery's `dadras/data/` directory.
    """
    return _cached(_DADRAS_STORAGE, _DADRAS_CACHE / "dadras_storage.cyc")


def dadras_trajectory() -> Path:
    """Return the local path to the Dadras detection trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `dadras/data/` directory. It is the point sequence the published storage
    indexes: storage index `i` is its detection point `i`, and its
    `parameters()` are integration times in Dadras time units.
    """
    return _cached(_DADRAS_TRAJECTORY, _DADRAS_CACHE / "dadras_trajectory.cyc")


def dadras_raw() -> Path:
    """Return the local path to the raw Dadras trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `dadras/data/` directory. Its raw rows are spaced by distance travelled
    rather than by time and the storage does not index them; `dadras_times()`
    gives the time of each raw row, and interpolating a detection parameter
    back through it gives the raw row coordinate of that detection point.
    """
    return _cached(_DADRAS_RAW, _DADRAS_CACHE / "dadras_raw.npy")


def dadras_times() -> Path:
    """Return the local path to the raw Dadras trajectory's row times.

    The file is fetched if absent and cached under the gallery's
    `dadras/data/` directory. It holds one strictly increasing integration
    time per row of the raw trajectory, in Dadras time units measured from the
    first raw row, so its first entry is zero.
    """
    return _cached(_DADRAS_TIMES, _DADRAS_CACHE / "dadras_times.npy")


def _normalized(red: int, green: int, blue: int) -> tuple[float, float, float]:
    return (red / 255, green / 255, blue / 255)


# Raw color values below are RGB triples in the 0-255 range, normalized to the
# [0, 1] floats matplotlib expects through `_normalized`.

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

# Five-stop colormap for purity values, running gray, pale yellow, orange,
# orange-red, dark red.
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

    Only the keys passed in receive colors, so a lookup for a class absent
    from ``class_keys`` raises ``KeyError``.
    """
    palette = signature_colors()
    nonzero = sorted(key for key in set(class_keys) if any(key))
    mapping = {key: palette[index] for index, key in enumerate(nonzero)}
    for key in class_keys:
        if not any(key):
            mapping[key] = (1.0, 1.0, 1.0)
    return mapping


def purity_colormap() -> LinearSegmentedColormap:
    """Return the gray-to-dark-red purity colormap."""
    stops = [(position, _normalized(*rgb)) for position, rgb in _PURITY_STOPS]
    return LinearSegmentedColormap.from_list("purity", stops)
