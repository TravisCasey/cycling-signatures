# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Shared helpers for the gallery examples: data fetching and color constants.

Each system publishes the raw position trajectory as ``.npy``, the detection
trajectory the storage was built over, and the cycle storage itself; Dadras
adds the integration time of each raw sample, since its raw samples are spaced
by distance rather than by time. A storage sample index is an index into the
detection trajectory, and that trajectory's ``parameters()`` carry the
integration time each detection point was sampled at, in the system's own time
units.

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
# https://zenodo.org/records/21794612
_ZENODO_RECORD = "21794612"


def _published(name: str, sha256: str) -> _RemoteFile:
    """Return the published example-data file `name` and its expected digest."""
    return _RemoteFile(
        url=f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{name}?download=1",
        sha256=sha256,
    )


_LORENZ_CACHE = Path(__file__).resolve().parent / "lorenz" / "data"

_LORENZ_STORAGE = _published(
    "lorenz_storage.cyc",
    "b10bcd42eed48fc27b774274c254012017a613bcbced8e60198218d0fa9dfc0e",
)
_LORENZ_TRAJECTORY = _published(
    "lorenz_trajectory.cyc",
    "31312068c1112b8fa3a20bec2f8593dbc89f23cd97566ee63e181ff58e550da1",
)
_LORENZ_RAW = _published(
    "lorenz_raw.npy",
    "74103f830bfc532f91a0a999a805b835f2444ed799de73ae631b372036993101",
)

# Real position units per cube: the divisor `data/generate_lorenz.py` scales
# the raw positions by. Multiplying a detection trajectory's position half by
# it recovers native Lorenz coordinates.
LORENZ_BOXSIZE = 5.0

# Time units per raw row: the fixed sampling interval `data/integrate_lorenz.py`
# recorded the raw trajectory at. Lorenz raw row `i` is time `i * LORENZ_DT`, so
# dividing a detection parameter by it gives the raw row coordinate.
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
    returned, so a cached file left over from a rebuilt pipeline cannot
    silently shadow regenerated data. A mismatch triggers a re-download rather
    than an error, since the cache may simply be a stale artifact. A build stays
    offline after the first fetch (or if the file is placed there manually) as
    long as its digest still matches.
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
    indexes: storage sample `i` is its point `i`, and its `parameters()` are
    integration times in Lorenz time units.
    """
    return _cached(_LORENZ_TRAJECTORY, _LORENZ_CACHE / "lorenz_trajectory.cyc")


def lorenz_raw() -> Path:
    """Return the local path to the raw Lorenz trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `lorenz/data/` directory. Its rows are the integrator's samples, taken
    `LORENZ_DT` time units apart, which the storage does not index; dividing a
    detection parameter by `LORENZ_DT` gives the raw row coordinate of that
    storage sample.
    """
    return _cached(_LORENZ_RAW, _LORENZ_CACHE / "lorenz_raw.npy")


_DADRAS_CACHE = Path(__file__).resolve().parent / "dadras" / "data"

_DADRAS_STORAGE = _published(
    "dadras_storage.cyc",
    "e76209b00569adeae47474a7fcfa8f04c54cfbef06018c45963fa3f574de9521",
)
_DADRAS_TRAJECTORY = _published(
    "dadras_trajectory.cyc",
    "6383726fd32b833353051572ee9b904c65c146673f164c8618969b9b6c5d18ae",
)
_DADRAS_RAW = _published(
    "dadras_raw.npy",
    "d9a57917ff8e44e2a9ca4b9879af3eedaf9b0c37aeb2ea4e5facd715d806e61e",
)
_DADRAS_TIMES = _published(
    "dadras_times.npy",
    "f3449347349c9015c0d8a46a262ca48c028fc8bf91123e283b5a9a6ffcb38972",
)

# Real position units per cube: the divisor `data/generate_dadras.py` scales
# the raw positions by. Multiplying a detection trajectory's position half by
# it recovers native Dadras coordinates.
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
    indexes: storage sample `i` is its point `i`, and its `parameters()` are
    integration times in Dadras time units.
    """
    return _cached(_DADRAS_TRAJECTORY, _DADRAS_CACHE / "dadras_trajectory.cyc")


def dadras_raw() -> Path:
    """Return the local path to the raw Dadras trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `dadras/data/` directory. Its rows are the integrator's samples, spaced by
    distance travelled rather than by time, which the storage does not index;
    `dadras_times()` gives the time of each raw row, and interpolating a
    detection parameter back through it gives the raw row coordinate of that
    storage sample.
    """
    return _cached(_DADRAS_RAW, _DADRAS_CACHE / "dadras_raw.npy")


def dadras_times() -> Path:
    """Return the local path to the raw Dadras trajectory's sample times.

    The file is fetched if absent and cached under the gallery's
    `dadras/data/` directory. It holds one strictly increasing integration
    time per row of the raw trajectory, in Dadras time units measured from the
    first raw row, so its first entry is zero.
    """
    return _cached(_DADRAS_TIMES, _DADRAS_CACHE / "dadras_times.npy")


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
