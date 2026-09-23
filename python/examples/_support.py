# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Shared helpers for the gallery examples: data fetching and color constants.

Lorenz and Dadras each publish their raw position trajectory as ``.npy``, their
detection points as a pair of position and time arrays, and their cycle
storage. Dadras also publishes the integration time of each raw row, since its
raw rows are spaced by distance travelled rather than by time; Lorenz's raw
rows are a fixed interval apart in time instead.

Charney-DeVore publishes its detection points and its cycle storages only. It
is covered at three cube sides rather than one, and at the middle cube side by
three placements of the cube lattice, so there is one storage per cover and one
set of detection points per cube side.

A storage index is an index into the detection points, and the detection times
carry the integration time of each one, in the system's own time units.
"""

import hashlib
import itertools
import math
import os
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from numpy.typing import NDArray

import cycling_signatures as cs


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
# https://zenodo.org/records/22909110
_ZENODO_RECORD = "22909110"


def _published(name: str, sha256: str) -> _RemoteFile:
    """Return the published example-data file `name` and its expected digest."""
    return _RemoteFile(
        url=f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{name}?download=1",
        sha256=sha256,
    )


@dataclass(frozen=True, eq=False)
class DetectionPoints:
    """A system's detection points, as parallel position and time arrays.

    Row `i` is detection point `i`, the index every storage cycle range uses.
    `positions` holds each point's position in the system's native coordinates.
    `times` holds each point's integration time, in the system's own time units,
    strictly increasing.
    """

    positions: NDArray[np.float32]
    times: NDArray[np.float64]


_LORENZ_CACHE = Path(__file__).resolve().parent / "lorenz" / "data"

_LORENZ_STORAGE = _published(
    "lorenz_storage.cyc",
    "76c351c0ce86753fa61265932322adc4a9bfd1336e302a5206f79148d1882ad8",
)
_LORENZ_DETECTION_POSITIONS = _published(
    "lorenz_detection_positions.npy",
    "f57c7008e40fc153d4789f15c08f0dc37c815369f36d396327cbf6e1cd3a32b3",
)
_LORENZ_DETECTION_TIMES = _published(
    "lorenz_detection_times.npy",
    "b517999285ae556fef2a0cdf57eb8504d95006027d900164a43849373453bf58",
)
_LORENZ_RAW = _published(
    "lorenz_raw.npy",
    "74103f830bfc532f91a0a999a805b835f2444ed799de73ae631b372036993101",
)

# Time units per raw row: the fixed interval the raw Lorenz trajectory was
# recorded at. Lorenz raw row `i` is time `i * LORENZ_DT`, so dividing a
# detection point's time by it gives the raw row coordinate.
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


def lorenz_detection() -> DetectionPoints:
    """Return the Lorenz detection points, fetching the arrays if absent.

    The arrays are cached under the gallery's `lorenz/data/` directory.
    """
    positions_path = _cached(
        _LORENZ_DETECTION_POSITIONS, _LORENZ_CACHE / "lorenz_detection_positions.npy"
    )
    times_path = _cached(_LORENZ_DETECTION_TIMES, _LORENZ_CACHE / "lorenz_detection_times.npy")
    return DetectionPoints(positions=np.load(positions_path), times=np.load(times_path))


def lorenz_raw() -> Path:
    """Return the local path to the raw Lorenz trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `lorenz/data/` directory. Its raw rows are taken `LORENZ_DT` time units
    apart and the storage does not index them; dividing a detection point's
    time by `LORENZ_DT` gives the raw row coordinate of that detection point.
    """
    return _cached(_LORENZ_RAW, _LORENZ_CACHE / "lorenz_raw.npy")


_DADRAS_CACHE = Path(__file__).resolve().parent / "dadras" / "data"

_DADRAS_STORAGE = _published(
    "dadras_storage.cyc",
    "33afed5d49b84537fce7b66b2343ea07b6fd41e8a380f104b30de3d48c9a342c",
)
_DADRAS_DETECTION_POSITIONS = _published(
    "dadras_detection_positions.npy",
    "ec70ac0a7ca98fc3193e481154fd8ef1ab2aa22b8cd0f9f63be616a0b7619f08",
)
_DADRAS_DETECTION_TIMES = _published(
    "dadras_detection_times.npy",
    "2082081cfe9f9abed65feb16ec6b9ebbbecaa7373559957f8b035e0ccbea4ed5",
)
_DADRAS_RAW = _published(
    "dadras_raw.npy",
    "d9a57917ff8e44e2a9ca4b9879af3eedaf9b0c37aeb2ea4e5facd715d806e61e",
)
_DADRAS_TIMES = _published(
    "dadras_times.npy",
    "f3449347349c9015c0d8a46a262ca48c028fc8bf91123e283b5a9a6ffcb38972",
)


def dadras_storage() -> Path:
    """Return the local path to the Dadras cycle storage, fetching it if absent.

    The storage is cached under the gallery's `dadras/data/` directory.
    """
    return _cached(_DADRAS_STORAGE, _DADRAS_CACHE / "dadras_storage.cyc")


def dadras_detection() -> DetectionPoints:
    """Return the Dadras detection points, fetching the arrays if absent.

    The arrays are cached under the gallery's `dadras/data/` directory.
    """
    positions_path = _cached(
        _DADRAS_DETECTION_POSITIONS, _DADRAS_CACHE / "dadras_detection_positions.npy"
    )
    times_path = _cached(_DADRAS_DETECTION_TIMES, _DADRAS_CACHE / "dadras_detection_times.npy")
    return DetectionPoints(positions=np.load(positions_path), times=np.load(times_path))


def dadras_raw() -> Path:
    """Return the local path to the raw Dadras trajectory.

    The trajectory is fetched if absent and cached under the gallery's
    `dadras/data/` directory. Its raw rows are spaced by distance travelled
    rather than by time and the storage does not index them; `dadras_times()`
    gives the time of each raw row, and interpolating a detection point's
    time back through it gives the raw row coordinate of that detection point.
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


_CHARNEY_DEVORE_CACHE = Path(__file__).resolve().parent / "charney_devore" / "data"

# Cube sides the Charney-DeVore covers are built at, in the model's own units,
# mapped to the suffix that names each one's files.
CHARNEY_DEVORE_SUFFIX = {0.005: "0005", 0.004: "0004", 0.006: "0006"}

# One storage per cover, keyed by the cover's name: the cube-side suffix for
# the lattice the coordinates arrive on, and the suffix with the placement
# appended for the two moved lattices, which are covered at 0.005 only.
_CHARNEY_DEVORE_STORAGES = {
    "0005": _published(
        "cdv_storage_0005.cyc",
        "d295709a02fad2d89390f24ee7734adaf47059cd221b87878c1af39afd4daf63",
    ),
    "0005_rotation": _published(
        "cdv_storage_0005_rotation.cyc",
        "1a98f923698549105bdb48a75a544f18550feb48d013f00f7ea914234240a101",
    ),
    "0005_shift": _published(
        "cdv_storage_0005_shift.cyc",
        "cd65fc8f29d9957d37eafea33ce8f7e0a858da11c4390f06c814add8af8046a9",
    ),
    "0004": _published(
        "cdv_storage_0004.cyc",
        "92e86039adf521aebe3d98b5061c69a2e70fed2be6e3f660fc656bc3668728a8",
    ),
    "0006": _published(
        "cdv_storage_0006.cyc",
        "982bdf53e2d39ab9b7b0d184724cc35f2c94c963c904ea34d68feb16100daf29",
    ),
}

_CHARNEY_DEVORE_DETECTION_POSITIONS = {
    "0005": _published(
        "cdv_detection_positions_0005.npy",
        "3a0f0c296926bb00cbadc8ca12733a60e5f2c053f8100fd0d0498559c9dbcfc7",
    ),
    "0004": _published(
        "cdv_detection_positions_0004.npy",
        "bc135ca09e87e1966cd46d212565d8cf713299da97065e47cb4160bfe5402e0b",
    ),
    "0006": _published(
        "cdv_detection_positions_0006.npy",
        "847fc09ee0b810d38275bce2645d593eadcd536a3625ca635106c4628293c375",
    ),
}
_CHARNEY_DEVORE_DETECTION_TIMES = {
    "0005": _published(
        "cdv_detection_times_0005.npy",
        "78678cfc41ff509d743dfa5f071f707c46d4e43cd30e4b91dfeeff4b4122079b",
    ),
    "0004": _published(
        "cdv_detection_times_0004.npy",
        "0906bcb4d97a8a2b576f79674231f744549e2003ef52159a24b2596530dc83eb",
    ),
    "0006": _published(
        "cdv_detection_times_0006.npy",
        "ab60ee1f16a6df4ac05ff2225b7ddbc8099999cbc6312b8f3aa6becd1663c853",
    ),
}


def _charney_devore_suffix(boxsize: float) -> str:
    """Return the file-name suffix for a published Charney-DeVore cube side.

    Raises
    ------
    ValueError
        If no Charney-DeVore data is published at `boxsize`.
    """
    suffix = CHARNEY_DEVORE_SUFFIX.get(boxsize)
    if suffix is None:
        published = ", ".join(str(side) for side in sorted(CHARNEY_DEVORE_SUFFIX))
        raise ValueError(f"no Charney-DeVore data at box size {boxsize}; published: {published}")
    return suffix


def charney_devore_storage(boxsize: float, grid: str = "identity") -> Path:
    """Return the path to a Charney-DeVore cycle storage, fetching it if absent.

    `boxsize` is the cube side the cover was built at, in the model's own
    units: 0.004, 0.005 or 0.006. `grid` is the placement of the cube lattice
    relative to the trajectory: ``"identity"`` for the lattice the coordinates
    arrive on, ``"rotation"`` for one turned by a fixed rotation of the
    six-dimensional state space, and ``"shift"`` for one offset by half a cube
    along every axis. The two moved lattices are published at 0.005 only.

    Each storage carries its classes in the generator basis of its own cover,
    so classes are comparable between two of these storages only through
    basis-invariant quantities such as ranks and cycle counts.

    The storage is cached under the gallery's `charney_devore/data` directory.

    Raises
    ------
    ValueError
        If no storage is published for `boxsize` and `grid`.
    """
    suffix = _charney_devore_suffix(boxsize)
    name = suffix if grid == "identity" else f"{suffix}_{grid}"
    remote = _CHARNEY_DEVORE_STORAGES.get(name)
    if remote is None:
        raise ValueError(f"no Charney-DeVore storage on the {grid} grid at box size {boxsize}")
    return _cached(remote, _CHARNEY_DEVORE_CACHE / f"cdv_storage_{name}.cyc")


def charney_devore_detection(boxsize: float) -> DetectionPoints:
    """Return the Charney-DeVore detection points, fetching them if absent.

    The arrays are cached under the gallery's `charney_devore/data` directory.

    Raises
    ------
    ValueError
        If no detection points are published at `boxsize`.
    """
    suffix = _charney_devore_suffix(boxsize)
    positions_path = _cached(
        _CHARNEY_DEVORE_DETECTION_POSITIONS[suffix],
        _CHARNEY_DEVORE_CACHE / f"cdv_detection_positions_{suffix}.npy",
    )
    times_path = _cached(
        _CHARNEY_DEVORE_DETECTION_TIMES[suffix],
        _CHARNEY_DEVORE_CACHE / f"cdv_detection_times_{suffix}.npy",
    )
    return DetectionPoints(positions=np.load(positions_path), times=np.load(times_path))


@dataclass(frozen=True)
class ClosedPrefix:
    """A prefix of the frequency order that is closed under addition.

    `size` is how many classes the prefix holds and `rank` the rank of the span
    they form. `share` is the prefix's cycles over the cycles carried by every
    nonzero class, so cycles carrying the trivial class are outside the
    denominator. `drop` is the last prefix class's cycles over the next class's,
    the frequency step down out of the prefix, and is `None` when no class
    follows the prefix.
    """

    size: int
    rank: int
    share: float
    drop: float | None


@dataclass(frozen=True)
class DominantSpan:
    """The span a storage's cycles overwhelmingly carry.

    `class_ids` indexes the storage's class list, in frequency order, and its
    classes are the whole nonzero part of a span of the stated `rank`. `share`
    is those classes' cycles over the cycles carried by every nonzero class.
    `drop` is the frequency step down to the first class outside the span, and
    is infinite when the span covers every nonzero class.
    """

    class_ids: list[int]
    rank: int
    share: float
    drop: float


def frequency_order(storage: cs.CycleStorage) -> list[tuple[int, int]]:
    """Return the storage's nonzero classes ordered by how often they occur.

    Each entry pairs a class's index in `storage.classes()` with the number of
    cycles carrying it, summed over the components of that class. The most
    frequent class comes first and ties are broken by ascending class index.
    Classes that no component carries, and the trivial class, are left out.
    """
    classes = storage.classes()
    cycles_by_class: Counter[int] = Counter()
    for component in storage.components():
        cycles_by_class[component.class_id()] += component.cycle_count()
    return sorted(
        (
            (class_id, count)
            for class_id, count in cycles_by_class.items()
            if not classes[class_id].is_zero()
        ),
        key=lambda entry: (-entry[1], entry[0]),
    )


def closed_prefixes(storage: cs.CycleStorage, order: list[tuple[int, int]]) -> list[ClosedPrefix]:
    """Return the prefixes of `order` that are closed under addition.

    `order` is a frequency order of `storage`, as `frequency_order` returns it.
    A prefix of `2 ** r - 1` classes is closed when those classes are distinct,
    span a subspace of rank `r`, and every sum of two of them is again one of
    them, which makes the prefix the whole nonzero part of that span. Prefixes
    of `2 ** r - 1` classes are tested for every `r` from 2 upward, shortest
    first.
    """
    classes = storage.classes()
    counts = [count for _, count in order]
    nonzero_cycles = sum(counts)
    prefixes = []
    for rank in itertools.count(2):
        size = 2**rank - 1
        if size > len(order):
            break
        selected = [classes[class_id] for class_id, _ in order[:size]]
        present = set(selected)
        if len(present) != size or cs.Subspace(selected).rank() != rank:
            continue
        sums_stay_inside = all(
            first ^ second in present
            for index, first in enumerate(selected)
            for second in selected[index + 1 :]
        )
        if not sums_stay_inside:
            continue
        prefixes.append(
            ClosedPrefix(
                size=size,
                rank=rank,
                share=sum(counts[:size]) / nonzero_cycles,
                drop=counts[size - 1] / counts[size] if size < len(order) else None,
            )
        )
    return prefixes


def dominant_span(storage: cs.CycleStorage) -> DominantSpan:
    """Return the span that `storage`'s cycles overwhelmingly carry.

    Ordering the nonzero classes by how often they occur, this is the closed
    prefix with the largest drop to the class after it. Choosing by the drop
    rather than by size keeps a rare closed extension of the span from
    displacing the span the cycles actually concentrate on.

    Raises
    ------
    ValueError
        If no prefix of the frequency order is closed under addition.
    """
    order = frequency_order(storage)
    prefixes = closed_prefixes(storage, order)
    if not prefixes:
        raise ValueError("no prefix of the frequency order is closed under addition")
    best = max(prefixes, key=lambda prefix: math.inf if prefix.drop is None else prefix.drop)
    return DominantSpan(
        class_ids=[class_id for class_id, _ in order[: best.size]],
        rank=best.rank,
        share=best.share,
        drop=math.inf if best.drop is None else best.drop,
    )


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

# Five-stop colormap for purity values, running white, pale yellow, orange,
# orange-red, dark red.
_PURITY_STOPS = [
    (0.00, (255, 255, 255)),
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
    """Return the white-to-dark-red purity colormap."""
    stops = [(position, _normalized(*rgb)) for position, rgb in _PURITY_STOPS]
    return LinearSegmentedColormap.from_list("purity", stops)


def purity_heatmap(
    axes: Axes,
    matrix: np.ma.MaskedArray,
    row_labels: list[str],
    column_labels: list[str],
) -> AxesImage:
    """Draw a matrix of purities on `axes` and return the image.

    Each entry is a share in `[0, 1]`, drawn on the white-to-dark-red purity
    colormap and written into its cell as a percentage. A masked entry is a
    cell with nothing to report and is left white.
    """
    colormap = purity_colormap()
    colormap.set_bad(color="white")
    image = axes.imshow(matrix, cmap=colormap, vmin=0.0, vmax=1.0, aspect="auto")
    axes.set_xticks(range(len(column_labels)), labels=column_labels)
    axes.set_yticks(range(len(row_labels)), labels=row_labels)
    mask = np.ma.getmaskarray(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if mask[row, column]:
                continue
            value = float(matrix[row, column])
            axes.text(
                column,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
                fontsize=9,
            )
    return image
