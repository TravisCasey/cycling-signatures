# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Charney-DeVore covers and cycle storages from the raw trajectory.

Reads the raw Charney-DeVore trajectory and its row times, embeds it through
the sphere-bundle pipeline, and writes, under ``charney_devore/data``, one
cover and one storage per cover named in ``COVERS``, plus the detection points
each cube side is thinned to. A cycle's point range indexes the detection
points directly; their times carry the integration time of each detection
point, in the model's own time units measured from the first raw row.

The five covers differ in the cube lattice the trajectory is laid on: three
cube sides on the lattice the coordinates arrive on, and at the middle cube
side two moved lattices, one turned by a fixed rotation of the six-dimensional
state space and one offset by half a cube along every axis. A moved lattice
thins the same dense trajectory to the same detection points, so its storage
indexes the arrays written for its cube side and writes none of its own.

Each cover is built in its own run, in this order::

    python examples/generate/generate_charney_devore.py --cover 0005
    python examples/generate/generate_charney_devore.py --cover 0005_rotation
    python examples/generate/generate_charney_devore.py --cover 0005_shift
    python examples/generate/generate_charney_devore.py --cover 0004
    python examples/generate/generate_charney_devore.py --cover 0006

The order is a requirement: each moved lattice checks its detection points
against the times array the ``0005`` run writes, and refuses to build without
it.

A run peaks while the dense trajectory and its cover are both in memory. For
the 60 M raw rows ``integrate_charney_devore.py`` produces, the resident peak
is about 19 GB at cube side 0.004 and about 15 GB at 0.005 and 0.006. Run with
20 GB free, one cover per process.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import cycling_signatures as cs

# The shared helper lives at the examples root, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrate_charney_devore import save_atomic

import _support

# Sphere-bundle parameters are interdependent; see the SphereBundle metric
# docs for the rationale. SPHERE_RADIUS sets the interpolator's direction
# normalization radius, which the metric measures against directly.
# RESAMPLE_SPACING is the dense-placement spacing that feeds the cover: tuned
# against the raw trajectory's own row spacing to stay fine enough that the
# cover resolves the attractor. DOWNSAMPLE_SPACING bounds the detection
# resolution, which has to stay below the cube side length, 1; `report` prints
# the resolution actually achieved. The box sizes live in `COVERS`: they are
# small enough that the cover resolves the attractor's sheets and large enough
# that recurrences are frequent.
SPHERE_RADIUS = 3.5
RESAMPLE_SPACING = 0.45
DOWNSAMPLE_SPACING = 0.5

# The storage covers the first EXTENT_TIME_UNITS of the trajectory while the
# cover is built from all of it, which keeps the quadratic detection cost
# affordable on a trajectory long enough to resolve the attractor.
# CAP_TIME_UNITS is the longest recurrence kept, as a duration: detection
# points are spaced by distance rather than by time, so it is converted to a
# point cap from the extent's own time per detection point, and every cube side
# then caps the same physical recurrence time.
EXTENT_TIME_UNITS = 100_000.0
CAP_TIME_UNITS = 750.0

# The two moved lattices at the middle cube side. The shift is in cube units,
# so half a cube means half a cube whatever the box size is.
HALF_CUBE_SHIFT = 0.5
ROTATION_SEED = 20260918


def rotation(seed: int) -> NDArray[np.float64]:
    """Return a reproducible uniformly random rotation of the state space."""
    generator = np.random.default_rng(seed)
    orthogonal, upper = np.linalg.qr(generator.standard_normal((6, 6)))
    # QR fixes the factors only up to the signs of the columns, and the
    # orthogonal factor may be a reflection. Matching the signs to the diagonal
    # of the upper factor and flipping one column where the determinant is
    # negative makes the result a proper rotation.
    orthogonal = orthogonal * np.sign(np.diag(upper))
    if np.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    return orthogonal


@dataclass(frozen=True)
class Cover:
    """One cube lattice to cover the trajectory with.

    `boxsize` is the cube side in the model's own units. `rotation` turns the
    trajectory in those units before scaling, and `shift` offsets it in cube
    units afterwards; either may be absent, and the lattice the coordinates
    arrive on has both absent. `partner_of` names the cover whose detection
    points this one shares, and is absent for a cover that writes its own.
    """

    name: str
    boxsize: float
    rotation: NDArray[np.float64] | None
    shift: float | None
    partner_of: str | None


COVERS = (
    Cover("0005", 0.005, None, None, None),
    Cover("0005_rotation", 0.005, rotation(ROTATION_SEED), None, "0005"),
    Cover("0005_shift", 0.005, None, HALF_CUBE_SHIFT, "0005"),
    Cover("0004", 0.004, None, None, None),
    Cover("0006", 0.006, None, None, None),
)

_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "charney_devore" / "data"
DEFAULT_RAW = _DATA_DIRECTORY / "cdv_raw.npy"
DEFAULT_TIMES = _DATA_DIRECTORY / "cdv_times.npy"


@dataclass(frozen=True)
class BuildResult:
    """The files one cover's build wrote and the measurements it took.

    `positions_path` and `times_path` are absent for a cover that shares
    another's detection points. `detection_points` counts the points inside the
    extent, which is what the storage indexes. `cap_time_units` is
    `max_length` back in time units at the extent's own time per detection
    point, and `resolution` is the detection points' achieved
    consecutive-point resolution, which stays below the cube side length, 1.
    """

    cover_path: Path
    storage_path: Path
    positions_path: Path | None
    times_path: Path | None
    dense_points: int
    cubes: int
    generators: int
    detection_points: int
    extent_time_units: float
    max_length: int
    cap_time_units: float
    resolution: float


def scaled_points(
    raw: NDArray[np.float64],
    boxsize: float,
    cover_rotation: NDArray[np.float64] | None,
    shift: float | None,
) -> NDArray[np.float64]:
    """Return `raw` in cube units under a cover's rigid motion.

    The rotation is applied in the model's own units and the shift in cube
    units. The result is a new float64 array and `raw` is left untouched, since
    one raw trajectory feeds every cover.
    """
    points = np.asarray(raw, dtype=np.float64)
    points = points @ cover_rotation.T if cover_rotation is not None else np.array(points)
    points /= boxsize
    if shift is not None:
        points += shift
    return points


def build_cover(
    raw: NDArray[np.float64],
    times: NDArray[np.float64],
    cover: Cover,
    output_directory: Path,
) -> tuple[cs.CycleStorage, BuildResult]:
    """Build and save `cover`, its storage, and its detection point arrays.

    The cover is built from the whole dense trajectory and the storage from the
    detection points inside the first `EXTENT_TIME_UNITS`. A cover that shares
    another's detection points writes no arrays and instead checks that its own
    detection points have the same times as the shared ones.

    Raises
    ------
    FileNotFoundError
        If `cover` shares another cover's detection points and that cover has
        not been built yet.
    RuntimeError
        If `cover` shares another cover's detection points but thins the
        trajectory to different ones.
    """
    suffix = _support.CHARNEY_DEVORE_SUFFIX[cover.boxsize]
    shared_times_path = (
        None
        if cover.partner_of is None
        else output_directory / f"cdv_detection_times_{cover.partner_of}.npy"
    )
    if shared_times_path is not None and not shared_times_path.exists():
        raise FileNotFoundError(
            f"cover {cover.name} shares the detection points of cover {cover.partner_of}, "
            f"which has to be built first; {shared_times_path} is missing"
        )

    metric = cs.SphereBundle()
    scaled = scaled_points(raw, cover.boxsize, cover.rotation, cover.shift)
    spline = cs.CubicSpline(np.arange(len(raw), dtype=np.float64), scaled)
    del scaled
    interpolator = cs.SphereBundleInterpolator(spline, SPHERE_RADIUS)
    dense = cs.Trajectory.resample(interpolator, metric, RESAMPLE_SPACING)
    del spline, interpolator
    cubical_cover = cs.CubicalCover(dense)
    cover_path = output_directory / f"cdv_cover_{cover.name}.cyc"
    cubical_cover.save(cover_path)
    dense_points = len(dense)

    detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
    del dense
    # Resampling records fractional raw row numbers, which thinning carries
    # through; convert them to time, which every later stage carries untouched.
    detection_parameters = detection.parameters()
    detection_times = np.interp(
        detection_parameters, np.arange(len(times), dtype=np.float64), times
    )
    del detection_parameters

    extent_stop = int(
        np.searchsorted(detection_times, detection_times[0] + EXTENT_TIME_UNITS, side="right")
    )
    extent_times = np.array(detection_times[:extent_stop])
    del detection_times
    extent_points = detection.segment(range(0, extent_stop)).points()
    detection = cs.Trajectory(extent_points, parameters=extent_times)

    embedded = cs.EmbeddedTrajectory(detection, cubical_cover, metric)
    extent_time_units = float(extent_times[-1] - extent_times[0])
    max_length = round(CAP_TIME_UNITS * (extent_stop - 1) / extent_time_units)
    storage = cs.CycleStorage.build(embedded, range(0, extent_stop), max_length)
    storage_path = output_directory / f"cdv_storage_{cover.name}.cyc"
    storage.save(storage_path)

    positions_path: Path | None = None
    times_path: Path | None = None
    if shared_times_path is None:
        dimension = extent_points.shape[1] // 2
        positions_path = output_directory / f"cdv_detection_positions_{suffix}.npy"
        times_path = output_directory / f"cdv_detection_times_{suffix}.npy"
        save_atomic(
            (extent_points[:, :dimension] * cover.boxsize).astype(np.float32), positions_path
        )
        save_atomic(extent_times, times_path)
    else:
        shared_times = np.load(shared_times_path)
        if len(shared_times) != extent_stop or not np.array_equal(extent_times, shared_times):
            raise RuntimeError(
                f"cover {cover.name} thins the trajectory to different detection points "
                f"than cover {cover.partner_of}, whose points its storage indexes"
            )

    return storage, BuildResult(
        cover_path=cover_path,
        storage_path=storage_path,
        positions_path=positions_path,
        times_path=times_path,
        dense_points=dense_points,
        cubes=len(cubical_cover),
        generators=cubical_cover.num_generators(),
        detection_points=extent_stop,
        extent_time_units=extent_time_units,
        max_length=max_length,
        cap_time_units=max_length * extent_time_units / (extent_stop - 1),
        resolution=embedded.resolution(),
    )


def report(cover: Cover, storage: cs.CycleStorage, result: BuildResult) -> None:
    """Print an artifact summary: sizes, contents, resolution, dominant span."""
    components = storage.components()
    order = _support.frequency_order(storage)
    prefixes = _support.closed_prefixes(storage, order)
    print(f"cover {cover.name}, box size {cover.boxsize}")
    print(f"{result.cover_path.name}  {result.cover_path.stat().st_size / 1e6:.1f} MB")
    print(f"{result.storage_path.name}  {result.storage_path.stat().st_size / 1e6:.1f} MB")
    for path in (result.positions_path, result.times_path):
        if path is not None:
            print(f"{path.name}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"dense points {result.dense_points}")
    print(f"cubes {result.cubes}, generators {result.generators}")
    print(
        f"detection points {result.detection_points} over {result.extent_time_units:.1f} time units"
    )
    print(f"cap {result.max_length} points ({result.cap_time_units:.1f} time units)")
    print(f"detection resolution {result.resolution:.6f}")
    print(
        f"components {len(components)}, classes {len(storage.classes())} "
        f"({len(order)} nonzero), "
        f"cycles {sum(component.cycle_count() for component in components)}"
    )
    print(f"closed prefixes {[(prefix.size, prefix.rank) for prefix in prefixes]}")
    # Last, because a storage whose frequency order has no closed prefix raises
    # here, and everything above should already have been printed by then.
    span = _support.dominant_span(storage)
    print(
        f"dominant span positions 1 to {len(span.class_ids)}, rank {span.rank}, "
        f"share {span.share:.1%}, drop {span.drop:.1f}x"
    )


def main() -> None:
    """Build the cover named on the command line and report on it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cover",
        required=True,
        choices=[cover.name for cover in COVERS],
        help="which cover to build",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW,
        help="input .npy path for the raw rows (default %(default)s)",
    )
    parser.add_argument(
        "--times",
        type=Path,
        default=DEFAULT_TIMES,
        help="input .npy path for the raw row times (default %(default)s)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_DATA_DIRECTORY,
        help="directory to write the cover, storage and arrays to (default %(default)s)",
    )
    arguments = parser.parse_args()

    cover = next(candidate for candidate in COVERS if candidate.name == arguments.cover)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    raw = np.load(arguments.raw, mmap_mode="r")
    times = np.asarray(np.load(arguments.times, mmap_mode="r"))
    if len(times) != len(raw):
        raise ValueError(
            f"{arguments.times} holds {len(times)} times for the {len(raw)} raw rows of "
            f"{arguments.raw}; they are not one trajectory"
        )
    storage, result = build_cover(raw, times, cover, arguments.output_directory)
    report(cover, storage, result)


if __name__ == "__main__":
    main()
