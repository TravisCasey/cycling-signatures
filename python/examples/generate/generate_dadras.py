# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Dadras cycle storage from the raw trajectory.

Reads the Dadras position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes two files under ``dadras/data``:
``dadras_trajectory.cyc``, the detection trajectory the storage indexes, and
``dadras_storage.cyc``, the detected cycles the gallery queries. A cycle's
point range indexes the detection trajectory directly; that trajectory's
``parameters()`` carry the integration time of each detection point, in Dadras
time units measured from the first raw row.

Because the raw rows are spaced by distance travelled rather than by time,
their times are read from the companion ``dadras_times.npy`` rather than
implied by the row number. The curve is fitted over row number and the
resulting parameters are converted to time afterwards, which keeps the fitted
geometry independent of how unevenly the flow moves through the rows.

The raw trajectory and its times are fetched from the Zenodo published dataset
and cached under ``dadras/data`` on first use; files already present there are
used as-is, and running ``integrate_dadras.py`` regenerates the same files in
place.
"""

import math
import sys
from pathlib import Path

import numpy as np

import cycling_signatures as cs

# The shared helper lives at the examples root, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support

# Sphere-bundle parameters are interdependent; see the SphereBundle metric
# docs for the rationale. SPHERE_RADIUS sets the interpolator's direction
# normalization radius, which the metric measures against directly.
# RESAMPLE_SPACING is the dense-placement spacing that feeds the cover: tuned
# against the raw trajectory's own row spacing to stay fine enough that the
# cover resolves the attractor, at the insertion cost `report` prints.
# DOWNSAMPLE_SPACING is the detection resolution: the build detects at an
# explicit threshold just under the cube side, so the stored band runs from
# the achieved resolution up to the threshold. Detection points are spaced by
# distance rather than time, so MAX_LENGTH, a count of them, caps cycles by
# their length through state space. The box size is large enough that
# recurrences are frequent while the cover still resolves the attractor.
BOXSIZE = _support.DADRAS_BOXSIZE
SPHERE_RADIUS = 3.5
RESAMPLE_SPACING = 0.45
MAX_LENGTH = 2400
THRESHOLD = math.nextafter(1.0, 0.0)
DOWNSAMPLE_SPACING = THRESHOLD / 2


def build() -> tuple[Path, Path, float, float]:
    """Build the artifacts; return their paths, inserted fraction, resolution.

    The two paths are the detection trajectory and the storage built over it.
    The fraction is the share of resample-inserted rows relative to the raw
    row count. The resolution is the detection trajectory's achieved
    consecutive-point resolution, the recorded band's lower end.
    """
    raw_path = _support.dadras_raw()
    points = np.load(raw_path)
    times = np.load(_support.dadras_times())
    row_count = len(points)
    rows = np.arange(row_count, dtype=np.float64)

    spline = cs.CubicSpline(rows, points / BOXSIZE)
    interpolator = cs.SphereBundleInterpolator(spline, SPHERE_RADIUS)
    metric = cs.SphereBundle()

    dense = cs.Trajectory.resample(interpolator, metric, RESAMPLE_SPACING)
    # Resampling records fractional row numbers; convert them to time, which
    # every later stage carries through untouched.
    dense = cs.Trajectory(dense.points(), parameters=np.interp(dense.parameters(), rows, times))
    cover = cs.CubicalCover(dense)
    detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
    inserted_fraction = (len(dense) - row_count) / row_count
    del dense, points, times, rows, spline, interpolator

    trajectory_target = raw_path.parent / "dadras_trajectory.cyc"
    detection.save(trajectory_target)

    embedded = cs.EmbeddedTrajectory(detection, cover, metric)
    storage = cs.CycleStorage.build(embedded, range(0, len(detection)), MAX_LENGTH, THRESHOLD)
    storage_target = raw_path.parent / "dadras_storage.cyc"
    storage.save(storage_target)
    return trajectory_target, storage_target, inserted_fraction, embedded.resolution()


def report(
    trajectory_path: Path,
    storage_path: Path,
    inserted_fraction: float,
    achieved_resolution: float,
) -> None:
    """Print an artifact summary: sizes, contents, band, and resample cost."""
    trajectory = cs.Trajectory.load(trajectory_path)
    storage = cs.CycleStorage.load(storage_path)
    print(f"dadras_trajectory.cyc  {trajectory_path.stat().st_size / 1e6:.1f} MB")
    print(f"dadras_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(f"detection points {len(trajectory)}")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )
    print(f"band [{achieved_resolution:.6f}, {storage.threshold():.6f}]")
    print(f"inserted points {inserted_fraction:.2%} of raw rows")


if __name__ == "__main__":
    trajectory_path, storage_path, inserted_fraction, achieved_resolution = build()
    report(trajectory_path, storage_path, inserted_fraction, achieved_resolution)
