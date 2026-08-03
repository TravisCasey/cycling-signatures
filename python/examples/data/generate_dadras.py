# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Dadras cycle storage from the raw trajectory.

Reads the Dadras position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes ``dadras/data/dadras_storage.cyc``: the
detected cycles the gallery queries. The trajectory is fetched from the Zenodo
published dataset and cached under ``dadras/data`` on first use; a file already
present there is used as-is, and running ``integrate_dadras.py`` regenerates
the same file in place.
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
# RESAMPLE_SPACING is the dense-placement spacing that feeds the cover:
# tuned against the raw trajectory's own sample spacing so resampling inserts
# a low number of extra points, fine enough that the cover resolves the
# attractor. DOWNSAMPLE_SPACING is the detection resolution: the build
# detects at an explicit threshold just under the cube side, so the stored
# band runs from the achieved resolution up to the threshold. The raw
# samples are spaced by distance rather than time, so MAX_LENGTH caps cycles
# by their length through state space. Boxsize is large enough that
# recurrences are frequent while the cover still resolves the attractor.
BOXSIZE = 12.0
SPHERE_RADIUS = 3.5
RESAMPLE_SPACING = 0.45
MAX_LENGTH = 1600
THRESHOLD = math.nextafter(1.0, 0.0)
DOWNSAMPLE_SPACING = THRESHOLD / 2


def build() -> tuple[Path, float, float]:
    """Build the storage; return its path, inserted fraction, and bound.

    The fraction is the share of resample-inserted rows relative to the raw
    sample count. The bound is the detection trajectory's achieved
    resolution, the recorded band's lower end.
    """
    raw_path = _support.dadras_raw()
    points = np.load(raw_path)
    sample_count = len(points)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), points / BOXSIZE)
    interpolator = cs.SphereBundleInterpolator(spline, SPHERE_RADIUS)
    metric = cs.SphereBundle()

    dense = cs.Trajectory.resample(interpolator, metric, RESAMPLE_SPACING)
    cover = cs.CubicalCover(dense)
    detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
    inserted_fraction = (len(dense) - sample_count) / sample_count
    del dense, points, spline, interpolator

    embedded = cs.EmbeddedTrajectory(detection, cover, metric)
    storage = cs.CycleStorage.build(embedded, range(0, len(detection)), MAX_LENGTH, THRESHOLD)
    target = raw_path.parent / "dadras_storage.cyc"
    storage.save(target)
    return target, inserted_fraction, embedded.bound()


def report(storage_path: Path, inserted_fraction: float, achieved_bound: float) -> None:
    """Print a storage summary: size, contents, band, and resample cost."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"dadras_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )
    print(f"band [{achieved_bound:.6f}, {storage.threshold():.6f}]")
    print(f"inserted points {inserted_fraction:.2%} of raw samples")


if __name__ == "__main__":
    storage_path, inserted_fraction, achieved_bound = build()
    report(storage_path, inserted_fraction, achieved_bound)
