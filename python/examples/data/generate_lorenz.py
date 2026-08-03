# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Lorenz cycle storage from the raw trajectory.

Reads the Lorenz position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes ``lorenz/data/lorenz_storage.cyc``: the
detected cycles the gallery queries. The trajectory is fetched from the Zenodo
published dataset and cached under ``lorenz/data`` on first use; a file already
present there is used as-is, and running ``integrate_lorenz.py`` regenerates
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
# tuned against the raw trajectory's own sampling interval so resampling
# inserts a low number of extra points, fine enough that the cover resolves
# the attractor. DOWNSAMPLE_SPACING is the detection resolution: the build
# detects at an explicit threshold just under the cube side, so the stored
# band runs from the achieved resolution up to the threshold, and MAX_LENGTH
# caps cycles at several time units of detection points. Boxsize is large
# enough that recurrences are frequent while the cover still resolves the
# attractor.
BOXSIZE = 5.0
SPHERE_RADIUS = 3.5
RESAMPLE_SPACING = 0.45
MAX_LENGTH = 800
THRESHOLD = math.nextafter(1.0, 0.0)
DOWNSAMPLE_SPACING = THRESHOLD / 2


def build() -> tuple[Path, float, float]:
    """Build the storage; return its path, inserted fraction, and resolution.

    The fraction is the share of resample-inserted rows relative to the raw
    sample count. The resolution is the detection trajectory's achieved
    consecutive-point resolution, the recorded band's lower end.
    """
    raw_path = _support.lorenz_raw()
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
    target = raw_path.parent / "lorenz_storage.cyc"
    storage.save(target)
    return target, inserted_fraction, embedded.resolution()


def report(storage_path: Path, inserted_fraction: float, achieved_resolution: float) -> None:
    """Print a storage summary: size, contents, band, and resample cost."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"lorenz_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )
    print(f"band [{achieved_resolution:.6f}, {storage.threshold():.6f}]")
    print(f"inserted points {inserted_fraction:.2%} of raw samples")


if __name__ == "__main__":
    storage_path, inserted_fraction, achieved_resolution = build()
    report(storage_path, inserted_fraction, achieved_resolution)
