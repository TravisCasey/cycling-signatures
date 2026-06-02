import numpy as np
import pytest

import cycling_signatures as cs


def test_points_roundtrip_through_numpy():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]])
    trajectory = cs.Trajectory(points)
    assert trajectory.original_count() == 3
    np.testing.assert_allclose(trajectory.points(), points)


def test_new_rejects_non_finite():
    with pytest.raises(ValueError):
        cs.Trajectory(np.array([[0.0, 0.0], [1.0, np.nan]]))


def test_resample_threads_interpolator_and_metric():
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
    spline = cs.CubicSpline(knots, values)
    trajectory = cs.Trajectory.resample(spline, cs.Euclidean(), 0.5)
    assert trajectory.original_count() == 5
    assert trajectory.points().shape[0] >= 5


def test_save_load_roundtrip(tmp_path):
    trajectory = cs.Trajectory(np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]]))
    path = str(tmp_path / "trajectory.cyc")
    trajectory.save(path)
    assert cs.Trajectory.load(path).fingerprint() == trajectory.fingerprint()


def test_load_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        cs.Trajectory.load(str(tmp_path / "missing.cyc"))
