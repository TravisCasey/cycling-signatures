import numpy as np
import pytest

import cycling_signatures as cs


def test_constructor_defaults_to_the_index_parameterization():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]])
    trajectory = cs.Trajectory(points)
    assert len(trajectory) == 3
    np.testing.assert_allclose(trajectory.points(), points)
    np.testing.assert_allclose(trajectory.parameters(), [0.0, 1.0, 2.0])


def test_constructor_accepts_a_supplied_parameterization():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]])
    parameters = np.array([0.25, 0.5, 4.0])
    trajectory = cs.Trajectory(points, parameters)
    np.testing.assert_allclose(trajectory.parameters(), parameters)


def test_constructor_rejects_parameter_count_mismatch():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]])
    with pytest.raises(ValueError):
        cs.Trajectory(points, np.array([0.0, 1.0]))


def test_constructor_rejects_non_increasing_parameters():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]])
    # A NaN parameter fails every comparison, so the guard must reject it
    # rather than let it silently pass.
    with pytest.raises(ValueError):
        cs.Trajectory(points, np.array([0.0, np.nan, 2.0]))


def test_new_rejects_non_finite():
    with pytest.raises(ValueError):
        cs.Trajectory(np.array([[0.0, 0.0], [1.0, np.nan]]))


def test_larger_downsample_spacing_yields_fewer_points():
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
    spline = cs.CubicSpline(knots, values)
    # Resampled finer than either thinning spacing, so neither downsample
    # sits on its own validation boundary.
    trajectory = cs.Trajectory.resample(spline, cs.Euclidean(), 0.25)
    coarse = trajectory.downsample(cs.Euclidean(), 5.0)
    fine = trajectory.downsample(cs.Euclidean(), 0.5)
    assert len(coarse) < len(fine)


def test_downsample_rejects_spacing_below_resolution():
    spline = cs.CubicSpline(np.array([0.0, 1.0]), np.array([[0.0, 0.0], [1.0, 0.0]]))
    trajectory = cs.Trajectory.resample(spline, cs.Euclidean(), 0.05)
    with pytest.raises(ValueError):
        trajectory.downsample(cs.Euclidean(), 0.01)
    # A NaN spacing fails every comparison, so the guard must reject it
    # rather than let it silently pass.
    with pytest.raises(ValueError):
        trajectory.downsample(cs.Euclidean(), float("nan"))


def test_save_load_roundtrip(tmp_path):
    trajectory = cs.Trajectory(np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]]))
    path = str(tmp_path / "trajectory.cyc")
    trajectory.save(path)
    assert cs.Trajectory.load(path).fingerprint() == trajectory.fingerprint()


def test_load_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        cs.Trajectory.load(str(tmp_path / "missing.cyc"))


def test_load_malformed_file_raises_valueerror(tmp_path):
    path = tmp_path / "malformed.cyc"
    path.write_bytes(b"not the expected format")
    with pytest.raises(ValueError):
        cs.Trajectory.load(str(path))
