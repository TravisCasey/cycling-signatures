from typing import Any, cast

import numpy as np
import pytest

import cycling_signatures as cs


def test_bundle_direction_radius_from_construction():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    assert cs.SphereBundleInterpolator(spline, 1.5).direction_radius() == 1.5


def test_cubic_spline_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cs.CubicSpline(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_sample_outside_domain_raises():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    with pytest.raises(ValueError):
        spline.sample(-0.1)
    np.testing.assert_allclose(spline.knots(), [0.0, 1.0, 2.0])


def test_bundle_sample_rejects_zero_derivative():
    # A constant spline (equal values at both knots) has a zero derivative
    # everywhere, so the sphere-bundle direction is undefined at any parameter.
    spline = cs.CubicSpline(np.array([0.0, 1.0]), np.array([[2.0, -3.0], [2.0, -3.0]]))
    bundle = cs.SphereBundleInterpolator(spline, 1.0)
    with pytest.raises(ValueError):
        bundle.sample(0.5)


def test_metric_extraction_rejects_unknown_type():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    with pytest.raises(TypeError):
        cs.Trajectory.resample(spline, cast(Any, object()), 0.5)
