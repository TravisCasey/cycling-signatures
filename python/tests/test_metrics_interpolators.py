from typing import Any, cast

import numpy as np
import pytest

import cycling_signatures as cs


def test_sphere_bundle_exposes_radius_floor():
    assert cs.SphereBundle(2).radius_floor() == 2


def test_bundle_radius_from_radius_floor():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    assert cs.ChebyshevSphereBundleInterpolator(spline, 1).radius() == 1.5


def test_cubic_spline_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cs.CubicSpline(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_metric_extraction_rejects_unknown_type():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    with pytest.raises(TypeError):
        cs.Trajectory.resample(spline, cast(Any, object()), 0.5)
