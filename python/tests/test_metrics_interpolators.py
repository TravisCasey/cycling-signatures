import numpy as np
import pytest

import cycling_signatures as cs


def test_sphere_bundle_exposes_weight():
    assert cs.SphereBundle(0.5).direction_weight == 0.5


def test_sphere_bundle_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        cs.SphereBundle(0.0)


def test_bundle_radius_from_halfspan():
    spline = cs.CubicSpline(
        np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    )
    assert cs.ChebyshevSphereBundleInterpolator(spline, 1).radius == 1.5


def test_cubic_spline_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cs.CubicSpline(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))
