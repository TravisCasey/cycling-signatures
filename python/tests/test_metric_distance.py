import numpy as np
import pytest

import cycling_signatures as cs


def test_euclidean_distance_matrix_matches_pairwise():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    matrix = cs.Euclidean().distance_matrix(points)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(3.0, abs=1e-12)
    assert matrix[0, 2] == pytest.approx(4.0, abs=1e-12)
    assert matrix[1, 2] == pytest.approx(5.0, abs=1e-12)
    assert np.allclose(matrix, matrix.T)


def test_sphere_bundle_distance_normalizes_direction():
    metric = cs.SphereBundle(0)
    left = np.array([0.0, 0.0, 1.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 2.0])
    # Direction halves normalize to unit vectors sqrt(2) apart; the derived
    # weight for radius_floor 0 is 0.5.
    assert metric.distance(left, right) == pytest.approx(0.5 * np.sqrt(2.0), abs=1e-12)


def test_sphere_bundle_distance_matrix_uses_its_own_metric():
    # The two vectors differ only in their (normalized) direction halves, so
    # the sphere-bundle distance is 0.5 * sqrt(2); a Euclidean reading would
    # be sqrt(5). This pins distance_matrix to the bundle metric, not a wrong
    # delegation.
    metric = cs.SphereBundle(0)
    points = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    matrix = metric.distance_matrix(points)
    assert matrix[0, 1] == pytest.approx(0.5 * np.sqrt(2.0), abs=1e-12)


def test_distance_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cs.Euclidean().distance(np.array([0.0, 0.0]), np.array([0.0]))
