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


def test_sphere_bundle_distance_does_not_normalize_direction():
    metric = cs.SphereBundle()
    left = np.array([0.0, 0.0, 1.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 2.0])
    # Direction halves are read directly off the stored coordinates: the
    # difference (1, -2) has L2 norm sqrt(5), which exceeds the (zero)
    # position term. A rescale of the direction half, which a normalizing
    # metric would absorb, changes this result.
    assert metric.distance(left, right) == pytest.approx(np.sqrt(5.0), abs=1e-12)


def test_sphere_bundle_distance_matrix_uses_its_own_metric():
    # Position halves differ (distance 5.0) and direction halves differ
    # (distance sqrt(5.0)), so the sphere-bundle distance is the max, 5.0,
    # while a Euclidean reading over all four coordinates would be
    # sqrt(30.0). The two disagree, which pins distance_matrix to the bundle
    # metric rather than a wrong delegation to Euclidean.
    metric = cs.SphereBundle()
    points = np.array([[0.0, 0.0, 1.0, 0.0], [3.0, 4.0, 0.0, 2.0]])
    matrix = metric.distance_matrix(points)
    assert matrix[0, 1] == pytest.approx(5.0, abs=1e-12)


def test_distance_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cs.Euclidean().distance(np.array([0.0, 0.0]), np.array([0.0]))
