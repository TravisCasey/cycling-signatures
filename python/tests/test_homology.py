import numpy as np
import pytest

import cycling_signatures as cs


def test_signature_span_basis_round_trips_to_homology_classes(square_loop_storage):
    start, stop = square_loop_storage.extent()
    signature = square_loop_storage.signature((start, stop))
    # The loop spans one class, so its basis is exactly that stored class.
    assert signature.rank() == 1
    assert signature.span().basis() == [square_loop_storage.classes()[0]]


def test_trivial_signature_has_empty_basis(square_loop_storage):
    # A single-sample window encloses no cycle, so its signature is trivial.
    trivial = square_loop_storage.signature((0, 1))
    assert trivial.rank() == 0
    assert trivial.span().basis() == []


def test_homology_class_is_zero(square_loop_storage, square_loop_embedded):
    # The loop encircles the hole, so its class is non-trivial.
    assert not square_loop_storage.classes()[0].is_zero()
    # A short arc that closes without enclosing the hole is the trivial class.
    assert square_loop_embedded.cycle_class((0, 10)).is_zero()


def _two_hole_points():
    """Two disjoint unit-square loops joined by a straight bridge.

    The bridge does not fill in either hole, so the cube set's homology has
    rank 2: one generator per loop, independent of one another.
    """
    side = 2.0
    steps = np.linspace(0.0, side, 50, endpoint=False)
    zeros = np.zeros_like(steps)
    full = np.full_like(steps, side)

    def square(offset):
        sides = [
            np.stack([steps + offset, zeros], axis=1),
            np.stack([full + offset, steps], axis=1),
            np.stack([side - steps + offset, full], axis=1),
            np.stack([zeros + offset, side - steps], axis=1),
        ]
        return np.concatenate(sides, axis=0)

    first = square(0.0)
    second = square(6.0)
    bridge = np.stack([np.linspace(0.0, 6.0, 20), np.zeros(20)], axis=1)
    return np.concatenate([first, first[:1], bridge, second, second[:1]])


def test_xor_computes_the_symmetric_difference(square_loop_storage, square_loop_embedded):
    nonzero = square_loop_storage.classes()[0]
    zero = square_loop_embedded.cycle_class((0, 10))
    assert (nonzero ^ nonzero).is_zero()
    assert nonzero ^ zero == nonzero

    two_hole_trajectory = cs.Trajectory(_two_hole_points())
    two_hole_cover = cs.CubicalCover(two_hole_trajectory)
    two_hole_embedded = cs.EmbeddedTrajectory(two_hole_trajectory, two_hole_cover, cs.Euclidean())
    other_count = len(two_hole_embedded)
    other_storage = cs.CycleStorage.build(two_hole_embedded, range(other_count), other_count)
    with pytest.raises(ValueError):
        nonzero ^ other_storage.classes()[0]
