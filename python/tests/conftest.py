import numpy as np
import pytest

import cycling_signatures as cs


@pytest.fixture
def square_loop_points():
    """A closed square loop that encircles one empty unit cube.

    The loop has side length 2.0, so under unit-side cubes it surrounds the
    single interior cube without covering it. The enclosed hole gives the loop
    first homology rank 1, which the signature tests rely on.
    """
    side = 2.0
    steps = np.linspace(0.0, side, 50, endpoint=False)
    zeros = np.zeros_like(steps)
    full = np.full_like(steps, side)
    sides = [
        np.stack([steps, zeros], axis=1),
        np.stack([full, steps], axis=1),
        np.stack([side - steps, full], axis=1),
        np.stack([zeros, side - steps], axis=1),
    ]
    loop = np.concatenate(sides, axis=0)
    return np.concatenate([loop, loop[:1]], axis=0)


@pytest.fixture
def square_loop_embedded(square_loop_points):
    """The square loop embedded under the Euclidean metric."""
    trajectory = cs.Trajectory(square_loop_points)
    cover = cs.CubicalCover(trajectory)
    return cs.EmbeddedTrajectory(trajectory, cover, cs.Euclidean())


@pytest.fixture
def square_loop_storage(square_loop_embedded):
    """A ``CycleStorage`` built over the entire square loop."""
    count = len(square_loop_embedded)
    return cs.CycleStorage.build(square_loop_embedded, range(count), count)
