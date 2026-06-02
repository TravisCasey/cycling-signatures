import numpy as np
import pytest


@pytest.fixture
def square_loop_points():
    """A closed unit-square loop, sampled for unit-cube adjacency."""
    steps = np.linspace(0.0, 1.0, 25, endpoint=False)
    sides = [
        np.stack([steps, np.zeros_like(steps)], axis=1),
        np.stack([np.ones_like(steps), steps], axis=1),
        np.stack([1.0 - steps, np.ones_like(steps)], axis=1),
        np.stack([np.zeros_like(steps), 1.0 - steps], axis=1),
    ]
    loop = np.concatenate(sides, axis=0)
    return np.concatenate([loop, loop[:1]], axis=0)
