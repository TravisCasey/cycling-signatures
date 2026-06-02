import pytest

import cycling_signatures as cs


def test_sphere_bundle_exposes_weight():
    assert cs.SphereBundle(0.5).direction_weight == 0.5


def test_sphere_bundle_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        cs.SphereBundle(0.0)
