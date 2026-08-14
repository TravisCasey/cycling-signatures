# cycling-signatures

Algebraic topological descriptions of recurrent motions in high-dimensional
dynamical systems.

Given a sampled trajectory, this package finds the stretches that nearly return
to where they started and classifies each cycle by the hole it encloses, so
that two cycles winding the same way around the same obstruction share a class.
The classes a trajectory visits, filtered by an adjacency threshold, form its
cycling signature.

The signature was introduced by Bauer, Hien, Junge, and Mischaikow (2023),
[arXiv:2312.04734](https://arxiv.org/abs/2312.04734). It is computed here by
embedding the trajectory into a cubical complex, a grid of unit cubes, and
finding classes via discrete Morse theory.

## Installation

The extension module is built from source with
[maturin](https://www.maturin.rs/), which needs a Rust toolchain. From the
`python` directory of a checkout:

```bash
uv run maturin develop --release
```

## Getting started

Every public name is reached through the `cs.` alias.

```python
import numpy as np

import cycling_signatures as cs

RESAMPLE_SPACING = 0.1
DOWNSAMPLE_SPACING = 0.4

# Two turns around a circle of radius 3.
angles = np.linspace(0.0, 4.0 * np.pi, 401)
points = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)], axis=1)
interpolator = cs.CubicSpline(np.arange(len(points), dtype=float), points)
metric = cs.Euclidean()

dense = cs.Trajectory.resample(interpolator, metric, RESAMPLE_SPACING)
cover = cs.CubicalCover(dense)
detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
embedded = cs.EmbeddedTrajectory(detection, cover, metric)

window = range(len(embedded))
storage = cs.CycleStorage.build(embedded, window, len(embedded))
print(storage.signature(window))
```

```
CyclingSignature(rank=1)
```

Rank 1 is one independent loop type: the circle encloses a single hole, and
both turns around it wind the same way, so they share a class.

The two spacings are separate knobs. The resample spacing sets the fidelity of
the dense trajectory the cover is built from; the downsample spacing sets how
sparse the detection trajectory is, which is what detection cost scales with.
Build the cover from the dense trajectory, before thinning: building it from
the detection trajectory validates successfully but perforates the cover and
reports classes the curve does not have.

Detection admits a pair of detection points as the endpoints of a cycle when
their distance falls strictly below 1. A signature can be read back at any 
distance threshold in `[0, 1]` through `span_at` and `rank_at`, which emit
the classes whose recurrences close within that threshold.

## Example gallery

The gallery renders worked pipelines end to end, with plots, over the Lorenz
and Dadras systems. Build it from the `python` directory:

```bash
uv run --group docs --group examples sphinx-build -b html docs docs/_build/html
```

then open `docs/_build/html/index.html`. It downloads its example trajectories
on first use and caches them locally after that.

## License

GPL-3.0-or-later. See `LICENSE`.
