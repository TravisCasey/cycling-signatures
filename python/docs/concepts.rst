Concepts
========

This page defines the vocabulary the gallery and the API reference both use.

Trajectories: raw, dense, and detection
----------------------------------------

The pipeline names three point sequences per system. They are roles rather than
separate types, all three are `Trajectory` values, and one trajectory can fill
multiple roles, but the roles are not interchangeable:

- The **raw trajectory** is the input data: any sequence of sampled points in a
  Euclidean coordinate space, one row per sample, ordered along the motion. Its
  indices are **raw rows**. The gallery's raw trajectories come from a numerical
  integrator and are published as ``lorenz_raw.npy`` and ``dadras_raw.npy``. A
  continuous curve (`CubicSpline` or `SphereBundleInterpolator`) is fitted
  through the raw trajectory, and everything downstream is built from that
  fitted curve rather than from the raw points directly.
- The **dense trajectory** is `Trajectory.resample`'s output: the fitted curve
  resampled finely enough that consecutive points land in adjacent cubes. The
  cubical cover is built from the dense trajectory. The gallery does not publish
  the dense trajectory, since it exists only to build the cover and is not
  itself analyzed.
- The **detection trajectory** is the trajectory that is embedded in the cover
  and detected on, normally `Trajectory.downsample`'s output: the dense
  trajectory thinned to the sparsity spacing. The gallery publishes the
  detection trajectory as ``lorenz_trajectory.cyc`` and
  ``dadras_trajectory.cyc``; a `CycleStorage` indexes it directly, so storage
  index ``i`` is detection point ``i``.

Raw and dense are easy to conflate because both are denser than a thinned
detection trajectory, but they come from different stages: raw is given and
dense is computed by resampling a curve fitted through the raw points. The
computational pipeline runs `Trajectory.resample`, then builds the cover, then
`Trajectory.downsample`, in that order; building the cover from a trajectory
thinner than the dense one perforates it and reports spurious classes.

**Downsampling is optional.** A dense trajectory can be embedded as it stands,
in which case it is the detection trajectory as well. Thinning first is the
usual choice because detection cost grows quadratically in the detection point
count, and because `CycleStorage.build`'s ``max_length`` caps a cycle's point
count, so the curve length a given cap reaches scales with the spacing. Any
detection trajectory is admissible as long as its `Trajectory.resolution` stays
below 1, which is the cube side length in the cubical cover.

**The parameterization tracks the sampling.** A `Trajectory` carries one
strictly increasing parameter per point, which the library itself does not
consume. `Trajectory.resample` records the interpolation parameter of every
point it emits and `Trajectory.downsample` carries the kept points' values
through unchanged, so `Trajectory.parameters` locates each detection point on
the fitted curve and, when the interpolator was fitted on raw row numbers, in
the raw trajectory. The gallery rebuilds the dense trajectory with those row
numbers mapped to integration time, so every later stage carries literal time
and the figures' time axes are read from `Trajectory.parameters`.

**The two spacings.** Both are named ``spacing`` and both mean the output's
maximum consecutive metric gap, but they answer to different things:

- **Resample spacing** controls cover fidelity: how finely `Trajectory.resample`
  samples the fitted curve. `resample` itself only requires a positive spacing;
  keeping it at or under 1 (the cube side) ensures consecutive dense points lie
  in intersecting cubes, which is required by the cover and is checked when the
  cover is built, not by `resample`.
- **Downsample spacing** is the sparsity knob and the primary cost lever: how
  coarsely `Trajectory.downsample` thins the dense trajectory down to the
  detection trajectory. It bounds the detection point count that cycle detection
  is quadratic in, and it bounds the detection trajectory's `resolution`, which
  must stay below the cube side length, 1 (see below). Downsampling is not
  required: when not used, the resample spacing is the bound instead.

Covers and cover generators
----------------------------

A `CubicalCover` is the set of unit cubes a dense trajectory visits, computed
once and reusable across several detection trajectories built from it. Building
the cover computes its **cover generators**: the basis of the cover's first
`F_2` cohomology, one coordinate per independent loop type in the cover's
cubical complex. A cycle's class records which combination of cover generators
it wraps.

Cycles, classes, and signatures
---------------------------------

A **cycle** is a detected near-recurrent segment of the detection trajectory:
its first and last points are strictly closer to each other than the adjacency
threshold (defined below). Its **birth** is the metric distance between those
two endpoints, and the cycle is admitted at every threshold above it.

Every cycle has a **class**: the homology object recording which cover
generators the cycle wraps. A **class vector** is that class's coordinates in
the cover's generator basis, a zero/one vector with one entry per cover
generator (`HomologyClass` in the API). The class is a mathematical object
independent of any basis, while the class vector names its coordinates in one
particular basis. See "Comparing across runs" below for when and why this
distinction matters.

A `Subspace` is an `F_2` subspace of cover homology, canonicalized so that two
spanning sets of the same space compare equal. Its **rank** is the dimension of
the subspace it represents. **Span** is the set of loop types closed under
addition that a subspace represents (matching the `span` and `span_at` methods).

A `CyclingSignature` is the filtered subspace a trajectory window visits: the
per-generator births at which each independent class first enters, ordered by
birth, together with those classes and the subspace they span. `span` returns
the subspace spanned by the entire collection, over the whole filtration band;
`span_at` returns the subspace spanned by only the classes born below a smaller
threshold, and `rank_at` its rank.

Windows, segments, and the storage extent
--------------------------------------------

A **segment** is any contiguous slice of point indices, written as a half-open
``(start, stop)`` pair; on its own it carries no analytical meaning. A
**window** is a segment used as an analysis domain: the range of detection
points a `CycleStorage.build` or `CycleStorage.signature` call considers.
Neither is a **stretch**, which is a period of time.

A `CycleStorage`'s **extent**, reported by `CycleStorage.extent`, is the window
it was built over. It is the outer bound on every later query: a signature
window must fit inside the extent, and a window starting before the extent's
start is out of bounds even at index ``0``.

Adjacency threshold, filtration band, birth cap, and box size
----------------------------------------------------------------

The **adjacency threshold** is the metric distance at which near-recurrent
cycles are detected: point pairs strictly closer than the threshold are
admitted as cycle endpoints. It is always `1`, the cube side length of the
cover, which is also the largest value the geometry below permits.

Two points farther apart than one cube side length may live in cubes two
positions apart on an axis, which would leave a cycle's closing step (the walk
between its two endpoints) undefined in the cubical cover; within one cube side
the cubes differ by at most one position per axis and meet. The same reasoning
constrains the detection trajectory: its `resolution` must stay below the cube
side, because the cycles of one component are homologous only if the three
endpoints involved in each component merge lie pairwise below it, and one of
those three distances is a consecutive pair which is bounded only by the
resolution. A detection trajectory whose `resolution` reaches the cube side
length is rejected when it is embedded; thin it more finely, or rescale the
trajectory to bring the resolution down.

A signature admits queries over the **filtration band**, `[0, 1]`. `span_at`
and `rank_at` reject a threshold outside it. Restricting a signature to a
threshold `t` keeps exactly the classes whose recurrences close within `t`: a
class is a per-cycle quantity, so it does not depend on the level the signature
is read at.

**Birth cap** is a presentation term the gallery uses: the threshold a single
figure or panel restricts its signatures to, passed to `span_at` or `rank_at`.
A panel admits every class born below its cap and hides the rest, so a
figure stacked by birth cap shows one window at a sequence of caps. Every cap
lies inside the filtration band.

**Box size** is the real position units per cube: the divisor the gallery's
generation scripts scale a system's raw positions by before constructing a
`Trajectory`. The cubical cover always tiles space with unit cubes, so box size
is what maps a system's native coordinates onto that unit grid; a larger box
size shrinks the scaled trajectory and makes recurrences more frequent, at the
cost of coarsening the cover.

Comparing across runs
------------------------

The cover generator basis is **not stable across runs**. Two runs over the same
input can compute different generator chains for the same cover, so a class
vector from one run's basis cannot be compared, entry by entry, to a class
vector from another run's basis. Concretely:

- Class vectors, `Subspace` equality, and `Subspace.contains` are meaningful
  only within one basis.
- One basis means one run, or every process loading one saved cover file:
  `CubicalCover.save` writes the exact generator chains, and `CubicalCover.load`
  restores them, so every process loading that file shares a basis and their
  class output is comparable.
- Rebuilding a cover from the same cubes gives no such guarantee: the
  fingerprint matches (it depends only on the cube set) but the basis need not,
  so a storage reunited with a rebuilt cover has class vectors indexed against a
  basis that no longer exists. Save the cover and the storage together, and load
  them together.
- Across independently built covers, compare only quantities a change of basis
  cannot alter: ranks and counts (components, cycles, distinct classes, cover
  generators), whether a class `is_zero`, and which components share a class
  (grouping keyed on something run-stable, such as each component's least
  cycle).
