// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Embedded trajectory: a `Trajectory` paired with a `CubicalCover` and the
//! mapping from trajectory point to cube index.

use std::{
    cmp::Ordering,
    ops::{Range, RangeBounds},
};

use chomp3rs::{Cube, ExecutionBackend, Orthant};
use ndarray::{Array2, ArrayView2};

use crate::{
    F2Vector,
    cover::CubicalCover,
    distance::detect_components,
    error::{Error, Result},
    metric::Metric,
    signature::{CycleComponent, CyclingSignature},
    trajectory::Trajectory,
    util::{fingerprint::Fingerprint, range::normalize_segment},
};

/// Pairs a [`Trajectory<M>`](Trajectory) with a [`CubicalCover`] and the
/// per-point cube-index map.
#[derive(Debug)]
pub struct EmbeddedTrajectory<M: Metric> {
    trajectory: Trajectory<M>,
    cover: CubicalCover,
    point_to_cube: Vec<usize>,
}

impl<M: Metric> EmbeddedTrajectory<M> {
    /// Pairs `trajectory` with a cover of exactly the integer cubes it visits,
    /// and records each point's cube index in `trajectory.points()` order.
    ///
    /// # Errors
    ///
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive trajectory
    ///   points land in cubes differing by more than 1 in some axis.
    /// - Any error from [`CubicalCover::from_cubes`].
    pub fn new(trajectory: Trajectory<M>, backend: &ExecutionBackend) -> Result<Self> {
        let points = trajectory.points();
        for point_index in 0..(points.nrows().saturating_sub(1)) {
            for axis in 0..points.ncols() {
                let current = points[(point_index, axis)].floor() as i64;
                let next = points[(point_index + 1, axis)].floor() as i64;
                let delta = next - current;
                if delta.abs() > 1 {
                    return Err(Error::ConsecutiveCubesNonAdjacent {
                        point_index,
                        axis,
                        delta,
                    });
                }
            }
        }

        let (canonical_cubes, point_to_cube) = walk_and_canonicalize(&trajectory);
        let cover = CubicalCover::from_cubes(canonical_cubes.view(), backend)?;
        Ok(Self {
            trajectory,
            cover,
            point_to_cube,
        })
    }

    /// The wrapped trajectory.
    #[must_use]
    pub fn trajectory(&self) -> &Trajectory<M> {
        &self.trajectory
    }

    /// The wrapped cover.
    #[must_use]
    pub fn cover(&self) -> &CubicalCover {
        &self.cover
    }

    /// The cube index in `cover().cubes()` of the dense-array row at
    /// `point_index` (an index into `trajectory.points()`, not a sample
    /// index).
    ///
    /// # Panics
    ///
    /// Panics if `point_index >= trajectory.len()`.
    #[must_use]
    pub(crate) fn point_to_cube(&self, point_index: usize) -> usize {
        self.point_to_cube[point_index]
    }

    /// Attaches a pre-built cover to a trajectory.
    ///
    /// Validates that the cover's dimension matches the trajectory's and that
    /// every point's cube is present in the cover.
    ///
    /// Unlike [`new`](Self::new), this constructor does **not** validate the
    /// consecutive-cube-adjacency invariant: it accepts trajectories where
    /// consecutive points land in cubes differing by more than 1 in some axis.
    /// Signature queries still detect non-adjacent forward steps and fail with
    /// [`Error::ConsecutiveCubesNonAdjacent`], but the failure is raised
    /// per-query rather than at construction.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedDimensionMismatch`] if dimensions disagree.
    /// - [`Error::EmbeddedCubeNotInCover`] if any point maps to a cube absent
    ///   from the cover.
    pub fn from_parts(trajectory: Trajectory<M>, cover: CubicalCover) -> Result<Self> {
        if trajectory.dimension() != cover.dimension() {
            return Err(Error::EmbeddedDimensionMismatch {
                trajectory: trajectory.dimension(),
                cover: cover.dimension(),
            });
        }

        let points = trajectory.points();
        let cubes = cover.cubes();
        let mut point_to_cube: Vec<usize> = Vec::with_capacity(points.nrows());
        let dimension = points.ncols();
        let mut buffer: Vec<i64> = vec![0; dimension];

        for point_index in 0..points.nrows() {
            for (axis, &value) in points.row(point_index).iter().enumerate() {
                buffer[axis] = value.floor() as i64;
            }
            match binary_search_cube(cubes, &buffer) {
                Some(cube_index) => point_to_cube.push(cube_index),
                None => {
                    return Err(Error::EmbeddedCubeNotInCover { point_index });
                },
            }
        }

        Ok(Self {
            trajectory,
            cover,
            point_to_cube,
        })
    }

    /// The sequence of 1-cube edges traversed when walking the cycle
    /// described by `segment`: forward along the trajectory from
    /// `points()[segment.start]` to `points()[segment.end - 1]`, then a
    /// closing direct cube-to-cube path back to `points()[segment.start]`.
    ///
    /// Useful for visualizing the cubical representation of a particular cycle.
    /// For the homology class only, prefer [`cycle_class`](Self::cycle_class),
    /// which avoids materializing the edge list.
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not normalize to a
    ///   valid sub-range.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if any forward step inside the
    ///   segment lands in cubes differing by more than 1 in some axis.
    ///   [`EmbeddedTrajectory::new`] catches this eagerly across the whole
    ///   trajectory; this variant surfaces here for
    ///   [`EmbeddedTrajectory::from_parts`]-constructed trajectories whose
    ///   queried segment violates the invariant.
    /// - [`Error::CycleEndpointsNonAdjacent`] if the cubes of
    ///   `points()[segment.start]` and `points()[segment.end - 1]` differ by
    ///   more than 1 in some axis.
    ///
    /// # Panics
    ///
    /// Panics if the normalized segment contains fewer than 2 points.
    pub fn walk_cycle(&self, segment: impl RangeBounds<usize>) -> Result<Vec<Cube>> {
        let segment = normalize_segment(segment, self.trajectory.original_count())?;
        assert!(
            segment.end > segment.start + 1,
            "cycle segment must contain at least two points",
        );

        let original_indices = self.trajectory.original_indices();
        let start_point = original_indices[segment.start];
        let end_point = original_indices[segment.end - 1];

        // Endpoint adjacency: the cycle's closing step requires the cubes of
        // the first and last original-index positions to differ by at most 1
        // in each axis.
        let cubes = self.cover.cubes();
        let start_cube_index = self.point_to_cube(start_point);
        let end_cube_index = self.point_to_cube(end_point);
        for axis in 0..cubes.ncols() {
            let delta = cubes[(start_cube_index, axis)] - cubes[(end_cube_index, axis)];
            if delta.abs() > 1 {
                return Err(Error::CycleEndpointsNonAdjacent {
                    start: segment.start,
                    end: segment.end,
                    axis,
                    delta,
                });
            }
        }

        let mut edges = Vec::new();
        for_each_cycle_edge(self, start_point..(end_point + 1), |edge| {
            edges.push(edge.clone());
        })?;
        Ok(edges)
    }

    /// The `F_2` homology class of the cycle described by `segment`,
    /// expressed in the cover's generator basis.
    ///
    /// Use this when only the class is needed. To inspect the underlying
    /// cubical edge sequence, call [`walk_cycle`](Self::walk_cycle) instead.
    ///
    /// # Errors
    ///
    /// Same as [`walk_cycle`](Self::walk_cycle).
    ///
    /// # Panics
    ///
    /// Same as [`walk_cycle`](Self::walk_cycle).
    pub fn cycle_class(&self, segment: impl RangeBounds<usize>) -> Result<F2Vector> {
        let edges = self.walk_cycle(segment)?;
        Ok(self.cover.chain_class(edges.iter()))
    }

    /// A stable 64-bit fingerprint combining the trajectory and cover.
    ///
    /// The per-point cube map is not hashed: it is fully determined by the
    /// trajectory points and the cover cubes, both already covered.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fingerprint::new();
        hasher.write(&self.trajectory.fingerprint().to_le_bytes());
        hasher.write(&self.cover.fingerprint().to_le_bytes());
        hasher.finish()
    }

    /// The cycling signature of the trajectory over the given segment.
    ///
    /// Returns the `F_2` subspace spanned by the non-trivial homology
    /// classes of recurrent cycles whose endpoint pairs fall within
    /// `segment`, together with the per-component decomposition (each
    /// component's cycle segments and shared class) accessible through
    /// [`CyclingSignature::components`].
    ///
    /// `threshold` is the adjacency threshold for cycle detection: pairs of
    /// trajectory points with metric distance `<= threshold` are candidate
    /// cycle endpoints, and union-find merges are gated by
    /// [`Metric::covers_triple`].
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not normalize to a
    ///   valid sub-range of the trajectory.
    /// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
    ///   self.trajectory().bound()`.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] when a detected cycle contains
    ///   consecutive points in non-adjacent cubes. This is only possible when
    ///   using [`from_parts`](Self::from_parts)-constructed trajectories that
    ///   bypassed the adjacency check in [`new`](Self::new).
    /// - [`Error::CycleEndpointsNonAdjacent`] when a detected cycle's endpoint
    ///   cubes differ by more than 1 in some axis.
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(
        &self,
        segment: impl RangeBounds<usize>,
        threshold: f64,
    ) -> Result<CyclingSignature> {
        let components = detect_components(
            &self.trajectory,
            segment,
            threshold,
            self.trajectory.original_count(),
        )?;

        let mut survivors: Vec<CycleComponent> = Vec::new();
        for cycles in components {
            let representative = cycles
                .first()
                .expect("connected components are nonempty by construction");
            let class = self.cycle_class(representative.clone())?;
            if !class.is_zero() {
                survivors.push(CycleComponent { cycles, class });
            }
        }

        Ok(CyclingSignature::from_components(
            survivors,
            self.cover.num_generators(),
        ))
    }
}

/// Walks the cubical path of the cycle described by `segment`, invoking
/// `visit` once per 1-cube edge traversed.
///
/// `segment` is the already-normalized half-open range in **point-index
/// space** (i.e., indices into `trajectory.points()`); the caller has
/// translated from sample-index space if needed and validated endpoint
/// cube adjacency. The forward path walks consecutive points
/// `segment.start..segment.end` in order; the closing path connects
/// `points[segment.end - 1]` back to `points[segment.start]` via direct
/// cube-to-cube steps.
///
/// Each step validates that the cube delta is in `{-1, 0, 1}` per axis,
/// returning `ConsecutiveCubesNonAdjacent` on violation.
fn for_each_cycle_edge<M, F>(
    embedded: &EmbeddedTrajectory<M>,
    segment: Range<usize>,
    mut visit: F,
) -> Result<()>
where
    M: Metric,
    F: FnMut(&Cube),
{
    let cubes = embedded.cover().cubes();
    let point_to_cube_coords = |point_index: usize| -> Vec<i64> {
        let cube_index = embedded.point_to_cube(point_index);
        (0..cubes.ncols())
            .map(|axis| cubes[(cube_index, axis)])
            .collect()
    };

    let dimension = cubes.ncols();
    let start_cube = point_to_cube_coords(segment.start);
    let mut base: Orthant = start_cube.iter().map(|&value| value as i32).collect();
    let mut dual: Orthant = start_cube.iter().map(|&value| value as i32 - 1).collect();

    // Forward path: each consecutive pair of points.
    for point_index in segment.start..(segment.end - 1) {
        let from = point_to_cube_coords(point_index);
        let to = point_to_cube_coords(point_index + 1);
        step_between_cubes(
            &from,
            &to,
            dimension,
            &mut base,
            &mut dual,
            &mut visit,
            point_index,
        )?;
    }

    // Closing step: from points[end - 1] to points[start].
    let end_cube = point_to_cube_coords(segment.end - 1);
    step_between_cubes(
        &end_cube,
        &start_cube,
        dimension,
        &mut base,
        &mut dual,
        &mut visit,
        segment.end - 1,
    )?;

    Ok(())
}

/// Steps from `from` cube to `to` cube one axis-aligned unit at a time.
/// Positive axis diffs are processed first (axis 0..dim), then negative
/// diffs. Each unit step emits one 1-cube edge via `visit`.
///
/// `point_index` is the trajectory-point index this step is leaving; it appears
/// in the error if any axis diff exceeds 1 in magnitude. For the cycle's
/// closing step the caller passes `point_index = segment.end - 1`.
fn step_between_cubes<F>(
    from: &[i64],
    to: &[i64],
    dimension: usize,
    base: &mut Orthant,
    dual: &mut Orthant,
    visit: &mut F,
    point_index: usize,
) -> Result<()>
where
    F: FnMut(&Cube),
{
    for axis in 0..dimension {
        let delta = to[axis] - from[axis];
        if delta.abs() > 1 {
            return Err(Error::ConsecutiveCubesNonAdjacent {
                point_index,
                axis,
                delta,
            });
        }
    }

    // Positive diffs first.
    for axis in 0..dimension {
        if to[axis] - from[axis] != 1 {
            continue;
        }
        dual[axis] += 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        base[axis] += 1;
    }

    // Negative diffs second.
    for axis in 0..dimension {
        if to[axis] - from[axis] != -1 {
            continue;
        }
        base[axis] -= 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        dual[axis] -= 1;
    }

    Ok(())
}

/// Walks `trajectory.points()`, floors each row to its `i64` cube, and returns
/// the canonical (sorted, deduplicated) cube array paired with a per-row
/// cube-index vector. The per-row indices reference the returned canonical
/// array.
fn walk_and_canonicalize<M: Metric>(trajectory: &Trajectory<M>) -> (Array2<i64>, Vec<usize>) {
    let points = trajectory.points();
    let dimension = points.ncols();
    let num_rows = points.nrows();

    // (cube_vector, point_index) pairs.
    let mut pairs: Vec<(Vec<i64>, usize)> = (0..num_rows)
        .map(|point_index| {
            let cube: Vec<i64> = points
                .row(point_index)
                .iter()
                .map(|&value| value.floor() as i64)
                .collect();
            (cube, point_index)
        })
        .collect();

    // Sort by cube vector lexicographically
    pairs.sort_by(|left, right| left.0.cmp(&right.0));

    // Build the deduplicated cube list and a point-index -> cube-index map.
    let mut unique_cubes: Vec<Vec<i64>> = Vec::new();
    let mut point_to_cube: Vec<usize> = vec![0; num_rows];
    for (cube, point_index) in pairs {
        let cube_index = match unique_cubes.last() {
            Some(last) if *last == cube => unique_cubes.len() - 1,
            _ => {
                unique_cubes.push(cube);
                unique_cubes.len() - 1
            },
        };
        point_to_cube[point_index] = cube_index;
    }

    let mut canonical = Array2::<i64>::zeros((unique_cubes.len(), dimension));
    for (cube_index, cube_vec) in unique_cubes.iter().enumerate() {
        for (column, &value) in cube_vec.iter().enumerate() {
            canonical[(cube_index, column)] = value;
        }
    }

    (canonical, point_to_cube)
}

/// Binary-searches `cubes` for a row equal to `target`, returning its row
/// index. `cubes` is assumed to be lexicographically sorted (the canonical
/// order [`CubicalCover`] enforces).
#[allow(clippy::needless_pass_by_value)]
fn binary_search_cube(cubes: ArrayView2<'_, i64>, target: &[i64]) -> Option<usize> {
    let mut low = 0_usize;
    let mut high = cubes.nrows();
    while low < high {
        let mid = low + (high - low) / 2;
        let row = cubes.row(mid);

        match row.iter().copied().cmp(target.iter().copied()) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Some(mid),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use chomp3rs::ExecutionBackend;
    use ndarray::array;

    use super::EmbeddedTrajectory;
    use crate::{
        cover::CubicalCover, error::Error, interpolation::CubicSpline, metric::Euclidean,
        trajectory::Trajectory,
    };

    #[test]
    fn fingerprint_is_stable_golden_value() {
        // Locks the whole fingerprint stack (trajectory feed, cubes-only cover
        // feed, and their composition) to a pinned value.
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        assert_eq!(embedded.fingerprint(), 0xe156_dbcc_88c8_c664);
    }

    #[test]
    fn walk_cycle_returns_expected_edges_for_known_loop() {
        // Square loop in 2D: 4 points, each adjacent in one axis to the next.
        // Cubes visited:  (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0).
        // Forward edges: 3 edges. Closing edge: 1 edge from (0,1)->(0,0)
        // is a unit step in axis 1 (negative direction).
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let edges = embedded.walk_cycle(0..4).unwrap();
        // 3 forward edges + 1 closing edge = 4 edges total.
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn cycle_class_of_boundary_loop_is_nontrivial() {
        // Boundary of a 3x3 grid, leaving the center cube (1,1) absent. The
        // cover's first cohomology has rank 1; the loop encircling the hole is
        // the generator.
        //
        // Cubes visited (in order): (0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1).
        // Each consecutive pair is adjacent, and the closing step (0,1)->(0,0)
        // differs by 1 in axis 1 only, so the cycle is valid.
        let points = array![
            [0.5, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [2.5, 1.5],
            [2.5, 2.5],
            [1.5, 2.5],
            [0.5, 2.5],
            [0.5, 1.5],
        ];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let class = embedded.cycle_class(0..8).unwrap();
        assert!(
            !class.is_zero(),
            "boundary-loop cycle class should be nontrivial; got {class:?}"
        );
    }

    #[test]
    fn walk_cycle_rejects_non_adjacent_endpoints() {
        // Trajectory whose first and last points are 3 cubes apart in axis 0:
        // (0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5). Endpoints of the
        // segment 0..4 are at cubes (0, 0) and (3, 0), which differ by 3
        // in axis 0. The forward path is fine (each consecutive pair
        // differs by 1); the closing step fails.
        let points = array![[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let err = embedded.walk_cycle(0..4).unwrap_err();
        assert!(matches!(
            err,
            Error::CycleEndpointsNonAdjacent {
                start: 0,
                end: 4,
                axis: 0,
                delta: -3 | 3,
            }
        ));
    }

    #[test]
    fn walk_cycle_rejects_non_adjacent_forward_step() {
        // A trajectory whose forward step from index 0 to index 1 jumps 4 cubes
        // in axis 0, but whose endpoints (indices 0 and 2) share a cube so the
        // endpoint-adjacency check passes. EmbeddedTrajectory::new would reject
        // this eagerly; we use `from_parts` to bypass that check and confirm the
        // walker surfaces the forward-step failure lazily.
        let points = array![[0.5, 0.5], [4.5, 0.5], [0.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let cubes = ndarray::array![[0_i64, 0], [4, 0]];
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        let embedded = EmbeddedTrajectory::from_parts(trajectory, cover).unwrap();

        let err = embedded.walk_cycle(0..3).unwrap_err();
        assert!(matches!(
            err,
            Error::ConsecutiveCubesNonAdjacent {
                point_index: 0,
                axis: 0,
                delta: 4,
            }
        ));
    }

    #[test]
    fn new_walks_trajectory_with_deduplication() {
        // Four points landing in three distinct cubes; rows 0 and 1
        // share a cube.
        //
        // (0.1, 0.1), (0.9, 0.9) -> cube (0, 0)
        // (1.5, 0.5)             -> cube (1, 0)
        // (2.5, 0.5)             -> cube (2, 0)
        let points = array![[0.1, 0.1], [0.9, 0.9], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let expected_cubes = array![[0_i64, 0], [1, 0], [2, 0]];
        assert_eq!(embedded.cover().cubes(), expected_cubes.view());
        assert_eq!(embedded.point_to_cube(0), 0);
        assert_eq!(embedded.point_to_cube(1), 0);
        assert_eq!(embedded.point_to_cube(2), 1);
        assert_eq!(embedded.point_to_cube(3), 2);
    }

    #[test]
    fn from_parts_matches_new() {
        // Build a trajectory, run new(), then re-pair via from_parts and
        // assert the same point_to_cube mapping.
        let points = array![[0.3, 0.7], [1.4, 0.2], [2.9, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded_via_new =
            EmbeddedTrajectory::new(trajectory.clone(), &ExecutionBackend::default()).unwrap();

        // Rebuild the cover from the same cube set, pair via from_parts.
        let cover_cubes = embedded_via_new.cover().cubes().to_owned();
        let fresh_cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();
        let embedded_via_from_parts =
            EmbeddedTrajectory::from_parts(trajectory, fresh_cover).unwrap();

        for point_index in 0..points.nrows() {
            assert_eq!(
                embedded_via_new.point_to_cube(point_index),
                embedded_via_from_parts.point_to_cube(point_index),
            );
        }
    }

    #[test]
    fn from_parts_rejects_missing_cube() {
        // Build a trajectory that visits cubes (0,0), (1,0), (2,0). Build
        // a cover containing only (0,0) and (1,0). from_parts must fail
        // at point_index 2 with EmbeddedCubeNotInCover.
        let points = array![[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedCubeNotInCover { point_index: 2 },
        ));
    }

    #[test]
    fn from_parts_rejects_dimension_mismatch() {
        let points = array![[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedDimensionMismatch {
                trajectory: 3,
                cover: 2,
            },
        ));
    }

    #[test]
    fn new_rejects_consecutive_cubes_non_adjacent() {
        // Two consecutive points whose floor cubes differ by 3 in axis 0.
        let points = array![[0.5, 0.5], [3.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let err = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap_err();
        assert!(matches!(
            err,
            Error::ConsecutiveCubesNonAdjacent {
                point_index: 0,
                axis: 0,
                delta: 3,
            }
        ));
    }

    #[test]
    fn signature_on_straight_line_trajectory_is_trivial() {
        // Four collinear points spaced 0.5 apart. Threshold 0.6 admits the
        // consecutive pairs (distance 0.5) but no genuine recurrence exists.
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let signature = embedded.signature(.., 0.6).unwrap();
        assert_eq!(signature.rank(), 0);
        assert!(signature.components().is_empty());
    }

    #[test]
    fn walk_cycle_on_resampled_trajectory_translates_segment() {
        // Two anchors in adjacent cubes (0, 0) and (1, 0). The cubic spline
        // through them resampled to a fine bound produces a strict subset
        // `original_indices = [0, N]` with intermediate fills at 1..N.
        let knots = array![0.0, 1.0];
        let values = array![[0.5, 0.5], [1.5, 0.5]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();
        let trajectory = Trajectory::resample(&spline, Euclidean, 0.2).unwrap();
        assert!(
            trajectory.len() > trajectory.original_count(),
            "fixture must have resampled fill points",
        );
        assert_eq!(trajectory.original_count(), 2);

        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();
        let edges = embedded.walk_cycle(0..2).unwrap();
        assert!(
            !edges.is_empty(),
            "expected a non-empty edge sequence for the two-anchor cycle",
        );
    }

    #[test]
    fn signature_on_recurrent_loop_returns_rank_one() {
        // 8-cube ring around a missing center cube at (1, 1). The cubical
        // cover has H^1 of rank 1; the loop's class is the generator.
        let points = array![
            [0.5, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [2.5, 1.5],
            [2.5, 2.5],
            [1.5, 2.5],
            [0.5, 2.5],
            [0.5, 1.5],
            [0.5, 0.5],
        ];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let signature = embedded.signature(.., 1.5).unwrap();
        assert_eq!(signature.rank(), 1);
        assert!(!signature.components().is_empty());
    }
}
