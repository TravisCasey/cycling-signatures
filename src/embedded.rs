// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Embedded trajectory: a `Trajectory` paired with a `CubicalCover` and the
//! mapping from trajectory point to cube index.

mod walker;

#[cfg(feature = "serde")]
use std::path::Path;
use std::{ops::RangeBounds, sync::Arc};

use chomp3rs::{Cube, ExecutionBackend};
use walker::{for_each_cycle_edge, walk_and_canonicalize};

use crate::{
    F2Vector,
    cover::{CubicalCover, floor_to_cube},
    distance::detect_components,
    error::{Error, Result},
    metric::Metric,
    signature::CyclingSignature,
    trajectory::{Trajectory, max_consecutive_distance},
    util::{fingerprint::Fingerprint, range::normalize_segment},
};

/// Columns each tile owns in the banded-distance passes that back cycle
/// detection.
///
/// The count bounds per-tile memory at `8 * max_length * owned_columns` bytes
/// per worker without affecting the result, and it is the largest lever on the
/// memory a detection pass needs.
///
/// Lowering tends to improve results to a certain point due to cache residence
/// and locality. There is a redundant `1 / owned_columns` portion that grows
/// as the column count is reduced, but it is outweighed in this regime. It
/// also sets parallel dispatch granularity, which is another positive of small
/// tiles.
pub(crate) const DEFAULT_OWNED_COLUMNS: usize = 256;

/// Pairs a [`Trajectory`] with a [`CubicalCover`], the metric used for queries,
/// and the per-point cube-index map.
#[derive(Debug)]
pub struct EmbeddedTrajectory {
    trajectory: Arc<Trajectory>,
    cover: CubicalCover,
    metric: Metric,
    point_to_cube: Vec<usize>,
    bound: f64,
}

impl EmbeddedTrajectory {
    /// Pairs `trajectory` with a cover of exactly the integer cubes it visits,
    /// records each point's cube index in `trajectory.points()` order, and
    /// caches the consecutive-distance bound under `metric`.
    ///
    /// Accepts an owned trajectory or one already shared with another holder;
    /// the trajectory is shared rather than copied.
    ///
    /// # Errors
    ///
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive trajectory
    ///   points land in cubes differing by more than 1 in some axis.
    /// - Any error from [`CubicalCover::from_cubes`].
    pub fn new(
        trajectory: impl Into<Arc<Trajectory>>,
        metric: Metric,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let trajectory = trajectory.into();
        let points = trajectory.points();
        let dimension = points.ncols();
        let mut current_cube: Vec<i64> = Vec::with_capacity(dimension);
        let mut next_cube: Vec<i64> = Vec::with_capacity(dimension);
        for point_index in 0..(points.nrows().saturating_sub(1)) {
            floor_to_cube(points.row(point_index), &mut current_cube);
            floor_to_cube(points.row(point_index + 1), &mut next_cube);
            for axis in 0..dimension {
                let delta = next_cube[axis] - current_cube[axis];
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
        let bound = max_consecutive_distance(trajectory.points(), metric);
        Ok(Self {
            trajectory,
            cover,
            metric,
            point_to_cube,
            bound,
        })
    }

    /// The wrapped trajectory.
    #[must_use]
    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// The metric used to compute distances over this trajectory.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The maximum metric distance between any pair of consecutive points: the
    /// achieved resolution of the trajectory under the embedded metric.
    #[must_use]
    pub fn bound(&self) -> f64 {
        self.bound
    }

    /// Returns an error if `threshold` is below the embedded trajectory's
    /// consecutive-distance bound under its metric, or at or above the unit
    /// cube side.
    pub(crate) fn check_threshold(&self, threshold: f64) -> Result<()> {
        if threshold < self.bound {
            return Err(Error::ThresholdBelowTrajectoryBound {
                threshold,
                trajectory_bound: self.bound,
            });
        }
        if threshold >= 1.0 {
            return Err(Error::ThresholdAboveCubeSide { threshold });
        }
        Ok(())
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
    /// every point's cube is present in the cover. Accepts an owned trajectory
    /// or one already shared with another holder; the trajectory is shared
    /// rather than copied.
    ///
    /// Unlike [`new`](Self::new), this constructor does **not** validate the
    /// consecutive-cube-adjacency invariant: it accepts trajectories where
    /// consecutive points land in cubes differing by more than 1 in some axis.
    /// [`walk_cycle`](Self::walk_cycle) and [`cycle_class`](Self::cycle_class)
    /// still detect non-adjacent forward steps and fail with
    /// [`Error::ConsecutiveCubesNonAdjacent`] per call, on whichever segment
    /// is queried.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedDimensionMismatch`] if dimensions disagree.
    /// - [`Error::EmbeddedCubeNotInCover`] if any point maps to a cube absent
    ///   from the cover.
    pub fn from_parts(
        trajectory: impl Into<Arc<Trajectory>>,
        cover: CubicalCover,
        metric: Metric,
    ) -> Result<Self> {
        let trajectory = trajectory.into();
        if trajectory.dimension() != cover.dimension() {
            return Err(Error::EmbeddedDimensionMismatch {
                trajectory: trajectory.dimension(),
                cover: cover.dimension(),
            });
        }

        let points = trajectory.points();
        let mut point_to_cube: Vec<usize> = Vec::with_capacity(points.nrows());

        for point_index in 0..points.nrows() {
            match cover.cube_index(points.row(point_index)) {
                Some(cube_index) => point_to_cube.push(cube_index),
                None => {
                    return Err(Error::EmbeddedCubeNotInCover { point_index });
                },
            }
        }

        let bound = max_consecutive_distance(trajectory.points(), metric);
        Ok(Self {
            trajectory,
            cover,
            metric,
            point_to_cube,
            bound,
        })
    }

    /// The sequence of 1-cube edges traversed when walking the cycle
    /// described by `segment`: forward along the trajectory from the sample at
    /// `segment.start` to the sample at `segment.end - 1`, then a closing
    /// direct cube-to-cube path back to the sample at `segment.start`.
    ///
    /// `segment` is interpreted in sample-index space
    /// (`0..trajectory.original_count()`). For resampled trajectories, sample
    /// indices are translated through
    /// [`Trajectory::original_indices`](crate::Trajectory::original_indices)
    /// to the corresponding dense-row positions before walking.
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
    /// - [`Error::CycleEndpointsNonAdjacent`] if the cubes of the trajectory
    ///   samples at `segment.start` and `segment.end - 1` differ by more than 1
    ///   in some axis.
    ///
    /// # Panics
    ///
    /// Panics if the normalized segment contains fewer than 2 points.
    pub fn walk_cycle(&self, segment: impl RangeBounds<usize>) -> Result<Vec<Cube>> {
        let segment = normalize_segment(segment, self.trajectory.original_count())?;
        assert!(
            segment.end > segment.start + 1,
            "cycle segment {}..{} must contain at least two points",
            segment.start,
            segment.end
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

    /// A stable 64-bit fingerprint combining the trajectory, the cover, and the
    /// metric identity.
    ///
    /// The metric contributes its discriminant as a single byte. The
    /// per-point cube map is not hashed: it is fully determined by the
    /// trajectory points and the cover cubes, both already covered.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fingerprint::new();
        hasher.write(&self.trajectory.fingerprint().to_le_bytes());
        hasher.write(&self.cover.fingerprint().to_le_bytes());
        hasher.write(&[self.metric as u8]);
        hasher.finish()
    }

    /// Writes the trajectory and cover to separate paths in the crate's binary
    /// format.
    ///
    /// The metric itself is not saved; supply it again when calling
    /// [`load`](Self::load).
    ///
    /// This is a convenience method, paired with [`load`](Self::load) which
    /// bundles the save methods on [`Trajectory`] and [`CubicalCover`].
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        trajectory_path: P,
        cover_path: Q,
    ) -> Result<()> {
        self.trajectory.save(trajectory_path)?;
        self.cover.save(cover_path)
    }

    /// Reads a trajectory and cover and reassembles an [`EmbeddedTrajectory`]
    /// using the supplied `metric`.
    ///
    /// This is a convenience method, paired with [`save`](Self::save) which
    /// bundles the load methods on [`Trajectory`] and [`CubicalCover`].
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedDimensionMismatch`] if the loaded trajectory and
    ///   cover disagree on spatial dimension.
    /// - [`Error::EmbeddedCubeNotInCover`] if a trajectory point maps to a cube
    ///   absent from the loaded cover.
    /// - [`Error::FormatVersionMismatch`] if either file's format version
    ///   differs.
    /// - [`Error::Io`] if either file could not be opened.
    /// - [`Error::Deserialize`] if either file's contents could not be read and
    ///   decoded.
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        trajectory_path: P,
        cover_path: Q,
        metric: Metric,
    ) -> Result<Self> {
        let trajectory = Trajectory::load(trajectory_path)?;
        let cover = CubicalCover::load(cover_path)?;
        Self::from_parts(trajectory, cover, metric)
    }

    /// The cycling signature of the trajectory over the given segment, at an
    /// explicit adjacency threshold.
    ///
    /// Returns the filtered `F_2` subspace spanned by the homology classes
    /// of recurrent cycles whose endpoint pairs fall within `segment`. A
    /// cycle's birth is the metric distance between its two endpoint
    /// samples, folded to a minimum across every cycle in its connected
    /// component; the signature is complete up to `threshold`
    /// ([`CyclingSignature::threshold_max`]).
    ///
    /// `threshold` is the adjacency threshold for cycle detection: pairs of
    /// trajectory points with metric distance `<= threshold` are candidate
    /// cycle endpoints, and union-find merges are gated by
    /// [`Metric::covers_triple`].
    ///
    /// This is not a cheap query. A signature has no cycle-length cap, so it
    /// evaluates the metric over every pair of samples in the segment, a cost
    /// growing with the square of the segment length. For a large window,
    /// prefer [`CycleStorage::build`](crate::CycleStorage::build) with an
    /// explicit `max_length`.
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not normalize to a
    ///   valid sub-range of the trajectory.
    /// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
    ///   self.bound()`.
    /// - [`Error::ThresholdAboveCubeSide`] if `threshold` is at or above the
    ///   unit cube side.
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(
        &self,
        segment: impl RangeBounds<usize>,
        threshold: f64,
    ) -> Result<CyclingSignature> {
        self.check_threshold(threshold)?;
        let range = normalize_segment(segment, self.trajectory.original_count())?;
        // A signature has no length cap, so every pair inside the segment is a
        // candidate. Detection clamps the cap to the segment's own length.
        let components = detect_components(
            &self.trajectory,
            self.metric,
            range,
            threshold,
            self.trajectory.original_count(),
            DEFAULT_OWNED_COLUMNS,
            &ExecutionBackend::Sequential,
        )?;

        let points = self.trajectory.points();
        let original_indices = self.trajectory.original_indices();
        let mut candidates: Vec<(f64, F2Vector)> = Vec::with_capacity(components.len());
        for cycles in components {
            // Every cycle in a component carries the same class, so the
            // shortest is chosen: walk cost grows with cycle length.
            let representative = cycles
                .iter()
                .min_by_key(|cycle| cycle.end - cycle.start)
                .expect("connected components are nonempty by construction");
            let class = self.cycle_class(representative.clone())?;
            let birth = cycles
                .iter()
                .map(|cycle| {
                    self.metric.distance(
                        points.row(original_indices[cycle.start]),
                        points.row(original_indices[cycle.end - 1]),
                    )
                })
                .fold(f64::INFINITY, f64::min);
            candidates.push((birth, class));
        }

        Ok(CyclingSignature::from_candidates(
            candidates,
            self.cover.num_generators(),
            threshold,
        ))
    }
}

#[cfg(test)]
mod tests {
    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::EmbeddedTrajectory;
    use crate::{
        cover::CubicalCover, error::Error, interpolation::CubicSpline, metric::Metric,
        trajectory::Trajectory,
    };
    #[cfg(feature = "serde")]
    use crate::{
        serialization::{load_from_reader, save_to_writer},
        storage::CycleStorage,
    };

    #[test]
    fn fingerprint_is_stable_golden_value() {
        // Locks the whole fingerprint stack (trajectory feed, cubes-only cover
        // feed, metric identity, and their composition) to a pinned value.
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

        assert_eq!(embedded.fingerprint(), 0xe26c_094a_37ac_7192);
    }

    #[test]
    fn walk_cycle_returns_expected_edges_for_known_loop() {
        // Square loop in 2D: 4 points, each adjacent in one axis to the next.
        // Cubes visited:  (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0).
        // Forward edges: 3 edges. Closing edge: 1 edge from (0,1)->(0,0)
        // is a unit step in axis 1 (negative direction).
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let cubes = ndarray::array![[0_i64, 0], [4, 0]];
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        let embedded =
            EmbeddedTrajectory::from_parts(trajectory, cover, Metric::Euclidean).unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded_via_new = EmbeddedTrajectory::new(
            trajectory.clone(),
            Metric::Euclidean,
            &ExecutionBackend::default(),
        )
        .unwrap();

        // Rebuild the cover from the same cube set, pair via from_parts.
        let cover_cubes = embedded_via_new.cover().cubes().to_owned();
        let fresh_cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();
        let embedded_via_from_parts =
            EmbeddedTrajectory::from_parts(trajectory, fresh_cover, Metric::Euclidean).unwrap();

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover, Metric::Euclidean);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedCubeNotInCover { point_index: 2 },
        ));
    }

    #[test]
    fn from_parts_rejects_dimension_mismatch() {
        let points = array![[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover, Metric::Euclidean);

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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let err =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap_err();
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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

        let signature = embedded.signature(.., 0.6).unwrap();
        assert_eq!(signature.rank(), 0);
    }

    #[test]
    fn walk_cycle_on_resampled_trajectory_translates_segment() {
        // Two anchors in adjacent cubes (0, 0) and (1, 0). The cubic spline
        // through them resampled to a fine bound produces a strict subset
        // `original_indices = [0, N]` with intermediate fills at 1..N.
        let knots = array![0.0, 1.0];
        let values = array![[0.5, 0.5], [1.5, 0.5]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();
        let trajectory = Trajectory::resample(&spline, Metric::Euclidean, 0.2).unwrap();
        assert!(
            trajectory.len() > trajectory.original_count(),
            "fixture must have resampled fill points",
        );
        assert_eq!(trajectory.original_count(), 2);

        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();
        let edges = embedded.walk_cycle(0..2).unwrap();
        assert!(
            !edges.is_empty(),
            "expected a non-empty edge sequence for the two-anchor cycle",
        );
    }

    /// Inserts evenly spaced intermediate points between consecutive
    /// `waypoints` so that no step's Euclidean distance exceeds `max_step`.
    ///
    /// Used to turn a short list of cube-center waypoints into a densely
    /// sampled trajectory whose consecutive-distance bound stays below the
    /// unit cube side, while every waypoint's cube membership (its
    /// coordinate floors) is unaffected: only points strictly between
    /// waypoints are added.
    fn densify_path(waypoints: &[[f64; 2]], max_step: f64) -> Array2<f64> {
        let mut points: Vec<[f64; 2]> = vec![waypoints[0]];
        for pair in waypoints.windows(2) {
            let start = pair[0];
            let end = pair[1];
            let distance = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
            let steps = ((distance / max_step).ceil() as usize).max(1);
            for step in 1..=steps {
                let fraction = step as f64 / steps as f64;
                points.push([
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                ]);
            }
        }
        let flat: Vec<f64> = points.iter().flatten().copied().collect();
        Array2::from_shape_vec((points.len(), 2), flat)
            .expect("flattened waypoint rows form a valid two-column matrix")
    }

    #[test]
    fn signature_on_recurrent_loop_returns_rank_one() {
        // 8-cube ring around a missing center cube at (1, 1), densely
        // sampled so consecutive steps stay under the unit cube side and
        // closing back to its own start. The cubical cover has H^1 of rank
        // 1; the loop's class is the generator.
        let waypoints = [
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
        let points = densify_path(&waypoints, 0.4);
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
                .unwrap();

        let signature = embedded.signature(.., 0.6).unwrap();
        assert_eq!(signature.rank(), 1);
    }

    /// Builds a densely sampled Euclidean square loop (a solid two-by-two
    /// block of cubes, no missing center) used in round-trip tests.
    #[cfg(feature = "serde")]
    fn euclidean_square_loop() -> EmbeddedTrajectory {
        let waypoints = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]];
        let points = densify_path(&waypoints, 0.4);
        let trajectory = Trajectory::new(points.view()).unwrap();
        EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::default())
            .unwrap()
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_then_load_round_trips() {
        let embedded = euclidean_square_loop();
        let mut trajectory_buffer = Vec::new();
        let mut cover_buffer = Vec::new();
        save_to_writer(&mut trajectory_buffer, embedded.trajectory()).unwrap();
        save_to_writer(&mut cover_buffer, embedded.cover()).unwrap();

        let trajectory: Trajectory = load_from_reader(&trajectory_buffer[..]).unwrap();
        let cover: CubicalCover = load_from_reader(&cover_buffer[..]).unwrap();
        let reloaded =
            EmbeddedTrajectory::from_parts(trajectory, cover, Metric::Euclidean).unwrap();

        assert_eq!(reloaded.fingerprint(), embedded.fingerprint());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn reload_with_different_metric_differs() {
        let embedded = euclidean_square_loop();
        let mut trajectory_buffer = Vec::new();
        let mut cover_buffer = Vec::new();
        save_to_writer(&mut trajectory_buffer, embedded.trajectory()).unwrap();
        save_to_writer(&mut cover_buffer, embedded.cover()).unwrap();

        let trajectory: Trajectory = load_from_reader(&trajectory_buffer[..]).unwrap();
        let cover: CubicalCover = load_from_reader(&cover_buffer[..]).unwrap();
        let reloaded =
            EmbeddedTrajectory::from_parts(trajectory, cover, Metric::SphereBundle).unwrap();

        assert_ne!(reloaded.fingerprint(), embedded.fingerprint());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn storage_provenance_matches_reassembled_embedded() {
        let embedded = euclidean_square_loop();
        let max_length = embedded.trajectory().original_count();
        let storage = CycleStorage::build(
            &embedded,
            ..,
            max_length,
            0.6,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let mut storage_buffer = Vec::new();
        save_to_writer(&mut storage_buffer, &storage).unwrap();
        let loaded_storage = load_from_reader::<CycleStorage, _>(&storage_buffer[..]).unwrap();

        let mut trajectory_buffer = Vec::new();
        let mut cover_buffer = Vec::new();
        save_to_writer(&mut trajectory_buffer, embedded.trajectory()).unwrap();
        save_to_writer(&mut cover_buffer, embedded.cover()).unwrap();
        let trajectory: Trajectory = load_from_reader(&trajectory_buffer[..]).unwrap();
        let cover: CubicalCover = load_from_reader(&cover_buffer[..]).unwrap();
        let reassembled =
            EmbeddedTrajectory::from_parts(trajectory, cover, Metric::Euclidean).unwrap();

        assert_eq!(loaded_storage.fingerprint(), reassembled.fingerprint());
    }
}
