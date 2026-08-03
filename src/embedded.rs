// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Embedded trajectory: a `Trajectory` paired with a `CubicalCover` and the
//! mapping from trajectory point to cube index.

mod signature;
mod walker;

#[cfg(feature = "serde")]
use std::path::Path;
use std::{ops::Range, sync::Arc};

use chomp3rs::ExecutionBackend;

use crate::{
    F2Vector,
    cover::{CubicalCover, non_adjacent_axis},
    error::{Error, Result},
    interpolation::Interpolator,
    metric::{Metric, MetricPoints},
    trajectory::Trajectory,
    util::fingerprint::Fingerprint,
};

/// Pairs a [`Trajectory`] with a [`CubicalCover`], the metric used for queries,
/// and the per-point cube-index map.
///
/// The trajectory's points are the vertices cycle detection and cycle walking
/// run over; the cover is the cubical complex their homology classes live in.
/// Both are shared rather than copied, so one cover built from a dense
/// trajectory can back several embeddings.
#[derive(Debug)]
pub struct EmbeddedTrajectory {
    trajectory: Arc<Trajectory>,
    cover: Arc<CubicalCover>,
    metric: Metric,
    point_to_cube: Vec<usize>,
    resolution: f64,
}

impl EmbeddedTrajectory {
    /// Places `trajectory`'s points in `cover`, recording each point's cube
    /// index and the consecutive-point resolution under `metric`.
    ///
    /// `cover` must contain the cube of every point, so it is normally built
    /// from this trajectory, or from the denser trajectory this one was
    /// thinned from; see [`CubicalCover::build`]. Accepts owned values or ones
    /// already shared with another holder.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedDimensionMismatch`] if the trajectory and the cover
    ///   disagree on dimension.
    /// - [`Error::EmbeddedCubeNotInCover`] if a point maps to a cube absent
    ///   from the cover.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive trajectory
    ///   points land in cubes differing by more than 1 in some axis.
    pub fn new(
        trajectory: impl Into<Arc<Trajectory>>,
        cover: impl Into<Arc<CubicalCover>>,
        metric: Metric,
    ) -> Result<Self> {
        let trajectory = trajectory.into();
        let cover = cover.into();
        if trajectory.dimension() != cover.dimension() {
            return Err(Error::EmbeddedDimensionMismatch {
                trajectory: trajectory.dimension(),
                cover: cover.dimension(),
            });
        }

        let points = trajectory.points();
        let point_to_cube = cover.cube_indices(points)?;

        // Consecutive points must land in adjacent cubes for every walk over
        // this trajectory to be well defined, so the whole trajectory is
        // checked once here rather than per queried segment.
        let cubes = cover.cubes();
        for point_index in 0..point_to_cube.len().saturating_sub(1) {
            let current = point_to_cube[point_index];
            let next = point_to_cube[point_index + 1];
            if current == next {
                continue;
            }
            if let Some((axis, delta)) = non_adjacent_axis(cubes.row(current), cubes.row(next)) {
                return Err(Error::ConsecutiveCubesNonAdjacent {
                    point_index,
                    axis,
                    delta,
                });
            }
        }

        let resolution = trajectory.resolution(metric);
        Ok(Self {
            trajectory,
            cover,
            metric,
            point_to_cube,
            resolution,
        })
    }

    /// Runs the whole embedding pipeline over `interpolator` in one call:
    /// resample at `resample_spacing`, build the cover from that dense
    /// trajectory, thin it to `downsample_spacing`, and embed the result.
    ///
    /// The dense trajectory is discarded. Run the stages separately
    /// ([`Trajectory::resample`], [`CubicalCover::build`],
    /// [`Trajectory::downsample`], [`new`](Self::new)) to keep it, for
    /// plotting or for re-thinning at another spacing.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
    /// let values =
    ///     array![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]];
    /// let spline = CubicSpline::new(knots, values.view()).unwrap();
    ///
    /// let embedded = EmbeddedTrajectory::from_interpolator(
    ///     &spline,
    ///     Metric::Euclidean,
    ///     0.2,
    ///     0.4,
    ///     &ExecutionBackend::default(),
    /// )
    /// .unwrap();
    /// assert!(embedded.resolution() <= 0.4);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::SpacingNotPositive`] if either spacing is zero, negative or
    ///   NaN.
    /// - [`Error::InterpolationKnotCount`] if the interpolator has fewer than
    ///   two knots.
    /// - [`Error::ResampleNonFinite`] if any interpolator output is not finite.
    /// - [`Error::ResampleStagnation`] if bisection cannot reach
    ///   `resample_spacing` at machine precision.
    /// - [`Error::SpacingBelowResolution`] if `downsample_spacing` is below the
    ///   resampled trajectory's own consecutive-point distance.
    /// - [`Error::CubicalCoverZeroDimension`] if the interpolator's samples
    ///   have zero columns.
    /// - [`Error::CubeCoordinateOutOfRange`] if a cube coordinate falls outside
    ///   `[i32::MIN, i32::MAX - 1]`.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if `resample_spacing` leaves
    ///   consecutive dense points more than one cube apart (surfaced at the
    ///   cover build), or `downsample_spacing` leaves consecutive kept points
    ///   more than one cube apart (surfaced at the embedding).
    ///
    /// # Panics
    ///
    /// Same as [`Trajectory::resample`].
    pub fn from_interpolator<I: Interpolator>(
        interpolator: &I,
        metric: Metric,
        resample_spacing: f64,
        downsample_spacing: f64,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let dense = Trajectory::resample(interpolator, metric, resample_spacing)?;
        let cover = CubicalCover::build(&dense, backend)?;
        let detection = dense.downsample(metric, downsample_spacing)?;
        // The dense trajectory is the largest allocation this call makes, and
        // nothing below reads it.
        drop(dense);
        Self::new(detection, cover, metric)
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

    /// The maximum metric distance between consecutive trajectory points: the
    /// detection resolution of this embedding, equal to
    /// [`Trajectory::resolution`] under the embedded metric.
    ///
    /// This is the smallest usable cycle-detection threshold: below it some
    /// consecutive pair of points is farther apart than the threshold, and
    /// cycles a single step apart can no longer be shown to be homologous.
    #[must_use]
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Returns an error if `threshold` is below the embedded trajectory's
    /// consecutive-point resolution under its metric (including when it is
    /// NaN), or at or above the cube side.
    pub(crate) fn check_threshold(&self, threshold: f64) -> Result<()> {
        // Negated form (rather than `threshold < self.resolution`) so a NaN
        // threshold fails loudly here instead of silently passing both band
        // checks; past this guard the threshold is a number, so a plain
        // comparison suffices below.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(threshold >= self.resolution) {
            return Err(Error::ThresholdBelowResolution {
                threshold,
                resolution: self.resolution,
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

    /// The cube index in `cover().cubes()` of the trajectory point at
    /// `point_index`.
    ///
    /// # Panics
    ///
    /// Panics if `point_index >= trajectory.len()`.
    #[must_use]
    fn point_to_cube(&self, point_index: usize) -> usize {
        self.point_to_cube[point_index]
    }

    /// The trajectory's points viewed through the metric this embedding was
    /// built with, ready for repeated indexed distance queries.
    #[must_use]
    pub(crate) fn metric_points(&self) -> MetricPoints<'_> {
        self.metric.over(self.trajectory.points())
    }

    /// The homology class of every component in `components`, in order.
    ///
    /// Each component's class is read off its shortest cycle: every cycle in a
    /// component carries the same class, and walk cost grows with cycle
    /// length.
    ///
    /// # Errors
    ///
    /// Same as [`cycle_class`](Self::cycle_class), for the representative
    /// walked out of each component.
    pub(crate) fn component_classes(
        &self,
        components: &[Vec<Range<usize>>],
    ) -> Result<Vec<F2Vector>> {
        let mut classes: Vec<F2Vector> = Vec::with_capacity(components.len());
        for cycles in components {
            let representative = cycles
                .iter()
                .min_by_key(|cycle| cycle.end - cycle.start)
                .expect("connected components are nonempty by construction");
            classes.push(self.cycle_class(representative.clone())?);
        }
        Ok(classes)
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
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive points of the
    ///   loaded trajectory land in cubes differing by more than 1 in some axis.
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
        Self::new(trajectory, cover, metric)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::EmbeddedTrajectory;
    use crate::{
        cover::CubicalCover,
        error::{Error, Result},
        interpolation::CubicSpline,
        metric::Metric,
        trajectory::Trajectory,
    };
    #[cfg(feature = "serde")]
    use crate::{
        serialization::{load_from_reader, save_to_writer},
        storage::CycleStorage,
    };

    /// Covers `trajectory`'s own cubes and embeds it under the `Euclidean`
    /// metric: the shortest spelling of the pipeline for a trajectory that is
    /// already at the resolution the test wants to detect at.
    fn embed_euclidean(trajectory: Trajectory) -> Result<EmbeddedTrajectory> {
        let cover = CubicalCover::build(&trajectory, &ExecutionBackend::default())?;
        EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean)
    }

    /// The centers of the eight cubes ringing the missing center cube `(1, 1)`,
    /// closing back on the first.
    ///
    /// Covering these cubes and no others leaves a one-cube hole, so the cover
    /// has `H^1` of rank one and a loop around the ring carries its generator.
    fn ring_waypoints() -> [[f64; 2]; 9] {
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [2.5, 1.5],
            [2.5, 2.5],
            [1.5, 2.5],
            [0.5, 2.5],
            [0.5, 1.5],
            [0.5, 0.5],
        ]
    }

    /// Stacks `points` into a two-column array, one row per point.
    fn stack_points(points: &[[f64; 2]]) -> Array2<f64> {
        let flat: Vec<f64> = points.iter().flatten().copied().collect();
        Array2::from_shape_vec((points.len(), 2), flat)
            .expect("flattened point rows form a valid two-column matrix")
    }

    /// Inserts evenly spaced intermediate points between consecutive
    /// `waypoints` so that no step's Euclidean distance exceeds `max_step`.
    ///
    /// Turns a short list of cube-center waypoints into a densely sampled
    /// trajectory whose consecutive-point resolution stays below the cube side,
    /// while every waypoint's cube membership (its coordinate floors) is
    /// unaffected: only points strictly between waypoints are added.
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
        stack_points(&points)
    }

    #[test]
    fn fingerprint_is_stable_golden_value() {
        // Locks the whole fingerprint stack (trajectory feed, cubes-only cover
        // feed, metric identity, and their composition) to a pinned value.
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

        assert_eq!(embedded.fingerprint(), 0x0412_381b_dab5_0d2c);
    }

    #[test]
    fn walk_cycle_returns_expected_edges_for_known_loop() {
        // Square loop in 2D: 4 points, each adjacent in one axis to the next.
        // Cubes visited:  (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0).
        // Forward edges: 3 edges. Closing edge: 1 edge from (0,1)->(0,0)
        // is a unit step in axis 1 (negative direction).
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

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
        // Each consecutive pair of cube centers is adjacent, and the closing
        // step (0,1)->(0,0) differs by 1 in axis 1 only, so the cycle is valid.
        let points = stack_points(&ring_waypoints()[..8]);
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

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
        let embedded = embed_euclidean(trajectory).unwrap();

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
    fn new_maps_each_point_to_its_cover_cube() {
        // Four points landing in three distinct cubes, which the cover holds
        // in sorted order as (0, 0), (1, 0), (2, 0). Rows 0 and 1 share a
        // cube, so the map is not the identity.
        //
        // (0.1, 0.1), (0.9, 0.9) -> cube (0, 0)
        // (1.5, 0.5)             -> cube (1, 0)
        // (2.5, 0.5)             -> cube (2, 0)
        let points = array![[0.1, 0.1], [0.9, 0.9], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

        assert_eq!(embedded.point_to_cube(0), 0);
        assert_eq!(embedded.point_to_cube(1), 0);
        assert_eq!(embedded.point_to_cube(2), 1);
        assert_eq!(embedded.point_to_cube(3), 2);
    }

    #[test]
    fn new_rejects_cube_missing_from_cover() {
        // A trajectory visiting cubes (0,0), (1,0), (2,0) against a cover
        // holding only (0,0) and (1,0): the point at index 2 has nowhere to
        // land.
        let points = array![[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedCubeNotInCover { point_index: 2 },
        ));
    }

    #[test]
    fn new_rejects_dimension_mismatch() {
        let points = array![[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean);

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

        let err = embed_euclidean(trajectory).unwrap_err();

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
    fn cycle_class_agrees_between_dense_and_downsampled_walks() {
        // One cover, built from the dense trajectory, shared by two
        // embeddings: the dense trajectory itself and a downsampled copy of
        // it. Downsampling always keeps the first and last point, so
        // `cycle_class(..)` describes the same closed loop in both, and the
        // two walks differ only in how many points they step through.
        //
        // The nonzero assertion is what keeps this from passing vacuously: on
        // a hole the fixture accidentally filled, both classes would agree at
        // zero and the comparison would evidence nothing.
        let points = densify_path(&ring_waypoints(), 0.1);
        let dense = Trajectory::new(points.view()).unwrap();
        let cover = Arc::new(CubicalCover::build(&dense, &ExecutionBackend::default()).unwrap());
        assert!(
            cover.num_generators() >= 1,
            "fixture cover has no generators, so every class below is zero",
        );

        let detection = dense.downsample(Metric::Euclidean, 0.5).unwrap();
        assert!(
            detection.len() < dense.len(),
            "fixture does not thin, so both walks step the same points",
        );

        let dense_embedded =
            EmbeddedTrajectory::new(dense, Arc::clone(&cover), Metric::Euclidean).unwrap();
        let detection_embedded =
            EmbeddedTrajectory::new(detection, cover, Metric::Euclidean).unwrap();

        let dense_class = dense_embedded.cycle_class(..).unwrap();
        let detection_class = detection_embedded.cycle_class(..).unwrap();

        assert!(
            !dense_class.is_zero(),
            "the loop encircles the missing center cube, so its class is nonzero",
        );
        assert_eq!(dense_class, detection_class);
    }

    #[test]
    fn from_interpolator_covers_the_dense_trajectory() {
        // A segment clipping the corner cube (1, 1): it crosses x = 1 well
        // before y = 1, so the cube is occupied over a short stretch that the
        // dense trajectory samples and the thinned one steps past. The two
        // cube sets therefore differ, which is what makes the cover's source
        // observable at all.
        const RESAMPLE_SPACING: f64 = 0.05;
        const DOWNSAMPLE_SPACING: f64 = 0.5;
        let knots = array![0.0, 1.0];
        let values = array![[0.2, 1.9], [1.8, 0.3]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();

        let dense = Trajectory::resample(&spline, Metric::Euclidean, RESAMPLE_SPACING).unwrap();
        let detection = dense
            .downsample(Metric::Euclidean, DOWNSAMPLE_SPACING)
            .unwrap();
        let dense_cover = CubicalCover::build(&dense, &ExecutionBackend::default()).unwrap();
        let detection_cover =
            CubicalCover::build(&detection, &ExecutionBackend::default()).unwrap();
        assert_ne!(
            dense_cover.fingerprint(),
            detection_cover.fingerprint(),
            "fixture's dense and thinned cube sets agree, so this test cannot tell them apart",
        );

        let embedded = EmbeddedTrajectory::from_interpolator(
            &spline,
            Metric::Euclidean,
            RESAMPLE_SPACING,
            DOWNSAMPLE_SPACING,
            &ExecutionBackend::default(),
        )
        .unwrap();

        assert_eq!(embedded.cover().fingerprint(), dense_cover.fingerprint());
    }

    #[test]
    fn signature_on_straight_line_trajectory_is_trivial() {
        // Four collinear points spaced 0.5 apart. Threshold 0.6 admits the
        // consecutive pairs (distance 0.5) but no genuine recurrence exists.
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

        let signature = embedded.signature(.., 0.6).unwrap();
        assert_eq!(signature.rank(), 0);
    }

    #[test]
    fn signature_on_recurrent_loop_returns_rank_one() {
        // 8-cube ring around a missing center cube at (1, 1), densely
        // sampled so consecutive steps stay under the cube side and closing
        // back to its own start. The cubical cover has H^1 of rank 1; the
        // loop's class is the generator.
        let points = densify_path(&ring_waypoints(), 0.4);
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();

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
        embed_euclidean(trajectory).unwrap()
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
        let reloaded = EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean).unwrap();

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
        let reloaded = EmbeddedTrajectory::new(trajectory, cover, Metric::SphereBundle).unwrap();

        assert_ne!(reloaded.fingerprint(), embedded.fingerprint());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn storage_provenance_matches_reassembled_embedded() {
        let embedded = euclidean_square_loop();
        let max_length = embedded.trajectory().len();
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
        let reassembled = EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean).unwrap();

        assert_eq!(loaded_storage.fingerprint(), reassembled.fingerprint());
    }
}
