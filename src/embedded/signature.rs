// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Direct signature queries over an embedded trajectory: detect, walk one
//! representative cycle per component, and reduce the classes to a filtered
//! signature.

use std::ops::RangeBounds;

use chomp3rs::ExecutionBackend;

use super::{DEFAULT_OWNED_COLUMNS, EmbeddedTrajectory};
use crate::{
    F2Vector, distance::detect_components, error::Result, signature::CyclingSignature,
    util::range::normalize_segment,
};

impl EmbeddedTrajectory {
    /// The cycling signature of the trajectory over the given segment, at an
    /// explicit adjacency threshold.
    ///
    /// Returns the filtered `F_2` subspace spanned by the homology classes
    /// of recurrent cycles whose endpoint pairs fall within `segment`. A
    /// cycle's birth is the metric distance between its two endpoints,
    /// folded to a minimum across every cycle in its connected component; the
    /// signature is complete up to `threshold`
    /// ([`CyclingSignature::threshold_max`]).
    ///
    /// `threshold` is the adjacency threshold for cycle detection: pairs of
    /// trajectory points with metric distance `<= threshold` are admitted as
    /// cycle endpoints, and union-find merges are gated by
    /// [`Metric::covers_triple`](crate::Metric::covers_triple).
    ///
    /// This is not a cheap query. A signature has no cycle-length cap, so it
    /// evaluates the metric over every pair of points in the segment, a cost
    /// growing with the square of the segment length. For a large window,
    /// prefer [`CycleStorage::build`](crate::CycleStorage::build) with an
    /// explicit `max_length`.
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`](crate::Error::WindowOutOfBounds) if
    ///   `segment` does not normalize to a valid sub-range of the trajectory.
    /// - [`Error::ThresholdBelowTrajectoryBound`](crate::Error::ThresholdBelowTrajectoryBound)
    ///   if `threshold < self.bound()`.
    /// - [`Error::ThresholdAboveCubeSide`](crate::Error::ThresholdAboveCubeSide)
    ///   if `threshold` is at or above the cube side.
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(
        &self,
        segment: impl RangeBounds<usize>,
        threshold: f64,
    ) -> Result<CyclingSignature> {
        self.check_threshold(threshold)?;
        let trajectory = self.trajectory();
        let metric = self.metric();
        let range = normalize_segment(segment, trajectory.len())?;
        // A signature has no length cap, so every pair inside the segment is
        // admitted. Detection clamps the cap to the segment's own length.
        let components = detect_components(
            trajectory,
            metric,
            range,
            threshold,
            trajectory.len(),
            DEFAULT_OWNED_COLUMNS,
            &ExecutionBackend::Sequential,
        )?;

        let points = trajectory.points();
        let mut births: Vec<(f64, F2Vector)> = Vec::with_capacity(components.len());
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
                .map(|cycle| metric.distance(points.row(cycle.start), points.row(cycle.end - 1)))
                .fold(f64::INFINITY, f64::min);
            births.push((birth, class));
        }

        Ok(CyclingSignature::from_candidates(
            births,
            self.cover().num_generators(),
            threshold,
        ))
    }
}
