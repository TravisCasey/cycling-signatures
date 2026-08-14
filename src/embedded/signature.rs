// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Direct signature queries over an embedded trajectory: detect, walk one
//! representative cycle per component, and reduce the classes to a filtered
//! signature.

use std::ops::RangeBounds;

use chomp3rs::ExecutionBackend;

use super::EmbeddedTrajectory;
use crate::{
    F2Vector,
    distance::{DEFAULT_OWNED_COLUMNS, detect_components},
    error::Result,
    signature::CyclingSignature,
    util::range::normalize_segment,
};

impl EmbeddedTrajectory {
    /// The cycling signature of the trajectory over the given segment.
    ///
    /// Returns the filtered `F_2` subspace spanned by the homology classes of
    /// recurrent cycles whose endpoint pairs fall within `segment`. A cycle's
    /// birth is the metric distance between its two endpoints, collected to a
    /// minimum over every cycle in its connected component.
    ///
    /// Pairs of trajectory points with metric distance strictly below 1, the
    /// cube side length, are admitted as cycle endpoints. The work is
    /// dispatched over `backend`.
    ///
    /// This is not a cheap query. A signature has no cycle-length cap, so it
    /// evaluates the metric over every pair of points in the segment, a cost
    /// growing with the square of the segment length. For a large window,
    /// prefer [`CycleStorage::build`](crate::CycleStorage::build) with an
    /// explicit `max_length`.
    ///
    /// # Errors
    ///
    /// - [`Error::SegmentOutOfBounds`](crate::Error::SegmentOutOfBounds) if
    ///   `segment` does not normalize to a valid sub-range of the trajectory.
    /// - [`Error::CycleEndpointsNonAdjacent`](crate::Error::CycleEndpointsNonAdjacent)
    ///   if a detected cycle's endpoint cubes differ by more than 1 in some
    ///   axis.
    pub fn signature(
        &self,
        segment: impl RangeBounds<usize>,
        backend: &ExecutionBackend,
    ) -> Result<CyclingSignature> {
        let trajectory = self.trajectory();
        let range = normalize_segment(segment, trajectory.len())?;
        let metric_points = self.metric_points();
        // A signature has no length cap, so every pair inside the segment is
        // admitted; detection clamps the cap to the segment's own length.
        let components = detect_components(
            &metric_points,
            range,
            trajectory.len(),
            DEFAULT_OWNED_COLUMNS,
            backend,
        )?;

        let classes = self.component_classes(&components)?;
        let births: Vec<(f64, F2Vector)> = components
            .into_iter()
            .zip(classes)
            .map(|(cycles, class)| {
                let birth = cycles
                    .iter()
                    .map(|cycle| metric_points.distance(cycle.start, cycle.end - 1))
                    .fold(f64::INFINITY, f64::min);
                (birth, class)
            })
            .collect();

        Ok(CyclingSignature::from_births(
            births,
            self.cover().num_generators(),
        ))
    }
}
