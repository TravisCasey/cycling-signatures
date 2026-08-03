// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Construction pipeline for [`CycleStorage`]: explicit-threshold cycle
//! detection over a trajectory segment, followed by per-component class
//! deduplication and assembly.

use std::ops::{Range, RangeBounds};

use chomp3rs::ExecutionBackend;

use super::{Component, Cycle, CycleStorage};
use crate::{
    EmbeddedTrajectory, F2Vector,
    distance::detect_components,
    embedded::DEFAULT_OWNED_COLUMNS,
    error::{Error, Result},
    storage::interval_subsumption::IntervalSubsumptionIndex,
    util::range::normalize_segment,
};

impl CycleStorage {
    /// Builds the storage by running cycle detection over `segment` at an
    /// explicit adjacency `threshold`, storing each surviving component's
    /// homology class and per-cycle birth.
    ///
    /// The returned storage's [`threshold`](Self::threshold) is `threshold`
    /// itself.
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    ///   `0..embedded.trajectory().len()`.
    /// - [`Error::InvalidMaxLength`] if `max_length < 2`.
    /// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
    ///   embedded.bound()`.
    /// - [`Error::ThresholdAboveCubeSide`] if `threshold` is at or above the
    ///   cube side.
    pub fn build(
        embedded: &EmbeddedTrajectory,
        segment: impl RangeBounds<usize>,
        max_length: usize,
        threshold: f64,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let range = normalize_segment(segment, embedded.trajectory().len())?;
        if max_length < 2 {
            return Err(Error::InvalidMaxLength { max_length });
        }
        embedded.check_threshold(threshold)?;

        Self::assemble(embedded, range, threshold, max_length, backend)
    }

    /// Assembly path for [`build`](Self::build): runs cycle detection over
    /// the already-validated `range` at `threshold`, then walks
    /// representatives, computes births, deduplicates classes, and builds the
    /// containment index.
    ///
    /// `range` must already be normalized and `max_length` already validated
    /// as at least 2; `threshold` must already be at least the embedded
    /// trajectory's consecutive-distance bound and strictly below the cube
    /// side.
    #[allow(clippy::missing_panics_doc)]
    fn assemble(
        embedded: &EmbeddedTrajectory,
        range: Range<usize>,
        threshold: f64,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let trajectory = embedded.trajectory();
        let metric = embedded.metric();
        let fingerprint = embedded.fingerprint();

        let raw_components = detect_components(
            trajectory,
            metric,
            range.clone(),
            threshold,
            max_length,
            DEFAULT_OWNED_COLUMNS,
            backend,
        )?;

        // Walk one representative per component for its class. Every cycle in a
        // component carries the same class, so the shortest is chosen:
        // walk cost grows with cycle length.
        let mut component_classes: Vec<F2Vector> = Vec::with_capacity(raw_components.len());
        for cycles in &raw_components {
            let representative = cycles
                .iter()
                .min_by_key(|cycle| cycle.end - cycle.start)
                .expect("connected components are nonempty by construction");
            component_classes.push(embedded.cycle_class(representative.clone())?);
        }

        // Deduplicate classes, recording the class index for each component.
        let mut classes: Vec<F2Vector> = Vec::new();
        let mut component_class_ids: Vec<u32> = Vec::with_capacity(component_classes.len());
        for class in component_classes {
            let class_index = classes
                .iter()
                .position(|existing| existing == &class)
                .unwrap_or_else(|| {
                    classes.push(class.clone());
                    classes.len() - 1
                });
            component_class_ids
                .push(u32::try_from(class_index).expect("class table size exceeds u32::MAX"));
        }

        // Compute birth and assemble Components.
        let points = trajectory.points();
        let mut components: Vec<Component> = Vec::with_capacity(raw_components.len());
        let mut all_cycle_records: Vec<(Range<u32>, u32, f64)> = Vec::new();

        for (component_index, cycles) in raw_components.into_iter().enumerate() {
            let mut cycle_records: Vec<Cycle> = Vec::with_capacity(cycles.len());
            for cycle in cycles {
                let birth = metric.distance(points.row(cycle.start), points.row(cycle.end - 1));
                let range_u32 = u32::try_from(cycle.start).expect("cycle start exceeds u32::MAX")
                    ..u32::try_from(cycle.end).expect("cycle end exceeds u32::MAX");
                cycle_records.push(Cycle {
                    range: range_u32.clone(),
                    birth,
                });
                all_cycle_records.push((
                    range_u32,
                    u32::try_from(component_index).expect("component count exceeds u32::MAX"),
                    birth,
                ));
            }
            let coverage_start = cycle_records
                .iter()
                .map(|cycle| cycle.range.start)
                .min()
                .expect("component has at least one cycle");
            let coverage_end = cycle_records
                .iter()
                .map(|cycle| cycle.range.end)
                .max()
                .expect("component has at least one cycle");
            components.push(Component {
                class_id: component_class_ids[component_index],
                coverage: coverage_start..coverage_end,
                cycles: cycle_records,
            });
        }

        let index = IntervalSubsumptionIndex::new(all_cycle_records);
        let num_generators = embedded.cover().num_generators();

        Ok(Self {
            fingerprint,
            extent: u32::try_from(range.start).expect("extent start exceeds u32::MAX")
                ..u32::try_from(range.end).expect("extent end exceeds u32::MAX"),
            max_length: u32::try_from(max_length).expect("cycle length cap exceeds u32::MAX"),
            threshold,
            num_generators,
            classes,
            components,
            index,
        })
    }
}
