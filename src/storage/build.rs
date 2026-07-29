// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Construction pipeline for [`CycleStorage`]: threshold-free and
//! explicit-threshold cycle detection over a trajectory segment, followed by
//! per-component class deduplication and assembly.

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
    /// Builds the storage by running cycle detection over `segment` at the
    /// largest threshold that keeps every admitted candidate pair in adjacent
    /// cubes.
    ///
    /// Runs the segment's empirical adjacency-bound sweep over candidate
    /// pairs within `max_length`, then detects cycles at the largest
    /// threshold strictly below that bound
    /// (its [`next_down`](f64::next_down)). The returned storage's
    /// [`threshold`](Self::threshold) is that effective threshold, and
    /// [`adjacency_bound`](Self::adjacency_bound) records the bound itself.
    /// For an explicit threshold instead, use
    /// [`build_with_threshold`](Self::build_with_threshold).
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    ///   `0..embedded.trajectory().original_count()`.
    /// - [`Error::InvalidMaxLength`] if `max_length < 2`.
    /// - [`Error::EmptyThresholdBand`] if the segment's empirical adjacency
    ///   bound does not exceed `embedded.bound()`: no threshold admits a
    ///   recurrence there.
    /// - [`Error::CycleEndpointsNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative on an `EmbeddedTrajectory` constructed via
    ///   [`EmbeddedTrajectory::from_parts`] with adjacency violations.
    pub fn build(
        embedded: &EmbeddedTrajectory,
        segment: impl RangeBounds<usize>,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let range = normalize_segment(segment, embedded.trajectory().original_count())?;
        if max_length < 2 {
            return Err(Error::InvalidMaxLength { max_length });
        }

        let threshold =
            embedded.threshold_free_detection_threshold(range.clone(), max_length, backend)?;

        Self::assemble(embedded, range, threshold, max_length, backend)
    }

    /// Builds the storage by running cycle detection over `segment` at an
    /// explicit adjacency `threshold`, storing each surviving component's
    /// homology class and per-cycle birth.
    ///
    /// The returned storage's [`threshold`](Self::threshold) is `threshold`
    /// itself, and [`adjacency_bound`](Self::adjacency_bound) records the
    /// segment's empirical adjacency bound as provenance. For the largest
    /// threshold that keeps every admitted candidate pair in adjacent cubes
    /// instead of an explicit value, use [`build`](Self::build).
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    ///   `0..embedded.trajectory().original_count()`.
    /// - [`Error::InvalidMaxLength`] if `max_length < 2`.
    /// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
    ///   embedded.bound()`.
    /// - [`Error::ThresholdExceedsAdjacencyBound`] if `threshold` admits a
    ///   candidate endpoint pair in non-adjacent cubes (at or above the
    ///   window's [`adjacency_bound`](EmbeddedTrajectory::adjacency_bound)).
    /// - [`Error::CycleEndpointsNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative on an `EmbeddedTrajectory` constructed via
    ///   [`EmbeddedTrajectory::from_parts`] with adjacency violations.
    pub fn build_with_threshold(
        embedded: &EmbeddedTrajectory,
        segment: impl RangeBounds<usize>,
        threshold: f64,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let range = normalize_segment(segment, embedded.trajectory().original_count())?;
        if max_length < 2 {
            return Err(Error::InvalidMaxLength { max_length });
        }
        embedded.check_threshold(threshold)?;

        Self::assemble(embedded, range, threshold, max_length, backend)
    }

    /// Shared assembly path for [`build`](Self::build) and
    /// [`build_with_threshold`](Self::build_with_threshold): runs cycle
    /// detection over the already-validated `range` at `threshold`, then
    /// walks representatives, computes births, deduplicates classes, and
    /// builds the containment index.
    ///
    /// `range` must already be normalized and `max_length` already validated
    /// as at least 2; `threshold` must already be at least the embedded
    /// trajectory's consecutive-distance bound and strictly below every
    /// non-adjacent candidate pair's distance in `range`.
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

        let (raw_components, adjacency_bound) = detect_components(
            trajectory,
            metric,
            range.clone(),
            threshold,
            max_length,
            DEFAULT_OWNED_COLUMNS,
            backend,
        )?;

        // Walk one representative per component for its class.
        let mut component_classes: Vec<F2Vector> = Vec::with_capacity(raw_components.len());
        for cycles in &raw_components {
            let representative = &cycles[0];
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
        let original_indices = trajectory.original_indices();
        let mut components: Vec<Component> = Vec::with_capacity(raw_components.len());
        let mut all_cycle_records: Vec<(Range<u32>, u32, f64)> = Vec::new();

        for (component_index, cycles) in raw_components.into_iter().enumerate() {
            let mut cycle_records: Vec<Cycle> = Vec::with_capacity(cycles.len());
            for cycle in cycles {
                let birth = metric.distance(
                    points.row(original_indices[cycle.start]),
                    points.row(original_indices[cycle.end - 1]),
                );
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
            adjacency_bound,
            num_generators,
            classes,
            components,
            index,
        })
    }
}
