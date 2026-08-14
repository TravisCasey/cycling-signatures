// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Construction pipeline for [`CycleStorage`]: cycle detection over a
//! trajectory segment, followed by per-component class deduplication and
//! assembly.

use std::ops::{Range, RangeBounds};

use chomp3rs::ExecutionBackend;
use rustc_hash::FxHashMap;

use super::{Component, Cycle, CycleStorage};
use crate::{
    EmbeddedTrajectory, F2Vector,
    distance::{DEFAULT_OWNED_COLUMNS, detect_components},
    error::{Error, Result},
    storage::interval_subsumption::IntervalSubsumptionIndex,
    util::range::normalize_segment,
};

impl CycleStorage {
    /// Builds the storage by running cycle detection over `segment`, storing
    /// each surviving component's homology class and per-cycle birth.
    ///
    /// Detected cycles have endpoints with metric distance strictly below 1,
    /// the cube side length.
    ///
    /// # Errors
    ///
    /// - [`Error::SegmentOutOfBounds`] if `segment` does not fit inside
    ///   `0..embedded.trajectory().len()`.
    /// - [`Error::MaxLengthBelowMinimum`] if `max_length < 2`.
    /// - [`Error::CycleEndpointsNonAdjacent`] if a detected cycle's endpoint
    ///   cubes differ by more than 1 in some axis.
    pub fn build(
        embedded: &EmbeddedTrajectory,
        segment: impl RangeBounds<usize>,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let range = normalize_segment(segment, embedded.trajectory().len())?;
        if max_length < 2 {
            return Err(Error::MaxLengthBelowMinimum { max_length });
        }

        Self::assemble(embedded, range, max_length, backend)
    }

    /// Assembly path for [`build`](Self::build): runs cycle detection over
    /// the already-validated `range`, then walks representatives, computes
    /// births, deduplicates classes, and builds the containment index.
    ///
    /// `range` must already be normalized and `max_length` already validated
    /// as at least 2.
    fn assemble(
        embedded: &EmbeddedTrajectory,
        range: Range<usize>,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let fingerprint = embedded.fingerprint();
        let metric_points = embedded.metric_points();

        let raw_components = detect_components(
            &metric_points,
            range.clone(),
            max_length,
            DEFAULT_OWNED_COLUMNS,
            backend,
        )?;

        let component_classes = embedded.component_classes(&raw_components)?;

        // Deduplicate classes, recording the class id for each component. Ids
        // are assigned in first-encounter order, which is the order the class
        // table is serialized in and therefore the order every stored
        // `class_id` is written against.
        let mut class_ids: FxHashMap<F2Vector, u32> = FxHashMap::default();
        let mut component_class_ids: Vec<u32> = Vec::with_capacity(component_classes.len());
        for class in component_classes {
            let next_id =
                u32::try_from(class_ids.len()).expect("class table size exceeds u32::MAX");
            component_class_ids.push(*class_ids.entry(class).or_insert(next_id));
        }
        let mut ordered_classes: Vec<(u32, F2Vector)> = class_ids
            .into_iter()
            .map(|(class, class_id)| (class_id, class))
            .collect();
        ordered_classes.sort_unstable_by_key(|&(class_id, _)| class_id);
        let classes: Vec<F2Vector> = ordered_classes
            .into_iter()
            .map(|(_, class)| class)
            .collect();

        // Compute birth and assemble Components.
        let mut components: Vec<Component> = Vec::with_capacity(raw_components.len());
        let mut all_cycle_records: Vec<(Range<u32>, u32, f64)> = Vec::new();

        for (component_index, cycles) in raw_components.into_iter().enumerate() {
            let mut cycle_records: Vec<Cycle> = Vec::with_capacity(cycles.len());
            for cycle in cycles {
                let birth = metric_points.distance(cycle.start, cycle.end - 1);
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
            num_generators,
            classes,
            components,
            index,
        })
    }
}
