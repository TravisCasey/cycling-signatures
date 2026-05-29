// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cache of all near-recurrent cycles over a trajectory extent.

use std::{
    cmp::Reverse,
    ops::{Range, RangeBounds},
};

use chomp3rs::ExecutionBackend;
use rustc_hash::FxHashSet;

use crate::{
    EmbeddedTrajectory, F2Subspace, F2Vector,
    distance::detect_components_streaming,
    error::{Error, Result},
    metric::Metric,
    storage::interval_subsumption::IntervalSubsumptionIndex,
    util::range::normalize_segment,
};

/// A detected cycle paired with the metric distance between its endpoints.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cycle {
    range: Range<u32>,
    birth: f64,
}

impl Cycle {
    /// The cycle's half-open segment in original-index space.
    #[must_use]
    pub fn range(&self) -> Range<u32> {
        self.range.clone()
    }

    /// The metric distance between the cycle's two endpoint points.
    #[must_use]
    pub fn birth(&self) -> f64 {
        self.birth
    }

    /// The cycle's point count (`range.end - range.start`).
    #[must_use]
    pub fn length(&self) -> u32 {
        self.range.end - self.range.start
    }
}

/// One connected component of below-threshold near-recurrence, together with
/// the homology class shared by every cycle in the component.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Component {
    class_id: u32,
    coverage: Range<u32>,
    cycles: Vec<Cycle>,
}

impl Component {
    /// Index into [`CycleStorage::classes`] for this component's homology
    /// class.
    #[must_use]
    pub fn class_id(&self) -> u32 {
        self.class_id
    }

    /// The bounding interval over every cycle's `range`.
    #[must_use]
    pub fn coverage(&self) -> Range<u32> {
        self.coverage.clone()
    }

    /// Every cycle detected in this component.
    #[must_use]
    pub fn cycles(&self) -> &[Cycle] {
        &self.cycles
    }

    /// The number of cycles stored in this component.
    #[must_use]
    pub fn cycle_count(&self) -> usize {
        self.cycles.len()
    }

    /// The longest cycle in this component. Ties broken by lowest
    /// `range.start`.
    ///
    /// # Panics
    ///
    /// Panics if this component has no cycles. Components produced by
    /// [`CycleStorage::build`] always have at least one cycle.
    #[must_use]
    pub fn longest_cycle(&self) -> &Cycle {
        self.cycles
            .iter()
            .max_by_key(|cycle| (cycle.length(), Reverse(cycle.range.start)))
            .expect("component has at least one cycle")
    }

    /// The shortest cycle in this component. Ties broken by lowest
    /// `range.start`.
    ///
    /// # Panics
    ///
    /// Panics if this component has no cycles.
    #[must_use]
    pub fn shortest_cycle(&self) -> &Cycle {
        self.cycles
            .iter()
            .min_by_key(|cycle| (cycle.length(), cycle.range.start))
            .expect("component has at least one cycle")
    }
}

/// Component-first cache of near-recurrent cycles over a trajectory extent.
#[derive(Clone, Debug)]
pub struct CycleStorage {
    extent: Range<u32>,
    max_length: u32,
    threshold: f64,
    num_generators: usize,
    classes: Vec<F2Vector>,
    components: Vec<Component>,
    index: IntervalSubsumptionIndex,
}

impl CycleStorage {
    /// Builds the storage by running cycle detection over `segment`, walking
    /// one representative cycle per surviving component to obtain its
    /// homology class, computing per-cycle birth distance, applying class
    /// deduplication, and assembling the segment-containment index.
    ///
    /// # Errors
    ///
    /// - [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    ///   `0..embedded.trajectory().original_count()`.
    /// - [`Error::InvalidMaxLength`] if `max_length < 2`.
    /// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
    ///   embedded.trajectory().bound()`.
    /// - [`Error::CycleEndpointsNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative.
    /// - [`Error::ConsecutiveCubesNonAdjacent`] from
    ///   [`EmbeddedTrajectory::cycle_class`] when walking a component
    ///   representative on an `EmbeddedTrajectory` constructed via
    ///   [`EmbeddedTrajectory::from_parts`] with adjacency violations.
    #[allow(clippy::missing_panics_doc)]
    pub fn build<M: Metric>(
        embedded: &EmbeddedTrajectory<M>,
        segment: impl RangeBounds<usize>,
        threshold: f64,
        max_length: usize,
        backend: &ExecutionBackend,
    ) -> Result<Self> {
        let trajectory = embedded.trajectory();
        let range = normalize_segment(segment, trajectory.original_count())?;
        if max_length < 2 {
            return Err(Error::InvalidMaxLength { value: max_length });
        }

        // ThresholdBelowTrajectoryBound is validated inside
        // detect_components_streaming.
        let tile_width = max_length.max(1024).min(range.len()).max(max_length);
        let raw_components = detect_components_streaming(
            trajectory,
            range.clone(),
            threshold,
            max_length,
            tile_width,
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
        let metric = trajectory.metric();
        let mut components: Vec<Component> = Vec::with_capacity(raw_components.len());
        let mut all_cycle_records: Vec<(Range<u32>, u32)> = Vec::new();

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

    /// Half-open range of original indices the storage was built over.
    #[must_use]
    pub fn extent(&self) -> Range<u32> {
        self.extent.clone()
    }

    /// Adjacency threshold the storage was built with.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Cycle-length cap (point count) the storage was built with.
    #[must_use]
    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    /// Number of homology generators in the cover the storage was built from.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Deduplicated homology classes referenced by [`Component::class_id`].
    #[must_use]
    pub fn classes(&self) -> &[F2Vector] {
        &self.classes
    }

    /// All components, in implementation-defined order.
    #[must_use]
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// The component at `component_id`.
    ///
    /// # Panics
    ///
    /// Panics if `component_id >= self.components().len()`.
    #[must_use]
    pub fn component(&self, component_id: usize) -> &Component {
        &self.components[component_id]
    }

    /// The homology class for the component at `component_id`.
    ///
    /// # Panics
    ///
    /// Panics if `component_id` is out of bounds.
    #[must_use]
    pub fn class(&self, component_id: usize) -> &F2Vector {
        &self.classes[self.components[component_id].class_id as usize]
    }

    /// The `F_2` subspace spanned by classes of all components having at least
    /// one stored cycle whose range is fully contained in `segment`.
    ///
    /// # Errors
    ///
    /// [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    /// [`Self::extent`].
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(&self, segment: impl RangeBounds<usize>) -> Result<F2Subspace> {
        let range = normalize_segment(segment, self.extent.end as usize)?;
        if (range.start as u32) < self.extent.start {
            return Err(Error::WindowOutOfBounds {
                start: range.start,
                end: range.end,
                trajectory_length: self.extent.end as usize,
            });
        }
        let query = (range.start as u32)..(range.end as u32);

        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut vectors: Vec<F2Vector> = Vec::new();
        for stored in self.index.contained_in(query) {
            if seen.insert(stored.payload) {
                let component = &self.components[stored.payload as usize];
                vectors.push(self.classes[component.class_id as usize].clone());
            }
        }
        Ok(F2Subspace::new(&vectors, self.num_generators)
            .expect("class vector lengths match num_generators by construction"))
    }

    /// Component IDs whose stored cycles cover trajectory point `point`.
    ///
    /// Returns an empty vector when `point` is outside [`Self::extent`]. IDs
    /// are sorted ascending.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn components_covering(&self, point: usize) -> Vec<u32> {
        let Ok(point_u32) = u32::try_from(point) else {
            return Vec::new();
        };
        if point_u32 < self.extent.start || point_u32 >= self.extent.end {
            return Vec::new();
        }
        let mut ids: Vec<u32> = Vec::new();
        for (component_id, component) in self.components.iter().enumerate() {
            if !component.coverage.contains(&point_u32) {
                continue;
            }
            if component
                .cycles
                .iter()
                .any(|cycle| cycle.range.contains(&point_u32))
            {
                ids.push(u32::try_from(component_id).expect("component id exceeds u32::MAX"));
            }
        }
        ids
    }

    /// Constructs a [`CycleStorage`] from already-computed parts.
    ///
    /// Used by tests to assemble fixtures without going through
    /// [`Self::build`]. Recomputes the internal subsumption index from
    /// `components`.
    ///
    /// Component invariants (`coverage` bounds the cycle ranges, `class_id` is
    /// in range, cycles are non-empty) are the caller's responsibility.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_parts(
        extent: Range<u32>,
        max_length: u32,
        threshold: f64,
        num_generators: usize,
        classes: Vec<F2Vector>,
        components: Vec<Component>,
    ) -> Self {
        let mut all_cycle_records: Vec<(Range<u32>, u32)> = Vec::new();
        for (component_index, component) in components.iter().enumerate() {
            for cycle in &component.cycles {
                all_cycle_records.push((
                    cycle.range.clone(),
                    u32::try_from(component_index).expect("component count exceeds u32::MAX"),
                ));
            }
        }
        let index = IntervalSubsumptionIndex::new(all_cycle_records);
        Self {
            extent,
            max_length,
            threshold,
            num_generators,
            classes,
            components,
            index,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use chomp3rs::ExecutionBackend;
    use ndarray::Array2;

    use super::{Component, Cycle, CycleStorage};
    use crate::{EmbeddedTrajectory, F2Vector, Trajectory, error::Error, metric::Euclidean};

    /// Builds an embedded 2D trajectory that traces a recurrent loop around a
    /// missing center cube, then returns near its starting point. The cubical
    /// cover has `H^1` of rank one; cycle detection at threshold `1.5`
    /// exposes at least one non-trivial component.
    fn loop_trajectory() -> EmbeddedTrajectory<Euclidean> {
        // 9-cube ring around a missing center cube at (1, 1), traversed twice
        // so that point 0 and point 8 are at identical positions and form a
        // detectable recurrence cycle.
        let positions = [
            [0.5_f64, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [2.5, 1.5],
            [2.5, 2.5],
            [1.5, 2.5],
            [0.5, 2.5],
            [0.5, 1.5],
            [0.5, 0.5],
        ];
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let points = Array2::from_shape_vec((positions.len(), 2), flat).unwrap();
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        EmbeddedTrajectory::new(trajectory, &ExecutionBackend::Sequential).unwrap()
    }

    #[test]
    fn build_smoke() {
        let embedded = loop_trajectory();
        let storage =
            CycleStorage::build(&embedded, .., 1.5, 9, &ExecutionBackend::Sequential).unwrap();
        assert!(!storage.components().is_empty());
        for component in storage.components() {
            assert!(component.cycle_count() >= 1);
            for cycle in component.cycles() {
                assert!(cycle.length() >= 2);
                assert!(cycle.range().start < cycle.range().end);
                assert!(component.coverage().start <= cycle.range().start);
                assert!(component.coverage().end >= cycle.range().end);
            }
            assert!((component.class_id() as usize) < storage.classes().len());
        }
    }

    #[test]
    fn max_length_below_minimum_is_rejected() {
        let embedded = loop_trajectory();
        let outcome = CycleStorage::build(&embedded, .., 1.5, 1, &ExecutionBackend::Sequential);
        assert!(matches!(outcome, Err(Error::InvalidMaxLength { value: 1 })));
    }

    fn cycle_set(storage: &CycleStorage) -> BTreeSet<(u32, u32)> {
        storage
            .components()
            .iter()
            .flat_map(|component| {
                component
                    .cycles()
                    .iter()
                    .map(|cycle| (cycle.range().start, cycle.range().end))
            })
            .collect()
    }

    #[test]
    fn max_length_behavior() {
        let embedded = loop_trajectory();
        let original_count = embedded.trajectory().original_count();
        let small =
            CycleStorage::build(&embedded, .., 1.5, 4, &ExecutionBackend::Sequential).unwrap();
        let medium = CycleStorage::build(
            &embedded,
            ..,
            1.5,
            original_count,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let huge = CycleStorage::build(
            &embedded,
            ..,
            1.5,
            10 * original_count,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let medium_cycles = cycle_set(&medium);
        let huge_cycles = cycle_set(&huge);
        assert_eq!(medium_cycles, huge_cycles);

        let small_cycles = cycle_set(&small);
        assert!(small_cycles.is_subset(&medium_cycles));
        for &(start, end) in &small_cycles {
            assert!(end - start <= 4);
        }
    }

    #[test]
    fn segment_signature_equivalence_with_embedded() {
        let embedded = loop_trajectory();
        let threshold = 1.5;
        let storage = CycleStorage::build(
            &embedded,
            ..,
            threshold,
            embedded.trajectory().original_count(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        // Several segments; each should produce the same span via the
        // in-memory walker path and the storage path.
        let segments: &[std::ops::Range<usize>] =
            &[0..embedded.trajectory().original_count(), 0..4, 4..9, 2..7];
        for segment in segments {
            let embedded_span = embedded.signature(segment.clone(), threshold).unwrap();
            let storage_span = storage.signature(segment.clone()).unwrap();
            assert_eq!(embedded_span.span(), &storage_span, "segment {segment:?}");
        }
    }

    #[test]
    fn coverage_disjoint_cycles_exact_and_union() {
        // Hand-build a storage with two components:
        //  - Component 0 has cycles [10, 15) and [50, 55) (disjoint; bounding [10,
        //    55)).
        //  - Component 1 has cycle [20, 25). Distinct class.
        //
        // Point 12: inside [10, 15) only (Component 0). Rank 1.
        // Point 22: inside [20, 25) only (Component 1). Rank 1.
        // Point 30: inside Component 0's bounding interval [10, 55) but outside
        //   its actual cycles; must NOT report Component 0 (exact second-pass
        //   correctness).

        let class_zero = F2Vector::from_nonzero(2, [0]);
        let class_one = F2Vector::from_nonzero(2, [1]);

        let component_zero = Component {
            class_id: 0,
            coverage: 10..55,
            cycles: vec![
                Cycle {
                    range: 10..15,
                    birth: 0.5,
                },
                Cycle {
                    range: 50..55,
                    birth: 0.5,
                },
            ],
        };
        let component_one = Component {
            class_id: 1,
            coverage: 20..25,
            cycles: vec![Cycle {
                range: 20..25,
                birth: 0.5,
            }],
        };

        let storage = CycleStorage::from_parts(
            0..100,
            10,
            1.5,
            2,
            vec![class_zero.clone(), class_one.clone()],
            vec![component_zero, component_one],
        );

        let covering_12 = storage.components_covering(12);
        assert_eq!(covering_12, vec![0]);

        let covering_30 = storage.components_covering(30);
        assert!(
            covering_30.is_empty(),
            "point 30 is in component 0's bounding interval but no actual cycle; got \
             {covering_30:?}"
        );

        // Rank-2 union: a second fixture where two components both have an
        // active cycle covering the same point.
        let component_a = Component {
            class_id: 0,
            coverage: 0..10,
            cycles: vec![Cycle {
                range: 0..10,
                birth: 0.5,
            }],
        };
        let component_b = Component {
            class_id: 1,
            coverage: 5..15,
            cycles: vec![Cycle {
                range: 5..15,
                birth: 0.5,
            }],
        };
        let union_storage = CycleStorage::from_parts(
            0..20,
            10,
            1.5,
            2,
            vec![class_zero, class_one],
            vec![component_a, component_b],
        );
        // Point 7 is in both [0, 10) and [5, 15).
        assert_eq!(union_storage.components_covering(7), vec![0, 1]);
    }
}
