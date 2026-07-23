// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cache of all near-recurrent cycles over a trajectory extent.

#[cfg(feature = "serde")]
use std::path::Path;
use std::{
    cmp::Reverse,
    ops::{Range, RangeBounds},
};

use chomp3rs::ExecutionBackend;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
use crate::serialization::{load_from_path, save_to_path};
use crate::{
    CyclingSignature, EmbeddedTrajectory, F2Vector,
    distance::{adjacency_bound_streaming, detect_components_streaming, detection_tile_width},
    error::{Error, Result},
    storage::interval_subsumption::IntervalSubsumptionIndex,
    util::range::normalize_segment,
};

/// A detected cycle paired with the metric distance between its endpoints.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    fingerprint: u64,
    extent: Range<u32>,
    max_length: u32,
    threshold: f64,
    adjacency_bound: f64,
    num_generators: usize,
    classes: Vec<F2Vector>,
    components: Vec<Component>,
    index: IntervalSubsumptionIndex,
}

impl CycleStorage {
    /// Builds the storage by running cycle detection over `segment` at the
    /// largest threshold that keeps every admitted candidate pair in adjacent
    /// cubes.
    ///
    /// Runs the segment's whole-band empirical adjacency bound sweep, then
    /// detects cycles at the largest threshold strictly below that bound
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

        let tile_width = detection_tile_width(range.len(), max_length);
        let empirical_bound = adjacency_bound_streaming(
            embedded.trajectory(),
            embedded.metric(),
            range.clone(),
            max_length,
            tile_width,
            backend,
        )?;
        if empirical_bound <= embedded.bound() {
            return Err(Error::EmptyThresholdBand {
                trajectory_bound: embedded.bound(),
                adjacency_bound: empirical_bound,
            });
        }

        Self::assemble(
            embedded,
            range,
            empirical_bound.next_down(),
            max_length,
            backend,
        )
    }

    /// Builds the storage by running cycle detection over `segment` at an
    /// explicit adjacency threshold, walking one representative cycle per
    /// surviving component to obtain its homology class, computing per-cycle
    /// birth distance, applying class deduplication, and assembling the
    /// segment-containment index.
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
        embedded.check_threshold(threshold)?;
        let range = normalize_segment(segment, embedded.trajectory().original_count())?;
        if max_length < 2 {
            return Err(Error::InvalidMaxLength { max_length });
        }

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

        let tile_width = detection_tile_width(range.len(), max_length);
        let (raw_components, adjacency_bound) = detect_components_streaming(
            trajectory,
            metric,
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

    /// Half-open range of original indices the storage was built over.
    #[must_use]
    pub fn extent(&self) -> Range<u32> {
        self.extent.clone()
    }

    /// The fingerprint of the embedded trajectory this storage was built from.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Inclusive upper end of the storage's valid query band (the effective
    /// detection threshold).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The empirical adjacency bound of the band this storage was built
    /// over: the smallest metric distance between two candidate endpoint
    /// samples in the build's segment whose cubes are not adjacent, or
    /// positive infinity if every candidate pair was adjacent.
    ///
    /// [`threshold`](Self::threshold) is always strictly below this value
    /// (or, for an infinite bound, equal to [`f64::MAX`]).
    #[must_use]
    pub fn adjacency_bound(&self) -> f64 {
        self.adjacency_bound
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

    /// The filtered cycling signature spanned by classes of all components
    /// having at least one stored cycle whose range is fully contained in
    /// `segment`.
    ///
    /// Each contributing component's birth is the minimum birth over its
    /// cycles contained in `segment`; the signature is complete up to
    /// [`Self::threshold`].
    ///
    /// # Errors
    ///
    /// [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    /// [`Self::extent`].
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(&self, segment: impl RangeBounds<usize>) -> Result<CyclingSignature> {
        let range = normalize_segment(segment, self.extent.end as usize)?;
        if (range.start as u32) < self.extent.start {
            return Err(Error::WindowOutOfBounds {
                start: range.start,
                end: range.end,
                trajectory_length: self.extent.end as usize,
            });
        }
        let query = (range.start as u32)..(range.end as u32);

        // One candidate per contributing component at its minimum finite
        // birth, in ascending component id: sorting by component id keeps
        // candidate order deterministic and independent of the index's
        // (begin, end) iteration order, and the finiteness filter drops
        // components with no finite birth before the fold (a lone non-finite
        // birth must not survive as a candidate).
        let mut admitted: Vec<(u32, f64)> = self
            .index
            .contained_in(query)
            .map(|stored| (stored.payload, stored.birth))
            .filter(|(_, birth)| birth.is_finite())
            .collect();
        admitted.sort_unstable_by_key(|&(component_id, _)| component_id);
        let mut component_minima: Vec<(u32, f64)> = Vec::new();
        for (component_id, birth) in admitted {
            match component_minima.last_mut() {
                Some((last_component_id, last_birth)) if *last_component_id == component_id => {
                    *last_birth = last_birth.min(birth);
                },
                _ => component_minima.push((component_id, birth)),
            }
        }
        let candidates = component_minima
            .into_iter()
            .map(|(component_id, birth)| {
                (
                    birth,
                    self.classes[self.components[component_id as usize].class_id as usize].clone(),
                )
            })
            .collect();
        Ok(CyclingSignature::from_candidates(
            candidates,
            self.num_generators,
            self.threshold,
        ))
    }

    /// Component IDs whose stored cycles cover sample index `point`.
    ///
    /// `point` is a sample index in original-index space, consistent with the
    /// original indices stored in [`Self::extent`]. Returns an empty vector
    /// when `point` is outside [`Self::extent`]. IDs are sorted ascending.
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

    /// Assembles a [`CycleStorage`] from already-computed parts, recomputing
    /// the internal subsumption index from `components`.
    ///
    /// Component invariants (`coverage` bounds the cycle ranges, `class_id` is
    /// in range, cycles are non-empty) are the caller's responsibility.
    #[cfg(any(test, feature = "serde"))]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        fingerprint: u64,
        extent: Range<u32>,
        max_length: u32,
        threshold: f64,
        adjacency_bound: f64,
        num_generators: usize,
        classes: Vec<F2Vector>,
        components: Vec<Component>,
    ) -> Self {
        let mut all_cycle_records: Vec<(Range<u32>, u32, f64)> = Vec::new();
        for (component_index, component) in components.iter().enumerate() {
            for cycle in &component.cycles {
                all_cycle_records.push((
                    cycle.range.clone(),
                    u32::try_from(component_index).expect("component count exceeds u32::MAX"),
                    cycle.birth,
                ));
            }
        }
        let index = IntervalSubsumptionIndex::new(all_cycle_records);
        Self {
            fingerprint,
            extent,
            max_length,
            threshold,
            adjacency_bound,
            num_generators,
            classes,
            components,
            index,
        }
    }

    /// Writes this storage to `path` in the crate's binary format.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_to_path(path, self)
    }

    /// Reads a storage written by [`save`](Self::save).
    ///
    /// The returned storage carries the fingerprint of the embedded trajectory
    /// it was built from; compare it against
    /// [`EmbeddedTrajectory::fingerprint`] to confirm provenance.
    ///
    /// # Errors
    ///
    /// - [`Error::FormatVersionMismatch`] if the file's format version differs.
    /// - [`Error::Io`] if the file could not be opened.
    /// - [`Error::Deserialize`] if the file contents could not be read and
    ///   decoded.
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load_from_path(path)
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct CycleStorageData {
    fingerprint: u64,
    extent: Range<u32>,
    max_length: u32,
    threshold: f64,
    adjacency_bound: f64,
    num_generators: usize,
    classes: Vec<F2Vector>,
    components: Vec<Component>,
}

#[cfg(feature = "serde")]
impl Serialize for CycleStorage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        CycleStorageData {
            fingerprint: self.fingerprint,
            extent: self.extent.clone(),
            max_length: self.max_length,
            threshold: self.threshold,
            adjacency_bound: self.adjacency_bound,
            num_generators: self.num_generators,
            classes: self.classes.clone(),
            components: self.components.clone(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for CycleStorage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = CycleStorageData::deserialize(deserializer)?;
        Ok(Self::from_parts(
            data.fingerprint,
            data.extent,
            data.max_length,
            data.threshold,
            data.adjacency_bound,
            data.num_generators,
            data.classes,
            data.components,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::{Component, Cycle, CycleStorage};
    use crate::{EmbeddedTrajectory, F2Vector, Trajectory, error::Error, metric::Metric};
    #[cfg(feature = "serde")]
    use crate::{SignatureGenerator, serialization::save_to_writer};

    /// Builds an embedded 2D trajectory that traces a recurrent loop around a
    /// missing center cube, then returns near its starting point. The cubical
    /// cover has `H^1` of rank one; cycle detection at threshold `1.5`
    /// exposes at least one non-trivial component.
    fn loop_trajectory() -> EmbeddedTrajectory {
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
        let trajectory = Trajectory::new(points.view()).unwrap();
        EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::Sequential)
            .unwrap()
    }

    /// Builds the small 4-point Euclidean square loop, whose consecutive and
    /// diagonal cube pairs are all adjacent: every candidate endpoint pair
    /// within `max_length` shares an adjacent cube, so the band has no upper
    /// constraint.
    #[cfg(feature = "serde")]
    fn euclidean_square_loop() -> EmbeddedTrajectory {
        let points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::Sequential)
            .unwrap()
    }

    #[test]
    fn build_smoke() {
        let embedded = loop_trajectory();
        let storage = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            1.5,
            9,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
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
        let outcome = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            1.5,
            1,
            &ExecutionBackend::Sequential,
        );
        assert!(matches!(
            outcome,
            Err(Error::InvalidMaxLength { max_length: 1 })
        ));
    }

    #[test]
    fn threshold_below_trajectory_bound_is_rejected() {
        // The loop_trajectory fixture has consecutive spacing of 1.0 (adjacent
        // cube centers), so bound() == 1.0. A threshold of 0.5 is below that.
        let embedded = loop_trajectory();
        let trajectory_bound = embedded.bound();
        let lower_threshold = trajectory_bound - 0.5;
        let outcome = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            lower_threshold,
            9,
            &ExecutionBackend::Sequential,
        );
        assert!(matches!(
            outcome,
            Err(Error::ThresholdBelowTrajectoryBound { threshold, trajectory_bound: bound })
                if (threshold - lower_threshold).abs() < 1e-12
                    && (bound - embedded.bound()).abs() < 1e-12
        ));
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
        let small = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            1.5,
            4,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let medium = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            1.5,
            original_count,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let huge = CycleStorage::build_with_threshold(
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
    fn signature_rank_and_span_agree_with_embedded_across_thresholds() {
        // Path agreement: filtering a threshold-free storage's signature down
        // to a threshold `t` must report the same rank and span as detecting
        // cycles directly at `t` through the in-memory walker, for every
        // segment and every threshold inside the valid band.
        let embedded = loop_trajectory();
        let max_length = embedded.trajectory().original_count();
        let storage =
            CycleStorage::build(&embedded, .., max_length, &ExecutionBackend::Sequential).unwrap();
        let trajectory_bound = embedded.bound();
        let band_top = storage.threshold();

        let segments: &[Range<usize>] = &[0..max_length, 0..4, 4..9, 2..7];
        let thresholds = [
            trajectory_bound,
            f64::midpoint(trajectory_bound, band_top),
            band_top,
        ];

        for segment in segments {
            let storage_signature = storage.signature(segment.clone()).unwrap();
            for &threshold in &thresholds {
                let embedded_signature = embedded
                    .signature_with_threshold(segment.clone(), threshold)
                    .unwrap();
                assert_eq!(
                    storage_signature.rank_at(threshold).unwrap(),
                    embedded_signature.rank(),
                    "rank mismatch for segment {segment:?} at threshold {threshold}"
                );
                let storage_span = storage_signature.span_at(threshold).unwrap();
                assert_eq!(
                    &storage_span,
                    embedded_signature.span(),
                    "span mismatch for segment {segment:?} at threshold {threshold}"
                );
            }
        }
    }

    #[test]
    fn window_signature_reports_per_component_minimum_birth() {
        // Component 0 has two cycles: the wide [10, 60) at birth 0.1 and the
        // narrow [40, 45) at birth 0.5. The container survives the
        // birth-aware subsumption dedup because its birth (0.1) is smaller
        // than that of the interval it contains (0.5). A window admitting
        // both cycles reports the smaller birth; a window admitting only the
        // narrow cycle reports its own, larger birth.
        let class = F2Vector::from_nonzero(1, [0]);
        let component = Component {
            class_id: 0,
            coverage: 10..60,
            cycles: vec![
                Cycle {
                    range: 10..60,
                    birth: 0.1,
                },
                Cycle {
                    range: 40..45,
                    birth: 0.5,
                },
            ],
        };
        let storage =
            CycleStorage::from_parts(0, 0..100, 60, 1.5, 2.0, 1, vec![class], vec![component]);

        let wide = storage.signature(0..100).unwrap();
        assert_eq!(wide.generators().len(), 1);
        assert!((wide.generators()[0].birth() - 0.1).abs() < 1e-12);

        let narrow = storage.signature(35..50).unwrap();
        assert_eq!(narrow.generators().len(), 1);
        assert!((narrow.generators()[0].birth() - 0.5).abs() < 1e-12);
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
            0,
            0..100,
            10,
            1.5,
            2.0,
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
            0,
            0..20,
            10,
            1.5,
            2.0,
            2,
            vec![class_zero, class_one],
            vec![component_a, component_b],
        );
        // Point 7 is in both [0, 10) and [5, 15).
        assert_eq!(union_storage.components_covering(7), vec![0, 1]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_round_trip_preserves_queries() {
        let embedded = loop_trajectory();
        let threshold = 1.5;
        let max_length = embedded.trajectory().original_count();
        let storage = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            threshold,
            max_length,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let mut storage_buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut storage_buffer, &storage).unwrap();
        let loaded_storage =
            crate::serialization::load_from_reader::<CycleStorage, _>(&storage_buffer[..]).unwrap();

        // The reloaded storage carries the same provenance fingerprint, so a
        // caller can confirm it against the embedded trajectory.
        assert_eq!(loaded_storage.fingerprint(), embedded.fingerprint());

        let segments: &[Range<usize>] =
            &[0..embedded.trajectory().original_count(), 0..4, 4..9, 2..7];
        for segment in segments {
            // `CyclingSignature` has no equality of its own, so compare its
            // full-band span, its per-generator births, and its band top
            // instead.
            let signature = storage.signature(segment.clone()).unwrap();
            let loaded_signature = loaded_storage.signature(segment.clone()).unwrap();
            assert_eq!(
                signature.span(),
                loaded_signature.span(),
                "span differs after round-trip for segment {segment:?}",
            );
            let births: Vec<f64> = signature
                .generators()
                .iter()
                .map(SignatureGenerator::birth)
                .collect();
            let loaded_births: Vec<f64> = loaded_signature
                .generators()
                .iter()
                .map(SignatureGenerator::birth)
                .collect();
            assert_eq!(
                births, loaded_births,
                "births differ after round-trip for segment {segment:?}",
            );
            assert!(
                signature
                    .threshold_max()
                    .total_cmp(&loaded_signature.threshold_max())
                    .is_eq(),
                "band top differs after round-trip for segment {segment:?}",
            );
        }
        for point in 0..embedded.trajectory().original_count() {
            assert_eq!(
                storage.components_covering(point),
                loaded_storage.components_covering(point),
                "coverage differs after round-trip at point {point}",
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn loaded_fingerprint_distinguishes_embedded() {
        let embedded = loop_trajectory();
        let storage = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            1.5,
            9,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &storage).unwrap();

        let loaded =
            crate::serialization::load_from_reader::<CycleStorage, _>(&buffer[..]).unwrap();

        // The matching embedded shares the loaded fingerprint.
        assert_eq!(loaded.fingerprint(), embedded.fingerprint());

        // A different embedded trajectory (different points) has a different
        // fingerprint, so a provenance check would reject it.
        let other_points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let other_trajectory = Trajectory::new(other_points.view()).unwrap();
        let other_embedded = EmbeddedTrajectory::new(
            other_trajectory,
            Metric::Euclidean,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        assert_ne!(loaded.fingerprint(), other_embedded.fingerprint());
    }

    #[test]
    fn build_free_and_capped_agree_at_empirical_bound() {
        let embedded = loop_trajectory();
        let max_length = embedded.trajectory().original_count();
        let empirical_bound = embedded
            .adjacency_bound(.., max_length, &ExecutionBackend::Sequential)
            .unwrap();

        let free =
            CycleStorage::build(&embedded, .., max_length, &ExecutionBackend::Sequential).unwrap();
        let capped = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            empirical_bound.next_down(),
            max_length,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        // Both paths derive from the same deterministic pipeline (the free
        // path's threshold is exactly the capped path's input, and both report
        // the same pass-2 piggybacked bound), so the comparison is exact.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(free.threshold(), capped.threshold());
            assert_eq!(free.adjacency_bound(), capped.adjacency_bound());
        }
        assert_eq!(free.classes(), capped.classes());
        assert_eq!(cycle_set(&free), cycle_set(&capped));
    }

    #[test]
    fn build_rejects_empty_threshold_band() {
        // Every consecutive step lands in an adjacent cube, but the smallest
        // non-adjacent candidate pair distance (samples 0 and 2) does not
        // exceed the trajectory's own consecutive-distance bound (samples 2
        // and 3), so no threshold admits a recurrence.
        let points = array![[0.501], [1.999], [2.001], [3.999]];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded =
            EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::Sequential)
                .unwrap();

        let outcome = CycleStorage::build(&embedded, .., 4, &ExecutionBackend::Sequential);
        assert!(matches!(outcome, Err(Error::EmptyThresholdBand { .. })));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn build_on_unconstrained_band_is_infinite_and_round_trips() {
        let embedded = euclidean_square_loop();
        let storage = CycleStorage::build(&embedded, .., 4, &ExecutionBackend::Sequential).unwrap();

        // Every candidate pair in this fixture is cube-adjacent, so the bound
        // and its derived threshold take on exact sentinel values that
        // round-trip unchanged through serde.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(storage.adjacency_bound(), f64::INFINITY);
            assert_eq!(storage.threshold(), f64::MAX);
        }

        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &storage).unwrap();
        let loaded =
            crate::serialization::load_from_reader::<CycleStorage, _>(&buffer[..]).unwrap();
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(loaded.adjacency_bound(), f64::INFINITY);
            assert_eq!(loaded.threshold(), f64::MAX);
        }
    }

    #[test]
    fn build_adjacency_bound_matches_standalone_computation() {
        let embedded = loop_trajectory();
        let max_length = embedded.trajectory().original_count();
        let standalone_bound = embedded
            .adjacency_bound(.., max_length, &ExecutionBackend::Sequential)
            .unwrap();

        let free =
            CycleStorage::build(&embedded, .., max_length, &ExecutionBackend::Sequential).unwrap();

        let capped = CycleStorage::build_with_threshold(
            &embedded,
            ..,
            standalone_bound.next_down(),
            max_length,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        // Both builds' pass-2 piggybacked minimum equals the standalone
        // pass-1 sweep exactly: same pair set, same pure minimum.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(free.adjacency_bound(), standalone_bound);
            assert_eq!(capped.adjacency_bound(), standalone_bound);
        }
    }
}
