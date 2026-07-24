// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cycle storage.
//!
//! On-disk cache for all near-recurrent cycles over a trajectory extent
//! (the range of original indices the storage was built over). Cycles are
//! grouped by their connected component in the below-threshold distance
//! graph, paired with the homology class shared across the component.

mod build;
pub(crate) mod interval_subsumption;
mod serialization;

use std::{
    cmp::Reverse,
    ops::{Bound, Range, RangeBounds},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    CyclingSignature, F2Vector,
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
    /// [`threshold`](Self::threshold) is always strictly below this value;
    /// the threshold-free [`build`](Self::build) records [`f64::MAX`] as the
    /// threshold when the bound is infinite.
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
    /// [`Self::threshold`]. An unbounded start (`..end` or `..`) resolves to
    /// [`Self::extent`]'s start rather than `0`; an explicit start below the
    /// extent's start is out of bounds even at `0`.
    ///
    /// # Errors
    ///
    /// [`Error::WindowOutOfBounds`] if `segment` does not fit inside
    /// [`Self::extent`].
    #[allow(clippy::missing_panics_doc)]
    pub fn signature(&self, segment: impl RangeBounds<usize>) -> Result<CyclingSignature> {
        let start_bound = match segment.start_bound() {
            Bound::Unbounded => Bound::Included(self.extent.start as usize),
            Bound::Included(&value) => Bound::Included(value),
            Bound::Excluded(&value) => Bound::Excluded(value),
        };
        let end_bound = match segment.end_bound() {
            Bound::Included(&value) => Bound::Included(value),
            Bound::Excluded(&value) => Bound::Excluded(value),
            Bound::Unbounded => Bound::Unbounded,
        };
        let range = normalize_segment((start_bound, end_bound), self.extent.end as usize)?;
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
        // birth must not survive as a candidate). Non-finite births only
        // arise from deserialized or hand-assembled storages; a storage
        // built by `build`/`build_with_threshold` always records finite
        // metric distances.
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
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::CycleStorage;
    #[cfg(feature = "serde")]
    use crate::serialization::{load_from_reader, save_to_writer};
    use crate::{EmbeddedTrajectory, SignatureGenerator, Trajectory, error::Error, metric::Metric};

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

    /// Two square-annulus holes (missing centers at cube `(1, 1)` and cube
    /// `(7, 1)`), each an 8-cube ring, joined by a detour of fresh cubes far
    /// enough from both rings that it introduces no third recurrence. The
    /// cover has `H^1` of rank two. Ring A's cut-corner closing (samples 0
    /// and 7) births at distance `1.0`; ring B's closing point is offset
    /// within its cube so its closing (samples 18 and 25) births at distance
    /// `1.2`, distinct from ring A's.
    fn two_hole_trajectory() -> EmbeddedTrajectory {
        let positions = [
            // Ring A: 8-cube ring around missing center (1, 1). Indices 0..8.
            [0.5_f64, 0.5],
            [1.5, 0.5],
            [2.5, 0.5],
            [2.5, 1.5],
            [2.5, 2.5],
            [1.5, 2.5],
            [0.5, 2.5],
            [0.5, 1.5],
            // Detour through fresh cubes, bridging ring A to ring B far
            // enough apart that the two rings never share or approach a
            // cube. Indices 8..18.
            [-0.5, 1.5],
            [-0.5, 0.5],
            [-0.5, -0.5],
            [0.5, -0.5],
            [1.5, -0.5],
            [2.5, -0.5],
            [3.5, -0.5],
            [4.5, -0.5],
            [5.5, -0.5],
            [6.5, -0.5],
            // Ring B: 8-cube ring around missing center (7, 1), offset
            // closing point. Indices 18..26.
            [6.5, 0.5],
            [7.5, 0.5],
            [8.5, 0.5],
            [8.5, 1.5],
            [8.5, 2.5],
            [7.5, 2.5],
            [6.5, 2.5],
            [6.5, 1.7],
        ];
        let flat: Vec<f64> = positions.iter().flatten().copied().collect();
        let points = Array2::from_shape_vec((positions.len(), 2), flat).unwrap();
        let trajectory = Trajectory::new(points.view()).unwrap();
        EmbeddedTrajectory::new(trajectory, Metric::Euclidean, &ExecutionBackend::Sequential)
            .unwrap()
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
    fn signature_rank_and_span_agree_with_embedded_across_two_hole_band() {
        // Rank-2 cross-path agreement: the two-hole cover carries two
        // independent generators (num_generators() == 2), so this exercises
        // filtered structure the single-hole `loop_trajectory` fixture never
        // reaches (its cover has rank one). Ring A's cut-corner recurrence
        // births at 1.0 (equal to the trajectory's own bound, since a
        // minimal ring's cut-corner closing distance is exactly its unit
        // step size); ring B's offset closing births at 1.2. Both are
        // strictly below the empirical adjacency bound of 2.0.
        let embedded = two_hole_trajectory();
        let max_length = embedded.trajectory().original_count();
        let storage =
            CycleStorage::build(&embedded, .., max_length, &ExecutionBackend::Sequential).unwrap();

        let signature = storage.signature(..).unwrap();
        assert_eq!(signature.rank(), 2);
        let births: Vec<f64> = signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert!(
            births[0] < births[1],
            "expected ascending births: {births:?}"
        );
        assert!((births[0] - 1.0).abs() < 1e-12);
        assert!((births[1] - 1.2).abs() < 1e-12);

        let birth_one = births[0];
        let birth_two = births[1];

        // Below the first birth, the trajectory's own resolution bound
        // already excludes the threshold from the embedded path's valid
        // domain (`threshold < bound()`): only the storage side can answer
        // a query there.
        assert_eq!(
            storage
                .signature(..)
                .unwrap()
                .rank_at(birth_one.next_down())
                .unwrap(),
            0
        );

        // From `birth_one` through the top of the band, every discontinuity
        // threshold is inside the embedded path's valid domain, so both
        // paths are compared directly.
        for (threshold, expected_rank) in
            [(birth_one, 1), (birth_two.next_down(), 1), (birth_two, 2)]
        {
            let storage_signature = storage.signature(..).unwrap();
            let embedded_signature = embedded.signature_with_threshold(.., threshold).unwrap();
            assert_eq!(
                storage_signature.rank_at(threshold).unwrap(),
                expected_rank,
                "storage rank mismatch at threshold {threshold}"
            );
            assert_eq!(
                embedded_signature.rank(),
                expected_rank,
                "embedded rank mismatch at threshold {threshold}"
            );
            assert_eq!(
                storage_signature.span_at(threshold).unwrap(),
                *embedded_signature.span(),
                "span mismatch at threshold {threshold}"
            );
        }

        // The theorem holds exactly for the threshold-free paths too: same
        // birth sequence, both sides.
        let embedded_signature = embedded.signature(..).unwrap();
        let embedded_births: Vec<f64> = embedded_signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert_eq!(births, embedded_births);
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
    fn signature_unbounded_start_resolves_to_extent_start() {
        // A storage built over a subsegment of the trajectory: `signature(..)`
        // must resolve the unbounded start to the storage's own extent start,
        // not to sample index 0, while an explicit `0..` still falls outside
        // the extent and errors.
        let embedded = loop_trajectory();
        let sub_segment = 4_usize..9_usize;
        let storage = CycleStorage::build(
            &embedded,
            sub_segment.clone(),
            sub_segment.len(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let unbounded_signature = storage.signature(..).unwrap();
        let explicit_signature = storage.signature(sub_segment.clone()).unwrap();
        assert_eq!(unbounded_signature.span(), explicit_signature.span());
        let unbounded_births: Vec<f64> = unbounded_signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        let explicit_births: Vec<f64> = explicit_signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert_eq!(unbounded_births, explicit_births);

        let error = storage.signature(0..sub_segment.end).unwrap_err();
        assert!(matches!(error, Error::WindowOutOfBounds { start: 0, .. }));
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

        // Both paths derive from the same deterministic pipeline, so the
        // comparison is exact. Both also agree with the standalone
        // `adjacency_bound` sweep computed above.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(free.threshold(), capped.threshold());
            assert_eq!(free.adjacency_bound(), capped.adjacency_bound());
            assert_eq!(free.adjacency_bound(), empirical_bound);
            assert_eq!(capped.adjacency_bound(), empirical_bound);
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
        let loaded_storage = load_from_reader::<CycleStorage, _>(&storage_buffer[..]).unwrap();

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

        let loaded = load_from_reader::<CycleStorage, _>(&buffer[..]).unwrap();

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

        // The same holds for the threshold-free direct path: with no
        // non-adjacent candidate pair anywhere in the window, detection runs
        // unconstrained and the signature's band top is the sentinel value.
        let signature = embedded.signature(..).unwrap();
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(signature.threshold_max(), f64::MAX);
        }

        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &storage).unwrap();
        let loaded = load_from_reader::<CycleStorage, _>(&buffer[..]).unwrap();
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(loaded.adjacency_bound(), f64::INFINITY);
            assert_eq!(loaded.threshold(), f64::MAX);
        }
    }
}
