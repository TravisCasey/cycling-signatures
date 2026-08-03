// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cycle storage.
//!
//! On-disk cache for all near-recurrent cycles over a trajectory extent
//! (the range of trajectory points the storage was built over). Cycles are
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
    /// The cycle's half-open segment of trajectory points.
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
    num_generators: usize,
    classes: Vec<F2Vector>,
    components: Vec<Component>,
    index: IntervalSubsumptionIndex,
}

impl CycleStorage {
    /// Half-open range of trajectory points the storage was built over.
    #[must_use]
    pub fn extent(&self) -> Range<u32> {
        self.extent.clone()
    }

    /// The fingerprint of the embedded trajectory this storage was built from.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The adjacency threshold detection ran at.
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

    /// All components, ordered by their least cycle under `(start, end)`, with
    /// each component's cycles in that order.
    ///
    /// No two components share a cycle, so the order is determined by the
    /// partition alone.
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

        // One entry per contributing component at its minimum finite birth,
        // in ascending component id: sorting by component id keeps the entry
        // order deterministic and independent of the index's (begin, end)
        // iteration order, and the finiteness filter drops components with no
        // finite birth before the fold.
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
        let births = component_minima
            .into_iter()
            .map(|(component_id, birth)| {
                (
                    birth,
                    self.classes[self.components[component_id as usize].class_id as usize].clone(),
                )
            })
            .collect();
        Ok(CyclingSignature::from_births(
            births,
            self.num_generators,
            self.threshold,
        ))
    }

    /// Component IDs whose stored cycles cover the trajectory point `point`.
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
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use chomp3rs::ExecutionBackend;
    use ndarray::Array2;
    #[cfg(feature = "serde")]
    use ndarray::array;

    use super::CycleStorage;
    #[cfg(feature = "serde")]
    use crate::serialization::{load_from_reader, save_to_writer};
    use crate::{
        CubicalCover, EmbeddedTrajectory, Metric, SignatureGenerator, Trajectory,
        error::{Error, Result},
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

    /// The center point of each cube in `cubes`.
    fn cube_centers(cubes: &[(i32, i32)]) -> Vec<[f64; 2]> {
        cubes
            .iter()
            .map(|&(x, y)| [f64::from(x) + 0.5, f64::from(y) + 0.5])
            .collect()
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

    /// Builds an embedded 2D trajectory that densely traces a recurrent loop
    /// around a missing center cube, then returns exactly to its starting
    /// point. The cubical cover has `H^1` of rank one; the closing recurrence
    /// births at distance `0`.
    fn loop_trajectory() -> EmbeddedTrajectory {
        let points = densify_path(&ring_waypoints(), 0.4);
        let trajectory = Trajectory::new(points.view()).unwrap();
        embed_euclidean(trajectory).unwrap()
    }

    /// Two square-annulus holes (missing centers at cube `(1, 1)` and cube
    /// `(5, 1)`), each an 8-cube ring, joined by a detour through fresh cubes
    /// far enough from both rings that it introduces no third recurrence. The
    /// cover has `H^1` of rank two. Ring A's cut-corner closing births at
    /// distance `0.8`; ring B's closing point is offset within its cube so
    /// its closing births at distance `0.9`, distinct from ring A's.
    fn two_hole_trajectory() -> EmbeddedTrajectory {
        let first_ring_centers = cube_centers(&[
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
        ]);
        let first_closing_point = [0.5, 1.3];

        let second_ring_centers = cube_centers(&[
            (4, 0),
            (5, 0),
            (6, 0),
            (6, 1),
            (6, 2),
            (5, 2),
            (4, 2),
            (4, 1),
        ]);
        let second_closing_point = [4.5, 1.4];

        // Combined waypoint path: the first ring, its cut-corner closing
        // point, a wide detour well clear of both rings' bounding boxes
        // (avoiding both missing cubes and staying far enough from either
        // ring that no shorter accidental recurrence forms), the second ring
        // (whose first waypoint continues directly from the detour), and the
        // second ring's own cut-corner closing point.
        let mut waypoints = first_ring_centers;
        waypoints.push(first_closing_point);
        waypoints.push([-2.0, 1.3]);
        waypoints.push([-2.0, -2.0]);
        waypoints.push([4.5, -2.0]);
        waypoints.extend_from_slice(&second_ring_centers);
        waypoints.push(second_closing_point);

        let points = densify_path(&waypoints, 0.4);
        let trajectory = Trajectory::new(points.view()).unwrap();
        embed_euclidean(trajectory).unwrap()
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
        // births at 0.8; ring B's offset closing births at 0.9.
        let embedded = two_hole_trajectory();
        let max_length = embedded.trajectory().len();
        let band_top = 0.95;
        let storage = CycleStorage::build(
            &embedded,
            ..,
            max_length,
            band_top,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let signature = storage.signature(..).unwrap();
        assert_eq!(signature.rank(), 2);
        assert_eq!(signature.num_generators(), 2);
        let births: Vec<f64> = signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert!(
            births[0] < births[1],
            "expected ascending births: {births:?}"
        );
        assert!((births[0] - 0.8).abs() < 1e-12);
        assert!((births[1] - 0.9).abs() < 1e-12);

        let birth_one = births[0];
        let birth_two = births[1];

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
            let embedded_signature = embedded.signature(.., threshold).unwrap();
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
    }

    #[test]
    fn signature_rank_and_span_agree_with_embedded_across_thresholds() {
        // Path agreement: filtering a storage's signature down to a
        // threshold `t` must report the same rank and span as detecting
        // cycles directly at `t` through the in-memory walker, for every
        // segment and every threshold inside the valid band.
        let embedded = loop_trajectory();
        let max_length = embedded.trajectory().len();
        let resolution = embedded.resolution();
        let band_top = 0.95;
        let storage = CycleStorage::build(
            &embedded,
            ..,
            max_length,
            band_top,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let segments: &[Range<usize>] = &[0..max_length, 0..8, 8..max_length, 4..13];
        let thresholds = [resolution, f64::midpoint(resolution, band_top), band_top];

        for segment in segments {
            let storage_signature = storage.signature(segment.clone()).unwrap();
            for &threshold in &thresholds {
                let embedded_signature = embedded.signature(segment.clone(), threshold).unwrap();
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
        // not to point index 0, while an explicit `0..` still falls outside
        // the extent and errors.
        let embedded = loop_trajectory();
        let sub_segment = 4_usize..13_usize;
        let storage = CycleStorage::build(
            &embedded,
            sub_segment.clone(),
            sub_segment.len(),
            0.95,
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
        let storage =
            CycleStorage::build(&embedded, .., 25, 0.95, &ExecutionBackend::Sequential).unwrap();
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
        let outcome = CycleStorage::build(&embedded, .., 1, 0.95, &ExecutionBackend::Sequential);
        assert!(matches!(
            outcome,
            Err(Error::InvalidMaxLength { max_length: 1 })
        ));
    }

    #[test]
    fn threshold_below_resolution_is_rejected() {
        let embedded = loop_trajectory();
        let lower_threshold = embedded.resolution() - 0.1;
        let outcome = CycleStorage::build(
            &embedded,
            ..,
            25,
            lower_threshold,
            &ExecutionBackend::Sequential,
        );
        assert!(matches!(
            outcome,
            Err(Error::ThresholdBelowResolution { threshold, resolution })
                if (threshold - lower_threshold).abs() < 1e-12
                    && (resolution - embedded.resolution()).abs() < 1e-12
        ));

        // A NaN threshold fails every comparison, so the band check must be
        // written to reject it rather than let it pass both ends.
        let nan_outcome =
            CycleStorage::build(&embedded, .., 25, f64::NAN, &ExecutionBackend::Sequential);
        assert!(matches!(
            nan_outcome,
            Err(Error::ThresholdBelowResolution { threshold, .. }) if threshold.is_nan()
        ));
    }

    #[test]
    fn threshold_at_cube_side_is_rejected() {
        // A threshold of exactly 1.0 (the cube side) is never valid: the
        // thickening that justifies a detected class has half-threshold
        // radius, which must stay strictly below 1/2. The largest
        // representable value below it is still in-band for this fixture
        // (the resolution is far below 1) and must succeed.
        let embedded = loop_trajectory();
        let outcome = CycleStorage::build(&embedded, .., 25, 1.0, &ExecutionBackend::Sequential);
        assert!(matches!(
            outcome,
            Err(Error::ThresholdAboveCubeSide { threshold }) if (threshold - 1.0).abs() < 1e-12
        ));
        assert!(
            CycleStorage::build(
                &embedded,
                ..,
                25,
                1.0_f64.next_down(),
                &ExecutionBackend::Sequential,
            )
            .is_ok()
        );
    }

    #[test]
    fn max_length_behavior() {
        let embedded = loop_trajectory();
        let point_count = embedded.trajectory().len();
        let small =
            CycleStorage::build(&embedded, .., 4, 0.95, &ExecutionBackend::Sequential).unwrap();
        let medium = CycleStorage::build(
            &embedded,
            ..,
            point_count,
            0.95,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let huge = CycleStorage::build(
            &embedded,
            ..,
            10 * point_count,
            0.95,
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

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_round_trip_preserves_queries() {
        let embedded = loop_trajectory();
        let threshold = 0.95;
        let max_length = embedded.trajectory().len();
        let storage = CycleStorage::build(
            &embedded,
            ..,
            max_length,
            threshold,
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
            &[0..embedded.trajectory().len(), 0..8, 8..max_length, 4..13];
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
        for point in 0..embedded.trajectory().len() {
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
        let storage =
            CycleStorage::build(&embedded, .., 25, 0.95, &ExecutionBackend::Sequential).unwrap();
        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &storage).unwrap();

        let loaded = load_from_reader::<CycleStorage, _>(&buffer[..]).unwrap();

        // The matching embedded shares the loaded fingerprint.
        assert_eq!(loaded.fingerprint(), embedded.fingerprint());

        // A different embedded trajectory (different points) has a different
        // fingerprint, so a provenance check would reject it.
        let other_points = array![[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        let other_trajectory = Trajectory::new(other_points.view()).unwrap();
        let other_embedded = embed_euclidean(other_trajectory).unwrap();
        assert_ne!(loaded.fingerprint(), other_embedded.fingerprint());
    }
}
