// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cycle storage.
//!
//! Cache for all near-recurrent cycles over a trajectory extent (the range of
//! trajectory points the storage was built over). Cycles are grouped by their
//! connected component in the below-threshold distance graph, paired with the
//! homology class shared across the component.

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
    /// [`Error::SegmentOutOfBounds`] if `segment` does not fit inside
    /// [`Self::extent`].
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
            return Err(Error::SegmentOutOfBounds {
                start: range.start,
                end: range.end,
                point_count: self.extent.end as usize,
            });
        }
        let query = (range.start as u32)..(range.end as u32);

        // One entry per contributing component at its minimum finite birth, in
        // ascending component id. Component ids are dense in `0..len`, so the
        // fold indexes them directly and the sweep below emits them in order,
        // independently of the index's own (begin, end) iteration order.
        //
        // Infinity doubles as the "nothing contributed" marker: a component
        // with no contained cycle keeps it, and one whose every contained cycle
        // has a non-finite birth cannot lower it, so the same test drops both.
        let mut minimum_births: Vec<f64> = vec![f64::INFINITY; self.components.len()];
        for stored in self.index.contained_in(query) {
            let slot = &mut minimum_births[stored.payload as usize];
            *slot = slot.min(stored.birth);
        }
        let births = minimum_births
            .into_iter()
            .enumerate()
            .filter(|(_, birth)| birth.is_finite())
            .map(|(component_id, birth)| {
                (
                    birth,
                    self.classes[self.components[component_id].class_id as usize].clone(),
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
    #[expect(
        clippy::missing_panics_doc,
        reason = "internal panic call is guarded, so the method advertises no panic"
    )]
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

    use super::CycleStorage;
    #[cfg(feature = "serde")]
    use crate::serialization::{load_from_reader, save_to_writer};
    use crate::{
        EmbeddedTrajectory, SignatureGenerator, Trajectory,
        error::Error,
        util::fixtures::{densify_path, embed_euclidean, ring_waypoints},
    };

    /// The center point of each cube in `cubes`.
    fn cube_centers(cubes: &[(i32, i32)]) -> Vec<[f64; 2]> {
        cubes
            .iter()
            .map(|&(x, y)| [f64::from(x) + 0.5, f64::from(y) + 0.5])
            .collect()
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

        // A filtered query restricts an already-computed filtration rather than
        // re-detecting, so it answers at thresholds detection itself rejects,
        // the trajectory's own resolution among them.
        assert_eq!(
            storage
                .signature(..)
                .unwrap()
                .rank_at(embedded.resolution())
                .unwrap(),
            0
        );

        // A class enters the filtration strictly above its birth, so each rank
        // steps between a birth and the next representable value above it.
        // Every threshold here is inside the embedded path's valid domain, so
        // both paths are compared directly.
        for (threshold, expected_rank) in [
            (birth_one, 0),
            (birth_one.next_up(), 1),
            (birth_two, 1),
            (birth_two.next_up(), 2),
        ] {
            let storage_signature = storage.signature(..).unwrap();
            let embedded_signature = embedded
                .signature(.., threshold, &ExecutionBackend::Sequential)
                .unwrap();
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

    #[cfg(feature = "rayon")]
    #[test]
    fn signature_agrees_between_rayon_and_sequential_backends() {
        let embedded = two_hole_trajectory();
        let band_top = 0.95;

        let sequential = embedded
            .signature(.., band_top, &ExecutionBackend::Sequential)
            .unwrap();
        let rayon = embedded
            .signature(.., band_top, &ExecutionBackend::Rayon)
            .unwrap();

        assert_eq!(sequential.span(), rayon.span());
        for threshold in [0.8, 0.9_f64.next_down(), 0.9] {
            assert_eq!(
                sequential.rank_at(threshold).unwrap(),
                rayon.rank_at(threshold).unwrap(),
                "rank mismatch at threshold {threshold}"
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
        // The band opens above the resolution, so the lowest threshold
        // detection can answer at is the next representable value above it.
        let thresholds = [
            resolution.next_up(),
            f64::midpoint(resolution, band_top),
            band_top,
        ];

        for segment in segments {
            let storage_signature = storage.signature(segment.clone()).unwrap();
            for &threshold in &thresholds {
                let embedded_signature = embedded
                    .signature(segment.clone(), threshold, &ExecutionBackend::Sequential)
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
        assert!(matches!(error, Error::SegmentOutOfBounds { start: 0, .. }));
    }

    #[test]
    fn max_length_below_minimum_is_rejected() {
        let embedded = loop_trajectory();
        let outcome = CycleStorage::build(&embedded, .., 1, 0.95, &ExecutionBackend::Sequential);
        assert!(matches!(
            outcome,
            Err(Error::MaxLengthBelowMinimum { max_length: 1 })
        ));
    }

    #[test]
    fn threshold_not_above_resolution_is_rejected() {
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
            Err(Error::ThresholdNotAboveResolution { threshold, resolution })
                if (threshold - lower_threshold).abs() < 1e-12
                    && (resolution - embedded.resolution()).abs() < 1e-12
        ));

        // The band opens strictly above the resolution, so the resolution
        // itself is out of band: at it, the widest consecutive pair of points
        // is not admitted.
        let boundary_outcome = CycleStorage::build(
            &embedded,
            ..,
            25,
            embedded.resolution(),
            &ExecutionBackend::Sequential,
        );
        assert!(matches!(
            boundary_outcome,
            Err(Error::ThresholdNotAboveResolution { threshold, .. })
                if (threshold - embedded.resolution()).abs() < 1e-12
        ));

        // A NaN threshold triggers the same negated guard.
        let nan_outcome =
            CycleStorage::build(&embedded, .., 25, f64::NAN, &ExecutionBackend::Sequential);
        assert!(matches!(
            nan_outcome,
            Err(Error::ThresholdNotAboveResolution { threshold, .. }) if threshold.is_nan()
        ));
    }

    #[test]
    fn threshold_above_cube_side_is_rejected() {
        // The band closes at the cube side: a pair is admitted only when it
        // is strictly closer than the threshold, so a threshold of exactly
        // 1.0 still leaves every admitted distance below the cube side.
        let embedded = loop_trajectory();
        assert!(CycleStorage::build(&embedded, .., 25, 1.0, &ExecutionBackend::Sequential).is_ok());

        let outcome = CycleStorage::build(
            &embedded,
            ..,
            25,
            1.0 + f64::EPSILON,
            &ExecutionBackend::Sequential,
        );
        assert!(matches!(
            outcome,
            Err(Error::ThresholdAboveCubeSide { threshold })
                if (threshold - (1.0 + f64::EPSILON)).abs() < 1e-12
        ));
    }

    #[test]
    fn coincident_points_reject_a_zero_threshold() {
        // A trajectory whose points all coincide has a resolution of zero, so
        // every threshold is out of band and the lowest one a caller can name
        // is reported rather than reaching detection, where the tile's
        // self-comparisons sit at distance zero.
        let points = Array2::from_shape_vec((4, 2), vec![0.5; 8]).unwrap();
        let embedded = embed_euclidean(Trajectory::new(points.view()).unwrap()).unwrap();
        assert!((embedded.resolution() - 0.0).abs() < 1e-12);

        let outcome = CycleStorage::build(&embedded, .., 3, 0.0, &ExecutionBackend::Sequential);
        assert!(matches!(
            outcome,
            Err(Error::ThresholdNotAboveResolution { threshold, resolution })
                if threshold.abs() < 1e-12 && resolution.abs() < 1e-12
        ));
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
        // Representative points rather than every index: the two ends of the
        // extent, an interior point, and one past the extent, which resolves to
        // no component at all.
        for point in [0, 12, 24, 25] {
            assert_eq!(
                storage.components_covering(point),
                loaded_storage.components_covering(point),
                "coverage differs after round-trip at point {point}",
            );
        }
        assert!(
            !storage.components_covering(0).is_empty(),
            "fixture covers no points, so the comparison above holds vacuously",
        );
    }
}
