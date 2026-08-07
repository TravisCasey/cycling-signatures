// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`crate::CycleStorage`]: a versioned on-disk representation
//! and the save/load round trip.

#[cfg(any(test, feature = "serde"))]
use std::ops::Range;
#[cfg(feature = "serde")]
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(any(test, feature = "serde"))]
use super::{Component, CycleStorage};
#[cfg(any(test, feature = "serde"))]
use crate::{F2Vector, storage::interval_subsumption::IntervalSubsumptionIndex};
#[cfg(feature = "serde")]
use crate::{
    error::Result,
    serialization::{load_from_path, save_to_path},
};

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct CycleStorageData {
    fingerprint: u64,
    extent: Range<u32>,
    max_length: u32,
    threshold: f64,
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
            data.num_generators,
            data.classes,
            data.components,
        ))
    }
}

#[cfg(any(test, feature = "serde"))]
impl CycleStorage {
    /// Assembles a [`CycleStorage`] from already-computed parts, recomputing
    /// the internal subsumption index from `components`.
    ///
    /// Component invariants (`coverage` bounds the cycle ranges, `class_id` is
    /// in range, cycles are non-empty) are the caller's responsibility.
    #[must_use]
    fn from_parts(
        fingerprint: u64,
        extent: Range<u32>,
        max_length: u32,
        threshold: f64,
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
            num_generators,
            classes,
            components,
            index,
        }
    }
}

#[cfg(feature = "serde")]
impl CycleStorage {
    /// Writes this storage to `path` in the crate's binary format.
    ///
    /// # Errors
    ///
    /// [`Error::Io`](crate::Error::Io) on file or serialization failure.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_to_path(path, self)
    }

    /// Reads a storage written by [`save`](Self::save).
    ///
    /// The returned storage carries the fingerprint of the embedded trajectory
    /// it was built from; compare it against
    /// [`EmbeddedTrajectory::fingerprint`](crate::EmbeddedTrajectory::fingerprint)
    /// to confirm provenance.
    ///
    /// # Errors
    ///
    /// - [`Error::FormatVersionMismatch`](crate::Error::FormatVersionMismatch)
    ///   if the file's format version differs.
    /// - [`Error::Io`](crate::Error::Io) if the file could not be opened.
    /// - [`Error::Deserialize`](crate::Error::Deserialize) if the file contents
    ///   could not be read and decoded.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load_from_path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Component, Cycle, CycleStorage};
    use crate::F2Vector;

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
        let storage = CycleStorage::from_parts(0, 0..100, 60, 1.5, 1, vec![class], vec![component]);

        let wide = storage.signature(0..100).unwrap();
        assert_eq!(wide.generators().len(), 1);
        assert!((wide.generators()[0].birth() - 0.1).abs() < 1e-12);

        let narrow = storage.signature(35..50).unwrap();
        assert_eq!(narrow.generators().len(), 1);
        assert!((narrow.generators()[0].birth() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn signature_drops_non_finite_birth_component() {
        // A non-finite birth only arises in a hand-assembled or deserialized
        // storage (never one produced by `build`); the finiteness filter in
        // `signature` must still exclude it from becoming a generator rather
        // than folding infinity into a component minimum. A second,
        // finite-birth component confirms only it survives.
        let infinite_class = F2Vector::from_nonzero(2, [0]);
        let finite_class = F2Vector::from_nonzero(2, [1]);
        let infinite_component = Component {
            class_id: 0,
            coverage: 10..20,
            cycles: vec![Cycle {
                range: 10..20,
                birth: f64::INFINITY,
            }],
        };
        let finite_component = Component {
            class_id: 1,
            coverage: 30..40,
            cycles: vec![Cycle {
                range: 30..40,
                birth: 0.5,
            }],
        };
        let storage = CycleStorage::from_parts(
            0,
            0..100,
            20,
            1.5,
            2,
            vec![infinite_class, finite_class.clone()],
            vec![infinite_component, finite_component],
        );

        let signature = storage.signature(0..100).unwrap();
        assert_eq!(signature.rank(), 1);
        assert_eq!(signature.generators()[0].class(), &finite_class);
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
            2,
            vec![class_zero, class_one],
            vec![component_a, component_b],
        );
        // Point 7 is in both [0, 10) and [5, 15).
        assert_eq!(union_storage.components_covering(7), vec![0, 1]);
    }
}
