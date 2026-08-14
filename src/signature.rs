// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Output types for trajectory cycling signatures.
//!
//! A [`CyclingSignature`] is the filtered `F_2` subspace of cubical-cover
//! homology classes spanned by a trajectory's recurrent cycles. At a threshold
//! `t` in the filtration band `[0, 1]`, it is the span of every admitted
//! generator whose birth (the endpoint distance at which it first becomes
//! independent of the generators before it) is below `t`.

use chomp3rs::{F2, Ring};

use crate::{
    F2Subspace, F2Vector,
    error::{Error, Result},
};

/// One retained generator of a filtered [`CyclingSignature`]: a homology
/// class together with the endpoint distance ("birth") above which it enters
/// the filtration.
///
/// The specific class a generator carries is an implementation detail of the
/// elimination that produced it; the contract a [`CyclingSignature`] honors
/// is its span at each threshold, not which vectors represent it.
#[derive(Debug, Clone)]
pub struct SignatureGenerator {
    birth: f64,
    class: F2Vector,
}

impl SignatureGenerator {
    /// The endpoint distance at which this generator's class first becomes
    /// independent of the generators with smaller birth. The class enters the
    /// filtration at every threshold above it.
    #[must_use]
    pub fn birth(&self) -> f64 {
        self.birth
    }

    /// The generator's homology class, in the cover's generator basis.
    #[must_use]
    pub fn class(&self) -> &F2Vector {
        &self.class
    }
}

/// A trajectory's cycling signature: the filtered `F_2` subspace of cover
/// homology spanned by its recurrent cycles.
///
/// Constructed by
/// [`EmbeddedTrajectory::signature`](crate::EmbeddedTrajectory::signature) or
/// [`CycleStorage::signature`](crate::CycleStorage::signature).
/// The filtration band is `[0, 1]`, closing at the cube side length.
/// [`span`](Self::span) and [`rank`](Self::rank) report every class detected;
/// [`span_at`](Self::span_at) and [`rank_at`](Self::rank_at) restrict to a
///  smaller threshold within the band.
#[derive(Debug, Clone)]
pub struct CyclingSignature {
    generators: Vec<SignatureGenerator>,
    span: F2Subspace,
}

impl CyclingSignature {
    /// Builds a filtered signature from `(birth, class)` pairs.
    ///
    /// The pairs are sorted by ascending birth (ties broken by input order),
    /// then reduced by incremental Gaussian elimination over the fixed
    /// ambient generator basis: each class is combined with the classes of
    /// already-retained generators; a class that reduces to zero is dropped,
    /// and a nonzero remainder is retained with its birth. A class matching
    /// an earlier one always reduces to zero and is dropped.
    ///
    /// `num_generators` is the ambient dimension every class must share.
    #[must_use]
    pub(crate) fn from_births(mut births: Vec<(f64, F2Vector)>, num_generators: usize) -> Self {
        births.sort_by(|left, right| left.0.total_cmp(&right.0));

        let mut generators: Vec<SignatureGenerator> = Vec::new();
        for (birth, mut class) in births {
            for pivot in &generators {
                let leading = pivot
                    .class
                    .first_nonzero_index()
                    .expect("retained pivots are nonzero");
                if class.get(leading) == F2::one() {
                    class ^= &pivot.class;
                }
            }
            if !class.is_zero() {
                generators.push(SignatureGenerator { birth, class });
            }
        }

        let classes: Vec<F2Vector> = generators
            .iter()
            .map(|generator| generator.class.clone())
            .collect();
        let span = F2Subspace::new(classes, num_generators)
            .expect("class vectors have the expected length by construction");

        Self { generators, span }
    }

    /// The dimension of the full-band spanned subspace: the number of
    /// independent cycling classes the signature carries at the top of the
    /// filtration band, i.e., the cube side length 1.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.generators.len()
    }

    /// The full-band `F_2` subspace, spanned by every retained generator.
    #[must_use]
    pub fn span(&self) -> &F2Subspace {
        &self.span
    }

    /// The number of independent cycling classes with birth below `threshold`.
    ///
    /// # Errors
    ///
    /// [`Error::ThresholdOutsideFiltrationBand`] if `threshold` falls outside
    /// the band `[0, 1]`, or is NaN.
    pub fn rank_at(&self, threshold: f64) -> Result<usize> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::ThresholdOutsideFiltrationBand { threshold });
        }
        Ok(self
            .generators
            .partition_point(|generator| generator.birth < threshold))
    }

    /// The `F_2` subspace spanned by every generator with birth below
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// [`Error::ThresholdOutsideFiltrationBand`] if `threshold` falls outside
    /// the band `[0, 1]`, or is NaN.
    #[expect(
        clippy::missing_panics_doc,
        reason = "internal panic call is guarded, so the method advertises no panic"
    )]
    pub fn span_at(&self, threshold: f64) -> Result<F2Subspace> {
        let rank = self.rank_at(threshold)?;
        let classes: Vec<F2Vector> = self.generators[..rank]
            .iter()
            .map(|generator| generator.class.clone())
            .collect();
        Ok(F2Subspace::new(classes, self.span.num_generators())
            .expect("class vectors have the expected length by construction"))
    }

    /// Every retained generator, ordered by ascending birth.
    #[must_use]
    pub fn generators(&self) -> &[SignatureGenerator] {
        &self.generators
    }

    /// The ambient dimension: the cover's generator count.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.span.num_generators()
    }
}

#[cfg(test)]
mod tests {
    use super::{CyclingSignature, SignatureGenerator};
    use crate::{F2Subspace, F2Vector, error::Error};

    #[test]
    fn from_births_eliminates_in_birth_order() {
        let basis_zero = F2Vector::from_nonzero(3, [0]);
        let basis_one = F2Vector::from_nonzero(3, [1]);
        let basis_two = F2Vector::from_nonzero(3, [2]);
        let basis_zero_one = &basis_zero ^ &basis_one;
        let basis_all = &basis_zero_one ^ &basis_two;

        // Deliberately unsorted, with a dependent class (0.9, the XOR of the
        // first two pivots) and a repeated class (0.8, matching the earlier
        // 0.2 entry) both expected to vanish.
        let births = vec![
            (0.5, basis_zero_one.clone()),
            (0.2, basis_two.clone()),
            (0.9, basis_all),
            (0.7, basis_zero.clone()),
            (0.8, basis_two.clone()),
        ];

        let signature = CyclingSignature::from_births(births, 3);

        let retained: Vec<f64> = signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert_eq!(retained, vec![0.2, 0.5, 0.7]);

        // A generator enters the filtration strictly above its birth, so the
        // step sits between 0.2 and the next representable value above it.
        assert_eq!(signature.rank_at(0.0).unwrap(), 0);
        assert_eq!(signature.rank_at(0.2).unwrap(), 0);
        assert_eq!(signature.rank_at(0.2_f64.next_up()).unwrap(), 1);
        assert_eq!(signature.rank_at(0.5).unwrap(), 1);
        assert_eq!(signature.rank_at(0.6).unwrap(), 2);
        // A query at the top of the band returns every retained generator.
        assert_eq!(signature.rank_at(1.0).unwrap(), signature.rank());

        let expected_span_at_0_6 = F2Subspace::new(vec![basis_two, basis_zero_one], 3).unwrap();
        assert_eq!(signature.span_at(0.6).unwrap(), expected_span_at_0_6);
    }

    #[test]
    fn rank_at_rejects_thresholds_outside_the_band() {
        let births = vec![(0.2, F2Vector::from_nonzero(3, [0]))];
        let signature = CyclingSignature::from_births(births, 3);

        // The band is closed at both ends, so a threshold is rejected only
        // outside them.
        assert!(matches!(
            signature.rank_at(1.5).unwrap_err(),
            Error::ThresholdOutsideFiltrationBand { threshold }
                if (threshold - 1.5).abs() < 1e-12
        ));
        assert!(matches!(
            signature.rank_at(-0.5).unwrap_err(),
            Error::ThresholdOutsideFiltrationBand { threshold }
                if (threshold + 0.5).abs() < 1e-12
        ));
        assert!(matches!(
            signature.rank_at(f64::NAN).unwrap_err(),
            Error::ThresholdOutsideFiltrationBand { .. }
        ));
    }
}
