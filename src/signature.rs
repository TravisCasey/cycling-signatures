// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Output types for trajectory cycling signatures.
//!
//! A [`CyclingSignature`] is the filtered `F_2` subspace of cubical-cover
//! homology classes spanned by a trajectory's recurrent cycles: at
//! adjacency threshold `t`, the signature is the span of every retained
//! generator whose birth (the adjacency threshold at which it first becomes
//! independent of the generators before it) is at most `t`.

use chomp3rs::{F2, Ring};

use crate::{
    F2Subspace, F2Vector,
    error::{Error, Result},
};

/// One retained generator of a filtered [`CyclingSignature`]: a homology
/// class together with the adjacency threshold ("birth") at which it enters
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
    /// The adjacency threshold at which this generator's class first becomes
    /// independent of the generators with smaller birth.
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
/// [`EmbeddedTrajectory::signature`](crate::EmbeddedTrajectory::signature),
/// [`EmbeddedTrajectory::signature_with_threshold`](crate::EmbeddedTrajectory::signature_with_threshold),
/// or [`CycleStorage::signature`](crate::CycleStorage::signature).
/// [`span`](Self::span) and [`rank`](Self::rank) report the full band;
/// [`span_at`](Self::span_at) and [`rank_at`](Self::rank_at) restrict to a
/// smaller threshold, up to [`threshold_max`](Self::threshold_max).
#[derive(Debug, Clone)]
pub struct CyclingSignature {
    generators: Vec<SignatureGenerator>,
    span: F2Subspace,
    threshold_max: f64,
}

impl CyclingSignature {
    /// Builds a filtered signature from candidate `(birth, class)` pairs.
    ///
    /// Candidates are sorted by ascending birth (ties broken by input order),
    /// then reduced by incremental Gaussian elimination over the fixed
    /// ambient generator basis: each candidate is combined by exclusive-or
    /// with the classes of already-retained generators with a set bit at
    /// that generator's leading index; a candidate that reduces to zero is
    /// dropped, and a nonzero remainder is retained with its birth. A
    /// candidate sharing an earlier candidate's class always reduces to
    /// zero, so the earliest birth wins when the same class is offered more
    /// than once.
    ///
    /// `num_generators` is the ambient dimension every candidate class must
    /// share; `threshold_max` is the inclusive upper end of the range this
    /// signature answers queries for.
    #[must_use]
    pub(crate) fn from_candidates(
        mut candidates: Vec<(f64, F2Vector)>,
        num_generators: usize,
        threshold_max: f64,
    ) -> Self {
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0));

        let mut generators: Vec<SignatureGenerator> = Vec::new();
        for (birth, mut class) in candidates {
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

        Self {
            generators,
            span,
            threshold_max,
        }
    }

    /// The dimension of the full-band spanned subspace: the number of
    /// independent cycling classes the signature carries at
    /// [`threshold_max`](Self::threshold_max).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.generators.len()
    }

    /// The full-band `F_2` subspace, spanned by every retained generator.
    #[must_use]
    pub fn span(&self) -> &F2Subspace {
        &self.span
    }

    /// The number of independent cycling classes with birth at most
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// [`Error::ThresholdExceedsFiltrationBand`] if `threshold` exceeds
    /// [`threshold_max`](Self::threshold_max), or is NaN.
    // The negated comparison (rather than `threshold > self.threshold_max`)
    // is deliberate: it also rejects a NaN threshold, which is neither
    // greater than nor less than or equal to any value.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn rank_at(&self, threshold: f64) -> Result<usize> {
        if !(threshold <= self.threshold_max) {
            return Err(Error::ThresholdExceedsFiltrationBand {
                threshold,
                threshold_max: self.threshold_max,
            });
        }
        Ok(self
            .generators
            .partition_point(|generator| generator.birth <= threshold))
    }

    /// The `F_2` subspace spanned by every generator with birth at most
    /// `threshold`.
    ///
    /// # Errors
    ///
    /// [`Error::ThresholdExceedsFiltrationBand`] if `threshold` exceeds
    /// [`threshold_max`](Self::threshold_max), or is NaN.
    #[allow(clippy::missing_panics_doc)]
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

    /// The largest threshold this signature answers queries for.
    #[must_use]
    pub fn threshold_max(&self) -> f64 {
        self.threshold_max
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
    fn from_candidates_eliminates_in_birth_order() {
        let e0 = F2Vector::from_nonzero(3, [0]);
        let e1 = F2Vector::from_nonzero(3, [1]);
        let e2 = F2Vector::from_nonzero(3, [2]);
        let e0_e1 = &e0 ^ &e1;
        let e0_e1_e2 = &e0_e1 ^ &e2;

        // Deliberately unsorted, with a dependent candidate (0.9, the XOR of
        // the first two pivots) and a repeated class (0.8, matching the
        // earlier 0.2 candidate) both expected to vanish.
        let candidates = vec![
            (0.5, e0_e1.clone()),
            (0.2, e2.clone()),
            (0.9, e0_e1_e2),
            (0.7, e0.clone()),
            (0.8, e2.clone()),
        ];

        let signature = CyclingSignature::from_candidates(candidates, 3, 1.0);

        let births: Vec<f64> = signature
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect();
        assert_eq!(births, vec![0.2, 0.5, 0.7]);

        assert_eq!(signature.rank_at(0.1).unwrap(), 0);
        assert_eq!(signature.rank_at(0.2).unwrap(), 1);
        assert_eq!(signature.rank_at(0.6).unwrap(), 2);
        assert_eq!(signature.rank_at(1.0).unwrap(), 3);
        assert_eq!(signature.rank_at(1.0).unwrap(), signature.rank());

        let expected_span_at_0_6 = F2Subspace::new(vec![e2, e0_e1], 3).unwrap();
        assert_eq!(signature.span_at(0.6).unwrap(), expected_span_at_0_6);

        assert!(matches!(
            signature.rank_at(1.5).unwrap_err(),
            Error::ThresholdExceedsFiltrationBand { threshold, threshold_max }
                if (threshold - 1.5).abs() < 1e-12 && (threshold_max - 1.0).abs() < 1e-12
        ));
        assert!(matches!(
            signature.rank_at(f64::NAN).unwrap_err(),
            Error::ThresholdExceedsFiltrationBand { .. }
        ));
    }
}
