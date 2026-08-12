// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Subspaces of vector spaces `F_2^n` stored in reduced row echelon form.

use std::{cmp::Ordering, fmt};

use chomp3rs::{F2, Ring};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    f2_vector::F2Vector,
};

/// A subspace of the vector space `F_2^n`.
///
/// Two subspaces compare equal if and only if they span the same set of
/// vectors.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct F2Subspace {
    basis: Vec<F2Vector>,
    num_generators: usize,
}

impl F2Subspace {
    /// Constructs the subspace spanned by `vectors`.
    ///
    /// `vectors` is a generating set in `F_2^n`; independence is not assumed,
    /// and the resulting subspace depends only on the space `vectors` spans.
    /// Passing an empty vector yields the trivial (rank 0) subspace.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::{F2Subspace, F2Vector};
    ///
    /// // The 1-dim subspace of F_2^3 spanned by [1, 1, 0].
    /// let first =
    ///     F2Subspace::new(vec![F2Vector::from_nonzero(3, [0, 1])], 3).unwrap();
    ///
    /// // The 2-dim subspace spanned by {[1, 1, 0], [0, 1, 0]} equals the
    /// // subspace spanned by {[1, 0, 0], [0, 1, 0]}; canonicalization makes
    /// // the two representations equal.
    /// let from_first_pair = F2Subspace::new(
    ///     vec![
    ///         F2Vector::from_nonzero(3, [0, 1]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    /// let from_second_pair = F2Subspace::new(
    ///     vec![
    ///         F2Vector::from_nonzero(3, [0]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    /// assert_eq!(from_first_pair, from_second_pair);
    /// assert_eq!(from_first_pair.rank(), 2);
    ///
    /// // Empty input yields the trivial (rank-0) subspace.
    /// let trivial = F2Subspace::new(Vec::new(), 3).unwrap();
    /// assert_eq!(trivial, F2Subspace::trivial(3));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::F2SubspaceVectorLength`] if any vector's length differs
    /// from `num_generators`.
    pub fn new(vectors: Vec<F2Vector>, num_generators: usize) -> Result<Self> {
        for (index, vector) in vectors.iter().enumerate() {
            if vector.len() != num_generators {
                return Err(Error::F2SubspaceVectorLength {
                    index,
                    actual: vector.len(),
                    expected: num_generators,
                });
            }
        }
        let basis = rref_f2(vectors, num_generators);
        Ok(Self {
            basis,
            num_generators,
        })
    }

    /// The trivial (rank 0) subspace for the given generator count.
    #[must_use]
    pub fn trivial(num_generators: usize) -> Self {
        Self {
            basis: Vec::new(),
            num_generators,
        }
    }

    /// The dimension of this subspace.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// The ambient dimension `n` for which this subspace lies in `F_2^n`.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// The basis vectors as a slice, in pivot-column order.
    #[must_use]
    pub fn basis_vectors(&self) -> &[F2Vector] {
        &self.basis
    }

    /// Tests whether `vector` lies in this subspace.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::{F2Subspace, F2Vector};
    ///
    /// let subspace = F2Subspace::new(
    ///     vec![
    ///         F2Vector::from_nonzero(3, [0]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    ///
    /// // [1, 1, 0] = [1, 0, 0] + [0, 1, 0] is in the span.
    /// assert!(subspace.contains(&F2Vector::from_nonzero(3, [0, 1])));
    ///
    /// // [0, 0, 1] is not in the span.
    /// assert!(!subspace.contains(&F2Vector::from_nonzero(3, [2])));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `vector.len()` does not equal `self.num_generators()`.
    #[must_use]
    pub fn contains(&self, vector: &F2Vector) -> bool {
        assert_eq!(
            vector.len(),
            self.num_generators,
            "vector length {} does not match generator count {}",
            vector.len(),
            self.num_generators
        );

        let mut residual = vector.clone();
        for basis_vector in &self.basis {
            let pivot = basis_vector
                .first_nonzero_index()
                .expect("RREF basis vector cannot be zero");
            if residual.get(pivot) == F2::one() {
                residual ^= basis_vector;
            }
        }
        residual.is_zero()
    }

    /// Compares this subspace and `other` under the subspace inclusion order.
    ///
    /// Returns `Ok(Some(..))` when one subspace includes the other, and
    /// `Ok(None)` when the two subspaces share an ambient space but neither
    /// includes the other. An `Err` result says only that the two subspaces'
    /// ambient dimensions differ, which is necessary but not sufficient for a
    /// comparison between them to be meaningful; comparing cover fingerprints
    /// is how a caller establishes that two subspaces come from the same cover.
    ///
    /// # Errors
    ///
    /// Returns [`Error::F2SubspaceGeneratorCountMismatch`] if `self` and
    /// `other` have different generator counts.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// use cycling_signatures::{F2Subspace, F2Vector};
    ///
    /// let small =
    ///     F2Subspace::new(vec![F2Vector::from_nonzero(3, [0])], 3).unwrap();
    /// let big = F2Subspace::new(
    ///     vec![
    ///         F2Vector::from_nonzero(3, [0]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    ///
    /// assert_eq!(small.inclusion(&big).unwrap(), Some(Ordering::Less));
    /// assert_eq!(big.inclusion(&small).unwrap(), Some(Ordering::Greater));
    ///
    /// // Two 1-dim subspaces along different axes are incomparable.
    /// let other =
    ///     F2Subspace::new(vec![F2Vector::from_nonzero(3, [2])], 3).unwrap();
    /// assert_eq!(small.inclusion(&other).unwrap(), None);
    ///
    /// // Equal rank alone does not decide the comparison: these two spanning
    /// // sets differ but span the same space, so each contains the other.
    /// let rewritten = F2Subspace::new(
    ///     vec![
    ///         F2Vector::from_nonzero(3, [0, 1]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    /// assert_eq!(big.inclusion(&rewritten).unwrap(), Some(Ordering::Equal));
    /// ```
    pub fn inclusion(&self, other: &Self) -> Result<Option<Ordering>> {
        if self.num_generators != other.num_generators {
            return Err(Error::F2SubspaceGeneratorCountMismatch {
                first: self.num_generators,
                second: other.num_generators,
            });
        }

        // Different ranks rule out equality and one inclusion direction; only
        // the larger-ranked side can possibly contain the smaller, so only one
        // direction is tested.
        let ordering = match self.rank().cmp(&other.rank()) {
            Ordering::Less => self
                .basis
                .iter()
                .all(|vector| other.contains(vector))
                .then_some(Ordering::Less),
            Ordering::Greater => other
                .basis
                .iter()
                .all(|vector| self.contains(vector))
                .then_some(Ordering::Greater),
            Ordering::Equal => self
                .basis
                .iter()
                .all(|vector| other.contains(vector))
                .then_some(Ordering::Equal),
        };
        Ok(ordering)
    }
}

impl fmt::Display for F2Subspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "F2Subspace(rank={}, generators={})",
            self.rank(),
            self.num_generators(),
        )?;
        for basis_vector in &self.basis {
            write!(formatter, "\n{basis_vector}")?;
        }
        Ok(())
    }
}

/// Computes the canonical RREF basis of the `F_2` subspace spanned by `rows`.
///
/// Mutates `rows` in place and returns the resulting basis. The output is
/// ordered by ascending pivot column with leading ones and zeros elsewhere
/// in each pivot column. Empty input or zero-only input yields an empty
/// vector.
///
/// All input vectors must have length `num_generators`; this is a
/// precondition validated by the caller.
fn rref_f2(mut rows: Vec<F2Vector>, num_generators: usize) -> Vec<F2Vector> {
    let mut pivot_row = 0;
    for column in 0..num_generators {
        let pivot_search = (pivot_row..rows.len()).find(|&row| rows[row].get(column) == F2::one());
        let Some(pivot_row_index) = pivot_search else {
            continue;
        };
        rows.swap(pivot_row, pivot_row_index);

        let pivot = rows[pivot_row].clone();
        let (head, tail) = rows.split_at_mut(pivot_row + 1);
        for row in head[..pivot_row].iter_mut().chain(tail.iter_mut()) {
            if row.get(column) == F2::one() {
                *row ^= &pivot;
            }
        }
        pivot_row += 1;
    }
    rows.truncate(pivot_row);
    rows
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use super::F2Subspace;
    use crate::{error::Error, f2_vector::F2Vector};

    fn vector(num_generators: usize, indices: &[usize]) -> F2Vector {
        F2Vector::from_nonzero(num_generators, indices.iter().copied())
    }

    #[test]
    fn contains_in_span() {
        // Subspace spanned by [1,0,1] and [0,1,0] in F_2^3.
        let subspace = F2Subspace::new(vec![vector(3, &[0, 2]), vector(3, &[1])], 3).unwrap();
        // [1,1,1] = [1,0,1] + [0,1,0] is in the span.
        assert!(subspace.contains(&vector(3, &[0, 1, 2])));
        // [0,0,1] is not in the span.
        assert!(!subspace.contains(&vector(3, &[2])));
    }

    #[test]
    #[should_panic(expected = "vector length 4 does not match generator count 3")]
    fn contains_dimension_mismatch_panics() {
        let subspace = F2Subspace::trivial(3);
        let _ = subspace.contains(&F2Vector::zeros(4));
    }

    #[test]
    fn inclusion_generator_count_mismatch_returns_err() {
        // Differing ranks and generator counts, with a non-empty basis on the
        // smaller-ranked side, are required here so the comparison actually
        // reaches `contains`'s generator-count assertion instead of
        // short-circuiting on an empty basis.
        let left = F2Subspace::new(vec![vector(3, &[0])], 3).unwrap();
        let right = F2Subspace::new(vec![vector(4, &[0]), vector(4, &[1])], 4).unwrap();

        let err = left.inclusion(&right).unwrap_err();
        assert!(matches!(
            err,
            Error::F2SubspaceGeneratorCountMismatch {
                first: 3,
                second: 4
            }
        ));
    }

    #[test]
    fn display_shows_the_header_and_one_line_per_basis_vector() {
        // Built from a spanning set that is not already reduced, so the
        // rendered lines are the canonical basis rather than the input.
        let subspace = F2Subspace::new(vec![vector(3, &[0, 1]), vector(3, &[1])], 3).unwrap();

        assert_eq!(
            subspace.to_string(),
            "F2Subspace(rank=2, generators=3)\n100\n010"
        );
    }

    #[test]
    fn new_row_length_mismatch_returns_err() {
        let mismatched = vec![vector(3, &[0]), vector(4, &[1])];
        let err = F2Subspace::new(mismatched, 3).unwrap_err();

        assert!(matches!(
            err,
            Error::F2SubspaceVectorLength {
                index: 1,
                actual: 4,
                expected: 3
            }
        ));
    }

    #[test]
    fn new_dependent_and_zero_rows_drop() {
        let subspace = F2Subspace::new(
            vec![
                vector(3, &[0]),
                vector(3, &[0]),
                F2Vector::zeros(3),
                vector(3, &[1]),
            ],
            3,
        )
        .unwrap();
        assert_eq!(subspace.rank(), 2);
    }

    #[test]
    fn hash_matches_eq() {
        let first = F2Subspace::new(vec![vector(3, &[0, 1]), vector(3, &[1])], 3).unwrap();
        let second = F2Subspace::new(vec![vector(3, &[0]), vector(3, &[1])], 3).unwrap();
        assert_eq!(first, second);

        let mut hasher_first = DefaultHasher::new();
        let mut hasher_second = DefaultHasher::new();
        first.hash(&mut hasher_first);
        second.hash(&mut hasher_second);

        assert_eq!(hasher_first.finish(), hasher_second.finish());
    }
}
