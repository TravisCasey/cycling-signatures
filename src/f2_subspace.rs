// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Subspaces of `F_2^n` stored in reduced row echelon form.

use std::{cmp::Ordering, fmt};

use chomp3rs::{F2, Ring};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    f2_vector::F2Vector,
};

/// A subspace of `F_2^n`.
///
/// Two subspaces compare equal if and only if they span the same set of
/// vectors. The internal canonical form makes this comparison structurally
/// cheap.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct F2Subspace {
    basis: Vec<F2Vector>,
    num_generators: usize,
}

impl F2Subspace {
    /// Constructs the subspace spanned by `vectors`.
    ///
    /// `vectors` is a generating set in `F_2^n`. Independence is not assumed;
    /// the implementation reduces to RREF to canonicalize. Passing an empty
    /// slice yields the trivial (rank 0) subspace.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::{F2Subspace, F2Vector};
    ///
    /// // The 1-dim subspace of F_2^3 spanned by [1, 1, 0].
    /// let first =
    ///     F2Subspace::new(&[F2Vector::from_nonzero(3, [0, 1])], 3).unwrap();
    ///
    /// // The 2-dim subspace spanned by {[1, 1, 0], [0, 1, 0]} equals the
    /// // subspace spanned by {[1, 0, 0], [0, 1, 0]}; canonicalization makes
    /// // the two representations equal.
    /// let from_first_pair = F2Subspace::new(
    ///     &[
    ///         F2Vector::from_nonzero(3, [0, 1]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    /// let from_second_pair = F2Subspace::new(
    ///     &[
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
    /// let empty: &[F2Vector] = &[];
    /// let trivial = F2Subspace::new(empty, 3).unwrap();
    /// assert_eq!(trivial, F2Subspace::trivial(3));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::F2SubspaceVectorLength`] if any vector's length
    /// differs from `num_generators`.
    pub fn new<V: AsRef<F2Vector>>(vectors: &[V], num_generators: usize) -> Result<Self> {
        let mut owned: Vec<F2Vector> = Vec::with_capacity(vectors.len());
        for (index, vector) in vectors.iter().enumerate() {
            let vector = vector.as_ref();
            if vector.len() != num_generators {
                return Err(Error::F2SubspaceVectorLength {
                    index,
                    actual: vector.len(),
                    expected: num_generators,
                });
            }
            owned.push(vector.clone());
        }
        let basis = rref_f2(owned, num_generators);
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

    /// The basis vector at the given index, in pivot-column order.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[must_use]
    pub fn column(&self, index: usize) -> &F2Vector {
        &self.basis[index]
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
    ///     &[
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
            "vector length does not match generator count",
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
}

impl PartialOrd for F2Subspace {
    /// Subspace inclusion partial order.
    ///
    /// Returns `Less`, `Equal`, or `Greater` for comparable subspaces and
    /// `None` for incomparable ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::{F2Subspace, F2Vector};
    ///
    /// let small = F2Subspace::new(&[F2Vector::from_nonzero(3, [0])], 3).unwrap();
    /// let big = F2Subspace::new(
    ///     &[
    ///         F2Vector::from_nonzero(3, [0]),
    ///         F2Vector::from_nonzero(3, [1]),
    ///     ],
    ///     3,
    /// )
    /// .unwrap();
    ///
    /// assert!(small < big);
    /// assert!(big > small);
    ///
    /// // Two 1-dim subspaces along different axes are incomparable.
    /// let other = F2Subspace::new(&[F2Vector::from_nonzero(3, [2])], 3).unwrap();
    /// assert_eq!(small.partial_cmp(&other), None);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the two subspaces have different generator counts.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_eq!(
            self.num_generators, other.num_generators,
            "comparing subspaces with different generator counts",
        );
        // Different ranks rule out equality and one inclusion direction; only
        // the larger-ranked side can possibly contain the smaller, so we test
        // a single direction.
        match self.rank().cmp(&other.rank()) {
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
        }
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
            writeln!(formatter)?;
            for entry in basis_vector {
                let glyph = if entry == F2::one() { '1' } else { '0' };
                write!(formatter, "{glyph}")?;
            }
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
        let candidate = (pivot_row..rows.len()).find(|&row| rows[row].get(column) == F2::one());
        let Some(found) = candidate else { continue };
        rows.swap(pivot_row, found);
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
    fn new_canonicalizes_equivalent_spanning_sets() {
        // Two spanning sets of the same 2-dim subspace of F_2^3:
        // {[1,1,0], [0,1,0]}  and  {[1,0,0], [0,1,0]}.
        let first = F2Subspace::new(&[vector(3, &[0, 1]), vector(3, &[1])], 3).unwrap();
        let second = F2Subspace::new(&[vector(3, &[0]), vector(3, &[1])], 3).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn column_basis_vectors_consistency() {
        let subspace = F2Subspace::new(&[vector(4, &[0, 2]), vector(4, &[1])], 4).unwrap();
        assert_eq!(subspace.rank(), 2);
        assert_eq!(subspace.num_generators(), 4);
        for index in 0..subspace.rank() {
            assert_eq!(subspace.column(index), &subspace.basis_vectors()[index]);
        }
    }

    #[test]
    fn contains_in_span() {
        // Subspace spanned by [1,0,1] and [0,1,0] in F_2^3.
        let subspace = F2Subspace::new(&[vector(3, &[0, 2]), vector(3, &[1])], 3).unwrap();
        // [1,1,1] = [1,0,1] + [0,1,0] is in the span.
        assert!(subspace.contains(&vector(3, &[0, 1, 2])));
        // [0,0,1] is not in the span.
        assert!(!subspace.contains(&vector(3, &[2])));
    }

    #[test]
    fn contains_zero_vector_in_trivial() {
        let trivial = F2Subspace::trivial(5);
        assert!(trivial.contains(&F2Vector::zeros(5)));
        assert!(!trivial.contains(&F2Vector::from_nonzero(5, [2])));
    }

    #[test]
    #[should_panic(expected = "vector length")]
    fn contains_dimension_mismatch_panics() {
        let subspace = F2Subspace::trivial(3);
        let _ = subspace.contains(&F2Vector::zeros(4));
    }

    #[test]
    fn partial_ord_inclusion() {
        let small = F2Subspace::new(&[vector(3, &[0])], 3).unwrap();
        let large = F2Subspace::new(&[vector(3, &[0]), vector(3, &[1])], 3).unwrap();
        assert!(small < large);
        assert!(large > small);
        let other = F2Subspace::new(&[vector(3, &[2])], 3).unwrap();
        assert_eq!(small.partial_cmp(&other), None);
    }

    #[test]
    #[should_panic(expected = "generator count")]
    fn partial_ord_dimension_mismatch_panics() {
        let left = F2Subspace::trivial(3);
        let right = F2Subspace::trivial(4);
        let _ = left.partial_cmp(&right);
    }

    #[test]
    fn new_row_length_mismatch_returns_err() {
        let mismatched = vec![vector(3, &[0]), vector(4, &[1])];
        let outcome = F2Subspace::new(&mismatched, 3);
        let err = outcome.unwrap_err();
        let Error::F2SubspaceVectorLength {
            index,
            actual,
            expected,
        } = err;
        assert_eq!(index, 1);
        assert_eq!(actual, 4);
        assert_eq!(expected, 3);
    }

    #[test]
    fn new_dependent_and_zero_rows_drop() {
        let subspace = F2Subspace::new(
            &[
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
        let first = F2Subspace::new(&[vector(3, &[0, 1]), vector(3, &[1])], 3).unwrap();
        let second = F2Subspace::new(&[vector(3, &[0]), vector(3, &[1])], 3).unwrap();
        assert_eq!(first, second);
        let mut hasher_first = DefaultHasher::new();
        let mut hasher_second = DefaultHasher::new();
        first.hash(&mut hasher_first);
        second.hash(&mut hasher_second);
        assert_eq!(hasher_first.finish(), hasher_second.finish());
    }
}
