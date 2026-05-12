// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Vector type over the field with two elements.

use std::{iter::FromIterator, ops::BitXorAssign};

use chomp3rs::{F2, Ring};

/// A bit-packed vector over the field with two elements.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct F2Vector {
    bits: Vec<u64>,
    len: usize,
}

impl F2Vector {
    /// The zero vector of the given length.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            len,
        }
    }

    /// A vector of the given length whose entries at the named indices are 1
    /// and all other entries are 0. Repeated indices are tolerated.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    #[must_use]
    pub fn from_nonzero(len: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut vector = Self::zeros(len);
        for index in indices {
            vector.set(index, F2::one());
        }
        vector
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether every entry is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|word| *word == 0)
    }

    /// The entry at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> F2 {
        assert!(index < self.len, "index out of bounds");
        let word = self.bits[index / 64];
        let bit = (word >> (index % 64)) & 1;
        F2::from(bit)
    }

    /// Sets the entry at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: F2) {
        assert!(index < self.len, "index out of bounds");
        let mask = 1u64 << (index % 64);
        if value == F2::one() {
            self.bits[index / 64] |= mask;
        } else {
            self.bits[index / 64] &= !mask;
        }
    }

    /// The smallest index at which the vector is nonzero, if any.
    #[must_use]
    pub fn first_nonzero_index(&self) -> Option<usize> {
        for (word_index, word) in self.bits.iter().enumerate() {
            if *word != 0 {
                return Some(word_index * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Iterates the entries in index order.
    pub fn iter(&self) -> impl Iterator<Item = F2> + '_ {
        (0..self.len).map(|index| self.get(index))
    }

    /// Iterates the indices at which the vector is nonzero, in ascending order.
    pub fn nonzero_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(word_index, word)| {
            let mut remaining = *word;
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                Some(word_index * 64 + bit)
            })
        })
    }
}

impl FromIterator<F2> for F2Vector {
    fn from_iter<I: IntoIterator<Item = F2>>(iter: I) -> Self {
        let entries: Vec<F2> = iter.into_iter().collect();
        let len = entries.len();
        let mut vector = Self::zeros(len);
        for (index, value) in entries.into_iter().enumerate() {
            vector.set(index, value);
        }
        vector
    }
}

impl BitXorAssign<&F2Vector> for F2Vector {
    /// XORs `other` into `self`.
    ///
    /// # Panics
    ///
    /// Panics if the two vectors have different lengths.
    fn bitxor_assign(&mut self, other: &F2Vector) {
        assert_eq!(self.len, other.len, "vector length mismatch");
        for (left, right) in self.bits.iter_mut().zip(other.bits.iter()) {
            *left ^= *right;
        }
    }
}

impl<'a> IntoIterator for &'a F2Vector {
    type IntoIter = Box<dyn Iterator<Item = F2> + 'a>;
    type Item = F2;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl From<&F2Vector> for Vec<F2> {
    fn from(vector: &F2Vector) -> Self {
        vector.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use chomp3rs::{F2, Ring};

    use super::F2Vector;

    fn hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn equality_across_construction_paths() {
        let from_set = {
            let mut vector = F2Vector::zeros(70);
            vector.set(3, F2::one());
            vector.set(65, F2::one());
            vector
        };
        let from_indices = F2Vector::from_nonzero(70, [3, 65]);
        let from_iter: F2Vector = (0..70)
            .map(|index| {
                if index == 3 || index == 65 {
                    F2::one()
                } else {
                    F2::zero()
                }
            })
            .collect();

        assert_eq!(from_set, from_indices);
        assert_eq!(from_set, from_iter);
        assert_eq!(hash(&from_set), hash(&from_indices));
        assert_eq!(hash(&from_set), hash(&from_iter));
    }

    #[test]
    fn xor_with_self_gives_zero() {
        let original = F2Vector::from_nonzero(130, [3, 64, 65, 129]);
        let copy = original.clone();
        let mut residual = original;
        residual ^= &copy;
        assert!(residual.is_zero());
    }

    #[test]
    fn first_nonzero_index_for_sparse_vector() {
        let vector = F2Vector::from_nonzero(200, [128]);
        assert_eq!(vector.first_nonzero_index(), Some(128));
        assert_eq!(F2Vector::zeros(200).first_nonzero_index(), None);
    }

    #[test]
    fn nonzero_indices_ascending() {
        let vector = F2Vector::from_nonzero(150, [42, 5, 100, 0]);
        let indices: Vec<usize> = vector.nonzero_indices().collect();
        assert_eq!(indices, vec![0, 5, 42, 100]);
    }

    #[test]
    #[should_panic(expected = "vector length mismatch")]
    fn xor_assign_dimension_mismatch_panics() {
        let mut left = F2Vector::zeros(64);
        let right = F2Vector::zeros(65);
        left ^= &right;
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn set_out_of_bounds_panics() {
        let mut vector = F2Vector::zeros(10);
        vector.set(10, F2::one());
    }

    #[test]
    fn vec_f2_round_trip() {
        let original = F2Vector::from_nonzero(130, [0, 5, 64, 65, 129]);
        let as_vec: Vec<F2> = (&original).into();
        let recovered: F2Vector = as_vec.into_iter().collect();
        assert_eq!(original, recovered);
    }
}
