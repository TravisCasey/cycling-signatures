// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the homology value types (classes and subspaces)
//! returned by signature queries. The signature type itself lives in
//! [`crate::signature`].

use std::hash::{Hash, Hasher};

use cycling_signatures::{F2, F2Subspace, F2Vector};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
};
use rustc_hash::FxHasher;

use crate::convert::resolve_index;

/// Computes a hash of a value, used to give the equatable value types a
/// `__hash__` consistent with their `__eq__`.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The homology class of a recurrent cycle, as a vector over ``F_2``.
///
/// A class records which cover generators a near-recurrent trajectory cycle
/// wraps. The coefficients live in the two-element field ``F_2`` (the integers
/// modulo 2), so every entry is either zero or one: the class is a zero/one
/// vector with one entry per cover generator.
///
/// The entries are coordinates in the cover's generator basis, which is not
/// stable across builds (see ``CubicalCover``), so comparing two classes entry
/// by entry is meaningful only when both were computed against one basis, and
/// is rarely the operation wanted. To compare cycling results, use
/// ``Subspace``.
///
/// ``len`` gives the number of cover generators, and indexing returns the zero
/// or one entry at a generator (negative indices count from the end). Equality
/// compares entry by entry.
#[pyclass(name = "HomologyClass")]
pub(crate) struct PyHomologyClass {
    pub(crate) inner: F2Vector,
}

/// A cycle space: the ``F_2`` subspace of cover homology spanned by some
/// classes.
///
/// Equality and membership depend only on the space a subspace spans, not on
/// the particular set of classes used to form it: two subspaces formed from
/// different spanning sets compare equal whenever they span the same space.
/// Compare cycling results at this level, since a signature's components may
/// contribute linearly dependent classes that no entry-by-entry comparison of
/// individual classes accounts for.
///
/// The comparison is still taken in the cover's generator basis, so it requires
/// both subspaces to use that same basis (see ``CubicalCover``). A cycling
/// signature over a fixed cover is independent of the basis chosen for its
/// homology; its coordinates are not, and the library offers no
/// basis-independent comparison.
#[pyclass(name = "Subspace")]
pub(crate) struct PySubspace {
    pub(crate) inner: F2Subspace,
}

#[pymethods]
impl PyHomologyClass {
    /// Returns the number of cover generators, the length of the class vector.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns the zero or one entry at generator ``index``.
    fn __getitem__(&self, index: isize) -> PyResult<u8> {
        let resolved = resolve_index(index, self.inner.len()).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "generator index {index} out of bounds for {} generators",
                self.inner.len()
            ))
        })?;
        Ok(u8::from(self.inner.get(resolved) == F2::from(1u64)))
    }

    /// Returns the indices of the class's nonzero entries, ascending.
    ///
    /// Returns
    /// -------
    /// list of int
    ///     The generator indices at which the class has value one.
    #[must_use]
    fn nonzero_indices(&self) -> Vec<usize> {
        self.inner.nonzero_indices().collect()
    }

    /// Returns whether this is the trivial (all-zero) class.
    ///
    /// Returns
    /// -------
    /// bool
    ///     ``True`` if every entry is zero.
    #[must_use]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns whether this class equals ``other`` entry by entry.
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Returns a hash consistent with equality.
    fn __hash__(&self) -> u64 {
        hash_of(&self.inner)
    }

    /// Returns the symmetric difference of this class and ``other``, entry by
    /// entry.
    ///
    /// Both classes must share one cover's generator basis.
    ///
    /// Parameters
    /// ----------
    /// other : ``HomologyClass``
    ///     A class with the same number of generators as this one.
    ///
    /// Returns
    /// -------
    /// ``HomologyClass``
    ///     The XOR of the two class vectors, entry by entry.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the two classes have different lengths.
    fn __xor__(&self, other: &Self) -> PyResult<Self> {
        if self.inner.len() != other.inner.len() {
            return Err(PyValueError::new_err(format!(
                "homology class length {} does not match the other class's length {}",
                self.inner.len(),
                other.inner.len()
            )));
        }
        Ok(Self {
            inner: &self.inner ^ &other.inner,
        })
    }

    /// Returns a string representation of the class.
    fn __repr__(&self) -> String {
        let set: Vec<String> = self
            .inner
            .nonzero_indices()
            .map(|index| index.to_string())
            .collect();
        format!(
            "HomologyClass(generators={}, set={{{}}})",
            self.inner.len(),
            set.join(", ")
        )
    }

    /// Returns the class as a dense zero or one array over the cover's
    /// generators.
    ///
    /// The array has one entry per cover generator, with ``1`` at each
    /// generator the class includes and ``0`` elsewhere. Its entry order
    /// follows the cover's generator basis.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A one-dimensional array of ``uint8`` zeros and ones.
    #[must_use]
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let mut values = vec![0_u8; self.inner.len()];
        for index in self.inner.nonzero_indices() {
            values[index] = 1;
        }
        PyArray1::from_vec(py, values)
    }
}

#[pymethods]
impl PySubspace {
    /// Returns the dimension of this subspace.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of independent classes the subspace spans.
    #[must_use]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns the number of cover generators, the dimension of the ambient
    /// space the subspace lies in.
    ///
    /// Returns
    /// -------
    /// int
    ///     Equal to the length of each class vector.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Returns the canonical basis classes that span this subspace.
    ///
    /// Two subspaces spanning the same space return the same basis, so a
    /// subspace of rank ``r`` returns ``r`` classes and the trivial subspace
    /// returns an empty list.
    ///
    /// Returns
    /// -------
    /// list of ``HomologyClass``
    ///     One class per dimension.
    #[must_use]
    fn basis(&self) -> Vec<PyHomologyClass> {
        self.inner
            .basis_vectors()
            .iter()
            .map(|vector| PyHomologyClass {
                inner: vector.clone(),
            })
            .collect()
    }

    /// Returns whether this subspace equals ``other`` by span comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Returns a hash consistent with equality.
    fn __hash__(&self) -> u64 {
        hash_of(&self.inner)
    }

    /// Returns a string representation of the subspace.
    fn __repr__(&self) -> String {
        format!(
            "Subspace(rank={}, generators={})",
            self.inner.rank(),
            self.inner.num_generators()
        )
    }

    /// Returns whether ``homology_class`` lies in this subspace.
    ///
    /// The class and the subspace must share one cover's generator basis.
    ///
    /// Parameters
    /// ----------
    /// homology_class : ``HomologyClass``
    ///     A class whose length matches the generator count of this subspace.
    ///
    /// Returns
    /// -------
    /// bool
    ///     ``True`` if ``homology_class`` is a member of the subspace.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the length of ``homology_class`` does not match the number of
    ///     generators of this subspace.
    fn contains(&self, homology_class: &PyHomologyClass) -> PyResult<bool> {
        if homology_class.inner.len() != self.inner.num_generators() {
            return Err(PyValueError::new_err(format!(
                "homology class length {} does not match the subspace generator count {}",
                homology_class.inner.len(),
                self.inner.num_generators()
            )));
        }
        Ok(self.inner.contains(&homology_class.inner))
    }
}

/// Registers the homology value types on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHomologyClass>()?;
    module.add_class::<PySubspace>()?;
    Ok(())
}
