// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the homology value types returned by signature queries.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use cycling_signatures::{CycleComponent, CyclingSignature, F2, F2Subspace, F2Vector};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
};

use crate::segment::resolve_index;

/// Computes a hash of a value, used to give the equatable value types a
/// `__hash__` consistent with their `__eq__`.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The homology class of a recurrent cycle, as a vector over ``F_2``.
///
/// The cubical cover of a trajectory contains a detected set of independent
/// loops (its first rank homology generators); a cycle's homology class records
/// which of them a near-recurrent trajectory cycle wraps. Because the
/// coefficients live in the two-element field ``F_2`` (the integers modulo 2),
/// every entry is either zero or one: the class is a zero/one vector with one
/// entry per homology generator of the cover.
///
/// The entries are coordinates in a chosen basis of generators, and that basis
/// is not canonical: the same cover can present its generators in a different
/// basis from one computation to the next, so the same class may appear with
/// different coordinates. Comparing two classes entry by entry is therefore
/// only meaningful when both use the same basis, and is rarely the operation
/// you want. To compare a span of cycling results in the same basis, use
/// ``Subspace``. Classes from different covers or bases are not currently
/// comparable.
///
/// ``len`` gives the number of generators, and indexing returns the zero or one
/// entry at a generator (negative indices count from the end). Equality
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
/// This is the right unit for comparing cycling results, since a signature's
/// components may contribute linearly dependent classes.
///
/// The comparison is still taken in the cover's generator basis, so it requires
/// both subspaces to use that same basis. A cycling signature over a fixed
/// cover is expected to be independent of the basis chosen for its homology,
/// but the library provides no basis-independent comparison: subspaces from
/// different covers, or from the same cover expressed in a different generator
/// basis, are not currently comparable.
#[pyclass(name = "Subspace")]
pub(crate) struct PySubspace {
    pub(crate) inner: F2Subspace,
}

/// One connected group of recurrent cycle segments sharing a homology class.
///
/// A cycling signature decomposes into connected components of near-recurrent
/// segments; every segment in a component carries the same homology class, and
/// every component holds at least one segment.
#[pyclass(name = "CycleComponent")]
pub(crate) struct PyCycleComponent {
    cycles: Vec<(usize, usize)>,
    homology_class: F2Vector,
}

/// The cycling signature of a trajectory segment.
///
/// The signature is the cycle space (a ``Subspace``) spanned by the homology
/// classes of the detected recurrence components. Its ``rank`` is the number of
/// independent cycling classes it carries, and ``components`` exposes the
/// per-component breakdown.
#[pyclass(name = "CyclingSignature")]
pub(crate) struct PyCyclingSignature {
    pub(crate) inner: CyclingSignature,
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
    /// The array has one entry per generator, with ``1`` at each generator the
    /// class includes and ``0`` elsewhere. The layout depends on the cover's
    /// non-canonical choice of generator basis, so it is meaningful only
    /// relative to the basis that produced this class.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A one-dimensional array of ``uint8`` zeros and ones, one per
    ///     generator.
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
    ///     The ambient dimension, equal to the length of each class vector.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Returns the basis classes that span this subspace.
    ///
    /// The basis is the reduced row echelon form of the spanning classes, in
    /// the cover's generator coordinates, so a rank-r subspace returns r
    /// classes and the trivial subspace returns an empty list.
    ///
    /// Returns
    /// -------
    /// list of ``HomologyClass``
    ///     The canonical basis of the subspace, one class per dimension.
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
    /// The class and the subspace must be expressed in the same cover's
    /// generator basis; the result is not meaningful across covers or across
    /// differing generator bases.
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

#[pymethods]
impl PyCycleComponent {
    /// Returns the cycle segments in this component as ``(start, stop)`` pairs.
    ///
    /// Each pair is a half-open range of sample indices covering the cycle
    /// ``start..stop``. Sample indices index into the original input data,
    /// the range ``0`` up to ``trajectory.original_count()``.
    ///
    /// Returns
    /// -------
    /// list of tuple of int
    ///     One ``(start, stop)`` pair per cycle in the component.
    #[must_use]
    fn cycles(&self) -> Vec<(usize, usize)> {
        self.cycles.clone()
    }

    /// Returns the homology class shared by every cycle in this component.
    ///
    /// Returns
    /// -------
    /// ``HomologyClass``
    ///     The class carried by all cycles in the component.
    #[must_use]
    fn homology_class(&self) -> PyHomologyClass {
        PyHomologyClass {
            inner: self.homology_class.clone(),
        }
    }

    /// Returns a string representation of the component.
    fn __repr__(&self) -> String {
        format!("CycleComponent(cycles={})", self.cycles.len())
    }
}

#[pymethods]
impl PyCyclingSignature {
    /// Returns the subspace spanned by the component classes.
    ///
    /// This is the signature's value identity: two signatures with the same
    /// span compare equal regardless of how many components contributed.
    ///
    /// Returns
    /// -------
    /// ``Subspace``
    ///     The cycle space spanned by the component classes.
    #[must_use]
    fn span(&self) -> PySubspace {
        PySubspace {
            inner: self.inner.span().clone(),
        }
    }

    /// Returns the number of independent cycling classes in the signature.
    ///
    /// Returns
    /// -------
    /// int
    ///     The rank of the spanned cycle space.
    #[must_use]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns the per-component decomposition of the signature.
    ///
    /// Each entry pairs the cycle segments of one connected recurrence
    /// component with the homology class those cycles share.
    ///
    /// Returns
    /// -------
    /// list of ``CycleComponent``
    ///     One entry per connected recurrence component.
    #[must_use]
    fn components(&self) -> Vec<PyCycleComponent> {
        self.inner
            .components()
            .iter()
            .map(component_to_py)
            .collect()
    }

    /// Returns whether this signature equals ``other`` by span comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Returns a hash consistent with equality, derived from the spanned
    /// subspace.
    fn __hash__(&self) -> u64 {
        hash_of(self.inner.span())
    }

    /// Returns a string representation of the signature.
    fn __repr__(&self) -> String {
        format!(
            "CyclingSignature(rank={}, components={})",
            self.inner.rank(),
            self.inner.components().len()
        )
    }
}

fn component_to_py(component: &CycleComponent) -> PyCycleComponent {
    PyCycleComponent {
        cycles: component
            .cycles()
            .iter()
            .map(|range| (range.start, range.end))
            .collect(),
        homology_class: component.class().clone(),
    }
}

/// Registers the homology value types on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHomologyClass>()?;
    module.add_class::<PySubspace>()?;
    module.add_class::<PyCycleComponent>()?;
    module.add_class::<PyCyclingSignature>()?;
    Ok(())
}
