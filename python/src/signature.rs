// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the filtered cycling signature returned by signature
//! queries.

use cycling_signatures::{CyclingSignature, SignatureGenerator};
use pyo3::prelude::*;

use crate::{
    errors::to_pyerr,
    homology::{PyHomologyClass, PySubspace},
};

/// The cycling signature of a trajectory segment: a filtered ``F_2`` subspace
/// of cover homology, ordered by the adjacency threshold ("birth") at which
/// each independent class first enters.
///
/// ``span`` is the full-band subspace, complete up to ``threshold_max``;
/// ``span_at`` and ``rank_at`` restrict to a smaller threshold. ``births``
/// and ``generators`` give the per-generator breakdown, aligned by index.
#[pyclass(name = "CyclingSignature")]
pub(crate) struct PyCyclingSignature {
    pub(crate) inner: CyclingSignature,
}

#[pymethods]
impl PyCyclingSignature {
    /// Returns the full-band subspace spanned by every generator.
    ///
    /// Returns
    /// -------
    /// ``Subspace``
    ///     The cycle space spanned by every generator, complete up to
    ///     ``threshold_max``.
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
    ///     The rank of the full-band spanned cycle space.
    #[must_use]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Returns the number of independent cycling classes with birth at most
    /// ``threshold``.
    ///
    /// Parameters
    /// ----------
    /// threshold : float
    ///     The adjacency threshold to query.
    ///
    /// Returns
    /// -------
    /// int
    ///     The rank of the subspace at ``threshold``.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``threshold`` exceeds ``threshold_max``, or is NaN.
    fn rank_at(&self, threshold: f64) -> PyResult<usize> {
        self.inner.rank_at(threshold).map_err(to_pyerr)
    }

    /// Returns the subspace spanned by generators with birth at most
    /// ``threshold``.
    ///
    /// Parameters
    /// ----------
    /// threshold : float
    ///     The adjacency threshold to query.
    ///
    /// Returns
    /// -------
    /// ``Subspace``
    ///     The cycle space at ``threshold``.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``threshold`` exceeds ``threshold_max``, or is NaN.
    fn span_at(&self, threshold: f64) -> PyResult<PySubspace> {
        self.inner
            .span_at(threshold)
            .map(|span| PySubspace { inner: span })
            .map_err(to_pyerr)
    }

    /// Returns the birth threshold of every generator, ascending.
    ///
    /// Returns
    /// -------
    /// list of float
    ///     One birth per generator, aligned by index with ``generators``.
    #[must_use]
    fn births(&self) -> Vec<f64> {
        self.inner
            .generators()
            .iter()
            .map(SignatureGenerator::birth)
            .collect()
    }

    /// Returns the homology class of every generator, in birth-ascending order.
    ///
    /// Returns
    /// -------
    /// list of ``HomologyClass``
    ///     One class per generator, aligned by index with ``births``.
    #[must_use]
    fn generators(&self) -> Vec<PyHomologyClass> {
        self.inner
            .generators()
            .iter()
            .map(|generator| PyHomologyClass {
                inner: generator.class().clone(),
            })
            .collect()
    }

    /// Returns the largest adjacency threshold this signature is complete for.
    ///
    /// Returns
    /// -------
    /// float
    ///     The inclusive upper end of the valid query range.
    #[must_use]
    fn threshold_max(&self) -> f64 {
        self.inner.threshold_max()
    }

    /// Returns the number of cover generators, the ambient dimension of the
    /// spanned subspace.
    ///
    /// Returns
    /// -------
    /// int
    ///     The ambient dimension.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Returns a string representation of the signature.
    fn __repr__(&self) -> String {
        format!(
            "CyclingSignature(rank={}, threshold_max={:?})",
            self.inner.rank(),
            self.inner.threshold_max()
        )
    }
}

/// Registers the cycling signature type on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCyclingSignature>()?;
    Ok(())
}
