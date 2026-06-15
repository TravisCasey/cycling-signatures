// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the `EmbeddedTrajectory` type.

use std::path::PathBuf;

use cycling_signatures::{EmbeddedTrajectory, ExecutionBackend};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    errors::to_pyerr,
    homology::{PyCyclingSignature, PyHomologyClass},
    metric::metric_from_py,
    segment::segment_from_py,
    trajectory::PyTrajectory,
};

/// A trajectory embedded in a cubical complex, ready for homological analysis.
///
/// Wraps the output of embedding a ``Trajectory`` into cubical covers and
/// computing the cover's homology generators. The result can be saved with
/// ``save`` and reloaded with ``load``.
///
/// The ``bound`` method reports the largest consecutive-point distance seen
/// during embedding; any adjacency threshold passed to ``signature`` must be at
/// least this value.
///
/// Parameters
/// ----------
/// trajectory : ``Trajectory``
///     The trajectory to embed.
/// metric : ``Euclidean``, ``Chebyshev``, or ``SphereBundle``
///     The metric used to build the cubical cover.
///
/// Raises
/// ------
/// ``ValueError``
///     If consecutive trajectory points fall in non-adjacent cubes, or if a
///     coordinate lies outside the supported integer range.
/// ``TypeError``
///     If ``metric`` is not a recognized metric type.
#[pyclass(name = "EmbeddedTrajectory")]
pub(crate) struct PyEmbeddedTrajectory {
    pub(crate) inner: EmbeddedTrajectory,
}

#[pymethods]
impl PyEmbeddedTrajectory {
    /// Embeds ``trajectory`` in a cubical complex under ``metric``.
    ///
    /// Builds the cubical cover for the trajectory and computes its homology
    /// generators. For large, high-dimensional trajectories, consider saving
    /// the result with ``save`` and reloading it later with ``load`` rather
    /// than recomputing it.
    #[new]
    fn new(
        py: Python<'_>,
        trajectory: &Bound<'_, PyTrajectory>,
        metric: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        let trajectory = trajectory.borrow().inner.clone();
        let inner = py
            .detach(move || EmbeddedTrajectory::new(trajectory, metric, &ExecutionBackend::Rayon))
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Returns the cycling signature of the trajectory segment ``segment``.
    ///
    /// Detects all near-recurrent cycles within ``segment`` whose endpoint
    /// distance is at most ``threshold``, and returns the signature describing
    /// the homological content of those cycles.
    ///
    /// Parameters
    /// ----------
    /// segment : range or tuple of int
    ///     A half-open range of sample indices, given as a Python ``range`` or
    ///     a ``(start, stop)`` integer tuple.
    /// threshold : float
    ///     The largest endpoint distance admitted as a cycle. Must be at least
    ///     ``bound``.
    ///
    /// Returns
    /// -------
    /// ``CyclingSignature``
    ///     The homological content of the detected cycles.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range, if ``threshold`` is below
    ///     ``bound``, if the segment indices are out of range, or if a detected
    ///     cycle's consecutive or endpoint points fall in non-adjacent cubes.
    fn signature(
        &self,
        py: Python<'_>,
        segment: &Bound<'_, PyAny>,
        threshold: f64,
    ) -> PyResult<PyCyclingSignature> {
        let range = segment_from_py(segment)?;
        let embedded = &self.inner;
        let cycling_signature = py
            .detach(move || embedded.signature(range, threshold))
            .map_err(to_pyerr)?;
        Ok(PyCyclingSignature {
            inner: cycling_signature,
        })
    }

    /// Returns the homology class of the cycle described by ``segment``.
    ///
    /// Walks the forward path from the segment start to one before the segment
    /// stop and closes it back to the start, then returns the resulting cycle's
    /// class in the cover's homology.
    ///
    /// Parameters
    /// ----------
    /// segment : range or tuple of int
    ///     A half-open range of sample indices, given as a Python ``range`` or
    ///     a ``(start, stop)`` integer tuple. Must contain at least two points.
    ///
    /// Returns
    /// -------
    /// ``HomologyClass``
    ///     The homology class of the closed cycle.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range, if it contains fewer than two
    ///     points, if the segment indices are out of bounds, or if the
    ///     segment's consecutive or endpoint points fall in non-adjacent cubes.
    fn cycle_class(&self, segment: &Bound<'_, PyAny>) -> PyResult<PyHomologyClass> {
        let range = segment_from_py(segment)?;
        if range.end < range.start + 2 {
            return Err(PyValueError::new_err(
                "cycle segment must contain at least two points",
            ));
        }
        let homology_class = self.inner.cycle_class(range).map_err(to_pyerr)?;
        Ok(PyHomologyClass {
            inner: homology_class,
        })
    }

    /// Returns the largest consecutive-point distance in the embedded
    /// trajectory.
    ///
    /// Any adjacency threshold passed to ``signature`` must be at least this
    /// value.
    ///
    /// Returns
    /// -------
    /// float
    ///     The largest consecutive-point distance.
    #[must_use]
    fn bound(&self) -> f64 {
        self.inner.bound()
    }

    /// Returns a content fingerprint of the embedded trajectory.
    ///
    /// Two embedded trajectories built from identical trajectory data, cover
    /// structure, and metric have the same fingerprint. Typically used to
    /// verify correct serialization and deserialization.
    ///
    /// Returns
    /// -------
    /// int
    ///     A fingerprint identifying the embedded trajectory.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Saves the embedded trajectory to a pair of files.
    ///
    /// The trajectory data is written to ``trajectory_path`` and the cubical
    /// cover data to ``cover_path``. Both files must be loaded together via
    /// ``load``.
    ///
    /// Parameters
    /// ----------
    /// trajectory_path : str or ``os.PathLike``
    ///     The destination for the trajectory data.
    /// cover_path : str or ``os.PathLike``
    ///     The destination for the cubical cover data.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If either file cannot be written.
    fn save(&self, trajectory_path: PathBuf, cover_path: PathBuf) -> PyResult<()> {
        self.inner
            .save(trajectory_path, cover_path)
            .map_err(to_pyerr)
    }

    /// Loads an embedded trajectory from a pair of previously saved files.
    ///
    /// Reads the trajectory from ``trajectory_path`` and the cubical cover from
    /// ``cover_path``, then reconstructs the embedded trajectory using
    /// ``metric``.
    ///
    /// Parameters
    /// ----------
    /// trajectory_path : str or ``os.PathLike``
    ///     The source of the trajectory data.
    /// cover_path : str or ``os.PathLike``
    ///     The source of the cubical cover data.
    /// metric : ``Euclidean``, ``Chebyshev``, or ``SphereBundle``
    ///     The metric the embedding was built with.
    ///
    /// Returns
    /// -------
    /// ``EmbeddedTrajectory``
    ///     The reloaded embedded trajectory.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If either file cannot be read.
    /// ``FormatVersionMismatchError``
    ///     If a file was written by an incompatible version of the library.
    /// ``ValueError``
    ///     If the stored data is inconsistent.
    /// ``TypeError``
    ///     If ``metric`` is not a recognized metric type.
    #[staticmethod]
    fn load(
        trajectory_path: PathBuf,
        cover_path: PathBuf,
        metric: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        let inner =
            EmbeddedTrajectory::load(trajectory_path, cover_path, metric).map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Registers the ``EmbeddedTrajectory`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEmbeddedTrajectory>()?;
    Ok(())
}
