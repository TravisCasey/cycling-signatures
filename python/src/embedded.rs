// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the `EmbeddedTrajectory` type.

use std::{path::PathBuf, sync::Arc};

use cycling_signatures::{EmbeddedTrajectory, ExecutionBackend};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    cover::PyCubicalCover, errors::to_pyerr, homology::PyHomologyClass, metric::metric_from_py,
    segment::segment_from_py, signature::PyCyclingSignature, trajectory::PyTrajectory,
};

/// A trajectory embedded in a cubical cover, ready for homological analysis.
///
/// Pairs a ``Trajectory`` with a ``CubicalCover`` built from it (or from a
/// denser trajectory through the same curve), validating that every point's
/// cube is present in the cover and that consecutive points land in adjacent
/// cubes. The result can be saved with ``save`` and reloaded with ``load``.
///
/// Typical pipeline, building the cover once from a dense trajectory and
/// embedding a thinned one against it (``resample_spacing`` and
/// ``downsample_spacing`` are the caller's own tuned values, fine and coarse
/// respectively)::
///
///     dense = cs.Trajectory.resample(interpolator, metric, resample_spacing)
///     cover = cs.CubicalCover(dense)
///     detection = dense.downsample(metric, downsample_spacing)
///     embedded = cs.EmbeddedTrajectory(detection, cover, metric)
///
/// The ``resolution`` method reports the largest distance between consecutive
/// points; any adjacency threshold passed to ``signature`` must be at least
/// this value.
///
/// Parameters
/// ----------
/// trajectory : ``Trajectory``
///     The trajectory to embed.
/// cover : ``CubicalCover``
///     The cubical cover to embed it in.
/// metric : ``Euclidean`` or ``SphereBundle``
///     The metric used to measure point spacing.
///
/// Raises
/// ------
/// ``ValueError``
///     If the trajectory and cover disagree on dimension, if a point's cube
///     is absent from the cover, or if consecutive points fall in
///     non-adjacent cubes.
/// ``TypeError``
///     If ``metric`` is not a recognized metric type.
#[pyclass(name = "EmbeddedTrajectory")]
pub(crate) struct PyEmbeddedTrajectory {
    pub(crate) inner: EmbeddedTrajectory,
}

#[pymethods]
impl PyEmbeddedTrajectory {
    /// Embeds ``trajectory`` in ``cover`` under ``metric``.
    ///
    /// For large, high-dimensional trajectories, consider saving the result
    /// with ``save`` and reloading it later with ``load`` rather than
    /// recomputing it.
    #[new]
    fn new(
        py: Python<'_>,
        trajectory: &Bound<'_, PyTrajectory>,
        cover: &Bound<'_, PyCubicalCover>,
        metric: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        let trajectory = Arc::clone(&trajectory.borrow().inner);
        let cover = Arc::clone(&cover.borrow().inner);
        let inner = py
            .detach(move || EmbeddedTrajectory::new(trajectory, cover, metric))
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Returns the cycling signature of the trajectory segment ``segment`` at
    /// an explicit adjacency ``threshold``.
    ///
    /// Detects all near-recurrent cycles within ``segment`` at ``threshold``
    /// and returns the filtered signature describing their homological
    /// content, ordered by birth (the adjacency threshold at which each
    /// independent class first enters). The signature is complete up to
    /// ``threshold``.
    ///
    /// This is not a cheap query. A signature has no cycle-length cap, so it
    /// evaluates the metric over every pair of points in the segment, a cost
    /// growing with the square of the segment length. For a large window,
    /// prefer ``CycleStorage.build`` with an explicit ``max_length``.
    ///
    /// Parameters
    /// ----------
    /// segment : range or tuple of int
    ///     A half-open range of point indices, given as a Python ``range`` or
    ///     a ``(start, stop)`` integer tuple.
    /// threshold : float
    ///     The largest endpoint distance admitted as a cycle. Must be at
    ///     least ``resolution`` and strictly below ``1.0`` (the cube side).
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
    ///     ``resolution`` or not below ``1.0``, or if a detected cycle's
    ///     endpoint points fall in non-adjacent cubes.
    /// ``IndexError``
    ///     If the segment indices are out of range.
    fn signature(
        &self,
        py: Python<'_>,
        segment: &Bound<'_, PyAny>,
        threshold: f64,
    ) -> PyResult<PyCyclingSignature> {
        let range = segment_from_py(segment)?;
        let embedded = &self.inner;
        let cycling_signature = py
            .detach(move || embedded.signature(range, threshold, &ExecutionBackend::Rayon))
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
    ///     A half-open range of point indices, given as a Python ``range`` or
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
    ///     points, or if the segment's endpoint points fall in non-adjacent
    ///     cubes.
    /// ``IndexError``
    ///     If the segment indices are out of bounds.
    fn cycle_class(&self, segment: &Bound<'_, PyAny>) -> PyResult<PyHomologyClass> {
        let range = segment_from_py(segment)?;
        if range.end < range.start + 2 {
            return Err(PyValueError::new_err(format!(
                "cycle segment {}..{} must contain at least two points",
                range.start, range.end
            )));
        }
        let homology_class = self.inner.cycle_class(range).map_err(to_pyerr)?;
        Ok(PyHomologyClass {
            inner: homology_class,
        })
    }

    /// Returns the largest distance between consecutive points in the
    /// embedded trajectory: its detection resolution.
    ///
    /// Any adjacency threshold passed to ``signature`` must be at least this
    /// value. Equals ``Trajectory.resolution`` under the embedded metric.
    ///
    /// Returns
    /// -------
    /// float
    ///     The largest distance between consecutive points.
    #[must_use]
    fn resolution(&self) -> f64 {
        self.inner.resolution()
    }

    /// Returns the cubical cover this trajectory is embedded in.
    ///
    /// The returned cover shares its underlying data with this embedded
    /// trajectory's cover rather than copying it, so it carries the exact
    /// generator basis this embedding's homology classes were computed
    /// against; saving it and reloading it later maintains that basis for
    /// comparison against classes computed here.
    ///
    /// Returns
    /// -------
    /// ``CubicalCover``
    ///     The wrapped cover.
    #[must_use]
    fn cover(&self) -> PyCubicalCover {
        PyCubicalCover {
            inner: Arc::clone(self.inner.cover()),
        }
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
    /// metric : ``Euclidean`` or ``SphereBundle``
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
