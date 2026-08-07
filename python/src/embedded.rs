// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the `EmbeddedTrajectory` type.

use std::{path::PathBuf, sync::Arc};

use cycling_signatures::{EmbeddedTrajectory, ExecutionBackend, Interpolator, Metric};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{
    convert::{parallel_backend, segment_from_py},
    cover::PyCubicalCover,
    errors::to_pyerr,
    homology::PyHomologyClass,
    interpolation::{PyCubicSpline, PySphereBundleInterpolator},
    metric::{metric_from_py, metric_to_py},
    signature::PyCyclingSignature,
    trajectory::PyTrajectory,
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

    /// Runs the entire embedding pipeline over ``interpolator`` in one call.
    ///
    /// Resamples ``interpolator`` at ``resample_spacing`` into a dense
    /// trajectory, builds a cubical cover from that dense trajectory, thins
    /// the dense trajectory to ``downsample_spacing``, and embeds the thinned
    /// trajectory in the cover. The dense intermediate is discarded once the
    /// cover is built.
    ///
    /// Finer control, and retention of the dense trajectory, is possible by
    /// instead composing ``Trajectory.resample``, ``CubicalCover``,
    /// ``Trajectory.downsample``, and ``EmbeddedTrajectory``.
    ///
    /// Parameters
    /// ----------
    /// interpolator : ``CubicSpline`` or ``SphereBundleInterpolator``
    ///     The interpolator to sample.
    /// metric : ``Euclidean`` or ``SphereBundle``
    ///     The metric used to measure point spacing and build the embedding.
    /// resample_spacing : float
    ///     The largest allowed distance between consecutive points of the
    ///     dense trajectory the cover is built from. Must be positive.
    /// downsample_spacing : float
    ///     The largest allowed distance between consecutive points of the
    ///     thinned, embedded trajectory.
    /// parallel : bool, optional
    ///     Whether to distribute the cover build across a thread pool.
    ///     Defaults to ``True``; pass ``False`` to run sequentially on the
    ///     calling thread.
    ///
    /// Returns
    /// -------
    /// ``EmbeddedTrajectory``
    ///     The embedded, thinned trajectory.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``resample_spacing`` is not positive (including NaN), if the
    ///     interpolator has fewer than two knots, if a sampled value is not
    ///     finite, if bisection cannot reach ``resample_spacing``, if the
    ///     interpolator's samples have zero columns, if a cube coordinate falls
    ///     outside the supported integer range, if consecutive dense points
    ///     fall in non-adjacent cubes, if ``downsample_spacing`` is below the
    ///     dense trajectory's own consecutive-point distance, or if consecutive
    ///     thinned points fall in non-adjacent cubes.
    /// ``TypeError``
    ///     If ``interpolator`` or ``metric`` is not a recognized type.
    #[staticmethod]
    #[pyo3(signature = (interpolator, metric, resample_spacing, downsample_spacing, *, parallel = true))]
    fn from_interpolator(
        py: Python<'_>,
        interpolator: &Bound<'_, PyAny>,
        metric: &Bound<'_, PyAny>,
        resample_spacing: f64,
        downsample_spacing: f64,
        parallel: bool,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        let backend = parallel_backend(parallel);
        if let Ok(spline) = interpolator.cast::<PyCubicSpline>() {
            let spline = Arc::clone(&spline.borrow().inner);
            return Self::embedded_from_interpolator(
                py,
                spline,
                metric,
                resample_spacing,
                downsample_spacing,
                backend,
            );
        }
        if let Ok(bundle) = interpolator.cast::<PySphereBundleInterpolator>() {
            let bundle = bundle.borrow().inner.clone();
            return Self::embedded_from_interpolator(
                py,
                bundle,
                metric,
                resample_spacing,
                downsample_spacing,
                backend,
            );
        }
        Err(PyTypeError::new_err(format!(
            "expected a CubicSpline or SphereBundleInterpolator, got {}",
            interpolator.get_type().name()?
        )))
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
    /// parallel : bool, optional
    ///     Whether to distribute the work across a thread pool. Defaults to
    ///     ``True``; pass ``False`` to run sequentially on the calling
    ///     thread.
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
    #[pyo3(signature = (segment, threshold, *, parallel = true))]
    fn signature(
        &self,
        py: Python<'_>,
        segment: &Bound<'_, PyAny>,
        threshold: f64,
        parallel: bool,
    ) -> PyResult<PyCyclingSignature> {
        let range = segment_from_py(segment)?;
        let backend = parallel_backend(parallel);
        let embedded = &self.inner;
        let cycling_signature = py
            .detach(move || embedded.signature(range, threshold, &backend))
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

    /// Returns the embedded trajectory.
    ///
    /// The returned trajectory shares its underlying data rather than copying
    /// it.
    ///
    /// Returns
    /// -------
    /// ``Trajectory``
    ///     The wrapped trajectory.
    #[must_use]
    fn trajectory(&self) -> PyTrajectory {
        PyTrajectory {
            inner: Arc::clone(self.inner.trajectory()),
        }
    }

    /// Returns the number of points in the embedded trajectory.
    fn __len__(&self) -> usize {
        self.inner.trajectory().len()
    }

    /// Returns the metric this embedded trajectory measures distances under.
    ///
    /// Useful for introspection: a caller holding an embedding built or
    /// returned elsewhere can recover which metric it uses without having
    /// tracked the value separately.
    ///
    /// Returns
    /// -------
    /// ``Euclidean`` or ``SphereBundle``
    ///     The metric this embedding was constructed with.
    fn metric(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metric_to_py(py, self.inner.metric())
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

    /// Saves the embedded trajectory to three files.
    ///
    /// The trajectory data is written to ``trajectory_path``, the cubical
    /// cover data to ``cover_path``, and an envelope recording this
    /// embedding's metric and both files' fingerprints to ``embedded_path``.
    /// All three files must be loaded together via ``load``.
    ///
    /// Parameters
    /// ----------
    /// embedded_path : str or ``os.PathLike``
    ///     The destination for the envelope.
    /// trajectory_path : str or ``os.PathLike``
    ///     The destination for the trajectory data.
    /// cover_path : str or ``os.PathLike``
    ///     The destination for the cubical cover data.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If any of the three files cannot be written.
    fn save(
        &self,
        embedded_path: PathBuf,
        trajectory_path: PathBuf,
        cover_path: PathBuf,
    ) -> PyResult<()> {
        self.inner
            .save(embedded_path, trajectory_path, cover_path)
            .map_err(to_pyerr)
    }

    /// Loads an embedded trajectory from three previously saved files.
    ///
    /// Reads the envelope from ``embedded_path``, the trajectory from
    /// ``trajectory_path``, and the cubical cover from ``cover_path``, then
    /// reconstructs the embedded trajectory using the metric recorded in the
    /// envelope, after verifying that the loaded trajectory and cover match
    /// the fingerprints the envelope recorded.
    ///
    /// Parameters
    /// ----------
    /// embedded_path : str or ``os.PathLike``
    ///     The source of the envelope.
    /// trajectory_path : str or ``os.PathLike``
    ///     The source of the trajectory data.
    /// cover_path : str or ``os.PathLike``
    ///     The source of the cubical cover data.
    ///
    /// Returns
    /// -------
    /// ``EmbeddedTrajectory``
    ///     The reloaded embedded trajectory.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If any of the three files cannot be read.
    /// ``FormatVersionMismatchError``
    ///     If a file was written by an incompatible version of the library.
    /// ``ValueError``
    ///     If the stored data is inconsistent, or if the loaded trajectory or
    ///     cover does not match the fingerprint the envelope recorded.
    #[staticmethod]
    fn load(
        embedded_path: PathBuf,
        trajectory_path: PathBuf,
        cover_path: PathBuf,
    ) -> PyResult<Self> {
        let inner = EmbeddedTrajectory::load(embedded_path, trajectory_path, cover_path)
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

impl PyEmbeddedTrajectory {
    /// Runs [`EmbeddedTrajectory::from_interpolator`] with the interpreter
    /// detached, so other Python threads run during the pipeline.
    fn embedded_from_interpolator<I: Interpolator + Send>(
        py: Python<'_>,
        interpolator: I,
        metric: Metric,
        resample_spacing: f64,
        downsample_spacing: f64,
        backend: ExecutionBackend,
    ) -> PyResult<Self> {
        let inner = py
            .detach(move || {
                EmbeddedTrajectory::from_interpolator(
                    &interpolator,
                    metric,
                    resample_spacing,
                    downsample_spacing,
                    &backend,
                )
            })
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Registers the ``EmbeddedTrajectory`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEmbeddedTrajectory>()?;
    Ok(())
}
