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
/// The ``resolution`` method reports the largest distance between consecutive
/// points, which must be below ``1.0``, the cube side length.
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
///     If the trajectory and cover disagree on dimension, if ``metric`` is
///     ``SphereBundle`` and the trajectory has an odd coordinate count, if a
///     point's cube is absent from the cover, if consecutive points fall in
///     non-adjacent cubes, or if the largest distance between consecutive
///     points reaches ``1.0``, the cube side length.
/// ``TypeError``
///     If ``metric`` is not a recognized metric type.
///
/// Examples
/// --------
/// Build the cover once from a dense trajectory, then embed a thinned copy of
/// it against that cover::
///
///     import numpy as np
///     import cycling_signatures as cs
///
///     RESAMPLE_SPACING = 0.1
///     DOWNSAMPLE_SPACING = 0.3
///
///     knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
///     values = np.array(
///         [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]
///     )
///     spline = cs.CubicSpline(knots, values)
///     metric = cs.Euclidean()
///
///     dense = cs.Trajectory.resample(spline, metric, RESAMPLE_SPACING)
///     cover = cs.CubicalCover(dense)
///     detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
///     embedded = cs.EmbeddedTrajectory(detection, cover, metric)
#[pyclass(name = "EmbeddedTrajectory")]
pub(crate) struct PyEmbeddedTrajectory {
    pub(crate) inner: Arc<EmbeddedTrajectory>,
}

#[pymethods]
impl PyEmbeddedTrajectory {
    /// Embeds ``trajectory`` in ``cover`` under ``metric``.
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
        Ok(Self {
            inner: Arc::new(inner),
        })
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
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``resample_spacing`` is not positive (including NaN), if the
    ///     interpolator has fewer than two knots, if a sampled value is not
    ///     finite, if bisection cannot reach ``resample_spacing``, if
    ///     ``metric`` is ``SphereBundle`` and the interpolator emits an odd
    ///     coordinate count, if the interpolator's samples have zero columns,
    ///     if a cube coordinate falls outside the supported integer range, if
    ///     consecutive dense points fall in non-adjacent cubes, if
    ///     ``downsample_spacing`` is below the dense trajectory's own
    ///     consecutive-point distance, if consecutive thinned points fall in
    ///     non-adjacent cubes, or if the detection trajectory's largest
    ///     distance between consecutive points reaches ``1.0``, the cube side
    ///     length.
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

    /// Returns the cycling signature of the trajectory segment ``segment``.
    ///
    /// Detects all near-recurrent cycles within ``segment`` and returns the
    /// filtered signature describing their homological content, ordered by
    /// birth (the endpoint distance at which each independent class first
    /// enters). Point pairs strictly closer than ``1.0``, the cube side
    /// length, are admitted as endpoints of near-recurrent cycles.
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
    /// parallel : bool, optional
    ///     Whether to distribute the work across a thread pool. Defaults to
    ///     ``True``; pass ``False`` to run sequentially on the calling
    ///     thread.
    ///
    /// Returns
    /// -------
    /// ``CyclingSignature``
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range, or if a detected cycle's
    ///     endpoints lie in non-adjacent cubes.
    /// ``IndexError``
    ///     If the segment indices are out of range.
    #[pyo3(signature = (segment, *, parallel = true))]
    fn signature(
        &self,
        py: Python<'_>,
        segment: &Bound<'_, PyAny>,
        parallel: bool,
    ) -> PyResult<PyCyclingSignature> {
        let range = segment_from_py(segment)?;
        let backend = parallel_backend(parallel);
        let embedded = Arc::clone(&self.inner);
        let cycling_signature = py
            .detach(move || embedded.signature(range, &backend))
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
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range, if it contains fewer than two
    ///     points, or if the segment's endpoint points fall in non-adjacent
    ///     cubes.
    /// ``IndexError``
    ///     If the segment indices are out of bounds.
    fn cycle_class(&self, py: Python<'_>, segment: &Bound<'_, PyAny>) -> PyResult<PyHomologyClass> {
        let range = segment_from_py(segment)?;
        if range.end < range.start + 2 {
            return Err(PyValueError::new_err(format!(
                "cycle segment {}..{} must contain at least two points",
                range.start, range.end
            )));
        }
        let embedded = Arc::clone(&self.inner);
        let homology_class = py
            .detach(move || embedded.cycle_class(range))
            .map_err(to_pyerr)?;
        Ok(PyHomologyClass {
            inner: homology_class,
        })
    }

    /// Returns the largest distance between consecutive points in the
    /// embedded trajectory: its detection resolution.
    ///
    /// The constructor validates that this value is below ``1.0``, the cube
    /// side length. Equals ``Trajectory.resolution`` under the embedded metric.
    ///
    /// Returns
    /// -------
    /// float
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
    /// Recovers the metric an embedding was constructed with, including one
    /// reloaded with ``load``, without the caller tracking it separately.
    ///
    /// Returns
    /// -------
    /// ``Euclidean`` or ``SphereBundle``
    fn metric(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metric_to_py(py, self.inner.metric())
    }

    /// Returns the cubical cover this trajectory is embedded in.
    ///
    /// The returned cover shares its underlying data rather than copying it,
    /// so it carries the generator basis this embedding's homology classes
    /// were computed against.
    ///
    /// Returns
    /// -------
    /// ``CubicalCover``
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
    /// For a large, high-dimensional trajectory, saving an embedding and
    /// reloading it later costs less than rebuilding the cover.
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
        py: Python<'_>,
        embedded_path: PathBuf,
        trajectory_path: PathBuf,
        cover_path: PathBuf,
    ) -> PyResult<()> {
        let embedded = Arc::clone(&self.inner);
        py.detach(move || embedded.save(embedded_path, trajectory_path, cover_path))
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
        py: Python<'_>,
        embedded_path: PathBuf,
        trajectory_path: PathBuf,
        cover_path: PathBuf,
    ) -> PyResult<Self> {
        let inner = py
            .detach(move || EmbeddedTrajectory::load(embedded_path, trajectory_path, cover_path))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
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
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// Registers the ``EmbeddedTrajectory`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEmbeddedTrajectory>()?;
    Ok(())
}
