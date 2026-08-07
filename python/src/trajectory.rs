// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the core `Trajectory` type.

use std::{path::PathBuf, sync::Arc};

use cycling_signatures::{Interpolator, Metric, Trajectory};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{
    errors::to_pyerr,
    interpolation::{PyCubicSpline, PySphereBundleInterpolator},
    metric::{check_even_dimension, metric_from_py},
};

/// An ordered array of points together with a strictly increasing
/// parameterization.
///
/// Constructed directly from points, densely from an interpolator via
/// ``resample``, or thinned from another trajectory via ``downsample``.
/// Nothing in cube covering, cycle detection, or walking reads the
/// parameterization: it is carried through unchanged for the caller to
/// interpret (integration time, arc length, or a raw sample's row number).
///
/// Parameters
/// ----------
/// points : ndarray
///     A two-dimensional array with at least one row, each row one point.
/// parameters : ndarray, optional
///     A one-dimensional array of strictly increasing parameter values, one
///     per row of ``points``. Defaults to the index parameterization
///     ``0.0, 1.0, ...``.
///
/// Raises
/// ------
/// ``ValueError``
///     If ``points`` has zero rows, if any coordinate is not finite, if
///     ``parameters`` does not have one value per row, or if ``parameters``
///     is not strictly increasing.
#[pyclass(name = "Trajectory")]
pub(crate) struct PyTrajectory {
    pub(crate) inner: Arc<Trajectory>,
}

#[pymethods]
impl PyTrajectory {
    /// Constructs a trajectory from a two-dimensional array of points.
    #[new]
    #[pyo3(signature = (points, parameters=None))]
    fn new(
        points: PyReadonlyArray2<'_, f64>,
        parameters: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let inner = match parameters {
            Some(parameters) => {
                let parameters: Vec<f64> = parameters.as_array().iter().copied().collect();
                Trajectory::with_parameters(points.as_array(), &parameters).map_err(to_pyerr)?
            },
            None => Trajectory::new(points.as_array()).map_err(to_pyerr)?,
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Constructs a trajectory by sampling an interpolator, inserting points
    /// between its knots until consecutive points are no more than
    /// ``spacing`` apart under ``metric``. Records each emitted point's
    /// interpolation parameter.
    ///
    /// Parameters
    /// ----------
    /// interpolator : ``CubicSpline`` or ``SphereBundleInterpolator``
    ///     The interpolator to sample.
    /// metric : ``Euclidean`` or ``SphereBundle``
    ///     The metric that measures point spacing.
    /// spacing : float
    ///     The largest allowed distance between consecutive points. Must be
    ///     positive. A spacing of at most the cube side, ``1.0``, keeps
    ///     consecutive points in intersecting cubes; a coarser spacing is
    ///     accepted here and rejected when a cover is built from the result.
    ///
    /// Returns
    /// -------
    /// ``Trajectory``
    ///     The resampled trajectory with inserted points and their recorded
    ///     parameters.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``spacing`` is not positive (including if it is NaN), if the
    ///     interpolator has fewer than two knots, if a sampled value is not
    ///     finite, or if bisection cannot reach ``spacing``.
    /// ``TypeError``
    ///     If ``interpolator`` or ``metric`` is not a recognized type.
    #[staticmethod]
    fn resample(
        py: Python<'_>,
        interpolator: &Bound<'_, PyAny>,
        metric: &Bound<'_, PyAny>,
        spacing: f64,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        if let Ok(spline) = interpolator.cast::<PyCubicSpline>() {
            let spline = Arc::clone(&spline.borrow().inner);
            return Self::resampled(py, spline, metric, spacing);
        }
        if let Ok(bundle) = interpolator.cast::<PySphereBundleInterpolator>() {
            let bundle = bundle.borrow().inner.clone();
            return Self::resampled(py, bundle, metric, spacing);
        }
        Err(PyTypeError::new_err(format!(
            "expected a CubicSpline or SphereBundleInterpolator, got {}",
            interpolator.get_type().name()?
        )))
    }

    /// Returns a new trajectory thinned to a subset of this trajectory's
    /// points at most ``spacing`` apart under ``metric``.
    ///
    /// A greedy forward walk always keeps the first and last point, and
    /// keeps an intermediate point once the next point would fall further
    /// than ``spacing`` from the last kept point. Any spacing up to the
    /// intended detection threshold is valid: the threshold has to clear the
    /// output's own consecutive-point distance, which this spacing bounds.
    ///
    /// Only the lower end is validated here. A spacing coarse enough to put
    /// consecutive kept points more than one cube apart surfaces later, when
    /// embedding, as a non-adjacent-cubes error.
    ///
    /// Parameters
    /// ----------
    /// metric : ``Euclidean`` or ``SphereBundle``
    ///     The metric that measures point spacing.
    /// spacing : float
    ///     The largest allowed distance between consecutive kept points.
    ///
    /// Returns
    /// -------
    /// ``Trajectory``
    ///     A new, thinned trajectory.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``spacing`` is less than this trajectory's own maximum
    ///     consecutive-point distance (including if it is NaN), or if
    ///     ``metric`` is ``SphereBundle`` and this trajectory has an odd
    ///     coordinate count.
    /// ``TypeError``
    ///     If ``metric`` is not a recognized type.
    fn downsample(
        &self,
        py: Python<'_>,
        metric: &Bound<'_, PyAny>,
        spacing: f64,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        check_even_dimension(metric, self.inner.dimension())?;
        let trajectory = Arc::clone(&self.inner);
        let inner = py
            .detach(move || trajectory.downsample(metric, spacing))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Returns the maximum distance between consecutive points under
    /// ``metric``: the finest separation this trajectory resolves. Zero when
    /// the trajectory has fewer than two points.
    ///
    /// This is the floor for a valid ``downsample`` spacing, and the
    /// value ``EmbeddedTrajectory.resolution`` reports for the trajectory an
    /// embedding was built over.
    ///
    /// Parameters
    /// ----------
    /// metric : ``Euclidean`` or ``SphereBundle``
    ///     The metric that measures point spacing.
    ///
    /// Returns
    /// -------
    /// float
    ///     The maximum distance between consecutive points.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``metric`` is ``SphereBundle`` and this trajectory has an odd
    ///     coordinate count.
    /// ``TypeError``
    ///     If ``metric`` is not a recognized type.
    fn resolution(&self, metric: &Bound<'_, PyAny>) -> PyResult<f64> {
        let metric = metric_from_py(metric)?;
        check_even_dimension(metric, self.inner.dimension())?;
        Ok(self.inner.resolution(metric))
    }

    /// Returns the trajectory points as a two-dimensional array.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A two-dimensional array whose rows are the trajectory points.
    #[must_use]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.points().to_pyarray(py)
    }

    /// Returns the trajectory's parameterization.
    ///
    /// One value per point, strictly increasing. Defaults to the index
    /// parameterization ``0.0, 1.0, ...`` unless supplied explicitly at
    /// construction, recorded by ``resample`` (the interpolation parameter
    /// each point was sampled at), or carried through by ``downsample`` (the
    /// entries of the points it kept).
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A one-dimensional array of parameter values, one per point.
    #[must_use]
    fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.parameters().to_pyarray(py)
    }

    /// Returns the number of points in the trajectory.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns a content fingerprint of the trajectory.
    ///
    /// Two trajectories with identical point and parameter data have the
    /// same fingerprint. Typically used to verify correct serialization and
    /// deserialization.
    ///
    /// Returns
    /// -------
    /// int
    ///     A fingerprint identifying the trajectory's content.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Saves the trajectory to a file at ``path``.
    ///
    /// Parameters
    /// ----------
    /// path : str or ``os.PathLike``
    ///     The destination file path.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If the file cannot be written.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    /// Loads a trajectory from the file at ``path``.
    ///
    /// Parameters
    /// ----------
    /// path : str or ``os.PathLike``
    ///     The source file path.
    ///
    /// Returns
    /// -------
    /// ``Trajectory``
    ///     The reloaded trajectory.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If the file cannot be read.
    /// ``FormatVersionMismatchError``
    ///     If the file was written by an incompatible version of the library.
    /// ``ValueError``
    ///     If the stored data cannot be decoded.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = Trajectory::load(path).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl PyTrajectory {
    /// Resamples `interpolator` with the interpreter detached, so other
    /// Python threads run during the sampling.
    fn resampled<I: Interpolator + Send>(
        py: Python<'_>,
        interpolator: I,
        metric: Metric,
        spacing: f64,
    ) -> PyResult<Self> {
        let inner = py
            .detach(move || Trajectory::resample(&interpolator, metric, spacing))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// Registers the ``Trajectory`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTrajectory>()?;
    Ok(())
}
