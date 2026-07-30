// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the core `Trajectory` type.

use std::{path::PathBuf, sync::Arc};

use cycling_signatures::Trajectory;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{
    errors::to_pyerr,
    interpolation::{PyChebyshevSphereBundleInterpolator, PyCubicSpline},
    metric::metric_from_py,
};

/// A sampled trajectory, optionally with fill points inserted between samples.
///
/// Wraps a sequence of points in d-dimensional space, enforcing that all
/// coordinates are finite. When constructed via ``resample``, additional
/// interpolated points are inserted between consecutive samples so that no two
/// adjacent points are more than ``bound`` apart under the given metric.
///
/// Parameters
/// ----------
/// points : ndarray
///     A two-dimensional array with at least one row, each row one sample.
///
/// Raises
/// ------
/// ``ValueError``
///     If ``points`` has zero rows or if any coordinate is not finite.
#[pyclass(name = "Trajectory")]
pub(crate) struct PyTrajectory {
    pub(crate) inner: Arc<Trajectory>,
}

#[pymethods]
impl PyTrajectory {
    /// Constructs a trajectory from a two-dimensional array of sample points.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(points: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let inner = Trajectory::new(points.as_array()).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Constructs a trajectory by sampling an interpolator and inserting fill
    /// points between the original samples.
    ///
    /// Starting from the interpolator's original sample points, bisects each
    /// interval until consecutive points are no more than ``bound`` apart under
    /// ``metric``. The ``original_count`` of the returned trajectory equals the
    /// number of knots in the interpolator.
    ///
    /// Parameters
    /// ----------
    /// interpolator : ``CubicSpline`` or ``ChebyshevSphereBundleInterpolator``
    ///     The interpolator to sample.
    /// metric : ``Euclidean`` or ``SphereBundle``
    ///     The metric that measures consecutive-point spacing.
    /// bound : float
    ///     The largest allowed distance between consecutive points.
    ///
    /// Returns
    /// -------
    /// ``Trajectory``
    ///     The resampled trajectory with fill points inserted.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the interpolator has fewer than two knots, if a sampled value is
    ///     not finite, or if bisection cannot reach ``bound``.
    /// ``TypeError``
    ///     If ``interpolator`` or ``metric`` is not a recognized type.
    #[staticmethod]
    fn resample(
        interpolator: &Bound<'_, PyAny>,
        metric: &Bound<'_, PyAny>,
        bound: f64,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        if let Ok(spline) = interpolator.cast::<PyCubicSpline>() {
            let inner =
                Trajectory::resample(&spline.borrow().inner, metric, bound).map_err(to_pyerr)?;
            return Ok(Self {
                inner: Arc::new(inner),
            });
        }
        if let Ok(bundle) = interpolator.cast::<PyChebyshevSphereBundleInterpolator>() {
            let inner =
                Trajectory::resample(&bundle.borrow().inner, metric, bound).map_err(to_pyerr)?;
            return Ok(Self {
                inner: Arc::new(inner),
            });
        }
        Err(PyTypeError::new_err(format!(
            "expected a CubicSpline or ChebyshevSphereBundleInterpolator, got {}",
            interpolator.get_type().name()?
        )))
    }

    /// Returns the trajectory points as a two-dimensional array.
    ///
    /// For a resampled trajectory, the array includes the interpolated fill
    /// rows in addition to the original sample rows. Returns a fresh copy.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A two-dimensional array whose rows are the trajectory points.
    #[must_use]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.points().to_pyarray(py)
    }

    /// Returns the number of original sample points.
    ///
    /// For a trajectory built with ``Trajectory(points)``, this equals the row
    /// count of ``points``. For a resampled trajectory, this equals the number
    /// of knots in the interpolator.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of original samples.
    #[must_use]
    fn original_count(&self) -> usize {
        self.inner.original_count()
    }

    /// Returns a content fingerprint of the trajectory.
    ///
    /// Two trajectories with identical point data have the same fingerprint.
    /// Typically used to verify correct serialization and deserialization.
    ///
    /// Returns
    /// -------
    /// int
    ///     A fingerprint identifying the point data.
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
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = Trajectory::load(path).map_err(to_pyerr)?;
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
