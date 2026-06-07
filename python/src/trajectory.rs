// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the core `Trajectory` type.

use std::path::PathBuf;

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
/// coordinates are finite. When constructed via `resample`, additional
/// interpolated points are inserted between consecutive samples so that no two
/// adjacent points are more than `bound` apart under the given metric.
#[pyclass(name = "Trajectory")]
pub(crate) struct PyTrajectory {
    pub(crate) inner: Trajectory,
}

#[pymethods]
impl PyTrajectory {
    /// Constructs a trajectory from a 2D `NumPy` array of sample points.
    ///
    /// `points` must have at least one row; each row is one sample.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if `points` has zero rows or if any coordinate is
    /// not finite.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(points: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let inner = Trajectory::new(points.as_array()).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Constructs a trajectory by sampling an interpolator and inserting fill
    /// points between the original samples.
    ///
    /// Starting from the interpolator's original sample points, bisects each
    /// interval until consecutive points are no more than `bound` apart under
    /// `metric`. The `original_count` method of the returned trajectory equals
    /// the number of knots in the interpolator.
    ///
    /// `interpolator` must be a `CubicSpline` or
    /// `ChebyshevSphereBundleInterpolator`. `metric` must be `Euclidean`,
    /// `Chebyshev`, or `SphereBundle` metric.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if the interpolator has fewer than two knots, if a
    /// sampled value is not finite, or if bisection cannot reach `bound`.
    /// Raises `TypeError` if `interpolator` or `metric` is not a recognized
    /// type.
    #[staticmethod]
    fn resample(
        interpolator: &Bound<'_, PyAny>,
        metric: &Bound<'_, PyAny>,
        bound: f64,
    ) -> PyResult<Self> {
        let metric = metric_from_py(metric)?;
        if let Ok(spline) = interpolator.cast::<PyCubicSpline>() {
            let inner = Trajectory::resample(&spline.borrow().inner, metric.as_ref(), bound)
                .map_err(to_pyerr)?;
            return Ok(Self { inner });
        }
        if let Ok(bundle) = interpolator.cast::<PyChebyshevSphereBundleInterpolator>() {
            let inner = Trajectory::resample(&bundle.borrow().inner, metric.as_ref(), bound)
                .map_err(to_pyerr)?;
            return Ok(Self { inner });
        }
        Err(PyTypeError::new_err(
            "expected a CubicSpline or ChebyshevSphereBundleInterpolator",
        ))
    }

    /// Returns the trajectory points as a 2D `NumPy` array.
    ///
    /// For a resampled trajectory, the array includes the interpolated fill
    /// rows in addition to the original sample rows. Returns a fresh copy.
    #[must_use]
    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.points().to_pyarray(py)
    }

    /// Returns the number of original sample points.
    ///
    /// For a trajectory built with `Trajectory(points)`, this equals
    /// `points().shape[0]`. For a resampled trajectory, this equals the number
    /// of knots in the interpolator.
    #[must_use]
    fn original_count(&self) -> usize {
        self.inner.original_count()
    }

    /// Returns a content fingerprint of the trajectory.
    ///
    /// Two trajectories with identical point data have the same fingerprint.
    /// Typically used to verify correct serialization and deserialization.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Saves the trajectory to a file at `path`.
    ///
    /// # Errors
    ///
    /// Raises `OSError` if the file cannot be written.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    /// Loads a trajectory from the file at `path`.
    ///
    /// # Errors
    ///
    /// Raises `OSError` if the file cannot be read. Raises
    /// `FormatVersionMismatchError` if the file was written by an incompatible
    /// version of the library.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = Trajectory::load(path).map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Registers the `Trajectory` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTrajectory>()?;
    Ok(())
}
