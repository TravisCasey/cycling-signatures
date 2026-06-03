// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the built-in metric types and the metric extractor
//! used by every method that accepts a user-supplied metric.

use cycling_signatures::{Chebyshev, Euclidean, Metric, SphereBundleMetric};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::errors::to_pyerr;

/// The standard Euclidean metric.
///
/// Distance is the square root of the sum of squared coordinate differences.
#[pyclass(name = "Euclidean", frozen)]
pub(crate) struct PyEuclidean;

#[pymethods]
impl PyEuclidean {
    /// Creates a new Euclidean metric.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// The Chebyshev (L-infinity) metric.
///
/// Distance is the largest absolute coordinate difference across all axes.
#[pyclass(name = "Chebyshev", frozen)]
pub(crate) struct PyChebyshev;

#[pymethods]
impl PyChebyshev {
    /// Creates a new Chebyshev metric.
    #[new]
    fn new() -> Self {
        Self
    }
}

/// A distance metric on the L2 sphere bundle.
///
/// Operates on even-length coordinate vectors whose first half is a spatial
/// position and whose second half is a nonzero direction (velocity) vector.
/// The direction half is L2-normalized before computing distances. The combined
/// metric is
///
/// ```text
/// max(
///     euclidean(position_left, position_right),
///     direction_weight * euclidean(direction_left_unit, direction_right_unit),
/// )
/// ```
#[pyclass(name = "SphereBundle", frozen)]
pub(crate) struct PySphereBundle {
    metric: SphereBundleMetric,
}

#[pymethods]
impl PySphereBundle {
    /// Creates a sphere-bundle metric with the given direction weight.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if `direction_weight` is not finite or not strictly
    /// positive.
    #[new]
    fn new(direction_weight: f64) -> PyResult<Self> {
        let metric = SphereBundleMetric::new(direction_weight).map_err(to_pyerr)?;
        Ok(Self { metric })
    }

    /// Returns the configured direction weight.
    #[must_use]
    fn direction_weight(&self) -> f64 {
        self.metric.direction_weight()
    }
}

/// Builds a boxed core metric from any built-in Python metric object, or raises
/// `TypeError` if the object is not a recognized metric.
pub(crate) fn metric_from_py(object: &Bound<'_, PyAny>) -> PyResult<Box<dyn Metric>> {
    if object.cast::<PyEuclidean>().is_ok() {
        return Ok(Box::new(Euclidean));
    }
    if object.cast::<PyChebyshev>().is_ok() {
        return Ok(Box::new(Chebyshev));
    }
    if let Ok(sphere) = object.cast::<PySphereBundle>() {
        return Ok(Box::new(sphere.get().metric));
    }
    Err(PyTypeError::new_err(
        "expected a Euclidean, Chebyshev, or SphereBundle metric",
    ))
}

/// Registers the metric classes on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEuclidean>()?;
    module.add_class::<PyChebyshev>()?;
    module.add_class::<PySphereBundle>()?;
    Ok(())
}
