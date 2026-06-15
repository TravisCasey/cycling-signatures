// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the built-in metric types and the metric extractor
//! used by every method that accepts a user-supplied metric.

use cycling_signatures::{Chebyshev, Euclidean, Metric, SphereBundleMetric};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray, ndarray::Array2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::errors::to_pyerr;

/// Computes the scalar distance between two coordinate vectors.
///
/// Returns an error if the vectors differ in length.
fn scalar_distance(
    metric: &dyn Metric,
    point: &PyReadonlyArray1<'_, f64>,
    other: &PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    if point.as_array().len() != other.as_array().len() {
        return Err(PyValueError::new_err(
            "coordinate vectors must have equal length",
        ));
    }
    Ok(metric.distance(point.as_array(), other.as_array()))
}

/// Computes an N x N symmetric matrix of pairwise distances among the rows of
/// ``points``.
fn pairwise_matrix<'py>(
    metric: &dyn Metric,
    points: &PyReadonlyArray2<'_, f64>,
    py: Python<'py>,
) -> Bound<'py, PyArray2<f64>> {
    let array = points.as_array();
    let count = array.nrows();
    let pairs: Vec<(usize, usize)> = (0..count)
        .flat_map(|left| (left + 1..count).map(move |right| (left, right)))
        .collect();

    let mut distances = vec![0.0_f64; pairs.len()];
    metric.fill_distances(array, &pairs, &mut distances);
    let mut matrix = Array2::zeros((count, count));
    for (index, &(left, right)) in pairs.iter().enumerate() {
        matrix[[left, right]] = distances[index];
        matrix[[right, left]] = distances[index];
    }
    matrix.to_pyarray(py)
}

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

    /// Returns the distance between two coordinate vectors.
    ///
    /// Parameters
    /// ----------
    /// point : ndarray
    ///     A one-dimensional coordinate vector.
    /// other : ndarray
    ///     A one-dimensional coordinate vector of the same length as ``point``.
    ///
    /// Returns
    /// -------
    /// float
    ///     The Euclidean distance between the two vectors.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the two vectors differ in length.
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn distance(
        &self,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        scalar_distance(&Euclidean, &point, &other)
    }

    /// Returns the matrix of pairwise distances among the rows of ``points``.
    ///
    /// Parameters
    /// ----------
    /// points : ndarray
    ///     A two-dimensional array whose rows are coordinate vectors.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A square symmetric matrix whose entry ``(i, j)`` is the distance
    ///     between row ``i`` and row ``j``. The diagonal is zero.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        pairwise_matrix(&Euclidean, &points, py)
    }

    /// Returns a string representation of the metric.
    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "Euclidean()".to_string()
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

    /// Returns the distance between two coordinate vectors.
    ///
    /// Parameters
    /// ----------
    /// point : ndarray
    ///     A one-dimensional coordinate vector.
    /// other : ndarray
    ///     A one-dimensional coordinate vector of the same length as ``point``.
    ///
    /// Returns
    /// -------
    /// float
    ///     The largest absolute coordinate difference between the two vectors.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the two vectors differ in length.
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn distance(
        &self,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        scalar_distance(&Chebyshev, &point, &other)
    }

    /// Returns the matrix of pairwise distances among the rows of ``points``.
    ///
    /// Parameters
    /// ----------
    /// points : ndarray
    ///     A two-dimensional array whose rows are coordinate vectors.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A square symmetric matrix whose entry ``(i, j)`` is the distance
    ///     between row ``i`` and row ``j``. The diagonal is zero.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        pairwise_matrix(&Chebyshev, &points, py)
    }

    /// Returns a string representation of the metric.
    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "Chebyshev()".to_string()
    }
}

/// A distance metric on the L2 sphere bundle.
///
/// Operates on even-length coordinate vectors whose first half is a spatial
/// position and whose second half is a nonzero direction (velocity) vector.
/// The distance is the maximum of two quantities: the Euclidean distance
/// between the position halves, and ``direction_weight`` times the Euclidean
/// distance between the L2-normalized direction halves.
///
/// Parameters
/// ----------
/// direction_weight : float
///     A finite, strictly positive weight applied to the direction half of the
///     distance.
///
/// Raises
/// ------
/// ``ValueError``
///     If ``direction_weight`` is not finite or not strictly positive.
#[pyclass(name = "SphereBundle", frozen)]
pub(crate) struct PySphereBundle {
    metric: SphereBundleMetric,
}

#[pymethods]
impl PySphereBundle {
    /// Creates a sphere-bundle metric with the given direction weight.
    #[new]
    fn new(direction_weight: f64) -> PyResult<Self> {
        let metric = SphereBundleMetric::new(direction_weight).map_err(to_pyerr)?;
        Ok(Self { metric })
    }

    /// Returns the configured direction weight.
    ///
    /// Returns
    /// -------
    /// float
    ///     The direction weight set at construction.
    #[must_use]
    fn direction_weight(&self) -> f64 {
        self.metric.direction_weight()
    }

    /// Returns the distance between two coordinate vectors.
    ///
    /// The direction half of each vector is L2-normalized before the distance
    /// is taken, so the result is independent of the direction magnitudes.
    ///
    /// Parameters
    /// ----------
    /// point : ndarray
    ///     A one-dimensional, even-length coordinate vector whose first half is
    ///     a position and whose second half is a nonzero direction.
    /// other : ndarray
    ///     A coordinate vector of the same length as ``point``.
    ///
    /// Returns
    /// -------
    /// float
    ///     The sphere-bundle distance between the two vectors.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If the two vectors differ in length.
    #[allow(clippy::needless_pass_by_value)]
    fn distance(
        &self,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        scalar_distance(&self.metric, &point, &other)
    }

    /// Returns the matrix of pairwise distances among the rows of ``points``.
    ///
    /// Parameters
    /// ----------
    /// points : ndarray
    ///     A two-dimensional array whose rows are even-length sphere-bundle
    ///     coordinate vectors.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A square symmetric matrix whose entry ``(i, j)`` is the distance
    ///     between row ``i`` and row ``j``. The diagonal is zero.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        pairwise_matrix(&self.metric, &points, py)
    }

    /// Returns a string representation of the metric.
    fn __repr__(&self) -> String {
        format!(
            "SphereBundle(direction_weight={:?})",
            self.metric.direction_weight()
        )
    }
}

/// Builds a boxed core metric from any built-in Python metric object, or raises
/// ``TypeError`` if the object is not a recognized metric.
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
