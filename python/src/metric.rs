// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the built-in metric types and the metric extractor
//! used by every method that accepts a user-supplied metric.

use cycling_signatures::Metric;
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
    metric: Metric,
    point: &PyReadonlyArray1<'_, f64>,
    other: &PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    if point.as_array().len() != other.as_array().len() {
        return Err(PyValueError::new_err(format!(
            "coordinate vectors have mismatched lengths: first {}, second {}",
            point.as_array().len(),
            other.as_array().len()
        )));
    }
    metric
        .check_dimension(point.as_array().len())
        .map_err(to_pyerr)?;
    Ok(metric.distance(point.as_array(), other.as_array()))
}

/// Computes an N x N symmetric matrix of pairwise distances among the rows of
/// ``points``.
fn pairwise_matrix<'py>(
    metric: Metric,
    points: &PyReadonlyArray2<'_, f64>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = points.as_array();
    metric.check_dimension(array.ncols()).map_err(to_pyerr)?;
    let count = array.nrows();

    // The coordinates are copied because the fill runs with the interpreter
    // detached, and a view into the caller's array would be open to mutation
    // from another Python thread while it does.
    let coordinates = array.to_owned();
    let matrix = py.detach(move || {
        let array = coordinates.view();
        let mut matrix = Array2::zeros((count, count));
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(count);
        let mut distances = vec![0.0_f64; count];
        for left in 0..count {
            pairs.clear();
            pairs.extend((left + 1..count).map(|right| (left, right)));
            metric.fill_distances(array, &pairs, &mut distances[..pairs.len()]);
            for (index, &(_, right)) in pairs.iter().enumerate() {
                matrix[[left, right]] = distances[index];
                matrix[[right, left]] = distances[index];
            }
        }
        matrix
    });
    Ok(matrix.to_pyarray(py))
}

/// The standard Euclidean metric.
///
/// Distance is the square root of the sum of squared coordinate differences.
#[pyclass(name = "Euclidean", frozen)]
pub(crate) struct PyEuclidean;

#[pymethods]
#[expect(clippy::unused_self, reason = "Python methods must take self")]
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
    fn distance(
        &self,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        scalar_distance(Metric::Euclidean, &point, &other)
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
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        pairwise_matrix(Metric::Euclidean, &points, py)
    }

    /// Returns a string representation of the metric.
    fn __repr__(&self) -> String {
        "Euclidean()".to_string()
    }
}

/// A distance metric on the L2 sphere bundle.
///
/// Operates on even-length coordinate vectors whose first half is a spatial
/// position and whose second half is a direction (velocity) vector. Neither
/// half is normalized: the distance is the maximum of the two halves'
/// Euclidean distances, measured directly on the given coordinates. The
/// maximum of the two halves gives a distance of at most ``t`` its reading:
/// within ``t`` in position *and* within ``t`` in direction.
///
/// This metric is calibrated against ``SphereBundleInterpolator``, which
/// stores each direction half as the unit tangent scaled to its configured
/// direction radius. That radius does double duty: it is the resolution of
/// the direction half and the exchange rate between position and direction.
#[pyclass(name = "SphereBundle", frozen)]
pub(crate) struct PySphereBundle;

#[pymethods]
#[expect(clippy::unused_self, reason = "Python methods must take self")]
impl PySphereBundle {
    /// Creates a sphere-bundle metric.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Returns the distance between two coordinate vectors.
    ///
    /// Parameters
    /// ----------
    /// point : ndarray
    ///     A one-dimensional, even-length coordinate vector whose first half is
    ///     a position and whose second half is a direction.
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
    ///     If the two vectors differ in length, or if their common length is
    ///     odd.
    fn distance(
        &self,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        scalar_distance(Metric::SphereBundle, &point, &other)
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
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``points`` has an odd number of columns.
    fn distance_matrix<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        pairwise_matrix(Metric::SphereBundle, &points, py)
    }

    /// Returns a string representation of the metric.
    fn __repr__(&self) -> String {
        "SphereBundle()".to_string()
    }
}

/// Builds a core metric from any built-in Python metric object, or raises
/// ``TypeError`` if the object is not a recognized metric.
pub(crate) fn metric_from_py(object: &Bound<'_, PyAny>) -> PyResult<Metric> {
    if object.cast::<PyEuclidean>().is_ok() {
        return Ok(Metric::Euclidean);
    }
    if object.cast::<PySphereBundle>().is_ok() {
        return Ok(Metric::SphereBundle);
    }
    Err(PyTypeError::new_err(format!(
        "expected a Euclidean or SphereBundle metric, got {}",
        object.get_type().name()?
    )))
}

/// Builds the Python metric object corresponding to a core metric, the inverse
/// of [`metric_from_py`].
///
/// Lets a caller holding a value that carries a metric (an embedded trajectory,
/// for instance) ask which metric it measures under, for introspection or for
/// passing on to another call that expects a metric object.
pub(crate) fn metric_to_py(py: Python<'_>, metric: Metric) -> PyResult<Py<PyAny>> {
    match metric {
        Metric::Euclidean => Ok(Py::new(py, PyEuclidean)?.into_any()),
        Metric::SphereBundle => Ok(Py::new(py, PySphereBundle)?.into_any()),
    }
}

/// Registers the metric classes on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEuclidean>()?;
    module.add_class::<PySphereBundle>()?;
    Ok(())
}
