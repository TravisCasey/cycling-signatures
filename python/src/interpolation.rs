// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the interpolation types used when constructing
//! sphere-bundle trajectories.

use std::sync::Arc;

use cycling_signatures::{CubicSpline, SphereBundleInterpolator};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::errors::to_pyerr;

/// A natural cubic spline interpolator fit over a set of knots.
///
/// The spline passes exactly through the supplied data points. Within each
/// interval it is a cubic polynomial, and across knot boundaries the first
/// and second derivatives are continuous. At the two endpoints the second
/// derivative is zero (the natural boundary condition).
///
/// Parameters
/// ----------
/// knots : ndarray
///     A one-dimensional array of strictly increasing parameter values with at
///     least two elements.
/// values : ndarray
///     A two-dimensional array whose row count matches the length of ``knots``
///     and whose column count is the output dimension.
///
/// Raises
/// ------
/// ``ValueError``
///     If ``knots`` has fewer than two elements, if the number of rows in
///     ``values`` does not match the length of ``knots``, or if ``knots`` is
///     not strictly increasing.
///
/// Examples
/// --------
/// Fit a spline through four planar knots::
///
///     import numpy as np
///     import cycling_signatures as cs
///
///     knots = np.array([0.0, 1.0, 2.0, 3.0])
///     values = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
///     spline = cs.CubicSpline(knots, values)
#[pyclass(name = "CubicSpline")]
pub(crate) struct PyCubicSpline {
    pub(crate) inner: Arc<CubicSpline>,
}

#[pymethods]
impl PyCubicSpline {
    /// Fits a natural cubic spline through the given knots and values.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(knots: PyReadonlyArray1<'_, f64>, values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let inner =
            CubicSpline::new(knots.as_array().to_owned(), values.as_array()).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// A sphere-bundle interpolator that wraps a ``CubicSpline``.
///
/// At each parameter value the output concatenates the position from the
/// inner spline with a direction vector: the inner derivative, normalized to
/// unit L2 length, then scaled to ``direction_radius``. Each sample has
/// length twice that of the inner spline.
///
/// ``direction_radius`` is the L2 norm of every stored direction, and it sets
/// the angular resolution of the embedding. Two directions separated by an
/// angle ``theta`` are stored ``2 * direction_radius * sin(theta / 2)``
/// apart, so a cycle-detection threshold ``t`` admits directions within
/// ``2 * arcsin(t / (2 * direction_radius))`` of each other: a larger radius
/// distinguishes directions more finely at the same threshold. Pair this
/// interpolator with the ``SphereBundle`` metric to measure distances on the
/// resulting embedding.
///
/// Parameters
/// ----------
/// inner : ``CubicSpline``
///     The spline supplying positions and derivatives.
/// direction_radius : float
///     The direction normalization radius. Must be positive and finite.
///
/// Raises
/// ------
/// ``ValueError``
///     If ``direction_radius`` is not positive and finite.
///
/// Examples
/// --------
/// Wrap a spline and read back the normalization radius::
///
///     import numpy as np
///     import cycling_signatures as cs
///
///     spline = cs.CubicSpline(
///         np.array([0.0, 1.0, 2.0]),
///         np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]]),
///     )
///     bundle = cs.SphereBundleInterpolator(spline, 1.5)
///     assert bundle.direction_radius() == 1.5
#[pyclass(name = "SphereBundleInterpolator")]
pub(crate) struct PySphereBundleInterpolator {
    pub(crate) inner: SphereBundleInterpolator<Arc<CubicSpline>>,
}

#[pymethods]
impl PySphereBundleInterpolator {
    /// Wraps a ``CubicSpline`` with the given direction radius.
    #[new]
    fn new(inner: &Bound<'_, PyCubicSpline>, direction_radius: f64) -> PyResult<Self> {
        if !(direction_radius.is_finite() && direction_radius > 0.0) {
            return Err(PyValueError::new_err(format!(
                "direction radius must be positive and finite, got {direction_radius}"
            )));
        }
        let spline = Arc::clone(&inner.borrow().inner);
        let bundle = SphereBundleInterpolator::new(spline, direction_radius);
        Ok(Self { inner: bundle })
    }

    /// Returns the direction normalization radius given at construction.
    ///
    /// Returns
    /// -------
    /// float
    ///     The direction normalization radius.
    #[must_use]
    fn direction_radius(&self) -> f64 {
        self.inner.direction_radius()
    }
}

/// Registers the interpolation classes on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCubicSpline>()?;
    module.add_class::<PySphereBundleInterpolator>()?;
    Ok(())
}
