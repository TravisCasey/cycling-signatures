// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the interpolation types used when constructing
//! sphere-bundle trajectories.

use cycling_signatures::{ChebyshevSphereBundleInterpolator, CubicSpline};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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
    pub(crate) inner: CubicSpline,
}

#[pymethods]
impl PyCubicSpline {
    /// Fits a natural cubic spline through the given knots and values.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(knots: PyReadonlyArray1<'_, f64>, values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let inner =
            CubicSpline::new(knots.as_array().to_owned(), values.as_array()).map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// A sphere-bundle interpolator that wraps a ``CubicSpline``.
///
/// At each parameter value the output concatenates the position from the inner
/// spline with a direction vector: the inner derivative, *normalized by its
/// Chebyshev norm*, then scaled to the configured radius. Each sample has
/// length twice that of the inner spline.
///
/// The radius is ``radius_floor + 0.5``, where ``radius_floor`` is given at
/// construction. The half-integer offset keeps every direction coordinate
/// strictly between two integers, ensuring that cube-floor assignments for the
/// direction components are unambiguous at extremal values.
///
/// Parameters
/// ----------
/// inner : ``CubicSpline``
///     The spline supplying positions and derivatives.
/// radius_floor : int
///     Sets the normalization radius to ``radius_floor + 0.5``.
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
///     bundle = cs.ChebyshevSphereBundleInterpolator(spline, 1)
///     assert bundle.radius() == 1.5
#[pyclass(name = "ChebyshevSphereBundleInterpolator")]
pub(crate) struct PyChebyshevSphereBundleInterpolator {
    pub(crate) inner: ChebyshevSphereBundleInterpolator<CubicSpline>,
}

#[pymethods]
impl PyChebyshevSphereBundleInterpolator {
    /// Wraps a ``CubicSpline`` with the given radius floor.
    #[new]
    fn new(inner: &Bound<'_, PyCubicSpline>, radius_floor: u32) -> Self {
        let spline = inner.borrow().inner.clone();
        let bundle = ChebyshevSphereBundleInterpolator::new(spline, radius_floor);
        Self { inner: bundle }
    }

    /// Returns the normalization radius, ``radius_floor + 0.5`` from
    /// construction.
    ///
    /// Returns
    /// -------
    /// float
    ///     The direction normalization radius.
    #[must_use]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }
}

/// Registers the interpolation classes on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCubicSpline>()?;
    module.add_class::<PyChebyshevSphereBundleInterpolator>()?;
    Ok(())
}
