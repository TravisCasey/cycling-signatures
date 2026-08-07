// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Natural cubic spline interpolation.

use ndarray::{Array1, Array2, Array3, ArrayView2};

use crate::{
    error::{Error, Result},
    interpolation::{DerivativeInterpolator, Interpolator},
};

/// A natural cubic spline fit over a set of knots.
///
/// The spline passes exactly through the supplied data points. Within each
/// interval it is a cubic polynomial, and across knot boundaries the spline
/// has continuous first and second derivatives. At the two endpoints the
/// second derivative is zero (the natural boundary condition).
///
/// # Examples
///
/// ```
/// use cycling_signatures::interpolation::{CubicSpline, Interpolator};
/// use ndarray::array;
///
/// let knots = array![0.0, 1.0, 2.0, 3.0];
/// let values = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
/// let spline = CubicSpline::new(knots, values.view()).unwrap();
/// let sample = spline.sample(1.5);
/// assert!((sample[0] - 1.5).abs() < 1e-10);
/// assert!((sample[1] - 0.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct CubicSpline {
    knots: Array1<f64>,
    coefficients: Array3<f64>,
}

impl CubicSpline {
    /// Constructs a natural cubic spline through the given knots and values.
    ///
    /// `knots` must be strictly increasing and have at least two elements.
    /// `values` must have the same number of rows as `knots`.
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::InterpolationKnotCount`] if fewer than two knots are
    ///   supplied.
    /// - [`Error::InterpolationShapeMismatch`] if the number of rows in
    ///   `values` does not match the number of knots.
    /// - [`Error::InterpolationKnotsNotIncreasing`] if `knots` is not strictly
    ///   increasing.
    #[expect(
        clippy::missing_panics_doc,
        reason = "internal panic call is guarded, so the method advertises no panic"
    )]
    pub fn new(knots: Array1<f64>, values: ArrayView2<'_, f64>) -> Result<Self> {
        let num_knots = knots.len();
        if num_knots < 2 {
            return Err(Error::InterpolationKnotCount { knots: num_knots });
        }

        let num_value_rows = values.nrows();
        if num_value_rows != num_knots {
            return Err(Error::InterpolationShapeMismatch {
                knots: num_knots,
                value_rows: num_value_rows,
            });
        }

        let knots_slice = knots.as_slice().expect("knots stored contiguously");
        for index in 0..num_knots - 1 {
            if knots_slice[index + 1] <= knots_slice[index] {
                return Err(Error::InterpolationKnotsNotIncreasing { index });
            }
        }

        let coefficients = compute_coefficients(knots_slice, values);
        Ok(Self {
            knots,
            coefficients,
        })
    }

    /// Fits a natural cubic spline through `values` with knots
    /// `0, 1, ..., values.nrows() - 1`.
    ///
    /// # Errors
    ///
    /// - [`Error::InterpolationKnotCount`] if `values` has fewer than two rows.
    pub fn with_integer_knots(values: ArrayView2<'_, f64>) -> Result<Self> {
        let knots = Array1::from_iter((0..values.nrows()).map(|index| index as f64));
        Self::new(knots, values)
    }

    /// Evaluates the spline value or one of its derivatives at `parameter`.
    ///
    /// `order` 0 returns the sampled value, `1` the first derivative, `2` the
    /// second, `3` the third. Orders four and above return the zero vector
    /// (a cubic polynomial has no higher derivatives).
    ///
    /// # Panics
    ///
    /// Panics if `parameter` is outside `[knots[0], knots[last]]`.
    #[must_use]
    pub fn sample_with_order(&self, parameter: f64, order: usize) -> Array1<f64> {
        let knots = self.knots.as_slice().expect("knots stored contiguously");
        let first_knot = knots[0];
        let last_knot = *knots.last().expect("at least two knots");
        assert!(
            parameter >= first_knot && parameter <= last_knot,
            "parameter {parameter} outside of domain [{first_knot}, {last_knot}]",
        );

        // Binary search for the interval index.
        let num_intervals = knots.len() - 1;
        let mut low = 0_usize;
        let mut high = num_intervals - 1;
        while low < high {
            let middle = (low + high).div_ceil(2);
            if knots[middle] <= parameter {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let interval = low;

        let offset = parameter - knots[interval];
        let dimension = self.coefficients.dim().2;
        let mut result = Array1::<f64>::zeros(dimension);

        match order {
            0 => {
                for axis in 0..dimension {
                    result[axis] = self.coefficients[[interval, 0, axis]]
                        + offset * self.coefficients[[interval, 1, axis]]
                        + offset.powi(2) * self.coefficients[[interval, 2, axis]]
                        + offset.powi(3) * self.coefficients[[interval, 3, axis]];
                }
            },
            1 => {
                for axis in 0..dimension {
                    result[axis] = self.coefficients[[interval, 1, axis]]
                        + 2.0 * offset * self.coefficients[[interval, 2, axis]]
                        + 3.0 * offset.powi(2) * self.coefficients[[interval, 3, axis]];
                }
            },
            2 => {
                for axis in 0..dimension {
                    result[axis] = 2.0 * self.coefficients[[interval, 2, axis]]
                        + 6.0 * offset * self.coefficients[[interval, 3, axis]];
                }
            },
            3 => {
                for axis in 0..dimension {
                    result[axis] = 6.0 * self.coefficients[[interval, 3, axis]];
                }
            },
            _ => {
                // Cubic polynomial; orders four and above are identically zero.
            },
        }

        result
    }
}

/// Computes the cubic spline coefficients for all intervals and all output
/// dimensions.
///
/// Uses the natural boundary condition (zero second derivative at the
/// endpoints) and a tridiagonal algorithm to solve for the second derivatives
/// at each knot in time linear over the number of rows. The moment
/// (second-derivative) formulation and Thomas-algorithm solve follow Stoer &
/// Bulirsch, *Introduction to Numerical Analysis*, section 2.4.
fn compute_coefficients(knots: &[f64], values: ArrayView2<'_, f64>) -> Array3<f64> {
    let num_knots = knots.len();
    let num_intervals = num_knots - 1;
    let dimension = values.ncols();

    // Interval widths.
    let widths: Vec<f64> = (0..num_intervals)
        .map(|index| knots[index + 1] - knots[index])
        .collect();

    // Solve for second derivatives via the natural cubic spline tridiagonal
    // system, with boundary conditions second[0] = second[n-1] = 0.
    let mut second_derivatives = Array2::<f64>::zeros((num_knots, dimension));

    // Forward sweep.
    let mut forward_coeff = vec![0.0_f64; num_knots];
    let mut forward_rhs = Array2::<f64>::zeros((num_knots, dimension));

    for index in 1..num_intervals {
        let diagonal = 2.0 * (widths[index - 1] + widths[index]);
        let factor = 1.0 / (diagonal - widths[index - 1] * forward_coeff[index - 1]);
        forward_coeff[index] = widths[index] * factor;

        for axis in 0..dimension {
            let rhs_value = 6.0
                * ((values[[index + 1, axis]] - values[[index, axis]]) / widths[index]
                    - (values[[index, axis]] - values[[index - 1, axis]]) / widths[index - 1]);
            forward_rhs[[index, axis]] =
                (rhs_value - widths[index - 1] * forward_rhs[[index - 1, axis]]) * factor;
        }
    }

    // Back substitution.
    for index in (1..num_intervals).rev() {
        for axis in 0..dimension {
            second_derivatives[[index, axis]] = forward_rhs[[index, axis]]
                - forward_coeff[index] * second_derivatives[[index + 1, axis]];
        }
    }

    // Build the coefficient array: shape [num_intervals, 4, dimension].
    let mut coefficients = Array3::<f64>::zeros((num_intervals, 4, dimension));

    for interval in 0..num_intervals {
        let width = widths[interval];
        for axis in 0..dimension {
            let value_left = values[[interval, axis]];
            let value_right = values[[interval + 1, axis]];
            let second_left = second_derivatives[[interval, axis]];
            let second_right = second_derivatives[[interval + 1, axis]];

            // Coefficients for: a + b*u + c*u^2 + d*u^3
            coefficients[[interval, 0, axis]] = value_left;
            coefficients[[interval, 1, axis]] = (value_right - value_left) / width
                - width * (2.0 * second_left + second_right) / 6.0;
            coefficients[[interval, 2, axis]] = second_left / 2.0;
            coefficients[[interval, 3, axis]] = (second_right - second_left) / (6.0 * width);
        }
    }

    coefficients
}

impl Interpolator for CubicSpline {
    fn sample(&self, parameter: f64) -> Array1<f64> {
        self.sample_with_order(parameter, 0)
    }

    fn knots(&self) -> &[f64] {
        self.knots.as_slice().expect("knots stored contiguously")
    }
}

impl DerivativeInterpolator for CubicSpline {
    fn derivative(&self, parameter: f64) -> Array1<f64> {
        self.sample_with_order(parameter, 1)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::CubicSpline;
    use crate::{
        error::Error,
        interpolation::{DerivativeInterpolator, Interpolator},
    };

    fn linear_spline() -> CubicSpline {
        // Linear data: values[k] = k along axis 0, constant 0 along axis 1.
        let knots = array![0.0, 1.0, 2.0, 3.0];
        let values = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        CubicSpline::new(knots, values.view()).unwrap()
    }

    #[test]
    fn construction_rejects_single_knot() {
        let knots = array![0.0];
        let values = array![[1.0]];

        let result = CubicSpline::new(knots, values.view());

        assert!(matches!(
            result,
            Err(Error::InterpolationKnotCount { knots: 1 })
        ));
    }

    #[test]
    fn construction_rejects_shape_mismatch() {
        let knots = array![0.0, 1.0];
        let values = array![[1.0], [2.0], [3.0]];

        let result = CubicSpline::new(knots, values.view());

        assert!(matches!(
            result,
            Err(Error::InterpolationShapeMismatch {
                knots: 2,
                value_rows: 3
            })
        ));
    }

    #[test]
    fn construction_rejects_non_increasing_knots() {
        let knots = array![0.0, 2.0, 1.0];
        let values = array![[0.0], [1.0], [2.0]];

        let result = CubicSpline::new(knots, values.view());

        assert!(matches!(
            result,
            Err(Error::InterpolationKnotsNotIncreasing { index: 1 })
        ));
    }

    #[test]
    fn samples_exactly_at_knots() {
        let knots = array![0.0, 1.0, 3.0, 6.0];
        let values = array![[1.0, 4.0], [3.0, 1.0], [2.0, 5.0], [7.0, 2.0]];
        let spline = CubicSpline::new(knots.clone(), values.view()).unwrap();

        for (row, &knot) in knots.iter().enumerate() {
            let sample = spline.sample(knot);
            for axis in 0..2 {
                assert!((sample[axis] - values[[row, axis]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn two_knot_linear_interpolation() {
        // With only two knots the system collapses to linear interpolation.
        let knots = array![0.0, 2.0];
        let values = array![[1.0, 4.0], [3.0, 0.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();

        let mid = spline.sample(1.0);
        assert!((mid[0] - 2.0).abs() < 1e-12);
        assert!((mid[1] - 2.0).abs() < 1e-12);

        for parameter in [0.0, 0.5, 1.0, 1.5, 2.0] {
            let slope = spline.derivative(parameter);
            assert!((slope[0] - 1.0).abs() < 1e-12);
            assert!((slope[1] - (-2.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_data_at_all_orders() {
        // A natural cubic spline through linear data is itself linear:
        // order 0 reproduces the line,
        // order 1 is the constant slope, and
        // orders 2 and above vanish.
        let spline = linear_spline();
        for parameter in [0.25, 0.5, 0.75, 1.0, 1.3, 1.7, 2.0, 2.5, 2.99] {
            let sample = spline.sample_with_order(parameter, 0);
            let derivative = spline.sample_with_order(parameter, 1);

            assert!((sample[0] - parameter).abs() < 1e-10);
            assert!(sample[1].abs() < 1e-10);
            assert!((derivative[0] - 1.0).abs() < 1e-10);
            assert!(derivative[1].abs() < 1e-10);

            for order in 2..=4 {
                let higher = spline.sample_with_order(parameter, order);
                assert!(higher.iter().all(|component| component.abs() < 1e-10));
            }
        }
    }

    fn oscillating_spline() -> CubicSpline {
        let knots = array![0.0, 1.0, 2.0, 3.0];
        let values = array![[0.0], [1.0], [0.0], [1.0]];
        CubicSpline::new(knots, values.view()).unwrap()
    }

    #[test]
    fn matches_scipy_natural_spline_values() {
        // Reference values from scipy.interpolate.CubicSpline with
        // bc_type="natural" on the same knots and values.
        let spline = oscillating_spline();
        let references = [(0.5, 0.75), (1.5, 0.49999999999999994), (2.5, 0.25)];
        for (parameter, expected) in references {
            assert!(
                (spline.sample(parameter)[0] - expected).abs() < 1e-10,
                "sample at {parameter} deviates from the scipy reference {expected}",
            );
        }
    }

    #[test]
    fn second_derivative_vanishes_at_endpoints() {
        // The defining property of the natural boundary condition.
        let spline = oscillating_spline();
        for endpoint in [0.0, 3.0] {
            assert!(spline.sample_with_order(endpoint, 2)[0].abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "parameter -0.001 outside of domain [0, 3]")]
    fn sample_outside_domain_panics() {
        let spline = linear_spline();
        let _ = spline.sample(-0.001);
    }

    #[test]
    fn matches_hand_solved_system_at_non_uniform_non_unit_knots() {
        let knots = array![0.0, 1.0, 4.0, 6.0];
        let values = array![[0.0], [1.0], [0.0], [2.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();

        let references = [(0.5, 42.0 / 71.0), (2.0, 66.0 / 71.0), (5.0, 49.0 / 71.0)];
        for (parameter, expected) in references {
            let sample = spline.sample(parameter)[0];
            assert!(
                (sample - expected).abs() < 1e-12,
                "sample at {parameter} was {sample}, expected {expected}",
            );
        }
    }
}
