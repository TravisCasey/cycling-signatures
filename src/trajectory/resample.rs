// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Resampling: knot-rooted bisection that builds a trajectory from an
//! interpolator, inserting points between its knots until consecutive ones
//! meet a metric spacing.

use ndarray::{Array1, Array2, Axis};

use super::Trajectory;
use crate::{
    error::{Error, Result},
    interpolation::Interpolator,
    metric::Metric,
};

impl Trajectory {
    /// Resamples `interpolator` under `metric` by knot-rooted bisection until
    /// consecutive points are within `spacing` of each other.
    ///
    /// `spacing` is the fidelity knob for the cubical model of the curve: a
    /// spacing of at most 1 (the cube side) puts consecutive points in
    /// intersecting cubes, which is what makes
    /// [`CubicalCover::build`](crate::CubicalCover::build) over the result a
    /// faithful tube around the curve. That requirement belongs to the cover
    /// and is checked there: a coarser spacing resamples without error and
    /// surfaces at the cover build as
    /// [`Error::ConsecutiveCubesNonAdjacent`]. Each output point records the
    /// interpolator parameter it was sampled at, so the knots appear among
    /// [`parameters`](Self::parameters) alongside the fill points bracketed
    /// between them.
    ///
    /// Thinning to the coarser resolution cycle detection runs at is a
    /// separate step: see [`downsample`](Self::downsample).
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let knots = array![0.0, 1.0, 2.0];
    /// let values = array![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]];
    /// let spline = CubicSpline::new(knots, values.view()).unwrap();
    /// let trajectory =
    ///     Trajectory::resample(&spline, Metric::Euclidean, 0.5).unwrap();
    /// let cover =
    ///     CubicalCover::build(&trajectory, &ExecutionBackend::default()).unwrap();
    /// let embedded =
    ///     EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean).unwrap();
    /// assert!(embedded.bound() <= 0.5);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::SpacingNotPositive`] if `spacing` is zero, negative or NaN.
    /// - [`Error::InterpolationKnotCount`] if `interpolator.knots().len() < 2`.
    /// - [`Error::ResampleNonFinite`] if any interpolator output is not finite.
    /// - [`Error::ResampleStagnation`] if bisection cannot reduce the metric
    ///   distance below `spacing` at machine precision.
    ///
    /// # Panics
    ///
    /// Panics if `interpolator.knots()` is not strictly increasing. The
    /// emitted parameters inherit their order from the knots, so a knot
    /// sequence breaking that contract cannot produce a valid trajectory.
    pub fn resample<I: Interpolator>(
        interpolator: &I,
        metric: Metric,
        spacing: f64,
    ) -> Result<Self> {
        // Negated form (rather than `spacing <= 0.0`) so a NaN spacing fails
        // loudly instead of silently passing the comparison.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(spacing > 0.0) {
            return Err(Error::SpacingNotPositive { spacing });
        }

        let knots = interpolator.knots();
        if knots.len() < 2 {
            return Err(Error::InterpolationKnotCount { knots: knots.len() });
        }
        // Strictly increasing knots are part of the `Interpolator` contract.
        assert!(
            knots.windows(2).all(|pair| pair[0] < pair[1]),
            "interpolator knots must be strictly increasing"
        );

        let mut samples: Vec<Array1<f64>> = Vec::new();
        let mut parameters: Vec<f64> = Vec::new();
        let first_sample = interpolator.sample(knots[0]);
        check_finite_sample(&first_sample, knots[0])?;
        samples.push(first_sample);
        parameters.push(knots[0]);

        for pair in knots.windows(2) {
            let (parameter_lower, parameter_upper) = (pair[0], pair[1]);
            let sample_upper = interpolator.sample(parameter_upper);
            check_finite_sample(&sample_upper, parameter_upper)?;
            let mut stack = vec![Interval {
                parameter_lower,
                sample_lower: samples
                    .last()
                    .expect("at least one sample exists at this point")
                    .clone(),
                parameter_upper,
                sample_upper,
                depth: 0,
            }];
            while let Some(interval) = stack.pop() {
                if metric.distance(interval.sample_lower.view(), interval.sample_upper.view())
                    <= spacing
                {
                    samples.push(interval.sample_upper);
                    parameters.push(interval.parameter_upper);
                    continue;
                }
                let parameter_mid = interval.midpoint_parameter();
                if interval.is_stagnant(parameter_mid) {
                    return Err(Error::ResampleStagnation {
                        parameter: interval.parameter_lower,
                    });
                }
                let sample_mid = interpolator.sample(parameter_mid);
                check_finite_sample(&sample_mid, parameter_mid)?;
                let (left, right) = interval.split(parameter_mid, sample_mid);
                // Push right first so left is popped (and emitted) first.
                stack.push(right);
                stack.push(left);
            }
        }

        let dimension = samples[0].len();
        let mut points = Array2::<f64>::zeros((samples.len(), dimension));
        for (row, sample) in samples.iter().enumerate() {
            points.index_axis_mut(Axis(0), row).assign(sample);
        }

        Ok(Self { points, parameters })
    }
}

// 2^40 subdivisions of any parameter interval is the f64 precision limit.
const MAX_DEPTH: u32 = 40;

/// A bisection-stack entry: an interval of parameter space and the inner
/// interpolator samples at its endpoints.
struct Interval {
    parameter_lower: f64,
    sample_lower: Array1<f64>,
    parameter_upper: f64,
    sample_upper: Array1<f64>,
    depth: u32,
}

impl Interval {
    fn midpoint_parameter(&self) -> f64 {
        self.parameter_lower.midpoint(self.parameter_upper)
    }

    #[allow(clippy::float_cmp)]
    fn is_stagnant(&self, parameter_mid: f64) -> bool {
        parameter_mid == self.parameter_lower
            || parameter_mid == self.parameter_upper
            || self.depth >= MAX_DEPTH
    }

    /// Splits at `parameter_mid` (where the inner sample is `sample_mid`).
    /// Consumes `self`; returns the left and right halves.
    fn split(self, parameter_mid: f64, sample_mid: Array1<f64>) -> (Self, Self) {
        let left = Self {
            parameter_lower: self.parameter_lower,
            sample_lower: self.sample_lower,
            parameter_upper: parameter_mid,
            sample_upper: sample_mid.clone(),
            depth: self.depth + 1,
        };
        let right = Self {
            parameter_lower: parameter_mid,
            sample_lower: sample_mid,
            parameter_upper: self.parameter_upper,
            sample_upper: self.sample_upper,
            depth: self.depth + 1,
        };
        (left, right)
    }
}

/// Rejects a non-finite coordinate in the interpolator sample taken at
/// `parameter`.
fn check_finite_sample(sample: &Array1<f64>, parameter: f64) -> Result<()> {
    for (column, coordinate) in sample.iter().enumerate() {
        if !coordinate.is_finite() {
            return Err(Error::ResampleNonFinite { parameter, column });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::Trajectory;
    use crate::{
        error::Error,
        interpolation::{CubicSpline, Interpolator},
        metric::Metric,
    };

    #[test]
    fn resample_meets_spacing_and_records_parameters() {
        let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = array![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]];
        let spline = CubicSpline::new(knots.clone(), values.view()).unwrap();
        let spacing = 0.5;

        let trajectory = Trajectory::resample(&spline, Metric::Euclidean, spacing).unwrap();

        let points = trajectory.points();
        for point_index in 0..trajectory.len() - 1 {
            let distance =
                Metric::Euclidean.distance(points.row(point_index), points.row(point_index + 1));
            assert!(distance <= spacing + 1e-12);
        }

        let parameters = trajectory.parameters();
        assert_eq!(parameters.len(), trajectory.len());
        assert!(
            parameters.windows(2).all(|pair| pair[0] < pair[1]),
            "resampled parameters are not strictly increasing: {parameters:?}",
        );
        // Knots are recorded verbatim, so the match is exact rather than
        // approximate.
        for knot in &knots {
            assert!(
                parameters
                    .iter()
                    .any(|parameter| parameter.total_cmp(knot).is_eq()),
                "knot {knot} is missing from the recorded parameters",
            );
        }
    }

    #[test]
    fn resample_preserves_final_knot_sample() {
        // The final resampled point must equal the interpolator's sample at
        // the last knot. This depends on bisection correctly threading the
        // outer `sample_upper` through every right-half split.
        let knots = array![0.0, 1.0, 2.0];
        let values = array![[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]];
        let spline = CubicSpline::new(knots.clone(), values.view()).unwrap();

        let trajectory = Trajectory::resample(&spline, Metric::Euclidean, 0.1).unwrap();

        let last = trajectory.points().row(trajectory.len() - 1).to_owned();
        let knot_last = spline.sample(knots[knots.len() - 1]);
        for axis in 0..2 {
            assert!((last[axis] - knot_last[axis]).abs() < 1e-12);
        }
    }

    #[test]
    fn resample_returns_err_on_single_knot_interpolator() {
        struct SingleKnotInterpolator;

        impl Interpolator for SingleKnotInterpolator {
            fn sample(&self, _parameter: f64) -> Array1<f64> {
                array![0.0, 0.0]
            }

            fn knots(&self) -> &[f64] {
                &[0.0]
            }
        }

        let outcome = Trajectory::resample(&SingleKnotInterpolator, Metric::Euclidean, 0.1);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::InterpolationKnotCount { knots: 1 },
        ));
    }

    #[test]
    fn resample_returns_err_on_stagnation() {
        // A pathological interpolator: the first knot sample is at [0, 0] and
        // all other samples (including the second knot and any midpoints) are
        // at [1000, 0], so consecutive distances always exceed any reasonable
        // bound and bisection can never help.
        struct PathologicalInterpolator;

        impl Interpolator for PathologicalInterpolator {
            fn sample(&self, parameter: f64) -> Array1<f64> {
                // Return [0, 0] at the first knot; [1000, 0] everywhere else.
                #[allow(clippy::float_cmp)]
                if parameter == 0.0 {
                    array![0.0, 0.0]
                } else {
                    array![1000.0, 0.0]
                }
            }

            fn knots(&self) -> &[f64] {
                &[0.0, 1.0]
            }
        }

        let outcome = Trajectory::resample(&PathologicalInterpolator, Metric::Euclidean, 0.1);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::ResampleStagnation { .. }
        ));
    }

    #[test]
    fn resample_rejects_non_positive_spacing() {
        let knots = array![0.0, 1.0];
        let values = array![[0.0, 0.0], [1.0, 0.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();

        // A spacing of zero admits no distance at all, so bisection could
        // only stagnate on it; it must be rejected up front instead.
        let zero_outcome = Trajectory::resample(&spline, Metric::Euclidean, 0.0);
        assert!(matches!(
            zero_outcome.unwrap_err(),
            Error::SpacingNotPositive { .. }
        ));

        // A NaN spacing fails every comparison, so the guard must be written
        // to reject it rather than let it silently pass.
        let nan_outcome = Trajectory::resample(&spline, Metric::Euclidean, f64::NAN);
        assert!(matches!(
            nan_outcome.unwrap_err(),
            Error::SpacingNotPositive { .. }
        ));
    }

    #[test]
    fn resample_reports_the_parameter_of_a_non_finite_sample() {
        // An interpolator whose second knot sample is non-finite. The report
        // names the parameter it was taken at, which is knowable before the
        // trajectory it would have belonged to exists.
        struct NonFiniteInterpolator;

        impl Interpolator for NonFiniteInterpolator {
            fn sample(&self, parameter: f64) -> Array1<f64> {
                #[allow(clippy::float_cmp)]
                if parameter == 2.0 {
                    array![f64::INFINITY, 0.0]
                } else {
                    array![parameter, 0.0]
                }
            }

            fn knots(&self) -> &[f64] {
                &[0.0, 2.0]
            }
        }

        let outcome = Trajectory::resample(&NonFiniteInterpolator, Metric::Euclidean, 0.5);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::ResampleNonFinite { parameter, column: 0 } if (parameter - 2.0).abs() < 1e-12
        ));
    }
}
