// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! A trajectory of points in a metric space.

use ndarray::{Array2, ArrayView2, Axis};

use crate::{
    error::{Error, Result},
    interpolation::Interpolator,
    metric::Metric,
    util::fingerprint::Fingerprint,
};

/// A trajectory of points in a metric space.
///
/// Pure data: the dense point array together with the map from each user-facing
/// sample to its row in that array.
///
/// # Sample vs. point indices
///
/// A trajectory carries two index spaces:
///
/// - **Sample index** (`0..self.original_count()`): an index into the user's
///   input data. This is the default user-facing index space. Every public
///   method elsewhere in the crate that takes a trajectory index (segment
///   ranges for
///   [`EmbeddedTrajectory::walk_cycle`](crate::EmbeddedTrajectory::walk_cycle),
///   [`EmbeddedTrajectory::cycle_class`](crate::EmbeddedTrajectory::cycle_class),
///   [`EmbeddedTrajectory::signature`](crate::EmbeddedTrajectory::signature),
///   etc.) interprets it as a sample index unless explicitly stated otherwise.
///
/// - **Point index** (`0..self.len()`): an index into
///   [`points()`](Self::points), the dense row array stored internally. For
///   trajectories built with [`new`](Self::new), point index equals sample
///   index. For trajectories built with [`resample`](Self::resample), point
///   indices additionally cover the bisection-inserted fill rows that densify
///   the trajectory for downstream cube-adjacency invariants.
///
/// [`original_indices()`](Self::original_indices) is the bridge: sample
/// index `i` corresponds to point index `original_indices()[i]`. Every
/// sample is a point; not every point is a sample.
///
/// Direct access to the dense form ([`points()`](Self::points),
/// [`len()`](Self::len)) is advanced. Most callers interact with the
/// trajectory by passing it to higher-level types and never touch the dense
/// form directly.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trajectory {
    #[cfg_attr(feature = "serde", serde(with = "crate::persistence::npy_field"))]
    points: Array2<f64>,
    original_indices: Vec<usize>,
}

impl Trajectory {
    /// Builds a trajectory from a dense point array, mapping each row to a
    /// sample index in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
    /// let trajectory = Trajectory::new(points.view()).unwrap();
    /// assert_eq!(trajectory.original_count(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::TrajectoryEmpty`] if `points` has zero rows.
    /// - [`Error::TrajectoryNonFinite`] if any coordinate is not finite.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(points: ArrayView2<'_, f64>) -> Result<Self> {
        if points.nrows() == 0 {
            return Err(Error::TrajectoryEmpty);
        }
        for (row, point) in points.outer_iter().enumerate() {
            for (column, coordinate) in point.iter().enumerate() {
                if !coordinate.is_finite() {
                    return Err(Error::TrajectoryNonFinite { row, column });
                }
            }
        }
        let original_indices = (0..points.nrows()).collect();
        Ok(Self {
            points: points.to_owned(),
            original_indices,
        })
    }

    /// Resamples `interpolator` so that consecutive output samples are within
    /// `bound` metric distance under `metric`.
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
    /// let trajectory = Trajectory::resample(&spline, &Euclidean, 0.5).unwrap();
    /// let embedded = EmbeddedTrajectory::new(
    ///     trajectory,
    ///     Box::new(Euclidean),
    ///     &ExecutionBackend::default(),
    /// )
    /// .unwrap();
    /// assert!(embedded.bound() <= 0.5);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::InterpolationKnotCount`] if `interpolator.knots().len() < 2`.
    /// - [`Error::TrajectoryNonFinite`] if any interpolator output is not
    ///   finite.
    /// - [`Error::ResampleStagnation`] if bisection cannot reduce the metric
    ///   distance below `bound` at machine precision.
    #[allow(clippy::missing_panics_doc)]
    pub fn resample<I: Interpolator>(
        interpolator: &I,
        metric: &dyn Metric,
        bound: f64,
    ) -> Result<Self> {
        let knots = interpolator.knots();
        if knots.len() < 2 {
            return Err(Error::InterpolationKnotCount {
                actual: knots.len(),
            });
        }

        let mut samples: Vec<ndarray::Array1<f64>> = Vec::new();
        let first_sample = interpolator.sample(knots[0]);
        assert_finite_sample(&first_sample, 0)?;
        samples.push(first_sample);
        let mut original_indices: Vec<usize> = vec![0];

        for pair in knots.windows(2) {
            let (parameter_lower, parameter_upper) = (pair[0], pair[1]);
            let sample_upper = interpolator.sample(parameter_upper);
            assert_finite_sample(&sample_upper, samples.len())?;
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
                    <= bound
                {
                    samples.push(interval.sample_upper);
                    continue;
                }
                let parameter_mid = interval.midpoint_parameter();
                if interval.is_stagnant(parameter_mid) {
                    return Err(Error::ResampleStagnation {
                        time: interval.parameter_lower,
                    });
                }
                let sample_mid = interpolator.sample(parameter_mid);
                assert_finite_sample(&sample_mid, samples.len())?;
                let (left, right) = interval.split(parameter_mid, sample_mid);
                // Push right first so left is popped (and emitted) first.
                stack.push(right);
                stack.push(left);
            }
            original_indices.push(samples.len() - 1);
        }

        let dimension = samples[0].len();
        let mut points = Array2::<f64>::zeros((samples.len(), dimension));
        for (row, sample) in samples.iter().enumerate() {
            points.index_axis_mut(Axis(0), row).assign(sample);
        }
        Ok(Self {
            points,
            original_indices,
        })
    }

    /// The dense row array as a 2D view, indexed by point index.
    ///
    /// Advanced: most callers operate in sample-index space and reach for the
    /// sample-only accessors instead. The dense form includes any
    /// bisection-inserted fill rows from [`resample`](Self::resample).
    #[must_use]
    pub fn points(&self) -> ArrayView2<'_, f64> {
        self.points.view()
    }

    /// The number of dense rows in [`points`](Self::points).
    ///
    /// Advanced. Equals [`original_count`](Self::original_count) for
    /// trajectories built with [`new`](Self::new); strictly greater for
    /// trajectories built with [`resample`](Self::resample).
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.nrows()
    }

    /// The embedding dimension of each point.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.points.ncols()
    }

    /// The point-index of each sample.
    ///
    /// The bridge between sample-index space (`0..original_count()`) and
    /// point-index space (`0..len()`): sample `i` lives at point
    /// `original_indices()[i]`. Bisection-inserted fill rows from
    /// [`resample`](Self::resample) have no sample index.
    ///
    /// For trajectories built with [`new`](Self::new), this is the identity
    /// map `0..len()`.
    #[must_use]
    pub fn original_indices(&self) -> &[usize] {
        &self.original_indices
    }

    /// The number of samples in the trajectory.
    ///
    /// The user-facing length. For trajectories built with
    /// [`new`](Self::new), this equals [`len`](Self::len); for trajectories
    /// built with [`resample`](Self::resample), it equals the interpolator's
    /// knot count.
    #[must_use]
    pub fn original_count(&self) -> usize {
        self.original_indices.len()
    }

    /// A stable 64-bit fingerprint of this trajectory's content.
    ///
    /// Derived from the points and the sample-to-point index map. Two
    /// trajectories with the same content fingerprint identically; changing
    /// either input changes the fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fingerprint::new();
        hasher.write(&(self.points.nrows() as u64).to_le_bytes());
        hasher.write(&(self.points.ncols() as u64).to_le_bytes());
        for &value in &self.points {
            hasher.write(&value.to_le_bytes());
        }
        hasher.write(&(self.original_indices.len() as u64).to_le_bytes());
        for &index in &self.original_indices {
            hasher.write(&(index as u64).to_le_bytes());
        }
        hasher.finish()
    }
}

// 2^40 subdivisions of any parameter interval is the f64 precision wall.
const MAX_DEPTH: u32 = 40;

/// A bisection-stack entry: an interval of parameter space and the inner
/// interpolator samples at its endpoints.
struct Interval {
    parameter_lower: f64,
    sample_lower: ndarray::Array1<f64>,
    parameter_upper: f64,
    sample_upper: ndarray::Array1<f64>,
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
    fn split(self, parameter_mid: f64, sample_mid: ndarray::Array1<f64>) -> (Self, Self) {
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

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn max_consecutive_distance(points: ArrayView2<'_, f64>, metric: &dyn Metric) -> f64 {
    let mut max = 0.0_f64;
    for point_index in 0..points.nrows().saturating_sub(1) {
        let distance = metric.distance(points.row(point_index), points.row(point_index + 1));
        if distance > max {
            max = distance;
        }
    }
    max
}

fn assert_finite_sample(sample: &ndarray::Array1<f64>, row: usize) -> Result<()> {
    for (column, coordinate) in sample.iter().enumerate() {
        if !coordinate.is_finite() {
            return Err(Error::TrajectoryNonFinite { row, column });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};

    use super::{Trajectory, max_consecutive_distance};
    use crate::{
        error::Error,
        interpolation::{CubicSpline, Interpolator},
        metric::{Euclidean, Metric},
    };

    #[test]
    fn new_records_points_and_index_map() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.dimension(), 2);
        assert_eq!(trajectory.original_indices(), &[0, 1, 2]);
        assert_eq!(trajectory.original_count(), 3);
    }

    #[test]
    fn new_returns_err_on_empty() {
        let points = Array2::<f64>::zeros((0, 3));
        let outcome = Trajectory::new(points.view());

        assert!(matches!(outcome.unwrap_err(), Error::TrajectoryEmpty));
    }

    #[test]
    fn new_returns_err_on_non_finite() {
        let points = array![[0.0, 0.0], [1.0, f64::NAN]];
        let outcome = Trajectory::new(points.view());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::TrajectoryNonFinite { row: 1, column: 1 },
        ));
    }

    #[test]
    fn resample_meets_bound_under_euclidean() {
        let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = array![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();
        let bound = 0.5;

        let trajectory = Trajectory::resample(&spline, &Euclidean, bound).unwrap();

        assert!(max_consecutive_distance(trajectory.points(), &Euclidean) <= bound);
        for point_index in 0..trajectory.len() - 1 {
            let distance = Euclidean.distance(
                trajectory.points().row(point_index),
                trajectory.points().row(point_index + 1),
            );
            assert!(distance <= bound + 1e-12);
        }

        let indices = trajectory.original_indices();
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[4], trajectory.len() - 1);
        for window in indices.windows(2) {
            assert!(window[0] < window[1]);
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

        let trajectory = Trajectory::resample(&spline, &Euclidean, 0.1).unwrap();

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

            fn dimension(&self) -> usize {
                2
            }
        }

        let outcome = Trajectory::resample(&SingleKnotInterpolator, &Euclidean, 0.1);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::InterpolationKnotCount { actual: 1 },
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

            fn dimension(&self) -> usize {
                2
            }
        }

        let outcome = Trajectory::resample(&PathologicalInterpolator, &Euclidean, 0.1);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::ResampleStagnation { .. }
        ));
    }
}
