// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Downsampling: a greedy forward walk that thins a trajectory to a
//! metric-spaced subset of its own points.

use ndarray::Array2;

use super::{Trajectory, max_consecutive_distance};
use crate::{
    error::{Error, Result},
    metric::Metric,
};

impl Trajectory {
    /// Thins `self` to a subset of its points at most `spacing` apart under
    /// `metric`, returning a new trajectory of the kept points and their
    /// parameters.
    ///
    /// A greedy forward walk always keeps the first and last point, and keeps
    /// an intermediate point once the following one would fall further than
    /// `spacing` from the last kept point. Re-running at a coarser spacing
    /// thins further; to thin more finely, start again from the denser
    /// trajectory.
    ///
    /// This is the detection resolution: the output is the vertex set cycle
    /// detection runs over, and its size is the dominant cost lever of a
    /// detection pass. Any spacing up to the intended detection threshold is
    /// valid: the threshold has to clear the output's own consecutive-point
    /// distance, which this spacing bounds.
    ///
    /// Only the lower end is validated here. A spacing coarse enough to put
    /// consecutive kept points more than one cube apart is rejected later, by
    /// the embedding, as [`Error::ConsecutiveCubesNonAdjacent`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
    /// let values =
    ///     array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
    /// let spline = CubicSpline::new(knots, values.view()).unwrap();
    /// let threshold = 0.5;
    ///
    /// let dense = Trajectory::resample(&spline, Metric::Euclidean, 0.05).unwrap();
    /// let cover =
    ///     CubicalCover::build(&dense, &ExecutionBackend::default()).unwrap();
    /// let detection = dense.downsample(Metric::Euclidean, threshold).unwrap();
    /// let embedded =
    ///     EmbeddedTrajectory::new(detection, cover, Metric::Euclidean).unwrap();
    /// assert!(embedded.bound() <= threshold);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::SpacingBelowResolution`] if `spacing` is less than the
    /// trajectory's own maximum consecutive-point distance (including when
    /// `spacing` is NaN): no subset of the points is spaced more finely than
    /// the points themselves are.
    #[allow(clippy::missing_panics_doc)]
    pub fn downsample(&self, metric: Metric, spacing: f64) -> Result<Self> {
        let points = self.points();
        let resolution = max_consecutive_distance(points, metric);
        // Negated form (rather than `spacing < resolution`) so a NaN spacing
        // fails loudly instead of silently passing the comparison.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(spacing >= resolution) {
            return Err(Error::SpacingBelowResolution {
                spacing,
                resolution,
            });
        }

        let row_count = points.nrows();

        // Greedy forward walk: keep a point once the following one would
        // exceed `spacing` from the last kept point.
        let mut kept: Vec<usize> = vec![0];
        for row in 1..row_count {
            let anchor = *kept.last().expect("seeded with the first point");
            if metric.distance(points.row(anchor), points.row(row)) > spacing {
                kept.push(row - 1);
            }
        }
        if *kept.last().expect("nonempty") != row_count - 1 {
            kept.push(row_count - 1);
        }

        let mut thinned_points = Array2::<f64>::zeros((kept.len(), points.ncols()));
        for (row, &source) in kept.iter().enumerate() {
            thinned_points.row_mut(row).assign(&points.row(source));
        }
        let thinned_parameters: Vec<f64> =
            kept.iter().map(|&source| self.parameters[source]).collect();

        Ok(Self {
            points: thinned_points,
            parameters: thinned_parameters,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::Trajectory;
    use crate::{error::Error, interpolation::CubicSpline, metric::Metric};

    #[test]
    fn downsample_thins_to_spacing_keeping_a_point_subset() {
        // Consecutive knot gaps grow from knot to knot (varying speed) while
        // every value stays on the x-axis (collinear), so density varies
        // across the trajectory while the walk still must keep every
        // consecutive gap within `spacing`.
        let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let values = array![[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [9.0, 0.0], [16.0, 0.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();
        let spacing = 1.0;

        let dense = Trajectory::resample(&spline, Metric::Euclidean, 0.05).unwrap();
        let thinned = dense.downsample(Metric::Euclidean, spacing).unwrap();

        // Points and parameters are carried over verbatim, so both are matched
        // exactly rather than approximately: walking the kept parameters
        // forward through the dense ones must never have to search backward,
        // and each match must bring its own point along.
        let dense_parameters = dense.parameters();
        let parameters = thinned.parameters();
        let mut search_from = 0;
        for (row, &parameter) in parameters.iter().enumerate() {
            let dense_row = (search_from..dense.len())
                .find(|&index| dense_parameters[index].total_cmp(&parameter).is_eq())
                .unwrap_or_else(|| {
                    panic!("kept parameter {parameter} is not a later dense parameter")
                });
            assert_eq!(thinned.points().row(row), dense.points().row(dense_row));
            search_from = dense_row + 1;
        }

        // First and last points always survive.
        assert_eq!(thinned.points().row(0), dense.points().row(0));
        assert_eq!(
            thinned.points().row(thinned.len() - 1),
            dense.points().row(dense.len() - 1),
        );

        let points = thinned.points();
        for row in 0..thinned.len() - 1 {
            let distance = Metric::Euclidean.distance(points.row(row), points.row(row + 1));
            assert!(distance <= spacing + 1e-12);
        }
    }

    #[test]
    fn downsample_coarsens_closely_spaced_points() {
        // A spacing far wider than the whole path collapses the trajectory to
        // just its first and last point: fewer points than went in, the
        // property downsampling exists to produce.
        let knots = array![0.0, 1.0, 2.0];
        let values = array![[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();

        let dense = Trajectory::resample(&spline, Metric::Euclidean, 0.1).unwrap();
        let thinned = dense.downsample(Metric::Euclidean, 10.0).unwrap();

        assert!(thinned.len() < dense.len());
    }

    #[test]
    fn downsample_rejects_spacing_below_resolution() {
        let knots = array![0.0, 1.0];
        let values = array![[0.0, 0.0], [1.0, 0.0]];
        let spline = CubicSpline::new(knots, values.view()).unwrap();
        let trajectory = Trajectory::resample(&spline, Metric::Euclidean, 0.05).unwrap();

        let outcome = trajectory.downsample(Metric::Euclidean, 0.01);
        assert!(matches!(
            outcome.unwrap_err(),
            Error::SpacingBelowResolution { .. }
        ));

        // A NaN spacing fails every comparison, so the guard must be written
        // to reject it rather than let it silently pass.
        let nan_outcome = trajectory.downsample(Metric::Euclidean, f64::NAN);
        assert!(matches!(
            nan_outcome.unwrap_err(),
            Error::SpacingBelowResolution { .. }
        ));
    }
}
