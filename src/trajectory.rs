// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! A trajectory of points in a metric space.

mod downsample;
mod resample;

#[cfg(feature = "serde")]
use std::path::Path;

use ndarray::{Array2, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    metric::Metric,
    util::fingerprint::Fingerprint,
};

/// An ordered array of points in a metric space together with a strictly
/// increasing parameterization, one value per point.
///
/// The parameterization is carried, never consumed. Nothing in cube covering,
/// cycle detection, or walking reads it: the caller decides what it means
/// (integration time, arc length, the row number of a raw sample) and reads it
/// back through [`parameters`](Self::parameters). [`new`](Self::new) assigns
/// `0.0, 1.0, ...`, so parameters default to point indices rather than
/// physical time; [`resample`](Self::resample) records the interpolator
/// parameter of each emitted point, and [`downsample`](Self::downsample)
/// carries the surviving points' entries through unchanged.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trajectory {
    #[cfg_attr(feature = "serde", serde(with = "crate::serialization::npy_field"))]
    points: Array2<f64>,
    parameters: Vec<f64>,
}

impl Trajectory {
    /// Builds a trajectory from a point array under the index
    /// parameterization `0.0, 1.0, ...`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
    /// let trajectory = Trajectory::new(points.view()).unwrap();
    /// assert_eq!(trajectory.len(), 3);
    /// assert_eq!(trajectory.parameters(), &[0.0, 1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::TrajectoryEmpty`] if `points` has zero rows.
    /// - [`Error::TrajectoryNonFinite`] if any coordinate is not finite.
    pub fn new(points: ArrayView2<'_, f64>) -> Result<Self> {
        check_points(points)?;
        let parameters = (0..points.nrows()).map(|index| index as f64).collect();
        Ok(Self {
            points: points.to_owned(),
            parameters,
        })
    }

    /// Builds a trajectory from a point array under a caller-supplied
    /// parameterization.
    ///
    /// Use this where the points carry a meaning of their own along the curve,
    /// such as the integration times a solver emitted them at. For the index
    /// parameterization, use [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// Returns
    ///
    /// - [`Error::TrajectoryEmpty`] if `points` has zero rows.
    /// - [`Error::TrajectoryNonFinite`] if any coordinate is not finite.
    /// - [`Error::TrajectoryParameterCount`] if `parameters` does not have one
    ///   value per point.
    /// - [`Error::TrajectoryParametersNotIncreasing`] if `parameters` is not
    ///   strictly increasing (including when a value is NaN).
    pub fn with_parameters(points: ArrayView2<'_, f64>, parameters: &[f64]) -> Result<Self> {
        check_points(points)?;
        if parameters.len() != points.nrows() {
            return Err(Error::TrajectoryParameterCount {
                parameters: parameters.len(),
                points: points.nrows(),
            });
        }
        for pair in parameters.windows(2) {
            // Negated form (rather than `pair[1] <= pair[0]`) so a NaN
            // parameter fails loudly instead of silently passing the
            // comparison.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(pair[0] < pair[1]) {
                return Err(Error::TrajectoryParametersNotIncreasing);
            }
        }
        Ok(Self {
            points: points.to_owned(),
            parameters: parameters.to_vec(),
        })
    }

    /// The point array as a 2D view, one row per point.
    #[must_use]
    pub fn points(&self) -> ArrayView2<'_, f64> {
        self.points.view()
    }

    /// The parameter value of each point, strictly increasing.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// The number of points in the trajectory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.nrows()
    }

    /// The embedding dimension of each point.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.points.ncols()
    }

    /// A stable 64-bit fingerprint of this trajectory's content.
    ///
    /// Derived from the points and the parameterization. Two trajectories with
    /// the same content fingerprint identically; changing either input changes
    /// the fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fingerprint::new();
        hasher.write(&(self.points.nrows() as u64).to_le_bytes());
        hasher.write(&(self.points.ncols() as u64).to_le_bytes());
        for &value in &self.points {
            hasher.write(&value.to_le_bytes());
        }
        hasher.write(&(self.parameters.len() as u64).to_le_bytes());
        for &parameter in &self.parameters {
            hasher.write(&parameter.to_le_bytes());
        }
        hasher.finish()
    }

    /// Writes this trajectory to `path` in the crate's binary format.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::serialization::save_to_path(path, self)
    }

    /// Reads a trajectory written by [`save`](Self::save).
    ///
    /// # Errors
    ///
    /// - [`Error::FormatVersionMismatch`] if the file's format version differs.
    /// - [`Error::Io`] if the file could not be opened.
    /// - [`Error::Deserialize`] if the file contents could not be read and
    ///   decoded.
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        crate::serialization::load_from_path(path)
    }
}

/// Rejects an empty or non-finite point array.
fn check_points(points: ArrayView2<'_, f64>) -> Result<()> {
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
    Ok(())
}

/// The maximum metric distance between consecutive rows of `points`, or `0.0`
/// when there are fewer than two rows.
pub(crate) fn max_consecutive_distance(points: ArrayView2<'_, f64>, metric: Metric) -> f64 {
    let mut max = 0.0_f64;
    for point_index in 0..points.nrows().saturating_sub(1) {
        let distance = metric.distance(points.row(point_index), points.row(point_index + 1));
        if distance > max {
            max = distance;
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::Trajectory;
    use crate::error::Error;

    #[test]
    fn new_records_points_under_the_index_parameterization() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.dimension(), 2);
        assert_eq!(trajectory.parameters(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn with_parameters_records_the_supplied_parameterization() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
        let trajectory = Trajectory::with_parameters(points.view(), &[0.25, 0.5, 4.0]).unwrap();

        assert_eq!(trajectory.parameters(), &[0.25, 0.5, 4.0]);
    }

    #[test]
    fn with_parameters_returns_err_on_count_mismatch() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
        let outcome = Trajectory::with_parameters(points.view(), &[0.0, 1.0]);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::TrajectoryParameterCount {
                parameters: 2,
                points: 3,
            },
        ));
    }

    #[test]
    fn with_parameters_returns_err_on_non_increasing() {
        // A NaN parameter fails every comparison, so the guard must be written
        // to reject it rather than let it silently pass.
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]];
        let outcome = Trajectory::with_parameters(points.view(), &[0.0, f64::NAN, 2.0]);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::TrajectoryParametersNotIncreasing
        ));
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
}
