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
#[cfg_attr(feature = "serde", serde(try_from = "TrajectoryData"))]
pub struct Trajectory {
    #[cfg_attr(feature = "serde", serde(with = "crate::serialization::npy_field"))]
    points: Array2<f64>,
    parameters: Vec<f64>,
}

/// The wire form of a [`Trajectory`], decoded before validation.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct TrajectoryData {
    #[serde(with = "crate::serialization::npy_field")]
    points: Array2<f64>,
    parameters: Vec<f64>,
}

#[cfg(feature = "serde")]
impl TryFrom<TrajectoryData> for Trajectory {
    type Error = Error;

    fn try_from(data: TrajectoryData) -> Result<Self> {
        Self::from_parts(data.points, data.parameters)
    }
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
        Self::from_parts(points.to_owned(), parameters.to_vec())
    }

    /// Builds a trajectory from owned points and an owned parameterization,
    /// applying the validation [`with_parameters`](Self::with_parameters)
    /// documents.
    fn from_parts(points: Array2<f64>, parameters: Vec<f64>) -> Result<Self> {
        check_points(points.view())?;
        if parameters.len() != points.nrows() {
            return Err(Error::TrajectoryParameterCount {
                parameters: parameters.len(),
                points: points.nrows(),
            });
        }
        for pair in parameters.windows(2) {
            // Negated: also rejects a NaN parameter.
            if !(pair[0] < pair[1]) {
                return Err(Error::TrajectoryParametersNotIncreasing);
            }
        }
        Ok(Self { points, parameters })
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

    /// The maximum metric distance between consecutive points: the finest
    /// separation this trajectory resolves. `0.0` when there are fewer than
    /// two points.
    ///
    /// This is the floor for a valid [`downsample`](Self::downsample)
    /// spacing, and the quantity
    /// [`EmbeddedTrajectory::resolution`](crate::EmbeddedTrajectory::resolution)
    /// reports for the trajectory an embedding was built over.
    ///
    /// # Panics
    ///
    /// Under [`Metric::SphereBundle`], panics if the points have odd
    /// dimension, which the metric cannot split into position and direction
    /// halves.
    #[must_use]
    pub fn resolution(&self, metric: Metric) -> f64 {
        metric.over(self.points()).max_consecutive_distance()
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

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::Trajectory;
    use crate::error::Error;
    #[cfg(feature = "serde")]
    use crate::serialization::{load_from_reader, save_to_writer};

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
        // A NaN parameter is rejected by the guard's negated form.
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

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_rejects_non_increasing_parameters() {
        // Built field by field so the payload carries a parameterization no
        // constructor would have accepted, which is the shape a corrupted or
        // hand-built file arrives in.
        let unchecked = Trajectory {
            points: array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0]],
            parameters: vec![0.0, 2.0, 1.0],
        };
        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &unchecked).unwrap();

        assert!(matches!(
            load_from_reader::<Trajectory, _>(&buffer[..]).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }
}
