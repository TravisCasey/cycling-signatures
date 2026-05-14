// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Crate-level error types.

/// Errors specific to the cycling-signatures crate.
///
/// The enum is `#[non_exhaustive]`; downstream `match` statements should
/// include a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A spanning vector passed to [`crate::F2Subspace::new`] did not have
    /// the expected length.
    #[error("spanning vector at index {index} has length {actual}, expected {expected}")]
    F2SubspaceVectorLength {
        /// Position of the offending vector in the input slice.
        index: usize,
        /// Length of the offending vector.
        actual: usize,
        /// The expected length.
        expected: usize,
    },

    /// An interpolator was constructed with fewer than two knots.
    #[error("interpolation requires at least two knots, got {actual}")]
    InterpolationKnotCount {
        /// Number of knots supplied.
        actual: usize,
    },

    /// Knot count and value row count disagreed.
    #[error("interpolation has {knots} knots and {value_rows} value rows")]
    InterpolationShapeMismatch {
        /// Number of knots supplied.
        knots: usize,
        /// Number of value rows supplied.
        value_rows: usize,
    },

    /// Interpolation values had zero columns.
    #[error("interpolation values have zero columns")]
    InterpolationEmptyValues,

    /// Interpolation knots were not strictly increasing.
    #[error("interpolation knots are not strictly increasing at index {index}")]
    InterpolationKnotsNotIncreasing {
        /// Position in the knots array of the first non-increasing pair.
        /// The offending pair is `knots[index]` and `knots[index + 1]`.
        index: usize,
    },

    /// Sphere bundle metric direction weight was not finite or not strictly
    /// positive.
    #[error("sphere bundle direction weight {value} is not finite or not strictly positive")]
    SphereBundleMetricWeight {
        /// The rejected weight.
        value: f64,
    },

    /// A cube coordinate is outside the range the cubical-homology backend
    /// accepts.
    #[error(
        "cube coordinate {value} at axis {axis} is outside the allowed range [{}, {}]",
        i16::MIN,
        i16::MAX - 1
    )]
    CubeCoordinateOutOfRange {
        /// The axis (cube column) of the offending coordinate.
        axis: usize,
        /// The rejected coordinate.
        value: i64,
    },

    /// [`CubicalCover::from_cubes`](crate::CubicalCover::from_cubes) was called
    /// with zero rows.
    #[error("cubical cover requires at least one cube, got zero")]
    CubicalCoverEmptyCubes,

    /// [`CubicalCover::from_cubes`](crate::CubicalCover::from_cubes) was called
    /// with zero columns.
    #[error("cubical cover requires at least one spatial dimension, got zero")]
    CubicalCoverZeroDimension,

    /// An [`EmbeddedTrajectory`](crate::EmbeddedTrajectory)'s trajectory and
    /// cover disagree on dimension.
    #[error("embedded trajectory dimension mismatch: trajectory {trajectory}, cover {cover}")]
    EmbeddedDimensionMismatch {
        /// The trajectory's embedding dimension.
        trajectory: usize,
        /// The cover's spatial dimension.
        cover: usize,
    },

    /// A trajectory point's cube is not present in the supplied cover.
    #[error("trajectory point at index {point_index} maps to a cube not present in the cover")]
    EmbeddedCubeNotInCover {
        /// The index in `trajectory.points()` of the offending point.
        point_index: usize,
    },

    /// Trajectory input had zero rows.
    #[error("trajectory input has zero rows")]
    TrajectoryEmpty,

    /// A trajectory coordinate was non-finite.
    #[error("trajectory coordinate at row {row}, column {column} is not finite")]
    TrajectoryNonFinite {
        /// Row of the offending coordinate in the trajectory points matrix.
        row: usize,
        /// Column of the offending coordinate.
        column: usize,
    },

    /// Bisection in `Trajectory::resample` could not bring consecutive
    /// samples within the requested bound at machine precision.
    #[error("trajectory bisection stagnated near parameter {time}")]
    ResampleStagnation {
        /// The parameter value at which bisection stagnated.
        time: f64,
    },
}

/// Convenience alias for [`std::result::Result`] with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
