// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Crate-level error-handling and reporting.

/// Error type specific to the cycling-signatures crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A spanning vector passed to [`crate::F2Subspace::new`] did not have the
    /// expected length.
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
    #[error("interpolation requires at least two knots, got {knots}")]
    InterpolationKnotCount {
        /// Number of knots supplied.
        knots: usize,
    },

    /// Knot count and value row count disagreed.
    #[error("interpolation has {knots} knots but {value_rows} rows of values")]
    InterpolationShapeMismatch {
        /// Number of knots supplied.
        knots: usize,
        /// Number of value rows supplied.
        value_rows: usize,
    },

    /// Interpolation knots were not strictly increasing.
    #[error("interpolation knots are not strictly increasing at index {index}")]
    InterpolationKnotsNotIncreasing {
        /// Position in the knots array of the first non-increasing pair.
        /// The offending pair is `knots[index]` and `knots[index + 1]`.
        index: usize,
    },

    /// A cube coordinate is outside the range the cubical-homology backend
    /// accepts.
    #[error(
        "cube coordinate {coordinate} at axis {axis} is outside the allowed range [{}, {}]",
        i32::MIN,
        i32::MAX - 1
    )]
    CubeCoordinateOutOfRange {
        /// The axis (cube column) of the offending coordinate.
        axis: usize,
        /// The rejected coordinate.
        coordinate: i64,
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

    /// Consecutive trajectory points land in cubes differing by more than 1 in
    /// some axis. The cube embedding requires trajectory sampling fine enough
    /// that consecutive points stay in adjacent cubes; callers should resample
    /// with a smaller bound or rescale coordinates.
    #[error(
        "consecutive trajectory points {point_index} and {} land in cubes differing by {delta} in \
        axis {axis}",
        point_index + 1
    )]
    ConsecutiveCubesNonAdjacent {
        /// Index in `trajectory.points()` of the first point of the offending
        /// pair.
        point_index: usize,
        /// Spatial axis on which the cubes differ by more than 1.
        axis: usize,
        /// Signed coordinate difference between the two cubes on that axis.
        delta: i64,
    },

    /// A trajectory segment passed to a window or cycle query falls outside
    /// the valid range of the trajectory.
    #[error(
        "trajectory segment {start}..{end} does not lie within a trajectory of length \
         {trajectory_length}"
    )]
    WindowOutOfBounds {
        /// Start of the offending range.
        start: usize,
        /// End (exclusive) of the offending range.
        end: usize,
        /// Number of points in the trajectory.
        trajectory_length: usize,
    },

    /// A cycle-detection threshold below the embedded trajectory's
    /// consecutive-distance bound under its metric: cycles below this
    /// resolution are not meaningfully detectable.
    #[error(
        "adjacency threshold {threshold} is below the consecutive-distance bound \
         {trajectory_bound}"
    )]
    ThresholdBelowTrajectoryBound {
        /// The threshold the caller supplied.
        threshold: f64,
        /// The embedded trajectory's consecutive-distance bound under its
        /// metric.
        trajectory_bound: f64,
    },

    /// A cycle's endpoint cubes differ by more than 1 in some axis; the
    /// closing step of the walker requires adjacency.
    #[error(
        "cycle endpoints at trajectory indices {start} and {} land in cubes differing by {delta} in axis {axis}",
        end - 1
    )]
    CycleEndpointsNonAdjacent {
        /// Inclusive start of the offending cycle's segment.
        start: usize,
        /// Exclusive end of the offending cycle's segment.
        end: usize,
        /// Spatial axis on which the endpoint cubes differ by more than 1.
        axis: usize,
        /// Signed coordinate difference between the two cubes on that axis.
        delta: i64,
    },

    /// A `max_length` value below the structure's minimum, or above the
    /// streaming tile width.
    #[error("cycle length cap {max_length} is invalid")]
    InvalidMaxLength {
        /// The rejected length cap.
        max_length: usize,
    },

    /// A saved file's format version does not match this build's
    /// supported version.
    #[error("serialized format version {found} does not match the supported version {expected}")]
    FormatVersionMismatch {
        /// The version this build writes and expects.
        expected: u32,
        /// The version found in the file.
        found: u32,
    },

    /// An I/O operation on a saved-artifact file failed.
    #[cfg(feature = "serde")]
    #[error("file input/output failed: {source}")]
    Io {
        /// The underlying I/O failure.
        source: std::io::Error,
    },

    /// A saved artifact could not be decoded.
    #[cfg(feature = "serde")]
    #[error("saved data could not be decoded: {source}")]
    Deserialize {
        /// The underlying decode failure.
        source: rmp_serde::decode::Error,
    },
}

/// Convenience alias for [`std::result::Result`] with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "serde")]
impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Error::Io { source }
    }
}

// Encode failures for the crate's plain serializable structs can only arise
// from the underlying writer, so they are reported as I/O faults.
#[cfg(feature = "serde")]
impl From<rmp_serde::encode::Error> for Error {
    fn from(source: rmp_serde::encode::Error) -> Self {
        Error::Io {
            source: std::io::Error::other(source),
        }
    }
}

#[cfg(feature = "serde")]
impl From<rmp_serde::decode::Error> for Error {
    fn from(source: rmp_serde::decode::Error) -> Self {
        Error::Deserialize { source }
    }
}
