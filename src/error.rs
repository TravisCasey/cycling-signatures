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
    /// A spanning vector passed to `CyclingSignature::new` did not have the
    /// expected length.
    #[error("spanning vector at index {index} has length {actual}, expected {expected}")]
    SignatureVectorLength {
        /// Position of the offending vector in the input slice.
        index: usize,
        /// Length of the offending vector.
        actual: usize,
        /// The expected length.
        expected: usize,
    },
}

/// Convenience alias for [`std::result::Result`] with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
