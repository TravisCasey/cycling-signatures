// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Translation of core errors into Python exceptions.

use cycling_signatures::Error;
use pyo3::{
    create_exception,
    exceptions::{PyException, PyIndexError, PyOSError, PyValueError},
    prelude::*,
};

create_exception!(
    _core,
    FormatVersionMismatchError,
    PyException,
    "Raised when a saved file was written by an incompatible version of the library. Regenerate \
     the file from its source data with the current version rather than loading it."
);

/// Maps a core error onto the appropriate Python exception: file input/output
/// failures become `OSError`, a format-version mismatch becomes custom
/// `FormatVersionMismatchError`, an out-of-range segment becomes `IndexError`,
/// and every other validation, input, or decode failure becomes `ValueError`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::FormatVersionMismatch { .. } => FormatVersionMismatchError::new_err(message),
        Error::Io { .. } => PyOSError::new_err(message),
        Error::SegmentOutOfBounds { .. } => PyIndexError::new_err(message),
        // ValueError is the default mapping. Error is #[non_exhaustive], so
        // this wildcard arm is required; a new variant needing a different
        // mapping goes above it.
        _ => PyValueError::new_err(message),
    }
}

/// Registers the exception type on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "FormatVersionMismatchError",
        module.py().get_type::<FormatVersionMismatchError>(),
    )?;
    Ok(())
}
