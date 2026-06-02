// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Translation of core errors into Python exceptions.

use cycling_signatures::Error;
use pyo3::{
    create_exception,
    exceptions::{PyException, PyOSError, PyValueError},
    prelude::*,
};

create_exception!(_core, FormatVersionMismatchError, PyException);

/// Maps a core error onto the appropriate Python exception: validation and
/// input failures become `ValueError`, storage and input/output failures
/// become `OSError`, and a format-version mismatch becomes custom
/// `FormatVersionMismatchError`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn to_pyerr(error: Error) -> PyErr {
    let message = error.to_string();
    match error {
        Error::FormatVersionMismatch { .. } => FormatVersionMismatchError::new_err(message),
        Error::Storage { .. } => PyOSError::new_err(message),
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
