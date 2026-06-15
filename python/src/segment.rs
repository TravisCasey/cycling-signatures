// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Shared helpers for converting Python index and segment arguments.

use std::ops::Range;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyRange, PyRangeMethods},
};

/// Resolves a Python sequence index against a container length.
///
/// Negative indices count from the end, as Python sequences do. Returns `None`
/// when the index is out of range.
pub(crate) fn resolve_index(index: isize, length: usize) -> Option<usize> {
    let resolved = if index < 0 {
        index.checked_add(length as isize)?
    } else {
        index
    };
    if resolved < 0 || resolved as usize >= length {
        return None;
    }
    Some(resolved as usize)
}

/// Converts a Python segment argument to a half-open `Range<usize>`.
///
/// Accepts either:
///
/// - A Python `range` with `step == 1`, e.g. `range(2, 10)`.
/// - A `(start, stop)` tuple of integers, e.g. `(2, 10)`.
///
/// Both forms describe the half-open range `start..stop` of sample indices.
///
/// # Errors
///
/// Raises `ValueError` if:
///
/// - The argument is not a `range` or a two-element tuple.
/// - A `range` has a step other than `1`.
/// - Either bound is negative.
/// - `start > stop`.
pub(crate) fn segment_from_py(object: &Bound<'_, PyAny>) -> PyResult<Range<usize>> {
    if let Ok(python_range) = object.cast::<PyRange>() {
        let start = python_range.start()?;
        let stop = python_range.stop()?;
        let increment = python_range.step()?;
        if increment != 1 {
            return Err(PyValueError::new_err("segment range must have step 1"));
        }
        return validate_bounds(start, stop);
    }
    if let Ok((start, stop)) = object.extract::<(isize, isize)>() {
        return validate_bounds(start, stop);
    }
    Err(PyValueError::new_err(
        "segment must be a range or a (start, stop) tuple",
    ))
}

/// Validates that `start` and `stop` are non-negative and ordered, then
/// converts them to a `Range<usize>`.
fn validate_bounds(start: isize, stop: isize) -> PyResult<Range<usize>> {
    if start < 0 || stop < 0 {
        return Err(PyValueError::new_err("segment bounds must be non-negative"));
    }
    let start = start as usize;
    let stop = stop as usize;
    if start > stop {
        return Err(PyValueError::new_err("segment start must not exceed stop"));
    }
    Ok(start..stop)
}
