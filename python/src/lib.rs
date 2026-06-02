// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python bindings for the cycling-signatures core pipeline.

mod errors;
mod interpolation;
mod metric;
mod trajectory;

use pyo3::prelude::*;

/// The compiled extension module, re-exported to Python as
/// `cycling_signatures`.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    metric::register(module)?;
    interpolation::register(module)?;
    trajectory::register(module)?;
    Ok(())
}
