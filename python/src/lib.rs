// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python bindings for the cycling-signatures core pipeline.

use pyo3::prelude::*;

/// The compiled extension module, re-exported to Python as
/// `cycling_signatures`.
// The `#[pymodule]` macro fixes the `PyResult<()>` return type.
#[allow(clippy::unnecessary_wraps)]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}
