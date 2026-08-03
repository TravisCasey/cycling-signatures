// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the `CubicalCover` type.

use std::sync::Arc;

use cycling_signatures::{CubicalCover, ExecutionBackend};
use pyo3::prelude::*;

use crate::{errors::to_pyerr, trajectory::PyTrajectory};

/// The cubical cover of the integer cubes a trajectory visits, with its
/// cohomology generators computed over ``F_2``.
///
/// Build the cover from the densest trajectory in hand: a cover built from a
/// thinned trajectory is a coarser model of the same curve and can report
/// first-homology classes the curve does not have. A cover is reusable, so
/// one built once from a dense trajectory can back several
/// ``EmbeddedTrajectory`` instances, each over a different thinned copy of
/// it.
///
/// Parameters
/// ----------
/// trajectory : ``Trajectory``
///     The trajectory whose visited cubes define the cover.
///
/// Raises
/// ------
/// ``ValueError``
///     If consecutive trajectory points fall in non-adjacent cubes (resample
///     at a smaller spacing), if the trajectory's points have zero columns,
///     or if a point's cube coordinate falls outside the supported integer
///     range.
#[pyclass(name = "CubicalCover")]
pub(crate) struct PyCubicalCover {
    pub(crate) inner: Arc<CubicalCover>,
}

#[pymethods]
impl PyCubicalCover {
    /// Builds the cover of exactly the integer cubes ``trajectory`` visits,
    /// and computes its cohomology generators.
    #[new]
    fn new(py: Python<'_>, trajectory: &Bound<'_, PyTrajectory>) -> PyResult<Self> {
        let trajectory = Arc::clone(&trajectory.borrow().inner);
        let inner = py
            .detach(move || CubicalCover::build(&trajectory, &ExecutionBackend::Rayon))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Returns a content fingerprint of the cover.
    ///
    /// Two covers built from the same visited cube set have the same
    /// fingerprint, regardless of the trajectory that produced it or the
    /// generator basis computed for it.
    ///
    /// Returns
    /// -------
    /// int
    ///     A fingerprint identifying the cover's cube set.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }
}

/// Registers the ``CubicalCover`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCubicalCover>()?;
    Ok(())
}
