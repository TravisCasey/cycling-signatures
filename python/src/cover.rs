// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrapper for the `CubicalCover` type.

use std::{path::PathBuf, sync::Arc};

use cycling_signatures::CubicalCover;
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::{convert::parallel_backend, errors::to_pyerr, trajectory::PyTrajectory};

/// The cubical cover of the integer cubes a trajectory visits, together with
/// its cover generators: the cover's ``F_2`` cohomology generators, one
/// coordinate per independent loop type in the cover.
///
/// Build the cover from the densest trajectory in hand: a cover built from a
/// thinned trajectory is a coarser model of the same curve and can report
/// first-homology classes the curve does not have. Where consecutive points
/// land in non-adjacent cubes, resample the curve at a smaller spacing. A
/// cover is reusable, so one built once from a dense trajectory can back
/// several ``EmbeddedTrajectory`` instances, each over a different thinned
/// copy of the trajectory.
///
/// The generator basis is not stable across builds. Two builds over the same
/// cubes can compute their homology classes in different generator bases, so a
/// class vector, ``Subspace`` equality, and ``Subspace.contains`` are
/// meaningful only within one basis. One basis means one build, or every
/// process that loads one saved cover file: ``save`` writes the cover's
/// homology data in the exact basis the build computed and ``load`` restores
/// it, while rebuilding from the same cubes reproduces the ``fingerprint`` but
/// not necessarily the basis. Across independently built covers, compare only
/// what a change of basis cannot alter: ranks and counts, whether a class
/// ``is_zero``, and which components share a class.
///
/// By default the work is distributed across a thread pool; set the
/// ``RAYON_NUM_THREADS`` environment variable to cap the number of threads it
/// uses.
///
/// Parameters
/// ----------
/// trajectory : ``Trajectory``
///     The trajectory whose visited cubes define the cover.
/// parallel : bool, optional
///     Whether to distribute the work across a thread pool. Defaults to
///     ``True``; pass ``False`` to run sequentially on the calling thread.
///
/// Raises
/// ------
/// ``ValueError``
///     If consecutive trajectory points fall in non-adjacent cubes, if the
///     trajectory's points have zero columns, or if a point's cube coordinate
///     falls outside the supported integer range.
#[pyclass(name = "CubicalCover")]
pub(crate) struct PyCubicalCover {
    pub(crate) inner: Arc<CubicalCover>,
}

#[pymethods]
impl PyCubicalCover {
    /// Builds the cover of exactly the integer cubes ``trajectory`` visits,
    /// and computes its cover generators.
    #[new]
    #[pyo3(signature = (trajectory, *, parallel = true))]
    fn new(py: Python<'_>, trajectory: &Bound<'_, PyTrajectory>, parallel: bool) -> PyResult<Self> {
        let backend = parallel_backend(parallel);
        let trajectory = Arc::clone(&trajectory.borrow().inner);
        let inner = py
            .detach(move || CubicalCover::build(&trajectory, &backend))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Returns a content fingerprint of the cover's cube set.
    ///
    /// Two covers built from the same visited cubes have the same fingerprint,
    /// regardless of the trajectory that produced them or the generator basis
    /// computed for them.
    ///
    /// Returns
    /// -------
    /// int
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Returns the number of cubes in the cover.
    fn __len__(&self) -> usize {
        self.inner.cubes().nrows()
    }

    /// Returns the spatial dimension of each cube.
    ///
    /// Must equal the embedded trajectory's own ``dimension`` when the two
    /// are paired in an ``EmbeddedTrajectory``.
    ///
    /// Returns
    /// -------
    /// int
    ///     The ambient coordinate dimension, the column count of ``cubes``.
    #[must_use]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Returns the cover's cubes as a two-dimensional array.
    ///
    /// Each row holds the integer coordinates of one cube, the component-wise
    /// floor of the trajectory points that landed in it. Rows are deduplicated
    /// and in lexicographic order.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A two-dimensional array whose rows are the cover's cubes.
    #[must_use]
    fn cubes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        self.inner.cubes().to_pyarray(py)
    }

    /// Returns the number of cover generators.
    ///
    /// Returns
    /// -------
    /// int
    ///     One per independent loop type in the cover, and the length of every
    ///     class vector computed against it.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Saves the cover to a file at ``path``, including its cube set and its
    /// exact generator basis.
    ///
    /// This is how a basis is pinned for later runs: a cover reloaded with
    /// ``load`` carries the basis saved here.
    ///
    /// Parameters
    /// ----------
    /// path : str or ``os.PathLike``
    ///     The destination file path.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If the file cannot be written.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let cover = Arc::clone(&self.inner);
        py.detach(move || cover.save(path)).map_err(to_pyerr)
    }

    /// Loads a cover from the file at ``path``.
    ///
    /// Parameters
    /// ----------
    /// path : str or ``os.PathLike``
    ///     The source file path.
    ///
    /// Returns
    /// -------
    /// ``CubicalCover``
    ///     The reloaded cover, carrying the exact generator basis it was
    ///     saved with.
    ///
    /// Raises
    /// ------
    /// ``OSError``
    ///     If the file cannot be read.
    /// ``FormatVersionMismatchError``
    ///     If the file was written by an incompatible version of the library.
    /// ``ValueError``
    ///     If the stored data cannot be decoded.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(move || CubicalCover::load(path))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// Registers the ``CubicalCover`` class on the module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCubicalCover>()?;
    Ok(())
}
