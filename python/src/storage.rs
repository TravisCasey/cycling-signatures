// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the `CycleStorage` type and its query results.

use std::path::PathBuf;

use cycling_signatures::{Component, Cycle, CycleStorage};
use pyo3::{exceptions::PyIndexError, prelude::*};

use crate::{
    convert::{parallel_backend, resolve_index, segment_from_py},
    embedded::PyEmbeddedTrajectory,
    errors::to_pyerr,
    homology::PyHomologyClass,
    signature::PyCyclingSignature,
};

/// Resolves a component id against a storage's component count, raising
/// `IndexError` if it falls outside `range(len(storage))`.
///
/// Unlike `resolve_index`, this does not accept Python-style negative
/// indexing: component ids are commonly computed rather than literals, so a
/// negative id here is more likely to be an arithmetic error than a deliberate
/// negative index.
fn checked_component_id(component_id: isize, component_count: usize) -> PyResult<usize> {
    if component_id < 0 || component_id as usize >= component_count {
        return Err(PyIndexError::new_err(format!(
            "component index {component_id} out of bounds for {component_count} components"
        )));
    }
    Ok(component_id as usize)
}

/// A single detected near-recurrent cycle in a trajectory.
///
/// A cycle is a segment of trajectory points whose first and last points are
/// within the adjacency threshold of each other. ``range`` gives the half-open
/// point-index span, ``birth`` gives the distance between the endpoints, and
/// ``length`` gives the number of points the cycle covers.
#[pyclass(name = "Cycle")]
pub(crate) struct PyCycle {
    pub(crate) inner: Cycle,
}

/// A connected group of detected cycles sharing the same homology class.
///
/// The cycles in one component all contribute the same class to cover homology.
/// ``class_id`` identifies which homology class that is within the parent
/// ``CycleStorage``, and ``coverage`` gives the overall point-index span that
/// the component's cycles collectively cover.
///
/// ``len`` gives the number of cycles, and indexing returns the ``Cycle`` at a
/// position (negative indices count from the end).
#[pyclass(name = "Component")]
pub(crate) struct PyComponent {
    pub(crate) inner: Component,
}

/// A stored set of detected cycles and their homology classes over a
/// trajectory segment.
///
/// Holds every near-recurrent cycle detected within a segment of an
/// ``EmbeddedTrajectory``, grouped into components that each carry a homology
/// class. The contents can be queried with ``signature``, ``component``, and
/// ``homology_class``, and can be saved and reloaded.
///
/// The ``fingerprint`` matches the fingerprint of the ``EmbeddedTrajectory`` it
/// was built from; save and load preserve this identity.
///
/// ``len`` gives the number of components, and indexing returns the
/// ``Component`` with that component id (negative indices count from the end).
#[pyclass(name = "CycleStorage")]
pub(crate) struct PyCycleStorage {
    pub(crate) inner: CycleStorage,
}

#[pymethods]
impl PyCycle {
    /// Returns the half-open point-index range ``(start, stop)`` covered by
    /// this cycle.
    ///
    /// Returns
    /// -------
    /// tuple of int
    ///     The ``(start, stop)`` point indices of the cycle.
    #[must_use]
    fn range(&self) -> (u32, u32) {
        let range = self.inner.range();
        (range.start, range.end)
    }

    /// Returns the metric distance between the cycle's first and last points.
    ///
    /// Returns
    /// -------
    /// float
    ///     The endpoint distance at which the cycle closes.
    #[must_use]
    fn birth(&self) -> f64 {
        self.inner.birth()
    }

    /// Returns the number of points covered by this cycle.
    ///
    /// Returns
    /// -------
    /// int
    ///     The point count of the cycle.
    #[must_use]
    fn length(&self) -> u32 {
        self.inner.length()
    }

    /// Returns a string representation of the cycle.
    fn __repr__(&self) -> String {
        let range = self.inner.range();
        format!(
            "Cycle(start={}, stop={}, birth={:?}, length={})",
            range.start,
            range.end,
            self.inner.birth(),
            self.inner.length()
        )
    }
}

#[pymethods]
impl PyComponent {
    /// Returns the index into the storage's deduplicated classes that gives
    /// this component's homology class.
    ///
    /// Several components may share one class, so distinct components can
    /// return the same ``class_id``. The id indexes
    /// ``CycleStorage.classes()``.
    ///
    /// Returns
    /// -------
    /// int
    ///     The class index within the parent storage.
    #[must_use]
    fn class_id(&self) -> u32 {
        self.inner.class_id()
    }

    /// Returns the half-open point-index range ``(start, stop)`` collectively
    /// covered by all cycles in this component.
    ///
    /// Returns
    /// -------
    /// tuple of int
    ///     The combined ``(start, stop)`` coverage of the component's cycles.
    #[must_use]
    fn coverage(&self) -> (u32, u32) {
        let range = self.inner.coverage();
        (range.start, range.end)
    }

    /// Returns all cycles belonging to this component.
    ///
    /// Returns
    /// -------
    /// list of ``Cycle``
    ///     Every cycle grouped into the component.
    #[must_use]
    fn cycles(&self) -> Vec<PyCycle> {
        self.inner
            .cycles()
            .iter()
            .map(|cycle| PyCycle {
                inner: cycle.clone(),
            })
            .collect()
    }

    /// Returns the number of cycles in this component.
    ///
    /// Returns
    /// -------
    /// int
    ///     The cycle count, also available as ``len(component)``.
    #[must_use]
    fn cycle_count(&self) -> usize {
        self.inner.cycle_count()
    }

    /// Returns the cycle in this component with the greatest point count.
    ///
    /// Returns
    /// -------
    /// ``Cycle``
    ///     The longest cycle in the component.
    #[must_use]
    fn longest_cycle(&self) -> PyCycle {
        PyCycle {
            inner: self.inner.longest_cycle().clone(),
        }
    }

    /// Returns the cycle in this component with the smallest point count.
    ///
    /// Returns
    /// -------
    /// ``Cycle``
    ///     The shortest cycle in the component.
    #[must_use]
    fn shortest_cycle(&self) -> PyCycle {
        PyCycle {
            inner: self.inner.shortest_cycle().clone(),
        }
    }

    /// Returns the number of cycles in this component.
    fn __len__(&self) -> usize {
        self.inner.cycle_count()
    }

    /// Returns the cycle at ``index`` in this component.
    fn __getitem__(&self, index: isize) -> PyResult<PyCycle> {
        let cycles = self.inner.cycles();
        let resolved = resolve_index(index, cycles.len()).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "cycle index {index} out of bounds for {} cycles",
                cycles.len()
            ))
        })?;
        Ok(PyCycle {
            inner: cycles[resolved].clone(),
        })
    }

    /// Returns a string representation of the component.
    fn __repr__(&self) -> String {
        let coverage = self.inner.coverage();
        format!(
            "Component(class_id={}, coverage=({}, {}), cycles={})",
            self.inner.class_id(),
            coverage.start,
            coverage.end,
            self.inner.cycle_count()
        )
    }
}

#[pymethods]
impl PyCycleStorage {
    /// Builds a ``CycleStorage`` from an embedded trajectory.
    ///
    /// Detects all near-recurrent cycles within ``segment`` of ``embedded``
    /// whose point count does not exceed ``max_length`` and whose endpoint
    /// distance does not exceed ``threshold``, and computes the homology
    /// class each one represents.
    ///
    /// By default the work is distributed across a thread pool; set the
    /// ``RAYON_NUM_THREADS`` environment variable to cap the number of
    /// threads it uses.
    ///
    /// Parameters
    /// ----------
    /// embedded : ``EmbeddedTrajectory``
    ///     The embedded trajectory to detect cycles in.
    /// segment : range or tuple of int
    ///     A half-open range of point indices, given as a Python ``range`` or
    ///     a ``(start, stop)`` integer tuple.
    /// max_length : int
    ///     The largest cycle point count to detect. Must be at least ``2``.
    /// threshold : float
    ///     The largest endpoint distance admitted as a cycle. Must be at
    ///     least the embedded trajectory's ``resolution`` and strictly below
    ///     ``1.0`` (the cube side).
    /// parallel : bool, optional
    ///     Whether to distribute the work across a thread pool. Defaults to
    ///     ``True``; pass ``False`` to run sequentially on the calling
    ///     thread.
    ///
    /// Returns
    /// -------
    /// ``CycleStorage``
    ///     The detected cycles and their homology classes.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range, if ``max_length`` is less
    ///     than ``2``, if ``threshold`` is below the embedded trajectory's
    ///     ``resolution`` or not below ``1.0``, or if a detected cycle's
    ///     endpoint points fall in non-adjacent cubes.
    /// ``IndexError``
    ///     If the segment indices are out of bounds.
    #[staticmethod]
    #[pyo3(signature = (embedded, segment, max_length, threshold, *, parallel = true))]
    fn build(
        py: Python<'_>,
        embedded: &Bound<'_, PyEmbeddedTrajectory>,
        segment: &Bound<'_, PyAny>,
        max_length: usize,
        threshold: f64,
        parallel: bool,
    ) -> PyResult<Self> {
        let range = segment_from_py(segment)?;
        let backend = parallel_backend(parallel);
        let embedded_ref = &embedded.borrow().inner;
        let inner = py
            .detach(move || {
                CycleStorage::build(embedded_ref, range, max_length, threshold, &backend)
            })
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Returns the half-open point-index range ``(start, stop)`` covered by
    /// this storage.
    ///
    /// Returns
    /// -------
    /// tuple of int
    ///     The ``(start, stop)`` point indices of the analyzed extent.
    #[must_use]
    fn extent(&self) -> (u32, u32) {
        let range = self.inner.extent();
        (range.start, range.end)
    }

    /// Returns a content fingerprint of this storage.
    ///
    /// The fingerprint matches the fingerprint of the ``EmbeddedTrajectory``
    /// used to build it. Save and load preserve this value.
    ///
    /// Returns
    /// -------
    /// int
    ///     A fingerprint identifying the storage contents.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Returns the adjacency threshold this storage was built with.
    ///
    /// Returns
    /// -------
    /// float
    ///     The threshold detection ran at.
    #[must_use]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    /// Returns the maximum cycle point count used when building this storage.
    ///
    /// Returns
    /// -------
    /// int
    ///     The cycle-length cap passed to ``build``.
    #[must_use]
    fn max_length(&self) -> u32 {
        self.inner.max_length()
    }

    /// Returns the number of cover generators, the ambient dimension shared by
    /// every homology class in this storage.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of homology generators in the cover.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Returns the deduplicated homology classes in this storage.
    ///
    /// Each distinct class appears once. A component's ``class_id`` indexes
    /// into this list, so several components may reference the same entry. The
    /// length is the number of distinct classes, not the number of components.
    ///
    /// Returns
    /// -------
    /// list of ``HomologyClass``
    ///     The distinct homology classes, indexed by ``Component.class_id``.
    #[must_use]
    fn classes(&self) -> Vec<PyHomologyClass> {
        self.inner
            .classes()
            .iter()
            .map(|vector| PyHomologyClass {
                inner: vector.clone(),
            })
            .collect()
    }

    /// Returns all detected components.
    ///
    /// A component's position in this list is its component id: the value that
    /// ``components_covering`` returns and that ``component`` and indexing
    /// accept. Components are ordered by their least cycle under
    /// ``(start, stop)``, with each component's cycles in that order.
    ///
    /// Returns
    /// -------
    /// list of ``Component``
    ///     Every detected component.
    #[must_use]
    fn components(&self) -> Vec<PyComponent> {
        self.inner
            .components()
            .iter()
            .map(|component| PyComponent {
                inner: component.clone(),
            })
            .collect()
    }

    /// Returns the component at ``component_id``.
    ///
    /// Parameters
    /// ----------
    /// component_id : int
    ///     A component index in ``range(len(storage))``.
    ///
    /// Returns
    /// -------
    /// ``Component``
    ///     The component at ``component_id``.
    ///
    /// Raises
    /// ------
    /// ``IndexError``
    ///     If ``component_id`` is out of bounds.
    fn component(&self, component_id: isize) -> PyResult<PyComponent> {
        let resolved = checked_component_id(component_id, self.inner.components().len())?;
        Ok(PyComponent {
            inner: self.inner.component(resolved).clone(),
        })
    }

    /// Returns the homology class of the component at ``component_id``.
    ///
    /// Parameters
    /// ----------
    /// component_id : int
    ///     A component index in ``range(len(storage))``.
    ///
    /// Returns
    /// -------
    /// ``HomologyClass``
    ///     The class of the component at ``component_id``.
    ///
    /// Raises
    /// ------
    /// ``IndexError``
    ///     If ``component_id`` is out of bounds.
    fn homology_class(&self, component_id: isize) -> PyResult<PyHomologyClass> {
        let resolved = checked_component_id(component_id, self.inner.components().len())?;
        Ok(PyHomologyClass {
            inner: self.inner.class(resolved).clone(),
        })
    }

    /// Returns the filtered cycling signature spanned by the classes of all
    /// components with a stored cycle fully contained in ``segment``.
    ///
    /// Each contributing component adds its class once, with birth equal to
    /// the minimum birth over its cycles contained in ``segment``, so a
    /// component with several contained cycles is not counted more than once.
    ///
    /// Parameters
    /// ----------
    /// segment : range or tuple of int
    ///     A half-open range of point indices, given as a Python ``range`` or
    ///     a ``(start, stop)`` integer tuple. Must fit inside the extent.
    ///
    /// Returns
    /// -------
    /// ``CyclingSignature``
    ///     The filtered cycling signature over the segment.
    ///
    /// Raises
    /// ------
    /// ``ValueError``
    ///     If ``segment`` is not a valid range.
    /// ``IndexError``
    ///     If ``segment`` falls outside the stored extent.
    fn signature(&self, segment: &Bound<'_, PyAny>) -> PyResult<PyCyclingSignature> {
        let range = segment_from_py(segment)?;
        let signature = self.inner.signature(range).map_err(to_pyerr)?;
        Ok(PyCyclingSignature { inner: signature })
    }

    /// Returns the ids of all components with a stored cycle covering the
    /// point index ``point``, in ascending order.
    ///
    /// Parameters
    /// ----------
    /// point : int
    ///     A point index to test for coverage. A negative value is always
    ///     outside the stored extent.
    ///
    /// Returns
    /// -------
    /// list of int
    ///     The covering component ids, ascending. Empty when ``point`` lies
    ///     outside the stored extent.
    #[must_use]
    fn components_covering(&self, point: isize) -> Vec<u32> {
        if point < 0 {
            return Vec::new();
        }
        self.inner.components_covering(point as usize)
    }

    /// Returns the number of detected components.
    fn __len__(&self) -> usize {
        self.inner.components().len()
    }

    /// Returns the component at ``index``.
    fn __getitem__(&self, index: isize) -> PyResult<PyComponent> {
        let resolved = resolve_index(index, self.inner.components().len()).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "component index {index} out of bounds for {} components",
                self.inner.components().len()
            ))
        })?;
        Ok(PyComponent {
            inner: self.inner.component(resolved).clone(),
        })
    }

    /// Returns a string representation of the storage.
    fn __repr__(&self) -> String {
        let extent = self.inner.extent();
        format!(
            "CycleStorage(extent=({}, {}), components={}, classes={}, threshold={:?}, \
             max_length={})",
            extent.start,
            extent.end,
            self.inner.components().len(),
            self.inner.classes().len(),
            self.inner.threshold(),
            self.inner.max_length()
        )
    }

    /// Saves this ``CycleStorage`` to a file at ``path``.
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
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    /// Loads a ``CycleStorage`` from a previously saved file at ``path``.
    ///
    /// Parameters
    /// ----------
    /// path : str or ``os.PathLike``
    ///     The source file path.
    ///
    /// Returns
    /// -------
    /// ``CycleStorage``
    ///     The reloaded storage.
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
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = CycleStorage::load(path).map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Registers the ``Cycle``, ``Component``, and ``CycleStorage`` classes on the
/// module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCycle>()?;
    module.add_class::<PyComponent>()?;
    module.add_class::<PyCycleStorage>()?;
    Ok(())
}
