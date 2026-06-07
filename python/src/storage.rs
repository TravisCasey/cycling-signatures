// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Python wrappers for the `CycleStorage` type and its query results.

use std::path::PathBuf;

use cycling_signatures::{Component, Cycle, CycleStorage, ExecutionBackend};
use pyo3::{exceptions::PyIndexError, prelude::*};

use crate::{
    embedded::PyEmbeddedTrajectory,
    errors::to_pyerr,
    homology::{PyHomologyClass, PySubspace},
    segment::segment_from_py,
};

/// A single detected near-recurrent cycle in a trajectory.
///
/// A cycle is a segment of trajectory samples whose first and last points are
/// within the adjacency threshold of each other. `range` gives the half-open
/// sample-index span, `birth` gives the distance between the endpoints, and
/// `length` gives the number of sample points the cycle covers.
#[pyclass(name = "Cycle")]
pub(crate) struct PyCycle {
    pub(crate) inner: Cycle,
}

/// A connected group of detected cycles sharing the same homology class.
///
/// The cycles in one component all contribute the same class to cover homology.
/// `class_id` identifies which homology class that is within the parent
/// `CycleStorage`, and `coverage` gives the overall sample-index span that the
/// component's cycles collectively cover.
#[pyclass(name = "Component")]
pub(crate) struct PyComponent {
    pub(crate) inner: Component,
}

/// A stored set of detected cycles and their homology classes over a
/// trajectory segment.
///
/// Holds every near-recurrent cycle detected within a segment of an
/// `EmbeddedTrajectory`, grouped into components that each carry a homology
/// class. The contents can be queried with `signature`, `component`, and
/// `homology_class`, and can be saved and reloaded.
///
/// The `fingerprint` matches the fingerprint of the `EmbeddedTrajectory` it was
/// built from; save and load preserve this identity.
#[pyclass(name = "CycleStorage")]
pub(crate) struct PyCycleStorage {
    pub(crate) inner: CycleStorage,
}

#[pymethods]
impl PyCycle {
    /// Returns the half-open sample-index range `(start, stop)` covered by
    /// this cycle.
    #[must_use]
    fn range(&self) -> (u32, u32) {
        let range = self.inner.range();
        (range.start, range.end)
    }

    /// Returns the metric distance between the cycle's first and last points.
    #[must_use]
    fn birth(&self) -> f64 {
        self.inner.birth()
    }

    /// Returns the number of sample points covered by this cycle.
    #[must_use]
    fn length(&self) -> u32 {
        self.inner.length()
    }
}

#[pymethods]
impl PyComponent {
    /// Returns the index of the homology class assigned to this component
    /// within its parent `CycleStorage`.
    #[must_use]
    fn class_id(&self) -> u32 {
        self.inner.class_id()
    }

    /// Returns the half-open sample-index range `(start, stop)` collectively
    /// covered by all cycles in this component.
    #[must_use]
    fn coverage(&self) -> (u32, u32) {
        let range = self.inner.coverage();
        (range.start, range.end)
    }

    /// Returns all cycles belonging to this component.
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
    #[must_use]
    fn cycle_count(&self) -> usize {
        self.inner.cycle_count()
    }

    /// Returns the cycle in this component with the greatest point count.
    #[must_use]
    fn longest_cycle(&self) -> PyCycle {
        PyCycle {
            inner: self.inner.longest_cycle().clone(),
        }
    }

    /// Returns the cycle in this component with the smallest point count.
    #[must_use]
    fn shortest_cycle(&self) -> PyCycle {
        PyCycle {
            inner: self.inner.shortest_cycle().clone(),
        }
    }
}

#[pymethods]
impl PyCycleStorage {
    /// Builds a `CycleStorage` from an embedded trajectory.
    ///
    /// Detects all near-recurrent cycles within `segment` of `embedded` whose
    /// endpoint metric distance is at most `threshold` and whose point count
    /// does not exceed `max_length`, and computes the homology class each one
    /// represents.
    ///
    /// `segment` is a half-open range of sample indices and may be a Python
    /// `range` or a `(start, stop)` integer tuple.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if `segment` is not a valid range, if the segment
    /// indices are out of bounds, if `max_length` is less than 2, if
    /// `threshold` is below the embedded trajectory's `bound`, or if a
    /// detected cycle's consecutive or endpoint points fall in non-adjacent
    /// cubes.
    #[staticmethod]
    fn build(
        py: Python<'_>,
        embedded: &Bound<'_, PyEmbeddedTrajectory>,
        segment: &Bound<'_, PyAny>,
        threshold: f64,
        max_length: usize,
    ) -> PyResult<Self> {
        let range = segment_from_py(segment)?;
        let embedded_ref = &embedded.borrow().inner;
        let inner = py
            .detach(move || {
                CycleStorage::build(
                    embedded_ref,
                    range,
                    threshold,
                    max_length,
                    &ExecutionBackend::Rayon,
                )
            })
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Returns the half-open sample-index range `(start, stop)` covered by
    /// this storage.
    #[must_use]
    fn extent(&self) -> (u32, u32) {
        let range = self.inner.extent();
        (range.start, range.end)
    }

    /// Returns a content fingerprint of this storage.
    ///
    /// The fingerprint matches the `fingerprint` of the `EmbeddedTrajectory`
    /// used to build it. Save and load preserve this value.
    #[must_use]
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Returns the adjacency threshold used when building this storage.
    #[must_use]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    /// Returns the maximum cycle point count used when building this storage.
    #[must_use]
    fn max_length(&self) -> u32 {
        self.inner.max_length()
    }

    /// Returns the number of cover generators, the ambient dimension shared by
    /// every homology class in this storage.
    #[must_use]
    fn num_generators(&self) -> usize {
        self.inner.num_generators()
    }

    /// Returns all homology classes stored, one per detected component.
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

    /// Returns the component at `component_id`.
    ///
    /// # Errors
    ///
    /// Raises `IndexError` if `component_id` is out of bounds.
    fn component(&self, component_id: usize) -> PyResult<PyComponent> {
        if component_id >= self.inner.components().len() {
            return Err(PyIndexError::new_err("component index out of bounds"));
        }
        Ok(PyComponent {
            inner: self.inner.component(component_id).clone(),
        })
    }

    /// Returns the homology class of the component at `component_id`.
    ///
    /// # Errors
    ///
    /// Raises `IndexError` if `component_id` is out of bounds.
    fn homology_class(&self, component_id: usize) -> PyResult<PyHomologyClass> {
        if component_id >= self.inner.components().len() {
            return Err(PyIndexError::new_err("component index out of bounds"));
        }
        Ok(PyHomologyClass {
            inner: self.inner.class(component_id).clone(),
        })
    }

    /// Returns the subspace spanned by the homology classes of all cycles
    /// within `segment`.
    ///
    /// `segment` is a half-open range of sample indices and may be a Python
    /// `range` or a `(start, stop)` integer tuple.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if `segment` is not a valid range or if it falls
    /// outside the stored extent.
    fn signature(&self, segment: &Bound<'_, PyAny>) -> PyResult<PySubspace> {
        let range = segment_from_py(segment)?;
        let subspace = self.inner.signature(range).map_err(to_pyerr)?;
        Ok(PySubspace { inner: subspace })
    }

    /// Returns the indices of all components that have a stored cycle covering
    /// the sample index `point`, in ascending order.
    ///
    /// The result is empty when `point` lies outside the stored extent.
    #[must_use]
    fn components_covering(&self, point: usize) -> Vec<u32> {
        self.inner.components_covering(point)
    }

    /// Saves this `CycleStorage` to a file at `path`.
    ///
    /// # Errors
    ///
    /// Raises `OSError` if the file cannot be written.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    /// Loads a `CycleStorage` from a previously saved file at `path`.
    ///
    /// # Errors
    ///
    /// Raises `OSError` if the file cannot be read. Raises
    /// `FormatVersionMismatchError` if the file was written by an incompatible
    /// version of the library.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = CycleStorage::load(path).map_err(to_pyerr)?;
        Ok(Self { inner })
    }
}

/// Registers the `Cycle`, `Component`, and `CycleStorage` classes on the
/// module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCycle>()?;
    module.add_class::<PyComponent>()?;
    module.add_class::<PyCycleStorage>()?;
    Ok(())
}
