// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`CubicalCover`]: the cube set, the discrete Morse
//! reduction's critical cells, and the class of each critical 1-cell.

use std::cmp::Ordering;

use chomp3rs::Cube;
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize, de::Error as DeserializeError};

use super::{CubicalCover, classifier::EdgeClassifier};
use crate::{error::Error, f2_vector::F2Vector};

#[derive(Serialize, Deserialize)]
struct CoverData {
    #[serde(with = "crate::serialization::npy_field")]
    cubes: Array2<i64>,
    num_generators: usize,
    critical_cells: Vec<Cube>,
    classes: Vec<(u32, F2Vector)>,
}

/// Checks that every cube coordinate is one the cubical-homology backend
/// accepts.
///
/// # Errors
///
/// Returns a description of the first offending coordinate, the one
/// [`Error::CubeCoordinateOutOfRange`] carries, so a decoded cover and an
/// explicitly built one report the same condition in the same words.
fn check_cube_range(cubes: ArrayView2<'_, i64>) -> Result<(), String> {
    for (row, cube) in cubes.outer_iter().enumerate() {
        for (axis, &coordinate) in cube.iter().enumerate() {
            if coordinate < i64::from(i32::MIN) || coordinate > i64::from(i32::MAX) - 1 {
                return Err(Error::CubeCoordinateOutOfRange {
                    row,
                    axis,
                    coordinate,
                }
                .to_string());
            }
        }
    }
    Ok(())
}

/// Checks that the cube rows are strictly increasing in lexicographic order.
///
/// That ordering is the canonical form a cover holds, and cube lookup is a
/// binary search over it: an unordered or duplicated row set answers lookups
/// wrongly rather than loudly.
///
/// # Errors
///
/// Returns a description of the first offending pair of rows.
fn check_lexicographic_order(cubes: ArrayView2<'_, i64>) -> Result<(), String> {
    for row in 1..cubes.nrows() {
        let previous = cubes.row(row - 1);
        let current = cubes.row(row);
        if previous.iter().cmp(current.iter()) != Ordering::Less {
            return Err(format!(
                "cube rows {} and {row} are not in strictly increasing lexicographic order",
                row - 1
            ));
        }
    }
    Ok(())
}

/// Checks that the class entries are strictly increasing by Morse cell index.
///
/// The strict order is the canonical form the file promises; it also rules
/// out duplicate indices, which would silently shadow one another when the
/// classes are placed back into critical-cell order.
///
/// # Errors
///
/// Returns a description of the first offending pair of entries.
fn check_class_indices_increasing(classes: &[(u32, F2Vector)]) -> Result<(), String> {
    for entry in 1..classes.len() {
        if classes[entry - 1].0 >= classes[entry].0 {
            return Err(format!(
                "class entries {} and {entry} are not in strictly increasing index order",
                entry - 1
            ));
        }
    }
    Ok(())
}

/// Checks that every class entry names a Morse cell index within
/// `critical_cell_count`.
///
/// # Errors
///
/// Returns a description of the first offending entry.
fn check_class_indices_in_range(
    classes: &[(u32, F2Vector)],
    critical_cell_count: usize,
) -> Result<(), String> {
    for (entry, &(index, _)) in classes.iter().enumerate() {
        if index as usize >= critical_cell_count {
            return Err(format!(
                "class entry {entry} names Morse cell index {index}, out of range for \
                 {critical_cell_count} critical cells"
            ));
        }
    }
    Ok(())
}

/// Checks that every class entry names a critical cell of dimension 1.
///
/// A class is meaningful only against a critical 1-cell; an entry naming
/// another dimension could never be produced by a valid classifier.
///
/// # Errors
///
/// Returns a description of the first offending entry. Assumes every index
/// already lies within `critical_cells`.
fn check_class_cells_are_edges(
    classes: &[(u32, F2Vector)],
    critical_cells: &[Cube],
) -> Result<(), String> {
    for (entry, &(index, _)) in classes.iter().enumerate() {
        let dimension = critical_cells[index as usize].dimension();
        if dimension != 1 {
            return Err(format!(
                "class entry {entry} names critical cell {index} of dimension {dimension}, \
                 expected 1"
            ));
        }
    }
    Ok(())
}

/// Checks that every class vector has length `num_generators`.
///
/// Every class computed against one cover shares that length; a vector of
/// another length would fail arithmetic far from the load that admitted it.
///
/// # Errors
///
/// Returns a description of the first offending entry.
fn check_class_lengths(classes: &[(u32, F2Vector)], num_generators: usize) -> Result<(), String> {
    for (entry, (_, class)) in classes.iter().enumerate() {
        if class.len() != num_generators {
            return Err(format!(
                "class entry {entry} carries a class vector of length {}, expected the generator \
                 count {num_generators}",
                class.len()
            ));
        }
    }
    Ok(())
}

/// Checks that every dimension-1 critical cell has a stored class entry.
///
/// # Errors
///
/// Returns a description of the first critical 1-cell missing its row.
fn check_classes_complete(
    classes: &[(u32, F2Vector)],
    critical_cells: &[Cube],
) -> Result<(), String> {
    let mut listed = classes.iter();
    let mut next = listed.next();
    for (index, cell) in critical_cells.iter().enumerate() {
        if cell.dimension() != 1 {
            continue;
        }
        match next {
            Some(&(listed_index, _)) if listed_index as usize == index => {
                next = listed.next();
            },
            _ => {
                return Err(format!(
                    "critical cell {index} has dimension 1 but no stored class row"
                ));
            },
        }
    }
    Ok(())
}

impl Serialize for CubicalCover {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let critical_cells = self.classifier.critical_cells().to_vec();
        let classes: Vec<(u32, F2Vector)> = self
            .classifier
            .classes()
            .iter()
            .enumerate()
            .filter_map(|(index, class)| class.as_ref().map(|class| (index as u32, class.clone())))
            .collect();
        CoverData {
            cubes: self.cubes.clone(),
            num_generators: self.classifier.num_generators(),
            critical_cells,
            classes,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CubicalCover {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = CoverData::deserialize(deserializer)?;
        if data.cubes.nrows() == 0 {
            return Err(D::Error::custom("cubical cover requires at least one cube"));
        }
        check_cube_range(data.cubes.view()).map_err(D::Error::custom)?;
        check_lexicographic_order(data.cubes.view()).map_err(D::Error::custom)?;
        check_class_indices_increasing(&data.classes).map_err(D::Error::custom)?;
        check_class_indices_in_range(&data.classes, data.critical_cells.len())
            .map_err(D::Error::custom)?;
        check_class_cells_are_edges(&data.classes, &data.critical_cells)
            .map_err(D::Error::custom)?;
        check_class_lengths(&data.classes, data.num_generators).map_err(D::Error::custom)?;
        check_classes_complete(&data.classes, &data.critical_cells).map_err(D::Error::custom)?;

        let mut classes: Vec<Option<F2Vector>> = vec![None; data.critical_cells.len()];
        for (index, class) in data.classes {
            classes[index as usize] = Some(class);
        }

        let classifier = EdgeClassifier::from_parts(
            &data.cubes,
            data.critical_cells,
            classes,
            data.num_generators,
        );

        Ok(Self {
            cubes: data.cubes,
            classifier,
        })
    }
}

#[cfg(test)]
mod tests {
    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::{CoverData, CubicalCover};
    use crate::{
        error::Error,
        f2_vector::F2Vector,
        serialization::{load_from_reader, save_to_writer},
    };

    /// Encodes a cover payload and decodes it back.
    fn round_trip(data: CoverData) -> Result<CubicalCover, Error> {
        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &data).unwrap();
        load_from_reader(&buffer[..])
    }

    /// A class-free payload over `cubes` with no critical cells and no
    /// generators.
    fn payload(cubes: Array2<i64>) -> CoverData {
        CoverData {
            cubes,
            num_generators: 0,
            critical_cells: Vec::new(),
            classes: Vec::new(),
        }
    }

    /// The twelve cubes on the boundary of a `4x4` grid, leaving a `2x2` hole
    /// in the middle, matching `cover.rs`'s ring fixture. A cover of these
    /// cubes and no others has exactly one generator.
    fn ring_cubes() -> Array2<i64> {
        array![
            [0_i64, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [2, 3],
            [1, 3],
            [0, 3],
            [0, 2],
            [0, 1],
        ]
    }

    /// Two copies of [`ring_cubes`] separated along axis 0 so they share no
    /// cube and stay topologically independent: a cover of these cubes has
    /// two generators, and correspondingly at least two critical 1-cells.
    fn two_ring_cubes() -> Array2<i64> {
        let first = ring_cubes();
        let mut rows: Vec<Vec<i64>> = first.outer_iter().map(|row| row.to_vec()).collect();
        for row in first.outer_iter() {
            rows.push(vec![row[0] + 10, row[1]]);
        }
        rows.sort();
        let dimension = first.ncols();
        let flat: Vec<i64> = rows.iter().flatten().copied().collect();
        Array2::from_shape_vec((rows.len(), dimension), flat).unwrap()
    }

    /// A genuine payload for a cover built from `cubes`.
    ///
    /// Built through [`CubicalCover::from_cubes`], so the critical cells are
    /// exactly what the discrete Morse matching computes for these cubes: the
    /// payload round-trips successfully unmodified, and tests corrupt exactly
    /// one field to isolate the check it exercises.
    fn cover_parts(cubes: Array2<i64>) -> CoverData {
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        let critical_cells = cover.classifier.critical_cells().to_vec();
        let classes: Vec<(u32, F2Vector)> = cover
            .classifier
            .classes()
            .iter()
            .enumerate()
            .filter_map(|(index, class)| class.as_ref().map(|class| (index as u32, class.clone())))
            .collect();
        CoverData {
            cubes: cover.cubes,
            num_generators: cover.classifier.num_generators(),
            critical_cells,
            classes,
        }
    }

    #[test]
    fn deserialize_rejects_out_of_range_cube_coordinate() {
        // The largest valid coordinate is i32::MAX - 1; i32::MAX is one past it.
        let cubes = array![[0_i64, 0], [1, i64::from(i32::MAX)]];
        assert!(matches!(
            round_trip(payload(cubes)).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_cubes_out_of_lexicographic_order() {
        let cubes = array![[1_i64, 0], [0, 0]];
        assert!(matches!(
            round_trip(payload(cubes)).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn generator_count_survives_when_every_stored_class_is_zero() {
        // The generator count is carried explicitly rather than inferred from
        // the stored classes, so a table of all-zero vectors cannot deflate
        // it: every real class row is replaced with a zero vector of a
        // different, explicitly stated length.
        let mut data = cover_parts(ring_cubes());
        for (_, class) in &mut data.classes {
            *class = F2Vector::zeros(2);
        }
        data.num_generators = 2;

        let cover = round_trip(data).unwrap();
        assert_eq!(cover.num_generators(), 2);
    }

    #[test]
    fn deserialize_rejects_indices_out_of_order() {
        // `two_ring_cubes` has two generators, so its class table has at
        // least two entries to reorder.
        let mut data = cover_parts(two_ring_cubes());
        assert!(
            data.classes.len() >= 2,
            "fixture needs at least two critical 1-cells to exercise ordering"
        );
        data.classes.swap(0, 1);

        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_index_out_of_critical_cell_range() {
        let mut data = cover_parts(ring_cubes());
        let out_of_range = u32::try_from(data.critical_cells.len()).unwrap();
        data.classes.last_mut().unwrap().0 = out_of_range;

        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_class_naming_a_non_edge_critical_cell() {
        // A class entry naming a critical cell of dimension other than 1
        // could never be produced by a valid classifier.
        let mut data = cover_parts(ring_cubes());
        let (non_edge_index, _) = data
            .critical_cells
            .iter()
            .enumerate()
            .find(|(_, cell)| cell.dimension() != 1)
            .expect("fixture has at least one non-edge critical cell");
        data.classes = vec![(
            u32::try_from(non_edge_index).unwrap(),
            F2Vector::zeros(data.num_generators),
        )];

        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_class_vector_of_wrong_length() {
        let mut data = cover_parts(ring_cubes());
        let num_generators = data.num_generators;
        data.classes.first_mut().unwrap().1 = F2Vector::zeros(num_generators + 1);

        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_an_edge_critical_cell_missing_its_class() {
        // Dropping one row would otherwise surface only at walk time, as a
        // panic deep inside the classifier.
        let mut data = cover_parts(ring_cubes());
        data.classes.remove(0);

        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }
}
