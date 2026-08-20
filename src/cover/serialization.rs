// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`CubicalCover`]: the cube set, the cohomology generator
//! count, and the class of every recognized edge.

use std::cmp::Ordering;

use chomp3rs::Cube;
use ndarray::{Array2, ArrayView2};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize, de::Error as DeserializeError};

use super::CubicalCover;
use crate::{error::Error, f2_vector::F2Vector};

#[derive(Serialize, Deserialize)]
struct CoverData {
    #[serde(with = "crate::serialization::npy_field")]
    cubes: Array2<i64>,
    num_generators: usize,
    edges: Vec<(Cube, F2Vector)>,
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

/// Checks that every edge entry is a 1-cube edge in the cubes' ambient
/// dimension.
///
/// The decoded map answers edge lookups by exact cube equality, so an entry
/// of another dimension or shape could never be looked up and would only
/// misstate what the cover recognizes.
///
/// # Errors
///
/// Returns a description of the first offending entry.
fn check_edge_shapes(edges: &[(Cube, F2Vector)], dimension: usize) -> Result<(), String> {
    for (entry, (edge, _)) in edges.iter().enumerate() {
        if edge.dimension() != 1 {
            return Err(format!("edge entry {entry} is not a 1-cube edge"));
        }
        if edge.ambient_dimension() as usize != dimension {
            return Err(format!(
                "edge entry {entry} has ambient dimension {}, expected the cubes' dimension \
                 {dimension}",
                edge.ambient_dimension()
            ));
        }
    }
    Ok(())
}

/// Checks that every edge's class vector has length `num_generators`.
///
/// Every class computed against one cover shares that length; a vector of
/// another length would fail arithmetic far from the load that admitted it.
///
/// # Errors
///
/// Returns a description of the first offending entry.
fn check_class_lengths(edges: &[(Cube, F2Vector)], num_generators: usize) -> Result<(), String> {
    for (entry, (_, class)) in edges.iter().enumerate() {
        if class.len() != num_generators {
            return Err(format!(
                "edge entry {entry} carries a class vector of length {}, expected the generator \
                 count {num_generators}",
                class.len()
            ));
        }
    }
    Ok(())
}

/// Checks that the edge entries are strictly increasing by base coordinates
/// then extent.
///
/// The strict order is the canonical form the file promises; it also rules
/// out duplicate edges, which would silently shadow one another in the
/// decoded map.
///
/// # Errors
///
/// Returns a description of the first offending pair of entries.
fn check_edge_order(edges: &[(Cube, F2Vector)]) -> Result<(), String> {
    for entry in 1..edges.len() {
        let (previous, _) = &edges[entry - 1];
        let (current, _) = &edges[entry];
        if (previous.base().as_slice(), previous.extent())
            .cmp(&(current.base().as_slice(), current.extent()))
            != Ordering::Less
        {
            return Err(format!(
                "edge entries {} and {entry} are not in strictly increasing order",
                entry - 1
            ));
        }
    }
    Ok(())
}

impl Serialize for CubicalCover {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut edges: Vec<(Cube, F2Vector)> = self
            .edge_classes
            .iter()
            .map(|(edge, class)| (edge.clone(), class.clone()))
            .collect();
        edges.sort_unstable_by(|(left, _), (right, _)| {
            (left.base().as_slice(), left.extent()).cmp(&(right.base().as_slice(), right.extent()))
        });
        CoverData {
            cubes: self.cubes.clone(),
            num_generators: self.num_generators,
            edges,
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
        check_edge_shapes(&data.edges, data.cubes.ncols()).map_err(D::Error::custom)?;
        check_class_lengths(&data.edges, data.num_generators).map_err(D::Error::custom)?;
        check_edge_order(&data.edges).map_err(D::Error::custom)?;
        let edge_classes: FxHashMap<Cube, F2Vector> = data.edges.into_iter().collect();
        Ok(Self {
            cubes: data.cubes,
            num_generators: data.num_generators,
            edge_classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use chomp3rs::{Cube, Orthant};
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

    /// An edge-free payload over `cubes` with no generators.
    fn payload(cubes: Array2<i64>) -> CoverData {
        CoverData {
            cubes,
            num_generators: 0,
            edges: Vec::new(),
        }
    }

    /// A 1-cube edge at integer `base` along `axis`.
    fn edge(base: [i32; 2], axis: u32) -> Cube {
        Cube::from_extent(Orthant::from(base), 1_u32 << axis)
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
        // the stored classes, so a map of all-zero vectors cannot deflate it.
        let data = CoverData {
            cubes: array![[0_i64, 0]],
            num_generators: 2,
            edges: vec![
                (edge([0, 0], 0), F2Vector::zeros(2)),
                (edge([0, 0], 1), F2Vector::zeros(2)),
            ],
        };
        let cover = round_trip(data).unwrap();
        assert_eq!(cover.num_generators(), 2);
    }

    #[test]
    fn deserialize_rejects_non_edge_entry() {
        // A 2-cube (two extent bits) is not an edge and could never be looked
        // up by a walk.
        let data = CoverData {
            cubes: array![[0_i64, 0]],
            num_generators: 1,
            edges: vec![(
                Cube::from_extent(Orthant::from([0_i32, 0]), 0b11),
                F2Vector::zeros(1),
            )],
        };
        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_class_vector_of_wrong_length() {
        let data = CoverData {
            cubes: array![[0_i64, 0]],
            num_generators: 2,
            edges: vec![(edge([0, 0], 0), F2Vector::zeros(3))],
        };
        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_edges_out_of_order() {
        let data = CoverData {
            cubes: array![[0_i64, 0]],
            num_generators: 1,
            edges: vec![
                (edge([0, 0], 1), F2Vector::zeros(1)),
                (edge([0, 0], 0), F2Vector::zeros(1)),
            ],
        };
        assert!(matches!(
            round_trip(data).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }
}
