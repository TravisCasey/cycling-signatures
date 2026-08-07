// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`CubicalCover`]: the cube set together with each
//! cohomology generator flattened to its sorted list of edges.

use std::cmp::Ordering;

use chomp3rs::{Chain, Cube, F2, Ring};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize, de::Error as DeserializeError};

use super::{CubicalCover, generators::compute_edge_classes};
use crate::error::Error;

#[derive(Serialize, Deserialize)]
struct CoverData {
    #[serde(with = "crate::serialization::npy_field")]
    cubes: Array2<i64>,
    generators: Vec<Vec<Cube>>,
}

/// Checks that every cube coordinate is one the cubical-homology backend
/// accepts, returning a description of the first offender.
///
/// The description is the one [`Error::CubeCoordinateOutOfRange`] carries, so
/// a decoded cover and an explicitly built one report the same condition in
/// the same words.
fn check_cube_range(cubes: ArrayView2<'_, i64>) -> std::result::Result<(), String> {
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

/// Checks that the cube rows are strictly increasing in lexicographic order,
/// returning a description of the first offending pair.
///
/// That ordering is the canonical form a cover holds, and cube lookup is a
/// binary search over it: an unordered or duplicated row set answers lookups
/// wrongly rather than loudly.
fn check_lexicographic_order(cubes: ArrayView2<'_, i64>) -> std::result::Result<(), String> {
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

impl Serialize for CubicalCover {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let generators: Vec<Vec<Cube>> = self
            .generators
            .iter()
            .map(|generator| {
                let mut cubes: Vec<Cube> = generator
                    .iter()
                    .filter(|(_, coefficient)| **coefficient != F2::zero())
                    .map(|(cube, _)| cube.clone())
                    .collect();
                cubes.sort();
                cubes
            })
            .collect();
        CoverData {
            cubes: self.cubes.clone(),
            generators,
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
        let generators: Vec<Chain<Cube, F2>> = data
            .generators
            .into_iter()
            .map(|cubes| cubes.into_iter().map(|cube| (cube, F2::one())).collect())
            .collect();
        let edge_classes = compute_edge_classes(&generators);
        Ok(Self {
            cubes: data.cubes,
            generators,
            edge_classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::{CoverData, CubicalCover};
    use crate::{
        error::Error,
        serialization::{load_from_reader, save_to_writer},
    };

    /// Encodes `cubes` as a generator-free cover payload and decodes it back.
    fn round_trip(cubes: Array2<i64>) -> Result<CubicalCover, Error> {
        let data = CoverData {
            cubes,
            generators: Vec::new(),
        };
        let mut buffer: Vec<u8> = Vec::new();
        save_to_writer(&mut buffer, &data).unwrap();
        load_from_reader(&buffer[..])
    }

    #[test]
    fn deserialize_rejects_out_of_range_cube_coordinate() {
        // The largest valid coordinate is i32::MAX - 1; i32::MAX is one past it.
        let cubes = array![[0_i64, 0], [1, i64::from(i32::MAX)]];
        assert!(matches!(
            round_trip(cubes).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }

    #[test]
    fn deserialize_rejects_cubes_out_of_lexicographic_order() {
        let cubes = array![[1_i64, 0], [0, 0]];
        assert!(matches!(
            round_trip(cubes).unwrap_err(),
            Error::Deserialize { .. }
        ));
    }
}
