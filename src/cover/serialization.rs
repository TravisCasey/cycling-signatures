// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Wire format for [`CubicalCover`]: the cube set together with each
//! cohomology generator flattened to its sorted list of edges.

use chomp3rs::{Chain, Cube, F2, Ring};
use ndarray::Array2;
use serde::{Deserialize, Serialize, de::Error as DeserializeError};

use super::{CubicalCover, generators::compute_edge_classes};

#[derive(Serialize, Deserialize)]
struct CoverData {
    #[serde(with = "crate::serialization::npy_field")]
    cubes: Array2<i64>,
    generators: Vec<Vec<Cube>>,
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
