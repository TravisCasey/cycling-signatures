// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Writes a square-loop `EmbeddedTrajectory` to disk under the `Euclidean`
//! metric, as a trajectory file and a cover file. The output directory is the
//! single command-line argument.

use std::{error::Error, path::Path};

use cycling_signatures::{EmbeddedTrajectory, Euclidean, ExecutionBackend, Trajectory};
use ndarray::Array2;

/// Builds a closed square loop of side length 2.0, sampled with 50 points per
/// side and closed by repeating the first point (201 points total). Under
/// unit-side cubes the loop encircles one empty cube, giving it first homology
/// rank 1.
fn square_loop() -> Array2<f64> {
    let side = 2.0;
    let steps_per_side = 50;
    let step = side / steps_per_side as f64;

    let mut points: Vec<[f64; 2]> = Vec::with_capacity(steps_per_side * 4 + 1);
    for index in 0..steps_per_side {
        points.push([index as f64 * step, 0.0]);
    }
    for index in 0..steps_per_side {
        points.push([side, index as f64 * step]);
    }
    for index in 0..steps_per_side {
        points.push([side - index as f64 * step, side]);
    }
    for index in 0..steps_per_side {
        points.push([0.0, side - index as f64 * step]);
    }
    let first = points[0];
    points.push(first);

    let rows = points.len();
    let coordinates: Vec<f64> = points.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, 2), coordinates).expect("each point contributes two coordinates")
}

fn main() -> Result<(), Box<dyn Error>> {
    let output_directory = std::env::args()
        .nth(1)
        .ok_or("usage: generate_python_fixture <output-directory>")?;
    let output_directory = Path::new(&output_directory);

    let points = square_loop();
    let trajectory = Trajectory::new(points.view())?;
    let embedded = EmbeddedTrajectory::new(
        trajectory,
        Box::new(Euclidean),
        &ExecutionBackend::default(),
    )?;
    embedded.save(
        output_directory.join("trajectory.cyc"),
        output_directory.join("cover.cyc"),
    )?;
    Ok(())
}
