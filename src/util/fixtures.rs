// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Trajectory and embedding fixtures shared by the crate's test modules.

use chomp3rs::ExecutionBackend;
use ndarray::Array2;

use crate::{
    cover::CubicalCover, embedded::EmbeddedTrajectory, error::Result, metric::Metric,
    trajectory::Trajectory,
};

/// Covers `trajectory`'s own cubes and embeds it under the `Euclidean` metric.
pub(crate) fn embed_euclidean(trajectory: Trajectory) -> Result<EmbeddedTrajectory> {
    let cover = CubicalCover::build(&trajectory, &ExecutionBackend::default())?;
    EmbeddedTrajectory::new(trajectory, cover, Metric::Euclidean)
}

/// The centers of the eight cubes ringing the missing center cube `(1, 1)`,
/// closing back on the first.
///
/// Covering these cubes and no others leaves a one-cube hole, so the cover
/// has `H^1` of rank one and a loop around the ring carries its generator.
pub(crate) fn ring_waypoints() -> [[f64; 2]; 9] {
    [
        [0.5, 0.5],
        [1.5, 0.5],
        [2.5, 0.5],
        [2.5, 1.5],
        [2.5, 2.5],
        [1.5, 2.5],
        [0.5, 2.5],
        [0.5, 1.5],
        [0.5, 0.5],
    ]
}

/// Stacks `points` into a two-column array, one row per point.
fn stack_points(points: &[[f64; 2]]) -> Array2<f64> {
    let flat: Vec<f64> = points.iter().flatten().copied().collect();
    Array2::from_shape_vec((points.len(), 2), flat)
        .expect("flattened point rows form a valid two-column matrix")
}

/// Inserts evenly spaced intermediate points between consecutive `waypoints` so
/// that no step's Euclidean distance exceeds `max_step`.
///
/// Turns a short list of cube-center waypoints into a densely sampled
/// trajectory whose consecutive-point resolution stays below the cube side,
/// while every waypoint's cube membership (its coordinate floors) is
/// unaffected: only points strictly between waypoints are added.
pub(crate) fn densify_path(waypoints: &[[f64; 2]], max_step: f64) -> Array2<f64> {
    let mut points: Vec<[f64; 2]> = vec![waypoints[0]];
    for pair in waypoints.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let distance = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
        let steps = ((distance / max_step).ceil() as usize).max(1);
        for step in 1..=steps {
            let fraction = step as f64 / steps as f64;
            points.push([
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            ]);
        }
    }
    stack_points(&points)
}
