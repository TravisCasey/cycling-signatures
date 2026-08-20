// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The deterministic axis-aligned staircase between two adjacent cubes: rising
//! axes in ascending order, then falling axes in ascending order.

use chomp3rs::{Cube, Orthant};

use super::non_adjacent_axis;

/// Steps from the `from` cube to the `to` cube one axis-aligned unit at a time,
/// emitting one 1-cube edge per step.
///
/// The two passes are ordered so that every emitted edge lies in one of the
/// two endpoint cubes: every axis whose cube coordinate increases is stepped
/// first, then every axis whose coordinate decreases. In the rising pass, the
/// axis being stepped is still at its `from` coordinate and every other axis
/// lies within one unit above its `from` coordinate, so the edge lies in the
/// closed `from` cube; in the falling pass, the axis being stepped has already
/// reached its `to` coordinate and the others lie within one unit above theirs,
/// so the edge lies in the closed `to` cube.
///
/// The entire staircase is therefore trapped in the union of two cubes that
/// meet, which is contractible, and any two staircases obeying this ordering
/// are homotopic there: every 1-cocycle scores them identically, so the class a
/// cycle walk reports does not depend on which staircase was taken.
///
/// # Errors
///
/// Returns the offending axis and its signed cube delta if any axis
/// difference exceeds one unit in magnitude.
pub(crate) fn step_between_cubes<F>(
    from: &[i64],
    to: &[i64],
    dimension: usize,
    position: &mut Orthant,
    visit: &mut F,
) -> Result<(), (usize, i64)>
where
    F: FnMut(&Cube),
{
    if let Some(gap) = non_adjacent_axis(from, to) {
        return Err(gap);
    }

    // Rising axes first: their edges lie in the `from` cube.
    for axis in 0..dimension {
        if to[axis] - from[axis] != 1 {
            continue;
        }
        let edge = Cube::from_extent(position.clone(), 1_u32 << axis);
        visit(&edge);
        position[axis] += 1;
    }

    // Falling axes second: their edges lie in the `to` cube.
    for axis in 0..dimension {
        if to[axis] - from[axis] != -1 {
            continue;
        }
        position[axis] -= 1;
        let edge = Cube::from_extent(position.clone(), 1_u32 << axis);
        visit(&edge);
    }

    Ok(())
}
