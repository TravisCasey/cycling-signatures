// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cube-path traversal for cycle segments.

use std::ops::{Range, RangeBounds};

use chomp3rs::{Cube, Orthant};

use super::EmbeddedTrajectory;
use crate::{
    F2Vector,
    cover::non_adjacent_axis,
    error::{Error, Result},
    util::range::normalize_segment,
};

impl EmbeddedTrajectory {
    /// The sequence of 1-cube edges traversed when walking the cycle
    /// described by `segment`: forward along the trajectory from the point at
    /// `segment.start` to the point at `segment.end - 1`, then a closing
    /// cube-to-cube path back to the point at `segment.start`.
    ///
    /// Every step, the closing one included, is realized as one axis-aligned
    /// staircase between the cubes of its two points.
    ///
    /// Useful for visualizing the cubical representation of a particular cycle.
    /// For the homology class alone, call [`cycle_class`](Self::cycle_class)
    /// instead.
    ///
    /// # Errors
    ///
    /// - [`Error::SegmentOutOfBounds`] if `segment` does not normalize to a
    ///   valid sub-range.
    /// - [`Error::CycleEndpointsNonAdjacent`] if the cubes of the trajectory
    ///   points at `segment.start` and `segment.end - 1` differ by more than 1
    ///   in some axis.
    ///
    /// # Panics
    ///
    /// Panics if the normalized segment contains fewer than 2 points, or if a
    /// forward step inside the segment lands in cubes differing by more than 1
    /// in some axis.
    pub fn walk_cycle(&self, segment: impl RangeBounds<usize>) -> Result<Vec<Cube>> {
        let segment = self.cycle_segment(segment)?;
        let mut edges = Vec::new();
        for_each_cycle_edge(self, segment, |edge| {
            edges.push(edge.clone());
        })?;
        Ok(edges)
    }

    /// The `F_2` homology class of the cycle described by `segment`,
    /// expressed in the cover's generator basis.
    ///
    /// Use this when only the class is needed. To inspect the underlying
    /// cubical edge sequence, call [`walk_cycle`](Self::walk_cycle) instead.
    ///
    /// # Errors
    ///
    /// Same as [`walk_cycle`](Self::walk_cycle).
    ///
    /// # Panics
    ///
    /// Same as [`walk_cycle`](Self::walk_cycle).
    pub fn cycle_class(&self, segment: impl RangeBounds<usize>) -> Result<F2Vector> {
        let segment = self.cycle_segment(segment)?;
        // Accumulated as the walk proceeds rather than over a collected edge
        // list: `F_2` addition is commutative, so the sum does not depend on
        // the order the edges arrive in.
        let mut accumulator = F2Vector::zeros(self.cover().num_generators());
        for_each_cycle_edge(self, segment, |edge| {
            if let Some(class) = self.cover().edge_class(edge) {
                accumulator ^= class;
            }
        })?;
        Ok(accumulator)
    }

    /// Normalizes `segment` against the trajectory and checks that it holds
    /// enough points to describe a cycle.
    ///
    /// # Errors
    ///
    /// - [`Error::SegmentOutOfBounds`] if `segment` does not normalize to a
    ///   valid sub-range.
    ///
    /// # Panics
    ///
    /// Panics if the normalized segment contains fewer than 2 points.
    fn cycle_segment(&self, segment: impl RangeBounds<usize>) -> Result<Range<usize>> {
        let segment = normalize_segment(segment, self.trajectory().len())?;
        assert!(
            segment.end > segment.start + 1,
            "cycle segment {}..{} must contain at least two points",
            segment.start,
            segment.end
        );
        Ok(segment)
    }
}

/// Walks the cubical path of the cycle described by `segment`, invoking
/// `visit` once per 1-cube edge traversed.
///
/// `segment` is the already-normalized half-open range of trajectory points.
/// The forward path walks consecutive points `segment.start..segment.end` in
/// order; the closing path connects `points[segment.end - 1]` back to
/// `points[segment.start]`. Each step, closing included, is one staircase
/// between the cubes of its two points.
///
/// Each step validates that the cube delta is in `{-1, 0, 1}` per axis. The
/// closing step, whose two points are the caller's cycle endpoints, returns
/// `CycleEndpointsNonAdjacent` when it exceeds that.
///
/// # Panics
///
/// Panics if a forward step exceeds the per-axis delta. Adjacency of
/// consecutive points holds across the whole trajectory once an embedding
/// exists, so this is a violated invariant rather than caller input error.
pub(super) fn for_each_cycle_edge<F>(
    embedded: &EmbeddedTrajectory,
    segment: Range<usize>,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(&Cube),
{
    let cubes = embedded.cover().cubes();
    let dimension = cubes.ncols();
    // Infallible: a cover owns its cube array and every way of producing one
    // leaves it in the standard row-major layout, so the view is contiguous.
    let coordinates = cubes
        .to_slice()
        .expect("cover cubes are held as one contiguous row-major block");
    let point_cube_coordinates = |point_index: usize| {
        let cube_index = embedded.point_to_cube(point_index);
        &coordinates[cube_index * dimension..][..dimension]
    };

    let start_cube = point_cube_coordinates(segment.start);
    // Cube coordinates fit in i32: every way of obtaining a cover, building
    // one or decoding a saved one, enforces the [i32::MIN, i32::MAX - 1] range.
    let mut base: Orthant = start_cube.iter().map(|&value| value as i32).collect();
    let mut dual: Orthant = start_cube.iter().map(|&value| value as i32 - 1).collect();

    // Forward path: each consecutive pair of points. Every consecutive pair of
    // the trajectory was shown to land in adjacent cubes before the embedding
    // existed.
    for point_index in segment.start..(segment.end - 1) {
        let from = point_cube_coordinates(point_index);
        let to = point_cube_coordinates(point_index + 1);
        if let Err((axis, delta)) =
            step_between_cubes(from, to, dimension, &mut base, &mut dual, &mut visit)
        {
            panic!(
                "consecutive trajectory points {point_index} and {} land in cubes differing by \
                 {delta} in axis {axis}",
                point_index + 1
            );
        }
    }

    // Closing step: from points[end - 1] to points[start].
    let end_cube = point_cube_coordinates(segment.end - 1);
    step_between_cubes(
        end_cube, start_cube, dimension, &mut base, &mut dual, &mut visit,
    )
    .map_err(|(axis, delta)| Error::CycleEndpointsNonAdjacent {
        start: segment.start,
        end: segment.end,
        axis,
        delta,
    })?;

    Ok(())
}

/// Steps from `from` cube to `to` cube one axis-aligned unit at a time,
/// emitting one 1-cube edge per step.
///
/// The two passes are ordered so that every emitted edge lies in one of the
/// two endpoint cubes: every axis whose cube coordinate increases is stepped
/// first, then every axis whose coordinate decreases. In the rising pass, the
/// axis being stepped is still at its `from` coordinate and every other axis
/// lies within one unit above its `from` coordinate, so the edge lies in the
/// closed `from` cube; in the falling pass, the axis being stepped has
/// already reached its `to` coordinate and the others lie within one unit
/// above theirs, so the edge lies in the closed `to` cube. The whole
/// staircase is therefore trapped in the union of two cubes that meet, which
/// is contractible, and any two staircases obeying this ordering are
/// homotopic there: every 1-cocycle scores them identically, so the class a
/// cycle walk reports does not depend on which staircase was taken.
///
/// Returns the offending axis and its signed cube delta if any axis difference
/// exceeds one unit in magnitude.
fn step_between_cubes<F>(
    from: &[i64],
    to: &[i64],
    dimension: usize,
    base: &mut Orthant,
    dual: &mut Orthant,
    visit: &mut F,
) -> std::result::Result<(), (usize, i64)>
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
        dual[axis] += 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        base[axis] += 1;
    }

    // Falling axes second: their edges lie in the `to` cube.
    for axis in 0..dimension {
        if to[axis] - from[axis] != -1 {
            continue;
        }
        base[axis] -= 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        dual[axis] -= 1;
    }

    Ok(())
}
