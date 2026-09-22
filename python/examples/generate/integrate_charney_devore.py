# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Integrate the Charney-DeVore trajectory the gallery's data derives from.

Writes two files under ``charney_devore/data``: ``cdv_raw.npy``, the raw state
trajectory of the six-mode Charney-DeVore model recorded after a discarded
transient, and ``cdv_times.npy``, the integration time of each of its raw rows.
Integration uses fourth-order Runge-Kutta steps with size adapted to the local
speed of the flow.

A raw row is recorded whenever the trajectory has moved a fixed Euclidean
distance from the previously recorded row, which spaces the rows evenly along
the curve rather than evenly in time. The spacing controls how densely the
attractor is covered relative to the downstream cubical cover: tighter spacing
shrinks the metric distance between consecutive raw rows, which reduces the
resample densification the embedding pipeline needs.

Because the raw rows are spaced by distance, the time between consecutive rows
varies with the local speed of the flow, which spans roughly two orders of
magnitude between the zonal and blocked phases. The times array carries that
information: it holds one strictly increasing entry per raw row, measured in
the model's own time units from the first saved row, so its first entry is zero
and the discarded transient contributes nothing to it.
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

# Model parameters. The zonal forcing ratio and the wave numbers place the
# system in the chaotic regime-transition window rather than at a stable
# blocked or zonal equilibrium.
DAMPING = 0.1
ZONAL_FORCING_ONE = 0.95
FORCING_RATIO = -0.801
ZONAL_FORCING_FOUR = FORCING_RATIO * ZONAL_FORCING_ONE
CHANNEL_ASPECT = 0.5
BETA = 1.25
TOPOGRAPHY = 0.2

INITIAL_STATE = (ZONAL_FORCING_ONE, 0.0, 0.0, ZONAL_FORCING_FOUR, 0.0, 0.0)

# Internal Runge-Kutta step targets: each step advances the state by about
# this arc length, never exceeding this much time. The arc target binds
# through most of the attractor; the time limit takes over in the slow
# blocked phases, where the speed falls far enough that the arc target alone
# would ask for an enormous step.
ARC_TARGET = 0.001
STEP_LIMIT = 0.05

DEFAULT_ROW_SPACING = 0.001
DEFAULT_ROW_COUNT = 60_000_000
DEFAULT_TRANSIENT_TIME = 10_000.0
_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "charney_devore" / "data"
DEFAULT_OUTPUT = _DATA_DIRECTORY / "cdv_raw.npy"
DEFAULT_TIMES_OUTPUT = _DATA_DIRECTORY / "cdv_times.npy"

# Row numbers compared against a reference trajectory, spread across the run so
# that a divergence anywhere in it is caught. Two integrations of this model
# that differ at all diverge exponentially, so an agreement at these rows is an
# agreement everywhere before the last of them.
CHECKED_ROWS = (0, 1, 1_000_000, 19_999_999, 20_000_000, 33_333_333, 39_999_999)

State = tuple[float, float, float, float, float, float]

_ROOT_TWO = math.sqrt(2.0)


def _wave_advection(wave_number: float) -> float:
    """The nonlinear advection coefficient for mode `wave_number`."""
    return (
        8.0
        * _ROOT_TWO
        / math.pi
        * wave_number**2
        / (4.0 * wave_number**2 - 1.0)
        * (CHANNEL_ASPECT**2 + wave_number**2 - 1.0)
        / (CHANNEL_ASPECT**2 + wave_number**2)
    )


def _rossby(wave_number: float) -> float:
    """The Rossby wave frequency for mode `wave_number`."""
    return BETA * CHANNEL_ASPECT**2 / (CHANNEL_ASPECT**2 + wave_number**2)


def _wave_coupling(wave_number: float) -> float:
    """The coupling coefficient between the two wave pairs."""
    return (
        64.0
        * _ROOT_TWO
        / (15.0 * math.pi)
        * (CHANNEL_ASPECT**2 - wave_number**2 + 1.0)
        / (CHANNEL_ASPECT**2 + wave_number**2)
    )


def _topographic_zonal(wave_number: float) -> float:
    """The topographic forcing felt by the zonal mode `wave_number`."""
    return (
        TOPOGRAPHY
        * 4.0
        * wave_number
        / (4.0 * wave_number**2 - 1.0)
        * (_ROOT_TWO * CHANNEL_ASPECT)
        / math.pi
    )


def _topographic_wave(wave_number: float) -> float:
    """The topographic forcing felt by the wave mode `wave_number`."""
    return (
        TOPOGRAPHY
        * 4.0
        * wave_number**3
        / (4.0 * wave_number**2 - 1.0)
        * (_ROOT_TWO * CHANNEL_ASPECT)
        / (math.pi * (CHANNEL_ASPECT**2 + wave_number**2))
    )


ADVECTION_ONE = _wave_advection(1.0)
ADVECTION_TWO = _wave_advection(2.0)
ROSSBY_ONE = _rossby(1.0)
ROSSBY_TWO = _rossby(2.0)
COUPLING_ONE = _wave_coupling(1.0)
COUPLING_TWO = _wave_coupling(2.0)
TOPOGRAPHIC_WAVE_ONE = _topographic_wave(1.0)
TOPOGRAPHIC_WAVE_TWO = _topographic_wave(2.0)
TOPOGRAPHIC_ZONAL_ONE = _topographic_zonal(1.0)
TOPOGRAPHIC_ZONAL_TWO = _topographic_zonal(2.0)
WAVE_INTERACTION = 16.0 * _ROOT_TWO / (5.0 * math.pi)


def derivative(state: State) -> State:
    """Return the Charney-DeVore vector field at `state`."""
    first, second, third, fourth, fifth, sixth = state
    return (
        TOPOGRAPHIC_ZONAL_ONE * third - DAMPING * (first - ZONAL_FORCING_ONE),
        -(ADVECTION_ONE * first - ROSSBY_ONE) * third
        - DAMPING * second
        - COUPLING_ONE * fourth * sixth,
        (ADVECTION_ONE * first - ROSSBY_ONE) * second
        - TOPOGRAPHIC_WAVE_ONE * first
        - DAMPING * third
        + COUPLING_ONE * fourth * fifth,
        TOPOGRAPHIC_ZONAL_TWO * sixth
        - DAMPING * (fourth - ZONAL_FORCING_FOUR)
        + WAVE_INTERACTION * (second * sixth - third * fifth),
        -(ADVECTION_TWO * first - ROSSBY_TWO) * sixth
        - DAMPING * fifth
        - COUPLING_TWO * fourth * third,
        (ADVECTION_TWO * first - ROSSBY_TWO) * fifth
        - TOPOGRAPHIC_WAVE_TWO * fourth
        - DAMPING * sixth
        + COUPLING_TWO * fourth * second,
    )


def add_scaled(state: State, factor: float, slope: State) -> State:
    """Return `state` displaced by `factor` times `slope`."""
    first, second, third, fourth, fifth, sixth = (
        value + factor * rate for value, rate in zip(state, slope, strict=True)
    )
    return (first, second, third, fourth, fifth, sixth)


def runge_kutta_step(state: State, step: float) -> State:
    """Advance `state` by one step of classic fourth-order Runge-Kutta."""
    slope_start = derivative(state)
    slope_mid_one = derivative(add_scaled(state, 0.5 * step, slope_start))
    slope_mid_two = derivative(add_scaled(state, 0.5 * step, slope_mid_one))
    slope_end = derivative(add_scaled(state, step, slope_mid_two))
    sixth_step = step / 6.0
    first, second, third, fourth, fifth, sixth = (
        value + sixth_step * (start + 2.0 * (mid_one + mid_two) + end)
        for value, start, mid_one, mid_two, end in zip(
            state, slope_start, slope_mid_one, slope_mid_two, slope_end, strict=True
        )
    )
    return (first, second, third, fourth, fifth, sixth)


def adaptive_step(state: State) -> float:
    """Return the step size advancing `state` by about `ARC_TARGET`.

    The step is the arc target divided by the local speed, capped at
    `STEP_LIMIT` where the flow is slow.
    """
    slope = derivative(state)
    speed = math.sqrt(sum(rate * rate for rate in slope))
    if speed * STEP_LIMIT <= ARC_TARGET:
        return STEP_LIMIT
    return ARC_TARGET / speed


def integrate(
    row_spacing: float, row_count: int, transient_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return `row_count` spaced raw rows, with the time of each.

    The trajectory starts from a fixed initial state and discards
    `transient_time` time units before recording, so the raw rows lie on the
    attractor. Each subsequent row is the first integration state at least
    `row_spacing` away from its predecessor in Euclidean distance.

    The second array holds the integration time of each raw row, measured from
    the first saved row, so its first entry is zero and it increases strictly.
    """
    state = INITIAL_STATE
    elapsed = 0.0
    while elapsed < transient_time:
        step = adaptive_step(state)
        state = runge_kutta_step(state, step)
        elapsed += step

    rows = np.empty((row_count, len(INITIAL_STATE)))
    times = np.empty(row_count)
    rows[0] = state
    times[0] = 0.0
    elapsed = 0.0
    previous = state
    spacing_squared = row_spacing * row_spacing
    for row_index in range(1, row_count):
        while True:
            step = adaptive_step(state)
            state = runge_kutta_step(state, step)
            elapsed += step
            gap = sum(
                (value - earlier) ** 2 for value, earlier in zip(state, previous, strict=True)
            )
            if gap >= spacing_squared:
                break
        rows[row_index] = state
        times[row_index] = elapsed
        previous = state
    return rows, times


def compare_against(rows: np.ndarray, reference_path: Path) -> tuple[list[int], list[int]]:
    """Return the row numbers compared against a reference, and those differing.

    The reference is a raw trajectory saved by an earlier run, read from
    `reference_path`. Only the row numbers in `CHECKED_ROWS` that lie inside
    both trajectories are compared, and they are compared exactly, so two
    integrations agree only if they reproduce each other bit for bit. The first
    list is empty when the two trajectories are too short to share any checked
    row, which is no evidence either way.
    """
    reference = np.load(reference_path, mmap_mode="r")
    limit = min(len(rows), len(reference))
    compared = [row_index for row_index in CHECKED_ROWS if row_index < limit]
    differing = [
        row_index
        for row_index in compared
        if not np.array_equal(rows[row_index], reference[row_index])
    ]
    return compared, differing


def save_atomic(array: np.ndarray, target: Path) -> None:
    """Save `array` to `target` so the file appears whole or not at all."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".npy")
    try:
        with os.fdopen(handle, "wb") as sink:
            np.save(sink, array)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    """Integrate and save the trajectory described by the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-spacing",
        type=float,
        default=DEFAULT_ROW_SPACING,
        help="Euclidean distance between saved raw rows (default %(default)s)",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=DEFAULT_ROW_COUNT,
        help="number of raw rows to save (default %(default)s)",
    )
    parser.add_argument(
        "--transient-time",
        type=float,
        default=DEFAULT_TRANSIENT_TIME,
        help="time units to discard before recording (default %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output .npy path for the raw rows (default %(default)s)",
    )
    parser.add_argument(
        "--times-output",
        type=Path,
        default=DEFAULT_TIMES_OUTPUT,
        help="output .npy path for the raw row times (default %(default)s)",
    )
    parser.add_argument(
        "--check-against",
        type=Path,
        default=None,
        help="raw trajectory .npy to reproduce; nothing is saved if it is not reproduced",
    )
    arguments = parser.parse_args()

    rows, times = integrate(arguments.row_spacing, arguments.row_count, arguments.transient_time)
    if arguments.check_against is not None:
        compared, differing = compare_against(rows, arguments.check_against)
        if not compared:
            raise SystemExit(
                f"no checked row lies inside both this trajectory and "
                f"{arguments.check_against}, so nothing was compared"
            )
        if differing:
            raise SystemExit(
                f"the integrated trajectory does not reproduce {arguments.check_against}: "
                f"rows {differing} differ"
            )
        print(f"{arguments.check_against}  reproduced at {len(compared)} rows")
    save_atomic(rows, arguments.output)
    save_atomic(times, arguments.times_output)
    print(f"{arguments.output}  {rows.shape[0]} raw rows")
    print(f"{arguments.times_output}  {times[-1]:.1f} time units")


if __name__ == "__main__":
    main()
