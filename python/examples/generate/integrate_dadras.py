# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Integrate the Dadras trajectory the gallery's example data derives from.

Writes two files under ``dadras/data``: ``dadras_raw.npy``, the position
samples of the four-wing chaotic Dadras system (parameters a 8, b 40, c 14.9)
recorded after a discarded transient, and ``dadras_times.npy``, the integration
time each of those samples was recorded at. Integration uses fourth-order
Runge-Kutta steps with size adapted to the local speed of the flow.

A sample is recorded whenever the trajectory has moved a fixed Euclidean
distance from the previously recorded sample. The spacing controls how densely
the attractor is sampled relative to the downstream cubical cover: tighter
spacing shrinks the metric distance between consecutive samples, which
reduces the resample densification the embedding pipeline needs.

Because the samples are spaced by distance, the time between consecutive samples
varies with the local speed of the flow. The times array carries that
information: it holds one strictly increasing entry per saved sample,
measured in the system's own time units from the first saved sample, so its
first entry is zero and the discarded transient contributes nothing to it.
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

PARAMETER_A = 8.0
PARAMETER_B = 40.0
PARAMETER_C = 14.9

INITIAL_STATE = (10.0, 1.0, 10.0, 1.0)

# Internal Runge-Kutta step targets: each step advances the state by about
# this arc length, never exceeding this much time.
ARC_TARGET = 0.25
STEP_LIMIT = 0.002

DEFAULT_SAMPLE_SPACING = 1.0
DEFAULT_SAMPLE_COUNT = 1_500_000
DEFAULT_TRANSIENT_TIME = 10_000.0
_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "dadras" / "data"
DEFAULT_OUTPUT = _DATA_DIRECTORY / "dadras_raw.npy"
DEFAULT_TIMES_OUTPUT = _DATA_DIRECTORY / "dadras_times.npy"

State = tuple[float, float, float, float]


def derivative(state: State) -> State:
    """Return the Dadras vector field at ``state``."""
    x, y, z, w = state
    return (
        PARAMETER_A * x - y * z + w,
        x * z - PARAMETER_B * y,
        x * y - PARAMETER_C * z + x * w,
        -y,
    )


def add_scaled(state: State, factor: float, slope: State) -> State:
    """Return ``state`` displaced by ``factor`` times ``slope``."""
    return (
        state[0] + factor * slope[0],
        state[1] + factor * slope[1],
        state[2] + factor * slope[2],
        state[3] + factor * slope[3],
    )


def runge_kutta_step(state: State, step: float) -> State:
    """Advance ``state`` by one step of classic fourth-order Runge-Kutta."""
    slope_start = derivative(state)
    slope_mid_one = derivative(add_scaled(state, 0.5 * step, slope_start))
    slope_mid_two = derivative(add_scaled(state, 0.5 * step, slope_mid_one))
    slope_end = derivative(add_scaled(state, step, slope_mid_two))
    return (
        state[0]
        + (step / 6.0)
        * (slope_start[0] + 2.0 * (slope_mid_one[0] + slope_mid_two[0]) + slope_end[0]),
        state[1]
        + (step / 6.0)
        * (slope_start[1] + 2.0 * (slope_mid_one[1] + slope_mid_two[1]) + slope_end[1]),
        state[2]
        + (step / 6.0)
        * (slope_start[2] + 2.0 * (slope_mid_one[2] + slope_mid_two[2]) + slope_end[2]),
        state[3]
        + (step / 6.0)
        * (slope_start[3] + 2.0 * (slope_mid_one[3] + slope_mid_two[3]) + slope_end[3]),
    )


def adaptive_step(state: State) -> float:
    """Return the step size advancing ``state`` by about ``ARC_TARGET``.

    The step is the arc target divided by the local speed, capped at
    ``STEP_LIMIT`` where the flow is slow.
    """
    slope = derivative(state)
    speed = math.sqrt(slope[0] ** 2 + slope[1] ** 2 + slope[2] ** 2 + slope[3] ** 2)
    if speed * STEP_LIMIT <= ARC_TARGET:
        return STEP_LIMIT
    return ARC_TARGET / speed


def integrate(
    sample_spacing: float, sample_count: int, transient_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``sample_count`` spaced samples, with the time of each.

    The trajectory starts from a fixed initial state and discards
    ``transient_time`` time units before recording, so the samples lie on the
    attractor. Each subsequent sample is the first integration state at least
    ``sample_spacing`` away from its predecessor in Euclidean distance.

    The second array holds the integration time of each sample, measured from
    the first saved sample, so its first entry is zero and it increases
    strictly.
    """
    state = INITIAL_STATE
    elapsed = 0.0
    while elapsed < transient_time:
        step = adaptive_step(state)
        state = runge_kutta_step(state, step)
        elapsed += step

    samples = np.empty((sample_count, 4))
    times = np.empty(sample_count)
    samples[0] = state
    times[0] = 0.0
    elapsed = 0.0
    previous = state
    spacing_squared = sample_spacing * sample_spacing
    for sample_index in range(1, sample_count):
        while True:
            step = adaptive_step(state)
            state = runge_kutta_step(state, step)
            elapsed += step
            gap = (
                (state[0] - previous[0]) ** 2
                + (state[1] - previous[1]) ** 2
                + (state[2] - previous[2]) ** 2
                + (state[3] - previous[3]) ** 2
            )
            if gap >= spacing_squared:
                break
        samples[sample_index] = state
        times[sample_index] = elapsed
        previous = state
    return samples, times


def save_atomic(array: np.ndarray, target: Path) -> None:
    """Save ``array`` to ``target`` so the file appears whole or not at all."""
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
        "--sample-spacing",
        type=float,
        default=DEFAULT_SAMPLE_SPACING,
        help="Euclidean distance between saved samples (default %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="number of samples to save (default %(default)s)",
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
        help="output .npy path for the position samples (default %(default)s)",
    )
    parser.add_argument(
        "--times-output",
        type=Path,
        default=DEFAULT_TIMES_OUTPUT,
        help="output .npy path for the sample times (default %(default)s)",
    )
    arguments = parser.parse_args()

    samples, times = integrate(
        arguments.sample_spacing, arguments.sample_count, arguments.transient_time
    )
    save_atomic(samples, arguments.output)
    save_atomic(times, arguments.times_output)
    print(f"{arguments.output}  {samples.shape[0]} samples")
    print(f"{arguments.times_output}  {times[-1]:.1f} time units")


if __name__ == "__main__":
    main()
