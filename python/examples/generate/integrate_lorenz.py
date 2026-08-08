# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Integrate the Lorenz trajectory the gallery's example data derives from.

Writes ``lorenz/data/lorenz_raw.npy``: the raw position trajectory of the
classic Lorenz system (sigma 10, rho 28, beta 8/3), recorded at a fixed time
interval after a transient burn-in. Integration uses fixed-step fourth-order
Runge-Kutta with several substeps per raw row, so identical arguments
deterministically reproduce the trajectory.

The row interval controls how densely the attractor is covered relative to
the cubical cover downstream: a finer interval shrinks the metric distance
between consecutive raw rows, which reduces the resample densification the
embedding pipeline needs.
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

INITIAL_STATE = (1.0, 1.0, 1.0)

# Internal Runge-Kutta step target; each row interval is integrated in equal
# substeps no longer than this.
STEP_TARGET = 0.0025

DEFAULT_ROW_INTERVAL = 0.007
DEFAULT_ROW_COUNT = 600_000
DEFAULT_TRANSIENT_TIME = 100.0
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "lorenz" / "data" / "lorenz_raw.npy"


def derivative(state: np.ndarray) -> np.ndarray:
    """Return the Lorenz vector field at ``state``."""
    x, y, z = state
    return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def runge_kutta_step(state: np.ndarray, step: float) -> np.ndarray:
    """Advance ``state`` by one step of classic fourth-order Runge-Kutta."""
    slope_start = derivative(state)
    slope_mid_one = derivative(state + 0.5 * step * slope_start)
    slope_mid_two = derivative(state + 0.5 * step * slope_mid_one)
    slope_end = derivative(state + step * slope_mid_two)
    return state + (step / 6.0) * (
        slope_start + 2.0 * slope_mid_one + 2.0 * slope_mid_two + slope_end
    )


def integrate(row_interval: float, row_count: int, transient_time: float) -> np.ndarray:
    """Return ``row_count`` raw rows spaced ``row_interval`` apart.

    The trajectory starts from a fixed initial state and discards
    ``transient_time`` time units before recording, so the raw rows lie on
    the attractor.
    """
    substeps = max(1, int(np.ceil(row_interval / STEP_TARGET)))
    step = row_interval / substeps

    state = np.array(INITIAL_STATE)
    transient_steps = int(np.ceil(transient_time / step))
    for _ in range(transient_steps):
        state = runge_kutta_step(state, step)

    rows = np.empty((row_count, 3))
    for row_index in range(row_count):
        rows[row_index] = state
        for _ in range(substeps):
            state = runge_kutta_step(state, step)
    return rows


def main() -> None:
    """Integrate and save the trajectory described by the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-interval",
        type=float,
        default=DEFAULT_ROW_INTERVAL,
        help="time between saved raw rows (default %(default)s)",
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
        help="output .npy path (default %(default)s)",
    )
    arguments = parser.parse_args()

    rows = integrate(arguments.row_interval, arguments.row_count, arguments.transient_time)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    # Write through a temporary file so the output appears atomically.
    handle, temporary = tempfile.mkstemp(dir=arguments.output.parent, suffix=".npy")
    try:
        with os.fdopen(handle, "wb") as sink:
            np.save(sink, rows)
        os.replace(temporary, arguments.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    duration = arguments.row_interval * arguments.row_count
    print(f"{arguments.output}  {rows.shape[0]} raw rows, {duration:.1f} time units")


if __name__ == "__main__":
    main()
