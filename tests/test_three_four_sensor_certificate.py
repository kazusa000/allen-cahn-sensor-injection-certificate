import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_three_four_sensor_certificate import (
    FOUR_SENSOR_TARGET_MARGIN,
    TOTAL_OBSERVATION_LENGTH,
    _signed_rate_summary,
    intervals_from_centers,
    select_four_sensor_geometry,
    select_three_sensor_geometry,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    local_average_matrix,
    mass_adjoint_injection,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)


def test_equal_width_intervals_preserve_total_observation_length() -> None:
    intervals = intervals_from_centers(np.asarray([0.2, 0.4, 0.6, 0.8]))

    assert np.sum(intervals[:, 1] - intervals[:, 0]) == pytest.approx(
        TOTAL_OBSERVATION_LENGTH
    )
    assert np.all(intervals[1:, 0] >= intervals[:-1, 1])


def test_three_sensor_mass_adjoint_has_rank_obstruction() -> None:
    grid = AllenCahnGrid(31)
    intervals = intervals_from_centers(np.asarray([0.2, 0.5, 0.8]))
    observation = local_average_matrix(grid, intervals)
    modal = unstable_modal_system(grid, 0.005, observation)

    assert modal.dimension == 4
    assert np.linalg.matrix_rank(mass_adjoint_injection(grid, observation)) == 3
    assert symmetric_allen_cahn_margin(grid, 0.005, observation) < 0.0


def test_four_interior_sensors_meet_frozen_global_target() -> None:
    grid = AllenCahnGrid(31)
    intervals = intervals_from_centers(np.asarray([0.2, 0.4, 0.6, 0.8]))
    observation = local_average_matrix(grid, intervals)

    margins = [
        symmetric_allen_cahn_margin(grid, nu, observation, gain=0.5)
        for nu in (0.005, 0.010, 0.020)
    ]

    assert min(margins) >= FOUR_SENSOR_TARGET_MARGIN


def test_frozen_linear_selection_is_reproducible() -> None:
    pytest.importorskip("cvxpy")

    three_name, three_intervals, _ = select_three_sensor_geometry()
    four_name, gain, four_intervals, _ = select_four_sensor_geometry()

    assert three_name == "wide-interior"
    assert np.mean(three_intervals, axis=1) == pytest.approx([0.2, 0.5, 0.8])
    assert four_name == "interior-fifths"
    assert gain == pytest.approx(0.5)
    assert np.mean(four_intervals, axis=1) == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_signed_rate_summary_checks_requested_margin() -> None:
    summary = _signed_rate_summary(np.asarray([0.02, 0.05, 0.10]), 0.01)

    assert summary["min"] == pytest.approx(0.02)
    assert summary["requested_margin_min"] == pytest.approx(0.01)
    assert summary["requested_rate_fraction"] == pytest.approx(1.0)
