import numpy as np
import pytest

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    lmi_modal_injection,
    linearized_error_matrix,
    local_average_matrix,
    mass_adjoint_injection,
    normalized_modal_transform,
    pole_placement_modal_injection,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)
from allen_cahn_certified_observer.solver import allen_cahn_rhs


TWO_SENSOR_INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]])
FIVE_SENSOR_INTERVALS = np.array(
    [
        [0.08, 0.12],
        [0.28, 0.32],
        [0.48, 0.52],
        [0.68, 0.72],
        [0.88, 0.92],
    ]
)


def test_five_sensor_mass_adjoint_has_positive_global_margin() -> None:
    expected = {
        0.005: 0.224483,
        0.010: 1.270446,
        0.020: 2.326218,
    }
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, FIVE_SENSOR_INTERVALS)

    for nu, target in expected.items():
        margin = symmetric_allen_cahn_margin(grid, nu, matrix)
        assert margin == pytest.approx(target, abs=5e-7)


def test_two_sensor_adjoint_cannot_remove_all_weak_diffusion_modes() -> None:
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
    unstable = unstable_modal_system(grid, 0.005, matrix)
    closed_loop = linearized_error_matrix(
        grid, 0.005, matrix, mass_adjoint_injection(grid, matrix, gain=1.0)
    )

    assert unstable.dimension == 4
    assert np.linalg.matrix_rank(mass_adjoint_injection(grid, matrix)) == 2
    assert np.max(np.linalg.eigvalsh(closed_loop)) > 0.0


@pytest.mark.parametrize(
    ("nu", "expected_dimension"), [(0.005, 4), (0.010, 3), (0.020, 2)]
)
def test_two_sensors_are_dynamically_observable_on_unstable_modes(
    nu: float, expected_dimension: int
) -> None:
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
    unstable = unstable_modal_system(grid, nu, matrix)

    assert unstable.dimension == expected_dimension
    assert unstable.observability_rank == expected_dimension
    assert unstable.observability_min_singular_value > 0.0


def test_modal_pole_placement_stabilizes_the_full_linearization() -> None:
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
    design = pole_placement_modal_injection(grid, 0.005, matrix)

    assert design.closed_loop_spectral_abscissa < 0.0
    assert design.modal_contraction_rate > 0.0
    assert design.mass_scaled_gain_norm < 10.0


def test_lmi_design_meets_requested_rate_with_declared_condition_bound() -> None:
    pytest.importorskip("cvxpy")
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
    rate = 0.1 * 0.005 * np.pi**2
    design = lmi_modal_injection(
        grid,
        0.005,
        matrix,
        decay_rate=rate,
        metric_condition_bound=256.0,
    )

    assert design.closed_loop_spectral_abscissa < 0.0
    assert design.modal_contraction_rate >= rate - 1e-7
    assert design.modal_metric_condition <= 256.0 + 1e-5

    modal = unstable_modal_system(grid, 0.005, matrix)
    transform = normalized_modal_transform(grid, modal, design.modal_metric)
    singular_values = np.linalg.svd(transform, compute_uv=False)
    assert singular_values[-1] * singular_values[0] == pytest.approx(1.0, abs=1e-10)
    assert singular_values[0] / singular_values[-1] == pytest.approx(
        design.transform_condition, rel=1e-8
    )


def test_general_output_injection_is_zero_at_zero_innovation() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
    injection = np.ones((grid.n, matrix.shape[0]))
    observer = CausalOutputInjection(grid, 0.01, matrix, injection)
    state = 0.2 * np.sin(np.pi * grid.x)

    assert np.allclose(
        observer.rhs(state, matrix @ state), allen_cahn_rhs(grid, 0.01, state)
    )
