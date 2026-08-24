import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)
from allen_cahn_certified_observer.solver import energy_derivative_identity


def test_discrete_energy_identity_matches_mass_norm() -> None:
    grid = AllenCahnGrid(31)
    state = 0.6 * np.sin(np.pi * grid.x) - 0.1 * np.sin(2.0 * np.pi * grid.x)

    derivative, mass_norm = energy_derivative_identity(grid, 0.01, state)

    assert derivative < 0.0
    assert np.isclose(derivative, mass_norm, rtol=0.0, atol=1e-14)


def test_reference_solution_has_decreasing_energy() -> None:
    grid = AllenCahnGrid(31)
    initial = 0.5 * np.sin(np.pi * grid.x)
    solution = solve_allen_cahn(
        grid,
        0.01,
        initial,
        t_span=(0.0, 0.4),
        output_times=np.linspace(0.0, 0.4, 41),
        rtol=1e-10,
        atol=1e-12,
    )

    assert solution.solver_status == 0
    assert solution.states.shape == (41, grid.n)
    assert solution.energy_drop_max <= 1e-11
    assert solution.energies[-1] < solution.energies[0]
    assert np.all(np.isfinite(solution.states))


def test_dense_output_agrees_with_saved_endpoint() -> None:
    grid = AllenCahnGrid(15)
    initial = 0.25 * np.sin(np.pi * grid.x)
    solution = solve_allen_cahn(grid, 0.02, initial, t_span=(0.0, 0.1))

    assert np.allclose(
        solution.state_at(0.1), solution.states[-1], rtol=1e-10, atol=1e-12
    )
    assert np.allclose(
        allen_cahn_rhs(grid, 0.02, solution.states[0]),
        allen_cahn_rhs(grid, 0.02, initial),
    )
    assert np.isclose(allen_cahn_energy(grid, 0.02, initial), solution.energies[0])
