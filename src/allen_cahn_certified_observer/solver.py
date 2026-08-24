"""High-accuracy reference integration and discrete energy diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from .grid import AllenCahnGrid

Array = np.ndarray
Rhs = Callable[[float, Array], Array]


def _validate_viscosity(nu: float) -> float:
    value = float(nu)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("nu must be a positive finite scalar")
    return value


def allen_cahn_rhs(grid: AllenCahnGrid, nu: float, state: Array) -> Array:
    """Return ``nu * D2 u + u - u**3`` on the interior grid."""

    viscosity = _validate_viscosity(nu)
    values = np.asarray(state, dtype=float)
    if values.shape != (grid.n,):
        raise ValueError(f"expected shape {(grid.n,)}, got {values.shape}")
    return viscosity * (grid.laplacian @ values) + values - values**3


def allen_cahn_energy(grid: AllenCahnGrid, nu: float, state: Array) -> float:
    """Compute the mass-weighted discrete Allen–Cahn energy."""

    viscosity = _validate_viscosity(nu)
    values = np.asarray(state, dtype=float)
    if values.shape != (grid.n,):
        raise ValueError(f"expected shape {(grid.n,)}, got {values.shape}")
    full = grid.with_boundary(values)
    gradient = np.diff(full) / grid.h
    potential = 0.25 * (values**2 - 1.0) ** 2
    return float(
        grid.h * (0.5 * viscosity * np.dot(gradient, gradient) + potential.sum())
    )


def energy_derivative_identity(
    grid: AllenCahnGrid, nu: float, state: Array
) -> tuple[float, float]:
    """Return ``dE/dt`` from the chain rule and ``-||u_t||_M^2``."""

    velocity = allen_cahn_rhs(grid, nu, state)
    derivative = float(-grid.h * np.dot(velocity, velocity))
    mass_norm = float(-velocity @ (grid.mass @ velocity))
    return derivative, mass_norm


@dataclass(frozen=True)
class AllenCahnSolution:
    """Dense output and sampled diagnostics for a reference trajectory."""

    grid: AllenCahnGrid
    nu: float
    times: Array
    states: Array
    energies: Array
    solver_status: int
    solver_message: str
    dense: OdeSolution | None = None

    @property
    def energy_drop_max(self) -> float:
        """Largest positive energy increment in the saved samples."""

        if self.energies.size < 2:
            return 0.0
        return float(np.max(np.diff(self.energies)))

    def state_at(self, time: float) -> Array:
        if self.dense is None:
            raise RuntimeError("dense output was not requested")
        value = np.asarray(self.dense(float(time)), dtype=float)
        return value.reshape(self.grid.n)


def solve_allen_cahn(
    grid: AllenCahnGrid,
    nu: float,
    initial_state: Array,
    *,
    t_span: tuple[float, float] = (0.0, 1.0),
    output_times: Array | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    dense_output: bool = True,
) -> AllenCahnSolution:
    """Integrate the reference semi-discrete Allen–Cahn equation with DOP853."""

    viscosity = _validate_viscosity(nu)
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (grid.n,):
        raise ValueError(f"expected shape {(grid.n,)}, got {initial.shape}")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain finite values")
    if len(t_span) != 2 or not t_span[1] > t_span[0]:
        raise ValueError("t_span must be an increasing pair")
    if rtol <= 0.0 or atol <= 0.0 or max_step <= 0.0:
        raise ValueError("rtol, atol, and max_step must be positive")

    sample_times = (
        np.asarray(output_times, dtype=float)
        if output_times is not None
        else np.linspace(t_span[0], t_span[1], 101)
    )
    if sample_times.ndim != 1 or sample_times.size == 0:
        raise ValueError("output_times must be a non-empty one-dimensional array")
    if (
        np.any(np.diff(sample_times) < 0.0)
        or sample_times[0] < t_span[0]
        or sample_times[-1] > t_span[1]
    ):
        raise ValueError("output_times must be sorted and lie inside t_span")

    result = solve_ivp(
        lambda time, state: allen_cahn_rhs(grid, viscosity, state),
        t_span,
        initial,
        method="DOP853",
        t_eval=sample_times,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
    )
    if result.y.shape != (grid.n, sample_times.size):
        raise RuntimeError("reference solver returned an incomplete trajectory")
    states = np.asarray(result.y.T, dtype=float)
    energies = np.asarray(
        [allen_cahn_energy(grid, viscosity, state) for state in states]
    )
    return AllenCahnSolution(
        grid=grid,
        nu=viscosity,
        times=sample_times,
        states=states,
        energies=energies,
        solver_status=int(result.status),
        solver_message=str(result.message),
        dense=result.sol,
    )
