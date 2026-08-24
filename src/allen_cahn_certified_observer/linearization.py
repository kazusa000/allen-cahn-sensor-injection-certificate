"""Local incremental dynamics for the Allen–Cahn nonlinearity."""

from __future__ import annotations

import numpy as np

from .grid import AllenCahnGrid
from .solver import allen_cahn_rhs


def _checked_state(grid: AllenCahnGrid, state: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if values.shape != (grid.n,):
        raise ValueError(f"{name} must have shape {(grid.n,)}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def allen_cahn_jacobian(
    grid: AllenCahnGrid, nu: float, base_state: np.ndarray
) -> np.ndarray:
    """Return the exact Jacobian of ``F_h`` at ``base_state``."""

    base = _checked_state(grid, base_state, "base_state")
    return nu * grid.laplacian + np.eye(grid.n) - np.diag(3.0 * base**2)


def incremental_remainder(
    grid: AllenCahnGrid,
    nu: float,
    base_state: np.ndarray,
    increment: np.ndarray,
) -> np.ndarray:
    """Return ``F(base+increment)-F(base)-J(base)increment``."""

    base = _checked_state(grid, base_state, "base_state")
    delta = _checked_state(grid, increment, "increment")
    return -3.0 * base * delta**2 - delta**3


def local_incremental_rhs(
    grid: AllenCahnGrid,
    nu: float,
    base_state: np.ndarray,
    increment: np.ndarray,
) -> np.ndarray:
    """Return the exact nonlinear increment RHS ``F(base+e)-F(base)``."""

    base = _checked_state(grid, base_state, "base_state")
    delta = _checked_state(grid, increment, "increment")
    return allen_cahn_rhs(grid, nu, base + delta) - allen_cahn_rhs(grid, nu, base)
