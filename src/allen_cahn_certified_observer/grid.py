"""Finite-difference grid primitives for the one-dimensional Allen–Cahn model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllenCahnGrid:
    """Interior grid with homogeneous Dirichlet boundary values."""

    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 3:
            raise ValueError("n must be an integer at least 3")

    @property
    def h(self) -> float:
        return 1.0 / (self.n + 1)

    @property
    def x(self) -> np.ndarray:
        return self.h * np.arange(1, self.n + 1, dtype=float)

    @property
    def x_with_boundary(self) -> np.ndarray:
        return self.h * np.arange(self.n + 2, dtype=float)

    @property
    def mass(self) -> np.ndarray:
        return self.h * np.eye(self.n)

    @property
    def laplacian(self) -> np.ndarray:
        diagonal = -2.0 * np.ones(self.n)
        off_diagonal = np.ones(self.n - 1)
        return (
            np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
        ) / self.h**2

    def with_boundary(self, interior: np.ndarray) -> np.ndarray:
        values = np.asarray(interior, dtype=float)
        if values.shape != (self.n,):
            raise ValueError(f"expected shape {(self.n,)}, got {values.shape}")
        return np.concatenate(([0.0], values, [0.0]))

    def interior(self, full: np.ndarray) -> np.ndarray:
        values = np.asarray(full, dtype=float)
        if values.shape != (self.n + 2,):
            raise ValueError(f"expected shape {(self.n + 2,)}, got {values.shape}")
        if not np.allclose(values[[0, -1]], 0.0, atol=0.0, rtol=0.0):
            raise ValueError("boundary values must be zero")
        return values[1:-1]
