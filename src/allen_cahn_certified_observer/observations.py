"""Causal fixed-width local-average observation operators."""

from __future__ import annotations

import numpy as np

from .grid import AllenCahnGrid


def _hat_integral(grid: AllenCahnGrid, node: int, left: float, right: float) -> float:
    """Integrate the piecewise-linear hat at interior node ``node``."""

    x_node = node * grid.h
    support_left = (node - 1) * grid.h
    support_right = (node + 1) * grid.h
    a = max(float(left), support_left)
    b = min(float(right), support_right)
    if b <= a:
        return 0.0

    total = 0.0
    midpoint = min(max(x_node, a), b)
    if midpoint > a:
        # Integral of (x - support_left) / h.
        total += ((midpoint - support_left) ** 2 - (a - support_left) ** 2) / (
            2.0 * grid.h
        )
    if b > midpoint:
        # Integral of (support_right - x) / h.
        total += ((support_right - midpoint) ** 2 - (support_right - b) ** 2) / (
            2.0 * grid.h
        )
    return total


def local_average_matrix(grid: AllenCahnGrid, intervals: np.ndarray) -> np.ndarray:
    """Build exact hat-function averages over fixed physical intervals.

    ``intervals`` has shape ``(q, 2)`` and uses the physical domain ``[0, 1]``.
    Boundary values are fixed to zero and are therefore not columns of the matrix.
    """

    bounds = np.asarray(intervals, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
        raise ValueError("intervals must have shape (q, 2) with q > 0")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("interval endpoints must be finite")
    if (
        np.any(bounds[:, 0] < 0.0)
        or np.any(bounds[:, 1] > 1.0)
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("intervals must satisfy 0 <= left < right <= 1")

    matrix = np.empty((bounds.shape[0], grid.n), dtype=float)
    for row, (left, right) in enumerate(bounds):
        width = right - left
        matrix[row] = [
            _hat_integral(grid, node, left, right) / width
            for node in range(1, grid.n + 1)
        ]
    return matrix
