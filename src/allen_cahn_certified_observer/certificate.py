"""Offline-only fiber certificates and their numerical audit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

Array = np.ndarray
CertificateMap = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class CertificateAudit:
    """Numerical diagnostics for a certificate on a declared sample set."""

    sample_count: int
    max_zero_fiber_residual: float
    max_direction_residual: float
    min_jacobian_singular_value: float
    max_jacobian_singular_value: float

    @property
    def passed_basic_constraints(self) -> bool:
        return (
            self.max_zero_fiber_residual <= 1e-10
            and self.max_direction_residual <= 1e-10
            and self.min_jacobian_singular_value > 0.0
            and np.isfinite(self.max_jacobian_singular_value)
        )


@dataclass(frozen=True)
class IdentityCertificate:
    """The identity map, retained as the zero-complexity certificate baseline."""

    def __call__(self, state: Array, error: Array) -> Array:
        del state
        return np.asarray(error, dtype=float)


@dataclass(frozen=True)
class NullspaceCertificate:
    """A small state-conditioned map that preserves the measured direction.

    Let ``N`` span ``ker(C)``. The map is

    ``T(u,e) = e + N diag(amplitude * tanh(scale * N.T @ u)) N.T @ e``.

    It is an audit/training scaffold, not a claim that this family is optimal.
    The nullspace construction enforces ``C T = C e`` up to numerical precision.
    """

    observation_matrix: Array
    amplitude: float = 0.05
    state_scale: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.observation_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("observation_matrix must have shape (q, n)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("observation_matrix must be finite")
        if not np.isfinite(self.amplitude) or not np.isfinite(self.state_scale):
            raise ValueError("certificate parameters must be finite")
        if self.state_scale < 0.0:
            raise ValueError("state_scale must be non-negative")
        basis = null_space(matrix)
        if basis.shape[1] == 0:
            raise ValueError("observation matrix has no nullspace")
        object.__setattr__(self, "observation_matrix", matrix)
        object.__setattr__(self, "_null_basis", basis)

    @property
    def null_basis(self) -> Array:
        return self._null_basis

    def __call__(self, state: Array, error: Array) -> Array:
        base = np.asarray(state, dtype=float)
        delta = np.asarray(error, dtype=float)
        if base.shape != (self.observation_matrix.shape[1],):
            raise ValueError("state has the wrong shape")
        if delta.shape != base.shape:
            raise ValueError("error has the wrong shape")
        coordinates = self._null_basis.T @ base
        error_coordinates = self._null_basis.T @ delta
        gate = self.amplitude * np.tanh(self.state_scale * coordinates)
        return delta + self._null_basis @ (gate * error_coordinates)


def _finite_difference_jacobian(
    certificate: CertificateMap,
    state: Array,
    error: Array,
    step: float,
) -> Array:
    dimension = error.size
    jacobian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        perturbation = np.zeros(dimension, dtype=float)
        perturbation[column] = step
        plus = certificate(state, error + perturbation)
        minus = certificate(state, error - perturbation)
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def audit_certificate(
    certificate: CertificateMap,
    observation_matrix: Array,
    states: Array,
    errors: Array,
    *,
    finite_difference_step: float = 1e-6,
) -> CertificateAudit:
    """Audit fiber, direction, and local invertibility constraints on samples."""

    matrix = np.asarray(observation_matrix, dtype=float)
    state_samples = np.asarray(states, dtype=float)
    error_samples = np.asarray(errors, dtype=float)
    if state_samples.ndim != 2 or error_samples.ndim != 2:
        raise ValueError("states and errors must be two-dimensional")
    if state_samples.shape != error_samples.shape or state_samples.shape[0] == 0:
        raise ValueError("states and errors must have the same non-empty shape")
    if matrix.ndim != 2 or matrix.shape[1] != state_samples.shape[1]:
        raise ValueError("observation_matrix and samples have incompatible shapes")
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")

    zero_fiber = []
    direction = []
    minimum_singular = []
    maximum_singular = []
    for state, error in zip(state_samples, error_samples, strict=True):
        transformed = np.asarray(certificate(state, error), dtype=float)
        zero = np.asarray(certificate(state, np.zeros_like(error)), dtype=float)
        if transformed.shape != error.shape or zero.shape != error.shape:
            raise ValueError("certificate returned an incompatible shape")
        zero_fiber.append(float(np.linalg.norm(zero)))
        direction.append(float(np.linalg.norm(matrix @ transformed - matrix @ error)))
        singular_values = np.linalg.svd(
            _finite_difference_jacobian(
                certificate, state, error, finite_difference_step
            ),
            compute_uv=False,
        )
        minimum_singular.append(float(np.min(singular_values)))
        maximum_singular.append(float(np.max(singular_values)))
    return CertificateAudit(
        sample_count=state_samples.shape[0],
        max_zero_fiber_residual=max(zero_fiber),
        max_direction_residual=max(direction),
        min_jacobian_singular_value=min(minimum_singular),
        max_jacobian_singular_value=max(maximum_singular),
    )
