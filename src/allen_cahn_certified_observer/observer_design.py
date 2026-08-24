"""Linear feasibility and modal output-injection design for R5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_continuous_lyapunov
from scipy.signal import place_poles

from .grid import AllenCahnGrid
from .linearization import allen_cahn_jacobian

Array = np.ndarray


def _checked_observation(grid: AllenCahnGrid, observation_matrix: Array) -> Array:
    matrix = np.asarray(observation_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != grid.n:
        raise ValueError("observation_matrix must have shape (q, n)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observation_matrix must be finite")
    return matrix


@dataclass(frozen=True)
class UnstableModalSystem:
    """Positive linearized Allen--Cahn modes and their measurements."""

    eigenvalues: Array
    modes: Array
    observed_modes: Array
    observability_matrix: Array
    observability_rank: int
    observability_min_singular_value: float
    observability_condition: float

    @property
    def dimension(self) -> int:
        return int(self.eigenvalues.size)


@dataclass(frozen=True)
class ModalInjectionDesign:
    """A lifted output injection and its low-mode contraction metric."""

    method: str
    injection_matrix: Array
    modal_gain: Array
    modal_metric: Array
    modal_contraction_rate: float
    modal_metric_condition: float
    transform_condition: float
    closed_loop_spectral_abscissa: float
    mass_scaled_gain_norm: float
    solver_status: str


def linearized_error_matrix(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    injection_matrix: Array,
) -> Array:
    """Return ``A + F'(0) - B C`` for the declared observer sign convention."""

    observation = _checked_observation(grid, observation_matrix)
    injection = np.asarray(injection_matrix, dtype=float)
    if injection.shape != (grid.n, observation.shape[0]):
        raise ValueError("injection_matrix must have shape (n, q)")
    if not np.all(np.isfinite(injection)):
        raise ValueError("injection_matrix must be finite")
    return (
        allen_cahn_jacobian(grid, nu, np.zeros(grid.n))
        - injection @ observation
    )


def mass_adjoint_injection(
    grid: AllenCahnGrid, observation_matrix: Array, gain: float = 1.0
) -> Array:
    """Return the scalar physical-mass adjoint injection ``gain * C.T / h``."""

    observation = _checked_observation(grid, observation_matrix)
    value = float(gain)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("gain must be a non-negative finite scalar")
    return value * observation.T / grid.h


def symmetric_allen_cahn_margin(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    *,
    gain: float = 1.0,
) -> float:
    """Global semidiscrete contraction margin for mass-adjoint injection.

    Allen--Cahn's cubic contribution is non-positive in the error energy, so a
    positive return value certifies global error decay in the physical mass norm.
    """

    observation = _checked_observation(grid, observation_matrix)
    closed_loop = linearized_error_matrix(
        grid,
        nu,
        observation,
        mass_adjoint_injection(grid, observation, gain),
    )
    return float(-np.max(np.linalg.eigvalsh(closed_loop)))


def unstable_modal_system(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    *,
    tolerance: float = 1e-10,
) -> UnstableModalSystem:
    """Extract every positive mode and its finite-horizon observability matrix."""

    observation = _checked_observation(grid, observation_matrix)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    linearization = allen_cahn_jacobian(grid, nu, np.zeros(grid.n))
    eigenvalues, modes = np.linalg.eigh(linearization)
    indices = np.flatnonzero(eigenvalues > tolerance)
    indices = indices[np.argsort(eigenvalues[indices])[::-1]]
    unstable_values = eigenvalues[indices]
    unstable_modes = modes[:, indices]
    observed_modes = observation @ unstable_modes
    if indices.size == 0:
        observability = np.empty((0, 0), dtype=float)
        singular_values = np.empty(0, dtype=float)
        rank = 0
        minimum = float("inf")
        condition = 1.0
    else:
        diagonal = np.diag(unstable_values)
        observability = np.vstack(
            [
                observed_modes @ np.linalg.matrix_power(diagonal, power)
                for power in range(indices.size)
            ]
        )
        singular_values = np.linalg.svd(observability, compute_uv=False)
        rank = int(np.linalg.matrix_rank(observability, tol=tolerance))
        minimum = float(np.min(singular_values))
        condition = float(np.max(singular_values) / minimum)
    return UnstableModalSystem(
        eigenvalues=unstable_values,
        modes=unstable_modes,
        observed_modes=observed_modes,
        observability_matrix=observability,
        observability_rank=rank,
        observability_min_singular_value=minimum,
        observability_condition=condition,
    )


def _metric_diagnostics(
    modal_matrix: Array, metric: Array
) -> tuple[float, float, float]:
    symmetric_metric = 0.5 * (metric + metric.T)
    metric_eigenvalues = np.linalg.eigvalsh(symmetric_metric)
    if np.min(metric_eigenvalues) <= 0.0:
        raise RuntimeError("modal metric is not positive definite")
    derivative = modal_matrix.T @ symmetric_metric + symmetric_metric @ modal_matrix
    cholesky = np.linalg.cholesky(symmetric_metric)
    inverse = np.linalg.inv(cholesky)
    normalized = inverse @ derivative @ inverse.T
    contraction_rate = float(-0.5 * np.max(np.linalg.eigvalsh(normalized)))
    condition = float(np.max(metric_eigenvalues) / np.min(metric_eigenvalues))
    return contraction_rate, condition, float(np.sqrt(condition))


def _build_design(
    method: str,
    grid: AllenCahnGrid,
    nu: float,
    observation: Array,
    modal: UnstableModalSystem,
    modal_gain: Array,
    modal_metric: Array,
    solver_status: str,
) -> ModalInjectionDesign:
    injection = modal.modes @ modal_gain
    closed_loop = linearized_error_matrix(grid, nu, observation, injection)
    modal_matrix = np.diag(modal.eigenvalues) - modal_gain @ modal.observed_modes
    rate, condition, transform_condition = _metric_diagnostics(
        modal_matrix, modal_metric
    )
    return ModalInjectionDesign(
        method=method,
        injection_matrix=injection,
        modal_gain=modal_gain,
        modal_metric=modal_metric,
        modal_contraction_rate=rate,
        modal_metric_condition=condition,
        transform_condition=transform_condition,
        closed_loop_spectral_abscissa=float(
            np.max(np.real(np.linalg.eigvals(closed_loop)))
        ),
        mass_scaled_gain_norm=float(
            np.sqrt(grid.h) * np.linalg.norm(injection, ord=2)
        ),
        solver_status=solver_status,
    )


def pole_placement_modal_injection(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    *,
    slowest_pole: float = -0.4,
    pole_spacing: float = 0.2,
) -> ModalInjectionDesign:
    """Place all unstable low-mode observer poles and lift the gain to the grid."""

    observation = _checked_observation(grid, observation_matrix)
    modal = unstable_modal_system(grid, nu, observation)
    if modal.dimension == 0:
        raise ValueError("the linearization has no unstable modes")
    if modal.observability_rank != modal.dimension:
        raise RuntimeError("unstable modes are not observable")
    if slowest_pole >= 0.0 or pole_spacing <= 0.0:
        raise ValueError("poles must be strictly negative and distinctly spaced")
    poles = slowest_pole - pole_spacing * np.arange(modal.dimension, dtype=float)
    diagonal = np.diag(modal.eigenvalues)
    gain = place_poles(
        diagonal.T,
        modal.observed_modes.T,
        poles,
        method="YT",
    ).gain_matrix.T
    modal_matrix = diagonal - gain @ modal.observed_modes
    metric = solve_continuous_lyapunov(modal_matrix.T, -np.eye(modal.dimension))
    return _build_design(
        "pole-placement",
        grid,
        nu,
        observation,
        modal,
        gain,
        metric,
        "analytic",
    )


def riccati_modal_injection(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    *,
    measurement_weight: float = 1.0,
) -> ModalInjectionDesign:
    """Construct a dual continuous-time Riccati observer gain on unstable modes."""

    observation = _checked_observation(grid, observation_matrix)
    weight = float(measurement_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("measurement_weight must be positive and finite")
    modal = unstable_modal_system(grid, nu, observation)
    if modal.dimension == 0:
        raise ValueError("the linearization has no unstable modes")
    if modal.observability_rank != modal.dimension:
        raise RuntimeError("unstable modes are not observable")
    diagonal = np.diag(modal.eigenvalues)
    riccati = solve_continuous_are(
        diagonal.T,
        modal.observed_modes.T,
        np.eye(modal.dimension),
        weight * np.eye(observation.shape[0]),
    )
    gain = riccati @ modal.observed_modes.T / weight
    modal_matrix = diagonal - gain @ modal.observed_modes
    metric = solve_continuous_lyapunov(modal_matrix.T, -np.eye(modal.dimension))
    return _build_design(
        f"riccati-{weight:g}",
        grid,
        nu,
        observation,
        modal,
        gain,
        metric,
        "analytic",
    )


def lmi_modal_injection(
    grid: AllenCahnGrid,
    nu: float,
    observation_matrix: Array,
    *,
    decay_rate: float,
    metric_condition_bound: float = 256.0,
    solver: str = "CLARABEL",
) -> ModalInjectionDesign:
    """Solve the low-mode convex metric/output-injection inequality.

    CVXPY is an optional design dependency because the resulting fixed injection
    does not require CVXPY at deployment time.
    """

    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("lmi_modal_injection requires the 'design' extra") from error

    observation = _checked_observation(grid, observation_matrix)
    requested_rate = float(decay_rate)
    condition_bound = float(metric_condition_bound)
    if not np.isfinite(requested_rate) or requested_rate <= 0.0:
        raise ValueError("decay_rate must be positive and finite")
    if not np.isfinite(condition_bound) or condition_bound <= 1.0:
        raise ValueError("metric_condition_bound must be greater than one")
    modal = unstable_modal_system(grid, nu, observation)
    if modal.dimension == 0:
        raise ValueError("the linearization has no unstable modes")
    if modal.observability_rank != modal.dimension:
        raise RuntimeError("unstable modes are not observable")

    dimension = modal.dimension
    diagonal = np.diag(modal.eigenvalues)
    metric = cp.Variable((dimension, dimension), symmetric=True)
    weighted_gain = cp.Variable((dimension, observation.shape[0]))
    derivative = (
        diagonal.T @ metric
        + metric @ diagonal
        - modal.observed_modes.T @ weighted_gain.T
        - weighted_gain @ modal.observed_modes
        + 2.0 * requested_rate * metric
    )
    identity = np.eye(dimension)
    problem = cp.Problem(
        cp.Minimize(cp.norm(weighted_gain, "fro")),
        [
            metric >> identity,
            metric << condition_bound * identity,
            derivative << -1e-8 * identity,
        ],
    )
    problem.solve(solver=solver)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"modal LMI was not solved: {problem.status}")
    metric_value = np.asarray(metric.value, dtype=float)
    metric_value = 0.5 * (metric_value + metric_value.T)
    weighted_gain_value = np.asarray(weighted_gain.value, dtype=float)
    gain = np.linalg.solve(metric_value, weighted_gain_value)
    return _build_design(
        f"lmi-{requested_rate:.8g}",
        grid,
        nu,
        observation,
        modal,
        gain,
        metric_value,
        str(problem.status),
    )


def finite_horizon_transient_amplification(
    matrix: Array,
    *,
    horizon: float = 1.0,
    sample_count: int = 41,
) -> tuple[float, float]:
    """Return sampled ``max ||exp(matrix*t)||_2`` and the maximizing time."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if horizon <= 0.0 or sample_count < 2:
        raise ValueError("horizon must be positive and sample_count at least two")
    best_value = -np.inf
    best_time = 0.0
    for time in np.linspace(0.0, horizon, sample_count):
        value = float(np.linalg.norm(expm(values * time), ord=2))
        if value > best_value:
            best_value = value
            best_time = float(time)
    return best_value, best_time


def normalized_modal_transform(
    grid: AllenCahnGrid,
    modal: UnstableModalSystem,
    modal_metric: Array,
) -> Array:
    """Lift a balanced square root of a modal metric to the full grid.

    Multiplying a Lyapunov metric by a positive scalar leaves its contraction
    inequality unchanged. The balancing used here makes the smallest and largest
    singular values reciprocal, avoiding an arbitrary overall scale in ``T_phi``.
    """

    metric = np.asarray(modal_metric, dtype=float)
    if metric.shape != (modal.dimension, modal.dimension):
        raise ValueError("modal_metric has the wrong shape")
    metric = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("modal_metric must be positive definite")
    scale = 1.0 / np.sqrt(float(np.min(eigenvalues) * np.max(eigenvalues)))
    modal_transform = (
        eigenvectors
        @ np.diag(np.sqrt(scale * eigenvalues))
        @ eigenvectors.T
    )
    projector = modal.modes @ modal.modes.T
    return (
        np.eye(grid.n)
        - projector
        + modal.modes @ modal_transform @ modal.modes.T
    )
