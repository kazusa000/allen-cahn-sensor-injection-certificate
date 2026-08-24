"""Small CPU-only learned-correction baseline for R5-D."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .grid import AllenCahnGrid
from .observer import ObserverRollout
from .solver import allen_cahn_rhs


@dataclass(frozen=True)
class StateConditionedLinearCorrection:
    """Causal correction with an optional fixed physical-gain safeguard."""

    grid: AllenCahnGrid
    observation_matrix: np.ndarray
    weights: np.ndarray
    baseline_gain: float = 0.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.observation_matrix, dtype=float)
        weight_array = np.asarray(self.weights, dtype=float)
        q = matrix.shape[0] if matrix.ndim == 2 else -1
        expected = (self.grid.n, 2 * q)
        if matrix.ndim != 2 or matrix.shape[1] != self.grid.n or q < 1:
            raise ValueError("observation_matrix must have shape (q, n)")
        if weight_array.shape != expected:
            raise ValueError(
                f"weights must have shape {expected}, got {weight_array.shape}"
            )
        if not np.isfinite(self.baseline_gain) or self.baseline_gain < 0.0:
            raise ValueError("baseline_gain must be a non-negative finite scalar")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(weight_array)):
            raise ValueError("correction parameters must be finite")
        object.__setattr__(self, "observation_matrix", matrix)
        object.__setattr__(self, "weights", weight_array)

    def features(self, estimate: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        state = np.asarray(estimate, dtype=float)
        observed = np.asarray(measurement, dtype=float)
        innovation = observed - self.observation_matrix @ state
        state_scale = np.sqrt(self.grid.h * np.dot(state, state))
        return np.concatenate((innovation, state_scale * innovation))

    def correction(self, estimate: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        return self.weights @ self.features(estimate, measurement)

    def rhs(
        self,
        nu: float,
        estimate: np.ndarray,
        measurement: np.ndarray,
    ) -> np.ndarray:
        state = np.asarray(estimate, dtype=float)
        innovation = np.asarray(measurement, dtype=float) - (
            self.observation_matrix @ state
        )
        physical_baseline = (
            self.baseline_gain * (self.observation_matrix.T @ innovation) / self.grid.h
        )
        return (
            allen_cahn_rhs(self.grid, nu, state)
            + physical_baseline
            + self.correction(state, measurement)
        )


def fit_state_conditioned_linear_correction(
    grid: AllenCahnGrid,
    observation_matrix: np.ndarray,
    estimates: np.ndarray,
    measurements: np.ndarray,
    target_corrections: np.ndarray,
    *,
    ridge: float = 1e-8,
    baseline_gain: float = 0.0,
) -> StateConditionedLinearCorrection:
    """Fit a legal causal correction by ridge least squares.

    ``target_corrections`` is generated offline from the reference state. The
    returned model itself only consumes ``estimate`` and ``measurement``.
    """

    matrix = np.asarray(observation_matrix, dtype=float)
    estimate_array = np.asarray(estimates, dtype=float)
    measurement_array = np.asarray(measurements, dtype=float)
    target_array = np.asarray(target_corrections, dtype=float)
    if estimate_array.ndim != 2 or estimate_array.shape[1] != grid.n:
        raise ValueError("estimates must have shape (samples, n)")
    if (
        measurement_array.ndim != 2
        or measurement_array.shape[0] != estimate_array.shape[0]
    ):
        raise ValueError("measurements must have the same sample count as estimates")
    if target_array.shape != estimate_array.shape:
        raise ValueError("target_corrections must have the same shape as estimates")
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a positive finite scalar")
    if baseline_gain < 0.0 or not np.isfinite(baseline_gain):
        raise ValueError("baseline_gain must be a non-negative finite scalar")

    features = []
    residual_targets = []
    for index, (estimate, measurement) in enumerate(
        zip(estimate_array, measurement_array, strict=True)
    ):
        innovation = measurement - matrix @ estimate
        scale = np.sqrt(grid.h * np.dot(estimate, estimate))
        features.append(np.concatenate((innovation, scale * innovation)))
        residual_targets.append(
            target_array[index] - baseline_gain * (matrix.T @ innovation) / grid.h
        )
    design = np.asarray(features, dtype=float)
    residual_target_array = np.asarray(residual_targets, dtype=float)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ residual_target_array).T
    return StateConditionedLinearCorrection(grid, matrix, weights, baseline_gain)


def simulate_learned_correction(
    model: StateConditionedLinearCorrection,
    nu: float,
    truth_initial: np.ndarray,
    estimate_initial: np.ndarray,
    *,
    t_span: tuple[float, float] = (0.0, 1.0),
    output_times: np.ndarray | None = None,
    noise: Callable[[float], np.ndarray] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> ObserverRollout:
    """Roll out a fitted causal correction with offline-generated observations."""

    n = model.grid.n
    truth = np.asarray(truth_initial, dtype=float)
    estimate = np.asarray(estimate_initial, dtype=float)
    if truth.shape != (n,) or estimate.shape != (n,):
        raise ValueError(f"truth_initial and estimate_initial must have shape {(n,)}")
    times = (
        np.asarray(output_times, dtype=float)
        if output_times is not None
        else np.linspace(t_span[0], t_span[1], 101)
    )
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) < 0.0):
        raise ValueError(
            "output_times must be a sorted, non-empty one-dimensional array"
        )

    def rhs(time: float, combined_state: np.ndarray) -> np.ndarray:
        truth_state = combined_state[:n]
        estimate_state = combined_state[n:]
        measurement = model.observation_matrix @ truth_state
        if noise is not None:
            perturbation = np.asarray(noise(float(time)), dtype=float)
            if perturbation.shape != measurement.shape:
                raise ValueError("noise function returned the wrong shape")
            measurement = measurement + perturbation
        return np.concatenate(
            (
                allen_cahn_rhs(model.grid, nu, truth_state),
                model.rhs(nu, estimate_state, measurement),
            )
        )

    result = solve_ivp(
        rhs,
        t_span,
        np.concatenate((truth, estimate)),
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if result.y.shape != (2 * n, times.size):
        raise RuntimeError(
            "learned-correction rollout returned an incomplete trajectory"
        )
    trajectories = result.y.T
    truth_trajectory = trajectories[:, :n]
    estimate_trajectory = trajectories[:, n:]
    measurements = np.asarray(
        [model.observation_matrix @ value for value in truth_trajectory],
        dtype=float,
    )
    if noise is not None:
        measurements += np.asarray([noise(float(time)) for time in times], dtype=float)
    return ObserverRollout(
        times=times,
        truth=truth_trajectory,
        estimate=estimate_trajectory,
        measurements=measurements,
        solver_status=int(result.status),
        solver_message=str(result.message),
    )
