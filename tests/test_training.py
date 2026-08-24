import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    fit_state_conditioned_linear_correction,
    local_average_matrix,
    simulate_learned_correction,
)


def test_causal_correction_has_zero_output_at_zero_innovation() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35], [0.65, 0.75]]))
    estimates = np.zeros((12, grid.n))
    measurements = np.zeros((12, matrix.shape[0]))
    targets = np.zeros_like(estimates)
    model = fit_state_conditioned_linear_correction(
        grid, matrix, estimates, measurements, targets
    )

    state = 0.2 * np.sin(np.pi * grid.x)
    assert np.allclose(model.correction(state, matrix @ state), 0.0)


def test_causal_correction_recovers_a_synthetic_linear_rule() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35], [0.65, 0.75]]))
    rng = np.random.Generator(np.random.PCG64DXSM(1234))
    estimates = rng.normal(size=(128, grid.n)) * 0.1
    innovations = rng.normal(size=(128, matrix.shape[0])) * 0.05
    measurements = estimates @ matrix.T + innovations
    features = np.concatenate(
        (
            innovations,
            innovations * np.sqrt(grid.h * np.sum(estimates**2, axis=1))[:, None],
        ),
        axis=1,
    )
    true_weights = rng.normal(size=(grid.n, 2 * matrix.shape[0])) * 0.1
    targets = features @ true_weights.T

    model = fit_state_conditioned_linear_correction(
        grid, matrix, estimates, measurements, targets, ridge=1e-10
    )

    predictions = np.asarray(
        [
            model.correction(state, measurement)
            for state, measurement in zip(estimates, measurements, strict=True)
        ]
    )
    assert np.linalg.norm(predictions - targets) / np.linalg.norm(targets) < 1e-7


def test_learned_rollout_is_finite_and_uses_measurement_noise() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35], [0.65, 0.75]]))
    samples = np.zeros((8, grid.n))
    measurements = np.zeros((8, matrix.shape[0]))
    targets = np.zeros_like(samples)
    model = fit_state_conditioned_linear_correction(
        grid, matrix, samples, measurements, targets
    )
    truth = 0.2 * np.sin(np.pi * grid.x)
    estimate = truth + 0.1 * np.sin(2.0 * np.pi * grid.x)
    rollout = simulate_learned_correction(
        model,
        0.01,
        truth,
        estimate,
        output_times=np.linspace(0.0, 0.2, 5),
        noise=lambda _time: np.array([0.001, -0.001]),
    )
    assert rollout.solver_status == 0
    assert np.all(np.isfinite(rollout.truth))
    assert np.all(np.isfinite(rollout.estimate))
    assert np.allclose(rollout.measurements[0], matrix @ truth + [0.001, -0.001])
