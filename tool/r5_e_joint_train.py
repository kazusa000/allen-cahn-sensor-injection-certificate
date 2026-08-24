"""R5-E GPU pilot: jointly fit a causal correction and offline certificate.

The correction network receives only the estimated field, current measurement,
innovation, known viscosity, and estimated-field mass norm. The certificate
network is offline-only and is trained against the declared nullspace scaffold;
its fiber and direction constraints are enforced by construction.

This runner is deliberately separate from the CPU R5-D smoke runner. It is a
pilot for the multi-seed GPU gate, not a theorem or a claim of global stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    NullspaceCertificate,
    allen_cahn_energy,
    allen_cahn_rhs,
    generate_pilot_cases,
    local_average_matrix,
    simulate_causal_nudging,
)

INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]], dtype=float)
OUTPUT_TIMES = np.linspace(0.0, 1.0, 51)
BASELINE_GAIN = 0.02


@dataclass(frozen=True)
class SampleSet:
    features: np.ndarray
    correction_targets: np.ndarray
    states: np.ndarray
    errors: np.ndarray
    certificate_targets: np.ndarray


def _feature_rows(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    estimates: np.ndarray,
    measurements: np.ndarray,
    nus: np.ndarray,
) -> np.ndarray:
    innovations = measurements - estimates @ matrix.T
    scales = np.sqrt(grid.h * np.sum(estimates**2, axis=1))
    viscosity = (nus - 0.01) / 0.01
    return np.concatenate(
        (
            estimates,
            measurements,
            innovations,
            viscosity[:, None],
            scales[:, None],
        ),
        axis=1,
    )


def _rhs_rows(grid: AllenCahnGrid, nus: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.asarray(
        [allen_cahn_rhs(grid, float(nu), state) for nu, state in zip(nus, states)]
    )


def _collect_samples(
    cases: list[object], grid: AllenCahnGrid, matrix: np.ndarray
) -> SampleSet:
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    nus: list[float] = []
    for case in cases:
        truth = case.initial_truth(grid)
        estimate = case.initial_estimate(grid)
        rollout = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.0),
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        estimates.extend(rollout.estimate)
        measurements.extend(rollout.measurements)
        truths.extend(rollout.truth)
        nus.extend([case.nu] * OUTPUT_TIMES.size)

    estimate_array = np.asarray(estimates, dtype=float)
    measurement_array = np.asarray(measurements, dtype=float)
    truth_array = np.asarray(truths, dtype=float)
    nu_array = np.asarray(nus, dtype=float)
    innovation = measurement_array - estimate_array @ matrix.T
    full_target = _rhs_rows(grid, nu_array, truth_array) - _rhs_rows(
        grid, nu_array, estimate_array
    )
    baseline = BASELINE_GAIN * (innovation @ matrix) / grid.h
    certificate = NullspaceCertificate(matrix, amplitude=0.05, state_scale=1.0)
    errors = truth_array - estimate_array
    certificate_targets = np.asarray(
        [certificate(state, error) for state, error in zip(truth_array, errors)]
    )
    return SampleSet(
        features=_feature_rows(
            grid, matrix, estimate_array, measurement_array, nu_array
        ),
        correction_targets=full_target - baseline,
        states=truth_array,
        errors=errors,
        certificate_targets=certificate_targets,
    )


def _split_cases(split: str, grid_size: int) -> list[object]:
    return [
        case
        for case in generate_pilot_cases()
        if case.split == split and case.n == grid_size
    ]


def _build_models(
    torch: object, input_dim: int, n: int, basis: np.ndarray
) -> tuple[object, object]:
    nn = torch.nn

    class CorrectionNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, n),
            )

        def forward(self, features: object) -> object:
            return self.network(features)

    class CertificateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            null_dimension = basis.shape[1]
            self.network = nn.Sequential(
                nn.Linear(2 * n, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, null_dimension),
            )
            self.register_buffer(
                "null_basis", torch.as_tensor(basis, dtype=torch.float32)
            )

        def forward(self, states: object, errors: object) -> object:
            coordinates = errors @ self.null_basis
            gates = 0.05 * torch.tanh(self.network(torch.cat((states, errors), dim=1)))
            return errors + (gates * coordinates) @ self.null_basis.T

    return CorrectionNet(), CertificateNet()


def _train_one(
    torch: object,
    sample_set: SampleSet,
    matrix: np.ndarray,
    grid: AllenCahnGrid,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    device: str,
) -> tuple[object, object, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    basis = NullspaceCertificate(matrix).null_basis
    correction, certificate = _build_models(
        torch, sample_set.features.shape[1], grid.n, basis
    )
    correction.to(device)
    certificate.to(device)
    features = torch.as_tensor(sample_set.features, dtype=torch.float32, device=device)
    correction_targets = torch.as_tensor(
        sample_set.correction_targets, dtype=torch.float32, device=device
    )
    states = torch.as_tensor(sample_set.states, dtype=torch.float32, device=device)
    errors = torch.as_tensor(sample_set.errors, dtype=torch.float32, device=device)
    certificate_targets = torch.as_tensor(
        sample_set.certificate_targets, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        list(correction.parameters()) + list(certificate.parameters()), lr=2e-3
    )
    mse = torch.nn.functional.mse_loss
    sample_count = features.shape[0]
    for _epoch in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            index = permutation[start : start + batch_size]
            predicted_correction = correction(features[index])
            predicted_certificate = certificate(states[index], errors[index])
            correction_loss = mse(predicted_correction, correction_targets[index])
            certificate_loss = mse(predicted_certificate, certificate_targets[index])
            loss = correction_loss + certificate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        correction_loss = float(mse(correction(features), correction_targets).item())
        certificate_loss = float(
            mse(certificate(states, errors), certificate_targets).item()
        )
    return (
        correction,
        certificate,
        {
            "correction_training_mse": correction_loss,
            "certificate_training_mse": certificate_loss,
        },
    )


def _numpy_correction(
    torch: object, model: object, device: str, value: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(value[None, :], dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy()[0]


def _simulate_neural(
    torch: object,
    model: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    nu: float,
    truth_initial: np.ndarray,
    estimate_initial: np.ndarray,
    *,
    noise: object = None,
) -> dict[str, float | int]:
    n = grid.n

    def rhs(time: float, combined: np.ndarray) -> np.ndarray:
        truth = combined[:n]
        estimate = combined[n:]
        measurement = matrix @ truth
        if noise is not None:
            measurement = measurement + noise(float(time))
        features = _feature_rows(
            grid,
            matrix,
            estimate[None, :],
            measurement[None, :],
            np.asarray([nu]),
        )[0]
        correction = _numpy_correction(torch, model, device, features)
        innovation = measurement - matrix @ estimate
        physical_baseline = BASELINE_GAIN * (matrix.T @ innovation) / grid.h
        return np.concatenate(
            (
                allen_cahn_rhs(grid, nu, truth),
                allen_cahn_rhs(grid, nu, estimate) + physical_baseline + correction,
            )
        )

    result = solve_ivp(
        rhs,
        (0.0, 1.0),
        np.concatenate((truth_initial, estimate_initial)),
        method="DOP853",
        t_eval=OUTPUT_TIMES,
        rtol=1e-8,
        atol=1e-10,
    )
    trajectories = result.y.T
    errors = trajectories[:, n:] - trajectories[:, :n]
    error_mass = np.sqrt(grid.h * np.sum(errors**2, axis=1))
    energies = np.asarray(
        [allen_cahn_energy(grid, nu, state) for state in trajectories[:, n:]]
    )
    return {
        "solver_status": int(result.status),
        "terminal_error_mass": float(error_mass[-1]),
        "peak_error_mass": float(np.max(error_mass)),
        "energy_defect": float(
            max(0.0, np.max(np.diff(energies, prepend=energies[0])))
        ),
    }


def _median(records: list[dict[str, float | int]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def _rollout_summary(
    rollout: object, grid: AllenCahnGrid, nu: float
) -> dict[str, float | int]:
    error = rollout.error_mass_norm
    energies = np.asarray(
        [allen_cahn_energy(grid, nu, state) for state in rollout.estimate]
    )
    return {
        "solver_status": int(rollout.solver_status),
        "terminal_error_mass": float(error[-1]),
        "peak_error_mass": float(np.max(error)),
        "energy_defect": float(
            max(0.0, np.max(np.diff(energies, prepend=energies[0])))
        ),
    }


def _certificate_diagnostics(
    torch: object,
    certificate: object,
    matrix: np.ndarray,
    sample_set: SampleSet,
    device: str,
    *,
    sample_limit: int = 3,
) -> dict[str, float | int]:
    count = min(sample_limit, sample_set.states.shape[0])
    states = torch.as_tensor(
        sample_set.states[:count], dtype=torch.float32, device=device
    )
    errors = torch.as_tensor(
        sample_set.errors[:count], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        transformed = certificate(states, errors).detach().cpu().numpy()
        zero = certificate(states, torch.zeros_like(errors)).detach().cpu().numpy()
    direction_residual = np.linalg.norm(
        (transformed - sample_set.errors[:count]) @ matrix.T, axis=1
    )
    minimum_singular: list[float] = []
    maximum_singular: list[float] = []
    for index in range(count):
        state = states[index].detach()
        error = errors[index].detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value, state=state: certificate(state[None, :], value[None, :])[0],
            error,
        )
        singular_values = np.linalg.svd(
            jacobian.detach().cpu().numpy(), compute_uv=False
        )
        minimum_singular.append(float(np.min(singular_values)))
        maximum_singular.append(float(np.max(singular_values)))
    return {
        "certificate_audit_sample_count": count,
        "certificate_max_zero_fiber_residual": float(
            np.max(np.linalg.norm(zero, axis=1))
        ),
        "certificate_max_direction_residual": float(np.max(direction_residual)),
        "certificate_min_jacobian_singular_value": min(minimum_singular),
        "certificate_max_jacobian_singular_value": max(maximum_singular),
    }


def run(
    torch: object,
    grid_sizes: list[int],
    seeds: list[int],
    *,
    epochs: int,
    batch_size: int,
    device: str,
    output_path: Path,
    rollout_limit: int,
    test_limit: int,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for grid_size in grid_sizes:
        grid = AllenCahnGrid(grid_size)
        matrix = local_average_matrix(grid, INTERVALS)
        train = _collect_samples(_split_cases("train", grid_size), grid, matrix)
        validation_cases = _split_cases("validation", grid_size)
        test_cases = _split_cases("test", grid_size)
        validation = _collect_samples(validation_cases, grid, matrix)
        validation_rollout_cases = validation_cases[:rollout_limit]
        grid_result: dict[str, object] = {
            "grid_size": grid_size,
            "train_case_count": len(_split_cases("train", grid_size)),
            "validation_case_count": len(validation_cases),
            "test_case_count": len(test_cases),
            "training_sample_count": int(train.features.shape[0]),
            "seed_results": [],
        }
        trained_models: dict[int, object] = {}
        for seed in seeds:
            correction, _certificate, losses = _train_one(
                torch,
                train,
                matrix,
                grid,
                seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
            with torch.no_grad():
                validation_features = torch.as_tensor(
                    validation.features, dtype=torch.float32, device=device
                )
                validation_targets = torch.as_tensor(
                    validation.correction_targets, dtype=torch.float32, device=device
                )
                validation_mse = float(
                    torch.nn.functional.mse_loss(
                        correction(validation_features), validation_targets
                    ).item()
                )
            rollout_records = [
                _simulate_neural(
                    torch,
                    correction,
                    device,
                    grid,
                    matrix,
                    case.nu,
                    case.initial_truth(grid),
                    case.initial_estimate(grid),
                )
                for case in validation_rollout_cases
            ]
            certificate_diagnostics = _certificate_diagnostics(
                torch, _certificate, matrix, train, device
            )
            result = {
                "seed": seed,
                **losses,
                "correction_validation_mse": validation_mse,
                "validation_rollout_case_count": len(rollout_records),
                "validation_rollout_median_terminal_error_mass": _median(
                    rollout_records, "terminal_error_mass"
                ),
                "validation_rollout_median_peak_error_mass": _median(
                    rollout_records, "peak_error_mass"
                ),
                "validation_rollout_median_energy_defect": _median(
                    rollout_records, "energy_defect"
                ),
                **certificate_diagnostics,
                "certificate_family": "nullspace-gated neural certificate",
                "certificate_fiber_constraint": "exact by construction up to floating point",
            }
            grid_result["seed_results"].append(result)
            trained_models[seed] = correction
            torch.save(
                {
                    "grid_size": grid_size,
                    "seed": seed,
                    "observation_matrix": matrix,
                    "correction_state_dict": correction.state_dict(),
                    "certificate_state_dict": _certificate.state_dict(),
                },
                output_path.parent / f"checkpoint-grid-{grid_size}-seed-{seed}.pt",
            )
        best_seed_result = min(
            grid_result["seed_results"],
            key=lambda item: item["correction_validation_mse"],
        )
        best_seed = int(best_seed_result["seed"])
        best_model = trained_models[best_seed]
        test_rollouts = [
            _simulate_neural(
                torch,
                best_model,
                device,
                grid,
                matrix,
                case.nu,
                case.initial_truth(grid),
                case.initial_estimate(grid),
            )
            for case in test_cases[:test_limit]
        ]
        grid_result["selected_seed"] = best_seed
        grid_result["test_rollout_case_count"] = len(test_rollouts)
        grid_result["test_rollout_median_terminal_error_mass"] = _median(
            test_rollouts, "terminal_error_mass"
        )
        grid_result["test_rollout_median_peak_error_mass"] = _median(
            test_rollouts, "peak_error_mass"
        )
        grid_result["test_rollout_median_energy_defect"] = _median(
            test_rollouts, "energy_defect"
        )
        results.append(grid_result)
    return {
        "kind": "r5-e-gpu-pilot",
        "information_contract": "correction uses estimate, measurement, innovation, nu, and estimate mass norm; certificate is offline-only",
        "baseline_gain": BASELINE_GAIN,
        "grid_results": results,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_rollout_limit": rollout_limit,
        "test_rollout_limit": test_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[31, 63, 127])
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503, 504])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-limit", type=int, default=48)
    parser.add_argument("--test-limit", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        torch,
        args.grid_sizes,
        args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
        rollout_limit=args.rollout_limit,
        test_limit=args.test_limit,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"grid_count": len(result["grid_results"]), "device": args.device})
    )


if __name__ == "__main__":
    main()
