"""R5 ablation sweep for online and certificate design factors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from r5_e_joint_train import (
    BASELINE_GAIN,
    INTERVALS,
    OUTPUT_TIMES,
    SampleSet,
    _build_models,
    _collect_samples,
    _median,
    _split_cases,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    NullspaceCertificate,
    allen_cahn_energy,
    allen_cahn_rhs,
    local_average_matrix,
    noise_waveform,
)


@dataclass(frozen=True)
class Ablation:
    name: str
    fixed_gain: float
    state_conditioning: bool
    certificate: str


ABLATIONS = (
    Ablation("full", 0.02, True, "constrained"),
    Ablation("no_fixed_gain", 0.0, True, "constrained"),
    Ablation("no_state_conditioning", 0.02, False, "constrained"),
    Ablation("no_certificate", 0.02, True, "identity"),
    Ablation("no_direction_constraint", 0.02, True, "unconstrained"),
    Ablation("no_lipschitz_control", 0.02, True, "nullspace_unbounded"),
)


class UnconstrainedCertificate:
    def __init__(self, torch: object, n: int) -> None:
        nn = torch.nn
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(2 * n, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n),
        )

    def parameters(self) -> object:
        return self.model.parameters()

    def to(self, device: str) -> UnconstrainedCertificate:
        self.model.to(device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, states: object, errors: object) -> object:
        return errors + 0.05 * self.torch.tanh(
            self.model(self.torch.cat((states, errors), dim=1))
        )


class NullspaceUnboundedCertificate:
    def __init__(self, torch: object, basis: np.ndarray, n: int) -> None:
        nn = torch.nn
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(2 * n, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, basis.shape[1]),
        )
        self.null_basis = torch.as_tensor(basis, dtype=torch.float32)

    def parameters(self) -> object:
        return self.model.parameters()

    def to(self, device: str) -> NullspaceUnboundedCertificate:
        self.model.to(device)
        self.null_basis = self.null_basis.to(device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, states: object, errors: object) -> object:
        coordinates = errors @ self.null_basis
        gates = self.model(self.torch.cat((states, errors), dim=1))
        return errors + (gates * coordinates) @ self.null_basis.T


class IdentityCertificate:
    def parameters(self) -> list[object]:
        return []

    def to(self, device: str) -> IdentityCertificate:
        del device
        return self

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def __call__(self, states: object, errors: object) -> object:
        del states
        return errors


def _make_certificate(torch: object, mode: str, matrix: np.ndarray, n: int) -> object:
    if mode == "identity":
        return IdentityCertificate()
    if mode == "unconstrained":
        return UnconstrainedCertificate(torch, n)
    if mode == "nullspace_unbounded":
        return NullspaceUnboundedCertificate(
            torch, NullspaceCertificate(matrix).null_basis, n
        )
    return _build_models(torch, n + 6, n, NullspaceCertificate(matrix).null_basis)[1]


def _features(sample: SampleSet, state_conditioning: bool) -> np.ndarray:
    return sample.features if state_conditioning else sample.features[:, :-1]


def _targets(
    sample: SampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    fixed_gain: float,
) -> np.ndarray:
    if fixed_gain == BASELINE_GAIN:
        return sample.correction_targets
    q = matrix.shape[0]
    innovation = sample.features[:, grid.n + q : grid.n + 2 * q]
    baseline_change = (BASELINE_GAIN - fixed_gain) * (innovation @ matrix) / grid.h
    return sample.correction_targets + baseline_change


def _train_one(
    torch: object,
    sample: SampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    ablation: Ablation,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> tuple[object, object, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    features = _features(sample, ablation.state_conditioning)
    targets = _targets(sample, grid, matrix, ablation.fixed_gain)
    correction, _ = _build_models(
        torch, features.shape[1], grid.n, NullspaceCertificate(matrix).null_basis
    )
    certificate = _make_certificate(torch, ablation.certificate, matrix, grid.n)
    correction.to(device)
    certificate.to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    states = torch.as_tensor(sample.states, dtype=torch.float32, device=device)
    errors = torch.as_tensor(sample.errors, dtype=torch.float32, device=device)
    cert_targets = torch.as_tensor(
        sample.certificate_targets, dtype=torch.float32, device=device
    )
    params = list(correction.parameters()) + list(certificate.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-3)
    mse = torch.nn.functional.mse_loss
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            index = permutation[start : start + batch_size]
            correction_loss = mse(correction(x[index]), y[index])
            certificate_loss = mse(
                certificate(states[index], errors[index]), cert_targets[index]
            )
            optimizer.zero_grad(set_to_none=True)
            (correction_loss + certificate_loss).backward()
            optimizer.step()
    with torch.no_grad():
        train_mse = float(mse(correction(x), y).item())
    correction.eval()
    return correction, certificate, train_mse, float(features.shape[1])


def _feature_value(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    estimate: np.ndarray,
    measurement: np.ndarray,
    nu: float,
    state_conditioning: bool,
) -> np.ndarray:
    innovation = measurement - matrix @ estimate
    values = np.concatenate(
        (
            estimate,
            measurement,
            innovation,
            np.asarray([(nu - 0.01) / 0.01]),
            np.asarray([np.sqrt(grid.h * np.dot(estimate, estimate))]),
        )
    )
    return values if state_conditioning else values[:-1]


def _simulate(
    torch: object,
    correction: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: object,
    ablation: Ablation,
    *,
    noise: object = None,
) -> dict[str, float | int]:
    from scipy.integrate import solve_ivp

    n = grid.n

    def rhs(time: float, combined: np.ndarray) -> np.ndarray:
        truth, estimate = combined[:n], combined[n:]
        measurement = matrix @ truth
        if noise is not None:
            measurement = measurement + noise(float(time))
        feature = _feature_value(
            grid, matrix, estimate, measurement, case.nu, ablation.state_conditioning
        )
        with torch.no_grad():
            value = torch.as_tensor(
                feature[None, :], dtype=torch.float32, device=device
            )
            learned = correction(value).cpu().numpy()[0]
        innovation = measurement - matrix @ estimate
        baseline = ablation.fixed_gain * (matrix.T @ innovation) / grid.h
        return np.concatenate(
            (
                allen_cahn_rhs(grid, case.nu, truth),
                allen_cahn_rhs(grid, case.nu, estimate) + baseline + learned,
            )
        )

    result = solve_ivp(
        rhs,
        (0.0, 1.0),
        np.concatenate((case.initial_truth(grid), case.initial_estimate(grid))),
        method="DOP853",
        t_eval=OUTPUT_TIMES,
        rtol=1e-8,
        atol=1e-10,
    )
    trajectories = result.y.T
    error = trajectories[:, n:] - trajectories[:, :n]
    error_mass = np.sqrt(grid.h * np.sum(error**2, axis=1))
    energies = np.asarray(
        [allen_cahn_energy(grid, case.nu, state) for state in trajectories[:, n:]]
    )
    return {
        "solver_status": int(result.status),
        "terminal_error_mass": float(error_mass[-1]),
        "peak_error_mass": float(np.max(error_mass)),
        "energy_defect": float(
            max(0.0, np.max(np.diff(energies, prepend=energies[0])))
        ),
    }


def _audit(
    torch: object,
    certificate: object,
    matrix: np.ndarray,
    grid: AllenCahnGrid,
    device: str,
) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(10000 + grid.n))
    states = rng.normal(size=(3, grid.n)) * 0.1
    errors = rng.normal(size=(3, grid.n)) * 0.05
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    error_tensor = torch.as_tensor(errors, dtype=torch.float32, device=device)
    with torch.no_grad():
        transformed = certificate(state_tensor, error_tensor).cpu().numpy()
        zero = certificate(state_tensor, torch.zeros_like(error_tensor)).cpu().numpy()
    direction = np.linalg.norm((transformed - errors) @ matrix.T, axis=1)
    minimum_singular: list[float] = []
    maximum_singular: list[float] = []
    for index in range(states.shape[0]):
        state = state_tensor[index].detach()
        error = error_tensor[index].detach().requires_grad_(True)
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
        "max_zero_fiber_residual": float(np.max(np.linalg.norm(zero, axis=1))),
        "max_direction_residual": float(np.max(direction)),
        "min_jacobian_singular_value": min(minimum_singular),
        "max_jacobian_singular_value": max(maximum_singular),
    }


def run(
    torch: object,
    grid_sizes: list[int],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    eval_limit: int,
    noise_limit: int,
    device: str,
) -> dict[str, object]:
    results = []
    for grid_size in grid_sizes:
        grid = AllenCahnGrid(grid_size)
        matrix = local_average_matrix(grid, INTERVALS)
        train = _collect_samples(_split_cases("train", grid_size), grid, matrix)
        validation = _collect_samples(
            _split_cases("validation", grid_size), grid, matrix
        )
        test_cases = _split_cases("test", grid_size)
        grid_results = []
        for ablation in ABLATIONS:
            print(
                f"[grid={grid_size} ablation={ablation.name}] training "
                f"{len(seeds)} seeds",
                flush=True,
            )
            seed_results = []
            models = {}
            for seed in seeds:
                correction, certificate, train_mse, feature_dim = _train_one(
                    torch,
                    train,
                    grid,
                    matrix,
                    ablation,
                    seed,
                    epochs,
                    batch_size,
                    device,
                )
                x_val = torch.as_tensor(
                    _features(validation, ablation.state_conditioning),
                    dtype=torch.float32,
                    device=device,
                )
                y_val = torch.as_tensor(
                    _targets(validation, grid, matrix, ablation.fixed_gain),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    val_mse = float(
                        torch.nn.functional.mse_loss(correction(x_val), y_val).item()
                    )
                seed_results.append(
                    {
                        "seed": seed,
                        "train_mse": train_mse,
                        "validation_mse": val_mse,
                        "feature_dimension": int(feature_dim),
                    }
                )
                models[seed] = (correction, certificate)
            best = min(seed_results, key=lambda item: item["validation_mse"])
            best_seed = int(best["seed"])
            correction, certificate = models[best_seed]
            replay = [
                _simulate(torch, correction, device, grid, matrix, case, ablation)
                for case in test_cases[:eval_limit]
            ]
            noisy = lambda time, q=matrix.shape[0]: noise_waveform(
                "common-sine", 0.01, q, time
            )
            noisy_replay = [
                _simulate(
                    torch,
                    correction,
                    device,
                    grid,
                    matrix,
                    case,
                    ablation,
                    noise=noisy,
                )
                for case in test_cases[:noise_limit]
            ]
            grid_results.append(
                {
                    "ablation": ablation.name,
                    "fixed_gain": ablation.fixed_gain,
                    "state_conditioning": ablation.state_conditioning,
                    "certificate": ablation.certificate,
                    "selected_seed": best_seed,
                    "seed_results": seed_results,
                    "test_case_count": len(replay),
                    "test_median_terminal_error_mass": _median(
                        replay, "terminal_error_mass"
                    ),
                    "test_median_peak_error_mass": _median(replay, "peak_error_mass"),
                    "noisy_case_count": len(noisy_replay),
                    "noisy_median_terminal_error_mass": _median(
                        noisy_replay, "terminal_error_mass"
                    ),
                    "certificate_audit": _audit(
                        torch, certificate, matrix, grid, device
                    ),
                }
            )
            print(
                f"[grid={grid_size} ablation={ablation.name}] "
                f"seed={best_seed} test={grid_results[-1]['test_median_terminal_error_mass']:.6g} "
                f"noisy={grid_results[-1]['noisy_median_terminal_error_mass']:.6g}",
                flush=True,
            )
        results.append({"grid_size": grid_size, "ablations": grid_results})
    return {
        "kind": "r5-ablation-sweep",
        "grid_sizes": grid_sizes,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_limit": eval_limit,
        "noise_limit": noise_limit,
        "device": device,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[31, 63, 127])
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503, 504])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=48)
    parser.add_argument("--noise-limit", type=int, default=12)
    parser.add_argument("--device", default="cuda")
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
        args.epochs,
        args.batch_size,
        args.eval_limit,
        args.noise_limit,
        args.device,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_count": len(result["results"]), "device": args.device}))


if __name__ == "__main__":
    main()
