"""Train bounded low-mode residuals around the certified two-sensor LMI design."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    allen_cahn_rhs,
    generate_pilot_cases,
    lmi_modal_injection,
    local_average_matrix,
    mass_adjoint_injection,
    noise_waveform,
    normalized_modal_transform,
    simulate_causal_nudging,
    unstable_modal_system,
)


INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]])
NU_VALUES = (0.005, 0.010, 0.020)
OUTPUT_TIMES = np.linspace(0.0, 1.0, 101)
LMI_CONDITION_BOUNDS = (16.0, 32.0, 64.0, 128.0, 256.0)
MAX_POLICY_RHS_EVALUATIONS = 20_000


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    nu: float
    truth_initial: np.ndarray
    estimate_initial: np.ndarray


@dataclass(frozen=True)
class JointSamples:
    states: np.ndarray
    estimates: np.ndarray
    measurements: np.ndarray
    next_states: np.ndarray
    nus: np.ndarray
    nu_indices: np.ndarray
    times: np.ndarray
    dt: float


def _mode_basis(grid: AllenCahnGrid, mode_count: int = 4) -> np.ndarray:
    modes = np.arange(1, mode_count + 1, dtype=float)
    return np.sqrt(2.0 * grid.h) * np.sin(
        np.pi * grid.x[:, None] * modes[None, :]
    )


def _design_bases(
    grid: AllenCahnGrid, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    gains: list[np.ndarray] = []
    transforms: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for nu in NU_VALUES:
        selected = None
        selected_bound = None
        for condition_bound in LMI_CONDITION_BOUNDS:
            try:
                selected = lmi_modal_injection(
                    grid,
                    nu,
                    matrix,
                    decay_rate=0.1 * nu * np.pi**2,
                    metric_condition_bound=condition_bound,
                )
            except RuntimeError:
                continue
            selected_bound = condition_bound
            break
        if selected is None or selected_bound is None:
            raise RuntimeError(f"no LMI base design was feasible for nu={nu}")
        modal = unstable_modal_system(grid, nu, matrix)
        gains.append(selected.injection_matrix)
        transforms.append(
            normalized_modal_transform(grid, modal, selected.modal_metric)
        )
        singular_values = np.linalg.svd(transforms[-1], compute_uv=False)
        diagnostics.append(
            {
                "nu": nu,
                "condition_bound": selected_bound,
                "closed_loop_spectral_abscissa": (
                    selected.closed_loop_spectral_abscissa
                ),
                "modal_contraction_rate": selected.modal_contraction_rate,
                "mass_scaled_gain_norm": selected.mass_scaled_gain_norm,
                "modal_metric_condition": selected.modal_metric_condition,
                "transform_min_singular_value": float(singular_values[-1]),
                "transform_max_singular_value": float(singular_values[0]),
            }
        )
    return np.asarray(gains), np.asarray(transforms), diagnostics


def _case_split(
    split: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    limit_per_nu: int,
    stress_truths_per_nu: int,
) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    for nu in NU_VALUES:
        base = [
            case
            for case in generate_pilot_cases()
            if case.split == split and case.n == grid.n and case.nu == nu
        ][:limit_per_nu]
        for case in base:
            cases.append(
                ExperimentCase(
                    case.case_id,
                    case.nu,
                    case.initial_truth(grid),
                    case.initial_estimate(grid),
                )
            )
        target_mass_norm = 0.25 / np.sqrt(2.0)
        fourth = np.sin(4.0 * np.pi * grid.x)
        fourth *= target_mass_norm / np.sqrt(grid.h * np.dot(fourth, fourth))
        modal = unstable_modal_system(grid, nu, matrix)
        _, _, right = np.linalg.svd(modal.observed_modes, full_matrices=True)
        hard = modal.modes @ right[-1]
        hard *= target_mass_norm / np.sqrt(grid.h * np.dot(hard, hard))
        for index, case in enumerate(base[:stress_truths_per_nu]):
            truth = case.initial_truth(grid)
            for name, direction in (("fourth", fourth), ("min-observation", hard)):
                for sign in (-1.0, 1.0):
                    cases.append(
                        ExperimentCase(
                            f"{split}-stress-{name}-{index}-sign-{sign:+g}"
                            f"__n-{grid.n}__nu-{nu:.3f}",
                            nu,
                            truth,
                            truth + sign * direction,
                        )
                    )
    return cases


def _feature_numpy(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    estimate: np.ndarray,
    measurement: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - matrix @ estimate
    feature = np.concatenate(
        (
            estimate,
            measurement,
            innovation,
            np.asarray([(nu - 0.01) / 0.01]),
            np.asarray([np.sqrt(grid.h * np.dot(estimate, estimate))]),
        )
    )
    return feature, innovation


def _feature_tensor(
    torch: object,
    estimates: object,
    measurements: object,
    nus: object,
    matrix: object,
    h: float,
) -> tuple[object, object]:
    innovations = measurements - estimates @ matrix.T
    scales = torch.sqrt(h * torch.sum(estimates**2, dim=1))
    viscosity = (nus - 0.01) / 0.01
    features = torch.cat(
        (
            estimates,
            measurements,
            innovations,
            viscosity[:, None],
            scales[:, None],
        ),
        dim=1,
    )
    return features, innovations


def _build_models(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
    base_transforms: np.ndarray,
    *,
    gain_trust_ratio: float,
    certificate_log_scale: float,
) -> tuple[object, object]:
    nn = torch.nn
    q = matrix.shape[0]
    mode_count = 4
    feature_dimension = grid.n + 2 * q + 2
    basis = _mode_basis(grid, mode_count)

    class GainResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_dimension, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, mode_count * q),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.register_buffer(
                "base_gains", torch.as_tensor(base_gains, dtype=torch.float32)
            )
            self.register_buffer(
                "mode_basis",
                torch.as_tensor(basis / np.sqrt(grid.h), dtype=torch.float32),
            )
            self.gain_trust_ratio = gain_trust_ratio

        def forward(self, features: object, nu_indices: object) -> object:
            base = self.base_gains[nu_indices]
            raw = self.network(features).reshape(-1, mode_count, q)
            modal_delta = torch.tanh(raw)
            delta = torch.einsum("nm,bmq->bnq", self.mode_basis, modal_delta)
            delta_norm = torch.sqrt(
                grid.h * torch.sum(delta**2, dim=(1, 2), keepdim=True) + 1.0e-12
            )
            base_norm = torch.sqrt(
                grid.h * torch.sum(base**2, dim=(1, 2), keepdim=True)
            )
            limit = self.gain_trust_ratio * base_norm
            scale = torch.clamp(limit / (delta_norm + 1.0e-12), max=1.0)
            return base + scale * delta

    class CertificateResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(grid.n + 1, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, mode_count),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.register_buffer(
                "base_transforms",
                torch.as_tensor(base_transforms, dtype=torch.float32),
            )
            self.register_buffer(
                "mode_basis", torch.as_tensor(basis, dtype=torch.float32)
            )
            self.log_scale = certificate_log_scale

        def forward(
            self, states: object, errors: object, nu_indices: object, nus: object
        ) -> object:
            viscosity = ((nus - 0.01) / 0.01)[:, None]
            raw = self.network(torch.cat((states, viscosity), dim=1))
            scales = torch.exp(self.log_scale * torch.tanh(raw))
            coordinates = errors @ self.mode_basis
            residual_error = errors + (
                coordinates * (scales - 1.0)
            ) @ self.mode_basis.T
            return torch.bmm(
                self.base_transforms[nu_indices], residual_error[:, :, None]
            ).squeeze(-1)

    return GainResidual(), CertificateResidual()


def _nu_index(nu: float) -> int:
    return NU_VALUES.index(float(nu))


def _fixed_samples(
    cases: list[ExperimentCase],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
) -> JointSamples:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    nus: list[float] = []
    indices: list[int] = []
    times: list[float] = []
    for case in cases:
        index = _nu_index(case.nu)
        observer = CausalOutputInjection(grid, case.nu, matrix, base_gains[index])
        rollout = simulate_causal_nudging(
            observer,
            case.truth_initial,
            case.estimate_initial,
            output_times=OUTPUT_TIMES,
        )
        if rollout.solver_status != 0:
            raise RuntimeError(f"base rollout failed for {case.case_id}")
        states.extend(rollout.truth[:-1])
        estimates.extend(rollout.estimate[:-1])
        measurements.extend(rollout.measurements[:-1])
        next_states.extend(rollout.truth[1:])
        count = OUTPUT_TIMES.size - 1
        nus.extend([case.nu] * count)
        indices.extend([index] * count)
        times.extend(OUTPUT_TIMES[:-1])
    return JointSamples(
        states=np.asarray(states),
        estimates=np.asarray(estimates),
        measurements=np.asarray(measurements),
        next_states=np.asarray(next_states),
        nus=np.asarray(nus),
        nu_indices=np.asarray(indices),
        times=np.asarray(times),
        dt=float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0]),
    )


def _policy_rollout(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: ExperimentCase,
    *,
    noise: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    evaluations = 0
    index = _nu_index(case.nu)

    def rhs(time: float, combined: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > MAX_POLICY_RHS_EVALUATIONS:
            raise RuntimeError("policy rollout exceeded the RHS budget")
        truth, estimate = combined[: grid.n], combined[grid.n :]
        measurement = matrix @ truth
        if noise is not None:
            measurement = measurement + noise(time)
        features, innovation = _feature_numpy(
            grid, matrix, estimate, measurement, case.nu
        )
        with torch.no_grad():
            feature_tensor = torch.as_tensor(
                features[None, :], dtype=torch.float32, device=device
            )
            index_tensor = torch.as_tensor([index], dtype=torch.long, device=device)
            injection = gain(feature_tensor, index_tensor)[0].cpu().numpy()
        return np.concatenate(
            (
                allen_cahn_rhs(grid, case.nu, truth),
                allen_cahn_rhs(grid, case.nu, estimate) + injection @ innovation,
            )
        )

    result = solve_ivp(
        rhs,
        (0.0, 1.0),
        np.concatenate((case.truth_initial, case.estimate_initial)),
        method="DOP853",
        t_eval=OUTPUT_TIMES,
        rtol=1e-8,
        atol=1e-10,
    )
    trajectories = result.y.T
    truth = trajectories[:, : grid.n]
    estimate = trajectories[:, grid.n :]
    measurements = np.asarray(
        [
            matrix @ state
            + (np.zeros(matrix.shape[0]) if noise is None else noise(float(time)))
            for time, state in zip(result.t, truth, strict=True)
        ]
    )
    return truth, estimate, measurements, int(result.status)


def _policy_samples(
    torch: object,
    gain: object,
    device: str,
    cases: list[ExperimentCase],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
) -> JointSamples:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    nus: list[float] = []
    indices: list[int] = []
    times: list[float] = []
    gain.eval()
    for case in cases:
        truth, estimate, observed, status = _policy_rollout(
            torch, gain, device, grid, matrix, case
        )
        if status != 0 or truth.shape[0] != OUTPUT_TIMES.size:
            raise RuntimeError(f"policy rollout failed for {case.case_id}")
        states.extend(truth[:-1])
        estimates.extend(estimate[:-1])
        measurements.extend(observed[:-1])
        next_states.extend(truth[1:])
        count = OUTPUT_TIMES.size - 1
        nus.extend([case.nu] * count)
        indices.extend([_nu_index(case.nu)] * count)
        times.extend(OUTPUT_TIMES[:-1])
    return JointSamples(
        states=np.asarray(states),
        estimates=np.asarray(estimates),
        measurements=np.asarray(measurements),
        next_states=np.asarray(next_states),
        nus=np.asarray(nus),
        nu_indices=np.asarray(indices),
        times=np.asarray(times),
        dt=float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0]),
    )


def _tensorize(
    torch: object, samples: JointSamples, grid: AllenCahnGrid, device: str
) -> dict[str, object]:
    return {
        "states": torch.as_tensor(samples.states, dtype=torch.float32, device=device),
        "estimates": torch.as_tensor(
            samples.estimates, dtype=torch.float32, device=device
        ),
        "measurements": torch.as_tensor(
            samples.measurements, dtype=torch.float32, device=device
        ),
        "next_states": torch.as_tensor(
            samples.next_states, dtype=torch.float32, device=device
        ),
        "nus": torch.as_tensor(samples.nus, dtype=torch.float32, device=device),
        "nu_indices": torch.as_tensor(
            samples.nu_indices, dtype=torch.long, device=device
        ),
        "laplacian": torch.as_tensor(
            grid.laplacian, dtype=torch.float32, device=device
        ),
        "dt": samples.dt,
    }


def _target_operators(grid: AllenCahnGrid) -> tuple[np.ndarray, np.ndarray]:
    generators: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for nu in NU_VALUES:
        lam = 0.1 * nu * np.pi**2
        generator = nu * grid.laplacian - lam * np.eye(grid.n)
        generators.append(generator)
        maps.append(expm((OUTPUT_TIMES[1] - OUTPUT_TIMES[0]) * generator))
    return np.asarray(generators), np.asarray(maps)


def _loss_components(
    torch: object,
    gain: object,
    certificate: object,
    samples: dict[str, object],
    matrix: object,
    target_generators: object,
    target_maps: object,
    indices: object,
    grid: AllenCahnGrid,
    *,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    bi_weight: float,
    gain_reg_weight: float,
    lower_bound: float,
    upper_bound: float,
    create_graph: bool,
) -> dict[str, object]:
    states = samples["states"][indices]
    estimates = samples["estimates"][indices]
    measurements = samples["measurements"][indices]
    next_states = samples["next_states"][indices]
    nus = samples["nus"][indices]
    nu_indices = samples["nu_indices"][indices]
    features, innovations = _feature_tensor(
        torch, estimates, measurements, nus, matrix, grid.h
    )
    gains = gain(features, nu_indices)
    correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
    rhs_truth = (
        nus[:, None] * (states @ samples["laplacian"].T)
        + states
        - states**3
    )
    rhs_estimate = (
        nus[:, None] * (estimates @ samples["laplacian"].T)
        + estimates
        - estimates**3
    )
    error_rhs = rhs_estimate + correction - rhs_truth
    errors = estimates - states
    transformed = certificate(states, errors, nu_indices, nus)

    def transform(state: object, error: object) -> object:
        return certificate(state, error, nu_indices, nus)

    _, transformed_rhs = torch.autograd.functional.jvp(
        transform,
        (states, errors),
        (rhs_truth, error_rhs),
        create_graph=create_graph,
    )
    transformed_squared = grid.h * torch.sum(transformed**2, dim=1)
    transformed_inner = grid.h * torch.sum(transformed * transformed_rhs, dim=1)
    lam = 0.1 * nus * np.pi**2
    normalized_violation = (
        transformed_inner + lam * transformed_squared
    ) / (transformed_squared + 1e-8)
    contraction_loss = torch.mean(torch.relu(normalized_violation) ** 2)

    generator = torch.bmm(
        target_generators[nu_indices], transformed[:, :, None]
    ).squeeze(-1)
    error_squared = grid.h * torch.sum(errors**2, dim=1)
    defect_squared = grid.h * torch.sum((transformed_rhs - generator) ** 2, dim=1)
    defect_loss = torch.mean(defect_squared / (error_squared + 1e-8))

    next_estimates = estimates + samples["dt"] * (rhs_estimate + correction)
    next_errors = next_estimates - next_states
    next_transformed = certificate(
        next_states, next_errors, nu_indices, nus
    )
    target = torch.bmm(target_maps[nu_indices], transformed[:, :, None]).squeeze(-1)
    stable_squared = grid.h * torch.sum((next_transformed - target) ** 2, dim=1)
    stable_loss = torch.mean(
        stable_squared / (samples["dt"] ** 2 * (error_squared + 1e-8))
    )

    error_norm = torch.sqrt(error_squared + 1e-12)
    transformed_norm = torch.sqrt(transformed_squared + 1e-12)
    bi_loss = torch.mean(
        torch.relu(lower_bound * error_norm - transformed_norm) ** 2
        + torch.relu(transformed_norm - upper_bound * error_norm) ** 2
    )
    base = gain.base_gains[nu_indices]
    gain_reg = torch.mean(
        grid.h * torch.sum((gains - base) ** 2, dim=(1, 2))
        / (grid.h * torch.sum(base**2, dim=(1, 2)) + 1e-8)
    )
    total = (
        stable_weight * stable_loss
        + defect_weight * defect_loss
        + contraction_weight * contraction_loss
        + bi_weight * bi_loss
        + gain_reg_weight * gain_reg
    )
    rates = -transformed_inner / (transformed_squared + 1e-8)
    return {
        "stable": stable_loss,
        "defect": defect_loss,
        "contraction": contraction_loss,
        "bi": bi_loss,
        "gain_reg": gain_reg,
        "total": total,
        "rates": rates,
        "requested": lam,
    }


def _train_seed(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    base_gains: np.ndarray,
    base_transforms: np.ndarray,
    train_cases: list[ExperimentCase],
    validation_cases: list[ExperimentCase],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    bi_weight: float,
    gain_reg_weight: float,
) -> tuple[object, object, dict[str, object]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gains,
        base_transforms,
        gain_trust_ratio=gain_trust_ratio,
        certificate_log_scale=certificate_log_scale,
    )
    gain.to(device)
    certificate.to(device)
    initial = _fixed_samples(train_cases, grid, matrix, base_gains)
    samples = _tensorize(torch, initial, grid, device)
    generators, maps = _target_operators(grid)
    generator_tensor = torch.as_tensor(generators, dtype=torch.float32, device=device)
    map_tensor = torch.as_tensor(maps, dtype=torch.float32, device=device)
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        [
            {"params": gain.parameters(), "lr": gain_learning_rate},
            {"params": certificate.parameters(), "lr": certificate_learning_rate},
        ]
    )
    history: list[dict[str, float]] = []
    refresh_count = 0
    for epoch in range(epochs):
        if epoch > 0 and refresh_interval > 0 and epoch % refresh_interval == 0:
            refreshed = _policy_samples(
                torch, gain, device, train_cases, grid, matrix
            )
            samples = _tensorize(torch, refreshed, grid, device)
            refresh_count += 1
        gain.train()
        certificate.train()
        sample_count = samples["states"].shape[0]
        permutation = torch.randperm(sample_count, device=device)
        totals: dict[str, float] = {
            name: 0.0
            for name in ("stable", "defect", "contraction", "bi", "gain_reg", "total")
        }
        batch_count = 0
        for start in range(0, sample_count, batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            components = _loss_components(
                torch,
                gain,
                certificate,
                samples,
                matrix_tensor,
                generator_tensor,
                map_tensor,
                batch_indices,
                grid,
                stable_weight=stable_weight,
                defect_weight=defect_weight,
                contraction_weight=contraction_weight,
                bi_weight=bi_weight,
                gain_reg_weight=gain_reg_weight,
                lower_bound=0.25,
                upper_bound=3.5,
                create_graph=True,
            )
            if not torch.isfinite(components["total"]):
                raise RuntimeError("non-finite joint loss")
            components["total"].backward()
            if any(
                parameter.grad is not None
                and not torch.all(torch.isfinite(parameter.grad))
                for parameter in list(gain.parameters())
                + list(certificate.parameters())
            ):
                raise RuntimeError("non-finite joint gradient")
            torch.nn.utils.clip_grad_norm_(
                list(gain.parameters()) + list(certificate.parameters()), 1.0
            )
            optimizer.step()
            for name in totals:
                totals[name] += float(components[name].detach().cpu())
            batch_count += 1
        history.append({name: value / batch_count for name, value in totals.items()})
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[seed={seed}] epoch={epoch + 1}/{epochs} "
                f"total={history[-1]['total']:.6g} "
                f"contraction={history[-1]['contraction']:.6g}",
                flush=True,
            )

    validation = _policy_samples(
        torch, gain, device, validation_cases, grid, matrix
    )
    validation_tensor = _tensorize(torch, validation, grid, device)
    all_indices = torch.arange(
        validation_tensor["states"].shape[0], device=device
    )
    gain.eval()
    certificate.eval()
    validation_components = _loss_components(
        torch,
        gain,
        certificate,
        validation_tensor,
        matrix_tensor,
        generator_tensor,
        map_tensor,
        all_indices,
        grid,
        stable_weight=stable_weight,
        defect_weight=defect_weight,
        contraction_weight=contraction_weight,
        bi_weight=bi_weight,
        gain_reg_weight=gain_reg_weight,
        lower_bound=0.25,
        upper_bound=3.5,
        create_graph=False,
    )
    rates = validation_components["rates"].detach().cpu().numpy()
    requested = validation_components["requested"].detach().cpu().numpy()
    result = {
        "seed": seed,
        "refresh_count": refresh_count,
        "final_training": history[-1],
        "validation": {
            name: float(validation_components[name].detach().cpu())
            for name in ("stable", "defect", "contraction", "bi", "gain_reg", "total")
        },
        "validation_contraction": {
            "sample_count": int(rates.size),
            "min": float(np.min(rates)),
            "p05": float(np.quantile(rates, 0.05)),
            "median": float(np.median(rates)),
            "positive_fraction": float(np.mean(rates > 0.0)),
            "requested_margin_fraction": float(np.mean(rates >= requested)),
        },
    }
    return gain, certificate, result


def _rollout_summary(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    cases: list[ExperimentCase],
    *,
    noise: Callable[[float], np.ndarray] | None = None,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in cases:
        truth, estimate, _, status = _policy_rollout(
            torch, gain, device, grid, matrix, case, noise=noise
        )
        if status != 0:
            raise RuntimeError(f"evaluation rollout failed for {case.case_id}")
        error = estimate - truth
        norms = np.sqrt(grid.h * np.sum(error**2, axis=1))
        records.append(
            {
                "case_id": case.case_id,
                "nu": case.nu,
                "terminal_error_mass": float(norms[-1]),
                "peak_error_mass": float(np.max(norms)),
            }
        )
    by_nu: dict[str, object] = {}
    for nu in NU_VALUES:
        subset = [record for record in records if record["nu"] == nu]
        by_nu[f"{nu:.3f}"] = {
            "case_count": len(subset),
            "terminal_error_mass_median": float(
                np.median([record["terminal_error_mass"] for record in subset])
            ),
            "terminal_error_mass_max": float(
                np.max([record["terminal_error_mass"] for record in subset])
            ),
            "peak_error_mass_max": float(
                np.max([record["peak_error_mass"] for record in subset])
            ),
        }
    return {"case_count": len(records), "by_nu": by_nu, "records": records}


def _fixed_rollout_summary(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    cases: list[ExperimentCase],
    injections: np.ndarray,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in cases:
        observer = CausalOutputInjection(
            grid, case.nu, matrix, injections[_nu_index(case.nu)]
        )
        rollout = simulate_causal_nudging(
            observer,
            case.truth_initial,
            case.estimate_initial,
            output_times=OUTPUT_TIMES,
        )
        records.append(
            {
                "case_id": case.case_id,
                "nu": case.nu,
                "terminal_error_mass": float(rollout.error_mass_norm[-1]),
                "peak_error_mass": float(np.max(rollout.error_mass_norm)),
            }
        )
    by_nu: dict[str, object] = {}
    for nu in NU_VALUES:
        subset = [record for record in records if record["nu"] == nu]
        by_nu[f"{nu:.3f}"] = {
            "case_count": len(subset),
            "terminal_error_mass_median": float(
                np.median([record["terminal_error_mass"] for record in subset])
            ),
            "terminal_error_mass_max": float(
                np.max([record["terminal_error_mass"] for record in subset])
            ),
            "peak_error_mass_max": float(
                np.max([record["peak_error_mass"] for record in subset])
            ),
        }
    return {"case_count": len(records), "by_nu": by_nu, "records": records}


def run(
    torch: object,
    *,
    grid_size: int,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    refresh_interval: int,
    device: str,
    train_limit_per_nu: int,
    validation_limit_per_nu: int,
    test_limit_per_nu: int,
    stress_truths_per_nu: int,
    gain_trust_ratio: float,
    certificate_log_scale: float,
    gain_learning_rate: float,
    certificate_learning_rate: float,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    bi_weight: float,
    gain_reg_weight: float,
    checkpoint_dir: Path | None,
    run_test: bool = True,
) -> dict[str, object]:
    grid = AllenCahnGrid(grid_size)
    matrix = local_average_matrix(grid, INTERVALS)
    base_gains, base_transforms, base_diagnostics = _design_bases(grid, matrix)
    train_cases = _case_split(
        "train",
        grid,
        matrix,
        limit_per_nu=train_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    validation_cases = _case_split(
        "validation",
        grid,
        matrix,
        limit_per_nu=validation_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    test_cases = _case_split(
        "test",
        grid,
        matrix,
        limit_per_nu=test_limit_per_nu,
        stress_truths_per_nu=stress_truths_per_nu,
    )
    fixed_injections = np.asarray(
        [mass_adjoint_injection(grid, matrix, gain=0.1) for _ in NU_VALUES]
    )
    validation_baselines = {
        "fixed-0.1": _fixed_rollout_summary(
            grid, matrix, validation_cases, fixed_injections
        ),
        "lmi-base": _fixed_rollout_summary(
            grid, matrix, validation_cases, base_gains
        ),
    }

    models: dict[int, tuple[object, object]] = {}
    seed_results: list[dict[str, object]] = []
    for seed in seeds:
        gain, certificate, result = _train_seed(
            torch,
            grid,
            matrix,
            base_gains,
            base_transforms,
            train_cases,
            validation_cases,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            refresh_interval=refresh_interval,
            device=device,
            gain_trust_ratio=gain_trust_ratio,
            certificate_log_scale=certificate_log_scale,
            gain_learning_rate=gain_learning_rate,
            certificate_learning_rate=certificate_learning_rate,
            stable_weight=stable_weight,
            defect_weight=defect_weight,
            contraction_weight=contraction_weight,
            bi_weight=bi_weight,
            gain_reg_weight=gain_reg_weight,
        )
        result["validation_rollout"] = _rollout_summary(
            torch, gain, device, grid, matrix, validation_cases
        )
        models[seed] = (gain, certificate)
        seed_results.append(result)
    selected = max(
        seed_results,
        key=lambda item: (
            item["validation_contraction"]["min"],
            item["validation_contraction"]["p05"],
            -item["validation_rollout"]["by_nu"]["0.005"][
                "terminal_error_mass_median"
            ],
        ),
    )
    selected_seed = int(selected["seed"])
    gain, certificate = models[selected_seed]
    test = None
    noisy_test = None
    if run_test:
        noise = lambda time: noise_waveform("common-sine", 0.01, 2, time)
        test = _rollout_summary(torch, gain, device, grid, matrix, test_cases)
        noisy_test = _rollout_summary(
            torch, gain, device, grid, matrix, test_cases, noise=noise
        )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "grid_size": grid_size,
                "seed": selected_seed,
                "gain_state_dict": gain.state_dict(),
                "certificate_state_dict": certificate.state_dict(),
                "base_gains": base_gains,
                "base_transforms": base_transforms,
                "nu_values": NU_VALUES,
                "gain_trust_ratio": gain_trust_ratio,
                "certificate_log_scale": certificate_log_scale,
            },
            checkpoint_dir / f"grid-{grid_size}__seed-{selected_seed}.pt",
        )
    fixed_nu005 = validation_baselines["fixed-0.1"]["by_nu"]["0.005"][
        "terminal_error_mass_median"
    ]
    learned_nu005 = selected["validation_rollout"]["by_nu"]["0.005"][
        "terminal_error_mass_median"
    ]
    return {
        "kind": "r5-oblique-joint-training",
        "grid_size": grid_size,
        "device": device,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "refresh_interval": refresh_interval,
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "test_case_count": len(test_cases),
        "gain_trust_ratio": gain_trust_ratio,
        "certificate_log_scale": certificate_log_scale,
        "loss_weights": {
            "stable": stable_weight,
            "defect": defect_weight,
            "contraction": contraction_weight,
            "bi": bi_weight,
            "gain_reg": gain_reg_weight,
        },
        "base_diagnostics": base_diagnostics,
        "validation_baselines": validation_baselines,
        "selected_seed": selected_seed,
        "seed_results": seed_results,
        "test": test,
        "noisy_test": noisy_test,
        "test_evaluated": run_test,
        "gates": {
            "positive_worst_validation_contraction": bool(
                selected["validation_contraction"]["min"] > 0.0
            ),
            "nu005_validation_no_regression": bool(
                learned_nu005 <= fixed_nu005
            ),
            "multi_grid_expansion_eligible": bool(
                selected["validation_contraction"]["min"] > 0.0
                and learned_nu005 <= fixed_nu005
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit-per-nu", type=int, default=16)
    parser.add_argument("--validation-limit-per-nu", type=int, default=8)
    parser.add_argument("--test-limit-per-nu", type=int, default=8)
    parser.add_argument("--stress-truths-per-nu", type=int, default=2)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.25)
    parser.add_argument("--certificate-log-scale", type=float, default=0.2231435513)
    parser.add_argument("--gain-learning-rate", type=float, default=5e-4)
    parser.add_argument("--certificate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stable-weight", type=float, default=1.0)
    parser.add_argument("--defect-weight", type=float, default=0.05)
    parser.add_argument("--contraction-weight", type=float, default=10.0)
    parser.add_argument("--bi-weight", type=float, default=1.0)
    parser.add_argument("--gain-reg-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if not 0.0 < args.gain_trust_ratio < 1.0:
        raise SystemExit("--gain-trust-ratio must lie in (0, 1)")
    if args.certificate_log_scale < 0.0:
        raise SystemExit("--certificate-log-scale must be non-negative")
    if min(
        args.train_limit_per_nu,
        args.validation_limit_per_nu,
        args.test_limit_per_nu,
        args.stress_truths_per_nu,
    ) < 1:
        raise SystemExit("case limits must be positive")
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        torch,
        grid_size=args.grid_size,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        refresh_interval=args.refresh_interval,
        device=args.device,
        train_limit_per_nu=args.train_limit_per_nu,
        validation_limit_per_nu=args.validation_limit_per_nu,
        test_limit_per_nu=args.test_limit_per_nu,
        stress_truths_per_nu=args.stress_truths_per_nu,
        gain_trust_ratio=args.gain_trust_ratio,
        certificate_log_scale=args.certificate_log_scale,
        gain_learning_rate=args.gain_learning_rate,
        certificate_learning_rate=args.certificate_learning_rate,
        stable_weight=args.stable_weight,
        defect_weight=args.defect_weight,
        contraction_weight=args.contraction_weight,
        bi_weight=args.bi_weight,
        gain_reg_weight=args.gain_reg_weight,
        checkpoint_dir=args.checkpoint_dir,
        run_test=not args.validation_only,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"]), flush=True)


if __name__ == "__main__":
    main()
