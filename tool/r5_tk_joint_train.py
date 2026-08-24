"""R5 T--K-style joint training against the stable target map."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from r5_e_joint_train import INTERVALS, OUTPUT_TIMES
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    NullspaceCertificate,
    allen_cahn_energy,
    allen_cahn_rhs,
    generate_pilot_cases,
    local_average_matrix,
    noise_waveform,
    simulate_causal_nudging,
)


@dataclass(frozen=True)
class JointSampleSet:
    states: np.ndarray
    estimates: np.ndarray
    measurements: np.ndarray
    next_states: np.ndarray
    nus: np.ndarray
    nu_indices: np.ndarray
    nu_values: tuple[float, ...]
    times: np.ndarray
    dt: float


ABLATION_SEEDS = (501, 502, 503, 504)
MAX_POLICY_RHS_EVALUATIONS = 20_000


class PolicyRolloutBudgetExceeded(RuntimeError):
    """Raised when an adaptive policy rollout becomes computationally unsafe."""


def _assemble_sample_set(
    states: list[np.ndarray],
    estimates: list[np.ndarray],
    measurements: list[np.ndarray],
    next_states: list[np.ndarray],
    nus: list[float],
    times: list[float],
    *,
    dt: float,
) -> JointSampleSet:
    nu_array = np.asarray(nus, dtype=float)
    nu_values = tuple(sorted({float(value) for value in nu_array}))
    nu_lookup = {value: index for index, value in enumerate(nu_values)}
    nu_indices = np.asarray([nu_lookup[float(value)] for value in nu_array], dtype=int)
    return JointSampleSet(
        states=np.asarray(states, dtype=float),
        estimates=np.asarray(estimates, dtype=float),
        measurements=np.asarray(measurements, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
        nus=nu_array,
        nu_indices=nu_indices,
        nu_values=nu_values,
        times=np.asarray(times, dtype=float),
        dt=dt,
    )


def _concatenate_sample_sets(*sample_sets: JointSampleSet) -> JointSampleSet:
    if not sample_sets:
        raise ValueError("at least one sample set is required")
    dt = sample_sets[0].dt
    if any(not np.isclose(item.dt, dt) for item in sample_sets[1:]):
        raise ValueError("sample sets must use the same time step")
    return _assemble_sample_set(
        [row for item in sample_sets for row in item.states],
        [row for item in sample_sets for row in item.estimates],
        [row for item in sample_sets for row in item.measurements],
        [row for item in sample_sets for row in item.next_states],
        [float(value) for item in sample_sets for value in item.nus],
        [float(value) for item in sample_sets for value in item.times],
        dt=dt,
    )


def _split_cases(split: str, grid_size: int) -> list[object]:
    return [
        case
        for case in generate_pilot_cases()
        if case.split == split and case.n == grid_size
    ]


def _collect_samples(
    cases: list[object],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    base_gain: float,
) -> JointSampleSet:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    nus: list[float] = []
    times: list[float] = []
    dt = float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])
    for case in cases:
        rollout = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=base_gain),
            case.initial_truth(grid),
            case.initial_estimate(grid),
            output_times=OUTPUT_TIMES,
        )
        states.extend(rollout.truth[:-1])
        estimates.extend(rollout.estimate[:-1])
        measurements.extend(rollout.measurements[:-1])
        next_states.extend(rollout.truth[1:])
        nus.extend([case.nu] * (OUTPUT_TIMES.size - 1))
        times.extend(OUTPUT_TIMES[:-1])
    return _assemble_sample_set(
        states,
        estimates,
        measurements,
        next_states,
        nus,
        times,
        dt=dt,
    )


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


def _feature_numpy(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    estimate: np.ndarray,
    measurement: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - matrix @ estimate
    features = np.concatenate(
        (
            estimate,
            measurement,
            innovation,
            np.asarray([(nu - 0.01) / 0.01]),
            np.asarray([np.sqrt(grid.h * np.dot(estimate, estimate))]),
        )
    )
    return features, innovation


def _policy_rollout(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: object,
    *,
    noise: object = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = grid.n
    rhs_evaluations = 0

    def rhs(time: float, combined: np.ndarray) -> np.ndarray:
        nonlocal rhs_evaluations
        rhs_evaluations += 1
        if rhs_evaluations > MAX_POLICY_RHS_EVALUATIONS:
            raise PolicyRolloutBudgetExceeded(
                f"policy rollout exceeded {MAX_POLICY_RHS_EVALUATIONS} RHS evaluations"
            )
        truth, estimate = combined[:n], combined[n:]
        measurement = matrix @ truth
        if noise is not None:
            measurement = measurement + noise(float(time))
        features, innovation = _feature_numpy(
            grid, matrix, estimate, measurement, case.nu
        )
        with torch.no_grad():
            value = torch.as_tensor(
                features[None, :], dtype=torch.float32, device=device
            )
            gain_value = gain(value)[0].cpu().numpy()
        correction = gain_value @ innovation
        return np.concatenate(
            (
                allen_cahn_rhs(grid, case.nu, truth),
                allen_cahn_rhs(grid, case.nu, estimate) + correction,
            )
        )

    try:
        result = solve_ivp(
            rhs,
            (0.0, 1.0),
            np.concatenate((case.initial_truth(grid), case.initial_estimate(grid))),
            method="DOP853",
            t_eval=OUTPUT_TIMES,
            rtol=1.0e-8,
            atol=1.0e-10,
        )
    except PolicyRolloutBudgetExceeded:
        empty_states = np.empty((0, n), dtype=float)
        empty_measurements = np.empty((0, matrix.shape[0]), dtype=float)
        return empty_states, empty_states.copy(), empty_measurements, -2
    trajectories = result.y.T
    truth = trajectories[:, :n]
    estimate = trajectories[:, n:]
    measurements = np.asarray(
        [
            matrix @ state
            + (
                np.zeros(matrix.shape[0], dtype=float)
                if noise is None
                else noise(float(time))
            )
            for time, state in zip(result.t, truth, strict=True)
        ]
    )
    return truth, estimate, measurements, int(result.status)


def _collect_policy_samples(
    torch: object,
    gain: object,
    device: str,
    cases: list[object],
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    noise: object = None,
) -> JointSampleSet:
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    nus: list[float] = []
    times: list[float] = []
    dt = float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])
    for case in cases:
        truth, estimate, observed, status = _policy_rollout(
            torch, gain, device, grid, matrix, case, noise=noise
        )
        if status != 0 or truth.shape[0] != OUTPUT_TIMES.size:
            raise RuntimeError(f"on-policy rollout failed for {case.case_id}")
        states.extend(truth[:-1])
        estimates.extend(estimate[:-1])
        measurements.extend(observed[:-1])
        next_states.extend(truth[1:])
        nus.extend([case.nu] * (OUTPUT_TIMES.size - 1))
        times.extend(OUTPUT_TIMES[:-1])
    return _assemble_sample_set(
        states,
        estimates,
        measurements,
        next_states,
        nus,
        times,
        dt=dt,
    )


def _allen_cahn_rhs_tensor(
    torch: object, grid: AllenCahnGrid, states: object, nus: object, laplacian: object
) -> object:
    return nus[:, None] * (states @ laplacian.T) + states - states**3


def _target_operators(
    grid: AllenCahnGrid,
    nu_values: tuple[float, ...],
    lambda_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(grid.n, dtype=float)
    generators = []
    maps = []
    for nu in nu_values:
        linear = nu * grid.laplacian
        lam = lambda_ratio * nu * np.pi**2
        generator = linear - lam * identity
        generators.append(generator)
        maps.append(expm(grid_step(grid) * generator))
    return np.asarray(generators, dtype=float), np.asarray(maps, dtype=float)


def grid_step(grid: AllenCahnGrid) -> float:
    return float(OUTPUT_TIMES[1] - OUTPUT_TIMES[0])


def _build_models(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
    lower_lipschitz: float,
    upper_lipschitz: float,
    certificate_kind: str = "diagonal",
    mixing_layers: int = 0,
    shear_norm_limit: float = 0.0,
    gain_trust_ratio: float = 0.0,
    gain_kind: str = "dense",
) -> tuple[object, object]:
    nn = torch.nn
    n = grid.n
    q = matrix.shape[0]
    feature_dim = n + 2 * q + 2
    basis = NullspaceCertificate(matrix).null_basis
    row_basis = np.linalg.qr(matrix.T, mode="reduced")[0]
    if gain_kind not in {"dense", "mass-adjoint", "mass-adjoint-constant"}:
        raise ValueError(f"unknown gain kind: {gain_kind}")
    if gain_kind.startswith("mass-adjoint") and not 0.0 < gain_trust_ratio < 1.0:
        raise ValueError("mass-adjoint gain requires 0 < trust ratio < 1")

    class ConstantSensorGain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(q, dtype=torch.float32))

        def forward(self, features: object) -> object:
            return self.logits[None, :].expand(features.shape[0], -1)

    class GainNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if gain_kind == "mass-adjoint-constant":
                self.network = ConstantSensorGain()
            else:
                self.network = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, n * q if gain_kind == "dense" else q),
                )
                nn.init.zeros_(self.network[-1].weight)
                nn.init.zeros_(self.network[-1].bias)
            self.register_buffer(
                "base_gain",
                torch.as_tensor(base_gain * matrix.T / grid.h, dtype=torch.float32),
            )
            self.register_buffer(
                "base_gain_norm",
                torch.linalg.vector_norm(self.base_gain),
            )
            self.register_buffer(
                "injection_basis",
                torch.as_tensor(matrix.T / grid.h, dtype=torch.float32),
            )
            self.gain_trust_ratio = gain_trust_ratio
            self.gain_kind = gain_kind

        def forward(self, features: object) -> object:
            raw = self.network(features)
            if self.gain_kind.startswith("mass-adjoint"):
                sensor_gain = base_gain * (
                    1.0 + self.gain_trust_ratio * torch.tanh(raw)
                )
                return self.injection_basis[None, :, :] * sensor_gain[:, None, :]
            raw = raw.reshape(-1, n, q)
            delta = gain_scale * torch.tanh(raw)
            if self.gain_trust_ratio > 0.0:
                delta_norm = torch.linalg.vector_norm(
                    delta, dim=(1, 2), keepdim=True
                )
                limit = self.gain_trust_ratio * self.base_gain_norm
                scale = limit / (limit + delta_norm)
                delta = scale * delta
            return self.base_gain[None, :, :] + delta

    class CertificateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            null_dimension = basis.shape[1]
            if certificate_kind not in {"diagonal", "givens", "triangular"}:
                raise ValueError(f"unknown certificate kind: {certificate_kind}")
            if certificate_kind in {"givens", "triangular"} and mixing_layers < 1:
                raise ValueError("mixed certificate requires at least one mixing layer")
            if certificate_kind == "triangular" and not 0.0 < shear_norm_limit < 1.0:
                raise ValueError("triangular certificate requires 0 < shear limit < 1")
            if certificate_kind == "triangular":
                internal_lower = lower_lipschitz * (1.0 + shear_norm_limit)
                internal_upper = upper_lipschitz / (1.0 + shear_norm_limit)
                if not internal_lower < 1.0 < internal_upper:
                    raise ValueError("triangular internal scale bounds must contain 1")
            pair_count = null_dimension // 2
            output_dimension = null_dimension
            if certificate_kind in {"givens", "triangular"}:
                output_dimension += mixing_layers * pair_count
            if certificate_kind == "triangular":
                output_dimension += null_dimension * q
            self.network = nn.Sequential(
                nn.Linear(n, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, output_dimension),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.certificate_kind = certificate_kind
            self.null_dimension = null_dimension
            self.pair_count = pair_count
            self.mixing_layers = mixing_layers
            self.shear_norm_limit = shear_norm_limit
            self.register_buffer(
                "null_basis", torch.as_tensor(basis, dtype=torch.float32)
            )
            self.register_buffer(
                "row_basis", torch.as_tensor(row_basis, dtype=torch.float32)
            )

        def _rotate(
            self,
            values: object,
            angles: object,
            layer: int,
            *,
            inverse: bool,
        ) -> object:
            shifted = torch.roll(values, shifts=-layer, dims=1)
            left = shifted[:, : 2 * self.pair_count : 2]
            right = shifted[:, 1 : 2 * self.pair_count : 2]
            sine = torch.sin(angles)
            if inverse:
                sine = -sine
            cosine = torch.cos(angles)
            rotated_pairs = torch.stack(
                (cosine * left - sine * right, sine * left + cosine * right),
                dim=2,
            ).reshape(values.shape[0], 2 * self.pair_count)
            rotated = torch.cat(
                (rotated_pairs, shifted[:, 2 * self.pair_count :]), dim=1
            )
            return torch.roll(rotated, shifts=layer, dims=1)

        def forward(self, states: object, errors: object) -> object:
            coordinates = errors @ self.null_basis
            raw = self.network(states)
            if self.certificate_kind == "diagonal":
                scales = lower_lipschitz + (
                    upper_lipschitz - lower_lipschitz
                ) * torch.sigmoid(certificate_scale * raw)
                transformed_coordinates = scales * coordinates
            else:
                scale_lower = lower_lipschitz
                scale_upper = upper_lipschitz
                if self.certificate_kind == "triangular":
                    scale_lower *= 1.0 + self.shear_norm_limit
                    scale_upper /= 1.0 + self.shear_norm_limit
                scale_fraction = (1.0 - scale_lower) / (
                    scale_upper - scale_lower
                )
                identity_logit = float(np.log(scale_fraction / (1.0 - scale_fraction)))
                scales = scale_lower + (scale_upper - scale_lower) * torch.sigmoid(
                    identity_logit
                    + certificate_scale * raw[:, : self.null_dimension]
                )
                angle_end = (
                    self.null_dimension + self.mixing_layers * self.pair_count
                )
                angles = (np.pi / 4.0) * torch.tanh(
                    certificate_scale
                    * raw[:, self.null_dimension : angle_end].reshape(
                        -1, self.mixing_layers, self.pair_count
                    )
                )
                transformed_coordinates = coordinates
                if self.certificate_kind == "triangular":
                    shear_candidate = torch.tanh(
                        certificate_scale
                        * raw[:, angle_end:].reshape(
                            -1, self.null_dimension, q
                        )
                    )
                    shear_squared_norm = torch.sum(
                        shear_candidate**2, dim=(1, 2), keepdim=True
                    )
                    shear = (
                        self.shear_norm_limit
                        * shear_candidate
                        / torch.sqrt(torch.clamp(shear_squared_norm, min=1.0))
                    )
                    observed_coordinates = errors @ self.row_basis
                    transformed_coordinates = transformed_coordinates + torch.bmm(
                        shear, observed_coordinates[:, :, None]
                    ).squeeze(-1)
                for layer in reversed(range(self.mixing_layers)):
                    transformed_coordinates = self._rotate(
                        transformed_coordinates,
                        angles[:, layer],
                        layer,
                        inverse=True,
                    )
                transformed_coordinates = scales * transformed_coordinates
                for layer in range(self.mixing_layers):
                    transformed_coordinates = self._rotate(
                        transformed_coordinates,
                        angles[:, layer],
                        layer,
                        inverse=False,
                    )
            return errors + (transformed_coordinates - coordinates) @ self.null_basis.T

    return GainNet(), CertificateNet()


def _joint_loss_components(
    torch: object,
    gain: object,
    certificate: object,
    samples: dict[str, object],
    target_generators: object,
    target_maps: object,
    grid: AllenCahnGrid,
    matrix: object,
    indices: object,
    *,
    stable_normalization: str,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    contraction_margin_ratio: float,
    bi_weight: float,
    gain_reg_weight: float,
    lower_lipschitz: float,
    upper_lipschitz: float,
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
    gains = gain(features)
    correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
    laplacian = samples["laplacian"]
    rhs_truth = _allen_cahn_rhs_tensor(torch, grid, states, nus, laplacian)
    rhs_estimate = _allen_cahn_rhs_tensor(torch, grid, estimates, nus, laplacian)
    error_rhs = rhs_estimate + correction - rhs_truth
    next_estimates = estimates + samples["dt"] * (rhs_estimate + correction)
    errors = estimates - states
    next_errors = next_estimates - next_states
    transformed = certificate(states, errors)
    next_transformed = certificate(next_states, next_errors)
    stable_target = torch.bmm(target_maps[nu_indices], transformed[:, :, None]).squeeze(
        -1
    )
    residual = next_transformed - stable_target
    error_squared_mass = grid.h * torch.sum(errors**2, dim=1)
    stable_squared_mass = grid.h * torch.sum(residual**2, dim=1)
    stable_raw_loss = torch.mean(stable_squared_mass)
    if stable_normalization == "error-time":
        stable_loss = torch.mean(
            stable_squared_mass
            / (samples["dt"] ** 2 * (error_squared_mass + 1.0e-8))
        )
    elif stable_normalization == "none":
        stable_loss = stable_raw_loss
    else:
        raise ValueError(f"unknown stable normalization: {stable_normalization}")

    def transform(state: object, error: object) -> object:
        return certificate(state, error)

    _, directional = torch.autograd.functional.jvp(
        transform,
        (states, errors),
        (rhs_truth, error_rhs),
        create_graph=True,
    )
    generator = torch.bmm(
        target_generators[nu_indices], transformed[:, :, None]
    ).squeeze(-1)
    defect_residual = directional - generator
    defect_squared_mass = grid.h * torch.sum(defect_residual**2, dim=1)
    defect_loss = torch.mean(defect_squared_mass / (error_squared_mass + 1.0e-8))
    contraction_metrics = _contraction_tensor_metrics(
        torch,
        transformed,
        directional,
        nus,
        h=grid.h,
        margin_ratio=contraction_margin_ratio,
    )
    contraction_loss = contraction_metrics["loss"]

    error_norm = torch.sqrt(error_squared_mass + 1.0e-12)
    transformed_norm = torch.sqrt(
        grid.h * torch.sum(transformed**2, dim=1) + 1.0e-12
    )
    lower_violation = torch.relu(lower_lipschitz * error_norm - transformed_norm)
    upper_violation = torch.relu(transformed_norm - upper_lipschitz * error_norm)
    bi_loss = torch.mean(lower_violation**2 + upper_violation**2)
    gain_deviation_loss = torch.mean(
        torch.sum((gains - gain.base_gain[None, :, :]) ** 2, dim=(1, 2))
        / (gain.base_gain_norm**2 + 1.0e-12)
    )
    total_loss = (
        stable_weight * stable_loss
        + defect_weight * defect_loss
        + contraction_weight * contraction_loss
        + bi_weight * bi_loss
        + gain_reg_weight * gain_deviation_loss
    )
    return {
        "stable": stable_loss,
        "stable_raw": stable_raw_loss,
        "defect": defect_loss,
        "contraction": contraction_loss,
        "bi": bi_loss,
        "gain_deviation": gain_deviation_loss,
        "total": total_loss,
    }


def _tensorize_samples(
    torch: object,
    sample_set: JointSampleSet,
    grid: AllenCahnGrid,
    device: str,
) -> dict[str, object]:
    return {
        "states": torch.as_tensor(
            sample_set.states, dtype=torch.float32, device=device
        ),
        "estimates": torch.as_tensor(
            sample_set.estimates, dtype=torch.float32, device=device
        ),
        "measurements": torch.as_tensor(
            sample_set.measurements, dtype=torch.float32, device=device
        ),
        "next_states": torch.as_tensor(
            sample_set.next_states, dtype=torch.float32, device=device
        ),
        "nus": torch.as_tensor(sample_set.nus, dtype=torch.float32, device=device),
        "nu_indices": torch.as_tensor(
            sample_set.nu_indices, dtype=torch.long, device=device
        ),
        "times": torch.as_tensor(
            sample_set.times, dtype=torch.float32, device=device
        ),
        "laplacian": torch.as_tensor(
            grid.laplacian, dtype=torch.float32, device=device
        ),
        "dt": sample_set.dt,
    }


def _train_one(
    torch: object,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    train: JointSampleSet,
    train_cases: list[object],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
    lambda_ratio: float,
    stable_normalization: str,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    contraction_margin_ratio: float,
    bi_weight: float,
    gain_reg_weight: float,
    lower_lipschitz: float,
    upper_lipschitz: float,
    refresh_interval: int,
    certificate_kind: str = "diagonal",
    mixing_layers: int = 0,
    shear_norm_limit: float = 0.0,
    replay_snapshots: int = 0,
    gain_warmup_epochs: int = 0,
    certificate_warmup_epochs: int = 0,
    gain_learning_rate: float = 2.0e-3,
    certificate_learning_rate: float = 2.0e-3,
    gradient_clip_norm: float = 0.0,
    gain_trust_ratio: float = 0.0,
    gain_kind: str = "dense",
) -> tuple[object, object, dict[str, float | int]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=base_gain,
        gain_scale=gain_scale,
        certificate_scale=certificate_scale,
        lower_lipschitz=lower_lipschitz,
        upper_lipschitz=upper_lipschitz,
        certificate_kind=certificate_kind,
        mixing_layers=mixing_layers,
        shear_norm_limit=shear_norm_limit,
        gain_trust_ratio=gain_trust_ratio,
        gain_kind=gain_kind,
    )
    gain.to(device)
    certificate.to(device)
    samples = _tensorize_samples(torch, train, grid, device)
    target_generators, target_maps = _target_operators(
        grid, train.nu_values, lambda_ratio
    )
    target_generators_tensor = torch.as_tensor(
        target_generators, dtype=torch.float32, device=device
    )
    target_maps_tensor = torch.as_tensor(
        target_maps,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(
        (
            {"params": list(gain.parameters()), "lr": gain_learning_rate},
            {
                "params": list(certificate.parameters()),
                "lr": certificate_learning_rate,
            },
        )
    )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    sample_count = samples["states"].shape[0]
    refresh_count = 0
    policy_replay: list[JointSampleSet] = []
    for epoch in range(epochs):
        gain_active = epoch < gain_warmup_epochs or epoch >= (
            gain_warmup_epochs + certificate_warmup_epochs
        )
        certificate_active = epoch >= gain_warmup_epochs
        for parameter in gain.parameters():
            parameter.requires_grad_(gain_active)
        for parameter in certificate.parameters():
            parameter.requires_grad_(certificate_active)
        if epoch > 0 and refresh_interval > 0 and epoch % refresh_interval == 0:
            gain.eval()
            refreshed = _collect_policy_samples(
                torch, gain, device, train_cases, grid, matrix
            )
            if replay_snapshots > 0:
                policy_replay.append(refreshed)
                policy_replay = policy_replay[-replay_snapshots:]
                replay = _concatenate_sample_sets(train, *policy_replay)
            else:
                replay = refreshed
            samples = _tensorize_samples(torch, replay, grid, device)
            sample_count = samples["states"].shape[0]
            gain.train()
            certificate.train()
            refresh_count += 1
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            components = _joint_loss_components(
                torch,
                gain,
                certificate,
                samples,
                target_generators_tensor,
                target_maps_tensor,
                grid,
                matrix_tensor,
                indices,
                stable_normalization=stable_normalization,
                stable_weight=stable_weight,
                defect_weight=defect_weight,
                contraction_weight=contraction_weight,
                contraction_margin_ratio=contraction_margin_ratio,
                bi_weight=bi_weight,
                gain_reg_weight=gain_reg_weight,
                lower_lipschitz=lower_lipschitz,
                upper_lipschitz=upper_lipschitz,
            )
            loss = components["total"]
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError(
                    f"non-finite joint loss at epoch={epoch}, batch={start}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters_with_gradients = [
                parameter
                for parameter in list(gain.parameters())
                + list(certificate.parameters())
                if parameter.grad is not None
            ]
            if any(
                not bool(
                    torch.all(torch.isfinite(parameter.grad)).detach().cpu().item()
                )
                for parameter in parameters_with_gradients
            ):
                raise RuntimeError(
                    f"non-finite joint gradient at epoch={epoch}, batch={start}"
                )
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    parameters_with_gradients,
                    gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            optimizer.step()
        history.append(
            {
                name: float(value.detach().cpu().item())
                for name, value in components.items()
            }
        )
    gain.eval()
    certificate.eval()
    with torch.enable_grad():
        indices = torch.arange(sample_count, device=device)
        final_components = _joint_loss_components(
            torch,
            gain,
            certificate,
            samples,
            target_generators_tensor,
            target_maps_tensor,
            grid,
            matrix_tensor,
            indices,
            stable_normalization=stable_normalization,
            stable_weight=stable_weight,
            defect_weight=defect_weight,
            contraction_weight=contraction_weight,
            contraction_margin_ratio=contraction_margin_ratio,
            bi_weight=bi_weight,
            gain_reg_weight=gain_reg_weight,
            lower_lipschitz=lower_lipschitz,
            upper_lipschitz=upper_lipschitz,
        )
        final_losses = {
            name: float(value.detach().cpu().item())
            for name, value in final_components.items()
        }
    return (
        gain,
        certificate,
        {
            "stable_training_loss": final_losses["stable"],
            "stable_raw_training_loss": final_losses["stable_raw"],
            "defect_training_loss": final_losses["defect"],
            "contraction_training_loss": final_losses["contraction"],
            "bi_training_loss": final_losses["bi"],
            "gain_deviation_training_loss": final_losses["gain_deviation"],
            "total_training_loss": final_losses["total"],
            "stable_initial_last_batch_loss": history[0]["stable"],
            "stable_final_last_batch_loss": history[-1]["stable"],
            "stable_raw_initial_last_batch_loss": history[0]["stable_raw"],
            "stable_raw_final_last_batch_loss": history[-1]["stable_raw"],
            "defect_initial_last_batch_loss": history[0]["defect"],
            "defect_final_last_batch_loss": history[-1]["defect"],
            "contraction_initial_last_batch_loss": history[0]["contraction"],
            "contraction_final_last_batch_loss": history[-1]["contraction"],
            "bi_initial_last_batch_loss": history[0]["bi"],
            "bi_final_last_batch_loss": history[-1]["bi"],
            "gain_deviation_initial_last_batch_loss": history[0]["gain_deviation"],
            "gain_deviation_final_last_batch_loss": history[-1]["gain_deviation"],
            "total_initial_last_batch_loss": history[0]["total"],
            "total_final_last_batch_loss": history[-1]["total"],
            "on_policy_refresh_count": refresh_count,
            "training_sample_count": int(sample_count),
            "replay_snapshot_count": len(policy_replay),
            "gain_learning_rate": gain_learning_rate,
            "certificate_learning_rate": certificate_learning_rate,
            "gradient_clip_norm": gradient_clip_norm,
            "gain_trust_ratio": gain_trust_ratio,
            "gain_kind": gain_kind,
            "gain_reg_weight": gain_reg_weight,
            "contraction_weight": contraction_weight,
            "contraction_margin_ratio": contraction_margin_ratio,
        },
    )


def _validation_loss(
    torch: object,
    gain: object,
    certificate: object,
    validation: JointSampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    lambda_ratio: float,
    stable_normalization: str,
    stable_weight: float,
    defect_weight: float,
    contraction_weight: float,
    contraction_margin_ratio: float,
    bi_weight: float,
    gain_reg_weight: float,
    lower_lipschitz: float,
    upper_lipschitz: float,
    device: str,
) -> dict[str, float]:
    samples = _tensorize_samples(torch, validation, grid, device)
    target_generators, target_maps = _target_operators(
        grid, validation.nu_values, lambda_ratio
    )
    target_generators_tensor = torch.as_tensor(
        target_generators, dtype=torch.float32, device=device
    )
    target_maps_tensor = torch.as_tensor(
        target_maps,
        dtype=torch.float32,
        device=device,
    )
    with torch.enable_grad():
        indices = torch.arange(samples["states"].shape[0], device=device)
        components = _joint_loss_components(
            torch,
            gain,
            certificate,
            samples,
            target_generators_tensor,
            target_maps_tensor,
            grid,
            torch.as_tensor(matrix, dtype=torch.float32, device=device),
            indices,
            stable_normalization=stable_normalization,
            stable_weight=stable_weight,
            defect_weight=defect_weight,
            contraction_weight=contraction_weight,
            contraction_margin_ratio=contraction_margin_ratio,
            bi_weight=bi_weight,
            gain_reg_weight=gain_reg_weight,
            lower_lipschitz=lower_lipschitz,
            upper_lipschitz=upper_lipschitz,
        )
        return {
            name: float(value.detach().cpu().item())
            for name, value in components.items()
        }


def _ratio_summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "rms": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _signed_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize signed sample values without hiding the worst case in an RMS."""
    if values.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "positive_fraction": float("nan"),
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def _contraction_tensor_metrics(
    torch: object,
    transformed: object,
    directional: object,
    nus: object,
    *,
    h: float,
    margin_ratio: float,
) -> dict[str, object]:
    """Return the direct energy-contraction rate and its normalized hinge loss.

    For ``z = T_phi(e)``, the reported rate is

        - <z, d_t z>_M / ||z||_M^2.

    A positive value is a finite-sample contraction margin.  The optional
    requested margin is ``margin_ratio * nu * pi**2`` so it remains tied to the
    first physical diffusion rate on every grid.
    """
    transformed_squared_mass = h * torch.sum(transformed**2, dim=1)
    energy_pairing = h * torch.sum(transformed * directional, dim=1)
    requested_margin = margin_ratio * nus * np.pi**2
    denominator = transformed_squared_mass + 1.0e-8
    rates = -energy_pairing / denominator
    violation = torch.relu(
        energy_pairing + requested_margin * transformed_squared_mass
    )
    loss = torch.mean(violation**2 / (transformed_squared_mass**2 + 1.0e-12))
    return {
        "rates": rates,
        "requested_margin": requested_margin,
        "transformed_squared_mass": transformed_squared_mass,
        "energy_pairing": energy_pairing,
        "loss": loss,
    }


def _contraction_audit(
    torch: object,
    gain: object,
    certificate: object,
    sample_set: JointSampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    margin_ratio: float,
    device: str,
    batch_size: int = 512,
) -> dict[str, object]:
    """Audit direct contraction of ``z=T_phi(e)`` on a finite trajectory set."""
    samples = _tensorize_samples(torch, sample_set, grid, device)
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    rates: list[np.ndarray] = []
    requested_margins: list[np.ndarray] = []
    error_norms: list[np.ndarray] = []
    transformed_norms: list[np.ndarray] = []
    losses: list[tuple[int, float]] = []
    count = samples["states"].shape[0]
    with torch.enable_grad():
        for start in range(0, count, batch_size):
            indices = torch.arange(
                start, min(start + batch_size, count), device=device
            )
            states = samples["states"][indices]
            estimates = samples["estimates"][indices]
            measurements = samples["measurements"][indices]
            nus = samples["nus"][indices]
            features, innovations = _feature_tensor(
                torch, estimates, measurements, nus, matrix_tensor, grid.h
            )
            gains = gain(features)
            correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
            rhs_truth = _allen_cahn_rhs_tensor(
                torch, grid, states, nus, samples["laplacian"]
            )
            rhs_estimate = _allen_cahn_rhs_tensor(
                torch, grid, estimates, nus, samples["laplacian"]
            )
            errors = estimates - states
            error_rhs = rhs_estimate + correction - rhs_truth

            def transform(state: object, error: object) -> object:
                return certificate(state, error)

            transformed = certificate(states, errors)
            _, directional = torch.autograd.functional.jvp(
                transform,
                (states, errors),
                (rhs_truth, error_rhs),
                create_graph=False,
            )
            metrics = _contraction_tensor_metrics(
                torch,
                transformed,
                directional,
                nus,
                h=grid.h,
                margin_ratio=margin_ratio,
            )
            batch_count = int(indices.shape[0])
            rates.append(metrics["rates"].detach().cpu().numpy())
            requested_margins.append(
                metrics["requested_margin"].detach().cpu().numpy()
            )
            error_norms.append(
                torch.sqrt(grid.h * torch.sum(errors**2, dim=1))
                .detach()
                .cpu()
                .numpy()
            )
            transformed_norms.append(
                torch.sqrt(metrics["transformed_squared_mass"])
                .detach()
                .cpu()
                .numpy()
            )
            losses.append((batch_count, float(metrics["loss"].detach().cpu().item())))

    rate_values = np.concatenate(rates)
    margin_values = np.concatenate(requested_margins)
    error_values = np.concatenate(error_norms)
    transformed_values = np.concatenate(transformed_norms)
    by_nu: dict[str, object] = {}
    for nu_index, nu in enumerate(sample_set.nu_values):
        mask = sample_set.nu_indices == nu_index
        summary = _signed_summary(rate_values[mask])
        requested_margin = float(margin_ratio * nu * np.pi**2)
        by_nu[f"{nu:.6g}"] = {
            **summary,
            "requested_margin": requested_margin,
            "requested_margin_fraction": float(
                np.mean(rate_values[mask] >= requested_margin)
            ),
            "positive_worst_sample_margin": bool(summary["min"] > 0.0),
            "requested_margin_passed": bool(summary["min"] >= requested_margin),
        }
    time_masks = {
        "early": sample_set.times < 0.25,
        "middle": (sample_set.times >= 0.25) & (sample_set.times < 0.75),
        "late": sample_set.times >= 0.75,
    }
    lower_error, upper_error = np.quantile(error_values, [0.25, 0.75])
    error_masks = {
        "small": error_values <= lower_error,
        "middle": (error_values > lower_error) & (error_values < upper_error),
        "large": error_values >= upper_error,
    }
    total_weight = sum(weight for weight, _value in losses)
    audit_loss = sum(weight * value for weight, value in losses) / total_weight
    overall = _signed_summary(rate_values)
    return {
        "definition": "-<T_phi(e), d_t T_phi(e)>_M / ||T_phi(e)||_M^2",
        "requested_margin_ratio": margin_ratio,
        "overall": {
            **overall,
            "requested_margin_fraction": float(
                np.mean(rate_values >= margin_values)
            ),
            "positive_worst_sample_margin": bool(overall["min"] > 0.0),
            "requested_margin_passed": bool(np.all(rate_values >= margin_values)),
        },
        "by_nu": by_nu,
        "by_time": {
            name: _signed_summary(rate_values[mask])
            for name, mask in time_masks.items()
        },
        "by_error_size": {
            name: _signed_summary(rate_values[mask])
            for name, mask in error_masks.items()
        },
        "error_norm": _ratio_summary(error_values),
        "transformed_norm": _ratio_summary(transformed_values),
        "normalized_hinge_loss": float(audit_loss),
        "finite_sample_only": True,
    }


def _defect_audit(
    torch: object,
    gain: object,
    certificate: object,
    sample_set: JointSampleSet,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    *,
    lambda_ratio: float,
    lower_lipschitz: float,
    gain_scale: float,
    device: str,
    batch_size: int = 512,
) -> dict[str, object]:
    samples = _tensorize_samples(torch, sample_set, grid, device)
    target_generators, _target_maps = _target_operators(
        grid, sample_set.nu_values, lambda_ratio
    )
    target_generators_tensor = torch.as_tensor(
        target_generators, dtype=torch.float32, device=device
    )
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    ratios: list[np.ndarray] = []
    identity_ratios: list[np.ndarray] = []
    no_correction_ratios: list[np.ndarray] = []
    error_norms: list[np.ndarray] = []
    innovation_norms: list[np.ndarray] = []
    saturation: list[np.ndarray] = []
    gain_deviation_ratios: list[np.ndarray] = []
    count = samples["states"].shape[0]
    with torch.enable_grad():
        for start in range(0, count, batch_size):
            indices = torch.arange(
                start, min(start + batch_size, count), device=device
            )
            states = samples["states"][indices]
            estimates = samples["estimates"][indices]
            measurements = samples["measurements"][indices]
            nus = samples["nus"][indices]
            nu_indices = samples["nu_indices"][indices]
            features, innovations = _feature_tensor(
                torch, estimates, measurements, nus, matrix_tensor, grid.h
            )
            raw_gain = gain.network(features)
            gains = gain(features)
            correction = torch.bmm(gains, innovations[:, :, None]).squeeze(-1)
            rhs_truth = _allen_cahn_rhs_tensor(
                torch, grid, states, nus, samples["laplacian"]
            )
            rhs_estimate = _allen_cahn_rhs_tensor(
                torch, grid, estimates, nus, samples["laplacian"]
            )
            errors = estimates - states
            error_rhs = rhs_estimate + correction - rhs_truth
            error_rhs_no_correction = rhs_estimate - rhs_truth

            def transform(state: object, error: object) -> object:
                return certificate(state, error)

            transformed = certificate(states, errors)
            _, directional = torch.autograd.functional.jvp(
                transform,
                (states, errors),
                (rhs_truth, error_rhs),
                create_graph=False,
            )
            _, directional_no_correction = torch.autograd.functional.jvp(
                transform,
                (states, errors),
                (rhs_truth, error_rhs_no_correction),
                create_graph=False,
            )
            generator = torch.bmm(
                target_generators_tensor[nu_indices], transformed[:, :, None]
            ).squeeze(-1)
            identity_generator = torch.bmm(
                target_generators_tensor[nu_indices], errors[:, :, None]
            ).squeeze(-1)
            error_squared_mass = grid.h * torch.sum(errors**2, dim=1)

            def relative_norm(
                residual: object, denominator: object = error_squared_mass
            ) -> object:
                squared_mass = grid.h * torch.sum(residual**2, dim=1)
                return torch.sqrt(squared_mass / (denominator + 1.0e-8))

            ratios.append(
                relative_norm(directional - generator).detach().cpu().numpy()
            )
            identity_ratios.append(
                relative_norm(error_rhs - identity_generator).detach().cpu().numpy()
            )
            no_correction_ratios.append(
                relative_norm(directional_no_correction - generator)
                .detach()
                .cpu()
                .numpy()
            )
            error_norms.append(
                torch.sqrt(error_squared_mass).detach().cpu().numpy()
            )
            innovation_norms.append(
                torch.linalg.vector_norm(innovations, dim=1).detach().cpu().numpy()
            )
            saturation.append(
                torch.mean(
                    (torch.abs(torch.tanh(raw_gain)) >= 0.95).to(torch.float32),
                    dim=1,
                ).detach().cpu().numpy()
            )
            gain_deviation_ratios.append(
                (
                    torch.linalg.vector_norm(
                        gains - gain.base_gain[None, :, :], dim=(1, 2)
                    )
                    / (gain.base_gain_norm + 1.0e-12)
                )
                .detach()
                .cpu()
                .numpy()
            )

    ratio_values = np.concatenate(ratios)
    identity_values = np.concatenate(identity_ratios)
    no_correction_values = np.concatenate(no_correction_ratios)
    error_values = np.concatenate(error_norms)
    innovation_values = np.concatenate(innovation_norms)
    saturation_values = np.concatenate(saturation)
    gain_deviation_values = np.concatenate(gain_deviation_ratios)
    by_nu: dict[str, object] = {}
    for nu_index, nu in enumerate(sample_set.nu_values):
        mask = sample_set.nu_indices == nu_index
        alpha = float(-np.max(np.linalg.eigvalsh(target_generators[nu_index])))
        threshold = lower_lipschitz * alpha
        summary = _ratio_summary(ratio_values[mask])
        by_nu[f"{nu:.6g}"] = {
            **summary,
            "target_slowest_decay": alpha,
            "conservative_defect_threshold": threshold,
            "rms_gate_passed": bool(summary["rms"] < threshold),
            "sample_max_gate_passed": bool(summary["max"] < threshold),
        }
    time_masks = {
        "early": sample_set.times < 0.25,
        "middle": (sample_set.times >= 0.25) & (sample_set.times < 0.75),
        "late": sample_set.times >= 0.75,
    }
    lower_error, upper_error = np.quantile(error_values, [0.25, 0.75])
    error_masks = {
        "small": error_values <= lower_error,
        "middle": (error_values > lower_error) & (error_values < upper_error),
        "large": error_values >= upper_error,
    }
    return {
        "overall": _ratio_summary(ratio_values),
        "identity_transform": _ratio_summary(identity_values),
        "without_observer_correction": _ratio_summary(no_correction_values),
        "by_nu": by_nu,
        "by_time": {
            name: _ratio_summary(ratio_values[mask])
            for name, mask in time_masks.items()
        },
        "by_error_size": {
            name: _ratio_summary(ratio_values[mask])
            for name, mask in error_masks.items()
        },
        "error_norm": _ratio_summary(error_values),
        "innovation_norm": _ratio_summary(innovation_values),
        "gain_saturation_fraction_mean": float(np.mean(saturation_values)),
        "gain_deviation_ratio": _ratio_summary(gain_deviation_values),
        "gain_trust_boundary_fraction": float(
            np.mean(
                gain_deviation_values
                >= 0.99 * float(getattr(gain, "gain_trust_ratio", 0.0))
            )
        )
        if getattr(gain, "gain_trust_ratio", 0.0) > 0.0
        else 0.0,
        "all_rms_gates_passed": bool(
            all(item["rms_gate_passed"] for item in by_nu.values())
        ),
        "all_sample_max_gates_passed": bool(
            all(item["sample_max_gate_passed"] for item in by_nu.values())
        ),
    }


def _simulate(
    torch: object,
    gain: object,
    device: str,
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: object,
    *,
    noise: object = None,
) -> dict[str, float | int]:
    truth, estimate, _measurements, solver_status = _policy_rollout(
        torch,
        gain,
        device,
        grid,
        matrix,
        case,
        noise=noise,
    )
    if solver_status != 0 or truth.shape[0] != OUTPUT_TIMES.size:
        raise RuntimeError("online observer rollout exceeded its solver budget")
    error = estimate - truth
    error_mass = np.sqrt(grid.h * np.sum(error**2, axis=1))
    energies = np.asarray(
        [allen_cahn_energy(grid, case.nu, state) for state in estimate]
    )
    return {
        "solver_status": solver_status,
        "terminal_error_mass": float(error_mass[-1]),
        "peak_error_mass": float(np.max(error_mass)),
        "energy_defect": float(
            max(0.0, np.max(np.diff(energies, prepend=energies[0])))
        ),
    }


def _simulate_fixed_gain(
    grid: AllenCahnGrid,
    matrix: np.ndarray,
    case: object,
    *,
    gain: float,
) -> dict[str, float | int]:
    rollout = simulate_causal_nudging(
        CausalNudging(grid, case.nu, matrix, gain=gain),
        case.initial_truth(grid),
        case.initial_estimate(grid),
        output_times=OUTPUT_TIMES,
    )
    error = rollout.estimate - rollout.truth
    error_mass = np.sqrt(grid.h * np.sum(error**2, axis=1))
    return {
        "solver_status": int(rollout.solver_status),
        "terminal_error_mass": float(error_mass[-1]),
        "peak_error_mass": float(np.max(error_mass)),
    }


def _median(records: list[dict[str, float | int]], key: str) -> float:
    return float(np.median([record[key] for record in records]))


def _audit(
    torch: object,
    certificate: object,
    matrix: np.ndarray,
    grid: AllenCahnGrid,
    device: str,
) -> dict[str, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(20000 + grid.n))
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
    *,
    epochs: int,
    batch_size: int,
    eval_limit: int,
    noise_limit: int,
    device: str,
    lambda_ratio: float,
    base_gain: float,
    gain_scale: float,
    certificate_scale: float,
    stable_normalization: str,
    stable_weight: float,
    defect_weight: float,
    bi_weight: float,
    lower_lipschitz: float,
    upper_lipschitz: float,
    refresh_interval: int,
    selection_limit: int,
    selection_baseline_gain: float,
    certificate_kind: str = "diagonal",
    mixing_layers: int = 0,
    shear_norm_limit: float = 0.0,
    replay_snapshots: int = 0,
    gain_warmup_epochs: int = 0,
    certificate_warmup_epochs: int = 0,
    gain_learning_rate: float = 2.0e-3,
    certificate_learning_rate: float = 2.0e-3,
    gradient_clip_norm: float = 0.0,
    gain_trust_ratio: float = 0.0,
    gain_reg_weight: float = 0.0,
    gain_kind: str = "dense",
    selection_mode: str = "rollout-first",
    run_defect_audit: bool = False,
    contraction_weight: float = 0.0,
    contraction_margin_ratio: float = 0.0,
    run_contraction_audit: bool = False,
    checkpoint_dir: Path | None = None,
) -> dict[str, object]:
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for grid_size in grid_sizes:
        grid = AllenCahnGrid(grid_size)
        matrix = local_average_matrix(grid, INTERVALS)
        train_cases = _split_cases("train", grid_size)
        validation_cases = _split_cases("validation", grid_size)
        train = _collect_samples(
            train_cases, grid, matrix, base_gain=base_gain
        )
        validation = _collect_samples(
            validation_cases, grid, matrix, base_gain=base_gain
        )
        test_cases = _split_cases("test", grid_size)
        selection_cases = validation_cases[:selection_limit]
        baseline_selection = [
            _simulate_fixed_gain(
                grid, matrix, case, gain=selection_baseline_gain
            )
            for case in selection_cases
        ]
        models: dict[int, tuple[object, object]] = {}
        seed_results = []
        print(f"[grid={grid_size}] training {len(seeds)} seeds", flush=True)
        for seed in seeds:
            gain, certificate, losses = _train_one(
                torch,
                grid,
                matrix,
                train,
                train_cases,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                base_gain=base_gain,
                gain_scale=gain_scale,
                certificate_scale=certificate_scale,
                lambda_ratio=lambda_ratio,
                stable_normalization=stable_normalization,
                stable_weight=stable_weight,
                defect_weight=defect_weight,
                contraction_weight=contraction_weight,
                contraction_margin_ratio=contraction_margin_ratio,
                bi_weight=bi_weight,
                gain_reg_weight=gain_reg_weight,
                lower_lipschitz=lower_lipschitz,
                upper_lipschitz=upper_lipschitz,
                refresh_interval=refresh_interval,
                certificate_kind=certificate_kind,
                mixing_layers=mixing_layers,
                shear_norm_limit=shear_norm_limit,
                replay_snapshots=replay_snapshots,
                gain_warmup_epochs=gain_warmup_epochs,
                certificate_warmup_epochs=certificate_warmup_epochs,
                gain_learning_rate=gain_learning_rate,
                certificate_learning_rate=certificate_learning_rate,
                gradient_clip_norm=gradient_clip_norm,
                gain_trust_ratio=gain_trust_ratio,
                gain_kind=gain_kind,
            )
            validation_loss = _validation_loss(
                torch,
                gain,
                certificate,
                validation,
                grid,
                matrix,
                lambda_ratio=lambda_ratio,
                stable_normalization=stable_normalization,
                stable_weight=stable_weight,
                defect_weight=defect_weight,
                contraction_weight=contraction_weight,
                contraction_margin_ratio=contraction_margin_ratio,
                bi_weight=bi_weight,
                gain_reg_weight=gain_reg_weight,
                lower_lipschitz=lower_lipschitz,
                upper_lipschitz=upper_lipschitz,
                device=device,
            )
            selection_replay = [
                _simulate(torch, gain, device, grid, matrix, case)
                for case in selection_cases
            ]
            selection_contraction: dict[str, object] = {}
            if selection_mode == "contraction-first":
                selection_policy = _collect_policy_samples(
                    torch, gain, device, selection_cases, grid, matrix
                )
                selection_contraction = _contraction_audit(
                    torch,
                    gain,
                    certificate,
                    selection_policy,
                    grid,
                    matrix,
                    margin_ratio=contraction_margin_ratio,
                    device=device,
                    batch_size=batch_size,
                )
            certificate_audit = _audit(
                torch, certificate, matrix, grid, device
            )
            seed_results.append(
                {
                    "seed": seed,
                    **losses,
                    "stable_validation_loss": validation_loss["stable"],
                    "stable_raw_validation_loss": validation_loss["stable_raw"],
                    "defect_validation_loss": validation_loss["defect"],
                    "contraction_validation_loss": validation_loss["contraction"],
                    "bi_validation_loss": validation_loss["bi"],
                    "total_validation_loss": validation_loss["total"],
                    "selection_case_count": len(selection_replay),
                    "validation_median_terminal_error_mass": _median(
                        selection_replay, "terminal_error_mass"
                    ),
                    "certificate_audit": certificate_audit,
                    "selection_contraction_audit": selection_contraction,
                }
            )
            models[seed] = (gain, certificate)
        eligible = [
            item
            for item in seed_results
            if item["certificate_audit"]["min_jacobian_singular_value"]
            >= lower_lipschitz - 1.0e-5
            and item["certificate_audit"]["max_zero_fiber_residual"] <= 1.0e-7
            and item["certificate_audit"]["max_direction_residual"] <= 1.0e-7
        ]
        if selection_mode == "defect-first":
            selection_key = lambda item: (
                item["defect_validation_loss"],
                item["validation_median_terminal_error_mass"],
            )
        elif selection_mode == "rollout-first":
            selection_key = lambda item: (
                item["validation_median_terminal_error_mass"],
                item["total_validation_loss"],
            )
        elif selection_mode == "contraction-first":
            selection_key = lambda item: (
                -item["selection_contraction_audit"]["overall"]["min"],
                -item["selection_contraction_audit"]["overall"]["p05"],
                item["validation_median_terminal_error_mass"],
            )
        else:
            raise ValueError(f"unknown selection mode: {selection_mode}")
        best = min(eligible or seed_results, key=selection_key)
        best_seed = int(best["seed"])
        gain, certificate = models[best_seed]
        defect_audits: dict[str, object] = {}
        contraction_audits: dict[str, object] = {}
        on_policy_validation_loss: dict[str, float] | None = None
        train_policy: JointSampleSet | None = None
        validation_policy: JointSampleSet | None = None
        if run_defect_audit or run_contraction_audit:
            train_policy = _collect_policy_samples(
                torch, gain, device, train_cases, grid, matrix
            )
            validation_policy = _collect_policy_samples(
                torch, gain, device, validation_cases, grid, matrix
            )
            on_policy_validation_loss = _validation_loss(
                torch,
                gain,
                certificate,
                validation_policy,
                grid,
                matrix,
                lambda_ratio=lambda_ratio,
                stable_normalization=stable_normalization,
                stable_weight=stable_weight,
                defect_weight=defect_weight,
                contraction_weight=contraction_weight,
                contraction_margin_ratio=contraction_margin_ratio,
                bi_weight=bi_weight,
                gain_reg_weight=gain_reg_weight,
                lower_lipschitz=lower_lipschitz,
                upper_lipschitz=upper_lipschitz,
                device=device,
            )
        if run_defect_audit:
            assert train_policy is not None and validation_policy is not None
            defect_audit_sets = {
                "fixed_gain_train": train,
                "current_observer_train": train_policy,
                "fixed_gain_validation": validation,
                "current_observer_validation": validation_policy,
            }
            defect_audits = {
                name: _defect_audit(
                    torch,
                    gain,
                    certificate,
                    sample_set,
                    grid,
                    matrix,
                    lambda_ratio=lambda_ratio,
                    lower_lipschitz=lower_lipschitz,
                    gain_scale=gain_scale,
                    device=device,
                    batch_size=batch_size,
                )
                for name, sample_set in defect_audit_sets.items()
            }
        if run_contraction_audit:
            assert train_policy is not None and validation_policy is not None
            noisy = lambda time, q=matrix.shape[0]: noise_waveform(
                "common-sine", 0.01, q, time
            )
            noisy_validation_policy = _collect_policy_samples(
                torch,
                gain,
                device,
                validation_cases[:noise_limit],
                grid,
                matrix,
                noise=noisy,
            )
            contraction_audit_sets = {
                "fixed_gain_train": train,
                "current_observer_train": train_policy,
                "fixed_gain_validation": validation,
                "current_observer_validation": validation_policy,
                "noisy_current_observer_validation": noisy_validation_policy,
            }
            contraction_audits = {
                name: _contraction_audit(
                    torch,
                    gain,
                    certificate,
                    sample_set,
                    grid,
                    matrix,
                    margin_ratio=contraction_margin_ratio,
                    device=device,
                    batch_size=batch_size,
                )
                for name, sample_set in contraction_audit_sets.items()
            }
        if checkpoint_dir is not None:
            torch.save(
                {
                    "grid_size": grid_size,
                    "seed": best_seed,
                    "gain_state_dict": gain.state_dict(),
                    "certificate_state_dict": certificate.state_dict(),
                    "certificate_kind": certificate_kind,
                    "mixing_layers": mixing_layers,
                    "shear_norm_limit": shear_norm_limit,
                    "lower_lipschitz": lower_lipschitz,
                    "upper_lipschitz": upper_lipschitz,
                    "base_gain": base_gain,
                    "gain_scale": gain_scale,
                    "certificate_scale": certificate_scale,
                    "lambda_ratio": lambda_ratio,
                    "gain_learning_rate": gain_learning_rate,
                    "certificate_learning_rate": certificate_learning_rate,
                    "gradient_clip_norm": gradient_clip_norm,
                    "gain_trust_ratio": gain_trust_ratio,
                    "gain_reg_weight": gain_reg_weight,
                    "gain_kind": gain_kind,
                    "contraction_weight": contraction_weight,
                    "contraction_margin_ratio": contraction_margin_ratio,
                },
                checkpoint_dir / f"grid-{grid_size}__seed-{best_seed}.pt",
            )
        replay = [
            _simulate(torch, gain, device, grid, matrix, case)
            for case in test_cases[:eval_limit]
        ]
        noisy = lambda time, q=matrix.shape[0]: noise_waveform(
            "common-sine", 0.01, q, time
        )
        noisy_replay = [
            _simulate(
                torch,
                gain,
                device,
                grid,
                matrix,
                case,
                noise=noisy,
            )
            for case in test_cases[:noise_limit]
        ]
        grid_result = {
            "grid_size": grid_size,
            "lambda_ratio": lambda_ratio,
            "base_gain": base_gain,
            "gain_scale": gain_scale,
            "certificate_scale": certificate_scale,
            "stable_normalization": stable_normalization,
            "stable_weight": stable_weight,
            "defect_weight": defect_weight,
            "contraction_weight": contraction_weight,
            "contraction_margin_ratio": contraction_margin_ratio,
            "bi_weight": bi_weight,
            "gain_reg_weight": gain_reg_weight,
            "gain_kind": gain_kind,
            "lower_lipschitz": lower_lipschitz,
            "upper_lipschitz": upper_lipschitz,
            "certificate_kind": certificate_kind,
            "mixing_layers": mixing_layers,
            "shear_norm_limit": shear_norm_limit,
            "replay_snapshots": replay_snapshots,
            "gain_warmup_epochs": gain_warmup_epochs,
            "certificate_warmup_epochs": certificate_warmup_epochs,
            "gain_learning_rate": gain_learning_rate,
            "certificate_learning_rate": certificate_learning_rate,
            "gradient_clip_norm": gradient_clip_norm,
            "gain_trust_ratio": gain_trust_ratio,
            "refresh_interval": refresh_interval,
            "selection_limit": selection_limit,
            "selection_baseline_gain": selection_baseline_gain,
            "selection_baseline_median_terminal_error_mass": _median(
                baseline_selection, "terminal_error_mass"
            ),
            "selection_constraint_passed": bool(eligible),
            "selected_seed": best_seed,
            "seed_results": seed_results,
            "test_case_count": len(replay),
            "test_median_terminal_error_mass": _median(replay, "terminal_error_mass"),
            "test_median_peak_error_mass": _median(replay, "peak_error_mass"),
            "noisy_case_count": len(noisy_replay),
            "noisy_median_terminal_error_mass": _median(
                noisy_replay, "terminal_error_mass"
            ),
            "certificate_audit": best["certificate_audit"],
            "defect_audits": defect_audits,
            "contraction_audits": contraction_audits,
            "on_policy_validation_loss": on_policy_validation_loss,
        }
        results.append(grid_result)
        print(
            f"[grid={grid_size}] seed={best_seed} "
            f"total={best['total_validation_loss']:.6g} "
            f"defect={best['defect_validation_loss']:.6g} "
            f"bi={best['bi_validation_loss']:.6g} "
            f"validation={best['validation_median_terminal_error_mass']:.6g} "
            f"test={grid_result['test_median_terminal_error_mass']:.6g} "
            f"noisy={grid_result['noisy_median_terminal_error_mass']:.6g}",
            flush=True,
        )
    return {
        "kind": "r5-tk-joint-training",
        "lambda_ratio": lambda_ratio,
        "base_gain": base_gain,
        "gain_scale": gain_scale,
        "certificate_scale": certificate_scale,
        "stable_normalization": stable_normalization,
        "stable_weight": stable_weight,
        "defect_weight": defect_weight,
        "contraction_weight": contraction_weight,
        "contraction_margin_ratio": contraction_margin_ratio,
        "bi_weight": bi_weight,
        "gain_reg_weight": gain_reg_weight,
        "gain_kind": gain_kind,
        "lower_lipschitz": lower_lipschitz,
        "upper_lipschitz": upper_lipschitz,
        "certificate_kind": certificate_kind,
        "mixing_layers": mixing_layers,
        "shear_norm_limit": shear_norm_limit,
        "replay_snapshots": replay_snapshots,
        "gain_warmup_epochs": gain_warmup_epochs,
        "certificate_warmup_epochs": certificate_warmup_epochs,
        "gain_learning_rate": gain_learning_rate,
        "certificate_learning_rate": certificate_learning_rate,
        "gradient_clip_norm": gradient_clip_norm,
        "gain_trust_ratio": gain_trust_ratio,
        "selection_mode": selection_mode,
        "run_defect_audit": run_defect_audit,
        "run_contraction_audit": run_contraction_audit,
        "refresh_interval": refresh_interval,
        "selection_limit": selection_limit,
        "selection_baseline_gain": selection_baseline_gain,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ABLATION_SEEDS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=48)
    parser.add_argument("--noise-limit", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda-ratio", type=float, default=0.5)
    parser.add_argument("--base-gain", type=float, default=0.02)
    parser.add_argument("--gain-scale", type=float, default=0.5)
    parser.add_argument("--certificate-scale", type=float, default=1.0)
    parser.add_argument(
        "--certificate-kind",
        choices=("diagonal", "givens", "triangular"),
        default="diagonal",
    )
    parser.add_argument("--mixing-layers", type=int, default=0)
    parser.add_argument("--shear-norm-limit", type=float, default=0.0)
    parser.add_argument(
        "--stable-normalization",
        choices=("none", "error-time"),
        default="error-time",
    )
    parser.add_argument("--stable-weight", type=float, default=1.0)
    parser.add_argument("--defect-weight", type=float, default=1.0)
    parser.add_argument("--contraction-weight", type=float, default=0.0)
    parser.add_argument("--contraction-margin-ratio", type=float, default=0.0)
    parser.add_argument("--bi-weight", type=float, default=1.0)
    parser.add_argument("--lower-lipschitz", type=float, default=0.5)
    parser.add_argument("--upper-lipschitz", type=float, default=2.0)
    parser.add_argument("--refresh-interval", type=int, default=50)
    parser.add_argument("--replay-snapshots", type=int, default=0)
    parser.add_argument("--gain-warmup-epochs", type=int, default=0)
    parser.add_argument("--certificate-warmup-epochs", type=int, default=0)
    parser.add_argument("--gain-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--certificate-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--gain-trust-ratio", type=float, default=0.0)
    parser.add_argument("--gain-reg-weight", type=float, default=0.0)
    parser.add_argument(
        "--gain-kind",
        choices=("dense", "mass-adjoint", "mass-adjoint-constant"),
        default="dense",
    )
    parser.add_argument("--selection-limit", type=int, default=48)
    parser.add_argument("--selection-baseline-gain", type=float, default=0.10)
    parser.add_argument(
        "--selection-mode",
        choices=("rollout-first", "defect-first", "contraction-first"),
        default="rollout-first",
    )
    parser.add_argument("--run-defect-audit", action="store_true")
    parser.add_argument("--run-contraction-audit", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if args.lambda_ratio <= 0.0:
        raise SystemExit("--lambda-ratio must be positive")
    if (
        args.stable_weight < 0.0
        or args.defect_weight < 0.0
        or args.contraction_weight < 0.0
        or args.bi_weight < 0.0
    ):
        raise SystemExit("loss weights must be nonnegative")
    if args.contraction_margin_ratio < 0.0:
        raise SystemExit("--contraction-margin-ratio must be nonnegative")
    if args.lower_lipschitz <= 0.0 or args.upper_lipschitz < args.lower_lipschitz:
        raise SystemExit("lipschitz bounds must satisfy 0 < lower <= upper")
    if args.refresh_interval < 0:
        raise SystemExit("--refresh-interval must be nonnegative")
    if args.replay_snapshots < 0:
        raise SystemExit("--replay-snapshots must be nonnegative")
    if args.gain_warmup_epochs < 0 or args.certificate_warmup_epochs < 0:
        raise SystemExit("warmup epochs must be nonnegative")
    if args.gain_warmup_epochs + args.certificate_warmup_epochs > args.epochs:
        raise SystemExit("warmup epochs must not exceed total epochs")
    if args.gain_learning_rate <= 0.0 or args.certificate_learning_rate <= 0.0:
        raise SystemExit("learning rates must be positive")
    if args.gradient_clip_norm < 0.0:
        raise SystemExit("--gradient-clip-norm must be nonnegative")
    if args.gain_trust_ratio < 0.0 or args.gain_reg_weight < 0.0:
        raise SystemExit("gain trust ratio and regularization must be nonnegative")
    if args.gain_kind.startswith("mass-adjoint") and not (
        0.0 < args.gain_trust_ratio < 1.0
    ):
        raise SystemExit("mass-adjoint gain requires 0 < trust ratio < 1")
    if args.certificate_kind in {"givens", "triangular"}:
        if args.mixing_layers < 1:
            raise SystemExit("mixed certificate requires --mixing-layers >= 1")
        if not args.lower_lipschitz < 1.0 < args.upper_lipschitz:
            raise SystemExit("mixed certificate bounds must strictly contain 1")
    if args.certificate_kind == "triangular":
        if not 0.0 < args.shear_norm_limit < 1.0:
            raise SystemExit("triangular certificate requires 0 < shear limit < 1")
        scale_lower = args.lower_lipschitz * (1.0 + args.shear_norm_limit)
        scale_upper = args.upper_lipschitz / (1.0 + args.shear_norm_limit)
        if not scale_lower < 1.0 < scale_upper:
            raise SystemExit("triangular internal scale bounds must contain 1")
    if args.selection_limit < 1:
        raise SystemExit("--selection-limit must be positive")
    if args.selection_baseline_gain < 0.0:
        raise SystemExit("--selection-baseline-gain must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        torch,
        args.grid_sizes,
        args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_limit=args.eval_limit,
        noise_limit=args.noise_limit,
        device=args.device,
        lambda_ratio=args.lambda_ratio,
        base_gain=args.base_gain,
        gain_scale=args.gain_scale,
        certificate_scale=args.certificate_scale,
        stable_normalization=args.stable_normalization,
        stable_weight=args.stable_weight,
        defect_weight=args.defect_weight,
        contraction_weight=args.contraction_weight,
        contraction_margin_ratio=args.contraction_margin_ratio,
        bi_weight=args.bi_weight,
        lower_lipschitz=args.lower_lipschitz,
        upper_lipschitz=args.upper_lipschitz,
        certificate_kind=args.certificate_kind,
        mixing_layers=args.mixing_layers,
        shear_norm_limit=args.shear_norm_limit,
        replay_snapshots=args.replay_snapshots,
        gain_warmup_epochs=args.gain_warmup_epochs,
        certificate_warmup_epochs=args.certificate_warmup_epochs,
        gain_learning_rate=args.gain_learning_rate,
        certificate_learning_rate=args.certificate_learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        gain_trust_ratio=args.gain_trust_ratio,
        gain_reg_weight=args.gain_reg_weight,
        gain_kind=args.gain_kind,
        refresh_interval=args.refresh_interval,
        selection_limit=args.selection_limit,
        selection_baseline_gain=args.selection_baseline_gain,
        selection_mode=args.selection_mode,
        run_defect_audit=args.run_defect_audit,
        run_contraction_audit=args.run_contraction_audit,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_count": len(result["results"]), "device": args.device}))


if __name__ == "__main__":
    main()
