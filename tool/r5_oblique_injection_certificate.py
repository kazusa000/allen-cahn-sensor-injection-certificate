"""CPU certificate and nonlinear gate for the R5 output-injection repair."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    finite_horizon_transient_amplification,
    generate_pilot_cases,
    linearized_error_matrix,
    lmi_modal_injection,
    local_average_matrix,
    mass_adjoint_injection,
    noise_waveform,
    pole_placement_modal_injection,
    riccati_modal_injection,
    simulate_causal_nudging,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)


TWO_SENSOR_INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]])
FIVE_SENSOR_INTERVALS = np.array(
    [
        [0.08, 0.12],
        [0.28, 0.32],
        [0.48, 0.52],
        [0.68, 0.72],
        [0.88, 0.92],
    ]
)
GRID_SIZES = (31, 63, 127)
NU_VALUES = (0.005, 0.010, 0.020)
RICCATI_WEIGHTS = (0.1, 1.0, 10.0)
LMI_CONDITION_BOUNDS = (16.0, 32.0, 64.0, 128.0, 256.0)


def _design_record(
    grid: AllenCahnGrid,
    nu: float,
    observation: np.ndarray,
    design: object,
) -> dict[str, object]:
    closed_loop = linearized_error_matrix(
        grid, nu, observation, design.injection_matrix
    )
    transient, transient_time = finite_horizon_transient_amplification(
        closed_loop, horizon=1.0, sample_count=41
    )
    return {
        "method": design.method,
        "solver_status": design.solver_status,
        "closed_loop_spectral_abscissa": design.closed_loop_spectral_abscissa,
        "mass_scaled_gain_norm": design.mass_scaled_gain_norm,
        "modal_contraction_rate": design.modal_contraction_rate,
        "modal_metric_condition": design.modal_metric_condition,
        "transform_condition": design.transform_condition,
        "sampled_transient_amplification": transient,
        "sampled_transient_maximizing_time": transient_time,
    }


def _design_two_sensor_injection(
    grid: AllenCahnGrid, nu: float, observation: np.ndarray
) -> tuple[object, dict[str, object]]:
    modal = unstable_modal_system(grid, nu, observation)
    candidates: list[dict[str, object]] = []
    pole = pole_placement_modal_injection(grid, nu, observation)
    candidates.append(_design_record(grid, nu, observation, pole))
    for weight in RICCATI_WEIGHTS:
        riccati = riccati_modal_injection(
            grid, nu, observation, measurement_weight=weight
        )
        candidates.append(_design_record(grid, nu, observation, riccati))

    target_rate = 0.1 * nu * np.pi**2
    selected = None
    lmi_attempts: list[dict[str, object]] = []
    for condition_bound in LMI_CONDITION_BOUNDS:
        try:
            design = lmi_modal_injection(
                grid,
                nu,
                observation,
                decay_rate=target_rate,
                metric_condition_bound=condition_bound,
            )
        except RuntimeError as error:
            lmi_attempts.append(
                {
                    "condition_bound": condition_bound,
                    "status": "infeasible",
                    "detail": str(error),
                }
            )
            continue
        record = _design_record(grid, nu, observation, design)
        record["condition_bound"] = condition_bound
        lmi_attempts.append(
            {
                "condition_bound": condition_bound,
                "status": design.solver_status,
                "modal_metric_condition": design.modal_metric_condition,
                "modal_contraction_rate": design.modal_contraction_rate,
            }
        )
        selected = design
        candidates.append(record)
        break
    if selected is None:
        raise RuntimeError(f"no declared LMI condition bound was feasible for n={grid.n}, nu={nu}")

    result = {
        "n": grid.n,
        "nu": nu,
        "unstable_dimension": modal.dimension,
        "unstable_eigenvalues": modal.eigenvalues.tolist(),
        "observability_rank": modal.observability_rank,
        "mass_normalized_observability_min_singular_value": (
            modal.observability_min_singular_value / np.sqrt(grid.h)
        ),
        "observability_condition": modal.observability_condition,
        "target_contraction_rate": target_rate,
        "lmi_attempts": lmi_attempts,
        "candidate_diagnostics": candidates,
        "selected_method": selected.method,
        "selected_diagnostics": _design_record(
            grid, nu, observation, selected
        ),
    }
    return selected, result


def _validation_initial_conditions(
    grid: AllenCahnGrid,
    nu: float,
    observation: np.ndarray,
    limit: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    selected_cases = [
        case
        for case in generate_pilot_cases()
        if case.split == "validation" and case.n == grid.n and case.nu == nu
    ][:limit]
    cases = [
        (
            case.case_id,
            case.initial_truth(grid),
            case.initial_estimate(grid),
        )
        for case in selected_cases
    ]
    if not selected_cases:
        raise RuntimeError("no validation cases matched the declared combination")

    truth = selected_cases[0].initial_truth(grid)
    target_mass_norm = 0.25 / np.sqrt(2.0)
    fourth_mode = np.sin(4.0 * np.pi * grid.x)
    fourth_mode *= target_mass_norm / np.sqrt(grid.h * np.dot(fourth_mode, fourth_mode))
    cases.append((f"stress-fourth-mode__n-{grid.n}__nu-{nu:.3f}", truth, truth + fourth_mode))

    modal = unstable_modal_system(grid, nu, observation)
    _, _, right = np.linalg.svd(modal.observed_modes, full_matrices=True)
    hard_direction = modal.modes @ right[-1]
    hard_direction *= target_mass_norm / np.sqrt(
        grid.h * np.dot(hard_direction, hard_direction)
    )
    cases.append((f"stress-min-observation__n-{grid.n}__nu-{nu:.3f}", truth, truth + hard_direction))
    return cases


def _rollout_records(
    grid: AllenCahnGrid,
    nu: float,
    observation: np.ndarray,
    injection: np.ndarray,
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    noise: Callable[[float], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    observer = CausalOutputInjection(grid, nu, observation, injection)
    records: list[dict[str, object]] = []
    times = np.linspace(0.0, 1.0, 101)
    for case_id, truth, estimate in cases:
        rollout = simulate_causal_nudging(
            observer,
            truth,
            estimate,
            output_times=times,
            noise=noise,
        )
        initial = float(rollout.error_mass_norm[0])
        terminal = float(rollout.error_mass_norm[-1])
        records.append(
            {
                "case_id": case_id,
                "solver_status": rollout.solver_status,
                "initial_error_mass": initial,
                "peak_error_mass": float(np.max(rollout.error_mass_norm)),
                "terminal_error_mass": terminal,
                "terminal_to_initial_ratio": terminal / initial,
            }
        )
    return records


def _summarize_rollouts(records: list[dict[str, object]]) -> dict[str, object]:
    terminal = np.asarray([record["terminal_error_mass"] for record in records])
    peak = np.asarray([record["peak_error_mass"] for record in records])
    ratio = np.asarray([record["terminal_to_initial_ratio"] for record in records])
    return {
        "case_count": len(records),
        "all_solver_status_zero": bool(
            all(record["solver_status"] == 0 for record in records)
        ),
        "terminal_error_mass_median": float(np.median(terminal)),
        "terminal_error_mass_max": float(np.max(terminal)),
        "peak_error_mass_median": float(np.median(peak)),
        "peak_error_mass_max": float(np.max(peak)),
        "terminal_to_initial_ratio_median": float(np.median(ratio)),
        "terminal_to_initial_ratio_max": float(np.max(ratio)),
        "records": records,
    }


def run(validation_limit: int) -> dict[str, object]:
    certificate_records: list[dict[str, object]] = []
    design_records: list[dict[str, object]] = []
    nonlinear_records: list[dict[str, object]] = []
    selected_designs: dict[tuple[int, float], object] = {}

    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        two_sensor = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
        five_sensor = local_average_matrix(grid, FIVE_SENSOR_INTERVALS)
        for nu in NU_VALUES:
            two_modal = unstable_modal_system(grid, nu, two_sensor)
            two_adjoint_margin = symmetric_allen_cahn_margin(
                grid, nu, two_sensor, gain=1.0
            )
            five_margin = symmetric_allen_cahn_margin(
                grid, nu, five_sensor, gain=1.0
            )
            certificate_records.append(
                {
                    "n": n,
                    "nu": nu,
                    "unstable_dimension": two_modal.dimension,
                    "two_sensor_mass_adjoint_margin": two_adjoint_margin,
                    "five_sensor_mass_adjoint_margin": five_margin,
                    "five_sensor_passed": bool(five_margin > 0.0),
                }
            )
            selected, design_result = _design_two_sensor_injection(
                grid, nu, two_sensor
            )
            selected_designs[(n, nu)] = selected
            design_records.append(design_result)

    method_summaries: dict[str, list[dict[str, object]]] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        two_sensor = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
        five_sensor = local_average_matrix(grid, FIVE_SENSOR_INTERVALS)
        for nu in NU_VALUES:
            cases = _validation_initial_conditions(
                grid, nu, two_sensor, validation_limit
            )
            methods = {
                "two-sensor-open-loop": (two_sensor, np.zeros((n, 2))),
                "two-sensor-fixed-0.1": (
                    two_sensor,
                    mass_adjoint_injection(grid, two_sensor, gain=0.1),
                ),
                "five-sensor-certified": (
                    five_sensor,
                    mass_adjoint_injection(grid, five_sensor, gain=1.0),
                ),
                "two-sensor-lmi": (
                    two_sensor,
                    selected_designs[(n, nu)].injection_matrix,
                ),
            }
            combination: dict[str, object] = {"n": n, "nu": nu, "methods": {}}
            for name, (observation, injection) in methods.items():
                records = _rollout_records(
                    grid, nu, observation, injection, cases
                )
                summary = _summarize_rollouts(records)
                combination["methods"][name] = summary
                method_summaries.setdefault(name, []).append(summary)

            for name in ("five-sensor-certified", "two-sensor-lmi"):
                observation, injection = methods[name]
                noise = lambda time, q=observation.shape[0]: noise_waveform(
                    "common-sine", 0.01, q, time
                )
                noisy_records = _rollout_records(
                    grid,
                    nu,
                    observation,
                    injection,
                    cases,
                    noise=noise,
                )
                combination["methods"][f"{name}-noise-0.01"] = _summarize_rollouts(
                    noisy_records
                )
            nonlinear_records.append(combination)

    five_sensor_passed = all(
        record["five_sensor_passed"] for record in certificate_records
    )
    two_sensor_linear_passed = all(
        record["observability_rank"] == record["unstable_dimension"]
        and record["selected_diagnostics"]["closed_loop_spectral_abscissa"] < 0.0
        and record["selected_diagnostics"]["modal_contraction_rate"] > 0.0
        for record in design_records
    )
    two_sensor_online_passed = all(
        combination["methods"]["two-sensor-lmi"]["terminal_error_mass_median"]
        <= combination["methods"]["two-sensor-fixed-0.1"][
            "terminal_error_mass_median"
        ]
        for combination in nonlinear_records
    )
    return {
        "kind": "r5-oblique-injection-certificate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "declared": {
            "grid_sizes": GRID_SIZES,
            "nu_values": NU_VALUES,
            "two_sensor_intervals": TWO_SENSOR_INTERVALS.tolist(),
            "five_sensor_intervals": FIVE_SENSOR_INTERVALS.tolist(),
            "five_sensor_total_length": float(
                np.sum(FIVE_SENSOR_INTERVALS[:, 1] - FIVE_SENSOR_INTERVALS[:, 0])
            ),
            "validation_limit_per_combination": validation_limit,
            "riccati_weights": RICCATI_WEIGHTS,
            "lmi_condition_bounds": LMI_CONDITION_BOUNDS,
            "lmi_target": "lambda = 0.1 * nu * pi**2",
        },
        "certificate_records": certificate_records,
        "two_sensor_design_records": design_records,
        "nonlinear_validation": nonlinear_records,
        "gates": {
            "five_sensor_global_semidiscrete_certificate": five_sensor_passed,
            "two_sensor_linear_feasibility": two_sensor_linear_passed,
            "two_sensor_online_no_regression": two_sensor_online_passed,
            "remote_joint_training_required": bool(
                two_sensor_linear_passed and not two_sensor_online_passed
            ),
        },
        "scope": (
            "The five-sensor eigenvalue bound is a global semidiscrete Allen-Cahn "
            "error-energy certificate. Two-sensor LMI and nonlinear rollout claims "
            "remain finite-dimensional and within the declared trajectory set."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-limit-per-combination", type=int, default=4)
    args = parser.parse_args()
    if args.validation_limit_per_combination < 1:
        raise SystemExit("--validation-limit-per-combination must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.validation_limit_per_combination)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"]), flush=True)


if __name__ == "__main__":
    main()
