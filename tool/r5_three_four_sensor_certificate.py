"""Select and validate three- and four-sensor R5 observer configurations."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalOutputInjection,
    generate_pilot_cases,
    local_average_matrix,
    mass_adjoint_injection,
    noise_waveform,
    normalized_modal_transform,
    simulate_causal_nudging,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)
from allen_cahn_certified_observer.solver import allen_cahn_rhs
from r5_oblique_injection_certificate import (
    FIVE_SENSOR_INTERVALS,
    GRID_SIZES,
    NU_VALUES,
    TWO_SENSOR_INTERVALS,
    _design_two_sensor_injection,
    _rollout_records,
    _summarize_rollouts,
)


TOTAL_OBSERVATION_LENGTH = 0.20
FOUR_SENSOR_GAINS = (0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)
FOUR_SENSOR_TARGET_MARGIN = 0.10
OUTPUT_TIMES = np.linspace(0.0, 1.0, 101)

THREE_SENSOR_CENTERS = {
    "cell-centered": np.asarray([1.0 / 6.0, 0.5, 5.0 / 6.0]),
    "quarter-centered": np.asarray([0.25, 0.50, 0.75]),
    "wide-interior": np.asarray([0.20, 0.50, 0.80]),
    "asymmetric-control": np.asarray([0.20, 0.45, 0.80]),
}
FOUR_SENSOR_CENTERS = {
    "cell-centered": np.asarray([0.125, 0.375, 0.625, 0.875]),
    "interior-fifths": np.asarray([0.20, 0.40, 0.60, 0.80]),
}


def intervals_from_centers(centers: np.ndarray) -> np.ndarray:
    """Return equal-width, non-overlapping intervals of total length 0.20."""

    values = np.sort(np.asarray(centers, dtype=float))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("centers must be a finite non-empty vector")
    width = TOTAL_OBSERVATION_LENGTH / values.size
    intervals = np.column_stack((values - 0.5 * width, values + 0.5 * width))
    if (
        intervals[0, 0] < 0.0
        or intervals[-1, 1] > 1.0
        or np.any(intervals[1:, 0] < intervals[:-1, 1] - 1e-12)
    ):
        raise ValueError("sensor intervals must be inside [0, 1] and non-overlapping")
    return intervals


def select_three_sensor_geometry() -> tuple[str, np.ndarray, list[dict[str, object]]]:
    """Select the frozen three-sensor candidate using n=31 linear metrics only."""

    grid = AllenCahnGrid(31)
    records: list[dict[str, object]] = []
    eligible: list[tuple[tuple[float, float, float, str], str, np.ndarray]] = []
    for name, centers in THREE_SENSOR_CENTERS.items():
        intervals = intervals_from_centers(centers)
        observation = local_average_matrix(grid, intervals)
        designs: list[dict[str, object]] = []
        all_feasible = True
        hard_metrics: dict[str, float] | None = None
        for nu in NU_VALUES:
            modal = unstable_modal_system(grid, nu, observation)
            item: dict[str, object] = {
                "nu": nu,
                "unstable_dimension": modal.dimension,
                "observability_rank": modal.observability_rank,
                "observability_condition": modal.observability_condition,
            }
            if modal.observability_rank != modal.dimension:
                all_feasible = False
                item["status"] = "unobservable"
                designs.append(item)
                continue
            design, design_record = _design_two_sensor_injection(
                grid, nu, observation
            )
            diagnostics = design_record["selected_diagnostics"]
            item.update(
                {
                    "status": "feasible",
                    "selected_method": design_record["selected_method"],
                    "closed_loop_spectral_abscissa": diagnostics[
                        "closed_loop_spectral_abscissa"
                    ],
                    "modal_contraction_rate": diagnostics[
                        "modal_contraction_rate"
                    ],
                    "transform_condition": diagnostics["transform_condition"],
                    "mass_scaled_gain_norm": diagnostics["mass_scaled_gain_norm"],
                    "sampled_transient_amplification": diagnostics[
                        "sampled_transient_amplification"
                    ],
                }
            )
            if nu == 0.005:
                hard_metrics = diagnostics
            designs.append(item)
        record = {
            "name": name,
            "centers": centers.tolist(),
            "intervals": intervals.tolist(),
            "total_observation_length": float(
                np.sum(intervals[:, 1] - intervals[:, 0])
            ),
            "all_linear_feasible": all_feasible,
            "designs": designs,
        }
        records.append(record)
        if all_feasible and hard_metrics is not None:
            score = (
                float(hard_metrics["sampled_transient_amplification"]),
                float(hard_metrics["transform_condition"]),
                float(hard_metrics["mass_scaled_gain_norm"]),
                name,
            )
            eligible.append((score, name, intervals))
    if not eligible:
        raise RuntimeError("no three-sensor candidate passed the linear gate")
    _, selected_name, selected_intervals = min(eligible, key=lambda item: item[0])
    return selected_name, selected_intervals, records


def select_four_sensor_geometry(
) -> tuple[str, float, np.ndarray, list[dict[str, object]]]:
    """Select the smallest n=31 gain meeting the frozen global-margin target."""

    grid = AllenCahnGrid(31)
    records: list[dict[str, object]] = []
    eligible: list[
        tuple[tuple[float, float, str], str, float, np.ndarray]
    ] = []
    for name, centers in FOUR_SENSOR_CENTERS.items():
        intervals = intervals_from_centers(centers)
        observation = local_average_matrix(grid, intervals)
        gain_records: list[dict[str, object]] = []
        for gain in FOUR_SENSOR_GAINS:
            margins = {
                f"{nu:.3f}": symmetric_allen_cahn_margin(
                    grid, nu, observation, gain=gain
                )
                for nu in NU_VALUES
            }
            worst = min(margins.values())
            passed = bool(worst >= FOUR_SENSOR_TARGET_MARGIN)
            gain_records.append(
                {
                    "gain": gain,
                    "margins": margins,
                    "worst_margin": worst,
                    "passed": passed,
                }
            )
            if passed:
                eligible.append(
                    ((gain, -worst, name), name, gain, intervals)
                )
                break
        records.append(
            {
                "name": name,
                "centers": centers.tolist(),
                "intervals": intervals.tolist(),
                "total_observation_length": float(
                    np.sum(intervals[:, 1] - intervals[:, 0])
                ),
                "gain_scan": gain_records,
            }
        )
    if not eligible:
        raise RuntimeError("no four-sensor candidate passed the global-margin gate")
    _, selected_name, selected_gain, selected_intervals = min(
        eligible, key=lambda item: item[0]
    )
    return selected_name, selected_gain, selected_intervals, records


def _split_initial_conditions(
    split: str,
    grid: AllenCahnGrid,
    nu: float,
    observation: np.ndarray,
    limit: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    selected = [
        case
        for case in generate_pilot_cases()
        if case.split == split and case.n == grid.n and case.nu == nu
    ][:limit]
    if not selected:
        raise RuntimeError(f"no {split} cases matched n={grid.n}, nu={nu}")
    cases = [
        (case.case_id, case.initial_truth(grid), case.initial_estimate(grid))
        for case in selected
    ]
    truth = selected[0].initial_truth(grid)
    target_mass_norm = 0.25 / np.sqrt(2.0)
    fourth_mode = np.sin(4.0 * np.pi * grid.x)
    fourth_mode *= target_mass_norm / np.sqrt(
        grid.h * np.dot(fourth_mode, fourth_mode)
    )
    cases.append(
        (
            f"{split}-stress-fourth-mode__n-{grid.n}__nu-{nu:.3f}",
            truth,
            truth + fourth_mode,
        )
    )
    modal = unstable_modal_system(grid, nu, observation)
    _, _, right = np.linalg.svd(modal.observed_modes, full_matrices=True)
    hard = modal.modes @ right[-1]
    hard *= target_mass_norm / np.sqrt(grid.h * np.dot(hard, hard))
    cases.append(
        (
            f"{split}-stress-min-observation__n-{grid.n}__nu-{nu:.3f}",
            truth,
            truth + hard,
        )
    )
    return cases


def _signed_rate_summary(rates: np.ndarray, requested_rate: float) -> dict[str, object]:
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rates must be a finite non-empty vector")
    return {
        "sample_count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "requested_rate": requested_rate,
        "requested_margin_min": float(np.min(values - requested_rate)),
        "requested_rate_fraction": float(np.mean(values >= requested_rate)),
    }


def _three_sensor_contraction_summary(
    grid: AllenCahnGrid,
    nu: float,
    observation: np.ndarray,
    design: object,
    cases: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, object]:
    modal = unstable_modal_system(grid, nu, observation)
    transform = normalized_modal_transform(grid, modal, design.modal_metric)
    observer = CausalOutputInjection(
        grid, nu, observation, design.injection_matrix
    )
    all_rates: list[float] = []
    records: list[dict[str, object]] = []
    for case_id, truth_initial, estimate_initial in cases:
        rollout = simulate_causal_nudging(
            observer,
            truth_initial,
            estimate_initial,
            output_times=OUTPUT_TIMES,
        )
        errors = rollout.estimate - rollout.truth
        error_rhs = np.asarray(
            [
                allen_cahn_rhs(grid, nu, estimate)
                - allen_cahn_rhs(grid, nu, truth)
                - design.injection_matrix @ (observation @ (estimate - truth))
                for truth, estimate in zip(rollout.truth, rollout.estimate)
            ]
        )
        transformed = errors @ transform.T
        transformed_rhs = error_rhs @ transform.T
        rates = -np.sum(transformed * transformed_rhs, axis=1) / (
            np.sum(transformed**2, axis=1) + 1e-12
        )
        all_rates.extend(rates.tolist())
        records.append(
            {
                "case_id": case_id,
                "min": float(np.min(rates)),
                "p05": float(np.quantile(rates, 0.05)),
                "median": float(np.median(rates)),
                "positive_fraction": float(np.mean(rates > 0.0)),
            }
        )
    requested_rate = 0.1 * nu * np.pi**2
    summary = _signed_rate_summary(np.asarray(all_rates), requested_rate)
    summary["records"] = records
    return summary


def _linear_records(
    three_intervals: np.ndarray,
    four_intervals: np.ndarray,
    four_gain: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, float], object]]:
    three_records: list[dict[str, object]] = []
    four_records: list[dict[str, object]] = []
    designs: dict[tuple[int, float], object] = {}
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        three = local_average_matrix(grid, three_intervals)
        four = local_average_matrix(grid, four_intervals)
        for nu in NU_VALUES:
            modal = unstable_modal_system(grid, nu, three)
            design, record = _design_two_sensor_injection(grid, nu, three)
            designs[(n, nu)] = design
            diagnostics = record["selected_diagnostics"]
            three_records.append(
                {
                    "n": n,
                    "nu": nu,
                    "unstable_dimension": modal.dimension,
                    "mass_adjoint_rank": int(
                        np.linalg.matrix_rank(mass_adjoint_injection(grid, three))
                    ),
                    "mass_adjoint_margin_gain_1": symmetric_allen_cahn_margin(
                        grid, nu, three, gain=1.0
                    ),
                    "observability_rank": modal.observability_rank,
                    "observability_condition": modal.observability_condition,
                    "selected_method": record["selected_method"],
                    "closed_loop_spectral_abscissa": diagnostics[
                        "closed_loop_spectral_abscissa"
                    ],
                    "modal_contraction_rate": diagnostics[
                        "modal_contraction_rate"
                    ],
                    "transform_condition": diagnostics["transform_condition"],
                    "mass_scaled_gain_norm": diagnostics["mass_scaled_gain_norm"],
                    "sampled_transient_amplification": diagnostics[
                        "sampled_transient_amplification"
                    ],
                }
            )
            margin = symmetric_allen_cahn_margin(
                grid, nu, four, gain=four_gain
            )
            four_records.append(
                {
                    "n": n,
                    "nu": nu,
                    "gain": four_gain,
                    "global_semidiscrete_margin": margin,
                    "passed_zero": bool(margin > 0.0),
                    "passed_target": bool(margin >= FOUR_SENSOR_TARGET_MARGIN),
                }
            )
    return three_records, four_records, designs


def _evaluate_split(
    split: str,
    limit: int,
    three_intervals: np.ndarray,
    four_intervals: np.ndarray,
    four_gain: float,
    designs: dict[tuple[int, float], object],
) -> list[dict[str, object]]:
    combinations: list[dict[str, object]] = []
    for n in GRID_SIZES:
        grid = AllenCahnGrid(n)
        two = local_average_matrix(grid, TWO_SENSOR_INTERVALS)
        three = local_average_matrix(grid, three_intervals)
        four = local_average_matrix(grid, four_intervals)
        five = local_average_matrix(grid, FIVE_SENSOR_INTERVALS)
        for nu in NU_VALUES:
            cases = _split_initial_conditions(split, grid, nu, three, limit)
            design = designs[(n, nu)]
            methods = {
                "two-sensor-fixed-0.1": (
                    two,
                    mass_adjoint_injection(grid, two, gain=0.1),
                ),
                "three-sensor-fixed-0.1": (
                    three,
                    mass_adjoint_injection(grid, three, gain=0.1),
                ),
                "three-sensor-lmi": (three, design.injection_matrix),
                "four-sensor-certified": (
                    four,
                    mass_adjoint_injection(grid, four, gain=four_gain),
                ),
                "five-sensor-certified": (
                    five,
                    mass_adjoint_injection(grid, five, gain=1.0),
                ),
            }
            result: dict[str, object] = {
                "split": split,
                "n": n,
                "nu": nu,
                "methods": {},
                "three_sensor_contraction": _three_sensor_contraction_summary(
                    grid, nu, three, design, cases
                ),
            }
            for name, (observation, injection) in methods.items():
                result["methods"][name] = _summarize_rollouts(
                    _rollout_records(
                        grid, nu, observation, injection, cases
                    )
                )
            for name in (
                "three-sensor-lmi",
                "four-sensor-certified",
                "five-sensor-certified",
            ):
                observation, injection = methods[name]
                noise = lambda time, q=observation.shape[0]: noise_waveform(
                    "common-sine", 0.01, q, time
                )
                result["methods"][f"{name}-noise-0.01"] = _summarize_rollouts(
                    _rollout_records(
                        grid,
                        nu,
                        observation,
                        injection,
                        cases,
                        noise=noise,
                    )
                )
            combinations.append(result)
    return combinations


def _online_no_regression(
    combinations: list[dict[str, object]], method: str
) -> bool:
    return all(
        combination["methods"][method]["terminal_error_mass_median"]
        <= combination["methods"]["two-sensor-fixed-0.1"][
            "terminal_error_mass_median"
        ]
        for combination in combinations
    )


def _noise_robust(
    combinations: list[dict[str, object]], method: str
) -> bool:
    return all(
        combination["methods"][f"{method}-noise-0.01"][
            "terminal_error_mass_median"
        ]
        <= 1.10
        * combination["methods"][method]["terminal_error_mass_median"]
        for combination in combinations
    )


def run(validation_limit: int, test_limit: int) -> dict[str, object]:
    three_name, three_intervals, three_candidates = select_three_sensor_geometry()
    four_name, four_gain, four_intervals, four_candidates = (
        select_four_sensor_geometry()
    )
    three_linear, four_linear, designs = _linear_records(
        three_intervals, four_intervals, four_gain
    )
    validation = _evaluate_split(
        "validation",
        validation_limit,
        three_intervals,
        four_intervals,
        four_gain,
        designs,
    )

    three_rank_obstruction = all(
        record["mass_adjoint_rank"] < record["unstable_dimension"]
        for record in three_linear
        if record["nu"] == 0.005
    )
    three_linear_passed = all(
        record["observability_rank"] == record["unstable_dimension"]
        and record["closed_loop_spectral_abscissa"] < 0.0
        and record["modal_contraction_rate"] > 0.0
        and record["transform_condition"] <= 4.0
        and record["sampled_transient_amplification"] <= 2.5
        for record in three_linear
    )
    three_contraction_passed = all(
        combination["three_sensor_contraction"]["requested_margin_min"] > 0.0
        for combination in validation
    )
    three_online_passed = _online_no_regression(validation, "three-sensor-lmi")
    three_noise_passed = _noise_robust(validation, "three-sensor-lmi")
    three_validation_passed = all(
        (
            three_rank_obstruction,
            three_linear_passed,
            three_contraction_passed,
            three_online_passed,
            three_noise_passed,
        )
    )

    four_global_passed = all(record["passed_target"] for record in four_linear)
    four_online_passed = _online_no_regression(
        validation, "four-sensor-certified"
    )
    four_noise_passed = _noise_robust(validation, "four-sensor-certified")
    four_validation_passed = all(
        (four_global_passed, four_online_passed, four_noise_passed)
    )

    test_eligible = bool(three_validation_passed and four_validation_passed)
    test = (
        _evaluate_split(
            "test",
            test_limit,
            three_intervals,
            four_intervals,
            four_gain,
            designs,
        )
        if test_eligible
        else None
    )
    test_confirmation = None
    if test is not None:
        test_confirmation = {
            "three_sensor_requested_contraction": all(
                combination["three_sensor_contraction"][
                    "requested_margin_min"
                ]
                > 0.0
                for combination in test
            ),
            "three_sensor_online_no_regression": _online_no_regression(
                test, "three-sensor-lmi"
            ),
            "three_sensor_noise_robust": _noise_robust(
                test, "three-sensor-lmi"
            ),
            "four_sensor_online_no_regression": _online_no_regression(
                test, "four-sensor-certified"
            ),
            "four_sensor_noise_robust": _noise_robust(
                test, "four-sensor-certified"
            ),
        }

    return {
        "kind": "r5-three-four-sensor-certificate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "declared": {
            "grid_sizes": GRID_SIZES,
            "nu_values": NU_VALUES,
            "total_observation_length": TOTAL_OBSERVATION_LENGTH,
            "four_sensor_gain_candidates": FOUR_SENSOR_GAINS,
            "four_sensor_target_margin": FOUR_SENSOR_TARGET_MARGIN,
            "validation_limit_per_combination": validation_limit,
            "test_limit_per_combination": test_limit,
            "noise_kind": "common-sine",
            "noise_amplitude": 0.01,
            "requested_rate": "lambda = 0.1 * nu * pi**2",
        },
        "selection": {
            "three_sensor_candidates": three_candidates,
            "selected_three_sensor": {
                "name": three_name,
                "intervals": three_intervals.tolist(),
            },
            "four_sensor_candidates": four_candidates,
            "selected_four_sensor": {
                "name": four_name,
                "gain": four_gain,
                "intervals": four_intervals.tolist(),
            },
            "selection_data": "n=31 linear matrix diagnostics only",
        },
        "three_sensor_linear": three_linear,
        "four_sensor_certificates": four_linear,
        "validation": validation,
        "test": test,
        "gates": {
            "three_sensor_mass_adjoint_rank_obstruction_verified": (
                three_rank_obstruction
            ),
            "three_sensor_general_injection_linear_gate": three_linear_passed,
            "three_sensor_validation_requested_contraction": (
                three_contraction_passed
            ),
            "three_sensor_validation_online_no_regression": three_online_passed,
            "three_sensor_validation_noise_robust": three_noise_passed,
            "three_sensor_validation_passed": three_validation_passed,
            "four_sensor_global_semidiscrete_target": four_global_passed,
            "four_sensor_validation_online_no_regression": four_online_passed,
            "four_sensor_validation_noise_robust": four_noise_passed,
            "four_sensor_validation_passed": four_validation_passed,
            "test_evaluated": test is not None,
            "test_confirmation": test_confirmation,
            "remote_joint_training_required": bool(
                three_linear_passed
                and three_online_passed
                and not three_contraction_passed
            ),
        },
        "scope": (
            "The four-sensor eigenvalue bound is a global semidiscrete Allen-Cahn "
            "error-energy certificate. The three-sensor transformed contraction "
            "claim is finite-sample and limited to the declared trajectories."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-limit-per-combination", type=int, default=4)
    parser.add_argument("--test-limit-per-combination", type=int, default=4)
    args = parser.parse_args()
    if args.validation_limit_per_combination < 1:
        raise SystemExit("--validation-limit-per-combination must be positive")
    if args.test_limit_per_combination < 1:
        raise SystemExit("--test-limit-per-combination must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        args.validation_limit_per_combination,
        args.test_limit_per_combination,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"]), flush=True)


if __name__ == "__main__":
    main()
