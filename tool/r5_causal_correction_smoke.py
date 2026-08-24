"""Run the local R5-D causal-correction pilot on a held-out split.

This is a CPU smoke experiment, not the formal multi-seed R5-E runner. The
offline fit uses truth-derived target corrections; rollout of the fitted model
uses only the current estimate and current observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    allen_cahn_energy,
    allen_cahn_rhs,
    fit_state_conditioned_linear_correction,
    generate_pilot_cases,
    local_average_matrix,
    noise_waveform,
    simulate_causal_nudging,
    simulate_learned_correction,
)

INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]], dtype=float)
OUTPUT_TIMES = np.linspace(0.0, 1.0, 51)


def _rollout_record(
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


def _rhs_trajectory(grid: AllenCahnGrid, nu: float, states: np.ndarray) -> np.ndarray:
    return np.asarray([allen_cahn_rhs(grid, nu, state) for state in states])


def run(grid_size: int, train_limit: int, validation_limit: int) -> dict[str, object]:
    all_cases = generate_pilot_cases()
    train_cases = [
        case for case in all_cases if case.split == "train" and case.n == grid_size
    ][:train_limit]
    validation_cases = [
        case for case in all_cases if case.split == "validation" and case.n == grid_size
    ][:validation_limit]
    if not train_cases or not validation_cases:
        raise ValueError("train and validation selections must both be non-empty")

    grid = AllenCahnGrid(grid_size)
    matrix = local_average_matrix(grid, INTERVALS)
    training_estimates: list[np.ndarray] = []
    training_measurements: list[np.ndarray] = []
    training_targets: list[np.ndarray] = []
    for case in train_cases:
        truth = case.initial_truth(grid)
        estimate = case.initial_estimate(grid)
        open_loop = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.0),
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        training_estimates.extend(open_loop.estimate)
        training_measurements.extend(open_loop.measurements)
        training_targets.extend(
            _rhs_trajectory(grid, case.nu, open_loop.truth)
            - _rhs_trajectory(grid, case.nu, open_loop.estimate)
        )

    model = fit_state_conditioned_linear_correction(
        grid,
        matrix,
        np.asarray(training_estimates),
        np.asarray(training_measurements),
        np.asarray(training_targets),
        ridge=1e-8,
        baseline_gain=0.02,
    )

    records: list[dict[str, object]] = []
    for case in validation_cases:
        truth = case.initial_truth(grid)
        estimate = case.initial_estimate(grid)
        open_loop = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.0),
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        fixed_gain = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.10),
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        fixed_gain_low = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain=0.02),
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        learned = simulate_learned_correction(
            model,
            case.nu,
            truth,
            estimate,
            output_times=OUTPUT_TIMES,
        )
        records.append(
            {
                "case_id": case.case_id,
                "nu": case.nu,
                "open_loop": _rollout_record(open_loop, grid, case.nu),
                "fixed_gain_0p02": _rollout_record(fixed_gain_low, grid, case.nu),
                "fixed_gain_0p10": _rollout_record(fixed_gain, grid, case.nu),
                "learned": _rollout_record(learned, grid, case.nu),
            }
        )

        noisy = lambda time: noise_waveform("common-sine", 0.01, matrix.shape[0], time)
        noisy_rollouts = {
            "open_loop": simulate_causal_nudging(
                CausalNudging(grid, case.nu, matrix, gain=0.0),
                truth,
                estimate,
                output_times=OUTPUT_TIMES,
                noise=noisy,
            ),
            "fixed_gain_0p10": simulate_causal_nudging(
                CausalNudging(grid, case.nu, matrix, gain=0.10),
                truth,
                estimate,
                output_times=OUTPUT_TIMES,
                noise=noisy,
            ),
            "fixed_gain_0p02": simulate_causal_nudging(
                CausalNudging(grid, case.nu, matrix, gain=0.02),
                truth,
                estimate,
                output_times=OUTPUT_TIMES,
                noise=noisy,
            ),
            "learned": simulate_learned_correction(
                model,
                case.nu,
                truth,
                estimate,
                output_times=OUTPUT_TIMES,
                noise=noisy,
            ),
        }
        records[-1]["common_sine_noise_0p01"] = {
            method: _rollout_record(rollout, grid, case.nu)
            for method, rollout in noisy_rollouts.items()
        }

    def median(method: str, metric: str) -> float:
        return float(np.median([record[method][metric] for record in records]))

    def median_noise(method: str, metric: str) -> float:
        return float(
            np.median(
                [record["common_sine_noise_0p01"][method][metric] for record in records]
            )
        )

    learned_terminal = [record["learned"]["terminal_error_mass"] for record in records]
    open_terminal = [record["open_loop"]["terminal_error_mass"] for record in records]
    return {
        "kind": "exploratory",
        "experiment": "r5-d-causal-correction-smoke",
        "grid_size": grid_size,
        "train_case_count": len(train_cases),
        "validation_case_count": len(validation_cases),
        "training_sample_count": len(training_estimates),
        "training_information": "offline truth targets; rollout uses estimate and measurement only",
        "summary": {
            method: {
                "median_terminal_error_mass": median(method, "terminal_error_mass"),
                "median_peak_error_mass": median(method, "peak_error_mass"),
                "median_energy_defect": median(method, "energy_defect"),
                "median_noisy_terminal_error_mass": median_noise(
                    method, "terminal_error_mass"
                ),
            }
            for method in ("open_loop", "fixed_gain_0p02", "fixed_gain_0p10", "learned")
        },
        "learned_terminal_better_than_open_loop_fraction": float(
            np.mean(np.asarray(learned_terminal) < np.asarray(open_terminal))
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--validation-limit", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path("out/r5-d-causal-correction-smoke.json")
    )
    args = parser.parse_args()
    result = run(args.grid_size, args.train_limit, args.validation_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
