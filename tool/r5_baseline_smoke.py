"""Run a small, exploratory R5 causal-baseline sweep.

This is intentionally not the formal R5 runner. It is a local profiling tool
used before any remote multi-seed training job is authorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from allen_cahn_certified_observer.dataset import generate_pilot_cases, noise_waveform
from allen_cahn_certified_observer.grid import AllenCahnGrid
from allen_cahn_certified_observer.observations import local_average_matrix
from allen_cahn_certified_observer.observer import (
    CausalNudging,
    simulate_causal_nudging,
)

INTERVALS = np.array([[0.20, 0.30], [0.65, 0.75]], dtype=float)
GAINS = (0.0, 0.02, 0.05, 0.10)


def run(limit: int) -> dict[str, object]:
    cases = [case for case in generate_pilot_cases() if case.split == "train"][:limit]
    records: list[dict[str, object]] = []
    for case in cases:
        grid = AllenCahnGrid(case.n)
        matrix = local_average_matrix(grid, INTERVALS)
        truth = case.initial_truth(grid)
        estimate = case.initial_estimate(grid)
        for gain in GAINS:
            observer = CausalNudging(grid, case.nu, matrix, gain)
            rollout = simulate_causal_nudging(
                observer,
                truth,
                estimate,
                t_span=(0.0, 1.0),
                output_times=np.linspace(0.0, 1.0, 101),
                noise=lambda time, output_dimension=matrix.shape[0]: noise_waveform(
                    "common-sine", 0.01, output_dimension, time
                ),
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "gain": gain,
                    "solver_status": rollout.solver_status,
                    "initial_error_mass": float(rollout.error_mass_norm[0]),
                    "terminal_error_mass": float(rollout.error_mass_norm[-1]),
                    "peak_error_mass": float(np.max(rollout.error_mass_norm)),
                }
            )
    return {
        "kind": "exploratory",
        "experiment": "r5-baseline-smoke",
        "case_count": len(cases),
        "record_count": len(records),
        "intervals": INTERVALS.tolist(),
        "gains": list(GAINS),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path("out/r5-baseline-smoke.json")
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    result = run(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("case_count", "record_count")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
