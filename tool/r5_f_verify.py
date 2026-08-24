"""Independent replay and contract checks for an R5-E GPU pilot run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from r5_e_joint_train import (
    INTERVALS,
    SampleSet,
    _build_models,
    _certificate_diagnostics,
    _median,
    _rollout_summary,
    _simulate_neural,
    _split_cases,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    CausalNudging,
    local_average_matrix,
    noise_waveform,
    simulate_causal_nudging,
)


def _baseline_summary(
    cases: list[object], grid: AllenCahnGrid, matrix: np.ndarray, gain: float
) -> dict[str, float | int]:
    records = []
    for case in cases:
        rollout = simulate_causal_nudging(
            CausalNudging(grid, case.nu, matrix, gain),
            case.initial_truth(grid),
            case.initial_estimate(grid),
        )
        records.append(_rollout_summary(rollout, grid, case.nu))
    return {
        "case_count": len(records),
        "median_terminal_error_mass": _median(records, "terminal_error_mass"),
        "median_peak_error_mass": _median(records, "peak_error_mass"),
        "median_energy_defect": _median(records, "energy_defect"),
    }


def run(
    torch: object, run_dir: Path, device: str, test_limit: int, noise_limit: int
) -> dict[str, object]:
    source = run_dir / "results.json"
    frozen = json.loads(source.read_text(encoding="utf-8"))
    grid_results = []
    for frozen_grid in frozen["grid_results"]:
        grid_size = int(frozen_grid["grid_size"])
        grid = AllenCahnGrid(grid_size)
        matrix = local_average_matrix(grid, INTERVALS)
        selected_seed = int(frozen_grid["selected_seed"])
        checkpoint = torch.load(
            run_dir / f"checkpoint-grid-{grid_size}-seed-{selected_seed}.pt",
            map_location=device,
            weights_only=False,
        )
        correction, certificate = _build_models(
            torch,
            grid_size + 6,
            grid_size,
            checkpoint["certificate_state_dict"]["null_basis"].detach().cpu().numpy(),
        )
        correction.load_state_dict(checkpoint["correction_state_dict"])
        certificate.load_state_dict(checkpoint["certificate_state_dict"])
        correction.to(device).eval()
        certificate.to(device).eval()
        test_cases = _split_cases("test", grid_size)
        replay_cases = test_cases[:test_limit]
        replay = [
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
            for case in replay_cases
        ]
        noise_cases = test_cases[:noise_limit]
        noisy = lambda time, q=matrix.shape[0]: noise_waveform(
            "common-sine", 0.01, q, time
        )
        noisy_replay = [
            _simulate_neural(
                torch,
                correction,
                device,
                grid,
                matrix,
                case.nu,
                case.initial_truth(grid),
                case.initial_estimate(grid),
                noise=noisy,
            )
            for case in noise_cases
        ]
        noise_baseline = []
        for case in noise_cases:
            rollout = simulate_causal_nudging(
                CausalNudging(grid, case.nu, matrix, 0.10),
                case.initial_truth(grid),
                case.initial_estimate(grid),
                noise=noisy,
            )
            noise_baseline.append(_rollout_summary(rollout, grid, case.nu))
        rng = np.random.Generator(np.random.PCG64DXSM(9000 + grid_size))
        audit_states = rng.normal(size=(3, grid_size)) * 0.1
        audit_errors = rng.normal(size=(3, grid_size)) * 0.05
        audit_samples = SampleSet(
            features=np.zeros((3, grid_size + 6)),
            correction_targets=np.zeros_like(audit_errors),
            states=audit_states,
            errors=audit_errors,
            certificate_targets=np.zeros_like(audit_errors),
        )
        audit = _certificate_diagnostics(
            torch, certificate, matrix, audit_samples, device, sample_limit=3
        )
        grid_results.append(
            {
                "grid_size": grid_size,
                "selected_seed": selected_seed,
                "replay_case_count": len(replay),
                "replay_median_terminal_error_mass": _median(
                    replay, "terminal_error_mass"
                ),
                "replay_median_peak_error_mass": _median(replay, "peak_error_mass"),
                "replay_median_energy_defect": _median(replay, "energy_defect"),
                "fixed_gain_0p10": _baseline_summary(replay_cases, grid, matrix, 0.10),
                "noise_case_count": len(noisy_replay),
                "noisy_replay_median_terminal_error_mass": _median(
                    noisy_replay, "terminal_error_mass"
                ),
                "noisy_fixed_gain_0p10_median_terminal_error_mass": _median(
                    noise_baseline, "terminal_error_mass"
                ),
                "certificate_audit": audit,
            }
        )
    return {
        "kind": "r5-f-independent-replay",
        "source_kind": frozen["kind"],
        "device": device,
        "test_limit": test_limit,
        "noise_limit": noise_limit,
        "grid_results": grid_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-limit", type=int, default=48)
    parser.add_argument("--noise-limit", type=int, default=12)
    args = parser.parse_args()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    result = run(torch, args.run_dir, args.device, args.test_limit, args.noise_limit)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"grid_count": len(result["grid_results"]), "device": args.device})
    )


if __name__ == "__main__":
    main()
