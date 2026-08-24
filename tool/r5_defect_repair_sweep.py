"""Pre-registered R5 screen for dynamics-defect generalization repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from r5_tk_joint_train import run

HISTORICAL_N31_VALIDATION_DEFECT = 0.538230836391449

CONFIGURATIONS = (
    {
        "name": "diagonal-current-policy",
        "certificate_kind": "diagonal",
        "mixing_layers": 0,
        "replay_snapshots": 0,
        "gain_warmup_epochs": 0,
        "certificate_warmup_epochs": 0,
    },
    {
        "name": "diagonal-mixed-replay",
        "certificate_kind": "diagonal",
        "mixing_layers": 0,
        "replay_snapshots": 2,
        "gain_warmup_epochs": 0,
        "certificate_warmup_epochs": 0,
    },
    {
        "name": "givens-mixed-replay",
        "certificate_kind": "givens",
        "mixing_layers": 2,
        "replay_snapshots": 2,
        "gain_warmup_epochs": 0,
        "certificate_warmup_epochs": 0,
    },
    {
        "name": "givens-mixed-curriculum",
        "certificate_kind": "givens",
        "mixing_layers": 2,
        "replay_snapshots": 2,
        "gain_warmup_epochs": 20,
        "certificate_warmup_epochs": 20,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.epochs < 41:
        raise SystemExit("--epochs must exceed the 40-epoch curriculum")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        print(f"[repair-screen] configuration={name}", flush=True)
        result = run(
            torch,
            [31],
            args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_limit=12,
            noise_limit=4,
            device=args.device,
            lambda_ratio=0.1,
            base_gain=0.02,
            gain_scale=0.5,
            certificate_scale=1.0,
            stable_normalization="error-time",
            stable_weight=1.0,
            defect_weight=1.0,
            bi_weight=1.0,
            lower_lipschitz=0.5,
            upper_lipschitz=2.0,
            refresh_interval=20,
            selection_limit=12,
            selection_baseline_gain=0.10,
            certificate_kind=str(configuration["certificate_kind"]),
            mixing_layers=int(configuration["mixing_layers"]),
            replay_snapshots=int(configuration["replay_snapshots"]),
            gain_warmup_epochs=int(configuration["gain_warmup_epochs"]),
            certificate_warmup_epochs=int(
                configuration["certificate_warmup_epochs"]
            ),
            selection_mode="defect-first",
            run_defect_audit=True,
            checkpoint_dir=args.checkpoint_dir / name,
        )
        grid = result["results"][0]
        audits = grid["defect_audits"]
        fixed_rms = audits["fixed_gain_validation"]["overall"]["rms"]
        current_rms = audits["current_observer_validation"]["overall"]["rms"]
        runs.append(
            {
                "configuration": configuration,
                "fixed_gain_validation_defect_rms": fixed_rms,
                "current_observer_validation_defect_rms": current_rms,
                "worst_validation_defect_rms": max(fixed_rms, current_rms),
                "validation_median_terminal_error_mass": grid[
                    "seed_results"
                ][
                    next(
                        index
                        for index, item in enumerate(grid["seed_results"])
                        if item["seed"] == grid["selected_seed"]
                    )
                ]["validation_median_terminal_error_mass"],
                "selection_baseline_median_terminal_error_mass": grid[
                    "selection_baseline_median_terminal_error_mass"
                ],
                "test_median_terminal_error_mass": grid[
                    "test_median_terminal_error_mass"
                ],
                "certificate_constraints_passed": grid[
                    "selection_constraint_passed"
                ],
                "result": result,
            }
        )

    eligible = [
        item
        for item in runs
        if item["certificate_constraints_passed"]
        and item["validation_median_terminal_error_mass"]
        <= item["selection_baseline_median_terminal_error_mass"]
    ]
    selected = min(
        eligible or runs,
        key=lambda item: (
            item["worst_validation_defect_rms"],
            item["validation_median_terminal_error_mass"],
        ),
    )
    historical_rms = float(np.sqrt(HISTORICAL_N31_VALIDATION_DEFECT))
    output = {
        "kind": "r5-dynamics-defect-repair-screen",
        "pre_registered_configurations": CONFIGURATIONS,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "historical_n31_validation_defect_rms": historical_rms,
        "selection_rule": (
            "certificate constraints and no validation regression, then minimum "
            "worst fixed/current-policy validation defect RMS"
        ),
        "selected_configuration": selected["configuration"],
        "selected_worst_validation_defect_rms": selected[
            "worst_validation_defect_rms"
        ],
        "half_defect_progress_gate_passed": bool(
            selected["worst_validation_defect_rms"] <= 0.5 * historical_rms
        ),
        "runs": runs,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["configuration"]["name"],
                "worst_validation_defect_rms": selected[
                    "worst_validation_defect_rms"
                ],
                "half_defect_progress_gate_passed": output[
                    "half_defect_progress_gate_passed"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
