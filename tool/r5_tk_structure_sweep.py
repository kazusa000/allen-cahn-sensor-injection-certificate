"""Stage-two R5 screen for the T--K triangular-transform structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from r5_tk_joint_train import run

HISTORICAL_N31_VALIDATION_DEFECT = 0.538230836391449

CONFIGURATIONS = (
    {
        "name": "triangular-current-policy",
        "replay_snapshots": 0,
        "stable_weight": 1.0,
        "gain_scale": 0.5,
        "gain_trust_ratio": 0.5,
    },
    {
        "name": "triangular-mixed-replay",
        "replay_snapshots": 2,
        "stable_weight": 1.0,
        "gain_scale": 0.5,
        "gain_trust_ratio": 0.5,
    },
    {
        "name": "triangular-balanced",
        "replay_snapshots": 2,
        "stable_weight": 0.5,
        "gain_scale": 0.5,
        "gain_trust_ratio": 0.5,
    },
    {
        "name": "triangular-wide-gain",
        "replay_snapshots": 2,
        "stable_weight": 1.0,
        "gain_scale": 0.5,
        "gain_trust_ratio": 0.9,
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        print(f"[tk-structure-screen] configuration={name}", flush=True)
        try:
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
                base_gain=0.10,
                gain_scale=float(configuration["gain_scale"]),
                certificate_scale=1.0,
                stable_normalization="error-time",
                stable_weight=float(configuration["stable_weight"]),
                defect_weight=1.0,
                bi_weight=1.0,
                lower_lipschitz=0.5,
                upper_lipschitz=2.0,
                refresh_interval=20,
                selection_limit=12,
                selection_baseline_gain=0.10,
                certificate_kind="triangular",
                mixing_layers=2,
                shear_norm_limit=0.2,
                replay_snapshots=int(configuration["replay_snapshots"]),
                gain_warmup_epochs=0,
                certificate_warmup_epochs=20,
                gain_learning_rate=5.0e-4,
                certificate_learning_rate=2.0e-3,
                gradient_clip_norm=1.0,
                gain_trust_ratio=float(configuration["gain_trust_ratio"]),
                gain_reg_weight=1.0,
                gain_kind="mass-adjoint-constant",
                selection_mode="defect-first",
                run_defect_audit=True,
                checkpoint_dir=args.checkpoint_dir / name,
            )
        except RuntimeError as error:
            print(f"[tk-structure-screen] failed={name}: {error}", flush=True)
            runs.append(
                {
                    "configuration": configuration,
                    "run_status": "failed",
                    "failure": str(error),
                }
            )
            continue
        grid = result["results"][0]
        selected_seed_result = next(
            item
            for item in grid["seed_results"]
            if item["seed"] == grid["selected_seed"]
        )
        audits = grid["defect_audits"]
        fixed_rms = audits["fixed_gain_validation"]["overall"]["rms"]
        current_rms = audits["current_observer_validation"]["overall"]["rms"]
        runs.append(
            {
                "configuration": configuration,
                "run_status": "completed",
                "fixed_gain_validation_defect_rms": fixed_rms,
                "current_observer_validation_defect_rms": current_rms,
                "worst_validation_defect_rms": max(fixed_rms, current_rms),
                "validation_median_terminal_error_mass": selected_seed_result[
                    "validation_median_terminal_error_mass"
                ],
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

    completed = [item for item in runs if item["run_status"] == "completed"]
    if not completed:
        output = {
            "kind": "r5-tk-triangular-structure-screen",
            "pre_registered_configurations": CONFIGURATIONS,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "status": "all-configurations-failed",
            "runs": runs,
        }
        args.output.write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": output["status"]}), flush=True)
        return
    eligible = [
        item
        for item in completed
        if item["certificate_constraints_passed"]
        and item["validation_median_terminal_error_mass"]
        <= item["selection_baseline_median_terminal_error_mass"]
    ]
    selected = min(
        eligible or completed,
        key=lambda item: (
            item["worst_validation_defect_rms"],
            item["validation_median_terminal_error_mass"],
        ),
    )
    historical_rms = float(np.sqrt(HISTORICAL_N31_VALIDATION_DEFECT))
    output = {
        "kind": "r5-tk-triangular-structure-screen",
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
