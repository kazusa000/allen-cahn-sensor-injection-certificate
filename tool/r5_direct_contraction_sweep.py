"""Pre-registered R5 screen for direct transformed-error contraction training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r5_tk_joint_train import run


CONFIGURATIONS = (
    {"name": "direct-weight-0.1", "contraction_weight": 0.1},
    {"name": "direct-weight-1", "contraction_weight": 1.0},
    {"name": "direct-weight-10", "contraction_weight": 10.0},
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
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        print(f"[direct-contraction] configuration={name}", flush=True)
        try:
            result = run(
                torch,
                [31],
                args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_limit=48,
                noise_limit=12,
                device=args.device,
                lambda_ratio=0.1,
                base_gain=0.10,
                gain_scale=0.5,
                certificate_scale=1.0,
                stable_normalization="error-time",
                stable_weight=1.0,
                defect_weight=1.0,
                contraction_weight=float(configuration["contraction_weight"]),
                contraction_margin_ratio=0.1,
                bi_weight=1.0,
                lower_lipschitz=0.5,
                upper_lipschitz=2.0,
                refresh_interval=20,
                selection_limit=12,
                selection_baseline_gain=0.10,
                certificate_kind="triangular",
                mixing_layers=2,
                shear_norm_limit=0.2,
                replay_snapshots=0,
                gain_warmup_epochs=0,
                certificate_warmup_epochs=20,
                gain_learning_rate=5.0e-4,
                certificate_learning_rate=2.0e-3,
                gradient_clip_norm=1.0,
                gain_trust_ratio=0.5,
                gain_reg_weight=1.0,
                gain_kind="mass-adjoint-constant",
                selection_mode="contraction-first",
                run_defect_audit=True,
                run_contraction_audit=True,
                checkpoint_dir=args.checkpoint_dir / name,
            )
        except RuntimeError as error:
            print(f"[direct-contraction] failed={name}: {error}", flush=True)
            runs.append(
                {
                    "configuration": configuration,
                    "run_status": "failed",
                    "failure": str(error),
                }
            )
            continue
        grid = result["results"][0]
        validation = grid["contraction_audits"][
            "current_observer_validation"
        ]["overall"]
        noisy_validation = grid["contraction_audits"][
            "noisy_current_observer_validation"
        ]["overall"]
        selected_seed = next(
            item
            for item in grid["seed_results"]
            if item["seed"] == grid["selected_seed"]
        )
        runs.append(
            {
                "configuration": configuration,
                "run_status": "completed",
                "selected_seed": grid["selected_seed"],
                "validation_contraction": validation,
                "noisy_validation_contraction": noisy_validation,
                "validation_median_terminal_error_mass": selected_seed[
                    "validation_median_terminal_error_mass"
                ],
                "selection_baseline_median_terminal_error_mass": grid[
                    "selection_baseline_median_terminal_error_mass"
                ],
                "certificate_constraints_passed": grid[
                    "selection_constraint_passed"
                ],
                "result": result,
            }
        )

    completed = [item for item in runs if item["run_status"] == "completed"]
    eligible = [
        item
        for item in completed
        if item["certificate_constraints_passed"]
        and item["validation_median_terminal_error_mass"]
        <= item["selection_baseline_median_terminal_error_mass"]
    ]
    selected = (
        max(
            eligible or completed,
            key=lambda item: (
                item["validation_contraction"]["min"],
                item["validation_contraction"]["p05"],
                -item["validation_median_terminal_error_mass"],
            ),
        )
        if completed
        else None
    )
    output = {
        "kind": "r5-direct-contraction-training-screen",
        "pre_registered_configurations": CONFIGURATIONS,
        "margin": "lambda = 0.1 * nu * pi**2",
        "selection_rule": (
            "certificate constraints and no validation rollout regression, then "
            "maximum worst sampled validation contraction rate"
        ),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "status": "completed" if completed else "all-configurations-failed",
        "selected_configuration": (
            selected["configuration"] if selected is not None else None
        ),
        "positive_validation_margin": bool(
            selected is not None and selected["validation_contraction"]["min"] > 0.0
        ),
        "requested_validation_margin_passed": bool(
            selected is not None
            and selected["validation_contraction"]["requested_margin_passed"]
        ),
        "runs": runs,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "selected_configuration": output["selected_configuration"],
                "positive_validation_margin": output["positive_validation_margin"],
                "requested_validation_margin_passed": output[
                    "requested_validation_margin_passed"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
