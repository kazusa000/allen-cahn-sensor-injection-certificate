"""Screen normalized, on-policy R5 joint-training settings on one grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r5_tk_joint_train import ABLATION_SEEDS, run


def _selected_seed_result(grid_result: dict[str, object]) -> dict[str, object]:
    selected_seed = int(grid_result["selected_seed"])
    return next(
        item
        for item in grid_result["seed_results"]
        if int(item["seed"]) == selected_seed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument(
        "--lambda-ratios", type=float, nargs="+", default=[0.1, 0.5, 1.0]
    )
    parser.add_argument(
        "--defect-weights", type=float, nargs="+", default=[0.1, 1.0]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ABLATION_SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument("--noise-limit", type=int, default=4)
    parser.add_argument("--selection-limit", type=int, default=12)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-gain", type=float, default=0.02)
    parser.add_argument("--gain-scale", type=float, default=0.5)
    parser.add_argument("--certificate-scale", type=float, default=1.0)
    parser.add_argument("--bi-weight", type=float, default=1.0)
    parser.add_argument("--lower-lipschitz", type=float, default=0.5)
    parser.add_argument("--upper-lipschitz", type=float, default=2.0)
    parser.add_argument("--selection-baseline-gain", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if any(value <= 0.0 for value in args.lambda_ratios):
        raise SystemExit("lambda ratios must be positive")
    if any(value < 0.0 for value in args.defect_weights):
        raise SystemExit("defect weights must be nonnegative")

    runs: list[dict[str, object]] = []
    for lambda_ratio in args.lambda_ratios:
        for defect_weight in args.defect_weights:
            print(
                f"[screen] lambda_ratio={lambda_ratio:g} "
                f"defect_weight={defect_weight:g}",
                flush=True,
            )
            result = run(
                torch,
                [args.grid_size],
                args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_limit=args.eval_limit,
                noise_limit=args.noise_limit,
                device=args.device,
                lambda_ratio=lambda_ratio,
                base_gain=args.base_gain,
                gain_scale=args.gain_scale,
                certificate_scale=args.certificate_scale,
                stable_normalization="error-time",
                stable_weight=1.0,
                defect_weight=defect_weight,
                bi_weight=args.bi_weight,
                lower_lipschitz=args.lower_lipschitz,
                upper_lipschitz=args.upper_lipschitz,
                refresh_interval=args.refresh_interval,
                selection_limit=args.selection_limit,
                selection_baseline_gain=args.selection_baseline_gain,
            )
            grid_result = result["results"][0]
            selected = _selected_seed_result(grid_result)
            runs.append(
                {
                    "lambda_ratio": lambda_ratio,
                    "defect_weight": defect_weight,
                    "selected_seed": grid_result["selected_seed"],
                    "selection_constraint_passed": grid_result[
                        "selection_constraint_passed"
                    ],
                    "validation_median_terminal_error_mass": selected[
                        "validation_median_terminal_error_mass"
                    ],
                    "selection_baseline_median_terminal_error_mass": grid_result[
                        "selection_baseline_median_terminal_error_mass"
                    ],
                    "stable_validation_loss": selected["stable_validation_loss"],
                    "stable_raw_validation_loss": selected[
                        "stable_raw_validation_loss"
                    ],
                    "defect_validation_loss": selected[
                        "defect_validation_loss"
                    ],
                    "bi_validation_loss": selected["bi_validation_loss"],
                    "test_median_terminal_error_mass": grid_result[
                        "test_median_terminal_error_mass"
                    ],
                    "noisy_median_terminal_error_mass": grid_result[
                        "noisy_median_terminal_error_mass"
                    ],
                    "certificate_audit": grid_result["certificate_audit"],
                    "result": result,
                }
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    eligible = [item for item in runs if item["selection_constraint_passed"]]
    best = min(
        eligible or runs,
        key=lambda item: (
            item["validation_median_terminal_error_mass"],
            item["defect_validation_loss"],
        ),
    )
    output = {
        "kind": "r5-normalized-refresh-screen",
        "grid_size": args.grid_size,
        "lambda_ratios": args.lambda_ratios,
        "defect_weights": args.defect_weights,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "selection_limit": args.selection_limit,
        "refresh_interval": args.refresh_interval,
        "device": args.device,
        "selected": {
            "lambda_ratio": best["lambda_ratio"],
            "defect_weight": best["defect_weight"],
            "selected_seed": best["selected_seed"],
            "validation_median_terminal_error_mass": best[
                "validation_median_terminal_error_mass"
            ],
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["selected"]), flush=True)


if __name__ == "__main__":
    main()
