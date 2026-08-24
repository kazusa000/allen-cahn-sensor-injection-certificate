"""Validation-only screen for the bounded R5 oblique joint trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r5_oblique_joint_train import run


CONFIGURATIONS = (
    {
        "name": "gain-only-trust-0.25",
        "gain_trust_ratio": 0.25,
        "certificate_log_scale": 0.0,
    },
    {
        "name": "joint-balanced",
        "gain_trust_ratio": 0.25,
        "certificate_log_scale": 0.2231435513,
    },
    {
        "name": "joint-flexible",
        "gain_trust_ratio": 0.50,
        "certificate_log_scale": 0.4054651081,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-limit-per-nu", type=int, default=16)
    parser.add_argument("--validation-limit-per-nu", type=int, default=8)
    parser.add_argument("--stress-truths-per-nu", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    runs: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        print(f"[configuration={configuration['name']}]", flush=True)
        result = run(
            torch,
            grid_size=args.grid_size,
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            refresh_interval=args.refresh_interval,
            device=args.device,
            train_limit_per_nu=args.train_limit_per_nu,
            validation_limit_per_nu=args.validation_limit_per_nu,
            test_limit_per_nu=1,
            stress_truths_per_nu=args.stress_truths_per_nu,
            gain_trust_ratio=float(configuration["gain_trust_ratio"]),
            certificate_log_scale=float(configuration["certificate_log_scale"]),
            gain_learning_rate=5e-4,
            certificate_learning_rate=1e-3,
            stable_weight=1.0,
            defect_weight=0.05,
            contraction_weight=10.0,
            bi_weight=1.0,
            gain_reg_weight=0.1,
            checkpoint_dir=args.checkpoint_root / str(configuration["name"]),
            run_test=False,
        )
        runs.append({"configuration": configuration, "result": result})

    def selection_key(item: dict[str, object]) -> tuple[float, ...]:
        result = item["result"]
        selected_seed = result["selected_seed"]
        selected = next(
            seed
            for seed in result["seed_results"]
            if seed["seed"] == selected_seed
        )
        return (
            float(result["gates"]["multi_grid_expansion_eligible"]),
            selected["validation_contraction"]["min"],
            selected["validation_contraction"]["p05"],
            -selected["validation_rollout"]["by_nu"]["0.005"][
                "terminal_error_mass_median"
            ],
        )

    selected = max(runs, key=selection_key)
    output = {
        "kind": "r5-oblique-joint-validation-screen",
        "configurations": CONFIGURATIONS,
        "selection_rule": (
            "validation only: expansion gates, then worst contraction, 5% contraction, "
            "then nu=0.005 median terminal error; test is not evaluated"
        ),
        "selected_configuration": selected["configuration"],
        "selected_gates": selected["result"]["gates"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_configuration": output["selected_configuration"],
                "selected_gates": output["selected_gates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
