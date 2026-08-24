"""Audit direct transformed-error contraction for an existing R5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from r5_e_joint_train import INTERVALS
from r5_tk_joint_train import (
    _audit,
    _build_models,
    _collect_policy_samples,
    _collect_samples,
    _contraction_audit,
    _split_cases,
)

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    local_average_matrix,
    noise_waveform,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_models(
    torch: object, checkpoint_path: Path, device: str
) -> tuple[dict, object, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    grid = AllenCahnGrid(int(checkpoint["grid_size"]))
    matrix = local_average_matrix(grid, INTERVALS)
    gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=float(checkpoint["base_gain"]),
        gain_scale=float(checkpoint["gain_scale"]),
        certificate_scale=float(checkpoint["certificate_scale"]),
        lower_lipschitz=float(checkpoint["lower_lipschitz"]),
        upper_lipschitz=float(checkpoint["upper_lipschitz"]),
        certificate_kind=str(checkpoint["certificate_kind"]),
        mixing_layers=int(checkpoint["mixing_layers"]),
        shear_norm_limit=float(checkpoint["shear_norm_limit"]),
        gain_trust_ratio=float(checkpoint["gain_trust_ratio"]),
        gain_kind=str(checkpoint["gain_kind"]),
    )
    gain.load_state_dict(checkpoint["gain_state_dict"])
    certificate.load_state_dict(checkpoint["certificate_state_dict"])
    gain.to(device).eval()
    certificate.to(device).eval()
    return checkpoint, gain, certificate


def run(
    torch: object,
    checkpoint_path: Path,
    *,
    device: str,
    margin_ratio: float,
    batch_size: int,
    noise_limit: int,
) -> dict[str, object]:
    checkpoint, gain, certificate = _load_models(torch, checkpoint_path, device)
    grid = AllenCahnGrid(int(checkpoint["grid_size"]))
    matrix = local_average_matrix(grid, INTERVALS)
    train_cases = _split_cases("train", grid.n)
    validation_cases = _split_cases("validation", grid.n)
    fixed_train = _collect_samples(
        train_cases, grid, matrix, base_gain=float(checkpoint["base_gain"])
    )
    fixed_validation = _collect_samples(
        validation_cases, grid, matrix, base_gain=float(checkpoint["base_gain"])
    )
    current_train = _collect_policy_samples(
        torch, gain, device, train_cases, grid, matrix
    )
    current_validation = _collect_policy_samples(
        torch, gain, device, validation_cases, grid, matrix
    )
    noisy = lambda time, q=matrix.shape[0]: noise_waveform(
        "common-sine", 0.01, q, time
    )
    noisy_validation = _collect_policy_samples(
        torch,
        gain,
        device,
        validation_cases[:noise_limit],
        grid,
        matrix,
        noise=noisy,
    )
    sample_sets = {
        "fixed_gain_train": fixed_train,
        "current_observer_train": current_train,
        "fixed_gain_validation": fixed_validation,
        "current_observer_validation": current_validation,
        "noisy_current_observer_validation": noisy_validation,
    }
    audits = {
        name: _contraction_audit(
            torch,
            gain,
            certificate,
            sample_set,
            grid,
            matrix,
            margin_ratio=margin_ratio,
            device=device,
            batch_size=batch_size,
        )
        for name, sample_set in sample_sets.items()
    }
    validation = audits["current_observer_validation"]["overall"]
    return {
        "kind": "r5-direct-contraction-checkpoint-audit",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "grid_size": grid.n,
        "seed": int(checkpoint["seed"]),
        "device": device,
        "margin_ratio": margin_ratio,
        "noise": {
            "kind": "common-sine",
            "amplitude": 0.01,
            "validation_case_count": noise_limit,
        },
        "certificate_audit": _audit(
            torch, certificate, matrix, grid, device
        ),
        "audits": audits,
        "positive_noiseless_validation_margin": bool(
            validation["positive_worst_sample_margin"]
        ),
        "requested_noiseless_validation_margin_passed": bool(
            validation["requested_margin_passed"]
        ),
        "conclusion_scope": (
            "finite sampled R5 trajectories only; not a continuous-domain uniform theorem"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--noise-limit", type=int, default=12)
    args = parser.parse_args()
    if args.margin_ratio < 0.0:
        raise SystemExit("--margin-ratio must be nonnegative")
    if args.noise_limit < 1:
        raise SystemExit("--noise-limit must be positive")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        torch,
        args.checkpoint,
        device=args.device,
        margin_ratio=args.margin_ratio,
        batch_size=args.batch_size,
        noise_limit=args.noise_limit,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "positive_noiseless_validation_margin": result[
                    "positive_noiseless_validation_margin"
                ],
                "requested_noiseless_validation_margin_passed": result[
                    "requested_noiseless_validation_margin_passed"
                ],
                "validation": result["audits"]["current_observer_validation"][
                    "overall"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
