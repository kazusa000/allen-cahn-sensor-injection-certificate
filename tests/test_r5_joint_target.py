import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_tk_joint_train import _target_operators

from allen_cahn_certified_observer import AllenCahnGrid


def test_r5_target_uses_stable_diffusion_shift() -> None:
    grid = AllenCahnGrid(15)
    nu_values = (0.005, 0.01, 0.02)
    lambda_ratio = 0.5

    generators, maps = _target_operators(grid, nu_values, lambda_ratio)

    identity = np.eye(grid.n)
    for nu, generator, step_map in zip(
        nu_values, generators, maps, strict=True
    ):
        expected = (
            nu * grid.laplacian
            - lambda_ratio * nu * np.pi**2 * identity
        )
        assert np.allclose(generator, expected)
        assert np.max(np.linalg.eigvalsh(generator)) < 0.0
        assert np.max(np.abs(np.linalg.eigvals(step_map))) < 1.0
