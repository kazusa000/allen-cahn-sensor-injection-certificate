import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    allen_cahn_jacobian,
    allen_cahn_rhs,
    incremental_remainder,
    local_incremental_rhs,
)


def test_jacobian_and_remainder_reconstruct_incremental_rhs() -> None:
    grid = AllenCahnGrid(31)
    base = 0.3 * np.sin(np.pi * grid.x)
    increment = 0.05 * np.cos(2.0 * np.pi * grid.x)
    jacobian = allen_cahn_jacobian(grid, 0.01, base)

    direct = local_incremental_rhs(grid, 0.01, base, increment)
    reconstructed = jacobian @ increment + incremental_remainder(
        grid, 0.01, base, increment
    )

    assert np.allclose(direct, reconstructed, rtol=0.0, atol=2e-14)
    assert np.allclose(
        direct,
        allen_cahn_rhs(grid, 0.01, base + increment) - allen_cahn_rhs(grid, 0.01, base),
        rtol=0.0,
        atol=2e-14,
    )


def test_local_jacobian_is_symmetric() -> None:
    grid = AllenCahnGrid(15)
    base = 0.2 * np.sin(np.pi * grid.x)

    jacobian = allen_cahn_jacobian(grid, 0.02, base)

    assert np.allclose(jacobian, jacobian.T, rtol=0.0, atol=1e-14)
    assert np.all(np.isfinite(np.linalg.eigvalsh(jacobian)))
