import numpy as np

from allen_cahn_certified_observer import AllenCahnGrid


def test_dirichlet_grid_operator_and_mass_are_well_formed() -> None:
    grid = AllenCahnGrid(15)

    assert grid.h == 1.0 / 16.0
    assert grid.x.shape == (15,)
    assert np.allclose(grid.laplacian, grid.laplacian.T)
    assert np.linalg.eigvalsh(grid.laplacian)[-1] < 0.0
    assert np.allclose(grid.mass, grid.h * np.eye(15))


def test_boundary_conversion_is_explicit() -> None:
    grid = AllenCahnGrid(7)
    interior = np.arange(7, dtype=float)
    full = grid.with_boundary(interior)

    assert full.shape == (9,)
    assert full[0] == 0.0 and full[-1] == 0.0
    assert np.array_equal(grid.interior(full), interior)
