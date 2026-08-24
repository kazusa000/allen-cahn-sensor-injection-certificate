import numpy as np

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix


def test_local_average_preserves_linear_functions_away_from_boundary() -> None:
    grid = AllenCahnGrid(63)
    intervals = np.array([[0.20, 0.30], [0.65, 0.80]])
    matrix = local_average_matrix(grid, intervals)
    nodal_values = grid.x.copy()
    expected = np.mean(intervals, axis=1)

    assert np.allclose(matrix.sum(axis=1), 1.0, atol=2e-14)
    assert np.allclose(matrix @ nodal_values, expected, atol=2e-14)


def test_local_average_rejects_invalid_intervals() -> None:
    grid = AllenCahnGrid(15)

    for intervals in (
        np.array([[0.4, 0.4]]),
        np.array([[-0.1, 0.2]]),
        np.array([[0.8, 1.2]]),
    ):
        try:
            local_average_matrix(grid, intervals)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid interval was accepted")
