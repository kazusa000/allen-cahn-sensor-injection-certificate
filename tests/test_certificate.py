import numpy as np

from allen_cahn_certified_observer import (
    AllenCahnGrid,
    IdentityCertificate,
    NullspaceCertificate,
    audit_certificate,
    local_average_matrix,
)


def _samples(grid: AllenCahnGrid) -> tuple[np.ndarray, np.ndarray]:
    states = []
    errors = []
    for scale in (0.5, 1.0, 1.5):
        states.append(scale * np.sin(np.pi * grid.x))
        errors.append(0.1 * scale * np.sin(2.0 * np.pi * grid.x))
    return np.asarray(states), np.asarray(errors)


def test_identity_certificate_passes_basic_audit() -> None:
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, np.array([[0.25, 0.35], [0.65, 0.75]]))
    states, errors = _samples(grid)

    audit = audit_certificate(IdentityCertificate(), matrix, states, errors)

    assert audit.sample_count == 3
    assert audit.passed_basic_constraints
    assert audit.max_zero_fiber_residual == 0.0
    assert audit.max_direction_residual == 0.0
    assert np.isclose(audit.min_jacobian_singular_value, 1.0, atol=1e-8)
    assert np.isclose(audit.max_jacobian_singular_value, 1.0, atol=1e-8)


def test_nullspace_certificate_preserves_observed_direction() -> None:
    grid = AllenCahnGrid(31)
    matrix = local_average_matrix(grid, np.array([[0.20, 0.30], [0.65, 0.80]]))
    states, errors = _samples(grid)
    certificate = NullspaceCertificate(matrix, amplitude=0.05, state_scale=0.5)

    audit = audit_certificate(certificate, matrix, states, errors)

    assert audit.passed_basic_constraints
    assert audit.max_zero_fiber_residual <= 1e-12
    assert audit.max_direction_residual <= 1e-12
    assert audit.min_jacobian_singular_value > 0.9
    assert audit.max_jacobian_singular_value < 1.1
