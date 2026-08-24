import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from r5_e_joint_train import INTERVALS
from r5_tk_joint_train import (
    JointSampleSet,
    _build_models,
    _concatenate_sample_sets,
    _contraction_tensor_metrics,
    _ratio_summary,
    _signed_summary,
)

from allen_cahn_certified_observer import AllenCahnGrid, local_average_matrix


def _sample_set(nu: float, time: float) -> JointSampleSet:
    return JointSampleSet(
        states=np.asarray([[1.0, 2.0]]),
        estimates=np.asarray([[1.1, 2.1]]),
        measurements=np.asarray([[0.2]]),
        next_states=np.asarray([[1.2, 2.2]]),
        nus=np.asarray([nu]),
        nu_indices=np.asarray([0]),
        nu_values=(nu,),
        times=np.asarray([time]),
        dt=0.02,
    )


def test_concatenate_sample_sets_reindexes_viscosity_and_keeps_time() -> None:
    combined = _concatenate_sample_sets(
        _sample_set(0.02, 0.4), _sample_set(0.005, 0.8)
    )

    assert combined.nu_values == (0.005, 0.02)
    assert combined.nu_indices.tolist() == [1, 0]
    assert combined.times.tolist() == [0.4, 0.8]
    assert combined.states.shape == (2, 2)


def test_ratio_summary_reports_rms_and_tail() -> None:
    summary = _ratio_summary(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4
    assert summary["rms"] == pytest.approx(np.sqrt(7.5))
    assert summary["median"] == pytest.approx(2.5)
    assert summary["max"] == pytest.approx(4.0)


def test_signed_summary_preserves_worst_contraction_rate() -> None:
    summary = _signed_summary(np.asarray([-0.5, 0.25, 1.0]))

    assert summary["min"] == pytest.approx(-0.5)
    assert summary["positive_fraction"] == pytest.approx(2.0 / 3.0)


def test_direct_contraction_rate_matches_exponential_decay() -> None:
    torch = pytest.importorskip("torch")
    transformed = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    directional = -2.0 * transformed
    nus = torch.tensor([0.01, 0.02])

    metrics = _contraction_tensor_metrics(
        torch,
        transformed,
        directional,
        nus,
        h=0.25,
        margin_ratio=1.0,
    )

    assert torch.allclose(metrics["rates"], torch.full((2,), 2.0))
    assert metrics["loss"].item() == pytest.approx(0.0)


def test_givens_certificate_is_identity_at_initialization_and_stays_bounded() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    _gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        certificate_kind="givens",
        mixing_layers=2,
    )
    generator = torch.Generator().manual_seed(42)
    states = torch.randn((3, grid.n), generator=generator)
    errors = torch.randn((3, grid.n), generator=generator)

    initial = certificate(states, errors)
    assert torch.allclose(initial, errors, atol=1.0e-6)

    torch.nn.init.normal_(certificate.network[-1].weight, std=0.1)
    torch.nn.init.normal_(certificate.network[-1].bias, std=0.1)
    transformed = certificate(states, errors)
    direction = (transformed - errors) @ torch.as_tensor(
        matrix.T, dtype=torch.float32
    )
    assert torch.max(torch.abs(direction)).item() < 1.0e-5
    assert torch.max(torch.abs(certificate(states, torch.zeros_like(errors)))).item() == 0

    error = errors[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: certificate(states[0:1], value[None, :])[0], error
    )
    singular_values = torch.linalg.svdvals(jacobian)
    assert torch.min(singular_values).item() >= 0.5 - 1.0e-5
    assert torch.max(singular_values).item() <= 2.0 + 1.0e-5


def test_triangular_certificate_adds_bounded_observed_to_nullspace_shear() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    _gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        certificate_kind="triangular",
        mixing_layers=2,
        shear_norm_limit=0.2,
    )
    generator = torch.Generator().manual_seed(43)
    states = torch.randn((3, grid.n), generator=generator)
    errors = torch.randn((3, grid.n), generator=generator)
    assert torch.allclose(certificate(states, errors), errors, atol=1.0e-6)

    torch.nn.init.normal_(certificate.network[-1].weight, std=0.1)
    torch.nn.init.normal_(certificate.network[-1].bias, std=0.1)
    transformed = certificate(states, errors)
    direction = (transformed - errors) @ torch.as_tensor(
        matrix.T, dtype=torch.float32
    )
    assert torch.max(torch.abs(direction)).item() < 1.0e-5
    assert torch.linalg.vector_norm(transformed - errors).item() > 1.0e-5

    error = errors[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: certificate(states[0:1], value[None, :])[0], error
    )
    singular_values = torch.linalg.svdvals(jacobian)
    assert torch.min(singular_values).item() >= 0.5 - 1.0e-5
    assert torch.max(singular_values).item() <= 2.0 + 1.0e-5


def test_triangular_certificate_has_finite_directional_loss_gradients_at_identity() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    _gain, certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        certificate_kind="triangular",
        mixing_layers=2,
        shear_norm_limit=0.2,
    )
    generator = torch.Generator().manual_seed(44)
    states = torch.randn((3, grid.n), generator=generator)
    errors = torch.randn((3, grid.n), generator=generator)
    state_directions = torch.randn((3, grid.n), generator=generator)
    error_directions = torch.randn((3, grid.n), generator=generator)

    _, directional = torch.autograd.functional.jvp(
        certificate,
        (states, errors),
        (state_directions, error_directions),
        create_graph=True,
    )
    torch.mean(directional**2).backward()

    gradients = [
        parameter.grad
        for parameter in certificate.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_gain_trust_region_bounds_state_conditioned_deviation() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    gain, _certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        gain_trust_ratio=0.25,
    )
    torch.nn.init.normal_(gain.network[-1].weight, std=0.2)
    torch.nn.init.normal_(gain.network[-1].bias, std=0.2)
    features = torch.randn((8, grid.n + 2 * matrix.shape[0] + 2))

    learned = gain(features)
    deviation = torch.linalg.vector_norm(
        learned - gain.base_gain[None, :, :], dim=(1, 2)
    )

    assert torch.all(deviation < 0.25 * gain.base_gain_norm)


def test_constant_mass_adjoint_gain_keeps_positive_sensor_injection_structure() -> None:
    torch = pytest.importorskip("torch")
    grid = AllenCahnGrid(15)
    matrix = local_average_matrix(grid, INTERVALS)
    gain, _certificate = _build_models(
        torch,
        grid,
        matrix,
        base_gain=0.02,
        gain_scale=0.5,
        certificate_scale=1.0,
        lower_lipschitz=0.5,
        upper_lipschitz=2.0,
        gain_trust_ratio=0.5,
        gain_kind="mass-adjoint-constant",
    )
    torch.nn.init.normal_(gain.network.logits, std=0.2)
    features = torch.randn((8, grid.n + 2 * matrix.shape[0] + 2))

    learned = gain(features)
    support = torch.abs(gain.base_gain) > 1.0e-12
    ratios = learned[:, support] / gain.base_gain[support]

    assert torch.all(learned[:, ~support] == 0.0)
    assert torch.all(ratios > 0.5)
    assert torch.all(ratios < 1.5)
    assert torch.allclose(learned[0], learned[-1])
