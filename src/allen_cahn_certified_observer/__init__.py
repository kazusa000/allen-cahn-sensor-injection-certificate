"""Allen–Cahn reference solver and certified-observer research tools."""

from .certificate import (
    CertificateAudit,
    IdentityCertificate,
    NullspaceCertificate,
    audit_certificate,
)
from .dataset import PilotCase, generate_pilot_cases, noise_waveform
from .grid import AllenCahnGrid
from .linearization import (
    allen_cahn_jacobian,
    incremental_remainder,
    local_incremental_rhs,
)
from .observations import local_average_matrix
from .observer import (
    CausalNudging,
    CausalOutputInjection,
    ObserverRollout,
    simulate_causal_nudging,
)
from .observer_design import (
    ModalInjectionDesign,
    UnstableModalSystem,
    finite_horizon_transient_amplification,
    linearized_error_matrix,
    lmi_modal_injection,
    mass_adjoint_injection,
    normalized_modal_transform,
    pole_placement_modal_injection,
    riccati_modal_injection,
    symmetric_allen_cahn_margin,
    unstable_modal_system,
)
from .solver import (
    AllenCahnSolution,
    allen_cahn_energy,
    allen_cahn_rhs,
    solve_allen_cahn,
)
from .training import (
    StateConditionedLinearCorrection,
    fit_state_conditioned_linear_correction,
    simulate_learned_correction,
)

__all__ = [
    "AllenCahnGrid",
    "AllenCahnSolution",
    "CausalNudging",
    "CausalOutputInjection",
    "CertificateAudit",
    "IdentityCertificate",
    "NullspaceCertificate",
    "ModalInjectionDesign",
    "ObserverRollout",
    "PilotCase",
    "StateConditionedLinearCorrection",
    "UnstableModalSystem",
    "allen_cahn_energy",
    "allen_cahn_jacobian",
    "allen_cahn_rhs",
    "audit_certificate",
    "fit_state_conditioned_linear_correction",
    "finite_horizon_transient_amplification",
    "generate_pilot_cases",
    "incremental_remainder",
    "linearized_error_matrix",
    "lmi_modal_injection",
    "local_average_matrix",
    "local_incremental_rhs",
    "mass_adjoint_injection",
    "normalized_modal_transform",
    "noise_waveform",
    "pole_placement_modal_injection",
    "riccati_modal_injection",
    "simulate_causal_nudging",
    "simulate_learned_correction",
    "solve_allen_cahn",
    "symmetric_allen_cahn_margin",
    "unstable_modal_system",
]
