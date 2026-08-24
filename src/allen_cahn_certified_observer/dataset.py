"""Deterministic R5 pilot case generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .grid import AllenCahnGrid

SPLIT_SEEDS: dict[str, tuple[int, ...]] = {
    "train": tuple(range(501, 509)),
    "validation": tuple(range(601, 605)),
    "test": tuple(range(701, 705)),
}
NU_VALUES = (0.005, 0.01, 0.02)
GRID_SIZES = (31, 63, 127)


@dataclass(frozen=True)
class PilotCase:
    split: str
    seed: int
    draw: int
    n: int
    nu: float
    state_coefficients: tuple[float, float, float]
    error_coefficients: tuple[float, float, float]

    @property
    def case_id(self) -> str:
        nu_token = f"{self.nu:.3f}".replace(".", "p")
        return (
            f"r5-pilot-v1__split-{self.split}__seed-{self.seed}"
            f"__draw-{self.draw}__n-{self.n}__nu-{nu_token}"
        )

    def initial_truth(self, grid: AllenCahnGrid) -> np.ndarray:
        if grid.n != self.n:
            raise ValueError("case and grid sizes do not match")
        return _modal_state(grid, self.state_coefficients)

    def initial_estimate(self, grid: AllenCahnGrid) -> np.ndarray:
        return self.initial_truth(grid) + _modal_state(grid, self.error_coefficients)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["case_id"] = self.case_id
        return value


def _modal_state(
    grid: AllenCahnGrid, coefficients: tuple[float, float, float]
) -> np.ndarray:
    return sum(
        coefficient * np.sin(mode * np.pi * grid.x)
        for mode, coefficient in enumerate(coefficients, start=1)
    )


def _coefficient_draw(
    seed: int, draw: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    generator = np.random.Generator(np.random.PCG64DXSM(seed + 1009 * draw))
    state = generator.uniform(-0.5, 0.5, size=3)
    error_direction = generator.normal(size=3)
    error_direction /= np.linalg.norm(error_direction)
    error = error_direction * generator.uniform(0.05, 0.25)
    return tuple(float(value) for value in state), tuple(
        float(value) for value in error
    )


def generate_pilot_cases() -> tuple[PilotCase, ...]:
    """Generate paired train/validation/test cases for the frozen pilot contract."""

    cases: list[PilotCase] = []
    for split, seeds in SPLIT_SEEDS.items():
        for seed in seeds:
            for draw in range(4):
                state, error = _coefficient_draw(seed, draw)
                for n in GRID_SIZES:
                    for nu in NU_VALUES:
                        cases.append(
                            PilotCase(
                                split=split,
                                seed=seed,
                                draw=draw,
                                n=n,
                                nu=nu,
                                state_coefficients=state,
                                error_coefficients=error,
                            )
                        )
    return tuple(cases)


def noise_waveform(
    kind: str,
    amplitude: float,
    output_dimension: int,
    time: float,
) -> np.ndarray:
    """Return one of the fixed deterministic measurement stress waveforms."""

    if amplitude < 0.0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be a non-negative finite scalar")
    if output_dimension < 1:
        raise ValueError("output_dimension must be positive")
    unit = np.ones(output_dimension, dtype=float) / np.sqrt(output_dimension)
    if kind == "zero":
        return np.zeros(output_dimension, dtype=float)
    if kind == "constant-plus":
        return amplitude * unit
    if kind == "constant-minus":
        return -amplitude * unit
    if kind == "common-sine":
        return amplitude * np.sin(2.0 * np.pi * time) * unit
    raise ValueError(f"unknown deterministic noise waveform: {kind}")
