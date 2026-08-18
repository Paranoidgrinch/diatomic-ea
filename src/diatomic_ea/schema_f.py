"""Frozen Schema F workflow specification.

The public Schema F preset represents the reproducible diatomic
electron-affinity workflow used by DiatomicEA. Scientific parameters
must not be silently changed by the GUI or CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Numerical settings for one bond-length scan stage."""

    step_angstrom: float
    grid_level: int
    conv_tol: float
    max_cycle: int
    max_memory_mb: int

    def __post_init__(self) -> None:
        if self.step_angstrom <= 0:
            raise ValueError(
                "step_angstrom must be positive."
            )

        if self.grid_level < 0:
            raise ValueError(
                "grid_level must not be negative."
            )

        if self.conv_tol <= 0:
            raise ValueError(
                "conv_tol must be positive."
            )

        if self.max_cycle < 1:
            raise ValueError(
                "max_cycle must be at least 1."
            )

        if self.max_memory_mb < 1:
            raise ValueError(
                "max_memory_mb must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class RefinementSpec:
    """Large-basis local refinement settings."""

    basis: str
    window_angstrom: float
    grid: GridSpec
    max_spins_per_charge: int

    def __post_init__(self) -> None:
        if self.window_angstrom <= 0:
            raise ValueError(
                "window_angstrom must be positive."
            )

        if self.max_spins_per_charge < 1:
            raise ValueError(
                "max_spins_per_charge must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class SCFRescueSpec:
    """SCF recovery strategy inherited from the validated workflow."""

    level_shift: float
    newton_max_cycle: int

    def __post_init__(self) -> None:
        if self.level_shift <= 0:
            raise ValueError(
                "level_shift must be positive."
            )

        if self.newton_max_cycle < 1:
            raise ValueError(
                "newton_max_cycle must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class SchemaFSpec:
    """Immutable scientific definition of Schema F."""

    schema_id: str
    reference_pyscf_version: str
    electronic_structure_method: str
    functionals: tuple[str, ...]
    fast_bases: tuple[str, ...]
    fast_grid: GridSpec
    refinement: RefinementSpec
    scf_rescue: SCFRescueSpec
    all_electron_through_atomic_number: int

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError(
                "schema_id must not be empty."
            )

        if not self.functionals:
            raise ValueError(
                "At least one functional is required."
            )

        if not self.fast_bases:
            raise ValueError(
                "At least one fast-grid basis is required."
            )


SCHEMA_F = SchemaFSpec(
    schema_id="schema-f-v1",
    reference_pyscf_version="2.13.0",
    electronic_structure_method="UKS-DFT",
    functionals=(
        "PBE",
        "B3LYP",
        "PBE0",
        "TPSSh",
    ),
    fast_bases=(
        "def2-svp",
        "def2-tzvp",
        "def2-tzvpp",
        "def2-svpd",
        "def2-tzvpd",
    ),
    fast_grid=GridSpec(
        step_angstrom=0.025,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    ),
    refinement=RefinementSpec(
        basis="def2-qzvpd",
        window_angstrom=0.10,
        grid=GridSpec(
            step_angstrom=0.01,
            grid_level=4,
            conv_tol=1.0e-8,
            max_cycle=250,
            max_memory_mb=6000,
        ),
        max_spins_per_charge=2,
    ),
    scf_rescue=SCFRescueSpec(
        level_shift=0.25,
        newton_max_cycle=80,
    ),
    all_electron_through_atomic_number=36,
)


def adiabatic_ea_ev(
    neutral_energy_hartree: float,
    anion_energy_hartree: float,
) -> float:
    """Calculate an adiabatic EA from optimized total energies."""
    return (
        neutral_energy_hartree
        - anion_energy_hartree
    ) * HARTREE_TO_EV