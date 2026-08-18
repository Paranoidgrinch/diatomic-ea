"""One-dimensional geometries for diatomic molecules."""

from __future__ import annotations

from dataclasses import dataclass

from diatomic_ea.molecule import DiatomicMolecule


@dataclass(frozen=True, slots=True)
class DiatomicGeometry:
    """A diatomic molecule at one internuclear distance."""

    molecule: DiatomicMolecule
    bond_length_angstrom: float

    def __post_init__(self) -> None:
        if self.bond_length_angstrom <= 0:
            raise ValueError(
                "bond_length_angstrom must be positive."
            )

    @property
    def atom_string(self) -> str:
        """Return a PySCF-compatible molecular geometry."""
        return (
            f"{self.molecule.atom_a} 0.0 0.0 0.0; "
            f"{self.molecule.atom_b} 0.0 0.0 "
            f"{self.bond_length_angstrom:.10f}"
        )