"""Diatomic molecule definitions."""

from __future__ import annotations

from dataclasses import dataclass


ELEMENT_SYMBOLS = frozenset(
    {
        "H", "He",
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
        "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }
)


def normalize_element_symbol(symbol: str) -> str:
    """Normalize and validate an elemental symbol."""
    cleaned = symbol.strip()

    if not cleaned:
        raise ValueError("Element symbol must not be empty.")

    normalized = cleaned[0].upper() + cleaned[1:].lower()

    if normalized not in ELEMENT_SYMBOLS:
        raise ValueError(
            f"Unknown element symbol: {symbol!r}"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class DiatomicMolecule:
    """A two-atom molecular composition."""

    atom_a: str
    atom_b: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_a",
            normalize_element_symbol(self.atom_a),
        )
        object.__setattr__(
            self,
            "atom_b",
            normalize_element_symbol(self.atom_b),
        )

    @property
    def atoms(self) -> tuple[str, str]:
        """Return the two atomic symbols."""
        return self.atom_a, self.atom_b

    @property
    def formula(self) -> str:
        """Return a compact molecular formula."""
        if self.atom_a == self.atom_b:
            return f"{self.atom_a}2"

        return f"{self.atom_a}{self.atom_b}"

    def __str__(self) -> str:
        return self.formula