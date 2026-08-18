"""Tests for diatomic molecule definitions."""

import pytest

from diatomic_ea.molecule import (
    DiatomicMolecule,
    normalize_element_symbol,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Al", "Al"),
        ("al", "Al"),
        ("AL", "Al"),
        (" o ", "O"),
        ("fe", "Fe"),
        ("CL", "Cl"),
    ],
)
def test_element_symbol_normalization(
    raw: str,
    expected: str,
) -> None:
    assert normalize_element_symbol(raw) == expected


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "Xx",
        "ABC",
        "1",
    ],
)
def test_invalid_element_symbols(symbol: str) -> None:
    with pytest.raises(ValueError):
        normalize_element_symbol(symbol)


def test_heteronuclear_formula() -> None:
    molecule = DiatomicMolecule("al", "o")

    assert molecule.atom_a == "Al"
    assert molecule.atom_b == "O"
    assert molecule.formula == "AlO"
    assert molecule.atoms == ("Al", "O")


def test_homonuclear_formula() -> None:
    molecule = DiatomicMolecule("O", "O")

    assert molecule.formula == "O2"


def test_string_representation() -> None:
    molecule = DiatomicMolecule("Mg", "O")

    assert str(molecule) == "MgO"