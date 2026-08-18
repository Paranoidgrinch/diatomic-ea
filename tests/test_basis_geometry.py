"""Tests for geometry and basis/ECP logic."""

import pytest

from diatomic_ea.basis import (
    bse_def2_name,
    ecp_candidates_for_basis,
    split_bse_nwchem_basis_ecp,
    uses_def2_ecp,
)
from diatomic_ea.geometry import DiatomicGeometry
from diatomic_ea.molecule import DiatomicMolecule


def test_diatomic_geometry() -> None:
    geometry = DiatomicGeometry(
        molecule=DiatomicMolecule("Al", "O"),
        bond_length_angstrom=1.62,
    )

    assert geometry.atom_string == (
        "Al 0.0 0.0 0.0; "
        "O 0.0 0.0 1.6200000000"
    )


def test_invalid_bond_length() -> None:
    with pytest.raises(ValueError):
        DiatomicGeometry(
            molecule=DiatomicMolecule("Al", "O"),
            bond_length_angstrom=0.0,
        )


def test_def2_is_all_electron_through_kr() -> None:
    assert not uses_def2_ecp("H")
    assert not uses_def2_ecp("Fe")
    assert not uses_def2_ecp("Kr")


def test_def2_uses_ecp_after_kr() -> None:
    assert uses_def2_ecp("Rb")
    assert uses_def2_ecp("Zr")
    assert uses_def2_ecp("W")
    assert uses_def2_ecp("Bi")


def test_diffuse_tzvpd_ecp_candidates() -> None:
    candidates = ecp_candidates_for_basis(
        "def2-tzvpd"
    )

    assert candidates == (
        "def2-tzvpd",
        "def2-tzvp",
        "def2-tzvp",
        "def2-ecp",
    ) or candidates == (
        "def2-tzvpd",
        "def2-tzvp",
        "def2-ecp",
    )


def test_qzvpd_ecp_candidates_are_unique() -> None:
    candidates = ecp_candidates_for_basis(
        "def2-qzvpd"
    )

    assert candidates == (
        "def2-qzvpd",
        "def2-qzvp",
        "def2-qzvpp",
        "def2-ecp",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("def2-svp", "def2-SVP"),
        ("DEF2-TZVPD", "def2-TZVPD"),
        ("def2-qzvpd", "def2-QZVPD"),
    ],
)
def test_bse_name_mapping(
    raw: str,
    expected: str,
) -> None:
    assert bse_def2_name(raw) == expected


def test_bse_block_split() -> None:
    text = """header
BASIS "ao basis" SPHERICAL
H S
  1.0 1.0
END
comment
ECP
Rb nelec 28
END
"""

    basis, ecp = split_bse_nwchem_basis_ecp(
        text
    )

    assert basis.startswith("BASIS")
    assert "H S" in basis
    assert ecp.startswith("ECP")
    assert "Rb nelec 28" in ecp