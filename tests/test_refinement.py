"""Tests for QZVPD candidate selection."""

from diatomic_ea.analysis import (
    ChargeMinimum,
    FastGridAnalysis,
    RawGridPoint,
    StateMinimum,
)
from diatomic_ea.refinement import (
    select_qzvpd_candidates,
)


def make_state(
    *,
    charge: int,
    spin: int,
    functional: str,
    basis: str,
    energy: float,
    r: float,
    hard_warnings: tuple[str, ...] = (),
) -> StateMinimum:
    point = RawGridPoint(
        task_id=(
            f"{charge}_{spin}_"
            f"{functional}_{basis}_{r}"
        ),
        molecule="AlO",
        charge=charge,
        spin=spin,
        multiplicity=spin + 1,
        functional=functional,
        basis=basis,
        bond_length_angstrom=r,
        status="ok",
        energy_hartree=energy,
        converged=True,
        positive_homo_warning=False,
        spin_contamination_warning=False,
    )

    return StateMinimum(
        point=point,
        grid_edge_warning=(
            "minimum_at_grid_edge"
            in hard_warnings
        ),
        hard_warnings=hard_warnings,
        diagnostic_warnings=(),
    )


def make_analysis() -> FastGridAnalysis:
    neutral_spin0_svp = make_state(
        charge=0,
        spin=0,
        functional="PBE",
        basis="def2-svp",
        energy=-100.5,
        r=1.50,
    )

    neutral_spin0_tzvpd = make_state(
        charge=0,
        spin=0,
        functional="PBE",
        basis="def2-tzvpd",
        energy=-100.0,
        r=1.60,
    )

    neutral_spin0_b3lyp = make_state(
        charge=0,
        spin=0,
        functional="B3LYP",
        basis="def2-tzvpd",
        energy=-101.0,
        r=1.61,
    )

    neutral_spin2 = make_state(
        charge=0,
        spin=2,
        functional="PBE0",
        basis="def2-tzvpd",
        energy=-102.0,
        r=1.70,
    )

    neutral_spin4 = make_state(
        charge=0,
        spin=4,
        functional="TPSSh",
        basis="def2-tzvpd",
        energy=-103.0,
        r=1.80,
    )

    anion_spin1_pbe = make_state(
        charge=-1,
        spin=1,
        functional="PBE",
        basis="def2-tzvpd",
        energy=-101.0,
        r=1.65,
    )

    anion_spin1_b3lyp = make_state(
        charge=-1,
        spin=1,
        functional="B3LYP",
        basis="def2-tzvpd",
        energy=-102.0,
        r=1.66,
    )

    anion_spin3 = make_state(
        charge=-1,
        spin=3,
        functional="PBE0",
        basis="def2-tzvpd",
        energy=-103.0,
        r=1.75,
    )

    states = (
        neutral_spin0_svp,
        neutral_spin0_tzvpd,
        neutral_spin0_b3lyp,
        neutral_spin2,
        neutral_spin4,
        anion_spin1_pbe,
        anion_spin1_b3lyp,
        anion_spin3,
    )

    charge_minima = (
        ChargeMinimum(
            neutral_spin0_svp
        ),
        ChargeMinimum(
            neutral_spin0_b3lyp
        ),
        ChargeMinimum(
            neutral_spin2
        ),
        ChargeMinimum(
            neutral_spin4
        ),
        ChargeMinimum(
            anion_spin1_pbe
        ),
        ChargeMinimum(
            anion_spin1_b3lyp
        ),
        ChargeMinimum(
            anion_spin3
        ),
    )

    return FastGridAnalysis(
        points=(),
        state_minima=states,
        charge_minima=charge_minima,
        method_eas=(),
    )


def test_top_two_neutral_spins_are_selected() -> None:
    candidates = select_qzvpd_candidates(
        make_analysis()
    )

    neutral_spins = {
        candidate.spin
        for candidate in candidates
        if candidate.charge == 0
    }

    assert neutral_spins == {
        0,
        2,
    }


def test_spin_win_count_is_preserved() -> None:
    candidates = select_qzvpd_candidates(
        make_analysis()
    )

    spin_zero = [
        candidate
        for candidate in candidates
        if (
            candidate.charge == 0
            and candidate.spin == 0
        )
    ]

    assert spin_zero

    assert all(
        candidate.method_minimum_count_for_spin
        == 2
        for candidate in spin_zero
    )


def test_tzvpd_is_preferred_as_geometry_source() -> None:
    candidates = select_qzvpd_candidates(
        make_analysis()
    )

    candidate = next(
        item
        for item in candidates
        if (
            item.charge == 0
            and item.spin == 0
            and item.functional == "PBE"
        )
    )

    assert (
        candidate.source_basis
        == "def2-tzvpd"
    )

    assert (
        candidate.r_center_angstrom
        == 1.60
    )


def test_qzvpd_window_is_plus_minus_point_one() -> None:
    candidates = select_qzvpd_candidates(
        make_analysis()
    )

    candidate = next(
        item
        for item in candidates
        if (
            item.charge == 0
            and item.spin == 0
            and item.functional == "PBE"
        )
    )

    assert (
        candidate.r_min_angstrom
        == 1.50
    )

    assert (
        candidate.r_max_angstrom
        == 1.70
    )


def test_frozen_qzvpd_basis_is_used() -> None:
    candidates = select_qzvpd_candidates(
        make_analysis()
    )

    assert candidates

    assert all(
        candidate.qzvpd_basis
        == "def2-qzvpd"
        for candidate in candidates
    )


def test_hard_warning_winner_does_not_vote() -> None:
    reliable = make_state(
        charge=0,
        spin=0,
        functional="PBE",
        basis="def2-tzvpd",
        energy=-100.0,
        r=1.6,
    )

    bad = make_state(
        charge=0,
        spin=6,
        functional="PBE0",
        basis="def2-tzvpd",
        energy=-200.0,
        r=1.7,
        hard_warnings=(
            "minimum_at_grid_edge",
        ),
    )

    analysis = FastGridAnalysis(
        points=(),
        state_minima=(
            reliable,
            bad,
        ),
        charge_minima=(
            ChargeMinimum(
                reliable
            ),
            ChargeMinimum(
                bad
            ),
        ),
        method_eas=(),
    )

    candidates = select_qzvpd_candidates(
        analysis
    )

    spins = {
        candidate.spin
        for candidate in candidates
    }

    assert 0 in spins
    assert 6 not in spins


def test_hard_warning_source_is_not_used() -> None:
    good_svp = make_state(
        charge=0,
        spin=0,
        functional="PBE",
        basis="def2-svp",
        energy=-100.0,
        r=1.55,
    )

    bad_tzvpd = make_state(
        charge=0,
        spin=0,
        functional="PBE",
        basis="def2-tzvpd",
        energy=-101.0,
        r=1.60,
        hard_warnings=(
            "not_converged",
        ),
    )

    analysis = FastGridAnalysis(
        points=(),
        state_minima=(
            good_svp,
            bad_tzvpd,
        ),
        charge_minima=(
            ChargeMinimum(
                good_svp
            ),
        ),
        method_eas=(),
    )

    candidates = select_qzvpd_candidates(
        analysis
    )

    candidate = candidates[0]

    assert (
        candidate.source_basis
        == "def2-svp"
    )

    assert (
        candidate.r_center_angstrom
        == 1.55
    )