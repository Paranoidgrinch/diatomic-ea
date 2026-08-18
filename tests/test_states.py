"""Tests for electronic-state handling."""

import pytest

from diatomic_ea.states import (
    ChargeState,
    ElectronicState,
    allowed_spins_from_nelectron,
    build_state_scan_plan,
    is_spin_contaminated,
)


def test_even_electron_spin_parity() -> None:
    assert allowed_spins_from_nelectron(
        20,
        7,
    ) == (0, 2, 4, 6)


def test_odd_electron_spin_parity() -> None:
    assert allowed_spins_from_nelectron(
        21,
        7,
    ) == (1, 3, 5, 7)


def test_spin_is_limited_by_electron_count() -> None:
    assert allowed_spins_from_nelectron(
        3,
        15,
    ) == (1, 3)


def test_invalid_electron_count() -> None:
    with pytest.raises(ValueError):
        allowed_spins_from_nelectron(
            0,
            5,
        )


def test_multiplicity_conversion() -> None:
    state = ElectronicState(
        charge=ChargeState.NEUTRAL,
        spin=4,
    )

    assert state.multiplicity == 5


def test_expected_s2() -> None:
    state = ElectronicState(
        charge=ChargeState.NEUTRAL,
        spin=2,
    )

    assert state.expected_s2 == pytest.approx(2.0)


def test_state_scan_plan() -> None:
    plan = build_state_scan_plan(
        neutral_electrons=20,
        anion_electrons=21,
        spin_max=7,
    )

    assert tuple(
        state.spin
        for state in plan.neutral.states
    ) == (0, 2, 4, 6)

    assert tuple(
        state.spin
        for state in plan.anion.states
    ) == (1, 3, 5, 7)


def test_small_spin_deviation_is_not_flagged() -> None:
    assert not is_spin_contaminated(
        input_spin=2,
        observed_s2=2.2,
        observed_multiplicity=3.1,
    )


def test_large_s2_deviation_is_flagged() -> None:
    assert is_spin_contaminated(
        input_spin=2,
        observed_s2=3.0,
        observed_multiplicity=3.0,
    )


def test_large_multiplicity_deviation_is_flagged() -> None:
    assert is_spin_contaminated(
        input_spin=2,
        observed_s2=2.0,
        observed_multiplicity=4.5,
    )