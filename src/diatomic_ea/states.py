"""Electronic-state scan model for DiatomicEA.

PySCF convention:
    spin = N_alpha - N_beta = 2S
    multiplicity = spin + 1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ChargeState(IntEnum):
    """Charge states used for electron-affinity calculations."""

    NEUTRAL = 0
    ANION = -1


@dataclass(frozen=True, slots=True)
class ElectronicState:
    """One electronic spin state."""

    charge: ChargeState
    spin: int

    def __post_init__(self) -> None:
        if self.spin < 0:
            raise ValueError(
                "PySCF spin must not be negative."
            )

    @property
    def multiplicity(self) -> int:
        """Return 2S + 1."""
        return self.spin + 1

    @property
    def expected_s2(self) -> float:
        """Return the ideal S(S+1) value."""
        s = self.spin / 2.0
        return s * (s + 1.0)


@dataclass(frozen=True, slots=True)
class ChargeStateScan:
    """Spin states scanned for one molecular charge."""

    charge: ChargeState
    electron_count: int
    states: tuple[ElectronicState, ...]

    def __post_init__(self) -> None:
        if self.electron_count < 1:
            raise ValueError(
                "electron_count must be at least 1."
            )

        if not self.states:
            raise ValueError(
                "At least one state must be scanned."
            )


@dataclass(frozen=True, slots=True)
class StateScanPlan:
    """Neutral and anionic state spaces for one molecule."""

    neutral: ChargeStateScan
    anion: ChargeStateScan


def allowed_spins_from_nelectron(
    nelectron: int,
    spin_max: int,
) -> tuple[int, ...]:
    """Return all PySCF-compatible spins through spin_max.

    This reproduces the parity logic used in the original
    diatomic calculation workflow.
    """
    if nelectron < 1:
        raise ValueError(
            "nelectron must be at least 1."
        )

    if spin_max < 0:
        raise ValueError(
            "spin_max must not be negative."
        )

    start = nelectron % 2
    upper = min(
        int(spin_max),
        int(nelectron),
    )

    spins = tuple(
        range(
            start,
            upper + 1,
            2,
        )
    )

    if not spins:
        raise ValueError(
            "No allowed spin states for the given "
            "electron count and spin limit."
        )

    return spins


def build_charge_state_scan(
    charge: ChargeState,
    *,
    electron_count: int,
    spin_max: int,
) -> ChargeStateScan:
    """Build the spin-state search space for one charge."""
    spins = allowed_spins_from_nelectron(
        electron_count,
        spin_max,
    )

    return ChargeStateScan(
        charge=charge,
        electron_count=electron_count,
        states=tuple(
            ElectronicState(
                charge=charge,
                spin=spin,
            )
            for spin in spins
        ),
    )


def build_state_scan_plan(
    *,
    neutral_electrons: int,
    anion_electrons: int,
    spin_max: int,
) -> StateScanPlan:
    """Build neutral and anionic spin scans."""
    return StateScanPlan(
        neutral=build_charge_state_scan(
            ChargeState.NEUTRAL,
            electron_count=neutral_electrons,
            spin_max=spin_max,
        ),
        anion=build_charge_state_scan(
            ChargeState.ANION,
            electron_count=anion_electrons,
            spin_max=spin_max,
        ),
    )


def is_spin_contaminated(
    *,
    input_spin: int,
    observed_s2: float,
    observed_multiplicity: float,
) -> bool:
    """Apply the diagnostic thresholds from the legacy workflow."""
    state = ElectronicState(
        charge=ChargeState.NEUTRAL,
        spin=input_spin,
    )

    s2_delta = abs(
        observed_s2 - state.expected_s2
    )

    multiplicity_delta = abs(
        observed_multiplicity
        - state.multiplicity
    )

    return (
        s2_delta > 0.75
        or multiplicity_delta > 1.0
    )