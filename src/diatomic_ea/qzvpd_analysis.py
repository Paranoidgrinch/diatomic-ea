"""Scientific analysis of Schema F QZVPD refinement results."""

from __future__ import annotations

from dataclasses import dataclass

from diatomic_ea.analysis import (
    RawGridPoint,
    load_latest_grid_points,
)
from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.qzvpd import QZVPDPlan
from diatomic_ea.refinement import QZVPDCandidate
from diatomic_ea.schema_f import HARTREE_TO_EV


@dataclass(frozen=True, slots=True)
class QZVPDMinimum:
    """Minimum of one selected charge/spin/functional state."""

    candidate: QZVPDCandidate
    point: RawGridPoint
    grid_edge_warning: bool
    hard_warnings: tuple[str, ...]
    diagnostic_warnings: tuple[str, ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        return (
            self.hard_warnings
            + self.diagnostic_warnings
        )


@dataclass(frozen=True, slots=True)
class QZVPDChargeMinimum:
    """Lowest QZVPD spin state for one functional and charge."""

    minimum: QZVPDMinimum

    @property
    def molecule(self) -> str:
        return self.minimum.point.molecule

    @property
    def charge(self) -> int:
        return self.minimum.point.charge

    @property
    def spin(self) -> int:
        return self.minimum.point.spin

    @property
    def multiplicity(self) -> int:
        return self.minimum.point.multiplicity

    @property
    def functional(self) -> str:
        return self.minimum.point.functional

    @property
    def basis(self) -> str:
        return self.minimum.point.basis

    @property
    def energy_hartree(self) -> float:
        return self.minimum.point.energy_hartree

    @property
    def bond_length_angstrom(self) -> float:
        return (
            self.minimum
            .point
            .bond_length_angstrom
        )


@dataclass(frozen=True, slots=True)
class QZVPDEA:
    """Final QZVPD EA for one density functional."""

    molecule: str
    functional: str
    basis: str
    neutral: QZVPDChargeMinimum
    anion: QZVPDChargeMinimum
    ea_ev: float
    hard_warnings: tuple[str, ...]
    diagnostic_warnings: tuple[str, ...]

    @property
    def recommended_for_summary(self) -> bool:
        return not self.hard_warnings


@dataclass(frozen=True, slots=True)
class QZVPDAnalysis:
    """Complete reduction of a QZVPD calculation stage."""

    points: tuple[RawGridPoint, ...]
    state_minima: tuple[QZVPDMinimum, ...]
    charge_minima: tuple[
        QZVPDChargeMinimum,
        ...,
    ]
    functional_eas: tuple[QZVPDEA, ...]


def _merge_warnings(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    output: list[str] = []

    for group in groups:
        for warning in group:
            if warning not in output:
                output.append(warning)

    return tuple(output)


def _candidate_matches_point(
    candidate: QZVPDCandidate,
    point: RawGridPoint,
) -> bool:
    return (
        candidate.molecule
        == point.molecule
        and candidate.charge
        == point.charge
        and candidate.spin
        == point.spin
        and candidate.functional
        == point.functional
        and candidate.qzvpd_basis.lower()
        == point.basis.lower()
    )


def _minimum_warnings(
    candidate: QZVPDCandidate,
    point: RawGridPoint,
) -> tuple[
    bool,
    tuple[str, ...],
    tuple[str, ...],
]:
    tolerance = 1.0e-9

    at_edge = (
        point.bond_length_angstrom
        <= (
            candidate.r_min_angstrom
            + tolerance
        )
        or point.bond_length_angstrom
        >= (
            candidate.r_max_angstrom
            - tolerance
        )
    )

    hard: list[str] = []
    diagnostic: list[str] = []

    if not point.converged:
        hard.append(
            "not_converged"
        )

    if at_edge:
        hard.append(
            "minimum_at_qzvpd_edge"
        )

    if point.positive_homo_warning:
        diagnostic.append(
            "positive_HOMO"
        )

    if point.spin_contamination_warning:
        diagnostic.append(
            "spin_contamination"
        )

    return (
        at_edge,
        tuple(hard),
        tuple(diagnostic),
    )


def build_qzvpd_state_minima(
    points: tuple[RawGridPoint, ...],
    plan: QZVPDPlan,
) -> tuple[QZVPDMinimum, ...]:
    """Find the QZVPD minimum for every selected candidate."""
    output: list[QZVPDMinimum] = []

    for candidate in plan.candidates:
        matching = [
            point
            for point in points
            if (
                point.usable_energy
                and _candidate_matches_point(
                    candidate,
                    point,
                )
            )
        ]

        if not matching:
            continue

        point = min(
            matching,
            key=lambda item: (
                item.energy_hartree
            ),
        )

        (
            edge_warning,
            hard_warnings,
            diagnostic_warnings,
        ) = _minimum_warnings(
            candidate,
            point,
        )

        output.append(
            QZVPDMinimum(
                candidate=candidate,
                point=point,
                grid_edge_warning=edge_warning,
                hard_warnings=hard_warnings,
                diagnostic_warnings=diagnostic_warnings,
            )
        )

    return tuple(
        sorted(
            output,
            key=lambda minimum: (
                minimum.point.molecule,
                minimum.point.functional,
                -minimum.point.charge,
                minimum.point.spin,
            ),
        )
    )


def build_qzvpd_charge_minima(
    state_minima: tuple[QZVPDMinimum, ...],
) -> tuple[QZVPDChargeMinimum, ...]:
    """Select the lowest QZVPD spin for each functional and charge."""
    winners: dict[
        tuple[
            str,
            str,
            int,
        ],
        QZVPDMinimum,
    ] = {}

    for minimum in state_minima:
        key = (
            minimum.point.molecule,
            minimum.point.functional,
            minimum.point.charge,
        )

        previous = winners.get(key)

        if (
            previous is None
            or minimum.point.energy_hartree
            < previous.point.energy_hartree
        ):
            winners[key] = minimum

    output = tuple(
        QZVPDChargeMinimum(
            minimum=minimum
        )
        for minimum in winners.values()
    )

    return tuple(
        sorted(
            output,
            key=lambda minimum: (
                minimum.molecule,
                minimum.functional,
                -minimum.charge,
            ),
        )
    )


def build_qzvpd_eas(
    charge_minima: tuple[
        QZVPDChargeMinimum,
        ...,
    ],
) -> tuple[QZVPDEA, ...]:
    """Calculate one QZVPD EA per functional."""
    grouped: dict[
        tuple[str, str],
        dict[int, QZVPDChargeMinimum],
    ] = {}

    for minimum in charge_minima:
        key = (
            minimum.molecule,
            minimum.functional,
        )

        grouped.setdefault(
            key,
            {},
        )[minimum.charge] = minimum

    output: list[QZVPDEA] = []

    for (
        molecule,
        functional,
    ), charges in grouped.items():
        neutral = charges.get(0)
        anion = charges.get(-1)

        if neutral is None or anion is None:
            continue

        hard = _merge_warnings(
            neutral.minimum.hard_warnings,
            anion.minimum.hard_warnings,
        )

        diagnostic = _merge_warnings(
            neutral.minimum.diagnostic_warnings,
            anion.minimum.diagnostic_warnings,
        )

        if neutral.basis.lower() != anion.basis.lower():
            hard = _merge_warnings(
                hard,
                ("basis_mismatch",),
            )

        output.append(
            QZVPDEA(
                molecule=molecule,
                functional=functional,
                basis=neutral.basis,
                neutral=neutral,
                anion=anion,
                ea_ev=(
                    neutral.energy_hartree
                    - anion.energy_hartree
                ) * HARTREE_TO_EV,
                hard_warnings=hard,
                diagnostic_warnings=diagnostic,
            )
        )

    return tuple(
        sorted(
            output,
            key=lambda result: (
                result.molecule,
                result.functional,
            ),
        )
    )


def analyze_qzvpd(
    store: RawResultStore,
    plan: QZVPDPlan,
) -> QZVPDAnalysis:
    """Run the complete QZVPD scientific reduction."""
    points = load_latest_grid_points(
        store
    )

    state_minima = (
        build_qzvpd_state_minima(
            points,
            plan,
        )
    )

    charge_minima = (
        build_qzvpd_charge_minima(
            state_minima
        )
    )

    functional_eas = (
        build_qzvpd_eas(
            charge_minima
        )
    )

    return QZVPDAnalysis(
        points=points,
        state_minima=state_minima,
        charge_minima=charge_minima,
        functional_eas=functional_eas,
    )