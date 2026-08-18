"""Scientific reduction of Schema F fast-grid results."""

from __future__ import annotations

import math
from dataclasses import dataclass

from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.grid import BondGrid
from diatomic_ea.schema_f import HARTREE_TO_EV


TRUE_VALUES = frozenset(
    {
        "true",
        "1",
        "yes",
        "y",
        "ok",
    }
)


def _as_bool(value: object) -> bool:
    """Parse CSV-style boolean values."""
    if isinstance(value, bool):
        return value

    return (
        str(value).strip().lower()
        in TRUE_VALUES
    )


def _as_float(value: object) -> float:
    """Parse a floating-point CSV field."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass(frozen=True, slots=True)
class RawGridPoint:
    """One parsed latest-result CSV row."""

    task_id: str
    molecule: str
    charge: int
    spin: int
    multiplicity: int
    functional: str
    basis: str
    bond_length_angstrom: float
    status: str
    energy_hartree: float
    converged: bool
    positive_homo_warning: bool
    spin_contamination_warning: bool

    @property
    def usable_energy(self) -> bool:
        """Return whether this point contains a usable energy."""
        return (
            self.status.strip().lower() == "ok"
            and math.isfinite(
                self.energy_hartree
            )
        )


@dataclass(frozen=True, slots=True)
class StateMinimum:
    """Minimum for one functional/basis/charge/spin state."""

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
class ChargeMinimum:
    """Lowest spin state for one method and charge."""

    state_minimum: StateMinimum

    @property
    def molecule(self) -> str:
        return self.state_minimum.point.molecule

    @property
    def charge(self) -> int:
        return self.state_minimum.point.charge

    @property
    def functional(self) -> str:
        return self.state_minimum.point.functional

    @property
    def basis(self) -> str:
        return self.state_minimum.point.basis

    @property
    def spin(self) -> int:
        return self.state_minimum.point.spin

    @property
    def multiplicity(self) -> int:
        return self.state_minimum.point.multiplicity

    @property
    def energy_hartree(self) -> float:
        return (
            self.state_minimum
            .point
            .energy_hartree
        )

    @property
    def bond_length_angstrom(self) -> float:
        return (
            self.state_minimum
            .point
            .bond_length_angstrom
        )


@dataclass(frozen=True, slots=True)
class MethodEA:
    """Adiabatic EA for one functional/basis pair."""

    molecule: str
    functional: str
    basis: str
    neutral: ChargeMinimum
    anion: ChargeMinimum
    ea_ev: float
    hard_warnings: tuple[str, ...]
    diagnostic_warnings: tuple[str, ...]

    @property
    def recommended_for_fast_summary(self) -> bool:
        """Return whether no hard diagnostic excludes this estimate."""
        return not self.hard_warnings


@dataclass(frozen=True, slots=True)
class FastGridAnalysis:
    """Complete scientific reduction of one fast grid."""

    points: tuple[RawGridPoint, ...]
    state_minima: tuple[StateMinimum, ...]
    charge_minima: tuple[ChargeMinimum, ...]
    method_eas: tuple[MethodEA, ...]


def _point_from_row(
    row: dict[str, str],
) -> RawGridPoint:
    return RawGridPoint(
        task_id=row.get("task_id", ""),
        molecule=row.get("molecule", ""),
        charge=int(
            row.get("charge", "0")
        ),
        spin=int(
            row.get("spin", "0")
        ),
        multiplicity=int(
            row.get("multiplicity", "1")
        ),
        functional=row.get(
            "functional",
            "",
        ),
        basis=row.get(
            "basis",
            "",
        ),
        bond_length_angstrom=_as_float(
            row.get(
                "bond_length_angstrom",
                "",
            )
        ),
        status=row.get(
            "status",
            "",
        ),
        energy_hartree=_as_float(
            row.get(
                "energy_hartree",
                "",
            )
        ),
        converged=_as_bool(
            row.get(
                "converged",
                "",
            )
        ),
        positive_homo_warning=_as_bool(
            row.get(
                "positive_homo_warning",
                "",
            )
        ),
        spin_contamination_warning=_as_bool(
            row.get(
                "spin_contamination_warning",
                "",
            )
        ),
    )


def load_latest_grid_points(
    store: RawResultStore,
) -> tuple[RawGridPoint, ...]:
    """Load the newest stored result for every task."""
    points = tuple(
        _point_from_row(row)
        for row
        in store.latest_rows().values()
    )

    return tuple(
        sorted(
            points,
            key=lambda point: point.task_id,
        )
    )


def _minimum_warnings(
    point: RawGridPoint,
    bond_grid: BondGrid,
) -> tuple[
    bool,
    tuple[str, ...],
    tuple[str, ...],
]:
    tolerance = 1.0e-9

    at_edge = (
        point.bond_length_angstrom
        <= (
            bond_grid.minimum_angstrom
            + tolerance
        )
        or point.bond_length_angstrom
        >= (
            bond_grid.maximum_angstrom
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
            "minimum_at_grid_edge"
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


def build_state_minima(
    points: tuple[RawGridPoint, ...],
    bond_grid: BondGrid,
) -> tuple[StateMinimum, ...]:
    """Find the lowest-energy geometry for every electronic state."""
    minima: dict[
        tuple[
            str,
            int,
            str,
            str,
            int,
            int,
        ],
        RawGridPoint,
    ] = {}

    for point in points:
        if not point.usable_energy:
            continue

        key = (
            point.molecule,
            point.charge,
            point.functional,
            point.basis,
            point.spin,
            point.multiplicity,
        )

        previous = minima.get(key)

        if (
            previous is None
            or point.energy_hartree
            < previous.energy_hartree
        ):
            minima[key] = point

    output: list[StateMinimum] = []

    for point in minima.values():
        (
            edge_warning,
            hard_warnings,
            diagnostic_warnings,
        ) = _minimum_warnings(
            point,
            bond_grid,
        )

        output.append(
            StateMinimum(
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
                minimum.point.basis,
                -minimum.point.charge,
                minimum.point.spin,
            ),
        )
    )


def build_charge_minima(
    state_minima: tuple[StateMinimum, ...],
) -> tuple[ChargeMinimum, ...]:
    """Select the lowest spin-state energy for each method and charge."""
    winners: dict[
        tuple[
            str,
            int,
            str,
            str,
        ],
        StateMinimum,
    ] = {}

    for minimum in state_minima:
        point = minimum.point

        key = (
            point.molecule,
            point.charge,
            point.functional,
            point.basis,
        )

        previous = winners.get(key)

        if (
            previous is None
            or point.energy_hartree
            < previous.point.energy_hartree
        ):
            winners[key] = minimum

    output = tuple(
        ChargeMinimum(
            state_minimum=minimum
        )
        for minimum in winners.values()
    )

    return tuple(
        sorted(
            output,
            key=lambda minimum: (
                minimum.molecule,
                minimum.functional,
                minimum.basis,
                -minimum.charge,
            ),
        )
    )


def _merge_warnings(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    merged: list[str] = []

    for group in groups:
        for warning in group:
            if warning not in merged:
                merged.append(warning)

    return tuple(merged)


def build_method_eas(
    charge_minima: tuple[ChargeMinimum, ...],
) -> tuple[MethodEA, ...]:
    """Pair neutral and anionic minima for each method."""
    grouped: dict[
        tuple[str, str, str],
        dict[int, ChargeMinimum],
    ] = {}

    for minimum in charge_minima:
        key = (
            minimum.molecule,
            minimum.functional,
            minimum.basis,
        )

        grouped.setdefault(
            key,
            {},
        )[minimum.charge] = minimum

    output: list[MethodEA] = []

    for (
        molecule,
        functional,
        basis,
    ), charges in grouped.items():
        neutral = charges.get(0)
        anion = charges.get(-1)

        if neutral is None or anion is None:
            continue

        neutral_state = (
            neutral.state_minimum
        )
        anion_state = (
            anion.state_minimum
        )

        ea_ev = (
            neutral.energy_hartree
            - anion.energy_hartree
        ) * HARTREE_TO_EV

        output.append(
            MethodEA(
                molecule=molecule,
                functional=functional,
                basis=basis,
                neutral=neutral,
                anion=anion,
                ea_ev=ea_ev,
                hard_warnings=_merge_warnings(
                    neutral_state.hard_warnings,
                    anion_state.hard_warnings,
                ),
                diagnostic_warnings=_merge_warnings(
                    neutral_state.diagnostic_warnings,
                    anion_state.diagnostic_warnings,
                ),
            )
        )

    return tuple(
        sorted(
            output,
            key=lambda result: (
                result.molecule,
                result.functional,
                result.basis,
            ),
        )
    )


def analyze_fast_grid(
    store: RawResultStore,
    bond_grid: BondGrid,
) -> FastGridAnalysis:
    """Run the complete scientific reduction of a fast grid."""
    points = load_latest_grid_points(
        store
    )

    state_minima = build_state_minima(
        points,
        bond_grid,
    )

    charge_minima = build_charge_minima(
        state_minima
    )

    method_eas = build_method_eas(
        charge_minima
    )

    return FastGridAnalysis(
        points=points,
        state_minima=state_minima,
        charge_minima=charge_minima,
        method_eas=method_eas,
    )