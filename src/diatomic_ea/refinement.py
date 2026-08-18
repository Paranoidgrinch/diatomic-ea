"""Selection of Schema F QZVPD refinement candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from diatomic_ea.analysis import (
    FastGridAnalysis,
    StateMinimum,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
    SchemaFSpec,
)


BASIS_PRIORITY = {
    "def2-tzvpd": 0,
    "def2-tzvpp": 1,
    "def2-tzvp": 2,
    "def2-svpd": 3,
    "def2-svp": 4,
}


@dataclass(frozen=True, slots=True)
class QZVPDCandidate:
    """One selected state/function pair for QZVPD refinement."""

    molecule: str
    charge: int
    spin: int
    multiplicity: int
    functional: str
    qzvpd_basis: str
    r_center_angstrom: float
    r_min_angstrom: float
    r_max_angstrom: float
    source_basis: str
    source_energy_hartree: float
    method_minimum_count_for_spin: int
    source_warnings: tuple[str, ...]

    @property
    def charge_label(self) -> str:
        if self.charge == 0:
            return "neutral"

        if self.charge == -1:
            return "anion"

        return f"q{self.charge:+d}"


def _selected_spin_counts(
    analysis: FastGridAnalysis,
    *,
    charge: int,
    maximum: int,
) -> tuple[
    tuple[int, int, int],
    ...,
]:
    """Rank reliable spin states by method-level wins."""
    if maximum < 1:
        raise ValueError(
            "maximum must be at least 1."
        )

    counts: Counter[
        tuple[int, int]
    ] = Counter()

    for minimum in analysis.charge_minima:
        if minimum.charge != charge:
            continue

        if minimum.state_minimum.hard_warnings:
            continue

        counts[
            (
                minimum.spin,
                minimum.multiplicity,
            )
        ] += 1

    ranked = sorted(
        (
            (
                spin,
                multiplicity,
                count,
            )
            for (
                spin,
                multiplicity,
            ), count in counts.items()
        ),
        key=lambda item: (
            -item[2],
            item[0],
        ),
    )

    return tuple(
        ranked[:maximum]
    )


def _source_for_functional(
    state_minima: tuple[StateMinimum, ...],
    *,
    charge: int,
    spin: int,
    functional: str,
) -> StateMinimum | None:
    """Choose the preferred reliable fast-grid geometry source."""
    matching = [
        minimum
        for minimum in state_minima
        if (
            minimum.point.charge == charge
            and minimum.point.spin == spin
            and minimum.point.functional == functional
            and not minimum.hard_warnings
        )
    ]

    if not matching:
        return None

    return min(
        matching,
        key=lambda minimum: (
            BASIS_PRIORITY.get(
                minimum.point.basis.lower(),
                99,
            ),
            minimum.point.energy_hartree,
            minimum.point.basis.lower(),
        ),
    )


def select_qzvpd_candidates(
    analysis: FastGridAnalysis,
    *,
    schema: SchemaFSpec = SCHEMA_F,
) -> tuple[QZVPDCandidate, ...]:
    """Select the Schema F QZVPD refinement candidates."""
    maximum = (
        schema.refinement
        .max_spins_per_charge
    )

    window = (
        schema.refinement
        .window_angstrom
    )

    qzvpd_basis = (
        schema.refinement.basis
    )

    candidates: list[
        QZVPDCandidate
    ] = []

    for charge in (
        0,
        -1,
    ):
        selected_spins = (
            _selected_spin_counts(
                analysis,
                charge=charge,
                maximum=maximum,
            )
        )

        for (
            spin,
            multiplicity,
            win_count,
        ) in selected_spins:
            functionals = sorted(
                {
                    minimum.point.functional
                    for minimum
                    in analysis.state_minima
                    if (
                        minimum.point.charge == charge
                        and minimum.point.spin == spin
                        and not minimum.hard_warnings
                    )
                }
            )

            for functional in functionals:
                source = _source_for_functional(
                    analysis.state_minima,
                    charge=charge,
                    spin=spin,
                    functional=functional,
                )

                if source is None:
                    continue

                center = (
                    source.point
                    .bond_length_angstrom
                )

                candidates.append(
                    QZVPDCandidate(
                        molecule=source.point.molecule,
                        charge=charge,
                        spin=spin,
                        multiplicity=multiplicity,
                        functional=functional,
                        qzvpd_basis=qzvpd_basis,
                        r_center_angstrom=center,
                        r_min_angstrom=round(
                            max(
                                0.2,
                                center - window,
                            ),
                            6,
                        ),
                        r_max_angstrom=round(
                            center + window,
                            6,
                        ),
                        source_basis=(
                            source.point.basis
                        ),
                        source_energy_hartree=(
                            source.point
                            .energy_hartree
                        ),
                        method_minimum_count_for_spin=(
                            win_count
                        ),
                        source_warnings=(
                            source.warnings
                        ),
                    )
                )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.charge,
                candidate.spin,
                candidate.functional,
            ),
        )
    )