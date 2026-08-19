"""Final-result presentation helpers for the desktop application."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


class ResultReadError(ValueError):
    """Raised when a completed result cannot be presented safely."""


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    confidence_percent: int
    lower_ev: float
    upper_ev: float
    half_width_ev: float


@dataclass(frozen=True, slots=True)
class CalculationResultSummary:
    molecule: str
    predicted_ea_ev: float
    median_ea_ev: float
    functional_half_range_ev: float
    model_scale_ev: float
    functional_eas_ev: tuple[tuple[str, float], ...]
    intervals: tuple[PredictionInterval, ...]
    source_path: str

    def interval(
        self,
        confidence_percent: int,
    ) -> PredictionInterval:
        for interval in self.intervals:
            if interval.confidence_percent == confidence_percent:
                return interval

        raise KeyError(
            confidence_percent
        )


def _required_text(
    row: dict[str, str],
    name: str,
) -> str:
    value = str(
        row.get(
            name,
            "",
        )
    ).strip()

    if not value:
        raise ResultReadError(
            f"Result is missing {name!r}."
        )

    return value


def _required_int(
    row: dict[str, str],
    name: str,
) -> int:
    try:
        return int(
            _required_text(
                row,
                name,
            )
        )
    except ValueError as exc:
        raise ResultReadError(
            f"Result field {name!r} is invalid."
        ) from exc


def _required_float(
    row: dict[str, str],
    name: str,
) -> float:
    try:
        value = float(
            _required_text(
                row,
                name,
            )
        )
    except ValueError as exc:
        raise ResultReadError(
            f"Result field {name!r} is invalid."
        ) from exc

    if not math.isfinite(
        value
    ):
        raise ResultReadError(
            f"Result field {name!r} is not finite."
        )

    return value


def read_calculation_result(
    path: str | Path,
) -> CalculationResultSummary:
    source = Path(
        path
    )

    try:
        with source.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(
                csv.DictReader(
                    handle
                )
            )
    except OSError as exc:
        raise ResultReadError(
            f"Could not read result file: {source}"
        ) from exc

    if len(rows) != 1:
        raise ResultReadError(
            "Final result must contain exactly one row."
        )

    row = rows[0]

    if (
        _required_int(
            row,
            "n_functionals",
        )
        != 4
    ):
        raise ResultReadError(
            "A complete standard calculation requires all four functionals."
        )

    functional_values = (
        (
            "PBE",
            _required_float(
                row,
                "ea_pbe_eV",
            ),
        ),
        (
            "B3LYP",
            _required_float(
                row,
                "ea_b3lyp_eV",
            ),
        ),
        (
            "PBE0",
            _required_float(
                row,
                "ea_pbe0_eV",
            ),
        ),
        (
            "TPSSh",
            _required_float(
                row,
                "ea_tpssh_eV",
            ),
        ),
    )

    intervals = tuple(
        PredictionInterval(
            confidence_percent=confidence,
            lower_ev=_required_float(
                row,
                f"pi{confidence}_lower_eV",
            ),
            upper_ev=_required_float(
                row,
                f"pi{confidence}_upper_eV",
            ),
            half_width_ev=_required_float(
                row,
                f"pi{confidence}_half_width_eV",
            ),
        )
        for confidence in (
            80,
            90,
            95,
        )
    )

    return CalculationResultSummary(
        molecule=_required_text(
            row,
            "molecule",
        ),
        predicted_ea_ev=_required_float(
            row,
            "predicted_ea_eV",
        ),
        median_ea_ev=_required_float(
            row,
            "median_qz_eV",
        ),
        functional_half_range_ev=_required_float(
            row,
            "half_range_qz_eV",
        ),
        model_scale_ev=_required_float(
            row,
            "scale_eV",
        ),
        functional_eas_ev=functional_values,
        intervals=intervals,
        source_path=str(
            source.resolve()
        ),
    )


def format_energy(
    value_ev: float,
) -> str:
    return f"{value_ev:.4f} eV"


def format_interval(
    interval: PredictionInterval,
) -> str:
    return (
        f"{interval.lower_ev:.4f} to "
        f"{interval.upper_ev:.4f} eV"
    )
