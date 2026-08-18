"""Calibrated Paper-1 Schema F statistical EA estimator.

Schema F uses the four hard-warning-free QZVPD electron affinities:

    PBE, B3LYP, PBE0, TPSSh

The calibrated model is

    m_QZ = median(q_f)
    h_QZ = (max(q_f) - min(q_f)) / 2

    EA_pred = m_QZ + 0.0825 eV

    s(h_QZ) = exp(-2.4496 + 3.0511 * h_QZ)

with scaled-conformal prediction intervals

    EA_pred +/- q_c * s(h_QZ)

where

    q_80 = 1.334
    q_90 = 1.874
    q_95 = 2.186

The calibration slope of the central prediction is fixed at one.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from diatomic_ea.qzvpd_analysis import QZVPDAnalysis
from diatomic_ea.schema_f import SCHEMA_F


@dataclass(frozen=True, slots=True)
class SchemaFCalibration:
    """Frozen calibration constants for Paper-1 Schema F."""

    model_id: str
    central_slope: float
    bias_correction_ev: float
    scale_intercept: float
    scale_half_range_coefficient: float
    conformal_quantiles: tuple[
        tuple[int, float],
        ...,
    ]

    def quantile(
        self,
        confidence_percent: int,
    ) -> float:
        """Return the calibrated conformal quantile."""
        for confidence, quantile in self.conformal_quantiles:
            if confidence == confidence_percent:
                return quantile

        raise ValueError(
            "Unsupported Schema F confidence level: "
            f"{confidence_percent}%."
        )


SCHEMA_F_CALIBRATION = SchemaFCalibration(
    model_id="schema-f-paper1-v1",
    central_slope=1.0,
    bias_correction_ev=0.0825,
    scale_intercept=-2.4496,
    scale_half_range_coefficient=3.0511,
    conformal_quantiles=(
        (80, 1.334),
        (90, 1.874),
        (95, 2.186),
    ),
)


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """One calibrated Schema F prediction interval."""

    confidence_percent: int
    conformal_quantile: float
    half_width_ev: float
    lower_ev: float
    upper_ev: float


@dataclass(frozen=True, slots=True)
class SchemaFEstimate:
    """Final calibrated Schema F electron-affinity estimate."""

    molecule: str
    model_id: str
    functional_eas_ev: tuple[
        tuple[str, float],
        ...,
    ]
    median_qz_ev: float
    half_range_qz_ev: float
    bias_correction_ev: float
    predicted_ea_ev: float
    scale_ev: float
    intervals: tuple[
        PredictionInterval,
        ...,
    ]

    @property
    def functional_count(self) -> int:
        return len(
            self.functional_eas_ev
        )

    def interval(
        self,
        confidence_percent: int,
    ) -> PredictionInterval:
        """Return one confidence interval."""
        for interval in self.intervals:
            if (
                interval.confidence_percent
                == confidence_percent
            ):
                return interval

        raise ValueError(
            "Prediction interval not available for "
            f"{confidence_percent}%."
        )


def _canonical_functional_values(
    values: Mapping[str, float],
) -> tuple[
    tuple[str, float],
    ...,
]:
    expected = tuple(
        SCHEMA_F.functionals
    )

    expected_by_key = {
        name.casefold(): name
        for name in expected
    }

    normalized: dict[
        str,
        float,
    ] = {}

    for supplied_name, supplied_value in values.items():
        key = supplied_name.strip().casefold()

        canonical = expected_by_key.get(
            key
        )

        if canonical is None:
            raise ValueError(
                "Unsupported functional for Schema F: "
                f"{supplied_name!r}."
            )

        if canonical in normalized:
            raise ValueError(
                "Duplicate Schema F functional: "
                f"{canonical}."
            )

        value = float(
            supplied_value
        )

        if not math.isfinite(value):
            raise ValueError(
                "Schema F functional EA must be finite: "
                f"{canonical}={value!r}."
            )

        normalized[
            canonical
        ] = value

    missing = [
        functional
        for functional in expected
        if functional not in normalized
    ]

    if missing:
        raise ValueError(
            "Strict Schema F requires all four "
            "hard-warning-free QZVPD functionals. "
            "Missing: "
            + ", ".join(missing)
        )

    return tuple(
        (
            functional,
            normalized[functional],
        )
        for functional in expected
    )


def estimate_schema_f_from_values(
    *,
    molecule: str,
    functional_eas_ev: Mapping[str, float],
    calibration: SchemaFCalibration = SCHEMA_F_CALIBRATION,
) -> SchemaFEstimate:
    """Evaluate calibrated Schema F from four QZVPD EAs."""
    if not molecule.strip():
        raise ValueError(
            "molecule must not be empty."
        )

    ordered = _canonical_functional_values(
        functional_eas_ev
    )

    values = [
        value
        for _, value in ordered
    ]

    median_qz = float(
        statistics.median(values)
    )

    half_range_qz = (
        max(values) - min(values)
    ) / 2.0

    predicted = (
        calibration.central_slope
        * median_qz
        + calibration.bias_correction_ev
    )

    scale = math.exp(
        calibration.scale_intercept
        + (
            calibration
            .scale_half_range_coefficient
            * half_range_qz
        )
    )

    intervals: list[
        PredictionInterval
    ] = []

    for (
        confidence,
        quantile,
    ) in calibration.conformal_quantiles:
        half_width = (
            quantile * scale
        )

        intervals.append(
            PredictionInterval(
                confidence_percent=confidence,
                conformal_quantile=quantile,
                half_width_ev=half_width,
                lower_ev=(
                    predicted - half_width
                ),
                upper_ev=(
                    predicted + half_width
                ),
            )
        )

    return SchemaFEstimate(
        molecule=molecule.strip(),
        model_id=calibration.model_id,
        functional_eas_ev=ordered,
        median_qz_ev=median_qz,
        half_range_qz_ev=half_range_qz,
        bias_correction_ev=(
            calibration.bias_correction_ev
        ),
        predicted_ea_ev=predicted,
        scale_ev=scale,
        intervals=tuple(intervals),
    )


def estimate_schema_f(
    analysis: QZVPDAnalysis,
    *,
    calibration: SchemaFCalibration = SCHEMA_F_CALIBRATION,
) -> SchemaFEstimate:
    """Evaluate strict Schema F from a QZVPD analysis."""
    reliable = [
        result
        for result in analysis.functional_eas
        if result.recommended_for_summary
    ]

    if not reliable:
        raise ValueError(
            "No hard-warning-free QZVPD electron "
            "affinities are available for Schema F."
        )

    molecules = {
        result.molecule
        for result in reliable
    }

    if len(molecules) != 1:
        raise ValueError(
            "Schema F analysis must contain exactly "
            "one molecule."
        )

    values: dict[
        str,
        float,
    ] = {}

    for result in reliable:
        if result.functional in values:
            raise ValueError(
                "Duplicate QZVPD functional result: "
                f"{result.functional}."
            )

        values[
            result.functional
        ] = result.ea_ev

    return estimate_schema_f_from_values(
        molecule=next(
            iter(molecules)
        ),
        functional_eas_ev=values,
        calibration=calibration,
    )