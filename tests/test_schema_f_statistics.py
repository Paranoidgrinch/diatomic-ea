"""Tests for calibrated Paper-1 Schema F statistics."""

import math
from types import SimpleNamespace

import pytest

from diatomic_ea.schema_f_statistics import (
    SCHEMA_F_CALIBRATION,
    estimate_schema_f,
    estimate_schema_f_from_values,
)


def values():
    return {
        "PBE": 1.00,
        "B3LYP": 1.10,
        "PBE0": 1.20,
        "TPSSh": 1.30,
    }


def test_calibration_constants_are_frozen() -> None:
    calibration = SCHEMA_F_CALIBRATION

    assert calibration.central_slope == 1.0
    assert calibration.bias_correction_ev == 0.0825
    assert calibration.scale_intercept == -2.4496
    assert (
        calibration.scale_half_range_coefficient
        == 3.0511
    )

    assert calibration.quantile(80) == 1.334
    assert calibration.quantile(90) == 1.874
    assert calibration.quantile(95) == 2.186


def test_schema_f_median_and_half_range() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev=values(),
    )

    assert result.median_qz_ev == pytest.approx(
        1.15
    )

    assert result.half_range_qz_ev == pytest.approx(
        0.15
    )


def test_schema_f_bias_correction() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev=values(),
    )

    assert result.predicted_ea_ev == pytest.approx(
        1.15 + 0.0825
    )


def test_schema_f_heteroscedastic_scale() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev=values(),
    )

    expected = math.exp(
        -2.4496
        + 3.0511 * 0.15
    )

    assert result.scale_ev == pytest.approx(
        expected
    )


def test_schema_f_95_interval() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev=values(),
    )

    interval = result.interval(95)

    expected_half_width = (
        2.186 * result.scale_ev
    )

    assert interval.half_width_ev == pytest.approx(
        expected_half_width
    )

    assert interval.lower_ev == pytest.approx(
        result.predicted_ea_ev
        - expected_half_width
    )

    assert interval.upper_ev == pytest.approx(
        result.predicted_ea_ev
        + expected_half_width
    )


def test_all_three_intervals_are_present() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev=values(),
    )

    assert [
        interval.confidence_percent
        for interval in result.intervals
    ] == [
        80,
        90,
        95,
    ]


def test_missing_functional_is_rejected() -> None:
    incomplete = values()
    incomplete.pop("TPSSh")

    with pytest.raises(
        ValueError,
        match="Missing: TPSSh",
    ):
        estimate_schema_f_from_values(
            molecule="AlO",
            functional_eas_ev=incomplete,
        )


def test_unknown_functional_is_rejected() -> None:
    bad = values()
    bad["M06"] = 1.0

    with pytest.raises(
        ValueError,
        match="Unsupported functional",
    ):
        estimate_schema_f_from_values(
            molecule="AlO",
            functional_eas_ev=bad,
        )


def test_functional_names_are_case_insensitive() -> None:
    result = estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev={
            "pbe": 1.0,
            "b3lyp": 1.1,
            "pbe0": 1.2,
            "tpssh": 1.3,
        },
    )

    assert result.functional_count == 4

    assert result.functional_eas_ev[0][0] == "PBE"
    assert result.functional_eas_ev[-1][0] == "TPSSh"


def test_hard_warning_qzvpd_result_is_excluded() -> None:
    analysis = SimpleNamespace(
        functional_eas=(
            SimpleNamespace(
                molecule="AlO",
                functional="PBE",
                ea_ev=1.0,
                recommended_for_summary=True,
            ),
            SimpleNamespace(
                molecule="AlO",
                functional="B3LYP",
                ea_ev=1.1,
                recommended_for_summary=True,
            ),
            SimpleNamespace(
                molecule="AlO",
                functional="PBE0",
                ea_ev=1.2,
                recommended_for_summary=True,
            ),
            SimpleNamespace(
                molecule="AlO",
                functional="TPSSh",
                ea_ev=1.3,
                recommended_for_summary=False,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="Missing: TPSSh",
    ):
        estimate_schema_f(
            analysis
        )


def test_complete_qzvpd_analysis_is_accepted() -> None:
    analysis = SimpleNamespace(
        functional_eas=tuple(
            SimpleNamespace(
                molecule="AlO",
                functional=functional,
                ea_ev=value,
                recommended_for_summary=True,
            )
            for functional, value in values().items()
        )
    )

    result = estimate_schema_f(
        analysis
    )

    assert result.molecule == "AlO"
    assert result.functional_count == 4