"""Tests for final Schema F result export."""

import csv

import pytest

from diatomic_ea.result_export import (
    FINAL_RESULT_COLUMNS,
    final_record_from_estimate,
    final_record_row,
    write_final_result_csv,
)
from diatomic_ea.schema_f_statistics import (
    estimate_schema_f_from_values,
)


def estimate():
    return estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev={
            "PBE": 1.0,
            "B3LYP": 1.1,
            "PBE0": 1.2,
            "TPSSh": 1.3,
        },
    )


def test_final_record_contains_functional_values() -> None:
    record = final_record_from_estimate(
        estimate()
    )

    assert record.molecule == "AlO"
    assert record.n_functionals == 4
    assert record.ea_pbe_ev == pytest.approx(1.0)
    assert record.ea_b3lyp_ev == pytest.approx(1.1)
    assert record.ea_pbe0_ev == pytest.approx(1.2)
    assert record.ea_tpssh_ev == pytest.approx(1.3)


def test_final_record_contains_prediction() -> None:
    result = estimate()

    record = final_record_from_estimate(
        result
    )

    assert record.median_qz_ev == pytest.approx(
        result.median_qz_ev
    )

    assert record.half_range_qz_ev == pytest.approx(
        result.half_range_qz_ev
    )

    assert record.predicted_ea_ev == pytest.approx(
        result.predicted_ea_ev
    )


def test_final_record_contains_all_intervals() -> None:
    result = estimate()

    record = final_record_from_estimate(
        result
    )

    assert record.pi80_lower_ev == pytest.approx(
        result.interval(80).lower_ev
    )

    assert record.pi90_lower_ev == pytest.approx(
        result.interval(90).lower_ev
    )

    assert record.pi95_lower_ev == pytest.approx(
        result.interval(95).lower_ev
    )


def test_final_row_uses_expected_columns() -> None:
    record = final_record_from_estimate(
        estimate()
    )

    row = final_record_row(
        record
    )

    assert tuple(row.keys()) == (
        FINAL_RESULT_COLUMNS
    )


def test_write_final_result_csv(
    tmp_path,
) -> None:
    record = final_record_from_estimate(
        estimate()
    )

    path = (
        tmp_path
        / "AlO"
        / "final_result.csv"
    )

    written = write_final_result_csv(
        path,
        record,
    )

    assert written == path
    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(
        reader.fieldnames or ()
    ) == FINAL_RESULT_COLUMNS

    assert len(rows) == 1

    assert rows[0]["molecule"] == "AlO"

    assert float(
        rows[0]["predicted_ea_eV"]
    ) == pytest.approx(
        record.predicted_ea_ev
    )


def test_write_replaces_existing_result(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "final_result.csv"
    )

    path.write_text(
        "old data",
        encoding="utf-8",
    )

    record = final_record_from_estimate(
        estimate()
    )

    write_final_result_csv(
        path,
        record,
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert "old data" not in text
    assert "predicted_ea_eV" in text