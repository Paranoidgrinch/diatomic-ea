"""Tests for final-result presentation."""

import csv

import pytest

from diatomic_ea.gui_results import (
    ResultReadError,
    format_energy,
    format_interval,
    read_calculation_result,
)


FIELDS = [
    "molecule",
    "model_id",
    "n_functionals",
    "ea_pbe_eV",
    "ea_b3lyp_eV",
    "ea_pbe0_eV",
    "ea_tpssh_eV",
    "median_qz_eV",
    "half_range_qz_eV",
    "bias_correction_eV",
    "predicted_ea_eV",
    "scale_eV",
    "pi80_half_width_eV",
    "pi80_lower_eV",
    "pi80_upper_eV",
    "pi90_half_width_eV",
    "pi90_lower_eV",
    "pi90_upper_eV",
    "pi95_half_width_eV",
    "pi95_lower_eV",
    "pi95_upper_eV",
]


def result_row():
    return {
        "molecule": "OH",
        "model_id": "internal-only",
        "n_functionals": "4",
        "ea_pbe_eV": "1.9829184758392013",
        "ea_b3lyp_eV": "1.8566987970372117",
        "ea_pbe0_eV": "1.6131671704161916",
        "ea_tpssh_eV": "1.6239377722463588",
        "median_qz_eV": "1.7403182846417853",
        "half_range_qz_eV": "0.18487565271150486",
        "bias_correction_eV": "0.0825",
        "predicted_ea_eV": "1.8228182846417853",
        "scale_eV": "0.1517492341243265",
        "pi80_half_width_eV": "0.20243347832185157",
        "pi80_lower_eV": "1.6203848063199338",
        "pi80_upper_eV": "2.025251762963637",
        "pi90_half_width_eV": "0.2843780647489879",
        "pi90_lower_eV": "1.5384402198927973",
        "pi90_upper_eV": "2.1071963493907733",
        "pi95_half_width_eV": "0.33172382579577775",
        "pi95_lower_eV": "1.4910944588460076",
        "pi95_upper_eV": "2.1545421104375633",
    }


def write_result(
    path,
    row=None,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        writer.writerow(
            result_row()
            if row is None
            else row
        )


def test_realistic_result_is_read(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "final_result.csv"
    )

    write_result(
        path
    )

    result = read_calculation_result(
        path
    )

    assert result.molecule == "OH"

    assert (
        result.predicted_ea_ev
        == pytest.approx(
            1.8228182846417853
        )
    )

    assert (
        format_energy(
            result.predicted_ea_ev
        )
        == "1.8228 eV"
    )

    assert (
        format_interval(
            result.interval(
                90
            )
        )
        == "1.5384 to 2.1072 eV"
    )


def test_incomplete_result_is_rejected(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "final_result.csv"
    )

    row = result_row()

    row[
        "n_functionals"
    ] = "3"

    write_result(
        path,
        row,
    )

    with pytest.raises(
        ResultReadError,
        match="four functionals",
    ):
        read_calculation_result(
            path
        )
