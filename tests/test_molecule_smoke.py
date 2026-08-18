"""Tests for the synthetic molecule smoke test."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.molecule_smoke import (
    SMOKE_NOTICE,
    run_molecule_smoke_test,
    synthetic_electron_count,
    synthetic_single_point,
)
from diatomic_ea.single_point import SinglePointTask
from diatomic_ea.states import ChargeState


def make_task(
    *,
    charge: ChargeState,
    r: float,
) -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        charge=charge,
        spin=(
            0
            if charge
            is ChargeState.NEUTRAL
            else 1
        ),
        functional="PBE",
        basis="def2-svp",
        bond_length_angstrom=r,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    )


def test_synthetic_electron_counts() -> None:
    neutral = synthetic_electron_count(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        charge=ChargeState.NEUTRAL,
        basis="def2-svp",
        bond_length_angstrom=1.5,
        max_memory_mb=4000,
    )

    anion = synthetic_electron_count(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        charge=ChargeState.ANION,
        basis="def2-svp",
        bond_length_angstrom=1.5,
        max_memory_mb=4000,
    )

    assert neutral == 10
    assert anion == 11


def test_synthetic_curve_has_interior_minimum() -> None:
    center = synthetic_single_point(
        make_task(
            charge=ChargeState.NEUTRAL,
            r=1.525,
        )
    )

    side = synthetic_single_point(
        make_task(
            charge=ChargeState.NEUTRAL,
            r=1.50,
        )
    )

    assert (
        center.energy_hartree
        < side.energy_hartree
    )


def test_synthetic_anion_is_lower() -> None:
    neutral = synthetic_single_point(
        make_task(
            charge=ChargeState.NEUTRAL,
            r=1.525,
        )
    )

    anion = synthetic_single_point(
        make_task(
            charge=ChargeState.ANION,
            r=1.525,
        )
    )

    assert (
        anion.energy_hartree
        < neutral.energy_hartree
    )


def test_smoke_notice_is_unambiguous() -> None:
    assert (
        "NOT A SCIENTIFIC"
        in SMOKE_NOTICE
    )

    assert (
        "SYNTHETIC"
        in SMOKE_NOTICE
    )


def test_smoke_wrapper_marks_outputs(
    tmp_path,
) -> None:
    run_dir = (
        tmp_path
        / "HF"
        / "fake-run"
    )

    final_dir = (
        run_dir
        / "04_final"
    )

    final_dir.mkdir(
        parents=True
    )

    final_csv = (
        final_dir
        / "final_result.csv"
    )

    final_csv.write_text(
        "synthetic",
        encoding="utf-8",
    )

    manifest = (
        final_dir
        / "manifest.json"
    )

    manifest.write_text(
        "{}",
        encoding="utf-8",
    )

    fake_result = SimpleNamespace(
        paths=SimpleNamespace(
            run_dir=run_dir,
            final_dir=final_dir,
            final_result_csv=final_csv,
            manifest_json=manifest,
        ),
        estimate=SimpleNamespace(
            predicted_ea_ev=2.5
        ),
        fast_plan=SimpleNamespace(
            task_count=120
        ),
        qzvpd_plan=SimpleNamespace(
            task_count=168
        ),
    )

    with patch(
        "diatomic_ea.molecule_smoke.run_schema_f_pipeline",
        return_value=fake_result,
    ):
        report = run_molecule_smoke_test(
            output_root=tmp_path,
            workers=1,
        )

    run_marker = (
        report.run_dir
        / "SMOKE_TEST_ONLY.txt"
    )

    final_marker = (
        final_dir
        / "SMOKE_TEST_ONLY.txt"
    )

    assert run_marker.exists()
    assert final_marker.exists()

    assert (
        "NOT A SCIENTIFIC"
        in run_marker.read_text(
            encoding="utf-8"
        )
    )

    stored_manifest = json.loads(
        manifest.read_text(
            encoding="utf-8"
        )
    )

    assert (
        stored_manifest[
            "smoke_test"
        ][
            "synthetic"
        ]
        is True
    )

    assert (
        stored_manifest[
            "smoke_test"
        ][
            "scientific_prediction"
        ]
        is False
    )

    assert (
        report.synthetic_predicted_ea_ev
        == pytest.approx(2.5)
    )