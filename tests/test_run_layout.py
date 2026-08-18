"""Tests for the standard DiatomicEA run layout."""

import re

import pytest

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.run_layout import (
    create_run_paths,
    new_run_id,
)


def test_explicit_run_layout(
    tmp_path,
) -> None:
    paths = create_run_paths(
        output_root=tmp_path,
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        run_id="test-run-001",
    )

    assert paths.run_id == "test-run-001"

    assert paths.run_dir == (
        tmp_path
        / "AlO"
        / "test-run-001"
    )

    assert (
        paths.fast_grid_dir.name
        == "01_fast_grid"
    )

    assert (
        paths.fast_analysis_dir.name
        == "02_fast_analysis"
    )

    assert (
        paths.qzvpd_dir.name
        == "03_qzvpd"
    )

    assert (
        paths.final_dir.name
        == "04_final"
    )

    assert paths.logs_dir.name == "logs"


def test_all_stage_directories_are_created(
    tmp_path,
) -> None:
    paths = create_run_paths(
        output_root=tmp_path,
        molecule=DiatomicMolecule(
            "Fe",
            "H",
        ),
        run_id="run-a",
    )

    assert paths.fast_grid_dir.is_dir()
    assert paths.fast_analysis_dir.is_dir()
    assert paths.qzvpd_dir.is_dir()
    assert paths.final_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_standard_output_files(
    tmp_path,
) -> None:
    paths = create_run_paths(
        output_root=tmp_path,
        molecule=DiatomicMolecule(
            "Mg",
            "O",
        ),
        run_id="run-b",
    )

    assert paths.fast_grid_csv == (
        paths.fast_grid_dir
        / "raw_results.csv"
    )

    assert paths.qzvpd_csv == (
        paths.qzvpd_dir
        / "raw_results.csv"
    )

    assert paths.final_result_csv == (
        paths.final_dir
        / "final_result.csv"
    )

    assert paths.manifest_json == (
        paths.final_dir
        / "manifest.json"
    )


def test_existing_run_can_be_reopened(
    tmp_path,
) -> None:
    first = create_run_paths(
        output_root=tmp_path,
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        run_id="resume-me",
    )

    marker = (
        first.fast_grid_dir
        / "marker.txt"
    )

    marker.write_text(
        "existing",
        encoding="utf-8",
    )

    second = create_run_paths(
        output_root=tmp_path,
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        run_id="resume-me",
    )

    assert first.run_dir == second.run_dir
    assert marker.exists()


def test_generated_run_id_is_sortable_and_unique() -> None:
    first = new_run_id()
    second = new_run_id()

    pattern = re.compile(
        r"^\d{8}T\d{6}Z-[0-9a-f]{8}$"
    )

    assert pattern.fullmatch(first)
    assert pattern.fullmatch(second)
    assert first != second


@pytest.mark.parametrize(
    "bad_run_id",
    [
        "../outside",
        "..",
        "run/test",
        "run\\test",
        "bad:name",
        "",
    ],
)
def test_unsafe_run_id_is_rejected(
    tmp_path,
    bad_run_id: str,
) -> None:
    with pytest.raises(ValueError):
        create_run_paths(
            output_root=tmp_path,
            molecule=DiatomicMolecule(
                "Al",
                "O",
            ),
            run_id=bad_run_id,
        )