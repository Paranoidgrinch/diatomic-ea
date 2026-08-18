"""Tests for DiatomicEA reproducibility manifests."""

import hashlib
import json

from diatomic_ea.manifest import (
    build_reproducibility_manifest,
    sha256_file,
    write_reproducibility_manifest,
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


def test_sha256_file(
    tmp_path,
) -> None:
    path = tmp_path / "data.txt"

    path.write_bytes(
        b"DiatomicEA"
    )

    expected = hashlib.sha256(
        b"DiatomicEA"
    ).hexdigest()

    assert sha256_file(path) == expected


def test_manifest_contains_schema_f_calibration(
    tmp_path,
) -> None:
    manifest = build_reproducibility_manifest(
        estimate=estimate(),
        fast_grid_task_count=100,
        qzvpd_task_count=84,
    )

    calibration = manifest[
        "schema_f_calibration"
    ]

    assert calibration[
        "bias_correction_eV"
    ] == 0.0825

    assert calibration[
        "scale_intercept"
    ] == -2.4496

    assert calibration[
        "scale_half_range_coefficient"
    ] == 3.0511

    assert calibration[
        "conformal_quantiles"
    ] == {
        "80": 1.334,
        "90": 1.874,
        "95": 2.186,
    }


def test_manifest_contains_workflow(
    tmp_path,
) -> None:
    manifest = build_reproducibility_manifest(
        estimate=estimate(),
        fast_grid_task_count=100,
        qzvpd_task_count=84,
    )

    workflow = manifest[
        "workflow"
    ]

    assert workflow[
        "schema_id"
    ] == "schema-f-v1"

    assert workflow[
        "functionals"
    ] == [
        "PBE",
        "B3LYP",
        "PBE0",
        "TPSSh",
    ]

    assert workflow[
        "qzvpd_refinement"
    ][
        "basis"
    ] == "def2-qzvpd"


def test_manifest_contains_final_result(
    tmp_path,
) -> None:
    result = estimate()

    manifest = build_reproducibility_manifest(
        estimate=result,
        fast_grid_task_count=100,
        qzvpd_task_count=84,
    )

    stored = manifest[
        "schema_f_result"
    ]

    assert stored[
        "predicted_ea_eV"
    ] == result.predicted_ea_ev

    assert stored[
        "functional_eas_eV"
    ] == {
        "PBE": 1.0,
        "B3LYP": 1.1,
        "PBE0": 1.2,
        "TPSSh": 1.3,
    }


def test_manifest_hashes_raw_files(
    tmp_path,
) -> None:
    fast = tmp_path / "fast.csv"
    qz = tmp_path / "qz.csv"

    fast.write_text(
        "fast-grid",
        encoding="utf-8",
    )

    qz.write_text(
        "qzvpd",
        encoding="utf-8",
    )

    manifest = build_reproducibility_manifest(
        estimate=estimate(),
        fast_grid_task_count=10,
        qzvpd_task_count=20,
        fast_grid_raw_csv=fast,
        qzvpd_raw_csv=qz,
    )

    assert manifest[
        "raw_files"
    ][
        "fast_grid"
    ][
        "sha256"
    ] == sha256_file(fast)

    assert manifest[
        "raw_files"
    ][
        "qzvpd"
    ][
        "sha256"
    ] == sha256_file(qz)


def test_missing_raw_file_is_recorded(
    tmp_path,
) -> None:
    missing = (
        tmp_path / "missing.csv"
    )

    manifest = build_reproducibility_manifest(
        estimate=estimate(),
        fast_grid_task_count=0,
        qzvpd_task_count=0,
        fast_grid_raw_csv=missing,
    )

    entry = manifest[
        "raw_files"
    ][
        "fast_grid"
    ]

    assert entry[
        "exists"
    ] is False

    assert entry[
        "sha256"
    ] is None


def test_write_manifest_is_valid_json(
    tmp_path,
) -> None:
    manifest = build_reproducibility_manifest(
        estimate=estimate(),
        fast_grid_task_count=100,
        qzvpd_task_count=84,
    )

    path = (
        tmp_path
        / "AlO"
        / "manifest.json"
    )

    write_reproducibility_manifest(
        path,
        manifest,
    )

    loaded = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded[
        "manifest_version"
    ] == 1

    assert loaded[
        "calculation"
    ][
        "molecule"
    ] == "AlO"

    assert loaded[
        "schema_f_result"
    ][
        "predicted_ea_eV"
    ] == manifest[
        "schema_f_result"
    ][
        "predicted_ea_eV"
    ]