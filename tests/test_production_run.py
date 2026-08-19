"""Tests for the safe resumable production launcher."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.pipeline import (
    PipelineRequest,
)
from diatomic_ea.production_plan import (
    PRODUCTION_PLAN_FILENAME,
    PRODUCTION_PLAN_VERSION,
)
from diatomic_ea.production_run import (
    PRODUCTION_LOCK_FILENAME,
    ProductionLaunchError,
    _atomic_write_json,
    _production_lock,
    validate_prepared_production,
)


HASH = (
    "0123456789abcdef"
    * 4
)


def provenance(
    *,
    wheel_hash=HASH,
):
    return {
        "backend": "wsl",
        "ready": True,
        "compute": {
            "distribution": (
                "Ubuntu-24.04"
            ),
            "python_version": (
                "3.12.3"
            ),
            "pyscf_version": (
                "2.13.0"
            ),
            "worker_wheel_sha256": (
                wheel_hash
            ),
        },
        "compatibility": {
            "verified": True,
        },
    }


def plan_payload(
    run_directory: Path,
):
    return {
        "plan_version": (
            PRODUCTION_PLAN_VERSION
        ),
        "schema_id": "schema-f-v1",
        "molecule": "OH",
        "run_id": "test-run",
        "run_directory": str(
            run_directory.resolve()
        ),
        "minimum_angstrom": 0.75,
        "maximum_angstrom": 1.35,
        "spin_max": 3,
        "neutral_electrons": 9,
        "anion_electrons": 10,
        "neutral_spins": [
            1,
            3,
        ],
        "anion_spins": [
            0,
            2,
        ],
        "functionals": [
            "PBE",
            "B3LYP",
            "PBE0",
            "TPSSh",
        ],
        "fast_bases": [
            "def2-svp",
            "def2-tzvp",
            "def2-tzvpp",
            "def2-svpd",
            "def2-tzvpd",
        ],
        "fast_grid_step_angstrom": 0.025,
        "fast_grid_points": 25,
        "fast_grid_tasks": 2000,
        "qzvpd_basis": "def2-qzvpd",
        "qzvpd_window_angstrom": 0.10,
        "qzvpd_step_angstrom": 0.01,
        "qzvpd_points_per_candidate": 21,
        "qzvpd_candidate_upper_bound": 16,
        "qzvpd_task_upper_bound": 336,
        "total_task_upper_bound": 2336,
        "physical_cores": 8,
        "logical_cores": 16,
        "detected_memory_mb": 32768,
        "requested_workers": 4,
        "cpu_recommended_workers": 7,
        "memory_recommended_workers": 6,
        "recommended_workers": 6,
        "threads_per_worker": 1,
        "fast_memory_per_worker_mb": 2000,
        "qzvpd_memory_per_worker_mb": 4000,
        "qzvpd_concurrent_memory_upper_mb": 16000,
        "compute_backend": "wsl",
        "compute_distribution": (
            "Ubuntu-24.04"
        ),
        "compute_python": "3.12.3",
        "pyscf_version": "2.13.0",
        "worker_wheel_sha256": HASH,
        "provenance_verified": True,
        "scientific_execution_started": False,
        "resumable": True,
    }


def create_plan(
    tmp_path: Path,
):
    run_directory = (
        tmp_path
        / "OH"
        / "test-run"
    )

    run_directory.mkdir(
        parents=True,
    )

    plan = (
        run_directory
        / PRODUCTION_PLAN_FILENAME
    )

    _atomic_write_json(
        plan,
        plan_payload(
            run_directory
        ),
    )

    return (
        plan,
        run_directory,
    )


def test_valid_plan_reconstructs_pipeline_request(
    tmp_path,
) -> None:
    (
        plan,
        run_directory,
    ) = create_plan(
        tmp_path
    )

    validated = (
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=provenance(),
        )
    )

    assert (
        validated.molecule
        == DiatomicMolecule(
            "O",
            "H",
        )
    )

    assert isinstance(
        validated.request,
        PipelineRequest,
    )

    assert (
        validated.request.run_id
        == "test-run"
    )

    assert (
        validated.request.workers
        == 4
    )

    assert (
        validated.output_root
        == tmp_path.resolve()
    )

    assert (
        validated.run_directory
        == run_directory.resolve()
    )


def test_molecule_mismatch_is_rejected(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    with pytest.raises(
        ProductionLaunchError,
        match="planned molecule",
    ):
        validate_prepared_production(
            plan,
            atom_a="N",
            atom_b="H",
            provenance=provenance(),
        )


def test_worker_hash_change_is_rejected(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    with pytest.raises(
        ProductionLaunchError,
        match="worker wheel SHA-256 changed",
    ):
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=provenance(
                wheel_hash=(
                    "f" * 64
                )
            ),
        )


def test_unverified_current_provenance_is_rejected(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    broken = provenance()

    broken[
        "compatibility"
    ][
        "verified"
    ] = False

    with pytest.raises(
        ProductionLaunchError,
        match="not verified",
    ):
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=broken,
        )


def test_plan_must_live_in_recorded_directory(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    payload = json.loads(
        plan.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "run_directory"
    ] = str(
        tmp_path
        / "wrong"
    )

    _atomic_write_json(
        plan,
        payload,
    )

    with pytest.raises(
        ProductionLaunchError,
        match="recorded run directory",
    ):
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=provenance(),
        )


def test_atomic_status_write_replaces_file(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "status.json"
    )

    _atomic_write_json(
        path,
        {
            "state": "running",
        },
    )

    _atomic_write_json(
        path,
        {
            "state": "completed",
        },
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert payload == {
        "state": "completed"
    }

    assert not (
        tmp_path
        / "status.json.tmp"
    ).exists()


def test_production_lock_prevents_second_launcher(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    validated = (
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=provenance(),
        )
    )

    with _production_lock(
        validated,
        recover_stale_lock=False,
    ):
        assert (
            validated.lock_path.name
            == PRODUCTION_LOCK_FILENAME
        )

        assert (
            validated.lock_path.exists()
        )

        with pytest.raises(
            ProductionLaunchError,
            match="already locked",
        ):
            with _production_lock(
                validated,
                recover_stale_lock=False,
            ):
                pass

    assert not (
        validated.lock_path.exists()
    )


def test_explicit_stale_lock_recovery(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    validated = (
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
            provenance=provenance(),
        )
    )

    validated.lock_path.write_text(
        "stale",
        encoding="utf-8",
    )

    with _production_lock(
        validated,
        recover_stale_lock=True,
    ):
        assert (
            validated.lock_path.exists()
        )

    assert not (
        validated.lock_path.exists()
    )


def test_validation_collects_real_provenance_when_unsupplied(
    tmp_path,
) -> None:
    plan, _ = create_plan(
        tmp_path
    )

    with patch(
        "diatomic_ea.production_run.collect_compute_provenance",
        return_value=provenance(),
    ) as collector:
        validate_prepared_production(
            plan,
            atom_a="O",
            atom_b="H",
        )

    collector.assert_called_once_with()
