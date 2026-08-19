"""Tests for full Schema F production-run planning."""

import json

import pytest

from diatomic_ea.grid import (
    BondGrid,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.production_plan import (
    PRODUCTION_PLAN_VERSION,
    ProductionRunRequest,
    build_production_plan,
    estimate_wall_time_hours,
    write_production_plan,
)
from diatomic_ea.resources import (
    CpuResources,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
)


def provenance():
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
                "a" * 64
            ),
        },
        "compatibility": {
            "verified": True,
        },
    }


def request(
    *,
    run_id="test-run",
):
    return ProductionRunRequest(
        molecule=DiatomicMolecule(
            "O",
            "H",
        ),
        minimum_angstrom=0.75,
        maximum_angstrom=1.35,
        spin_max=3,
        workers=4,
        threads_per_worker=1,
        run_id=run_id,
    )


def resources():
    return CpuResources(
        physical_cores=8,
        logical_cores=16,
        recommended_workers=7,
    )


def test_full_oh_fast_task_count(tmp_path) -> None:
    (
        plan,
        fast_plan,
        _paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    expected_points = len(
        BondGrid(
            minimum_angstrom=0.75,
            maximum_angstrom=1.35,
            step_angstrom=(
                SCHEMA_F
                .fast_grid
                .step_angstrom
            ),
        ).values
    )

    expected_states = 4

    expected_tasks = (
        expected_points
        * expected_states
        * len(
            SCHEMA_F.functionals
        )
        * len(
            SCHEMA_F.fast_bases
        )
    )

    assert (
        fast_plan.task_count
        == expected_tasks
    )

    assert (
        plan.fast_grid_tasks
        == expected_tasks
    )

    assert (
        plan.fast_grid_points
        == expected_points
    )

    assert plan.neutral_spins == (
        1,
        3,
    )

    assert plan.anion_spins == (
        0,
        2,
    )


def test_qzvpd_upper_bound_is_exact_formula(
    tmp_path,
) -> None:
    (
        plan,
        _fast_plan,
        _paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    expected_candidates = (
        len(
            SCHEMA_F.functionals
        )
        * 2
        * SCHEMA_F
        .refinement
        .max_spins_per_charge
    )

    assert (
        plan.qzvpd_candidate_upper_bound
        == expected_candidates
    )

    assert (
        plan.qzvpd_points_per_candidate
        == 21
    )

    assert (
        plan.qzvpd_task_upper_bound
        == expected_candidates
        * 21
    )

    assert (
        plan.total_task_upper_bound
        == (
            plan.fast_grid_tasks
            + plan.qzvpd_task_upper_bound
        )
    )


def test_memory_recommendation_limits_workers(
    tmp_path,
) -> None:
    (
        plan,
        _fast_plan,
        _paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=16000,
    )

    assert (
        plan.memory_recommended_workers
        == 2
    )

    assert (
        plan.recommended_workers
        == 2
    )


def test_plan_is_marked_not_started_and_resumable(
    tmp_path,
) -> None:
    (
        plan,
        _fast_plan,
        _paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    assert (
        plan.plan_version
        == PRODUCTION_PLAN_VERSION
    )

    assert (
        plan.schema_id
        == SCHEMA_F.schema_id
    )

    assert (
        plan.scientific_execution_started
        is False
    )

    assert plan.resumable is True

    assert (
        plan.provenance_verified
        is True
    )


def test_plan_file_can_be_reopened_idempotently(
    tmp_path,
) -> None:
    (
        plan,
        _fast_plan,
        paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    first = write_production_plan(
        plan,
        paths,
    )

    second = write_production_plan(
        plan,
        paths,
    )

    assert first == second

    payload = json.loads(
        first.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "run_id"
        ]
        == "test-run"
    )

    assert (
        payload[
            "scientific_execution_started"
        ]
        is False
    )


def test_same_run_id_rejects_changed_plan(
    tmp_path,
) -> None:
    (
        first_plan,
        _fast_plan,
        paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    write_production_plan(
        first_plan,
        paths,
    )

    changed_request = (
        ProductionRunRequest(
            molecule=DiatomicMolecule(
                "O",
                "H",
            ),
            minimum_angstrom=0.80,
            maximum_angstrom=1.35,
            spin_max=3,
            workers=4,
            threads_per_worker=1,
            run_id="test-run",
        )
    )

    (
        changed,
        _changed_fast,
        _changed_paths,
    ) = build_production_plan(
        changed_request,
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    with pytest.raises(
        RuntimeError,
        match="different production plan",
    ):
        write_production_plan(
            changed,
            paths,
        )


def test_wall_time_estimate_uses_worker_parallelism(
    tmp_path,
) -> None:
    (
        plan,
        _fast_plan,
        _paths,
    ) = build_production_plan(
        request(),
        output_root=tmp_path,
        provenance=provenance(),
        neutral_electrons=9,
        anion_electrons=10,
        resources=resources(),
        total_memory_mb=32768,
    )

    fast_h, qz_h, total_h = (
        estimate_wall_time_hours(
            plan=plan,
            fast_seconds_per_task=4.0,
            qzvpd_seconds_per_task=20.0,
        )
    )

    assert fast_h > 0
    assert qz_h > 0

    assert total_h == pytest.approx(
        fast_h
        + qz_h
    )
