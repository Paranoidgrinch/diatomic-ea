"""Tests for the reduced-cost real workflow canary."""

import json

from diatomic_ea.qzvpd import (
    QZVPDPlan,
)
from diatomic_ea.qzvpd_analysis import (
    QZVPDAnalysis,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
)
from diatomic_ea.schema_f_canary import (
    CANARY_ID,
    CANARY_SCHEMA,
    CANARY_WARNING,
    build_canary_fast_plan,
    build_canary_report_payload,
    write_canary_report,
)
from diatomic_ea.schema_f_statistics import (
    estimate_schema_f_from_values,
)


def test_canary_cannot_be_confused_with_schema_f() -> None:
    assert (
        CANARY_ID
        != SCHEMA_F.schema_id
    )

    assert (
        CANARY_SCHEMA.schema_id
        == CANARY_ID
    )

    assert (
        CANARY_SCHEMA.functionals
        == SCHEMA_F.functionals
    )

    assert (
        CANARY_SCHEMA.fast_bases
        != SCHEMA_F.fast_bases
    )

    assert (
        CANARY_WARNING.startswith(
            "WORKFLOW CANARY ONLY"
        )
    )


def test_canary_fast_plan_is_reduced() -> None:
    plan = (
        build_canary_fast_plan(
            neutral_electrons=9,
            anion_electrons=10,
        )
    )

    assert (
        plan.task_count
        == 32
    )

    assert (
        plan.bond_point_count
        == 4
    )

    assert {
        task.functional
        for task
        in plan.tasks
    } == set(
        SCHEMA_F.functionals
    )

    assert {
        task.basis
        for task
        in plan.tasks
    } == {
        "def2-svp"
    }

    assert {
        task.spin
        for task
        in plan.tasks
        if int(
            task.charge
        ) == 0
    } == {
        1
    }

    assert {
        task.spin
        for task
        in plan.tasks
        if int(
            task.charge
        ) == -1
    } == {
        0
    }


def test_report_is_explicitly_non_scientific(
    tmp_path,
) -> None:
    estimate = (
        estimate_schema_f_from_values(
            molecule="OH",
            functional_eas_ev={
                "PBE": 1.70,
                "B3LYP": 1.75,
                "PBE0": 1.80,
                "TPSSh": 1.78,
            },
        )
    )

    fast_plan = (
        build_canary_fast_plan(
            neutral_electrons=9,
            anion_electrons=10,
        )
    )

    qzvpd_plan = QZVPDPlan(
        molecule=fast_plan.molecule,
        candidates=(),
        tasks=(),
    )

    qzvpd_analysis = QZVPDAnalysis(
        points=(),
        state_minima=(),
        charge_minima=(),
        functional_eas=(),
    )

    payload = (
        build_canary_report_payload(
            neutral_electrons=9,
            anion_electrons=10,
            fast_plan=fast_plan,
            qzvpd_plan=qzvpd_plan,
            qzvpd_analysis=(
                qzvpd_analysis
            ),
            estimate=estimate,
            provenance={
                "compatibility": {
                    "verified": True,
                },
            },
            fast_grid_csv=(
                tmp_path
                / "fast.csv"
            ),
            qzvpd_csv=(
                tmp_path
                / "qz.csv"
            ),
        )
    )

    assert (
        payload[
            "scientific_result"
        ]
        is False
    )

    assert (
        payload[
            "schema_f_result"
        ]
        is False
    )

    assert (
        payload[
            "warning"
        ]
        == CANARY_WARNING
    )

    assert (
        payload[
            "report_type"
        ]
        == (
            "diatomic_ea_workflow_canary"
        )
    )

    path = (
        write_canary_report(
            tmp_path
            / "report.json",
            payload,
        )
    )

    restored = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        restored[
            "scientific_result"
        ]
        is False
    )

    assert (
        restored[
            "schema_f_result"
        ]
        is False
    )


def test_canary_uses_real_qzvpd_basis() -> None:
    assert (
        CANARY_SCHEMA
        .refinement
        .basis
        == "def2-qzvpd"
    )

    assert (
        CANARY_SCHEMA
        .refinement
        .max_spins_per_charge
        == 1
    )
