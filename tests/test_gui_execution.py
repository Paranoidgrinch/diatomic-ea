"""Tests for GUI calculation execution helpers."""

from pathlib import Path

import pytest

from diatomic_ea.gui_execution import (
    GuiCalculationSpec,
    build_plan_command,
    build_run_command,
    calculation_mode_label,
    job_status_label,
    make_gui_run_id,
)
from diatomic_ea.jobs import (
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)


def spec():
    molecule = DiatomicMolecule(
        "Al",
        "O",
    )

    return GuiCalculationSpec(
        job_id="abcdef1234567890",
        molecule=molecule,
        minimum_angstrom=1.20,
        maximum_angstrom=2.20,
        spin_max=5,
        workers=4,
        run_id=make_gui_run_id(
            molecule,
            "abcdef1234567890",
        ),
    )


def test_internal_method_name_is_not_exposed() -> None:
    label = calculation_mode_label(
        CalculationMode.SCHEMA_F
    )

    assert (
        label
        == "Standard calculation"
    )

    assert (
        "schema"
        not in label.casefold()
    )


@pytest.mark.parametrize(
    (
        "status",
        "expected",
    ),
    (
        (
            JobStatus.QUEUED,
            "Waiting",
        ),
        (
            JobStatus.RUNNING,
            "Running",
        ),
        (
            JobStatus.COMPLETED,
            "Completed",
        ),
        (
            JobStatus.FAILED,
            "Failed",
        ),
    ),
)
def test_job_status_labels(
    status,
    expected,
) -> None:
    assert (
        job_status_label(
            status
        )
        == expected
    )


def test_gui_run_id_is_stable() -> None:
    molecule = DiatomicMolecule(
        "Al",
        "O",
    )

    assert (
        make_gui_run_id(
            molecule,
            "abcdef1234567890",
        )
        == "alo-abcdef123456"
    )


def test_plan_command_contains_frozen_job_settings(
    tmp_path,
) -> None:
    command = build_plan_command(
        spec(),
        output_root=tmp_path,
        python_executable="python-test",
    )

    assert (
        command.program
        == "python-test"
    )

    arguments = list(
        command.arguments
    )

    assert (
        arguments[:3]
        == [
            "-m",
            "diatomic_ea.production_plan",
            "Al",
        ]
    )

    assert (
        "--minimum"
        in arguments
    )

    assert (
        arguments[
            arguments.index(
                "--minimum"
            )
            + 1
        ]
        == "1.2"
    )

    assert (
        arguments[
            arguments.index(
                "--maximum"
            )
            + 1
        ]
        == "2.2"
    )

    assert (
        arguments[
            arguments.index(
                "--workers"
            )
            + 1
        ]
        == "4"
    )


def test_run_command_uses_prepared_plan(
    tmp_path,
) -> None:
    calculation = spec()

    command = build_run_command(
        calculation,
        output_root=tmp_path,
        python_executable="python-test",
    )

    assert (
        command.program
        == "python-test"
    )

    arguments = list(
        command.arguments
    )

    plan = (
        calculation.plan_path(
            tmp_path
        )
    )

    assert (
        str(
            plan
        )
        in arguments
    )

    assert (
        "--start"
        in arguments
    )

    assert (
        "--recover-stale-lock"
        not in arguments
    )


def test_spec_paths_share_same_run_directory(
    tmp_path,
) -> None:
    calculation = spec()

    run_directory = (
        calculation.run_directory(
            tmp_path
        )
    )

    assert (
        calculation.plan_path(
            tmp_path
        ).parent
        == run_directory
    )

    assert (
        calculation.status_path(
            tmp_path
        ).parents[1]
        == run_directory
    )

    assert (
        calculation.final_result_path(
            tmp_path
        ).parents[1]
        == run_directory
    )


def test_invalid_range_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater",
    ):
        GuiCalculationSpec(
            job_id="x",
            molecule=DiatomicMolecule(
                "O",
                "H",
            ),
            minimum_angstrom=2.0,
            maximum_angstrom=1.0,
            spin_max=3,
            workers=2,
            run_id="test",
        )
