"""Tests for the platform single-point adapter."""

from unittest.mock import patch

import pytest

from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    WSL_COMPUTE_PYTHON,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.single_point_adapter import (
    SinglePointExecutionError,
    run_platform_single_point,
    run_wsl_single_point,
)
from diatomic_ea.single_point_protocol import (
    dumps_result,
    dumps_task,
)
from diatomic_ea.states import (
    ChargeState,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
)


def task() -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        charge=ChargeState.NEUTRAL,
        spin=0,
        functional="PBE",
        basis="def2-SVP",
        bond_length_angstrom=0.92,
        grid_level=1,
        conv_tol=1.0e-8,
        max_cycle=80,
        max_memory_mb=1000,
        threads_per_worker=1,
    )


def result_for(
    current: SinglePointTask,
    *,
    task_id: str | None = None,
) -> SinglePointResult:
    return SinglePointResult(
        task_id=(
            task_id
            if task_id is not None
            else current.task_id
        ),
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=-100.10,
        energy_ev=-2723.9,
        converged=True,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=10,
        alpha_electrons=5,
        beta_electrons=5,
        basis_label_a="def2-SVP",
        basis_label_b="def2-SVP",
        ecp_label_a="",
        ecp_label_b="",
        frontier=None,
        s2=0.0,
        observed_multiplicity=1.0,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.5,
    )


def ready_environment():
    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .READY_WSL
        ),
        system="Windows",
        backend="wsl",
        distribution="Ubuntu-24.04",
        python_version="3.12.3",
        pyscf_version="2.13.0",
        basis_set_exchange_version="0.12",
        message="ready",
    )


def command_result(
    *,
    returncode=0,
    stdout="",
    stderr="",
):
    return WSLCommandResult(
        command=("wsl.exe",),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_wsl_adapter_transports_task_and_result() -> None:
    current = task()
    expected = result_for(
        current
    )

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command",
            return_value=command_result(
                stdout=(
                    dumps_result(
                        expected
                    )
                    + "\n"
                )
            ),
        ) as command,
    ):
        restored = run_wsl_single_point(
            current,
            timeout=123.0,
        )

    assert restored == expected

    arguments = (
        command.call_args.args[0]
    )

    assert arguments == (
        WSL_COMPUTE_PYTHON,
        "-m",
        "diatomic_ea.single_point_worker",
    )

    assert (
        command.call_args.kwargs[
            "distribution"
        ]
        == "Ubuntu-24.04"
    )

    assert (
        command.call_args.kwargs[
            "input_text"
        ]
        == dumps_task(
            current
        )
    )

    assert (
        command.call_args.kwargs[
            "timeout"
        ]
        == 123.0
    )


def test_worker_stdout_may_contain_nonprotocol_lines() -> None:
    current = task()
    expected = result_for(
        current
    )

    stdout = (
        "irrelevant diagnostic line\n"
        + dumps_result(
            expected
        )
        + "\n"
    )

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command",
            return_value=command_result(
                stdout=stdout
            ),
        ),
    ):
        restored = run_wsl_single_point(
            current
        )

    assert restored == expected


def test_unready_environment_is_rejected() -> None:
    current = task()

    unready = ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .WSL_ENV_INVALID
        ),
        system="Windows",
        backend="wsl",
        distribution="Ubuntu-24.04",
        python_version=None,
        pyscf_version=None,
        basis_set_exchange_version=None,
        message="broken",
    )

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=unready,
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command"
        ) as command,
    ):
        with pytest.raises(
            SinglePointExecutionError,
            match="not ready",
        ):
            run_wsl_single_point(
                current
            )

    command.assert_not_called()


def test_worker_process_failure_is_reported() -> None:
    current = task()

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command",
            return_value=command_result(
                returncode=3,
                stderr="WORKER_ERROR: boom",
            ),
        ),
    ):
        with pytest.raises(
            SinglePointExecutionError,
            match="WORKER_ERROR",
        ):
            run_wsl_single_point(
                current
            )


def test_malformed_worker_output_is_rejected() -> None:
    current = task()

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command",
            return_value=command_result(
                stdout="not-json\n"
            ),
        ),
    ):
        with pytest.raises(
            SinglePointExecutionError,
            match="no valid",
        ):
            run_wsl_single_point(
                current
            )


def test_wrong_task_id_is_rejected() -> None:
    current = task()

    wrong = result_for(
        current,
        task_id="wrong-task-id",
    )

    with (
        patch(
            "diatomic_ea.single_point_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.single_point_adapter.run_wsl_command",
            return_value=command_result(
                stdout=dumps_result(
                    wrong
                )
            ),
        ),
    ):
        with pytest.raises(
            SinglePointExecutionError,
            match="wrong task",
        ):
            run_wsl_single_point(
                current
            )


def test_linux_uses_native_single_point_runner() -> None:
    current = task()
    expected = result_for(
        current
    )

    with patch(
        "diatomic_ea.single_point_adapter.run_pyscf_single_point",
        return_value=expected,
    ) as native:
        restored = run_platform_single_point(
            current,
            system_name="Linux",
        )

    assert restored == expected

    native.assert_called_once_with(
        current
    )


def test_windows_uses_wsl_adapter() -> None:
    current = task()
    expected = result_for(
        current
    )

    with patch(
        "diatomic_ea.single_point_adapter.run_wsl_single_point",
        return_value=expected,
    ) as wsl:
        restored = run_platform_single_point(
            current,
            system_name="Windows",
        )

    assert restored == expected

    wsl.assert_called_once()
