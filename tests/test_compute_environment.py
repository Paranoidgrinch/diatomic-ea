"""Tests for cross-platform compute-environment detection."""

import json
from unittest.mock import patch

from diatomic_ea.backend import (
    BackendAvailability,
)
from diatomic_ea.compute_environment import (
    ComputeEnvironmentState,
    EXPECTED_PYSCF_VERSION,
    WSL_COMPUTE_PYTHON,
    inspect_compute_environment,
)
from diatomic_ea.wsl import (
    WSLAvailability,
    WSLCommandResult,
)


def wsl_result(
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


def availability(
    distributions=("Ubuntu-24.04",),
):
    return WSLAvailability(
        executable="wsl.exe",
        distributions=distributions,
        message="ready",
    )


def test_windows_without_wsl() -> None:
    state = WSLAvailability(
        executable=None,
        distributions=(),
        message="missing",
    )

    with patch(
        "diatomic_ea.compute_environment.inspect_wsl",
        return_value=state,
    ):
        report = inspect_compute_environment(
            system_name="Windows"
        )

    assert (
        report.state
        is ComputeEnvironmentState.WSL_MISSING
    )

    assert not report.ready


def test_windows_without_distribution() -> None:
    state = WSLAvailability(
        executable="wsl.exe",
        distributions=(),
        message="none",
    )

    with patch(
        "diatomic_ea.compute_environment.inspect_wsl",
        return_value=state,
    ):
        report = inspect_compute_environment(
            system_name="Windows"
        )

    assert (
        report.state
        is ComputeEnvironmentState
        .WSL_NO_DISTRIBUTION
    )


def test_windows_launch_failure() -> None:
    with (
        patch(
            "diatomic_ea.compute_environment.inspect_wsl",
            return_value=availability(),
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_command",
            return_value=wsl_result(
                returncode=1,
                stderr="cannot launch",
            ),
        ),
    ):
        report = inspect_compute_environment(
            system_name="Windows"
        )

    assert (
        report.state
        is ComputeEnvironmentState
        .WSL_LAUNCH_FAILED
    )


def test_windows_missing_compute_environment() -> None:
    with (
        patch(
            "diatomic_ea.compute_environment.inspect_wsl",
            return_value=availability(),
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_command",
            side_effect=[
                wsl_result(),
                wsl_result(
                    returncode=1
                ),
            ],
        ),
    ):
        report = inspect_compute_environment(
            system_name="Windows"
        )

    assert (
        report.state
        is ComputeEnvironmentState
        .WSL_ENV_MISSING
    )


def test_ready_windows_environment() -> None:
    payload = json.dumps(
        {
            "python_version": "3.12.3",
            "pyscf_version": (
                EXPECTED_PYSCF_VERSION
            ),
            "basis_set_exchange_version": (
                "0.12"
            ),
        }
    )

    with (
        patch(
            "diatomic_ea.compute_environment.inspect_wsl",
            return_value=availability(),
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_command",
            side_effect=[
                wsl_result(),
                wsl_result(),
                wsl_result(
                    stdout=payload
                ),
            ],
        ) as command,
    ):
        report = inspect_compute_environment(
            system_name="Windows"
        )

    assert report.ready

    assert (
        report.state
        is ComputeEnvironmentState.READY_WSL
    )

    assert (
        report.pyscf_version
        == EXPECTED_PYSCF_VERSION
    )

    calls = command.call_args_list

    assert calls[1].args[0] == (
        "test",
        "-x",
        WSL_COMPUTE_PYTHON,
    )

    assert calls[2].args[0][0] == (
        WSL_COMPUTE_PYTHON
    )

    assert calls[2].args[0][1] == "-c"


def test_ready_native_linux_environment() -> None:
    backend = BackendAvailability(
        backend="PySCF",
        platform_supported=True,
        installed=True,
        version=EXPECTED_PYSCF_VERSION,
        message="ready",
    )

    with (
        patch(
            "diatomic_ea.compute_environment.PySCFBackend.availability",
            return_value=backend,
        ),
        patch(
            "diatomic_ea.compute_environment._package_version",
            return_value="0.12",
        ),
    ):
        report = inspect_compute_environment(
            system_name="Linux"
        )

    assert report.ready

    assert (
        report.state
        is ComputeEnvironmentState
        .READY_NATIVE
    )

    assert report.backend == "native"


def test_native_version_mismatch() -> None:
    backend = BackendAvailability(
        backend="PySCF",
        platform_supported=True,
        installed=True,
        version="9.9.9",
        message="ready",
    )

    with (
        patch(
            "diatomic_ea.compute_environment.PySCFBackend.availability",
            return_value=backend,
        ),
        patch(
            "diatomic_ea.compute_environment._package_version",
            return_value="0.12",
        ),
    ):
        report = inspect_compute_environment(
            system_name="Linux"
        )

    assert not report.ready

    assert (
        report.state
        is ComputeEnvironmentState
        .NATIVE_PYSCF_VERSION_MISMATCH
    )
