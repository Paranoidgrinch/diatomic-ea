"""Tests for cross-platform compute-environment detection."""

import json
from unittest.mock import patch

from diatomic_ea.backend import BackendAvailability
from diatomic_ea.compute_environment import (
    ComputeEnvironmentState,
    EXPECTED_PYSCF_VERSION,
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


def test_windows_without_wsl() -> None:
    availability = WSLAvailability(
        executable=None,
        distributions=(),
        message="missing",
    )

    with patch(
        "diatomic_ea.compute_environment.inspect_wsl",
        return_value=availability,
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
    availability = WSLAvailability(
        executable="wsl.exe",
        distributions=(),
        message="none",
    )

    with patch(
        "diatomic_ea.compute_environment.inspect_wsl",
        return_value=availability,
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
    availability = WSLAvailability(
        executable="wsl.exe",
        distributions=("Ubuntu-24.04",),
        message="ready",
    )

    with (
        patch(
            "diatomic_ea.compute_environment.inspect_wsl",
            return_value=availability,
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
    availability = WSLAvailability(
        executable="wsl.exe",
        distributions=("Ubuntu-24.04",),
        message="ready",
    )

    with (
        patch(
            "diatomic_ea.compute_environment.inspect_wsl",
            return_value=availability,
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_command",
            return_value=wsl_result(),
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_shell",
            return_value=wsl_result(
                returncode=20
            ),
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
    availability = WSLAvailability(
        executable="wsl.exe",
        distributions=("Ubuntu-24.04",),
        message="ready",
    )

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
            return_value=availability,
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_command",
            return_value=wsl_result(),
        ),
        patch(
            "diatomic_ea.compute_environment.run_wsl_shell",
            return_value=wsl_result(
                stdout=payload
            ),
        ),
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