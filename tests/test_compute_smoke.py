"""Tests for the real platform-aware PySCF smoke test."""

import json
from unittest.mock import patch

import pytest

from diatomic_ea.backend import (
    BackendAvailability,
    BackendSmokeReport,
)
from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    WSL_COMPUTE_PYTHON,
)
from diatomic_ea.compute_smoke import (
    run_compute_smoke,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
)


def ready_wsl_report():
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


def missing_wsl_report():
    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .WSL_ENV_MISSING
        ),
        system="Windows",
        backend="wsl",
        distribution="Ubuntu-24.04",
        python_version=None,
        pyscf_version=None,
        basis_set_exchange_version=None,
        message="missing",
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


def test_successful_wsl_smoke() -> None:
    payload = json.dumps(
        {
            "pyscf_version": "2.13.0",
            "converged": True,
            "energy_hartree": -1.15,
            "finite": True,
        }
    )

    with (
        patch(
            "diatomic_ea.compute_smoke.inspect_compute_environment",
            return_value=ready_wsl_report(),
        ),
        patch(
            "diatomic_ea.compute_smoke.run_wsl_command",
            return_value=command_result(
                stdout=payload
            ),
        ) as command,
    ):
        report = run_compute_smoke(
            system_name="Windows"
        )

    assert report.passed
    assert report.backend == "wsl"
    assert report.converged is True

    assert (
        report.energy_hartree
        == pytest.approx(
            -1.15
        )
    )

    arguments = (
        command.call_args.args[0]
    )

    assert arguments[0] == (
        WSL_COMPUTE_PYTHON
    )

    assert arguments[1] == "-c"


def test_missing_wsl_environment_stops_smoke() -> None:
    with (
        patch(
            "diatomic_ea.compute_smoke.inspect_compute_environment",
            return_value=missing_wsl_report(),
        ),
        patch(
            "diatomic_ea.compute_smoke.run_wsl_command"
        ) as command,
    ):
        report = run_compute_smoke(
            system_name="Windows"
        )

    assert not report.passed

    command.assert_not_called()


def test_nonconverged_wsl_smoke_fails() -> None:
    payload = json.dumps(
        {
            "pyscf_version": "2.13.0",
            "converged": False,
            "energy_hartree": -1.15,
            "finite": True,
        }
    )

    with (
        patch(
            "diatomic_ea.compute_smoke.inspect_compute_environment",
            return_value=ready_wsl_report(),
        ),
        patch(
            "diatomic_ea.compute_smoke.run_wsl_command",
            return_value=command_result(
                stdout=payload
            ),
        ),
    ):
        report = run_compute_smoke(
            system_name="Windows"
        )

    assert not report.passed
    assert report.converged is False


def test_wsl_version_mismatch_fails() -> None:
    payload = json.dumps(
        {
            "pyscf_version": "9.9.9",
            "converged": True,
            "energy_hartree": -1.15,
            "finite": True,
        }
    )

    with (
        patch(
            "diatomic_ea.compute_smoke.inspect_compute_environment",
            return_value=ready_wsl_report(),
        ),
        patch(
            "diatomic_ea.compute_smoke.run_wsl_command",
            return_value=command_result(
                stdout=payload
            ),
        ),
    ):
        report = run_compute_smoke(
            system_name="Windows"
        )

    assert not report.passed


def test_native_linux_smoke() -> None:
    availability = BackendAvailability(
        backend="PySCF",
        platform_supported=True,
        installed=True,
        version="2.13.0",
        message="ready",
    )

    smoke = BackendSmokeReport(
        backend="PySCF",
        passed=True,
        message="passed",
        energy_hartree=-1.15,
    )

    with (
        patch(
            "diatomic_ea.compute_smoke.PySCFBackend.availability",
            return_value=availability,
        ),
        patch(
            "diatomic_ea.compute_smoke.PySCFBackend.smoke_test",
            return_value=smoke,
        ),
    ):
        report = run_compute_smoke(
            system_name="Linux"
        )

    assert report.passed
    assert report.backend == "native"

    assert (
        report.energy_hartree
        == pytest.approx(
            -1.15
        )
    )


def test_native_version_mismatch_fails() -> None:
    availability = BackendAvailability(
        backend="PySCF",
        platform_supported=True,
        installed=True,
        version="9.9.9",
        message="ready",
    )

    with patch(
        "diatomic_ea.compute_smoke.PySCFBackend.availability",
        return_value=availability,
    ):
        report = run_compute_smoke(
            system_name="Linux"
        )

    assert not report.passed
