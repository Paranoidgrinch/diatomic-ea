"""Tests for WSL compute-environment bootstrap."""

from unittest.mock import patch

from diatomic_ea.compute_bootstrap import (
    bootstrap_wsl_compute_environment,
)
from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
    run_wsl_command,
)


def command_result(
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


def ready_report():
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


def missing_environment_report():
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


def no_distribution_report():
    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .WSL_NO_DISTRIBUTION
        ),
        system="Windows",
        backend="wsl",
        distribution=None,
        python_version=None,
        pyscf_version=None,
        basis_set_exchange_version=None,
        message="no distro",
    )


def test_ready_environment_is_not_reinstalled() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            return_value=ready_report(),
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command"
        ) as command,
    ):
        report = (
            bootstrap_wsl_compute_environment()
        )

    assert report.success
    assert report.already_ready

    command.assert_not_called()


def test_missing_environment_is_installed() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            side_effect=[
                missing_environment_report(),
                ready_report(),
            ],
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command",
            return_value=command_result(),
        ) as command,
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_shell",
            return_value=command_result(),
        ) as shell,
    ):
        report = (
            bootstrap_wsl_compute_environment()
        )

    assert report.success
    assert not report.already_ready
    assert command.call_count == 2

    shell.assert_called_once()


def test_distribution_must_exist_before_bootstrap() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            return_value=no_distribution_report(),
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command"
        ) as command,
    ):
        report = (
            bootstrap_wsl_compute_environment()
        )

    assert not report.success
    command.assert_not_called()


def test_apt_failure_stops_bootstrap() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            return_value=missing_environment_report(),
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command",
            return_value=command_result(
                returncode=1,
                stderr="apt failed",
            ),
        ),
    ):
        report = (
            bootstrap_wsl_compute_environment()
        )

    assert not report.success
    assert "apt failed" in report.message


def test_wsl_bridge_supports_root_user() -> None:
    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run"
        ) as mocked,
    ):
        mocked.return_value.returncode = 0
        mocked.return_value.stdout = b""
        mocked.return_value.stderr = b""

        run_wsl_command(
            (
                "id",
                "-u",
            ),
            distribution="Ubuntu-24.04",
            user="root",
        )

    command = mocked.call_args.args[0]

    assert command == [
        "wsl.exe",
        "--distribution",
        "Ubuntu-24.04",
        "--user",
        "root",
        "--",
        "id",
        "-u",
    ]