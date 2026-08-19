"""Regression tests for the deterministic WSL compute path."""

from unittest.mock import patch

from diatomic_ea.compute_bootstrap import (
    bootstrap_wsl_compute_environment,
)
from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    WSL_COMPUTE_PYTHON,
    WSL_COMPUTE_ROOT,
    WSL_COMPUTE_VENV,
    _wsl_probe_python,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
)


def command_result():
    return WSLCommandResult(
        command=("wsl.exe",),
        returncode=0,
        stdout="",
        stderr="",
    )


def missing_report():
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


def test_compute_paths_are_absolute() -> None:
    assert (
        WSL_COMPUTE_ROOT
        == "/opt/diatomic-ea"
    )

    assert (
        WSL_COMPUTE_VENV
        == "/opt/diatomic-ea/venv"
    )

    assert (
        WSL_COMPUTE_PYTHON
        == "/opt/diatomic-ea/venv/bin/python"
    )


def test_probe_is_pure_python() -> None:
    program = _wsl_probe_python()

    assert "import pyscf" in program
    assert "PY=" not in program
    assert "test -x" not in program
    assert "$HOME" not in program


def test_bootstrap_creates_fixed_path_as_root() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            side_effect=[
                missing_report(),
                ready_report(),
            ],
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command",
            return_value=command_result(),
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_shell",
            return_value=command_result(),
        ) as shell,
    ):
        result = (
            bootstrap_wsl_compute_environment()
        )

    assert result.success
    assert not result.already_ready

    shell.assert_called_once()

    call = shell.call_args

    setup_command = call.args[0]

    assert (
        "/opt/diatomic-ea/venv"
        in setup_command
    )

    assert "$HOME" not in setup_command

    assert (
        call.kwargs["user"]
        == "root"
    )


def test_ready_environment_remains_idempotent() -> None:
    with (
        patch(
            "diatomic_ea.compute_bootstrap.inspect_compute_environment",
            return_value=ready_report(),
        ),
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_command"
        ) as command,
        patch(
            "diatomic_ea.compute_bootstrap.run_wsl_shell"
        ) as shell,
    ):
        result = (
            bootstrap_wsl_compute_environment()
        )

    assert result.success
    assert result.already_ready

    command.assert_not_called()
    shell.assert_not_called()
