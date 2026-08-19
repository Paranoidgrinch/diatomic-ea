"""Tests for platform-aware electron-count resolution."""

import json
from unittest.mock import patch

import pytest

from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    WSL_COMPUTE_PYTHON,
)
from diatomic_ea.electron_count_adapter import (
    ElectronCountExecutionError,
    run_platform_electron_count,
    run_wsl_electron_count,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.states import (
    ChargeState,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
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


def test_windows_electron_count_uses_wsl() -> None:
    molecule = DiatomicMolecule(
        "H",
        "F",
    )

    with (
        patch(
            "diatomic_ea.electron_count_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.electron_count_adapter.run_wsl_command",
            return_value=command_result(
                stdout=(
                    '{"electron_count":10}\n'
                )
            ),
        ) as command,
    ):
        count = run_wsl_electron_count(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            basis="def2-SVP",
            bond_length_angstrom=0.92,
            max_memory_mb=1000,
        )

    assert count == 10

    arguments = (
        command.call_args.args[0]
    )

    assert arguments[0] == (
        WSL_COMPUTE_PYTHON
    )

    assert arguments[1] == "-c"

    payload = json.loads(
        command.call_args.kwargs[
            "input_text"
        ]
    )

    assert payload == {
        "atom_a": "H",
        "atom_b": "F",
        "charge": 0,
        "basis": "def2-SVP",
        "bond_length_angstrom": 0.92,
        "max_memory_mb": 1000,
    }


def test_unready_wsl_is_rejected() -> None:
    broken = ComputeEnvironmentReport(
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
            "diatomic_ea.electron_count_adapter.inspect_compute_environment",
            return_value=broken,
        ),
        patch(
            "diatomic_ea.electron_count_adapter.run_wsl_command"
        ) as command,
    ):
        with pytest.raises(
            ElectronCountExecutionError,
            match="not ready",
        ):
            run_wsl_electron_count(
                molecule=DiatomicMolecule(
                    "H",
                    "F",
                ),
                charge=ChargeState.NEUTRAL,
                basis="def2-SVP",
                bond_length_angstrom=0.92,
                max_memory_mb=1000,
            )

    command.assert_not_called()


def test_invalid_worker_output_is_rejected() -> None:
    with (
        patch(
            "diatomic_ea.electron_count_adapter.inspect_compute_environment",
            return_value=ready_environment(),
        ),
        patch(
            "diatomic_ea.electron_count_adapter.run_wsl_command",
            return_value=command_result(
                stdout="not-json\n"
            ),
        ),
    ):
        with pytest.raises(
            ElectronCountExecutionError,
            match="no valid",
        ):
            run_wsl_electron_count(
                molecule=DiatomicMolecule(
                    "H",
                    "F",
                ),
                charge=ChargeState.NEUTRAL,
                basis="def2-SVP",
                bond_length_angstrom=0.92,
                max_memory_mb=1000,
            )


def test_linux_uses_native_resolver() -> None:
    molecule = DiatomicMolecule(
        "H",
        "F",
    )

    with patch(
        "diatomic_ea.electron_count_adapter.determine_electron_count",
        return_value=10,
    ) as native:
        count = run_platform_electron_count(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            basis="def2-SVP",
            bond_length_angstrom=0.92,
            max_memory_mb=1000,
            system_name="Linux",
        )

    assert count == 10

    native.assert_called_once_with(
        molecule=molecule,
        charge=ChargeState.NEUTRAL,
        basis="def2-SVP",
        bond_length_angstrom=0.92,
        max_memory_mb=1000,
    )


def test_windows_uses_wsl_resolver() -> None:
    molecule = DiatomicMolecule(
        "H",
        "F",
    )

    with patch(
        "diatomic_ea.electron_count_adapter.run_wsl_electron_count",
        return_value=10,
    ) as wsl:
        count = run_platform_electron_count(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            basis="def2-SVP",
            bond_length_angstrom=0.92,
            max_memory_mb=1000,
            system_name="Windows",
        )

    assert count == 10
    wsl.assert_called_once()
