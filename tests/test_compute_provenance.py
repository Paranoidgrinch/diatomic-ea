"""Tests for compute-backend provenance."""

import json
from unittest.mock import patch

from diatomic_ea import __version__
from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    WSL_COMPUTE_VENV,
)
from diatomic_ea.compute_provenance import (
    collect_compute_provenance,
)
from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
)


VALID_HASH = (
    "0123456789abcdef"
    * 4
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


def ready_wsl():
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


def ready_native():
    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .READY_NATIVE
        ),
        system="Linux",
        backend="native",
        distribution=None,
        python_version="3.12.3",
        pyscf_version="2.13.0",
        basis_set_exchange_version="0.12",
        message="ready",
    )


def worker_payload(
    *,
    protocol=(
        SINGLE_POINT_PROTOCOL_VERSION
    ),
    wheel_hash=VALID_HASH,
):
    return {
        "system": "Linux",
        "platform": "Linux-test",
        "machine": "x86_64",
        "kernel_release": "6.8-test",
        "python_version": "3.12.3",
        "pyscf_version": "2.13.0",
        "basis_set_exchange_version": "0.12",
        "diatomic_ea_version": __version__,
        "single_point_protocol_version": (
            protocol
        ),
        "module_path": (
            WSL_COMPUTE_VENV
            + "/lib/python3.12/"
            "site-packages/"
            "diatomic_ea/__init__.py"
        ),
        "worker_module_path": (
            WSL_COMPUTE_VENV
            + "/lib/python3.12/"
            "site-packages/"
            "diatomic_ea/"
            "single_point_worker.py"
        ),
        "worker_wheel_sha256": (
            wheel_hash
        ),
    }


def test_ready_wsl_provenance_is_verified() -> None:
    with (
        patch(
            "diatomic_ea.compute_provenance.inspect_compute_environment",
            return_value=ready_wsl(),
        ),
        patch(
            "diatomic_ea.compute_provenance.run_wsl_command",
            return_value=command_result(
                stdout=(
                    json.dumps(
                        worker_payload()
                    )
                    + "\n"
                )
            ),
        ),
    ):
        provenance = (
            collect_compute_provenance(
                system_name="Windows"
            )
        )

    assert provenance[
        "backend"
    ] == "wsl"

    assert provenance[
        "ready"
    ] is True

    assert provenance[
        "compatibility"
    ][
        "verified"
    ] is True

    assert provenance[
        "compute"
    ][
        "distribution"
    ] == "Ubuntu-24.04"

    assert provenance[
        "compute"
    ][
        "kernel_release"
    ] == "6.8-test"

    assert provenance[
        "compute"
    ][
        "worker_wheel_sha256"
    ] == VALID_HASH


def test_protocol_mismatch_is_not_verified() -> None:
    payload = worker_payload(
        protocol=999
    )

    with (
        patch(
            "diatomic_ea.compute_provenance.inspect_compute_environment",
            return_value=ready_wsl(),
        ),
        patch(
            "diatomic_ea.compute_provenance.run_wsl_command",
            return_value=command_result(
                stdout=json.dumps(
                    payload
                )
            ),
        ),
    ):
        provenance = (
            collect_compute_provenance(
                system_name="Windows"
            )
        )

    assert provenance[
        "compatibility"
    ][
        "verified"
    ] is False

    assert provenance[
        "compatibility"
    ][
        "protocol_version_match"
    ] is False


def test_invalid_wheel_hash_is_not_verified() -> None:
    payload = worker_payload(
        wheel_hash="not-a-hash"
    )

    with (
        patch(
            "diatomic_ea.compute_provenance.inspect_compute_environment",
            return_value=ready_wsl(),
        ),
        patch(
            "diatomic_ea.compute_provenance.run_wsl_command",
            return_value=command_result(
                stdout=json.dumps(
                    payload
                )
            ),
        ),
    ):
        provenance = (
            collect_compute_provenance(
                system_name="Windows"
            )
        )

    assert provenance[
        "compatibility"
    ][
        "verified"
    ] is False

    assert provenance[
        "compatibility"
    ][
        "wheel_hash_valid"
    ] is False


def test_unready_wsl_does_not_probe_worker() -> None:
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
            "diatomic_ea.compute_provenance.inspect_compute_environment",
            return_value=broken,
        ),
        patch(
            "diatomic_ea.compute_provenance.run_wsl_command"
        ) as command,
    ):
        provenance = (
            collect_compute_provenance(
                system_name="Windows"
            )
        )

    assert provenance[
        "ready"
    ] is False

    assert provenance[
        "compatibility"
    ][
        "verified"
    ] is False

    command.assert_not_called()


def test_native_provenance_is_supported() -> None:
    with patch(
        "diatomic_ea.compute_provenance.inspect_compute_environment",
        return_value=ready_native(),
    ):
        provenance = (
            collect_compute_provenance(
                system_name="Linux"
            )
        )

    assert provenance[
        "backend"
    ] == "native"

    assert provenance[
        "compatibility"
    ][
        "verified"
    ] is True

    assert provenance[
        "compute"
    ][
        "worker_wheel_sha256"
    ] is None
