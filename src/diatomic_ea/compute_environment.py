"""Cross-platform compute-environment detection for DiatomicEA."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import dataclass
from enum import Enum

from diatomic_ea.backend import PySCFBackend
from diatomic_ea.schema_f import SCHEMA_F
from diatomic_ea.wsl import (
    inspect_wsl,
    run_wsl_command,
)


DEFAULT_WSL_DISTRIBUTION = "Ubuntu-24.04"

EXPECTED_PYSCF_VERSION = (
    SCHEMA_F.reference_pyscf_version
)

WSL_COMPUTE_ROOT = "/opt/diatomic-ea"

WSL_COMPUTE_VENV = (
    WSL_COMPUTE_ROOT
    + "/venv"
)

WSL_COMPUTE_PYTHON = (
    WSL_COMPUTE_VENV
    + "/bin/python"
)


class ComputeEnvironmentState(str, Enum):
    READY_NATIVE = "ready_native"
    READY_WSL = "ready_wsl"

    NATIVE_PYSCF_MISSING = (
        "native_pyscf_missing"
    )

    NATIVE_PYSCF_VERSION_MISMATCH = (
        "native_pyscf_version_mismatch"
    )

    NATIVE_BSE_MISSING = (
        "native_bse_missing"
    )

    WSL_MISSING = "wsl_missing"
    WSL_UNRESPONSIVE = "wsl_unresponsive"

    WSL_NO_DISTRIBUTION = (
        "wsl_no_distribution"
    )

    WSL_DISTRIBUTION_NOT_FOUND = (
        "wsl_distribution_not_found"
    )

    WSL_LAUNCH_FAILED = (
        "wsl_launch_failed"
    )

    WSL_ENV_MISSING = (
        "wsl_env_missing"
    )

    WSL_ENV_INVALID = (
        "wsl_env_invalid"
    )

    WSL_PYSCF_VERSION_MISMATCH = (
        "wsl_pyscf_version_mismatch"
    )

    WSL_BSE_MISSING = (
        "wsl_bse_missing"
    )


@dataclass(frozen=True, slots=True)
class ComputeEnvironmentReport:
    """Current compute-backend readiness."""

    state: ComputeEnvironmentState
    system: str
    backend: str
    distribution: str | None
    python_version: str | None
    pyscf_version: str | None
    basis_set_exchange_version: str | None
    message: str

    @property
    def ready(self) -> bool:
        return self.state in {
            ComputeEnvironmentState.READY_NATIVE,
            ComputeEnvironmentState.READY_WSL,
        }


def _package_version(
    name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def _native_report(
    system_name: str,
) -> ComputeEnvironmentReport:
    availability = (
        PySCFBackend().availability()
    )

    if not availability.installed:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .NATIVE_PYSCF_MISSING
            ),
            system=system_name,
            backend="native",
            distribution=None,
            python_version=(
                platform.python_version()
            ),
            pyscf_version=None,
            basis_set_exchange_version=(
                _package_version(
                    "basis-set-exchange"
                )
            ),
            message=availability.message,
        )

    if (
        availability.version
        != EXPECTED_PYSCF_VERSION
    ):
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .NATIVE_PYSCF_VERSION_MISMATCH
            ),
            system=system_name,
            backend="native",
            distribution=None,
            python_version=(
                platform.python_version()
            ),
            pyscf_version=availability.version,
            basis_set_exchange_version=(
                _package_version(
                    "basis-set-exchange"
                )
            ),
            message=(
                "Native PySCF version mismatch: "
                f"{availability.version!r}; expected "
                f"{EXPECTED_PYSCF_VERSION!r}."
            ),
        )

    bse_version = _package_version(
        "basis-set-exchange"
    )

    if bse_version is None:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .NATIVE_BSE_MISSING
            ),
            system=system_name,
            backend="native",
            distribution=None,
            python_version=(
                platform.python_version()
            ),
            pyscf_version=availability.version,
            basis_set_exchange_version=None,
            message=(
                "basis-set-exchange is missing "
                "from the native compute environment."
            ),
        )

    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .READY_NATIVE
        ),
        system=system_name,
        backend="native",
        distribution=None,
        python_version=(
            platform.python_version()
        ),
        pyscf_version=availability.version,
        basis_set_exchange_version=(
            bse_version
        ),
        message=(
            "Native PySCF compute environment "
            "is ready."
        ),
    )


def _select_distribution(
    distributions: tuple[str, ...],
    requested: str | None,
) -> str | None:
    if requested is not None:
        if requested in distributions:
            return requested

        return None

    if (
        DEFAULT_WSL_DISTRIBUTION
        in distributions
    ):
        return DEFAULT_WSL_DISTRIBUTION

    if distributions:
        return distributions[0]

    return None


def _wsl_probe_python() -> str:
    """Return pure Python source for the WSL environment probe."""
    return """
import importlib.metadata as metadata
import json
import platform

import pyscf

payload = {
    "python_version": platform.python_version(),
    "pyscf_version": str(pyscf.__version__),
    "basis_set_exchange_version": metadata.version(
        "basis-set-exchange"
    ),
}

print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)
""".strip()


def _extract_payload(
    stdout: str,
) -> dict[str, object] | None:
    for line in reversed(
        stdout.splitlines()
    ):
        line = line.strip()

        if not line:
            continue

        try:
            payload = json.loads(
                line
            )
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            return payload

    return None


def _windows_report(
    *,
    distribution: str | None,
) -> ComputeEnvironmentReport:
    availability = inspect_wsl()

    if not availability.executable_found:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_MISSING
            ),
            system="Windows",
            backend="wsl",
            distribution=None,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "WSL is not installed. "
                "DiatomicEA requires WSL 2 "
                "for PySCF calculations on Windows."
            ),
        )

    if availability.timed_out:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_UNRESPONSIVE
            ),
            system="Windows",
            backend="wsl",
            distribution=None,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=availability.message,
        )

    if not availability.distributions:
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
            message=(
                "WSL is installed, but no Linux "
                "distribution is installed."
            ),
        )

    selected = _select_distribution(
        availability.distributions,
        distribution,
    )

    if selected is None:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_DISTRIBUTION_NOT_FOUND
            ),
            system="Windows",
            backend="wsl",
            distribution=distribution,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "Requested WSL distribution "
                f"{distribution!r} is not installed."
            ),
        )

    launch = run_wsl_command(
        ("true",),
        distribution=selected,
        timeout=20.0,
    )

    if not launch.succeeded:
        detail = (
            launch.stderr.strip()
            or launch.stdout.strip()
            or "unknown WSL launch error"
        )

        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_LAUNCH_FAILED
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "The Linux distribution could not "
                "be started. A Windows restart, "
                "Virtual Machine Platform, or "
                "firmware virtualization may need "
                "attention. Details: "
                + detail
            ),
        )

    executable_check = run_wsl_command(
        (
            "test",
            "-x",
            WSL_COMPUTE_PYTHON,
        ),
        distribution=selected,
        timeout=20.0,
    )

    if not executable_check.succeeded:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_ENV_MISSING
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "The dedicated DiatomicEA WSL "
                "compute environment is missing."
            ),
        )

    result = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _wsl_probe_python(),
        ),
        distribution=selected,
        timeout=60.0,
    )

    if not result.succeeded:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown environment error"
        )

        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_ENV_INVALID
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "The DiatomicEA WSL environment "
                "could not be imported: "
                + detail
            ),
        )

    payload = _extract_payload(
        result.stdout
    )

    if payload is None:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_ENV_INVALID
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=None,
            pyscf_version=None,
            basis_set_exchange_version=None,
            message=(
                "The WSL environment probe "
                "returned no valid payload."
            ),
        )

    python_version = str(
        payload.get(
            "python_version",
            "",
        )
    ) or None

    pyscf_version = str(
        payload.get(
            "pyscf_version",
            "",
        )
    ) or None

    bse_version = str(
        payload.get(
            "basis_set_exchange_version",
            "",
        )
    ) or None

    if (
        pyscf_version
        != EXPECTED_PYSCF_VERSION
    ):
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_PYSCF_VERSION_MISMATCH
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=python_version,
            pyscf_version=pyscf_version,
            basis_set_exchange_version=(
                bse_version
            ),
            message=(
                "WSL PySCF version mismatch: "
                f"{pyscf_version!r}; expected "
                f"{EXPECTED_PYSCF_VERSION!r}."
            ),
        )

    if bse_version is None:
        return ComputeEnvironmentReport(
            state=(
                ComputeEnvironmentState
                .WSL_BSE_MISSING
            ),
            system="Windows",
            backend="wsl",
            distribution=selected,
            python_version=python_version,
            pyscf_version=pyscf_version,
            basis_set_exchange_version=None,
            message=(
                "basis-set-exchange is missing "
                "from the WSL environment."
            ),
        )

    return ComputeEnvironmentReport(
        state=(
            ComputeEnvironmentState
            .READY_WSL
        ),
        system="Windows",
        backend="wsl",
        distribution=selected,
        python_version=python_version,
        pyscf_version=pyscf_version,
        basis_set_exchange_version=(
            bse_version
        ),
        message=(
            "DiatomicEA WSL compute environment "
            "is ready."
        ),
    )


def inspect_compute_environment(
    *,
    system_name: str | None = None,
    distribution: str | None = None,
) -> ComputeEnvironmentReport:
    """Inspect the platform-appropriate compute backend."""
    resolved_system = (
        system_name
        or platform.system()
    )

    if (
        resolved_system.casefold()
        == "windows"
    ):
        return _windows_report(
            distribution=distribution
        )

    return _native_report(
        resolved_system
    )


def main() -> int:
    """Print current compute-environment readiness."""
    report = inspect_compute_environment()

    print(
        "System:",
        report.system,
    )

    print(
        "Backend:",
        report.backend,
    )

    print(
        "State:",
        report.state.value,
    )

    print(
        "Distribution:",
        report.distribution
        or "n/a",
    )

    print(
        "Python:",
        report.python_version
        or "not available",
    )

    print(
        "PySCF:",
        report.pyscf_version
        or "not available",
    )

    print(
        "basis-set-exchange:",
        report.basis_set_exchange_version
        or "not available",
    )

    print(
        "Status:",
        report.message,
    )

    return (
        0
        if report.ready
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
