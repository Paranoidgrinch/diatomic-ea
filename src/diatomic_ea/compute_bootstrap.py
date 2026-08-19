"""Bootstrap the dedicated DiatomicEA compute environment in WSL."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass

from diatomic_ea.compute_environment import (
    DEFAULT_WSL_DISTRIBUTION,
    EXPECTED_PYSCF_VERSION,
    WSL_COMPUTE_ROOT,
    WSL_COMPUTE_VENV,
    ComputeEnvironmentReport,
    ComputeEnvironmentState,
    inspect_compute_environment,
)
from diatomic_ea.wsl import (
    run_wsl_command,
    run_wsl_shell,
)


@dataclass(frozen=True, slots=True)
class ComputeBootstrapResult:
    """Result of preparing the Windows WSL compute backend."""

    success: bool
    already_ready: bool
    distribution: str
    message: str
    environment: ComputeEnvironmentReport | None


_BOOTSTRAPPABLE_STATES = {
    ComputeEnvironmentState.WSL_ENV_MISSING,
    ComputeEnvironmentState.WSL_ENV_INVALID,
    ComputeEnvironmentState.WSL_PYSCF_VERSION_MISMATCH,
    ComputeEnvironmentState.WSL_BSE_MISSING,
}


def _failure_detail(result) -> str:
    return (
        result.stderr.strip()
        or result.stdout.strip()
        or "unknown error"
    )


def bootstrap_wsl_compute_environment(
    *,
    distribution: str = DEFAULT_WSL_DISTRIBUTION,
) -> ComputeBootstrapResult:
    """Install Linux prerequisites and the pinned PySCF environment."""
    before = inspect_compute_environment(
        system_name="Windows",
        distribution=distribution,
    )

    if before.ready:
        return ComputeBootstrapResult(
            success=True,
            already_ready=True,
            distribution=distribution,
            message=(
                "DiatomicEA compute environment "
                "is already ready."
            ),
            environment=before,
        )

    if (
        before.state
        not in _BOOTSTRAPPABLE_STATES
    ):
        return ComputeBootstrapResult(
            success=False,
            already_ready=False,
            distribution=distribution,
            message=(
                "The WSL compute environment cannot "
                "be bootstrapped yet: "
                + before.message
            ),
            environment=before,
        )

    apt_update = run_wsl_command(
        (
            "apt-get",
            "update",
        ),
        distribution=distribution,
        user="root",
        timeout=600.0,
    )

    if not apt_update.succeeded:
        return ComputeBootstrapResult(
            success=False,
            already_ready=False,
            distribution=distribution,
            message=(
                "Could not update Linux package metadata: "
                + _failure_detail(
                    apt_update
                )
            ),
            environment=None,
        )

    apt_install = run_wsl_command(
        (
            "apt-get",
            "install",
            "-y",
            "python3-venv",
            "python3-pip",
        ),
        distribution=distribution,
        user="root",
        timeout=900.0,
    )

    if not apt_install.succeeded:
        return ComputeBootstrapResult(
            success=False,
            already_ready=False,
            distribution=distribution,
            message=(
                "Could not install Linux Python prerequisites: "
                + _failure_detail(
                    apt_install
                )
            ),
            environment=None,
        )

    pyscf_spec = shlex.quote(
        "pyscf=="
        + EXPECTED_PYSCF_VERSION
    )

    root_path = shlex.quote(
        WSL_COMPUTE_ROOT
    )

    venv_path = shlex.quote(
        WSL_COMPUTE_VENV
    )

    venv_python = shlex.quote(
        WSL_COMPUTE_VENV
        + "/bin/python"
    )

    setup_command = " ".join(
        (
            "set -eu;",
            "install -d -m 0755",
            root_path + ";",
            "python3 -m venv",
            venv_path + ";",
            venv_python,
            "-m pip install --upgrade pip;",
            venv_python,
            "-m pip install --prefer-binary",
            pyscf_spec,
            shlex.quote(
                "basis-set-exchange"
            ),
        )
    )

    setup = run_wsl_shell(
        setup_command,
        distribution=distribution,
        user="root",
        timeout=1200.0,
    )

    if not setup.succeeded:
        return ComputeBootstrapResult(
            success=False,
            already_ready=False,
            distribution=distribution,
            message=(
                "Could not create the DiatomicEA "
                "Python environment: "
                + _failure_detail(
                    setup
                )
            ),
            environment=None,
        )

    after = inspect_compute_environment(
        system_name="Windows",
        distribution=distribution,
    )

    if not after.ready:
        return ComputeBootstrapResult(
            success=False,
            already_ready=False,
            distribution=distribution,
            message=(
                "Environment installation completed, "
                "but validation failed: "
                + after.message
            ),
            environment=after,
        )

    return ComputeBootstrapResult(
        success=True,
        already_ready=False,
        distribution=distribution,
        message=(
            "DiatomicEA WSL compute environment "
            "was installed successfully."
        ),
        environment=after,
    )


def main() -> int:
    """Install or validate the Windows WSL compute environment."""
    parser = argparse.ArgumentParser(
        description=(
            "Install or validate the dedicated "
            "DiatomicEA WSL compute environment."
        )
    )

    parser.add_argument(
        "--distribution",
        default=DEFAULT_WSL_DISTRIBUTION,
    )

    args = parser.parse_args()

    result = bootstrap_wsl_compute_environment(
        distribution=args.distribution
    )

    print(
        "Distribution:",
        result.distribution,
    )

    print(
        "Already ready:",
        result.already_ready,
    )

    print(
        "Status:",
        result.message,
    )

    if result.environment is not None:
        print(
            "Python:",
            result.environment.python_version
            or "not available",
        )

        print(
            "PySCF:",
            result.environment.pyscf_version
            or "not available",
        )

        print(
            "basis-set-exchange:",
            result.environment.basis_set_exchange_version
            or "not available",
        )

    return (
        0
        if result.success
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
