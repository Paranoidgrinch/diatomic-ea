"""Real platform-aware PySCF backend smoke test.

The calculation validates the compute backend only.
It is not a Schema F electron-affinity prediction.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass

from diatomic_ea.backend import PySCFBackend
from diatomic_ea.compute_environment import (
    EXPECTED_PYSCF_VERSION,
    WSL_COMPUTE_PYTHON,
    inspect_compute_environment,
)
from diatomic_ea.wsl import run_wsl_command


@dataclass(frozen=True, slots=True)
class ComputeSmokeReport:
    """Result of one real compute-backend smoke test."""

    passed: bool
    backend: str
    distribution: str | None
    pyscf_version: str | None
    converged: bool | None
    energy_hartree: float | None
    message: str


def _wsl_smoke_program() -> str:
    """Return pure Python source for the WSL PySCF smoke calculation."""
    return """
import json
import math
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pyscf
from pyscf import dft, gto, lib

lib.num_threads(1)

mol = gto.Mole()
mol.atom = (
    "H 0.0 0.0 0.0; "
    "H 0.0 0.0 0.74"
)
mol.unit = "Angstrom"
mol.charge = 0
mol.spin = 0
mol.basis = "sto-3g"
mol.verbose = 0
mol.build()

mf = dft.UKS(mol)
mf.xc = "PBE"
mf.grids.level = 0
mf.conv_tol = 1.0e-8
mf.max_cycle = 50
mf.verbose = 0

energy = float(
    mf.kernel()
)

payload = {
    "pyscf_version": str(
        pyscf.__version__
    ),
    "converged": bool(
        mf.converged
    ),
    "energy_hartree": energy,
    "finite": bool(
        math.isfinite(
            energy
        )
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

        if isinstance(
            payload,
            dict,
        ):
            return payload

    return None


def _native_smoke() -> ComputeSmokeReport:
    backend = PySCFBackend()

    availability = (
        backend.availability()
    )

    if not availability.ready:
        return ComputeSmokeReport(
            passed=False,
            backend="native",
            distribution=None,
            pyscf_version=availability.version,
            converged=None,
            energy_hartree=None,
            message=availability.message,
        )

    if (
        availability.version
        != EXPECTED_PYSCF_VERSION
    ):
        return ComputeSmokeReport(
            passed=False,
            backend="native",
            distribution=None,
            pyscf_version=availability.version,
            converged=None,
            energy_hartree=None,
            message=(
                "Native PySCF version mismatch: "
                f"{availability.version!r}; expected "
                f"{EXPECTED_PYSCF_VERSION!r}."
            ),
        )

    smoke = backend.smoke_test()

    return ComputeSmokeReport(
        passed=smoke.passed,
        backend="native",
        distribution=None,
        pyscf_version=availability.version,
        converged=(
            True
            if smoke.passed
            else None
        ),
        energy_hartree=(
            smoke.energy_hartree
        ),
        message=smoke.message,
    )


def _wsl_smoke(
    distribution: str | None,
) -> ComputeSmokeReport:
    environment = (
        inspect_compute_environment(
            system_name="Windows",
            distribution=distribution,
        )
    )

    if not environment.ready:
        return ComputeSmokeReport(
            passed=False,
            backend="wsl",
            distribution=environment.distribution,
            pyscf_version=environment.pyscf_version,
            converged=None,
            energy_hartree=None,
            message=environment.message,
        )

    selected = (
        environment.distribution
    )

    result = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _wsl_smoke_program(),
        ),
        distribution=selected,
        timeout=180.0,
    )

    if not result.succeeded:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown PySCF error"
        )

        return ComputeSmokeReport(
            passed=False,
            backend="wsl",
            distribution=selected,
            pyscf_version=environment.pyscf_version,
            converged=None,
            energy_hartree=None,
            message=(
                "WSL PySCF smoke calculation "
                "failed: "
                + detail
            ),
        )

    payload = _extract_payload(
        result.stdout
    )

    if payload is None:
        return ComputeSmokeReport(
            passed=False,
            backend="wsl",
            distribution=selected,
            pyscf_version=None,
            converged=None,
            energy_hartree=None,
            message=(
                "WSL PySCF smoke calculation "
                "returned no valid JSON payload."
            ),
        )

    pyscf_version = str(
        payload.get(
            "pyscf_version",
            "",
        )
    ) or None

    converged = bool(
        payload.get(
            "converged",
            False,
        )
    )

    try:
        energy = float(
            payload.get(
                "energy_hartree"
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        energy = math.nan

    finite = (
        bool(
            payload.get(
                "finite",
                False,
            )
        )
        and math.isfinite(
            energy
        )
    )

    version_ok = (
        pyscf_version
        == EXPECTED_PYSCF_VERSION
    )

    passed = (
        version_ok
        and converged
        and finite
    )

    if passed:
        message = (
            "Real WSL PySCF UKS/PBE "
            "smoke calculation passed."
        )

    elif not version_ok:
        message = (
            "PySCF version mismatch during "
            "the WSL smoke calculation."
        )

    elif not converged:
        message = (
            "The WSL PySCF smoke calculation "
            "did not converge."
        )

    else:
        message = (
            "The WSL PySCF smoke calculation "
            "returned a non-finite energy."
        )

    return ComputeSmokeReport(
        passed=passed,
        backend="wsl",
        distribution=selected,
        pyscf_version=pyscf_version,
        converged=converged,
        energy_hartree=(
            energy
            if math.isfinite(
                energy
            )
            else None
        ),
        message=message,
    )


def run_compute_smoke(
    *,
    system_name: str | None = None,
    distribution: str | None = None,
) -> ComputeSmokeReport:
    """Run a real smoke calculation using the platform backend."""
    resolved_system = (
        system_name
        or platform.system()
    )

    if (
        resolved_system.casefold()
        == "windows"
    ):
        return _wsl_smoke(
            distribution
        )

    return _native_smoke()


def main() -> int:
    """Run and print the real backend smoke test."""
    report = run_compute_smoke()

    print()
    print(
        "DiatomicEA real compute smoke test"
    )

    print(
        "================================="
    )

    print()
    print(
        "BACKEND VALIDATION ONLY - "
        "NOT A SCIENTIFIC EA PREDICTION"
    )
    print()

    print(
        "Status:",
        (
            "PASS"
            if report.passed
            else "FAIL"
        ),
    )

    print(
        "Backend:",
        report.backend,
    )

    print(
        "Distribution:",
        report.distribution
        or "n/a",
    )

    print(
        "PySCF:",
        report.pyscf_version
        or "not available",
    )

    print(
        "Converged:",
        report.converged,
    )

    print(
        "Energy / Ha:",
        report.energy_hartree,
    )

    print(
        "Message:",
        report.message,
    )

    return (
        0
        if report.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
