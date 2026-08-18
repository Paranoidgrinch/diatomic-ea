"""Quantum-chemistry backend interfaces.

PySCF is loaded lazily so that the DiatomicEA GUI and project tools
remain usable on systems where the compute backend is unavailable.
"""

from __future__ import annotations

import importlib.util
import math
import platform
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Availability of a quantum-chemistry backend."""

    backend: str
    platform_supported: bool
    installed: bool
    version: str | None
    message: str

    @property
    def ready(self) -> bool:
        return (
            self.platform_supported
            and self.installed
        )


@dataclass(frozen=True, slots=True)
class BackendSmokeReport:
    """Result of a small backend calculation."""

    backend: str
    passed: bool
    message: str
    energy_hartree: float | None = None


def native_windows_supported(
    system_name: str,
) -> bool:
    """Return whether native PySCF execution is supported."""
    return system_name.lower() != "windows"


class PySCFBackend:
    """Lazy interface to the reference PySCF backend."""

    name = "PySCF"

    def availability(self) -> BackendAvailability:
        """Inspect the current environment without running a calculation."""
        system_name = platform.system()

        if not native_windows_supported(
            system_name
        ):
            return BackendAvailability(
                backend=self.name,
                platform_supported=False,
                installed=False,
                version=None,
                message=(
                    "PySCF is not supported natively on Windows. "
                    "Use the DiatomicEA compute environment in WSL."
                ),
            )

        if importlib.util.find_spec("pyscf") is None:
            return BackendAvailability(
                backend=self.name,
                platform_supported=True,
                installed=False,
                version=None,
                message=(
                    "PySCF is not installed. Install the "
                    "DiatomicEA compute extra."
                ),
            )

        try:
            import pyscf  # type: ignore
        except Exception as exc:
            return BackendAvailability(
                backend=self.name,
                platform_supported=True,
                installed=False,
                version=None,
                message=(
                    "PySCF was found but could not be imported: "
                    f"{exc}"
                ),
            )

        return BackendAvailability(
            backend=self.name,
            platform_supported=True,
            installed=True,
            version=str(pyscf.__version__),
            message="PySCF is available.",
        )

    def smoke_test(self) -> BackendSmokeReport:
        """Run a tiny UKS/PBE calculation.

        This is an installation test only. It is not a scientific
        electron-affinity calculation and is not part of Schema F.
        """
        availability = self.availability()

        if not availability.ready:
            return BackendSmokeReport(
                backend=self.name,
                passed=False,
                message=availability.message,
            )

        try:
            from pyscf import dft, gto, lib  # type: ignore

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

            energy = float(mf.kernel())

            if not mf.converged:
                return BackendSmokeReport(
                    backend=self.name,
                    passed=False,
                    message=(
                        "The PySCF smoke calculation "
                        "did not converge."
                    ),
                    energy_hartree=energy,
                )

            if not math.isfinite(energy):
                return BackendSmokeReport(
                    backend=self.name,
                    passed=False,
                    message=(
                        "The PySCF smoke calculation "
                        "returned a non-finite energy."
                    ),
                    energy_hartree=energy,
                )

        except Exception as exc:
            return BackendSmokeReport(
                backend=self.name,
                passed=False,
                message=(
                    "PySCF smoke calculation failed: "
                    f"{exc}"
                ),
            )

        return BackendSmokeReport(
            backend=self.name,
            passed=True,
            message=(
                "Tiny UKS/PBE smoke calculation "
                "completed successfully."
            ),
            energy_hartree=energy,
        )