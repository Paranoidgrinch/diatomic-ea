"""Platform-aware electron-count resolution.

Windows uses the managed WSL PySCF environment.
Linux and other supported native platforms call PySCF directly.
"""

from __future__ import annotations

import json
import platform

from diatomic_ea.compute_environment import (
    DEFAULT_WSL_DISTRIBUTION,
    WSL_COMPUTE_PYTHON,
    inspect_compute_environment,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.single_point import (
    determine_electron_count,
)
from diatomic_ea.states import (
    ChargeState,
)
from diatomic_ea.wsl import (
    run_wsl_command,
)


class ElectronCountExecutionError(RuntimeError):
    """Raised when electron-count resolution cannot be completed."""


def _wsl_electron_count_program() -> str:
    """Return pure Python source for one WSL electron-count request."""
    return """
import json
import sys

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.single_point import determine_electron_count
from diatomic_ea.states import ChargeState

payload = json.loads(
    sys.stdin.read()
)

electron_count = determine_electron_count(
    molecule=DiatomicMolecule(
        str(
            payload["atom_a"]
        ),
        str(
            payload["atom_b"]
        ),
    ),
    charge=ChargeState(
        int(
            payload["charge"]
        )
    ),
    basis=str(
        payload["basis"]
    ),
    bond_length_angstrom=float(
        payload[
            "bond_length_angstrom"
        ]
    ),
    max_memory_mb=int(
        payload[
            "max_memory_mb"
        ]
    ),
)

print(
    json.dumps(
        {
            "electron_count": (
                electron_count
            )
        },
        sort_keys=True,
    )
)
""".strip()


def _extract_electron_count(
    stdout: str,
) -> int:
    """Extract one positive electron count from worker output."""
    for line in reversed(
        stdout.splitlines()
    ):
        candidate = line.strip()

        if not candidate:
            continue

        try:
            payload = json.loads(
                candidate
            )
        except json.JSONDecodeError:
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        try:
            value = int(
                payload[
                    "electron_count"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if value < 1:
            raise ElectronCountExecutionError(
                "WSL electron-count worker returned "
                f"an invalid count: {value}."
            )

        return value

    raise ElectronCountExecutionError(
        "WSL electron-count worker returned "
        "no valid result payload."
    )


def run_wsl_electron_count(
    *,
    molecule: DiatomicMolecule,
    charge: ChargeState,
    basis: str,
    bond_length_angstrom: float,
    max_memory_mb: int,
    distribution: str | None = None,
    timeout: float = 120.0,
) -> int:
    """Resolve one post-ECP electron count in managed WSL PySCF."""
    environment = inspect_compute_environment(
        system_name="Windows",
        distribution=distribution,
    )

    if not environment.ready:
        raise ElectronCountExecutionError(
            "WSL compute environment is not ready: "
            + environment.message
        )

    if environment.backend != "wsl":
        raise ElectronCountExecutionError(
            "Windows electron-count resolution "
            "expected a WSL compute backend."
        )

    selected = environment.distribution

    if selected is None:
        raise ElectronCountExecutionError(
            "No WSL distribution was selected."
        )

    payload = json.dumps(
        {
            "atom_a": molecule.atom_a,
            "atom_b": molecule.atom_b,
            "charge": int(
                charge
            ),
            "basis": basis,
            "bond_length_angstrom": (
                bond_length_angstrom
            ),
            "max_memory_mb": (
                max_memory_mb
            ),
        },
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    )

    result = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _wsl_electron_count_program(),
        ),
        distribution=selected,
        input_text=payload,
        timeout=timeout,
    )

    if not result.succeeded:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown electron-count error"
        )

        raise ElectronCountExecutionError(
            "WSL electron-count worker failed: "
            + detail
        )

    return _extract_electron_count(
        result.stdout
    )


def run_platform_electron_count(
    *,
    molecule: DiatomicMolecule,
    charge: ChargeState,
    basis: str,
    bond_length_angstrom: float,
    max_memory_mb: int,
    system_name: str | None = None,
    distribution: str | None = None,
) -> int:
    """Resolve electron count using the platform-appropriate backend."""
    resolved_system = (
        system_name
        or platform.system()
    )

    if (
        resolved_system.casefold()
        == "windows"
    ):
        return run_wsl_electron_count(
            molecule=molecule,
            charge=charge,
            basis=basis,
            bond_length_angstrom=(
                bond_length_angstrom
            ),
            max_memory_mb=max_memory_mb,
            distribution=distribution,
        )

    return determine_electron_count(
        molecule=molecule,
        charge=charge,
        basis=basis,
        bond_length_angstrom=(
            bond_length_angstrom
        ),
        max_memory_mb=max_memory_mb,
    )


def main() -> int:
    """Run a real lightweight electron-count backend validation."""
    molecule = DiatomicMolecule(
        "H",
        "F",
    )

    print()
    print(
        "DiatomicEA electron-count backend test"
    )

    print(
        "====================================="
    )

    print()

    print(
        "BACKEND VALIDATION ONLY - "
        "NOT A SCIENTIFIC EA PREDICTION"
    )

    print()

    try:
        neutral = run_platform_electron_count(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            basis="def2-SVP",
            bond_length_angstrom=0.92,
            max_memory_mb=1000,
        )

        anion = run_platform_electron_count(
            molecule=molecule,
            charge=ChargeState.ANION,
            basis="def2-SVP",
            bond_length_angstrom=0.92,
            max_memory_mb=1000,
        )

    except Exception as exc:
        print(
            "Status: FAIL"
        )

        print(
            "Error:",
            str(
                exc
            ),
        )

        return 1

    passed = (
        neutral == 10
        and anion == 11
    )

    print(
        "Molecule:",
        molecule.formula,
    )

    print(
        "Neutral electrons:",
        neutral,
    )

    print(
        "Anion electrons:",
        anion,
    )

    print(
        "Status:",
        (
            "PASS"
            if passed
            else "FAIL"
        ),
    )

    return (
        0
        if passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
