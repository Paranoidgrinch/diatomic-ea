"""Platform adapter for real DiatomicEA single-point calculations.

Windows:
    SinglePointTask
        -> strict JSON stdin
        -> managed WSL Python
        -> diatomic_ea.single_point_worker
        -> run_pyscf_single_point
        -> strict JSON stdout
        -> SinglePointResult

Linux:
    SinglePointTask
        -> native run_pyscf_single_point
        -> SinglePointResult
"""

from __future__ import annotations

import math
import platform

from diatomic_ea.compute_environment import (
    DEFAULT_WSL_DISTRIBUTION,
    EXPECTED_PYSCF_VERSION,
    WSL_COMPUTE_PYTHON,
    inspect_compute_environment,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
    run_pyscf_single_point,
)
from diatomic_ea.single_point_protocol import (
    SinglePointProtocolError,
    dumps_task,
    loads_result,
)
from diatomic_ea.states import (
    ChargeState,
)
from diatomic_ea.wsl import (
    run_wsl_command,
)


class SinglePointExecutionError(RuntimeError):
    """Raised when task transport or backend execution fails."""


def _result_from_worker_stdout(
    stdout: str,
) -> SinglePointResult:
    """Extract the final protocol result from worker stdout."""
    for line in reversed(
        stdout.splitlines()
    ):
        candidate = line.strip()

        if not candidate:
            continue

        try:
            return loads_result(
                candidate
            )
        except SinglePointProtocolError:
            continue

    raise SinglePointExecutionError(
        "WSL single-point worker returned no "
        "valid SinglePointResult payload."
    )


def run_wsl_single_point(
    task: SinglePointTask,
    *,
    distribution: str | None = None,
    timeout: float = 900.0,
) -> SinglePointResult:
    """Execute one real task using the managed WSL worker."""
    environment = inspect_compute_environment(
        system_name="Windows",
        distribution=distribution,
    )

    if not environment.ready:
        raise SinglePointExecutionError(
            "WSL compute environment is not ready: "
            + environment.message
        )

    if environment.backend != "wsl":
        raise SinglePointExecutionError(
            "Windows single-point execution expected "
            "a WSL compute backend."
        )

    selected = environment.distribution

    if selected is None:
        raise SinglePointExecutionError(
            "No WSL distribution was selected."
        )

    task_json = dumps_task(
        task
    )

    result = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-m",
            "diatomic_ea.single_point_worker",
        ),
        distribution=selected,
        input_text=task_json,
        timeout=timeout,
    )

    if not result.succeeded:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown worker error"
        )

        raise SinglePointExecutionError(
            "WSL single-point worker failed: "
            + detail
        )

    restored = _result_from_worker_stdout(
        result.stdout
    )

    if restored.task_id != task.task_id:
        raise SinglePointExecutionError(
            "WSL worker returned a result for the "
            "wrong task. Expected "
            f"{task.task_id!r}, received "
            f"{restored.task_id!r}."
        )

    return restored


def run_platform_single_point(
    task: SinglePointTask,
    *,
    system_name: str | None = None,
    distribution: str | None = None,
    timeout: float = 900.0,
) -> SinglePointResult:
    """Execute one task using the correct platform backend."""
    resolved_system = (
        system_name
        or platform.system()
    )

    if (
        resolved_system.casefold()
        == "windows"
    ):
        return run_wsl_single_point(
            task,
            distribution=distribution,
            timeout=timeout,
        )

    return run_pyscf_single_point(
        task
    )


def real_hf_smoke_task() -> SinglePointTask:
    """Return a small real task for end-to-end adapter validation."""
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        charge=ChargeState.NEUTRAL,
        spin=0,
        functional="PBE",
        basis="def2-SVP",
        bond_length_angstrom=0.92,
        grid_level=1,
        conv_tol=1.0e-8,
        max_cycle=80,
        max_memory_mb=1000,
        threads_per_worker=1,
    )


def main() -> int:
    """Run one real HF task through the platform adapter."""
    task = real_hf_smoke_task()

    print()
    print(
        "DiatomicEA real single-point adapter test"
    )

    print(
        "========================================="
    )

    print()

    print(
        "BACKEND VALIDATION ONLY - "
        "NOT A SCIENTIFIC EA PREDICTION"
    )

    print()

    print(
        "Molecule:",
        task.molecule.formula,
    )

    print(
        "Charge:",
        int(
            task.charge
        ),
    )

    print(
        "Spin:",
        task.spin,
    )

    print(
        "Method:",
        task.functional,
        "/",
        task.basis,
    )

    print(
        "Bond length / Angstrom:",
        task.bond_length_angstrom,
    )

    print(
        "Task ID:",
        task.task_id,
    )

    try:
        result = run_platform_single_point(
            task,
            distribution=(
                DEFAULT_WSL_DISTRIBUTION
            ),
            timeout=300.0,
        )

    except Exception as exc:
        print()
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
        result.status
        is SinglePointStatus.OK
        and result.converged
        and math.isfinite(
            result.energy_hartree
        )
        and result.electron_count == 10
        and result.alpha_electrons == 5
        and result.beta_electrons == 5
        and result.pyscf_version
        == EXPECTED_PYSCF_VERSION
    )

    print()

    print(
        "Status:",
        (
            "PASS"
            if passed
            else "FAIL"
        ),
    )

    print(
        "Result status:",
        result.status.value,
    )

    print(
        "Converged:",
        result.converged,
    )

    print(
        "Energy / Ha:",
        result.energy_hartree,
    )

    print(
        "Energy / eV:",
        result.energy_ev,
    )

    print(
        "Electrons:",
        result.electron_count,
    )

    print(
        "Alpha / beta:",
        result.alpha_electrons,
        "/",
        result.beta_electrons,
    )

    print(
        "Level-shift rescue:",
        result.used_level_shift_retry,
    )

    print(
        "Newton rescue:",
        result.used_newton_retry,
    )

    print(
        "PySCF:",
        result.pyscf_version,
    )

    print(
        "Elapsed / s:",
        result.elapsed_seconds,
    )

    if result.frontier is not None:
        print(
            "HOMO / eV:",
            result.frontier.homo_ev,
        )

        print(
            "LUMO / eV:",
            result.frontier.lumo_ev,
        )

        print(
            "Gap / eV:",
            result.frontier.gap_ev,
        )

    print(
        "<S^2>:",
        result.s2,
    )

    print(
        "Observed multiplicity:",
        result.observed_multiplicity,
    )

    print(
        "Spin warning:",
        result.spin_contamination_warning,
    )

    if result.error:
        print(
            "Backend message:",
            result.error,
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
