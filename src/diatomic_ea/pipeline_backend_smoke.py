"""Real multiprocessing validation for the platform compute worker."""

from __future__ import annotations

import math

from diatomic_ea.compute_environment import (
    EXPECTED_PYSCF_VERSION,
)
from diatomic_ea.executor import (
    execute_process_batch,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.single_point import (
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.single_point_adapter import (
    run_platform_single_point,
)
from diatomic_ea.states import (
    ChargeState,
)


def _tasks() -> tuple[
    SinglePointTask,
    ...,
]:
    molecule = DiatomicMolecule(
        "H",
        "F",
    )

    return (
        SinglePointTask(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            spin=0,
            functional="PBE",
            basis="def2-SVP",
            bond_length_angstrom=0.90,
            grid_level=0,
            conv_tol=1.0e-8,
            max_cycle=80,
            max_memory_mb=1000,
            threads_per_worker=1,
        ),
        SinglePointTask(
            molecule=molecule,
            charge=ChargeState.NEUTRAL,
            spin=0,
            functional="PBE",
            basis="def2-SVP",
            bond_length_angstrom=0.94,
            grid_level=0,
            conv_tol=1.0e-8,
            max_cycle=80,
            max_memory_mb=1000,
            threads_per_worker=1,
        ),
    )


def main() -> int:
    """Run two real tasks concurrently through the production worker."""
    tasks = _tasks()

    print()
    print(
        "DiatomicEA platform multiprocessing test"
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

    try:
        results = execute_process_batch(
            tasks,
            worker=run_platform_single_point,
            max_workers=2,
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
        len(results) == 2
        and all(
            result.status
            is SinglePointStatus.OK
            for result in results
        )
        and all(
            result.converged
            for result in results
        )
        and all(
            math.isfinite(
                result.energy_hartree
            )
            for result in results
        )
        and all(
            result.electron_count == 10
            for result in results
        )
        and all(
            result.pyscf_version
            == EXPECTED_PYSCF_VERSION
            for result in results
        )
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            "Task",
            index,
            ":",
            result.status.value,
            "| converged =",
            result.converged,
            "| energy / Ha =",
            result.energy_hartree,
            "| PySCF =",
            result.pyscf_version,
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

    return (
        0
        if passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
