"""Tests for the single-point stdin/stdout worker."""

import math
from unittest.mock import patch

import pytest

from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.single_point_protocol import (
    SinglePointProtocolError,
    dumps_task,
    loads_result,
)
from diatomic_ea.single_point_worker import (
    execute_task_json,
)
from diatomic_ea.states import (
    ChargeState,
)


def example_task() -> SinglePointTask:
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
        max_cycle=50,
        max_memory_mb=1000,
        threads_per_worker=1,
    )


def example_result(
    task: SinglePointTask,
) -> SinglePointResult:
    return SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=-100.123,
        energy_ev=-2724.0,
        converged=True,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=10,
        alpha_electrons=5,
        beta_electrons=5,
        basis_label_a="def2-SVP",
        basis_label_b="def2-SVP",
        ecp_label_a="",
        ecp_label_b="",
        frontier=FrontierOrbitals(
            homo_hartree=-0.3,
            lumo_hartree=0.1,
            homo_ev=-8.16,
            lumo_ev=2.72,
            gap_ev=10.88,
            positive_homo_warning=False,
        ),
        s2=0.0,
        observed_multiplicity=1.0,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.25,
    )


def test_worker_executes_deserialized_task() -> None:
    task = example_task()
    expected = example_result(
        task
    )

    with patch(
        "diatomic_ea.single_point_worker.run_pyscf_single_point",
        return_value=expected,
    ) as runner:
        output = execute_task_json(
            dumps_task(
                task
            )
        )

    runner.assert_called_once_with(
        task
    )

    restored = loads_result(
        output
    )

    assert restored == expected


def test_worker_rejects_empty_input() -> None:
    with pytest.raises(
        SinglePointProtocolError,
        match="empty input",
    ):
        execute_task_json(
            ""
        )


def test_worker_rejects_invalid_json() -> None:
    with pytest.raises(
        SinglePointProtocolError,
        match="Invalid task JSON",
    ):
        execute_task_json(
            "{broken"
        )


def test_worker_preserves_error_result() -> None:
    task = example_task()

    failed = SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.ERROR,
        error="synthetic SCF failure",
        energy_hartree=math.nan,
        energy_ev=math.nan,
        converged=False,
        used_level_shift_retry=True,
        used_newton_retry=True,
        electron_count=None,
        alpha_electrons=None,
        beta_electrons=None,
        basis_label_a="def2-SVP",
        basis_label_b="def2-SVP",
        ecp_label_a="",
        ecp_label_b="",
        frontier=None,
        s2=math.nan,
        observed_multiplicity=math.nan,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=2.0,
    )

    with patch(
        "diatomic_ea.single_point_worker.run_pyscf_single_point",
        return_value=failed,
    ):
        restored = loads_result(
            execute_task_json(
                dumps_task(
                    task
                )
            )
        )

    assert (
        restored.status
        is SinglePointStatus.ERROR
    )

    assert (
        restored.error
        == "synthetic SCF failure"
    )

    assert math.isnan(
        restored.energy_hartree
    )
