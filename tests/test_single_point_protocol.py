"""Tests for the versioned single-point transport protocol."""

import math

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
    SINGLE_POINT_PROTOCOL_VERSION,
    SinglePointProtocolError,
    dumps_result,
    dumps_task,
    loads_result,
    loads_task,
)
from diatomic_ea.states import (
    ChargeState,
)


def example_task() -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        charge=ChargeState.ANION,
        spin=1,
        functional="PBE0",
        basis="def2-tzvpd",
        bond_length_angstrom=1.625,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
        threads_per_worker=2,
    )


def example_result() -> SinglePointResult:
    task = example_task()

    return SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=-317.123456789,
        energy_ev=-8629.123456,
        converged=True,
        used_level_shift_retry=True,
        used_newton_retry=False,
        electron_count=20,
        alpha_electrons=11,
        beta_electrons=9,
        basis_label_a="def2-tzvpd",
        basis_label_b="def2-tzvpd",
        ecp_label_a="",
        ecp_label_b="",
        frontier=FrontierOrbitals(
            homo_hartree=-0.10,
            lumo_hartree=0.02,
            homo_ev=-2.72,
            lumo_ev=0.54,
            gap_ev=3.26,
            positive_homo_warning=False,
        ),
        s2=0.75,
        observed_multiplicity=2.0,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=1.25,
    )


def test_protocol_version_is_one() -> None:
    assert (
        SINGLE_POINT_PROTOCOL_VERSION
        == 1
    )


def test_task_round_trip_is_lossless() -> None:
    original = example_task()

    encoded = dumps_task(
        original
    )

    restored = loads_task(
        encoded
    )

    assert restored == original

    assert (
        restored.task_id
        == original.task_id
    )


def test_result_round_trip_is_lossless() -> None:
    original = example_result()

    encoded = dumps_result(
        original
    )

    restored = loads_result(
        encoded
    )

    assert restored == original


def test_error_result_preserves_nonfinite_values() -> None:
    task = example_task()

    original = SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.ERROR,
        error="synthetic failure",
        energy_hartree=math.nan,
        energy_ev=math.nan,
        converged=False,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=None,
        alpha_electrons=None,
        beta_electrons=None,
        basis_label_a="",
        basis_label_b="",
        ecp_label_a="",
        ecp_label_b="",
        frontier=None,
        s2=math.nan,
        observed_multiplicity=math.nan,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.10,
    )

    encoded = dumps_result(
        original
    )

    assert ":NaN" not in encoded
    assert ":Infinity" not in encoded

    restored = loads_result(
        encoded
    )

    assert (
        restored.status
        is SinglePointStatus.ERROR
    )

    assert (
        restored.error
        == "synthetic failure"
    )

    assert math.isnan(
        restored.energy_hartree
    )

    assert math.isnan(
        restored.energy_ev
    )

    assert math.isnan(
        restored.s2
    )

    assert math.isnan(
        restored.observed_multiplicity
    )


def test_frontier_nonfinite_values_round_trip() -> None:
    original = example_result()

    modified = SinglePointResult(
        task_id=original.task_id,
        status=original.status,
        error=original.error,
        energy_hartree=original.energy_hartree,
        energy_ev=original.energy_ev,
        converged=original.converged,
        used_level_shift_retry=(
            original.used_level_shift_retry
        ),
        used_newton_retry=(
            original.used_newton_retry
        ),
        electron_count=original.electron_count,
        alpha_electrons=original.alpha_electrons,
        beta_electrons=original.beta_electrons,
        basis_label_a=original.basis_label_a,
        basis_label_b=original.basis_label_b,
        ecp_label_a=original.ecp_label_a,
        ecp_label_b=original.ecp_label_b,
        frontier=FrontierOrbitals(
            homo_hartree=math.nan,
            lumo_hartree=math.inf,
            homo_ev=math.nan,
            lumo_ev=math.inf,
            gap_ev=math.inf,
            positive_homo_warning=False,
        ),
        s2=original.s2,
        observed_multiplicity=(
            original.observed_multiplicity
        ),
        spin_contamination_warning=(
            original.spin_contamination_warning
        ),
        pyscf_version=original.pyscf_version,
        elapsed_seconds=original.elapsed_seconds,
    )

    restored = loads_result(
        dumps_result(
            modified
        )
    )

    assert restored.frontier is not None

    assert math.isnan(
        restored.frontier.homo_hartree
    )

    assert (
        restored.frontier.lumo_hartree
        == math.inf
    )


def test_wrong_protocol_version_is_rejected() -> None:
    text = dumps_task(
        example_task()
    )

    text = text.replace(
        '"protocol_version":1',
        '"protocol_version":999',
    )

    with pytest.raises(
        SinglePointProtocolError,
        match="Unsupported protocol version",
    ):
        loads_task(
            text
        )


def test_wrong_payload_kind_is_rejected() -> None:
    text = dumps_task(
        example_task()
    )

    text = text.replace(
        '"single_point_task"',
        '"something_else"',
    )

    with pytest.raises(
        SinglePointProtocolError,
        match="Unexpected payload kind",
    ):
        loads_task(
            text
        )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(
        SinglePointProtocolError,
        match="Invalid task JSON",
    ):
        loads_task(
            "{broken"
        )


def test_missing_required_result_field_is_rejected() -> None:
    text = dumps_result(
        example_result()
    )

    import json

    payload = json.loads(
        text
    )

    del payload[
        "energy_hartree"
    ]

    malformed = json.dumps(
        payload
    )

    with pytest.raises(
        SinglePointProtocolError,
        match="Invalid single-point result payload",
    ):
        loads_result(
            malformed
        )


def test_non_boolean_diagnostic_is_rejected() -> None:
    import json

    payload = json.loads(
        dumps_result(
            example_result()
        )
    )

    payload[
        "converged"
    ] = "yes"

    with pytest.raises(
        SinglePointProtocolError,
        match="converged must be a boolean",
    ):
        loads_result(
            json.dumps(
                payload
            )
        )
