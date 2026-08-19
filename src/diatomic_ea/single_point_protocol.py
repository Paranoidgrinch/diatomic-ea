"""Versioned JSON transport for single-point tasks and results."""

from __future__ import annotations

import json
import math
from typing import Any

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.states import ChargeState


SINGLE_POINT_PROTOCOL_VERSION = 1


class SinglePointProtocolError(ValueError):
    """Raised when a transport payload is malformed or unsupported."""


def _encode_float(
    value: float,
) -> float | str:
    """Encode non-finite floats without non-standard JSON constants."""
    resolved = float(
        value
    )

    if math.isnan(
        resolved
    ):
        return "nan"

    if resolved == math.inf:
        return "inf"

    if resolved == -math.inf:
        return "-inf"

    return resolved


def _decode_float(
    value: object,
) -> float:
    """Decode one transport floating-point value."""
    if value == "nan":
        return math.nan

    if value == "inf":
        return math.inf

    if value == "-inf":
        return -math.inf

    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SinglePointProtocolError(
            "Invalid floating-point value: "
            f"{value!r}."
        ) from exc


def _require_bool(
    value: object,
    *,
    field: str,
) -> bool:
    if not isinstance(
        value,
        bool,
    ):
        raise SinglePointProtocolError(
            f"{field} must be a boolean."
        )

    return value


def _validate_envelope(
    payload: object,
    *,
    expected_kind: str,
) -> dict[str, Any]:
    """Validate the common protocol envelope."""
    if not isinstance(
        payload,
        dict,
    ):
        raise SinglePointProtocolError(
            "Protocol payload must be a JSON object."
        )

    version = payload.get(
        "protocol_version"
    )

    if (
        version
        != SINGLE_POINT_PROTOCOL_VERSION
    ):
        raise SinglePointProtocolError(
            "Unsupported protocol version: "
            f"{version!r}."
        )

    kind = payload.get(
        "kind"
    )

    if kind != expected_kind:
        raise SinglePointProtocolError(
            "Unexpected payload kind: "
            f"{kind!r}; expected "
            f"{expected_kind!r}."
        )

    return payload


def task_to_payload(
    task: SinglePointTask,
) -> dict[str, Any]:
    """Convert one single-point task to JSON-compatible data."""
    return {
        "protocol_version": (
            SINGLE_POINT_PROTOCOL_VERSION
        ),
        "kind": "single_point_task",
        "molecule": {
            "atom_a": task.molecule.atom_a,
            "atom_b": task.molecule.atom_b,
        },
        "charge": int(
            task.charge
        ),
        "spin": task.spin,
        "functional": task.functional,
        "basis": task.basis,
        "bond_length_angstrom": (
            task.bond_length_angstrom
        ),
        "grid_level": task.grid_level,
        "conv_tol": task.conv_tol,
        "max_cycle": task.max_cycle,
        "max_memory_mb": (
            task.max_memory_mb
        ),
        "threads_per_worker": (
            task.threads_per_worker
        ),
    }


def task_from_payload(
    payload: object,
) -> SinglePointTask:
    """Reconstruct and validate one single-point task."""
    data = _validate_envelope(
        payload,
        expected_kind="single_point_task",
    )

    molecule_data = data.get(
        "molecule"
    )

    if not isinstance(
        molecule_data,
        dict,
    ):
        raise SinglePointProtocolError(
            "Task molecule must be a JSON object."
        )

    try:
        return SinglePointTask(
            molecule=DiatomicMolecule(
                atom_a=str(
                    molecule_data[
                        "atom_a"
                    ]
                ),
                atom_b=str(
                    molecule_data[
                        "atom_b"
                    ]
                ),
            ),
            charge=ChargeState(
                int(
                    data[
                        "charge"
                    ]
                )
            ),
            spin=int(
                data[
                    "spin"
                ]
            ),
            functional=str(
                data[
                    "functional"
                ]
            ),
            basis=str(
                data[
                    "basis"
                ]
            ),
            bond_length_angstrom=float(
                data[
                    "bond_length_angstrom"
                ]
            ),
            grid_level=int(
                data[
                    "grid_level"
                ]
            ),
            conv_tol=float(
                data[
                    "conv_tol"
                ]
            ),
            max_cycle=int(
                data[
                    "max_cycle"
                ]
            ),
            max_memory_mb=int(
                data[
                    "max_memory_mb"
                ]
            ),
            threads_per_worker=int(
                data[
                    "threads_per_worker"
                ]
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise SinglePointProtocolError(
            "Invalid single-point task payload."
        ) from exc


def _frontier_to_payload(
    frontier: FrontierOrbitals | None,
) -> dict[str, Any] | None:
    if frontier is None:
        return None

    return {
        "homo_hartree": _encode_float(
            frontier.homo_hartree
        ),
        "lumo_hartree": _encode_float(
            frontier.lumo_hartree
        ),
        "homo_ev": _encode_float(
            frontier.homo_ev
        ),
        "lumo_ev": _encode_float(
            frontier.lumo_ev
        ),
        "gap_ev": _encode_float(
            frontier.gap_ev
        ),
        "positive_homo_warning": (
            frontier.positive_homo_warning
        ),
    }


def _frontier_from_payload(
    payload: object,
) -> FrontierOrbitals | None:
    if payload is None:
        return None

    if not isinstance(
        payload,
        dict,
    ):
        raise SinglePointProtocolError(
            "Result frontier must be a JSON object or null."
        )

    try:
        return FrontierOrbitals(
            homo_hartree=_decode_float(
                payload[
                    "homo_hartree"
                ]
            ),
            lumo_hartree=_decode_float(
                payload[
                    "lumo_hartree"
                ]
            ),
            homo_ev=_decode_float(
                payload[
                    "homo_ev"
                ]
            ),
            lumo_ev=_decode_float(
                payload[
                    "lumo_ev"
                ]
            ),
            gap_ev=_decode_float(
                payload[
                    "gap_ev"
                ]
            ),
            positive_homo_warning=(
                _require_bool(
                    payload[
                        "positive_homo_warning"
                    ],
                    field=(
                        "positive_homo_warning"
                    ),
                )
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(
            exc,
            SinglePointProtocolError,
        ):
            raise

        raise SinglePointProtocolError(
            "Invalid frontier-orbital payload."
        ) from exc


def result_to_payload(
    result: SinglePointResult,
) -> dict[str, Any]:
    """Convert one single-point result to JSON-compatible data."""
    return {
        "protocol_version": (
            SINGLE_POINT_PROTOCOL_VERSION
        ),
        "kind": "single_point_result",
        "task_id": result.task_id,
        "status": result.status.value,
        "error": result.error,
        "energy_hartree": _encode_float(
            result.energy_hartree
        ),
        "energy_ev": _encode_float(
            result.energy_ev
        ),
        "converged": result.converged,
        "used_level_shift_retry": (
            result.used_level_shift_retry
        ),
        "used_newton_retry": (
            result.used_newton_retry
        ),
        "electron_count": (
            result.electron_count
        ),
        "alpha_electrons": (
            result.alpha_electrons
        ),
        "beta_electrons": (
            result.beta_electrons
        ),
        "basis_label_a": (
            result.basis_label_a
        ),
        "basis_label_b": (
            result.basis_label_b
        ),
        "ecp_label_a": (
            result.ecp_label_a
        ),
        "ecp_label_b": (
            result.ecp_label_b
        ),
        "frontier": _frontier_to_payload(
            result.frontier
        ),
        "s2": _encode_float(
            result.s2
        ),
        "observed_multiplicity": (
            _encode_float(
                result.observed_multiplicity
            )
        ),
        "spin_contamination_warning": (
            result.spin_contamination_warning
        ),
        "pyscf_version": (
            result.pyscf_version
        ),
        "elapsed_seconds": (
            _encode_float(
                result.elapsed_seconds
            )
        ),
    }


def result_from_payload(
    payload: object,
) -> SinglePointResult:
    """Reconstruct and validate one single-point result."""
    data = _validate_envelope(
        payload,
        expected_kind="single_point_result",
    )

    try:
        status = SinglePointStatus(
            str(
                data[
                    "status"
                ]
            )
        )

        frontier = _frontier_from_payload(
            data[
                "frontier"
            ]
        )

        converged = _require_bool(
            data[
                "converged"
            ],
            field="converged",
        )

        level_shift = _require_bool(
            data[
                "used_level_shift_retry"
            ],
            field="used_level_shift_retry",
        )

        newton = _require_bool(
            data[
                "used_newton_retry"
            ],
            field="used_newton_retry",
        )

        spin_warning = _require_bool(
            data[
                "spin_contamination_warning"
            ],
            field=(
                "spin_contamination_warning"
            ),
        )

        electron_count = (
            None
            if data[
                "electron_count"
            ] is None
            else int(
                data[
                    "electron_count"
                ]
            )
        )

        alpha_electrons = (
            None
            if data[
                "alpha_electrons"
            ] is None
            else int(
                data[
                    "alpha_electrons"
                ]
            )
        )

        beta_electrons = (
            None
            if data[
                "beta_electrons"
            ] is None
            else int(
                data[
                    "beta_electrons"
                ]
            )
        )

        return SinglePointResult(
            task_id=str(
                data[
                    "task_id"
                ]
            ),
            status=status,
            error=str(
                data[
                    "error"
                ]
            ),
            energy_hartree=_decode_float(
                data[
                    "energy_hartree"
                ]
            ),
            energy_ev=_decode_float(
                data[
                    "energy_ev"
                ]
            ),
            converged=converged,
            used_level_shift_retry=(
                level_shift
            ),
            used_newton_retry=(
                newton
            ),
            electron_count=electron_count,
            alpha_electrons=alpha_electrons,
            beta_electrons=beta_electrons,
            basis_label_a=str(
                data[
                    "basis_label_a"
                ]
            ),
            basis_label_b=str(
                data[
                    "basis_label_b"
                ]
            ),
            ecp_label_a=str(
                data[
                    "ecp_label_a"
                ]
            ),
            ecp_label_b=str(
                data[
                    "ecp_label_b"
                ]
            ),
            frontier=frontier,
            s2=_decode_float(
                data[
                    "s2"
                ]
            ),
            observed_multiplicity=(
                _decode_float(
                    data[
                        "observed_multiplicity"
                    ]
                )
            ),
            spin_contamination_warning=(
                spin_warning
            ),
            pyscf_version=str(
                data[
                    "pyscf_version"
                ]
            ),
            elapsed_seconds=_decode_float(
                data[
                    "elapsed_seconds"
                ]
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(
            exc,
            SinglePointProtocolError,
        ):
            raise

        raise SinglePointProtocolError(
            "Invalid single-point result payload."
        ) from exc


def dumps_task(
    task: SinglePointTask,
) -> str:
    """Serialize one task using strict standard JSON."""
    return json.dumps(
        task_to_payload(
            task
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    )


def loads_task(
    text: str,
) -> SinglePointTask:
    """Deserialize one task from JSON."""
    try:
        payload = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise SinglePointProtocolError(
            "Invalid task JSON."
        ) from exc

    return task_from_payload(
        payload
    )


def dumps_result(
    result: SinglePointResult,
) -> str:
    """Serialize one result using strict standard JSON."""
    return json.dumps(
        result_to_payload(
            result
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    )


def loads_result(
    text: str,
) -> SinglePointResult:
    """Deserialize one result from JSON."""
    try:
        payload = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise SinglePointProtocolError(
            "Invalid result JSON."
        ) from exc

    return result_from_payload(
        payload
    )
