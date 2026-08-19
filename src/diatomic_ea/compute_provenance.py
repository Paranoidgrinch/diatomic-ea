"""Reproducible identity of the actual compute backend."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from diatomic_ea import __version__
from diatomic_ea.compute_deploy import (
    WSL_WHEEL_HASH_MARKER,
)
from diatomic_ea.compute_environment import (
    EXPECTED_PYSCF_VERSION,
    WSL_COMPUTE_PYTHON,
    WSL_COMPUTE_VENV,
    inspect_compute_environment,
)
from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)
from diatomic_ea.wsl import (
    run_wsl_command,
)


def _host_identity() -> dict[str, object]:
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": (
            platform.python_version()
        ),
        "application_version": (
            __version__
        ),
        "single_point_protocol_version": (
            SINGLE_POINT_PROTOCOL_VERSION
        ),
    }


def _extract_json_object(
    stdout: str,
) -> dict[str, object] | None:
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

        if isinstance(
            payload,
            dict,
        ):
            return payload

    return None


def _valid_sha256(
    value: object,
) -> bool:
    if not isinstance(
        value,
        str,
    ):
        return False

    if len(
        value
    ) != 64:
        return False

    return all(
        character
        in "0123456789abcdefABCDEF"
        for character in value
    )


def _wsl_provenance_program() -> str:
    """Return pure Python code executed inside the managed WSL venv."""
    return f"""
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import diatomic_ea
import diatomic_ea.single_point_worker

from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)

payload = {{
    "system": platform.system(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "kernel_release": platform.release(),
    "python_version": platform.python_version(),
    "pyscf_version": metadata.version(
        "pyscf"
    ),
    "basis_set_exchange_version": metadata.version(
        "basis-set-exchange"
    ),
    "diatomic_ea_version": metadata.version(
        "diatomic-ea"
    ),
    "single_point_protocol_version": (
        SINGLE_POINT_PROTOCOL_VERSION
    ),
    "module_path": str(
        Path(
            diatomic_ea.__file__
        ).resolve()
    ),
    "worker_module_path": str(
        Path(
            diatomic_ea.single_point_worker.__file__
        ).resolve()
    ),
    "worker_wheel_sha256": (
        Path(
            {WSL_WHEEL_HASH_MARKER!r}
        )
        .read_text(
            encoding="ascii"
        )
        .strip()
    ),
}}

print(
    json.dumps(
        payload,
        sort_keys=True,
    )
)
""".strip()


def _native_provenance(
    *,
    system_name: str,
) -> dict[str, Any]:
    environment = inspect_compute_environment(
        system_name=system_name
    )

    pyscf_match = (
        environment.pyscf_version
        == EXPECTED_PYSCF_VERSION
    )

    verified = (
        environment.ready
        and pyscf_match
    )

    return {
        "backend": "native",
        "ready": environment.ready,
        "host": _host_identity(),
        "compute": {
            "system": system_name,
            "distribution": None,
            "platform": (
                platform.platform()
            ),
            "machine": (
                platform.machine()
            ),
            "kernel_release": (
                platform.release()
            ),
            "python_version": (
                environment.python_version
            ),
            "pyscf_version": (
                environment.pyscf_version
            ),
            "basis_set_exchange_version": (
                environment
                .basis_set_exchange_version
            ),
            "diatomic_ea_version": (
                __version__
            ),
            "single_point_protocol_version": (
                SINGLE_POINT_PROTOCOL_VERSION
            ),
            "module_path": str(
                Path(
                    __file__
                ).resolve()
            ),
            "worker_module_path": None,
            "worker_wheel_sha256": None,
        },
        "compatibility": {
            "verified": verified,
            "application_version_match": True,
            "protocol_version_match": True,
            "pyscf_version_match": (
                pyscf_match
            ),
            "environment_version_match": True,
            "managed_venv": None,
            "worker_module_managed": None,
            "wheel_hash_valid": None,
        },
        "message": (
            "Native compute environment "
            "provenance verified."
            if verified
            else environment.message
        ),
    }


def _unready_wsl_provenance(
    environment,
) -> dict[str, Any]:
    return {
        "backend": "wsl",
        "ready": False,
        "host": _host_identity(),
        "compute": {
            "system": "Linux",
            "distribution": (
                environment.distribution
            ),
            "platform": None,
            "machine": None,
            "kernel_release": None,
            "python_version": (
                environment.python_version
            ),
            "pyscf_version": (
                environment.pyscf_version
            ),
            "basis_set_exchange_version": (
                environment
                .basis_set_exchange_version
            ),
            "diatomic_ea_version": None,
            "single_point_protocol_version": None,
            "module_path": None,
            "worker_module_path": None,
            "worker_wheel_sha256": None,
        },
        "compatibility": {
            "verified": False,
            "application_version_match": False,
            "protocol_version_match": False,
            "pyscf_version_match": False,
            "environment_version_match": False,
            "managed_venv": False,
            "worker_module_managed": False,
            "wheel_hash_valid": False,
        },
        "message": environment.message,
    }


def _wsl_provenance(
    *,
    distribution: str | None,
) -> dict[str, Any]:
    environment = inspect_compute_environment(
        system_name="Windows",
        distribution=distribution,
    )

    if not environment.ready:
        return _unready_wsl_provenance(
            environment
        )

    selected = environment.distribution

    if selected is None:
        return _unready_wsl_provenance(
            environment
        )

    result = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _wsl_provenance_program(),
        ),
        distribution=selected,
        timeout=60.0,
    )

    if not result.succeeded:
        provenance = (
            _unready_wsl_provenance(
                environment
            )
        )

        provenance["ready"] = True

        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown provenance probe error"
        )

        provenance["message"] = (
            "WSL compute environment is ready, "
            "but worker provenance could not "
            "be collected: "
            + detail
        )

        return provenance

    payload = _extract_json_object(
        result.stdout
    )

    if payload is None:
        provenance = (
            _unready_wsl_provenance(
                environment
            )
        )

        provenance["ready"] = True

        provenance["message"] = (
            "WSL compute environment is ready, "
            "but worker provenance returned "
            "no valid JSON payload."
        )

        return provenance

    worker_version = str(
        payload.get(
            "diatomic_ea_version",
            "",
        )
    ) or None

    worker_protocol_raw = payload.get(
        "single_point_protocol_version"
    )

    try:
        worker_protocol = int(
            worker_protocol_raw
        )
    except (
        TypeError,
        ValueError,
    ):
        worker_protocol = None

    worker_pyscf = str(
        payload.get(
            "pyscf_version",
            "",
        )
    ) or None

    worker_bse = str(
        payload.get(
            "basis_set_exchange_version",
            "",
        )
    ) or None

    worker_python = str(
        payload.get(
            "python_version",
            "",
        )
    ) or None

    module_path = str(
        payload.get(
            "module_path",
            "",
        )
    ) or None

    worker_module_path = str(
        payload.get(
            "worker_module_path",
            "",
        )
    ) or None

    wheel_hash = str(
        payload.get(
            "worker_wheel_sha256",
            "",
        )
    ) or None

    application_version_match = (
        worker_version
        == __version__
    )

    protocol_version_match = (
        worker_protocol
        == SINGLE_POINT_PROTOCOL_VERSION
    )

    pyscf_version_match = (
        worker_pyscf
        == EXPECTED_PYSCF_VERSION
    )

    environment_version_match = (
        worker_python
        == environment.python_version
        and worker_pyscf
        == environment.pyscf_version
        and worker_bse
        == environment.basis_set_exchange_version
    )

    managed_venv = (
        module_path is not None
        and module_path.startswith(
            WSL_COMPUTE_VENV
            + "/"
        )
    )

    worker_module_managed = (
        worker_module_path is not None
        and worker_module_path.startswith(
            WSL_COMPUTE_VENV
            + "/"
        )
    )

    wheel_hash_valid = (
        _valid_sha256(
            wheel_hash
        )
    )

    checks = {
        "application_version_match": (
            application_version_match
        ),
        "protocol_version_match": (
            protocol_version_match
        ),
        "pyscf_version_match": (
            pyscf_version_match
        ),
        "environment_version_match": (
            environment_version_match
        ),
        "managed_venv": (
            managed_venv
        ),
        "worker_module_managed": (
            worker_module_managed
        ),
        "wheel_hash_valid": (
            wheel_hash_valid
        ),
    }

    verified = all(
        checks.values()
    )

    if verified:
        message = (
            "Managed WSL compute worker "
            "provenance verified."
        )

    else:
        failed_checks = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        message = (
            "WSL compute worker provenance "
            "failed compatibility checks: "
            + ", ".join(
                failed_checks
            )
            + "."
        )

    return {
        "backend": "wsl",
        "ready": True,
        "host": _host_identity(),
        "compute": {
            "system": str(
                payload.get(
                    "system",
                    "Linux",
                )
            ),
            "distribution": selected,
            "platform": str(
                payload.get(
                    "platform",
                    "",
                )
            ) or None,
            "machine": str(
                payload.get(
                    "machine",
                    "",
                )
            ) or None,
            "kernel_release": str(
                payload.get(
                    "kernel_release",
                    "",
                )
            ) or None,
            "python_version": (
                worker_python
            ),
            "pyscf_version": (
                worker_pyscf
            ),
            "basis_set_exchange_version": (
                worker_bse
            ),
            "diatomic_ea_version": (
                worker_version
            ),
            "single_point_protocol_version": (
                worker_protocol
            ),
            "module_path": module_path,
            "worker_module_path": (
                worker_module_path
            ),
            "worker_wheel_sha256": (
                wheel_hash
            ),
        },
        "compatibility": {
            "verified": verified,
            **checks,
        },
        "message": message,
    }


def collect_compute_provenance(
    *,
    system_name: str | None = None,
    distribution: str | None = None,
) -> dict[str, Any]:
    """Collect identity of the backend that performs calculations."""
    resolved_system = (
        system_name
        or platform.system()
    )

    if (
        resolved_system.casefold()
        == "windows"
    ):
        return _wsl_provenance(
            distribution=distribution
        )

    return _native_provenance(
        system_name=resolved_system
    )


def main() -> int:
    """Print and validate the current compute provenance."""
    provenance = (
        collect_compute_provenance()
    )

    print()
    print(
        "DiatomicEA compute provenance"
    )

    print(
        "============================="
    )

    print()

    print(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
    )

    print()

    verified = bool(
        provenance.get(
            "compatibility",
            {},
        ).get(
            "verified",
            False,
        )
    )

    print(
        "Status:",
        (
            "PASS"
            if verified
            else "FAIL"
        ),
    )

    return (
        0
        if verified
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
