"""Deploy the DiatomicEA compute package into the managed WSL venv."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.compute_environment import (
    DEFAULT_WSL_DISTRIBUTION,
    WSL_COMPUTE_ROOT,
    WSL_COMPUTE_VENV,
    WSL_COMPUTE_PYTHON,
)
from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)
from diatomic_ea.wsl import (
    run_wsl_command,
)


WSL_WHEEL_HASH_MARKER = (
    WSL_COMPUTE_ROOT
    + "/worker-wheel.sha256"
)


@dataclass(frozen=True, slots=True)
class WSLDeploymentResult:
    """Result of deploying one DiatomicEA wheel into WSL."""

    success: bool
    distribution: str
    wheel_path: str
    wheel_sha256: str
    package_version: str | None
    module_path: str | None
    message: str


def sha256_file(
    path: str | Path,
) -> str:
    """Return the SHA-256 digest of one file."""
    resolved = Path(
        path
    )

    digest = hashlib.sha256()

    with resolved.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _failure_detail(
    result,
) -> str:
    return (
        result.stderr.strip()
        or result.stdout.strip()
        or "unknown error"
    )


def _extract_json_object(
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


def windows_path_to_wsl(
    path: str | Path,
    *,
    distribution: str,
) -> str:
    """Translate one absolute Windows path using the selected WSL distro."""
    resolved = Path(
        path
    ).resolve()

    windows_path = str(
        resolved
    ).replace(
        "\\",
        "/",
    )

    result = run_wsl_command(
        (
            "wslpath",
            "-a",
            "-u",
            windows_path,
        ),
        distribution=distribution,
        timeout=30.0,
    )

    if not result.succeeded:
        raise RuntimeError(
            "Could not translate Windows path "
            "to WSL path: "
            + _failure_detail(
                result
            )
        )

    translated = (
        result.stdout
        .replace(
            "\x00",
            "",
        )
        .strip()
    )

    if not translated:
        raise RuntimeError(
            "WSL path translation returned "
            "an empty path."
        )

    if not translated.startswith(
        "/"
    ):
        raise RuntimeError(
            "WSL path translation did not "
            "return an absolute Linux path: "
            f"{translated!r}."
        )

    return translated


def _marker_program(
    wheel_sha256: str,
) -> str:
    return (
        "from pathlib import Path; "
        f"Path({WSL_WHEEL_HASH_MARKER!r})"
        f".write_text({(wheel_sha256 + chr(10))!r}, "
        "encoding='ascii')"
    )


def _verification_program() -> str:
    return f"""
import importlib.metadata as metadata
import json
from pathlib import Path

import diatomic_ea
import diatomic_ea.single_point_worker
from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)

payload = {{
    "package_version": metadata.version(
        "diatomic-ea"
    ),
    "module_path": str(
        Path(
            diatomic_ea.__file__
        ).resolve()
    ),
    "worker_module": str(
        Path(
            diatomic_ea.single_point_worker.__file__
        ).resolve()
    ),
    "protocol_version": (
        SINGLE_POINT_PROTOCOL_VERSION
    ),
    "wheel_sha256": (
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


def deploy_wsl_wheel(
    wheel_path: str | Path,
    *,
    distribution: str = DEFAULT_WSL_DISTRIBUTION,
) -> WSLDeploymentResult:
    """Install and verify one exact DiatomicEA wheel in WSL."""
    wheel = Path(
        wheel_path
    ).resolve()

    if not wheel.is_file():
        raise ValueError(
            f"Wheel does not exist: {wheel}"
        )

    if wheel.suffix.casefold() != ".whl":
        raise ValueError(
            "wheel_path must point to a .whl file."
        )

    wheel_hash = sha256_file(
        wheel
    )

    try:
        linux_wheel = windows_path_to_wsl(
            wheel,
            distribution=distribution,
        )

    except RuntimeError as exc:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=None,
            module_path=None,
            message=str(
                exc
            ),
        )

    install = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            linux_wheel,
        ),
        distribution=distribution,
        user="root",
        timeout=600.0,
    )

    if not install.succeeded:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=None,
            module_path=None,
            message=(
                "Could not install DiatomicEA "
                "wheel in WSL: "
                + _failure_detail(
                    install
                )
            ),
        )

    marker = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _marker_program(
                wheel_hash
            ),
        ),
        distribution=distribution,
        user="root",
        timeout=30.0,
    )

    if not marker.succeeded:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=None,
            module_path=None,
            message=(
                "DiatomicEA was installed, but "
                "the deployment marker could not "
                "be written: "
                + _failure_detail(
                    marker
                )
            ),
        )

    verify = run_wsl_command(
        (
            WSL_COMPUTE_PYTHON,
            "-c",
            _verification_program(),
        ),
        distribution=distribution,
        timeout=60.0,
    )

    if not verify.succeeded:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=None,
            module_path=None,
            message=(
                "Installed DiatomicEA package "
                "could not be imported: "
                + _failure_detail(
                    verify
                )
            ),
        )

    payload = _extract_json_object(
        verify.stdout
    )

    if payload is None:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=None,
            module_path=None,
            message=(
                "WSL deployment verification "
                "returned no valid JSON payload."
            ),
        )

    package_version = str(
        payload.get(
            "package_version",
            "",
        )
    ) or None

    module_path = str(
        payload.get(
            "module_path",
            "",
        )
    ) or None

    worker_module = str(
        payload.get(
            "worker_module",
            "",
        )
    ) or None

    installed_hash = str(
        payload.get(
            "wheel_sha256",
            "",
        )
    )

    try:
        protocol_version = int(
            payload.get(
                "protocol_version"
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        protocol_version = -1

    if installed_hash != wheel_hash:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=package_version,
            module_path=module_path,
            message=(
                "Installed wheel hash does not "
                "match the deployed wheel."
            ),
        )

    if (
        protocol_version
        != SINGLE_POINT_PROTOCOL_VERSION
    ):
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=package_version,
            module_path=module_path,
            message=(
                "Installed WSL protocol version "
                "does not match the Windows package."
            ),
        )

    if module_path is None:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=package_version,
            module_path=None,
            message=(
                "Installed DiatomicEA module path "
                "was not reported."
            ),
        )

    if not module_path.startswith(
        WSL_COMPUTE_VENV
    ):
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=package_version,
            module_path=module_path,
            message=(
                "DiatomicEA was imported from an "
                "unexpected Linux environment: "
                + module_path
            ),
        )

    if not worker_module:
        return WSLDeploymentResult(
            success=False,
            distribution=distribution,
            wheel_path=str(
                wheel
            ),
            wheel_sha256=wheel_hash,
            package_version=package_version,
            module_path=module_path,
            message=(
                "The installed single-point worker "
                "could not be located."
            ),
        )

    return WSLDeploymentResult(
        success=True,
        distribution=distribution,
        wheel_path=str(
            wheel
        ),
        wheel_sha256=wheel_hash,
        package_version=package_version,
        module_path=module_path,
        message=(
            "Exact DiatomicEA compute wheel "
            "is installed and verified in WSL."
        ),
    )


def main() -> int:
    """Deploy one wheel supplied on the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Install an exact DiatomicEA wheel "
            "into the managed WSL compute venv."
        )
    )

    parser.add_argument(
        "wheel",
    )

    parser.add_argument(
        "--distribution",
        default=DEFAULT_WSL_DISTRIBUTION,
    )

    args = parser.parse_args()

    result = deploy_wsl_wheel(
        args.wheel,
        distribution=args.distribution,
    )

    print(
        "Distribution:",
        result.distribution,
    )

    print(
        "Wheel:",
        result.wheel_path,
    )

    print(
        "SHA-256:",
        result.wheel_sha256,
    )

    print(
        "Package version:",
        result.package_version
        or "not available",
    )

    print(
        "Module:",
        result.module_path
        or "not available",
    )

    print(
        "Status:",
        result.message,
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
