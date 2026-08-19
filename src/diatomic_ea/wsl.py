"""Windows Subsystem for Linux discovery and command execution."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence


WSL_TIMEOUT_RETURN_CODE = 124


@dataclass(frozen=True, slots=True)
class WSLCommandResult:
    """Captured result of one WSL command."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def timed_out(self) -> bool:
        return (
            self.returncode
            == WSL_TIMEOUT_RETURN_CODE
        )


@dataclass(frozen=True, slots=True)
class WSLAvailability:
    """Availability of the Windows WSL interface."""

    executable: str | None
    distributions: tuple[str, ...]
    message: str
    timed_out: bool = False

    @property
    def executable_found(self) -> bool:
        return self.executable is not None

    @property
    def ready(self) -> bool:
        return (
            self.executable_found
            and bool(self.distributions)
            and not self.timed_out
        )


class WSLCommandError(RuntimeError):
    """Raised when a checked WSL command fails."""

    def __init__(
        self,
        result: WSLCommandResult,
    ) -> None:
        self.result = result

        message = (
            "WSL command failed with exit code "
            f"{result.returncode}: "
            + " ".join(result.command)
        )

        if result.stderr.strip():
            message += (
                "\n"
                + result.stderr.strip()
            )

        super().__init__(message)


def decode_wsl_output(
    data: bytes | str | None,
) -> str:
    """Decode WSL output captured through Windows pipes."""
    if data is None:
        return ""

    if isinstance(
        data,
        str,
    ):
        return data

    if not data:
        return ""

    if data.startswith(
        b"\xff\xfe"
    ):
        return (
            data.decode(
                "utf-16-le"
            )
            .lstrip("\ufeff")
        )

    if data.startswith(
        b"\xfe\xff"
    ):
        return (
            data.decode(
                "utf-16-be"
            )
            .lstrip("\ufeff")
        )

    if b"\x00" in data:
        try:
            return (
                data.decode(
                    "utf-16-le"
                )
                .lstrip("\ufeff")
            )
        except UnicodeDecodeError:
            pass

    try:
        return data.decode(
            "utf-8-sig"
        )
    except UnicodeDecodeError:
        return data.decode(
            errors="replace"
        )


def wsl_executable() -> str | None:
    """Return the installed WSL executable, if available."""
    executable = shutil.which(
        "wsl.exe"
    )

    if executable is not None:
        return executable

    return shutil.which(
        "wsl"
    )


def _run_windows_command(
    command: Sequence[str],
    *,
    timeout: float,
) -> WSLCommandResult:
    """Run a Windows command without allowing timeout exceptions to escape."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired as exc:
        stdout = decode_wsl_output(
            exc.stdout
        )

        stderr = decode_wsl_output(
            exc.stderr
        ).strip()

        timeout_message = (
            "Command timed out after "
            f"{timeout:g} seconds."
        )

        if stderr:
            stderr = (
                stderr
                + "\n"
                + timeout_message
            )

        if not stderr:
            stderr = timeout_message

        return WSLCommandResult(
            command=tuple(command),
            returncode=(
                WSL_TIMEOUT_RETURN_CODE
            ),
            stdout=stdout,
            stderr=stderr,
        )

    return WSLCommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=decode_wsl_output(
            completed.stdout
        ),
        stderr=decode_wsl_output(
            completed.stderr
        ),
    )


def _distribution_list_result(
    *,
    timeout: float,
) -> WSLCommandResult | None:
    executable = wsl_executable()

    if executable is None:
        return None

    return _run_windows_command(
        (
            executable,
            "--list",
            "--quiet",
        ),
        timeout=timeout,
    )


def _parse_distribution_names(
    stdout: str,
) -> tuple[str, ...]:
    names: list[str] = []

    for raw_line in stdout.splitlines():
        name = (
            raw_line
            .replace("\x00", "")
            .strip()
        )

        if not name:
            continue

        if name not in names:
            names.append(name)

    return tuple(names)


def list_wsl_distributions(
    *,
    timeout: float = 15.0,
) -> tuple[str, ...]:
    """Return installed WSL distribution names.

    Failure or timeout returns an empty tuple. Use inspect_wsl()
    when diagnostic status information is required.
    """
    result = _distribution_list_result(
        timeout=timeout
    )

    if result is None:
        return ()

    if not result.succeeded:
        return ()

    return _parse_distribution_names(
        result.stdout
    )


def inspect_wsl(
    *,
    timeout: float = 15.0,
) -> WSLAvailability:
    """Inspect WSL without allowing a hung service to crash the app."""
    executable = wsl_executable()

    if executable is None:
        return WSLAvailability(
            executable=None,
            distributions=(),
            message=(
                "WSL executable was not found."
            ),
            timed_out=False,
        )

    result = _distribution_list_result(
        timeout=timeout
    )

    if result is None:
        return WSLAvailability(
            executable=None,
            distributions=(),
            message=(
                "WSL executable was not found."
            ),
            timed_out=False,
        )

    if result.timed_out:
        return WSLAvailability(
            executable=executable,
            distributions=(),
            message=(
                "WSL is installed, but distribution "
                "discovery timed out after "
                f"{timeout:g} seconds. "
                "The WSL service may be unresponsive."
            ),
            timed_out=True,
        )

    if not result.succeeded:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown WSL error"
        )

        return WSLAvailability(
            executable=executable,
            distributions=(),
            message=(
                "WSL distribution discovery failed: "
                + detail
            ),
            timed_out=False,
        )

    distributions = (
        _parse_distribution_names(
            result.stdout
        )
    )

    if not distributions:
        return WSLAvailability(
            executable=executable,
            distributions=(),
            message=(
                "WSL is installed but no Linux "
                "distribution was detected."
            ),
            timed_out=False,
        )

    return WSLAvailability(
        executable=executable,
        distributions=distributions,
        message=(
            "WSL is available with "
            f"{len(distributions)} distribution(s)."
        ),
        timed_out=False,
    )


def run_wsl_command(
    arguments: Sequence[str],
    *,
    distribution: str | None = None,
    timeout: float = 60.0,
    check: bool = False,
) -> WSLCommandResult:
    """Run an argument-vector command inside WSL."""
    executable = wsl_executable()

    if executable is None:
        raise FileNotFoundError(
            "WSL executable was not found."
        )

    command: list[str] = [
        executable,
    ]

    if distribution is not None:
        distribution = (
            distribution.strip()
        )

        if not distribution:
            raise ValueError(
                "distribution must not be empty."
            )

        command.extend(
            [
                "--distribution",
                distribution,
            ]
        )

    command.append(
        "--"
    )

    command.extend(
        str(argument)
        for argument in arguments
    )

    result = _run_windows_command(
        command,
        timeout=timeout,
    )

    if (
        check
        and not result.succeeded
    ):
        raise WSLCommandError(
            result
        )

    return result


def run_wsl_shell(
    shell_command: str,
    *,
    distribution: str | None = None,
    timeout: float = 60.0,
    check: bool = False,
) -> WSLCommandResult:
    """Run one POSIX shell command inside WSL."""
    if not shell_command.strip():
        raise ValueError(
            "shell_command must not be empty."
        )

    return run_wsl_command(
        (
            "sh",
            "-lc",
            shell_command,
        ),
        distribution=distribution,
        timeout=timeout,
        check=check,
    )