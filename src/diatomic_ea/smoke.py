"""System smoke tests for DiatomicEA."""

from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from diatomic_ea.resources import detect_cpu_resources


class SmokeStatus(str, Enum):
    """Outcome of one smoke-test check."""

    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    """Result of one system check."""

    name: str
    status: SmokeStatus
    message: str

    @property
    def passed(self) -> bool:
        return self.status is SmokeStatus.PASS


@dataclass(frozen=True, slots=True)
class SmokeTestReport:
    """Complete system smoke-test report."""

    checks: tuple[SmokeCheck, ...]

    @property
    def passed(self) -> bool:
        return all(
            check.passed
            for check in self.checks
        )

    @property
    def passed_count(self) -> int:
        return sum(
            check.passed
            for check in self.checks
        )

    @property
    def failed_count(self) -> int:
        return len(self.checks) - self.passed_count


def _check_python() -> SmokeCheck:
    version = sys.version_info

    if version >= (3, 10):
        return SmokeCheck(
            name="Python",
            status=SmokeStatus.PASS,
            message=(
                f"Python {version.major}."
                f"{version.minor}.{version.micro}"
            ),
        )

    return SmokeCheck(
        name="Python",
        status=SmokeStatus.FAIL,
        message=(
            "DiatomicEA requires Python 3.10 "
            "or newer."
        ),
    )


def _check_cpu_detection() -> SmokeCheck:
    try:
        resources = detect_cpu_resources()
    except Exception as exc:
        return SmokeCheck(
            name="CPU detection",
            status=SmokeStatus.FAIL,
            message=str(exc),
        )

    if (
        resources.physical_cores < 1
        or resources.logical_cores < 1
    ):
        return SmokeCheck(
            name="CPU detection",
            status=SmokeStatus.FAIL,
            message="Invalid CPU counts detected.",
        )

    return SmokeCheck(
        name="CPU detection",
        status=SmokeStatus.PASS,
        message=(
            f"{resources.physical_cores} physical / "
            f"{resources.logical_cores} logical cores; "
            f"{resources.recommended_workers} "
            "workers recommended."
        ),
    )


def _check_temporary_directory() -> SmokeCheck:
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.txt"
            path.write_text(
                "DiatomicEA smoke test",
                encoding="utf-8",
            )

            content = path.read_text(
                encoding="utf-8"
            )

            if content != "DiatomicEA smoke test":
                raise RuntimeError(
                    "Temporary-file verification failed."
                )

    except Exception as exc:
        return SmokeCheck(
            name="Temporary directory",
            status=SmokeStatus.FAIL,
            message=str(exc),
        )

    return SmokeCheck(
        name="Temporary directory",
        status=SmokeStatus.PASS,
        message="Temporary files can be written and read.",
    )


def _check_output_directory(
    output_dir: Path,
) -> SmokeCheck:
    try:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        probe = (
            output_dir
            / f".diatomic_ea_probe_{os.getpid()}"
        )

        probe.write_text(
            "write-test",
            encoding="utf-8",
        )

        if (
            probe.read_text(encoding="utf-8")
            != "write-test"
        ):
            raise RuntimeError(
                "Output-file verification failed."
            )

        probe.unlink()

    except Exception as exc:
        return SmokeCheck(
            name="Output directory",
            status=SmokeStatus.FAIL,
            message=str(exc),
        )

    return SmokeCheck(
        name="Output directory",
        status=SmokeStatus.PASS,
        message=f"Writable: {output_dir.resolve()}",
    )


def _multiprocessing_probe(
    result_queue,
) -> None:
    """Small worker used by the spawn smoke test."""
    result_queue.put(
        {
            "pid": os.getpid(),
            "ok": True,
        }
    )


def _check_multiprocessing() -> SmokeCheck:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_multiprocessing_probe,
        args=(result_queue,),
    )

    try:
        process.start()
        process.join(timeout=15)

        if process.is_alive():
            process.terminate()
            process.join()

            return SmokeCheck(
                name="Multiprocessing",
                status=SmokeStatus.FAIL,
                message=(
                    "Spawned worker did not finish "
                    "within 15 seconds."
                ),
            )

        if process.exitcode != 0:
            return SmokeCheck(
                name="Multiprocessing",
                status=SmokeStatus.FAIL,
                message=(
                    "Spawned worker exited with code "
                    f"{process.exitcode}."
                ),
            )

        try:
            result = result_queue.get(
                timeout=2
            )
        except Exception as exc:
            return SmokeCheck(
                name="Multiprocessing",
                status=SmokeStatus.FAIL,
                message=(
                    "Worker returned no result: "
                    f"{exc}"
                ),
            )

        if not result.get("ok"):
            return SmokeCheck(
                name="Multiprocessing",
                status=SmokeStatus.FAIL,
                message="Worker probe reported failure.",
            )

    except Exception as exc:
        return SmokeCheck(
            name="Multiprocessing",
            status=SmokeStatus.FAIL,
            message=str(exc),
        )

    finally:
        result_queue.close()
        result_queue.join_thread()

    return SmokeCheck(
        name="Multiprocessing",
        status=SmokeStatus.PASS,
        message=(
            "Spawned worker process completed "
            "successfully."
        ),
    )


def run_system_smoke_test(
    output_dir: str | Path = "results",
) -> SmokeTestReport:
    """Run the basic DiatomicEA system checks."""
    destination = Path(output_dir)

    checks = (
        _check_python(),
        _check_cpu_detection(),
        _check_temporary_directory(),
        _check_output_directory(destination),
        _check_multiprocessing(),
    )

    return SmokeTestReport(checks=checks)