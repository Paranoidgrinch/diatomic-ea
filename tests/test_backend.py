"""Tests for quantum-chemistry backend handling."""

from diatomic_ea.backend import (
    BackendAvailability,
    BackendSmokeReport,
    PySCFBackend,
    native_windows_supported,
)


def test_native_windows_is_not_supported() -> None:
    assert not native_windows_supported(
        "Windows"
    )


def test_linux_is_supported() -> None:
    assert native_windows_supported(
        "Linux"
    )


def test_macos_is_supported() -> None:
    assert native_windows_supported(
        "Darwin"
    )


def test_backend_availability_ready_property() -> None:
    ready = BackendAvailability(
        backend="test",
        platform_supported=True,
        installed=True,
        version="1.0",
        message="OK",
    )

    unavailable = BackendAvailability(
        backend="test",
        platform_supported=True,
        installed=False,
        version=None,
        message="Missing",
    )

    assert ready.ready
    assert not unavailable.ready


def test_smoke_report_can_store_energy() -> None:
    report = BackendSmokeReport(
        backend="test",
        passed=True,
        message="OK",
        energy_hartree=-1.0,
    )

    assert report.passed
    assert report.energy_hartree == -1.0


def test_pyscf_availability_is_safe_to_query() -> None:
    backend = PySCFBackend()

    status = backend.availability()

    assert status.backend == "PySCF"

    if status.ready:
        assert status.version is not None