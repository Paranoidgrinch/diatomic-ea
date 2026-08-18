"""Tests for the system smoke-test framework."""

from diatomic_ea.smoke import (
    SmokeCheck,
    SmokeStatus,
    SmokeTestReport,
    run_system_smoke_test,
)


def test_smoke_check_passed_property() -> None:
    check = SmokeCheck(
        name="Example",
        status=SmokeStatus.PASS,
        message="OK",
    )

    assert check.passed


def test_failed_check_is_not_passed() -> None:
    check = SmokeCheck(
        name="Example",
        status=SmokeStatus.FAIL,
        message="Failed",
    )

    assert not check.passed


def test_report_summary() -> None:
    report = SmokeTestReport(
        checks=(
            SmokeCheck(
                "A",
                SmokeStatus.PASS,
                "OK",
            ),
            SmokeCheck(
                "B",
                SmokeStatus.FAIL,
                "No",
            ),
        )
    )

    assert not report.passed
    assert report.passed_count == 1
    assert report.failed_count == 1


def test_system_smoke_test(tmp_path) -> None:
    output = tmp_path / "results"

    report = run_system_smoke_test(output)

    assert report.passed
    assert report.failed_count == 0
    assert len(report.checks) == 5
    assert output.exists()