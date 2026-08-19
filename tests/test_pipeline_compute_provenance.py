"""Tests for provenance selection in the Schema F pipeline."""

from unittest.mock import patch

from diatomic_ea.electron_count_adapter import (
    run_platform_electron_count,
)
from diatomic_ea.pipeline import (
    _resolve_compute_provenance,
)
from diatomic_ea.single_point_adapter import (
    run_platform_single_point,
)


def test_supplied_provenance_wins() -> None:
    supplied = {
        "backend": "test",
    }

    with patch(
        "diatomic_ea.pipeline.collect_compute_provenance"
    ) as collector:
        resolved = (
            _resolve_compute_provenance(
                worker=(
                    run_platform_single_point
                ),
                electron_count_resolver=(
                    run_platform_electron_count
                ),
                supplied=supplied,
            )
        )

    assert resolved is supplied

    collector.assert_not_called()


def test_production_backends_collect_real_provenance() -> None:
    expected = {
        "backend": "wsl",
        "compatibility": {
            "verified": True,
        },
    }

    with patch(
        "diatomic_ea.pipeline.collect_compute_provenance",
        return_value=expected,
    ) as collector:
        resolved = (
            _resolve_compute_provenance(
                worker=(
                    run_platform_single_point
                ),
                electron_count_resolver=(
                    run_platform_electron_count
                ),
                supplied=None,
            )
        )

    assert resolved == expected

    collector.assert_called_once_with()


def test_custom_test_backends_do_not_touch_real_environment() -> None:
    def fake_worker(task):
        return task

    def fake_count(**kwargs):
        return 10

    with patch(
        "diatomic_ea.pipeline.collect_compute_provenance"
    ) as collector:
        resolved = (
            _resolve_compute_provenance(
                worker=fake_worker,
                electron_count_resolver=(
                    fake_count
                ),
                supplied=None,
            )
        )

    assert resolved is None

    collector.assert_not_called()
