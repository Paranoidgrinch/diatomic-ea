"""Tests for CPU-resource handling."""

import pytest

from diatomic_ea.resources import recommended_worker_count


@pytest.mark.parametrize(
    ("physical_cores", "expected"),
    [
        (1, 1),
        (2, 1),
        (4, 3),
        (8, 7),
        (16, 14),
        (20, 18),
        (32, 29),
        (64, 58),
    ],
)
def test_recommended_worker_count(
    physical_cores: int,
    expected: int,
) -> None:
    assert recommended_worker_count(physical_cores) == expected


def test_invalid_core_count() -> None:
    with pytest.raises(ValueError):
        recommended_worker_count(0)
