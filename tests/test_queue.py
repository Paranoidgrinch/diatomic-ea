"""Tests for the sequential calculation queue."""

import pytest

from diatomic_ea.jobs import (
    CalculationJob,
    JobStatus,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.queue import CalculationQueue


def make_job(
    atom_a: str,
    atom_b: str,
) -> CalculationJob:
    return CalculationJob(
        molecule=DiatomicMolecule(atom_a, atom_b)
    )


def test_jobs_are_added_in_fifo_order() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue()
    queue.add(first)
    queue.add(second)

    assert queue.jobs == (first, second)


def test_start_next_starts_first_waiting_job() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])

    started = queue.start_next()

    assert started is first
    assert first.status is JobStatus.RUNNING
    assert second.status is JobStatus.QUEUED


def test_only_one_job_can_run() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])
    queue.start_next()

    with pytest.raises(RuntimeError):
        queue.start_next()


def test_next_job_starts_after_completion() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])

    running = queue.start_next()
    assert running is not None

    running.transition_to(JobStatus.COMPLETED)

    next_job = queue.start_next()

    assert next_job is second
    assert second.status is JobStatus.RUNNING


def test_remove_waiting_job() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])

    removed = queue.remove(first.job_id)

    assert removed is first
    assert queue.jobs == (second,)


def test_move_job_up() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])

    assert queue.move_up(second.job_id)
    assert queue.jobs == (second, first)


def test_move_job_down() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])

    assert queue.move_down(first.job_id)
    assert queue.jobs == (second, first)


def test_running_job_cannot_be_reordered() -> None:
    first = make_job("Al", "O")
    second = make_job("Mg", "O")

    queue = CalculationQueue([first, second])
    queue.start_next()

    with pytest.raises(ValueError):
        queue.move_down(first.job_id)


def test_duplicate_job_is_rejected() -> None:
    job = make_job("Al", "O")

    queue = CalculationQueue([job])

    with pytest.raises(ValueError):
        queue.add(job)