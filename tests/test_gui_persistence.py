"""Tests for persistent desktop state."""

from diatomic_ea.gui_execution import (
    GuiCalculationSpec,
)
from diatomic_ea.gui_persistence import (
    GuiPreferences,
    load_preferences,
    load_queue_session,
    save_preferences,
    save_queue_session,
)
from diatomic_ea.jobs import (
    CalculationJob,
    JobStatus,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)


def make_job(
    status=JobStatus.QUEUED,
):
    molecule = DiatomicMolecule(
        "Al",
        "O",
    )

    job = CalculationJob(
        molecule=molecule,
        job_id="persistent-job",
        status=status,
    )

    spec = GuiCalculationSpec(
        job_id=job.job_id,
        molecule=molecule,
        minimum_angstrom=1.0,
        maximum_angstrom=2.5,
        spin_max=5,
        workers=3,
        run_id="alo-persistent-job",
    )

    return (
        job,
        spec,
    )


def test_preferences_round_trip(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "preferences.json"
    )

    expected = GuiPreferences(
        minimum_angstrom=0.8,
        maximum_angstrom=2.8,
        spin_max=7,
        workers=4,
    )

    save_preferences(
        path,
        expected,
    )

    assert (
        load_preferences(
            path,
            fallback_workers=1,
        )
        == expected
    )


def test_queue_round_trip(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "queue.json"
    )

    job, spec = make_job()

    save_queue_session(
        path,
        jobs=(
            job,
        ),
        specs={
            job.job_id: spec,
        },
    )

    loaded = load_queue_session(
        path
    )

    assert len(
        loaded.jobs
    ) == 1

    assert (
        loaded.specs[
            job.job_id
        ]
        == spec
    )


def test_running_job_is_recovered_as_waiting(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "queue.json"
    )

    job, spec = make_job(
        JobStatus.RUNNING
    )

    save_queue_session(
        path,
        jobs=(
            job,
        ),
        specs={
            job.job_id: spec,
        },
    )

    loaded = load_queue_session(
        path
    )

    assert (
        loaded.jobs[0].status
        is JobStatus.QUEUED
    )

    assert (
        loaded.recovered_running_jobs
        == 1
    )
