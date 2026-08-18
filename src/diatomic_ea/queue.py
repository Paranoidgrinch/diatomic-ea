"""Sequential calculation queue."""

from __future__ import annotations

from collections.abc import Iterable

from diatomic_ea.jobs import CalculationJob, JobStatus


class CalculationQueue:
    """FIFO queue for molecular calculation jobs.

    Only one job may be running at a time. This matches the initial
    DiatomicEA execution model: parallelize work within one molecule,
    then continue with the next queued molecule.
    """

    def __init__(
        self,
        jobs: Iterable[CalculationJob] | None = None,
    ) -> None:
        self._jobs = list(jobs or [])

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self):
        return iter(self._jobs)

    @property
    def jobs(self) -> tuple[CalculationJob, ...]:
        """Return an immutable snapshot of queue order."""
        return tuple(self._jobs)

    def add(self, job: CalculationJob) -> None:
        """Add a job to the end of the queue."""
        if job.status is not JobStatus.QUEUED:
            raise ValueError(
                "Only queued jobs can be added."
            )

        if any(
            existing.job_id == job.job_id
            for existing in self._jobs
        ):
            raise ValueError(
                f"Job already exists: {job.job_id}"
            )

        self._jobs.append(job)

    def get(self, job_id: str) -> CalculationJob:
        """Return a job by identifier."""
        for job in self._jobs:
            if job.job_id == job_id:
                return job

        raise KeyError(job_id)

    def start_next(self) -> CalculationJob | None:
        """Start the next waiting job.

        Returns None if no queued jobs remain.
        """
        if any(
            job.status is JobStatus.RUNNING
            for job in self._jobs
        ):
            raise RuntimeError(
                "A calculation job is already running."
            )

        for job in self._jobs:
            if job.status is JobStatus.QUEUED:
                job.transition_to(JobStatus.RUNNING)
                return job

        return None

    def remove(self, job_id: str) -> CalculationJob:
        """Remove a waiting job from the queue."""
        job = self.get(job_id)

        if job.status is not JobStatus.QUEUED:
            raise ValueError(
                "Only queued jobs can be removed."
            )

        self._jobs.remove(job)
        return job

    def move_up(self, job_id: str) -> bool:
        """Move a queued job one position upward."""
        index = self._index(job_id)
        job = self._jobs[index]

        if job.status is not JobStatus.QUEUED:
            raise ValueError(
                "Only queued jobs can be reordered."
            )

        if index == 0:
            return False

        previous = self._jobs[index - 1]

        if previous.status is not JobStatus.QUEUED:
            return False

        self._jobs[index - 1], self._jobs[index] = (
            self._jobs[index],
            self._jobs[index - 1],
        )
        return True

    def move_down(self, job_id: str) -> bool:
        """Move a queued job one position downward."""
        index = self._index(job_id)
        job = self._jobs[index]

        if job.status is not JobStatus.QUEUED:
            raise ValueError(
                "Only queued jobs can be reordered."
            )

        if index >= len(self._jobs) - 1:
            return False

        following = self._jobs[index + 1]

        if following.status is not JobStatus.QUEUED:
            return False

        self._jobs[index], self._jobs[index + 1] = (
            self._jobs[index + 1],
            self._jobs[index],
        )
        return True

    def _index(self, job_id: str) -> int:
        for index, job in enumerate(self._jobs):
            if job.job_id == job_id:
                return index

        raise KeyError(job_id)