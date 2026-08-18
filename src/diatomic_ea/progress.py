"""Calculation progress events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CalculationStage(str, Enum):
    """High-level stages of a Schema F calculation."""

    PREPARATION = "preparation"
    FAST_GRID = "fast-grid"
    FAST_GRID_ANALYSIS = "fast-grid-analysis"
    QZVPD_REFINEMENT = "qzvpd-refinement"
    STATISTICAL_EA = "statistical-ea"
    EXPORT = "export"


class ProgressEventType(str, Enum):
    """Types of progress events emitted by calculations."""

    JOB_STARTED = "job-started"
    STAGE_STARTED = "stage-started"
    ADVANCE = "advance"
    MESSAGE = "message"
    WARNING = "warning"
    STAGE_COMPLETED = "stage-completed"
    JOB_COMPLETED = "job-completed"
    JOB_FAILED = "job-failed"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Serializable progress message from a calculation worker."""

    job_id: str
    event_type: ProgressEventType
    stage: CalculationStage | None = None
    completed: int | None = None
    total: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError(
                "job_id must not be empty."
            )

        if (
            self.completed is None
            and self.total is not None
        ):
            raise ValueError(
                "completed is required when total is provided."
            )

        if (
            self.completed is not None
            and self.total is None
        ):
            raise ValueError(
                "total is required when completed is provided."
            )

        if self.total is not None:
            if self.total < 1:
                raise ValueError(
                    "total must be at least 1."
                )

            if self.completed is None:
                raise ValueError(
                    "completed must be provided."
                )

            if not 0 <= self.completed <= self.total:
                raise ValueError(
                    "completed must be between 0 and total."
                )

    @property
    def fraction(self) -> float | None:
        """Return progress as a fraction from 0 to 1."""
        if (
            self.completed is None
            or self.total is None
        ):
            return None

        return self.completed / self.total

    @property
    def percent(self) -> float | None:
        """Return progress as a percentage."""
        fraction = self.fraction

        if fraction is None:
            return None

        return fraction * 100.0


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Emit calculation progress without depending on a GUI."""

    def __init__(
        self,
        job_id: str,
        callback: ProgressCallback | None = None,
    ) -> None:
        if not job_id:
            raise ValueError(
                "job_id must not be empty."
            )

        self.job_id = job_id
        self.callback = callback

    def emit(
        self,
        event_type: ProgressEventType,
        *,
        stage: CalculationStage | None = None,
        completed: int | None = None,
        total: int | None = None,
        message: str = "",
    ) -> ProgressEvent:
        """Create and optionally publish a progress event."""
        event = ProgressEvent(
            job_id=self.job_id,
            event_type=event_type,
            stage=stage,
            completed=completed,
            total=total,
            message=message,
        )

        if self.callback is not None:
            self.callback(event)

        return event

    def job_started(
        self,
        message: str = "",
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.JOB_STARTED,
            message=message,
        )

    def stage_started(
        self,
        stage: CalculationStage,
        *,
        message: str = "",
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.STAGE_STARTED,
            stage=stage,
            message=message,
        )

    def advance(
        self,
        stage: CalculationStage,
        *,
        completed: int,
        total: int,
        message: str = "",
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.ADVANCE,
            stage=stage,
            completed=completed,
            total=total,
            message=message,
        )

    def warning(
        self,
        message: str,
        *,
        stage: CalculationStage | None = None,
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.WARNING,
            stage=stage,
            message=message,
        )

    def stage_completed(
        self,
        stage: CalculationStage,
        *,
        message: str = "",
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.STAGE_COMPLETED,
            stage=stage,
            message=message,
        )

    def job_completed(
        self,
        message: str = "",
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.JOB_COMPLETED,
            message=message,
        )

    def job_failed(
        self,
        message: str,
    ) -> ProgressEvent:
        return self.emit(
            ProgressEventType.JOB_FAILED,
            message=message,
        )