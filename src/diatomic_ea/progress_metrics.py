"""Rate and ETA telemetry derived from calculation progress events.

This module is independent of Qt and console output.

The same metrics can therefore be consumed by:
- the command-line production runner,
- production_status.json,
- the future desktop GUI.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from diatomic_ea.progress import (
    CalculationStage,
    ProgressEvent,
    ProgressEventType,
)


@dataclass(frozen=True, slots=True)
class ProgressMetrics:
    """Calculated performance metrics for one active stage."""

    stage: CalculationStage | None
    completed: int | None
    total: int | None
    elapsed_seconds: float | None
    tasks_per_second: float | None
    eta_seconds: float | None


class ProgressRateTracker:
    """Calculate a smoothed task rate and ETA from progress events."""

    def __init__(
        self,
        *,
        window_seconds: float = 30.0,
        minimum_elapsed_seconds: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive."
            )

        if minimum_elapsed_seconds < 0:
            raise ValueError(
                "minimum_elapsed_seconds must not be negative."
            )

        self.window_seconds = float(
            window_seconds
        )

        self.minimum_elapsed_seconds = float(
            minimum_elapsed_seconds
        )

        self._clock = clock

        self._stage: CalculationStage | None = None
        self._stage_started_at: float | None = None

        self._samples: deque[
            tuple[float, int]
        ] = deque()

        self._last_metrics = ProgressMetrics(
            stage=None,
            completed=None,
            total=None,
            elapsed_seconds=None,
            tasks_per_second=None,
            eta_seconds=None,
        )

    def _start_stage(
        self,
        stage: CalculationStage | None,
        now: float,
    ) -> None:
        self._stage = stage
        self._stage_started_at = now

        self._samples.clear()

        self._samples.append(
            (
                now,
                0,
            )
        )

        self._last_metrics = ProgressMetrics(
            stage=stage,
            completed=0,
            total=None,
            elapsed_seconds=0.0,
            tasks_per_second=None,
            eta_seconds=None,
        )

    def _prune_samples(
        self,
        now: float,
    ) -> None:
        cutoff = (
            now
            - self.window_seconds
        )

        while (
            len(
                self._samples
            )
            > 2
            and self._samples[1][0]
            <= cutoff
        ):
            self._samples.popleft()

    def _calculate_rate(
        self,
        *,
        now: float,
        completed: int,
    ) -> float | None:
        if self._stage_started_at is None:
            return None

        elapsed = (
            now
            - self._stage_started_at
        )

        if (
            elapsed
            < self.minimum_elapsed_seconds
        ):
            return None

        self._prune_samples(
            now
        )

        if len(
            self._samples
        ) >= 2:
            first_time, first_completed = (
                self._samples[0]
            )

            delta_time = (
                now
                - first_time
            )

            delta_tasks = (
                completed
                - first_completed
            )

            if (
                delta_time
                >= self.minimum_elapsed_seconds
                and delta_tasks > 0
            ):
                return (
                    delta_tasks
                    / delta_time
                )

        if (
            elapsed > 0
            and completed > 0
        ):
            return (
                completed
                / elapsed
            )

        return None

    def update(
        self,
        event: ProgressEvent,
        *,
        now: float | None = None,
    ) -> ProgressMetrics:
        """Consume one event and return the latest stage telemetry."""
        timestamp = (
            self._clock()
            if now is None
            else float(
                now
            )
        )

        if (
            event.event_type
            is ProgressEventType.STAGE_STARTED
        ):
            self._start_stage(
                event.stage,
                timestamp,
            )

            return self._last_metrics

        if (
            event.stage is not None
            and event.stage
            != self._stage
        ):
            self._start_stage(
                event.stage,
                timestamp,
            )

        if (
            event.event_type
            is ProgressEventType.ADVANCE
            and event.completed is not None
            and event.total is not None
        ):
            if self._stage_started_at is None:
                self._start_stage(
                    event.stage,
                    timestamp,
                )

            if (
                self._samples
                and event.completed
                < self._samples[-1][1]
            ):
                self._start_stage(
                    event.stage,
                    timestamp,
                )

            self._samples.append(
                (
                    timestamp,
                    event.completed,
                )
            )

            rate = self._calculate_rate(
                now=timestamp,
                completed=event.completed,
            )

            if self._stage_started_at is None:
                elapsed = None
            else:
                elapsed = max(
                    0.0,
                    timestamp
                    - self._stage_started_at,
                )

            remaining = max(
                0,
                event.total
                - event.completed,
            )

            eta = (
                None
                if rate is None
                or rate <= 0
                else (
                    remaining
                    / rate
                )
            )

            self._last_metrics = ProgressMetrics(
                stage=event.stage,
                completed=event.completed,
                total=event.total,
                elapsed_seconds=elapsed,
                tasks_per_second=rate,
                eta_seconds=eta,
            )

            return self._last_metrics

        if (
            self._stage_started_at is not None
        ):
            elapsed = max(
                0.0,
                timestamp
                - self._stage_started_at,
            )

            previous = (
                self._last_metrics
            )

            self._last_metrics = ProgressMetrics(
                stage=(
                    event.stage
                    if event.stage is not None
                    else previous.stage
                ),
                completed=previous.completed,
                total=previous.total,
                elapsed_seconds=elapsed,
                tasks_per_second=(
                    previous.tasks_per_second
                ),
                eta_seconds=(
                    previous.eta_seconds
                ),
            )

        return self._last_metrics


def format_duration(
    seconds: float | None,
) -> str:
    """Return a compact human-readable duration for console or GUI use."""
    if seconds is None:
        return "--"

    seconds = max(
        0,
        int(
            round(
                seconds
            )
        ),
    )

    if seconds < 60:
        return (
            f"{seconds}s"
        )

    minutes, remaining_seconds = divmod(
        seconds,
        60,
    )

    if minutes < 60:
        return (
            f"{minutes}m "
            f"{remaining_seconds:02d}s"
        )

    hours, remaining_minutes = divmod(
        minutes,
        60,
    )

    return (
        f"{hours}h "
        f"{remaining_minutes:02d}m"
    )
