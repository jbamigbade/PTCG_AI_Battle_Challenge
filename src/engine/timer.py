"""Timing utilities for time-limited game-tree search."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SearchTimeLimit:
    """High-resolution timer used by time-limited search."""

    max_seconds: float
    start_time: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.max_seconds = float(self.max_seconds)

        if self.max_seconds <= 0:
            raise ValueError(
                "max_seconds must be greater than 0."
            )

    def start(self) -> None:
        """Start or restart the timer."""

        self.start_time = time.perf_counter()

    def _require_started(self) -> float:
        if self.start_time is None:
            raise RuntimeError(
                "The timer has not been started."
            )

        return self.start_time

    def elapsed(self) -> float:
        """Return elapsed seconds."""

        start_time = self._require_started()

        return time.perf_counter() - start_time

    def remaining(self) -> float:
        """Return the remaining nonnegative time."""

        return max(
            0.0,
            self.max_seconds - self.elapsed(),
        )

    def expired(self) -> bool:
        """Return whether the time budget has expired."""

        return self.elapsed() >= self.max_seconds
