from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .replay import ReplayRecorder


@dataclass(slots=True)
class SelfPlayStatistics:
    total_games: int = 0
    total_steps: int = 0
    average_turns: float = 0.0
    average_score: float = 0.0
    winner_counts: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_statistics(
    recorder: ReplayRecorder,
) -> SelfPlayStatistics:
    """
    Calculate summary statistics from recorded self-play games.
    """

    games = list(recorder.iter_games())

    if not games:
        return SelfPlayStatistics(
            total_games=0,
            total_steps=0,
            average_turns=0.0,
            average_score=0.0,
            winner_counts={},
        )

    winner_counts = Counter(
        game.winner or "Unknown"
        for game in games
    )

    average_turns = sum(
        game.total_turns
        for game in games
    ) / len(games)

    average_score = sum(
        game.final_score
        for game in games
    ) / len(games)

    return SelfPlayStatistics(
        total_games=len(games),
        total_steps=recorder.total_steps,
        average_turns=average_turns,
        average_score=average_score,
        winner_counts=dict(winner_counts),
    )


__all__ = [
    "SelfPlayStatistics",
    "calculate_statistics",
]
