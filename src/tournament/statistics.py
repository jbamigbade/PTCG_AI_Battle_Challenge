"""Tournament statistics models and update functions."""

from __future__ import annotations

from dataclasses import dataclass

from .match import TournamentResult


@dataclass
class TournamentStatistics:
    """Aggregate results across multiple tournament matches."""

    matches_played: int = 0
    player_wins: int = 0
    opponent_wins: int = 0
    draws: int = 0
    total_turns: int = 0
    average_turns: float = 0.0
    total_score: float = 0.0
    average_score: float = 0.0

    @property
    def player_win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0

        return self.player_wins / self.matches_played

    @property
    def opponent_win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0

        return self.opponent_wins / self.matches_played

    @property
    def draw_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0

        return self.draws / self.matches_played


def update_statistics(
    stats: TournamentStatistics,
    result: TournamentResult,
    *,
    player_name: str,
    opponent_name: str,
) -> TournamentStatistics:
    """Update aggregate statistics with one match result."""

    stats.matches_played += 1

    if result.winner == player_name:
        stats.player_wins += 1
    elif result.winner == opponent_name:
        stats.opponent_wins += 1
    else:
        stats.draws += 1

    stats.total_turns += result.turns
    stats.total_score += result.final_score

    stats.average_turns = (
        stats.total_turns
        / stats.matches_played
    )

    stats.average_score = (
        stats.total_score
        / stats.matches_played
    )

    return stats
