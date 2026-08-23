"""Tournament match configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass

from ..battle_agent import PokemonBattleAgent


@dataclass
class TournamentMatch:
    """Configuration for one AI-versus-AI tournament match."""

    player_agent: PokemonBattleAgent
    opponent_agent: PokemonBattleAgent
    player_name: str = "Player"
    opponent_name: str = "Opponent"
    search_depth: int = 6
    max_turns: int = 100

    def __post_init__(self) -> None:
        if self.search_depth < 1:
            raise ValueError(
                "search_depth must be at least 1."
            )

        if self.max_turns < 1:
            raise ValueError(
                "max_turns must be at least 1."
            )


@dataclass
class TournamentResult:
    """Summary of one completed tournament match."""

    winner: str
    turns: int
    final_score: float
    transcript: str
