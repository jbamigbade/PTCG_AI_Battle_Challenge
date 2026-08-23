from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class ReplayStep:
    """
    One decision recorded during a self-play battle.
    """

    game_id: str
    turn: int
    player: str
    state: dict[str, Any]
    legal_moves: list[str]
    chosen_move: str
    evaluation: float = 0.0
    search_depth: int = 0


@dataclass(slots=True)
class ReplayGame:
    """
    Complete replay record for one self-play game.
    """

    game_id: str
    winner: str = ""
    total_turns: int = 0
    final_score: float = 0.0
    transcript: str = ""
    steps: list[ReplayStep] = field(default_factory=list)

    def add_step(self, step: ReplayStep) -> None:
        if step.game_id != self.game_id:
            raise ValueError(
                "ReplayStep game_id does not match ReplayGame game_id."
            )

        self.steps.append(step)


class ReplayRecorder:
    """
    Stores replay games and exposes reusable replay utilities.
    """

    def __init__(self) -> None:
        self.games: list[ReplayGame] = []

    def add_game(self, game: ReplayGame) -> None:
        if any(existing.game_id == game.game_id for existing in self.games):
            raise ValueError(
                f"Duplicate replay game_id: {game.game_id}"
            )

        self.games.append(game)

    def clear(self) -> None:
        self.games.clear()

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def total_steps(self) -> int:
        return sum(len(game.steps) for game in self.games)

    def iter_games(self) -> Iterable[ReplayGame]:
        yield from self.games

    def iter_steps(self) -> Iterable[ReplayStep]:
        for game in self.games:
            yield from game.steps

    def get_game(self, game_id: str) -> ReplayGame | None:
        for game in self.games:
            if game.game_id == game_id:
                return game

        return None


__all__ = [
    "ReplayStep",
    "ReplayGame",
    "ReplayRecorder",
]
