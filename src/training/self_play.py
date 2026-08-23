from __future__ import annotations

import uuid
from dataclasses import dataclass

from src.tournament.match import TournamentMatch
from src.tournament.runner import run_two_agent_match

from .replay import ReplayGame, ReplayRecorder


@dataclass(slots=True)
class SelfPlayRunSummary:
    requested_games: int
    completed_games: int
    recorder: ReplayRecorder


def run_self_play_session(
    *,
    match_factory,
    initial_state_factory,
    num_games: int,
    verbose: bool = False,
) -> SelfPlayRunSummary:
    """
    Run repeated self-play matches using caller-provided factories.

    Parameters
    ----------
    match_factory:
        Callable accepting game_index and returning TournamentMatch.

    initial_state_factory:
        Callable accepting game_index and returning BattleState.

    num_games:
        Number of matches to execute.
    """

    if num_games < 1:
        raise ValueError("num_games must be at least 1.")

    recorder = ReplayRecorder()

    for game_index in range(num_games):
        match: TournamentMatch = match_factory(game_index)
        initial_state = initial_state_factory(game_index)

        result = run_two_agent_match(
            match,
            initial_state,
            verbose=verbose,
        )

        game_id = str(uuid.uuid4())

        replay_game = ReplayGame(
            game_id=game_id,
            winner=result.winner,
            total_turns=result.turns,
            final_score=result.final_score,
            transcript=result.transcript,
            steps=[],
        )

        recorder.add_game(replay_game)

    return SelfPlayRunSummary(
        requested_games=num_games,
        completed_games=recorder.total_games,
        recorder=recorder,
    )


__all__ = [
    "SelfPlayRunSummary",
    "run_self_play_session",
]
