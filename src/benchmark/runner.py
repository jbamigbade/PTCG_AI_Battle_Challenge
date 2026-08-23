"""Reusable benchmark match and matrix execution."""

from __future__ import annotations

import time

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..battle_agent import PokemonBattleAgent
from ..battle_state import BattleState
from ..tournament import (
    TournamentMatch,
    run_two_agent_match,
)


BattleFactory = Callable[[str], BattleState]


def run_benchmark_match(
    *,
    benchmark_agents: Mapping[str, PokemonBattleAgent],
    battle_factory: BattleFactory,
    player_name: str,
    opponent_name: str,
    search_depth: int,
    starting_player: str = "Player",
    max_turns: int = 20,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one timed benchmark match."""

    if player_name not in benchmark_agents:
        raise KeyError(
            f"Unknown benchmark agent: {player_name!r}"
        )

    if opponent_name not in benchmark_agents:
        raise KeyError(
            f"Unknown benchmark agent: {opponent_name!r}"
        )

    if player_name == opponent_name:
        raise ValueError(
            "A benchmark agent cannot play against itself."
        )

    if search_depth < 1:
        raise ValueError(
            "search_depth must be at least 1."
        )

    if max_turns < 1:
        raise ValueError(
            "max_turns must be at least 1."
        )

    match = TournamentMatch(
        player_agent=benchmark_agents[player_name],
        opponent_agent=benchmark_agents[opponent_name],
        player_name=player_name,
        opponent_name=opponent_name,
        search_depth=search_depth,
        max_turns=max_turns,
    )

    initial_state = battle_factory(
        starting_player
    )

    start_time = time.perf_counter()

    result = run_two_agent_match(
        match=match,
        initial_state=initial_state,
        verbose=verbose,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return {
        "player": player_name,
        "opponent": opponent_name,
        "starting_player": starting_player,
        "search_depth": search_depth,
        "winner": result.winner,
        "turns": result.turns,
        "final_score": result.final_score,
        "elapsed_seconds": elapsed_seconds,
        "transcript_characters": len(
            result.transcript
        ),
    }


def run_benchmark_matrix(
    *,
    benchmark_agents: Mapping[str, PokemonBattleAgent],
    battle_factory: BattleFactory,
    search_depths: Iterable[int],
    pairings: Iterable[tuple[str, str]],
    starting_player: str = "Player",
    max_turns: int = 20,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run every requested pairing at every search depth."""

    depth_values = list(
        search_depths
    )

    pairing_values = list(
        pairings
    )

    if not depth_values:
        raise ValueError(
            "At least one search depth is required."
        )

    if not pairing_values:
        raise ValueError(
            "At least one benchmark pairing is required."
        )

    benchmark_rows: list[dict[str, Any]] = []
    benchmark_number = 1

    for search_depth in depth_values:
        for player_name, opponent_name in pairing_values:
            record = run_benchmark_match(
                benchmark_agents=benchmark_agents,
                battle_factory=battle_factory,
                player_name=player_name,
                opponent_name=opponent_name,
                search_depth=search_depth,
                starting_player=starting_player,
                max_turns=max_turns,
                verbose=verbose,
            )

            record[
                "benchmark_number"
            ] = benchmark_number

            benchmark_rows.append(
                record
            )

            benchmark_number += 1

    return benchmark_rows
