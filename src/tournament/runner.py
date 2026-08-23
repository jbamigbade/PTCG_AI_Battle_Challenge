"""Tournament match execution utilities."""

from __future__ import annotations

from copy import deepcopy

from ..battle_simulation import (
    BattleSimulationResult,
    BattleTurnRecord,
    create_battle_transcript,
    determine_battle_winner,
)
from ..battle_state import BattleState
from ..simulator import apply_move
from .match import TournamentMatch, TournamentResult


def run_two_agent_match(
    match: TournamentMatch,
    initial_state: BattleState,
    *,
    verbose: bool = False,
) -> TournamentResult:
    """Run one complete match using a separate agent for each side."""

    current_state = deepcopy(initial_state)
    turn_records: list[BattleTurnRecord] = []

    for _ in range(match.max_turns):
        winner = determine_battle_winner(
            current_state
        )

        if winner is not None:
            break

        acting_side = current_state.current_player

        if acting_side == "Player":
            acting_agent = match.player_agent
            active_pokemon = current_state.player.active
        elif acting_side == "Opponent":
            acting_agent = match.opponent_agent
            active_pokemon = current_state.opponent.active
        else:
            raise ValueError(
                "current_player must be 'Player' or 'Opponent'."
            )

        decision = acting_agent.choose_move(
            state=current_state,
            depth=match.search_depth,
        )

        next_state = apply_move(
            current_state,
            decision.move,
        )

        record = BattleTurnRecord(
            turn_number=current_state.turn_number,
            acting_side=acting_side,
            pokemon_name=active_pokemon.card["name"],
            move_name=decision.move["name"],
            damage=float(
                decision.move.get("damage", 0) or 0
            ),
            score=decision.score,
            search_depth=decision.search_depth,
            nodes=decision.nodes,
            player_hp_after=(
                next_state.player.active.current_hp
            ),
            opponent_hp_after=(
                next_state.opponent.active.current_hp
            ),
            next_side=next_state.current_player,
        )

        turn_records.append(record)

        if verbose:
            print(
                f"Turn {record.turn_number}: "
                f"{record.acting_side} — "
                f"{record.pokemon_name} used "
                f"{record.move_name}"
            )
            print(
                f"  Player HP: {record.player_hp_after} | "
                f"Opponent HP: {record.opponent_hp_after}"
            )

        current_state = next_state

    winner = determine_battle_winner(
        current_state
    )

    simulation = BattleSimulationResult(
        final_state=current_state,
        turns=turn_records,
        winner=winner,
        stop_reason=(
            "Battle reached a terminal state."
            if winner is not None
            else "Maximum turn limit reached."
        ),
    )

    if winner == "Player":
        winner_name = match.player_name
    elif winner == "Opponent":
        winner_name = match.opponent_name
    elif winner == "Draw":
        winner_name = "Draw"
    else:
        winner_name = "No winner"

    final_score = (
        current_state.player.active.current_hp
        - current_state.opponent.active.current_hp
    )

    return TournamentResult(
        winner=winner_name,
        turns=len(turn_records),
        final_score=final_score,
        transcript=create_battle_transcript(
            simulation
        ),
    )
