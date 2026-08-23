"""Adapters connecting the Pokémon battle engine to generic search."""

from __future__ import annotations

from typing import Hashable, Tuple

from .battle_state import BattleState
from .evaluation import evaluate_position
from .legal_moves import get_current_legal_moves


def generate_moves_adapter(
    battle_state: BattleState,
):
    """Return legal moves for the current side."""

    return get_current_legal_moves(
        battle_state
    )


def evaluate_state_adapter(
    battle_state: BattleState,
    root_player: str,
) -> float:
    """
    Evaluate from the requested root player's perspective.

    evaluate_position evaluates from Player's perspective, so the
    score is negated when Opponent is the search root.
    """

    score = evaluate_position(
        battle_state
    )

    if root_player == "Player":
        return score

    if root_player == "Opponent":
        return -score

    raise ValueError(
        "root_player must be 'Player' or 'Opponent'."
    )


def current_player_adapter(
    battle_state: BattleState,
) -> str:
    """Return the side whose turn it is."""

    return battle_state.current_player


def terminal_state_adapter(
    battle_state: BattleState,
) -> bool:
    """Return True when the simplified battle has ended."""

    if (
        battle_state.player.prize_cards_remaining <= 0
        or battle_state.opponent.prize_cards_remaining <= 0
    ):
        return True

    if (
        battle_state.player.active.current_hp <= 0
        or battle_state.opponent.active.current_hp <= 0
    ):
        return True

    return False


def pokemon_state_features(
    battle_state: BattleState,
) -> Tuple[Hashable, ...]:
    """
    Convert a simplified battle state into hashable features.

    These features can be passed to Notebook 11's ZobristHasher.
    """

    player_card = battle_state.player.active.card
    opponent_card = battle_state.opponent.active.card

    return (
        ("game", "pokemon_tcg"),
        ("current_player", battle_state.current_player),
        ("turn_number", battle_state.turn_number),

        (
            "player_active",
            player_card.get("id"),
            player_card.get("name"),
        ),
        (
            "player_hp",
            float(battle_state.player.active.current_hp),
        ),
        (
            "player_energy",
            battle_state.player.active.attached_energy,
        ),
        (
            "player_status",
            battle_state.player.active.status,
        ),
        (
            "player_prizes",
            battle_state.player.prize_cards_remaining,
        ),
        (
            "player_hand",
            battle_state.player.hand_size,
        ),

        (
            "opponent_active",
            opponent_card.get("id"),
            opponent_card.get("name"),
        ),
        (
            "opponent_hp",
            float(battle_state.opponent.active.current_hp),
        ),
        (
            "opponent_energy",
            battle_state.opponent.active.attached_energy,
        ),
        (
            "opponent_status",
            battle_state.opponent.active.status,
        ),
        (
            "opponent_prizes",
            battle_state.opponent.prize_cards_remaining,
        ),
        (
            "opponent_hand",
            battle_state.opponent.hand_size,
        ),
    )
