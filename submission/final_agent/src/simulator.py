"""Battle-state transition functions for minimax search."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

from .battle_state import BattleState


Move = Dict[str, object]


def apply_move(
    battle_state: BattleState,
    move: Move,
) -> BattleState:
    """
    Apply one move for the side identified by current_player.

    A new BattleState is returned. The original state is unchanged.
    """

    future_state = deepcopy(battle_state)

    if future_state.current_player == "Player":
        acting_player = future_state.player
        defending_player = future_state.opponent
        next_player = "Opponent"

    elif future_state.current_player == "Opponent":
        acting_player = future_state.opponent
        defending_player = future_state.player
        next_player = "Player"

    else:
        raise ValueError(
            "current_player must be 'Player' or 'Opponent'."
        )

    try:
        damage = float(
            move.get("damage", 0) or 0
        )
    except (TypeError, ValueError):
        damage = 0.0

    defending_player.active.current_hp = max(
        0.0,
        defending_player.active.current_hp - damage,
    )

    # Keep the separate damage field synchronized.
    defending_player.active.damage += max(
        0.0,
        damage,
    )

    knockout_occurred = (
        defending_player.active.current_hp <= 0
    )

    if knockout_occurred:
        acting_player.prize_cards_remaining = max(
            0,
            acting_player.prize_cards_remaining - 1,
        )

    future_state.turn_number += 1
    future_state.current_player = next_player

    return future_state
