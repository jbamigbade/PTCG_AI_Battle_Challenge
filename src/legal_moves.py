"""Legal attack and current-side move generation."""

from __future__ import annotations

from typing import Dict, List

from .battle_state import BattleState, PokemonState


Move = Dict[str, object]


def get_legal_moves(
    pokemon_state: PokemonState,
) -> List[Move]:
    """
    Return every attack currently usable by the selected Pokémon.

    Saved attack dictionaries use:

    - Move Name
    - damage_numeric
    - energy_cost
    - Effect Explanation
    """

    legal_moves: List[Move] = []

    attacks = pokemon_state.card.get("attacks", [])

    if not isinstance(attacks, list):
        return legal_moves

    for attack in attacks:
        if not isinstance(attack, dict):
            continue

        move_name = attack.get(
            "Move Name",
            "Unknown Move",
        )

        try:
            damage = float(
                attack.get("damage_numeric", 0) or 0
            )
        except (TypeError, ValueError):
            damage = 0.0

        try:
            required_energy = int(
                attack.get("energy_cost", 0) or 0
            )
        except (TypeError, ValueError):
            required_energy = 0

        usable = (
            pokemon_state.attached_energy
            >= required_energy
        )

        if usable:
            legal_moves.append(
                {
                    "name": str(move_name),
                    "damage": damage,
                    "energy_cost": required_energy,
                    "effect": attack.get(
                        "Effect Explanation"
                    ),
                }
            )

    return legal_moves


def get_current_side_pokemon(
    battle_state: BattleState,
) -> PokemonState:
    """Return the Active Pokémon belonging to the current side."""

    if battle_state.current_player == "Player":
        return battle_state.player.active

    if battle_state.current_player == "Opponent":
        return battle_state.opponent.active

    raise ValueError(
        "current_player must be 'Player' or 'Opponent'."
    )


def get_current_legal_moves(
    battle_state: BattleState,
) -> List[Move]:
    """
    Return legal moves for whichever side currently has the turn.

    When no attack is available, return a Pass action so the search
    tree can continue to the next turn.
    """

    active_pokemon = get_current_side_pokemon(
        battle_state
    )

    legal_moves = get_legal_moves(
        active_pokemon
    )

    if not legal_moves:
        return [
            {
                "name": "Pass",
                "damage": 0.0,
                "energy_cost": 0,
                "effect": "No legal attack was available.",
            }
        ]

    return legal_moves
