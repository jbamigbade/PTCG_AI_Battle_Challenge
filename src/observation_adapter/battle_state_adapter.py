from __future__ import annotations

from typing import Any

from src.battle_state import BattleState


def move_label(
    move: Any,
) -> str:
    if isinstance(move, dict):
        return str(
            move.get("name")
            or move.get("Move Name")
            or move
        )

    for attribute in [
        "name",
        "move_name",
        "action",
        "attack_name",
    ]:
        value = getattr(
            move,
            attribute,
            None,
        )

        if value:
            return str(value)

    return str(move)


def numeric_move_value(
    move: Any,
    *candidate_names: str,
    default: float = 0.0,
) -> float:
    if isinstance(move, dict):
        for name in candidate_names:
            value = move.get(name)

            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return default

    for name in candidate_names:
        value = getattr(
            move,
            name,
            None,
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return default


def build_policy_state(
    battle_state: BattleState,
    legal_moves: list[Any],
) -> dict[str, Any]:
    current_side = (
        battle_state.player
        if battle_state.current_player == "Player"
        else battle_state.opponent
    )

    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    current_hp = float(
        current_side.active.current_hp
    )

    opponent_hp = float(
        opposing_side.active.current_hp
    )

    move_scores: dict[str, float] = {}
    heuristic_scores: dict[str, float] = {}
    search_scores: dict[str, float] = {}
    damage_scores: dict[str, float] = {}
    knockout_scores: dict[str, float] = {}
    defense_scores: dict[str, float] = {}
    healing_scores: dict[str, float] = {}
    move_lookup: dict[str, Any] = {}
    policy_legal_moves: list[str] = []

    for index, move in enumerate(legal_moves):
        base_label = move_label(move)
        label = base_label

        if label in move_lookup:
            label = f"{base_label} #{index + 1}"

        move_lookup[label] = move
        policy_legal_moves.append(label)

        damage = numeric_move_value(
            move,
            "damage",
            "damage_numeric",
            default=0.0,
        )

        knockout_value = (
            100.0
            if damage >= opponent_hp
            else 0.0
        )

        defensive_value = (
            50.0
            if "retreat" in label.lower()
            else 0.0
        )

        healing_value = (
            40.0
            if "heal" in label.lower()
            else 0.0
        )

        damage_scores[label] = damage
        knockout_scores[label] = knockout_value
        defense_scores[label] = defensive_value
        healing_scores[label] = healing_value
        move_scores[label] = damage

        heuristic_scores[label] = (
            damage
            + knockout_value
            + defensive_value * 0.25
            + healing_value * 0.25
        )

        search_scores[label] = (
            heuristic_scores[label]
        )

    return {
        "turn_number": battle_state.turn_number,
        "current_player": battle_state.current_player,
        "current_hp": current_hp,
        "opponent_hp": opponent_hp,
        "current_hand_size": current_side.hand_size,
        "opponent_hand_size": opposing_side.hand_size,
        "current_prizes": (
            current_side.prize_cards_remaining
        ),
        "opponent_prizes": (
            opposing_side.prize_cards_remaining
        ),
        "policy_legal_moves": policy_legal_moves,
        "move_lookup": move_lookup,
        "move_scores": move_scores,
        "heuristic_scores": heuristic_scores,
        "search_scores": search_scores,
        "damage_scores": damage_scores,
        "knockout_scores": knockout_scores,
        "defense_scores": defense_scores,
        "healing_scores": healing_scores,
    }


__all__ = [
    "build_policy_state",
    "move_label",
    "numeric_move_value",
]
