"""AI-versus-AI Pokémon battle simulation utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from .battle_agent import PokemonBattleAgent
from .battle_state import BattleState
from .simulator import apply_move


@dataclass
class BattleTurnRecord:
    """Record one completed simulated battle turn."""

    turn_number: int
    acting_side: str
    pokemon_name: str
    move_name: str
    damage: float
    score: float
    search_depth: int
    nodes: int
    player_hp_after: float
    opponent_hp_after: float
    next_side: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "acting_side": self.acting_side,
            "pokemon_name": self.pokemon_name,
            "move_name": self.move_name,
            "damage": self.damage,
            "score": self.score,
            "search_depth": self.search_depth,
            "nodes": self.nodes,
            "player_hp_after": self.player_hp_after,
            "opponent_hp_after": self.opponent_hp_after,
            "next_side": self.next_side,
        }


@dataclass
class BattleSimulationResult:
    """Store a complete simulated battle result."""

    final_state: BattleState
    turns: list[BattleTurnRecord] = field(default_factory=list)
    winner: Optional[str] = None
    stop_reason: str = ""

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "stop_reason": self.stop_reason,
            "turn_count": self.turn_count,
            "player_hp": (
                self.final_state.player.active.current_hp
            ),
            "opponent_hp": (
                self.final_state.opponent.active.current_hp
            ),
            "turns": [
                turn.as_dict()
                for turn in self.turns
            ],
        }


def determine_battle_winner(
    state: BattleState,
) -> Optional[str]:
    """Return the winner when the battle reaches a terminal state."""

    player_hp = state.player.active.current_hp
    opponent_hp = state.opponent.active.current_hp

    if player_hp <= 0 and opponent_hp <= 0:
        return "Draw"

    if opponent_hp <= 0:
        return "Player"

    if player_hp <= 0:
        return "Opponent"

    if state.player.prize_cards_remaining <= 0:
        return "Player"

    if state.opponent.prize_cards_remaining <= 0:
        return "Opponent"

    return None


def simulate_ai_battle(
    initial_state: BattleState,
    agent: PokemonBattleAgent,
    search_depth: int = 6,
    max_turns: int = 20,
    verbose: bool = True,
) -> BattleSimulationResult:
    """Run a complete alternating AI-versus-AI battle."""

    current_state = deepcopy(initial_state)
    turn_records: list[BattleTurnRecord] = []

    for _ in range(max_turns):
        winner = determine_battle_winner(current_state)

        if winner is not None:
            return BattleSimulationResult(
                final_state=current_state,
                turns=turn_records,
                winner=winner,
                stop_reason="Battle reached a terminal state.",
            )

        acting_side = current_state.current_player

        if acting_side == "Player":
            active_pokemon = current_state.player.active
        elif acting_side == "Opponent":
            active_pokemon = current_state.opponent.active
        else:
            raise ValueError(
                "current_player must be 'Player' or 'Opponent'."
            )

        decision = agent.choose_move(
            state=current_state,
            depth=search_depth,
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

    winner = determine_battle_winner(current_state)

    return BattleSimulationResult(
        final_state=current_state,
        turns=turn_records,
        winner=winner,
        stop_reason=(
            "Maximum turn limit reached."
            if winner is None
            else "Battle reached a terminal state."
        ),
    )


def create_battle_transcript(
    simulation: BattleSimulationResult,
) -> str:
    """Create a readable text transcript for a simulation."""

    lines = [
        "=" * 70,
        "POKÉMON AI BATTLE TRANSCRIPT",
        "=" * 70,
    ]

    for turn in simulation.turns:
        lines.append(
            f"Turn {turn.turn_number}: "
            f"{turn.acting_side} — "
            f"{turn.pokemon_name} used "
            f"{turn.move_name}"
        )

        lines.append(
            f"  Damage: {turn.damage:.1f}"
        )

        lines.append(
            f"  Search score: {turn.score:.2f}"
        )

        lines.append(
            f"  Search depth: {turn.search_depth}"
        )

        lines.append(
            f"  Nodes searched: {turn.nodes}"
        )

        lines.append(
            "  HP after move — "
            f"Player: {turn.player_hp_after:.1f}, "
            f"Opponent: {turn.opponent_hp_after:.1f}"
        )

        lines.append(
            f"  Next side: {turn.next_side}"
        )

        lines.append("-" * 70)

    lines.append(f"Winner: {simulation.winner}")
    lines.append(f"Stop reason: {simulation.stop_reason}")

    return "\n".join(lines)
