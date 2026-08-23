"""Lightweight battle-state models used by the PTCG search engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PokemonState:
    """Represents one Pokémon currently in play."""

    card: dict
    current_hp: float
    attached_energy: int = 0
    status: Optional[str] = None
    damage: float = 0.0
    is_active: bool = False


@dataclass
class PlayerState:
    """Represents one player's current battlefield."""

    active: PokemonState
    bench: List[PokemonState]
    prize_cards_remaining: int = 6
    hand_size: int = 7


@dataclass
class BattleState:
    """Complete position used by minimax and advanced search."""

    player: PlayerState
    opponent: PlayerState
    turn_number: int
    current_player: str

    def __post_init__(self) -> None:
        if self.current_player not in {"Player", "Opponent"}:
            raise ValueError(
                "current_player must be 'Player' or 'Opponent'."
            )
