"""Dataclasses used by the official card database."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CardMove:
    """One move, attack, ability, or effect row belonging to a card."""

    name: str | None
    cost: str | None
    damage: str | None
    effect: str | None
    damage_numeric: int | None
    energy_cost: int | None


@dataclass
class CardRecord:
    """Canonical card record containing metadata and all associated moves."""

    card_id: int
    name: str
    expansion: str | None
    collection_number: str | None
    stage_or_type: str | None
    rule: str | None
    category: str | None
    previous_stage: str | None
    hp: int | None
    card_type: str | None
    weakness: str | None
    resistance: str | None
    retreat: int | None

    is_pokemon: bool
    is_trainer: bool
    is_energy: bool
    is_basic: bool
    is_stage1: bool
    is_stage2: bool
    has_rule_box: bool
    has_attack: bool
    has_ability: bool

    attack_power_score: float | None
    tank_score: float | None
    mobility_score: float | None
    evolution_score: float | None
    overall_battle_score: float | None

    moves: list[CardMove] = field(default_factory=list)
