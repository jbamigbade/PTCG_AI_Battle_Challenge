from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Mapping

from .base_agent import BaseAgent
from .factories import AgentFactory


@dataclass(frozen=True, slots=True)
class OpponentProfile:
    name: str
    policy_type: str
    description: str

    seed: int | None = None
    search_depth: int = 0

    aggression_weight: float = 1.0
    defense_weight: float = 1.0
    immediate_value_weight: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


def create_opponent(
    profile: OpponentProfile,
) -> BaseAgent:
    return AgentFactory.create(
        name=profile.name,
        policy_type=profile.policy_type,
        seed=profile.seed,
        search_depth=profile.search_depth,
        aggression_weight=profile.aggression_weight,
        defense_weight=profile.defense_weight,
        immediate_value_weight=(
            profile.immediate_value_weight
        ),
        metadata=profile.metadata,
    )


def build_round_robin_schedule(
    opponent_names: list[str],
    num_games: int,
) -> list[str]:
    if not opponent_names:
        raise ValueError(
            "opponent_names cannot be empty."
        )

    if num_games < 1:
        raise ValueError(
            "num_games must be at least 1."
        )

    scheduler = cycle(opponent_names)

    return [
        next(scheduler)
        for _ in range(num_games)
    ]
