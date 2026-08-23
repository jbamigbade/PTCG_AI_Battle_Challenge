from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    name: str
    policy_type: str

    seed: int | None = None

    aggression_weight: float = 1.0
    defense_weight: float = 1.0
    immediate_value_weight: float = 1.0
    randomness: float = 0.0

    search_depth: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
