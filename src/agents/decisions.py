from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentDecision:
    move: Any
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
