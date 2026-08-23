from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .decisions import AgentDecision
from .policies import BasePolicy


@dataclass(slots=True)
class BaseAgent:
    name: str
    policy: BasePolicy

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        return self.policy.choose_move(
            state=state,
            legal_moves=legal_moves,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"policy={self.policy.name!r})"
        )
