"""Decision result returned by the Pokémon battle agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentDecision:
    """Store one completed search decision."""

    move: dict[str, Any]
    score: float
    search_depth: int
    nodes: int
    principal_variation: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        """Return a compact readable decision summary."""

        return {
            "move": (
                self.move.get("name")
                if self.move is not None
                else None
            ),
            "score": self.score,
            "depth": self.search_depth,
            "nodes": self.nodes,
            "pv": [
                move.get("name")
                for move in self.principal_variation
            ],
        }
