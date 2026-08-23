"""Reusable Pokémon AI battle agent."""

from __future__ import annotations

from typing import Any

from .agent_decision import AgentDecision


class PokemonBattleAgent:
    """Use an AdvancedSearchEngine to choose Pokémon actions."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def choose_move(
        self,
        state: Any,
        depth: int = 6,
    ) -> AgentDecision:
        """Search the current state and return the selected move."""

        result = self.engine.search_depth(
            state=state,
            depth=depth,
            clear_table=True,
        )

        if result.best_move is None:
            raise RuntimeError(
                "The search engine did not return a legal move."
            )

        return AgentDecision(
            move=result.best_move,
            score=result.score,
            search_depth=result.completed_depth,
            nodes=result.stats.nodes,
            principal_variation=result.principal_variation,
        )
