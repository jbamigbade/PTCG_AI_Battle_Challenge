from __future__ import annotations

from typing import Any

from src.battle_state import BattleState
from src.engine import SearchResult, SearchStats
from src.legal_moves import get_current_legal_moves

from .battle_state_adapter import build_policy_state


class PolicySearchEngineAdapter:
    def __init__(
        self,
        policy_agent: Any,
    ) -> None:
        self.policy_agent = policy_agent

    def search_depth(
        self,
        *,
        state: BattleState,
        depth: int,
        clear_table: bool = True,
    ) -> SearchResult:
        legal_moves = get_current_legal_moves(
            state
        )

        if not legal_moves:
            return SearchResult(
                best_move=None,
                score=0.0,
                completed_depth=depth,
                principal_variation=[],
                stats=SearchStats(),
            )

        policy_state = build_policy_state(
            state,
            legal_moves,
        )

        decision = self.policy_agent.choose_move(
            state=policy_state,
            legal_moves=(
                policy_state["policy_legal_moves"]
            ),
        )

        selected_move = policy_state[
            "move_lookup"
        ][decision.move]

        return SearchResult(
            best_move=selected_move,
            score=float(decision.score),
            completed_depth=depth,
            principal_variation=[
                selected_move
            ],
            stats=SearchStats(
                nodes=1,
            ),
        )


__all__ = [
    "PolicySearchEngineAdapter",
]
