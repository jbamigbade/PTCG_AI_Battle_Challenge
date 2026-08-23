"""
Final PPO-powered Pokémon battle agent.

This module wraps the production PPOPolicyEngine and the existing
PokemonBattleAgent interface.

The current checkpoint is a bootstrap validation checkpoint. The agent
therefore includes deterministic legal-action fallback protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.battle_agent import PokemonBattleAgent
from src.legal_moves import get_current_legal_moves
from src.ppo.ppo_policy_engine import PPOPolicyEngine


class FinalPPOBattleAgent:
    """Competition-facing PPO battle agent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        deterministic: bool = True,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        ).resolve()

        self.engine = PPOPolicyEngine(
            checkpoint_path=self.checkpoint_path,
            device=device,
            deterministic=deterministic,
        )

        self.agent = PokemonBattleAgent(
            engine=self.engine
        )

    @staticmethod
    def _safe_fallback_move(
        state: Any,
    ) -> Any:
        """
        Select the highest-damage legal move.

        The legal-move module already provides a Pass action when no
        attack is available.
        """

        legal_moves = list(
            get_current_legal_moves(state)
        )

        if not legal_moves:
            raise RuntimeError(
                "No legal move was available."
            )

        return max(
            legal_moves,
            key=lambda move: float(
                move.get("damage", 0.0) or 0.0
            )
            if isinstance(move, dict)
            else 0.0,
        )

    def choose_move(
        self,
        state: Any,
        depth: int = 1,
    ) -> Any:
        """Return an AgentDecision compatible with the simulator."""

        try:
            decision = self.agent.choose_move(
                state=state,
                depth=depth,
            )

            legal_moves = list(
                get_current_legal_moves(state)
            )

            if decision.move not in legal_moves:
                raise RuntimeError(
                    "PPO selected a move outside the legal move set."
                )

            return decision

        except Exception:

            fallback_move = (
                self._safe_fallback_move(state)
            )

            # Reuse the search-compatible PPO result pathway.
            result = self.engine.search_depth(
                state=state,
                depth=1,
                clear_table=True,
            )

            result = type(result)(
                best_move=fallback_move,
                score=float(
                    fallback_move.get(
                        "damage",
                        0.0,
                    )
                )
                if isinstance(
                    fallback_move,
                    dict,
                )
                else 0.0,
                completed_depth=1,
                stats=result.stats,
                principal_variation=[
                    fallback_move
                ],
                action_index=0,
                confidence=0.0,
                used_fallback=True,
            )

            from src.agent_decision import (
                AgentDecision,
            )

            return AgentDecision(
                move=result.best_move,
                score=result.score,
                search_depth=result.completed_depth,
                nodes=result.stats.nodes,
                principal_variation=(
                    result.principal_variation
                ),
            )


def build_final_agent(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> FinalPPOBattleAgent:
    """Factory used by local evaluation or submission code."""

    return FinalPPOBattleAgent(
        checkpoint_path=checkpoint_path,
        device=device,
        deterministic=True,
    )
