from __future__ import annotations

import random

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .configs import PolicyConfig
from .decisions import AgentDecision


class BasePolicy(ABC):

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        self.config = config
        self.name = config.name
        self.policy_type = config.policy_type
        self._rng = random.Random(config.seed)

    @abstractmethod
    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:
        ...

    def validate_legal_moves(
        self,
        legal_moves: Sequence[Any],
    ) -> None:
        if not legal_moves:
            raise ValueError(
                f"{self.name} received no legal moves."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"policy_type={self.policy_type!r})"
        )


class RandomPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        move = self._rng.choice(
            list(legal_moves)
        )

        return AgentDecision(
            move=move,
            score=0.0,
            reason="Random legal move",
            metadata={"policy": self.name},
        )


class GreedyPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "move_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                0.0,
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Highest immediate value",
            metadata={"policy": self.name},
        )


class HeuristicPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "heuristic_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                float("-inf"),
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Highest heuristic evaluation",
            metadata={"policy": self.name},
        )


class SearchPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "search_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                float("-inf"),
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Search engine recommendation",
            metadata={
                "policy": self.name,
                "search_depth": (
                    self.config.search_depth
                ),
            },
        )


class AggressivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        damage_scores = state.get(
            "damage_scores",
            {},
        )

        knockout_scores = state.get(
            "knockout_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.aggression_weight
                * float(
                    damage_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    knockout_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason=(
                "Highest damage and knockout value"
            ),
            metadata={"policy": self.name},
        )


class DefensivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        defense_scores = state.get(
            "defense_scores",
            {},
        )

        healing_scores = state.get(
            "healing_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.defense_weight
                * float(
                    defense_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    healing_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason="Highest defensive value",
            metadata={"policy": self.name},
        )
