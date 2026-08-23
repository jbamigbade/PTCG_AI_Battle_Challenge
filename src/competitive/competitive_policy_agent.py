from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.competitive.competitive_policy_engine import (
    CompetitivePolicyEngine,
)

from src.legal_moves import (
    get_current_legal_moves,
)


@dataclass
class CompetitivePolicyDecision:
    move: Any
    confidence: float
    selected_move_name: str | None
    probabilities: list[float]
    logits: list[float]
    move_names: list[str]
    fallback: bool

    # Required by the production battle simulator.
    score: float = 0.0
    search_depth: int = 0
    nodes: int = 0

    reason: str | None = None


class CompetitivePolicyAgent:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        name: str = "Competitive Policy Agent",
    ) -> None:
        self.name = name

        self.policy_engine = (
            CompetitivePolicyEngine(
                checkpoint_path=
                    checkpoint_path,

                device=device,
            )
        )

        self.last_decision: (
            CompetitivePolicyDecision
            | None
        ) = None

    def choose_move(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> CompetitivePolicyDecision:
        resolved_state = (
            state
            if state is not None
            else battle_state
        )

        if resolved_state is None:
            raise ValueError(
                "A battle state must be supplied."
            )

        legal_moves = (
            get_current_legal_moves(
                resolved_state
            )
        )

        if not legal_moves:
            self.last_decision = (
                CompetitivePolicyDecision(
                    move=None,

                    confidence=0.0,

                    selected_move_name=None,

                    probabilities=[],

                    logits=[],

                    move_names=[],

                    fallback=True,

                    score=0.0,

                    search_depth=int(
                        depth or 0
                    ),

                    nodes=0,

                    reason=
                        "No legal moves available.",
                )
            )

            return self.last_decision

        result = (
            self.policy_engine
            .score_moves(
                battle_state=
                    resolved_state,

                legal_moves=
                    legal_moves,
            )
        )

        self.last_decision = (
            CompetitivePolicyDecision(
                move=
                    result[
                        "selected_move"
                    ],

                confidence=float(
                    result[
                        "confidence"
                    ]
                ),

                selected_move_name=
                    result[
                        "selected_move_name"
                    ],

                probabilities=list(
                    result[
                        "probabilities"
                    ]
                ),

                logits=list(
                    result[
                        "logits"
                    ]
                ),

                move_names=list(
                    result[
                        "move_names"
                    ]
                ),

                fallback=bool(
                    result[
                        "fallback"
                    ]
                ),

                score=float(
                    result[
                        "confidence"
                    ]
                ),

                search_depth=int(
                    depth or 0
                ),

                nodes=0,

                reason=None,
            )
        )

        return self.last_decision

    def select_move(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> CompetitivePolicyDecision:
        return self.choose_move(
            state=state,
            depth=depth,
            battle_state=battle_state,
            **kwargs,
        )

    def act(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> CompetitivePolicyDecision:
        return self.choose_move(
            state=state,
            depth=depth,
            battle_state=battle_state,
            **kwargs,
        )

    def get_last_decision(
        self,
    ) -> CompetitivePolicyDecision | None:
        return self.last_decision
