"""
Production PPO policy engine.

This module adapts the trained PPO Actor network to the search_depth()
interface expected by PokemonBattleAgent.

The checkpoint currently represents a bootstrap validation model.
The observation schema used for simulator integration is explicitly:

    0. current active Pokémon HP ratio
    1. opposing active Pokémon HP ratio
    2. current player's prize progress
    3. opposing player's prize progress

Legal moves are mapped to action slots in their existing order, with a
maximum of four action slots matching the trained Actor output dimension.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.legal_moves import get_current_legal_moves


OBSERVATION_DIM = 4
ACTION_DIM = 4


class PPOActor(nn.Module):
    """Actor architecture used by Notebooks 41–44."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observations)


@dataclass(frozen=True)
class PPOPolicyStats:
    """Minimal statistics compatible with PokemonBattleAgent."""

    nodes: int = 1


@dataclass(frozen=True)
class PPOPolicyResult:
    """Search-compatible PPO result."""

    best_move: Any
    score: float
    completed_depth: int
    stats: PPOPolicyStats
    principal_variation: list[Any]
    action_index: int
    confidence: float
    used_fallback: bool


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(
    value: float,
) -> float:
    return max(
        0.0,
        min(1.0, value),
    )


def _pokemon_max_hp(
    pokemon_state: Any,
) -> float:
    """
    Read maximum HP from the underlying card dictionary.

    Multiple common field names are supported because historical dataset
    exports may use slightly different capitalization.
    """

    card = getattr(
        pokemon_state,
        "card",
        {},
    )

    if not isinstance(card, dict):
        return max(
            1.0,
            _safe_float(
                getattr(
                    pokemon_state,
                    "current_hp",
                    1.0,
                ),
                1.0,
            ),
        )

    candidate_fields = (
        "hp",
        "HP",
        "Hit Points",
        "hit_points",
        "max_hp",
    )

    for field in candidate_fields:
        value = _safe_float(
            card.get(field),
            0.0,
        )

        if value > 0.0:
            return value

    return max(
        1.0,
        _safe_float(
            getattr(
                pokemon_state,
                "current_hp",
                1.0,
            ),
            1.0,
        ),
    )


def encode_battle_observation(
    battle_state: Any,
) -> np.ndarray:
    """
    Encode one BattleState into the four PPO input features.

    Feature order:
        0 current active HP ratio
        1 opponent active HP ratio
        2 current prize progress
        3 opponent prize progress
    """

    if battle_state.current_player == "Player":
        current_side = battle_state.player
        opposing_side = battle_state.opponent

    elif battle_state.current_player == "Opponent":
        current_side = battle_state.opponent
        opposing_side = battle_state.player

    else:
        raise ValueError(
            "current_player must be 'Player' or 'Opponent'."
        )

    current_hp = _safe_float(
        current_side.active.current_hp,
    )

    opponent_hp = _safe_float(
        opposing_side.active.current_hp,
    )

    current_max_hp = _pokemon_max_hp(
        current_side.active
    )

    opponent_max_hp = _pokemon_max_hp(
        opposing_side.active
    )

    current_hp_ratio = _clamp01(
        current_hp / current_max_hp
    )

    opponent_hp_ratio = _clamp01(
        opponent_hp / opponent_max_hp
    )

    current_prizes_remaining = _safe_float(
        current_side.prize_cards_remaining,
        6.0,
    )

    opponent_prizes_remaining = _safe_float(
        opposing_side.prize_cards_remaining,
        6.0,
    )

    current_prize_progress = _clamp01(
        (6.0 - current_prizes_remaining) / 6.0
    )

    opponent_prize_progress = _clamp01(
        (6.0 - opponent_prizes_remaining) / 6.0
    )

    return np.asarray(
        [
            current_hp_ratio,
            opponent_hp_ratio,
            current_prize_progress,
            opponent_prize_progress,
        ],
        dtype=np.float32,
    )


class PPOPolicyEngine:
    """
    PPO-backed engine compatible with PokemonBattleAgent.

    PokemonBattleAgent calls:

        engine.search_depth(
            state=state,
            depth=depth,
            clear_table=True,
        )
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        deterministic: bool = True,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        ).resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.deterministic = bool(
            deterministic
        )

        self.actor = PPOActor().to(
            self.device
        )

        self._load_checkpoint()

        self.actor.eval()

    def _load_checkpoint(
        self,
    ) -> None:

        with self.checkpoint_path.open(
            "rb"
        ) as file:
            checkpoint = pickle.load(file)

        actor_state_dict = checkpoint.get(
            "actor_state_dict"
        )

        if actor_state_dict is None:
            raise KeyError(
                "Checkpoint does not contain "
                "'actor_state_dict'."
            )

        self.actor.load_state_dict(
            actor_state_dict
        )

        self.metadata = checkpoint.get(
            "metadata",
            {},
        )

    @staticmethod
    def _fallback_action_index(
        legal_moves: Sequence[Any],
    ) -> int:
        """
        Deterministic fallback.

        Select the move with the greatest numeric damage. Ties preserve
        the original legal-move ordering.
        """

        if not legal_moves:
            raise RuntimeError(
                "No legal moves were supplied."
            )

        best_index = 0
        best_damage = float("-inf")

        for index, move in enumerate(
            legal_moves
        ):
            damage = 0.0

            if isinstance(move, dict):
                damage = _safe_float(
                    move.get(
                        "damage",
                        move.get(
                            "damage_numeric",
                            0.0,
                        ),
                    )
                )

            if damage > best_damage:
                best_damage = damage
                best_index = index

        return best_index

    def select_move(
        self,
        state: Any,
    ) -> PPOPolicyResult:
        """Select one legal move from a BattleState."""

        legal_moves = list(
            get_current_legal_moves(state)
        )

        if not legal_moves:
            raise RuntimeError(
                "The simulator returned no legal moves."
            )

        # Actor supports four action slots.
        candidate_moves = legal_moves[
            :ACTION_DIM
        ]

        observation = (
            encode_battle_observation(state)
        )

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action_mask = torch.zeros(
            (1, ACTION_DIM),
            dtype=torch.bool,
            device=self.device,
        )

        action_mask[
            0,
            :len(candidate_moves),
        ] = True

        used_fallback = False

        try:
            with torch.no_grad():

                logits = self.actor(
                    observation_tensor
                )

                masked_logits = logits.masked_fill(
                    ~action_mask,
                    torch.finfo(
                        logits.dtype
                    ).min,
                )

                distribution = Categorical(
                    logits=masked_logits
                )

                if self.deterministic:
                    selected_index = int(
                        distribution.probs.argmax(
                            dim=1
                        ).item()
                    )
                else:
                    selected_index = int(
                        distribution.sample().item()
                    )

                confidence = float(
                    distribution.probs[
                        0,
                        selected_index,
                    ].item()
                )

                score = float(
                    masked_logits[
                        0,
                        selected_index,
                    ].item()
                )

        except Exception:
            selected_index = (
                self._fallback_action_index(
                    candidate_moves
                )
            )

            confidence = 0.0
            score = _safe_float(
                candidate_moves[
                    selected_index
                ].get("damage", 0.0)
                if isinstance(
                    candidate_moves[
                        selected_index
                    ],
                    dict,
                )
                else 0.0
            )

            used_fallback = True

        if (
            selected_index < 0
            or selected_index
            >= len(candidate_moves)
        ):
            selected_index = (
                self._fallback_action_index(
                    candidate_moves
                )
            )

            confidence = 0.0
            used_fallback = True

        selected_move = candidate_moves[
            selected_index
        ]

        return PPOPolicyResult(
            best_move=selected_move,
            score=score,
            completed_depth=1,
            stats=PPOPolicyStats(
                nodes=1
            ),
            principal_variation=[
                selected_move
            ],
            action_index=selected_index,
            confidence=confidence,
            used_fallback=used_fallback,
        )

    def search_depth(
        self,
        state: Any,
        depth: int = 1,
        clear_table: bool = True,
    ) -> PPOPolicyResult:
        """
        Search-compatible entry point.

        depth and clear_table are accepted for compatibility with the
        existing PokemonBattleAgent interface.
        """

        del depth
        del clear_table

        return self.select_move(state)
