from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


STATE_DIM = 8
MOVE_DIM = 8
HIDDEN_DIM = 128


class CompetitivePolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        move_dim: int = MOVE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.move_encoder = nn.Sequential(
            nn.Linear(
                move_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        move_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = (
            state_features.shape[0]
        )

        maximum_moves = (
            move_features.shape[1]
        )

        state_embedding = (
            self.state_encoder(
                state_features
            )
        )

        state_embedding = (
            state_embedding
            .unsqueeze(1)
            .expand(
                batch_size,
                maximum_moves,
                -1,
            )
        )

        move_embedding = (
            self.move_encoder(
                move_features
            )
        )

        joint_embedding = torch.cat(
            [
                state_embedding,
                move_embedding,
            ],
            dim=-1,
        )

        logits = (
            self.policy_head(
                joint_embedding
            )
            .squeeze(-1)
        )

        return logits.masked_fill(
            ~action_mask,
            -1e9,
        )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _move_value(
    move: Any,
    *keys: str,
    default: Any = None,
) -> Any:
    if isinstance(
        move,
        dict,
    ):
        for key in keys:
            if key in move:
                return move[key]

    for key in keys:
        if hasattr(
            move,
            key,
        ):
            return getattr(
                move,
                key,
            )

    return default


def move_name(
    move: Any,
) -> str:
    if isinstance(
        move,
        dict,
    ):
        return str(
            move.get("name")
            or move.get("Move Name")
            or move.get("move_name")
            or move
        )

    for attribute_name in (
        "name",
        "move_name",
        "attack_name",
    ):
        value = getattr(
            move,
            attribute_name,
            None,
        )

        if value is not None:
            return str(value)

    return str(move)


def _card_max_hp(
    pokemon: Any,
) -> float:
    card = getattr(
        pokemon,
        "card",
        None,
    )

    current_hp = _safe_float(
        getattr(
            pokemon,
            "current_hp",
            1.0,
        ),
        default=1.0,
    )

    if isinstance(
        card,
        dict,
    ):
        maximum_hp = card.get(
            "hp",
            card.get(
                "HP",
                current_hp,
            ),
        )

    else:
        maximum_hp = getattr(
            card,
            "hp",
            current_hp,
        )

    return max(
        _safe_float(
            maximum_hp,
            default=current_hp,
        ),
        1.0,
    )


def actor_and_defender(
    battle_state: Any,
) -> tuple[Any, Any, str]:
    current_player = str(
        getattr(
            battle_state,
            "current_player",
            "",
        )
    ).strip().lower()

    if current_player == "player":
        return (
            battle_state.player,
            battle_state.opponent,
            "Player",
        )

    if current_player == "opponent":
        return (
            battle_state.opponent,
            battle_state.player,
            "Opponent",
        )

    raise ValueError(
        "Unsupported current_player value: "
        f"{getattr(battle_state, 'current_player', None)!r}"
    )


def encode_state(
    battle_state: Any,
) -> np.ndarray:
    actor, defender, actor_side = (
        actor_and_defender(
            battle_state
        )
    )

    actor_active = actor.active
    defender_active = defender.active

    actor_hp_ratio = (
        _safe_float(
            actor_active.current_hp
        )
        / _card_max_hp(
            actor_active
        )
    )

    defender_hp_ratio = (
        _safe_float(
            defender_active.current_hp
        )
        / _card_max_hp(
            defender_active
        )
    )

    features = np.asarray(
        [
            np.clip(
                actor_hp_ratio,
                0.0,
                1.0,
            ),

            np.clip(
                _safe_float(
                    actor_active
                    .attached_energy
                )
                / 10.0,
                0.0,
                1.0,
            ),

            np.clip(
                _safe_float(
                    actor
                    .prize_cards_remaining
                )
                / 6.0,
                0.0,
                1.0,
            ),

            np.clip(
                defender_hp_ratio,
                0.0,
                1.0,
            ),

            np.clip(
                _safe_float(
                    defender_active
                    .attached_energy
                )
                / 10.0,
                0.0,
                1.0,
            ),

            np.clip(
                _safe_float(
                    defender
                    .prize_cards_remaining
                )
                / 6.0,
                0.0,
                1.0,
            ),

            np.clip(
                _safe_float(
                    battle_state.turn_number
                )
                / 50.0,
                0.0,
                1.0,
            ),

            (
                1.0
                if actor_side == "Player"
                else 0.0
            ),
        ],
        dtype=np.float32,
    )

    if features.shape != (
        STATE_DIM,
    ):
        raise ValueError(
            "Unexpected encoded state shape: "
            f"{features.shape}"
        )

    return features


def encode_move(
    battle_state: Any,
    move: Any,
) -> np.ndarray:
    actor, defender, _ = (
        actor_and_defender(
            battle_state
        )
    )

    damage = _safe_float(
        _move_value(
            move,
            "damage",
            "damage_numeric",
            "base_damage",
            default=0.0,
        )
    )

    energy_cost = _safe_float(
        _move_value(
            move,
            "energy_cost",
            "cost",
            "required_energy",
            default=0.0,
        )
    )

    defender_hp = _safe_float(
        defender.active.current_hp
    )

    actor_energy = _safe_float(
        actor.active.attached_energy
    )

    effect_text = str(
        _move_value(
            move,
            "effect",
            "Effect Explanation",
            default="",
        )
        or ""
    ).lower()

    is_knockout = float(
        defender_hp > 0
        and damage >= defender_hp
    )

    is_heal = float(
        any(
            term in effect_text
            for term in (
                "heal",
                "recover",
                "restore",
            )
        )
    )

    is_switch = float(
        any(
            term in effect_text
            for term in (
                "switch",
                "retreat",
                "bench",
            )
        )
    )

    is_draw = float(
        "draw" in effect_text
    )

    is_status = float(
        any(
            term in effect_text
            for term in (
                "poison",
                "burn",
                "paraly",
                "asleep",
                "confus",
                "status",
            )
        )
    )

    energy_affordable = float(
        actor_energy >= energy_cost
    )

    efficiency = (
        damage
        / max(
            energy_cost,
            1.0,
        )
    )

    expected_reward = (
        damage / 250.0
        + is_knockout
        + 0.15 * is_heal
        + 0.10 * is_switch
        + 0.10 * is_draw
        + 0.10 * is_status
        + 0.10 * energy_affordable
        + min(
            efficiency / 100.0,
            0.5,
        )
    )

    features = np.asarray(
        [
            np.clip(
                damage / 250.0,
                0.0,
                1.0,
            ),

            np.clip(
                energy_cost / 10.0,
                0.0,
                1.0,
            ),

            is_knockout,
            is_heal,
            is_switch,
            is_draw,
            is_status,

            np.clip(
                expected_reward,
                -2.0,
                2.0,
            ),
        ],
        dtype=np.float32,
    )

    if features.shape != (
        MOVE_DIM,
    ):
        raise ValueError(
            "Unexpected encoded move shape: "
            f"{features.shape}"
        )

    return features


class CompetitivePolicyEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Competitive policy checkpoint "
                f"not found: {self.checkpoint_path}"
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

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        configuration = checkpoint.get(
            "configuration",
            {},
        )

        self.model = CompetitivePolicyNetwork(
            state_dim=int(
                configuration.get(
                    "state_dim",
                    STATE_DIM,
                )
            ),

            move_dim=int(
                configuration.get(
                    "move_dim",
                    MOVE_DIM,
                )
            ),

            hidden_dim=int(
                configuration.get(
                    "hidden_dim",
                    HIDDEN_DIM,
                )
            ),
        ).to(
            self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.checkpoint_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        self.configuration = (
            configuration
        )

    @torch.no_grad()
    def score_moves(
        self,
        battle_state: Any,
        legal_moves: Sequence[Any],
    ) -> dict[str, Any]:
        legal_moves = list(
            legal_moves
        )

        if not legal_moves:
            raise ValueError(
                "At least one legal move is required."
            )

        state_array = encode_state(
            battle_state
        )

        move_array = np.stack(
            [
                encode_move(
                    battle_state,
                    move,
                )
                for move in legal_moves
            ],
            axis=0,
        )

        state_tensor = torch.tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        move_tensor = torch.tensor(
            move_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action_mask = torch.ones(
            (
                1,
                len(legal_moves),
            ),
            dtype=torch.bool,
            device=self.device,
        )

        logits = self.model(
            state_features=
                state_tensor,

            move_features=
                move_tensor,

            action_mask=
                action_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        selected_index = int(
            logits.argmax(
                dim=1
            ).item()
        )

        probability_values = (
            probabilities
            .squeeze(0)
            .detach()
            .cpu()
            .tolist()
        )

        logit_values = (
            logits
            .squeeze(0)
            .detach()
            .cpu()
            .tolist()
        )

        return {
            "selected_index":
                selected_index,

            "selected_move":
                legal_moves[
                    selected_index
                ],

            "selected_move_name":
                move_name(
                    legal_moves[
                        selected_index
                    ]
                ),

            "confidence":
                float(
                    probability_values[
                        selected_index
                    ]
                ),

            "probabilities":
                [
                    float(value)
                    for value in probability_values
                ],

            "logits":
                [
                    float(value)
                    for value in logit_values
                ],

            "move_names":
                [
                    move_name(move)
                    for move in legal_moves
                ],

            "checkpoint_epoch":
                self.checkpoint_epoch,

            "fallback":
                False,
        }

    def choose_move(
        self,
        battle_state: Any,
        legal_moves: Sequence[Any],
    ) -> Any:
        result = self.score_moves(
            battle_state,
            legal_moves,
        )

        return result[
            "selected_move"
        ]
