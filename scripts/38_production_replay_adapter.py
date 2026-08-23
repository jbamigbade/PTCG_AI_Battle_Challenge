#!/usr/bin/env python
# coding: utf-8

# # Notebook 38 — Production Replay Adapter
# 
# ## Objectives
# 
# - Load the replay and training specifications from Notebooks 35–37
# - Reuse the existing `ReplayStep`, `ReplayGame`, and `ReplayRecorder`
# - Add PPO-ready transition fields without breaking the current replay system
# - Generate observation vectors
# - Build legal-action masks
# - Compute chosen-action indices
# - Attach rewards and terminal flags
# - Validate the adapter against the canonical production contract
# - Export RL-ready CSV and Parquet datasets

# ## Section 1 — Project Setup
# 
# #### Locate the project root, configure source and report paths, initialize reproducibility settings, and prepare Notebook 38 for replay adaptation.

# In[1]:


# ============================================================
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current_path = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current_path,
        *current_path.parents,
    ]:
        if (
            (candidate / "src").is_dir()
            and (candidate / "notebooks").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOK35_REPORT_DIR = (
    REPORTS_DIR / "notebook35"
)

NOTEBOOK37_REPORT_DIR = (
    REPORTS_DIR / "notebook37"
)

NOTEBOOK38_REPORT_DIR = (
    REPORTS_DIR / "notebook38"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK38_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 38

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option(
    "display.max_columns",
    140,
)

pd.set_option(
    "display.width",
    220,
)

pd.set_option(
    "display.max_colwidth",
    180,
)

print("NOTEBOOK 38 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
)

print(
    "Notebook 35 reports    :",
    NOTEBOOK35_REPORT_DIR,
)

print(
    "Notebook 37 reports    :",
    NOTEBOOK37_REPORT_DIR,
)

print(
    "Notebook 38 reports    :",
    NOTEBOOK38_REPORT_DIR,
)

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK35_REPORT_DIR.exists()
assert NOTEBOOK38_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# ## Section 2 — Load the Production Contract
# 
# #### Load the canonical RL transition schema defined in Notebook 35 and use it as the validation contract for the adapter.

# In[2]:


# ============================================================
# SECTION 2 — LOAD PRODUCTION CONTRACT
# ============================================================

CONTRACT_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_dataset_contract.csv"
)

MAPPING_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_field_mapping.csv"
)

assert CONTRACT_FILE.exists()
assert MAPPING_FILE.exists()

production_contract_df = pd.read_csv(
    CONTRACT_FILE
)

production_mapping_df = pd.read_csv(
    MAPPING_FILE
)

required_contract_fields = (
    production_contract_df["field"]
    .astype(str)
    .tolist()
)

print("NOTEBOOK 38 — CONTRACT LOAD")
print("=" * 70)

print(
    "Contract fields        :",
    len(required_contract_fields),
)

display(
    production_contract_df
)

assert len(required_contract_fields) == 16

assert {
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
    "chosen_move",
    "chosen_move_index",
    "reward",
    "done",
}.issubset(
    set(required_contract_fields)
)

print()
print(
    "✅ SECTION 2 CONTRACT LOAD PASSED"
)


# ## Section 3 — Define the RL Transition Structure
# 
# #### Define the production `RLTransition` record that extends the existing replay information with PPO- and Behavior-Cloning-ready fields.

# In[3]:


# ============================================================
# SECTION 3 — RL TRANSITION STRUCTURE
# ============================================================

@dataclass(slots=True)
class RLTransition:
    sample_id: str
    game_index: int
    turn: int
    actor: str

    observation_vector: list[float]

    legal_moves: list[str]
    legal_move_mask: list[int]

    chosen_move: str
    chosen_move_index: int

    search_score: float
    search_depth: int
    nodes_searched: int

    reward: float
    done: bool
    winner: str

    source_dataset: str


transition_fields = list(
    RLTransition.__dataclass_fields__.keys()
)

print("NOTEBOOK 38 — RL TRANSITION STRUCTURE")
print("=" * 70)

print(
    "Transition fields      :",
    len(transition_fields),
)

for index, field in enumerate(
    transition_fields,
    start=1,
):
    print(
        f"{index:>2}. {field}"
    )

missing_contract_fields = sorted(
    set(required_contract_fields)
    - set(transition_fields)
)

extra_transition_fields = sorted(
    set(transition_fields)
    - set(required_contract_fields)
)

print()
print(
    "Missing contract fields:",
    missing_contract_fields,
)

print(
    "Extra transition fields:",
    extra_transition_fields,
)

assert not missing_contract_fields
assert not extra_transition_fields

print()
print(
    "✅ SECTION 3 RL TRANSITION STRUCTURE PASSED"
)


# ## Section 4 — Action-Mask Utilities
# 
# #### Implement reusable functions for normalizing legal moves, building action masks, and locating the chosen action within the legal-action list.

# In[4]:


# ============================================================
# SECTION 4 — ACTION-MASK UTILITIES
# ============================================================

def normalize_move_name(
    move: Any,
) -> str:
    return str(move).strip()


def normalize_legal_moves(
    legal_moves: list[Any],
) -> list[str]:
    normalized = [
        normalize_move_name(move)
        for move in legal_moves
    ]

    return list(
        dict.fromkeys(
            normalized
        )
    )


def build_legal_move_mask(
    legal_moves: list[str],
    action_vocabulary: list[str],
) -> list[int]:
    legal_move_set = set(
        legal_moves
    )

    return [
        int(action in legal_move_set)
        for action in action_vocabulary
    ]


def chosen_move_index(
    chosen_move: str,
    legal_moves: list[str],
) -> int:
    normalized_choice = normalize_move_name(
        chosen_move
    )

    if normalized_choice not in legal_moves:
        raise ValueError(
            "Chosen move is not present in the "
            f"legal move list: {normalized_choice}"
        )

    return legal_moves.index(
        normalized_choice
    )


sample_legal_moves = [
    "Attack A",
    "Retreat",
    "Pass",
    "Attack A",
]

sample_vocabulary = [
    "Attack A",
    "Attack B",
    "Retreat",
    "Pass",
]

normalized_sample_moves = normalize_legal_moves(
    sample_legal_moves
)

sample_mask = build_legal_move_mask(
    normalized_sample_moves,
    sample_vocabulary,
)

sample_choice_index = chosen_move_index(
    "Retreat",
    normalized_sample_moves,
)

print("NOTEBOOK 38 — ACTION-MASK UTILITIES")
print("=" * 70)

print(
    "Normalized legal moves:",
    normalized_sample_moves,
)

print(
    "Action mask           :",
    sample_mask,
)

print(
    "Chosen move index     :",
    sample_choice_index,
)

assert normalized_sample_moves == [
    "Attack A",
    "Retreat",
    "Pass",
]

assert sample_mask == [
    1,
    0,
    1,
    1,
]

assert sample_choice_index == 1

print()
print(
    "✅ SECTION 4 ACTION-MASK UTILITIES PASSED"
)


# ## Section 5 — Production Replay Adapter
# 
# #### Create the adapter responsible for converting production replay objects into PPO-ready RLTransition records.

# In[5]:


# ============================================================
# SECTION 5 — PRODUCTION REPLAY ADAPTER
# ============================================================

class ProductionReplayAdapter:

    def __init__(
        self,
        action_vocabulary: list[str],
    ):
        self.action_vocabulary = list(action_vocabulary)

    def create_mask(
        self,
        legal_moves: list[str],
    ) -> list[int]:

        return build_legal_move_mask(
            legal_moves,
            self.action_vocabulary,
        )

    def action_index(
        self,
        chosen_move: str,
        legal_moves: list[str],
    ) -> int:

        return chosen_move_index(
            chosen_move,
            legal_moves,
        )

    def normalize_moves(
        self,
        legal_moves: list[str],
    ) -> list[str]:

        return normalize_legal_moves(
            legal_moves
        )


adapter = ProductionReplayAdapter(
    action_vocabulary=[
        "Attack A",
        "Attack B",
        "Retreat",
        "Pass",
    ]
)

normalized = adapter.normalize_moves(
    [
        "Attack A",
        "Retreat",
        "Pass",
    ]
)

mask = adapter.create_mask(
    normalized
)

index = adapter.action_index(
    "Retreat",
    normalized,
)

print("NOTEBOOK 38 — PRODUCTION REPLAY ADAPTER")
print("=" * 70)

print("Normalized :", normalized)
print("Mask       :", mask)
print("Index      :", index)

assert index == 1
assert mask == [1,0,1,1]

print()
print("✅ SECTION 5 REPLAY ADAPTER PASSED")


# ## Section 6 — Observation Adapter Interface
# 
# #### Define the interface that converts a battle state into the production observation vector.

# In[6]:


# ============================================================
# SECTION 6 — OBSERVATION INTERFACE
# ============================================================

class ObservationProvider:

    def build_observation(
        self,
        battle_state,
    ):

        raise NotImplementedError(
            "Notebook 33 supplies this implementation."
        )


class MockObservationProvider(
    ObservationProvider
):

    def build_observation(
        self,
        battle_state,
    ):

        return [
            0.10,
            0.25,
            0.75,
            1.00,
        ]


provider = MockObservationProvider()

vector = provider.build_observation(
    None
)

print("NOTEBOOK 38 — OBSERVATION INTERFACE")
print("=" * 70)

print("Observation length :", len(vector))
print(vector)

assert isinstance(vector, list)
assert len(vector) == 4

print()
print("✅ SECTION 6 OBSERVATION INTERFACE PASSED")


# ## Section 7 — Reward Provider
# 
# #### Define the reward interface used during replay conversion.

# In[7]:


# ============================================================
# SECTION 7 — REWARD PROVIDER
# ============================================================

class RewardProvider:

    def reward(
        self,
        replay_step,
    ) -> float:

        raise NotImplementedError


class MockRewardProvider(
    RewardProvider
):

    def reward(
        self,
        replay_step,
    ):

        return 1.0


reward_provider = MockRewardProvider()

reward = reward_provider.reward(
    None
)

print("NOTEBOOK 38 — REWARD PROVIDER")
print("=" * 70)

print("Reward :", reward)

assert reward == 1.0

print()
print("✅ SECTION 7 REWARD PROVIDER PASSED")


# ## Section 8 — RL Transition Builder
# 
# #### Combine the observation provider, replay adapter, and reward provider into a single PPO transition builder.

# In[8]:


# ============================================================
# SECTION 8 — CONTRACT-COMPATIBLE TRANSITION BUILDER
# ============================================================

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RLTransition:
    sample_id: str
    game_index: int
    turn: int
    actor: str

    observation_vector: list[float]

    legal_moves: list[str]
    legal_move_mask: list[int]

    chosen_move: str
    chosen_move_index: int

    search_score: float
    search_depth: int
    nodes_searched: int

    reward: float
    done: bool
    winner: str

    source_dataset: str


def get_record_value(
    record: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    if isinstance(record, dict):
        return record.get(
            field_name,
            default,
        )

    return getattr(
        record,
        field_name,
        default,
    )


class TransitionBuilder:

    def __init__(
        self,
        observation_provider,
        replay_adapter,
        reward_provider,
        source_dataset: str = (
            "production_replay_adapter"
        ),
    ):
        self.observation_provider = (
            observation_provider
        )

        self.replay_adapter = (
            replay_adapter
        )

        self.reward_provider = (
            reward_provider
        )

        self.source_dataset = (
            source_dataset
        )

    def build(
        self,
        replay_step: Any,
        *,
        game_index: int = 0,
        winner: str = "",
        done: bool = False,
        nodes_searched: int = 0,
    ) -> RLTransition:
        legal_moves = (
            self.replay_adapter
            .normalize_moves(
                get_record_value(
                    replay_step,
                    "legal_moves",
                    [],
                )
            )
        )

        if not legal_moves:
            raise ValueError(
                "Replay step contains no legal moves."
            )

        chosen_move = normalize_move_name(
            get_record_value(
                replay_step,
                "chosen_move",
                "",
            )
        )

        chosen_index = (
            self.replay_adapter
            .action_index(
                chosen_move,
                legal_moves,
            )
        )

        observation = (
            self.observation_provider
            .build_observation(
                replay_step
            )
        )

        reward = float(
            self.reward_provider.reward(
                replay_step
            )
        )

        game_id = str(
            get_record_value(
                replay_step,
                "game_id",
                game_index,
            )
        )

        turn = int(
            get_record_value(
                replay_step,
                "turn",
                0,
            )
        )

        actor = str(
            get_record_value(
                replay_step,
                "player",
                "Unknown",
            )
        )

        search_score = float(
            get_record_value(
                replay_step,
                "evaluation",
                0.0,
            )
        )

        search_depth = int(
            get_record_value(
                replay_step,
                "search_depth",
                0,
            )
        )

        sample_id = (
            f"{game_id}_turn_{turn}_"
            f"{actor}_{chosen_index}"
        )

        return RLTransition(
            sample_id=sample_id,
            game_index=int(game_index),
            turn=turn,
            actor=actor,
            observation_vector=[
                float(value)
                for value in observation
            ],
            legal_moves=legal_moves,
            legal_move_mask=(
                self.replay_adapter
                .create_mask(
                    legal_moves
                )
            ),
            chosen_move=chosen_move,
            chosen_move_index=chosen_index,
            search_score=search_score,
            search_depth=search_depth,
            nodes_searched=int(
                nodes_searched
            ),
            reward=reward,
            done=bool(done),
            winner=str(winner),
            source_dataset=(
                self.source_dataset
            ),
        )


builder = TransitionBuilder(
    provider,
    adapter,
    reward_provider,
)

sample_replay_step = {
    "game_id": "test_game_001",
    "turn": 1,
    "player": "Player",
    "legal_moves": [
        "Attack A",
        "Retreat",
        "Pass",
    ],
    "chosen_move": "Retreat",
    "evaluation": 25.0,
    "search_depth": 3,
}

transition = builder.build(
    sample_replay_step,
    game_index=0,
    winner="Player",
    done=False,
    nodes_searched=12,
)

print("NOTEBOOK 38 — TRANSITION BUILDER")
print("=" * 70)

display(
    pd.DataFrame(
        [
            asdict(
                transition
            )
        ]
    )
)

assert transition.sample_id
assert transition.chosen_move_index == 1
assert transition.search_score == 25.0
assert transition.search_depth == 3
assert transition.nodes_searched == 12
assert transition.done is False

print()
print(
    "✅ SECTION 8 TRANSITION BUILDER PASSED"
)


# ## Section 9 — Transition Validation
# 
# #### Verify that every required PPO training field exists.

# In[9]:


# ============================================================
# SECTION 9 — FULL CONTRACT VALIDATION
# ============================================================

from dataclasses import fields

transition_fields = [
    field.name
    for field in fields(
        transition
    )
]

contract_validation_df = pd.DataFrame(
    {
        "field": required_contract_fields,
        "exists": [
            field in transition_fields
            for field in (
                required_contract_fields
            )
        ],
    }
)

display(
    contract_validation_df
)

missing_fields = (
    contract_validation_df.loc[
        ~contract_validation_df["exists"],
        "field",
    ]
    .tolist()
)

assert not missing_fields, (
    "Missing contract fields: "
    f"{missing_fields}"
)

assert len(transition_fields) == 16

print()
print("NOTEBOOK 38 — CONTRACT VALIDATION")
print("=" * 70)

print(
    "Fields validated       :",
    int(
        contract_validation_df[
            "exists"
        ].sum()
    ),
    "/",
    len(contract_validation_df),
)

print(
    "Transition field count :",
    len(transition_fields),
)

print()
print(
    "✅ SECTION 9 CONTRACT VALIDATION PASSED"
)


# ## Section 10 — Batch Transition Builder
# 
# #### Convert a sequence of replay-step records into validated RL transitions.
# 
# #### The builder accepts both dictionaries and replay objects, allowing it to work with the existing `ReplayStep` dataclass without modifying the original replay system.

# In[10]:


# ============================================================
# SECTION 10 — FULL BATCH TRANSITION BUILDER
# ============================================================

class BatchTransitionBuilder:

    def __init__(
        self,
        transition_builder: TransitionBuilder,
    ):
        self.transition_builder = (
            transition_builder
        )

    def build_many(
        self,
        replay_steps: list[Any],
        *,
        game_index: int,
        winner: str,
    ) -> list[RLTransition]:
        transitions = []

        last_index = (
            len(replay_steps) - 1
        )

        for index, replay_step in enumerate(
            replay_steps
        ):
            transitions.append(
                self.transition_builder.build(
                    replay_step,
                    game_index=game_index,
                    winner=winner,
                    done=(
                        index == last_index
                    ),
                )
            )

        return transitions

    @staticmethod
    def to_dataframe(
        transitions: list[RLTransition],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                asdict(
                    transition
                )
                for transition
                in transitions
            ]
        )


batch_builder = BatchTransitionBuilder(
    builder
)

sample_replay_steps = [
    {
        "game_id": "batch_game_001",
        "turn": 1,
        "player": "Player",
        "legal_moves": [
            "Attack A",
            "Retreat",
            "Pass",
        ],
        "chosen_move": "Attack A",
        "evaluation": 20.0,
        "search_depth": 2,
    },
    {
        "game_id": "batch_game_001",
        "turn": 2,
        "player": "Opponent",
        "legal_moves": [
            "Attack A",
            "Attack B",
            "Pass",
        ],
        "chosen_move": "Attack B",
        "evaluation": 15.0,
        "search_depth": 2,
    },
    {
        "game_id": "batch_game_001",
        "turn": 3,
        "player": "Player",
        "legal_moves": [
            "Retreat",
            "Pass",
        ],
        "chosen_move": "Pass",
        "evaluation": 5.0,
        "search_depth": 1,
    },
]

batch_transitions = (
    batch_builder.build_many(
        sample_replay_steps,
        game_index=0,
        winner="Player",
    )
)

transition_batch_df = (
    batch_builder.to_dataframe(
        batch_transitions
    )
)

print("NOTEBOOK 38 — BATCH TRANSITION BUILDER")
print("=" * 70)

print(
    "Transitions generated  :",
    len(batch_transitions),
)

display(
    transition_batch_df
)

assert len(
    transition_batch_df
) == 3

assert set(
    required_contract_fields
) == set(
    transition_batch_df.columns
)

assert (
    transition_batch_df[
        "done"
    ].tolist()
    == [
        False,
        False,
        True,
    ]
)

print()
print(
    "✅ SECTION 10 BATCH BUILDER PASSED"
)


# In[14]:


def build(
    self,
    replay_step,
):
    observation = (
        self.observation_provider
        .build_observation(
            replay_step
        )
    )

    legal_moves = (
        self.replay_adapter
        .normalize_moves(
            get_record_value(
                replay_step,
                "legal_moves",
                [],
            )
        )
    )

    chosen_move = normalize_move_name(
        get_record_value(
            replay_step,
            "chosen_move",
            "",
        )
    )

    if not legal_moves:
        raise ValueError(
            "Replay step contains no legal moves."
        )

    mask = (
        self.replay_adapter
        .create_mask(
            legal_moves
        )
    )

    move_index = (
        self.replay_adapter
        .action_index(
            chosen_move,
            legal_moves,
        )
    )

    reward = (
        self.reward_provider
        .reward(
            replay_step
        )
    )

    done = bool(
        get_record_value(
            replay_step,
            "done",
            False,
        )
    )

    return RLTransition(
        observation=observation,
        legal_moves=legal_moves,
        legal_move_mask=mask,
        chosen_move=chosen_move,
        chosen_move_index=move_index,
        reward=float(reward),
        done=done,
    )


# In[15]:


# ============================================================
# SECTION 10 — FULL BATCH TRANSITION BUILDER
# ============================================================

class BatchTransitionBuilder:

    def __init__(
        self,
        transition_builder: TransitionBuilder,
    ):
        self.transition_builder = (
            transition_builder
        )

    def build_many(
        self,
        replay_steps: list[Any],
        *,
        game_index: int,
        winner: str,
    ) -> list[RLTransition]:
        transitions = []

        last_index = (
            len(replay_steps) - 1
        )

        for index, replay_step in enumerate(
            replay_steps
        ):
            transition = (
                self.transition_builder.build(
                    replay_step,
                    game_index=game_index,
                    winner=winner,
                    done=(
                        index == last_index
                    ),
                )
            )

            transitions.append(
                transition
            )

        return transitions

    @staticmethod
    def to_dataframe(
        transitions: list[RLTransition],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                asdict(
                    transition
                )
                for transition
                in transitions
            ]
        )


batch_builder = BatchTransitionBuilder(
    builder
)

sample_replay_steps = [
    {
        "game_id": "batch_game_001",
        "turn": 1,
        "player": "Player",
        "legal_moves": [
            "Attack A",
            "Retreat",
            "Pass",
        ],
        "chosen_move": "Attack A",
        "evaluation": 20.0,
        "search_depth": 2,
    },
    {
        "game_id": "batch_game_001",
        "turn": 2,
        "player": "Opponent",
        "legal_moves": [
            "Attack A",
            "Attack B",
            "Pass",
        ],
        "chosen_move": "Attack B",
        "evaluation": 15.0,
        "search_depth": 2,
    },
    {
        "game_id": "batch_game_001",
        "turn": 3,
        "player": "Player",
        "legal_moves": [
            "Retreat",
            "Pass",
        ],
        "chosen_move": "Pass",
        "evaluation": 5.0,
        "search_depth": 1,
    },
]

batch_transitions = (
    batch_builder.build_many(
        sample_replay_steps,
        game_index=0,
        winner="Player",
    )
)

transition_batch_df = (
    batch_builder.to_dataframe(
        batch_transitions
    )
)

print("NOTEBOOK 38 — BATCH TRANSITION BUILDER")
print("=" * 70)

print(
    "Replay steps           :",
    len(sample_replay_steps),
)

print(
    "Transitions generated  :",
    len(batch_transitions),
)

display(
    transition_batch_df
)

assert len(
    batch_transitions
) == 3

assert len(
    transition_batch_df
) == 3

assert set(
    required_contract_fields
) == set(
    transition_batch_df.columns
)

assert (
    transition_batch_df[
        "chosen_move_index"
    ].tolist()
    == [
        0,
        1,
        1,
    ]
)

assert (
    transition_batch_df[
        "done"
    ].tolist()
    == [
        False,
        False,
        True,
    ]
)

print()
print(
    "✅ SECTION 10 BATCH BUILDER PASSED"
)


# ## Section 11 — Batch Dataset Validation
# 
# #### Validate the generated transition batch for schema completeness, action-mask consistency, valid chosen-action indices, and terminal-state coverage.

# In[17]:


# ============================================================
# SECTION 11 — FULL BATCH DATASET VALIDATION
# ============================================================

batch_required_columns = list(
    required_contract_fields
)

missing_batch_columns = [
    column
    for column in batch_required_columns
    if column not in transition_batch_df.columns
]

validation_rows = []

for row_index, row in transition_batch_df.iterrows():
    legal_moves = row[
        "legal_moves"
    ]

    legal_mask = row[
        "legal_move_mask"
    ]

    chosen_index = int(
        row["chosen_move_index"]
    )

    chosen_move = row[
        "chosen_move"
    ]

    observation_vector = row[
        "observation_vector"
    ]

    index_in_range = (
        0
        <= chosen_index
        < len(legal_moves)
    )

    selected_move_matches = (
        index_in_range
        and legal_moves[
            chosen_index
        ]
        == chosen_move
    )

    mask_length_valid = (
        len(legal_mask)
        == len(
            adapter.action_vocabulary
        )
    )

    chosen_move_is_legal = (
        chosen_move
        in legal_moves
    )

    observation_valid = (
        isinstance(
            observation_vector,
            list,
        )
        and len(
            observation_vector
        ) > 0
    )

    sample_id_valid = bool(
        str(
            row["sample_id"]
        ).strip()
    )

    validation_rows.append(
        {
            "row_index": row_index,
            "index_in_range": (
                index_in_range
            ),
            "selected_move_matches": (
                selected_move_matches
            ),
            "mask_length_valid": (
                mask_length_valid
            ),
            "chosen_move_is_legal": (
                chosen_move_is_legal
            ),
            "observation_valid": (
                observation_valid
            ),
            "sample_id_valid": (
                sample_id_valid
            ),
        }
    )

batch_validation_df = pd.DataFrame(
    validation_rows
)

print("NOTEBOOK 38 — BATCH DATASET VALIDATION")
print("=" * 70)

display(
    batch_validation_df
)

print()
print(
    "Missing columns        :",
    missing_batch_columns,
)

print(
    "Terminal rows          :",
    int(
        transition_batch_df[
            "done"
        ].sum()
    ),
)

print(
    "Unique chosen moves    :",
    transition_batch_df[
        "chosen_move"
    ].nunique(),
)

print(
    "Contract columns       :",
    len(
        transition_batch_df.columns
    ),
)

quality_columns = [
    "index_in_range",
    "selected_move_matches",
    "mask_length_valid",
    "chosen_move_is_legal",
    "observation_valid",
    "sample_id_valid",
]

assert not missing_batch_columns, (
    "Missing batch columns: "
    f"{missing_batch_columns}"
)

assert (
    batch_validation_df[
        quality_columns
    ]
    .all()
    .all()
)

assert (
    transition_batch_df[
        "done"
    ]
    .any()
)

assert (
    transition_batch_df[
        "chosen_move"
    ]
    .nunique()
    >= 2
)

assert set(
    transition_batch_df.columns
) == set(
    required_contract_fields
)

print()
print(
    "✅ SECTION 11 BATCH VALIDATION PASSED"
)


# ## Section 12 — Production Replay Conversion
# 
# #### Verify that ReplayStep dataclass objects from the production replay system can be converted into PPO transitions without modification.

# In[18]:


# ============================================================
# SECTION 12 — PRODUCTION REPLAY CONVERSION
# ============================================================

from dataclasses import fields

from src.training.replay import ReplayStep


replay_step_fields = [
    field.name
    for field in fields(
        ReplayStep
    )
]

required_fields = [
    "legal_moves",
    "chosen_move",
]

conversion_audit = pd.DataFrame(
    {
        "required_field": required_fields,
        "exists": [
            field in replay_step_fields
            for field in required_fields
        ],
    }
)

display(
    conversion_audit
)

missing = (
    conversion_audit.loc[
        ~conversion_audit["exists"],
        "required_field",
    ]
    .tolist()
)

print("NOTEBOOK 38 — PRODUCTION REPLAY")
print("=" * 70)

print(
    "ReplayStep fields      :",
    len(replay_step_fields),
)

print(
    "Required fields        :",
    len(required_fields),
)

print(
    "Missing fields         :",
    missing,
)

assert not missing, (
    "ReplayStep is missing required fields: "
    f"{missing}"
)

print()
print(
    "✅ SECTION 12 PRODUCTION REPLAY PASSED"
)


# ## Section 13 — Production Integration Summary
# 
# #### Document how the production replay adapter connects the existing replay system, observation pipeline, legal-action representation, and RL transition contract.

# In[19]:


# ============================================================
# SECTION 13 — PRODUCTION INTEGRATION SUMMARY
# ============================================================

integration_df = pd.DataFrame(
    [
        {
            "stage": 1,
            "source": "src.training.replay.ReplayStep",
            "output": "Replay decision record",
            "status": "reused",
        },
        {
            "stage": 2,
            "source": "ObservationProvider",
            "output": "observation_vector",
            "status": "adapter interface ready",
        },
        {
            "stage": 3,
            "source": "ProductionReplayAdapter",
            "output": (
                "legal_moves, legal_move_mask, "
                "chosen_move_index"
            ),
            "status": "implemented",
        },
        {
            "stage": 4,
            "source": "RewardProvider",
            "output": "reward",
            "status": "adapter interface ready",
        },
        {
            "stage": 5,
            "source": "TransitionBuilder",
            "output": "16-field RLTransition",
            "status": "implemented",
        },
        {
            "stage": 6,
            "source": "BatchTransitionBuilder",
            "output": "RL transition dataset",
            "status": "implemented",
        },
    ]
)

print("NOTEBOOK 38 — INTEGRATION SUMMARY")
print("=" * 70)

display(
    integration_df
)

print()
print(
    "Integration stages     :",
    len(integration_df),
)

print(
    "Transition columns     :",
    len(transition_batch_df.columns),
)

assert len(integration_df) == 6

assert set(
    transition_batch_df.columns
) == set(
    required_contract_fields
)

print()
print(
    "✅ SECTION 13 INTEGRATION SUMMARY PASSED"
)


# ## Section 14 — Export Adapter Artifacts
# 
# #### Export the validated transition fixture, adapter contract, integration summary, and quality audit.
# 
# #### The generated transition rows are controlled validation fixtures, not yet full simulator-generated training data.

# In[20]:


# ============================================================
# SECTION 14 — EXPORT ADAPTER ARTIFACTS
# ============================================================

REPORT_DIR = NOTEBOOK38_REPORT_DIR

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRANSITION_CSV_FILE = (
    REPORT_DIR
    / "rl_transition_validation_fixture.csv"
)

TRANSITION_PARQUET_FILE = (
    REPORT_DIR
    / "rl_transition_validation_fixture.parquet"
)

CONTRACT_FILE = (
    REPORT_DIR
    / "rl_transition_contract.csv"
)

QUALITY_FILE = (
    REPORT_DIR
    / "rl_transition_quality_audit.csv"
)

INTEGRATION_FILE = (
    REPORT_DIR
    / "replay_adapter_integration.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "replay_adapter_summary.json"
)

# CSV requires nested values to be serialized.
transition_csv_df = (
    transition_batch_df.copy()
)

nested_columns = [
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
]

for column in nested_columns:
    transition_csv_df[column] = (
        transition_csv_df[column]
        .apply(
            lambda value: json.dumps(
                value,
                ensure_ascii=False,
            )
        )
    )

transition_csv_df.to_csv(
    TRANSITION_CSV_FILE,
    index=False,
)

# Parquet preserves nested list columns.
transition_batch_df.to_parquet(
    TRANSITION_PARQUET_FILE,
    index=False,
)

production_contract_df.to_csv(
    CONTRACT_FILE,
    index=False,
)

batch_validation_df.to_csv(
    QUALITY_FILE,
    index=False,
)

integration_df.to_csv(
    INTEGRATION_FILE,
    index=False,
)

adapter_summary = {
    "notebook": 38,
    "name": "Production Replay Adapter",
    "schema_field_count": len(
        required_contract_fields
    ),
    "validation_transition_count": len(
        transition_batch_df
    ),
    "terminal_transition_count": int(
        transition_batch_df["done"].sum()
    ),
    "unique_chosen_moves": int(
        transition_batch_df[
            "chosen_move"
        ].nunique()
    ),
    "all_quality_checks_passed": bool(
        batch_validation_df[
            [
                "index_in_range",
                "selected_move_matches",
                "mask_length_valid",
                "chosen_move_is_legal",
                "observation_valid",
                "sample_id_valid",
            ]
        ]
        .all()
        .all()
    ),
    "dataset_status": "VALIDATION_FIXTURE",
    "production_simulator_data": False,
}

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        adapter_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("NOTEBOOK 38 — ADAPTER EXPORT")
print("=" * 70)

for label, path in [
    ("CSV fixture", TRANSITION_CSV_FILE),
    ("Parquet fixture", TRANSITION_PARQUET_FILE),
    ("Contract", CONTRACT_FILE),
    ("Quality audit", QUALITY_FILE),
    ("Integration", INTEGRATION_FILE),
    ("Summary", SUMMARY_FILE),
]:
    print(
        f"{label:<20}:",
        path.relative_to(
            PROJECT_ROOT
        ),
    )

assert TRANSITION_CSV_FILE.exists()
assert TRANSITION_PARQUET_FILE.exists()
assert CONTRACT_FILE.exists()
assert QUALITY_FILE.exists()
assert INTEGRATION_FILE.exists()
assert SUMMARY_FILE.exists()

print()
print(
    "✅ SECTION 14 ADAPTER EXPORT PASSED"
)


# ## Section 15 — Reload and Final Validation
# 
# #### Reload every exported artifact and verify that the adapter outputs remain complete, readable, and consistent with the production contract.

# In[21]:


# ============================================================
# SECTION 15 — RELOAD AND FINAL VALIDATION
# ============================================================

csv_reload_df = pd.read_csv(
    TRANSITION_CSV_FILE
)

parquet_reload_df = pd.read_parquet(
    TRANSITION_PARQUET_FILE
)

contract_reload_df = pd.read_csv(
    CONTRACT_FILE
)

quality_reload_df = pd.read_csv(
    QUALITY_FILE
)

integration_reload_df = pd.read_csv(
    INTEGRATION_FILE
)

with SUMMARY_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    summary_reload = json.load(
        file
    )

print("NOTEBOOK 38 — FINAL VALIDATION")
print("=" * 70)

print(
    "CSV rows              :",
    len(csv_reload_df),
)

print(
    "Parquet rows          :",
    len(parquet_reload_df),
)

print(
    "Parquet columns       :",
    len(parquet_reload_df.columns),
)

print(
    "Contract fields       :",
    len(contract_reload_df),
)

print(
    "Quality rows          :",
    len(quality_reload_df),
)

print(
    "Integration stages    :",
    len(integration_reload_df),
)

print(
    "Dataset status        :",
    summary_reload[
        "dataset_status"
    ],
)

print(
    "Simulator data        :",
    summary_reload[
        "production_simulator_data"
    ],
)

assert len(
    csv_reload_df
) == len(
    transition_batch_df
)

assert len(
    parquet_reload_df
) == len(
    transition_batch_df
)

assert set(
    parquet_reload_df.columns
) == set(
    required_contract_fields
)

assert len(
    contract_reload_df
) == 16

assert (
    summary_reload[
        "all_quality_checks_passed"
    ]
    is True
)

assert (
    summary_reload[
        "dataset_status"
    ]
    == "VALIDATION_FIXTURE"
)

assert (
    summary_reload[
        "production_simulator_data"
    ]
    is False
)

display(
    parquet_reload_df.head()
)

print()
print(
    "🏆 NOTEBOOK 38 COMPLETE"
)


# In[ ]:




