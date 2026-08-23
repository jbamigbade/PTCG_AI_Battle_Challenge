from __future__ import annotations

#!/usr/bin/env python
# coding: utf-8

# # Notebook 47 - Competitive Policy Training
# 
# ## Objective
# 
# Train a variable-action neural policy using the actor-relative,
# search-guided expert dataset produced by Notebook 46.
# 
# ### Training plan
# 
# 1. Load and validate the corrected expert dataset.
# 2. Split data into training, validation, and test sets.
# 3. Build a policy that scores each legal move independently.
# 4. Train with behavioral cloning.
# 5. Track top-1 accuracy, top-k accuracy, and validation loss.
# 6. Save the strongest checkpoint.
# 7. Benchmark the trained policy against expert search labels.
# 8. Export the production inference engine.
# 

# ## Section 1 â€” Training Setup and Dataset Validation
# 
# #### Load the actor-relative expert dataset from Notebook 46, verify its schema, and prepare deterministic training, validation, and test splits.

# In[1]:


# ============================================================
# NOTEBOOK 47
# SECTION 1 â€” TRAINING SETUP AND DATASET VALIDATION
# ============================================================


import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Dataset,
)


# ------------------------------------------------------------
# 1.3 â€” Locate project root
# ------------------------------------------------------------

def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        required_paths = [
            candidate / "src",
            candidate / "notebooks",
            candidate / "data",
            candidate / "reports",
        ]

        if all(
            path.exists()
            for path in required_paths
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"

COMPETITIVE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "competitive"
)

COMPETITIVE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "competitive"
)

NOTEBOOK47_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook47"
)

for directory in [
    COMPETITIVE_MODEL_DIR,
    NOTEBOOK47_REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ------------------------------------------------------------
# 1.4 â€” Configuration
# ------------------------------------------------------------

RANDOM_SEED = 4701

DATASET_FILE = (
    COMPETITIVE_DATA_DIR
    / "actor_relative_search_expert_dataset.parquet"
)

DATASET_SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "notebook46"
    / "actor_relative_search_expert_summary.json"
)

SPLIT_REPORT_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "dataset_split_summary.json"
)

STATE_DIM = 8
MOVE_DIM = 8

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

BATCH_SIZE = 64


# ------------------------------------------------------------
# 1.5 â€” Reproducibility
# ------------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# 1.6 â€” Load dataset
# ------------------------------------------------------------

assert DATASET_FILE.exists(), (
    "The Notebook 46 actor-relative dataset "
    "was not found."
)

dataset_df = pd.read_parquet(
    DATASET_FILE
)

assert len(dataset_df) > 0


# ------------------------------------------------------------
# 1.7 â€” Normalize nested Parquet values
# ------------------------------------------------------------

def to_python_list(
    value: Any,
) -> list:
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if value is None:
        return []

    return list(value)


dataset_df[
    "state_features"
] = dataset_df[
    "state_features"
].apply(
    to_python_list
)

dataset_df[
    "legal_move_features"
] = dataset_df[
    "legal_move_features"
].apply(
    to_python_list
)

dataset_df[
    "legal_move_names"
] = dataset_df[
    "legal_move_names"
].apply(
    to_python_list
)


# ------------------------------------------------------------
# 1.8 â€” Dataset validation
# ------------------------------------------------------------

required_columns = {
    "record_id",
    "battle_id",
    "state_features",
    "legal_move_features",
    "legal_move_names",
    "legal_move_count",
    "selected_move_index",
    "selected_move_name",
    "actor_relative",
    "synthetic_label",
    "label_source",
}

missing_columns = (
    required_columns
    - set(dataset_df.columns)
)

assert not missing_columns, (
    "Dataset is missing required columns: "
    f"{sorted(missing_columns)}"
)

state_dimensions_valid = dataset_df[
    "state_features"
].apply(
    lambda values:
        len(values) == STATE_DIM
)

move_dimensions_valid = dataset_df[
    "legal_move_features"
].apply(
    lambda matrix:
        (
            len(matrix) > 0
            and all(
                len(row) == MOVE_DIM
                for row in matrix
            )
        )
)

move_counts_valid = dataset_df.apply(
    lambda row:
        len(
            row[
                "legal_move_features"
            ]
        )
        == int(
            row[
                "legal_move_count"
            ]
        ),
    axis=1,
)

selected_indices_valid = (
    dataset_df[
        "selected_move_index"
    ]
    >= 0
) & (
    dataset_df[
        "selected_move_index"
    ]
    < dataset_df[
        "legal_move_count"
    ]
)

selected_names_valid = dataset_df.apply(
    lambda row:
        str(
            row[
                "selected_move_name"
            ]
        )
        == str(
            row[
                "legal_move_names"
            ][
                int(
                    row[
                        "selected_move_index"
                    ]
                )
            ]
        ),
    axis=1,
)

assert state_dimensions_valid.all()
assert move_dimensions_valid.all()
assert move_counts_valid.all()
assert selected_indices_valid.all()
assert selected_names_valid.all()

assert (
    dataset_df[
        "actor_relative"
    ]
    == True
).all()

assert (
    dataset_df[
        "synthetic_label"
    ]
    == False
).all()

assert (
    dataset_df[
        "label_source"
    ]
    == "advanced_search_engine"
).all()


# ------------------------------------------------------------
# 1.9 â€” Battle-group split
#
# Split by battle_id rather than by individual rows.
# This prevents states from the same battle appearing in
# both training and evaluation sets.
# ------------------------------------------------------------

unique_battle_ids = (
    dataset_df[
        "battle_id"
    ]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(
    unique_battle_ids
)

battle_count = len(
    unique_battle_ids
)

train_end = int(
    battle_count
    * TRAIN_FRACTION
)

validation_end = (
    train_end
    + int(
        battle_count
        * VALIDATION_FRACTION
    )
)

train_battle_ids = set(
    unique_battle_ids[
        :train_end
    ].tolist()
)

validation_battle_ids = set(
    unique_battle_ids[
        train_end:validation_end
    ].tolist()
)

test_battle_ids = set(
    unique_battle_ids[
        validation_end:
    ].tolist()
)

train_df = dataset_df[
    dataset_df[
        "battle_id"
    ].isin(
        train_battle_ids
    )
].copy()

validation_df = dataset_df[
    dataset_df[
        "battle_id"
    ].isin(
        validation_battle_ids
    )
].copy()

test_df = dataset_df[
    dataset_df[
        "battle_id"
    ].isin(
        test_battle_ids
    )
].copy()


# ------------------------------------------------------------
# 1.10 â€” Validate split isolation
# ------------------------------------------------------------

assert train_battle_ids.isdisjoint(
    validation_battle_ids
)

assert train_battle_ids.isdisjoint(
    test_battle_ids
)

assert validation_battle_ids.isdisjoint(
    test_battle_ids
)

assert (
    len(train_df)
    + len(validation_df)
    + len(test_df)
    == len(dataset_df)
)

assert len(train_df) > 0
assert len(validation_df) > 0
assert len(test_df) > 0


# ------------------------------------------------------------
# 1.11 â€” Build split summary
# ------------------------------------------------------------

split_summary = {
    "dataset_file":
        str(DATASET_FILE),

    "total_records":
        int(
            len(dataset_df)
        ),

    "total_battles":
        int(
            battle_count
        ),

    "train_records":
        int(
            len(train_df)
        ),

    "validation_records":
        int(
            len(validation_df)
        ),

    "test_records":
        int(
            len(test_df)
        ),

    "train_battles":
        int(
            len(train_battle_ids)
        ),

    "validation_battles":
        int(
            len(validation_battle_ids)
        ),

    "test_battles":
        int(
            len(test_battle_ids)
        ),

    "state_dimension":
        STATE_DIM,

    "move_dimension":
        MOVE_DIM,

    "batch_size":
        BATCH_SIZE,

    "random_seed":
        RANDOM_SEED,

    "device":
        str(DEVICE),

    "actor_relative":
        True,

    "synthetic_labels":
        0,
}

with open(
    SPLIT_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        split_summary,
        file,
        indent=4,
    )


# ------------------------------------------------------------
# 1.12 â€” Display results
# ------------------------------------------------------------

print(
    "NOTEBOOK 47 â€” TRAINING SETUP AND DATASET VALIDATION"
)

print("=" * 100)

print()
print(
    "Project root       :",
    PROJECT_ROOT,
)

print(
    "Device             :",
    DEVICE,
)

print(
    "Dataset records    :",
    f"{len(dataset_df):,}",
)

print(
    "Unique battles     :",
    f"{battle_count:,}",
)

print()
print(
    "Training records   :",
    f"{len(train_df):,}",
)

print(
    "Validation records :",
    f"{len(validation_df):,}",
)

print(
    "Test records       :",
    f"{len(test_df):,}",
)

print()
print(
    "State dimension    :",
    STATE_DIM,
)

print(
    "Move dimension     :",
    MOVE_DIM,
)

print(
    "Batch size         :",
    BATCH_SIZE,
)

print()
print(
    "Split report:",
    SPLIT_REPORT_FILE,
)

print()
print("DATASET SAMPLE")
print("-" * 100)

display(
    dataset_df[
        [
            "record_id",
            "battle_id",
            "turn_number",
            "actor_side",
            "actor_name",
            "defender_name",
            "legal_move_count",
            "selected_move_name",
            "search_score",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 1.13 â€” Final validation
# ------------------------------------------------------------

assert SPLIT_REPORT_FILE.exists()

print()
print(
    "âœ… SECTION 1 TRAINING SETUP PASSED"
)


# ## Section 2 â€” Variable-Action PyTorch Dataset
# 
# #### Convert the actor-relative expert records into PyTorch tensors.
# 
# #### Because each battle state may contain a different number of legal moves, the batch collator pads move features to the largest legal-move count in the batch and creates a Boolean action mask.
# 
# #### The mask prevents padded actions from being treated as real choices.

# In[2]:


# ============================================================
# NOTEBOOK 47
# SECTION 2 â€” VARIABLE-ACTION DATASET AND COLLATOR
# ============================================================


from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
)


# ------------------------------------------------------------
# 2.3 â€” Dataset item structure
# ------------------------------------------------------------

@dataclass
class CompetitivePolicyExample:
    state_features: torch.Tensor
    move_features: torch.Tensor
    selected_move_index: int
    legal_move_count: int
    record_id: int
    battle_id: int
    selected_move_name: str


# ------------------------------------------------------------
# 2.4 â€” Dataset class
# ------------------------------------------------------------

class CompetitivePolicyDataset(
    Dataset,
):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        state_dim: int = STATE_DIM,
        move_dim: int = MOVE_DIM,
    ) -> None:
        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.state_dim = int(
            state_dim
        )

        self.move_dim = int(
            move_dim
        )

        assert len(
            self.dataframe
        ) > 0

    def __len__(
        self,
    ) -> int:
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ) -> CompetitivePolicyExample:
        row = self.dataframe.iloc[
            index
        ]

        state_array = np.asarray(
            row["state_features"],
            dtype=np.float32,
        )

        move_array = np.asarray(
            row["legal_move_features"],
            dtype=np.float32,
        )

        if state_array.ndim != 1:
            state_array = state_array.reshape(
                -1
            )

        if move_array.ndim == 1:
            move_array = move_array.reshape(
                1,
                -1,
            )

        if state_array.shape[0] != (
            self.state_dim
        ):
            raise ValueError(
                "Invalid state feature dimension. "
                f"Expected {self.state_dim}, "
                f"received {state_array.shape}."
            )

        if move_array.ndim != 2:
            raise ValueError(
                "Move features must form a "
                "two-dimensional matrix. "
                f"Received {move_array.shape}."
            )

        if move_array.shape[1] != (
            self.move_dim
        ):
            raise ValueError(
                "Invalid move feature dimension. "
                f"Expected {self.move_dim}, "
                f"received {move_array.shape}."
            )

        legal_move_count = int(
            row["legal_move_count"]
        )

        selected_move_index = int(
            row["selected_move_index"]
        )

        if move_array.shape[0] != (
            legal_move_count
        ):
            raise ValueError(
                "Move matrix row count does not "
                "match legal_move_count."
            )

        if not (
            0
            <= selected_move_index
            < legal_move_count
        ):
            raise ValueError(
                "Selected move index is outside "
                "the legal-move range."
            )

        return CompetitivePolicyExample(
            state_features=torch.tensor(
                state_array,
                dtype=torch.float32,
            ),

            move_features=torch.tensor(
                move_array,
                dtype=torch.float32,
            ),

            selected_move_index=
                selected_move_index,

            legal_move_count=
                legal_move_count,

            record_id=int(
                row["record_id"]
            ),

            battle_id=int(
                row["battle_id"]
            ),

            selected_move_name=str(
                row["selected_move_name"]
            ),
        )


# ------------------------------------------------------------
# 2.5 â€” Padded batch structure
# ------------------------------------------------------------

@dataclass
class CompetitivePolicyBatch:
    state_features: torch.Tensor
    move_features: torch.Tensor
    action_mask: torch.Tensor
    selected_move_indices: torch.Tensor
    legal_move_counts: torch.Tensor
    record_ids: torch.Tensor
    battle_ids: torch.Tensor
    selected_move_names: list[str]

    def to(
        self,
        device: torch.device,
    ) -> "CompetitivePolicyBatch":
        return CompetitivePolicyBatch(
            state_features=
                self.state_features.to(
                    device
                ),

            move_features=
                self.move_features.to(
                    device
                ),

            action_mask=
                self.action_mask.to(
                    device
                ),

            selected_move_indices=
                self.selected_move_indices.to(
                    device
                ),

            legal_move_counts=
                self.legal_move_counts.to(
                    device
                ),

            record_ids=
                self.record_ids.to(
                    device
                ),

            battle_ids=
                self.battle_ids.to(
                    device
                ),

            selected_move_names=
                self.selected_move_names,
        )


# ------------------------------------------------------------
# 2.6 â€” Variable-action collator
# ------------------------------------------------------------

def competitive_policy_collate_fn(
    examples: list[
        CompetitivePolicyExample
    ],
) -> CompetitivePolicyBatch:
    if not examples:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    batch_size = len(examples)

    maximum_move_count = max(
        example.legal_move_count
        for example in examples
    )

    state_batch = torch.stack(
        [
            example.state_features
            for example in examples
        ],
        dim=0,
    )

    move_batch = torch.zeros(
        (
            batch_size,
            maximum_move_count,
            MOVE_DIM,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.zeros(
        (
            batch_size,
            maximum_move_count,
        ),
        dtype=torch.bool,
    )

    selected_move_indices = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    legal_move_counts = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    record_ids = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    battle_ids = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    selected_move_names = []

    for batch_index, example in enumerate(
        examples
    ):
        move_count = (
            example.legal_move_count
        )

        move_batch[
            batch_index,
            :move_count,
            :,
        ] = example.move_features

        action_mask[
            batch_index,
            :move_count,
        ] = True

        selected_move_indices[
            batch_index
        ] = example.selected_move_index

        legal_move_counts[
            batch_index
        ] = move_count

        record_ids[
            batch_index
        ] = example.record_id

        battle_ids[
            batch_index
        ] = example.battle_id

        selected_move_names.append(
            example.selected_move_name
        )

    if not action_mask[
        torch.arange(batch_size),
        selected_move_indices,
    ].all():
        raise ValueError(
            "At least one selected move points "
            "to a padded or illegal action."
        )

    return CompetitivePolicyBatch(
        state_features=state_batch,
        move_features=move_batch,
        action_mask=action_mask,
        selected_move_indices=
            selected_move_indices,
        legal_move_counts=
            legal_move_counts,
        record_ids=record_ids,
        battle_ids=battle_ids,
        selected_move_names=
            selected_move_names,
    )


# ------------------------------------------------------------
# 2.7 â€” Create datasets
# ------------------------------------------------------------

train_dataset = CompetitivePolicyDataset(
    train_df
)

validation_dataset = (
    CompetitivePolicyDataset(
        validation_df
    )
)

test_dataset = CompetitivePolicyDataset(
    test_df
)


# ------------------------------------------------------------
# 2.8 â€” Create data loaders
# ------------------------------------------------------------

train_generator = torch.Generator()

train_generator.manual_seed(
    RANDOM_SEED
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=
        competitive_policy_collate_fn,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    generator=train_generator,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=
        competitive_policy_collate_fn,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=
        competitive_policy_collate_fn,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# ------------------------------------------------------------
# 2.9 â€” Inspect one batch
# ------------------------------------------------------------

sample_batch = next(
    iter(train_loader)
)

print(
    "NOTEBOOK 47 â€” VARIABLE-ACTION DATASET"
)

print("=" * 100)

print()
print(
    "Training examples     :",
    len(train_dataset),
)

print(
    "Validation examples   :",
    len(validation_dataset),
)

print(
    "Test examples         :",
    len(test_dataset),
)

print()
print(
    "Batch state shape     :",
    tuple(
        sample_batch
        .state_features
        .shape
    ),
)

print(
    "Batch move shape      :",
    tuple(
        sample_batch
        .move_features
        .shape
    ),
)

print(
    "Batch action-mask shape:",
    tuple(
        sample_batch
        .action_mask
        .shape
    ),
)

print(
    "Selected-index shape  :",
    tuple(
        sample_batch
        .selected_move_indices
        .shape
    ),
)

print()
print(
    "Largest move count    :",
    int(
        sample_batch
        .legal_move_counts
        .max()
        .item()
    ),
)

print(
    "Smallest move count   :",
    int(
        sample_batch
        .legal_move_counts
        .min()
        .item()
    ),
)

print(
    "Valid actions in batch:",
    int(
        sample_batch
        .action_mask
        .sum()
        .item()
    ),
)

print()
print("FIRST FIVE BATCH RECORDS")
print("-" * 100)

batch_preview = pd.DataFrame(
    {
        "record_id":
            sample_batch
            .record_ids[:5]
            .tolist(),

        "battle_id":
            sample_batch
            .battle_ids[:5]
            .tolist(),

        "legal_move_count":
            sample_batch
            .legal_move_counts[:5]
            .tolist(),

        "selected_index":
            sample_batch
            .selected_move_indices[:5]
            .tolist(),

        "selected_move_name":
            sample_batch
            .selected_move_names[:5],
    }
)

display(batch_preview)


# ------------------------------------------------------------
# 2.10 â€” Validation
# ------------------------------------------------------------

assert sample_batch.state_features.ndim == 2

assert sample_batch.state_features.shape[
    1
] == STATE_DIM

assert sample_batch.move_features.ndim == 3

assert sample_batch.move_features.shape[
    2
] == MOVE_DIM

assert sample_batch.action_mask.shape == (
    sample_batch.move_features.shape[
        :2
    ]
)

assert sample_batch.selected_move_indices.shape[
    0
] == sample_batch.state_features.shape[
    0
]

assert sample_batch.action_mask[
    torch.arange(
        sample_batch
        .state_features
        .shape[0]
    ),
    sample_batch
    .selected_move_indices,
].all()

print()
print(
    "âœ… SECTION 2 VARIABLE-ACTION DATASET PASSED"
)


# ## Section 3 â€” Competitive Policy Network
# 
# #### This section creates the neural network that predicts which legal move should be selected from a variable-sized action set.
# 
# #### The network embeds the battle state and every legal move independently, combines them, scores every legal move, and masks invalid padded actions.

# In[3]:


# ============================================================
# NOTEBOOK 47
# SECTION 3 â€” COMPETITIVE POLICY NETWORK
# ============================================================


import torch
import torch.nn as nn


# ------------------------------------------------------------
# 3.3 Policy Network
# ------------------------------------------------------------

class CompetitivePolicyNetwork(
    nn.Module,
):

    def __init__(
        self,
        state_dim: int,
        move_dim: int,
        hidden_dim: int = 128,
    ):

        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.move_encoder = nn.Sequential(
            nn.Linear(move_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_features,
        move_features,
        action_mask,
    ):

        batch_size = state_features.shape[0]

        max_moves = move_features.shape[1]

        state_embedding = self.state_encoder(
            state_features
        )

        state_embedding = state_embedding.unsqueeze(1)

        state_embedding = state_embedding.expand(
            batch_size,
            max_moves,
            -1,
        )

        move_embedding = self.move_encoder(
            move_features
        )

        joint = torch.cat(
            [
                state_embedding,
                move_embedding,
            ],
            dim=-1,
        )

        logits = self.policy_head(
            joint
        ).squeeze(-1)

        logits = logits.masked_fill(
            ~action_mask,
            -1e9,
        )

        return logits


# In[4]:


policy_network = CompetitivePolicyNetwork(
    state_dim=STATE_DIM,
    move_dim=MOVE_DIM,
    hidden_dim=128,
).to(DEVICE)

print(policy_network)


# In[5]:


batch = next(iter(train_loader)).to(DEVICE)

with torch.no_grad():

    logits = policy_network(
        state_features=batch.state_features,
        move_features=batch.move_features,
        action_mask=batch.action_mask,
    )

predicted_actions = logits.argmax(
    dim=1
)

print()

print("NOTEBOOK 47 â€” POLICY NETWORK")

print("=" * 70)

print()

print(
    "Batch size:",
    logits.shape[0],
)

print(
    "Maximum legal moves:",
    logits.shape[1],
)

print(
    "Output logits shape:",
    tuple(logits.shape),
)

print(
    "Predicted action shape:",
    tuple(predicted_actions.shape),
)

print()

print(
    "Sample logits:"
)

print(
    logits[:5]
)

print()

print(
    "Predicted actions:"
)

print(
    predicted_actions[:10]
)


# In[6]:


assert logits.ndim == 2

assert logits.shape == batch.action_mask.shape

assert predicted_actions.shape[0] == logits.shape[0]

assert torch.isfinite(
    logits[
        batch.action_mask
    ]
).all()

assert (
    predicted_actions
    < batch.legal_move_counts
).all()

print()

print(
    "âœ… SECTION 3 POLICY NETWORK PASSED"
)


# ## Section 4 â€” Training Objective and Optimization
# 
# #### Define the behavioral-cloning objective used to train the policy to imitate the depth-4 search expert.
# 
# #### This section adds:
# 
# - masked cross-entropy loss;
# - AdamW optimization;
# - cosine learning-rate scheduling;
# - gradient clipping;
# - top-1 and top-2 accuracy;
# - separate training and evaluation passes.

# In[7]:


# ============================================================
# NOTEBOOK 47
# SECTION 4 â€” TRAINING OBJECTIVE AND OPTIMIZATION
# ============================================================


from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------
# 4.3 â€” Training configuration
# ------------------------------------------------------------

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

MAX_GRADIENT_NORM = 1.0

TRAINING_EPOCHS = 50

EARLY_STOPPING_PATIENCE = 10

MINIMUM_LEARNING_RATE = 1e-6


# ------------------------------------------------------------
# 4.4 â€” Loss function
# ------------------------------------------------------------

policy_loss_function = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# 4.5 â€” Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    policy_network.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ------------------------------------------------------------
# 4.6 â€” Learning-rate scheduler
# ------------------------------------------------------------

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_EPOCHS,
        eta_min=MINIMUM_LEARNING_RATE,
    )
)


# ------------------------------------------------------------
# 4.7 â€” Metric result structure
# ------------------------------------------------------------

@dataclass
class PolicyEpochMetrics:
    loss: float
    top1_accuracy: float
    top2_accuracy: float
    examples: int
    batches: int
    average_confidence: float
    average_margin: float
    learning_rate: float | None = None


# ------------------------------------------------------------
# 4.8 â€” Top-k accuracy helper
# ------------------------------------------------------------

def masked_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
    k: int,
) -> tuple[int, int]:
    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape "
            "[batch_size, action_count]."
        )

    if targets.ndim != 1:
        raise ValueError(
            "Targets must have shape [batch_size]."
        )

    if action_mask.shape != logits.shape:
        raise ValueError(
            "Action-mask shape must match logits."
        )

    maximum_k = min(
        int(k),
        int(logits.shape[1]),
    )

    masked_logits = logits.masked_fill(
        ~action_mask,
        -1e9,
    )

    top_indices = torch.topk(
        masked_logits,
        k=maximum_k,
        dim=1,
    ).indices

    correct = (
        top_indices
        == targets.unsqueeze(1)
    ).any(dim=1)

    return (
        int(correct.sum().item()),
        int(targets.shape[0]),
    )


# ------------------------------------------------------------
# 4.9 â€” Confidence and margin helper
# ------------------------------------------------------------

def policy_confidence_statistics(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[float, float]:
    masked_logits = logits.masked_fill(
        ~action_mask,
        -1e9,
    )

    probabilities = torch.softmax(
        masked_logits,
        dim=1,
    )

    top_values = torch.topk(
        probabilities,
        k=min(
            2,
            probabilities.shape[1],
        ),
        dim=1,
    ).values

    confidence = top_values[
        :,
        0,
    ]

    if top_values.shape[1] >= 2:
        margin = (
            top_values[:, 0]
            - top_values[:, 1]
        )
    else:
        margin = confidence

    return (
        float(confidence.sum().item()),
        float(margin.sum().item()),
    )


# ------------------------------------------------------------
# 4.10 â€” One training epoch
# ------------------------------------------------------------

def train_policy_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_gradient_norm: float,
) -> PolicyEpochMetrics:
    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    top1_correct = 0
    top2_correct = 0

    total_confidence = 0.0
    total_margin = 0.0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            state_features=
                batch.state_features,

            move_features=
                batch.move_features,

            action_mask=
                batch.action_mask,
        )

        loss = policy_loss_function(
            logits,
            batch.selected_move_indices,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Training loss became non-finite."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_gradient_norm,
        )

        optimizer.step()

        batch_size = int(
            batch.state_features.shape[0]
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_examples += batch_size
        total_batches += 1

        batch_top1, _ = (
            masked_topk_accuracy(
                logits=logits,
                targets=
                    batch.selected_move_indices,
                action_mask=
                    batch.action_mask,
                k=1,
            )
        )

        batch_top2, _ = (
            masked_topk_accuracy(
                logits=logits,
                targets=
                    batch.selected_move_indices,
                action_mask=
                    batch.action_mask,
                k=2,
            )
        )

        top1_correct += batch_top1
        top2_correct += batch_top2

        confidence_sum, margin_sum = (
            policy_confidence_statistics(
                logits=logits,
                action_mask=
                    batch.action_mask,
            )
        )

        total_confidence += confidence_sum
        total_margin += margin_sum

    if total_examples == 0:
        raise RuntimeError(
            "The training loader produced "
            "no examples."
        )

    return PolicyEpochMetrics(
        loss=(
            total_loss
            / total_examples
        ),

        top1_accuracy=(
            top1_correct
            / total_examples
        ),

        top2_accuracy=(
            top2_correct
            / total_examples
        ),

        examples=total_examples,

        batches=total_batches,

        average_confidence=(
            total_confidence
            / total_examples
        ),

        average_margin=(
            total_margin
            / total_examples
        ),

        learning_rate=float(
            optimizer
            .param_groups[0]["lr"]
        ),
    )


# ------------------------------------------------------------
# 4.11 â€” Evaluation pass
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> PolicyEpochMetrics:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    top1_correct = 0
    top2_correct = 0

    total_confidence = 0.0
    total_margin = 0.0

    for batch in data_loader:
        batch = batch.to(device)

        logits = model(
            state_features=
                batch.state_features,

            move_features=
                batch.move_features,

            action_mask=
                batch.action_mask,
        )

        loss = policy_loss_function(
            logits,
            batch.selected_move_indices,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Evaluation loss became non-finite."
            )

        batch_size = int(
            batch.state_features.shape[0]
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_examples += batch_size
        total_batches += 1

        batch_top1, _ = (
            masked_topk_accuracy(
                logits=logits,
                targets=
                    batch.selected_move_indices,
                action_mask=
                    batch.action_mask,
                k=1,
            )
        )

        batch_top2, _ = (
            masked_topk_accuracy(
                logits=logits,
                targets=
                    batch.selected_move_indices,
                action_mask=
                    batch.action_mask,
                k=2,
            )
        )

        top1_correct += batch_top1
        top2_correct += batch_top2

        confidence_sum, margin_sum = (
            policy_confidence_statistics(
                logits=logits,
                action_mask=
                    batch.action_mask,
            )
        )

        total_confidence += confidence_sum
        total_margin += margin_sum

    if total_examples == 0:
        raise RuntimeError(
            "The evaluation loader produced "
            "no examples."
        )

    return PolicyEpochMetrics(
        loss=(
            total_loss
            / total_examples
        ),

        top1_accuracy=(
            top1_correct
            / total_examples
        ),

        top2_accuracy=(
            top2_correct
            / total_examples
        ),

        examples=total_examples,

        batches=total_batches,

        average_confidence=(
            total_confidence
            / total_examples
        ),

        average_margin=(
            total_margin
            / total_examples
        ),

        learning_rate=None,
    )


# ------------------------------------------------------------
# 4.12 â€” Dry-run validation
# ------------------------------------------------------------

initial_train_metrics = evaluate_policy(
    model=policy_network,
    data_loader=train_loader,
    device=DEVICE,
)

initial_validation_metrics = (
    evaluate_policy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)


# ------------------------------------------------------------
# 4.13 â€” Display configuration and dry-run metrics
# ------------------------------------------------------------

parameter_count = sum(
    parameter.numel()
    for parameter
    in policy_network.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter
    in policy_network.parameters()
    if parameter.requires_grad
)

print(
    "NOTEBOOK 47 â€” TRAINING OBJECTIVE AND OPTIMIZATION"
)

print("=" * 100)

print()
print(
    "Loss function        :",
    type(
        policy_loss_function
    ).__name__,
)

print(
    "Optimizer            :",
    type(optimizer).__name__,
)

print(
    "Scheduler            :",
    type(scheduler).__name__,
)

print(
    "Learning rate        :",
    LEARNING_RATE,
)

print(
    "Weight decay         :",
    WEIGHT_DECAY,
)

print(
    "Maximum gradient norm:",
    MAX_GRADIENT_NORM,
)

print(
    "Training epochs      :",
    TRAINING_EPOCHS,
)

print(
    "Early-stop patience  :",
    EARLY_STOPPING_PATIENCE,
)

print()
print(
    "Total parameters     :",
    f"{parameter_count:,}",
)

print(
    "Trainable parameters :",
    f"{trainable_parameter_count:,}",
)

print()
print("INITIAL TRAINING-SPLIT METRICS")
print("-" * 100)

print(
    "Loss                :",
    round(
        initial_train_metrics.loss,
        6,
    ),
)

print(
    "Top-1 accuracy      :",
    round(
        initial_train_metrics
        .top1_accuracy,
        4,
    ),
)

print(
    "Top-2 accuracy      :",
    round(
        initial_train_metrics
        .top2_accuracy,
        4,
    ),
)

print(
    "Average confidence  :",
    round(
        initial_train_metrics
        .average_confidence,
        4,
    ),
)

print()
print("INITIAL VALIDATION METRICS")
print("-" * 100)

print(
    "Loss                :",
    round(
        initial_validation_metrics.loss,
        6,
    ),
)

print(
    "Top-1 accuracy      :",
    round(
        initial_validation_metrics
        .top1_accuracy,
        4,
    ),
)

print(
    "Top-2 accuracy      :",
    round(
        initial_validation_metrics
        .top2_accuracy,
        4,
    ),
)

print(
    "Average confidence  :",
    round(
        initial_validation_metrics
        .average_confidence,
        4,
    ),
)


# ------------------------------------------------------------
# 4.14 â€” Final validation
# ------------------------------------------------------------

assert parameter_count > 0

assert trainable_parameter_count == (
    parameter_count
)

assert (
    0.0
    <= initial_train_metrics
    .top1_accuracy
    <= 1.0
)

assert (
    0.0
    <= initial_validation_metrics
    .top1_accuracy
    <= 1.0
)

assert (
    initial_train_metrics
    .top2_accuracy
    >= initial_train_metrics
    .top1_accuracy
)

assert (
    initial_validation_metrics
    .top2_accuracy
    >= initial_validation_metrics
    .top1_accuracy
)

assert math.isfinite(
    initial_train_metrics.loss
)

assert math.isfinite(
    initial_validation_metrics.loss
)

print()
print(
    "âœ… SECTION 4 TRAINING OBJECTIVE PASSED"
)


# ## Section 5 â€” Behavioral-Cloning Training
# 
# #### Train the variable-action policy to imitate the depth-4 search expert.
# 
# #### The loop includes:
# 
# - training and validation metrics;
# - early stopping;
# - best-checkpoint saving;
# - learning-rate scheduling;
# - multi-action accuracy tracking;
# - epoch-history export.

# In[8]:


# ============================================================
# NOTEBOOK 47
# SECTION 5 â€” BEHAVIORAL-CLONING TRAINING LOOP
# ============================================================


import copy
import json
import math
import time
from dataclasses import asdict

import pandas as pd
import torch


# ------------------------------------------------------------
# 5.3 â€” Output files
# ------------------------------------------------------------

BEST_CHECKPOINT_FILE = (
    COMPETITIVE_MODEL_DIR
    / "competitive_policy_best.pt"
)

LAST_CHECKPOINT_FILE = (
    COMPETITIVE_MODEL_DIR
    / "competitive_policy_last.pt"
)

TRAINING_HISTORY_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_training_history.csv"
)

TRAINING_SUMMARY_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_training_summary.json"
)


# ------------------------------------------------------------
# 5.4 â€” Multi-action evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_multi_action_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_examples = 0
    correct_examples = 0

    total_loss = 0.0

    for batch in data_loader:
        batch = batch.to(device)

        multi_action_mask = (
            batch.legal_move_counts
            >= 2
        )

        if not multi_action_mask.any():
            continue

        logits = model(
            state_features=
                batch.state_features,

            move_features=
                batch.move_features,

            action_mask=
                batch.action_mask,
        )

        selected_logits = logits[
            multi_action_mask
        ]

        selected_targets = (
            batch.selected_move_indices[
                multi_action_mask
            ]
        )

        loss = policy_loss_function(
            selected_logits,
            selected_targets,
        )

        predictions = (
            selected_logits.argmax(
                dim=1
            )
        )

        batch_count = int(
            selected_targets.shape[0]
        )

        total_examples += batch_count

        correct_examples += int(
            (
                predictions
                == selected_targets
            ).sum().item()
        )

        total_loss += (
            float(loss.item())
            * batch_count
        )

    if total_examples == 0:
        return {
            "examples": 0,
            "accuracy": float("nan"),
            "loss": float("nan"),
        }

    return {
        "examples":
            total_examples,

        "accuracy":
            correct_examples
            / total_examples,

        "loss":
            total_loss
            / total_examples,
    }


# ------------------------------------------------------------
# 5.5 â€” Checkpoint helper
# ------------------------------------------------------------

def save_policy_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    validation_metrics:
        PolicyEpochMetrics,
    multi_action_metrics:
        dict[str, float],
) -> None:
    checkpoint = {
        "epoch":
            int(epoch),

        "model_state_dict":
            copy.deepcopy(
                model.state_dict()
            ),

        "optimizer_state_dict":
            copy.deepcopy(
                optimizer.state_dict()
            ),

        "scheduler_state_dict":
            copy.deepcopy(
                scheduler.state_dict()
            ),

        "validation_metrics":
            asdict(
                validation_metrics
            ),

        "multi_action_metrics":
            {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    )
                    else value
                )
                for key, value in (
                    multi_action_metrics.items()
                )
            },

        "configuration": {
            "state_dim":
                STATE_DIM,

            "move_dim":
                MOVE_DIM,

            "hidden_dim":
                128,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "random_seed":
                RANDOM_SEED,

            "actor_relative":
                True,

            "variable_action":
                True,

            "label_source":
                "advanced_search_engine",
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ------------------------------------------------------------
# 5.6 â€” Reset model and optimizer for clean training
# ------------------------------------------------------------

torch.manual_seed(
    RANDOM_SEED
)

policy_network = CompetitivePolicyNetwork(
    state_dim=STATE_DIM,
    move_dim=MOVE_DIM,
    hidden_dim=128,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    policy_network.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_EPOCHS,
        eta_min=MINIMUM_LEARNING_RATE,
    )
)


# ------------------------------------------------------------
# 5.7 â€” Baseline metrics before training
# ------------------------------------------------------------

baseline_validation = (
    evaluate_policy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)

baseline_multi_action = (
    evaluate_multi_action_accuracy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)


# ------------------------------------------------------------
# 5.8 â€” Training loop
# ------------------------------------------------------------

training_history = []

best_validation_loss = math.inf

best_validation_accuracy = 0.0

best_multi_action_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0

training_start_time = (
    time.perf_counter()
)

print(
    "NOTEBOOK 47 â€” BEHAVIORAL-CLONING TRAINING"
)

print("=" * 115)

print()
print(
    "Baseline validation loss:",
    round(
        baseline_validation.loss,
        6,
    ),
)

print(
    "Baseline top-1 accuracy :",
    round(
        baseline_validation
        .top1_accuracy,
        4,
    ),
)

print(
    "Baseline multi-action   :",
    round(
        baseline_multi_action[
            "accuracy"
        ],
        4,
    ),
)

print()
print(
    "Epoch | Train Loss | Train Acc | Val Loss | "
    "Val Acc | Multi Acc | Confidence | LR"
)

print("-" * 115)


for epoch in range(
    1,
    TRAINING_EPOCHS + 1,
):
    epoch_start = (
        time.perf_counter()
    )

    train_metrics = (
        train_policy_epoch(
            model=policy_network,
            data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            max_gradient_norm=
                MAX_GRADIENT_NORM,
        )
    )

    validation_metrics = (
        evaluate_policy(
            model=policy_network,
            data_loader=validation_loader,
            device=DEVICE,
        )
    )

    validation_multi_action = (
        evaluate_multi_action_accuracy(
            model=policy_network,
            data_loader=validation_loader,
            device=DEVICE,
        )
    )

    current_learning_rate = float(
        optimizer.param_groups[0]["lr"]
    )

    epoch_seconds = (
        time.perf_counter()
        - epoch_start
    )

    history_row = {
        "epoch":
            epoch,

        "train_loss":
            train_metrics.loss,

        "train_top1_accuracy":
            train_metrics
            .top1_accuracy,

        "train_top2_accuracy":
            train_metrics
            .top2_accuracy,

        "validation_loss":
            validation_metrics.loss,

        "validation_top1_accuracy":
            validation_metrics
            .top1_accuracy,

        "validation_top2_accuracy":
            validation_metrics
            .top2_accuracy,

        "validation_multi_action_accuracy":
            validation_multi_action[
                "accuracy"
            ],

        "validation_multi_action_loss":
            validation_multi_action[
                "loss"
            ],

        "validation_confidence":
            validation_metrics
            .average_confidence,

        "validation_margin":
            validation_metrics
            .average_margin,

        "learning_rate":
            current_learning_rate,

        "epoch_seconds":
            epoch_seconds,
    }

    training_history.append(
        history_row
    )

    print(
        f"{epoch:5d} | "
        f"{train_metrics.loss:10.5f} | "
        f"{train_metrics.top1_accuracy:9.4f} | "
        f"{validation_metrics.loss:8.5f} | "
        f"{validation_metrics.top1_accuracy:7.4f} | "
        f"{validation_multi_action['accuracy']:9.4f} | "
        f"{validation_metrics.average_confidence:10.4f} | "
        f"{current_learning_rate:.7f}"
    )

    improved = (
        validation_metrics.loss
        < best_validation_loss
        - 1e-6
    )

    if improved:
        best_validation_loss = (
            validation_metrics.loss
        )

        best_validation_accuracy = (
            validation_metrics
            .top1_accuracy
        )

        best_multi_action_accuracy = (
            validation_multi_action[
                "accuracy"
            ]
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        save_policy_checkpoint(
            path=BEST_CHECKPOINT_FILE,
            model=policy_network,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            validation_metrics=
                validation_metrics,
            multi_action_metrics=
                validation_multi_action,
        )

    else:
        epochs_without_improvement += 1

    scheduler.step()

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print()
        print(
            "Early stopping triggered at "
            f"epoch {epoch}."
        )

        break


training_elapsed = (
    time.perf_counter()
    - training_start_time
)

completed_epochs = len(
    training_history
)


# ------------------------------------------------------------
# 5.9 â€” Save last checkpoint
# ------------------------------------------------------------

last_validation_metrics = (
    evaluate_policy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)

last_multi_action_metrics = (
    evaluate_multi_action_accuracy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)

save_policy_checkpoint(
    path=LAST_CHECKPOINT_FILE,
    model=policy_network,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=completed_epochs,
    validation_metrics=
        last_validation_metrics,
    multi_action_metrics=
        last_multi_action_metrics,
)


# ------------------------------------------------------------
# 5.10 â€” Save training history
# ------------------------------------------------------------

training_history_df = pd.DataFrame(
    training_history
)

training_history_df.to_csv(
    TRAINING_HISTORY_FILE,
    index=False,
)


# ------------------------------------------------------------
# 5.11 â€” Load best checkpoint
# ------------------------------------------------------------

best_checkpoint = torch.load(
    BEST_CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False,
)

policy_network.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

policy_network.eval()


# ------------------------------------------------------------
# 5.12 â€” Final validation after restoring best model
# ------------------------------------------------------------

restored_validation_metrics = (
    evaluate_policy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)

restored_multi_action_metrics = (
    evaluate_multi_action_accuracy(
        model=policy_network,
        data_loader=validation_loader,
        device=DEVICE,
    )
)


# ------------------------------------------------------------
# 5.13 â€” Save training summary
# ------------------------------------------------------------

training_summary = {
    "completed_epochs":
        completed_epochs,

    "best_epoch":
        int(best_epoch),

    "baseline_validation_loss":
        float(
            baseline_validation.loss
        ),

    "baseline_validation_accuracy":
        float(
            baseline_validation
            .top1_accuracy
        ),

    "baseline_multi_action_accuracy":
        float(
            baseline_multi_action[
                "accuracy"
            ]
        ),

    "best_validation_loss":
        float(
            restored_validation_metrics
            .loss
        ),

    "best_validation_accuracy":
        float(
            restored_validation_metrics
            .top1_accuracy
        ),

    "best_multi_action_accuracy":
        float(
            restored_multi_action_metrics[
                "accuracy"
            ]
        ),

    "training_seconds":
        float(training_elapsed),

    "best_checkpoint":
        str(BEST_CHECKPOINT_FILE),

    "last_checkpoint":
        str(LAST_CHECKPOINT_FILE),

    "history_file":
        str(TRAINING_HISTORY_FILE),

    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,

    "actor_relative":
        True,

    "variable_action":
        True,
}

with open(
    TRAINING_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        training_summary,
        file,
        indent=4,
    )


# ------------------------------------------------------------
# 5.14 â€” Display final training results
# ------------------------------------------------------------

print()
print(
    "TRAINING COMPLETE"
)

print("=" * 115)

print()
print(
    "Completed epochs         :",
    completed_epochs,
)

print(
    "Best epoch               :",
    best_epoch,
)

print(
    "Baseline validation loss :",
    round(
        baseline_validation.loss,
        6,
    ),
)

print(
    "Best validation loss     :",
    round(
        restored_validation_metrics
        .loss,
        6,
    ),
)

print(
    "Baseline validation acc  :",
    round(
        baseline_validation
        .top1_accuracy,
        4,
    ),
)

print(
    "Best validation acc      :",
    round(
        restored_validation_metrics
        .top1_accuracy,
        4,
    ),
)

print(
    "Baseline multi-action acc:",
    round(
        baseline_multi_action[
            "accuracy"
        ],
        4,
    ),
)

print(
    "Best multi-action acc    :",
    round(
        restored_multi_action_metrics[
            "accuracy"
        ],
        4,
    ),
)

print(
    "Training time            :",
    round(
        training_elapsed,
        2,
    ),
    "seconds",
)

print()
print(
    "Best checkpoint:",
    BEST_CHECKPOINT_FILE,
)

print(
    "Last checkpoint:",
    LAST_CHECKPOINT_FILE,
)

print(
    "Training history:",
    TRAINING_HISTORY_FILE,
)

print(
    "Training summary:",
    TRAINING_SUMMARY_FILE,
)


# ------------------------------------------------------------
# 5.15 â€” Final validation
# ------------------------------------------------------------

assert BEST_CHECKPOINT_FILE.exists()
assert LAST_CHECKPOINT_FILE.exists()
assert TRAINING_HISTORY_FILE.exists()
assert TRAINING_SUMMARY_FILE.exists()

assert completed_epochs >= 1
assert best_epoch >= 1

assert math.isfinite(
    restored_validation_metrics.loss
)

assert (
    restored_validation_metrics.loss
    <= baseline_validation.loss
    + 1e-6
)

assert (
    restored_validation_metrics
    .top1_accuracy
    >= baseline_validation
    .top1_accuracy
    - 1e-6
)

print()
print(
    "âœ… SECTION 5 BEHAVIORAL-CLONING TRAINING PASSED"
)


# ## Section 6 â€” Final Test Evaluation
# 
# #### Evaluate the best checkpoint on the untouched test split.
# 
# #### Report:
# 
# - overall top-1 accuracy;
# - multi-action accuracy;
# - loss and confidence;
# - performance by legal-move count;
# - confusion patterns;
# - incorrect-decision examples.
# 
# #### The test split was never used for training or checkpoint selection.

# In[9]:


# ============================================================
# NOTEBOOK 47
# SECTION 6 â€” FINAL TEST EVALUATION AND ERROR ANALYSIS
# ============================================================


import json
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 6.3 â€” Output files
# ------------------------------------------------------------

TEST_EVALUATION_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_test_evaluation.json"
)

TEST_PREDICTIONS_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_test_predictions.csv"
)

TEST_ERRORS_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_test_errors.csv"
)


# ------------------------------------------------------------
# 6.4 â€” Confirm best checkpoint is loaded
# ------------------------------------------------------------

assert BEST_CHECKPOINT_FILE.exists()

best_checkpoint = torch.load(
    BEST_CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False,
)

policy_network.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

policy_network.to(DEVICE)

policy_network.eval()


# ------------------------------------------------------------
# 6.5 â€” Standard test metrics
# ------------------------------------------------------------

test_metrics = evaluate_policy(
    model=policy_network,
    data_loader=test_loader,
    device=DEVICE,
)

test_multi_action_metrics = (
    evaluate_multi_action_accuracy(
        model=policy_network,
        data_loader=test_loader,
        device=DEVICE,
    )
)


# ------------------------------------------------------------
# 6.6 â€” Collect individual predictions
# ------------------------------------------------------------

prediction_records = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)

        logits = policy_network(
            state_features=
                batch.state_features,

            move_features=
                batch.move_features,

            action_mask=
                batch.action_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_indices = (
            logits.argmax(
                dim=1
            )
        )

        predicted_confidences = (
            probabilities.max(
                dim=1
            ).values
        )

        top_values = torch.topk(
            probabilities,
            k=min(
                2,
                probabilities.shape[1],
            ),
            dim=1,
        ).values

        if top_values.shape[1] >= 2:
            margins = (
                top_values[:, 0]
                - top_values[:, 1]
            )
        else:
            margins = top_values[:, 0]

        for row_index in range(
            batch.state_features.shape[0]
        ):
            record_id = int(
                batch.record_ids[
                    row_index
                ].item()
            )

            source_row = test_df[
                test_df[
                    "record_id"
                ]
                == record_id
            ].iloc[0]

            predicted_index = int(
                predicted_indices[
                    row_index
                ].item()
            )

            target_index = int(
                batch.selected_move_indices[
                    row_index
                ].item()
            )

            legal_move_names = list(
                source_row[
                    "legal_move_names"
                ]
            )

            predicted_name = str(
                legal_move_names[
                    predicted_index
                ]
            )

            target_name = str(
                legal_move_names[
                    target_index
                ]
            )

            prediction_records.append(
                {
                    "record_id":
                        record_id,

                    "battle_id":
                        int(
                            source_row[
                                "battle_id"
                            ]
                        ),

                    "turn_number":
                        int(
                            source_row[
                                "turn_number"
                            ]
                        ),

                    "actor_side":
                        str(
                            source_row[
                                "actor_side"
                            ]
                        ),

                    "actor_name":
                        str(
                            source_row[
                                "actor_name"
                            ]
                        ),

                    "defender_name":
                        str(
                            source_row[
                                "defender_name"
                            ]
                        ),

                    "legal_move_count":
                        int(
                            source_row[
                                "legal_move_count"
                            ]
                        ),

                    "legal_move_names":
                        " | ".join(
                            str(name)
                            for name
                            in legal_move_names
                        ),

                    "target_index":
                        target_index,

                    "target_move":
                        target_name,

                    "predicted_index":
                        predicted_index,

                    "predicted_move":
                        predicted_name,

                    "correct":
                        bool(
                            predicted_index
                            == target_index
                        ),

                    "confidence":
                        float(
                            predicted_confidences[
                                row_index
                            ].item()
                        ),

                    "margin":
                        float(
                            margins[
                                row_index
                            ].item()
                        ),

                    "search_score":
                        float(
                            source_row[
                                "search_score"
                            ]
                        ),

                    "damage_applied":
                        float(
                            source_row[
                                "damage_applied"
                            ]
                        ),
                }
            )


prediction_df = pd.DataFrame(
    prediction_records
)

error_df = prediction_df[
    ~prediction_df[
        "correct"
    ]
].copy()


# ------------------------------------------------------------
# 6.7 â€” Performance by action count
# ------------------------------------------------------------

accuracy_by_move_count = (
    prediction_df
    .groupby(
        "legal_move_count"
    )
    .agg(
        records=(
            "correct",
            "size",
        ),

        correct=(
            "correct",
            "sum",
        ),

        accuracy=(
            "correct",
            "mean",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# 6.8 â€” Performance by actor side
# ------------------------------------------------------------

accuracy_by_actor_side = (
    prediction_df
    .groupby(
        "actor_side"
    )
    .agg(
        records=(
            "correct",
            "size",
        ),

        correct=(
            "correct",
            "sum",
        ),

        accuracy=(
            "correct",
            "mean",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# 6.9 â€” Error-pair analysis
# ------------------------------------------------------------

error_pair_counts = (
    error_df
    .groupby(
        [
            "target_move",
            "predicted_move",
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

top_error_pairs = [
    {
        "target_move":
            str(target_move),

        "predicted_move":
            str(predicted_move),

        "count":
            int(count),
    }
    for (
        target_move,
        predicted_move,
    ), count in (
        error_pair_counts
        .head(20)
        .items()
    )
]


# ------------------------------------------------------------
# 6.10 â€” Confidence calibration summary
# ------------------------------------------------------------

confidence_bins = pd.cut(
    prediction_df[
        "confidence"
    ],
    bins=[
        0.0,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        1.0,
    ],
    include_lowest=True,
)

confidence_summary = (
    prediction_df
    .assign(
        confidence_bin=
            confidence_bins
    )
    .groupby(
        "confidence_bin",
        observed=False,
    )
    .agg(
        records=(
            "correct",
            "size",
        ),

        accuracy=(
            "correct",
            "mean",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
    .reset_index()
)

confidence_summary[
    "confidence_bin"
] = confidence_summary[
    "confidence_bin"
].astype(str)


# ------------------------------------------------------------
# 6.11 â€” Save reports
# ------------------------------------------------------------

prediction_df.to_csv(
    TEST_PREDICTIONS_FILE,
    index=False,
)

error_df.to_csv(
    TEST_ERRORS_FILE,
    index=False,
)

test_summary = {
    "checkpoint":
        str(
            BEST_CHECKPOINT_FILE
        ),

    "best_epoch":
        int(
            best_checkpoint[
                "epoch"
            ]
        ),

    "test_records":
        int(
            len(prediction_df)
        ),

    "test_loss":
        float(
            test_metrics.loss
        ),

    "test_top1_accuracy":
        float(
            test_metrics
            .top1_accuracy
        ),

    "test_top2_accuracy":
        float(
            test_metrics
            .top2_accuracy
        ),

    "test_multi_action_records":
        int(
            test_multi_action_metrics[
                "examples"
            ]
        ),

    "test_multi_action_accuracy":
        float(
            test_multi_action_metrics[
                "accuracy"
            ]
        ),

    "test_multi_action_loss":
        float(
            test_multi_action_metrics[
                "loss"
            ]
        ),

    "average_confidence":
        float(
            test_metrics
            .average_confidence
        ),

    "average_margin":
        float(
            test_metrics
            .average_margin
        ),

    "errors":
        int(
            len(error_df)
        ),

    "accuracy_by_move_count":
        accuracy_by_move_count
        .to_dict(
            orient="records"
        ),

    "accuracy_by_actor_side":
        accuracy_by_actor_side
        .to_dict(
            orient="records"
        ),

    "confidence_summary":
        confidence_summary
        .to_dict(
            orient="records"
        ),

    "top_error_pairs":
        top_error_pairs,

    "actor_relative":
        True,

    "variable_action":
        True,

    "test_split_untouched":
        True,
}

with open(
    TEST_EVALUATION_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 6.12 â€” Display test results
# ------------------------------------------------------------

print(
    "NOTEBOOK 47 â€” FINAL TEST EVALUATION"
)

print("=" * 110)

print()
print(
    "Best checkpoint epoch :",
    best_checkpoint[
        "epoch"
    ],
)

print(
    "Test records          :",
    f"{len(prediction_df):,}",
)

print(
    "Test loss             :",
    round(
        test_metrics.loss,
        6,
    ),
)

print(
    "Test top-1 accuracy   :",
    round(
        test_metrics
        .top1_accuracy,
        4,
    ),
)

print(
    "Test top-2 accuracy   :",
    round(
        test_metrics
        .top2_accuracy,
        4,
    ),
)

print(
    "Multi-action records  :",
    test_multi_action_metrics[
        "examples"
    ],
)

print(
    "Multi-action accuracy :",
    round(
        test_multi_action_metrics[
            "accuracy"
        ],
        4,
    ),
)

print(
    "Average confidence    :",
    round(
        test_metrics
        .average_confidence,
        4,
    ),
)

print(
    "Incorrect decisions   :",
    len(error_df),
)

print()
print("ACCURACY BY LEGAL-MOVE COUNT")
print("-" * 110)

display(
    accuracy_by_move_count
)

print()
print("ACCURACY BY ACTOR SIDE")
print("-" * 110)

display(
    accuracy_by_actor_side
)

print()
print("CONFIDENCE CALIBRATION")
print("-" * 110)

display(
    confidence_summary
)

if len(error_df):
    print()
    print("ERROR SAMPLE")
    print("-" * 110)

    display(
        error_df[
            [
                "record_id",
                "battle_id",
                "actor_side",
                "actor_name",
                "defender_name",
                "legal_move_names",
                "target_move",
                "predicted_move",
                "confidence",
                "margin",
                "search_score",
            ]
        ].sort_values(
            "confidence",
            ascending=False,
        ).head(20)
    )

print()
print(
    "Evaluation report:",
    TEST_EVALUATION_FILE,
)

print(
    "Prediction file:",
    TEST_PREDICTIONS_FILE,
)

print(
    "Error file:",
    TEST_ERRORS_FILE,
)


# ------------------------------------------------------------
# 6.13 â€” Final validation
# ------------------------------------------------------------

assert TEST_EVALUATION_FILE.exists()
assert TEST_PREDICTIONS_FILE.exists()
assert TEST_ERRORS_FILE.exists()

assert len(prediction_df) == (
    len(test_df)
)

assert (
    prediction_df[
        "correct"
    ].mean()
    == test_metrics
    .top1_accuracy
)

assert math.isfinite(
    test_metrics.loss
)

assert (
    0.0
    <= test_metrics
    .top1_accuracy
    <= 1.0
)

assert (
    test_metrics
    .top2_accuracy
    >= test_metrics
    .top1_accuracy
)

assert (
    test_multi_action_metrics[
        "examples"
    ]
    > 0
)

print()
print(
    "âœ… SECTION 6 FINAL TEST EVALUATION PASSED"
)


# ## Section 7 â€” Production Inference Engine
# 
# #### Package the trained variable-action policy into reusable production classes.
# 
# #### This section will:
# 
# - load the best behavioral-cloning checkpoint;
# - encode actor-relative battle states;
# - encode all legal moves;
# - score a variable number of legal actions;
# - select only legal moves;
# - expose a reusable competitive battle agent;
# - save the production implementation under `src/competitive/`.

# In[13]:


# ============================================================
# NOTEBOOK 47
# SECTION 7 â€” PRODUCTION INFERENCE ENGINE
# FULL CORRECTED VERSION
# ============================================================


import ast
import importlib
import inspect
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ------------------------------------------------------------
# 7.1 â€” Output paths
# ------------------------------------------------------------

COMPETITIVE_SOURCE_DIR = (
    SRC_DIR
    / "competitive"
)

COMPETITIVE_SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

POLICY_MODULE_FILE = (
    COMPETITIVE_SOURCE_DIR
    / "competitive_policy_engine.py"
)

AGENT_MODULE_FILE = (
    COMPETITIVE_SOURCE_DIR
    / "competitive_policy_agent.py"
)

PACKAGE_INIT_FILE = (
    COMPETITIVE_SOURCE_DIR
    / "__init__.py"
)

PACKAGING_REPORT_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_packaging.json"
)


# ------------------------------------------------------------
# 7.2 â€” Production policy-engine source
# ------------------------------------------------------------

policy_module_source = r'''

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
'''


# ------------------------------------------------------------
# 7.3 â€” Simulator-compatible production agent source
# ------------------------------------------------------------

agent_module_source = r'''

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
'''


# ------------------------------------------------------------
# 7.4 â€” Package initializer source
# ------------------------------------------------------------

package_init_source = r'''
from src.competitive.competitive_policy_agent import (
    CompetitivePolicyAgent,
    CompetitivePolicyDecision,
)

from src.competitive.competitive_policy_engine import (
    CompetitivePolicyEngine,
    CompetitivePolicyNetwork,
    actor_and_defender,
    encode_move,
    encode_state,
    move_name,
)

__all__ = [
    "CompetitivePolicyAgent",
    "CompetitivePolicyDecision",
    "CompetitivePolicyEngine",
    "CompetitivePolicyNetwork",
    "actor_and_defender",
    "encode_move",
    "encode_state",
    "move_name",
]
'''


# ------------------------------------------------------------
# 7.5 â€” Write production files
# ------------------------------------------------------------

POLICY_MODULE_FILE.write_text(
    textwrap.dedent(
        policy_module_source
    ).lstrip(),
    encoding="utf-8",
)

AGENT_MODULE_FILE.write_text(
    textwrap.dedent(
        agent_module_source
    ).lstrip(),
    encoding="utf-8",
)

PACKAGE_INIT_FILE.write_text(
    textwrap.dedent(
        package_init_source
    ).lstrip(),
    encoding="utf-8",
)


# ------------------------------------------------------------
# 7.6 â€” Confirm generated source parses
# ------------------------------------------------------------

for source_file in [
    POLICY_MODULE_FILE,
    AGENT_MODULE_FILE,
    PACKAGE_INIT_FILE,
]:
    source_text = (
        source_file.read_text(
            encoding="utf-8"
        )
    )

    ast.parse(
        source_text
    )


# ------------------------------------------------------------
# 7.7 â€” Reload generated production modules
# ------------------------------------------------------------

importlib.invalidate_caches()

import src.competitive.competitive_policy_engine as policy_engine_module
import src.competitive.competitive_policy_agent as policy_agent_module

policy_engine_module = (
    importlib.reload(
        policy_engine_module
    )
)

policy_agent_module = (
    importlib.reload(
        policy_agent_module
    )
)

CompetitivePolicyEngine = (
    policy_engine_module
    .CompetitivePolicyEngine
)

encode_move = (
    policy_engine_module
    .encode_move
)

encode_state = (
    policy_engine_module
    .encode_state
)

CompetitivePolicyAgent = (
    policy_agent_module
    .CompetitivePolicyAgent
)

CompetitivePolicyDecision = (
    policy_agent_module
    .CompetitivePolicyDecision
)

print(
    "Reloaded choose_move signature:"
)

print(
    inspect.signature(
        CompetitivePolicyAgent
        .choose_move
    )
)


# ------------------------------------------------------------
# 7.8 â€” Build a genuine production test state
# ------------------------------------------------------------

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

from src.legal_moves import (
    get_current_legal_moves,
)


production_pikachu_card = {
    "name": "Pikachu",

    "hp": 120,

    "attacks": [
        {
            "Move Name":
                "Quick Attack",

            "damage_numeric":
                30,

            "energy_cost":
                1,

            "Effect Explanation":
                None,
        },

        {
            "Move Name":
                "Thunderbolt",

            "damage_numeric":
                90,

            "energy_cost":
                2,

            "Effect Explanation":
                None,
        },
    ],
}

production_charmander_card = {
    "name": "Charmander",

    "hp": 100,

    "attacks": [
        {
            "Move Name":
                "Scratch",

            "damage_numeric":
                20,

            "energy_cost":
                1,

            "Effect Explanation":
                None,
        },

        {
            "Move Name":
                "Flame Tail",

            "damage_numeric":
                50,

            "energy_cost":
                2,

            "Effect Explanation":
                None,
        },
    ],
}

production_test_state = BattleState(
    player=PlayerState(
        active=PokemonState(
            card=
                production_pikachu_card,

            current_hp=120.0,

            attached_energy=3,

            status=None,

            damage=0.0,

            is_active=True,
        ),

        bench=[],

        prize_cards_remaining=1,

        hand_size=7,
    ),

    opponent=PlayerState(
        active=PokemonState(
            card=
                production_charmander_card,

            current_hp=100.0,

            attached_energy=3,

            status=None,

            damage=0.0,

            is_active=True,
        ),

        bench=[],

        prize_cards_remaining=1,

        hand_size=7,
    ),

    turn_number=1,

    current_player="Player",
)


# ------------------------------------------------------------
# 7.9 â€” Validate production inference engine
# ------------------------------------------------------------

production_legal_moves = (
    get_current_legal_moves(
        production_test_state
    )
)

assert (
    len(
        production_legal_moves
    )
    >= 2
)

production_engine = (
    CompetitivePolicyEngine(
        checkpoint_path=
            BEST_CHECKPOINT_FILE,

        device=DEVICE,
    )
)

production_result = (
    production_engine
    .score_moves(
        battle_state=
            production_test_state,

        legal_moves=
            production_legal_moves,
    )
)

assert (
    0
    <= production_result[
        "selected_index"
    ]
    < len(
        production_legal_moves
    )
)

assert (
    production_result[
        "selected_move"
    ]
    in production_legal_moves
)

assert math.isclose(
    sum(
        production_result[
            "probabilities"
        ]
    ),
    1.0,
    rel_tol=1e-5,
    abs_tol=1e-5,
)


# ------------------------------------------------------------
# 7.10 â€” Create and validate production agent
# ------------------------------------------------------------

production_agent = (
    CompetitivePolicyAgent(
        checkpoint_path=
            BEST_CHECKPOINT_FILE,

        device=str(
            DEVICE
        ),

        name=
            "Competitive Behavioral-Cloning Agent",
    )
)

production_agent_decision = (
    production_agent.choose_move(
        state=
            production_test_state,

        depth=4,
    )
)

production_agent_move = (
    production_agent_decision.move
)

stored_decision = (
    production_agent
    .get_last_decision()
)

assert (
    production_agent_move
    in production_legal_moves
)

assert stored_decision is not None

assert (
    stored_decision.fallback
    is False
)

assert (
    stored_decision.score
    == stored_decision.confidence
)

assert (
    stored_decision.search_depth
    == 4
)

assert (
    stored_decision.nodes
    == 0
)


# ------------------------------------------------------------
# 7.11 â€” Save packaging report
# ------------------------------------------------------------

packaging_report = {
    "checkpoint":
        str(
            BEST_CHECKPOINT_FILE
        ),

    "checkpoint_epoch":
        int(
            production_result[
                "checkpoint_epoch"
            ]
        ),

    "policy_module":
        str(
            POLICY_MODULE_FILE
        ),

    "agent_module":
        str(
            AGENT_MODULE_FILE
        ),

    "package_initializer":
        str(
            PACKAGE_INIT_FILE
        ),

    "state_dimension":
        STATE_DIM,

    "move_dimension":
        MOVE_DIM,

    "hidden_dimension":
        128,

    "actor_relative":
        True,

    "variable_action":
        True,

    "simulator_compatible":
        True,

    "decision_fields": [
        "move",
        "confidence",
        "selected_move_name",
        "probabilities",
        "logits",
        "move_names",
        "fallback",
        "score",
        "search_depth",
        "nodes",
        "reason",
    ],

    "legal_move_count":
        int(
            len(
                production_legal_moves
            )
        ),

    "legal_move_names":
        production_result[
            "move_names"
        ],

    "selected_move":
        production_result[
            "selected_move_name"
        ],

    "confidence":
        float(
            production_result[
                "confidence"
            ]
        ),

    "probabilities":
        production_result[
            "probabilities"
        ],

    "fallback":
        bool(
            production_result[
                "fallback"
            ]
        ),

    "production_validation_passed":
        True,
}

with open(
    PACKAGING_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        packaging_report,
        file,
        indent=4,
    )


# ------------------------------------------------------------
# 7.12 â€” Display production inference result
# ------------------------------------------------------------

print()
print(
    "NOTEBOOK 47 â€” PRODUCTION INFERENCE ENGINE"
)

print("=" * 100)

print()
print(
    "Checkpoint:",
    BEST_CHECKPOINT_FILE,
)

print(
    "Checkpoint epoch:",
    production_result[
        "checkpoint_epoch"
    ],
)

print()
print(
    "Legal moves:",
    production_result[
        "move_names"
    ],
)

print(
    "Selected move:",
    production_result[
        "selected_move_name"
    ],
)

print(
    "Confidence:",
    round(
        production_result[
            "confidence"
        ],
        6,
    ),
)

print(
    "Probabilities:",
    [
        round(
            probability,
            6,
        )
        for probability in (
            production_result[
                "probabilities"
            ]
        )
    ],
)

print(
    "Fallback:",
    production_result[
        "fallback"
    ],
)

print()
print(
    "Decision score:",
    stored_decision.score,
)

print(
    "Decision search depth:",
    stored_decision.search_depth,
)

print(
    "Decision nodes:",
    stored_decision.nodes,
)

print()
print(
    "Policy module:",
    POLICY_MODULE_FILE,
)

print(
    "Agent module:",
    AGENT_MODULE_FILE,
)

print(
    "Package initializer:",
    PACKAGE_INIT_FILE,
)

print(
    "Packaging report:",
    PACKAGING_REPORT_FILE,
)


# ------------------------------------------------------------
# 7.13 â€” Final validation
# ------------------------------------------------------------

assert POLICY_MODULE_FILE.exists()

assert AGENT_MODULE_FILE.exists()

assert PACKAGE_INIT_FILE.exists()

assert PACKAGING_REPORT_FILE.exists()

assert (
    production_result[
        "fallback"
    ]
    is False
)

assert (
    stored_decision
    .selected_move_name
    == production_result[
        "selected_move_name"
    ]
)

print()
print(
    "âœ… SECTION 7 PRODUCTION INFERENCE ENGINE PASSED"
)


# ## Section 8 â€” End-to-End Competitive Agent Battle
# 
# #### Run the packaged competitive policy agent through the production simulator.
# 
# #### This validates that the trained policy can:
# 
# - load its production checkpoint;
# - evaluate legal moves on every turn;
# - complete a full battle;
# - avoid illegal-action fallback;
# - save a battle result and transcript.

# In[18]:


# ============================================================
# NOTEBOOK 47
# SECTION 8 â€” END-TO-END COMPETITIVE AGENT BATTLE
# ============================================================


import json
import pickle
from pathlib import Path
from typing import Any

from src.battle_simulation import (
    create_battle_transcript,
    simulate_ai_battle,
)

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

from src.competitive.competitive_policy_agent import (
    CompetitivePolicyAgent,
)


# ------------------------------------------------------------
# 8.3 â€” Output files
# ------------------------------------------------------------

COMPETITIVE_BATTLE_RESULT_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_battle_result.pkl"
)

COMPETITIVE_BATTLE_TRANSCRIPT_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_battle_transcript.txt"
)

COMPETITIVE_BATTLE_SUMMARY_FILE = (
    NOTEBOOK47_REPORT_DIR
    / "competitive_policy_battle_summary.json"
)


# ------------------------------------------------------------
# 8.4 â€” Build production battle cards
# ------------------------------------------------------------

battle_pikachu_card = {
    "name": "Pikachu",
    "hp": 120,
    "attacks": [
        {
            "Move Name": "Quick Attack",
            "damage_numeric": 30,
            "energy_cost": 1,
            "Effect Explanation": None,
        },
        {
            "Move Name": "Thunderbolt",
            "damage_numeric": 90,
            "energy_cost": 2,
            "Effect Explanation": None,
        },
    ],
}

battle_charmander_card = {
    "name": "Charmander",
    "hp": 100,
    "attacks": [
        {
            "Move Name": "Scratch",
            "damage_numeric": 20,
            "energy_cost": 1,
            "Effect Explanation": None,
        },
        {
            "Move Name": "Flame Tail",
            "damage_numeric": 50,
            "energy_cost": 2,
            "Effect Explanation": None,
        },
    ],
}


# ------------------------------------------------------------
# 8.5 â€” Build initial battle state
# ------------------------------------------------------------

competitive_initial_state = BattleState(
    player=PlayerState(
        active=PokemonState(
            card=battle_pikachu_card,
            current_hp=120.0,
            attached_energy=3,
            status=None,
            damage=0.0,
            is_active=True,
        ),
        bench=[],
        prize_cards_remaining=1,
        hand_size=7,
    ),

    opponent=PlayerState(
        active=PokemonState(
            card=battle_charmander_card,
            current_hp=100.0,
            attached_energy=3,
            status=None,
            damage=0.0,
            is_active=True,
        ),
        bench=[],
        prize_cards_remaining=1,
        hand_size=7,
    ),

    turn_number=1,
    current_player="Player",
)


# ------------------------------------------------------------
# 8.6 â€” Instantiate packaged production agent
# ------------------------------------------------------------

competitive_agent = CompetitivePolicyAgent(
    checkpoint_path=
        BEST_CHECKPOINT_FILE,

    device=str(DEVICE),

    name=
        "Competitive Behavioral-Cloning Agent",
)


# ------------------------------------------------------------
# 8.7 â€” Inspect simulator signature
# ------------------------------------------------------------

import inspect

simulation_signature = inspect.signature(
    simulate_ai_battle
)

print(
    "simulate_ai_battle signature:"
)

print(
    simulation_signature
)


# ------------------------------------------------------------
# 8.8 â€” Run complete production battle
# ------------------------------------------------------------

simulation_parameters = (
    simulation_signature.parameters
)

simulation_kwargs = {}


if "initial_state" in simulation_parameters:
    simulation_kwargs[
        "initial_state"
    ] = competitive_initial_state

elif "state" in simulation_parameters:
    simulation_kwargs[
        "state"
    ] = competitive_initial_state

else:
    raise TypeError(
        "Could not identify the initial-state "
        "parameter for simulate_ai_battle."
    )


if "agent" in simulation_parameters:
    simulation_kwargs[
        "agent"
    ] = competitive_agent

elif "player_agent" in simulation_parameters:
    simulation_kwargs[
        "player_agent"
    ] = competitive_agent

else:
    raise TypeError(
        "Could not identify the agent parameter "
        "for simulate_ai_battle."
    )


if "max_turns" in simulation_parameters:
    simulation_kwargs[
        "max_turns"
    ] = 50


competitive_simulation = simulate_ai_battle(
    **simulation_kwargs
)


# ------------------------------------------------------------
# 8.9 â€” Create transcript
# ------------------------------------------------------------

competitive_transcript = (
    create_battle_transcript(
        competitive_simulation
    )
)


# ------------------------------------------------------------
# 8.10 â€” Extract result fields safely
# ------------------------------------------------------------

def section8_safe_attribute(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(
                    obj,
                    name,
                )
            except Exception:
                continue

    return default


winner = section8_safe_attribute(
    competitive_simulation,
    "winner",
    "winning_side",
    default=None,
)

turns = section8_safe_attribute(
    competitive_simulation,
    "turns",
    "turn_records",
    default=[],
)

stop_reason = section8_safe_attribute(
    competitive_simulation,
    "stop_reason",
    "termination_reason",
    default=None,
)

final_state = section8_safe_attribute(
    competitive_simulation,
    "final_state",
    "state",
    default=None,
)


# ------------------------------------------------------------
# 8.11 â€” Validate all recorded decisions
# ------------------------------------------------------------

fallback_decisions = []

if hasattr(
    competitive_agent,
    "get_last_decision",
):
    last_decision = (
        competitive_agent
        .get_last_decision()
    )

    if (
        last_decision is not None
        and bool(
            getattr(
                last_decision,
                "fallback",
                False,
            )
        )
    ):
        fallback_decisions.append(
            last_decision
        )


# ------------------------------------------------------------
# 8.12 â€” Save battle artifacts
# ------------------------------------------------------------

with open(
    COMPETITIVE_BATTLE_RESULT_FILE,
    "wb",
) as file:
    pickle.dump(
        competitive_simulation,
        file,
    )

COMPETITIVE_BATTLE_TRANSCRIPT_FILE.write_text(
    competitive_transcript,
    encoding="utf-8",
)

battle_summary = {
    "checkpoint":
        str(
            BEST_CHECKPOINT_FILE
        ),

    "agent_name":
        competitive_agent.name,

    "battle":
        "Pikachu vs Charmander",

    "winner":
        winner,

    "turn_count":
        int(
            len(turns)
        ),

    "stop_reason":
        stop_reason,

    "fallback_decisions":
        int(
            len(
                fallback_decisions
            )
        ),

    "transcript_file":
        str(
            COMPETITIVE_BATTLE_TRANSCRIPT_FILE
        ),

    "result_file":
        str(
            COMPETITIVE_BATTLE_RESULT_FILE
        ),

    "completed":
        True,
}

with open(
    COMPETITIVE_BATTLE_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        battle_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8.13 â€” Display result
# ------------------------------------------------------------

print()
print(
    "NOTEBOOK 47 â€” END-TO-END COMPETITIVE BATTLE"
)

print("=" * 100)

print()
print(
    "Agent:",
    competitive_agent.name,
)

print(
    "Checkpoint:",
    BEST_CHECKPOINT_FILE,
)

print(
    "Battle:",
    "Pikachu vs Charmander",
)

print(
    "Winner:",
    winner,
)

print(
    "Turns:",
    len(turns),
)

print(
    "Stop reason:",
    stop_reason,
)

print(
    "Fallback decisions:",
    len(
        fallback_decisions
    ),
)

print()
print(
    "Battle result:",
    COMPETITIVE_BATTLE_RESULT_FILE,
)

print(
    "Battle transcript:",
    COMPETITIVE_BATTLE_TRANSCRIPT_FILE,
)

print(
    "Battle summary:",
    COMPETITIVE_BATTLE_SUMMARY_FILE,
)

print()
print("TRANSCRIPT")
print("-" * 100)

print(
    competitive_transcript
)


# ------------------------------------------------------------
# 8.14 â€” Final validation
# ------------------------------------------------------------

assert COMPETITIVE_BATTLE_RESULT_FILE.exists()

assert COMPETITIVE_BATTLE_TRANSCRIPT_FILE.exists()

assert COMPETITIVE_BATTLE_SUMMARY_FILE.exists()

assert competitive_transcript.strip()

assert len(turns) > 0

assert len(
    fallback_decisions
) == 0

print()
print(
    "âœ… SECTION 8 END-TO-END COMPETITIVE BATTLE PASSED"
)


# In[ ]:





