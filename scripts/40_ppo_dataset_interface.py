#!/usr/bin/env python
# coding: utf-8

# # Notebook 40 - PPO Dataset Interface
# 
# ## Objectives
# 
# - Load the validated transition dataset from Notebook 39
# - Convert observation vectors and action masks into model-ready arrays
# - Build a reusable PPO dataset object
# - Create deterministic batching and shuffling utilities
# - Validate tensor shapes and data types
# - Prepare the exact input interface required by the PPO trainer
# - Export interface manifests and validation reports
# 

# # Section 1 — Imports & Project Setup

# In[2]:


# ============================================================
# NOTEBOOK 40 — PPO DATASET INTERFACE
# SECTION 1 — IMPORTS & PROJECT SETUP
# ============================================================

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd


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
        if (
            (candidate / "src").is_dir()
            and (candidate / "notebooks").is_dir()
            and (candidate / "reports").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOK39_REPORT_DIR = (
    REPORTS_DIR / "notebook39"
)

NOTEBOOK40_REPORT_DIR = (
    REPORTS_DIR / "notebook40"
)

CSV_FILE = (
    NOTEBOOK39_REPORT_DIR
    / "training_dataset.csv"
)

PARQUET_FILE = (
    NOTEBOOK39_REPORT_DIR
    / "training_dataset.parquet"
)

NOTEBOOK40_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RANDOM_SEED = 40

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option(
    "display.max_columns",
    160,
)

pd.set_option(
    "display.width",
    240,
)

pd.set_option(
    "display.max_colwidth",
    200,
)

print("NOTEBOOK 40 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
)

print(
    "Notebook 39 reports    :",
    NOTEBOOK39_REPORT_DIR,
)

print(
    "Notebook 40 reports    :",
    NOTEBOOK40_REPORT_DIR,
)

print(
    "CSV file               :",
    CSV_FILE,
)

print(
    "CSV exists             :",
    CSV_FILE.exists(),
)

print(
    "Parquet file           :",
    PARQUET_FILE,
)

print(
    "Parquet exists         :",
    PARQUET_FILE.exists(),
)

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert SCRIPTS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK39_REPORT_DIR.exists()
assert CSV_FILE.exists()
assert PARQUET_FILE.exists()

print()
print(
    "✅ SECTION 1 PROJECT SETUP PASSED"
)


# ## Section 2 — Load Production Dataset
# 
# #### Load the validated production dataset from Notebook 39, normalize nested columns, and confirm that CSV and Parquet contain the expected 16-field schema.

# In[3]:


# ============================================================
# SECTION 2 — LOAD PRODUCTION DATASET
# ============================================================

nested_columns = [
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
]


def normalize_nested_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        tuple,
    ):
        return list(
            value
        )

    if isinstance(
        value,
        str,
    ):
        try:
            parsed = json.loads(
                value
            )

            if isinstance(
                parsed,
                list,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

    return value


csv_dataset_df = pd.read_csv(
    CSV_FILE
)

parquet_dataset_df = pd.read_parquet(
    PARQUET_FILE
)

for column in nested_columns:
    assert column in csv_dataset_df.columns
    assert column in parquet_dataset_df.columns

    csv_dataset_df[column] = (
        csv_dataset_df[column]
        .apply(
            normalize_nested_value
        )
    )

    parquet_dataset_df[column] = (
        parquet_dataset_df[column]
        .apply(
            normalize_nested_value
        )
    )

dataset_df = (
    parquet_dataset_df.copy()
)

expected_columns = [
    "sample_id",
    "game_index",
    "turn",
    "actor",
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
    "chosen_move",
    "chosen_move_index",
    "search_score",
    "search_depth",
    "nodes_searched",
    "reward",
    "done",
    "winner",
    "source_dataset",
]

missing_columns = [
    column
    for column in expected_columns
    if column not in dataset_df.columns
]

extra_columns = [
    column
    for column in dataset_df.columns
    if column not in expected_columns
]

print("NOTEBOOK 40 — DATASET LOAD")
print("=" * 70)

print(
    "CSV rows               :",
    len(csv_dataset_df),
)

print(
    "Parquet rows           :",
    len(parquet_dataset_df),
)

print(
    "Dataset columns        :",
    len(dataset_df.columns),
)

print(
    "Missing columns        :",
    missing_columns,
)

print(
    "Extra columns          :",
    extra_columns,
)

display(
    dataset_df.head()
)

assert len(
    csv_dataset_df
) == len(
    parquet_dataset_df
)

assert len(
    dataset_df.columns
) == 16

assert not missing_columns
assert not extra_columns

assert (
    dataset_df.columns.tolist()
    == expected_columns
)

print()
print(
    "✅ SECTION 2 DATASET LOAD PASSED"
)


# ## Section 3 — PPO Dataset Class
# 
# #### Create a dataset wrapper around the validated production dataset. This object will provide indexed access to transitions and serve as the foundation for PPO training.

# In[4]:


# ============================================================
# SECTION 3 — PPO DATASET CLASS
# ============================================================

from dataclasses import dataclass


@dataclass
class PPOTransition:

    sample_id: str

    game_index: int

    turn: int

    actor: str

    observation_vector: list

    legal_moves: list

    legal_move_mask: list

    chosen_move: str

    chosen_move_index: int

    search_score: float

    search_depth: int

    nodes_searched: int

    reward: float

    done: bool

    winner: str

    source_dataset: str


class PPODataset:

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

    def __len__(
        self,
    ):

        return len(
            self.dataframe
        )

    def transition(
        self,
        index: int,
    ) -> PPOTransition:

        row = self.dataframe.iloc[
            index
        ]

        return PPOTransition(
            sample_id=row.sample_id,
            game_index=int(row.game_index),
            turn=int(row.turn),
            actor=row.actor,
            observation_vector=list(
                row.observation_vector
            ),
            legal_moves=list(
                row.legal_moves
            ),
            legal_move_mask=list(
                row.legal_move_mask
            ),
            chosen_move=row.chosen_move,
            chosen_move_index=int(
                row.chosen_move_index
            ),
            search_score=float(
                row.search_score
            ),
            search_depth=int(
                row.search_depth
            ),
            nodes_searched=int(
                row.nodes_searched
            ),
            reward=float(
                row.reward
            ),
            done=bool(
                row.done
            ),
            winner=row.winner,
            source_dataset=row.source_dataset,
        )


ppo_dataset = PPODataset(
    dataset_df
)

print(
    "NOTEBOOK 40 — PPO DATASET"
)
print("=" * 70)

print(
    "Dataset size :",
    len(ppo_dataset),
)

display(
    dataset_df.head(1)
)

assert len(
    ppo_dataset
) == 3

print()

print(
    "✅ SECTION 3 PPO DATASET PASSED"
)


# ## Section 4 — Dataset Indexing
# 
# #### Implement indexed access so each PPO transition can be retrieved using standard dataset syntax.

# In[5]:


# ============================================================
# SECTION 4 — DATASET INDEXING
# ============================================================

class PPODataset(PPODataset):

    def __getitem__(
        self,
        index: int,
    ) -> PPOTransition:

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                "Dataset index out of range."
            )

        return self.transition(index)


ppo_dataset = PPODataset(
    dataset_df
)

first_transition = ppo_dataset[0]
last_transition = ppo_dataset[-1]

print("NOTEBOOK 40 — DATASET INDEXING")
print("=" * 70)

print("Dataset length :", len(ppo_dataset))
print()

print("First sample")
display(
    pd.DataFrame(
        [first_transition.__dict__]
    )
)

print("Last sample")
display(
    pd.DataFrame(
        [last_transition.__dict__]
    )
)

assert first_transition.sample_id.startswith(
    "batch_game_"
)

assert last_transition.done is True

assert first_transition.chosen_move == "Attack A"

assert last_transition.chosen_move == "Pass"

print()

print("✅ SECTION 4 DATASET INDEXING PASSED")


# ## Section 5 — Observation Tensor Preparation
# 
# #### Validate and convert observation vectors into consistent NumPy arrays suitable for PPO training.

# In[6]:


# ============================================================
# SECTION 5 — OBSERVATION TENSOR PREPARATION
# ============================================================

observation_arrays = []

for transition in (
    ppo_dataset[i]
    for i in range(len(ppo_dataset))
):

    observation = np.asarray(
        transition.observation_vector,
        dtype=np.float32,
    )

    observation_arrays.append(
        observation
    )

observation_matrix = np.stack(
    observation_arrays
)

print("NOTEBOOK 40 — OBSERVATION TENSORS")
print("=" * 70)

print(
    "Transitions          :",
    len(observation_arrays),
)

print(
    "Observation shape    :",
    observation_matrix.shape,
)

print(
    "Observation dtype    :",
    observation_matrix.dtype,
)

display(
    pd.DataFrame(
        observation_matrix
    )
)

assert observation_matrix.shape == (
    len(ppo_dataset),
    4,
)

assert observation_matrix.dtype == np.float32

assert np.isfinite(
    observation_matrix
).all()

print()
print("✅ SECTION 5 OBSERVATION TENSORS PASSED")


# ## Section 6 — Action Encoding
# 
# #### Create a deterministic action vocabulary and encode each chosen move as an integer action ID suitable for PPO training.

# In[7]:


# ============================================================
# SECTION 6 — ACTION ENCODING
# ============================================================

# Build deterministic action vocabulary
action_vocabulary = sorted(
    {
        move
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
        for move in transition.legal_moves
    }
)

action_to_id = {
    action: index
    for index, action in enumerate(
        action_vocabulary
    )
}

id_to_action = {
    index: action
    for action, index in action_to_id.items()
}

encoded_actions = np.asarray(
    [
        action_to_id[
            transition.chosen_move
        ]
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
    ],
    dtype=np.int64,
)

encoding_df = pd.DataFrame(
    {
        "chosen_move": [
            ppo_dataset[i].chosen_move
            for i in range(len(ppo_dataset))
        ],
        "encoded_action": encoded_actions,
    }
)

print("NOTEBOOK 40 — ACTION ENCODING")
print("=" * 70)

print(
    "Vocabulary size :",
    len(action_vocabulary),
)

print()

print("Vocabulary")

display(
    pd.DataFrame(
        {
            "action": action_vocabulary,
            "id": list(
                range(
                    len(action_vocabulary)
                )
            ),
        }
    )
)

print()

print("Encoded actions")

display(
    encoding_df
)

assert len(action_vocabulary) == 4

assert encoded_actions.dtype == np.int64

assert encoded_actions.min() >= 0

assert encoded_actions.max() < len(action_vocabulary)

assert (
    id_to_action[
        action_to_id["Attack A"]
    ]
    == "Attack A"
)

print()

print("✅ SECTION 6 ACTION ENCODING PASSED")


# ## Section 7 — Legal Action Masks
# 
# #### Prepare fixed-size legal action masks for PPO policy training.

# In[8]:


# ============================================================
# SECTION 7 — LEGAL ACTION MASKS
# ============================================================

mask_tensor = np.asarray(
    [
        transition.legal_move_mask
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
    ],
    dtype=np.float32,
)

mask_df = pd.DataFrame(mask_tensor)

print("NOTEBOOK 40 — LEGAL ACTION MASKS")
print("=" * 70)

print(
    "Transitions :",
    len(mask_tensor),
)

print(
    "Mask shape  :",
    mask_tensor.shape,
)

print(
    "Mask dtype  :",
    mask_tensor.dtype,
)

print()

display(mask_df)

assert mask_tensor.shape == (3, 4)

assert mask_tensor.dtype == np.float32

assert np.all(
    (mask_tensor == 0)
    | (mask_tensor == 1)
)

assert (
    mask_tensor.sum(axis=1) >= 1
).all()

print()

print("✅ SECTION 7 LEGAL ACTION MASKS PASSED")


# ## Section 8 — Rewards Tensor
# 
# #### Convert rewards into a float32 tensor for PPO value-function learning.

# In[9]:


# ============================================================
# SECTION 8 — REWARD TENSORS
# ============================================================

reward_tensor = np.asarray(
    [
        transition.reward
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
    ],
    dtype=np.float32,
)

reward_df = pd.DataFrame(
    reward_tensor,
    columns=["reward"],
)

print("NOTEBOOK 40 — REWARD TENSORS")
print("=" * 70)

print(
    "Transitions :",
    len(reward_tensor),
)

print(
    "Tensor shape:",
    reward_tensor.shape,
)

print(
    "Tensor dtype:",
    reward_tensor.dtype,
)

print()

display(reward_df)

assert reward_tensor.shape == (3,)

assert reward_tensor.dtype == np.float32

assert np.isfinite(
    reward_tensor
).all()

print()

print("✅ SECTION 8 REWARD TENSORS PASSED")


# ## Section 9 — Terminal Flags
# 
# #### Prepare terminal state flags for PPO return and advantage computation.

# In[11]:


# ============================================================
# SECTION 9 — TERMINAL FLAGS
# ============================================================

done_tensor = np.asarray(
    [
        transition.done
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
    ],
    dtype=np.bool_,
)

done_df = pd.DataFrame(
    done_tensor,
    columns=["done"],
)

print("NOTEBOOK 40 — TERMINAL FLAGS")
print("=" * 70)

print(
    "Transitions :",
    len(done_tensor),
)

print(
    "Tensor shape:",
    done_tensor.shape,
)

print(
    "Terminal states:",
    done_tensor.sum(),
)

print()

display(done_df)

assert done_tensor.shape == (3,)

assert done_tensor.dtype == np.bool_

assert bool(done_tensor[-1]) is True

assert done_tensor[:-1].sum() == 0

print()
print("✅ SECTION 9 TERMINAL FLAGS PASSED")


# ## Section 10 — Action Target Tensor
# 
# #### Prepare the chosen action IDs as an integer target tensor for PPO policy optimization.

# In[12]:


# ============================================================
# SECTION 10 — ACTION TARGET TENSOR
# ============================================================

action_target_tensor = np.asarray(
    [
        action_to_id[
            transition.chosen_move
        ]
        for transition in (
            ppo_dataset[i]
            for i in range(len(ppo_dataset))
        )
    ],
    dtype=np.int64,
)

action_target_df = pd.DataFrame(
    {
        "chosen_move": [
            ppo_dataset[i].chosen_move
            for i in range(len(ppo_dataset))
        ],
        "action_target": (
            action_target_tensor
        ),
    }
)

print("NOTEBOOK 40 — ACTION TARGETS")
print("=" * 70)

print(
    "Transitions :",
    len(action_target_tensor),
)

print(
    "Tensor shape:",
    action_target_tensor.shape,
)

print(
    "Tensor dtype:",
    action_target_tensor.dtype,
)

print()

display(
    action_target_df
)

assert (
    action_target_tensor.shape
    == (len(ppo_dataset),)
)

assert (
    action_target_tensor.dtype
    == np.int64
)

assert (
    action_target_tensor.min()
    >= 0
)

assert (
    action_target_tensor.max()
    < len(action_vocabulary)
)

for row_index in range(
    len(ppo_dataset)
):
    target_id = int(
        action_target_tensor[
            row_index
        ]
    )

    assert (
        id_to_action[
            target_id
        ]
        == ppo_dataset[
            row_index
        ].chosen_move
    )

print()
print(
    "✅ SECTION 10 ACTION TARGETS PASSED"
)


# ## Section 11 — PPO Batch Dictionary
# 
# #### Combine all PPO tensors into one batch dictionary for policy training.

# In[15]:


# ============================================================
# SECTION 11 — PPO BATCH DICTIONARY
# ============================================================

ppo_batch = {
    "observations": observation_matrix,
    "action_masks": mask_tensor,
    "actions": action_target_tensor,
    "rewards": reward_tensor,
    "dones": done_tensor,
}

batch_summary = pd.DataFrame(
    [
        {
            "tensor": key,
            "shape": value.shape,
            "dtype": str(value.dtype),
        }
        for key, value in ppo_batch.items()
    ]
)

print("NOTEBOOK 40 — PPO BATCH")
print("=" * 70)

display(batch_summary)

assert len(ppo_batch) == 5

assert ppo_batch["observations"].shape == (
    len(ppo_dataset),
    4,
)

assert ppo_batch["action_masks"].shape == (
    len(ppo_dataset),
    4,
)

assert ppo_batch["actions"].shape == (
    len(ppo_dataset),
)

assert ppo_batch["rewards"].shape == (
    len(ppo_dataset),
)

assert ppo_batch["dones"].shape == (
    len(ppo_dataset),
)

assert (
    ppo_batch["observations"].dtype
    == np.float32
)

assert (
    ppo_batch["action_masks"].dtype
    == np.float32
)

assert (
    ppo_batch["actions"].dtype
    == np.int64
)

assert (
    ppo_batch["rewards"].dtype
    == np.float32
)

assert (
    ppo_batch["dones"].dtype
    == np.bool_
)

print()
print(
    "✅ SECTION 11 PPO BATCH PASSED"
)


# ## Section 12 — Mini-Batch Iterator
# 
# #### Create deterministic PPO mini-batches from the complete training batch.

# In[18]:


# ============================================================
# SECTION 12 — MINI-BATCH ITERATOR
# ============================================================

class PPOBatchIterator:

    def __init__(
        self,
        batch: dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 40,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self.batch = batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        lengths = {
            len(value)
            for value in batch.values()
        }

        if len(lengths) != 1:
            raise ValueError(
                "All batch tensors must have "
                "the same first dimension."
            )

        self.dataset_size = lengths.pop()

    def __len__(
        self,
    ) -> int:
        return math.ceil(
            self.dataset_size
            / self.batch_size
        )

    def __iter__(
        self,
    ) -> Iterator[
        dict[str, np.ndarray]
    ]:
        indices = np.arange(
            self.dataset_size
        )

        if self.shuffle:
            rng = np.random.default_rng(
                self.seed
            )

            rng.shuffle(
                indices
            )

        for start in range(
            0,
            self.dataset_size,
            self.batch_size,
        ):
            batch_indices = indices[
                start:
                start + self.batch_size
            ]

            yield {
                key: value[
                    batch_indices
                ]
                for key, value
                in self.batch.items()
            }


batch_iterator = PPOBatchIterator(
    batch=ppo_batch,
    batch_size=2,
    shuffle=False,
)

mini_batches = list(
    batch_iterator
)

mini_batch_summary = pd.DataFrame(
    [
        {
            "batch_index": index,
            "batch_size": len(
                batch["actions"]
            ),
            "observation_shape": (
                batch[
                    "observations"
                ].shape
            ),
            "mask_shape": (
                batch[
                    "action_masks"
                ].shape
            ),
        }
        for index, batch
        in enumerate(
            mini_batches
        )
    ]
)

print("NOTEBOOK 40 — MINI-BATCH ITERATOR")
print("=" * 70)

print(
    "Dataset size          :",
    batch_iterator.dataset_size,
)

print(
    "Configured batch size :",
    batch_iterator.batch_size,
)

print(
    "Mini-batches          :",
    len(batch_iterator),
)

display(
    mini_batch_summary
)

assert len(
    mini_batches
) == 2

assert mini_batches[
    0
]["observations"].shape == (
    2,
    4,
)

assert mini_batches[
    1
]["observations"].shape == (
    1,
    4,
)

assert mini_batches[
    0
]["actions"].shape == (
    2,
)

assert mini_batches[
    1
]["actions"].shape == (
    1,
)

print()
print(
    "✅ SECTION 12 MINI-BATCH ITERATOR PASSED"
)


# ## Section 13 — PPO Dataset Statistics
# 
# #### Generate a complete summary of the PPO tensors before training.

# In[20]:


# ============================================================
# SECTION 13 — PPO DATASET STATISTICS
# ============================================================

statistics = pd.DataFrame(
    [
        (
            "Transitions",
            len(ppo_dataset),
        ),
        (
            "Observation dimension",
            observation_matrix.shape[1],
        ),
        (
            "Action vocabulary",
            len(action_vocabulary),
        ),
        (
            "Legal actions",
            int(
                mask_tensor.sum()
            ),
        ),
        (
            "Average reward",
            float(
                reward_tensor.mean()
            ),
        ),
        (
            "Terminal transitions",
            int(
                done_tensor.sum()
            ),
        ),
        (
            "Mini-batches",
            len(mini_batches),
        ),
    ],
    columns=[
        "Metric",
        "Value",
    ],
)

print("NOTEBOOK 40 — PPO DATASET STATISTICS")
print("=" * 70)

display(
    statistics
)

assert statistics.shape == (
    7,
    2,
)

assert statistics.iloc[
    0
]["Value"] == 3

assert statistics.iloc[
    1
]["Value"] == 4

assert statistics.iloc[
    2
]["Value"] == 4

assert statistics.iloc[
    5
]["Value"] == 1

print()
print(
    "✅ SECTION 13 PPO DATASET STATISTICS PASSED"
)


# ## Section 14 — Production Readiness Validation
# 
# #### Verify that every tensor required for PPO policy optimization is present and internally consistent.

# In[21]:


# ============================================================
# SECTION 14 — PPO PRODUCTION READINESS
# ============================================================

checklist = pd.DataFrame(
    [
        (
            "Dataset loaded",
            len(ppo_dataset) > 0,
        ),
        (
            "Observation tensor",
            observation_matrix.shape == (3, 4),
        ),
        (
            "Action mask tensor",
            mask_tensor.shape == (3, 4),
        ),
        (
            "Action targets",
            action_target_tensor.shape == (3,),
        ),
        (
            "Reward tensor",
            reward_tensor.shape == (3,),
        ),
        (
            "Done tensor",
            done_tensor.shape == (3,),
        ),
        (
            "Mini-batches",
            len(mini_batches) == 2,
        ),
        (
            "PPO batch",
            len(ppo_batch) == 5,
        ),
    ],
    columns=[
        "Component",
        "Ready",
    ],
)

ready_count = int(
    checklist["Ready"].sum()
)

print("NOTEBOOK 40 — PPO READINESS")
print("=" * 70)

display(checklist)

print(
    "Ready components :",
    ready_count,
    "/",
    len(checklist),
)

assert checklist["Ready"].all()

print()
print("✅ SECTION 14 PPO READINESS PASSED")


# ## Section 15 — Export PPO Training Package
# 
# #### Export the complete PPO training package and verify it can be reloaded without modification.

# In[22]:


# ============================================================
# SECTION 15 — EXPORT PPO TRAINING PACKAGE
# ============================================================

import pickle

NOTEBOOK40_REPORT_DIR = REPORTS_DIR / "notebook40"
NOTEBOOK40_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPORT_FILE = (
    NOTEBOOK40_REPORT_DIR
    / "ppo_training_package.pkl"
)

metadata = {
    "num_samples": len(ppo_dataset),
    "observation_dim": observation_matrix.shape[1],
    "action_dim": len(action_vocabulary),
    "num_batches": len(mini_batches),
}

export_package = {
    "metadata": metadata,
    "ppo_batch": ppo_batch,
}

with open(
    EXPORT_FILE,
    "wb",
) as f:
    pickle.dump(
        export_package,
        f,
    )

assert EXPORT_FILE.exists()

with open(
    EXPORT_FILE,
    "rb",
) as f:
    reload_package = pickle.load(
        f
    )

print("NOTEBOOK 40 — PPO EXPORT")
print("=" * 70)

print(
    "Export file :",
    EXPORT_FILE,
)

print()

metadata_df = pd.DataFrame(
    reload_package["metadata"].items(),
    columns=[
        "Field",
        "Value",
    ],
)

display(
    metadata_df
)

assert (
    reload_package["metadata"]["num_samples"]
    == 3
)

assert (
    reload_package["metadata"]["observation_dim"]
    == 4
)

assert (
    reload_package["metadata"]["action_dim"]
    == 4
)

assert (
    reload_package["metadata"]["num_batches"]
    == 2
)

assert len(
    reload_package["ppo_batch"]
) == 5

print()
print("🏆 NOTEBOOK 40 COMPLETE")


# In[ ]:




