#!/usr/bin/env python
# coding: utf-8

# # Notebook 39 — Production Dataset Loader
# 
# ## Objectives
# 
# - Load RL transition datasets exported by Notebook 38
# - Support CSV and Parquet formats
# - Deserialize nested observation, legal-move, and mask fields
# - Validate the 16-field production contract
# - Build deterministic train, validation, and test splits
# - Create batching and shuffling utilities
# - Verify tensor shapes and data types
# - Detect invalid masks, illegal chosen actions, and duplicate sample IDs
# - Export dataset manifests and split reports
# - Prepare clean inputs for the PPO policy/value model

# ## Section 1 — Project Setup
# 
# #### Locate the project root, configure Notebook 38 input paths and Notebook 39 output paths, initialize deterministic settings, and prepare the dataset-loading environment.

# In[1]:


# ============================================================
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import json
import random
import sys
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

NOTEBOOK38_REPORT_DIR = (
    REPORTS_DIR / "notebook38"
)

NOTEBOOK39_REPORT_DIR = (
    REPORTS_DIR / "notebook39"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK39_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 39

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

print("NOTEBOOK 39 — PROJECT SETUP")
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
    "Notebook 38 reports    :",
    NOTEBOOK38_REPORT_DIR,
)

print(
    "Notebook 39 reports    :",
    NOTEBOOK39_REPORT_DIR,
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
assert NOTEBOOK39_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# ## Section 2 — Load Dataset Contract and Manifest
# 
# #### Load the canonical 16-field dataset contract from Notebook 35 and the adapter summary from Notebook 38.
# 
# #### This section also confirms whether the current input is a validation fixture or true simulator-generated data.

# In[2]:


# ============================================================
# SECTION 2 — LOAD CONTRACT AND MANIFEST
# ============================================================

CONTRACT_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_dataset_contract.csv"
)

ADAPTER_SUMMARY_FILE = (
    NOTEBOOK38_REPORT_DIR
    / "replay_adapter_summary.json"
)

assert CONTRACT_FILE.exists(), (
    f"Missing contract file: {CONTRACT_FILE}"
)

assert ADAPTER_SUMMARY_FILE.exists(), (
    "Missing Notebook 38 adapter summary: "
    f"{ADAPTER_SUMMARY_FILE}"
)

production_contract_df = pd.read_csv(
    CONTRACT_FILE
)

with ADAPTER_SUMMARY_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    adapter_summary = json.load(
        file
    )

required_contract_fields = (
    production_contract_df[
        "field"
    ]
    .astype(str)
    .tolist()
)

print("NOTEBOOK 39 — CONTRACT AND MANIFEST LOAD")
print("=" * 70)

print(
    "Contract fields        :",
    len(required_contract_fields),
)

print(
    "Dataset status         :",
    adapter_summary.get(
        "dataset_status"
    ),
)

print(
    "Simulator data         :",
    adapter_summary.get(
        "production_simulator_data"
    ),
)

print(
    "Validation rows        :",
    adapter_summary.get(
        "validation_transition_count"
    ),
)

display(
    production_contract_df
)

assert len(
    required_contract_fields
) == 16

assert adapter_summary.get(
    "dataset_status"
) == "VALIDATION_FIXTURE"

assert adapter_summary.get(
    "production_simulator_data"
) is False

print()
print(
    "✅ SECTION 2 CONTRACT AND MANIFEST LOAD PASSED"
)


# ## Section 3 — Load CSV and Parquet Sources
# 
# #### Load both Notebook 38 dataset formats and confirm they contain the same rows and the same 16-field schema.
# 
# #### Nested list values in CSV are deserialized from JSON strings. Parquet should preserve those values directly.

# In[7]:


# ============================================================
# SECTION 3 — LOAD CSV AND PARQUET SOURCES
# ============================================================

CSV_DATASET_FILE = (
    NOTEBOOK38_REPORT_DIR
    / "rl_transition_validation_fixture.csv"
)

PARQUET_DATASET_FILE = (
    NOTEBOOK38_REPORT_DIR
    / "rl_transition_validation_fixture.parquet"
)

assert CSV_DATASET_FILE.exists(), (
    f"Missing CSV dataset: {CSV_DATASET_FILE}"
)

assert PARQUET_DATASET_FILE.exists(), (
    f"Missing Parquet dataset: {PARQUET_DATASET_FILE}"
)

csv_dataset_df = pd.read_csv(
    CSV_DATASET_FILE
)

parquet_dataset_df = pd.read_parquet(
    PARQUET_DATASET_FILE
)

nested_columns = [
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
]


def deserialize_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        str,
    ):
        return json.loads(
            value
        )

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

    return value


for column in nested_columns:
    assert column in csv_dataset_df.columns
    assert column in parquet_dataset_df.columns

    csv_dataset_df[column] = (
        csv_dataset_df[column]
        .apply(
            deserialize_json_value
        )
    )

print("NOTEBOOK 39 — DATASET SOURCE LOAD")
print("=" * 70)

print(
    "CSV rows               :",
    len(csv_dataset_df),
)

print(
    "CSV columns            :",
    len(csv_dataset_df.columns),
)

print(
    "Parquet rows           :",
    len(parquet_dataset_df),
)

print(
    "Parquet columns        :",
    len(parquet_dataset_df.columns),
)

print()
print("CSV preview")
print("-" * 70)

display(
    csv_dataset_df.head()
)

print()
print("Parquet preview")
print("-" * 70)

display(
    parquet_dataset_df.head()
)

assert len(
    csv_dataset_df
) == len(
    parquet_dataset_df
)

assert set(
    csv_dataset_df.columns
) == set(
    required_contract_fields
)

assert set(
    parquet_dataset_df.columns
) == set(
    required_contract_fields
)

assert len(
    csv_dataset_df.columns
) == 16

assert len(
    parquet_dataset_df.columns
) == 16

print()
print(
    "✅ SECTION 3 DATASET SOURCE LOAD PASSED"
)


# ## Section 4 — Cross-Format Equality Audit
# 
# #### Normalize the CSV and Parquet datasets and verify that both formats preserve equivalent values for every production field.

# In[8]:


# ============================================================
# SECTION 4 — CROSS-FORMAT EQUALITY AUDIT
# ============================================================

def canonicalize_nested_value(
    value: Any,
) -> str:
    if isinstance(
        value,
        np.ndarray,
    ):
        value = value.tolist()

    if isinstance(
        value,
        tuple,
    ):
        value = list(
            value
        )

    if isinstance(
        value,
        list,
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(
        value
    )


csv_compare_df = (
    csv_dataset_df[
        required_contract_fields
    ]
    .copy()
)

parquet_compare_df = (
    parquet_dataset_df[
        required_contract_fields
    ]
    .copy()
)

for column in nested_columns:
    csv_compare_df[column] = (
        csv_compare_df[column]
        .apply(
            canonicalize_nested_value
        )
    )

    parquet_compare_df[column] = (
        parquet_compare_df[column]
        .apply(
            canonicalize_nested_value
        )
    )

comparison_rows = []

for column in required_contract_fields:
    csv_values = (
        csv_compare_df[column]
        .astype(str)
        .tolist()
    )

    parquet_values = (
        parquet_compare_df[column]
        .astype(str)
        .tolist()
    )

    comparison_rows.append(
        {
            "field": column,
            "equal": (
                csv_values
                == parquet_values
            ),
        }
    )

format_equality_df = pd.DataFrame(
    comparison_rows
)

print("NOTEBOOK 39 — CROSS-FORMAT EQUALITY")
print("=" * 70)

display(
    format_equality_df
)

matching_fields = int(
    format_equality_df[
        "equal"
    ].sum()
)

print()
print(
    "Matching fields        :",
    matching_fields,
    "/",
    len(format_equality_df),
)

nonmatching_fields = (
    format_equality_df.loc[
        ~format_equality_df[
            "equal"
        ],
        "field",
    ]
    .tolist()
)

print(
    "Nonmatching fields     :",
    nonmatching_fields,
)

assert format_equality_df[
    "equal"
].all(), (
    "CSV and Parquet differ for: "
    f"{nonmatching_fields}"
)

print()
print(
    "✅ SECTION 4 CROSS-FORMAT EQUALITY PASSED"
)


# ## Section 5 — Production Dataset Loader
# 
# #### Create a reusable loader that supports CSV and Parquet sources, deserializes nested values, enforces the canonical 16-field contract, and returns a consistently ordered DataFrame.

# In[9]:


# ============================================================
# SECTION 5 — PRODUCTION DATASET LOADER
# ============================================================

class ProductionDatasetLoader:

    def __init__(
        self,
        contract_fields: list[str],
        nested_columns: list[str],
    ):
        self.contract_fields = list(
            contract_fields
        )

        self.nested_columns = list(
            nested_columns
        )

    def _deserialize_nested_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        result = dataframe.copy()

        for column in self.nested_columns:
            if column not in result.columns:
                raise ValueError(
                    "Missing nested column: "
                    f"{column}"
                )

            result[column] = (
                result[column]
                .apply(
                    deserialize_json_value
                )
            )

        return result

    def validate_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        actual_columns = set(
            dataframe.columns
        )

        required_columns = set(
            self.contract_fields
        )

        missing_columns = sorted(
            required_columns
            - actual_columns
        )

        extra_columns = sorted(
            actual_columns
            - required_columns
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing contract "
                f"columns: {missing_columns}"
            )

        if extra_columns:
            raise ValueError(
                "Dataset contains unexpected "
                f"columns: {extra_columns}"
            )

    def load_csv(
        self,
        path: Path,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {path}"
            )

        dataframe = pd.read_csv(
            path
        )

        dataframe = (
            self._deserialize_nested_columns(
                dataframe
            )
        )

        self.validate_schema(
            dataframe
        )

        return dataframe[
            self.contract_fields
        ].copy()

    def load_parquet(
        self,
        path: Path,
    ) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path}"
            )

        dataframe = pd.read_parquet(
            path
        )

        dataframe = (
            self._deserialize_nested_columns(
                dataframe
            )
        )

        self.validate_schema(
            dataframe
        )

        return dataframe[
            self.contract_fields
        ].copy()

    def load(
        self,
        path: Path,
    ) -> pd.DataFrame:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self.load_csv(
                path
            )

        if suffix == ".parquet":
            return self.load_parquet(
                path
            )

        raise ValueError(
            "Unsupported dataset format: "
            f"{suffix}"
        )


dataset_loader = ProductionDatasetLoader(
    contract_fields=(
        required_contract_fields
    ),
    nested_columns=nested_columns,
)

loaded_csv_df = dataset_loader.load(
    CSV_DATASET_FILE
)

loaded_parquet_df = dataset_loader.load(
    PARQUET_DATASET_FILE
)

print("NOTEBOOK 39 — PRODUCTION DATASET LOADER")
print("=" * 70)

print(
    "CSV loaded rows        :",
    len(loaded_csv_df),
)

print(
    "Parquet loaded rows    :",
    len(loaded_parquet_df),
)

print(
    "Ordered columns        :",
    len(loaded_parquet_df.columns),
)

assert (
    loaded_csv_df.columns.tolist()
    == required_contract_fields
)

assert (
    loaded_parquet_df.columns.tolist()
    == required_contract_fields
)

assert len(
    loaded_csv_df
) == len(
    loaded_parquet_df
)

print()
print(
    "✅ SECTION 5 DATASET LOADER PASSED"
)


# ## Section 6 — Dataset Integrity Audit
# 
# #### Audit the loaded dataset for duplicate sample identifiers, missing values, invalid observation vectors, malformed masks, illegal selected actions, invalid action indices, and inconsistent terminal states.

# In[10]:


# ============================================================
# SECTION 6 — DATASET INTEGRITY AUDIT
# ============================================================

dataset_df = loaded_parquet_df.copy()

action_vocabulary_size = max(
    len(mask)
    for mask in dataset_df[
        "legal_move_mask"
    ]
)


def validate_dataset_row(
    row: pd.Series,
    action_vocabulary_size: int,
) -> dict[str, Any]:
    observation = row[
        "observation_vector"
    ]

    legal_moves = row[
        "legal_moves"
    ]

    legal_mask = row[
        "legal_move_mask"
    ]

    chosen_move = str(
        row["chosen_move"]
    )

    chosen_index = int(
        row["chosen_move_index"]
    )

    observation_valid = (
        isinstance(
            observation,
            list,
        )
        and len(
            observation
        ) > 0
        and all(
            isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            for value in observation
        )
    )

    legal_moves_valid = (
        isinstance(
            legal_moves,
            list,
        )
        and len(
            legal_moves
        ) > 0
        and all(
            isinstance(
                move,
                str,
            )
            for move in legal_moves
        )
    )

    mask_valid = (
        isinstance(
            legal_mask,
            list,
        )
        and len(
            legal_mask
        )
        == action_vocabulary_size
        and set(
            legal_mask
        ).issubset(
            {
                0,
                1,
            }
        )
    )

    index_in_range = (
        0
        <= chosen_index
        < len(
            legal_moves
        )
    )

    chosen_move_matches = (
        index_in_range
        and legal_moves[
            chosen_index
        ]
        == chosen_move
    )

    chosen_move_legal = (
        chosen_move
        in legal_moves
    )

    sample_id_valid = bool(
        str(
            row["sample_id"]
        ).strip()
    )

    return {
        "sample_id": row[
            "sample_id"
        ],
        "observation_valid": (
            observation_valid
        ),
        "legal_moves_valid": (
            legal_moves_valid
        ),
        "mask_valid": (
            mask_valid
        ),
        "index_in_range": (
            index_in_range
        ),
        "chosen_move_matches": (
            chosen_move_matches
        ),
        "chosen_move_legal": (
            chosen_move_legal
        ),
        "sample_id_valid": (
            sample_id_valid
        ),
    }


integrity_rows = [
    validate_dataset_row(
        row,
        action_vocabulary_size,
    )
    for _, row in dataset_df.iterrows()
]

integrity_audit_df = pd.DataFrame(
    integrity_rows
)

duplicate_sample_ids = int(
    dataset_df[
        "sample_id"
    ].duplicated().sum()
)

missing_value_counts = (
    dataset_df.isna()
    .sum()
)

columns_with_missing_values = (
    missing_value_counts.loc[
        missing_value_counts > 0
    ]
    .to_dict()
)

quality_columns = [
    "observation_valid",
    "legal_moves_valid",
    "mask_valid",
    "index_in_range",
    "chosen_move_matches",
    "chosen_move_legal",
    "sample_id_valid",
]

print("NOTEBOOK 39 — DATASET INTEGRITY AUDIT")
print("=" * 70)

display(
    integrity_audit_df
)

print()
print(
    "Rows audited           :",
    len(integrity_audit_df),
)

print(
    "Duplicate sample IDs   :",
    duplicate_sample_ids,
)

print(
    "Columns with missing   :",
    columns_with_missing_values,
)

print(
    "Action vocabulary size :",
    action_vocabulary_size,
)

print(
    "Terminal transitions   :",
    int(
        dataset_df[
            "done"
        ].sum()
    ),
)

assert duplicate_sample_ids == 0

assert not columns_with_missing_values

assert (
    integrity_audit_df[
        quality_columns
    ]
    .all()
    .all()
)

assert (
    dataset_df[
        "done"
    ]
    .any()
)

print()
print(
    "✅ SECTION 6 DATASET INTEGRITY PASSED"
)


# ## Section 7 — Type and Shape Audit
# 
# #### Verify the observation-vector shape, action-mask shape, numeric fields, Boolean terminal flags, and categorical fields required by the future PPO model.

# In[11]:


# ============================================================
# SECTION 7 — TYPE AND SHAPE AUDIT
# ============================================================

observation_lengths = (
    dataset_df[
        "observation_vector"
    ]
    .apply(
        len
    )
)

mask_lengths = (
    dataset_df[
        "legal_move_mask"
    ]
    .apply(
        len
    )
)

numeric_columns = [
    "game_index",
    "turn",
    "chosen_move_index",
    "search_score",
    "search_depth",
    "nodes_searched",
    "reward",
]

categorical_columns = [
    "sample_id",
    "actor",
    "chosen_move",
    "winner",
    "source_dataset",
]

type_audit_rows = []

for column in numeric_columns:
    is_numeric = pd.api.types.is_numeric_dtype(
        dataset_df[column]
    )

    type_audit_rows.append(
        {
            "field": column,
            "expected_type": "numeric",
            "valid": bool(
                is_numeric
            ),
        }
    )

for column in categorical_columns:
    valid = (
        dataset_df[column]
        .apply(
            lambda value: isinstance(
                value,
                str,
            )
        )
        .all()
    )

    type_audit_rows.append(
        {
            "field": column,
            "expected_type": "string",
            "valid": bool(
                valid
            ),
        }
    )

done_valid = (
    dataset_df["done"]
    .apply(
        lambda value: isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        )
    )
    .all()
)

type_audit_rows.append(
    {
        "field": "done",
        "expected_type": "boolean",
        "valid": bool(
            done_valid
        ),
    }
)

type_shape_audit_df = pd.DataFrame(
    type_audit_rows
)

observation_dimension = int(
    observation_lengths.iloc[
        0
    ]
)

mask_dimension = int(
    mask_lengths.iloc[
        0
    ]
)

print("NOTEBOOK 39 — TYPE AND SHAPE AUDIT")
print("=" * 70)

display(
    type_shape_audit_df
)

print()
print(
    "Observation dimension  :",
    observation_dimension,
)

print(
    "Observation lengths    :",
    sorted(
        observation_lengths
        .unique()
        .tolist()
    ),
)

print(
    "Mask dimension         :",
    mask_dimension,
)

print(
    "Mask lengths           :",
    sorted(
        mask_lengths
        .unique()
        .tolist()
    ),
)

assert (
    observation_lengths
    == observation_dimension
).all()

assert (
    mask_lengths
    == mask_dimension
).all()

assert observation_dimension > 0
assert mask_dimension > 0

assert type_shape_audit_df[
    "valid"
].all()

print()
print(
    "✅ SECTION 7 TYPE AND SHAPE AUDIT PASSED"
)


# ## Section 8 — Production Dataset Object
# 
# #### Create a lightweight dataset wrapper that exposes deterministic indexing and prepares the dataset for PPO batching.

# In[12]:


# ============================================================
# SECTION 8 — PRODUCTION DATASET OBJECT
# ============================================================

class PPOTransitionDataset:

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

    def __len__(self):
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.dataframe.iloc[index]
            .to_dict()
        )

    def head(
        self,
        rows=5,
    ):
        return self.dataframe.head(
            rows
        )

dataset = PPOTransitionDataset(
    loaded_parquet_df
)

print("NOTEBOOK 39 — DATASET OBJECT")
print("=" * 70)

print(
    "Dataset length :",
    len(dataset),
)

display(
    pd.DataFrame(
        [dataset[0]]
    )
)

assert len(dataset) == 3

print()
print("✅ SECTION 8 DATASET OBJECT PASSED")


# ## Section 9 — Deterministic Dataset Split
# 
# #### Split the PPO dataset into train, validation, and test sets using deterministic ordering. Small validation fixtures automatically remain entirely in the training split.

# In[13]:


# ============================================================
# SECTION 9 — DETERMINISTIC DATASET SPLIT
# ============================================================

dataset_size = len(dataset)

if dataset_size < 10:
    train_indices = list(range(dataset_size))
    validation_indices = []
    test_indices = []
else:
    train_end = int(dataset_size * 0.8)
    validation_end = int(dataset_size * 0.9)

    train_indices = list(range(train_end))
    validation_indices = list(range(train_end, validation_end))
    test_indices = list(range(validation_end, dataset_size))

print("NOTEBOOK 39 — DATASET SPLIT")
print("=" * 70)

print("Dataset size        :", dataset_size)
print("Training samples    :", len(train_indices))
print("Validation samples  :", len(validation_indices))
print("Testing samples     :", len(test_indices))

split_summary = pd.DataFrame(
    {
        "split": [
            "train",
            "validation",
            "test",
        ],
        "count": [
            len(train_indices),
            len(validation_indices),
            len(test_indices),
        ],
    }
)

display(split_summary)

assert (
    len(train_indices)
    + len(validation_indices)
    + len(test_indices)
    == dataset_size
)

print()
print("✅ SECTION 9 DATASET SPLIT PASSED")


# ## Section 10 — Mini-Batch Generator
# 
# #### Create deterministic mini-batches from the PPO dataset.

# In[14]:


# ============================================================
# SECTION 10 — MINI-BATCH GENERATOR
# ============================================================

class MiniBatchGenerator:

    def __init__(
        self,
        dataset,
        batch_size=32,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):

        for start in range(
            0,
            len(self.dataset),
            self.batch_size,
        ):
            end = min(
                start + self.batch_size,
                len(self.dataset),
            )

            yield [
                self.dataset[index]
                for index in range(start, end)
            ]


generator = MiniBatchGenerator(
    dataset,
    batch_size=2,
)

batches = list(generator)

print("NOTEBOOK 39 — MINI-BATCH GENERATOR")
print("=" * 70)

print("Dataset size :", len(dataset))
print("Batch size   :", generator.batch_size)
print("Batches      :", len(batches))

summary = pd.DataFrame(
    {
        "batch": list(range(len(batches))),
        "batch_size": [
            len(batch)
            for batch in batches
        ],
    }
)

display(summary)

assert len(batches) == 2
assert len(batches[0]) == 2
assert len(batches[1]) == 1

print()
print("✅ SECTION 10 MINI-BATCH GENERATOR PASSED")


# ## Section 11 — Dataset Statistics
# 
# #### Summarize important statistics of the PPO dataset before training.

# In[15]:


# ============================================================
# SECTION 11 — DATASET STATISTICS
# ============================================================

statistics = {
    "Rows": len(dataset_df),
    "Columns": len(dataset_df.columns),
    "Observation dimension": len(dataset_df.iloc[0]["observation_vector"]),
    "Action vocabulary": len(
        {
            action
            for actions in dataset_df["legal_moves"]
            for action in actions
        }
    ),
    "Terminal transitions": int(dataset_df["done"].sum()),
    "Unique winners": dataset_df["winner"].nunique(),
    "Average reward": float(dataset_df["reward"].mean()),
    "Average search score": float(dataset_df["search_score"].mean()),
}

stats_df = pd.DataFrame(
    {
        "Metric": statistics.keys(),
        "Value": statistics.values(),
    }
)

print("NOTEBOOK 39 — DATASET STATISTICS")
print("=" * 70)

display(stats_df)

assert statistics["Rows"] == len(dataset_df)
assert statistics["Columns"] == len(dataset_df.columns)
assert statistics["Observation dimension"] > 0
assert statistics["Action vocabulary"] >= 2

print()
print("✅ SECTION 11 DATASET STATISTICS PASSED")


# ## Section 12 — Production Readiness Checklist
# 
# #### Verify every required PPO dataset subsystem is available before training.

# In[17]:


# ============================================================
# SECTION 12 — PRODUCTION READINESS
# ============================================================

checklist = pd.DataFrame(
    [
        (
            "CSV Export",
            CSV_DATASET_FILE.exists(),
        ),
        (
            "Parquet Export",
            PARQUET_DATASET_FILE.exists(),
        ),
        (
            "Dataset Object",
            len(dataset) > 0,
        ),
        (
            "Mini-Batch Generator",
            len(batches) > 0,
        ),
        (
            "Dataset Split",
            len(train_indices) > 0,
        ),
        (
            "Integrity Audit",
            bool(
                integrity_audit_df[
                    quality_columns
                ]
                .all()
                .all()
            ),
        ),
        (
            "Cross-format Equality",
            bool(
                format_equality_df[
                    "equal"
                ].all()
            ),
        ),
        (
            "Statistics",
            len(stats_df) > 0,
        ),
    ],
    columns=[
        "Component",
        "Ready",
    ],
)

display(
    checklist
)

ready_count = int(
    checklist["Ready"].sum()
)

not_ready_components = (
    checklist.loc[
        ~checklist["Ready"],
        "Component",
    ]
    .tolist()
)

print("NOTEBOOK 39 — PRODUCTION CHECKLIST")
print("=" * 70)

print(
    "Production components  :",
    ready_count,
    "/",
    len(checklist),
)

print(
    "Not ready components   :",
    not_ready_components,
)

assert checklist["Ready"].all(), (
    "One or more components are not ready: "
    f"{not_ready_components}"
)

print()
print(
    "✅ SECTION 12 PRODUCTION CHECKLIST PASSED"
)


# ## Section 13 — Export Loader Reports
# 
# #### Export the dataset integrity audit, type-and-shape audit, split summary, statistics, readiness checklist, and a loader manifest for downstream PPO training.

# In[18]:


# ============================================================
# SECTION 13 — EXPORT LOADER REPORTS
# ============================================================

REPORT_DIR = NOTEBOOK39_REPORT_DIR

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INTEGRITY_FILE = (
    REPORT_DIR
    / "dataset_integrity_audit.csv"
)

TYPE_SHAPE_FILE = (
    REPORT_DIR
    / "dataset_type_shape_audit.csv"
)

SPLIT_FILE = (
    REPORT_DIR
    / "dataset_split_summary.csv"
)

STATISTICS_FILE = (
    REPORT_DIR
    / "dataset_statistics.csv"
)

CHECKLIST_FILE = (
    REPORT_DIR
    / "production_readiness.csv"
)

MANIFEST_FILE = (
    REPORT_DIR
    / "dataset_loader_manifest.json"
)

integrity_audit_df.to_csv(
    INTEGRITY_FILE,
    index=False,
)

type_shape_audit_df.to_csv(
    TYPE_SHAPE_FILE,
    index=False,
)

split_summary.to_csv(
    SPLIT_FILE,
    index=False,
)

stats_df.to_csv(
    STATISTICS_FILE,
    index=False,
)

checklist.to_csv(
    CHECKLIST_FILE,
    index=False,
)

loader_manifest = {
    "notebook": 39,
    "name": "Production Dataset Loader",
    "dataset_status": (
        adapter_summary[
            "dataset_status"
        ]
    ),
    "production_simulator_data": (
        adapter_summary[
            "production_simulator_data"
        ]
    ),
    "row_count": int(
        len(dataset_df)
    ),
    "column_count": int(
        len(dataset_df.columns)
    ),
    "observation_dimension": int(
        observation_dimension
    ),
    "action_mask_dimension": int(
        mask_dimension
    ),
    "train_count": int(
        len(train_indices)
    ),
    "validation_count": int(
        len(validation_indices)
    ),
    "test_count": int(
        len(test_indices)
    ),
    "duplicate_sample_ids": int(
        duplicate_sample_ids
    ),
    "terminal_transitions": int(
        dataset_df[
            "done"
        ].sum()
    ),
    "all_integrity_checks_passed": bool(
        integrity_audit_df[
            quality_columns
        ]
        .all()
        .all()
    ),
    "all_type_checks_passed": bool(
        type_shape_audit_df[
            "valid"
        ].all()
    ),
    "all_readiness_checks_passed": bool(
        checklist[
            "Ready"
        ].all()
    ),
}

with MANIFEST_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        loader_manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("NOTEBOOK 39 — REPORT EXPORT")
print("=" * 70)

for label, path in [
    ("Integrity audit", INTEGRITY_FILE),
    ("Type/shape audit", TYPE_SHAPE_FILE),
    ("Split summary", SPLIT_FILE),
    ("Statistics", STATISTICS_FILE),
    ("Readiness", CHECKLIST_FILE),
    ("Manifest", MANIFEST_FILE),
]:
    print(
        f"{label:<20}:",
        path.relative_to(
            PROJECT_ROOT
        ),
    )

assert INTEGRITY_FILE.exists()
assert TYPE_SHAPE_FILE.exists()
assert SPLIT_FILE.exists()
assert STATISTICS_FILE.exists()
assert CHECKLIST_FILE.exists()
assert MANIFEST_FILE.exists()

print()
print(
    "✅ SECTION 13 REPORT EXPORT PASSED"
)


# ## Section 14 — Final Reload Validation
# 
# #### Reload the exported reports and manifest, verify their consistency, and record the final Notebook 39 status.

# In[19]:


# ============================================================
# SECTION 14 — FINAL RELOAD VALIDATION
# ============================================================

integrity_reload_df = pd.read_csv(
    INTEGRITY_FILE
)

type_shape_reload_df = pd.read_csv(
    TYPE_SHAPE_FILE
)

split_reload_df = pd.read_csv(
    SPLIT_FILE
)

statistics_reload_df = pd.read_csv(
    STATISTICS_FILE
)

checklist_reload_df = pd.read_csv(
    CHECKLIST_FILE
)

with MANIFEST_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    manifest_reload = json.load(
        file
    )

print("NOTEBOOK 39 — FINAL VALIDATION")
print("=" * 70)

print(
    "Integrity rows         :",
    len(integrity_reload_df),
)

print(
    "Type/shape rows        :",
    len(type_shape_reload_df),
)

print(
    "Split rows             :",
    len(split_reload_df),
)

print(
    "Statistics rows        :",
    len(statistics_reload_df),
)

print(
    "Checklist rows         :",
    len(checklist_reload_df),
)

print(
    "Dataset rows           :",
    manifest_reload[
        "row_count"
    ],
)

print(
    "Dataset columns        :",
    manifest_reload[
        "column_count"
    ],
)

print(
    "Observation dimension :",
    manifest_reload[
        "observation_dimension"
    ],
)

print(
    "Mask dimension        :",
    manifest_reload[
        "action_mask_dimension"
    ],
)

print(
    "Dataset status        :",
    manifest_reload[
        "dataset_status"
    ],
)

print(
    "Simulator data        :",
    manifest_reload[
        "production_simulator_data"
    ],
)

assert len(
    integrity_reload_df
) == len(
    dataset_df
)

assert len(
    split_reload_df
) == 3

assert manifest_reload[
    "row_count"
] == len(
    dataset_df
)

assert manifest_reload[
    "column_count"
] == 16

assert manifest_reload[
    "all_integrity_checks_passed"
] is True

assert manifest_reload[
    "all_type_checks_passed"
] is True

assert manifest_reload[
    "all_readiness_checks_passed"
] is True

assert manifest_reload[
    "dataset_status"
] == "VALIDATION_FIXTURE"

assert manifest_reload[
    "production_simulator_data"
] is False

print()
print(
    "🏆 NOTEBOOK 39 COMPLETE"
)


# In[ ]:




