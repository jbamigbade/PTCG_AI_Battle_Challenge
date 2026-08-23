#!/usr/bin/env python
# coding: utf-8

# # Notebook 35 — Policy Dataset Enrichment
# 
# ## Objectives
# 
# - Audit the bootstrap policy dataset produced by Notebook 34
# - Compare the available data with the production observation schema
# - Define the missing fields required for meaningful policy learning
# - Represent all legal actions available before each decision
# - Preserve the chosen action and search-engine supervision
# - Add richer pre-decision state features
# - Detect label leakage and deterministic policies
# - Generate a training-ready policy dataset
# - Export validated CSV, Parquet, JSON, and audit artifacts

# # Section 1 — Project Setup

# ## Section 1 — Project Setup
# 
# #### Locate the project root, configure paths, initialize reproducibility settings, and prepare the Notebook 35 report directory.

# In[1]:


# ============================================================
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import importlib
import json
import math
import random
import re
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

NOTEBOOK33_REPORT_DIR = (
    REPORTS_DIR / "notebook33"
)

NOTEBOOK34_REPORT_DIR = (
    REPORTS_DIR / "notebook34"
)

NOTEBOOK35_REPORT_DIR = (
    REPORTS_DIR / "notebook35"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK35_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 35

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option(
    "display.max_columns",
    100,
)

pd.set_option(
    "display.width",
    180,
)

pd.set_option(
    "display.max_colwidth",
    140,
)

print("NOTEBOOK 35 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
)

print(
    "Notebook 33 reports    :",
    NOTEBOOK33_REPORT_DIR,
)

print(
    "Notebook 34 reports    :",
    NOTEBOOK34_REPORT_DIR,
)

print(
    "Notebook 35 reports    :",
    NOTEBOOK35_REPORT_DIR,
)

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK33_REPORT_DIR.exists()
assert NOTEBOOK34_REPORT_DIR.exists()
assert NOTEBOOK35_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# # Section 2 — Load Notebook 34 Artifacts

# ## Section 2 — Load Notebook 34 Policy Artifacts
# 
# #### Load the bootstrap policy dataset, label mapping, and fitness report produced by Notebook 34. These artifacts establish the current baseline and document why richer decision data is required.

# In[2]:


# ============================================================
# SECTION 2 — LOAD NOTEBOOK 34 ARTIFACTS
# ============================================================

BOOTSTRAP_CSV_FILE = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_bootstrap.csv"
)

BOOTSTRAP_PARQUET_FILE = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_bootstrap.parquet"
)

POLICY_MAPPING_FILE = (
    NOTEBOOK34_REPORT_DIR
    / "policy_label_mapping.json"
)

FITNESS_REPORT_FILE = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_fitness.json"
)

QUALITY_SUMMARY_FILE = (
    NOTEBOOK34_REPORT_DIR
    / "policy_quality_summary.csv"
)

required_notebook34_files = [
    BOOTSTRAP_CSV_FILE,
    POLICY_MAPPING_FILE,
    FITNESS_REPORT_FILE,
    QUALITY_SUMMARY_FILE,
]

missing_notebook34_files = [
    path
    for path in required_notebook34_files
    if not path.exists()
]

assert not missing_notebook34_files, (
    "Missing Notebook 34 artifacts:\n"
    + "\n".join(
        str(path)
        for path in missing_notebook34_files
    )
)

if BOOTSTRAP_PARQUET_FILE.exists():
    bootstrap_policy_df = pd.read_parquet(
        BOOTSTRAP_PARQUET_FILE
    )

    bootstrap_source = (
        BOOTSTRAP_PARQUET_FILE
    )
else:
    bootstrap_policy_df = pd.read_csv(
        BOOTSTRAP_CSV_FILE
    )

    bootstrap_source = (
        BOOTSTRAP_CSV_FILE
    )

with POLICY_MAPPING_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    policy_mapping = json.load(
        file
    )

with FITNESS_REPORT_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    fitness_report = json.load(
        file
    )

quality_summary_df = pd.read_csv(
    QUALITY_SUMMARY_FILE
)

print("NOTEBOOK 35 — NOTEBOOK 34 ARTIFACT LOAD")
print("=" * 70)

print(
    "Bootstrap source       :",
    bootstrap_source.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Bootstrap rows         :",
    f"{len(bootstrap_policy_df):,}",
)

print(
    "Bootstrap columns      :",
    len(bootstrap_policy_df.columns),
)

print(
    "Fitness status         :",
    fitness_report[
        "fitness_status"
    ],
)

print(
    "Training ready         :",
    fitness_report[
        "training_ready"
    ],
)

print(
    "Policy classes         :",
    fitness_report[
        "unique_policy_classes"
    ],
)

print()
print("Bootstrap preview")
print("-" * 70)

display(
    bootstrap_policy_df.head(10)
)

print()
print("Notebook 34 quality summary")
print("-" * 70)

display(
    quality_summary_df
)

assert len(bootstrap_policy_df) > 0

assert (
    fitness_report[
        "fitness_status"
    ]
    == "BOOTSTRAP_ONLY"
)

assert (
    fitness_report[
        "training_ready"
    ]
    is False
)

print()
print(
    "✅ SECTION 2 NOTEBOOK 34 LOAD PASSED"
)


# # Section 3 — Load Notebook 33 Observation Schema

# In[3]:


# ============================================================
# SECTION 3 — LOAD NOTEBOOK 33 OBSERVATION SCHEMA
# ============================================================

FEATURE_SCHEMA_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "feature_schema.csv"
)

MOVE_FEATURES_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "move_features.csv"
)

SAMPLE_OBSERVATIONS_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "sample_observations.csv"
)

required_notebook33_files = [
    FEATURE_SCHEMA_FILE,
    MOVE_FEATURES_FILE,
    SAMPLE_OBSERVATIONS_FILE,
]

missing_notebook33_files = [
    path
    for path in required_notebook33_files
    if not path.exists()
]

assert not missing_notebook33_files, (
    "Missing Notebook 33 artifacts:\n"
    + "\n".join(
        str(path)
        for path in missing_notebook33_files
    )
)

feature_schema_df = pd.read_csv(
    FEATURE_SCHEMA_FILE
)

move_features_df = pd.read_csv(
    MOVE_FEATURES_FILE
)

sample_observations_df = pd.read_csv(
    SAMPLE_OBSERVATIONS_FILE
)

print("NOTEBOOK 35 — OBSERVATION SCHEMA LOAD")
print("=" * 70)

print(
    "Feature schema rows    :",
    len(feature_schema_df),
)

print(
    "Move feature rows      :",
    len(move_features_df),
)

print(
    "Sample observations    :",
    len(sample_observations_df),
)

print()
print("Feature schema")
print("-" * 70)

display(
    feature_schema_df
)

print()
print("Move features")
print("-" * 70)

display(
    move_features_df
)

print()
print("Sample observations")
print("-" * 70)

display(
    sample_observations_df.head(10)
)

assert len(feature_schema_df) > 0
assert len(move_features_df) > 0
assert len(sample_observations_df) > 0

print()
print(
    "✅ SECTION 3 OBSERVATION SCHEMA LOAD PASSED"
)


# # Section 4 — Policy Dataset Gap Analysis

# ## Section 4 — Policy Dataset Gap Analysis
# 
# Compare the bootstrap policy dataset with the production observation schema and define the additional fields required for meaningful policy learning.
# 
# The goal is to distinguish:
# 
# - fields already available;
# - fields available only after the move;
# - fields missing before the decision;
# - fields required to represent legal-action choices;
# - fields that may leak the policy label.

# In[5]:


# ============================================================
# SECTION 4 — POLICY DATASET GAP ANALYSIS
# ============================================================

print("NOTEBOOK 35 — POLICY DATASET GAP ANALYSIS")
print("=" * 70)

bootstrap_columns = set(
    bootstrap_policy_df.columns
)

observation_columns = set(
    sample_observations_df.columns
)

move_feature_columns = set(
    move_features_df.columns
)

required_policy_fields = {
    "game_index",
    "turn",
    "actor",
    "pokemon_name",
    "chosen_move",
    "chosen_move_index",
    "legal_moves",
    "legal_move_count",
    "observation_vector",
    "search_score",
    "search_depth",
    "nodes_searched",
    "winner",
    "reward",
}

available_fields = sorted(
    required_policy_fields
    & bootstrap_columns
)

missing_fields = sorted(
    required_policy_fields
    - bootstrap_columns
)

observation_overlap = sorted(
    bootstrap_columns
    & observation_columns
)

move_feature_overlap = sorted(
    bootstrap_columns
    & move_feature_columns
)

post_move_fields = [
    column
    for column in bootstrap_policy_df.columns
    if any(
        token in column.lower()
        for token in (
            "after",
            "final",
            "winner",
            "next_side",
        )
    )
]

potential_label_leakage_fields = [
    column
    for column in bootstrap_policy_df.columns
    if column in {
        "actor",
        "pokemon_name",
        "move_name",
        "policy_index",
        "is_training_agent_turn",
    }
]

gap_analysis_df = pd.DataFrame(
    {
        "category": [
            "bootstrap_columns",
            "required_policy_fields",
            "available_required_fields",
            "missing_required_fields",
            "observation_overlap",
            "move_feature_overlap",
            "post_move_fields",
            "potential_label_leakage_fields",
        ],
        "count": [
            len(bootstrap_columns),
            len(required_policy_fields),
            len(available_fields),
            len(missing_fields),
            len(observation_overlap),
            len(move_feature_overlap),
            len(post_move_fields),
            len(potential_label_leakage_fields),
        ],
        "fields": [
            ", ".join(
                sorted(bootstrap_columns)
            ),
            ", ".join(
                sorted(required_policy_fields)
            ),
            ", ".join(
                available_fields
            ),
            ", ".join(
                missing_fields
            ),
            ", ".join(
                observation_overlap
            ),
            ", ".join(
                move_feature_overlap
            ),
            ", ".join(
                post_move_fields
            ),
            ", ".join(
                potential_label_leakage_fields
            ),
        ],
    }
)

display(
    gap_analysis_df
)

print()
print("Critical missing fields")
print("-" * 70)

for field in missing_fields:
    print(
        f"  - {field}"
    )

print()
print("Post-move fields")
print("-" * 70)

for field in post_move_fields:
    print(
        f"  - {field}"
    )

print()
print("Potential label-leakage fields")
print("-" * 70)

for field in potential_label_leakage_fields:
    print(
        f"  - {field}"
    )

assert "legal_moves" in missing_fields
assert "legal_move_count" in missing_fields
assert "observation_vector" in missing_fields
assert "reward" in missing_fields

print()
print(
    "✅ SECTION 4 GAP ANALYSIS PASSED"
)


# # Section 5 — Define the Enriched Policy Record Schema

# ## Section 5 — Enriched Policy Record Schema
# 
# Define a stable schema for one pre-decision policy-learning example.
# 
# Each row should describe:
# 
# - the board state before a decision;
# - the acting player;
# - every legal action;
# - the selected action;
# - search supervision;
# - the game outcome;
# - data-quality metadata.

# In[6]:


# ============================================================
# SECTION 5 — ENRICHED POLICY RECORD SCHEMA
# ============================================================

ENRICHED_POLICY_SCHEMA = [
    {
        "field": "sample_id",
        "dtype": "string",
        "role": "identifier",
        "required": True,
        "description": (
            "Stable unique identifier for the "
            "decision sample."
        ),
    },
    {
        "field": "game_index",
        "dtype": "int64",
        "role": "identifier",
        "required": True,
        "description": (
            "Source self-play game identifier."
        ),
    },
    {
        "field": "turn",
        "dtype": "int64",
        "role": "state",
        "required": True,
        "description": (
            "Decision turn number."
        ),
    },
    {
        "field": "actor",
        "dtype": "string",
        "role": "state",
        "required": True,
        "description": (
            "Side making the decision."
        ),
    },
    {
        "field": "observation_vector",
        "dtype": "json",
        "role": "feature",
        "required": True,
        "description": (
            "Production observation features "
            "captured before the move."
        ),
    },
    {
        "field": "legal_moves",
        "dtype": "json",
        "role": "action_space",
        "required": True,
        "description": (
            "All legal actions available before "
            "the decision."
        ),
    },
    {
        "field": "legal_move_count",
        "dtype": "int64",
        "role": "action_space",
        "required": True,
        "description": (
            "Number of legal actions."
        ),
    },
    {
        "field": "chosen_move",
        "dtype": "string",
        "role": "target",
        "required": True,
        "description": (
            "Action selected by the policy or "
            "search agent."
        ),
    },
    {
        "field": "chosen_move_index",
        "dtype": "int64",
        "role": "target",
        "required": True,
        "description": (
            "Index of the selected action within "
            "the legal-action list."
        ),
    },
    {
        "field": "search_score",
        "dtype": "float64",
        "role": "supervision",
        "required": False,
        "description": (
            "Search evaluation assigned to the "
            "selected action."
        ),
    },
    {
        "field": "search_depth",
        "dtype": "int64",
        "role": "supervision",
        "required": False,
        "description": (
            "Search depth used for the decision."
        ),
    },
    {
        "field": "nodes_searched",
        "dtype": "int64",
        "role": "supervision",
        "required": False,
        "description": (
            "Number of search nodes evaluated."
        ),
    },
    {
        "field": "winner",
        "dtype": "string",
        "role": "outcome",
        "required": True,
        "description": (
            "Final game winner."
        ),
    },
    {
        "field": "reward",
        "dtype": "float64",
        "role": "outcome",
        "required": True,
        "description": (
            "Outcome reward from the acting "
            "side's perspective."
        ),
    },
    {
        "field": "source_dataset",
        "dtype": "string",
        "role": "metadata",
        "required": True,
        "description": (
            "Name of the source generator or "
            "benchmark."
        ),
    },
    {
        "field": "schema_version",
        "dtype": "string",
        "role": "metadata",
        "required": True,
        "description": (
            "Version of the enriched policy schema."
        ),
    },
]

enriched_schema_df = pd.DataFrame(
    ENRICHED_POLICY_SCHEMA
)

print("NOTEBOOK 35 — ENRICHED POLICY SCHEMA")
print("=" * 70)

display(
    enriched_schema_df
)

required_schema_fields = (
    enriched_schema_df.loc[
        enriched_schema_df["required"],
        "field",
    ]
    .tolist()
)

print()
print(
    "Total schema fields    :",
    len(enriched_schema_df),
)

print(
    "Required fields        :",
    len(required_schema_fields),
)

print(
    "Target fields          :",
    enriched_schema_df.loc[
        enriched_schema_df["role"]
        .eq("target"),
        "field",
    ].tolist(),
)

assert (
    enriched_schema_df["field"]
    .is_unique
)

assert {
    "legal_moves",
    "chosen_move",
    "chosen_move_index",
    "observation_vector",
    "reward",
}.issubset(
    set(
        enriched_schema_df["field"]
    )
)

print()
print(
    "✅ SECTION 5 ENRICHED SCHEMA PASSED"
)


# # SECTION 6 — Locate the Source of Legal Actions

# ## Section 6 — Locate Legal Action Sources
# 
# The enriched policy dataset requires the legal action set available before each decision.
# 
# This section searches all observation tables and move-feature tables for any representation of:
# 
# - legal moves
# - action masks
# - candidate moves
# - available actions
# - move indices
# 
# These will determine whether Notebook 35 can build a true policy-learning dataset or whether Notebook 36 must regenerate observations directly from the simulator.

# In[7]:


# ============================================================
# SECTION 6 — LEGAL ACTION DISCOVERY
# ============================================================

print("NOTEBOOK 35 — LEGAL ACTION DISCOVERY")
print("=" * 70)

datasets = {
    "observations": sample_observations_df,
    "move_features": move_features_df,
    "bootstrap_policy": bootstrap_policy_df,
}

keywords = [
    "legal",
    "action",
    "candidate",
    "move",
    "mask",
    "choice",
    "available",
    "index",
]

results = []

for name, df in datasets.items():

    matching = []

    for column in df.columns:

        lower = column.lower()

        if any(k in lower for k in keywords):
            matching.append(column)

    results.append(
        {
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "matching_columns": len(matching),
            "column_list": ", ".join(matching),
        }
    )

results_df = pd.DataFrame(results)

display(results_df)

print()

for _, row in results_df.iterrows():

    print("-" * 70)
    print(row["dataset"])

    if row["matching_columns"] == 0:
        print("  No candidate action columns found.")
    else:
        print(row["column_list"])

assert len(results_df) == 3

print()
print("✅ SECTION 6 LEGAL ACTION DISCOVERY PASSED")


# ## Section 7 — Missing Production Fields
# 
# The bootstrap dataset contains only the executed action.
# 
# A production PPO / Behavior Cloning dataset additionally requires:
# 
# - complete observation tensor
# - legal action list
# - legal action mask
# - chosen action index
# - reward
# - terminal flag
# 
# These fields cannot be reconstructed perfectly after gameplay.
# 
# Therefore Notebook 36 will regenerate them directly from the simulator.

# In[8]:


# ============================================================
# SECTION 7 — MISSING PRODUCTION FEATURES
# ============================================================

required_training_fields = [
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
    "chosen_move_index",
    "reward",
    "done",
]

existing_fields = set(bootstrap_policy_df.columns)

audit = []

for field in required_training_fields:

    audit.append(
        {
            "field": field,
            "present": field in existing_fields,
        }
    )

missing_df = pd.DataFrame(audit)

display(missing_df)

missing_count = (~missing_df["present"]).sum()

print()
print("Missing production fields :", missing_count)

production_ready = missing_count == 0

print("Production ready          :", production_ready)

assert missing_count >= 1

print()
print("✅ SECTION 7 PRODUCTION AUDIT PASSED")


# # SECTION 8 — Define the Production Dataset Contract

# ## Section 8 — Production Dataset Contract
# 
# This notebook establishes the complete dataset specification required for
# Behavior Cloning and PPO training.
# 
# Notebook 36 will satisfy every required field by querying the simulator
# before each action is selected.
# 
# Each decision sample will contain:
# 
# - observation vector
# - legal action list
# - legal action mask
# - chosen action
# - chosen action index
# - reward
# - terminal flag
# 
# This schema becomes the canonical policy-training format for the project.

# ## Section 8 — Production Dataset Contract
# 
# This notebook establishes the complete dataset specification required for
# Behavior Cloning and PPO training.
# 
# Notebook 36 will satisfy every required field by querying the simulator
# before each action is selected.
# 
# Each decision sample will contain:
# 
# - observation vector
# - legal action list
# - legal action mask
# - chosen action
# - chosen action index
# - reward
# - terminal flag
# 
# This schema becomes the canonical policy-training format for the project.

# In[11]:


# ============================================================
# SECTION 8 — PRODUCTION DATASET CONTRACT
# ============================================================

production_contract = pd.DataFrame(
    [
        ("sample_id", "identifier"),
        ("game_index", "identifier"),
        ("turn", "state"),
        ("actor", "state"),
        ("observation_vector", "state"),
        ("legal_moves", "action_space"),
        ("legal_move_mask", "action_space"),
        ("chosen_move", "target"),
        ("chosen_move_index", "target"),
        ("search_score", "optional"),
        ("search_depth", "optional"),
        ("nodes_searched", "optional"),
        ("reward", "training"),
        ("done", "training"),
        ("winner", "training"),
        ("source_dataset", "metadata"),
    ],
    columns=[
        "field",
        "category",
    ],
)

display(production_contract)

category_summary = (
    production_contract
    .groupby("category")
    .size()
    .reset_index(name="count")
)

print()
print("Category summary")
display(category_summary)

assert len(production_contract) == 16

print()
print("✅ SECTION 8 DATASET CONTRACT PASSED")


# # SECTION 9 — Bootstrap → Production Mapping

# ## Section 9 — Bootstrap Dataset Mapping
# 
# This table documents how every field in the bootstrap dataset maps to the
# final production dataset.
# 
# Fields marked "Generated" cannot be recovered from Notebook 34 and must
# be produced directly by the simulator in Notebook 36.

# In[12]:


# ============================================================
# SECTION 9 — BOOTSTRAP → PRODUCTION MAPPING
# ============================================================

mapping = pd.DataFrame(
    [
        ("sample_id", "-", "Generated"),
        ("game_index", "game_index", "Direct"),
        ("turn", "turn", "Direct"),
        ("actor", "actor", "Direct"),
        ("observation_vector", "-", "Simulator"),
        ("legal_moves", "-", "Simulator"),
        ("legal_move_mask", "-", "Simulator"),
        ("chosen_move", "move_name", "Direct"),
        ("chosen_move_index", "policy_index", "Direct"),
        ("search_score", "search_score", "Direct"),
        ("search_depth", "search_depth", "Direct"),
        ("nodes_searched", "nodes_searched", "Direct"),
        ("reward", "-", "Simulator"),
        ("done", "-", "Simulator"),
        ("winner", "winner", "Direct"),
        ("source_dataset", "bootstrap_policy", "Generated"),
    ],
    columns=[
        "production_field",
        "bootstrap_source",
        "generation_method",
    ],
)

display(mapping)

summary = (
    mapping.groupby("generation_method")
    .size()
    .reset_index(name="count")
)

print()

print("Generation summary")

display(summary)

assert len(mapping) == 16

print()

print("✅ SECTION 9 FIELD MAPPING PASSED")


# # SECTION 10 — Notebook 36 Generation Pipeline

# ## Section 10 — Production Generation Workflow
# 
# Notebook 36 will generate production-quality policy samples by interacting
# with the simulator before every action selection.
# 
# Workflow:
# 
# 1. Reset simulator.
# 2. Observe current state.
# 3. Collect legal actions.
# 4. Build observation vector.
# 5. Query the search/policy agent.
# 6. Execute chosen action.
# 7. Receive reward.
# 8. Save complete training sample.
# 9. Repeat until terminal state.
# 
# This process produces a Behavior Cloning / PPO-ready dataset.

# In[14]:


# ============================================================
# SECTION 10 — NOTEBOOK 36 WORKFLOW
# ============================================================

workflow = [
    "Reset environment",
    "Observe game state",
    "Extract observation vector",
    "Collect legal actions",
    "Build legal-action mask",
    "Select action",
    "Execute action",
    "Receive reward",
    "Check terminal state",
    "Store training sample",
    "Repeat",
]

workflow_df = pd.DataFrame(
    {
        "Step": range(1, len(workflow) + 1),
        "Operation": workflow,
    }
)

display(workflow_df)

print()

print("Workflow length :", len(workflow_df))

assert len(workflow_df) == 11

print()

print("✅ SECTION 10 WORKFLOW PASSED")


# # SECTION 11 — Notebook 36 Interface Specification

# ## Section 11 — Notebook 36 Interface Specification
# 
# Notebook 36 will consume:
# 
# - Production Observation Pipeline (Notebook 33)
# - Battle Engine
# - Policy/Search Agent
# 
# Notebook 36 will produce:
# 
# - PPO training dataset
# - Behavior Cloning dataset
# - Observation vectors
# - Legal action masks
# - Training metadata
# 
# This interface specification allows Notebook 36 to remain modular and independent
# from downstream training notebooks.

# In[16]:


# ============================================================
# SECTION 11 — NOTEBOOK 36 INTERFACE
# ============================================================

inputs = [
    "Observation Pipeline (Notebook 33)",
    "Battle Engine",
    "Policy Agent",
    "Search Agent",
]

outputs = [
    "Observation vectors",
    "Legal move lists",
    "Legal move masks",
    "Chosen actions",
    "Rewards",
    "Terminal flags",
    "Training dataset",
]

interface_df = pd.DataFrame(
    {
        "Inputs": pd.Series(inputs),
        "Outputs": pd.Series(outputs),
    }
)

display(interface_df)

print()

print("Inputs :", len(inputs))
print("Outputs:", len(outputs))

assert len(inputs) == 4
assert len(outputs) == 7

print()
print("✅ SECTION 11 INTERFACE SPECIFICATION PASSED")


# # SECTION 12 — EXPORT DESIGN DOCUMENTS

# ## Section 12 — Export Design Documents
# 
# Export the production training specification.
# 
# Notebook 36 will load these documents directly instead of
# hardcoding field names.
# 
# Exports:
# 
# • production_dataset_contract.csv
# • production_field_mapping.csv
# • notebook35_workflow.csv
# • notebook35_interface.json

# In[18]:


# ============================================================
# SECTION 12 — EXPORT DESIGN DOCUMENTS
# ============================================================

REPORT_DIR = NOTEBOOK35_REPORT_DIR

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONTRACT_FILE = (
    REPORT_DIR
    / "production_dataset_contract.csv"
)

MAPPING_FILE = (
    REPORT_DIR
    / "production_field_mapping.csv"
)

WORKFLOW_FILE = (
    REPORT_DIR
    / "production_workflow.csv"
)

INTERFACE_FILE = (
    REPORT_DIR
    / "production_interface.json"
)

# ------------------------------------------------------------
# Export contract
# ------------------------------------------------------------

production_contract.to_csv(
    CONTRACT_FILE,
    index=False,
)

# ------------------------------------------------------------
# Export field mapping
# ------------------------------------------------------------

mapping.to_csv(
    MAPPING_FILE,
    index=False,
)

# ------------------------------------------------------------
# Export workflow
# ------------------------------------------------------------

workflow_df.to_csv(
    WORKFLOW_FILE,
    index=False,
)

# ------------------------------------------------------------
# Export interface
# ------------------------------------------------------------

interface_json = {
    "inputs": inputs,
    "outputs": outputs,
}

with INTERFACE_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        interface_json,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("NOTEBOOK 35 — DESIGN DOCUMENT EXPORT")
print("=" * 70)

print(
    "Contract               :",
    CONTRACT_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Field mapping          :",
    MAPPING_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Workflow               :",
    WORKFLOW_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Interface              :",
    INTERFACE_FILE.relative_to(
        PROJECT_ROOT
    ),
)

assert CONTRACT_FILE.exists()
assert MAPPING_FILE.exists()
assert WORKFLOW_FILE.exists()
assert INTERFACE_FILE.exists()

print()
print(
    "✅ SECTION 12 EXPORT PASSED"
)


# # SECTION 13 — VALIDATE EXPORTS

# ## Section 13 — Validate Exported Design Documents
# 
# Reload every exported design document and verify:
# 
# - contract reloads
# - mapping reloads
# - workflow reloads
# - interface reloads
# 
# This guarantees Notebook 36 can immediately use the exported specifications.

# In[19]:


# ============================================================
# SECTION 13 — VALIDATE EXPORTS
# ============================================================

contract_reload = pd.read_csv(CONTRACT_FILE)

mapping_reload = pd.read_csv(MAPPING_FILE)

workflow_reload = pd.read_csv(WORKFLOW_FILE)

with INTERFACE_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    interface_reload = json.load(file)

print("NOTEBOOK 35 — EXPORT VALIDATION")
print("=" * 70)

print(
    "Contract rows      :",
    len(contract_reload),
)

print(
    "Mapping rows       :",
    len(mapping_reload),
)

print(
    "Workflow steps     :",
    len(workflow_reload),
)

print(
    "Interface inputs   :",
    len(interface_reload["inputs"]),
)

print(
    "Interface outputs  :",
    len(interface_reload["outputs"]),
)

display(contract_reload.head())

assert len(contract_reload) == len(production_contract)
assert len(mapping_reload) == len(mapping)
assert len(workflow_reload) == len(workflow_df)

assert "inputs" in interface_reload
assert "outputs" in interface_reload

print()
print("✅ NOTEBOOK 35 COMPLETE")


# In[ ]:




