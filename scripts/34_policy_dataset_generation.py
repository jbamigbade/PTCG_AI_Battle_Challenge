#!/usr/bin/env python
# coding: utf-8

# # Notebook 34 - Policy Dataset Generation
# 
# ## Objectives
# 
# - Reuse the production observation pipeline from Notebook 33
# - Generate one training row per legal move
# - Record selected moves and policy labels
# - Build a reproducible policy-learning dataset
# - Validate schema, labels, and numerical features
# - Export CSV and Parquet datasets for later model training

# ## 1. Project Setup
# 
# This section:
# 
# - locates the project root
# - imports the production battle and observation modules
# - configures reproducibility
# - creates the Notebook 34 report directory
# - validates required Notebook 33 outputs

# In[1]:


from __future__ import annotations

import importlib
import json
import math
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
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK34_REPORT_DIR = (
    REPORTS_DIR / "notebook34"
)
NOTEBOOK33_REPORT_DIR = (
    REPORTS_DIR / "notebook33"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK34_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 34

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
    120,
)

print("NOTEBOOK 34 — PROJECT SETUP")
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
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert NOTEBOOK34_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# ## Section 2 — Discover Notebook 33 Observation Outputs
# 
# This section inspects the artifacts created by Notebook 33 and identifies the observation dataset that will be converted into policy-learning samples.

# In[2]:


# ============================================================
# SECTION 2 — DISCOVER NOTEBOOK 33 OUTPUTS
# ============================================================

print("NOTEBOOK 34 — DISCOVERING NOTEBOOK 33 OUTPUTS")
print("=" * 70)

if not NOTEBOOK33_REPORT_DIR.exists():
    raise FileNotFoundError(
        "Notebook 33 report directory does not exist:\n"
        f"{NOTEBOOK33_REPORT_DIR}"
    )

notebook33_files = sorted(
    path
    for path in NOTEBOOK33_REPORT_DIR.rglob("*")
    if path.is_file()
)

print(
    "Files discovered       :",
    len(notebook33_files),
)

for path in notebook33_files:
    relative_path = path.relative_to(
        PROJECT_ROOT
    )

    size_kb = path.stat().st_size / 1024

    print(
        f"  - {relative_path} "
        f"({size_kb:,.2f} KB)"
    )

supported_suffixes = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".pkl",
    ".pickle",
}

candidate_data_files = [
    path
    for path in notebook33_files
    if path.suffix.lower()
    in supported_suffixes
]

print()
print(
    "Candidate data files   :",
    len(candidate_data_files),
)

for path in candidate_data_files:
    print(
        "  -",
        path.relative_to(
            PROJECT_ROOT
        ),
    )

assert notebook33_files, (
    "Notebook 33 report directory is empty."
)

print()
print(
    "✅ SECTION 2 OUTPUT DISCOVERY PASSED"
)


# # Section 3 — Load Notebook 33 Data

# ## Section 3 — Load Observation Dataset
# 
# #### Load the observation dataset and feature schema generated in Notebook 33.

# In[3]:


# ============================================================
# SECTION 3 — LOAD NOTEBOOK 33 DATA
# ============================================================

SCHEMA_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "feature_schema.csv"
)

MOVE_FEATURE_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "move_features.csv"
)

OBSERVATION_FILE = (
    NOTEBOOK33_REPORT_DIR
    / "sample_observations.csv"
)

assert SCHEMA_FILE.exists()
assert MOVE_FEATURE_FILE.exists()
assert OBSERVATION_FILE.exists()

feature_schema_df = pd.read_csv(
    SCHEMA_FILE
)

move_features_df = pd.read_csv(
    MOVE_FEATURE_FILE
)

observation_df = pd.read_csv(
    OBSERVATION_FILE
)

print("Feature schema rows :", len(feature_schema_df))
print("Move features rows  :", len(move_features_df))
print("Observations rows   :", len(observation_df))

print()
print("✅ SECTION 3 PASSED")


# # Section 4 — Inspect Dataset

# In[4]:


# ============================================================
# SECTION 4 — DATASET PREVIEW
# ============================================================

print("Observation Dataset")

display(observation_df.head())

print()

print("Move Features")

display(move_features_df.head())

print()

print("Feature Schema")

display(feature_schema_df.head())

print()
print("✅ SECTION 4 PASSED")


# # Section 5 — Discover Notebook 32 Self-Play Artifacts

# ## Section 5 — Discover Notebook 32 Self-Play Artifacts
# 
# #### Notebook 32 already generated self-play benchmark data. This section locates those artifacts so Notebook 34 can reuse real game states, legal moves, and selected actions instead of generating placeholder labels.
# 

# In[5]:


# ============================================================
# SECTION 5 — DISCOVER NOTEBOOK 32 SELF-PLAY ARTIFACTS
# ============================================================

NOTEBOOK32_REPORT_DIR = (
    REPORTS_DIR / "notebook32"
)

print("NOTEBOOK 34 — DISCOVERING NOTEBOOK 32 OUTPUTS")
print("=" * 70)

print(
    "Notebook 32 reports    :",
    NOTEBOOK32_REPORT_DIR,
)

if not NOTEBOOK32_REPORT_DIR.exists():
    raise FileNotFoundError(
        "Notebook 32 report directory does not exist:\n"
        f"{NOTEBOOK32_REPORT_DIR}"
    )

notebook32_files = sorted(
    path
    for path in NOTEBOOK32_REPORT_DIR.rglob("*")
    if path.is_file()
)

print(
    "Files discovered       :",
    len(notebook32_files),
)

for path in notebook32_files:
    relative_path = path.relative_to(
        PROJECT_ROOT
    )

    size_kb = path.stat().st_size / 1024

    print(
        f"  - {relative_path} "
        f"({size_kb:,.2f} KB)"
    )

supported_suffixes = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".pkl",
    ".pickle",
}

notebook32_data_files = [
    path
    for path in notebook32_files
    if path.suffix.lower()
    in supported_suffixes
]

self_play_candidates = [
    path
    for path in notebook32_data_files
    if any(
        keyword in path.name.lower()
        for keyword in (
            "self_play",
            "self-play",
            "benchmark",
            "game",
            "episode",
            "trajectory",
            "decision",
            "move",
        )
    )
]

print()
print(
    "Candidate data files   :",
    len(notebook32_data_files),
)

for path in notebook32_data_files:
    print(
        "  -",
        path.relative_to(
            PROJECT_ROOT
        ),
    )

print()
print(
    "Likely self-play files :",
    len(self_play_candidates),
)

for path in self_play_candidates:
    print(
        "  -",
        path.relative_to(
            PROJECT_ROOT
        ),
    )

assert notebook32_files, (
    "Notebook 32 report directory is empty."
)

assert notebook32_data_files, (
    "Notebook 32 contains no supported data files."
)

print()
print(
    "✅ SECTION 5 NOTEBOOK 32 DISCOVERY PASSED"
)


# # Section 6 — Load and Inspect Notebook 32 Benchmark

# ###  Section 6 — Load the Notebook 32 Self-Play Benchmark
# 
# ##### This section loads the real mixed self-play benchmark and inspects its schema, data types, completeness, and sample records before policy labels are derived.

# In[6]:


# ============================================================
# SECTION 6 — LOAD NOTEBOOK 32 MIXED BENCHMARK
# ============================================================

SELF_PLAY_BENCHMARK_FILE = (
    NOTEBOOK32_REPORT_DIR
    / "notebook32_mixed_benchmark.csv"
)

assert SELF_PLAY_BENCHMARK_FILE.exists(), (
    "Missing Notebook 32 mixed benchmark:\n"
    f"{SELF_PLAY_BENCHMARK_FILE}"
)

self_play_df = pd.read_csv(
    SELF_PLAY_BENCHMARK_FILE,
    low_memory=False,
)

print("NOTEBOOK 34 — NOTEBOOK 32 BENCHMARK INSPECTION")
print("=" * 70)

print(
    "Benchmark file         :",
    SELF_PLAY_BENCHMARK_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Rows                   :",
    f"{len(self_play_df):,}",
)

print(
    "Columns                :",
    len(self_play_df.columns),
)

print(
    "Duplicate rows         :",
    f"{self_play_df.duplicated().sum():,}",
)

print(
    "Completely empty rows  :",
    f"{self_play_df.isna().all(axis=1).sum():,}",
)

print()
print("Column names")
print("-" * 70)

for index, column in enumerate(
    self_play_df.columns,
    start=1,
):
    print(
        f"{index:>3}. {column}"
    )

print()
print("Data types")
print("-" * 70)

display(
    self_play_df.dtypes
    .astype(str)
    .rename("dtype")
    .to_frame()
)

print()
print("Sample records")
print("-" * 70)

display(
    self_play_df.head(10)
)

print()
print(
    "✅ SECTION 6 BENCHMARK LOAD PASSED"
)


# # Section 7 — Inspect Missing Values and Candidate Decision Columns

# In[7]:


# ============================================================
# SECTION 7 — IDENTIFY POLICY-RELEVANT COLUMNS
# ============================================================

missing_summary_df = (
    self_play_df
    .isna()
    .sum()
    .rename("missing_count")
    .to_frame()
)

missing_summary_df["missing_percent"] = (
    missing_summary_df["missing_count"]
    / len(self_play_df)
    * 100
)

missing_summary_df = (
    missing_summary_df
    .sort_values(
        [
            "missing_count",
            "missing_percent",
        ],
        ascending=False,
    )
)

decision_keywords = (
    "move",
    "action",
    "choice",
    "chosen",
    "selected",
    "legal",
    "state",
    "observation",
    "turn",
    "player",
    "score",
    "value",
    "winner",
    "result",
)

candidate_decision_columns = [
    column
    for column in self_play_df.columns
    if any(
        keyword in column.lower()
        for keyword in decision_keywords
    )
]

print("NOTEBOOK 34 — POLICY COLUMN DISCOVERY")
print("=" * 70)

print(
    "Candidate columns      :",
    len(candidate_decision_columns),
)

for column in candidate_decision_columns:
    unique_count = (
        self_play_df[column]
        .nunique(
            dropna=True
        )
    )

    missing_count = (
        self_play_df[column]
        .isna()
        .sum()
    )

    print(
        f"  - {column}"
        f" | unique={unique_count:,}"
        f" | missing={missing_count:,}"
    )

print()
print("Missing-value summary")
print("-" * 70)

display(
    missing_summary_df.head(30)
)

print()
print("Candidate-column preview")
print("-" * 70)

if candidate_decision_columns:
    display(
        self_play_df[
            candidate_decision_columns
        ].head(15)
    )
else:
    print(
        "No policy-relevant columns were identified "
        "from the current keyword search."
    )

assert len(self_play_df) > 0
assert len(self_play_df.columns) > 0

print()
print(
    "✅ SECTION 7 POLICY COLUMN DISCOVERY PASSED"
)


# # Section 8 — Parse Battle Transcripts

# ## Section 8 — Parse Self-Play Battle Transcripts
# 
# #### Extract individual battle events from the Notebook 32 transcripts. These parsed events will later be aligned with the observation features generated in Notebook 33 to build the policy-learning dataset.

# In[8]:


# ============================================================
# SECTION 8 — PARSE BATTLE TRANSCRIPTS
# ============================================================

import re

def split_transcript(transcript: str):

    if pd.isna(transcript):
        return []

    lines = [
        line.strip()
        for line in transcript.splitlines()
        if line.strip()
    ]

    return lines


parsed_games = []

for _, row in self_play_df.iterrows():

    events = split_transcript(
        row["transcript"]
    )

    parsed_games.append(
        {
            "game_index": row["game_index"],
            "opponent": row["opponent"],
            "winner": row["winner"],
            "turns": row["turns"],
            "final_score": row["final_score"],
            "events": events,
            "num_events": len(events),
        }
    )

parsed_df = pd.DataFrame(parsed_games)

display(parsed_df.head())

print()
print(
    "Average events/game:",
    round(
        parsed_df["num_events"].mean(),
        2,
    ),
)

print()

print(
    "✅ SECTION 8 PASSED"
)


# # Section 9 — Expand Into Event-Level Dataset

# In[9]:


# ============================================================
# SECTION 9 — EXPAND EVENTS
# ============================================================

event_rows = []

for game in parsed_games:

    for event_number, text in enumerate(
        game["events"],
        start=1,
    ):

        event_rows.append(
            {
                "game_index": game["game_index"],
                "event_number": event_number,
                "event_text": text,
                "winner": game["winner"],
                "opponent": game["opponent"],
                "final_score": game["final_score"],
            }
        )

events_df = pd.DataFrame(event_rows)

print("Total events:", len(events_df))

display(events_df.head(20))

print()

print(
    "✅ SECTION 9 PASSED"
)


# # Section 10 — Extract Decision Events

# ## Section 10 — Extract Search Decisions
# 
# #### Filter the transcript to retain only events corresponding to AI decisions and search metrics. These events become the foundation of the supervised policy-learning dataset.

# In[10]:


# ============================================================
# SECTION 10 — EXTRACT DECISION EVENTS
# ============================================================

decision_keywords = (
    "used",
    "search score",
    "search depth",
    "nodes searched",
    "hp after move",
    "next side",
)

decision_df = events_df[
    events_df["event_text"]
    .str.lower()
    .str.contains(
        "|".join(decision_keywords),
        regex=True,
        na=False,
    )
].copy()

decision_df.reset_index(
    drop=True,
    inplace=True,
)

print("Decision events :", len(decision_df))

display(decision_df.head(25))

print()
print("✅ SECTION 10 PASSED")


# # Section 11 — Parse Search Metrics

# In[11]:


# ============================================================
# SECTION 11 — PARSE SEARCH METRICS
# ============================================================

import re

decision_df["search_score"] = (
    decision_df["event_text"]
    .str.extract(
        r"Search score:\s*([-+]?\d*\.?\d+)",
        expand=False,
    )
)

decision_df["search_depth"] = (
    decision_df["event_text"]
    .str.extract(
        r"Search depth:\s*(\d+)",
        expand=False,
    )
)

decision_df["nodes_searched"] = (
    decision_df["event_text"]
    .str.extract(
        r"Nodes searched:\s*(\d+)",
        expand=False,
    )
)

decision_df["move_name"] = (
    decision_df["event_text"]
    .str.extract(
        r"used\s+(.*)",
        expand=False,
    )
)

display(
    decision_df.head(30)
)

print()

print("Search scores parsed:",
      decision_df["search_score"].notna().sum())

print("Moves parsed:",
      decision_df["move_name"].notna().sum())

print()

print("✅ SECTION 11 PASSED")


# # Section 12 — Consolidate Decision Records

# ## Section 12 — Consolidate Decision Records
# 
# #### Merge consecutive transcript events into a single structured decision record. Each record represents one search decision and becomes one training example for the policy dataset.

# In[12]:


# ============================================================
# SECTION 12 — BUILD DECISION RECORDS
# ============================================================

import re

decision_records = []

current = None

for _, row in decision_df.iterrows():

    text = row["event_text"]

    # Start of a new move
    if "used" in text.lower():

        if current is not None:
            decision_records.append(current)

        current = {
            "game_index": row["game_index"],
            "winner": row["winner"],
            "opponent": row["opponent"],
            "final_score": row["final_score"],
            "turn": None,
            "move_name": row["move_name"],
            "search_score": None,
            "search_depth": None,
            "nodes_searched": None,
            "player_hp": None,
            "opponent_hp": None,
            "next_side": None,
        }

        turn_match = re.search(r"Turn\s+(\d+)", text)
        if turn_match:
            current["turn"] = int(turn_match.group(1))

        continue

    if current is None:
        continue

    # Search score
    if pd.notna(row["search_score"]):
        current["search_score"] = float(row["search_score"])

    # Search depth
    if pd.notna(row["search_depth"]):
        current["search_depth"] = int(row["search_depth"])

    # Nodes searched
    if pd.notna(row["nodes_searched"]):
        current["nodes_searched"] = int(row["nodes_searched"])

    # HP after move
    if "HP after move" in text:

        hp_match = re.search(
            r"Player:\s*([0-9.]+),\s*Opponent:\s*([0-9.]+)",
            text,
        )

        if hp_match:
            current["player_hp"] = float(hp_match.group(1))
            current["opponent_hp"] = float(hp_match.group(2))

    # Next side
    if "Next side:" in text:
        current["next_side"] = (
            text.split(":")[-1]
            .strip()
        )

# Save last record
if current is not None:
    decision_records.append(current)

policy_df = pd.DataFrame(decision_records)

print("Structured decisions:", len(policy_df))

display(policy_df.head(20))

print()
print("✅ SECTION 12 PASSED")


# # Section 13 — Add Actor and Policy Labels

# ## Section 13 — Add Actor and Policy Labels
# 
# #### Extract the acting side from each move event and encode the selected move as a reproducible policy target. Policy indices are generated from sorted move names rather than assigned manually.

# In[13]:


# ============================================================
# SECTION 13 — ADD ACTOR AND POLICY LABELS
# RESTART-SAFE VERSION
# ============================================================

# Rebuild the clean Section 12 DataFrame so rerunning this
# cell cannot create actor_x, actor_y, or duplicate columns.
policy_df = pd.DataFrame(
    decision_records
).copy()

move_event_pattern = re.compile(
    r"Turn\s+(?P<turn>\d+):\s*"
    r"(?P<actor>Player|Opponent)\s*[—-]\s*"
    r"(?P<pokemon>.+?)\s+used\s+"
    r"(?P<move>.+)$",
    flags=re.IGNORECASE,
)


def parse_move_event(
    text: str,
) -> dict[str, Any]:
    match = move_event_pattern.search(
        str(text).strip()
    )

    if match is None:
        return {
            "turn": None,
            "actor": None,
            "pokemon_name": None,
            "parsed_move_name": None,
        }

    return {
        "turn": int(
            match.group("turn")
        ),
        "actor": match.group(
            "actor"
        ).title(),
        "pokemon_name": match.group(
            "pokemon"
        ).strip(),
        "parsed_move_name": match.group(
            "move"
        ).strip(),
    }


# Keep only transcript rows that contain an actual move.
move_event_rows = (
    decision_df.loc[
        decision_df["move_name"].notna(),
        [
            "game_index",
            "event_text",
        ],
    ]
    .copy()
    .reset_index(drop=True)
)

parsed_move_metadata = (
    move_event_rows["event_text"]
    .apply(parse_move_event)
    .apply(pd.Series)
)

move_event_rows = pd.concat(
    [
        move_event_rows,
        parsed_move_metadata,
    ],
    axis=1,
)

# Validate parsing before merging.
assert move_event_rows["turn"].notna().all(), (
    "One or more turn numbers could not be parsed."
)

assert move_event_rows["actor"].notna().all(), (
    "One or more actors could not be parsed."
)

assert move_event_rows["pokemon_name"].notna().all(), (
    "One or more Pokémon names could not be parsed."
)

# Confirm one move record per game and turn.
duplicate_move_keys = (
    move_event_rows.duplicated(
        subset=[
            "game_index",
            "turn",
        ]
    )
    .sum()
)

assert duplicate_move_keys == 0, (
    "Duplicate move rows were found for the same game and turn."
)

# Merge the transcript metadata with the consolidated decisions.
policy_df = policy_df.merge(
    move_event_rows[
        [
            "game_index",
            "turn",
            "actor",
            "pokemon_name",
            "parsed_move_name",
        ]
    ],
    on=[
        "game_index",
        "turn",
    ],
    how="left",
    validate="one_to_one",
)

# Normalize move names before comparing.
policy_df["move_name"] = (
    policy_df["move_name"]
    .astype(str)
    .str.strip()
)

policy_df["parsed_move_name"] = (
    policy_df["parsed_move_name"]
    .astype(str)
    .str.strip()
)

move_name_matches = (
    policy_df["move_name"]
    .eq(
        policy_df["parsed_move_name"]
    )
)

assert move_name_matches.all(), (
    "Parsed move names do not match the "
    "consolidated decision records."
)

policy_df.drop(
    columns=[
        "parsed_move_name",
    ],
    inplace=True,
)

# Create reproducible policy-label mapping.
policy_classes = sorted(
    policy_df["move_name"]
    .dropna()
    .unique()
    .tolist()
)

policy_label_to_index = {
    move_name: index
    for index, move_name
    in enumerate(policy_classes)
}

policy_index_to_label = {
    index: move_name
    for move_name, index
    in policy_label_to_index.items()
}

policy_df["policy_index"] = (
    policy_df["move_name"]
    .map(policy_label_to_index)
    .astype("Int64")
)

policy_df["is_training_agent_turn"] = (
    policy_df["actor"]
    .eq("Player")
)

print("NOTEBOOK 34 — POLICY LABEL ENCODING")
print("=" * 70)

print(
    "Structured decisions   :",
    f"{len(policy_df):,}",
)

print(
    "Unique policy classes  :",
    len(policy_classes),
)

print()
print("Policy mapping")
print("-" * 70)

for index, move_name in (
    policy_index_to_label.items()
):
    print(
        f"{index:>3} -> {move_name}"
    )

print()
print("Actor distribution")
print("-" * 70)

display(
    policy_df["actor"]
    .value_counts(
        dropna=False
    )
    .rename_axis("actor")
    .reset_index(
        name="decision_count"
    )
)

print()
print("Policy preview")
print("-" * 70)

display(
    policy_df[
        [
            "game_index",
            "turn",
            "actor",
            "pokemon_name",
            "move_name",
            "policy_index",
            "is_training_agent_turn",
        ]
    ].head(20)
)

assert len(policy_df) == 3500
assert policy_df["actor"].notna().all()
assert policy_df["pokemon_name"].notna().all()
assert policy_df["policy_index"].notna().all()

print()
print(
    "✅ SECTION 13 POLICY LABELS PASSED"
)


# # Section 14 — Policy Dataset Quality Audit

# ## Section 14 — Policy Dataset Quality Audit
# 
# #### Evaluate class balance, actor-to-action dependence, duplicate decision patterns, missing values, and whether the current benchmark contains enough action diversity for meaningful policy learning.

# In[14]:


# ============================================================
# SECTION 14 — POLICY DATASET QUALITY AUDIT
# ============================================================

print("NOTEBOOK 34 — POLICY DATASET QUALITY AUDIT")
print("=" * 70)

required_columns = [
    "game_index",
    "turn",
    "actor",
    "pokemon_name",
    "move_name",
    "policy_index",
    "search_score",
    "search_depth",
    "nodes_searched",
    "player_hp",
    "opponent_hp",
    "next_side",
]

missing_required_columns = [
    column
    for column in required_columns
    if column not in policy_df.columns
]

assert not missing_required_columns, (
    "Required policy columns are missing: "
    f"{missing_required_columns}"
)

# ------------------------------------------------------------
# Basic completeness
# ------------------------------------------------------------

quality_summary = pd.DataFrame(
    {
        "metric": [
            "decision_rows",
            "unique_games",
            "unique_moves",
            "unique_actors",
            "missing_cells",
            "duplicate_game_turn_keys",
        ],
        "value": [
            len(policy_df),
            policy_df["game_index"].nunique(),
            policy_df["move_name"].nunique(),
            policy_df["actor"].nunique(),
            int(policy_df[required_columns].isna().sum().sum()),
            int(
                policy_df.duplicated(
                    subset=[
                        "game_index",
                        "turn",
                    ]
                ).sum()
            ),
        ],
    }
)

print("Quality summary")
print("-" * 70)

display(quality_summary)

# ------------------------------------------------------------
# Policy class distribution
# ------------------------------------------------------------

policy_distribution_df = (
    policy_df["move_name"]
    .value_counts(
        dropna=False
    )
    .rename_axis("move_name")
    .reset_index(
        name="decision_count"
    )
)

policy_distribution_df["decision_percent"] = (
    policy_distribution_df["decision_count"]
    / len(policy_df)
    * 100
).round(2)

print()
print("Policy class distribution")
print("-" * 70)

display(policy_distribution_df)

# ------------------------------------------------------------
# Actor and policy dependence
# ------------------------------------------------------------

actor_policy_counts = pd.crosstab(
    policy_df["actor"],
    policy_df["move_name"],
    margins=True,
)

actor_policy_percent = pd.crosstab(
    policy_df["actor"],
    policy_df["move_name"],
    normalize="index",
).mul(100).round(2)

print()
print("Actor × policy counts")
print("-" * 70)

display(actor_policy_counts)

print()
print("Actor × policy row percentages")
print("-" * 70)

display(actor_policy_percent)

# ------------------------------------------------------------
# Pokémon and policy dependence
# ------------------------------------------------------------

pokemon_policy_counts = pd.crosstab(
    policy_df["pokemon_name"],
    policy_df["move_name"],
    margins=True,
)

print()
print("Pokémon × policy counts")
print("-" * 70)

display(pokemon_policy_counts)

# ------------------------------------------------------------
# TrainingAgent-only subset
# ------------------------------------------------------------

training_policy_df = (
    policy_df.loc[
        policy_df["is_training_agent_turn"]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

training_move_count = (
    training_policy_df["move_name"]
    .nunique()
)

print()
print("TrainingAgent subset")
print("-" * 70)

print(
    "Rows                   :",
    f"{len(training_policy_df):,}",
)

print(
    "Unique policy classes  :",
    training_move_count,
)

display(
    training_policy_df["move_name"]
    .value_counts()
    .rename_axis("move_name")
    .reset_index(
        name="decision_count"
    )
)

# ------------------------------------------------------------
# Detect deterministic actor-to-policy mapping
# ------------------------------------------------------------

moves_per_actor = (
    policy_df.groupby(
        "actor"
    )["move_name"]
    .nunique()
)

actor_determines_move = bool(
    moves_per_actor.eq(1).all()
)

moves_per_pokemon = (
    policy_df.groupby(
        "pokemon_name"
    )["move_name"]
    .nunique()
)

pokemon_determines_move = bool(
    moves_per_pokemon.eq(1).all()
)

policy_dataset_trainable = bool(
    training_move_count >= 2
    and not actor_determines_move
    and not pokemon_determines_move
)

print()
print("Dataset fitness assessment")
print("-" * 70)

print(
    "Actor determines move  :",
    actor_determines_move,
)

print(
    "Pokémon determines move:",
    pokemon_determines_move,
)

print(
    "Training-ready policy  :",
    policy_dataset_trainable,
)

if policy_dataset_trainable:
    fitness_status = "TRAINING_READY"
    fitness_message = (
        "The dataset contains sufficient action diversity "
        "for an initial policy-learning experiment."
    )
else:
    fitness_status = "BOOTSTRAP_ONLY"
    fitness_message = (
        "The dataset validates the extraction pipeline, "
        "but it is not suitable for meaningful policy-model "
        "training because the selected move is deterministic "
        "from the actor and Pokémon."
    )

print()
print(
    "Fitness status         :",
    fitness_status,
)

print(
    "Assessment             :",
    fitness_message,
)

assert len(policy_df) == 3500
assert policy_df["game_index"].nunique() == 1000
assert (
    policy_df.duplicated(
        subset=[
            "game_index",
            "turn",
        ]
    ).sum()
    == 0
)

print()
print(
    "✅ SECTION 14 QUALITY AUDIT PASSED"
)


# # Section 15 — Export Policy Dataset and Audit Artifacts

# ## Section 15 — Export Policy Dataset Artifacts
# 
# #### Export the structured bootstrap policy dataset, policy-label mapping, class distributions, and dataset fitness assessment for reproducibility and future model-development notebooks.

# In[15]:


# ============================================================
# SECTION 15 — EXPORT POLICY DATASET ARTIFACTS
# ============================================================

POLICY_DATASET_CSV = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_bootstrap.csv"
)

POLICY_DATASET_PARQUET = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_bootstrap.parquet"
)

POLICY_MAPPING_JSON = (
    NOTEBOOK34_REPORT_DIR
    / "policy_label_mapping.json"
)

POLICY_DISTRIBUTION_CSV = (
    NOTEBOOK34_REPORT_DIR
    / "policy_class_distribution.csv"
)

QUALITY_SUMMARY_CSV = (
    NOTEBOOK34_REPORT_DIR
    / "policy_quality_summary.csv"
)

FITNESS_REPORT_JSON = (
    NOTEBOOK34_REPORT_DIR
    / "policy_dataset_fitness.json"
)

export_columns = [
    "game_index",
    "turn",
    "actor",
    "pokemon_name",
    "move_name",
    "policy_index",
    "is_training_agent_turn",
    "search_score",
    "search_depth",
    "nodes_searched",
    "player_hp",
    "opponent_hp",
    "next_side",
    "winner",
    "opponent",
    "final_score",
]

export_policy_df = (
    policy_df[export_columns]
    .copy()
    .sort_values(
        [
            "game_index",
            "turn",
        ]
    )
    .reset_index(
        drop=True
    )
)

export_policy_df.to_csv(
    POLICY_DATASET_CSV,
    index=False,
)

parquet_exported = True
parquet_error = None

try:
    export_policy_df.to_parquet(
        POLICY_DATASET_PARQUET,
        index=False,
    )
except Exception as exc:
    parquet_exported = False
    parquet_error = str(exc)

with POLICY_MAPPING_JSON.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "label_to_index": (
                policy_label_to_index
            ),
            "index_to_label": {
                str(index): move_name
                for index, move_name
                in policy_index_to_label.items()
            },
        },
        file,
        indent=2,
        ensure_ascii=False,
    )

policy_distribution_df.to_csv(
    POLICY_DISTRIBUTION_CSV,
    index=False,
)

quality_summary.to_csv(
    QUALITY_SUMMARY_CSV,
    index=False,
)

fitness_report = {
    "dataset_name": (
        "policy_dataset_bootstrap"
    ),
    "fitness_status": (
        fitness_status
    ),
    "training_ready": (
        policy_dataset_trainable
    ),
    "decision_rows": int(
        len(policy_df)
    ),
    "unique_games": int(
        policy_df["game_index"].nunique()
    ),
    "unique_policy_classes": int(
        policy_df["move_name"].nunique()
    ),
    "training_agent_rows": int(
        len(training_policy_df)
    ),
    "training_agent_policy_classes": int(
        training_move_count
    ),
    "actor_determines_move": (
        actor_determines_move
    ),
    "pokemon_determines_move": (
        pokemon_determines_move
    ),
    "assessment": (
        fitness_message
    ),
    "parquet_exported": (
        parquet_exported
    ),
    "parquet_error": (
        parquet_error
    ),
}

with FITNESS_REPORT_JSON.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        fitness_report,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("NOTEBOOK 34 — EXPORT RESULTS")
print("=" * 70)

print(
    "CSV dataset            :",
    POLICY_DATASET_CSV.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "CSV rows               :",
    f"{len(export_policy_df):,}",
)

print(
    "Parquet exported       :",
    parquet_exported,
)

if parquet_exported:
    print(
        "Parquet dataset        :",
        POLICY_DATASET_PARQUET.relative_to(
            PROJECT_ROOT
        ),
    )
else:
    print(
        "Parquet warning        :",
        parquet_error,
    )

print(
    "Policy mapping         :",
    POLICY_MAPPING_JSON.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Class distribution     :",
    POLICY_DISTRIBUTION_CSV.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Quality summary        :",
    QUALITY_SUMMARY_CSV.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Fitness report         :",
    FITNESS_REPORT_JSON.relative_to(
        PROJECT_ROOT
    ),
)

assert POLICY_DATASET_CSV.exists()
assert POLICY_MAPPING_JSON.exists()
assert POLICY_DISTRIBUTION_CSV.exists()
assert QUALITY_SUMMARY_CSV.exists()
assert FITNESS_REPORT_JSON.exists()

print()
print(
    "✅ SECTION 15 EXPORT PASSED"
)


# # Section 16 — Validate Exported Dataset

# In[16]:


# ============================================================
# SECTION 16 — VALIDATE EXPORTED ARTIFACTS
# ============================================================

reloaded_policy_df = pd.read_csv(
    POLICY_DATASET_CSV
)

with POLICY_MAPPING_JSON.open(
    "r",
    encoding="utf-8",
) as file:
    reloaded_policy_mapping = json.load(
        file
    )

with FITNESS_REPORT_JSON.open(
    "r",
    encoding="utf-8",
) as file:
    reloaded_fitness_report = json.load(
        file
    )

print("NOTEBOOK 34 — EXPORT VALIDATION")
print("=" * 70)

print(
    "Reloaded rows          :",
    f"{len(reloaded_policy_df):,}",
)

print(
    "Reloaded columns       :",
    len(reloaded_policy_df.columns),
)

print(
    "Fitness status         :",
    reloaded_fitness_report[
        "fitness_status"
    ],
)

print(
    "Training ready         :",
    reloaded_fitness_report[
        "training_ready"
    ],
)

display(
    reloaded_policy_df.head(10)
)

assert (
    len(reloaded_policy_df)
    == len(export_policy_df)
)

assert (
    set(reloaded_policy_df.columns)
    == set(export_policy_df.columns)
)

assert (
    reloaded_policy_df[
        "policy_index"
    ]
    .notna()
    .all()
)

assert (
    reloaded_fitness_report[
        "fitness_status"
    ]
    == "BOOTSTRAP_ONLY"
)

assert (
    reloaded_policy_mapping[
        "label_to_index"
    ]
    == policy_label_to_index
)

print()
print(
    "✅ SECTION 16 EXPORT VALIDATION PASSED"
)


# In[17]:


import pyarrow

print(pyarrow.__version__)


# In[18]:


import pandas as pd

df = pd.DataFrame(
    {
        "A": [1, 2, 3]
    }
)

df.to_parquet("test.parquet")

print("✅ Parquet works!")


# In[ ]:




