#!/usr/bin/env python
# coding: utf-8

# # Notebook 49:  Side-Balanced Search-Guided Self-Play
# 
# ## Objective
# 
# #### Improve second-player performance and reduce the large side-performance gap identified in Notebook 48.
# 
# ### Primary targets
# 
# - Raise Opponent-side performance above 50%
# - Improve side-neutral score toward 70%
# - Reduce estimated first-player Elo advantage
# - Maintain zero tournament failures
# - Improve performance against the Depth-6 Search baseline
# 

# ## Section 1 — Setup and Baseline Validation
# 
# #### Load and validate the competitive-policy checkpoint, actor-relative dataset, production search engine, Notebook 48 tournament reports, and side-bias measurements.
# #### This section establishes deterministic configuration for side-balanced search-guided self-play.

# In[1]:


# ============================================================
# NOTEBOOK 49
# SECTION 1 — SETUP AND ARTIFACT VALIDATION
# ============================================================

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 1.3 — Locate project root
# ------------------------------------------------------------

def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current = (
        start_path
        or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        required_paths = [
            candidate / "src",
            candidate / "notebooks",
            candidate / "models",
            candidate / "reports",
            candidate / "data",
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

SELF_PLAY_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "self_play"
)

COMPETITIVE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "competitive"
)

NOTEBOOK49_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook49"
)

for directory in [
    SELF_PLAY_DATA_DIR,
    COMPETITIVE_MODEL_DIR,
    NOTEBOOK49_REPORT_DIR,
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
# 1.4 — Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 4901

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

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
# 1.5 — Required input artifacts
# ------------------------------------------------------------

BASE_POLICY_CHECKPOINT = (
    COMPETITIVE_MODEL_DIR
    / "competitive_policy_best.pt"
)

ACTOR_RELATIVE_DATASET = (
    COMPETITIVE_DATA_DIR
    / "actor_relative_search_expert_dataset.parquet"
)

POLICY_ENGINE_FILE = (
    SRC_DIR
    / "competitive"
    / "competitive_policy_engine.py"
)

POLICY_AGENT_FILE = (
    SRC_DIR
    / "competitive"
    / "competitive_policy_agent.py"
)

SEARCH_ENGINE_FILE = (
    SRC_DIR
    / "engine"
    / "advanced_search.py"
)

ADAPTERS_FILE = (
    SRC_DIR
    / "adapters.py"
)

SIMULATOR_FILE = (
    SRC_DIR
    / "simulator.py"
)

BATTLE_STATE_FILE = (
    SRC_DIR
    / "battle_state.py"
)

LEGAL_MOVES_FILE = (
    SRC_DIR
    / "legal_moves.py"
)

NOTEBOOK48_STRENGTH_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "notebook48"
    / "final_competitive_strength_report.json"
)

NOTEBOOK48_ELO_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "notebook48"
    / "competitive_policy_elo_summary.json"
)

NOTEBOOK48_SIDE_BIAS_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "notebook48"
    / "competitive_policy_side_bias.csv"
)

NOTEBOOK48_TOURNAMENT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "tournaments"
    / "competitive_policy_baseline_tournament.csv"
)


# ------------------------------------------------------------
# 1.6 — Output artifacts
# ------------------------------------------------------------

SETUP_REPORT_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "side_balanced_setup_summary.json"
)

SIDE_BALANCED_DATASET_FILE = (
    SELF_PLAY_DATA_DIR
    / "side_balanced_search_guided_dataset.parquet"
)

SIDE_BALANCED_JSONL_FILE = (
    SELF_PLAY_DATA_DIR
    / "side_balanced_search_guided_dataset.jsonl"
)

UPGRADED_CHECKPOINT_FILE = (
    COMPETITIVE_MODEL_DIR
    / "competitive_policy_side_balanced.pt"
)


# ------------------------------------------------------------
# 1.7 — Configuration
# ------------------------------------------------------------

TARGET_PLAYER_RECORDS = 2000

TARGET_OPPONENT_RECORDS = 2000

TARGET_TOTAL_RECORDS = (
    TARGET_PLAYER_RECORDS
    + TARGET_OPPONENT_RECORDS
)

MINIMUM_MULTI_ACTION_FRACTION = 0.75

SEARCH_DEPTH = 6

MAX_SELF_PLAY_TURNS = 30

SELF_PLAY_BATCH_SIZE = 128

FINE_TUNING_EPOCHS = 40

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

STATE_DIM = 8

MOVE_DIM = 8


# ------------------------------------------------------------
# 1.8 — Load JSON helper
# ------------------------------------------------------------

def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    if not file_path.exists():
        return {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ------------------------------------------------------------
# 1.9 — Artifact inventory
# ------------------------------------------------------------

artifact_rows = [
    {
        "artifact":
            "Base policy checkpoint",

        "path":
            str(
                BASE_POLICY_CHECKPOINT
            ),

        "exists":
            BASE_POLICY_CHECKPOINT.exists(),
    },

    {
        "artifact":
            "Actor-relative expert dataset",

        "path":
            str(
                ACTOR_RELATIVE_DATASET
            ),

        "exists":
            ACTOR_RELATIVE_DATASET.exists(),
    },

    {
        "artifact":
            "Competitive policy engine",

        "path":
            str(
                POLICY_ENGINE_FILE
            ),

        "exists":
            POLICY_ENGINE_FILE.exists(),
    },

    {
        "artifact":
            "Competitive policy agent",

        "path":
            str(
                POLICY_AGENT_FILE
            ),

        "exists":
            POLICY_AGENT_FILE.exists(),
    },

    {
        "artifact":
            "Advanced search engine",

        "path":
            str(
                SEARCH_ENGINE_FILE
            ),

        "exists":
            SEARCH_ENGINE_FILE.exists(),
    },

    {
        "artifact":
            "Search adapters",

        "path":
            str(
                ADAPTERS_FILE
            ),

        "exists":
            ADAPTERS_FILE.exists(),
    },

    {
        "artifact":
            "Simulator",

        "path":
            str(
                SIMULATOR_FILE
            ),

        "exists":
            SIMULATOR_FILE.exists(),
    },

    {
        "artifact":
            "Battle-state module",

        "path":
            str(
                BATTLE_STATE_FILE
            ),

        "exists":
            BATTLE_STATE_FILE.exists(),
    },

    {
        "artifact":
            "Legal-move module",

        "path":
            str(
                LEGAL_MOVES_FILE
            ),

        "exists":
            LEGAL_MOVES_FILE.exists(),
    },

    {
        "artifact":
            "Notebook 48 strength report",

        "path":
            str(
                NOTEBOOK48_STRENGTH_REPORT
            ),

        "exists":
            NOTEBOOK48_STRENGTH_REPORT.exists(),
    },

    {
        "artifact":
            "Notebook 48 Elo report",

        "path":
            str(
                NOTEBOOK48_ELO_REPORT
            ),

        "exists":
            NOTEBOOK48_ELO_REPORT.exists(),
    },

    {
        "artifact":
            "Notebook 48 side-bias report",

        "path":
            str(
                NOTEBOOK48_SIDE_BIAS_REPORT
            ),

        "exists":
            NOTEBOOK48_SIDE_BIAS_REPORT.exists(),
    },

    {
        "artifact":
            "Notebook 48 tournament results",

        "path":
            str(
                NOTEBOOK48_TOURNAMENT_RESULTS
            ),

        "exists":
            NOTEBOOK48_TOURNAMENT_RESULTS.exists(),
    },
]

artifact_df = pd.DataFrame(
    artifact_rows
)


# ------------------------------------------------------------
# 1.10 — Validate required artifacts
# ------------------------------------------------------------

assert artifact_df[
    "exists"
].all(), (
    "One or more required Notebook 49 "
    "artifacts are missing."
)

base_checkpoint = torch.load(
    BASE_POLICY_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

checkpoint_configuration = (
    base_checkpoint.get(
        "configuration",
        {},
    )
)

assert checkpoint_configuration.get(
    "actor_relative"
) is True

assert checkpoint_configuration.get(
    "variable_action"
) is True

assert int(
    checkpoint_configuration.get(
        "state_dim",
        0,
    )
) == STATE_DIM

assert int(
    checkpoint_configuration.get(
        "move_dim",
        0,
    )
) == MOVE_DIM


# ------------------------------------------------------------
# 1.11 — Load Notebook 48 results
# ------------------------------------------------------------

strength_report = load_json_file(
    NOTEBOOK48_STRENGTH_REPORT
)

elo_report = load_json_file(
    NOTEBOOK48_ELO_REPORT
)

side_bias_df = pd.read_csv(
    NOTEBOOK48_SIDE_BIAS_REPORT
)

notebook48_tournament_df = pd.read_csv(
    NOTEBOOK48_TOURNAMENT_RESULTS
)


baseline_player_score = float(
    strength_report[
        "side_performance"
    ][
        "player_score"
    ]
)

baseline_opponent_score = float(
    strength_report[
        "side_performance"
    ][
        "opponent_score"
    ]
)

baseline_side_neutral_score = float(
    strength_report[
        "side_performance"
    ][
        "side_neutral_score"
    ]
)

baseline_side_advantage_elo = float(
    strength_report[
        "elo"
    ][
        "estimated_side_advantage_elo"
    ]
)

baseline_side_neutral_elo = float(
    strength_report[
        "elo"
    ][
        "side_neutral_elo_difference"
    ]
)

baseline_search_win_rate = float(
    strength_report[
        "tournament"
    ][
        "depth6_search_win_rate"
    ]
)


# ------------------------------------------------------------
# 1.12 — Load and inspect original dataset
# ------------------------------------------------------------

original_dataset_df = pd.read_parquet(
    ACTOR_RELATIVE_DATASET
)

assert len(
    original_dataset_df
) > 0

original_side_counts = (
    original_dataset_df[
        "actor_side"
    ]
    .value_counts()
    .to_dict()
)

original_multi_action_fraction = float(
    (
        original_dataset_df[
            "legal_move_count"
        ]
        >= 2
    ).mean()
)


# ------------------------------------------------------------
# 1.13 — Setup report
# ------------------------------------------------------------

setup_report = {
    "project_root":
        str(PROJECT_ROOT),

    "device":
        str(DEVICE),

    "random_seed":
        RANDOM_SEED,

    "base_checkpoint":
        str(
            BASE_POLICY_CHECKPOINT
        ),

    "base_checkpoint_epoch":
        int(
            base_checkpoint.get(
                "epoch",
                0,
            )
        ),

    "original_dataset":
        str(
            ACTOR_RELATIVE_DATASET
        ),

    "original_dataset_records":
        int(
            len(
                original_dataset_df
            )
        ),

    "original_side_counts":
        {
            str(key):
                int(value)
            for key, value in (
                original_side_counts.items()
            )
        },

    "original_multi_action_fraction":
        original_multi_action_fraction,

    "baseline_player_score":
        baseline_player_score,

    "baseline_opponent_score":
        baseline_opponent_score,

    "baseline_side_neutral_score":
        baseline_side_neutral_score,

    "baseline_side_advantage_elo":
        baseline_side_advantage_elo,

    "baseline_side_neutral_elo":
        baseline_side_neutral_elo,

    "baseline_search_win_rate":
        baseline_search_win_rate,

    "target_player_records":
        TARGET_PLAYER_RECORDS,

    "target_opponent_records":
        TARGET_OPPONENT_RECORDS,

    "target_total_records":
        TARGET_TOTAL_RECORDS,

    "minimum_multi_action_fraction":
        MINIMUM_MULTI_ACTION_FRACTION,

    "search_depth":
        SEARCH_DEPTH,

    "max_self_play_turns":
        MAX_SELF_PLAY_TURNS,

    "fine_tuning_epochs":
        FINE_TUNING_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "upgraded_checkpoint":
        str(
            UPGRADED_CHECKPOINT_FILE
        ),

    "all_required_artifacts_exist":
        True,
}

with open(
    SETUP_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        setup_report,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 1.14 — Display setup
# ------------------------------------------------------------

print(
    "NOTEBOOK 49 — SIDE-BALANCED SELF-PLAY SETUP"
)

print("=" * 105)

print()
print(
    "Project root              :",
    PROJECT_ROOT,
)

print(
    "Device                    :",
    DEVICE,
)

print(
    "Random seed               :",
    RANDOM_SEED,
)

print()
print(
    "Base checkpoint epoch     :",
    base_checkpoint.get(
        "epoch"
    ),
)

print(
    "Original dataset records  :",
    f"{len(original_dataset_df):,}",
)

print(
    "Original Player records   :",
    original_side_counts.get(
        "Player",
        0,
    ),
)

print(
    "Original Opponent records :",
    original_side_counts.get(
        "Opponent",
        0,
    ),
)

print(
    "Original multi-action rate:",
    round(
        original_multi_action_fraction,
        4,
    ),
)

print()
print(
    "Notebook 48 Player score  :",
    round(
        baseline_player_score,
        4,
    ),
)

print(
    "Notebook 48 Opponent score:",
    round(
        baseline_opponent_score,
        4,
    ),
)

print(
    "Side-neutral score        :",
    round(
        baseline_side_neutral_score,
        4,
    ),
)

print(
    "Side-neutral Elo          :",
    round(
        baseline_side_neutral_elo,
        2,
    ),
)

print(
    "Estimated side advantage  :",
    round(
        baseline_side_advantage_elo,
        2,
    ),
    "Elo",
)

print(
    "Depth-6 search win rate   :",
    round(
        baseline_search_win_rate,
        4,
    ),
)

print()
print(
    "Target Player records     :",
    f"{TARGET_PLAYER_RECORDS:,}",
)

print(
    "Target Opponent records   :",
    f"{TARGET_OPPONENT_RECORDS:,}",
)

print(
    "Target total records      :",
    f"{TARGET_TOTAL_RECORDS:,}",
)

print(
    "Search depth              :",
    SEARCH_DEPTH,
)

print(
    "Minimum multi-action rate :",
    MINIMUM_MULTI_ACTION_FRACTION,
)

print()
print("REQUIRED ARTIFACTS")
print("-" * 105)

display(
    artifact_df
)

print()
print(
    "Setup report:",
    SETUP_REPORT_FILE,
)


# ------------------------------------------------------------
# 1.15 — Final validation
# ------------------------------------------------------------

assert SETUP_REPORT_FILE.exists()

assert TARGET_PLAYER_RECORDS == (
    TARGET_OPPONENT_RECORDS
)

assert TARGET_TOTAL_RECORDS == (
    TARGET_PLAYER_RECORDS
    + TARGET_OPPONENT_RECORDS
)

assert baseline_opponent_score < (
    baseline_player_score
)

assert baseline_side_advantage_elo > 0

assert (
    0.0
    <= original_multi_action_fraction
    <= 1.0
)

print()
print(
    "✅ SECTION 1 SIDE-BALANCED SETUP PASSED"
)


# ## Section 2 — Opponent-Side Difficulty Analysis
# 
# #### This section analyzes the actor-relative expert dataset from Notebook 46 to identify structural differences between Player-side and Opponent-side training examples.
# 
# #### Rather than assuming the cause of the side-performance gap, we measure differences in:
# 
# - legal move counts
# - search confidence
# - move entropy
# - tactical complexity
# - action diversity
# 
# #### These measurements guide the targeted self-play strategy used later in this notebook.

# In[2]:


# ============================================================
# SECTION 2
# OPPONENT-SIDE DIFFICULTY ANALYSIS
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

print("=" * 100)
print("SECTION 2 — OPPONENT-SIDE DIFFICULTY ANALYSIS")
print("=" * 100)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

dataset = pd.read_parquet(
    ACTOR_RELATIVE_DATASET
)

print()
print(f"Total records : {len(dataset):,}")

# ------------------------------------------------------------
# Identify actor column
# ------------------------------------------------------------

possible_side_columns = [
    "actor_side",
    "side",
    "player_side",
]

side_column = None

for c in possible_side_columns:
    if c in dataset.columns:
        side_column = c
        break

assert side_column is not None, "Could not find actor-side column."

print("Side column :", side_column)

# ------------------------------------------------------------
# Split dataset
# ------------------------------------------------------------

player_df = dataset[
    dataset[side_column] == "Player"
]

opponent_df = dataset[
    dataset[side_column] == "Opponent"
]

print()
print(f"Player records   : {len(player_df):,}")
print(f"Opponent records : {len(opponent_df):,}")

# ------------------------------------------------------------
# Numeric feature comparison
# ------------------------------------------------------------

candidate_columns = [
    "legal_move_count",
    "search_score",
    "best_move_score",
    "selected_move_score",
    "evaluation",
    "value",
    "policy_confidence",
]

available = [
    c
    for c in candidate_columns
    if c in dataset.columns
]

summary_rows = []

for column in available:

    summary_rows.append({

        "feature": column,

        "player_mean":
            float(player_df[column].mean()),

        "opponent_mean":
            float(opponent_df[column].mean()),

        "difference":
            float(
                player_df[column].mean()
                -
                opponent_df[column].mean()
            ),

        "player_std":
            float(player_df[column].std()),

        "opponent_std":
            float(opponent_df[column].std())
    })

summary = pd.DataFrame(summary_rows)

print()
print("NUMERIC FEATURE COMPARISON")
display(summary)

# ------------------------------------------------------------
# Multi-action comparison
# ------------------------------------------------------------

if "legal_move_count" in dataset.columns:

    player_multi = (
        player_df["legal_move_count"] >= 2
    ).mean()

    opponent_multi = (
        opponent_df["legal_move_count"] >= 2
    ).mean()

    print()
    print("MULTI-ACTION RATE")
    print("-----------------------------")
    print(
        "Player   :",
        round(player_multi,4)
    )
    print(
        "Opponent :",
        round(opponent_multi,4)
    )

# ------------------------------------------------------------
# Action diversity
# ------------------------------------------------------------

candidate_action_columns = [
    "selected_move",
    "best_move",
    "move_name",
]

action_column = None

for c in candidate_action_columns:
    if c in dataset.columns:
        action_column = c
        break

if action_column:

    print()
    print("ACTION DIVERSITY")
    print("-----------------------------")

    print(
        "Player unique actions :",
        player_df[action_column].nunique()
    )

    print(
        "Opponent unique actions :",
        opponent_df[action_column].nunique()
    )

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print()
print("Missing values")

missing = dataset.isna().sum()

missing = missing[
    missing > 0
]

display(missing)

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

analysis_report = {

    "records":
        int(len(dataset)),

    "player_records":
        int(len(player_df)),

    "opponent_records":
        int(len(opponent_df)),

    "available_numeric_features":
        available,

    "summary":
        summary.to_dict(
            orient="records"
        )
}

report_file = (
    NOTEBOOK49_REPORT_DIR
    /
    "section2_difficulty_analysis.json"
)

with open(
    report_file,
    "w",
    encoding="utf8",
) as f:

    json.dump(
        analysis_report,
        f,
        indent=4
    )

print()
print("Saved:", report_file)

print()
print("✅ SECTION 2 PASSED")


# ## Section 3 — Tournament Loss Analysis
# 
# #### This section analyzes the complete Notebook 48 tournament results to identify patterns associated with losses.
# 
# #### Rather than generating new self-play data indiscriminately, we identify the specific situations where the Competitive Policy underperformed.
# 
# #### These loss patterns will become the primary source of additional training data later in this notebook.

# In[3]:


# ============================================================
# SECTION 3
# TOURNAMENT LOSS ANALYSIS
# ============================================================

from __future__ import annotations

import json

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 3 — TOURNAMENT LOSS ANALYSIS")
print("=" * 100)


# ------------------------------------------------------------
# 3.1 — Load Notebook 48 tournament results
# ------------------------------------------------------------

tournament = pd.read_csv(
    NOTEBOOK48_TOURNAMENT_RESULTS
)

print()
print("Rows :", len(tournament))

print()
print("Columns")
print("-" * 30)

for column in tournament.columns:
    print(column)


# ------------------------------------------------------------
# 3.2 — Validate required columns
# ------------------------------------------------------------

required_columns = [
    "match_id",
    "baseline_name",
    "starting_side",
    "agent_result",
    "competitive_win",
    "competitive_loss",
    "turns",
    "fallback_count",
    "elapsed_seconds",
    "evaluated_agent_side",
]

missing_required_columns = [
    column
    for column in required_columns
    if column not in tournament.columns
]

assert not missing_required_columns, (
    "Missing required tournament columns: "
    f"{missing_required_columns}"
)


# ------------------------------------------------------------
# 3.3 — Normalize Boolean result columns
# ------------------------------------------------------------

def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common Boolean-like values into True/False.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "win",
        "won",
    }

    return normalized.isin(
        true_values
    )


competitive_win_mask = normalize_boolean_series(
    tournament["competitive_win"]
)

competitive_loss_mask = normalize_boolean_series(
    tournament["competitive_loss"]
)


# ------------------------------------------------------------
# 3.4 — Build win, loss, and draw subsets
# ------------------------------------------------------------

wins = tournament[
    competitive_win_mask
].copy()

losses = tournament[
    competitive_loss_mask
].copy()

if "draw" in tournament.columns:
    draw_mask = normalize_boolean_series(
        tournament["draw"]
    )
else:
    draw_mask = (
        ~competitive_win_mask
        & ~competitive_loss_mask
    )

draws = tournament[
    draw_mask
].copy()


# ------------------------------------------------------------
# 3.5 — Validate tournament accounting
# ------------------------------------------------------------

total_accounted = (
    len(wins)
    + len(losses)
    + len(draws)
)

assert total_accounted == len(tournament), (
    "Tournament result accounting failed: "
    f"{total_accounted} results accounted for "
    f"out of {len(tournament)} rows."
)

assert len(
    set(wins.index)
    & set(losses.index)
) == 0, (
    "Some matches are marked as both wins and losses."
)


print()
print("TOURNAMENT RESULT SUMMARY")
print("-" * 40)

print("Games :", len(tournament))
print("Wins  :", len(wins))
print("Losses:", len(losses))
print("Draws :", len(draws))

overall_win_rate = (
    len(wins)
    / len(tournament)
    if len(tournament) > 0
    else 0.0
)

print(
    "Overall competitive win rate:",
    round(
        overall_win_rate,
        4,
    ),
)


# ------------------------------------------------------------
# 3.6 — Identify side and opponent columns
# ------------------------------------------------------------

side_column = "evaluated_agent_side"

if (
    tournament[side_column]
    .isna()
    .all()
):
    side_column = "starting_side"

opponent_column = "baseline_name"

turn_column = "turns"


print()
print("Detected side column     :", side_column)
print("Detected opponent column :", opponent_column)
print("Detected turn column     :", turn_column)


# ------------------------------------------------------------
# 3.7 — Normalize side labels
# ------------------------------------------------------------

def normalize_side_label(
    value: object,
) -> str:
    text = str(value).strip().lower()

    player_labels = {
        "player",
        "first",
        "player_side",
        "competitive_player",
        "agent_player",
    }

    opponent_labels = {
        "opponent",
        "second",
        "opponent_side",
        "competitive_opponent",
        "agent_opponent",
    }

    if text in player_labels:
        return "Player"

    if text in opponent_labels:
        return "Opponent"

    return str(value)


tournament["normalized_side"] = (
    tournament[side_column]
    .map(
        normalize_side_label
    )
)

wins["normalized_side"] = (
    wins[side_column]
    .map(
        normalize_side_label
    )
)

losses["normalized_side"] = (
    losses[side_column]
    .map(
        normalize_side_label
    )
)

draws["normalized_side"] = (
    draws[side_column]
    .map(
        normalize_side_label
    )
)


# ------------------------------------------------------------
# 3.8 — Results by side
# ------------------------------------------------------------

side_result_table = pd.crosstab(
    tournament["normalized_side"],
    tournament["agent_result"],
)

print()
print("RESULTS BY SIDE")
print("-" * 40)

display(
    side_result_table
)


side_summary_rows = []

for side_name, side_df in tournament.groupby(
    "normalized_side"
):
    side_wins = int(
        normalize_boolean_series(
            side_df["competitive_win"]
        ).sum()
    )

    side_losses = int(
        normalize_boolean_series(
            side_df["competitive_loss"]
        ).sum()
    )

    if "draw" in side_df.columns:
        side_draws = int(
            normalize_boolean_series(
                side_df["draw"]
            ).sum()
        )
    else:
        side_draws = int(
            len(side_df)
            - side_wins
            - side_losses
        )

    side_games = int(
        len(side_df)
    )

    side_score = (
        (
            side_wins
            + 0.5 * side_draws
        )
        / side_games
        if side_games > 0
        else 0.0
    )

    side_summary_rows.append(
        {
            "side":
                side_name,

            "games":
                side_games,

            "wins":
                side_wins,

            "losses":
                side_losses,

            "draws":
                side_draws,

            "score":
                side_score,

            "average_turns":
                float(
                    side_df[
                        turn_column
                    ].mean()
                ),

            "average_elapsed_seconds":
                float(
                    side_df[
                        "elapsed_seconds"
                    ].mean()
                ),

            "fallbacks":
                int(
                    side_df[
                        "fallback_count"
                    ].sum()
                ),
        }
    )

side_summary_df = pd.DataFrame(
    side_summary_rows
).sort_values(
    "side"
).reset_index(
    drop=True
)

print()
print("SIDE PERFORMANCE SUMMARY")
print("-" * 40)

display(
    side_summary_df
)


# ------------------------------------------------------------
# 3.9 — Losses by side
# ------------------------------------------------------------

losses_by_side = (
    losses[
        "normalized_side"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "side"
    )
    .reset_index(
        name="losses"
    )
)

print()
print("LOSSES BY SIDE")
print("-" * 40)

display(
    losses_by_side
)


# ------------------------------------------------------------
# 3.10 — Results by baseline
# ------------------------------------------------------------

baseline_summary_rows = []

for baseline_name, baseline_df in tournament.groupby(
    opponent_column
):
    baseline_wins = int(
        normalize_boolean_series(
            baseline_df["competitive_win"]
        ).sum()
    )

    baseline_losses = int(
        normalize_boolean_series(
            baseline_df["competitive_loss"]
        ).sum()
    )

    if "draw" in baseline_df.columns:
        baseline_draws = int(
            normalize_boolean_series(
                baseline_df["draw"]
            ).sum()
        )
    else:
        baseline_draws = int(
            len(baseline_df)
            - baseline_wins
            - baseline_losses
        )

    baseline_games = int(
        len(baseline_df)
    )

    baseline_score = (
        (
            baseline_wins
            + 0.5 * baseline_draws
        )
        / baseline_games
        if baseline_games > 0
        else 0.0
    )

    baseline_summary_rows.append(
        {
            "baseline":
                baseline_name,

            "games":
                baseline_games,

            "wins":
                baseline_wins,

            "losses":
                baseline_losses,

            "draws":
                baseline_draws,

            "score":
                baseline_score,

            "average_turns":
                float(
                    baseline_df[
                        turn_column
                    ].mean()
                ),

            "average_elapsed_seconds":
                float(
                    baseline_df[
                        "elapsed_seconds"
                    ].mean()
                ),

            "fallbacks":
                int(
                    baseline_df[
                        "fallback_count"
                    ].sum()
                ),
        }
    )

baseline_summary_df = pd.DataFrame(
    baseline_summary_rows
).sort_values(
    "score"
).reset_index(
    drop=True
)

print()
print("RESULTS BY BASELINE")
print("-" * 40)

display(
    baseline_summary_df
)


# ------------------------------------------------------------
# 3.11 — Losses by baseline
# ------------------------------------------------------------

losses_by_baseline = (
    losses[
        opponent_column
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "baseline"
    )
    .reset_index(
        name="losses"
    )
)

print()
print("LOSSES BY BASELINE")
print("-" * 40)

display(
    losses_by_baseline
)


# ------------------------------------------------------------
# 3.12 — Side-by-baseline performance matrix
# ------------------------------------------------------------

side_baseline_rows = []

for (
    side_name,
    baseline_name,
), group_df in tournament.groupby(
    [
        "normalized_side",
        opponent_column,
    ]
):
    group_wins = int(
        normalize_boolean_series(
            group_df["competitive_win"]
        ).sum()
    )

    group_losses = int(
        normalize_boolean_series(
            group_df["competitive_loss"]
        ).sum()
    )

    if "draw" in group_df.columns:
        group_draws = int(
            normalize_boolean_series(
                group_df["draw"]
            ).sum()
        )
    else:
        group_draws = int(
            len(group_df)
            - group_wins
            - group_losses
        )

    group_games = int(
        len(group_df)
    )

    group_score = (
        (
            group_wins
            + 0.5 * group_draws
        )
        / group_games
        if group_games > 0
        else 0.0
    )

    side_baseline_rows.append(
        {
            "side":
                side_name,

            "baseline":
                baseline_name,

            "games":
                group_games,

            "wins":
                group_wins,

            "losses":
                group_losses,

            "draws":
                group_draws,

            "score":
                group_score,

            "average_turns":
                float(
                    group_df[
                        turn_column
                    ].mean()
                ),
        }
    )

side_baseline_df = pd.DataFrame(
    side_baseline_rows
).sort_values(
    [
        "side",
        "score",
    ]
).reset_index(
    drop=True
)

print()
print("SIDE-BY-BASELINE PERFORMANCE")
print("-" * 40)

display(
    side_baseline_df
)


# ------------------------------------------------------------
# 3.13 — Turn statistics
# ------------------------------------------------------------

turn_statistics = {
    "average_winning_turns":
        float(
            wins[
                turn_column
            ].mean()
        )
        if len(wins) > 0
        else None,

    "median_winning_turns":
        float(
            wins[
                turn_column
            ].median()
        )
        if len(wins) > 0
        else None,

    "average_losing_turns":
        float(
            losses[
                turn_column
            ].mean()
        )
        if len(losses) > 0
        else None,

    "median_losing_turns":
        float(
            losses[
                turn_column
            ].median()
        )
        if len(losses) > 0
        else None,

    "minimum_losing_turns":
        int(
            losses[
                turn_column
            ].min()
        )
        if len(losses) > 0
        else None,

    "maximum_losing_turns":
        int(
            losses[
                turn_column
            ].max()
        )
        if len(losses) > 0
        else None,
}

print()
print("TURN STATISTICS")
print("-" * 40)

for key, value in turn_statistics.items():
    print(
        f"{key:28s}:",
        value,
    )


# ------------------------------------------------------------
# 3.14 — Stop-reason analysis
# ------------------------------------------------------------

stop_reason_summary = (
    tournament[
        "stop_reason"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "stop_reason"
    )
    .reset_index(
        name="games"
    )
)

loss_stop_reason_summary = (
    losses[
        "stop_reason"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "stop_reason"
    )
    .reset_index(
        name="losses"
    )
)

print()
print("ALL STOP REASONS")
print("-" * 40)

display(
    stop_reason_summary
)

print()
print("LOSS STOP REASONS")
print("-" * 40)

display(
    loss_stop_reason_summary
)


# ------------------------------------------------------------
# 3.15 — Runtime and fallback analysis
# ------------------------------------------------------------

runtime_summary = {
    "average_elapsed_seconds":
        float(
            tournament[
                "elapsed_seconds"
            ].mean()
        ),

    "maximum_elapsed_seconds":
        float(
            tournament[
                "elapsed_seconds"
            ].max()
        ),

    "total_fallbacks":
        int(
            tournament[
                "fallback_count"
            ].sum()
        ),

    "loss_fallbacks":
        int(
            losses[
                "fallback_count"
            ].sum()
        ),
}

print()
print("RUNTIME AND STABILITY")
print("-" * 40)

for key, value in runtime_summary.items():
    print(
        f"{key:28s}:",
        value,
    )


# ------------------------------------------------------------
# 3.16 — Identify high-priority loss groups
# ------------------------------------------------------------

priority_loss_groups = (
    losses
    .groupby(
        [
            "normalized_side",
            opponent_column,
        ],
        dropna=False,
    )
    .agg(
        losses=(
            "match_id",
            "count",
        ),

        average_turns=(
            turn_column,
            "mean",
        ),

        average_elapsed_seconds=(
            "elapsed_seconds",
            "mean",
        ),

        total_fallbacks=(
            "fallback_count",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "losses",
            "average_turns",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

print()
print("HIGH-PRIORITY LOSS GROUPS")
print("-" * 40)

display(
    priority_loss_groups
)


# ------------------------------------------------------------
# 3.17 — Save detailed CSV reports
# ------------------------------------------------------------

side_summary_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_side_performance.csv"
)

baseline_summary_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_baseline_performance.csv"
)

side_baseline_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_side_by_baseline_performance.csv"
)

priority_loss_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_priority_loss_groups.csv"
)

loss_matches_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_loss_matches.csv"
)

side_summary_df.to_csv(
    side_summary_file,
    index=False,
)

baseline_summary_df.to_csv(
    baseline_summary_file,
    index=False,
)

side_baseline_df.to_csv(
    side_baseline_file,
    index=False,
)

priority_loss_groups.to_csv(
    priority_loss_file,
    index=False,
)

losses.to_csv(
    loss_matches_file,
    index=False,
)


# ------------------------------------------------------------
# 3.18 — Save JSON summary
# ------------------------------------------------------------

analysis_summary = {
    "games":
        int(
            len(tournament)
        ),

    "wins":
        int(
            len(wins)
        ),

    "losses":
        int(
            len(losses)
        ),

    "draws":
        int(
            len(draws)
        ),

    "overall_win_rate":
        float(
            overall_win_rate
        ),

    "side_summary":
        side_summary_df.to_dict(
            orient="records"
        ),

    "baseline_summary":
        baseline_summary_df.to_dict(
            orient="records"
        ),

    "side_by_baseline_summary":
        side_baseline_df.to_dict(
            orient="records"
        ),

    "turn_statistics":
        turn_statistics,

    "runtime_summary":
        runtime_summary,

    "priority_loss_groups":
        priority_loss_groups.to_dict(
            orient="records"
        ),
}

analysis_report_file = (
    NOTEBOOK49_REPORT_DIR
    / "section3_loss_analysis.json"
)

with open(
    analysis_report_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 3.19 — Final validation
# ------------------------------------------------------------

assert len(
    tournament
) == 300, (
    "Expected 300 Notebook 48 tournament matches."
)

assert len(
    wins
) == 194, (
    "Expected 194 competitive-policy wins."
)

assert len(
    losses
) == 106, (
    "Expected 106 competitive-policy losses."
)

assert int(
    tournament[
        "fallback_count"
    ].sum()
) == 0, (
    "Notebook 48 tournament contained fallbacks."
)

assert analysis_report_file.exists()

assert side_summary_file.exists()

assert baseline_summary_file.exists()

assert side_baseline_file.exists()

assert priority_loss_file.exists()

assert loss_matches_file.exists()


print()
print("SAVED REPORTS")
print("-" * 40)

print(
    analysis_report_file
)

print(
    side_summary_file
)

print(
    baseline_summary_file
)

print(
    side_baseline_file
)

print(
    priority_loss_file
)

print(
    loss_matches_file
)

print()
print(
    "✅ SECTION 3 TOURNAMENT LOSS ANALYSIS PASSED"
)


# ## Section 4 — Targeted Recovery Curriculum
# 
# #### This section converts the Notebook 48 tournament-loss analysis into a targeted training curriculum.
# 
# #### Instead of generating generic self-play positions, the curriculum prioritizes:
# 
# - Opponent-side situations
# - losses against stronger baselines
# - longer and more complex games
# - side–baseline combinations with the lowest score
# - recovery positions associated with repeated tournament failures
# 
# #### The resulting curriculum becomes the sampling policy for search-guided self-play generation.

# In[4]:


# ============================================================
# SECTION 4
# TARGETED RECOVERY CURRICULUM
# ============================================================

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 4 — TARGETED RECOVERY CURRICULUM")
print("=" * 100)


# ------------------------------------------------------------
# 4.1 — Input report paths
# ------------------------------------------------------------

SIDE_PERFORMANCE_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_side_performance.csv"
)

BASELINE_PERFORMANCE_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_baseline_performance.csv"
)

SIDE_BASELINE_PERFORMANCE_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_side_by_baseline_performance.csv"
)

PRIORITY_LOSS_GROUPS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_priority_loss_groups.csv"
)

LOSS_MATCHES_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_loss_matches.csv"
)


required_report_files = [
    SIDE_PERFORMANCE_FILE,
    BASELINE_PERFORMANCE_FILE,
    SIDE_BASELINE_PERFORMANCE_FILE,
    PRIORITY_LOSS_GROUPS_FILE,
    LOSS_MATCHES_FILE,
]

missing_report_files = [
    str(path)
    for path in required_report_files
    if not path.exists()
]

assert not missing_report_files, (
    "Missing Section 3 reports: "
    f"{missing_report_files}"
)


# ------------------------------------------------------------
# 4.2 — Load Section 3 reports
# ------------------------------------------------------------

side_performance_df = pd.read_csv(
    SIDE_PERFORMANCE_FILE
)

baseline_performance_df = pd.read_csv(
    BASELINE_PERFORMANCE_FILE
)

side_baseline_df = pd.read_csv(
    SIDE_BASELINE_PERFORMANCE_FILE
)

priority_loss_groups_df = pd.read_csv(
    PRIORITY_LOSS_GROUPS_FILE
)

loss_matches_df = pd.read_csv(
    LOSS_MATCHES_FILE
)


print()
print("Loaded Section 3 reports")

print(
    "Side rows                :",
    len(side_performance_df),
)

print(
    "Baseline rows            :",
    len(baseline_performance_df),
)

print(
    "Side-baseline rows       :",
    len(side_baseline_df),
)

print(
    "Priority loss groups     :",
    len(priority_loss_groups_df),
)

print(
    "Individual loss matches  :",
    len(loss_matches_df),
)


# ------------------------------------------------------------
# 4.3 — Validate expected columns
# ------------------------------------------------------------

required_side_baseline_columns = [
    "side",
    "baseline",
    "games",
    "wins",
    "losses",
    "draws",
    "score",
    "average_turns",
]

missing_columns = [
    column
    for column in required_side_baseline_columns
    if column not in side_baseline_df.columns
]

assert not missing_columns, (
    "Missing side-baseline columns: "
    f"{missing_columns}"
)

assert len(
    loss_matches_df
) == 106, (
    "Expected 106 Notebook 48 loss matches."
)


# ------------------------------------------------------------
# 4.4 — Normalize side labels
# ------------------------------------------------------------

def normalize_side(
    value: object,
) -> str:
    text = str(value).strip().lower()

    if text in {
        "player",
        "first",
        "agent_player",
    }:
        return "Player"

    if text in {
        "opponent",
        "second",
        "agent_opponent",
    }:
        return "Opponent"

    return str(value)


side_baseline_df["side"] = (
    side_baseline_df["side"]
    .map(
        normalize_side
    )
)

priority_loss_groups_df["normalized_side"] = (
    priority_loss_groups_df[
        "normalized_side"
    ]
    .map(
        normalize_side
    )
)


# ------------------------------------------------------------
# 4.5 — Baseline-strength weights
# ------------------------------------------------------------

def baseline_strength_weight(
    baseline_name: object,
) -> float:
    text = str(
        baseline_name
    ).strip().lower()

    if (
        "depth" in text
        or "search" in text
    ):
        return 1.50

    if "greedy" in text:
        return 1.20

    if "random" in text:
        return 0.75

    return 1.00


side_baseline_df[
    "baseline_strength_weight"
] = (
    side_baseline_df[
        "baseline"
    ]
    .map(
        baseline_strength_weight
    )
)


# ------------------------------------------------------------
# 4.6 — Side-priority weights
# ------------------------------------------------------------

side_baseline_df[
    "side_priority_weight"
] = np.where(
    side_baseline_df[
        "side"
    ].eq(
        "Opponent"
    ),
    1.75,
    1.00,
)


# ------------------------------------------------------------
# 4.7 — Weakness weights
# ------------------------------------------------------------

side_baseline_df[
    "weakness_weight"
] = (
    1.0
    - side_baseline_df[
        "score"
    ].clip(
        lower=0.0,
        upper=1.0,
    )
)

side_baseline_df[
    "weakness_weight"
] = (
    side_baseline_df[
        "weakness_weight"
    ]
    + 0.25
)


# ------------------------------------------------------------
# 4.8 — Loss-volume weights
# ------------------------------------------------------------

maximum_losses = max(
    int(
        side_baseline_df[
            "losses"
        ].max()
    ),
    1,
)

side_baseline_df[
    "loss_volume_weight"
] = (
    side_baseline_df[
        "losses"
    ]
    / maximum_losses
)

side_baseline_df[
    "loss_volume_weight"
] = (
    side_baseline_df[
        "loss_volume_weight"
    ]
    + 0.25
)


# ------------------------------------------------------------
# 4.9 — Game-length complexity weights
# ------------------------------------------------------------

median_turns = float(
    side_baseline_df[
        "average_turns"
    ].median()
)

if (
    not np.isfinite(
        median_turns
    )
    or median_turns <= 0
):
    median_turns = 1.0

side_baseline_df[
    "game_length_weight"
] = (
    side_baseline_df[
        "average_turns"
    ]
    / median_turns
)

side_baseline_df[
    "game_length_weight"
] = (
    side_baseline_df[
        "game_length_weight"
    ]
    .clip(
        lower=0.75,
        upper=1.50,
    )
)


# ------------------------------------------------------------
# 4.10 — Combined raw priority
# ------------------------------------------------------------

side_baseline_df[
    "raw_priority"
] = (
    side_baseline_df[
        "baseline_strength_weight"
    ]
    * side_baseline_df[
        "side_priority_weight"
    ]
    * side_baseline_df[
        "weakness_weight"
    ]
    * side_baseline_df[
        "loss_volume_weight"
    ]
    * side_baseline_df[
        "game_length_weight"
    ]
)

raw_priority_sum = float(
    side_baseline_df[
        "raw_priority"
    ].sum()
)

assert raw_priority_sum > 0


side_baseline_df[
    "sampling_probability"
] = (
    side_baseline_df[
        "raw_priority"
    ]
    / raw_priority_sum
)


# ------------------------------------------------------------
# 4.11 — Curriculum generation target
# ------------------------------------------------------------

CURRICULUM_TOTAL_RECORDS = 4000

MINIMUM_GROUP_RECORDS = 100

side_baseline_df[
    "initial_target_records"
] = np.rint(
    side_baseline_df[
        "sampling_probability"
    ]
    * CURRICULUM_TOTAL_RECORDS
).astype(
    int
)

side_baseline_df[
    "target_records"
] = (
    side_baseline_df[
        "initial_target_records"
    ]
    .clip(
        lower=MINIMUM_GROUP_RECORDS
    )
)


# ------------------------------------------------------------
# 4.12 — Correct rounding and minimum-allocation drift
# ------------------------------------------------------------

allocation_difference = (
    CURRICULUM_TOTAL_RECORDS
    - int(
        side_baseline_df[
            "target_records"
        ].sum()
    )
)

if allocation_difference != 0:

    adjustment_order = (
        side_baseline_df
        .sort_values(
            "raw_priority",
            ascending=(
                allocation_difference < 0
            ),
        )
        .index
        .tolist()
    )

    adjustment_sign = (
        1
        if allocation_difference > 0
        else -1
    )

    remaining_adjustment = abs(
        allocation_difference
    )

    cursor = 0

    while remaining_adjustment > 0:

        row_index = adjustment_order[
            cursor
            % len(
                adjustment_order
            )
        ]

        current_target = int(
            side_baseline_df.loc[
                row_index,
                "target_records",
            ]
        )

        if (
            adjustment_sign > 0
            or current_target
            > MINIMUM_GROUP_RECORDS
        ):
            side_baseline_df.loc[
                row_index,
                "target_records",
            ] = (
                current_target
                + adjustment_sign
            )

            remaining_adjustment -= 1

        cursor += 1


assert int(
    side_baseline_df[
        "target_records"
    ].sum()
) == CURRICULUM_TOTAL_RECORDS


# ------------------------------------------------------------
# 4.13 — Curriculum tier assignment
# ------------------------------------------------------------

side_baseline_df[
    "priority_rank"
] = (
    side_baseline_df[
        "raw_priority"
    ]
    .rank(
        method="dense",
        ascending=False,
    )
    .astype(
        int
    )
)


def assign_curriculum_tier(
    row: pd.Series,
) -> str:

    side = str(
        row["side"]
    )

    baseline = str(
        row["baseline"]
    ).lower()

    score = float(
        row["score"]
    )

    if (
        side == "Opponent"
        and (
            "depth" in baseline
            or "search" in baseline
        )
    ):
        return "Tier A — Opponent vs Search"

    if (
        side == "Opponent"
        and score < 0.50
    ):
        return "Tier A — Opponent Recovery"

    if (
        side == "Opponent"
    ):
        return "Tier B — Opponent Reinforcement"

    if (
        "depth" in baseline
        or "search" in baseline
    ):
        return "Tier B — Player vs Search"

    return "Tier C — General Maintenance"


side_baseline_df[
    "curriculum_tier"
] = side_baseline_df.apply(
    assign_curriculum_tier,
    axis=1,
)


# ------------------------------------------------------------
# 4.14 — Desired position composition
# ------------------------------------------------------------

side_baseline_df[
    "multi_action_target_fraction"
] = np.where(
    side_baseline_df[
        "curriculum_tier"
    ].str.startswith(
        "Tier A"
    ),
    0.85,
    np.where(
        side_baseline_df[
            "curriculum_tier"
        ].str.startswith(
            "Tier B"
        ),
        0.78,
        0.65,
    ),
)

side_baseline_df[
    "policy_disagreement_target_fraction"
] = np.where(
    side_baseline_df[
        "curriculum_tier"
    ].str.startswith(
        "Tier A"
    ),
    0.60,
    np.where(
        side_baseline_df[
            "curriculum_tier"
        ].str.startswith(
            "Tier B"
        ),
        0.40,
        0.20,
    ),
)

side_baseline_df[
    "late_game_target_fraction"
] = np.where(
    side_baseline_df[
        "curriculum_tier"
    ].str.startswith(
        "Tier A"
    ),
    0.35,
    0.20,
)


# ------------------------------------------------------------
# 4.15 — Sort final curriculum plan
# ------------------------------------------------------------

curriculum_plan_df = (
    side_baseline_df
    .sort_values(
        [
            "priority_rank",
            "side",
            "baseline",
        ]
    )
    .reset_index(
        drop=True
    )
)


print()
print("TARGETED CURRICULUM PLAN")
print("-" * 100)

display_columns = [
    "priority_rank",
    "curriculum_tier",
    "side",
    "baseline",
    "games",
    "wins",
    "losses",
    "score",
    "raw_priority",
    "sampling_probability",
    "target_records",
    "multi_action_target_fraction",
    "policy_disagreement_target_fraction",
    "late_game_target_fraction",
]

display(
    curriculum_plan_df[
        display_columns
    ]
)


# ------------------------------------------------------------
# 4.16 — Side-level allocation
# ------------------------------------------------------------

side_allocation_df = (
    curriculum_plan_df
    .groupby(
        "side",
        as_index=False,
    )
    .agg(
        target_records=(
            "target_records",
            "sum",
        ),

        average_score=(
            "score",
            "mean",
        ),

        total_losses=(
            "losses",
            "sum",
        ),
    )
)

side_allocation_df[
    "allocation_fraction"
] = (
    side_allocation_df[
        "target_records"
    ]
    / CURRICULUM_TOTAL_RECORDS
)


print()
print("CURRICULUM ALLOCATION BY SIDE")
print("-" * 60)

display(
    side_allocation_df
)


# ------------------------------------------------------------
# 4.17 — Baseline-level allocation
# ------------------------------------------------------------

baseline_allocation_df = (
    curriculum_plan_df
    .groupby(
        "baseline",
        as_index=False,
    )
    .agg(
        target_records=(
            "target_records",
            "sum",
        ),

        average_score=(
            "score",
            "mean",
        ),

        total_losses=(
            "losses",
            "sum",
        ),
    )
)

baseline_allocation_df[
    "allocation_fraction"
] = (
    baseline_allocation_df[
        "target_records"
    ]
    / CURRICULUM_TOTAL_RECORDS
)


print()
print("CURRICULUM ALLOCATION BY BASELINE")
print("-" * 60)

display(
    baseline_allocation_df
)


# ------------------------------------------------------------
# 4.18 — Curriculum quality checks
# ------------------------------------------------------------

opponent_target_records = int(
    side_allocation_df.loc[
        side_allocation_df[
            "side"
        ].eq(
            "Opponent"
        ),
        "target_records",
    ].sum()
)

player_target_records = int(
    side_allocation_df.loc[
        side_allocation_df[
            "side"
        ].eq(
            "Player"
        ),
        "target_records",
    ].sum()
)

opponent_allocation_fraction = (
    opponent_target_records
    / CURRICULUM_TOTAL_RECORDS
)

search_target_records = int(
    curriculum_plan_df.loc[
        curriculum_plan_df[
            "baseline"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "search|depth",
            regex=True,
        ),
        "target_records",
    ].sum()
)

search_allocation_fraction = (
    search_target_records
    / CURRICULUM_TOTAL_RECORDS
)


print()
print("CURRICULUM QUALITY CHECKS")
print("-" * 60)

print(
    "Total records              :",
    int(
        curriculum_plan_df[
            "target_records"
        ].sum()
    ),
)

print(
    "Opponent target records    :",
    opponent_target_records,
)

print(
    "Player target records      :",
    player_target_records,
)

print(
    "Opponent allocation        :",
    round(
        opponent_allocation_fraction,
        4,
    ),
)

print(
    "Search-focused records     :",
    search_target_records,
)

print(
    "Search allocation          :",
    round(
        search_allocation_fraction,
        4,
    ),
)


# ------------------------------------------------------------
# 4.19 — Save curriculum reports
# ------------------------------------------------------------

CURRICULUM_PLAN_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4_targeted_curriculum_plan.csv"
)

CURRICULUM_SIDE_ALLOCATION_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4_side_allocation.csv"
)

CURRICULUM_BASELINE_ALLOCATION_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4_baseline_allocation.csv"
)

CURRICULUM_SUMMARY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4_targeted_curriculum_summary.json"
)

curriculum_plan_df.to_csv(
    CURRICULUM_PLAN_FILE,
    index=False,
)

side_allocation_df.to_csv(
    CURRICULUM_SIDE_ALLOCATION_FILE,
    index=False,
)

baseline_allocation_df.to_csv(
    CURRICULUM_BASELINE_ALLOCATION_FILE,
    index=False,
)


curriculum_summary = {
    "curriculum_total_records":
        CURRICULUM_TOTAL_RECORDS,

    "minimum_group_records":
        MINIMUM_GROUP_RECORDS,

    "opponent_target_records":
        opponent_target_records,

    "player_target_records":
        player_target_records,

    "opponent_allocation_fraction":
        opponent_allocation_fraction,

    "search_target_records":
        search_target_records,

    "search_allocation_fraction":
        search_allocation_fraction,

    "curriculum_plan":
        curriculum_plan_df.to_dict(
            orient="records"
        ),

    "side_allocation":
        side_allocation_df.to_dict(
            orient="records"
        ),

    "baseline_allocation":
        baseline_allocation_df.to_dict(
            orient="records"
        ),
}

with open(
    CURRICULUM_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        curriculum_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 4.20 — Final validation
# ------------------------------------------------------------

assert CURRICULUM_PLAN_FILE.exists()

assert CURRICULUM_SIDE_ALLOCATION_FILE.exists()

assert CURRICULUM_BASELINE_ALLOCATION_FILE.exists()

assert CURRICULUM_SUMMARY_FILE.exists()

assert int(
    curriculum_plan_df[
        "target_records"
    ].sum()
) == CURRICULUM_TOTAL_RECORDS

assert opponent_target_records > (
    player_target_records
), (
    "The targeted curriculum should prioritize "
    "Opponent-side records."
)

assert opponent_allocation_fraction >= 0.55, (
    "Opponent-side allocation should be at least 55%."
)

assert search_allocation_fraction >= 0.30, (
    "At least 30% of the curriculum should target "
    "the Search baseline."
)

assert (
    curriculum_plan_df[
        "target_records"
    ]
    >= MINIMUM_GROUP_RECORDS
).all()


print()
print("SAVED REPORTS")
print("-" * 60)

print(
    CURRICULUM_PLAN_FILE
)

print(
    CURRICULUM_SIDE_ALLOCATION_FILE
)

print(
    CURRICULUM_BASELINE_ALLOCATION_FILE
)

print(
    CURRICULUM_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 4 TARGETED RECOVERY CURRICULUM PASSED"
)


# ### Section 4B — Curriculum Rebalancing Guardrail
# 
# #### The raw weakness-weighted curriculum allocated more than 90% of records to the Opponent side. Although this reflects the observed loss concentration, such an extreme distribution could cause catastrophic forgetting of the agent's strong Player-side policy.
# 
# #### This guardrail preserves targeted learning while capping the final curriculum at 70% Opponent-side and 30% Player-side records.

# In[5]:


# ============================================================
# SECTION 4B
# CURRICULUM REBALANCING GUARDRAIL
# ============================================================

from __future__ import annotations

import json

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 4B — CURRICULUM REBALANCING GUARDRAIL")
print("=" * 100)


# ------------------------------------------------------------
# 4B.1 — Final side allocation targets
# ------------------------------------------------------------

FINAL_CURRICULUM_TOTAL_RECORDS = 4000

FINAL_OPPONENT_FRACTION = 0.70

FINAL_PLAYER_FRACTION = (
    1.0
    - FINAL_OPPONENT_FRACTION
)

FINAL_OPPONENT_RECORDS = int(
    FINAL_CURRICULUM_TOTAL_RECORDS
    * FINAL_OPPONENT_FRACTION
)

FINAL_PLAYER_RECORDS = (
    FINAL_CURRICULUM_TOTAL_RECORDS
    - FINAL_OPPONENT_RECORDS
)


print()
print(
    "Final curriculum total :",
    FINAL_CURRICULUM_TOTAL_RECORDS,
)

print(
    "Opponent target        :",
    FINAL_OPPONENT_RECORDS,
)

print(
    "Player target          :",
    FINAL_PLAYER_RECORDS,
)


# ------------------------------------------------------------
# 4B.2 — Allocate records within each side
# ------------------------------------------------------------

def allocate_side_records(
    side_df: pd.DataFrame,
    side_target: int,
    minimum_per_group: int = 100,
) -> pd.DataFrame:
    """
    Allocate a fixed number of records across one side's
    baseline groups while preserving relative raw priority.
    """

    result = side_df.copy()

    group_count = len(
        result
    )

    minimum_total = (
        group_count
        * minimum_per_group
    )

    assert side_target >= minimum_total, (
        f"Side target {side_target} is below the "
        f"minimum required allocation {minimum_total}."
    )

    remaining_records = (
        side_target
        - minimum_total
    )

    priority_values = (
        result[
            "raw_priority"
        ]
        .astype(float)
        .clip(
            lower=0.0
        )
    )

    priority_sum = float(
        priority_values.sum()
    )

    if priority_sum <= 0:
        probabilities = np.full(
            group_count,
            1.0 / group_count,
        )
    else:
        probabilities = (
            priority_values
            / priority_sum
        ).to_numpy()

    additional_records = np.floor(
        probabilities
        * remaining_records
    ).astype(
        int
    )

    result[
        "target_records"
    ] = (
        minimum_per_group
        + additional_records
    )

    allocation_difference = (
        side_target
        - int(
            result[
                "target_records"
            ].sum()
        )
    )

    fractional_remainders = (
        probabilities
        * remaining_records
        - additional_records
    )

    remainder_order = np.argsort(
        -fractional_remainders
    )

    for position in range(
        allocation_difference
    ):
        row_position = remainder_order[
            position
            % group_count
        ]

        row_index = result.index[
            row_position
        ]

        result.loc[
            row_index,
            "target_records",
        ] += 1

    assert int(
        result[
            "target_records"
        ].sum()
    ) == side_target

    assert (
        result[
            "target_records"
        ]
        >= minimum_per_group
    ).all()

    return result


player_curriculum_df = (
    curriculum_plan_df[
        curriculum_plan_df[
            "side"
        ].eq(
            "Player"
        )
    ]
    .copy()
)

opponent_curriculum_df = (
    curriculum_plan_df[
        curriculum_plan_df[
            "side"
        ].eq(
            "Opponent"
        )
    ]
    .copy()
)


assert len(
    player_curriculum_df
) > 0

assert len(
    opponent_curriculum_df
) > 0


player_curriculum_df = allocate_side_records(
    side_df=player_curriculum_df,
    side_target=FINAL_PLAYER_RECORDS,
    minimum_per_group=100,
)

opponent_curriculum_df = allocate_side_records(
    side_df=opponent_curriculum_df,
    side_target=FINAL_OPPONENT_RECORDS,
    minimum_per_group=100,
)


# ------------------------------------------------------------
# 4B.3 — Combine final curriculum
# ------------------------------------------------------------

final_curriculum_plan_df = pd.concat(
    [
        opponent_curriculum_df,
        player_curriculum_df,
    ],
    ignore_index=True,
)

final_curriculum_plan_df[
    "final_sampling_probability"
] = (
    final_curriculum_plan_df[
        "target_records"
    ]
    / FINAL_CURRICULUM_TOTAL_RECORDS
)

final_curriculum_plan_df = (
    final_curriculum_plan_df
    .sort_values(
        [
            "priority_rank",
            "side",
            "baseline",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 4B.4 — Final side allocation
# ------------------------------------------------------------

final_side_allocation_df = (
    final_curriculum_plan_df
    .groupby(
        "side",
        as_index=False,
    )
    .agg(
        target_records=(
            "target_records",
            "sum",
        ),

        total_losses=(
            "losses",
            "sum",
        ),

        average_score=(
            "score",
            "mean",
        ),
    )
)

final_side_allocation_df[
    "allocation_fraction"
] = (
    final_side_allocation_df[
        "target_records"
    ]
    / FINAL_CURRICULUM_TOTAL_RECORDS
)


# ------------------------------------------------------------
# 4B.5 — Final baseline allocation
# ------------------------------------------------------------

final_baseline_allocation_df = (
    final_curriculum_plan_df
    .groupby(
        "baseline",
        as_index=False,
    )
    .agg(
        target_records=(
            "target_records",
            "sum",
        ),

        total_losses=(
            "losses",
            "sum",
        ),

        average_score=(
            "score",
            "mean",
        ),
    )
)

final_baseline_allocation_df[
    "allocation_fraction"
] = (
    final_baseline_allocation_df[
        "target_records"
    ]
    / FINAL_CURRICULUM_TOTAL_RECORDS
)


# ------------------------------------------------------------
# 4B.6 — Search-focused allocation
# ------------------------------------------------------------

search_mask = (
    final_curriculum_plan_df[
        "baseline"
    ]
    .astype(str)
    .str.lower()
    .str.contains(
        "search|depth",
        regex=True,
    )
)

final_search_records = int(
    final_curriculum_plan_df.loc[
        search_mask,
        "target_records",
    ].sum()
)

final_search_fraction = (
    final_search_records
    / FINAL_CURRICULUM_TOTAL_RECORDS
)


# ------------------------------------------------------------
# 4B.7 — Display final plan
# ------------------------------------------------------------

display_columns = [
    "priority_rank",
    "curriculum_tier",
    "side",
    "baseline",
    "score",
    "losses",
    "target_records",
    "final_sampling_probability",
    "multi_action_target_fraction",
    "policy_disagreement_target_fraction",
    "late_game_target_fraction",
]


print()
print("FINAL REBALANCED CURRICULUM")
print("-" * 100)

display(
    final_curriculum_plan_df[
        display_columns
    ]
)


print()
print("FINAL SIDE ALLOCATION")
print("-" * 60)

display(
    final_side_allocation_df
)


print()
print("FINAL BASELINE ALLOCATION")
print("-" * 60)

display(
    final_baseline_allocation_df
)


print()
print("FINAL QUALITY CHECKS")
print("-" * 60)

print(
    "Total records          :",
    int(
        final_curriculum_plan_df[
            "target_records"
        ].sum()
    ),
)

print(
    "Opponent records       :",
    int(
        final_side_allocation_df.loc[
            final_side_allocation_df[
                "side"
            ].eq(
                "Opponent"
            ),
            "target_records",
        ].sum()
    ),
)

print(
    "Player records         :",
    int(
        final_side_allocation_df.loc[
            final_side_allocation_df[
                "side"
            ].eq(
                "Player"
            ),
            "target_records",
        ].sum()
    ),
)

print(
    "Search-focused records :",
    final_search_records,
)

print(
    "Search fraction        :",
    round(
        final_search_fraction,
        4,
    ),
)


# ------------------------------------------------------------
# 4B.8 — Save final guarded curriculum
# ------------------------------------------------------------

FINAL_CURRICULUM_PLAN_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4b_final_curriculum_plan.csv"
)

FINAL_SIDE_ALLOCATION_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4b_final_side_allocation.csv"
)

FINAL_BASELINE_ALLOCATION_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4b_final_baseline_allocation.csv"
)

FINAL_CURRICULUM_SUMMARY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4b_final_curriculum_summary.json"
)


final_curriculum_plan_df.to_csv(
    FINAL_CURRICULUM_PLAN_FILE,
    index=False,
)

final_side_allocation_df.to_csv(
    FINAL_SIDE_ALLOCATION_FILE,
    index=False,
)

final_baseline_allocation_df.to_csv(
    FINAL_BASELINE_ALLOCATION_FILE,
    index=False,
)


final_curriculum_summary = {
    "total_records":
        FINAL_CURRICULUM_TOTAL_RECORDS,

    "opponent_records":
        FINAL_OPPONENT_RECORDS,

    "player_records":
        FINAL_PLAYER_RECORDS,

    "opponent_fraction":
        FINAL_OPPONENT_FRACTION,

    "player_fraction":
        FINAL_PLAYER_FRACTION,

    "search_records":
        final_search_records,

    "search_fraction":
        final_search_fraction,

    "reason":
        (
            "Opponent-side emphasis capped at 70% to target "
            "the measured weakness without catastrophic "
            "forgetting of Player-side strategy."
        ),

    "curriculum_plan":
        final_curriculum_plan_df.to_dict(
            orient="records"
        ),
}

with open(
    FINAL_CURRICULUM_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_curriculum_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 4B.9 — Final validation
# ------------------------------------------------------------

assert int(
    final_curriculum_plan_df[
        "target_records"
    ].sum()
) == 4000

assert int(
    final_side_allocation_df.loc[
        final_side_allocation_df[
            "side"
        ].eq(
            "Opponent"
        ),
        "target_records",
    ].sum()
) == 2800

assert int(
    final_side_allocation_df.loc[
        final_side_allocation_df[
            "side"
        ].eq(
            "Player"
        ),
        "target_records",
    ].sum()
) == 1200

assert final_search_fraction >= 0.30

assert (
    final_curriculum_plan_df[
        "target_records"
    ]
    >= 100
).all()

assert FINAL_CURRICULUM_PLAN_FILE.exists()

assert FINAL_SIDE_ALLOCATION_FILE.exists()

assert FINAL_BASELINE_ALLOCATION_FILE.exists()

assert FINAL_CURRICULUM_SUMMARY_FILE.exists()


print()
print("SAVED FINAL CURRICULUM REPORTS")
print("-" * 60)

print(
    FINAL_CURRICULUM_PLAN_FILE
)

print(
    FINAL_SIDE_ALLOCATION_FILE
)

print(
    FINAL_BASELINE_ALLOCATION_FILE
)

print(
    FINAL_CURRICULUM_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 4B REBALANCED CURRICULUM PASSED"
)


# ## Section 5 — Production Self-Play API Validation
# 
# #### This section inspects and validates the production battle simulator, search engine, competitive-policy agent, legal-move system, and battle-state APIs.
# 
# #### The goal is to construct the search-guided self-play generator using the actual production interfaces rather than assumptions about class names or method signatures.

# In[6]:


# ============================================================
# SECTION 5
# PRODUCTION SELF-PLAY API VALIDATION
# ============================================================

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


print("=" * 100)
print("SECTION 5 — PRODUCTION SELF-PLAY API VALIDATION")
print("=" * 100)


# ------------------------------------------------------------
# 5.1 — Ensure project root is importable
# ------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ------------------------------------------------------------
# 5.2 — Candidate production modules
# ------------------------------------------------------------

candidate_modules = [
    "src.simulator",
    "src.battle_state",
    "src.legal_moves",
    "src.adapters",
    "src.engine.advanced_search",
    "src.competitive.competitive_policy_engine",
    "src.competitive.competitive_policy_agent",
]


module_inventory: list[dict[str, Any]] = []

loaded_modules: dict[str, ModuleType] = {}


# ------------------------------------------------------------
# 5.3 — Load and reload production modules
# ------------------------------------------------------------

for module_name in candidate_modules:

    try:
        module = importlib.import_module(
            module_name
        )

        module = importlib.reload(
            module
        )

        loaded_modules[
            module_name
        ] = module

        module_inventory.append(
            {
                "module":
                    module_name,

                "loaded":
                    True,

                "file":
                    str(
                        getattr(
                            module,
                            "__file__",
                            "",
                        )
                    ),

                "error":
                    None,
            }
        )

    except Exception as exc:

        module_inventory.append(
            {
                "module":
                    module_name,

                "loaded":
                    False,

                "file":
                    None,

                "error":
                    f"{type(exc).__name__}: {exc}",
            }
        )


module_inventory_df = pd.DataFrame(
    module_inventory
)


print()
print("MODULE LOAD INVENTORY")
print("-" * 100)

display(
    module_inventory_df
)


assert module_inventory_df[
    "loaded"
].all(), (
    "One or more production modules failed to load."
)


# ------------------------------------------------------------
# 5.4 — Safe signature helper
# ------------------------------------------------------------

def safe_signature(
    obj: Any,
) -> str:
    try:
        return str(
            inspect.signature(
                obj
            )
        )

    except Exception:
        return "<signature unavailable>"


# ------------------------------------------------------------
# 5.5 — Public symbol inventory
# ------------------------------------------------------------

symbol_rows: list[dict[str, Any]] = []

for module_name, module in loaded_modules.items():

    for symbol_name in dir(
        module
    ):

        if symbol_name.startswith(
            "_"
        ):
            continue

        symbol = getattr(
            module,
            symbol_name,
        )

        if inspect.isclass(
            symbol
        ):
            symbol_type = "class"

        elif inspect.isfunction(
            symbol
        ):
            symbol_type = "function"

        else:
            continue

        symbol_rows.append(
            {
                "module":
                    module_name,

                "symbol":
                    symbol_name,

                "symbol_type":
                    symbol_type,

                "signature":
                    safe_signature(
                        symbol
                    ),

                "defined_in_module":
                    getattr(
                        symbol,
                        "__module__",
                        None,
                    ) == module_name,
            }
        )


symbol_inventory_df = pd.DataFrame(
    symbol_rows
)

if len(
    symbol_inventory_df
) > 0:
    symbol_inventory_df = (
        symbol_inventory_df
        .sort_values(
            [
                "module",
                "symbol_type",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )


print()
print("PUBLIC CLASS AND FUNCTION INVENTORY")
print("-" * 100)

display(
    symbol_inventory_df
)


# ------------------------------------------------------------
# 5.6 — Identify likely production classes
# ------------------------------------------------------------

def find_symbols_by_keywords(
    inventory_df: pd.DataFrame,
    keywords: list[str],
    symbol_type: str | None = None,
) -> pd.DataFrame:

    mask = pd.Series(
        True,
        index=inventory_df.index,
    )

    if symbol_type is not None:
        mask &= inventory_df[
            "symbol_type"
        ].eq(
            symbol_type
        )

    keyword_pattern = "|".join(
        keywords
    )

    mask &= (
        inventory_df[
            "symbol"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            keyword_pattern,
            regex=True,
        )
    )

    return (
        inventory_df[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


simulator_candidates_df = find_symbols_by_keywords(
    symbol_inventory_df,
    [
        "simulator",
        "battle",
        "match",
    ],
    symbol_type="class",
)

search_candidates_df = find_symbols_by_keywords(
    symbol_inventory_df,
    [
        "search",
        "engine",
        "minimax",
        "iddfs",
    ],
    symbol_type="class",
)

agent_candidates_df = find_symbols_by_keywords(
    symbol_inventory_df,
    [
        "agent",
        "policy",
    ],
    symbol_type="class",
)

state_candidates_df = find_symbols_by_keywords(
    symbol_inventory_df,
    [
        "state",
        "observation",
    ],
    symbol_type="class",
)

move_candidates_df = find_symbols_by_keywords(
    symbol_inventory_df,
    [
        "move",
        "action",
        "legal",
    ],
)


print()
print("SIMULATOR CANDIDATES")
print("-" * 100)

display(
    simulator_candidates_df
)

print()
print("SEARCH-ENGINE CANDIDATES")
print("-" * 100)

display(
    search_candidates_df
)

print()
print("AGENT CANDIDATES")
print("-" * 100)

display(
    agent_candidates_df
)

print()
print("STATE CANDIDATES")
print("-" * 100)

display(
    state_candidates_df
)

print()
print("MOVE/ACTION CANDIDATES")
print("-" * 100)

display(
    move_candidates_df
)


# ------------------------------------------------------------
# 5.7 — Inspect methods of important classes
# ------------------------------------------------------------

method_rows: list[dict[str, Any]] = []

important_keywords = [
    "simulator",
    "battle",
    "search",
    "engine",
    "agent",
    "policy",
    "state",
    "adapter",
]


for module_name, module in loaded_modules.items():

    for symbol_name in dir(
        module
    ):

        if symbol_name.startswith(
            "_"
        ):
            continue

        symbol = getattr(
            module,
            symbol_name,
        )

        if not inspect.isclass(
            symbol
        ):
            continue

        lower_name = symbol_name.lower()

        if not any(
            keyword in lower_name
            for keyword in important_keywords
        ):
            continue

        for method_name, method in inspect.getmembers(
            symbol,
            predicate=inspect.isfunction,
        ):

            if method_name.startswith(
                "_"
            ):
                continue

            method_rows.append(
                {
                    "module":
                        module_name,

                    "class":
                        symbol_name,

                    "method":
                        method_name,

                    "signature":
                        safe_signature(
                            method
                        ),
                }
            )


method_inventory_df = pd.DataFrame(
    method_rows
)

if len(
    method_inventory_df
) > 0:
    method_inventory_df = (
        method_inventory_df
        .sort_values(
            [
                "module",
                "class",
                "method",
            ]
        )
        .reset_index(
            drop=True
        )
    )


print()
print("IMPORTANT PUBLIC METHOD INVENTORY")
print("-" * 100)

display(
    method_inventory_df
)


# ------------------------------------------------------------
# 5.8 — Detect likely callable entry points
# ------------------------------------------------------------

entry_point_keywords = [
    "run",
    "play",
    "battle",
    "simulate",
    "choose",
    "select",
    "decide",
    "search",
    "legal",
    "generate",
    "step",
    "reset",
]


if len(
    method_inventory_df
) > 0:

    entry_point_mask = (
        method_inventory_df[
            "method"
        ]
        .astype(str)
        .str.lower()
        .apply(
            lambda value: any(
                keyword in value
                for keyword in entry_point_keywords
            )
        )
    )

    entry_point_df = (
        method_inventory_df[
            entry_point_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

else:
    entry_point_df = pd.DataFrame()


print()
print("LIKELY SELF-PLAY ENTRY POINTS")
print("-" * 100)

display(
    entry_point_df
)


# ------------------------------------------------------------
# 5.9 — Inspect previously validated Notebook 47/48 objects
# ------------------------------------------------------------

notebook_object_names = [
    "production_agent",
    "competitive_agent",
    "competitive_policy_agent",
    "policy_engine",
    "competitive_engine",
    "search_engine",
    "battle_simulator",
    "simulator",
]


runtime_object_rows = []

for object_name in notebook_object_names:

    object_exists = (
        object_name in globals()
    )

    if object_exists:
        runtime_object = globals()[
            object_name
        ]

        runtime_object_rows.append(
            {
                "name":
                    object_name,

                "exists":
                    True,

                "type":
                    type(
                        runtime_object
                    ).__name__,

                "module":
                    type(
                        runtime_object
                    ).__module__,

                "callable":
                    callable(
                        runtime_object
                    ),

                "signature":
                    safe_signature(
                        runtime_object
                    ),
            }
        )

    else:
        runtime_object_rows.append(
            {
                "name":
                    object_name,

                "exists":
                    False,

                "type":
                    None,

                "module":
                    None,

                "callable":
                    False,

                "signature":
                    None,
            }
        )


runtime_object_df = pd.DataFrame(
    runtime_object_rows
)


print()
print("CURRENT NOTEBOOK RUNTIME OBJECTS")
print("-" * 100)

display(
    runtime_object_df
)


# ------------------------------------------------------------
# 5.10 — Inspect production checkpoint structure
# ------------------------------------------------------------

checkpoint_keys = sorted(
    list(
        base_checkpoint.keys()
    )
)

checkpoint_structure = {
    key: type(
        base_checkpoint[
            key
        ]
    ).__name__
    for key in checkpoint_keys
}


print()
print("BASE CHECKPOINT STRUCTURE")
print("-" * 100)

for key, value_type in checkpoint_structure.items():
    print(
        f"{key:35s}: {value_type}"
    )


# ------------------------------------------------------------
# 5.11 — Inspect actor-relative dataset schema
# ------------------------------------------------------------

dataset_schema_rows = []

for column in original_dataset_df.columns:

    sample_value = None

    non_null_values = (
        original_dataset_df[
            column
        ]
        .dropna()
    )

    if len(
        non_null_values
    ) > 0:
        sample_value = non_null_values.iloc[
            0
        ]

    sample_text = repr(
        sample_value
    )

    if len(
        sample_text
    ) > 150:
        sample_text = (
            sample_text[:147]
            + "..."
        )

    dataset_schema_rows.append(
        {
            "column":
                column,

            "dtype":
                str(
                    original_dataset_df[
                        column
                    ].dtype
                ),

            "non_null":
                int(
                    original_dataset_df[
                        column
                    ].notna()
                    .sum()
                ),

            "sample":
                sample_text,
        }
    )


dataset_schema_df = pd.DataFrame(
    dataset_schema_rows
)


print()
print("ACTOR-RELATIVE DATASET SCHEMA")
print("-" * 100)

display(
    dataset_schema_df
)


# ------------------------------------------------------------
# 5.12 — Determine minimum readiness
# ------------------------------------------------------------

readiness_checks = {
    "all_modules_loaded":
        bool(
            module_inventory_df[
                "loaded"
            ].all()
        ),

    "simulator_candidate_found":
        bool(
            len(
                simulator_candidates_df
            ) > 0
        ),

    "search_candidate_found":
        bool(
            len(
                search_candidates_df
            ) > 0
        ),

    "agent_candidate_found":
        bool(
            len(
                agent_candidates_df
            ) > 0
        ),

    "state_candidate_found":
        bool(
            len(
                state_candidates_df
            ) > 0
        ),

    "entry_points_found":
        bool(
            len(
                entry_point_df
            ) > 0
        ),

    "checkpoint_available":
        bool(
            BASE_POLICY_CHECKPOINT.exists()
        ),

    "dataset_available":
        bool(
            ACTOR_RELATIVE_DATASET.exists()
        ),

    "curriculum_available":
        bool(
            FINAL_CURRICULUM_PLAN_FILE.exists()
        ),
}


readiness_pass_count = int(
    sum(
        readiness_checks.values()
    )
)

readiness_total_count = int(
    len(
        readiness_checks
    )
)

readiness_fraction = (
    readiness_pass_count
    / readiness_total_count
)


print()
print("SELF-PLAY API READINESS")
print("-" * 100)

for check_name, check_passed in readiness_checks.items():
    status = (
        "PASS"
        if check_passed
        else "FAIL"
    )

    print(
        f"{check_name:35s}: {status}"
    )

print()
print(
    "Readiness:",
    f"{readiness_pass_count}/{readiness_total_count}",
    f"({readiness_fraction:.2%})",
)


# ------------------------------------------------------------
# 5.13 — Save API inventory reports
# ------------------------------------------------------------

SECTION5_MODULE_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_module_inventory.csv"
)

SECTION5_SYMBOL_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_symbol_inventory.csv"
)

SECTION5_METHOD_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_method_inventory.csv"
)

SECTION5_ENTRY_POINT_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_entry_point_inventory.csv"
)

SECTION5_DATASET_SCHEMA_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_dataset_schema.csv"
)

SECTION5_API_SUMMARY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_api_validation_summary.json"
)


module_inventory_df.to_csv(
    SECTION5_MODULE_INVENTORY_FILE,
    index=False,
)

symbol_inventory_df.to_csv(
    SECTION5_SYMBOL_INVENTORY_FILE,
    index=False,
)

method_inventory_df.to_csv(
    SECTION5_METHOD_INVENTORY_FILE,
    index=False,
)

entry_point_df.to_csv(
    SECTION5_ENTRY_POINT_FILE,
    index=False,
)

dataset_schema_df.to_csv(
    SECTION5_DATASET_SCHEMA_FILE,
    index=False,
)


section5_summary = {
    "readiness_checks":
        readiness_checks,

    "readiness_pass_count":
        readiness_pass_count,

    "readiness_total_count":
        readiness_total_count,

    "readiness_fraction":
        readiness_fraction,

    "checkpoint_structure":
        checkpoint_structure,

    "runtime_objects":
        runtime_object_df.to_dict(
            orient="records"
        ),

    "simulator_candidates":
        simulator_candidates_df.to_dict(
            orient="records"
        ),

    "search_candidates":
        search_candidates_df.to_dict(
            orient="records"
        ),

    "agent_candidates":
        agent_candidates_df.to_dict(
            orient="records"
        ),

    "state_candidates":
        state_candidates_df.to_dict(
            orient="records"
        ),

    "entry_points":
        entry_point_df.to_dict(
            orient="records"
        ),
}


with open(
    SECTION5_API_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        section5_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 5.14 — Final validation
# ------------------------------------------------------------

assert SECTION5_MODULE_INVENTORY_FILE.exists()

assert SECTION5_SYMBOL_INVENTORY_FILE.exists()

assert SECTION5_METHOD_INVENTORY_FILE.exists()

assert SECTION5_ENTRY_POINT_FILE.exists()

assert SECTION5_DATASET_SCHEMA_FILE.exists()

assert SECTION5_API_SUMMARY_FILE.exists()

assert readiness_checks[
    "all_modules_loaded"
]

assert readiness_checks[
    "search_candidate_found"
]

assert readiness_checks[
    "agent_candidate_found"
]

assert readiness_checks[
    "entry_points_found"
]


print()
print("SAVED API REPORTS")
print("-" * 100)

print(
    SECTION5_MODULE_INVENTORY_FILE
)

print(
    SECTION5_SYMBOL_INVENTORY_FILE
)

print(
    SECTION5_METHOD_INVENTORY_FILE
)

print(
    SECTION5_ENTRY_POINT_FILE
)

print(
    SECTION5_DATASET_SCHEMA_FILE
)

print(
    SECTION5_API_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 5 PRODUCTION SELF-PLAY API VALIDATION PASSED"
)


# ## Section 6 — Hard-Example Replay Contract
# 
# #### This section prepares the hard-example mining pipeline.
# 
# #### It combines:
# 
# - Notebook 48 tournament losses
# - the production API inventory from Section 5
# - available match-identification fields
# - simulator and agent method signatures
# - card and side information needed for deterministic replay
# 
# #### The section does not yet generate training labels. It first establishes a validated replay contract so that hard examples are reconstructed from genuine tournament failures rather than from guessed simulator calls.

# In[7]:


# ============================================================
# SECTION 6
# HARD-EXAMPLE REPLAY CONTRACT
# ============================================================

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 6 — HARD-EXAMPLE REPLAY CONTRACT")
print("=" * 100)


# ------------------------------------------------------------
# 6.1 — Required Section 3 and Section 5 reports
# ------------------------------------------------------------

SECTION6_LOSS_MATCHES_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section3_loss_matches.csv"
)

SECTION6_MODULE_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_module_inventory.csv"
)

SECTION6_SYMBOL_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_symbol_inventory.csv"
)

SECTION6_METHOD_INVENTORY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_method_inventory.csv"
)

SECTION6_ENTRY_POINT_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_entry_point_inventory.csv"
)

SECTION6_API_SUMMARY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section5_api_validation_summary.json"
)

SECTION6_CURRICULUM_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section4b_final_curriculum_plan.csv"
)


required_section6_files = [
    SECTION6_LOSS_MATCHES_FILE,
    SECTION6_MODULE_INVENTORY_FILE,
    SECTION6_SYMBOL_INVENTORY_FILE,
    SECTION6_METHOD_INVENTORY_FILE,
    SECTION6_ENTRY_POINT_FILE,
    SECTION6_API_SUMMARY_FILE,
    SECTION6_CURRICULUM_FILE,
]

missing_section6_files = [
    str(path)
    for path in required_section6_files
    if not path.exists()
]

assert not missing_section6_files, (
    "Missing required reports for Section 6: "
    f"{missing_section6_files}"
)


# ------------------------------------------------------------
# 6.2 — Load diagnostic reports
# ------------------------------------------------------------

loss_matches_df = pd.read_csv(
    SECTION6_LOSS_MATCHES_FILE
)

module_inventory_df = pd.read_csv(
    SECTION6_MODULE_INVENTORY_FILE
)

symbol_inventory_df = pd.read_csv(
    SECTION6_SYMBOL_INVENTORY_FILE
)

method_inventory_df = pd.read_csv(
    SECTION6_METHOD_INVENTORY_FILE
)

entry_point_df = pd.read_csv(
    SECTION6_ENTRY_POINT_FILE
)

curriculum_plan_df = pd.read_csv(
    SECTION6_CURRICULUM_FILE
)

with open(
    SECTION6_API_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:
    section5_api_summary = json.load(
        file
    )


print()
print("LOADED INPUTS")
print("-" * 60)

print(
    "Loss matches          :",
    len(loss_matches_df),
)

print(
    "Modules               :",
    len(module_inventory_df),
)

print(
    "Symbols               :",
    len(symbol_inventory_df),
)

print(
    "Methods               :",
    len(method_inventory_df),
)

print(
    "Likely entry points   :",
    len(entry_point_df),
)

print(
    "Curriculum groups     :",
    len(curriculum_plan_df),
)


# ------------------------------------------------------------
# 6.3 — Validate loss-match schema
# ------------------------------------------------------------

required_loss_columns = [
    "match_id",
    "agent_name",
    "baseline_name",
    "player_card",
    "opponent_card",
    "starting_side",
    "winner",
    "turns",
    "stop_reason",
    "agent_result",
    "fallback_count",
    "elapsed_seconds",
    "evaluated_agent_side",
    "competitive_win",
    "competitive_loss",
]

missing_loss_columns = [
    column
    for column in required_loss_columns
    if column not in loss_matches_df.columns
]

assert not missing_loss_columns, (
    "Loss-match report is missing required columns: "
    f"{missing_loss_columns}"
)

assert len(
    loss_matches_df
) == 106, (
    "Expected exactly 106 Notebook 48 losses."
)


# ------------------------------------------------------------
# 6.4 — Normalize loss-match fields
# ------------------------------------------------------------

def normalize_text(
    value: Any,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip()


def normalize_side(
    value: Any,
) -> str:
    text = normalize_text(
        value
    ).lower()

    player_values = {
        "player",
        "first",
        "player_side",
        "agent_player",
        "competitive_player",
    }

    opponent_values = {
        "opponent",
        "second",
        "opponent_side",
        "agent_opponent",
        "competitive_opponent",
    }

    if text in player_values:
        return "Player"

    if text in opponent_values:
        return "Opponent"

    return normalize_text(
        value
    )


loss_matches_df[
    "normalized_side"
] = (
    loss_matches_df[
        "evaluated_agent_side"
    ]
    .map(
        normalize_side
    )
)

empty_side_mask = (
    loss_matches_df[
        "normalized_side"
    ]
    .eq("")
)

loss_matches_df.loc[
    empty_side_mask,
    "normalized_side",
] = (
    loss_matches_df.loc[
        empty_side_mask,
        "starting_side",
    ]
    .map(
        normalize_side
    )
)


# ------------------------------------------------------------
# 6.5 — Examine reproducibility fields
# ------------------------------------------------------------

reproducibility_keywords = [
    "seed",
    "random",
    "rng",
    "match_id",
    "game_id",
    "episode",
    "player_card",
    "opponent_card",
    "starting_side",
    "side",
    "deck",
    "state",
    "observation",
    "history",
    "trajectory",
    "action",
    "move",
]


reproducibility_rows = []

for column in loss_matches_df.columns:

    column_lower = str(
        column
    ).lower()

    matched_keywords = [
        keyword
        for keyword in reproducibility_keywords
        if keyword in column_lower
    ]

    if not matched_keywords:
        continue

    unique_count = int(
        loss_matches_df[
            column
        ].nunique(
            dropna=True
        )
    )

    missing_count = int(
        loss_matches_df[
            column
        ].isna()
        .sum()
    )

    sample_values = (
        loss_matches_df[
            column
        ]
        .dropna()
        .astype(str)
        .head(3)
        .tolist()
    )

    reproducibility_rows.append(
        {
            "column":
                column,

            "matched_keywords":
                ", ".join(
                    matched_keywords
                ),

            "dtype":
                str(
                    loss_matches_df[
                        column
                    ].dtype
                ),

            "unique_count":
                unique_count,

            "missing_count":
                missing_count,

            "sample_values":
                " | ".join(
                    sample_values
                ),
        }
    )


reproducibility_fields_df = pd.DataFrame(
    reproducibility_rows
)


print()
print("AVAILABLE REPRODUCIBILITY FIELDS")
print("-" * 100)

display(
    reproducibility_fields_df
)


# ------------------------------------------------------------
# 6.6 — Detect direct replay-related methods
# ------------------------------------------------------------

replay_method_keywords = [
    "replay",
    "reconstruct",
    "restore",
    "load",
    "reset",
    "initialize",
    "initialise",
    "create",
    "build",
    "setup",
    "simulate",
    "run",
    "play",
    "battle",
    "match",
    "step",
]


def contains_any_keyword(
    value: Any,
    keywords: list[str],
) -> bool:
    text = normalize_text(
        value
    ).lower()

    return any(
        keyword in text
        for keyword in keywords
    )


replay_method_mask = (
    method_inventory_df[
        "method"
    ]
    .apply(
        lambda value: contains_any_keyword(
            value,
            replay_method_keywords,
        )
    )
)

replay_method_candidates_df = (
    method_inventory_df[
        replay_method_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print()
print("REPLAY-RELATED METHOD CANDIDATES")
print("-" * 100)

display(
    replay_method_candidates_df
)


# ------------------------------------------------------------
# 6.7 — Detect move-selection and search methods
# ------------------------------------------------------------

move_selection_keywords = [
    "choose",
    "select",
    "decide",
    "act",
    "policy",
    "predict",
    "move",
    "action",
]

search_keywords = [
    "search",
    "minimax",
    "iddfs",
    "iterative",
    "evaluate",
    "score",
    "best",
]


move_selection_mask = (
    method_inventory_df[
        "method"
    ]
    .apply(
        lambda value: contains_any_keyword(
            value,
            move_selection_keywords,
        )
    )
)

search_method_mask = (
    method_inventory_df[
        "method"
    ]
    .apply(
        lambda value: contains_any_keyword(
            value,
            search_keywords,
        )
    )
)


move_selection_candidates_df = (
    method_inventory_df[
        move_selection_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

search_method_candidates_df = (
    method_inventory_df[
        search_method_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print()
print("POLICY MOVE-SELECTION CANDIDATES")
print("-" * 100)

display(
    move_selection_candidates_df
)

print()
print("SEARCH LABELING CANDIDATES")
print("-" * 100)

display(
    search_method_candidates_df
)


# ------------------------------------------------------------
# 6.8 — Detect legal-move and state-transition methods
# ------------------------------------------------------------

legal_move_keywords = [
    "legal",
    "valid",
    "available",
    "actions",
    "moves",
]

transition_keywords = [
    "step",
    "apply",
    "transition",
    "advance",
    "execute",
    "update",
]


legal_move_mask = (
    method_inventory_df[
        "method"
    ]
    .apply(
        lambda value: contains_any_keyword(
            value,
            legal_move_keywords,
        )
    )
)

transition_mask = (
    method_inventory_df[
        "method"
    ]
    .apply(
        lambda value: contains_any_keyword(
            value,
            transition_keywords,
        )
    )
)


legal_move_candidates_df = (
    method_inventory_df[
        legal_move_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

transition_candidates_df = (
    method_inventory_df[
        transition_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


print()
print("LEGAL-MOVE CANDIDATES")
print("-" * 100)

display(
    legal_move_candidates_df
)

print()
print("STATE-TRANSITION CANDIDATES")
print("-" * 100)

display(
    transition_candidates_df
)


# ------------------------------------------------------------
# 6.9 — Parse method signature parameter names
# ------------------------------------------------------------

def extract_signature_parameters(
    signature_text: Any,
) -> list[str]:
    """
    Extract approximate parameter names from an inspected
    Python signature string.

    This is diagnostic only. It does not call the method.
    """

    text = normalize_text(
        signature_text
    )

    if not text.startswith(
        "("
    ):
        return []

    inner_text = text[
        1:
    ]

    if ")" in inner_text:
        inner_text = inner_text.split(
            ")",
            1,
        )[0]

    parameters = []

    for raw_part in inner_text.split(
        ","
    ):
        part = raw_part.strip()

        if not part:
            continue

        part = part.lstrip(
            "*"
        )

        parameter_name = re.split(
            r"[:=]",
            part,
            maxsplit=1,
        )[0].strip()

        if (
            parameter_name
            and parameter_name
            not in {
                "self",
                "cls",
            }
        ):
            parameters.append(
                parameter_name
            )

    return parameters


method_inventory_df[
    "parameter_names"
] = (
    method_inventory_df[
        "signature"
    ]
    .map(
        extract_signature_parameters
    )
)

method_inventory_df[
    "parameter_count"
] = (
    method_inventory_df[
        "parameter_names"
    ]
    .map(
        len
    )
)


# ------------------------------------------------------------
# 6.10 — Rank methods for replay suitability
# ------------------------------------------------------------

def score_replay_method(
    row: pd.Series,
) -> float:
    method_name = normalize_text(
        row.get(
            "method",
            "",
        )
    ).lower()

    class_name = normalize_text(
        row.get(
            "class",
            "",
        )
    ).lower()

    parameters = row.get(
        "parameter_names",
        [],
    )

    if not isinstance(
        parameters,
        list,
    ):
        parameters = []

    score = 0.0

    method_weights = {
        "replay":
            8.0,

        "reconstruct":
            8.0,

        "restore":
            7.0,

        "simulate":
            6.0,

        "run_match":
            6.0,

        "play_match":
            6.0,

        "battle":
            5.0,

        "match":
            4.0,

        "reset":
            3.0,

        "create":
            2.0,

        "build":
            2.0,

        "setup":
            2.0,
    }

    for keyword, weight in method_weights.items():
        if keyword in method_name:
            score += weight

    if (
        "simulator" in class_name
        or "battle" in class_name
    ):
        score += 3.0

    parameter_weights = {
        "seed":
            4.0,

        "player_card":
            3.0,

        "opponent_card":
            3.0,

        "starting_side":
            3.0,

        "side":
            1.5,

        "state":
            2.0,

        "agent":
            1.0,

        "opponent":
            1.0,
    }

    for parameter in parameters:
        parameter_lower = str(
            parameter
        ).lower()

        for keyword, weight in parameter_weights.items():
            if keyword in parameter_lower:
                score += weight

    return float(
        score
    )


method_inventory_df[
    "replay_suitability_score"
] = method_inventory_df.apply(
    score_replay_method,
    axis=1,
)

ranked_replay_methods_df = (
    method_inventory_df
    .sort_values(
        [
            "replay_suitability_score",
            "module",
            "class",
            "method",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("RANKED REPLAY METHODS")
print("-" * 100)

display(
    ranked_replay_methods_df.head(
        30
    )
)


# ------------------------------------------------------------
# 6.11 — Rank policy-labeling methods
# ------------------------------------------------------------

def score_labeling_method(
    row: pd.Series,
) -> float:
    method_name = normalize_text(
        row.get(
            "method",
            "",
        )
    ).lower()

    class_name = normalize_text(
        row.get(
            "class",
            "",
        )
    ).lower()

    parameters = row.get(
        "parameter_names",
        [],
    )

    if not isinstance(
        parameters,
        list,
    ):
        parameters = []

    score = 0.0

    if "search" in method_name:
        score += 7.0

    if "best" in method_name:
        score += 5.0

    if "choose" in method_name:
        score += 4.0

    if "select" in method_name:
        score += 4.0

    if "move" in method_name:
        score += 3.0

    if "action" in method_name:
        score += 3.0

    if "evaluate" in method_name:
        score += 2.0

    if (
        "search" in class_name
        or "engine" in class_name
    ):
        score += 3.0

    for parameter in parameters:
        parameter_lower = str(
            parameter
        ).lower()

        if "state" in parameter_lower:
            score += 3.0

        if "depth" in parameter_lower:
            score += 2.0

        if "time" in parameter_lower:
            score += 1.0

        if "legal" in parameter_lower:
            score += 2.0

    return float(
        score
    )


method_inventory_df[
    "labeling_suitability_score"
] = method_inventory_df.apply(
    score_labeling_method,
    axis=1,
)

ranked_labeling_methods_df = (
    method_inventory_df
    .sort_values(
        [
            "labeling_suitability_score",
            "module",
            "class",
            "method",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("RANKED SEARCH-LABELING METHODS")
print("-" * 100)

display(
    ranked_labeling_methods_df.head(
        30
    )
)


# ------------------------------------------------------------
# 6.12 — Determine available replay evidence
# ------------------------------------------------------------

seed_columns = [
    column
    for column in loss_matches_df.columns
    if (
        "seed" in str(
            column
        ).lower()
        or "rng" in str(
            column
        ).lower()
    )
]

trajectory_columns = [
    column
    for column in loss_matches_df.columns
    if any(
        keyword in str(
            column
        ).lower()
        for keyword in [
            "trajectory",
            "history",
            "state",
            "observation",
            "action_sequence",
            "move_sequence",
        ]
    )
]

card_columns_present = all(
    column in loss_matches_df.columns
    for column in [
        "player_card",
        "opponent_card",
    ]
)

side_column_present = any(
    column in loss_matches_df.columns
    for column in [
        "evaluated_agent_side",
        "starting_side",
    ]
)

match_id_present = (
    "match_id"
    in loss_matches_df.columns
)


replay_evidence = {
    "match_id_available":
        match_id_present,

    "card_pair_available":
        card_columns_present,

    "side_available":
        side_column_present,

    "seed_columns":
        seed_columns,

    "seed_available":
        bool(
            seed_columns
        ),

    "trajectory_columns":
        trajectory_columns,

    "trajectory_available":
        bool(
            trajectory_columns
        ),

    "replay_methods_found":
        bool(
            len(
                replay_method_candidates_df
            ) > 0
        ),

    "search_labeling_methods_found":
        bool(
            len(
                search_method_candidates_df
            ) > 0
        ),

    "legal_move_methods_found":
        bool(
            len(
                legal_move_candidates_df
            ) > 0
        ),

    "transition_methods_found":
        bool(
            len(
                transition_candidates_df
            ) > 0
        ),
}


# ------------------------------------------------------------
# 6.13 — Classify replay level
# ------------------------------------------------------------

if (
    replay_evidence[
        "trajectory_available"
    ]
):
    replay_level = (
        "LEVEL_3_EXACT_TRAJECTORY_REPLAY"
    )

    replay_description = (
        "Exact per-turn state or action trajectory data is "
        "available for direct hard-example reconstruction."
    )

elif (
    replay_evidence[
        "seed_available"
    ]
    and replay_evidence[
        "card_pair_available"
    ]
    and replay_evidence[
        "side_available"
    ]
):
    replay_level = (
        "LEVEL_2_DETERMINISTIC_MATCH_REPLAY"
    )

    replay_description = (
        "Losses can potentially be reproduced from the saved "
        "random seed, card pair, and evaluated side."
    )

elif (
    replay_evidence[
        "card_pair_available"
    ]
    and replay_evidence[
        "side_available"
    ]
):
    replay_level = (
        "LEVEL_1_SCENARIO_REGENERATION"
    )

    replay_description = (
        "Exact trajectories are not stored, but matching card "
        "and side scenarios can be regenerated and mined for "
        "new policy-search disagreements."
    )

else:
    replay_level = (
        "LEVEL_0_INSUFFICIENT_REPLAY_METADATA"
    )

    replay_description = (
        "Tournament results do not currently contain enough "
        "metadata to reconstruct or regenerate loss scenarios."
    )


print()
print("REPLAY CAPABILITY")
print("-" * 100)

print(
    "Replay level       :",
    replay_level,
)

print(
    "Description        :",
    replay_description,
)

print(
    "Seed columns       :",
    seed_columns,
)

print(
    "Trajectory columns :",
    trajectory_columns,
)


# ------------------------------------------------------------
# 6.14 — Build the hard-example source manifest
# ------------------------------------------------------------

baseline_strength_map = {
    "random":
        1.0,

    "greedy":
        2.0,

    "depth":
        3.0,

    "search":
        3.0,
}


def get_baseline_strength(
    baseline_name: Any,
) -> float:
    text = normalize_text(
        baseline_name
    ).lower()

    for keyword, value in baseline_strength_map.items():
        if keyword in text:
            return float(
                value
            )

    return 1.5


loss_matches_df[
    "baseline_strength"
] = (
    loss_matches_df[
        "baseline_name"
    ]
    .map(
        get_baseline_strength
    )
)

loss_matches_df[
    "side_priority"
] = np.where(
    loss_matches_df[
        "normalized_side"
    ].eq(
        "Opponent"
    ),
    2.0,
    1.0,
)

median_loss_turns = float(
    loss_matches_df[
        "turns"
    ].median()
)

if (
    not np.isfinite(
        median_loss_turns
    )
    or median_loss_turns <= 0
):
    median_loss_turns = 1.0

loss_matches_df[
    "turn_complexity"
] = (
    loss_matches_df[
        "turns"
    ]
    / median_loss_turns
).clip(
    lower=0.75,
    upper=1.50,
)

loss_matches_df[
    "hard_example_priority"
] = (
    loss_matches_df[
        "baseline_strength"
    ]
    * loss_matches_df[
        "side_priority"
    ]
    * loss_matches_df[
        "turn_complexity"
    ]
)

loss_matches_df[
    "hard_example_rank"
] = (
    loss_matches_df[
        "hard_example_priority"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(
        int
    )
)

loss_matches_df[
    "source_scenario_id"
] = (
    "NB48_LOSS_"
    + loss_matches_df[
        "match_id"
    ]
    .astype(str)
)

loss_matches_df[
    "replay_level"
] = replay_level

loss_matches_df[
    "mining_status"
] = (
    "PENDING_REPLAY"
)

loss_matches_df[
    "records_mined"
] = 0


manifest_columns = [
    "source_scenario_id",
    "match_id",
    "agent_name",
    "baseline_name",
    "player_card",
    "opponent_card",
    "starting_side",
    "evaluated_agent_side",
    "normalized_side",
    "winner",
    "turns",
    "stop_reason",
    "fallback_count",
    "elapsed_seconds",
    "baseline_strength",
    "side_priority",
    "turn_complexity",
    "hard_example_priority",
    "hard_example_rank",
    "replay_level",
    "mining_status",
    "records_mined",
]


hard_example_manifest_df = (
    loss_matches_df[
        manifest_columns
    ]
    .sort_values(
        [
            "hard_example_rank",
            "match_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


print()
print("HARD-EXAMPLE SOURCE MANIFEST")
print("-" * 100)

display(
    hard_example_manifest_df.head(
        30
    )
)


# ------------------------------------------------------------
# 6.15 — Summarize the hard-example pool
# ------------------------------------------------------------

hard_example_summary_df = (
    hard_example_manifest_df
    .groupby(
        [
            "normalized_side",
            "baseline_name",
        ],
        as_index=False,
    )
    .agg(
        loss_scenarios=(
            "source_scenario_id",
            "count",
        ),

        average_turns=(
            "turns",
            "mean",
        ),

        average_priority=(
            "hard_example_priority",
            "mean",
        ),

        maximum_priority=(
            "hard_example_priority",
            "max",
        ),
    )
    .sort_values(
        [
            "average_priority",
            "loss_scenarios",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("HARD-EXAMPLE POOL SUMMARY")
print("-" * 100)

display(
    hard_example_summary_df
)


# ------------------------------------------------------------
# 6.16 — Define hard-example mining acceptance rules
# ------------------------------------------------------------

hard_example_acceptance_rules = {
    "require_multi_action_state":
        True,

    "minimum_legal_moves":
        2,

    "require_policy_search_disagreement":
        True,

    "minimum_search_depth":
        6,

    "preferred_search_depth":
        8,

    "minimum_score_margin":
        0.05,

    "require_finite_search_score":
        True,

    "reject_duplicate_observations":
        True,

    "reject_illegal_search_labels":
        True,

    "opponent_side_sampling_fraction":
        0.70,

    "player_side_sampling_fraction":
        0.30,

    "target_training_records":
        4000,

    "maximum_records_per_source_match":
        100,

    "preserve_original_dataset_replay_fraction":
        0.30,

    "hard_example_training_fraction":
        0.70,
}


print()
print("HARD-EXAMPLE ACCEPTANCE RULES")
print("-" * 100)

for key, value in hard_example_acceptance_rules.items():
    print(
        f"{key:42s}: {value}"
    )


# ------------------------------------------------------------
# 6.17 — Build the replay contract
# ------------------------------------------------------------

top_replay_methods = (
    ranked_replay_methods_df.head(
        10
    )
)

top_labeling_methods = (
    ranked_labeling_methods_df.head(
        10
    )
)

replay_contract = {
    "replay_level":
        replay_level,

    "replay_description":
        replay_description,

    "replay_evidence":
        replay_evidence,

    "loss_scenario_count":
        int(
            len(
                hard_example_manifest_df
            )
        ),

    "opponent_loss_scenarios":
        int(
            hard_example_manifest_df[
                "normalized_side"
            ]
            .eq(
                "Opponent"
            )
            .sum()
        ),

    "player_loss_scenarios":
        int(
            hard_example_manifest_df[
                "normalized_side"
            ]
            .eq(
                "Player"
            )
            .sum()
        ),

    "acceptance_rules":
        hard_example_acceptance_rules,

    "top_replay_methods":
        top_replay_methods.to_dict(
            orient="records"
        ),

    "top_labeling_methods":
        top_labeling_methods.to_dict(
            orient="records"
        ),

    "legal_move_candidates":
        legal_move_candidates_df.to_dict(
            orient="records"
        ),

    "transition_candidates":
        transition_candidates_df.to_dict(
            orient="records"
        ),
}


# ------------------------------------------------------------
# 6.18 — Save Section 6 reports
# ------------------------------------------------------------

SECTION6_REPRODUCIBILITY_FIELDS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_reproducibility_fields.csv"
)

SECTION6_REPLAY_METHODS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_ranked_replay_methods.csv"
)

SECTION6_LABELING_METHODS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_ranked_labeling_methods.csv"
)

SECTION6_LEGAL_MOVE_METHODS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_legal_move_methods.csv"
)

SECTION6_TRANSITION_METHODS_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_transition_methods.csv"
)

SECTION6_HARD_EXAMPLE_MANIFEST_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_hard_example_manifest.csv"
)

SECTION6_HARD_EXAMPLE_SUMMARY_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_hard_example_summary.csv"
)

SECTION6_REPLAY_CONTRACT_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_replay_contract.json"
)


reproducibility_fields_df.to_csv(
    SECTION6_REPRODUCIBILITY_FIELDS_FILE,
    index=False,
)

ranked_replay_methods_df.to_csv(
    SECTION6_REPLAY_METHODS_FILE,
    index=False,
)

ranked_labeling_methods_df.to_csv(
    SECTION6_LABELING_METHODS_FILE,
    index=False,
)

legal_move_candidates_df.to_csv(
    SECTION6_LEGAL_MOVE_METHODS_FILE,
    index=False,
)

transition_candidates_df.to_csv(
    SECTION6_TRANSITION_METHODS_FILE,
    index=False,
)

hard_example_manifest_df.to_csv(
    SECTION6_HARD_EXAMPLE_MANIFEST_FILE,
    index=False,
)

hard_example_summary_df.to_csv(
    SECTION6_HARD_EXAMPLE_SUMMARY_FILE,
    index=False,
)

with open(
    SECTION6_REPLAY_CONTRACT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        replay_contract,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 6.19 — Final validation
# ------------------------------------------------------------

assert replay_level != (
    "LEVEL_0_INSUFFICIENT_REPLAY_METADATA"
), (
    "Not enough tournament metadata exists to build "
    "the hard-example replay pipeline."
)

assert len(
    hard_example_manifest_df
) == 106

assert hard_example_manifest_df[
    "source_scenario_id"
].is_unique

assert hard_example_manifest_df[
    "match_id"
].notna().all()

assert hard_example_manifest_df[
    "player_card"
].notna().all()

assert hard_example_manifest_df[
    "opponent_card"
].notna().all()

assert hard_example_manifest_df[
    "normalized_side"
].isin(
        [
            "Player",
            "Opponent",
        ]
    ).all()

assert len(
    replay_method_candidates_df
) > 0

assert len(
    search_method_candidates_df
) > 0

assert len(
    legal_move_candidates_df
) > 0

assert len(
    transition_candidates_df
) > 0

assert SECTION6_REPRODUCIBILITY_FIELDS_FILE.exists()

assert SECTION6_REPLAY_METHODS_FILE.exists()

assert SECTION6_LABELING_METHODS_FILE.exists()

assert SECTION6_LEGAL_MOVE_METHODS_FILE.exists()

assert SECTION6_TRANSITION_METHODS_FILE.exists()

assert SECTION6_HARD_EXAMPLE_MANIFEST_FILE.exists()

assert SECTION6_HARD_EXAMPLE_SUMMARY_FILE.exists()

assert SECTION6_REPLAY_CONTRACT_FILE.exists()


print()
print("REPLAY CONTRACT QUALITY CHECKS")
print("-" * 100)

print(
    "Replay level              :",
    replay_level,
)

print(
    "Loss scenarios            :",
    len(
        hard_example_manifest_df
    ),
)

print(
    "Opponent loss scenarios   :",
    int(
        hard_example_manifest_df[
            "normalized_side"
        ]
        .eq(
            "Opponent"
        )
        .sum()
    ),
)

print(
    "Player loss scenarios     :",
    int(
        hard_example_manifest_df[
            "normalized_side"
        ]
        .eq(
            "Player"
        )
        .sum()
    ),
)

print(
    "Replay method candidates  :",
    len(
        replay_method_candidates_df
    ),
)

print(
    "Search-label candidates   :",
    len(
        search_method_candidates_df
    ),
)

print(
    "Legal-move candidates     :",
    len(
        legal_move_candidates_df
    ),
)

print(
    "Transition candidates     :",
    len(
        transition_candidates_df
    ),
)


print()
print("SAVED SECTION 6 REPORTS")
print("-" * 100)

print(
    SECTION6_REPRODUCIBILITY_FIELDS_FILE
)

print(
    SECTION6_REPLAY_METHODS_FILE
)

print(
    SECTION6_LABELING_METHODS_FILE
)

print(
    SECTION6_LEGAL_MOVE_METHODS_FILE
)

print(
    SECTION6_TRANSITION_METHODS_FILE
)

print(
    SECTION6_HARD_EXAMPLE_MANIFEST_FILE
)

print(
    SECTION6_HARD_EXAMPLE_SUMMARY_FILE
)

print(
    SECTION6_REPLAY_CONTRACT_FILE
)

print()
print(
    "✅ SECTION 6 HARD-EXAMPLE REPLAY CONTRACT PASSED"
)


# ## Section 7 — State-Based Hard-Example Regeneration
# 
# #### The production codebase does not expose a standalone battle simulator or match runner. Instead, it provides a state-based decision stack:
# 
# 1. `BattleState` represents the current game position.
# 2. `get_current_legal_moves` generates valid actions.
# 3. `CompetitivePolicyEngine` scores or selects actions.
# 4. `AdvancedSearchEngine` performs deeper search over compatible states.
# 
# #### Section 7 therefore regenerates decision points rather than replaying complete matches through a simulator.
# 
# #### The workflow is resumable. Each prioritized loss scenario is tracked in a progress ledger, and generated state records will be stored separately for search labeling in Section 8.

# In[8]:


# ============================================================
# SECTION 7A
# STATE-BASED RUNTIME CONTRACT AND RESUMABLE QUEUE
# ============================================================

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 7A — STATE-BASED RUNTIME CONTRACT")
print("=" * 100)


# ------------------------------------------------------------
# 7A.1 — Paths
# ------------------------------------------------------------

SECTION7_REPORT_DIR = (
    NOTEBOOK49_REPORT_DIR
    / "section7"
)

SECTION7_TRACE_DIR = (
    SECTION7_REPORT_DIR
    / "scenario_traces"
)

SECTION7_CHECKPOINT_DIR = (
    SECTION7_REPORT_DIR
    / "checkpoints"
)

SECTION7_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION7_TRACE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION7_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION7_MANIFEST_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_hard_example_manifest.csv"
)

SECTION7_REPLAY_CONTRACT_FILE = (
    NOTEBOOK49_REPORT_DIR
    / "section6_replay_contract.json"
)

SECTION7_PROGRESS_FILE = (
    SECTION7_CHECKPOINT_DIR
    / "scenario_progress.csv"
)

SECTION7_RUNTIME_CONTRACT_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_state_runtime_contract.json"
)

SECTION7_QUEUE_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_prioritized_scenario_queue.csv"
)


required_files = [
    SECTION7_MANIFEST_FILE,
    SECTION7_REPLAY_CONTRACT_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

assert not missing_files, (
    "Missing required Section 6 reports: "
    f"{missing_files}"
)


# ------------------------------------------------------------
# 7A.2 — Load Section 6 outputs
# ------------------------------------------------------------

hard_example_manifest_df = pd.read_csv(
    SECTION7_MANIFEST_FILE
)

with open(
    SECTION7_REPLAY_CONTRACT_FILE,
    "r",
    encoding="utf-8",
) as file:

    replay_contract = json.load(
        file
    )


assert (
    replay_contract.get(
        "replay_level"
    )
    == "LEVEL_1_SCENARIO_REGENERATION"
), (
    "Section 7 expects Level 1 scenario regeneration."
)

assert not hard_example_manifest_df.empty

assert (
    hard_example_manifest_df[
        "source_scenario_id"
    ].is_unique
), (
    "source_scenario_id must be unique."
)


# ------------------------------------------------------------
# 7A.3 — Build prioritized scenario queue
# ------------------------------------------------------------

sort_columns = [
    column
    for column in [
        "hard_example_rank",
        "hard_example_priority",
        "match_id",
    ]
    if column in hard_example_manifest_df.columns
]

ascending_lookup = {
    "hard_example_rank":
        True,

    "hard_example_priority":
        False,

    "match_id":
        True,
}

scenario_queue_df = (
    hard_example_manifest_df
    .sort_values(
        sort_columns,
        ascending=[
            ascending_lookup[column]
            for column in sort_columns
        ],
    )
    .reset_index(
        drop=True
    )
)

scenario_queue_df[
    "queue_position"
] = np.arange(
    1,
    len(
        scenario_queue_df
    ) + 1,
)

scenario_queue_df.to_csv(
    SECTION7_QUEUE_FILE,
    index=False,
)


print()
print("PRIORITIZED SCENARIO QUEUE")
print("-" * 100)

queue_display_columns = [
    column
    for column in [
        "queue_position",
        "source_scenario_id",
        "normalized_side",
        "baseline_name",
        "player_card",
        "opponent_card",
        "turns",
        "hard_example_priority",
        "hard_example_rank",
    ]
    if column in scenario_queue_df.columns
]

display(
    scenario_queue_df[
        queue_display_columns
    ].head(
        20
    )
)


# ------------------------------------------------------------
# 7A.4 — Initialize or repair progress ledger
# ------------------------------------------------------------

progress_defaults = {
    "status":
        "PENDING",

    "attempts":
        0,

    "generated_states":
        0,

    "accepted_states":
        0,

    "last_seed":
        np.nan,

    "last_error":
        "",

    "trace_file":
        "",
}


if SECTION7_PROGRESS_FILE.exists():

    existing_progress_df = pd.read_csv(
        SECTION7_PROGRESS_FILE
    )

    preserved_columns = [
        column
        for column in existing_progress_df.columns
        if column not in {
            "queue_position"
        }
    ]

    scenario_progress_df = (
        scenario_queue_df[
            [
                "source_scenario_id",
                "queue_position",
            ]
        ]
        .merge(
            existing_progress_df[
                preserved_columns
            ],
            on="source_scenario_id",
            how="left",
        )
    )

    for column, default_value in progress_defaults.items():

        if column not in scenario_progress_df.columns:
            scenario_progress_df[
                column
            ] = default_value

        else:
            scenario_progress_df[
                column
            ] = scenario_progress_df[
                column
            ].fillna(
                default_value
            )

else:

    scenario_progress_df = scenario_queue_df[
        [
            "source_scenario_id",
            "queue_position",
        ]
    ].copy()

    for column, default_value in progress_defaults.items():
        scenario_progress_df[
            column
        ] = default_value


scenario_progress_df.to_csv(
    SECTION7_PROGRESS_FILE,
    index=False,
)


assert len(
    scenario_progress_df
) == len(
    scenario_queue_df
)

assert scenario_progress_df[
    "source_scenario_id"
].is_unique


print()
print("RESUMABLE PROGRESS LEDGER")
print("-" * 100)

print(
    "Progress file :",
    SECTION7_PROGRESS_FILE,
)

for status_name in [
    "PENDING",
    "IN_PROGRESS",
    "COMPLETED",
    "FAILED",
]:

    status_count = int(
        scenario_progress_df[
            "status"
        ]
        .astype(str)
        .eq(
            status_name
        )
        .sum()
    )

    print(
        f"{status_name:13s}:",
        status_count,
    )


# ------------------------------------------------------------
# 7A.5 — Import the real production state stack
# ------------------------------------------------------------

runtime_import_specs = {
    "battle_state_module":
        "src.battle_state",

    "legal_moves_module":
        "src.legal_moves",

    "adapters_module":
        "src.adapters",

    "policy_engine_module":
        "src.competitive.competitive_policy_engine",

    "policy_agent_module":
        "src.competitive.competitive_policy_agent",

    "search_module":
        "src.engine.advanced_search",
}


runtime_modules: dict[str, Any] = {}

module_rows = []


for runtime_name, module_name in runtime_import_specs.items():

    try:

        module = importlib.import_module(
            module_name
        )

        runtime_modules[
            runtime_name
        ] = module

        module_rows.append(
            {
                "runtime_name":
                    runtime_name,

                "module":
                    module_name,

                "loaded":
                    True,

                "file":
                    str(
                        getattr(
                            module,
                            "__file__",
                            "",
                        )
                    ),

                "error":
                    "",
            }
        )

    except Exception as exc:

        runtime_modules[
            runtime_name
        ] = None

        module_rows.append(
            {
                "runtime_name":
                    runtime_name,

                "module":
                    module_name,

                "loaded":
                    False,

                "file":
                    "",

                "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }
        )


module_status_df = pd.DataFrame(
    module_rows
)


print()
print("STATE-BASED PRODUCTION MODULES")
print("-" * 100)

display(
    module_status_df
)


assert module_status_df[
    "loaded"
].all(), (
    "One or more required production modules failed to import."
)


# ------------------------------------------------------------
# 7A.6 — Resolve required symbols explicitly
# ------------------------------------------------------------

battle_state_module = runtime_modules[
    "battle_state_module"
]

legal_moves_module = runtime_modules[
    "legal_moves_module"
]

adapters_module = runtime_modules[
    "adapters_module"
]

policy_engine_module = runtime_modules[
    "policy_engine_module"
]

policy_agent_module = runtime_modules[
    "policy_agent_module"
]

search_module = runtime_modules[
    "search_module"
]


BattleState = getattr(
    battle_state_module,
    "BattleState",
    None,
)

PlayerState = getattr(
    battle_state_module,
    "PlayerState",
    None,
)

PokemonState = getattr(
    battle_state_module,
    "PokemonState",
    None,
)

get_current_legal_moves = getattr(
    legal_moves_module,
    "get_current_legal_moves",
    None,
)

generate_moves_adapter = getattr(
    adapters_module,
    "generate_moves_adapter",
    None,
)

terminal_state_adapter = getattr(
    adapters_module,
    "terminal_state_adapter",
    None,
)

evaluate_state_adapter = getattr(
    adapters_module,
    "evaluate_state_adapter",
    None,
)

current_player_adapter = getattr(
    adapters_module,
    "current_player_adapter",
    None,
)

CompetitivePolicyEngine = getattr(
    policy_engine_module,
    "CompetitivePolicyEngine",
    None,
)

CompetitivePolicyAgent = getattr(
    policy_agent_module,
    "CompetitivePolicyAgent",
    None,
)

AdvancedSearchEngine = getattr(
    search_module,
    "AdvancedSearchEngine",
    None,
)


required_symbols = {
    "BattleState":
        BattleState,

    "PlayerState":
        PlayerState,

    "PokemonState":
        PokemonState,

    "get_current_legal_moves":
        get_current_legal_moves,

    "generate_moves_adapter":
        generate_moves_adapter,

    "terminal_state_adapter":
        terminal_state_adapter,

    "evaluate_state_adapter":
        evaluate_state_adapter,

    "current_player_adapter":
        current_player_adapter,

    "CompetitivePolicyEngine":
        CompetitivePolicyEngine,

    "CompetitivePolicyAgent":
        CompetitivePolicyAgent,

    "AdvancedSearchEngine":
        AdvancedSearchEngine,
}


def safe_signature(
    obj: Any,
) -> str:

    if obj is None:
        return "<missing>"

    try:
        return str(
            inspect.signature(
                obj
            )
        )

    except Exception:
        return "<signature unavailable>"


symbol_rows = []

for symbol_name, symbol_object in required_symbols.items():

    symbol_rows.append(
        {
            "symbol":
                symbol_name,

            "available":
                symbol_object is not None,

            "object_type":
                (
                    type(
                        symbol_object
                    ).__name__
                    if symbol_object is not None
                    else ""
                ),

            "module":
                (
                    getattr(
                        symbol_object,
                        "__module__",
                        "",
                    )
                    if symbol_object is not None
                    else ""
                ),

            "signature":
                safe_signature(
                    symbol_object
                ),
        }
    )


symbol_status_df = pd.DataFrame(
    symbol_rows
)


print()
print("STATE-BASED RUNTIME SYMBOLS")
print("-" * 100)

display(
    symbol_status_df
)


missing_required_symbols = (
    symbol_status_df.loc[
        ~symbol_status_df[
            "available"
        ],
        "symbol",
    ]
    .tolist()
)

assert not missing_required_symbols, (
    "Missing required state-runtime symbols: "
    f"{missing_required_symbols}"
)


# ------------------------------------------------------------
# 7A.7 — Validate callable contracts
# ------------------------------------------------------------

assert inspect.isclass(
    BattleState
)

assert inspect.isclass(
    PlayerState
)

assert inspect.isclass(
    PokemonState
)

assert callable(
    get_current_legal_moves
)

assert callable(
    generate_moves_adapter
)

assert callable(
    terminal_state_adapter
)

assert callable(
    evaluate_state_adapter
)

assert callable(
    current_player_adapter
)

assert inspect.isclass(
    CompetitivePolicyEngine
)

assert inspect.isclass(
    CompetitivePolicyAgent
)

assert inspect.isclass(
    AdvancedSearchEngine
)


battle_state_parameters = list(
    inspect.signature(
        BattleState
    ).parameters
)

required_battle_state_parameters = {
    "player",
    "opponent",
    "turn_number",
    "current_player",
}

assert required_battle_state_parameters.issubset(
    battle_state_parameters
), (
    "BattleState constructor does not match the "
    "expected production contract."
)


legal_move_parameters = list(
    inspect.signature(
        get_current_legal_moves
    ).parameters
)

assert legal_move_parameters, (
    "get_current_legal_moves must accept a battle state."
)


# ------------------------------------------------------------
# 7A.8 — Search and policy method validation
# ------------------------------------------------------------

required_policy_methods = {
    "choose_move",
    "score_moves",
}

available_policy_methods = {
    method_name
    for method_name in dir(
        CompetitivePolicyEngine
    )
    if callable(
        getattr(
            CompetitivePolicyEngine,
            method_name,
            None,
        )
    )
}

missing_policy_methods = (
    required_policy_methods
    - available_policy_methods
)

assert not missing_policy_methods, (
    "CompetitivePolicyEngine is missing: "
    f"{sorted(missing_policy_methods)}"
)


required_search_methods = {
    "search_depth",
    "search_for_time",
    "iterative_deepening",
}

available_search_methods = {
    method_name
    for method_name in dir(
        AdvancedSearchEngine
    )
    if callable(
        getattr(
            AdvancedSearchEngine,
            method_name,
            None,
        )
    )
}

missing_search_methods = (
    required_search_methods
    - available_search_methods
)

assert not missing_search_methods, (
    "AdvancedSearchEngine is missing: "
    f"{sorted(missing_search_methods)}"
)


contract_method_rows = []

for class_name, class_object, method_names in [
    (
        "CompetitivePolicyEngine",
        CompetitivePolicyEngine,
        sorted(
            required_policy_methods
        ),
    ),
    (
        "AdvancedSearchEngine",
        AdvancedSearchEngine,
        sorted(
            required_search_methods
        ),
    ),
]:

    for method_name in method_names:

        method_object = getattr(
            class_object,
            method_name,
        )

        contract_method_rows.append(
            {
                "class":
                    class_name,

                "method":
                    method_name,

                "signature":
                    safe_signature(
                        method_object
                    ),

                "status":
                    "PASS",
            }
        )


contract_method_df = pd.DataFrame(
    contract_method_rows
)


print()
print("POLICY AND SEARCH CONTRACTS")
print("-" * 100)

display(
    contract_method_df
)


# ------------------------------------------------------------
# 7A.9 — Determine state-construction readiness
# ------------------------------------------------------------

state_factory_candidates = {
    name:
        globals().get(
            name
        )

    for name in [
        "build_battle_state",
        "create_battle_state",
        "make_battle_state",
        "scenario_to_battle_state",
        "card_pair_to_battle_state",
    ]
}

state_factory_candidates = {
    name:
        candidate

    for name, candidate in state_factory_candidates.items()
    if callable(
        candidate
    )
}


existing_state_candidates = {
    name:
        globals().get(
            name
        )

    for name in [
        "battle_state",
        "initial_state",
        "sample_battle_state",
        "example_battle_state",
    ]
}

existing_state_candidates = {
    name:
        candidate

    for name, candidate in existing_state_candidates.items()
    if candidate is not None
    and isinstance(
        candidate,
        BattleState,
    )
}


if state_factory_candidates:

    state_construction_status = (
        "STATE_FACTORY_AVAILABLE"
    )

elif existing_state_candidates:

    state_construction_status = (
        "EXISTING_STATE_AVAILABLE"
    )

else:

    state_construction_status = (
        "STATE_FACTORY_REQUIRED"
    )


print()
print("STATE-CONSTRUCTION READINESS")
print("-" * 100)

print(
    "Status                  :",
    state_construction_status,
)

print(
    "Factory candidates      :",
    sorted(
        state_factory_candidates
    ),
)

print(
    "Existing state objects  :",
    sorted(
        existing_state_candidates
    ),
)


# ------------------------------------------------------------
# 7A.10 — Optional legal-move smoke test
# ------------------------------------------------------------

legal_move_smoke_test = {
    "executed":
        False,

    "passed":
        False,

    "state_source":
        None,

    "legal_move_count":
        None,

    "error":
        None,
}


if existing_state_candidates:

    state_name = next(
        iter(
            existing_state_candidates
        )
    )

    sample_state = existing_state_candidates[
        state_name
    ]

    legal_move_smoke_test[
        "executed"
    ] = True

    legal_move_smoke_test[
        "state_source"
    ] = state_name

    try:

        sample_legal_moves = get_current_legal_moves(
            sample_state
        )

        sample_legal_moves = list(
            sample_legal_moves
        )

        legal_move_smoke_test[
            "passed"
        ] = True

        legal_move_smoke_test[
            "legal_move_count"
        ] = len(
            sample_legal_moves
        )

    except Exception as exc:

        legal_move_smoke_test[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )


print()
print("LEGAL-MOVE SMOKE TEST")
print("-" * 100)

for key, value in legal_move_smoke_test.items():
    print(
        f"{key:20s}: {value}"
    )


# ------------------------------------------------------------
# 7A.11 — Save corrected runtime contract
# ------------------------------------------------------------

runtime_contract = {
    "architecture":
        "STATE_BASED_DECISION_PIPELINE",

    "replay_level":
        replay_contract.get(
            "replay_level"
        ),

    "scenario_count":
        int(
            len(
                scenario_queue_df
            )
        ),

    "pipeline":
        [
            "BattleState",
            "get_current_legal_moves",
            "CompetitivePolicyEngine",
            "AdvancedSearchEngine",
        ],

    "simulator_required":
        False,

    "simulator_available":
        False,

    "state_construction_status":
        state_construction_status,

    "state_factory_candidates":
        sorted(
            state_factory_candidates
        ),

    "existing_state_candidates":
        sorted(
            existing_state_candidates
        ),

    "module_status":
        module_status_df.to_dict(
            orient="records"
        ),

    "symbol_status":
        symbol_status_df.to_dict(
            orient="records"
        ),

    "method_contracts":
        contract_method_df.to_dict(
            orient="records"
        ),

    "legal_move_smoke_test":
        legal_move_smoke_test,

    "next_stage":
        (
            "BUILD_SCENARIO_STATE_FACTORY"
            if state_construction_status
            == "STATE_FACTORY_REQUIRED"
            else
            "GENERATE_CANDIDATE_STATES"
        ),
}


with open(
    SECTION7_RUNTIME_CONTRACT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        runtime_contract,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 7A.12 — Final validation
# ------------------------------------------------------------

assert SECTION7_REPORT_DIR.exists()

assert SECTION7_TRACE_DIR.exists()

assert SECTION7_CHECKPOINT_DIR.exists()

assert SECTION7_PROGRESS_FILE.exists()

assert SECTION7_QUEUE_FILE.exists()

assert SECTION7_RUNTIME_CONTRACT_FILE.exists()

assert len(
    scenario_queue_df
) == len(
    hard_example_manifest_df
)

assert scenario_progress_df[
    "source_scenario_id"
].is_unique

assert runtime_contract[
    "architecture"
] == "STATE_BASED_DECISION_PIPELINE"

assert runtime_contract[
    "simulator_required"
] is False


print()
print("CORRECTED SECTION 7A SUMMARY")
print("-" * 100)

print(
    "Architecture             :",
    runtime_contract[
        "architecture"
    ],
)

print(
    "Simulator required       :",
    runtime_contract[
        "simulator_required"
    ],
)

print(
    "Scenarios queued         :",
    runtime_contract[
        "scenario_count"
    ],
)

print(
    "State construction       :",
    state_construction_status,
)

print(
    "Next stage               :",
    runtime_contract[
        "next_stage"
    ],
)

print()
print("SAVED REPORTS")
print("-" * 100)

print(
    SECTION7_QUEUE_FILE
)

print(
    SECTION7_PROGRESS_FILE
)

print(
    SECTION7_RUNTIME_CONTRACT_FILE
)

print()
print(
    "✅ SECTION 7A STATE-BASED RUNTIME CONTRACT PASSED"
)


# In[9]:


import inspect

print(inspect.signature(BattleState))


# In[10]:


print(inspect.signature(PlayerState))


# In[11]:


print(inspect.signature(PokemonState))


# ## Section 7B — Manifest-to-BattleState Factory
# 
# #### Section 7B converts each prioritized hard-example scenario into a valid BattleState`.
# 
# #### Each manifest row provides the player card, opponent card, evaluated side, and original match metadata. The factory resolves both cards from the production card dataset and creates:
# 
# - one active `PokemonState` for each side,
# - one `PlayerState` for each active Pokémon,
# - an empty bench,
# - six remaining prize cards,
# - the default hand size,
# - and a deterministic starting turn.
# 
# #### The resulting state is then validated with the production legal-move generator. Only states that can be constructed and evaluated successfully are accepted for later hard-example mining.

# In[12]:


# ============================================================
# SECTION 7B
# MANIFEST-TO-BATTLESTATE FACTORY
# FULL CORRECTED VERSION
# ============================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 7B — MANIFEST-TO-BATTLESTATE FACTORY")
print("=" * 100)


# ------------------------------------------------------------
# 7B.1 — Output paths
# ------------------------------------------------------------

SECTION7B_FACTORY_REPORT_FILE = (
    SECTION7_REPORT_DIR
    / "section7b_state_factory_report.json"
)

SECTION7B_CARD_RESOLUTION_FILE = (
    SECTION7_REPORT_DIR
    / "section7b_card_resolution.csv"
)

SECTION7B_SMOKE_TEST_FILE = (
    SECTION7_REPORT_DIR
    / "section7b_state_smoke_tests.csv"
)


# ------------------------------------------------------------
# 7B.2 — General normalization helpers
# ------------------------------------------------------------

def normalize_lookup_text(
    value: Any,
) -> str:
    """
    Normalize IDs, names, and side labels for stable matching.
    """

    if value is None:
        return ""

    text = str(
        value
    ).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )

    return text


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """
    Return the first dataframe column matching any candidate name.
    """

    normalized_columns = {
        normalize_lookup_text(column):
            column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = (
            normalize_lookup_text(
                candidate
            )
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[
                normalized_candidate
            ]

    return None


def safe_numeric(
    value: Any,
    default: float,
) -> float:
    """
    Convert a value to finite float form safely.
    """

    try:
        numeric_value = float(
            value
        )

        if np.isfinite(
            numeric_value
        ):
            return numeric_value

    except Exception:
        pass

    return float(
        default
    )


# ------------------------------------------------------------
# 7B.3 — Collect manifest card references
# ------------------------------------------------------------

required_manifest_cards = sorted(
    {
        str(value).strip()
        for column_name in [
            "player_card",
            "opponent_card",
        ]
        if column_name in scenario_queue_df.columns
        for value in (
            scenario_queue_df[
                column_name
            ]
            .dropna()
            .tolist()
        )
        if str(value).strip()
    }
)

required_manifest_cards_normalized = {
    normalize_lookup_text(
        value
    )
    for value in required_manifest_cards
}


print()
print("REQUIRED MANIFEST CARDS")
print("-" * 100)

print(
    required_manifest_cards
)


assert required_manifest_cards, (
    "No required manifest cards were found in scenario_queue_df."
)


# ------------------------------------------------------------
# 7B.4 — Locate likely card datasets
# ------------------------------------------------------------

candidate_card_files = sorted(
    {
        path.resolve()
        for path in PROJECT_ROOT.rglob(
            "*.csv"
        )
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in [
                "card",
                "pokemon",
                "metadata",
            ]
        )
        and "reports" not in {
            part.lower()
            for part in path.parts
        }
        and "section7" not in {
            part.lower()
            for part in path.parts
        }
        and not path.name.lower().startswith(
            "section"
        )
    }
)


assert candidate_card_files, (
    "No likely card-data CSV files were found under PROJECT_ROOT."
)


dataset_candidate_rows = []
loaded_dataset_candidates = []


for candidate_path in candidate_card_files:
    try:
        candidate_df = pd.read_csv(
            candidate_path,
            low_memory=False,
        )

    except Exception as exc:
        dataset_candidate_rows.append(
            {
                "path":
                    str(candidate_path),

                "rows":
                    0,

                "columns":
                    0,

                "best_name_column":
                    None,

                "exact_matches":
                    0,

                "partial_matches":
                    0,

                "total_matches":
                    0,

                "error":
                    f"{type(exc).__name__}: {exc}",
            }
        )

        continue

    best_column = None
    best_exact_matches = set()
    best_partial_matches = set()

    for column_name in candidate_df.columns:
        column_series = candidate_df[
            column_name
        ]

        if not (
            pd.api.types.is_object_dtype(
                column_series
            )
            or pd.api.types.is_string_dtype(
                column_series
            )
        ):
            continue

        normalized_values = (
            column_series
            .dropna()
            .astype(str)
            .map(
                normalize_lookup_text
            )
        )

        normalized_value_set = set(
            normalized_values
        )

        exact_matches = (
            required_manifest_cards_normalized
            & normalized_value_set
        )

        partial_matches = {
            required_name
            for required_name
            in required_manifest_cards_normalized
            if any(
                required_name in dataset_value
                or dataset_value in required_name
                for dataset_value
                in normalized_value_set
                if dataset_value
            )
        }

        current_total = len(
            exact_matches
            | partial_matches
        )

        best_total = len(
            best_exact_matches
            | best_partial_matches
        )

        if current_total > best_total:
            best_column = column_name
            best_exact_matches = (
                exact_matches
            )
            best_partial_matches = (
                partial_matches
            )

    total_matches = (
        best_exact_matches
        | best_partial_matches
    )

    dataset_candidate_rows.append(
        {
            "path":
                str(candidate_path),

            "rows":
                int(
                    len(candidate_df)
                ),

            "columns":
                int(
                    len(
                        candidate_df.columns
                    )
                ),

            "best_name_column":
                best_column,

            "exact_matches":
                int(
                    len(
                        best_exact_matches
                    )
                ),

            "partial_matches":
                int(
                    len(
                        best_partial_matches
                    )
                ),

            "total_matches":
                int(
                    len(
                        total_matches
                    )
                ),

            "error":
                "",
        }
    )

    loaded_dataset_candidates.append(
        {
            "path":
                candidate_path,

            "dataframe":
                candidate_df,

            "best_name_column":
                best_column,

            "exact_matches":
                best_exact_matches,

            "partial_matches":
                best_partial_matches,

            "total_matches":
                total_matches,
        }
    )


dataset_candidate_df = (
    pd.DataFrame(
        dataset_candidate_rows
    )
    .sort_values(
        [
            "total_matches",
            "exact_matches",
            "rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("CARD DATASET CANDIDATES")
print("-" * 100)

display(
    dataset_candidate_df.head(
        30
    )
)


# ------------------------------------------------------------
# 7B.5 — Select production card dataset
# ------------------------------------------------------------

valid_dataset_candidates = [
    candidate
    for candidate
    in loaded_dataset_candidates
    if candidate[
        "best_name_column"
    ] is not None
    and len(
        candidate[
            "total_matches"
        ]
    ) > 0
    and len(
        candidate[
            "dataframe"
        ]
    ) > 100
    and first_existing_column(
        candidate[
            "dataframe"
        ],
        [
            "hp",
            "hit_points",
            "hit points",
            "health",
            "maximum_hp",
            "max_hp",
        ],
    ) is not None
]


assert valid_dataset_candidates, (
    "No discovered card dataset contains the required "
    "manifest Pokémon names and an HP column."
)


selected_dataset = max(
    valid_dataset_candidates,
    key=lambda candidate: (
        len(
            candidate[
                "total_matches"
            ]
        ),
        len(
            candidate[
                "exact_matches"
            ]
        ),
        len(
            candidate[
                "dataframe"
            ]
        ),
    ),
)


selected_card_file = (
    selected_dataset[
        "path"
    ]
)

card_data_df = (
    selected_dataset[
        "dataframe"
    ]
    .copy()
)

selected_card_dataframe_name = str(
    selected_card_file
)

card_name_column = (
    selected_dataset[
        "best_name_column"
    ]
)


assert "reports" not in {
    part.lower()
    for part in selected_card_file.parts
}, (
    "Section 7B selected a generated report instead "
    "of the production card dataset."
)

assert len(
    card_data_df
) > 100, (
    "Selected card dataset is unexpectedly small."
)


# ------------------------------------------------------------
# 7B.6 — Detect required card columns
# ------------------------------------------------------------

card_id_column = first_existing_column(
    card_data_df,
    [
        "card_id",
        "card id",
        "id",
        "identifier",
        "card_identifier",
        "card_number",
        "local_id",
    ],
)

card_hp_column = first_existing_column(
    card_data_df,
    [
        "hp",
        "hit_points",
        "hit points",
        "health",
        "maximum_hp",
        "max_hp",
    ],
)

move_name_column = first_existing_column(
    card_data_df,
    [
        "move name",
        "move_name",
        "attack name",
        "attack_name",
    ],
)

damage_numeric_column = first_existing_column(
    card_data_df,
    [
        "damage_numeric",
        "damage numeric",
        "numeric_damage",
    ],
)

energy_cost_column = first_existing_column(
    card_data_df,
    [
        "energy_cost",
        "energy cost",
        "cost_numeric",
    ],
)

effect_column = first_existing_column(
    card_data_df,
    [
        "effect explanation",
        "effect_explanation",
        "effect",
        "description",
    ],
)


assert card_name_column is not None, (
    "No card-name column could be identified."
)

assert card_hp_column is not None, (
    "No HP column could be identified."
)

assert move_name_column is not None, (
    "No move-name column could be identified."
)


print()
print("SELECTED CARD DATASET")
print("-" * 100)

print(
    "Selected file       :",
    selected_card_file,
)

print(
    "Rows                :",
    len(
        card_data_df
    ),
)

print(
    "Columns             :",
    len(
        card_data_df.columns
    ),
)

print(
    "Card-name column    :",
    card_name_column,
)

print(
    "Card-ID column      :",
    card_id_column,
)

print(
    "HP column           :",
    card_hp_column,
)

print(
    "Move-name column    :",
    move_name_column,
)

print(
    "Damage column       :",
    damage_numeric_column,
)

print(
    "Energy-cost column  :",
    energy_cost_column,
)

print(
    "Effect column       :",
    effect_column,
)


# ------------------------------------------------------------
# 7B.7 — Build lookup dataframe
# ------------------------------------------------------------

card_lookup_df = (
    card_data_df.copy()
)

card_lookup_df[
    "__normalized_card_name"
] = (
    card_lookup_df[
        card_name_column
    ]
    .fillna("")
    .astype(str)
    .map(
        normalize_lookup_text
    )
)


if card_id_column is not None:
    card_lookup_df[
        "__normalized_card_id"
    ] = (
        card_lookup_df[
            card_id_column
        ]
        .fillna("")
        .astype(str)
        .map(
            normalize_lookup_text
        )
    )

else:
    card_lookup_df[
        "__normalized_card_id"
    ] = ""


# ------------------------------------------------------------
# 7B.8 — Resolve cards and build nested attack lists
# ------------------------------------------------------------

def resolve_card_record(
    card_reference: Any,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any],
]:
    """
    Resolve a manifest card reference and return one card record
    containing a nested 'attacks' list.
    """

    normalized_reference = (
        normalize_lookup_text(
            card_reference
        )
    )

    resolution = {
        "requested_reference":
            card_reference,

        "normalized_reference":
            normalized_reference,

        "resolved":
            False,

        "match_type":
            None,

        "matched_row_index":
            None,

        "matched_card_id":
            None,

        "matched_card_name":
            None,

        "candidate_count":
            0,

        "attack_count":
            0,

        "attack_names":
            [],
    }

    if not normalized_reference:
        return (
            None,
            resolution,
        )


    matches = (
        card_lookup_df.iloc[
            0:0
        ].copy()
    )


    # Exact card ID
    if card_id_column is not None:
        exact_id_matches = (
            card_lookup_df.loc[
                card_lookup_df[
                    "__normalized_card_id"
                ].eq(
                    normalized_reference
                )
            ]
        )

        if not exact_id_matches.empty:
            matches = (
                exact_id_matches
            )

            resolution[
                "match_type"
            ] = "EXACT_CARD_ID"


    # Exact card name
    if matches.empty:
        exact_name_matches = (
            card_lookup_df.loc[
                card_lookup_df[
                    "__normalized_card_name"
                ].eq(
                    normalized_reference
                )
            ]
        )

        if not exact_name_matches.empty:
            matches = (
                exact_name_matches
            )

            resolution[
                "match_type"
            ] = "EXACT_CARD_NAME"


    # Bidirectional partial match
    if matches.empty:
        partial_mask = (
            card_lookup_df[
                "__normalized_card_name"
            ]
            .fillna("")
            .astype(str)
            .map(
                lambda value: (
                    bool(value)
                    and (
                        normalized_reference
                        in value
                        or value
                        in normalized_reference
                    )
                )
            )
        )

        if card_id_column is not None:
            partial_mask |= (
                card_lookup_df[
                    "__normalized_card_id"
                ]
                .fillna("")
                .astype(str)
                .map(
                    lambda value: (
                        bool(value)
                        and (
                            normalized_reference
                            in value
                            or value
                            in normalized_reference
                        )
                    )
                )
            )

        matches = card_lookup_df.loc[
            partial_mask
        ]

        if not matches.empty:
            resolution[
                "match_type"
            ] = "PARTIAL_MATCH"


    resolution[
        "candidate_count"
    ] = int(
        len(
            matches
        )
    )


    if matches.empty:
        return (
            None,
            resolution,
        )


    # Prefer the shortest matching card name.
    selected_index = (
        matches[
            "__normalized_card_name"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .sort_values()
        .index[0]
    )

    selected_row = (
        card_data_df.loc[
            selected_index
        ]
    )

    card_record = (
        selected_row.to_dict()
    )


    # Collect all rows belonging to the selected physical card.
    if card_id_column is not None:
        selected_card_id = (
            selected_row.get(
                card_id_column
            )
        )

        selected_card_rows = (
            card_data_df.loc[
                card_data_df[
                    card_id_column
                ]
                .astype(str)
                .eq(
                    str(
                        selected_card_id
                    )
                )
            ]
            .copy()
        )

    else:
        selected_card_name = (
            selected_row.get(
                card_name_column
            )
        )

        selected_card_rows = (
            card_data_df.loc[
                card_data_df[
                    card_name_column
                ]
                .astype(str)
                .eq(
                    str(
                        selected_card_name
                    )
                )
            ]
            .copy()
        )


    attacks = []

    for _, attack_row in (
        selected_card_rows.iterrows()
    ):
        move_name_value = (
            attack_row.get(
                move_name_column
            )
        )

        if pd.isna(
            move_name_value
        ):
            continue

        move_name_text = str(
            move_name_value
        ).strip()

        if not move_name_text:
            continue

        normalized_move_name = (
            move_name_text
            .strip()
            .lower()
        )

        # Exclude Pokémon abilities.
        if (
            "[ability]"
            in normalized_move_name
            or normalized_move_name.startswith(
                "ability"
            )
        ):
            continue


        attack_damage = safe_numeric(
            attack_row.get(
                damage_numeric_column
            )
            if damage_numeric_column
            is not None
            else 0,
            default=0.0,
        )

        attack_energy_cost = int(
            max(
                0,
                safe_numeric(
                    attack_row.get(
                        energy_cost_column
                    )
                    if energy_cost_column
                    is not None
                    else 0,
                    default=0.0,
                ),
            )
        )

        attack_effect = (
            attack_row.get(
                effect_column
            )
            if effect_column
            is not None
            else None
        )

        if pd.isna(
            attack_effect
        ):
            attack_effect = None

        attacks.append(
            {
                "Move Name":
                    move_name_text,

                "damage_numeric":
                    float(
                        attack_damage
                    ),

                "energy_cost":
                    int(
                        attack_energy_cost
                    ),

                "Effect Explanation":
                    attack_effect,
            }
        )


    # Remove duplicate attacks.
    unique_attacks = []
    seen_attack_keys = set()

    for attack in attacks:
        attack_key = (
            attack[
                "Move Name"
            ]
            .strip()
            .lower(),

            float(
                attack[
                    "damage_numeric"
                ]
            ),

            int(
                attack[
                    "energy_cost"
                ]
            ),
        )

        if attack_key in seen_attack_keys:
            continue

        seen_attack_keys.add(
            attack_key
        )

        unique_attacks.append(
            attack
        )

    attacks = (
        unique_attacks
    )

    card_record[
        "attacks"
    ] = attacks


    resolution[
        "resolved"
    ] = True

    resolution[
        "matched_row_index"
    ] = (
        int(
            selected_index
        )
        if isinstance(
            selected_index,
            (
                int,
                np.integer,
            ),
        )
        else str(
            selected_index
        )
    )

    if card_id_column is not None:
        resolution[
            "matched_card_id"
        ] = (
            selected_row.get(
                card_id_column
            )
        )

    resolution[
        "matched_card_name"
    ] = (
        selected_row.get(
            card_name_column
        )
    )

    resolution[
        "attack_count"
    ] = int(
        len(
            attacks
        )
    )

    resolution[
        "attack_names"
    ] = [
        attack[
            "Move Name"
        ]
        for attack in attacks
    ]


    return (
        card_record,
        resolution,
    )


# ------------------------------------------------------------
# 7B.9 — Starting HP helper
# ------------------------------------------------------------

def card_starting_hp(
    card_record: Mapping[
        str,
        Any,
    ],
    default_hp: float = 100.0,
) -> float:
    """
    Return a valid positive starting HP value.
    """

    hp_value = card_record.get(
        card_hp_column
    )

    starting_hp = safe_numeric(
        hp_value,
        default=default_hp,
    )

    if starting_hp <= 0:
        starting_hp = float(
            default_hp
        )

    return float(
        starting_hp
    )


# ------------------------------------------------------------
# 7B.10 — Normalize active side
# ------------------------------------------------------------

def normalize_current_player(
    side_value: Any,
) -> str:
    """
    Normalize side references into Player or Opponent.
    """

    normalized_side = (
        normalize_lookup_text(
            side_value
        )
    )

    player_aliases = {
        "player",
        "agent",
        "evaluatedagent",
        "competitive",
        "p1",
        "first",
    }

    opponent_aliases = {
        "opponent",
        "baseline",
        "enemy",
        "p2",
        "second",
    }

    if normalized_side in opponent_aliases:
        return "Opponent"

    if "opponent" in normalized_side:
        return "Opponent"

    if normalized_side in player_aliases:
        return "Player"

    return "Player"


# ------------------------------------------------------------
# 7B.11 — Construct battle state
# ------------------------------------------------------------

def build_battle_state(
    scenario: Mapping[
        str,
        Any,
    ],
    *,
    turn_number: int = 1,
    player_energy: int = 0,
    opponent_energy: int = 0,
    player_damage: float = 0.0,
    opponent_damage: float = 0.0,
    prize_cards_remaining: int = 6,
    hand_size: int = 7,
) -> tuple[
    Any,
    dict[str, Any],
]:
    """
    Build one simulator-compatible BattleState.
    """

    player_reference = (
        scenario.get(
            "player_card"
        )
    )

    opponent_reference = (
        scenario.get(
            "opponent_card"
        )
    )


    (
        player_card_record,
        player_resolution,
    ) = resolve_card_record(
        player_reference
    )

    (
        opponent_card_record,
        opponent_resolution,
    ) = resolve_card_record(
        opponent_reference
    )


    if player_card_record is None:
        raise LookupError(
            "Could not resolve player card: "
            f"{player_reference}"
        )

    if opponent_card_record is None:
        raise LookupError(
            "Could not resolve opponent card: "
            f"{opponent_reference}"
        )


    player_max_hp = (
        card_starting_hp(
            player_card_record
        )
    )

    opponent_max_hp = (
        card_starting_hp(
            opponent_card_record
        )
    )


    player_current_hp = max(
        1.0,
        player_max_hp
        - float(
            player_damage
        ),
    )

    opponent_current_hp = max(
        1.0,
        opponent_max_hp
        - float(
            opponent_damage
        ),
    )


    player_pokemon = PokemonState(
        card=player_card_record,
        current_hp=player_current_hp,
        attached_energy=int(
            player_energy
        ),
        status=None,
        damage=float(
            player_damage
        ),
        is_active=True,
    )

    opponent_pokemon = PokemonState(
        card=opponent_card_record,
        current_hp=opponent_current_hp,
        attached_energy=int(
            opponent_energy
        ),
        status=None,
        damage=float(
            opponent_damage
        ),
        is_active=True,
    )


    player_state = PlayerState(
        active=player_pokemon,
        bench=[],
        prize_cards_remaining=int(
            prize_cards_remaining
        ),
        hand_size=int(
            hand_size
        ),
    )

    opponent_state = PlayerState(
        active=opponent_pokemon,
        bench=[],
        prize_cards_remaining=int(
            prize_cards_remaining
        ),
        hand_size=int(
            hand_size
        ),
    )


    current_player = (
        normalize_current_player(
            scenario.get(
                "normalized_side",
                scenario.get(
                    "current_side",
                    scenario.get(
                        "source_side",
                        "Player",
                    ),
                ),
            )
        )
    )


    battle_state = BattleState(
        player=player_state,
        opponent=opponent_state,
        turn_number=max(
            1,
            int(
                turn_number
            ),
        ),
        current_player=current_player,
    )


    metadata = {
        "source_scenario_id":
            scenario.get(
                "source_scenario_id"
            ),

        "player_reference":
            player_reference,

        "opponent_reference":
            opponent_reference,

        "player_resolution":
            player_resolution,

        "opponent_resolution":
            opponent_resolution,

        "player_starting_hp":
            player_max_hp,

        "opponent_starting_hp":
            opponent_max_hp,

        "player_current_hp":
            player_current_hp,

        "opponent_current_hp":
            opponent_current_hp,

        "player_energy":
            int(
                player_energy
            ),

        "opponent_energy":
            int(
                opponent_energy
            ),

        "turn_number":
            max(
                1,
                int(
                    turn_number
                ),
            ),

        "current_player":
            current_player,
    }


    return (
        battle_state,
        metadata,
    )


# Make factory visible to later notebook sections.
scenario_to_battle_state = (
    build_battle_state
)


# ------------------------------------------------------------
# 7B.12 — Validate required card resolution
# ------------------------------------------------------------

required_card_resolution_rows = []


for card_name in required_manifest_cards:
    (
        card_record,
        resolution,
    ) = resolve_card_record(
        card_name
    )

    required_card_resolution_rows.append(
        {
            "requested_card":
                card_name,

            "resolved":
                card_record is not None,

            "match_type":
                resolution.get(
                    "match_type"
                ),

            "candidate_count":
                resolution.get(
                    "candidate_count"
                ),

            "matched_card_id":
                resolution.get(
                    "matched_card_id"
                ),

            "matched_card_name":
                resolution.get(
                    "matched_card_name"
                ),

            "attack_count":
                resolution.get(
                    "attack_count"
                ),

            "attack_names":
                json.dumps(
                    resolution.get(
                        "attack_names",
                        [],
                    ),
                    ensure_ascii=False,
                ),
        }
    )


required_card_resolution_df = (
    pd.DataFrame(
        required_card_resolution_rows
    )
)


resolved_required_cards = (
    required_card_resolution_df.loc[
        required_card_resolution_df[
            "resolved"
        ],
        "requested_card",
    ]
    .tolist()
)

unresolved_required_cards = (
    required_card_resolution_df.loc[
        ~required_card_resolution_df[
            "resolved"
        ],
        "requested_card",
    ]
    .tolist()
)

cards_without_attacks = (
    required_card_resolution_df.loc[
        required_card_resolution_df[
            "attack_count"
        ]
        .fillna(0)
        .le(0),
        "requested_card",
    ]
    .tolist()
)


print()
print("CARD RESOLUTION")
print("-" * 100)

display(
    required_card_resolution_df
)


assert not unresolved_required_cards, (
    "Some manifest card references could not be resolved: "
    f"{unresolved_required_cards}"
)

assert not cards_without_attacks, (
    "Some resolved cards contain no usable attacks: "
    f"{cards_without_attacks}"
)


# ------------------------------------------------------------
# 7B.13 — State smoke tests
# ------------------------------------------------------------

smoke_test_rows = []

smoke_test_source_df = (
    scenario_queue_df.head(
        min(
            20,
            len(
                scenario_queue_df
            ),
        )
    )
    .copy()
)


for _, scenario_row in (
    smoke_test_source_df.iterrows()
):
    scenario_record = (
        scenario_row.to_dict()
    )

    smoke_player_energy = int(
        safe_numeric(
            scenario_record.get(
                "player_energy",
                3,
            ),
            default=3,
        )
    )

    smoke_opponent_energy = int(
        safe_numeric(
            scenario_record.get(
                "opponent_energy",
                3,
            ),
            default=3,
        )
    )

    battle_state, metadata = (
        build_battle_state(
            scenario_record,
            turn_number=int(
                safe_numeric(
                    scenario_record.get(
                        "turn_number",
                        5,
                    ),
                    default=5,
                )
            ),
            player_energy=smoke_player_energy,
            opponent_energy=smoke_opponent_energy,
            player_damage=safe_numeric(
                scenario_record.get(
                    "player_damage",
                    0.0,
                ),
                default=0.0,
            ),
            opponent_damage=safe_numeric(
                scenario_record.get(
                    "opponent_damage",
                    0.0,
                ),
                default=0.0,
            ),
            prize_cards_remaining=int(
                safe_numeric(
                    scenario_record.get(
                        "prize_cards_remaining",
                        4,
                    ),
                    default=4,
                )
            ),
            hand_size=int(
                safe_numeric(
                    scenario_record.get(
                        "hand_size",
                        5,
                    ),
                    default=5,
                )
            ),
        )
    )

    legal_moves = (
        get_current_legal_moves(
            battle_state
        )
    )

    legal_move_names = [
        (
            move.get(
                "name",
                "Unknown Move",
            )
            if isinstance(
                move,
                dict,
            )
            else str(
                move
            )
        )
        for move in legal_moves
    ]

    smoke_test_rows.append(
        {
            "source_scenario_id":
                scenario_record.get(
                    "source_scenario_id"
                ),

            "player_card":
                scenario_record.get(
                    "player_card"
                ),

            "opponent_card":
                scenario_record.get(
                    "opponent_card"
                ),

            "current_player":
                battle_state.current_player,

            "player_energy":
                smoke_player_energy,

            "opponent_energy":
                smoke_opponent_energy,

            "player_attack_count":
                metadata[
                    "player_resolution"
                ].get(
                    "attack_count",
                    0,
                ),

            "opponent_attack_count":
                metadata[
                    "opponent_resolution"
                ].get(
                    "attack_count",
                    0,
                ),

            "legal_moves_generated":
                bool(
                    legal_moves
                ),

            "legal_move_count":
                int(
                    len(
                        legal_moves
                    )
                ),

            "legal_move_names":
                json.dumps(
                    legal_move_names,
                    ensure_ascii=False,
                ),

            "only_pass":
                bool(
                    len(
                        legal_move_names
                    ) == 1
                    and legal_move_names[
                        0
                    ].strip().lower()
                    == "pass"
                ),
        }
    )


state_smoke_test_df = (
    pd.DataFrame(
        smoke_test_rows
    )
)


print()
print("STATE SMOKE TESTS")
print("-" * 100)

display(
    state_smoke_test_df
)


assert not state_smoke_test_df.empty

assert state_smoke_test_df[
    "legal_moves_generated"
].all(), (
    "At least one battle state generated no legal-move output."
)

assert state_smoke_test_df[
    "player_attack_count"
].gt(0).all(), (
    "At least one player card resolved without attacks."
)

assert state_smoke_test_df[
    "opponent_attack_count"
].gt(0).all(), (
    "At least one opponent card resolved without attacks."
)

assert (
    ~state_smoke_test_df[
        "only_pass"
    ]
).any(), (
    "Every smoke-test state still generated only Pass."
)


# ------------------------------------------------------------
# 7B.14 — Save reports
# ------------------------------------------------------------

required_card_resolution_df.to_csv(
    SECTION7B_CARD_RESOLUTION_FILE,
    index=False,
)

state_smoke_test_df.to_csv(
    SECTION7B_SMOKE_TEST_FILE,
    index=False,
)


section7b_factory_report = {
    "section":
        "7B",

    "selected_card_file":
        str(
            selected_card_file
        ),

    "selected_card_rows":
        int(
            len(
                card_data_df
            )
        ),

    "card_name_column":
        card_name_column,

    "card_id_column":
        card_id_column,

    "card_hp_column":
        card_hp_column,

    "move_name_column":
        move_name_column,

    "damage_numeric_column":
        damage_numeric_column,

    "energy_cost_column":
        energy_cost_column,

    "effect_column":
        effect_column,

    "required_card_count":
        int(
            len(
                required_manifest_cards
            )
        ),

    "resolved_required_card_count":
        int(
            required_card_resolution_df[
                "resolved"
            ].sum()
        ),

    "cards_without_attacks":
        cards_without_attacks,

    "smoke_test_count":
        int(
            len(
                state_smoke_test_df
            )
        ),

    "smoke_tests_with_non_pass_moves":
        int(
            (
                ~state_smoke_test_df[
                    "only_pass"
                ]
            ).sum()
        ),
}


with open(
    SECTION7B_FACTORY_REPORT_FILE,
    "w",
    encoding="utf-8",
) as report_file:
    json.dump(
        section7b_factory_report,
        report_file,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


print()
print("SAVED SECTION 7B REPORTS")
print("-" * 100)

print(
    SECTION7B_FACTORY_REPORT_FILE
)

print(
    SECTION7B_CARD_RESOLUTION_FILE
)

print(
    SECTION7B_SMOKE_TEST_FILE
)

print()
print(
    "✅ SECTION 7B MANIFEST-TO-BATTLESTATE FACTORY PASSED"
)


# ## Section 7C — Deterministic State Variant Generation
# 
# #### Section 7C expands each prioritized hard-example scenario into multiple, reproducible battle-state variants.
# 
# The variant generator changes strategically relevant state conditions:
# 
# - current turn,
# - attached energy,
# - accumulated damage,
# - cards remaining in hand,
# - prize-card pressure,
# - and current player.
# 
# #### Most variants preserve the original evaluated side. A smaller number are counterfactual side variants used to measure side sensitivity.
# 
# Each generated state must:
# 
# 1. construct successfully,
# 2. be nonterminal,
# 3. produce at least one legal move,
# 4. retain its source scenario metadata,
# 5. and be reproducible from a deterministic variant ID.
# 
# #### The generated records remain unlabeled. Search-guided move labeling occurs in Section 8.

# In[13]:


# ============================================================
# SECTION 7C
# DETERMINISTIC STATE VARIANT GENERATION
# ============================================================

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 7C — DETERMINISTIC STATE VARIANT GENERATION")
print("=" * 100)


# ------------------------------------------------------------
# 7C.1 — Output paths
# ------------------------------------------------------------

SECTION7C_VARIANT_DIR = (
    SECTION7_TRACE_DIR
    / "state_variants"
)

SECTION7C_VARIANT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION7C_VARIANT_MANIFEST_FILE = (
    SECTION7_REPORT_DIR
    / "section7c_state_variant_manifest.csv"
)

SECTION7C_VARIANT_SUMMARY_FILE = (
    SECTION7_REPORT_DIR
    / "section7c_state_variant_summary.json"
)

SECTION7C_SCENARIO_SUMMARY_FILE = (
    SECTION7_REPORT_DIR
    / "section7c_scenario_generation_summary.csv"
)

SECTION7C_FAILURE_FILE = (
    SECTION7_REPORT_DIR
    / "section7c_variant_failures.csv"
)


# ------------------------------------------------------------
# 7C.2 — Generation configuration
# ------------------------------------------------------------

SECTION7C_BASE_SEED = 490700

SECTION7C_MINIMUM_LEGAL_MOVES = 1

SECTION7C_SKIP_COMPLETED_SCENARIOS = True

SECTION7C_OVERWRITE_EXISTING = False


# Ten deterministic variants per source scenario:
#
# Eight preserve the original side.
# Two create counterfactual side conditions.
#
# 106 source scenarios × 10 variants = 1,060 candidate states.

STATE_VARIANT_PROFILES = [
    {
        "variant_name":
            "opening_zero_energy",

        "turn_number":
            1,

        "player_energy":
            0,

        "opponent_energy":
            0,

        "player_damage_fraction":
            0.00,

        "opponent_damage_fraction":
            0.00,

        "prize_cards_remaining":
            6,

        "hand_size":
            7,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "opening_one_energy",

        "turn_number":
            2,

        "player_energy":
            1,

        "opponent_energy":
            1,

        "player_damage_fraction":
            0.00,

        "opponent_damage_fraction":
            0.00,

        "prize_cards_remaining":
            6,

        "hand_size":
            6,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "early_pressure",

        "turn_number":
            3,

        "player_energy":
            1,

        "opponent_energy":
            2,

        "player_damage_fraction":
            0.15,

        "opponent_damage_fraction":
            0.05,

        "prize_cards_remaining":
            5,

        "hand_size":
            6,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "balanced_midgame",

        "turn_number":
            5,

        "player_energy":
            2,

        "opponent_energy":
            2,

        "player_damage_fraction":
            0.25,

        "opponent_damage_fraction":
            0.25,

        "prize_cards_remaining":
            4,

        "hand_size":
            5,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "player_under_pressure",

        "turn_number":
            6,

        "player_energy":
            2,

        "opponent_energy":
            3,

        "player_damage_fraction":
            0.50,

        "opponent_damage_fraction":
            0.15,

        "prize_cards_remaining":
            4,

        "hand_size":
            4,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "opponent_under_pressure",

        "turn_number":
            6,

        "player_energy":
            3,

        "opponent_energy":
            2,

        "player_damage_fraction":
            0.15,

        "opponent_damage_fraction":
            0.50,

        "prize_cards_remaining":
            4,

        "hand_size":
            5,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "late_game_prize_pressure",

        "turn_number":
            9,

        "player_energy":
            3,

        "opponent_energy":
            3,

        "player_damage_fraction":
            0.40,

        "opponent_damage_fraction":
            0.40,

        "prize_cards_remaining":
            2,

        "hand_size":
            3,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "critical_hp_decision",

        "turn_number":
            10,

        "player_energy":
            4,

        "opponent_energy":
            4,

        "player_damage_fraction":
            0.70,

        "opponent_damage_fraction":
            0.55,

        "prize_cards_remaining":
            2,

        "hand_size":
            3,

        "side_mode":
            "PRESERVE",
    },
    {
        "variant_name":
            "counterfactual_side_early",

        "turn_number":
            3,

        "player_energy":
            1,

        "opponent_energy":
            1,

        "player_damage_fraction":
            0.10,

        "opponent_damage_fraction":
            0.10,

        "prize_cards_remaining":
            5,

        "hand_size":
            6,

        "side_mode":
            "FLIP",
    },
    {
        "variant_name":
            "counterfactual_side_late",

        "turn_number":
            8,

        "player_energy":
            3,

        "opponent_energy":
            3,

        "player_damage_fraction":
            0.45,

        "opponent_damage_fraction":
            0.45,

        "prize_cards_remaining":
            3,

        "hand_size":
            4,

        "side_mode":
            "FLIP",
    },
]


VARIANTS_PER_SCENARIO = len(
    STATE_VARIANT_PROFILES
)

EXPECTED_VARIANT_COUNT = (
    len(scenario_queue_df)
    * VARIANTS_PER_SCENARIO
)


print()
print("GENERATION CONFIGURATION")
print("-" * 100)

print(
    "Source scenarios       :",
    len(scenario_queue_df),
)

print(
    "Variants per scenario  :",
    VARIANTS_PER_SCENARIO,
)

print(
    "Expected variants      :",
    EXPECTED_VARIANT_COUNT,
)

print(
    "Minimum legal moves    :",
    SECTION7C_MINIMUM_LEGAL_MOVES,
)

print(
    "Skip completed         :",
    SECTION7C_SKIP_COMPLETED_SCENARIOS,
)


# ------------------------------------------------------------
# 7C.3 — Serialization helpers
# ------------------------------------------------------------

def serialize_runtime_object(
    value: Any,
    maximum_depth: int = 4,
    current_depth: int = 0,
) -> Any:

    if current_depth >= maximum_depth:
        return repr(value)

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)

        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return serialize_runtime_object(
            asdict(value),
            maximum_depth=maximum_depth,
            current_depth=current_depth + 1,
        )

    if isinstance(value, Mapping):
        return {
            str(key): serialize_runtime_object(
                item,
                maximum_depth=maximum_depth,
                current_depth=current_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            serialize_runtime_object(
                item,
                maximum_depth=maximum_depth,
                current_depth=current_depth + 1,
            )
            for item in value
        ]

    if hasattr(value, "__dict__"):
        public_attributes = {
            key: item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

        return {
            "__type__":
                type(value).__name__,

            "__module__":
                type(value).__module__,

            "attributes":
                serialize_runtime_object(
                    public_attributes,
                    maximum_depth=maximum_depth,
                    current_depth=current_depth + 1,
                ),
        }

    return repr(value)


def move_to_serializable_record(
    move: Any,
    move_index: int,
) -> dict[str, Any]:

    serialized_move = serialize_runtime_object(
        move,
        maximum_depth=4,
    )

    return {
        "move_index":
            int(move_index),

        "move_type":
            type(move).__name__,

        "move_module":
            type(move).__module__,

        "move_repr":
            repr(move),

        "move_payload":
            serialized_move,
    }


# ------------------------------------------------------------
# 7C.4 — Deterministic ID and seed helpers
# ------------------------------------------------------------

def deterministic_variant_seed(
    source_scenario_id: Any,
    variant_name: str,
) -> int:

    seed_text = (
        f"{SECTION7C_BASE_SEED}|"
        f"{source_scenario_id}|"
        f"{variant_name}"
    )

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def deterministic_variant_id(
    source_scenario_id: Any,
    variant_name: str,
) -> str:

    raw_value = (
        f"{source_scenario_id}|"
        f"{variant_name}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{source_scenario_id}"
        f"__{variant_name}"
        f"__{digest}"
    )


def scenario_trace_file(
    source_scenario_id: Any,
) -> Path:

    safe_scenario_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(source_scenario_id),
    )

    return (
        SECTION7C_VARIANT_DIR
        / f"{safe_scenario_id}.jsonl"
    )


# ------------------------------------------------------------
# 7C.5 — Side-variant helpers
# ------------------------------------------------------------

def flip_manifest_side(
    side_value: Any,
) -> str:

    normalized_side = normalize_lookup_text(
        side_value
    )

    if (
        normalized_side == "opponent"
        or "opponent" in normalized_side
    ):
        return "Player"

    return "Opponent"


def scenario_for_variant(
    source_scenario: Mapping[str, Any],
    variant_profile: Mapping[str, Any],
) -> dict[str, Any]:

    variant_scenario = dict(
        source_scenario
    )

    if variant_profile["side_mode"] == "FLIP":

        variant_scenario[
            "normalized_side"
        ] = flip_manifest_side(
            source_scenario.get(
                "normalized_side"
            )
        )

    return variant_scenario


# ------------------------------------------------------------
# 7C.6 — Damage calculation
# ------------------------------------------------------------

def damage_from_fraction(
    card_reference: Any,
    damage_fraction: float,
) -> float:

    card_record, _ = resolve_card_record(
        card_reference
    )

    if card_record is None:
        raise LookupError(
            f"Could not resolve card for damage calculation: "
            f"{card_reference}"
        )

    maximum_hp = card_starting_hp(
        card_record
    )

    damage_value = round(
        maximum_hp
        * float(damage_fraction),
        2,
    )

    # Prevent construction of a knocked-out active Pokémon.
    maximum_allowed_damage = max(
        0.0,
        maximum_hp - 1.0,
    )

    return float(
        min(
            damage_value,
            maximum_allowed_damage,
        )
    )


# ------------------------------------------------------------
# 7C.7 — Construct one deterministic variant
# ------------------------------------------------------------

def generate_state_variant(
    source_scenario: Mapping[str, Any],
    variant_profile: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:

    source_scenario_id = source_scenario.get(
        "source_scenario_id"
    )

    variant_name = variant_profile[
        "variant_name"
    ]

    variant_id = deterministic_variant_id(
        source_scenario_id,
        variant_name,
    )

    variant_seed = deterministic_variant_seed(
        source_scenario_id,
        variant_name,
    )

    variant_scenario = scenario_for_variant(
        source_scenario,
        variant_profile,
    )

    player_damage = damage_from_fraction(
        variant_scenario.get(
            "player_card"
        ),
        variant_profile[
            "player_damage_fraction"
        ],
    )

    opponent_damage = damage_from_fraction(
        variant_scenario.get(
            "opponent_card"
        ),
        variant_profile[
            "opponent_damage_fraction"
        ],
    )

    try:

        battle_state, state_metadata = build_battle_state(
            variant_scenario,
            turn_number=variant_profile[
                "turn_number"
            ],
            player_energy=variant_profile[
                "player_energy"
            ],
            opponent_energy=variant_profile[
                "opponent_energy"
            ],
            player_damage=player_damage,
            opponent_damage=opponent_damage,
            prize_cards_remaining=variant_profile[
                "prize_cards_remaining"
            ],
            hand_size=variant_profile[
                "hand_size"
            ],
        )

        legal_moves = list(
            get_current_legal_moves(
                battle_state
            )
        )

        terminal = bool(
            terminal_state_adapter(
                battle_state
            )
        )

        accepted = (
            not terminal
            and len(legal_moves)
            >= SECTION7C_MINIMUM_LEGAL_MOVES
        )

        record = {
            "variant_id":
                variant_id,

            "source_scenario_id":
                source_scenario_id,

            "source_match_id":
                source_scenario.get(
                    "match_id"
                ),

            "queue_position":
                source_scenario.get(
                    "queue_position"
                ),

            "variant_name":
                variant_name,

            "variant_seed":
                variant_seed,

            "side_mode":
                variant_profile[
                    "side_mode"
                ],

            "source_normalized_side":
                source_scenario.get(
                    "normalized_side"
                ),

            "variant_normalized_side":
                variant_scenario.get(
                    "normalized_side"
                ),

            "current_player":
                battle_state.current_player,

            "baseline_name":
                source_scenario.get(
                    "baseline_name"
                ),

            "player_card":
                source_scenario.get(
                    "player_card"
                ),

            "opponent_card":
                source_scenario.get(
                    "opponent_card"
                ),

            "turn_number":
                int(
                    variant_profile[
                        "turn_number"
                    ]
                ),

            "player_energy":
                int(
                    variant_profile[
                        "player_energy"
                    ]
                ),

            "opponent_energy":
                int(
                    variant_profile[
                        "opponent_energy"
                    ]
                ),

            "player_damage":
                float(
                    player_damage
                ),

            "opponent_damage":
                float(
                    opponent_damage
                ),

            "player_damage_fraction":
                float(
                    variant_profile[
                        "player_damage_fraction"
                    ]
                ),

            "opponent_damage_fraction":
                float(
                    variant_profile[
                        "opponent_damage_fraction"
                    ]
                ),

            "prize_cards_remaining":
                int(
                    variant_profile[
                        "prize_cards_remaining"
                    ]
                ),

            "hand_size":
                int(
                    variant_profile[
                        "hand_size"
                    ]
                ),

            "terminal":
                terminal,

            "legal_move_count":
                int(
                    len(legal_moves)
                ),

            "accepted":
                bool(
                    accepted
                ),

            "legal_moves":
                [
                    move_to_serializable_record(
                        move,
                        move_index,
                    )
                    for move_index, move in enumerate(
                        legal_moves
                    )
                ],

            "state_metadata":
                serialize_runtime_object(
                    state_metadata,
                    maximum_depth=5,
                ),

            "battle_state":
                serialize_runtime_object(
                    battle_state,
                    maximum_depth=6,
                ),

            "curriculum_priority":
                source_scenario.get(
                    "hard_example_priority"
                ),

            "hard_example_rank":
                source_scenario.get(
                    "hard_example_rank"
                ),
        }

        return record, None

    except Exception as exc:

        failure = {
            "variant_id":
                variant_id,

            "source_scenario_id":
                source_scenario_id,

            "variant_name":
                variant_name,

            "variant_seed":
                variant_seed,

            "player_card":
                source_scenario.get(
                    "player_card"
                ),

            "opponent_card":
                source_scenario.get(
                    "opponent_card"
                ),

            "normalized_side":
                source_scenario.get(
                    "normalized_side"
                ),

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),
        }

        return None, failure


# ------------------------------------------------------------
# 7C.8 — Existing trace validation
# ------------------------------------------------------------

def read_jsonl_file(
    file_path: Path,
) -> list[dict[str, Any]]:

    records = []

    if not file_path.exists():
        return records

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            stripped_line = line.strip()

            if not stripped_line:
                continue

            records.append(
                json.loads(
                    stripped_line
                )
            )

    return records


def scenario_trace_is_complete(
    file_path: Path,
) -> bool:

    try:

        records = read_jsonl_file(
            file_path
        )

    except Exception:

        return False

    expected_variant_names = {
        profile["variant_name"]
        for profile in STATE_VARIANT_PROFILES
    }

    observed_variant_names = {
        record.get(
            "variant_name"
        )
        for record in records
    }

    return (
        len(records)
        == VARIANTS_PER_SCENARIO
        and observed_variant_names
        == expected_variant_names
    )


# ------------------------------------------------------------
# 7C.9 — Generate or resume all scenario variants
# ------------------------------------------------------------

variant_manifest_rows = []

variant_failure_rows = []

scenario_generation_rows = []


for scenario in scenario_queue_df.to_dict(
    orient="records"
):

    source_scenario_id = scenario[
        "source_scenario_id"
    ]

    trace_file = scenario_trace_file(
        source_scenario_id
    )

    skipped_existing = False


    if (
        SECTION7C_SKIP_COMPLETED_SCENARIOS
        and not SECTION7C_OVERWRITE_EXISTING
        and scenario_trace_is_complete(
            trace_file
        )
    ):

        scenario_records = read_jsonl_file(
            trace_file
        )

        skipped_existing = True

    else:

        scenario_records = []

        scenario_failures = []


        for variant_profile in STATE_VARIANT_PROFILES:

            record, failure = generate_state_variant(
                source_scenario=scenario,
                variant_profile=variant_profile,
            )

            if record is not None:
                scenario_records.append(
                    record
                )

            if failure is not None:
                scenario_failures.append(
                    failure
                )
                variant_failure_rows.append(
                    failure
                )


        if (
            scenario_records
            or SECTION7C_OVERWRITE_EXISTING
        ):

            with open(
                trace_file,
                "w",
                encoding="utf-8",
            ) as file:

                for record in scenario_records:

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )


    accepted_scenario_records = [
        record
        for record in scenario_records
        if record.get(
            "accepted"
        )
    ]


    for record in scenario_records:

        manifest_row = {
            key: value
            for key, value in record.items()
            if key not in {
                "legal_moves",
                "state_metadata",
                "battle_state",
            }
        }

        manifest_row[
            "trace_file"
        ] = str(
            trace_file
        )

        variant_manifest_rows.append(
            manifest_row
        )


    scenario_generation_rows.append(
        {
            "source_scenario_id":
                source_scenario_id,

            "queue_position":
                scenario.get(
                    "queue_position"
                ),

            "normalized_side":
                scenario.get(
                    "normalized_side"
                ),

            "baseline_name":
                scenario.get(
                    "baseline_name"
                ),

            "generated_variants":
                int(
                    len(
                        scenario_records
                    )
                ),

            "accepted_variants":
                int(
                    len(
                        accepted_scenario_records
                    )
                ),

            "rejected_variants":
                int(
                    len(
                        scenario_records
                    )
                    - len(
                        accepted_scenario_records
                    )
                ),

            "skipped_existing":
                skipped_existing,

            "trace_file":
                str(
                    trace_file
                ),
        }
    )


variant_manifest_df = pd.DataFrame(
    variant_manifest_rows
)

scenario_generation_df = pd.DataFrame(
    scenario_generation_rows
)

variant_failure_df = pd.DataFrame(
    variant_failure_rows
)


# ------------------------------------------------------------
# 7C.10 — Save reports
# ------------------------------------------------------------

variant_manifest_df.to_csv(
    SECTION7C_VARIANT_MANIFEST_FILE,
    index=False,
)

scenario_generation_df.to_csv(
    SECTION7C_SCENARIO_SUMMARY_FILE,
    index=False,
)

variant_failure_df.to_csv(
    SECTION7C_FAILURE_FILE,
    index=False,
)


# ------------------------------------------------------------
# 7C.11 — Update progress ledger
# ------------------------------------------------------------

progress_update_df = (
    scenario_generation_df[
        [
            "source_scenario_id",
            "generated_variants",
            "accepted_variants",
            "trace_file",
        ]
    ]
    .copy()
)


scenario_progress_df = (
    scenario_progress_df
    .drop(
        columns=[
            column
            for column in [
                "generated_states",
                "accepted_states",
                "trace_file",
            ]
            if column in scenario_progress_df.columns
        ]
    )
    .merge(
        progress_update_df,
        on="source_scenario_id",
        how="left",
    )
)


scenario_progress_df[
    "generated_states"
] = (
    scenario_progress_df[
        "generated_variants"
    ]
    .fillna(0)
    .astype(int)
)

scenario_progress_df[
    "accepted_states"
] = (
    scenario_progress_df[
        "accepted_variants"
    ]
    .fillna(0)
    .astype(int)
)


scenario_progress_df[
    "status"
] = np.where(
    scenario_progress_df[
        "accepted_states"
    ].ge(
        VARIANTS_PER_SCENARIO
    ),
    "COMPLETED",
    np.where(
        scenario_progress_df[
            "generated_states"
        ].gt(0),
        "PARTIAL",
        "FAILED",
    ),
)


scenario_progress_df = scenario_progress_df.drop(
    columns=[
        "generated_variants",
        "accepted_variants",
    ],
)

scenario_progress_df.to_csv(
    SECTION7_PROGRESS_FILE,
    index=False,
)


# ------------------------------------------------------------
# 7C.12 — Summary metrics
# ------------------------------------------------------------

generated_variant_count = int(
    len(
        variant_manifest_df
    )
)

accepted_variant_count = int(
    variant_manifest_df[
        "accepted"
    ].sum()
)

rejected_variant_count = int(
    generated_variant_count
    - accepted_variant_count
)

failed_variant_count = int(
    len(
        variant_failure_df
    )
)

completed_scenario_count = int(
    scenario_generation_df[
        "accepted_variants"
    ]
    .ge(
        VARIANTS_PER_SCENARIO
    )
    .sum()
)

preserved_side_count = int(
    variant_manifest_df[
        "side_mode"
    ]
    .eq(
        "PRESERVE"
    )
    .sum()
)

counterfactual_side_count = int(
    variant_manifest_df[
        "side_mode"
    ]
    .eq(
        "FLIP"
    )
    .sum()
)


variant_summary = {
    "status":
        (
            "READY"
            if accepted_variant_count > 0
            else "FAILED"
        ),

    "source_scenarios":
        int(
            len(
                scenario_queue_df
            )
        ),

    "variants_per_scenario":
        int(
            VARIANTS_PER_SCENARIO
        ),

    "expected_variants":
        int(
            EXPECTED_VARIANT_COUNT
        ),

    "generated_variants":
        generated_variant_count,

    "accepted_variants":
        accepted_variant_count,

    "rejected_variants":
        rejected_variant_count,

    "failed_variants":
        failed_variant_count,

    "completed_scenarios":
        completed_scenario_count,

    "preserved_side_variants":
        preserved_side_count,

    "counterfactual_side_variants":
        counterfactual_side_count,

    "minimum_legal_moves":
        SECTION7C_MINIMUM_LEGAL_MOVES,

    "next_stage":
        (
            "SEARCH_GUIDED_STATE_MINING"
            if accepted_variant_count > 0
            else "REPAIR_VARIANT_GENERATION"
        ),
}


with open(
    SECTION7C_VARIANT_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        variant_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 7C.13 — Display results
# ------------------------------------------------------------

print()
print("STATE VARIANT GENERATION SUMMARY")
print("-" * 100)

print(
    "Source scenarios          :",
    variant_summary[
        "source_scenarios"
    ],
)

print(
    "Expected variants         :",
    variant_summary[
        "expected_variants"
    ],
)

print(
    "Generated variants        :",
    variant_summary[
        "generated_variants"
    ],
)

print(
    "Accepted variants         :",
    variant_summary[
        "accepted_variants"
    ],
)

print(
    "Rejected variants         :",
    variant_summary[
        "rejected_variants"
    ],
)

print(
    "Generation failures       :",
    variant_summary[
        "failed_variants"
    ],
)

print(
    "Completed scenarios       :",
    (
        f"{variant_summary['completed_scenarios']}/"
        f"{variant_summary['source_scenarios']}"
    ),
)

print(
    "Preserved-side variants   :",
    variant_summary[
        "preserved_side_variants"
    ],
)

print(
    "Counterfactual variants   :",
    variant_summary[
        "counterfactual_side_variants"
    ],
)

print(
    "Next stage                :",
    variant_summary[
        "next_stage"
    ],
)


print()
print("VARIANT PROFILE PERFORMANCE")
print("-" * 100)

variant_profile_summary_df = (
    variant_manifest_df
    .groupby(
        [
            "variant_name",
            "side_mode",
        ],
        dropna=False,
    )
    .agg(
        generated_variants=(
            "variant_id",
            "count",
        ),
        accepted_variants=(
            "accepted",
            "sum",
        ),
        average_legal_moves=(
            "legal_move_count",
            "mean",
        ),
    )
    .reset_index()
)

display(
    variant_profile_summary_df
)


print()
print("SCENARIO GENERATION SAMPLE")
print("-" * 100)

display(
    scenario_generation_df.head(
        20
    )
)


if not variant_failure_df.empty:

    print()
    print("VARIANT FAILURES")
    print("-" * 100)

    display(
        variant_failure_df.head(
            30
        )
    )


# ------------------------------------------------------------
# 7C.14 — Final validation
# ------------------------------------------------------------

assert SECTION7C_VARIANT_DIR.exists()

assert SECTION7C_VARIANT_MANIFEST_FILE.exists()

assert SECTION7C_VARIANT_SUMMARY_FILE.exists()

assert SECTION7C_SCENARIO_SUMMARY_FILE.exists()

assert SECTION7C_FAILURE_FILE.exists()

assert generated_variant_count > 0, (
    "Section 7C did not generate any state variants."
)

assert accepted_variant_count > 0, (
    "Section 7C did not accept any valid state variants."
)

assert variant_manifest_df[
    "variant_id"
].is_unique, (
    "Generated variant IDs must be unique."
)

assert variant_manifest_df[
    "source_scenario_id"
].nunique() > 0

assert variant_manifest_df[
    "legal_move_count"
].ge(
    SECTION7C_MINIMUM_LEGAL_MOVES
).any()

assert variant_summary[
    "next_stage"
] == "SEARCH_GUIDED_STATE_MINING"


print()
print("SAVED SECTION 7C REPORTS")
print("-" * 100)

print(
    SECTION7C_VARIANT_MANIFEST_FILE
)

print(
    SECTION7C_SCENARIO_SUMMARY_FILE
)

print(
    SECTION7C_FAILURE_FILE
)

print(
    SECTION7C_VARIANT_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 7C DETERMINISTIC STATE VARIANT GENERATION PASSED"
)


# In[14]:


# ============================================================
# SECTION 8A REPAIR
# LOAD ONLY THE REQUIRED PRODUCTION APPLY_MOVE FUNCTION
# ============================================================

import importlib.util
import inspect


print("=" * 100)
print("SECTION 8A REPAIR — LOAD REQUIRED APPLY_MOVE")
print("=" * 100)


apply_move_script_path = (
    PROJECT_ROOT
    / "scripts"
    / "10_minimax_opponent_ai.py"
)


assert apply_move_script_path.exists(), (
    f"Could not find apply_move source: "
    f"{apply_move_script_path}"
)


module_spec = importlib.util.spec_from_file_location(
    "nb49_production_transition_module",
    apply_move_script_path,
)

assert module_spec is not None
assert module_spec.loader is not None


production_transition_module = (
    importlib.util.module_from_spec(
        module_spec
    )
)

module_spec.loader.exec_module(
    production_transition_module
)


apply_move_function = getattr(
    production_transition_module,
    "apply_move",
    None,
)


assert callable(
    apply_move_function
), (
    "The production script does not expose a callable apply_move."
)


print()
print("PRODUCTION TRANSITION BINDING")
print("-" * 100)

print(
    "Source    :",
    apply_move_script_path,
)

print(
    "Function  :",
    getattr(
        apply_move_function,
        "__qualname__",
        repr(apply_move_function),
    ),
)

print(
    "Signature :",
    inspect.signature(
        apply_move_function
    ),
)

print()
print(
    "✅ SECTION 8A REQUIRED APPLY_MOVE LOADED"
)


# In[15]:


# ============================================================
# SECTION 8A REPAIR
# HASHABLE STATE AND MOVE KEYS
# ============================================================

from typing import Any, Mapping


print("=" * 100)
print("SECTION 8A REPAIR — HASHABLE STATE AND MOVE KEYS")
print("=" * 100)


def stable_card_identifier(
    card: Any,
) -> tuple[str, str]:

    if not isinstance(card, Mapping):
        return (
            type(card).__name__,
            repr(card),
        )

    card_id = (
        card.get("Card ID")
        or card.get("card_id")
        or card.get("id")
        or ""
    )

    card_name = (
        card.get("Card Name")
        or card.get("card_name")
        or card.get("name")
        or ""
    )

    return (
        str(card_id),
        str(card_name),
    )


def pokemon_state_key(
    pokemon_state: Any,
) -> tuple[Any, ...]:

    if pokemon_state is None:
        return ("NONE",)

    return (
        stable_card_identifier(
            getattr(
                pokemon_state,
                "card",
                None,
            )
        ),
        float(
            getattr(
                pokemon_state,
                "current_hp",
                0.0,
            )
        ),
        int(
            getattr(
                pokemon_state,
                "attached_energy",
                0,
            )
        ),
        str(
            getattr(
                pokemon_state,
                "status",
                None,
            )
        ),
        float(
            getattr(
                pokemon_state,
                "damage",
                0.0,
            )
        ),
        bool(
            getattr(
                pokemon_state,
                "is_active",
                False,
            )
        ),
    )


def player_state_key(
    player_state: Any,
) -> tuple[Any, ...]:

    if player_state is None:
        return ("NONE",)

    bench = getattr(
        player_state,
        "bench",
        [],
    )

    return (
        pokemon_state_key(
            getattr(
                player_state,
                "active",
                None,
            )
        ),
        tuple(
            pokemon_state_key(
                pokemon
            )
            for pokemon in bench
        ),
        int(
            getattr(
                player_state,
                "prize_cards_remaining",
                0,
            )
        ),
        int(
            getattr(
                player_state,
                "hand_size",
                0,
            )
        ),
    )


def notebook49_state_key(
    battle_state: Any,
) -> tuple[Any, ...]:

    return (
        player_state_key(
            getattr(
                battle_state,
                "player",
                None,
            )
        ),
        player_state_key(
            getattr(
                battle_state,
                "opponent",
                None,
            )
        ),
        int(
            getattr(
                battle_state,
                "turn_number",
                0,
            )
        ),
        str(
            getattr(
                battle_state,
                "current_player",
                "",
            )
        ),
    )


def notebook49_move_key(
    move: Any,
) -> tuple[Any, ...]:

    if isinstance(move, Mapping):

        return (
            str(
                move.get("name")
                or move.get("Move Name")
                or ""
            ),
            float(
                move.get("damage")
                or move.get("damage_numeric")
                or 0.0
            ),
            int(
                move.get("energy_cost")
                or move.get("cost")
                or 0
            ),
            str(
                move.get("effect")
                or move.get("Effect Explanation")
                or ""
            ),
        )

    return (
        type(move).__name__,
        repr(move),
    )


print()
print("KEY FUNCTION VALIDATION")
print("-" * 100)

print(
    "State key callable :",
    callable(
        notebook49_state_key
    ),
)

print(
    "Move key callable  :",
    callable(
        notebook49_move_key
    ),
)

assert callable(
    notebook49_state_key
)

assert callable(
    notebook49_move_key
)


print()
print(
    "✅ SECTION 8A HASHABLE STATE AND MOVE KEYS PASSED"
)


# In[16]:


# ------------------------------------------------------------
# 8A.4 — Build constructor argument pool
# ------------------------------------------------------------

search_constructor_pool = {
    "generate_moves":
        generate_moves_adapter,

    "apply_move":
        apply_move_function,

    "evaluate_state":
        evaluate_state_adapter,

    "is_terminal":
        terminal_state_adapter,

    "current_player":
        current_player_adapter,

    "state_key":
        notebook49_state_key,

    "move_key":
        notebook49_move_key,
}


# In[17]:


def resolve_signature_arguments(
    callable_object: Any,
    argument_pool: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:

    signature = inspect.signature(
        callable_object
    )

    resolved = {}
    unresolved = []

    for parameter_name, parameter in signature.parameters.items():

        if parameter_name in {
            "self",
            "cls",
        }:
            continue

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if parameter_name in argument_pool:

            resolved[
                parameter_name
            ] = argument_pool[
                parameter_name
            ]

        elif parameter.default is inspect.Parameter.empty:

            unresolved.append(
                parameter_name
            )

    return resolved, unresolved


# In[18]:


constructor_arguments, unresolved_constructor_arguments = (
    resolve_signature_arguments(
        AdvancedSearchEngine,
        search_constructor_pool,
    )
)


print()
print("SEARCH CONSTRUCTOR RESOLUTION")
print("-" * 100)

print(
    "Resolved arguments   :",
    sorted(
        constructor_arguments
    ),
)

print(
    "Unresolved arguments :",
    unresolved_constructor_arguments,
)


assert not unresolved_constructor_arguments, (
    "AdvancedSearchEngine has unresolved constructor arguments: "
    f"{unresolved_constructor_arguments}"
)


search_engine = AdvancedSearchEngine(
    **constructor_arguments
)


# In[19]:


# ============================================================
# SECTION 8A SETUP REPAIR
# RELOAD ACCEPTED SECTION 7C VARIANTS
# ============================================================

import pandas as pd


print("=" * 100)
print("SECTION 8A SETUP — RELOAD ACCEPTED SECTION 7C VARIANTS")
print("=" * 100)


section7c_variant_df = pd.read_csv(
    SECTION7C_VARIANT_MANIFEST_FILE
)


accepted_variant_df = (
    section7c_variant_df[
        section7c_variant_df[
            "accepted"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    ]
    .copy()
    .reset_index(drop=True)
)


assert not accepted_variant_df.empty, (
    "No accepted Section 7C variants were found."
)


print()
print(
    "Accepted variants :",
    len(accepted_variant_df),
)

print(
    "Source scenarios  :",
    accepted_variant_df[
        "source_scenario_id"
    ].nunique(),
)

print()
print(
    "✅ SECTION 8A ACCEPTED VARIANTS RELOADED"
)


# In[20]:


# ============================================================
# SECTION 8A SETUP REPAIR
# RESTORE SECTION 8 OUTPUT PATHS
# ============================================================

from pathlib import Path


print("=" * 100)
print("SECTION 8A SETUP — RESTORE OUTPUT PATHS")
print("=" * 100)


SECTION8_REPORT_DIR = (
    NOTEBOOK49_REPORT_DIR
    / "section8"
)

SECTION8_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION8A_BINDING_FILE = (
    SECTION8_REPORT_DIR
    / "section8a_search_binding.json"
)

SECTION8A_DRY_RUN_FILE = (
    SECTION8_REPORT_DIR
    / "section8a_search_dry_run.json"
)


print()
print("SECTION 8A OUTPUT PATHS")
print("-" * 100)

print(
    "Report directory :",
    SECTION8_REPORT_DIR,
)

print(
    "Binding report   :",
    SECTION8A_BINDING_FILE,
)

print(
    "Dry-run report   :",
    SECTION8A_DRY_RUN_FILE,
)


assert SECTION8_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 8A OUTPUT PATHS RESTORED"
)


# In[21]:


# ============================================================
# SECTION 8A CONTINUATION
# ONE-STATE SEARCH DRY RUN AND FINAL VALIDATION
# ============================================================

print("=" * 100)
print("SECTION 8A CONTINUATION — ONE-STATE SEARCH DRY RUN")
print("=" * 100)


# ------------------------------------------------------------
# 8A.5 — Reconstruct one accepted state variant
# ------------------------------------------------------------

assert "search_engine" in globals(), (
    "search_engine is not available. "
    "Run the constructor-resolution cell first."
)

assert not accepted_variant_df.empty, (
    "No accepted Section 7C variants are available."
)


dry_run_variant = (
    accepted_variant_df
    .iloc[0]
    .to_dict()
)


source_scenario_matches = (
    scenario_queue_df[
        scenario_queue_df[
            "source_scenario_id"
        ]
        .astype(str)
        .eq(
            str(
                dry_run_variant[
                    "source_scenario_id"
                ]
            )
        )
    ]
)


assert not source_scenario_matches.empty, (
    "Could not recover the source scenario for "
    f"{dry_run_variant['source_scenario_id']}."
)


dry_run_scenario = (
    source_scenario_matches
    .iloc[0]
    .to_dict()
)


# Use the side stored in the generated variant.
dry_run_scenario[
    "normalized_side"
] = dry_run_variant[
    "variant_normalized_side"
]


dry_run_state, dry_run_state_metadata = (
    build_battle_state(
        dry_run_scenario,

        turn_number=int(
            dry_run_variant[
                "turn_number"
            ]
        ),

        player_energy=int(
            dry_run_variant[
                "player_energy"
            ]
        ),

        opponent_energy=int(
            dry_run_variant[
                "opponent_energy"
            ]
        ),

        player_damage=float(
            dry_run_variant[
                "player_damage"
            ]
        ),

        opponent_damage=float(
            dry_run_variant[
                "opponent_damage"
            ]
        ),

        prize_cards_remaining=int(
            dry_run_variant[
                "prize_cards_remaining"
            ]
        ),

        hand_size=int(
            dry_run_variant[
                "hand_size"
            ]
        ),
    )
)


dry_run_legal_moves = list(
    get_current_legal_moves(
        dry_run_state
    )
)


assert dry_run_legal_moves, (
    "The reconstructed dry-run state has no legal moves."
)


print()
print("DRY-RUN STATE")
print("-" * 100)

print(
    "Variant ID       :",
    dry_run_variant[
        "variant_id"
    ],
)

print(
    "Scenario ID      :",
    dry_run_variant[
        "source_scenario_id"
    ],
)

print(
    "Variant name     :",
    dry_run_variant[
        "variant_name"
    ],
)

print(
    "Player card      :",
    dry_run_variant[
        "player_card"
    ],
)

print(
    "Opponent card    :",
    dry_run_variant[
        "opponent_card"
    ],
)

print(
    "Current player   :",
    dry_run_state.current_player,
)

print(
    "Turn number      :",
    dry_run_state.turn_number,
)

print(
    "Legal move count :",
    len(
        dry_run_legal_moves
    ),
)


print()
print("LEGAL MOVES")
print("-" * 100)

for move_index, move in enumerate(
    dry_run_legal_moves,
    start=1,
):

    print(
        f"{move_index:02d}: "
        f"{repr(move)}"
    )


# ------------------------------------------------------------
# 8A.6 — Resolve search_depth arguments
# ------------------------------------------------------------

SECTION8_DRY_RUN_DEPTH = 2


search_argument_pool = {
    "state":
        dry_run_state,

    "battle_state":
        dry_run_state,

    "depth":
        SECTION8_DRY_RUN_DEPTH,

    "max_depth":
        SECTION8_DRY_RUN_DEPTH,

    "root_player":
        dry_run_state.current_player,

    "player":
        dry_run_state.current_player,

    "verbose":
        False,

    "clear_table":
        True,

    "clear_killers":
        True,

    "clear_history":
        True,
}


search_depth_method = (
    search_engine.search_depth
)


search_arguments, unresolved_search_arguments = (
    resolve_signature_arguments(
        search_depth_method,
        search_argument_pool,
    )
)


print()
print("SEARCH DEPTH ARGUMENT RESOLUTION")
print("-" * 100)

print(
    "Resolved arguments   :",
    sorted(
        search_arguments
    ),
)

print(
    "Unresolved arguments :",
    unresolved_search_arguments,
)


assert not unresolved_search_arguments, (
    "search_depth has unresolved required arguments: "
    f"{unresolved_search_arguments}"
)


# ------------------------------------------------------------
# 8A.7 — Run one controlled production search
# ------------------------------------------------------------

dry_run_search_passed = False
dry_run_search_result = None
dry_run_error = None


try:

    dry_run_search_result = (
        search_depth_method(
            **search_arguments
        )
    )

    dry_run_search_passed = True

except Exception as exc:

    dry_run_error = (
        f"{type(exc).__name__}: "
        f"{exc}"
    )


print()
print("SEARCH DRY RUN")
print("-" * 100)

print(
    "Passed      :",
    dry_run_search_passed,
)

print(
    "Result type :",
    (
        type(
            dry_run_search_result
        ).__name__
        if dry_run_search_result is not None
        else None
    ),
)

print(
    "Result      :",
    repr(
        dry_run_search_result
    ),
)

print(
    "Error       :",
    dry_run_error,
)


# ------------------------------------------------------------
# 8A.7B — Inspect returned SearchResult fields
# ------------------------------------------------------------

search_result_attributes = {}


if dry_run_search_result is not None:

    if hasattr(
        dry_run_search_result,
        "__dict__",
    ):

        search_result_attributes = {
            key:
                value

            for key, value in vars(
                dry_run_search_result
            ).items()

            if not str(
                key
            ).startswith(
                "_"
            )
        }

    elif isinstance(
        dry_run_search_result,
        dict,
    ):

        search_result_attributes = dict(
            dry_run_search_result
        )


print()
print("SEARCH RESULT ATTRIBUTES")
print("-" * 100)

if search_result_attributes:

    for key, value in search_result_attributes.items():

        print(
            f"{key:25s}: "
            f"{repr(value)}"
        )

else:

    print(
        "No public result attributes were discovered."
    )


# ------------------------------------------------------------
# 8A.8 — Save search binding and dry-run reports
# ------------------------------------------------------------

search_binding_report = {
    "search_engine_class":
        type(
            search_engine
        ).__name__,

    "search_engine_module":
        type(
            search_engine
        ).__module__,

    "constructor_signature":
        str(
            inspect.signature(
                AdvancedSearchEngine
            )
        ),

    "constructor_arguments":
        sorted(
            constructor_arguments
        ),

    "search_depth_signature":
        str(
            inspect.signature(
                search_depth_method
            )
        ),

    "search_depth_arguments":
        sorted(
            search_arguments
        ),

    "apply_move_source":
        str(
            apply_move_script_path
        ),

    "apply_move_signature":
        str(
            inspect.signature(
                apply_move_function
            )
        ),

    "state_key_function":
        getattr(
            notebook49_state_key,
            "__name__",
            repr(
                notebook49_state_key
            ),
        ),

    "move_key_function":
        getattr(
            notebook49_move_key,
            "__name__",
            repr(
                notebook49_move_key
            ),
        ),
}


search_dry_run_report = {
    "variant_id":
        dry_run_variant[
            "variant_id"
        ],

    "source_scenario_id":
        dry_run_variant[
            "source_scenario_id"
        ],

    "variant_name":
        dry_run_variant[
            "variant_name"
        ],

    "player_card":
        dry_run_variant[
            "player_card"
        ],

    "opponent_card":
        dry_run_variant[
            "opponent_card"
        ],

    "depth_requested":
        SECTION8_DRY_RUN_DEPTH,

    "current_player":
        dry_run_state.current_player,

    "turn_number":
        int(
            dry_run_state.turn_number
        ),

    "legal_move_count":
        len(
            dry_run_legal_moves
        ),

    "legal_moves":
        [
            repr(
                move
            )
            for move in dry_run_legal_moves
        ],

    "state_key":
        notebook49_state_key(
            dry_run_state
        ),

    "passed":
        dry_run_search_passed,

    "result_type":
        (
            type(
                dry_run_search_result
            ).__name__
            if dry_run_search_result is not None
            else None
        ),

    "result_repr":
        repr(
            dry_run_search_result
        ),

    "result_attributes":
        search_result_attributes,

    "error":
        dry_run_error,
}


with open(
    SECTION8A_BINDING_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        search_binding_report,
        file,
        indent=4,
        default=str,
    )


with open(
    SECTION8A_DRY_RUN_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        search_dry_run_report,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8A.9 — Final validation
# ------------------------------------------------------------

assert SECTION8_REPORT_DIR.exists()

assert SECTION8A_BINDING_FILE.exists()

assert SECTION8A_DRY_RUN_FILE.exists()

assert callable(
    apply_move_function
)

assert callable(
    notebook49_state_key
)

assert callable(
    notebook49_move_key
)

assert len(
    dry_run_legal_moves
) > 0

assert dry_run_search_passed, (
    "The production search engine dry run failed: "
    f"{dry_run_error}"
)

assert dry_run_search_result is not None, (
    "The search completed without an exception but "
    "returned no SearchResult."
)


print()
print("SECTION 8A SUMMARY")
print("-" * 100)

print(
    "Search engine       :",
    type(
        search_engine
    ).__name__,
)

print(
    "Dry-run variant     :",
    dry_run_variant[
        "variant_id"
    ],
)

print(
    "Dry-run depth       :",
    SECTION8_DRY_RUN_DEPTH,
)

print(
    "Current player      :",
    dry_run_state.current_player,
)

print(
    "Legal move count    :",
    len(
        dry_run_legal_moves
    ),
)

print(
    "Search result type  :",
    type(
        dry_run_search_result
    ).__name__,
)

print(
    "Next stage          :",
    "BATCH_SEARCH_GUIDED_MINING",
)


print()
print("SAVED SECTION 8A REPORTS")
print("-" * 100)

print(
    SECTION8A_BINDING_FILE
)

print(
    SECTION8A_DRY_RUN_FILE
)


print()
print(
    "✅ SECTION 8A SEARCH ENGINE BINDING AND DRY RUN PASSED"
)


# In[22]:


# ============================================================
# SECTION 8B RESET
# DELETE STALE PRE-REPAIR SEARCH RESULTS
# ============================================================

from pathlib import Path


stale_section8b_files = [
    SECTION8B_RESULTS_FILE,
    SECTION8B_FAILURES_FILE,
    SECTION8B_PROGRESS_FILE,
    SECTION8B_SUMMARY_FILE,
    SECTION8B_JSONL_FILE,
]


print("=" * 100)
print("RESETTING STALE SECTION 8B OUTPUTS")
print("=" * 100)


for file_path in stale_section8b_files:

    file_path = Path(
        file_path
    )

    if file_path.exists():

        file_path.unlink()

        print(
            "Deleted:",
            file_path,
        )

    else:

        print(
            "Not found:",
            file_path,
        )


print()
print(
    "✅ OLD SECTION 8B SEARCH OUTPUTS CLEARED"
)


# ## Section 8B — Resumable Batch Search Labeling
# 
# Section 8B applies the validated production search engine to every accepted
# state variant created in Section 7C.
# 
# #### The process is resumable:
# 
# - previously completed variants are skipped,
# - each result is checkpointed,
# - failures are preserved for diagnosis,
# - and search labels are saved independently from the larger state traces.
# 
# #### The search output records the recommended move, evaluation score, principal variation, node statistics, and search metadata whenever those fields are available from the production `SearchResult`.
# 
# #### These search labels will be compared against the competitive policy in the next section.

# In[ ]:


# ============================================================
# SECTION 8B
# RESUMABLE BATCH SEARCH LABELING
# ============================================================

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 8B — RESUMABLE BATCH SEARCH LABELING")
print("=" * 100)


# ------------------------------------------------------------
# 8B.1 — Preconditions
# ------------------------------------------------------------

assert "search_engine" in globals(), (
    "Run Section 8A successfully before Section 8B."
)

assert "accepted_variant_df" in globals(), (
    "Accepted Section 7C variants are not loaded."
)

assert callable(
    globals().get(
        "apply_move_function"
    )
)

assert callable(
    globals().get(
        "notebook49_state_key"
    )
)

assert callable(
    globals().get(
        "notebook49_move_key"
    )
)

assert not accepted_variant_df.empty


# ------------------------------------------------------------
# 8B.2 — Output paths
# ------------------------------------------------------------

SECTION8B_REPORT_DIR = (
    SECTION8_REPORT_DIR
    / "batch_search"
)

SECTION8B_CHECKPOINT_DIR = (
    SECTION8B_REPORT_DIR
    / "checkpoints"
)

SECTION8B_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION8B_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION8B_RESULTS_FILE = (
    SECTION8B_REPORT_DIR
    / "section8b_search_labels.csv"
)

SECTION8B_FAILURES_FILE = (
    SECTION8B_REPORT_DIR
    / "section8b_search_failures.csv"
)

SECTION8B_PROGRESS_FILE = (
    SECTION8B_CHECKPOINT_DIR
    / "section8b_search_progress.csv"
)

SECTION8B_SUMMARY_FILE = (
    SECTION8B_REPORT_DIR
    / "section8b_search_summary.json"
)

SECTION8B_JSONL_FILE = (
    SECTION8B_REPORT_DIR
    / "section8b_search_labels.jsonl"
)


# ------------------------------------------------------------
# 8B.3 — Search configuration
# ------------------------------------------------------------

SECTION8B_SEARCH_DEPTH = 4

SECTION8B_BATCH_LIMIT = None

SECTION8B_CHECKPOINT_INTERVAL = 25

## SECTION8B_SKIP_COMPLETED = True

SECTION8B_SKIP_COMPLETED = False

SECTION8B_RETRY_FAILED = True

SECTION8B_CLEAR_TABLE_EACH_STATE = True

SECTION8B_CLEAR_KILLERS_EACH_STATE = True

SECTION8B_CLEAR_HISTORY_EACH_STATE = True


print()
print("SEARCH CONFIGURATION")
print("-" * 100)

print(
    "Search depth             :",
    SECTION8B_SEARCH_DEPTH,
)

print(
    "Accepted input variants  :",
    len(
        accepted_variant_df
    ),
)

print(
    "Batch limit              :",
    SECTION8B_BATCH_LIMIT,
)

print(
    "Checkpoint interval      :",
    SECTION8B_CHECKPOINT_INTERVAL,
)

print(
    "Skip completed           :",
    SECTION8B_SKIP_COMPLETED,
)

print(
    "Retry failed             :",
    SECTION8B_RETRY_FAILED,
)


# ------------------------------------------------------------
# 8B.4 — Generic serialization helpers
# ------------------------------------------------------------

def json_safe_value(
    value: Any,
    maximum_depth: int = 5,
    current_depth: int = 0,
) -> Any:

    if current_depth >= maximum_depth:
        return repr(
            value
        )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        numeric_value = float(
            value
        )

        if math.isfinite(
            numeric_value
        ):
            return numeric_value

        return str(
            numeric_value
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key):
                json_safe_value(
                    item,
                    maximum_depth=maximum_depth,
                    current_depth=current_depth + 1,
                )

            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            json_safe_value(
                item,
                maximum_depth=maximum_depth,
                current_depth=current_depth + 1,
            )

            for item in value
        ]

    if hasattr(
        value,
        "__dict__",
    ):

        public_attributes = {
            key:
                item

            for key, item in vars(
                value
            ).items()

            if not str(
                key
            ).startswith(
                "_"
            )
        }

        return {
            "__type__":
                type(
                    value
                ).__name__,

            "__module__":
                type(
                    value
                ).__module__,

            "attributes":
                json_safe_value(
                    public_attributes,
                    maximum_depth=maximum_depth,
                    current_depth=current_depth + 1,
                ),
        }

    return repr(
        value
    )


def first_available_attribute(
    obj: Any,
    names: list[str],
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(
        obj,
        Mapping,
    ):

        for name in names:

            if name in obj:
                return obj[
                    name
                ]

        return default

    for name in names:

        if hasattr(
            obj,
            name,
        ):

            try:
                return getattr(
                    obj,
                    name,
                )

            except Exception:
                continue

    return default


def move_display_string(
    move: Any,
) -> str:

    if move is None:
        return ""

    if isinstance(
        move,
        Mapping,
    ):

        for key in [
            "name",
            "Move Name",
            "move_name",
            "action",
            "type",
        ]:

            if key in move:
                return str(
                    move[
                        key
                    ]
                )

    return repr(
        move
    )


# ------------------------------------------------------------
# 8B.5 — SearchResult parser
# ------------------------------------------------------------

def parse_search_result(
    search_result: Any,
) -> dict[str, Any]:

    best_move = first_available_attribute(
        search_result,
        [
            "best_move",
            "move",
            "selected_move",
            "recommended_move",
            "action",
        ],
    )

    score = first_available_attribute(
        search_result,
        [
            "score",
            "value",
            "evaluation",
            "best_score",
            "utility",
        ],
    )

    depth_reached = first_available_attribute(
        search_result,
        [
            "depth",
            "depth_reached",
            "completed_depth",
            "search_depth",
        ],
    )

    nodes = first_available_attribute(
        search_result,
        [
            "nodes",
            "nodes_searched",
            "node_count",
            "positions",
        ],
    )

    principal_variation = first_available_attribute(
        search_result,
        [
            "principal_variation",
            "pv",
            "line",
            "best_line",
        ],
        default=[],
    )

    elapsed_seconds = first_available_attribute(
        search_result,
        [
            "elapsed_seconds",
            "elapsed_time",
            "seconds",
            "duration",
        ],
    )

    timed_out = first_available_attribute(
        search_result,
        [
            "timed_out",
            "timeout",
            "was_interrupted",
        ],
        default=False,
    )

    return {
        "search_result_type":
            type(
                search_result
            ).__name__,

        "search_result_module":
            type(
                search_result
            ).__module__,

        "best_move":
            best_move,

        "best_move_repr":
            move_display_string(
                best_move
            ),

        "best_move_key":
            (
                notebook49_move_key(
                    best_move
                )
                if best_move is not None
                else None
            ),

        "score":
            score,

        "depth_reached":
            depth_reached,

        "nodes":
            nodes,

        "principal_variation":
            principal_variation,

        "elapsed_seconds_reported":
            elapsed_seconds,

        "timed_out":
            bool(
                timed_out
            ),

        "raw_result":
            json_safe_value(
                search_result,
                maximum_depth=6,
            ),
    }


# ------------------------------------------------------------
# 8B.6 — Reconstruct a Section 7C variant
# ------------------------------------------------------------

def reconstruct_variant_state(
    variant_row: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[Any]]:

    source_scenario_matches = scenario_queue_df[
        scenario_queue_df[
            "source_scenario_id"
        ].astype(str).eq(
            str(
                variant_row[
                    "source_scenario_id"
                ]
            )
        )
    ]

    if source_scenario_matches.empty:

        raise LookupError(
            "Could not locate source scenario: "
            f"{variant_row['source_scenario_id']}"
        )

    source_scenario = (
        source_scenario_matches
        .iloc[
            0
        ]
        .to_dict()
    )

    source_scenario[
        "normalized_side"
    ] = variant_row[
        "variant_normalized_side"
    ]

    battle_state, state_metadata = (
        build_battle_state(
            source_scenario,
            turn_number=int(
                variant_row[
                    "turn_number"
                ]
            ),
            player_energy=int(
                variant_row[
                    "player_energy"
                ]
            ),
            opponent_energy=int(
                variant_row[
                    "opponent_energy"
                ]
            ),
            player_damage=float(
                variant_row[
                    "player_damage"
                ]
            ),
            opponent_damage=float(
                variant_row[
                    "opponent_damage"
                ]
            ),
            prize_cards_remaining=int(
                variant_row[
                    "prize_cards_remaining"
                ]
            ),
            hand_size=int(
                variant_row[
                    "hand_size"
                ]
            ),
        )
    )

    legal_moves = list(
        get_current_legal_moves(
            battle_state
        )
    )

    return (
        battle_state,
        state_metadata,
        legal_moves,
    )


# ------------------------------------------------------------
# 8B.7 — Initialize or load progress
# ------------------------------------------------------------

variant_work_df = (
    accepted_variant_df
    .copy()
    .reset_index(
        drop=True
    )
)


if SECTION8B_BATCH_LIMIT is not None:

    variant_work_df = (
        variant_work_df
        .head(
            int(
                SECTION8B_BATCH_LIMIT
            )
        )
        .copy()
    )


required_progress_columns = {
    "variant_id":
        "",

    "status":
        "PENDING",

    "attempts":
        0,

    "last_error":
        "",

    "search_depth":
        SECTION8B_SEARCH_DEPTH,

    "completed_at":
        "",
}


if SECTION8B_PROGRESS_FILE.exists():

    existing_progress_df = pd.read_csv(
        SECTION8B_PROGRESS_FILE
    )

else:

    existing_progress_df = pd.DataFrame(
        columns=list(
            required_progress_columns
        )
    )


progress_base_df = variant_work_df[
    [
        "variant_id",
    ]
].copy()


search_progress_df = (
    progress_base_df
    .merge(
        existing_progress_df,
        on="variant_id",
        how="left",
    )
)


for column, default_value in required_progress_columns.items():

    if column == "variant_id":
        continue

    if column not in search_progress_df.columns:

        search_progress_df[
            column
        ] = default_value

    else:

        search_progress_df[
            column
        ] = (
            search_progress_df[
                column
            ]
            .fillna(
                default_value
            )
            .infer_objects(
                copy=False
            )
        )


search_progress_df[
    "attempts"
] = (
    search_progress_df[
        "attempts"
    ]
    .astype(int)
)


search_progress_df.to_csv(
    SECTION8B_PROGRESS_FILE,
    index=False,
)


print()
print("INITIAL PROGRESS")
print("-" * 100)

display(
    search_progress_df[
        "status"
    ].value_counts(
        dropna=False
    )
)


# ------------------------------------------------------------
# 8B.8 — Load prior results
# ------------------------------------------------------------

if SECTION8B_RESULTS_FILE.exists():

    prior_results_df = pd.read_csv(
        SECTION8B_RESULTS_FILE
    )

else:

    prior_results_df = pd.DataFrame()


if (
    SECTION8B_FAILURES_FILE.exists()
    and SECTION8B_FAILURES_FILE.stat().st_size > 0
):
    try:
        prior_failures_df = pd.read_csv(
            SECTION8B_FAILURES_FILE
        )
    except pd.errors.EmptyDataError:
        prior_failures_df = pd.DataFrame()
else:
    prior_failures_df = pd.DataFrame()

completed_variant_ids = set()


if (
    SECTION8B_SKIP_COMPLETED
    and not prior_results_df.empty
    and "variant_id" in prior_results_df.columns
):

    completed_variant_ids = set(
        prior_results_df[
            "variant_id"
        ].astype(str)
    )


print()
print("RESUME STATUS")
print("-" * 100)

print(
    "Prior completed results :",
    len(
        completed_variant_ids
    ),
)

print(
    "Variants in this run    :",
    len(
        variant_work_df
    ),
)


# ------------------------------------------------------------
# 8B.9 — One-variant search function
# ------------------------------------------------------------

def search_one_variant(
    variant_row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:

    variant_id = str(
        variant_row[
            "variant_id"
        ]
    )

    started_at = time.perf_counter()

    try:

        battle_state, state_metadata, legal_moves = (
            reconstruct_variant_state(
                variant_row
            )
        )

        if not legal_moves:

            raise ValueError(
                "Reconstructed state has no legal moves."
            )

        search_result = search_engine.search_depth(
            state=battle_state,
            depth=SECTION8B_SEARCH_DEPTH,
            root_player=battle_state.current_player,
            clear_table=SECTION8B_CLEAR_TABLE_EACH_STATE,
            clear_killers=SECTION8B_CLEAR_KILLERS_EACH_STATE,
            clear_history=SECTION8B_CLEAR_HISTORY_EACH_STATE,
        )

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        parsed_result = parse_search_result(
            search_result
        )

        result_record = {
            "variant_id":
                variant_id,

            "source_scenario_id":
                variant_row[
                    "source_scenario_id"
                ],

            "variant_name":
                variant_row[
                    "variant_name"
                ],

            "side_mode":
                variant_row[
                    "side_mode"
                ],

            "source_normalized_side":
                variant_row[
                    "source_normalized_side"
                ],

            "variant_normalized_side":
                variant_row[
                    "variant_normalized_side"
                ],

            "current_player":
                battle_state.current_player,

            "baseline_name":
                variant_row.get(
                    "baseline_name"
                ),

            "player_card":
                variant_row[
                    "player_card"
                ],

            "opponent_card":
                variant_row[
                    "opponent_card"
                ],

            "turn_number":
                int(
                    variant_row[
                        "turn_number"
                    ]
                ),

            "player_energy":
                int(
                    variant_row[
                        "player_energy"
                    ]
                ),

            "opponent_energy":
                int(
                    variant_row[
                        "opponent_energy"
                    ]
                ),

            "player_damage":
                float(
                    variant_row[
                        "player_damage"
                    ]
                ),

            "opponent_damage":
                float(
                    variant_row[
                        "opponent_damage"
                    ]
                ),

            "prize_cards_remaining":
                int(
                    variant_row[
                        "prize_cards_remaining"
                    ]
                ),

            "hand_size":
                int(
                    variant_row[
                        "hand_size"
                    ]
                ),

            "legal_move_count":
                int(
                    len(
                        legal_moves
                    )
                ),

            "legal_move_reprs":
                json.dumps(
                    [
                        move_display_string(
                            move
                        )
                        for move in legal_moves
                    ],
                    ensure_ascii=False,
                ),

            "search_depth_requested":
                int(
                    SECTION8B_SEARCH_DEPTH
                ),

            "search_elapsed_seconds":
                float(
                    elapsed_seconds
                ),

            "search_result_type":
                parsed_result[
                    "search_result_type"
                ],

            "search_result_module":
                parsed_result[
                    "search_result_module"
                ],

            "search_best_move_repr":
                parsed_result[
                    "best_move_repr"
                ],

            "search_best_move_key":
                json.dumps(
                    json_safe_value(
                        parsed_result[
                            "best_move_key"
                        ]
                    ),
                    ensure_ascii=False,
                ),

            "search_score":
                parsed_result[
                    "score"
                ],

            "search_depth_reached":
                parsed_result[
                    "depth_reached"
                ],

            "search_nodes":
                parsed_result[
                    "nodes"
                ],

            "search_timed_out":
                parsed_result[
                    "timed_out"
                ],

            "principal_variation":
                json.dumps(
                    json_safe_value(
                        parsed_result[
                            "principal_variation"
                        ]
                    ),
                    ensure_ascii=False,
                ),

            "state_key":
                json.dumps(
                    json_safe_value(
                        notebook49_state_key(
                            battle_state
                        )
                    ),
                    ensure_ascii=False,
                ),

            "raw_result":
                json.dumps(
                    parsed_result[
                        "raw_result"
                    ],
                    ensure_ascii=False,
                    default=str,
                ),

            "status":
                "COMPLETED",
        }

        return (
            result_record,
            None,
        )

    except Exception as exc:

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        failure_record = {
            "variant_id":
                variant_id,

            "source_scenario_id":
                variant_row.get(
                    "source_scenario_id"
                ),

            "variant_name":
                variant_row.get(
                    "variant_name"
                ),

            "player_card":
                variant_row.get(
                    "player_card"
                ),

            "opponent_card":
                variant_row.get(
                    "opponent_card"
                ),

            "current_side":
                variant_row.get(
                    "variant_normalized_side"
                ),

            "search_depth":
                SECTION8B_SEARCH_DEPTH,

            "elapsed_seconds":
                elapsed_seconds,

            "error_type":
                type(
                    exc
                ).__name__,

            "error":
                str(
                    exc
                ),

            "status":
                "FAILED",
        }

        return (
            None,
            failure_record,
        )


# ------------------------------------------------------------
# 8B.10 — Batch processing
# ------------------------------------------------------------

new_result_rows = []

new_failure_rows = []


variant_records = variant_work_df.to_dict(
    orient="records"
)


for position, variant_row in enumerate(
    variant_records,
    start=1,
):

    variant_id = str(
        variant_row[
            "variant_id"
        ]
    )

    if (
        SECTION8B_SKIP_COMPLETED
        and variant_id in completed_variant_ids
    ):

        continue


    progress_index = search_progress_df[
        "variant_id"
    ].astype(str).eq(
        variant_id
    )


    current_status = (
        search_progress_df.loc[
            progress_index,
            "status",
        ]
        .iloc[
            0
        ]
    )


    if (
        current_status == "FAILED"
        and not SECTION8B_RETRY_FAILED
    ):

        continue


    search_progress_df.loc[
        progress_index,
        "status",
    ] = "IN_PROGRESS"

    search_progress_df.loc[
        progress_index,
        "attempts",
    ] = (
        search_progress_df.loc[
            progress_index,
            "attempts",
        ]
        .astype(int)
        + 1
    )


    result_record, failure_record = search_one_variant(
        variant_row
    )


    if result_record is not None:

        new_result_rows.append(
            result_record
        )

        search_progress_df.loc[
            progress_index,
            "status",
        ] = "COMPLETED"

        search_progress_df.loc[
            progress_index,
            "last_error",
        ] = ""

        search_progress_df.loc[
            progress_index,
            "completed_at",
        ] = pd.Timestamp.now().isoformat()


        with open(
            SECTION8B_JSONL_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    result_record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


    if failure_record is not None:

        new_failure_rows.append(
            failure_record
        )

        search_progress_df.loc[
            progress_index,
            "status",
        ] = "FAILED"

        search_progress_df.loc[
            progress_index,
            "last_error",
        ] = failure_record[
            "error"
        ]


    if (
        position % SECTION8B_CHECKPOINT_INTERVAL == 0
        or position == len(
            variant_records
        )
    ):

        search_progress_df.to_csv(
            SECTION8B_PROGRESS_FILE,
            index=False,
        )

        print(
            f"Checkpoint {position}/"
            f"{len(variant_records)} | "
            f"new completed={len(new_result_rows)} | "
            f"new failed={len(new_failure_rows)}"
        )


# ------------------------------------------------------------
# 8B.11 — Combine and save results
# ------------------------------------------------------------

new_results_df = pd.DataFrame(
    new_result_rows
)

new_failures_df = pd.DataFrame(
    new_failure_rows
)


if prior_results_df.empty:

    combined_results_df = (
        new_results_df.copy()
    )

elif new_results_df.empty:

    combined_results_df = (
        prior_results_df.copy()
    )

else:

    combined_results_df = pd.concat(
        [
            prior_results_df,
            new_results_df,
        ],
        ignore_index=True,
    )


if not combined_results_df.empty:

    combined_results_df = (
        combined_results_df
        .drop_duplicates(
            subset=[
                "variant_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "source_scenario_id",
                "variant_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


if prior_failures_df.empty:

    combined_failures_df = (
        new_failures_df.copy()
    )

elif new_failures_df.empty:

    combined_failures_df = (
        prior_failures_df.copy()
    )

else:

    combined_failures_df = pd.concat(
        [
            prior_failures_df,
            new_failures_df,
        ],
        ignore_index=True,
    )


if not combined_failures_df.empty:

    combined_failures_df = (
        combined_failures_df
        .drop_duplicates(
            subset=[
                "variant_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


combined_results_df.to_csv(
    SECTION8B_RESULTS_FILE,
    index=False,
)

combined_failures_df.to_csv(
    SECTION8B_FAILURES_FILE,
    index=False,
)

search_progress_df.to_csv(
    SECTION8B_PROGRESS_FILE,
    index=False,
)


# ------------------------------------------------------------
# 8B.12 — Summary metrics
# ------------------------------------------------------------

completed_count = int(
    search_progress_df[
        "status"
    ].eq(
        "COMPLETED"
    ).sum()
)

failed_count = int(
    search_progress_df[
        "status"
    ].eq(
        "FAILED"
    ).sum()
)

pending_count = int(
    search_progress_df[
        "status"
    ].isin(
        [
            "PENDING",
            "IN_PROGRESS",
        ]
    ).sum()
)


if not combined_results_df.empty:

    average_search_seconds = float(
        combined_results_df[
            "search_elapsed_seconds"
        ].mean()
    )

    average_legal_moves = float(
        combined_results_df[
            "legal_move_count"
        ].mean()
    )

    results_with_best_move = int(
        combined_results_df[
            "search_best_move_repr"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .sum()
    )

else:

    average_search_seconds = 0.0

    average_legal_moves = 0.0

    results_with_best_move = 0


search_summary = {
    "status":
        (
            "READY"
            if completed_count > 0
            else "FAILED"
        ),

    "search_depth":
        int(
            SECTION8B_SEARCH_DEPTH
        ),

    "input_variants":
        int(
            len(
                variant_work_df
            )
        ),

    "completed_variants":
        completed_count,

    "failed_variants":
        failed_count,

    "pending_variants":
        pending_count,

    "results_with_best_move":
        results_with_best_move,

    "average_search_seconds":
        average_search_seconds,

    "average_legal_moves":
        average_legal_moves,

    "completion_rate":
        (
            completed_count
            / len(
                variant_work_df
            )
            if len(
                variant_work_df
            ) > 0
            else 0.0
        ),

    "next_stage":
        (
            "POLICY_SEARCH_DISAGREEMENT_MINING"
            if completed_count > 0
            else "REPAIR_BATCH_SEARCH"
        ),
}


with open(
    SECTION8B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        search_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8B.13 — Display summary
# ------------------------------------------------------------

print()
print("SECTION 8B SEARCH SUMMARY")
print("-" * 100)

print(
    "Input variants          :",
    search_summary[
        "input_variants"
    ],
)

print(
    "Completed variants      :",
    search_summary[
        "completed_variants"
    ],
)

print(
    "Failed variants         :",
    search_summary[
        "failed_variants"
    ],
)

print(
    "Pending variants        :",
    search_summary[
        "pending_variants"
    ],
)

print(
    "Results with best move  :",
    search_summary[
        "results_with_best_move"
    ],
)

print(
    "Average search seconds  :",
    round(
        search_summary[
            "average_search_seconds"
        ],
        4,
    ),
)

print(
    "Completion rate         :",
    f"{search_summary['completion_rate']:.2%}",
)

print(
    "Next stage              :",
    search_summary[
        "next_stage"
    ],
)


if not combined_results_df.empty:

    print()
    print("SEARCH RESULT SAMPLE")
    print("-" * 100)

    display(
        combined_results_df[
            [
                column
                for column in [
                    "variant_id",
                    "source_scenario_id",
                    "variant_name",
                    "current_player",
                    "legal_move_count",
                    "search_best_move_repr",
                    "search_score",
                    "search_depth_reached",
                    "search_nodes",
                    "search_elapsed_seconds",
                ]
                if column in combined_results_df.columns
            ]
        ].head(
            20
        )
    )


if not combined_failures_df.empty:

    print()
    print("SEARCH FAILURE SUMMARY")
    print("-" * 100)

    display(
        combined_failures_df[
            "error"
        ].value_counts(
            dropna=False
        ).head(
            20
        )
    )


# ------------------------------------------------------------
# 8B.14 — Final validation
# ------------------------------------------------------------

assert SECTION8B_RESULTS_FILE.exists()

assert SECTION8B_FAILURES_FILE.exists()

assert SECTION8B_PROGRESS_FILE.exists()

assert SECTION8B_SUMMARY_FILE.exists()

assert completed_count > 0, (
    "Section 8B did not complete any search labels."
)

assert not combined_results_df.empty

assert combined_results_df[
    "variant_id"
].is_unique

assert search_summary[
    "next_stage"
] == "POLICY_SEARCH_DISAGREEMENT_MINING"


print()
print("SAVED SECTION 8B REPORTS")
print("-" * 100)

print(
    SECTION8B_RESULTS_FILE
)

print(
    SECTION8B_FAILURES_FILE
)

print(
    SECTION8B_PROGRESS_FILE
)

print(
    SECTION8B_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 8B RESUMABLE BATCH SEARCH LABELING PASSED"
)


# ## Section 8C — Expert Policy Dataset Construction
# 
# #### Section 8C converts the validated search labels from Section 8B into a clean,deduplicated expert-policy dataset.
# 
# #### Each accepted record represents one battle-state variant and contains:
# 
# - the reconstructed state identifier,
# - the legal actions available,
# - the expert move selected by search,
# - the search evaluation score,
# - completed search depth,
# - principal variation,
# - side and matchup metadata,
# - and curriculum-priority information.
# 
# #### Records with failed searches, missing expert moves, invalid state identifiers, or duplicate variant IDs are excluded.
# 
# #### The exported dataset becomes the supervised expert dataset used for policy improvement in the next notebook.

# In[ ]:


# ============================================================
# SECTION 8C
# EXPERT POLICY DATASET CONSTRUCTION
# ============================================================

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 8C — EXPERT POLICY DATASET CONSTRUCTION")
print("=" * 100)


# ------------------------------------------------------------
# 8C.1 — Preconditions
# ------------------------------------------------------------

assert "SECTION8B_RESULTS_FILE" in globals(), (
    "SECTION8B_RESULTS_FILE is unavailable. "
    "Run Section 8B first."
)

assert SECTION8B_RESULTS_FILE.exists(), (
    f"Missing Section 8B search labels: "
    f"{SECTION8B_RESULTS_FILE}"
)

assert "SECTION7C_VARIANT_MANIFEST_FILE" in globals(), (
    "SECTION7C_VARIANT_MANIFEST_FILE is unavailable. "
    "Run Section 7C first."
)

assert SECTION7C_VARIANT_MANIFEST_FILE.exists(), (
    f"Missing Section 7C variant manifest: "
    f"{SECTION7C_VARIANT_MANIFEST_FILE}"
)


# ------------------------------------------------------------
# 8C.2 — Output paths
# ------------------------------------------------------------

SECTION8C_REPORT_DIR = (
    SECTION8_REPORT_DIR
    / "policy_dataset"
)

SECTION8C_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION8C_POLICY_DATASET_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_expert_policy_dataset.csv"
)

SECTION8C_POLICY_DATASET_JSONL_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_expert_policy_dataset.jsonl"
)

SECTION8C_REJECTED_RECORDS_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_rejected_records.csv"
)

SECTION8C_DATASET_STATISTICS_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_dataset_statistics.csv"
)

SECTION8C_SIDE_SUMMARY_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_side_summary.csv"
)

SECTION8C_VARIANT_SUMMARY_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_variant_summary.csv"
)

SECTION8C_SUMMARY_FILE = (
    SECTION8C_REPORT_DIR
    / "section8c_summary.json"
)


# ------------------------------------------------------------
# 8C.3 — Load Section 7C and Section 8B data
# ------------------------------------------------------------

search_labels_df = pd.read_csv(
    SECTION8B_RESULTS_FILE,
    low_memory=False,
)

variant_manifest_df = pd.read_csv(
    SECTION7C_VARIANT_MANIFEST_FILE,
    low_memory=False,
)


assert not search_labels_df.empty, (
    "Section 8B search-label file is empty."
)

assert not variant_manifest_df.empty, (
    "Section 7C variant manifest is empty."
)


print()
print("INPUT DATA")
print("-" * 100)

print(
    "Search labels          :",
    len(search_labels_df),
)

print(
    "Variant manifest rows  :",
    len(variant_manifest_df),
)

print(
    "Search scenarios       :",
    search_labels_df[
        "source_scenario_id"
    ].nunique(),
)


# ------------------------------------------------------------
# 8C.4 — Parsing and normalization helpers
# ------------------------------------------------------------

def normalize_boolean(
    value: Any,
    default: bool = False,
) -> bool:

    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, float) and pd.isna(value):
        return default

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "y",
        "completed",
        "ready",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "n",
        "failed",
    }:
        return False

    return default


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:

    try:

        result = float(value)

        if math.isfinite(result):
            return result

    except Exception:
        pass

    return default


def safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:

    try:

        if value is None:
            return default

        if isinstance(value, float) and pd.isna(value):
            return default

        return int(float(value))

    except Exception:
        return default


def parse_json_like(
    value: Any,
    default: Any,
) -> Any:

    if value is None:
        return default

    if isinstance(value, float) and pd.isna(value):
        return default

    if isinstance(
        value,
        (
            list,
            tuple,
            dict,
        ),
    ):
        return value

    text = str(value).strip()

    if not text:
        return default

    try:
        return json.loads(text)

    except Exception:
        pass

    try:
        return ast.literal_eval(text)

    except Exception:
        return default


def normalized_move_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return text


def normalized_side_name(
    value: Any,
) -> str:

    normalized = (
        str(value)
        .strip()
        .lower()
    )

    if "opponent" in normalized:
        return "Opponent"

    if "player" in normalized:
        return "Player"

    return str(value).strip()


# ------------------------------------------------------------
# 8C.5 — Select merge columns from the variant manifest
# ------------------------------------------------------------

variant_metadata_columns = [
    column
    for column in [
        "variant_id",
        "variant_seed",
        "source_match_id",
        "queue_position",
        "curriculum_priority",
        "hard_example_rank",
        "side_mode",
        "source_normalized_side",
        "variant_normalized_side",
        "baseline_name",
        "player_card",
        "opponent_card",
        "turn_number",
        "player_energy",
        "opponent_energy",
        "player_damage",
        "opponent_damage",
        "player_damage_fraction",
        "opponent_damage_fraction",
        "prize_cards_remaining",
        "hand_size",
        "legal_move_count",
        "trace_file",
        "state_key",
    ]
    if column in variant_manifest_df.columns
]


variant_metadata_df = (
    variant_manifest_df[
        variant_metadata_columns
    ]
    .copy()
)


# Rename overlapping fields before merging.
overlapping_columns = [
    column
    for column in variant_metadata_df.columns
    if column != "variant_id"
    and column in search_labels_df.columns
]


variant_metadata_df = variant_metadata_df.rename(
    columns={
        column:
            f"manifest_{column}"

        for column in overlapping_columns
    }
)


merged_policy_df = search_labels_df.merge(
    variant_metadata_df,
    on="variant_id",
    how="left",
    validate="one_to_one",
)


assert len(merged_policy_df) == len(search_labels_df), (
    "Search labels changed size during manifest merge."
)


print()
print("MERGED DATASET")
print("-" * 100)

print(
    "Merged rows            :",
    len(merged_policy_df),
)

print(
    "Matched manifest rows  :",
    int(
        merged_policy_df[
            [
                column
                for column in merged_policy_df.columns
                if column.startswith(
                    "manifest_"
                )
            ][0]
        ]
        .notna()
        .sum()
    )
    if any(
        column.startswith("manifest_")
        for column in merged_policy_df.columns
    )
    else len(merged_policy_df),
)


# ------------------------------------------------------------
# 8C.6 — Build normalized expert-policy records
# ------------------------------------------------------------

expert_record_rows = []

rejected_record_rows = []


for source_row in merged_policy_df.to_dict(
    orient="records"
):

    variant_id = str(
        source_row.get(
            "variant_id",
            "",
        )
    ).strip()

    status = str(
        source_row.get(
            "status",
            "",
        )
    ).strip().upper()

    best_move_text = normalized_move_text(
        source_row.get(
            "search_best_move_repr"
        )
    )

    state_key_text = normalized_move_text(
        source_row.get(
            "state_key"
        )
    )

    legal_move_reprs = parse_json_like(
        source_row.get(
            "legal_move_reprs"
        ),
        default=[],
    )

    principal_variation = parse_json_like(
        source_row.get(
            "principal_variation"
        ),
        default=[],
    )

    search_best_move_key = parse_json_like(
        source_row.get(
            "search_best_move_key"
        ),
        default=None,
    )

    raw_search_result = parse_json_like(
        source_row.get(
            "raw_result"
        ),
        default={},
    )

    search_score = safe_float(
        source_row.get(
            "search_score"
        )
    )

    search_depth_requested = safe_int(
        source_row.get(
            "search_depth_requested"
        )
    )

    search_depth_reached = safe_int(
        source_row.get(
            "search_depth_reached"
        )
    )

    legal_move_count = safe_int(
        source_row.get(
            "legal_move_count"
        ),
        default=0,
    )

    rejection_reasons = []

    if not variant_id:
        rejection_reasons.append(
            "MISSING_VARIANT_ID"
        )

    if status != "COMPLETED":
        rejection_reasons.append(
            "SEARCH_NOT_COMPLETED"
        )

    if not best_move_text:
        rejection_reasons.append(
            "MISSING_BEST_MOVE"
        )

    if search_score is None:
        rejection_reasons.append(
            "MISSING_SEARCH_SCORE"
        )

    if not state_key_text:
        rejection_reasons.append(
            "MISSING_STATE_KEY"
        )

    if legal_move_count is None or legal_move_count <= 0:
        rejection_reasons.append(
            "NO_LEGAL_MOVES"
        )

    if not isinstance(legal_move_reprs, list):
        rejection_reasons.append(
            "INVALID_LEGAL_MOVE_LIST"
        )

    if rejection_reasons:

        rejected_record_rows.append(
            {
                "variant_id":
                    variant_id,

                "source_scenario_id":
                    source_row.get(
                        "source_scenario_id"
                    ),

                "variant_name":
                    source_row.get(
                        "variant_name"
                    ),

                "status":
                    status,

                "search_best_move_repr":
                    best_move_text,

                "search_score":
                    search_score,

                "legal_move_count":
                    legal_move_count,

                "rejection_reasons":
                    "|".join(
                        rejection_reasons
                    ),
            }
        )

        continue


    source_side = (
        source_row.get(
            "source_normalized_side"
        )
        or source_row.get(
            "manifest_source_normalized_side"
        )
    )

    variant_side = (
        source_row.get(
            "variant_normalized_side"
        )
        or source_row.get(
            "manifest_variant_normalized_side"
        )
    )

    side_mode = (
        source_row.get(
            "side_mode"
        )
        or source_row.get(
            "manifest_side_mode"
        )
    )

    expert_record = {
        "example_id":
            variant_id,

        "variant_id":
            variant_id,

        "source_scenario_id":
            str(
                source_row.get(
                    "source_scenario_id",
                    "",
                )
            ),

        "source_match_id":
            (
                source_row.get(
                    "source_match_id"
                )
                or source_row.get(
                    "manifest_source_match_id"
                )
            ),

        "variant_name":
            str(
                source_row.get(
                    "variant_name",
                    "",
                )
            ),

        "variant_seed":
            safe_int(
                source_row.get(
                    "variant_seed"
                )
                or source_row.get(
                    "manifest_variant_seed"
                )
            ),

        "queue_position":
            safe_int(
                source_row.get(
                    "queue_position"
                )
                or source_row.get(
                    "manifest_queue_position"
                )
            ),

        "curriculum_priority":
            safe_float(
                source_row.get(
                    "curriculum_priority"
                )
                or source_row.get(
                    "manifest_curriculum_priority"
                )
            ),

        "hard_example_rank":
            safe_int(
                source_row.get(
                    "hard_example_rank"
                )
                or source_row.get(
                    "manifest_hard_example_rank"
                )
            ),

        "side_mode":
            str(
                side_mode
                if side_mode is not None
                else ""
            ),

        "source_side":
            normalized_side_name(
                source_side
            ),

        "current_side":
            normalized_side_name(
                source_row.get(
                    "current_player"
                )
                or variant_side
            ),

        "baseline_name":
            (
                source_row.get(
                    "baseline_name"
                )
                or source_row.get(
                    "manifest_baseline_name"
                )
            ),

        "player_card":
            str(
                source_row.get(
                    "player_card"
                )
                or source_row.get(
                    "manifest_player_card"
                )
                or ""
            ),

        "opponent_card":
            str(
                source_row.get(
                    "opponent_card"
                )
                or source_row.get(
                    "manifest_opponent_card"
                )
                or ""
            ),

        "turn_number":
            safe_int(
                source_row.get(
                    "turn_number"
                )
                or source_row.get(
                    "manifest_turn_number"
                ),
                default=1,
            ),

        "player_energy":
            safe_int(
                source_row.get(
                    "player_energy"
                )
                or source_row.get(
                    "manifest_player_energy"
                ),
                default=0,
            ),

        "opponent_energy":
            safe_int(
                source_row.get(
                    "opponent_energy"
                )
                or source_row.get(
                    "manifest_opponent_energy"
                ),
                default=0,
            ),

        "player_damage":
            safe_float(
                source_row.get(
                    "player_damage"
                )
                or source_row.get(
                    "manifest_player_damage"
                ),
                default=0.0,
            ),

        "opponent_damage":
            safe_float(
                source_row.get(
                    "opponent_damage"
                )
                or source_row.get(
                    "manifest_opponent_damage"
                ),
                default=0.0,
            ),

        "prize_cards_remaining":
            safe_int(
                source_row.get(
                    "prize_cards_remaining"
                )
                or source_row.get(
                    "manifest_prize_cards_remaining"
                ),
                default=6,
            ),

        "hand_size":
            safe_int(
                source_row.get(
                    "hand_size"
                )
                or source_row.get(
                    "manifest_hand_size"
                ),
                default=7,
            ),

        "state_key":
            state_key_text,

        "legal_move_count":
            int(
                legal_move_count
            ),

        "legal_moves":
            json.dumps(
                legal_move_reprs,
                ensure_ascii=False,
            ),

        "expert_move":
            best_move_text,

        "expert_move_key":
            json.dumps(
                search_best_move_key,
                ensure_ascii=False,
                default=str,
            ),

        "value_target":
            float(
                search_score
            ),

        "search_depth_requested":
            search_depth_requested,

        "search_depth_reached":
            search_depth_reached,

        "search_nodes":
            safe_int(
                source_row.get(
                    "search_nodes"
                )
            ),

        "search_elapsed_seconds":
            safe_float(
                source_row.get(
                    "search_elapsed_seconds"
                ),
                default=0.0,
            ),

        "search_timed_out":
            normalize_boolean(
                source_row.get(
                    "search_timed_out"
                ),
                default=False,
            ),

        "principal_variation":
            json.dumps(
                principal_variation,
                ensure_ascii=False,
                default=str,
            ),

        "raw_search_result":
            json.dumps(
                raw_search_result,
                ensure_ascii=False,
                default=str,
            ),

        "search_success":
            True,

        "sample_weight":
            1.0,
    }

    expert_record_rows.append(
        expert_record
    )


expert_policy_df = pd.DataFrame(
    expert_record_rows
)

rejected_records_df = pd.DataFrame(
    rejected_record_rows
)


print()
print("INITIAL QUALITY FILTER")
print("-" * 100)

print(
    "Accepted examples  :",
    len(expert_policy_df),
)

print(
    "Rejected examples  :",
    len(rejected_records_df),
)


# ------------------------------------------------------------
# 8C.7 — Remove exact duplicate examples
# ------------------------------------------------------------

rows_before_deduplication = len(
    expert_policy_df
)


expert_policy_df = (
    expert_policy_df
    .drop_duplicates(
        subset=[
            "variant_id",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


duplicates_removed = (
    rows_before_deduplication
    - len(
        expert_policy_df
    )
)


# ------------------------------------------------------------
# 8C.8 — Assign curriculum-aware sample weights
# ------------------------------------------------------------

def calculate_sample_weight(
    row: pd.Series,
) -> float:

    weight = 1.0

    current_side = str(
        row.get(
            "current_side",
            "",
        )
    ).strip().lower()

    side_mode_value = str(
        row.get(
            "side_mode",
            "",
        )
    ).strip().upper()

    depth_reached = safe_int(
        row.get(
            "search_depth_reached"
        ),
        default=0,
    )

    legal_moves = safe_int(
        row.get(
            "legal_move_count"
        ),
        default=1,
    )

    if current_side == "opponent":
        weight *= 1.35

    if side_mode_value == "PRESERVE":
        weight *= 1.10

    if side_mode_value == "FLIP":
        weight *= 0.90

    if depth_reached is not None and depth_reached >= 4:
        weight *= 1.10

    if legal_moves is not None and legal_moves > 1:
        weight *= 1.15

    return float(
        round(
            weight,
            4,
        )
    )


if not expert_policy_df.empty:

    expert_policy_df[
        "sample_weight"
    ] = expert_policy_df.apply(
        calculate_sample_weight,
        axis=1,
    )


# ------------------------------------------------------------
# 8C.9 — Final schema ordering
# ------------------------------------------------------------

final_policy_columns = [
    "example_id",
    "variant_id",
    "source_scenario_id",
    "source_match_id",
    "variant_name",
    "variant_seed",
    "queue_position",
    "curriculum_priority",
    "hard_example_rank",
    "side_mode",
    "source_side",
    "current_side",
    "baseline_name",
    "player_card",
    "opponent_card",
    "turn_number",
    "player_energy",
    "opponent_energy",
    "player_damage",
    "opponent_damage",
    "prize_cards_remaining",
    "hand_size",
    "state_key",
    "legal_move_count",
    "legal_moves",
    "expert_move",
    "expert_move_key",
    "value_target",
    "search_depth_requested",
    "search_depth_reached",
    "search_nodes",
    "search_elapsed_seconds",
    "search_timed_out",
    "principal_variation",
    "raw_search_result",
    "search_success",
    "sample_weight",
]


for column in final_policy_columns:

    if column not in expert_policy_df.columns:
        expert_policy_df[
            column
        ] = np.nan


expert_policy_df = expert_policy_df[
    final_policy_columns
]


# ------------------------------------------------------------
# 8C.10 — Dataset integrity checks
# ------------------------------------------------------------

missing_required_field_counts = {
    "variant_id":
        int(
            expert_policy_df[
                "variant_id"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),

    "state_key":
        int(
            expert_policy_df[
                "state_key"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),

    "expert_move":
        int(
            expert_policy_df[
                "expert_move"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),

    "value_target":
        int(
            expert_policy_df[
                "value_target"
            ]
            .isna()
            .sum()
        ),
}


duplicate_variant_count = int(
    expert_policy_df[
        "variant_id"
    ].duplicated().sum()
)


invalid_legal_move_count = int(
    expert_policy_df[
        "legal_move_count"
    ].le(0).sum()
)


invalid_search_success_count = int(
    (
        ~expert_policy_df[
            "search_success"
        ].astype(bool)
    ).sum()
)


print()
print("DATASET INTEGRITY")
print("-" * 100)

print(
    "Final examples             :",
    len(expert_policy_df),
)

print(
    "Duplicates removed         :",
    duplicates_removed,
)

print(
    "Duplicate variants remain  :",
    duplicate_variant_count,
)

print(
    "Invalid legal-move rows     :",
    invalid_legal_move_count,
)

print(
    "Invalid success rows        :",
    invalid_search_success_count,
)

print(
    "Missing required fields     :",
    missing_required_field_counts,
)


# ------------------------------------------------------------
# 8C.11 — Dataset statistics
# ------------------------------------------------------------

dataset_statistics_rows = [
    {
        "metric":
            "search_label_rows",

        "value":
            int(
                len(
                    search_labels_df
                )
            ),
    },
    {
        "metric":
            "accepted_policy_examples",

        "value":
            int(
                len(
                    expert_policy_df
                )
            ),
    },
    {
        "metric":
            "rejected_examples",

        "value":
            int(
                len(
                    rejected_records_df
                )
            ),
    },
    {
        "metric":
            "duplicates_removed",

        "value":
            int(
                duplicates_removed
            ),
    },
    {
        "metric":
            "source_scenarios",

        "value":
            int(
                expert_policy_df[
                    "source_scenario_id"
                ].nunique()
            ),
    },
    {
        "metric":
            "unique_matchups",

        "value":
            int(
                expert_policy_df[
                    [
                        "player_card",
                        "opponent_card",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
    },
    {
        "metric":
            "opponent_side_examples",

        "value":
            int(
                expert_policy_df[
                    "current_side"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "opponent"
                )
                .sum()
            ),
    },
    {
        "metric":
            "player_side_examples",

        "value":
            int(
                expert_policy_df[
                    "current_side"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "player"
                )
                .sum()
            ),
    },
    {
        "metric":
            "average_value_target",

        "value":
            float(
                expert_policy_df[
                    "value_target"
                ].mean()
            ),
    },
    {
        "metric":
            "average_search_depth_reached",

        "value":
            float(
                expert_policy_df[
                    "search_depth_reached"
                ].mean()
            ),
    },
    {
        "metric":
            "average_search_seconds",

        "value":
            float(
                expert_policy_df[
                    "search_elapsed_seconds"
                ].mean()
            ),
    },
    {
        "metric":
            "average_sample_weight",

        "value":
            float(
                expert_policy_df[
                    "sample_weight"
                ].mean()
            ),
    },
]


dataset_statistics_df = pd.DataFrame(
    dataset_statistics_rows
)


side_summary_df = (
    expert_policy_df
    .groupby(
        [
            "current_side",
            "side_mode",
        ],
        dropna=False,
    )
    .agg(
        examples=(
            "variant_id",
            "count",
        ),
        source_scenarios=(
            "source_scenario_id",
            "nunique",
        ),
        average_value_target=(
            "value_target",
            "mean",
        ),
        average_sample_weight=(
            "sample_weight",
            "mean",
        ),
        average_search_depth=(
            "search_depth_reached",
            "mean",
        ),
    )
    .reset_index()
)


variant_summary_df = (
    expert_policy_df
    .groupby(
        [
            "variant_name",
            "current_side",
        ],
        dropna=False,
    )
    .agg(
        examples=(
            "variant_id",
            "count",
        ),
        source_scenarios=(
            "source_scenario_id",
            "nunique",
        ),
        average_value_target=(
            "value_target",
            "mean",
        ),
        average_search_seconds=(
            "search_elapsed_seconds",
            "mean",
        ),
        average_sample_weight=(
            "sample_weight",
            "mean",
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8C.12 — Save the expert-policy dataset
# ------------------------------------------------------------

expert_policy_df.to_csv(
    SECTION8C_POLICY_DATASET_FILE,
    index=False,
)


with open(
    SECTION8C_POLICY_DATASET_JSONL_FILE,
    "w",
    encoding="utf-8",
) as file:

    for record in expert_policy_df.to_dict(
        orient="records"
    ):

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


rejected_records_df.to_csv(
    SECTION8C_REJECTED_RECORDS_FILE,
    index=False,
)

dataset_statistics_df.to_csv(
    SECTION8C_DATASET_STATISTICS_FILE,
    index=False,
)

side_summary_df.to_csv(
    SECTION8C_SIDE_SUMMARY_FILE,
    index=False,
)

variant_summary_df.to_csv(
    SECTION8C_VARIANT_SUMMARY_FILE,
    index=False,
)


# ------------------------------------------------------------
# 8C.13 — Save summary report
# ------------------------------------------------------------

section8c_summary = {
    "status":
        (
            "READY"
            if len(
                expert_policy_df
            ) > 0
            else "FAILED"
        ),

    "input_search_labels":
        int(
            len(
                search_labels_df
            )
        ),

    "accepted_policy_examples":
        int(
            len(
                expert_policy_df
            )
        ),

    "rejected_examples":
        int(
            len(
                rejected_records_df
            )
        ),

    "duplicates_removed":
        int(
            duplicates_removed
        ),

    "source_scenarios":
        int(
            expert_policy_df[
                "source_scenario_id"
            ].nunique()
        ),

    "player_side_examples":
        int(
            expert_policy_df[
                "current_side"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "player"
            )
            .sum()
        ),

    "opponent_side_examples":
        int(
            expert_policy_df[
                "current_side"
            ]
            .astype(str)
            .str.lower()
            .eq(
                "opponent"
            )
            .sum()
        ),

    "preserved_side_examples":
        int(
            expert_policy_df[
                "side_mode"
            ]
            .astype(str)
            .str.upper()
            .eq(
                "PRESERVE"
            )
            .sum()
        ),

    "counterfactual_side_examples":
        int(
            expert_policy_df[
                "side_mode"
            ]
            .astype(str)
            .str.upper()
            .eq(
                "FLIP"
            )
            .sum()
        ),

    "missing_required_fields":
        missing_required_field_counts,

    "duplicate_variant_count":
        duplicate_variant_count,

    "invalid_legal_move_count":
        invalid_legal_move_count,

    "average_value_target":
        float(
            expert_policy_df[
                "value_target"
            ].mean()
        ),

    "average_sample_weight":
        float(
            expert_policy_df[
                "sample_weight"
            ].mean()
        ),

    "policy_dataset_file":
        str(
            SECTION8C_POLICY_DATASET_FILE
        ),

    "policy_jsonl_file":
        str(
            SECTION8C_POLICY_DATASET_JSONL_FILE
        ),

    "next_stage":
        (
            "FINAL_DATASET_VALIDATION"
            if len(
                expert_policy_df
            ) > 0
            else "REPAIR_POLICY_DATASET"
        ),
}


with open(
    SECTION8C_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section8c_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8C.14 — Display results
# ------------------------------------------------------------

print()
print("SECTION 8C POLICY DATASET SUMMARY")
print("-" * 100)

print(
    "Input search labels       :",
    section8c_summary[
        "input_search_labels"
    ],
)

print(
    "Accepted policy examples  :",
    section8c_summary[
        "accepted_policy_examples"
    ],
)

print(
    "Rejected examples         :",
    section8c_summary[
        "rejected_examples"
    ],
)

print(
    "Duplicates removed        :",
    section8c_summary[
        "duplicates_removed"
    ],
)

print(
    "Source scenarios          :",
    section8c_summary[
        "source_scenarios"
    ],
)

print(
    "Player-side examples      :",
    section8c_summary[
        "player_side_examples"
    ],
)

print(
    "Opponent-side examples    :",
    section8c_summary[
        "opponent_side_examples"
    ],
)

print(
    "Preserved-side examples   :",
    section8c_summary[
        "preserved_side_examples"
    ],
)

print(
    "Counterfactual examples   :",
    section8c_summary[
        "counterfactual_side_examples"
    ],
)

print(
    "Average value target      :",
    round(
        section8c_summary[
            "average_value_target"
        ],
        4,
    ),
)

print(
    "Average sample weight     :",
    round(
        section8c_summary[
            "average_sample_weight"
        ],
        4,
    ),
)

print(
    "Next stage                :",
    section8c_summary[
        "next_stage"
    ],
)


print()
print("SIDE DISTRIBUTION")
print("-" * 100)

display(
    side_summary_df
)


print()
print("VARIANT DISTRIBUTION")
print("-" * 100)

display(
    variant_summary_df
)


print()
print("EXPERT POLICY DATASET SAMPLE")
print("-" * 100)

display(
    expert_policy_df[
        [
            "variant_id",
            "source_scenario_id",
            "variant_name",
            "current_side",
            "player_card",
            "opponent_card",
            "legal_move_count",
            "expert_move",
            "value_target",
            "search_depth_reached",
            "sample_weight",
        ]
    ].head(
        20
    )
)


if not rejected_records_df.empty:

    print()
    print("REJECTION SUMMARY")
    print("-" * 100)

    display(
        rejected_records_df[
            "rejection_reasons"
        ]
        .value_counts(
            dropna=False
        )
    )


# ------------------------------------------------------------
# 8C.15 — Final validation
# ------------------------------------------------------------

assert SECTION8C_POLICY_DATASET_FILE.exists()

assert SECTION8C_POLICY_DATASET_JSONL_FILE.exists()

assert SECTION8C_REJECTED_RECORDS_FILE.exists()

assert SECTION8C_DATASET_STATISTICS_FILE.exists()

assert SECTION8C_SIDE_SUMMARY_FILE.exists()

assert SECTION8C_VARIANT_SUMMARY_FILE.exists()

assert SECTION8C_SUMMARY_FILE.exists()

assert not expert_policy_df.empty, (
    "Section 8C produced no expert-policy examples."
)

assert expert_policy_df[
    "variant_id"
].is_unique, (
    "Expert-policy variant IDs are not unique."
)

assert duplicate_variant_count == 0, (
    "Duplicate variant IDs remain in the expert-policy dataset."
)

assert invalid_legal_move_count == 0, (
    "Expert-policy dataset contains rows without legal moves."
)

assert invalid_search_success_count == 0, (
    "Expert-policy dataset contains unsuccessful search rows."
)

assert all(
    count == 0
    for count in missing_required_field_counts.values()
), (
    "Required expert-policy fields contain missing values: "
    f"{missing_required_field_counts}"
)

assert section8c_summary[
    "next_stage"
] == "FINAL_DATASET_VALIDATION"


print()
print("SAVED SECTION 8C REPORTS")
print("-" * 100)

print(
    SECTION8C_POLICY_DATASET_FILE
)

print(
    SECTION8C_POLICY_DATASET_JSONL_FILE
)

print(
    SECTION8C_REJECTED_RECORDS_FILE
)

print(
    SECTION8C_DATASET_STATISTICS_FILE
)

print(
    SECTION8C_SIDE_SUMMARY_FILE
)

print(
    SECTION8C_VARIANT_SUMMARY_FILE
)

print(
    SECTION8C_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 8C EXPERT POLICY DATASET CONSTRUCTION PASSED"
)


# In[ ]:


# ============================================================
# SECTION 8D DIAGNOSTIC
# TRACE ATTACKS, ENERGY, AND LEGAL-MOVE GENERATION
# ============================================================

from pprint import pprint

import pandas as pd


print("=" * 100)
print("SECTION 8D DIAGNOSTIC — ATTACK AND ENERGY TRACE")
print("=" * 100)


# ------------------------------------------------------------
# 1. Select useful diagnostic variants
# ------------------------------------------------------------

diagnostic_candidates_df = (
    accepted_variant_df.loc[
        (
            accepted_variant_df["player_energy"].astype(float).gt(0)
            | accepted_variant_df["opponent_energy"].astype(float).gt(0)
        )
        & ~accepted_variant_df["variant_name"].astype(str).eq(
            "opening_zero_energy"
        )
    ]
    .copy()
)


assert not diagnostic_candidates_df.empty, (
    "No non-zero-energy variants were found."
)


# Prefer examples from both current-player sides.
diagnostic_rows = []

for desired_side in ["Player", "Opponent"]:

    side_matches = diagnostic_candidates_df.loc[
        diagnostic_candidates_df[
            "variant_normalized_side"
        ].astype(str).eq(
            desired_side
        )
    ]

    if not side_matches.empty:

        # Prefer the row with the most Energy for the active side.
        if desired_side == "Player":

            selected_index = (
                side_matches["player_energy"]
                .astype(float)
                .idxmax()
            )

        else:

            selected_index = (
                side_matches["opponent_energy"]
                .astype(float)
                .idxmax()
            )

        diagnostic_rows.append(
            diagnostic_candidates_df.loc[
                selected_index
            ].to_dict()
        )


assert diagnostic_rows, (
    "Could not select diagnostic variants."
)


# ------------------------------------------------------------
# 2. Helper for displaying one PokémonState
# ------------------------------------------------------------

def inspect_pokemon_state(
    label,
    pokemon_state,
):

    print()
    print(label)
    print("-" * 100)

    print(
        "PokemonState type   :",
        type(pokemon_state).__name__,
    )

    print(
        "Attached Energy     :",
        getattr(
            pokemon_state,
            "attached_energy",
            "<missing attribute>",
        ),
    )

    print(
        "Current HP          :",
        getattr(
            pokemon_state,
            "current_hp",
            "<missing attribute>",
        ),
    )

    print(
        "Damage              :",
        getattr(
            pokemon_state,
            "damage",
            "<missing attribute>",
        ),
    )

    card_record = getattr(
        pokemon_state,
        "card",
        None,
    )

    print(
        "Card type           :",
        type(card_record).__name__,
    )

    if not isinstance(
        card_record,
        dict,
    ):

        print(
            "Card record         :",
            repr(card_record),
        )

        return

    card_name = (
        card_record.get(card_name_column)
        if "card_name_column" in globals()
        else card_record.get(
            "Name",
            card_record.get(
                "name",
                "<unknown>",
            ),
        )
    )

    attacks = card_record.get(
        "attacks",
        [],
    )

    print(
        "Card name           :",
        card_name,
    )

    print(
        "Has attacks key     :",
        "attacks" in card_record,
    )

    print(
        "Attack count        :",
        len(attacks)
        if isinstance(attacks, list)
        else "<not a list>",
    )

    print(
        "Attack container    :",
        type(attacks).__name__,
    )

    print(
        "Attacks:"
    )

    pprint(
        attacks
    )


# ------------------------------------------------------------
# 3. Reconstruct and inspect representative states
# ------------------------------------------------------------

diagnostic_summary_rows = []


for diagnostic_number, variant_row in enumerate(
    diagnostic_rows,
    start=1,
):

    print()
    print("=" * 100)
    print(
        f"DIAGNOSTIC VARIANT {diagnostic_number}"
    )
    print("=" * 100)

    print(
        "Variant ID          :",
        variant_row["variant_id"],
    )

    print(
        "Variant name        :",
        variant_row["variant_name"],
    )

    print(
        "Requested side      :",
        variant_row["variant_normalized_side"],
    )

    print(
        "Player card         :",
        variant_row["player_card"],
    )

    print(
        "Opponent card       :",
        variant_row["opponent_card"],
    )

    print(
        "Requested P Energy  :",
        variant_row["player_energy"],
    )

    print(
        "Requested O Energy  :",
        variant_row["opponent_energy"],
    )


    battle_state, state_metadata, legal_moves = (
        reconstruct_variant_state(
            variant_row
        )
    )


    print()
    print("BATTLE STATE")
    print("-" * 100)

    print(
        "Current player      :",
        battle_state.current_player,
    )

    print(
        "Turn number         :",
        battle_state.turn_number,
    )


    inspect_pokemon_state(
        "PLAYER ACTIVE POKÉMON",
        battle_state.player.active,
    )

    inspect_pokemon_state(
        "OPPONENT ACTIVE POKÉMON",
        battle_state.opponent.active,
    )


    if str(
        battle_state.current_player
    ).strip().lower() == "opponent":

        active_pokemon = (
            battle_state.opponent.active
        )

        active_label = "Opponent"

    else:

        active_pokemon = (
            battle_state.player.active
        )

        active_label = "Player"


    active_attacks = (
        active_pokemon.card.get(
            "attacks",
            [],
        )
        if isinstance(
            active_pokemon.card,
            dict,
        )
        else []
    )

    active_energy = getattr(
        active_pokemon,
        "attached_energy",
        None,
    )


    print()
    print("ACTIVE-SIDE LEGAL-MOVE CHECK")
    print("-" * 100)

    print(
        "Active side         :",
        active_label,
    )

    print(
        "Active Energy       :",
        active_energy,
    )

    print(
        "Active attacks      :",
        len(active_attacks),
    )


    affordability_rows = []

    for attack in active_attacks:

        attack_name = attack.get(
            "Move Name",
            attack.get(
                "name",
                "<unknown>",
            ),
        )

        energy_cost = attack.get(
            "energy_cost",
            attack.get(
                "cost",
                None,
            ),
        )

        try:

            affordable = (
                float(active_energy)
                >= float(energy_cost)
            )

        except Exception:

            affordable = False

        affordability_rows.append(
            {
                "attack_name":
                    attack_name,

                "energy_cost":
                    energy_cost,

                "attached_energy":
                    active_energy,

                "affordable":
                    affordable,

                "damage_numeric":
                    attack.get(
                        "damage_numeric",
                        attack.get(
                            "damage",
                            None,
                        ),
                    ),
            }
        )


    affordability_df = pd.DataFrame(
        affordability_rows
    )

    if affordability_df.empty:

        print(
            "No attacks were available on the active card."
        )

    else:

        display(
            affordability_df
        )


    direct_legal_moves = list(
        get_current_legal_moves(
            battle_state
        )
    )


    print()
    print("LEGAL MOVES RETURNED")
    print("-" * 100)

    print(
        "Legal move count    :",
        len(direct_legal_moves),
    )

    pprint(
        direct_legal_moves
    )


    diagnostic_summary_rows.append(
        {
            "variant_id":
                variant_row["variant_id"],

            "variant_name":
                variant_row["variant_name"],

            "requested_side":
                variant_row["variant_normalized_side"],

            "actual_current_player":
                battle_state.current_player,

            "requested_player_energy":
                variant_row["player_energy"],

            "stored_player_energy":
                getattr(
                    battle_state.player.active,
                    "attached_energy",
                    None,
                ),

            "requested_opponent_energy":
                variant_row["opponent_energy"],

            "stored_opponent_energy":
                getattr(
                    battle_state.opponent.active,
                    "attached_energy",
                    None,
                ),

            "active_attack_count":
                len(active_attacks),

            "affordable_attack_count":
                (
                    int(
                        affordability_df[
                            "affordable"
                        ].sum()
                    )
                    if not affordability_df.empty
                    else 0
                ),

            "legal_move_count":
                len(direct_legal_moves),

            "legal_moves":
                [
                    move_display_string(move)
                    for move in direct_legal_moves
                ],
        }
    )


# ------------------------------------------------------------
# 4. Diagnostic summary
# ------------------------------------------------------------

section8d_diagnostic_df = pd.DataFrame(
    diagnostic_summary_rows
)


print()
print("=" * 100)
print("SECTION 8D DIAGNOSTIC SUMMARY")
print("=" * 100)

display(
    section8d_diagnostic_df
)


problem_rows = section8d_diagnostic_df.loc[
    (
        section8d_diagnostic_df[
            "affordable_attack_count"
        ].gt(0)
    )
    & (
        section8d_diagnostic_df[
            "legal_move_count"
        ].le(1)
    )
]


if not problem_rows.empty:

    print()
    print(
        "⚠️ DIAGNOSIS: At least one state has an affordable "
        "attack, but legal-move generation still returns only one move."
    )

    print(
        "The next repair should be made inside "
        "get_legal_moves() or get_current_legal_moves()."
    )

else:

    zero_affordable_rows = (
        section8d_diagnostic_df.loc[
            section8d_diagnostic_df[
                "affordable_attack_count"
            ].eq(0)
        ]
    )

    if not zero_affordable_rows.empty:

        print()
        print(
            "⚠️ DIAGNOSIS: The inspected states contain no "
            "affordable attacks for the active Pokémon."
        )

        print(
            "Check attack energy costs and the Section 7C "
            "variant Energy assignments."
        )

    else:

        print()
        print(
            "✅ Attacks, Energy, and legal-move generation "
            "appear consistent in the inspected states."
        )


# ## Section 8D — Final Dataset Validation and Notebook 49 Handoff
# 
# #### Section 8D performs the final integrity review for Notebook 49.
# 
# #### It verifies that:
# 
# - all required Section 7 and Section 8 artifacts exist,
# - the expert-policy dataset is nonempty,
# - variant identifiers are unique,
# - required training fields are complete,
# - search labels are valid,
# - side and curriculum distributions are acceptable,
# - no failed or rejected records have leaked into the final dataset,
# - and the exported files are ready for the next training notebook.
# 
# #### If all checks pass, Notebook 49 is considered complete and ready for handoff.

# In[ ]:


# ============================================================
# SECTION 8D
# FINAL DATASET VALIDATION AND NOTEBOOK 49 HANDOFF
# ============================================================

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


print("=" * 100)
print("SECTION 8D — FINAL DATASET VALIDATION AND NOTEBOOK 49 HANDOFF")
print("=" * 100)


# ------------------------------------------------------------
# 8D.1 — Output paths
# ------------------------------------------------------------

SECTION8D_REPORT_DIR = (
    SECTION8_REPORT_DIR
    / "final_validation"
)

SECTION8D_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION8D_ARTIFACT_INVENTORY_FILE = (
    SECTION8D_REPORT_DIR
    / "section8d_artifact_inventory.csv"
)

SECTION8D_VALIDATION_CHECKS_FILE = (
    SECTION8D_REPORT_DIR
    / "section8d_validation_checks.csv"
)

SECTION8D_FINAL_DATASET_PROFILE_FILE = (
    SECTION8D_REPORT_DIR
    / "section8d_final_dataset_profile.csv"
)

SECTION8D_HANDOFF_MANIFEST_FILE = (
    SECTION8D_REPORT_DIR
    / "section8d_handoff_manifest.json"
)

SECTION8D_FINAL_SUMMARY_FILE = (
    SECTION8D_REPORT_DIR
    / "section8d_final_summary.json"
)


# ------------------------------------------------------------
# 8D.2 — Required artifact inventory
# ------------------------------------------------------------

required_artifacts = {
    # Section 7
    "section7a_prioritized_scenario_queue":
        SECTION7_QUEUE_FILE,

    "section7a_progress_ledger":
        SECTION7_PROGRESS_FILE,

    "section7a_runtime_contract":
        SECTION7_RUNTIME_CONTRACT_FILE,

    "section7b_factory_report":
        SECTION7B_FACTORY_REPORT_FILE,

    "section7b_card_resolution":
        SECTION7B_CARD_RESOLUTION_FILE,

    "section7b_state_smoke_tests":
        SECTION7B_SMOKE_TEST_FILE,

    "section7c_variant_manifest":
        SECTION7C_VARIANT_MANIFEST_FILE,

    "section7c_scenario_summary":
        SECTION7C_SCENARIO_SUMMARY_FILE,

    "section7c_variant_failures":
        SECTION7C_FAILURE_FILE,

    "section7c_variant_summary":
        SECTION7C_VARIANT_SUMMARY_FILE,

    # Section 8A
    "section8a_search_binding":
        SECTION8A_BINDING_FILE,

    "section8a_search_dry_run":
        SECTION8A_DRY_RUN_FILE,

    # Section 8B
    "section8b_search_labels":
        SECTION8B_RESULTS_FILE,

    "section8b_search_failures":
        SECTION8B_FAILURES_FILE,

    "section8b_search_progress":
        SECTION8B_PROGRESS_FILE,

    "section8b_search_summary":
        SECTION8B_SUMMARY_FILE,

    # Section 8C
    "section8c_policy_dataset_csv":
        SECTION8C_POLICY_DATASET_FILE,

    "section8c_policy_dataset_jsonl":
        SECTION8C_POLICY_DATASET_JSONL_FILE,

    "section8c_rejected_records":
        SECTION8C_REJECTED_RECORDS_FILE,

    "section8c_dataset_statistics":
        SECTION8C_DATASET_STATISTICS_FILE,

    "section8c_side_summary":
        SECTION8C_SIDE_SUMMARY_FILE,

    "section8c_variant_summary":
        SECTION8C_VARIANT_SUMMARY_FILE,

    "section8c_summary":
        SECTION8C_SUMMARY_FILE,
}


artifact_inventory_rows = []


for artifact_name, artifact_path in required_artifacts.items():

    artifact_path = Path(
        artifact_path
    )

    artifact_inventory_rows.append(
        {
            "artifact_name":
                artifact_name,

            "path":
                str(
                    artifact_path
                ),

            "exists":
                artifact_path.exists(),

            "is_file":
                artifact_path.is_file(),

            "size_bytes":
                (
                    int(
                        artifact_path.stat().st_size
                    )
                    if artifact_path.exists()
                    and artifact_path.is_file()
                    else 0
                ),

            "nonempty":
                (
                    artifact_path.exists()
                    and artifact_path.is_file()
                    and artifact_path.stat().st_size > 0
                ),
        }
    )


artifact_inventory_df = pd.DataFrame(
    artifact_inventory_rows
)

artifact_inventory_df.to_csv(
    SECTION8D_ARTIFACT_INVENTORY_FILE,
    index=False,
)


print()
print("ARTIFACT INVENTORY")
print("-" * 100)

display(
    artifact_inventory_df
)


missing_artifacts = (
    artifact_inventory_df.loc[
        ~artifact_inventory_df[
            "exists"
        ],
        "artifact_name",
    ]
    .tolist()
)

# Some artifacts are allowed to be empty because they represent
# zero failures or zero rejected records.

allowed_empty_artifacts = {
    "section7c_variant_failures",
    "section8b_search_failures",
    "section8c_rejected_records",
}

empty_artifacts = (
    artifact_inventory_df.loc[
        artifact_inventory_df["exists"]
        & ~artifact_inventory_df["nonempty"]
        & ~artifact_inventory_df["artifact_name"].isin(
            allowed_empty_artifacts
        ),
        "artifact_name",
    ]
    .tolist()
)


# ------------------------------------------------------------
# 8D.3 — Load final dataset and summaries
# ------------------------------------------------------------

final_policy_df = pd.read_csv(
    SECTION8C_POLICY_DATASET_FILE,
    low_memory=False,
)

search_labels_df = pd.read_csv(
    SECTION8B_RESULTS_FILE,
    low_memory=False,
)

variant_manifest_df = pd.read_csv(
    SECTION7C_VARIANT_MANIFEST_FILE,
    low_memory=False,
)

# Safely load rejected records.
# If there were no rejected records, the CSV may legitimately be empty.

if (
    SECTION8C_REJECTED_RECORDS_FILE.exists()
    and SECTION8C_REJECTED_RECORDS_FILE.stat().st_size > 0
):
    try:
        rejected_records_df = pd.read_csv(
            SECTION8C_REJECTED_RECORDS_FILE,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        rejected_records_df = pd.DataFrame()
else:
    rejected_records_df = pd.DataFrame()

print(f"Rejected records loaded: {len(rejected_records_df)}")


with open(
    SECTION8C_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    section8c_summary = json.load(
        file
    )


with open(
    SECTION8B_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    section8b_summary = json.load(
        file
    )


with open(
    SECTION7C_VARIANT_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    section7c_summary = json.load(
        file
    )


assert not final_policy_df.empty, (
    "The final expert-policy dataset is empty."
)


# ------------------------------------------------------------
# 8D.4 — Required final dataset schema
# ------------------------------------------------------------

required_final_columns = [
    "example_id",
    "variant_id",
    "source_scenario_id",
    "variant_name",
    "side_mode",
    "source_side",
    "current_side",
    "player_card",
    "opponent_card",
    "state_key",
    "legal_move_count",
    "legal_moves",
    "expert_move",
    "expert_move_key",
    "value_target",
    "search_depth_requested",
    "search_depth_reached",
    "search_success",
    "sample_weight",
]


missing_final_columns = [
    column
    for column in required_final_columns
    if column not in final_policy_df.columns
]


# ------------------------------------------------------------
# 8D.5 — Core dataset integrity metrics
# ------------------------------------------------------------

final_example_count = int(
    len(
        final_policy_df
    )
)

unique_variant_count = int(
    final_policy_df[
        "variant_id"
    ].nunique()
)

duplicate_variant_count = int(
    final_policy_df[
        "variant_id"
    ].duplicated().sum()
)

unique_example_count = int(
    final_policy_df[
        "example_id"
    ].nunique()
)

duplicate_example_count = int(
    final_policy_df[
        "example_id"
    ].duplicated().sum()
)

source_scenario_count = int(
    final_policy_df[
        "source_scenario_id"
    ].nunique()
)

unique_matchup_count = int(
    final_policy_df[
        [
            "player_card",
            "opponent_card",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ------------------------------------------------------------
# 8D.6 — Required field completeness
# ------------------------------------------------------------

required_nonempty_columns = [
    "variant_id",
    "example_id",
    "source_scenario_id",
    "variant_name",
    "current_side",
    "player_card",
    "opponent_card",
    "state_key",
    "legal_moves",
    "expert_move",
    "expert_move_key",
]


required_numeric_columns = [
    "legal_move_count",
    "value_target",
    "search_depth_requested",
    "search_depth_reached",
    "sample_weight",
]


missing_field_counts = {}


for column in required_nonempty_columns:

    missing_field_counts[
        column
    ] = int(
        final_policy_df[
            column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


for column in required_numeric_columns:

    missing_field_counts[
        column
    ] = int(
        pd.to_numeric(
            final_policy_df[
                column
            ],
            errors="coerce",
        )
        .isna()
        .sum()
    )


# ------------------------------------------------------------
# 8D.7 — Search-label integrity
# ------------------------------------------------------------

invalid_legal_move_count = int(
    pd.to_numeric(
        final_policy_df[
            "legal_move_count"
        ],
        errors="coerce",
    )
    .le(0)
    .sum()
)

invalid_search_success_count = int(
    (
        ~final_policy_df[
            "search_success"
        ]
        .astype(bool)
    )
    .sum()
)

invalid_depth_count = int(
    (
        pd.to_numeric(
            final_policy_df[
                "search_depth_reached"
            ],
            errors="coerce",
        )
        .fillna(0)
        .le(0)
    )
    .sum()
)

invalid_weight_count = int(
    (
        pd.to_numeric(
            final_policy_df[
                "sample_weight"
            ],
            errors="coerce",
        )
        .fillna(0)
        .le(0)
    )
    .sum()
)

invalid_value_target_count = int(
    (
        ~np.isfinite(
            pd.to_numeric(
                final_policy_df[
                    "value_target"
                ],
                errors="coerce",
            )
        )
    )
    .sum()
)


# ------------------------------------------------------------
# 8D.8 — Cross-file consistency
# ------------------------------------------------------------

search_label_variant_ids = set(
    search_labels_df[
        "variant_id"
    ].astype(str)
)

final_policy_variant_ids = set(
    final_policy_df[
        "variant_id"
    ].astype(str)
)

variant_manifest_ids = set(
    variant_manifest_df[
        "variant_id"
    ].astype(str)
)


final_not_in_search = sorted(
    final_policy_variant_ids
    - search_label_variant_ids
)

final_not_in_manifest = sorted(
    final_policy_variant_ids
    - variant_manifest_ids
)

search_not_in_final = sorted(
    search_label_variant_ids
    - final_policy_variant_ids
)


# ------------------------------------------------------------
# 8D.9 — Side and curriculum validation
# ------------------------------------------------------------

current_side_normalized = (
    final_policy_df[
        "current_side"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

side_mode_normalized = (
    final_policy_df[
        "side_mode"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


player_side_count = int(
    current_side_normalized.eq(
        "player"
    ).sum()
)

opponent_side_count = int(
    current_side_normalized.eq(
        "opponent"
    ).sum()
)

unknown_side_count = int(
    (
        ~current_side_normalized.isin(
            [
                "player",
                "opponent",
            ]
        )
    )
    .sum()
)

preserved_side_count = int(
    side_mode_normalized.eq(
        "PRESERVE"
    ).sum()
)

counterfactual_side_count = int(
    side_mode_normalized.eq(
        "FLIP"
    ).sum()
)

unknown_side_mode_count = int(
    (
        ~side_mode_normalized.isin(
            [
                "PRESERVE",
                "FLIP",
            ]
        )
    )
    .sum()
)


opponent_side_fraction = (
    opponent_side_count
    / final_example_count
    if final_example_count > 0
    else 0.0
)

counterfactual_fraction = (
    counterfactual_side_count
    / final_example_count
    if final_example_count > 0
    else 0.0
)


# ------------------------------------------------------------
# 8D.10 — Rejection and failure review
# ------------------------------------------------------------

rejected_record_count = int(
    len(
        rejected_records_df
    )
)

batch_failure_count = int(
    section8b_summary.get(
        "failed_variants",
        0,
    )
)

batch_pending_count = int(
    section8b_summary.get(
        "pending_variants",
        0,
    )
)

variant_failure_count = int(
    section7c_summary.get(
        "failed_variants",
        0,
    )
)


# ------------------------------------------------------------
# 8D.11 — Validation checks
# ------------------------------------------------------------

validation_checks = [
    {
        "check":
            "all_required_artifacts_exist",

        "passed":
            len(
                missing_artifacts
            ) == 0,

        "value":
            len(
                missing_artifacts
            ),

        "expected":
            0,

        "details":
            json.dumps(
                missing_artifacts
            ),
    },
    {
        "check":
            "all_required_artifacts_nonempty",

        "passed":
            len(
                empty_artifacts
            ) == 0,

        "value":
            len(
                empty_artifacts
            ),

        "expected":
            0,

        "details":
            json.dumps(
                empty_artifacts
            ),
    },
    {
        "check":
            "final_dataset_nonempty",

        "passed":
            final_example_count > 0,

        "value":
            final_example_count,

        "expected":
            "> 0",

        "details":
            "",
    },
    {
        "check":
            "required_schema_complete",

        "passed":
            len(
                missing_final_columns
            ) == 0,

        "value":
            len(
                missing_final_columns
            ),

        "expected":
            0,

        "details":
            json.dumps(
                missing_final_columns
            ),
    },
    {
        "check":
            "variant_ids_unique",

        "passed":
            duplicate_variant_count == 0,

        "value":
            duplicate_variant_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "example_ids_unique",

        "passed":
            duplicate_example_count == 0,

        "value":
            duplicate_example_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "required_fields_complete",

        "passed":
            all(
                value == 0
                for value in missing_field_counts.values()
            ),

        "value":
            int(
                sum(
                    missing_field_counts.values()
                )
            ),

        "expected":
            0,

        "details":
            json.dumps(
                missing_field_counts
            ),
    },
    {
        "check":
            "all_rows_have_legal_moves",

        "passed":
            invalid_legal_move_count == 0,

        "value":
            invalid_legal_move_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "all_searches_successful",

        "passed":
            invalid_search_success_count == 0,

        "value":
            invalid_search_success_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "search_depth_valid",

        "passed":
            invalid_depth_count == 0,

        "value":
            invalid_depth_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "sample_weights_positive",

        "passed":
            invalid_weight_count == 0,

        "value":
            invalid_weight_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "value_targets_finite",

        "passed":
            invalid_value_target_count == 0,

        "value":
            invalid_value_target_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "final_rows_present_in_search_labels",

        "passed":
            len(
                final_not_in_search
            ) == 0,

        "value":
            len(
                final_not_in_search
            ),

        "expected":
            0,

        "details":
            json.dumps(
                final_not_in_search[:20]
            ),
    },
    {
        "check":
            "final_rows_present_in_variant_manifest",

        "passed":
            len(
                final_not_in_manifest
            ) == 0,

        "value":
            len(
                final_not_in_manifest
            ),

        "expected":
            0,

        "details":
            json.dumps(
                final_not_in_manifest[:20]
            ),
    },
    {
        "check":
            "all_search_labels_exported",

        "passed":
            len(
                search_not_in_final
            ) == rejected_record_count,

        "value":
            len(
                search_not_in_final
            ),

        "expected":
            rejected_record_count,

        "details":
            (
                "Search rows not in final should equal "
                "the number of rejected Section 8C records."
            ),
    },
    {
        "check":
            "side_values_valid",

        "passed":
            unknown_side_count == 0,

        "value":
            unknown_side_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "side_mode_values_valid",

        "passed":
            unknown_side_mode_count == 0,

        "value":
            unknown_side_mode_count,

        "expected":
            0,

        "details":
            "",
    },
    {
        "check":
            "both_sides_represented",

        "passed":
            (
                player_side_count > 0
                and opponent_side_count > 0
            ),

        "value":
            (
                f"Player={player_side_count}, "
                f"Opponent={opponent_side_count}"
            ),

        "expected":
            "Both > 0",

        "details":
            "",
    },
    {
        "check":
            "counterfactual_variants_present",

        "passed":
            counterfactual_side_count > 0,

        "value":
            counterfactual_side_count,

        "expected":
            "> 0",

        "details":
            "",
    },
    {
        "check":
            "source_scenario_coverage",

        "passed":
            source_scenario_count == int(
                section7c_summary.get(
                    "source_scenarios",
                    source_scenario_count,
                )
            ),

        "value":
            source_scenario_count,

        "expected":
            int(
                section7c_summary.get(
                    "source_scenarios",
                    source_scenario_count,
                )
            ),

        "details":
            "",
    },
    {
        "check":
            "no_batch_search_pending",

        "passed":
            batch_pending_count == 0,

        "value":
            batch_pending_count,

        "expected":
            0,

        "details":
            "",
    },
]


validation_checks_df = pd.DataFrame(
    validation_checks
)

validation_checks_df.to_csv(
    SECTION8D_VALIDATION_CHECKS_FILE,
    index=False,
)


passed_check_count = int(
    validation_checks_df[
        "passed"
    ].sum()
)

failed_check_count = int(
    (
        ~validation_checks_df[
            "passed"
        ]
    ).sum()
)


# ------------------------------------------------------------
# 8D.12 — Final dataset profile
# ------------------------------------------------------------

final_dataset_profile_rows = [
    {
        "metric":
            "final_examples",

        "value":
            final_example_count,
    },
    {
        "metric":
            "unique_variants",

        "value":
            unique_variant_count,
    },
    {
        "metric":
            "unique_examples",

        "value":
            unique_example_count,
    },
    {
        "metric":
            "source_scenarios",

        "value":
            source_scenario_count,
    },
    {
        "metric":
            "unique_matchups",

        "value":
            unique_matchup_count,
    },
    {
        "metric":
            "player_side_examples",

        "value":
            player_side_count,
    },
    {
        "metric":
            "opponent_side_examples",

        "value":
            opponent_side_count,
    },
    {
        "metric":
            "opponent_side_fraction",

        "value":
            opponent_side_fraction,
    },
    {
        "metric":
            "preserved_side_examples",

        "value":
            preserved_side_count,
    },
    {
        "metric":
            "counterfactual_side_examples",

        "value":
            counterfactual_side_count,
    },
    {
        "metric":
            "counterfactual_fraction",

        "value":
            counterfactual_fraction,
    },
    {
        "metric":
            "average_legal_moves",

        "value":
            float(
                pd.to_numeric(
                    final_policy_df[
                        "legal_move_count"
                    ],
                    errors="coerce",
                ).mean()
            ),
    },
    {
        "metric":
            "average_value_target",

        "value":
            float(
                pd.to_numeric(
                    final_policy_df[
                        "value_target"
                    ],
                    errors="coerce",
                ).mean()
            ),
    },
    {
        "metric":
            "average_search_depth_reached",

        "value":
            float(
                pd.to_numeric(
                    final_policy_df[
                        "search_depth_reached"
                    ],
                    errors="coerce",
                ).mean()
            ),
    },
    {
        "metric":
            "average_sample_weight",

        "value":
            float(
                pd.to_numeric(
                    final_policy_df[
                        "sample_weight"
                    ],
                    errors="coerce",
                ).mean()
            ),
    },
    {
        "metric":
            "rejected_records",

        "value":
            rejected_record_count,
    },
    {
        "metric":
            "section7c_variant_failures",

        "value":
            variant_failure_count,
    },
    {
        "metric":
            "section8b_search_failures",

        "value":
            batch_failure_count,
    },
    {
        "metric":
            "section8b_pending",

        "value":
            batch_pending_count,
    },
    {
        "metric":
            "validation_checks_passed",

        "value":
            passed_check_count,
    },
    {
        "metric":
            "validation_checks_failed",

        "value":
            failed_check_count,
    },
]


final_dataset_profile_df = pd.DataFrame(
    final_dataset_profile_rows
)

final_dataset_profile_df.to_csv(
    SECTION8D_FINAL_DATASET_PROFILE_FILE,
    index=False,
)


# ------------------------------------------------------------
# 8D.13 — Handoff manifest
# ------------------------------------------------------------

handoff_ready = (
    failed_check_count == 0
)


handoff_manifest = {
    "notebook":
        "49_side_balanced_search_guided_self_play",

    "status":
        (
            "READY_FOR_POLICY_TRAINING"
            if handoff_ready
            else "VALIDATION_FAILED"
        ),

    "completed_sections":
        [
            "7A",
            "7B",
            "7C",
            "8A",
            "8B",
            "8C",
            "8D",
        ],

    "final_dataset_rows":
        final_example_count,

    "source_scenarios":
        source_scenario_count,

    "unique_matchups":
        unique_matchup_count,

    "player_side_examples":
        player_side_count,

    "opponent_side_examples":
        opponent_side_count,

    "opponent_side_fraction":
        opponent_side_fraction,

    "preserved_side_examples":
        preserved_side_count,

    "counterfactual_side_examples":
        counterfactual_side_count,

    "rejected_records":
        rejected_record_count,

    "variant_failures":
        variant_failure_count,

    "search_failures":
        batch_failure_count,

    "search_pending":
        batch_pending_count,

    "validation_checks_passed":
        passed_check_count,

    "validation_checks_failed":
        failed_check_count,

    "failed_checks":
        validation_checks_df.loc[
            ~validation_checks_df[
                "passed"
            ],
            "check",
        ].tolist(),

    "training_dataset_csv":
        str(
            SECTION8C_POLICY_DATASET_FILE
        ),

    "training_dataset_jsonl":
        str(
            SECTION8C_POLICY_DATASET_JSONL_FILE
        ),

    "dataset_statistics":
        str(
            SECTION8C_DATASET_STATISTICS_FILE
        ),

    "side_summary":
        str(
            SECTION8C_SIDE_SUMMARY_FILE
        ),

    "variant_summary":
        str(
            SECTION8C_VARIANT_SUMMARY_FILE
        ),

    "artifact_inventory":
        str(
            SECTION8D_ARTIFACT_INVENTORY_FILE
        ),

    "validation_checks":
        str(
            SECTION8D_VALIDATION_CHECKS_FILE
        ),

    "dataset_profile":
        str(
            SECTION8D_FINAL_DATASET_PROFILE_FILE
        ),

    "next_notebook":
        "Notebook 50 — Side-Balanced Policy Fine-Tuning",

    "next_stage":
        (
            "POLICY_FINE_TUNING"
            if handoff_ready
            else "REPAIR_FINAL_VALIDATION"
        ),
}


with open(
    SECTION8D_HANDOFF_MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        handoff_manifest,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8D.14 — Final summary
# ------------------------------------------------------------

final_summary = {
    "status":
        handoff_manifest[
            "status"
        ],

    "notebook_complete":
        handoff_ready,

    "required_artifacts":
        int(
            len(
                artifact_inventory_df
            )
        ),

    "missing_artifacts":
        missing_artifacts,

    "empty_artifacts":
        empty_artifacts,

    "validation_checks":
        int(
            len(
                validation_checks_df
            )
        ),

    "validation_checks_passed":
        passed_check_count,

    "validation_checks_failed":
        failed_check_count,

    "final_dataset_rows":
        final_example_count,

    "source_scenarios":
        source_scenario_count,

    "player_side_examples":
        player_side_count,

    "opponent_side_examples":
        opponent_side_count,

    "counterfactual_examples":
        counterfactual_side_count,

    "rejected_records":
        rejected_record_count,

    "next_stage":
        handoff_manifest[
            "next_stage"
        ],
}


with open(
    SECTION8D_FINAL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 8D.15 — Display validation results
# ------------------------------------------------------------

print()
print("FINAL VALIDATION CHECKS")
print("-" * 100)

display(
    validation_checks_df
)


print()
print("FINAL DATASET PROFILE")
print("-" * 100)

display(
    final_dataset_profile_df
)


print()
print("NOTEBOOK 49 HANDOFF SUMMARY")
print("-" * 100)

print(
    "Final status               :",
    handoff_manifest[
        "status"
    ],
)

print(
    "Final dataset rows         :",
    final_example_count,
)

print(
    "Source scenarios           :",
    source_scenario_count,
)

print(
    "Unique matchups            :",
    unique_matchup_count,
)

print(
    "Player-side examples       :",
    player_side_count,
)

print(
    "Opponent-side examples     :",
    opponent_side_count,
)

print(
    "Opponent-side fraction     :",
    f"{opponent_side_fraction:.2%}",
)

print(
    "Counterfactual examples    :",
    counterfactual_side_count,
)

print(
    "Rejected records           :",
    rejected_record_count,
)

print(
    "Validation checks passed   :",
    (
        f"{passed_check_count}/"
        f"{len(validation_checks_df)}"
    ),
)

print(
    "Validation checks failed   :",
    failed_check_count,
)

print(
    "Next notebook              :",
    handoff_manifest[
        "next_notebook"
    ],
)

print(
    "Next stage                 :",
    handoff_manifest[
        "next_stage"
    ],
)


if failed_check_count > 0:

    print()
    print("FAILED VALIDATION CHECKS")
    print("-" * 100)

    display(
        validation_checks_df[
            ~validation_checks_df[
                "passed"
            ]
        ]
    )


# ------------------------------------------------------------
# 8D.16 — Final assertions
# ------------------------------------------------------------

assert SECTION8D_ARTIFACT_INVENTORY_FILE.exists()

assert SECTION8D_VALIDATION_CHECKS_FILE.exists()

assert SECTION8D_FINAL_DATASET_PROFILE_FILE.exists()

assert SECTION8D_HANDOFF_MANIFEST_FILE.exists()

assert SECTION8D_FINAL_SUMMARY_FILE.exists()

assert len(
    missing_artifacts
) == 0, (
    "Required Notebook 49 artifacts are missing: "
    f"{missing_artifacts}"
)

assert len(
    empty_artifacts
) == 0, (
    "Required Notebook 49 artifacts are empty: "
    f"{empty_artifacts}"
)

assert len(
    missing_final_columns
) == 0, (
    "The final dataset is missing required columns: "
    f"{missing_final_columns}"
)

assert failed_check_count == 0, (
    "Notebook 49 final validation failed. "
    f"Failed checks: "
    f"{handoff_manifest['failed_checks']}"
)

assert handoff_manifest[
    "status"
] == "READY_FOR_POLICY_TRAINING"

assert handoff_manifest[
    "next_stage"
] == "POLICY_FINE_TUNING"


print()
print("SAVED SECTION 8D REPORTS")
print("-" * 100)

print(
    SECTION8D_ARTIFACT_INVENTORY_FILE
)

print(
    SECTION8D_VALIDATION_CHECKS_FILE
)

print(
    SECTION8D_FINAL_DATASET_PROFILE_FILE
)

print(
    SECTION8D_HANDOFF_MANIFEST_FILE
)

print(
    SECTION8D_FINAL_SUMMARY_FILE
)

print()
print(
    "✅ SECTION 8D FINAL DATASET VALIDATION PASSED"
)

print()
print(
    "🎉 NOTEBOOK 49 COMPLETE — READY FOR NOTEBOOK 50"
)


# In[ ]:


# =============================================================================
# NOTEBOOK 49
# LEGAL-MOVE GENERATION SOURCE DIAGNOSTIC
# =============================================================================

import inspect
import json
import re


print("=" * 100)
print("NOTEBOOK 49 — LEGAL-MOVE GENERATION SOURCE DIAGNOSTIC")
print("=" * 100)


# =============================================================================
# SEARCH CURRENT NOTEBOOK NAMESPACE FOR RELEVANT FUNCTIONS
# =============================================================================

search_keywords = [
    "legal",
    "move",
    "attack",
    "action",
    "expert",
    "search",
    "card",
]

candidate_function_records = []

for object_name, object_value in list(globals().items()):
    if not callable(object_value):
        continue

    normalized_name = str(object_name).lower()

    matched_keywords = [
        keyword
        for keyword in search_keywords
        if keyword in normalized_name
    ]

    if not matched_keywords:
        continue

    try:
        source_code = inspect.getsource(
            object_value
        )
    except (
        OSError,
        TypeError,
    ):
        source_code = None

    candidate_function_records.append(
        {
            "object_name": object_name,
            "object_type": type(
                object_value
            ).__name__,
            "matched_keywords": "|".join(
                matched_keywords
            ),
            "has_source_code": bool(
                source_code
            ),
            "source_preview": (
                source_code[:1000]
                if source_code
                else None
            ),
        }
    )

candidate_function_df = (
    pd.DataFrame(
        candidate_function_records
    )
    .sort_values(
        [
            "has_source_code",
            "object_name",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

print()
print("CANDIDATE LEGAL-MOVE / SEARCH FUNCTIONS")
print("-" * 100)

display(
    candidate_function_df
)


# =============================================================================
# PRINT FULL SOURCE FOR MOST RELEVANT FUNCTIONS
# =============================================================================

priority_function_names = [
    object_name
    for object_name in candidate_function_df[
        "object_name"
    ].tolist()
    if any(
        keyword in object_name.lower()
        for keyword in [
            "legal_move",
            "legal_action",
            "generate_move",
            "generate_action",
            "available_move",
            "available_action",
            "attack",
            "expert_move",
            "search_best",
            "choose_move",
        ]
    )
]

print()
print("PRIORITY FUNCTION SOURCE")
print("-" * 100)

if not priority_function_names:
    print(
        "No obvious priority functions were found."
    )

for function_name in priority_function_names:
    function_object = globals().get(
        function_name
    )

    print()
    print("=" * 100)
    print(f"FUNCTION: {function_name}")
    print("=" * 100)

    try:
        print(
            inspect.getsource(
                function_object
            )
        )
    except (
        OSError,
        TypeError,
    ) as exc:
        print(
            f"Could not retrieve source: {exc}"
        )


# =============================================================================
# SEARCH DATAFRAME COLUMNS FOR CARD METADATA
# =============================================================================

dataframe_records = []

for object_name, object_value in list(globals().items()):
    if not isinstance(
        object_value,
        pd.DataFrame,
    ):
        continue

    related_columns = [
        column_name
        for column_name in object_value.columns
        if any(
            keyword in str(
                column_name
            ).lower()
            for keyword in [
                "attack",
                "move",
                "energy",
                "cost",
                "damage",
                "card",
            ]
        )
    ]

    if not related_columns:
        continue

    dataframe_records.append(
        {
            "dataframe_name": object_name,
            "row_count": int(
                len(object_value)
            ),
            "column_count": int(
                len(object_value.columns)
            ),
            "related_columns": json.dumps(
                related_columns
            ),
        }
    )

related_dataframe_df = pd.DataFrame(
    dataframe_records
)

print()
print("DATAFRAMES WITH CARD / ATTACK METADATA")
print("-" * 100)

display(
    related_dataframe_df
)


# =============================================================================
# PREVIEW RELATED DATAFRAMES
# =============================================================================

for dataframe_name in (
    related_dataframe_df[
        "dataframe_name"
    ].tolist()
    if not related_dataframe_df.empty
    else []
):
    dataframe_object = globals().get(
        dataframe_name
    )

    related_columns = [
        column_name
        for column_name
        in dataframe_object.columns
        if any(
            keyword in str(
                column_name
            ).lower()
            for keyword in [
                "attack",
                "move",
                "energy",
                "cost",
                "damage",
                "card",
            ]
        )
    ]

    print()
    print("=" * 100)
    print(
        f"DATAFRAME PREVIEW: {dataframe_name}"
    )
    print("=" * 100)

    display(
        dataframe_object[
            related_columns
        ].head(20)
    )


# =============================================================================
# CHECK POLICY DATASET LEGAL-MOVE OUTPUT
# =============================================================================

possible_policy_dataframes = []

for object_name, object_value in list(globals().items()):
    if not isinstance(
        object_value,
        pd.DataFrame,
    ):
        continue

    required_columns = {
        "legal_moves",
        "legal_move_count",
        "expert_move",
    }

    if required_columns.issubset(
        object_value.columns
    ):
        possible_policy_dataframes.append(
            object_name
        )

print()
print("POLICY DATAFRAME CANDIDATES")
print("-" * 100)

print(
    possible_policy_dataframes
)

for dataframe_name in possible_policy_dataframes:
    dataframe_object = globals().get(
        dataframe_name
    )

    preview_columns = [
        column_name
        for column_name in [
            "variant_id",
            "source_scenario_id",
            "current_side",
            "player_card",
            "opponent_card",
            "player_energy",
            "opponent_energy",
            "legal_moves",
            "legal_move_count",
            "expert_move",
            "expert_move_key",
        ]
        if column_name
        in dataframe_object.columns
    ]

    print()
    print("=" * 100)
    print(
        f"POLICY DATASET PREVIEW: {dataframe_name}"
    )
    print("=" * 100)

    display(
        dataframe_object[
            preview_columns
        ].head(30)
    )


print()
print(
    "✅ NOTEBOOK 49 LEGAL-MOVE SOURCE DIAGNOSTIC COMPLETED"
)


# In[ ]:




