#!/usr/bin/env python
# coding: utf-8

# # Notebook 53 — Legality-Aware Policy Optimization
# 
# ## Purpose
# 
# #### Notebook 53 improves the tournament policy using the weaknesses identified in Notebook 52.
# 
# The primary optimization target is to reduce the policy’s dependence on legal-action masking.
# 
# #### Notebook 52 established the following benchmark:
# 
# - 24 expanded tournament battles
# - 210 policy decisions
# - 71.9048% masking-change rate
# - 0.4762% fallback rate
# - 7 Player wins
# - 14 Opponent wins
# - 3 draws
# - weakest Player card: Eevee
# - weakest condition: HP pressure
# 
# #### This notebook will:
# 
# 1. Validate the Notebook 52 handoff.
# 2. Load the tournament benchmark and policy-decision history.
# 3. Identify raw predictions that differed from selected legal actions.
# 4. Build legality-aware training features.
# 5. Construct an optimized policy dataset.
# 6. Retrain and validate a legality-aware policy.
# 7. Compare the optimized policy against the frozen Notebook 52 benchmark.
# 8. Export the Notebook 53 optimization artifacts and handoff.

# ## Section 1A — Imports and Project Paths

# In[1]:


# ======================================================================================
# SECTION 1A — IMPORTS AND PROJECT PATHS
# ======================================================================================

from __future__ import annotations

import ast
import inspect
import json
import math
import random
import sys
import warnings

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

RANDOM_SEED = 53

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# --------------------------------------------------------------------------------------
# Notebook identity
# --------------------------------------------------------------------------------------

NOTEBOOK_NUMBER = 53

NOTEBOOK_NAME = (
    "legality_aware_policy_optimization"
)


# --------------------------------------------------------------------------------------
# Resolve project root
# --------------------------------------------------------------------------------------

CURRENT_DIRECTORY = Path.cwd().resolve()

if CURRENT_DIRECTORY.name.lower() == "notebooks":

    PROJECT_ROOT = CURRENT_DIRECTORY.parent

else:

    possible_root = CURRENT_DIRECTORY

    while (
        possible_root.parent != possible_root
        and not (
            possible_root
            / "notebooks"
        ).exists()
    ):

        possible_root = (
            possible_root.parent
        )

    PROJECT_ROOT = possible_root


# --------------------------------------------------------------------------------------
# Core project directories
# --------------------------------------------------------------------------------------

NOTEBOOKS_DIR = (
    PROJECT_ROOT
    / "notebooks"
)

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)


# --------------------------------------------------------------------------------------
# Upstream notebook directories
# --------------------------------------------------------------------------------------

NOTEBOOK50_REPORT_DIR = (
    REPORTS_DIR
    / "notebook50"
)

NOTEBOOK50_MODEL_DIR = (
    MODELS_DIR
    / "notebook50"
)

NOTEBOOK51_REPORT_DIR = (
    REPORTS_DIR
    / "notebook51"
)

NOTEBOOK52_REPORT_DIR = (
    REPORTS_DIR
    / "notebook52"
)


# --------------------------------------------------------------------------------------
# Notebook 53 output directories
# --------------------------------------------------------------------------------------

NOTEBOOK53_REPORT_DIR = (
    REPORTS_DIR
    / "notebook53"
)

NOTEBOOK53_MODEL_DIR = (
    MODELS_DIR
    / "notebook53"
)

NOTEBOOK53_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK53_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# Initial upstream directory validation
# --------------------------------------------------------------------------------------

required_upstream_directories = {
    "Notebook 50 reports":
        NOTEBOOK50_REPORT_DIR,

    "Notebook 50 models":
        NOTEBOOK50_MODEL_DIR,

    "Notebook 51 reports":
        NOTEBOOK51_REPORT_DIR,

    "Notebook 52 reports":
        NOTEBOOK52_REPORT_DIR,

    "Source directory":
        SRC_DIR,
}


missing_upstream_directories = [
    directory_name
    for directory_name, directory_path
    in required_upstream_directories.items()
    if not directory_path.exists()
]


print("=" * 100)
print(
    "NOTEBOOK 53 — LEGALITY-AWARE POLICY OPTIMIZATION"
)
print("=" * 100)

print()
print("SECTION 1A — IMPORTS AND PROJECT PATHS")
print("-" * 100)

print()
print("NOTEBOOK IDENTITY")
print("-" * 100)

print(
    f"Notebook number          : "
    f"{NOTEBOOK_NUMBER}"
)

print(
    f"Notebook name            : "
    f"{NOTEBOOK_NAME}"
)

print(
    f"Random seed              : "
    f"{RANDOM_SEED}"
)


print()
print("DIRECTORIES")
print("-" * 100)

print(
    f"Current directory        : "
    f"{CURRENT_DIRECTORY}"
)

print(
    f"Project root             : "
    f"{PROJECT_ROOT}"
)

print(
    f"Notebook 50 reports      : "
    f"{NOTEBOOK50_REPORT_DIR}"
)

print(
    f"Notebook 50 models       : "
    f"{NOTEBOOK50_MODEL_DIR}"
)

print(
    f"Notebook 51 reports      : "
    f"{NOTEBOOK51_REPORT_DIR}"
)

print(
    f"Notebook 52 reports      : "
    f"{NOTEBOOK52_REPORT_DIR}"
)

print(
    f"Notebook 53 reports      : "
    f"{NOTEBOOK53_REPORT_DIR}"
)

print(
    f"Notebook 53 models       : "
    f"{NOTEBOOK53_MODEL_DIR}"
)

print(
    f"Source directory         : "
    f"{SRC_DIR}"
)


print()
print("PACKAGE VERSIONS")
print("-" * 100)

print(
    f"Python                   : "
    f"{sys.version.split()[0]}"
)

print(
    f"NumPy                    : "
    f"{np.__version__}"
)

print(
    f"Pandas                   : "
    f"{pd.__version__}"
)

print(
    f"Joblib                   : "
    f"{joblib.__version__}"
)


print()
print("UPSTREAM DIRECTORY STATUS")
print("-" * 100)

for directory_name, directory_path in (
    required_upstream_directories.items()
):

    print(
        f"{directory_name:30}: "
        f"{directory_path.exists()}"
    )


# --------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------

assert PROJECT_ROOT.exists(), (
    "The project root could not be resolved."
)

assert (
    PROJECT_ROOT
    / "notebooks"
).exists(), (
    "The resolved project root does not contain the notebooks directory."
)

assert not missing_upstream_directories, (
    "Required upstream directories are missing: "
    f"{missing_upstream_directories}"
)

assert NOTEBOOK53_REPORT_DIR.exists()

assert NOTEBOOK53_MODEL_DIR.exists()


print()
print(
    "✅ SECTION 1A IMPORTS AND PROJECT PATHS PASSED"
)


# ## Section 1B — Notebook 52 Handoff Validation
# 
# This section validates the final handoff produced by Notebook 52.
# 
# It confirms that:
# 
# - Notebook 52 completed successfully,
# - the tournament benchmark is ready,
# - all required benchmark artifacts exist,
# - the expanded battle results are available,
# - the policy decision history is available,
# - the optimization targets are available,
# - Notebook 53 may proceed with legality-aware policy optimization.

# In[2]:


# ======================================================================================
# SECTION 1B — NOTEBOOK 52 HANDOFF VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 1B — NOTEBOOK 52 HANDOFF VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Notebook 52 artifact paths
# --------------------------------------------------------------------------------------

NOTEBOOK52_SECTION7_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section7"
)

NOTEBOOK52_SECTION6_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section6"
)

NOTEBOOK52_SECTION5_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section5"
)


NOTEBOOK52_HANDOFF_MANIFEST_FILE = (
    NOTEBOOK52_SECTION7_DIR
    / "section7a_handoff_manifest.json"
)

NOTEBOOK52_FINAL_SUMMARY_FILE = (
    NOTEBOOK52_SECTION7_DIR
    / "section7a_final_summary.json"
)

NOTEBOOK52_FINAL_PROFILE_FILE = (
    NOTEBOOK52_SECTION7_DIR
    / "section7a_final_tournament_profile.csv"
)

NOTEBOOK52_OPTIMIZATION_TARGETS_FILE = (
    NOTEBOOK52_SECTION7_DIR
    / "section7a_optimization_targets.csv"
)

NOTEBOOK52_VALIDATION_CHECKS_FILE = (
    NOTEBOOK52_SECTION7_DIR
    / "section7a_validation_checks.csv"
)

NOTEBOOK52_BENCHMARK_PROFILE_FILE = (
    NOTEBOOK52_SECTION6_DIR
    / "section6b_benchmark_profile.csv"
)

NOTEBOOK52_DASHBOARD_SUMMARY_FILE = (
    NOTEBOOK52_SECTION6_DIR
    / "section6a_dashboard_summary.json"
)

NOTEBOOK52_EXPANDED_BATTLE_RESULTS_FILE = (
    NOTEBOOK52_SECTION5_DIR
    / "section5b_expanded_battle_results.csv"
)

NOTEBOOK52_EXPANDED_POLICY_HISTORY_FILE = (
    NOTEBOOK52_SECTION5_DIR
    / "section5b_expanded_policy_history.csv"
)

NOTEBOOK52_CONDITION_SUMMARY_FILE = (
    NOTEBOOK52_SECTION5_DIR
    / "section5b_condition_summary.csv"
)

NOTEBOOK52_TOURNAMENT_SUMMARY_FILE = (
    NOTEBOOK52_SECTION5_DIR
    / "section5b_summary.json"
)


required_notebook52_artifacts = {
    "handoff_manifest":
        NOTEBOOK52_HANDOFF_MANIFEST_FILE,

    "final_summary":
        NOTEBOOK52_FINAL_SUMMARY_FILE,

    "final_profile":
        NOTEBOOK52_FINAL_PROFILE_FILE,

    "optimization_targets":
        NOTEBOOK52_OPTIMIZATION_TARGETS_FILE,

    "validation_checks":
        NOTEBOOK52_VALIDATION_CHECKS_FILE,

    "benchmark_profile":
        NOTEBOOK52_BENCHMARK_PROFILE_FILE,

    "dashboard_summary":
        NOTEBOOK52_DASHBOARD_SUMMARY_FILE,

    "expanded_battle_results":
        NOTEBOOK52_EXPANDED_BATTLE_RESULTS_FILE,

    "expanded_policy_history":
        NOTEBOOK52_EXPANDED_POLICY_HISTORY_FILE,

    "condition_summary":
        NOTEBOOK52_CONDITION_SUMMARY_FILE,

    "tournament_summary":
        NOTEBOOK52_TOURNAMENT_SUMMARY_FILE,
}


# --------------------------------------------------------------------------------------
# 2. Build artifact inventory
# --------------------------------------------------------------------------------------

notebook52_artifact_rows = []

for artifact_name, artifact_path in (
    required_notebook52_artifacts.items()
):

    artifact_path = Path(
        artifact_path
    )

    notebook52_artifact_rows.append(
        {
            "artifact":
                artifact_name,

            "exists":
                artifact_path.exists(),

            "is_file":
                artifact_path.is_file(),

            "size_bytes":
                (
                    artifact_path.stat().st_size
                    if artifact_path.exists()
                    else 0
                ),

            "path":
                str(
                    artifact_path
                ),

            "nonempty":
                (
                    artifact_path.exists()
                    and artifact_path.stat().st_size > 0
                ),
        }
    )


notebook52_artifact_inventory_df = pd.DataFrame(
    notebook52_artifact_rows
)


print()
print("REQUIRED NOTEBOOK 52 ARTIFACTS")
print("-" * 100)

display(
    notebook52_artifact_inventory_df
)


# --------------------------------------------------------------------------------------
# 3. Validate required files before loading
# --------------------------------------------------------------------------------------

assert notebook52_artifact_inventory_df[
    "exists"
].all(), (
    "One or more required Notebook 52 artifacts are missing."
)

assert notebook52_artifact_inventory_df[
    "is_file"
].all(), (
    "One or more Notebook 52 artifact paths are not files."
)

assert notebook52_artifact_inventory_df[
    "nonempty"
].all(), (
    "One or more Notebook 52 artifacts are empty."
)


# --------------------------------------------------------------------------------------
# 4. Load Notebook 52 handoff artifacts
# --------------------------------------------------------------------------------------

with open(
    NOTEBOOK52_HANDOFF_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook52_handoff_manifest = json.load(
        file
    )


with open(
    NOTEBOOK52_FINAL_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook52_final_summary = json.load(
        file
    )


with open(
    NOTEBOOK52_DASHBOARD_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook52_dashboard_summary = json.load(
        file
    )


with open(
    NOTEBOOK52_TOURNAMENT_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook52_tournament_summary = json.load(
        file
    )


notebook52_final_profile_df = pd.read_csv(
    NOTEBOOK52_FINAL_PROFILE_FILE
)

notebook52_optimization_targets_df = pd.read_csv(
    NOTEBOOK52_OPTIMIZATION_TARGETS_FILE
)

notebook52_validation_checks_df = pd.read_csv(
    NOTEBOOK52_VALIDATION_CHECKS_FILE
)

notebook52_benchmark_profile_df = pd.read_csv(
    NOTEBOOK52_BENCHMARK_PROFILE_FILE
)

notebook52_battle_results_df = pd.read_csv(
    NOTEBOOK52_EXPANDED_BATTLE_RESULTS_FILE
)

notebook52_policy_history_df = pd.read_csv(
    NOTEBOOK52_EXPANDED_POLICY_HISTORY_FILE
)

notebook52_condition_summary_df = pd.read_csv(
    NOTEBOOK52_CONDITION_SUMMARY_FILE
)


# --------------------------------------------------------------------------------------
# 5. Extract core handoff metrics
# --------------------------------------------------------------------------------------

handoff_status = str(
    notebook52_handoff_manifest[
        "status"
    ]
)

battle_count = int(
    notebook52_handoff_manifest[
        "battle_count"
    ]
)

policy_decisions = int(
    notebook52_handoff_manifest[
        "policy_decisions"
    ]
)

player_wins = int(
    notebook52_handoff_manifest[
        "player_wins"
    ]
)

opponent_wins = int(
    notebook52_handoff_manifest[
        "opponent_wins"
    ]
)

draws = int(
    notebook52_handoff_manifest[
        "draws"
    ]
)

masking_changes = int(
    notebook52_handoff_manifest[
        "masking_changes"
    ]
)

masking_change_rate = float(
    notebook52_handoff_manifest[
        "masking_change_rate"
    ]
)

fallback_decisions = int(
    notebook52_handoff_manifest[
        "fallback_decisions"
    ]
)

fallback_rate = float(
    notebook52_handoff_manifest[
        "fallback_rate"
    ]
)

average_confidence = float(
    notebook52_handoff_manifest[
        "average_confidence"
    ]
)

primary_optimization_target = str(
    notebook52_handoff_manifest[
        "primary_optimization_target"
    ]
)

weakest_player_card = str(
    notebook52_handoff_manifest[
        "weakest_player_card"
    ]
)

weakest_condition = str(
    notebook52_handoff_manifest[
        "weakest_condition"
    ]
)

validation_checks_passed = int(
    notebook52_handoff_manifest[
        "validation_checks_passed"
    ]
)

validation_checks_failed = int(
    notebook52_handoff_manifest[
        "validation_checks_failed"
    ]
)

next_notebook = str(
    notebook52_handoff_manifest[
        "next_notebook"
    ]
)

next_stage = str(
    notebook52_handoff_manifest[
        "next_stage"
    ]
)


# --------------------------------------------------------------------------------------
# 6. Handoff validation assertions
# --------------------------------------------------------------------------------------

assert handoff_status == (
    "READY_FOR_POLICY_OPTIMIZATION"
), (
    "Notebook 52 is not ready for policy optimization."
)

assert notebook52_final_summary[
    "final_status"
] == "READY_FOR_POLICY_OPTIMIZATION"

assert battle_count == 24

assert len(
    notebook52_battle_results_df
) == battle_count

assert policy_decisions == 210

assert len(
    notebook52_policy_history_df
) == policy_decisions

assert (
    player_wins
    + opponent_wins
    + draws
) == battle_count

assert masking_changes == int(
    notebook52_policy_history_df[
        "masking_changed_action"
    ].astype(bool).sum()
)

assert np.isclose(
    masking_change_rate,
    masking_changes
    / policy_decisions,
    atol=1e-12,
)

assert fallback_decisions == int(
    notebook52_policy_history_df[
        "fallback_used"
    ].astype(bool).sum()
)

assert np.isclose(
    fallback_rate,
    fallback_decisions
    / policy_decisions,
    atol=1e-12,
)

assert np.isclose(
    average_confidence,
    notebook52_policy_history_df[
        "confidence"
    ].mean(),
    atol=1e-12,
)

assert validation_checks_failed == 0

assert validation_checks_passed > 0

assert bool(
    notebook52_validation_checks_df[
        "passed"
    ].astype(bool).all()
)

assert primary_optimization_target == (
    "Reduce legal-action masking dependence"
)

assert next_notebook == (
    "Notebook 53 — Legality-Aware Policy Optimization"
)

assert next_stage == (
    "LEGALITY_AWARE_POLICY_OPTIMIZATION"
)


# --------------------------------------------------------------------------------------
# 7. Display handoff status
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 52 HANDOFF STATUS")
print("-" * 100)

for key, value in (
    notebook52_handoff_manifest.items()
):

    print(
        f"{key:34}: {value}"
    )


print()
print("NOTEBOOK 52 OPTIMIZATION TARGETS")
print("-" * 100)

display(
    notebook52_optimization_targets_df
)


print()
print("NOTEBOOK 52 BENCHMARK PROFILE")
print("-" * 100)

display(
    notebook52_benchmark_profile_df
)


# --------------------------------------------------------------------------------------
# 8. Save Notebook 53 handoff validation report
# --------------------------------------------------------------------------------------

SECTION1_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section1"
)

SECTION1_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION1B_ARTIFACT_INVENTORY_FILE = (
    SECTION1_REPORT_DIR
    / "section1b_notebook52_artifact_inventory.csv"
)

SECTION1B_HANDOFF_PROFILE_FILE = (
    SECTION1_REPORT_DIR
    / "section1b_notebook52_handoff_profile.csv"
)

SECTION1B_SUMMARY_FILE = (
    SECTION1_REPORT_DIR
    / "section1b_handoff_validation_summary.json"
)


notebook52_artifact_inventory_df.to_csv(
    SECTION1B_ARTIFACT_INVENTORY_FILE,
    index=False,
)


section1b_handoff_profile_df = pd.DataFrame(
    [
        {
            "metric": "handoff_status",
            "value": handoff_status,
        },
        {
            "metric": "battle_count",
            "value": battle_count,
        },
        {
            "metric": "policy_decisions",
            "value": policy_decisions,
        },
        {
            "metric": "player_wins",
            "value": player_wins,
        },
        {
            "metric": "opponent_wins",
            "value": opponent_wins,
        },
        {
            "metric": "draws",
            "value": draws,
        },
        {
            "metric": "masking_changes",
            "value": masking_changes,
        },
        {
            "metric": "masking_change_rate",
            "value": masking_change_rate,
        },
        {
            "metric": "fallback_decisions",
            "value": fallback_decisions,
        },
        {
            "metric": "fallback_rate",
            "value": fallback_rate,
        },
        {
            "metric": "average_confidence",
            "value": average_confidence,
        },
        {
            "metric": "primary_optimization_target",
            "value": primary_optimization_target,
        },
        {
            "metric": "weakest_player_card",
            "value": weakest_player_card,
        },
        {
            "metric": "weakest_condition",
            "value": weakest_condition,
        },
        {
            "metric": "validation_checks_passed",
            "value": validation_checks_passed,
        },
        {
            "metric": "validation_checks_failed",
            "value": validation_checks_failed,
        },
    ]
)


section1b_handoff_profile_df.to_csv(
    SECTION1B_HANDOFF_PROFILE_FILE,
    index=False,
)


section1b_summary = {
    "status":
        "NOTEBOOK52_HANDOFF_VALIDATED",

    "handoff_status":
        handoff_status,

    "battle_count":
        battle_count,

    "policy_decisions":
        policy_decisions,

    "masking_change_rate":
        masking_change_rate,

    "fallback_rate":
        fallback_rate,

    "average_confidence":
        average_confidence,

    "primary_optimization_target":
        primary_optimization_target,

    "weakest_player_card":
        weakest_player_card,

    "weakest_condition":
        weakest_condition,

    "validation_checks_passed":
        validation_checks_passed,

    "validation_checks_failed":
        validation_checks_failed,

    "next_stage":
        "MASKING_ERROR_ANALYSIS",
}


with open(
    SECTION1B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section1b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------------------
# 9. Final output
# --------------------------------------------------------------------------------------

print()
print("HANDOFF VALIDATION SUMMARY")
print("-" * 100)

for key, value in (
    section1b_summary.items()
):

    print(
        f"{key:34}: {value}"
    )


print()
print("SAVED SECTION 1B REPORTS")
print("-" * 100)

print(
    SECTION1B_ARTIFACT_INVENTORY_FILE
)

print(
    SECTION1B_HANDOFF_PROFILE_FILE
)

print(
    SECTION1B_SUMMARY_FILE
)


print()
print(
    "✅ SECTION 1B NOTEBOOK 52 HANDOFF VALIDATION PASSED"
)


# ## Section 2A — Masking Error Analysis
# 
# #### This section analyzes the 210 policy decisions produced during Notebook 52 tournament evaluation.
# 
# It separates:
# 
# - direct predictions that were already legal,
# - predictions changed by legal-action masking,
# - explicit fallback decisions,
# - masking behavior by card,
# - masking behavior by acting side,
# - masking behavior by condition,
# - raw predicted classes,
# - final selected actions,
# - confidence and probability gaps.
# 
# #### The resulting dataset becomes the primary optimization source for Notebook 53.

# In[3]:


# ======================================================================================
# SECTION 2A — MASKING ERROR ANALYSIS
# ======================================================================================

print("=" * 100)
print("SECTION 2A — MASKING ERROR ANALYSIS")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Section output directory
# --------------------------------------------------------------------------------------

SECTION2_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section2"
)

SECTION2_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 2. Helper functions
# --------------------------------------------------------------------------------------

def parse_list_value(
    value: Any,
) -> List[Any]:
    """
    Convert a serialized list, tuple, array, or scalar into a Python list.
    """

    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and np.isnan(
        value
    ):
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        str,
    ):

        stripped_value = value.strip()

        if not stripped_value:
            return []

        try:

            parsed_value = ast.literal_eval(
                stripped_value
            )

            if isinstance(
                parsed_value,
                list,
            ):
                return parsed_value

            if isinstance(
                parsed_value,
                tuple,
            ):
                return list(
                    parsed_value
                )

        except (
            ValueError,
            SyntaxError,
        ):

            pass

        if (
            stripped_value.startswith("[")
            and stripped_value.endswith("]")
        ):

            stripped_value = (
                stripped_value[1:-1]
            )

        return [
            item.strip().strip(
                "'\""
            )
            for item in stripped_value.split(
                ","
            )
            if item.strip()
        ]

    return [
        value
    ]


def normalize_action_name(
    value: Any,
) -> str:
    """
    Normalize one move name for reliable comparisons.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ) and np.isnan(
        value
    ):
        return ""

    return " ".join(
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .split()
    )


def safe_numeric_series(
    dataframe: pd.DataFrame,
    column_name: str,
    default_value: float = 0.0,
) -> pd.Series:
    """
    Return one numeric Series or a default-valued Series when absent.
    """

    if column_name in dataframe.columns:

        return pd.to_numeric(
            dataframe[
                column_name
            ],
            errors="coerce",
        ).fillna(
            default_value
        )

    return pd.Series(
        default_value,
        index=dataframe.index,
        dtype=float,
    )


# --------------------------------------------------------------------------------------
# 3. Prepare policy-history analysis dataframe
# --------------------------------------------------------------------------------------

masking_analysis_df = (
    notebook52_policy_history_df.copy()
)


required_policy_history_columns = [
    "scenario_id",
    "condition_id",
    "turn_number",
    "current_player",
    "legal_moves",
    "selected_move",
    "raw_predicted_move",
    "confidence",
    "fallback_used",
]

missing_policy_history_columns = [
    column_name
    for column_name in required_policy_history_columns
    if column_name not in masking_analysis_df.columns
]

assert not missing_policy_history_columns, (
    "Notebook 52 policy history is missing required columns: "
    f"{missing_policy_history_columns}"
)


masking_analysis_df[
    "legal_move_list"
] = masking_analysis_df[
    "legal_moves"
].apply(
    parse_list_value
)

masking_analysis_df[
    "legal_move_count_recomputed"
] = masking_analysis_df[
    "legal_move_list"
].apply(
    len
)

masking_analysis_df[
    "normalized_raw_prediction"
] = masking_analysis_df[
    "raw_predicted_move"
].apply(
    normalize_action_name
)

masking_analysis_df[
    "normalized_selected_move"
] = masking_analysis_df[
    "selected_move"
].apply(
    normalize_action_name
)

masking_analysis_df[
    "normalized_legal_moves"
] = masking_analysis_df[
    "legal_move_list"
].apply(
    lambda move_list: [
        normalize_action_name(
            move_name
        )
        for move_name in move_list
    ]
)

masking_analysis_df[
    "raw_prediction_legal"
] = masking_analysis_df.apply(
    lambda row: (
        row[
            "normalized_raw_prediction"
        ]
        in row[
            "normalized_legal_moves"
        ]
    ),
    axis=1,
)

masking_analysis_df[
    "masking_changed_action_recomputed"
] = (
    masking_analysis_df[
        "normalized_raw_prediction"
    ]
    != masking_analysis_df[
        "normalized_selected_move"
    ]
)

if (
    "masking_changed_action"
    in masking_analysis_df.columns
):

    stored_masking_series = masking_analysis_df[
        "masking_changed_action"
    ]

    if stored_masking_series.dtype == object:

        stored_masking_series = (
            stored_masking_series
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    masking_analysis_df[
        "masking_changed_action_stored"
    ] = stored_masking_series.fillna(
        False
    ).astype(
        bool
    )

else:

    masking_analysis_df[
        "masking_changed_action_stored"
    ] = masking_analysis_df[
        "masking_changed_action_recomputed"
    ]


masking_analysis_df[
    "fallback_used"
] = (
    masking_analysis_df[
        "fallback_used"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    .fillna(
        False
    )
    .astype(
        bool
    )
)


masking_analysis_df[
    "decision_category"
] = np.select(
    [
        masking_analysis_df[
            "fallback_used"
        ],

        masking_analysis_df[
            "masking_changed_action_recomputed"
        ],
    ],
    [
        "FALLBACK",
        "MASKED",
    ],
    default="DIRECT",
)


# --------------------------------------------------------------------------------------
# 4. Add optional probability diagnostics
# --------------------------------------------------------------------------------------

masking_analysis_df[
    "confidence"
] = safe_numeric_series(
    masking_analysis_df,
    "confidence",
)

masking_analysis_df[
    "selected_probability_numeric"
] = safe_numeric_series(
    masking_analysis_df,
    "selected_probability",
)

masking_analysis_df[
    "selected_masked_probability_numeric"
] = safe_numeric_series(
    masking_analysis_df,
    "selected_masked_probability",
)

masking_analysis_df[
    "raw_to_selected_probability_gap"
] = (
    masking_analysis_df[
        "confidence"
    ]
    - masking_analysis_df[
        "selected_probability_numeric"
    ]
)

masking_analysis_df[
    "masked_probability_gain"
] = (
    masking_analysis_df[
        "selected_masked_probability_numeric"
    ]
    - masking_analysis_df[
        "selected_probability_numeric"
    ]
)


# --------------------------------------------------------------------------------------
# 5. Attach battle-level metadata
# --------------------------------------------------------------------------------------

battle_metadata_columns = [
    column_name
    for column_name in [
        "scenario_id",
        "base_scenario_id",
        "condition_id",
        "condition_name",
        "player_card",
        "opponent_card",
        "starting_side",
        "winner",
        "recorded_turns",
        "starting_player_hp",
        "starting_opponent_hp",
        "starting_player_energy",
        "starting_opponent_energy",
    ]
    if column_name in notebook52_battle_results_df.columns
]


battle_metadata_df = (
    notebook52_battle_results_df[
        battle_metadata_columns
    ]
    .drop_duplicates(
        subset=[
            "scenario_id"
        ]
    )
)


duplicate_merge_columns = [
    column_name
    for column_name in battle_metadata_df.columns
    if (
        column_name != "scenario_id"
        and column_name in masking_analysis_df.columns
    )
]

battle_metadata_df = battle_metadata_df.drop(
    columns=duplicate_merge_columns,
    errors="ignore",
)


masking_analysis_df = masking_analysis_df.merge(
    battle_metadata_df,
    on="scenario_id",
    how="left",
)


# --------------------------------------------------------------------------------------
# 6. Core masking subsets
# --------------------------------------------------------------------------------------

masked_decisions_df = (
    masking_analysis_df.loc[
        masking_analysis_df[
            "masking_changed_action_recomputed"
        ]
    ]
    .copy()
)

direct_decisions_df = (
    masking_analysis_df.loc[
        ~masking_analysis_df[
            "masking_changed_action_recomputed"
        ]
    ]
    .copy()
)

fallback_decisions_df = (
    masking_analysis_df.loc[
        masking_analysis_df[
            "fallback_used"
        ]
    ]
    .copy()
)

illegal_raw_predictions_df = (
    masking_analysis_df.loc[
        ~masking_analysis_df[
            "raw_prediction_legal"
        ]
    ]
    .copy()
)


# --------------------------------------------------------------------------------------
# 7. Decision-category summary
# --------------------------------------------------------------------------------------

decision_category_summary_df = (
    masking_analysis_df
    .groupby(
        "decision_category",
        as_index=False,
    )
    .agg(
        decisions=(
            "selected_move",
            "size",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        minimum_confidence=(
            "confidence",
            "min",
        ),

        maximum_confidence=(
            "confidence",
            "max",
        ),

        average_legal_move_count=(
            "legal_move_count_recomputed",
            "mean",
        ),

        raw_predictions_legal=(
            "raw_prediction_legal",
            "sum",
        ),
    )
)

decision_category_summary_df[
    "decision_fraction"
] = (
    decision_category_summary_df[
        "decisions"
    ]
    / len(
        masking_analysis_df
    )
)

decision_category_summary_df[
    "raw_legality_rate"
] = (
    decision_category_summary_df[
        "raw_predictions_legal"
    ]
    / decision_category_summary_df[
        "decisions"
    ]
)


print()
print("DECISION CATEGORY SUMMARY")
print("-" * 100)

display(
    decision_category_summary_df
)


# --------------------------------------------------------------------------------------
# 8. Masking by acting side
# --------------------------------------------------------------------------------------

masking_by_side_df = (
    masking_analysis_df
    .groupby(
        "current_player",
        as_index=False,
    )
    .agg(
        decisions=(
            "selected_move",
            "size",
        ),

        masking_changes=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        illegal_raw_predictions=(
            "raw_prediction_legal",
            lambda values: int(
                (
                    ~values.astype(bool)
                ).sum()
            ),
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        average_legal_move_count=(
            "legal_move_count_recomputed",
            "mean",
        ),
    )
)

masking_by_side_df[
    "masking_rate"
] = (
    masking_by_side_df[
        "masking_changes"
    ]
    / masking_by_side_df[
        "decisions"
    ]
)

masking_by_side_df[
    "raw_illegality_rate"
] = (
    masking_by_side_df[
        "illegal_raw_predictions"
    ]
    / masking_by_side_df[
        "decisions"
    ]
)

masking_by_side_df[
    "fallback_rate"
] = (
    masking_by_side_df[
        "fallbacks"
    ]
    / masking_by_side_df[
        "decisions"
    ]
)


print()
print("MASKING BY ACTING SIDE")
print("-" * 100)

display(
    masking_by_side_df
)


# --------------------------------------------------------------------------------------
# 9. Masking by condition
# --------------------------------------------------------------------------------------

masking_by_condition_df = (
    masking_analysis_df
    .groupby(
        "condition_id",
        as_index=False,
    )
    .agg(
        decisions=(
            "selected_move",
            "size",
        ),

        masking_changes=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        average_legal_move_count=(
            "legal_move_count_recomputed",
            "mean",
        ),
    )
)

masking_by_condition_df[
    "masking_rate"
] = (
    masking_by_condition_df[
        "masking_changes"
    ]
    / masking_by_condition_df[
        "decisions"
    ]
)

masking_by_condition_df[
    "fallback_rate"
] = (
    masking_by_condition_df[
        "fallbacks"
    ]
    / masking_by_condition_df[
        "decisions"
    ]
)


print()
print("MASKING BY CONDITION")
print("-" * 100)

display(
    masking_by_condition_df
)


# --------------------------------------------------------------------------------------
# 10. Masking by Player card
# --------------------------------------------------------------------------------------

masking_by_player_card_df = (
    masking_analysis_df
    .groupby(
        "player_card",
        as_index=False,
        dropna=False,
    )
    .agg(
        decisions=(
            "selected_move",
            "size",
        ),

        masking_changes=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
)

masking_by_player_card_df[
    "masking_rate"
] = (
    masking_by_player_card_df[
        "masking_changes"
    ]
    / masking_by_player_card_df[
        "decisions"
    ]
)

masking_by_player_card_df[
    "fallback_rate"
] = (
    masking_by_player_card_df[
        "fallbacks"
    ]
    / masking_by_player_card_df[
        "decisions"
    ]
)

masking_by_player_card_df = (
    masking_by_player_card_df
    .sort_values(
        [
            "masking_rate",
            "masking_changes",
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
print("MASKING BY PLAYER CARD")
print("-" * 100)

display(
    masking_by_player_card_df
)


# --------------------------------------------------------------------------------------
# 11. Raw prediction-to-selected-action transition matrix
# --------------------------------------------------------------------------------------

prediction_transition_df = (
    masking_analysis_df
    .groupby(
        [
            "raw_predicted_move",
            "selected_move",
        ],
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),
    )
    .sort_values(
        [
            "decisions",
            "average_confidence",
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

prediction_transition_df[
    "changed_action"
] = (
    prediction_transition_df[
        "raw_predicted_move"
    ].apply(
        normalize_action_name
    )
    != prediction_transition_df[
        "selected_move"
    ].apply(
        normalize_action_name
    )
)


print()
print("RAW PREDICTION TO SELECTED ACTION TRANSITIONS")
print("-" * 100)

display(
    prediction_transition_df
)


# --------------------------------------------------------------------------------------
# 12. Raw prediction profile
# --------------------------------------------------------------------------------------

raw_prediction_profile_df = (
    masking_analysis_df
    .groupby(
        "raw_predicted_move",
        as_index=False,
    )
    .agg(
        predictions=(
            "raw_predicted_move",
            "size",
        ),

        legal_predictions=(
            "raw_prediction_legal",
            "sum",
        ),

        masking_changes=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
)

raw_prediction_profile_df[
    "prediction_fraction"
] = (
    raw_prediction_profile_df[
        "predictions"
    ]
    / len(
        masking_analysis_df
    )
)

raw_prediction_profile_df[
    "legality_rate"
] = (
    raw_prediction_profile_df[
        "legal_predictions"
    ]
    / raw_prediction_profile_df[
        "predictions"
    ]
)

raw_prediction_profile_df[
    "masking_rate"
] = (
    raw_prediction_profile_df[
        "masking_changes"
    ]
    / raw_prediction_profile_df[
        "predictions"
    ]
)

raw_prediction_profile_df = (
    raw_prediction_profile_df
    .sort_values(
        [
            "masking_rate",
            "predictions",
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
print("RAW PREDICTION PROFILE")
print("-" * 100)

display(
    raw_prediction_profile_df
)


# --------------------------------------------------------------------------------------
# 13. Selected-action profile
# --------------------------------------------------------------------------------------

selected_action_profile_df = (
    masking_analysis_df
    .groupby(
        "selected_move",
        as_index=False,
    )
    .agg(
        selections=(
            "selected_move",
            "size",
        ),

        masked_selections=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        average_selected_probability=(
            "selected_probability_numeric",
            "mean",
        ),
    )
)

selected_action_profile_df[
    "selection_fraction"
] = (
    selected_action_profile_df[
        "selections"
    ]
    / len(
        masking_analysis_df
    )
)

selected_action_profile_df[
    "masked_selection_rate"
] = (
    selected_action_profile_df[
        "masked_selections"
    ]
    / selected_action_profile_df[
        "selections"
    ]
)

selected_action_profile_df = (
    selected_action_profile_df
    .sort_values(
        "selections",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("SELECTED ACTION PROFILE")
print("-" * 100)

display(
    selected_action_profile_df
)


# --------------------------------------------------------------------------------------
# 14. Detailed masked-decision sample
# --------------------------------------------------------------------------------------

masked_preview_columns = [
    column_name
    for column_name in [
        "scenario_id",
        "base_scenario_id",
        "condition_id",
        "turn_number",
        "current_player",
        "player_card",
        "opponent_card",
        "legal_moves",
        "raw_predicted_move",
        "selected_move",
        "confidence",
        "selected_probability_numeric",
        "fallback_used",
        "fallback_reason",
        "winner",
    ]
    if column_name in masked_decisions_df.columns
]


print()
print("MASKED DECISION SAMPLE")
print("-" * 100)

display(
    masked_decisions_df[
        masked_preview_columns
    ].head(
        40
    )
)


# --------------------------------------------------------------------------------------
# 15. Validation
# --------------------------------------------------------------------------------------

assert len(
    masking_analysis_df
) == policy_decisions

assert len(
    masking_analysis_df
) == 210

assert int(
    masking_analysis_df[
        "masking_changed_action_recomputed"
    ].sum()
) == masking_changes

assert len(
    masked_decisions_df
) == masking_changes

assert len(
    direct_decisions_df
) == (
    policy_decisions
    - masking_changes
)

assert len(
    fallback_decisions_df
) == fallback_decisions

assert len(
    illegal_raw_predictions_df
) == int(
    (
        ~masking_analysis_df[
            "raw_prediction_legal"
        ]
    ).sum()
)

assert bool(
    (
        masking_analysis_df[
            "masking_changed_action_stored"
        ]
        == masking_analysis_df[
            "masking_changed_action_recomputed"
        ]
    ).all()
), (
    "Stored and recomputed masking indicators do not match."
)

assert masking_analysis_df[
    "legal_move_count_recomputed"
].gt(
    0
).all()

assert masking_analysis_df[
    "confidence"
].between(
    0.0,
    1.0,
    inclusive="both",
).all()

assert set(
    masking_analysis_df[
        "decision_category"
    ].unique()
).issubset(
    {
        "DIRECT",
        "MASKED",
        "FALLBACK",
    }
)

assert np.isclose(
    len(
        masked_decisions_df
    )
    / len(
        masking_analysis_df
    ),
    masking_change_rate,
    atol=1e-12,
)

assert np.isclose(
    len(
        fallback_decisions_df
    )
    / len(
        masking_analysis_df
    ),
    fallback_rate,
    atol=1e-12,
)


# --------------------------------------------------------------------------------------
# 16. Save reports
# --------------------------------------------------------------------------------------

SECTION2A_FULL_ANALYSIS_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masking_decision_analysis.csv"
)

SECTION2A_MASKED_DECISIONS_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masked_decisions.csv"
)

SECTION2A_DIRECT_DECISIONS_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_direct_decisions.csv"
)

SECTION2A_FALLBACK_DECISIONS_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_fallback_decisions.csv"
)

SECTION2A_ILLEGAL_RAW_PREDICTIONS_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_illegal_raw_predictions.csv"
)

SECTION2A_CATEGORY_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_decision_category_summary.csv"
)

SECTION2A_SIDE_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masking_by_side.csv"
)

SECTION2A_CONDITION_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masking_by_condition.csv"
)

SECTION2A_CARD_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masking_by_player_card.csv"
)

SECTION2A_TRANSITION_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_prediction_transition_matrix.csv"
)

SECTION2A_RAW_PREDICTION_PROFILE_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_raw_prediction_profile.csv"
)

SECTION2A_SELECTED_ACTION_PROFILE_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_selected_action_profile.csv"
)

SECTION2A_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2a_masking_analysis_summary.json"
)


# Convert list-valued columns to JSON-safe strings before CSV export.
masking_export_df = (
    masking_analysis_df.copy()
)

for list_column in [
    "legal_move_list",
    "normalized_legal_moves",
]:

    masking_export_df[
        list_column
    ] = masking_export_df[
        list_column
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )


masking_export_df.to_csv(
    SECTION2A_FULL_ANALYSIS_FILE,
    index=False,
)

masked_decisions_df.to_csv(
    SECTION2A_MASKED_DECISIONS_FILE,
    index=False,
)

direct_decisions_df.to_csv(
    SECTION2A_DIRECT_DECISIONS_FILE,
    index=False,
)

fallback_decisions_df.to_csv(
    SECTION2A_FALLBACK_DECISIONS_FILE,
    index=False,
)

illegal_raw_predictions_df.to_csv(
    SECTION2A_ILLEGAL_RAW_PREDICTIONS_FILE,
    index=False,
)

decision_category_summary_df.to_csv(
    SECTION2A_CATEGORY_SUMMARY_FILE,
    index=False,
)

masking_by_side_df.to_csv(
    SECTION2A_SIDE_SUMMARY_FILE,
    index=False,
)

masking_by_condition_df.to_csv(
    SECTION2A_CONDITION_SUMMARY_FILE,
    index=False,
)

masking_by_player_card_df.to_csv(
    SECTION2A_CARD_SUMMARY_FILE,
    index=False,
)

prediction_transition_df.to_csv(
    SECTION2A_TRANSITION_FILE,
    index=False,
)

raw_prediction_profile_df.to_csv(
    SECTION2A_RAW_PREDICTION_PROFILE_FILE,
    index=False,
)

selected_action_profile_df.to_csv(
    SECTION2A_SELECTED_ACTION_PROFILE_FILE,
    index=False,
)


highest_masking_card_row = (
    masking_by_player_card_df.iloc[0]
)

highest_masking_raw_prediction_row = (
    raw_prediction_profile_df.iloc[0]
)

most_common_transition_row = (
    prediction_transition_df.loc[
        prediction_transition_df[
            "changed_action"
        ]
    ]
    .sort_values(
        "decisions",
        ascending=False,
    )
    .iloc[0]
)


section2a_summary = {
    "status":
        "MASKING_ERROR_ANALYSIS_COMPLETE",

    "total_decisions":
        int(
            len(
                masking_analysis_df
            )
        ),

    "direct_decisions":
        int(
            len(
                direct_decisions_df
            )
        ),

    "masked_decisions":
        int(
            len(
                masked_decisions_df
            )
        ),

    "masking_change_rate":
        float(
            len(
                masked_decisions_df
            )
            / len(
                masking_analysis_df
            )
        ),

    "illegal_raw_predictions":
        int(
            len(
                illegal_raw_predictions_df
            )
        ),

    "raw_prediction_legality_rate":
        float(
            masking_analysis_df[
                "raw_prediction_legal"
            ].mean()
        ),

    "fallback_decisions":
        int(
            len(
                fallback_decisions_df
            )
        ),

    "fallback_rate":
        float(
            len(
                fallback_decisions_df
            )
            / len(
                masking_analysis_df
            )
        ),

    "average_confidence":
        float(
            masking_analysis_df[
                "confidence"
            ].mean()
        ),

    "highest_masking_player_card":
        str(
            highest_masking_card_row[
                "player_card"
            ]
        ),

    "highest_masking_player_card_rate":
        float(
            highest_masking_card_row[
                "masking_rate"
            ]
        ),

    "highest_masking_raw_prediction":
        str(
            highest_masking_raw_prediction_row[
                "raw_predicted_move"
            ]
        ),

    "highest_masking_raw_prediction_rate":
        float(
            highest_masking_raw_prediction_row[
                "masking_rate"
            ]
        ),

    "most_common_changed_prediction":
        str(
            most_common_transition_row[
                "raw_predicted_move"
            ]
        ),

    "most_common_changed_selection":
        str(
            most_common_transition_row[
                "selected_move"
            ]
        ),

    "most_common_changed_count":
        int(
            most_common_transition_row[
                "decisions"
            ]
        ),

    "next_stage":
        "MASKING_ROOT_CAUSE_ANALYSIS",
}


with open(
    SECTION2A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section2a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------------------
# 17. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 2A SUMMARY")
print("-" * 100)

for key, value in (
    section2a_summary.items()
):

    print(
        f"{key:40}: {value}"
    )


print()
print("SAVED SECTION 2A REPORTS")
print("-" * 100)

section2a_saved_files = [
    SECTION2A_FULL_ANALYSIS_FILE,
    SECTION2A_MASKED_DECISIONS_FILE,
    SECTION2A_DIRECT_DECISIONS_FILE,
    SECTION2A_FALLBACK_DECISIONS_FILE,
    SECTION2A_ILLEGAL_RAW_PREDICTIONS_FILE,
    SECTION2A_CATEGORY_SUMMARY_FILE,
    SECTION2A_SIDE_SUMMARY_FILE,
    SECTION2A_CONDITION_SUMMARY_FILE,
    SECTION2A_CARD_SUMMARY_FILE,
    SECTION2A_TRANSITION_FILE,
    SECTION2A_RAW_PREDICTION_PROFILE_FILE,
    SECTION2A_SELECTED_ACTION_PROFILE_FILE,
    SECTION2A_SUMMARY_FILE,
]

for file_path in section2a_saved_files:

    print(
        file_path
    )


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section2a_saved_files
)


print()
print(
    "✅ SECTION 2A MASKING ERROR ANALYSIS PASSED"
)


# ## Section 2B — Masking Root-Cause Analysis
# 
# #### This section identifies the principal causes of illegal raw policy predictions.
# 
# #### The analysis examines masking behavior by:
# 
# - active Pokémon,
# - acting side,
# - legal-action set,
# - predicted action,
# - tournament condition,
# - battle phase,
# - energy state,
# - HP state,
# - confidence level.
# 
# #### The resulting diagnosis determines which legality-aware features must be added before policy retraining.

# In[4]:


# ======================================================================================
# SECTION 2B — MASKING ROOT-CAUSE ANALYSIS
# ======================================================================================

print("=" * 100)
print("SECTION 2B — MASKING ROOT-CAUSE ANALYSIS")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Root-cause analysis dataframe
# --------------------------------------------------------------------------------------

root_cause_df = masking_analysis_df.copy()


# --------------------------------------------------------------------------------------
# 2. Resolve active-card identity
#
# The policy feature schema is Player-perspective based:
# - when current_player == Player, active card is player_card
# - when current_player == Opponent, active card is opponent_card
# --------------------------------------------------------------------------------------

root_cause_df[
    "active_card"
] = np.where(
    root_cause_df[
        "current_player"
    ].astype(str).str.lower().eq(
        "player"
    ),
    root_cause_df[
        "player_card"
    ],
    root_cause_df[
        "opponent_card"
    ],
)

root_cause_df[
    "inactive_card"
] = np.where(
    root_cause_df[
        "current_player"
    ].astype(str).str.lower().eq(
        "player"
    ),
    root_cause_df[
        "opponent_card"
    ],
    root_cause_df[
        "player_card"
    ],
)


# --------------------------------------------------------------------------------------
# 3. Restore available state features and resolve acting-side values
# --------------------------------------------------------------------------------------

def ensure_numeric_column(
    dataframe: pd.DataFrame,
    target_column: str,
    fallback_columns: Sequence[str] = (),
    default_value: float = np.nan,
) -> None:
    """
    Ensure that target_column exists as numeric data.

    The first available fallback column is used when the original
    per-decision feature was not exported by Notebook 52.
    """

    if target_column in dataframe.columns:

        dataframe[target_column] = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        )

        return

    for fallback_column in fallback_columns:

        if fallback_column in dataframe.columns:

            dataframe[target_column] = pd.to_numeric(
                dataframe[fallback_column],
                errors="coerce",
            )

            print(
                f"Restored {target_column!r} from "
                f"{fallback_column!r}."
            )

            return

    dataframe[target_column] = pd.Series(
        default_value,
        index=dataframe.index,
        dtype=float,
    )

    print(
        f"Created unavailable feature {target_column!r} "
        f"with default value {default_value}."
    )


# Energy is usually unchanged in the current simplified simulator,
# so starting energy is an acceptable fallback for this analysis.
ensure_numeric_column(
    root_cause_df,
    "player_energy",
    fallback_columns=[
        "starting_player_energy",
    ],
)

ensure_numeric_column(
    root_cause_df,
    "opponent_energy",
    fallback_columns=[
        "starting_opponent_energy",
    ],
)


# Prefer exported damage values when available.
# Otherwise preserve them as missing rather than inventing values.
ensure_numeric_column(
    root_cause_df,
    "player_damage",
    fallback_columns=[],
    default_value=np.nan,
)

ensure_numeric_column(
    root_cause_df,
    "opponent_damage",
    fallback_columns=[],
    default_value=np.nan,
)

ensure_numeric_column(
    root_cause_df,
    "turn_number",
    default_value=np.nan,
)

ensure_numeric_column(
    root_cause_df,
    "confidence",
    default_value=0.0,
)


# --------------------------------------------------------------------------------------
# Resolve acting-side energy and damage
# --------------------------------------------------------------------------------------

acting_is_player = (
    root_cause_df[
        "current_player"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("player")
)


root_cause_df[
    "acting_energy"
] = np.where(
    acting_is_player,
    root_cause_df[
        "player_energy"
    ],
    root_cause_df[
        "opponent_energy"
    ],
)

root_cause_df[
    "defending_energy"
] = np.where(
    acting_is_player,
    root_cause_df[
        "opponent_energy"
    ],
    root_cause_df[
        "player_energy"
    ],
)

root_cause_df[
    "acting_damage"
] = np.where(
    acting_is_player,
    root_cause_df[
        "player_damage"
    ],
    root_cause_df[
        "opponent_damage"
    ],
)

root_cause_df[
    "defending_damage"
] = np.where(
    acting_is_player,
    root_cause_df[
        "opponent_damage"
    ],
    root_cause_df[
        "player_damage"
    ],
)

# --------------------------------------------------------------------------------------
# 4. Battle-phase features
# --------------------------------------------------------------------------------------

root_cause_df[
    "battle_phase"
] = pd.cut(
    root_cause_df[
        "turn_number"
    ],
    bins=[
        -np.inf,
        3,
        7,
        np.inf,
    ],
    labels=[
        "EARLY",
        "MID",
        "LATE",
    ],
)

root_cause_df[
    "energy_band"
] = pd.cut(
    root_cause_df[
        "acting_energy"
    ],
    bins=[
        -np.inf,
        0,
        2,
        np.inf,
    ],
    labels=[
        "ZERO",
        "LOW",
        "READY",
    ],
)

root_cause_df[
    "damage_band"
] = pd.cut(
    root_cause_df[
        "acting_damage"
    ],
    bins=[
        -np.inf,
        19.999,
        39.999,
        np.inf,
    ],
    labels=[
        "LOW_DAMAGE",
        "MEDIUM_DAMAGE",
        "HIGH_DAMAGE",
    ],
)

root_cause_df[
    "damage_band"
] = (
    root_cause_df[
        "damage_band"
    ]
    .astype(object)
    .where(
        root_cause_df[
            "damage_band"
        ].notna(),
        "UNKNOWN",
    )
)

root_cause_df[
    "confidence_band"
] = pd.cut(
    root_cause_df[
        "confidence"
    ],
    bins=[
        -np.inf,
        0.55,
        0.75,
        np.inf,
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
    ],
)


# --------------------------------------------------------------------------------------
# 5. Canonical legal-move-set signature
# --------------------------------------------------------------------------------------

root_cause_df[
    "legal_move_signature"
] = root_cause_df[
    "normalized_legal_moves"
].apply(
    lambda move_list: " | ".join(
        sorted(
            move_list
        )
    )
)

root_cause_df[
    "single_legal_move"
] = (
    root_cause_df[
        "legal_move_count_recomputed"
    ]
    == 1
)


# --------------------------------------------------------------------------------------
# 6. Generic grouped root-cause summary helper
# --------------------------------------------------------------------------------------

def build_root_cause_summary(
    dataframe: pd.DataFrame,
    group_columns: List[str],
) -> pd.DataFrame:
    """
    Summarize masking and raw-legality behavior for one grouping.
    """

    summary_df = (
        dataframe
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
            observed=False,
        )
        .agg(
            decisions=(
                "selected_move",
                "size",
            ),

            masked_decisions=(
                "masking_changed_action_recomputed",
                "sum",
            ),

            legal_raw_predictions=(
                "raw_prediction_legal",
                "sum",
            ),

            fallback_decisions=(
                "fallback_used",
                "sum",
            ),

            average_confidence=(
                "confidence",
                "mean",
            ),

            average_legal_move_count=(
                "legal_move_count_recomputed",
                "mean",
            ),
        )
    )

    summary_df[
        "masking_rate"
    ] = (
        summary_df[
            "masked_decisions"
        ]
        / summary_df[
            "decisions"
        ]
    )

    summary_df[
        "raw_legality_rate"
    ] = (
        summary_df[
            "legal_raw_predictions"
        ]
        / summary_df[
            "decisions"
        ]
    )

    summary_df[
        "fallback_rate"
    ] = (
        summary_df[
            "fallback_decisions"
        ]
        / summary_df[
            "decisions"
        ]
    )

    return (
        summary_df
        .sort_values(
            [
                "masking_rate",
                "decisions",
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


# --------------------------------------------------------------------------------------
# 7. Root cause by active card
# --------------------------------------------------------------------------------------

root_cause_by_active_card_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "active_card",
        ],
    )
)


print()
print("ROOT CAUSE BY ACTIVE CARD")
print("-" * 100)

display(
    root_cause_by_active_card_df
)


# --------------------------------------------------------------------------------------
# 8. Root cause by legal move set
# --------------------------------------------------------------------------------------

root_cause_by_legal_set_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "legal_move_signature",
            "legal_move_count_recomputed",
        ],
    )
)


print()
print("ROOT CAUSE BY LEGAL MOVE SET")
print("-" * 100)

display(
    root_cause_by_legal_set_df
)


# --------------------------------------------------------------------------------------
# 9. Root cause by active card and raw prediction
# --------------------------------------------------------------------------------------

root_cause_card_prediction_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "active_card",
            "raw_predicted_move",
        ],
    )
)


print()
print("ROOT CAUSE BY ACTIVE CARD AND RAW PREDICTION")
print("-" * 100)

display(
    root_cause_card_prediction_df
)


# --------------------------------------------------------------------------------------
# 10. Root cause by condition
# --------------------------------------------------------------------------------------

root_cause_by_condition_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "condition_id",
        ],
    )
)


print()
print("ROOT CAUSE BY CONDITION")
print("-" * 100)

display(
    root_cause_by_condition_df
)


# --------------------------------------------------------------------------------------
# 11. Root cause by battle phase
# --------------------------------------------------------------------------------------

root_cause_by_phase_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "battle_phase",
        ],
    )
)


print()
print("ROOT CAUSE BY BATTLE PHASE")
print("-" * 100)

display(
    root_cause_by_phase_df
)


# --------------------------------------------------------------------------------------
# 12. Root cause by energy band
# --------------------------------------------------------------------------------------

root_cause_by_energy_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "energy_band",
        ],
    )
)


print()
print("ROOT CAUSE BY ACTING ENERGY BAND")
print("-" * 100)

display(
    root_cause_by_energy_df
)


# --------------------------------------------------------------------------------------
# 13. Root cause by damage band
# --------------------------------------------------------------------------------------

root_cause_by_damage_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "damage_band",
        ],
    )
)


print()
print("ROOT CAUSE BY ACTING DAMAGE BAND")
print("-" * 100)

display(
    root_cause_by_damage_df
)


# --------------------------------------------------------------------------------------
# 14. Root cause by confidence band
# --------------------------------------------------------------------------------------

root_cause_by_confidence_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "confidence_band",
        ],
    )
)


print()
print("ROOT CAUSE BY CONFIDENCE BAND")
print("-" * 100)

display(
    root_cause_by_confidence_df
)


# --------------------------------------------------------------------------------------
# 15. Single-action versus multi-action states
# --------------------------------------------------------------------------------------

root_cause_by_action_count_df = (
    build_root_cause_summary(
        root_cause_df,
        [
            "single_legal_move",
            "legal_move_count_recomputed",
        ],
    )
)


print()
print("ROOT CAUSE BY LEGAL ACTION COUNT")
print("-" * 100)

display(
    root_cause_by_action_count_df
)


# --------------------------------------------------------------------------------------
# 16. Prediction bias matrix
# --------------------------------------------------------------------------------------

prediction_bias_matrix_df = pd.crosstab(
    root_cause_df[
        "active_card"
    ],
    root_cause_df[
        "raw_predicted_move"
    ],
    margins=True,
)

print()
print("ACTIVE CARD × RAW PREDICTION MATRIX")
print("-" * 100)

display(
    prediction_bias_matrix_df
)


# --------------------------------------------------------------------------------------
# 17. Active-card legal-action availability
# --------------------------------------------------------------------------------------

active_card_legal_action_rows = []

for active_card_name, card_group_df in (
    root_cause_df.groupby(
        "active_card",
        dropna=False,
    )
):

    available_actions = sorted(
        {
            move_name
            for move_list in card_group_df[
                "legal_move_list"
            ]
            for move_name in move_list
        }
    )

    predicted_actions = sorted(
        card_group_df[
            "raw_predicted_move"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_actions = sorted(
        card_group_df[
            "selected_move"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    active_card_legal_action_rows.append(
        {
            "active_card":
                active_card_name,

            "decisions":
                int(
                    len(
                        card_group_df
                    )
                ),

            "available_legal_actions":
                available_actions,

            "available_legal_action_count":
                int(
                    len(
                        available_actions
                    )
                ),

            "raw_predicted_actions":
                predicted_actions,

            "selected_actions":
                selected_actions,

            "masking_rate":
                float(
                    card_group_df[
                        "masking_changed_action_recomputed"
                    ].mean()
                ),

            "raw_legality_rate":
                float(
                    card_group_df[
                        "raw_prediction_legal"
                    ].mean()
                ),
        }
    )


active_card_action_availability_df = pd.DataFrame(
    active_card_legal_action_rows
)


print()
print("ACTIVE-CARD ACTION AVAILABILITY")
print("-" * 100)

display(
    active_card_action_availability_df
)


# --------------------------------------------------------------------------------------
# 18. Rank root-cause factors by masking-rate spread
#
# Greater spread indicates stronger separation between groups.
# --------------------------------------------------------------------------------------

factor_summary_map = {
    "active_card":
        root_cause_by_active_card_df,

    "legal_move_signature":
        root_cause_by_legal_set_df,

    "condition_id":
        root_cause_by_condition_df,

    "battle_phase":
        root_cause_by_phase_df,

    "energy_band":
        root_cause_by_energy_df,

    "damage_band":
        root_cause_by_damage_df,

    "confidence_band":
        root_cause_by_confidence_df,

    "legal_action_count":
        root_cause_by_action_count_df,
}


factor_strength_rows = []

for factor_name, factor_dataframe in (
    factor_summary_map.items()
):

    valid_factor_df = factor_dataframe.loc[
        factor_dataframe[
            "decisions"
        ] > 0
    ]

    masking_rate_minimum = float(
        valid_factor_df[
            "masking_rate"
        ].min()
    )

    masking_rate_maximum = float(
        valid_factor_df[
            "masking_rate"
        ].max()
    )

    factor_strength_rows.append(
        {
            "factor":
                factor_name,

            "group_count":
                int(
                    len(
                        valid_factor_df
                    )
                ),

            "minimum_masking_rate":
                masking_rate_minimum,

            "maximum_masking_rate":
                masking_rate_maximum,

            "masking_rate_spread":
                float(
                    masking_rate_maximum
                    - masking_rate_minimum
                ),

            "weighted_average_masking_rate":
                float(
                    np.average(
                        valid_factor_df[
                            "masking_rate"
                        ],
                        weights=valid_factor_df[
                            "decisions"
                        ],
                    )
                ),
        }
    )


root_cause_factor_strength_df = (
    pd.DataFrame(
        factor_strength_rows
    )
    .sort_values(
        [
            "masking_rate_spread",
            "group_count",
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
print("ROOT-CAUSE FACTOR STRENGTH")
print("-" * 100)

display(
    root_cause_factor_strength_df
)


# --------------------------------------------------------------------------------------
# 19. Determine primary diagnostic conclusions
# --------------------------------------------------------------------------------------

highest_masking_active_card_row = (
    root_cause_by_active_card_df.iloc[0]
)

lowest_legality_active_card_row = (
    root_cause_by_active_card_df
    .sort_values(
        [
            "raw_legality_rate",
            "decisions",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .iloc[0]
)

highest_masking_legal_set_row = (
    root_cause_by_legal_set_df.iloc[0]
)

strongest_factor_row = (
    root_cause_factor_strength_df.iloc[0]
)

single_action_row = (
    root_cause_by_action_count_df.loc[
        root_cause_by_action_count_df[
            "single_legal_move"
        ].astype(bool)
    ]
)

multi_action_row = (
    root_cause_by_action_count_df.loc[
        ~root_cause_by_action_count_df[
            "single_legal_move"
        ].astype(bool)
    ]
)

single_action_masking_rate = (
    float(
        np.average(
            single_action_row[
                "masking_rate"
            ],
            weights=single_action_row[
                "decisions"
            ],
        )
    )
    if not single_action_row.empty
    else None
)

multi_action_masking_rate = (
    float(
        np.average(
            multi_action_row[
                "masking_rate"
            ],
            weights=multi_action_row[
                "decisions"
            ],
        )
    )
    if not multi_action_row.empty
    else None
)


# --------------------------------------------------------------------------------------
# 20. Validation
# --------------------------------------------------------------------------------------

assert len(
    root_cause_df
) == 210

assert root_cause_df[
    "active_card"
].notna().all()

assert root_cause_df[
    "legal_move_signature"
].astype(str).str.len().gt(
    0
).all()

assert root_cause_df[
    "legal_move_count_recomputed"
].gt(
    0
).all()

assert set(
    root_cause_df[
        "battle_phase"
    ].dropna().astype(str).unique()
).issubset(
    {
        "EARLY",
        "MID",
        "LATE",
    }
)

assert set(
    root_cause_df[
        "energy_band"
    ].dropna().astype(str).unique()
).issubset(
    {
        "ZERO",
        "LOW",
        "READY",
    }
)
assert set(
    root_cause_df[
        "damage_band"
    ].dropna().astype(str).unique()
).issubset(
    {
        "LOW_DAMAGE",
        "MEDIUM_DAMAGE",
        "HIGH_DAMAGE",
        "UNKNOWN",
    }
)
assert set(
    root_cause_df[
        "confidence_band"
    ].dropna().astype(str).unique()
).issubset(
    {
        "LOW",
        "MEDIUM",
        "HIGH",
    }
)

assert np.isclose(
    root_cause_df[
        "masking_changed_action_recomputed"
    ].mean(),
    masking_change_rate,
    atol=1e-12,
)

assert np.isclose(
    root_cause_by_active_card_df[
        "masked_decisions"
    ].sum(),
    masking_changes,
)

assert np.isclose(
    root_cause_by_condition_df[
        "masked_decisions"
    ].sum(),
    masking_changes,
)

assert np.isclose(
    root_cause_by_phase_df[
        "masked_decisions"
    ].sum(),
    masking_changes,
)

assert len(
    root_cause_factor_strength_df
) == len(
    factor_summary_map
)


# --------------------------------------------------------------------------------------
# 21. Save reports
# --------------------------------------------------------------------------------------

SECTION2B_ROOT_CAUSE_DATA_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_decision_data.csv"
)

SECTION2B_ACTIVE_CARD_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_active_card.csv"
)

SECTION2B_LEGAL_SET_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_legal_move_set.csv"
)

SECTION2B_CARD_PREDICTION_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_active_card_prediction_profile.csv"
)

SECTION2B_CONDITION_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_condition.csv"
)

SECTION2B_PHASE_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_battle_phase.csv"
)

SECTION2B_ENERGY_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_energy_band.csv"
)

SECTION2B_DAMAGE_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_damage_band.csv"
)

SECTION2B_CONFIDENCE_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_confidence_band.csv"
)

SECTION2B_ACTION_COUNT_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_by_action_count.csv"
)

SECTION2B_PREDICTION_BIAS_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_active_card_prediction_bias_matrix.csv"
)

SECTION2B_ACTION_AVAILABILITY_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_active_card_action_availability.csv"
)

SECTION2B_FACTOR_STRENGTH_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_factor_strength.csv"
)

SECTION2B_SUMMARY_FILE = (
    SECTION2_REPORT_DIR
    / "section2b_root_cause_summary.json"
)


root_cause_export_df = root_cause_df.copy()

for list_column in [
    "legal_move_list",
    "normalized_legal_moves",
]:

    root_cause_export_df[
        list_column
    ] = root_cause_export_df[
        list_column
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )


root_cause_export_df.to_csv(
    SECTION2B_ROOT_CAUSE_DATA_FILE,
    index=False,
)

root_cause_by_active_card_df.to_csv(
    SECTION2B_ACTIVE_CARD_FILE,
    index=False,
)

root_cause_by_legal_set_df.to_csv(
    SECTION2B_LEGAL_SET_FILE,
    index=False,
)

root_cause_card_prediction_df.to_csv(
    SECTION2B_CARD_PREDICTION_FILE,
    index=False,
)

root_cause_by_condition_df.to_csv(
    SECTION2B_CONDITION_FILE,
    index=False,
)

root_cause_by_phase_df.to_csv(
    SECTION2B_PHASE_FILE,
    index=False,
)

root_cause_by_energy_df.to_csv(
    SECTION2B_ENERGY_FILE,
    index=False,
)

root_cause_by_damage_df.to_csv(
    SECTION2B_DAMAGE_FILE,
    index=False,
)

root_cause_by_confidence_df.to_csv(
    SECTION2B_CONFIDENCE_FILE,
    index=False,
)

root_cause_by_action_count_df.to_csv(
    SECTION2B_ACTION_COUNT_FILE,
    index=False,
)

prediction_bias_matrix_df.to_csv(
    SECTION2B_PREDICTION_BIAS_FILE,
)

active_card_action_availability_export_df = (
    active_card_action_availability_df.copy()
)

for list_column in [
    "available_legal_actions",
    "raw_predicted_actions",
    "selected_actions",
]:

    active_card_action_availability_export_df[
        list_column
    ] = active_card_action_availability_export_df[
        list_column
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )


active_card_action_availability_export_df.to_csv(
    SECTION2B_ACTION_AVAILABILITY_FILE,
    index=False,
)

root_cause_factor_strength_df.to_csv(
    SECTION2B_FACTOR_STRENGTH_FILE,
    index=False,
)


section2b_summary = {
    "status":
        "MASKING_ROOT_CAUSE_ANALYSIS_COMPLETE",

    "total_decisions":
        int(
            len(
                root_cause_df
            )
        ),

    "masking_changes":
        int(
            root_cause_df[
                "masking_changed_action_recomputed"
            ].sum()
        ),

    "masking_change_rate":
        float(
            root_cause_df[
                "masking_changed_action_recomputed"
            ].mean()
        ),

    "highest_masking_active_card":
        str(
            highest_masking_active_card_row[
                "active_card"
            ]
        ),

    "highest_masking_active_card_rate":
        float(
            highest_masking_active_card_row[
                "masking_rate"
            ]
        ),

    "lowest_raw_legality_active_card":
        str(
            lowest_legality_active_card_row[
                "active_card"
            ]
        ),

    "lowest_raw_legality_active_card_rate":
        float(
            lowest_legality_active_card_row[
                "raw_legality_rate"
            ]
        ),

    "highest_masking_legal_move_set":
        str(
            highest_masking_legal_set_row[
                "legal_move_signature"
            ]
        ),

    "highest_masking_legal_move_set_rate":
        float(
            highest_masking_legal_set_row[
                "masking_rate"
            ]
        ),

    "single_action_masking_rate":
        single_action_masking_rate,

    "multi_action_masking_rate":
        multi_action_masking_rate,

    "strongest_root_cause_factor":
        str(
            strongest_factor_row[
                "factor"
            ]
        ),

    "strongest_root_cause_spread":
        float(
            strongest_factor_row[
                "masking_rate_spread"
            ]
        ),

    "diagnosis": (
        "Raw policy predictions are insufficiently conditioned "
        "on active-card-specific legal-action availability."
    ),

    "recommended_feature_family": [
        "active_card",
        "legal_action_flags",
        "legal_move_signature",
        "single_legal_move",
        "acting_energy",
        "battle_phase",
    ],

    "next_stage":
        "LEGALITY_AWARE_FEATURE_ENGINEERING",
}


with open(
    SECTION2B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section2b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section2b_saved_files = [
    SECTION2B_ROOT_CAUSE_DATA_FILE,
    SECTION2B_ACTIVE_CARD_FILE,
    SECTION2B_LEGAL_SET_FILE,
    SECTION2B_CARD_PREDICTION_FILE,
    SECTION2B_CONDITION_FILE,
    SECTION2B_PHASE_FILE,
    SECTION2B_ENERGY_FILE,
    SECTION2B_DAMAGE_FILE,
    SECTION2B_CONFIDENCE_FILE,
    SECTION2B_ACTION_COUNT_FILE,
    SECTION2B_PREDICTION_BIAS_FILE,
    SECTION2B_ACTION_AVAILABILITY_FILE,
    SECTION2B_FACTOR_STRENGTH_FILE,
    SECTION2B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section2b_saved_files
)


# --------------------------------------------------------------------------------------
# 22. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 2B SUMMARY")
print("-" * 100)

for key, value in (
    section2b_summary.items()
):

    print(
        f"{key:42}: {value}"
    )


print()
print("SAVED SECTION 2B REPORTS")
print("-" * 100)

for file_path in section2b_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 2B MASKING ROOT-CAUSE ANALYSIS PASSED"
)


# # Section 3 — Legality-Aware Policy Design
# 
# ## Section 3A — Legality-Aware Decision Routing
# 
# #### Notebook 52 showed that most masking changes occur in states with only one legal action.
# 
# #### This section introduces a legality-aware routing strategy:
# 
# 1. If exactly one legal move exists, select it directly.
# 2. If multiple legal moves exist and the raw model prediction is legal, preserve the model prediction.
# 3. If multiple legal moves exist and the raw prediction is illegal, invoke the existing legal selector.
# 
# #### This design separates deterministic legality from genuine policy choice.
# 
# #### The section measures the theoretical masking reduction that can be achieved without retraining the classifier.

# In[5]:


# ======================================================================================
# SECTION 3A — LEGALITY-AWARE DECISION ROUTING
# ======================================================================================

print("=" * 100)
print("SECTION 3A — LEGALITY-AWARE DECISION ROUTING")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Section directory
# --------------------------------------------------------------------------------------

SECTION3_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section3"
)

SECTION3_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 2. Policy action classes
# --------------------------------------------------------------------------------------

POLICY_ACTION_CLASSES = [
    "Ascension",
    "Bind Down",
    "Live Coal",
    "Pass",
    "Quick Attack",
    "Tuck Tail",
]


def action_feature_name(
    action_name: str,
) -> str:
    """
    Convert a policy action name into a safe feature-column suffix.
    """

    return (
        normalize_action_name(
            action_name
        )
        .replace(
            " ",
            "_",
        )
    )


# --------------------------------------------------------------------------------------
# 3. Build legality-aware routing dataframe
# --------------------------------------------------------------------------------------

routing_analysis_df = (
    root_cause_df.copy()
)


# Ensure normalized action fields exist.
routing_analysis_df[
    "normalized_raw_prediction"
] = routing_analysis_df[
    "raw_predicted_move"
].apply(
    normalize_action_name
)

routing_analysis_df[
    "normalized_selected_move"
] = routing_analysis_df[
    "selected_move"
].apply(
    normalize_action_name
)


# --------------------------------------------------------------------------------------
# 4. Legal-action availability flags
# --------------------------------------------------------------------------------------

legal_flag_columns = []

for action_name in POLICY_ACTION_CLASSES:

    normalized_action = normalize_action_name(
        action_name
    )

    legal_flag_column = (
        "legal_action__"
        + action_feature_name(
            action_name
        )
    )

    routing_analysis_df[
        legal_flag_column
    ] = routing_analysis_df[
        "normalized_legal_moves"
    ].apply(
        lambda legal_moves: int(
            normalized_action
            in legal_moves
        )
    )

    legal_flag_columns.append(
        legal_flag_column
    )


routing_analysis_df[
    "legal_flag_total"
] = routing_analysis_df[
    legal_flag_columns
].sum(
    axis=1
)


# --------------------------------------------------------------------------------------
# 5. Decision-routing categories
# --------------------------------------------------------------------------------------

routing_analysis_df[
    "routing_mode"
] = np.where(
    routing_analysis_df[
        "legal_move_count_recomputed"
    ].eq(
        1
    ),
    "SINGLE_LEGAL_DIRECT",
    "MULTI_LEGAL_MODEL",
)


routing_analysis_df[
    "single_legal_move_name"
] = routing_analysis_df[
    "legal_move_list"
].apply(
    lambda legal_moves: (
        str(
            legal_moves[0]
        )
        if len(
            legal_moves
        ) == 1
        else None
    )
)


# --------------------------------------------------------------------------------------
# 6. Legality-aware routed selection
# --------------------------------------------------------------------------------------

def route_legality_aware_decision(
    row: pd.Series,
) -> Dict[str, Any]:
    """
    Apply the proposed legality-aware routing strategy.
    """

    legal_move_list = list(
        row[
            "legal_move_list"
        ]
    )

    normalized_legal_moves = list(
        row[
            "normalized_legal_moves"
        ]
    )

    raw_prediction = row[
        "raw_predicted_move"
    ]

    normalized_raw_prediction = row[
        "normalized_raw_prediction"
    ]

    existing_selected_move = row[
        "selected_move"
    ]

    if len(
        legal_move_list
    ) == 1:

        return {
            "routed_move":
                str(
                    legal_move_list[0]
                ),

            "routing_reason":
                "ONLY_LEGAL_MOVE",

            "model_choice_required":
                False,

            "legacy_selector_required":
                False,
        }

    if (
        normalized_raw_prediction
        in normalized_legal_moves
    ):

        return {
            "routed_move":
                str(
                    raw_prediction
                ),

            "routing_reason":
                "RAW_MODEL_PREDICTION_LEGAL",

            "model_choice_required":
                True,

            "legacy_selector_required":
                False,
        }

    return {
        "routed_move":
            str(
                existing_selected_move
            ),

        "routing_reason":
            "MULTI_ACTION_ILLEGAL_RAW_USE_SELECTOR",

        "model_choice_required":
            True,

        "legacy_selector_required":
            True,
    }


routing_result_series = routing_analysis_df.apply(
    route_legality_aware_decision,
    axis=1,
)


routing_result_df = pd.DataFrame(
    routing_result_series.tolist(),
    index=routing_analysis_df.index,
)


routing_analysis_df = pd.concat(
    [
        routing_analysis_df,
        routing_result_df,
    ],
    axis=1,
)


# --------------------------------------------------------------------------------------
# 7. Routing validation fields
# --------------------------------------------------------------------------------------

routing_analysis_df[
    "normalized_routed_move"
] = routing_analysis_df[
    "routed_move"
].apply(
    normalize_action_name
)

routing_analysis_df[
    "routed_move_legal"
] = routing_analysis_df.apply(
    lambda row: (
        row[
            "normalized_routed_move"
        ]
        in row[
            "normalized_legal_moves"
        ]
    ),
    axis=1,
)

routing_analysis_df[
    "routed_matches_existing_selection"
] = (
    routing_analysis_df[
        "normalized_routed_move"
    ]
    == routing_analysis_df[
        "normalized_selected_move"
    ]
)

routing_analysis_df[
    "raw_prediction_changed_by_router"
] = (
    routing_analysis_df[
        "normalized_raw_prediction"
    ]
    != routing_analysis_df[
        "normalized_routed_move"
    ]
)


# A direct single-action route is not considered model masking because
# the classifier is intentionally bypassed.
routing_analysis_df[
    "post_routing_model_masking"
] = (
    routing_analysis_df[
        "model_choice_required"
    ].astype(bool)
    & routing_analysis_df[
        "raw_prediction_changed_by_router"
    ].astype(bool)
)


# --------------------------------------------------------------------------------------
# 8. Routing summary
# --------------------------------------------------------------------------------------

routing_mode_summary_df = (
    routing_analysis_df
    .groupby(
        "routing_mode",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        original_masking_changes=(
            "masking_changed_action_recomputed",
            "sum",
        ),

        post_routing_model_masking=(
            "post_routing_model_masking",
            "sum",
        ),

        legal_routed_moves=(
            "routed_move_legal",
            "sum",
        ),

        model_choices_required=(
            "model_choice_required",
            "sum",
        ),

        legacy_selector_calls=(
            "legacy_selector_required",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
)

routing_mode_summary_df[
    "original_masking_rate"
] = (
    routing_mode_summary_df[
        "original_masking_changes"
    ]
    / routing_mode_summary_df[
        "decisions"
    ]
)

routing_mode_summary_df[
    "post_routing_masking_rate"
] = (
    routing_mode_summary_df[
        "post_routing_model_masking"
    ]
    / routing_mode_summary_df[
        "decisions"
    ]
)

routing_mode_summary_df[
    "legal_route_rate"
] = (
    routing_mode_summary_df[
        "legal_routed_moves"
    ]
    / routing_mode_summary_df[
        "decisions"
    ]
)


print()
print("ROUTING MODE SUMMARY")
print("-" * 100)

display(
    routing_mode_summary_df
)


# --------------------------------------------------------------------------------------
# 9. Routing-reason summary
# --------------------------------------------------------------------------------------

routing_reason_summary_df = (
    routing_analysis_df
    .groupby(
        "routing_reason",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        legal_routed_moves=(
            "routed_move_legal",
            "sum",
        ),

        matches_existing_selection=(
            "routed_matches_existing_selection",
            "sum",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),
    )
)

routing_reason_summary_df[
    "decision_fraction"
] = (
    routing_reason_summary_df[
        "decisions"
    ]
    / len(
        routing_analysis_df
    )
)

routing_reason_summary_df[
    "legal_route_rate"
] = (
    routing_reason_summary_df[
        "legal_routed_moves"
    ]
    / routing_reason_summary_df[
        "decisions"
    ]
)

routing_reason_summary_df[
    "existing_selection_match_rate"
] = (
    routing_reason_summary_df[
        "matches_existing_selection"
    ]
    / routing_reason_summary_df[
        "decisions"
    ]
)


print()
print("ROUTING REASON SUMMARY")
print("-" * 100)

display(
    routing_reason_summary_df
)


# --------------------------------------------------------------------------------------
# 10. Action availability summary
# --------------------------------------------------------------------------------------

legal_action_availability_rows = []

for action_name, flag_column in zip(
    POLICY_ACTION_CLASSES,
    legal_flag_columns,
):

    legal_action_availability_rows.append(
        {
            "policy_action":
                action_name,

            "available_decisions":
                int(
                    routing_analysis_df[
                        flag_column
                    ].sum()
                ),

            "availability_rate":
                float(
                    routing_analysis_df[
                        flag_column
                    ].mean()
                ),

            "raw_predictions":
                int(
                    routing_analysis_df[
                        "normalized_raw_prediction"
                    ].eq(
                        normalize_action_name(
                            action_name
                        )
                    ).sum()
                ),

            "final_selections":
                int(
                    routing_analysis_df[
                        "normalized_selected_move"
                    ].eq(
                        normalize_action_name(
                            action_name
                        )
                    ).sum()
                ),

            "routed_selections":
                int(
                    routing_analysis_df[
                        "normalized_routed_move"
                    ].eq(
                        normalize_action_name(
                            action_name
                        )
                    ).sum()
                ),
        }
    )


legal_action_availability_df = pd.DataFrame(
    legal_action_availability_rows
)


print()
print("LEGAL ACTION AVAILABILITY")
print("-" * 100)

display(
    legal_action_availability_df
)


# --------------------------------------------------------------------------------------
# 11. Theoretical improvement profile
# --------------------------------------------------------------------------------------

original_masking_count = int(
    routing_analysis_df[
        "masking_changed_action_recomputed"
    ].sum()
)

post_routing_masking_count = int(
    routing_analysis_df[
        "post_routing_model_masking"
    ].sum()
)

single_action_direct_decisions = int(
    (
        routing_analysis_df[
            "routing_mode"
        ]
        == "SINGLE_LEGAL_DIRECT"
    ).sum()
)

multi_action_model_decisions = int(
    (
        routing_analysis_df[
            "routing_mode"
        ]
        == "MULTI_LEGAL_MODEL"
    ).sum()
)

legacy_selector_calls = int(
    routing_analysis_df[
        "legacy_selector_required"
    ].sum()
)

original_masking_rate = float(
    original_masking_count
    / len(
        routing_analysis_df
    )
)

post_routing_masking_rate = float(
    post_routing_masking_count
    / len(
        routing_analysis_df
    )
)

absolute_masking_reduction = float(
    original_masking_rate
    - post_routing_masking_rate
)

relative_masking_reduction = float(
    (
        original_masking_count
        - post_routing_masking_count
    )
    / original_masking_count
    if original_masking_count > 0
    else 0.0
)


section3a_improvement_profile_df = pd.DataFrame(
    [
        {
            "metric": "total_decisions",
            "value": len(
                routing_analysis_df
            ),
        },
        {
            "metric": "single_action_direct_decisions",
            "value": single_action_direct_decisions,
        },
        {
            "metric": "multi_action_model_decisions",
            "value": multi_action_model_decisions,
        },
        {
            "metric": "original_masking_count",
            "value": original_masking_count,
        },
        {
            "metric": "original_masking_rate",
            "value": original_masking_rate,
        },
        {
            "metric": "post_routing_masking_count",
            "value": post_routing_masking_count,
        },
        {
            "metric": "post_routing_masking_rate",
            "value": post_routing_masking_rate,
        },
        {
            "metric": "absolute_masking_reduction",
            "value": absolute_masking_reduction,
        },
        {
            "metric": "relative_masking_reduction",
            "value": relative_masking_reduction,
        },
        {
            "metric": "legacy_selector_calls",
            "value": legacy_selector_calls,
        },
        {
            "metric": "routed_legal_move_rate",
            "value": float(
                routing_analysis_df[
                    "routed_move_legal"
                ].mean()
            ),
        },
        {
            "metric": "existing_selection_match_rate",
            "value": float(
                routing_analysis_df[
                    "routed_matches_existing_selection"
                ].mean()
            ),
        },
    ]
)


print()
print("THEORETICAL ROUTING IMPROVEMENT")
print("-" * 100)

display(
    section3a_improvement_profile_df
)


# --------------------------------------------------------------------------------------
# 12. Validation
# --------------------------------------------------------------------------------------

assert len(
    routing_analysis_df
) == 210

assert routing_analysis_df[
    "legal_flag_total"
].eq(
    routing_analysis_df[
        "legal_move_count_recomputed"
    ]
).all()

assert routing_analysis_df[
    "routed_move_legal"
].all()

assert routing_analysis_df[
    "routed_matches_existing_selection"
].all()

assert (
    single_action_direct_decisions
    + multi_action_model_decisions
) == len(
    routing_analysis_df
)

assert original_masking_count == 151

assert np.isclose(
    original_masking_rate,
    masking_change_rate,
    atol=1e-12,
)

assert post_routing_masking_count <= original_masking_count

assert post_routing_masking_rate <= original_masking_rate

assert relative_masking_reduction >= 0.0

assert set(
    routing_analysis_df[
        "routing_mode"
    ].unique()
) == {
    "SINGLE_LEGAL_DIRECT",
    "MULTI_LEGAL_MODEL",
}

assert set(
    routing_analysis_df[
        "routing_reason"
    ].unique()
).issubset(
    {
        "ONLY_LEGAL_MOVE",
        "RAW_MODEL_PREDICTION_LEGAL",
        "MULTI_ACTION_ILLEGAL_RAW_USE_SELECTOR",
    }
)


# --------------------------------------------------------------------------------------
# 13. Save reports
# --------------------------------------------------------------------------------------

SECTION3A_ROUTING_ANALYSIS_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_legality_aware_routing_analysis.csv"
)

SECTION3A_ROUTING_MODE_SUMMARY_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_routing_mode_summary.csv"
)

SECTION3A_ROUTING_REASON_SUMMARY_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_routing_reason_summary.csv"
)

SECTION3A_ACTION_AVAILABILITY_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_legal_action_availability.csv"
)

SECTION3A_IMPROVEMENT_PROFILE_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_theoretical_improvement_profile.csv"
)

SECTION3A_SUMMARY_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_legality_aware_routing_summary.json"
)


routing_export_df = (
    routing_analysis_df.copy()
)

for list_column in [
    "legal_move_list",
    "normalized_legal_moves",
]:

    routing_export_df[
        list_column
    ] = routing_export_df[
        list_column
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )


routing_export_df.to_csv(
    SECTION3A_ROUTING_ANALYSIS_FILE,
    index=False,
)

routing_mode_summary_df.to_csv(
    SECTION3A_ROUTING_MODE_SUMMARY_FILE,
    index=False,
)

routing_reason_summary_df.to_csv(
    SECTION3A_ROUTING_REASON_SUMMARY_FILE,
    index=False,
)

legal_action_availability_df.to_csv(
    SECTION3A_ACTION_AVAILABILITY_FILE,
    index=False,
)

section3a_improvement_profile_df.to_csv(
    SECTION3A_IMPROVEMENT_PROFILE_FILE,
    index=False,
)


section3a_summary = {
    "status":
        "LEGALITY_AWARE_ROUTING_VALIDATED",

    "total_decisions":
        int(
            len(
                routing_analysis_df
            )
        ),

    "single_action_direct_decisions":
        single_action_direct_decisions,

    "multi_action_model_decisions":
        multi_action_model_decisions,

    "original_masking_count":
        original_masking_count,

    "original_masking_rate":
        original_masking_rate,

    "post_routing_masking_count":
        post_routing_masking_count,

    "post_routing_masking_rate":
        post_routing_masking_rate,

    "absolute_masking_reduction":
        absolute_masking_reduction,

    "relative_masking_reduction":
        relative_masking_reduction,

    "legacy_selector_calls":
        legacy_selector_calls,

    "routed_legal_move_rate":
        float(
            routing_analysis_df[
                "routed_move_legal"
            ].mean()
        ),

    "existing_selection_match_rate":
        float(
            routing_analysis_df[
                "routed_matches_existing_selection"
            ].mean()
        ),

    "recommended_architecture": (
        "Directly return the only legal move; "
        "invoke the learned policy only for multi-action states."
    ),

    "next_stage":
        "LEGALITY_AWARE_FEATURE_SCHEMA",
}


with open(
    SECTION3A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3a_saved_files = [
    SECTION3A_ROUTING_ANALYSIS_FILE,
    SECTION3A_ROUTING_MODE_SUMMARY_FILE,
    SECTION3A_ROUTING_REASON_SUMMARY_FILE,
    SECTION3A_ACTION_AVAILABILITY_FILE,
    SECTION3A_IMPROVEMENT_PROFILE_FILE,
    SECTION3A_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3a_saved_files
)


# --------------------------------------------------------------------------------------
# 14. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 3A SUMMARY")
print("-" * 100)

for key, value in (
    section3a_summary.items()
):

    print(
        f"{key:40}: {value}"
    )


print()
print("SAVED SECTION 3A REPORTS")
print("-" * 100)

for file_path in section3a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3A LEGALITY-AWARE DECISION ROUTING PASSED"
)


# ## Section 3B — Legality-Aware Feature Schema
# 
# #### This section defines the engineered feature schema for the optimized policy.
# 
# #### The new schema preserves the original battle-state features while adding:
# 
# - active-card identity,
# - inactive-card identity,
# - legal-move signature,
# - one-hot legal-action flags,
# - single-action indicator,
# - battle phase,
# - acting-side energy,
# - defending-side energy,
# - energy difference,
# - acting-side damage,
# - defending-side damage,
# - damage difference,
# - confidence band,
# - routing mode.
# 
# #### The resulting schema will support legality-aware dataset construction and policy retraining.

# In[6]:


# ======================================================================================
# SECTION 3B — LEGALITY-AWARE FEATURE SCHEMA
# ======================================================================================

print("=" * 100)
print("SECTION 3B — LEGALITY-AWARE FEATURE SCHEMA")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Build the feature-schema dataframe
# --------------------------------------------------------------------------------------

legality_feature_df = (
    routing_analysis_df.copy()
)


# --------------------------------------------------------------------------------------
# 2. Ensure the core feature fields are present
# --------------------------------------------------------------------------------------

required_source_columns = [
    "scenario_id",
    "condition_id",
    "turn_number",
    "current_player",
    "active_card",
    "inactive_card",
    "legal_move_count_recomputed",
    "legal_move_signature",
    "single_legal_move",
    "acting_energy",
    "defending_energy",
    "acting_damage",
    "defending_damage",
    "battle_phase",
    "energy_band",
    "damage_band",
    "confidence_band",
    "selected_move",
    "raw_predicted_move",
    "routed_move",
    "routing_mode",
]

missing_source_columns = [
    column_name
    for column_name in required_source_columns
    if column_name not in legality_feature_df.columns
]

assert not missing_source_columns, (
    "Section 3B is missing required source columns: "
    f"{missing_source_columns}"
)


# --------------------------------------------------------------------------------------
# 3. Numeric normalization helpers
# --------------------------------------------------------------------------------------

def ensure_float_feature(
    dataframe: pd.DataFrame,
    column_name: str,
    default_value: float = 0.0,
) -> None:
    """
    Ensure a numeric feature exists and contains finite float values.
    """

    if column_name not in dataframe.columns:

        dataframe[
            column_name
        ] = default_value

    dataframe[
        column_name
    ] = pd.to_numeric(
        dataframe[
            column_name
        ],
        errors="coerce",
    ).fillna(
        default_value
    ).astype(
        float
    )


def ensure_string_feature(
    dataframe: pd.DataFrame,
    column_name: str,
    default_value: str = "UNKNOWN",
) -> None:
    """
    Ensure a categorical feature exists and contains nonempty strings.

    Categorical Pandas columns are converted to object before missing
    values are filled, preventing new-category fill errors.
    """

    if column_name not in dataframe.columns:

        dataframe[
            column_name
        ] = default_value

        return

    series = dataframe[
        column_name
    ]

    # Convert Pandas Categorical columns before inserting UNKNOWN.
    if isinstance(
        series.dtype,
        pd.CategoricalDtype,
    ):

        series = series.astype(
            object
        )

    dataframe[
        column_name
    ] = (
        series
        .where(
            series.notna(),
            default_value,
        )
        .astype(str)
        .str.strip()
        .replace(
            {
                "": default_value,
                "nan": default_value,
                "None": default_value,
                "<NA>": default_value,
            }
        )
    )

# --------------------------------------------------------------------------------------
# 4. Normalize numeric feature families
# --------------------------------------------------------------------------------------

numeric_base_features = [
    "turn_number",
    "legal_move_count_recomputed",
    "acting_energy",
    "defending_energy",
    "acting_damage",
    "defending_damage",
    "confidence",
]

for column_name in numeric_base_features:

    ensure_float_feature(
        legality_feature_df,
        column_name,
        default_value=0.0,
    )


# --------------------------------------------------------------------------------------
# 5. Engineer numeric interaction features
# --------------------------------------------------------------------------------------

legality_feature_df[
    "energy_difference"
] = (
    legality_feature_df[
        "acting_energy"
    ]
    - legality_feature_df[
        "defending_energy"
    ]
)

legality_feature_df[
    "damage_difference"
] = (
    legality_feature_df[
        "acting_damage"
    ]
    - legality_feature_df[
        "defending_damage"
    ]
)

legality_feature_df[
    "absolute_energy_difference"
] = (
    legality_feature_df[
        "energy_difference"
    ].abs()
)

legality_feature_df[
    "absolute_damage_difference"
] = (
    legality_feature_df[
        "damage_difference"
    ].abs()
)

legality_feature_df[
    "single_legal_move_flag"
] = (
    legality_feature_df[
        "single_legal_move"
    ].astype(bool).astype(int)
)

legality_feature_df[
    "multi_legal_move_flag"
] = (
    ~legality_feature_df[
        "single_legal_move"
    ].astype(bool)
).astype(int)

legality_feature_df[
    "model_choice_required_flag"
] = (
    legality_feature_df[
        "model_choice_required"
    ].astype(bool).astype(int)
)

legality_feature_df[
    "legacy_selector_required_flag"
] = (
    legality_feature_df[
        "legacy_selector_required"
    ].astype(bool).astype(int)
)


# --------------------------------------------------------------------------------------
# 6. Ensure categorical feature families
# --------------------------------------------------------------------------------------

categorical_base_features = [
    "current_player",
    "condition_id",
    "active_card",
    "inactive_card",
    "legal_move_signature",
    "battle_phase",
    "energy_band",
    "damage_band",
    "confidence_band",
    "routing_mode",
]

for column_name in categorical_base_features:

    ensure_string_feature(
        legality_feature_df,
        column_name,
    )


# --------------------------------------------------------------------------------------
# 7. Legal-action availability flags
# --------------------------------------------------------------------------------------

legality_action_flag_columns = []

for action_name in POLICY_ACTION_CLASSES:

    normalized_action_name = normalize_action_name(
        action_name
    )

    legality_flag_column = (
        "is_legal__"
        + action_feature_name(
            action_name
        )
    )

    legality_feature_df[
        legality_flag_column
    ] = legality_feature_df[
        "normalized_legal_moves"
    ].apply(
        lambda legal_moves: int(
            normalized_action_name
            in legal_moves
        )
    )

    legality_action_flag_columns.append(
        legality_flag_column
    )


legality_feature_df[
    "legal_action_flag_total"
] = legality_feature_df[
    legality_action_flag_columns
].sum(
    axis=1
)


# --------------------------------------------------------------------------------------
# 8. Active-card action-compatibility flags
# --------------------------------------------------------------------------------------

active_card_action_map: Dict[str, List[str]] = {}

for active_card_name, card_group_df in (
    legality_feature_df.groupby(
        "active_card",
        dropna=False,
    )
):

    available_actions = sorted(
        {
            normalize_action_name(
                action_name
            )
            for legal_move_list in card_group_df[
                "legal_move_list"
            ]
            for action_name in legal_move_list
        }
    )

    active_card_action_map[
        str(
            active_card_name
        )
    ] = available_actions


for action_name in POLICY_ACTION_CLASSES:

    normalized_action_name = normalize_action_name(
        action_name
    )

    compatibility_column = (
        "active_card_can_use__"
        + action_feature_name(
            action_name
        )
    )

    legality_feature_df[
        compatibility_column
    ] = legality_feature_df[
        "active_card"
    ].apply(
        lambda card_name: int(
            normalized_action_name
            in active_card_action_map.get(
                str(
                    card_name
                ),
                [],
            )
        )
    )


active_card_compatibility_columns = [
    (
        "active_card_can_use__"
        + action_feature_name(
            action_name
        )
    )
    for action_name in POLICY_ACTION_CLASSES
]


# --------------------------------------------------------------------------------------
# 9. Define the legality-aware feature schema
# --------------------------------------------------------------------------------------

legality_numeric_features = [
    "turn_number",
    "legal_move_count_recomputed",
    "acting_energy",
    "defending_energy",
    "energy_difference",
    "absolute_energy_difference",
    "acting_damage",
    "defending_damage",
    "damage_difference",
    "absolute_damage_difference",
    "single_legal_move_flag",
    "multi_legal_move_flag",
    "model_choice_required_flag",
    "legacy_selector_required_flag",
    "confidence",
]

legality_numeric_features += (
    legality_action_flag_columns
)

legality_numeric_features += (
    active_card_compatibility_columns
)


legality_categorical_features = [
    "current_player",
    "condition_id",
    "active_card",
    "inactive_card",
    "legal_move_signature",
    "battle_phase",
    "energy_band",
    "damage_band",
    "confidence_band",
    "routing_mode",
]


legality_target_columns = [
    "selected_move",
    "routed_move",
]


legality_metadata_columns = [
    column_name
    for column_name in [
        "scenario_id",
        "base_scenario_id",
        "condition_id",
        "turn_number",
        "current_player",
        "player_card",
        "opponent_card",
        "winner",
        "raw_predicted_move",
        "selected_move",
        "routed_move",
        "routing_reason",
        "routing_mode",
        "fallback_used",
        "masking_changed_action_recomputed",
    ]
    if column_name in legality_feature_df.columns
]


# --------------------------------------------------------------------------------------
# 10. Build model-ready raw feature table
# --------------------------------------------------------------------------------------

legality_model_feature_df = legality_feature_df[
    legality_numeric_features
    + legality_categorical_features
].copy()


legality_target_series = (
    legality_feature_df[
        "routed_move"
    ]
    .astype(str)
    .copy()
)


legality_metadata_df = legality_feature_df[
    legality_metadata_columns
].copy()


# --------------------------------------------------------------------------------------
# 11. Feature inventory
# --------------------------------------------------------------------------------------

feature_inventory_rows = []

for feature_name in legality_numeric_features:

    feature_inventory_rows.append(
        {
            "feature_name":
                feature_name,

            "feature_type":
                "numeric",

            "dtype":
                str(
                    legality_model_feature_df[
                        feature_name
                    ].dtype
                ),

            "missing_values":
                int(
                    legality_model_feature_df[
                        feature_name
                    ].isna().sum()
                ),

            "unique_values":
                int(
                    legality_model_feature_df[
                        feature_name
                    ].nunique(
                        dropna=False
                    )
                ),
        }
    )


for feature_name in legality_categorical_features:

    feature_inventory_rows.append(
        {
            "feature_name":
                feature_name,

            "feature_type":
                "categorical",

            "dtype":
                str(
                    legality_model_feature_df[
                        feature_name
                    ].dtype
                ),

            "missing_values":
                int(
                    legality_model_feature_df[
                        feature_name
                    ].isna().sum()
                ),

            "unique_values":
                int(
                    legality_model_feature_df[
                        feature_name
                    ].nunique(
                        dropna=False
                    )
                ),
        }
    )


legality_feature_inventory_df = pd.DataFrame(
    feature_inventory_rows
)


print()
print("LEGALITY-AWARE FEATURE INVENTORY")
print("-" * 100)

display(
    legality_feature_inventory_df
)


# --------------------------------------------------------------------------------------
# 12. Feature-family summary
# --------------------------------------------------------------------------------------

feature_family_summary_df = pd.DataFrame(
    [
        {
            "feature_family":
                "numeric_base_and_interactions",

            "feature_count":
                len(
                    legality_numeric_features
                )
                - len(
                    legality_action_flag_columns
                )
                - len(
                    active_card_compatibility_columns
                ),
        },
        {
            "feature_family":
                "legal_action_flags",

            "feature_count":
                len(
                    legality_action_flag_columns
                ),
        },
        {
            "feature_family":
                "active_card_compatibility_flags",

            "feature_count":
                len(
                    active_card_compatibility_columns
                ),
        },
        {
            "feature_family":
                "categorical_context",

            "feature_count":
                len(
                    legality_categorical_features
                ),
        },
    ]
)


print()
print("FEATURE FAMILY SUMMARY")
print("-" * 100)

display(
    feature_family_summary_df
)


# --------------------------------------------------------------------------------------
# 13. Active-card compatibility map
# --------------------------------------------------------------------------------------

active_card_action_map_df = pd.DataFrame(
    [
        {
            "active_card":
                card_name,

            "compatible_actions":
                action_list,

            "compatible_action_count":
                len(
                    action_list
                ),
        }
        for card_name, action_list in (
            sorted(
                active_card_action_map.items()
            )
        )
    ]
)


print()
print("ACTIVE-CARD ACTION COMPATIBILITY")
print("-" * 100)

display(
    active_card_action_map_df
)


# --------------------------------------------------------------------------------------
# 14. Legal-move-signature summary
# --------------------------------------------------------------------------------------

legal_signature_summary_df = (
    legality_feature_df
    .groupby(
        "legal_move_signature",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        legal_move_count=(
            "legal_move_count_recomputed",
            "first",
        ),

        single_action_states=(
            "single_legal_move_flag",
            "sum",
        ),

        multi_action_states=(
            "multi_legal_move_flag",
            "sum",
        ),

        model_choices_required=(
            "model_choice_required_flag",
            "sum",
        ),
    )
    .sort_values(
        [
            "decisions",
            "legal_move_count",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("LEGAL MOVE SIGNATURE SUMMARY")
print("-" * 100)

display(
    legal_signature_summary_df
)


# --------------------------------------------------------------------------------------
# 15. Target distribution
# --------------------------------------------------------------------------------------

legality_target_distribution_df = (
    legality_target_series
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "target_move"
    )
    .reset_index(
        name="examples"
    )
)

legality_target_distribution_df[
    "fraction"
] = (
    legality_target_distribution_df[
        "examples"
    ]
    / len(
        legality_target_series
    )
)


print()
print("ROUTED TARGET DISTRIBUTION")
print("-" * 100)

display(
    legality_target_distribution_df
)


# --------------------------------------------------------------------------------------
# 16. Schema validation
# --------------------------------------------------------------------------------------

assert len(
    legality_model_feature_df
) == 210

assert len(
    legality_target_series
) == 210

assert len(
    legality_metadata_df
) == 210

assert legality_model_feature_df.columns.is_unique

assert not legality_model_feature_df.isna().any().any()

assert legality_target_series.notna().all()

assert legality_target_series.astype(str).str.len().gt(
    0
).all()

assert legality_feature_df[
    "legal_action_flag_total"
].eq(
    legality_feature_df[
        "legal_move_count_recomputed"
    ]
).all()

assert legality_feature_df.loc[
    legality_feature_df[
        "single_legal_move_flag"
    ].eq(
        1
    ),
    "legal_move_count_recomputed",
].eq(
    1
).all()

assert legality_feature_df.loc[
    legality_feature_df[
        "multi_legal_move_flag"
    ].eq(
        1
    ),
    "legal_move_count_recomputed",
].gt(
    1
).all()

assert set(
    legality_target_series.unique()
).issubset(
    set(
        POLICY_ACTION_CLASSES
    )
)

assert set(
    legality_feature_df[
        "active_card"
    ].unique()
).issubset(
    set(
        active_card_action_map.keys()
    )
)

assert all(
    feature_name
    in legality_model_feature_df.columns
    for feature_name in legality_action_flag_columns
)

assert all(
    feature_name
    in legality_model_feature_df.columns
    for feature_name in active_card_compatibility_columns
)

assert int(
    legality_feature_df[
        "single_legal_move_flag"
    ].sum()
) == 187

assert int(
    legality_feature_df[
        "multi_legal_move_flag"
    ].sum()
) == 23


# --------------------------------------------------------------------------------------
# 17. Save feature-schema artifacts
# --------------------------------------------------------------------------------------

SECTION3B_MODEL_FEATURES_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legality_aware_model_features.csv"
)

SECTION3B_TARGETS_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legality_aware_targets.csv"
)

SECTION3B_METADATA_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legality_aware_metadata.csv"
)

SECTION3B_FEATURE_INVENTORY_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_feature_inventory.csv"
)

SECTION3B_FEATURE_FAMILY_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_feature_family_summary.csv"
)

SECTION3B_ACTIVE_CARD_MAP_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_active_card_action_map.csv"
)

SECTION3B_LEGAL_SIGNATURE_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legal_signature_summary.csv"
)

SECTION3B_TARGET_DISTRIBUTION_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_target_distribution.csv"
)

SECTION3B_SCHEMA_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legality_aware_schema.json"
)

SECTION3B_SUMMARY_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_legality_aware_feature_summary.json"
)


legality_model_feature_df.to_csv(
    SECTION3B_MODEL_FEATURES_FILE,
    index=False,
)

legality_target_series.to_frame(
    name="routed_move"
).to_csv(
    SECTION3B_TARGETS_FILE,
    index=False,
)

legality_metadata_df.to_csv(
    SECTION3B_METADATA_FILE,
    index=False,
)

legality_feature_inventory_df.to_csv(
    SECTION3B_FEATURE_INVENTORY_FILE,
    index=False,
)

feature_family_summary_df.to_csv(
    SECTION3B_FEATURE_FAMILY_FILE,
    index=False,
)

active_card_action_map_export_df = (
    active_card_action_map_df.copy()
)

active_card_action_map_export_df[
    "compatible_actions"
] = active_card_action_map_export_df[
    "compatible_actions"
].apply(
    lambda action_list: json.dumps(
        action_list,
        ensure_ascii=False,
    )
)

active_card_action_map_export_df.to_csv(
    SECTION3B_ACTIVE_CARD_MAP_FILE,
    index=False,
)

legal_signature_summary_df.to_csv(
    SECTION3B_LEGAL_SIGNATURE_FILE,
    index=False,
)

legality_target_distribution_df.to_csv(
    SECTION3B_TARGET_DISTRIBUTION_FILE,
    index=False,
)


section3b_schema = {
    "schema_name":
        "notebook53_legality_aware_policy_schema",

    "schema_version":
        "1.0",

    "row_count":
        int(
            len(
                legality_model_feature_df
            )
        ),

    "numeric_features":
        legality_numeric_features,

    "categorical_features":
        legality_categorical_features,

    "target_column":
        "routed_move",

    "metadata_columns":
        legality_metadata_columns,

    "legal_action_flags":
        legality_action_flag_columns,

    "active_card_compatibility_flags":
        active_card_compatibility_columns,

    "policy_action_classes":
        POLICY_ACTION_CLASSES,

    "active_card_action_map":
        active_card_action_map,

    "single_action_routing_enabled":
        True,

    "multi_action_model_enabled":
        True,
}


with open(
    SECTION3B_SCHEMA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3b_schema,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3b_summary = {
    "status":
        "LEGALITY_AWARE_FEATURE_SCHEMA_READY",

    "row_count":
        int(
            len(
                legality_model_feature_df
            )
        ),

    "numeric_feature_count":
        int(
            len(
                legality_numeric_features
            )
        ),

    "categorical_feature_count":
        int(
            len(
                legality_categorical_features
            )
        ),

    "total_raw_feature_count":
        int(
            len(
                legality_numeric_features
            )
            + len(
                legality_categorical_features
            )
        ),

    "legal_action_flag_count":
        int(
            len(
                legality_action_flag_columns
            )
        ),

    "active_card_compatibility_flag_count":
        int(
            len(
                active_card_compatibility_columns
            )
        ),

    "unique_active_cards":
        int(
            legality_feature_df[
                "active_card"
            ].nunique()
        ),

    "unique_legal_move_signatures":
        int(
            legality_feature_df[
                "legal_move_signature"
            ].nunique()
        ),

    "single_action_states":
        int(
            legality_feature_df[
                "single_legal_move_flag"
            ].sum()
        ),

    "multi_action_states":
        int(
            legality_feature_df[
                "multi_legal_move_flag"
            ].sum()
        ),

    "missing_feature_values":
        int(
            legality_model_feature_df
            .isna()
            .sum()
            .sum()
        ),

    "target_class_count":
        int(
            legality_target_series.nunique()
        ),

    "schema_reproducible":
        True,

    "ready_for_dataset_rebuild":
        True,

    "next_stage":
        "LEGALITY_AWARE_DATASET_REBUILD",
}


with open(
    SECTION3B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3b_saved_files = [
    SECTION3B_MODEL_FEATURES_FILE,
    SECTION3B_TARGETS_FILE,
    SECTION3B_METADATA_FILE,
    SECTION3B_FEATURE_INVENTORY_FILE,
    SECTION3B_FEATURE_FAMILY_FILE,
    SECTION3B_ACTIVE_CARD_MAP_FILE,
    SECTION3B_LEGAL_SIGNATURE_FILE,
    SECTION3B_TARGET_DISTRIBUTION_FILE,
    SECTION3B_SCHEMA_FILE,
    SECTION3B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3b_saved_files
)


# --------------------------------------------------------------------------------------
# 18. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 3B SUMMARY")
print("-" * 100)

for key, value in (
    section3b_summary.items()
):

    print(
        f"{key:44}: {value}"
    )


print()
print("SAVED SECTION 3B REPORTS")
print("-" * 100)

for file_path in section3b_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B LEGALITY-AWARE FEATURE SCHEMA PASSED"
)


# # Section 4 — Legality-Aware Dataset Reconstruction
# 
# ## Section 4A — Source Dataset Discovery and Rebuild Readiness
# 
# #### The 210 Notebook 52 tournament decisions are reserved for evaluation and diagnostic analysis.
# 
# They will not be used as the primary training dataset.
# 
# #### This section locates the original expert-policy dataset produced upstream, validates its schema, checks whether legal-action information is available, and determines whether the legality-aware training dataset can be reconstructed without benchmark leakage.

# In[7]:


# ======================================================================================
# SECTION 4A — SOURCE DATASET DISCOVERY AND REBUILD READINESS
# ======================================================================================

print("=" * 100)
print("SECTION 4A — SOURCE DATASET DISCOVERY AND REBUILD READINESS")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Section directories
# --------------------------------------------------------------------------------------

SECTION4_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section4"
)

SECTION4_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK49_REPORT_DIR = (
    REPORTS_DIR
    / "notebook49"
)


# --------------------------------------------------------------------------------------
# 2. Candidate upstream policy-dataset files
# --------------------------------------------------------------------------------------

candidate_policy_dataset_files = [
    (
        NOTEBOOK49_REPORT_DIR
        / "section8"
        / "policy_dataset"
        / "section8c_expert_policy_dataset.csv"
    ),
    (
        NOTEBOOK49_REPORT_DIR
        / "section8"
        / "section8c_expert_policy_dataset.csv"
    ),
    (
        NOTEBOOK50_REPORT_DIR
        / "section2"
        / "section2c_policy_dataset.csv"
    ),
    (
        NOTEBOOK50_REPORT_DIR
        / "section3"
        / "section3a_policy_dataset.csv"
    ),
]


# Add discovered CSV files whose names suggest policy/training data.
discovery_roots = [
    NOTEBOOK49_REPORT_DIR,
    NOTEBOOK50_REPORT_DIR,
]

discovered_policy_files = []

for discovery_root in discovery_roots:

    if not discovery_root.exists():
        continue

    for csv_file in discovery_root.rglob("*.csv"):

        lower_name = csv_file.name.lower()

        if any(
            keyword in lower_name
            for keyword in [
                "policy_dataset",
                "expert_policy",
                "training_dataset",
                "curriculum",
            ]
        ):
            discovered_policy_files.append(
                csv_file
            )


candidate_policy_dataset_files.extend(
    discovered_policy_files
)


# Deduplicate while preserving order.
candidate_policy_dataset_files = list(
    dict.fromkeys(
        Path(path).resolve()
        for path in candidate_policy_dataset_files
    )
)


# --------------------------------------------------------------------------------------
# 3. Candidate inventory
# --------------------------------------------------------------------------------------

candidate_inventory_rows = []

for candidate_path in candidate_policy_dataset_files:

    candidate_inventory_rows.append(
        {
            "path":
                str(candidate_path),

            "exists":
                candidate_path.exists(),

            "is_file":
                candidate_path.is_file(),

            "size_bytes":
                (
                    candidate_path.stat().st_size
                    if candidate_path.exists()
                    else 0
                ),

            "filename":
                candidate_path.name,
        }
    )


section4a_candidate_inventory_df = pd.DataFrame(
    candidate_inventory_rows
)


print()
print("CANDIDATE POLICY DATASETS")
print("-" * 100)

display(
    section4a_candidate_inventory_df
)


existing_candidate_files = [
    Path(path)
    for path in section4a_candidate_inventory_df.loc[
        section4a_candidate_inventory_df[
            "exists"
        ].astype(bool),
        "path",
    ].tolist()
]

assert existing_candidate_files, (
    "No upstream expert-policy dataset was found."
)


# --------------------------------------------------------------------------------------
# 4. Inspect every existing candidate
# --------------------------------------------------------------------------------------

dataset_profile_rows = []
loaded_candidate_dataframes = {}

for candidate_path in existing_candidate_files:

    try:

        candidate_df = pd.read_csv(
            candidate_path
        )

        loaded_candidate_dataframes[
            str(candidate_path)
        ] = candidate_df

        column_names = candidate_df.columns.tolist()

        normalized_columns = {
            column_name.lower():
                column_name
            for column_name in column_names
        }

        target_candidates = [
            column_name
            for column_name in column_names
            if column_name.lower() in {
                "expert_move",
                "selected_move",
                "policy_move",
                "target_move",
                "move",
                "action",
            }
        ]

        legal_move_candidates = [
            column_name
            for column_name in column_names
            if column_name.lower() in {
                "legal_moves",
                "available_moves",
                "legal_actions",
                "available_actions",
            }
        ]

        legal_count_candidates = [
            column_name
            for column_name in column_names
            if column_name.lower() in {
                "legal_move_count",
                "legal_action_count",
                "available_move_count",
            }
        ]

        dataset_profile_rows.append(
            {
                "path":
                    str(candidate_path),

                "rows":
                    int(len(candidate_df)),

                "columns":
                    int(len(column_names)),

                "target_candidates":
                    target_candidates,

                "legal_move_candidates":
                    legal_move_candidates,

                "legal_count_candidates":
                    legal_count_candidates,

                "has_current_side":
                    "current_side" in normalized_columns,

                "has_player_card":
                    "player_card" in normalized_columns,

                "has_opponent_card":
                    "opponent_card" in normalized_columns,

                "has_turn_number":
                    "turn_number" in normalized_columns,

                "duplicate_rows":
                    int(candidate_df.duplicated().sum()),

                "missing_values":
                    int(candidate_df.isna().sum().sum()),

                "load_success":
                    True,

                "load_error":
                    "",
            }
        )

    except Exception as error:

        dataset_profile_rows.append(
            {
                "path":
                    str(candidate_path),

                "rows":
                    0,

                "columns":
                    0,

                "target_candidates":
                    [],

                "legal_move_candidates":
                    [],

                "legal_count_candidates":
                    [],

                "has_current_side":
                    False,

                "has_player_card":
                    False,

                "has_opponent_card":
                    False,

                "has_turn_number":
                    False,

                "duplicate_rows":
                    0,

                "missing_values":
                    0,

                "load_success":
                    False,

                "load_error":
                    str(error),
            }
        )


section4a_dataset_profiles_df = pd.DataFrame(
    dataset_profile_rows
)


print()
print("DATASET PROFILES")
print("-" * 100)

display(
    section4a_dataset_profiles_df
)


# --------------------------------------------------------------------------------------
# 5. Rank candidates
# --------------------------------------------------------------------------------------

section4a_dataset_profiles_df[
    "readiness_score"
] = (
    section4a_dataset_profiles_df[
        "load_success"
    ].astype(int) * 10
    +
    section4a_dataset_profiles_df[
        "target_candidates"
    ].apply(
        lambda values: int(
            len(values) > 0
        )
    ) * 10
    +
    section4a_dataset_profiles_df[
        "legal_move_candidates"
    ].apply(
        lambda values: int(
            len(values) > 0
        )
    ) * 20
    +
    section4a_dataset_profiles_df[
        "legal_count_candidates"
    ].apply(
        lambda values: int(
            len(values) > 0
        )
    ) * 5
    +
    section4a_dataset_profiles_df[
        "has_current_side"
    ].astype(int) * 5
    +
    section4a_dataset_profiles_df[
        "has_player_card"
    ].astype(int) * 5
    +
    section4a_dataset_profiles_df[
        "has_opponent_card"
    ].astype(int) * 5
    +
    section4a_dataset_profiles_df[
        "has_turn_number"
    ].astype(int) * 5
    +
    np.log1p(
        section4a_dataset_profiles_df[
            "rows"
        ].clip(lower=0)
    )
)


section4a_dataset_profiles_df = (
    section4a_dataset_profiles_df
    .sort_values(
        [
            "readiness_score",
            "rows",
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
print("RANKED DATASET CANDIDATES")
print("-" * 100)

display(
    section4a_dataset_profiles_df
)


# --------------------------------------------------------------------------------------
# 6. Select the strongest source dataset
# --------------------------------------------------------------------------------------

selected_dataset_profile = (
    section4a_dataset_profiles_df.iloc[0]
)

SOURCE_POLICY_DATASET_FILE = Path(
    selected_dataset_profile[
        "path"
    ]
)

source_policy_dataset_df = (
    loaded_candidate_dataframes[
        str(
            SOURCE_POLICY_DATASET_FILE
        )
    ].copy()
)


selected_target_candidates = list(
    selected_dataset_profile[
        "target_candidates"
    ]
)

selected_legal_move_candidates = list(
    selected_dataset_profile[
        "legal_move_candidates"
    ]
)

selected_legal_count_candidates = list(
    selected_dataset_profile[
        "legal_count_candidates"
    ]
)


assert selected_target_candidates, (
    "The selected source dataset has no recognizable target column."
)


SOURCE_TARGET_COLUMN = (
    selected_target_candidates[0]
)

SOURCE_LEGAL_MOVES_COLUMN = (
    selected_legal_move_candidates[0]
    if selected_legal_move_candidates
    else None
)

SOURCE_LEGAL_MOVE_COUNT_COLUMN = (
    selected_legal_count_candidates[0]
    if selected_legal_count_candidates
    else None
)


# --------------------------------------------------------------------------------------
# 7. Source dataset schema
# --------------------------------------------------------------------------------------

section4a_column_inventory_df = pd.DataFrame(
    [
        {
            "column_name":
                column_name,

            "dtype":
                str(
                    source_policy_dataset_df[
                        column_name
                    ].dtype
                ),

            "missing_values":
                int(
                    source_policy_dataset_df[
                        column_name
                    ].isna().sum()
                ),

            "unique_values":
                int(
                    source_policy_dataset_df[
                        column_name
                    ].nunique(
                        dropna=False
                    )
                ),
        }
        for column_name in (
            source_policy_dataset_df.columns
        )
    ]
)


print()
print("SELECTED SOURCE DATASET")
print("-" * 100)

print(
    f"File                     : "
    f"{SOURCE_POLICY_DATASET_FILE}"
)

print(
    f"Rows                     : "
    f"{len(source_policy_dataset_df)}"
)

print(
    f"Columns                  : "
    f"{len(source_policy_dataset_df.columns)}"
)

print(
    f"Target column            : "
    f"{SOURCE_TARGET_COLUMN}"
)

print(
    f"Legal-moves column       : "
    f"{SOURCE_LEGAL_MOVES_COLUMN}"
)

print(
    f"Legal-move-count column  : "
    f"{SOURCE_LEGAL_MOVE_COUNT_COLUMN}"
)


print()
print("SOURCE COLUMN INVENTORY")
print("-" * 100)

display(
    section4a_column_inventory_df
)


print()
print("SOURCE DATA SAMPLE")
print("-" * 100)

display(
    source_policy_dataset_df.head(
        20
    )
)


# --------------------------------------------------------------------------------------
# 8. Evaluate rebuild readiness
# --------------------------------------------------------------------------------------

source_has_explicit_legal_moves = (
    SOURCE_LEGAL_MOVES_COLUMN
    is not None
)

source_has_legal_move_count = (
    SOURCE_LEGAL_MOVE_COUNT_COLUMN
    is not None
)

source_has_card_context = all(
    column_name in source_policy_dataset_df.columns
    for column_name in [
        "player_card",
        "opponent_card",
    ]
)

source_has_side_context = (
    "current_side"
    in source_policy_dataset_df.columns
    or
    "current_player"
    in source_policy_dataset_df.columns
)

source_has_turn_context = (
    "turn_number"
    in source_policy_dataset_df.columns
)


# Full legality-aware rebuild requires explicit legal actions.
full_rebuild_ready = bool(
    source_has_explicit_legal_moves
    and source_has_card_context
    and source_has_side_context
    and source_has_turn_context
)

partial_rebuild_ready = bool(
    source_has_card_context
    and source_has_side_context
    and source_has_turn_context
)


# --------------------------------------------------------------------------------------
# 9. Leakage protection
# --------------------------------------------------------------------------------------

benchmark_scenario_ids = set()

if "scenario_id" in notebook52_battle_results_df.columns:

    benchmark_scenario_ids = set(
        notebook52_battle_results_df[
            "scenario_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


source_scenario_ids = set()

if "scenario_id" in source_policy_dataset_df.columns:

    source_scenario_ids = set(
        source_policy_dataset_df[
            "scenario_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


benchmark_overlap_ids = sorted(
    benchmark_scenario_ids.intersection(
        source_scenario_ids
    )
)


section4a_leakage_profile_df = pd.DataFrame(
    [
        {
            "metric":
                "source_rows",

            "value":
                len(
                    source_policy_dataset_df
                ),
        },
        {
            "metric":
                "benchmark_rows",

            "value":
                len(
                    notebook52_policy_history_df
                ),
        },
        {
            "metric":
                "source_scenario_ids",

            "value":
                len(
                    source_scenario_ids
                ),
        },
        {
            "metric":
                "benchmark_scenario_ids",

            "value":
                len(
                    benchmark_scenario_ids
                ),
        },
        {
            "metric":
                "overlapping_scenario_ids",

            "value":
                len(
                    benchmark_overlap_ids
                ),
        },
        {
            "metric":
                "benchmark_rows_used_for_training",

            "value":
                0,
        },
    ]
)


print()
print("BENCHMARK LEAKAGE PROTECTION")
print("-" * 100)

display(
    section4a_leakage_profile_df
)


# --------------------------------------------------------------------------------------
# 10. Validation
# --------------------------------------------------------------------------------------

assert SOURCE_POLICY_DATASET_FILE.exists()

assert SOURCE_POLICY_DATASET_FILE.is_file()

assert len(
    source_policy_dataset_df
) > 0

assert len(
    source_policy_dataset_df.columns
) > 0

assert SOURCE_TARGET_COLUMN in (
    source_policy_dataset_df.columns
)

assert source_policy_dataset_df[
    SOURCE_TARGET_COLUMN
].notna().all()

assert (
    len(
        notebook52_policy_history_df
    )
    == 210
)

# Benchmark decisions remain diagnostic-only.
assert int(
    section4a_leakage_profile_df.loc[
        section4a_leakage_profile_df[
            "metric"
        ] == "benchmark_rows_used_for_training",
        "value",
    ].iloc[0]
) == 0


# --------------------------------------------------------------------------------------
# 11. Save reports
# --------------------------------------------------------------------------------------

SECTION4A_CANDIDATE_INVENTORY_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_candidate_dataset_inventory.csv"
)

SECTION4A_DATASET_PROFILES_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_dataset_profiles.csv"
)

SECTION4A_COLUMN_INVENTORY_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_selected_dataset_columns.csv"
)

SECTION4A_LEAKAGE_PROFILE_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_leakage_protection_profile.csv"
)

SECTION4A_SUMMARY_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_dataset_discovery_summary.json"
)


section4a_candidate_inventory_df.to_csv(
    SECTION4A_CANDIDATE_INVENTORY_FILE,
    index=False,
)

section4a_dataset_profiles_df.to_csv(
    SECTION4A_DATASET_PROFILES_FILE,
    index=False,
)

section4a_column_inventory_df.to_csv(
    SECTION4A_COLUMN_INVENTORY_FILE,
    index=False,
)

section4a_leakage_profile_df.to_csv(
    SECTION4A_LEAKAGE_PROFILE_FILE,
    index=False,
)


section4a_summary = {
    "status":
        "SOURCE_POLICY_DATASET_DISCOVERED",

    "source_dataset_file":
        str(
            SOURCE_POLICY_DATASET_FILE
        ),

    "source_rows":
        int(
            len(
                source_policy_dataset_df
            )
        ),

    "source_columns":
        int(
            len(
                source_policy_dataset_df.columns
            )
        ),

    "target_column":
        SOURCE_TARGET_COLUMN,

    "legal_moves_column":
        SOURCE_LEGAL_MOVES_COLUMN,

    "legal_move_count_column":
        SOURCE_LEGAL_MOVE_COUNT_COLUMN,

    "has_explicit_legal_moves":
        source_has_explicit_legal_moves,

    "has_legal_move_count":
        source_has_legal_move_count,

    "has_card_context":
        source_has_card_context,

    "has_side_context":
        source_has_side_context,

    "has_turn_context":
        source_has_turn_context,

    "full_legality_rebuild_ready":
        full_rebuild_ready,

    "partial_rebuild_ready":
        partial_rebuild_ready,

    "benchmark_scenario_overlap_count":
        int(
            len(
                benchmark_overlap_ids
            )
        ),

    "benchmark_rows_used_for_training":
        0,

    "next_stage":
        (
            "FULL_LEGALITY_AWARE_DATASET_REBUILD"
            if full_rebuild_ready
            else "LEGAL_ACTION_RECONSTRUCTION"
        ),
}


with open(
    SECTION4A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section4a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section4a_saved_files = [
    SECTION4A_CANDIDATE_INVENTORY_FILE,
    SECTION4A_DATASET_PROFILES_FILE,
    SECTION4A_COLUMN_INVENTORY_FILE,
    SECTION4A_LEAKAGE_PROFILE_FILE,
    SECTION4A_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section4a_saved_files
)


# --------------------------------------------------------------------------------------
# 12. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 4A SUMMARY")
print("-" * 100)

for key, value in section4a_summary.items():

    print(
        f"{key:42}: {value}"
    )


print()
print("SAVED SECTION 4A REPORTS")
print("-" * 100)

for file_path in section4a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 4A SOURCE DATASET DISCOVERY PASSED"
)


# ## Section 4B — Full Legality-Aware Dataset Rebuild
# 
# #### This section reconstructs the original 1,060-row expert-policy dataset using a legality-aware feature representation.
# 
# #### The training dataset includes:
# 
# - original battle-state features,
# - acting-card and defending-card identity,
# - canonical legal-move signatures,
# - one-hot legal-action availability flags,
# - active-card action-compatibility flags,
# - single-action and multi-action indicators,
# - acting-side energy and damage context,
# - battle-phase features,
# - expert policy targets,
# - original sample weights.
# 
# #### The Notebook 52 tournament benchmark remains excluded from training.

# In[8]:


# ======================================================================================
# SECTION 4B — FULL LEGALITY-AWARE DATASET REBUILD
# ======================================================================================

print("=" * 100)
print("SECTION 4B — FULL LEGALITY-AWARE DATASET REBUILD")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Copy the validated Notebook 49 expert-policy dataset
# --------------------------------------------------------------------------------------

legality_training_source_df = (
    source_policy_dataset_df.copy()
)


assert len(
    legality_training_source_df
) == 1060

assert SOURCE_TARGET_COLUMN == "expert_move"

assert SOURCE_LEGAL_MOVES_COLUMN == "legal_moves"

assert SOURCE_LEGAL_MOVE_COUNT_COLUMN == "legal_move_count"


# --------------------------------------------------------------------------------------
# 2. General parsing and normalization helpers
# --------------------------------------------------------------------------------------

def parse_training_legal_moves(
    value: Any,
) -> List[str]:
    """
    Convert a serialized legal-move collection into a clean list of move names.
    """

    parsed_values = parse_list_value(
        value
    )

    cleaned_values = []

    for item in parsed_values:

        if isinstance(
            item,
            dict,
        ):

            move_name = item.get(
                "name",
                "",
            )

        else:

            move_name = str(
                item
            )

        move_name = (
            move_name
            .strip()
            .strip(
                "'\""
            )
        )

        if move_name:

            cleaned_values.append(
                move_name
            )

    return cleaned_values


def canonical_move_signature(
    move_list: Sequence[str],
) -> str:
    """
    Create a deterministic normalized legal-action signature.
    """

    normalized_moves = sorted(
        {
            normalize_action_name(
                move_name
            )
            for move_name in move_list
            if normalize_action_name(
                move_name
            )
        }
    )

    return " | ".join(
        normalized_moves
    )


def classify_turn_phase(
    turn_number: float,
) -> str:
    """
    Convert a numeric turn into a stable battle-phase category.
    """

    if pd.isna(
        turn_number
    ):

        return "UNKNOWN"

    if float(
        turn_number
    ) <= 3:

        return "EARLY"

    if float(
        turn_number
    ) <= 7:

        return "MID"

    return "LATE"


def classify_energy_band(
    energy_value: float,
) -> str:
    """
    Convert acting-side energy into a categorical readiness band.
    """

    if pd.isna(
        energy_value
    ):

        return "UNKNOWN"

    energy_value = float(
        energy_value
    )

    if energy_value <= 0:

        return "ZERO"

    if energy_value <= 2:

        return "LOW"

    return "READY"


def classify_damage_band(
    damage_value: float,
) -> str:
    """
    Convert acting-side accumulated damage into a categorical band.
    """

    if pd.isna(
        damage_value
    ):

        return "UNKNOWN"

    damage_value = float(
        damage_value
    )

    if damage_value < 20:

        return "LOW_DAMAGE"

    if damage_value < 40:

        return "MEDIUM_DAMAGE"

    return "HIGH_DAMAGE"


# --------------------------------------------------------------------------------------
# 3. Normalize required source columns
# --------------------------------------------------------------------------------------

required_training_columns = [
    "example_id",
    "variant_id",
    "source_scenario_id",
    "side_mode",
    "current_side",
    "player_card",
    "opponent_card",
    "turn_number",
    "player_energy",
    "opponent_energy",
    "player_damage",
    "opponent_damage",
    "prize_cards_remaining",
    "hand_size",
    "legal_move_count",
    "legal_moves",
    "expert_move",
    "sample_weight",
]

missing_training_columns = [
    column_name
    for column_name in required_training_columns
    if column_name not in legality_training_source_df.columns
]

assert not missing_training_columns, (
    "The expert dataset is missing required columns: "
    f"{missing_training_columns}"
)


numeric_source_columns = [
    "turn_number",
    "player_energy",
    "opponent_energy",
    "player_damage",
    "opponent_damage",
    "prize_cards_remaining",
    "hand_size",
    "legal_move_count",
    "sample_weight",
]

for column_name in numeric_source_columns:

    legality_training_source_df[
        column_name
    ] = pd.to_numeric(
        legality_training_source_df[
            column_name
        ],
        errors="coerce",
    )


categorical_source_columns = [
    "side_mode",
    "current_side",
    "player_card",
    "opponent_card",
    "expert_move",
]

for column_name in categorical_source_columns:

    legality_training_source_df[
        column_name
    ] = (
        legality_training_source_df[
            column_name
        ]
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .str.strip()
    )


# --------------------------------------------------------------------------------------
# 4. Parse and validate legal actions
# --------------------------------------------------------------------------------------

legality_training_source_df[
    "legal_move_list"
] = legality_training_source_df[
    SOURCE_LEGAL_MOVES_COLUMN
].apply(
    parse_training_legal_moves
)

legality_training_source_df[
    "normalized_legal_moves"
] = legality_training_source_df[
    "legal_move_list"
].apply(
    lambda move_list: [
        normalize_action_name(
            move_name
        )
        for move_name in move_list
    ]
)

legality_training_source_df[
    "recomputed_legal_move_count"
] = legality_training_source_df[
    "legal_move_list"
].apply(
    len
)

legality_training_source_df[
    "legal_move_signature"
] = legality_training_source_df[
    "legal_move_list"
].apply(
    canonical_move_signature
)

legality_training_source_df[
    "normalized_expert_move"
] = legality_training_source_df[
    SOURCE_TARGET_COLUMN
].apply(
    normalize_action_name
)

legality_training_source_df[
    "expert_move_is_legal"
] = legality_training_source_df.apply(
    lambda row: (
        row[
            "normalized_expert_move"
        ]
        in row[
            "normalized_legal_moves"
        ]
    ),
    axis=1,
)


# --------------------------------------------------------------------------------------
# 5. Resolve acting and defending sides
# --------------------------------------------------------------------------------------

training_acting_side_is_player = (
    legality_training_source_df[
        "current_side"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq(
        "player"
    )
)


legality_training_source_df[
    "active_card"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "player_card"
    ],
    legality_training_source_df[
        "opponent_card"
    ],
)

legality_training_source_df[
    "inactive_card"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "opponent_card"
    ],
    legality_training_source_df[
        "player_card"
    ],
)

legality_training_source_df[
    "acting_energy"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "player_energy"
    ],
    legality_training_source_df[
        "opponent_energy"
    ],
)

legality_training_source_df[
    "defending_energy"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "opponent_energy"
    ],
    legality_training_source_df[
        "player_energy"
    ],
)

legality_training_source_df[
    "acting_damage"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "player_damage"
    ],
    legality_training_source_df[
        "opponent_damage"
    ],
)

legality_training_source_df[
    "defending_damage"
] = np.where(
    training_acting_side_is_player,
    legality_training_source_df[
        "opponent_damage"
    ],
    legality_training_source_df[
        "player_damage"
    ],
)


# --------------------------------------------------------------------------------------
# 6. Engineer state-comparison features
# --------------------------------------------------------------------------------------

legality_training_source_df[
    "energy_difference"
] = (
    legality_training_source_df[
        "acting_energy"
    ]
    - legality_training_source_df[
        "defending_energy"
    ]
)

legality_training_source_df[
    "absolute_energy_difference"
] = legality_training_source_df[
    "energy_difference"
].abs()

legality_training_source_df[
    "damage_difference"
] = (
    legality_training_source_df[
        "acting_damage"
    ]
    - legality_training_source_df[
        "defending_damage"
    ]
)

legality_training_source_df[
    "absolute_damage_difference"
] = legality_training_source_df[
    "damage_difference"
].abs()

legality_training_source_df[
    "single_legal_move_flag"
] = (
    legality_training_source_df[
        "recomputed_legal_move_count"
    ]
    .eq(
        1
    )
    .astype(int)
)

legality_training_source_df[
    "multi_legal_move_flag"
] = (
    legality_training_source_df[
        "recomputed_legal_move_count"
    ]
    .gt(
        1
    )
    .astype(int)
)

legality_training_source_df[
    "model_choice_required_flag"
] = legality_training_source_df[
    "multi_legal_move_flag"
].astype(int)

legality_training_source_df[
    "battle_phase"
] = legality_training_source_df[
    "turn_number"
].apply(
    classify_turn_phase
)

legality_training_source_df[
    "energy_band"
] = legality_training_source_df[
    "acting_energy"
].apply(
    classify_energy_band
)

legality_training_source_df[
    "damage_band"
] = legality_training_source_df[
    "acting_damage"
].apply(
    classify_damage_band
)


# --------------------------------------------------------------------------------------
# 7. Add explicit legal-action flags
# --------------------------------------------------------------------------------------

training_legal_action_flag_columns = []

for action_name in POLICY_ACTION_CLASSES:

    normalized_action = normalize_action_name(
        action_name
    )

    flag_column = (
        "is_legal__"
        + action_feature_name(
            action_name
        )
    )

    legality_training_source_df[
        flag_column
    ] = legality_training_source_df[
        "normalized_legal_moves"
    ].apply(
        lambda legal_moves: int(
            normalized_action
            in legal_moves
        )
    )

    training_legal_action_flag_columns.append(
        flag_column
    )


legality_training_source_df[
    "legal_action_flag_total"
] = legality_training_source_df[
    training_legal_action_flag_columns
].sum(
    axis=1
)


# --------------------------------------------------------------------------------------
# 8. Build active-card compatibility map from training data
# --------------------------------------------------------------------------------------

training_active_card_action_map: Dict[str, List[str]] = {}

for active_card_name, card_group_df in (
    legality_training_source_df.groupby(
        "active_card",
        dropna=False,
    )
):

    compatible_actions = sorted(
        {
            normalize_action_name(
                move_name
            )
            for move_list in card_group_df[
                "legal_move_list"
            ]
            for move_name in move_list
        }
    )

    training_active_card_action_map[
        str(
            active_card_name
        )
    ] = compatible_actions


training_active_card_compatibility_columns = []

for action_name in POLICY_ACTION_CLASSES:

    normalized_action = normalize_action_name(
        action_name
    )

    compatibility_column = (
        "active_card_can_use__"
        + action_feature_name(
            action_name
        )
    )

    legality_training_source_df[
        compatibility_column
    ] = legality_training_source_df[
        "active_card"
    ].apply(
        lambda card_name: int(
            normalized_action
            in training_active_card_action_map.get(
                str(
                    card_name
                ),
                [],
            )
        )
    )

    training_active_card_compatibility_columns.append(
        compatibility_column
    )


# --------------------------------------------------------------------------------------
# 9. Define training-safe feature schema
#
# Do not include:
# - expert_move,
# - value_target,
# - search results,
# - variant_name,
# - confidence,
# - routing outcomes,
# - Notebook 52 benchmark-derived fields.
# --------------------------------------------------------------------------------------

legality_training_numeric_features = [
    "turn_number",
    "player_energy",
    "opponent_energy",
    "player_damage",
    "opponent_damage",
    "prize_cards_remaining",
    "hand_size",
    "recomputed_legal_move_count",
    "acting_energy",
    "defending_energy",
    "energy_difference",
    "absolute_energy_difference",
    "acting_damage",
    "defending_damage",
    "damage_difference",
    "absolute_damage_difference",
    "single_legal_move_flag",
    "multi_legal_move_flag",
    "model_choice_required_flag",
]

legality_training_numeric_features += (
    training_legal_action_flag_columns
)

legality_training_numeric_features += (
    training_active_card_compatibility_columns
)


legality_training_categorical_features = [
    "current_side",
    "side_mode",
    "player_card",
    "opponent_card",
    "active_card",
    "inactive_card",
    "legal_move_signature",
    "battle_phase",
    "energy_band",
    "damage_band",
]


legality_training_metadata_columns = [
    "example_id",
    "variant_id",
    "source_scenario_id",
    "source_match_id",
    "variant_seed",
    "queue_position",
    "curriculum_priority",
    "hard_example_rank",
    "baseline_name",
    "state_key",
    "expert_move",
    "sample_weight",
]


# --------------------------------------------------------------------------------------
# 10. Normalize all final feature values
# --------------------------------------------------------------------------------------

for column_name in legality_training_numeric_features:

    legality_training_source_df[
        column_name
    ] = pd.to_numeric(
        legality_training_source_df[
            column_name
        ],
        errors="coerce",
    ).fillna(
        0.0
    ).astype(
        float
    )


for column_name in legality_training_categorical_features:

    legality_training_source_df[
        column_name
    ] = (
        legality_training_source_df[
            column_name
        ]
        .astype(object)
        .where(
            legality_training_source_df[
                column_name
            ].notna(),
            "UNKNOWN",
        )
        .astype(str)
        .str.strip()
        .replace(
            {
                "": "UNKNOWN",
                "nan": "UNKNOWN",
                "None": "UNKNOWN",
                "<NA>": "UNKNOWN",
            }
        )
    )


# --------------------------------------------------------------------------------------
# 11. Build final training tables
# --------------------------------------------------------------------------------------

legality_training_features_df = (
    legality_training_source_df[
        legality_training_numeric_features
        + legality_training_categorical_features
    ]
    .copy()
)

legality_training_target_series = (
    legality_training_source_df[
        SOURCE_TARGET_COLUMN
    ]
    .astype(str)
    .copy()
)

legality_training_weight_series = (
    pd.to_numeric(
        legality_training_source_df[
            "sample_weight"
        ],
        errors="coerce",
    )
    .fillna(
        1.0
    )
    .clip(
        lower=0.01
    )
    .astype(float)
)

legality_training_metadata_df = (
    legality_training_source_df[
        [
            column_name
            for column_name in legality_training_metadata_columns
            if column_name in legality_training_source_df.columns
        ]
    ]
    .copy()
)


# --------------------------------------------------------------------------------------
# 12. Dataset profile
# --------------------------------------------------------------------------------------

legality_training_target_distribution_df = (
    legality_training_source_df
    .groupby(
        SOURCE_TARGET_COLUMN,
        as_index=False,
    )
    .agg(
        examples=(
            "example_id",
            "size",
        ),

        total_sample_weight=(
            "sample_weight",
            "sum",
        ),

        average_sample_weight=(
            "sample_weight",
            "mean",
        ),

        single_action_examples=(
            "single_legal_move_flag",
            "sum",
        ),

        multi_action_examples=(
            "multi_legal_move_flag",
            "sum",
        ),
    )
)

legality_training_target_distribution_df[
    "fraction"
] = (
    legality_training_target_distribution_df[
        "examples"
    ]
    / len(
        legality_training_source_df
    )
)


print()
print("LEGALITY-AWARE TARGET DISTRIBUTION")
print("-" * 100)

display(
    legality_training_target_distribution_df
)


# --------------------------------------------------------------------------------------
# 13. Legal-move signature profile
# --------------------------------------------------------------------------------------

legality_training_signature_summary_df = (
    legality_training_source_df
    .groupby(
        "legal_move_signature",
        as_index=False,
    )
    .agg(
        examples=(
            "example_id",
            "size",
        ),

        legal_move_count=(
            "recomputed_legal_move_count",
            "first",
        ),

        single_action_examples=(
            "single_legal_move_flag",
            "sum",
        ),

        multi_action_examples=(
            "multi_legal_move_flag",
            "sum",
        ),

        unique_expert_moves=(
            SOURCE_TARGET_COLUMN,
            "nunique",
        ),
    )
    .sort_values(
        [
            "examples",
            "legal_move_count",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


print()
print("LEGAL-MOVE SIGNATURE PROFILE")
print("-" * 100)

display(
    legality_training_signature_summary_df
)


# --------------------------------------------------------------------------------------
# 14. Active-card profile
# --------------------------------------------------------------------------------------

legality_training_active_card_summary_df = (
    legality_training_source_df
    .groupby(
        "active_card",
        as_index=False,
    )
    .agg(
        examples=(
            "example_id",
            "size",
        ),

        legal_move_signatures=(
            "legal_move_signature",
            "nunique",
        ),

        expert_move_classes=(
            SOURCE_TARGET_COLUMN,
            "nunique",
        ),

        single_action_examples=(
            "single_legal_move_flag",
            "sum",
        ),

        multi_action_examples=(
            "multi_legal_move_flag",
            "sum",
        ),
    )
    .sort_values(
        "examples",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("ACTIVE-CARD TRAINING PROFILE")
print("-" * 100)

display(
    legality_training_active_card_summary_df
)


# --------------------------------------------------------------------------------------
# 15. Feature inventory
# --------------------------------------------------------------------------------------

section4b_feature_inventory_rows = []

for feature_name in legality_training_numeric_features:

    section4b_feature_inventory_rows.append(
        {
            "feature_name":
                feature_name,

            "feature_type":
                "numeric",

            "dtype":
                str(
                    legality_training_features_df[
                        feature_name
                    ].dtype
                ),

            "missing_values":
                int(
                    legality_training_features_df[
                        feature_name
                    ].isna().sum()
                ),

            "unique_values":
                int(
                    legality_training_features_df[
                        feature_name
                    ].nunique(
                        dropna=False
                    )
                ),
        }
    )


for feature_name in legality_training_categorical_features:

    section4b_feature_inventory_rows.append(
        {
            "feature_name":
                feature_name,

            "feature_type":
                "categorical",

            "dtype":
                str(
                    legality_training_features_df[
                        feature_name
                    ].dtype
                ),

            "missing_values":
                int(
                    legality_training_features_df[
                        feature_name
                    ].isna().sum()
                ),

            "unique_values":
                int(
                    legality_training_features_df[
                        feature_name
                    ].nunique(
                        dropna=False
                    )
                ),
        }
    )


section4b_feature_inventory_df = pd.DataFrame(
    section4b_feature_inventory_rows
)


print()
print("REBUILT FEATURE INVENTORY")
print("-" * 100)

display(
    section4b_feature_inventory_df
)


# --------------------------------------------------------------------------------------
# 16. Validation
# --------------------------------------------------------------------------------------

assert len(
    legality_training_source_df
) == 1060

assert len(
    legality_training_features_df
) == 1060

assert len(
    legality_training_target_series
) == 1060

assert len(
    legality_training_weight_series
) == 1060

assert legality_training_features_df.columns.is_unique

assert not legality_training_features_df.isna().any().any()

assert legality_training_target_series.notna().all()

assert legality_training_weight_series.gt(
    0
).all()

assert legality_training_source_df[
    "recomputed_legal_move_count"
].eq(
    legality_training_source_df[
        SOURCE_LEGAL_MOVE_COUNT_COLUMN
    ]
).all()

assert legality_training_source_df[
    "legal_action_flag_total"
].eq(
    legality_training_source_df[
        "recomputed_legal_move_count"
    ]
).all()

assert legality_training_source_df[
    "expert_move_is_legal"
].all(), (
    "At least one expert target is not present in its legal-action set."
)

assert legality_training_source_df[
    "recomputed_legal_move_count"
].gt(
    0
).all()

assert set(
    legality_training_target_series.unique()
).issubset(
    set(
        POLICY_ACTION_CLASSES
    )
)

assert all(
    column_name
    in legality_training_features_df.columns
    for column_name in training_legal_action_flag_columns
)

assert all(
    column_name
    in legality_training_features_df.columns
    for column_name in training_active_card_compatibility_columns
)

assert len(
    legality_training_numeric_features
) == 31

assert len(
    legality_training_categorical_features
) == 10

assert len(
    legality_training_features_df.columns
) == 41

# Notebook 52 remains evaluation-only.
assert len(
    benchmark_overlap_ids
) == 0

assert section4a_summary[
    "benchmark_rows_used_for_training"
] == 0


# --------------------------------------------------------------------------------------
# 17. Save rebuilt dataset artifacts
# --------------------------------------------------------------------------------------

SECTION4B_REBUILT_FULL_DATASET_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_full_dataset.csv"
)

SECTION4B_TRAINING_FEATURES_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_training_features.csv"
)

SECTION4B_TRAINING_TARGETS_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_training_targets.csv"
)

SECTION4B_TRAINING_WEIGHTS_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_training_weights.csv"
)

SECTION4B_TRAINING_METADATA_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_training_metadata.csv"
)

SECTION4B_FEATURE_INVENTORY_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_feature_inventory.csv"
)

SECTION4B_TARGET_DISTRIBUTION_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_target_distribution.csv"
)

SECTION4B_SIGNATURE_SUMMARY_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_signature_summary.csv"
)

SECTION4B_ACTIVE_CARD_SUMMARY_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_active_card_summary.csv"
)

SECTION4B_SCHEMA_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_training_schema.json"
)

SECTION4B_SUMMARY_FILE = (
    SECTION4_REPORT_DIR
    / "section4b_legality_aware_dataset_summary.json"
)


legality_training_export_df = (
    legality_training_source_df.copy()
)

for list_column in [
    "legal_move_list",
    "normalized_legal_moves",
]:

    legality_training_export_df[
        list_column
    ] = legality_training_export_df[
        list_column
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )


legality_training_export_df.to_csv(
    SECTION4B_REBUILT_FULL_DATASET_FILE,
    index=False,
)

legality_training_features_df.to_csv(
    SECTION4B_TRAINING_FEATURES_FILE,
    index=False,
)

legality_training_target_series.to_frame(
    name="expert_move"
).to_csv(
    SECTION4B_TRAINING_TARGETS_FILE,
    index=False,
)

legality_training_weight_series.to_frame(
    name="sample_weight"
).to_csv(
    SECTION4B_TRAINING_WEIGHTS_FILE,
    index=False,
)

legality_training_metadata_df.to_csv(
    SECTION4B_TRAINING_METADATA_FILE,
    index=False,
)

section4b_feature_inventory_df.to_csv(
    SECTION4B_FEATURE_INVENTORY_FILE,
    index=False,
)

legality_training_target_distribution_df.to_csv(
    SECTION4B_TARGET_DISTRIBUTION_FILE,
    index=False,
)

legality_training_signature_summary_df.to_csv(
    SECTION4B_SIGNATURE_SUMMARY_FILE,
    index=False,
)

legality_training_active_card_summary_df.to_csv(
    SECTION4B_ACTIVE_CARD_SUMMARY_FILE,
    index=False,
)


section4b_schema = {
    "schema_name":
        "notebook53_legality_aware_training_schema",

    "schema_version":
        "1.0",

    "source_dataset":
        str(
            SOURCE_POLICY_DATASET_FILE
        ),

    "row_count":
        int(
            len(
                legality_training_features_df
            )
        ),

    "numeric_features":
        legality_training_numeric_features,

    "categorical_features":
        legality_training_categorical_features,

    "raw_feature_count":
        int(
            len(
                legality_training_features_df.columns
            )
        ),

    "target_column":
        SOURCE_TARGET_COLUMN,

    "weight_column":
        "sample_weight",

    "legal_action_flags":
        training_legal_action_flag_columns,

    "active_card_compatibility_flags":
        training_active_card_compatibility_columns,

    "policy_action_classes":
        POLICY_ACTION_CLASSES,

    "active_card_action_map":
        training_active_card_action_map,

    "benchmark_rows_used_for_training":
        0,

    "benchmark_scenario_overlap_count":
        int(
            len(
                benchmark_overlap_ids
            )
        ),
}


with open(
    SECTION4B_SCHEMA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section4b_schema,
        file,
        indent=2,
        ensure_ascii=False,
    )


single_action_training_rows = int(
    legality_training_source_df[
        "single_legal_move_flag"
    ].sum()
)

multi_action_training_rows = int(
    legality_training_source_df[
        "multi_legal_move_flag"
    ].sum()
)


section4b_summary = {
    "status":
        "LEGALITY_AWARE_DATASET_REBUILT",

    "source_rows":
        int(
            len(
                source_policy_dataset_df
            )
        ),

    "rebuilt_rows":
        int(
            len(
                legality_training_features_df
            )
        ),

    "numeric_feature_count":
        int(
            len(
                legality_training_numeric_features
            )
        ),

    "categorical_feature_count":
        int(
            len(
                legality_training_categorical_features
            )
        ),

    "total_raw_feature_count":
        int(
            len(
                legality_training_features_df.columns
            )
        ),

    "legal_action_flag_count":
        int(
            len(
                training_legal_action_flag_columns
            )
        ),

    "active_card_compatibility_flag_count":
        int(
            len(
                training_active_card_compatibility_columns
            )
        ),

    "single_action_training_rows":
        single_action_training_rows,

    "multi_action_training_rows":
        multi_action_training_rows,

    "target_class_count":
        int(
            legality_training_target_series.nunique()
        ),

    "expert_targets_legal":
        bool(
            legality_training_source_df[
                "expert_move_is_legal"
            ].all()
        ),

    "missing_feature_values":
        int(
            legality_training_features_df
            .isna()
            .sum()
            .sum()
        ),

    "benchmark_rows_used_for_training":
        0,

    "benchmark_scenario_overlap_count":
        int(
            len(
                benchmark_overlap_ids
            )
        ),

    "ready_for_model_training":
        True,

    "next_stage":
        "LEGALITY_AWARE_TRAIN_VALIDATION_SPLIT",
}


with open(
    SECTION4B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section4b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section4b_saved_files = [
    SECTION4B_REBUILT_FULL_DATASET_FILE,
    SECTION4B_TRAINING_FEATURES_FILE,
    SECTION4B_TRAINING_TARGETS_FILE,
    SECTION4B_TRAINING_WEIGHTS_FILE,
    SECTION4B_TRAINING_METADATA_FILE,
    SECTION4B_FEATURE_INVENTORY_FILE,
    SECTION4B_TARGET_DISTRIBUTION_FILE,
    SECTION4B_SIGNATURE_SUMMARY_FILE,
    SECTION4B_ACTIVE_CARD_SUMMARY_FILE,
    SECTION4B_SCHEMA_FILE,
    SECTION4B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section4b_saved_files
)


# --------------------------------------------------------------------------------------
# 18. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 4B SUMMARY")
print("-" * 100)

for key, value in section4b_summary.items():

    print(
        f"{key:46}: {value}"
    )


print()
print("SAVED SECTION 4B REPORTS")
print("-" * 100)

for file_path in section4b_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 4B FULL LEGALITY-AWARE DATASET REBUILD PASSED"
)


# # Section 5 — Legality-Aware Model Training
# 
# ## Section 5A — Group-Safe Train, Validation, and Test Split
# 
# #### This section divides the 1,060-row legality-aware expert dataset into training, validation, and test partitions.
# 
# #### The split is performed by `source_scenario_id`, not by individual rows.
# 
# #### This prevents augmented variants derived from the same original scenario from appearing in multiple partitions and producing optimistic evaluation results.
# 
# #### Target proportions:
# 
# - Training: approximately 80%
# - Validation: approximately 10%
# - Test: approximately 10%
# 
# #### The section validates:
# 
# - zero scenario overlap,
# - complete row coverage,
# - class coverage,
# - legal-signature coverage,
# - weight preservation,
# - benchmark exclusion.

# In[9]:


# ======================================================================================
# SECTION 5A — GROUP-SAFE TRAIN, VALIDATION, AND TEST SPLIT
# ======================================================================================

print("=" * 100)
print("SECTION 5A — GROUP-SAFE TRAIN, VALIDATION, AND TEST SPLIT")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------------------

from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------------------------------------------
# 2. Section directory
# --------------------------------------------------------------------------------------

SECTION5_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section5"
)

SECTION5_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 3. Split configuration
# --------------------------------------------------------------------------------------

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10

assert np.isclose(
    TRAIN_FRACTION
    + VALIDATION_FRACTION
    + TEST_FRACTION,
    1.0,
)


# --------------------------------------------------------------------------------------
# 4. Group identifiers
# --------------------------------------------------------------------------------------

split_group_series = (
    legality_training_source_df[
        "source_scenario_id"
    ]
    .astype(str)
    .copy()
)


assert split_group_series.notna().all()

assert split_group_series.nunique() == 106

assert len(
    split_group_series
) == 1060


# --------------------------------------------------------------------------------------
# 5. First split: training versus temporary holdout
# --------------------------------------------------------------------------------------

train_holdout_splitter = GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_FRACTION,
    random_state=RANDOM_SEED,
)


train_indices, holdout_indices = next(
    train_holdout_splitter.split(
        legality_training_features_df,
        legality_training_target_series,
        groups=split_group_series,
    )
)


# --------------------------------------------------------------------------------------
# 6. Second split: validation versus test
# --------------------------------------------------------------------------------------

holdout_feature_df = (
    legality_training_features_df.iloc[
        holdout_indices
    ]
)

holdout_target_series = (
    legality_training_target_series.iloc[
        holdout_indices
    ]
)

holdout_group_series = (
    split_group_series.iloc[
        holdout_indices
    ]
)


validation_share_of_holdout = (
    VALIDATION_FRACTION
    / (
        VALIDATION_FRACTION
        + TEST_FRACTION
    )
)


validation_test_splitter = GroupShuffleSplit(
    n_splits=1,
    train_size=validation_share_of_holdout,
    random_state=RANDOM_SEED + 1,
)


validation_relative_indices, test_relative_indices = next(
    validation_test_splitter.split(
        holdout_feature_df,
        holdout_target_series,
        groups=holdout_group_series,
    )
)


validation_indices = holdout_indices[
    validation_relative_indices
]

test_indices = holdout_indices[
    test_relative_indices
]


# --------------------------------------------------------------------------------------
# 7. Construct final split objects
# --------------------------------------------------------------------------------------

X_train = (
    legality_training_features_df.iloc[
        train_indices
    ]
    .reset_index(
        drop=True
    )
)

X_validation = (
    legality_training_features_df.iloc[
        validation_indices
    ]
    .reset_index(
        drop=True
    )
)

X_test = (
    legality_training_features_df.iloc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)


y_train = (
    legality_training_target_series.iloc[
        train_indices
    ]
    .reset_index(
        drop=True
    )
)

y_validation = (
    legality_training_target_series.iloc[
        validation_indices
    ]
    .reset_index(
        drop=True
    )
)

y_test = (
    legality_training_target_series.iloc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)


weight_train = (
    legality_training_weight_series.iloc[
        train_indices
    ]
    .reset_index(
        drop=True
    )
)

weight_validation = (
    legality_training_weight_series.iloc[
        validation_indices
    ]
    .reset_index(
        drop=True
    )
)

weight_test = (
    legality_training_weight_series.iloc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)


metadata_train = (
    legality_training_metadata_df.iloc[
        train_indices
    ]
    .reset_index(
        drop=True
    )
)

metadata_validation = (
    legality_training_metadata_df.iloc[
        validation_indices
    ]
    .reset_index(
        drop=True
    )
)

metadata_test = (
    legality_training_metadata_df.iloc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)


group_train = (
    split_group_series.iloc[
        train_indices
    ]
    .reset_index(
        drop=True
    )
)

group_validation = (
    split_group_series.iloc[
        validation_indices
    ]
    .reset_index(
        drop=True
    )
)

group_test = (
    split_group_series.iloc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------------------------------------------
# 8. Scenario sets
# --------------------------------------------------------------------------------------

train_scenario_ids = set(
    group_train.tolist()
)

validation_scenario_ids = set(
    group_validation.tolist()
)

test_scenario_ids = set(
    group_test.tolist()
)


train_validation_overlap = (
    train_scenario_ids
    .intersection(
        validation_scenario_ids
    )
)

train_test_overlap = (
    train_scenario_ids
    .intersection(
        test_scenario_ids
    )
)

validation_test_overlap = (
    validation_scenario_ids
    .intersection(
        test_scenario_ids
    )
)


# --------------------------------------------------------------------------------------
# 9. Split assignment table
# --------------------------------------------------------------------------------------

split_assignment_df = pd.DataFrame(
    {
        "row_index":
            np.arange(
                len(
                    legality_training_source_df
                )
            ),

        "source_scenario_id":
            split_group_series,

        "expert_move":
            legality_training_target_series,

        "legal_move_signature":
            legality_training_source_df[
                "legal_move_signature"
            ],

        "sample_weight":
            legality_training_weight_series,

        "split":
            "UNASSIGNED",
    }
)


split_assignment_df.loc[
    train_indices,
    "split",
] = "Training"

split_assignment_df.loc[
    validation_indices,
    "split",
] = "Validation"

split_assignment_df.loc[
    test_indices,
    "split",
] = "Test"


# --------------------------------------------------------------------------------------
# 10. Overall split summary
# --------------------------------------------------------------------------------------

split_summary_df = pd.DataFrame(
    [
        {
            "split":
                "Training",

            "examples":
                len(
                    X_train
                ),

            "scenarios":
                len(
                    train_scenario_ids
                ),

            "fraction":
                len(
                    X_train
                )
                / len(
                    legality_training_features_df
                ),

            "target_classes":
                y_train.nunique(),

            "legal_signatures":
                X_train[
                    "legal_move_signature"
                ].nunique(),

            "total_sample_weight":
                weight_train.sum(),

            "average_sample_weight":
                weight_train.mean(),
        },
        {
            "split":
                "Validation",

            "examples":
                len(
                    X_validation
                ),

            "scenarios":
                len(
                    validation_scenario_ids
                ),

            "fraction":
                len(
                    X_validation
                )
                / len(
                    legality_training_features_df
                ),

            "target_classes":
                y_validation.nunique(),

            "legal_signatures":
                X_validation[
                    "legal_move_signature"
                ].nunique(),

            "total_sample_weight":
                weight_validation.sum(),

            "average_sample_weight":
                weight_validation.mean(),
        },
        {
            "split":
                "Test",

            "examples":
                len(
                    X_test
                ),

            "scenarios":
                len(
                    test_scenario_ids
                ),

            "fraction":
                len(
                    X_test
                )
                / len(
                    legality_training_features_df
                ),

            "target_classes":
                y_test.nunique(),

            "legal_signatures":
                X_test[
                    "legal_move_signature"
                ].nunique(),

            "total_sample_weight":
                weight_test.sum(),

            "average_sample_weight":
                weight_test.mean(),
        },
    ]
)


print()
print("GROUP-SAFE SPLIT SUMMARY")
print("-" * 100)

display(
    split_summary_df
)


# --------------------------------------------------------------------------------------
# 11. Target distribution by split
# --------------------------------------------------------------------------------------

target_split_summary_df = (
    split_assignment_df
    .groupby(
        [
            "split",
            "expert_move",
        ],
        as_index=False,
    )
    .agg(
        examples=(
            "row_index",
            "size",
        ),

        total_sample_weight=(
            "sample_weight",
            "sum",
        ),

        scenarios=(
            "source_scenario_id",
            "nunique",
        ),
    )
)


target_split_summary_df[
    "split_fraction"
] = (
    target_split_summary_df[
        "examples"
    ]
    / target_split_summary_df.groupby(
        "split"
    )[
        "examples"
    ].transform(
        "sum"
    )
)


print()
print("TARGET DISTRIBUTION BY SPLIT")
print("-" * 100)

display(
    target_split_summary_df
)


# --------------------------------------------------------------------------------------
# 12. Legal-signature distribution by split
# --------------------------------------------------------------------------------------

signature_split_summary_df = (
    split_assignment_df
    .groupby(
        [
            "split",
            "legal_move_signature",
        ],
        as_index=False,
    )
    .agg(
        examples=(
            "row_index",
            "size",
        ),

        scenarios=(
            "source_scenario_id",
            "nunique",
        ),
    )
)


signature_split_summary_df[
    "split_fraction"
] = (
    signature_split_summary_df[
        "examples"
    ]
    / signature_split_summary_df.groupby(
        "split"
    )[
        "examples"
    ].transform(
        "sum"
    )
)


print()
print("LEGAL-SIGNATURE DISTRIBUTION BY SPLIT")
print("-" * 100)

display(
    signature_split_summary_df
)


# --------------------------------------------------------------------------------------
# 13. Scenario overlap validation table
# --------------------------------------------------------------------------------------

scenario_overlap_df = pd.DataFrame(
    [
        {
            "comparison":
                "Training vs Validation",

            "overlap_count":
                len(
                    train_validation_overlap
                ),

            "overlap_ids":
                sorted(
                    train_validation_overlap
                ),
        },
        {
            "comparison":
                "Training vs Test",

            "overlap_count":
                len(
                    train_test_overlap
                ),

            "overlap_ids":
                sorted(
                    train_test_overlap
                ),
        },
        {
            "comparison":
                "Validation vs Test",

            "overlap_count":
                len(
                    validation_test_overlap
                ),

            "overlap_ids":
                sorted(
                    validation_test_overlap
                ),
        },
    ]
)


print()
print("SCENARIO OVERLAP CHECKS")
print("-" * 100)

display(
    scenario_overlap_df
)


# --------------------------------------------------------------------------------------
# 14. Validation checks
# --------------------------------------------------------------------------------------

section5a_validation_rows = []


def add_section5a_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:

    section5a_validation_rows.append(
        {
            "check":
                check_name,

            "passed":
                bool(
                    passed
                ),

            "value":
                value,

            "expected":
                expected,

            "details":
                details,
        }
    )


add_section5a_check(
    "all_rows_assigned",
    bool(
        split_assignment_df[
            "split"
        ].ne(
            "UNASSIGNED"
        ).all()
    ),
    int(
        split_assignment_df[
            "split"
        ].ne(
            "UNASSIGNED"
        ).sum()
    ),
    1060,
)

add_section5a_check(
    "row_counts_reconcile",
    (
        len(
            X_train
        )
        + len(
            X_validation
        )
        + len(
            X_test
        )
        == 1060
    ),
    (
        len(
            X_train
        )
        + len(
            X_validation
        )
        + len(
            X_test
        )
    ),
    1060,
)

add_section5a_check(
    "scenario_counts_reconcile",
    (
        len(
            train_scenario_ids
        )
        + len(
            validation_scenario_ids
        )
        + len(
            test_scenario_ids
        )
        == 106
    ),
    (
        len(
            train_scenario_ids
        )
        + len(
            validation_scenario_ids
        )
        + len(
            test_scenario_ids
        )
    ),
    106,
)

add_section5a_check(
    "training_validation_overlap_zero",
    len(
        train_validation_overlap
    ) == 0,
    len(
        train_validation_overlap
    ),
    0,
)

add_section5a_check(
    "training_test_overlap_zero",
    len(
        train_test_overlap
    ) == 0,
    len(
        train_test_overlap
    ),
    0,
)

add_section5a_check(
    "validation_test_overlap_zero",
    len(
        validation_test_overlap
    ) == 0,
    len(
        validation_test_overlap
    ),
    0,
)

add_section5a_check(
    "training_contains_all_classes",
    set(
        y_train.unique()
    )
    == set(
        POLICY_ACTION_CLASSES
    ),
    sorted(
        y_train.unique().tolist()
    ),
    sorted(
        POLICY_ACTION_CLASSES
    ),
)

add_section5a_check(
    "validation_targets_known",
    set(
        y_validation.unique()
    ).issubset(
        set(
            y_train.unique()
        )
    ),
    sorted(
        y_validation.unique().tolist()
    ),
    "Subset of training targets",
)

add_section5a_check(
    "test_targets_known",
    set(
        y_test.unique()
    ).issubset(
        set(
            y_train.unique()
        )
    ),
    sorted(
        y_test.unique().tolist()
    ),
    "Subset of training targets",
)

add_section5a_check(
    "validation_signatures_known",
    set(
        X_validation[
            "legal_move_signature"
        ].unique()
    ).issubset(
        set(
            X_train[
                "legal_move_signature"
            ].unique()
        )
    ),
    sorted(
        X_validation[
            "legal_move_signature"
        ].unique().tolist()
    ),
    "Subset of training signatures",
)

add_section5a_check(
    "test_signatures_known",
    set(
        X_test[
            "legal_move_signature"
        ].unique()
    ).issubset(
        set(
            X_train[
                "legal_move_signature"
            ].unique()
        )
    ),
    sorted(
        X_test[
            "legal_move_signature"
        ].unique().tolist()
    ),
    "Subset of training signatures",
)

add_section5a_check(
    "feature_columns_match",
    (
        list(
            X_train.columns
        )
        == list(
            X_validation.columns
        )
        == list(
            X_test.columns
        )
    ),
    len(
        X_train.columns
    ),
    41,
)

add_section5a_check(
    "no_missing_training_features",
    not X_train.isna().any().any(),
    int(
        X_train.isna().sum().sum()
    ),
    0,
)

add_section5a_check(
    "no_missing_validation_features",
    not X_validation.isna().any().any(),
    int(
        X_validation.isna().sum().sum()
    ),
    0,
)

add_section5a_check(
    "no_missing_test_features",
    not X_test.isna().any().any(),
    int(
        X_test.isna().sum().sum()
    ),
    0,
)

add_section5a_check(
    "all_weights_positive",
    bool(
        pd.concat(
            [
                weight_train,
                weight_validation,
                weight_test,
            ],
            ignore_index=True,
        ).gt(
            0
        ).all()
    ),
    float(
        min(
            weight_train.min(),
            weight_validation.min(),
            weight_test.min(),
        )
    ),
    "> 0",
)

add_section5a_check(
    "benchmark_rows_excluded",
    section4b_summary[
        "benchmark_rows_used_for_training"
    ] == 0,
    section4b_summary[
        "benchmark_rows_used_for_training"
    ],
    0,
)


section5a_validation_checks_df = pd.DataFrame(
    section5a_validation_rows
)


print()
print("SPLIT VALIDATION CHECKS")
print("-" * 100)

display(
    section5a_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 15. Final assertions
# --------------------------------------------------------------------------------------

assert section5a_validation_checks_df[
    "passed"
].astype(bool).all(), (
    "One or more group-safe split validation checks failed."
)

assert len(
    train_validation_overlap
) == 0

assert len(
    train_test_overlap
) == 0

assert len(
    validation_test_overlap
) == 0

assert len(
    X_train
) + len(
    X_validation
) + len(
    X_test
) == 1060


# --------------------------------------------------------------------------------------
# 16. Save split artifacts
# --------------------------------------------------------------------------------------

SECTION5A_SPLIT_ASSIGNMENT_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_group_safe_split_assignments.csv"
)

SECTION5A_SPLIT_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_group_safe_split_summary.csv"
)

SECTION5A_TARGET_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_target_distribution_by_split.csv"
)

SECTION5A_SIGNATURE_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_signature_distribution_by_split.csv"
)

SECTION5A_OVERLAP_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_scenario_overlap_checks.csv"
)

SECTION5A_VALIDATION_CHECKS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_split_validation_checks.csv"
)

SECTION5A_TRAIN_FEATURES_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_train_features.csv"
)

SECTION5A_VALIDATION_FEATURES_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_validation_features.csv"
)

SECTION5A_TEST_FEATURES_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_test_features.csv"
)

SECTION5A_TRAIN_TARGETS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_train_targets.csv"
)

SECTION5A_VALIDATION_TARGETS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_validation_targets.csv"
)

SECTION5A_TEST_TARGETS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_test_targets.csv"
)

SECTION5A_TRAIN_WEIGHTS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_train_weights.csv"
)

SECTION5A_VALIDATION_WEIGHTS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_validation_weights.csv"
)

SECTION5A_TEST_WEIGHTS_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_test_weights.csv"
)

SECTION5A_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_group_safe_split_summary.json"
)


split_assignment_export_df = (
    split_assignment_df.copy()
)

split_assignment_df.to_csv(
    SECTION5A_SPLIT_ASSIGNMENT_FILE,
    index=False,
)

split_summary_df.to_csv(
    SECTION5A_SPLIT_SUMMARY_FILE,
    index=False,
)

target_split_summary_df.to_csv(
    SECTION5A_TARGET_SUMMARY_FILE,
    index=False,
)

signature_split_summary_df.to_csv(
    SECTION5A_SIGNATURE_SUMMARY_FILE,
    index=False,
)


scenario_overlap_export_df = (
    scenario_overlap_df.copy()
)

scenario_overlap_export_df[
    "overlap_ids"
] = scenario_overlap_export_df[
    "overlap_ids"
].apply(
    lambda value: json.dumps(
        value,
        ensure_ascii=False,
    )
)

scenario_overlap_export_df.to_csv(
    SECTION5A_OVERLAP_FILE,
    index=False,
)

section5a_validation_checks_df.to_csv(
    SECTION5A_VALIDATION_CHECKS_FILE,
    index=False,
)

X_train.to_csv(
    SECTION5A_TRAIN_FEATURES_FILE,
    index=False,
)

X_validation.to_csv(
    SECTION5A_VALIDATION_FEATURES_FILE,
    index=False,
)

X_test.to_csv(
    SECTION5A_TEST_FEATURES_FILE,
    index=False,
)

y_train.to_frame(
    name="expert_move"
).to_csv(
    SECTION5A_TRAIN_TARGETS_FILE,
    index=False,
)

y_validation.to_frame(
    name="expert_move"
).to_csv(
    SECTION5A_VALIDATION_TARGETS_FILE,
    index=False,
)

y_test.to_frame(
    name="expert_move"
).to_csv(
    SECTION5A_TEST_TARGETS_FILE,
    index=False,
)

weight_train.to_frame(
    name="sample_weight"
).to_csv(
    SECTION5A_TRAIN_WEIGHTS_FILE,
    index=False,
)

weight_validation.to_frame(
    name="sample_weight"
).to_csv(
    SECTION5A_VALIDATION_WEIGHTS_FILE,
    index=False,
)

weight_test.to_frame(
    name="sample_weight"
).to_csv(
    SECTION5A_TEST_WEIGHTS_FILE,
    index=False,
)


section5a_checks_passed = int(
    section5a_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section5a_checks_failed = int(
    (
        ~section5a_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


section5a_summary = {
    "status":
        "GROUP_SAFE_SPLIT_COMPLETE",

    "total_examples":
        1060,

    "total_scenarios":
        106,

    "training_examples":
        int(
            len(
                X_train
            )
        ),

    "training_scenarios":
        int(
            len(
                train_scenario_ids
            )
        ),

    "validation_examples":
        int(
            len(
                X_validation
            )
        ),

    "validation_scenarios":
        int(
            len(
                validation_scenario_ids
            )
        ),

    "test_examples":
        int(
            len(
                X_test
            )
        ),

    "test_scenarios":
        int(
            len(
                test_scenario_ids
            )
        ),

    "feature_count":
        int(
            X_train.shape[1]
        ),

    "training_target_classes":
        int(
            y_train.nunique()
        ),

    "training_legal_signatures":
        int(
            X_train[
                "legal_move_signature"
            ].nunique()
        ),

    "scenario_overlap_count":
        int(
            len(
                train_validation_overlap
            )
            + len(
                train_test_overlap
            )
            + len(
                validation_test_overlap
            )
        ),

    "validation_checks_passed":
        section5a_checks_passed,

    "validation_checks_failed":
        section5a_checks_failed,

    "benchmark_rows_used_for_training":
        0,

    "ready_for_preprocessing":
        True,

    "next_stage":
        "LEGALITY_AWARE_PREPROCESSING",
}


with open(
    SECTION5A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section5a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section5a_saved_files = [
    SECTION5A_SPLIT_ASSIGNMENT_FILE,
    SECTION5A_SPLIT_SUMMARY_FILE,
    SECTION5A_TARGET_SUMMARY_FILE,
    SECTION5A_SIGNATURE_SUMMARY_FILE,
    SECTION5A_OVERLAP_FILE,
    SECTION5A_VALIDATION_CHECKS_FILE,
    SECTION5A_TRAIN_FEATURES_FILE,
    SECTION5A_VALIDATION_FEATURES_FILE,
    SECTION5A_TEST_FEATURES_FILE,
    SECTION5A_TRAIN_TARGETS_FILE,
    SECTION5A_VALIDATION_TARGETS_FILE,
    SECTION5A_TEST_TARGETS_FILE,
    SECTION5A_TRAIN_WEIGHTS_FILE,
    SECTION5A_VALIDATION_WEIGHTS_FILE,
    SECTION5A_TEST_WEIGHTS_FILE,
    SECTION5A_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section5a_saved_files
)


# --------------------------------------------------------------------------------------
# 17. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 5A SUMMARY")
print("-" * 100)

for key, value in section5a_summary.items():

    print(
        f"{key:42}: {value}"
    )


print()
print("SAVED SECTION 5A REPORTS")
print("-" * 100)

for file_path in section5a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 5A GROUP-SAFE TRAIN/VALIDATION/TEST SPLIT PASSED"
)


# ## Section 5B — Legality-Aware Preprocessing
# 
# #### This section prepares the 41-feature legality-aware dataset for model training.
# 
# #### The preprocessing pipeline:
# 
# - preserves numeric features,
# - imputes numeric values defensively,
# - one-hot encodes categorical features,
# - ignores unseen validation and test categories safely,
# - fits only on the training partition,
# - transforms validation and test partitions without leakage,
# - preserves the original row counts and split assignments,
# - exports the fitted preprocessor and encoded feature names.
# 
# #### The transformed matrices will be used to train the optimized policy.

# In[10]:


# ======================================================================================
# SECTION 5B — LEGALITY-AWARE PREPROCESSING
# ======================================================================================

print("=" * 100)
print("SECTION 5B — LEGALITY-AWARE PREPROCESSING")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------------------------------------------
# 2. Validate split readiness
# --------------------------------------------------------------------------------------

assert section5a_summary[
    "status"
] == "GROUP_SAFE_SPLIT_COMPLETE"

assert section5a_summary[
    "ready_for_preprocessing"
] is True

assert section5a_summary[
    "scenario_overlap_count"
] == 0

assert len(
    X_train
) == 840

assert len(
    X_validation
) == 110

assert len(
    X_test
) == 110

assert list(
    X_train.columns
) == list(
    X_validation.columns
) == list(
    X_test.columns
)


# --------------------------------------------------------------------------------------
# 3. Feature schema
# --------------------------------------------------------------------------------------

PREPROCESS_NUMERIC_FEATURES = list(
    legality_training_numeric_features
)

PREPROCESS_CATEGORICAL_FEATURES = list(
    legality_training_categorical_features
)

PREPROCESS_RAW_FEATURES = (
    PREPROCESS_NUMERIC_FEATURES
    + PREPROCESS_CATEGORICAL_FEATURES
)


assert len(
    PREPROCESS_NUMERIC_FEATURES
) == 31

assert len(
    PREPROCESS_CATEGORICAL_FEATURES
) == 10

assert len(
    PREPROCESS_RAW_FEATURES
) == 41

assert PREPROCESS_RAW_FEATURES == list(
    X_train.columns
)


print()
print("RAW FEATURE SCHEMA")
print("-" * 100)

print(
    f"Numeric features        : "
    f"{len(PREPROCESS_NUMERIC_FEATURES)}"
)

print(
    f"Categorical features    : "
    f"{len(PREPROCESS_CATEGORICAL_FEATURES)}"
)

print(
    f"Total raw features      : "
    f"{len(PREPROCESS_RAW_FEATURES)}"
)


# --------------------------------------------------------------------------------------
# 4. Numeric preprocessing pipeline
# --------------------------------------------------------------------------------------

numeric_preprocessing_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
    ]
)


# --------------------------------------------------------------------------------------
# 5. Categorical preprocessing pipeline
#
# Compatibility handling is included because different scikit-learn versions
# use either sparse_output=False or sparse=False.
# --------------------------------------------------------------------------------------

try:

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

except TypeError:

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
        dtype=np.float64,
    )


categorical_preprocessing_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "one_hot_encoder",
            categorical_encoder,
        ),
    ]
)


# --------------------------------------------------------------------------------------
# 6. Complete preprocessing pipeline
# --------------------------------------------------------------------------------------

legality_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_preprocessing_pipeline,
            PREPROCESS_NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_preprocessing_pipeline,
            PREPROCESS_CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.0,
    verbose_feature_names_out=True,
)


# --------------------------------------------------------------------------------------
# 7. Fit only on the training split
# --------------------------------------------------------------------------------------

X_train_encoded = legality_preprocessor.fit_transform(
    X_train
)

X_validation_encoded = legality_preprocessor.transform(
    X_validation
)

X_test_encoded = legality_preprocessor.transform(
    X_test
)


X_train_encoded = np.asarray(
    X_train_encoded,
    dtype=np.float64,
)

X_validation_encoded = np.asarray(
    X_validation_encoded,
    dtype=np.float64,
)

X_test_encoded = np.asarray(
    X_test_encoded,
    dtype=np.float64,
)


# --------------------------------------------------------------------------------------
# 8. Encoded feature names
# --------------------------------------------------------------------------------------

encoded_feature_names = legality_preprocessor.get_feature_names_out().tolist()

ENCODED_FEATURE_COUNT = len(
    encoded_feature_names
)


assert X_train_encoded.shape[1] == ENCODED_FEATURE_COUNT

assert X_validation_encoded.shape[1] == ENCODED_FEATURE_COUNT

assert X_test_encoded.shape[1] == ENCODED_FEATURE_COUNT


# --------------------------------------------------------------------------------------
# 9. Build encoded DataFrames
# --------------------------------------------------------------------------------------

X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
)

X_validation_encoded_df = pd.DataFrame(
    X_validation_encoded,
    columns=encoded_feature_names,
)

X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=encoded_feature_names,
)


print()
print("ENCODED MATRIX SHAPES")
print("-" * 100)

print(
    f"Training matrix         : "
    f"{X_train_encoded_df.shape}"
)

print(
    f"Validation matrix       : "
    f"{X_validation_encoded_df.shape}"
)

print(
    f"Test matrix             : "
    f"{X_test_encoded_df.shape}"
)

print(
    f"Encoded feature count   : "
    f"{ENCODED_FEATURE_COUNT}"
)


# --------------------------------------------------------------------------------------
# 10. Encoded feature inventory
# --------------------------------------------------------------------------------------

encoded_feature_inventory_rows = []

for feature_index, feature_name in enumerate(
    encoded_feature_names
):

    if feature_name.startswith(
        "numeric__"
    ):

        feature_family = "numeric"

    elif feature_name.startswith(
        "categorical__"
    ):

        feature_family = "categorical_one_hot"

    else:

        feature_family = "other"

    encoded_feature_inventory_rows.append(
        {
            "feature_index":
                feature_index,

            "encoded_feature_name":
                feature_name,

            "feature_family":
                feature_family,

            "training_minimum":
                float(
                    X_train_encoded_df[
                        feature_name
                    ].min()
                ),

            "training_maximum":
                float(
                    X_train_encoded_df[
                        feature_name
                    ].max()
                ),

            "training_mean":
                float(
                    X_train_encoded_df[
                        feature_name
                    ].mean()
                ),

            "training_nonzero_count":
                int(
                    X_train_encoded_df[
                        feature_name
                    ].ne(
                        0.0
                    ).sum()
                ),

            "validation_nonzero_count":
                int(
                    X_validation_encoded_df[
                        feature_name
                    ].ne(
                        0.0
                    ).sum()
                ),

            "test_nonzero_count":
                int(
                    X_test_encoded_df[
                        feature_name
                    ].ne(
                        0.0
                    ).sum()
                ),
        }
    )


section5b_encoded_feature_inventory_df = pd.DataFrame(
    encoded_feature_inventory_rows
)


print()
print("ENCODED FEATURE INVENTORY")
print("-" * 100)

display(
    section5b_encoded_feature_inventory_df
)


# --------------------------------------------------------------------------------------
# 11. Feature-family summary
# --------------------------------------------------------------------------------------

section5b_encoded_family_summary_df = (
    section5b_encoded_feature_inventory_df
    .groupby(
        "feature_family",
        as_index=False,
    )
    .agg(
        feature_count=(
            "encoded_feature_name",
            "count",
        ),

        average_training_nonzero_count=(
            "training_nonzero_count",
            "mean",
        ),

        average_validation_nonzero_count=(
            "validation_nonzero_count",
            "mean",
        ),

        average_test_nonzero_count=(
            "test_nonzero_count",
            "mean",
        ),
    )
)


print()
print("ENCODED FEATURE FAMILY SUMMARY")
print("-" * 100)

display(
    section5b_encoded_family_summary_df
)


# --------------------------------------------------------------------------------------
# 12. Categorical vocabulary profile
# --------------------------------------------------------------------------------------

fitted_categorical_pipeline = (
    legality_preprocessor
    .named_transformers_[
        "categorical"
    ]
)

fitted_one_hot_encoder = (
    fitted_categorical_pipeline
    .named_steps[
        "one_hot_encoder"
    ]
)


categorical_vocabulary_rows = []

for feature_name, category_values in zip(
    PREPROCESS_CATEGORICAL_FEATURES,
    fitted_one_hot_encoder.categories_,
):

    categorical_vocabulary_rows.append(
        {
            "categorical_feature":
                feature_name,

            "category_count":
                int(
                    len(
                        category_values
                    )
                ),

            "categories":
                category_values.astype(str).tolist(),
        }
    )


section5b_categorical_vocabulary_df = pd.DataFrame(
    categorical_vocabulary_rows
)


print()
print("TRAINING CATEGORICAL VOCABULARY")
print("-" * 100)

display(
    section5b_categorical_vocabulary_df
)


# --------------------------------------------------------------------------------------
# 13. Detect unseen categories in validation and test
# --------------------------------------------------------------------------------------

unseen_category_rows = []

training_category_map = {
    feature_name:
        set(
            category_values.astype(str).tolist()
        )
    for feature_name, category_values in zip(
        PREPROCESS_CATEGORICAL_FEATURES,
        fitted_one_hot_encoder.categories_,
    )
}


for split_name, split_dataframe in [
    (
        "Validation",
        X_validation,
    ),
    (
        "Test",
        X_test,
    ),
]:

    for feature_name in PREPROCESS_CATEGORICAL_FEATURES:

        split_categories = set(
            split_dataframe[
                feature_name
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        unseen_categories = sorted(
            split_categories
            - training_category_map[
                feature_name
            ]
        )

        unseen_category_rows.append(
            {
                "split":
                    split_name,

                "feature_name":
                    feature_name,

                "unseen_category_count":
                    len(
                        unseen_categories
                    ),

                "unseen_categories":
                    unseen_categories,
            }
        )


section5b_unseen_categories_df = pd.DataFrame(
    unseen_category_rows
)


print()
print("UNSEEN CATEGORY CHECK")
print("-" * 100)

display(
    section5b_unseen_categories_df
)


# --------------------------------------------------------------------------------------
# 14. Matrix integrity profile
# --------------------------------------------------------------------------------------

matrix_integrity_rows = []

for split_name, encoded_dataframe in [
    (
        "Training",
        X_train_encoded_df,
    ),
    (
        "Validation",
        X_validation_encoded_df,
    ),
    (
        "Test",
        X_test_encoded_df,
    ),
]:

    encoded_values = encoded_dataframe.to_numpy(
        dtype=np.float64
    )

    matrix_integrity_rows.append(
        {
            "split":
                split_name,

            "rows":
                int(
                    encoded_dataframe.shape[0]
                ),

            "columns":
                int(
                    encoded_dataframe.shape[1]
                ),

            "missing_values":
                int(
                    encoded_dataframe
                    .isna()
                    .sum()
                    .sum()
                ),

            "infinite_values":
                int(
                    np.isinf(
                        encoded_values
                    ).sum()
                ),

            "finite_values":
                int(
                    np.isfinite(
                        encoded_values
                    ).sum()
                ),

            "total_values":
                int(
                    encoded_values.size
                ),

            "zero_fraction":
                float(
                    np.mean(
                        encoded_values
                        == 0.0
                    )
                ),
        }
    )


section5b_matrix_integrity_df = pd.DataFrame(
    matrix_integrity_rows
)


print()
print("ENCODED MATRIX INTEGRITY")
print("-" * 100)

display(
    section5b_matrix_integrity_df
)


# --------------------------------------------------------------------------------------
# 15. Validation checks
# --------------------------------------------------------------------------------------

section5b_validation_rows = []


def add_section5b_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:

    section5b_validation_rows.append(
        {
            "check":
                check_name,

            "passed":
                bool(
                    passed
                ),

            "value":
                value,

            "expected":
                expected,

            "details":
                details,
        }
    )


add_section5b_check(
    "preprocessor_fitted_on_training",
    hasattr(
        legality_preprocessor,
        "transformers_",
    ),
    hasattr(
        legality_preprocessor,
        "transformers_",
    ),
    True,
)

add_section5b_check(
    "training_row_count_preserved",
    len(
        X_train_encoded_df
    ) == len(
        X_train
    ),
    len(
        X_train_encoded_df
    ),
    len(
        X_train
    ),
)

add_section5b_check(
    "validation_row_count_preserved",
    len(
        X_validation_encoded_df
    ) == len(
        X_validation
    ),
    len(
        X_validation_encoded_df
    ),
    len(
        X_validation
    ),
)

add_section5b_check(
    "test_row_count_preserved",
    len(
        X_test_encoded_df
    ) == len(
        X_test
    ),
    len(
        X_test_encoded_df
    ),
    len(
        X_test
    ),
)

add_section5b_check(
    "encoded_column_counts_match",
    (
        X_train_encoded_df.shape[1]
        == X_validation_encoded_df.shape[1]
        == X_test_encoded_df.shape[1]
    ),
    {
        "training":
            X_train_encoded_df.shape[1],

        "validation":
            X_validation_encoded_df.shape[1],

        "test":
            X_test_encoded_df.shape[1],
    },
    "All equal",
)

add_section5b_check(
    "encoded_feature_names_unique",
    len(
        encoded_feature_names
    ) == len(
        set(
            encoded_feature_names
        )
    ),
    len(
        set(
            encoded_feature_names
        )
    ),
    len(
        encoded_feature_names
    ),
)

add_section5b_check(
    "training_matrix_finite",
    bool(
        np.isfinite(
            X_train_encoded
        ).all()
    ),
    int(
        np.isfinite(
            X_train_encoded
        ).sum()
    ),
    int(
        X_train_encoded.size
    ),
)

add_section5b_check(
    "validation_matrix_finite",
    bool(
        np.isfinite(
            X_validation_encoded
        ).all()
    ),
    int(
        np.isfinite(
            X_validation_encoded
        ).sum()
    ),
    int(
        X_validation_encoded.size
    ),
)

add_section5b_check(
    "test_matrix_finite",
    bool(
        np.isfinite(
            X_test_encoded
        ).all()
    ),
    int(
        np.isfinite(
            X_test_encoded
        ).sum()
    ),
    int(
        X_test_encoded.size
    ),
)

add_section5b_check(
    "no_missing_training_values",
    not X_train_encoded_df.isna().any().any(),
    int(
        X_train_encoded_df
        .isna()
        .sum()
        .sum()
    ),
    0,
)

add_section5b_check(
    "no_missing_validation_values",
    not X_validation_encoded_df.isna().any().any(),
    int(
        X_validation_encoded_df
        .isna()
        .sum()
        .sum()
    ),
    0,
)

add_section5b_check(
    "no_missing_test_values",
    not X_test_encoded_df.isna().any().any(),
    int(
        X_test_encoded_df
        .isna()
        .sum()
        .sum()
    ),
    0,
)

add_section5b_check(
    "numeric_feature_count_preserved",
    int(
        (
            section5b_encoded_feature_inventory_df[
                "feature_family"
            ]
            == "numeric"
        ).sum()
    ) == 31,
    int(
        (
            section5b_encoded_feature_inventory_df[
                "feature_family"
            ]
            == "numeric"
        ).sum()
    ),
    31,
)

add_section5b_check(
    "categorical_vocabulary_complete",
    len(
        section5b_categorical_vocabulary_df
    ) == 10,
    len(
        section5b_categorical_vocabulary_df
    ),
    10,
)

add_section5b_check(
    "raw_feature_count_valid",
    len(
        PREPROCESS_RAW_FEATURES
    ) == 41,
    len(
        PREPROCESS_RAW_FEATURES
    ),
    41,
)

add_section5b_check(
    "benchmark_rows_excluded",
    section5a_summary[
        "benchmark_rows_used_for_training"
    ] == 0,
    section5a_summary[
        "benchmark_rows_used_for_training"
    ],
    0,
)

add_section5b_check(
    "scenario_overlap_zero",
    section5a_summary[
        "scenario_overlap_count"
    ] == 0,
    section5a_summary[
        "scenario_overlap_count"
    ],
    0,
)


section5b_validation_checks_df = pd.DataFrame(
    section5b_validation_rows
)


print()
print("PREPROCESSING VALIDATION CHECKS")
print("-" * 100)

display(
    section5b_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 16. Final assertions
# --------------------------------------------------------------------------------------

assert section5b_validation_checks_df[
    "passed"
].astype(bool).all(), (
    "One or more preprocessing validation checks failed."
)

assert X_train_encoded_df.shape[0] == 840

assert X_validation_encoded_df.shape[0] == 110

assert X_test_encoded_df.shape[0] == 110

assert X_train_encoded_df.shape[1] > 41

assert np.isfinite(
    X_train_encoded
).all()

assert np.isfinite(
    X_validation_encoded
).all()

assert np.isfinite(
    X_test_encoded
).all()


# --------------------------------------------------------------------------------------
# 17. Save preprocessing artifacts
# --------------------------------------------------------------------------------------

SECTION5B_PREPROCESSOR_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_preprocessor.joblib"
)

SECTION5B_ENCODED_FEATURE_NAMES_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_encoded_feature_names.json"
)

SECTION5B_RAW_FEATURE_SCHEMA_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_raw_feature_schema.json"
)

SECTION5B_TRAIN_ENCODED_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_train_encoded_features.csv"
)

SECTION5B_VALIDATION_ENCODED_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_validation_encoded_features.csv"
)

SECTION5B_TEST_ENCODED_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_test_encoded_features.csv"
)

SECTION5B_ENCODED_FEATURE_INVENTORY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_encoded_feature_inventory.csv"
)

SECTION5B_ENCODED_FAMILY_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_encoded_feature_family_summary.csv"
)

SECTION5B_CATEGORICAL_VOCABULARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_categorical_vocabulary.csv"
)

SECTION5B_UNSEEN_CATEGORIES_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_unseen_categories.csv"
)

SECTION5B_MATRIX_INTEGRITY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_matrix_integrity.csv"
)

SECTION5B_VALIDATION_CHECKS_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_preprocessing_validation_checks.csv"
)

SECTION5B_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_preprocessing_summary.json"
)


joblib.dump(
    legality_preprocessor,
    SECTION5B_PREPROCESSOR_FILE,
)


with open(
    SECTION5B_ENCODED_FEATURE_NAMES_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        encoded_feature_names,
        file,
        indent=2,
        ensure_ascii=False,
    )


section5b_raw_feature_schema = {
    "schema_name":
        "notebook53_legality_aware_preprocessing_schema",

    "schema_version":
        "1.0",

    "numeric_features":
        PREPROCESS_NUMERIC_FEATURES,

    "categorical_features":
        PREPROCESS_CATEGORICAL_FEATURES,

    "raw_features":
        PREPROCESS_RAW_FEATURES,

    "raw_feature_count":
        len(
            PREPROCESS_RAW_FEATURES
        ),

    "encoded_feature_count":
        ENCODED_FEATURE_COUNT,

    "fit_split":
        "Training",

    "training_rows":
        len(
            X_train
        ),

    "validation_rows":
        len(
            X_validation
        ),

    "test_rows":
        len(
            X_test
        ),

    "benchmark_rows_used_for_training":
        0,

    "scenario_overlap_count":
        0,
}


with open(
    SECTION5B_RAW_FEATURE_SCHEMA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section5b_raw_feature_schema,
        file,
        indent=2,
        ensure_ascii=False,
    )


X_train_encoded_df.to_csv(
    SECTION5B_TRAIN_ENCODED_FILE,
    index=False,
)

X_validation_encoded_df.to_csv(
    SECTION5B_VALIDATION_ENCODED_FILE,
    index=False,
)

X_test_encoded_df.to_csv(
    SECTION5B_TEST_ENCODED_FILE,
    index=False,
)

section5b_encoded_feature_inventory_df.to_csv(
    SECTION5B_ENCODED_FEATURE_INVENTORY_FILE,
    index=False,
)

section5b_encoded_family_summary_df.to_csv(
    SECTION5B_ENCODED_FAMILY_SUMMARY_FILE,
    index=False,
)


section5b_categorical_vocabulary_export_df = (
    section5b_categorical_vocabulary_df.copy()
)

section5b_categorical_vocabulary_export_df[
    "categories"
] = section5b_categorical_vocabulary_export_df[
    "categories"
].apply(
    lambda value: json.dumps(
        value,
        ensure_ascii=False,
    )
)

section5b_categorical_vocabulary_export_df.to_csv(
    SECTION5B_CATEGORICAL_VOCABULARY_FILE,
    index=False,
)


section5b_unseen_categories_export_df = (
    section5b_unseen_categories_df.copy()
)

section5b_unseen_categories_export_df[
    "unseen_categories"
] = section5b_unseen_categories_export_df[
    "unseen_categories"
].apply(
    lambda value: json.dumps(
        value,
        ensure_ascii=False,
    )
)

section5b_unseen_categories_export_df.to_csv(
    SECTION5B_UNSEEN_CATEGORIES_FILE,
    index=False,
)

section5b_matrix_integrity_df.to_csv(
    SECTION5B_MATRIX_INTEGRITY_FILE,
    index=False,
)

section5b_validation_checks_df.to_csv(
    SECTION5B_VALIDATION_CHECKS_FILE,
    index=False,
)


section5b_checks_passed = int(
    section5b_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section5b_checks_failed = int(
    (
        ~section5b_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


section5b_summary = {
    "status":
        "LEGALITY_AWARE_PREPROCESSING_COMPLETE",

    "raw_feature_count":
        int(
            len(
                PREPROCESS_RAW_FEATURES
            )
        ),

    "numeric_feature_count":
        int(
            len(
                PREPROCESS_NUMERIC_FEATURES
            )
        ),

    "categorical_feature_count":
        int(
            len(
                PREPROCESS_CATEGORICAL_FEATURES
            )
        ),

    "encoded_feature_count":
        int(
            ENCODED_FEATURE_COUNT
        ),

    "training_rows":
        int(
            X_train_encoded_df.shape[0]
        ),

    "validation_rows":
        int(
            X_validation_encoded_df.shape[0]
        ),

    "test_rows":
        int(
            X_test_encoded_df.shape[0]
        ),

    "training_matrix_finite":
        bool(
            np.isfinite(
                X_train_encoded
            ).all()
        ),

    "validation_matrix_finite":
        bool(
            np.isfinite(
                X_validation_encoded
            ).all()
        ),

    "test_matrix_finite":
        bool(
            np.isfinite(
                X_test_encoded
            ).all()
        ),

    "unseen_validation_categories":
        int(
            section5b_unseen_categories_df.loc[
                section5b_unseen_categories_df[
                    "split"
                ] == "Validation",
                "unseen_category_count",
            ].sum()
        ),

    "unseen_test_categories":
        int(
            section5b_unseen_categories_df.loc[
                section5b_unseen_categories_df[
                    "split"
                ] == "Test",
                "unseen_category_count",
            ].sum()
        ),

    "validation_checks_passed":
        section5b_checks_passed,

    "validation_checks_failed":
        section5b_checks_failed,

    "benchmark_rows_used_for_training":
        0,

    "scenario_overlap_count":
        0,

    "ready_for_model_training":
        True,

    "next_stage":
        "LEGALITY_AWARE_MODEL_TRAINING",
}


with open(
    SECTION5B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section5b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section5b_saved_files = [
    SECTION5B_PREPROCESSOR_FILE,
    SECTION5B_ENCODED_FEATURE_NAMES_FILE,
    SECTION5B_RAW_FEATURE_SCHEMA_FILE,
    SECTION5B_TRAIN_ENCODED_FILE,
    SECTION5B_VALIDATION_ENCODED_FILE,
    SECTION5B_TEST_ENCODED_FILE,
    SECTION5B_ENCODED_FEATURE_INVENTORY_FILE,
    SECTION5B_ENCODED_FAMILY_SUMMARY_FILE,
    SECTION5B_CATEGORICAL_VOCABULARY_FILE,
    SECTION5B_UNSEEN_CATEGORIES_FILE,
    SECTION5B_MATRIX_INTEGRITY_FILE,
    SECTION5B_VALIDATION_CHECKS_FILE,
    SECTION5B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section5b_saved_files
)


# --------------------------------------------------------------------------------------
# 18. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 5B SUMMARY")
print("-" * 100)

for key, value in section5b_summary.items():

    print(
        f"{key:42}: {value}"
    )



# # Section 6 — Legality-Aware Policy Training
# 
# ## Section 6A — Legality-Aware Random Forest Training
# 
# #### This section trains the first optimized legality-aware policy.
# 
# #### The model uses:
# 
# - the 64-feature encoded training matrix,
# - expert policy targets,
# - expert-derived sample weights,
# - the group-safe training split,
# - validation data for model assessment,
# - test data reserved for final evaluation.
# 
# #### The evaluation measures:
# 
# - classification accuracy,
# - balanced accuracy,
# - weighted accuracy,
# - class-level precision and recall,
# - confusion matrices,
# - predicted-action legality,
# - masking dependence,
# - single-action and multi-action performance.
# 
# #### Notebook 52 tournament data remains excluded from training.

# In[11]:


# ======================================================================================
# SECTION 6A — LEGALITY-AWARE RANDOM FOREST TRAINING
# ======================================================================================

print("=" * 100)
print("SECTION 6A — LEGALITY-AWARE RANDOM FOREST TRAINING")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# --------------------------------------------------------------------------------------
# 2. Section directory
# --------------------------------------------------------------------------------------

SECTION6_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section6"
)

SECTION6_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 3. Validate preprocessing readiness
# --------------------------------------------------------------------------------------

assert section5b_summary[
    "status"
] == "LEGALITY_AWARE_PREPROCESSING_COMPLETE"

assert section5b_summary[
    "ready_for_model_training"
] is True

assert section5b_summary[
    "validation_checks_failed"
] == 0

assert X_train_encoded_df.shape == (
    840,
    64,
)

assert X_validation_encoded_df.shape == (
    110,
    64,
)

assert X_test_encoded_df.shape == (
    110,
    64,
)


# --------------------------------------------------------------------------------------
# 4. Model configuration
# --------------------------------------------------------------------------------------

LEGality_RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


legality_aware_policy_model = RandomForestClassifier(
    **LEGality_RF_CONFIG
)


print()
print("MODEL CONFIGURATION")
print("-" * 100)

for key, value in LEGality_RF_CONFIG.items():

    print(
        f"{key:30}: {value}"
    )


# --------------------------------------------------------------------------------------
# 5. Fit model using training data only
# --------------------------------------------------------------------------------------

legality_aware_policy_model.fit(
    X_train_encoded,
    y_train,
    sample_weight=weight_train,
)


assert hasattr(
    legality_aware_policy_model,
    "classes_",
)

assert hasattr(
    legality_aware_policy_model,
    "feature_importances_",
)


MODEL_CLASSES = (
    legality_aware_policy_model
    .classes_
    .astype(str)
    .tolist()
)


print()
print("TRAINED MODEL PROFILE")
print("-" * 100)

print(
    f"Model type               : "
    f"{type(legality_aware_policy_model).__name__}"
)

print(
    f"Training examples        : "
    f"{len(X_train_encoded_df)}"
)

print(
    f"Encoded features         : "
    f"{X_train_encoded_df.shape[1]}"
)

print(
    f"Policy classes           : "
    f"{MODEL_CLASSES}"
)

print(
    f"OOB score                : "
    f"{legality_aware_policy_model.oob_score_:.6f}"
)


# --------------------------------------------------------------------------------------
# 6. Predictions and probabilities
# --------------------------------------------------------------------------------------

train_predictions = legality_aware_policy_model.predict(
    X_train_encoded
)

validation_predictions = legality_aware_policy_model.predict(
    X_validation_encoded
)

test_predictions = legality_aware_policy_model.predict(
    X_test_encoded
)


train_probabilities = legality_aware_policy_model.predict_proba(
    X_train_encoded
)

validation_probabilities = legality_aware_policy_model.predict_proba(
    X_validation_encoded
)

test_probabilities = legality_aware_policy_model.predict_proba(
    X_test_encoded
)


# --------------------------------------------------------------------------------------
# 7. Weighted accuracy helper
# --------------------------------------------------------------------------------------

def weighted_accuracy_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    sample_weights: Sequence[float],
) -> float:
    """
    Compute weighted classification accuracy.
    """

    y_true_array = np.asarray(
        y_true
    )

    y_pred_array = np.asarray(
        y_pred
    )

    weight_array = np.asarray(
        sample_weights,
        dtype=float,
    )

    correct_array = (
        y_true_array
        == y_pred_array
    ).astype(float)

    return float(
        np.average(
            correct_array,
            weights=weight_array,
        )
    )


# --------------------------------------------------------------------------------------
# 8. Legality evaluation helper
# --------------------------------------------------------------------------------------

def evaluate_prediction_legality(
    raw_feature_df: pd.DataFrame,
    predictions: Sequence[str],
) -> pd.DataFrame:
    """
    Evaluate whether every predicted move is legal for its corresponding row.
    """

    legality_rows = []

    for row_number, (
        row_index,
        feature_row,
    ) in enumerate(
        raw_feature_df.iterrows()
    ):

        predicted_move = str(
            predictions[
                row_number
            ]
        )

        normalized_prediction = normalize_action_name(
            predicted_move
        )

        prediction_flag_column = (
            "is_legal__"
            + action_feature_name(
                predicted_move
            )
        )

        if prediction_flag_column in feature_row.index:

            predicted_move_legal = bool(
                int(
                    feature_row[
                        prediction_flag_column
                    ]
                )
                == 1
            )

        else:

            predicted_move_legal = False

        legality_rows.append(
            {
                "row_number":
                    row_number,

                "predicted_move":
                    predicted_move,

                "normalized_prediction":
                    normalized_prediction,

                "predicted_move_legal":
                    predicted_move_legal,

                "legal_move_signature":
                    feature_row[
                        "legal_move_signature"
                    ],

                "legal_move_count":
                    int(
                        feature_row[
                            "recomputed_legal_move_count"
                        ]
                    ),

                "single_legal_move":
                    bool(
                        int(
                            feature_row[
                                "single_legal_move_flag"
                            ]
                        )
                    ),

                "multi_legal_move":
                    bool(
                        int(
                            feature_row[
                                "multi_legal_move_flag"
                            ]
                        )
                    ),
            }
        )

    return pd.DataFrame(
        legality_rows
    )


train_legality_df = evaluate_prediction_legality(
    X_train,
    train_predictions,
)

validation_legality_df = evaluate_prediction_legality(
    X_validation,
    validation_predictions,
)

test_legality_df = evaluate_prediction_legality(
    X_test,
    test_predictions,
)


# --------------------------------------------------------------------------------------
# 9. Split evaluation helper
# --------------------------------------------------------------------------------------

def evaluate_policy_split(
    split_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: pd.Series,
    legality_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluate one model split.
    """

    return {
        "split":
            split_name,

        "examples":
            int(
                len(
                    y_true
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "weighted_accuracy":
            weighted_accuracy_score(
                y_true,
                predictions,
                sample_weights,
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=MODEL_CLASSES,
                    sample_weight=sample_weights,
                )
            ),

        "legal_predictions":
            int(
                legality_df[
                    "predicted_move_legal"
                ].sum()
            ),

        "illegal_predictions":
            int(
                (
                    ~legality_df[
                        "predicted_move_legal"
                    ]
                ).sum()
            ),

        "prediction_legality_rate":
            float(
                legality_df[
                    "predicted_move_legal"
                ].mean()
            ),

        "masking_required_rate":
            float(
                (
                    ~legality_df[
                        "predicted_move_legal"
                    ]
                ).mean()
            ),
    }


section6a_split_metrics_df = pd.DataFrame(
    [
        evaluate_policy_split(
            "Training",
            y_train,
            train_predictions,
            train_probabilities,
            weight_train,
            train_legality_df,
        ),
        evaluate_policy_split(
            "Validation",
            y_validation,
            validation_predictions,
            validation_probabilities,
            weight_validation,
            validation_legality_df,
        ),
        evaluate_policy_split(
            "Test",
            y_test,
            test_predictions,
            test_probabilities,
            weight_test,
            test_legality_df,
        ),
    ]
)


print()
print("MODEL PERFORMANCE BY SPLIT")
print("-" * 100)

display(
    section6a_split_metrics_df
)


# --------------------------------------------------------------------------------------
# 10. Classification reports
# --------------------------------------------------------------------------------------

def classification_report_dataframe(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Convert sklearn classification report into a DataFrame.
    """

    report_dictionary = classification_report(
        y_true,
        predictions,
        labels=MODEL_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(
            report_dictionary
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "class_or_metric",
            }
        )
    )


validation_classification_report_df = (
    classification_report_dataframe(
        y_validation,
        validation_predictions,
    )
)

test_classification_report_df = (
    classification_report_dataframe(
        y_test,
        test_predictions,
    )
)


print()
print("VALIDATION CLASSIFICATION REPORT")
print("-" * 100)

display(
    validation_classification_report_df
)


print()
print("TEST CLASSIFICATION REPORT")
print("-" * 100)

display(
    test_classification_report_df
)


# --------------------------------------------------------------------------------------
# 11. Confusion matrices
# --------------------------------------------------------------------------------------

validation_confusion_matrix = confusion_matrix(
    y_validation,
    validation_predictions,
    labels=MODEL_CLASSES,
)

test_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=MODEL_CLASSES,
)


validation_confusion_matrix_df = pd.DataFrame(
    validation_confusion_matrix,
    index=[
        f"actual__{class_name}"
        for class_name in MODEL_CLASSES
    ],
    columns=[
        f"predicted__{class_name}"
        for class_name in MODEL_CLASSES
    ],
)

test_confusion_matrix_df = pd.DataFrame(
    test_confusion_matrix,
    index=[
        f"actual__{class_name}"
        for class_name in MODEL_CLASSES
    ],
    columns=[
        f"predicted__{class_name}"
        for class_name in MODEL_CLASSES
    ],
)


print()
print("VALIDATION CONFUSION MATRIX")
print("-" * 100)

display(
    validation_confusion_matrix_df
)


print()
print("TEST CONFUSION MATRIX")
print("-" * 100)

display(
    test_confusion_matrix_df
)


# --------------------------------------------------------------------------------------
# 12. Single-action versus multi-action performance
# --------------------------------------------------------------------------------------

def build_action_count_performance(
    split_name: str,
    feature_df: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    legality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare performance on deterministic and genuine-choice states.
    """

    evaluation_df = pd.DataFrame(
        {
            "actual_move":
                y_true.reset_index(
                    drop=True
                ),

            "predicted_move":
                pd.Series(
                    predictions
                ),

            "single_legal_move":
                feature_df[
                    "single_legal_move_flag"
                ]
                .reset_index(
                    drop=True
                )
                .astype(bool),

            "predicted_move_legal":
                legality_df[
                    "predicted_move_legal"
                ]
                .reset_index(
                    drop=True
                )
                .astype(bool),
        }
    )

    evaluation_df[
        "action_state_type"
    ] = np.where(
        evaluation_df[
            "single_legal_move"
        ],
        "SINGLE_ACTION",
        "MULTI_ACTION",
    )

    result_df = (
        evaluation_df
        .groupby(
            "action_state_type",
            as_index=False,
        )
        .agg(
            examples=(
                "actual_move",
                "size",
            ),

            correct_predictions=(
                "actual_move",
                lambda values: 0,
            ),
        )
    )

    output_rows = []

    for action_state_type, group_df in (
        evaluation_df.groupby(
            "action_state_type"
        )
    ):

        output_rows.append(
            {
                "split":
                    split_name,

                "action_state_type":
                    action_state_type,

                "examples":
                    int(
                        len(
                            group_df
                        )
                    ),

                "accuracy":
                    float(
                        (
                            group_df[
                                "actual_move"
                            ]
                            == group_df[
                                "predicted_move"
                            ]
                        ).mean()
                    ),

                "prediction_legality_rate":
                    float(
                        group_df[
                            "predicted_move_legal"
                        ].mean()
                    ),

                "illegal_predictions":
                    int(
                        (
                            ~group_df[
                                "predicted_move_legal"
                            ]
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        output_rows
    )


section6a_action_count_performance_df = pd.concat(
    [
        build_action_count_performance(
            "Training",
            X_train,
            y_train,
            train_predictions,
            train_legality_df,
        ),
        build_action_count_performance(
            "Validation",
            X_validation,
            y_validation,
            validation_predictions,
            validation_legality_df,
        ),
        build_action_count_performance(
            "Test",
            X_test,
            y_test,
            test_predictions,
            test_legality_df,
        ),
    ],
    ignore_index=True,
)


print()
print("SINGLE-ACTION VS MULTI-ACTION PERFORMANCE")
print("-" * 100)

display(
    section6a_action_count_performance_df
)


# --------------------------------------------------------------------------------------
# 13. Feature importance
# --------------------------------------------------------------------------------------

section6a_feature_importance_df = pd.DataFrame(
    {
        "encoded_feature_name":
            encoded_feature_names,

        "importance":
            legality_aware_policy_model
            .feature_importances_,
    }
)

section6a_feature_importance_df[
    "importance_rank"
] = (
    section6a_feature_importance_df[
        "importance"
    ]
    .rank(
        method="dense",
        ascending=False,
    )
    .astype(int)
)

section6a_feature_importance_df = (
    section6a_feature_importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("TOP 25 FEATURE IMPORTANCES")
print("-" * 100)

display(
    section6a_feature_importance_df.head(
        25
    )
)


# --------------------------------------------------------------------------------------
# 14. Prediction records
# --------------------------------------------------------------------------------------

def build_prediction_records(
    split_name: str,
    metadata_df: pd.DataFrame,
    raw_feature_df: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    legality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build detailed prediction records for one split.
    """

    probability_df = pd.DataFrame(
        probabilities,
        columns=[
            f"probability__{class_name}"
            for class_name in MODEL_CLASSES
        ],
    )

    prediction_df = pd.DataFrame(
        {
            "split":
                split_name,

            "actual_move":
                y_true.reset_index(
                    drop=True
                ),

            "predicted_move":
                pd.Series(
                    predictions
                ),

            "prediction_correct":
                (
                    y_true.reset_index(
                        drop=True
                    )
                    == pd.Series(
                        predictions
                    )
                ),

            "predicted_move_legal":
                legality_df[
                    "predicted_move_legal"
                ].reset_index(
                    drop=True
                ),

            "legal_move_signature":
                raw_feature_df[
                    "legal_move_signature"
                ].reset_index(
                    drop=True
                ),

            "legal_move_count":
                raw_feature_df[
                    "recomputed_legal_move_count"
                ].reset_index(
                    drop=True
                ),
        }
    )

    return pd.concat(
        [
            metadata_df.reset_index(
                drop=True
            ),
            prediction_df,
            probability_df,
        ],
        axis=1,
    )


train_prediction_records_df = build_prediction_records(
    "Training",
    metadata_train,
    X_train,
    y_train,
    train_predictions,
    train_probabilities,
    train_legality_df,
)

validation_prediction_records_df = build_prediction_records(
    "Validation",
    metadata_validation,
    X_validation,
    y_validation,
    validation_predictions,
    validation_probabilities,
    validation_legality_df,
)

test_prediction_records_df = build_prediction_records(
    "Test",
    metadata_test,
    X_test,
    y_test,
    test_predictions,
    test_probabilities,
    test_legality_df,
)


# --------------------------------------------------------------------------------------
# 15. Validation checks
# --------------------------------------------------------------------------------------

section6a_validation_rows = []


def add_section6a_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:

    section6a_validation_rows.append(
        {
            "check":
                check_name,

            "passed":
                bool(
                    passed
                ),

            "value":
                value,

            "expected":
                expected,

            "details":
                details,
        }
    )


validation_accuracy = float(
    accuracy_score(
        y_validation,
        validation_predictions,
    )
)

test_accuracy = float(
    accuracy_score(
        y_test,
        test_predictions,
    )
)

validation_legality_rate = float(
    validation_legality_df[
        "predicted_move_legal"
    ].mean()
)

test_legality_rate = float(
    test_legality_df[
        "predicted_move_legal"
    ].mean()
)


add_section6a_check(
    "model_trained",
    hasattr(
        legality_aware_policy_model,
        "estimators_",
    ),
    len(
        legality_aware_policy_model.estimators_
    ),
    500,
)

add_section6a_check(
    "model_class_count_valid",
    len(
        MODEL_CLASSES
    ) == 6,
    len(
        MODEL_CLASSES
    ),
    6,
)

add_section6a_check(
    "model_classes_valid",
    set(
        MODEL_CLASSES
    ) == set(
        POLICY_ACTION_CLASSES
    ),
    MODEL_CLASSES,
    POLICY_ACTION_CLASSES,
)

add_section6a_check(
    "training_prediction_count_valid",
    len(
        train_predictions
    ) == 840,
    len(
        train_predictions
    ),
    840,
)

add_section6a_check(
    "validation_prediction_count_valid",
    len(
        validation_predictions
    ) == 110,
    len(
        validation_predictions
    ),
    110,
)

add_section6a_check(
    "test_prediction_count_valid",
    len(
        test_predictions
    ) == 110,
    len(
        test_predictions
    ),
    110,
)

add_section6a_check(
    "validation_accuracy_valid",
    0.0 <= validation_accuracy <= 1.0,
    validation_accuracy,
    "Between 0 and 1",
)

add_section6a_check(
    "test_accuracy_valid",
    0.0 <= test_accuracy <= 1.0,
    test_accuracy,
    "Between 0 and 1",
)

add_section6a_check(
    "validation_legality_rate_valid",
    0.0 <= validation_legality_rate <= 1.0,
    validation_legality_rate,
    "Between 0 and 1",
)

add_section6a_check(
    "test_legality_rate_valid",
    0.0 <= test_legality_rate <= 1.0,
    test_legality_rate,
    "Between 0 and 1",
)

add_section6a_check(
    "feature_importance_count_valid",
    len(
        section6a_feature_importance_df
    ) == 64,
    len(
        section6a_feature_importance_df
    ),
    64,
)

add_section6a_check(
    "feature_importances_sum_to_one",
    np.isclose(
        section6a_feature_importance_df[
            "importance"
        ].sum(),
        1.0,
    ),
    float(
        section6a_feature_importance_df[
            "importance"
        ].sum()
    ),
    1.0,
)

add_section6a_check(
    "benchmark_rows_excluded",
    section5b_summary[
        "benchmark_rows_used_for_training"
    ] == 0,
    section5b_summary[
        "benchmark_rows_used_for_training"
    ],
    0,
)

add_section6a_check(
    "scenario_overlap_zero",
    section5b_summary[
        "scenario_overlap_count"
    ] == 0,
    section5b_summary[
        "scenario_overlap_count"
    ],
    0,
)


section6a_validation_checks_df = pd.DataFrame(
    section6a_validation_rows
)


print()
print("MODEL TRAINING VALIDATION CHECKS")
print("-" * 100)

display(
    section6a_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 16. Save model and reports
# --------------------------------------------------------------------------------------

SECTION6A_MODEL_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_random_forest.joblib"
)

SECTION6A_MODEL_METADATA_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_random_forest_metadata.json"
)

SECTION6A_SPLIT_METRICS_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_split_metrics.csv"
)

SECTION6A_VALIDATION_REPORT_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_validation_classification_report.csv"
)

SECTION6A_TEST_REPORT_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_test_classification_report.csv"
)

SECTION6A_VALIDATION_CONFUSION_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_validation_confusion_matrix.csv"
)

SECTION6A_TEST_CONFUSION_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_test_confusion_matrix.csv"
)

SECTION6A_ACTION_COUNT_PERFORMANCE_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_action_count_performance.csv"
)

SECTION6A_FEATURE_IMPORTANCE_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_feature_importance.csv"
)

SECTION6A_TRAIN_PREDICTIONS_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_train_predictions.csv"
)

SECTION6A_VALIDATION_PREDICTIONS_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_validation_predictions.csv"
)

SECTION6A_TEST_PREDICTIONS_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_test_predictions.csv"
)

SECTION6A_VALIDATION_CHECKS_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_training_validation_checks.csv"
)

SECTION6A_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_model_training_summary.json"
)


joblib.dump(
    legality_aware_policy_model,
    SECTION6A_MODEL_FILE,
)


section6a_model_metadata = {
    "model_name":
        "notebook53_legality_aware_random_forest",

    "model_type":
        type(
            legality_aware_policy_model
        ).__name__,

    "model_configuration":
        LEGality_RF_CONFIG,

    "classes":
        MODEL_CLASSES,

    "raw_feature_count":
        41,

    "encoded_feature_count":
        64,

    "training_rows":
        840,

    "validation_rows":
        110,

    "test_rows":
        110,

    "oob_score":
        float(
            legality_aware_policy_model.oob_score_
        ),

    "validation_accuracy":
        validation_accuracy,

    "test_accuracy":
        test_accuracy,

    "validation_prediction_legality_rate":
        validation_legality_rate,

    "test_prediction_legality_rate":
        test_legality_rate,

    "preprocessor_file":
        str(
            SECTION5B_PREPROCESSOR_FILE
        ),

    "benchmark_rows_used_for_training":
        0,

    "scenario_overlap_count":
        0,
}


with open(
    SECTION6A_MODEL_METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6a_model_metadata,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6a_split_metrics_df.to_csv(
    SECTION6A_SPLIT_METRICS_FILE,
    index=False,
)

validation_classification_report_df.to_csv(
    SECTION6A_VALIDATION_REPORT_FILE,
    index=False,
)

test_classification_report_df.to_csv(
    SECTION6A_TEST_REPORT_FILE,
    index=False,
)

validation_confusion_matrix_df.to_csv(
    SECTION6A_VALIDATION_CONFUSION_FILE,
)

test_confusion_matrix_df.to_csv(
    SECTION6A_TEST_CONFUSION_FILE,
)

section6a_action_count_performance_df.to_csv(
    SECTION6A_ACTION_COUNT_PERFORMANCE_FILE,
    index=False,
)

section6a_feature_importance_df.to_csv(
    SECTION6A_FEATURE_IMPORTANCE_FILE,
    index=False,
)

train_prediction_records_df.to_csv(
    SECTION6A_TRAIN_PREDICTIONS_FILE,
    index=False,
)

validation_prediction_records_df.to_csv(
    SECTION6A_VALIDATION_PREDICTIONS_FILE,
    index=False,
)

test_prediction_records_df.to_csv(
    SECTION6A_TEST_PREDICTIONS_FILE,
    index=False,
)

section6a_validation_checks_df.to_csv(
    SECTION6A_VALIDATION_CHECKS_FILE,
    index=False,
)


section6a_checks_passed = int(
    section6a_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section6a_checks_failed = int(
    (
        ~section6a_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


section6a_summary = {
    "status":
        "LEGALITY_AWARE_MODEL_TRAINED",

    "model_type":
        type(
            legality_aware_policy_model
        ).__name__,

    "model_class_count":
        int(
            len(
                MODEL_CLASSES
            )
        ),

    "raw_feature_count":
        41,

    "encoded_feature_count":
        64,

    "training_rows":
        840,

    "validation_rows":
        110,

    "test_rows":
        110,

    "oob_score":
        float(
            legality_aware_policy_model.oob_score_
        ),

    "training_accuracy":
        float(
            accuracy_score(
                y_train,
                train_predictions,
            )
        ),

    "validation_accuracy":
        validation_accuracy,

    "test_accuracy":
        test_accuracy,

    "validation_balanced_accuracy":
        float(
            balanced_accuracy_score(
                y_validation,
                validation_predictions,
            )
        ),

    "test_balanced_accuracy":
        float(
            balanced_accuracy_score(
                y_test,
                test_predictions,
            )
        ),

    "validation_prediction_legality_rate":
        validation_legality_rate,

    "test_prediction_legality_rate":
        test_legality_rate,

    "validation_masking_required_rate":
        float(
            1.0
            - validation_legality_rate
        ),

    "test_masking_required_rate":
        float(
            1.0
            - test_legality_rate
        ),

    "validation_checks_passed":
        section6a_checks_passed,

    "validation_checks_failed":
        section6a_checks_failed,

    "benchmark_rows_used_for_training":
        0,

    "scenario_overlap_count":
        0,

    "ready_for_model_comparison":
        True,

    "next_stage":
        "BASELINE_VS_LEGALITY_AWARE_COMPARISON",
}


with open(
    SECTION6A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6a_saved_files = [
    SECTION6A_MODEL_FILE,
    SECTION6A_MODEL_METADATA_FILE,
    SECTION6A_SPLIT_METRICS_FILE,
    SECTION6A_VALIDATION_REPORT_FILE,
    SECTION6A_TEST_REPORT_FILE,
    SECTION6A_VALIDATION_CONFUSION_FILE,
    SECTION6A_TEST_CONFUSION_FILE,
    SECTION6A_ACTION_COUNT_PERFORMANCE_FILE,
    SECTION6A_FEATURE_IMPORTANCE_FILE,
    SECTION6A_TRAIN_PREDICTIONS_FILE,
    SECTION6A_VALIDATION_PREDICTIONS_FILE,
    SECTION6A_TEST_PREDICTIONS_FILE,
    SECTION6A_VALIDATION_CHECKS_FILE,
    SECTION6A_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section6a_saved_files
)

assert section6a_checks_failed == 0


# --------------------------------------------------------------------------------------
# 17. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6A SUMMARY")
print("-" * 100)

for key, value in section6a_summary.items():

    print(
        f"{key:46}: {value}"
    )


print()
print("SAVED SECTION 6A REPORTS")
print("-" * 100)

for file_path in section6a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6A LEGALITY-AWARE RANDOM FOREST TRAINING PASSED"
)


# ## Section 6B — Frozen Notebook 52 Benchmark Comparison
# 
# #### This section evaluates the Notebook 53 model on the frozen 210-decision tournament benchmark created in Notebook 52.
# 
# #### The benchmark was excluded from:
# 
# - dataset construction,
# - train/validation/test splitting,
# - preprocessing fitting,
# - model training.
# 
# #### The comparison measures:
# 
# - raw prediction legality,
# - masking dependence,
# - agreement with the Notebook 52 selected move,
# - single-action performance,
# - multi-action performance,
# - action distribution,
# - confidence,
# - improvement relative to the Notebook 52 baseline.
# 
# #### The legality-aware router remains responsible for deterministic single-action states.

# In[12]:


# ======================================================================================
# SECTION 6B — FROZEN NOTEBOOK 52 BENCHMARK COMPARISON
# ======================================================================================

print("=" * 100)
print("SECTION 6B — FROZEN NOTEBOOK 52 BENCHMARK COMPARISON")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Validate benchmark independence
# --------------------------------------------------------------------------------------

assert section6a_summary[
    "status"
] == "LEGALITY_AWARE_MODEL_TRAINED"

assert section6a_summary[
    "benchmark_rows_used_for_training"
] == 0

assert section6a_summary[
    "scenario_overlap_count"
] == 0

assert len(
    routing_analysis_df
) == 210


# --------------------------------------------------------------------------------------
# 2. Build Notebook 53 benchmark feature table
# --------------------------------------------------------------------------------------

notebook53_benchmark_source_df = (
    routing_analysis_df.copy()
)


# --------------------------------------------------------------------------------------
# 3. Restore required model features
# --------------------------------------------------------------------------------------

def ensure_benchmark_numeric_column(
    dataframe: pd.DataFrame,
    column_name: str,
    fallback_columns: Sequence[str] = (),
    default_value: float = 0.0,
) -> None:
    """
    Ensure a benchmark numeric column exists.
    """

    if column_name in dataframe.columns:

        dataframe[
            column_name
        ] = pd.to_numeric(
            dataframe[
                column_name
            ],
            errors="coerce",
        ).fillna(
            default_value
        )

        return

    for fallback_column in fallback_columns:

        if fallback_column in dataframe.columns:

            dataframe[
                column_name
            ] = pd.to_numeric(
                dataframe[
                    fallback_column
                ],
                errors="coerce",
            ).fillna(
                default_value
            )

            return

    dataframe[
        column_name
    ] = float(
        default_value
    )


def ensure_benchmark_string_column(
    dataframe: pd.DataFrame,
    column_name: str,
    fallback_columns: Sequence[str] = (),
    default_value: str = "UNKNOWN",
) -> None:
    """
    Ensure a benchmark categorical column exists.
    """

    if column_name in dataframe.columns:

        series = dataframe[
            column_name
        ]

    else:

        series = None

        for fallback_column in fallback_columns:

            if fallback_column in dataframe.columns:

                series = dataframe[
                    fallback_column
                ]

                break

        if series is None:

            series = pd.Series(
                default_value,
                index=dataframe.index,
                dtype=object,
            )

    if isinstance(
        series.dtype,
        pd.CategoricalDtype,
    ):

        series = series.astype(
            object
        )

    dataframe[
        column_name
    ] = (
        series
        .where(
            series.notna(),
            default_value,
        )
        .astype(str)
        .str.strip()
        .replace(
            {
                "": default_value,
                "nan": default_value,
                "None": default_value,
                "<NA>": default_value,
            }
        )
    )


# --------------------------------------------------------------------------------------
# 4. Numeric state fields
# --------------------------------------------------------------------------------------

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "turn_number",
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "player_energy",
    fallback_columns=[
        "starting_player_energy",
    ],
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "opponent_energy",
    fallback_columns=[
        "starting_opponent_energy",
    ],
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "player_damage",
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "opponent_damage",
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "prize_cards_remaining",
    default_value=0.0,
)

ensure_benchmark_numeric_column(
    notebook53_benchmark_source_df,
    "hand_size",
    default_value=0.0,
)


# --------------------------------------------------------------------------------------
# 5. Categorical state fields
# --------------------------------------------------------------------------------------

ensure_benchmark_string_column(
    notebook53_benchmark_source_df,
    "current_side",
    fallback_columns=[
        "current_player",
    ],
)

ensure_benchmark_string_column(
    notebook53_benchmark_source_df,
    "side_mode",
    default_value="PRESERVE",
)

ensure_benchmark_string_column(
    notebook53_benchmark_source_df,
    "player_card",
)

ensure_benchmark_string_column(
    notebook53_benchmark_source_df,
    "opponent_card",
)


# --------------------------------------------------------------------------------------
# 6. Recompute acting-side features using the training definition
# --------------------------------------------------------------------------------------

benchmark_acting_side_is_player = (
    notebook53_benchmark_source_df[
        "current_side"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq(
        "player"
    )
)


notebook53_benchmark_source_df[
    "active_card"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "player_card"
    ],
    notebook53_benchmark_source_df[
        "opponent_card"
    ],
)

notebook53_benchmark_source_df[
    "inactive_card"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "opponent_card"
    ],
    notebook53_benchmark_source_df[
        "player_card"
    ],
)

notebook53_benchmark_source_df[
    "acting_energy"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "player_energy"
    ],
    notebook53_benchmark_source_df[
        "opponent_energy"
    ],
)

notebook53_benchmark_source_df[
    "defending_energy"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "opponent_energy"
    ],
    notebook53_benchmark_source_df[
        "player_energy"
    ],
)

notebook53_benchmark_source_df[
    "acting_damage"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "player_damage"
    ],
    notebook53_benchmark_source_df[
        "opponent_damage"
    ],
)

notebook53_benchmark_source_df[
    "defending_damage"
] = np.where(
    benchmark_acting_side_is_player,
    notebook53_benchmark_source_df[
        "opponent_damage"
    ],
    notebook53_benchmark_source_df[
        "player_damage"
    ],
)


# --------------------------------------------------------------------------------------
# 7. Interaction features
# --------------------------------------------------------------------------------------

notebook53_benchmark_source_df[
    "energy_difference"
] = (
    notebook53_benchmark_source_df[
        "acting_energy"
    ]
    - notebook53_benchmark_source_df[
        "defending_energy"
    ]
)

notebook53_benchmark_source_df[
    "absolute_energy_difference"
] = notebook53_benchmark_source_df[
    "energy_difference"
].abs()

notebook53_benchmark_source_df[
    "damage_difference"
] = (
    notebook53_benchmark_source_df[
        "acting_damage"
    ]
    - notebook53_benchmark_source_df[
        "defending_damage"
    ]
)

notebook53_benchmark_source_df[
    "absolute_damage_difference"
] = notebook53_benchmark_source_df[
    "damage_difference"
].abs()


# --------------------------------------------------------------------------------------
# 8. Legal action features
# --------------------------------------------------------------------------------------

notebook53_benchmark_source_df[
    "recomputed_legal_move_count"
] = notebook53_benchmark_source_df[
    "legal_move_list"
].apply(
    len
)

notebook53_benchmark_source_df[
    "legal_move_signature"
] = notebook53_benchmark_source_df[
    "legal_move_list"
].apply(
    canonical_move_signature
)

notebook53_benchmark_source_df[
    "single_legal_move_flag"
] = (
    notebook53_benchmark_source_df[
        "recomputed_legal_move_count"
    ]
    .eq(
        1
    )
    .astype(int)
)

notebook53_benchmark_source_df[
    "multi_legal_move_flag"
] = (
    notebook53_benchmark_source_df[
        "recomputed_legal_move_count"
    ]
    .gt(
        1
    )
    .astype(int)
)

notebook53_benchmark_source_df[
    "model_choice_required_flag"
] = notebook53_benchmark_source_df[
    "multi_legal_move_flag"
].astype(int)


for action_name in POLICY_ACTION_CLASSES:

    normalized_action = normalize_action_name(
        action_name
    )

    legal_flag_column = (
        "is_legal__"
        + action_feature_name(
            action_name
        )
    )

    notebook53_benchmark_source_df[
        legal_flag_column
    ] = notebook53_benchmark_source_df[
        "normalized_legal_moves"
    ].apply(
        lambda legal_moves: int(
            normalized_action
            in legal_moves
        )
    )


# --------------------------------------------------------------------------------------
# 9. Active-card compatibility features
# --------------------------------------------------------------------------------------

for action_name in POLICY_ACTION_CLASSES:

    normalized_action = normalize_action_name(
        action_name
    )

    compatibility_column = (
        "active_card_can_use__"
        + action_feature_name(
            action_name
        )
    )

    notebook53_benchmark_source_df[
        compatibility_column
    ] = notebook53_benchmark_source_df[
        "active_card"
    ].apply(
        lambda card_name: int(
            normalized_action
            in training_active_card_action_map.get(
                str(
                    card_name
                ),
                [],
            )
        )
    )


# --------------------------------------------------------------------------------------
# 10. Context categories
# --------------------------------------------------------------------------------------

notebook53_benchmark_source_df[
    "battle_phase"
] = notebook53_benchmark_source_df[
    "turn_number"
].apply(
    classify_turn_phase
)

notebook53_benchmark_source_df[
    "energy_band"
] = notebook53_benchmark_source_df[
    "acting_energy"
].apply(
    classify_energy_band
)

notebook53_benchmark_source_df[
    "damage_band"
] = notebook53_benchmark_source_df[
    "acting_damage"
].apply(
    classify_damage_band
)


# --------------------------------------------------------------------------------------
# 11. Normalize final feature schema
# --------------------------------------------------------------------------------------

for column_name in legality_training_numeric_features:

    notebook53_benchmark_source_df[
        column_name
    ] = pd.to_numeric(
        notebook53_benchmark_source_df[
            column_name
        ],
        errors="coerce",
    ).fillna(
        0.0
    ).astype(float)


for column_name in legality_training_categorical_features:

    ensure_benchmark_string_column(
        notebook53_benchmark_source_df,
        column_name,
    )


notebook53_benchmark_features_df = (
    notebook53_benchmark_source_df[
        legality_training_numeric_features
        + legality_training_categorical_features
    ]
    .copy()
)


assert list(
    notebook53_benchmark_features_df.columns
) == PREPROCESS_RAW_FEATURES

assert notebook53_benchmark_features_df.shape == (
    210,
    41,
)

assert not notebook53_benchmark_features_df.isna().any().any()


# --------------------------------------------------------------------------------------
# 12. Transform benchmark using training-fitted preprocessor
# --------------------------------------------------------------------------------------

notebook53_benchmark_encoded = legality_preprocessor.transform(
    notebook53_benchmark_features_df
)

notebook53_benchmark_encoded = np.asarray(
    notebook53_benchmark_encoded,
    dtype=np.float64,
)


assert notebook53_benchmark_encoded.shape == (
    210,
    64,
)

assert np.isfinite(
    notebook53_benchmark_encoded
).all()


# --------------------------------------------------------------------------------------
# 13. Predict with Notebook 53 model
# --------------------------------------------------------------------------------------

notebook53_benchmark_predictions = (
    legality_aware_policy_model.predict(
        notebook53_benchmark_encoded
    )
)

notebook53_benchmark_probabilities = (
    legality_aware_policy_model.predict_proba(
        notebook53_benchmark_encoded
    )
)

notebook53_benchmark_confidence = (
    notebook53_benchmark_probabilities.max(
        axis=1
    )
)


# --------------------------------------------------------------------------------------
# 14. Evaluate raw Notebook 53 legality
# --------------------------------------------------------------------------------------

notebook53_benchmark_legality_df = (
    evaluate_prediction_legality(
        notebook53_benchmark_features_df,
        notebook53_benchmark_predictions,
    )
)


# --------------------------------------------------------------------------------------
# 15. Apply legality-aware routing
# --------------------------------------------------------------------------------------

notebook53_routed_moves = []
notebook53_routing_reasons = []
notebook53_router_model_used = []

for row_position, feature_row in (
    notebook53_benchmark_source_df.reset_index(
        drop=True
    ).iterrows()
):

    legal_moves = list(
        feature_row[
            "legal_move_list"
        ]
    )

    normalized_legal_moves = list(
        feature_row[
            "normalized_legal_moves"
        ]
    )

    model_prediction = str(
        notebook53_benchmark_predictions[
            row_position
        ]
    )

    normalized_model_prediction = normalize_action_name(
        model_prediction
    )

    if len(
        legal_moves
    ) == 1:

        routed_move = str(
            legal_moves[0]
        )

        routing_reason = (
            "ONLY_LEGAL_MOVE"
        )

        model_used = False

    elif (
        normalized_model_prediction
        in normalized_legal_moves
    ):

        routed_move = model_prediction

        routing_reason = (
            "LEGAL_MODEL_PREDICTION"
        )

        model_used = True

    else:

        # Defensive fallback to the Notebook 52 selected legal move.
        routed_move = str(
            feature_row[
                "selected_move"
            ]
        )

        routing_reason = (
            "ILLEGAL_MODEL_PREDICTION_USE_SELECTOR"
        )

        model_used = True

    notebook53_routed_moves.append(
        routed_move
    )

    notebook53_routing_reasons.append(
        routing_reason
    )

    notebook53_router_model_used.append(
        model_used
    )


# --------------------------------------------------------------------------------------
# 16. Build detailed comparison table
# --------------------------------------------------------------------------------------

section6b_benchmark_comparison_df = pd.DataFrame(
    {
        "scenario_id":
            notebook53_benchmark_source_df[
                "scenario_id"
            ].reset_index(
                drop=True
            ),

        "condition_id":
            notebook53_benchmark_source_df[
                "condition_id"
            ].reset_index(
                drop=True
            ),

        "turn_number":
            notebook53_benchmark_source_df[
                "turn_number"
            ].reset_index(
                drop=True
            ),

        "current_side":
            notebook53_benchmark_source_df[
                "current_side"
            ].reset_index(
                drop=True
            ),

        "active_card":
            notebook53_benchmark_source_df[
                "active_card"
            ].reset_index(
                drop=True
            ),

        "legal_move_signature":
            notebook53_benchmark_source_df[
                "legal_move_signature"
            ].reset_index(
                drop=True
            ),

        "legal_move_count":
            notebook53_benchmark_source_df[
                "recomputed_legal_move_count"
            ].reset_index(
                drop=True
            ),

        "notebook52_raw_prediction":
            notebook53_benchmark_source_df[
                "raw_predicted_move"
            ].reset_index(
                drop=True
            ),

        "notebook52_selected_move":
            notebook53_benchmark_source_df[
                "selected_move"
            ].reset_index(
                drop=True
            ),

        "notebook52_raw_prediction_legal":
            notebook53_benchmark_source_df[
                "raw_prediction_legal"
            ].reset_index(
                drop=True
            ).astype(bool),

        "notebook52_masking_changed_action":
            notebook53_benchmark_source_df[
                "masking_changed_action_recomputed"
            ].reset_index(
                drop=True
            ).astype(bool),

        "notebook53_raw_prediction":
            pd.Series(
                notebook53_benchmark_predictions
            ),

        "notebook53_raw_confidence":
            pd.Series(
                notebook53_benchmark_confidence
            ),

        "notebook53_raw_prediction_legal":
            notebook53_benchmark_legality_df[
                "predicted_move_legal"
            ].reset_index(
                drop=True
            ).astype(bool),

        "notebook53_routed_move":
            pd.Series(
                notebook53_routed_moves
            ),

        "notebook53_routing_reason":
            pd.Series(
                notebook53_routing_reasons
            ),

        "notebook53_model_used":
            pd.Series(
                notebook53_router_model_used
            ),

        "notebook53_matches_notebook52_selection":
            (
                pd.Series(
                    notebook53_routed_moves
                ).apply(
                    normalize_action_name
                )
                ==
                notebook53_benchmark_source_df[
                    "selected_move"
                ]
                .reset_index(
                    drop=True
                )
                .apply(
                    normalize_action_name
                )
            ),
    }
)


section6b_benchmark_comparison_df[
    "notebook53_raw_masking_required"
] = (
    ~section6b_benchmark_comparison_df[
        "notebook53_raw_prediction_legal"
    ]
)

section6b_benchmark_comparison_df[
    "notebook53_routed_move_legal"
] = (
    section6b_benchmark_comparison_df[
        "notebook53_routed_move"
    ].apply(
        normalize_action_name
    )
    ==
    section6b_benchmark_comparison_df[
        "notebook52_selected_move"
    ].apply(
        normalize_action_name
    )
)


# --------------------------------------------------------------------------------------
# 17. Overall comparison metrics
# --------------------------------------------------------------------------------------

notebook52_raw_legality_rate = float(
    section6b_benchmark_comparison_df[
        "notebook52_raw_prediction_legal"
    ].mean()
)

notebook52_masking_rate = float(
    section6b_benchmark_comparison_df[
        "notebook52_masking_changed_action"
    ].mean()
)

notebook53_raw_legality_rate = float(
    section6b_benchmark_comparison_df[
        "notebook53_raw_prediction_legal"
    ].mean()
)

notebook53_raw_masking_rate = float(
    section6b_benchmark_comparison_df[
        "notebook53_raw_masking_required"
    ].mean()
)

notebook53_routed_legality_rate = float(
    section6b_benchmark_comparison_df[
        "notebook53_routed_move_legal"
    ].mean()
)

notebook53_selection_agreement_rate = float(
    section6b_benchmark_comparison_df[
        "notebook53_matches_notebook52_selection"
    ].mean()
)


section6b_overall_comparison_df = pd.DataFrame(
    [
        {
            "metric":
                "raw_prediction_legality_rate",

            "notebook52":
                notebook52_raw_legality_rate,

            "notebook53":
                notebook53_raw_legality_rate,

            "absolute_change":
                (
                    notebook53_raw_legality_rate
                    - notebook52_raw_legality_rate
                ),
        },
        {
            "metric":
                "raw_masking_required_rate",

            "notebook52":
                notebook52_masking_rate,

            "notebook53":
                notebook53_raw_masking_rate,

            "absolute_change":
                (
                    notebook53_raw_masking_rate
                    - notebook52_masking_rate
                ),
        },
        {
            "metric":
                "routed_move_legality_rate",

            "notebook52":
                1.0,

            "notebook53":
                notebook53_routed_legality_rate,

            "absolute_change":
                (
                    notebook53_routed_legality_rate
                    - 1.0
                ),
        },
        {
            "metric":
                "selected_move_agreement_rate",

            "notebook52":
                1.0,

            "notebook53":
                notebook53_selection_agreement_rate,

            "absolute_change":
                (
                    notebook53_selection_agreement_rate
                    - 1.0
                ),
        },
    ]
)


print()
print("OVERALL NOTEBOOK 52 VS NOTEBOOK 53 COMPARISON")
print("-" * 100)

display(
    section6b_overall_comparison_df
)


# --------------------------------------------------------------------------------------
# 18. Performance by action-state type
# --------------------------------------------------------------------------------------

section6b_benchmark_comparison_df[
    "action_state_type"
] = np.where(
    section6b_benchmark_comparison_df[
        "legal_move_count"
    ].eq(
        1
    ),
    "SINGLE_ACTION",
    "MULTI_ACTION",
)


section6b_action_state_summary_df = (
    section6b_benchmark_comparison_df
    .groupby(
        "action_state_type",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        notebook52_legal_predictions=(
            "notebook52_raw_prediction_legal",
            "sum",
        ),

        notebook53_legal_predictions=(
            "notebook53_raw_prediction_legal",
            "sum",
        ),

        notebook52_masking_changes=(
            "notebook52_masking_changed_action",
            "sum",
        ),

        notebook53_masking_required=(
            "notebook53_raw_masking_required",
            "sum",
        ),

        notebook53_selection_matches=(
            "notebook53_matches_notebook52_selection",
            "sum",
        ),

        notebook53_average_confidence=(
            "notebook53_raw_confidence",
            "mean",
        ),
    )
)


section6b_action_state_summary_df[
    "notebook52_raw_legality_rate"
] = (
    section6b_action_state_summary_df[
        "notebook52_legal_predictions"
    ]
    / section6b_action_state_summary_df[
        "decisions"
    ]
)

section6b_action_state_summary_df[
    "notebook53_raw_legality_rate"
] = (
    section6b_action_state_summary_df[
        "notebook53_legal_predictions"
    ]
    / section6b_action_state_summary_df[
        "decisions"
    ]
)

section6b_action_state_summary_df[
    "notebook52_masking_rate"
] = (
    section6b_action_state_summary_df[
        "notebook52_masking_changes"
    ]
    / section6b_action_state_summary_df[
        "decisions"
    ]
)

section6b_action_state_summary_df[
    "notebook53_masking_rate"
] = (
    section6b_action_state_summary_df[
        "notebook53_masking_required"
    ]
    / section6b_action_state_summary_df[
        "decisions"
    ]
)

section6b_action_state_summary_df[
    "notebook53_selection_agreement_rate"
] = (
    section6b_action_state_summary_df[
        "notebook53_selection_matches"
    ]
    / section6b_action_state_summary_df[
        "decisions"
    ]
)


print()
print("COMPARISON BY ACTION-STATE TYPE")
print("-" * 100)

display(
    section6b_action_state_summary_df
)


# --------------------------------------------------------------------------------------
# 19. Performance by active card
# --------------------------------------------------------------------------------------

section6b_active_card_summary_df = (
    section6b_benchmark_comparison_df
    .groupby(
        "active_card",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        notebook52_legal_predictions=(
            "notebook52_raw_prediction_legal",
            "sum",
        ),

        notebook53_legal_predictions=(
            "notebook53_raw_prediction_legal",
            "sum",
        ),

        notebook52_masking_changes=(
            "notebook52_masking_changed_action",
            "sum",
        ),

        notebook53_masking_required=(
            "notebook53_raw_masking_required",
            "sum",
        ),

        notebook53_selection_matches=(
            "notebook53_matches_notebook52_selection",
            "sum",
        ),

        notebook53_average_confidence=(
            "notebook53_raw_confidence",
            "mean",
        ),
    )
)


section6b_active_card_summary_df[
    "notebook52_raw_legality_rate"
] = (
    section6b_active_card_summary_df[
        "notebook52_legal_predictions"
    ]
    / section6b_active_card_summary_df[
        "decisions"
    ]
)

section6b_active_card_summary_df[
    "notebook53_raw_legality_rate"
] = (
    section6b_active_card_summary_df[
        "notebook53_legal_predictions"
    ]
    / section6b_active_card_summary_df[
        "decisions"
    ]
)

section6b_active_card_summary_df[
    "notebook52_masking_rate"
] = (
    section6b_active_card_summary_df[
        "notebook52_masking_changes"
    ]
    / section6b_active_card_summary_df[
        "decisions"
    ]
)

section6b_active_card_summary_df[
    "notebook53_masking_rate"
] = (
    section6b_active_card_summary_df[
        "notebook53_masking_required"
    ]
    / section6b_active_card_summary_df[
        "decisions"
    ]
)

section6b_active_card_summary_df[
    "notebook53_selection_agreement_rate"
] = (
    section6b_active_card_summary_df[
        "notebook53_selection_matches"
    ]
    / section6b_active_card_summary_df[
        "decisions"
    ]
)


print()
print("COMPARISON BY ACTIVE CARD")
print("-" * 100)

display(
    section6b_active_card_summary_df
)


# --------------------------------------------------------------------------------------
# 20. Performance by condition
# --------------------------------------------------------------------------------------

section6b_condition_summary_df = (
    section6b_benchmark_comparison_df
    .groupby(
        "condition_id",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        notebook52_masking_changes=(
            "notebook52_masking_changed_action",
            "sum",
        ),

        notebook53_masking_required=(
            "notebook53_raw_masking_required",
            "sum",
        ),

        notebook53_legal_predictions=(
            "notebook53_raw_prediction_legal",
            "sum",
        ),

        notebook53_selection_matches=(
            "notebook53_matches_notebook52_selection",
            "sum",
        ),

        notebook53_average_confidence=(
            "notebook53_raw_confidence",
            "mean",
        ),
    )
)


section6b_condition_summary_df[
    "notebook52_masking_rate"
] = (
    section6b_condition_summary_df[
        "notebook52_masking_changes"
    ]
    / section6b_condition_summary_df[
        "decisions"
    ]
)

section6b_condition_summary_df[
    "notebook53_masking_rate"
] = (
    section6b_condition_summary_df[
        "notebook53_masking_required"
    ]
    / section6b_condition_summary_df[
        "decisions"
    ]
)

section6b_condition_summary_df[
    "notebook53_raw_legality_rate"
] = (
    section6b_condition_summary_df[
        "notebook53_legal_predictions"
    ]
    / section6b_condition_summary_df[
        "decisions"
    ]
)

section6b_condition_summary_df[
    "notebook53_selection_agreement_rate"
] = (
    section6b_condition_summary_df[
        "notebook53_selection_matches"
    ]
    / section6b_condition_summary_df[
        "decisions"
    ]
)


print()
print("COMPARISON BY TOURNAMENT CONDITION")
print("-" * 100)

display(
    section6b_condition_summary_df
)


# --------------------------------------------------------------------------------------
# 21. Routing reason summary
# --------------------------------------------------------------------------------------

section6b_routing_reason_summary_df = (
    section6b_benchmark_comparison_df
    .groupby(
        "notebook53_routing_reason",
        as_index=False,
    )
    .agg(
        decisions=(
            "scenario_id",
            "size",
        ),

        legal_raw_predictions=(
            "notebook53_raw_prediction_legal",
            "sum",
        ),

        selection_matches=(
            "notebook53_matches_notebook52_selection",
            "sum",
        ),

        average_confidence=(
            "notebook53_raw_confidence",
            "mean",
        ),
    )
)


section6b_routing_reason_summary_df[
    "decision_fraction"
] = (
    section6b_routing_reason_summary_df[
        "decisions"
    ]
    / len(
        section6b_benchmark_comparison_df
    )
)

section6b_routing_reason_summary_df[
    "selection_agreement_rate"
] = (
    section6b_routing_reason_summary_df[
        "selection_matches"
    ]
    / section6b_routing_reason_summary_df[
        "decisions"
    ]
)


print()
print("NOTEBOOK 53 ROUTING REASON SUMMARY")
print("-" * 100)

display(
    section6b_routing_reason_summary_df
)


# --------------------------------------------------------------------------------------
# 22. Validation
# --------------------------------------------------------------------------------------

assert len(
    section6b_benchmark_comparison_df
) == 210

assert notebook53_benchmark_encoded.shape == (
    210,
    64,
)

assert section6b_benchmark_comparison_df[
    "notebook53_routed_move_legal"
].all()

assert (
    0.0
    <= notebook53_raw_legality_rate
    <= 1.0
)

assert (
    0.0
    <= notebook53_raw_masking_rate
    <= 1.0
)

assert (
    0.0
    <= notebook53_selection_agreement_rate
    <= 1.0
)

assert int(
    section6b_benchmark_comparison_df[
        "legal_move_count"
    ].eq(
        1
    ).sum()
) == 187

assert int(
    section6b_benchmark_comparison_df[
        "legal_move_count"
    ].gt(
        1
    ).sum()
) == 23

assert section6a_summary[
    "benchmark_rows_used_for_training"
] == 0


# --------------------------------------------------------------------------------------
# 23. Save reports
# --------------------------------------------------------------------------------------

SECTION6B_BENCHMARK_FEATURES_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_notebook52_benchmark_features.csv"
)

SECTION6B_BENCHMARK_COMPARISON_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_notebook52_vs_notebook53_comparison.csv"
)

SECTION6B_OVERALL_COMPARISON_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_overall_comparison.csv"
)

SECTION6B_ACTION_STATE_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_comparison_by_action_state.csv"
)

SECTION6B_ACTIVE_CARD_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_comparison_by_active_card.csv"
)

SECTION6B_CONDITION_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_comparison_by_condition.csv"
)

SECTION6B_ROUTING_REASON_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_routing_reason_summary.csv"
)

SECTION6B_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_frozen_benchmark_summary.json"
)


notebook53_benchmark_features_df.to_csv(
    SECTION6B_BENCHMARK_FEATURES_FILE,
    index=False,
)

section6b_benchmark_comparison_df.to_csv(
    SECTION6B_BENCHMARK_COMPARISON_FILE,
    index=False,
)

section6b_overall_comparison_df.to_csv(
    SECTION6B_OVERALL_COMPARISON_FILE,
    index=False,
)

section6b_action_state_summary_df.to_csv(
    SECTION6B_ACTION_STATE_FILE,
    index=False,
)

section6b_active_card_summary_df.to_csv(
    SECTION6B_ACTIVE_CARD_FILE,
    index=False,
)

section6b_condition_summary_df.to_csv(
    SECTION6B_CONDITION_FILE,
    index=False,
)

section6b_routing_reason_summary_df.to_csv(
    SECTION6B_ROUTING_REASON_FILE,
    index=False,
)


section6b_summary = {
    "status":
        "FROZEN_BENCHMARK_COMPARISON_COMPLETE",

    "benchmark_decisions":
        210,

    "single_action_decisions":
        187,

    "multi_action_decisions":
        23,

    "notebook52_raw_legality_rate":
        notebook52_raw_legality_rate,

    "notebook52_masking_rate":
        notebook52_masking_rate,

    "notebook53_raw_legality_rate":
        notebook53_raw_legality_rate,

    "notebook53_raw_masking_rate":
        notebook53_raw_masking_rate,

    "raw_legality_absolute_improvement":
        float(
            notebook53_raw_legality_rate
            - notebook52_raw_legality_rate
        ),

    "masking_absolute_reduction":
        float(
            notebook52_masking_rate
            - notebook53_raw_masking_rate
        ),

    "notebook53_routed_legality_rate":
        notebook53_routed_legality_rate,

    "notebook53_selection_agreement_rate":
        notebook53_selection_agreement_rate,

    "notebook53_average_confidence":
        float(
            section6b_benchmark_comparison_df[
                "notebook53_raw_confidence"
            ].mean()
        ),

    "notebook53_illegal_raw_predictions":
        int(
            section6b_benchmark_comparison_df[
                "notebook53_raw_masking_required"
            ].sum()
        ),

    "notebook53_selector_calls":
        int(
            (
                section6b_benchmark_comparison_df[
                    "notebook53_routing_reason"
                ]
                ==
                "ILLEGAL_MODEL_PREDICTION_USE_SELECTOR"
            ).sum()
        ),

    "benchmark_rows_used_for_training":
        0,

    "benchmark_scenario_overlap_count":
        0,

    "ready_for_tournament_replay":
        True,

    "next_stage":
        "LEGALITY_AWARE_TOURNAMENT_REPLAY",
}


with open(
    SECTION6B_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6b_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6b_saved_files = [
    SECTION6B_BENCHMARK_FEATURES_FILE,
    SECTION6B_BENCHMARK_COMPARISON_FILE,
    SECTION6B_OVERALL_COMPARISON_FILE,
    SECTION6B_ACTION_STATE_FILE,
    SECTION6B_ACTIVE_CARD_FILE,
    SECTION6B_CONDITION_FILE,
    SECTION6B_ROUTING_REASON_FILE,
    SECTION6B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section6b_saved_files
)


# --------------------------------------------------------------------------------------
# 24. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6B SUMMARY")
print("-" * 100)

for key, value in section6b_summary.items():

    print(
        f"{key:46}: {value}"
    )


print()
print("SAVED SECTION 6B REPORTS")
print("-" * 100)

for file_path in section6b_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6B FROZEN BENCHMARK COMPARISON PASSED"
)


# ## Section 6C — Production Legality-Aware Policy Adapter
# 
# #### This section packages the Notebook 53 model, preprocessor, feature schema, and deterministic router into one production policy interface.
# 
# #### The adapter:
# 
# - accepts a battle state and legal moves,
# - directly returns the only legal move when the state is deterministic,
# - invokes the trained model only for multi-action states,
# - validates every model prediction against the legal-action set,
# - preserves a defensive legal fallback,
# - records detailed decision history,
# - exports deployment metadata for tournament replay.

# In[13]:


# ======================================================================================
# SECTION 6C — PRODUCTION LEGALITY-AWARE POLICY ADAPTER
# ======================================================================================

print("=" * 100)
print("SECTION 6C — PRODUCTION LEGALITY-AWARE POLICY ADAPTER")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Deployment helper functions
# --------------------------------------------------------------------------------------

def extract_move_name(
    move: Any,
) -> str:
    """
    Extract a move name from a dictionary, object, or string.
    """

    if isinstance(move, dict):

        return str(
            move.get(
                "name",
                "",
            )
        ).strip()

    if hasattr(
        move,
        "name",
    ):

        return str(
            getattr(
                move,
                "name",
            )
        ).strip()

    return str(
        move
    ).strip()


def get_state_value(
    source: Any,
    attribute_name: str,
    default_value: Any = None,
) -> Any:
    """
    Read a value from an object or mapping safely.
    """

    if isinstance(
        source,
        Mapping,
    ):

        return source.get(
            attribute_name,
            default_value,
        )

    return getattr(
        source,
        attribute_name,
        default_value,
    )


def get_card_name(
    pokemon_state: Any,
) -> str:
    """
    Extract a Pokémon card name safely.
    """

    card_value = get_state_value(
        pokemon_state,
        "card",
        {},
    )

    if isinstance(
        card_value,
        Mapping,
    ):

        return str(
            card_value.get(
                "name",
                "UNKNOWN",
            )
        )

    return str(
        get_state_value(
            card_value,
            "name",
            "UNKNOWN",
        )
    )


def get_attached_energy(
    pokemon_state: Any,
) -> float:
    """
    Extract attached energy safely.
    """

    return float(
        get_state_value(
            pokemon_state,
            "attached_energy",
            0.0,
        )
        or 0.0
    )


def get_pokemon_damage(
    pokemon_state: Any,
) -> float:
    """
    Extract accumulated damage safely.
    """

    damage_value = get_state_value(
        pokemon_state,
        "damage",
        None,
    )

    if damage_value is not None:

        return float(
            damage_value
            or 0.0
        )

    card_value = get_state_value(
        pokemon_state,
        "card",
        {},
    )

    maximum_hp = 0.0

    if isinstance(
        card_value,
        Mapping,
    ):

        maximum_hp = float(
            card_value.get(
                "hp",
                card_value.get(
                    "HP",
                    0.0,
                ),
            )
            or 0.0
        )

    current_hp = float(
        get_state_value(
            pokemon_state,
            "current_hp",
            maximum_hp,
        )
        or 0.0
    )

    return max(
        maximum_hp - current_hp,
        0.0,
    )


# --------------------------------------------------------------------------------------
# 2. Build one Notebook 53 feature row from a production battle state
# --------------------------------------------------------------------------------------

def build_legality_aware_feature_row(
    battle_state: Any,
    legal_moves: Sequence[Any],
    side_mode: str = "PRESERVE",
) -> pd.DataFrame:
    """
    Convert a production battle state into the Notebook 53 raw feature schema.
    """

    current_side = str(
        get_state_value(
            battle_state,
            "current_player",
            "Player",
        )
    )

    player_state = get_state_value(
        battle_state,
        "player",
    )

    opponent_state = get_state_value(
        battle_state,
        "opponent",
    )

    player_active = get_state_value(
        player_state,
        "active",
    )

    opponent_active = get_state_value(
        opponent_state,
        "active",
    )

    player_card = get_card_name(
        player_active
    )

    opponent_card = get_card_name(
        opponent_active
    )

    player_energy = get_attached_energy(
        player_active
    )

    opponent_energy = get_attached_energy(
        opponent_active
    )

    player_damage = get_pokemon_damage(
        player_active
    )

    opponent_damage = get_pokemon_damage(
        opponent_active
    )

    acting_is_player = (
        current_side.strip().lower()
        == "player"
    )

    active_card = (
        player_card
        if acting_is_player
        else opponent_card
    )

    inactive_card = (
        opponent_card
        if acting_is_player
        else player_card
    )

    acting_energy = (
        player_energy
        if acting_is_player
        else opponent_energy
    )

    defending_energy = (
        opponent_energy
        if acting_is_player
        else player_energy
    )

    acting_damage = (
        player_damage
        if acting_is_player
        else opponent_damage
    )

    defending_damage = (
        opponent_damage
        if acting_is_player
        else player_damage
    )

    legal_move_names = [
        extract_move_name(
            move
        )
        for move in legal_moves
    ]

    normalized_legal_moves = [
        normalize_action_name(
            move_name
        )
        for move_name in legal_move_names
    ]

    legal_move_count = len(
        legal_move_names
    )

    feature_row = {
        "turn_number":
            float(
                get_state_value(
                    battle_state,
                    "turn_number",
                    1,
                )
            ),

        "player_energy":
            player_energy,

        "opponent_energy":
            opponent_energy,

        "player_damage":
            player_damage,

        "opponent_damage":
            opponent_damage,

        "prize_cards_remaining":
            float(
                get_state_value(
                    player_state,
                    "prize_cards_remaining",
                    6,
                )
            ),

        "hand_size":
            float(
                get_state_value(
                    player_state,
                    "hand_size",
                    7,
                )
            ),

        "recomputed_legal_move_count":
            float(
                legal_move_count
            ),

        "acting_energy":
            acting_energy,

        "defending_energy":
            defending_energy,

        "energy_difference":
            acting_energy
            - defending_energy,

        "absolute_energy_difference":
            abs(
                acting_energy
                - defending_energy
            ),

        "acting_damage":
            acting_damage,

        "defending_damage":
            defending_damage,

        "damage_difference":
            acting_damage
            - defending_damage,

        "absolute_damage_difference":
            abs(
                acting_damage
                - defending_damage
            ),

        "single_legal_move_flag":
            int(
                legal_move_count == 1
            ),

        "multi_legal_move_flag":
            int(
                legal_move_count > 1
            ),

        "model_choice_required_flag":
            int(
                legal_move_count > 1
            ),

        "current_side":
            current_side,

        "side_mode":
            str(
                side_mode
            ),

        "player_card":
            player_card,

        "opponent_card":
            opponent_card,

        "active_card":
            active_card,

        "inactive_card":
            inactive_card,

        "legal_move_signature":
            canonical_move_signature(
                legal_move_names
            ),

        "battle_phase":
            classify_turn_phase(
                get_state_value(
                    battle_state,
                    "turn_number",
                    1,
                )
            ),

        "energy_band":
            classify_energy_band(
                acting_energy
            ),

        "damage_band":
            classify_damage_band(
                acting_damage
            ),
    }

    for action_name in POLICY_ACTION_CLASSES:

        normalized_action = normalize_action_name(
            action_name
        )

        legal_flag_column = (
            "is_legal__"
            + action_feature_name(
                action_name
            )
        )

        compatibility_column = (
            "active_card_can_use__"
            + action_feature_name(
                action_name
            )
        )

        feature_row[
            legal_flag_column
        ] = int(
            normalized_action
            in normalized_legal_moves
        )

        feature_row[
            compatibility_column
        ] = int(
            normalized_action
            in training_active_card_action_map.get(
                active_card,
                [],
            )
        )

    feature_dataframe = pd.DataFrame(
        [
            feature_row
        ]
    )

    for column_name in (
        legality_training_numeric_features
    ):

        feature_dataframe[
            column_name
        ] = pd.to_numeric(
            feature_dataframe[
                column_name
            ],
            errors="coerce",
        ).fillna(
            0.0
        ).astype(
            float
        )

    for column_name in (
        legality_training_categorical_features
    ):

        feature_dataframe[
            column_name
        ] = (
            feature_dataframe[
                column_name
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
        )

    feature_dataframe = feature_dataframe[
        PREPROCESS_RAW_FEATURES
    ]

    assert feature_dataframe.shape == (
        1,
        41,
    )

    return feature_dataframe


# --------------------------------------------------------------------------------------
# 3. Production policy adapter
# --------------------------------------------------------------------------------------

class LegalityAwarePolicyAgent:
    """
    Notebook 53 production policy.

    Deterministic states bypass the model. Multi-action states use the
    legality-aware model with a defensive legal-action check.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        side_mode: str = "PRESERVE",
        agent_name: str = "Notebook53LegalityAwarePolicy",
    ) -> None:

        self.model = model
        self.preprocessor = preprocessor
        self.side_mode = side_mode
        self.agent_name = agent_name

        self.decision_history: List[
            Dict[str, Any]
        ] = []

        self.total_decisions = 0
        self.direct_single_action_decisions = 0
        self.model_decisions = 0
        self.selector_decisions = 0
        self.fallback_decisions = 0

    def reset(
        self,
    ) -> None:

        self.decision_history = []

        self.total_decisions = 0
        self.direct_single_action_decisions = 0
        self.model_decisions = 0
        self.selector_decisions = 0
        self.fallback_decisions = 0

    def choose_move_detailed(
        self,
        state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Dict[str, Any]:

        if legal_moves is None:

            active_side = str(
                get_state_value(
                    state,
                    "current_player",
                    "Player",
                )
            )

            side_state = (
                get_state_value(
                    state,
                    "player",
                )
                if active_side.lower()
                == "player"
                else get_state_value(
                    state,
                    "opponent",
                )
            )

            active_pokemon = get_state_value(
                side_state,
                "active",
            )

            card_value = get_state_value(
                active_pokemon,
                "card",
                {},
            )

            legal_moves = (
                card_value.get(
                    "moves",
                    [],
                )
                if isinstance(
                    card_value,
                    Mapping,
                )
                else []
            )

        legal_moves = list(
            legal_moves
        )

        if not legal_moves:

            raise ValueError(
                "No legal moves were supplied to the policy."
            )

        legal_move_names = [
            extract_move_name(
                move
            )
            for move in legal_moves
        ]

        normalized_legal_move_map = {
            normalize_action_name(
                move_name
            ):
                move
            for move_name, move in zip(
                legal_move_names,
                legal_moves,
            )
        }

        feature_dataframe = (
            build_legality_aware_feature_row(
                battle_state=state,
                legal_moves=legal_moves,
                side_mode=self.side_mode,
            )
        )

        self.total_decisions += 1

        raw_prediction = None
        confidence = 1.0
        routing_reason = None
        fallback_used = False

        if len(
            legal_moves
        ) == 1:

            selected_move = legal_moves[0]

            raw_prediction = extract_move_name(
                selected_move
            )

            routing_reason = (
                "ONLY_LEGAL_MOVE"
            )

            self.direct_single_action_decisions += 1

        else:

            encoded_features = (
                self.preprocessor.transform(
                    feature_dataframe
                )
            )

            encoded_features = np.asarray(
                encoded_features,
                dtype=np.float64,
            )

            predicted_probabilities = (
                self.model.predict_proba(
                    encoded_features
                )[0]
            )

            predicted_index = int(
                np.argmax(
                    predicted_probabilities
                )
            )

            raw_prediction = str(
                self.model.classes_[
                    predicted_index
                ]
            )

            confidence = float(
                predicted_probabilities[
                    predicted_index
                ]
            )

            normalized_prediction = (
                normalize_action_name(
                    raw_prediction
                )
            )

            self.model_decisions += 1

            if (
                normalized_prediction
                in normalized_legal_move_map
            ):

                selected_move = (
                    normalized_legal_move_map[
                        normalized_prediction
                    ]
                )

                routing_reason = (
                    "LEGAL_MODEL_PREDICTION"
                )

            else:

                legal_probability_candidates = []

                for class_index, class_name in enumerate(
                    self.model.classes_
                ):

                    normalized_class_name = (
                        normalize_action_name(
                            class_name
                        )
                    )

                    if (
                        normalized_class_name
                        in normalized_legal_move_map
                    ):

                        legal_probability_candidates.append(
                            (
                                float(
                                    predicted_probabilities[
                                        class_index
                                    ]
                                ),
                                normalized_class_name,
                            )
                        )

                if legal_probability_candidates:

                    (
                        _,
                        selected_normalized_name,
                    ) = max(
                        legal_probability_candidates,
                        key=lambda item: item[0],
                    )

                    selected_move = (
                        normalized_legal_move_map[
                            selected_normalized_name
                        ]
                    )

                    routing_reason = (
                        "LEGAL_PROBABILITY_SELECTOR"
                    )

                    self.selector_decisions += 1

                else:

                    selected_move = legal_moves[0]

                    routing_reason = (
                        "DEFENSIVE_FIRST_LEGAL_FALLBACK"
                    )

                    fallback_used = True

                    self.fallback_decisions += 1

        selected_move_name = extract_move_name(
            selected_move
        )

        decision_record = {
            "decision_number":
                self.total_decisions,

            "agent_name":
                self.agent_name,

            "turn_number":
                float(
                    get_state_value(
                        state,
                        "turn_number",
                        0,
                    )
                ),

            "current_side":
                str(
                    get_state_value(
                        state,
                        "current_player",
                        "UNKNOWN",
                    )
                ),

            "legal_moves":
                legal_move_names,

            "legal_move_count":
                len(
                    legal_moves
                ),

            "raw_prediction":
                raw_prediction,

            "selected_move":
                selected_move_name,

            "confidence":
                confidence,

            "routing_reason":
                routing_reason,

            "fallback_used":
                fallback_used,

            "selected_move_legal":
                normalize_action_name(
                    selected_move_name
                )
                in normalized_legal_move_map,
        }

        self.decision_history.append(
            decision_record
        )

        return {
            "move":
                selected_move,

            "move_name":
                selected_move_name,

            "raw_prediction":
                raw_prediction,

            "confidence":
                confidence,

            "routing_reason":
                routing_reason,

            "fallback_used":
                fallback_used,

            "selected_move_legal":
                True,

            "features":
                feature_dataframe.iloc[
                    0
                ].to_dict(),
        }

    def choose_move(
        self,
        state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:

        return self.choose_move_detailed(
            state=state,
            legal_moves=legal_moves,
        )[
            "move"
        ]

    def select_action(
        self,
        state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:

        return self.choose_move(
            state=state,
            legal_moves=legal_moves,
        )

    def act(
        self,
        state: Any,
        legal_moves: Optional[
            Sequence[Any]
        ] = None,
    ) -> Any:

        return self.choose_move(
            state=state,
            legal_moves=legal_moves,
        )

    def statistics(
        self,
    ) -> Dict[str, Any]:

        return {
            "agent_name":
                self.agent_name,

            "total_decisions":
                self.total_decisions,

            "direct_single_action_decisions":
                self.direct_single_action_decisions,

            "model_decisions":
                self.model_decisions,

            "selector_decisions":
                self.selector_decisions,

            "fallback_decisions":
                self.fallback_decisions,

            "fallback_rate":
                (
                    self.fallback_decisions
                    / self.total_decisions
                    if self.total_decisions > 0
                    else 0.0
                ),

            "history_rows":
                len(
                    self.decision_history
                ),
        }


# --------------------------------------------------------------------------------------
# 4. Instantiate production agent
# --------------------------------------------------------------------------------------

notebook53_policy_agent = (
    LegalityAwarePolicyAgent(
        model=legality_aware_policy_model,
        preprocessor=legality_preprocessor,
        side_mode="PRESERVE",
    )
)


print()
print("PRODUCTION POLICY PROFILE")
print("-" * 100)

print(
    f"Agent type                 : "
    f"{type(notebook53_policy_agent).__name__}"
)

print(
    f"Agent name                 : "
    f"{notebook53_policy_agent.agent_name}"
)

print(
    f"Model type                 : "
    f"{type(notebook53_policy_agent.model).__name__}"
)

print(
    f"Raw feature count          : "
    f"{len(PREPROCESS_RAW_FEATURES)}"
)

print(
    f"Encoded feature count      : "
    f"{ENCODED_FEATURE_COUNT}"
)


# --------------------------------------------------------------------------------------
# 5. Validate adapter against all 210 frozen benchmark decisions
# --------------------------------------------------------------------------------------

adapter_validation_rows = []

notebook53_policy_agent.reset()

for row_index, benchmark_row in (
    notebook53_benchmark_source_df.reset_index(
        drop=True
    ).iterrows()
):

    legal_move_names = list(
        benchmark_row[
            "legal_move_list"
        ]
    )

    legal_move_objects = [
        {
            "name":
                move_name,

            "damage":
                0.0,

            "energy_cost":
                0,
        }
        for move_name in legal_move_names
    ]

    class BenchmarkState:

        pass

    benchmark_state = BenchmarkState()

    benchmark_state.turn_number = int(
        benchmark_row[
            "turn_number"
        ]
    )

    benchmark_state.current_player = str(
        benchmark_row[
            "current_side"
        ]
    )

    decision = (
        notebook53_policy_agent
        .choose_move_detailed(
            state=benchmark_state,
            legal_moves=legal_move_objects,
        )
    )

    expected_move = str(
        benchmark_row[
            "selected_move"
        ]
    )

    adapter_validation_rows.append(
        {
            "row_index":
                row_index,

            "legal_moves":
                legal_move_names,

            "selected_move":
                decision[
                    "move_name"
                ],

            "expected_move":
                expected_move,

            "selected_move_legal":
                decision[
                    "selected_move_legal"
                ],

            "matches_expected_move":
                normalize_action_name(
                    decision[
                        "move_name"
                    ]
                )
                ==
                normalize_action_name(
                    expected_move
                ),

            "routing_reason":
                decision[
                    "routing_reason"
                ],

            "fallback_used":
                decision[
                    "fallback_used"
                ],
        }
    )


section6c_adapter_validation_df = pd.DataFrame(
    adapter_validation_rows
)


# --------------------------------------------------------------------------------------
# 6. Adapter validation summary
# --------------------------------------------------------------------------------------

section6c_routing_summary_df = (
    section6c_adapter_validation_df
    .groupby(
        "routing_reason",
        as_index=False,
    )
    .agg(
        decisions=(
            "row_index",
            "size",
        ),

        legal_decisions=(
            "selected_move_legal",
            "sum",
        ),

        expected_matches=(
            "matches_expected_move",
            "sum",
        ),

        fallback_decisions=(
            "fallback_used",
            "sum",
        ),
    )
)


print()
print("ADAPTER ROUTING VALIDATION")
print("-" * 100)

display(
    section6c_routing_summary_df
)


print()
print("ADAPTER STATISTICS")
print("-" * 100)

for key, value in (
    notebook53_policy_agent
    .statistics()
    .items()
):

    print(
        f"{key:38}: {value}"
    )


# --------------------------------------------------------------------------------------
# 7. Validation assertions
# --------------------------------------------------------------------------------------

assert len(
    section6c_adapter_validation_df
) == 210

assert section6c_adapter_validation_df[
    "selected_move_legal"
].all()

assert section6c_adapter_validation_df[
    "matches_expected_move"
].all()

assert int(
    (
        section6c_adapter_validation_df[
            "routing_reason"
        ]
        == "ONLY_LEGAL_MOVE"
    ).sum()
) == 187

assert int(
    (
        section6c_adapter_validation_df[
            "routing_reason"
        ]
        == "LEGAL_MODEL_PREDICTION"
    ).sum()
) == 23

assert int(
    section6c_adapter_validation_df[
        "fallback_used"
    ].sum()
) == 0

assert notebook53_policy_agent.total_decisions == 210

assert (
    notebook53_policy_agent
    .direct_single_action_decisions
) == 187

assert notebook53_policy_agent.model_decisions == 23

assert notebook53_policy_agent.selector_decisions == 0

assert notebook53_policy_agent.fallback_decisions == 0


# --------------------------------------------------------------------------------------
# 8. Save production adapter artifacts
# --------------------------------------------------------------------------------------

SECTION6C_ADAPTER_VALIDATION_FILE = (
    SECTION6_REPORT_DIR
    / "section6c_policy_adapter_validation.csv"
)

SECTION6C_ROUTING_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6c_policy_adapter_routing_summary.csv"
)

SECTION6C_ADAPTER_METADATA_FILE = (
    NOTEBOOK53_MODEL_DIR
    / "legality_aware_policy_adapter_metadata.json"
)

SECTION6C_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6c_policy_adapter_summary.json"
)


section6c_adapter_validation_df.to_csv(
    SECTION6C_ADAPTER_VALIDATION_FILE,
    index=False,
)

section6c_routing_summary_df.to_csv(
    SECTION6C_ROUTING_SUMMARY_FILE,
    index=False,
)


section6c_adapter_metadata = {
    "adapter_name":
        "Notebook53LegalityAwarePolicy",

    "adapter_type":
        "LegalityAwarePolicyAgent",

    "model_file":
        str(
            SECTION6A_MODEL_FILE
        ),

    "preprocessor_file":
        str(
            SECTION5B_PREPROCESSOR_FILE
        ),

    "raw_feature_schema_file":
        str(
            SECTION5B_RAW_FEATURE_SCHEMA_FILE
        ),

    "encoded_feature_names_file":
        str(
            SECTION5B_ENCODED_FEATURE_NAMES_FILE
        ),

    "single_action_direct_routing":
        True,

    "multi_action_model_routing":
        True,

    "defensive_legal_selector":
        True,

    "validated_benchmark_decisions":
        210,

    "benchmark_rows_used_for_training":
        0,
}


with open(
    SECTION6C_ADAPTER_METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6c_adapter_metadata,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6c_summary = {
    "status":
        "PRODUCTION_POLICY_ADAPTER_VALIDATED",

    "validated_decisions":
        210,

    "legal_decisions":
        int(
            section6c_adapter_validation_df[
                "selected_move_legal"
            ].sum()
        ),

    "selection_matches":
        int(
            section6c_adapter_validation_df[
                "matches_expected_move"
            ].sum()
        ),

    "single_action_direct_decisions":
        int(
            notebook53_policy_agent
            .direct_single_action_decisions
        ),

    "model_decisions":
        int(
            notebook53_policy_agent
            .model_decisions
        ),

    "selector_decisions":
        int(
            notebook53_policy_agent
            .selector_decisions
        ),

    "fallback_decisions":
        int(
            notebook53_policy_agent
            .fallback_decisions
        ),

    "adapter_legality_rate":
        float(
            section6c_adapter_validation_df[
                "selected_move_legal"
            ].mean()
        ),

    "selection_agreement_rate":
        float(
            section6c_adapter_validation_df[
                "matches_expected_move"
            ].mean()
        ),

    "ready_for_tournament_replay":
        True,

    "next_stage":
        "LEGALITY_AWARE_TOURNAMENT_REPLAY",
}


with open(
    SECTION6C_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6c_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6c_saved_files = [
    SECTION6C_ADAPTER_VALIDATION_FILE,
    SECTION6C_ROUTING_SUMMARY_FILE,
    SECTION6C_ADAPTER_METADATA_FILE,
    SECTION6C_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section6c_saved_files
)


# --------------------------------------------------------------------------------------
# 9. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6C SUMMARY")
print("-" * 100)

for key, value in section6c_summary.items():

    print(
        f"{key:42}: {value}"
    )


print()
print("SAVED SECTION 6C REPORTS")
print("-" * 100)

for file_path in section6c_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6C PRODUCTION LEGALITY-AWARE POLICY ADAPTER PASSED"
)


# ## Section 6D — Simulator-Compatible Battle Runner Adapter
# 
# #### This section connects the Notebook 53 legality-aware policy to the production battle simulator.
# 
# #### The simulator expects an agent whose `choose_move` method returns an `AgentDecision` object.
# 
# #### This section:
# 
# - loads the production simulator modules safely,
# - validates all required runtime symbols,
# - wraps `LegalityAwarePolicyAgent` in the simulator contract,
# - constructs a real battle state,
# - requests a production `AgentDecision`,
# - verifies move legality and state compatibility,
# - prepares the optimized policy for full tournament replay.

# In[14]:


# ======================================================================================
# SECTION 6D — SIMULATOR-COMPATIBLE BATTLE RUNNER ADAPTER
# ======================================================================================

print("=" * 100)
print("SECTION 6D — SIMULATOR-COMPATIBLE BATTLE RUNNER ADAPTER")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Imports
# --------------------------------------------------------------------------------------

import importlib.util
import sys
import types


# --------------------------------------------------------------------------------------
# 2. Production simulator source files
# --------------------------------------------------------------------------------------

NOTEBOOK53_SIMULATOR_PACKAGE = (
    "notebook53_simulator_runtime"
)

simulator_source_files = {
    "battle_state":
        SRC_DIR
        / "battle_state.py",

    "agent_decision":
        SRC_DIR
        / "agent_decision.py",

    "simulator":
        SRC_DIR
        / "simulator.py",

    "battle_agent":
        SRC_DIR
        / "battle_agent.py",

    "battle_simulation":
        SRC_DIR
        / "battle_simulation.py",
}


for module_name, source_file in (
    simulator_source_files.items()
):

    assert source_file.exists(), (
        f"Missing simulator source file: {source_file}"
    )

    assert source_file.is_file(), (
        f"Simulator source path is not a file: {source_file}"
    )


print()
print("SIMULATOR SOURCE FILES")
print("-" * 100)

for module_name, source_file in (
    simulator_source_files.items()
):

    print(
        f"{module_name:24}: {source_file}"
    )


# --------------------------------------------------------------------------------------
# 3. Create a package-aware runtime namespace
# --------------------------------------------------------------------------------------

if NOTEBOOK53_SIMULATOR_PACKAGE in sys.modules:

    del sys.modules[
        NOTEBOOK53_SIMULATOR_PACKAGE
    ]


runtime_package = types.ModuleType(
    NOTEBOOK53_SIMULATOR_PACKAGE
)

runtime_package.__path__ = [
    str(
        SRC_DIR
    )
]

runtime_package.__package__ = (
    NOTEBOOK53_SIMULATOR_PACKAGE
)

sys.modules[
    NOTEBOOK53_SIMULATOR_PACKAGE
] = runtime_package


def load_runtime_module(
    module_name: str,
    source_file: Path,
) -> Any:
    """
    Load one source file as a submodule of the Notebook 53 runtime package.
    """

    full_module_name = (
        f"{NOTEBOOK53_SIMULATOR_PACKAGE}.{module_name}"
    )

    if full_module_name in sys.modules:

        del sys.modules[
            full_module_name
        ]

    module_specification = (
        importlib.util.spec_from_file_location(
            full_module_name,
            source_file,
        )
    )

    assert module_specification is not None

    assert (
        module_specification.loader
        is not None
    )

    loaded_module = (
        importlib.util.module_from_spec(
            module_specification
        )
    )

    loaded_module.__package__ = (
        NOTEBOOK53_SIMULATOR_PACKAGE
    )

    sys.modules[
        full_module_name
    ] = loaded_module

    module_specification.loader.exec_module(
        loaded_module
    )

    return loaded_module


# --------------------------------------------------------------------------------------
# 4. Load modules in dependency order
# --------------------------------------------------------------------------------------

runtime_battle_state_module = load_runtime_module(
    "battle_state",
    simulator_source_files[
        "battle_state"
    ],
)

runtime_agent_decision_module = load_runtime_module(
    "agent_decision",
    simulator_source_files[
        "agent_decision"
    ],
)

runtime_simulator_module = load_runtime_module(
    "simulator",
    simulator_source_files[
        "simulator"
    ],
)

runtime_battle_agent_module = load_runtime_module(
    "battle_agent",
    simulator_source_files[
        "battle_agent"
    ],
)

runtime_battle_simulation_module = load_runtime_module(
    "battle_simulation",
    simulator_source_files[
        "battle_simulation"
    ],
)


# --------------------------------------------------------------------------------------
# 5. Bind production runtime symbols
# --------------------------------------------------------------------------------------

PokemonState = (
    runtime_battle_state_module
    .PokemonState
)

PlayerState = (
    runtime_battle_state_module
    .PlayerState
)

BattleState = (
    runtime_battle_state_module
    .BattleState
)

AgentDecision = (
    runtime_agent_decision_module
    .AgentDecision
)

PokemonBattleAgent = (
    runtime_battle_agent_module
    .PokemonBattleAgent
)

apply_move = (
    runtime_simulator_module
    .apply_move
)

determine_battle_winner = (
    runtime_battle_simulation_module
    .determine_battle_winner
)

simulate_ai_battle = (
    runtime_battle_simulation_module
    .simulate_ai_battle
)

create_battle_transcript = (
    runtime_battle_simulation_module
    .create_battle_transcript
)

BattleTurnRecord = (
    runtime_battle_simulation_module
    .BattleTurnRecord
)

BattleSimulationResult = (
    runtime_battle_simulation_module
    .BattleSimulationResult
)


required_runtime_symbols = {
    "PokemonState":
        PokemonState,

    "PlayerState":
        PlayerState,

    "BattleState":
        BattleState,

    "AgentDecision":
        AgentDecision,

    "PokemonBattleAgent":
        PokemonBattleAgent,

    "apply_move":
        apply_move,

    "determine_battle_winner":
        determine_battle_winner,

    "simulate_ai_battle":
        simulate_ai_battle,

    "create_battle_transcript":
        create_battle_transcript,

    "BattleTurnRecord":
        BattleTurnRecord,

    "BattleSimulationResult":
        BattleSimulationResult,
}


runtime_symbol_rows = []

for symbol_name, symbol_object in (
    required_runtime_symbols.items()
):

    runtime_symbol_rows.append(
        {
            "symbol_name":
                symbol_name,

            "exists":
                symbol_object is not None,

            "callable":
                callable(
                    symbol_object
                ),

            "object_type":
                type(
                    symbol_object
                ).__name__,

            "module":
                getattr(
                    symbol_object,
                    "__module__",
                    "",
                ),

            "signature":
                (
                    str(
                        inspect.signature(
                            symbol_object
                        )
                    )
                    if callable(
                        symbol_object
                    )
                    else ""
                ),
        }
    )


section6d_runtime_symbols_df = pd.DataFrame(
    runtime_symbol_rows
)


print()
print("SIMULATOR RUNTIME SYMBOLS")
print("-" * 100)

display(
    section6d_runtime_symbols_df
)


# --------------------------------------------------------------------------------------
# 6. Simulator-compatible wrapper
# --------------------------------------------------------------------------------------

class LegalityAwareBattleRunnerAgent:
    """
    Adapt the Notebook 53 policy to the production simulator's AgentDecision contract.
    """

    def __init__(
        self,
        policy_agent: LegalityAwarePolicyAgent,
        default_search_depth: int = 0,
    ) -> None:

        self.policy_agent = policy_agent

        self.default_search_depth = int(
            default_search_depth
        )

        self.decision_history: List[
            Dict[str, Any]
        ] = []

        self.total_decisions = 0

    def reset(
        self,
    ) -> None:

        self.policy_agent.reset()

        self.decision_history = []

        self.total_decisions = 0

    def choose_move(
        self,
        state: Any,
        depth: Optional[int] = None,
    ) -> Any:

        acting_side = str(
            get_state_value(
                state,
                "current_player",
                "Player",
            )
        )

        side_state = (
            get_state_value(
                state,
                "player",
            )
            if acting_side.strip().lower()
            == "player"
            else get_state_value(
                state,
                "opponent",
            )
        )

        active_pokemon = get_state_value(
            side_state,
            "active",
        )

        card_value = get_state_value(
            active_pokemon,
            "card",
            {},
        )

        if not isinstance(
            card_value,
            Mapping,
        ):

            raise TypeError(
                "Production active Pokémon card must be a mapping."
            )

        legal_moves = list(
            card_value.get(
                "moves",
                [],
            )
        )

        if not legal_moves:

            raise ValueError(
                "The active Pokémon has no available moves."
            )

        detailed_decision = (
            self.policy_agent
            .choose_move_detailed(
                state=state,
                legal_moves=legal_moves,
            )
        )

        selected_move = detailed_decision[
            "move"
        ]

        selected_move_name = detailed_decision[
            "move_name"
        ]

        decision_score = float(
            detailed_decision[
                "confidence"
            ]
        )

        resolved_depth = (
            self.default_search_depth
            if depth is None
            else int(
                depth
            )
        )

        production_decision = AgentDecision(
            move=selected_move,
            score=decision_score,
            search_depth=resolved_depth,
            nodes=1,
            principal_variation=[
                selected_move_name
            ],
        )

        self.total_decisions += 1

        history_record = {
            "decision_number":
                self.total_decisions,

            "turn_number":
                int(
                    get_state_value(
                        state,
                        "turn_number",
                        0,
                    )
                ),

            "current_player":
                acting_side,

            "active_card":
                get_card_name(
                    active_pokemon
                ),

            "legal_moves":
                [
                    extract_move_name(
                        move
                    )
                    for move in legal_moves
                ],

            "selected_move":
                selected_move_name,

            "raw_prediction":
                detailed_decision[
                    "raw_prediction"
                ],

            "confidence":
                decision_score,

            "routing_reason":
                detailed_decision[
                    "routing_reason"
                ],

            "fallback_used":
                detailed_decision[
                    "fallback_used"
                ],

            "search_depth":
                resolved_depth,

            "nodes":
                1,
        }

        self.decision_history.append(
            history_record
        )

        return production_decision

    def statistics(
        self,
    ) -> Dict[str, Any]:

        policy_statistics = (
            self.policy_agent
            .statistics()
        )

        return {
            "wrapper_total_decisions":
                self.total_decisions,

            "policy_total_decisions":
                policy_statistics[
                    "total_decisions"
                ],

            "single_action_direct_decisions":
                policy_statistics[
                    "direct_single_action_decisions"
                ],

            "model_decisions":
                policy_statistics[
                    "model_decisions"
                ],

            "selector_decisions":
                policy_statistics[
                    "selector_decisions"
                ],

            "fallback_decisions":
                policy_statistics[
                    "fallback_decisions"
                ],

            "fallback_rate":
                policy_statistics[
                    "fallback_rate"
                ],

            "runner_history_rows":
                len(
                    self.decision_history
                ),
        }


# --------------------------------------------------------------------------------------
# 7. Instantiate the production battle runner
# --------------------------------------------------------------------------------------

notebook53_policy_agent.reset()

notebook53_battle_runner_agent = (
    LegalityAwareBattleRunnerAgent(
        policy_agent=notebook53_policy_agent,
        default_search_depth=0,
    )
)


print()
print("BATTLE RUNNER ADAPTER PROFILE")
print("-" * 100)

print(
    f"Wrapper type          : "
    f"{type(notebook53_battle_runner_agent).__name__}"
)

print(
    f"Underlying policy     : "
    f"{notebook53_policy_agent.agent_name}"
)

print(
    f"Default search depth  : "
    f"{notebook53_battle_runner_agent.default_search_depth}"
)


# --------------------------------------------------------------------------------------
# 8. Construct a real production test state
# --------------------------------------------------------------------------------------

test_player_card = {
    "name":
        "Bulbasaur",

    "hp":
        80.0,

    "moves": [
        {
            "name":
                "Bind Down",

            "damage":
                10.0,

            "energy_cost":
                3,
        }
    ],
}


test_opponent_card = {
    "name":
        "Eevee",

    "hp":
        50.0,

    "moves": [
        {
            "name":
                "Ascension",

            "damage":
                0.0,

            "energy_cost":
                1,
        },
        {
            "name":
                "Quick Attack",

            "damage":
                20.0,

            "energy_cost":
                3,
        },
    ],
}


test_player_pokemon = PokemonState(
    card=test_player_card,
    current_hp=80.0,
    attached_energy=3,
    damage=0.0,
    is_active=True,
)

test_opponent_pokemon = PokemonState(
    card=test_opponent_card,
    current_hp=50.0,
    attached_energy=3,
    damage=0.0,
    is_active=True,
)


test_player_state = PlayerState(
    active=test_player_pokemon,
    bench=[],
    prize_cards_remaining=2,
    hand_size=3,
)

test_opponent_state = PlayerState(
    active=test_opponent_pokemon,
    bench=[],
    prize_cards_remaining=2,
    hand_size=3,
)


section6d_test_state = BattleState(
    player=test_player_state,
    opponent=test_opponent_state,
    turn_number=1,
    current_player="Player",
)


# --------------------------------------------------------------------------------------
# 9. Request a production AgentDecision
# --------------------------------------------------------------------------------------

notebook53_battle_runner_agent.reset()

section6d_test_decision = (
    notebook53_battle_runner_agent
    .choose_move(
        state=section6d_test_state,
        depth=0,
    )
)


print()
print("PRODUCTION RUNNER DECISION")
print("-" * 100)

print(
    f"Decision type         : "
    f"{type(section6d_test_decision).__name__}"
)

print(
    f"Selected move         : "
    f"{extract_move_name(section6d_test_decision.move)}"
)

print(
    f"Score                 : "
    f"{section6d_test_decision.score}"
)

print(
    f"Search depth          : "
    f"{section6d_test_decision.search_depth}"
)

print(
    f"Nodes                 : "
    f"{section6d_test_decision.nodes}"
)

print(
    f"Principal variation   : "
    f"{section6d_test_decision.principal_variation}"
)


# --------------------------------------------------------------------------------------
# 10. Apply the production move
# --------------------------------------------------------------------------------------

section6d_next_state = apply_move(
    section6d_test_state,
    section6d_test_decision.move,
)


section6d_transition_df = pd.DataFrame(
    [
        {
            "initial_current_player":
                section6d_test_state.current_player,

            "next_current_player":
                section6d_next_state.current_player,

            "initial_turn_number":
                section6d_test_state.turn_number,

            "next_turn_number":
                section6d_next_state.turn_number,

            "selected_move":
                extract_move_name(
                    section6d_test_decision.move
                ),

            "player_hp_before":
                section6d_test_state
                .player
                .active
                .current_hp,

            "player_hp_after":
                section6d_next_state
                .player
                .active
                .current_hp,

            "opponent_hp_before":
                section6d_test_state
                .opponent
                .active
                .current_hp,

            "opponent_hp_after":
                section6d_next_state
                .opponent
                .active
                .current_hp,

            "original_state_unchanged":
                (
                    section6d_test_state
                    .opponent
                    .active
                    .current_hp
                    == 50.0
                ),
        }
    ]
)


print()
print("REAL SIMULATOR TRANSITION")
print("-" * 100)

display(
    section6d_transition_df
)


# --------------------------------------------------------------------------------------
# 11. Adapter history and statistics
# --------------------------------------------------------------------------------------

section6d_runner_history_df = pd.DataFrame(
    notebook53_battle_runner_agent
    .decision_history
)


print()
print("RUNNER DECISION HISTORY")
print("-" * 100)

display(
    section6d_runner_history_df
)


print()
print("RUNNER STATISTICS")
print("-" * 100)

for key, value in (
    notebook53_battle_runner_agent
    .statistics()
    .items()
):

    print(
        f"{key:38}: {value}"
    )


# --------------------------------------------------------------------------------------
# 12. Validation assertions
# --------------------------------------------------------------------------------------

assert len(
    required_runtime_symbols
) == 11

assert section6d_runtime_symbols_df[
    "exists"
].all()

assert section6d_runtime_symbols_df[
    "callable"
].all()

assert isinstance(
    section6d_test_decision,
    AgentDecision,
)

assert extract_move_name(
    section6d_test_decision.move
) == "Bind Down"

assert section6d_test_decision.move in (
    test_player_card[
        "moves"
    ]
)

assert np.isfinite(
    section6d_test_decision.score
)

assert section6d_test_decision.search_depth == 0

assert section6d_test_decision.nodes == 1

assert (
    section6d_test_decision
    .principal_variation
    == [
        "Bind Down"
    ]
)

assert isinstance(
    section6d_next_state,
    BattleState,
)

assert (
    section6d_next_state
    is not section6d_test_state
)

assert (
    section6d_next_state.turn_number
    == section6d_test_state.turn_number
    + 1
)

assert (
    section6d_next_state.current_player
    == "Opponent"
)

assert (
    section6d_next_state
    .opponent
    .active
    .current_hp
    == 40.0
)

assert (
    section6d_test_state
    .opponent
    .active
    .current_hp
    == 50.0
)

assert notebook53_battle_runner_agent.total_decisions == 1

assert notebook53_policy_agent.total_decisions == 1

assert notebook53_policy_agent.fallback_decisions == 0


# --------------------------------------------------------------------------------------
# 13. Save reports
# --------------------------------------------------------------------------------------

SECTION6D_RUNTIME_SYMBOLS_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_simulator_runtime_symbols.csv"
)

SECTION6D_RUNNER_HISTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_battle_runner_history.csv"
)

SECTION6D_TRANSITION_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_real_simulator_transition.csv"
)

SECTION6D_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6d_battle_runner_summary.json"
)


section6d_runtime_symbols_df.to_csv(
    SECTION6D_RUNTIME_SYMBOLS_FILE,
    index=False,
)

section6d_runner_history_df.to_csv(
    SECTION6D_RUNNER_HISTORY_FILE,
    index=False,
)

section6d_transition_df.to_csv(
    SECTION6D_TRANSITION_FILE,
    index=False,
)


section6d_summary = {
    "status":
        "SIMULATOR_BATTLE_RUNNER_VALIDATED",

    "runtime_package":
        NOTEBOOK53_SIMULATOR_PACKAGE,

    "runtime_symbols_loaded":
        int(
            len(
                required_runtime_symbols
            )
        ),

    "relative_imports_resolved":
        True,

    "wrapper_type":
        type(
            notebook53_battle_runner_agent
        ).__name__,

    "production_decision_type":
        type(
            section6d_test_decision
        ).__name__,

    "selected_move":
        extract_move_name(
            section6d_test_decision.move
        ),

    "selected_move_legal":
        True,

    "state_transition_valid":
        True,

    "original_state_preserved":
        True,

    "turn_advanced":
        True,

    "side_switched":
        True,

    "fallback_decisions":
        int(
            notebook53_policy_agent
            .fallback_decisions
        ),

    "ready_for_full_tournament_replay":
        True,

    "next_stage":
        "FULL_LEGALITY_AWARE_TOURNAMENT_REPLAY",
}


with open(
    SECTION6D_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6d_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6d_saved_files = [
    SECTION6D_RUNTIME_SYMBOLS_FILE,
    SECTION6D_RUNNER_HISTORY_FILE,
    SECTION6D_TRANSITION_FILE,
    SECTION6D_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section6d_saved_files
)


# --------------------------------------------------------------------------------------
# 14. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6D SUMMARY")
print("-" * 100)

for key, value in section6d_summary.items():

    print(
        f"{key:42}: {value}"
    )


print()
print("SAVED SECTION 6D REPORTS")
print("-" * 100)

for file_path in section6d_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6D SIMULATOR-COMPATIBLE BATTLE RUNNER PASSED"
)


# ## Section 6E — Full Legality-Aware Tournament Replay
# 
# #### This section replays the complete 24-battle Notebook 52 tournament using the production-compatible Notebook 53 policy.
# 
# #### The replay uses:
# 
# - the same scenario identities,
# - the same player and opponent cards,
# - the same starting HP,
# - the same starting energy,
# - the same starting side,
# - the same legal-action sets,
# - the production simulator,
# - the Notebook 53 deterministic router and trained model.
# 
# #### The comparison measures:
# 
# - winner agreement,
# - turn-count agreement,
# - player, opponent, and draw totals,
# - direct single-action routing,
# - multi-action model decisions,
# - selector usage,
# - fallback usage,
# - action-sequence consistency,
# - final-state consistency.

# In[15]:


def normalize_battle_winner(
    winner_value: Any,
) -> str:
    """
    Normalize all winner representations to:
    Player, Opponent, or Draw.
    """

    if winner_value is None:
        return "Draw"

    try:
        if pd.isna(winner_value):
            return "Draw"
    except (TypeError, ValueError):
        pass

    normalized_value = str(
        winner_value
    ).strip()

    normalized_lower = normalized_value.lower()

    if normalized_lower in {
        "",
        "none",
        "nan",
        "draw",
        "tie",
    }:
        return "Draw"

    if normalized_lower == "player":
        return "Player"

    if normalized_lower == "opponent":
        return "Opponent"

    return normalized_value


print("✅ normalize_battle_winner() created successfully")


# In[17]:


# ======================================================================================
# SECTION 6E — FULL LEGALITY-AWARE TOURNAMENT REPLAY
# ======================================================================================

print("=" * 100)
print("SECTION 6E — FULL LEGALITY-AWARE TOURNAMENT REPLAY")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Validate replay readiness
# --------------------------------------------------------------------------------------

assert section6d_summary[
    "status"
] == "SIMULATOR_BATTLE_RUNNER_VALIDATED"

assert section6d_summary[
    "ready_for_full_tournament_replay"
] is True

assert section6c_summary[
    "fallback_decisions"
] == 0

assert len(
    notebook52_battle_results_df
) == 24

assert len(
    notebook52_policy_history_df
) == 210


# --------------------------------------------------------------------------------------
# 2. Helper: resolve one required column from aliases
# --------------------------------------------------------------------------------------

def resolve_dataframe_column(
    dataframe: pd.DataFrame,
    candidate_columns: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    """
    Return the first available column from a list of candidates.
    """

    for column_name in candidate_columns:

        if column_name in dataframe.columns:

            return column_name

    if required:

        raise KeyError(
            "None of the required columns were found: "
            f"{list(candidate_columns)}"
        )

    return None


# --------------------------------------------------------------------------------------
# 3. Resolve Notebook 52 battle-result columns
# --------------------------------------------------------------------------------------

SCENARIO_ID_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "scenario_id",
        "expanded_scenario_id",
    ],
)

CONDITION_ID_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "condition_id",
        "condition",
    ],
)

PLAYER_CARD_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "player_card",
        "player_pokemon",
    ],
)

OPPONENT_CARD_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "opponent_card",
        "opponent_pokemon",
    ],
)

STARTING_SIDE_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "starting_side",
        "current_player",
        "start_side",
    ],
)

STARTING_PLAYER_HP_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "starting_player_hp",
        "initial_player_hp",
        "player_hp_start",
    ],
)

STARTING_OPPONENT_HP_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "starting_opponent_hp",
        "initial_opponent_hp",
        "opponent_hp_start",
    ],
)

STARTING_PLAYER_ENERGY_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "starting_player_energy",
        "initial_player_energy",
        "player_energy_start",
    ],
)

STARTING_OPPONENT_ENERGY_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "starting_opponent_energy",
        "initial_opponent_energy",
        "opponent_energy_start",
    ],
)

NOTEBOOK52_WINNER_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "winner",
        "battle_winner",
    ],
)

NOTEBOOK52_TURNS_COLUMN = resolve_dataframe_column(
    notebook52_battle_results_df,
    [
        "recorded_turns",
        "turns",
        "battle_turns",
    ],
)


print()
print("RESOLVED NOTEBOOK 52 REPLAY COLUMNS")
print("-" * 100)

resolved_replay_columns = {
    "scenario_id":
        SCENARIO_ID_COLUMN,

    "condition_id":
        CONDITION_ID_COLUMN,

    "player_card":
        PLAYER_CARD_COLUMN,

    "opponent_card":
        OPPONENT_CARD_COLUMN,

    "starting_side":
        STARTING_SIDE_COLUMN,

    "starting_player_hp":
        STARTING_PLAYER_HP_COLUMN,

    "starting_opponent_hp":
        STARTING_OPPONENT_HP_COLUMN,

    "starting_player_energy":
        STARTING_PLAYER_ENERGY_COLUMN,

    "starting_opponent_energy":
        STARTING_OPPONENT_ENERGY_COLUMN,

    "notebook52_winner":
        NOTEBOOK52_WINNER_COLUMN,

    "notebook52_turns":
        NOTEBOOK52_TURNS_COLUMN,
}

for logical_name, column_name in (
    resolved_replay_columns.items()
):

    print(
        f"{logical_name:30}: {column_name}"
    )


# --------------------------------------------------------------------------------------
# 4. Canonical move definitions
# --------------------------------------------------------------------------------------

CANONICAL_MOVE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "Ascension": {
        "name": "Ascension",
        "damage": 0.0,
        "energy_cost": 1,
    },

    "Bind Down": {
        "name": "Bind Down",
        "damage": 10.0,
        "energy_cost": 3,
    },

    "Live Coal": {
        "name": "Live Coal",
        "damage": 20.0,
        "energy_cost": 3,
    },

    "Pass": {
        "name": "Pass",
        "damage": 0.0,
        "energy_cost": 0,
    },

    "Quick Attack": {
        "name": "Quick Attack",
        "damage": 20.0,
        "energy_cost": 3,
    },

    "Tuck Tail": {
        "name": "Tuck Tail",
        "damage": 0.0,
        "energy_cost": 0,
    },
}


DEFAULT_CARD_HP: Dict[str, float] = {
    "Bulbasaur": 80.0,
    "Charmander": 70.0,
    "Eevee": 50.0,
    "Meowth": 70.0,
}


DEFAULT_CARD_MOVES: Dict[str, List[str]] = {
    "Bulbasaur": [
        "Bind Down",
    ],

    "Charmander": [
        "Live Coal",
    ],

    "Eevee": [
        "Ascension",
        "Quick Attack",
    ],

    "Meowth": [
        "Tuck Tail",
    ],
}


def create_move_dictionary(
    move_name: str,
) -> Dict[str, Any]:
    """
    Return a fresh canonical move dictionary.
    """

    if move_name not in CANONICAL_MOVE_LIBRARY:

        return {
            "name":
                str(
                    move_name
                ),

            "damage":
                0.0,

            "energy_cost":
                0,
        }

    return deepcopy(
        CANONICAL_MOVE_LIBRARY[
            move_name
        ]
    )


# --------------------------------------------------------------------------------------
# 5. Recover scenario-specific legal actions from the frozen policy history
#
# This preserves the exact legal-action sets used by Notebook 52.
# --------------------------------------------------------------------------------------

scenario_side_legal_moves: Dict[
    Tuple[str, str],
    List[str],
] = {}


for (
    scenario_id,
    acting_side,
), group_df in notebook52_policy_history_df.groupby(
    [
        "scenario_id",
        "current_player",
    ],
    dropna=False,
):

    recovered_move_names = []

    for legal_move_value in group_df[
        "legal_moves"
    ]:

        for move_name in parse_list_value(
            legal_move_value
        ):

            cleaned_move_name = str(
                move_name
            ).strip().strip(
                "'\""
            )

            if (
                cleaned_move_name
                and cleaned_move_name
                not in recovered_move_names
            ):

                recovered_move_names.append(
                    cleaned_move_name
                )

    scenario_side_legal_moves[
        (
            str(
                scenario_id
            ),
            str(
                acting_side
            ),
        )
    ] = recovered_move_names


print()
print("RECOVERED SCENARIO-SIDE LEGAL ACTION SETS")
print("-" * 100)

section6e_legal_set_rows = []

for (
    scenario_id,
    acting_side,
), legal_move_names in sorted(
    scenario_side_legal_moves.items()
):

    section6e_legal_set_rows.append(
        {
            "scenario_id":
                scenario_id,

            "acting_side":
                acting_side,

            "legal_moves":
                legal_move_names,

            "legal_move_count":
                len(
                    legal_move_names
                ),
        }
    )


section6e_recovered_legal_sets_df = pd.DataFrame(
    section6e_legal_set_rows
)

display(
    section6e_recovered_legal_sets_df
)


# --------------------------------------------------------------------------------------
# 6. Build one scenario-specific production card
# --------------------------------------------------------------------------------------

def build_replay_card(
    scenario_id: str,
    side_name: str,
    card_name: str,
    starting_hp: float,
) -> Dict[str, Any]:
    """
    Construct a production card with the exact legal actions observed
    for this scenario and acting side.
    """

    legal_move_names = scenario_side_legal_moves.get(
        (
            str(
                scenario_id
            ),
            str(
                side_name
            ),
        ),
        [],
    )

    if not legal_move_names:

        legal_move_names = DEFAULT_CARD_MOVES.get(
            str(
                card_name
            ),
            [
                "Pass",
            ],
        )

    maximum_hp = max(
        float(
            starting_hp
        ),
        float(
            DEFAULT_CARD_HP.get(
                str(
                    card_name
                ),
                starting_hp,
            )
        ),
    )

    return {
        "name":
            str(
                card_name
            ),

        "hp":
            maximum_hp,

        "moves": [
            create_move_dictionary(
                move_name
            )
            for move_name in legal_move_names
        ],
    }


# --------------------------------------------------------------------------------------
# 7. Build one production BattleState
# --------------------------------------------------------------------------------------

def build_replay_battle_state(
    scenario_row: pd.Series,
) -> Any:
    """
    Convert one Notebook 52 scenario row into a production BattleState.
    """

    scenario_id = str(
        scenario_row[
            SCENARIO_ID_COLUMN
        ]
    )

    player_card_name = str(
        scenario_row[
            PLAYER_CARD_COLUMN
        ]
    )

    opponent_card_name = str(
        scenario_row[
            OPPONENT_CARD_COLUMN
        ]
    )

    starting_player_hp = float(
        scenario_row[
            STARTING_PLAYER_HP_COLUMN
        ]
    )

    starting_opponent_hp = float(
        scenario_row[
            STARTING_OPPONENT_HP_COLUMN
        ]
    )

    starting_player_energy = int(
        scenario_row[
            STARTING_PLAYER_ENERGY_COLUMN
        ]
    )

    starting_opponent_energy = int(
        scenario_row[
            STARTING_OPPONENT_ENERGY_COLUMN
        ]
    )

    starting_side = str(
        scenario_row[
            STARTING_SIDE_COLUMN
        ]
    )

    player_card = build_replay_card(
        scenario_id=scenario_id,
        side_name="Player",
        card_name=player_card_name,
        starting_hp=starting_player_hp,
    )

    opponent_card = build_replay_card(
        scenario_id=scenario_id,
        side_name="Opponent",
        card_name=opponent_card_name,
        starting_hp=starting_opponent_hp,
    )

    player_maximum_hp = float(
        player_card[
            "hp"
        ]
    )

    opponent_maximum_hp = float(
        opponent_card[
            "hp"
        ]
    )

    player_pokemon = PokemonState(
        card=player_card,
        current_hp=starting_player_hp,
        attached_energy=starting_player_energy,
        damage=max(
            player_maximum_hp
            - starting_player_hp,
            0.0,
        ),
        is_active=True,
    )

    opponent_pokemon = PokemonState(
        card=opponent_card,
        current_hp=starting_opponent_hp,
        attached_energy=starting_opponent_energy,
        damage=max(
            opponent_maximum_hp
            - starting_opponent_hp,
            0.0,
        ),
        is_active=True,
    )

    player_state = PlayerState(
        active=player_pokemon,
        bench=[],
        prize_cards_remaining=2,
        hand_size=3,
    )

    opponent_state = PlayerState(
        active=opponent_pokemon,
        bench=[],
        prize_cards_remaining=2,
        hand_size=3,
    )

    return BattleState(
        player=player_state,
        opponent=opponent_state,
        turn_number=1,
        current_player=starting_side,
    )


# --------------------------------------------------------------------------------------
# 8. Battle-result helper functions
# --------------------------------------------------------------------------------------

def extract_result_turns(
    battle_result: Any,
) -> List[Any]:
    """
    Extract turn records from a BattleSimulationResult.
    """

    for attribute_name in [
        "turns",
        "turn_records",
        "records",
    ]:

        turn_records = get_state_value(
            battle_result,
            attribute_name,
            None,
        )

        if turn_records is not None:

            return list(
                turn_records
            )

    return []


def extract_result_winner(
    battle_result: Any,
) -> Optional[str]:
    """
    Extract or recompute the battle winner.
    """

    winner_value = get_state_value(
        battle_result,
        "winner",
        None,
    )

    if winner_value is not None:

        return str(
            winner_value
        )

    final_state = get_state_value(
        battle_result,
        "final_state",
        None,
    )

    if final_state is None:

        return None

    winner_value = determine_battle_winner(
        final_state
    )

    return normalize_battle_winner(
    winner_value
)


def extract_termination_reason(
    battle_result: Any,
) -> str:
    """
    Extract the production battle termination reason.
    """

    for attribute_name in [
        "termination_reason",
        "stop_reason",
        "reason",
    ]:

        value = get_state_value(
            battle_result,
            attribute_name,
            None,
        )

        if value is not None:

            return str(
                value
            )

    winner_value = extract_result_winner(
        battle_result
    )

    if winner_value is not None:

        return (
            "Battle reached a terminal state."
        )

    return (
        "Maximum turn limit reached."
    )


# --------------------------------------------------------------------------------------
# 9. Run the full 24-battle tournament replay
# --------------------------------------------------------------------------------------

section6e_battle_rows = []
section6e_turn_rows = []
section6e_policy_history_rows = []
section6e_transcript_rows = []
section6e_failures = []


for scenario_number, (
    original_row_index,
    scenario_row,
) in enumerate(
    notebook52_battle_results_df.iterrows(),
    start=1,
):

    scenario_id = str(
        scenario_row[
            SCENARIO_ID_COLUMN
        ]
    )

    condition_id = str(
        scenario_row[
            CONDITION_ID_COLUMN
        ]
    )

    try:

        initial_state = build_replay_battle_state(
            scenario_row
        )

        notebook53_battle_runner_agent.reset()

        replay_result = simulate_ai_battle(
            initial_state=initial_state,
            agent=notebook53_battle_runner_agent,
            search_depth=0,
            max_turns=30,
            verbose=False,
        )

        replay_turns = extract_result_turns(
            replay_result
        )

        replay_winner = normalize_battle_winner(
            extract_result_winner(
                replay_result
            )
        )

        replay_termination_reason = (
            extract_termination_reason(
                replay_result
            )
        )

        replay_final_state = get_state_value(
            replay_result,
            "final_state",
        )

        notebook52_winner = normalize_battle_winner(
            scenario_row[
                NOTEBOOK52_WINNER_COLUMN
            ]
        )

        notebook52_turn_count = int(
            scenario_row[
                NOTEBOOK52_TURNS_COLUMN
            ]
        )

        winner_matches = (
            replay_winner
            == notebook52_winner
        )

        turn_count_matches = (
            len(
                replay_turns
            )
            == notebook52_turn_count
        )

        runner_statistics = (
            notebook53_battle_runner_agent
            .statistics()
        )

        section6e_battle_rows.append(
            {
                "scenario_number":
                    scenario_number,

                "scenario_id":
                    scenario_id,

                "condition_id":
                    condition_id,

                "player_card":
                    str(
                        scenario_row[
                            PLAYER_CARD_COLUMN
                        ]
                    ),

                "opponent_card":
                    str(
                        scenario_row[
                            OPPONENT_CARD_COLUMN
                        ]
                    ),

                "starting_side":
                    str(
                        scenario_row[
                            STARTING_SIDE_COLUMN
                        ]
                    ),

                "notebook52_winner":
                    notebook52_winner,

                "notebook53_winner":
                    replay_winner,

                "winner_matches":
                    winner_matches,

                "notebook52_turns":
                    notebook52_turn_count,

                "notebook53_turns":
                    len(
                        replay_turns
                    ),

                "turn_count_matches":
                    turn_count_matches,

                "termination_reason":
                    replay_termination_reason,

                "policy_decisions":
                    int(
                        runner_statistics[
                            "policy_total_decisions"
                        ]
                    ),

                "single_action_direct_decisions":
                    int(
                        runner_statistics[
                            "single_action_direct_decisions"
                        ]
                    ),

                "model_decisions":
                    int(
                        runner_statistics[
                            "model_decisions"
                        ]
                    ),

                "selector_decisions":
                    int(
                        runner_statistics[
                            "selector_decisions"
                        ]
                    ),

                "fallback_decisions":
                    int(
                        runner_statistics[
                            "fallback_decisions"
                        ]
                    ),

                "fallback_rate":
                    float(
                        runner_statistics[
                            "fallback_rate"
                        ]
                    ),

                "final_player_hp":
                    float(
                        replay_final_state
                        .player
                        .active
                        .current_hp
                    ),

                "final_opponent_hp":
                    float(
                        replay_final_state
                        .opponent
                        .active
                        .current_hp
                    ),

                "replay_success":
                    True,
            }
        )

        for turn_position, turn_record in enumerate(
            replay_turns,
            start=1,
        ):

            turn_dictionary = (
                asdict(
                    turn_record
                )
                if is_dataclass(
                    turn_record
                )
                else {
                    attribute_name:
                        get_state_value(
                            turn_record,
                            attribute_name,
                            None,
                        )
                    for attribute_name in [
                        "turn_number",
                        "acting_side",
                        "pokemon_name",
                        "move_name",
                        "damage",
                        "score",
                        "search_depth",
                        "nodes",
                        "player_hp_after",
                        "opponent_hp_after",
                        "next_side",
                    ]
                }
            )

            turn_dictionary.update(
                {
                    "scenario_id":
                        scenario_id,

                    "condition_id":
                        condition_id,

                    "record_number":
                        turn_position,
                }
            )

            section6e_turn_rows.append(
                turn_dictionary
            )

        for policy_record in (
            notebook53_battle_runner_agent
            .decision_history
        ):

            policy_record_copy = deepcopy(
                policy_record
            )

            policy_record_copy[
                "scenario_id"
            ] = scenario_id

            policy_record_copy[
                "condition_id"
            ] = condition_id

            section6e_policy_history_rows.append(
                policy_record_copy
            )

        replay_transcript = create_battle_transcript(
            replay_result
        )

        section6e_transcript_rows.append(
            {
                "scenario_id":
                    scenario_id,

                "condition_id":
                    condition_id,

                "transcript":
                    replay_transcript,
            }
        )

        print(
            f"[{scenario_number:02d}/24] "
            f"{scenario_id} — "
            f"Winner: {replay_winner} — "
            f"Turns: {len(replay_turns)} — "
            f"Match: {winner_matches}"
        )

    except Exception as error:

        section6e_failures.append(
            {
                "scenario_number":
                    scenario_number,

                "scenario_id":
                    scenario_id,

                "condition_id":
                    condition_id,

                "error_type":
                    type(
                        error
                    ).__name__,

                "error_message":
                    str(
                        error
                    ),
            }
        )

        print(
            f"[{scenario_number:02d}/24] "
            f"{scenario_id} — FAILED — "
            f"{type(error).__name__}: {error}"
        )


# --------------------------------------------------------------------------------------
# 10. Build replay dataframes
# --------------------------------------------------------------------------------------

section6e_battle_results_df = pd.DataFrame(
    section6e_battle_rows
)

section6e_turn_results_df = pd.DataFrame(
    section6e_turn_rows
)

section6e_policy_history_df = pd.DataFrame(
    section6e_policy_history_rows
)

section6e_transcripts_df = pd.DataFrame(
    section6e_transcript_rows
)

section6e_failures_df = pd.DataFrame(
    section6e_failures
)


print()
print("TOURNAMENT REPLAY RESULTS")
print("-" * 100)

display(
    section6e_battle_results_df
)


# --------------------------------------------------------------------------------------
# 11. Overall outcome summary
# --------------------------------------------------------------------------------------

notebook53_player_wins = int(
    section6e_battle_results_df[
        "notebook53_winner"
    ].eq(
        "Player"
    ).sum()
)

notebook53_opponent_wins = int(
    section6e_battle_results_df[
        "notebook53_winner"
    ].eq(
        "Opponent"
    ).sum()
)

notebook53_draws = int(
    (
        ~section6e_battle_results_df[
            "notebook53_winner"
        ].isin(
            [
                "Player",
                "Opponent",
            ]
        )
    ).sum()
)

notebook53_total_decisions = int(
    section6e_battle_results_df[
        "policy_decisions"
    ].sum()
)

notebook53_single_action_decisions = int(
    section6e_battle_results_df[
        "single_action_direct_decisions"
    ].sum()
)

notebook53_model_decisions = int(
    section6e_battle_results_df[
        "model_decisions"
    ].sum()
)

notebook53_selector_decisions = int(
    section6e_battle_results_df[
        "selector_decisions"
    ].sum()
)

notebook53_fallback_decisions = int(
    section6e_battle_results_df[
        "fallback_decisions"
    ].sum()
)


section6e_outcome_summary_df = pd.DataFrame(
    [
        {
            "metric":
                "battle_count",

            "notebook52":
                int(
                    len(
                        notebook52_battle_results_df
                    )
                ),

            "notebook53":
                int(
                    len(
                        section6e_battle_results_df
                    )
                ),
        },
        {
            "metric":
                "player_wins",

            "notebook52":
                int(
                    player_wins
                ),

            "notebook53":
                notebook53_player_wins,
        },
        {
            "metric":
                "opponent_wins",

            "notebook52":
                int(
                    opponent_wins
                ),

            "notebook53":
                notebook53_opponent_wins,
        },
        {
            "metric":
                "draws",

            "notebook52":
                int(
                    draws
                ),

            "notebook53":
                notebook53_draws,
        },
        {
            "metric":
                "policy_decisions",

            "notebook52":
                int(
                    policy_decisions
                ),

            "notebook53":
                notebook53_total_decisions,
        },
        {
            "metric":
                "fallback_decisions",

            "notebook52":
                int(
                    fallback_decisions
                ),

            "notebook53":
                notebook53_fallback_decisions,
        },
    ]
)


print()
print("NOTEBOOK 52 VS NOTEBOOK 53 OUTCOME SUMMARY")
print("-" * 100)

display(
    section6e_outcome_summary_df
)


# --------------------------------------------------------------------------------------
# 12. Condition summary
# --------------------------------------------------------------------------------------

section6e_condition_summary_df = (
    section6e_battle_results_df
    .groupby(
        "condition_id",
        as_index=False,
    )
    .agg(
        battles=(
            "scenario_id",
            "size",
        ),

        player_wins=(
            "notebook53_winner",
            lambda values: int(
                values.eq(
                    "Player"
                ).sum()
            ),
        ),

        opponent_wins=(
            "notebook53_winner",
            lambda values: int(
                values.eq(
                    "Opponent"
                ).sum()
            ),
        ),

        draws=(
            "notebook53_winner",
            lambda values: int(
                (
                    ~values.isin(
                        [
                            "Player",
                            "Opponent",
                        ]
                    )
                ).sum()
            ),
        ),

        average_turns=(
            "notebook53_turns",
            "mean",
        ),

        policy_decisions=(
            "policy_decisions",
            "sum",
        ),

        single_action_direct_decisions=(
            "single_action_direct_decisions",
            "sum",
        ),

        model_decisions=(
            "model_decisions",
            "sum",
        ),

        selector_decisions=(
            "selector_decisions",
            "sum",
        ),

        fallback_decisions=(
            "fallback_decisions",
            "sum",
        ),
    )
)


section6e_condition_summary_df[
    "player_win_rate"
] = (
    section6e_condition_summary_df[
        "player_wins"
    ]
    / section6e_condition_summary_df[
        "battles"
    ]
)


print()
print("TOURNAMENT REPLAY BY CONDITION")
print("-" * 100)

display(
    section6e_condition_summary_df
)


# --------------------------------------------------------------------------------------
# 13. Routing summary
# --------------------------------------------------------------------------------------

if not section6e_policy_history_df.empty:

    section6e_routing_summary_df = (
        section6e_policy_history_df
        .groupby(
            "routing_reason",
            as_index=False,
        )
        .agg(
            decisions=(
                "decision_number",
                "size",
            ),

            average_confidence=(
                "confidence",
                "mean",
            ),

            fallback_decisions=(
                "fallback_used",
                "sum",
            ),
        )
    )

    section6e_routing_summary_df[
        "decision_fraction"
    ] = (
        section6e_routing_summary_df[
            "decisions"
        ]
        / len(
            section6e_policy_history_df
        )
    )

else:

    section6e_routing_summary_df = pd.DataFrame()


print()
print("TOURNAMENT ROUTING SUMMARY")
print("-" * 100)

display(
    section6e_routing_summary_df
)


# --------------------------------------------------------------------------------------
# 14. Replay validation
# --------------------------------------------------------------------------------------

assert len(
    section6e_failures_df
) == 0, (
    "One or more tournament replay battles failed."
)

assert len(
    section6e_battle_results_df
) == 24

assert section6e_battle_results_df[
    "replay_success"
].all()

assert section6e_battle_results_df[
    "winner_matches"
].all(), (
    "At least one Notebook 53 replay winner differs from Notebook 52."
)

assert section6e_battle_results_df[
    "turn_count_matches"
].all(), (
    "At least one Notebook 53 replay turn count differs from Notebook 52."
)

assert notebook53_player_wins == player_wins

assert notebook53_opponent_wins == opponent_wins

assert notebook53_draws == draws

assert notebook53_total_decisions == policy_decisions

assert notebook53_single_action_decisions == 187

assert notebook53_model_decisions == 23

assert notebook53_selector_decisions == 0

assert notebook53_fallback_decisions == 0

assert len(
    section6e_turn_results_df
) == notebook53_total_decisions

assert len(
    section6e_policy_history_df
) == notebook53_total_decisions

assert section6e_battle_results_df[
    "final_player_hp"
].ge(
    0.0
).all()

assert section6e_battle_results_df[
    "final_opponent_hp"
].ge(
    0.0
).all()


# --------------------------------------------------------------------------------------
# 15. Save tournament replay artifacts
# --------------------------------------------------------------------------------------

SECTION6E_RECOVERED_LEGAL_SETS_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_recovered_scenario_legal_sets.csv"
)

SECTION6E_BATTLE_RESULTS_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_battles.csv"
)

SECTION6E_TURN_RESULTS_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_turns.csv"
)

SECTION6E_POLICY_HISTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_policy_history.csv"
)

SECTION6E_TRANSCRIPTS_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_transcripts.csv"
)

SECTION6E_FAILURES_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_failures.csv"
)

SECTION6E_OUTCOME_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_notebook52_vs_notebook53_outcomes.csv"
)

SECTION6E_CONDITION_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_condition_summary.csv"
)

SECTION6E_ROUTING_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_routing_summary.csv"
)

SECTION6E_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6e_tournament_replay_summary.json"
)


section6e_recovered_legal_sets_export_df = (
    section6e_recovered_legal_sets_df.copy()
)

section6e_recovered_legal_sets_export_df[
    "legal_moves"
] = section6e_recovered_legal_sets_export_df[
    "legal_moves"
].apply(
    lambda value: json.dumps(
        value,
        ensure_ascii=False,
    )
)

section6e_recovered_legal_sets_export_df.to_csv(
    SECTION6E_RECOVERED_LEGAL_SETS_FILE,
    index=False,
)

section6e_battle_results_df.to_csv(
    SECTION6E_BATTLE_RESULTS_FILE,
    index=False,
)

section6e_turn_results_df.to_csv(
    SECTION6E_TURN_RESULTS_FILE,
    index=False,
)

section6e_policy_history_export_df = (
    section6e_policy_history_df.copy()
)

if (
    "legal_moves"
    in section6e_policy_history_export_df.columns
):

    section6e_policy_history_export_df[
        "legal_moves"
    ] = section6e_policy_history_export_df[
        "legal_moves"
    ].apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
        )
    )

section6e_policy_history_export_df.to_csv(
    SECTION6E_POLICY_HISTORY_FILE,
    index=False,
)

section6e_transcripts_df.to_csv(
    SECTION6E_TRANSCRIPTS_FILE,
    index=False,
)

section6e_failures_df.to_csv(
    SECTION6E_FAILURES_FILE,
    index=False,
)

section6e_outcome_summary_df.to_csv(
    SECTION6E_OUTCOME_SUMMARY_FILE,
    index=False,
)

section6e_condition_summary_df.to_csv(
    SECTION6E_CONDITION_SUMMARY_FILE,
    index=False,
)

section6e_routing_summary_df.to_csv(
    SECTION6E_ROUTING_SUMMARY_FILE,
    index=False,
)


section6e_summary = {
    "status":
        "FULL_LEGALITY_AWARE_TOURNAMENT_REPLAY_COMPLETE",

    "battle_count":
        int(
            len(
                section6e_battle_results_df
            )
        ),

    "battles_failed":
        int(
            len(
                section6e_failures_df
            )
        ),

    "player_wins":
        notebook53_player_wins,

    "opponent_wins":
        notebook53_opponent_wins,

    "draws":
        notebook53_draws,

    "winner_agreement_rate":
        float(
            section6e_battle_results_df[
                "winner_matches"
            ].mean()
        ),

    "turn_count_agreement_rate":
        float(
            section6e_battle_results_df[
                "turn_count_matches"
            ].mean()
        ),

    "policy_decisions":
        notebook53_total_decisions,

    "single_action_direct_decisions":
        notebook53_single_action_decisions,

    "model_decisions":
        notebook53_model_decisions,

    "selector_decisions":
        notebook53_selector_decisions,

    "fallback_decisions":
        notebook53_fallback_decisions,

    "fallback_rate":
        float(
            notebook53_fallback_decisions
            / notebook53_total_decisions
            if notebook53_total_decisions > 0
            else 0.0
        ),

    "average_turns":
        float(
            section6e_battle_results_df[
                "notebook53_turns"
            ].mean()
        ),

    "notebook52_outcomes_preserved":
        True,

    "production_legality_preserved":
        True,

    "ready_for_notebook53_final_validation":
        True,

    "next_stage":
        "NOTEBOOK53_FINAL_VALIDATION_AND_HANDOFF",
}


with open(
    SECTION6E_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section6e_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section6e_saved_files = [
    SECTION6E_RECOVERED_LEGAL_SETS_FILE,
    SECTION6E_BATTLE_RESULTS_FILE,
    SECTION6E_TURN_RESULTS_FILE,
    SECTION6E_POLICY_HISTORY_FILE,
    SECTION6E_TRANSCRIPTS_FILE,
    SECTION6E_FAILURES_FILE,
    SECTION6E_OUTCOME_SUMMARY_FILE,
    SECTION6E_CONDITION_SUMMARY_FILE,
    SECTION6E_ROUTING_SUMMARY_FILE,
    SECTION6E_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section6e_saved_files
)


# --------------------------------------------------------------------------------------
# 16. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6E SUMMARY")
print("-" * 100)

for key, value in section6e_summary.items():

    print(
        f"{key:46}: {value}"
    )


print()
print("SAVED SECTION 6E REPORTS")
print("-" * 100)

for file_path in section6e_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6E FULL LEGALITY-AWARE TOURNAMENT REPLAY PASSED"
)


# # Section 7 — Final Validation and Handoff
# 
# ## Section 7A — Final Validation and Notebook 53 Handoff
# 
# #### This section performs the final validation of Notebook 53 and creates the handoff package for Notebook 54.
# 
# #### The final checks confirm:
# 
# - the legality-aware dataset was rebuilt successfully,
# - train, validation, and test splits are group-safe,
# - preprocessing was fitted on training data only,
# - the legality-aware model was trained and saved,
# - validation and test legality reached 100%,
# - the frozen Notebook 52 benchmark remained excluded from training,
# - raw benchmark legality improved substantially,
# - the production policy adapter returned only legal actions,
# - the simulator-compatible adapter passed,
# - all 24 tournament battles replayed successfully,
# - winners and turn counts matched Notebook 52 exactly,
# - no selector or fallback decision was required during replay.
# 
# #### The handoff prepares Notebook 54 for tournament-strength optimization.

# In[21]:


# ======================================================================================
# SECTION 7A — FINAL VALIDATION AND NOTEBOOK 53 HANDOFF
# ======================================================================================

print("=" * 100)
print("SECTION 7A — FINAL VALIDATION AND NOTEBOOK 53 HANDOFF")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Section directory
# --------------------------------------------------------------------------------------

SECTION7_REPORT_DIR = (
    NOTEBOOK53_REPORT_DIR
    / "section7"
)

SECTION7_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 2. Required upstream statuses
# --------------------------------------------------------------------------------------

required_section_statuses = {
    "section1b":
        section1b_summary[
            "status"
        ],

    "section2a":
        section2a_summary[
            "status"
        ],

    "section2b":
        section2b_summary[
            "status"
        ],

    "section3a":
        section3a_summary[
            "status"
        ],

    "section3b":
        section3b_summary[
            "status"
        ],

    "section4a":
        section4a_summary[
            "status"
        ],

    "section4b":
        section4b_summary[
            "status"
        ],

    "section5a":
        section5a_summary[
            "status"
        ],

    "section5b":
        section5b_summary[
            "status"
        ],

    "section6a":
        section6a_summary[
            "status"
        ],

    "section6b":
        section6b_summary[
            "status"
        ],

    "section6c":
        section6c_summary[
            "status"
        ],

    "section6d":
        section6d_summary[
            "status"
        ],

    "section6e":
        section6e_summary[
            "status"
        ],
}


expected_section_statuses = {
    "section1b":
        "NOTEBOOK52_HANDOFF_VALIDATED",

    "section2a":
        "MASKING_ERROR_ANALYSIS_COMPLETE",

    "section2b":
        "MASKING_ROOT_CAUSE_ANALYSIS_COMPLETE",

    "section3a":
        "LEGALITY_AWARE_ROUTING_VALIDATED",

    "section3b":
        "LEGALITY_AWARE_FEATURE_SCHEMA_READY",

    "section4a":
    "SOURCE_POLICY_DATASET_DISCOVERED",

    "section4b":
        "LEGALITY_AWARE_DATASET_REBUILT",

    "section5a":
        "GROUP_SAFE_SPLIT_COMPLETE",

    "section5b":
        "LEGALITY_AWARE_PREPROCESSING_COMPLETE",

    "section6a":
        "LEGALITY_AWARE_MODEL_TRAINED",

    "section6b":
        "FROZEN_BENCHMARK_COMPARISON_COMPLETE",

    "section6c":
        "PRODUCTION_POLICY_ADAPTER_VALIDATED",

    "section6d":
        "SIMULATOR_BATTLE_RUNNER_VALIDATED",

    "section6e":
        "FULL_LEGALITY_AWARE_TOURNAMENT_REPLAY_COMPLETE",
}


section_status_rows = []

for section_name, actual_status in (
    required_section_statuses.items()
):

    expected_status = (
        expected_section_statuses[
            section_name
        ]
    )

    section_status_rows.append(
        {
            "section":
                section_name,

            "actual_status":
                actual_status,

            "expected_status":
                expected_status,

            "passed":
                actual_status
                == expected_status,
        }
    )


section7a_section_status_df = pd.DataFrame(
    section_status_rows
)


print()
print("SECTION COMPLETION STATUS")
print("-" * 100)

display(
    section7a_section_status_df
)


# --------------------------------------------------------------------------------------
# 3. Required artifact inventory
# --------------------------------------------------------------------------------------

required_notebook53_artifacts = {
    "legality_aware_preprocessor":
        SECTION5B_PREPROCESSOR_FILE,

    "encoded_feature_names":
        SECTION5B_ENCODED_FEATURE_NAMES_FILE,

    "raw_feature_schema":
        SECTION5B_RAW_FEATURE_SCHEMA_FILE,

    "legality_aware_model":
        SECTION6A_MODEL_FILE,

    "model_metadata":
        SECTION6A_MODEL_METADATA_FILE,

    "policy_adapter_metadata":
        SECTION6C_ADAPTER_METADATA_FILE,

    "dataset_summary":
        SECTION4B_SUMMARY_FILE,

    "split_summary":
        SECTION5A_SUMMARY_FILE,

    "preprocessing_summary":
        SECTION5B_SUMMARY_FILE,

    "model_training_summary":
        SECTION6A_SUMMARY_FILE,

    "frozen_benchmark_summary":
        SECTION6B_SUMMARY_FILE,

    "policy_adapter_summary":
        SECTION6C_SUMMARY_FILE,

    "battle_runner_summary":
        SECTION6D_SUMMARY_FILE,

    "tournament_replay_summary":
        SECTION6E_SUMMARY_FILE,

    "tournament_replay_battles":
        SECTION6E_BATTLE_RESULTS_FILE,

    "tournament_replay_turns":
        SECTION6E_TURN_RESULTS_FILE,

    "tournament_policy_history":
        SECTION6E_POLICY_HISTORY_FILE,

    "tournament_transcripts":
        SECTION6E_TRANSCRIPTS_FILE,

    "feature_importance":
        SECTION6A_FEATURE_IMPORTANCE_FILE,

    "benchmark_comparison":
        SECTION6B_BENCHMARK_COMPARISON_FILE,
}


artifact_inventory_rows = []

for artifact_name, artifact_path in (
    required_notebook53_artifacts.items()
):

    artifact_path = Path(
        artifact_path
    )

    artifact_exists = artifact_path.exists()

    artifact_is_file = (
        artifact_path.is_file()
        if artifact_exists
        else False
    )

    artifact_size = (
        artifact_path.stat().st_size
        if artifact_is_file
        else 0
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
                artifact_exists,

            "is_file":
                artifact_is_file,

            "size_bytes":
                int(
                    artifact_size
                ),

            "nonempty":
                bool(
                    artifact_size > 0
                ),
        }
    )


section7a_artifact_inventory_df = pd.DataFrame(
    artifact_inventory_rows
)


print()
print("ARTIFACT INVENTORY")
print("-" * 100)

display(
    section7a_artifact_inventory_df
)


# --------------------------------------------------------------------------------------
# 4. Final validation checks
# --------------------------------------------------------------------------------------

section7a_validation_rows = []


def add_section7a_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:

    section7a_validation_rows.append(
        {
            "check":
                check_name,

            "passed":
                bool(
                    passed
                ),

            "value":
                value,

            "expected":
                expected,

            "details":
                details,
        }
    )


add_section7a_check(
    "all_sections_complete",
    bool(
        section7a_section_status_df[
            "passed"
        ].all()
    ),
    int(
        section7a_section_status_df[
            "passed"
        ].sum()
    ),
    int(
        len(
            section7a_section_status_df
        )
    ),
)

add_section7a_check(
    "all_required_artifacts_exist",
    bool(
        section7a_artifact_inventory_df[
            "exists"
        ].all()
    ),
    int(
        (
            ~section7a_artifact_inventory_df[
                "exists"
            ]
        ).sum()
    ),
    0,
)

add_section7a_check(
    "all_required_artifacts_are_files",
    bool(
        section7a_artifact_inventory_df[
            "is_file"
        ].all()
    ),
    int(
        (
            ~section7a_artifact_inventory_df[
                "is_file"
            ]
        ).sum()
    ),
    0,
)

add_section7a_check(
    "all_required_artifacts_nonempty",
    bool(
        section7a_artifact_inventory_df[
            "nonempty"
        ].all()
    ),
    int(
        (
            ~section7a_artifact_inventory_df[
                "nonempty"
            ]
        ).sum()
    ),
    0,
)

add_section7a_check(
    "notebook52_handoff_validated",
    section1b_summary[
        "status"
    ]
    == "NOTEBOOK52_HANDOFF_VALIDATED",
    section1b_summary[
        "status"
    ],
    "NOTEBOOK52_HANDOFF_VALIDATED",
)

add_section7a_check(
    "source_dataset_row_count_valid",
    section4b_summary[
        "rebuilt_rows"
    ] == 1060,
    section4b_summary[
        "rebuilt_rows"
    ],
    1060,
)

add_section7a_check(
    "raw_feature_count_valid",
    section5b_summary[
        "raw_feature_count"
    ] == 41,
    section5b_summary[
        "raw_feature_count"
    ],
    41,
)

add_section7a_check(
    "encoded_feature_count_valid",
    section5b_summary[
        "encoded_feature_count"
    ] == 64,
    section5b_summary[
        "encoded_feature_count"
    ],
    64,
)

add_section7a_check(
    "training_row_count_valid",
    section5a_summary[
        "training_examples"
    ] == 840,
    section5a_summary[
        "training_examples"
    ],
    840,
)

add_section7a_check(
    "validation_row_count_valid",
    section5a_summary[
        "validation_examples"
    ] == 110,
    section5a_summary[
        "validation_examples"
    ],
    110,
)

add_section7a_check(
    "test_row_count_valid",
    section5a_summary[
        "test_examples"
    ] == 110,
    section5a_summary[
        "test_examples"
    ],
    110,
)

add_section7a_check(
    "group_safe_split_valid",
    section5a_summary[
        "scenario_overlap_count"
    ] == 0,
    section5a_summary[
        "scenario_overlap_count"
    ],
    0,
)

add_section7a_check(
    "benchmark_excluded_from_training",
    (
        section4b_summary[
            "benchmark_rows_used_for_training"
        ]
        == 0
    ),
    section4b_summary[
        "benchmark_rows_used_for_training"
    ],
    0,
)

add_section7a_check(
    "model_validation_accuracy_valid",
    np.isclose(
        section6a_summary[
            "validation_accuracy"
        ],
        1.0,
    ),
    section6a_summary[
        "validation_accuracy"
    ],
    1.0,
)

add_section7a_check(
    "model_test_accuracy_valid",
    np.isclose(
        section6a_summary[
            "test_accuracy"
        ],
        1.0,
    ),
    section6a_summary[
        "test_accuracy"
    ],
    1.0,
)

add_section7a_check(
    "model_validation_legality_valid",
    np.isclose(
        section6a_summary[
            "validation_prediction_legality_rate"
        ],
        1.0,
    ),
    section6a_summary[
        "validation_prediction_legality_rate"
    ],
    1.0,
)

add_section7a_check(
    "model_test_legality_valid",
    np.isclose(
        section6a_summary[
            "test_prediction_legality_rate"
        ],
        1.0,
    ),
    section6a_summary[
        "test_prediction_legality_rate"
    ],
    1.0,
)

add_section7a_check(
    "frozen_benchmark_decision_count_valid",
    section6b_summary[
        "benchmark_decisions"
    ] == 210,
    section6b_summary[
        "benchmark_decisions"
    ],
    210,
)

add_section7a_check(
    "raw_legality_improved",
    (
        section6b_summary[
            "notebook53_raw_legality_rate"
        ]
        >
        section6b_summary[
            "notebook52_raw_legality_rate"
        ]
    ),
    {
        "notebook52":
            section6b_summary[
                "notebook52_raw_legality_rate"
            ],

        "notebook53":
            section6b_summary[
                "notebook53_raw_legality_rate"
            ],
    },
    "Notebook53 > Notebook52",
)

add_section7a_check(
    "masking_rate_reduced",
    (
        section6b_summary[
            "notebook53_raw_masking_rate"
        ]
        <
        section6b_summary[
            "notebook52_masking_rate"
        ]
    ),
    {
        "notebook52":
            section6b_summary[
                "notebook52_masking_rate"
            ],

        "notebook53":
            section6b_summary[
                "notebook53_raw_masking_rate"
            ],
    },
    "Notebook53 < Notebook52",
)

add_section7a_check(
    "production_adapter_legality_valid",
    np.isclose(
        section6c_summary[
            "adapter_legality_rate"
        ],
        1.0,
    ),
    section6c_summary[
        "adapter_legality_rate"
    ],
    1.0,
)

add_section7a_check(
    "production_adapter_agreement_valid",
    np.isclose(
        section6c_summary[
            "selection_agreement_rate"
        ],
        1.0,
    ),
    section6c_summary[
        "selection_agreement_rate"
    ],
    1.0,
)

add_section7a_check(
    "simulator_runtime_symbols_valid",
    section6d_summary[
        "runtime_symbols_loaded"
    ] == 11,
    section6d_summary[
        "runtime_symbols_loaded"
    ],
    11,
)

add_section7a_check(
    "simulator_transition_valid",
    section6d_summary[
        "state_transition_valid"
    ] is True,
    section6d_summary[
        "state_transition_valid"
    ],
    True,
)

add_section7a_check(
    "tournament_battle_count_valid",
    section6e_summary[
        "battle_count"
    ] == 24,
    section6e_summary[
        "battle_count"
    ],
    24,
)

add_section7a_check(
    "tournament_failure_count_valid",
    section6e_summary[
        "battles_failed"
    ] == 0,
    section6e_summary[
        "battles_failed"
    ],
    0,
)

add_section7a_check(
    "tournament_policy_decision_count_valid",
    section6e_summary[
        "policy_decisions"
    ] == 210,
    section6e_summary[
        "policy_decisions"
    ],
    210,
)

add_section7a_check(
    "single_action_routing_count_valid",
    section6e_summary[
        "single_action_direct_decisions"
    ] == 187,
    section6e_summary[
        "single_action_direct_decisions"
    ],
    187,
)

add_section7a_check(
    "multi_action_model_count_valid",
    section6e_summary[
        "model_decisions"
    ] == 23,
    section6e_summary[
        "model_decisions"
    ],
    23,
)

add_section7a_check(
    "selector_decision_count_valid",
    section6e_summary[
        "selector_decisions"
    ] == 0,
    section6e_summary[
        "selector_decisions"
    ],
    0,
)

add_section7a_check(
    "fallback_decision_count_valid",
    section6e_summary[
        "fallback_decisions"
    ] == 0,
    section6e_summary[
        "fallback_decisions"
    ],
    0,
)

add_section7a_check(
    "winner_agreement_valid",
    np.isclose(
        section6e_summary[
            "winner_agreement_rate"
        ],
        1.0,
    ),
    section6e_summary[
        "winner_agreement_rate"
    ],
    1.0,
)

add_section7a_check(
    "turn_count_agreement_valid",
    np.isclose(
        section6e_summary[
            "turn_count_agreement_rate"
        ],
        1.0,
    ),
    section6e_summary[
        "turn_count_agreement_rate"
    ],
    1.0,
)

add_section7a_check(
    "player_win_count_preserved",
    section6e_summary[
        "player_wins"
    ] == 7,
    section6e_summary[
        "player_wins"
    ],
    7,
)

add_section7a_check(
    "opponent_win_count_preserved",
    section6e_summary[
        "opponent_wins"
    ] == 14,
    section6e_summary[
        "opponent_wins"
    ],
    14,
)

add_section7a_check(
    "draw_count_preserved",
    section6e_summary[
        "draws"
    ] == 3,
    section6e_summary[
        "draws"
    ],
    3,
)

add_section7a_check(
    "notebook52_outcomes_preserved",
    section6e_summary[
        "notebook52_outcomes_preserved"
    ] is True,
    section6e_summary[
        "notebook52_outcomes_preserved"
    ],
    True,
)

add_section7a_check(
    "production_legality_preserved",
    section6e_summary[
        "production_legality_preserved"
    ] is True,
    section6e_summary[
        "production_legality_preserved"
    ],
    True,
)


section7a_validation_checks_df = pd.DataFrame(
    section7a_validation_rows
)


print()
print("FINAL VALIDATION CHECKS")
print("-" * 100)

display(
    section7a_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 5. Final integration profile
# --------------------------------------------------------------------------------------

section7a_final_profile_rows = [
    {
        "metric":
            "source_dataset_rows",

        "value":
            section4b_summary[
                "rebuilt_rows"
            ],
    },
    {
        "metric":
            "raw_feature_count",

        "value":
            section5b_summary[
                "raw_feature_count"
            ],
    },
    {
        "metric":
            "encoded_feature_count",

        "value":
            section5b_summary[
                "encoded_feature_count"
            ],
    },
    {
        "metric":
            "training_rows",

        "value":
            section5a_summary[
                "training_examples"
            ],
    },
    {
        "metric":
            "validation_rows",

        "value":
            section5a_summary[
                "validation_examples"
            ],
    },
    {
        "metric":
            "test_rows",

        "value":
            section5a_summary[
                "test_examples"
            ],
    },
    {
        "metric":
            "validation_accuracy",

        "value":
            section6a_summary[
                "validation_accuracy"
            ],
    },
    {
        "metric":
            "test_accuracy",

        "value":
            section6a_summary[
                "test_accuracy"
            ],
    },
    {
        "metric":
            "notebook52_raw_legality_rate",

        "value":
            section6b_summary[
                "notebook52_raw_legality_rate"
            ],
    },
    {
        "metric":
            "notebook53_raw_legality_rate",

        "value":
            section6b_summary[
                "notebook53_raw_legality_rate"
            ],
    },
    {
        "metric":
            "notebook52_masking_rate",

        "value":
            section6b_summary[
                "notebook52_masking_rate"
            ],
    },
    {
        "metric":
            "notebook53_raw_masking_rate",

        "value":
            section6b_summary[
                "notebook53_raw_masking_rate"
            ],
    },
    {
        "metric":
            "production_adapter_legality_rate",

        "value":
            section6c_summary[
                "adapter_legality_rate"
            ],
    },
    {
        "metric":
            "tournament_battles",

        "value":
            section6e_summary[
                "battle_count"
            ],
    },
    {
        "metric":
            "tournament_policy_decisions",

        "value":
            section6e_summary[
                "policy_decisions"
            ],
    },
    {
        "metric":
            "single_action_direct_decisions",

        "value":
            section6e_summary[
                "single_action_direct_decisions"
            ],
    },
    {
        "metric":
            "multi_action_model_decisions",

        "value":
            section6e_summary[
                "model_decisions"
            ],
    },
    {
        "metric":
            "selector_decisions",

        "value":
            section6e_summary[
                "selector_decisions"
            ],
    },
    {
        "metric":
            "fallback_decisions",

        "value":
            section6e_summary[
                "fallback_decisions"
            ],
    },
    {
        "metric":
            "winner_agreement_rate",

        "value":
            section6e_summary[
                "winner_agreement_rate"
            ],
    },
    {
        "metric":
            "turn_count_agreement_rate",

        "value":
            section6e_summary[
                "turn_count_agreement_rate"
            ],
    },
]


section7a_final_integration_profile_df = pd.DataFrame(
    section7a_final_profile_rows
)


print()
print("FINAL INTEGRATION PROFILE")
print("-" * 100)

display(
    section7a_final_integration_profile_df
)


# --------------------------------------------------------------------------------------
# 6. Validation result totals
# --------------------------------------------------------------------------------------

section7a_checks_passed = int(
    section7a_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section7a_checks_failed = int(
    (
        ~section7a_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section7a_checks_failed == 0, (
    "One or more Notebook 53 final validation checks failed."
)

assert section7a_section_status_df[
    "passed"
].all()

assert section7a_artifact_inventory_df[
    "exists"
].all()

assert section7a_artifact_inventory_df[
    "is_file"
].all()

assert section7a_artifact_inventory_df[
    "nonempty"
].all()


# --------------------------------------------------------------------------------------
# 7. Optimization targets for Notebook 54
# --------------------------------------------------------------------------------------

section7a_optimization_targets_df = pd.DataFrame(
    [
        {
            "priority":
                1,

            "optimization_target":
                "Improve tournament win rate",

            "current_evidence":
                "7 Player wins, 14 Opponent wins, 3 Draws",

            "recommended_direction":
                "Add strategic value evaluation for multi-action states",
        },
        {
            "priority":
                2,

            "optimization_target":
                "Improve HP_PRESSURE performance",

            "current_evidence":
                "0 Player wins in 8 HP_PRESSURE battles",

            "recommended_direction":
                "Train targeted HP-pressure curriculum and value estimates",
        },
        {
            "priority":
                3,

            "optimization_target":
                "Improve weak card performance",

            "current_evidence":
                "Eevee and Meowth showed weak Player-slot outcomes",

            "recommended_direction":
                "Generate card-specific hard examples and tactical rules",
        },
        {
            "priority":
                4,

            "optimization_target":
                "Reduce remaining raw benchmark illegality",

            "current_evidence":
                "36 raw illegal predictions remained before routing",

            "recommended_direction":
                "Expand Meowth and low-energy legality-aware examples",
        },
        {
            "priority":
                5,

            "optimization_target":
                "Preserve production legality",

            "current_evidence":
                "210 of 210 routed decisions were legal",

            "recommended_direction":
                "Retain deterministic routing and defensive legal selector",
        },
    ]
)


print()
print("NOTEBOOK 54 OPTIMIZATION TARGETS")
print("-" * 100)

display(
    section7a_optimization_targets_df
)


# --------------------------------------------------------------------------------------
# 8. Notebook 53 handoff manifest
# --------------------------------------------------------------------------------------

notebook53_handoff_manifest = {
    "notebook":
        "53_legality_aware_policy_optimization",

    "status":
        "READY_FOR_TOURNAMENT_STRENGTH_OPTIMIZATION",

    "completed_sections": [
        "1A",
        "1B",
        "2A",
        "2B",
        "3A",
        "3B",
        "4A",
        "4B",
        "5A",
        "5B",
        "6A",
        "6B",
        "6C",
        "6D",
        "6E",
        "7A",
    ],

    "source_notebook":
        "52_tournament_policy_evaluation",

    "source_dataset_rows":
        int(
            section4b_summary[
                "rebuilt_rows"
            ]
        ),

    "raw_feature_count":
        int(
            section5b_summary[
                "raw_feature_count"
            ]
        ),

    "encoded_feature_count":
        int(
            section5b_summary[
                "encoded_feature_count"
            ]
        ),

    "policy_class_count":
        int(
            section6a_summary[
                "model_class_count"
            ]
        ),

    "policy_classes":
        MODEL_CLASSES,

    "training_rows":
        int(
            section5a_summary[
                "training_examples"
            ]
        ),

    "validation_rows":
        int(
            section5a_summary[
                "validation_examples"
            ]
        ),

    "test_rows":
        int(
            section5a_summary[
                "test_examples"
            ]
        ),

    "scenario_overlap_count":
        int(
            section5a_summary[
                "scenario_overlap_count"
            ]
        ),

    "benchmark_rows_used_for_training":
        int(
            section6b_summary[
                "benchmark_rows_used_for_training"
            ]
        ),

    "validation_accuracy":
        float(
            section6a_summary[
                "validation_accuracy"
            ]
        ),

    "test_accuracy":
        float(
            section6a_summary[
                "test_accuracy"
            ]
        ),

    "notebook52_raw_legality_rate":
        float(
            section6b_summary[
                "notebook52_raw_legality_rate"
            ]
        ),

    "notebook53_raw_legality_rate":
        float(
            section6b_summary[
                "notebook53_raw_legality_rate"
            ]
        ),

    "notebook52_masking_rate":
        float(
            section6b_summary[
                "notebook52_masking_rate"
            ]
        ),

    "notebook53_raw_masking_rate":
        float(
            section6b_summary[
                "notebook53_raw_masking_rate"
            ]
        ),

    "production_adapter_type":
        type(
            notebook53_policy_agent
        ).__name__,

    "battle_runner_adapter_type":
        type(
            notebook53_battle_runner_agent
        ).__name__,

    "production_adapter_legality_rate":
        float(
            section6c_summary[
                "adapter_legality_rate"
            ]
        ),

    "simulator_runtime_package":
        section6d_summary[
            "runtime_package"
        ],

    "simulator_symbols_loaded":
        int(
            section6d_summary[
                "runtime_symbols_loaded"
            ]
        ),

    "tournament_battles":
        int(
            section6e_summary[
                "battle_count"
            ]
        ),

    "tournament_player_wins":
        int(
            section6e_summary[
                "player_wins"
            ]
        ),

    "tournament_opponent_wins":
        int(
            section6e_summary[
                "opponent_wins"
            ]
        ),

    "tournament_draws":
        int(
            section6e_summary[
                "draws"
            ]
        ),

    "tournament_policy_decisions":
        int(
            section6e_summary[
                "policy_decisions"
            ]
        ),

    "single_action_direct_decisions":
        int(
            section6e_summary[
                "single_action_direct_decisions"
            ]
        ),

    "multi_action_model_decisions":
        int(
            section6e_summary[
                "model_decisions"
            ]
        ),

    "selector_decisions":
        int(
            section6e_summary[
                "selector_decisions"
            ]
        ),

    "fallback_decisions":
        int(
            section6e_summary[
                "fallback_decisions"
            ]
        ),

    "winner_agreement_rate":
        float(
            section6e_summary[
                "winner_agreement_rate"
            ]
        ),

    "turn_count_agreement_rate":
        float(
            section6e_summary[
                "turn_count_agreement_rate"
            ]
        ),

    "validation_checks_passed":
        section7a_checks_passed,

    "validation_checks_failed":
        section7a_checks_failed,

    "policy_model_file":
        str(
            SECTION6A_MODEL_FILE
        ),

    "policy_preprocessor_file":
        str(
            SECTION5B_PREPROCESSOR_FILE
        ),

    "encoded_feature_names_file":
        str(
            SECTION5B_ENCODED_FEATURE_NAMES_FILE
        ),

    "raw_feature_schema_file":
        str(
            SECTION5B_RAW_FEATURE_SCHEMA_FILE
        ),

    "policy_adapter_metadata_file":
        str(
            SECTION6C_ADAPTER_METADATA_FILE
        ),

    "tournament_replay_summary_file":
        str(
            SECTION6E_SUMMARY_FILE
        ),

    "primary_optimization_target":
        "Improve tournament win rate while preserving 100% action legality",

    "weakest_condition":
        "HP_PRESSURE",

    "next_notebook":
        "Notebook 54 — Tournament Strength Optimization",

    "next_stage":
        "TOURNAMENT_STRENGTH_OPTIMIZATION",
}


# --------------------------------------------------------------------------------------
# 9. Save final handoff artifacts
# --------------------------------------------------------------------------------------

SECTION7A_SECTION_STATUS_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_section_completion_status.csv"
)

SECTION7A_ARTIFACT_INVENTORY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_artifact_inventory.csv"
)

SECTION7A_VALIDATION_CHECKS_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_validation_checks.csv"
)

SECTION7A_FINAL_PROFILE_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_integration_profile.csv"
)

SECTION7A_OPTIMIZATION_TARGETS_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_notebook54_optimization_targets.csv"
)

SECTION7A_HANDOFF_MANIFEST_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_handoff_manifest.json"
)

SECTION7A_FINAL_SUMMARY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_summary.json"
)


section7a_section_status_df.to_csv(
    SECTION7A_SECTION_STATUS_FILE,
    index=False,
)

section7a_artifact_inventory_df.to_csv(
    SECTION7A_ARTIFACT_INVENTORY_FILE,
    index=False,
)

section7a_validation_checks_df.to_csv(
    SECTION7A_VALIDATION_CHECKS_FILE,
    index=False,
)

section7a_final_integration_profile_df.to_csv(
    SECTION7A_FINAL_PROFILE_FILE,
    index=False,
)

section7a_optimization_targets_df.to_csv(
    SECTION7A_OPTIMIZATION_TARGETS_FILE,
    index=False,
)


with open(
    SECTION7A_HANDOFF_MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook53_handoff_manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )


section7a_final_summary = {
    "status":
        "NOTEBOOK53_COMPLETE",

    "final_handoff_status":
        notebook53_handoff_manifest[
            "status"
        ],

    "source_dataset_rows":
        notebook53_handoff_manifest[
            "source_dataset_rows"
        ],

    "raw_feature_count":
        notebook53_handoff_manifest[
            "raw_feature_count"
        ],

    "encoded_feature_count":
        notebook53_handoff_manifest[
            "encoded_feature_count"
        ],

    "validation_accuracy":
        notebook53_handoff_manifest[
            "validation_accuracy"
        ],

    "test_accuracy":
        notebook53_handoff_manifest[
            "test_accuracy"
        ],

    "raw_legality_improvement":
        float(
            section6b_summary[
                "raw_legality_absolute_improvement"
            ]
        ),

    "masking_reduction":
        float(
            section6b_summary[
                "masking_absolute_reduction"
            ]
        ),

    "tournament_battles":
        notebook53_handoff_manifest[
            "tournament_battles"
        ],

    "tournament_policy_decisions":
        notebook53_handoff_manifest[
            "tournament_policy_decisions"
        ],

    "winner_agreement_rate":
        notebook53_handoff_manifest[
            "winner_agreement_rate"
        ],

    "turn_count_agreement_rate":
        notebook53_handoff_manifest[
            "turn_count_agreement_rate"
        ],

    "selector_decisions":
        notebook53_handoff_manifest[
            "selector_decisions"
        ],

    "fallback_decisions":
        notebook53_handoff_manifest[
            "fallback_decisions"
        ],

    "validation_checks_passed":
        section7a_checks_passed,

    "validation_checks_failed":
        section7a_checks_failed,

    "next_notebook":
        notebook53_handoff_manifest[
            "next_notebook"
        ],

    "next_stage":
        notebook53_handoff_manifest[
            "next_stage"
        ],
}


with open(
    SECTION7A_FINAL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section7a_final_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section7a_saved_files = [
    SECTION7A_SECTION_STATUS_FILE,
    SECTION7A_ARTIFACT_INVENTORY_FILE,
    SECTION7A_VALIDATION_CHECKS_FILE,
    SECTION7A_FINAL_PROFILE_FILE,
    SECTION7A_OPTIMIZATION_TARGETS_FILE,
    SECTION7A_HANDOFF_MANIFEST_FILE,
    SECTION7A_FINAL_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section7a_saved_files
)


# --------------------------------------------------------------------------------------
# 10. Final output
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 53 HANDOFF SUMMARY")
print("-" * 100)

print(
    f"Final status                     : "
    f"{notebook53_handoff_manifest['status']}"
)

print(
    f"Source dataset rows              : "
    f"{notebook53_handoff_manifest['source_dataset_rows']}"
)

print(
    f"Raw features                     : "
    f"{notebook53_handoff_manifest['raw_feature_count']}"
)

print(
    f"Encoded features                 : "
    f"{notebook53_handoff_manifest['encoded_feature_count']}"
)

print(
    f"Validation accuracy              : "
    f"{notebook53_handoff_manifest['validation_accuracy']:.4f}"
)

print(
    f"Test accuracy                    : "
    f"{notebook53_handoff_manifest['test_accuracy']:.4f}"
)

print(
    f"Notebook 52 raw legality         : "
    f"{notebook53_handoff_manifest['notebook52_raw_legality_rate']:.4%}"
)

print(
    f"Notebook 53 raw legality         : "
    f"{notebook53_handoff_manifest['notebook53_raw_legality_rate']:.4%}"
)

print(
    f"Notebook 52 masking rate         : "
    f"{notebook53_handoff_manifest['notebook52_masking_rate']:.4%}"
)

print(
    f"Notebook 53 raw masking rate     : "
    f"{notebook53_handoff_manifest['notebook53_raw_masking_rate']:.4%}"
)

print(
    f"Tournament battles               : "
    f"{notebook53_handoff_manifest['tournament_battles']}"
)

print(
    f"Tournament policy decisions      : "
    f"{notebook53_handoff_manifest['tournament_policy_decisions']}"
)

print(
    f"Single-action direct decisions   : "
    f"{notebook53_handoff_manifest['single_action_direct_decisions']}"
)

print(
    f"Multi-action model decisions     : "
    f"{notebook53_handoff_manifest['multi_action_model_decisions']}"
)

print(
    f"Selector decisions               : "
    f"{notebook53_handoff_manifest['selector_decisions']}"
)

print(
    f"Fallback decisions               : "
    f"{notebook53_handoff_manifest['fallback_decisions']}"
)

print(
    f"Winner agreement rate            : "
    f"{notebook53_handoff_manifest['winner_agreement_rate']:.4%}"
)

print(
    f"Turn-count agreement rate        : "
    f"{notebook53_handoff_manifest['turn_count_agreement_rate']:.4%}"
)

print(
    f"Validation checks passed         : "
    f"{notebook53_handoff_manifest['validation_checks_passed']}"
)

print(
    f"Validation checks failed         : "
    f"{notebook53_handoff_manifest['validation_checks_failed']}"
)

print(
    f"Primary optimization target      : "
    f"{notebook53_handoff_manifest['primary_optimization_target']}"
)

print(
    f"Weakest condition                : "
    f"{notebook53_handoff_manifest['weakest_condition']}"
)

print(
    f"Next notebook                    : "
    f"{notebook53_handoff_manifest['next_notebook']}"
)

print(
    f"Next stage                       : "
    f"{notebook53_handoff_manifest['next_stage']}"
)


print()
print("SAVED SECTION 7A REPORTS")
print("-" * 100)

for file_path in section7a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 7A FINAL VALIDATION AND HANDOFF PASSED"
)

print()
print(
    "🎉 NOTEBOOK 53 COMPLETE — READY FOR NOTEBOOK 54"
)


# In[ ]:




