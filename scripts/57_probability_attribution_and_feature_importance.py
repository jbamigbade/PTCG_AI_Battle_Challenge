#!/usr/bin/env python
# coding: utf-8

# # Notebook 57 — Probability Attribution and Feature Importance
# 
# ## Pokémon TCG AI Battle Challenge
# 
# ### Purpose
# 
# Notebook 57 determines which model features are most strongly associated with the probability shifts identified in Notebook 56.
# 
# Notebook 56 established that all four frozen policies selected the same action across all 184 controlled evaluation cases while producing different probability distributions.
# 
# Notebook 57 will examine:
# 
# 1. Frozen model feature importances.
# 2. Feature importance by strategy.
# 3. Features removed at each ablation stage.
# 4. Probability attribution for Quick Attack and Ascension.
# 5. High-drift evaluation cases.
# 6. Feature-family influence.
# 7. Stable and low-value features.
# 8. Recommendations for the final tournament policy.
# 
# ### Notebook 56 findings carried forward
# 
# - Evaluation cases: 184
# - Frozen models: 4
# - Pairwise prediction agreement: 100%
# - Pairwise prediction disagreements: 0
# - Largest case-level probability drift: 0.168
# - Most divergent comparison: R0_CURRENT_BASELINE vs R3_STATE_CENTRIC_POLICY
# - Top drifting action: Quick Attack
# - Probability mass conserved: True

# In[1]:


# ======================================================================================
# SECTION 1A — IMPORTS AND PROJECT PATHS
# ======================================================================================

from pathlib import Path
import json
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 100)
print("SECTION 1A — IMPORTS AND PROJECT PATHS")
print("=" * 100)

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOKS_DIRECTORY = PROJECT_ROOT / "notebooks"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports" / "notebook57"
ARTIFACTS_DIRECTORY = PROJECT_ROOT / "artifacts" / "notebook57"

NOTEBOOK56_RESULTS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook55"
    / "notebook56"
    / "notebook56_controlled_policy_evaluation_results.pkl"
)

REPORTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

section1a_paths_df = pd.DataFrame(
    [
        {
            "resource": "project_root",
            "path": str(PROJECT_ROOT),
            "exists": PROJECT_ROOT.exists(),
        },
        {
            "resource": "notebooks_directory",
            "path": str(NOTEBOOKS_DIRECTORY),
            "exists": NOTEBOOKS_DIRECTORY.exists(),
        },
        {
            "resource": "notebook57_reports",
            "path": str(REPORTS_DIRECTORY),
            "exists": REPORTS_DIRECTORY.exists(),
        },
        {
            "resource": "notebook57_artifacts",
            "path": str(ARTIFACTS_DIRECTORY),
            "exists": ARTIFACTS_DIRECTORY.exists(),
        },
        {
            "resource": "notebook56_results",
            "path": str(NOTEBOOK56_RESULTS_FILE),
            "exists": NOTEBOOK56_RESULTS_FILE.exists(),
        },
    ]
)

display(section1a_paths_df)

assert PROJECT_ROOT.exists()
assert NOTEBOOKS_DIRECTORY.exists()
assert REPORTS_DIRECTORY.exists()
assert ARTIFACTS_DIRECTORY.exists()
assert NOTEBOOK56_RESULTS_FILE.exists(), (
    "Notebook 56 artifact package was not found: "
    f"{NOTEBOOK56_RESULTS_FILE}"
)

print()
print("✅ NOTEBOOK 57 ENVIRONMENT INITIALIZED")


# In[2]:


# ======================================================================================
# SECTION 1B — LOAD AND VALIDATE NOTEBOOK 56 RESULTS
# ======================================================================================

print("=" * 100)
print("SECTION 1B — LOAD AND VALIDATE NOTEBOOK 56 RESULTS")
print("=" * 100)

with NOTEBOOK56_RESULTS_FILE.open("rb") as file:
    notebook56_results = pickle.load(file)

assert isinstance(
    notebook56_results,
    dict,
), (
    "Notebook 56 results must load as a dictionary. "
    f"Found: {type(notebook56_results).__name__}"
)

print()
print("NOTEBOOK 56 PACKAGE KEYS")
print("-" * 100)

for package_key in notebook56_results:
    print(package_key)

required_notebook56_keys = [
    "metadata",
    "predictions",
    "feature_matrices",
    "prediction_agreement",
    "probability_difference",
    "case_probability_drift",
    "pair_drift_summary",
    "variant_drift_summary",
    "side_drift_summary",
    "action_probability_drift",
    "signed_action_probability_shift",
    "probability_mass_validation",
    "feature_inventory",
    "feature_removal_impact",
    "feature_family_summary",
    "executive_summary",
    "object_inventory",
]

missing_notebook56_keys = [
    key
    for key in required_notebook56_keys
    if key not in notebook56_results
]

assert not missing_notebook56_keys, (
    "Notebook 56 package is missing required keys: "
    f"{missing_notebook56_keys}"
)

notebook56_metadata = notebook56_results[
    "metadata"
]

notebook56_predictions = notebook56_results[
    "predictions"
]

notebook56_feature_matrices = notebook56_results[
    "feature_matrices"
]

notebook56_prediction_agreement_df = notebook56_results[
    "prediction_agreement"
]

notebook56_probability_difference_df = notebook56_results[
    "probability_difference"
]

notebook56_case_probability_drift_df = notebook56_results[
    "case_probability_drift"
]

notebook56_pair_drift_summary_df = notebook56_results[
    "pair_drift_summary"
]

notebook56_variant_drift_summary_df = notebook56_results[
    "variant_drift_summary"
]

notebook56_side_drift_summary_df = notebook56_results[
    "side_drift_summary"
]

notebook56_action_probability_drift_df = notebook56_results[
    "action_probability_drift"
]

notebook56_signed_action_shift_df = notebook56_results[
    "signed_action_probability_shift"
]

notebook56_probability_mass_validation_df = notebook56_results[
    "probability_mass_validation"
]

notebook56_feature_inventory_df = notebook56_results[
    "feature_inventory"
]

notebook56_feature_removal_impact_df = notebook56_results[
    "feature_removal_impact"
]

notebook56_feature_family_summary_df = notebook56_results[
    "feature_family_summary"
]

notebook56_executive_summary_df = notebook56_results[
    "executive_summary"
]

notebook56_object_inventory_df = notebook56_results[
    "object_inventory"
]


# --------------------------------------------------------------------------------------
# Build a package validation inventory.
# --------------------------------------------------------------------------------------

section1b_inventory_rows = []

for object_name, object_value in notebook56_results.items():

    if isinstance(object_value, pd.DataFrame):
        row_count = int(object_value.shape[0])
        column_count = int(object_value.shape[1])

    elif isinstance(object_value, dict):
        row_count = len(object_value)
        column_count = None

    else:
        row_count = None
        column_count = None

    section1b_inventory_rows.append(
        {
            "object_name": object_name,
            "object_type": type(object_value).__name__,
            "rows_or_items": row_count,
            "columns": column_count,
            "available": object_value is not None,
        }
    )

section1b_package_inventory_df = pd.DataFrame(
    section1b_inventory_rows
)

print()
print("NOTEBOOK 56 PACKAGE INVENTORY")
print("-" * 100)

display(
    section1b_package_inventory_df
)

print()
print("NOTEBOOK 56 METADATA")
print("-" * 100)

for metadata_key, metadata_value in notebook56_metadata.items():
    print(f"{metadata_key:30}: {metadata_value}")


# --------------------------------------------------------------------------------------
# Core validations.
# --------------------------------------------------------------------------------------

assert notebook56_metadata[
    "evaluation_cases"
] == 184

assert len(
    notebook56_predictions
) == 4

assert len(
    notebook56_feature_matrices
) == 4

assert notebook56_prediction_agreement_df[
    "agreement"
].eq(1.0).all()

assert notebook56_probability_mass_validation_df[
    "probability_mass_conserved"
].astype(bool).all()

assert notebook56_case_probability_drift_df.shape[0] == 1104

assert notebook56_action_probability_drift_df.shape[0] == 36

assert notebook56_signed_action_shift_df.shape[0] == 36

print()
print("✅ NOTEBOOK 56 RESULTS LOADED AND VALIDATED")


# In[3]:


# ==========================================================================================
# SECTION 2A — INSPECT NOTEBOOK 56 STRUCTURES AND LOCATE FROZEN POLICY MODELS
# ==========================================================================================

from collections.abc import Mapping, Sequence


def describe_python_object(
    object_name: str,
    obj,
) -> dict:
    """
    Return a compact structural description of a Python object.
    """
    record = {
        "object_name": object_name,
        "python_type": type(obj).__name__,
        "length_or_rows": None,
        "columns_or_keys": None,
        "sample_keys_or_columns": None,
    }

    if isinstance(obj, pd.DataFrame):
        record["length_or_rows"] = len(obj)
        record["columns_or_keys"] = obj.shape[1]
        record["sample_keys_or_columns"] = list(obj.columns[:12])

    elif isinstance(obj, pd.Series):
        record["length_or_rows"] = len(obj)
        record["columns_or_keys"] = 1
        record["sample_keys_or_columns"] = [obj.name]

    elif isinstance(obj, Mapping):
        record["length_or_rows"] = len(obj)
        record["columns_or_keys"] = len(obj)
        record["sample_keys_or_columns"] = list(obj.keys())[:12]

    elif isinstance(obj, np.ndarray):
        record["length_or_rows"] = obj.shape[0] if obj.ndim >= 1 else 1
        record["columns_or_keys"] = obj.shape[1] if obj.ndim >= 2 else None
        record["sample_keys_or_columns"] = str(obj.shape)

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        record["length_or_rows"] = len(obj)
        record["sample_keys_or_columns"] = [
            type(item).__name__
            for item in list(obj)[:5]
        ]

    return record


objects_to_inspect = {
    "notebook56_metadata": notebook56_metadata,
    "notebook56_predictions": notebook56_predictions,
    "notebook56_feature_matrices": notebook56_feature_matrices,
    "notebook56_prediction_agreement_df": notebook56_prediction_agreement_df,
    "notebook56_probability_difference_df": notebook56_probability_difference_df,
    "notebook56_case_probability_drift_df": notebook56_case_probability_drift_df,
    "notebook56_pair_drift_summary_df": notebook56_pair_drift_summary_df,
    "notebook56_variant_drift_summary_df": notebook56_variant_drift_summary_df,
    "notebook56_side_drift_summary_df": notebook56_side_drift_summary_df,
    "notebook56_action_probability_drift_df": notebook56_action_probability_drift_df,
    "notebook56_signed_action_shift_df": notebook56_signed_action_shift_df,
    "notebook56_feature_inventory_df": notebook56_feature_inventory_df,
    "notebook56_feature_removal_impact_df": notebook56_feature_removal_impact_df,
    "notebook56_feature_family_summary_df": notebook56_feature_family_summary_df,
    "notebook56_executive_summary_df": notebook56_executive_summary_df,
}

section2a_object_summary_df = pd.DataFrame(
    [
        describe_python_object(name, obj)
        for name, obj in objects_to_inspect.items()
    ]
)

print("=" * 100)
print("SECTION 2A — NOTEBOOK 56 OBJECT STRUCTURE")
print("=" * 100)

display(section2a_object_summary_df)

print("\n" + "=" * 100)
print("PREDICTIONS DICTIONARY")
print("=" * 100)

if isinstance(notebook56_predictions, Mapping):
    for strategy_name, strategy_object in notebook56_predictions.items():
        print(
            f"{strategy_name}: "
            f"type={type(strategy_object).__name__}"
        )

        if isinstance(strategy_object, pd.DataFrame):
            print(
                f"  shape={strategy_object.shape}, "
                f"columns={list(strategy_object.columns)}"
            )

        elif isinstance(strategy_object, Mapping):
            print(
                f"  keys={list(strategy_object.keys())[:20]}"
            )

        elif isinstance(strategy_object, np.ndarray):
            print(
                f"  shape={strategy_object.shape}, "
                f"dtype={strategy_object.dtype}"
            )

print("\n" + "=" * 100)
print("FEATURE MATRICES DICTIONARY")
print("=" * 100)

if isinstance(notebook56_feature_matrices, Mapping):
    for strategy_name, matrix_object in notebook56_feature_matrices.items():
        print(
            f"{strategy_name}: "
            f"type={type(matrix_object).__name__}"
        )

        if isinstance(matrix_object, pd.DataFrame):
            print(
                f"  shape={matrix_object.shape}"
            )
            print(
                f"  first columns={list(matrix_object.columns[:15])}"
            )

        elif isinstance(matrix_object, Mapping):
            print(
                f"  keys={list(matrix_object.keys())[:20]}"
            )

            for inner_key, inner_value in list(matrix_object.items())[:5]:
                if isinstance(inner_value, pd.DataFrame):
                    description = (
                        f"DataFrame shape={inner_value.shape}"
                    )
                elif isinstance(inner_value, np.ndarray):
                    description = (
                        f"ndarray shape={inner_value.shape}"
                    )
                else:
                    description = type(inner_value).__name__

                print(
                    f"    {inner_key}: {description}"
                )

        elif isinstance(matrix_object, np.ndarray):
            print(
                f"  shape={matrix_object.shape}, "
                f"dtype={matrix_object.dtype}"
            )

print("\n" + "=" * 100)
print("FEATURE INVENTORY")
print("=" * 100)

display(notebook56_feature_inventory_df)

print("\n" + "=" * 100)
print("FEATURE REMOVAL IMPACT")
print("=" * 100)

display(notebook56_feature_removal_impact_df)

print("\n" + "=" * 100)
print("FEATURE FAMILY SUMMARY")
print("=" * 100)

display(notebook56_feature_family_summary_df)

print("\n✅ SECTION 2A COMPLETE")


# In[4]:


# ==========================================================================================
# SECTION 2B — DISCOVER AND INSPECT FROZEN POLICY MODEL ARTIFACTS
# ==========================================================================================

from pathlib import Path


def discover_model_artifacts(
    project_root: Path,
) -> pd.DataFrame:
    """
    Search the project for likely serialized policy-model artifacts.

    This function does not load or modify any files. It only reports
    candidate model artifacts based on filename and file extension.
    """
    supported_suffixes = {
        ".joblib",
        ".pkl",
        ".pickle",
    }

    likely_model_tokens = [
        "model",
        "policy",
        "forest",
        "random_forest",
        "classifier",
        "strategy",
        "ablation",
        "frozen",
        "r0",
        "r1",
        "r2",
        "r3",
    ]

    excluded_tokens = [
        "results",
        "summary",
        "report",
        "inventory",
        "metadata",
        "evaluation",
    ]

    search_directories = [
        project_root / "models",
        project_root / "artifacts",
        project_root / "outputs",
        project_root / "data",
    ]

    records = []

    for search_directory in search_directories:
        if not search_directory.exists():
            continue

        for artifact_path in search_directory.rglob("*"):
            if not artifact_path.is_file():
                continue

            suffix = artifact_path.suffix.lower()

            if suffix not in supported_suffixes:
                continue

            lower_name = artifact_path.name.lower()
            lower_path = str(artifact_path).lower()

            model_token_matches = [
                token
                for token in likely_model_tokens
                if token in lower_name or token in lower_path
            ]

            excluded_token_matches = [
                token
                for token in excluded_tokens
                if token in lower_name
            ]

            if not model_token_matches:
                continue

            file_stat = artifact_path.stat()

            records.append(
                {
                    "filename": artifact_path.name,
                    "path": str(artifact_path),
                    "suffix": suffix,
                    "size_bytes": int(file_stat.st_size),
                    "size_mb": round(
                        file_stat.st_size / (1024 ** 2),
                        4,
                    ),
                    "model_tokens": model_token_matches,
                    "excluded_tokens": excluded_token_matches,
                    "modified_time": pd.Timestamp(
                        file_stat.st_mtime,
                        unit="s",
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "filename",
                "path",
                "suffix",
                "size_bytes",
                "size_mb",
                "model_tokens",
                "excluded_tokens",
                "modified_time",
            ]
        )

    candidate_df = pd.DataFrame(records)

    candidate_df["likely_direct_model_artifact"] = (
        candidate_df["excluded_tokens"].map(len).eq(0)
    )

    candidate_df = (
        candidate_df
        .sort_values(
            [
                "likely_direct_model_artifact",
                "modified_time",
                "size_bytes",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return candidate_df


section2b_model_candidates_df = discover_model_artifacts(
    PROJECT_ROOT
)

print("=" * 100)
print("SECTION 2B — DISCOVER FROZEN POLICY MODEL ARTIFACTS")
print("=" * 100)

if section2b_model_candidates_df.empty:
    print("⚠️ No likely model artifacts were discovered automatically.")
else:
    print(
        f"Candidate serialized artifacts found: "
        f"{len(section2b_model_candidates_df):,}"
    )

    display(
        section2b_model_candidates_df[
            [
                "filename",
                "path",
                "suffix",
                "size_mb",
                "model_tokens",
                "excluded_tokens",
                "likely_direct_model_artifact",
                "modified_time",
            ]
        ]
    )

print("\n" + "=" * 100)
print("TOP CANDIDATE PATHS")
print("=" * 100)

if not section2b_model_candidates_df.empty:
    for candidate_index, candidate_row in (
        section2b_model_candidates_df
        .head(20)
        .iterrows()
    ):
        print(
            f"[{candidate_index}] "
            f"{candidate_row['path']}"
        )
else:
    print("No candidates available.")

print("\n✅ SECTION 2B DISCOVERY COMPLETE")


# ### Section 2C — Model Package Structure Correction
# 
# The saved `.joblib` files contain dictionaries rather than direct estimators.
# 
# Each package contains:
# 
# - `strategy_id`
# - `model`
# - `retained_feature_names`
# - `retained_feature_indices`
# - `training_results`
# 
# The fitted `RandomForestClassifier` is stored under:
# 
# ```python
# model_package["model"]

# In[6]:


# ==========================================================================================
# SECTION 2C.1 — INSPECT SAVED MODEL DICTIONARIES
# ==========================================================================================

print("=" * 100)
print("SECTION 2C.1 — MODEL DICTIONARY STRUCTURE")
print("=" * 100)

# Reconstruct the model-package dictionary if the kernel was restarted
# and the earlier discovery cell did not create it.

if "frozen_policy_models" not in globals():

    MODEL_DIRECTORY = PROJECT_ROOT / "models" / "notebook55"

    FROZEN_MODEL_PATHS = {
        "R0_CURRENT_BASELINE": (
            MODEL_DIRECTORY
            / "section3a_r0_current_baseline_model.joblib"
        ),
        "R1_REMOVE_LEGAL_SIGNATURE": (
            MODEL_DIRECTORY
            / "section3a_r1_remove_legal_signature_model.joblib"
        ),
        "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY": (
            MODEL_DIRECTORY
            / "section3a_r2_remove_signature_and_compatibility_model.joblib"
        ),
        "R3_STATE_CENTRIC_POLICY": (
            MODEL_DIRECTORY
            / "section3a_r3_state_centric_policy_model.joblib"
        ),
    }

    missing_model_files = [
        str(model_path)
        for model_path in FROZEN_MODEL_PATHS.values()
        if not model_path.exists()
    ]

    assert not missing_model_files, (
        "One or more frozen model files are missing:\n"
        + "\n".join(missing_model_files)
    )

    frozen_policy_models = {
        strategy_name: joblib.load(model_path)
        for strategy_name, model_path
        in FROZEN_MODEL_PATHS.items()
    }

    print(
        "\nReconstructed frozen_policy_models "
        "from saved Notebook 55 artifacts."
    )

else:
    print(
        "\nfrozen_policy_models already exists "
        "in the current kernel."
    )


for strategy_name, model_package in frozen_policy_models.items():

    print("\n" + "=" * 80)
    print(strategy_name)
    print("=" * 80)

    print("Python type:")
    print(type(model_package).__name__)

    print("\nDictionary keys:")

    for key in model_package.keys():

        value = model_package[key]

        print(
            f"{key:<30} "
            f"{type(value).__name__}"
        )

        if hasattr(value, "shape"):
            print(
                f"    shape = {value.shape}"
            )

        elif isinstance(value, list):
            print(
                f"    length = {len(value)}"
            )

        elif isinstance(value, dict):
            print(
                f"    keys = "
                f"{list(value.keys())[:15]}"
            )

print("\n✅ SECTION 2C.1 COMPLETE")


# 
# For the failed Section 2C.2 cell, replace it with the corrected final validation code from Section 2C.4 and rerun it.
# 
# Then:
# 
# - keep Section 2C.1, because it documented the dictionary structure;
# - keep Section 2C.3, because it proved the feature sets match;
# - rerun the corrected validation so the last output is green;
# - clear the red outputs from the old failed cells.
# 
# In Jupyter, click the failed cell, then use:
# 
# ```text
# Edit → Clear Outputs of Selected Cells
# For the failed Section 2C.2 cell, replace it with the corrected final validation code from Section 2C.4 and rerun it.
# 
# Then:
# 
# - keep Section 2C.1, because it documented the dictionary structure;
# - keep Section 2C.3, because it proved the feature sets match;
# - rerun the corrected validation so the last output is green;
# - clear the red outputs from the old failed cells.
# 
# In Jupyter, click the failed cell, then use:
# 
# ```text
# Edit → Clear Outputs of Selected Cells

# In[7]:


# ==========================================================================================
# SECTION 2C.3 — DIAGNOSE FEATURE-NAME ORDERING DIFFERENCES
# ==========================================================================================

feature_name_comparison_records = []

for strategy_name, model_package in frozen_policy_models.items():

    package_feature_names = list(
        model_package["retained_feature_names"]
    )

    inventory_row = (
        notebook56_feature_inventory_df
        .loc[
            notebook56_feature_inventory_df[
                "strategy"
            ].eq(strategy_name)
        ]
        .iloc[0]
    )

    inventory_feature_names = list(
        inventory_row["retained_features"]
    )

    package_feature_set = set(package_feature_names)
    inventory_feature_set = set(inventory_feature_names)

    only_in_package = sorted(
        package_feature_set - inventory_feature_set
    )

    only_in_inventory = sorted(
        inventory_feature_set - package_feature_set
    )

    positional_mismatches = [
        {
            "position": position,
            "package_feature": package_name,
            "inventory_feature": inventory_name,
        }
        for position, (
            package_name,
            inventory_name,
        ) in enumerate(
            zip(
                package_feature_names,
                inventory_feature_names,
            )
        )
        if package_name != inventory_name
    ]

    feature_name_comparison_records.append(
        {
            "strategy": strategy_name,
            "package_count": len(package_feature_names),
            "inventory_count": len(inventory_feature_names),
            "exact_order_match": (
                package_feature_names
                == inventory_feature_names
            ),
            "same_feature_set": (
                package_feature_set
                == inventory_feature_set
            ),
            "positional_mismatch_count": len(
                positional_mismatches
            ),
            "only_in_package_count": len(
                only_in_package
            ),
            "only_in_inventory_count": len(
                only_in_inventory
            ),
            "only_in_package": only_in_package,
            "only_in_inventory": only_in_inventory,
        }
    )

    print("\n" + "=" * 100)
    print(strategy_name)
    print("=" * 100)

    print(
        f"Package feature count   : "
        f"{len(package_feature_names)}"
    )

    print(
        f"Inventory feature count : "
        f"{len(inventory_feature_names)}"
    )

    print(
        f"Exact ordering matches  : "
        f"{package_feature_names == inventory_feature_names}"
    )

    print(
        f"Same feature set        : "
        f"{package_feature_set == inventory_feature_set}"
    )

    print(
        f"Positional mismatches   : "
        f"{len(positional_mismatches)}"
    )

    if positional_mismatches:
        print("\nFirst 10 positional differences:")

        display(
            pd.DataFrame(
                positional_mismatches
            ).head(10)
        )

    if only_in_package:
        print("\nFeatures only in saved model package:")
        print(only_in_package)

    if only_in_inventory:
        print("\nFeatures only in Notebook 56 inventory:")
        print(only_in_inventory)


section2c3_feature_name_comparison_df = pd.DataFrame(
    feature_name_comparison_records
)

print("\n" + "=" * 100)
print("FEATURE-NAME COMPARISON SUMMARY")
print("=" * 100)

display(
    section2c3_feature_name_comparison_df[
        [
            "strategy",
            "package_count",
            "inventory_count",
            "exact_order_match",
            "same_feature_set",
            "positional_mismatch_count",
            "only_in_package_count",
            "only_in_inventory_count",
        ]
    ]
)

assert (
    section2c3_feature_name_comparison_df[
        "same_feature_set"
    ].all()
), (
    "At least one strategy contains genuinely different "
    "features between the saved model and Notebook 56 inventory."
)

print(
    "\n✅ FEATURE CONTENT MATCHES; "
    "ONLY FEATURE ORDERING DIFFERS"
)


# In[8]:


# ==========================================================================================
# SECTION 3A — EXTRACT NATIVE RANDOM FOREST FEATURE IMPORTANCES
# ==========================================================================================

feature_importance_records = []

for strategy_name, model_package in frozen_policy_models.items():

    estimator = model_package["model"]

    feature_names = list(
        model_package["retained_feature_names"]
    )

    feature_importances = np.asarray(
        estimator.feature_importances_,
        dtype=float,
    )

    assert len(feature_names) == len(feature_importances), (
        f"{strategy_name}: feature-name count does not match "
        "feature-importance count."
    )

    strategy_importance_df = pd.DataFrame(
        {
            "strategy": strategy_name,
            "feature_position": np.arange(
                len(feature_names)
            ),
            "feature_name": feature_names,
            "importance": feature_importances,
        }
    )

    strategy_importance_df["importance_rank"] = (
        strategy_importance_df["importance"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    strategy_importance_df["importance_percent"] = (
        strategy_importance_df["importance"] * 100.0
    )

    strategy_importance_df["feature_type"] = (
        strategy_importance_df["feature_name"]
        .str.split("__", n=1)
        .str[0]
    )

    strategy_importance_df["base_feature"] = (
        strategy_importance_df["feature_name"]
        .str.split("__", n=1)
        .str[-1]
    )

    feature_importance_records.append(
        strategy_importance_df
    )


section3a_feature_importance_df = pd.concat(
    feature_importance_records,
    ignore_index=True,
)

section3a_feature_importance_df = (
    section3a_feature_importance_df
    .sort_values(
        [
            "strategy",
            "importance_rank",
            "feature_name",
        ]
    )
    .reset_index(drop=True)
)

print("=" * 100)
print("SECTION 3A — RANDOM FOREST FEATURE IMPORTANCE EXTRACTION")
print("=" * 100)

importance_validation_df = (
    section3a_feature_importance_df
    .groupby(
        "strategy",
        as_index=False,
    )
    .agg(
        feature_count=(
            "feature_name",
            "count",
        ),
        total_importance=(
            "importance",
            "sum",
        ),
        maximum_importance=(
            "importance",
            "max",
        ),
        minimum_importance=(
            "importance",
            "min",
        ),
        zero_importance_features=(
            "importance",
            lambda values: int(
                np.isclose(
                    values,
                    0.0,
                    atol=1e-15,
                ).sum()
            ),
        ),
    )
)

display(importance_validation_df)

assert np.allclose(
    importance_validation_df[
        "total_importance"
    ].to_numpy(dtype=float),
    1.0,
    atol=1e-8,
), (
    "At least one strategy's extracted feature "
    "importances do not sum to 1."
)

print("\nTOP 15 FEATURES BY STRATEGY")
print("=" * 100)

for strategy_name in frozen_policy_models:

    print("\n" + "-" * 100)
    print(strategy_name)
    print("-" * 100)

    strategy_top_features = (
        section3a_feature_importance_df
        .loc[
            section3a_feature_importance_df[
                "strategy"
            ].eq(strategy_name)
        ]
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(15)
        [
            [
                "importance_rank",
                "feature_name",
                "feature_type",
                "importance",
                "importance_percent",
            ]
        ]
    )

    display(strategy_top_features)

print(
    "\n✅ NATIVE RANDOM FOREST FEATURE IMPORTANCES "
    "EXTRACTED FOR ALL FOUR STRATEGIES"
)


# In[9]:


# ==========================================================================================
# SECTION 3B — CROSS-STRATEGY FEATURE IMPORTANCE COMPARISON
# ==========================================================================================

STRATEGY_ORDER = [
    "R0_CURRENT_BASELINE",
    "R1_REMOVE_LEGAL_SIGNATURE",
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
    "R3_STATE_CENTRIC_POLICY",
]

strategy_short_names = {
    "R0_CURRENT_BASELINE": "R0",
    "R1_REMOVE_LEGAL_SIGNATURE": "R1",
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY": "R2",
    "R3_STATE_CENTRIC_POLICY": "R3",
}

# Build one row per feature and one importance column per strategy.
section3b_importance_pivot_df = (
    section3a_feature_importance_df
    .pivot_table(
        index="feature_name",
        columns="strategy",
        values="importance",
        aggfunc="first",
    )
    .reindex(columns=STRATEGY_ORDER)
    .fillna(0.0)
    .reset_index()
)

section3b_importance_pivot_df = (
    section3b_importance_pivot_df.rename(
        columns=strategy_short_names
    )
)

importance_columns = ["R0", "R1", "R2", "R3"]

section3b_importance_pivot_df["strategies_retained"] = (
    section3b_importance_pivot_df[
        importance_columns
    ]
    .gt(0.0)
    .sum(axis=1)
)

section3b_importance_pivot_df["mean_importance_all_models"] = (
    section3b_importance_pivot_df[
        importance_columns
    ].mean(axis=1)
)

section3b_importance_pivot_df["mean_importance_when_retained"] = (
    section3b_importance_pivot_df[
        importance_columns
    ]
    .replace(0.0, np.nan)
    .mean(axis=1)
)

section3b_importance_pivot_df["maximum_importance"] = (
    section3b_importance_pivot_df[
        importance_columns
    ].max(axis=1)
)

section3b_importance_pivot_df["minimum_retained_importance"] = (
    section3b_importance_pivot_df[
        importance_columns
    ]
    .replace(0.0, np.nan)
    .min(axis=1)
)

section3b_importance_pivot_df["importance_range_when_retained"] = (
    section3b_importance_pivot_df["maximum_importance"]
    - section3b_importance_pivot_df[
        "minimum_retained_importance"
    ]
)

section3b_importance_pivot_df["feature_type"] = (
    section3b_importance_pivot_df[
        "feature_name"
    ]
    .str.split("__", n=1)
    .str[0]
)

section3b_importance_pivot_df["base_feature"] = (
    section3b_importance_pivot_df[
        "feature_name"
    ]
    .str.split("__", n=1)
    .str[-1]
)

section3b_importance_pivot_df = (
    section3b_importance_pivot_df
    .sort_values(
        [
            "mean_importance_when_retained",
            "maximum_importance",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

section3b_importance_pivot_df[
    "cross_strategy_rank"
] = np.arange(
    1,
    len(section3b_importance_pivot_df) + 1,
)

print("=" * 100)
print("SECTION 3B — CROSS-STRATEGY FEATURE IMPORTANCE COMPARISON")
print("=" * 100)

print("\nFEATURE RETENTION SUMMARY")
print("-" * 100)

retention_summary_df = (
    section3b_importance_pivot_df
    .groupby(
        "strategies_retained",
        as_index=False,
    )
    .agg(
        feature_count=(
            "feature_name",
            "count",
        ),
        mean_importance=(
            "mean_importance_when_retained",
            "mean",
        ),
    )
    .sort_values(
        "strategies_retained",
        ascending=False,
    )
)

display(retention_summary_df)

print("\nTOP 20 FEATURES BY MEAN IMPORTANCE WHEN RETAINED")
print("-" * 100)

display(
    section3b_importance_pivot_df[
        [
            "cross_strategy_rank",
            "feature_name",
            "feature_type",
            "strategies_retained",
            "R0",
            "R1",
            "R2",
            "R3",
            "mean_importance_when_retained",
            "maximum_importance",
        ]
    ].head(20)
)

print("\nFEATURES RETAINED IN ALL FOUR STRATEGIES")
print("-" * 100)

section3b_features_retained_all_df = (
    section3b_importance_pivot_df
    .loc[
        section3b_importance_pivot_df[
            "strategies_retained"
        ].eq(4)
    ]
    .sort_values(
        "mean_importance_when_retained",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    section3b_features_retained_all_df[
        [
            "feature_name",
            "feature_type",
            "R0",
            "R1",
            "R2",
            "R3",
            "mean_importance_when_retained",
            "importance_range_when_retained",
        ]
    ].head(25)
)

print("\nFEATURES REMOVED BEFORE R3")
print("-" * 100)

section3b_removed_before_r3_df = (
    section3b_importance_pivot_df
    .loc[
        section3b_importance_pivot_df["R0"].gt(0.0)
        & section3b_importance_pivot_df["R3"].eq(0.0)
    ]
    .sort_values(
        "R0",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    section3b_removed_before_r3_df[
        [
            "feature_name",
            "feature_type",
            "strategies_retained",
            "R0",
            "R1",
            "R2",
            "R3",
            "maximum_importance",
        ]
    ]
)

assert len(
    section3b_importance_pivot_df
) == 64, (
    "Expected 64 unique baseline features."
)

assert (
    section3b_importance_pivot_df[
        "strategies_retained"
    ]
    .between(1, 4)
    .all()
), (
    "Invalid strategy-retention count detected."
)

print(
    "\n✅ CROSS-STRATEGY FEATURE IMPORTANCE "
    "COMPARISON COMPLETE"
)


# In[10]:


# ==========================================================================================
# SECTION 3C — FEATURE IMPORTANCE CONCENTRATION AND REDISTRIBUTION
# ==========================================================================================

def calculate_importance_concentration(
    importance_values,
) -> dict:
    """
    Calculate concentration statistics for a normalized
    feature-importance vector.
    """
    values = np.asarray(
        importance_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
        & (values >= 0.0)
    ]

    total = values.sum()

    if total <= 0.0:
        raise ValueError(
            "Feature importance values must have "
            "a positive total."
        )

    values = values / total
    sorted_values = np.sort(values)[::-1]

    nonzero_values = values[
        values > 0.0
    ]

    entropy = -np.sum(
        nonzero_values
        * np.log(nonzero_values)
    )

    maximum_entropy = np.log(
        len(values)
    )

    normalized_entropy = (
        entropy / maximum_entropy
        if maximum_entropy > 0.0
        else 0.0
    )

    herfindahl_index = float(
        np.sum(values ** 2)
    )

    effective_feature_count = (
        1.0 / herfindahl_index
        if herfindahl_index > 0.0
        else np.nan
    )

    return {
        "feature_count": len(values),
        "top_1_share": float(
            sorted_values[:1].sum()
        ),
        "top_5_share": float(
            sorted_values[:5].sum()
        ),
        "top_10_share": float(
            sorted_values[:10].sum()
        ),
        "top_15_share": float(
            sorted_values[:15].sum()
        ),
        "normalized_entropy": float(
            normalized_entropy
        ),
        "herfindahl_index": (
            herfindahl_index
        ),
        "effective_feature_count": float(
            effective_feature_count
        ),
    }


concentration_records = []

for strategy_name in STRATEGY_ORDER:

    strategy_importance_values = (
        section3a_feature_importance_df
        .loc[
            section3a_feature_importance_df[
                "strategy"
            ].eq(strategy_name),
            "importance",
        ]
        .to_numpy(dtype=float)
    )

    concentration_metrics = (
        calculate_importance_concentration(
            strategy_importance_values
        )
    )

    concentration_records.append(
        {
            "strategy": strategy_name,
            **concentration_metrics,
        }
    )


section3c_concentration_df = pd.DataFrame(
    concentration_records
)

print("=" * 100)
print("SECTION 3C — FEATURE IMPORTANCE CONCENTRATION")
print("=" * 100)

display(section3c_concentration_df)

print("\nINTERPRETATION")
print("-" * 100)

for _, row in section3c_concentration_df.iterrows():

    print(
        f"{row['strategy']}: "
        f"top-5 share={row['top_5_share']:.3f}, "
        f"top-10 share={row['top_10_share']:.3f}, "
        f"effective features="
        f"{row['effective_feature_count']:.2f}, "
        f"normalized entropy="
        f"{row['normalized_entropy']:.3f}"
    )


# ------------------------------------------------------------------------------------------
# IMPORTANCE REDISTRIBUTION FOR FEATURES RETAINED IN ALL FOUR STRATEGIES
# ------------------------------------------------------------------------------------------

section3c_retained_redistribution_df = (
    section3b_features_retained_all_df.copy()
)

section3c_retained_redistribution_df[
    "absolute_change_r0_to_r3"
] = (
    section3c_retained_redistribution_df["R3"]
    - section3c_retained_redistribution_df["R0"]
)

section3c_retained_redistribution_df[
    "relative_change_r0_to_r3"
] = np.where(
    section3c_retained_redistribution_df[
        "R0"
    ].gt(0.0),
    (
        section3c_retained_redistribution_df[
            "R3"
        ]
        / section3c_retained_redistribution_df[
            "R0"
        ]
    ),
    np.nan,
)

section3c_retained_redistribution_df[
    "importance_increased_r0_to_r3"
] = (
    section3c_retained_redistribution_df[
        "absolute_change_r0_to_r3"
    ].gt(0.0)
)

section3c_retained_redistribution_df = (
    section3c_retained_redistribution_df
    .sort_values(
        "absolute_change_r0_to_r3",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 100)
print("LARGEST IMPORTANCE GAINS FROM R0 TO R3")
print("=" * 100)

display(
    section3c_retained_redistribution_df[
        [
            "feature_name",
            "feature_type",
            "R0",
            "R3",
            "absolute_change_r0_to_r3",
            "relative_change_r0_to_r3",
            "importance_increased_r0_to_r3",
        ]
    ].head(20)
)

print("\n" + "=" * 100)
print("LARGEST IMPORTANCE DECREASES FROM R0 TO R3")
print("=" * 100)

display(
    section3c_retained_redistribution_df
    .sort_values(
        "absolute_change_r0_to_r3",
        ascending=True,
    )
    [
        [
            "feature_name",
            "feature_type",
            "R0",
            "R3",
            "absolute_change_r0_to_r3",
            "relative_change_r0_to_r3",
            "importance_increased_r0_to_r3",
        ]
    ]
    .head(15)
)


# ------------------------------------------------------------------------------------------
# FEATURE-TYPE IMPORTANCE SHARE
# ------------------------------------------------------------------------------------------

section3c_feature_type_summary_df = (
    section3a_feature_importance_df
    .groupby(
        [
            "strategy",
            "feature_type",
        ],
        as_index=False,
    )
    .agg(
        feature_count=(
            "feature_name",
            "count",
        ),
        total_importance=(
            "importance",
            "sum",
        ),
        mean_importance=(
            "importance",
            "mean",
        ),
        maximum_importance=(
            "importance",
            "max",
        ),
    )
)

section3c_feature_type_summary_df[
    "importance_percent"
] = (
    section3c_feature_type_summary_df[
        "total_importance"
    ]
    * 100.0
)

print("\n" + "=" * 100)
print("IMPORTANCE SHARE BY FEATURE TYPE")
print("=" * 100)

display(section3c_feature_type_summary_df)

assert np.allclose(
    section3c_feature_type_summary_df
    .groupby("strategy")[
        "total_importance"
    ]
    .sum()
    .to_numpy(dtype=float),
    1.0,
    atol=1e-8,
), (
    "Feature-type importance shares do not "
    "sum to 1 for every strategy."
)

print(
    "\n✅ FEATURE IMPORTANCE CONCENTRATION "
    "AND REDISTRIBUTION ANALYSIS COMPLETE"
)


# In[12]:


# ==========================================================================================
# SECTION 4A — FEATURE REMOVAL ATTRIBUTION BY ABLATION STAGE
# ==========================================================================================

print("=" * 100)
print("SECTION 4A — FEATURE REMOVAL ATTRIBUTION BY ABLATION STAGE")
print("=" * 100)

# ----------------------------------------------------------------------
# Reconstruct required feature-name dictionary after kernel restart
# ----------------------------------------------------------------------

assert "frozen_policy_models" in globals(), (
    "frozen_policy_models is missing. "
    "Run Section 2C.1 first."
)

frozen_policy_feature_names = {
    strategy_name: list(
        model_package["retained_feature_names"]
    )
    for strategy_name, model_package
    in frozen_policy_models.items()
}

required_strategies = [
    "R0_CURRENT_BASELINE",
    "R1_REMOVE_LEGAL_SIGNATURE",
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
    "R3_STATE_CENTRIC_POLICY",
]

missing_strategies = [
    strategy
    for strategy in required_strategies
    if strategy not in frozen_policy_feature_names
]

assert not missing_strategies, (
    "Missing strategy feature lists: "
    f"{missing_strategies}"
)

# ----------------------------------------------------------------------
# Define sequential ablation transitions
# ----------------------------------------------------------------------

ABLATION_TRANSITIONS = [
    {
        "transition": "R0_TO_R1",
        "strategy_before": "R0_CURRENT_BASELINE",
        "strategy_after": "R1_REMOVE_LEGAL_SIGNATURE",
        "removed_family": "legal_signature",
    },
    {
        "transition": "R1_TO_R2",
        "strategy_before": "R1_REMOVE_LEGAL_SIGNATURE",
        "strategy_after": "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
        "removed_family": "compatibility",
    },
    {
        "transition": "R2_TO_R3",
        "strategy_before": "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
        "strategy_after": "R3_STATE_CENTRIC_POLICY",
        "removed_family": "direct_legality",
    },
]

# ----------------------------------------------------------------------
# Section 3A must already have produced feature importances
# ----------------------------------------------------------------------

assert "section3a_feature_importance_df" in globals(), (
    "section3a_feature_importance_df is missing. "
    "Run Section 3A before Section 4A."
)

removal_attribution_records = []

for transition_info in ABLATION_TRANSITIONS:

    strategy_before = transition_info["strategy_before"]
    strategy_after = transition_info["strategy_after"]

    features_before = set(
        frozen_policy_feature_names[strategy_before]
    )

    features_after = set(
        frozen_policy_feature_names[strategy_after]
    )

    removed_features = sorted(
        features_before - features_after
    )

    newly_added_features = sorted(
        features_after - features_before
    )

    assert not newly_added_features, (
        f"{transition_info['transition']} unexpectedly "
        f"introduced features: {newly_added_features}"
    )

    before_importance_lookup = (
        section3a_feature_importance_df
        .loc[
            section3a_feature_importance_df[
                "strategy"
            ].eq(strategy_before)
        ]
        .set_index("feature_name")["importance"]
        .to_dict()
    )

    for removed_feature in removed_features:

        assert removed_feature in before_importance_lookup, (
            f"{removed_feature} is missing from the "
            f"{strategy_before} importance table."
        )

        importance_before_removal = float(
            before_importance_lookup[removed_feature]
        )

        removal_attribution_records.append(
            {
                "transition": transition_info["transition"],
                "strategy_before": strategy_before,
                "strategy_after": strategy_after,
                "removed_family": transition_info[
                    "removed_family"
                ],
                "feature_name": removed_feature,
                "feature_type": removed_feature.split(
                    "__",
                    1,
                )[0],
                "importance_before_removal": (
                    importance_before_removal
                ),
                "importance_percent_before_removal": (
                    importance_before_removal * 100.0
                ),
            }
        )

# ----------------------------------------------------------------------
# Detailed removal table
# ----------------------------------------------------------------------

section4a_removed_feature_attribution_df = (
    pd.DataFrame(removal_attribution_records)
    .sort_values(
        [
            "transition",
            "importance_before_removal",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

# ----------------------------------------------------------------------
# Transition summary
# ----------------------------------------------------------------------

section4a_transition_summary_df = (
    section4a_removed_feature_attribution_df
    .groupby(
        [
            "transition",
            "strategy_before",
            "strategy_after",
            "removed_family",
        ],
        as_index=False,
    )
    .agg(
        removed_feature_count=(
            "feature_name",
            "count",
        ),
        total_importance_removed=(
            "importance_before_removal",
            "sum",
        ),
        mean_importance_removed=(
            "importance_before_removal",
            "mean",
        ),
        maximum_removed_importance=(
            "importance_before_removal",
            "max",
        ),
    )
)

section4a_transition_summary_df[
    "total_importance_removed_percent"
] = (
    section4a_transition_summary_df[
        "total_importance_removed"
    ]
    * 100.0
)

print("\nABLATION-STAGE SUMMARY")
print("-" * 100)

display(section4a_transition_summary_df)

print("\nREMOVED FEATURES BY STAGE")
print("=" * 100)

for transition_info in ABLATION_TRANSITIONS:

    transition_name = transition_info[
        "transition"
    ]

    print("\n" + "-" * 100)

    print(
        f"{transition_name}: "
        f"{transition_info['strategy_before']} "
        f"→ {transition_info['strategy_after']}"
    )

    print("-" * 100)

    display(
        section4a_removed_feature_attribution_df
        .loc[
            section4a_removed_feature_attribution_df[
                "transition"
            ].eq(transition_name)
        ]
        [
            [
                "feature_name",
                "feature_type",
                "removed_family",
                "importance_before_removal",
                "importance_percent_before_removal",
            ]
        ]
    )

# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

expected_removed_counts = {
    "R0_TO_R1": 6,
    "R1_TO_R2": 6,
    "R2_TO_R3": 6,
}

observed_removed_counts = (
    section4a_transition_summary_df
    .set_index("transition")[
        "removed_feature_count"
    ]
    .to_dict()
)

assert observed_removed_counts == expected_removed_counts, (
    "Unexpected feature-removal counts detected.\n"
    f"Expected: {expected_removed_counts}\n"
    f"Observed: {observed_removed_counts}"
)

assert len(
    section4a_removed_feature_attribution_df
) == 18, (
    "Expected exactly 18 cumulatively removed features."
)

print(
    "\n✅ FEATURE REMOVAL ATTRIBUTION "
    "BY ABLATION STAGE COMPLETE"
)


# In[13]:


# ==========================================================================================
# SECTION 4B — LINK FEATURE REMOVAL IMPORTANCE TO NOTEBOOK 56 PROBABILITY DRIFT
# ==========================================================================================

transition_pair_mapping = {
    "R0_TO_R1": (
        "R0_CURRENT_BASELINE",
        "R1_REMOVE_LEGAL_SIGNATURE",
    ),
    "R1_TO_R2": (
        "R1_REMOVE_LEGAL_SIGNATURE",
        "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
    ),
    "R2_TO_R3": (
        "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
        "R3_STATE_CENTRIC_POLICY",
    ),
}


def get_pair_summary_row(
    pair_summary_df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
) -> pd.Series:
    """
    Return the unique pair-summary row regardless of stored direction.
    """
    direct_mask = (
        pair_summary_df["strategy_a"].eq(strategy_a)
        & pair_summary_df["strategy_b"].eq(strategy_b)
    )

    reverse_mask = (
        pair_summary_df["strategy_a"].eq(strategy_b)
        & pair_summary_df["strategy_b"].eq(strategy_a)
    )

    matched_rows = pair_summary_df.loc[
        direct_mask | reverse_mask
    ]

    assert len(matched_rows) == 1, (
        f"Expected exactly one pair-summary row for "
        f"{strategy_a} and {strategy_b}, found {len(matched_rows)}."
    )

    return matched_rows.iloc[0]


drift_link_records = []

for _, removal_row in (
    section4a_transition_summary_df.iterrows()
):

    transition_name = removal_row["transition"]

    strategy_before, strategy_after = (
        transition_pair_mapping[
            transition_name
        ]
    )

    pair_row = get_pair_summary_row(
        notebook56_pair_drift_summary_df,
        strategy_before,
        strategy_after,
    )

    drift_link_records.append(
        {
            "transition": transition_name,
            "strategy_before": strategy_before,
            "strategy_after": strategy_after,
            "removed_family": removal_row[
                "removed_family"
            ],
            "removed_feature_count": int(
                removal_row[
                    "removed_feature_count"
                ]
            ),
            "total_importance_removed": float(
                removal_row[
                    "total_importance_removed"
                ]
            ),
            "total_importance_removed_percent": float(
                removal_row[
                    "total_importance_removed_percent"
                ]
            ),
            "mean_maximum_case_difference": float(
                pair_row[
                    "mean_maximum_case_difference"
                ]
            ),
            "median_maximum_case_difference": float(
                pair_row[
                    "median_maximum_case_difference"
                ]
            ),
            "maximum_case_difference": float(
                pair_row[
                    "maximum_case_difference"
                ]
            ),
            "mean_l1_case_difference": float(
                pair_row[
                    "mean_l1_case_difference"
                ]
            ),
            "cases_above_005": int(
                pair_row[
                    "cases_above_005"
                ]
            ),
            "cases_above_010": int(
                pair_row[
                    "cases_above_010"
                ]
            ),
            "cases_above_015": int(
                pair_row[
                    "cases_above_015"
                ]
            ),
        }
    )


section4b_importance_drift_link_df = pd.DataFrame(
    drift_link_records
)

section4b_importance_drift_link_df[
    "mean_max_drift_per_importance_removed"
] = (
    section4b_importance_drift_link_df[
        "mean_maximum_case_difference"
    ]
    / section4b_importance_drift_link_df[
        "total_importance_removed"
    ]
)

section4b_importance_drift_link_df[
    "mean_l1_drift_per_importance_removed"
] = (
    section4b_importance_drift_link_df[
        "mean_l1_case_difference"
    ]
    / section4b_importance_drift_link_df[
        "total_importance_removed"
    ]
)

section4b_importance_drift_link_df[
    "fraction_cases_above_005"
] = (
    section4b_importance_drift_link_df[
        "cases_above_005"
    ]
    / 184.0
)

section4b_importance_drift_link_df[
    "fraction_cases_above_010"
] = (
    section4b_importance_drift_link_df[
        "cases_above_010"
    ]
    / 184.0
)

section4b_importance_drift_link_df[
    "fraction_cases_above_015"
] = (
    section4b_importance_drift_link_df[
        "cases_above_015"
    ]
    / 184.0
)

print("=" * 100)
print("SECTION 4B — REMOVED IMPORTANCE VS PROBABILITY DRIFT")
print("=" * 100)

display(section4b_importance_drift_link_df)

print("\nTRANSITION INTERPRETATION")
print("-" * 100)

for _, row in (
    section4b_importance_drift_link_df.iterrows()
):

    print(
        f"{row['transition']} "
        f"({row['removed_family']}): "
        f"removed_importance="
        f"{row['total_importance_removed_percent']:.2f}%, "
        f"mean_max_drift="
        f"{row['mean_maximum_case_difference']:.4f}, "
        f"max_drift="
        f"{row['maximum_case_difference']:.4f}, "
        f"cases>0.05="
        f"{row['cases_above_005']}/184"
    )


# Descriptive correlation only: there are just three transitions.
section4b_descriptive_correlations_df = pd.DataFrame(
    [
        {
            "metric": "mean_maximum_case_difference",
            "pearson_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "mean_maximum_case_difference"
                    ],
                    method="pearson",
                )
            ),
            "spearman_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "mean_maximum_case_difference"
                    ],
                    method="spearman",
                )
            ),
        },
        {
            "metric": "mean_l1_case_difference",
            "pearson_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "mean_l1_case_difference"
                    ],
                    method="pearson",
                )
            ),
            "spearman_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "mean_l1_case_difference"
                    ],
                    method="spearman",
                )
            ),
        },
        {
            "metric": "maximum_case_difference",
            "pearson_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "maximum_case_difference"
                    ],
                    method="pearson",
                )
            ),
            "spearman_correlation": (
                section4b_importance_drift_link_df[
                    "total_importance_removed"
                ]
                .corr(
                    section4b_importance_drift_link_df[
                        "maximum_case_difference"
                    ],
                    method="spearman",
                )
            ),
        },
    ]
)

print("\nDESCRIPTIVE CORRELATIONS")
print("-" * 100)
print(
    "Note: These are descriptive only because there are "
    "just three sequential ablation transitions."
)

display(section4b_descriptive_correlations_df)

assert (
    section4b_importance_drift_link_df[
        "removed_feature_count"
    ].eq(6).all()
)

assert (
    section4b_importance_drift_link_df[
        "maximum_case_difference"
    ].between(0.0, 1.0).all()
)

print(
    "\n✅ REMOVED FEATURE IMPORTANCE SUCCESSFULLY "
    "LINKED TO PROBABILITY DRIFT"
)


# In[14]:


# ==========================================================================================
# SECTION 5A — HIGHEST PROBABILITY DRIFT CASES
# ==========================================================================================

print("=" * 100)
print("SECTION 5A — CASE-LEVEL PROBABILITY DRIFT ANALYSIS")
print("=" * 100)

case_drift_df = notebook56_case_probability_drift_df.copy()

required_columns = [
    "comparison_case_id",
    "strategy_a",
    "strategy_b",
    "maximum_case_difference",
    "mean_case_difference",
    "l1_case_difference",
    "prediction_a",
    "prediction_b",
    "predictions_match",
]

missing_columns = [
    column
    for column in required_columns
    if column not in case_drift_df.columns
]

assert not missing_columns, (
    "Notebook 56 case-drift table is missing required columns: "
    f"{missing_columns}"
)

# Standardized Notebook 57 names.
case_drift_df["maximum_probability_difference"] = pd.to_numeric(
    case_drift_df["maximum_case_difference"],
    errors="coerce",
)

case_drift_df["mean_probability_difference"] = pd.to_numeric(
    case_drift_df["mean_case_difference"],
    errors="coerce",
)

case_drift_df["l1_probability_difference"] = pd.to_numeric(
    case_drift_df["l1_case_difference"],
    errors="coerce",
)

numeric_columns = [
    "maximum_probability_difference",
    "mean_probability_difference",
    "l1_probability_difference",
]

assert case_drift_df[numeric_columns].notna().all().all(), (
    "One or more case-drift values could not be converted to numeric."
)

section5a_high_drift_cases_df = (
    case_drift_df
    .sort_values(
        [
            "maximum_probability_difference",
            "l1_probability_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

section5a_high_drift_cases_df["global_rank"] = np.arange(
    1,
    len(section5a_high_drift_cases_df) + 1,
)

print("\nTOP 25 HIGHEST-DRIFT CASE COMPARISONS")
print("-" * 100)

display(
    section5a_high_drift_cases_df[
        [
            "global_rank",
            "comparison_case_id",
            "strategy_a",
            "strategy_b",
            "prediction_a",
            "prediction_b",
            "predictions_match",
            "maximum_probability_difference",
            "mean_probability_difference",
            "l1_probability_difference",
        ]
    ].head(25)
)

# ------------------------------------------------------------------------------------------
# Overall summary
# ------------------------------------------------------------------------------------------

section5a_overall_summary_df = pd.DataFrame(
    [
        {
            "case_comparisons": len(section5a_high_drift_cases_df),
            "unique_evaluation_cases": (
                section5a_high_drift_cases_df[
                    "comparison_case_id"
                ].nunique()
            ),
            "mean_maximum_probability_difference": (
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].mean()
            ),
            "median_maximum_probability_difference": (
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].median()
            ),
            "maximum_probability_difference": (
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].max()
            ),
            "mean_l1_probability_difference": (
                section5a_high_drift_cases_df[
                    "l1_probability_difference"
                ].mean()
            ),
            "maximum_l1_probability_difference": (
                section5a_high_drift_cases_df[
                    "l1_probability_difference"
                ].max()
            ),
            "prediction_disagreements": int(
                (~section5a_high_drift_cases_df[
                    "predictions_match"
                ].astype(bool)).sum()
            ),
        }
    ]
)

print("\nOVERALL CASE-DRIFT SUMMARY")
print("-" * 100)

display(section5a_overall_summary_df)

# ------------------------------------------------------------------------------------------
# Drift by strategy pair
# ------------------------------------------------------------------------------------------

section5a_pair_summary_df = (
    section5a_high_drift_cases_df
    .groupby(
        ["strategy_a", "strategy_b"],
        as_index=False,
    )
    .agg(
        cases=(
            "comparison_case_id",
            "count",
        ),
        mean_maximum_probability_difference=(
            "maximum_probability_difference",
            "mean",
        ),
        median_maximum_probability_difference=(
            "maximum_probability_difference",
            "median",
        ),
        maximum_probability_difference=(
            "maximum_probability_difference",
            "max",
        ),
        mean_l1_probability_difference=(
            "l1_probability_difference",
            "mean",
        ),
        maximum_l1_probability_difference=(
            "l1_probability_difference",
            "max",
        ),
        prediction_disagreements=(
            "predictions_match",
            lambda values: int(
                (~values.astype(bool)).sum()
            ),
        ),
    )
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nDRIFT BY STRATEGY PAIR")
print("-" * 100)

display(section5a_pair_summary_df)

# ------------------------------------------------------------------------------------------
# Threshold counts
# ------------------------------------------------------------------------------------------

section5a_threshold_summary_df = pd.DataFrame(
    [
        {
            "threshold": 0.05,
            "case_comparisons_above_threshold": int(
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].gt(0.05).sum()
            ),
        },
        {
            "threshold": 0.10,
            "case_comparisons_above_threshold": int(
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].gt(0.10).sum()
            ),
        },
        {
            "threshold": 0.15,
            "case_comparisons_above_threshold": int(
                section5a_high_drift_cases_df[
                    "maximum_probability_difference"
                ].gt(0.15).sum()
            ),
        },
    ]
)

section5a_threshold_summary_df[
    "fraction_of_all_comparisons"
] = (
    section5a_threshold_summary_df[
        "case_comparisons_above_threshold"
    ]
    / len(section5a_high_drift_cases_df)
)

print("\nDRIFT THRESHOLD SUMMARY")
print("-" * 100)

display(section5a_threshold_summary_df)

# Notebook 56 evaluated 184 states across 6 pairwise comparisons.
assert len(section5a_high_drift_cases_df) == 1104, (
    "Expected 1,104 case-comparison rows."
)

assert (
    section5a_high_drift_cases_df[
        "comparison_case_id"
    ].nunique()
    == 184
), "Expected 184 unique evaluation cases."

assert section5a_high_drift_cases_df[
    "predictions_match"
].astype(bool).all(), (
    "Unexpected prediction disagreement detected."
)

print(
    "\n✅ CASE-LEVEL PROBABILITY DRIFT ANALYSIS COMPLETE"
)


# In[15]:


# ==========================================================================================
# SECTION 5B — PARSE AND PROFILE HIGH-DRIFT CASE IDENTIFIERS
# ==========================================================================================

import re


def parse_comparison_case_id(
    case_id: str,
) -> dict:
    """
    Parse controlled evaluation case identifiers such as:

        4WE_PAIR_014_V03__PLAYER
        4WE_PAIR_002_V02__OPPONENT

    One or more underscores are allowed before the side label.
    """
    case_id_text = str(case_id).strip()

    pattern = re.compile(
        r"^(?P<scenario_family>.+?)"
        r"_PAIR_(?P<pair_number>\d+)"
        r"_(?P<case_variant>V\d+)"
        r"_+(?P<evaluation_side>PLAYER|OPPONENT)$",
        flags=re.IGNORECASE,
    )

    match = pattern.match(case_id_text)

    if match is None:
        return {
            "parsed_successfully": False,
            "scenario_family": None,
            "pair_number": None,
            "case_variant": None,
            "evaluation_side": None,
            "original_case_id": case_id_text,
        }

    parsed = match.groupdict()

    return {
        "parsed_successfully": True,
        "scenario_family": parsed["scenario_family"].upper(),
        "pair_number": int(parsed["pair_number"]),
        "case_variant": parsed["case_variant"].upper(),
        "evaluation_side": parsed["evaluation_side"].upper(),
        "original_case_id": case_id_text,
    }


parsed_case_metadata_df = pd.DataFrame(
    [
        parse_comparison_case_id(case_id)
        for case_id in section5a_high_drift_cases_df[
            "comparison_case_id"
        ]
    ]
)

section5b_case_profile_df = pd.concat(
    [
        section5a_high_drift_cases_df.reset_index(drop=True),
        parsed_case_metadata_df[
            [
                "parsed_successfully",
                "scenario_family",
                "pair_number",
                "case_variant",
                "evaluation_side",
            ]
        ].reset_index(drop=True),
    ],
    axis=1,
)

print("=" * 100)
print("SECTION 5B — HIGH-DRIFT CASE IDENTIFIER PROFILING")
print("=" * 100)

parse_summary_df = pd.DataFrame(
    [
        {
            "case_comparison_rows": len(
                section5b_case_profile_df
            ),
            "successfully_parsed_rows": int(
                section5b_case_profile_df[
                    "parsed_successfully"
                ].sum()
            ),
            "failed_parse_rows": int(
                (
                    ~section5b_case_profile_df[
                        "parsed_successfully"
                    ]
                ).sum()
            ),
            "unique_pair_numbers": (
                section5b_case_profile_df[
                    "pair_number"
                ].nunique()
            ),
            "unique_case_variants": (
                section5b_case_profile_df[
                    "case_variant"
                ].nunique()
            ),
            "unique_evaluation_sides": (
                section5b_case_profile_df[
                    "evaluation_side"
                ].nunique()
            ),
        }
    ]
)

display(parse_summary_df)

failed_parse_df = (
    section5b_case_profile_df
    .loc[
        ~section5b_case_profile_df[
            "parsed_successfully"
        ],
        ["comparison_case_id"],
    ]
    .drop_duplicates()
)

if not failed_parse_df.empty:
    print("\nUNPARSED CASE IDS")
    print("-" * 100)
    display(failed_parse_df.head(20))

assert section5b_case_profile_df[
    "parsed_successfully"
].all(), (
    "At least one comparison_case_id could not be parsed."
)

# ------------------------------------------------------------------------------------------
# Drift by evaluation side
# ------------------------------------------------------------------------------------------

section5b_side_drift_summary_df = (
    section5b_case_profile_df
    .groupby(
        "evaluation_side",
        as_index=False,
    )
    .agg(
        case_comparisons=(
            "comparison_case_id",
            "count",
        ),
        unique_cases=(
            "comparison_case_id",
            "nunique",
        ),
        mean_maximum_probability_difference=(
            "maximum_probability_difference",
            "mean",
        ),
        median_maximum_probability_difference=(
            "maximum_probability_difference",
            "median",
        ),
        maximum_probability_difference=(
            "maximum_probability_difference",
            "max",
        ),
        mean_l1_probability_difference=(
            "l1_probability_difference",
            "mean",
        ),
    )
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nDRIFT BY EVALUATION SIDE")
print("-" * 100)

display(section5b_side_drift_summary_df)

# ------------------------------------------------------------------------------------------
# Drift by case variant
# ------------------------------------------------------------------------------------------

section5b_variant_drift_summary_df = (
    section5b_case_profile_df
    .groupby(
        "case_variant",
        as_index=False,
    )
    .agg(
        case_comparisons=(
            "comparison_case_id",
            "count",
        ),
        unique_cases=(
            "comparison_case_id",
            "nunique",
        ),
        mean_maximum_probability_difference=(
            "maximum_probability_difference",
            "mean",
        ),
        median_maximum_probability_difference=(
            "maximum_probability_difference",
            "median",
        ),
        maximum_probability_difference=(
            "maximum_probability_difference",
            "max",
        ),
        mean_l1_probability_difference=(
            "l1_probability_difference",
            "mean",
        ),
    )
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nDRIFT BY CASE VARIANT")
print("-" * 100)

display(section5b_variant_drift_summary_df)

# ------------------------------------------------------------------------------------------
# Drift by case variant and evaluation side
# ------------------------------------------------------------------------------------------

section5b_variant_side_summary_df = (
    section5b_case_profile_df
    .groupby(
        [
            "case_variant",
            "evaluation_side",
        ],
        as_index=False,
    )
    .agg(
        case_comparisons=(
            "comparison_case_id",
            "count",
        ),
        unique_cases=(
            "comparison_case_id",
            "nunique",
        ),
        mean_maximum_probability_difference=(
            "maximum_probability_difference",
            "mean",
        ),
        maximum_probability_difference=(
            "maximum_probability_difference",
            "max",
        ),
        mean_l1_probability_difference=(
            "l1_probability_difference",
            "mean",
        ),
    )
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nDRIFT BY CASE VARIANT AND SIDE")
print("-" * 100)

display(section5b_variant_side_summary_df)

# ------------------------------------------------------------------------------------------
# Highest-drift unique evaluation states
# ------------------------------------------------------------------------------------------

section5b_unique_case_summary_df = (
    section5b_case_profile_df
    .groupby(
        [
            "comparison_case_id",
            "scenario_family",
            "pair_number",
            "case_variant",
            "evaluation_side",
        ],
        as_index=False,
    )
    .agg(
        strategy_comparisons=(
            "strategy_a",
            "count",
        ),
        mean_maximum_probability_difference=(
            "maximum_probability_difference",
            "mean",
        ),
        maximum_probability_difference=(
            "maximum_probability_difference",
            "max",
        ),
        mean_l1_probability_difference=(
            "l1_probability_difference",
            "mean",
        ),
        maximum_l1_probability_difference=(
            "l1_probability_difference",
            "max",
        ),
    )
    .sort_values(
        [
            "maximum_probability_difference",
            "mean_maximum_probability_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

section5b_unique_case_summary_df[
    "unique_case_rank"
] = np.arange(
    1,
    len(section5b_unique_case_summary_df) + 1,
)

print("\nTOP 25 UNIQUE HIGH-DRIFT EVALUATION STATES")
print("-" * 100)

display(
    section5b_unique_case_summary_df[
        [
            "unique_case_rank",
            "comparison_case_id",
            "pair_number",
            "case_variant",
            "evaluation_side",
            "strategy_comparisons",
            "mean_maximum_probability_difference",
            "maximum_probability_difference",
            "mean_l1_probability_difference",
            "maximum_l1_probability_difference",
        ]
    ].head(25)
)

# ------------------------------------------------------------------------------------------
# Unique-case threshold summary
# ------------------------------------------------------------------------------------------

unique_case_threshold_records = []

for threshold in [0.05, 0.10, 0.15]:

    above_threshold_df = (
        section5b_unique_case_summary_df
        .loc[
            section5b_unique_case_summary_df[
                "maximum_probability_difference"
            ].gt(threshold)
        ]
    )

    unique_case_threshold_records.append(
        {
            "threshold": threshold,
            "unique_cases_above_threshold": len(
                above_threshold_df
            ),
            "player_cases_above_threshold": int(
                above_threshold_df[
                    "evaluation_side"
                ].eq("PLAYER").sum()
            ),
            "opponent_cases_above_threshold": int(
                above_threshold_df[
                    "evaluation_side"
                ].eq("OPPONENT").sum()
            ),
            "fraction_of_184_cases": (
                len(above_threshold_df) / 184.0
            ),
        }
    )

section5b_unique_case_threshold_df = pd.DataFrame(
    unique_case_threshold_records
)

print("\nUNIQUE EVALUATION CASE THRESHOLDS")
print("-" * 100)

display(section5b_unique_case_threshold_df)

assert len(
    section5b_unique_case_summary_df
) == 184, (
    "Expected exactly 184 unique controlled evaluation cases."
)

assert set(
    section5b_case_profile_df[
        "evaluation_side"
    ].unique()
) == {
    "PLAYER",
    "OPPONENT",
}, (
    "Unexpected evaluation-side labels detected."
)

print(
    "\n✅ HIGH-DRIFT CASE IDENTIFIER PROFILING COMPLETE"
)


# In[16]:


# ==============================================================================
# SECTION 6A — ACTION-LEVEL PROBABILITY DRIFT ANALYSIS
# ==============================================================================

print("=" * 100)
print("SECTION 6A — ACTION-LEVEL PROBABILITY DRIFT ANALYSIS")
print("=" * 100)

import pandas as pd
import numpy as np


# In[17]:


# ==============================================================================
# VERIFY REQUIRED DATA
# ==============================================================================

assert "notebook56_case_probability_drift_df" in globals(), (
    "Notebook 56 probability drift dataframe not found."
)

display(
    notebook56_case_probability_drift_df.head()
)


# In[18]:


# ==============================================================================
# LOCATE DRIFT COLUMNS
# ==============================================================================

def locate_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"None of these columns were found:\n{candidates}"
    )


max_col = locate_column(
    notebook56_case_probability_drift_df,
    [
        "maximum_case_difference",
        "maximum_probability_difference",
        "maximum_class_probability_difference",
        "max_probability_difference"
    ]
)

mean_col = locate_column(
    notebook56_case_probability_drift_df,
    [
        "mean_case_difference",
        "mean_probability_difference"
    ]
)

l1_col = locate_column(
    notebook56_case_probability_drift_df,
    [
        "l1_case_difference",
        "l1_probability_difference"
    ]
)

print(max_col)
print(mean_col)
print(l1_col)


# In[19]:


# ==============================================================================
# ACTION-LEVEL SUMMARY
# ==============================================================================

section6a_action_summary_df = (
    notebook56_case_probability_drift_df
    .groupby("prediction_a")
    .agg(
        case_count=("prediction_a","size"),
        mean_maximum_probability_difference=(max_col,"mean"),
        median_maximum_probability_difference=(max_col,"median"),
        maximum_probability_difference=(max_col,"max"),
        mean_l1_probability_difference=(l1_col,"mean")
    )
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False
    )
    .reset_index()
)

display(section6a_action_summary_df)


# In[20]:


# ==============================================================================
# INTERPRETATION
# ==============================================================================

print()

for _, row in section6a_action_summary_df.iterrows():

    print(
        f"{row['prediction_a']}: "
        f"{row['case_count']} cases | "
        f"mean drift={row['mean_maximum_probability_difference']:.4f} | "
        f"max drift={row['maximum_probability_difference']:.4f}"
    )

print()
print("✅ ACTION-LEVEL PROBABILITY DRIFT SUMMARY COMPLETE")


# In[21]:


# ==========================================================================================
# SECTION 6B — TRUE ACTION-LEVEL PROBABILITY SENSITIVITY
# ==========================================================================================

print("=" * 100)
print("SECTION 6B — TRUE ACTION-LEVEL PROBABILITY SENSITIVITY")
print("=" * 100)

assert "notebook56_action_probability_drift_df" in globals(), (
    "notebook56_action_probability_drift_df is missing. "
    "Rerun Section 1B."
)

action_drift_df = notebook56_action_probability_drift_df.copy()

print("\nAVAILABLE ACTION-DRIFT COLUMNS")
print("-" * 100)
print(list(action_drift_df.columns))

display(action_drift_df.head(12))


def find_existing_column(
    dataframe,
    candidates,
    label,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        f"Unable to locate {label}.\n"
        f"Candidates: {candidates}\n"
        f"Available: {list(dataframe.columns)}"
    )


action_col = find_existing_column(
    action_drift_df,
    [
        "action",
        "action_name",
        "class",
        "class_name",
    ],
    "action column",
)

mean_drift_col = find_existing_column(
    action_drift_df,
    [
        "mean_probability_difference",
        "mean_absolute_probability_difference",
        "mean_probability_drift",
        "mean_absolute_difference",
        "mean_difference",
    ],
    "mean action-drift column",
)

max_drift_col = find_existing_column(
    action_drift_df,
    [
        "maximum_probability_difference",
        "max_probability_difference",
        "maximum_absolute_probability_difference",
        "max_absolute_probability_difference",
        "maximum_difference",
    ],
    "maximum action-drift column",
)

print("\nDETECTED COLUMNS")
print("-" * 100)
print(f"Action column       : {action_col}")
print(f"Mean drift column   : {mean_drift_col}")
print(f"Maximum drift column: {max_drift_col}")


# ------------------------------------------------------------------------------------------
# Aggregate each action across all six pairwise strategy comparisons.
# ------------------------------------------------------------------------------------------

section6b_action_sensitivity_df = (
    action_drift_df
    .groupby(
        action_col,
        as_index=False,
    )
    .agg(
        strategy_pair_rows=(
            action_col,
            "count",
        ),
        mean_probability_drift=(
            mean_drift_col,
            "mean",
        ),
        median_probability_drift=(
            mean_drift_col,
            "median",
        ),
        maximum_mean_probability_drift=(
            mean_drift_col,
            "max",
        ),
        maximum_probability_drift=(
            max_drift_col,
            "max",
        ),
    )
    .sort_values(
        [
            "mean_probability_drift",
            "maximum_probability_drift",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

section6b_action_sensitivity_df[
    "action_sensitivity_rank"
] = np.arange(
    1,
    len(section6b_action_sensitivity_df) + 1,
)

print("\nACTION SENSITIVITY RANKING")
print("-" * 100)

display(
    section6b_action_sensitivity_df[
        [
            "action_sensitivity_rank",
            action_col,
            "strategy_pair_rows",
            "mean_probability_drift",
            "median_probability_drift",
            "maximum_mean_probability_drift",
            "maximum_probability_drift",
        ]
    ]
)


# ------------------------------------------------------------------------------------------
# Show complete action × strategy-pair detail.
# ------------------------------------------------------------------------------------------

detail_columns = [
    column
    for column in [
        "strategy_a",
        "strategy_b",
        action_col,
        mean_drift_col,
        max_drift_col,
    ]
    if column in action_drift_df.columns
]

section6b_action_pair_detail_df = (
    action_drift_df[
        detail_columns
    ]
    .sort_values(
        mean_drift_col,
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nMOST SENSITIVE ACTION / STRATEGY-PAIR COMBINATIONS")
print("-" * 100)

display(
    section6b_action_pair_detail_df.head(20)
)


# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------

assert len(action_drift_df) == 36, (
    f"Expected 36 action-drift rows "
    f"(6 actions × 6 strategy pairs), "
    f"found {len(action_drift_df)}."
)

assert section6b_action_sensitivity_df[
    action_col
].nunique() == 6, (
    "Expected exactly six unique policy actions."
)

print("\nACTION RANKING")
print("-" * 100)

for _, row in section6b_action_sensitivity_df.iterrows():

    print(
        f"#{int(row['action_sensitivity_rank'])} "
        f"{row[action_col]}: "
        f"mean drift={row['mean_probability_drift']:.4f}, "
        f"max drift={row['maximum_probability_drift']:.4f}"
    )

print(
    "\n✅ TRUE ACTION-LEVEL PROBABILITY "
    "SENSITIVITY ANALYSIS COMPLETE"
)


# In[23]:


# ==========================================================================================
# SECTION 6C — SIGNED ACTION PROBABILITY REDISTRIBUTION
# ==========================================================================================

print("=" * 100)
print("SECTION 6C — SIGNED ACTION PROBABILITY REDISTRIBUTION")
print("=" * 100)

assert "notebook56_signed_action_shift_df" in globals(), (
    "notebook56_signed_action_shift_df is missing. "
    "Rerun Section 1B."
)

signed_shift_df = notebook56_signed_action_shift_df.copy()

print("\nAVAILABLE SIGNED-SHIFT COLUMNS")
print("-" * 100)
print(list(signed_shift_df.columns))

display(signed_shift_df.head(12))

# ------------------------------------------------------------------------------------------
# Use exact Notebook 56 schema
# ------------------------------------------------------------------------------------------

required_columns = [
    "strategy_a",
    "strategy_b",
    "action",
    "mean_signed_shift_b_minus_a",
    "median_signed_shift_b_minus_a",
    "minimum_signed_shift",
    "maximum_signed_shift",
    "cases_probability_increased",
    "cases_probability_decreased",
    "cases_probability_unchanged",
]

missing_columns = [
    column
    for column in required_columns
    if column not in signed_shift_df.columns
]

assert not missing_columns, (
    "Notebook 56 signed-action table is missing required columns: "
    f"{missing_columns}"
)

action_col = "action"
mean_signed_col = "mean_signed_shift_b_minus_a"
median_signed_col = "median_signed_shift_b_minus_a"
positive_count_col = "cases_probability_increased"
negative_count_col = "cases_probability_decreased"
unchanged_count_col = "cases_probability_unchanged"

# ------------------------------------------------------------------------------------------
# Global signed redistribution by action
# ------------------------------------------------------------------------------------------

section6c_signed_action_summary_df = (
    signed_shift_df
    .groupby(
        action_col,
        as_index=False,
    )
    .agg(
        strategy_pair_rows=(
            action_col,
            "count",
        ),
        mean_signed_probability_shift=(
            mean_signed_col,
            "mean",
        ),
        median_signed_probability_shift=(
            median_signed_col,
            "median",
        ),
        minimum_signed_probability_shift=(
            "minimum_signed_shift",
            "min",
        ),
        maximum_signed_probability_shift=(
            "maximum_signed_shift",
            "max",
        ),
        total_positive_cases=(
            positive_count_col,
            "sum",
        ),
        total_negative_cases=(
            negative_count_col,
            "sum",
        ),
        total_unchanged_cases=(
            unchanged_count_col,
            "sum",
        ),
    )
)

section6c_signed_action_summary_df[
    "net_case_direction"
] = (
    section6c_signed_action_summary_df[
        "total_positive_cases"
    ]
    - section6c_signed_action_summary_df[
        "total_negative_cases"
    ]
)

section6c_signed_action_summary_df[
    "absolute_mean_signed_shift"
] = (
    section6c_signed_action_summary_df[
        "mean_signed_probability_shift"
    ].abs()
)

section6c_signed_action_summary_df[
    "dominant_direction"
] = np.where(
    section6c_signed_action_summary_df[
        "mean_signed_probability_shift"
    ].gt(0.0),
    "GAINED_PROBABILITY",
    np.where(
        section6c_signed_action_summary_df[
            "mean_signed_probability_shift"
        ].lt(0.0),
        "LOST_PROBABILITY",
        "NEUTRAL",
    ),
)

section6c_signed_action_summary_df = (
    section6c_signed_action_summary_df
    .sort_values(
        "absolute_mean_signed_shift",
        ascending=False,
    )
    .reset_index(drop=True)
)

section6c_signed_action_summary_df[
    "redistribution_rank"
] = np.arange(
    1,
    len(section6c_signed_action_summary_df) + 1,
)

print("\nGLOBAL SIGNED PROBABILITY REDISTRIBUTION BY ACTION")
print("-" * 100)

display(
    section6c_signed_action_summary_df[
        [
            "redistribution_rank",
            "action",
            "strategy_pair_rows",
            "mean_signed_probability_shift",
            "median_signed_probability_shift",
            "minimum_signed_probability_shift",
            "maximum_signed_probability_shift",
            "total_positive_cases",
            "total_negative_cases",
            "total_unchanged_cases",
            "net_case_direction",
            "dominant_direction",
        ]
    ]
)

# ------------------------------------------------------------------------------------------
# Detailed pairwise signed redistribution
# ------------------------------------------------------------------------------------------

section6c_pair_detail_df = (
    signed_shift_df[
        [
            "strategy_a",
            "strategy_b",
            "action",
            "mean_signed_shift_b_minus_a",
            "median_signed_shift_b_minus_a",
            "minimum_signed_shift",
            "maximum_signed_shift",
            "cases_probability_increased",
            "cases_probability_decreased",
            "cases_probability_unchanged",
            "cases_increased_by_005",
            "cases_decreased_by_005",
            "cases_increased_by_010",
            "cases_decreased_by_010",
        ]
    ]
    .copy()
)

section6c_pair_detail_df[
    "absolute_mean_signed_shift"
] = (
    section6c_pair_detail_df[
        "mean_signed_shift_b_minus_a"
    ].abs()
)

section6c_pair_detail_df = (
    section6c_pair_detail_df
    .sort_values(
        "absolute_mean_signed_shift",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nLARGEST SIGNED ACTION SHIFTS BY STRATEGY PAIR")
print("-" * 100)

display(
    section6c_pair_detail_df.head(20)
)

# ------------------------------------------------------------------------------------------
# Probability mass conservation
# ------------------------------------------------------------------------------------------

section6c_mass_conservation_df = (
    signed_shift_df
    .groupby(
        [
            "strategy_a",
            "strategy_b",
        ],
        as_index=False,
    )
    .agg(
        summed_mean_signed_shift=(
            "mean_signed_shift_b_minus_a",
            "sum",
        ),
    )
)

section6c_mass_conservation_df[
    "absolute_residual"
] = (
    section6c_mass_conservation_df[
        "summed_mean_signed_shift"
    ].abs()
)

print("\nSIGNED-SHIFT PROBABILITY MASS CONSERVATION")
print("-" * 100)

display(section6c_mass_conservation_df)

# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------

assert len(signed_shift_df) == 36, (
    f"Expected 36 signed action-shift rows "
    f"(6 actions × 6 strategy pairs), "
    f"found {len(signed_shift_df)}."
)

assert section6c_signed_action_summary_df[
    "action"
].nunique() == 6, (
    "Expected exactly six unique actions."
)

assert np.allclose(
    section6c_mass_conservation_df[
        "summed_mean_signed_shift"
    ].to_numpy(dtype=float),
    0.0,
    atol=1e-8,
), (
    "Signed action shifts do not conserve probability mass "
    "for one or more strategy comparisons."
)

print("\nREDISTRIBUTION INTERPRETATION")
print("-" * 100)

for _, row in section6c_signed_action_summary_df.iterrows():

    if row["mean_signed_probability_shift"] > 0:
        direction_text = "gained probability on average"
    elif row["mean_signed_probability_shift"] < 0:
        direction_text = "lost probability on average"
    else:
        direction_text = "showed no net average change"

    print(
        f"{row['action']}: "
        f"{direction_text} "
        f"({row['mean_signed_probability_shift']:+.4f})"
    )

print(
    "\n✅ SIGNED ACTION PROBABILITY "
    "REDISTRIBUTION ANALYSIS COMPLETE"
)


# In[24]:


# ==========================================================================================
# SECTION 7 — EXECUTIVE EXPLAINABILITY FINDINGS AND POLICY RECOMMENDATION
# ==========================================================================================

print("=" * 100)
print("SECTION 7 — EXECUTIVE EXPLAINABILITY FINDINGS")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Collect key findings from completed Notebook 57 analyses
# ------------------------------------------------------------------------------------------

top_global_feature = (
    section3b_importance_pivot_df
    .sort_values(
        "mean_importance_when_retained",
        ascending=False,
    )
    .iloc[0]
)

top_r3_feature = (
    section3a_feature_importance_df
    .loc[
        section3a_feature_importance_df[
            "strategy"
        ].eq("R3_STATE_CENTRIC_POLICY")
    ]
    .sort_values(
        "importance",
        ascending=False,
    )
    .iloc[0]
)

largest_removal_stage = (
    section4a_transition_summary_df
    .sort_values(
        "total_importance_removed",
        ascending=False,
    )
    .iloc[0]
)

most_sensitive_action = (
    section6b_action_sensitivity_df
    .sort_values(
        "mean_probability_drift",
        ascending=False,
    )
    .iloc[0]
)

largest_single_action_drift = (
    section6b_action_sensitivity_df
    .sort_values(
        "maximum_probability_drift",
        ascending=False,
    )
    .iloc[0]
)

largest_probability_gain = (
    section6c_signed_action_summary_df
    .sort_values(
        "mean_signed_probability_shift",
        ascending=False,
    )
    .iloc[0]
)

largest_probability_loss = (
    section6c_signed_action_summary_df
    .sort_values(
        "mean_signed_probability_shift",
        ascending=True,
    )
    .iloc[0]
)

highest_drift_variant = (
    section5b_variant_drift_summary_df
    .sort_values(
        "mean_maximum_probability_difference",
        ascending=False,
    )
    .iloc[0]
)

# ------------------------------------------------------------------------------------------
# Build executive findings
# ------------------------------------------------------------------------------------------

executive_findings = [
    {
        "finding_id": "F01",
        "finding": "Predicted action stability",
        "result": (
            "All four frozen policies preserved identical predicted actions "
            "across all 184 controlled evaluation cases."
        ),
    },
    {
        "finding_id": "F02",
        "finding": "Probability sensitivity",
        "result": (
            f"The largest observed case-level probability drift was "
            f"{section5a_overall_summary_df.iloc[0]['maximum_probability_difference']:.3f}."
        ),
    },
    {
        "finding_id": "F03",
        "finding": "Most important shared feature",
        "result": (
            f"{top_global_feature['feature_name']} had the highest mean importance "
            f"when retained across the strategy set "
            f"({top_global_feature['mean_importance_when_retained']:.4f})."
        ),
    },
    {
        "finding_id": "F04",
        "finding": "Dominant R3 feature",
        "result": (
            f"{top_r3_feature['feature_name']} was the strongest feature in "
            f"R3_STATE_CENTRIC_POLICY with importance "
            f"{top_r3_feature['importance']:.4f}."
        ),
    },
    {
        "finding_id": "F05",
        "finding": "Largest ablation stage",
        "result": (
            f"{largest_removal_stage['transition']} removed "
            f"{largest_removal_stage['total_importance_removed_percent']:.2f}% "
            f"of the prior model importance."
        ),
    },
    {
        "finding_id": "F06",
        "finding": "Most sensitive action on average",
        "result": (
            f"{most_sensitive_action['action']} had the largest mean probability drift "
            f"({most_sensitive_action['mean_probability_drift']:.4f})."
        ),
    },
    {
        "finding_id": "F07",
        "finding": "Largest single-action drift",
        "result": (
            f"{largest_single_action_drift['action']} produced the largest observed "
            f"single-action probability drift "
            f"({largest_single_action_drift['maximum_probability_drift']:.3f})."
        ),
    },
    {
        "finding_id": "F08",
        "finding": "Primary probability gain",
        "result": (
            f"{largest_probability_gain['action']} gained the most probability "
            f"on average "
            f"({largest_probability_gain['mean_signed_probability_shift']:+.4f})."
        ),
    },
    {
        "finding_id": "F09",
        "finding": "Primary probability loss",
        "result": (
            f"{largest_probability_loss['action']} lost the most probability "
            f"on average "
            f"({largest_probability_loss['mean_signed_probability_shift']:+.4f})."
        ),
    },
    {
        "finding_id": "F10",
        "finding": "Most drift-sensitive variant",
        "result": (
            f"{highest_drift_variant['case_variant']} had the largest mean "
            f"case-level drift "
            f"({highest_drift_variant['mean_maximum_probability_difference']:.4f})."
        ),
    },
    {
        "finding_id": "F11",
        "finding": "Probability mass conservation",
        "result": (
            "Signed action shifts sum to approximately zero for every strategy pair, "
            "confirming numerically stable probability redistribution."
        ),
    },
    {
        "finding_id": "F12",
        "finding": "Interpretability conclusion",
        "result": (
            "Removing explicit legality and compatibility signals changes policy "
            "confidence substantially while preserving the decision boundary."
        ),
    },
]

section7_executive_findings_df = pd.DataFrame(
    executive_findings
)

print("\nEXECUTIVE FINDINGS")
print("-" * 100)

display(section7_executive_findings_df)

# ------------------------------------------------------------------------------------------
# Tournament-policy recommendation
# ------------------------------------------------------------------------------------------

section7_policy_recommendation_df = pd.DataFrame(
    [
        {
            "recommendation": "Primary candidate",
            "strategy": "R3_STATE_CENTRIC_POLICY",
            "rationale": (
                "R3 preserves action decisions across the controlled benchmark "
                "while relying more strongly on intrinsic battle-state features "
                "rather than handcrafted legality/signature signals."
            ),
        },
        {
            "recommendation": "Validation requirement",
            "strategy": "R3_STATE_CENTRIC_POLICY",
            "rationale": (
                "R3 should proceed to robustness, calibration, SHAP, and large-scale "
                "tournament evaluation before final submission selection."
            ),
        },
        {
            "recommendation": "Fallback reference",
            "strategy": "R0_CURRENT_BASELINE",
            "rationale": (
                "R0 remains the reference policy for measuring the effect of "
                "progressive feature removal and confidence redistribution."
            ),
        },
    ]
)

print("\nPOLICY RECOMMENDATION")
print("-" * 100)

display(section7_policy_recommendation_df)

# ------------------------------------------------------------------------------------------
# Final validation
# ------------------------------------------------------------------------------------------

assert len(section7_executive_findings_df) == 12
assert (
    section7_executive_findings_df[
        "result"
    ].notna().all()
)

print(
    "\n✅ EXECUTIVE EXPLAINABILITY FINDINGS "
    "AND POLICY RECOMMENDATION COMPLETE"
)


# In[26]:


# ==========================================================================================
# SECTION 8 — EXPORT NOTEBOOK 57 ARTIFACTS AND FINAL VALIDATION
# ==========================================================================================

print("=" * 100)
print("SECTION 8 — EXPORTING NOTEBOOK 57 ARTIFACTS")
print("=" * 100)

from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np

# ==========================================================================================
# 8A — RESOLVE OUTPUT DIRECTORY
# ==========================================================================================

# Use the Notebook 57 artifact directory created in Section 1A when available.
if "ARTIFACTS_DIRECTORY" in globals():

    output_dir = ARTIFACTS_DIRECTORY

else:

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "notebook57"
    )

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print(f"\nArtifact directory:\n{output_dir}")

# ==========================================================================================
# 8B — VERIFY REQUIRED NOTEBOOK 57 OBJECTS
# ==========================================================================================

required_objects = {
    # Section 3
    "section3a_feature_importance_df":
        section3a_feature_importance_df,

    "section3b_importance_pivot_df":
        section3b_importance_pivot_df,

    "section3c_concentration_df":
        section3c_concentration_df,

    "section3c_retained_redistribution_df":
        section3c_retained_redistribution_df,

    "section3c_feature_type_summary_df":
        section3c_feature_type_summary_df,

    # Section 4
    "section4a_removed_feature_attribution_df":
        section4a_removed_feature_attribution_df,

    "section4a_transition_summary_df":
        section4a_transition_summary_df,

    "section4b_importance_drift_link_df":
        section4b_importance_drift_link_df,

    "section4b_descriptive_correlations_df":
        section4b_descriptive_correlations_df,

    # Section 5
    "section5a_high_drift_cases_df":
        section5a_high_drift_cases_df,

    "section5a_overall_summary_df":
        section5a_overall_summary_df,

    "section5a_pair_summary_df":
        section5a_pair_summary_df,

    "section5a_threshold_summary_df":
        section5a_threshold_summary_df,

    "section5b_case_profile_df":
        section5b_case_profile_df,

    "section5b_side_drift_summary_df":
        section5b_side_drift_summary_df,

    "section5b_variant_drift_summary_df":
        section5b_variant_drift_summary_df,

    "section5b_variant_side_summary_df":
        section5b_variant_side_summary_df,

    "section5b_unique_case_summary_df":
        section5b_unique_case_summary_df,

    "section5b_unique_case_threshold_df":
        section5b_unique_case_threshold_df,

    # Section 6
    "section6a_action_summary_df":
        section6a_action_summary_df,

    "section6b_action_sensitivity_df":
        section6b_action_sensitivity_df,

    "section6b_action_pair_detail_df":
        section6b_action_pair_detail_df,

    "section6c_signed_action_summary_df":
        section6c_signed_action_summary_df,

    "section6c_pair_detail_df":
        section6c_pair_detail_df,

    "section6c_mass_conservation_df":
        section6c_mass_conservation_df,

    # Section 7
    "section7_executive_findings_df":
        section7_executive_findings_df,

    "section7_policy_recommendation_df":
        section7_policy_recommendation_df,
}

object_validation_records = []

for object_name, obj in required_objects.items():

    object_validation_records.append(
        {
            "object_name": object_name,
            "python_type": type(obj).__name__,
            "rows": (
                len(obj)
                if hasattr(obj, "__len__")
                else None
            ),
            "available": obj is not None,
        }
    )

section8_object_validation_df = pd.DataFrame(
    object_validation_records
)

print("\nNOTEBOOK 57 OBJECT VALIDATION")
print("-" * 100)

display(section8_object_validation_df)

assert section8_object_validation_df[
    "available"
].all(), (
    "One or more Notebook 57 export objects are unavailable."
)

# ==========================================================================================
# 8C — EXPORT CSV ANALYSIS TABLES
# ==========================================================================================

csv_exports = {
    # Section 3
    "section3a_feature_importance.csv":
        section3a_feature_importance_df,

    "section3b_cross_strategy_importance.csv":
        section3b_importance_pivot_df,

    "section3c_importance_concentration.csv":
        section3c_concentration_df,

    "section3c_importance_redistribution.csv":
        section3c_retained_redistribution_df,

    "section3c_feature_type_importance.csv":
        section3c_feature_type_summary_df,

    # Section 4
    "section4a_removed_feature_attribution.csv":
        section4a_removed_feature_attribution_df,

    "section4a_ablation_transition_summary.csv":
        section4a_transition_summary_df,

    "section4b_removed_importance_probability_drift.csv":
        section4b_importance_drift_link_df,

    "section4b_descriptive_correlations.csv":
        section4b_descriptive_correlations_df,

    # Section 5
    "section5a_case_probability_drift.csv":
        section5a_high_drift_cases_df,

    "section5a_case_drift_overall_summary.csv":
        section5a_overall_summary_df,

    "section5a_case_drift_pair_summary.csv":
        section5a_pair_summary_df,

    "section5a_case_drift_threshold_summary.csv":
        section5a_threshold_summary_df,

    "section5b_case_profiles.csv":
        section5b_case_profile_df,

    "section5b_side_drift_summary.csv":
        section5b_side_drift_summary_df,

    "section5b_variant_drift_summary.csv":
        section5b_variant_drift_summary_df,

    "section5b_variant_side_summary.csv":
        section5b_variant_side_summary_df,

    "section5b_unique_high_drift_cases.csv":
        section5b_unique_case_summary_df,

    "section5b_unique_case_thresholds.csv":
        section5b_unique_case_threshold_df,

    # Section 6
    "section6a_selected_action_drift.csv":
        section6a_action_summary_df,

    "section6b_action_sensitivity.csv":
        section6b_action_sensitivity_df,

    "section6b_action_pair_detail.csv":
        section6b_action_pair_detail_df,

    "section6c_signed_action_redistribution.csv":
        section6c_signed_action_summary_df,

    "section6c_signed_action_pair_detail.csv":
        section6c_pair_detail_df,

    "section6c_probability_mass_conservation.csv":
        section6c_mass_conservation_df,

    # Section 7
    "section7_executive_findings.csv":
        section7_executive_findings_df,

    "section7_policy_recommendation.csv":
        section7_policy_recommendation_df,
}

export_records = []

for filename, dataframe in csv_exports.items():

    export_path = output_dir / filename

    dataframe.to_csv(
        export_path,
        index=False,
    )

    export_records.append(
        {
            "artifact": filename,
            "artifact_type": "CSV",
            "rows": len(dataframe),
            "size_bytes": export_path.stat().st_size,
            "exists": export_path.exists(),
            "path": str(export_path),
        }
    )

# ==========================================================================================
# 8D — CREATE NOTEBOOK 57 MACHINE-READABLE RESULT PACKAGE
# ==========================================================================================

notebook57_metadata = {
    "notebook": 57,
    "title": "Probability Attribution and Feature Importance",
    "evaluation_cases": 184,
    "strategy_count": 4,
    "strategies": [
        "R0_CURRENT_BASELINE",
        "R1_REMOVE_LEGAL_SIGNATURE",
        "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
        "R3_STATE_CENTRIC_POLICY",
    ],
    "primary_candidate": "R3_STATE_CENTRIC_POLICY",
    "largest_case_probability_drift": 0.168,
    "most_sensitive_action_mean": "Quick Attack",
    "largest_single_action_drift": "Ascension",
    "primary_probability_gain": "Ascension",
    "primary_probability_loss": "Quick Attack",
    "most_drift_sensitive_variant": "V03",
    "status": "PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE",
}

notebook57_results = {
    "metadata":
        notebook57_metadata,

    "feature_importance":
        section3a_feature_importance_df,

    "cross_strategy_importance":
        section3b_importance_pivot_df,

    "importance_concentration":
        section3c_concentration_df,

    "importance_redistribution":
        section3c_retained_redistribution_df,

    "feature_type_summary":
        section3c_feature_type_summary_df,

    "removed_feature_attribution":
        section4a_removed_feature_attribution_df,

    "ablation_transition_summary":
        section4a_transition_summary_df,

    "importance_probability_drift_link":
        section4b_importance_drift_link_df,

    "high_drift_cases":
        section5a_high_drift_cases_df,

    "case_profile":
        section5b_case_profile_df,

    "unique_case_summary":
        section5b_unique_case_summary_df,

    "action_sensitivity":
        section6b_action_sensitivity_df,

    "signed_action_redistribution":
        section6c_signed_action_summary_df,

    "probability_mass_conservation":
        section6c_mass_conservation_df,

    "executive_findings":
        section7_executive_findings_df,

    "policy_recommendation":
        section7_policy_recommendation_df,
}

results_pickle_path = (
    output_dir
    / "notebook57_probability_attribution_results.pkl"
)

with open(
    results_pickle_path,
    "wb",
) as file:

    pickle.dump(
        notebook57_results,
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

export_records.append(
    {
        "artifact": results_pickle_path.name,
        "artifact_type": "PKL",
        "rows": len(notebook57_results),
        "size_bytes": results_pickle_path.stat().st_size,
        "exists": results_pickle_path.exists(),
        "path": str(results_pickle_path),
    }
)

# ==========================================================================================
# 8E — EXPORT METADATA JSON
# ==========================================================================================

metadata_json_path = (
    output_dir
    / "notebook57_summary.json"
)

with open(
    metadata_json_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook57_metadata,
        file,
        indent=2,
    )

export_records.append(
    {
        "artifact": metadata_json_path.name,
        "artifact_type": "JSON",
        "rows": 1,
        "size_bytes": metadata_json_path.stat().st_size,
        "exists": metadata_json_path.exists(),
        "path": str(metadata_json_path),
    }
)

# ==========================================================================================
# 8F — EXPORT EXECUTIVE REPORT
# ==========================================================================================

executive_report_path = (
    output_dir
    / "notebook57_executive_report.md"
)

executive_lines = [
    "# Notebook 57 — Probability Attribution and Feature Importance",
    "",
    "## Executive Findings",
    "",
]

for _, row in section7_executive_findings_df.iterrows():

    executive_lines.append(
        f"- **{row['finding']}**: {row['result']}"
    )

executive_lines.extend(
    [
        "",
        "## Policy Recommendation",
        "",
    ]
)

for _, row in section7_policy_recommendation_df.iterrows():

    executive_lines.append(
        f"- **{row['recommendation']} — "
        f"{row['strategy']}**: "
        f"{row['rationale']}"
    )

executive_lines.extend(
    [
        "",
        "## Notebook Status",
        "",
        "**PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE**",
        "",
    ]
)

executive_report_path.write_text(
    "\n".join(executive_lines),
    encoding="utf-8",
)

export_records.append(
    {
        "artifact": executive_report_path.name,
        "artifact_type": "Markdown",
        "rows": len(executive_lines),
        "size_bytes": executive_report_path.stat().st_size,
        "exists": executive_report_path.exists(),
        "path": str(executive_report_path),
    }
)

# ==========================================================================================
# 8G — EXPORT INVENTORY
# ==========================================================================================

section8_export_inventory_df = pd.DataFrame(
    export_records
)

inventory_path = (
    output_dir
    / "notebook57_artifact_inventory.csv"
)

section8_export_inventory_df.to_csv(
    inventory_path,
    index=False,
)

print("\nEXPORTED ARTIFACT INVENTORY")
print("-" * 100)

display(section8_export_inventory_df)

# ==========================================================================================
# 8H — FINAL COMPLETION CHECKLIST
# ==========================================================================================

section8_completion_checklist_df = pd.DataFrame(
    [
        {
            "requirement": "Feature importance extraction",
            "completed": True,
        },
        {
            "requirement": "Cross-strategy importance comparison",
            "completed": True,
        },
        {
            "requirement": "Importance concentration analysis",
            "completed": True,
        },
        {
            "requirement": "Feature-removal attribution",
            "completed": True,
        },
        {
            "requirement": "Probability-drift attribution",
            "completed": True,
        },
        {
            "requirement": "High-drift case profiling",
            "completed": True,
        },
        {
            "requirement": "Action sensitivity analysis",
            "completed": True,
        },
        {
            "requirement": "Signed probability redistribution",
            "completed": True,
        },
        {
            "requirement": "Probability-mass conservation",
            "completed": True,
        },
        {
            "requirement": "Executive findings",
            "completed": True,
        },
        {
            "requirement": "Policy recommendation",
            "completed": True,
        },
        {
            "requirement": "CSV artifact exports",
            "completed": all(
                item["exists"]
                for item in export_records
                if item["artifact_type"] == "CSV"
            ),
        },
        {
            "requirement": "Machine-readable result package",
            "completed": results_pickle_path.exists(),
        },
        {
            "requirement": "Executive report",
            "completed": executive_report_path.exists(),
        },
    ]
)

print("\nNOTEBOOK 57 FINAL COMPLETION CHECKLIST")
print("-" * 100)

display(section8_completion_checklist_df)

assert section8_completion_checklist_df[
    "completed"
].all(), (
    "Notebook 57 final validation failed."
)

assert results_pickle_path.exists()
assert metadata_json_path.exists()
assert executive_report_path.exists()
assert inventory_path.exists()

print("\n" + "=" * 100)
print("🏁 NOTEBOOK 57 COMPLETE")
print("=" * 100)

print(
    "\nStatus: "
    "PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE"
)

print(
    f"\nPrimary policy candidate: "
    f"{notebook57_metadata['primary_candidate']}"
)

print(
    f"\nArtifacts saved to:\n{output_dir}"
)

print(
    f"\nTotal exported artifacts: "
    f"{len(section8_export_inventory_df) + 1}"
)

print("\n✅ NOTEBOOK 57 RESULTS VALIDATED AND EXPORTED")
print("✅ READY FOR NOTEBOOK 58")


# In[28]:


# ==========================================================================================
# SECTION 8I — ALIGN NOTEBOOK 57 OUTPUTS WITH PROJECT DIRECTORY STRUCTURE
# ==========================================================================================

print("=" * 100)
print("SECTION 8I — PROJECT DIRECTORY ALIGNMENT")
print("=" * 100)

from pathlib import Path
import shutil
import pandas as pd

# ==========================================================================================
# PROJECT-ALIGNED DIRECTORIES
# ==========================================================================================

NOTEBOOK57_ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook57"
)

NOTEBOOK57_REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook57"
)

NOTEBOOK57_ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK57_REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nPROJECT-ALIGNED DIRECTORIES")
print("-" * 100)

directory_alignment_df = pd.DataFrame(
    [
        {
            "purpose": "machine_readable_artifacts",
            "directory": str(
                NOTEBOOK57_ARTIFACTS_DIR
            ),
            "exists": (
                NOTEBOOK57_ARTIFACTS_DIR.exists()
            ),
        },
        {
            "purpose": "human_readable_reports",
            "directory": str(
                NOTEBOOK57_REPORTS_DIR
            ),
            "exists": (
                NOTEBOOK57_REPORTS_DIR.exists()
            ),
        },
    ]
)

display(directory_alignment_df)

# ==========================================================================================
# COPY HUMAN-FACING REPORTS FROM ARTIFACTS INTO REPORTS
# ==========================================================================================

report_files = [
    "notebook57_executive_report.md",
    "section7_executive_findings.csv",
    "section7_policy_recommendation.csv",
]

report_alignment_records = []

for filename in report_files:

    source_path = (
        NOTEBOOK57_ARTIFACTS_DIR
        / filename
    )

    destination_path = (
        NOTEBOOK57_REPORTS_DIR
        / filename
    )

    assert source_path.exists(), (
        f"Source file not found:\n"
        f"{source_path}"
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    report_alignment_records.append(
        {
            "filename": filename,
            "artifact_source_exists": (
                source_path.exists()
            ),
            "report_destination_exists": (
                destination_path.exists()
            ),
            "destination": str(
                destination_path
            ),
        }
    )

section8i_report_alignment_df = pd.DataFrame(
    report_alignment_records
)

print("\nREPORT DIRECTORY ALIGNMENT")
print("-" * 100)

display(section8i_report_alignment_df)

assert (
    section8i_report_alignment_df[
        "report_destination_exists"
    ].all()
), (
    "One or more report files were not copied "
    "to reports/notebook57."
)

# ==========================================================================================
# VERIFY CORE NOTEBOOK 57 ARTIFACTS REMAIN IN ARTIFACTS DIRECTORY
# ==========================================================================================

required_artifact_files = [
    "notebook57_probability_attribution_results.pkl",
    "notebook57_summary.json",
    "notebook57_artifact_inventory.csv",
    "section3a_feature_importance.csv",
    "section6b_action_sensitivity.csv",
    "section6c_signed_action_redistribution.csv",
]

artifact_validation_records = []

for filename in required_artifact_files:

    artifact_path = (
        NOTEBOOK57_ARTIFACTS_DIR
        / filename
    )

    artifact_validation_records.append(
        {
            "filename": filename,
            "exists": artifact_path.exists(),
            "path": str(artifact_path),
        }
    )

section8i_artifact_validation_df = pd.DataFrame(
    artifact_validation_records
)

print("\nCORE ARTIFACT VALIDATION")
print("-" * 100)

display(section8i_artifact_validation_df)

assert (
    section8i_artifact_validation_df[
        "exists"
    ].all()
), (
    "One or more core Notebook 57 artifacts "
    "are missing."
)

# ==========================================================================================
# FINAL PROJECT-ALIGNED COMPLETION MESSAGE
# ==========================================================================================

print("\n" + "=" * 100)
print("🏁 NOTEBOOK 57 COMPLETE")
print("=" * 100)

print(
    "\nStatus: "
    "PROBABILITY_ATTRIBUTION_AND_FEATURE_IMPORTANCE_COMPLETE"
)

print(
    "\nPrimary policy candidate: "
    "R3_STATE_CENTRIC_POLICY"
)

print(
    "\nMachine-readable artifacts:"
)

print(
    NOTEBOOK57_ARTIFACTS_DIR
)

print(
    "\nHuman-readable reports:"
)

print(
    NOTEBOOK57_REPORTS_DIR
)

print(
    "\n✅ NOTEBOOK 57 ARTIFACT STRUCTURE VALIDATED"
)

print(
    "✅ NOTEBOOK 57 REPORT STRUCTURE VALIDATED"
)

print(
    "✅ PROJECT DIRECTORY STRUCTURE ALIGNED"
)

print(
    "✅ READY TO SAVE AND BACK UP NOTEBOOK 57"
)


# In[ ]:




