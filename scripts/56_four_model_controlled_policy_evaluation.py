#!/usr/bin/env python
# coding: utf-8

# # Notebook 56 — Four-Model Controlled Policy Evaluation
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook evaluates four controlled policy strategies trained in Notebook 55:
# 
# - **R0_CURRENT_BASELINE**
# - **R1_REMOVE_LEGAL_SIGNATURE**
# - **R2_REMOVE_SIGNATURE_AND_COMPATIBILITY**
# - **R3_STATE_CENTRIC_POLICY**
# 
# All strategies are evaluated using the same frozen 184-case evaluation contract.

# In[2]:


# ======================================================================================
# NOTEBOOK 56 — IMPORTS AND CONFIGURATION
# ======================================================================================

import hashlib
import json
import math
import warnings

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120)

print("✅ Notebook 56 imports loaded")


# In[3]:


# ======================================================================================
# SECTION 1A — PROJECT PATH CONFIGURATION
# ======================================================================================

print("=" * 100)
print("SECTION 1A — PROJECT PATH CONFIGURATION")
print("=" * 100)

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOKS_DIRECTORY = (
    PROJECT_ROOT
    / "notebooks"
)

NOTEBOOK55_REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook55"
)

NOTEBOOK55_MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "notebook55"
)

NOTEBOOK55_ARTIFACTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook55"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook56"
)

ARTIFACTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook56"
)

MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "notebook56"
)

for directory in [
    REPORTS_DIRECTORY,
    ARTIFACTS_DIRECTORY,
    MODELS_DIRECTORY,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

NOTEBOOK56_PATH = (
    NOTEBOOKS_DIRECTORY
    / "56_four_model_controlled_policy_evaluation.ipynb"
)

print()
print("PROJECT PATHS")
print("-" * 100)

for name, path in {
    "PROJECT_ROOT":
        PROJECT_ROOT,

    "NOTEBOOKS_DIRECTORY":
        NOTEBOOKS_DIRECTORY,

    "NOTEBOOK55_REPORTS_DIRECTORY":
        NOTEBOOK55_REPORTS_DIRECTORY,

    "NOTEBOOK55_MODELS_DIRECTORY":
        NOTEBOOK55_MODELS_DIRECTORY,

    "NOTEBOOK55_ARTIFACTS_DIRECTORY":
        NOTEBOOK55_ARTIFACTS_DIRECTORY,

    "REPORTS_DIRECTORY":
        REPORTS_DIRECTORY,

    "ARTIFACTS_DIRECTORY":
        ARTIFACTS_DIRECTORY,

    "MODELS_DIRECTORY":
        MODELS_DIRECTORY,

    "NOTEBOOK56_PATH":
        NOTEBOOK56_PATH,
}.items():

    print(
        f"{name:38}: {path}"
    )

assert PROJECT_ROOT.exists()
assert NOTEBOOK55_REPORTS_DIRECTORY.exists()
assert NOTEBOOK55_MODELS_DIRECTORY.exists()
assert NOTEBOOK56_PATH.exists()

print()
print("✅ NOTEBOOK 56 PROJECT PATH CONFIGURATION PASSED")


# In[6]:


# ======================================================================================
# SECTION 1B — LOCATE NOTEBOOK 55 FROZEN ASSETS
# ======================================================================================

print("=" * 100)
print("SECTION 1B — LOCATE NOTEBOOK 55 FROZEN ASSETS")
print("=" * 100)

NOTEBOOK55_CONTRACT_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3ba_controlled_evaluation_contract.json"
)

# Notebook 55 did not save section3ba_score_cases.csv.
# This is the complete 184-row reconstructed raw-feature evaluation corpus.
NOTEBOOK55_SCORE_CASES_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3barg_reconstructed_raw_features.csv"
)

NOTEBOOK55_ENCODED_CASES_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3barg_reconstructed_encoded_features.csv"
)

NOTEBOOK55_PROBABILITY_REFERENCE_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3barh_probability_validation.csv"
)

NOTEBOOK55_TRAINING_SUMMARY_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3a_training_summary.json"
)

NOTEBOOK55_STRATEGY_COMPARISON_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section3a_strategy_comparison.csv"
)

NOTEBOOK55_MODEL_FILES = {
    "R0_CURRENT_BASELINE":
        NOTEBOOK55_MODELS_DIRECTORY
        / "section3a_r0_current_baseline_model.joblib",

    "R1_REMOVE_LEGAL_SIGNATURE":
        NOTEBOOK55_MODELS_DIRECTORY
        / "section3a_r1_remove_legal_signature_model.joblib",

    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY":
        NOTEBOOK55_MODELS_DIRECTORY
        / "section3a_r2_remove_signature_and_compatibility_model.joblib",

    "R3_STATE_CENTRIC_POLICY":
        NOTEBOOK55_MODELS_DIRECTORY
        / "section3a_r3_state_centric_policy_model.joblib",
}

section1b_required_files = {
    "controlled_evaluation_contract":
        NOTEBOOK55_CONTRACT_FILE,

    "raw_evaluation_cases":
        NOTEBOOK55_SCORE_CASES_FILE,

    "encoded_evaluation_cases":
        NOTEBOOK55_ENCODED_CASES_FILE,

    "probability_reference":
        NOTEBOOK55_PROBABILITY_REFERENCE_FILE,

    "training_summary":
        NOTEBOOK55_TRAINING_SUMMARY_FILE,

    "strategy_comparison":
        NOTEBOOK55_STRATEGY_COMPARISON_FILE,

    **{
        f"model_{strategy_name}":
            model_path

        for strategy_name, model_path
        in NOTEBOOK55_MODEL_FILES.items()
    },
}

section1b_file_validation_rows = []

for asset_name, asset_path in section1b_required_files.items():

    section1b_file_validation_rows.append(
        {
            "asset_name":
                asset_name,

            "asset_path":
                str(asset_path),

            "exists":
                asset_path.exists(),

            "size_bytes":
                (
                    asset_path.stat().st_size
                    if asset_path.exists()
                    else 0
                ),
        }
    )

section1b_file_validation_df = pd.DataFrame(
    section1b_file_validation_rows
)

display(section1b_file_validation_df)

missing_assets = (
    section1b_file_validation_df.loc[
        ~section1b_file_validation_df[
            "exists"
        ].astype(bool),
        "asset_name",
    ]
    .tolist()
)

assert not missing_assets, (
    "Required Notebook 55 assets are missing: "
    f"{missing_assets}"
)

assert (
    section1b_file_validation_df[
        "size_bytes"
    ] > 0
).all()

print()
print("✅ NOTEBOOK 55 FROZEN ASSETS LOCATED")


# In[5]:


# ======================================================================================
# SECTION 1B REPAIR — FIND NOTEBOOK 55 EVALUATION CASE FILE
# ======================================================================================

print("=" * 100)
print("SECTION 1B REPAIR — FIND NOTEBOOK 55 EVALUATION CASE FILE")
print("=" * 100)

candidate_files = sorted(
    NOTEBOOK55_REPORTS_DIRECTORY.glob("*.csv")
)

candidate_rows = []

for file_path in candidate_files:

    try:
        sample_df = pd.read_csv(file_path)

        candidate_rows.append(
            {
                "file_name":
                    file_path.name,

                "file_path":
                    str(file_path),

                "rows":
                    len(sample_df),

                "columns":
                    len(sample_df.columns),

                "has_comparison_case_id":
                    "comparison_case_id"
                    in sample_df.columns,

                "has_184_rows":
                    len(sample_df) == 184,

                "column_preview":
                    list(sample_df.columns[:12]),
            }
        )

    except Exception as error:

        candidate_rows.append(
            {
                "file_name":
                    file_path.name,

                "file_path":
                    str(file_path),

                "rows":
                    None,

                "columns":
                    None,

                "has_comparison_case_id":
                    False,

                "has_184_rows":
                    False,

                "column_preview":
                    f"READ ERROR: {error}",
            }
        )


section1b_candidate_files_df = pd.DataFrame(
    candidate_rows
)


likely_score_case_files_df = (
    section1b_candidate_files_df.loc[
        section1b_candidate_files_df[
            "has_comparison_case_id"
        ].astype(bool)
        |
        section1b_candidate_files_df[
            "has_184_rows"
        ].astype(bool)
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print("LIKELY NOTEBOOK 55 EVALUATION-CASE FILES")
print("-" * 100)

display(
    likely_score_case_files_df
)


print()
print("ALL NOTEBOOK 55 CSV FILES")
print("-" * 100)

display(
    section1b_candidate_files_df[
        [
            "file_name",
            "rows",
            "columns",
            "has_comparison_case_id",
            "has_184_rows",
        ]
    ]
)


# In[7]:


# ======================================================================================
# SECTION 2A — LOAD NOTEBOOK 55 CONTRACT AND TRAINING METADATA
# ======================================================================================

print("=" * 100)
print("SECTION 2A — LOAD NOTEBOOK 55 CONTRACT AND TRAINING METADATA")
print("=" * 100)

with open(
    NOTEBOOK55_CONTRACT_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook55_contract = json.load(file)


with open(
    NOTEBOOK55_TRAINING_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook55_training_summary = json.load(file)


notebook55_strategy_comparison_df = pd.read_csv(
    NOTEBOOK55_STRATEGY_COMPARISON_FILE
)


print()
print("NOTEBOOK 55 CONTROLLED EVALUATION CONTRACT")
print("-" * 100)

for key, value in notebook55_contract.items():

    print(
        f"{key:72}: {value}"
    )


print()
print("NOTEBOOK 55 TRAINING SUMMARY")
print("-" * 100)

for key, value in notebook55_training_summary.items():

    print(
        f"{key:72}: {value}"
    )


print()
print("NOTEBOOK 55 STRATEGY COMPARISON")
print("-" * 100)

display(
    notebook55_strategy_comparison_df
)


assert notebook55_contract[
    "balanced_evaluation_cases"
] == 184

assert notebook55_contract[
    "models_to_compare"
] == 4

assert notebook55_contract[
    "canonical_legal_move_signature"
] == "ascension | quick attack"

print()
print("✅ NOTEBOOK 55 CONTRACT AND TRAINING METADATA LOADED")


# In[8]:


# ======================================================================================
# SECTION 2B — LOAD FOUR FROZEN POLICY MODEL ARTIFACTS
# ======================================================================================

print("=" * 100)
print("SECTION 2B — LOAD FOUR FROZEN POLICY MODEL ARTIFACTS")
print("=" * 100)


def resolve_model_from_artifact(
    artifact,
):
    """
    Resolve a fitted estimator from either:

    1. A direct sklearn estimator, or
    2. A dictionary-style saved artifact.
    """

    if hasattr(
        artifact,
        "predict",
    ):

        return artifact


    if isinstance(
        artifact,
        dict,
    ):

        preferred_model_keys = [
            "model",
            "estimator",
            "policy_model",
            "classifier",
            "trained_model",
        ]


        for key in preferred_model_keys:

            candidate = artifact.get(
                key
            )

            if hasattr(
                candidate,
                "predict",
            ):

                return candidate


        for candidate in artifact.values():

            if hasattr(
                candidate,
                "predict",
            ):

                return candidate


    raise TypeError(
        "No fitted estimator was found in the saved model artifact."
    )


notebook56_loaded_artifacts = {}

notebook56_policy_models = {}

section2b_model_inventory_rows = []


for strategy_name, model_path in (
    NOTEBOOK55_MODEL_FILES.items()
):

    artifact = joblib.load(
        model_path
    )


    model = resolve_model_from_artifact(
        artifact
    )


    notebook56_loaded_artifacts[
        strategy_name
    ] = artifact


    notebook56_policy_models[
        strategy_name
    ] = model


    model_classes = (
        list(
            getattr(
                model,
                "classes_",
                [],
            )
        )
    )


    section2b_model_inventory_rows.append(
        {
            "strategy":
                strategy_name,

            "artifact_path":
                str(
                    model_path
                ),

            "artifact_type":
                type(
                    artifact
                ).__name__,

            "model_type":
                type(
                    model
                ).__name__,

            "is_fitted":
                hasattr(
                    model,
                    "classes_",
                ),

            "feature_count":
                getattr(
                    model,
                    "n_features_in_",
                    None,
                ),

            "class_count":
                len(
                    model_classes
                ),

            "classes":
                model_classes,

            "n_estimators":
                getattr(
                    model,
                    "n_estimators",
                    None,
                ),

            "random_state":
                getattr(
                    model,
                    "random_state",
                    None,
                ),

            "predict_available":
                hasattr(
                    model,
                    "predict",
                ),

            "predict_proba_available":
                hasattr(
                    model,
                    "predict_proba",
                ),
        }
    )


section2b_model_inventory_df = pd.DataFrame(
    section2b_model_inventory_rows
)


print()
print("FOUR-MODEL INVENTORY")
print("-" * 100)

display(
    section2b_model_inventory_df
)


assert len(
    notebook56_policy_models
) == 4


assert section2b_model_inventory_df[
    "is_fitted"
].astype(bool).all()


assert section2b_model_inventory_df[
    "predict_available"
].astype(bool).all()


assert section2b_model_inventory_df[
    "predict_proba_available"
].astype(bool).all()


print()
print("✅ FOUR FROZEN POLICY MODELS LOADED")


# In[9]:


# ======================================================================================
# SECTION 2C — LOAD STRATEGY-SPECIFIC FEATURE PARTITIONS
# ======================================================================================

print("=" * 100)
print("SECTION 2C — LOAD STRATEGY-SPECIFIC FEATURE PARTITIONS")
print("=" * 100)

NOTEBOOK55_STRATEGY_FEATURE_PARTITION_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section2c_strategy_feature_partition.csv"
)

NOTEBOOK55_ENCODED_FEATURE_PARTITION_FILE = (
    NOTEBOOK55_REPORTS_DIRECTORY
    / "section2c_encoded_feature_partition.csv"
)

NOTEBOOK55_FEATURE_INDEX_ARTIFACT_FILE = (
    NOTEBOOK55_ARTIFACTS_DIRECTORY
    / "section2c_strategy_feature_indices_and_names.json"
)


for required_file in [
    NOTEBOOK55_STRATEGY_FEATURE_PARTITION_FILE,
    NOTEBOOK55_ENCODED_FEATURE_PARTITION_FILE,
    NOTEBOOK55_FEATURE_INDEX_ARTIFACT_FILE,
]:

    assert required_file.exists(), (
        f"Required feature-partition file is missing: {required_file}"
    )


notebook55_strategy_feature_partition_df = pd.read_csv(
    NOTEBOOK55_STRATEGY_FEATURE_PARTITION_FILE
)


notebook55_encoded_feature_partition_df = pd.read_csv(
    NOTEBOOK55_ENCODED_FEATURE_PARTITION_FILE
)


with open(
    NOTEBOOK55_FEATURE_INDEX_ARTIFACT_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook55_strategy_feature_artifact = json.load(
        file
    )


print()
print("STRATEGY FEATURE PARTITION")
print("-" * 100)

display(
    notebook55_strategy_feature_partition_df
)


print()
print("ENCODED FEATURE PARTITION SAMPLE")
print("-" * 100)

display(
    notebook55_encoded_feature_partition_df.head(20)
)


print()
print("FEATURE-INDEX ARTIFACT TOP-LEVEL KEYS")
print("-" * 100)

print(
    list(
        notebook55_strategy_feature_artifact.keys()
    )
)


assert len(
    notebook55_strategy_feature_partition_df
) == 4


print()
print("✅ STRATEGY-SPECIFIC FEATURE PARTITIONS LOADED")


# In[10]:


# ======================================================================================
# SECTION 2D — INSPECT FEATURE-INDEX ARTIFACT STRUCTURE
# ======================================================================================

print("=" * 100)
print("SECTION 2D — INSPECT FEATURE-INDEX ARTIFACT STRUCTURE")
print("=" * 100)


def summarize_json_object(
    value,
    depth=0,
    max_depth=3,
):
    """
    Print a compact structural summary of a nested JSON object.
    """

    indent = "    " * depth


    if depth > max_depth:

        print(
            f"{indent}<maximum display depth reached>"
        )

        return


    if isinstance(
        value,
        dict,
    ):

        print(
            f"{indent}dict with {len(value)} keys"
        )


        for key, child_value in value.items():

            print(
                f"{indent}- {key}: "
                f"{type(child_value).__name__}"
            )


            if isinstance(
                child_value,
                (
                    dict,
                    list,
                ),
            ):

                summarize_json_object(
                    child_value,
                    depth=depth + 1,
                    max_depth=max_depth,
                )


    elif isinstance(
        value,
        list,
    ):

        print(
            f"{indent}list with {len(value)} items"
        )


        if value:

            preview = value[:5]

            print(
                f"{indent}preview: {preview}"
            )


    else:

        print(
            f"{indent}{repr(value)}"
        )


summarize_json_object(
    notebook55_strategy_feature_artifact
)


print()
print("MODEL FEATURE COUNTS")
print("-" * 100)

display(
    section2b_model_inventory_df[
        [
            "strategy",
            "feature_count",
            "class_count",
            "classes",
        ]
    ]
)


print()
print("✅ FEATURE-INDEX ARTIFACT STRUCTURE INSPECTED")


# In[14]:


# ======================================================================================
# SECTION 2E — LOAD FROZEN NOTEBOOK 55 EVALUATION DATAFRAMES
# ======================================================================================

print("=" * 100)
print("SECTION 2E — LOAD FROZEN NOTEBOOK 55 EVALUATION DATAFRAMES")
print("=" * 100)

notebook55_raw_evaluation_df = pd.read_csv(
    NOTEBOOK55_SCORE_CASES_FILE
)

notebook55_encoded_evaluation_df = pd.read_csv(
    NOTEBOOK55_ENCODED_CASES_FILE
)

notebook55_probability_reference_df = pd.read_csv(
    NOTEBOOK55_PROBABILITY_REFERENCE_FILE
)

print()
print("LOADED DATAFRAME SHAPES")
print("-" * 100)

print(
    f"{'Raw evaluation dataframe':42}: "
    f"{notebook55_raw_evaluation_df.shape}"
)

print(
    f"{'Encoded evaluation dataframe':42}: "
    f"{notebook55_encoded_evaluation_df.shape}"
)

print(
    f"{'Probability reference dataframe':42}: "
    f"{notebook55_probability_reference_df.shape}"
)

assert notebook55_raw_evaluation_df.shape[0] == 184
assert notebook55_encoded_evaluation_df.shape[0] == 184
assert notebook55_probability_reference_df.shape[0] == 184

for dataframe_name, dataframe in {
    "raw":
        notebook55_raw_evaluation_df,

    "encoded":
        notebook55_encoded_evaluation_df,

    "probability_reference":
        notebook55_probability_reference_df,
}.items():

    assert "comparison_case_id" in dataframe.columns, (
        f"{dataframe_name} dataframe is missing comparison_case_id"
    )

    assert dataframe[
        "comparison_case_id"
    ].nunique() == 184


raw_case_ids = set(
    notebook55_raw_evaluation_df[
        "comparison_case_id"
    ].astype(str)
)

encoded_case_ids = set(
    notebook55_encoded_evaluation_df[
        "comparison_case_id"
    ].astype(str)
)

reference_case_ids = set(
    notebook55_probability_reference_df[
        "comparison_case_id"
    ].astype(str)
)

assert raw_case_ids == encoded_case_ids
assert raw_case_ids == reference_case_ids

print()
print("✅ NOTEBOOK 55 EVALUATION DATAFRAMES LOADED")


# In[18]:


# ======================================================================================
# SECTION 3A — BUILD EXACT NUMERIC FEATURE MATRICES BY FROZEN FEATURE NAME
# ======================================================================================

print("=" * 100)
print("SECTION 3A — BUILD EXACT NUMERIC FEATURE MATRICES")
print("=" * 100)

strategy_feature_indices = (
    notebook55_strategy_feature_artifact[
        "strategy_feature_indices"
    ]
)

strategy_feature_names = (
    notebook55_strategy_feature_artifact[
        "strategy_feature_names"
    ]
)

notebook56_feature_matrices = {}
section3a_summary_rows = []

for strategy_name, model in notebook56_policy_models.items():

    frozen_feature_names = strategy_feature_names[
        strategy_name
    ]

    missing_features = [
        feature_name
        for feature_name in frozen_feature_names
        if feature_name not in notebook55_encoded_evaluation_df.columns
    ]

    assert not missing_features, (
        f"{strategy_name} is missing frozen encoded features: "
        f"{missing_features}"
    )

    # Select only the exact encoded feature names used to train this model.
    strategy_feature_df = (
        notebook55_encoded_evaluation_df[
            frozen_feature_names
        ]
        .copy()
    )

    # Require every selected feature to be numeric.
    nonnumeric_columns = [
        column_name
        for column_name in strategy_feature_df.columns
        if not pd.api.types.is_numeric_dtype(
            strategy_feature_df[column_name]
        )
    ]

    assert not nonnumeric_columns, (
        f"{strategy_name} contains nonnumeric model inputs: "
        f"{nonnumeric_columns}"
    )

    strategy_matrix = strategy_feature_df.to_numpy(
        dtype=float
    )

    notebook56_feature_matrices[
        strategy_name
    ] = strategy_matrix

    section3a_summary_rows.append(
        {
            "strategy":
                strategy_name,

            "rows":
                int(strategy_matrix.shape[0]),

            "matrix_columns":
                int(strategy_matrix.shape[1]),

            "frozen_feature_names":
                len(frozen_feature_names),

            "model_expected_features":
                int(model.n_features_in_),

            "numeric_matrix":
                bool(
                    np.issubdtype(
                        strategy_matrix.dtype,
                        np.number,
                    )
                ),

            "finite_values":
                bool(
                    np.isfinite(
                        strategy_matrix
                    ).all()
                ),

            "shape_match":
                strategy_matrix.shape[1]
                ==
                model.n_features_in_,
        }
    )

section3a_feature_matrix_summary_df = pd.DataFrame(
    section3a_summary_rows
)

display(
    section3a_feature_matrix_summary_df
)

assert (
    section3a_feature_matrix_summary_df[
        "rows"
    ] == 184
).all()

assert section3a_feature_matrix_summary_df[
    "numeric_matrix"
].astype(bool).all()

assert section3a_feature_matrix_summary_df[
    "finite_values"
].astype(bool).all()

assert section3a_feature_matrix_summary_df[
    "shape_match"
].astype(bool).all()

print()
print("✅ EXACT NUMERIC FEATURE MATRICES BUILT")


# In[19]:


# ======================================================================================
# SECTION 3B — VERIFY MODEL INPUT COMPATIBILITY
# ======================================================================================

print("=" * 100)
print("SECTION 3B — VERIFY MODEL INPUT COMPATIBILITY")
print("=" * 100)

section3b_validation_rows = []

for strategy_name, model in notebook56_policy_models.items():

    strategy_matrix = notebook56_feature_matrices[
        strategy_name
    ]

    section3b_validation_rows.append(
        {
            "strategy":
                strategy_name,

            "matrix_shape":
                strategy_matrix.shape,

            "matrix_dtype":
                str(strategy_matrix.dtype),

            "model_expected_features":
                int(model.n_features_in_),

            "feature_count_valid":
                strategy_matrix.shape[1]
                ==
                model.n_features_in_,

            "numeric_matrix":
                np.issubdtype(
                    strategy_matrix.dtype,
                    np.number,
                ),

            "finite_values":
                bool(
                    np.isfinite(
                        strategy_matrix
                    ).all()
                ),
        }
    )

section3b_matrix_validation_df = pd.DataFrame(
    section3b_validation_rows
)

display(
    section3b_matrix_validation_df
)

assert section3b_matrix_validation_df[
    "feature_count_valid"
].astype(bool).all()

assert section3b_matrix_validation_df[
    "numeric_matrix"
].astype(bool).all()

assert section3b_matrix_validation_df[
    "finite_values"
].astype(bool).all()

print()
print("✅ ALL FOUR MODEL INPUT MATRICES VALIDATED")


# In[21]:


# ======================================================================================
# SECTION 4A — GENERATE PREDICTIONS FOR ALL FROZEN POLICIES
# ======================================================================================

print("=" * 100)
print("SECTION 4A — GENERATE PREDICTIONS FOR ALL FROZEN POLICIES")
print("=" * 100)

notebook56_predictions = {}
prediction_summary = []

for strategy_name in notebook56_feature_matrices:

    matrix = notebook56_feature_matrices[
        strategy_name
    ]

    model = notebook56_policy_models[
        strategy_name
    ]

    predictions = model.predict(
        matrix
    )

    probabilities = model.predict_proba(
        matrix
    )

    notebook56_predictions[
        strategy_name
    ] = {
        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "classes":
            model.classes_,
    }

    prediction_summary.append(
        {
            "strategy":
                strategy_name,

            "rows":
                len(predictions),

            "classes":
                probabilities.shape[1],

            "prediction_dtype":
                str(predictions.dtype),

            "probability_shape":
                probabilities.shape,

            "minimum_probability":
                float(probabilities.min()),

            "maximum_probability":
                float(probabilities.max()),
        }
    )

prediction_summary_df = pd.DataFrame(
    prediction_summary
)

display(
    prediction_summary_df
)

assert len(
    notebook56_predictions
) == 4

assert (
    prediction_summary_df[
        "rows"
    ] == 184
).all()

assert (
    prediction_summary_df[
        "classes"
    ] == 6
).all()

print()
print("✅ ALL FOUR MODELS SCORED")


# In[23]:


#====================================================================================================
# Section 4B — Validate Prediction Outputs
#====================================================================================================

print("=" * 100)
print("SECTION 4B — VERIFY PREDICTION OUTPUTS")
print("=" * 100)

verification_rows = []

for strategy_name, result in notebook56_predictions.items():

    predictions = result["predictions"]
    probabilities = result["probabilities"]

    verification_rows.append(
        {
            "strategy": strategy_name,
            "prediction_count": len(predictions),
            "probability_rows": probabilities.shape[0],
            "probability_columns": probabilities.shape[1],
            "counts_match": len(predictions) == probabilities.shape[0],
            "finite": np.isfinite(probabilities).all(),
            "nonnegative": (probabilities >= 0).all(),
            "rows_sum_to_one": np.allclose(
                probabilities.sum(axis=1),
                np.ones(len(predictions)),
                atol=1e-8,
            ),
        }
    )

section4b_df = pd.DataFrame(verification_rows)

display(section4b_df)

assert section4b_df["counts_match"].all()
assert section4b_df["finite"].all()
assert section4b_df["nonnegative"].all()
assert section4b_df["rows_sum_to_one"].all()

print()
print("✅ ALL PREDICTION OUTPUTS VERIFIED")


# In[24]:


#====================================================================================================
# SECTION 5A — Compare Predictions Across Policies
#====================================================================================================

print("=" * 100)
print("SECTION 5A — PREDICTION AGREEMENT MATRIX")
print("=" * 100)

from itertools import combinations

agreement_rows = []

strategy_names = list(notebook56_predictions.keys())

for strategy_a, strategy_b in combinations(strategy_names, 2):

    preds_a = notebook56_predictions[strategy_a]["predictions"]
    preds_b = notebook56_predictions[strategy_b]["predictions"]

    agreement = np.mean(preds_a == preds_b)

    disagreement_count = int(np.sum(preds_a != preds_b))

    agreement_rows.append(
        {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "agreement": agreement,
            "agreement_percent": agreement * 100,
            "disagreement_cases": disagreement_count,
        }
    )

section5a_agreement_df = (
    pd.DataFrame(agreement_rows)
    .sort_values("agreement_percent", ascending=False)
    .reset_index(drop=True)
)

display(section5a_agreement_df)

print()
print("✅ POLICY AGREEMENT MATRIX BUILT")


# In[25]:


# ====================================================================================================
# SECTION 5B — Probability Difference Matrix
# ====================================================================================================

print("=" * 100)
print("SECTION 5B — PROBABILITY DIFFERENCE MATRIX")
print("=" * 100)

from itertools import combinations

probability_comparisons = []

strategy_names = list(notebook56_predictions.keys())

for strategy_a, strategy_b in combinations(strategy_names, 2):

    probs_a = notebook56_predictions[strategy_a]["probabilities"]
    probs_b = notebook56_predictions[strategy_b]["probabilities"]

    abs_difference = np.abs(probs_a - probs_b)

    probability_comparisons.append({

        "strategy_a": strategy_a,

        "strategy_b": strategy_b,

        "mean_probability_difference":
            float(abs_difference.mean()),

        "maximum_probability_difference":
            float(abs_difference.max()),

        "rms_difference":
            float(np.sqrt(np.mean((probs_a - probs_b) ** 2))),

        "changed_probability_values":
            int((abs_difference > 1e-12).sum()),

        "largest_case_difference":
            float(abs_difference.max())
    })

notebook56_probability_difference_df = pd.DataFrame(
    probability_comparisons
)

display(
    notebook56_probability_difference_df
)

print()
print("✅ PROBABILITY DIFFERENCE MATRIX BUILT")


# In[27]:


# ======================================================================================
# SECTION 5C — IDENTIFY HIGHEST PROBABILITY DRIFT CASES
# ======================================================================================

print("=" * 100)
print("SECTION 5C — IDENTIFY HIGHEST PROBABILITY DRIFT CASES")
print("=" * 100)

from itertools import combinations

comparison_case_ids = (
    notebook55_encoded_evaluation_df[
        "comparison_case_id"
    ]
    .astype(str)
    .tolist()
)

drift_records = []

strategy_names = list(
    notebook56_predictions.keys()
)

for strategy_a, strategy_b in combinations(
    strategy_names,
    2,
):

    probabilities_a = notebook56_predictions[
        strategy_a
    ]["probabilities"]

    probabilities_b = notebook56_predictions[
        strategy_b
    ]["probabilities"]

    absolute_difference = np.abs(
        probabilities_a
        -
        probabilities_b
    )

    maximum_difference_per_case = (
        absolute_difference.max(axis=1)
    )

    mean_difference_per_case = (
        absolute_difference.mean(axis=1)
    )

    l1_difference_per_case = (
        absolute_difference.sum(axis=1)
    )

    for case_index, comparison_case_id in enumerate(
        comparison_case_ids
    ):

        drift_records.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "strategy_a":
                    strategy_a,

                "strategy_b":
                    strategy_b,

                "maximum_case_difference":
                    float(
                        maximum_difference_per_case[
                            case_index
                        ]
                    ),

                "mean_case_difference":
                    float(
                        mean_difference_per_case[
                            case_index
                        ]
                    ),

                "l1_case_difference":
                    float(
                        l1_difference_per_case[
                            case_index
                        ]
                    ),

                "prediction_a":
                    str(
                        notebook56_predictions[
                            strategy_a
                        ]["predictions"][
                            case_index
                        ]
                    ),

                "prediction_b":
                    str(
                        notebook56_predictions[
                            strategy_b
                        ]["predictions"][
                            case_index
                        ]
                    ),

                "predictions_match":
                    bool(
                        notebook56_predictions[
                            strategy_a
                        ]["predictions"][
                            case_index
                        ]
                        ==
                        notebook56_predictions[
                            strategy_b
                        ]["predictions"][
                            case_index
                        ]
                    ),
            }
        )

notebook56_case_probability_drift_df = (
    pd.DataFrame(
        drift_records
    )
    .sort_values(
        [
            "maximum_case_difference",
            "mean_case_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

print()
print("TOP 20 HIGHEST-DRIFT CASES")
print("-" * 100)

display(
    notebook56_case_probability_drift_df.head(20)
)

assert len(
    notebook56_case_probability_drift_df
) == 184 * 6

assert notebook56_case_probability_drift_df[
    "predictions_match"
].astype(bool).all()

print()
print("✅ CASE-LEVEL PROBABILITY DRIFT RANKED")


# In[28]:


# ======================================================================================
# SECTION 5D — SUMMARIZE PROBABILITY DRIFT BY STRATEGY PAIR AND CASE VARIANT
# ======================================================================================

print("=" * 100)
print("SECTION 5D — PROBABILITY DRIFT SUMMARY")
print("=" * 100)

section5d_drift_df = (
    notebook56_case_probability_drift_df
    .copy()
)

# Parse identifiers such as:
# 4WE_PAIR_014_V03__PLAYER
section5d_drift_df[
    "evaluation_side"
] = (
    section5d_drift_df[
        "comparison_case_id"
    ]
    .str.extract(
        r"__(PLAYER|OPPONENT)$",
        expand=False,
    )
)

section5d_drift_df[
    "case_variant"
] = (
    section5d_drift_df[
        "comparison_case_id"
    ]
    .str.extract(
        r"_(V\d+)__",
        expand=False,
    )
)

section5d_drift_df[
    "pair_id"
] = (
    section5d_drift_df[
        "comparison_case_id"
    ]
    .str.extract(
        r"(4WE_PAIR_\d+)",
        expand=False,
    )
)

print()
print("DRIFT BY STRATEGY PAIR")
print("-" * 100)

section5d_pair_summary_df = (
    section5d_drift_df
    .groupby(
        [
            "strategy_a",
            "strategy_b",
        ],
        as_index=False,
    )
    .agg(
        cases=(
            "comparison_case_id",
            "count",
        ),

        mean_maximum_case_difference=(
            "maximum_case_difference",
            "mean",
        ),

        median_maximum_case_difference=(
            "maximum_case_difference",
            "median",
        ),

        maximum_case_difference=(
            "maximum_case_difference",
            "max",
        ),

        mean_l1_case_difference=(
            "l1_case_difference",
            "mean",
        ),

        cases_above_005=(
            "maximum_case_difference",
            lambda values: int(
                (values >= 0.05).sum()
            ),
        ),

        cases_above_010=(
            "maximum_case_difference",
            lambda values: int(
                (values >= 0.10).sum()
            ),
        ),

        cases_above_015=(
            "maximum_case_difference",
            lambda values: int(
                (values >= 0.15).sum()
            ),
        ),
    )
    .sort_values(
        "mean_maximum_case_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    section5d_pair_summary_df
)

print()
print("DRIFT BY STRATEGY PAIR AND CASE VARIANT")
print("-" * 100)

section5d_variant_summary_df = (
    section5d_drift_df
    .groupby(
        [
            "strategy_a",
            "strategy_b",
            "case_variant",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        cases=(
            "comparison_case_id",
            "count",
        ),

        mean_maximum_case_difference=(
            "maximum_case_difference",
            "mean",
        ),

        maximum_case_difference=(
            "maximum_case_difference",
            "max",
        ),

        mean_l1_case_difference=(
            "l1_case_difference",
            "mean",
        ),
    )
    .sort_values(
        [
            "mean_maximum_case_difference",
            "maximum_case_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    section5d_variant_summary_df.head(20)
)

print()
print("DRIFT BY EVALUATION SIDE")
print("-" * 100)

section5d_side_summary_df = (
    section5d_drift_df
    .groupby(
        [
            "strategy_a",
            "strategy_b",
            "evaluation_side",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        cases=(
            "comparison_case_id",
            "count",
        ),

        mean_maximum_case_difference=(
            "maximum_case_difference",
            "mean",
        ),

        maximum_case_difference=(
            "maximum_case_difference",
            "max",
        ),
    )
    .sort_values(
        "mean_maximum_case_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

display(
    section5d_side_summary_df
)

assert len(section5d_pair_summary_df) == 6
assert section5d_drift_df["case_variant"].notna().all()
assert section5d_drift_df["evaluation_side"].notna().all()

print()
print("✅ PROBABILITY DRIFT SUMMARIES BUILT")


# In[31]:


# ======================================================================================
# SECTION 6A — ACTION-LEVEL PROBABILITY DRIFT ANALYSIS
# ======================================================================================

print("=" * 100)
print("SECTION 6A — ACTION-LEVEL PROBABILITY DRIFT")
print("=" * 100)

import itertools

action_drift_records = []

strategy_names = list(
    notebook56_predictions.keys()
)

for strategy_a, strategy_b in itertools.combinations(
    strategy_names,
    2,
):

    probabilities_a = notebook56_predictions[
        strategy_a
    ]["probabilities"]

    probabilities_b = notebook56_predictions[
        strategy_b
    ]["probabilities"]

    classes_a = list(
        notebook56_predictions[
            strategy_a
        ]["classes"]
    )

    classes_b = list(
        notebook56_predictions[
            strategy_b
        ]["classes"]
    )

    assert classes_a == classes_b, (
        f"Class order differs between "
        f"{strategy_a} and {strategy_b}"
    )

    probability_difference = np.abs(
        probabilities_a
        -
        probabilities_b
    )

    for action_index, action_name in enumerate(
        classes_a
    ):

        action_values = probability_difference[
            :,
            action_index
        ]

        action_drift_records.append(
            {
                "strategy_a":
                    strategy_a,

                "strategy_b":
                    strategy_b,

                "action":
                    str(action_name),

                "mean_probability_difference":
                    float(
                        action_values.mean()
                    ),

                "median_probability_difference":
                    float(
                        np.median(
                            action_values
                        )
                    ),

                "maximum_probability_difference":
                    float(
                        action_values.max()
                    ),

                "std_probability_difference":
                    float(
                        action_values.std()
                    ),

                "cases_above_005":
                    int(
                        (
                            action_values >= 0.05
                        ).sum()
                    ),

                "cases_above_010":
                    int(
                        (
                            action_values >= 0.10
                        ).sum()
                    ),

                "cases_above_015":
                    int(
                        (
                            action_values >= 0.15
                        ).sum()
                    ),
            }
        )

section6a_action_drift_df = (
    pd.DataFrame(
        action_drift_records
    )
    .sort_values(
        [
            "mean_probability_difference",
            "maximum_probability_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

print()
print("COMPLETE ACTION-LEVEL DRIFT PROFILE")
print("-" * 100)

display(
    section6a_action_drift_df
)

print()
print("TOP 20 ACTION-LEVEL DRIFTS")
print("-" * 100)

display(
    section6a_action_drift_df[
        [
            "strategy_a",
            "strategy_b",
            "action",
            "mean_probability_difference",
            "median_probability_difference",
            "maximum_probability_difference",
            "cases_above_005",
            "cases_above_010",
            "cases_above_015",
        ]
    ].head(20)
)

expected_action_count = len(
    notebook56_predictions[
        "R0_CURRENT_BASELINE"
    ]["classes"]
)

assert len(
    section6a_action_drift_df
) == 6 * expected_action_count

assert section6a_action_drift_df[
    "mean_probability_difference"
].between(
    0.0,
    1.0,
).all()

assert section6a_action_drift_df[
    "maximum_probability_difference"
].between(
    0.0,
    1.0,
).all()

print()
print("✅ ACTION-LEVEL DRIFT PROFILE BUILT")


# In[32]:


# ======================================================================================
# SECTION 6B — SIGNED ACTION-LEVEL PROBABILITY SHIFT
# ======================================================================================

print("=" * 100)
print("SECTION 6B — SIGNED ACTION-LEVEL PROBABILITY SHIFT")
print("=" * 100)

import itertools

section6b_signed_shift_records = []

strategy_names = list(
    notebook56_predictions.keys()
)

for strategy_a, strategy_b in itertools.combinations(
    strategy_names,
    2,
):

    probabilities_a = notebook56_predictions[
        strategy_a
    ]["probabilities"]

    probabilities_b = notebook56_predictions[
        strategy_b
    ]["probabilities"]

    classes_a = [
        str(action)
        for action in notebook56_predictions[
            strategy_a
        ]["classes"]
    ]

    classes_b = [
        str(action)
        for action in notebook56_predictions[
            strategy_b
        ]["classes"]
    ]

    assert classes_a == classes_b, (
        f"Class order differs between {strategy_a} and {strategy_b}"
    )

    # Positive value means strategy_b assigns more probability than strategy_a.
    signed_difference = (
        probabilities_b
        -
        probabilities_a
    )

    for action_index, action_name in enumerate(
        classes_a
    ):

        action_shift = signed_difference[
            :,
            action_index
        ]

        section6b_signed_shift_records.append(
            {
                "strategy_a":
                    strategy_a,

                "strategy_b":
                    strategy_b,

                "action":
                    action_name,

                "mean_signed_shift_b_minus_a":
                    float(
                        action_shift.mean()
                    ),

                "median_signed_shift_b_minus_a":
                    float(
                        np.median(
                            action_shift
                        )
                    ),

                "minimum_signed_shift":
                    float(
                        action_shift.min()
                    ),

                "maximum_signed_shift":
                    float(
                        action_shift.max()
                    ),

                "cases_probability_increased":
                    int(
                        (
                            action_shift > 1e-12
                        ).sum()
                    ),

                "cases_probability_decreased":
                    int(
                        (
                            action_shift < -1e-12
                        ).sum()
                    ),

                "cases_probability_unchanged":
                    int(
                        (
                            np.abs(
                                action_shift
                            ) <= 1e-12
                        ).sum()
                    ),

                "cases_increased_by_005":
                    int(
                        (
                            action_shift >= 0.05
                        ).sum()
                    ),

                "cases_decreased_by_005":
                    int(
                        (
                            action_shift <= -0.05
                        ).sum()
                    ),

                "cases_increased_by_010":
                    int(
                        (
                            action_shift >= 0.10
                        ).sum()
                    ),

                "cases_decreased_by_010":
                    int(
                        (
                            action_shift <= -0.10
                        ).sum()
                    ),
            }
        )

section6b_signed_action_shift_df = (
    pd.DataFrame(
        section6b_signed_shift_records
    )
    .sort_values(
        [
            "mean_signed_shift_b_minus_a",
            "maximum_signed_shift",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

print()
print("COMPLETE SIGNED ACTION-SHIFT PROFILE")
print("-" * 100)

display(
    section6b_signed_action_shift_df
)

print()
print("R0 BASELINE COMPARISONS")
print("-" * 100)

section6b_r0_comparisons_df = (
    section6b_signed_action_shift_df.loc[
        section6b_signed_action_shift_df[
            "strategy_a"
        ].eq(
            "R0_CURRENT_BASELINE"
        )
    ]
    .copy()
    .sort_values(
        [
            "strategy_b",
            "mean_signed_shift_b_minus_a",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

display(
    section6b_r0_comparisons_df
)

# For each strategy pair and case, probabilities must still sum to one.
section6b_pair_balance_rows = []

for strategy_a, strategy_b in itertools.combinations(
    strategy_names,
    2,
):

    probabilities_a = notebook56_predictions[
        strategy_a
    ]["probabilities"]

    probabilities_b = notebook56_predictions[
        strategy_b
    ]["probabilities"]

    row_shift_totals = (
        probabilities_b
        -
        probabilities_a
    ).sum(axis=1)

    section6b_pair_balance_rows.append(
        {
            "strategy_a":
                strategy_a,

            "strategy_b":
                strategy_b,

            "maximum_absolute_row_shift_sum":
                float(
                    np.abs(
                        row_shift_totals
                    ).max()
                ),

            "probability_mass_conserved":
                bool(
                    np.allclose(
                        row_shift_totals,
                        0.0,
                        atol=1e-10,
                    )
                ),
        }
    )

section6b_probability_balance_df = pd.DataFrame(
    section6b_pair_balance_rows
)

print()
print("PROBABILITY-MASS CONSERVATION")
print("-" * 100)

display(
    section6b_probability_balance_df
)

expected_action_count = len(
    notebook56_predictions[
        "R0_CURRENT_BASELINE"
    ]["classes"]
)

assert len(
    section6b_signed_action_shift_df
) == 6 * expected_action_count

assert section6b_probability_balance_df[
    "probability_mass_conserved"
].astype(bool).all()

print()
print("✅ SIGNED ACTION-LEVEL PROBABILITY SHIFT PROFILE BUILT")


# In[33]:


# ====================================================================================================
# SECTION 7A — BUILD FEATURE CONTRIBUTION INVENTORY
# ====================================================================================================

print("=" * 100)
print("SECTION 7A — BUILD FEATURE CONTRIBUTION INVENTORY")
print("=" * 100)

feature_records = []

baseline_features = notebook55_strategy_feature_artifact["strategy_feature_names"][
    "R0_CURRENT_BASELINE"
]

strategy_feature_names = notebook55_strategy_feature_artifact[
    "strategy_feature_names"
]

for strategy_name, feature_list in strategy_feature_names.items():

    removed_features = sorted(
        set(baseline_features) - set(feature_list)
    )

    retained_features = sorted(
        set(feature_list)
    )

    feature_records.append({

        "strategy": strategy_name,

        "feature_count": len(feature_list),

        "removed_feature_count": len(removed_features),

        "retained_feature_count": len(retained_features),

        "removed_features": removed_features,

        "retained_features": retained_features

    })

feature_inventory_df = pd.DataFrame(feature_records)

display(
    feature_inventory_df[
        [
            "strategy",
            "feature_count",
            "removed_feature_count"
        ]
    ]
)

print()
print("✅ FEATURE CONTRIBUTION INVENTORY CREATED")


# In[35]:


# ======================================================================================
# SECTION 7B — FEATURE REMOVAL IMPACT SUMMARY
# ======================================================================================

print("=" * 100)
print("SECTION 7B — FEATURE REMOVAL IMPACT SUMMARY")
print("=" * 100)

drift_summary_df = (
    section5d_pair_summary_df
    .copy()
)

feature_lookup_df = (
    feature_inventory_df
    .set_index("strategy")
)

section7b_impact_records = []

for _, row in drift_summary_df.iterrows():

    strategy_a = row[
        "strategy_a"
    ]

    strategy_b = row[
        "strategy_b"
    ]

    removed_features_a = int(
        feature_lookup_df.loc[
            strategy_a,
            "removed_feature_count",
        ]
    )

    removed_features_b = int(
        feature_lookup_df.loc[
            strategy_b,
            "removed_feature_count",
        ]
    )

    feature_difference = (
        removed_features_b
        -
        removed_features_a
    )

    section7b_impact_records.append(
        {
            "strategy_a":
                strategy_a,

            "strategy_b":
                strategy_b,

            "removed_features_a":
                removed_features_a,

            "removed_features_b":
                removed_features_b,

            "additional_features_removed":
                feature_difference,

            "cases":
                int(
                    row["cases"]
                ),

            "mean_maximum_case_difference":
                float(
                    row[
                        "mean_maximum_case_difference"
                    ]
                ),

            "median_maximum_case_difference":
                float(
                    row[
                        "median_maximum_case_difference"
                    ]
                ),

            "maximum_case_difference":
                float(
                    row[
                        "maximum_case_difference"
                    ]
                ),

            "mean_l1_case_difference":
                float(
                    row[
                        "mean_l1_case_difference"
                    ]
                ),

            "cases_above_005":
                int(
                    row[
                        "cases_above_005"
                    ]
                ),

            "cases_above_010":
                int(
                    row[
                        "cases_above_010"
                    ]
                ),

            "cases_above_015":
                int(
                    row[
                        "cases_above_015"
                    ]
                ),
        }
    )

section7b_feature_impact_df = (
    pd.DataFrame(
        section7b_impact_records
    )
    .sort_values(
        [
            "mean_maximum_case_difference",
            "maximum_case_difference",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

print()
print("FEATURE REMOVAL IMPACT")
print("-" * 100)

display(
    section7b_feature_impact_df
)

assert len(
    section7b_feature_impact_df
) == 6

assert (
    section7b_feature_impact_df[
        "cases"
    ] == 184
).all()

assert (
    section7b_feature_impact_df[
        "additional_features_removed"
    ] > 0
).all()

print()
print("✅ FEATURE REMOVAL IMPACT SUMMARY CREATED")


# In[36]:


# ====================================================================================================
# Section 7C — Feature Family Contribution Summary
# ====================================================================================================

print("=" * 100)
print("SECTION 7C — FEATURE FAMILY CONTRIBUTION SUMMARY")
print("=" * 100)

family_records = []

for _, row in feature_inventory_df.iterrows():

    strategy = row["strategy"]

    removed_features = row["removed_features"]

    family_counts = {}

    for feature in removed_features:

        family = feature.split("_")[0]

        family_counts[family] = family_counts.get(
            family,
            0
        ) + 1

    if len(family_counts) == 0:

        family_records.append({

            "strategy": strategy,

            "feature_family": "None",

            "removed_features": 0

        })

    else:

        for family, count in sorted(
            family_counts.items()
        ):

            family_records.append({

                "strategy": strategy,

                "feature_family": family,

                "removed_features": count

            })

section7c_feature_family_df = pd.DataFrame(
    family_records
)

display(
    section7c_feature_family_df
)

print()
print("✅ FEATURE FAMILY SUMMARY CREATED")


# In[38]:


# ======================================================================================
# SECTION 8A — NOTEBOOK 56 EXECUTIVE SUMMARY
# ======================================================================================

print("=" * 100)
print("SECTION 8A — NOTEBOOK 56 EXECUTIVE SUMMARY")
print("=" * 100)

# Identify the strategy pair with the largest mean case-level probability drift.
worst_comparison_row = (
    section7b_feature_impact_df
    .sort_values(
        [
            "mean_maximum_case_difference",
            "maximum_case_difference",
        ],
        ascending=False,
    )
    .iloc[0]
)

# Identify the action with the largest mean absolute probability drift.
top_action_row = (
    section6a_action_drift_df
    .sort_values(
        [
            "mean_probability_difference",
            "maximum_probability_difference",
        ],
        ascending=False,
    )
    .iloc[0]
)

section8a_summary_rows = [
    {
        "finding": "Evaluation cases",
        "value": 184,
    },
    {
        "finding": "Models compared",
        "value": len(notebook56_predictions),
    },
    {
        "finding": "Prediction agreement across all model pairs",
        "value": "100%",
    },
    {
        "finding": "Total pairwise prediction disagreements",
        "value": int(
            section5a_agreement_df[
                "disagreement_cases"
            ].sum()
        ),
    },
    {
        "finding": "Largest case-level probability drift",
        "value": round(
            float(
                section7b_feature_impact_df[
                    "maximum_case_difference"
                ].max()
            ),
            3,
        ),
    },
    {
        "finding": "Largest mean maximum case drift",
        "value": round(
            float(
                section7b_feature_impact_df[
                    "mean_maximum_case_difference"
                ].max()
            ),
            3,
        ),
    },
    {
        "finding": "Most divergent strategy comparison",
        "value": (
            str(
                worst_comparison_row[
                    "strategy_a"
                ]
            )
            + " vs "
            + str(
                worst_comparison_row[
                    "strategy_b"
                ]
            )
        ),
    },
    {
        "finding": "Additional features removed in most divergent comparison",
        "value": int(
            worst_comparison_row[
                "additional_features_removed"
            ]
        ),
    },
    {
        "finding": "Top drifting action",
        "value": str(
            top_action_row[
                "action"
            ]
        ),
    },
    {
        "finding": "Top action mean probability drift",
        "value": round(
            float(
                top_action_row[
                    "mean_probability_difference"
                ]
            ),
            3,
        ),
    },
    {
        "finding": "Top action maximum probability drift",
        "value": round(
            float(
                top_action_row[
                    "maximum_probability_difference"
                ]
            ),
            3,
        ),
    },
    {
        "finding": "Probability mass conserved",
        "value": bool(
            section6b_probability_balance_df[
                "probability_mass_conserved"
            ].astype(bool).all()
        ),
    },
]

section8a_summary_df = pd.DataFrame(
    section8a_summary_rows
)

print()
print("EXECUTIVE SUMMARY")
print("-" * 100)

display(
    section8a_summary_df
)

assert section8a_summary_df.shape[0] == 12

assert (
    section5a_agreement_df[
        "agreement"
    ] == 1.0
).all()

assert section6b_probability_balance_df[
    "probability_mass_conserved"
].astype(bool).all()

print()
print("✅ NOTEBOOK 56 EXECUTIVE SUMMARY CREATED")


# In[40]:


# ====================================================================================================
# SECTION 8B — EXPORT EXECUTIVE SUMMARY
# ====================================================================================================

print("=" * 100)
print("SECTION 8B — EXPORT EXECUTIVE SUMMARY")
print("=" * 100)

section8b_export_path = (
    NOTEBOOK55_REPORTS_DIRECTORY /
    "notebook56_executive_summary.csv"
)

section8a_summary_df.to_csv(
    section8b_export_path,
    index=False,
)

print()

print("Saved to:")
print(section8b_export_path)

assert section8b_export_path.exists()

print()
print("✅ EXECUTIVE SUMMARY EXPORTED")


# In[42]:


# ======================================================================================
# SECTION 8C — SAVE NOTEBOOK 56 ARTIFACTS
# ======================================================================================

print("=" * 100)
print("SECTION 8C — SAVE NOTEBOOK 56 ARTIFACTS")
print("=" * 100)

import pickle
from pathlib import Path

# Confirm that every required Notebook 56 object exists before saving.
section8c_required_objects = {
    "notebook56_predictions":
        notebook56_predictions,

    "notebook56_feature_matrices":
        notebook56_feature_matrices,

    "section5a_agreement_df":
        section5a_agreement_df,

    "notebook56_probability_difference_df":
        notebook56_probability_difference_df,

    "notebook56_case_probability_drift_df":
        notebook56_case_probability_drift_df,

    "section5d_pair_summary_df":
        section5d_pair_summary_df,

    "section5d_variant_summary_df":
        section5d_variant_summary_df,

    "section5d_side_summary_df":
        section5d_side_summary_df,

    "section6a_action_drift_df":
        section6a_action_drift_df,

    "section6b_signed_action_shift_df":
        section6b_signed_action_shift_df,

    "section6b_probability_balance_df":
        section6b_probability_balance_df,

    "feature_inventory_df":
        feature_inventory_df,

    "section7b_feature_impact_df":
        section7b_feature_impact_df,

    "section7c_feature_family_df":
        section7c_feature_family_df,

    "section8a_summary_df":
        section8a_summary_df,
}

print()
print("ARTIFACT OBJECT INVENTORY")
print("-" * 100)

section8c_object_inventory_rows = []

for object_name, object_value in section8c_required_objects.items():

    if isinstance(object_value, pd.DataFrame):
        object_rows = int(object_value.shape[0])
        object_columns = int(object_value.shape[1])

    elif isinstance(object_value, dict):
        object_rows = len(object_value)
        object_columns = None

    else:
        object_rows = None
        object_columns = None

    section8c_object_inventory_rows.append(
        {
            "object_name": object_name,
            "object_type": type(object_value).__name__,
            "rows_or_items": object_rows,
            "columns": object_columns,
            "available": object_value is not None,
        }
    )

section8c_object_inventory_df = pd.DataFrame(
    section8c_object_inventory_rows
)

display(section8c_object_inventory_df)

assert section8c_object_inventory_df[
    "available"
].astype(bool).all()


# Build a self-contained artifact package.
section8c_artifacts = {
    "metadata": {
        "notebook":
            56,

        "title":
            "Four Model Controlled Policy Evaluation",

        "evaluation_cases":
            184,

        "strategies":
            list(notebook56_predictions.keys()),

        "status":
            "CONTROLLED_FEATURE_ABLATION_EVALUATION_COMPLETE",
    },

    "predictions":
        notebook56_predictions,

    "feature_matrices":
        notebook56_feature_matrices,

    "prediction_agreement":
        section5a_agreement_df,

    "probability_difference":
        notebook56_probability_difference_df,

    "case_probability_drift":
        notebook56_case_probability_drift_df,

    "pair_drift_summary":
        section5d_pair_summary_df,

    "variant_drift_summary":
        section5d_variant_summary_df,

    "side_drift_summary":
        section5d_side_summary_df,

    "action_probability_drift":
        section6a_action_drift_df,

    "signed_action_probability_shift":
        section6b_signed_action_shift_df,

    "probability_mass_validation":
        section6b_probability_balance_df,

    "feature_inventory":
        feature_inventory_df,

    "feature_removal_impact":
        section7b_feature_impact_df,

    "feature_family_summary":
        section7c_feature_family_df,

    "executive_summary":
        section8a_summary_df,

    "object_inventory":
        section8c_object_inventory_df,
}


# Use the existing Notebook 55 artifact directory as the parent,
# but create a separate Notebook 56 folder.
NOTEBOOK56_ARTIFACTS_DIRECTORY = (
    Path(NOTEBOOK55_ARTIFACTS_DIRECTORY)
    / "notebook56"
)

NOTEBOOK56_ARTIFACTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK56_RESULTS_FILE = (
    NOTEBOOK56_ARTIFACTS_DIRECTORY
    / "notebook56_controlled_policy_evaluation_results.pkl"
)

with NOTEBOOK56_RESULTS_FILE.open("wb") as file:
    pickle.dump(
        section8c_artifacts,
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


# Reload immediately to confirm that the saved package is readable.
with NOTEBOOK56_RESULTS_FILE.open("rb") as file:
    section8c_reloaded_artifacts = pickle.load(file)


print()
print("SAVED NOTEBOOK 56 ARTIFACT")
print("-" * 100)
print(NOTEBOOK56_RESULTS_FILE)

print()
print("SAVED PACKAGE PROFILE")
print("-" * 100)

print(
    f"{'File exists':42}: "
    f"{NOTEBOOK56_RESULTS_FILE.exists()}"
)

print(
    f"{'File size bytes':42}: "
    f"{NOTEBOOK56_RESULTS_FILE.stat().st_size}"
)

print(
    f"{'Saved package keys':42}: "
    f"{len(section8c_artifacts)}"
)

print(
    f"{'Reloaded package keys':42}: "
    f"{len(section8c_reloaded_artifacts)}"
)

print(
    f"{'Reloaded status':42}: "
    f"{section8c_reloaded_artifacts['metadata']['status']}"
)


assert NOTEBOOK56_RESULTS_FILE.exists()

assert NOTEBOOK56_RESULTS_FILE.stat().st_size > 0

assert (
    set(section8c_reloaded_artifacts.keys())
    ==
    set(section8c_artifacts.keys())
)

assert (
    section8c_reloaded_artifacts[
        "metadata"
    ]["evaluation_cases"]
    == 184
)

assert (
    len(
        section8c_reloaded_artifacts[
            "predictions"
        ]
    )
    == 4
)

assert (
    section8c_reloaded_artifacts[
        "prediction_agreement"
    ]["agreement"].eq(1.0).all()
)

assert (
    section8c_reloaded_artifacts[
        "probability_mass_validation"
    ]["probability_mass_conserved"]
    .astype(bool)
    .all()
)

print()
print("✅ NOTEBOOK 56 ARTIFACTS SAVED AND RELOADED SUCCESSFULLY")


# In[2]:


# ======================================================================================
# SECTION 8D — RECOVER SAVED PACKAGE AND EXPORT NOTEBOOK 56 REPORTS
# ======================================================================================

print("=" * 100)
print("SECTION 8D — RECOVER SAVED PACKAGE AND EXPORT NOTEBOOK 56 REPORTS")
print("=" * 100)

from pathlib import Path
import json
import pickle

import pandas as pd


# --------------------------------------------------------------------------------------
# 1. DEFINE PATHS EXPLICITLY
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOK56_RESULTS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook55"
    / "notebook56"
    / "notebook56_controlled_policy_evaluation_results.pkl"
)

NOTEBOOK56_REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook56"
)

NOTEBOOK56_REPORTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

assert PROJECT_ROOT.exists(), (
    f"Project root was not found: {PROJECT_ROOT}"
)

assert NOTEBOOK56_RESULTS_FILE.exists(), (
    "The saved Notebook 56 artifact package was not found:\n"
    f"{NOTEBOOK56_RESULTS_FILE}"
)

print()
print("SOURCE ARTIFACT PACKAGE")
print("-" * 100)
print(NOTEBOOK56_RESULTS_FILE)

print()
print("REPORT OUTPUT DIRECTORY")
print("-" * 100)
print(NOTEBOOK56_REPORTS_DIRECTORY)


# --------------------------------------------------------------------------------------
# 2. LOAD THE SAVED NOTEBOOK 56 PACKAGE
# --------------------------------------------------------------------------------------

with NOTEBOOK56_RESULTS_FILE.open("rb") as file:
    notebook56_saved_results = pickle.load(file)

print()
print("SAVED PACKAGE KEYS")
print("-" * 100)

for key in notebook56_saved_results:
    print(key)

assert isinstance(
    notebook56_saved_results,
    dict,
)

assert len(
    notebook56_saved_results
) == 17


# --------------------------------------------------------------------------------------
# 3. MAP SAVED OBJECTS TO REPORT FILES
# --------------------------------------------------------------------------------------

report_exports = {
    "section5a_prediction_agreement.csv":
        notebook56_saved_results[
            "prediction_agreement"
        ],

    "section5b_probability_difference.csv":
        notebook56_saved_results[
            "probability_difference"
        ],

    "section5c_case_probability_drift.csv":
        notebook56_saved_results[
            "case_probability_drift"
        ],

    "section5d_pair_drift_summary.csv":
        notebook56_saved_results[
            "pair_drift_summary"
        ],

    "section5d_variant_drift_summary.csv":
        notebook56_saved_results[
            "variant_drift_summary"
        ],

    "section5d_side_drift_summary.csv":
        notebook56_saved_results[
            "side_drift_summary"
        ],

    "section6a_action_probability_drift.csv":
        notebook56_saved_results[
            "action_probability_drift"
        ],

    "section6b_signed_action_shift.csv":
        notebook56_saved_results[
            "signed_action_probability_shift"
        ],

    "section6b_probability_mass_validation.csv":
        notebook56_saved_results[
            "probability_mass_validation"
        ],

    "section7a_feature_inventory.csv":
        notebook56_saved_results[
            "feature_inventory"
        ],

    "section7b_feature_removal_impact.csv":
        notebook56_saved_results[
            "feature_removal_impact"
        ],

    "section7c_feature_family_summary.csv":
        notebook56_saved_results[
            "feature_family_summary"
        ],

    "section8a_executive_summary.csv":
        notebook56_saved_results[
            "executive_summary"
        ],

    "section8c_artifact_object_inventory.csv":
        notebook56_saved_results[
            "object_inventory"
        ],
}


# --------------------------------------------------------------------------------------
# 4. EXPORT ALL CSV REPORTS
# --------------------------------------------------------------------------------------

export_inventory_rows = []

for file_name, dataframe in report_exports.items():

    assert isinstance(
        dataframe,
        pd.DataFrame,
    ), (
        f"{file_name} did not resolve to a DataFrame. "
        f"Found: {type(dataframe).__name__}"
    )

    report_path = (
        NOTEBOOK56_REPORTS_DIRECTORY
        / file_name
    )

    dataframe.to_csv(
        report_path,
        index=False,
    )

    export_inventory_rows.append(
        {
            "file_name":
                file_name,

            "file_path":
                str(report_path),

            "rows":
                int(dataframe.shape[0]),

            "columns":
                int(dataframe.shape[1]),

            "exists":
                report_path.exists(),

            "size_bytes":
                (
                    report_path.stat().st_size
                    if report_path.exists()
                    else 0
                ),
        }
    )

section8d_export_inventory_df = pd.DataFrame(
    export_inventory_rows
)


# --------------------------------------------------------------------------------------
# 5. EXPORT JSON SUMMARY
# --------------------------------------------------------------------------------------

saved_metadata = notebook56_saved_results[
    "metadata"
]

executive_summary_df = notebook56_saved_results[
    "executive_summary"
]

notebook56_json_summary = {
    "notebook":
        int(
            saved_metadata[
                "notebook"
            ]
        ),

    "title":
        saved_metadata[
            "title"
        ],

    "status":
        saved_metadata[
            "status"
        ],

    "evaluation_cases":
        int(
            saved_metadata[
                "evaluation_cases"
            ]
        ),

    "strategies":
        list(
            saved_metadata[
                "strategies"
            ]
        ),

    "models_compared":
        4,

    "prediction_agreement":
        1.0,

    "prediction_disagreements":
        0,

    "largest_case_probability_drift":
        0.168,

    "most_divergent_comparison":
        (
            "R0_CURRENT_BASELINE vs "
            "R3_STATE_CENTRIC_POLICY"
        ),

    "top_drifting_action":
        "Quick Attack",

    "probability_mass_conserved":
        True,

    "csv_reports_exported":
        len(
            report_exports
        ),
}

NOTEBOOK56_JSON_SUMMARY_FILE = (
    NOTEBOOK56_REPORTS_DIRECTORY
    / "notebook56_evaluation_summary.json"
)

with NOTEBOOK56_JSON_SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook56_json_summary,
        file,
        indent=2,
    )


# --------------------------------------------------------------------------------------
# 6. SAVE THE REPORT INVENTORY ITSELF
# --------------------------------------------------------------------------------------

NOTEBOOK56_REPORT_INVENTORY_FILE = (
    NOTEBOOK56_REPORTS_DIRECTORY
    / "notebook56_report_inventory.csv"
)

section8d_export_inventory_df.to_csv(
    NOTEBOOK56_REPORT_INVENTORY_FILE,
    index=False,
)


# --------------------------------------------------------------------------------------
# 7. VALIDATE ALL EXPORTED FILES
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 56 REPORT EXPORT INVENTORY")
print("-" * 100)

display(
    section8d_export_inventory_df
)

all_csv_reports_exist = bool(
    section8d_export_inventory_df[
        "exists"
    ].astype(bool).all()
)

all_csv_reports_nonempty = bool(
    (
        section8d_export_inventory_df[
            "size_bytes"
        ] > 0
    ).all()
)

total_files_expected = (
    len(report_exports)
    + 2
)

total_files_found = len(
    list(
        NOTEBOOK56_REPORTS_DIRECTORY.glob("*")
    )
)

print()
print("EXPORT VALIDATION")
print("-" * 100)

print(
    f"{'CSV reports exported':42}: "
    f"{len(report_exports)}"
)

print(
    f"{'All CSV reports exist':42}: "
    f"{all_csv_reports_exist}"
)

print(
    f"{'All CSV reports are nonempty':42}: "
    f"{all_csv_reports_nonempty}"
)

print(
    f"{'JSON summary exists':42}: "
    f"{NOTEBOOK56_JSON_SUMMARY_FILE.exists()}"
)

print(
    f"{'Report inventory exists':42}: "
    f"{NOTEBOOK56_REPORT_INVENTORY_FILE.exists()}"
)

print(
    f"{'Total files expected':42}: "
    f"{total_files_expected}"
)

print(
    f"{'Total files currently found':42}: "
    f"{total_files_found}"
)

assert len(report_exports) == 14

assert all_csv_reports_exist

assert all_csv_reports_nonempty

assert NOTEBOOK56_JSON_SUMMARY_FILE.exists()

assert NOTEBOOK56_JSON_SUMMARY_FILE.stat().st_size > 0

assert NOTEBOOK56_REPORT_INVENTORY_FILE.exists()

assert NOTEBOOK56_REPORT_INVENTORY_FILE.stat().st_size > 0

assert total_files_found >= total_files_expected

print()
print("✅ NOTEBOOK 56 REPORTS RECOVERED AND EXPORTED SUCCESSFULLY")


# In[ ]:




