#!/usr/bin/env python
# coding: utf-8

# # Notebook 58 — Final Competitive Validation
# 
# ## Pokémon TCG AI Battle Challenge
# 
# ### Purpose
# 
# Notebook 58 performs the final pre-production competitive validation of the frozen policy candidates.
# 
# The primary candidate carried forward from Notebook 57 is:
# 
# **R3_STATE_CENTRIC_POLICY**
# 
# Notebook 58 will determine whether R3 is sufficiently stable and competitive to advance into production packaging.
# 
# ### Upstream evidence
# 
# Notebook 56 established:
# 
# - 184 controlled evaluation states
# - four frozen policy variants
# - 100% action-label agreement
# - probability drift up to 0.168
# 
# Notebook 57 established:
# 
# - feature importance attribution
# - ablation-stage attribution
# - high-drift case profiling
# - action-level sensitivity
# - signed probability redistribution
# - probability-mass conservation
# 
# The primary explainability-driven candidate is:
# 
# **R3_STATE_CENTRIC_POLICY**
# 
# ### Notebook 58 decision
# 
# This notebook must end with one of two outcomes:
# 
# - **SHIP R3**
# - **DO NOT SHIP R3**
# 
# No new policy family will be created here.

# In[1]:


# ==========================================================================================
# SECTION 1A — IMPORTS AND PROJECT PATHS
# ==========================================================================================

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

ARTIFACTS_55_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook55"
)

ARTIFACTS_56_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook56"
)

ARTIFACTS_57_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook57"
)

ARTIFACTS_58_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook58"
)

REPORTS_58_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook58"
)

MODELS_55_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "notebook55"
)

ARTIFACTS_58_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

REPORTS_58_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

paths_df = pd.DataFrame(
    [
        {
            "resource": "project_root",
            "path": str(PROJECT_ROOT),
            "exists": PROJECT_ROOT.exists(),
        },
        {
            "resource": "notebook55_artifacts",
            "path": str(ARTIFACTS_55_DIRECTORY),
            "exists": ARTIFACTS_55_DIRECTORY.exists(),
        },
        {
            "resource": "notebook56_artifacts",
            "path": str(ARTIFACTS_56_DIRECTORY),
            "exists": ARTIFACTS_56_DIRECTORY.exists(),
        },
        {
            "resource": "notebook57_artifacts",
            "path": str(ARTIFACTS_57_DIRECTORY),
            "exists": ARTIFACTS_57_DIRECTORY.exists(),
        },
        {
            "resource": "notebook58_artifacts",
            "path": str(ARTIFACTS_58_DIRECTORY),
            "exists": ARTIFACTS_58_DIRECTORY.exists(),
        },
        {
            "resource": "notebook58_reports",
            "path": str(REPORTS_58_DIRECTORY),
            "exists": REPORTS_58_DIRECTORY.exists(),
        },
        {
            "resource": "notebook55_models",
            "path": str(MODELS_55_DIRECTORY),
            "exists": MODELS_55_DIRECTORY.exists(),
        },
    ]
)

display(paths_df)

assert PROJECT_ROOT.exists()
assert ARTIFACTS_56_DIRECTORY.exists()
assert ARTIFACTS_57_DIRECTORY.exists()
assert MODELS_55_DIRECTORY.exists()

print("\n✅ SECTION 1A COMPLETE")


# In[5]:


# ==========================================================================================
# SECTION 1B — LOAD UPSTREAM RESULT PACKAGES
# ==========================================================================================

print("=" * 100)
print("SECTION 1B — LOAD UPSTREAM RESULT PACKAGES")
print("=" * 100)

NOTEBOOK56_RESULTS_FILE = (
    ARTIFACTS_56_DIRECTORY
    / "notebook56_controlled_policy_evaluation_results.pkl"
)

NOTEBOOK57_RESULTS_FILE = (
    ARTIFACTS_57_DIRECTORY
    / "notebook57_probability_attribution_results.pkl"
)

assert NOTEBOOK56_RESULTS_FILE.exists(), (
    f"Notebook 56 result package not found:\n"
    f"{NOTEBOOK56_RESULTS_FILE}"
)

assert NOTEBOOK57_RESULTS_FILE.exists(), (
    f"Notebook 57 result package not found:\n"
    f"{NOTEBOOK57_RESULTS_FILE}"
)

with open(
    NOTEBOOK56_RESULTS_FILE,
    "rb",
) as file:
    notebook56_results = pickle.load(file)

with open(
    NOTEBOOK57_RESULTS_FILE,
    "rb",
) as file:
    notebook57_results = pickle.load(file)

print("\nNOTEBOOK 56 PACKAGE KEYS")
print("-" * 100)

for key in notebook56_results.keys():
    print(key)

print("\nNOTEBOOK 57 PACKAGE KEYS")
print("-" * 100)

for key in notebook57_results.keys():
    print(key)

assert "metadata" in notebook56_results
assert "predictions" in notebook56_results
assert "feature_matrices" in notebook56_results

assert "metadata" in notebook57_results
assert "feature_importance" in notebook57_results
assert "action_sensitivity" in notebook57_results
assert "policy_recommendation" in notebook57_results

print("\n✅ NOTEBOOK 56 RESULTS LOADED")
print("✅ NOTEBOOK 57 RESULTS LOADED")
print("✅ SECTION 1B COMPLETE")


# In[6]:


# ==========================================================================================
# SECTION 2A — DISCOVER AND LOAD FROZEN POLICY MODELS
# ==========================================================================================

print("=" * 100)
print("SECTION 2A — DISCOVER AND LOAD FROZEN POLICY MODELS")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Expected frozen models
# ------------------------------------------------------------------------------------------

EXPECTED_MODELS = {
    "R0_CURRENT_BASELINE": (
        MODELS_55_DIRECTORY
        / "section3a_r0_current_baseline_model.joblib"
    ),

    "R1_REMOVE_LEGAL_SIGNATURE": (
        MODELS_55_DIRECTORY
        / "section3a_r1_remove_legal_signature_model.joblib"
    ),

    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY": (
        MODELS_55_DIRECTORY
        / "section3a_r2_remove_signature_and_compatibility_model.joblib"
    ),

    "R3_STATE_CENTRIC_POLICY": (
        MODELS_55_DIRECTORY
        / "section3a_r3_state_centric_policy_model.joblib"
    ),
}

# ------------------------------------------------------------------------------------------
# Validate files exist
# ------------------------------------------------------------------------------------------

model_file_records = []

for strategy, model_path in EXPECTED_MODELS.items():

    model_file_records.append(
        {
            "strategy": strategy,
            "filename": model_path.name,
            "path": str(model_path),
            "exists": model_path.exists(),
            "size_bytes": (
                model_path.stat().st_size
                if model_path.exists()
                else None
            ),
        }
    )

section2a_model_files_df = pd.DataFrame(
    model_file_records
)

print("\nFROZEN MODEL FILES")
print("-" * 100)

display(section2a_model_files_df)

assert section2a_model_files_df["exists"].all(), (
    "One or more frozen policy model files are missing."
)

# ------------------------------------------------------------------------------------------
# Load model packages
# ------------------------------------------------------------------------------------------

frozen_policy_packages = {}

for strategy, model_path in EXPECTED_MODELS.items():

    frozen_policy_packages[strategy] = joblib.load(
        model_path
    )

print("\n✅ ALL FOUR FROZEN POLICY PACKAGES LOADED")

# ------------------------------------------------------------------------------------------
# Display package structure
# ------------------------------------------------------------------------------------------

package_records = []

for strategy, package in frozen_policy_packages.items():

    if isinstance(package, dict):

        package_keys = list(package.keys())

        estimator = package.get("model")

        retained_feature_names = package.get(
            "retained_feature_names"
        )

        retained_feature_indices = package.get(
            "retained_feature_indices"
        )

        package_strategy_id = package.get(
            "strategy_id"
        )

    else:

        package_keys = []

        estimator = package

        retained_feature_names = None
        retained_feature_indices = None
        package_strategy_id = None

    package_records.append(
        {
            "strategy": strategy,
            "package_type": type(package).__name__,
            "package_strategy_id": package_strategy_id,
            "package_keys": package_keys,
            "estimator_type": (
                type(estimator).__name__
                if estimator is not None
                else None
            ),
            "retained_feature_name_count": (
                len(retained_feature_names)
                if retained_feature_names is not None
                else None
            ),
            "retained_feature_index_count": (
                len(retained_feature_indices)
                if retained_feature_indices is not None
                else None
            ),
        }
    )

section2a_package_structure_df = pd.DataFrame(
    package_records
)

print("\nFROZEN MODEL PACKAGE STRUCTURE")
print("-" * 100)

display(section2a_package_structure_df)

print("\n✅ SECTION 2A COMPLETE")


# In[8]:


# ==========================================================================================
# SECTION 2B — VALIDATE MODEL / FEATURE MATRIX COMPATIBILITY
# ==========================================================================================

print("=" * 100)
print("SECTION 2B — VALIDATE MODEL / FEATURE MATRIX COMPATIBILITY")
print("=" * 100)

EXPECTED_FEATURE_COUNTS = {
    "R0_CURRENT_BASELINE": 64,
    "R1_REMOVE_LEGAL_SIGNATURE": 58,
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY": 52,
    "R3_STATE_CENTRIC_POLICY": 46,
}

# Notebook 56 matrices
notebook56_feature_matrices = notebook56_results[
    "feature_matrices"
]

validation_records = []

for strategy, expected_count in EXPECTED_FEATURE_COUNTS.items():

    package = frozen_policy_packages[strategy]

    # ------------------------------------------------------------------
    # Extract estimator
    # ------------------------------------------------------------------

    if isinstance(package, dict):
        estimator = package.get("model")
        retained_feature_names = package.get(
            "retained_feature_names"
        )
    else:
        estimator = package
        retained_feature_names = None

    assert estimator is not None, (
        f"No estimator found for {strategy}."
    )

    # ------------------------------------------------------------------
    # Extract Notebook 56 feature matrix
    # ------------------------------------------------------------------

    assert strategy in notebook56_feature_matrices, (
        f"Notebook 56 feature matrix missing for {strategy}."
    )

    X = np.asarray(
        notebook56_feature_matrices[strategy]
    )

    # ------------------------------------------------------------------
    # Model dimensions
    # ------------------------------------------------------------------

    model_n_features = getattr(
        estimator,
        "n_features_in_",
        None,
    )

    feature_importances = getattr(
        estimator,
        "feature_importances_",
        None,
    )

    importance_length = (
        len(feature_importances)
        if feature_importances is not None
        else None
    )

    feature_name_count = (
        len(retained_feature_names)
        if retained_feature_names is not None
        else None
    )

    validation_records.append(
        {
            "strategy": strategy,
            "expected_feature_count": expected_count,
            "matrix_rows": X.shape[0],
            "matrix_columns": X.shape[1],
            "model_n_features_in": model_n_features,
            "retained_feature_name_count": feature_name_count,
            "feature_importance_length": importance_length,
            "matrix_count_matches": (
                X.shape[1] == expected_count
            ),
            "model_count_matches": (
                model_n_features == expected_count
            ),
            "feature_name_count_matches": (
                feature_name_count == expected_count
            ),
            "importance_length_matches": (
                importance_length == expected_count
            ),
        }
    )

section2b_model_validation_df = pd.DataFrame(
    validation_records
)

print("\nMODEL / FEATURE MATRIX VALIDATION")
print("-" * 100)

display(section2b_model_validation_df)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

required_boolean_columns = [
    "matrix_count_matches",
    "model_count_matches",
    "feature_name_count_matches",
    "importance_length_matches",
]

for column in required_boolean_columns:

    assert section2b_model_validation_df[
        column
    ].all(), (
        f"Frozen policy validation failed: {column}"
    )

assert (
    section2b_model_validation_df[
        "matrix_rows"
    ] == 184
).all(), (
    "Expected 184 controlled evaluation states "
    "for every frozen policy."
)

print("\nEXPECTED FEATURE COUNTS")
print("-" * 100)

for strategy, count in EXPECTED_FEATURE_COUNTS.items():
    print(f"{strategy}: {count}")

print("\n✅ ALL MODEL FEATURE COUNTS MATCH")
print("✅ ALL NOTEBOOK 56 MATRICES MATCH FROZEN MODELS")
print("✅ ALL FEATURE-IMPORTANCE VECTORS MATCH")
print("✅ SECTION 2B COMPLETE")


# In[9]:


# ==========================================================================================
# SECTION 3A — INDEPENDENT PREDICTION REPRODUCTION
# ==========================================================================================

print("=" * 100)
print("SECTION 3A — INDEPENDENT PREDICTION REPRODUCTION")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Notebook 56 reference predictions
# ------------------------------------------------------------------------------------------

notebook56_predictions = notebook56_results[
    "predictions"
]

reproduction_records = []

reproduced_predictions = {}
reproduced_probabilities = {}

for strategy in EXPECTED_FEATURE_COUNTS.keys():

    package = frozen_policy_packages[strategy]

    # ------------------------------------------------------------------
    # Extract estimator
    # ------------------------------------------------------------------

    if isinstance(package, dict):
        estimator = package.get("model")
    else:
        estimator = package

    assert estimator is not None, (
        f"No estimator available for {strategy}."
    )

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    X = np.asarray(
        notebook56_feature_matrices[strategy]
    )

    # ------------------------------------------------------------------
    # Reproduce predictions
    # ------------------------------------------------------------------

    current_predictions = estimator.predict(X)

    current_probabilities = estimator.predict_proba(X)

    reproduced_predictions[strategy] = current_predictions
    reproduced_probabilities[strategy] = current_probabilities

    # ------------------------------------------------------------------
    # Reference Notebook 56 outputs
    # ------------------------------------------------------------------

    reference_package = notebook56_predictions[
        strategy
    ]

    reference_predictions = np.asarray(
        reference_package["predictions"]
    )

    reference_probabilities = np.asarray(
        reference_package["probabilities"]
    )

    reference_classes = np.asarray(
        reference_package["classes"]
    )

    current_classes = np.asarray(
        estimator.classes_
    )

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    prediction_matches = np.array_equal(
        current_predictions,
        reference_predictions,
    )

    class_order_matches = np.array_equal(
        current_classes,
        reference_classes,
    )

    probability_shape_matches = (
        current_probabilities.shape
        == reference_probabilities.shape
    )

    if probability_shape_matches:

        maximum_probability_difference = float(
            np.max(
                np.abs(
                    current_probabilities
                    - reference_probabilities
                )
            )
        )

        mean_probability_difference = float(
            np.mean(
                np.abs(
                    current_probabilities
                    - reference_probabilities
                )
            )
        )

        probabilities_match = np.allclose(
            current_probabilities,
            reference_probabilities,
            rtol=0.0,
            atol=1e-12,
        )

    else:

        maximum_probability_difference = np.nan
        mean_probability_difference = np.nan
        probabilities_match = False

    reproduction_records.append(
        {
            "strategy": strategy,
            "cases": len(current_predictions),
            "class_count": len(current_classes),
            "prediction_matches": prediction_matches,
            "class_order_matches": class_order_matches,
            "probability_shape_matches": probability_shape_matches,
            "probabilities_match": probabilities_match,
            "mean_probability_difference": (
                mean_probability_difference
            ),
            "maximum_probability_difference": (
                maximum_probability_difference
            ),
        }
    )

section3a_prediction_reproduction_df = pd.DataFrame(
    reproduction_records
)

print("\nINDEPENDENT PREDICTION REPRODUCTION")
print("-" * 100)

display(
    section3a_prediction_reproduction_df
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert (
    section3a_prediction_reproduction_df[
        "cases"
    ] == 184
).all(), (
    "Expected 184 reproduced predictions "
    "for every frozen policy."
)

assert section3a_prediction_reproduction_df[
    "prediction_matches"
].all(), (
    "At least one frozen model failed "
    "to reproduce Notebook 56 predictions."
)

assert section3a_prediction_reproduction_df[
    "class_order_matches"
].all(), (
    "At least one model class ordering differs "
    "from Notebook 56."
)

assert section3a_prediction_reproduction_df[
    "probability_shape_matches"
].all(), (
    "At least one probability matrix shape differs "
    "from Notebook 56."
)

assert section3a_prediction_reproduction_df[
    "probabilities_match"
].all(), (
    "At least one frozen model failed "
    "to reproduce Notebook 56 probabilities."
)

assert (
    section3a_prediction_reproduction_df[
        "maximum_probability_difference"
    ] <= 1e-12
).all(), (
    "Probability reproduction exceeded "
    "the allowed numerical tolerance."
)

print("\n✅ ALL 184 PREDICTIONS REPRODUCED FOR ALL FOUR POLICIES")
print("✅ CLASS ORDERINGS MATCH NOTEBOOK 56")
print("✅ PROBABILITY MATRICES MATCH NOTEBOOK 56")
print("✅ FROZEN POLICY REPRODUCIBILITY CONFIRMED")
print("✅ SECTION 3A COMPLETE")


# In[10]:


# ==========================================================================================
# SECTION 3B — CROSS-POLICY PREDICTION CONSISTENCY
# ==========================================================================================

print("=" * 100)
print("SECTION 3B — CROSS-POLICY PREDICTION CONSISTENCY")
print("=" * 100)

from itertools import combinations

strategy_pairs = list(
    combinations(
        EXPECTED_FEATURE_COUNTS.keys(),
        2,
    )
)

comparison_records = []

for strategy_a, strategy_b in strategy_pairs:

    predictions_a = reproduced_predictions[strategy_a]
    predictions_b = reproduced_predictions[strategy_b]

    probabilities_a = reproduced_probabilities[strategy_a]
    probabilities_b = reproduced_probabilities[strategy_b]

    identical_predictions = (
        predictions_a == predictions_b
    )

    agreement_count = int(
        identical_predictions.sum()
    )

    disagreement_count = (
        len(predictions_a)
        - agreement_count
    )

    agreement_rate = (
        agreement_count
        / len(predictions_a)
    )

    confidence_a = np.max(
        probabilities_a,
        axis=1,
    )

    confidence_b = np.max(
        probabilities_b,
        axis=1,
    )

    confidence_difference = (
        confidence_b
        - confidence_a
    )

    comparison_records.append(
        {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "cases": len(predictions_a),
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "agreement_rate": agreement_rate,
            "mean_confidence_difference": float(
                np.mean(confidence_difference)
            ),
            "median_confidence_difference": float(
                np.median(confidence_difference)
            ),
            "maximum_confidence_difference": float(
                np.max(
                    np.abs(confidence_difference)
                )
            ),
        }
    )

section3b_prediction_consistency_df = pd.DataFrame(
    comparison_records
)

print("\nCROSS-POLICY PREDICTION CONSISTENCY")
print("-" * 100)

display(
    section3b_prediction_consistency_df
)

# ==========================================================================================
# SUMMARY
# ==========================================================================================

highest_agreement = (
    section3b_prediction_consistency_df
    .sort_values(
        "agreement_rate",
        ascending=False,
    )
    .iloc[0]
)

lowest_agreement = (
    section3b_prediction_consistency_df
    .sort_values(
        "agreement_rate",
        ascending=True,
    )
    .iloc[0]
)

print("\nMOST SIMILAR POLICIES")
print("-" * 100)

print(
    f"{highest_agreement.strategy_a} ↔ "
    f"{highest_agreement.strategy_b}"
)

print(
    f"Agreement rate: "
    f"{highest_agreement.agreement_rate:.4f}"
)

print("\nMOST DIFFERENT POLICIES")
print("-" * 100)

print(
    f"{lowest_agreement.strategy_a} ↔ "
    f"{lowest_agreement.strategy_b}"
)

print(
    f"Agreement rate: "
    f"{lowest_agreement.agreement_rate:.4f}"
)

assert (
    section3b_prediction_consistency_df["cases"]
    == 184
).all()

print("\n✅ ALL STRATEGY PAIRS VALIDATED")
print("✅ CROSS-POLICY CONSISTENCY COMPUTED")
print("✅ SECTION 3B COMPLETE")


# In[11]:


# ==========================================================================================
# SECTION 3C — POLICY DIVERGENCE CASE ANALYSIS
# ==========================================================================================

print("=" * 100)
print("SECTION 3C — POLICY DIVERGENCE CASE ANALYSIS")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Build pairwise decision-divergence records
# ------------------------------------------------------------------------------------------

decision_divergence_records = []
confidence_divergence_records = []

for strategy_a, strategy_b in strategy_pairs:

    predictions_a = reproduced_predictions[strategy_a]
    predictions_b = reproduced_predictions[strategy_b]

    probabilities_a = reproduced_probabilities[strategy_a]
    probabilities_b = reproduced_probabilities[strategy_b]

    decision_diff_mask = predictions_a != predictions_b

    # ------------------------------------------------------
    # Decision-level divergence summary
    # ------------------------------------------------------

    decision_divergence_records.append(
        {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "cases": len(predictions_a),
            "decision_disagreements": int(
                decision_diff_mask.sum()
            ),
            "decision_agreements": int(
                (~decision_diff_mask).sum()
            ),
            "agreement_rate": float(
                (~decision_diff_mask).mean()
            ),
        }
    )

    # ------------------------------------------------------
    # Confidence-level divergence per case
    # ------------------------------------------------------

    confidence_a = np.max(
        probabilities_a,
        axis=1,
    )

    confidence_b = np.max(
        probabilities_b,
        axis=1,
    )

    absolute_confidence_difference = np.abs(
        confidence_b
        - confidence_a
    )

    for case_index in range(
        len(predictions_a)
    ):

        confidence_divergence_records.append(
            {
                "case_index": case_index,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "prediction_a": predictions_a[
                    case_index
                ],
                "prediction_b": predictions_b[
                    case_index
                ],
                "predictions_match": bool(
                    predictions_a[
                        case_index
                    ]
                    == predictions_b[
                        case_index
                    ]
                ),
                "confidence_a": float(
                    confidence_a[
                        case_index
                    ]
                ),
                "confidence_b": float(
                    confidence_b[
                        case_index
                    ]
                ),
                "signed_confidence_difference": float(
                    confidence_b[
                        case_index
                    ]
                    - confidence_a[
                        case_index
                    ]
                ),
                "absolute_confidence_difference": float(
                    absolute_confidence_difference[
                        case_index
                    ]
                ),
            }
        )

section3c_decision_divergence_df = pd.DataFrame(
    decision_divergence_records
)

section3c_confidence_divergence_df = pd.DataFrame(
    confidence_divergence_records
)

# ------------------------------------------------------------------------------------------
# Decision divergence summary
# ------------------------------------------------------------------------------------------

print("\nDECISION DIVERGENCE SUMMARY")
print("-" * 100)

display(
    section3c_decision_divergence_df
)

# ------------------------------------------------------------------------------------------
# Top confidence-divergence cases
# ------------------------------------------------------------------------------------------

section3c_top_confidence_divergence_df = (
    section3c_confidence_divergence_df
    .sort_values(
        "absolute_confidence_difference",
        ascending=False,
    )
    .head(25)
    .reset_index(drop=True)
)

section3c_top_confidence_divergence_df.insert(
    0,
    "confidence_divergence_rank",
    np.arange(
        1,
        len(
            section3c_top_confidence_divergence_df
        ) + 1,
    ),
)

print("\nTOP 25 CONFIDENCE-DIVERGENCE CASES")
print("-" * 100)

display(
    section3c_top_confidence_divergence_df
)

# ------------------------------------------------------------------------------------------
# Pair-level confidence divergence summary
# ------------------------------------------------------------------------------------------

section3c_pair_confidence_summary_df = (
    section3c_confidence_divergence_df
    .groupby(
        [
            "strategy_a",
            "strategy_b",
        ],
        as_index=False,
    )
    .agg(
        cases=(
            "case_index",
            "size",
        ),
        mean_absolute_confidence_difference=(
            "absolute_confidence_difference",
            "mean",
        ),
        median_absolute_confidence_difference=(
            "absolute_confidence_difference",
            "median",
        ),
        maximum_absolute_confidence_difference=(
            "absolute_confidence_difference",
            "max",
        ),
        mean_signed_confidence_difference=(
            "signed_confidence_difference",
            "mean",
        ),
    )
    .sort_values(
        "maximum_absolute_confidence_difference",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nPAIR-LEVEL CONFIDENCE DIVERGENCE SUMMARY")
print("-" * 100)

display(
    section3c_pair_confidence_summary_df
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert (
    section3c_decision_divergence_df[
        "decision_disagreements"
    ] == 0
).all(), (
    "At least one policy pair produced "
    "a decision disagreement."
)

assert (
    section3c_decision_divergence_df[
        "agreement_rate"
    ] == 1.0
).all(), (
    "Expected 100% decision agreement "
    "across all frozen policy pairs."
)

assert (
    section3c_confidence_divergence_df[
        "predictions_match"
    ]
).all(), (
    "At least one case-level prediction differs."
)

# ------------------------------------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------------------------------------

most_divergent_pair = (
    section3c_pair_confidence_summary_df
    .iloc[0]
)

print("\nINTERPRETATION")
print("-" * 100)

print(
    "Decision disagreements across all "
    "strategy pairs: 0"
)

print(
    "All observed policy differences are "
    "confidence-level differences."
)

print(
    "Largest confidence-divergence pair: "
    f"{most_divergent_pair['strategy_a']} "
    "vs "
    f"{most_divergent_pair['strategy_b']}"
)

print(
    "Maximum absolute confidence difference: "
    f"{most_divergent_pair['maximum_absolute_confidence_difference']:.4f}"
)

print("\n✅ ZERO ACTION-LEVEL POLICY DIVERGENCE CONFIRMED")
print("✅ CONFIDENCE-LEVEL DIVERGENCE PROFILED")
print("✅ SECTION 3C COMPLETE")


# In[12]:


# ==========================================================================================
# SECTION 4A — INSPECT FROZEN MODEL PERFORMANCE METRICS
# ==========================================================================================

print("=" * 100)
print("SECTION 4A — INSPECT FROZEN MODEL PERFORMANCE METRICS")
print("=" * 100)

performance_structure_records = []

for strategy, package in frozen_policy_packages.items():

    print("\n" + "-" * 100)
    print(strategy)
    print("-" * 100)

    if not isinstance(package, dict):

        print(
            f"Package type: {type(package).__name__}"
        )

        performance_structure_records.append(
            {
                "strategy": strategy,
                "has_training_results": False,
                "training_result_keys": None,
                "train_metric_keys": None,
                "validation_metric_keys": None,
                "test_metric_keys": None,
            }
        )

        continue

    training_results = package.get(
        "training_results"
    )

    print(
        "training_results type:",
        type(training_results).__name__
    )

    if not isinstance(training_results, dict):

        performance_structure_records.append(
            {
                "strategy": strategy,
                "has_training_results": False,
                "training_result_keys": None,
                "train_metric_keys": None,
                "validation_metric_keys": None,
                "test_metric_keys": None,
            }
        )

        continue

    print(
        "\ntraining_results keys:"
    )

    for key in training_results.keys():

        value = training_results[key]

        print(
            f"  {key}: "
            f"{type(value).__name__}"
        )

        if isinstance(value, dict):

            print(
                f"      keys = "
                f"{list(value.keys())}"
            )

            # Print scalar metric values only
            scalar_items = {
                metric_name: metric_value
                for metric_name, metric_value
                in value.items()
                if np.isscalar(metric_value)
            }

            if scalar_items:

                print(
                    "      scalar values =",
                    scalar_items,
                )

    train_metrics = training_results.get(
        "train_metrics"
    )

    validation_metrics = training_results.get(
        "validation_metrics"
    )

    test_metrics = training_results.get(
        "test_metrics"
    )

    performance_structure_records.append(
        {
            "strategy": strategy,
            "has_training_results": True,
            "training_result_keys": list(
                training_results.keys()
            ),
            "train_metric_keys": (
                list(train_metrics.keys())
                if isinstance(
                    train_metrics,
                    dict,
                )
                else None
            ),
            "validation_metric_keys": (
                list(
                    validation_metrics.keys()
                )
                if isinstance(
                    validation_metrics,
                    dict,
                )
                else None
            ),
            "test_metric_keys": (
                list(test_metrics.keys())
                if isinstance(
                    test_metrics,
                    dict,
                )
                else None
            ),
        }
    )

section4a_metric_structure_df = pd.DataFrame(
    performance_structure_records
)

print(
    "\nFROZEN MODEL PERFORMANCE-METRIC STRUCTURE"
)

print("-" * 100)

display(
    section4a_metric_structure_df
)

assert section4a_metric_structure_df[
    "has_training_results"
].all(), (
    "At least one frozen model package "
    "does not contain training_results."
)

print(
    "\n✅ TRAINING RESULTS FOUND FOR ALL FOUR FROZEN POLICIES"
)

print(
    "✅ PERFORMANCE METRIC STRUCTURE INSPECTED"
)

print(
    "✅ SECTION 4A COMPLETE"
)


# In[13]:


# ==========================================================================================
# SECTION 4B — FROZEN POLICY PERFORMANCE LEADERBOARD
# ==========================================================================================

print("=" * 100)
print("SECTION 4B — FROZEN POLICY PERFORMANCE LEADERBOARD")
print("=" * 100)

performance_records = []

for strategy, package in frozen_policy_packages.items():

    training_results = package["training_results"]

    train_metrics = training_results["train_metrics"]
    validation_metrics = training_results["validation_metrics"]
    test_metrics = training_results["test_metrics"]

    retained_feature_names = package.get(
        "retained_feature_names",
        [],
    )

    performance_records.append(
        {
            "strategy": strategy,

            "feature_count": len(
                retained_feature_names
            ),

            "training_seconds": float(
                training_results[
                    "training_seconds"
                ]
            ),

            # TRAIN
            "train_rows": train_metrics["rows"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics[
                "weighted_f1"
            ],
            "train_log_loss": train_metrics["log_loss"],

            # VALIDATION
            "validation_rows": validation_metrics["rows"],
            "validation_accuracy": validation_metrics[
                "accuracy"
            ],
            "validation_balanced_accuracy": validation_metrics[
                "balanced_accuracy"
            ],
            "validation_macro_f1": validation_metrics[
                "macro_f1"
            ],
            "validation_weighted_f1": validation_metrics[
                "weighted_f1"
            ],
            "validation_log_loss": validation_metrics[
                "log_loss"
            ],

            # TEST
            "test_rows": test_metrics["rows"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics[
                "balanced_accuracy"
            ],
            "test_macro_f1": test_metrics[
                "macro_f1"
            ],
            "test_weighted_f1": test_metrics[
                "weighted_f1"
            ],
            "test_log_loss": test_metrics[
                "log_loss"
            ],
        }
    )

section4b_performance_leaderboard_df = pd.DataFrame(
    performance_records
)

# ------------------------------------------------------------------------------------------
# Complexity reduction relative to R0
# ------------------------------------------------------------------------------------------

baseline_feature_count = int(
    section4b_performance_leaderboard_df.loc[
        section4b_performance_leaderboard_df[
            "strategy"
        ] == "R0_CURRENT_BASELINE",
        "feature_count",
    ].iloc[0]
)

section4b_performance_leaderboard_df[
    "features_removed_vs_r0"
] = (
    baseline_feature_count
    - section4b_performance_leaderboard_df[
        "feature_count"
    ]
)

section4b_performance_leaderboard_df[
    "feature_reduction_percent_vs_r0"
] = (
    section4b_performance_leaderboard_df[
        "features_removed_vs_r0"
    ]
    / baseline_feature_count
    * 100.0
)

# ------------------------------------------------------------------------------------------
# Performance preservation flags
# ------------------------------------------------------------------------------------------

section4b_performance_leaderboard_df[
    "perfect_validation_accuracy"
] = (
    section4b_performance_leaderboard_df[
        "validation_accuracy"
    ] == 1.0
)

section4b_performance_leaderboard_df[
    "perfect_test_accuracy"
] = (
    section4b_performance_leaderboard_df[
        "test_accuracy"
    ] == 1.0
)

section4b_performance_leaderboard_df[
    "perfect_test_balanced_accuracy"
] = (
    section4b_performance_leaderboard_df[
        "test_balanced_accuracy"
    ] == 1.0
)

section4b_performance_leaderboard_df[
    "perfect_test_macro_f1"
] = (
    section4b_performance_leaderboard_df[
        "test_macro_f1"
    ] == 1.0
)

# ------------------------------------------------------------------------------------------
# Rank by:
# 1. Test performance
# 2. Fewer features
# ------------------------------------------------------------------------------------------

section4b_performance_leaderboard_df = (
    section4b_performance_leaderboard_df
    .sort_values(
        by=[
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "feature_count",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

section4b_performance_leaderboard_df.insert(
    0,
    "performance_rank",
    np.arange(
        1,
        len(
            section4b_performance_leaderboard_df
        ) + 1,
    ),
)

print("\nFROZEN POLICY PERFORMANCE LEADERBOARD")
print("-" * 100)

display(
    section4b_performance_leaderboard_df
)

# ==========================================================================================
# PERFORMANCE EQUIVALENCE CHECK
# ==========================================================================================

metric_columns = [
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "validation_weighted_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
]

metric_ranges = {}

for metric in metric_columns:

    metric_range = (
        section4b_performance_leaderboard_df[
            metric
        ].max()
        -
        section4b_performance_leaderboard_df[
            metric
        ].min()
    )

    metric_ranges[metric] = float(
        metric_range
    )

section4b_metric_equivalence_df = pd.DataFrame(
    [
        {
            "metric": metric,
            "minimum": float(
                section4b_performance_leaderboard_df[
                    metric
                ].min()
            ),
            "maximum": float(
                section4b_performance_leaderboard_df[
                    metric
                ].max()
            ),
            "range": metric_ranges[metric],
            "identical_across_strategies": (
                metric_ranges[metric] == 0.0
            ),
        }
        for metric in metric_columns
    ]
)

print("\nPERFORMANCE EQUIVALENCE ACROSS STRATEGIES")
print("-" * 100)

display(
    section4b_metric_equivalence_df
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert (
    section4b_performance_leaderboard_df[
        "validation_rows"
    ] == 110
).all()

assert (
    section4b_performance_leaderboard_df[
        "test_rows"
    ] == 110
).all()

assert section4b_performance_leaderboard_df[
    "perfect_validation_accuracy"
].all()

assert section4b_performance_leaderboard_df[
    "perfect_test_accuracy"
].all()

assert section4b_performance_leaderboard_df[
    "perfect_test_balanced_accuracy"
].all()

assert section4b_performance_leaderboard_df[
    "perfect_test_macro_f1"
].all()

assert section4b_metric_equivalence_df[
    "identical_across_strategies"
].all()

# ==========================================================================================
# R3 PERFORMANCE PRESERVATION
# ==========================================================================================

r3_row = (
    section4b_performance_leaderboard_df[
        section4b_performance_leaderboard_df[
            "strategy"
        ] == "R3_STATE_CENTRIC_POLICY"
    ]
    .iloc[0]
)

print("\nR3 PERFORMANCE PRESERVATION")
print("-" * 100)

print(
    f"Feature count: "
    f"{int(r3_row['feature_count'])}"
)

print(
    f"Features removed vs R0: "
    f"{int(r3_row['features_removed_vs_r0'])}"
)

print(
    f"Feature reduction vs R0: "
    f"{r3_row['feature_reduction_percent_vs_r0']:.2f}%"
)

print(
    f"Validation accuracy: "
    f"{r3_row['validation_accuracy']:.4f}"
)

print(
    f"Test accuracy: "
    f"{r3_row['test_accuracy']:.4f}"
)

print(
    f"Test balanced accuracy: "
    f"{r3_row['test_balanced_accuracy']:.4f}"
)

print(
    f"Test macro F1: "
    f"{r3_row['test_macro_f1']:.4f}"
)

print("\nINTERPRETATION")
print("-" * 100)

print(
    "All four frozen policies are indistinguishable "
    "on the stored validation/test classification metrics."
)

print(
    "R3 preserves the same stored validation and test "
    "performance while using the smallest feature set."
)

print(
    "Therefore, stored classification metrics alone "
    "cannot determine the final production winner."
)

print(
    "The final ship decision must also incorporate "
    "controlled ablation behavior, confidence stability, "
    "robustness, and competitive/live evaluation."
)

print("\n✅ ALL FROZEN POLICIES PRESERVE HELD-OUT PERFORMANCE")
print("✅ R3 PRESERVES PERFORMANCE WITH 46 FEATURES")
print("✅ PERFORMANCE EQUIVALENCE CONFIRMED")
print("✅ SECTION 4B COMPLETE")


# In[14]:


# ==========================================================================================
# SECTION 4C — FINAL EVIDENCE MATRIX
# ==========================================================================================

print("=" * 100)
print("SECTION 4C — FINAL EVIDENCE MATRIX")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Helper lookups
# ------------------------------------------------------------------------------------------

performance_lookup = (
    section4b_performance_leaderboard_df
    .set_index("strategy")
)

reproduction_lookup = (
    section3a_prediction_reproduction_df
    .set_index("strategy")
)

# Pairwise consistency summary for R0 comparisons
r0_pair_lookup = (
    section3b_prediction_consistency_df[
        section3b_prediction_consistency_df[
            "strategy_a"
        ] == "R0_CURRENT_BASELINE"
    ]
    .set_index("strategy_b")
)

# ------------------------------------------------------------------------------------------
# Build evidence matrix
# ------------------------------------------------------------------------------------------

evidence_records = []

for strategy in EXPECTED_FEATURE_COUNTS.keys():

    perf = performance_lookup.loc[strategy]
    repro = reproduction_lookup.loc[strategy]

    if strategy == "R0_CURRENT_BASELINE":

        agreement_with_r0 = 1.0
        mean_confidence_difference_vs_r0 = 0.0
        maximum_confidence_difference_vs_r0 = 0.0

    else:

        pair_row = r0_pair_lookup.loc[strategy]

        agreement_with_r0 = float(
            pair_row["agreement_rate"]
        )

        mean_confidence_difference_vs_r0 = float(
            pair_row[
                "mean_confidence_difference"
            ]
        )

        maximum_confidence_difference_vs_r0 = float(
            pair_row[
                "maximum_confidence_difference"
            ]
        )

    evidence_records.append(
        {
            "strategy": strategy,

            # Complexity
            "feature_count": int(
                perf["feature_count"]
            ),

            "features_removed_vs_r0": int(
                perf[
                    "features_removed_vs_r0"
                ]
            ),

            "feature_reduction_percent_vs_r0": float(
                perf[
                    "feature_reduction_percent_vs_r0"
                ]
            ),

            # Held-out metrics
            "validation_accuracy": float(
                perf["validation_accuracy"]
            ),

            "test_accuracy": float(
                perf["test_accuracy"]
            ),

            "test_balanced_accuracy": float(
                perf["test_balanced_accuracy"]
            ),

            "test_macro_f1": float(
                perf["test_macro_f1"]
            ),

            # Reproducibility
            "predictions_reproduced": bool(
                repro["prediction_matches"]
            ),

            "probabilities_reproduced": bool(
                repro["probabilities_match"]
            ),

            "maximum_reproduction_error": float(
                repro[
                    "maximum_probability_difference"
                ]
            ),

            # Decision stability
            "agreement_with_r0": (
                agreement_with_r0
            ),

            # Confidence behavior
            "mean_confidence_difference_vs_r0": (
                mean_confidence_difference_vs_r0
            ),

            "maximum_confidence_difference_vs_r0": (
                maximum_confidence_difference_vs_r0
            ),
        }
    )

section4c_final_evidence_matrix_df = pd.DataFrame(
    evidence_records
)

# ------------------------------------------------------------------------------------------
# Add validation flags
# ------------------------------------------------------------------------------------------

section4c_final_evidence_matrix_df[
    "held_out_performance_preserved"
] = (
    (
        section4c_final_evidence_matrix_df[
            "validation_accuracy"
        ] == 1.0
    )
    &
    (
        section4c_final_evidence_matrix_df[
            "test_accuracy"
        ] == 1.0
    )
    &
    (
        section4c_final_evidence_matrix_df[
            "test_balanced_accuracy"
        ] == 1.0
    )
    &
    (
        section4c_final_evidence_matrix_df[
            "test_macro_f1"
        ] == 1.0
    )
)

section4c_final_evidence_matrix_df[
    "reproducibility_passed"
] = (
    section4c_final_evidence_matrix_df[
        "predictions_reproduced"
    ]
    &
    section4c_final_evidence_matrix_df[
        "probabilities_reproduced"
    ]
)

section4c_final_evidence_matrix_df[
    "decision_stability_passed"
] = (
    section4c_final_evidence_matrix_df[
        "agreement_with_r0"
    ] == 1.0
)

# ------------------------------------------------------------------------------------------
# Production candidate rule
#
# IMPORTANT:
# This is not a claim of tournament superiority.
# It selects the smallest policy that:
#   1. preserves stored held-out performance,
#   2. reproduces exactly,
#   3. preserves R0 decisions.
# ------------------------------------------------------------------------------------------

eligible_mask = (
    section4c_final_evidence_matrix_df[
        "held_out_performance_preserved"
    ]
    &
    section4c_final_evidence_matrix_df[
        "reproducibility_passed"
    ]
    &
    section4c_final_evidence_matrix_df[
        "decision_stability_passed"
    ]
)

section4c_final_evidence_matrix_df[
    "production_gate_eligible"
] = eligible_mask

eligible_candidates = (
    section4c_final_evidence_matrix_df[
        eligible_mask
    ]
    .sort_values(
        by=[
            "feature_count",
            "maximum_confidence_difference_vs_r0",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(drop=True)
)

assert not eligible_candidates.empty, (
    "No frozen policy passed the Notebook 58 production gate."
)

primary_candidate = (
    eligible_candidates
    .iloc[0]["strategy"]
)

section4c_final_evidence_matrix_df[
    "primary_candidate"
] = (
    section4c_final_evidence_matrix_df[
        "strategy"
    ] == primary_candidate
)

# ------------------------------------------------------------------------------------------
# Display evidence
# ------------------------------------------------------------------------------------------

print("\nFINAL EVIDENCE MATRIX")
print("-" * 100)

display(
    section4c_final_evidence_matrix_df
)

print("\nPRODUCTION-GATE ELIGIBLE POLICIES")
print("-" * 100)

display(
    eligible_candidates
)

print("\nPRIMARY CANDIDATE")
print("-" * 100)

print(primary_candidate)

# ------------------------------------------------------------------------------------------
# R3-focused interpretation
# ------------------------------------------------------------------------------------------

r3_evidence = (
    section4c_final_evidence_matrix_df[
        section4c_final_evidence_matrix_df[
            "strategy"
        ] == "R3_STATE_CENTRIC_POLICY"
    ]
    .iloc[0]
)

print("\nR3 EVIDENCE SUMMARY")
print("-" * 100)

print(
    f"Feature count: "
    f"{int(r3_evidence['feature_count'])}"
)

print(
    f"Feature reduction vs R0: "
    f"{r3_evidence['feature_reduction_percent_vs_r0']:.3f}%"
)

print(
    f"Validation accuracy: "
    f"{r3_evidence['validation_accuracy']:.4f}"
)

print(
    f"Test accuracy: "
    f"{r3_evidence['test_accuracy']:.4f}"
)

print(
    f"Agreement with R0: "
    f"{r3_evidence['agreement_with_r0']:.4f}"
)

print(
    f"Maximum reproduction error: "
    f"{r3_evidence['maximum_reproduction_error']:.12f}"
)

print(
    f"Mean confidence difference vs R0: "
    f"{r3_evidence['mean_confidence_difference_vs_r0']:.6f}"
)

print(
    f"Maximum confidence difference vs R0: "
    f"{r3_evidence['maximum_confidence_difference_vs_r0']:.6f}"
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert section4c_final_evidence_matrix_df[
    "held_out_performance_preserved"
].all()

assert section4c_final_evidence_matrix_df[
    "reproducibility_passed"
].all()

assert section4c_final_evidence_matrix_df[
    "decision_stability_passed"
].all()

assert primary_candidate == (
    "R3_STATE_CENTRIC_POLICY"
), (
    "Expected R3_STATE_CENTRIC_POLICY "
    "to be the smallest eligible production candidate."
)

print("\nINTERPRETATION")
print("-" * 100)

print(
    "All four frozen policies preserve the stored "
    "held-out classification performance."
)

print(
    "All four policies reproduce their Notebook 56 "
    "predictions and probabilities exactly."
)

print(
    "All four policies preserve the same action decisions "
    "on all 184 controlled evaluation states."
)

print(
    "R3 is the smallest eligible policy, using 46 features "
    "instead of 64 while preserving the validated decision behavior."
)

print(
    "Confidence differences remain present and will be treated "
    "as a robustness / deployment consideration rather than "
    "as evidence of action-level failure."
)

print("\n✅ FINAL EVIDENCE MATRIX COMPLETE")
print("✅ R3 PASSES THE CURRENT PRODUCTION GATE")
print("✅ SECTION 4C COMPLETE")


# In[16]:


# ==========================================================================================
# SECTION 5A — INDEPENDENT RISK ASSESSMENT
# CORRECTED FLOATING-POINT BOUNDARY HANDLING
# ==========================================================================================

print("=" * 100)
print("SECTION 5A — INDEPENDENT RISK ASSESSMENT")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Risk thresholds
# ------------------------------------------------------------------------------------------

FEATURE_REDUCTION_LOW_THRESHOLD = 30.0
CONFIDENCE_DIFFERENCE_LOW_THRESHOLD = 0.15

# Numerical tolerance for boundary comparisons
NUMERICAL_TOLERANCE = 1e-12

risk_records = []

for _, row in section4c_final_evidence_matrix_df.iterrows():

    # ------------------------------------------------------------------
    # Feature complexity risk
    # ------------------------------------------------------------------

    feature_reduction = float(
        row["feature_reduction_percent_vs_r0"]
    )

    feature_risk = (
        "LOW"
        if feature_reduction
        < FEATURE_REDUCTION_LOW_THRESHOLD
        else "MEDIUM"
    )

    # ------------------------------------------------------------------
    # Reproduction risk
    # ------------------------------------------------------------------

    reproduction_error = float(
        row["maximum_reproduction_error"]
    )

    reproduction_risk = (
        "LOW"
        if np.isclose(
            reproduction_error,
            0.0,
            atol=NUMERICAL_TOLERANCE,
            rtol=0.0,
        )
        else "HIGH"
    )

    # ------------------------------------------------------------------
    # Decision risk
    # ------------------------------------------------------------------

    agreement_with_r0 = float(
        row["agreement_with_r0"]
    )

    decision_risk = (
        "LOW"
        if np.isclose(
            agreement_with_r0,
            1.0,
            atol=NUMERICAL_TOLERANCE,
            rtol=0.0,
        )
        else "HIGH"
    )

    # ------------------------------------------------------------------
    # Confidence risk
    # ------------------------------------------------------------------

    maximum_confidence_difference = abs(
        float(
            row[
                "maximum_confidence_difference_vs_r0"
            ]
        )
    )

    confidence_within_threshold = (
        maximum_confidence_difference
        <= (
            CONFIDENCE_DIFFERENCE_LOW_THRESHOLD
            + NUMERICAL_TOLERANCE
        )
    )

    confidence_risk = (
        "LOW"
        if confidence_within_threshold
        else "MEDIUM"
    )

    # ------------------------------------------------------------------
    # Overall controlled-validation risk
    # ------------------------------------------------------------------

    if (
        reproduction_risk == "LOW"
        and decision_risk == "LOW"
        and confidence_risk == "LOW"
    ):
        overall_risk = "LOW"

    elif (
        reproduction_risk == "HIGH"
        or decision_risk == "HIGH"
    ):
        overall_risk = "HIGH"

    else:
        overall_risk = "MEDIUM"

    risk_records.append(
        {
            "strategy": row["strategy"],

            "feature_reduction_percent": (
                feature_reduction
            ),

            "maximum_reproduction_error": (
                reproduction_error
            ),

            "agreement_with_r0": (
                agreement_with_r0
            ),

            "maximum_confidence_difference_vs_r0": (
                maximum_confidence_difference
            ),

            "feature_risk": feature_risk,

            "reproduction_risk": (
                reproduction_risk
            ),

            "decision_risk": decision_risk,

            "confidence_risk": confidence_risk,

            "overall_controlled_validation_risk": (
                overall_risk
            ),
        }
    )

section5a_risk_assessment_df = pd.DataFrame(
    risk_records
)

# ------------------------------------------------------------------------------------------
# Display
# ------------------------------------------------------------------------------------------

print("\nRISK ASSESSMENT")
print("-" * 100)

display(
    section5a_risk_assessment_df
)

# ------------------------------------------------------------------------------------------
# R3 diagnostic
# ------------------------------------------------------------------------------------------

r3_risk = (
    section5a_risk_assessment_df[
        section5a_risk_assessment_df[
            "strategy"
        ] == "R3_STATE_CENTRIC_POLICY"
    ]
    .iloc[0]
)

print("\nR3 RISK DIAGNOSTIC")
print("-" * 100)

print(
    "Maximum confidence difference: "
    f"{r3_risk['maximum_confidence_difference_vs_r0']:.15f}"
)

print(
    "Low-risk threshold: "
    f"{CONFIDENCE_DIFFERENCE_LOW_THRESHOLD:.15f}"
)

print(
    "Numerical tolerance: "
    f"{NUMERICAL_TOLERANCE:.1e}"
)

print(
    "Confidence risk: "
    f"{r3_risk['confidence_risk']}"
)

print(
    "Overall controlled-validation risk: "
    f"{r3_risk['overall_controlled_validation_risk']}"
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert (
    section5a_risk_assessment_df[
        "reproduction_risk"
    ] == "LOW"
).all(), (
    "At least one policy has reproduction risk."
)

assert (
    section5a_risk_assessment_df[
        "decision_risk"
    ] == "LOW"
).all(), (
    "At least one policy has decision-level risk."
)

assert (
    section5a_risk_assessment_df[
        "overall_controlled_validation_risk"
    ] == "LOW"
).all(), (
    "At least one policy exceeds the accepted "
    "controlled-validation risk threshold."
)

# ------------------------------------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------------------------------------

print("\nINTERPRETATION")
print("-" * 100)

print(
    "All four frozen policies are LOW risk under "
    "the Notebook 58 controlled-validation protocol."
)

print(
    "R3's observed maximum confidence difference is "
    "at the accepted 0.150 boundary and is treated "
    "using explicit floating-point tolerance."
)

print(
    "R3 has zero reproduction error, 100% action agreement "
    "with R0, and preserves all stored held-out metrics."
)

print(
    "This assessment applies to controlled validation only; "
    "full-game competitive performance still requires "
    "live/tournament evaluation."
)

print("\n✅ CONTROLLED-VALIDATION RISK ASSESSMENT PASSED")
print("✅ R3 REMAINS WITHIN THE ACCEPTED CONFIDENCE BOUNDARY")
print("✅ SECTION 5A COMPLETE")


# In[18]:


# ==========================================================================================
# SECTION 5B — PRODUCTION RELEASE DECISION
# ==========================================================================================

print("=" * 100)
print("SECTION 5B — PRODUCTION RELEASE DECISION")
print("=" * 100)

release_records = []

for _, row in section4c_final_evidence_matrix_df.iterrows():

    strategy = row["strategy"]

    production_ready = (
        row["validation_accuracy"] == 1.0
        and row["test_accuracy"] == 1.0
        and row["test_balanced_accuracy"] == 1.0
        and row["test_macro_f1"] == 1.0
        and bool(row["predictions_reproduced"])
        and bool(row["probabilities_reproduced"])
        and np.isclose(
            row["maximum_reproduction_error"],
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
        and np.isclose(
            row["agreement_with_r0"],
            1.0,
            atol=1e-12,
            rtol=0.0,
        )
    )

    release_records.append(
        {
            "strategy": strategy,
            "feature_count": int(
                row["feature_count"]
            ),
            "feature_reduction_percent": float(
                row["feature_reduction_percent_vs_r0"]
            ),
            "production_ready": production_ready,
            "release_status": (
                "GO"
                if production_ready
                else "NO-GO"
            ),
        }
    )

section5b_release_df = pd.DataFrame(
    release_records
)

print("\nPRODUCTION RELEASE MATRIX")
print("-" * 100)

display(
    section5b_release_df
)

assert section5b_release_df[
    "production_ready"
].all(), (
    "At least one frozen policy failed the production-readiness gate."
)

print("\n✅ ALL FOUR FROZEN POLICIES PASS THE CURRENT RELEASE GATE")
print("✅ SECTION 5B RELEASE MATRIX COMPLETE")


# In[19]:


# ==========================================================================================
# SECTION 5B.1 — SELECT PRIMARY PRODUCTION CANDIDATE
# ==========================================================================================

eligible = (
    section5b_release_df[
        section5b_release_df["production_ready"]
    ]
    .sort_values(
        by=[
            "feature_count",
            "feature_reduction_percent",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

assert not eligible.empty, (
    "No production-ready policy is available."
)

production_candidate = eligible.iloc[0]

print("=" * 100)
print("PRIMARY PRODUCTION POLICY")
print("=" * 100)

print(
    f"\nStrategy: "
    f"{production_candidate['strategy']}"
)

print(
    f"Feature count: "
    f"{int(production_candidate['feature_count'])}"
)

print(
    f"Feature reduction vs R0: "
    f"{production_candidate['feature_reduction_percent']:.3f}%"
)

print(
    f"Release status: "
    f"{production_candidate['release_status']}"
)

assert (
    production_candidate["strategy"]
    == "R3_STATE_CENTRIC_POLICY"
), (
    "Expected R3_STATE_CENTRIC_POLICY "
    "to be selected as the smallest eligible policy."
)

print("\n✅ R3 SELECTED AS PRIMARY PRODUCTION CANDIDATE")


# In[20]:


# ==========================================================================================
# SECTION 5B.2 — FINAL PRODUCTION GATE CHECKLIST
# ==========================================================================================

print("=" * 100)
print("SECTION 5B.2 — FINAL PRODUCTION GATE CHECKLIST")
print("=" * 100)

r3_row = (
    section4c_final_evidence_matrix_df[
        section4c_final_evidence_matrix_df["strategy"]
        == "R3_STATE_CENTRIC_POLICY"
    ]
    .iloc[0]
)

r3_risk_row = (
    section5a_risk_assessment_df[
        section5a_risk_assessment_df["strategy"]
        == "R3_STATE_CENTRIC_POLICY"
    ]
    .iloc[0]
)

gate_checklist = pd.DataFrame(
    [
        {
            "Gate": "Frozen model loaded",
            "Passed": True,
        },
        {
            "Gate": "46 retained features",
            "Passed": int(r3_row["feature_count"]) == 46,
        },
        {
            "Gate": "Validation accuracy preserved",
            "Passed": np.isclose(
                r3_row["validation_accuracy"],
                1.0,
                atol=1e-12,
            ),
        },
        {
            "Gate": "Test accuracy preserved",
            "Passed": np.isclose(
                r3_row["test_accuracy"],
                1.0,
                atol=1e-12,
            ),
        },
        {
            "Gate": "Predictions reproduced",
            "Passed": bool(
                r3_row["predictions_reproduced"]
            ),
        },
        {
            "Gate": "Probabilities reproduced",
            "Passed": bool(
                r3_row["probabilities_reproduced"]
            ),
        },
        {
            "Gate": "Zero reproduction error",
            "Passed": np.isclose(
                r3_row["maximum_reproduction_error"],
                0.0,
                atol=1e-12,
            ),
        },
        {
            "Gate": "100% agreement with R0",
            "Passed": np.isclose(
                r3_row["agreement_with_r0"],
                1.0,
                atol=1e-12,
            ),
        },
        {
            "Gate": "Controlled-validation risk LOW",
            "Passed": (
                r3_risk_row[
                    "overall_controlled_validation_risk"
                ]
                == "LOW"
            ),
        },
        {
            "Gate": "Release decision GO",
            "Passed": (
                production_candidate["release_status"]
                == "GO"
            ),
        },
    ]
)

print()
print("FINAL PRODUCTION GATE CHECKLIST")
print("-" * 100)

display(gate_checklist)

assert gate_checklist["Passed"].all(), (
    "One or more production gates failed."
)

FINAL_RELEASE_DECISION = "GO"
FINAL_PRODUCTION_POLICY = (
    "R3_STATE_CENTRIC_POLICY"
)

print()
print("=" * 100)
print("FINAL RELEASE DECISION")
print("=" * 100)

print(f"Decision           : {FINAL_RELEASE_DECISION}")
print(f"Production Policy  : {FINAL_PRODUCTION_POLICY}")
print("Deployment Status  : APPROVED FOR LIVE COMPETITIVE VALIDATION")
print("Notebook Status    : COMPLETE")

print()
print("Interpretation")
print("-" * 100)

print(
    "R3 preserves all held-out performance while reducing "
    "the feature space from 64 to 46 features."
)

print(
    "Controlled validation demonstrates zero prediction "
    "reproduction error and complete action agreement."
)

print(
    "Notebook 59 will evaluate real competitive performance "
    "rather than controlled validation."
)

print()
print("✅ ALL FINAL PRODUCTION GATES PASSED")
print("✅ R3 APPROVED FOR LIVE COMPETITIVE EVALUATION")
print("✅ NOTEBOOK 58 VALIDATION COMPLETE")


# In[21]:


# ==========================================================================================
# SECTION 6 — EXPORT NOTEBOOK 58 RESULTS
# ==========================================================================================

from pathlib import Path
import pickle
import json
import pandas as pd

print("=" * 100)
print("SECTION 6 — EXPORT NOTEBOOK 58 RESULTS")
print("=" * 100)

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "notebook58"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "notebook58"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================================================
# REPORTS
# ==========================================================================================

section4c_final_evidence_matrix_df.to_csv(
    REPORT_DIR / "notebook58_validation_report.csv",
    index=False,
)

section5b_release_df.to_csv(
    REPORT_DIR / "notebook58_release_summary.csv",
    index=False,
)

gate_checklist.to_csv(
    REPORT_DIR / "notebook58_gate_checklist.csv",
    index=False,
)

section5a_risk_assessment_df.to_csv(
    REPORT_DIR / "notebook58_risk_assessment.csv",
    index=False,
)

# ==========================================================================================
# CONSOLIDATED MACHINE-READABLE ARTIFACT
# ==========================================================================================

notebook58_results = {
    "release_matrix": section5b_release_df,
    "production_candidate": production_candidate.to_dict(),
    "gate_checklist": gate_checklist,
    "risk_assessment": section5a_risk_assessment_df,
    "final_evidence_matrix": section4c_final_evidence_matrix_df,
    "prediction_reproduction": section3a_prediction_reproduction_df,
    "prediction_consistency": section3b_prediction_consistency_df,
    "decision_divergence": section3c_decision_divergence_df,
    "confidence_divergence": section3c_confidence_divergence_df,
    "performance_leaderboard": section4b_performance_leaderboard_df,
    "release_decision": FINAL_RELEASE_DECISION,
    "production_policy": FINAL_PRODUCTION_POLICY,
}

with open(
    ARTIFACT_DIR / "notebook58_final_competitive_validation_results.pkl",
    "wb",
) as file:
    pickle.dump(notebook58_results, file)

# ==========================================================================================
# JSON SUMMARY
# ==========================================================================================

notebook58_summary = {
    "notebook": 58,
    "status": "FINAL_COMPETITIVE_VALIDATION_COMPLETE",
    "release_decision": FINAL_RELEASE_DECISION,
    "production_policy": FINAL_PRODUCTION_POLICY,
    "feature_count": 46,
    "feature_reduction_vs_r0_percent": 28.125,
    "controlled_validation_risk": "LOW",
    "action_agreement_with_r0": 1.0,
    "maximum_reproduction_error": 0.0,
    "ready_for_next_stage": True,
}

with open(
    ARTIFACT_DIR / "notebook58_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        notebook58_summary,
        file,
        indent=4,
    )

print("\nREPORT DIRECTORY")
print(REPORT_DIR)

print("\nARTIFACT DIRECTORY")
print(ARTIFACT_DIR)

print("\n✅ NOTEBOOK 58 REPORTS EXPORTED")
print("✅ NOTEBOOK 58 ARTIFACT PACKAGE EXPORTED")
print("✅ NOTEBOOK 58 JSON SUMMARY EXPORTED")
print("✅ SECTION 6 COMPLETE")


# In[ ]:




