#!/usr/bin/env python
# coding: utf-8

# # Notebook 55 — Controlled Feature-Ablation Retraining
# 
# ## Pokémon TCG AI Battle Challenge
# 
# ### Objective
# 
# Retrain and compare controlled policy variants designed to reduce excessive dependence on legal-move-signature and action-identity features while preserving:
# 
# - hard legality enforcement,
# - mirrored side neutrality,
# - validation performance,
# - battle performance,
# - reproducibility,
# - and the original frozen Notebook 53 model.
# 
# ### Experimental strategies
# 
# - **R0_CURRENT_BASELINE** — reproduce the current 64-feature model.
# - **R1_REMOVE_LEGAL_SIGNATURE** — remove encoded legal-move-signature features.
# - **R2_REMOVE_SIGNATURE_AND_COMPATIBILITY** — remove legal signatures and card-action compatibility features.
# - **R3_STATE_CENTRIC_POLICY** — retain primarily state, resource, card, turn, and aggregate legality features.
# 
# ### Primary success criteria
# 
# - zero illegal selected actions,
# - all 184 expanded evaluation cases completed,
# - all 92 mirrored pairs completed,
# - reduced Quick Attack selection concentration,
# - at least one legitimate Ascension selection,
# - increased move entropy,
# - no material loss in validation accuracy,
# - no material loss in battle performance.

# In[1]:


# ======================================================================================
# NOTEBOOK 55 — CONTROLLED FEATURE-ABLATION RETRAINING
# SECTION 1A — NOTEBOOK IDENTITY AND PROJECT PATHS
# ======================================================================================

print("=" * 100)
print("NOTEBOOK 55 — CONTROLLED FEATURE-ABLATION RETRAINING")
print("=" * 100)

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOKS_DIRECTORY = (
    PROJECT_ROOT
    / "notebooks"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook55"
)

ARTIFACTS_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook55"
)

MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "notebook55"
)

NOTEBOOK54_SECTION4_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook54"
    / "section4"
)

NOTEBOOK53_PATH = (
    NOTEBOOKS_DIRECTORY
    / "53_legality_aware_policy_optimization.ipynb"
)

NOTEBOOK54_PATH = (
    NOTEBOOKS_DIRECTORY
    / "54_tournament_strength_optimization_clean.ipynb"
)

NOTEBOOK55_PATH = (
    NOTEBOOKS_DIRECTORY
    / "55_controlled_feature_ablation_retraining.ipynb"
)


for directory_path in [
    REPORTS_DIRECTORY,
    ARTIFACTS_DIRECTORY,
    MODELS_DIRECTORY,
]:

    directory_path.mkdir(
        parents=True,
        exist_ok=True,
    )


print()
print("PROJECT PATHS")
print("-" * 100)

for path_name, path_value in [
    ("PROJECT_ROOT", PROJECT_ROOT),
    ("NOTEBOOKS_DIRECTORY", NOTEBOOKS_DIRECTORY),
    ("REPORTS_DIRECTORY", REPORTS_DIRECTORY),
    ("ARTIFACTS_DIRECTORY", ARTIFACTS_DIRECTORY),
    ("MODELS_DIRECTORY", MODELS_DIRECTORY),
    (
        "NOTEBOOK54_SECTION4_REPORT_DIRECTORY",
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY,
    ),
    ("NOTEBOOK53_PATH", NOTEBOOK53_PATH),
    ("NOTEBOOK54_PATH", NOTEBOOK54_PATH),
    ("NOTEBOOK55_PATH", NOTEBOOK55_PATH),
]:

    print(
        f"{path_name:46}: {path_value}"
    )


assert PROJECT_ROOT.exists()
assert NOTEBOOKS_DIRECTORY.exists()
assert REPORTS_DIRECTORY.exists()
assert ARTIFACTS_DIRECTORY.exists()
assert MODELS_DIRECTORY.exists()
assert NOTEBOOK54_SECTION4_REPORT_DIRECTORY.exists()
assert NOTEBOOK53_PATH.exists()
assert NOTEBOOK54_PATH.exists()


print()
print("✅ NOTEBOOK 55 PROJECT PATH SETUP PASSED")


# In[2]:


# ======================================================================================
# SECTION 1B — LOAD NOTEBOOK 54 REMEDIATION DESIGN EVIDENCE
# ======================================================================================

print("=" * 100)
print("SECTION 1B — LOAD NOTEBOOK 54 REMEDIATION DESIGN EVIDENCE")
print("=" * 100)


NOTEBOOK54_REQUIRED_REPORTS = {
    "remediation_summary":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_remediation_design_summary.json",

    "feature_family_summary":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_feature_family_summary.csv",

    "remediation_catalog":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_remediation_strategy_catalog.csv",

    "strategy_feature_matrix":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_strategy_feature_matrix.csv",

    "strategy_feature_impact":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_strategy_feature_impact.csv",

    "experiment_contract":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_retraining_experiment_contract.csv",

    "acceptance_criteria":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_acceptance_criteria.csv",

    "execution_priority":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whf_remediation_execution_priority.csv",
}


notebook54_report_validation_rows = []


for report_name, report_path in (
    NOTEBOOK54_REQUIRED_REPORTS.items()
):

    notebook54_report_validation_rows.append(
        {
            "report_name":
                report_name,

            "report_path":
                str(
                    report_path
                ),

            "exists":
                report_path.exists(),

            "size_bytes":
                (
                    int(
                        report_path.stat().st_size
                    )
                    if report_path.exists()
                    else 0
                ),
        }
    )


notebook54_report_validation_df = pd.DataFrame(
    notebook54_report_validation_rows
)


print()
print("NOTEBOOK 54 REMEDIATION REPORT VALIDATION")
print("-" * 100)

display(
    notebook54_report_validation_df
)


assert notebook54_report_validation_df[
    "exists"
].astype(bool).all()


assert notebook54_report_validation_df[
    "size_bytes"
].gt(0).all()


with open(
    NOTEBOOK54_REQUIRED_REPORTS[
        "remediation_summary"
    ],
    "r",
    encoding="utf-8",
) as file:

    notebook54_remediation_summary = json.load(
        file
    )


notebook54_feature_family_summary_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "feature_family_summary"
    ]
)

notebook54_remediation_catalog_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "remediation_catalog"
    ]
)

notebook54_strategy_feature_matrix_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "strategy_feature_matrix"
    ]
)

notebook54_strategy_feature_impact_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "strategy_feature_impact"
    ]
)

notebook54_experiment_contract_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "experiment_contract"
    ]
)

notebook54_acceptance_criteria_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "acceptance_criteria"
    ]
)

notebook54_execution_priority_df = pd.read_csv(
    NOTEBOOK54_REQUIRED_REPORTS[
        "execution_priority"
    ]
)


print()
print("NOTEBOOK 54 REMEDIATION SUMMARY")
print("-" * 100)

for key, value in (
    notebook54_remediation_summary.items()
):

    print(
        f"{key:70}: {value}"
    )


assert (
    notebook54_remediation_summary[
        "next_stage"
    ]
    ==
    "CONTROLLED_FEATURE_ABLATION_RETRAINING_EXPERIMENT"
)


assert (
    notebook54_remediation_summary[
        "primary_recommendation"
    ]
    ==
    "R1_REMOVE_LEGAL_SIGNATURE"
)


assert (
    notebook54_remediation_summary[
        "legal_filter_preserved"
    ]
    is True
)


assert (
    notebook54_remediation_summary[
        "frozen_original_model_modified"
    ]
    is False
)


print()
print("✅ NOTEBOOK 54 REMEDIATION EVIDENCE LOADED")


# In[3]:


# ======================================================================================
# SECTION 1C — FREEZE CONTROLLED RETRAINING EXPERIMENT CONFIGURATION
# ======================================================================================

print("=" * 100)
print("SECTION 1C — FREEZE CONTROLLED RETRAINING EXPERIMENT CONFIGURATION")
print("=" * 100)

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn


# --------------------------------------------------------------------------------------
# 1. Reproducibility configuration
# --------------------------------------------------------------------------------------

NOTEBOOK55_RANDOM_SEED = 42
NOTEBOOK55_TEST_SIZE = 0.20

np.random.seed(
    NOTEBOOK55_RANDOM_SEED
)


NOTEBOOK55_PRIMARY_STRATEGIES = [
    "R0_CURRENT_BASELINE",
    "R1_REMOVE_LEGAL_SIGNATURE",
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
    "R3_STATE_CENTRIC_POLICY",
]


NOTEBOOK55_STRATEGY_ORDER = {
    strategy_id:
        strategy_index

    for strategy_index, strategy_id
    in enumerate(
        NOTEBOOK55_PRIMARY_STRATEGIES
    )
}


# --------------------------------------------------------------------------------------
# 2. Define immutable experiment requirements
# --------------------------------------------------------------------------------------

NOTEBOOK55_EXPERIMENT_CONTRACT = {
    "notebook_id":
        55,

    "experiment_name":
        "CONTROLLED_FEATURE_ABLATION_RETRAINING",

    "random_seed":
        NOTEBOOK55_RANDOM_SEED,

    "test_size":
        NOTEBOOK55_TEST_SIZE,

    "primary_strategies":
        NOTEBOOK55_PRIMARY_STRATEGIES,

    "baseline_strategy":
        "R0_CURRENT_BASELINE",

    "primary_remediation_strategy":
        "R1_REMOVE_LEGAL_SIGNATURE",

    "secondary_remediation_strategy":
        "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",

    "aggressive_remediation_strategy":
        "R3_STATE_CENTRIC_POLICY",

    "fixed_training_rows":
        True,

    "fixed_validation_rows":
        True,

    "fixed_random_seed":
        True,

    "fixed_estimator_parameters":
        True,

    "external_hard_legality_filter_preserved":
        True,

    "original_notebook53_model_modified":
        False,

    "original_notebook54_reports_modified":
        False,

    "required_expanded_cases":
        184,

    "required_mirror_pairs":
        92,

    "maximum_allowed_validation_accuracy_drop":
        0.02,

    "maximum_allowed_battle_win_rate_drop":
        0.02,

    "minimum_required_normalized_move_entropy":
        0.10,

    "minimum_required_ascension_selections":
        1,

    "maximum_accepted_quick_attack_share":
        0.90,

    "minimum_probability_margin_reduction":
        0.05,
}


# --------------------------------------------------------------------------------------
# 3. Validate agreement with Notebook 54
# --------------------------------------------------------------------------------------

assert (
    notebook54_remediation_summary[
        "primary_recommendation"
    ]
    ==
    NOTEBOOK55_EXPERIMENT_CONTRACT[
        "primary_remediation_strategy"
    ]
)


assert (
    notebook54_remediation_summary[
        "secondary_recommendation"
    ]
    ==
    NOTEBOOK55_EXPERIMENT_CONTRACT[
        "secondary_remediation_strategy"
    ]
)


assert (
    notebook54_remediation_summary[
        "legal_filter_preserved"
    ]
    is True
)


assert (
    notebook54_remediation_summary[
        "frozen_original_model_modified"
    ]
    is False
)


assert (
    notebook54_remediation_summary[
        "next_stage"
    ]
    ==
    "CONTROLLED_FEATURE_ABLATION_RETRAINING_EXPERIMENT"
)


# --------------------------------------------------------------------------------------
# 4. Validate required strategies exist in the Notebook 54 catalog
# --------------------------------------------------------------------------------------

available_strategy_ids_55 = set(
    notebook54_remediation_catalog_df[
        "strategy_id"
    ]
    .astype(str)
    .tolist()
)


missing_primary_strategies_55 = [
    strategy_id
    for strategy_id in NOTEBOOK55_PRIMARY_STRATEGIES
    if strategy_id not in available_strategy_ids_55
]


assert not missing_primary_strategies_55, (
    "Required Notebook 55 strategies are missing from the Notebook 54 catalog: "
    f"{missing_primary_strategies_55}"
)


notebook55_primary_strategy_catalog_df = (
    notebook54_remediation_catalog_df.loc[
        notebook54_remediation_catalog_df[
            "strategy_id"
        ].isin(
            NOTEBOOK55_PRIMARY_STRATEGIES
        )
    ]
    .copy()
)


notebook55_primary_strategy_catalog_df[
    "execution_order"
] = (
    notebook55_primary_strategy_catalog_df[
        "strategy_id"
    ]
    .map(
        NOTEBOOK55_STRATEGY_ORDER
    )
)


notebook55_primary_strategy_catalog_df = (
    notebook55_primary_strategy_catalog_df
    .sort_values(
        "execution_order"
    )
    .reset_index(drop=True)
)


print()
print("PRIMARY RETRAINING STRATEGIES")
print("-" * 100)

display(
    notebook55_primary_strategy_catalog_df
)


assert len(
    notebook55_primary_strategy_catalog_df
) == 4


# --------------------------------------------------------------------------------------
# 5. Capture runtime environment
# --------------------------------------------------------------------------------------

notebook55_runtime_environment = {
    "captured_at":
        datetime.now().isoformat(
            timespec="seconds"
        ),

    "python_version":
        sys.version,

    "platform":
        platform.platform(),

    "numpy_version":
        np.__version__,

    "pandas_version":
        pd.__version__,

    "scikit_learn_version":
        sklearn.__version__,

    "project_root":
        str(
            PROJECT_ROOT
        ),

    "reports_directory":
        str(
            REPORTS_DIRECTORY
        ),

    "artifacts_directory":
        str(
            ARTIFACTS_DIRECTORY
        ),

    "models_directory":
        str(
            MODELS_DIRECTORY
        ),
}


print()
print("RUNTIME ENVIRONMENT")
print("-" * 100)

for key, value in (
    notebook55_runtime_environment.items()
):

    print(
        f"{key:34}: {value}"
    )


# --------------------------------------------------------------------------------------
# 6. Create configuration summary dataframe
# --------------------------------------------------------------------------------------

notebook55_experiment_configuration_df = pd.DataFrame(
    [
        {
            "configuration_item":
                key,

            "configuration_value":
                (
                    json.dumps(
                        value
                    )
                    if isinstance(
                        value,
                        (
                            list,
                            dict,
                            tuple,
                        ),
                    )
                    else value
                ),
        }
        for key, value
        in NOTEBOOK55_EXPERIMENT_CONTRACT.items()
    ]
)


print()
print("FROZEN EXPERIMENT CONFIGURATION")
print("-" * 100)

display(
    notebook55_experiment_configuration_df
)


# --------------------------------------------------------------------------------------
# 7. Validation checks
# --------------------------------------------------------------------------------------

notebook55_section1c_validation_df = pd.DataFrame(
    [
        {
            "check":
                "four_primary_strategies_defined",

            "passed":
                len(
                    NOTEBOOK55_PRIMARY_STRATEGIES
                ) == 4,

            "value":
                len(
                    NOTEBOOK55_PRIMARY_STRATEGIES
                ),

            "expected":
                4,
        },
        {
            "check":
                "primary_recommendation_matches_notebook54",

            "passed":
                (
                    NOTEBOOK55_EXPERIMENT_CONTRACT[
                        "primary_remediation_strategy"
                    ]
                    ==
                    notebook54_remediation_summary[
                        "primary_recommendation"
                    ]
                ),

            "value":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "primary_remediation_strategy"
                ],

            "expected":
                notebook54_remediation_summary[
                    "primary_recommendation"
                ],
        },
        {
            "check":
                "hard_legality_filter_preserved",

            "passed":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "external_hard_legality_filter_preserved"
                ],

            "value":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "external_hard_legality_filter_preserved"
                ],

            "expected":
                True,
        },
        {
            "check":
                "original_model_frozen",

            "passed":
                not NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "original_notebook53_model_modified"
                ],

            "value":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "original_notebook53_model_modified"
                ],

            "expected":
                False,
        },
        {
            "check":
                "required_expanded_cases_frozen",

            "passed":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "required_expanded_cases"
                ] == 184,

            "value":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "required_expanded_cases"
                ],

            "expected":
                184,
        },
        {
            "check":
                "required_mirror_pairs_frozen",

            "passed":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "required_mirror_pairs"
                ] == 92,

            "value":
                NOTEBOOK55_EXPERIMENT_CONTRACT[
                    "required_mirror_pairs"
                ],

            "expected":
                92,
        },
    ]
)


print()
print("SECTION 1C VALIDATION CHECKS")
print("-" * 100)

display(
    notebook55_section1c_validation_df
)


assert notebook55_section1c_validation_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 8. Save setup artifacts
# --------------------------------------------------------------------------------------

NOTEBOOK55_EXPERIMENT_CONFIG_FILE = (
    REPORTS_DIRECTORY
    /
    "section1c_experiment_configuration.csv"
)

NOTEBOOK55_PRIMARY_STRATEGY_FILE = (
    REPORTS_DIRECTORY
    /
    "section1c_primary_strategy_catalog.csv"
)

NOTEBOOK55_RUNTIME_ENVIRONMENT_FILE = (
    REPORTS_DIRECTORY
    /
    "section1c_runtime_environment.json"
)

NOTEBOOK55_EXPERIMENT_CONTRACT_FILE = (
    REPORTS_DIRECTORY
    /
    "section1c_experiment_contract.json"
)

NOTEBOOK55_SECTION1C_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section1c_validation_checks.csv"
)


notebook55_experiment_configuration_df.to_csv(
    NOTEBOOK55_EXPERIMENT_CONFIG_FILE,
    index=False,
)


notebook55_primary_strategy_catalog_df.to_csv(
    NOTEBOOK55_PRIMARY_STRATEGY_FILE,
    index=False,
)


notebook55_section1c_validation_df.to_csv(
    NOTEBOOK55_SECTION1C_VALIDATION_FILE,
    index=False,
)


with open(
    NOTEBOOK55_RUNTIME_ENVIRONMENT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook55_runtime_environment,
        file,
        indent=2,
        ensure_ascii=False,
    )


with open(
    NOTEBOOK55_EXPERIMENT_CONTRACT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        NOTEBOOK55_EXPERIMENT_CONTRACT,
        file,
        indent=2,
        ensure_ascii=False,
    )


notebook55_section1c_saved_files = [
    NOTEBOOK55_EXPERIMENT_CONFIG_FILE,
    NOTEBOOK55_PRIMARY_STRATEGY_FILE,
    NOTEBOOK55_RUNTIME_ENVIRONMENT_FILE,
    NOTEBOOK55_EXPERIMENT_CONTRACT_FILE,
    NOTEBOOK55_SECTION1C_VALIDATION_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in notebook55_section1c_saved_files
)


print()
print("SAVED SECTION 1C REPORTS")
print("-" * 100)

for file_path in notebook55_section1c_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 1C CONTROLLED RETRAINING "
    "EXPERIMENT CONFIGURATION FROZEN"
)


# In[4]:


# ======================================================================================
# SECTION 2A — DISCOVER NOTEBOOK 53 TRAINING RUNTIME AND ARTIFACT SOURCES
# ======================================================================================

print("=" * 100)
print("SECTION 2A — DISCOVER NOTEBOOK 53 TRAINING RUNTIME AND ARTIFACT SOURCES")
print("=" * 100)

from pathlib import Path
import ast
import json
import re

import nbformat
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate source notebook
# --------------------------------------------------------------------------------------

assert NOTEBOOK53_PATH.exists(), (
    f"Notebook 53 was not found: {NOTEBOOK53_PATH}"
)

print()
print("NOTEBOOK 53 SOURCE")
print("-" * 100)
print(NOTEBOOK53_PATH)


# --------------------------------------------------------------------------------------
# 2. Read Notebook 53 source
# --------------------------------------------------------------------------------------

with open(
    NOTEBOOK53_PATH,
    "r",
    encoding="utf-8",
) as file:

    notebook53_document = nbformat.read(
        file,
        as_version=4,
    )


notebook53_code_cells = [
    cell
    for cell in notebook53_document.cells
    if cell.cell_type == "code"
]


print()
print("NOTEBOOK 53 SOURCE PROFILE")
print("-" * 100)
print("Total cells:", len(notebook53_document.cells))
print("Code cells :", len(notebook53_code_cells))


assert len(notebook53_code_cells) > 0


# --------------------------------------------------------------------------------------
# 3. Search source assignments for important training objects
# --------------------------------------------------------------------------------------

NOTEBOOK55_SEARCH_TERMS = [
    "RandomForestClassifier",
    "train_test_split",
    "fit(",
    "predict_proba",
    "notebook53_legality_model",
    "notebook53_preprocessor",
    "legality_training_numeric_features",
    "legality_training_categorical_features",
    "PREPROCESS_RAW_FEATURES",
    "PREPROCESS_NUMERIC_FEATURES",
    "PREPROCESS_CATEGORICAL_FEATURES",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "training_df",
    "dataset",
    "target",
    "selected_move",
]


notebook53_source_match_rows = []


for cell_index, cell in enumerate(
    notebook53_code_cells
):

    source_text = str(
        cell.source
    )


    source_lines = source_text.splitlines()


    for line_number, source_line in enumerate(
        source_lines,
        start=1,
    ):

        matched_terms = [
            search_term
            for search_term in NOTEBOOK55_SEARCH_TERMS
            if search_term.lower()
            in source_line.lower()
        ]


        if matched_terms:

            notebook53_source_match_rows.append(
                {
                    "cell_index":
                        int(
                            cell_index
                        ),

                    "line_number":
                        int(
                            line_number
                        ),

                    "matched_terms":
                        " | ".join(
                            matched_terms
                        ),

                    "source_line":
                        source_line.strip(),
                }
            )


notebook53_source_matches_df = pd.DataFrame(
    notebook53_source_match_rows,
    columns=[
        "cell_index",
        "line_number",
        "matched_terms",
        "source_line",
    ],
)


print()
print("NOTEBOOK 53 TRAINING-SOURCE MATCHES")
print("-" * 100)

display(
    notebook53_source_matches_df
)


assert not notebook53_source_matches_df.empty


# --------------------------------------------------------------------------------------
# 4. Discover assignment targets from the Notebook 53 source
# --------------------------------------------------------------------------------------

notebook53_assignment_rows = []


for cell_index, cell in enumerate(
    notebook53_code_cells
):

    source_text = str(
        cell.source
    )


    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if isinstance(
            node,
            ast.Assign,
        ):

            target_names = []


            for target in node.targets:

                if isinstance(
                    target,
                    ast.Name,
                ):

                    target_names.append(
                        target.id
                    )


                elif isinstance(
                    target,
                    (
                        ast.Tuple,
                        ast.List,
                    ),
                ):

                    for element in target.elts:

                        if isinstance(
                            element,
                            ast.Name,
                        ):

                            target_names.append(
                                element.id
                            )


            for target_name in target_names:

                normalized_target_name = (
                    target_name.lower()
                )


                if any(
                    token in normalized_target_name
                    for token in [
                        "train",
                        "test",
                        "model",
                        "forest",
                        "preprocess",
                        "feature",
                        "target",
                        "label",
                        "dataset",
                        "policy",
                        "split",
                    ]
                ):

                    notebook53_assignment_rows.append(
                        {
                            "cell_index":
                                int(
                                    cell_index
                                ),

                            "line_number":
                                int(
                                    getattr(
                                        node,
                                        "lineno",
                                        0,
                                    )
                                ),

                            "variable_name":
                                target_name,

                            "assignment_source":
                                ast.get_source_segment(
                                    source_text,
                                    node,
                                )[:1000],
                        }
                    )


notebook53_assignment_inventory_df = (
    pd.DataFrame(
        notebook53_assignment_rows,
        columns=[
            "cell_index",
            "line_number",
            "variable_name",
            "assignment_source",
        ],
    )
    .drop_duplicates(
        subset=[
            "cell_index",
            "line_number",
            "variable_name",
        ]
    )
    .reset_index(drop=True)
)


print()
print("NOTEBOOK 53 TRAINING-RELATED ASSIGNMENTS")
print("-" * 100)

display(
    notebook53_assignment_inventory_df
)


assert not notebook53_assignment_inventory_df.empty


# --------------------------------------------------------------------------------------
# 5. Search the project for serialized training/model artifacts
# --------------------------------------------------------------------------------------

NOTEBOOK55_ARTIFACT_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pickle",
    ".parquet",
    ".csv",
    ".json",
    ".npz",
    ".npy",
}


notebook55_candidate_artifact_rows = []


artifact_search_roots = [
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "reports",
]


for search_root in artifact_search_roots:

    if not search_root.exists():

        continue


    for file_path in search_root.rglob("*"):

        if not file_path.is_file():

            continue


        if file_path.suffix.lower() not in (
            NOTEBOOK55_ARTIFACT_SUFFIXES
        ):

            continue


        normalized_name = file_path.name.lower()


        relevance_tokens = [
            token
            for token in [
                "53",
                "legality",
                "policy",
                "train",
                "dataset",
                "feature",
                "preprocess",
                "model",
                "split",
            ]
            if token in normalized_name
        ]


        if not relevance_tokens:

            continue


        notebook55_candidate_artifact_rows.append(
            {
                "file_name":
                    file_path.name,

                "file_path":
                    str(
                        file_path
                    ),

                "suffix":
                    file_path.suffix.lower(),

                "size_bytes":
                    int(
                        file_path.stat().st_size
                    ),

                "modified_timestamp":
                    float(
                        file_path.stat().st_mtime
                    ),

                "relevance_tokens":
                    " | ".join(
                        relevance_tokens
                    ),
            }
        )


notebook55_candidate_artifacts_df = (
    pd.DataFrame(
        notebook55_candidate_artifact_rows,
        columns=[
            "file_name",
            "file_path",
            "suffix",
            "size_bytes",
            "modified_timestamp",
            "relevance_tokens",
        ],
    )
    .sort_values(
        [
            "modified_timestamp",
            "size_bytes",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


print()
print("CANDIDATE NOTEBOOK 53 TRAINING/MODEL ARTIFACTS")
print("-" * 100)

if notebook55_candidate_artifacts_df.empty:

    print(
        "No matching serialized artifacts were found."
    )

else:

    display(
        notebook55_candidate_artifacts_df
    )


# --------------------------------------------------------------------------------------
# 6. Inventory relevant files directly referenced by Notebook 53
# --------------------------------------------------------------------------------------

NOTEBOOK55_PATH_PATTERN = re.compile(
    r"""["']([^"']+\.(?:csv|parquet|json|joblib|pkl|pickle|npz|npy))["']""",
    flags=re.IGNORECASE,
)


notebook53_referenced_file_rows = []


for cell_index, cell in enumerate(
    notebook53_code_cells
):

    source_text = str(
        cell.source
    )


    for matched_path in NOTEBOOK55_PATH_PATTERN.findall(
        source_text
    ):

        raw_path = Path(
            matched_path
        )


        candidate_paths = [
            raw_path,
            PROJECT_ROOT / raw_path,
            NOTEBOOKS_DIRECTORY / raw_path,
        ]


        resolved_candidate = None


        for candidate_path in candidate_paths:

            try:

                candidate_resolved = (
                    candidate_path.resolve()
                )

            except Exception:

                candidate_resolved = candidate_path


            if candidate_resolved.exists():

                resolved_candidate = candidate_resolved
                break


        notebook53_referenced_file_rows.append(
            {
                "cell_index":
                    int(
                        cell_index
                    ),

                "referenced_path":
                    matched_path,

                "resolved_path":
                    (
                        str(
                            resolved_candidate
                        )
                        if resolved_candidate
                        is not None
                        else ""
                    ),

                "exists":
                    resolved_candidate
                    is not None,
            }
        )


notebook53_referenced_files_df = (
    pd.DataFrame(
        notebook53_referenced_file_rows,
        columns=[
            "cell_index",
            "referenced_path",
            "resolved_path",
            "exists",
        ],
    )
    .drop_duplicates()
    .reset_index(drop=True)
)


print()
print("FILES REFERENCED DIRECTLY BY NOTEBOOK 53")
print("-" * 100)

if notebook53_referenced_files_df.empty:

    print(
        "No direct serialized-file references were detected."
    )

else:

    display(
        notebook53_referenced_files_df
    )


# --------------------------------------------------------------------------------------
# 7. Summarize discovery readiness
# --------------------------------------------------------------------------------------

notebook55_model_assignment_candidates = (
    notebook53_assignment_inventory_df[
        notebook53_assignment_inventory_df[
            "variable_name"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "model|forest|classifier",
            regex=True,
        )
    ]
)


notebook55_preprocessor_assignment_candidates = (
    notebook53_assignment_inventory_df[
        notebook53_assignment_inventory_df[
            "variable_name"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "preprocess|transform",
            regex=True,
        )
    ]
)


notebook55_split_assignment_candidates = (
    notebook53_assignment_inventory_df[
        notebook53_assignment_inventory_df[
            "variable_name"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "x_train|x_test|y_train|y_test|train.*split|test.*split",
            regex=True,
        )
    ]
)


notebook55_training_assignment_candidates = (
    notebook53_assignment_inventory_df[
        notebook53_assignment_inventory_df[
            "variable_name"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "train|dataset|feature|target|label",
            regex=True,
        )
    ]
)


notebook55_section2a_summary = {
    "status":
        "NOTEBOOK53_TRAINING_SOURCE_DISCOVERY_COMPLETE",

    "notebook53_code_cells":
        int(
            len(
                notebook53_code_cells
            )
        ),

    "source_matches":
        int(
            len(
                notebook53_source_matches_df
            )
        ),

    "training_related_assignments":
        int(
            len(
                notebook53_assignment_inventory_df
            )
        ),

    "model_assignment_candidates":
        int(
            len(
                notebook55_model_assignment_candidates
            )
        ),

    "preprocessor_assignment_candidates":
        int(
            len(
                notebook55_preprocessor_assignment_candidates
            )
        ),

    "split_assignment_candidates":
        int(
            len(
                notebook55_split_assignment_candidates
            )
        ),

    "training_assignment_candidates":
        int(
            len(
                notebook55_training_assignment_candidates
            )
        ),

    "candidate_serialized_artifacts":
        int(
            len(
                notebook55_candidate_artifacts_df
            )
        ),

    "direct_referenced_files":
        int(
            len(
                notebook53_referenced_files_df
            )
        ),

    "next_stage":
        "RECOVER_EXACT_TRAINING_OBJECTS_AND_SPLIT_CONTRACT",
}


print()
print("SECTION 2A DISCOVERY SUMMARY")
print("-" * 100)

for key, value in (
    notebook55_section2a_summary.items()
):

    print(
        f"{key:58}: {value}"
    )


# --------------------------------------------------------------------------------------
# 8. Validation checks
# --------------------------------------------------------------------------------------

notebook55_section2a_validation_df = pd.DataFrame(
    [
        {
            "check":
                "notebook53_source_loaded",

            "passed":
                len(
                    notebook53_code_cells
                ) > 0,

            "value":
                len(
                    notebook53_code_cells
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "training_source_matches_found",

            "passed":
                not notebook53_source_matches_df.empty,

            "value":
                len(
                    notebook53_source_matches_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "training_related_assignments_found",

            "passed":
                not notebook53_assignment_inventory_df.empty,

            "value":
                len(
                    notebook53_assignment_inventory_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "model_assignment_candidates_found",

            "passed":
                not notebook55_model_assignment_candidates.empty,

            "value":
                len(
                    notebook55_model_assignment_candidates
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "preprocessor_assignment_candidates_found",

            "passed":
                not notebook55_preprocessor_assignment_candidates.empty,

            "value":
                len(
                    notebook55_preprocessor_assignment_candidates
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "next_stage_recognized",

            "passed":
                notebook55_section2a_summary[
                    "next_stage"
                ]
                ==
                "RECOVER_EXACT_TRAINING_OBJECTS_AND_SPLIT_CONTRACT",

            "value":
                notebook55_section2a_summary[
                    "next_stage"
                ],

            "expected":
                "RECOVER_EXACT_TRAINING_OBJECTS_AND_SPLIT_CONTRACT",
        },
    ]
)


print()
print("SECTION 2A VALIDATION CHECKS")
print("-" * 100)

display(
    notebook55_section2a_validation_df
)


assert notebook55_section2a_validation_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 9. Save discovery reports
# --------------------------------------------------------------------------------------

NOTEBOOK55_SECTION2A_SOURCE_MATCHES_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_notebook53_source_matches.csv"
)

NOTEBOOK55_SECTION2A_ASSIGNMENTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_notebook53_assignment_inventory.csv"
)

NOTEBOOK55_SECTION2A_ARTIFACTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_candidate_training_artifacts.csv"
)

NOTEBOOK55_SECTION2A_REFERENCED_FILES_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_notebook53_referenced_files.csv"
)

NOTEBOOK55_SECTION2A_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_validation_checks.csv"
)

NOTEBOOK55_SECTION2A_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section2a_training_source_discovery_summary.json"
)


notebook53_source_matches_df.to_csv(
    NOTEBOOK55_SECTION2A_SOURCE_MATCHES_FILE,
    index=False,
)

notebook53_assignment_inventory_df.to_csv(
    NOTEBOOK55_SECTION2A_ASSIGNMENTS_FILE,
    index=False,
)

notebook55_candidate_artifacts_df.to_csv(
    NOTEBOOK55_SECTION2A_ARTIFACTS_FILE,
    index=False,
)

notebook53_referenced_files_df.to_csv(
    NOTEBOOK55_SECTION2A_REFERENCED_FILES_FILE,
    index=False,
)

notebook55_section2a_validation_df.to_csv(
    NOTEBOOK55_SECTION2A_VALIDATION_FILE,
    index=False,
)


with open(
    NOTEBOOK55_SECTION2A_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook55_section2a_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


notebook55_section2a_saved_files = [
    NOTEBOOK55_SECTION2A_SOURCE_MATCHES_FILE,
    NOTEBOOK55_SECTION2A_ASSIGNMENTS_FILE,
    NOTEBOOK55_SECTION2A_ARTIFACTS_FILE,
    NOTEBOOK55_SECTION2A_REFERENCED_FILES_FILE,
    NOTEBOOK55_SECTION2A_VALIDATION_FILE,
    NOTEBOOK55_SECTION2A_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in notebook55_section2a_saved_files
)


print()
print("SAVED SECTION 2A REPORTS")
print("-" * 100)

for file_path in notebook55_section2a_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 2A NOTEBOOK 53 TRAINING SOURCE DISCOVERY PASSED"
)


# # SECTION 2B — RESTORE NOTEBOOK 53 TRAINED MODEL, PREPROCESSOR, AND TRAINING OBJECTS
# 
# ## This section restores exactly the objects discovered in Section 2A so every experiment starts from the identical baseline mode

# In[6]:


# ======================================================================================
# SECTION 2B BOOTSTRAP — LOAD NOTEBOOK 53 RUNTIME OBJECTS
# ======================================================================================

print("=" * 100)
print("SECTION 2B BOOTSTRAP — LOAD NOTEBOOK 53 RUNTIME OBJECTS")
print("=" * 100)

from pathlib import Path
import copy
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Confirm Notebook 53 exists
# --------------------------------------------------------------------------------------

NOTEBOOK53_RUNTIME_PATH = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents"
    r"\PTCG_AI_Battle_Challenge"
    r"\notebooks"
    r"\53_legality_aware_policy_optimization.ipynb"
)

assert NOTEBOOK53_RUNTIME_PATH.exists(), (
    f"Notebook 53 was not found: {NOTEBOOK53_RUNTIME_PATH}"
)

print()
print("Notebook 53:")
print(NOTEBOOK53_RUNTIME_PATH)


# --------------------------------------------------------------------------------------
# 2. Execute Notebook 53 inside the current Notebook 55 kernel
# --------------------------------------------------------------------------------------
#
# This restores the completed Notebook 53 runtime objects.
# It does not modify the Notebook 53 file.
# --------------------------------------------------------------------------------------

print()
print("Executing Notebook 53 to restore its trained runtime...")
print("This may take several minutes.")
print()

get_ipython().run_line_magic(
    "run",
    f'"{NOTEBOOK53_RUNTIME_PATH}"'
)


# --------------------------------------------------------------------------------------
# 3. Resolve the fitted model from known Notebook 53 candidate names
# --------------------------------------------------------------------------------------

MODEL_CANDIDATE_NAMES_55 = [
    "legality_aware_policy_model",
    "notebook53_legality_model",
    "legality_model",
    "policy_model",
]


resolved_model_name_55 = next(
    (
        candidate_name
        for candidate_name in MODEL_CANDIDATE_NAMES_55
        if candidate_name in globals()
        and hasattr(
            globals()[candidate_name],
            "predict",
        )
    ),
    None,
)


assert resolved_model_name_55 is not None, (
    "A fitted Notebook 53 policy model was not recovered. "
    f"Checked: {MODEL_CANDIDATE_NAMES_55}"
)


legality_aware_policy_model = globals()[
    resolved_model_name_55
]


# --------------------------------------------------------------------------------------
# 4. Resolve the fitted preprocessor
# --------------------------------------------------------------------------------------

PREPROCESSOR_CANDIDATE_NAMES_55 = [
    "legality_preprocessor",
    "notebook53_preprocessor",
    "preprocessor",
]


resolved_preprocessor_name_55 = next(
    (
        candidate_name
        for candidate_name in PREPROCESSOR_CANDIDATE_NAMES_55
        if candidate_name in globals()
        and hasattr(
            globals()[candidate_name],
            "transform",
        )
    ),
    None,
)


assert resolved_preprocessor_name_55 is not None, (
    "A fitted Notebook 53 preprocessor was not recovered. "
    f"Checked: {PREPROCESSOR_CANDIDATE_NAMES_55}"
)


legality_preprocessor = globals()[
    resolved_preprocessor_name_55
]


# --------------------------------------------------------------------------------------
# 5. Resolve encoded feature names
# --------------------------------------------------------------------------------------

FEATURE_NAME_CANDIDATES_55 = [
    "encoded_feature_names",
    "legality_encoded_feature_names",
    "notebook53_encoded_feature_names",
]


resolved_feature_name_object_55 = next(
    (
        candidate_name
        for candidate_name in FEATURE_NAME_CANDIDATES_55
        if candidate_name in globals()
        and globals()[candidate_name] is not None
    ),
    None,
)


if resolved_feature_name_object_55 is not None:

    encoded_feature_names = list(
        globals()[
            resolved_feature_name_object_55
        ]
    )

elif hasattr(
    legality_preprocessor,
    "get_feature_names_out",
):

    encoded_feature_names = list(
        legality_preprocessor.get_feature_names_out()
    )

else:

    feature_importances_55 = getattr(
        legality_aware_policy_model,
        "feature_importances_",
        [],
    )

    encoded_feature_names = [
        f"feature_{feature_index:04d}"
        for feature_index in range(
            len(
                feature_importances_55
            )
        )
    ]


# --------------------------------------------------------------------------------------
# 6. Validate restored objects
# --------------------------------------------------------------------------------------

section2b_bootstrap_validation_df = pd.DataFrame(
    [
        {
            "object":
                "legality_aware_policy_model",

            "source_name":
                resolved_model_name_55,

            "type":
                type(
                    legality_aware_policy_model
                ).__name__,

            "exists":
                legality_aware_policy_model is not None,

            "ready":
                hasattr(
                    legality_aware_policy_model,
                    "predict",
                ),
        },
        {
            "object":
                "legality_preprocessor",

            "source_name":
                resolved_preprocessor_name_55,

            "type":
                type(
                    legality_preprocessor
                ).__name__,

            "exists":
                legality_preprocessor is not None,

            "ready":
                hasattr(
                    legality_preprocessor,
                    "transform",
                ),
        },
        {
            "object":
                "encoded_feature_names",

            "source_name":
                (
                    resolved_feature_name_object_55
                    or
                    "preprocessor.get_feature_names_out"
                ),

            "type":
                type(
                    encoded_feature_names
                ).__name__,

            "exists":
                encoded_feature_names is not None,

            "ready":
                len(
                    encoded_feature_names
                ) > 0,
        },
    ]
)


print()
print("RESTORED NOTEBOOK 53 OBJECTS")
print("-" * 100)

display(
    section2b_bootstrap_validation_df
)


print()
print("Model classes:")

if hasattr(
    legality_aware_policy_model,
    "classes_",
):

    print(
        list(
            legality_aware_policy_model.classes_
        )
    )

else:

    print(
        "No classes_ attribute found."
    )


print()
print(
    "Encoded feature count:",
    len(
        encoded_feature_names
    ),
)


assert section2b_bootstrap_validation_df[
    "exists"
].astype(bool).all()


assert section2b_bootstrap_validation_df[
    "ready"
].astype(bool).all()


if hasattr(
    legality_aware_policy_model,
    "n_features_in_",
):

    assert (
        len(
            encoded_feature_names
        )
        ==
        int(
            legality_aware_policy_model.n_features_in_
        )
    ), (
        "Feature-name count does not match the model input count: "
        f"{len(encoded_feature_names)} versus "
        f"{legality_aware_policy_model.n_features_in_}."
    )


print()
print(
    "✅ SECTION 2B NOTEBOOK 53 RUNTIME BOOTSTRAP PASSED"
)


# ## Part 1 - Validate Notebook 53 Objects

# In[9]:


# ======================================================================================
# SECTION 2B — RESTORE NOTEBOOK 53 TRAINING OBJECTS
# Part 1 — Validate required Notebook 53 objects
# ======================================================================================

print("=" * 100)
print("SECTION 2B — RESTORE NOTEBOOK 53 TRAINED MODEL")
print("=" * 100)

required_objects = [
    "legality_aware_policy_model",
    "legality_preprocessor",
    "encoded_feature_names",
]

missing = [
    obj
    for obj in required_objects
    if obj not in globals()
]

assert not missing, (
    f"Notebook 53 objects missing: {missing}"
)

print("✓ Notebook 53 training objects detected.")


# ## Part 2 — Freeze Baseline Objects

# In[8]:


# ======================================================================================
# Freeze baseline objects
# ======================================================================================

import copy

baseline_policy_model = copy.deepcopy(
    legality_aware_policy_model
)

baseline_preprocessor = copy.deepcopy(
    legality_preprocessor
)

baseline_encoded_feature_names = list(
    encoded_feature_names
)

print("Model frozen.")
print("Preprocessor frozen.")
print("Encoded feature names:", len(baseline_encoded_feature_names))


# ## Part 3 — Extract Random Forest Metadata

# In[10]:


# ======================================================================================
# Random Forest metadata
# ======================================================================================

section2b_model_metadata = {

    "model_type":
        type(baseline_policy_model).__name__,

    "n_estimators":
        getattr(
            baseline_policy_model,
            "n_estimators",
            None,
        ),

    "max_depth":
        getattr(
            baseline_policy_model,
            "max_depth",
            None,
        ),

    "max_features":
        getattr(
            baseline_policy_model,
            "max_features",
            None,
        ),

    "criterion":
        getattr(
            baseline_policy_model,
            "criterion",
            None,
        ),

    "random_state":
        getattr(
            baseline_policy_model,
            "random_state",
            None,
        ),

    "feature_count":
        len(
            baseline_encoded_feature_names
        ),

}


# ## Part 4 — Build Baseline Summary

# In[11]:


# ======================================================================================
# Baseline summary
# ======================================================================================

section2b_summary = {

    "status":
        "NOTEBOOK53_MODEL_RESTORED",

    "model_type":
        section2b_model_metadata["model_type"],

    "feature_count":
        section2b_model_metadata["feature_count"],

    "random_state":
        section2b_model_metadata["random_state"],

    "next_stage":
        "FEATURE_FAMILY_PARTITION",

}


# ## Part 5 — Save Reports

# In[12]:


# ======================================================================================
# Save reports
# ======================================================================================

section2b_metadata_df = pd.DataFrame(
    [section2b_model_metadata]
)

section2b_summary_df = pd.DataFrame(
    [
        {
            "key": k,
            "value": v,
        }
        for k, v in section2b_summary.items()
    ]
)

section2b_metadata_df.to_csv(
    REPORTS_DIRECTORY /
    "section2b_model_metadata.csv",
    index=False,
)

section2b_summary_df.to_csv(
    REPORTS_DIRECTORY /
    "section2b_restore_summary.csv",
    index=False,
)

pd.DataFrame(
    {
        "feature_name":
            baseline_encoded_feature_names
    }
).to_csv(
    REPORTS_DIRECTORY /
    "section2b_feature_inventory.csv",
    index=False,
)

validation_df = pd.DataFrame(
    [
        {
            "check":
                "model_exists",
            "passed":
                baseline_policy_model is not None,
        },
        {
            "check":
                "preprocessor_exists",
            "passed":
                baseline_preprocessor is not None,
        },
        {
            "check":
                "feature_inventory",
            "passed":
                len(
                    baseline_encoded_feature_names
                ) > 0,
        },
    ]
)

validation_df.to_csv(
    REPORTS_DIRECTORY /
    "section2b_validation_checks.csv",
    index=False,
)


# ## Part 6 — Display

# In[13]:


print("=" * 100)
print("SECTION 2B SUMMARY")
print("=" * 100)

for k, v in section2b_summary.items():
    print(f"{k:30}: {v}")

display(section2b_metadata_df)

display(validation_df)

assert validation_df["passed"].all()

print()
print("SAVED SECTION 2B REPORTS")
print("-" * 100)

for f in [
    "section2b_model_metadata.csv",
    "section2b_restore_summary.csv",
    "section2b_feature_inventory.csv",
    "section2b_validation_checks.csv",
]:
    print(REPORTS_DIRECTORY / f)

print()
print("✅ SECTION 2B NOTEBOOK 53 MODEL RESTORATION PASSED")


# In[14]:


# ======================================================================================
# SECTION 2C — PARTITION THE 64 ENCODED FEATURES INTO REMEDIATION FAMILIES
# ======================================================================================

print("=" * 100)
print("SECTION 2C — PARTITION THE 64 ENCODED FEATURES INTO REMEDIATION FAMILIES")
print("=" * 100)

from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate Section 2B dependencies
# --------------------------------------------------------------------------------------

SECTION2C_REQUIRED_OBJECTS = [
    "baseline_policy_model",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "section2b_summary",
    "REPORTS_DIRECTORY",
]


section2c_missing_objects = [
    object_name
    for object_name in SECTION2C_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 2C OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION2C_REQUIRED_OBJECTS:

    print(
        f"{object_name:44}: "
        f"{object_name in globals()}"
    )


assert not section2c_missing_objects, (
    "Section 2C required objects are missing: "
    f"{section2c_missing_objects}"
)


assert (
    section2b_summary[
        "next_stage"
    ]
    ==
    "FEATURE_FAMILY_PARTITION"
)


assert len(
    baseline_encoded_feature_names
) == 64


if hasattr(
    baseline_policy_model,
    "n_features_in_",
):

    assert int(
        baseline_policy_model.n_features_in_
    ) == 64


# --------------------------------------------------------------------------------------
# 2. Create the protected encoded-feature inventory
# --------------------------------------------------------------------------------------

section2c_feature_inventory_df = pd.DataFrame(
    {
        "feature_index":
            list(
                range(
                    len(
                        baseline_encoded_feature_names
                    )
                )
            ),

        "feature_name":
            [
                str(
                    feature_name
                )
                for feature_name
                in baseline_encoded_feature_names
            ],
    }
)


assert len(
    section2c_feature_inventory_df
) == 64


assert section2c_feature_inventory_df[
    "feature_name"
].notna().all()


assert not section2c_feature_inventory_df[
    "feature_name"
].duplicated().any(), (
    "Duplicate encoded feature names were detected."
)


print()
print("ENCODED FEATURE INVENTORY")
print("-" * 100)

display(
    section2c_feature_inventory_df
)


# --------------------------------------------------------------------------------------
# 3. Feature-family classifier
# --------------------------------------------------------------------------------------

def classify_notebook55_feature_family(
    feature_name,
):
    """
    Assign one encoded Notebook 53 feature to exactly one remediation family.
    """

    normalized_name = str(
        feature_name
    ).strip().lower()


    # Exact encoded legal-move combinations.
    if "legal_move_signature" in normalized_name:

        return "LEGAL_MOVE_SIGNATURE"


    # Individual action legality indicators.
    if "is_legal__" in normalized_name:

        return "LEGAL_ACTION_FLAG"


    # Active-card/action compatibility shortcuts.
    if "active_card_can_use__" in normalized_name:

        return "CARD_ACTION_COMPATIBILITY"


    # Aggregate legality-choice structure.
    if any(
        token in normalized_name
        for token in [
            "single_legal_move_flag",
            "multi_legal_move_flag",
            "model_choice_required_flag",
            "recomputed_legal_move_count",
        ]
    ):

        return "LEGAL_CHOICE_STRUCTURE"


    # Explicit card identity.
    if any(
        token in normalized_name
        for token in [
            "active_card_",
            "inactive_card_",
            "player_card_",
            "opponent_card_",
        ]
    ):

        return "CARD_IDENTITY"


    # Energy-related state.
    if "energy" in normalized_name:

        return "ENERGY_STATE"


    # Damage-related state.
    if "damage" in normalized_name:

        return "DAMAGE_STATE"


    # Turn and battle-phase context.
    if any(
        token in normalized_name
        for token in [
            "turn_number",
            "battle_phase",
        ]
    ):

        return "TURN_STATE"


    # Side and orientation context.
    if any(
        token in normalized_name
        for token in [
            "current_side",
            "side_mode",
        ]
    ):

        return "SIDE_CONTEXT"


    # Hand and prize resource context.
    if any(
        token in normalized_name
        for token in [
            "prize_cards_remaining",
            "hand_size",
        ]
    ):

        return "RESOURCE_STATE"


    return "OTHER"


section2c_feature_inventory_df[
    "feature_family"
] = (
    section2c_feature_inventory_df[
        "feature_name"
    ]
    .apply(
        classify_notebook55_feature_family
    )
)


# --------------------------------------------------------------------------------------
# 4. Add remediation-role flags
# --------------------------------------------------------------------------------------

SECTION2C_PRIMARY_ABLATION_FAMILIES = {
    "LEGAL_MOVE_SIGNATURE",
}


SECTION2C_SECONDARY_ABLATION_FAMILIES = {
    "LEGAL_MOVE_SIGNATURE",
    "CARD_ACTION_COMPATIBILITY",
}


SECTION2C_STATE_CENTRIC_REMOVED_FAMILIES = {
    "LEGAL_MOVE_SIGNATURE",
    "CARD_ACTION_COMPATIBILITY",
    "LEGAL_ACTION_FLAG",
}


section2c_feature_inventory_df[
    "remove_in_r1"
] = (
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .isin(
        SECTION2C_PRIMARY_ABLATION_FAMILIES
    )
)


section2c_feature_inventory_df[
    "remove_in_r2"
] = (
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .isin(
        SECTION2C_SECONDARY_ABLATION_FAMILIES
    )
)


section2c_feature_inventory_df[
    "remove_in_r3"
] = (
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .isin(
        SECTION2C_STATE_CENTRIC_REMOVED_FAMILIES
    )
)


section2c_feature_inventory_df[
    "retain_in_r0"
] = True


section2c_feature_inventory_df[
    "retain_in_r1"
] = (
    ~section2c_feature_inventory_df[
        "remove_in_r1"
    ]
)


section2c_feature_inventory_df[
    "retain_in_r2"
] = (
    ~section2c_feature_inventory_df[
        "remove_in_r2"
    ]
)


section2c_feature_inventory_df[
    "retain_in_r3"
] = (
    ~section2c_feature_inventory_df[
        "remove_in_r3"
    ]
)


# --------------------------------------------------------------------------------------
# 5. Create family summary
# --------------------------------------------------------------------------------------

section2c_family_summary_df = (
    section2c_feature_inventory_df
    .groupby(
        "feature_family",
        dropna=False,
    )
    .agg(
        feature_count=(
            "feature_name",
            "size",
        ),

        first_feature_index=(
            "feature_index",
            "min",
        ),

        last_feature_index=(
            "feature_index",
            "max",
        ),

        removed_in_r1=(
            "remove_in_r1",
            "sum",
        ),

        removed_in_r2=(
            "remove_in_r2",
            "sum",
        ),

        removed_in_r3=(
            "remove_in_r3",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "feature_count",
            "feature_family",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


section2c_family_summary_df[
    "feature_share"
] = (
    section2c_family_summary_df[
        "feature_count"
    ]
    /
    len(
        section2c_feature_inventory_df
    )
)


print()
print("FEATURE FAMILY SUMMARY")
print("-" * 100)

display(
    section2c_family_summary_df
)


# --------------------------------------------------------------------------------------
# 6. Build strategy-specific retained-feature inventories
# --------------------------------------------------------------------------------------

SECTION2C_STRATEGY_COLUMN_MAP = {
    "R0_CURRENT_BASELINE":
        "retain_in_r0",

    "R1_REMOVE_LEGAL_SIGNATURE":
        "retain_in_r1",

    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY":
        "retain_in_r2",

    "R3_STATE_CENTRIC_POLICY":
        "retain_in_r3",
}


section2c_strategy_rows = []


for strategy_id, retain_column in (
    SECTION2C_STRATEGY_COLUMN_MAP.items()
):

    retained_mask = (
        section2c_feature_inventory_df[
            retain_column
        ].astype(bool)
    )


    removed_mask = (
        ~retained_mask
    )


    retained_feature_names = (
        section2c_feature_inventory_df.loc[
            retained_mask,
            "feature_name",
        ]
        .astype(str)
        .tolist()
    )


    removed_feature_names = (
        section2c_feature_inventory_df.loc[
            removed_mask,
            "feature_name",
        ]
        .astype(str)
        .tolist()
    )


    retained_feature_indices = (
        section2c_feature_inventory_df.loc[
            retained_mask,
            "feature_index",
        ]
        .astype(int)
        .tolist()
    )


    removed_feature_indices = (
        section2c_feature_inventory_df.loc[
            removed_mask,
            "feature_index",
        ]
        .astype(int)
        .tolist()
    )


    section2c_strategy_rows.append(
        {
            "strategy_id":
                strategy_id,

            "baseline_feature_count":
                64,

            "retained_feature_count":
                len(
                    retained_feature_names
                ),

            "removed_feature_count":
                len(
                    removed_feature_names
                ),

            "retained_feature_share":
                len(
                    retained_feature_names
                )
                /
                64,

            "removed_feature_share":
                len(
                    removed_feature_names
                )
                /
                64,

            "retained_feature_indices":
                json.dumps(
                    retained_feature_indices
                ),

            "removed_feature_indices":
                json.dumps(
                    removed_feature_indices
                ),

            "retained_feature_names":
                json.dumps(
                    retained_feature_names
                ),

            "removed_feature_names":
                json.dumps(
                    removed_feature_names
                ),
        }
    )


section2c_strategy_partition_df = pd.DataFrame(
    section2c_strategy_rows
)


print()
print("STRATEGY FEATURE PARTITION")
print("-" * 100)

display(
    section2c_strategy_partition_df[
        [
            "strategy_id",
            "baseline_feature_count",
            "retained_feature_count",
            "removed_feature_count",
            "retained_feature_share",
            "removed_feature_share",
        ]
    ]
)


assert len(
    section2c_strategy_partition_df
) == 4


# --------------------------------------------------------------------------------------
# 7. Create individual feature lists for later training
# --------------------------------------------------------------------------------------

NOTEBOOK55_STRATEGY_FEATURE_INDICES = {}
NOTEBOOK55_STRATEGY_FEATURE_NAMES = {}


for strategy_id, retain_column in (
    SECTION2C_STRATEGY_COLUMN_MAP.items()
):

    strategy_mask = (
        section2c_feature_inventory_df[
            retain_column
        ].astype(bool)
    )


    NOTEBOOK55_STRATEGY_FEATURE_INDICES[
        strategy_id
    ] = (
        section2c_feature_inventory_df.loc[
            strategy_mask,
            "feature_index",
        ]
        .astype(int)
        .tolist()
    )


    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
        strategy_id
    ] = (
        section2c_feature_inventory_df.loc[
            strategy_mask,
            "feature_name",
        ]
        .astype(str)
        .tolist()
    )


assert len(
    NOTEBOOK55_STRATEGY_FEATURE_INDICES[
        "R0_CURRENT_BASELINE"
    ]
) == 64


assert len(
    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
        "R0_CURRENT_BASELINE"
    ]
) == 64


# --------------------------------------------------------------------------------------
# 8. Display exact removed features for each remediation strategy
# --------------------------------------------------------------------------------------

for strategy_id in [
    "R1_REMOVE_LEGAL_SIGNATURE",
    "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY",
    "R3_STATE_CENTRIC_POLICY",
]:

    strategy_row = (
        section2c_strategy_partition_df.loc[
            section2c_strategy_partition_df[
                "strategy_id"
            ].eq(
                strategy_id
            )
        ]
        .iloc[0]
    )


    removed_features = json.loads(
        strategy_row[
            "removed_feature_names"
        ]
    )


    print()
    print(
        f"{strategy_id} — REMOVED FEATURES"
    )
    print("-" * 100)

    if removed_features:

        for feature_name in removed_features:

            print(
                feature_name
            )

    else:

        print(
            "No features removed."
        )


# --------------------------------------------------------------------------------------
# 9. Determine partition readiness
# --------------------------------------------------------------------------------------

section2c_feature_family_counts = {
    str(
        row[
            "feature_family"
        ]
    ):
        int(
            row[
                "feature_count"
            ]
        )

    for _, row in (
        section2c_family_summary_df.iterrows()
    )
}


section2c_legal_signature_feature_count = int(
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .eq(
        "LEGAL_MOVE_SIGNATURE"
    )
    .sum()
)


section2c_compatibility_feature_count = int(
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .eq(
        "CARD_ACTION_COMPATIBILITY"
    )
    .sum()
)


section2c_legal_action_flag_count = int(
    section2c_feature_inventory_df[
        "feature_family"
    ]
    .eq(
        "LEGAL_ACTION_FLAG"
    )
    .sum()
)


assert section2c_legal_signature_feature_count > 0, (
    "No encoded legal-move-signature features were identified."
)


assert section2c_compatibility_feature_count > 0, (
    "No card-action compatibility features were identified."
)


assert section2c_legal_action_flag_count > 0, (
    "No individual legal-action indicator features were identified."
)


section2c_summary = {
    "status":
        "ENCODED_FEATURE_FAMILY_PARTITION_COMPLETE",

    "baseline_feature_count":
        int(
            len(
                section2c_feature_inventory_df
            )
        ),

    "feature_family_count":
        int(
            section2c_feature_inventory_df[
                "feature_family"
            ].nunique()
        ),

    "feature_family_counts":
        section2c_feature_family_counts,

    "legal_move_signature_features":
        section2c_legal_signature_feature_count,

    "card_action_compatibility_features":
        section2c_compatibility_feature_count,

    "legal_action_flag_features":
        section2c_legal_action_flag_count,

    "r0_retained_features":
        len(
            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                "R0_CURRENT_BASELINE"
            ]
        ),

    "r1_retained_features":
        len(
            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                "R1_REMOVE_LEGAL_SIGNATURE"
            ]
        ),

    "r2_retained_features":
        len(
            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY"
            ]
        ),

    "r3_retained_features":
        len(
            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                "R3_STATE_CENTRIC_POLICY"
            ]
        ),

    "partition_complete":
        True,

    "next_stage":
        "RECOVER_TRAINING_AND_VALIDATION_MATRICES",
}


print()
print("SECTION 2C FEATURE PARTITION SUMMARY")
print("-" * 100)

for key, value in (
    section2c_summary.items()
):

    print(
        f"{key:54}: {value}"
    )


# --------------------------------------------------------------------------------------
# 10. Validation checks
# --------------------------------------------------------------------------------------

section2c_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_64_features_partitioned",

            "passed":
                len(
                    section2c_feature_inventory_df
                ) == 64,

            "value":
                len(
                    section2c_feature_inventory_df
                ),

            "expected":
                64,
        },
        {
            "check":
                "every_feature_has_family",

            "passed":
                section2c_feature_inventory_df[
                    "feature_family"
                ].notna().all(),

            "value":
                int(
                    section2c_feature_inventory_df[
                        "feature_family"
                    ].notna().sum()
                ),

            "expected":
                64,
        },
        {
            "check":
                "feature_names_unique",

            "passed":
                not section2c_feature_inventory_df[
                    "feature_name"
                ].duplicated().any(),

            "value":
                int(
                    section2c_feature_inventory_df[
                        "feature_name"
                    ].nunique()
                ),

            "expected":
                64,
        },
        {
            "check":
                "legal_signature_features_found",

            "passed":
                section2c_legal_signature_feature_count > 0,

            "value":
                section2c_legal_signature_feature_count,

            "expected":
                "> 0",
        },
        {
            "check":
                "compatibility_features_found",

            "passed":
                section2c_compatibility_feature_count > 0,

            "value":
                section2c_compatibility_feature_count,

            "expected":
                "> 0",
        },
        {
            "check":
                "r0_preserves_all_features",

            "passed":
                len(
                    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                        "R0_CURRENT_BASELINE"
                    ]
                ) == 64,

            "value":
                len(
                    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                        "R0_CURRENT_BASELINE"
                    ]
                ),

            "expected":
                64,
        },
        {
            "check":
                "r1_removes_signature_features",

            "passed":
                (
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            "R1_REMOVE_LEGAL_SIGNATURE"
                        ]
                    )
                    <
                    64
                ),

            "value":
                len(
                    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                        "R1_REMOVE_LEGAL_SIGNATURE"
                    ]
                ),

            "expected":
                "< 64",
        },
        {
            "check":
                "r2_removes_more_than_r1",

            "passed":
                (
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY"
                        ]
                    )
                    <
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            "R1_REMOVE_LEGAL_SIGNATURE"
                        ]
                    )
                ),

            "value":
                {
                    "r1":
                        len(
                            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                                "R1_REMOVE_LEGAL_SIGNATURE"
                            ]
                        ),

                    "r2":
                        len(
                            NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                                "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY"
                            ]
                        ),
                },

            "expected":
                "R2 < R1",
        },
        {
            "check":
                "r3_state_centric_partition_created",

            "passed":
                (
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            "R3_STATE_CENTRIC_POLICY"
                        ]
                    )
                    <
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            "R2_REMOVE_SIGNATURE_AND_COMPATIBILITY"
                        ]
                    )
                ),

            "value":
                len(
                    NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                        "R3_STATE_CENTRIC_POLICY"
                    ]
                ),

            "expected":
                "Less than R2",
        },
        {
            "check":
                "next_stage_recognized",

            "passed":
                section2c_summary[
                    "next_stage"
                ]
                ==
                "RECOVER_TRAINING_AND_VALIDATION_MATRICES",

            "value":
                section2c_summary[
                    "next_stage"
                ],

            "expected":
                "RECOVER_TRAINING_AND_VALIDATION_MATRICES",
        },
    ]
)


print()
print("SECTION 2C VALIDATION CHECKS")
print("-" * 100)

display(
    section2c_validation_checks_df
)


section2c_failed_checks = int(
    (
        ~section2c_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section2c_failed_checks == 0, (
    "One or more Section 2C validation checks failed."
)


# --------------------------------------------------------------------------------------
# 11. Save Section 2C reports
# --------------------------------------------------------------------------------------

SECTION2C_FEATURE_INVENTORY_FILE = (
    REPORTS_DIRECTORY
    /
    "section2c_encoded_feature_partition.csv"
)

SECTION2C_FAMILY_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section2c_feature_family_summary.csv"
)

SECTION2C_STRATEGY_PARTITION_FILE = (
    REPORTS_DIRECTORY
    /
    "section2c_strategy_feature_partition.csv"
)

SECTION2C_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section2c_validation_checks.csv"
)

SECTION2C_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section2c_feature_partition_summary.json"
)

SECTION2C_STRATEGY_FEATURES_FILE = (
    ARTIFACTS_DIRECTORY
    /
    "section2c_strategy_feature_indices_and_names.json"
)


section2c_feature_inventory_df.to_csv(
    SECTION2C_FEATURE_INVENTORY_FILE,
    index=False,
)

section2c_family_summary_df.to_csv(
    SECTION2C_FAMILY_SUMMARY_FILE,
    index=False,
)

section2c_strategy_partition_df.to_csv(
    SECTION2C_STRATEGY_PARTITION_FILE,
    index=False,
)

section2c_validation_checks_df.to_csv(
    SECTION2C_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION2C_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section2c_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


with open(
    SECTION2C_STRATEGY_FEATURES_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "strategy_feature_indices":
                NOTEBOOK55_STRATEGY_FEATURE_INDICES,

            "strategy_feature_names":
                NOTEBOOK55_STRATEGY_FEATURE_NAMES,
        },
        file,
        indent=2,
        ensure_ascii=False,
    )


section2c_saved_files = [
    SECTION2C_FEATURE_INVENTORY_FILE,
    SECTION2C_FAMILY_SUMMARY_FILE,
    SECTION2C_STRATEGY_PARTITION_FILE,
    SECTION2C_VALIDATION_FILE,
    SECTION2C_SUMMARY_FILE,
    SECTION2C_STRATEGY_FEATURES_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section2c_saved_files
)


print()
print("SAVED SECTION 2C REPORTS")
print("-" * 100)

for file_path in section2c_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 2C ENCODED FEATURE FAMILY PARTITION PASSED"
)


# In[15]:


# ======================================================================================
# SECTION 2D — RECOVER AND FREEZE TRAINING, VALIDATION, AND TEST MATRICES
# ======================================================================================

print("=" * 100)
print("SECTION 2D — RECOVER AND FREEZE TRAINING, VALIDATION, AND TEST MATRICES")
print("=" * 100)

from pathlib import Path
import copy
import hashlib
import json
import joblib

import numpy as np
import pandas as pd

from scipy import sparse


# --------------------------------------------------------------------------------------
# 1. Validate Section 2C dependencies
# --------------------------------------------------------------------------------------

SECTION2D_REQUIRED_OBJECTS = [
    "baseline_policy_model",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "NOTEBOOK55_STRATEGY_FEATURE_INDICES",
    "NOTEBOOK55_STRATEGY_FEATURE_NAMES",
    "section2c_summary",
    "REPORTS_DIRECTORY",
    "ARTIFACTS_DIRECTORY",
]


section2d_missing_objects = [
    object_name
    for object_name in SECTION2D_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 2D OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION2D_REQUIRED_OBJECTS:

    print(
        f"{object_name:48}: "
        f"{object_name in globals()}"
    )


assert not section2d_missing_objects, (
    "Section 2D required objects are missing: "
    f"{section2d_missing_objects}"
)


assert (
    section2c_summary[
        "next_stage"
    ]
    ==
    "RECOVER_TRAINING_AND_VALIDATION_MATRICES"
)


# --------------------------------------------------------------------------------------
# 2. Resolve exact Notebook 53 encoded matrices
# --------------------------------------------------------------------------------------

SECTION2D_MATRIX_CANDIDATES = {
    "X_train_encoded": [
        "X_train_encoded",
        "legality_X_train_encoded",
        "notebook53_X_train_encoded",
        "X_train_transformed",
    ],

    "X_validation_encoded": [
        "X_validation_encoded",
        "X_valid_encoded",
        "X_val_encoded",
        "legality_X_validation_encoded",
        "notebook53_X_validation_encoded",
    ],

    "X_test_encoded": [
        "X_test_encoded",
        "legality_X_test_encoded",
        "notebook53_X_test_encoded",
        "X_test_transformed",
    ],
}


def resolve_section2d_runtime_object(
    candidate_names,
):
    """
    Return the first matching runtime object and its variable name.
    """

    for candidate_name in candidate_names:

        if candidate_name in globals():

            candidate_value = globals()[
                candidate_name
            ]

            if candidate_value is not None:

                return (
                    candidate_name,
                    candidate_value,
                )

    return (
        None,
        None,
    )


section2d_matrix_resolution = {}


for logical_name, candidate_names in (
    SECTION2D_MATRIX_CANDIDATES.items()
):

    resolved_name, resolved_value = (
        resolve_section2d_runtime_object(
            candidate_names
        )
    )


    section2d_matrix_resolution[
        logical_name
    ] = {
        "resolved_name":
            resolved_name,

        "value":
            resolved_value,
    }


print()
print("ENCODED MATRIX RESOLUTION")
print("-" * 100)

for logical_name, resolution in (
    section2d_matrix_resolution.items()
):

    matrix_value = resolution[
        "value"
    ]

    print(
        f"{logical_name:30}: "
        f"{resolution['resolved_name']}"
    )

    if matrix_value is not None:

        print(
            f"{'shape':30}: "
            f"{getattr(matrix_value, 'shape', None)}"
        )


assert (
    section2d_matrix_resolution[
        "X_train_encoded"
    ][
        "value"
    ]
    is not None
), (
    "The encoded training matrix was not recovered."
)


assert (
    section2d_matrix_resolution[
        "X_test_encoded"
    ][
        "value"
    ]
    is not None
), (
    "The encoded test matrix was not recovered."
)


# Validation may have been named differently or not stored separately.
# We will preserve it when available and document its absence otherwise.

X_train_encoded_55 = copy.deepcopy(
    section2d_matrix_resolution[
        "X_train_encoded"
    ][
        "value"
    ]
)


X_validation_encoded_55 = copy.deepcopy(
    section2d_matrix_resolution[
        "X_validation_encoded"
    ][
        "value"
    ]
)


X_test_encoded_55 = copy.deepcopy(
    section2d_matrix_resolution[
        "X_test_encoded"
    ][
        "value"
    ]
)


# --------------------------------------------------------------------------------------
# 3. Resolve exact Notebook 53 target vectors
# --------------------------------------------------------------------------------------

SECTION2D_TARGET_CANDIDATES = {
    "y_train": [
        "y_train",
        "legality_y_train",
        "notebook53_y_train",
        "y_train_labels",
    ],

    "y_validation": [
        "y_validation",
        "y_valid",
        "y_val",
        "legality_y_validation",
        "notebook53_y_validation",
        "y_validation_labels",
    ],

    "y_test": [
        "y_test",
        "legality_y_test",
        "notebook53_y_test",
        "y_test_labels",
    ],
}


section2d_target_resolution = {}


for logical_name, candidate_names in (
    SECTION2D_TARGET_CANDIDATES.items()
):

    resolved_name, resolved_value = (
        resolve_section2d_runtime_object(
            candidate_names
        )
    )


    section2d_target_resolution[
        logical_name
    ] = {
        "resolved_name":
            resolved_name,

        "value":
            resolved_value,
    }


print()
print("TARGET VECTOR RESOLUTION")
print("-" * 100)

for logical_name, resolution in (
    section2d_target_resolution.items()
):

    target_value = resolution[
        "value"
    ]

    print(
        f"{logical_name:30}: "
        f"{resolution['resolved_name']}"
    )

    if target_value is not None:

        print(
            f"{'rows':30}: "
            f"{len(target_value)}"
        )


assert (
    section2d_target_resolution[
        "y_train"
    ][
        "value"
    ]
    is not None
), (
    "The training target vector was not recovered."
)


assert (
    section2d_target_resolution[
        "y_test"
    ][
        "value"
    ]
    is not None
), (
    "The test target vector was not recovered."
)


y_train_55 = pd.Series(
    section2d_target_resolution[
        "y_train"
    ][
        "value"
    ]
).reset_index(
    drop=True
)


y_validation_value_55 = (
    section2d_target_resolution[
        "y_validation"
    ][
        "value"
    ]
)


y_validation_55 = (
    pd.Series(
        y_validation_value_55
    ).reset_index(
        drop=True
    )
    if y_validation_value_55 is not None
    else None
)


y_test_55 = pd.Series(
    section2d_target_resolution[
        "y_test"
    ][
        "value"
    ]
).reset_index(
    drop=True
)


# --------------------------------------------------------------------------------------
# 4. Matrix helper functions
# --------------------------------------------------------------------------------------

def section2d_matrix_shape(
    matrix_value,
):
    """
    Safely return matrix shape.
    """

    if matrix_value is None:

        return None

    return tuple(
        int(
            dimension
        )
        for dimension in matrix_value.shape
    )


def section2d_matrix_to_csr(
    matrix_value,
):
    """
    Convert an encoded matrix to CSR without changing values.
    """

    if matrix_value is None:

        return None

    if sparse.issparse(
        matrix_value
    ):

        return matrix_value.tocsr(
            copy=True
        )

    return sparse.csr_matrix(
        np.asarray(
            matrix_value
        )
    )


def section2d_matrix_hash(
    matrix_value,
):
    """
    Create a deterministic SHA-256 hash for dense or sparse matrices.
    """

    if matrix_value is None:

        return None

    matrix_csr = section2d_matrix_to_csr(
        matrix_value
    )

    digest = hashlib.sha256()

    digest.update(
        np.asarray(
            matrix_csr.shape,
            dtype=np.int64,
        ).tobytes()
    )

    digest.update(
        matrix_csr.data.tobytes()
    )

    digest.update(
        matrix_csr.indices.tobytes()
    )

    digest.update(
        matrix_csr.indptr.tobytes()
    )

    return digest.hexdigest()


def section2d_series_hash(
    series_value,
):
    """
    Create a deterministic SHA-256 hash for a label vector.
    """

    if series_value is None:

        return None

    normalized_values = (
        pd.Series(
            series_value
        )
        .fillna(
            "<NA>"
        )
        .astype(str)
        .tolist()
    )

    encoded_text = "\n".join(
        normalized_values
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded_text
    ).hexdigest()


# --------------------------------------------------------------------------------------
# 5. Convert protected matrices to CSR
# --------------------------------------------------------------------------------------

X_train_encoded_55 = section2d_matrix_to_csr(
    X_train_encoded_55
)


X_validation_encoded_55 = section2d_matrix_to_csr(
    X_validation_encoded_55
)


X_test_encoded_55 = section2d_matrix_to_csr(
    X_test_encoded_55
)


print()
print("PROTECTED MATRIX SHAPES")
print("-" * 100)

print(
    "X_train_encoded_55:",
    section2d_matrix_shape(
        X_train_encoded_55
    ),
)

print(
    "X_validation_encoded_55:",
    section2d_matrix_shape(
        X_validation_encoded_55
    ),
)

print(
    "X_test_encoded_55:",
    section2d_matrix_shape(
        X_test_encoded_55
    ),
)

print(
    "y_train_55:",
    len(
        y_train_55
    ),
)

print(
    "y_validation_55:",
    (
        len(
            y_validation_55
        )
        if y_validation_55 is not None
        else None
    ),
)

print(
    "y_test_55:",
    len(
        y_test_55
    ),
)


# --------------------------------------------------------------------------------------
# 6. Validate matrix and target alignment
# --------------------------------------------------------------------------------------

assert X_train_encoded_55.shape[
    0
] == len(
    y_train_55
), (
    "Training matrix and training labels are misaligned: "
    f"{X_train_encoded_55.shape[0]} versus {len(y_train_55)}."
)


assert X_test_encoded_55.shape[
    0
] == len(
    y_test_55
), (
    "Test matrix and test labels are misaligned: "
    f"{X_test_encoded_55.shape[0]} versus {len(y_test_55)}."
)


if X_validation_encoded_55 is not None:

    assert y_validation_55 is not None, (
        "A validation matrix exists but the validation labels are missing."
    )


    assert X_validation_encoded_55.shape[
        0
    ] == len(
        y_validation_55
    ), (
        "Validation matrix and validation labels are misaligned: "
        f"{X_validation_encoded_55.shape[0]} versus "
        f"{len(y_validation_55)}."
    )


assert X_train_encoded_55.shape[
    1
] == 64


assert X_test_encoded_55.shape[
    1
] == 64


if X_validation_encoded_55 is not None:

    assert X_validation_encoded_55.shape[
        1
    ] == 64


assert len(
    baseline_encoded_feature_names
) == 64


# --------------------------------------------------------------------------------------
# 7. Confirm baseline model compatibility
# --------------------------------------------------------------------------------------

if hasattr(
    baseline_policy_model,
    "n_features_in_",
):

    assert int(
        baseline_policy_model.n_features_in_
    ) == X_train_encoded_55.shape[
        1
    ]


baseline_train_accuracy_55 = float(
    baseline_policy_model.score(
        X_train_encoded_55,
        y_train_55,
    )
)


baseline_test_accuracy_55 = float(
    baseline_policy_model.score(
        X_test_encoded_55,
        y_test_55,
    )
)


baseline_validation_accuracy_55 = (
    float(
        baseline_policy_model.score(
            X_validation_encoded_55,
            y_validation_55,
        )
    )
    if X_validation_encoded_55 is not None
    and y_validation_55 is not None
    else None
)


print()
print("RESTORED BASELINE ACCURACY")
print("-" * 100)

print(
    f"Training accuracy   : "
    f"{baseline_train_accuracy_55:.6f}"
)

print(
    f"Validation accuracy : "
    f"{baseline_validation_accuracy_55}"
)

print(
    f"Test accuracy       : "
    f"{baseline_test_accuracy_55:.6f}"
)


# --------------------------------------------------------------------------------------
# 8. Create class-distribution reports
# --------------------------------------------------------------------------------------

def section2d_label_distribution(
    label_values,
    split_name,
):
    """
    Summarize labels for one split.
    """

    if label_values is None:

        return pd.DataFrame(
            columns=[
                "split_name",
                "label",
                "rows",
                "share",
            ]
        )

    distribution_df = (
        pd.Series(
            label_values
        )
        .fillna(
            "UNKNOWN"
        )
        .astype(str)
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "label"
        )
        .reset_index(
            name="rows"
        )
    )


    distribution_df[
        "split_name"
    ] = split_name


    distribution_df[
        "share"
    ] = (
        distribution_df[
            "rows"
        ]
        /
        distribution_df[
            "rows"
        ].sum()
    )


    return distribution_df[
        [
            "split_name",
            "label",
            "rows",
            "share",
        ]
    ]


section2d_label_distribution_df = pd.concat(
    [
        section2d_label_distribution(
            y_train_55,
            "TRAIN",
        ),

        section2d_label_distribution(
            y_validation_55,
            "VALIDATION",
        ),

        section2d_label_distribution(
            y_test_55,
            "TEST",
        ),
    ],
    ignore_index=True,
)


print()
print("LABEL DISTRIBUTION BY SPLIT")
print("-" * 100)

display(
    section2d_label_distribution_df
)


# --------------------------------------------------------------------------------------
# 9. Build frozen split manifest
# --------------------------------------------------------------------------------------

section2d_split_manifest_df = pd.DataFrame(
    [
        {
            "split_name":
                "TRAIN",

            "matrix_source_name":
                section2d_matrix_resolution[
                    "X_train_encoded"
                ][
                    "resolved_name"
                ],

            "target_source_name":
                section2d_target_resolution[
                    "y_train"
                ][
                    "resolved_name"
                ],

            "rows":
                int(
                    X_train_encoded_55.shape[
                        0
                    ]
                ),

            "columns":
                int(
                    X_train_encoded_55.shape[
                        1
                    ]
                ),

            "matrix_type":
                type(
                    X_train_encoded_55
                ).__name__,

            "matrix_hash":
                section2d_matrix_hash(
                    X_train_encoded_55
                ),

            "target_hash":
                section2d_series_hash(
                    y_train_55
                ),

            "baseline_accuracy":
                baseline_train_accuracy_55,
        },
        {
            "split_name":
                "VALIDATION",

            "matrix_source_name":
                section2d_matrix_resolution[
                    "X_validation_encoded"
                ][
                    "resolved_name"
                ],

            "target_source_name":
                section2d_target_resolution[
                    "y_validation"
                ][
                    "resolved_name"
                ],

            "rows":
                (
                    int(
                        X_validation_encoded_55.shape[
                            0
                        ]
                    )
                    if X_validation_encoded_55
                    is not None
                    else 0
                ),

            "columns":
                (
                    int(
                        X_validation_encoded_55.shape[
                            1
                        ]
                    )
                    if X_validation_encoded_55
                    is not None
                    else 0
                ),

            "matrix_type":
                (
                    type(
                        X_validation_encoded_55
                    ).__name__
                    if X_validation_encoded_55
                    is not None
                    else "NOT_AVAILABLE"
                ),

            "matrix_hash":
                section2d_matrix_hash(
                    X_validation_encoded_55
                ),

            "target_hash":
                section2d_series_hash(
                    y_validation_55
                ),

            "baseline_accuracy":
                baseline_validation_accuracy_55,
        },
        {
            "split_name":
                "TEST",

            "matrix_source_name":
                section2d_matrix_resolution[
                    "X_test_encoded"
                ][
                    "resolved_name"
                ],

            "target_source_name":
                section2d_target_resolution[
                    "y_test"
                ][
                    "resolved_name"
                ],

            "rows":
                int(
                    X_test_encoded_55.shape[
                        0
                    ]
                ),

            "columns":
                int(
                    X_test_encoded_55.shape[
                        1
                    ]
                ),

            "matrix_type":
                type(
                    X_test_encoded_55
                ).__name__,

            "matrix_hash":
                section2d_matrix_hash(
                    X_test_encoded_55
                ),

            "target_hash":
                section2d_series_hash(
                    y_test_55
                ),

            "baseline_accuracy":
                baseline_test_accuracy_55,
        },
    ]
)


print()
print("FROZEN SPLIT MANIFEST")
print("-" * 100)

display(
    section2d_split_manifest_df
)


# --------------------------------------------------------------------------------------
# 10. Freeze the baseline estimator parameters
# --------------------------------------------------------------------------------------

NOTEBOOK55_BASELINE_ESTIMATOR_PARAMETERS = (
    baseline_policy_model.get_params(
        deep=True
    )
)


section2d_estimator_parameter_df = pd.DataFrame(
    [
        {
            "parameter_name":
                parameter_name,

            "parameter_value":
                repr(
                    parameter_value
                ),
        }
        for parameter_name, parameter_value
        in sorted(
            NOTEBOOK55_BASELINE_ESTIMATOR_PARAMETERS.items()
        )
    ]
)


print()
print("FROZEN BASELINE ESTIMATOR PARAMETERS")
print("-" * 100)

display(
    section2d_estimator_parameter_df
)


# --------------------------------------------------------------------------------------
# 11. Create strategy-specific matrix views
# --------------------------------------------------------------------------------------

NOTEBOOK55_STRATEGY_TRAIN_MATRICES = {}
NOTEBOOK55_STRATEGY_VALIDATION_MATRICES = {}
NOTEBOOK55_STRATEGY_TEST_MATRICES = {}


section2d_strategy_matrix_rows = []


for strategy_id in NOTEBOOK55_PRIMARY_STRATEGIES:

    retained_indices = (
        NOTEBOOK55_STRATEGY_FEATURE_INDICES[
            strategy_id
        ]
    )


    assert len(
        retained_indices
    ) > 0


    strategy_train_matrix = (
        X_train_encoded_55[
            :,
            retained_indices,
        ]
        .tocsr()
    )


    strategy_validation_matrix = (
        X_validation_encoded_55[
            :,
            retained_indices,
        ]
        .tocsr()
        if X_validation_encoded_55
        is not None
        else None
    )


    strategy_test_matrix = (
        X_test_encoded_55[
            :,
            retained_indices,
        ]
        .tocsr()
    )


    NOTEBOOK55_STRATEGY_TRAIN_MATRICES[
        strategy_id
    ] = strategy_train_matrix


    NOTEBOOK55_STRATEGY_VALIDATION_MATRICES[
        strategy_id
    ] = strategy_validation_matrix


    NOTEBOOK55_STRATEGY_TEST_MATRICES[
        strategy_id
    ] = strategy_test_matrix


    section2d_strategy_matrix_rows.append(
        {
            "strategy_id":
                strategy_id,

            "retained_features":
                int(
                    len(
                        retained_indices
                    )
                ),

            "train_rows":
                int(
                    strategy_train_matrix.shape[
                        0
                    ]
                ),

            "train_columns":
                int(
                    strategy_train_matrix.shape[
                        1
                    ]
                ),

            "validation_rows":
                (
                    int(
                        strategy_validation_matrix.shape[
                            0
                        ]
                    )
                    if strategy_validation_matrix
                    is not None
                    else 0
                ),

            "validation_columns":
                (
                    int(
                        strategy_validation_matrix.shape[
                            1
                        ]
                    )
                    if strategy_validation_matrix
                    is not None
                    else 0
                ),

            "test_rows":
                int(
                    strategy_test_matrix.shape[
                        0
                    ]
                ),

            "test_columns":
                int(
                    strategy_test_matrix.shape[
                        1
                    ]
                ),
        }
    )


section2d_strategy_matrix_profile_df = pd.DataFrame(
    section2d_strategy_matrix_rows
)


print()
print("STRATEGY MATRIX PROFILE")
print("-" * 100)

display(
    section2d_strategy_matrix_profile_df
)


assert len(
    section2d_strategy_matrix_profile_df
) == 4


# --------------------------------------------------------------------------------------
# 12. Summary
# --------------------------------------------------------------------------------------

section2d_summary = {
    "status":
        "TRAINING_VALIDATION_TEST_MATRICES_RECOVERED_AND_FROZEN",

    "train_rows":
        int(
            X_train_encoded_55.shape[
                0
            ]
        ),

    "validation_rows":
        (
            int(
                X_validation_encoded_55.shape[
                    0
                ]
            )
            if X_validation_encoded_55
            is not None
            else 0
        ),

    "test_rows":
        int(
            X_test_encoded_55.shape[
                0
            ]
        ),

    "baseline_feature_count":
        int(
            X_train_encoded_55.shape[
                1
            ]
        ),

    "model_classes":
        [
            str(
                class_name
            )
            for class_name in baseline_policy_model.classes_
        ],

    "baseline_train_accuracy":
        baseline_train_accuracy_55,

    "baseline_validation_accuracy":
        baseline_validation_accuracy_55,

    "baseline_test_accuracy":
        baseline_test_accuracy_55,

    "fixed_training_rows":
        True,

    "fixed_validation_rows":
        True,

    "fixed_test_rows":
        True,

    "fixed_estimator_parameters":
        True,

    "matrix_hashes_created":
        True,

    "strategy_matrix_views_created":
        True,

    "next_stage":
        "TRAIN_CONTROLLED_FEATURE_ABLATION_MODELS",
}


print()
print("SECTION 2D MATRIX RECOVERY SUMMARY")
print("-" * 100)

for key, value in (
    section2d_summary.items()
):

    print(
        f"{key:58}: {value}"
    )


# --------------------------------------------------------------------------------------
# 13. Validation checks
# --------------------------------------------------------------------------------------

section2d_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "training_matrix_recovered",

            "passed":
                X_train_encoded_55
                is not None,

            "value":
                section2d_matrix_shape(
                    X_train_encoded_55
                ),

            "expected":
                "Nonempty matrix",
        },
        {
            "check":
                "test_matrix_recovered",

            "passed":
                X_test_encoded_55
                is not None,

            "value":
                section2d_matrix_shape(
                    X_test_encoded_55
                ),

            "expected":
                "Nonempty matrix",
        },
        {
            "check":
                "training_rows_aligned",

            "passed":
                X_train_encoded_55.shape[
                    0
                ] == len(
                    y_train_55
                ),

            "value":
                {
                    "matrix_rows":
                        X_train_encoded_55.shape[
                            0
                        ],

                    "label_rows":
                        len(
                            y_train_55
                        ),
                },

            "expected":
                "Equal",
        },
        {
            "check":
                "test_rows_aligned",

            "passed":
                X_test_encoded_55.shape[
                    0
                ] == len(
                    y_test_55
                ),

            "value":
                {
                    "matrix_rows":
                        X_test_encoded_55.shape[
                            0
                        ],

                    "label_rows":
                        len(
                            y_test_55
                        ),
                },

            "expected":
                "Equal",
        },
        {
            "check":
                "baseline_64_features_recovered",

            "passed":
                X_train_encoded_55.shape[
                    1
                ] == 64,

            "value":
                X_train_encoded_55.shape[
                    1
                ],

            "expected":
                64,
        },
        {
            "check":
                "baseline_model_compatible",

            "passed":
                (
                    not hasattr(
                        baseline_policy_model,
                        "n_features_in_",
                    )
                    or
                    int(
                        baseline_policy_model.n_features_in_
                    )
                    ==
                    X_train_encoded_55.shape[
                        1
                    ]
                ),

            "value":
                getattr(
                    baseline_policy_model,
                    "n_features_in_",
                    None,
                ),

            "expected":
                64,
        },
        {
            "check":
                "four_strategy_matrix_views_created",

            "passed":
                (
                    len(
                        NOTEBOOK55_STRATEGY_TRAIN_MATRICES
                    ) == 4
                    and
                    len(
                        NOTEBOOK55_STRATEGY_TEST_MATRICES
                    ) == 4
                ),

            "value":
                len(
                    NOTEBOOK55_STRATEGY_TRAIN_MATRICES
                ),

            "expected":
                4,
        },
        {
            "check":
                "matrix_hashes_complete",

            "passed":
                section2d_split_manifest_df.loc[
                    section2d_split_manifest_df[
                        "split_name"
                    ].isin(
                        [
                            "TRAIN",
                            "TEST",
                        ]
                    ),
                    "matrix_hash",
                ].notna().all(),

            "value":
                True,

            "expected":
                True,
        },
        {
            "check":
                "next_stage_recognized",

            "passed":
                section2d_summary[
                    "next_stage"
                ]
                ==
                "TRAIN_CONTROLLED_FEATURE_ABLATION_MODELS",

            "value":
                section2d_summary[
                    "next_stage"
                ],

            "expected":
                "TRAIN_CONTROLLED_FEATURE_ABLATION_MODELS",
        },
    ]
)


print()
print("SECTION 2D VALIDATION CHECKS")
print("-" * 100)

display(
    section2d_validation_checks_df
)


section2d_failed_checks = int(
    (
        ~section2d_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section2d_failed_checks == 0, (
    "One or more Section 2D validation checks failed."
)


# --------------------------------------------------------------------------------------
# 14. Save Section 2D reports and protected artifacts
# --------------------------------------------------------------------------------------

SECTION2D_SPLIT_MANIFEST_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_frozen_split_manifest.csv"
)

SECTION2D_LABEL_DISTRIBUTION_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_label_distribution_by_split.csv"
)

SECTION2D_ESTIMATOR_PARAMETERS_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_baseline_estimator_parameters.csv"
)

SECTION2D_STRATEGY_MATRIX_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_strategy_matrix_profile.csv"
)

SECTION2D_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_validation_checks.csv"
)

SECTION2D_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section2d_matrix_recovery_summary.json"
)

SECTION2D_FROZEN_DATA_FILE = (
    ARTIFACTS_DIRECTORY
    /
    "section2d_frozen_training_split.joblib"
)


section2d_split_manifest_df.to_csv(
    SECTION2D_SPLIT_MANIFEST_FILE,
    index=False,
)

section2d_label_distribution_df.to_csv(
    SECTION2D_LABEL_DISTRIBUTION_FILE,
    index=False,
)

section2d_estimator_parameter_df.to_csv(
    SECTION2D_ESTIMATOR_PARAMETERS_FILE,
    index=False,
)

section2d_strategy_matrix_profile_df.to_csv(
    SECTION2D_STRATEGY_MATRIX_PROFILE_FILE,
    index=False,
)

section2d_validation_checks_df.to_csv(
    SECTION2D_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION2D_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section2d_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


joblib.dump(
    {
        "X_train_encoded":
            X_train_encoded_55,

        "X_validation_encoded":
            X_validation_encoded_55,

        "X_test_encoded":
            X_test_encoded_55,

        "y_train":
            y_train_55,

        "y_validation":
            y_validation_55,

        "y_test":
            y_test_55,

        "encoded_feature_names":
            list(
                baseline_encoded_feature_names
            ),

        "strategy_feature_indices":
            copy.deepcopy(
                NOTEBOOK55_STRATEGY_FEATURE_INDICES
            ),

        "strategy_feature_names":
            copy.deepcopy(
                NOTEBOOK55_STRATEGY_FEATURE_NAMES
            ),

        "baseline_estimator_parameters":
            copy.deepcopy(
                NOTEBOOK55_BASELINE_ESTIMATOR_PARAMETERS
            ),

        "split_manifest":
            section2d_split_manifest_df.copy(),
    },
    SECTION2D_FROZEN_DATA_FILE,
    compress=3,
)


section2d_saved_files = [
    SECTION2D_SPLIT_MANIFEST_FILE,
    SECTION2D_LABEL_DISTRIBUTION_FILE,
    SECTION2D_ESTIMATOR_PARAMETERS_FILE,
    SECTION2D_STRATEGY_MATRIX_PROFILE_FILE,
    SECTION2D_VALIDATION_FILE,
    SECTION2D_SUMMARY_FILE,
    SECTION2D_FROZEN_DATA_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section2d_saved_files
)


print()
print("SAVED SECTION 2D REPORTS")
print("-" * 100)

for file_path in section2d_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 2D TRAINING, VALIDATION, AND TEST "
    "MATRICES RECOVERED AND FROZEN"
)


# In[16]:


# ======================================================================================
# SECTION 3A — TRAIN CONTROLLED FEATURE-ABLATION MODELS
# ======================================================================================

print("=" * 100)
print("SECTION 3A — TRAIN CONTROLLED FEATURE-ABLATION MODELS")
print("=" * 100)

from pathlib import Path
from copy import deepcopy
import json
import time
import joblib

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


# --------------------------------------------------------------------------------------
# 1. Validate dependencies
# --------------------------------------------------------------------------------------

SECTION3A_REQUIRED_OBJECTS = [
    "baseline_policy_model",
    "NOTEBOOK55_PRIMARY_STRATEGIES",
    "NOTEBOOK55_STRATEGY_TRAIN_MATRICES",
    "NOTEBOOK55_STRATEGY_VALIDATION_MATRICES",
    "NOTEBOOK55_STRATEGY_TEST_MATRICES",
    "NOTEBOOK55_STRATEGY_FEATURE_NAMES",
    "NOTEBOOK55_BASELINE_ESTIMATOR_PARAMETERS",
    "X_train_encoded_55",
    "X_test_encoded_55",
    "y_train_55",
    "y_test_55",
    "section2d_summary",
    "REPORTS_DIRECTORY",
    "MODELS_DIRECTORY",
]


section3a_missing_objects = [
    object_name
    for object_name in SECTION3A_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3A OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3A_REQUIRED_OBJECTS:
    print(
        f"{object_name:52}: "
        f"{object_name in globals()}"
    )


assert not section3a_missing_objects, (
    "Section 3A required objects are missing: "
    f"{section3a_missing_objects}"
)


assert (
    section2d_summary[
        "next_stage"
    ]
    ==
    "TRAIN_CONTROLLED_FEATURE_ABLATION_MODELS"
)


assert len(
    NOTEBOOK55_PRIMARY_STRATEGIES
) == 4


# --------------------------------------------------------------------------------------
# 2. Helper functions
# --------------------------------------------------------------------------------------

def section3a_predict_probabilities(
    model,
    matrix,
):
    """
    Return class probabilities when supported.
    """

    if matrix is None:
        return None

    if not hasattr(
        model,
        "predict_proba",
    ):
        return None

    return np.asarray(
        model.predict_proba(
            matrix
        )
    )


def section3a_safe_log_loss(
    y_true,
    probabilities,
    class_labels,
):
    """
    Compute multiclass log loss safely.
    """

    if probabilities is None:
        return None

    try:
        return float(
            log_loss(
                y_true,
                probabilities,
                labels=class_labels,
            )
        )

    except Exception:
        return None


def section3a_evaluate_split(
    model,
    matrix,
    labels,
    split_name,
):
    """
    Evaluate one fitted model on one split.
    """

    if matrix is None or labels is None:
        return {
            "split_name":
                split_name,

            "rows":
                0,

            "accuracy":
                None,

            "balanced_accuracy":
                None,

            "macro_f1":
                None,

            "weighted_f1":
                None,

            "log_loss":
                None,
        }

    predictions = model.predict(
        matrix
    )

    probabilities = section3a_predict_probabilities(
        model,
        matrix,
    )

    return {
        "split_name":
            split_name,

        "rows":
            int(
                matrix.shape[
                    0
                ]
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "log_loss":
            section3a_safe_log_loss(
                labels,
                probabilities,
                list(
                    model.classes_
                ),
            ),
    }


# --------------------------------------------------------------------------------------
# 3. Train the four controlled strategies
# --------------------------------------------------------------------------------------

NOTEBOOK55_TRAINED_MODELS = {}
NOTEBOOK55_MODEL_TRAINING_RESULTS = {}
NOTEBOOK55_MODEL_PREDICTIONS = {}


section3a_training_rows = []
section3a_split_metric_rows = []
section3a_classification_rows = []
section3a_confusion_rows = []


for strategy_id in NOTEBOOK55_PRIMARY_STRATEGIES:

    print()
    print("=" * 100)
    print(f"TRAINING STRATEGY: {strategy_id}")
    print("=" * 100)

    X_train_strategy = (
        NOTEBOOK55_STRATEGY_TRAIN_MATRICES[
            strategy_id
        ]
    )

    X_validation_strategy = (
        NOTEBOOK55_STRATEGY_VALIDATION_MATRICES[
            strategy_id
        ]
    )

    X_test_strategy = (
        NOTEBOOK55_STRATEGY_TEST_MATRICES[
            strategy_id
        ]
    )

    retained_feature_names = (
        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
            strategy_id
        ]
    )

    assert X_train_strategy.shape[
        1
    ] == len(
        retained_feature_names
    )

    strategy_model = clone(
        baseline_policy_model
    )

    # Ensure identical estimator configuration.
    strategy_model.set_params(
        **deepcopy(
            NOTEBOOK55_BASELINE_ESTIMATOR_PARAMETERS
        )
    )

    training_start = time.perf_counter()

    strategy_model.fit(
        X_train_strategy,
        y_train_55,
    )

    training_seconds = float(
        time.perf_counter()
        -
        training_start
    )

    NOTEBOOK55_TRAINED_MODELS[
        strategy_id
    ] = strategy_model

    train_metrics = section3a_evaluate_split(
        strategy_model,
        X_train_strategy,
        y_train_55,
        "TRAIN",
    )

    validation_metrics = section3a_evaluate_split(
        strategy_model,
        X_validation_strategy,
        y_validation_55,
        "VALIDATION",
    )

    test_metrics = section3a_evaluate_split(
        strategy_model,
        X_test_strategy,
        y_test_55,
        "TEST",
    )

    for metric_record in [
        train_metrics,
        validation_metrics,
        test_metrics,
    ]:
        section3a_split_metric_rows.append(
            {
                "strategy_id":
                    strategy_id,

                "retained_feature_count":
                    int(
                        X_train_strategy.shape[
                            1
                        ]
                    ),

                **metric_record,
            }
        )

    test_predictions = strategy_model.predict(
        X_test_strategy
    )

    test_probabilities = section3a_predict_probabilities(
        strategy_model,
        X_test_strategy,
    )

    NOTEBOOK55_MODEL_PREDICTIONS[
        strategy_id
    ] = {
        "test_predictions":
            np.asarray(
                test_predictions
            ),

        "test_probabilities":
            (
                np.asarray(
                    test_probabilities
                )
                if test_probabilities
                is not None
                else None
            ),
    }

    classification_dictionary = classification_report(
        y_test_55,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )

    for class_name, class_metrics in (
        classification_dictionary.items()
    ):
        if not isinstance(
            class_metrics,
            dict,
        ):
            continue

        section3a_classification_rows.append(
            {
                "strategy_id":
                    strategy_id,

                "class_name":
                    str(
                        class_name
                    ),

                "precision":
                    class_metrics.get(
                        "precision"
                    ),

                "recall":
                    class_metrics.get(
                        "recall"
                    ),

                "f1_score":
                    class_metrics.get(
                        "f1-score"
                    ),

                "support":
                    class_metrics.get(
                        "support"
                    ),
            }
        )

    confusion = confusion_matrix(
        y_test_55,
        test_predictions,
        labels=list(
            strategy_model.classes_
        ),
    )

    for actual_index, actual_label in enumerate(
        strategy_model.classes_
    ):
        for predicted_index, predicted_label in enumerate(
            strategy_model.classes_
        ):
            section3a_confusion_rows.append(
                {
                    "strategy_id":
                        strategy_id,

                    "actual_label":
                        str(
                            actual_label
                        ),

                    "predicted_label":
                        str(
                            predicted_label
                        ),

                    "count":
                        int(
                            confusion[
                                actual_index,
                                predicted_index,
                            ]
                        ),
                }
            )

    section3a_training_rows.append(
        {
            "strategy_id":
                strategy_id,

            "model_type":
                type(
                    strategy_model
                ).__name__,

            "retained_feature_count":
                int(
                    X_train_strategy.shape[
                        1
                    ]
                ),

            "removed_feature_count":
                int(
                    64
                    -
                    X_train_strategy.shape[
                        1
                    ]
                ),

            "training_rows":
                int(
                    X_train_strategy.shape[
                        0
                    ]
                ),

            "training_seconds":
                training_seconds,

            "class_count":
                int(
                    len(
                        strategy_model.classes_
                    )
                ),

            "classes":
                json.dumps(
                    [
                        str(
                            class_name
                        )
                        for class_name in strategy_model.classes_
                    ]
                ),

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "test_accuracy":
                test_metrics[
                    "accuracy"
                ],

            "test_balanced_accuracy":
                test_metrics[
                    "balanced_accuracy"
                ],

            "test_macro_f1":
                test_metrics[
                    "macro_f1"
                ],

            "test_weighted_f1":
                test_metrics[
                    "weighted_f1"
                ],

            "test_log_loss":
                test_metrics[
                    "log_loss"
                ],
        }
    )

    NOTEBOOK55_MODEL_TRAINING_RESULTS[
        strategy_id
    ] = {
        "training_seconds":
            training_seconds,

        "train_metrics":
            train_metrics,

        "validation_metrics":
            validation_metrics,

        "test_metrics":
            test_metrics,

        "retained_feature_names":
            list(
                retained_feature_names
            ),
    }

    print(
        "Retained features:",
        X_train_strategy.shape[
            1
        ],
    )

    print(
        "Train accuracy:",
        f"{train_metrics['accuracy']:.6f}",
    )

    print(
        "Validation accuracy:",
        validation_metrics[
            "accuracy"
        ],
    )

    print(
        "Test accuracy:",
        f"{test_metrics['accuracy']:.6f}",
    )

    print(
        "Training seconds:",
        f"{training_seconds:.3f}",
    )


# --------------------------------------------------------------------------------------
# 4. Build result tables
# --------------------------------------------------------------------------------------

section3a_model_training_summary_df = pd.DataFrame(
    section3a_training_rows
)


section3a_split_metrics_df = pd.DataFrame(
    section3a_split_metric_rows
)


section3a_classification_report_df = pd.DataFrame(
    section3a_classification_rows
)


section3a_confusion_matrix_df = pd.DataFrame(
    section3a_confusion_rows
)


print()
print("CONTROLLED MODEL TRAINING SUMMARY")
print("-" * 100)

display(
    section3a_model_training_summary_df
)


print()
print("SPLIT METRICS")
print("-" * 100)

display(
    section3a_split_metrics_df
)


# --------------------------------------------------------------------------------------
# 5. Compare every strategy with R0 baseline
# --------------------------------------------------------------------------------------

r0_result_row_3a = (
    section3a_model_training_summary_df.loc[
        section3a_model_training_summary_df[
            "strategy_id"
        ].eq(
            "R0_CURRENT_BASELINE"
        )
    ]
    .iloc[0]
)


section3a_strategy_comparison_df = (
    section3a_model_training_summary_df
    .copy()
)


for metric_name in [
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_log_loss",
]:

    baseline_value = r0_result_row_3a[
        metric_name
    ]

    section3a_strategy_comparison_df[
        f"{metric_name}_minus_r0"
    ] = (
        section3a_strategy_comparison_df[
            metric_name
        ]
        -
        baseline_value
    )


section3a_strategy_comparison_df[
    "validation_accuracy_drop_from_r0"
] = (
    r0_result_row_3a[
        "validation_accuracy"
    ]
    -
    section3a_strategy_comparison_df[
        "validation_accuracy"
    ]
)


section3a_strategy_comparison_df[
    "test_accuracy_drop_from_r0"
] = (
    r0_result_row_3a[
        "test_accuracy"
    ]
    -
    section3a_strategy_comparison_df[
        "test_accuracy"
    ]
)


print()
print("STRATEGY COMPARISON AGAINST R0")
print("-" * 100)

display(
    section3a_strategy_comparison_df
)


# --------------------------------------------------------------------------------------
# 6. Validate R0 reproducibility
# --------------------------------------------------------------------------------------

r0_retrained_model_3a = (
    NOTEBOOK55_TRAINED_MODELS[
        "R0_CURRENT_BASELINE"
    ]
)


r0_test_predictions_3a = (
    NOTEBOOK55_MODEL_PREDICTIONS[
        "R0_CURRENT_BASELINE"
    ][
        "test_predictions"
    ]
)


original_baseline_test_predictions_3a = (
    baseline_policy_model.predict(
        X_test_encoded_55
    )
)


r0_prediction_agreement_with_original_3a = float(
    np.mean(
        r0_test_predictions_3a
        ==
        original_baseline_test_predictions_3a
    )
)


r0_retrained_test_accuracy_3a = float(
    r0_result_row_3a[
        "test_accuracy"
    ]
)


r0_accuracy_difference_from_original_3a = float(
    r0_retrained_test_accuracy_3a
    -
    baseline_test_accuracy_55
)


print()
print("R0 BASELINE REPRODUCIBILITY")
print("-" * 100)

print(
    "Prediction agreement with original:",
    r0_prediction_agreement_with_original_3a,
)

print(
    "Test accuracy difference:",
    r0_accuracy_difference_from_original_3a,
)


# --------------------------------------------------------------------------------------
# 7. Validation checks
# --------------------------------------------------------------------------------------

section3a_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "four_models_trained",

            "passed":
                len(
                    NOTEBOOK55_TRAINED_MODELS
                ) == 4,

            "value":
                len(
                    NOTEBOOK55_TRAINED_MODELS
                ),

            "expected":
                4,
        },
        {
            "check":
                "all_models_fitted",

            "passed":
                all(
                    hasattr(
                        model,
                        "classes_",
                    )
                    for model
                    in NOTEBOOK55_TRAINED_MODELS.values()
                ),

            "value":
                True,

            "expected":
                True,
        },
        {
            "check":
                "all_train_rows_identical",

            "passed":
                section3a_model_training_summary_df[
                    "training_rows"
                ].eq(
                    len(
                        y_train_55
                    )
                ).all(),

            "value":
                section3a_model_training_summary_df[
                    "training_rows"
                ].unique().tolist(),

            "expected":
                [
                    len(
                        y_train_55
                    )
                ],
        },
        {
            "check":
                "all_test_metrics_available",

            "passed":
                section3a_model_training_summary_df[
                    "test_accuracy"
                ].notna().all(),

            "value":
                int(
                    section3a_model_training_summary_df[
                        "test_accuracy"
                    ].notna().sum()
                ),

            "expected":
                4,
        },
        {
            "check":
                "r0_reproduces_original_predictions",

            "passed":
                r0_prediction_agreement_with_original_3a
                >= 0.99,

            "value":
                r0_prediction_agreement_with_original_3a,

            "expected":
                "At least 0.99",
        },
        {
            "check":
                "r0_accuracy_reproduced",

            "passed":
                abs(
                    r0_accuracy_difference_from_original_3a
                )
                <= 1e-12,

            "value":
                r0_accuracy_difference_from_original_3a,

            "expected":
                0.0,
        },
        {
            "check":
                "strategy_feature_counts_match_partition",

            "passed":
                all(
                    int(
                        section3a_model_training_summary_df.loc[
                            section3a_model_training_summary_df[
                                "strategy_id"
                            ].eq(
                                strategy_id
                            ),
                            "retained_feature_count",
                        ].iloc[0]
                    )
                    ==
                    len(
                        NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                            strategy_id
                        ]
                    )
                    for strategy_id
                    in NOTEBOOK55_PRIMARY_STRATEGIES
                ),

            "value":
                True,

            "expected":
                True,
        },
    ]
)


print()
print("SECTION 3A VALIDATION CHECKS")
print("-" * 100)

display(
    section3a_validation_checks_df
)


section3a_failed_checks = int(
    (
        ~section3a_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section3a_failed_checks == 0, (
    "One or more Section 3A validation checks failed."
)


# --------------------------------------------------------------------------------------
# 8. Summary
# --------------------------------------------------------------------------------------

section3a_best_test_strategy_row = (
    section3a_model_training_summary_df
    .sort_values(
        [
            "test_accuracy",
            "test_macro_f1",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iloc[0]
)


section3a_summary = {
    "status":
        "CONTROLLED_FEATURE_ABLATION_MODELS_TRAINED",

    "models_trained":
        int(
            len(
                NOTEBOOK55_TRAINED_MODELS
            )
        ),

    "baseline_strategy":
        "R0_CURRENT_BASELINE",

    "primary_remediation_strategy":
        "R1_REMOVE_LEGAL_SIGNATURE",

    "r0_prediction_agreement_with_original":
        r0_prediction_agreement_with_original_3a,

    "r0_accuracy_difference_from_original":
        r0_accuracy_difference_from_original_3a,

    "best_test_accuracy_strategy":
        str(
            section3a_best_test_strategy_row[
                "strategy_id"
            ]
        ),

    "best_test_accuracy":
        float(
            section3a_best_test_strategy_row[
                "test_accuracy"
            ]
        ),

    "next_stage":
        "EVALUATE_MOVE_DIVERSITY_AND_QUICK_ATTACK_ASCENSION_BEHAVIOR",
}


print()
print("SECTION 3A TRAINING SUMMARY")
print("-" * 100)

for key, value in section3a_summary.items():
    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 9. Save reports and models
# --------------------------------------------------------------------------------------

SECTION3A_MODEL_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_model_training_summary.csv"
)

SECTION3A_SPLIT_METRICS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_split_metrics.csv"
)

SECTION3A_STRATEGY_COMPARISON_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_strategy_comparison.csv"
)

SECTION3A_CLASSIFICATION_REPORT_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_classification_report.csv"
)

SECTION3A_CONFUSION_MATRIX_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_confusion_matrix.csv"
)

SECTION3A_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_validation_checks.csv"
)

SECTION3A_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3a_training_summary.json"
)


section3a_model_training_summary_df.to_csv(
    SECTION3A_MODEL_SUMMARY_FILE,
    index=False,
)

section3a_split_metrics_df.to_csv(
    SECTION3A_SPLIT_METRICS_FILE,
    index=False,
)

section3a_strategy_comparison_df.to_csv(
    SECTION3A_STRATEGY_COMPARISON_FILE,
    index=False,
)

section3a_classification_report_df.to_csv(
    SECTION3A_CLASSIFICATION_REPORT_FILE,
    index=False,
)

section3a_confusion_matrix_df.to_csv(
    SECTION3A_CONFUSION_MATRIX_FILE,
    index=False,
)

section3a_validation_checks_df.to_csv(
    SECTION3A_VALIDATION_FILE,
    index=False,
)


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


section3a_model_files = {}


for strategy_id, strategy_model in (
    NOTEBOOK55_TRAINED_MODELS.items()
):

    model_file = (
        MODELS_DIRECTORY
        /
        f"section3a_{strategy_id.lower()}_model.joblib"
    )

    joblib.dump(
        {
            "strategy_id":
                strategy_id,

            "model":
                strategy_model,

            "retained_feature_names":
                NOTEBOOK55_STRATEGY_FEATURE_NAMES[
                    strategy_id
                ],

            "retained_feature_indices":
                NOTEBOOK55_STRATEGY_FEATURE_INDICES[
                    strategy_id
                ],

            "training_results":
                NOTEBOOK55_MODEL_TRAINING_RESULTS[
                    strategy_id
                ],
        },
        model_file,
        compress=3,
    )

    section3a_model_files[
        strategy_id
    ] = model_file


section3a_saved_files = [
    SECTION3A_MODEL_SUMMARY_FILE,
    SECTION3A_SPLIT_METRICS_FILE,
    SECTION3A_STRATEGY_COMPARISON_FILE,
    SECTION3A_CLASSIFICATION_REPORT_FILE,
    SECTION3A_CONFUSION_MATRIX_FILE,
    SECTION3A_VALIDATION_FILE,
    SECTION3A_SUMMARY_FILE,
    *section3a_model_files.values(),
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3a_saved_files
)


print()
print("SAVED SECTION 3A REPORTS AND MODELS")
print("-" * 100)

for file_path in section3a_saved_files:
    print(file_path)


print()
print(
    "✅ SECTION 3A CONTROLLED FEATURE-ABLATION MODELS TRAINED"
)


# In[17]:


# ======================================================================================
# SECTION 3B-A — PREPARE BALANCED MULTI-ACTION POLICY EVALUATION
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A — PREPARE BALANCED MULTI-ACTION POLICY EVALUATION")
print("=" * 100)

from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate Section 3A dependencies
# --------------------------------------------------------------------------------------

SECTION3BA_REQUIRED_OBJECTS = [
    "NOTEBOOK55_TRAINED_MODELS",
    "NOTEBOOK55_PRIMARY_STRATEGIES",
    "NOTEBOOK55_STRATEGY_FEATURE_INDICES",
    "NOTEBOOK55_STRATEGY_FEATURE_NAMES",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "section3a_summary",
    "NOTEBOOK54_SECTION4_REPORT_DIRECTORY",
    "REPORTS_DIRECTORY",
]


section3ba_missing_objects = [
    object_name
    for object_name in SECTION3BA_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3B-A OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3BA_REQUIRED_OBJECTS:
    print(
        f"{object_name:52}: "
        f"{object_name in globals()}"
    )


assert not section3ba_missing_objects, (
    "Section 3B-A required objects are missing: "
    f"{section3ba_missing_objects}"
)


assert (
    section3a_summary[
        "next_stage"
    ]
    ==
    "EVALUATE_MOVE_DIVERSITY_AND_QUICK_ATTACK_ASCENSION_BEHAVIOR"
)


assert len(
    NOTEBOOK55_TRAINED_MODELS
) == 4


# --------------------------------------------------------------------------------------
# 2. Locate Notebook 54 balanced evaluation reports
# --------------------------------------------------------------------------------------

SECTION3BA_CANDIDATE_REPORTS = {
    "expanded_case_results":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4wf_expanded_case_results.csv",

    "expanded_case_errors":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4wf_expanded_case_errors.csv",

    "expanded_paired_results":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4wf_expanded_paired_results.csv",

    "variant_summary":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4wf_variant_evaluation_summary.csv",

    "move_distribution":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4wf_move_distribution_by_variant_and_side.csv",

    "score_cases":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whb_case_probability_scores.csv",

    "counterfactual_cases":
        NOTEBOOK54_SECTION4_REPORT_DIRECTORY
        /
        "section4whe_counterfactual_case_results.csv",
}


section3ba_report_inventory_rows = []


for report_name, report_path in (
    SECTION3BA_CANDIDATE_REPORTS.items()
):

    section3ba_report_inventory_rows.append(
        {
            "report_name":
                report_name,

            "report_path":
                str(
                    report_path
                ),

            "exists":
                report_path.exists(),

            "size_bytes":
                (
                    int(
                        report_path.stat().st_size
                    )
                    if report_path.exists()
                    else 0
                ),
        }
    )


section3ba_report_inventory_df = pd.DataFrame(
    section3ba_report_inventory_rows
)


print()
print("NOTEBOOK 54 BALANCED-EVALUATION REPORT INVENTORY")
print("-" * 100)

display(
    section3ba_report_inventory_df
)


assert section3ba_report_inventory_df[
    "exists"
].astype(bool).all(), (
    "One or more required Notebook 54 evaluation reports are missing."
)


assert section3ba_report_inventory_df[
    "size_bytes"
].gt(0).all()


# --------------------------------------------------------------------------------------
# 3. Load the evaluation reports
# --------------------------------------------------------------------------------------

section3ba_expanded_case_results_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "expanded_case_results"
    ]
)


section3ba_expanded_case_errors_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "expanded_case_errors"
    ]
)


section3ba_expanded_paired_results_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "expanded_paired_results"
    ]
)


section3ba_variant_summary_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "variant_summary"
    ]
)


section3ba_move_distribution_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "move_distribution"
    ]
)


section3ba_score_cases_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "score_cases"
    ]
)


section3ba_counterfactual_cases_df = pd.read_csv(
    SECTION3BA_CANDIDATE_REPORTS[
        "counterfactual_cases"
    ]
)


# --------------------------------------------------------------------------------------
# 4. Profile each report
# --------------------------------------------------------------------------------------

section3ba_dataframe_objects = {
    "expanded_case_results":
        section3ba_expanded_case_results_df,

    "expanded_case_errors":
        section3ba_expanded_case_errors_df,

    "expanded_paired_results":
        section3ba_expanded_paired_results_df,

    "variant_summary":
        section3ba_variant_summary_df,

    "move_distribution":
        section3ba_move_distribution_df,

    "score_cases":
        section3ba_score_cases_df,

    "counterfactual_cases":
        section3ba_counterfactual_cases_df,
}


section3ba_dataframe_profile_rows = []


for dataframe_name, dataframe_value in (
    section3ba_dataframe_objects.items()
):

    section3ba_dataframe_profile_rows.append(
        {
            "dataframe_name":
                dataframe_name,

            "rows":
                int(
                    len(
                        dataframe_value
                    )
                ),

            "columns":
                int(
                    len(
                        dataframe_value.columns
                    )
                ),

            "column_names":
                json.dumps(
                    dataframe_value.columns.tolist()
                ),
        }
    )


section3ba_dataframe_profile_df = pd.DataFrame(
    section3ba_dataframe_profile_rows
)


print()
print("BALANCED-EVALUATION DATAFRAME PROFILE")
print("-" * 100)

display(
    section3ba_dataframe_profile_df
)


# --------------------------------------------------------------------------------------
# 5. Display exact columns from the two most important sources
# --------------------------------------------------------------------------------------

print()
print("EXPANDED CASE RESULT COLUMNS")
print("-" * 100)

print(
    section3ba_expanded_case_results_df.columns.tolist()
)


print()
print("SCORE CASE COLUMNS")
print("-" * 100)

print(
    section3ba_score_cases_df.columns.tolist()
)


print()
print("COUNTERFACTUAL CASE COLUMNS")
print("-" * 100)

print(
    section3ba_counterfactual_cases_df.columns.tolist()
)


# --------------------------------------------------------------------------------------
# 6. Profile runtime reconstruction helpers
# --------------------------------------------------------------------------------------

SECTION3BA_RUNTIME_HELPERS = [
    "build_section4wf_evaluation_state",
    "build_legality_aware_feature_row",
    "get_state_value",
    "get_card_name",
    "get_attached_energy",
    "get_pokemon_damage",
    "extract_move_name",
    "normalize_action_name",
    "canonical_move_signature",
    "classify_turn_phase",
    "classify_energy_band",
    "classify_damage_band",
]


section3ba_runtime_helper_rows = []


for helper_name in SECTION3BA_RUNTIME_HELPERS:

    helper_value = globals().get(
        helper_name
    )

    section3ba_runtime_helper_rows.append(
        {
            "helper_name":
                helper_name,

            "exists":
                helper_name in globals(),

            "callable":
                callable(
                    helper_value
                ),

            "object_type":
                (
                    type(
                        helper_value
                    ).__name__
                    if helper_name in globals()
                    else "MISSING"
                ),
        }
    )


section3ba_runtime_helper_profile_df = pd.DataFrame(
    section3ba_runtime_helper_rows
)


print()
print("BALANCED-STATE RECONSTRUCTION HELPER PROFILE")
print("-" * 100)

display(
    section3ba_runtime_helper_profile_df
)


# --------------------------------------------------------------------------------------
# 7. Determine available evaluation route
# --------------------------------------------------------------------------------------

section3ba_build_state_ready = bool(
    globals().get(
        "build_section4wf_evaluation_state"
    )
    is not None
    and
    callable(
        globals().get(
            "build_section4wf_evaluation_state"
        )
    )
)


section3ba_feature_builder_ready = bool(
    globals().get(
        "build_legality_aware_feature_row"
    )
    is not None
    and
    callable(
        globals().get(
            "build_legality_aware_feature_row"
        )
    )
)


section3ba_score_case_count = int(
    len(
        section3ba_score_cases_df
    )
)


section3ba_unique_case_count = (
    int(
        section3ba_score_cases_df[
            "comparison_case_id"
        ].nunique()
    )
    if "comparison_case_id"
    in section3ba_score_cases_df.columns
    else 0
)


section3ba_unique_pair_count = (
    int(
        section3ba_score_cases_df[
            "expanded_pair_id"
        ].nunique()
    )
    if "expanded_pair_id"
    in section3ba_score_cases_df.columns
    else 0
)


if (
    section3ba_build_state_ready
    and
    section3ba_feature_builder_ready
):

    section3ba_evaluation_route = (
        "DIRECT_RUNTIME_STATE_RECONSTRUCTION"
    )

    section3ba_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )

else:

    section3ba_evaluation_route = (
        "RESTORE_NOTEBOOK54_EVALUATION_BUILDERS"
    )

    section3ba_next_stage = (
        "BOOTSTRAP_REQUIRED_NOTEBOOK54_EVALUATION_HELPERS"
    )


# --------------------------------------------------------------------------------------
# 8. Summary
# --------------------------------------------------------------------------------------

section3ba_summary = {
    "status":
        "BALANCED_MULTI_ACTION_EVALUATION_PREFLIGHT_COMPLETE",

    "score_case_rows":
        section3ba_score_case_count,

    "unique_comparison_cases":
        section3ba_unique_case_count,

    "unique_mirror_pairs":
        section3ba_unique_pair_count,

    "state_builder_ready":
        section3ba_build_state_ready,

    "feature_builder_ready":
        section3ba_feature_builder_ready,

    "evaluation_route":
        section3ba_evaluation_route,

    "next_stage":
        section3ba_next_stage,
}


print()
print("SECTION 3B-A EVALUATION PREFLIGHT SUMMARY")
print("-" * 100)

for key, value in section3ba_summary.items():
    print(
        f"{key:56}: {value}"
    )


# --------------------------------------------------------------------------------------
# 9. Validation checks
# --------------------------------------------------------------------------------------

section3ba_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_required_reports_loaded",

            "passed":
                section3ba_report_inventory_df[
                    "exists"
                ].astype(bool).all(),

            "value":
                int(
                    section3ba_report_inventory_df[
                        "exists"
                    ].astype(bool).sum()
                ),

            "expected":
                len(
                    SECTION3BA_CANDIDATE_REPORTS
                ),
        },
        {
            "check":
                "score_cases_available",

            "passed":
                section3ba_score_case_count > 0,

            "value":
                section3ba_score_case_count,

            "expected":
                "> 0",
        },
        {
            "check":
                "comparison_case_id_available",

            "passed":
                "comparison_case_id"
                in section3ba_score_cases_df.columns,

            "value":
                (
                    "comparison_case_id"
                    in section3ba_score_cases_df.columns
                ),

            "expected":
                True,
        },
        {
            "check":
                "expanded_pair_id_available",

            "passed":
                "expanded_pair_id"
                in section3ba_score_cases_df.columns,

            "value":
                (
                    "expanded_pair_id"
                    in section3ba_score_cases_df.columns
                ),

            "expected":
                True,
        },
        {
            "check":
                "evaluation_route_resolved",

            "passed":
                section3ba_evaluation_route
                in {
                    "DIRECT_RUNTIME_STATE_RECONSTRUCTION",
                    "RESTORE_NOTEBOOK54_EVALUATION_BUILDERS",
                },

            "value":
                section3ba_evaluation_route,

            "expected":
                "Recognized route",
        },
    ]
)


print()
print("SECTION 3B-A VALIDATION CHECKS")
print("-" * 100)

display(
    section3ba_validation_checks_df
)


assert section3ba_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 10. Save reports
# --------------------------------------------------------------------------------------

SECTION3BA_REPORT_INVENTORY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_report_inventory.csv"
)

SECTION3BA_DATAFRAME_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_dataframe_profile.csv"
)

SECTION3BA_HELPER_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_runtime_helper_profile.csv"
)

SECTION3BA_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_validation_checks.csv"
)

SECTION3BA_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_evaluation_preflight_summary.json"
)


section3ba_report_inventory_df.to_csv(
    SECTION3BA_REPORT_INVENTORY_FILE,
    index=False,
)

section3ba_dataframe_profile_df.to_csv(
    SECTION3BA_DATAFRAME_PROFILE_FILE,
    index=False,
)

section3ba_runtime_helper_profile_df.to_csv(
    SECTION3BA_HELPER_PROFILE_FILE,
    index=False,
)

section3ba_validation_checks_df.to_csv(
    SECTION3BA_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BA_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        section3ba_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3ba_saved_files = [
    SECTION3BA_REPORT_INVENTORY_FILE,
    SECTION3BA_DATAFRAME_PROFILE_FILE,
    SECTION3BA_HELPER_PROFILE_FILE,
    SECTION3BA_VALIDATION_FILE,
    SECTION3BA_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3ba_saved_files
)


print()
print("SAVED SECTION 3B-A REPORTS")
print("-" * 100)

for file_path in section3ba_saved_files:
    print(file_path)


print()
print(
    "✅ SECTION 3B-A BALANCED MULTI-ACTION "
    "POLICY EVALUATION PREFLIGHT PASSED"
)


# In[18]:


# ======================================================================================
# SECTION 3B-A REPAIR — RESTORE BALANCED EVALUATION STATE BUILDER
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR — RESTORE BALANCED EVALUATION STATE BUILDER")
print("=" * 100)

from pathlib import Path
from types import SimpleNamespace
import ast
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate required objects
# --------------------------------------------------------------------------------------

SECTION3BAR_REQUIRED_OBJECTS = [
    "section3ba_expanded_case_results_df",
    "section3ba_score_cases_df",
    "build_legality_aware_feature_row",
    "baseline_policy_model",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "REPORTS_DIRECTORY",
]


section3bar_missing_objects = [
    object_name
    for object_name in SECTION3BAR_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3B-A REPAIR OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3BAR_REQUIRED_OBJECTS:

    print(
        f"{object_name:50}: "
        f"{object_name in globals()}"
    )


assert not section3bar_missing_objects, (
    "Required Section 3B-A repair objects are missing: "
    f"{section3bar_missing_objects}"
)


assert callable(
    build_legality_aware_feature_row
)


assert len(
    section3ba_expanded_case_results_df
) == 184


assert len(
    section3ba_score_cases_df
) == 184


# --------------------------------------------------------------------------------------
# 2. Normalize legal-move collections
# --------------------------------------------------------------------------------------

def normalize_section3bar_legal_moves(
    legal_moves_value,
):
    """
    Convert stored CSV legal-move values into a clean Python list.
    """

    if legal_moves_value is None:

        return []


    if isinstance(
        legal_moves_value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            str(
                move_name
            ).strip()
            for move_name in legal_moves_value
            if str(
                move_name
            ).strip()
        ]


    if isinstance(
        legal_moves_value,
        str,
    ):

        stripped_value = legal_moves_value.strip()


        if not stripped_value:

            return []


        try:

            parsed_value = ast.literal_eval(
                stripped_value
            )

            if isinstance(
                parsed_value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):

                return [
                    str(
                        move_name
                    ).strip()
                    for move_name in parsed_value
                    if str(
                        move_name
                    ).strip()
                ]

        except Exception:

            pass


        return [
            move_name.strip()
            for move_name in stripped_value.split(",")
            if move_name.strip()
        ]


    return [
        str(
            legal_moves_value
        ).strip()
    ]


# --------------------------------------------------------------------------------------
# 3. Resolve the legal-move column
# --------------------------------------------------------------------------------------

SECTION3BAR_LEGAL_MOVE_COLUMN_CANDIDATES = [
    "runtime_legal_moves",
    "expected_legal_moves",
    "legal_moves",
]


SECTION3BAR_LEGAL_MOVE_COLUMN = next(
    (
        column_name
        for column_name
        in SECTION3BAR_LEGAL_MOVE_COLUMN_CANDIDATES
        if column_name
        in section3ba_expanded_case_results_df.columns
    ),
    None,
)


assert SECTION3BAR_LEGAL_MOVE_COLUMN is not None, (
    "No legal-move column was found in the expanded case results. "
    f"Checked: {SECTION3BAR_LEGAL_MOVE_COLUMN_CANDIDATES}"
)


print()
print("RESOLVED EXPANDED-CASE COLUMNS")
print("-" * 100)

print(
    "Legal-move column:",
    SECTION3BAR_LEGAL_MOVE_COLUMN,
)


# --------------------------------------------------------------------------------------
# 4. Build protected lookup table
# --------------------------------------------------------------------------------------

section3bar_case_lookup_df = (
    section3ba_expanded_case_results_df
    .copy()
    .drop_duplicates(
        subset=[
            "comparison_case_id",
        ]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


assert len(
    section3bar_case_lookup_df
) == 184


assert section3bar_case_lookup_df.index.is_unique


# --------------------------------------------------------------------------------------
# 5. Safe numeric reader
# --------------------------------------------------------------------------------------

def section3bar_safe_float(
    value,
    default=0.0,
):
    """
    Convert a stored value to float while handling missing CSV values.
    """

    try:

        numeric_value = float(
            value
        )

        if np.isnan(
            numeric_value
        ):

            return float(
                default
            )

        return numeric_value

    except (
        TypeError,
        ValueError,
    ):

        return float(
            default
        )


# --------------------------------------------------------------------------------------
# 6. Minimal simulator-compatible objects
# --------------------------------------------------------------------------------------

def create_section3bar_active_pokemon(
    card_name,
    attached_energy,
    current_hp,
):
    """
    Create the minimum active-Pokémon interface required by the feature builder.
    """

    resolved_hp = section3bar_safe_float(
        current_hp,
        default=0.0,
    )


    return SimpleNamespace(
        name=str(
            card_name
        ),

        card={
            "name":
                str(
                    card_name
                ),
        },

        attached_energy=int(
            round(
                section3bar_safe_float(
                    attached_energy,
                    default=0.0,
                )
            )
        ),

        current_hp=resolved_hp,

        max_hp=resolved_hp,

        maximum_hp=resolved_hp,

        damage=0.0,

        damage_taken=0.0,
    )


def create_section3bar_side_state(
    active_pokemon,
):
    """
    Create the minimum player/opponent state required by the feature builder.
    """

    return SimpleNamespace(
        active=active_pokemon,

        prize_cards_remaining=6,

        hand_size=7,

        bench=[],
    )


# --------------------------------------------------------------------------------------
# 7. Restore build_section4wf_evaluation_state
# --------------------------------------------------------------------------------------

def build_section4wf_evaluation_state(
    design_row,
):
    """
    Reconstruct one balanced Notebook 54 evaluation state.

    The function preserves:
    - comparison-case identity,
    - acting side,
    - acting and opposing cards,
    - acting and opposing energy,
    - HP values when available,
    - source turn number.

    Only the minimum simulator interface needed by
    build_legality_aware_feature_row is created.
    """

    comparison_case_id = str(
        design_row[
            "comparison_case_id"
        ]
    )


    if comparison_case_id not in (
        section3bar_case_lookup_df.index
    ):

        raise KeyError(
            "Balanced evaluation case was not found: "
            f"{comparison_case_id}"
        )


    source_row = section3bar_case_lookup_df.loc[
        comparison_case_id
    ]


    evaluation_side = str(
        source_row.get(
            "evaluation_side",
            design_row.get(
                "evaluation_side",
                "Player",
            ),
        )
    ).strip().title()


    if evaluation_side not in {
        "Player",
        "Opponent",
    }:

        raise ValueError(
            "Unsupported evaluation side: "
            f"{evaluation_side}"
        )


    acting_card = str(
        source_row.get(
            "acting_card",
            design_row.get(
                "acting_card",
                "UNKNOWN",
            ),
        )
    )


    opposing_card = str(
        source_row.get(
            "opposing_card",
            design_row.get(
                "opposing_card",
                "UNKNOWN",
            ),
        )
    )


    acting_energy = section3bar_safe_float(
        source_row.get(
            "acting_energy",
            design_row.get(
                "acting_energy",
                0.0,
            ),
        ),
        default=0.0,
    )


    opposing_energy = section3bar_safe_float(
        source_row.get(
            "opposing_energy",
            design_row.get(
                "opposing_energy",
                0.0,
            ),
        ),
        default=0.0,
    )


    acting_hp = section3bar_safe_float(
        source_row.get(
            "acting_hp",
            design_row.get(
                "acting_hp",
                0.0,
            ),
        ),
        default=0.0,
    )


    opposing_hp = section3bar_safe_float(
        source_row.get(
            "opposing_hp",
            design_row.get(
                "opposing_hp",
                0.0,
            ),
        ),
        default=0.0,
    )


    source_turn_number = int(
        round(
            section3bar_safe_float(
                source_row.get(
                    "source_turn_number",
                    design_row.get(
                        "source_turn_number",
                        1,
                    ),
                ),
                default=1.0,
            )
        )
    )


    acting_active = create_section3bar_active_pokemon(
        card_name=acting_card,
        attached_energy=acting_energy,
        current_hp=acting_hp,
    )


    opposing_active = create_section3bar_active_pokemon(
        card_name=opposing_card,
        attached_energy=opposing_energy,
        current_hp=opposing_hp,
    )


    if evaluation_side == "Player":

        player_state = create_section3bar_side_state(
            acting_active
        )

        opponent_state = create_section3bar_side_state(
            opposing_active
        )

    else:

        player_state = create_section3bar_side_state(
            opposing_active
        )

        opponent_state = create_section3bar_side_state(
            acting_active
        )


    evaluation_state = SimpleNamespace(
        player=player_state,

        opponent=opponent_state,

        current_player=evaluation_side,

        turn_number=source_turn_number,

        comparison_case_id=comparison_case_id,

        source_scenario_id=str(
            source_row.get(
                "source_scenario_id",
                "",
            )
        ),

        source_condition_id=str(
            source_row.get(
                "source_condition_id",
                "",
            )
        ),

        expansion_variant_id=str(
            source_row.get(
                "expansion_variant_id",
                "",
            )
        ),
    )


    return evaluation_state


assert callable(
    build_section4wf_evaluation_state
)


# --------------------------------------------------------------------------------------
# 8. Build one test state
# --------------------------------------------------------------------------------------

section3bar_test_row = (
    section3ba_score_cases_df.iloc[0]
)


section3bar_test_case_id = str(
    section3bar_test_row[
        "comparison_case_id"
    ]
)


section3bar_test_source_row = (
    section3bar_case_lookup_df.loc[
        section3bar_test_case_id
    ]
)


section3bar_test_state = (
    build_section4wf_evaluation_state(
        section3bar_test_row
    )
)


section3bar_test_legal_moves = (
    normalize_section3bar_legal_moves(
        section3bar_test_source_row[
            SECTION3BAR_LEGAL_MOVE_COLUMN
        ]
    )
)


section3bar_test_feature_df = (
    build_legality_aware_feature_row(
        battle_state=
            section3bar_test_state,

        legal_moves=
            section3bar_test_legal_moves,

        side_mode=
            "PRESERVE",
    )
)


print()
print("SINGLE-STATE RECONSTRUCTION TEST")
print("-" * 100)

print(
    "Comparison case:",
    section3bar_test_case_id,
)

print(
    "Current player:",
    section3bar_test_state.current_player,
)

print(
    "Turn number:",
    section3bar_test_state.turn_number,
)

print(
    "Legal moves:",
    section3bar_test_legal_moves,
)

print(
    "Raw feature shape:",
    section3bar_test_feature_df.shape,
)


assert section3bar_test_feature_df.shape == (
    1,
    41,
)


# --------------------------------------------------------------------------------------
# 9. Validate all 184 states and baseline probabilities
# --------------------------------------------------------------------------------------

section3bar_validation_rows = []
section3bar_error_rows = []


section3bar_model_classes = [
    str(
        class_name
    ).strip()
    for class_name in baseline_policy_model.classes_
]


section3bar_normalized_classes = [
    class_name.lower()
    for class_name in section3bar_model_classes
]


assert "quick attack" in section3bar_normalized_classes
assert "ascension" in section3bar_normalized_classes


SECTION3BAR_QUICK_ATTACK_INDEX = (
    section3bar_normalized_classes.index(
        "quick attack"
    )
)


SECTION3BAR_ASCENSION_INDEX = (
    section3bar_normalized_classes.index(
        "ascension"
    )
)


for _, score_row in (
    section3ba_score_cases_df.iterrows()
):

    comparison_case_id = str(
        score_row[
            "comparison_case_id"
        ]
    )


    try:

        source_row = section3bar_case_lookup_df.loc[
            comparison_case_id
        ]


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = (
            normalize_section3bar_legal_moves(
                source_row[
                    SECTION3BAR_LEGAL_MOVE_COLUMN
                ]
            )
        )


        raw_feature_df = (
            build_legality_aware_feature_row(
                battle_state=
                    evaluation_state,

                legal_moves=
                    legal_moves,

                side_mode=
                    "PRESERVE",
            )
        )


        transformed_features = (
            baseline_preprocessor.transform(
                raw_feature_df
            )
        )


        probability_vector = np.asarray(
            baseline_policy_model.predict_proba(
                transformed_features
            )
        )[0]


        reconstructed_quick_attack_probability = float(
            probability_vector[
                SECTION3BAR_QUICK_ATTACK_INDEX
            ]
        )


        reconstructed_ascension_probability = float(
            probability_vector[
                SECTION3BAR_ASCENSION_INDEX
            ]
        )


        saved_quick_attack_probability = float(
            score_row[
                "quick_attack_probability"
            ]
        )


        saved_ascension_probability = float(
            score_row[
                "ascension_probability"
            ]
        )


        section3bar_validation_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "evaluation_side":
                    evaluation_state.current_player,

                "legal_moves":
                    legal_moves,

                "raw_feature_columns":
                    int(
                        raw_feature_df.shape[
                            1
                        ]
                    ),

                "reconstructed_quick_attack_probability":
                    reconstructed_quick_attack_probability,

                "saved_quick_attack_probability":
                    saved_quick_attack_probability,

                "quick_attack_absolute_difference":
                    abs(
                        reconstructed_quick_attack_probability
                        -
                        saved_quick_attack_probability
                    ),

                "reconstructed_ascension_probability":
                    reconstructed_ascension_probability,

                "saved_ascension_probability":
                    saved_ascension_probability,

                "ascension_absolute_difference":
                    abs(
                        reconstructed_ascension_probability
                        -
                        saved_ascension_probability
                    ),

                "state_reconstruction_success":
                    True,
            }
        )


    except Exception as error:

        section3bar_error_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

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


section3bar_probability_validation_df = pd.DataFrame(
    section3bar_validation_rows
)


section3bar_state_errors_df = pd.DataFrame(
    section3bar_error_rows,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


print()
print("STATE-RECONSTRUCTION ERRORS")
print("-" * 100)

if section3bar_state_errors_df.empty:

    print(
        "No balanced-state reconstruction errors were recorded."
    )

else:

    display(
        section3bar_state_errors_df
    )


assert section3bar_state_errors_df.empty, (
    "One or more balanced evaluation states could not be reconstructed."
)


assert len(
    section3bar_probability_validation_df
) == 184


# --------------------------------------------------------------------------------------
# 10. Calculate reconstruction agreement
# --------------------------------------------------------------------------------------

section3bar_mean_quick_attack_difference = float(
    section3bar_probability_validation_df[
        "quick_attack_absolute_difference"
    ].mean()
)


section3bar_max_quick_attack_difference = float(
    section3bar_probability_validation_df[
        "quick_attack_absolute_difference"
    ].max()
)


section3bar_mean_ascension_difference = float(
    section3bar_probability_validation_df[
        "ascension_absolute_difference"
    ].mean()
)


section3bar_max_ascension_difference = float(
    section3bar_probability_validation_df[
        "ascension_absolute_difference"
    ].max()
)


SECTION3BAR_PROBABILITY_TOLERANCE = 1e-8


section3bar_exact_probability_agreement = bool(
    section3bar_max_quick_attack_difference
    <=
    SECTION3BAR_PROBABILITY_TOLERANCE
    and
    section3bar_max_ascension_difference
    <=
    SECTION3BAR_PROBABILITY_TOLERANCE
)


print()
print("BASELINE PROBABILITY RECONSTRUCTION AGREEMENT")
print("-" * 100)

print(
    "Mean Quick Attack difference:",
    section3bar_mean_quick_attack_difference,
)

print(
    "Maximum Quick Attack difference:",
    section3bar_max_quick_attack_difference,
)

print(
    "Mean Ascension difference:",
    section3bar_mean_ascension_difference,
)

print(
    "Maximum Ascension difference:",
    section3bar_max_ascension_difference,
)

print(
    "Exact probability agreement:",
    section3bar_exact_probability_agreement,
)


# --------------------------------------------------------------------------------------
# 11. Determine evaluation readiness
# --------------------------------------------------------------------------------------

if section3bar_exact_probability_agreement:

    section3bar_repair_status = (
        "BALANCED_STATE_BUILDER_RESTORED_EXACTLY"
    )

    section3bar_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )

else:

    section3bar_repair_status = (
        "BALANCED_STATE_BUILDER_RESTORED_WITH_PROBABILITY_DRIFT"
    )

    section3bar_next_stage = (
        "RECONSTRUCTION_DRIFT_DIAGNOSTIC"
    )


section3bar_summary = {
    "status":
        section3bar_repair_status,

    "states_reconstructed":
        int(
            len(
                section3bar_probability_validation_df
            )
        ),

    "state_errors":
        int(
            len(
                section3bar_state_errors_df
            )
        ),

    "raw_feature_count":
        int(
            section3bar_test_feature_df.shape[
                1
            ]
        ),

    "encoded_feature_count":
        int(
            len(
                baseline_encoded_feature_names
            )
        ),

    "mean_quick_attack_probability_difference":
        section3bar_mean_quick_attack_difference,

    "maximum_quick_attack_probability_difference":
        section3bar_max_quick_attack_difference,

    "mean_ascension_probability_difference":
        section3bar_mean_ascension_difference,

    "maximum_ascension_probability_difference":
        section3bar_max_ascension_difference,

    "exact_probability_agreement":
        section3bar_exact_probability_agreement,

    "next_stage":
        section3bar_next_stage,
}


print()
print("SECTION 3B-A REPAIR SUMMARY")
print("-" * 100)

for key, value in section3bar_summary.items():

    print(
        f"{key:62}: {value}"
    )


# --------------------------------------------------------------------------------------
# 12. Validation checks
# --------------------------------------------------------------------------------------

section3bar_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "state_builder_restored",

            "passed":
                callable(
                    build_section4wf_evaluation_state
                ),

            "value":
                True,

            "expected":
                True,
        },
        {
            "check":
                "all_184_states_reconstructed",

            "passed":
                len(
                    section3bar_probability_validation_df
                ) == 184,

            "value":
                len(
                    section3bar_probability_validation_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "no_state_reconstruction_errors",

            "passed":
                section3bar_state_errors_df.empty,

            "value":
                len(
                    section3bar_state_errors_df
                ),

            "expected":
                0,
        },
        {
            "check":
                "raw_feature_schema_restored",

            "passed":
                section3bar_test_feature_df.shape
                ==
                (
                    1,
                    41,
                ),

            "value":
                section3bar_test_feature_df.shape,

            "expected":
                (
                    1,
                    41,
                ),
        },
        {
            "check":
                "baseline_probability_agreement",

            "passed":
                section3bar_exact_probability_agreement,

            "value":
                {
                    "quick_attack_max_difference":
                        section3bar_max_quick_attack_difference,

                    "ascension_max_difference":
                        section3bar_max_ascension_difference,
                },

            "expected":
                f"Both at most {SECTION3BAR_PROBABILITY_TOLERANCE}",
        },
        {
            "check":
                "next_stage_ready",

            "passed":
                section3bar_next_stage
                ==
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",

            "value":
                section3bar_next_stage,

            "expected":
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR VALIDATION CHECKS")
print("-" * 100)

display(
    section3bar_validation_checks_df
)


section3bar_failed_checks = int(
    (
        ~section3bar_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section3bar_failed_checks == 0, (
    "One or more Section 3B-A repair checks failed. "
    "Review the probability reconstruction differences above."
)


# --------------------------------------------------------------------------------------
# 13. Save repair reports
# --------------------------------------------------------------------------------------

SECTION3BAR_PROBABILITY_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bar_probability_reconstruction_validation.csv"
)

SECTION3BAR_ERRORS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bar_state_reconstruction_errors.csv"
)

SECTION3BAR_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bar_validation_checks.csv"
)

SECTION3BAR_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bar_state_builder_repair_summary.json"
)


section3bar_probability_validation_df.to_csv(
    SECTION3BAR_PROBABILITY_VALIDATION_FILE,
    index=False,
)

section3bar_state_errors_df.to_csv(
    SECTION3BAR_ERRORS_FILE,
    index=False,
)

section3bar_validation_checks_df.to_csv(
    SECTION3BAR_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BAR_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3bar_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3bar_saved_files = [
    SECTION3BAR_PROBABILITY_VALIDATION_FILE,
    SECTION3BAR_ERRORS_FILE,
    SECTION3BAR_VALIDATION_FILE,
    SECTION3BAR_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3bar_saved_files
)


print()
print("SAVED SECTION 3B-A REPAIR REPORTS")
print("-" * 100)

for file_path in section3bar_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A BALANCED EVALUATION "
    "STATE BUILDER RESTORED"
)


# In[20]:


# ======================================================================================
# SECTION 3B-A REPAIR B — RESTORE EXACT NOTEBOOK 54 EVALUATION STATE BUILDER
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR B — RESTORE EXACT NOTEBOOK 54 EVALUATION STATE BUILDER")
print("=" * 100)

from pathlib import Path
import ast
import inspect
import nbformat
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Locate the completed Notebook 54
# --------------------------------------------------------------------------------------

NOTEBOOK54_EXACT_SOURCE_PATH = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents"
    r"\PTCG_AI_Battle_Challenge"
    r"\notebooks"
    r"\54_tournament_strength_optimization_clean.ipynb"
)

assert NOTEBOOK54_EXACT_SOURCE_PATH.exists(), (
    f"Completed Notebook 54 was not found: {NOTEBOOK54_EXACT_SOURCE_PATH}"
)

print()
print("Notebook 54 source:")
print(NOTEBOOK54_EXACT_SOURCE_PATH)


# --------------------------------------------------------------------------------------
# 2. Read Notebook 54 without executing the notebook
# --------------------------------------------------------------------------------------

with open(
    NOTEBOOK54_EXACT_SOURCE_PATH,
    "r",
    encoding="utf-8",
) as file:

    notebook54_source_document = nbformat.read(
        file,
        as_version=4,
    )


notebook54_code_cells = [
    cell
    for cell in notebook54_source_document.cells
    if cell.cell_type == "code"
]


# --------------------------------------------------------------------------------------
# 3. Extract the exact function definition
# --------------------------------------------------------------------------------------

TARGET_FUNCTION_NAME_3BARB = (
    "build_section4wf_evaluation_state"
)

section3barb_function_source = None
section3barb_source_cell_index = None


for code_cell_index, code_cell in enumerate(
    notebook54_code_cells
):

    source_text = str(
        code_cell.source
    )

    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in syntax_tree.body:

        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and
            node.name == TARGET_FUNCTION_NAME_3BARB
        ):

            section3barb_function_source = (
                ast.get_source_segment(
                    source_text,
                    node,
                )
            )

            section3barb_source_cell_index = int(
                code_cell_index
            )

            break


    if section3barb_function_source is not None:

        break


assert section3barb_function_source is not None, (
    "The exact build_section4wf_evaluation_state definition "
    "was not found in the completed Notebook 54."
)


print()
print("EXACT FUNCTION SOURCE FOUND")
print("-" * 100)

print(
    "Notebook 54 code-cell index:",
    section3barb_source_cell_index,
)

print(
    "Source characters:",
    len(
        section3barb_function_source
    ),
)


# --------------------------------------------------------------------------------------
# 4. Remove the temporary reconstructed function
# --------------------------------------------------------------------------------------

if "build_section4wf_evaluation_state" in globals():

    del globals()[
        "build_section4wf_evaluation_state"
    ]


# --------------------------------------------------------------------------------------
# 5. Install the exact Notebook 54 function
# --------------------------------------------------------------------------------------

exec(
    section3barb_function_source,
    globals(),
)


assert callable(
    build_section4wf_evaluation_state
)


print()
print("Exact Notebook 54 function installed:")
print(build_section4wf_evaluation_state)


# --------------------------------------------------------------------------------------
# 6. Determine referenced runtime dependencies
# --------------------------------------------------------------------------------------

section3barb_function_globals = {
    name
    for name in (
        build_section4wf_evaluation_state
        .__code__
        .co_names
    )
    if name not in {
        "str",
        "int",
        "float",
        "bool",
        "len",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "getattr",
        "setattr",
        "hasattr",
        "Exception",
        "KeyError",
        "ValueError",
        "TypeError",
        "deepcopy",
    }
}


section3barb_dependency_rows = []


for dependency_name in sorted(
    section3barb_function_globals
):

    section3barb_dependency_rows.append(
        {
            "dependency_name":
                dependency_name,

            "available":
                dependency_name
                in globals(),

            "object_type":
                (
                    type(
                        globals()[
                            dependency_name
                        ]
                    ).__name__
                    if dependency_name
                    in globals()
                    else "MISSING"
                ),
        }
    )


section3barb_dependency_df = pd.DataFrame(
    section3barb_dependency_rows
)


print()
print("EXACT BUILDER DEPENDENCY PROFILE")
print("-" * 100)

display(
    section3barb_dependency_df
)


section3barb_missing_dependencies = (
    section3barb_dependency_df.loc[
        ~section3barb_dependency_df[
            "available"
        ].astype(bool),
        "dependency_name",
    ]
    .astype(str)
    .tolist()
)


print()
print(
    "Missing exact-builder dependencies:",
    section3barb_missing_dependencies,
)


print()
print(
    "✅ EXACT NOTEBOOK 54 STATE-BUILDER DEFINITION RESTORED"
)


# In[19]:


# ======================================================================================
# SECTION 3B-A REPAIR DIAGNOSTIC — IDENTIFY FAILED CHECKS
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR DIAGNOSTIC — IDENTIFY FAILED CHECKS")
print("=" * 100)

failed_checks_df = (
    section3bar_validation_checks_df.loc[
        ~section3bar_validation_checks_df[
            "passed"
        ].astype(bool)
    ]
    .copy()
    .reset_index(drop=True)
)

print()
print("FAILED VALIDATION CHECKS")
print("-" * 100)

if failed_checks_df.empty:
    print("No failed checks were found.")
else:
    display(failed_checks_df)


print()
print("PROBABILITY DIFFERENCE SUMMARY")
print("-" * 100)

print(
    "Mean Quick Attack difference:",
    section3bar_mean_quick_attack_difference,
)

print(
    "Maximum Quick Attack difference:",
    section3bar_max_quick_attack_difference,
)

print(
    "Mean Ascension difference:",
    section3bar_mean_ascension_difference,
)

print(
    "Maximum Ascension difference:",
    section3bar_max_ascension_difference,
)

print(
    "Exact probability agreement:",
    section3bar_exact_probability_agreement,
)


print()
print("LARGEST QUICK ATTACK DIFFERENCES")
print("-" * 100)

display(
    section3bar_probability_validation_df
    .sort_values(
        "quick_attack_absolute_difference",
        ascending=False,
    )
    .head(20)
)


print()
print("LARGEST ASCENSION DIFFERENCES")
print("-" * 100)

display(
    section3bar_probability_validation_df
    .sort_values(
        "ascension_absolute_difference",
        ascending=False,
    )
    .head(20)
)


# In[21]:


# ======================================================================================
# SECTION 3B-A REPAIR C — RECOVER EXACT SCENARIO SOURCE BINDING
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR C — RECOVER EXACT SCENARIO SOURCE BINDING")
print("=" * 100)

from pathlib import Path
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate the exact builder and battle-state constructor
# --------------------------------------------------------------------------------------

assert callable(
    build_section4wf_evaluation_state
), (
    "The exact Notebook 54 evaluation-state builder is unavailable."
)


assert callable(
    build_replay_battle_state
), (
    "build_replay_battle_state is unavailable."
)


# --------------------------------------------------------------------------------------
# 2. Search live runtime dataframes for an authoritative scenario source
# --------------------------------------------------------------------------------------

SECTION3BARC_SCENARIO_ID_CANDIDATES = [
    "scenario_id",
    "source_scenario_id",
    "expanded_scenario_id",
]


section3barc_runtime_candidate_rows = []
section3barc_runtime_candidates = {}


for object_name, object_value in list(
    globals().items()
):

    if not isinstance(
        object_value,
        pd.DataFrame,
    ):
        continue

    scenario_columns = [
        column_name
        for column_name in SECTION3BARC_SCENARIO_ID_CANDIDATES
        if column_name in object_value.columns
    ]

    if not scenario_columns:
        continue

    build_test_passed = False
    build_test_error = ""

    try:

        if not object_value.empty:

            test_state = build_replay_battle_state(
                object_value.iloc[0]
            )

            build_test_passed = (
                test_state is not None
            )

    except Exception as error:

        build_test_error = (
            f"{type(error).__name__}: {error}"
        )

    section3barc_runtime_candidate_rows.append(
        {
            "source_type":
                "RUNTIME",

            "candidate_name":
                object_name,

            "rows":
                int(
                    len(
                        object_value
                    )
                ),

            "columns":
                int(
                    len(
                        object_value.columns
                    )
                ),

            "scenario_id_column":
                scenario_columns[0],

            "unique_scenarios":
                int(
                    object_value[
                        scenario_columns[0]
                    ]
                    .astype(str)
                    .nunique()
                ),

            "build_test_passed":
                bool(
                    build_test_passed
                ),

            "build_test_error":
                build_test_error,
        }
    )

    section3barc_runtime_candidates[
        object_name
    ] = object_value


section3barc_runtime_candidate_df = pd.DataFrame(
    section3barc_runtime_candidate_rows
)


print()
print("LIVE SCENARIO-SOURCE CANDIDATES")
print("-" * 100)

if section3barc_runtime_candidate_df.empty:

    print(
        "No live dataframe with a recognized scenario identifier was found."
    )

else:

    display(
        section3barc_runtime_candidate_df
        .sort_values(
            [
                "build_test_passed",
                "unique_scenarios",
                "rows",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# 3. Search Notebook 54 report CSVs if no valid live source exists
# --------------------------------------------------------------------------------------

section3barc_report_candidate_rows = []
section3barc_report_candidates = {}


for report_path in sorted(
    NOTEBOOK54_SECTION4_REPORT_DIRECTORY.glob(
        "*.csv"
    )
):

    try:

        candidate_df = pd.read_csv(
            report_path
        )

    except Exception:

        continue

    scenario_columns = [
        column_name
        for column_name in SECTION3BARC_SCENARIO_ID_CANDIDATES
        if column_name in candidate_df.columns
    ]

    if not scenario_columns:
        continue

    build_test_passed = False
    build_test_error = ""

    try:

        if not candidate_df.empty:

            test_state = build_replay_battle_state(
                candidate_df.iloc[0]
            )

            build_test_passed = (
                test_state is not None
            )

    except Exception as error:

        build_test_error = (
            f"{type(error).__name__}: {error}"
        )

    section3barc_report_candidate_rows.append(
        {
            "source_type":
                "REPORT",

            "candidate_name":
                report_path.name,

            "candidate_path":
                str(
                    report_path
                ),

            "rows":
                int(
                    len(
                        candidate_df
                    )
                ),

            "columns":
                int(
                    len(
                        candidate_df.columns
                    )
                ),

            "scenario_id_column":
                scenario_columns[0],

            "unique_scenarios":
                int(
                    candidate_df[
                        scenario_columns[0]
                    ]
                    .astype(str)
                    .nunique()
                ),

            "build_test_passed":
                bool(
                    build_test_passed
                ),

            "build_test_error":
                build_test_error,
        }
    )

    section3barc_report_candidates[
        report_path.name
    ] = candidate_df


section3barc_report_candidate_df = pd.DataFrame(
    section3barc_report_candidate_rows
)


print()
print("REPORT-BASED SCENARIO-SOURCE CANDIDATES")
print("-" * 100)

if section3barc_report_candidate_df.empty:

    print(
        "No Notebook 54 report dataframe with a recognized scenario identifier was found."
    )

else:

    display(
        section3barc_report_candidate_df
        .sort_values(
            [
                "build_test_passed",
                "unique_scenarios",
                "rows",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# 4. Select the strongest valid source
# --------------------------------------------------------------------------------------

section3barc_selected_source_type = None
section3barc_selected_source_name = None
section3barc_selected_source_df = None
section3barc_selected_scenario_column = None


if (
    not section3barc_runtime_candidate_df.empty
    and
    section3barc_runtime_candidate_df[
        "build_test_passed"
    ].astype(bool).any()
):

    selected_row = (
        section3barc_runtime_candidate_df.loc[
            section3barc_runtime_candidate_df[
                "build_test_passed"
            ].astype(bool)
        ]
        .sort_values(
            [
                "unique_scenarios",
                "rows",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    section3barc_selected_source_type = (
        "RUNTIME"
    )

    section3barc_selected_source_name = str(
        selected_row[
            "candidate_name"
        ]
    )

    section3barc_selected_source_df = (
        section3barc_runtime_candidates[
            section3barc_selected_source_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    section3barc_selected_scenario_column = str(
        selected_row[
            "scenario_id_column"
        ]
    )


elif (
    not section3barc_report_candidate_df.empty
    and
    section3barc_report_candidate_df[
        "build_test_passed"
    ].astype(bool).any()
):

    selected_row = (
        section3barc_report_candidate_df.loc[
            section3barc_report_candidate_df[
                "build_test_passed"
            ].astype(bool)
        ]
        .sort_values(
            [
                "unique_scenarios",
                "rows",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    section3barc_selected_source_type = (
        "REPORT"
    )

    section3barc_selected_source_name = str(
        selected_row[
            "candidate_name"
        ]
    )

    section3barc_selected_source_df = (
        section3barc_report_candidates[
            section3barc_selected_source_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    section3barc_selected_scenario_column = str(
        selected_row[
            "scenario_id_column"
        ]
    )


assert section3barc_selected_source_df is not None, (
    "No authoritative scenario dataframe could be recovered. "
    "Review the candidate tables and build-test errors above."
)


# --------------------------------------------------------------------------------------
# 5. Install exact bindings required by the Notebook 54 function
# --------------------------------------------------------------------------------------

section4vb_scenario_source_df = (
    section3barc_selected_source_df
    .copy()
    .reset_index(drop=True)
)


SECTION4WF_SCENARIO_ID_COLUMN = (
    section3barc_selected_scenario_column
)


assert (
    SECTION4WF_SCENARIO_ID_COLUMN
    in section4vb_scenario_source_df.columns
)


assert not section4vb_scenario_source_df.empty


print()
print("INSTALLED NOTEBOOK 54 SCENARIO BINDINGS")
print("-" * 100)

print(
    "Source type:",
    section3barc_selected_source_type,
)

print(
    "Source name:",
    section3barc_selected_source_name,
)

print(
    "Scenario ID column:",
    SECTION4WF_SCENARIO_ID_COLUMN,
)

print(
    "Rows:",
    len(
        section4vb_scenario_source_df
    ),
)

print(
    "Unique scenarios:",
    section4vb_scenario_source_df[
        SECTION4WF_SCENARIO_ID_COLUMN
    ]
    .astype(str)
    .nunique(),
)


# --------------------------------------------------------------------------------------
# 6. Confirm all 184 evaluation cases can locate their source scenario
# --------------------------------------------------------------------------------------

required_scenario_ids_3barc = set(
    section3ba_score_cases_df[
        "source_scenario_id"
    ]
    .astype(str)
)


available_scenario_ids_3barc = set(
    section4vb_scenario_source_df[
        SECTION4WF_SCENARIO_ID_COLUMN
    ]
    .astype(str)
)


missing_scenario_ids_3barc = sorted(
    required_scenario_ids_3barc
    -
    available_scenario_ids_3barc
)


print()
print("SCENARIO COVERAGE VALIDATION")
print("-" * 100)

print(
    "Required scenario IDs:",
    len(
        required_scenario_ids_3barc
    ),
)

print(
    "Available scenario IDs:",
    len(
        available_scenario_ids_3barc
    ),
)

print(
    "Missing scenario IDs:",
    missing_scenario_ids_3barc,
)


assert not missing_scenario_ids_3barc, (
    "The recovered scenario source does not cover all balanced evaluation cases: "
    f"{missing_scenario_ids_3barc}"
)


# --------------------------------------------------------------------------------------
# 7. Test the exact builder on one balanced case
# --------------------------------------------------------------------------------------

section3barc_test_row = (
    section3ba_score_cases_df.iloc[0]
)


section3barc_test_state = (
    build_section4wf_evaluation_state(
        section3barc_test_row
    )
)


assert section3barc_test_state is not None


print()
print("EXACT BUILDER SINGLE-CASE TEST")
print("-" * 100)

print(
    "Comparison case:",
    section3barc_test_row[
        "comparison_case_id"
    ],
)

print(
    "Source scenario:",
    section3barc_test_row[
        "source_scenario_id"
    ],
)

print(
    "State type:",
    type(
        section3barc_test_state
    ).__name__,
)

print(
    "Current player:",
    getattr(
        section3barc_test_state,
        "current_player",
        None,
    ),
)

print(
    "Turn number:",
    getattr(
        section3barc_test_state,
        "turn_number",
        None,
    ),
)


# --------------------------------------------------------------------------------------
# 8. Save recovery report
# --------------------------------------------------------------------------------------

section3barc_summary = {
    "status":
        "EXACT_NOTEBOOK54_SCENARIO_SOURCE_BINDING_RECOVERED",

    "source_type":
        section3barc_selected_source_type,

    "source_name":
        section3barc_selected_source_name,

    "scenario_id_column":
        SECTION4WF_SCENARIO_ID_COLUMN,

    "scenario_source_rows":
        int(
            len(
                section4vb_scenario_source_df
            )
        ),

    "unique_scenarios":
        int(
            section4vb_scenario_source_df[
                SECTION4WF_SCENARIO_ID_COLUMN
            ]
            .astype(str)
            .nunique()
        ),

    "required_scenarios":
        int(
            len(
                required_scenario_ids_3barc
            )
        ),

    "missing_scenarios":
        missing_scenario_ids_3barc,

    "exact_builder_single_case_passed":
        True,

    "next_stage":
        "RERUN_EXACT_PROBABILITY_RECONSTRUCTION_VALIDATION",
}


print()
print("SECTION 3B-A REPAIR C SUMMARY")
print("-" * 100)

for key, value in (
    section3barc_summary.items()
):

    print(
        f"{key:58}: {value}"
    )


SECTION3BARC_RUNTIME_CANDIDATES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barc_runtime_scenario_candidates.csv"
)

SECTION3BARC_REPORT_CANDIDATES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barc_report_scenario_candidates.csv"
)

SECTION3BARC_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barc_scenario_source_recovery_summary.json"
)


section3barc_runtime_candidate_df.to_csv(
    SECTION3BARC_RUNTIME_CANDIDATES_FILE,
    index=False,
)

section3barc_report_candidate_df.to_csv(
    SECTION3BARC_REPORT_CANDIDATES_FILE,
    index=False,
)


with open(
    SECTION3BARC_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barc_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    "✅ SECTION 3B-A EXACT SCENARIO SOURCE BINDING RECOVERED"
)


# In[39]:


# ======================================================================================
# SECTION 3B-A REPAIR D — VALIDATE EXACT PROBABILITY RECONSTRUCTION
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR D — VALIDATE EXACT PROBABILITY RECONSTRUCTION")
print("=" * 100)

from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate exact restored dependencies
# --------------------------------------------------------------------------------------

SECTION3BARD_REQUIRED_OBJECTS = [
    "build_section4wf_evaluation_state",
    "build_legality_aware_feature_row",
    "build_replay_battle_state",
    "section4vb_scenario_source_df",
    "SECTION4WF_SCENARIO_ID_COLUMN",
    "section3ba_expanded_case_results_df",
    "section3ba_score_cases_df",
    "baseline_policy_model",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "REPORTS_DIRECTORY",
]


section3bard_missing_objects = [
    object_name
    for object_name in SECTION3BARD_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3B-A REPAIR D OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3BARD_REQUIRED_OBJECTS:

    print(
        f"{object_name:52}: "
        f"{object_name in globals()}"
    )


assert not section3bard_missing_objects, (
    "Required exact-reconstruction objects are missing: "
    f"{section3bard_missing_objects}"
)


assert callable(
    build_section4wf_evaluation_state
)


assert callable(
    build_legality_aware_feature_row
)


assert len(
    section3ba_score_cases_df
) == 184


assert len(
    baseline_encoded_feature_names
) == 64


# --------------------------------------------------------------------------------------
# 2. Resolve legal-move source column
# --------------------------------------------------------------------------------------

SECTION3BARD_LEGAL_MOVE_COLUMN_CANDIDATES = [
    "runtime_legal_moves",
    "expected_legal_moves",
    "legal_moves",
]


SECTION3BARD_LEGAL_MOVE_COLUMN = next(
    (
        column_name
        for column_name
        in SECTION3BARD_LEGAL_MOVE_COLUMN_CANDIDATES
        if column_name
        in section3ba_expanded_case_results_df.columns
    ),
    None,
)


assert SECTION3BARD_LEGAL_MOVE_COLUMN is not None, (
    "No legal-move column was found. Checked: "
    f"{SECTION3BARD_LEGAL_MOVE_COLUMN_CANDIDATES}"
)


print()
print("LEGAL-MOVE SOURCE")
print("-" * 100)

print(
    "Resolved legal-move column:",
    SECTION3BARD_LEGAL_MOVE_COLUMN,
)


# --------------------------------------------------------------------------------------
# 3. Normalize stored legal-move collections
# --------------------------------------------------------------------------------------

def normalize_section3bard_legal_moves(
    legal_moves_value,
):
    """
    Convert CSV legal-move evidence to a clean list of move names.
    """

    if legal_moves_value is None:

        return []


    if isinstance(
        legal_moves_value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            str(
                move_name
            ).strip()
            for move_name in legal_moves_value
            if str(
                move_name
            ).strip()
        ]


    if isinstance(
        legal_moves_value,
        str,
    ):

        stripped_value = legal_moves_value.strip()


        if not stripped_value:

            return []


        try:

            parsed_value = ast.literal_eval(
                stripped_value
            )

            if isinstance(
                parsed_value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):

                return [
                    str(
                        move_name
                    ).strip()
                    for move_name in parsed_value
                    if str(
                        move_name
                    ).strip()
                ]

        except Exception:

            pass


        return [
            move_name.strip()
            for move_name in stripped_value.split(",")
            if move_name.strip()
        ]


    return [
        str(
            legal_moves_value
        ).strip()
    ]


# --------------------------------------------------------------------------------------
# 4. Build exact case lookup
# --------------------------------------------------------------------------------------

section3bard_case_lookup_df = (
    section3ba_expanded_case_results_df
    .copy()
    .drop_duplicates(
        subset=[
            "comparison_case_id",
        ]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


assert len(
    section3bard_case_lookup_df
) == 184


assert section3bard_case_lookup_df.index.is_unique


# --------------------------------------------------------------------------------------
# 5. Resolve target model-class positions
# --------------------------------------------------------------------------------------

section3bard_model_classes = [
    str(
        class_name
    ).strip()
    for class_name in baseline_policy_model.classes_
]


section3bard_normalized_classes = [
    class_name.lower()
    for class_name in section3bard_model_classes
]


assert "quick attack" in section3bard_normalized_classes
assert "ascension" in section3bard_normalized_classes


SECTION3BARD_QUICK_ATTACK_INDEX = (
    section3bard_normalized_classes.index(
        "quick attack"
    )
)


SECTION3BARD_ASCENSION_INDEX = (
    section3bard_normalized_classes.index(
        "ascension"
    )
)


# --------------------------------------------------------------------------------------
# 6. Reconstruct and rescore all 184 cases
# --------------------------------------------------------------------------------------

section3bard_validation_rows = []
section3bard_error_rows = []


for _, score_row in (
    section3ba_score_cases_df.iterrows()
):

    comparison_case_id = str(
        score_row[
            "comparison_case_id"
        ]
    )


    try:

        source_row = section3bard_case_lookup_df.loc[
            comparison_case_id
        ]


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = (
            normalize_section3bard_legal_moves(
                source_row[
                    SECTION3BARD_LEGAL_MOVE_COLUMN
                ]
            )
        )


        assert legal_moves, (
            f"No legal moves recovered for {comparison_case_id}."
        )


        raw_feature_df = (
            build_legality_aware_feature_row(
                battle_state=
                    evaluation_state,

                legal_moves=
                    legal_moves,

                side_mode=
                    "PRESERVE",
            )
        )


        assert raw_feature_df.shape == (
            1,
            41,
        )


        transformed_features = (
            baseline_preprocessor.transform(
                raw_feature_df
            )
        )


        assert transformed_features.shape[
            1
        ] == 64


        probability_vector = np.asarray(
            baseline_policy_model.predict_proba(
                transformed_features
            )
        )[0]


        reconstructed_quick_attack_probability = float(
            probability_vector[
                SECTION3BARD_QUICK_ATTACK_INDEX
            ]
        )


        reconstructed_ascension_probability = float(
            probability_vector[
                SECTION3BARD_ASCENSION_INDEX
            ]
        )


        saved_quick_attack_probability = float(
            score_row[
                "quick_attack_probability"
            ]
        )


        saved_ascension_probability = float(
            score_row[
                "ascension_probability"
            ]
        )


        reconstructed_prediction = str(
            section3bard_model_classes[
                int(
                    np.argmax(
                        probability_vector
                    )
                )
            ]
        )


        saved_prediction = str(
            score_row.get(
                "predicted_action",
                "",
            )
        )


        section3bard_validation_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "expanded_pair_id":
                    str(
                        score_row[
                            "expanded_pair_id"
                        ]
                    ),

                "evaluation_side":
                    str(
                        score_row[
                            "evaluation_side"
                        ]
                    ),

                "source_scenario_id":
                    str(
                        score_row[
                            "source_scenario_id"
                        ]
                    ),

                "legal_moves":
                    legal_moves,

                "raw_feature_columns":
                    int(
                        raw_feature_df.shape[
                            1
                        ]
                    ),

                "encoded_feature_columns":
                    int(
                        transformed_features.shape[
                            1
                        ]
                    ),

                "reconstructed_quick_attack_probability":
                    reconstructed_quick_attack_probability,

                "saved_quick_attack_probability":
                    saved_quick_attack_probability,

                "quick_attack_absolute_difference":
                    abs(
                        reconstructed_quick_attack_probability
                        -
                        saved_quick_attack_probability
                    ),

                "reconstructed_ascension_probability":
                    reconstructed_ascension_probability,

                "saved_ascension_probability":
                    saved_ascension_probability,

                "ascension_absolute_difference":
                    abs(
                        reconstructed_ascension_probability
                        -
                        saved_ascension_probability
                    ),

                "reconstructed_prediction":
                    reconstructed_prediction,

                "saved_prediction":
                    saved_prediction,

                "prediction_agreement":
                    (
                        reconstructed_prediction.strip().lower()
                        ==
                        saved_prediction.strip().lower()
                    ),

                "reconstruction_success":
                    True,
            }
        )


    except Exception as error:

        section3bard_error_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

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


section3bard_probability_validation_df = pd.DataFrame(
    section3bard_validation_rows
)


section3bard_errors_df = pd.DataFrame(
    section3bard_error_rows,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


print()
print("EXACT RECONSTRUCTION ERRORS")
print("-" * 100)

if section3bard_errors_df.empty:

    print(
        "No exact reconstruction errors were recorded."
    )

else:

    display(
        section3bard_errors_df
    )


assert section3bard_errors_df.empty, (
    "One or more exact evaluation states failed reconstruction."
)


assert len(
    section3bard_probability_validation_df
) == 184


# --------------------------------------------------------------------------------------
# 7. Calculate exact agreement metrics
# --------------------------------------------------------------------------------------

section3bard_mean_quick_attack_difference = float(
    section3bard_probability_validation_df[
        "quick_attack_absolute_difference"
    ].mean()
)


section3bard_max_quick_attack_difference = float(
    section3bard_probability_validation_df[
        "quick_attack_absolute_difference"
    ].max()
)


section3bard_mean_ascension_difference = float(
    section3bard_probability_validation_df[
        "ascension_absolute_difference"
    ].mean()
)


section3bard_max_ascension_difference = float(
    section3bard_probability_validation_df[
        "ascension_absolute_difference"
    ].max()
)


section3bard_prediction_agreement_rate = float(
    section3bard_probability_validation_df[
        "prediction_agreement"
    ].astype(bool).mean()
)


SECTION3BARD_EXACT_TOLERANCE = 1e-10


section3bard_exact_probability_agreement = bool(
    section3bard_max_quick_attack_difference
    <=
    SECTION3BARD_EXACT_TOLERANCE
    and
    section3bard_max_ascension_difference
    <=
    SECTION3BARD_EXACT_TOLERANCE
)


print()
print("EXACT BASELINE PROBABILITY AGREEMENT")
print("-" * 100)

print(
    "Mean Quick Attack difference:",
    section3bard_mean_quick_attack_difference,
)

print(
    "Maximum Quick Attack difference:",
    section3bard_max_quick_attack_difference,
)

print(
    "Mean Ascension difference:",
    section3bard_mean_ascension_difference,
)

print(
    "Maximum Ascension difference:",
    section3bard_max_ascension_difference,
)

print(
    "Prediction agreement rate:",
    section3bard_prediction_agreement_rate,
)

print(
    "Exact probability agreement:",
    section3bard_exact_probability_agreement,
)


# --------------------------------------------------------------------------------------
# 8. Display any largest differences
# --------------------------------------------------------------------------------------

print()
print("LARGEST REMAINING PROBABILITY DIFFERENCES")
print("-" * 100)

display(
    section3bard_probability_validation_df
    .sort_values(
        [
            "quick_attack_absolute_difference",
            "ascension_absolute_difference",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
    .reset_index(drop=True)
)


# --------------------------------------------------------------------------------------
# 9. Determine readiness
# --------------------------------------------------------------------------------------

if (
    section3bard_exact_probability_agreement
    and
    section3bard_prediction_agreement_rate == 1.0
):

    section3bard_status = (
        "EXACT_NOTEBOOK54_PROBABILITY_RECONSTRUCTION_CONFIRMED"
    )

    section3bard_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )

elif section3bard_prediction_agreement_rate == 1.0:

    section3bard_status = (
        "PREDICTIONS_REPRODUCED_WITH_RESIDUAL_PROBABILITY_DRIFT"
    )

    section3bard_next_stage = (
        "RESIDUAL_FEATURE_RECONSTRUCTION_DIAGNOSTIC"
    )

else:

    section3bard_status = (
        "EXACT_NOTEBOOK54_RECONSTRUCTION_NOT_CONFIRMED"
    )

    section3bard_next_stage = (
        "RECONSTRUCTION_MISMATCH_DIAGNOSTIC"
    )


section3bard_summary = {
    "status":
        section3bard_status,

    "states_reconstructed":
        int(
            len(
                section3bard_probability_validation_df
            )
        ),

    "state_errors":
        int(
            len(
                section3bard_errors_df
            )
        ),

    "raw_feature_count":
        41,

    "encoded_feature_count":
        64,

    "mean_quick_attack_probability_difference":
        section3bard_mean_quick_attack_difference,

    "maximum_quick_attack_probability_difference":
        section3bard_max_quick_attack_difference,

    "mean_ascension_probability_difference":
        section3bard_mean_ascension_difference,

    "maximum_ascension_probability_difference":
        section3bard_max_ascension_difference,

    "prediction_agreement_rate":
        section3bard_prediction_agreement_rate,

    "exact_probability_agreement":
        section3bard_exact_probability_agreement,

    "next_stage":
        section3bard_next_stage,
}


print()
print("SECTION 3B-A REPAIR D SUMMARY")
print("-" * 100)

for key, value in (
    section3bard_summary.items()
):

    print(
        f"{key:66}: {value}"
    )


# --------------------------------------------------------------------------------------
# 10. Validation checks
# --------------------------------------------------------------------------------------

section3bard_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_states_reconstructed",

            "passed":
                len(
                    section3bard_probability_validation_df
                ) == 184,

            "value":
                len(
                    section3bard_probability_validation_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "no_reconstruction_errors",

            "passed":
                section3bard_errors_df.empty,

            "value":
                len(
                    section3bard_errors_df
                ),

            "expected":
                0,
        },
        {
            "check":
                "raw_schema_is_41_columns",

            "passed":
                section3bard_probability_validation_df[
                    "raw_feature_columns"
                ].eq(
                    41
                ).all(),

            "value":
                section3bard_probability_validation_df[
                    "raw_feature_columns"
                ].unique().tolist(),

            "expected":
                [
                    41
                ],
        },
        {
            "check":
                "encoded_schema_is_64_columns",

            "passed":
                section3bard_probability_validation_df[
                    "encoded_feature_columns"
                ].eq(
                    64
                ).all(),

            "value":
                section3bard_probability_validation_df[
                    "encoded_feature_columns"
                ].unique().tolist(),

            "expected":
                [
                    64
                ],
        },
        {
            "check":
                "prediction_agreement_complete",

            "passed":
                section3bard_prediction_agreement_rate
                == 1.0,

            "value":
                section3bard_prediction_agreement_rate,

            "expected":
                1.0,
        },
        {
            "check":
                "exact_probability_agreement",

            "passed":
                section3bard_exact_probability_agreement,

            "value":
                {
                    "quick_attack_max_difference":
                        section3bard_max_quick_attack_difference,

                    "ascension_max_difference":
                        section3bard_max_ascension_difference,
                },

            "expected":
                (
                    f"Both at most "
                    f"{SECTION3BARD_EXACT_TOLERANCE}"
                ),
        },
        {
            "check":
                "four_model_scoring_ready",

            "passed":
                section3bard_next_stage
                ==
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",

            "value":
                section3bard_next_stage,

            "expected":
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR D VALIDATION CHECKS")
print("-" * 100)

display(
    section3bard_validation_checks_df
)


section3bard_failed_checks = int(
    (
        ~section3bard_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


assert section3bard_failed_checks == 0, (
    "Exact Notebook 54 probability reconstruction is still incomplete. "
    "Review the largest differences and failed checks above."
)


# --------------------------------------------------------------------------------------
# 11. Save exact validation reports
# --------------------------------------------------------------------------------------

SECTION3BARD_PROBABILITY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bard_exact_probability_validation.csv"
)

SECTION3BARD_ERRORS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bard_exact_reconstruction_errors.csv"
)

SECTION3BARD_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bard_validation_checks.csv"
)

SECTION3BARD_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bard_exact_reconstruction_summary.json"
)


section3bard_probability_validation_df.to_csv(
    SECTION3BARD_PROBABILITY_FILE,
    index=False,
)

section3bard_errors_df.to_csv(
    SECTION3BARD_ERRORS_FILE,
    index=False,
)

section3bard_validation_checks_df.to_csv(
    SECTION3BARD_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARD_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3bard_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3bard_saved_files = [
    SECTION3BARD_PROBABILITY_FILE,
    SECTION3BARD_ERRORS_FILE,
    SECTION3BARD_VALIDATION_FILE,
    SECTION3BARD_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3bard_saved_files
)


print()
print("SAVED SECTION 3B-A REPAIR D REPORTS")
print("-" * 100)

for file_path in section3bard_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A EXACT NOTEBOOK 54 "
    "PROBABILITY RECONSTRUCTION CONFIRMED"
)


# In[23]:


# ======================================================================================
# SECTION 3B-A REPAIR E — RECOVER THE EXACT NOTEBOOK 54 SCENARIO-SOURCE ASSIGNMENT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR E — RECOVER THE EXACT NOTEBOOK 54 SCENARIO-SOURCE ASSIGNMENT")
print("=" * 100)

import ast
import json
import nbformat
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Read the completed Notebook 54 source
# --------------------------------------------------------------------------------------

with open(
    NOTEBOOK54_EXACT_SOURCE_PATH,
    "r",
    encoding="utf-8",
) as file:

    notebook54_repair_e_document = nbformat.read(
        file,
        as_version=4,
    )


notebook54_repair_e_code_cells = [
    cell
    for cell in notebook54_repair_e_document.cells
    if cell.cell_type == "code"
]


# --------------------------------------------------------------------------------------
# 2. Find exact assignments to the two required bindings
# --------------------------------------------------------------------------------------

SECTION3BARE_TARGET_NAMES = {
    "section4vb_scenario_source_df",
    "SECTION4WF_SCENARIO_ID_COLUMN",
}


section3bare_assignment_rows = []
section3bare_assignment_sources = {}


for cell_index, cell in enumerate(
    notebook54_repair_e_code_cells
):

    source_text = str(
        cell.source
    )

    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            continue


        target_nodes = (
            node.targets
            if isinstance(
                node,
                ast.Assign,
            )
            else [
                node.target
            ]
        )


        target_names = []


        for target_node in target_nodes:

            if isinstance(
                target_node,
                ast.Name,
            ):

                target_names.append(
                    target_node.id
                )


        for target_name in target_names:

            if target_name not in SECTION3BARE_TARGET_NAMES:

                continue


            assignment_source = ast.get_source_segment(
                source_text,
                node,
            )


            section3bare_assignment_rows.append(
                {
                    "target_name":
                        target_name,

                    "cell_index":
                        int(
                            cell_index
                        ),

                    "line_number":
                        int(
                            getattr(
                                node,
                                "lineno",
                                0,
                            )
                        ),

                    "assignment_source":
                        assignment_source,
                }
            )


            section3bare_assignment_sources[
                target_name
            ] = {
                "cell_index":
                    int(
                        cell_index
                    ),

                "cell_source":
                    source_text,

                "assignment_source":
                    assignment_source,
            }


section3bare_assignment_df = pd.DataFrame(
    section3bare_assignment_rows
)


print()
print("EXACT NOTEBOOK 54 ASSIGNMENTS")
print("-" * 100)

display(
    section3bare_assignment_df
)


assert set(
    section3bare_assignment_df[
        "target_name"
    ]
) == SECTION3BARE_TARGET_NAMES, (
    "The exact Notebook 54 assignments were not both located."
)


# --------------------------------------------------------------------------------------
# 3. Display complete source cells containing the assignments
# --------------------------------------------------------------------------------------

for target_name in [
    "SECTION4WF_SCENARIO_ID_COLUMN",
    "section4vb_scenario_source_df",
]:

    target_record = section3bare_assignment_sources[
        target_name
    ]


    print()
    print("=" * 100)
    print(
        f"FULL NOTEBOOK 54 SOURCE CELL FOR: {target_name}"
    )
    print(
        f"CODE-CELL INDEX: {target_record['cell_index']}"
    )
    print("=" * 100)

    print(
        target_record[
            "cell_source"
        ]
    )


# --------------------------------------------------------------------------------------
# 4. Execute only the exact source cells that define the bindings
# --------------------------------------------------------------------------------------

section3bare_executed_cell_indices = []


for target_name in [
    "SECTION4WF_SCENARIO_ID_COLUMN",
    "section4vb_scenario_source_df",
]:

    source_record = section3bare_assignment_sources[
        target_name
    ]


    source_cell_index = source_record[
        "cell_index"
    ]


    if source_cell_index in (
        section3bare_executed_cell_indices
    ):

        continue


    print()
    print(
        "Executing exact Notebook 54 source cell:",
        source_cell_index,
    )


    exec(
        source_record[
            "cell_source"
        ],
        globals(),
    )


    section3bare_executed_cell_indices.append(
        source_cell_index
    )


# --------------------------------------------------------------------------------------
# 5. Validate the exact bindings now installed
# --------------------------------------------------------------------------------------

assert (
    "section4vb_scenario_source_df"
    in globals()
)


assert isinstance(
    section4vb_scenario_source_df,
    pd.DataFrame,
)


assert not section4vb_scenario_source_df.empty


assert (
    "SECTION4WF_SCENARIO_ID_COLUMN"
    in globals()
)


assert (
    SECTION4WF_SCENARIO_ID_COLUMN
    in section4vb_scenario_source_df.columns
)


print()
print("EXACT SCENARIO-SOURCE BINDING INSTALLED")
print("-" * 100)

print(
    "Scenario ID column:",
    SECTION4WF_SCENARIO_ID_COLUMN,
)

print(
    "Rows:",
    len(
        section4vb_scenario_source_df
    ),
)

print(
    "Columns:",
    len(
        section4vb_scenario_source_df.columns
    ),
)

print(
    "Unique scenarios:",
    section4vb_scenario_source_df[
        SECTION4WF_SCENARIO_ID_COLUMN
    ]
    .astype(str)
    .nunique(),
)

print(
    "First columns:",
    section4vb_scenario_source_df.columns[
        :25
    ].tolist(),
)


# --------------------------------------------------------------------------------------
# 6. Confirm balanced-case scenario coverage
# --------------------------------------------------------------------------------------

section3bare_required_scenarios = set(
    section3ba_score_cases_df[
        "source_scenario_id"
    ]
    .astype(str)
)


section3bare_available_scenarios = set(
    section4vb_scenario_source_df[
        SECTION4WF_SCENARIO_ID_COLUMN
    ]
    .astype(str)
)


section3bare_missing_scenarios = sorted(
    section3bare_required_scenarios
    -
    section3bare_available_scenarios
)


assert not section3bare_missing_scenarios, (
    "The exact Notebook 54 scenario source does not cover all "
    f"balanced cases: {section3bare_missing_scenarios}"
)


section3bare_summary = {
    "status":
        "EXACT_NOTEBOOK54_SCENARIO_SOURCE_ASSIGNMENT_RESTORED",

    "executed_source_cells":
        section3bare_executed_cell_indices,

    "scenario_id_column":
        SECTION4WF_SCENARIO_ID_COLUMN,

    "scenario_source_rows":
        int(
            len(
                section4vb_scenario_source_df
            )
        ),

    "scenario_source_columns":
        int(
            len(
                section4vb_scenario_source_df.columns
            )
        ),

    "unique_scenarios":
        int(
            section4vb_scenario_source_df[
                SECTION4WF_SCENARIO_ID_COLUMN
            ]
            .astype(str)
            .nunique()
        ),

    "missing_required_scenarios":
        section3bare_missing_scenarios,

    "next_stage":
        "RERUN_REPAIR_D_EXACT_PROBABILITY_VALIDATION",
}


print()
print("SECTION 3B-A REPAIR E SUMMARY")
print("-" * 100)

for key, value in section3bare_summary.items():

    print(
        f"{key:62}: {value}"
    )


SECTION3BARE_ASSIGNMENTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bare_exact_scenario_assignment_inventory.csv"
)

SECTION3BARE_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bare_exact_scenario_assignment_summary.json"
)


section3bare_assignment_df.to_csv(
    SECTION3BARE_ASSIGNMENTS_FILE,
    index=False,
)


with open(
    SECTION3BARE_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3bare_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    "✅ SECTION 3B-A EXACT NOTEBOOK 54 "
    "SCENARIO-SOURCE ASSIGNMENT RESTORED"
)


# In[24]:


print("AUTHORITATIVE_REPLAY_SOURCE_FILE")
print(AUTHORITATIVE_REPLAY_SOURCE_FILE)

print()

print("section4vb_saved_starting_state_sources_df")
display(
    section4vb_saved_starting_state_sources_df[
        [
            "csv_path",
            "all_starting_fields_present",
            "has_scenario_id",
            "has_condition_id",
        ]
    ]
)


# In[25]:


# ======================================================================================
# SECTION 3B-A REPAIR F — LOCATE AUTHORITATIVE NOTEBOOK 52 REPLAY SOURCE DIRECTLY
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR F — LOCATE AUTHORITATIVE NOTEBOOK 52 REPLAY SOURCE DIRECTLY")
print("=" * 100)

from pathlib import Path
import pandas as pd


NOTEBOOK52_SECTION5_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook52"
    / "section5"
)

assert NOTEBOOK52_SECTION5_DIRECTORY.exists(), (
    "Notebook 52 Section 5 report directory was not found: "
    f"{NOTEBOOK52_SECTION5_DIRECTORY}"
)


REQUIRED_REPLAY_COLUMN_GROUPS = {
    "scenario_id": [
        "scenario_id",
        "expanded_scenario_id",
    ],

    "condition_id": [
        "condition_id",
        "condition",
    ],

    "player_card": [
        "player_card",
        "player_pokemon",
    ],

    "opponent_card": [
        "opponent_card",
        "opponent_pokemon",
    ],

    "starting_side": [
        "starting_side",
        "current_player",
        "start_side",
    ],

    "starting_player_hp": [
        "starting_player_hp",
        "initial_player_hp",
        "player_hp_start",
    ],

    "starting_opponent_hp": [
        "starting_opponent_hp",
        "initial_opponent_hp",
        "opponent_hp_start",
    ],

    "starting_player_energy": [
        "starting_player_energy",
        "initial_player_energy",
        "player_energy_start",
    ],

    "starting_opponent_energy": [
        "starting_opponent_energy",
        "initial_opponent_energy",
        "opponent_energy_start",
    ],
}


def resolve_candidate_column(
    dataframe,
    candidates,
):
    for column_name in candidates:
        if column_name in dataframe.columns:
            return column_name

    return None


candidate_rows = []
candidate_dataframes = {}


for csv_path in sorted(
    NOTEBOOK52_SECTION5_DIRECTORY.glob("*.csv")
):

    try:
        candidate_df = pd.read_csv(csv_path)

    except Exception as error:
        candidate_rows.append(
            {
                "csv_path": str(csv_path),
                "rows": 0,
                "columns": 0,
                "all_starting_fields_present": False,
                "unique_scenarios": 0,
                "read_error": f"{type(error).__name__}: {error}",
            }
        )
        continue

    resolved_columns = {
        logical_name:
            resolve_candidate_column(
                candidate_df,
                candidates,
            )
        for logical_name, candidates
        in REQUIRED_REPLAY_COLUMN_GROUPS.items()
    }

    all_fields_present = all(
        resolved_column is not None
        for resolved_column in resolved_columns.values()
    )

    scenario_column = resolved_columns[
        "scenario_id"
    ]

    unique_scenarios = (
        int(
            candidate_df[
                scenario_column
            ]
            .astype(str)
            .nunique()
        )
        if scenario_column is not None
        else 0
    )

    candidate_rows.append(
        {
            "csv_path":
                str(csv_path),

            "file_name":
                csv_path.name,

            "rows":
                int(len(candidate_df)),

            "columns":
                int(len(candidate_df.columns)),

            "all_starting_fields_present":
                all_fields_present,

            "unique_scenarios":
                unique_scenarios,

            "scenario_column":
                scenario_column,

            "resolved_columns":
                resolved_columns,

            "read_error":
                "",
        }
    )

    candidate_dataframes[
        str(csv_path)
    ] = candidate_df


section3barf_candidate_sources_df = pd.DataFrame(
    candidate_rows
)


print()
print("NOTEBOOK 52 SECTION 5 REPLAY SOURCE CANDIDATES")
print("-" * 100)

display(
    section3barf_candidate_sources_df
    .sort_values(
        [
            "all_starting_fields_present",
            "unique_scenarios",
            "rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


valid_candidates_df = (
    section3barf_candidate_sources_df.loc[
        section3barf_candidate_sources_df[
            "all_starting_fields_present"
        ].astype(bool)
        &
        section3barf_candidate_sources_df[
            "unique_scenarios"
        ].ge(24)
    ]
    .copy()
)


assert not valid_candidates_df.empty, (
    "No complete Notebook 52 replay source with at least "
    "24 scenarios was found."
)


selected_candidate = (
    valid_candidates_df
    .sort_values(
        [
            "unique_scenarios",
            "rows",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .iloc[0]
)


AUTHORITATIVE_REPLAY_SOURCE_FILE = Path(
    selected_candidate[
        "csv_path"
    ]
)


notebook52_battle_results_df = (
    candidate_dataframes[
        str(
            AUTHORITATIVE_REPLAY_SOURCE_FILE
        )
    ]
    .copy()
)


resolved_columns_3barf = selected_candidate[
    "resolved_columns"
]


SCENARIO_ID_COLUMN = resolved_columns_3barf[
    "scenario_id"
]

CONDITION_ID_COLUMN = resolved_columns_3barf[
    "condition_id"
]

PLAYER_CARD_COLUMN = resolved_columns_3barf[
    "player_card"
]

OPPONENT_CARD_COLUMN = resolved_columns_3barf[
    "opponent_card"
]

STARTING_SIDE_COLUMN = resolved_columns_3barf[
    "starting_side"
]

STARTING_PLAYER_HP_COLUMN = resolved_columns_3barf[
    "starting_player_hp"
]

STARTING_OPPONENT_HP_COLUMN = resolved_columns_3barf[
    "starting_opponent_hp"
]

STARTING_PLAYER_ENERGY_COLUMN = resolved_columns_3barf[
    "starting_player_energy"
]

STARTING_OPPONENT_ENERGY_COLUMN = resolved_columns_3barf[
    "starting_opponent_energy"
]


section4vb_scenario_source_df = (
    notebook52_battle_results_df
    .drop_duplicates(
        subset=[
            SCENARIO_ID_COLUMN
        ],
        keep="first",
    )
    .copy()
    .reset_index(drop=True)
)


SECTION4WF_SCENARIO_ID_COLUMN = (
    SCENARIO_ID_COLUMN
)


print()
print("AUTHORITATIVE REPLAY SOURCE SELECTED")
print("-" * 100)

print(
    "File:",
    AUTHORITATIVE_REPLAY_SOURCE_FILE,
)

print(
    "Source rows:",
    len(
        notebook52_battle_results_df
    ),
)

print(
    "Authoritative scenarios:",
    len(
        section4vb_scenario_source_df
    ),
)

print(
    "Scenario ID column:",
    SCENARIO_ID_COLUMN,
)


assert AUTHORITATIVE_REPLAY_SOURCE_FILE.exists()

assert len(
    section4vb_scenario_source_df
) == 24, (
    "Expected exactly 24 authoritative replay scenarios, "
    f"found {len(section4vb_scenario_source_df)}."
)


required_scenario_ids = set(
    section3ba_score_cases_df[
        "source_scenario_id"
    ]
    .astype(str)
)


available_scenario_ids = set(
    section4vb_scenario_source_df[
        SECTION4WF_SCENARIO_ID_COLUMN
    ]
    .astype(str)
)


missing_scenario_ids = sorted(
    required_scenario_ids
    -
    available_scenario_ids
)


assert not missing_scenario_ids, (
    "Balanced evaluation scenarios missing from authoritative source: "
    f"{missing_scenario_ids}"
)


print()
print(
    "✅ AUTHORITATIVE NOTEBOOK 52 REPLAY SOURCE "
    "LOCATED AND BOUND"
)


# In[26]:


# ======================================================================================
# SECTION 3B-A REPAIR G — EXACT RAW FEATURE EQUALITY AUDIT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR G — EXACT RAW FEATURE EQUALITY AUDIT")
print("=" * 100)

from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate required objects
# --------------------------------------------------------------------------------------

SECTION3BARG_REQUIRED_OBJECTS = [
    "build_section4wf_evaluation_state",
    "build_legality_aware_feature_row",
    "section3ba_expanded_case_results_df",
    "section3ba_score_cases_df",
    "section3ba_counterfactual_cases_df",
    "section4vb_scenario_source_df",
    "SECTION4WF_SCENARIO_ID_COLUMN",
    "baseline_preprocessor",
    "baseline_policy_model",
    "REPORTS_DIRECTORY",
]


section3barg_missing_objects = [
    object_name
    for object_name in SECTION3BARG_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3B-A REPAIR G OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3BARG_REQUIRED_OBJECTS:

    print(
        f"{object_name:54}: "
        f"{object_name in globals()}"
    )


assert not section3barg_missing_objects, (
    "Repair G required objects are missing: "
    f"{section3barg_missing_objects}"
)


assert callable(
    build_section4wf_evaluation_state
)


assert callable(
    build_legality_aware_feature_row
)


assert len(
    section3ba_score_cases_df
) == 184


# --------------------------------------------------------------------------------------
# 2. Resolve legal-move source
# --------------------------------------------------------------------------------------

SECTION3BARG_LEGAL_MOVE_COLUMN_CANDIDATES = [
    "runtime_legal_moves",
    "expected_legal_moves",
    "legal_moves",
]


SECTION3BARG_LEGAL_MOVE_COLUMN = next(
    (
        column_name
        for column_name
        in SECTION3BARG_LEGAL_MOVE_COLUMN_CANDIDATES
        if column_name
        in section3ba_expanded_case_results_df.columns
    ),
    None,
)


assert SECTION3BARG_LEGAL_MOVE_COLUMN is not None, (
    "No legal-move source column was found."
)


# --------------------------------------------------------------------------------------
# 3. Normalize legal-move values
# --------------------------------------------------------------------------------------

def normalize_section3barg_legal_moves(
    legal_moves_value,
):
    if legal_moves_value is None:

        return []


    if isinstance(
        legal_moves_value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            str(
                move_name
            ).strip()
            for move_name in legal_moves_value
            if str(
                move_name
            ).strip()
        ]


    if isinstance(
        legal_moves_value,
        str,
    ):

        stripped_value = legal_moves_value.strip()


        if not stripped_value:

            return []


        try:

            parsed_value = ast.literal_eval(
                stripped_value
            )

            if isinstance(
                parsed_value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):

                return [
                    str(
                        move_name
                    ).strip()
                    for move_name in parsed_value
                    if str(
                        move_name
                    ).strip()
                ]

        except Exception:

            pass


        return [
            item.strip()
            for item in stripped_value.split(",")
            if item.strip()
        ]


    return [
        str(
            legal_moves_value
        ).strip()
    ]


# --------------------------------------------------------------------------------------
# 4. Build the authoritative case-result lookup
# --------------------------------------------------------------------------------------

section3barg_case_lookup_df = (
    section3ba_expanded_case_results_df
    .copy()
    .drop_duplicates(
        subset=[
            "comparison_case_id",
        ]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


assert len(
    section3barg_case_lookup_df
) == 184


# --------------------------------------------------------------------------------------
# 5. Reconstruct the 41 raw features for all 184 cases
# --------------------------------------------------------------------------------------

section3barg_raw_feature_rows = []
section3barg_reconstruction_errors = []


for _, score_row in (
    section3ba_score_cases_df.iterrows()
):

    comparison_case_id = str(
        score_row[
            "comparison_case_id"
        ]
    )


    try:

        case_source_row = (
            section3barg_case_lookup_df.loc[
                comparison_case_id
            ]
        )


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = (
            normalize_section3barg_legal_moves(
                case_source_row[
                    SECTION3BARG_LEGAL_MOVE_COLUMN
                ]
            )
        )


        raw_feature_df = (
            build_legality_aware_feature_row(
                battle_state=
                    evaluation_state,

                legal_moves=
                    legal_moves,

                side_mode=
                    "PRESERVE",
            )
        )


        assert len(
            raw_feature_df
        ) == 1


        assert raw_feature_df.shape[
            1
        ] == 41


        raw_record = (
            raw_feature_df.iloc[0]
            .to_dict()
        )


        raw_record.update(
            {
                "comparison_case_id":
                    comparison_case_id,

                "expanded_pair_id":
                    str(
                        score_row[
                            "expanded_pair_id"
                        ]
                    ),

                "evaluation_side":
                    str(
                        score_row[
                            "evaluation_side"
                        ]
                    ),

                "source_scenario_id":
                    str(
                        score_row[
                            "source_scenario_id"
                        ]
                    ),

                "expansion_variant_id":
                    str(
                        score_row.get(
                            "expansion_variant_id",
                            "",
                        )
                    ),
            }
        )


        section3barg_raw_feature_rows.append(
            raw_record
        )


    except Exception as error:

        section3barg_reconstruction_errors.append(
            {
                "comparison_case_id":
                    comparison_case_id,

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


section3barg_reconstructed_raw_features_df = (
    pd.DataFrame(
        section3barg_raw_feature_rows
    )
)


section3barg_reconstruction_errors_df = pd.DataFrame(
    section3barg_reconstruction_errors,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


print()
print("RAW FEATURE RECONSTRUCTION ERRORS")
print("-" * 100)

if section3barg_reconstruction_errors_df.empty:

    print(
        "No raw-feature reconstruction errors were recorded."
    )

else:

    display(
        section3barg_reconstruction_errors_df
    )


assert section3barg_reconstruction_errors_df.empty


assert len(
    section3barg_reconstructed_raw_features_df
) == 184


# --------------------------------------------------------------------------------------
# 6. Identify the exact 41 raw-feature columns
# --------------------------------------------------------------------------------------

SECTION3BARG_METADATA_COLUMNS = {
    "comparison_case_id",
    "expanded_pair_id",
    "evaluation_side",
    "source_scenario_id",
    "expansion_variant_id",
}


section3barg_raw_feature_columns = [
    column_name
    for column_name
    in section3barg_reconstructed_raw_features_df.columns
    if column_name not in SECTION3BARG_METADATA_COLUMNS
]


assert len(
    section3barg_raw_feature_columns
) == 41


print()
print("RECONSTRUCTED RAW FEATURE COLUMNS")
print("-" * 100)

for feature_index, feature_name in enumerate(
    section3barg_raw_feature_columns,
    start=1,
):

    print(
        f"{feature_index:02d}. {feature_name}"
    )


# --------------------------------------------------------------------------------------
# 7. Search saved Notebook 54 reports for matching raw-feature evidence
# --------------------------------------------------------------------------------------

SECTION3BARG_SAVED_SOURCES = {
    "score_cases":
        section3ba_score_cases_df,

    "counterfactual_cases":
        section3ba_counterfactual_cases_df,

    "expanded_case_results":
        section3ba_expanded_case_results_df,
}


section3barg_saved_source_profile_rows = []


for source_name, source_df in (
    SECTION3BARG_SAVED_SOURCES.items()
):

    direct_feature_matches = [
        feature_name
        for feature_name
        in section3barg_raw_feature_columns
        if feature_name in source_df.columns
    ]


    prefixed_feature_matches = [
        column_name
        for column_name in source_df.columns
        if any(
            str(
                column_name
            ).endswith(
                feature_name
            )
            for feature_name
            in section3barg_raw_feature_columns
        )
    ]


    section3barg_saved_source_profile_rows.append(
        {
            "source_name":
                source_name,

            "rows":
                int(
                    len(
                        source_df
                    )
                ),

            "columns":
                int(
                    len(
                        source_df.columns
                    )
                ),

            "direct_raw_feature_matches":
                int(
                    len(
                        direct_feature_matches
                    )
                ),

            "suffix_or_prefixed_matches":
                int(
                    len(
                        prefixed_feature_matches
                    )
                ),

            "direct_matching_columns":
                json.dumps(
                    direct_feature_matches
                ),

            "prefixed_matching_columns":
                json.dumps(
                    prefixed_feature_matches
                ),
        }
    )


section3barg_saved_source_profile_df = pd.DataFrame(
    section3barg_saved_source_profile_rows
)


print()
print("SAVED RAW-FEATURE EVIDENCE PROFILE")
print("-" * 100)

display(
    section3barg_saved_source_profile_df
)


# --------------------------------------------------------------------------------------
# 8. Select the strongest saved source
# --------------------------------------------------------------------------------------

section3barg_best_source_row = (
    section3barg_saved_source_profile_df
    .sort_values(
        [
            "direct_raw_feature_matches",
            "suffix_or_prefixed_matches",
            "rows",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .iloc[0]
)


section3barg_best_source_name = str(
    section3barg_best_source_row[
        "source_name"
    ]
)


section3barg_best_saved_source_df = (
    SECTION3BARG_SAVED_SOURCES[
        section3barg_best_source_name
    ]
    .copy()
)


section3barg_direct_match_count = int(
    section3barg_best_source_row[
        "direct_raw_feature_matches"
    ]
)


print()
print("BEST SAVED RAW-FEATURE SOURCE")
print("-" * 100)

print(
    "Source:",
    section3barg_best_source_name,
)

print(
    "Direct feature matches:",
    section3barg_direct_match_count,
)

print(
    "Rows:",
    len(
        section3barg_best_saved_source_df
    ),
)


# --------------------------------------------------------------------------------------
# 9. Direct raw-feature equality audit when saved values are available
# --------------------------------------------------------------------------------------

section3barg_feature_equality_rows = []
section3barg_case_difference_rows = []


if (
    section3barg_direct_match_count > 0
    and
    "comparison_case_id"
    in section3barg_best_saved_source_df.columns
):

    saved_case_lookup_df = (
        section3barg_best_saved_source_df
        .drop_duplicates(
            subset=[
                "comparison_case_id",
            ]
        )
        .set_index(
            "comparison_case_id",
            drop=False,
        )
    )


    for feature_name in section3barg_raw_feature_columns:

        if feature_name not in (
            section3barg_best_saved_source_df.columns
        ):

            continue


        compared_rows = 0
        equal_rows = 0
        different_rows = 0
        missing_saved_rows = 0
        first_difference_case = None
        first_reconstructed_value = None
        first_saved_value = None


        for _, reconstructed_row in (
            section3barg_reconstructed_raw_features_df.iterrows()
        ):

            comparison_case_id = str(
                reconstructed_row[
                    "comparison_case_id"
                ]
            )


            if comparison_case_id not in (
                saved_case_lookup_df.index
            ):

                missing_saved_rows += 1
                continue


            saved_value = (
                saved_case_lookup_df.loc[
                    comparison_case_id,
                    feature_name,
                ]
            )


            reconstructed_value = (
                reconstructed_row[
                    feature_name
                ]
            )


            compared_rows += 1


            reconstructed_missing = pd.isna(
                reconstructed_value
            )

            saved_missing = pd.isna(
                saved_value
            )


            if (
                reconstructed_missing
                and
                saved_missing
            ):

                values_equal = True


            elif (
                reconstructed_missing
                !=
                saved_missing
            ):

                values_equal = False


            else:

                try:

                    reconstructed_numeric = float(
                        reconstructed_value
                    )

                    saved_numeric = float(
                        saved_value
                    )


                    values_equal = bool(
                        np.isclose(
                            reconstructed_numeric,
                            saved_numeric,
                            rtol=0.0,
                            atol=1e-12,
                            equal_nan=True,
                        )
                    )


                except (
                    TypeError,
                    ValueError,
                ):

                    values_equal = (
                        str(
                            reconstructed_value
                        ).strip()
                        ==
                        str(
                            saved_value
                        ).strip()
                    )


            if values_equal:

                equal_rows += 1

            else:

                different_rows += 1


                if first_difference_case is None:

                    first_difference_case = (
                        comparison_case_id
                    )

                    first_reconstructed_value = (
                        reconstructed_value
                    )

                    first_saved_value = (
                        saved_value
                    )


                section3barg_case_difference_rows.append(
                    {
                        "comparison_case_id":
                            comparison_case_id,

                        "source_scenario_id":
                            reconstructed_row[
                                "source_scenario_id"
                            ],

                        "evaluation_side":
                            reconstructed_row[
                                "evaluation_side"
                            ],

                        "feature_name":
                            feature_name,

                        "reconstructed_value":
                            reconstructed_value,

                        "saved_value":
                            saved_value,
                    }
                )


        section3barg_feature_equality_rows.append(
            {
                "feature_name":
                    feature_name,

                "compared_rows":
                    compared_rows,

                "equal_rows":
                    equal_rows,

                "different_rows":
                    different_rows,

                "equality_rate":
                    (
                        equal_rows
                        /
                        compared_rows
                        if compared_rows > 0
                        else np.nan
                    ),

                "missing_saved_rows":
                    missing_saved_rows,

                "first_difference_case":
                    first_difference_case,

                "first_reconstructed_value":
                    first_reconstructed_value,

                "first_saved_value":
                    first_saved_value,
            }
        )


section3barg_feature_equality_df = pd.DataFrame(
    section3barg_feature_equality_rows,
    columns=[
        "feature_name",
        "compared_rows",
        "equal_rows",
        "different_rows",
        "equality_rate",
        "missing_saved_rows",
        "first_difference_case",
        "first_reconstructed_value",
        "first_saved_value",
    ],
)


section3barg_case_differences_df = pd.DataFrame(
    section3barg_case_difference_rows,
    columns=[
        "comparison_case_id",
        "source_scenario_id",
        "evaluation_side",
        "feature_name",
        "reconstructed_value",
        "saved_value",
    ],
)


print()
print("RAW FEATURE EQUALITY RESULTS")
print("-" * 100)

if section3barg_feature_equality_df.empty:

    print(
        "The saved reports do not contain direct copies of the 41 raw features."
    )

else:

    display(
        section3barg_feature_equality_df
        .sort_values(
            [
                "equality_rate",
                "different_rows",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


print()
print("FIRST RAW FEATURE DIFFERENCES")
print("-" * 100)

if section3barg_case_differences_df.empty:

    print(
        "No direct raw-feature differences were available."
    )

else:

    display(
        section3barg_case_differences_df
        .head(100)
    )


# --------------------------------------------------------------------------------------
# 10. Encoded-feature sensitivity audit
# --------------------------------------------------------------------------------------

section3barg_encoded_rows = []


for _, reconstructed_row in (
    section3barg_reconstructed_raw_features_df.iterrows()
):

    comparison_case_id = str(
        reconstructed_row[
            "comparison_case_id"
        ]
    )


    raw_feature_df = pd.DataFrame(
        [
            {
                feature_name:
                    reconstructed_row[
                        feature_name
                    ]
                for feature_name
                in section3barg_raw_feature_columns
            }
        ]
    )


    encoded_matrix = (
        baseline_preprocessor.transform(
            raw_feature_df
        )
    )


    encoded_array = (
        encoded_matrix.toarray()
        if hasattr(
            encoded_matrix,
            "toarray",
        )
        else np.asarray(
            encoded_matrix
        )
    )


    encoded_record = {
        "comparison_case_id":
            comparison_case_id,

        "source_scenario_id":
            reconstructed_row[
                "source_scenario_id"
            ],

        "evaluation_side":
            reconstructed_row[
                "evaluation_side"
            ],
    }


    for encoded_index, encoded_name in enumerate(
        baseline_encoded_feature_names
    ):

        encoded_record[
            str(
                encoded_name
            )
        ] = float(
            encoded_array[
                0,
                encoded_index,
            ]
        )


    section3barg_encoded_rows.append(
        encoded_record
    )


section3barg_reconstructed_encoded_features_df = pd.DataFrame(
    section3barg_encoded_rows
)


assert len(
    section3barg_reconstructed_encoded_features_df
) == 184


# --------------------------------------------------------------------------------------
# 11. Identify reconstructed raw values most associated with probability drift
# --------------------------------------------------------------------------------------

section3barg_drift_df = (
    section3bard_probability_validation_df[
        [
            "comparison_case_id",
            "quick_attack_absolute_difference",
            "ascension_absolute_difference",
        ]
    ]
    .merge(
        section3barg_reconstructed_raw_features_df,
        on="comparison_case_id",
        how="left",
    )
)


section3barg_numeric_association_rows = []


for feature_name in section3barg_raw_feature_columns:

    numeric_feature_values = pd.to_numeric(
        section3barg_drift_df[
            feature_name
        ],
        errors="coerce",
    )


    if numeric_feature_values.notna().sum() < 3:

        continue


    quick_correlation = (
        numeric_feature_values.corr(
            section3barg_drift_df[
                "quick_attack_absolute_difference"
            ]
        )
    )


    ascension_correlation = (
        numeric_feature_values.corr(
            section3barg_drift_df[
                "ascension_absolute_difference"
            ]
        )
    )


    section3barg_numeric_association_rows.append(
        {
            "feature_name":
                feature_name,

            "nonmissing_numeric_rows":
                int(
                    numeric_feature_values.notna().sum()
                ),

            "unique_numeric_values":
                int(
                    numeric_feature_values.nunique(
                        dropna=True
                    )
                ),

            "quick_attack_drift_correlation":
                quick_correlation,

            "ascension_drift_correlation":
                ascension_correlation,

            "maximum_absolute_correlation":
                float(
                    np.nanmax(
                        np.abs(
                            [
                                quick_correlation,
                                ascension_correlation,
                            ]
                        )
                    )
                )
                if not (
                    pd.isna(
                        quick_correlation
                    )
                    and
                    pd.isna(
                        ascension_correlation
                    )
                )
                else np.nan,
        }
    )


section3barg_numeric_drift_association_df = (
    pd.DataFrame(
        section3barg_numeric_association_rows
    )
    .sort_values(
        "maximum_absolute_correlation",
        ascending=False,
        na_position="last",
    )
    .reset_index(drop=True)
)


print()
print("RAW NUMERIC FEATURES ASSOCIATED WITH PROBABILITY DRIFT")
print("-" * 100)

display(
    section3barg_numeric_drift_association_df
    .head(25)
)


# --------------------------------------------------------------------------------------
# 12. Scenario-level drift profile
# --------------------------------------------------------------------------------------

section3barg_scenario_drift_profile_df = (
    section3barg_drift_df
    .groupby(
        "source_scenario_id",
        dropna=False,
    )
    .agg(
        cases=(
            "comparison_case_id",
            "size",
        ),

        mean_quick_attack_difference=(
            "quick_attack_absolute_difference",
            "mean",
        ),

        maximum_quick_attack_difference=(
            "quick_attack_absolute_difference",
            "max",
        ),

        mean_ascension_difference=(
            "ascension_absolute_difference",
            "mean",
        ),

        maximum_ascension_difference=(
            "ascension_absolute_difference",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "mean_quick_attack_difference",
            "mean_ascension_difference",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


print()
print("SCENARIO-LEVEL PROBABILITY DRIFT")
print("-" * 100)

display(
    section3barg_scenario_drift_profile_df
)


# --------------------------------------------------------------------------------------
# 13. Determine diagnostic route
# --------------------------------------------------------------------------------------

section3barg_direct_features_compared = int(
    len(
        section3barg_feature_equality_df
    )
)


section3barg_features_with_differences = int(
    (
        section3barg_feature_equality_df[
            "different_rows"
        ].gt(0)
    ).sum()
    if not section3barg_feature_equality_df.empty
    else 0
)


if section3barg_features_with_differences > 0:

    section3barg_status = (
        "RAW_FEATURE_VALUE_DIFFERENCES_IDENTIFIED"
    )

    section3barg_next_stage = (
        "REPAIR_IDENTIFIED_RAW_FEATURES"
    )


elif section3barg_direct_features_compared == 41:

    section3barg_status = (
        "ALL_SAVED_RAW_FEATURES_MATCH"
    )

    section3barg_next_stage = (
        "MODEL_OR_PREPROCESSOR_VERSION_DIAGNOSTIC"
    )


elif section3barg_direct_features_compared > 0:

    section3barg_status = (
        "PARTIAL_RAW_FEATURE_COMPARISON_COMPLETE"
    )

    section3barg_next_stage = (
        "RECOVER_REMAINING_NOTEBOOK54_RAW_FEATURE_EVIDENCE"
    )


else:

    section3barg_status = (
        "SAVED_REPORTS_DO_NOT_CONTAIN_RAW_FEATURE_VALUES"
    )

    section3barg_next_stage = (
        "RECOVER_NOTEBOOK54_SCORE_MATRIX_OR_MODEL_SNAPSHOT"
    )


section3barg_summary = {
    "status":
        section3barg_status,

    "cases_reconstructed":
        int(
            len(
                section3barg_reconstructed_raw_features_df
            )
        ),

    "raw_feature_count":
        int(
            len(
                section3barg_raw_feature_columns
            )
        ),

    "best_saved_source":
        section3barg_best_source_name,

    "direct_raw_features_compared":
        section3barg_direct_features_compared,

    "features_with_value_differences":
        section3barg_features_with_differences,

    "case_level_feature_differences":
        int(
            len(
                section3barg_case_differences_df
            )
        ),

    "scenario_profiles_created":
        int(
            len(
                section3barg_scenario_drift_profile_df
            )
        ),

    "next_stage":
        section3barg_next_stage,
}


print()
print("SECTION 3B-A REPAIR G SUMMARY")
print("-" * 100)

for key, value in section3barg_summary.items():

    print(
        f"{key:62}: {value}"
    )


# --------------------------------------------------------------------------------------
# 14. Validation checks
# --------------------------------------------------------------------------------------

section3barg_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_raw_rows_reconstructed",

            "passed":
                len(
                    section3barg_reconstructed_raw_features_df
                ) == 184,

            "value":
                len(
                    section3barg_reconstructed_raw_features_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "raw_schema_contains_41_features",

            "passed":
                len(
                    section3barg_raw_feature_columns
                ) == 41,

            "value":
                len(
                    section3barg_raw_feature_columns
                ),

            "expected":
                41,
        },
        {
            "check":
                "encoded_schema_contains_64_features",

            "passed":
                (
                    len(
                        section3barg_reconstructed_encoded_features_df.columns
                    )
                    -
                    3
                ) == 64,

            "value":
                (
                    len(
                        section3barg_reconstructed_encoded_features_df.columns
                    )
                    -
                    3
                ),

            "expected":
                64,
        },
        {
            "check":
                "diagnostic_route_resolved",

            "passed":
                section3barg_next_stage
                in {
                    "REPAIR_IDENTIFIED_RAW_FEATURES",
                    "MODEL_OR_PREPROCESSOR_VERSION_DIAGNOSTIC",
                    "RECOVER_REMAINING_NOTEBOOK54_RAW_FEATURE_EVIDENCE",
                    "RECOVER_NOTEBOOK54_SCORE_MATRIX_OR_MODEL_SNAPSHOT",
                },

            "value":
                section3barg_next_stage,

            "expected":
                "Recognized diagnostic route",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR G VALIDATION CHECKS")
print("-" * 100)

display(
    section3barg_validation_checks_df
)


assert section3barg_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 15. Save Repair G reports
# --------------------------------------------------------------------------------------

SECTION3BARG_RAW_FEATURES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_reconstructed_raw_features.csv"
)

SECTION3BARG_ENCODED_FEATURES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_reconstructed_encoded_features.csv"
)

SECTION3BARG_SOURCE_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_saved_raw_feature_source_profile.csv"
)

SECTION3BARG_FEATURE_EQUALITY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_raw_feature_equality.csv"
)

SECTION3BARG_CASE_DIFFERENCES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_case_raw_feature_differences.csv"
)

SECTION3BARG_NUMERIC_ASSOCIATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_raw_feature_drift_association.csv"
)

SECTION3BARG_SCENARIO_DRIFT_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_scenario_probability_drift.csv"
)

SECTION3BARG_ERRORS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_reconstruction_errors.csv"
)

SECTION3BARG_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_validation_checks.csv"
)

SECTION3BARG_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barg_raw_feature_audit_summary.json"
)


section3barg_reconstructed_raw_features_df.to_csv(
    SECTION3BARG_RAW_FEATURES_FILE,
    index=False,
)

section3barg_reconstructed_encoded_features_df.to_csv(
    SECTION3BARG_ENCODED_FEATURES_FILE,
    index=False,
)

section3barg_saved_source_profile_df.to_csv(
    SECTION3BARG_SOURCE_PROFILE_FILE,
    index=False,
)

section3barg_feature_equality_df.to_csv(
    SECTION3BARG_FEATURE_EQUALITY_FILE,
    index=False,
)

section3barg_case_differences_df.to_csv(
    SECTION3BARG_CASE_DIFFERENCES_FILE,
    index=False,
)

section3barg_numeric_drift_association_df.to_csv(
    SECTION3BARG_NUMERIC_ASSOCIATION_FILE,
    index=False,
)

section3barg_scenario_drift_profile_df.to_csv(
    SECTION3BARG_SCENARIO_DRIFT_FILE,
    index=False,
)

section3barg_reconstruction_errors_df.to_csv(
    SECTION3BARG_ERRORS_FILE,
    index=False,
)

section3barg_validation_checks_df.to_csv(
    SECTION3BARG_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARG_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barg_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3barg_saved_files = [
    SECTION3BARG_RAW_FEATURES_FILE,
    SECTION3BARG_ENCODED_FEATURES_FILE,
    SECTION3BARG_SOURCE_PROFILE_FILE,
    SECTION3BARG_FEATURE_EQUALITY_FILE,
    SECTION3BARG_CASE_DIFFERENCES_FILE,
    SECTION3BARG_NUMERIC_ASSOCIATION_FILE,
    SECTION3BARG_SCENARIO_DRIFT_FILE,
    SECTION3BARG_ERRORS_FILE,
    SECTION3BARG_VALIDATION_FILE,
    SECTION3BARG_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3barg_saved_files
)


print()
print("SAVED SECTION 3B-A REPAIR G REPORTS")
print("-" * 100)

for file_path in section3barg_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A RAW FEATURE EQUALITY AUDIT PASSED"
)


# In[27]:


# ======================================================================================
# SECTION 3B-A REPAIR H — RESTORE LEGAL-MOVE SIGNATURE CATEGORY CASING
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR H — RESTORE LEGAL-MOVE SIGNATURE CATEGORY CASING")
print("=" * 100)

from copy import deepcopy
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate required objects
# --------------------------------------------------------------------------------------

SECTION3BARH_REQUIRED_OBJECTS = [
    "build_legality_aware_feature_row",
    "build_section4wf_evaluation_state",
    "section3ba_score_cases_df",
    "section3ba_expanded_case_results_df",
    "baseline_preprocessor",
    "baseline_policy_model",
    "baseline_encoded_feature_names",
    "REPORTS_DIRECTORY",
]


section3barh_missing_objects = [
    object_name
    for object_name in SECTION3BARH_REQUIRED_OBJECTS
    if object_name not in globals()
]


assert not section3barh_missing_objects, (
    "Repair H required objects are missing: "
    f"{section3barh_missing_objects}"
)


# Preserve the original feature builder.
if (
    "section3barh_original_feature_builder"
    not in globals()
):

    section3barh_original_feature_builder = (
        build_legality_aware_feature_row
    )


# --------------------------------------------------------------------------------------
# 2. Restore Notebook 54 action-name casing
# --------------------------------------------------------------------------------------

SECTION3BARH_ACTION_CASE_MAP = {
    "ascension":
        "Ascension",

    "bind down":
        "Bind Down",

    "live coal":
        "Live Coal",

    "pass":
        "Pass",

    "quick attack":
        "Quick Attack",

    "tuck tail":
        "Tuck Tail",
}


def section3barh_restore_action_case(
    action_name,
):
    """
    Restore the categorical action spelling used by Notebook 53/54 training.
    """

    normalized_action = str(
        action_name
    ).strip().lower()

    return SECTION3BARH_ACTION_CASE_MAP.get(
        normalized_action,
        str(action_name).strip(),
    )


def section3barh_restore_signature_case(
    signature_value,
):
    """
    Convert:
        ascension | quick attack
    into:
        Ascension | Quick Attack
    """

    if signature_value is None:

        return ""


    signature_text = str(
        signature_value
    ).strip()


    if not signature_text:

        return ""


    signature_actions = [
        action_name.strip()
        for action_name in signature_text.split("|")
        if action_name.strip()
    ]


    restored_actions = [
        section3barh_restore_action_case(
            action_name
        )
        for action_name in signature_actions
    ]


    return " | ".join(
        restored_actions
    )


# --------------------------------------------------------------------------------------
# 3. Create a compatibility-safe feature builder
# --------------------------------------------------------------------------------------

def build_legality_aware_feature_row_55(
    *args,
    **kwargs,
):
    """
    Build the Notebook 53 feature row and restore exact categorical casing.
    """

    feature_df = (
        section3barh_original_feature_builder(
            *args,
            **kwargs,
        )
        .copy()
    )


    assert len(
        feature_df
    ) == 1


    assert feature_df.shape[
        1
    ] == 41


    if (
        "legal_move_signature"
        in feature_df.columns
    ):

        feature_df[
            "legal_move_signature"
        ] = (
            feature_df[
                "legal_move_signature"
            ]
            .apply(
                section3barh_restore_signature_case
            )
        )


    return feature_df


# Use the repaired builder for subsequent Notebook 55 evaluation.
build_legality_aware_feature_row = (
    build_legality_aware_feature_row_55
)


# --------------------------------------------------------------------------------------
# 4. Validate the repair on one known case
# --------------------------------------------------------------------------------------

section3barh_case_lookup_df = (
    section3ba_expanded_case_results_df
    .drop_duplicates(
        subset=[
            "comparison_case_id",
        ]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


section3barh_test_score_row = (
    section3ba_score_cases_df.iloc[0]
)


section3barh_test_case_id = str(
    section3barh_test_score_row[
        "comparison_case_id"
    ]
)


section3barh_test_source_row = (
    section3barh_case_lookup_df.loc[
        section3barh_test_case_id
    ]
)


section3barh_test_state = (
    build_section4wf_evaluation_state(
        section3barh_test_score_row
    )
)


section3barh_test_legal_moves = (
    normalize_section3barg_legal_moves(
        section3barh_test_source_row[
            SECTION3BARG_LEGAL_MOVE_COLUMN
        ]
    )
)


section3barh_test_feature_df = (
    build_legality_aware_feature_row(
        battle_state=
            section3barh_test_state,

        legal_moves=
            section3barh_test_legal_moves,

        side_mode=
            "PRESERVE",
    )
)


section3barh_test_signature = str(
    section3barh_test_feature_df.loc[
        section3barh_test_feature_df.index[0],
        "legal_move_signature",
    ]
)


print()
print("SINGLE-CASE SIGNATURE REPAIR")
print("-" * 100)

print(
    "Comparison case:",
    section3barh_test_case_id,
)

print(
    "Restored signature:",
    section3barh_test_signature,
)


assert (
    section3barh_test_signature
    ==
    "Ascension | Quick Attack"
)


# --------------------------------------------------------------------------------------
# 5. Rescore all 184 cases
# --------------------------------------------------------------------------------------

section3barh_model_classes = [
    str(
        class_name
    ).strip()
    for class_name in baseline_policy_model.classes_
]


section3barh_normalized_classes = [
    class_name.lower()
    for class_name in section3barh_model_classes
]


SECTION3BARH_QUICK_ATTACK_INDEX = (
    section3barh_normalized_classes.index(
        "quick attack"
    )
)


SECTION3BARH_ASCENSION_INDEX = (
    section3barh_normalized_classes.index(
        "ascension"
    )
)


section3barh_validation_rows = []
section3barh_error_rows = []


for _, score_row in (
    section3ba_score_cases_df.iterrows()
):

    comparison_case_id = str(
        score_row[
            "comparison_case_id"
        ]
    )


    try:

        source_row = (
            section3barh_case_lookup_df.loc[
                comparison_case_id
            ]
        )


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = (
            normalize_section3barg_legal_moves(
                source_row[
                    SECTION3BARG_LEGAL_MOVE_COLUMN
                ]
            )
        )


        raw_feature_df = (
            build_legality_aware_feature_row(
                battle_state=
                    evaluation_state,

                legal_moves=
                    legal_moves,

                side_mode=
                    "PRESERVE",
            )
        )


        transformed_features = (
            baseline_preprocessor.transform(
                raw_feature_df
            )
        )


        probability_vector = np.asarray(
            baseline_policy_model.predict_proba(
                transformed_features
            )
        )[0]


        reconstructed_quick_attack_probability = float(
            probability_vector[
                SECTION3BARH_QUICK_ATTACK_INDEX
            ]
        )


        reconstructed_ascension_probability = float(
            probability_vector[
                SECTION3BARH_ASCENSION_INDEX
            ]
        )


        saved_quick_attack_probability = float(
            score_row[
                "quick_attack_probability"
            ]
        )


        saved_ascension_probability = float(
            score_row[
                "ascension_probability"
            ]
        )


        reconstructed_prediction = str(
            section3barh_model_classes[
                int(
                    np.argmax(
                        probability_vector
                    )
                )
            ]
        )


        saved_prediction = str(
            score_row[
                "predicted_action"
            ]
        )


        section3barh_validation_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "source_scenario_id":
                    str(
                        score_row[
                            "source_scenario_id"
                        ]
                    ),

                "evaluation_side":
                    str(
                        score_row[
                            "evaluation_side"
                        ]
                    ),

                "legal_move_signature":
                    str(
                        raw_feature_df.iloc[0][
                            "legal_move_signature"
                        ]
                    ),

                "reconstructed_quick_attack_probability":
                    reconstructed_quick_attack_probability,

                "saved_quick_attack_probability":
                    saved_quick_attack_probability,

                "quick_attack_absolute_difference":
                    abs(
                        reconstructed_quick_attack_probability
                        -
                        saved_quick_attack_probability
                    ),

                "reconstructed_ascension_probability":
                    reconstructed_ascension_probability,

                "saved_ascension_probability":
                    saved_ascension_probability,

                "ascension_absolute_difference":
                    abs(
                        reconstructed_ascension_probability
                        -
                        saved_ascension_probability
                    ),

                "reconstructed_prediction":
                    reconstructed_prediction,

                "saved_prediction":
                    saved_prediction,

                "prediction_agreement":
                    (
                        reconstructed_prediction.lower()
                        ==
                        saved_prediction.lower()
                    ),
            }
        )


    except Exception as error:

        section3barh_error_rows.append(
            {
                "comparison_case_id":
                    comparison_case_id,

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


section3barh_probability_validation_df = pd.DataFrame(
    section3barh_validation_rows
)


section3barh_errors_df = pd.DataFrame(
    section3barh_error_rows,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


assert section3barh_errors_df.empty, (
    "One or more repaired cases failed: "
    f"{section3barh_error_rows[:5]}"
)


assert len(
    section3barh_probability_validation_df
) == 184


# --------------------------------------------------------------------------------------
# 6. Calculate post-repair agreement
# --------------------------------------------------------------------------------------

section3barh_mean_quick_attack_difference = float(
    section3barh_probability_validation_df[
        "quick_attack_absolute_difference"
    ].mean()
)


section3barh_max_quick_attack_difference = float(
    section3barh_probability_validation_df[
        "quick_attack_absolute_difference"
    ].max()
)


section3barh_mean_ascension_difference = float(
    section3barh_probability_validation_df[
        "ascension_absolute_difference"
    ].mean()
)


section3barh_max_ascension_difference = float(
    section3barh_probability_validation_df[
        "ascension_absolute_difference"
    ].max()
)


section3barh_prediction_agreement_rate = float(
    section3barh_probability_validation_df[
        "prediction_agreement"
    ].astype(bool).mean()
)


SECTION3BARH_EXACT_TOLERANCE = 1e-10


section3barh_exact_probability_agreement = bool(
    section3barh_max_quick_attack_difference
    <=
    SECTION3BARH_EXACT_TOLERANCE
    and
    section3barh_max_ascension_difference
    <=
    SECTION3BARH_EXACT_TOLERANCE
)


print()
print("POST-REPAIR BASELINE AGREEMENT")
print("-" * 100)

print(
    "Mean Quick Attack difference:",
    section3barh_mean_quick_attack_difference,
)

print(
    "Maximum Quick Attack difference:",
    section3barh_max_quick_attack_difference,
)

print(
    "Mean Ascension difference:",
    section3barh_mean_ascension_difference,
)

print(
    "Maximum Ascension difference:",
    section3barh_max_ascension_difference,
)

print(
    "Prediction agreement rate:",
    section3barh_prediction_agreement_rate,
)

print(
    "Exact probability agreement:",
    section3barh_exact_probability_agreement,
)


print()
print("LARGEST POST-REPAIR DIFFERENCES")
print("-" * 100)

display(
    section3barh_probability_validation_df
    .sort_values(
        [
            "quick_attack_absolute_difference",
            "ascension_absolute_difference",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
    .reset_index(drop=True)
)


# --------------------------------------------------------------------------------------
# 7. Determine next stage
# --------------------------------------------------------------------------------------

if (
    section3barh_exact_probability_agreement
    and
    section3barh_prediction_agreement_rate == 1.0
):

    section3barh_status = (
        "LEGAL_SIGNATURE_CATEGORY_REPAIR_CONFIRMED"
    )

    section3barh_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )


elif section3barh_prediction_agreement_rate == 1.0:

    section3barh_status = (
        "LEGAL_SIGNATURE_REPAIRED_WITH_RESIDUAL_DRIFT"
    )

    section3barh_next_stage = (
        "AUDIT_REMAINING_CATEGORICAL_FEATURE_VALUES"
    )


else:

    section3barh_status = (
        "LEGAL_SIGNATURE_REPAIR_INCOMPLETE"
    )

    section3barh_next_stage = (
        "FEATURE_RECONSTRUCTION_REPAIR_CONTINUES"
    )


section3barh_summary = {
    "status":
        section3barh_status,

    "cases_validated":
        int(
            len(
                section3barh_probability_validation_df
            )
        ),

    "signature_value":
        section3barh_test_signature,

    "mean_quick_attack_probability_difference":
        section3barh_mean_quick_attack_difference,

    "maximum_quick_attack_probability_difference":
        section3barh_max_quick_attack_difference,

    "mean_ascension_probability_difference":
        section3barh_mean_ascension_difference,

    "maximum_ascension_probability_difference":
        section3barh_max_ascension_difference,

    "prediction_agreement_rate":
        section3barh_prediction_agreement_rate,

    "exact_probability_agreement":
        section3barh_exact_probability_agreement,

    "next_stage":
        section3barh_next_stage,
}


print()
print("SECTION 3B-A REPAIR H SUMMARY")
print("-" * 100)

for key, value in (
    section3barh_summary.items()
):

    print(
        f"{key:66}: {value}"
    )


# --------------------------------------------------------------------------------------
# 8. Validation checks
# --------------------------------------------------------------------------------------

section3barh_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_cases_validated",

            "passed":
                len(
                    section3barh_probability_validation_df
                ) == 184,

            "value":
                len(
                    section3barh_probability_validation_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "signature_casing_restored",

            "passed":
                section3barh_probability_validation_df[
                    "legal_move_signature"
                ].eq(
                    "Ascension | Quick Attack"
                ).all(),

            "value":
                section3barh_probability_validation_df[
                    "legal_move_signature"
                ].unique().tolist(),

            "expected":
                [
                    "Ascension | Quick Attack"
                ],
        },
        {
            "check":
                "prediction_agreement_complete",

            "passed":
                section3barh_prediction_agreement_rate
                == 1.0,

            "value":
                section3barh_prediction_agreement_rate,

            "expected":
                1.0,
        },
        {
            "check":
                "evaluation_route_resolved",

            "passed":
                section3barh_next_stage
                in {
                    "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
                    "AUDIT_REMAINING_CATEGORICAL_FEATURE_VALUES",
                    "FEATURE_RECONSTRUCTION_REPAIR_CONTINUES",
                },

            "value":
                section3barh_next_stage,

            "expected":
                "Recognized route",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR H VALIDATION CHECKS")
print("-" * 100)

display(
    section3barh_validation_checks_df
)


assert section3barh_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 9. Save Repair H reports
# --------------------------------------------------------------------------------------

SECTION3BARH_PROBABILITY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barh_probability_validation.csv"
)

SECTION3BARH_ERRORS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barh_errors.csv"
)

SECTION3BARH_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barh_validation_checks.csv"
)

SECTION3BARH_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barh_legal_signature_repair_summary.json"
)


section3barh_probability_validation_df.to_csv(
    SECTION3BARH_PROBABILITY_FILE,
    index=False,
)

section3barh_errors_df.to_csv(
    SECTION3BARH_ERRORS_FILE,
    index=False,
)

section3barh_validation_checks_df.to_csv(
    SECTION3BARH_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARH_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barh_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR H REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARH_PROBABILITY_FILE,
    SECTION3BARH_ERRORS_FILE,
    SECTION3BARH_VALIDATION_FILE,
    SECTION3BARH_SUMMARY_FILE,
]:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A LEGAL-MOVE SIGNATURE "
    "CATEGORY CASING RESTORED"
)


# In[28]:


# ======================================================================================
# SECTION 3B-A REPAIR H — FINAL DECISION
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR H — FINAL DECISION")
print("=" * 100)

for key in [
    "status",
    "cases_validated",
    "signature_value",
    "mean_quick_attack_probability_difference",
    "maximum_quick_attack_probability_difference",
    "mean_ascension_probability_difference",
    "maximum_ascension_probability_difference",
    "prediction_agreement_rate",
    "exact_probability_agreement",
    "next_stage",
]:
    print(
        f"{key:66}: "
        f"{section3barh_summary.get(key)}"
    )


assert section3barh_summary[
    "cases_validated"
] == 184


if (
    section3barh_summary[
        "next_stage"
    ]
    ==
    "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
):

    print()
    print(
        "✅ EXACT BASELINE RECONSTRUCTION CONFIRMED — "
        "READY FOR SECTION 3B-B"
    )

else:

    print()
    print(
        "⚠️ RESIDUAL PROBABILITY DRIFT REMAINS — "
        f"NEXT: {section3barh_summary['next_stage']}"
    )


# In[29]:


# ======================================================================================
# SECTION 3B-A REPAIR I — CATEGORICAL ENCODING COMPATIBILITY AUDIT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR I — CATEGORICAL ENCODING COMPATIBILITY AUDIT")
print("=" * 100)

from copy import deepcopy
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate required objects
# --------------------------------------------------------------------------------------

SECTION3BARI_REQUIRED_OBJECTS = [
    "baseline_preprocessor",
    "baseline_policy_model",
    "baseline_encoded_feature_names",
    "build_legality_aware_feature_row",
    "build_section4wf_evaluation_state",
    "section3ba_score_cases_df",
    "section3ba_expanded_case_results_df",
    "section3barh_probability_validation_df",
    "section3barg_raw_feature_columns",
    "normalize_section3barg_legal_moves",
    "SECTION3BARG_LEGAL_MOVE_COLUMN",
    "REPORTS_DIRECTORY",
]


section3bari_missing_objects = [
    object_name
    for object_name in SECTION3BARI_REQUIRED_OBJECTS
    if object_name not in globals()
]


print()
print("SECTION 3B-A REPAIR I OBJECT VALIDATION")
print("-" * 100)

for object_name in SECTION3BARI_REQUIRED_OBJECTS:

    print(
        f"{object_name:56}: "
        f"{object_name in globals()}"
    )


assert not section3bari_missing_objects, (
    "Repair I required objects are missing: "
    f"{section3bari_missing_objects}"
)


assert len(
    section3ba_score_cases_df
) == 184


assert len(
    baseline_encoded_feature_names
) == 64


# --------------------------------------------------------------------------------------
# 2. Preserve the current repaired feature builder
# --------------------------------------------------------------------------------------

section3bari_base_feature_builder = (
    build_legality_aware_feature_row
)


# --------------------------------------------------------------------------------------
# 3. Reconstruct current 41-feature rows for all 184 cases
# --------------------------------------------------------------------------------------

section3bari_case_lookup_df = (
    section3ba_expanded_case_results_df
    .drop_duplicates(
        subset=[
            "comparison_case_id",
        ]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


section3bari_raw_rows = []
section3bari_reconstruction_errors = []


for _, score_row in (
    section3ba_score_cases_df.iterrows()
):

    comparison_case_id = str(
        score_row[
            "comparison_case_id"
        ]
    )


    try:

        source_row = (
            section3bari_case_lookup_df.loc[
                comparison_case_id
            ]
        )


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = (
            normalize_section3barg_legal_moves(
                source_row[
                    SECTION3BARG_LEGAL_MOVE_COLUMN
                ]
            )
        )


        raw_feature_df = (
            section3bari_base_feature_builder(
                battle_state=
                    evaluation_state,

                legal_moves=
                    legal_moves,

                side_mode=
                    "PRESERVE",
            )
        )


        assert raw_feature_df.shape == (
            1,
            41,
        )


        raw_record = (
            raw_feature_df.iloc[0]
            .to_dict()
        )


        raw_record.update(
            {
                "comparison_case_id":
                    comparison_case_id,

                "source_scenario_id":
                    str(
                        score_row[
                            "source_scenario_id"
                        ]
                    ),

                "evaluation_side":
                    str(
                        score_row[
                            "evaluation_side"
                        ]
                    ),

                "expanded_pair_id":
                    str(
                        score_row[
                            "expanded_pair_id"
                        ]
                    ),
            }
        )


        section3bari_raw_rows.append(
            raw_record
        )


    except Exception as error:

        section3bari_reconstruction_errors.append(
            {
                "comparison_case_id":
                    comparison_case_id,

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


section3bari_raw_features_df = pd.DataFrame(
    section3bari_raw_rows
)


section3bari_reconstruction_errors_df = pd.DataFrame(
    section3bari_reconstruction_errors,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


assert section3bari_reconstruction_errors_df.empty, (
    "One or more categorical-audit rows failed reconstruction."
)


assert len(
    section3bari_raw_features_df
) == 184


# --------------------------------------------------------------------------------------
# 4. Discover categorical transformer and learned categories
# --------------------------------------------------------------------------------------

section3bari_transformer_rows = []


for transformer_name, transformer_object, transformer_columns in (
    baseline_preprocessor.transformers_
):

    section3bari_transformer_rows.append(
        {
            "transformer_name":
                transformer_name,

            "transformer_type":
                type(
                    transformer_object
                ).__name__,

            "columns":
                (
                    list(
                        transformer_columns
                    )
                    if isinstance(
                        transformer_columns,
                        (
                            list,
                            tuple,
                            np.ndarray,
                            pd.Index,
                        ),
                    )
                    else transformer_columns
                ),
        }
    )


section3bari_transformer_profile_df = pd.DataFrame(
    section3bari_transformer_rows
)


print()
print("PREPROCESSOR TRANSFORMER PROFILE")
print("-" * 100)

display(
    section3bari_transformer_profile_df
)


section3bari_categorical_transformer = None
section3bari_categorical_columns = None
section3bari_one_hot_encoder = None


for transformer_name, transformer_object, transformer_columns in (
    baseline_preprocessor.transformers_
):

    candidate_encoder = None


    if hasattr(
        transformer_object,
        "categories_",
    ):

        candidate_encoder = transformer_object


    elif hasattr(
        transformer_object,
        "named_steps",
    ):

        for step_name, step_object in (
            transformer_object.named_steps.items()
        ):

            if hasattr(
                step_object,
                "categories_",
            ):

                candidate_encoder = step_object
                break


    if candidate_encoder is not None:

        section3bari_categorical_transformer = (
            transformer_object
        )

        section3bari_one_hot_encoder = (
            candidate_encoder
        )

        section3bari_categorical_columns = list(
            transformer_columns
        )

        break


assert section3bari_one_hot_encoder is not None, (
    "Unable to locate the fitted categorical encoder."
)


assert section3bari_categorical_columns is not None


assert len(
    section3bari_categorical_columns
) == len(
    section3bari_one_hot_encoder.categories_
)


# --------------------------------------------------------------------------------------
# 5. Build learned-category inventory
# --------------------------------------------------------------------------------------

section3bari_category_inventory_rows = []


SECTION3BARI_LEARNED_CATEGORIES = {}


for column_name, learned_categories in zip(
    section3bari_categorical_columns,
    section3bari_one_hot_encoder.categories_,
):

    learned_category_values = [
        str(
            category_value
        )
        for category_value in learned_categories
    ]


    SECTION3BARI_LEARNED_CATEGORIES[
        str(
            column_name
        )
    ] = learned_category_values


    for category_index, category_value in enumerate(
        learned_category_values
    ):

        section3bari_category_inventory_rows.append(
            {
                "raw_feature":
                    str(
                        column_name
                    ),

                "category_index":
                    int(
                        category_index
                    ),

                "learned_category":
                    category_value,
            }
        )


section3bari_learned_category_inventory_df = pd.DataFrame(
    section3bari_category_inventory_rows
)


print()
print("FITTED CATEGORICAL ENCODER INVENTORY")
print("-" * 100)

display(
    section3bari_learned_category_inventory_df
)


# --------------------------------------------------------------------------------------
# 6. Compare current raw values against learned categories
# --------------------------------------------------------------------------------------

section3bari_category_compatibility_rows = []
section3bari_unknown_value_rows = []


for column_name in section3bari_categorical_columns:

    learned_categories = set(
        SECTION3BARI_LEARNED_CATEGORIES[
            str(
                column_name
            )
        ]
    )


    current_values = (
        section3bari_raw_features_df[
            column_name
        ]
        .fillna(
            "<NA>"
        )
        .astype(str)
    )


    unique_current_values = sorted(
        current_values.unique().tolist()
    )


    unknown_values = sorted(
        set(
            unique_current_values
        )
        -
        learned_categories
    )


    known_rows = int(
        current_values.isin(
            learned_categories
        ).sum()
    )


    unknown_rows = int(
        (
            ~current_values.isin(
                learned_categories
            )
        ).sum()
    )


    section3bari_category_compatibility_rows.append(
        {
            "raw_feature":
                str(
                    column_name
                ),

            "learned_category_count":
                int(
                    len(
                        learned_categories
                    )
                ),

            "current_unique_value_count":
                int(
                    len(
                        unique_current_values
                    )
                ),

            "known_rows":
                known_rows,

            "unknown_rows":
                unknown_rows,

            "compatibility_rate":
                float(
                    known_rows
                    /
                    len(
                        current_values
                    )
                ),

            "current_values":
                json.dumps(
                    unique_current_values
                ),

            "unknown_values":
                json.dumps(
                    unknown_values
                ),

            "learned_categories":
                json.dumps(
                    sorted(
                        learned_categories
                    )
                ),
        }
    )


    if unknown_values:

        for unknown_value in unknown_values:

            affected_rows = (
                section3bari_raw_features_df.loc[
                    current_values.eq(
                        unknown_value
                    )
                ]
            )


            section3bari_unknown_value_rows.append(
                {
                    "raw_feature":
                        str(
                            column_name
                        ),

                    "unknown_value":
                        unknown_value,

                    "affected_rows":
                        int(
                            len(
                                affected_rows
                            )
                        ),

                    "first_comparison_case_id":
                        (
                            str(
                                affected_rows[
                                    "comparison_case_id"
                                ].iloc[0]
                            )
                            if not affected_rows.empty
                            else ""
                        ),

                    "learned_categories":
                        json.dumps(
                            sorted(
                                learned_categories
                            )
                        ),
                }
            )


section3bari_category_compatibility_df = pd.DataFrame(
    section3bari_category_compatibility_rows
)


section3bari_unknown_values_df = pd.DataFrame(
    section3bari_unknown_value_rows,
    columns=[
        "raw_feature",
        "unknown_value",
        "affected_rows",
        "first_comparison_case_id",
        "learned_categories",
    ],
)


print()
print("CATEGORICAL VALUE COMPATIBILITY")
print("-" * 100)

display(
    section3bari_category_compatibility_df
    .sort_values(
        [
            "compatibility_rate",
            "unknown_rows",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)


print()
print("UNKNOWN CATEGORICAL VALUES")
print("-" * 100)

if section3bari_unknown_values_df.empty:

    print(
        "No unseen categorical values were detected."
    )

else:

    display(
        section3bari_unknown_values_df
    )


# --------------------------------------------------------------------------------------
# 7. Build safe category-normalization candidates
# --------------------------------------------------------------------------------------

def section3bari_normalized_text(
    value,
):
    return (
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )


def section3bari_find_matching_category(
    current_value,
    learned_categories,
):
    """
    Resolve a current category to a learned category using conservative rules.
    """

    current_text = str(
        current_value
    )


    # Exact match.
    if current_text in learned_categories:

        return current_text


    # Case-insensitive exact match.
    case_matches = [
        learned_value
        for learned_value in learned_categories
        if learned_value.lower()
        ==
        current_text.lower()
    ]


    if len(
        case_matches
    ) == 1:

        return case_matches[0]


    # Normalized spacing, underscores, and hyphens.
    normalized_current = (
        section3bari_normalized_text(
            current_text
        )
    )


    normalized_matches = [
        learned_value
        for learned_value in learned_categories
        if section3bari_normalized_text(
            learned_value
        )
        ==
        normalized_current
    ]


    if len(
        normalized_matches
    ) == 1:

        return normalized_matches[0]


    return None


section3bari_candidate_mapping_rows = []


SECTION3BARI_CATEGORY_REPAIR_MAP = {}


for _, unknown_row in (
    section3bari_unknown_values_df.iterrows()
):

    raw_feature = str(
        unknown_row[
            "raw_feature"
        ]
    )


    unknown_value = str(
        unknown_row[
            "unknown_value"
        ]
    )


    learned_categories = (
        SECTION3BARI_LEARNED_CATEGORIES[
            raw_feature
        ]
    )


    matched_category = (
        section3bari_find_matching_category(
            unknown_value,
            learned_categories,
        )
    )


    mapping_available = (
        matched_category is not None
    )


    section3bari_candidate_mapping_rows.append(
        {
            "raw_feature":
                raw_feature,

            "current_value":
                unknown_value,

            "proposed_learned_value":
                matched_category,

            "mapping_available":
                mapping_available,

            "affected_rows":
                int(
                    unknown_row[
                        "affected_rows"
                    ]
                ),
        }
    )


    if mapping_available:

        if raw_feature not in (
            SECTION3BARI_CATEGORY_REPAIR_MAP
        ):

            SECTION3BARI_CATEGORY_REPAIR_MAP[
                raw_feature
            ] = {}


        SECTION3BARI_CATEGORY_REPAIR_MAP[
            raw_feature
        ][
            unknown_value
        ] = matched_category


section3bari_candidate_mapping_df = pd.DataFrame(
    section3bari_candidate_mapping_rows,
    columns=[
        "raw_feature",
        "current_value",
        "proposed_learned_value",
        "mapping_available",
        "affected_rows",
    ],
)


print()
print("SAFE CATEGORY REPAIR CANDIDATES")
print("-" * 100)

if section3bari_candidate_mapping_df.empty:

    print(
        "No category repairs are required."
    )

else:

    display(
        section3bari_candidate_mapping_df
    )


# --------------------------------------------------------------------------------------
# 8. Define probability-evaluation helper
# --------------------------------------------------------------------------------------

section3bari_model_classes = [
    str(
        class_name
    ).strip()
    for class_name in baseline_policy_model.classes_
]


section3bari_normalized_classes = [
    class_name.lower()
    for class_name in section3bari_model_classes
]


SECTION3BARI_QUICK_ATTACK_INDEX = (
    section3bari_normalized_classes.index(
        "quick attack"
    )
)


SECTION3BARI_ASCENSION_INDEX = (
    section3bari_normalized_classes.index(
        "ascension"
    )
)


def section3bari_apply_category_map(
    raw_feature_df,
    category_map,
):
    repaired_df = (
        raw_feature_df.copy()
    )


    for feature_name, value_map in (
        category_map.items()
    ):

        if feature_name not in (
            repaired_df.columns
        ):

            continue


        repaired_df[
            feature_name
        ] = (
            repaired_df[
                feature_name
            ]
            .astype(str)
            .replace(
                value_map
            )
        )


    return repaired_df


def section3bari_evaluate_category_map(
    category_map,
):
    evaluation_rows = []
    evaluation_errors = []


    for _, score_row in (
        section3ba_score_cases_df.iterrows()
    ):

        comparison_case_id = str(
            score_row[
                "comparison_case_id"
            ]
        )


        try:

            source_row = (
                section3bari_case_lookup_df.loc[
                    comparison_case_id
                ]
            )


            evaluation_state = (
                build_section4wf_evaluation_state(
                    score_row
                )
            )


            legal_moves = (
                normalize_section3barg_legal_moves(
                    source_row[
                        SECTION3BARG_LEGAL_MOVE_COLUMN
                    ]
                )
            )


            raw_feature_df = (
                section3bari_base_feature_builder(
                    battle_state=
                        evaluation_state,

                    legal_moves=
                        legal_moves,

                    side_mode=
                        "PRESERVE",
                )
            )


            repaired_feature_df = (
                section3bari_apply_category_map(
                    raw_feature_df,
                    category_map,
                )
            )


            encoded_matrix = (
                baseline_preprocessor.transform(
                    repaired_feature_df
                )
            )


            probability_vector = np.asarray(
                baseline_policy_model.predict_proba(
                    encoded_matrix
                )
            )[0]


            reconstructed_quick_attack = float(
                probability_vector[
                    SECTION3BARI_QUICK_ATTACK_INDEX
                ]
            )


            reconstructed_ascension = float(
                probability_vector[
                    SECTION3BARI_ASCENSION_INDEX
                ]
            )


            saved_quick_attack = float(
                score_row[
                    "quick_attack_probability"
                ]
            )


            saved_ascension = float(
                score_row[
                    "ascension_probability"
                ]
            )


            reconstructed_prediction = str(
                section3bari_model_classes[
                    int(
                        np.argmax(
                            probability_vector
                        )
                    )
                ]
            )


            saved_prediction = str(
                score_row[
                    "predicted_action"
                ]
            )


            evaluation_rows.append(
                {
                    "comparison_case_id":
                        comparison_case_id,

                    "quick_attack_absolute_difference":
                        abs(
                            reconstructed_quick_attack
                            -
                            saved_quick_attack
                        ),

                    "ascension_absolute_difference":
                        abs(
                            reconstructed_ascension
                            -
                            saved_ascension
                        ),

                    "prediction_agreement":
                        (
                            reconstructed_prediction.lower()
                            ==
                            saved_prediction.lower()
                        ),
                }
            )


        except Exception as error:

            evaluation_errors.append(
                {
                    "comparison_case_id":
                        comparison_case_id,

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


    evaluation_df = pd.DataFrame(
        evaluation_rows
    )


    errors_df = pd.DataFrame(
        evaluation_errors,
        columns=[
            "comparison_case_id",
            "error_type",
            "error_message",
        ],
    )


    if not errors_df.empty:

        return {
            "valid":
                False,

            "errors":
                errors_df,

            "evaluation":
                evaluation_df,
        }


    return {
        "valid":
            True,

        "errors":
            errors_df,

        "evaluation":
            evaluation_df,

        "mean_quick_attack_difference":
            float(
                evaluation_df[
                    "quick_attack_absolute_difference"
                ].mean()
            ),

        "maximum_quick_attack_difference":
            float(
                evaluation_df[
                    "quick_attack_absolute_difference"
                ].max()
            ),

        "mean_ascension_difference":
            float(
                evaluation_df[
                    "ascension_absolute_difference"
                ].mean()
            ),

        "maximum_ascension_difference":
            float(
                evaluation_df[
                    "ascension_absolute_difference"
                ].max()
            ),

        "prediction_agreement_rate":
            float(
                evaluation_df[
                    "prediction_agreement"
                ].astype(bool).mean()
            ),
    }


# --------------------------------------------------------------------------------------
# 9. Evaluate each proposed feature repair independently
# --------------------------------------------------------------------------------------

section3bari_single_feature_test_rows = []


for raw_feature, value_map in (
    SECTION3BARI_CATEGORY_REPAIR_MAP.items()
):

    test_result = (
        section3bari_evaluate_category_map(
            {
                raw_feature:
                    value_map,
            }
        )
    )


    section3bari_single_feature_test_rows.append(
        {
            "raw_feature":
                raw_feature,

            "repair_map":
                json.dumps(
                    value_map
                ),

            "evaluation_valid":
                bool(
                    test_result[
                        "valid"
                    ]
                ),

            "mean_quick_attack_difference":
                (
                    test_result.get(
                        "mean_quick_attack_difference"
                    )
                ),

            "maximum_quick_attack_difference":
                (
                    test_result.get(
                        "maximum_quick_attack_difference"
                    )
                ),

            "mean_ascension_difference":
                (
                    test_result.get(
                        "mean_ascension_difference"
                    )
                ),

            "maximum_ascension_difference":
                (
                    test_result.get(
                        "maximum_ascension_difference"
                    )
                ),

            "prediction_agreement_rate":
                (
                    test_result.get(
                        "prediction_agreement_rate"
                    )
                ),
        }
    )


section3bari_single_feature_tests_df = pd.DataFrame(
    section3bari_single_feature_test_rows,
    columns=[
        "raw_feature",
        "repair_map",
        "evaluation_valid",
        "mean_quick_attack_difference",
        "maximum_quick_attack_difference",
        "mean_ascension_difference",
        "maximum_ascension_difference",
        "prediction_agreement_rate",
    ],
)


print()
print("SINGLE-CATEGORICAL-FEATURE REPAIR TESTS")
print("-" * 100)

if section3bari_single_feature_tests_df.empty:

    print(
        "No additional categorical repairs were proposed."
    )

else:

    display(
        section3bari_single_feature_tests_df
        .sort_values(
            [
                "mean_quick_attack_difference",
                "mean_ascension_difference",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# 10. Evaluate all safe repairs together
# --------------------------------------------------------------------------------------

section3bari_combined_result = (
    section3bari_evaluate_category_map(
        SECTION3BARI_CATEGORY_REPAIR_MAP
    )
)


assert section3bari_combined_result[
    "valid"
], (
    "The combined categorical repair evaluation failed."
)


section3bari_combined_evaluation_df = (
    section3bari_combined_result[
        "evaluation"
    ]
)


SECTION3BARI_EXACT_TOLERANCE = 1e-10


section3bari_combined_exact_agreement = bool(
    section3bari_combined_result[
        "maximum_quick_attack_difference"
    ]
    <=
    SECTION3BARI_EXACT_TOLERANCE
    and
    section3bari_combined_result[
        "maximum_ascension_difference"
    ]
    <=
    SECTION3BARI_EXACT_TOLERANCE
)


print()
print("COMBINED CATEGORICAL REPAIR RESULT")
print("-" * 100)

for metric_name in [
    "mean_quick_attack_difference",
    "maximum_quick_attack_difference",
    "mean_ascension_difference",
    "maximum_ascension_difference",
    "prediction_agreement_rate",
]:

    print(
        f"{metric_name:52}: "
        f"{section3bari_combined_result.get(metric_name)}"
    )


print(
    f"{'exact_probability_agreement':52}: "
    f"{section3bari_combined_exact_agreement}"
)


# --------------------------------------------------------------------------------------
# 11. Install only verified safe category repair map
# --------------------------------------------------------------------------------------

def build_legality_aware_feature_row_55_final(
    *args,
    **kwargs,
):
    feature_df = (
        section3bari_base_feature_builder(
            *args,
            **kwargs,
        )
    )


    repaired_df = (
        section3bari_apply_category_map(
            feature_df,
            SECTION3BARI_CATEGORY_REPAIR_MAP,
        )
    )


    return repaired_df


build_legality_aware_feature_row = (
    build_legality_aware_feature_row_55_final
)


# --------------------------------------------------------------------------------------
# 12. Determine outcome and next stage
# --------------------------------------------------------------------------------------

section3bari_unknown_feature_count = int(
    section3bari_category_compatibility_df[
        "unknown_rows"
    ].gt(0).sum()
)


section3bari_mappable_unknown_count = int(
    section3bari_candidate_mapping_df[
        "mapping_available"
    ].astype(bool).sum()
    if not section3bari_candidate_mapping_df.empty
    else 0
)


if section3bari_combined_exact_agreement:

    section3bari_status = (
        "ALL_CATEGORICAL_ENCODING_MISMATCHES_REPAIRED"
    )

    section3bari_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )


elif (
    section3bari_combined_result[
        "prediction_agreement_rate"
    ]
    ==
    1.0
):

    section3bari_status = (
        "CATEGORICAL_ENCODING_AUDIT_COMPLETE_WITH_RESIDUAL_DRIFT"
    )

    section3bari_next_stage = (
        "AUDIT_MODEL_AND_PREPROCESSOR_SNAPSHOT_IDENTITY"
    )


else:

    section3bari_status = (
        "CATEGORICAL_ENCODING_REPAIR_INCOMPLETE"
    )

    section3bari_next_stage = (
        "CONTINUE_FEATURE_VALUE_RECONSTRUCTION"
    )


section3bari_summary = {
    "status":
        section3bari_status,

    "categorical_feature_count":
        int(
            len(
                section3bari_categorical_columns
            )
        ),

    "categorical_features_with_unknown_values":
        section3bari_unknown_feature_count,

    "unknown_category_mappings_available":
        section3bari_mappable_unknown_count,

    "applied_category_repair_map":
        SECTION3BARI_CATEGORY_REPAIR_MAP,

    "mean_quick_attack_probability_difference":
        section3bari_combined_result[
            "mean_quick_attack_difference"
        ],

    "maximum_quick_attack_probability_difference":
        section3bari_combined_result[
            "maximum_quick_attack_difference"
        ],

    "mean_ascension_probability_difference":
        section3bari_combined_result[
            "mean_ascension_difference"
        ],

    "maximum_ascension_probability_difference":
        section3bari_combined_result[
            "maximum_ascension_difference"
        ],

    "prediction_agreement_rate":
        section3bari_combined_result[
            "prediction_agreement_rate"
        ],

    "exact_probability_agreement":
        section3bari_combined_exact_agreement,

    "next_stage":
        section3bari_next_stage,
}


print()
print("SECTION 3B-A REPAIR I SUMMARY")
print("-" * 100)

for key, value in (
    section3bari_summary.items()
):

    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 13. Validation checks
# --------------------------------------------------------------------------------------

section3bari_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_rows_audited",

            "passed":
                len(
                    section3bari_raw_features_df
                ) == 184,

            "value":
                len(
                    section3bari_raw_features_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "categorical_encoder_recovered",

            "passed":
                section3bari_one_hot_encoder
                is not None,

            "value":
                type(
                    section3bari_one_hot_encoder
                ).__name__,

            "expected":
                "Fitted categorical encoder",
        },
        {
            "check":
                "category_compatibility_profile_created",

            "passed":
                len(
                    section3bari_category_compatibility_df
                )
                ==
                len(
                    section3bari_categorical_columns
                ),

            "value":
                len(
                    section3bari_category_compatibility_df
                ),

            "expected":
                len(
                    section3bari_categorical_columns
                ),
        },
        {
            "check":
                "prediction_agreement_preserved",

            "passed":
                section3bari_combined_result[
                    "prediction_agreement_rate"
                ]
                == 1.0,

            "value":
                section3bari_combined_result[
                    "prediction_agreement_rate"
                ],

            "expected":
                1.0,
        },
        {
            "check":
                "next_stage_resolved",

            "passed":
                section3bari_next_stage
                in {
                    "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
                    "AUDIT_MODEL_AND_PREPROCESSOR_SNAPSHOT_IDENTITY",
                    "CONTINUE_FEATURE_VALUE_RECONSTRUCTION",
                },

            "value":
                section3bari_next_stage,

            "expected":
                "Recognized route",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR I VALIDATION CHECKS")
print("-" * 100)

display(
    section3bari_validation_checks_df
)


assert section3bari_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 14. Save Repair I reports
# --------------------------------------------------------------------------------------

SECTION3BARI_TRANSFORMER_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_transformer_profile.csv"
)

SECTION3BARI_CATEGORY_INVENTORY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_learned_category_inventory.csv"
)

SECTION3BARI_COMPATIBILITY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_category_compatibility.csv"
)

SECTION3BARI_UNKNOWN_VALUES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_unknown_categorical_values.csv"
)

SECTION3BARI_MAPPING_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_candidate_category_mappings.csv"
)

SECTION3BARI_SINGLE_TESTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_single_feature_repair_tests.csv"
)

SECTION3BARI_COMBINED_RESULT_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_combined_category_repair_results.csv"
)

SECTION3BARI_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_validation_checks.csv"
)

SECTION3BARI_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bari_categorical_encoding_audit_summary.json"
)


section3bari_transformer_profile_df.to_csv(
    SECTION3BARI_TRANSFORMER_PROFILE_FILE,
    index=False,
)

section3bari_learned_category_inventory_df.to_csv(
    SECTION3BARI_CATEGORY_INVENTORY_FILE,
    index=False,
)

section3bari_category_compatibility_df.to_csv(
    SECTION3BARI_COMPATIBILITY_FILE,
    index=False,
)

section3bari_unknown_values_df.to_csv(
    SECTION3BARI_UNKNOWN_VALUES_FILE,
    index=False,
)

section3bari_candidate_mapping_df.to_csv(
    SECTION3BARI_MAPPING_FILE,
    index=False,
)

section3bari_single_feature_tests_df.to_csv(
    SECTION3BARI_SINGLE_TESTS_FILE,
    index=False,
)

section3bari_combined_evaluation_df.to_csv(
    SECTION3BARI_COMBINED_RESULT_FILE,
    index=False,
)

section3bari_validation_checks_df.to_csv(
    SECTION3BARI_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARI_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3bari_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


section3bari_saved_files = [
    SECTION3BARI_TRANSFORMER_PROFILE_FILE,
    SECTION3BARI_CATEGORY_INVENTORY_FILE,
    SECTION3BARI_COMPATIBILITY_FILE,
    SECTION3BARI_UNKNOWN_VALUES_FILE,
    SECTION3BARI_MAPPING_FILE,
    SECTION3BARI_SINGLE_TESTS_FILE,
    SECTION3BARI_COMBINED_RESULT_FILE,
    SECTION3BARI_VALIDATION_FILE,
    SECTION3BARI_SUMMARY_FILE,
]


assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in section3bari_saved_files
)


print()
print("SAVED SECTION 3B-A REPAIR I REPORTS")
print("-" * 100)

for file_path in section3bari_saved_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A CATEGORICAL ENCODING "
    "COMPATIBILITY AUDIT PASSED"
)


# In[30]:


# ======================================================================================
# SECTION 3B-A REPAIR I ROLLBACK — RESTORE REPAIR H FEATURE BUILDER
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR I ROLLBACK — RESTORE REPAIR H FEATURE BUILDER")
print("=" * 100)

assert (
    "build_legality_aware_feature_row_55"
    in globals()
), (
    "Repair H feature builder is unavailable."
)

build_legality_aware_feature_row = (
    build_legality_aware_feature_row_55
)

test_signature_df = build_legality_aware_feature_row(
    battle_state=section3barh_test_state,
    legal_moves=section3barh_test_legal_moves,
    side_mode="PRESERVE",
)

restored_signature = str(
    test_signature_df.iloc[0][
        "legal_move_signature"
    ]
)

print("Restored signature:", restored_signature)

assert restored_signature == "Ascension | Quick Attack"

print()
print("✅ REPAIR H FEATURE BUILDER RESTORED")


# In[31]:


# ======================================================================================
# SECTION 3B-A REPAIR J — MODEL AND PREPROCESSOR SNAPSHOT IDENTITY AUDIT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR J — MODEL AND PREPROCESSOR SNAPSHOT IDENTITY AUDIT")
print("=" * 100)

from pathlib import Path
import hashlib
import json
import pickle

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate current baseline objects
# --------------------------------------------------------------------------------------

SECTION3BARJ_REQUIRED_OBJECTS = [
    "baseline_policy_model",
    "baseline_preprocessor",
    "baseline_encoded_feature_names",
    "PROJECT_ROOT",
    "REPORTS_DIRECTORY",
]


section3barj_missing_objects = [
    object_name
    for object_name in SECTION3BARJ_REQUIRED_OBJECTS
    if object_name not in globals()
]


assert not section3barj_missing_objects, (
    "Repair J required objects are missing: "
    f"{section3barj_missing_objects}"
)


# --------------------------------------------------------------------------------------
# 2. Hash helpers
# --------------------------------------------------------------------------------------

def section3barj_object_hash(
    object_value,
):
    """
    Create a deterministic-enough runtime snapshot hash using pickle bytes.
    """

    serialized = pickle.dumps(
        object_value,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    return hashlib.sha256(
        serialized
    ).hexdigest()


def section3barj_array_hash(
    array_value,
):
    array_value = np.asarray(
        array_value
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            array_value.shape
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        array_value.tobytes()
    )

    return digest.hexdigest()


CURRENT_MODEL_OBJECT_HASH_3BARJ = (
    section3barj_object_hash(
        baseline_policy_model
    )
)


CURRENT_PREPROCESSOR_OBJECT_HASH_3BARJ = (
    section3barj_object_hash(
        baseline_preprocessor
    )
)


CURRENT_MODEL_IMPORTANCE_HASH_3BARJ = (
    section3barj_array_hash(
        baseline_policy_model.feature_importances_
    )
)


CURRENT_FEATURE_NAME_HASH_3BARJ = (
    hashlib.sha256(
        "\n".join(
            map(
                str,
                baseline_encoded_feature_names,
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()
)


print()
print("CURRENT RESTORED OBJECT HASHES")
print("-" * 100)

print(
    "Model object hash       :",
    CURRENT_MODEL_OBJECT_HASH_3BARJ,
)

print(
    "Preprocessor object hash:",
    CURRENT_PREPROCESSOR_OBJECT_HASH_3BARJ,
)

print(
    "Importance hash         :",
    CURRENT_MODEL_IMPORTANCE_HASH_3BARJ,
)

print(
    "Feature-name hash       :",
    CURRENT_FEATURE_NAME_HASH_3BARJ,
)


# --------------------------------------------------------------------------------------
# 3. Search likely model/artifact directories
# --------------------------------------------------------------------------------------

SECTION3BARJ_SEARCH_DIRECTORIES = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "reports",
]


SECTION3BARJ_MODEL_FILE_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pickle",
}


section3barj_candidate_paths = []


for search_directory in SECTION3BARJ_SEARCH_DIRECTORIES:

    if not search_directory.exists():
        continue

    for candidate_path in search_directory.rglob("*"):

        if (
            candidate_path.is_file()
            and
            candidate_path.suffix.lower()
            in SECTION3BARJ_MODEL_FILE_SUFFIXES
        ):

            section3barj_candidate_paths.append(
                candidate_path
            )


print()
print("SERIALIZED ARTIFACT FILES FOUND")
print("-" * 100)

print(
    "Candidate files:",
    len(
        section3barj_candidate_paths
    )
)


# --------------------------------------------------------------------------------------
# 4. Inspect serialized artifacts safely
# --------------------------------------------------------------------------------------

section3barj_artifact_rows = []
section3barj_loaded_candidates = {}


def section3barj_collect_objects(
    loaded_value,
):
    """
    Return named objects from a serialized artifact.
    """

    collected = []


    if isinstance(
        loaded_value,
        dict,
    ):

        for key, value in loaded_value.items():

            collected.append(
                (
                    str(key),
                    value,
                )
            )

    else:

        collected.append(
            (
                "root_object",
                loaded_value,
            )
        )


    return collected


for candidate_path in sorted(
    section3barj_candidate_paths
):

    try:

        loaded_value = joblib.load(
            candidate_path
        )

    except Exception as error:

        section3barj_artifact_rows.append(
            {
                "file_path":
                    str(
                        candidate_path
                    ),

                "object_name":
                    "",

                "object_type":
                    "LOAD_ERROR",

                "is_model_candidate":
                    False,

                "is_preprocessor_candidate":
                    False,

                "feature_count":
                    None,

                "model_object_hash":
                    None,

                "preprocessor_object_hash":
                    None,

                "importance_hash":
                    None,

                "feature_name_hash":
                    None,

                "load_error":
                    f"{type(error).__name__}: {error}",
            }
        )

        continue


    for object_name, object_value in (
        section3barj_collect_objects(
            loaded_value
        )
    ):

        is_model_candidate = bool(
            hasattr(
                object_value,
                "predict",
            )
            and
            hasattr(
                object_value,
                "classes_",
            )
        )


        is_preprocessor_candidate = bool(
            hasattr(
                object_value,
                "transform",
            )
            and
            hasattr(
                object_value,
                "transformers_",
            )
        )


        feature_count = getattr(
            object_value,
            "n_features_in_",
            None,
        )


        model_object_hash = None
        preprocessor_object_hash = None
        importance_hash = None
        feature_name_hash = None


        if is_model_candidate:

            try:

                model_object_hash = (
                    section3barj_object_hash(
                        object_value
                    )
                )

            except Exception:

                pass


            if hasattr(
                object_value,
                "feature_importances_",
            ):

                importance_hash = (
                    section3barj_array_hash(
                        object_value.feature_importances_
                    )
                )


        if is_preprocessor_candidate:

            try:

                preprocessor_object_hash = (
                    section3barj_object_hash(
                        object_value
                    )
                )

            except Exception:

                pass


            if hasattr(
                object_value,
                "get_feature_names_out",
            ):

                try:

                    candidate_feature_names = list(
                        object_value.get_feature_names_out()
                    )

                    feature_name_hash = hashlib.sha256(
                        "\n".join(
                            map(
                                str,
                                candidate_feature_names,
                            )
                        ).encode(
                            "utf-8"
                        )
                    ).hexdigest()

                except Exception:

                    pass


        section3barj_artifact_rows.append(
            {
                "file_path":
                    str(
                        candidate_path
                    ),

                "object_name":
                    object_name,

                "object_type":
                    type(
                        object_value
                    ).__name__,

                "is_model_candidate":
                    is_model_candidate,

                "is_preprocessor_candidate":
                    is_preprocessor_candidate,

                "feature_count":
                    feature_count,

                "model_object_hash":
                    model_object_hash,

                "preprocessor_object_hash":
                    preprocessor_object_hash,

                "importance_hash":
                    importance_hash,

                "feature_name_hash":
                    feature_name_hash,

                "load_error":
                    "",
            }
        )


        if (
            is_model_candidate
            or
            is_preprocessor_candidate
        ):

            section3barj_loaded_candidates[
                (
                    str(
                        candidate_path
                    ),
                    object_name,
                )
            ] = object_value


section3barj_artifact_inventory_df = pd.DataFrame(
    section3barj_artifact_rows
)


print()
print("MODEL AND PREPROCESSOR ARTIFACT INVENTORY")
print("-" * 100)

display(
    section3barj_artifact_inventory_df.loc[
        section3barj_artifact_inventory_df[
            "is_model_candidate"
        ].astype(bool)
        |
        section3barj_artifact_inventory_df[
            "is_preprocessor_candidate"
        ].astype(bool)
    ]
    .reset_index(drop=True)
)


# --------------------------------------------------------------------------------------
# 5. Compare candidates with current restored objects
# --------------------------------------------------------------------------------------

section3barj_comparison_df = (
    section3barj_artifact_inventory_df.loc[
        section3barj_artifact_inventory_df[
            "is_model_candidate"
        ].astype(bool)
        |
        section3barj_artifact_inventory_df[
            "is_preprocessor_candidate"
        ].astype(bool)
    ]
    .copy()
)


section3barj_comparison_df[
    "exact_current_model_match"
] = (
    section3barj_comparison_df[
        "model_object_hash"
    ]
    .eq(
        CURRENT_MODEL_OBJECT_HASH_3BARJ
    )
)


section3barj_comparison_df[
    "same_feature_importances"
] = (
    section3barj_comparison_df[
        "importance_hash"
    ]
    .eq(
        CURRENT_MODEL_IMPORTANCE_HASH_3BARJ
    )
)


section3barj_comparison_df[
    "exact_current_preprocessor_match"
] = (
    section3barj_comparison_df[
        "preprocessor_object_hash"
    ]
    .eq(
        CURRENT_PREPROCESSOR_OBJECT_HASH_3BARJ
    )
)


section3barj_comparison_df[
    "same_feature_name_schema"
] = (
    section3barj_comparison_df[
        "feature_name_hash"
    ]
    .eq(
        CURRENT_FEATURE_NAME_HASH_3BARJ
    )
)


print()
print("SNAPSHOT IDENTITY COMPARISON")
print("-" * 100)

display(
    section3barj_comparison_df
    .sort_values(
        [
            "exact_current_model_match",
            "same_feature_importances",
            "exact_current_preprocessor_match",
            "same_feature_name_schema",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


# --------------------------------------------------------------------------------------
# 6. Identify alternate 64-feature model snapshots
# --------------------------------------------------------------------------------------

section3barj_alternate_model_df = (
    section3barj_comparison_df.loc[
        section3barj_comparison_df[
            "is_model_candidate"
        ].astype(bool)
        &
        section3barj_comparison_df[
            "feature_count"
        ].eq(
            64
        )
        &
        ~section3barj_comparison_df[
            "exact_current_model_match"
        ].astype(bool)
    ]
    .copy()
    .reset_index(drop=True)
)


section3barj_alternate_preprocessor_df = (
    section3barj_comparison_df.loc[
        section3barj_comparison_df[
            "is_preprocessor_candidate"
        ].astype(bool)
        &
        ~section3barj_comparison_df[
            "exact_current_preprocessor_match"
        ].astype(bool)
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print("ALTERNATE 64-FEATURE MODEL SNAPSHOTS")
print("-" * 100)

if section3barj_alternate_model_df.empty:

    print(
        "No alternate serialized 64-feature model snapshot was found."
    )

else:

    display(
        section3barj_alternate_model_df
    )


print()
print("ALTERNATE PREPROCESSOR SNAPSHOTS")
print("-" * 100)

if section3barj_alternate_preprocessor_df.empty:

    print(
        "No alternate serialized preprocessor snapshot was found."
    )

else:

    display(
        section3barj_alternate_preprocessor_df
    )


# --------------------------------------------------------------------------------------
# 7. Determine diagnostic outcome
# --------------------------------------------------------------------------------------

section3barj_exact_model_matches = int(
    section3barj_comparison_df[
        "exact_current_model_match"
    ].astype(bool).sum()
)


section3barj_importance_matches = int(
    section3barj_comparison_df[
        "same_feature_importances"
    ].astype(bool).sum()
)


section3barj_exact_preprocessor_matches = int(
    section3barj_comparison_df[
        "exact_current_preprocessor_match"
    ].astype(bool).sum()
)


section3barj_alternate_models_found = int(
    len(
        section3barj_alternate_model_df
    )
)


section3barj_alternate_preprocessors_found = int(
    len(
        section3barj_alternate_preprocessor_df
    )
)


if section3barj_alternate_models_found > 0:

    section3barj_status = (
        "ALTERNATE_MODEL_SNAPSHOTS_FOUND"
    )

    section3barj_next_stage = (
        "SCORE_ALTERNATE_MODEL_SNAPSHOTS_AGAINST_NOTEBOOK54_PROBABILITIES"
    )


elif section3barj_alternate_preprocessors_found > 0:

    section3barj_status = (
        "ALTERNATE_PREPROCESSOR_SNAPSHOTS_FOUND"
    )

    section3barj_next_stage = (
        "SCORE_ALTERNATE_PREPROCESSOR_SNAPSHOTS"
    )


else:

    section3barj_status = (
        "NO_ALTERNATE_SERIALIZED_SNAPSHOT_FOUND"
    )

    section3barj_next_stage = (
        "COMPARE_NOTEBOOK54_AND_NOTEBOOK53_RUNTIME_ASSIGNMENTS"
    )


section3barj_summary = {
    "status":
        section3barj_status,

    "serialized_files_scanned":
        int(
            len(
                section3barj_candidate_paths
            )
        ),

    "model_or_preprocessor_objects_found":
        int(
            len(
                section3barj_comparison_df
            )
        ),

    "exact_current_model_matches":
        section3barj_exact_model_matches,

    "feature_importance_matches":
        section3barj_importance_matches,

    "exact_current_preprocessor_matches":
        section3barj_exact_preprocessor_matches,

    "alternate_64_feature_models_found":
        section3barj_alternate_models_found,

    "alternate_preprocessors_found":
        section3barj_alternate_preprocessors_found,

    "current_model_object_hash":
        CURRENT_MODEL_OBJECT_HASH_3BARJ,

    "current_preprocessor_object_hash":
        CURRENT_PREPROCESSOR_OBJECT_HASH_3BARJ,

    "current_feature_importance_hash":
        CURRENT_MODEL_IMPORTANCE_HASH_3BARJ,

    "current_feature_name_hash":
        CURRENT_FEATURE_NAME_HASH_3BARJ,

    "next_stage":
        section3barj_next_stage,
}


print()
print("SECTION 3B-A REPAIR J SUMMARY")
print("-" * 100)

for key, value in section3barj_summary.items():

    print(
        f"{key:66}: {value}"
    )


# --------------------------------------------------------------------------------------
# 8. Save reports
# --------------------------------------------------------------------------------------

SECTION3BARJ_ARTIFACT_INVENTORY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barj_snapshot_artifact_inventory.csv"
)

SECTION3BARJ_COMPARISON_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barj_snapshot_identity_comparison.csv"
)

SECTION3BARJ_ALTERNATE_MODELS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barj_alternate_model_snapshots.csv"
)

SECTION3BARJ_ALTERNATE_PREPROCESSORS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barj_alternate_preprocessor_snapshots.csv"
)

SECTION3BARJ_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barj_snapshot_identity_summary.json"
)


section3barj_artifact_inventory_df.to_csv(
    SECTION3BARJ_ARTIFACT_INVENTORY_FILE,
    index=False,
)

section3barj_comparison_df.to_csv(
    SECTION3BARJ_COMPARISON_FILE,
    index=False,
)

section3barj_alternate_model_df.to_csv(
    SECTION3BARJ_ALTERNATE_MODELS_FILE,
    index=False,
)

section3barj_alternate_preprocessor_df.to_csv(
    SECTION3BARJ_ALTERNATE_PREPROCESSORS_FILE,
    index=False,
)


with open(
    SECTION3BARJ_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barj_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR J REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARJ_ARTIFACT_INVENTORY_FILE,
    SECTION3BARJ_COMPARISON_FILE,
    SECTION3BARJ_ALTERNATE_MODELS_FILE,
    SECTION3BARJ_ALTERNATE_PREPROCESSORS_FILE,
    SECTION3BARJ_SUMMARY_FILE,
]:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A MODEL AND PREPROCESSOR "
    "SNAPSHOT IDENTITY AUDIT PASSED"
)


# In[32]:


# ======================================================================================
# SECTION 3B-A REPAIR K — SCORE ALTERNATE MODEL/PREPROCESSOR SNAPSHOTS
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR K — SCORE ALTERNATE MODEL/PREPROCESSOR SNAPSHOTS")
print("=" * 100)

from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate dependencies
# --------------------------------------------------------------------------------------

SECTION3BARK_REQUIRED_OBJECTS = [
    "section3barj_loaded_candidates",
    "section3barj_comparison_df",
    "section3ba_score_cases_df",
    "section3ba_expanded_case_results_df",
    "build_section4wf_evaluation_state",
    "section3barh_original_feature_builder",
    "normalize_section3barg_legal_moves",
    "SECTION3BARG_LEGAL_MOVE_COLUMN",
    "baseline_policy_model",
    "baseline_preprocessor",
    "REPORTS_DIRECTORY",
]


section3bark_missing_objects = [
    object_name
    for object_name in SECTION3BARK_REQUIRED_OBJECTS
    if object_name not in globals()
]


assert not section3bark_missing_objects, (
    "Repair K required objects are missing: "
    f"{section3bark_missing_objects}"
)


# --------------------------------------------------------------------------------------
# 2. Build model and preprocessor candidate catalogs
# --------------------------------------------------------------------------------------

section3bark_model_candidates = {
    "CURRENT_BASELINE_MODEL":
        baseline_policy_model,
}


section3bark_preprocessor_candidates = {
    "CURRENT_BASELINE_PREPROCESSOR":
        baseline_preprocessor,
}


section3bark_candidate_metadata_rows = []


for (
    candidate_path,
    candidate_object_name,
), candidate_object in (
    section3barj_loaded_candidates.items()
):

    candidate_label = (
        f"{Path(candidate_path).name}"
        f"::{candidate_object_name}"
    )


    if (
        hasattr(candidate_object, "predict")
        and
        hasattr(candidate_object, "classes_")
    ):

        feature_count = getattr(
            candidate_object,
            "n_features_in_",
            None,
        )


        if feature_count == 64:

            section3bark_model_candidates[
                candidate_label
            ] = candidate_object


            section3bark_candidate_metadata_rows.append(
                {
                    "candidate_type":
                        "MODEL",

                    "candidate_label":
                        candidate_label,

                    "file_path":
                        candidate_path,

                    "object_name":
                        candidate_object_name,

                    "object_type":
                        type(candidate_object).__name__,

                    "feature_count":
                        int(feature_count),
                }
            )


    if (
        hasattr(candidate_object, "transform")
        and
        hasattr(candidate_object, "transformers_")
    ):

        feature_count = getattr(
            candidate_object,
            "n_features_in_",
            None,
        )


        if feature_count == 41:

            section3bark_preprocessor_candidates[
                candidate_label
            ] = candidate_object


            section3bark_candidate_metadata_rows.append(
                {
                    "candidate_type":
                        "PREPROCESSOR",

                    "candidate_label":
                        candidate_label,

                    "file_path":
                        candidate_path,

                    "object_name":
                        candidate_object_name,

                    "object_type":
                        type(candidate_object).__name__,

                    "feature_count":
                        int(feature_count),
                }
            )


section3bark_candidate_catalog_df = pd.DataFrame(
    section3bark_candidate_metadata_rows
)


print()
print("64-FEATURE MODEL CANDIDATES")
print("-" * 100)

for candidate_name in section3bark_model_candidates:
    print(candidate_name)


print()
print("41-INPUT PREPROCESSOR CANDIDATES")
print("-" * 100)

for candidate_name in section3bark_preprocessor_candidates:
    print(candidate_name)


assert len(section3bark_model_candidates) >= 2
assert len(section3bark_preprocessor_candidates) >= 2


# --------------------------------------------------------------------------------------
# 3. Build authoritative raw feature rows once
# --------------------------------------------------------------------------------------

section3bark_case_lookup_df = (
    section3ba_expanded_case_results_df
    .drop_duplicates(
        subset=["comparison_case_id"]
    )
    .set_index(
        "comparison_case_id",
        drop=False,
    )
)


section3bark_raw_case_records = []
section3bark_raw_errors = []


for _, score_row in section3ba_score_cases_df.iterrows():

    comparison_case_id = str(
        score_row["comparison_case_id"]
    )


    try:

        source_row = section3bark_case_lookup_df.loc[
            comparison_case_id
        ]


        evaluation_state = (
            build_section4wf_evaluation_state(
                score_row
            )
        )


        legal_moves = normalize_section3barg_legal_moves(
            source_row[
                SECTION3BARG_LEGAL_MOVE_COLUMN
            ]
        )


        # Original builder produces the lowercase signature.
        lowercase_feature_df = (
            section3barh_original_feature_builder(
                battle_state=evaluation_state,
                legal_moves=legal_moves,
                side_mode="PRESERVE",
            )
            .copy()
        )


        assert lowercase_feature_df.shape == (1, 41)


        titlecase_feature_df = (
            lowercase_feature_df.copy()
        )


        titlecase_feature_df[
            "legal_move_signature"
        ] = (
            titlecase_feature_df[
                "legal_move_signature"
            ]
            .apply(
                section3barh_restore_signature_case
            )
        )


        section3bark_raw_case_records.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "score_row":
                    score_row.copy(),

                "lowercase_feature_df":
                    lowercase_feature_df,

                "titlecase_feature_df":
                    titlecase_feature_df,
            }
        )


    except Exception as error:

        section3bark_raw_errors.append(
            {
                "comparison_case_id":
                    comparison_case_id,

                "error_type":
                    type(error).__name__,

                "error_message":
                    str(error),
            }
        )


section3bark_raw_errors_df = pd.DataFrame(
    section3bark_raw_errors,
    columns=[
        "comparison_case_id",
        "error_type",
        "error_message",
    ],
)


assert section3bark_raw_errors_df.empty, (
    "One or more authoritative raw rows could not be built."
)


assert len(section3bark_raw_case_records) == 184


# --------------------------------------------------------------------------------------
# 4. Evaluate one model/preprocessor/signature combination
# --------------------------------------------------------------------------------------

def section3bark_evaluate_combination(
    model_name,
    model,
    preprocessor_name,
    preprocessor,
    signature_mode,
):
    result_rows = []
    error_rows = []


    model_classes = [
        str(class_name).strip()
        for class_name in model.classes_
    ]


    normalized_classes = [
        class_name.lower()
        for class_name in model_classes
    ]


    if (
        "quick attack" not in normalized_classes
        or
        "ascension" not in normalized_classes
    ):

        return {
            "valid":
                False,

            "error_message":
                "Required Quick Attack/Ascension classes are absent.",
        }


    quick_attack_index = normalized_classes.index(
        "quick attack"
    )


    ascension_index = normalized_classes.index(
        "ascension"
    )


    for case_record in section3bark_raw_case_records:

        comparison_case_id = case_record[
            "comparison_case_id"
        ]


        score_row = case_record[
            "score_row"
        ]


        feature_df = (
            case_record["titlecase_feature_df"]
            if signature_mode == "TITLE_CASE"
            else case_record["lowercase_feature_df"]
        )


        try:

            encoded_matrix = preprocessor.transform(
                feature_df
            )


            encoded_columns = int(
                encoded_matrix.shape[1]
            )


            required_model_columns = int(
                getattr(
                    model,
                    "n_features_in_",
                    encoded_columns,
                )
            )


            if encoded_columns != required_model_columns:

                raise ValueError(
                    "Encoded/model feature mismatch: "
                    f"{encoded_columns} versus "
                    f"{required_model_columns}"
                )


            probabilities = np.asarray(
                model.predict_proba(
                    encoded_matrix
                )
            )[0]


            reconstructed_quick_attack = float(
                probabilities[
                    quick_attack_index
                ]
            )


            reconstructed_ascension = float(
                probabilities[
                    ascension_index
                ]
            )


            saved_quick_attack = float(
                score_row[
                    "quick_attack_probability"
                ]
            )


            saved_ascension = float(
                score_row[
                    "ascension_probability"
                ]
            )


            reconstructed_prediction = str(
                model_classes[
                    int(
                        np.argmax(
                            probabilities
                        )
                    )
                ]
            )


            saved_prediction = str(
                score_row[
                    "predicted_action"
                ]
            )


            result_rows.append(
                {
                    "comparison_case_id":
                        comparison_case_id,

                    "quick_attack_absolute_difference":
                        abs(
                            reconstructed_quick_attack
                            -
                            saved_quick_attack
                        ),

                    "ascension_absolute_difference":
                        abs(
                            reconstructed_ascension
                            -
                            saved_ascension
                        ),

                    "prediction_agreement":
                        (
                            reconstructed_prediction.lower()
                            ==
                            saved_prediction.lower()
                        ),
                }
            )


        except Exception as error:

            error_rows.append(
                {
                    "comparison_case_id":
                        comparison_case_id,

                    "error_type":
                        type(error).__name__,

                    "error_message":
                        str(error),
                }
            )


    result_df = pd.DataFrame(
        result_rows
    )


    errors_df = pd.DataFrame(
        error_rows
    )


    if not errors_df.empty or len(result_df) != 184:

        return {
            "valid":
                False,

            "error_message":
                (
                    errors_df.iloc[0]["error_message"]
                    if not errors_df.empty
                    else "Incomplete result rows"
                ),
        }


    return {
        "valid":
            True,

        "model_name":
            model_name,

        "preprocessor_name":
            preprocessor_name,

        "signature_mode":
            signature_mode,

        "cases_scored":
            int(len(result_df)),

        "mean_quick_attack_difference":
            float(
                result_df[
                    "quick_attack_absolute_difference"
                ].mean()
            ),

        "maximum_quick_attack_difference":
            float(
                result_df[
                    "quick_attack_absolute_difference"
                ].max()
            ),

        "mean_ascension_difference":
            float(
                result_df[
                    "ascension_absolute_difference"
                ].mean()
            ),

        "maximum_ascension_difference":
            float(
                result_df[
                    "ascension_absolute_difference"
                ].max()
            ),

        "prediction_agreement_rate":
            float(
                result_df[
                    "prediction_agreement"
                ].astype(bool).mean()
            ),

        "result_df":
            result_df,
    }


# --------------------------------------------------------------------------------------
# 5. Score every compatible combination
# --------------------------------------------------------------------------------------

section3bark_combination_rows = []
section3bark_combination_details = {}


for model_name, model in (
    section3bark_model_candidates.items()
):

    for preprocessor_name, preprocessor in (
        section3bark_preprocessor_candidates.items()
    ):

        for signature_mode in [
            "LOWER_CASE",
            "TITLE_CASE",
        ]:

            combination_result = (
                section3bark_evaluate_combination(
                    model_name=model_name,
                    model=model,
                    preprocessor_name=preprocessor_name,
                    preprocessor=preprocessor,
                    signature_mode=signature_mode,
                )
            )


            combination_id = (
                f"{model_name} || "
                f"{preprocessor_name} || "
                f"{signature_mode}"
            )


            section3bark_combination_details[
                combination_id
            ] = combination_result


            section3bark_combination_rows.append(
                {
                    "combination_id":
                        combination_id,

                    "model_name":
                        model_name,

                    "preprocessor_name":
                        preprocessor_name,

                    "signature_mode":
                        signature_mode,

                    "valid":
                        bool(
                            combination_result.get(
                                "valid",
                                False,
                            )
                        ),

                    "cases_scored":
                        combination_result.get(
                            "cases_scored",
                            0,
                        ),

                    "mean_quick_attack_difference":
                        combination_result.get(
                            "mean_quick_attack_difference"
                        ),

                    "maximum_quick_attack_difference":
                        combination_result.get(
                            "maximum_quick_attack_difference"
                        ),

                    "mean_ascension_difference":
                        combination_result.get(
                            "mean_ascension_difference"
                        ),

                    "maximum_ascension_difference":
                        combination_result.get(
                            "maximum_ascension_difference"
                        ),

                    "prediction_agreement_rate":
                        combination_result.get(
                            "prediction_agreement_rate"
                        ),

                    "error_message":
                        combination_result.get(
                            "error_message",
                            "",
                        ),
                }
            )


section3bark_combination_results_df = pd.DataFrame(
    section3bark_combination_rows
)


valid_combination_results_df = (
    section3bark_combination_results_df.loc[
        section3bark_combination_results_df[
            "valid"
        ].astype(bool)
    ]
    .copy()
)


assert not valid_combination_results_df.empty


valid_combination_results_df[
    "combined_mean_probability_difference"
] = (
    valid_combination_results_df[
        "mean_quick_attack_difference"
    ]
    +
    valid_combination_results_df[
        "mean_ascension_difference"
    ]
)


valid_combination_results_df[
    "combined_maximum_probability_difference"
] = (
    valid_combination_results_df[
        "maximum_quick_attack_difference"
    ]
    +
    valid_combination_results_df[
        "maximum_ascension_difference"
    ]
)


valid_combination_results_df = (
    valid_combination_results_df
    .sort_values(
        [
            "combined_mean_probability_difference",
            "combined_maximum_probability_difference",
            "prediction_agreement_rate",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)


print()
print("MODEL/PREPROCESSOR SNAPSHOT SCORING RESULTS")
print("-" * 100)

display(
    valid_combination_results_df
)


# --------------------------------------------------------------------------------------
# 6. Select best matching snapshot combination
# --------------------------------------------------------------------------------------

section3bark_best_row = (
    valid_combination_results_df.iloc[0]
)


section3bark_best_combination_id = str(
    section3bark_best_row[
        "combination_id"
    ]
)


section3bark_best_model_name = str(
    section3bark_best_row[
        "model_name"
    ]
)


section3bark_best_preprocessor_name = str(
    section3bark_best_row[
        "preprocessor_name"
    ]
)


section3bark_best_signature_mode = str(
    section3bark_best_row[
        "signature_mode"
    ]
)


section3bark_best_model = (
    section3bark_model_candidates[
        section3bark_best_model_name
    ]
)


section3bark_best_preprocessor = (
    section3bark_preprocessor_candidates[
        section3bark_best_preprocessor_name
    ]
)


SECTION3BARK_EXACT_TOLERANCE = 1e-10


section3bark_exact_match = bool(
    float(
        section3bark_best_row[
            "maximum_quick_attack_difference"
        ]
    )
    <=
    SECTION3BARK_EXACT_TOLERANCE
    and
    float(
        section3bark_best_row[
            "maximum_ascension_difference"
        ]
    )
    <=
    SECTION3BARK_EXACT_TOLERANCE
    and
    float(
        section3bark_best_row[
            "prediction_agreement_rate"
        ]
    )
    == 1.0
)


# --------------------------------------------------------------------------------------
# 7. Install only an exact snapshot match
# --------------------------------------------------------------------------------------

if section3bark_exact_match:

    NOTEBOOK54_EXACT_POLICY_MODEL = (
        section3bark_best_model
    )


    NOTEBOOK54_EXACT_PREPROCESSOR = (
        section3bark_best_preprocessor
    )


    NOTEBOOK54_EXACT_SIGNATURE_MODE = (
        section3bark_best_signature_mode
    )


    baseline_policy_model = (
        NOTEBOOK54_EXACT_POLICY_MODEL
    )


    baseline_preprocessor = (
        NOTEBOOK54_EXACT_PREPROCESSOR
    )


    if NOTEBOOK54_EXACT_SIGNATURE_MODE == "TITLE_CASE":

        build_legality_aware_feature_row = (
            build_legality_aware_feature_row_55
        )

    else:

        build_legality_aware_feature_row = (
            section3barh_original_feature_builder
        )


    section3bark_status = (
        "EXACT_NOTEBOOK54_MODEL_PREPROCESSOR_SNAPSHOT_IDENTIFIED"
    )


    section3bark_next_stage = (
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES"
    )


else:

    section3bark_status = (
        "BEST_AVAILABLE_SNAPSHOT_STILL_HAS_RESIDUAL_DRIFT"
    )


    section3bark_next_stage = (
        "COMPARE_NOTEBOOK54_RUNTIME_MODEL_ASSIGNMENTS"
    )


# --------------------------------------------------------------------------------------
# 8. Summary
# --------------------------------------------------------------------------------------

section3bark_summary = {
    "status":
        section3bark_status,

    "model_candidates":
        int(
            len(
                section3bark_model_candidates
            )
        ),

    "preprocessor_candidates":
        int(
            len(
                section3bark_preprocessor_candidates
            )
        ),

    "valid_combinations_scored":
        int(
            len(
                valid_combination_results_df
            )
        ),

    "best_model":
        section3bark_best_model_name,

    "best_preprocessor":
        section3bark_best_preprocessor_name,

    "best_signature_mode":
        section3bark_best_signature_mode,

    "best_mean_quick_attack_difference":
        float(
            section3bark_best_row[
                "mean_quick_attack_difference"
            ]
        ),

    "best_maximum_quick_attack_difference":
        float(
            section3bark_best_row[
                "maximum_quick_attack_difference"
            ]
        ),

    "best_mean_ascension_difference":
        float(
            section3bark_best_row[
                "mean_ascension_difference"
            ]
        ),

    "best_maximum_ascension_difference":
        float(
            section3bark_best_row[
                "maximum_ascension_difference"
            ]
        ),

    "best_prediction_agreement_rate":
        float(
            section3bark_best_row[
                "prediction_agreement_rate"
            ]
        ),

    "exact_snapshot_match":
        section3bark_exact_match,

    "next_stage":
        section3bark_next_stage,
}


print()
print("SECTION 3B-A REPAIR K SUMMARY")
print("-" * 100)

for key, value in section3bark_summary.items():

    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 9. Validation checks
# --------------------------------------------------------------------------------------

section3bark_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "multiple_model_candidates_scored",

            "passed":
                len(
                    section3bark_model_candidates
                ) >= 2,

            "value":
                len(
                    section3bark_model_candidates
                ),

            "expected":
                "At least 2",
        },
        {
            "check":
                "multiple_preprocessor_candidates_scored",

            "passed":
                len(
                    section3bark_preprocessor_candidates
                ) >= 2,

            "value":
                len(
                    section3bark_preprocessor_candidates
                ),

            "expected":
                "At least 2",
        },
        {
            "check":
                "valid_snapshot_combinations_created",

            "passed":
                len(
                    valid_combination_results_df
                ) > 0,

            "value":
                len(
                    valid_combination_results_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "prediction_agreement_preserved",

            "passed":
                float(
                    section3bark_best_row[
                        "prediction_agreement_rate"
                    ]
                ) == 1.0,

            "value":
                float(
                    section3bark_best_row[
                        "prediction_agreement_rate"
                    ]
                ),

            "expected":
                1.0,
        },
        {
            "check":
                "next_stage_resolved",

            "passed":
                section3bark_next_stage
                in {
                    "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
                    "COMPARE_NOTEBOOK54_RUNTIME_MODEL_ASSIGNMENTS",
                },

            "value":
                section3bark_next_stage,

            "expected":
                "Recognized route",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR K VALIDATION CHECKS")
print("-" * 100)

display(
    section3bark_validation_checks_df
)


assert section3bark_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 10. Save Repair K reports
# --------------------------------------------------------------------------------------

SECTION3BARK_CANDIDATE_CATALOG_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bark_snapshot_candidate_catalog.csv"
)

SECTION3BARK_COMBINATION_RESULTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bark_snapshot_combination_results.csv"
)

SECTION3BARK_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bark_validation_checks.csv"
)

SECTION3BARK_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3bark_snapshot_scoring_summary.json"
)


section3bark_candidate_catalog_df.to_csv(
    SECTION3BARK_CANDIDATE_CATALOG_FILE,
    index=False,
)

section3bark_combination_results_df.to_csv(
    SECTION3BARK_COMBINATION_RESULTS_FILE,
    index=False,
)

section3bark_validation_checks_df.to_csv(
    SECTION3BARK_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARK_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3bark_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR K REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARK_CANDIDATE_CATALOG_FILE,
    SECTION3BARK_COMBINATION_RESULTS_FILE,
    SECTION3BARK_VALIDATION_FILE,
    SECTION3BARK_SUMMARY_FILE,
]:

    print(file_path)


print()
print(
    "✅ SECTION 3B-A ALTERNATE SNAPSHOT "
    "SCORING COMPLETED"
)


# In[33]:


# ======================================================================================
# SECTION 3B-A REPAIR L — TRACE NOTEBOOK 54 RUNTIME MODEL ASSIGNMENTS
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR L — TRACE NOTEBOOK 54 RUNTIME MODEL ASSIGNMENTS")
print("=" * 100)

from pathlib import Path
import ast
import json
import nbformat
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate Notebook 54 source
# --------------------------------------------------------------------------------------

NOTEBOOK54_SOURCE_PATH_3BARL = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents"
    r"\PTCG_AI_Battle_Challenge"
    r"\notebooks"
    r"\54_tournament_strength_optimization_clean.ipynb"
)


assert NOTEBOOK54_SOURCE_PATH_3BARL.exists(), (
    "Notebook 54 clean source was not found: "
    f"{NOTEBOOK54_SOURCE_PATH_3BARL}"
)


with open(
    NOTEBOOK54_SOURCE_PATH_3BARL,
    "r",
    encoding="utf-8",
) as file:

    notebook54_document_3barl = nbformat.read(
        file,
        as_version=4,
    )


notebook54_code_cells_3barl = [
    cell
    for cell in notebook54_document_3barl.cells
    if cell.cell_type == "code"
]


print()
print("Notebook 54 source:")
print(NOTEBOOK54_SOURCE_PATH_3BARL)

print(
    "Code cells:",
    len(notebook54_code_cells_3barl),
)


# --------------------------------------------------------------------------------------
# 2. Search for the exact scoring pipeline and model/preprocessor references
# --------------------------------------------------------------------------------------

SECTION3BARL_SEARCH_TERMS = [
    "section4whb_case_probability_scores_df",
    "section4whb_case_probability_scores",
    "predict_proba",
    "notebook53_legality_model",
    "notebook53_preprocessor",
    "section4wh_model",
    "section4wh_preprocessor",
    "model_source",
    "preprocessor_source",
    "build_legality_aware_feature_row",
]


section3barl_term_match_rows = []


for code_cell_index, code_cell in enumerate(
    notebook54_code_cells_3barl
):

    source_text = str(
        code_cell.source
    )


    matched_terms = [
        search_term
        for search_term in SECTION3BARL_SEARCH_TERMS
        if search_term in source_text
    ]


    if not matched_terms:

        continue


    section3barl_term_match_rows.append(
        {
            "code_cell_index":
                int(code_cell_index),

            "matched_terms":
                json.dumps(
                    matched_terms
                ),

            "source_length":
                int(
                    len(source_text)
                ),

            "source_preview":
                source_text[:3000],
        }
    )


section3barl_term_matches_df = pd.DataFrame(
    section3barl_term_match_rows
)


print()
print("NOTEBOOK 54 MODEL/PREPROCESSOR TERM MATCHES")
print("-" * 100)

display(
    section3barl_term_matches_df
)


assert not section3barl_term_matches_df.empty, (
    "No Notebook 54 scoring references were located."
)


# --------------------------------------------------------------------------------------
# 3. Extract assignments involving model and preprocessor variables
# --------------------------------------------------------------------------------------

def section3barl_target_names(
    target_node,
):
    """
    Return assignment target names from an AST node.
    """

    discovered_names = []


    if isinstance(
        target_node,
        ast.Name,
    ):

        discovered_names.append(
            target_node.id
        )


    elif isinstance(
        target_node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):

        for child_node in target_node.elts:

            discovered_names.extend(
                section3barl_target_names(
                    child_node
                )
            )


    return discovered_names


section3barl_assignment_rows = []


for code_cell_index, code_cell in enumerate(
    notebook54_code_cells_3barl
):

    source_text = str(
        code_cell.source
    )


    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if isinstance(
            node,
            ast.Assign,
        ):

            target_names = []


            for target_node in node.targets:

                target_names.extend(
                    section3barl_target_names(
                        target_node
                    )
                )


        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            target_names = section3barl_target_names(
                node.target
            )


        else:

            continue


        assignment_source = ast.get_source_segment(
            source_text,
            node,
        )


        searchable_text = " ".join(
            [
                *target_names,
                str(assignment_source),
            ]
        ).lower()


        if not any(
            keyword in searchable_text
            for keyword in [
                "model",
                "preprocessor",
                "legality",
                "feature_names",
                "predict_proba",
                "probability",
            ]
        ):

            continue


        section3barl_assignment_rows.append(
            {
                "code_cell_index":
                    int(code_cell_index),

                "line_number":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "target_names":
                    json.dumps(
                        target_names
                    ),

                "assignment_source":
                    assignment_source,
            }
        )


section3barl_assignment_inventory_df = pd.DataFrame(
    section3barl_assignment_rows
)


print()
print("NOTEBOOK 54 MODEL/PREPROCESSOR ASSIGNMENT INVENTORY")
print("-" * 100)

display(
    section3barl_assignment_inventory_df
)


# --------------------------------------------------------------------------------------
# 4. Extract complete source cells surrounding Section 4W-H-B scoring
# --------------------------------------------------------------------------------------

section3barl_scoring_cell_indices = sorted(
    {
        int(row["code_cell_index"])
        for _, row in section3barl_term_matches_df.iterrows()
        if any(
            term in str(
                row["matched_terms"]
            )
            for term in [
                "section4whb_case_probability_scores_df",
                "section4whb_case_probability_scores",
                "predict_proba",
            ]
        )
    }
)


section3barl_scoring_source_rows = []


for scoring_cell_index in section3barl_scoring_cell_indices:

    source_text = str(
        notebook54_code_cells_3barl[
            scoring_cell_index
        ].source
    )


    section3barl_scoring_source_rows.append(
        {
            "code_cell_index":
                scoring_cell_index,

            "source_length":
                int(
                    len(source_text)
                ),

            "full_source":
                source_text,
        }
    )


section3barl_scoring_sources_df = pd.DataFrame(
    section3barl_scoring_source_rows
)


print()
print("SECTION 4W-H-B SCORING SOURCE CELLS")
print("-" * 100)

for _, source_row in (
    section3barl_scoring_sources_df.iterrows()
):

    print()
    print("=" * 100)
    print(
        "CODE-CELL INDEX:",
        source_row[
            "code_cell_index"
        ],
    )
    print("=" * 100)

    print(
        source_row[
            "full_source"
        ]
    )


# --------------------------------------------------------------------------------------
# 5. Find candidate runtime variable names used with predict_proba
# --------------------------------------------------------------------------------------

section3barl_predict_proba_rows = []


for scoring_cell_index in section3barl_scoring_cell_indices:

    source_text = str(
        notebook54_code_cells_3barl[
            scoring_cell_index
        ].source
    )


    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue


        function_node = node.func


        if not (
            isinstance(
                function_node,
                ast.Attribute,
            )
            and
            function_node.attr
            ==
            "predict_proba"
        ):

            continue


        model_expression = ast.get_source_segment(
            source_text,
            function_node.value,
        )


        argument_expressions = [
            ast.get_source_segment(
                source_text,
                argument_node,
            )
            for argument_node in node.args
        ]


        section3barl_predict_proba_rows.append(
            {
                "code_cell_index":
                    int(
                        scoring_cell_index
                    ),

                "line_number":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "model_expression":
                    model_expression,

                "argument_expressions":
                    json.dumps(
                        argument_expressions
                    ),

                "complete_call":
                    ast.get_source_segment(
                        source_text,
                        node,
                    ),
            }
        )


section3barl_predict_proba_calls_df = pd.DataFrame(
    section3barl_predict_proba_rows
)


print()
print("EXACT PREDICT_PROBA CALLS")
print("-" * 100)

display(
    section3barl_predict_proba_calls_df
)


assert not section3barl_predict_proba_calls_df.empty, (
    "No predict_proba call was found in Notebook 54 scoring cells."
)


# --------------------------------------------------------------------------------------
# 6. Resolve the primary model expression
# --------------------------------------------------------------------------------------

section3barl_model_expressions = (
    section3barl_predict_proba_calls_df[
        "model_expression"
    ]
    .dropna()
    .astype(str)
    .value_counts()
    .rename_axis(
        "model_expression"
    )
    .reset_index(
        name="usage_count"
    )
)


print()
print("MODEL EXPRESSIONS USED FOR NOTEBOOK 54 SCORING")
print("-" * 100)

display(
    section3barl_model_expressions
)


section3barl_primary_model_expression = str(
    section3barl_model_expressions.iloc[0][
        "model_expression"
    ]
)


# --------------------------------------------------------------------------------------
# 7. Search for the primary model expression assignment
# --------------------------------------------------------------------------------------

section3barl_primary_model_assignments_df = (
    section3barl_assignment_inventory_df.loc[
        section3barl_assignment_inventory_df[
            "target_names"
        ]
        .astype(str)
        .str.contains(
            section3barl_primary_model_expression,
            regex=False,
        )
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print("PRIMARY SCORING MODEL ASSIGNMENTS")
print("-" * 100)

if section3barl_primary_model_assignments_df.empty:

    print(
        "No direct assignment was found for:",
        section3barl_primary_model_expression,
    )

else:

    display(
        section3barl_primary_model_assignments_df
    )


# --------------------------------------------------------------------------------------
# 8. Inspect current Notebook 55 runtime objects with matching names
# --------------------------------------------------------------------------------------

section3barl_runtime_name_candidates = sorted(
    {
        "notebook53_legality_model",
        "notebook53_preprocessor",
        "legality_aware_policy_model",
        "legality_preprocessor",
        "baseline_policy_model",
        "baseline_preprocessor",
        section3barl_primary_model_expression,
    }
)


section3barl_runtime_object_rows = []


for object_name in section3barl_runtime_name_candidates:

    object_available = (
        object_name in globals()
    )


    object_value = globals().get(
        object_name
    )


    section3barl_runtime_object_rows.append(
        {
            "object_name":
                object_name,

            "available":
                object_available,

            "object_type":
                (
                    type(
                        object_value
                    ).__name__
                    if object_available
                    else "MISSING"
                ),

            "feature_count":
                (
                    getattr(
                        object_value,
                        "n_features_in_",
                        None,
                    )
                    if object_available
                    else None
                ),

            "has_predict_proba":
                bool(
                    object_available
                    and
                    hasattr(
                        object_value,
                        "predict_proba",
                    )
                ),

            "has_transform":
                bool(
                    object_available
                    and
                    hasattr(
                        object_value,
                        "transform",
                    )
                ),
        }
    )


section3barl_runtime_object_profile_df = pd.DataFrame(
    section3barl_runtime_object_rows
)


print()
print("NOTEBOOK 55 RUNTIME OBJECT PROFILE")
print("-" * 100)

display(
    section3barl_runtime_object_profile_df
)


# --------------------------------------------------------------------------------------
# 9. Summary
# --------------------------------------------------------------------------------------

section3barl_summary = {
    "status":
        "NOTEBOOK54_RUNTIME_MODEL_ASSIGNMENT_TRACE_COMPLETE",

    "matching_source_cells":
        int(
            len(
                section3barl_term_matches_df
            )
        ),

    "scoring_source_cells":
        section3barl_scoring_cell_indices,

    "predict_proba_calls":
        int(
            len(
                section3barl_predict_proba_calls_df
            )
        ),

    "primary_model_expression":
        section3barl_primary_model_expression,

    "direct_primary_model_assignments":
        int(
            len(
                section3barl_primary_model_assignments_df
            )
        ),

    "runtime_objects_profiled":
        int(
            len(
                section3barl_runtime_object_profile_df
            )
        ),

    "next_stage":
        "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",
}


print()
print("SECTION 3B-A REPAIR L SUMMARY")
print("-" * 100)

for key, value in section3barl_summary.items():

    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 10. Validation checks
# --------------------------------------------------------------------------------------

section3barl_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "notebook54_source_loaded",

            "passed":
                len(
                    notebook54_code_cells_3barl
                ) > 0,

            "value":
                len(
                    notebook54_code_cells_3barl
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "scoring_cells_located",

            "passed":
                len(
                    section3barl_scoring_cell_indices
                ) > 0,

            "value":
                section3barl_scoring_cell_indices,

            "expected":
                "At least one scoring cell",
        },
        {
            "check":
                "predict_proba_call_located",

            "passed":
                len(
                    section3barl_predict_proba_calls_df
                ) > 0,

            "value":
                len(
                    section3barl_predict_proba_calls_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "primary_model_expression_resolved",

            "passed":
                bool(
                    section3barl_primary_model_expression
                ),

            "value":
                section3barl_primary_model_expression,

            "expected":
                "Nonempty expression",
        },
        {
            "check":
                "next_stage_resolved",

            "passed":
                section3barl_summary[
                    "next_stage"
                ]
                ==
                "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",

            "value":
                section3barl_summary[
                    "next_stage"
                ],

            "expected":
                "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR L VALIDATION CHECKS")
print("-" * 100)

display(
    section3barl_validation_checks_df
)


assert section3barl_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 11. Save Repair L reports
# --------------------------------------------------------------------------------------

SECTION3BARL_TERM_MATCHES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_notebook54_term_matches.csv"
)

SECTION3BARL_ASSIGNMENTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_model_preprocessor_assignments.csv"
)

SECTION3BARL_SCORING_SOURCES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_scoring_source_cells.csv"
)

SECTION3BARL_PREDICT_CALLS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_predict_proba_calls.csv"
)

SECTION3BARL_RUNTIME_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_runtime_object_profile.csv"
)

SECTION3BARL_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_validation_checks.csv"
)

SECTION3BARL_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_runtime_assignment_trace_summary.json"
)


section3barl_term_matches_df.to_csv(
    SECTION3BARL_TERM_MATCHES_FILE,
    index=False,
)

section3barl_assignment_inventory_df.to_csv(
    SECTION3BARL_ASSIGNMENTS_FILE,
    index=False,
)

section3barl_scoring_sources_df.to_csv(
    SECTION3BARL_SCORING_SOURCES_FILE,
    index=False,
)

section3barl_predict_proba_calls_df.to_csv(
    SECTION3BARL_PREDICT_CALLS_FILE,
    index=False,
)

section3barl_runtime_object_profile_df.to_csv(
    SECTION3BARL_RUNTIME_PROFILE_FILE,
    index=False,
)

section3barl_validation_checks_df.to_csv(
    SECTION3BARL_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barl_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR L REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARL_TERM_MATCHES_FILE,
    SECTION3BARL_ASSIGNMENTS_FILE,
    SECTION3BARL_SCORING_SOURCES_FILE,
    SECTION3BARL_PREDICT_CALLS_FILE,
    SECTION3BARL_RUNTIME_PROFILE_FILE,
    SECTION3BARL_VALIDATION_FILE,
    SECTION3BARL_SUMMARY_FILE,
]:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A NOTEBOOK 54 RUNTIME "
    "MODEL ASSIGNMENT TRACE PASSED"
)


# In[34]:


# ======================================================================================
# SECTION 3B-A REPAIR L — TRACE NOTEBOOK 54 RUNTIME MODEL ASSIGNMENTS
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR L — TRACE NOTEBOOK 54 RUNTIME MODEL ASSIGNMENTS")
print("=" * 100)

from pathlib import Path
import ast
import json
import nbformat
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate Notebook 54 source
# --------------------------------------------------------------------------------------

NOTEBOOK54_SOURCE_PATH_3BARL = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents"
    r"\PTCG_AI_Battle_Challenge"
    r"\notebooks"
    r"\54_tournament_strength_optimization_clean.ipynb"
)


assert NOTEBOOK54_SOURCE_PATH_3BARL.exists(), (
    "Notebook 54 clean source was not found: "
    f"{NOTEBOOK54_SOURCE_PATH_3BARL}"
)


with open(
    NOTEBOOK54_SOURCE_PATH_3BARL,
    "r",
    encoding="utf-8",
) as file:

    notebook54_document_3barl = nbformat.read(
        file,
        as_version=4,
    )


notebook54_code_cells_3barl = [
    cell
    for cell in notebook54_document_3barl.cells
    if cell.cell_type == "code"
]


print()
print("Notebook 54 source:")
print(NOTEBOOK54_SOURCE_PATH_3BARL)

print(
    "Code cells:",
    len(notebook54_code_cells_3barl),
)


# --------------------------------------------------------------------------------------
# 2. Search for the exact scoring pipeline and model/preprocessor references
# --------------------------------------------------------------------------------------

SECTION3BARL_SEARCH_TERMS = [
    "section4whb_case_probability_scores_df",
    "section4whb_case_probability_scores",
    "predict_proba",
    "notebook53_legality_model",
    "notebook53_preprocessor",
    "section4wh_model",
    "section4wh_preprocessor",
    "model_source",
    "preprocessor_source",
    "build_legality_aware_feature_row",
]


section3barl_term_match_rows = []


for code_cell_index, code_cell in enumerate(
    notebook54_code_cells_3barl
):

    source_text = str(
        code_cell.source
    )


    matched_terms = [
        search_term
        for search_term in SECTION3BARL_SEARCH_TERMS
        if search_term in source_text
    ]


    if not matched_terms:

        continue


    section3barl_term_match_rows.append(
        {
            "code_cell_index":
                int(code_cell_index),

            "matched_terms":
                json.dumps(
                    matched_terms
                ),

            "source_length":
                int(
                    len(source_text)
                ),

            "source_preview":
                source_text[:3000],
        }
    )


section3barl_term_matches_df = pd.DataFrame(
    section3barl_term_match_rows
)


print()
print("NOTEBOOK 54 MODEL/PREPROCESSOR TERM MATCHES")
print("-" * 100)

display(
    section3barl_term_matches_df
)


assert not section3barl_term_matches_df.empty, (
    "No Notebook 54 scoring references were located."
)


# --------------------------------------------------------------------------------------
# 3. Extract assignments involving model and preprocessor variables
# --------------------------------------------------------------------------------------

def section3barl_target_names(
    target_node,
):
    """
    Return assignment target names from an AST node.
    """

    discovered_names = []


    if isinstance(
        target_node,
        ast.Name,
    ):

        discovered_names.append(
            target_node.id
        )


    elif isinstance(
        target_node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):

        for child_node in target_node.elts:

            discovered_names.extend(
                section3barl_target_names(
                    child_node
                )
            )


    return discovered_names


section3barl_assignment_rows = []


for code_cell_index, code_cell in enumerate(
    notebook54_code_cells_3barl
):

    source_text = str(
        code_cell.source
    )


    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if isinstance(
            node,
            ast.Assign,
        ):

            target_names = []


            for target_node in node.targets:

                target_names.extend(
                    section3barl_target_names(
                        target_node
                    )
                )


        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            target_names = section3barl_target_names(
                node.target
            )


        else:

            continue


        assignment_source = ast.get_source_segment(
            source_text,
            node,
        )


        searchable_text = " ".join(
            [
                *target_names,
                str(assignment_source),
            ]
        ).lower()


        if not any(
            keyword in searchable_text
            for keyword in [
                "model",
                "preprocessor",
                "legality",
                "feature_names",
                "predict_proba",
                "probability",
            ]
        ):

            continue


        section3barl_assignment_rows.append(
            {
                "code_cell_index":
                    int(code_cell_index),

                "line_number":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "target_names":
                    json.dumps(
                        target_names
                    ),

                "assignment_source":
                    assignment_source,
            }
        )


section3barl_assignment_inventory_df = pd.DataFrame(
    section3barl_assignment_rows
)


print()
print("NOTEBOOK 54 MODEL/PREPROCESSOR ASSIGNMENT INVENTORY")
print("-" * 100)

display(
    section3barl_assignment_inventory_df
)


# --------------------------------------------------------------------------------------
# 4. Extract complete source cells surrounding Section 4W-H-B scoring
# --------------------------------------------------------------------------------------

section3barl_scoring_cell_indices = sorted(
    {
        int(row["code_cell_index"])
        for _, row in section3barl_term_matches_df.iterrows()
        if any(
            term in str(
                row["matched_terms"]
            )
            for term in [
                "section4whb_case_probability_scores_df",
                "section4whb_case_probability_scores",
                "predict_proba",
            ]
        )
    }
)


section3barl_scoring_source_rows = []


for scoring_cell_index in section3barl_scoring_cell_indices:

    source_text = str(
        notebook54_code_cells_3barl[
            scoring_cell_index
        ].source
    )


    section3barl_scoring_source_rows.append(
        {
            "code_cell_index":
                scoring_cell_index,

            "source_length":
                int(
                    len(source_text)
                ),

            "full_source":
                source_text,
        }
    )


section3barl_scoring_sources_df = pd.DataFrame(
    section3barl_scoring_source_rows
)


print()
print("SECTION 4W-H-B SCORING SOURCE CELLS")
print("-" * 100)

for _, source_row in (
    section3barl_scoring_sources_df.iterrows()
):

    print()
    print("=" * 100)
    print(
        "CODE-CELL INDEX:",
        source_row[
            "code_cell_index"
        ],
    )
    print("=" * 100)

    print(
        source_row[
            "full_source"
        ]
    )


# --------------------------------------------------------------------------------------
# 5. Find candidate runtime variable names used with predict_proba
# --------------------------------------------------------------------------------------

section3barl_predict_proba_rows = []


for scoring_cell_index in section3barl_scoring_cell_indices:

    source_text = str(
        notebook54_code_cells_3barl[
            scoring_cell_index
        ].source
    )


    try:

        syntax_tree = ast.parse(
            source_text
        )

    except SyntaxError:

        continue


    for node in ast.walk(
        syntax_tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue


        function_node = node.func


        if not (
            isinstance(
                function_node,
                ast.Attribute,
            )
            and
            function_node.attr
            ==
            "predict_proba"
        ):

            continue


        model_expression = ast.get_source_segment(
            source_text,
            function_node.value,
        )


        argument_expressions = [
            ast.get_source_segment(
                source_text,
                argument_node,
            )
            for argument_node in node.args
        ]


        section3barl_predict_proba_rows.append(
            {
                "code_cell_index":
                    int(
                        scoring_cell_index
                    ),

                "line_number":
                    int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    ),

                "model_expression":
                    model_expression,

                "argument_expressions":
                    json.dumps(
                        argument_expressions
                    ),

                "complete_call":
                    ast.get_source_segment(
                        source_text,
                        node,
                    ),
            }
        )


section3barl_predict_proba_calls_df = pd.DataFrame(
    section3barl_predict_proba_rows
)


print()
print("EXACT PREDICT_PROBA CALLS")
print("-" * 100)

display(
    section3barl_predict_proba_calls_df
)


assert not section3barl_predict_proba_calls_df.empty, (
    "No predict_proba call was found in Notebook 54 scoring cells."
)


# --------------------------------------------------------------------------------------
# 6. Resolve the primary model expression
# --------------------------------------------------------------------------------------

section3barl_model_expressions = (
    section3barl_predict_proba_calls_df[
        "model_expression"
    ]
    .dropna()
    .astype(str)
    .value_counts()
    .rename_axis(
        "model_expression"
    )
    .reset_index(
        name="usage_count"
    )
)


print()
print("MODEL EXPRESSIONS USED FOR NOTEBOOK 54 SCORING")
print("-" * 100)

display(
    section3barl_model_expressions
)


section3barl_primary_model_expression = str(
    section3barl_model_expressions.iloc[0][
        "model_expression"
    ]
)


# --------------------------------------------------------------------------------------
# 7. Search for the primary model expression assignment
# --------------------------------------------------------------------------------------

section3barl_primary_model_assignments_df = (
    section3barl_assignment_inventory_df.loc[
        section3barl_assignment_inventory_df[
            "target_names"
        ]
        .astype(str)
        .str.contains(
            section3barl_primary_model_expression,
            regex=False,
        )
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print("PRIMARY SCORING MODEL ASSIGNMENTS")
print("-" * 100)

if section3barl_primary_model_assignments_df.empty:

    print(
        "No direct assignment was found for:",
        section3barl_primary_model_expression,
    )

else:

    display(
        section3barl_primary_model_assignments_df
    )


# --------------------------------------------------------------------------------------
# 8. Inspect current Notebook 55 runtime objects with matching names
# --------------------------------------------------------------------------------------

section3barl_runtime_name_candidates = sorted(
    {
        "notebook53_legality_model",
        "notebook53_preprocessor",
        "legality_aware_policy_model",
        "legality_preprocessor",
        "baseline_policy_model",
        "baseline_preprocessor",
        section3barl_primary_model_expression,
    }
)


section3barl_runtime_object_rows = []


for object_name in section3barl_runtime_name_candidates:

    object_available = (
        object_name in globals()
    )


    object_value = globals().get(
        object_name
    )


    section3barl_runtime_object_rows.append(
        {
            "object_name":
                object_name,

            "available":
                object_available,

            "object_type":
                (
                    type(
                        object_value
                    ).__name__
                    if object_available
                    else "MISSING"
                ),

            "feature_count":
                (
                    getattr(
                        object_value,
                        "n_features_in_",
                        None,
                    )
                    if object_available
                    else None
                ),

            "has_predict_proba":
                bool(
                    object_available
                    and
                    hasattr(
                        object_value,
                        "predict_proba",
                    )
                ),

            "has_transform":
                bool(
                    object_available
                    and
                    hasattr(
                        object_value,
                        "transform",
                    )
                ),
        }
    )


section3barl_runtime_object_profile_df = pd.DataFrame(
    section3barl_runtime_object_rows
)


print()
print("NOTEBOOK 55 RUNTIME OBJECT PROFILE")
print("-" * 100)

display(
    section3barl_runtime_object_profile_df
)


# --------------------------------------------------------------------------------------
# 9. Summary
# --------------------------------------------------------------------------------------

section3barl_summary = {
    "status":
        "NOTEBOOK54_RUNTIME_MODEL_ASSIGNMENT_TRACE_COMPLETE",

    "matching_source_cells":
        int(
            len(
                section3barl_term_matches_df
            )
        ),

    "scoring_source_cells":
        section3barl_scoring_cell_indices,

    "predict_proba_calls":
        int(
            len(
                section3barl_predict_proba_calls_df
            )
        ),

    "primary_model_expression":
        section3barl_primary_model_expression,

    "direct_primary_model_assignments":
        int(
            len(
                section3barl_primary_model_assignments_df
            )
        ),

    "runtime_objects_profiled":
        int(
            len(
                section3barl_runtime_object_profile_df
            )
        ),

    "next_stage":
        "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",
}


print()
print("SECTION 3B-A REPAIR L SUMMARY")
print("-" * 100)

for key, value in section3barl_summary.items():

    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 10. Validation checks
# --------------------------------------------------------------------------------------

section3barl_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "notebook54_source_loaded",

            "passed":
                len(
                    notebook54_code_cells_3barl
                ) > 0,

            "value":
                len(
                    notebook54_code_cells_3barl
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "scoring_cells_located",

            "passed":
                len(
                    section3barl_scoring_cell_indices
                ) > 0,

            "value":
                section3barl_scoring_cell_indices,

            "expected":
                "At least one scoring cell",
        },
        {
            "check":
                "predict_proba_call_located",

            "passed":
                len(
                    section3barl_predict_proba_calls_df
                ) > 0,

            "value":
                len(
                    section3barl_predict_proba_calls_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "primary_model_expression_resolved",

            "passed":
                bool(
                    section3barl_primary_model_expression
                ),

            "value":
                section3barl_primary_model_expression,

            "expected":
                "Nonempty expression",
        },
        {
            "check":
                "next_stage_resolved",

            "passed":
                section3barl_summary[
                    "next_stage"
                ]
                ==
                "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",

            "value":
                section3barl_summary[
                    "next_stage"
                ],

            "expected":
                "RESTORE_EXACT_NOTEBOOK54_SCORING_RUNTIME_OBJECT",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR L VALIDATION CHECKS")
print("-" * 100)

display(
    section3barl_validation_checks_df
)


assert section3barl_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 11. Save Repair L reports
# --------------------------------------------------------------------------------------

SECTION3BARL_TERM_MATCHES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_notebook54_term_matches.csv"
)

SECTION3BARL_ASSIGNMENTS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_model_preprocessor_assignments.csv"
)

SECTION3BARL_SCORING_SOURCES_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_scoring_source_cells.csv"
)

SECTION3BARL_PREDICT_CALLS_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_predict_proba_calls.csv"
)

SECTION3BARL_RUNTIME_PROFILE_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_runtime_object_profile.csv"
)

SECTION3BARL_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_validation_checks.csv"
)

SECTION3BARL_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3barl_runtime_assignment_trace_summary.json"
)


section3barl_term_matches_df.to_csv(
    SECTION3BARL_TERM_MATCHES_FILE,
    index=False,
)

section3barl_assignment_inventory_df.to_csv(
    SECTION3BARL_ASSIGNMENTS_FILE,
    index=False,
)

section3barl_scoring_sources_df.to_csv(
    SECTION3BARL_SCORING_SOURCES_FILE,
    index=False,
)

section3barl_predict_proba_calls_df.to_csv(
    SECTION3BARL_PREDICT_CALLS_FILE,
    index=False,
)

section3barl_runtime_object_profile_df.to_csv(
    SECTION3BARL_RUNTIME_PROFILE_FILE,
    index=False,
)

section3barl_validation_checks_df.to_csv(
    SECTION3BARL_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barl_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR L REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARL_TERM_MATCHES_FILE,
    SECTION3BARL_ASSIGNMENTS_FILE,
    SECTION3BARL_SCORING_SOURCES_FILE,
    SECTION3BARL_PREDICT_CALLS_FILE,
    SECTION3BARL_RUNTIME_PROFILE_FILE,
    SECTION3BARL_VALIDATION_FILE,
    SECTION3BARL_SUMMARY_FILE,
]:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A NOTEBOOK 54 RUNTIME "
    "MODEL ASSIGNMENT TRACE PASSED"
)


# In[35]:


# ======================================================================================
# SECTION 3B-A REPAIR L — FINAL TRACE DECISION
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR L — FINAL TRACE DECISION")
print("=" * 100)

for key in [
    "status",
    "scoring_source_cells",
    "predict_proba_calls",
    "primary_model_expression",
    "direct_primary_model_assignments",
    "runtime_objects_profiled",
    "next_stage",
]:
    print(
        f"{key:68}: "
        f"{section3barl_summary.get(key)}"
    )


print()
print("EXACT PREDICT_PROBA CALLS")
print("-" * 100)

display(
    section3barl_predict_proba_calls_df
)


print()
print("PRIMARY SCORING MODEL ASSIGNMENTS")
print("-" * 100)

if section3barl_primary_model_assignments_df.empty:

    print(
        "No direct assignment was found for:",
        section3barl_primary_model_expression,
    )

else:

    display(
        section3barl_primary_model_assignments_df
    )


print()
print("RELEVANT RUNTIME OBJECTS")
print("-" * 100)

display(
    section3barl_runtime_object_profile_df.loc[
        section3barl_runtime_object_profile_df[
            "available"
        ].astype(bool)
    ].reset_index(drop=True)
)


# In[36]:


print("=" * 100)
print("SECTION 3B-A REPAIR M — SHOW NOTEBOOK 54 CELL 66")
print("=" * 100)

import nbformat
from pathlib import Path

nb = nbformat.read(
    NOTEBOOK54_SOURCE_PATH_3BARL,
    as_version=4,
)

code_cells = [
    c
    for c in nb.cells
    if c.cell_type == "code"
]

CELL_INDEX = 66

print(f"Notebook 54 code cell {CELL_INDEX}")
print("-" * 100)
print(code_cells[CELL_INDEX].source)


# In[37]:


# ======================================================================================
# SECTION 3B-A REPAIR N — RECOVER NOTEBOOK 54 RESOLVED RUNTIME OBJECTS
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR N — RECOVER NOTEBOOK 54 RESOLVED RUNTIME OBJECTS")
print("=" * 100)

from pathlib import Path
import json
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Locate the reports created by Notebook 54 Cell 66
# --------------------------------------------------------------------------------------

SECTION3BARN_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "notebook54"
    / "section4"
)


SECTION3BARN_RUNTIME_INVENTORY_FILE = (
    SECTION3BARN_REPORT_DIRECTORY
    / "section4wha_runtime_object_inventory.csv"
)


SECTION3BARN_MODEL_INTERFACE_FILE = (
    SECTION3BARN_REPORT_DIRECTORY
    / "section4wha_model_interface_profile.csv"
)


SECTION3BARN_SUMMARY_FILE = (
    SECTION3BARN_REPORT_DIRECTORY
    / "section4wha_root_cause_preflight_summary.json"
)


for required_file in [
    SECTION3BARN_RUNTIME_INVENTORY_FILE,
    SECTION3BARN_MODEL_INTERFACE_FILE,
    SECTION3BARN_SUMMARY_FILE,
]:

    assert required_file.exists(), (
        f"Required Notebook 54 report was not found: {required_file}"
    )

    assert required_file.stat().st_size > 0


# --------------------------------------------------------------------------------------
# 2. Load the exact original Notebook 54 evidence
# --------------------------------------------------------------------------------------

section3barn_runtime_inventory_df = pd.read_csv(
    SECTION3BARN_RUNTIME_INVENTORY_FILE
)


section3barn_model_interface_df = pd.read_csv(
    SECTION3BARN_MODEL_INTERFACE_FILE
)


with open(
    SECTION3BARN_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    section3barn_notebook54_summary = json.load(
        file
    )


# --------------------------------------------------------------------------------------
# 3. Display the runtime inventory exactly as Notebook 54 saved it
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 54 ORIGINAL RUNTIME OBJECT INVENTORY")
print("-" * 100)

display(
    section3barn_runtime_inventory_df
)


print()
print("NOTEBOOK 54 ORIGINAL MODEL INTERFACE")
print("-" * 100)

display(
    section3barn_model_interface_df
)


# --------------------------------------------------------------------------------------
# 4. Recover the exact resolved model and preprocessor sources
# --------------------------------------------------------------------------------------

section3barn_model_source = str(
    section3barn_notebook54_summary.get(
        "model_source",
        "",
    )
)


section3barn_preprocessor_source = str(
    section3barn_notebook54_summary.get(
        "preprocessor_source",
        "",
    )
)


section3barn_model_type = str(
    section3barn_notebook54_summary.get(
        "model_type",
        "",
    )
)


section3barn_preprocessor_available = bool(
    section3barn_notebook54_summary.get(
        "preprocessor_available",
        False,
    )
)


print()
print("NOTEBOOK 54 ORIGINAL RESOLUTION DECISION")
print("-" * 100)

print(
    f"{'model_source':56}: "
    f"{section3barn_model_source}"
)

print(
    f"{'model_type':56}: "
    f"{section3barn_model_type}"
)

print(
    f"{'preprocessor_source':56}: "
    f"{section3barn_preprocessor_source}"
)

print(
    f"{'preprocessor_available':56}: "
    f"{section3barn_preprocessor_available}"
)


assert section3barn_model_source, (
    "Notebook 54 did not save a resolved model source."
)


assert section3barn_preprocessor_source, (
    "Notebook 54 did not save a resolved preprocessor source."
)


# --------------------------------------------------------------------------------------
# 5. Resolve those exact source expressions in the Notebook 55 runtime
# --------------------------------------------------------------------------------------

def section3barn_resolve_source_expression(
    source_expression,
):
    """
    Resolve simple source expressions such as:

        globals.notebook53_legality_model
        notebook54_legality_policy.model
    """

    source_expression = str(
        source_expression
    ).strip()


    if source_expression.startswith(
        "globals."
    ):

        object_name = source_expression.split(
            ".",
            1,
        )[1]

        return (
            object_name,
            globals().get(
                object_name
            ),
        )


    if source_expression.startswith(
        "notebook54_legality_policy."
    ):

        attribute_name = source_expression.split(
            ".",
            1,
        )[1]

        policy_object = globals().get(
            "notebook54_legality_policy"
        )

        return (
            source_expression,
            getattr(
                policy_object,
                attribute_name,
                None,
            )
            if policy_object is not None
            else None,
        )


    return (
        source_expression,
        globals().get(
            source_expression
        ),
    )


(
    section3barn_model_runtime_name,
    section3barn_exact_model_object,
) = section3barn_resolve_source_expression(
    section3barn_model_source
)


(
    section3barn_preprocessor_runtime_name,
    section3barn_exact_preprocessor_object,
) = section3barn_resolve_source_expression(
    section3barn_preprocessor_source
)


print()
print("NOTEBOOK 55 MATCHING RUNTIME OBJECTS")
print("-" * 100)

print(
    f"{'model runtime name':56}: "
    f"{section3barn_model_runtime_name}"
)

print(
    f"{'model available':56}: "
    f"{section3barn_exact_model_object is not None}"
)

print(
    f"{'model runtime type':56}: "
    f"{type(section3barn_exact_model_object).__name__ if section3barn_exact_model_object is not None else 'MISSING'}"
)

print(
    f"{'preprocessor runtime name':56}: "
    f"{section3barn_preprocessor_runtime_name}"
)

print(
    f"{'preprocessor available':56}: "
    f"{section3barn_exact_preprocessor_object is not None}"
)

print(
    f"{'preprocessor runtime type':56}: "
    f"{type(section3barn_exact_preprocessor_object).__name__ if section3barn_exact_preprocessor_object is not None else 'MISSING'}"
)


# --------------------------------------------------------------------------------------
# 6. Determine the next exact recovery route
# --------------------------------------------------------------------------------------

if (
    section3barn_exact_model_object is not None
    and
    section3barn_exact_preprocessor_object is not None
):

    section3barn_status = (
        "ORIGINAL_NOTEBOOK54_RUNTIME_OBJECTS_AVAILABLE"
    )

    section3barn_next_stage = (
        "SCORE_ORIGINAL_RESOLVED_RUNTIME_OBJECTS"
    )

else:

    missing_runtime_bindings = []

    if section3barn_exact_model_object is None:

        missing_runtime_bindings.append(
            section3barn_model_source
        )

    if section3barn_exact_preprocessor_object is None:

        missing_runtime_bindings.append(
            section3barn_preprocessor_source
        )


    section3barn_status = (
        "ORIGINAL_NOTEBOOK54_RUNTIME_BINDINGS_REQUIRE_RESTORATION"
    )

    section3barn_next_stage = (
        "RESTORE_MISSING_ORIGINAL_RUNTIME_BINDINGS"
    )


section3barn_summary = {
    "status":
        section3barn_status,

    "original_model_source":
        section3barn_model_source,

    "original_model_type":
        section3barn_model_type,

    "original_preprocessor_source":
        section3barn_preprocessor_source,

    "matching_model_available":
        section3barn_exact_model_object
        is not None,

    "matching_preprocessor_available":
        section3barn_exact_preprocessor_object
        is not None,

    "next_stage":
        section3barn_next_stage,
}


print()
print("SECTION 3B-A REPAIR N SUMMARY")
print("-" * 100)

for key, value in section3barn_summary.items():

    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 7. Validation checks
# --------------------------------------------------------------------------------------

section3barn_validation_checks_df = pd.DataFrame(
    [
        {
            "check":
                "runtime_inventory_loaded",

            "passed":
                not section3barn_runtime_inventory_df.empty,

            "value":
                len(
                    section3barn_runtime_inventory_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "model_interface_loaded",

            "passed":
                not section3barn_model_interface_df.empty,

            "value":
                len(
                    section3barn_model_interface_df
                ),

            "expected":
                "> 0",
        },
        {
            "check":
                "original_model_source_recovered",

            "passed":
                bool(
                    section3barn_model_source
                ),

            "value":
                section3barn_model_source,

            "expected":
                "Nonempty source",
        },
        {
            "check":
                "original_preprocessor_source_recovered",

            "passed":
                bool(
                    section3barn_preprocessor_source
                ),

            "value":
                section3barn_preprocessor_source,

            "expected":
                "Nonempty source",
        },
        {
            "check":
                "next_stage_resolved",

            "passed":
                section3barn_next_stage
                in {
                    "SCORE_ORIGINAL_RESOLVED_RUNTIME_OBJECTS",
                    "RESTORE_MISSING_ORIGINAL_RUNTIME_BINDINGS",
                },

            "value":
                section3barn_next_stage,

            "expected":
                "Recognized recovery route",
        },
    ]
)


print()
print("SECTION 3B-A REPAIR N VALIDATION CHECKS")
print("-" * 100)

display(
    section3barn_validation_checks_df
)


assert section3barn_validation_checks_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 8. Save Repair N reports
# --------------------------------------------------------------------------------------

SECTION3BARN_RUNTIME_COPY_FILE = (
    REPORTS_DIRECTORY
    / "section3barn_original_runtime_inventory.csv"
)


SECTION3BARN_INTERFACE_COPY_FILE = (
    REPORTS_DIRECTORY
    / "section3barn_original_model_interface.csv"
)


SECTION3BARN_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    / "section3barn_validation_checks.csv"
)


SECTION3BARN_SUMMARY_OUTPUT_FILE = (
    REPORTS_DIRECTORY
    / "section3barn_runtime_resolution_summary.json"
)


section3barn_runtime_inventory_df.to_csv(
    SECTION3BARN_RUNTIME_COPY_FILE,
    index=False,
)


section3barn_model_interface_df.to_csv(
    SECTION3BARN_INTERFACE_COPY_FILE,
    index=False,
)


section3barn_validation_checks_df.to_csv(
    SECTION3BARN_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BARN_SUMMARY_OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3barn_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("SAVED SECTION 3B-A REPAIR N REPORTS")
print("-" * 100)

for file_path in [
    SECTION3BARN_RUNTIME_COPY_FILE,
    SECTION3BARN_INTERFACE_COPY_FILE,
    SECTION3BARN_VALIDATION_FILE,
    SECTION3BARN_SUMMARY_OUTPUT_FILE,
]:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 3B-A ORIGINAL NOTEBOOK 54 "
    "RUNTIME RESOLUTION RECOVERED"
)


# In[38]:


# =====================================================================================
# SECTION 3B-A REPAIR O — RESTORE NOTEBOOK 53 ORIGINAL RUNTIME BINDINGS
# =====================================================================================

print("=" * 100)
print("SECTION 3B-A REPAIR O — RESTORE ORIGINAL NOTEBOOK 53 RUNTIME BINDINGS")
print("=" * 100)

binding_candidates = {
    "notebook53_legality_model": [
        "legality_aware_policy_model",
        "baseline_policy_model",
        "legality_aware_model",
        "trained_model",
        "policy_model",
    ],

    "notebook53_preprocessor": [
        "legality_preprocessor",
        "baseline_preprocessor",
        "preprocessor",
    ],

    "notebook53_encoded_feature_names": [
        "baseline_encoded_feature_names",
        "encoded_feature_names",
    ],
}

binding_results = []

for target_name, candidates in binding_candidates.items():

    restored = False

    for candidate in candidates:

        if candidate in globals():

            globals()[target_name] = globals()[candidate]

            binding_results.append({
                "runtime_name": target_name,
                "bound_to": candidate,
                "restored": True,
            })

            restored = True
            break

    if not restored:

        binding_results.append({
            "runtime_name": target_name,
            "bound_to": None,
            "restored": False,
        })

import pandas as pd

section3baro_runtime_bindings_df = pd.DataFrame(binding_results)

display(section3baro_runtime_bindings_df)

assert section3baro_runtime_bindings_df["restored"].all(), \
    "Some original Notebook 53 runtime bindings could not be restored."

print()
print("✅ ORIGINAL NOTEBOOK 53 RUNTIME NAMES RESTORED")


# In[41]:


# ======================================================================================
# SECTION 3B-A CLOSURE — FREEZE CONTROLLED EVALUATION CONTRACT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A CLOSURE — FREEZE CONTROLLED EVALUATION CONTRACT")
print("=" * 100)

import json
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate the evidence supporting closure
# --------------------------------------------------------------------------------------

required_objects = [
    "section3a_summary",
    "section3bard_summary",
    "section3bark_summary",
    "NOTEBOOK55_TRAINED_MODELS",
    "section3barh_original_feature_builder",
    "REPORTS_DIRECTORY",
]

missing_objects = [
    name
    for name in required_objects
    if name not in globals()
]

assert not missing_objects, (
    "Section 3B-A closure dependencies are missing: "
    f"{missing_objects}"
)


assert section3a_summary[
    "r0_prediction_agreement_with_original"
] == 1.0


assert section3a_summary[
    "r0_accuracy_difference_from_original"
] == 0.0


assert section3bard_summary[
    "states_reconstructed"
] == 184


assert section3bard_summary[
    "state_errors"
] == 0


assert section3bard_summary[
    "prediction_agreement_rate"
] == 1.0


assert section3bark_summary[
    "best_prediction_agreement_rate"
] == 1.0


assert len(
    NOTEBOOK55_TRAINED_MODELS
) == 4


# --------------------------------------------------------------------------------------
# 2. Restore the training-compatible feature builder
# --------------------------------------------------------------------------------------
# The fitted categorical encoder learned lowercase legal-move signatures.
# All R0–R3 models must therefore receive the same canonical training-compatible
# representation during the controlled comparison.

build_legality_aware_feature_row = (
    section3barh_original_feature_builder
)


test_feature_df = build_legality_aware_feature_row(
    battle_state=section3barh_test_state,
    legal_moves=section3barh_test_legal_moves,
    side_mode="PRESERVE",
)


canonical_signature = str(
    test_feature_df.iloc[0][
        "legal_move_signature"
    ]
)


print()
print("CANONICAL EVALUATION SIGNATURE")
print("-" * 100)
print(canonical_signature)


assert canonical_signature == "ascension | quick attack"


# --------------------------------------------------------------------------------------
# 3. Freeze the evaluation contract
# --------------------------------------------------------------------------------------

section3ba_closure_summary = {
    "status":
        "CONTROLLED_EVALUATION_CONTRACT_FROZEN",

    "balanced_evaluation_cases":
        184,

    "models_to_compare":
        4,

    "r0_prediction_agreement_with_original":
        float(
            section3a_summary[
                "r0_prediction_agreement_with_original"
            ]
        ),

    "r0_accuracy_difference_from_original":
        float(
            section3a_summary[
                "r0_accuracy_difference_from_original"
            ]
        ),

    "notebook54_reconstructed_prediction_agreement":
        float(
            section3bard_summary[
                "prediction_agreement_rate"
            ]
        ),

    "notebook54_exact_probability_reconstruction":
        False,

    "historical_probability_drift_documented":
        True,

    "exact_historical_estimator_snapshot_available":
        False,

    "canonical_legal_move_signature":
        canonical_signature,

    "evaluation_representation":
        "TRAINING_COMPATIBLE_CANONICAL_FEATURES",

    "comparison_rule":
        (
            "Score R0, R1, R2, and R3 on the same 184 states using "
            "the same reconstructed raw rows, strategy-specific feature "
            "subsets, and each strategy's frozen trained estimator."
        ),

    "interpretation_rule":
        (
            "Use R0 from Notebook 55 as the controlled comparison baseline. "
            "Notebook 54 remains historical behavioral evidence; its exact "
            "probability calibration is not treated as reproducible because "
            "the original runtime estimator snapshot was not preserved."
        ),

    "next_stage":
        "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
}


print()
print("SECTION 3B-A CLOSURE SUMMARY")
print("-" * 100)

for key, value in section3ba_closure_summary.items():
    print(
        f"{key:68}: {value}"
    )


# --------------------------------------------------------------------------------------
# 4. Validation checks
# --------------------------------------------------------------------------------------

section3ba_closure_validation_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_states_available",

            "passed":
                section3bard_summary[
                    "states_reconstructed"
                ] == 184,

            "value":
                section3bard_summary[
                    "states_reconstructed"
                ],

            "expected":
                184,
        },
        {
            "check":
                "no_state_errors",

            "passed":
                section3bard_summary[
                    "state_errors"
                ] == 0,

            "value":
                section3bard_summary[
                    "state_errors"
                ],

            "expected":
                0,
        },
        {
            "check":
                "historical_predictions_reproduced",

            "passed":
                section3bard_summary[
                    "prediction_agreement_rate"
                ] == 1.0,

            "value":
                section3bard_summary[
                    "prediction_agreement_rate"
                ],

            "expected":
                1.0,
        },
        {
            "check":
                "r0_frozen_baseline_reproduced",

            "passed":
                (
                    section3a_summary[
                        "r0_prediction_agreement_with_original"
                    ] == 1.0
                    and
                    section3a_summary[
                        "r0_accuracy_difference_from_original"
                    ] == 0.0
                ),

            "value":
                {
                    "prediction_agreement":
                        section3a_summary[
                            "r0_prediction_agreement_with_original"
                        ],

                    "accuracy_difference":
                        section3a_summary[
                            "r0_accuracy_difference_from_original"
                        ],
                },

            "expected":
                {
                    "prediction_agreement":
                        1.0,

                    "accuracy_difference":
                        0.0,
                },
        },
        {
            "check":
                "four_models_ready",

            "passed":
                len(
                    NOTEBOOK55_TRAINED_MODELS
                ) == 4,

            "value":
                len(
                    NOTEBOOK55_TRAINED_MODELS
                ),

            "expected":
                4,
        },
        {
            "check":
                "training_compatible_signature_restored",

            "passed":
                canonical_signature
                ==
                "ascension | quick attack",

            "value":
                canonical_signature,

            "expected":
                "ascension | quick attack",
        },
        {
            "check":
                "next_stage_ready",

            "passed":
                section3ba_closure_summary[
                    "next_stage"
                ]
                ==
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",

            "value":
                section3ba_closure_summary[
                    "next_stage"
                ],

            "expected":
                "SCORE_ALL_FOUR_MODELS_ON_184_BALANCED_CASES",
        },
    ]
)


print()
print("SECTION 3B-A CLOSURE VALIDATION")
print("-" * 100)

display(
    section3ba_closure_validation_df
)


assert section3ba_closure_validation_df[
    "passed"
].astype(bool).all()


# --------------------------------------------------------------------------------------
# 5. Save the closure evidence
# --------------------------------------------------------------------------------------

SECTION3BA_CLOSURE_VALIDATION_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_closure_validation.csv"
)

SECTION3BA_CLOSURE_SUMMARY_FILE = (
    REPORTS_DIRECTORY
    /
    "section3ba_controlled_evaluation_contract.json"
)


section3ba_closure_validation_df.to_csv(
    SECTION3BA_CLOSURE_VALIDATION_FILE,
    index=False,
)


with open(
    SECTION3BA_CLOSURE_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section3ba_closure_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


assert SECTION3BA_CLOSURE_VALIDATION_FILE.exists()
assert SECTION3BA_CLOSURE_SUMMARY_FILE.exists()


print()
print("SAVED SECTION 3B-A CLOSURE REPORTS")
print("-" * 100)

print(
    SECTION3BA_CLOSURE_VALIDATION_FILE
)

print(
    SECTION3BA_CLOSURE_SUMMARY_FILE
)


print()
print(
    "✅ SECTION 3B-A CONTROLLED EVALUATION "
    "CONTRACT FROZEN — READY FOR SECTION 3B-B"
)


# In[ ]:




