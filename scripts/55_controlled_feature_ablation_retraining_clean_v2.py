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

# In[5]:


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

# In[6]:


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

# In[7]:


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

# In[8]:


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

# In[9]:


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

# In[10]:


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

# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# # Section 3B-A Closure — Controlled Evaluation Contract
# 
# The exploratory Notebook 54 probability-reconstruction repair branch was removed from this clean version.
# 
# The historical Notebook 54 action selections remain useful behavioral evidence, but its exact runtime estimator snapshot was not preserved. Notebook 55 therefore freezes its own reproduced R0 estimator as the controlled baseline for comparing R1, R2, and R3 on the same 184 balanced cases.
# 

# In[16]:


# ======================================================================================
# SECTION 3B-A CLOSURE — FREEZE CONTROLLED EVALUATION CONTRACT
# ======================================================================================

print("=" * 100)
print("SECTION 3B-A CLOSURE — FREEZE CONTROLLED EVALUATION CONTRACT")
print("=" * 100)

import json
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Validate stable Notebook 55 evidence
# --------------------------------------------------------------------------------------

SECTION3BA_CLOSURE_REQUIRED_OBJECTS = [
    "section3a_summary",
    "section3ba_summary",
    "section3ba_score_cases_df",
    "NOTEBOOK55_TRAINED_MODELS",
    "REPORTS_DIRECTORY",
]

section3ba_closure_missing_objects = [
    object_name
    for object_name in SECTION3BA_CLOSURE_REQUIRED_OBJECTS
    if object_name not in globals()
]

assert not section3ba_closure_missing_objects, (
    "Section 3B-A closure dependencies are missing: "
    f"{section3ba_closure_missing_objects}"
)

assert section3a_summary[
    "r0_prediction_agreement_with_original"
] == 1.0

assert section3a_summary[
    "r0_accuracy_difference_from_original"
] == 0.0

assert len(
    NOTEBOOK55_TRAINED_MODELS
) == 4

assert len(
    section3ba_score_cases_df
) == 184

assert section3ba_score_cases_df[
    "comparison_case_id"
].nunique() == 184


# --------------------------------------------------------------------------------------
# 2. Freeze the training-compatible representation
# --------------------------------------------------------------------------------------

canonical_legal_move_signature = (
    "ascension | quick attack"
)


# --------------------------------------------------------------------------------------
# 3. Freeze the controlled evaluation contract
# --------------------------------------------------------------------------------------

section3ba_closure_summary = {
    "status":
        "CONTROLLED_EVALUATION_CONTRACT_FROZEN",

    "balanced_evaluation_cases":
        int(
            len(
                section3ba_score_cases_df
            )
        ),

    "unique_comparison_cases":
        int(
            section3ba_score_cases_df[
                "comparison_case_id"
            ].nunique()
        ),

    "models_to_compare":
        int(
            len(
                NOTEBOOK55_TRAINED_MODELS
            )
        ),

    "model_strategies":
        sorted(
            NOTEBOOK55_TRAINED_MODELS.keys()
        ),

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

    "historical_notebook54_probability_reconstruction_required":
        False,

    "historical_probability_drift_documented":
        True,

    "exact_historical_estimator_snapshot_available":
        False,

    "canonical_legal_move_signature":
        canonical_legal_move_signature,

    "evaluation_representation":
        "TRAINING_COMPATIBLE_CANONICAL_FEATURES",

    "comparison_rule":
        (
            "Score R0, R1, R2, and R3 on the same 184 balanced cases "
            "using each strategy's frozen estimator and feature subset."
        ),

    "interpretation_rule":
        (
            "Use Notebook 55 R0 as the controlled comparison baseline. "
            "Notebook 54 remains historical behavioral evidence; exact "
            "historical probability calibration is not required because "
            "the original runtime estimator snapshot was not preserved."
        ),

    "next_stage":
        "NOTEBOOK_56_FOUR_MODEL_CONTROLLED_POLICY_EVALUATION",
}


print()
print("SECTION 3B-A CLOSURE SUMMARY")
print("-" * 100)

for key, value in section3ba_closure_summary.items():

    print(
        f"{key:72}: {value}"
    )


# --------------------------------------------------------------------------------------
# 4. Validation checks
# --------------------------------------------------------------------------------------

section3ba_closure_validation_df = pd.DataFrame(
    [
        {
            "check":
                "all_184_cases_available",

            "passed":
                len(
                    section3ba_score_cases_df
                ) == 184,

            "value":
                len(
                    section3ba_score_cases_df
                ),

            "expected":
                184,
        },
        {
            "check":
                "all_case_ids_unique",

            "passed":
                section3ba_score_cases_df[
                    "comparison_case_id"
                ].nunique() == 184,

            "value":
                section3ba_score_cases_df[
                    "comparison_case_id"
                ].nunique(),

            "expected":
                184,
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
                "canonical_signature_frozen",

            "passed":
                canonical_legal_move_signature
                ==
                "ascension | quick attack",

            "value":
                canonical_legal_move_signature,

            "expected":
                "ascension | quick attack",
        },
        {
            "check":
                "next_notebook_resolved",

            "passed":
                section3ba_closure_summary[
                    "next_stage"
                ]
                ==
                "NOTEBOOK_56_FOUR_MODEL_CONTROLLED_POLICY_EVALUATION",

            "value":
                section3ba_closure_summary[
                    "next_stage"
                ],

            "expected":
                "NOTEBOOK_56_FOUR_MODEL_CONTROLLED_POLICY_EVALUATION",
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
# 5. Save closure reports
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
assert SECTION3BA_CLOSURE_VALIDATION_FILE.stat().st_size > 0
assert SECTION3BA_CLOSURE_SUMMARY_FILE.exists()
assert SECTION3BA_CLOSURE_SUMMARY_FILE.stat().st_size > 0


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
    "✅ NOTEBOOK 55 CONTROLLED EVALUATION CONTRACT "
    "FROZEN — READY FOR NOTEBOOK 56"
)

