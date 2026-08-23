from __future__ import annotations

#!/usr/bin/env python
# coding: utf-8

# # Notebook 48 - Tournament Evaluation and Elo Benchmarking
# 
# ## Objective
# 
# Evaluate the trained competitive policy through repeated simulator battles.
# 
# ### Main goals
# 
# - Compare the competitive policy with baseline agents
# - Run side-swapped matches
# - Measure win rate and average battle length
# - Track legal-action reliability
# - Calculate Elo ratings
# - Produce a final competitive-strength report
# - Identify whether the trained policy should replace the earlier bootstrap agent
# 

# ## Section 1 â€” Tournament Setup and Artifact Validation
# 
# #### Verify the trained competitive-policy checkpoint, production agent modules, simulator components, tournament modules, and Notebook 47 evaluation artifacts.
# 
# #### This section creates the output directories and establishes deterministic benchmarking configuration.

# In[1]:


# ============================================================
# NOTEBOOK 48
# SECTION 1 â€” TOURNAMENT SETUP AND ARTIFACT VALIDATION
# ============================================================


import inspect
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 1.3 â€” Locate project root
# ------------------------------------------------------------

def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current = (
        start_path or Path.cwd()
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

NOTEBOOK48_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook48"
)

TOURNAMENT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "tournaments"
)

TOURNAMENT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "tournament"
)

for directory in [
    NOTEBOOK48_REPORT_DIR,
    TOURNAMENT_DATA_DIR,
    TOURNAMENT_MODEL_DIR,
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
# 1.4 â€” Reproducibility and device
# ------------------------------------------------------------

RANDOM_SEED = 4801

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
# 1.5 â€” Required artifacts
# ------------------------------------------------------------

BEST_POLICY_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "competitive"
    / "competitive_policy_best.pt"
)

LAST_POLICY_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "competitive"
    / "competitive_policy_last.pt"
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

COMPETITIVE_PACKAGE_FILE = (
    SRC_DIR
    / "competitive"
    / "__init__.py"
)

BATTLE_SIMULATION_FILE = (
    SRC_DIR
    / "battle_simulation.py"
)

BATTLE_STATE_FILE = (
    SRC_DIR
    / "battle_state.py"
)

SIMULATOR_FILE = (
    SRC_DIR
    / "simulator.py"
)

LEGAL_MOVES_FILE = (
    SRC_DIR
    / "legal_moves.py"
)

NOTEBOOK47_TEST_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "notebook47"
    / "competitive_policy_test_evaluation.json"
)

NOTEBOOK47_BATTLE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "notebook47"
    / "competitive_policy_battle_summary.json"
)

NOTEBOOK47_TRAINING_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "notebook47"
    / "competitive_policy_training_summary.json"
)


# ------------------------------------------------------------
# 1.6 â€” Candidate tournament modules
# ------------------------------------------------------------

TOURNAMENT_CANDIDATES = [
    SRC_DIR / "tournament.py",
    SRC_DIR / "tournament_runner.py",
    SRC_DIR / "self_play.py",
    SRC_DIR / "battle_agent.py",
    SRC_DIR / "agents",
    SRC_DIR / "engine",
]


# ------------------------------------------------------------
# 1.7 â€” Artifact inventory
# ------------------------------------------------------------

artifact_rows = [
    {
        "Artifact":
            "Best competitive checkpoint",

        "Path":
            str(
                BEST_POLICY_CHECKPOINT
            ),

        "Exists":
            BEST_POLICY_CHECKPOINT.exists(),
    },

    {
        "Artifact":
            "Last competitive checkpoint",

        "Path":
            str(
                LAST_POLICY_CHECKPOINT
            ),

        "Exists":
            LAST_POLICY_CHECKPOINT.exists(),
    },

    {
        "Artifact":
            "Competitive policy engine",

        "Path":
            str(
                POLICY_ENGINE_FILE
            ),

        "Exists":
            POLICY_ENGINE_FILE.exists(),
    },

    {
        "Artifact":
            "Competitive policy agent",

        "Path":
            str(
                POLICY_AGENT_FILE
            ),

        "Exists":
            POLICY_AGENT_FILE.exists(),
    },

    {
        "Artifact":
            "Competitive package initializer",

        "Path":
            str(
                COMPETITIVE_PACKAGE_FILE
            ),

        "Exists":
            COMPETITIVE_PACKAGE_FILE.exists(),
    },

    {
        "Artifact":
            "Battle simulation module",

        "Path":
            str(
                BATTLE_SIMULATION_FILE
            ),

        "Exists":
            BATTLE_SIMULATION_FILE.exists(),
    },

    {
        "Artifact":
            "Battle-state module",

        "Path":
            str(
                BATTLE_STATE_FILE
            ),

        "Exists":
            BATTLE_STATE_FILE.exists(),
    },

    {
        "Artifact":
            "Simulator transition module",

        "Path":
            str(
                SIMULATOR_FILE
            ),

        "Exists":
            SIMULATOR_FILE.exists(),
    },

    {
        "Artifact":
            "Legal-move module",

        "Path":
            str(
                LEGAL_MOVES_FILE
            ),

        "Exists":
            LEGAL_MOVES_FILE.exists(),
    },

    {
        "Artifact":
            "Notebook 47 test report",

        "Path":
            str(
                NOTEBOOK47_TEST_REPORT
            ),

        "Exists":
            NOTEBOOK47_TEST_REPORT.exists(),
    },

    {
        "Artifact":
            "Notebook 47 battle summary",

        "Path":
            str(
                NOTEBOOK47_BATTLE_SUMMARY
            ),

        "Exists":
            NOTEBOOK47_BATTLE_SUMMARY.exists(),
    },

    {
        "Artifact":
            "Notebook 47 training summary",

        "Path":
            str(
                NOTEBOOK47_TRAINING_SUMMARY
            ),

        "Exists":
            NOTEBOOK47_TRAINING_SUMMARY.exists(),
    },
]

artifact_df = pd.DataFrame(
    artifact_rows
)


# ------------------------------------------------------------
# 1.8 â€” Load best checkpoint metadata
# ------------------------------------------------------------

assert BEST_POLICY_CHECKPOINT.exists(), (
    "The best competitive-policy checkpoint "
    "was not found."
)

best_checkpoint = torch.load(
    BEST_POLICY_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

checkpoint_configuration = (
    best_checkpoint.get(
        "configuration",
        {},
    )
)

checkpoint_epoch = int(
    best_checkpoint.get(
        "epoch",
        0,
    )
)

checkpoint_validation_metrics = (
    best_checkpoint.get(
        "validation_metrics",
        {},
    )
)

checkpoint_multi_action_metrics = (
    best_checkpoint.get(
        "multi_action_metrics",
        {},
    )
)


# ------------------------------------------------------------
# 1.9 â€” Load Notebook 47 reports
# ------------------------------------------------------------

def load_json_file(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


notebook47_test_report = (
    load_json_file(
        NOTEBOOK47_TEST_REPORT
    )
)

notebook47_battle_summary = (
    load_json_file(
        NOTEBOOK47_BATTLE_SUMMARY
    )
)

notebook47_training_summary = (
    load_json_file(
        NOTEBOOK47_TRAINING_SUMMARY
    )
)


# ------------------------------------------------------------
# 1.10 â€” Tournament configuration
# ------------------------------------------------------------

INITIAL_MATCHES_PER_PAIRING = 100

MAX_TURNS_PER_MATCH = 50

SIMULATOR_SEARCH_DEPTH = 6

INITIAL_ELO = 1000.0

ELO_K_FACTOR = 32.0

SIDE_SWAP_ENABLED = True

VERBOSE_MATCHES = False


# ------------------------------------------------------------
# 1.11 â€” Discover available tournament paths
# ------------------------------------------------------------

tournament_candidate_rows = []

for candidate in TOURNAMENT_CANDIDATES:
    tournament_candidate_rows.append(
        {
            "Candidate":
                candidate.name,

            "Path":
                str(candidate),

            "Exists":
                candidate.exists(),

            "Type": (
                "directory"
                if candidate.is_dir()
                else "file"
            ),
        }
    )

tournament_candidate_df = pd.DataFrame(
    tournament_candidate_rows
)


# ------------------------------------------------------------
# 1.12 â€” Save setup report
# ------------------------------------------------------------

SETUP_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "tournament_setup_summary.json"
)

setup_report = {
    "project_root":
        str(PROJECT_ROOT),

    "device":
        str(DEVICE),

    "random_seed":
        RANDOM_SEED,

    "checkpoint":
        str(
            BEST_POLICY_CHECKPOINT
        ),

    "checkpoint_epoch":
        checkpoint_epoch,

    "checkpoint_configuration":
        checkpoint_configuration,

    "checkpoint_validation_metrics":
        checkpoint_validation_metrics,

    "checkpoint_multi_action_metrics":
        checkpoint_multi_action_metrics,

    "notebook47_test_accuracy":
        notebook47_test_report.get(
            "test_top1_accuracy"
        ),

    "notebook47_multi_action_accuracy":
        notebook47_test_report.get(
            "test_multi_action_accuracy"
        ),

    "notebook47_battle_winner":
        notebook47_battle_summary.get(
            "winner"
        ),

    "matches_per_pairing":
        INITIAL_MATCHES_PER_PAIRING,

    "max_turns_per_match":
        MAX_TURNS_PER_MATCH,

    "simulator_search_depth":
        SIMULATOR_SEARCH_DEPTH,

    "initial_elo":
        INITIAL_ELO,

    "elo_k_factor":
        ELO_K_FACTOR,

    "side_swap_enabled":
        SIDE_SWAP_ENABLED,

    "verbose_matches":
        VERBOSE_MATCHES,

    "all_required_artifacts_exist":
        bool(
            artifact_df[
                "Exists"
            ].all()
        ),
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
# 1.13 â€” Display results
# ------------------------------------------------------------

print(
    "NOTEBOOK 48 â€” TOURNAMENT SETUP"
)

print("=" * 100)

print()
print(
    "Project root          :",
    PROJECT_ROOT,
)

print(
    "Device                :",
    DEVICE,
)

print(
    "Random seed           :",
    RANDOM_SEED,
)

print()
print(
    "Best checkpoint epoch :",
    checkpoint_epoch,
)

print(
    "Validation accuracy   :",
    checkpoint_validation_metrics.get(
        "top1_accuracy"
    ),
)

print(
    "Multi-action accuracy :",
    checkpoint_multi_action_metrics.get(
        "accuracy"
    ),
)

print(
    "Notebook 47 test acc  :",
    notebook47_test_report.get(
        "test_top1_accuracy"
    ),
)

print(
    "Notebook 47 multi acc :",
    notebook47_test_report.get(
        "test_multi_action_accuracy"
    ),
)

print()
print(
    "Matches per pairing   :",
    INITIAL_MATCHES_PER_PAIRING,
)

print(
    "Maximum turns         :",
    MAX_TURNS_PER_MATCH,
)

print(
    "Simulator depth       :",
    SIMULATOR_SEARCH_DEPTH,
)

print(
    "Initial Elo           :",
    INITIAL_ELO,
)

print(
    "Elo K-factor          :",
    ELO_K_FACTOR,
)

print()
print("REQUIRED ARTIFACTS")
print("-" * 100)

display(
    artifact_df
)

print()
print("TOURNAMENT MODULE CANDIDATES")
print("-" * 100)

display(
    tournament_candidate_df
)

print()
print(
    "Setup report:",
    SETUP_REPORT_FILE,
)


# ------------------------------------------------------------
# 1.14 â€” Final validation
# ------------------------------------------------------------

assert SETUP_REPORT_FILE.exists()

assert artifact_df[
    "Exists"
].all(), (
    "One or more required Notebook 48 "
    "artifacts are missing."
)

assert checkpoint_epoch >= 1

assert checkpoint_configuration.get(
    "actor_relative"
) is True

assert checkpoint_configuration.get(
    "variable_action"
) is True

assert notebook47_test_report.get(
    "test_split_untouched"
) is True

assert (
    notebook47_test_report.get(
        "test_multi_action_accuracy",
        0.0,
    )
    > 0.0
)

print()
print(
    "âœ… SECTION 1 TOURNAMENT SETUP PASSED"
)


# ## Section 2 â€” Baseline Agent and Match Interface Discovery
# 
# #### Inspect the production battle-agent, baseline-agent, search-engine, and match simulation APIs.
# 
# #### The goal is to identify the exact callable interfaces for:
# 
# - random or simple baseline agents;
# - greedy or heuristic agents;
# - search-based agents;
# - complete battle simulation;
# - tournament-compatible decision objects.
# 
# #### This prevents the benchmark from relying on assumed class names or signatures.

# In[2]:


# ============================================================
# NOTEBOOK 48
# SECTION 2 â€” BASELINE AGENT AND MATCH INTERFACE DISCOVERY
# ============================================================


import ast
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd


# ------------------------------------------------------------
# 2.3 â€” Files and directories to inspect
# ------------------------------------------------------------

DISCOVERY_PATHS = [
    SRC_DIR / "battle_agent.py",
    SRC_DIR / "battle_simulation.py",
    SRC_DIR / "legal_moves.py",
    SRC_DIR / "simulator.py",
    SRC_DIR / "agents",
    SRC_DIR / "engine",
]

python_files = []

for discovery_path in DISCOVERY_PATHS:
    if discovery_path.is_file():
        python_files.append(
            discovery_path
        )

    elif discovery_path.is_dir():
        python_files.extend(
            sorted(
                discovery_path.rglob(
                    "*.py"
                )
            )
        )

python_files = [
    path
    for path in python_files
    if (
        path.name != "__init__.py"
        and "__pycache__"
        not in path.parts
    )
]


# ------------------------------------------------------------
# 2.4 â€” AST definition extraction
# ------------------------------------------------------------

def module_name_from_path(
    file_path: Path,
) -> str:
    relative_path = (
        file_path
        .relative_to(
            PROJECT_ROOT
        )
        .with_suffix("")
    )

    return ".".join(
        relative_path.parts
    )


def extract_module_definitions(
    file_path: Path,
) -> dict[str, Any]:
    source_text = file_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source_text
    )

    functions = []
    classes = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                node.name
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            methods = [
                child.name
                for child in node.body
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
            ]

            classes.append(
                {
                    "name":
                        node.name,

                    "methods":
                        methods,
                }
            )

    return {
        "file":
            str(file_path),

        "module":
            module_name_from_path(
                file_path
            ),

        "functions":
            functions,

        "classes":
            classes,
    }


definition_inventory = []

for file_path in python_files:
    try:
        definition_inventory.append(
            extract_module_definitions(
                file_path
            )
        )

    except Exception as exc:
        definition_inventory.append(
            {
                "file":
                    str(file_path),

                "module":
                    module_name_from_path(
                        file_path
                    ),

                "functions":
                    [],

                "classes":
                    [],

                "parse_error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }
        )


# ------------------------------------------------------------
# 2.5 â€” Candidate classification
# ------------------------------------------------------------

BASELINE_TERMS = (
    "random",
    "greedy",
    "heuristic",
    "baseline",
    "simple",
)

AGENT_TERMS = (
    "agent",
    "policy",
    "player",
)

SEARCH_TERMS = (
    "search",
    "minimax",
    "alpha",
    "iddfs",
    "mcts",
)

MATCH_TERMS = (
    "simulate",
    "battle",
    "match",
    "tournament",
    "run_game",
)

DECISION_TERMS = (
    "decision",
    "choose_move",
    "select_move",
    "act",
)


def contains_term(
    value: str,
    terms: tuple[str, ...],
) -> bool:
    lowered = value.lower()

    return any(
        term in lowered
        for term in terms
    )


candidate_rows = []

for module_info in definition_inventory:
    module_name = module_info[
        "module"
    ]

    for function_name in module_info.get(
        "functions",
        [],
    ):
        categories = []

        if contains_term(
            function_name,
            BASELINE_TERMS,
        ):
            categories.append(
                "baseline"
            )

        if contains_term(
            function_name,
            AGENT_TERMS,
        ):
            categories.append(
                "agent"
            )

        if contains_term(
            function_name,
            SEARCH_TERMS,
        ):
            categories.append(
                "search"
            )

        if contains_term(
            function_name,
            MATCH_TERMS,
        ):
            categories.append(
                "match"
            )

        if contains_term(
            function_name,
            DECISION_TERMS,
        ):
            categories.append(
                "decision"
            )

        if categories:
            candidate_rows.append(
                {
                    "module":
                        module_name,

                    "object_type":
                        "function",

                    "object_name":
                        function_name,

                    "categories":
                        ", ".join(
                            categories
                        ),
                }
            )

    for class_info in module_info.get(
        "classes",
        [],
    ):
        class_name = class_info[
            "name"
        ]

        class_categories = []

        if contains_term(
            class_name,
            BASELINE_TERMS,
        ):
            class_categories.append(
                "baseline"
            )

        if contains_term(
            class_name,
            AGENT_TERMS,
        ):
            class_categories.append(
                "agent"
            )

        if contains_term(
            class_name,
            SEARCH_TERMS,
        ):
            class_categories.append(
                "search"
            )

        if contains_term(
            class_name,
            MATCH_TERMS,
        ):
            class_categories.append(
                "match"
            )

        if contains_term(
            class_name,
            DECISION_TERMS,
        ):
            class_categories.append(
                "decision"
            )

        if class_categories:
            candidate_rows.append(
                {
                    "module":
                        module_name,

                    "object_type":
                        "class",

                    "object_name":
                        class_name,

                    "categories":
                        ", ".join(
                            class_categories
                        ),
                }
            )

        for method_name in class_info.get(
            "methods",
            [],
        ):
            method_categories = []

            if contains_term(
                method_name,
                BASELINE_TERMS,
            ):
                method_categories.append(
                    "baseline"
                )

            if contains_term(
                method_name,
                AGENT_TERMS,
            ):
                method_categories.append(
                    "agent"
                )

            if contains_term(
                method_name,
                SEARCH_TERMS,
            ):
                method_categories.append(
                    "search"
                )

            if contains_term(
                method_name,
                MATCH_TERMS,
            ):
                method_categories.append(
                    "match"
                )

            if contains_term(
                method_name,
                DECISION_TERMS,
            ):
                method_categories.append(
                    "decision"
                )

            if method_categories:
                candidate_rows.append(
                    {
                        "module":
                            module_name,

                        "object_type":
                            "method",

                        "object_name":
                            (
                                f"{class_name}."
                                f"{method_name}"
                            ),

                        "categories":
                            ", ".join(
                                method_categories
                            ),
                    }
                )


candidate_df = pd.DataFrame(
    candidate_rows
).drop_duplicates()


# ------------------------------------------------------------
# 2.6 â€” Runtime import and signature inspection
# ------------------------------------------------------------

runtime_rows = []

for candidate in candidate_rows:
    module_name = candidate[
        "module"
    ]

    object_type = candidate[
        "object_type"
    ]

    object_name = candidate[
        "object_name"
    ]

    try:
        module = importlib.import_module(
            module_name
        )

        if object_type == "method":
            class_name, method_name = (
                object_name.split(
                    ".",
                    maxsplit=1,
                )
            )

            parent_class = getattr(
                module,
                class_name,
            )

            target_object = getattr(
                parent_class,
                method_name,
            )

        else:
            target_object = getattr(
                module,
                object_name,
            )

        try:
            signature = str(
                inspect.signature(
                    target_object
                )
            )

        except Exception:
            signature = (
                "<signature unavailable>"
            )

        try:
            source = inspect.getsource(
                target_object
            )

        except Exception as exc:
            source = (
                "<source unavailable: "
                f"{type(exc).__name__}: "
                f"{exc}>"
            )

        runtime_rows.append(
            {
                "module":
                    module_name,

                "object_type":
                    object_type,

                "object_name":
                    object_name,

                "categories":
                    candidate[
                        "categories"
                    ],

                "signature":
                    signature,

                "import_status":
                    "success",

                "source":
                    source,
            }
        )

    except Exception as exc:
        runtime_rows.append(
            {
                "module":
                    module_name,

                "object_type":
                    object_type,

                "object_name":
                    object_name,

                "categories":
                    candidate[
                        "categories"
                    ],

                "signature":
                    None,

                "import_status":
                    (
                        "failed: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                "source":
                    None,
            }
        )


runtime_df = pd.DataFrame(
    runtime_rows
).drop_duplicates(
    subset=[
        "module",
        "object_type",
        "object_name",
    ]
)


# ------------------------------------------------------------
# 2.7 â€” Filter useful groups
# ------------------------------------------------------------

successful_df = runtime_df[
    runtime_df[
        "import_status"
    ]
    == "success"
].copy()

baseline_df = successful_df[
    successful_df[
        "categories"
    ].str.contains(
        "baseline",
        case=False,
        na=False,
    )
].copy()

agent_df = successful_df[
    successful_df[
        "categories"
    ].str.contains(
        "agent|decision",
        case=False,
        na=False,
        regex=True,
    )
].copy()

search_df = successful_df[
    successful_df[
        "categories"
    ].str.contains(
        "search",
        case=False,
        na=False,
    )
].copy()

match_df = successful_df[
    successful_df[
        "categories"
    ].str.contains(
        "match",
        case=False,
        na=False,
    )
].copy()


# ------------------------------------------------------------
# 2.8 â€” Save discovery reports
# ------------------------------------------------------------

DEFINITION_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "baseline_agent_definition_inventory.json"
)

SIGNATURE_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "baseline_agent_signature_inventory.json"
)

SOURCE_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "baseline_agent_candidate_sources.txt"
)

with open(
    DEFINITION_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        definition_inventory,
        file,
        indent=4,
    )

with open(
    SIGNATURE_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        [
            {
                key: value
                for key, value in row.items()
                if key != "source"
            }
            for row in runtime_rows
        ],
        file,
        indent=4,
    )

with open(
    SOURCE_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    for row in runtime_rows:
        if row[
            "import_status"
        ] != "success":
            continue

        file.write(
            "=" * 100
            + "\n"
        )

        file.write(
            f"{row['module']}."
            f"{row['object_name']}\n"
        )

        file.write(
            f"Categories: "
            f"{row['categories']}\n"
        )

        file.write(
            f"Signature: "
            f"{row['signature']}\n"
        )

        file.write(
            "-" * 100
            + "\n"
        )

        file.write(
            row[
                "source"
            ]
            or ""
        )

        file.write(
            "\n\n"
        )


# ------------------------------------------------------------
# 2.9 â€” Display discovery results
# ------------------------------------------------------------

display_columns = [
    "module",
    "object_type",
    "object_name",
    "signature",
]

print(
    "NOTEBOOK 48 â€” BASELINE AGENT INTERFACE DISCOVERY"
)

print("=" * 100)

print()
print(
    "Python files inspected       :",
    len(
        python_files
    ),
)

print(
    "Candidate definitions        :",
    len(
        candidate_df
    ),
)

print(
    "Successful runtime imports   :",
    len(
        successful_df
    ),
)

print()
print(
    "BASELINE CANDIDATES"
)

print("-" * 100)

if len(
    baseline_df
):
    display(
        baseline_df[
            display_columns
        ].reset_index(
            drop=True
        )
    )

else:
    print(
        "No explicit baseline classes "
        "or functions were found."
    )

print()
print(
    "AGENT AND DECISION CANDIDATES"
)

print("-" * 100)

if len(
    agent_df
):
    display(
        agent_df[
            display_columns
        ].reset_index(
            drop=True
        )
    )

else:
    print(
        "No agent or decision "
        "candidates were found."
    )

print()
print(
    "SEARCH CANDIDATES"
)

print("-" * 100)

if len(
    search_df
):
    display(
        search_df[
            display_columns
        ].reset_index(
            drop=True
        )
    )

else:
    print(
        "No search candidates "
        "were found."
    )

print()
print(
    "MATCH AND SIMULATION CANDIDATES"
)

print("-" * 100)

if len(
    match_df
):
    display(
        match_df[
            display_columns
        ].reset_index(
            drop=True
        )
    )

else:
    print(
        "No match or simulation "
        "candidates were found."
    )

print()
print(
    "Definition report:",
    DEFINITION_REPORT_FILE,
)

print(
    "Signature report:",
    SIGNATURE_REPORT_FILE,
)

print(
    "Source report:",
    SOURCE_REPORT_FILE,
)


# ------------------------------------------------------------
# 2.10 â€” Final validation
# ------------------------------------------------------------

assert DEFINITION_REPORT_FILE.exists()

assert SIGNATURE_REPORT_FILE.exists()

assert SOURCE_REPORT_FILE.exists()

assert len(
    successful_df
) > 0

assert len(
    agent_df
) > 0, (
    "No usable agent or decision "
    "interfaces were discovered."
)

assert len(
    match_df
) > 0, (
    "No usable match or simulation "
    "interfaces were discovered."
)

print()
print(
    "âœ… SECTION 2 BASELINE INTERFACE DISCOVERY PASSED"
)


# ## Section 3 â€” Tournament Baseline Agents and Match Runner
# 
# #### Create simulator-compatible baseline agents and a reusable tournament match runner.
# 
# #### The baselines include:
# 
# - random legal-move selection;
# - greedy highest-damage selection;
# - production search-agent selection.
# 
# #### All agents return decision objects compatible with `simulate_ai_battle`.

# In[4]:


# ============================================================
# NOTEBOOK 48
# SECTION 3 â€” TOURNAMENT BASELINES AND MATCH RUNNER
# ============================================================


import copy
import dataclasses
import inspect
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.battle_agent import (
    PokemonBattleAgent,
)

from src.battle_simulation import (
    create_battle_transcript,
    simulate_ai_battle,
)

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

from src.competitive.competitive_policy_agent import (
    CompetitivePolicyAgent,
)

from src.legal_moves import (
    get_current_legal_moves,
)


# ------------------------------------------------------------
# 3.3 â€” Decision object compatible with simulator
# ------------------------------------------------------------

@dataclass
class TournamentDecision:
    move: Any
    score: float
    search_depth: int
    nodes: int

    confidence: float = 0.0
    selected_move_name: str | None = None
    fallback: bool = False
    reason: str | None = None


# ------------------------------------------------------------
# 3.4 â€” Move helpers
# ------------------------------------------------------------

def tournament_move_name(
    move: Any,
) -> str:
    if isinstance(move, dict):
        return str(
            move.get("name")
            or move.get("Move Name")
            or move.get("move_name")
            or move
        )

    for attribute_name in (
        "name",
        "move_name",
        "attack_name",
    ):
        value = getattr(
            move,
            attribute_name,
            None,
        )

        if value is not None:
            return str(value)

    return str(move)


def tournament_move_damage(
    move: Any,
) -> float:
    if isinstance(move, dict):
        for key in (
            "damage",
            "damage_numeric",
            "base_damage",
        ):
            if key in move:
                try:
                    return float(
                        move[key] or 0.0
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    return 0.0

        return 0.0

    for attribute_name in (
        "damage",
        "damage_numeric",
        "base_damage",
    ):
        value = getattr(
            move,
            attribute_name,
            None,
        )

        if value is not None:
            try:
                return float(value or 0.0)
            except (
                TypeError,
                ValueError,
            ):
                return 0.0

    return 0.0


# ------------------------------------------------------------
# 3.5 â€” Random baseline agent
# ------------------------------------------------------------

class RandomLegalMoveAgent:
    def __init__(
        self,
        seed: int = RANDOM_SEED,
        name: str = "Random Legal Agent",
    ) -> None:
        self.name = name
        self.random = random.Random(
            seed
        )
        self.last_decision = None

    def choose_move(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> TournamentDecision:
        resolved_state = (
            state
            if state is not None
            else battle_state
        )

        if resolved_state is None:
            raise ValueError(
                "A battle state is required."
            )

        legal_moves = (
            get_current_legal_moves(
                resolved_state
            )
        )

        if not legal_moves:
            self.last_decision = (
                TournamentDecision(
                    move=None,
                    score=0.0,
                    search_depth=int(
                        depth or 0
                    ),
                    nodes=0,
                    confidence=0.0,
                    selected_move_name=None,
                    fallback=True,
                    reason=
                        "No legal moves available.",
                )
            )

            return self.last_decision

        selected_move = (
            self.random.choice(
                legal_moves
            )
        )

        confidence = (
            1.0
            / len(legal_moves)
        )

        self.last_decision = (
            TournamentDecision(
                move=
                    selected_move,

                score=
                    confidence,

                search_depth=int(
                    depth or 0
                ),

                nodes=0,

                confidence=
                    confidence,

                selected_move_name=
                    tournament_move_name(
                        selected_move
                    ),

                fallback=False,
            )
        )

        return self.last_decision

    def get_last_decision(
        self,
    ) -> TournamentDecision | None:
        return self.last_decision


# ------------------------------------------------------------
# 3.6 â€” Greedy damage baseline
# ------------------------------------------------------------

class GreedyDamageAgent:
    def __init__(
        self,
        name: str = "Greedy Damage Agent",
    ) -> None:
        self.name = name
        self.last_decision = None

    def choose_move(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> TournamentDecision:
        resolved_state = (
            state
            if state is not None
            else battle_state
        )

        if resolved_state is None:
            raise ValueError(
                "A battle state is required."
            )

        legal_moves = (
            get_current_legal_moves(
                resolved_state
            )
        )

        if not legal_moves:
            self.last_decision = (
                TournamentDecision(
                    move=None,
                    score=0.0,
                    search_depth=int(
                        depth or 0
                    ),
                    nodes=0,
                    confidence=0.0,
                    selected_move_name=None,
                    fallback=True,
                    reason=
                        "No legal moves available.",
                )
            )

            return self.last_decision

        selected_index = max(
            range(
                len(legal_moves)
            ),
            key=lambda index: (
                tournament_move_damage(
                    legal_moves[index]
                ),
                -index,
            ),
        )

        selected_move = (
            legal_moves[
                selected_index
            ]
        )

        maximum_damage = (
            tournament_move_damage(
                selected_move
            )
        )

        self.last_decision = (
            TournamentDecision(
                move=
                    selected_move,

                score=
                    maximum_damage,

                search_depth=int(
                    depth or 0
                ),

                nodes=0,

                confidence=1.0,

                selected_move_name=
                    tournament_move_name(
                        selected_move
                    ),

                fallback=False,
            )
        )

        return self.last_decision

    def get_last_decision(
        self,
    ) -> TournamentDecision | None:
        return self.last_decision


# ------------------------------------------------------------
# 3.7 â€” Search baseline adapter
# ------------------------------------------------------------

from src.adapters import (
    current_player_adapter,
    evaluate_state_adapter,
    generate_moves_adapter,
    terminal_state_adapter,
)

from src.engine.advanced_search import (
    AdvancedSearchEngine,
)

from src.simulator import (
    apply_move,
)


def tournament_state_key(
    state: BattleState,
) -> tuple:
    def pokemon_key(
        pokemon: Any,
    ) -> tuple | None:
        if pokemon is None:
            return None

        card = getattr(
            pokemon,
            "card",
            None,
        )

        if isinstance(
            card,
            dict,
        ):
            card_name = (
                card.get("name")
                or card.get("Name")
                or ""
            )
        else:
            card_name = getattr(
                card,
                "name",
                "",
            )

        return (
            str(card_name),

            float(
                getattr(
                    pokemon,
                    "current_hp",
                    0.0,
                )
            ),

            int(
                getattr(
                    pokemon,
                    "attached_energy",
                    0,
                )
            ),

            str(
                getattr(
                    pokemon,
                    "status",
                    None,
                )
            ),

            float(
                getattr(
                    pokemon,
                    "damage",
                    0.0,
                )
            ),

            bool(
                getattr(
                    pokemon,
                    "is_active",
                    False,
                )
            ),
        )

    def player_key(
        player: Any,
    ) -> tuple | None:
        if player is None:
            return None

        bench = getattr(
            player,
            "bench",
            [],
        )

        return (
            pokemon_key(
                getattr(
                    player,
                    "active",
                    None,
                )
            ),

            tuple(
                pokemon_key(pokemon)
                for pokemon in bench
            ),

            int(
                getattr(
                    player,
                    "prize_cards_remaining",
                    0,
                )
            ),

            int(
                getattr(
                    player,
                    "hand_size",
                    0,
                )
            ),
        )

    return (
        player_key(
            getattr(
                state,
                "player",
                None,
            )
        ),

        player_key(
            getattr(
                state,
                "opponent",
                None,
            )
        ),

        int(
            getattr(
                state,
                "turn_number",
                0,
            )
        ),

        str(
            getattr(
                state,
                "current_player",
                "",
            )
        ),
    )


def tournament_move_key(
    move: Any,
) -> tuple:
    if isinstance(
        move,
        dict,
    ):
        return (
            tournament_move_name(
                move
            ),

            tournament_move_damage(
                move
            ),

            int(
                move.get(
                    "energy_cost",
                    move.get(
                        "cost",
                        0,
                    ),
                )
                or 0
            ),

            repr(
                sorted(
                    move.items(),
                    key=lambda item:
                        str(item[0]),
                )
            ),
        )

    return (
        tournament_move_name(
            move
        ),

        tournament_move_damage(
            move
        ),

        int(
            getattr(
                move,
                "energy_cost",
                getattr(
                    move,
                    "cost",
                    0,
                ),
            )
            or 0
        ),

        repr(move),
    )


class SearchAgentAdapter:
    def __init__(
        self,
        search_depth: int =
            SIMULATOR_SEARCH_DEPTH,
        name: str =
            "Production Search Agent",
    ) -> None:
        self.name = name

        self.search_depth = int(
            search_depth
        )

        self.search_engine = (
            AdvancedSearchEngine(
                generate_moves=
                    generate_moves_adapter,

                apply_move=
                    apply_move,

                evaluate_state=
                    evaluate_state_adapter,

                is_terminal=
                    terminal_state_adapter,

                current_player=
                    current_player_adapter,

                state_key=
                    tournament_state_key,

                move_key=
                    tournament_move_key,
            )
        )

        self.inner_agent = (
            PokemonBattleAgent(
                engine=
                    self.search_engine
            )
        )

        self.last_decision = None

    def choose_move(
        self,
        state: Any = None,
        depth: int | None = None,
        battle_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        resolved_state = (
            state
            if state is not None
            else battle_state
        )

        if resolved_state is None:
            raise ValueError(
                "A battle state is required."
            )

        resolved_depth = int(
            depth
            if depth is not None
            else self.search_depth
        )

        self.last_decision = (
            self.inner_agent
            .choose_move(
                state=
                    resolved_state,

                depth=
                    resolved_depth,
            )
        )

        return self.last_decision

    def get_last_decision(
        self,
    ) -> Any:
        return self.last_decision
# ------------------------------------------------------------
# 3.8 â€” Tournament battle templates
# ------------------------------------------------------------

TOURNAMENT_CARD_TEMPLATES = [
    {
        "name": "Pikachu",
        "hp": 120,
        "attacks": [
            {
                "Move Name":
                    "Quick Attack",
                "damage_numeric":
                    30,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Thunderbolt",
                "damage_numeric":
                    90,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },

    {
        "name": "Charmander",
        "hp": 100,
        "attacks": [
            {
                "Move Name":
                    "Scratch",
                "damage_numeric":
                    20,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Flame Tail",
                "damage_numeric":
                    50,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },

    {
        "name": "Squirtle",
        "hp": 110,
        "attacks": [
            {
                "Move Name":
                    "Tackle",
                "damage_numeric":
                    20,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Water Pulse",
                "damage_numeric":
                    60,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },

    {
        "name": "Bulbasaur",
        "hp": 110,
        "attacks": [
            {
                "Move Name":
                    "Vine Whip",
                "damage_numeric":
                    30,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Razor Leaf",
                "damage_numeric":
                    70,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },

    {
        "name": "Eevee",
        "hp": 100,
        "attacks": [
            {
                "Move Name":
                    "Tail Whap",
                "damage_numeric":
                    20,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Bite",
                "damage_numeric":
                    50,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },

    {
        "name": "Meowth",
        "hp": 90,
        "attacks": [
            {
                "Move Name":
                    "Scratch",
                "damage_numeric":
                    20,
                "energy_cost":
                    1,
                "Effect Explanation":
                    None,
            },
            {
                "Move Name":
                    "Fury Swipes",
                "damage_numeric":
                    60,
                "energy_cost":
                    2,
                "Effect Explanation":
                    None,
            },
        ],
    },
]


# ------------------------------------------------------------
# 3.9 â€” Build deterministic initial battle state
# ------------------------------------------------------------

def create_tournament_state(
    player_card: dict[str, Any],
    opponent_card: dict[str, Any],
    starting_side: str = "Player",
    player_energy: int = 3,
    opponent_energy: int = 3,
) -> BattleState:
    player_card_copy = copy.deepcopy(
        player_card
    )

    opponent_card_copy = copy.deepcopy(
        opponent_card
    )

    return BattleState(
        player=PlayerState(
            active=PokemonState(
                card=
                    player_card_copy,

                current_hp=float(
                    player_card_copy[
                        "hp"
                    ]
                ),

                attached_energy=int(
                    player_energy
                ),

                status=None,

                damage=0.0,

                is_active=True,
            ),

            bench=[],

            prize_cards_remaining=1,

            hand_size=7,
        ),

        opponent=PlayerState(
            active=PokemonState(
                card=
                    opponent_card_copy,

                current_hp=float(
                    opponent_card_copy[
                        "hp"
                    ]
                ),

                attached_energy=int(
                    opponent_energy
                ),

                status=None,

                damage=0.0,

                is_active=True,
            ),

            bench=[],

            prize_cards_remaining=1,

            hand_size=7,
        ),

        turn_number=1,

        current_player=
            starting_side,
    )


# ------------------------------------------------------------
# 3.10 â€” Match result structure
# ------------------------------------------------------------

@dataclass
class TournamentMatchResult:
    match_id: int
    agent_name: str
    baseline_name: str
    player_card: str
    opponent_card: str
    starting_side: str
    winner: str | None
    turns: int
    stop_reason: str | None
    agent_result: float
    fallback_count: int
    elapsed_seconds: float


# ------------------------------------------------------------
# 3.11 â€” Run one simulator match
# ------------------------------------------------------------

def run_tournament_match(
    match_id: int,
    evaluated_agent: Any,
    baseline_agent: Any,
    player_card: dict[str, Any],
    opponent_card: dict[str, Any],
    starting_side: str,
    evaluated_agent_side: str,
    max_turns: int =
        MAX_TURNS_PER_MATCH,
    search_depth: int =
        SIMULATOR_SEARCH_DEPTH,
    verbose: bool =
        VERBOSE_MATCHES,
) -> TournamentMatchResult:
    if evaluated_agent_side not in {
        "Player",
        "Opponent",
    }:
        raise ValueError(
            "evaluated_agent_side must be "
            "'Player' or 'Opponent'."
        )

    initial_state = (
        create_tournament_state(
            player_card=
                player_card,

            opponent_card=
                opponent_card,

            starting_side=
                starting_side,
        )
    )

    class SideAwareAgent:
        def __init__(
            self,
            player_agent: Any,
            opponent_agent: Any,
        ) -> None:
            self.player_agent = (
                player_agent
            )

            self.opponent_agent = (
                opponent_agent
            )

            self.fallback_count = 0

        def choose_move(
            self,
            state: Any = None,
            depth: int | None = None,
            battle_state: Any = None,
            **kwargs: Any,
        ) -> Any:
            resolved_state = (
                state
                if state is not None
                else battle_state
            )

            if resolved_state is None:
                raise ValueError(
                    "A battle state is required."
                )

            side = str(
                resolved_state
                .current_player
            )

            selected_agent = (
                self.player_agent
                if side == "Player"
                else self.opponent_agent
            )

            decision = (
                selected_agent
                .choose_move(
                    state=
                        resolved_state,

                    depth=
                        depth,
                )
            )

            if bool(
                getattr(
                    decision,
                    "fallback",
                    False,
                )
            ):
                self.fallback_count += 1

            return decision

    if evaluated_agent_side == "Player":
        side_agent = SideAwareAgent(
            player_agent=
                evaluated_agent,

            opponent_agent=
                baseline_agent,
        )

    else:
        side_agent = SideAwareAgent(
            player_agent=
                baseline_agent,

            opponent_agent=
                evaluated_agent,
        )

    start_time = time.perf_counter()

    simulation = simulate_ai_battle(
        initial_state=
            initial_state,

        agent=
            side_agent,

        search_depth=
            search_depth,

        max_turns=
            max_turns,

        verbose=
            verbose,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    winner = getattr(
        simulation,
        "winner",
        None,
    )

    turns = getattr(
        simulation,
        "turns",
        [],
    )

    stop_reason = getattr(
        simulation,
        "stop_reason",
        None,
    )

    if winner is None:
        agent_result = 0.5

    elif winner == evaluated_agent_side:
        agent_result = 1.0

    else:
        agent_result = 0.0

    return TournamentMatchResult(
        match_id=
            int(match_id),

        agent_name=
            str(
                getattr(
                    evaluated_agent,
                    "name",
                    type(
                        evaluated_agent
                    ).__name__,
                )
            ),

        baseline_name=
            str(
                getattr(
                    baseline_agent,
                    "name",
                    type(
                        baseline_agent
                    ).__name__,
                )
            ),

        player_card=
            str(
                player_card[
                    "name"
                ]
            ),

        opponent_card=
            str(
                opponent_card[
                    "name"
                ]
            ),

        starting_side=
            str(
                starting_side
            ),

        winner=winner,

        turns=
            len(turns),

        stop_reason=
            stop_reason,

        agent_result=
            float(
                agent_result
            ),

        fallback_count=
            int(
                side_agent
                .fallback_count
            ),

        elapsed_seconds=
            float(
                elapsed_seconds
            ),
    )


# ------------------------------------------------------------
# 3.12 â€” Instantiate agents
# ------------------------------------------------------------

competitive_tournament_agent = (
    CompetitivePolicyAgent(
        checkpoint_path=
            BEST_POLICY_CHECKPOINT,

        device=str(
            DEVICE
        ),

        name=
            "Competitive Policy",
    )
)

random_baseline_agent = (
    RandomLegalMoveAgent(
        seed=
            RANDOM_SEED,

        name=
            "Random Baseline",
    )
)

greedy_baseline_agent = (
    GreedyDamageAgent(
        name=
            "Greedy Damage Baseline",
    )
)

search_baseline_agent = (
    SearchAgentAdapter(
        search_depth=
            SIMULATOR_SEARCH_DEPTH,

        name=
            "Depth-6 Search Baseline",
    )
)


# ------------------------------------------------------------
# 3.13 â€” Smoke-test one match per baseline
# ------------------------------------------------------------

smoke_test_results = []

smoke_player_card = (
    TOURNAMENT_CARD_TEMPLATES[0]
)

smoke_opponent_card = (
    TOURNAMENT_CARD_TEMPLATES[1]
)

for smoke_index, baseline_agent in enumerate(
    [
        random_baseline_agent,
        greedy_baseline_agent,
        search_baseline_agent,
    ],
    start=1,
):
    smoke_result = run_tournament_match(
        match_id=
            smoke_index,

        evaluated_agent=
            competitive_tournament_agent,

        baseline_agent=
            baseline_agent,

        player_card=
            smoke_player_card,

        opponent_card=
            smoke_opponent_card,

        starting_side=
            "Player",

        evaluated_agent_side=
            "Player",

        max_turns=
            20,

        search_depth=
            SIMULATOR_SEARCH_DEPTH,

        verbose=False,
    )

    smoke_test_results.append(
        dataclasses.asdict(
            smoke_result
        )
    )


smoke_test_df = pd.DataFrame(
    smoke_test_results
)


# ------------------------------------------------------------
# 3.14 â€” Save adapter report
# ------------------------------------------------------------

MATCH_ADAPTER_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "tournament_match_adapter_summary.json"
)

match_adapter_report = {
    "competitive_agent":
        competitive_tournament_agent.name,

    "baselines": [
        random_baseline_agent.name,
        greedy_baseline_agent.name,
        search_baseline_agent.name,
    ],

    "simulator_signature":
        str(
            inspect.signature(
                simulate_ai_battle
            )
        ),

    "smoke_tests":
        smoke_test_results,

    "all_smoke_tests_completed":
        bool(
            len(smoke_test_results)
            == 3
        ),

    "fallbacks":
        int(
            smoke_test_df[
                "fallback_count"
            ].sum()
        ),
}

with open(
    MATCH_ADAPTER_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        match_adapter_report,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 3.15 â€” Display smoke-test results
# ------------------------------------------------------------

print(
    "NOTEBOOK 48 â€” TOURNAMENT MATCH ADAPTER"
)

print("=" * 110)

print()
print(
    "Competitive agent:",
    competitive_tournament_agent.name,
)

print()
print(
    "Baselines:"
)

print(
    " -",
    random_baseline_agent.name,
)

print(
    " -",
    greedy_baseline_agent.name,
)

print(
    " -",
    search_baseline_agent.name,
)

print()
print("SMOKE-TEST MATCHES")
print("-" * 110)

display(
    smoke_test_df
)

print()
print(
    "Match adapter report:",
    MATCH_ADAPTER_REPORT_FILE,
)


# ------------------------------------------------------------
# 3.16 â€” Final validation
# ------------------------------------------------------------

assert MATCH_ADAPTER_REPORT_FILE.exists()

assert len(
    smoke_test_df
) == 3

assert (
    smoke_test_df[
        "turns"
    ]
    > 0
).all()

assert (
    smoke_test_df[
        "fallback_count"
    ]
    == 0
).all()

assert smoke_test_df[
    "agent_result"
].between(
    0.0,
    1.0,
).all()

print()
print(
    "âœ… SECTION 3 TOURNAMENT MATCH ADAPTER PASSED"
)


# ## Section 4 â€” Full Baseline Tournament
# 
# #### Run repeated side-swapped matches between the Competitive Policy agent and each baseline.
# 
# #### The tournament evaluates:
# 
# - Random Baseline
# - Greedy Damage Baseline
# - Depth-6 Search Baseline
# 
# #### Each pairing uses multiple card matchups and both evaluated-agent sides.

# In[6]:


# ============================================================
# NOTEBOOK 48
# SECTION 4 â€” FULL BASELINE TOURNAMENT
# ============================================================


import dataclasses
import json
import math
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 4.3 â€” Tournament configuration
# ------------------------------------------------------------

TOURNAMENT_MATCHES_PER_BASELINE = (
    INITIAL_MATCHES_PER_PAIRING
)

TOURNAMENT_PROGRESS_INTERVAL = 25

TOURNAMENT_RESULTS_FILE = (
    TOURNAMENT_DATA_DIR
    / "competitive_policy_baseline_tournament.csv"
)

TOURNAMENT_SUMMARY_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "competitive_policy_baseline_tournament_summary.json"
)

TOURNAMENT_FAILURE_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "competitive_policy_baseline_tournament_failures.json"
)

TOURNAMENT_RANDOM_SEED = 4804

tournament_random = random.Random(
    TOURNAMENT_RANDOM_SEED
)


# ------------------------------------------------------------
# 4.4 â€” Fresh-agent factory
#
# New instances prevent state, random-generator, or search
# tables from leaking between matches.
# ------------------------------------------------------------

def create_competitive_agent() -> CompetitivePolicyAgent:
    return CompetitivePolicyAgent(
        checkpoint_path=
            BEST_POLICY_CHECKPOINT,

        device=str(
            DEVICE
        ),

        name=
            "Competitive Policy",
    )


def create_random_baseline(
    seed: int,
) -> RandomLegalMoveAgent:
    return RandomLegalMoveAgent(
        seed=seed,
        name="Random Baseline",
    )


def create_greedy_baseline() -> GreedyDamageAgent:
    return GreedyDamageAgent(
        name="Greedy Damage Baseline",
    )


def create_search_baseline() -> SearchAgentAdapter:
    return SearchAgentAdapter(
        search_depth=
            SIMULATOR_SEARCH_DEPTH,

        name=
            "Depth-6 Search Baseline",
    )


baseline_factories = {
    "Random Baseline":
        lambda seed:
            create_random_baseline(
                seed
            ),

    "Greedy Damage Baseline":
        lambda seed:
            create_greedy_baseline(),

    "Depth-6 Search Baseline":
        lambda seed:
            create_search_baseline(),
}


# ------------------------------------------------------------
# 4.5 â€” Balanced tournament schedule
# ------------------------------------------------------------

def build_balanced_schedule(
    baseline_name: str,
    match_count: int,
) -> list[dict[str, Any]]:
    schedule = []

    card_count = len(
        TOURNAMENT_CARD_TEMPLATES
    )

    for match_index in range(
        match_count
    ):
        evaluated_side = (
            "Player"
            if match_index % 2 == 0
            else "Opponent"
        )

        starting_side = (
            "Player"
            if (
                match_index // 2
            ) % 2 == 0
            else "Opponent"
        )

        first_card_index = (
            match_index
            % card_count
        )

        second_card_index = (
            (
                match_index * 3
            )
            + 1
        ) % card_count

        if (
            second_card_index
            == first_card_index
        ):
            second_card_index = (
                second_card_index + 1
            ) % card_count

        player_card = (
            TOURNAMENT_CARD_TEMPLATES[
                first_card_index
            ]
        )

        opponent_card = (
            TOURNAMENT_CARD_TEMPLATES[
                second_card_index
            ]
        )

        schedule.append(
            {
                "baseline_name":
                    baseline_name,

                "evaluated_agent_side":
                    evaluated_side,

                "starting_side":
                    starting_side,

                "player_card":
                    player_card,

                "opponent_card":
                    opponent_card,
            }
        )

    return schedule


full_schedule = []

for baseline_name in (
    baseline_factories
):
    full_schedule.extend(
        build_balanced_schedule(
            baseline_name=
                baseline_name,

            match_count=
                TOURNAMENT_MATCHES_PER_BASELINE,
        )
    )


# ------------------------------------------------------------
# 4.6 â€” Run tournament
# ------------------------------------------------------------

tournament_rows = []

tournament_failures = []

tournament_start = time.perf_counter()

for global_match_index, scheduled_match in enumerate(
    full_schedule,
    start=1,
):
    baseline_name = (
        scheduled_match[
            "baseline_name"
        ]
    )

    try:
        evaluated_agent = (
            create_competitive_agent()
        )

        baseline_agent = (
            baseline_factories[
                baseline_name
            ](
                TOURNAMENT_RANDOM_SEED
                + global_match_index
            )
        )

        result = run_tournament_match(
            match_id=
                global_match_index,

            evaluated_agent=
                evaluated_agent,

            baseline_agent=
                baseline_agent,

            player_card=
                scheduled_match[
                    "player_card"
                ],

            opponent_card=
                scheduled_match[
                    "opponent_card"
                ],

            starting_side=
                scheduled_match[
                    "starting_side"
                ],

            evaluated_agent_side=
                scheduled_match[
                    "evaluated_agent_side"
                ],

            max_turns=
                MAX_TURNS_PER_MATCH,

            search_depth=
                SIMULATOR_SEARCH_DEPTH,

            verbose=False,
        )

        result_row = dataclasses.asdict(
            result
        )

        result_row[
            "evaluated_agent_side"
        ] = scheduled_match[
            "evaluated_agent_side"
        ]

        result_row[
            "baseline_result"
        ] = (
            1.0
            - float(
                result.agent_result
            )
        )

        result_row[
            "draw"
        ] = bool(
            float(
                result.agent_result
            )
            == 0.5
        )

        result_row[
            "competitive_win"
        ] = bool(
            float(
                result.agent_result
            )
            == 1.0
        )

        result_row[
            "competitive_loss"
        ] = bool(
            float(
                result.agent_result
            )
            == 0.0
        )

        tournament_rows.append(
            result_row
        )

    except Exception as exc:
        tournament_failures.append(
            {
                "match_id":
                    global_match_index,

                "baseline_name":
                    baseline_name,

                "evaluated_agent_side":
                    scheduled_match[
                        "evaluated_agent_side"
                    ],

                "starting_side":
                    scheduled_match[
                        "starting_side"
                    ],

                "player_card":
                    scheduled_match[
                        "player_card"
                    ][
                        "name"
                    ],

                "opponent_card":
                    scheduled_match[
                        "opponent_card"
                    ][
                        "name"
                    ],

                "error_type":
                    type(exc).__name__,

                "error":
                    str(exc),
            }
        )

    if (
        global_match_index
        % TOURNAMENT_PROGRESS_INTERVAL
        == 0
        or global_match_index
        == len(full_schedule)
    ):
        elapsed = (
            time.perf_counter()
            - tournament_start
        )

        completed = len(
            tournament_rows
        )

        speed = (
            completed
            / max(
                elapsed,
                0.001,
            )
        )

        print(
            f"Completed "
            f"{global_match_index:,}/"
            f"{len(full_schedule):,} "
            f"scheduled matches | "
            f"{completed:,} successful | "
            f"{len(tournament_failures):,} failed | "
            f"{speed:.2f} matches/sec"
        )


tournament_elapsed = (
    time.perf_counter()
    - tournament_start
)


# ------------------------------------------------------------
# 4.7 â€” Build results DataFrame
# ------------------------------------------------------------

tournament_df = pd.DataFrame(
    tournament_rows
)

assert len(
    tournament_df
) > 0, (
    "The tournament produced no successful matches."
)

tournament_df.to_csv(
    TOURNAMENT_RESULTS_FILE,
    index=False,
)


# ------------------------------------------------------------
# 4.8 â€” Aggregate baseline results
# ------------------------------------------------------------

baseline_summary_df = (
    tournament_df
    .groupby(
        "baseline_name"
    )
    .agg(
        matches=(
            "match_id",
            "size",
        ),

        competitive_wins=(
            "competitive_win",
            "sum",
        ),

        competitive_losses=(
            "competitive_loss",
            "sum",
        ),

        draws=(
            "draw",
            "sum",
        ),

        score=(
            "agent_result",
            "mean",
        ),

        average_turns=(
            "turns",
            "mean",
        ),

        median_turns=(
            "turns",
            "median",
        ),

        average_seconds=(
            "elapsed_seconds",
            "mean",
        ),

        fallback_count=(
            "fallback_count",
            "sum",
        ),
    )
    .reset_index()
)

baseline_summary_df[
    "win_rate"
] = (
    baseline_summary_df[
        "competitive_wins"
    ]
    / baseline_summary_df[
        "matches"
    ]
)

baseline_summary_df[
    "loss_rate"
] = (
    baseline_summary_df[
        "competitive_losses"
    ]
    / baseline_summary_df[
        "matches"
    ]
)

baseline_summary_df[
    "draw_rate"
] = (
    baseline_summary_df[
        "draws"
    ]
    / baseline_summary_df[
        "matches"
    ]
)


# ------------------------------------------------------------
# 4.9 â€” Side-swap analysis
# ------------------------------------------------------------

side_summary_df = (
    tournament_df
    .groupby(
        [
            "baseline_name",
            "evaluated_agent_side",
        ]
    )
    .agg(
        matches=(
            "match_id",
            "size",
        ),

        wins=(
            "competitive_win",
            "sum",
        ),

        losses=(
            "competitive_loss",
            "sum",
        ),

        draws=(
            "draw",
            "sum",
        ),

        score=(
            "agent_result",
            "mean",
        ),

        average_turns=(
            "turns",
            "mean",
        ),
    )
    .reset_index()
)

side_summary_df[
    "win_rate"
] = (
    side_summary_df[
        "wins"
    ]
    / side_summary_df[
        "matches"
    ]
)


# ------------------------------------------------------------
# 4.10 â€” Matchup analysis
# ------------------------------------------------------------

matchup_summary_df = (
    tournament_df
    .groupby(
        [
            "baseline_name",
            "player_card",
            "opponent_card",
        ]
    )
    .agg(
        matches=(
            "match_id",
            "size",
        ),

        score=(
            "agent_result",
            "mean",
        ),

        wins=(
            "competitive_win",
            "sum",
        ),

        average_turns=(
            "turns",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "baseline_name",
            "matches",
            "score",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
)


# ------------------------------------------------------------
# 4.11 â€” Overall statistics
# ------------------------------------------------------------

overall_matches = len(
    tournament_df
)

overall_wins = int(
    tournament_df[
        "competitive_win"
    ].sum()
)

overall_losses = int(
    tournament_df[
        "competitive_loss"
    ].sum()
)

overall_draws = int(
    tournament_df[
        "draw"
    ].sum()
)

overall_score = float(
    tournament_df[
        "agent_result"
    ].mean()
)

overall_win_rate = (
    overall_wins
    / overall_matches
)

overall_fallbacks = int(
    tournament_df[
        "fallback_count"
    ].sum()
)


# ------------------------------------------------------------
# 4.12 â€” Save failure and summary reports
# ------------------------------------------------------------

with open(
    TOURNAMENT_FAILURE_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        tournament_failures,
        file,
        indent=4,
        default=str,
    )

tournament_summary = {
    "scheduled_matches":
        int(
            len(full_schedule)
        ),

    "successful_matches":
        int(
            overall_matches
        ),

    "failed_matches":
        int(
            len(
                tournament_failures
            )
        ),

    "matches_per_baseline":
        int(
            TOURNAMENT_MATCHES_PER_BASELINE
        ),

    "competitive_wins":
        overall_wins,

    "competitive_losses":
        overall_losses,

    "draws":
        overall_draws,

    "overall_score":
        overall_score,

    "overall_win_rate":
        overall_win_rate,

    "average_turns":
        float(
            tournament_df[
                "turns"
            ].mean()
        ),

    "fallback_count":
        overall_fallbacks,

    "elapsed_seconds":
        float(
            tournament_elapsed
        ),

    "matches_per_second":
        float(
            overall_matches
            / max(
                tournament_elapsed,
                0.001,
            )
        ),

    "baseline_results":
        baseline_summary_df
        .to_dict(
            orient="records"
        ),

    "side_results":
        side_summary_df
        .to_dict(
            orient="records"
        ),

    "results_file":
        str(
            TOURNAMENT_RESULTS_FILE
        ),

    "failure_file":
        str(
            TOURNAMENT_FAILURE_FILE
        ),
}

with open(
    TOURNAMENT_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        tournament_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 4.13 â€” Display tournament results
# ------------------------------------------------------------

print()
print(
    "NOTEBOOK 48 â€” FULL BASELINE TOURNAMENT"
)

print("=" * 115)

print()
print(
    "Scheduled matches :",
    f"{len(full_schedule):,}",
)

print(
    "Successful matches:",
    f"{overall_matches:,}",
)

print(
    "Failed matches    :",
    f"{len(tournament_failures):,}",
)

print(
    "Competitive wins  :",
    f"{overall_wins:,}",
)

print(
    "Competitive losses:",
    f"{overall_losses:,}",
)

print(
    "Draws             :",
    f"{overall_draws:,}",
)

print(
    "Overall win rate  :",
    round(
        overall_win_rate,
        4,
    ),
)

print(
    "Overall score     :",
    round(
        overall_score,
        4,
    ),
)

print(
    "Average turns     :",
    round(
        float(
            tournament_df[
                "turns"
            ].mean()
        ),
        3,
    ),
)

print(
    "Fallback count    :",
    overall_fallbacks,
)

print(
    "Tournament time   :",
    round(
        tournament_elapsed,
        2,
    ),
    "seconds",
)

print()
print(
    "RESULTS BY BASELINE"
)

print("-" * 115)

display(
    baseline_summary_df
)

print()
print(
    "RESULTS BY SIDE"
)

print("-" * 115)

display(
    side_summary_df
)

print()
print(
    "TOP MATCHUP RESULTS"
)

print("-" * 115)

display(
    matchup_summary_df
    .head(30)
)

print()
print(
    "Results file:",
    TOURNAMENT_RESULTS_FILE,
)

print(
    "Summary file:",
    TOURNAMENT_SUMMARY_FILE,
)

print(
    "Failure file:",
    TOURNAMENT_FAILURE_FILE,
)


# ------------------------------------------------------------
# 4.14 â€” Final validation
# ------------------------------------------------------------

assert TOURNAMENT_RESULTS_FILE.exists()

assert TOURNAMENT_SUMMARY_FILE.exists()

assert TOURNAMENT_FAILURE_FILE.exists()

assert len(
    tournament_df
) == len(
    full_schedule
), (
    "One or more scheduled tournament "
    "matches failed."
)

assert len(
    tournament_failures
) == 0

assert overall_fallbacks == 0

assert tournament_df[
    "agent_result"
].between(
    0.0,
    1.0,
).all()

assert set(
    baseline_summary_df[
        "baseline_name"
    ]
) == set(
    baseline_factories.keys()
)

assert (
    side_summary_df[
        "evaluated_agent_side"
    ]
    .isin(
        [
            "Player",
            "Opponent",
        ]
    )
    .all()
)

print()
print(
    "âœ… SECTION 4 FULL BASELINE TOURNAMENT PASSED"
)


# ## Section 5 â€” Elo Ratings and Side-Bias Analysis
# 
# #### Calculate Elo ratings from the completed tournament matches.
# 
# #### This section reports:
# 
# - sequential Elo ratings;
# - Elo by baseline pairing;
# - Player-side and Opponent-side performance;
# - estimated first-player advantage;
# - side-adjusted policy strength;
# - confidence intervals for observed win rates.

# In[7]:


# ============================================================
# NOTEBOOK 48
# SECTION 5 â€” ELO RATINGS AND SIDE-BIAS ANALYSIS
# ============================================================


import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 5.3 â€” Output files
# ------------------------------------------------------------

ELO_HISTORY_FILE = (
    TOURNAMENT_DATA_DIR
    / "competitive_policy_elo_history.csv"
)

ELO_SUMMARY_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "competitive_policy_elo_summary.json"
)

SIDE_BIAS_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "competitive_policy_side_bias.csv"
)


# ------------------------------------------------------------
# 5.4 â€” Elo helpers
# ------------------------------------------------------------

def expected_elo_score(
    rating_a: float,
    rating_b: float,
) -> float:
    return 1.0 / (
        1.0
        + 10.0 ** (
            (
                rating_b
                - rating_a
            )
            / 400.0
        )
    )


def update_elo_pair(
    rating_a: float,
    rating_b: float,
    result_a: float,
    k_factor: float,
) -> tuple[float, float]:
    expected_a = expected_elo_score(
        rating_a,
        rating_b,
    )

    expected_b = 1.0 - expected_a

    result_b = 1.0 - result_a

    updated_a = (
        rating_a
        + k_factor
        * (
            result_a
            - expected_a
        )
    )

    updated_b = (
        rating_b
        + k_factor
        * (
            result_b
            - expected_b
        )
    )

    return (
        float(updated_a),
        float(updated_b),
    )


def elo_difference_from_score(
    score: float,
) -> float:
    clipped_score = float(
        np.clip(
            score,
            1e-6,
            1.0 - 1e-6,
        )
    )

    return float(
        400.0
        * math.log10(
            clipped_score
            / (
                1.0
                - clipped_score
            )
        )
    )


def wilson_interval(
    wins: int,
    matches: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    if matches <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    proportion = (
        wins
        / matches
    )

    denominator = (
        1.0
        + (
            z_value ** 2
            / matches
        )
    )

    center = (
        proportion
        + (
            z_value ** 2
            / (
                2.0
                * matches
            )
        )
    ) / denominator

    margin = (
        z_value
        * math.sqrt(
            (
                proportion
                * (
                    1.0
                    - proportion
                )
                / matches
            )
            + (
                z_value ** 2
                / (
                    4.0
                    * matches ** 2
                )
            )
        )
        / denominator
    )

    return (
        float(
            max(
                0.0,
                center - margin,
            )
        ),
        float(
            min(
                1.0,
                center + margin,
            )
        ),
    )


# ------------------------------------------------------------
# 5.5 â€” Initialize ratings
# ------------------------------------------------------------

agent_names = [
    "Competitive Policy",
    *sorted(
        baseline_factories.keys()
    ),
]

elo_ratings = {
    agent_name:
        float(INITIAL_ELO)
    for agent_name in agent_names
}

elo_history_rows = []


# ------------------------------------------------------------
# 5.6 â€” Sequential Elo calculation
# ------------------------------------------------------------

ordered_tournament_df = (
    tournament_df
    .sort_values(
        "match_id"
    )
    .reset_index(
        drop=True
    )
)

for _, match_row in (
    ordered_tournament_df.iterrows()
):
    competitive_name = (
        "Competitive Policy"
    )

    baseline_name = str(
        match_row[
            "baseline_name"
        ]
    )

    competitive_result = float(
        match_row[
            "agent_result"
        ]
    )

    competitive_before = (
        elo_ratings[
            competitive_name
        ]
    )

    baseline_before = (
        elo_ratings[
            baseline_name
        ]
    )

    competitive_after, baseline_after = (
        update_elo_pair(
            rating_a=
                competitive_before,

            rating_b=
                baseline_before,

            result_a=
                competitive_result,

            k_factor=
                ELO_K_FACTOR,
        )
    )

    elo_ratings[
        competitive_name
    ] = competitive_after

    elo_ratings[
        baseline_name
    ] = baseline_after

    elo_history_rows.append(
        {
            "match_id":
                int(
                    match_row[
                        "match_id"
                    ]
                ),

            "baseline_name":
                baseline_name,

            "evaluated_agent_side":
                str(
                    match_row[
                        "evaluated_agent_side"
                    ]
                ),

            "result":
                competitive_result,

            "competitive_elo_before":
                competitive_before,

            "baseline_elo_before":
                baseline_before,

            "competitive_elo_after":
                competitive_after,

            "baseline_elo_after":
                baseline_after,
        }
    )


elo_history_df = pd.DataFrame(
    elo_history_rows
)

elo_history_df.to_csv(
    ELO_HISTORY_FILE,
    index=False,
)


# ------------------------------------------------------------
# 5.7 â€” Observed rating differences by baseline
# ------------------------------------------------------------

pairing_rating_rows = []

for _, summary_row in (
    baseline_summary_df.iterrows()
):
    score = float(
        summary_row[
            "score"
        ]
    )

    match_count = int(
        summary_row[
            "matches"
        ]
    )

    wins = int(
        summary_row[
            "competitive_wins"
        ]
    )

    confidence_low, confidence_high = (
        wilson_interval(
            wins=wins,
            matches=match_count,
        )
    )

    pairing_rating_rows.append(
        {
            "baseline_name":
                str(
                    summary_row[
                        "baseline_name"
                    ]
                ),

            "matches":
                match_count,

            "wins":
                wins,

            "losses":
                int(
                    summary_row[
                        "competitive_losses"
                    ]
                ),

            "score":
                score,

            "observed_elo_difference":
                elo_difference_from_score(
                    score
                ),

            "win_rate_ci_low":
                confidence_low,

            "win_rate_ci_high":
                confidence_high,
        }
    )


pairing_rating_df = pd.DataFrame(
    pairing_rating_rows
)


# ------------------------------------------------------------
# 5.8 â€” Side-specific statistics
# ------------------------------------------------------------

side_bias_rows = []

for (
    baseline_name,
    evaluated_side,
), side_group in (
    tournament_df.groupby(
        [
            "baseline_name",
            "evaluated_agent_side",
        ]
    )
):
    matches = len(
        side_group
    )

    wins = int(
        side_group[
            "competitive_win"
        ].sum()
    )

    score = float(
        side_group[
            "agent_result"
        ].mean()
    )

    confidence_low, confidence_high = (
        wilson_interval(
            wins=wins,
            matches=matches,
        )
    )

    side_bias_rows.append(
        {
            "baseline_name":
                str(
                    baseline_name
                ),

            "evaluated_agent_side":
                str(
                    evaluated_side
                ),

            "matches":
                int(matches),

            "wins":
                wins,

            "losses":
                int(
                    side_group[
                        "competitive_loss"
                    ].sum()
                ),

            "score":
                score,

            "observed_elo_difference":
                elo_difference_from_score(
                    score
                ),

            "win_rate_ci_low":
                confidence_low,

            "win_rate_ci_high":
                confidence_high,

            "average_turns":
                float(
                    side_group[
                        "turns"
                    ].mean()
                ),
        }
    )


side_bias_df = pd.DataFrame(
    side_bias_rows
)

side_bias_df.to_csv(
    SIDE_BIAS_FILE,
    index=False,
)


# ------------------------------------------------------------
# 5.9 â€” Estimate Player-side advantage
# ------------------------------------------------------------

player_side_group = tournament_df[
    tournament_df[
        "evaluated_agent_side"
    ]
    == "Player"
]

opponent_side_group = tournament_df[
    tournament_df[
        "evaluated_agent_side"
    ]
    == "Opponent"
]

player_side_score = float(
    player_side_group[
        "agent_result"
    ].mean()
)

opponent_side_score = float(
    opponent_side_group[
        "agent_result"
    ].mean()
)

player_side_elo = (
    elo_difference_from_score(
        player_side_score
    )
)

opponent_side_elo = (
    elo_difference_from_score(
        opponent_side_score
    )
)

estimated_side_advantage_elo = float(
    (
        player_side_elo
        - opponent_side_elo
    )
    / 2.0
)

side_neutral_score = float(
    (
        player_side_score
        + opponent_side_score
    )
    / 2.0
)

side_neutral_elo_difference = (
    elo_difference_from_score(
        side_neutral_score
    )
)


# ------------------------------------------------------------
# 5.10 â€” Final Elo standings
# ------------------------------------------------------------

final_elo_df = pd.DataFrame(
    [
        {
            "agent_name":
                agent_name,

            "final_elo":
                float(
                    rating
                ),

            "elo_change":
                float(
                    rating
                    - INITIAL_ELO
                ),
        }
        for agent_name, rating in (
            elo_ratings.items()
        )
    ]
).sort_values(
    "final_elo",
    ascending=False,
).reset_index(
    drop=True
)

final_elo_df[
    "rank"
] = (
    np.arange(
        1,
        len(final_elo_df) + 1,
    )
)

final_elo_df = final_elo_df[
    [
        "rank",
        "agent_name",
        "final_elo",
        "elo_change",
    ]
]


# ------------------------------------------------------------
# 5.11 â€” Save Elo summary
# ------------------------------------------------------------

elo_summary = {
    "initial_elo":
        INITIAL_ELO,

    "k_factor":
        ELO_K_FACTOR,

    "matches_processed":
        int(
            len(
                elo_history_df
            )
        ),

    "final_ratings":
        final_elo_df
        .to_dict(
            orient="records"
        ),

    "pairing_rating_differences":
        pairing_rating_df
        .to_dict(
            orient="records"
        ),

    "player_side_score":
        player_side_score,

    "opponent_side_score":
        opponent_side_score,

    "player_side_elo_difference":
        player_side_elo,

    "opponent_side_elo_difference":
        opponent_side_elo,

    "estimated_player_side_advantage_elo":
        estimated_side_advantage_elo,

    "side_neutral_score":
        side_neutral_score,

    "side_neutral_elo_difference":
        side_neutral_elo_difference,

    "raw_overall_score":
        overall_score,

    "raw_overall_elo_difference":
        elo_difference_from_score(
            overall_score
        ),

    "elo_history_file":
        str(
            ELO_HISTORY_FILE
        ),

    "side_bias_file":
        str(
            SIDE_BIAS_FILE
        ),
}

with open(
    ELO_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        elo_summary,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 5.12 â€” Display Elo and side-bias results
# ------------------------------------------------------------

print(
    "NOTEBOOK 48 â€” ELO AND SIDE-BIAS ANALYSIS"
)

print("=" * 110)

print()
print(
    "Matches processed           :",
    len(
        elo_history_df
    ),
)

print(
    "Competitive Player score    :",
    round(
        player_side_score,
        4,
    ),
)

print(
    "Competitive Opponent score  :",
    round(
        opponent_side_score,
        4,
    ),
)

print(
    "Side-neutral score          :",
    round(
        side_neutral_score,
        4,
    ),
)

print(
    "Raw overall Elo difference  :",
    round(
        elo_difference_from_score(
            overall_score
        ),
        2,
    ),
)

print(
    "Side-neutral Elo difference :",
    round(
        side_neutral_elo_difference,
        2,
    ),
)

print(
    "Estimated side advantage Elo:",
    round(
        estimated_side_advantage_elo,
        2,
    ),
)

print()
print(
    "FINAL SEQUENTIAL ELO RATINGS"
)

print("-" * 110)

display(
    final_elo_df
)

print()
print(
    "OBSERVED STRENGTH BY BASELINE"
)

print("-" * 110)

display(
    pairing_rating_df
)

print()
print(
    "SIDE-BIAS RESULTS"
)

print("-" * 110)

display(
    side_bias_df
)

print()
print(
    "Elo history:",
    ELO_HISTORY_FILE,
)

print(
    "Elo summary:",
    ELO_SUMMARY_FILE,
)

print(
    "Side-bias report:",
    SIDE_BIAS_FILE,
)


# ------------------------------------------------------------
# 5.13 â€” Final validation
# ------------------------------------------------------------

assert ELO_HISTORY_FILE.exists()

assert ELO_SUMMARY_FILE.exists()

assert SIDE_BIAS_FILE.exists()

assert len(
    elo_history_df
) == len(
    tournament_df
)

assert set(
    final_elo_df[
        "agent_name"
    ]
) == set(
    agent_names
)

assert math.isfinite(
    side_neutral_elo_difference
)

assert math.isfinite(
    estimated_side_advantage_elo
)

assert (
    0.0
    <= player_side_score
    <= 1.0
)

assert (
    0.0
    <= opponent_side_score
    <= 1.0
)

print()
print(
    "âœ… SECTION 5 ELO AND SIDE-BIAS ANALYSIS PASSED"
)


# ## Section 6 â€” Final Competitive Strength Report
# 
# #### Consolidate training, test, tournament, Elo, and side-bias results into one competition-readiness report.
# 
# ##### This section determines:
# 
# - whether the competitive policy should replace the old bootstrap policy;
# - its current measured strength;
# - its principal weaknesses;
# - the highest-value improvements for Notebook 49.

# In[8]:


# ============================================================
# NOTEBOOK 48
# SECTION 6 â€” FINAL COMPETITIVE STRENGTH REPORT
# ============================================================


import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


# ------------------------------------------------------------
# 6.3 â€” Output files
# ------------------------------------------------------------

FINAL_STRENGTH_REPORT_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "final_competitive_strength_report.json"
)

FINAL_STRENGTH_TABLE_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "final_competitive_strength_table.csv"
)

FINAL_RECOMMENDATION_FILE = (
    NOTEBOOK48_REPORT_DIR
    / "final_competitive_recommendation.txt"
)


# ------------------------------------------------------------
# 6.4 â€” Collect Notebook 47 metrics
# ------------------------------------------------------------

test_accuracy = float(
    notebook47_test_report.get(
        "test_top1_accuracy",
        0.0,
    )
)

test_multi_action_accuracy = float(
    notebook47_test_report.get(
        "test_multi_action_accuracy",
        0.0,
    )
)

test_loss = float(
    notebook47_test_report.get(
        "test_loss",
        float("nan"),
    )
)

training_best_accuracy = float(
    notebook47_training_summary.get(
        "best_validation_accuracy",
        0.0,
    )
)

training_best_multi_accuracy = float(
    notebook47_training_summary.get(
        "best_multi_action_accuracy",
        0.0,
    )
)

best_epoch = int(
    notebook47_training_summary.get(
        "best_epoch",
        0,
    )
)


# ------------------------------------------------------------
# 6.5 â€” Collect tournament metrics
# ------------------------------------------------------------

random_row = baseline_summary_df[
    baseline_summary_df[
        "baseline_name"
    ]
    == "Random Baseline"
].iloc[0]

greedy_row = baseline_summary_df[
    baseline_summary_df[
        "baseline_name"
    ]
    == "Greedy Damage Baseline"
].iloc[0]

search_row = baseline_summary_df[
    baseline_summary_df[
        "baseline_name"
    ]
    == "Depth-6 Search Baseline"
].iloc[0]

random_win_rate = float(
    random_row[
        "win_rate"
    ]
)

greedy_win_rate = float(
    greedy_row[
        "win_rate"
    ]
)

search_win_rate = float(
    search_row[
        "win_rate"
    ]
)

final_competitive_elo = float(
    final_elo_df[
        final_elo_df[
            "agent_name"
        ]
        == "Competitive Policy"
    ][
        "final_elo"
    ].iloc[0]
)

final_competitive_rank = int(
    final_elo_df[
        final_elo_df[
            "agent_name"
        ]
        == "Competitive Policy"
    ][
        "rank"
    ].iloc[0]
)


# ------------------------------------------------------------
# 6.6 â€” Strength classification
# ------------------------------------------------------------

if (
    search_win_rate >= 0.60
    and test_multi_action_accuracy >= 0.80
):
    strength_tier = (
        "Highly Competitive"
    )

elif (
    search_win_rate >= 0.50
    and test_multi_action_accuracy >= 0.70
):
    strength_tier = (
        "Competitive"
    )

elif (
    random_win_rate >= 0.60
):
    strength_tier = (
        "Developing Competitive"
    )

else:
    strength_tier = (
        "Prototype"
    )


# ------------------------------------------------------------
# 6.7 â€” Replacement decision
# ------------------------------------------------------------

replace_bootstrap_policy = bool(
    test_accuracy >= 0.75
    and test_multi_action_accuracy >= 0.70
    and random_win_rate >= 0.65
    and search_win_rate >= 0.50
    and overall_fallbacks == 0
)

replacement_reason = (
    "The trained variable-action policy should replace "
    "the earlier bootstrap PPO policy. It was trained on "
    "real depth-4 search demonstrations, passed untouched "
    "test evaluation, completed 300 tournament matches "
    "without fallback, and achieved a positive score against "
    "the depth-6 search baseline."
    if replace_bootstrap_policy
    else
    "The trained policy should not yet replace the bootstrap "
    "policy because one or more readiness thresholds were not met."
)


# ------------------------------------------------------------
# 6.8 â€” Identify weaknesses
# ------------------------------------------------------------

identified_weaknesses = []

if opponent_side_score < 0.50:
    identified_weaknesses.append(
        {
            "issue":
                "Opponent-side weakness",

            "evidence":
                (
                    f"Opponent-side score was "
                    f"{opponent_side_score:.4f}."
                ),

            "priority":
                "Critical",

            "recommended_fix":
                (
                    "Generate more actor-relative expert positions "
                    "where the trained policy acts from the Opponent "
                    "side, and rebalance training batches by side."
                ),
        }
    )

if test_multi_action_accuracy < 0.85:
    identified_weaknesses.append(
        {
            "issue":
                "Multi-action imitation gap",

            "evidence":
                (
                    f"Untouched-test multi-action accuracy was "
                    f"{test_multi_action_accuracy:.4f}."
                ),

            "priority":
                "High",

            "recommended_fix":
                (
                    "Increase expert dataset size and oversample "
                    "positions containing two or more legal actions."
                ),
        }
    )

if search_win_rate < 0.65:
    identified_weaknesses.append(
        {
            "issue":
                "Search-baseline margin is modest",

            "evidence":
                (
                    f"Win rate against the depth-6 search baseline "
                    f"was {search_win_rate:.4f}."
                ),

            "priority":
                "High",

            "recommended_fix":
                (
                    "Use search-guided self-play and policy distillation "
                    "from deeper or time-limited search."
                ),
        }
    )

if estimated_side_advantage_elo > 100:
    identified_weaknesses.append(
        {
            "issue":
                "Large first-player advantage",

            "evidence":
                (
                    f"Estimated side advantage was "
                    f"{estimated_side_advantage_elo:.2f} Elo."
                ),

            "priority":
                "High",

            "recommended_fix":
                (
                    "Evaluate balanced mirrored states and train with "
                    "equal Player/Opponent examples for every matchup."
                ),
        }
    )


# ------------------------------------------------------------
# 6.9 â€” Competitive scorecard
# ------------------------------------------------------------

scorecard_rows = [
    {
        "metric":
            "Validation accuracy",

        "value":
            training_best_accuracy,

        "target":
            0.80,

        "passed":
            training_best_accuracy
            >= 0.80,
    },

    {
        "metric":
            "Validation multi-action accuracy",

        "value":
            training_best_multi_accuracy,

        "target":
            0.75,

        "passed":
            training_best_multi_accuracy
            >= 0.75,
    },

    {
        "metric":
            "Test accuracy",

        "value":
            test_accuracy,

        "target":
            0.80,

        "passed":
            test_accuracy
            >= 0.80,
    },

    {
        "metric":
            "Test multi-action accuracy",

        "value":
            test_multi_action_accuracy,

        "target":
            0.75,

        "passed":
            test_multi_action_accuracy
            >= 0.75,
    },

    {
        "metric":
            "Random baseline win rate",

        "value":
            random_win_rate,

        "target":
            0.70,

        "passed":
            random_win_rate
            >= 0.70,
    },

    {
        "metric":
            "Greedy baseline win rate",

        "value":
            greedy_win_rate,

        "target":
            0.50,

        "passed":
            greedy_win_rate
            >= 0.50,
    },

    {
        "metric":
            "Depth-6 search win rate",

        "value":
            search_win_rate,

        "target":
            0.50,

        "passed":
            search_win_rate
            >= 0.50,
    },

    {
        "metric":
            "Opponent-side score",

        "value":
            opponent_side_score,

        "target":
            0.50,

        "passed":
            opponent_side_score
            >= 0.50,
    },

    {
        "metric":
            "Fallback-free tournament",

        "value":
            float(
                overall_fallbacks
                == 0
            ),

        "target":
            1.0,

        "passed":
            overall_fallbacks
            == 0,
    },
]

scorecard_df = pd.DataFrame(
    scorecard_rows
)

scorecard_df.to_csv(
    FINAL_STRENGTH_TABLE_FILE,
    index=False,
)

passed_scorecard_items = int(
    scorecard_df[
        "passed"
    ].sum()
)

total_scorecard_items = int(
    len(
        scorecard_df
    )
)

readiness_percentage = float(
    passed_scorecard_items
    / total_scorecard_items
)


# ------------------------------------------------------------
# 6.10 â€” Final report
# ------------------------------------------------------------

final_strength_report = {
    "policy_name":
        "Competitive Policy",

    "strength_tier":
        strength_tier,

    "replace_bootstrap_policy":
        replace_bootstrap_policy,

    "replacement_reason":
        replacement_reason,

    "training": {
        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            training_best_accuracy,

        "best_validation_multi_action_accuracy":
            training_best_multi_accuracy,
    },

    "test": {
        "records":
            int(
                notebook47_test_report.get(
                    "test_records",
                    0,
                )
            ),

        "loss":
            test_loss,

        "top1_accuracy":
            test_accuracy,

        "multi_action_accuracy":
            test_multi_action_accuracy,
    },

    "tournament": {
        "matches":
            int(
                overall_matches
            ),

        "wins":
            overall_wins,

        "losses":
            overall_losses,

        "draws":
            overall_draws,

        "overall_win_rate":
            overall_win_rate,

        "random_baseline_win_rate":
            random_win_rate,

        "greedy_baseline_win_rate":
            greedy_win_rate,

        "depth6_search_win_rate":
            search_win_rate,

        "fallback_count":
            overall_fallbacks,
    },

    "elo": {
        "final_sequential_elo":
            final_competitive_elo,

        "final_rank":
            final_competitive_rank,

        "raw_elo_difference":
            elo_difference_from_score(
                overall_score
            ),

        "side_neutral_elo_difference":
            side_neutral_elo_difference,

        "estimated_side_advantage_elo":
            estimated_side_advantage_elo,
    },

    "side_performance": {
        "player_score":
            player_side_score,

        "opponent_score":
            opponent_side_score,

        "side_neutral_score":
            side_neutral_score,
    },

    "readiness": {
        "passed_items":
            passed_scorecard_items,

        "total_items":
            total_scorecard_items,

        "readiness_percentage":
            readiness_percentage,
    },

    "identified_weaknesses":
        identified_weaknesses,

    "recommended_next_notebook":
        (
            "49_side_balanced_search_guided_self_play.ipynb"
        ),
}

with open(
    FINAL_STRENGTH_REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_strength_report,
        file,
        indent=4,
        default=str,
    )


# ------------------------------------------------------------
# 6.11 â€” Recommendation text
# ------------------------------------------------------------

recommendation_text = f"""
NOTEBOOK 48 â€” FINAL COMPETITIVE RECOMMENDATION
============================================================

Policy: Competitive Policy
Strength tier: {strength_tier}

Replace bootstrap policy: {replace_bootstrap_policy}

{replacement_reason}

Measured results
----------------
Validation accuracy: {training_best_accuracy:.4f}
Validation multi-action accuracy: {training_best_multi_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}
Test multi-action accuracy: {test_multi_action_accuracy:.4f}

Tournament matches: {overall_matches}
Overall win rate: {overall_win_rate:.4f}
Random baseline win rate: {random_win_rate:.4f}
Greedy baseline win rate: {greedy_win_rate:.4f}
Depth-6 search win rate: {search_win_rate:.4f}

Final sequential Elo: {final_competitive_elo:.2f}
Side-neutral Elo difference: {side_neutral_elo_difference:.2f}
Estimated first-player advantage: {estimated_side_advantage_elo:.2f} Elo

Player-side score: {player_side_score:.4f}
Opponent-side score: {opponent_side_score:.4f}

Readiness score: {passed_scorecard_items}/{total_scorecard_items}
Readiness percentage: {readiness_percentage:.2%}

Recommended next notebook
-------------------------
49_side_balanced_search_guided_self_play.ipynb
""".strip()

FINAL_RECOMMENDATION_FILE.write_text(
    recommendation_text,
    encoding="utf-8",
)


# ------------------------------------------------------------
# 6.12 â€” Display final report
# ------------------------------------------------------------

print(
    "NOTEBOOK 48 â€” FINAL COMPETITIVE STRENGTH REPORT"
)

print("=" * 110)

print()
print(
    "Strength tier              :",
    strength_tier,
)

print(
    "Replace bootstrap policy   :",
    replace_bootstrap_policy,
)

print(
    "Readiness score            :",
    (
        f"{passed_scorecard_items}/"
        f"{total_scorecard_items}"
    ),
)

print(
    "Readiness percentage       :",
    f"{readiness_percentage:.2%}",
)

print(
    "Final sequential Elo       :",
    round(
        final_competitive_elo,
        2,
    ),
)

print(
    "Side-neutral Elo difference:",
    round(
        side_neutral_elo_difference,
        2,
    ),
)

print()
print(
    "COMPETITIVE SCORECARD"
)

print("-" * 110)

display(
    scorecard_df
)

print()
print(
    "IDENTIFIED WEAKNESSES"
)

print("-" * 110)

if identified_weaknesses:
    display(
        pd.DataFrame(
            identified_weaknesses
        )
    )

else:
    print(
        "No major weaknesses detected."
    )

print()
print(
    "Final report:",
    FINAL_STRENGTH_REPORT_FILE,
)

print(
    "Scorecard:",
    FINAL_STRENGTH_TABLE_FILE,
)

print(
    "Recommendation:",
    FINAL_RECOMMENDATION_FILE,
)

print()
print(recommendation_text)


# ------------------------------------------------------------
# 6.13 â€” Final validation
# ------------------------------------------------------------

assert FINAL_STRENGTH_REPORT_FILE.exists()

assert FINAL_STRENGTH_TABLE_FILE.exists()

assert FINAL_RECOMMENDATION_FILE.exists()

assert replace_bootstrap_policy is True

assert strength_tier in {
    "Highly Competitive",
    "Competitive",
    "Developing Competitive",
    "Prototype",
}

assert passed_scorecard_items >= 7

assert overall_fallbacks == 0

print()
print(
    "âœ… SECTION 6 FINAL COMPETITIVE REPORT PASSED"
)


# In[ ]:





