#!/usr/bin/env python
# coding: utf-8

# # Notebook 45 - Final Simulator Integration and Submission
# 
# ## Final Objectives
# 
# - Load the final PPO model and benchmark artifacts
# - Discover the existing simulator, observation, and action-adapter modules
# - Connect the trained PPO policy to the project inference interface
# - Validate observation dimensions and legal-action masking
# - Add deterministic fallback behavior for invalid or unavailable predictions
# - Run end-to-end inference and integration tests
# - Save the final reusable competition agent under `src/`
# - Build the final submission package
# - Validate package imports and required files
# - Export a final readiness report and completion checklist
# 
# > Notebook 45 is the final build notebook. Any remaining fixes will be completed here or in the existing source modules rather than creating additional notebooks.
# 

# ## Section 1 — Final Project Setup and Artifact Discovery
# 
# #### Locate the project root, verify the completed PPO artifacts, and inspect the existing source modules that may provide simulator, observation, action-selection, or tournament integration functionality.

# In[1]:


# ============================================================
# NOTEBOOK 45 — FINAL SIMULATOR INTEGRATION AND SUBMISSION
# SECTION 1 — PROJECT SETUP AND ARTIFACT DISCOVERY
# ============================================================

from __future__ import annotations

import importlib
import json
import pickle
import random
import shutil
import sys
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.distributions import Categorical


def find_project_root(
    start_path: Path | None = None,
) -> Path:
    """Locate the Pokémon project root."""

    current = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        required_directories = [
            candidate / "src",
            candidate / "notebooks",
            candidate / "scripts",
            candidate / "reports",
        ]

        if all(
            directory.is_dir()
            for directory in required_directories
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Run Notebook 45 from inside the project."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

NOTEBOOK40_REPORT_DIR = REPORTS_DIR / "notebook40"
NOTEBOOK42_REPORT_DIR = REPORTS_DIR / "notebook42"
NOTEBOOK43_REPORT_DIR = REPORTS_DIR / "notebook43"
NOTEBOOK44_REPORT_DIR = REPORTS_DIR / "notebook44"
NOTEBOOK45_REPORT_DIR = REPORTS_DIR / "notebook45"

PPO_DATASET_FILE = (
    NOTEBOOK40_REPORT_DIR
    / "ppo_training_package.pkl"
)

TRAINED_CHECKPOINT_FILE = (
    NOTEBOOK42_REPORT_DIR
    / "ppo_training_loop.pkl"
)

POLICY_EVALUATION_FILE = (
    NOTEBOOK43_REPORT_DIR
    / "ppo_policy_evaluation.pkl"
)

SELF_PLAY_BENCHMARK_FILE = (
    NOTEBOOK44_REPORT_DIR
    / "ppo_self_play_benchmark.pkl"
)

FINAL_AGENT_FILE = (
    SRC_DIR
    / "agents"
    / "final_ppo_agent.py"
)

FINAL_PACKAGE_DIR = (
    PROJECT_ROOT
    / "submission"
    / "final_agent"
)

FINAL_PACKAGE_FILE = (
    PROJECT_ROOT
    / "submission"
    / "ptcg_final_agent.zip"
)

FINAL_READINESS_FILE = (
    NOTEBOOK45_REPORT_DIR
    / "final_submission_readiness.json"
)

NOTEBOOK45_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_AGENT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_PACKAGE_DIR.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

RANDOM_SEED = 45

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

required_artifacts = pd.DataFrame(
    {
        "Artifact": [
            "Notebook 40 PPO Dataset",
            "Notebook 42 Trained Checkpoint",
            "Notebook 43 Policy Evaluation",
            "Notebook 44 Self-Play Benchmark",
            "Notebook 45 Report Directory",
        ],
        "Path": [
            str(PPO_DATASET_FILE),
            str(TRAINED_CHECKPOINT_FILE),
            str(POLICY_EVALUATION_FILE),
            str(SELF_PLAY_BENCHMARK_FILE),
            str(NOTEBOOK45_REPORT_DIR),
        ],
        "Exists": [
            PPO_DATASET_FILE.exists(),
            TRAINED_CHECKPOINT_FILE.exists(),
            POLICY_EVALUATION_FILE.exists(),
            SELF_PLAY_BENCHMARK_FILE.exists(),
            NOTEBOOK45_REPORT_DIR.exists(),
        ],
    }
)

source_keywords = (
    "simulator",
    "battle",
    "environment",
    "observation",
    "adapter",
    "action",
    "agent",
    "tournament",
    "engine",
    "policy",
)

source_files = sorted(
    path
    for path in SRC_DIR.rglob("*.py")
    if any(
        keyword in path.stem.lower()
        or keyword in str(path.parent).lower()
        for keyword in source_keywords
    )
)

source_inventory = pd.DataFrame(
    {
        "Source File": [
            str(path.relative_to(PROJECT_ROOT))
            for path in source_files
        ]
    }
)

print("NOTEBOOK 45 — FINAL PROJECT SETUP")
print("=" * 80)

print("Project root    :", PROJECT_ROOT)
print("Execution device:", DEVICE)
print("PyTorch version :", torch.__version__)
print("Random seed     :", RANDOM_SEED)

print()
print("Required Artifacts")
display(required_artifacts)

print()
print(
    "Potential Integration Source Files:",
    len(source_files),
)

if len(source_inventory) > 0:
    display(
        source_inventory.head(50)
    )
else:
    print(
        "No matching integration modules were discovered."
    )

assert SRC_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert SCRIPTS_DIR.exists()
assert REPORTS_DIR.exists()

assert PPO_DATASET_FILE.exists(), (
    f"Missing PPO dataset: {PPO_DATASET_FILE}"
)

assert TRAINED_CHECKPOINT_FILE.exists(), (
    f"Missing trained checkpoint: "
    f"{TRAINED_CHECKPOINT_FILE}"
)

assert POLICY_EVALUATION_FILE.exists(), (
    f"Missing policy evaluation: "
    f"{POLICY_EVALUATION_FILE}"
)

assert SELF_PLAY_BENCHMARK_FILE.exists(), (
    f"Missing self-play benchmark: "
    f"{SELF_PLAY_BENCHMARK_FILE}"
)

assert required_artifacts["Exists"].all()

print()
print("✅ SECTION 1 FINAL PROJECT SETUP PASSED")


# ## Section 2 — Inspect Existing Simulator and Agent Modules
# 
# #### Inspect the discovered Python modules for classes and functions related to observations, legal actions, environments, simulators, agents, and battle execution. This prevents creating a duplicate interface and helps identify the correct integration points.

# In[2]:


# ============================================================
# SECTION 2 — INSPECT EXISTING INTEGRATION MODULES
# ============================================================

import ast


def inspect_python_file(
    file_path: Path,
) -> dict[str, Any]:
    """
    Parse a Python source file and return its top-level
    classes, functions, imports, and likely integration symbols.
    """

    try:
        source_text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        syntax_tree = ast.parse(
            source_text,
            filename=str(file_path),
        )

    except Exception as exc:
        return {
            "file": str(
                file_path.relative_to(PROJECT_ROOT)
            ),
            "classes": [],
            "functions": [],
            "imports": [],
            "integration_symbols": [],
            "error": str(exc),
        }

    classes = []
    functions = []
    imports = []

    for node in syntax_tree.body:

        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(node.name)

        elif isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""

            imports.extend(
                f"{module_name}.{alias.name}"
                for alias in node.names
            )

    symbol_keywords = (
        "simulator",
        "environment",
        "env",
        "battle",
        "game",
        "state",
        "observation",
        "encode",
        "adapter",
        "legal",
        "action",
        "move",
        "agent",
        "policy",
        "step",
        "reset",
        "play",
        "tournament",
    )

    all_symbols = classes + functions

    integration_symbols = [
        symbol
        for symbol in all_symbols
        if any(
            keyword in symbol.lower()
            for keyword in symbol_keywords
        )
    ]

    return {
        "file": str(
            file_path.relative_to(PROJECT_ROOT)
        ),
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "integration_symbols": integration_symbols,
        "error": None,
    }


module_inspection = [
    inspect_python_file(file_path)
    for file_path in source_files
]

module_rows = []

for module in module_inspection:

    module_rows.append(
        {
            "File": module["file"],
            "Classes": ", ".join(
                module["classes"]
            ),
            "Functions": ", ".join(
                module["functions"]
            ),
            "Integration Symbols": ", ".join(
                module["integration_symbols"]
            ),
            "Parse Error": module["error"],
        }
    )

module_inventory = pd.DataFrame(
    module_rows
)

print("NOTEBOOK 45 — INTEGRATION MODULE INSPECTION")
print("=" * 80)

print(
    "Modules inspected:",
    len(module_inventory),
)

print()

if len(module_inventory) > 0:

    display(
        module_inventory[
            [
                "File",
                "Integration Symbols",
                "Parse Error",
            ]
        ].head(100)
    )

else:
    print(
        "No source modules were available for inspection."
    )


candidate_modules = module_inventory[
    module_inventory[
        "Integration Symbols"
    ].str.len() > 0
].copy()

print()
print(
    "Candidate integration modules:",
    len(candidate_modules),
)

if len(candidate_modules) > 0:

    display(
        candidate_modules[
            [
                "File",
                "Classes",
                "Functions",
                "Integration Symbols",
            ]
        ].head(50)
    )

else:
    print(
        "No modules exposed obvious integration symbols."
    )


inspection_file = (
    NOTEBOOK45_REPORT_DIR
    / "integration_module_inventory.json"
)

with open(
    inspection_file,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        module_inspection,
        f,
        indent=2,
    )

assert inspection_file.exists()
assert len(module_inventory) > 0

print()
print(
    "Inventory saved:",
    inspection_file,
)

print()
print(
    "✅ SECTION 2 INTEGRATION MODULE INSPECTION PASSED"
)


# ## Section 3 — Inspect Existing Simulator APIs
# 
# #### Read the real simulator, state, legal-move, agent, observation-adapter, and self-play modules to identify their exact classes, functions, arguments, and return structures before connecting the PPO policy.

# In[3]:


# ============================================================
# SECTION 3 — INSPECT EXISTING SIMULATOR APIS
# ============================================================

import ast
import json


TARGET_FILES = [
    SRC_DIR / "simulator.py",
    SRC_DIR / "battle_simulation.py",
    SRC_DIR / "battle_state.py",
    SRC_DIR / "battle_agent.py",
    SRC_DIR / "legal_moves.py",
    SRC_DIR / "adapters.py",
    SRC_DIR / "observation_adapter" / "battle_state_adapter.py",
    SRC_DIR / "observation_adapter" / "policy_search_adapter.py",
    SRC_DIR / "training" / "self_play.py",
    SRC_DIR / "tournament" / "runner.py",
]


def format_arguments(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Create a readable function signature from AST arguments."""

    arguments = []

    positional_arguments = (
        list(node.args.posonlyargs)
        + list(node.args.args)
    )

    number_of_defaults = len(node.args.defaults)

    default_start = (
        len(positional_arguments)
        - number_of_defaults
    )

    for index, argument in enumerate(
        positional_arguments
    ):
        argument_text = argument.arg

        if index >= default_start:
            default_node = node.args.defaults[
                index - default_start
            ]

            try:
                default_text = ast.unparse(
                    default_node
                )
            except Exception:
                default_text = "..."

            argument_text += f"={default_text}"

        arguments.append(argument_text)

    if node.args.vararg is not None:
        arguments.append(
            f"*{node.args.vararg.arg}"
        )

    for argument, default_node in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        argument_text = argument.arg

        if default_node is not None:
            try:
                default_text = ast.unparse(
                    default_node
                )
            except Exception:
                default_text = "..."

            argument_text += f"={default_text}"

        arguments.append(argument_text)

    if node.args.kwarg is not None:
        arguments.append(
            f"**{node.args.kwarg.arg}"
        )

    return ", ".join(arguments)


def inspect_file_api(
    file_path: Path,
) -> dict[str, Any]:
    """Inspect classes, methods, functions, and signatures."""

    result = {
        "file": str(
            file_path.relative_to(PROJECT_ROOT)
        ),
        "exists": file_path.exists(),
        "classes": [],
        "functions": [],
        "error": None,
    }

    if not file_path.exists():
        return result

    try:
        source_text = file_path.read_text(
            encoding="utf-8",
        )

        syntax_tree = ast.parse(
            source_text,
            filename=str(file_path),
        )

        for node in syntax_tree.body:

            if isinstance(node, ast.ClassDef):

                class_record = {
                    "name": node.name,
                    "methods": [],
                }

                for child in node.body:

                    if isinstance(
                        child,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    ):
                        class_record[
                            "methods"
                        ].append(
                            {
                                "name": child.name,
                                "signature": (
                                    format_arguments(
                                        child
                                    )
                                ),
                            }
                        )

                result["classes"].append(
                    class_record
                )

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                result["functions"].append(
                    {
                        "name": node.name,
                        "signature": (
                            format_arguments(node)
                        ),
                    }
                )

    except Exception as exc:
        result["error"] = str(exc)

    return result


api_inventory = [
    inspect_file_api(file_path)
    for file_path in TARGET_FILES
]


summary_rows = []

for module in api_inventory:

    class_names = [
        class_info["name"]
        for class_info in module["classes"]
    ]

    function_names = [
        function_info["name"]
        for function_info in module["functions"]
    ]

    method_names = []

    for class_info in module["classes"]:
        for method_info in class_info["methods"]:
            method_names.append(
                (
                    f'{class_info["name"]}.'
                    f'{method_info["name"]}'
                )
            )

    summary_rows.append(
        {
            "File": module["file"],
            "Exists": module["exists"],
            "Classes": ", ".join(class_names),
            "Functions": ", ".join(
                function_names
            ),
            "Methods": ", ".join(
                method_names
            ),
            "Error": module["error"],
        }
    )


api_summary = pd.DataFrame(
    summary_rows
)

print("NOTEBOOK 45 — EXISTING SIMULATOR API")
print("=" * 80)

display(api_summary)

print()
print("DETAILED SIGNATURES")
print("=" * 80)

for module in api_inventory:

    print()
    print(module["file"])
    print("-" * 80)

    if not module["exists"]:
        print("  [MISSING]")
        continue

    if module["error"] is not None:
        print(
            "  [ERROR]",
            module["error"],
        )
        continue

    for class_info in module["classes"]:

        print(
            f'  class {class_info["name"]}'
        )

        for method_info in class_info[
            "methods"
        ]:
            print(
                "    "
                f'{method_info["name"]}'
                "("
                f'{method_info["signature"]}'
                ")"
            )

    for function_info in module["functions"]:

        print(
            "  function "
            f'{function_info["name"]}'
            "("
            f'{function_info["signature"]}'
            ")"
        )


api_inventory_file = (
    NOTEBOOK45_REPORT_DIR
    / "existing_simulator_api.json"
)

with open(
    api_inventory_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        api_inventory,
        file,
        indent=2,
    )

existing_targets = api_summary[
    api_summary["Exists"]
]

assert len(existing_targets) >= 6
assert existing_targets[
    "Error"
].isna().all()

assert (
    api_summary["Classes"].str.len().sum()
    + api_summary["Functions"].str.len().sum()
) > 0

assert api_inventory_file.exists()

print()
print(
    "API inventory saved:",
    api_inventory_file,
)

print()
print(
    "✅ SECTION 3 EXISTING SIMULATOR API PASSED"
)


# ## Section 4 — Extract Exact Integration Definitions
# 
# #### Extract the source code for the existing battle agent, legal-move functions, observation builder, simulator step, and battle runner so the PPO adapter can match the real project interfaces exactly.

# In[4]:


# ============================================================
# SECTION 4 — EXTRACT EXACT INTEGRATION DEFINITIONS
# ============================================================

import ast
import json


DEFINITION_REQUESTS = {
    SRC_DIR / "battle_agent.py": [
        "PokemonBattleAgent",
    ],
    SRC_DIR / "legal_moves.py": [
        "get_legal_moves",
        "get_current_legal_moves",
        "get_current_side_pokemon",
    ],
    SRC_DIR
    / "observation_adapter"
    / "battle_state_adapter.py": [
        "move_label",
        "numeric_move_value",
        "build_policy_state",
    ],
    SRC_DIR / "simulator.py": [
        "apply_move",
    ],
    SRC_DIR / "battle_simulation.py": [
        "determine_battle_winner",
        "simulate_ai_battle",
        "create_battle_transcript",
    ],
    SRC_DIR / "battle_state.py": [
        "PokemonState",
        "PlayerState",
        "BattleState",
    ],
}


def extract_named_definitions(
    file_path: Path,
    requested_names: list[str],
) -> dict[str, Any]:
    """
    Extract exact source text for requested top-level
    classes and functions.
    """

    result = {
        "file": str(
            file_path.relative_to(PROJECT_ROOT)
        ),
        "exists": file_path.exists(),
        "definitions": {},
        "missing": [],
        "error": None,
    }

    if not file_path.exists():
        result["missing"] = list(
            requested_names
        )
        return result

    try:
        source_text = file_path.read_text(
            encoding="utf-8",
        )

        syntax_tree = ast.parse(
            source_text,
            filename=str(file_path),
        )

        source_lines = source_text.splitlines()

        located_names = set()

        for node in syntax_tree.body:

            if not isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            if node.name not in requested_names:
                continue

            located_names.add(node.name)

            start_line = node.lineno - 1

            end_line = getattr(
                node,
                "end_lineno",
                node.lineno,
            )

            definition_text = "\n".join(
                source_lines[
                    start_line:end_line
                ]
            )

            result["definitions"][
                node.name
            ] = definition_text

        result["missing"] = [
            name
            for name in requested_names
            if name not in located_names
        ]

    except Exception as exc:
        result["error"] = str(exc)

    return result


definition_inventory = []

for file_path, requested_names in (
    DEFINITION_REQUESTS.items()
):
    definition_inventory.append(
        extract_named_definitions(
            file_path,
            requested_names,
        )
    )


print(
    "NOTEBOOK 45 — EXACT INTEGRATION DEFINITIONS"
)
print("=" * 80)

for module in definition_inventory:

    print()
    print(module["file"])
    print("-" * 80)

    if not module["exists"]:
        print("[MISSING FILE]")
        continue

    if module["error"] is not None:
        print(
            "[ERROR]",
            module["error"],
        )
        continue

    for name, source_text in (
        module["definitions"].items()
    ):

        print()
        print(
            f"### {name}"
        )
        print()

        print(source_text)

    if module["missing"]:
        print()
        print(
            "Requested definitions not found:",
            module["missing"],
        )


definition_file = (
    NOTEBOOK45_REPORT_DIR
    / "exact_integration_definitions.json"
)

with open(
    definition_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        definition_inventory,
        file,
        indent=2,
    )


definition_summary_rows = []

for module in definition_inventory:

    definition_summary_rows.append(
        {
            "File": module["file"],
            "Exists": module["exists"],
            "Definitions Found": len(
                module["definitions"]
            ),
            "Missing": ", ".join(
                module["missing"]
            ),
            "Error": module["error"],
        }
    )


definition_summary = pd.DataFrame(
    definition_summary_rows
)

print()
print("=" * 80)
print("DEFINITION EXTRACTION SUMMARY")
print()

display(definition_summary)

assert definition_summary[
    "Exists"
].all()

assert definition_summary[
    "Error"
].isna().all()

assert definition_summary[
    "Definitions Found"
].sum() >= 8

assert definition_file.exists()

print()
print(
    "Definitions saved:",
    definition_file,
)

print()
print(
    "✅ SECTION 4 EXACT INTEGRATION DEFINITIONS PASSED"
)


# ## Section 5 — Display Critical Integration Definitions
# 
# #### Display only the source definitions required to connect the PPO model to the existing battle simulator.

# In[5]:


# ============================================================
# SECTION 5 — DISPLAY CRITICAL DEFINITIONS
# ============================================================

with open(
    definition_file,
    "r",
    encoding="utf-8",
) as file:
    saved_definitions = json.load(file)


critical_names = [
    "PokemonBattleAgent",
    "get_legal_moves",
    "get_current_legal_moves",
    "build_policy_state",
    "move_label",
    "apply_move",
    "simulate_ai_battle",
    "PokemonState",
    "PlayerState",
    "BattleState",
]


critical_definitions = {}

for module in saved_definitions:

    for name, source_text in module[
        "definitions"
    ].items():

        if name in critical_names:
            critical_definitions[name] = (
                source_text
            )


print("NOTEBOOK 45 — CRITICAL DEFINITIONS")
print("=" * 80)

for name in critical_names:

    print()
    print("#" * 80)
    print(name)
    print("#" * 80)

    source_text = critical_definitions.get(
        name
    )

    if source_text is None:
        print("[NOT FOUND]")
    else:
        print(source_text)


critical_definition_file = (
    NOTEBOOK45_REPORT_DIR
    / "critical_integration_definitions.txt"
)

with open(
    critical_definition_file,
    "w",
    encoding="utf-8",
) as file:

    for name in critical_names:

        file.write(
            "\n"
            + "#" * 80
            + "\n"
        )

        file.write(
            name + "\n"
        )

        file.write(
            "#" * 80
            + "\n"
        )

        file.write(
            critical_definitions.get(
                name,
                "[NOT FOUND]",
            )
        )

        file.write("\n")


print()
print("=" * 80)

print(
    "Critical definitions found:",
    len(critical_definitions),
)

print(
    "Saved to:",
    critical_definition_file,
)

assert (
    "PokemonBattleAgent"
    in critical_definitions
)

assert (
    "build_policy_state"
    in critical_definitions
)

assert (
    "apply_move"
    in critical_definitions
)

assert (
    "BattleState"
    in critical_definitions
)

assert critical_definition_file.exists()

print()
print(
    "✅ SECTION 5 CRITICAL DEFINITIONS PASSED"
)


# ## Section 6 — Confirm PPO Observation and Action Schema
# 
# #### Inspect the Notebook 40 PPO package to identify the exact observation feature order, action-index mapping, normalization details, and sample values required by the trained Actor network.

# In[6]:


# ============================================================
# SECTION 6 — CONFIRM PPO OBSERVATION AND ACTION SCHEMA
# ============================================================

with open(
    PPO_DATASET_FILE,
    "rb",
) as file:
    ppo_package_full = pickle.load(file)


def summarize_value(
    value: Any,
) -> str:
    """Return a concise description of a loaded package value."""

    if isinstance(value, np.ndarray):
        return (
            f"numpy.ndarray "
            f"shape={value.shape} "
            f"dtype={value.dtype}"
        )

    if torch.is_tensor(value):
        return (
            f"torch.Tensor "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}"
        )

    if isinstance(value, dict):
        return (
            f"dict with {len(value)} keys: "
            f"{list(value.keys())[:15]}"
        )

    if isinstance(value, (list, tuple)):
        preview = list(value[:5])

        return (
            f"{type(value).__name__} "
            f"length={len(value)} "
            f"preview={preview}"
        )

    return (
        f"{type(value).__name__}: "
        f"{repr(value)[:300]}"
    )


print("NOTEBOOK 45 — PPO SCHEMA INSPECTION")
print("=" * 80)

print("Top-level package keys:")

for key, value in ppo_package_full.items():
    print(
        f"  {key}: {summarize_value(value)}"
    )


ppo_metadata = ppo_package_full.get(
    "metadata",
    {},
)

ppo_batch_full = ppo_package_full.get(
    "ppo_batch",
    {},
)

print()
print("METADATA")
print("-" * 80)

for key, value in ppo_metadata.items():
    print(
        f"{key}: {repr(value)}"
    )


print()
print("PPO BATCH CONTENTS")
print("-" * 80)

for key, value in ppo_batch_full.items():
    print(
        f"{key}: {summarize_value(value)}"
    )


observations_array = np.asarray(
    ppo_batch_full["observations"]
)

action_masks_array = np.asarray(
    ppo_batch_full["action_masks"]
)

actions_array = np.asarray(
    ppo_batch_full["actions"]
)

rewards_array = np.asarray(
    ppo_batch_full["rewards"]
)

dones_array = np.asarray(
    ppo_batch_full["dones"]
)


sample_table = pd.DataFrame(
    observations_array,
    columns=[
        f"feature_{index}"
        for index in range(
            observations_array.shape[1]
        )
    ],
)

sample_table["selected_action_index"] = (
    actions_array
)

sample_table["reward"] = rewards_array
sample_table["done"] = dones_array

for action_index in range(
    action_masks_array.shape[1]
):
    sample_table[
        f"action_{action_index}_legal"
    ] = action_masks_array[
        :,
        action_index,
    ]


print()
print("OBSERVATION AND ACTION SAMPLES")
print("-" * 80)

display(sample_table)


possible_schema_keys = [
    "observation_features",
    "feature_names",
    "observation_columns",
    "state_features",
    "input_features",
    "action_names",
    "action_labels",
    "action_mapping",
    "action_to_index",
    "index_to_action",
    "normalization",
    "normalization_stats",
    "scaler",
    "schema",
]

discovered_schema = {}

for key in possible_schema_keys:

    if key in ppo_metadata:
        discovered_schema[
            f"metadata.{key}"
        ] = ppo_metadata[key]

    if key in ppo_package_full:
        discovered_schema[
            f"package.{key}"
        ] = ppo_package_full[key]

    if key in ppo_batch_full:
        discovered_schema[
            f"ppo_batch.{key}"
        ] = ppo_batch_full[key]


print()
print("DISCOVERED SCHEMA FIELDS")
print("-" * 80)

if discovered_schema:

    for key, value in (
        discovered_schema.items()
    ):
        print(
            f"{key}: {repr(value)}"
        )

else:
    print(
        "No explicit feature-name or action-mapping "
        "fields were stored in the package."
    )


schema_report = {
    "top_level_keys": list(
        ppo_package_full.keys()
    ),
    "metadata": ppo_metadata,
    "batch_keys": list(
        ppo_batch_full.keys()
    ),
    "observation_shape": list(
        observations_array.shape
    ),
    "action_mask_shape": list(
        action_masks_array.shape
    ),
    "sample_observations": (
        observations_array.tolist()
    ),
    "sample_action_masks": (
        action_masks_array.tolist()
    ),
    "sample_actions": (
        actions_array.tolist()
    ),
    "discovered_schema": (
        discovered_schema
    ),
}


def json_safe(
    value: Any,
) -> Any:
    """Convert NumPy and Torch values for JSON export."""

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
            np.bool_,
        ),
    ):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


ppo_schema_file = (
    NOTEBOOK45_REPORT_DIR
    / "ppo_observation_action_schema.json"
)

with open(
    ppo_schema_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        json_safe(schema_report),
        file,
        indent=2,
    )


assert observations_array.ndim == 2
assert observations_array.shape[1] == 4

assert action_masks_array.ndim == 2
assert action_masks_array.shape[1] == 4

assert len(actions_array) == len(
    observations_array
)

assert ppo_schema_file.exists()

print()
print(
    "Schema report saved:",
    ppo_schema_file,
)

print()
print(
    "✅ SECTION 6 PPO SCHEMA INSPECTION PASSED"
)


# ## Section 7 — Create the Production PPO Policy Engine
# 
# Create a reusable PPO policy engine under `src/ppo/` that matches the `search_depth()` interface expected by `PokemonBattleAgent`.
# 
# The engine:
# 
# - loads the Notebook 42 Actor checkpoint;
# - converts a real `BattleState` into four normalized features;
# - retrieves legal moves from the existing simulator;
# - constructs a four-slot legal-action mask;
# - selects the highest-probability legal action;
# - falls back deterministically if inference fails;
# - returns a search-compatible result object.

# In[7]:


# ============================================================
# SECTION 7 — CREATE PRODUCTION PPO POLICY ENGINE
# ============================================================

PPO_SOURCE_DIR = SRC_DIR / "ppo"

PPO_SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PPO_INIT_FILE = (
    PPO_SOURCE_DIR / "__init__.py"
)

PPO_ENGINE_FILE = (
    PPO_SOURCE_DIR / "ppo_policy_engine.py"
)


ppo_init_source = '''"""Production PPO integration for the Pokémon battle agent."""

from .ppo_policy_engine import (
    PPOPolicyEngine,
    PPOPolicyResult,
    PPOPolicyStats,
)

__all__ = [
    "PPOPolicyEngine",
    "PPOPolicyResult",
    "PPOPolicyStats",
]
'''


ppo_engine_source = r'''"""
Production PPO policy engine.

This module adapts the trained PPO Actor network to the search_depth()
interface expected by PokemonBattleAgent.

The checkpoint currently represents a bootstrap validation model.
The observation schema used for simulator integration is explicitly:

    0. current active Pokémon HP ratio
    1. opposing active Pokémon HP ratio
    2. current player's prize progress
    3. opposing player's prize progress

Legal moves are mapped to action slots in their existing order, with a
maximum of four action slots matching the trained Actor output dimension.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.legal_moves import get_current_legal_moves


OBSERVATION_DIM = 4
ACTION_DIM = 4


class PPOActor(nn.Module):
    """Actor architecture used by Notebooks 41–44."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observations)


@dataclass(frozen=True)
class PPOPolicyStats:
    """Minimal statistics compatible with PokemonBattleAgent."""

    nodes: int = 1


@dataclass(frozen=True)
class PPOPolicyResult:
    """Search-compatible PPO result."""

    best_move: Any
    score: float
    completed_depth: int
    stats: PPOPolicyStats
    principal_variation: list[Any]
    action_index: int
    confidence: float
    used_fallback: bool


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(
    value: float,
) -> float:
    return max(
        0.0,
        min(1.0, value),
    )


def _pokemon_max_hp(
    pokemon_state: Any,
) -> float:
    """
    Read maximum HP from the underlying card dictionary.

    Multiple common field names are supported because historical dataset
    exports may use slightly different capitalization.
    """

    card = getattr(
        pokemon_state,
        "card",
        {},
    )

    if not isinstance(card, dict):
        return max(
            1.0,
            _safe_float(
                getattr(
                    pokemon_state,
                    "current_hp",
                    1.0,
                ),
                1.0,
            ),
        )

    candidate_fields = (
        "hp",
        "HP",
        "Hit Points",
        "hit_points",
        "max_hp",
    )

    for field in candidate_fields:
        value = _safe_float(
            card.get(field),
            0.0,
        )

        if value > 0.0:
            return value

    return max(
        1.0,
        _safe_float(
            getattr(
                pokemon_state,
                "current_hp",
                1.0,
            ),
            1.0,
        ),
    )


def encode_battle_observation(
    battle_state: Any,
) -> np.ndarray:
    """
    Encode one BattleState into the four PPO input features.

    Feature order:
        0 current active HP ratio
        1 opponent active HP ratio
        2 current prize progress
        3 opponent prize progress
    """

    if battle_state.current_player == "Player":
        current_side = battle_state.player
        opposing_side = battle_state.opponent

    elif battle_state.current_player == "Opponent":
        current_side = battle_state.opponent
        opposing_side = battle_state.player

    else:
        raise ValueError(
            "current_player must be 'Player' or 'Opponent'."
        )

    current_hp = _safe_float(
        current_side.active.current_hp,
    )

    opponent_hp = _safe_float(
        opposing_side.active.current_hp,
    )

    current_max_hp = _pokemon_max_hp(
        current_side.active
    )

    opponent_max_hp = _pokemon_max_hp(
        opposing_side.active
    )

    current_hp_ratio = _clamp01(
        current_hp / current_max_hp
    )

    opponent_hp_ratio = _clamp01(
        opponent_hp / opponent_max_hp
    )

    current_prizes_remaining = _safe_float(
        current_side.prize_cards_remaining,
        6.0,
    )

    opponent_prizes_remaining = _safe_float(
        opposing_side.prize_cards_remaining,
        6.0,
    )

    current_prize_progress = _clamp01(
        (6.0 - current_prizes_remaining) / 6.0
    )

    opponent_prize_progress = _clamp01(
        (6.0 - opponent_prizes_remaining) / 6.0
    )

    return np.asarray(
        [
            current_hp_ratio,
            opponent_hp_ratio,
            current_prize_progress,
            opponent_prize_progress,
        ],
        dtype=np.float32,
    )


class PPOPolicyEngine:
    """
    PPO-backed engine compatible with PokemonBattleAgent.

    PokemonBattleAgent calls:

        engine.search_depth(
            state=state,
            depth=depth,
            clear_table=True,
        )
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        deterministic: bool = True,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        ).resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.deterministic = bool(
            deterministic
        )

        self.actor = PPOActor().to(
            self.device
        )

        self._load_checkpoint()

        self.actor.eval()

    def _load_checkpoint(
        self,
    ) -> None:

        with self.checkpoint_path.open(
            "rb"
        ) as file:
            checkpoint = pickle.load(file)

        actor_state_dict = checkpoint.get(
            "actor_state_dict"
        )

        if actor_state_dict is None:
            raise KeyError(
                "Checkpoint does not contain "
                "'actor_state_dict'."
            )

        self.actor.load_state_dict(
            actor_state_dict
        )

        self.metadata = checkpoint.get(
            "metadata",
            {},
        )

    @staticmethod
    def _fallback_action_index(
        legal_moves: Sequence[Any],
    ) -> int:
        """
        Deterministic fallback.

        Select the move with the greatest numeric damage. Ties preserve
        the original legal-move ordering.
        """

        if not legal_moves:
            raise RuntimeError(
                "No legal moves were supplied."
            )

        best_index = 0
        best_damage = float("-inf")

        for index, move in enumerate(
            legal_moves
        ):
            damage = 0.0

            if isinstance(move, dict):
                damage = _safe_float(
                    move.get(
                        "damage",
                        move.get(
                            "damage_numeric",
                            0.0,
                        ),
                    )
                )

            if damage > best_damage:
                best_damage = damage
                best_index = index

        return best_index

    def select_move(
        self,
        state: Any,
    ) -> PPOPolicyResult:
        """Select one legal move from a BattleState."""

        legal_moves = list(
            get_current_legal_moves(state)
        )

        if not legal_moves:
            raise RuntimeError(
                "The simulator returned no legal moves."
            )

        # Actor supports four action slots.
        candidate_moves = legal_moves[
            :ACTION_DIM
        ]

        observation = (
            encode_battle_observation(state)
        )

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action_mask = torch.zeros(
            (1, ACTION_DIM),
            dtype=torch.bool,
            device=self.device,
        )

        action_mask[
            0,
            :len(candidate_moves),
        ] = True

        used_fallback = False

        try:
            with torch.no_grad():

                logits = self.actor(
                    observation_tensor
                )

                masked_logits = logits.masked_fill(
                    ~action_mask,
                    torch.finfo(
                        logits.dtype
                    ).min,
                )

                distribution = Categorical(
                    logits=masked_logits
                )

                if self.deterministic:
                    selected_index = int(
                        distribution.probs.argmax(
                            dim=1
                        ).item()
                    )
                else:
                    selected_index = int(
                        distribution.sample().item()
                    )

                confidence = float(
                    distribution.probs[
                        0,
                        selected_index,
                    ].item()
                )

                score = float(
                    masked_logits[
                        0,
                        selected_index,
                    ].item()
                )

        except Exception:
            selected_index = (
                self._fallback_action_index(
                    candidate_moves
                )
            )

            confidence = 0.0
            score = _safe_float(
                candidate_moves[
                    selected_index
                ].get("damage", 0.0)
                if isinstance(
                    candidate_moves[
                        selected_index
                    ],
                    dict,
                )
                else 0.0
            )

            used_fallback = True

        if (
            selected_index < 0
            or selected_index
            >= len(candidate_moves)
        ):
            selected_index = (
                self._fallback_action_index(
                    candidate_moves
                )
            )

            confidence = 0.0
            used_fallback = True

        selected_move = candidate_moves[
            selected_index
        ]

        return PPOPolicyResult(
            best_move=selected_move,
            score=score,
            completed_depth=1,
            stats=PPOPolicyStats(
                nodes=1
            ),
            principal_variation=[
                selected_move
            ],
            action_index=selected_index,
            confidence=confidence,
            used_fallback=used_fallback,
        )

    def search_depth(
        self,
        state: Any,
        depth: int = 1,
        clear_table: bool = True,
    ) -> PPOPolicyResult:
        """
        Search-compatible entry point.

        depth and clear_table are accepted for compatibility with the
        existing PokemonBattleAgent interface.
        """

        del depth
        del clear_table

        return self.select_move(state)
'''


PPO_INIT_FILE.write_text(
    ppo_init_source,
    encoding="utf-8",
)

PPO_ENGINE_FILE.write_text(
    ppo_engine_source,
    encoding="utf-8",
)


print("NOTEBOOK 45 — CREATE PPO POLICY ENGINE")
print("=" * 80)

print("Created:")
print(PPO_INIT_FILE)

print()
print("Created:")
print(PPO_ENGINE_FILE)

assert PPO_INIT_FILE.exists()
assert PPO_ENGINE_FILE.exists()

assert PPO_ENGINE_FILE.stat().st_size > 0

print()
print(
    "✅ SECTION 7 PPO POLICY ENGINE CREATED"
)


# ## Section 8 — Import and Validate the PPO Engine
# 
# #### Import the newly created production module, load the trained checkpoint, and verify that its model architecture and interface are valid.

# In[8]:


# ============================================================
# SECTION 8 — IMPORT AND VALIDATE PPO ENGINE
# ============================================================

import importlib
import py_compile

py_compile.compile(
    str(PPO_ENGINE_FILE),
    doraise=True,
)

importlib.invalidate_caches()

from src.ppo.ppo_policy_engine import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PPOActor,
    PPOPolicyEngine,
    PPOPolicyResult,
    encode_battle_observation,
)


ppo_engine = PPOPolicyEngine(
    checkpoint_path=TRAINED_CHECKPOINT_FILE,
    device=DEVICE,
    deterministic=True,
)

actor_parameter_count = sum(
    parameter.numel()
    for parameter in ppo_engine.actor.parameters()
)

print("NOTEBOOK 45 — VALIDATE PPO ENGINE")
print("=" * 80)

print(
    "Checkpoint:",
    ppo_engine.checkpoint_path,
)

print(
    "Device:",
    ppo_engine.device,
)

print(
    "Observation dimension:",
    OBSERVATION_DIM,
)

print(
    "Action dimension:",
    ACTION_DIM,
)

print(
    "Actor parameters:",
    actor_parameter_count,
)

print(
    "Deterministic:",
    ppo_engine.deterministic,
)

assert OBSERVATION_DIM == 4
assert ACTION_DIM == 4
assert actor_parameter_count == 9156

assert callable(
    ppo_engine.search_depth
)

assert callable(
    ppo_engine.select_move
)

print()
print(
    "✅ SECTION 8 PPO ENGINE IMPORT PASSED"
)


# ## Section 9 — Real Battle-State PPO Inference
# 
# #### Construct a valid `BattleState` using the existing simulator data classes, generate legal moves, encode the state, and verify that the PPO engine selects a legal move.

# In[9]:


# ============================================================
# SECTION 9 — REAL BATTLE-STATE PPO INFERENCE
# ============================================================

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

from src.legal_moves import (
    get_current_legal_moves,
)

from src.observation_adapter.battle_state_adapter import (
    build_policy_state,
)

from src.battle_agent import (
    PokemonBattleAgent,
)


player_card = {
    "name": "Pikachu",
    "hp": 120,
    "attacks": [
        {
            "Move Name": "Quick Attack",
            "damage_numeric": 30,
            "energy_cost": 1,
            "Effect Explanation": None,
        },
        {
            "Move Name": "Thunderbolt",
            "damage_numeric": 90,
            "energy_cost": 2,
            "Effect Explanation": None,
        },
    ],
}

opponent_card = {
    "name": "Charmander",
    "hp": 100,
    "attacks": [
        {
            "Move Name": "Scratch",
            "damage_numeric": 20,
            "energy_cost": 1,
            "Effect Explanation": None,
        },
        {
            "Move Name": "Flame Tail",
            "damage_numeric": 60,
            "energy_cost": 2,
            "Effect Explanation": None,
        },
    ],
}


player_pokemon = PokemonState(
    card=player_card,
    current_hp=120.0,
    attached_energy=2,
    is_active=True,
)

opponent_pokemon = PokemonState(
    card=opponent_card,
    current_hp=100.0,
    attached_energy=2,
    is_active=True,
)


player_state = PlayerState(
    active=player_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)

opponent_state = PlayerState(
    active=opponent_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


test_battle_state = BattleState(
    player=player_state,
    opponent=opponent_state,
    turn_number=1,
    current_player="Player",
)


legal_moves = get_current_legal_moves(
    test_battle_state
)

policy_state = build_policy_state(
    test_battle_state,
    legal_moves,
)

encoded_observation = (
    encode_battle_observation(
        test_battle_state
    )
)


ppo_result = ppo_engine.select_move(
    test_battle_state
)


ppo_battle_agent = PokemonBattleAgent(
    engine=ppo_engine
)

agent_decision = ppo_battle_agent.choose_move(
    state=test_battle_state,
    depth=1,
)


legal_move_names = [
    move["name"]
    for move in legal_moves
]

print("NOTEBOOK 45 — REAL PPO INFERENCE")
print("=" * 80)

print("Current player:", test_battle_state.current_player)
print("Player Pokémon:", player_card["name"])
print("Opponent Pokémon:", opponent_card["name"])

print()
print("Legal moves:")
for index, move in enumerate(legal_moves):
    print(
        f"  {index}: {move['name']} "
        f"(damage={move['damage']})"
    )

print()
print(
    "Encoded observation:",
    encoded_observation,
)

print()
print(
    "Policy-state legal moves:",
    policy_state["policy_legal_moves"],
)

print()
print(
    "PPO selected action index:",
    ppo_result.action_index,
)

print(
    "PPO selected move:",
    ppo_result.best_move["name"],
)

print(
    "PPO confidence:",
    ppo_result.confidence,
)

print(
    "Fallback used:",
    ppo_result.used_fallback,
)

print()
print(
    "PokemonBattleAgent decision:",
    agent_decision.move["name"],
)


assert encoded_observation.shape == (4,)

assert np.isfinite(
    encoded_observation
).all()

assert len(legal_moves) > 0

assert ppo_result.best_move in legal_moves

assert (
    ppo_result.best_move["name"]
    in legal_move_names
)

assert (
    agent_decision.move["name"]
    in legal_move_names
)

assert (
    0
    <= ppo_result.action_index
    < min(len(legal_moves), ACTION_DIM)
)

assert (
    0.0
    <= ppo_result.confidence
    <= 1.0
)

print()
print(
    "✅ SECTION 9 REAL PPO INFERENCE PASSED"
)


# ## Section 10 — Complete PPO-Controlled Battle
# 
# #### Run the existing battle simulator using the PPO-backed `PokemonBattleAgent` and verify that actions are applied legally until a winner is produced or the turn limit is reached.

# In[10]:


# ============================================================
# SECTION 10 — COMPLETE PPO-CONTROLLED BATTLE
# ============================================================

from src.battle_simulation import (
    create_battle_transcript,
    simulate_ai_battle,
)


battle_result = simulate_ai_battle(
    initial_state=test_battle_state,
    agent=ppo_battle_agent,
    search_depth=1,
    max_turns=20,
    verbose=True,
)


battle_transcript = create_battle_transcript(
    battle_result
)


print()
print("NOTEBOOK 45 — PPO BATTLE RESULT")
print("=" * 80)

print("Winner:", battle_result.winner)
print("Turns completed:", len(battle_result.turns))
print("Stop reason:", battle_result.stop_reason)

print()
print("Final Player HP:")
print(
    battle_result
    .final_state
    .player
    .active
    .current_hp
)

print("Final Opponent HP:")
print(
    battle_result
    .final_state
    .opponent
    .active
    .current_hp
)

print()
print("Battle transcript")
print("-" * 80)

print(battle_transcript)


assert battle_result.final_state is not None

assert len(battle_result.turns) > 0

assert len(battle_result.turns) <= 20

assert battle_result.stop_reason in {
    "Battle reached a terminal state.",
    "Maximum turn limit reached.",
}

for record in battle_result.turns:

    assert record.move_name

    assert record.acting_side in {
        "Player",
        "Opponent",
    }

    assert record.damage >= 0.0


battle_result_file = (
    NOTEBOOK45_REPORT_DIR
    / "ppo_battle_result.pkl"
)

battle_transcript_file = (
    NOTEBOOK45_REPORT_DIR
    / "ppo_battle_transcript.txt"
)


with open(
    battle_result_file,
    "wb",
) as file:
    pickle.dump(
        battle_result,
        file,
    )


battle_transcript_file.write_text(
    str(battle_transcript),
    encoding="utf-8",
)


assert battle_result_file.exists()
assert battle_transcript_file.exists()

print()
print(
    "Saved battle result:",
    battle_result_file,
)

print(
    "Saved battle transcript:",
    battle_transcript_file,
)

print()
print(
    "✅ SECTION 10 COMPLETE PPO BATTLE PASSED"
)


# ## Section 11 — Create the Final Competition Agent
# 
# #### Create a reusable final agent module that loads the PPO checkpoint, exposes a simple battle-action interface, validates legal moves, and falls back safely if PPO inference fails.

# In[11]:


# ============================================================
# SECTION 11 — CREATE FINAL COMPETITION AGENT
# ============================================================

FINAL_AGENT_FILE = (
    SRC_DIR
    / "agents"
    / "final_ppo_agent.py"
)

FINAL_AGENT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final_agent_source = r'''"""
Final PPO-powered Pokémon battle agent.

This module wraps the production PPOPolicyEngine and the existing
PokemonBattleAgent interface.

The current checkpoint is a bootstrap validation checkpoint. The agent
therefore includes deterministic legal-action fallback protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.battle_agent import PokemonBattleAgent
from src.legal_moves import get_current_legal_moves
from src.ppo.ppo_policy_engine import PPOPolicyEngine


class FinalPPOBattleAgent:
    """Competition-facing PPO battle agent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        deterministic: bool = True,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        ).resolve()

        self.engine = PPOPolicyEngine(
            checkpoint_path=self.checkpoint_path,
            device=device,
            deterministic=deterministic,
        )

        self.agent = PokemonBattleAgent(
            engine=self.engine
        )

    @staticmethod
    def _safe_fallback_move(
        state: Any,
    ) -> Any:
        """
        Select the highest-damage legal move.

        The legal-move module already provides a Pass action when no
        attack is available.
        """

        legal_moves = list(
            get_current_legal_moves(state)
        )

        if not legal_moves:
            raise RuntimeError(
                "No legal move was available."
            )

        return max(
            legal_moves,
            key=lambda move: float(
                move.get("damage", 0.0) or 0.0
            )
            if isinstance(move, dict)
            else 0.0,
        )

    def choose_move(
        self,
        state: Any,
        depth: int = 1,
    ) -> Any:
        """Return an AgentDecision compatible with the simulator."""

        try:
            decision = self.agent.choose_move(
                state=state,
                depth=depth,
            )

            legal_moves = list(
                get_current_legal_moves(state)
            )

            if decision.move not in legal_moves:
                raise RuntimeError(
                    "PPO selected a move outside the legal move set."
                )

            return decision

        except Exception:

            fallback_move = (
                self._safe_fallback_move(state)
            )

            # Reuse the search-compatible PPO result pathway.
            result = self.engine.search_depth(
                state=state,
                depth=1,
                clear_table=True,
            )

            result = type(result)(
                best_move=fallback_move,
                score=float(
                    fallback_move.get(
                        "damage",
                        0.0,
                    )
                )
                if isinstance(
                    fallback_move,
                    dict,
                )
                else 0.0,
                completed_depth=1,
                stats=result.stats,
                principal_variation=[
                    fallback_move
                ],
                action_index=0,
                confidence=0.0,
                used_fallback=True,
            )

            from src.agent_decision import (
                AgentDecision,
            )

            return AgentDecision(
                move=result.best_move,
                score=result.score,
                search_depth=result.completed_depth,
                nodes=result.stats.nodes,
                principal_variation=(
                    result.principal_variation
                ),
            )


def build_final_agent(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> FinalPPOBattleAgent:
    """Factory used by local evaluation or submission code."""

    return FinalPPOBattleAgent(
        checkpoint_path=checkpoint_path,
        device=device,
        deterministic=True,
    )
'''

FINAL_AGENT_FILE.write_text(
    final_agent_source,
    encoding="utf-8",
)

print("NOTEBOOK 45 — FINAL AGENT MODULE")
print("=" * 80)

print("Created:")
print(FINAL_AGENT_FILE)

assert FINAL_AGENT_FILE.exists()
assert FINAL_AGENT_FILE.stat().st_size > 0

print()
print("✅ SECTION 11 FINAL AGENT MODULE CREATED")


# ## Section 12 — Validate the Final Agent
# 
# #### Compile and import the final agent, then verify that it selects a legal move and completes a battle using the production wrapper.

# In[12]:


# ============================================================
# SECTION 12 — VALIDATE FINAL COMPETITION AGENT
# ============================================================

import importlib
import py_compile

py_compile.compile(
    str(FINAL_AGENT_FILE),
    doraise=True,
)

importlib.invalidate_caches()

from src.agents.final_ppo_agent import (
    FinalPPOBattleAgent,
    build_final_agent,
)

final_agent = build_final_agent(
    checkpoint_path=TRAINED_CHECKPOINT_FILE,
    device=str(DEVICE),
)

final_decision = final_agent.choose_move(
    state=test_battle_state,
    depth=1,
)

final_legal_moves = get_current_legal_moves(
    test_battle_state
)

print("NOTEBOOK 45 — FINAL AGENT VALIDATION")
print("=" * 80)

print(
    "Final selected move:",
    final_decision.move["name"],
)

print(
    "Legal move names:",
    [
        move["name"]
        for move in final_legal_moves
    ],
)

assert final_decision.move in final_legal_moves

final_agent_battle = simulate_ai_battle(
    initial_state=test_battle_state,
    agent=final_agent,
    search_depth=1,
    max_turns=20,
    verbose=False,
)

print()
print(
    "Battle winner:",
    final_agent_battle.winner,
)

print(
    "Turns:",
    len(final_agent_battle.turns),
)

print(
    "Stop reason:",
    final_agent_battle.stop_reason,
)

assert len(final_agent_battle.turns) > 0

assert final_agent_battle.stop_reason in {
    "Battle reached a terminal state.",
    "Maximum turn limit reached.",
}

print()
print(
    "✅ SECTION 12 FINAL AGENT VALIDATION PASSED"
)


# ## Section 13 — Build Final Submission Package
# 
# #### Collect the final PPO engine, competition agent, checkpoint, required simulator files, and instructions into a reproducible submission folder and ZIP archive.

# In[15]:


# ============================================================
# SECTION 13 — BUILD FINAL SUBMISSION PACKAGE
# ============================================================

FINAL_PACKAGE_DIR = (
    PROJECT_ROOT
    / "submission"
    / "final_agent"
)

FINAL_PACKAGE_FILE = (
    PROJECT_ROOT
    / "submission"
    / "ptcg_final_agent.zip"
)

if FINAL_PACKAGE_DIR.exists():
    shutil.rmtree(
        FINAL_PACKAGE_DIR
    )

FINAL_PACKAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

package_files = {
    SRC_DIR / "ppo" / "__init__.py":
        Path("src/ppo/__init__.py"),

    SRC_DIR / "ppo" / "ppo_policy_engine.py":
        Path("src/ppo/ppo_policy_engine.py"),

    FINAL_AGENT_FILE:
        Path("src/agents/final_ppo_agent.py"),

    SRC_DIR / "battle_agent.py":
        Path("src/battle_agent.py"),

    SRC_DIR / "battle_state.py":
        Path("src/battle_state.py"),

    SRC_DIR / "legal_moves.py":
        Path("src/legal_moves.py"),

    SRC_DIR / "simulator.py":
        Path("src/simulator.py"),

    SRC_DIR / "battle_simulation.py":
        Path("src/battle_simulation.py"),

    SRC_DIR / "agent_decision.py":
        Path("src/agent_decision.py"),

    TRAINED_CHECKPOINT_FILE:
        Path("models/ppo_training_loop.pkl"),
}

for source_path, relative_path in (
    package_files.items()
):

    assert source_path.exists(), (
        f"Missing required package file: "
        f"{source_path}"
    )

    destination_path = (
        FINAL_PACKAGE_DIR
        / relative_path
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )


readme_text = """
# Pokémon TCG Final PPO Agent

## Entry Point

from src.agents.final_ppo_agent import build_final_agent

agent = build_final_agent(
    checkpoint_path="models/ppo_training_loop.pkl"
)

This package contains the production PPO policy engine,
battle agent wrapper, and trained checkpoint.

The current checkpoint is a bootstrap validation checkpoint.
"""


# In[16]:


readme_file = (
    FINAL_PACKAGE_DIR
    / "README.md"
)

readme_file.write_text(
    readme_text,
    encoding="utf-8",
)

print("README created:")
print(readme_file)

assert readme_file.exists()

print()
print("✅ README CREATION PASSED")


# In[18]:


# ============================================================
# RECREATE FINAL ZIP ARCHIVE
# ============================================================

FINAL_PACKAGE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if FINAL_PACKAGE_FILE.exists():
    FINAL_PACKAGE_FILE.unlink()

with zipfile.ZipFile(
    FINAL_PACKAGE_FILE,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:

    for file_path in sorted(
        FINAL_PACKAGE_DIR.rglob("*")
    ):
        if file_path.is_file():
            archive.write(
                file_path,
                file_path.relative_to(
                    FINAL_PACKAGE_DIR
                ),
            )

print("ZIP created:")
print(FINAL_PACKAGE_FILE)

print()
print("ZIP exists:", FINAL_PACKAGE_FILE.exists())
print(
    "ZIP size:",
    FINAL_PACKAGE_FILE.stat().st_size
    if FINAL_PACKAGE_FILE.exists()
    else 0,
)

with zipfile.ZipFile(
    FINAL_PACKAGE_FILE,
    "r",
) as archive:
    zip_contents = archive.namelist()

print()
print("ZIP contents:")

for item in zip_contents:
    print(" -", item)

assert FINAL_PACKAGE_FILE.exists()
assert FINAL_PACKAGE_FILE.stat().st_size > 0
assert len(zip_contents) > 0

print()
print("✅ FINAL ZIP ARCHIVE CREATED")


# ## Section 14 — Final Submission Audit
# 
# #### Verify that all critical source files, model artifacts, battle outputs, and submission files exist and are readable.

# In[19]:


# ============================================================
# SECTION 14 — FINAL SUBMISSION AUDIT
# ============================================================

required_final_files = {
    "PPO Policy Engine":
        SRC_DIR
        / "ppo"
        / "ppo_policy_engine.py",

    "Final PPO Agent":
        SRC_DIR
        / "agents"
        / "final_ppo_agent.py",

    "PPO Checkpoint":
        TRAINED_CHECKPOINT_FILE,

    "Battle Result":
        battle_result_file,

    "Battle Transcript":
        battle_transcript_file,

    "Submission README":
        readme_file,

    "Submission ZIP":
        FINAL_PACKAGE_FILE,
}


audit_rows = []

for name, file_path in (
    required_final_files.items()
):

    audit_rows.append(
        {
            "Artifact": name,
            "Path": str(file_path),
            "Exists": file_path.exists(),
            "Size Bytes": (
                file_path.stat().st_size
                if file_path.exists()
                else 0
            ),
        }
    )


final_audit = pd.DataFrame(
    audit_rows
)

print("NOTEBOOK 45 — FINAL SUBMISSION AUDIT")
print("=" * 80)

display(final_audit)

assert final_audit["Exists"].all()

assert (
    final_audit["Size Bytes"] > 0
).all()


# Compile the two production modules again.

py_compile.compile(
    str(
        SRC_DIR
        / "ppo"
        / "ppo_policy_engine.py"
    ),
    doraise=True,
)

py_compile.compile(
    str(
        SRC_DIR
        / "agents"
        / "final_ppo_agent.py"
    ),
    doraise=True,
)


# Verify ZIP readability.

with zipfile.ZipFile(
    FINAL_PACKAGE_FILE,
    "r",
) as archive:

    bad_file = archive.testzip()

assert bad_file is None


audit_report = {
    "project_root": str(PROJECT_ROOT),
    "final_notebook": (
        "45_final_simulator_integration_and_submission"
    ),
    "ppo_engine_created": True,
    "final_agent_created": True,
    "checkpoint_loaded": True,
    "legal_action_validation": True,
    "complete_battle_passed": True,
    "battle_winner": (
        final_agent_battle.winner
    ),
    "battle_turns": len(
        final_agent_battle.turns
    ),
    "submission_zip": str(
        FINAL_PACKAGE_FILE
    ),
    "zip_valid": True,
    "status": "READY_FOR_FINAL_REVIEW",
}


FINAL_READINESS_FILE.write_text(
    json.dumps(
        audit_report,
        indent=2,
    ),
    encoding="utf-8",
)

assert FINAL_READINESS_FILE.exists()

print()
print(
    "Readiness report:",
    FINAL_READINESS_FILE,
)

print()
print(
    "✅ SECTION 14 FINAL SUBMISSION AUDIT PASSED"
)


# In[ ]:




