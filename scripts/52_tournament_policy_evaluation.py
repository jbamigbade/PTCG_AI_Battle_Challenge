#!/usr/bin/env python
# coding: utf-8

# # Notebook 52 — Tournament Policy Evaluation
# 
# ## Purpose
# 
# #### This notebook evaluates the Notebook 50 side-balanced policy agent across repeated production-simulator battles.
# 
# #### The tournament evaluation pipeline will:
# 
# 1. Validate the Notebook 51 simulator-integration handoff.
# 2. Load the trained policy artifacts.
# 3. Load the production simulator runtime.
# 4. reconstruct the policy agent and battle-runner adapter.
# 5. build reusable tournament battle scenarios.
# 6. evaluate the policy from both Player and Opponent starting sides.
# 7. compare performance across matchups and battle conditions.
# 8. measure wins, losses, draws, turns, confidence, and fallback rates.
# 9. analyze side bias and matchup weaknesses.
# 10. export tournament reports and the Notebook 52 handoff.
# 
# #### Notebook 51 handoff
# 
# #### Expected upstream status:
# 
# - `READY_FOR_TOURNAMENT_EVALUATION`
# - 6 policy classes
# - 12 raw policy features
# - 18 encoded features
# - 11 simulator runtime symbols loaded
# - real simulator transition validated
# - full production battle completed
# - 27/27 validation checks passed
# - 0 validation checks failed

# ## Section 1A — Imports and Project Paths

# In[1]:


# ======================================================================================
# SECTION 1A — IMPORTS AND PROJECT PATHS
# ======================================================================================

from __future__ import annotations

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

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
# Upstream Notebook 50 and Notebook 51 directories
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


# --------------------------------------------------------------------------------------
# Notebook 52 output directories
# --------------------------------------------------------------------------------------

NOTEBOOK52_REPORT_DIR = (
    REPORTS_DIR
    / "notebook52"
)

NOTEBOOK52_MODEL_DIR = (
    MODELS_DIR
    / "notebook52"
)

NOTEBOOK52_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NOTEBOOK52_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# Initial directory validation
# --------------------------------------------------------------------------------------

required_upstream_directories = {
    "Notebook 50 reports":
        NOTEBOOK50_REPORT_DIR,

    "Notebook 50 models":
        NOTEBOOK50_MODEL_DIR,

    "Notebook 51 reports":
        NOTEBOOK51_REPORT_DIR,

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
print("SECTION 1A — IMPORTS AND PROJECT PATHS")
print("=" * 100)

print()
print("DIRECTORIES")
print("-" * 100)

print(
    f"Current directory       : "
    f"{CURRENT_DIRECTORY}"
)

print(
    f"Project root            : "
    f"{PROJECT_ROOT}"
)

print(
    f"Notebook 50 reports     : "
    f"{NOTEBOOK50_REPORT_DIR}"
)

print(
    f"Notebook 50 models      : "
    f"{NOTEBOOK50_MODEL_DIR}"
)

print(
    f"Notebook 51 reports     : "
    f"{NOTEBOOK51_REPORT_DIR}"
)

print(
    f"Notebook 52 reports     : "
    f"{NOTEBOOK52_REPORT_DIR}"
)

print(
    f"Notebook 52 models      : "
    f"{NOTEBOOK52_MODEL_DIR}"
)

print(
    f"Source directory        : "
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


print()
print("UPSTREAM DIRECTORY STATUS")
print("-" * 100)

for directory_name, directory_path in (
    required_upstream_directories.items()
):

    print(
        f"{directory_name:28}: "
        f"{directory_path.exists()}"
    )


assert PROJECT_ROOT.exists(), (
    "The project root could not be resolved."
)

assert not missing_upstream_directories, (
    "Required upstream directories are missing: "
    f"{missing_upstream_directories}"
)

assert NOTEBOOK52_REPORT_DIR.exists()

assert NOTEBOOK52_MODEL_DIR.exists()


print()
print(
    "✅ SECTION 1A IMPORTS AND PROJECT PATHS PASSED"
)


# ## Section 1B — Notebook 51 Handoff Validation
# 
# #### This section validates the completed Notebook 51 integration handoff before tournament evaluation begins.
# 
# #### It confirms that:
# 
# - Notebook 51 finished successfully,
# - all final validation checks passed,
# - the trained policy was connected to the production simulator,
# - a complete policy-controlled battle was executed,
# - tournament evaluation is the approved next stage.

# In[2]:


# ======================================================================================
# SECTION 1B — NOTEBOOK 51 HANDOFF VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 1B — NOTEBOOK 51 HANDOFF VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Required Notebook 51 artifacts
# --------------------------------------------------------------------------------------

NOTEBOOK51_SECTION7_DIR = (
    NOTEBOOK51_REPORT_DIR
    / "section7"
)

NOTEBOOK51_SECTION6_DIR = (
    NOTEBOOK51_REPORT_DIR
    / "section6"
)


NOTEBOOK51_HANDOFF_MANIFEST_FILE = (
    NOTEBOOK51_SECTION7_DIR
    / "section7a_handoff_manifest.json"
)

NOTEBOOK51_FINAL_SUMMARY_FILE = (
    NOTEBOOK51_SECTION7_DIR
    / "section7a_final_summary.json"
)

NOTEBOOK51_VALIDATION_CHECKS_FILE = (
    NOTEBOOK51_SECTION7_DIR
    / "section7a_validation_checks.csv"
)

NOTEBOOK51_FINAL_PROFILE_FILE = (
    NOTEBOOK51_SECTION7_DIR
    / "section7a_final_integration_profile.csv"
)

NOTEBOOK51_BATTLE_SUMMARY_FILE = (
    NOTEBOOK51_SECTION6_DIR
    / "section6f_first_full_battle_summary.json"
)

NOTEBOOK51_BATTLE_TURNS_FILE = (
    NOTEBOOK51_SECTION6_DIR
    / "section6f_first_full_battle_turns.csv"
)

NOTEBOOK51_POLICY_HISTORY_FILE = (
    NOTEBOOK51_SECTION6_DIR
    / "section6f_first_full_battle_policy_history.csv"
)

NOTEBOOK51_BATTLE_TRANSCRIPT_FILE = (
    NOTEBOOK51_SECTION6_DIR
    / "section6f_first_full_battle_transcript.txt"
)

NOTEBOOK51_RUNTIME_BINDING_SUMMARY_FILE = (
    NOTEBOOK51_SECTION6_DIR
    / "section6c_simulator_binding_summary.json"
)


required_notebook51_artifacts = {
    "handoff_manifest":
        NOTEBOOK51_HANDOFF_MANIFEST_FILE,

    "final_summary":
        NOTEBOOK51_FINAL_SUMMARY_FILE,

    "validation_checks":
        NOTEBOOK51_VALIDATION_CHECKS_FILE,

    "final_profile":
        NOTEBOOK51_FINAL_PROFILE_FILE,

    "battle_summary":
        NOTEBOOK51_BATTLE_SUMMARY_FILE,

    "battle_turns":
        NOTEBOOK51_BATTLE_TURNS_FILE,

    "policy_history":
        NOTEBOOK51_POLICY_HISTORY_FILE,

    "battle_transcript":
        NOTEBOOK51_BATTLE_TRANSCRIPT_FILE,

    "runtime_binding_summary":
        NOTEBOOK51_RUNTIME_BINDING_SUMMARY_FILE,
}


artifact_rows = []

for artifact_name, artifact_path in (
    required_notebook51_artifacts.items()
):

    artifact_path = Path(
        artifact_path
    )

    artifact_rows.append(
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
        }
    )


notebook51_artifact_status_df = pd.DataFrame(
    artifact_rows
)

notebook51_artifact_status_df[
    "nonempty"
] = (
    notebook51_artifact_status_df[
        "size_bytes"
    ] > 0
)


print()
print("REQUIRED ARTIFACTS")
print("-" * 100)

display(
    notebook51_artifact_status_df
)


assert notebook51_artifact_status_df[
    "exists"
].all(), (
    "One or more required Notebook 51 artifacts are missing."
)

assert notebook51_artifact_status_df[
    "is_file"
].all(), (
    "One or more Notebook 51 artifact paths are not files."
)

assert notebook51_artifact_status_df[
    "nonempty"
].all(), (
    "One or more Notebook 51 artifacts are empty."
)


# --------------------------------------------------------------------------------------
# 2. Load Notebook 51 handoff data
# --------------------------------------------------------------------------------------

with open(
    NOTEBOOK51_HANDOFF_MANIFEST_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook51_handoff_manifest = json.load(
        file
    )


with open(
    NOTEBOOK51_FINAL_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook51_final_summary = json.load(
        file
    )


with open(
    NOTEBOOK51_BATTLE_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook51_battle_summary = json.load(
        file
    )


with open(
    NOTEBOOK51_RUNTIME_BINDING_SUMMARY_FILE,
    "r",
    encoding="utf-8",
) as file:

    notebook51_runtime_binding_summary = (
        json.load(
            file
        )
    )


notebook51_validation_checks_df = pd.read_csv(
    NOTEBOOK51_VALIDATION_CHECKS_FILE
)

notebook51_final_profile_df = pd.read_csv(
    NOTEBOOK51_FINAL_PROFILE_FILE
)

notebook51_battle_turns_df = pd.read_csv(
    NOTEBOOK51_BATTLE_TURNS_FILE
)

notebook51_policy_history_df = pd.read_csv(
    NOTEBOOK51_POLICY_HISTORY_FILE
)

notebook51_battle_transcript = (
    NOTEBOOK51_BATTLE_TRANSCRIPT_FILE
    .read_text(
        encoding="utf-8"
    )
)


# --------------------------------------------------------------------------------------
# 3. Resolve handoff values
# --------------------------------------------------------------------------------------

handoff_status = (
    notebook51_handoff_manifest.get(
        "status"
    )
)

validation_checks_passed = int(
    notebook51_handoff_manifest.get(
        "validation_checks_passed",
        0,
    )
)

validation_checks_failed = int(
    notebook51_handoff_manifest.get(
        "validation_checks_failed",
        0,
    )
)

policy_class_count = int(
    notebook51_handoff_manifest.get(
        "policy_class_count",
        0,
    )
)

raw_feature_count = int(
    notebook51_handoff_manifest.get(
        "raw_feature_count",
        0,
    )
)

encoded_feature_count = int(
    notebook51_handoff_manifest.get(
        "encoded_feature_count",
        0,
    )
)

simulator_symbols_loaded = int(
    notebook51_handoff_manifest.get(
        "simulator_symbols_loaded",
        0,
    )
)

full_battle_completed = bool(
    notebook51_handoff_manifest.get(
        "full_battle_completed",
        False,
    )
)

full_battle_winner = (
    notebook51_handoff_manifest.get(
        "full_battle_winner"
    )
)

full_battle_turns = int(
    notebook51_handoff_manifest.get(
        "full_battle_turns",
        0,
    )
)

next_notebook = (
    notebook51_handoff_manifest.get(
        "next_notebook"
    )
)

next_stage = (
    notebook51_handoff_manifest.get(
        "next_stage"
    )
)


# --------------------------------------------------------------------------------------
# 4. Display handoff status
# --------------------------------------------------------------------------------------

print()
print("HANDOFF STATUS")
print("-" * 100)

for key, value in (
    notebook51_handoff_manifest.items()
):

    print(
        f"{key:34}: {value}"
    )


print()
print("NOTEBOOK 51 FINAL PROFILE")
print("-" * 100)

display(
    notebook51_final_profile_df
)


print()
print("NOTEBOOK 51 BATTLE PROFILE")
print("-" * 100)

print(
    f"Battle winner                 : "
    f"{notebook51_battle_summary.get('winner')}"
)

print(
    f"Recorded turns                : "
    f"{notebook51_battle_summary.get('recorded_turns')}"
)

print(
    f"Policy decisions              : "
    f"{notebook51_battle_summary.get('policy_decisions')}"
)

print(
    f"Fallback decisions            : "
    f"{notebook51_battle_summary.get('fallback_decisions')}"
)

print(
    f"Fallback rate                 : "
    f"{notebook51_battle_summary.get('fallback_rate')}"
)

print(
    f"Termination reason            : "
    f"{notebook51_battle_summary.get('termination_reason')}"
)


# --------------------------------------------------------------------------------------
# 5. Final validation assertions
# --------------------------------------------------------------------------------------

assert (
    handoff_status
    == "READY_FOR_TOURNAMENT_EVALUATION"
), (
    "Notebook 51 is not ready for tournament evaluation."
)

assert (
    notebook51_final_summary[
        "final_status"
    ]
    == "READY_FOR_TOURNAMENT_EVALUATION"
)

assert validation_checks_passed == 27

assert validation_checks_failed == 0

assert notebook51_validation_checks_df[
    "passed"
].astype(bool).all()

assert policy_class_count == 6

assert raw_feature_count == 12

assert encoded_feature_count == 18

assert simulator_symbols_loaded == 11

assert (
    notebook51_runtime_binding_summary[
        "status"
    ]
    == "SIMULATOR_MODULES_BOUND"
)

assert (
    notebook51_runtime_binding_summary[
        "relative_imports_resolved"
    ]
    is True
)

assert full_battle_completed is True

assert full_battle_turns > 0

assert (
    len(
        notebook51_battle_turns_df
    )
    == full_battle_turns
)

assert (
    len(
        notebook51_policy_history_df
    )
    == full_battle_turns
)

assert isinstance(
    notebook51_battle_transcript,
    str,
)

assert len(
    notebook51_battle_transcript
) > 0

assert next_notebook == (
    "Notebook 52 — Tournament Policy Evaluation"
)

assert next_stage == (
    "TOURNAMENT_POLICY_EVALUATION"
)


print()
print("HANDOFF VALIDATION SUMMARY")
print("-" * 100)

print(
    f"Status                        : "
    f"{handoff_status}"
)

print(
    f"Policy classes                : "
    f"{policy_class_count}"
)

print(
    f"Raw policy features           : "
    f"{raw_feature_count}"
)

print(
    f"Encoded features              : "
    f"{encoded_feature_count}"
)

print(
    f"Simulator symbols loaded      : "
    f"{simulator_symbols_loaded}"
)

print(
    f"Validation checks passed      : "
    f"{validation_checks_passed}"
)

print(
    f"Validation checks failed      : "
    f"{validation_checks_failed}"
)

print(
    f"Full battle completed         : "
    f"{full_battle_completed}"
)

print(
    f"Full battle winner            : "
    f"{full_battle_winner}"
)

print(
    f"Full battle turns             : "
    f"{full_battle_turns}"
)

print(
    f"Next stage                    : "
    f"{next_stage}"
)


print()
print(
    "✅ SECTION 1B NOTEBOOK 51 HANDOFF VALIDATION PASSED"
)


# # Section 2 — Tournament Runtime Reconstruction
# 
# ## Section 2A — Load Policy and Simulator Artifacts
# 
# #### This section reloads the trained Notebook 50 policy artifacts and reconstructs the production simulator runtime required for tournament evaluation.
# 
# #### It validates:
# 
# - policy model loading,
# - preprocessing pipeline loading,
# - label encoder loading,
# - raw and encoded feature schema,
# - package-aware simulator module loading,
# - required production runtime symbols.

# In[3]:


# ======================================================================================
# SECTION 2A — LOAD POLICY AND SIMULATOR ARTIFACTS
# ======================================================================================

import importlib.util
import types

print("=" * 100)
print("SECTION 2A — LOAD POLICY AND SIMULATOR ARTIFACTS")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Resolve Notebook 50 model files from the validated Notebook 51 handoff
# --------------------------------------------------------------------------------------

POLICY_MODEL_FILE = Path(
    notebook51_handoff_manifest[
        "policy_model_file"
    ]
)

POLICY_PREPROCESSOR_FILE = Path(
    notebook51_handoff_manifest[
        "policy_preprocessor_file"
    ]
)

POLICY_LABEL_ENCODER_FILE = Path(
    notebook51_handoff_manifest[
        "policy_label_encoder_file"
    ]
)

POLICY_FEATURE_NAMES_FILE = Path(
    notebook51_handoff_manifest[
        "policy_feature_names_file"
    ]
)

POLICY_METADATA_FILE = (
    NOTEBOOK50_MODEL_DIR
    / "model_metadata.json"
)


required_policy_files = {
    "policy_model":
        POLICY_MODEL_FILE,

    "preprocessor":
        POLICY_PREPROCESSOR_FILE,

    "label_encoder":
        POLICY_LABEL_ENCODER_FILE,

    "feature_names":
        POLICY_FEATURE_NAMES_FILE,

    "model_metadata":
        POLICY_METADATA_FILE,
}


policy_file_rows = []

for artifact_name, artifact_path in (
    required_policy_files.items()
):

    artifact_path = Path(
        artifact_path
    )

    policy_file_rows.append(
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
        }
    )


policy_file_inventory_df = pd.DataFrame(
    policy_file_rows
)

policy_file_inventory_df[
    "nonempty"
] = (
    policy_file_inventory_df[
        "size_bytes"
    ] > 0
)


print()
print("POLICY ARTIFACT INVENTORY")
print("-" * 100)

display(
    policy_file_inventory_df
)


assert policy_file_inventory_df[
    "exists"
].all()

assert policy_file_inventory_df[
    "is_file"
].all()

assert policy_file_inventory_df[
    "nonempty"
].all()


# --------------------------------------------------------------------------------------
# 2. Load trained policy artifacts
# --------------------------------------------------------------------------------------

policy_model = joblib.load(
    POLICY_MODEL_FILE
)

policy_preprocessor = joblib.load(
    POLICY_PREPROCESSOR_FILE
)

policy_label_encoder = joblib.load(
    POLICY_LABEL_ENCODER_FILE
)


with open(
    POLICY_FEATURE_NAMES_FILE,
    "r",
    encoding="utf-8",
) as file:

    policy_feature_names = json.load(
        file
    )


with open(
    POLICY_METADATA_FILE,
    "r",
    encoding="utf-8",
) as file:

    policy_model_metadata = json.load(
        file
    )


if isinstance(
    policy_feature_names,
    dict,
):

    policy_feature_names = (
        policy_feature_names.get(
            "feature_names",
            policy_feature_names.get(
                "encoded_feature_names",
                [],
            ),
        )
    )


policy_feature_names = list(
    policy_feature_names
)


# --------------------------------------------------------------------------------------
# 3. Define deployment feature schema
# --------------------------------------------------------------------------------------

POLICY_RAW_FEATURE_COLUMNS = [
    "current_side",
    "side_mode",
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
]


print()
print("POLICY MODEL PROFILE")
print("-" * 100)

print(
    f"Model type              : "
    f"{type(policy_model).__name__}"
)

print(
    f"Policy classes          : "
    f"{list(policy_label_encoder.classes_)}"
)

print(
    f"Policy class count      : "
    f"{len(policy_label_encoder.classes_)}"
)

print(
    f"Raw feature count       : "
    f"{len(POLICY_RAW_FEATURE_COLUMNS)}"
)

print(
    f"Encoded feature count   : "
    f"{len(policy_feature_names)}"
)


assert len(
    policy_label_encoder.classes_
) == policy_class_count

assert len(
    POLICY_RAW_FEATURE_COLUMNS
) == raw_feature_count

assert len(
    policy_feature_names
) == encoded_feature_count

assert (
    int(
        policy_model_metadata[
            "encoded_feature_count"
        ]
    )
    == encoded_feature_count
)

assert (
    "variant_name"
    not in POLICY_RAW_FEATURE_COLUMNS
)


# --------------------------------------------------------------------------------------
# 4. Resolve production simulator source files
# --------------------------------------------------------------------------------------

SIMULATOR_RUNTIME_PACKAGE = (
    "notebook52_simulator_runtime"
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


simulator_source_rows = []

for module_name, module_path in (
    simulator_source_files.items()
):

    simulator_source_rows.append(
        {
            "module_name":
                module_name,

            "exists":
                module_path.exists(),

            "size_bytes":
                (
                    module_path.stat().st_size
                    if module_path.exists()
                    else 0
                ),

            "path":
                str(
                    module_path.resolve()
                ),
        }
    )


simulator_source_inventory_df = pd.DataFrame(
    simulator_source_rows
)


print()
print("SIMULATOR SOURCE INVENTORY")
print("-" * 100)

display(
    simulator_source_inventory_df
)


assert simulator_source_inventory_df[
    "exists"
].all()


# --------------------------------------------------------------------------------------
# 5. Remove stale runtime modules
# --------------------------------------------------------------------------------------

for module_name in list(
    sys.modules
):

    if (
        module_name
        == SIMULATOR_RUNTIME_PACKAGE
        or module_name.startswith(
            SIMULATOR_RUNTIME_PACKAGE
            + "."
        )
    ):
        del sys.modules[
            module_name
        ]


# --------------------------------------------------------------------------------------
# 6. Create synthetic package for relative imports
# --------------------------------------------------------------------------------------

simulator_runtime_package = types.ModuleType(
    SIMULATOR_RUNTIME_PACKAGE
)

simulator_runtime_package.__path__ = [
    str(
        SRC_DIR.resolve()
    )
]

simulator_runtime_package.__package__ = (
    SIMULATOR_RUNTIME_PACKAGE
)

sys.modules[
    SIMULATOR_RUNTIME_PACKAGE
] = simulator_runtime_package


def load_simulator_submodule(
    short_module_name: str,
    module_path: Path,
) -> Any:
    """
    Load one simulator source file as a package submodule.
    """

    qualified_module_name = (
        f"{SIMULATOR_RUNTIME_PACKAGE}."
        f"{short_module_name}"
    )

    module_spec = (
        importlib.util.spec_from_file_location(
            qualified_module_name,
            module_path,
        )
    )

    if (
        module_spec is None
        or module_spec.loader is None
    ):
        raise ImportError(
            "Unable to build module specification for "
            f"{module_path}"
        )

    module_object = (
        importlib.util.module_from_spec(
            module_spec
        )
    )

    sys.modules[
        qualified_module_name
    ] = module_object

    try:

        module_spec.loader.exec_module(
            module_object
        )

    except Exception:

        sys.modules.pop(
            qualified_module_name,
            None,
        )

        raise

    setattr(
        simulator_runtime_package,
        short_module_name,
        module_object,
    )

    return module_object


# --------------------------------------------------------------------------------------
# 7. Load simulator modules in dependency order
# --------------------------------------------------------------------------------------

battle_state_module = (
    load_simulator_submodule(
        "battle_state",
        simulator_source_files[
            "battle_state"
        ],
    )
)

agent_decision_module = (
    load_simulator_submodule(
        "agent_decision",
        simulator_source_files[
            "agent_decision"
        ],
    )
)

simulator_module = (
    load_simulator_submodule(
        "simulator",
        simulator_source_files[
            "simulator"
        ],
    )
)

battle_agent_module = (
    load_simulator_submodule(
        "battle_agent",
        simulator_source_files[
            "battle_agent"
        ],
    )
)

battle_simulation_module = (
    load_simulator_submodule(
        "battle_simulation",
        simulator_source_files[
            "battle_simulation"
        ],
    )
)


# --------------------------------------------------------------------------------------
# 8. Resolve production runtime symbols
# --------------------------------------------------------------------------------------

SimulatorPokemonState = getattr(
    battle_state_module,
    "PokemonState",
)

SimulatorPlayerState = getattr(
    battle_state_module,
    "PlayerState",
)

SimulatorBattleState = getattr(
    battle_state_module,
    "BattleState",
)

SimulatorAgentDecision = getattr(
    agent_decision_module,
    "AgentDecision",
)

simulator_apply_move = getattr(
    simulator_module,
    "apply_move",
)

SimulatorPokemonBattleAgent = getattr(
    battle_agent_module,
    "PokemonBattleAgent",
)

simulator_determine_winner = getattr(
    battle_simulation_module,
    "determine_battle_winner",
)

simulator_run_battle = getattr(
    battle_simulation_module,
    "simulate_ai_battle",
)

simulator_create_transcript = getattr(
    battle_simulation_module,
    "create_battle_transcript",
)

SimulatorBattleTurnRecord = getattr(
    battle_simulation_module,
    "BattleTurnRecord",
)

SimulatorBattleSimulationResult = getattr(
    battle_simulation_module,
    "BattleSimulationResult",
)


runtime_symbols = {
    "PokemonState":
        SimulatorPokemonState,

    "PlayerState":
        SimulatorPlayerState,

    "BattleState":
        SimulatorBattleState,

    "AgentDecision":
        SimulatorAgentDecision,

    "PokemonBattleAgent":
        SimulatorPokemonBattleAgent,

    "apply_move":
        simulator_apply_move,

    "determine_battle_winner":
        simulator_determine_winner,

    "simulate_ai_battle":
        simulator_run_battle,

    "create_battle_transcript":
        simulator_create_transcript,

    "BattleTurnRecord":
        SimulatorBattleTurnRecord,

    "BattleSimulationResult":
        SimulatorBattleSimulationResult,
}


runtime_symbol_rows = []

for symbol_name, symbol_object in (
    runtime_symbols.items()
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

            "module":
                getattr(
                    symbol_object,
                    "__module__",
                    None,
                ),
        }
    )


runtime_symbol_inventory_df = pd.DataFrame(
    runtime_symbol_rows
)


print()
print("SIMULATOR RUNTIME SYMBOLS")
print("-" * 100)

display(
    runtime_symbol_inventory_df
)


# --------------------------------------------------------------------------------------
# 9. Validate runtime consistency
# --------------------------------------------------------------------------------------

assert runtime_symbol_inventory_df[
    "exists"
].all()

assert len(
    runtime_symbol_inventory_df
) == simulator_symbols_loaded

assert inspect.isclass(
    SimulatorPokemonState
)

assert inspect.isclass(
    SimulatorPlayerState
)

assert inspect.isclass(
    SimulatorBattleState
)

assert inspect.isclass(
    SimulatorAgentDecision
)

assert callable(
    simulator_apply_move
)

assert callable(
    simulator_run_battle
)

assert callable(
    simulator_determine_winner
)

assert (
    getattr(
        simulator_module,
        "BattleState",
    )
    is SimulatorBattleState
)

assert (
    getattr(
        battle_simulation_module,
        "BattleState",
    )
    is SimulatorBattleState
)


print()
print("RUNTIME LOAD SUMMARY")
print("-" * 100)

print(
    f"Policy model loaded         : "
    f"{policy_model is not None}"
)

print(
    f"Policy preprocessor loaded  : "
    f"{policy_preprocessor is not None}"
)

print(
    f"Label encoder loaded        : "
    f"{policy_label_encoder is not None}"
)

print(
    f"Runtime package             : "
    f"{SIMULATOR_RUNTIME_PACKAGE}"
)

print(
    f"Simulator symbols loaded    : "
    f"{len(runtime_symbol_inventory_df)}"
)

print(
    f"Relative imports resolved   : "
    f"{True}"
)


print()
print(
    "✅ SECTION 2A POLICY AND SIMULATOR ARTIFACT LOADING PASSED"
)


# ## Section 2B — Tournament Policy Agent Reconstruction
# 
# #### This section reconstructs the tournament-ready policy agent from the validated Notebook 50 artifacts.
# 
# #### It provides:
# 
# - live battle-state feature extraction,
# - policy inference,
# - legal-move generation,
# - legal-action masking,
# - simulator-compatible `AgentDecision` output,
# - policy decision history and fallback tracking.

# In[5]:


# ======================================================================================
# SECTION 2B — TOURNAMENT POLICY AGENT RECONSTRUCTION
# ======================================================================================

print("=" * 100)
print("SECTION 2B — TOURNAMENT POLICY AGENT RECONSTRUCTION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Generic helper functions
# --------------------------------------------------------------------------------------

def read_first(
    source: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Read the first available value from a mapping or object.
    """

    for name in names:

        if isinstance(source, Mapping):

            if name in source:
                return source[name]

        elif hasattr(source, name):

            return getattr(
                source,
                name,
            )

    return default


def normalize_side_name(
    side_value: Any,
) -> str:
    """
    Normalize a side value to Player or Opponent.
    """

    normalized_value = str(
        side_value
    ).strip().lower()

    if normalized_value in {
        "opponent",
        "enemy",
        "second",
        "p2",
        "player2",
    }:
        return "Opponent"

    return "Player"


def normalize_move_name(
    move_value: Any,
) -> str:
    """
    Normalize a move name for matching.
    """

    if move_value is None:
        return ""

    if isinstance(
        move_value,
        Mapping,
    ):

        move_value = read_first(
            move_value,
            [
                "name",
                "move_name",
                "Move Name",
                "action",
            ],
            default="",
        )

    elif not isinstance(
        move_value,
        str,
    ):

        move_value = read_first(
            move_value,
            [
                "name",
                "move_name",
                "action",
            ],
            default=move_value,
        )

    return " ".join(
        str(
            move_value
        )
        .strip()
        .lower()
        .split()
    )


def move_display_name(
    move_value: Any,
) -> str:
    """
    Return a readable move name.
    """

    if isinstance(
        move_value,
        Mapping,
    ):

        return str(
            read_first(
                move_value,
                [
                    "name",
                    "move_name",
                    "Move Name",
                    "action",
                ],
                default="Unknown Move",
            )
        )

    if isinstance(
        move_value,
        str,
    ):
        return move_value

    return str(
        read_first(
            move_value,
            [
                "name",
                "move_name",
                "action",
            ],
            default="Unknown Move",
        )
    )


# --------------------------------------------------------------------------------------
# 2. Resolve the current active Pokémon
# --------------------------------------------------------------------------------------

def tournament_current_active_pokemon(
    state: Any,
) -> Any:
    """
    Return the active Pokémon for the current side.
    """

    current_side = normalize_side_name(
        read_first(
            state,
            [
                "current_player",
                "current_side",
            ],
            default="Player",
        )
    )

    if current_side == "Opponent":
        return state.opponent.active

    return state.player.active


# --------------------------------------------------------------------------------------
# 3. Generate legal moves from the production state
# --------------------------------------------------------------------------------------

def tournament_generate_legal_moves(
    state: Any,
) -> List[Dict[str, Any]]:
    """
    Return all affordable attacks for the current active Pokémon.

    If no attack is affordable, return Pass.
    """

    active_pokemon = (
        tournament_current_active_pokemon(
            state
        )
    )

    attached_energy = int(
        read_first(
            active_pokemon,
            [
                "attached_energy",
                "energy",
            ],
            default=0,
        )
        or 0
    )

    card_record = read_first(
        active_pokemon,
        [
            "card",
            "card_record",
        ],
        default={},
    )

    attacks = read_first(
        card_record,
        [
            "attacks",
            "Attacks",
        ],
        default=[],
    )

    if not isinstance(
        attacks,
        list,
    ):
        attacks = []

    legal_moves: List[
        Dict[str, Any]
    ] = []

    for attack in attacks:

        if not isinstance(
            attack,
            Mapping,
        ):
            continue

        move_name = read_first(
            attack,
            [
                "Move Name",
                "name",
                "move_name",
            ],
            default="Unknown Move",
        )

        try:

            energy_cost = int(
                read_first(
                    attack,
                    [
                        "energy_cost",
                        "Energy Cost",
                        "cost",
                    ],
                    default=0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            energy_cost = 0

        try:

            move_damage = float(
                read_first(
                    attack,
                    [
                        "damage_numeric",
                        "damage",
                        "Damage",
                    ],
                    default=0.0,
                )
                or 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            move_damage = 0.0

        if attached_energy < energy_cost:
            continue

        legal_moves.append(
            {
                "name":
                    str(
                        move_name
                    ),

                "damage":
                    float(
                        move_damage
                    ),

                "energy_cost":
                    int(
                        energy_cost
                    ),

                "effect":
                    read_first(
                        attack,
                        [
                            "Effect Explanation",
                            "effect",
                        ],
                        default=None,
                    ),
            }
        )

    if not legal_moves:

        legal_moves.append(
            {
                "name":
                    "Pass",

                "damage":
                    0.0,

                "energy_cost":
                    0,

                "effect":
                    "No legal attack was available.",
            }
        )

    return legal_moves


# --------------------------------------------------------------------------------------
# 4. Live policy-feature extraction
# --------------------------------------------------------------------------------------

def battle_state_to_policy_features(
    battle_state: Any,
    legal_moves: Optional[
        Sequence[Any]
    ] = None,
    side_mode: str = "PRESERVE",
) -> pd.DataFrame:
    """
    Convert a live production BattleState into the raw Notebook 50 schema.
    """

    current_side = normalize_side_name(
        read_first(
            battle_state,
            [
                "current_player",
                "current_side",
            ],
            default="Player",
        )
    )

    resolved_legal_moves = list(
        legal_moves
        if legal_moves is not None
        else tournament_generate_legal_moves(
            battle_state
        )
    )

    player_active = (
        battle_state.player.active
    )

    opponent_active = (
        battle_state.opponent.active
    )

    player_card = read_first(
        player_active,
        [
            "card",
            "card_record",
        ],
        default={},
    )

    opponent_card = read_first(
        opponent_active,
        [
            "card",
            "card_record",
        ],
        default={},
    )

    player_card_name = read_first(
        player_card,
        [
            "Card Name",
            "name",
            "card_name",
        ],
        default="Unknown",
    )

    opponent_card_name = read_first(
        opponent_card,
        [
            "Card Name",
            "name",
            "card_name",
        ],
        default="Unknown",
    )

    feature_record = {
        "current_side":
            current_side,

        "side_mode":
            str(
                side_mode
            ).strip().upper(),

        "player_card":
            str(
                player_card_name
            ),

        "opponent_card":
            str(
                opponent_card_name
            ),

        "turn_number":
            float(
                battle_state.turn_number
            ),

        "player_energy":
            float(
                player_active.attached_energy
            ),

        "opponent_energy":
            float(
                opponent_active.attached_energy
            ),

        "player_damage":
            float(
                player_active.damage
            ),

        "opponent_damage":
            float(
                opponent_active.damage
            ),

        "prize_cards_remaining":
            float(
                (
                    battle_state.player
                    .prize_cards_remaining
                )
                if current_side == "Player"
                else (
                    battle_state.opponent
                    .prize_cards_remaining
                )
            ),

        "hand_size":
            float(
                (
                    battle_state.player
                    .hand_size
                )
                if current_side == "Player"
                else (
                    battle_state.opponent
                    .hand_size
                )
            ),

        "legal_move_count":
            float(
                len(
                    resolved_legal_moves
                )
            ),
    }

    feature_df = pd.DataFrame(
        [
            feature_record
        ]
    )

    return feature_df[
        POLICY_RAW_FEATURE_COLUMNS
    ]


# --------------------------------------------------------------------------------------
# 5. Raw policy inference
# --------------------------------------------------------------------------------------

def predict_policy_move(
    policy_features: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run the Notebook 50 preprocessor and policy model.
    """

    transformed_features = (
        policy_preprocessor.transform(
            policy_features[
                POLICY_RAW_FEATURE_COLUMNS
            ]
        )
    )

    predicted_label_id = int(
        policy_model.predict(
            transformed_features
        )[0]
    )

    predicted_move = str(
        policy_label_encoder.inverse_transform(
            [
                predicted_label_id
            ]
        )[0]
    )

    class_probabilities = (
        policy_model.predict_proba(
            transformed_features
        )[0]
    )

    probabilities = {
        str(
            move_name
        ):
            float(
                probability
            )
        for move_name, probability
        in zip(
            policy_label_encoder.classes_,
            class_probabilities,
        )
    }

    confidence = float(
        max(
            probabilities.values()
        )
    )

    return {
        "predicted_label_id":
            predicted_label_id,

        "predicted_move":
            predicted_move,

        "confidence":
            confidence,

        "probabilities":
            probabilities,
    }


# --------------------------------------------------------------------------------------
# 6. Legal-action masking
# --------------------------------------------------------------------------------------

def select_legal_policy_move(
    battle_state: Any,
    legal_moves: Sequence[Any],
    side_mode: str = "PRESERVE",
) -> Dict[str, Any]:
    """
    Choose the highest-probability legal move.
    """

    resolved_legal_moves = list(
        legal_moves
    )

    if not resolved_legal_moves:

        raise ValueError(
            "At least one legal move is required."
        )

    policy_features = (
        battle_state_to_policy_features(
            battle_state=battle_state,
            legal_moves=resolved_legal_moves,
            side_mode=side_mode,
        )
    )

    prediction = predict_policy_move(
        policy_features
    )

    legal_move_lookup = {
        normalize_move_name(
            move
        ):
            move
        for move in resolved_legal_moves
        if normalize_move_name(
            move
        )
    }

    matching_rows = []

    for move_name, probability in (
        prediction[
            "probabilities"
        ].items()
    ):

        normalized_name = (
            normalize_move_name(
                move_name
            )
        )

        if normalized_name in legal_move_lookup:

            matching_rows.append(
                {
                    "policy_move":
                        str(
                            move_name
                        ),

                    "normalized_move":
                        normalized_name,

                    "probability":
                        float(
                            probability
                        ),
                }
            )

    fallback_used = False
    fallback_reason = None

    if matching_rows:

        legal_probability_df = (
            pd.DataFrame(
                matching_rows
            )
            .sort_values(
                [
                    "probability",
                    "policy_move",
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

        probability_total = float(
            legal_probability_df[
                "probability"
            ].sum()
        )

        if probability_total > 0.0:

            legal_probability_df[
                "masked_probability"
            ] = (
                legal_probability_df[
                    "probability"
                ]
                / probability_total
            )

        else:

            legal_probability_df[
                "masked_probability"
            ] = 0.0

            fallback_used = True

            fallback_reason = (
                "All matching legal actions had zero policy probability."
            )

        selected_row = (
            legal_probability_df
            .iloc[0]
        )

        selected_move = (
            legal_move_lookup[
                selected_row[
                    "normalized_move"
                ]
            ]
        )

        selected_probability = float(
            selected_row[
                "probability"
            ]
        )

        selected_masked_probability = float(
            selected_row[
                "masked_probability"
            ]
        )

    else:

        selected_move = (
            resolved_legal_moves[0]
        )

        selected_probability = 0.0

        selected_masked_probability = 1.0

        fallback_used = True

        fallback_reason = (
            "No legal move matched a trained policy class."
        )

    return {
        "selected_move":
            selected_move,

        "selected_move_name":
            move_display_name(
                selected_move
            ),

        "raw_predicted_move":
            prediction[
                "predicted_move"
            ],

        "raw_confidence":
            float(
                prediction[
                    "confidence"
                ]
            ),

        "selected_probability":
            selected_probability,

        "selected_masked_probability":
            selected_masked_probability,

        "fallback_used":
            fallback_used,

        "fallback_reason":
            fallback_reason,

        "policy_features":
            policy_features,

        "raw_probabilities":
            prediction[
                "probabilities"
            ],
    }


# --------------------------------------------------------------------------------------
# 7. Tournament policy agent
# --------------------------------------------------------------------------------------

class TournamentPolicyAgent:
    """
    Production-compatible policy agent for tournament evaluation.
    """

    def __init__(
        self,
        name: str = "Notebook50TournamentPolicy",
        side_mode: str = "PRESERVE",
    ) -> None:

        self.name = str(
            name
        )

        self.side_mode = str(
            side_mode
        ).strip().upper()

        self.total_decisions = 0

        self.fallback_decisions = 0

        self.decision_records: List[
            Dict[str, Any]
        ] = []

    def reset(
        self,
    ) -> None:

        self.total_decisions = 0

        self.fallback_decisions = 0

        self.decision_records = []

    def choose_move(
        self,
        state: Any,
        depth: int = 0,
    ) -> Any:
        """
        Return a production AgentDecision.
        """

        legal_moves = (
            tournament_generate_legal_moves(
                state
            )
        )

        selection = (
            select_legal_policy_move(
                battle_state=state,
                legal_moves=legal_moves,
                side_mode=self.side_mode,
            )
        )

        selected_move = (
            selection[
                "selected_move"
            ]
        )

        decision = SimulatorAgentDecision(
            move=selected_move,
            score=float(
                selection[
                    "raw_confidence"
                ]
            ),
            search_depth=int(
                depth
            ),
            nodes=1,
            principal_variation=[
                selected_move
            ],
        )

        self.total_decisions += 1

        if selection[
            "fallback_used"
        ]:

            self.fallback_decisions += 1

        self.decision_records.append(
            {
                "turn_number":
                    int(
                        state.turn_number
                    ),

                "current_player":
                    normalize_side_name(
                        state.current_player
                    ),

                "legal_moves":
                    [
                        move_display_name(
                            move
                        )
                        for move in legal_moves
                    ],

                "selected_move":
                    selection[
                        "selected_move_name"
                    ],

                "raw_predicted_move":
                    selection[
                        "raw_predicted_move"
                    ],

                "confidence":
                    float(
                        selection[
                            "raw_confidence"
                        ]
                    ),

                "selected_probability":
                    float(
                        selection[
                            "selected_probability"
                        ]
                    ),

                "fallback_used":
                    bool(
                        selection[
                            "fallback_used"
                        ]
                    ),

                "fallback_reason":
                    selection[
                        "fallback_reason"
                    ],
            }
        )

        return decision

    def history_dataframe(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            self.decision_records
        )

    def statistics(
        self,
    ) -> Dict[str, Any]:

        fallback_rate = (
            self.fallback_decisions
            / self.total_decisions
            if self.total_decisions > 0
            else 0.0
        )

        return {
            "agent_name":
                self.name,

            "side_mode":
                self.side_mode,

            "total_decisions":
                int(
                    self.total_decisions
                ),

            "fallback_decisions":
                int(
                    self.fallback_decisions
                ),

            "fallback_rate":
                float(
                    fallback_rate
                ),

            "history_rows":
                int(
                    len(
                        self.decision_records
                    )
                ),
        }


# --------------------------------------------------------------------------------------
# 8. Instantiate and validate
# --------------------------------------------------------------------------------------

tournament_policy_agent = TournamentPolicyAgent(
    name="Notebook50TournamentPolicy",
    side_mode="PRESERVE",
)


print()
print("TOURNAMENT POLICY AGENT PROFILE")
print("-" * 100)

print(
    f"Agent name             : "
    f"{tournament_policy_agent.name}"
)

print(
    f"Side mode              : "
    f"{tournament_policy_agent.side_mode}"
)

print(
    f"Policy classes         : "
    f"{list(policy_label_encoder.classes_)}"
)

print(
    f"Raw feature count      : "
    f"{len(POLICY_RAW_FEATURE_COLUMNS)}"
)


assert callable(
    tournament_policy_agent.choose_move
)

assert tournament_policy_agent.total_decisions == 0

assert tournament_policy_agent.fallback_decisions == 0

assert len(
    tournament_policy_agent.decision_records
) == 0


print()
print(
    "✅ SECTION 2B TOURNAMENT POLICY AGENT RECONSTRUCTION PASSED"
)


# # Section 3 — Tournament Scenario Construction
# 
# ## Section 3A — Tournament Scenario Catalog
# 
# #### This section defines the reusable battle scenarios used in tournament evaluation.
# 
# #### The catalog varies:
# 
# - player and opponent Pokémon,
# - starting side,
# - HP conditions,
# - attached Energy,
# - prize-card pressure,
# - hand size,
# - early-, mid-, and late-game states.
# 
# #### Each scenario is converted into a production `BattleState`.

# In[6]:


# ======================================================================================
# SECTION 3A — TOURNAMENT SCENARIO CATALOG
# ======================================================================================

print("=" * 100)
print("SECTION 3A — TOURNAMENT SCENARIO CATALOG")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Reusable card records
# --------------------------------------------------------------------------------------

TOURNAMENT_CARD_LIBRARY = {
    "Bulbasaur": {
        "Card ID": 650,
        "Card Name": "Bulbasaur",
        "name": "Bulbasaur",
        "HP": 80.0,
        "hp": 80.0,
        "attacks": [
            {
                "Move Name": "Bind Down",
                "damage_numeric": 10.0,
                "energy_cost": 1,
                "Effect Explanation":
                    "During your opponent’s next turn, "
                    "the Defending Pokémon can’t retreat.",
            }
        ],
    },

    "Eevee": {
        "Card ID": 43,
        "Card Name": "Eevee",
        "name": "Eevee",
        "HP": 50.0,
        "hp": 50.0,
        "attacks": [
            {
                "Move Name": "Ascension",
                "damage_numeric": 0.0,
                "energy_cost": 1,
                "Effect Explanation":
                    "Search your deck for a card that evolves "
                    "from this Pokémon and put it onto this Pokémon.",
            },
            {
                "Move Name": "Quick Attack",
                "damage_numeric": 20.0,
                "energy_cost": 3,
                "Effect Explanation":
                    "Flip a coin. If heads, this attack does "
                    "20 more damage.",
            },
        ],
    },

    "Charmander": {
        "Card ID": 4,
        "Card Name": "Charmander",
        "name": "Charmander",
        "HP": 80.0,
        "hp": 80.0,
        "attacks": [
            {
                "Move Name": "Live Coal",
                "damage_numeric": 20.0,
                "energy_cost": 1,
                "Effect Explanation": None,
            }
        ],
    },

    "Meowth": {
        "Card ID": 52,
        "Card Name": "Meowth",
        "name": "Meowth",
        "HP": 60.0,
        "hp": 60.0,
        "attacks": [
            {
                "Move Name": "Tuck Tail",
                "damage_numeric": 0.0,
                "energy_cost": 1,
                "Effect Explanation":
                    "Move all Energy from this Pokémon "
                    "to one of your Benched Pokémon.",
            }
        ],
    },
}


# --------------------------------------------------------------------------------------
# 2. Scenario specification
# --------------------------------------------------------------------------------------

TOURNAMENT_SCENARIO_SPECS = [
    {
        "scenario_id": "T52_S001",
        "scenario_name": "Bulbasaur_vs_Eevee_Player_Start",
        "player_card": "Bulbasaur",
        "opponent_card": "Eevee",
        "current_player": "Player",
        "turn_number": 1,
        "player_hp": 80.0,
        "opponent_hp": 50.0,
        "player_energy": 3,
        "opponent_energy": 3,
        "player_prizes": 1,
        "opponent_prizes": 1,
        "player_hand": 5,
        "opponent_hand": 5,
    },
    {
        "scenario_id": "T52_S002",
        "scenario_name": "Bulbasaur_vs_Eevee_Opponent_Start",
        "player_card": "Bulbasaur",
        "opponent_card": "Eevee",
        "current_player": "Opponent",
        "turn_number": 1,
        "player_hp": 80.0,
        "opponent_hp": 50.0,
        "player_energy": 3,
        "opponent_energy": 3,
        "player_prizes": 1,
        "opponent_prizes": 1,
        "player_hand": 5,
        "opponent_hand": 5,
    },
    {
        "scenario_id": "T52_S003",
        "scenario_name": "Charmander_vs_Eevee_Player_Start",
        "player_card": "Charmander",
        "opponent_card": "Eevee",
        "current_player": "Player",
        "turn_number": 1,
        "player_hp": 80.0,
        "opponent_hp": 50.0,
        "player_energy": 3,
        "opponent_energy": 3,
        "player_prizes": 1,
        "opponent_prizes": 1,
        "player_hand": 5,
        "opponent_hand": 5,
    },
    {
        "scenario_id": "T52_S004",
        "scenario_name": "Charmander_vs_Eevee_Opponent_Start",
        "player_card": "Charmander",
        "opponent_card": "Eevee",
        "current_player": "Opponent",
        "turn_number": 1,
        "player_hp": 80.0,
        "opponent_hp": 50.0,
        "player_energy": 3,
        "opponent_energy": 3,
        "player_prizes": 1,
        "opponent_prizes": 1,
        "player_hand": 5,
        "opponent_hand": 5,
    },
    {
        "scenario_id": "T52_S005",
        "scenario_name": "Meowth_vs_Bulbasaur_Player_Start",
        "player_card": "Meowth",
        "opponent_card": "Bulbasaur",
        "current_player": "Player",
        "turn_number": 1,
        "player_hp": 60.0,
        "opponent_hp": 80.0,
        "player_energy": 2,
        "opponent_energy": 2,
        "player_prizes": 2,
        "opponent_prizes": 2,
        "player_hand": 4,
        "opponent_hand": 4,
    },
    {
        "scenario_id": "T52_S006",
        "scenario_name": "Meowth_vs_Bulbasaur_Opponent_Start",
        "player_card": "Meowth",
        "opponent_card": "Bulbasaur",
        "current_player": "Opponent",
        "turn_number": 1,
        "player_hp": 60.0,
        "opponent_hp": 80.0,
        "player_energy": 2,
        "opponent_energy": 2,
        "player_prizes": 2,
        "opponent_prizes": 2,
        "player_hand": 4,
        "opponent_hand": 4,
    },
    {
        "scenario_id": "T52_S007",
        "scenario_name": "Critical_HP_Bulbasaur_vs_Eevee",
        "player_card": "Bulbasaur",
        "opponent_card": "Eevee",
        "current_player": "Opponent",
        "turn_number": 9,
        "player_hp": 20.0,
        "opponent_hp": 20.0,
        "player_energy": 3,
        "opponent_energy": 3,
        "player_prizes": 1,
        "opponent_prizes": 1,
        "player_hand": 3,
        "opponent_hand": 3,
    },
    {
        "scenario_id": "T52_S008",
        "scenario_name": "Low_Energy_Eevee_vs_Charmander",
        "player_card": "Eevee",
        "opponent_card": "Charmander",
        "current_player": "Player",
        "turn_number": 2,
        "player_hp": 50.0,
        "opponent_hp": 80.0,
        "player_energy": 1,
        "opponent_energy": 1,
        "player_prizes": 4,
        "opponent_prizes": 4,
        "player_hand": 6,
        "opponent_hand": 6,
    },
]


# --------------------------------------------------------------------------------------
# 3. Build production BattleState objects
# --------------------------------------------------------------------------------------

def build_tournament_state(
    scenario_spec: Mapping[str, Any],
) -> Any:
    """
    Convert one scenario specification into a production BattleState.
    """

    player_card_name = str(
        scenario_spec[
            "player_card"
        ]
    )

    opponent_card_name = str(
        scenario_spec[
            "opponent_card"
        ]
    )

    player_card = deepcopy(
        TOURNAMENT_CARD_LIBRARY[
            player_card_name
        ]
    )

    opponent_card = deepcopy(
        TOURNAMENT_CARD_LIBRARY[
            opponent_card_name
        ]
    )

    player_max_hp = float(
        player_card[
            "HP"
        ]
    )

    opponent_max_hp = float(
        opponent_card[
            "HP"
        ]
    )

    player_current_hp = float(
        scenario_spec[
            "player_hp"
        ]
    )

    opponent_current_hp = float(
        scenario_spec[
            "opponent_hp"
        ]
    )

    player_damage = max(
        0.0,
        player_max_hp
        - player_current_hp,
    )

    opponent_damage = max(
        0.0,
        opponent_max_hp
        - opponent_current_hp,
    )

    player_pokemon = SimulatorPokemonState(
        card=player_card,
        current_hp=player_current_hp,
        attached_energy=int(
            scenario_spec[
                "player_energy"
            ]
        ),
        status=None,
        damage=player_damage,
        is_active=True,
    )

    opponent_pokemon = SimulatorPokemonState(
        card=opponent_card,
        current_hp=opponent_current_hp,
        attached_energy=int(
            scenario_spec[
                "opponent_energy"
            ]
        ),
        status=None,
        damage=opponent_damage,
        is_active=True,
    )

    player_state = SimulatorPlayerState(
        active=player_pokemon,
        bench=[],
        prize_cards_remaining=int(
            scenario_spec[
                "player_prizes"
            ]
        ),
        hand_size=int(
            scenario_spec[
                "player_hand"
            ]
        ),
    )

    opponent_state = SimulatorPlayerState(
        active=opponent_pokemon,
        bench=[],
        prize_cards_remaining=int(
            scenario_spec[
                "opponent_prizes"
            ]
        ),
        hand_size=int(
            scenario_spec[
                "opponent_hand"
            ]
        ),
    )

    return SimulatorBattleState(
        player=player_state,
        opponent=opponent_state,
        turn_number=int(
            scenario_spec[
                "turn_number"
            ]
        ),
        current_player=str(
            scenario_spec[
                "current_player"
            ]
        ),
    )


tournament_scenarios = {}

scenario_catalog_rows = []

for scenario_spec in TOURNAMENT_SCENARIO_SPECS:

    scenario_state = build_tournament_state(
        scenario_spec
    )

    scenario_id = str(
        scenario_spec[
            "scenario_id"
        ]
    )

    tournament_scenarios[
        scenario_id
    ] = scenario_state

    scenario_catalog_rows.append(
        {
            **dict(
                scenario_spec
            ),
            "player_damage":
                float(
                    scenario_state
                    .player
                    .active
                    .damage
                ),
            "opponent_damage":
                float(
                    scenario_state
                    .opponent
                    .active
                    .damage
                ),
            "player_legal_moves":
                [
                    move_display_name(
                        move
                    )
                    for move in (
                        tournament_generate_legal_moves(
                            scenario_state
                        )
                        if scenario_state.current_player
                        == "Player"
                        else []
                    )
                ],
            "opponent_legal_moves":
                [
                    move_display_name(
                        move
                    )
                    for move in (
                        tournament_generate_legal_moves(
                            scenario_state
                        )
                        if scenario_state.current_player
                        == "Opponent"
                        else []
                    )
                ],
        }
    )


tournament_scenario_catalog_df = pd.DataFrame(
    scenario_catalog_rows
)


print()
print("TOURNAMENT SCENARIO CATALOG")
print("-" * 100)

display(
    tournament_scenario_catalog_df
)


# --------------------------------------------------------------------------------------
# 4. Validate scenario catalog
# --------------------------------------------------------------------------------------

assert len(
    tournament_scenarios
) == len(
    TOURNAMENT_SCENARIO_SPECS
)

assert tournament_scenario_catalog_df[
    "scenario_id"
].is_unique

assert tournament_scenario_catalog_df[
    "scenario_name"
].is_unique

assert set(
    tournament_scenario_catalog_df[
        "current_player"
    ]
) == {
    "Player",
    "Opponent",
}

assert tournament_scenario_catalog_df[
    "player_hp"
].gt(0).all()

assert tournament_scenario_catalog_df[
    "opponent_hp"
].gt(0).all()

assert all(
    isinstance(
        state,
        SimulatorBattleState,
    )
    for state in tournament_scenarios.values()
)


# --------------------------------------------------------------------------------------
# 5. Save scenario catalog
# --------------------------------------------------------------------------------------

SECTION3_REPORT_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section3"
)

SECTION3_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION3A_SCENARIO_CATALOG_FILE = (
    SECTION3_REPORT_DIR
    / "section3a_tournament_scenario_catalog.csv"
)

tournament_scenario_catalog_df.to_csv(
    SECTION3A_SCENARIO_CATALOG_FILE,
    index=False,
)


print()
print("SCENARIO SUMMARY")
print("-" * 100)

print(
    f"Scenario count         : "
    f"{len(tournament_scenarios)}"
)

print(
    f"Starting Player states : "
    f"{(
        tournament_scenario_catalog_df[
            'current_player'
        ]
        == 'Player'
    ).sum()}"
)

print(
    f"Starting Opponent states: "
    f"{(
        tournament_scenario_catalog_df[
            'current_player'
        ]
        == 'Opponent'
    ).sum()}"
)

print(
    f"Unique player cards    : "
    f"{tournament_scenario_catalog_df['player_card'].nunique()}"
)

print(
    f"Unique opponent cards  : "
    f"{tournament_scenario_catalog_df['opponent_card'].nunique()}"
)

print()
print("SAVED SECTION 3A REPORT")
print("-" * 100)

print(
    SECTION3A_SCENARIO_CATALOG_FILE
)

print()
print(
    "✅ SECTION 3A TOURNAMENT SCENARIO CATALOG PASSED"
)


# ## Section 3B — Tournament Scenario Validation
# 
# #### This section validates every tournament scenario before running repeated battles.
# 
# #### For each scenario, it confirms that:
# 
# - the state is a valid production `BattleState`,
# - legal moves can be generated,
# - deployment features can be extracted,
# - policy inference succeeds,
# - the selected move is legal,
# - confidence and probability values are valid,
# - fallback behavior is recorded,
# - the original scenario state remains unchanged.
# 
# #### Scenarios that pass this section are approved for tournament execution.

# In[7]:


# ======================================================================================
# SECTION 3B — TOURNAMENT SCENARIO VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 3B — TOURNAMENT SCENARIO VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Helper for compact state snapshots
# --------------------------------------------------------------------------------------

def snapshot_tournament_state(
    state: Any,
) -> Dict[str, Any]:
    """
    Return the key state values used to verify that validation does not mutate
    the original tournament scenario.
    """

    return {
        "turn_number":
            int(
                state.turn_number
            ),

        "current_player":
            str(
                state.current_player
            ),

        "player_hp":
            float(
                state.player.active.current_hp
            ),

        "opponent_hp":
            float(
                state.opponent.active.current_hp
            ),

        "player_damage":
            float(
                state.player.active.damage
            ),

        "opponent_damage":
            float(
                state.opponent.active.damage
            ),

        "player_energy":
            int(
                state.player.active.attached_energy
            ),

        "opponent_energy":
            int(
                state.opponent.active.attached_energy
            ),

        "player_prizes":
            int(
                state.player.prize_cards_remaining
            ),

        "opponent_prizes":
            int(
                state.opponent.prize_cards_remaining
            ),

        "player_hand":
            int(
                state.player.hand_size
            ),

        "opponent_hand":
            int(
                state.opponent.hand_size
            ),
    }


# --------------------------------------------------------------------------------------
# 2. Validate every scenario
# --------------------------------------------------------------------------------------

scenario_validation_rows = []
scenario_probability_rows = []
scenario_validation_failures = []

for scenario_spec in TOURNAMENT_SCENARIO_SPECS:

    scenario_id = str(
        scenario_spec[
            "scenario_id"
        ]
    )

    scenario_name = str(
        scenario_spec[
            "scenario_name"
        ]
    )

    scenario_state = (
        tournament_scenarios[
            scenario_id
        ]
    )

    initial_snapshot = (
        snapshot_tournament_state(
            scenario_state
        )
    )

    try:

        # --------------------------------------------------------------------------
        # Production state validation
        # --------------------------------------------------------------------------

        assert isinstance(
            scenario_state,
            SimulatorBattleState,
        )

        assert isinstance(
            scenario_state.player,
            SimulatorPlayerState,
        )

        assert isinstance(
            scenario_state.opponent,
            SimulatorPlayerState,
        )

        assert isinstance(
            scenario_state.player.active,
            SimulatorPokemonState,
        )

        assert isinstance(
            scenario_state.opponent.active,
            SimulatorPokemonState,
        )


        # --------------------------------------------------------------------------
        # Legal moves
        # --------------------------------------------------------------------------

        legal_moves = (
            tournament_generate_legal_moves(
                scenario_state
            )
        )

        legal_move_names = [
            move_display_name(
                move
            )
            for move in legal_moves
        ]

        assert len(
            legal_moves
        ) > 0

        assert all(
            isinstance(
                move,
                Mapping,
            )
            for move in legal_moves
        )

        assert all(
            bool(
                move_display_name(
                    move
                ).strip()
            )
            for move in legal_moves
        )


        # --------------------------------------------------------------------------
        # Feature extraction
        # --------------------------------------------------------------------------

        policy_features_df = (
            battle_state_to_policy_features(
                battle_state=scenario_state,
                legal_moves=legal_moves,
                side_mode="PRESERVE",
            )
        )

        assert list(
            policy_features_df.columns
        ) == POLICY_RAW_FEATURE_COLUMNS

        assert len(
            policy_features_df
        ) == 1

        assert not policy_features_df[
            POLICY_RAW_FEATURE_COLUMNS
        ].isna().any().any()


        # --------------------------------------------------------------------------
        # Raw policy inference
        # --------------------------------------------------------------------------

        raw_prediction = (
            predict_policy_move(
                policy_features_df
            )
        )

        raw_predicted_move = str(
            raw_prediction[
                "predicted_move"
            ]
        )

        raw_confidence = float(
            raw_prediction[
                "confidence"
            ]
        )

        raw_probabilities = dict(
            raw_prediction[
                "probabilities"
            ]
        )

        probability_sum = float(
            sum(
                raw_probabilities.values()
            )
        )

        assert (
            raw_predicted_move
            in set(
                policy_label_encoder.classes_
            )
        )

        assert (
            0.0
            <= raw_confidence
            <= 1.0
        )

        assert np.isclose(
            probability_sum,
            1.0,
            atol=1e-9,
        )


        # --------------------------------------------------------------------------
        # Legal-action masking
        # --------------------------------------------------------------------------

        legal_selection = (
            select_legal_policy_move(
                battle_state=scenario_state,
                legal_moves=legal_moves,
                side_mode="PRESERVE",
            )
        )

        selected_move = (
            legal_selection[
                "selected_move"
            ]
        )

        selected_move_name = str(
            legal_selection[
                "selected_move_name"
            ]
        )

        fallback_used = bool(
            legal_selection[
                "fallback_used"
            ]
        )

        fallback_reason = (
            legal_selection[
                "fallback_reason"
            ]
        )

        selected_probability = float(
            legal_selection[
                "selected_probability"
            ]
        )

        selected_masked_probability = float(
            legal_selection[
                "selected_masked_probability"
            ]
        )

        assert (
            normalize_move_name(
                selected_move
            )
            in {
                normalize_move_name(
                    legal_move
                )
                for legal_move
                in legal_moves
            }
        )

        assert (
            selected_move_name
            in legal_move_names
        )

        assert (
            0.0
            <= selected_probability
            <= 1.0
        )

        assert (
            0.0
            <= selected_masked_probability
            <= 1.0
        )


        # --------------------------------------------------------------------------
        # AgentDecision contract
        # --------------------------------------------------------------------------

        validation_agent = (
            TournamentPolicyAgent(
                name=(
                    f"ValidationAgent_{scenario_id}"
                ),
                side_mode="PRESERVE",
            )
        )

        agent_decision = (
            validation_agent.choose_move(
                state=scenario_state,
                depth=0,
            )
        )

        assert isinstance(
            agent_decision,
            SimulatorAgentDecision,
        )

        assert (
            normalize_move_name(
                agent_decision.move
            )
            in {
                normalize_move_name(
                    legal_move
                )
                for legal_move
                in legal_moves
            }
        )

        assert np.isfinite(
            float(
                agent_decision.score
            )
        )

        assert (
            agent_decision.search_depth
            == 0
        )

        assert (
            agent_decision.nodes
            == 1
        )

        assert len(
            agent_decision.principal_variation
        ) == 1


        # --------------------------------------------------------------------------
        # Original-state preservation
        # --------------------------------------------------------------------------

        final_snapshot = (
            snapshot_tournament_state(
                scenario_state
            )
        )

        state_unchanged = (
            final_snapshot
            == initial_snapshot
        )

        assert state_unchanged, (
            f"Scenario {scenario_id} was mutated during validation."
        )


        # --------------------------------------------------------------------------
        # Store probability records
        # --------------------------------------------------------------------------

        for policy_move, probability in (
            raw_probabilities.items()
        ):

            scenario_probability_rows.append(
                {
                    "scenario_id":
                        scenario_id,

                    "scenario_name":
                        scenario_name,

                    "current_player":
                        str(
                            scenario_state.current_player
                        ),

                    "policy_move":
                        str(
                            policy_move
                        ),

                    "probability":
                        float(
                            probability
                        ),

                    "is_legal":
                        (
                            normalize_move_name(
                                policy_move
                            )
                            in {
                                normalize_move_name(
                                    legal_move
                                )
                                for legal_move
                                in legal_moves
                            }
                        ),

                    "is_selected":
                        (
                            normalize_move_name(
                                policy_move
                            )
                            == normalize_move_name(
                                selected_move
                            )
                        ),
                }
            )


        # --------------------------------------------------------------------------
        # Store successful validation row
        # --------------------------------------------------------------------------

        scenario_validation_rows.append(
            {
                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "player_card":
                    str(
                        scenario_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        scenario_spec[
                            "opponent_card"
                        ]
                    ),

                "current_player":
                    str(
                        scenario_state.current_player
                    ),

                "turn_number":
                    int(
                        scenario_state.turn_number
                    ),

                "legal_move_count":
                    int(
                        len(
                            legal_moves
                        )
                    ),

                "legal_moves":
                    legal_move_names,

                "raw_predicted_move":
                    raw_predicted_move,

                "selected_move":
                    selected_move_name,

                "raw_confidence":
                    raw_confidence,

                "selected_probability":
                    selected_probability,

                "selected_masked_probability":
                    selected_masked_probability,

                "fallback_used":
                    fallback_used,

                "fallback_reason":
                    fallback_reason,

                "probability_sum":
                    probability_sum,

                "agent_decision_type":
                    type(
                        agent_decision
                    ).__name__,

                "selected_move_legal":
                    True,

                "state_unchanged":
                    state_unchanged,

                "validation_passed":
                    True,

                "error_type":
                    None,

                "error_message":
                    None,
            }
        )

    except Exception as error:

        scenario_validation_failures.append(
            {
                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

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

        scenario_validation_rows.append(
            {
                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "player_card":
                    str(
                        scenario_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        scenario_spec[
                            "opponent_card"
                        ]
                    ),

                "current_player":
                    str(
                        scenario_spec[
                            "current_player"
                        ]
                    ),

                "turn_number":
                    int(
                        scenario_spec[
                            "turn_number"
                        ]
                    ),

                "legal_move_count":
                    None,

                "legal_moves":
                    None,

                "raw_predicted_move":
                    None,

                "selected_move":
                    None,

                "raw_confidence":
                    None,

                "selected_probability":
                    None,

                "selected_masked_probability":
                    None,

                "fallback_used":
                    None,

                "fallback_reason":
                    None,

                "probability_sum":
                    None,

                "agent_decision_type":
                    None,

                "selected_move_legal":
                    False,

                "state_unchanged":
                    (
                        snapshot_tournament_state(
                            scenario_state
                        )
                        == initial_snapshot
                    ),

                "validation_passed":
                    False,

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


# --------------------------------------------------------------------------------------
# 3. Build validation dataframes
# --------------------------------------------------------------------------------------

tournament_scenario_validation_df = (
    pd.DataFrame(
        scenario_validation_rows
    )
)

tournament_scenario_probabilities_df = (
    pd.DataFrame(
        scenario_probability_rows
    )
)

tournament_scenario_failures_df = (
    pd.DataFrame(
        scenario_validation_failures,
        columns=[
            "scenario_id",
            "scenario_name",
            "error_type",
            "error_message",
        ],
    )
)


# --------------------------------------------------------------------------------------
# 4. Display validation results
# --------------------------------------------------------------------------------------

print()
print("SCENARIO VALIDATION RESULTS")
print("-" * 100)

validation_preview_columns = [
    "scenario_id",
    "scenario_name",
    "current_player",
    "legal_move_count",
    "legal_moves",
    "raw_predicted_move",
    "selected_move",
    "raw_confidence",
    "fallback_used",
    "selected_move_legal",
    "state_unchanged",
    "validation_passed",
]

display(
    tournament_scenario_validation_df[
        validation_preview_columns
    ]
)


print()
print("SCENARIO VALIDATION FAILURES")
print("-" * 100)

if tournament_scenario_failures_df.empty:

    print(
        "No scenario validation failures."
    )

else:

    display(
        tournament_scenario_failures_df
    )


print()
print("POLICY PROBABILITY SAMPLE")
print("-" * 100)

display(
    tournament_scenario_probabilities_df.head(
        24
    )
)


# --------------------------------------------------------------------------------------
# 5. Validation assertions
# --------------------------------------------------------------------------------------

assert len(
    tournament_scenario_validation_df
) == len(
    TOURNAMENT_SCENARIO_SPECS
)

assert tournament_scenario_validation_df[
    "scenario_id"
].is_unique

assert tournament_scenario_failures_df.empty, (
    "One or more tournament scenarios failed validation."
)

assert tournament_scenario_validation_df[
    "validation_passed"
].all()

assert tournament_scenario_validation_df[
    "selected_move_legal"
].all()

assert tournament_scenario_validation_df[
    "state_unchanged"
].all()

assert tournament_scenario_validation_df[
    "legal_move_count"
].gt(0).all()

assert tournament_scenario_validation_df[
    "raw_confidence"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert tournament_scenario_validation_df[
    "selected_probability"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert tournament_scenario_validation_df[
    "selected_masked_probability"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert np.allclose(
    tournament_scenario_validation_df[
        "probability_sum"
    ].astype(float),
    1.0,
    atol=1e-9,
)

assert (
    len(
        tournament_scenario_probabilities_df
    )
    == (
        len(
            TOURNAMENT_SCENARIO_SPECS
        )
        * len(
            policy_label_encoder.classes_
        )
    )
)


# --------------------------------------------------------------------------------------
# 6. Save validation reports
# --------------------------------------------------------------------------------------

SECTION3B_SCENARIO_VALIDATION_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_tournament_scenario_validation.csv"
)

SECTION3B_SCENARIO_PROBABILITIES_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_tournament_scenario_probabilities.csv"
)

SECTION3B_SCENARIO_FAILURES_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_tournament_scenario_failures.csv"
)

SECTION3B_SUMMARY_FILE = (
    SECTION3_REPORT_DIR
    / "section3b_tournament_scenario_summary.json"
)


tournament_scenario_validation_df.to_csv(
    SECTION3B_SCENARIO_VALIDATION_FILE,
    index=False,
)

tournament_scenario_probabilities_df.to_csv(
    SECTION3B_SCENARIO_PROBABILITIES_FILE,
    index=False,
)

tournament_scenario_failures_df.to_csv(
    SECTION3B_SCENARIO_FAILURES_FILE,
    index=False,
)


section3b_summary = {
    "status":
        "TOURNAMENT_SCENARIOS_VALIDATED",

    "scenario_count":
        int(
            len(
                tournament_scenario_validation_df
            )
        ),

    "scenarios_passed":
        int(
            tournament_scenario_validation_df[
                "validation_passed"
            ].sum()
        ),

    "scenarios_failed":
        int(
            (
                ~tournament_scenario_validation_df[
                    "validation_passed"
                ]
            ).sum()
        ),

    "legal_selections":
        int(
            tournament_scenario_validation_df[
                "selected_move_legal"
            ].sum()
        ),

    "fallback_scenarios":
        int(
            tournament_scenario_validation_df[
                "fallback_used"
            ].astype(bool).sum()
        ),

    "average_confidence":
        float(
            tournament_scenario_validation_df[
                "raw_confidence"
            ].mean()
        ),

    "minimum_confidence":
        float(
            tournament_scenario_validation_df[
                "raw_confidence"
            ].min()
        ),

    "maximum_confidence":
        float(
            tournament_scenario_validation_df[
                "raw_confidence"
            ].max()
        ),

    "average_legal_move_count":
        float(
            tournament_scenario_validation_df[
                "legal_move_count"
            ].mean()
        ),

    "states_unchanged":
        bool(
            tournament_scenario_validation_df[
                "state_unchanged"
            ].all()
        ),

    "next_stage":
        "TOURNAMENT_BATTLE_EXECUTION",
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


# --------------------------------------------------------------------------------------
# 7. Final summary
# --------------------------------------------------------------------------------------

print()
print("SECTION 3B SUMMARY")
print("-" * 100)

for key, value in (
    section3b_summary.items()
):

    print(
        f"{key:30}: {value}"
    )


print()
print("SAVED SECTION 3B REPORTS")
print("-" * 100)

print(
    SECTION3B_SCENARIO_VALIDATION_FILE
)

print(
    SECTION3B_SCENARIO_PROBABILITIES_FILE
)

print(
    SECTION3B_SCENARIO_FAILURES_FILE
)

print(
    SECTION3B_SUMMARY_FILE
)


print()
print(
    "✅ SECTION 3B TOURNAMENT SCENARIO VALIDATION PASSED"
)


# # Section 4 — Tournament Battle Execution
# 
# ## Section 4A — Single-Pass Scenario Tournament
# 
# #### This section executes one complete production-simulator battle for every validated tournament scenario.
# 
# #### For each battle, it records:
# 
# - winner,
# - starting side,
# - battle length,
# - final HP,
# - policy decision count,
# - fallback count and rate,
# - average confidence,
# - termination reason,
# - turn-level actions,
# - transcript output.
# 
# #### This first pass validates the tournament runner across the full scenario catalog before repeated tournament rounds begin.

# In[8]:


# ======================================================================================
# SECTION 4A — SINGLE-PASS SCENARIO TOURNAMENT
# ======================================================================================

print("=" * 100)
print("SECTION 4A — SINGLE-PASS SCENARIO TOURNAMENT")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Tournament configuration
# --------------------------------------------------------------------------------------

SECTION4A_SEARCH_DEPTH = 0
SECTION4A_MAX_TURNS = 30
SECTION4A_VERBOSE = False


SECTION4_REPORT_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section4"
)

SECTION4_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SECTION4A_TRANSCRIPT_DIR = (
    SECTION4_REPORT_DIR
    / "transcripts"
)

SECTION4A_TRANSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("TOURNAMENT CONFIGURATION")
print("-" * 100)

print(
    f"Scenario count         : "
    f"{len(TOURNAMENT_SCENARIO_SPECS)}"
)

print(
    f"Search depth           : "
    f"{SECTION4A_SEARCH_DEPTH}"
)

print(
    f"Maximum turns          : "
    f"{SECTION4A_MAX_TURNS}"
)

print(
    f"Verbose battles        : "
    f"{SECTION4A_VERBOSE}"
)


# --------------------------------------------------------------------------------------
# 2. Helper for converting production turn records
# --------------------------------------------------------------------------------------

def turn_record_to_dictionary(
    turn_record: Any,
) -> Dict[str, Any]:
    """
    Convert one production turn record into a plain dictionary.
    """

    if is_dataclass(
        turn_record
    ):
        return asdict(
            turn_record
        )

    if isinstance(
        turn_record,
        Mapping,
    ):
        return dict(
            turn_record
        )

    output = {}

    for attribute_name in dir(
        turn_record
    ):

        if attribute_name.startswith(
            "_"
        ):
            continue

        attribute_value = getattr(
            turn_record,
            attribute_name,
        )

        if callable(
            attribute_value
        ):
            continue

        output[
            attribute_name
        ] = attribute_value

    return output


# --------------------------------------------------------------------------------------
# 3. Execute all scenarios
# --------------------------------------------------------------------------------------

tournament_battle_rows = []
tournament_turn_rows = []
tournament_policy_rows = []
tournament_failure_rows = []

for battle_number, scenario_spec in enumerate(
    TOURNAMENT_SCENARIO_SPECS,
    start=1,
):

    scenario_id = str(
        scenario_spec[
            "scenario_id"
        ]
    )

    scenario_name = str(
        scenario_spec[
            "scenario_name"
        ]
    )

    print()
    print(
        f"Running {battle_number:02d}/"
        f"{len(TOURNAMENT_SCENARIO_SPECS):02d} — "
        f"{scenario_id}: {scenario_name}"
    )

    initial_state = deepcopy(
        tournament_scenarios[
            scenario_id
        ]
    )

    battle_agent = TournamentPolicyAgent(
        name=(
            f"Notebook50TournamentPolicy_"
            f"{scenario_id}"
        ),
        side_mode="PRESERVE",
    )

    try:

        battle_result = simulator_run_battle(
            initial_state=initial_state,
            agent=battle_agent,
            search_depth=SECTION4A_SEARCH_DEPTH,
            max_turns=SECTION4A_MAX_TURNS,
            verbose=SECTION4A_VERBOSE,
        )

        assert isinstance(
            battle_result,
            SimulatorBattleSimulationResult,
        )

        final_state = read_first(
            battle_result,
            [
                "final_state",
                "state",
            ],
            default=None,
        )

        turn_records = list(
            read_first(
                battle_result,
                [
                    "turns",
                    "turn_records",
                    "history",
                ],
                default=[],
            )
            or []
        )

        winner = read_first(
            battle_result,
            [
                "winner",
                "winning_side",
            ],
            default=None,
        )

        if (
            winner is None
            and final_state is not None
        ):
            winner = simulator_determine_winner(
                final_state
            )

        termination_reason = read_first(
            battle_result,
            [
                "stop_reason",
                "termination_reason",
                "reason",
                "status",
            ],
            default=None,
        )

        transcript = simulator_create_transcript(
            battle_result
        )

        policy_history_df = (
            battle_agent.history_dataframe()
        )

        agent_statistics = (
            battle_agent.statistics()
        )

        decision_count = int(
            agent_statistics[
                "total_decisions"
            ]
        )

        fallback_count = int(
            agent_statistics[
                "fallback_decisions"
            ]
        )

        fallback_rate = float(
            agent_statistics[
                "fallback_rate"
            ]
        )

        average_confidence = (
            float(
                policy_history_df[
                    "confidence"
                ].mean()
            )
            if not policy_history_df.empty
            else 0.0
        )

        minimum_confidence = (
            float(
                policy_history_df[
                    "confidence"
                ].min()
            )
            if not policy_history_df.empty
            else 0.0
        )

        maximum_confidence = (
            float(
                policy_history_df[
                    "confidence"
                ].max()
            )
            if not policy_history_df.empty
            else 0.0
        )

        final_player_hp = float(
            final_state
            .player
            .active
            .current_hp
        )

        final_opponent_hp = float(
            final_state
            .opponent
            .active
            .current_hp
        )

        battle_turn_count = int(
            len(
                turn_records
            )
        )

        starting_side = str(
            scenario_spec[
                "current_player"
            ]
        )

        starter_won = (
            winner == starting_side
        )

        player_won = (
            winner == "Player"
        )

        opponent_won = (
            winner == "Opponent"
        )

        draw = (
            winner == "Draw"
            or winner is None
        )

        transcript_file = (
            SECTION4A_TRANSCRIPT_DIR
            / (
                f"{scenario_id}_"
                f"battle_transcript.txt"
            )
        )

        transcript_file.write_text(
            transcript,
            encoding="utf-8",
        )


        # --------------------------------------------------------------------------
        # Battle summary row
        # --------------------------------------------------------------------------

        tournament_battle_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "player_card":
                    str(
                        scenario_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        scenario_spec[
                            "opponent_card"
                        ]
                    ),

                "starting_side":
                    starting_side,

                "starting_turn_number":
                    int(
                        scenario_spec[
                            "turn_number"
                        ]
                    ),

                "winner":
                    winner,

                "starter_won":
                    bool(
                        starter_won
                    ),

                "player_won":
                    bool(
                        player_won
                    ),

                "opponent_won":
                    bool(
                        opponent_won
                    ),

                "draw":
                    bool(
                        draw
                    ),

                "recorded_turns":
                    battle_turn_count,

                "final_turn_number":
                    int(
                        final_state.turn_number
                    ),

                "final_player_hp":
                    final_player_hp,

                "final_opponent_hp":
                    final_opponent_hp,

                "policy_decisions":
                    decision_count,

                "fallback_decisions":
                    fallback_count,

                "fallback_rate":
                    fallback_rate,

                "average_confidence":
                    average_confidence,

                "minimum_confidence":
                    minimum_confidence,

                "maximum_confidence":
                    maximum_confidence,

                "termination_reason":
                    termination_reason,

                "transcript_file":
                    str(
                        transcript_file
                    ),

                "battle_completed":
                    True,

                "error_type":
                    None,

                "error_message":
                    None,
            }
        )


        # --------------------------------------------------------------------------
        # Production turn records
        # --------------------------------------------------------------------------

        for turn_index, turn_record in enumerate(
            turn_records,
            start=1,
        ):

            record_dict = (
                turn_record_to_dictionary(
                    turn_record
                )
            )

            tournament_turn_rows.append(
                {
                    "battle_number":
                        battle_number,

                    "scenario_id":
                        scenario_id,

                    "scenario_name":
                        scenario_name,

                    "record_number":
                        turn_index,

                    "turn_number":
                        read_first(
                            record_dict,
                            [
                                "turn_number",
                                "turn",
                            ],
                            default=turn_index,
                        ),

                    "acting_side":
                        read_first(
                            record_dict,
                            [
                                "acting_side",
                                "current_player",
                                "side",
                            ],
                            default=None,
                        ),

                    "pokemon_name":
                        read_first(
                            record_dict,
                            [
                                "pokemon_name",
                                "active_pokemon",
                            ],
                            default=None,
                        ),

                    "move_name":
                        read_first(
                            record_dict,
                            [
                                "move_name",
                                "action_name",
                            ],
                            default=None,
                        ),

                    "damage":
                        read_first(
                            record_dict,
                            [
                                "damage",
                            ],
                            default=None,
                        ),

                    "score":
                        read_first(
                            record_dict,
                            [
                                "score",
                                "decision_score",
                            ],
                            default=None,
                        ),

                    "search_depth":
                        read_first(
                            record_dict,
                            [
                                "search_depth",
                                "depth",
                            ],
                            default=None,
                        ),

                    "nodes":
                        read_first(
                            record_dict,
                            [
                                "nodes",
                                "search_nodes",
                            ],
                            default=None,
                        ),

                    "player_hp_after":
                        read_first(
                            record_dict,
                            [
                                "player_hp_after",
                                "player_hp",
                            ],
                            default=None,
                        ),

                    "opponent_hp_after":
                        read_first(
                            record_dict,
                            [
                                "opponent_hp_after",
                                "opponent_hp",
                            ],
                            default=None,
                        ),

                    "next_side":
                        read_first(
                            record_dict,
                            [
                                "next_side",
                                "next_player",
                            ],
                            default=None,
                        ),
                }
            )


        # --------------------------------------------------------------------------
        # Policy decision history
        # --------------------------------------------------------------------------

        if not policy_history_df.empty:

            policy_history_copy = (
                policy_history_df.copy()
            )

            policy_history_copy.insert(
                0,
                "scenario_name",
                scenario_name,
            )

            policy_history_copy.insert(
                0,
                "scenario_id",
                scenario_id,
            )

            policy_history_copy.insert(
                0,
                "battle_number",
                battle_number,
            )

            tournament_policy_rows.extend(
                policy_history_copy.to_dict(
                    orient="records"
                )
            )


        print(
            f"  Winner={winner} | "
            f"Turns={battle_turn_count} | "
            f"Fallbacks={fallback_count} | "
            f"Average confidence="
            f"{average_confidence:.4f}"
        )

    except Exception as error:

        tournament_failure_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

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

        tournament_battle_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "player_card":
                    str(
                        scenario_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        scenario_spec[
                            "opponent_card"
                        ]
                    ),

                "starting_side":
                    str(
                        scenario_spec[
                            "current_player"
                        ]
                    ),

                "starting_turn_number":
                    int(
                        scenario_spec[
                            "turn_number"
                        ]
                    ),

                "winner":
                    None,

                "starter_won":
                    False,

                "player_won":
                    False,

                "opponent_won":
                    False,

                "draw":
                    False,

                "recorded_turns":
                    0,

                "final_turn_number":
                    None,

                "final_player_hp":
                    None,

                "final_opponent_hp":
                    None,

                "policy_decisions":
                    0,

                "fallback_decisions":
                    0,

                "fallback_rate":
                    None,

                "average_confidence":
                    None,

                "minimum_confidence":
                    None,

                "maximum_confidence":
                    None,

                "termination_reason":
                    None,

                "transcript_file":
                    None,

                "battle_completed":
                    False,

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
            f"  FAILED: "
            f"{type(error).__name__}: {error}"
        )


# --------------------------------------------------------------------------------------
# 4. Build tournament result dataframes
# --------------------------------------------------------------------------------------

section4a_battle_results_df = pd.DataFrame(
    tournament_battle_rows
)

section4a_turn_results_df = pd.DataFrame(
    tournament_turn_rows
)

section4a_policy_history_df = pd.DataFrame(
    tournament_policy_rows
)

section4a_failures_df = pd.DataFrame(
    tournament_failure_rows,
    columns=[
        "battle_number",
        "scenario_id",
        "scenario_name",
        "error_type",
        "error_message",
    ],
)


# --------------------------------------------------------------------------------------
# 5. Display battle results
# --------------------------------------------------------------------------------------

print()
print("TOURNAMENT BATTLE RESULTS")
print("-" * 100)

battle_preview_columns = [
    "battle_number",
    "scenario_id",
    "scenario_name",
    "starting_side",
    "winner",
    "starter_won",
    "recorded_turns",
    "final_player_hp",
    "final_opponent_hp",
    "policy_decisions",
    "fallback_decisions",
    "fallback_rate",
    "average_confidence",
    "termination_reason",
    "battle_completed",
]

display(
    section4a_battle_results_df[
        battle_preview_columns
    ]
)


print()
print("TOURNAMENT FAILURES")
print("-" * 100)

if section4a_failures_df.empty:

    print(
        "No tournament battle failures."
    )

else:

    display(
        section4a_failures_df
    )


print()
print("TURN RECORD SAMPLE")
print("-" * 100)

display(
    section4a_turn_results_df.head(
        30
    )
)


print()
print("POLICY HISTORY SAMPLE")
print("-" * 100)

display(
    section4a_policy_history_df.head(
        30
    )
)


# --------------------------------------------------------------------------------------
# 6. Validation checks
# --------------------------------------------------------------------------------------

assert len(
    section4a_battle_results_df
) == len(
    TOURNAMENT_SCENARIO_SPECS
)

assert section4a_battle_results_df[
    "scenario_id"
].is_unique

assert section4a_failures_df.empty, (
    "One or more tournament battles failed."
)

assert section4a_battle_results_df[
    "battle_completed"
].all()

assert section4a_battle_results_df[
    "recorded_turns"
].gt(0).all()

assert section4a_battle_results_df[
    "policy_decisions"
].gt(0).all()

assert section4a_battle_results_df[
    "fallback_rate"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert section4a_battle_results_df[
    "average_confidence"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert section4a_battle_results_df[
    "final_player_hp"
].ge(0.0).all()

assert section4a_battle_results_df[
    "final_opponent_hp"
].ge(0.0).all()

assert section4a_battle_results_df[
    "winner"
].isin(
        [
            "Player",
            "Opponent",
            "Draw",
        ]
    ).all()

assert (
    len(
        section4a_turn_results_df
    )
    == int(
        section4a_battle_results_df[
            "recorded_turns"
        ].sum()
    )
)

assert (
    len(
        section4a_policy_history_df
    )
    == int(
        section4a_battle_results_df[
            "policy_decisions"
        ].sum()
    )
)


# --------------------------------------------------------------------------------------
# 7. Save tournament reports
# --------------------------------------------------------------------------------------

SECTION4A_BATTLE_RESULTS_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_single_pass_battle_results.csv"
)

SECTION4A_TURN_RESULTS_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_single_pass_turn_results.csv"
)

SECTION4A_POLICY_HISTORY_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_single_pass_policy_history.csv"
)

SECTION4A_FAILURES_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_single_pass_failures.csv"
)

SECTION4A_SUMMARY_FILE = (
    SECTION4_REPORT_DIR
    / "section4a_single_pass_summary.json"
)


section4a_battle_results_df.to_csv(
    SECTION4A_BATTLE_RESULTS_FILE,
    index=False,
)

section4a_turn_results_df.to_csv(
    SECTION4A_TURN_RESULTS_FILE,
    index=False,
)

section4a_policy_history_df.to_csv(
    SECTION4A_POLICY_HISTORY_FILE,
    index=False,
)

section4a_failures_df.to_csv(
    SECTION4A_FAILURES_FILE,
    index=False,
)


section4a_summary = {
    "status":
        "SINGLE_PASS_TOURNAMENT_COMPLETED",

    "battle_count":
        int(
            len(
                section4a_battle_results_df
            )
        ),

    "battles_completed":
        int(
            section4a_battle_results_df[
                "battle_completed"
            ].sum()
        ),

    "battles_failed":
        int(
            (
                ~section4a_battle_results_df[
                    "battle_completed"
                ]
            ).sum()
        ),

    "player_wins":
        int(
            section4a_battle_results_df[
                "player_won"
            ].sum()
        ),

    "opponent_wins":
        int(
            section4a_battle_results_df[
                "opponent_won"
            ].sum()
        ),

    "draws":
        int(
            section4a_battle_results_df[
                "draw"
            ].sum()
        ),

    "starter_wins":
        int(
            section4a_battle_results_df[
                "starter_won"
            ].sum()
        ),

    "starter_win_rate":
        float(
            section4a_battle_results_df[
                "starter_won"
            ].mean()
        ),

    "average_turns":
        float(
            section4a_battle_results_df[
                "recorded_turns"
            ].mean()
        ),

    "total_policy_decisions":
        int(
            section4a_battle_results_df[
                "policy_decisions"
            ].sum()
        ),

    "total_fallback_decisions":
        int(
            section4a_battle_results_df[
                "fallback_decisions"
            ].sum()
        ),

    "overall_fallback_rate":
        float(
            (
                section4a_battle_results_df[
                    "fallback_decisions"
                ].sum()
                / section4a_battle_results_df[
                    "policy_decisions"
                ].sum()
            )
            if section4a_battle_results_df[
                "policy_decisions"
            ].sum() > 0
            else 0.0
        ),

    "average_confidence":
        float(
            section4a_policy_history_df[
                "confidence"
            ].mean()
        ),

    "next_stage":
        "TOURNAMENT_RESULT_ANALYSIS",
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


# --------------------------------------------------------------------------------------
# 8. Final summary
# --------------------------------------------------------------------------------------

print()
print("SECTION 4A SUMMARY")
print("-" * 100)

for key, value in (
    section4a_summary.items()
):

    print(
        f"{key:32}: {value}"
    )


print()
print("SAVED SECTION 4A REPORTS")
print("-" * 100)

print(SECTION4A_BATTLE_RESULTS_FILE)
print(SECTION4A_TURN_RESULTS_FILE)
print(SECTION4A_POLICY_HISTORY_FILE)
print(SECTION4A_FAILURES_FILE)
print(SECTION4A_SUMMARY_FILE)

print()
print(
    "✅ SECTION 4A SINGLE-PASS SCENARIO TOURNAMENT PASSED"
)


# ## Section 4B — Single-Pass Tournament Result Analysis
# 
# #### This section analyzes the completed single-pass tournament.
# 
# #### It separates:
# 
# - Player-side and Opponent-side outcomes,
# - starting-side advantage,
# - matchup performance,
# - battle length,
# - policy confidence,
# - fallback behavior,
# - raw-prediction legality,
# - legal-action masking dependence,
# - action-selection frequency.
# 
# #### Because the same trained policy controls both sides, win counts are interpreted as scenario and matchup outcomes rather than direct policy-versus-opponent performance.

# In[9]:


# ======================================================================================
# SECTION 4B — SINGLE-PASS TOURNAMENT RESULT ANALYSIS
# ======================================================================================

print("=" * 100)
print("SECTION 4B — SINGLE-PASS TOURNAMENT RESULT ANALYSIS")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Prepare analysis directory
# --------------------------------------------------------------------------------------

SECTION4B_ANALYSIS_DIR = (
    SECTION4_REPORT_DIR
    / "analysis"
)

SECTION4B_ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 2. Add useful derived battle-level fields
# --------------------------------------------------------------------------------------

section4b_battle_analysis_df = (
    section4a_battle_results_df.copy()
)

section4b_battle_analysis_df[
    "winner_card"
] = np.where(
    section4b_battle_analysis_df[
        "winner"
    ] == "Player",
    section4b_battle_analysis_df[
        "player_card"
    ],
    np.where(
        section4b_battle_analysis_df[
            "winner"
        ] == "Opponent",
        section4b_battle_analysis_df[
            "opponent_card"
        ],
        "Draw",
    ),
)

section4b_battle_analysis_df[
    "loser_card"
] = np.where(
    section4b_battle_analysis_df[
        "winner"
    ] == "Player",
    section4b_battle_analysis_df[
        "opponent_card"
    ],
    np.where(
        section4b_battle_analysis_df[
            "winner"
        ] == "Opponent",
        section4b_battle_analysis_df[
            "player_card"
        ],
        "Draw",
    ),
)

section4b_battle_analysis_df[
    "hp_margin"
] = (
    section4b_battle_analysis_df[
        "final_player_hp"
    ]
    - section4b_battle_analysis_df[
        "final_opponent_hp"
    ]
)

section4b_battle_analysis_df[
    "absolute_hp_margin"
] = (
    section4b_battle_analysis_df[
        "hp_margin"
    ].abs()
)

section4b_battle_analysis_df[
    "required_fallback"
] = (
    section4b_battle_analysis_df[
        "fallback_decisions"
    ] > 0
)


# --------------------------------------------------------------------------------------
# 3. Side-outcome summary
# --------------------------------------------------------------------------------------

side_outcome_rows = []

for side_name in [
    "Player",
    "Opponent",
]:

    side_started_df = (
        section4b_battle_analysis_df.loc[
            section4b_battle_analysis_df[
                "starting_side"
            ] == side_name
        ]
    )

    side_wins = int(
        (
            side_started_df[
                "winner"
            ] == side_name
        ).sum()
    )

    side_battles = int(
        len(
            side_started_df
        )
    )

    side_outcome_rows.append(
        {
            "starting_side":
                side_name,

            "battles":
                side_battles,

            "starter_wins":
                side_wins,

            "starter_losses":
                int(
                    side_battles
                    - side_wins
                ),

            "starter_win_rate":
                float(
                    side_wins
                    / side_battles
                    if side_battles > 0
                    else 0.0
                ),

            "average_turns":
                float(
                    side_started_df[
                        "recorded_turns"
                    ].mean()
                ),

            "average_confidence":
                float(
                    side_started_df[
                        "average_confidence"
                    ].mean()
                ),

            "fallback_decisions":
                int(
                    side_started_df[
                        "fallback_decisions"
                    ].sum()
                ),
        }
    )


section4b_side_outcome_df = pd.DataFrame(
    side_outcome_rows
)


print()
print("STARTING-SIDE OUTCOME SUMMARY")
print("-" * 100)

display(
    section4b_side_outcome_df
)


# --------------------------------------------------------------------------------------
# 4. Matchup summary
# --------------------------------------------------------------------------------------

section4b_matchup_summary_df = (
    section4b_battle_analysis_df
    .groupby(
        [
            "player_card",
            "opponent_card",
        ],
        as_index=False,
    )
    .agg(
        battles=(
            "battle_number",
            "count",
        ),

        player_wins=(
            "player_won",
            "sum",
        ),

        opponent_wins=(
            "opponent_won",
            "sum",
        ),

        draws=(
            "draw",
            "sum",
        ),

        starter_wins=(
            "starter_won",
            "sum",
        ),

        average_turns=(
            "recorded_turns",
            "mean",
        ),

        average_confidence=(
            "average_confidence",
            "mean",
        ),

        fallback_decisions=(
            "fallback_decisions",
            "sum",
        ),

        average_absolute_hp_margin=(
            "absolute_hp_margin",
            "mean",
        ),
    )
)

section4b_matchup_summary_df[
    "player_win_rate"
] = (
    section4b_matchup_summary_df[
        "player_wins"
    ]
    / section4b_matchup_summary_df[
        "battles"
    ]
)

section4b_matchup_summary_df[
    "opponent_win_rate"
] = (
    section4b_matchup_summary_df[
        "opponent_wins"
    ]
    / section4b_matchup_summary_df[
        "battles"
    ]
)

section4b_matchup_summary_df[
    "starter_win_rate"
] = (
    section4b_matchup_summary_df[
        "starter_wins"
    ]
    / section4b_matchup_summary_df[
        "battles"
    ]
)


print()
print("MATCHUP SUMMARY")
print("-" * 100)

display(
    section4b_matchup_summary_df
)


# --------------------------------------------------------------------------------------
# 5. Card-level outcome summary
#
# Each card may appear on either the Player or Opponent side.
# --------------------------------------------------------------------------------------

card_result_rows = []

for _, battle_row in (
    section4b_battle_analysis_df.iterrows()
):

    card_result_rows.append(
        {
            "battle_number":
                int(
                    battle_row[
                        "battle_number"
                    ]
                ),

            "scenario_id":
                battle_row[
                    "scenario_id"
                ],

            "card_name":
                battle_row[
                    "player_card"
                ],

            "battle_side":
                "Player",

            "started_battle":
                (
                    battle_row[
                        "starting_side"
                    ]
                    == "Player"
                ),

            "won":
                (
                    battle_row[
                        "winner"
                    ]
                    == "Player"
                ),

            "lost":
                (
                    battle_row[
                        "winner"
                    ]
                    == "Opponent"
                ),

            "draw":
                bool(
                    battle_row[
                        "draw"
                    ]
                ),

            "final_hp":
                float(
                    battle_row[
                        "final_player_hp"
                    ]
                ),
        }
    )

    card_result_rows.append(
        {
            "battle_number":
                int(
                    battle_row[
                        "battle_number"
                    ]
                ),

            "scenario_id":
                battle_row[
                    "scenario_id"
                ],

            "card_name":
                battle_row[
                    "opponent_card"
                ],

            "battle_side":
                "Opponent",

            "started_battle":
                (
                    battle_row[
                        "starting_side"
                    ]
                    == "Opponent"
                ),

            "won":
                (
                    battle_row[
                        "winner"
                    ]
                    == "Opponent"
                ),

            "lost":
                (
                    battle_row[
                        "winner"
                    ]
                    == "Player"
                ),

            "draw":
                bool(
                    battle_row[
                        "draw"
                    ]
                ),

            "final_hp":
                float(
                    battle_row[
                        "final_opponent_hp"
                    ]
                ),
        }
    )


section4b_card_results_df = pd.DataFrame(
    card_result_rows
)

section4b_card_summary_df = (
    section4b_card_results_df
    .groupby(
        "card_name",
        as_index=False,
    )
    .agg(
        battles=(
            "battle_number",
            "count",
        ),

        wins=(
            "won",
            "sum",
        ),

        losses=(
            "lost",
            "sum",
        ),

        draws=(
            "draw",
            "sum",
        ),

        started_battles=(
            "started_battle",
            "sum",
        ),

        average_final_hp=(
            "final_hp",
            "mean",
        ),
    )
)

section4b_card_summary_df[
    "win_rate"
] = (
    section4b_card_summary_df[
        "wins"
    ]
    / section4b_card_summary_df[
        "battles"
    ]
)

section4b_card_summary_df = (
    section4b_card_summary_df
    .sort_values(
        [
            "win_rate",
            "wins",
            "average_final_hp",
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
print("CARD-LEVEL OUTCOME SUMMARY")
print("-" * 100)

display(
    section4b_card_summary_df
)


# --------------------------------------------------------------------------------------
# 6. Policy action and legality analysis
# --------------------------------------------------------------------------------------

section4b_policy_analysis_df = (
    section4a_policy_history_df.copy()
)

section4b_policy_analysis_df[
    "raw_prediction_legal"
] = section4b_policy_analysis_df.apply(
    lambda row: (
        normalize_move_name(
            row[
                "raw_predicted_move"
            ]
        )
        in {
            normalize_move_name(
                move_name
            )
            for move_name in (
                row[
                    "legal_moves"
                ]
                if isinstance(
                    row[
                        "legal_moves"
                    ],
                    list,
                )
                else []
            )
        }
    ),
    axis=1,
)

# CSV or notebook display paths may stringify lists.
# Recompute robustly by comparing selected and raw moves when needed.
section4b_policy_analysis_df[
    "raw_prediction_matches_selected"
] = (
    section4b_policy_analysis_df.apply(
        lambda row: (
            normalize_move_name(
                row[
                    "raw_predicted_move"
                ]
            )
            == normalize_move_name(
                row[
                    "selected_move"
                ]
            )
        ),
        axis=1,
    )
)

section4b_policy_analysis_df[
    "masking_changed_action"
] = (
    ~section4b_policy_analysis_df[
        "raw_prediction_matches_selected"
    ]
)

section4b_policy_analysis_df[
    "selected_probability_gap"
] = (
    section4b_policy_analysis_df[
        "confidence"
    ]
    - section4b_policy_analysis_df[
        "selected_probability"
    ]
)


section4b_action_summary_df = (
    section4b_policy_analysis_df
    .groupby(
        "selected_move",
        as_index=False,
    )
    .agg(
        selections=(
            "selected_move",
            "size",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        average_selected_probability=(
            "selected_probability",
            "mean",
        ),

        fallback_decisions=(
            "fallback_used",
            "sum",
        ),

        masking_changes=(
            "masking_changed_action",
            "sum",
        ),
    )
)

section4b_action_summary_df[
    "selection_fraction"
] = (
    section4b_action_summary_df[
        "selections"
    ]
    / len(
        section4b_policy_analysis_df
    )
)

section4b_action_summary_df[
    "masking_change_rate"
] = (
    section4b_action_summary_df[
        "masking_changes"
    ]
    / section4b_action_summary_df[
        "selections"
    ]
)

section4b_action_summary_df = (
    section4b_action_summary_df
    .sort_values(
        "selections",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("ACTION-SELECTION SUMMARY")
print("-" * 100)

display(
    section4b_action_summary_df
)


# --------------------------------------------------------------------------------------
# 7. Side-level policy-decision analysis
# --------------------------------------------------------------------------------------

section4b_policy_side_summary_df = (
    section4b_policy_analysis_df
    .groupby(
        "current_player",
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

        average_selected_probability=(
            "selected_probability",
            "mean",
        ),

        fallbacks=(
            "fallback_used",
            "sum",
        ),

        masking_changes=(
            "masking_changed_action",
            "sum",
        ),
    )
)

section4b_policy_side_summary_df[
    "fallback_rate"
] = (
    section4b_policy_side_summary_df[
        "fallbacks"
    ]
    / section4b_policy_side_summary_df[
        "decisions"
    ]
)

section4b_policy_side_summary_df[
    "masking_change_rate"
] = (
    section4b_policy_side_summary_df[
        "masking_changes"
    ]
    / section4b_policy_side_summary_df[
        "decisions"
    ]
)


print()
print("POLICY DECISIONS BY ACTING SIDE")
print("-" * 100)

display(
    section4b_policy_side_summary_df
)


# --------------------------------------------------------------------------------------
# 8. Diagnostic findings
# --------------------------------------------------------------------------------------

total_decisions = int(
    len(
        section4b_policy_analysis_df
    )
)

masking_changes = int(
    section4b_policy_analysis_df[
        "masking_changed_action"
    ].sum()
)

masking_change_rate = float(
    masking_changes
    / total_decisions
    if total_decisions > 0
    else 0.0
)

fallback_decisions = int(
    section4b_policy_analysis_df[
        "fallback_used"
    ].sum()
)

fallback_rate = float(
    fallback_decisions
    / total_decisions
    if total_decisions > 0
    else 0.0
)

starter_win_rate = float(
    section4b_battle_analysis_df[
        "starter_won"
    ].mean()
)

player_win_rate = float(
    section4b_battle_analysis_df[
        "player_won"
    ].mean()
)

opponent_win_rate = float(
    section4b_battle_analysis_df[
        "opponent_won"
    ].mean()
)


diagnostic_rows = [
    {
        "finding":
            "starter_win_rate",

        "value":
            starter_win_rate,

        "interpretation":
            (
                "No overall first-move advantage detected."
                if np.isclose(
                    starter_win_rate,
                    0.5,
                )
                else (
                    "Starting side shows an advantage."
                    if starter_win_rate > 0.5
                    else "Starting side shows a disadvantage."
                )
            ),
    },
    {
        "finding":
            "player_slot_win_rate",

        "value":
            player_win_rate,

        "interpretation":
            "Outcome by fixed Player slot, not independent agent strength.",
    },
    {
        "finding":
            "opponent_slot_win_rate",

        "value":
            opponent_win_rate,

        "interpretation":
            "Outcome by fixed Opponent slot, not independent agent strength.",
    },
    {
        "finding":
            "legal_masking_change_rate",

        "value":
            masking_change_rate,

        "interpretation":
            (
                "Fraction of decisions where the raw classifier prediction "
                "differed from the final legal move."
            ),
    },
    {
        "finding":
            "fallback_rate",

        "value":
            fallback_rate,

        "interpretation":
            "Fraction of decisions requiring explicit fallback behavior.",
    },
]


section4b_diagnostics_df = pd.DataFrame(
    diagnostic_rows
)


print()
print("TOURNAMENT DIAGNOSTICS")
print("-" * 100)

display(
    section4b_diagnostics_df
)


# --------------------------------------------------------------------------------------
# 9. Validation assertions
# --------------------------------------------------------------------------------------

assert len(
    section4b_battle_analysis_df
) == len(
    section4a_battle_results_df
)

assert int(
    section4b_side_outcome_df[
        "battles"
    ].sum()
) == len(
    section4a_battle_results_df
)

assert int(
    section4b_card_summary_df[
        "battles"
    ].sum()
) == (
    2
    * len(
        section4a_battle_results_df
    )
)

assert total_decisions == int(
    section4a_summary[
        "total_policy_decisions"
    ]
)

assert fallback_decisions == int(
    section4a_summary[
        "total_fallback_decisions"
    ]
)

assert 0.0 <= masking_change_rate <= 1.0

assert 0.0 <= fallback_rate <= 1.0

assert np.isclose(
    player_win_rate
    + opponent_win_rate
    + float(
        section4b_battle_analysis_df[
            "draw"
        ].mean()
    ),
    1.0,
)

assert np.isclose(
    starter_win_rate,
    float(
        section4a_summary[
            "starter_win_rate"
        ]
    ),
)


# --------------------------------------------------------------------------------------
# 10. Save Section 4B reports
# --------------------------------------------------------------------------------------

SECTION4B_BATTLE_ANALYSIS_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_battle_analysis.csv"
)

SECTION4B_SIDE_OUTCOME_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_starting_side_summary.csv"
)

SECTION4B_MATCHUP_SUMMARY_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_matchup_summary.csv"
)

SECTION4B_CARD_SUMMARY_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_card_summary.csv"
)

SECTION4B_ACTION_SUMMARY_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_action_summary.csv"
)

SECTION4B_POLICY_SIDE_SUMMARY_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_policy_side_summary.csv"
)

SECTION4B_DIAGNOSTICS_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_diagnostics.csv"
)

SECTION4B_SUMMARY_FILE = (
    SECTION4B_ANALYSIS_DIR
    / "section4b_summary.json"
)


section4b_battle_analysis_df.to_csv(
    SECTION4B_BATTLE_ANALYSIS_FILE,
    index=False,
)

section4b_side_outcome_df.to_csv(
    SECTION4B_SIDE_OUTCOME_FILE,
    index=False,
)

section4b_matchup_summary_df.to_csv(
    SECTION4B_MATCHUP_SUMMARY_FILE,
    index=False,
)

section4b_card_summary_df.to_csv(
    SECTION4B_CARD_SUMMARY_FILE,
    index=False,
)

section4b_action_summary_df.to_csv(
    SECTION4B_ACTION_SUMMARY_FILE,
    index=False,
)

section4b_policy_side_summary_df.to_csv(
    SECTION4B_POLICY_SIDE_SUMMARY_FILE,
    index=False,
)

section4b_diagnostics_df.to_csv(
    SECTION4B_DIAGNOSTICS_FILE,
    index=False,
)


section4b_summary = {
    "status":
        "SINGLE_PASS_TOURNAMENT_ANALYZED",

    "battle_count":
        int(
            len(
                section4b_battle_analysis_df
            )
        ),

    "total_decisions":
        total_decisions,

    "player_slot_wins":
        int(
            section4b_battle_analysis_df[
                "player_won"
            ].sum()
        ),

    "opponent_slot_wins":
        int(
            section4b_battle_analysis_df[
                "opponent_won"
            ].sum()
        ),

    "draws":
        int(
            section4b_battle_analysis_df[
                "draw"
            ].sum()
        ),

    "starter_win_rate":
        starter_win_rate,

    "masking_changes":
        masking_changes,

    "masking_change_rate":
        masking_change_rate,

    "fallback_decisions":
        fallback_decisions,

    "fallback_rate":
        fallback_rate,

    "highest_win_rate_card":
        (
            str(
                section4b_card_summary_df.iloc[0][
                    "card_name"
                ]
            )
            if not section4b_card_summary_df.empty
            else None
        ),

    "highest_win_rate":
        (
            float(
                section4b_card_summary_df.iloc[0][
                    "win_rate"
                ]
            )
            if not section4b_card_summary_df.empty
            else None
        ),

    "next_stage":
        "REPEATED_TOURNAMENT_EVALUATION",
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


# --------------------------------------------------------------------------------------
# 11. Final summary
# --------------------------------------------------------------------------------------

print()
print("SECTION 4B SUMMARY")
print("-" * 100)

for key, value in (
    section4b_summary.items()
):

    print(
        f"{key:32}: {value}"
    )


print()
print("SAVED SECTION 4B REPORTS")
print("-" * 100)

print(SECTION4B_BATTLE_ANALYSIS_FILE)
print(SECTION4B_SIDE_OUTCOME_FILE)
print(SECTION4B_MATCHUP_SUMMARY_FILE)
print(SECTION4B_CARD_SUMMARY_FILE)
print(SECTION4B_ACTION_SUMMARY_FILE)
print(SECTION4B_POLICY_SIDE_SUMMARY_FILE)
print(SECTION4B_DIAGNOSTICS_FILE)
print(SECTION4B_SUMMARY_FILE)

print()
print(
    "✅ SECTION 4B SINGLE-PASS TOURNAMENT RESULT ANALYSIS PASSED"
)


# # Section 5 — Expanded Tournament Evaluation
# 
# ## Section 5A — Controlled Scenario Expansion
# 
# #### The policy and simulator are deterministic, so repeating identical states would reproduce identical results.
# 
# #### This section creates a meaningful expanded tournament by applying controlled perturbations to every validated base scenario:
# 
# - baseline conditions,
# - HP-pressure conditions,
# - Energy-pressure conditions.
# 
# #### The expanded matrix preserves valid simulator states while testing whether the policy remains stable outside the exact baseline configuration.

# In[10]:


# ======================================================================================
# SECTION 5A — CONTROLLED SCENARIO EXPANSION
# ======================================================================================

print("=" * 100)
print("SECTION 5A — CONTROLLED SCENARIO EXPANSION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Expansion configuration
# --------------------------------------------------------------------------------------

SECTION5_REPORT_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section5"
)

SECTION5_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TOURNAMENT_CONDITION_SPECS = [
    {
        "condition_id": "BASELINE",
        "condition_name": "Baseline",
        "player_hp_factor": 1.00,
        "opponent_hp_factor": 1.00,
        "player_energy_delta": 0,
        "opponent_energy_delta": 0,
        "turn_delta": 0,
    },
    {
        "condition_id": "HP_PRESSURE",
        "condition_name": "Player HP Pressure",
        "player_hp_factor": 0.50,
        "opponent_hp_factor": 1.00,
        "player_energy_delta": 0,
        "opponent_energy_delta": 0,
        "turn_delta": 4,
    },
    {
        "condition_id": "ENERGY_PRESSURE",
        "condition_name": "Opponent Energy Pressure",
        "player_hp_factor": 1.00,
        "opponent_hp_factor": 1.00,
        "player_energy_delta": 0,
        "opponent_energy_delta": -2,
        "turn_delta": 2,
    },
]


# --------------------------------------------------------------------------------------
# 2. Expand one base scenario
# --------------------------------------------------------------------------------------

def expand_tournament_scenario(
    base_spec: Mapping[str, Any],
    condition_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Apply one controlled tournament condition to one base scenario.
    """

    expanded_spec = deepcopy(
        dict(
            base_spec
        )
    )

    condition_id = str(
        condition_spec[
            "condition_id"
        ]
    )

    base_scenario_id = str(
        base_spec[
            "scenario_id"
        ]
    )

    base_scenario_name = str(
        base_spec[
            "scenario_name"
        ]
    )

    player_card_name = str(
        base_spec[
            "player_card"
        ]
    )

    opponent_card_name = str(
        base_spec[
            "opponent_card"
        ]
    )

    player_max_hp = float(
        TOURNAMENT_CARD_LIBRARY[
            player_card_name
        ][
            "HP"
        ]
    )

    opponent_max_hp = float(
        TOURNAMENT_CARD_LIBRARY[
            opponent_card_name
        ][
            "HP"
        ]
    )

    requested_player_hp = (
        float(
            base_spec[
                "player_hp"
            ]
        )
        * float(
            condition_spec[
                "player_hp_factor"
            ]
        )
    )

    requested_opponent_hp = (
        float(
            base_spec[
                "opponent_hp"
            ]
        )
        * float(
            condition_spec[
                "opponent_hp_factor"
            ]
        )
    )

    expanded_player_hp = min(
        player_max_hp,
        max(
            10.0,
            requested_player_hp,
        ),
    )

    expanded_opponent_hp = min(
        opponent_max_hp,
        max(
            10.0,
            requested_opponent_hp,
        ),
    )

    expanded_player_energy = max(
        0,
        int(
            base_spec[
                "player_energy"
            ]
        )
        + int(
            condition_spec[
                "player_energy_delta"
            ]
        ),
    )

    expanded_opponent_energy = max(
        0,
        int(
            base_spec[
                "opponent_energy"
            ]
        )
        + int(
            condition_spec[
                "opponent_energy_delta"
            ]
        ),
    )

    expanded_spec.update(
        {
            "scenario_id":
                (
                    f"{base_scenario_id}"
                    f"__{condition_id}"
                ),

            "scenario_name":
                (
                    f"{base_scenario_name}"
                    f"__{condition_id}"
                ),

            "base_scenario_id":
                base_scenario_id,

            "base_scenario_name":
                base_scenario_name,

            "condition_id":
                condition_id,

            "condition_name":
                str(
                    condition_spec[
                        "condition_name"
                    ]
                ),

            "player_hp":
                float(
                    expanded_player_hp
                ),

            "opponent_hp":
                float(
                    expanded_opponent_hp
                ),

            "player_energy":
                int(
                    expanded_player_energy
                ),

            "opponent_energy":
                int(
                    expanded_opponent_energy
                ),

            "turn_number":
                int(
                    base_spec[
                        "turn_number"
                    ]
                )
                + int(
                    condition_spec[
                        "turn_delta"
                    ]
                ),
        }
    )

    return expanded_spec


# --------------------------------------------------------------------------------------
# 3. Build expanded specification matrix
# --------------------------------------------------------------------------------------

expanded_tournament_specs = []

for base_spec in TOURNAMENT_SCENARIO_SPECS:

    for condition_spec in (
        TOURNAMENT_CONDITION_SPECS
    ):

        expanded_tournament_specs.append(
            expand_tournament_scenario(
                base_spec=base_spec,
                condition_spec=condition_spec,
            )
        )


expanded_tournament_spec_df = pd.DataFrame(
    expanded_tournament_specs
)


print()
print("EXPANDED TOURNAMENT SPECIFICATIONS")
print("-" * 100)

display(
    expanded_tournament_spec_df
)


# --------------------------------------------------------------------------------------
# 4. Construct production states
# --------------------------------------------------------------------------------------

expanded_tournament_states = {}

expanded_state_rows = []
expanded_state_failures = []

for expanded_spec in expanded_tournament_specs:

    expanded_scenario_id = str(
        expanded_spec[
            "scenario_id"
        ]
    )

    try:

        expanded_state = build_tournament_state(
            expanded_spec
        )

        expanded_tournament_states[
            expanded_scenario_id
        ] = expanded_state

        legal_moves = (
            tournament_generate_legal_moves(
                expanded_state
            )
        )

        policy_features = (
            battle_state_to_policy_features(
                battle_state=expanded_state,
                legal_moves=legal_moves,
                side_mode="PRESERVE",
            )
        )

        selection = (
            select_legal_policy_move(
                battle_state=expanded_state,
                legal_moves=legal_moves,
                side_mode="PRESERVE",
            )
        )

        expanded_state_rows.append(
            {
                "scenario_id":
                    expanded_scenario_id,

                "base_scenario_id":
                    expanded_spec[
                        "base_scenario_id"
                    ],

                "condition_id":
                    expanded_spec[
                        "condition_id"
                    ],

                "scenario_name":
                    expanded_spec[
                        "scenario_name"
                    ],

                "player_card":
                    expanded_spec[
                        "player_card"
                    ],

                "opponent_card":
                    expanded_spec[
                        "opponent_card"
                    ],

                "current_player":
                    expanded_state.current_player,

                "turn_number":
                    int(
                        expanded_state.turn_number
                    ),

                "player_hp":
                    float(
                        expanded_state
                        .player
                        .active
                        .current_hp
                    ),

                "opponent_hp":
                    float(
                        expanded_state
                        .opponent
                        .active
                        .current_hp
                    ),

                "player_energy":
                    int(
                        expanded_state
                        .player
                        .active
                        .attached_energy
                    ),

                "opponent_energy":
                    int(
                        expanded_state
                        .opponent
                        .active
                        .attached_energy
                    ),

                "legal_move_count":
                    int(
                        len(
                            legal_moves
                        )
                    ),

                "legal_moves":
                    [
                        move_display_name(
                            move
                        )
                        for move in legal_moves
                    ],

                "raw_predicted_move":
                    selection[
                        "raw_predicted_move"
                    ],

                "selected_move":
                    selection[
                        "selected_move_name"
                    ],

                "confidence":
                    float(
                        selection[
                            "raw_confidence"
                        ]
                    ),

                "fallback_used":
                    bool(
                        selection[
                            "fallback_used"
                        ]
                    ),

                "feature_count":
                    int(
                        len(
                            policy_features.columns
                        )
                    ),

                "state_valid":
                    True,

                "error_type":
                    None,

                "error_message":
                    None,
            }
        )

    except Exception as error:

        expanded_state_failures.append(
            {
                "scenario_id":
                    expanded_scenario_id,

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


expanded_tournament_validation_df = pd.DataFrame(
    expanded_state_rows
)

expanded_tournament_failures_df = pd.DataFrame(
    expanded_state_failures,
    columns=[
        "scenario_id",
        "error_type",
        "error_message",
    ],
)


# --------------------------------------------------------------------------------------
# 5. Display validation profile
# --------------------------------------------------------------------------------------

print()
print("EXPANDED STATE VALIDATION")
print("-" * 100)

display(
    expanded_tournament_validation_df
)


print()
print("EXPANSION FAILURES")
print("-" * 100)

if expanded_tournament_failures_df.empty:

    print(
        "No expanded scenario construction failures."
    )

else:

    display(
        expanded_tournament_failures_df
    )


condition_summary_df = (
    expanded_tournament_validation_df
    .groupby(
        "condition_id",
        as_index=False,
    )
    .agg(
        scenarios=(
            "scenario_id",
            "count",
        ),

        player_start_scenarios=(
            "current_player",
            lambda values: int(
                (
                    values
                    == "Player"
                ).sum()
            ),
        ),

        opponent_start_scenarios=(
            "current_player",
            lambda values: int(
                (
                    values
                    == "Opponent"
                ).sum()
            ),
        ),

        average_player_hp=(
            "player_hp",
            "mean",
        ),

        average_opponent_hp=(
            "opponent_hp",
            "mean",
        ),

        average_player_energy=(
            "player_energy",
            "mean",
        ),

        average_opponent_energy=(
            "opponent_energy",
            "mean",
        ),

        average_confidence=(
            "confidence",
            "mean",
        ),

        fallback_scenarios=(
            "fallback_used",
            "sum",
        ),
    )
)


print()
print("CONDITION SUMMARY")
print("-" * 100)

display(
    condition_summary_df
)


# --------------------------------------------------------------------------------------
# 6. Validation
# --------------------------------------------------------------------------------------

expected_expanded_count = (
    len(
        TOURNAMENT_SCENARIO_SPECS
    )
    * len(
        TOURNAMENT_CONDITION_SPECS
    )
)

assert len(
    expanded_tournament_specs
) == expected_expanded_count

assert len(
    expanded_tournament_states
) == expected_expanded_count

assert len(
    expanded_tournament_validation_df
) == expected_expanded_count

assert expanded_tournament_spec_df[
    "scenario_id"
].is_unique

assert expanded_tournament_validation_df[
    "scenario_id"
].is_unique

assert expanded_tournament_failures_df.empty

assert expanded_tournament_validation_df[
    "state_valid"
].all()

assert expanded_tournament_validation_df[
    "legal_move_count"
].gt(0).all()

assert expanded_tournament_validation_df[
    "feature_count"
].eq(
        len(
            POLICY_RAW_FEATURE_COLUMNS
        )
    ).all()

assert expanded_tournament_validation_df[
    "confidence"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert set(
    expanded_tournament_validation_df[
        "condition_id"
    ]
) == {
    "BASELINE",
    "HP_PRESSURE",
    "ENERGY_PRESSURE",
}


# --------------------------------------------------------------------------------------
# 7. Save reports
# --------------------------------------------------------------------------------------

SECTION5A_EXPANDED_SPEC_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_expanded_scenario_specs.csv"
)

SECTION5A_VALIDATION_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_expanded_scenario_validation.csv"
)

SECTION5A_CONDITION_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_condition_summary.csv"
)

SECTION5A_FAILURES_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_expansion_failures.csv"
)

SECTION5A_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5a_summary.json"
)


expanded_tournament_spec_df.to_csv(
    SECTION5A_EXPANDED_SPEC_FILE,
    index=False,
)

expanded_tournament_validation_df.to_csv(
    SECTION5A_VALIDATION_FILE,
    index=False,
)

condition_summary_df.to_csv(
    SECTION5A_CONDITION_SUMMARY_FILE,
    index=False,
)

expanded_tournament_failures_df.to_csv(
    SECTION5A_FAILURES_FILE,
    index=False,
)


section5a_summary = {
    "status":
        "EXPANDED_TOURNAMENT_MATRIX_READY",

    "base_scenario_count":
        int(
            len(
                TOURNAMENT_SCENARIO_SPECS
            )
        ),

    "condition_count":
        int(
            len(
                TOURNAMENT_CONDITION_SPECS
            )
        ),

    "expanded_scenario_count":
        int(
            len(
                expanded_tournament_validation_df
            )
        ),

    "expanded_states_valid":
        int(
            expanded_tournament_validation_df[
                "state_valid"
            ].sum()
        ),

    "expanded_states_failed":
        int(
            len(
                expanded_tournament_failures_df
            )
        ),

    "fallback_scenarios":
        int(
            expanded_tournament_validation_df[
                "fallback_used"
            ].sum()
        ),

    "average_confidence":
        float(
            expanded_tournament_validation_df[
                "confidence"
            ].mean()
        ),

    "next_stage":
        "EXPANDED_TOURNAMENT_EXECUTION",
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


# --------------------------------------------------------------------------------------
# 8. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 5A SUMMARY")
print("-" * 100)

for key, value in (
    section5a_summary.items()
):

    print(
        f"{key:34}: {value}"
    )


print()
print("SAVED SECTION 5A REPORTS")
print("-" * 100)

print(SECTION5A_EXPANDED_SPEC_FILE)
print(SECTION5A_VALIDATION_FILE)
print(SECTION5A_CONDITION_SUMMARY_FILE)
print(SECTION5A_FAILURES_FILE)
print(SECTION5A_SUMMARY_FILE)

print()
print(
    "✅ SECTION 5A CONTROLLED SCENARIO EXPANSION PASSED"
)


# ## Section 5B — Expanded Tournament Execution
# 
# #### This section runs one complete production-simulator battle for every state in the expanded tournament matrix.
# 
# #### The evaluation measures how the policy behaves under:
# 
# - baseline conditions,
# - Player HP pressure,
# - Opponent Energy pressure.
# 
# #### For each battle, it records:
# 
# - winner,
# - battle length,
# - final HP,
# - policy decisions,
# - fallback decisions,
# - masking dependence,
# - confidence,
# - condition-specific results,
# - turn-level actions,
# - policy history,
# - execution failures.

# In[12]:


# ======================================================================================
# SECTION 5B — EXPANDED TOURNAMENT EXECUTION
# ======================================================================================

print("=" * 100)
print("SECTION 5B — EXPANDED TOURNAMENT EXECUTION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Execution configuration
# --------------------------------------------------------------------------------------

SECTION5B_SEARCH_DEPTH = 0
SECTION5B_MAX_TURNS = 30
SECTION5B_VERBOSE = False

SECTION5B_TRANSCRIPT_DIR = (
    SECTION5_REPORT_DIR
    / "expanded_transcripts"
)

SECTION5B_TRANSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("EXPANDED TOURNAMENT CONFIGURATION")
print("-" * 100)

print(
    f"Expanded scenarios     : "
    f"{len(expanded_tournament_specs)}"
)

print(
    f"Search depth           : "
    f"{SECTION5B_SEARCH_DEPTH}"
)

print(
    f"Maximum turns          : "
    f"{SECTION5B_MAX_TURNS}"
)

print(
    f"Verbose battles        : "
    f"{SECTION5B_VERBOSE}"
)


# --------------------------------------------------------------------------------------
# 2. Execute all expanded battles
# --------------------------------------------------------------------------------------

section5b_battle_rows = []
section5b_turn_rows = []
section5b_policy_rows = []
section5b_failure_rows = []

for battle_number, expanded_spec in enumerate(
    expanded_tournament_specs,
    start=1,
):

    scenario_id = str(
        expanded_spec[
            "scenario_id"
        ]
    )

    scenario_name = str(
        expanded_spec[
            "scenario_name"
        ]
    )

    base_scenario_id = str(
        expanded_spec[
            "base_scenario_id"
        ]
    )

    condition_id = str(
        expanded_spec[
            "condition_id"
        ]
    )

    condition_name = str(
        expanded_spec[
            "condition_name"
        ]
    )

    print()
    print(
        f"Running {battle_number:02d}/"
        f"{len(expanded_tournament_specs):02d} — "
        f"{scenario_id}"
    )

    initial_state = deepcopy(
        expanded_tournament_states[
            scenario_id
        ]
    )

    expanded_agent = TournamentPolicyAgent(
        name=(
            f"ExpandedTournamentPolicy_"
            f"{scenario_id}"
        ),
        side_mode="PRESERVE",
    )

    try:

        battle_result = simulator_run_battle(
            initial_state=initial_state,
            agent=expanded_agent,
            search_depth=SECTION5B_SEARCH_DEPTH,
            max_turns=SECTION5B_MAX_TURNS,
            verbose=SECTION5B_VERBOSE,
        )

        assert isinstance(
            battle_result,
            SimulatorBattleSimulationResult,
        )

        final_state = read_first(
            battle_result,
            [
                "final_state",
                "state",
            ],
            default=None,
        )

        assert final_state is not None

        turn_records = list(
            read_first(
                battle_result,
                [
                    "turns",
                    "turn_records",
                    "history",
                ],
                default=[],
            )
            or []
        )

        winner = read_first(
            battle_result,
            [
                "winner",
                "winning_side",
            ],
            default=None,
        )

        if winner is None:

            winner = simulator_determine_winner(
                final_state
            )

        termination_reason = read_first(
            battle_result,
            [
                "stop_reason",
                "termination_reason",
                "reason",
                "status",
            ],
            default=None,
        )

        transcript = simulator_create_transcript(
            battle_result
        )

        expanded_policy_history_df = (
            expanded_agent.history_dataframe()
        )

        expanded_statistics = (
            expanded_agent.statistics()
        )

        policy_decisions = int(
            expanded_statistics[
                "total_decisions"
            ]
        )

        fallback_decisions = int(
            expanded_statistics[
                "fallback_decisions"
            ]
        )

        fallback_rate = float(
            expanded_statistics[
                "fallback_rate"
            ]
        )

        average_confidence = (
            float(
                expanded_policy_history_df[
                    "confidence"
                ].mean()
            )
            if not expanded_policy_history_df.empty
            else 0.0
        )

        minimum_confidence = (
            float(
                expanded_policy_history_df[
                    "confidence"
                ].min()
            )
            if not expanded_policy_history_df.empty
            else 0.0
        )

        maximum_confidence = (
            float(
                expanded_policy_history_df[
                    "confidence"
                ].max()
            )
            if not expanded_policy_history_df.empty
            else 0.0
        )

        masking_changes = (
            int(
                (
                    expanded_policy_history_df[
                        "selected_move"
                    ].map(
                        normalize_move_name
                    )
                    != expanded_policy_history_df[
                        "raw_predicted_move"
                    ].map(
                        normalize_move_name
                    )
                ).sum()
            )
            if not expanded_policy_history_df.empty
            else 0
        )

        masking_change_rate = (
            float(
                masking_changes
                / policy_decisions
            )
            if policy_decisions > 0
            else 0.0
        )

        recorded_turns = int(
            len(
                turn_records
            )
        )

        final_player_hp = float(
            final_state
            .player
            .active
            .current_hp
        )

        final_opponent_hp = float(
            final_state
            .opponent
            .active
            .current_hp
        )

        starting_side = str(
            expanded_spec[
                "current_player"
            ]
        )

        starter_won = (
            winner == starting_side
        )

        player_won = (
            winner == "Player"
        )

        opponent_won = (
            winner == "Opponent"
        )

        draw = (
            winner == "Draw"
            or winner is None
        )

        transcript_file = (
            SECTION5B_TRANSCRIPT_DIR
            / f"{scenario_id}_transcript.txt"
        )

        transcript_file.write_text(
            transcript,
            encoding="utf-8",
        )


        # --------------------------------------------------------------------------
        # Battle summary
        # --------------------------------------------------------------------------

        section5b_battle_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "base_scenario_id":
                    base_scenario_id,

                "condition_id":
                    condition_id,

                "condition_name":
                    condition_name,

                "player_card":
                    str(
                        expanded_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        expanded_spec[
                            "opponent_card"
                        ]
                    ),

                "starting_side":
                    starting_side,

                "starting_turn_number":
                    int(
                        expanded_spec[
                            "turn_number"
                        ]
                    ),

                "starting_player_hp":
                    float(
                        expanded_spec[
                            "player_hp"
                        ]
                    ),

                "starting_opponent_hp":
                    float(
                        expanded_spec[
                            "opponent_hp"
                        ]
                    ),

                "starting_player_energy":
                    int(
                        expanded_spec[
                            "player_energy"
                        ]
                    ),

                "starting_opponent_energy":
                    int(
                        expanded_spec[
                            "opponent_energy"
                        ]
                    ),

                "winner":
                    winner,

                "starter_won":
                    bool(
                        starter_won
                    ),

                "player_won":
                    bool(
                        player_won
                    ),

                "opponent_won":
                    bool(
                        opponent_won
                    ),

                "draw":
                    bool(
                        draw
                    ),

                "recorded_turns":
                    recorded_turns,

                "final_turn_number":
                    int(
                        final_state.turn_number
                    ),

                "final_player_hp":
                    final_player_hp,

                "final_opponent_hp":
                    final_opponent_hp,

                "policy_decisions":
                    policy_decisions,

                "fallback_decisions":
                    fallback_decisions,

                "fallback_rate":
                    fallback_rate,

                "masking_changes":
                    masking_changes,

                "masking_change_rate":
                    masking_change_rate,

                "average_confidence":
                    average_confidence,

                "minimum_confidence":
                    minimum_confidence,

                "maximum_confidence":
                    maximum_confidence,

                "termination_reason":
                    termination_reason,

                "transcript_file":
                    str(
                        transcript_file
                    ),

                "battle_completed":
                    True,

                "error_type":
                    None,

                "error_message":
                    None,
            }
        )


        # --------------------------------------------------------------------------
        # Turn-level records
        # --------------------------------------------------------------------------

        for turn_index, turn_record in enumerate(
            turn_records,
            start=1,
        ):

            record_dict = turn_record_to_dictionary(
                turn_record
            )

            section5b_turn_rows.append(
                {
                    "battle_number":
                        battle_number,

                    "scenario_id":
                        scenario_id,

                    "base_scenario_id":
                        base_scenario_id,

                    "condition_id":
                        condition_id,

                    "record_number":
                        turn_index,

                    "turn_number":
                        read_first(
                            record_dict,
                            [
                                "turn_number",
                                "turn",
                            ],
                            default=turn_index,
                        ),

                    "acting_side":
                        read_first(
                            record_dict,
                            [
                                "acting_side",
                                "current_player",
                                "side",
                            ],
                            default=None,
                        ),

                    "pokemon_name":
                        read_first(
                            record_dict,
                            [
                                "pokemon_name",
                                "active_pokemon",
                            ],
                            default=None,
                        ),

                    "move_name":
                        read_first(
                            record_dict,
                            [
                                "move_name",
                                "action_name",
                            ],
                            default=None,
                        ),

                    "damage":
                        read_first(
                            record_dict,
                            [
                                "damage",
                            ],
                            default=None,
                        ),

                    "score":
                        read_first(
                            record_dict,
                            [
                                "score",
                                "decision_score",
                            ],
                            default=None,
                        ),

                    "player_hp_after":
                        read_first(
                            record_dict,
                            [
                                "player_hp_after",
                                "player_hp",
                            ],
                            default=None,
                        ),

                    "opponent_hp_after":
                        read_first(
                            record_dict,
                            [
                                "opponent_hp_after",
                                "opponent_hp",
                            ],
                            default=None,
                        ),

                    "next_side":
                        read_first(
                            record_dict,
                            [
                                "next_side",
                                "next_player",
                            ],
                            default=None,
                        ),
                }
            )


        # --------------------------------------------------------------------------
        # Policy decision history
        # --------------------------------------------------------------------------

        if not expanded_policy_history_df.empty:

            expanded_policy_history_copy = (
                expanded_policy_history_df.copy()
            )

            expanded_policy_history_copy.insert(
                0,
                "condition_id",
                condition_id,
            )

            expanded_policy_history_copy.insert(
                0,
                "base_scenario_id",
                base_scenario_id,
            )

            expanded_policy_history_copy.insert(
                0,
                "scenario_id",
                scenario_id,
            )

            expanded_policy_history_copy.insert(
                0,
                "battle_number",
                battle_number,
            )

            expanded_policy_history_copy[
                "masking_changed_action"
            ] = (
                expanded_policy_history_copy[
                    "selected_move"
                ].map(
                    normalize_move_name
                )
                != expanded_policy_history_copy[
                    "raw_predicted_move"
                ].map(
                    normalize_move_name
                )
            )

            section5b_policy_rows.extend(
                expanded_policy_history_copy
                .to_dict(
                    orient="records"
                )
            )


        print(
            f"  Winner={winner} | "
            f"Turns={recorded_turns} | "
            f"Masking={masking_changes} | "
            f"Fallbacks={fallback_decisions} | "
            f"Confidence={average_confidence:.4f}"
        )

    except Exception as error:

        section5b_failure_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "base_scenario_id":
                    base_scenario_id,

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

        section5b_battle_rows.append(
            {
                "battle_number":
                    battle_number,

                "scenario_id":
                    scenario_id,

                "scenario_name":
                    scenario_name,

                "base_scenario_id":
                    base_scenario_id,

                "condition_id":
                    condition_id,

                "condition_name":
                    condition_name,

                "player_card":
                    str(
                        expanded_spec[
                            "player_card"
                        ]
                    ),

                "opponent_card":
                    str(
                        expanded_spec[
                            "opponent_card"
                        ]
                    ),

                "starting_side":
                    str(
                        expanded_spec[
                            "current_player"
                        ]
                    ),

                "winner":
                    None,

                "recorded_turns":
                    0,

                "policy_decisions":
                    0,

                "fallback_decisions":
                    0,

                "masking_changes":
                    0,

                "battle_completed":
                    False,

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
            f"  FAILED — "
            f"{type(error).__name__}: {error}"
        )


# --------------------------------------------------------------------------------------
# 3. Build result dataframes
# --------------------------------------------------------------------------------------

section5b_battle_results_df = pd.DataFrame(
    section5b_battle_rows
)

section5b_turn_results_df = pd.DataFrame(
    section5b_turn_rows
)

section5b_policy_history_df = pd.DataFrame(
    section5b_policy_rows
)

section5b_failures_df = pd.DataFrame(
    section5b_failure_rows,
    columns=[
        "battle_number",
        "scenario_id",
        "base_scenario_id",
        "condition_id",
        "error_type",
        "error_message",
    ],
)


# --------------------------------------------------------------------------------------
# 4. Display execution results
# --------------------------------------------------------------------------------------

print()
print("EXPANDED TOURNAMENT RESULTS")
print("-" * 100)

expanded_preview_columns = [
    "battle_number",
    "scenario_id",
    "condition_id",
    "starting_side",
    "winner",
    "starter_won",
    "recorded_turns",
    "final_player_hp",
    "final_opponent_hp",
    "policy_decisions",
    "masking_changes",
    "masking_change_rate",
    "fallback_decisions",
    "fallback_rate",
    "average_confidence",
    "battle_completed",
]

display(
    section5b_battle_results_df[
        expanded_preview_columns
    ]
)


print()
print("EXPANDED TOURNAMENT FAILURES")
print("-" * 100)

if section5b_failures_df.empty:

    print(
        "No expanded tournament failures."
    )

else:

    display(
        section5b_failures_df
    )


print()
print("EXPANDED TURN SAMPLE")
print("-" * 100)

display(
    section5b_turn_results_df.head(
        30
    )
)


print()
print("EXPANDED POLICY HISTORY SAMPLE")
print("-" * 100)

display(
    section5b_policy_history_df.head(
        30
    )
)


# --------------------------------------------------------------------------------------
# 5. Condition-level summary
# --------------------------------------------------------------------------------------

section5b_condition_summary_df = (
    section5b_battle_results_df
    .groupby(
        "condition_id",
        as_index=False,
    )
    .agg(
        battles=(
            "battle_number",
            "count",
        ),

        player_wins=(
            "player_won",
            "sum",
        ),

        opponent_wins=(
            "opponent_won",
            "sum",
        ),

        draws=(
            "draw",
            "sum",
        ),

        starter_wins=(
            "starter_won",
            "sum",
        ),

        average_turns=(
            "recorded_turns",
            "mean",
        ),

        total_decisions=(
            "policy_decisions",
            "sum",
        ),

        total_masking_changes=(
            "masking_changes",
            "sum",
        ),

        total_fallbacks=(
            "fallback_decisions",
            "sum",
        ),

        average_confidence=(
            "average_confidence",
            "mean",
        ),

        average_final_player_hp=(
            "final_player_hp",
            "mean",
        ),

        average_final_opponent_hp=(
            "final_opponent_hp",
            "mean",
        ),
    )
)

section5b_condition_summary_df[
    "player_win_rate"
] = (
    section5b_condition_summary_df[
        "player_wins"
    ]
    / section5b_condition_summary_df[
        "battles"
    ]
)

section5b_condition_summary_df[
    "opponent_win_rate"
] = (
    section5b_condition_summary_df[
        "opponent_wins"
    ]
    / section5b_condition_summary_df[
        "battles"
    ]
)

section5b_condition_summary_df[
    "starter_win_rate"
] = (
    section5b_condition_summary_df[
        "starter_wins"
    ]
    / section5b_condition_summary_df[
        "battles"
    ]
)

section5b_condition_summary_df[
    "masking_change_rate"
] = (
    section5b_condition_summary_df[
        "total_masking_changes"
    ]
    / section5b_condition_summary_df[
        "total_decisions"
    ]
)

section5b_condition_summary_df[
    "fallback_rate"
] = (
    section5b_condition_summary_df[
        "total_fallbacks"
    ]
    / section5b_condition_summary_df[
        "total_decisions"
    ]
)


print()
print("CONDITION-LEVEL TOURNAMENT SUMMARY")
print("-" * 100)

display(
    section5b_condition_summary_df
)

# --------------------------------------------------------------------------------------
# Normalize completed turn-limit games into Draws
# --------------------------------------------------------------------------------------

missing_winner_mask = (
    section5b_battle_results_df["battle_completed"].astype(bool)
    &
    section5b_battle_results_df["winner"].isna()
)

section5b_battle_results_df.loc[
    missing_winner_mask,
    "winner"
] = "Draw"

section5b_battle_results_df.loc[
    missing_winner_mask,
    "draw"
] = True

section5b_battle_results_df.loc[
    missing_winner_mask,
    "player_won"
] = False

section5b_battle_results_df.loc[
    missing_winner_mask,
    "opponent_won"
] = False

section5b_battle_results_df.loc[
    missing_winner_mask,
    "starter_won"
] = False

print()
print("TURN-LIMIT OUTCOME NORMALIZATION")
print("-" * 100)

print(
    f"Completed battles normalized as draws: "
    f"{int(missing_winner_mask.sum())}"
)

if missing_winner_mask.any():

    display(
        section5b_battle_results_df.loc[
            missing_winner_mask,
            [
                "scenario_id",
                "condition_id",
                "winner",
                "recorded_turns",
                "termination_reason",
            ],
        ]
    )


# --------------------------------------------------------------------------------------
# 6. Validation
# --------------------------------------------------------------------------------------

assert len(
    section5b_battle_results_df
) == len(
    expanded_tournament_specs
)

assert section5b_battle_results_df[
    "scenario_id"
].is_unique

assert section5b_failures_df.empty, (
    "One or more expanded tournament battles failed."
)

assert section5b_battle_results_df[
    "battle_completed"
].all()

assert section5b_battle_results_df[
    "recorded_turns"
].gt(0).all()

assert section5b_battle_results_df[
    "policy_decisions"
].gt(0).all()

assert section5b_battle_results_df[
    "winner"
].isin(
        [
            "Player",
            "Opponent",
            "Draw",
        ]
    ).all()

assert section5b_battle_results_df[
    "final_player_hp"
].ge(0.0).all()

assert section5b_battle_results_df[
    "final_opponent_hp"
].ge(0.0).all()

assert section5b_battle_results_df[
    "fallback_rate"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert section5b_battle_results_df[
    "masking_change_rate"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert section5b_battle_results_df[
    "average_confidence"
].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

assert (
    len(
        section5b_turn_results_df
    )
    == int(
        section5b_battle_results_df[
            "recorded_turns"
        ].sum()
    )
)

assert (
    len(
        section5b_policy_history_df
    )
    == int(
        section5b_battle_results_df[
            "policy_decisions"
        ].sum()
    )
)

assert set(
    section5b_condition_summary_df[
        "condition_id"
    ]
) == {
    "BASELINE",
    "HP_PRESSURE",
    "ENERGY_PRESSURE",
}


# --------------------------------------------------------------------------------------
# 7. Save reports
# --------------------------------------------------------------------------------------

SECTION5B_BATTLE_RESULTS_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_expanded_battle_results.csv"
)

SECTION5B_TURN_RESULTS_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_expanded_turn_results.csv"
)

SECTION5B_POLICY_HISTORY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_expanded_policy_history.csv"
)

SECTION5B_CONDITION_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_condition_summary.csv"
)

SECTION5B_FAILURES_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_expanded_failures.csv"
)

SECTION5B_SUMMARY_FILE = (
    SECTION5_REPORT_DIR
    / "section5b_summary.json"
)


section5b_battle_results_df.to_csv(
    SECTION5B_BATTLE_RESULTS_FILE,
    index=False,
)

section5b_turn_results_df.to_csv(
    SECTION5B_TURN_RESULTS_FILE,
    index=False,
)

section5b_policy_history_df.to_csv(
    SECTION5B_POLICY_HISTORY_FILE,
    index=False,
)

section5b_condition_summary_df.to_csv(
    SECTION5B_CONDITION_SUMMARY_FILE,
    index=False,
)

section5b_failures_df.to_csv(
    SECTION5B_FAILURES_FILE,
    index=False,
)


total_expanded_decisions = int(
    section5b_battle_results_df[
        "policy_decisions"
    ].sum()
)

total_expanded_masking_changes = int(
    section5b_battle_results_df[
        "masking_changes"
    ].sum()
)

total_expanded_fallbacks = int(
    section5b_battle_results_df[
        "fallback_decisions"
    ].sum()
)


section5b_summary = {
    "status":
        "EXPANDED_TOURNAMENT_COMPLETED",

    "battle_count":
        int(
            len(
                section5b_battle_results_df
            )
        ),

    "battles_completed":
        int(
            section5b_battle_results_df[
                "battle_completed"
            ].sum()
        ),

    "battles_failed":
        int(
            len(
                section5b_failures_df
            )
        ),

    "player_wins":
        int(
            section5b_battle_results_df[
                "player_won"
            ].sum()
        ),

    "opponent_wins":
        int(
            section5b_battle_results_df[
                "opponent_won"
            ].sum()
        ),

    "draws":
        int(
            section5b_battle_results_df[
                "draw"
            ].sum()
        ),

    "starter_win_rate":
        float(
            section5b_battle_results_df[
                "starter_won"
            ].mean()
        ),

    "average_turns":
        float(
            section5b_battle_results_df[
                "recorded_turns"
            ].mean()
        ),

    "total_policy_decisions":
        total_expanded_decisions,

    "total_masking_changes":
        total_expanded_masking_changes,

    "overall_masking_change_rate":
        float(
            total_expanded_masking_changes
            / total_expanded_decisions
            if total_expanded_decisions > 0
            else 0.0
        ),

    "total_fallback_decisions":
        total_expanded_fallbacks,

    "overall_fallback_rate":
        float(
            total_expanded_fallbacks
            / total_expanded_decisions
            if total_expanded_decisions > 0
            else 0.0
        ),

    "average_confidence":
        float(
            section5b_policy_history_df[
                "confidence"
            ].mean()
        ),

    "next_stage":
        "EXPANDED_TOURNAMENT_ANALYSIS",
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


# --------------------------------------------------------------------------------------
# 8. Final summary
# --------------------------------------------------------------------------------------

print()
print("SECTION 5B SUMMARY")
print("-" * 100)

for key, value in (
    section5b_summary.items()
):

    print(
        f"{key:36}: {value}"
    )


print()
print("SAVED SECTION 5B REPORTS")
print("-" * 100)

print(SECTION5B_BATTLE_RESULTS_FILE)
print(SECTION5B_TURN_RESULTS_FILE)
print(SECTION5B_POLICY_HISTORY_FILE)
print(SECTION5B_CONDITION_SUMMARY_FILE)
print(SECTION5B_FAILURES_FILE)
print(SECTION5B_SUMMARY_FILE)

print()
print(
    "✅ SECTION 5B EXPANDED TOURNAMENT EXECUTION PASSED"
)


# # Section 5C — Expanded Tournament Analysis
# 
# ## This section analyzes the results from the expanded tournament evaluation.
# 
# #### The analysis summarizes:
# 
# - tournament outcomes,
# - condition-specific performance,
# - Pokémon matchup performance,
# - policy confidence,
# - masking behavior,
# - fallback usage,
# - action frequencies,
# - strengths and weaknesses of the deployed policy.
# 
# #### The resulting reports become the primary benchmark for future notebook comparisons.

# In[13]:


# ======================================================================================
# SECTION 5C — EXPANDED TOURNAMENT ANALYSIS
# ======================================================================================

print("=" * 100)
print("SECTION 5C — EXPANDED TOURNAMENT ANALYSIS")
print("=" * 100)

SECTION5C_ANALYSIS_DIR = (
    SECTION5_REPORT_DIR / "analysis"
)

SECTION5C_ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ## Overall Tournament Summary

# In[14]:


overall_summary = pd.DataFrame([
{
    "Battles": len(section5b_battle_results_df),
    "Player Wins": int(section5b_battle_results_df["player_won"].sum()),
    "Opponent Wins": int(section5b_battle_results_df["opponent_won"].sum()),
    "Draws": int(section5b_battle_results_df["draw"].sum()),
    "Starter Win Rate":
        float(section5b_battle_results_df["starter_won"].mean()),
    "Average Turns":
        float(section5b_battle_results_df["recorded_turns"].mean()),
    "Average Confidence":
        float(section5b_battle_results_df["average_confidence"].mean()),
    "Masking Rate":
        float(section5b_battle_results_df["masking_change_rate"].mean()),
    "Fallback Rate":
        float(section5b_battle_results_df["fallback_rate"].mean()),
}
])

print()
print("OVERALL TOURNAMENT SUMMARY")
print("-"*100)

display(overall_summary)


# ## Win Rates by Condition

# In[15]:


condition_analysis = (
    section5b_battle_results_df
    .groupby("condition_id", as_index=False)
    .agg(
        Battles=("battle_number","count"),
        PlayerWins=("player_won","sum"),
        OpponentWins=("opponent_won","sum"),
        Draws=("draw","sum"),
        AvgTurns=("recorded_turns","mean"),
        AvgConfidence=("average_confidence","mean"),
        AvgMasking=("masking_change_rate","mean"),
        AvgFallback=("fallback_rate","mean"),
    )
)

condition_analysis["PlayerWinRate"] = (
    condition_analysis["PlayerWins"] /
    condition_analysis["Battles"]
)

condition_analysis["OpponentWinRate"] = (
    condition_analysis["OpponentWins"] /
    condition_analysis["Battles"]
)

print()
print("CONDITION ANALYSIS")
print("-"*100)

display(condition_analysis)


# ## Card Performance

# In[16]:


card_analysis = (
    section5b_battle_results_df
    .groupby("player_card", as_index=False)
    .agg(
        Battles=("battle_number","count"),
        Wins=("player_won","sum"),
        AvgConfidence=("average_confidence","mean"),
        AvgTurns=("recorded_turns","mean"),
    )
)

card_analysis["WinRate"] = (
    card_analysis["Wins"] /
    card_analysis["Battles"]
)

card_analysis = (
    card_analysis
    .sort_values(
        "WinRate",
        ascending=False
    )
)

print()
print("PLAYER CARD PERFORMANCE")
print("-"*100)

display(card_analysis)


# ## Action Usage

# In[17]:


action_analysis = (
    section5b_policy_history_df
    .groupby("selected_move", as_index=False)
    .agg(
        Uses=("selected_move","count"),
        AvgConfidence=("confidence","mean"),
    )
    .sort_values(
        "Uses",
        ascending=False
    )
)

print()
print("ACTION USAGE")
print("-"*100)

display(action_analysis)


# ## Masking Analysis

# In[18]:


masking_analysis = (
    section5b_policy_history_df
    .groupby(
        "masking_changed_action",
        as_index=False
    )
    .size()
)

print()
print("MASKING ANALYSIS")
print("-"*100)

display(masking_analysis)


# ## Confidence Distribution

# In[19]:


confidence_summary = (
    section5b_policy_history_df[
        "confidence"
    ].describe()
)

print()
print("CONFIDENCE SUMMARY")
print("-"*100)

print(confidence_summary)


# ## Save Reports

# In[20]:


SECTION5C_OVERALL_FILE = (
    SECTION5C_ANALYSIS_DIR /
    "section5c_overall_summary.csv"
)

SECTION5C_CONDITION_FILE = (
    SECTION5C_ANALYSIS_DIR /
    "section5c_condition_analysis.csv"
)

SECTION5C_CARD_FILE = (
    SECTION5C_ANALYSIS_DIR /
    "section5c_card_analysis.csv"
)

SECTION5C_ACTION_FILE = (
    SECTION5C_ANALYSIS_DIR /
    "section5c_action_analysis.csv"
)

overall_summary.to_csv(
    SECTION5C_OVERALL_FILE,
    index=False
)

condition_analysis.to_csv(
    SECTION5C_CONDITION_FILE,
    index=False
)

card_analysis.to_csv(
    SECTION5C_CARD_FILE,
    index=False
)

action_analysis.to_csv(
    SECTION5C_ACTION_FILE,
    index=False
)


# ## Final Summary

# In[21]:


print()
print("SECTION 5C SUMMARY")
print("-"*100)

print(
    f"Battles analyzed        : {len(section5b_battle_results_df)}"
)

print(
    f"Policy decisions        : {len(section5b_policy_history_df)}"
)

print(
    f"Unique actions          : {action_analysis.shape[0]}"
)

print(
    f"Cards analyzed          : {card_analysis.shape[0]}"
)

print(
    f"Conditions analyzed     : {condition_analysis.shape[0]}"
)

print()
print("SAVED SECTION 5C REPORTS")
print("-"*100)

print(SECTION5C_OVERALL_FILE)
print(SECTION5C_CONDITION_FILE)
print(SECTION5C_CARD_FILE)
print(SECTION5C_ACTION_FILE)

print()
print("✅ SECTION 5C EXPANDED TOURNAMENT ANALYSIS PASSED")


# ## Section 6A TOURNAMENT PERFORMANCE DASHBOARD

# In[24]:


# ======================================================================================
# SECTION 6A — TOURNAMENT PERFORMANCE DASHBOARD
# ======================================================================================

print("=" * 100)
print("SECTION 6A — TOURNAMENT PERFORMANCE DASHBOARD")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Validate required Section 5C objects
# --------------------------------------------------------------------------------------

required_section5c_objects = {
    "overall_summary": overall_summary,
    "condition_analysis": condition_analysis,
    "card_analysis": card_analysis,
    "action_analysis": action_analysis,
    "confidence_summary": confidence_summary,
}

for object_name, object_value in required_section5c_objects.items():

    assert object_value is not None, (
        f"Required Section 5C object is missing: {object_name}"
    )


# --------------------------------------------------------------------------------------
# 2. Overall tournament dashboard
# --------------------------------------------------------------------------------------

overall_dashboard = pd.DataFrame(
    [
        {
            "Metric": "Battles",
            "Value": int(
                len(
                    section5b_battle_results_df
                )
            ),
        },
        {
            "Metric": "Player Wins",
            "Value": int(
                (
                    section5b_battle_results_df[
                        "winner"
                    ]
                    == "Player"
                ).sum()
            ),
        },
        {
            "Metric": "Opponent Wins",
            "Value": int(
                (
                    section5b_battle_results_df[
                        "winner"
                    ]
                    == "Opponent"
                ).sum()
            ),
        },
        {
            "Metric": "Draws",
            "Value": int(
                (
                    section5b_battle_results_df[
                        "winner"
                    ]
                    == "Draw"
                ).sum()
            ),
        },
        {
            "Metric": "Starter Win Rate",
            "Value": float(
                overall_summary.iloc[0][
                    "Starter Win Rate"
                ]
            ),
        },
        {
            "Metric": "Average Turns",
            "Value": float(
                overall_summary.iloc[0][
                    "Average Turns"
                ]
            ),
        },
        {
            "Metric": "Average Confidence",
            "Value": float(
                confidence_summary[
                    "mean"
                ]
            ),
        },
        {
            "Metric": "Masking Rate",
            "Value": float(
                section5b_summary[
                    "overall_masking_change_rate"
                ]
            ),
        },
        {
            "Metric": "Fallback Rate",
            "Value": float(
                section5b_summary[
                    "overall_fallback_rate"
                ]
            ),
        },
        {
            "Metric": "Total Policy Decisions",
            "Value": int(
                len(
                    section5b_policy_history_df
                )
            ),
        },
    ]
)


print()
print("OVERALL PERFORMANCE")
print("-" * 100)

display(
    overall_dashboard
)


# --------------------------------------------------------------------------------------
# 3. Player-card leaderboard
# --------------------------------------------------------------------------------------

leaderboard = (
    card_analysis
    .sort_values(
        [
            "WinRate",
            "Wins",
            "AvgConfidence",
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
print("PLAYER CARD LEADERBOARD")
print("-" * 100)

display(
    leaderboard
)


# --------------------------------------------------------------------------------------
# 4. Tournament-condition leaderboard
# --------------------------------------------------------------------------------------

condition_leaderboard = (
    condition_analysis
    .sort_values(
        [
            "PlayerWinRate",
            "OpponentWinRate",
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
print("TOURNAMENT CONDITION LEADERBOARD")
print("-" * 100)

display(
    condition_leaderboard
)


# --------------------------------------------------------------------------------------
# 5. Action-distribution dashboard
# --------------------------------------------------------------------------------------

action_dashboard = (
    action_analysis
    .sort_values(
        "Uses",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

action_dashboard[
    "UsageFraction"
] = (
    action_dashboard[
        "Uses"
    ]
    / action_dashboard[
        "Uses"
    ].sum()
)


print()
print("ACTION DISTRIBUTION")
print("-" * 100)

display(
    action_dashboard
)


# --------------------------------------------------------------------------------------
# 6. Confidence dashboard
# --------------------------------------------------------------------------------------

confidence_dashboard = pd.DataFrame(
    [
        {
            "Statistic": "Minimum",
            "Value": float(
                confidence_summary[
                    "min"
                ]
            ),
        },
        {
            "Statistic": "25%",
            "Value": float(
                confidence_summary[
                    "25%"
                ]
            ),
        },
        {
            "Statistic": "Median",
            "Value": float(
                confidence_summary[
                    "50%"
                ]
            ),
        },
        {
            "Statistic": "Mean",
            "Value": float(
                confidence_summary[
                    "mean"
                ]
            ),
        },
        {
            "Statistic": "75%",
            "Value": float(
                confidence_summary[
                    "75%"
                ]
            ),
        },
        {
            "Statistic": "Maximum",
            "Value": float(
                confidence_summary[
                    "max"
                ]
            ),
        },
        {
            "Statistic": "Standard Deviation",
            "Value": float(
                confidence_summary[
                    "std"
                ]
            ),
        },
    ]
)


print()
print("CONFIDENCE METRICS")
print("-" * 100)

display(
    confidence_dashboard
)


# --------------------------------------------------------------------------------------
# 7. Masking and fallback dashboard
# --------------------------------------------------------------------------------------

masking_dashboard = pd.DataFrame(
    [
        {
            "Metric": "Total Decisions",
            "Value": int(
                len(
                    section5b_policy_history_df
                )
            ),
        },
        {
            "Metric": "Masking Changes",
            "Value": int(
                section5b_policy_history_df[
                    "masking_changed_action"
                ].sum()
            ),
        },
        {
            "Metric": "Masking Change Rate",
            "Value": float(
                section5b_summary[
                    "overall_masking_change_rate"
                ]
            ),
        },
        {
            "Metric": "Fallback Decisions",
            "Value": int(
                section5b_policy_history_df[
                    "fallback_used"
                ].sum()
            ),
        },
        {
            "Metric": "Fallback Rate",
            "Value": float(
                section5b_summary[
                    "overall_fallback_rate"
                ]
            ),
        },
    ]
)


print()
print("MASKING AND FALLBACK METRICS")
print("-" * 100)

display(
    masking_dashboard
)


# --------------------------------------------------------------------------------------
# 8. Save dashboard reports
# --------------------------------------------------------------------------------------

SECTION6_REPORT_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section6"
)

SECTION6_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SECTION6A_OVERALL_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_overall_dashboard.csv"
)

SECTION6A_CARD_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_card_leaderboard.csv"
)

SECTION6A_CONDITION_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_condition_leaderboard.csv"
)

SECTION6A_ACTION_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_action_distribution.csv"
)

SECTION6A_CONFIDENCE_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_confidence_metrics.csv"
)

SECTION6A_MASKING_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_masking_fallback_metrics.csv"
)

SECTION6A_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6a_dashboard_summary.json"
)


overall_dashboard.to_csv(
    SECTION6A_OVERALL_FILE,
    index=False,
)

leaderboard.to_csv(
    SECTION6A_CARD_FILE,
    index=False,
)

condition_leaderboard.to_csv(
    SECTION6A_CONDITION_FILE,
    index=False,
)

action_dashboard.to_csv(
    SECTION6A_ACTION_FILE,
    index=False,
)

confidence_dashboard.to_csv(
    SECTION6A_CONFIDENCE_FILE,
    index=False,
)

masking_dashboard.to_csv(
    SECTION6A_MASKING_FILE,
    index=False,
)


section6a_summary = {
    "status":
        "TOURNAMENT_PERFORMANCE_DASHBOARD_COMPLETE",

    "battles":
        int(
            len(
                section5b_battle_results_df
            )
        ),

    "policy_decisions":
        int(
            len(
                section5b_policy_history_df
            )
        ),

    "player_wins":
        int(
            (
                section5b_battle_results_df[
                    "winner"
                ]
                == "Player"
            ).sum()
        ),

    "opponent_wins":
        int(
            (
                section5b_battle_results_df[
                    "winner"
                ]
                == "Opponent"
            ).sum()
        ),

    "draws":
        int(
            (
                section5b_battle_results_df[
                    "winner"
                ]
                == "Draw"
            ).sum()
        ),

    "starter_win_rate":
        float(
            overall_summary.iloc[0][
                "Starter Win Rate"
            ]
        ),

    "average_turns":
        float(
            overall_summary.iloc[0][
                "Average Turns"
            ]
        ),

    "average_confidence":
        float(
            confidence_summary[
                "mean"
            ]
        ),

    "masking_change_rate":
        float(
            section5b_summary[
                "overall_masking_change_rate"
            ]
        ),

    "fallback_rate":
        float(
            section5b_summary[
                "overall_fallback_rate"
            ]
        ),

    "top_player_card":
        str(
            leaderboard.iloc[0][
                "player_card"
            ]
        ),

    "top_player_card_win_rate":
        float(
            leaderboard.iloc[0][
                "WinRate"
            ]
        ),

    "best_condition":
        str(
            condition_leaderboard.iloc[0][
                "condition_id"
            ]
        ),

    "best_condition_player_win_rate":
        float(
            condition_leaderboard.iloc[0][
                "PlayerWinRate"
            ]
        ),

    "next_stage":
        "TOURNAMENT_PERFORMANCE_VALIDATION",
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


# --------------------------------------------------------------------------------------
# 9. Validation
# --------------------------------------------------------------------------------------

required_section6a_files = [
    SECTION6A_OVERALL_FILE,
    SECTION6A_CARD_FILE,
    SECTION6A_CONDITION_FILE,
    SECTION6A_ACTION_FILE,
    SECTION6A_CONFIDENCE_FILE,
    SECTION6A_MASKING_FILE,
    SECTION6A_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    for file_path in required_section6a_files
)

assert all(
    file_path.stat().st_size > 0
    for file_path in required_section6a_files
)

assert len(
    overall_dashboard
) == 10

assert int(
    overall_dashboard.loc[
        overall_dashboard[
            "Metric"
        ] == "Battles",
        "Value",
    ].iloc[0]
) == 24

assert int(
    overall_dashboard.loc[
        overall_dashboard[
            "Metric"
        ] == "Total Policy Decisions",
        "Value",
    ].iloc[0]
) == 210

assert len(
    leaderboard
) == 4

assert len(
    condition_leaderboard
) == 3

assert len(
    action_dashboard
) == 6


# --------------------------------------------------------------------------------------
# 10. Final summary
# --------------------------------------------------------------------------------------

print()
print("SECTION 6A SUMMARY")
print("-" * 100)

for key, value in section6a_summary.items():

    print(
        f"{key:38}: {value}"
    )


print()
print("SAVED SECTION 6A REPORTS")
print("-" * 100)

for file_path in required_section6a_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6A TOURNAMENT PERFORMANCE DASHBOARD PASSED"
)


# ## Section 6B — Tournament Performance Validation
# 
# #### This section validates the complete Notebook 52 tournament benchmark.
# 
# #### It confirms that:
# 
# - all 24 expanded battles completed,
# - all outcome counts reconcile,
# - every policy decision is represented,
# - masking and fallback totals match the saved summaries,
# - condition, card, action, and confidence dashboards are complete,
# - all Section 6A artifacts exist and are nonempty,
# - the benchmark is ready for final handoff to Notebook 53.

# In[25]:


# ======================================================================================
# SECTION 6B — TOURNAMENT PERFORMANCE VALIDATION
# ======================================================================================

print("=" * 100)
print("SECTION 6B — TOURNAMENT PERFORMANCE VALIDATION")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Reconcile core tournament totals
# --------------------------------------------------------------------------------------

validated_battle_count = int(
    len(
        section5b_battle_results_df
    )
)

validated_player_wins = int(
    (
        section5b_battle_results_df[
            "winner"
        ]
        == "Player"
    ).sum()
)

validated_opponent_wins = int(
    (
        section5b_battle_results_df[
            "winner"
        ]
        == "Opponent"
    ).sum()
)

validated_draws = int(
    (
        section5b_battle_results_df[
            "winner"
        ]
        == "Draw"
    ).sum()
)

validated_outcome_total = (
    validated_player_wins
    + validated_opponent_wins
    + validated_draws
)

validated_policy_decisions = int(
    len(
        section5b_policy_history_df
    )
)

validated_turn_records = int(
    len(
        section5b_turn_results_df
    )
)

validated_masking_changes = int(
    section5b_policy_history_df[
        "masking_changed_action"
    ].astype(bool).sum()
)

validated_fallback_decisions = int(
    section5b_policy_history_df[
        "fallback_used"
    ].astype(bool).sum()
)

validated_masking_rate = float(
    validated_masking_changes
    / validated_policy_decisions
    if validated_policy_decisions > 0
    else 0.0
)

validated_fallback_rate = float(
    validated_fallback_decisions
    / validated_policy_decisions
    if validated_policy_decisions > 0
    else 0.0
)

validated_average_confidence = float(
    section5b_policy_history_df[
        "confidence"
    ].mean()
)

validated_average_turns = float(
    section5b_battle_results_df[
        "recorded_turns"
    ].mean()
)

validated_starter_win_rate = float(
    section5b_battle_results_df[
        "starter_won"
    ].astype(bool).mean()
)


# --------------------------------------------------------------------------------------
# 2. Artifact inventory
# --------------------------------------------------------------------------------------

section6a_artifact_map = {
    "overall_dashboard":
        SECTION6A_OVERALL_FILE,

    "card_leaderboard":
        SECTION6A_CARD_FILE,

    "condition_leaderboard":
        SECTION6A_CONDITION_FILE,

    "action_distribution":
        SECTION6A_ACTION_FILE,

    "confidence_metrics":
        SECTION6A_CONFIDENCE_FILE,

    "masking_fallback_metrics":
        SECTION6A_MASKING_FILE,

    "dashboard_summary":
        SECTION6A_SUMMARY_FILE,
}


section6b_artifact_rows = []

for artifact_name, artifact_path in (
    section6a_artifact_map.items()
):

    artifact_path = Path(
        artifact_path
    )

    section6b_artifact_rows.append(
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
                    artifact_path.stat().st_size
                    if artifact_path.exists()
                    else 0
                ),

            "nonempty":
                (
                    artifact_path.exists()
                    and artifact_path.stat().st_size > 0
                ),
        }
    )


section6b_artifact_inventory_df = pd.DataFrame(
    section6b_artifact_rows
)


print()
print("SECTION 6A ARTIFACT INVENTORY")
print("-" * 100)

display(
    section6b_artifact_inventory_df
)


# --------------------------------------------------------------------------------------
# 3. Validation checks
# --------------------------------------------------------------------------------------

section6b_validation_rows = []


def add_section6b_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:
    """
    Append one validation check.
    """

    section6b_validation_rows.append(
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


add_section6b_check(
    "battle_count_valid",
    validated_battle_count == 24,
    validated_battle_count,
    24,
)

add_section6b_check(
    "all_battles_completed",
    bool(
        section5b_battle_results_df[
            "battle_completed"
        ].astype(bool).all()
    ),
    int(
        section5b_battle_results_df[
            "battle_completed"
        ].astype(bool).sum()
    ),
    24,
)

add_section6b_check(
    "outcomes_reconcile",
    validated_outcome_total == validated_battle_count,
    validated_outcome_total,
    validated_battle_count,
)

add_section6b_check(
    "winner_values_valid",
    bool(
        section5b_battle_results_df[
            "winner"
        ].isin(
            [
                "Player",
                "Opponent",
                "Draw",
            ]
        ).all()
    ),
    sorted(
        section5b_battle_results_df[
            "winner"
        ].dropna().unique().tolist()
    ),
    [
        "Draw",
        "Opponent",
        "Player",
    ],
)

add_section6b_check(
    "policy_decision_count_valid",
    validated_policy_decisions == 210,
    validated_policy_decisions,
    210,
)

add_section6b_check(
    "turn_record_count_valid",
    validated_turn_records
    == int(
        section5b_battle_results_df[
            "recorded_turns"
        ].sum()
    ),
    validated_turn_records,
    int(
        section5b_battle_results_df[
            "recorded_turns"
        ].sum()
    ),
)

add_section6b_check(
    "policy_history_matches_battle_totals",
    validated_policy_decisions
    == int(
        section5b_battle_results_df[
            "policy_decisions"
        ].sum()
    ),
    validated_policy_decisions,
    int(
        section5b_battle_results_df[
            "policy_decisions"
        ].sum()
    ),
)

add_section6b_check(
    "masking_change_total_matches_summary",
    validated_masking_changes
    == int(
        section5b_summary[
            "total_masking_changes"
        ]
    ),
    validated_masking_changes,
    int(
        section5b_summary[
            "total_masking_changes"
        ]
    ),
)

add_section6b_check(
    "masking_rate_matches_summary",
    np.isclose(
        validated_masking_rate,
        float(
            section5b_summary[
                "overall_masking_change_rate"
            ]
        ),
        atol=1e-12,
    ),
    validated_masking_rate,
    float(
        section5b_summary[
            "overall_masking_change_rate"
        ]
    ),
)

add_section6b_check(
    "fallback_total_matches_summary",
    validated_fallback_decisions
    == int(
        section5b_summary[
            "total_fallback_decisions"
        ]
    ),
    validated_fallback_decisions,
    int(
        section5b_summary[
            "total_fallback_decisions"
        ]
    ),
)

add_section6b_check(
    "fallback_rate_matches_summary",
    np.isclose(
        validated_fallback_rate,
        float(
            section5b_summary[
                "overall_fallback_rate"
            ]
        ),
        atol=1e-12,
    ),
    validated_fallback_rate,
    float(
        section5b_summary[
            "overall_fallback_rate"
        ]
    ),
)

add_section6b_check(
    "average_confidence_matches_dashboard",
    np.isclose(
        validated_average_confidence,
        float(
            section6a_summary[
                "average_confidence"
            ]
        ),
        atol=1e-12,
    ),
    validated_average_confidence,
    float(
        section6a_summary[
            "average_confidence"
        ]
    ),
)

add_section6b_check(
    "average_turns_matches_dashboard",
    np.isclose(
        validated_average_turns,
        float(
            section6a_summary[
                "average_turns"
            ]
        ),
        atol=1e-12,
    ),
    validated_average_turns,
    float(
        section6a_summary[
            "average_turns"
        ]
    ),
)

add_section6b_check(
    "starter_win_rate_matches_dashboard",
    np.isclose(
        validated_starter_win_rate,
        float(
            section6a_summary[
                "starter_win_rate"
            ]
        ),
        atol=1e-12,
    ),
    validated_starter_win_rate,
    float(
        section6a_summary[
            "starter_win_rate"
        ]
    ),
)

add_section6b_check(
    "card_leaderboard_complete",
    len(
        leaderboard
    ) == 4,
    len(
        leaderboard
    ),
    4,
)

add_section6b_check(
    "condition_leaderboard_complete",
    len(
        condition_leaderboard
    ) == 3,
    len(
        condition_leaderboard
    ),
    3,
)

add_section6b_check(
    "action_distribution_complete",
    len(
        action_dashboard
    ) == 6,
    len(
        action_dashboard
    ),
    6,
)

add_section6b_check(
    "all_policy_classes_represented",
    set(
        action_dashboard[
            "selected_move"
        ]
    )
    == set(
        policy_label_encoder.classes_
    ),
    sorted(
        action_dashboard[
            "selected_move"
        ].tolist()
    ),
    sorted(
        policy_label_encoder.classes_.tolist()
    ),
)

add_section6b_check(
    "confidence_values_valid",
    bool(
        section5b_policy_history_df[
            "confidence"
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()
    ),
    {
        "minimum":
            float(
                section5b_policy_history_df[
                    "confidence"
                ].min()
            ),

        "maximum":
            float(
                section5b_policy_history_df[
                    "confidence"
                ].max()
            ),
    },
    "All values between 0 and 1",
)

add_section6b_check(
    "all_section6a_artifacts_exist",
    bool(
        section6b_artifact_inventory_df[
            "exists"
        ].all()
    ),
    int(
        (
            ~section6b_artifact_inventory_df[
                "exists"
            ]
        ).sum()
    ),
    0,
)

add_section6b_check(
    "all_section6a_artifacts_are_files",
    bool(
        section6b_artifact_inventory_df[
            "is_file"
        ].all()
    ),
    int(
        (
            ~section6b_artifact_inventory_df[
                "is_file"
            ]
        ).sum()
    ),
    0,
)

add_section6b_check(
    "all_section6a_artifacts_nonempty",
    bool(
        section6b_artifact_inventory_df[
            "nonempty"
        ].all()
    ),
    int(
        (
            ~section6b_artifact_inventory_df[
                "nonempty"
            ]
        ).sum()
    ),
    0,
)


section6b_validation_checks_df = pd.DataFrame(
    section6b_validation_rows
)


print()
print("TOURNAMENT PERFORMANCE VALIDATION CHECKS")
print("-" * 100)

display(
    section6b_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 4. Final benchmark profile
# --------------------------------------------------------------------------------------

section6b_benchmark_profile_df = pd.DataFrame(
    [
        {
            "metric":
                "battle_count",

            "value":
                validated_battle_count,
        },
        {
            "metric":
                "player_wins",

            "value":
                validated_player_wins,
        },
        {
            "metric":
                "opponent_wins",

            "value":
                validated_opponent_wins,
        },
        {
            "metric":
                "draws",

            "value":
                validated_draws,
        },
        {
            "metric":
                "policy_decisions",

            "value":
                validated_policy_decisions,
        },
        {
            "metric":
                "turn_records",

            "value":
                validated_turn_records,
        },
        {
            "metric":
                "starter_win_rate",

            "value":
                validated_starter_win_rate,
        },
        {
            "metric":
                "average_turns",

            "value":
                validated_average_turns,
        },
        {
            "metric":
                "average_confidence",

            "value":
                validated_average_confidence,
        },
        {
            "metric":
                "masking_changes",

            "value":
                validated_masking_changes,
        },
        {
            "metric":
                "masking_change_rate",

            "value":
                validated_masking_rate,
        },
        {
            "metric":
                "fallback_decisions",

            "value":
                validated_fallback_decisions,
        },
        {
            "metric":
                "fallback_rate",

            "value":
                validated_fallback_rate,
        },
        {
            "metric":
                "top_player_card",

            "value":
                str(
                    section6a_summary[
                        "top_player_card"
                    ]
                ),
        },
        {
            "metric":
                "top_player_card_win_rate",

            "value":
                float(
                    section6a_summary[
                        "top_player_card_win_rate"
                    ]
                ),
        },
        {
            "metric":
                "best_condition",

            "value":
                str(
                    section6a_summary[
                        "best_condition"
                    ]
                ),
        },
        {
            "metric":
                "best_condition_player_win_rate",

            "value":
                float(
                    section6a_summary[
                        "best_condition_player_win_rate"
                    ]
                ),
        },
    ]
)


print()
print("FINAL TOURNAMENT BENCHMARK PROFILE")
print("-" * 100)

display(
    section6b_benchmark_profile_df
)


# --------------------------------------------------------------------------------------
# 5. Save Section 6B reports
# --------------------------------------------------------------------------------------

SECTION6B_ARTIFACT_INVENTORY_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_artifact_inventory.csv"
)

SECTION6B_VALIDATION_CHECKS_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_validation_checks.csv"
)

SECTION6B_BENCHMARK_PROFILE_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_benchmark_profile.csv"
)

SECTION6B_SUMMARY_FILE = (
    SECTION6_REPORT_DIR
    / "section6b_validation_summary.json"
)


section6b_artifact_inventory_df.to_csv(
    SECTION6B_ARTIFACT_INVENTORY_FILE,
    index=False,
)

section6b_validation_checks_df.to_csv(
    SECTION6B_VALIDATION_CHECKS_FILE,
    index=False,
)

section6b_benchmark_profile_df.to_csv(
    SECTION6B_BENCHMARK_PROFILE_FILE,
    index=False,
)


section6b_checks_passed = int(
    section6b_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section6b_checks_failed = int(
    (
        ~section6b_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


section6b_summary = {
    "status":
        (
            "TOURNAMENT_BENCHMARK_VALIDATED"
            if section6b_checks_failed == 0
            else "TOURNAMENT_BENCHMARK_VALIDATION_FAILED"
        ),

    "validation_checks_total":
        int(
            len(
                section6b_validation_checks_df
            )
        ),

    "validation_checks_passed":
        section6b_checks_passed,

    "validation_checks_failed":
        section6b_checks_failed,

    "battle_count":
        validated_battle_count,

    "policy_decisions":
        validated_policy_decisions,

    "masking_change_rate":
        validated_masking_rate,

    "fallback_rate":
        validated_fallback_rate,

    "average_confidence":
        validated_average_confidence,

    "benchmark_ready":
        bool(
            section6b_checks_failed == 0
        ),

    "next_stage":
        "NOTEBOOK52_FINAL_HANDOFF",
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


# --------------------------------------------------------------------------------------
# 6. Final assertions
# --------------------------------------------------------------------------------------

assert section6b_checks_failed == 0, (
    "One or more tournament performance validation checks failed."
)

assert section6b_summary[
    "status"
] == "TOURNAMENT_BENCHMARK_VALIDATED"

assert section6b_summary[
    "benchmark_ready"
] is True


required_section6b_files = [
    SECTION6B_ARTIFACT_INVENTORY_FILE,
    SECTION6B_VALIDATION_CHECKS_FILE,
    SECTION6B_BENCHMARK_PROFILE_FILE,
    SECTION6B_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in required_section6b_files
)


# --------------------------------------------------------------------------------------
# 7. Final output
# --------------------------------------------------------------------------------------

print()
print("SECTION 6B SUMMARY")
print("-" * 100)

for key, value in (
    section6b_summary.items()
):

    print(
        f"{key:36}: {value}"
    )


print()
print("SAVED SECTION 6B REPORTS")
print("-" * 100)

for file_path in required_section6b_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 6B TOURNAMENT PERFORMANCE VALIDATION PASSED"
)


# # Section 7 — Final Notebook Handoff
# 
# ## Section 7A — Final Validation and Notebook 52 Handoff
# 
# #### This section performs the final Notebook 52 validation and prepares the benchmark handoff for Notebook 53.
# 
# It records:
# 
# - completed sections,
# - tournament benchmark metrics,
# - validated artifact inventory,
# - final performance profile,
# - identified policy weaknesses,
# - recommended optimization targets,
# - readiness for policy improvement.

# In[26]:


# ======================================================================================
# SECTION 7A — FINAL VALIDATION AND NOTEBOOK 52 HANDOFF
# ======================================================================================

print("=" * 100)
print("SECTION 7A — FINAL VALIDATION AND NOTEBOOK 52 HANDOFF")
print("=" * 100)


# --------------------------------------------------------------------------------------
# 1. Final report directory
# --------------------------------------------------------------------------------------

SECTION7_REPORT_DIR = (
    NOTEBOOK52_REPORT_DIR
    / "section7"
)

SECTION7_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------------------
# 2. Required final artifacts
# --------------------------------------------------------------------------------------

required_notebook52_artifacts = {
    "section3a_scenario_catalog":
        SECTION3A_SCENARIO_CATALOG_FILE,

    "section3b_scenario_validation":
        SECTION3B_SCENARIO_VALIDATION_FILE,

    "section3b_scenario_probabilities":
        SECTION3B_SCENARIO_PROBABILITIES_FILE,

    "section3b_scenario_summary":
        SECTION3B_SUMMARY_FILE,

    "section4a_battle_results":
        SECTION4A_BATTLE_RESULTS_FILE,

    "section4a_turn_results":
        SECTION4A_TURN_RESULTS_FILE,

    "section4a_policy_history":
        SECTION4A_POLICY_HISTORY_FILE,

    "section4a_summary":
        SECTION4A_SUMMARY_FILE,

    "section4b_battle_analysis":
        SECTION4B_BATTLE_ANALYSIS_FILE,

    "section4b_card_summary":
        SECTION4B_CARD_SUMMARY_FILE,

    "section4b_action_summary":
        SECTION4B_ACTION_SUMMARY_FILE,

    "section4b_summary":
        SECTION4B_SUMMARY_FILE,

    "section5a_expanded_specs":
        SECTION5A_EXPANDED_SPEC_FILE,

    "section5a_validation":
        SECTION5A_VALIDATION_FILE,

    "section5a_summary":
        SECTION5A_SUMMARY_FILE,

    "section5b_battle_results":
        SECTION5B_BATTLE_RESULTS_FILE,

    "section5b_turn_results":
        SECTION5B_TURN_RESULTS_FILE,

    "section5b_policy_history":
        SECTION5B_POLICY_HISTORY_FILE,

    "section5b_condition_summary":
        SECTION5B_CONDITION_SUMMARY_FILE,

    "section5b_summary":
        SECTION5B_SUMMARY_FILE,

    "section5c_overall_summary":
        SECTION5C_OVERALL_FILE,

    "section5c_condition_analysis":
        SECTION5C_CONDITION_FILE,

    "section5c_card_analysis":
        SECTION5C_CARD_FILE,

    "section5c_action_analysis":
        SECTION5C_ACTION_FILE,

    "section6a_overall_dashboard":
        SECTION6A_OVERALL_FILE,

    "section6a_card_leaderboard":
        SECTION6A_CARD_FILE,

    "section6a_condition_leaderboard":
        SECTION6A_CONDITION_FILE,

    "section6a_action_distribution":
        SECTION6A_ACTION_FILE,

    "section6a_confidence_metrics":
        SECTION6A_CONFIDENCE_FILE,

    "section6a_masking_metrics":
        SECTION6A_MASKING_FILE,

    "section6a_summary":
        SECTION6A_SUMMARY_FILE,

    "section6b_artifact_inventory":
        SECTION6B_ARTIFACT_INVENTORY_FILE,

    "section6b_validation_checks":
        SECTION6B_VALIDATION_CHECKS_FILE,

    "section6b_benchmark_profile":
        SECTION6B_BENCHMARK_PROFILE_FILE,

    "section6b_summary":
        SECTION6B_SUMMARY_FILE,
}


section7_artifact_rows = []

for artifact_name, artifact_path in (
    required_notebook52_artifacts.items()
):

    artifact_path = Path(
        artifact_path
    )

    section7_artifact_rows.append(
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
                    artifact_path.stat().st_size
                    if artifact_path.exists()
                    else 0
                ),

            "nonempty":
                (
                    artifact_path.exists()
                    and artifact_path.stat().st_size > 0
                ),
        }
    )


section7_artifact_inventory_df = pd.DataFrame(
    section7_artifact_rows
)


print()
print("FINAL ARTIFACT INVENTORY")
print("-" * 100)

display(
    section7_artifact_inventory_df
)


# --------------------------------------------------------------------------------------
# 3. Final Notebook 52 validation checks
# --------------------------------------------------------------------------------------

section7_validation_rows = []


def add_section7_check(
    check_name: str,
    passed: bool,
    value: Any,
    expected: Any,
    details: str = "",
) -> None:

    section7_validation_rows.append(
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


add_section7_check(
    "all_required_artifacts_exist",
    bool(
        section7_artifact_inventory_df[
            "exists"
        ].all()
    ),
    int(
        (
            ~section7_artifact_inventory_df[
                "exists"
            ]
        ).sum()
    ),
    0,
)

add_section7_check(
    "all_required_artifacts_are_files",
    bool(
        section7_artifact_inventory_df[
            "is_file"
        ].all()
    ),
    int(
        (
            ~section7_artifact_inventory_df[
                "is_file"
            ]
        ).sum()
    ),
    0,
)

add_section7_check(
    "all_required_artifacts_nonempty",
    bool(
        section7_artifact_inventory_df[
            "nonempty"
        ].all()
    ),
    int(
        (
            ~section7_artifact_inventory_df[
                "nonempty"
            ]
        ).sum()
    ),
    0,
)

add_section7_check(
    "notebook51_handoff_ready",
    (
        notebook51_handoff_manifest[
            "status"
        ]
        == "READY_FOR_TOURNAMENT_EVALUATION"
    ),
    notebook51_handoff_manifest[
        "status"
    ],
    "READY_FOR_TOURNAMENT_EVALUATION",
)

add_section7_check(
    "base_scenarios_validated",
    int(
        section3b_summary[
            "scenarios_passed"
        ]
    ) == 8,
    int(
        section3b_summary[
            "scenarios_passed"
        ]
    ),
    8,
)

add_section7_check(
    "base_scenario_failures_zero",
    int(
        section3b_summary[
            "scenarios_failed"
        ]
    ) == 0,
    int(
        section3b_summary[
            "scenarios_failed"
        ]
    ),
    0,
)

add_section7_check(
    "single_pass_tournament_completed",
    (
        section4a_summary[
            "status"
        ]
        == "SINGLE_PASS_TOURNAMENT_COMPLETED"
    ),
    section4a_summary[
        "status"
    ],
    "SINGLE_PASS_TOURNAMENT_COMPLETED",
)

add_section7_check(
    "expanded_matrix_ready",
    (
        section5a_summary[
            "status"
        ]
        == "EXPANDED_TOURNAMENT_MATRIX_READY"
    ),
    section5a_summary[
        "status"
    ],
    "EXPANDED_TOURNAMENT_MATRIX_READY",
)

add_section7_check(
    "expanded_tournament_completed",
    (
        section5b_summary[
            "status"
        ]
        == "EXPANDED_TOURNAMENT_COMPLETED"
    ),
    section5b_summary[
        "status"
    ],
    "EXPANDED_TOURNAMENT_COMPLETED",
)

add_section7_check(
    "expanded_battle_count_valid",
    int(
        section5b_summary[
            "battle_count"
        ]
    ) == 24,
    int(
        section5b_summary[
            "battle_count"
        ]
    ),
    24,
)

add_section7_check(
    "expanded_battles_failed_zero",
    int(
        section5b_summary[
            "battles_failed"
        ]
    ) == 0,
    int(
        section5b_summary[
            "battles_failed"
        ]
    ),
    0,
)

add_section7_check(
    "policy_decision_count_valid",
    int(
        section5b_summary[
            "total_policy_decisions"
        ]
    ) == 210,
    int(
        section5b_summary[
            "total_policy_decisions"
        ]
    ),
    210,
)

add_section7_check(
    "outcomes_reconcile",
    (
        int(
            section5b_summary[
                "player_wins"
            ]
        )
        + int(
            section5b_summary[
                "opponent_wins"
            ]
        )
        + int(
            section5b_summary[
                "draws"
            ]
        )
        == 24
    ),
    {
        "player_wins":
            int(
                section5b_summary[
                    "player_wins"
                ]
            ),

        "opponent_wins":
            int(
                section5b_summary[
                    "opponent_wins"
                ]
            ),

        "draws":
            int(
                section5b_summary[
                    "draws"
                ]
            ),
    },
    "Total equals 24",
)

add_section7_check(
    "masking_rate_valid",
    (
        0.0
        <= float(
            section5b_summary[
                "overall_masking_change_rate"
            ]
        )
        <= 1.0
    ),
    float(
        section5b_summary[
            "overall_masking_change_rate"
        ]
    ),
    "Between 0 and 1",
)

add_section7_check(
    "fallback_rate_valid",
    (
        0.0
        <= float(
            section5b_summary[
                "overall_fallback_rate"
            ]
        )
        <= 1.0
    ),
    float(
        section5b_summary[
            "overall_fallback_rate"
        ]
    ),
    "Between 0 and 1",
)

add_section7_check(
    "dashboard_complete",
    (
        section6a_summary[
            "status"
        ]
        == "TOURNAMENT_PERFORMANCE_DASHBOARD_COMPLETE"
    ),
    section6a_summary[
        "status"
    ],
    "TOURNAMENT_PERFORMANCE_DASHBOARD_COMPLETE",
)

add_section7_check(
    "benchmark_validated",
    (
        section6b_summary[
            "status"
        ]
        == "TOURNAMENT_BENCHMARK_VALIDATED"
    ),
    section6b_summary[
        "status"
    ],
    "TOURNAMENT_BENCHMARK_VALIDATED",
)

add_section7_check(
    "benchmark_ready",
    section6b_summary[
        "benchmark_ready"
    ] is True,
    section6b_summary[
        "benchmark_ready"
    ],
    True,
)

add_section7_check(
    "section6b_checks_all_passed",
    int(
        section6b_summary[
            "validation_checks_failed"
        ]
    ) == 0,
    int(
        section6b_summary[
            "validation_checks_failed"
        ]
    ),
    0,
)

add_section7_check(
    "all_policy_classes_used",
    set(
        action_dashboard[
            "selected_move"
        ]
    )
    == set(
        policy_label_encoder.classes_
    ),
    sorted(
        action_dashboard[
            "selected_move"
        ].tolist()
    ),
    sorted(
        policy_label_encoder.classes_.tolist()
    ),
)

add_section7_check(
    "confidence_profile_valid",
    (
        float(
            section6a_summary[
                "average_confidence"
            ]
        )
        >= 0.0
        and float(
            section6a_summary[
                "average_confidence"
            ]
        )
        <= 1.0
    ),
    float(
        section6a_summary[
            "average_confidence"
        ]
    ),
    "Between 0 and 1",
)

add_section7_check(
    "optimization_signal_present",
    float(
        section6a_summary[
            "masking_change_rate"
        ]
    ) > 0.0,
    float(
        section6a_summary[
            "masking_change_rate"
        ]
    ),
    "> 0",
    "A measurable masking-dependence signal is available for Notebook 53.",
)


section7_validation_checks_df = pd.DataFrame(
    section7_validation_rows
)


print()
print("FINAL NOTEBOOK 52 VALIDATION CHECKS")
print("-" * 100)

display(
    section7_validation_checks_df
)


# --------------------------------------------------------------------------------------
# 4. Optimization-target profile
# --------------------------------------------------------------------------------------

lowest_card_row = (
    leaderboard
    .sort_values(
        [
            "WinRate",
            "Wins",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .iloc[0]
)

lowest_condition_row = (
    condition_leaderboard
    .sort_values(
        [
            "PlayerWinRate",
            "OpponentWinRate",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .iloc[0]
)


section7_optimization_targets = {
    "primary_target":
        "Reduce legal-action masking dependence",

    "masking_change_rate":
        float(
            section6a_summary[
                "masking_change_rate"
            ]
        ),

    "fallback_rate":
        float(
            section6a_summary[
                "fallback_rate"
            ]
        ),

    "weakest_player_card":
        str(
            lowest_card_row[
                "player_card"
            ]
        ),

    "weakest_player_card_win_rate":
        float(
            lowest_card_row[
                "WinRate"
            ]
        ),

    "weakest_condition":
        str(
            lowest_condition_row[
                "condition_id"
            ]
        ),

    "weakest_condition_player_win_rate":
        float(
            lowest_condition_row[
                "PlayerWinRate"
            ]
        ),

    "secondary_targets": [
        "Improve raw policy legality before masking",
        "Improve HP-pressure decisions",
        "Improve Eevee and Meowth outcomes",
        "Reduce zero-damage and turn-limit draws",
        "Compare optimized policy against this frozen benchmark",
    ],
}


section7_optimization_profile_df = pd.DataFrame(
    [
        {
            "metric":
                key,

            "value":
                (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                    if isinstance(
                        value,
                        list,
                    )
                    else value
                ),
        }
        for key, value in (
            section7_optimization_targets.items()
        )
    ]
)


print()
print("NOTEBOOK 53 OPTIMIZATION TARGETS")
print("-" * 100)

display(
    section7_optimization_profile_df
)


# --------------------------------------------------------------------------------------
# 5. Final benchmark profile
# --------------------------------------------------------------------------------------

section7_final_profile_df = pd.DataFrame(
    [
        {
            "metric": "base_scenarios",
            "value": 8,
        },
        {
            "metric": "expanded_scenarios",
            "value": 24,
        },
        {
            "metric": "player_wins",
            "value": int(
                section5b_summary[
                    "player_wins"
                ]
            ),
        },
        {
            "metric": "opponent_wins",
            "value": int(
                section5b_summary[
                    "opponent_wins"
                ]
            ),
        },
        {
            "metric": "draws",
            "value": int(
                section5b_summary[
                    "draws"
                ]
            ),
        },
        {
            "metric": "policy_decisions",
            "value": int(
                section5b_summary[
                    "total_policy_decisions"
                ]
            ),
        },
        {
            "metric": "average_turns",
            "value": float(
                section5b_summary[
                    "average_turns"
                ]
            ),
        },
        {
            "metric": "starter_win_rate",
            "value": float(
                section5b_summary[
                    "starter_win_rate"
                ]
            ),
        },
        {
            "metric": "average_confidence",
            "value": float(
                section5b_summary[
                    "average_confidence"
                ]
            ),
        },
        {
            "metric": "masking_change_rate",
            "value": float(
                section5b_summary[
                    "overall_masking_change_rate"
                ]
            ),
        },
        {
            "metric": "fallback_rate",
            "value": float(
                section5b_summary[
                    "overall_fallback_rate"
                ]
            ),
        },
        {
            "metric": "top_player_card",
            "value": str(
                section6a_summary[
                    "top_player_card"
                ]
            ),
        },
        {
            "metric": "top_player_card_win_rate",
            "value": float(
                section6a_summary[
                    "top_player_card_win_rate"
                ]
            ),
        },
        {
            "metric": "best_condition",
            "value": str(
                section6a_summary[
                    "best_condition"
                ]
            ),
        },
        {
            "metric": "best_condition_player_win_rate",
            "value": float(
                section6a_summary[
                    "best_condition_player_win_rate"
                ]
            ),
        },
        {
            "metric": "section6b_validation_checks_passed",
            "value": int(
                section6b_summary[
                    "validation_checks_passed"
                ]
            ),
        },
        {
            "metric": "section6b_validation_checks_failed",
            "value": int(
                section6b_summary[
                    "validation_checks_failed"
                ]
            ),
        },
    ]
)


print()
print("FINAL NOTEBOOK 52 PROFILE")
print("-" * 100)

display(
    section7_final_profile_df
)


# --------------------------------------------------------------------------------------
# 6. Save final handoff artifacts
# --------------------------------------------------------------------------------------

SECTION7_ARTIFACT_INVENTORY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_artifact_inventory.csv"
)

SECTION7_VALIDATION_CHECKS_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_validation_checks.csv"
)

SECTION7_FINAL_PROFILE_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_tournament_profile.csv"
)

SECTION7_OPTIMIZATION_PROFILE_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_optimization_targets.csv"
)

SECTION7_HANDOFF_MANIFEST_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_handoff_manifest.json"
)

SECTION7_FINAL_SUMMARY_FILE = (
    SECTION7_REPORT_DIR
    / "section7a_final_summary.json"
)


section7_artifact_inventory_df.to_csv(
    SECTION7_ARTIFACT_INVENTORY_FILE,
    index=False,
)

section7_validation_checks_df.to_csv(
    SECTION7_VALIDATION_CHECKS_FILE,
    index=False,
)

section7_final_profile_df.to_csv(
    SECTION7_FINAL_PROFILE_FILE,
    index=False,
)

section7_optimization_profile_df.to_csv(
    SECTION7_OPTIMIZATION_PROFILE_FILE,
    index=False,
)


section7_checks_passed = int(
    section7_validation_checks_df[
        "passed"
    ].astype(bool).sum()
)

section7_checks_failed = int(
    (
        ~section7_validation_checks_df[
            "passed"
        ].astype(bool)
    ).sum()
)


notebook52_handoff_manifest = {
    "notebook":
        "52_tournament_policy_evaluation",

    "status":
        (
            "READY_FOR_POLICY_OPTIMIZATION"
            if section7_checks_failed == 0
            else "FINAL_VALIDATION_FAILED"
        ),

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
        "5C",
        "6A",
        "6B",
        "7A",
    ],

    "source_policy_notebook":
        "50_side_balanced_policy_fine_tuning",

    "source_integration_notebook":
        "51_policy_simulator_integration",

    "base_scenario_count":
        8,

    "expanded_scenario_count":
        24,

    "condition_count":
        3,

    "battle_count":
        int(
            section5b_summary[
                "battle_count"
            ]
        ),

    "player_wins":
        int(
            section5b_summary[
                "player_wins"
            ]
        ),

    "opponent_wins":
        int(
            section5b_summary[
                "opponent_wins"
            ]
        ),

    "draws":
        int(
            section5b_summary[
                "draws"
            ]
        ),

    "policy_decisions":
        int(
            section5b_summary[
                "total_policy_decisions"
            ]
        ),

    "starter_win_rate":
        float(
            section5b_summary[
                "starter_win_rate"
            ]
        ),

    "average_turns":
        float(
            section5b_summary[
                "average_turns"
            ]
        ),

    "average_confidence":
        float(
            section5b_summary[
                "average_confidence"
            ]
        ),

    "masking_changes":
        int(
            section5b_summary[
                "total_masking_changes"
            ]
        ),

    "masking_change_rate":
        float(
            section5b_summary[
                "overall_masking_change_rate"
            ]
        ),

    "fallback_decisions":
        int(
            section5b_summary[
                "total_fallback_decisions"
            ]
        ),

    "fallback_rate":
        float(
            section5b_summary[
                "overall_fallback_rate"
            ]
        ),

    "top_player_card":
        str(
            section6a_summary[
                "top_player_card"
            ]
        ),

    "top_player_card_win_rate":
        float(
            section6a_summary[
                "top_player_card_win_rate"
            ]
        ),

    "best_condition":
        str(
            section6a_summary[
                "best_condition"
            ]
        ),

    "best_condition_player_win_rate":
        float(
            section6a_summary[
                "best_condition_player_win_rate"
            ]
        ),

    "primary_optimization_target":
        section7_optimization_targets[
            "primary_target"
        ],

    "weakest_player_card":
        section7_optimization_targets[
            "weakest_player_card"
        ],

    "weakest_condition":
        section7_optimization_targets[
            "weakest_condition"
        ],

    "validation_checks_passed":
        section7_checks_passed,

    "validation_checks_failed":
        section7_checks_failed,

    "benchmark_profile_file":
        str(
            SECTION6B_BENCHMARK_PROFILE_FILE
        ),

    "expanded_battle_results_file":
        str(
            SECTION5B_BATTLE_RESULTS_FILE
        ),

    "expanded_policy_history_file":
        str(
            SECTION5B_POLICY_HISTORY_FILE
        ),

    "optimization_targets_file":
        str(
            SECTION7_OPTIMIZATION_PROFILE_FILE
        ),

    "next_notebook":
        "Notebook 53 — Legality-Aware Policy Optimization",

    "next_stage":
        "LEGALITY_AWARE_POLICY_OPTIMIZATION",
}


with open(
    SECTION7_HANDOFF_MANIFEST_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        notebook52_handoff_manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )


section7_final_summary = {
    "final_status":
        notebook52_handoff_manifest[
            "status"
        ],

    "battle_count":
        notebook52_handoff_manifest[
            "battle_count"
        ],

    "policy_decisions":
        notebook52_handoff_manifest[
            "policy_decisions"
        ],

    "masking_change_rate":
        notebook52_handoff_manifest[
            "masking_change_rate"
        ],

    "fallback_rate":
        notebook52_handoff_manifest[
            "fallback_rate"
        ],

    "validation_checks_passed":
        notebook52_handoff_manifest[
            "validation_checks_passed"
        ],

    "validation_checks_failed":
        notebook52_handoff_manifest[
            "validation_checks_failed"
        ],

    "next_notebook":
        notebook52_handoff_manifest[
            "next_notebook"
        ],

    "next_stage":
        notebook52_handoff_manifest[
            "next_stage"
        ],
}


with open(
    SECTION7_FINAL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        section7_final_summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------------------
# 7. Final assertions
# --------------------------------------------------------------------------------------

assert section7_checks_failed == 0, (
    "One or more final Notebook 52 validation checks failed."
)

assert (
    notebook52_handoff_manifest[
        "status"
    ]
    == "READY_FOR_POLICY_OPTIMIZATION"
)

assert (
    section7_final_summary[
        "final_status"
    ]
    == "READY_FOR_POLICY_OPTIMIZATION"
)


required_section7_files = [
    SECTION7_ARTIFACT_INVENTORY_FILE,
    SECTION7_VALIDATION_CHECKS_FILE,
    SECTION7_FINAL_PROFILE_FILE,
    SECTION7_OPTIMIZATION_PROFILE_FILE,
    SECTION7_HANDOFF_MANIFEST_FILE,
    SECTION7_FINAL_SUMMARY_FILE,
]

assert all(
    file_path.exists()
    and file_path.stat().st_size > 0
    for file_path in required_section7_files
)


# --------------------------------------------------------------------------------------
# 8. Final output
# --------------------------------------------------------------------------------------

print()
print("NOTEBOOK 52 HANDOFF SUMMARY")
print("-" * 100)

print(
    f"Final status               : "
    f"{notebook52_handoff_manifest['status']}"
)

print(
    f"Expanded battles           : "
    f"{notebook52_handoff_manifest['battle_count']}"
)

print(
    f"Policy decisions           : "
    f"{notebook52_handoff_manifest['policy_decisions']}"
)

print(
    f"Player wins                : "
    f"{notebook52_handoff_manifest['player_wins']}"
)

print(
    f"Opponent wins              : "
    f"{notebook52_handoff_manifest['opponent_wins']}"
)

print(
    f"Draws                      : "
    f"{notebook52_handoff_manifest['draws']}"
)

print(
    f"Masking change rate        : "
    f"{notebook52_handoff_manifest['masking_change_rate']:.4%}"
)

print(
    f"Fallback rate              : "
    f"{notebook52_handoff_manifest['fallback_rate']:.4%}"
)

print(
    f"Primary optimization target: "
    f"{notebook52_handoff_manifest['primary_optimization_target']}"
)

print(
    f"Weakest player card        : "
    f"{notebook52_handoff_manifest['weakest_player_card']}"
)

print(
    f"Weakest condition          : "
    f"{notebook52_handoff_manifest['weakest_condition']}"
)

print(
    f"Validation checks passed   : "
    f"{notebook52_handoff_manifest['validation_checks_passed']}"
)

print(
    f"Validation checks failed   : "
    f"{notebook52_handoff_manifest['validation_checks_failed']}"
)

print(
    f"Next notebook              : "
    f"{notebook52_handoff_manifest['next_notebook']}"
)

print(
    f"Next stage                 : "
    f"{notebook52_handoff_manifest['next_stage']}"
)


print()
print("SAVED SECTION 7A REPORTS")
print("-" * 100)

for file_path in required_section7_files:

    print(
        file_path
    )


print()
print(
    "✅ SECTION 7A FINAL VALIDATION AND HANDOFF PASSED"
)

print()
print(
    "🎉 NOTEBOOK 52 COMPLETE — READY FOR NOTEBOOK 53"
)


# In[ ]:




