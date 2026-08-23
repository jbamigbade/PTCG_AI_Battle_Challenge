#!/usr/bin/env python
# coding: utf-8

# # Notebook 59 — Live Competitive Evaluation
# 
# ## Pokémon TCG AI Battle Challenge
# 
# ### Primary Policy
# 
# **R3_STATE_CENTRIC_POLICY**
# 
# ### Purpose
# 
# Notebook 59 evaluates the frozen R3 production candidate in the actual project battle environment.
# 
# Notebook 58 established that R3:
# 
# - uses 46 retained features,
# - reduces the R0 feature set by 28.125%,
# - reproduces frozen predictions exactly,
# - reproduces probabilities exactly,
# - preserves all controlled action decisions,
# - passes the controlled-validation production gate.
# 
# Notebook 59 now moves beyond controlled classification validation.
# 
# ### Primary question
# 
# > How does R3 perform when it must actually play battles?
# 
# ### Evaluation priorities
# 
# - battle execution success
# - legal-action integrity
# - win rate
# - first-player performance
# - second-player performance
# - opponent-pool performance
# - runtime stability
# - reproducibility
# 
# No new policy family will be trained in this notebook.

# In[2]:


# ==========================================================================================
# SECTION 1A — ENVIRONMENT AND PROJECT PATHS
# ==========================================================================================

from pathlib import Path
import json
import pickle
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 100)
print("SECTION 1A — ENVIRONMENT AND PROJECT PATHS")
print("=" * 100)

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
)

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

ARTIFACTS_58_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook58"
)

ARTIFACTS_59_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook59"
)

REPORTS_59_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook59"
)

BACKUPS_59_DIR = (
    PROJECT_ROOT
    / "backups"
    / "notebook59"
)

MODEL_55_DIR = (
    PROJECT_ROOT
    / "models"
    / "notebook55"
)

R3_MODEL_PATH = (
    MODEL_55_DIR
    / "section3a_r3_state_centric_policy_model.joblib"
)

NOTEBOOK58_RESULTS_PATH = (
    ARTIFACTS_58_DIR
    / "notebook58_final_competitive_validation_results.pkl"
)

# Create NB59 output directories
ARTIFACTS_59_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_59_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BACKUPS_59_DIR.mkdir(
    parents=True,
    exist_ok=True
)

path_records = [
    {
        "resource": "PROJECT_ROOT",
        "path": str(PROJECT_ROOT),
        "exists": PROJECT_ROOT.exists(),
    },
    {
        "resource": "SRC_DIR",
        "path": str(SRC_DIR),
        "exists": SRC_DIR.exists(),
    },
    {
        "resource": "NOTEBOOK58_RESULTS",
        "path": str(NOTEBOOK58_RESULTS_PATH),
        "exists": NOTEBOOK58_RESULTS_PATH.exists(),
    },
    {
        "resource": "R3_MODEL",
        "path": str(R3_MODEL_PATH),
        "exists": R3_MODEL_PATH.exists(),
    },
    {
        "resource": "ARTIFACTS_59",
        "path": str(ARTIFACTS_59_DIR),
        "exists": ARTIFACTS_59_DIR.exists(),
    },
    {
        "resource": "REPORTS_59",
        "path": str(REPORTS_59_DIR),
        "exists": REPORTS_59_DIR.exists(),
    },
]

section1a_paths_df = pd.DataFrame(
    path_records
)

display(section1a_paths_df)

assert PROJECT_ROOT.exists()
assert SRC_DIR.exists()
assert NOTEBOOK58_RESULTS_PATH.exists()
assert R3_MODEL_PATH.exists()

print("\nPython version:")
print(sys.version)

print("\n✅ NOTEBOOK 59 DIRECTORIES READY")
print("✅ R3 FROZEN MODEL FOUND")
print("✅ NOTEBOOK 58 RESULT PACKAGE FOUND")
print("✅ SECTION 1A COMPLETE")


# In[3]:


# ==========================================================================================
# SECTION 1B — LOAD APPROVED R3 PRODUCTION CANDIDATE
# ==========================================================================================

print("=" * 100)
print("SECTION 1B — LOAD APPROVED R3 PRODUCTION CANDIDATE")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Load Notebook 58 results
# ------------------------------------------------------------------------------------------

with open(
    NOTEBOOK58_RESULTS_PATH,
    "rb",
) as file:

    notebook58_results = pickle.load(file)

print("\nNOTEBOOK 58 RESULT KEYS")
print("-" * 100)

for key in notebook58_results.keys():
    print(key)

# ------------------------------------------------------------------------------------------
# Validate release decision
# ------------------------------------------------------------------------------------------

release_decision = notebook58_results[
    "release_decision"
]

production_policy = notebook58_results[
    "production_policy"
]

print("\nUPSTREAM RELEASE DECISION")
print("-" * 100)

print(
    f"Release decision : {release_decision}"
)

print(
    f"Production policy: {production_policy}"
)

assert release_decision == "GO", (
    "Notebook 58 did not approve release."
)

assert production_policy == (
    "R3_STATE_CENTRIC_POLICY"
), (
    "Notebook 58 production policy is not R3."
)

# ------------------------------------------------------------------------------------------
# Load frozen R3 package
# ------------------------------------------------------------------------------------------

r3_package = joblib.load(
    R3_MODEL_PATH
)

assert isinstance(
    r3_package,
    dict,
), (
    "Expected R3 model artifact to be a dictionary package."
)

r3_model = r3_package.get(
    "model"
)

r3_feature_names = r3_package.get(
    "retained_feature_names"
)

assert r3_model is not None

assert r3_feature_names is not None

assert len(
    r3_feature_names
) == 46

print("\nR3 PACKAGE VALIDATION")
print("-" * 100)

print(
    f"Estimator type : "
    f"{type(r3_model).__name__}"
)

print(
    f"Feature count  : "
    f"{len(r3_feature_names)}"
)

print(
    f"Class count    : "
    f"{len(r3_model.classes_)}"
)

print(
    f"Classes        : "
    f"{list(r3_model.classes_)}"
)

print("\n✅ NOTEBOOK 58 RELEASE DECISION VERIFIED")
print("✅ R3 PRODUCTION PACKAGE LOADED")
print("✅ R3 46-FEATURE STRUCTURE VERIFIED")
print("✅ SECTION 1B COMPLETE")


# In[4]:


# ==========================================================================================
# SECTION 1C — DISCOVER CURRENT BATTLE INFRASTRUCTURE
# ==========================================================================================

print("=" * 100)
print("SECTION 1C — DISCOVER CURRENT BATTLE INFRASTRUCTURE")
print("=" * 100)

candidate_directories = [
    SRC_DIR / "engine",
    SRC_DIR / "tournament",
    SRC_DIR / "full_game_simulation",
    SRC_DIR / "evaluation_harness",
    SRC_DIR / "production_agent_benchmark",
    SRC_DIR / "policy_engine",
    SRC_DIR / "agents",
    SRC_DIR / "competitive",
    SRC_DIR / "kaggle_agent",
]

directory_records = []

for directory in candidate_directories:

    python_files = []

    if directory.exists():

        python_files = sorted(
            [
                file.name
                for file in directory.rglob("*.py")
                if "__pycache__"
                not in file.parts
            ]
        )

    directory_records.append(
        {
            "component": directory.name,
            "path": str(directory),
            "exists": directory.exists(),
            "python_file_count": len(
                python_files
            ),
            "python_files": python_files,
        }
    )

section1c_battle_infrastructure_df = pd.DataFrame(
    directory_records
)

display(
    section1c_battle_infrastructure_df
)

print("\nTOP-LEVEL BATTLE-RELATED PYTHON FILES")
print("-" * 100)

battle_keywords = [
    "battle",
    "simulat",
    "tournament",
    "agent",
    "evaluation",
    "competitive",
]

top_level_battle_files = []

for file in SRC_DIR.glob("*.py"):

    lower_name = file.name.lower()

    if any(
        keyword in lower_name
        for keyword in battle_keywords
    ):

        top_level_battle_files.append(
            {
                "filename": file.name,
                "path": str(file),
                "size_bytes": file.stat().st_size,
            }
        )

section1c_top_level_files_df = pd.DataFrame(
    top_level_battle_files
)

display(
    section1c_top_level_files_df
)

print("\n✅ BATTLE INFRASTRUCTURE INVENTORIED")
print("✅ SECTION 1C COMPLETE")


# In[5]:


# ==========================================================================================
# SECTION 2A — DISCOVER LIVE EVALUATION ENTRY POINTS
# ==========================================================================================

print("=" * 100)
print("SECTION 2A — DISCOVER LIVE EVALUATION ENTRY POINTS")
print("=" * 100)

import ast

candidate_files = [
    SRC_DIR / "battle_simulation.py",
    SRC_DIR / "simulator.py",
    SRC_DIR / "battle_agent.py",
    SRC_DIR / "evaluation.py",
    SRC_DIR / "engine" / "benchmark.py",
    SRC_DIR / "tournament" / "manager.py",
    SRC_DIR / "competitive" / "competitive_policy_agent.py",
]

entry_records = []

for file_path in candidate_files:

    if not file_path.exists():
        entry_records.append(
            {
                "file": str(file_path),
                "exists": False,
                "classes": [],
                "functions": [],
            }
        )
        continue

    source = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    classes = []
    functions = []

    for node in tree.body:

        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            functions.append(node.name)

    entry_records.append(
        {
            "file": str(file_path),
            "exists": True,
            "classes": classes,
            "functions": functions,
        }
    )

section2a_entry_points_df = pd.DataFrame(
    entry_records
)

display(section2a_entry_points_df)

print("\nDETAILED ENTRY POINTS")
print("-" * 100)

for _, row in section2a_entry_points_df.iterrows():

    print("\nFILE")
    print(row["file"])

    print("Classes:")
    print(row["classes"])

    print("Functions:")
    print(row["functions"])

print("\n✅ LIVE EVALUATION ENTRY POINTS DISCOVERED")
print("✅ SECTION 2A COMPLETE")


# In[6]:


# ==========================================================================================
# SECTION 2B — INSPECT LIVE EVALUATION INTERFACES
# ==========================================================================================

print("=" * 100)
print("SECTION 2B — INSPECT LIVE EVALUATION INTERFACES")
print("=" * 100)

import ast

target_files = [
    SRC_DIR / "battle_simulation.py",
    SRC_DIR / "battle_agent.py",
    SRC_DIR / "tournament" / "manager.py",
    SRC_DIR / "competitive" / "competitive_policy_agent.py",
]

records = []

for file_path in target_files:

    if not file_path.exists():
        continue

    source = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    for node in tree.body:

        if isinstance(node, ast.ClassDef):

            methods = []

            for child in node.body:

                if isinstance(child, ast.FunctionDef):

                    methods.append(child.name)

            records.append(
                {
                    "type": "CLASS",
                    "file": file_path.name,
                    "name": node.name,
                    "methods": methods,
                }
            )

        elif isinstance(node, ast.FunctionDef):

            args = [
                arg.arg
                for arg in node.args.args
            ]

            records.append(
                {
                    "type": "FUNCTION",
                    "file": file_path.name,
                    "name": node.name,
                    "methods": args,
                }
            )

section2b_interface_df = pd.DataFrame(records)

display(section2b_interface_df)

print()
print("KEY INTERFACES")
print("-" * 100)

for _, row in section2b_interface_df.iterrows():

    print(f"{row['type']}: {row['name']}")
    print(f"File: {row['file']}")
    print(f"Methods / Args:")
    print(row["methods"])
    print()

print("✅ LIVE INTERFACES DISCOVERED")
print("✅ SECTION 2B COMPLETE")


# In[7]:


# ==========================================================================================
# SECTION 3A — INSPECT COMPETITIVE AGENT CONSTRUCTOR
# ==========================================================================================

print("=" * 100)
print("SECTION 3A — INSPECT COMPETITIVE AGENT CONSTRUCTOR")
print("=" * 100)

import ast

inspect_classes = {
    "competitive_policy_agent.py": [
        "CompetitivePolicyAgent",
        "CompetitivePolicyDecision",
    ],
    "battle_agent.py": [
        "PokemonBattleAgent",
    ],
}

records = []

for filename, wanted_classes in inspect_classes.items():

    file_path = None

    if filename == "competitive_policy_agent.py":
        file_path = (
            SRC_DIR
            / "competitive"
            / filename
        )

    else:
        file_path = SRC_DIR / filename

    source = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    for node in tree.body:

        if not isinstance(node, ast.ClassDef):
            continue

        if node.name not in wanted_classes:
            continue

        constructor = None

        methods = []

        for child in node.body:

            if isinstance(child, ast.FunctionDef):

                methods.append(child.name)

                if child.name == "__init__":

                    constructor = [
                        arg.arg
                        for arg in child.args.args
                    ]

        records.append(
            {
                "class": node.name,
                "constructor_args": constructor,
                "methods": methods,
            }
        )

section3a_constructor_df = pd.DataFrame(records)

display(section3a_constructor_df)

print()

for _, row in section3a_constructor_df.iterrows():

    print("=" * 80)
    print(row["class"])
    print("=" * 80)

    print("Constructor:")
    print(row["constructor_args"])

    print()

    print("Methods:")
    print(row["methods"])

print()
print("✅ COMPETITIVE AGENT CONSTRUCTOR DISCOVERED")
print("✅ SECTION 3A COMPLETE")


# In[8]:


# ==========================================================================================
# SECTION 3B — INSPECT COMPETITIVE POLICY AGENT INITIALIZATION
# ==========================================================================================

print("=" * 100)
print("SECTION 3B — INSPECT COMPETITIVE POLICY AGENT INITIALIZATION")
print("=" * 100)

from pathlib import Path
import ast

agent_file = (
    SRC_DIR
    / "competitive"
    / "competitive_policy_agent.py"
)

source = agent_file.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source)

for node in tree.body:

    if isinstance(node, ast.ClassDef) and node.name == "CompetitivePolicyAgent":

        print("CLASS FOUND:")
        print(node.name)
        print()

        for child in node.body:

            if (
                isinstance(child, ast.FunctionDef)
                and child.name == "__init__"
            ):

                print("__init__ source")
                print("-" * 100)

                init_source = ast.get_source_segment(
                    source,
                    child,
                )

                print(init_source)

print()
print("✅ CONSTRUCTOR SOURCE DISPLAYED")
print("✅ SECTION 3B COMPLETE")


# In[10]:


# ==========================================================================================
# SECTION 3C — LOCATE COMPETITIVE POLICY ENGINE SOURCE
# ==========================================================================================

print("=" * 100)
print("SECTION 3C — LOCATE COMPETITIVE POLICY ENGINE SOURCE")
print("=" * 100)

from pathlib import Path

search_term = "CompetitivePolicyEngine"

matches = []

for file_path in SRC_DIR.rglob("*.py"):

    if "__pycache__" in file_path.parts:
        continue

    try:
        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    if search_term in text:

        matching_lines = []

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            if search_term in line:

                matching_lines.append(
                    {
                        "line_number": line_number,
                        "line": line.strip(),
                    }
                )

        matches.append(
            {
                "file": str(file_path),
                "filename": file_path.name,
                "match_count": len(
                    matching_lines
                ),
                "matching_lines": matching_lines,
            }
        )

section3c_engine_location_df = pd.DataFrame(
    matches
)

print("\nCOMPETITIVE POLICY ENGINE SEARCH RESULTS")
print("-" * 100)

display(
    section3c_engine_location_df
)

print("\nDETAILED MATCHES")
print("-" * 100)

for match in matches:

    print("\nFILE:")
    print(match["file"])

    for item in match["matching_lines"]:

        print(
            f"  line {item['line_number']}: "
            f"{item['line']}"
        )

assert len(matches) > 0, (
    "CompetitivePolicyEngine was not found "
    "anywhere under src."
)

print("\n✅ COMPETITIVE POLICY ENGINE SOURCE LOCATED")
print("✅ SECTION 3C COMPLETE")


# In[11]:


# ==========================================================================================
# SECTION 3D — INSPECT COMPETITIVE POLICY ENGINE IMPLEMENTATION
# ==========================================================================================

print("=" * 100)
print("SECTION 3D — INSPECT COMPETITIVE POLICY ENGINE IMPLEMENTATION")
print("=" * 100)

engine_file = (
    SRC_DIR
    / "competitive"
    / "competitive_policy_engine.py"
)

assert engine_file.exists(), (
    f"Competitive policy engine file not found:\n{engine_file}"
)

source = engine_file.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source)

found = False

for node in tree.body:

    if (
        isinstance(node, ast.ClassDef)
        and node.name == "CompetitivePolicyEngine"
    ):

        found = True

        print("CLASS FOUND:")
        print(node.name)
        print()

        for child in node.body:

            if not isinstance(child, ast.FunctionDef):
                continue

            if child.name in [
                "__init__",
                "choose_move",
                "select_move",
                "predict",
                "act",
                "load_checkpoint",
                "_load_checkpoint",
            ]:

                print("=" * 100)
                print(f"METHOD: {child.name}")
                print("=" * 100)

                method_source = ast.get_source_segment(
                    source,
                    child,
                )

                print(method_source)
                print()

assert found, (
    "CompetitivePolicyEngine class was not found."
)

print("✅ COMPETITIVE POLICY ENGINE IMPLEMENTATION INSPECTED")
print("✅ SECTION 3D COMPLETE")


# In[13]:


# ==========================================================================================
# SECTION 3E — INSPECT REQUIRED LIVE AGENT INTERFACE
# ==========================================================================================

print("=" * 100)
print("SECTION 3E — INSPECT REQUIRED LIVE AGENT INTERFACE")
print("=" * 100)

targets = [
    (
        SRC_DIR / "battle_agent.py",
        "PokemonBattleAgent",
        ["__init__", "choose_move"],
    ),
    (
        SRC_DIR / "competitive" / "competitive_policy_agent.py",
        "CompetitivePolicyAgent",
        [
            "__init__",
            "choose_move",
            "select_move",
            "act",
            "get_last_decision",
        ],
    ),
]

interface_records = []

for file_path, class_name, method_names in targets:

    assert file_path.exists(), (
        f"Missing source file:\n{file_path}"
    )

    source = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    for node in tree.body:

        if not (
            isinstance(node, ast.ClassDef)
            and node.name == class_name
        ):
            continue

        for child in node.body:

            if not (
                isinstance(child, ast.FunctionDef)
                and child.name in method_names
            ):
                continue

            arguments = [
                arg.arg
                for arg in child.args.args
            ]

            method_source = ast.get_source_segment(
                source,
                child,
            )

            interface_records.append(
                {
                    "class_name": class_name,
                    "method_name": child.name,
                    "arguments": arguments,
                    "source": method_source,
                }
            )

section3e_required_interface_df = pd.DataFrame(
    interface_records
)

print("\nLIVE AGENT METHOD SIGNATURES")
print("-" * 100)

display(
    section3e_required_interface_df[
        [
            "class_name",
            "method_name",
            "arguments",
        ]
    ]
)

print("\nDETAILED METHOD IMPLEMENTATIONS")
print("-" * 100)

for _, row in section3e_required_interface_df.iterrows():

    print("\n" + "=" * 90)
    print(
        f"{row['class_name']}.{row['method_name']}"
    )
    print("=" * 90)

    print(row["source"])

print("\n✅ REQUIRED LIVE AGENT INTERFACE INSPECTED")
print("✅ SECTION 3E COMPLETE")


# In[19]:


# ==========================================================================================
# SECTION 3F — LOCATE LIVE STATE-TO-FEATURE PIPELINE
# ROBUST PROJECT-WIDE SEARCH
# ==========================================================================================

print("=" * 100)
print("SECTION 3F — LOCATE LIVE STATE-TO-FEATURE PIPELINE")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Search roots
# ------------------------------------------------------------------------------------------

SEARCH_ROOTS = [
    SRC_DIR,
    PROJECT_ROOT / "scripts",
]

# ------------------------------------------------------------------------------------------
# High-value search terms
# ------------------------------------------------------------------------------------------

search_terms = [
    "retained_feature_names",
    "decision_features",
    "feature_vector",
    "feature_matrix",
    "build_features",
    "extract_features",
    "transform_features",
    "encode_features",
    "observation_adapter",
    "damage_difference",
    "acting_damage",
    "multi_legal_move_flag",
    "recomputed_legal_move_count",
    "model_choice_required_flag",
    "single_legal_move_flag",
    "absolute_damage_difference",
    "turn_number",
    "hand_size",
    "player_damage",
    "player_energy",
    "opponent_energy",
    "defending_damage",
    "defending_energy",
]

feature_search_records = []

for search_root in SEARCH_ROOTS:

    if not search_root.exists():
        continue

    for file_path in search_root.rglob("*.py"):

        if "__pycache__" in file_path.parts:
            continue

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        text_lower = text.lower()

        matched_terms = []
        matching_lines = []

        lines = text.splitlines()

        for term in search_terms:

            term_lower = term.lower()

            if term_lower not in text_lower:
                continue

            matched_terms.append(term)

            for line_number, line in enumerate(
                lines,
                start=1,
            ):

                if term_lower in line.lower():

                    matching_lines.append(
                        {
                            "term": term,
                            "line_number": line_number,
                            "line": line.strip(),
                        }
                    )

        if matched_terms:

            feature_search_records.append(
                {
                    "filename": file_path.name,
                    "file": str(file_path),
                    "search_root": str(search_root),
                    "matched_terms": sorted(
                        set(matched_terms)
                    ),
                    "match_count": len(
                        matching_lines
                    ),
                    "matching_lines": matching_lines,
                }
            )

# ------------------------------------------------------------------------------------------
# Construct robust dataframe
# ------------------------------------------------------------------------------------------

expected_columns = [
    "filename",
    "file",
    "search_root",
    "matched_terms",
    "match_count",
    "matching_lines",
]

section3f_feature_search_df = pd.DataFrame(
    feature_search_records,
    columns=expected_columns,
)

print("\nSTATE / FEATURE PIPELINE SEARCH RESULTS")
print("-" * 100)

if section3f_feature_search_df.empty:

    print(
        "No direct feature-pipeline matches were found "
        "under src/ or scripts/."
    )

else:

    display(
        section3f_feature_search_df[
            [
                "filename",
                "file",
                "matched_terms",
                "match_count",
            ]
        ]
    )

# ------------------------------------------------------------------------------------------
# Detailed high-value matches
# ------------------------------------------------------------------------------------------

print("\nDETAILED HIGH-VALUE MATCHES")
print("-" * 100)

priority_terms = {
    "retained_feature_names",
    "decision_features",
    "feature_vector",
    "feature_matrix",
    "build_features",
    "extract_features",
    "transform_features",
    "observation_adapter",
    "recomputed_legal_move_count",
    "model_choice_required_flag",
    "single_legal_move_flag",
    "damage_difference",
    "absolute_damage_difference",
}

important_match_count = 0

for record in feature_search_records:

    important = [
        item
        for item in record["matching_lines"]
        if item["term"] in priority_terms
    ]

    if not important:
        continue

    important_match_count += len(important)

    print("\nFILE:")
    print(record["file"])

    for item in important[:40]:

        print(
            f"  line {item['line_number']:>4}: "
            f"[{item['term']}] "
            f"{item['line']}"
        )

if important_match_count == 0:

    print(
        "No high-value live feature-builder references "
        "were located in the searched Python files."
    )

# ------------------------------------------------------------------------------------------
# Search summary
# ------------------------------------------------------------------------------------------

print("\nSEARCH SUMMARY")
print("-" * 100)

print(
    f"Files with matches: "
    f"{len(section3f_feature_search_df)}"
)

print(
    f"High-value matching lines: "
    f"{important_match_count}"
)

print("\n✅ LIVE FEATURE PIPELINE SEARCH COMPLETE")
print("✅ SECTION 3F COMPLETE")


# In[20]:


# ==========================================================================================
# SECTION 4A — LOAD APPROVED R3 LIVE POLICY PACKAGE
# ==========================================================================================

print("=" * 100)
print("SECTION 4A — LOAD APPROVED R3 LIVE POLICY PACKAGE")
print("=" * 100)

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------
# Resolve approved R3 model path
# ------------------------------------------------------------------------------------------

R3_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "notebook55"
    / "section3a_r3_state_centric_policy_model.joblib"
)

print("\nR3 MODEL PATH")
print("-" * 100)
print(R3_MODEL_PATH)

assert R3_MODEL_PATH.exists(), (
    f"Approved R3 model package not found:\n"
    f"{R3_MODEL_PATH}"
)

# ------------------------------------------------------------------------------------------
# Load R3 model package
# ------------------------------------------------------------------------------------------

r3_live_package = joblib.load(
    R3_MODEL_PATH
)

assert isinstance(
    r3_live_package,
    dict,
), (
    "Expected the approved R3 artifact "
    "to be a dictionary model package."
)

print("\nR3 PACKAGE KEYS")
print("-" * 100)

for key in r3_live_package.keys():
    print(key)

# ------------------------------------------------------------------------------------------
# Extract model and metadata
# ------------------------------------------------------------------------------------------

r3_live_model = r3_live_package.get(
    "model"
)

r3_live_feature_names = r3_live_package.get(
    "retained_feature_names"
)

r3_live_feature_indices = r3_live_package.get(
    "retained_feature_indices"
)

r3_live_strategy_id = r3_live_package.get(
    "strategy_id"
)

r3_live_training_results = r3_live_package.get(
    "training_results"
)

# ------------------------------------------------------------------------------------------
# Validate estimator
# ------------------------------------------------------------------------------------------

assert r3_live_model is not None, (
    "R3 package does not contain a model."
)

assert r3_live_feature_names is not None, (
    "R3 package does not contain retained_feature_names."
)

assert len(
    r3_live_feature_names
) == 46, (
    f"Expected 46 R3 features, "
    f"found {len(r3_live_feature_names)}."
)

model_feature_count = getattr(
    r3_live_model,
    "n_features_in_",
    None,
)

assert model_feature_count == 46, (
    f"R3 model expects {model_feature_count} features, "
    "not 46."
)

# ------------------------------------------------------------------------------------------
# Validate class structure
# ------------------------------------------------------------------------------------------

r3_live_classes = np.asarray(
    getattr(
        r3_live_model,
        "classes_",
        [],
    )
)

assert len(r3_live_classes) > 0, (
    "R3 estimator does not expose model classes."
)

# ------------------------------------------------------------------------------------------
# Build validation summary
# ------------------------------------------------------------------------------------------

section4a_r3_live_validation_df = pd.DataFrame(
    [
        {
            "strategy": (
                r3_live_strategy_id
                if r3_live_strategy_id is not None
                else "R3_STATE_CENTRIC_POLICY"
            ),
            "model_type": type(
                r3_live_model
            ).__name__,
            "model_path": str(
                R3_MODEL_PATH
            ),
            "model_exists": (
                R3_MODEL_PATH.exists()
            ),
            "retained_feature_count": len(
                r3_live_feature_names
            ),
            "model_n_features_in": (
                model_feature_count
            ),
            "class_count": len(
                r3_live_classes
            ),
            "classes": list(
                r3_live_classes
            ),
            "training_results_available": (
                isinstance(
                    r3_live_training_results,
                    dict,
                )
            ),
        }
    ]
)

print("\nR3 LIVE POLICY VALIDATION")
print("-" * 100)

display(
    section4a_r3_live_validation_df
)

# ------------------------------------------------------------------------------------------
# Display retained feature order
# ------------------------------------------------------------------------------------------

section4a_r3_feature_order_df = pd.DataFrame(
    {
        "feature_position": np.arange(
            1,
            len(r3_live_feature_names) + 1,
        ),
        "feature_name": (
            r3_live_feature_names
        ),
    }
)

print("\nR3 REQUIRED FEATURE ORDER")
print("-" * 100)

display(
    section4a_r3_feature_order_df
)

# ==========================================================================================
# HARD VALIDATION GATES
# ==========================================================================================

assert (
    section4a_r3_live_validation_df[
        "model_exists"
    ].all()
)

assert (
    section4a_r3_live_validation_df[
        "retained_feature_count"
    ] == 46
).all()

assert (
    section4a_r3_live_validation_df[
        "model_n_features_in"
    ] == 46
).all()

# ------------------------------------------------------------------------------------------
# Important architecture check
# ------------------------------------------------------------------------------------------

print("\nLIVE INTEGRATION ARCHITECTURE")
print("-" * 100)

print(
    "Approved production model : "
    "R3 Random Forest / joblib package"
)

print(
    "Existing CompetitivePolicyAgent : "
    "PyTorch checkpoint interface"
)

print(
    "Integration approach : "
    "R3 live-policy adapter required"
)

print(
    "No retraining or model conversion will be performed."
)

print("\n✅ APPROVED R3 MODEL PACKAGE LOADED")
print("✅ 46-FEATURE ORDER VERIFIED")
print("✅ R3 MODEL CLASSES VERIFIED")
print("✅ LIVE ADAPTER PATH CONFIRMED")
print("✅ SECTION 4A COMPLETE")


# In[21]:


# ==========================================================================================
# SECTION 4B — VALIDATE EXISTING PRODUCTION FEATURE PIPELINE
# ==========================================================================================

print("=" * 100)
print("SECTION 4B — VALIDATE EXISTING PRODUCTION FEATURE PIPELINE")
print("=" * 100)

import ast
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------
# Candidate production feature modules discovered earlier
# ------------------------------------------------------------------------------------------

candidate_feature_files = [
    SRC_DIR / "decision_features" / "decision_features.py",
    SRC_DIR / "decision_features" / "feature_extractor.py",
    SRC_DIR / "decision_features" / "battle_feature_extractor.py",
    SRC_DIR / "observation_adapter" / "battle_state_adapter.py",
    SRC_DIR / "observation_adapter" / "policy_search_adapter.py",
]

# Keep only files that actually exist
existing_feature_files = [
    path
    for path in candidate_feature_files
    if path.exists()
]

print("\nEXISTING FEATURE PIPELINE FILES")
print("-" * 100)

for path in existing_feature_files:
    print(path)

# ------------------------------------------------------------------------------------------
# Also inventory every Python file inside the two production packages
# ------------------------------------------------------------------------------------------

package_files = []

for package_dir in [
    SRC_DIR / "decision_features",
    SRC_DIR / "observation_adapter",
]:

    if not package_dir.exists():
        continue

    for path in sorted(
        package_dir.rglob("*.py")
    ):

        if "__pycache__" in path.parts:
            continue

        package_files.append(path)

print("\nPRODUCTION FEATURE PACKAGE INVENTORY")
print("-" * 100)

for path in package_files:
    print(path)

assert len(package_files) > 0, (
    "No production feature-pipeline Python files were found."
)

# ==========================================================================================
# DISCOVER CLASSES AND FUNCTIONS
# ==========================================================================================

interface_records = []

for file_path in package_files:

    source = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in tree.body:

        if isinstance(node, ast.ClassDef):

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

            interface_records.append(
                {
                    "file": str(file_path),
                    "object_type": "CLASS",
                    "name": node.name,
                    "methods_or_args": methods,
                }
            )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            args = [
                arg.arg
                for arg in node.args.args
            ]

            interface_records.append(
                {
                    "file": str(file_path),
                    "object_type": "FUNCTION",
                    "name": node.name,
                    "methods_or_args": args,
                }
            )

section4b_feature_interfaces_df = pd.DataFrame(
    interface_records
)

print("\nPRODUCTION FEATURE PIPELINE INTERFACES")
print("-" * 100)

display(
    section4b_feature_interfaces_df
)

# ==========================================================================================
# SEARCH FOR FEATURE-BUILDING ENTRY POINTS
# ==========================================================================================

feature_keywords = [
    "feature",
    "extract",
    "build",
    "encode",
    "transform",
    "adapt",
    "observation",
    "state",
]

candidate_entry_points = []

for _, row in section4b_feature_interfaces_df.iterrows():

    name_lower = str(
        row["name"]
    ).lower()

    if any(
        keyword in name_lower
        for keyword in feature_keywords
    ):

        candidate_entry_points.append(
            row.to_dict()
        )

section4b_candidate_entry_points_df = pd.DataFrame(
    candidate_entry_points
)

print("\nCANDIDATE FEATURE / ADAPTER ENTRY POINTS")
print("-" * 100)

if section4b_candidate_entry_points_df.empty:

    print(
        "No function/class names matched the feature keywords."
    )

else:

    display(
        section4b_candidate_entry_points_df
    )

# ==========================================================================================
# VERIFY R3 FEATURE CONTRACT
# ==========================================================================================

r3_feature_contract_df = pd.DataFrame(
    {
        "feature_position": np.arange(
            1,
            len(r3_live_feature_names) + 1,
        ),
        "feature_name": r3_live_feature_names,
    }
)

print("\nR3 LIVE FEATURE CONTRACT")
print("-" * 100)

display(
    r3_feature_contract_df
)

assert len(
    r3_feature_contract_df
) == 46

assert r3_feature_contract_df[
    "feature_name"
].is_unique

# ==========================================================================================
# SEARCH PRODUCTION SOURCE FOR R3 FEATURE REFERENCES
# ==========================================================================================

feature_reference_records = []

for file_path in package_files:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    text_lower = text.lower()

    matched_features = []

    for full_feature_name in r3_live_feature_names:

        # Remove preprocessing prefix:
        # numeric__turn_number -> turn_number
        # categorical__active_card_Eevee -> active_card_Eevee
        raw_feature_name = (
            str(full_feature_name)
            .split("__", 1)[-1]
        )

        if raw_feature_name.lower() in text_lower:

            matched_features.append(
                raw_feature_name
            )

    if matched_features:

        feature_reference_records.append(
            {
                "file": str(file_path),
                "matched_r3_features": sorted(
                    set(matched_features)
                ),
                "matched_feature_count": len(
                    set(matched_features)
                ),
            }
        )

section4b_r3_source_coverage_df = pd.DataFrame(
    feature_reference_records
)

print("\nR3 FEATURE REFERENCES IN PRODUCTION SOURCE")
print("-" * 100)

if section4b_r3_source_coverage_df.empty:

    print(
        "No exact R3 feature references were found "
        "inside the production source packages."
    )

else:

    display(
        section4b_r3_source_coverage_df
        .sort_values(
            "matched_feature_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

# ==========================================================================================
# SUMMARY
# ==========================================================================================

print("\nFEATURE PIPELINE VALIDATION SUMMARY")
print("-" * 100)

print(
    f"Production package files discovered : "
    f"{len(package_files)}"
)

print(
    f"Functions/classes discovered        : "
    f"{len(section4b_feature_interfaces_df)}"
)

print(
    f"Candidate feature entry points      : "
    f"{len(section4b_candidate_entry_points_df)}"
)

print(
    f"R3 required features                : "
    f"{len(r3_live_feature_names)}"
)

print("\n✅ EXISTING PRODUCTION FEATURE PIPELINE INVENTORIED")
print("✅ R3 46-FEATURE CONTRACT VERIFIED")
print("✅ SECTION 4B COMPLETE")


# In[23]:


# ==========================================================================================
# SECTION 4C — LOCATE AND INSPECT PRODUCTION POLICY STATE BUILDER
# ==========================================================================================

print("=" * 100)
print("SECTION 4C — LOCATE AND INSPECT PRODUCTION POLICY STATE BUILDER")
print("=" * 100)

import ast
import pandas as pd

TARGET_FUNCTION = "build_policy_state"

search_directories = [
    SRC_DIR / "decision_features",
    SRC_DIR / "observation_adapter",
]

definition_records = []

for directory in search_directories:

    if not directory.exists():
        continue

    for file_path in directory.rglob("*.py"):

        if "__pycache__" in file_path.parts:
            continue

        source = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:

            if (
                isinstance(node, ast.FunctionDef)
                and node.name == TARGET_FUNCTION
            ):

                definition_records.append(
                    {
                        "file": str(file_path),
                        "filename": file_path.name,
                        "line_number": node.lineno,
                        "arguments": [
                            arg.arg
                            for arg in node.args.args
                        ],
                        "source": ast.get_source_segment(
                            source,
                            node,
                        ),
                    }
                )

section4c_builder_locations_df = pd.DataFrame(
    definition_records
)

print("\nBUILD_POLICY_STATE DEFINITIONS")
print("-" * 100)

if section4c_builder_locations_df.empty:

    print(
        "No direct build_policy_state() definition "
        "was found in the production packages."
    )

else:

    display(
        section4c_builder_locations_df[
            [
                "filename",
                "file",
                "line_number",
                "arguments",
            ]
        ]
    )

assert not section4c_builder_locations_df.empty, (
    "build_policy_state() definition could not be located."
)

# ------------------------------------------------------------------------------------------
# Display exact implementation
# ------------------------------------------------------------------------------------------

builder_record = (
    section4c_builder_locations_df
    .iloc[0]
)

print("\nBUILD_POLICY_STATE SOURCE")
print("-" * 100)

print(
    builder_record["source"]
)

# ==========================================================================================
# LOCATE POLICY SEARCH ADAPTER
# ==========================================================================================

adapter_file = (
    SRC_DIR
    / "observation_adapter"
    / "policy_search_adapter.py"
)

assert adapter_file.exists(), (
    f"Missing policy search adapter:\n"
    f"{adapter_file}"
)

adapter_source = adapter_file.read_text(
    encoding="utf-8",
    errors="ignore",
)

adapter_tree = ast.parse(
    adapter_source
)

adapter_methods = []

for node in adapter_tree.body:

    if (
        isinstance(node, ast.ClassDef)
        and node.name == "PolicySearchEngineAdapter"
    ):

        for child in node.body:

            if isinstance(child, ast.FunctionDef):

                adapter_methods.append(
                    {
                        "method": child.name,
                        "arguments": [
                            arg.arg
                            for arg in child.args.args
                        ],
                        "source": ast.get_source_segment(
                            adapter_source,
                            child,
                        ),
                    }
                )

section4c_adapter_methods_df = pd.DataFrame(
    adapter_methods
)

print("\nPOLICY SEARCH ADAPTER METHODS")
print("-" * 100)

display(
    section4c_adapter_methods_df[
        [
            "method",
            "arguments",
        ]
    ]
)

# ------------------------------------------------------------------------------------------
# Display search_depth implementation
# ------------------------------------------------------------------------------------------

search_depth_rows = (
    section4c_adapter_methods_df[
        section4c_adapter_methods_df[
            "method"
        ] == "search_depth"
    ]
)

assert not search_depth_rows.empty, (
    "PolicySearchEngineAdapter.search_depth() "
    "was not found."
)

print("\nPOLICY SEARCH ADAPTER — search_depth()")
print("-" * 100)

print(
    search_depth_rows.iloc[0][
        "source"
    ]
)

print("\n✅ BUILD_POLICY_STATE LOCATION CONFIRMED")
print("✅ BUILD_POLICY_STATE IMPLEMENTATION INSPECTED")
print("✅ POLICY SEARCH ADAPTER FLOW INSPECTED")
print("✅ SECTION 4C COMPLETE")


# In[24]:


# ==========================================================================================
# SECTION 5A — INSPECT COMPETITIVE POLICY ENGINE SCORE_MOVES CONTRACT
# ==========================================================================================

print("=" * 100)
print("SECTION 5A — INSPECT COMPETITIVE POLICY ENGINE SCORE_MOVES CONTRACT")
print("=" * 100)

import ast

engine_file = (
    SRC_DIR
    / "competitive"
    / "competitive_policy_engine.py"
)

assert engine_file.exists(), (
    f"Competitive policy engine file not found:\n"
    f"{engine_file}"
)

source = engine_file.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source)

score_moves_source = None
engine_method_records = []

for node in tree.body:

    if not (
        isinstance(node, ast.ClassDef)
        and node.name == "CompetitivePolicyEngine"
    ):
        continue

    for child in node.body:

        if not isinstance(
            child,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        arguments = [
            arg.arg
            for arg in child.args.args
        ]

        engine_method_records.append(
            {
                "method": child.name,
                "arguments": arguments,
            }
        )

        if child.name == "score_moves":

            score_moves_source = (
                ast.get_source_segment(
                    source,
                    child,
                )
            )

section5a_engine_methods_df = pd.DataFrame(
    engine_method_records
)

print("\nCOMPETITIVE POLICY ENGINE METHODS")
print("-" * 100)

display(
    section5a_engine_methods_df
)

assert score_moves_source is not None, (
    "CompetitivePolicyEngine.score_moves() was not found."
)

print("\nSCORE_MOVES IMPLEMENTATION")
print("-" * 100)

print(
    score_moves_source
)

# ==========================================================================================
# Identify helper methods called by score_moves()
# ==========================================================================================

score_tree = ast.parse(
    score_moves_source
)

called_methods = set()

for node in ast.walk(score_tree):

    if isinstance(node, ast.Call):

        if isinstance(node.func, ast.Attribute):

            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):

                called_methods.add(
                    node.func.attr
                )

print("\nHELPER METHODS CALLED BY SCORE_MOVES")
print("-" * 100)

for method in sorted(
    called_methods
):
    print(method)

# ==========================================================================================
# Display helper implementations
# ==========================================================================================

helper_sources = {}

for node in tree.body:

    if not (
        isinstance(node, ast.ClassDef)
        and node.name == "CompetitivePolicyEngine"
    ):
        continue

    for child in node.body:

        if not isinstance(child, ast.FunctionDef):
            continue

        if child.name in called_methods:

            helper_sources[
                child.name
            ] = ast.get_source_segment(
                source,
                child,
            )

print("\nSCORE_MOVES HELPER IMPLEMENTATIONS")
print("-" * 100)

for method_name, method_source in helper_sources.items():

    print()
    print("=" * 90)
    print(method_name)
    print("=" * 90)

    print(
        method_source
    )

# ==========================================================================================
# Search returned dictionary keys
# ==========================================================================================

return_key_records = []

for node in ast.walk(score_tree):

    if not isinstance(node, ast.Return):
        continue

    if isinstance(node.value, ast.Dict):

        keys = []

        for key in node.value.keys:

            if isinstance(
                key,
                ast.Constant,
            ):

                keys.append(
                    key.value
                )

            else:

                keys.append(
                    ast.dump(key)
                )

        return_key_records.append(
            {
                "return_keys": keys
            }
        )

section5a_return_contract_df = pd.DataFrame(
    return_key_records
)

print("\nSCORE_MOVES RETURN CONTRACT")
print("-" * 100)

if section5a_return_contract_df.empty:

    print(
        "No literal dictionary return was detected. "
        "Inspect the printed source above."
    )

else:

    display(
        section5a_return_contract_df
    )

print("\n✅ SCORE_MOVES IMPLEMENTATION INSPECTED")
print("✅ SCORE_MOVES HELPER METHODS IDENTIFIED")
print("✅ RETURN CONTRACT INSPECTED")
print("✅ SECTION 5A COMPLETE")


# In[25]:


# ==========================================================================================
# SECTION 5B — LOCATE EXACT R3 FEATURE-CONSTRUCTION LOGIC
# ==========================================================================================

print("=" * 100)
print("SECTION 5B — LOCATE EXACT R3 FEATURE-CONSTRUCTION LOGIC")
print("=" * 100)

import ast
import pandas as pd

# ------------------------------------------------------------------------------------------
# Search the final pre-ablation / ablation scripts that actually produced the R3 features
# ------------------------------------------------------------------------------------------

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean_v2.py",
]

required_feature_terms = [
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
    "battle_phase",
    "energy_band",
    "damage_band",
    "active_card",
    "inactive_card",
]

function_records = []

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    print(f"\nScanning: {script_path.name}")

    source = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(
            f"Skipping {script_path.name} "
            f"because AST parsing failed: {exc}"
        )
        continue

    # ------------------------------------------------------------------
    # Inspect every function and find those containing R3 feature logic
    # ------------------------------------------------------------------

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        function_source = ast.get_source_segment(
            source,
            node,
        )

        if function_source is None:
            continue

        function_lower = function_source.lower()

        matched_terms = [
            term
            for term in required_feature_terms
            if term.lower() in function_lower
        ]

        if matched_terms:

            function_records.append(
                {
                    "script": script_path.name,
                    "function": node.name,
                    "line_number": node.lineno,
                    "matched_term_count": len(
                        matched_terms
                    ),
                    "matched_terms": matched_terms,
                    "arguments": [
                        arg.arg
                        for arg in node.args.args
                    ],
                    "source": function_source,
                }
            )

section5b_feature_builder_candidates_df = pd.DataFrame(
    function_records
)

print("\nR3 FEATURE-BUILDER CANDIDATES")
print("-" * 100)

assert not section5b_feature_builder_candidates_df.empty, (
    "No function containing the R3 feature-construction logic "
    "was found in the Notebook 54/55 scripts."
)

display(
    section5b_feature_builder_candidates_df[
        [
            "script",
            "function",
            "line_number",
            "matched_term_count",
            "matched_terms",
            "arguments",
        ]
    ]
    .sort_values(
        [
            "matched_term_count",
            "script",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

# ==========================================================================================
# Select the strongest candidate
# ==========================================================================================

best_candidate = (
    section5b_feature_builder_candidates_df
    .sort_values(
        "matched_term_count",
        ascending=False,
    )
    .iloc[0]
)

print("\nSTRONGEST FEATURE-BUILDER CANDIDATE")
print("-" * 100)

print(
    f"Script        : {best_candidate['script']}"
)

print(
    f"Function      : {best_candidate['function']}"
)

print(
    f"Line          : {best_candidate['line_number']}"
)

print(
    f"Matched terms : {best_candidate['matched_term_count']}"
)

print(
    f"Arguments     : {best_candidate['arguments']}"
)

print("\nSOURCE")
print("-" * 100)

print(
    best_candidate["source"]
)

# ==========================================================================================
# Determine term coverage
# ==========================================================================================

best_terms = set(
    best_candidate["matched_terms"]
)

missing_terms = [
    term
    for term in required_feature_terms
    if term not in best_terms
]

print("\nFEATURE-CONSTRUCTION COVERAGE")
print("-" * 100)

print(
    f"Required terms : {len(required_feature_terms)}"
)

print(
    f"Located terms  : {len(best_terms)}"
)

print(
    f"Missing terms  : {len(missing_terms)}"
)

if missing_terms:

    print("\nTerms not present in the strongest function:")

    for term in missing_terms:
        print(f"  - {term}")

else:

    print(
        "\nAll targeted R3 derived-feature concepts "
        "occur in the selected function."
    )

print("\n✅ R3 FEATURE-CONSTRUCTION CANDIDATES LOCATED")
print("✅ STRONGEST EXISTING FEATURE BUILDER IDENTIFIED")
print("✅ SECTION 5B COMPLETE")


# In[26]:


# ==========================================================================================
# SECTION 5C — LOCATE EXACT R3 PREPROCESSING PIPELINE
# ==========================================================================================

print("=" * 100)
print("SECTION 5C — LOCATE EXACT R3 PREPROCESSING PIPELINE")
print("=" * 100)

import ast
import pandas as pd

# ------------------------------------------------------------------------------------------
# Search the final Notebook 54 and Notebook 55 scripts.
# We need the transformation from the 41 raw columns into encoded model features.
# ------------------------------------------------------------------------------------------

candidate_scripts = [
    PROJECT_ROOT
    / "scripts"
    / "54_tournament_strength_optimization_clean.py",

    PROJECT_ROOT
    / "scripts"
    / "55_controlled_feature_ablation_retraining_clean.py",

    PROJECT_ROOT
    / "scripts"
    / "55_controlled_feature_ablation_retraining_clean_v2.py",
]

search_terms = [
    "ColumnTransformer",
    "OneHotEncoder",
    "StandardScaler",
    "get_feature_names_out",
    "fit_transform",
    "transform",
    "PREPROCESS_RAW_FEATURES",
    "legality_training_numeric_features",
    "legality_training_categorical_features",
]

records = []

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    # ----------------------------------------------------------------------
    # Functions containing preprocessing-related logic
    # ----------------------------------------------------------------------

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        function_source = ast.get_source_segment(
            source,
            node,
        )

        if function_source is None:
            continue

        lower_source = function_source.lower()

        matched_terms = [
            term
            for term in search_terms
            if term.lower() in lower_source
        ]

        if not matched_terms:
            continue

        records.append(
            {
                "script": script_path.name,
                "object_type": "FUNCTION",
                "name": node.name,
                "line_number": node.lineno,
                "matched_term_count": len(
                    matched_terms
                ),
                "matched_terms": matched_terms,
                "arguments": [
                    arg.arg
                    for arg in node.args.args
                ],
                "source": function_source,
            }
        )

# ------------------------------------------------------------------------------------------
# Also capture assignment blocks that define preprocessing objects.
# ------------------------------------------------------------------------------------------

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = source.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            term.lower() in line.lower()
            for term in search_terms
        ):

            start = max(
                0,
                line_number - 8,
            )

            end = min(
                len(lines),
                line_number + 18,
            )

            context = "\n".join(
                lines[start:end]
            )

            records.append(
                {
                    "script": script_path.name,
                    "object_type": "SOURCE_CONTEXT",
                    "name": (
                        f"context_line_{line_number}"
                    ),
                    "line_number": line_number,
                    "matched_term_count": sum(
                        term.lower()
                        in context.lower()
                        for term in search_terms
                    ),
                    "matched_terms": [
                        term
                        for term in search_terms
                        if term.lower()
                        in context.lower()
                    ],
                    "arguments": None,
                    "source": context,
                }
            )

section5c_preprocessing_candidates_df = pd.DataFrame(
    records
)

assert not section5c_preprocessing_candidates_df.empty, (
    "No preprocessing implementation was found "
    "in the Notebook 54/55 scripts."
)

print("\nPREPROCESSING CANDIDATES")
print("-" * 100)

display(
    section5c_preprocessing_candidates_df[
        [
            "script",
            "object_type",
            "name",
            "line_number",
            "matched_term_count",
            "matched_terms",
            "arguments",
        ]
    ]
    .sort_values(
        [
            "matched_term_count",
            "script",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .head(30)
    .reset_index(drop=True)
)

# ==========================================================================================
# Display strongest FUNCTION candidate
# ==========================================================================================

function_candidates = (
    section5c_preprocessing_candidates_df[
        section5c_preprocessing_candidates_df[
            "object_type"
        ] == "FUNCTION"
    ]
)

if not function_candidates.empty:

    best_function = (
        function_candidates
        .sort_values(
            "matched_term_count",
            ascending=False,
        )
        .iloc[0]
    )

    print("\nSTRONGEST PREPROCESSING FUNCTION")
    print("-" * 100)

    print(
        f"Script   : {best_function['script']}"
    )

    print(
        f"Function : {best_function['name']}"
    )

    print(
        f"Line     : {best_function['line_number']}"
    )

    print(
        f"Args     : {best_function['arguments']}"
    )

    print("\nSOURCE")
    print("-" * 100)

    print(
        best_function["source"]
    )

else:

    print(
        "\nNo self-contained preprocessing function "
        "was detected."
    )

# ==========================================================================================
# Display strongest source contexts
# ==========================================================================================

print("\nSTRONGEST PREPROCESSING SOURCE CONTEXTS")
print("-" * 100)

context_candidates = (
    section5c_preprocessing_candidates_df[
        section5c_preprocessing_candidates_df[
            "object_type"
        ] == "SOURCE_CONTEXT"
    ]
    .sort_values(
        "matched_term_count",
        ascending=False,
    )
    .head(5)
)

for _, row in context_candidates.iterrows():

    print("\n" + "=" * 90)

    print(
        f"{row['script']} "
        f"around line {row['line_number']}"
    )

    print("=" * 90)

    print(
        row["source"]
    )

print("\n✅ R3 PREPROCESSING PIPELINE SEARCH COMPLETE")
print("✅ SECTION 5C COMPLETE")


# In[27]:


# ==========================================================================================
# SECTION 5D — RECOVER TRAINING PREPROCESSOR IMPLEMENTATION
# ==========================================================================================

print("=" * 100)
print("SECTION 5D — RECOVER TRAINING PREPROCESSOR IMPLEMENTATION")
print("=" * 100)

import ast
import pandas as pd

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

interesting_names = [
    "PREPROCESS_NUMERIC_FEATURES",
    "PREPROCESS_CATEGORICAL_FEATURES",
    "PREPROCESS_RAW_FEATURES",
    "preprocessor",
    "ColumnTransformer",
    "OneHotEncoder",
    "get_feature_names_out",
    "fit_transform",
    "transform",
]

matches = []

for script in candidate_scripts:

    if not script.exists():
        continue

    lines = script.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    for i, line in enumerate(lines, start=1):

        if any(
            token.lower() in line.lower()
            for token in interesting_names
        ):

            start = max(0, i - 20)
            end = min(len(lines), i + 40)

            matches.append(
                {
                    "script": script.name,
                    "line": i,
                    "context": "\n".join(lines[start:end]),
                }
            )

section5d_matches_df = pd.DataFrame(matches)

display(
    section5d_matches_df[
        ["script", "line"]
    ]
)

print("\nMATCHED CONTEXTS")
print("-" * 100)

for _, row in section5d_matches_df.iterrows():

    print("\n" + "=" * 90)
    print(
        f"{row['script']}  line {row['line']}"
    )
    print("=" * 90)

    print(row["context"])

print()
print("✅ TRAINING PREPROCESSOR SOURCE RECOVERED")
print("✅ SECTION 5D COMPLETE")


# In[33]:


# ==========================================================================================
# SECTION 6A — DIAGNOSE FROZEN PREPROCESSOR VERSION COMPATIBILITY
# ==========================================================================================

print("=" * 100)
print("SECTION 6A — DIAGNOSE FROZEN PREPROCESSOR VERSION COMPATIBILITY")
print("=" * 100)

import sys
import sklearn
import joblib
import pickle
from pathlib import Path

print("\nCURRENT RUNTIME")
print("-" * 100)

print("Python      :", sys.version)
print("scikit-learn:", sklearn.__version__)
print("joblib      :", joblib.__version__)

print("\nFROZEN PREPROCESSOR")
print("-" * 100)

print(NOTEBOOK53_PREPROCESSOR_PATH_59)

assert NOTEBOOK53_PREPROCESSOR_PATH_59.exists()
assert NOTEBOOK53_PREPROCESSOR_PATH_59.stat().st_size > 0

# ------------------------------------------------------------------------------------------
# Confirm whether current sklearn contains the serialized private class
# ------------------------------------------------------------------------------------------

import sklearn.compose._column_transformer as sklearn_column_transformer

has_remainder_cols_list = hasattr(
    sklearn_column_transformer,
    "_RemainderColsList",
)

print("\nPRIVATE CLASS COMPATIBILITY")
print("-" * 100)

print(
    "Current sklearn exposes _RemainderColsList:",
    has_remainder_cols_list,
)

# ------------------------------------------------------------------------------------------
# Attempt normal load and capture exact failure
# ------------------------------------------------------------------------------------------

normal_load_success = False
normal_load_error = None
normal_load_error_type = None

try:

    notebook53_preprocessor_probe = joblib.load(
        NOTEBOOK53_PREPROCESSOR_PATH_59
    )

    normal_load_success = True

except Exception as exc:

    normal_load_error = str(exc)
    normal_load_error_type = type(exc).__name__

print("\nNORMAL JOBLIB LOAD")
print("-" * 100)

print("Success    :", normal_load_success)
print("Error type :", normal_load_error_type)
print("Error      :", normal_load_error)

# ------------------------------------------------------------------------------------------
# Diagnostic conclusion
# ------------------------------------------------------------------------------------------

version_compatibility_issue = (
    not normal_load_success
    and normal_load_error is not None
    and "_RemainderColsList" in normal_load_error
)

print("\nDIAGNOSTIC RESULT")
print("-" * 100)

print(
    "Version compatibility issue:",
    version_compatibility_issue,
)

if version_compatibility_issue:

    print(
        "\nThe frozen Notebook 53 preprocessor was serialized "
        "under a different scikit-learn implementation than "
        "the currently installed runtime."
    )

    print(
        "\nNo artifact corruption has been detected."
    )

    print(
        "\nDo NOT overwrite or rebuild the frozen preprocessor."
    )

assert version_compatibility_issue, (
    "The observed error does not match the expected "
    "_RemainderColsList compatibility failure."
)

print("\n✅ PREPROCESSOR ARTIFACT EXISTS")
print("✅ LOAD FAILURE ISOLATED TO SCIKIT-LEARN COMPATIBILITY")
print("✅ FROZEN ARTIFACT LEFT UNCHANGED")
print("✅ SECTION 6A DIAGNOSTIC COMPLETE")


# In[34]:


# ==========================================================================================
# SECTION 6B — BUILD EXACT 46-FEATURE R3 LIVE ENCODER
# ==========================================================================================

print("=" * 100)
print("SECTION 6B — BUILD EXACT 46-FEATURE R3 LIVE ENCODER")
print("=" * 100)

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------
# 1. Use the exact retained feature order frozen inside the approved R3 package
# ------------------------------------------------------------------------------------------

R3_ENCODED_FEATURE_NAMES = list(
    r3_live_feature_names
)

assert len(
    R3_ENCODED_FEATURE_NAMES
) == 46

assert len(
    R3_ENCODED_FEATURE_NAMES
) == len(
    set(
        R3_ENCODED_FEATURE_NAMES
    )
)

# ------------------------------------------------------------------------------------------
# 2. Split retained features by preprocessing family
# ------------------------------------------------------------------------------------------

R3_NUMERIC_FEATURES = [
    feature_name
    for feature_name in R3_ENCODED_FEATURE_NAMES
    if feature_name.startswith(
        "numeric__"
    )
]

R3_CATEGORICAL_FEATURES = [
    feature_name
    for feature_name in R3_ENCODED_FEATURE_NAMES
    if feature_name.startswith(
        "categorical__"
    )
]

assert len(
    R3_NUMERIC_FEATURES
) == 19, (
    f"Expected 19 numeric R3 features, "
    f"found {len(R3_NUMERIC_FEATURES)}."
)

assert len(
    R3_CATEGORICAL_FEATURES
) == 27, (
    f"Expected 27 categorical R3 features, "
    f"found {len(R3_CATEGORICAL_FEATURES)}."
)

print("\nR3 RETAINED FEATURE FAMILY COUNTS")
print("-" * 100)

print(
    "Numeric retained features     :",
    len(
        R3_NUMERIC_FEATURES
    ),
)

print(
    "Categorical retained features :",
    len(
        R3_CATEGORICAL_FEATURES
    ),
)

# ==========================================================================================
# 3. Define exact categorical source columns
# ==========================================================================================

R3_CATEGORICAL_SOURCE_COLUMNS = [
    "current_side",
    "side_mode",
    "player_card",
    "opponent_card",
    "active_card",
    "inactive_card",
    "battle_phase",
    "energy_band",
    "damage_band",
]

# Sort longest first so active_card does not accidentally
# match inactive_card.
R3_CATEGORICAL_SOURCE_COLUMNS = sorted(
    R3_CATEGORICAL_SOURCE_COLUMNS,
    key=len,
    reverse=True,
)

# ==========================================================================================
# 4. Parse an encoded categorical feature name
# ==========================================================================================

def parse_r3_categorical_feature(
    encoded_feature_name,
):
    """
    Convert for example

        categorical__active_card_Eevee

    into

        ("active_card", "Eevee")
    """

    assert encoded_feature_name.startswith(
        "categorical__"
    )

    encoded_body = encoded_feature_name[
        len("categorical__"):
    ]

    for source_column in (
        R3_CATEGORICAL_SOURCE_COLUMNS
    ):

        prefix = (
            source_column
            + "_"
        )

        if encoded_body.startswith(
            prefix
        ):

            category_value = encoded_body[
                len(prefix):
            ]

            return (
                source_column,
                category_value,
            )

    raise ValueError(
        "Unable to resolve categorical source "
        f"for encoded feature: {encoded_feature_name}"
    )

# ------------------------------------------------------------------------------------------
# Validate parsing of all retained categorical columns
# ------------------------------------------------------------------------------------------

categorical_mapping_records = []

for encoded_name in (
    R3_CATEGORICAL_FEATURES
):

    source_column, category_value = (
        parse_r3_categorical_feature(
            encoded_name
        )
    )

    categorical_mapping_records.append(
        {
            "encoded_feature":
                encoded_name,

            "source_column":
                source_column,

            "category_value":
                category_value,
        }
    )

section6b_categorical_mapping_df = (
    pd.DataFrame(
        categorical_mapping_records
    )
)

print("\nR3 CATEGORICAL FEATURE MAPPING")
print("-" * 100)

display(
    section6b_categorical_mapping_df
)

# ==========================================================================================
# 5. Build exact R3 encoded row
# ==========================================================================================

def encode_r3_feature_row(
    raw_feature_dataframe,
):
    """
    Convert the 1x41 raw production feature dataframe
    into the exact 1x46 feature matrix expected by R3.

    This intentionally reconstructs ONLY the retained
    R3 features and does not recreate discarded features.
    """

    assert isinstance(
        raw_feature_dataframe,
        pd.DataFrame,
    )

    assert len(
        raw_feature_dataframe
    ) == 1, (
        "R3 live encoder expects exactly one raw row."
    )

    raw_row = (
        raw_feature_dataframe
        .iloc[0]
    )

    encoded_values = {}

    # ------------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------------

    for encoded_name in (
        R3_NUMERIC_FEATURES
    ):

        raw_column = encoded_name[
            len("numeric__"):
        ]

        assert raw_column in raw_row.index, (
            f"Required numeric raw feature missing: "
            f"{raw_column}"
        )

        numeric_value = pd.to_numeric(
            raw_row[
                raw_column
            ],
            errors="coerce",
        )

        if pd.isna(
            numeric_value
        ):
            numeric_value = 0.0

        encoded_values[
            encoded_name
        ] = float(
            numeric_value
        )

    # ------------------------------------------------------------------
    # Categorical one-hot columns
    # ------------------------------------------------------------------

    for encoded_name in (
        R3_CATEGORICAL_FEATURES
    ):

        (
            source_column,
            category_value,
        ) = parse_r3_categorical_feature(
            encoded_name
        )

        assert source_column in raw_row.index, (
            f"Required categorical raw feature missing: "
            f"{source_column}"
        )

        actual_value = str(
            raw_row[
                source_column
            ]
        )

        encoded_values[
            encoded_name
        ] = float(
            actual_value
            ==
            category_value
        )

    # ------------------------------------------------------------------
    # Freeze exact R3 order
    # ------------------------------------------------------------------

    encoded_dataframe = pd.DataFrame(
        [
            encoded_values
        ],
        columns=
            R3_ENCODED_FEATURE_NAMES,
        dtype=float,
    )

    assert encoded_dataframe.shape == (
        1,
        46,
    ), (
        "Unexpected R3 encoded matrix shape: "
        f"{encoded_dataframe.shape}"
    )

    assert list(
        encoded_dataframe.columns
    ) == R3_ENCODED_FEATURE_NAMES

    assert np.isfinite(
        encoded_dataframe.to_numpy(
            dtype=float
        )
    ).all()

    return encoded_dataframe


# ==========================================================================================
# 6. Structural validation
# ==========================================================================================

numeric_raw_columns = [
    name.replace(
        "numeric__",
        "",
        1,
    )
    for name in (
        R3_NUMERIC_FEATURES
    )
]

categorical_source_columns = sorted(
    set(
        section6b_categorical_mapping_df[
            "source_column"
        ]
        .astype(str)
        .tolist()
    )
)

section6b_encoder_contract_df = (
    pd.DataFrame(
        [
            {
                "contract":
                    "R3 retained features",

                "value":
                    len(
                        R3_ENCODED_FEATURE_NAMES
                    ),

                "expected":
                    46,

                "passed":
                    (
                        len(
                            R3_ENCODED_FEATURE_NAMES
                        )
                        == 46
                    ),
            },

            {
                "contract":
                    "R3 numeric features",

                "value":
                    len(
                        R3_NUMERIC_FEATURES
                    ),

                "expected":
                    19,

                "passed":
                    (
                        len(
                            R3_NUMERIC_FEATURES
                        )
                        == 19
                    ),
            },

            {
                "contract":
                    "R3 categorical indicators",

                "value":
                    len(
                        R3_CATEGORICAL_FEATURES
                    ),

                "expected":
                    27,

                "passed":
                    (
                        len(
                            R3_CATEGORICAL_FEATURES
                        )
                        == 27
                    ),
            },

            {
                "contract":
                    "R3 model input features",

                "value":
                    int(
                        r3_live_model
                        .n_features_in_
                    ),

                "expected":
                    46,

                "passed":
                    (
                        int(
                            r3_live_model
                            .n_features_in_
                        )
                        == 46
                    ),
            },
        ]
    )
)

print("\nR3 LIVE ENCODER CONTRACT")
print("-" * 100)

display(
    section6b_encoder_contract_df
)

assert section6b_encoder_contract_df[
    "passed"
].all()

print("\nNUMERIC RAW INPUTS")
print("-" * 100)

for column_name in (
    numeric_raw_columns
):
    print(column_name)

print("\nCATEGORICAL RAW INPUTS")
print("-" * 100)

for column_name in (
    categorical_source_columns
):
    print(column_name)

print(
    "\n✅ EXACT 46-FEATURE R3 ENCODER CREATED"
)

print(
    "✅ NUMERIC FEATURE MAPPING VERIFIED"
)

print(
    "✅ CATEGORICAL ONE-HOT MAPPING VERIFIED"
)

print(
    "✅ FROZEN PREPROCESSOR VERSION CONFLICT BYPASSED SAFELY"
)

print(
    "✅ NO MODEL OR ARTIFACT MODIFICATION PERFORMED"
)

print(
    "✅ SECTION 6B COMPLETE"
)


# In[80]:


# ==========================================================================================
# SECTION 6C — VALIDATE END-TO-END R3 LIVE INFERENCE PIPELINE
# ==========================================================================================

import numpy as np
import pandas as pd

print("=" * 100)
print("SECTION 6C — VALIDATE END-TO-END R3 LIVE INFERENCE PIPELINE")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# 1. Confirm all required live components are loaded
# ------------------------------------------------------------------------------------------

required_objects_6c = [
    "build_legality_aware_feature_row",
    "encode_r3_feature_row",
    "r3_live_model",
    "r3_live_feature_names",
]

missing_objects_6c = [
    name
    for name in required_objects_6c
    if name not in globals()
]

assert not missing_objects_6c, (
    "Missing required Section 6C objects: "
    f"{missing_objects_6c}"
)

print("\nLIVE COMPONENT CHECK")
print("-" * 100)

for name in required_objects_6c:
    print(f"{name}: LOADED")


# ------------------------------------------------------------------------------------------
# 2. Confirm the test state and legal moves from Section 6C.2 still exist
# ------------------------------------------------------------------------------------------

assert "section6c2_test_state" in globals(), (
    "section6c2_test_state is not available."
)

assert "section6c2_legal_moves" in globals(), (
    "section6c2_legal_moves is not available."
)

assert len(section6c2_legal_moves) > 0, (
    "section6c2_legal_moves is empty."
)

print("\nTEST INPUT")
print("-" * 100)

print(
    "Battle state type :",
    type(section6c2_test_state).__name__,
)

print(
    "Legal move count  :",
    len(section6c2_legal_moves),
)


# ------------------------------------------------------------------------------------------
# 3. Build exact 41-column live raw feature row
#
# IMPORTANT:
# build_legality_aware_feature_row requires BOTH:
#     battle_state
#     legal_moves
# ------------------------------------------------------------------------------------------

section6c_raw_feature_row = (
    build_legality_aware_feature_row(
        section6c2_test_state,
        section6c2_legal_moves,
    )
)

assert isinstance(
    section6c_raw_feature_row,
    pd.DataFrame,
), (
    "Raw feature builder did not return a pandas DataFrame."
)

assert section6c_raw_feature_row.shape == (1, 41), (
    "Unexpected raw feature shape: "
    f"{section6c_raw_feature_row.shape}; "
    "expected (1, 41)."
)

print("\nRAW FEATURE ROW")
print("-" * 100)

print(
    "Type  :",
    type(section6c_raw_feature_row).__name__,
)

print(
    "Shape :",
    section6c_raw_feature_row.shape,
)

display(section6c_raw_feature_row)


# ------------------------------------------------------------------------------------------
# 4. Encode raw row into exact R3 46-feature model matrix
# ------------------------------------------------------------------------------------------

section6c_r3_matrix = (
    encode_r3_feature_row(
        section6c_raw_feature_row
    )
)

section6c_r3_matrix = np.asarray(
    section6c_r3_matrix,
    dtype=np.float64,
)

if section6c_r3_matrix.ndim == 1:
    section6c_r3_matrix = (
        section6c_r3_matrix.reshape(1, -1)
    )

assert section6c_r3_matrix.shape == (1, 46), (
    "Unexpected R3 encoded feature shape: "
    f"{section6c_r3_matrix.shape}; "
    "expected (1, 46)."
)

assert len(r3_live_feature_names) == 46, (
    "R3 live feature-name contract is not 46 features."
)

assert np.isfinite(
    section6c_r3_matrix
).all(), (
    "R3 feature matrix contains NaN or infinite values."
)

print("\nR3 ENCODED MATRIX")
print("-" * 100)

print(
    "Shape :",
    section6c_r3_matrix.shape,
)

section6c_encoded_df = pd.DataFrame(
    section6c_r3_matrix,
    columns=r3_live_feature_names,
)

display(section6c_encoded_df)


# ------------------------------------------------------------------------------------------
# 5. Execute live R3 model prediction
# ------------------------------------------------------------------------------------------

section6c_prediction = (
    r3_live_model.predict(
        section6c_r3_matrix
    )
)

assert len(section6c_prediction) == 1, (
    "R3 model did not return exactly one prediction."
)

section6c_predicted_action = (
    section6c_prediction[0]
)

print("\nR3 LIVE PREDICTION")
print("-" * 100)

print(
    "Predicted action :",
    section6c_predicted_action,
)


# ------------------------------------------------------------------------------------------
# 6. Execute and validate probability vector
# ------------------------------------------------------------------------------------------

assert hasattr(
    r3_live_model,
    "predict_proba",
), (
    "R3 model does not expose predict_proba()."
)

section6c_probabilities = (
    r3_live_model.predict_proba(
        section6c_r3_matrix
    )
)

assert section6c_probabilities.ndim == 2
assert section6c_probabilities.shape[0] == 1

section6c_probability_vector = (
    section6c_probabilities[0]
)

section6c_classes = list(
    r3_live_model.classes_
)

assert (
    len(section6c_probability_vector)
    ==
    len(section6c_classes)
), (
    "Probability vector length does not match "
    "R3 class count."
)

assert np.isfinite(
    section6c_probability_vector
).all()

assert np.isclose(
    float(
        np.sum(
            section6c_probability_vector
        )
    ),
    1.0,
    atol=1e-9,
), (
    "R3 probability vector does not sum to 1."
)

section6c_probability_df = pd.DataFrame(
    {
        "action": section6c_classes,
        "probability": (
            section6c_probability_vector
        ),
    }
).sort_values(
    "probability",
    ascending=False,
).reset_index(
    drop=True
)

print("\nR3 ACTION PROBABILITIES")
print("-" * 100)

display(section6c_probability_df)


# ------------------------------------------------------------------------------------------
# 7. Final validation summary
# ------------------------------------------------------------------------------------------

section6c_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "raw_feature_row_shape",

            "observed":
                str(
                    section6c_raw_feature_row.shape
                ),

            "expected":
                "(1, 41)",

            "passed":
                section6c_raw_feature_row.shape
                == (1, 41),
        },

        {
            "validation":
                "r3_encoded_matrix_shape",

            "observed":
                str(
                    section6c_r3_matrix.shape
                ),

            "expected":
                "(1, 46)",

            "passed":
                section6c_r3_matrix.shape
                == (1, 46),
        },

        {
            "validation":
                "r3_feature_name_count",

            "observed":
                len(
                    r3_live_feature_names
                ),

            "expected":
                46,

            "passed":
                len(
                    r3_live_feature_names
                ) == 46,
        },

        {
            "validation":
                "prediction_count",

            "observed":
                len(
                    section6c_prediction
                ),

            "expected":
                1,

            "passed":
                len(
                    section6c_prediction
                ) == 1,
        },

        {
            "validation":
                "probability_sum",

            "observed":
                float(
                    np.sum(
                        section6c_probability_vector
                    )
                ),

            "expected":
                1.0,

            "passed":
                np.isclose(
                    float(
                        np.sum(
                            section6c_probability_vector
                        )
                    ),
                    1.0,
                    atol=1e-9,
                ),
        },
    ]
)

print("\nEND-TO-END R3 LIVE INFERENCE VALIDATION")
print("-" * 100)

display(
    section6c_validation_df
)

assert section6c_validation_df[
    "passed"
].all()


# ------------------------------------------------------------------------------------------
# 8. Completion
# ------------------------------------------------------------------------------------------

print("\n" + "=" * 100)

print("✅ RAW LIVE FEATURE ROW BUILT")
print("✅ EXACT 41-COLUMN RAW FEATURE CONTRACT VERIFIED")
print("✅ EXACT 46-FEATURE R3 MATRIX CREATED")
print("✅ R3 PREDICTION EXECUTED")
print("✅ R3 PROBABILITY VECTOR VALIDATED")
print("✅ END-TO-END INFERENCE PATH CONFIRMED")
print("✅ SECTION 6C COMPLETE")

print("=" * 100)


# In[73]:


# ==========================================================================================
# SECTION 6C.1 — LOAD EXISTING LEGALITY-AWARE FEATURE BUILDER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.1 — LOAD EXISTING LEGALITY-AWARE FEATURE BUILDER")
print("=" * 100)

import ast
from pathlib import Path

# ------------------------------------------------------------------------------------------
# Search known Notebook 54/55 scripts for the exact function definition
# ------------------------------------------------------------------------------------------

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

target_function_name = "build_legality_aware_feature_row"

resolved_function_source = None
resolved_function_script = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == target_function_name
        ):

            resolved_function_source = ast.get_source_segment(
                source,
                node,
            )

            resolved_function_script = script_path

            break

    if resolved_function_source is not None:
        break

assert resolved_function_source is not None, (
    "Could not locate build_legality_aware_feature_row()."
)

print("\nFUNCTION SOURCE LOCATED IN")
print("-" * 100)

print(resolved_function_script)

# ------------------------------------------------------------------------------------------
# Execute ONLY the recovered function definition in Notebook 59
# ------------------------------------------------------------------------------------------

exec(
    resolved_function_source,
    globals(),
)

assert (
    "build_legality_aware_feature_row"
    in globals()
)

assert callable(
    build_legality_aware_feature_row
)

print("\n✅ build_legality_aware_feature_row() LOADED")
print("✅ FUNCTION IS CALLABLE")
print("✅ SECTION 6C.1 COMPLETE")


# In[38]:


# ==========================================================================================
# SECTION 6C.1A — LOAD FUNCTION DEPENDENCIES
# ==========================================================================================

from typing import Any, Sequence
import numpy as np
import pandas as pd

print("✅ Any loaded")
print("✅ Sequence loaded")
print("✅ numpy loaded")
print("✅ pandas loaded")
print("✅ SECTION 6C.1A COMPLETE")


# In[40]:


# ==========================================================================================
# SECTION 6C.1B — FORCE-LOAD LEGALITY-AWARE FEATURE BUILDER SAFELY
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.1B — FORCE-LOAD LEGALITY-AWARE FEATURE BUILDER SAFELY")
print("=" * 100)

import ast
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

TARGET_FUNCTION = "build_legality_aware_feature_row"

candidate_scripts = [
    PROJECT_ROOT
    / "scripts"
    / "54_tournament_strength_optimization_clean.py",

    PROJECT_ROOT
    / "scripts"
    / "55_controlled_feature_ablation_retraining_clean.py",
]

function_source = None
function_file = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source_text = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TARGET_FUNCTION
        ):

            function_source = ast.get_source_segment(
                source_text,
                node,
            )

            function_file = script_path
            break

    if function_source is not None:
        break

assert function_source is not None, (
    "Could not recover build_legality_aware_feature_row()."
)

print("\nRecovered from:")
print(function_file)

# ------------------------------------------------------------------------------------------
# Postpone annotation evaluation.
# This prevents Any / Sequence / other typing names from breaking function creation.
# ------------------------------------------------------------------------------------------

safe_function_source = (
    "from __future__ import annotations\n\n"
    + function_source
)

exec(
    safe_function_source,
    globals(),
)

# ------------------------------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------------------------------

builder_loaded = (
    TARGET_FUNCTION in globals()
    and callable(
        globals()[TARGET_FUNCTION]
    )
)

print("\nBuilder loaded:", builder_loaded)

assert builder_loaded, (
    "Builder was still not created."
)

print("\n✅ build_legality_aware_feature_row() LOADED")
print("✅ FUNCTION IS CALLABLE")
print("✅ SECTION 6C.1B COMPLETE")


# In[41]:


print(
    "Builder loaded:",
    "build_legality_aware_feature_row" in globals()
)

print(
    "Encoder loaded:",
    "encode_r3_feature_row" in globals()
)

print(
    "R3 model loaded:",
    "r3_live_model" in globals()
)


# In[77]:


# ==========================================================================================
# SECTION 6C.2 — TEST RAW LEGALITY-AWARE FEATURE BUILDER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2 — TEST RAW LEGALITY-AWARE FEATURE BUILDER")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# Controlled live-style battle state
# ------------------------------------------------------------------------------------------

section6c2_test_state = {
    "turn_number": 3,
    "current_side": "Player",
    "side_mode": "PRESERVE",

    "player_card": "Charmander",
    "opponent_card": "Eevee",

    "player_energy": 2,
    "opponent_energy": 1,

    "player_damage": 10,
    "opponent_damage": 20,

    "prize_cards_remaining": 4,
    "hand_size": 5,
}

# ------------------------------------------------------------------------------------------
# Legal moves are a SEPARATE argument
# ------------------------------------------------------------------------------------------

section6c2_legal_moves = [
    "Quick Attack",
    "Pass",
]

print("\nCalling build_legality_aware_feature_row()...")

section6c2_raw_row = (
    build_legality_aware_feature_row(
        section6c2_test_state,
        section6c2_legal_moves,
    )
)

print("\nBuilder returned successfully.")

# ------------------------------------------------------------------------------------------
# Inspect result
# ------------------------------------------------------------------------------------------

print("\nRETURN TYPE")
print("-" * 100)

print(
    type(
        section6c2_raw_row
    ).__name__
)

print("\nRETURN SHAPE")
print("-" * 100)

if hasattr(
    section6c2_raw_row,
    "shape",
):
    print(
        section6c2_raw_row.shape
    )
else:
    print(
        "No shape attribute"
    )

print("\nRAW FEATURE OUTPUT")
print("-" * 100)

display(
    section6c2_raw_row
)

# ------------------------------------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------------------------------------

assert isinstance(
    section6c2_raw_row,
    pd.DataFrame,
), (
    "Expected build_legality_aware_feature_row() "
    "to return a pandas DataFrame."
)

assert len(
    section6c2_raw_row
) == 1, (
    "Expected exactly one feature row."
)

print("\n✅ RAW FEATURE BUILDER EXECUTED")
print("✅ ONE FEATURE ROW RETURNED")
print("✅ SECTION 6C.2 COMPLETE")


# In[49]:


# ==========================================================================================
# SECTION 6C.2A — LOAD get_state_value HELPER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2A — LOAD get_state_value HELPER")
print("=" * 100)

import ast

TARGET_HELPER = "get_state_value"

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

helper_source = None
helper_file = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source_text = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TARGET_HELPER
        ):

            helper_source = ast.get_source_segment(
                source_text,
                node,
            )

            helper_file = script_path
            break

    if helper_source is not None:
        break

assert helper_source is not None, (
    "Could not locate get_state_value()."
)

safe_helper_source = (
    "from __future__ import annotations\n\n"
    + helper_source
)

exec(
    safe_helper_source,
    globals(),
)

assert (
    "get_state_value"
    in globals()
)

assert callable(
    get_state_value
)

print("\nRecovered from:")
print(helper_file)

print("\n✅ get_state_value() LOADED")
print("✅ HELPER IS CALLABLE")
print("✅ SECTION 6C.2A COMPLETE")


# In[51]:


# ==========================================================================================
# SECTION 6C.2B — LOAD get_card_name HELPER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2B — LOAD get_card_name HELPER")
print("=" * 100)

import ast

TARGET_HELPER = "get_card_name"

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

helper_source = None
helper_file = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source_text = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TARGET_HELPER
        ):
            helper_source = ast.get_source_segment(
                source_text,
                node,
            )
            helper_file = script_path
            break

    if helper_source is not None:
        break

assert helper_source is not None, (
    "Could not locate get_card_name()."
)

safe_helper_source = (
    "from __future__ import annotations\n\n"
    + helper_source
)

exec(
    safe_helper_source,
    globals(),
)

assert "get_card_name" in globals()
assert callable(get_card_name)

print("\nRecovered from:")
print(helper_file)

print("\n✅ get_card_name() LOADED")
print("✅ HELPER IS CALLABLE")
print("✅ SECTION 6C.2B COMPLETE")


# In[53]:


# ==========================================================================================
# SECTION 6C.2C — LOAD get_attached_energy HELPER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2C — LOAD get_attached_energy HELPER")
print("=" * 100)

import ast

TARGET_HELPER = "get_attached_energy"

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

helper_source = None
helper_file = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source_text = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TARGET_HELPER
        ):
            helper_source = ast.get_source_segment(
                source_text,
                node,
            )
            helper_file = script_path
            break

    if helper_source is not None:
        break

assert helper_source is not None, (
    "Could not locate get_attached_energy()."
)

safe_helper_source = (
    "from __future__ import annotations\n\n"
    + helper_source
)

exec(
    safe_helper_source,
    globals(),
)

assert "get_attached_energy" in globals()
assert callable(get_attached_energy)

print("\nRecovered from:")
print(helper_file)

print("\n✅ get_attached_energy() LOADED")
print("✅ HELPER IS CALLABLE")
print("✅ SECTION 6C.2C COMPLETE")


# In[55]:


# ==========================================================================================
# SECTION 6C.2D — LOAD get_pokemon_damage HELPER
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2D — LOAD get_pokemon_damage HELPER")
print("=" * 100)

import ast

TARGET_HELPER = "get_pokemon_damage"

candidate_scripts = [
    PROJECT_ROOT / "scripts" / "54_tournament_strength_optimization_clean.py",
    PROJECT_ROOT / "scripts" / "55_controlled_feature_ablation_retraining_clean.py",
]

helper_source = None
helper_file = None

for script_path in candidate_scripts:

    if not script_path.exists():
        continue

    source_text = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        continue

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name == TARGET_HELPER
        ):
            helper_source = ast.get_source_segment(
                source_text,
                node,
            )
            helper_file = script_path
            break

    if helper_source is not None:
        break

assert helper_source is not None, (
    "Could not locate get_pokemon_damage()."
)

safe_helper_source = (
    "from __future__ import annotations\n\n"
    + helper_source
)

exec(
    safe_helper_source,
    globals(),
)

assert "get_pokemon_damage" in globals()
assert callable(get_pokemon_damage)

print("\nRecovered from:")
print(helper_file)

print("\n✅ get_pokemon_damage() LOADED")
print("✅ HELPER IS CALLABLE")
print("✅ SECTION 6C.2D COMPLETE")


# In[57]:


# ==========================================================================================
# SECTION 6C.2E — LOAD REMAINING FEATURE-BUILDER HELPERS
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2E — LOAD REMAINING FEATURE-BUILDER HELPERS")
print("=" * 100)

import ast

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "54_tournament_strength_optimization_clean.py"
)

assert SCRIPT_PATH.exists()

source_text = SCRIPT_PATH.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source_text)

# Helpers already encountered + likely direct dependencies
TARGET_HELPERS = {
    "get_state_value",
    "get_card_name",
    "get_attached_energy",
    "get_pokemon_damage",
    "extract_move_name",
}

loaded_helpers = []

for node in tree.body:

    if (
        isinstance(node, ast.FunctionDef)
        and node.name in TARGET_HELPERS
    ):

        function_source = ast.get_source_segment(
            source_text,
            node,
        )

        safe_source = (
            "from __future__ import annotations\n\n"
            + function_source
        )

        exec(
            safe_source,
            globals(),
        )

        loaded_helpers.append(
            node.name
        )

print("\nHELPERS LOADED")
print("-" * 100)

for helper_name in sorted(
    loaded_helpers
):
    print(helper_name)

# ------------------------------------------------------------------------------------------
# Verify all expected helpers
# ------------------------------------------------------------------------------------------

missing_helpers = [
    helper_name
    for helper_name in TARGET_HELPERS
    if (
        helper_name not in globals()
        or not callable(
            globals()[helper_name]
        )
    )
]

print("\nMissing helpers:", missing_helpers)

assert not missing_helpers, (
    f"Missing helpers: {missing_helpers}"
)

print("\n✅ FEATURE-BUILDER HELPERS LOADED")
print("✅ SECTION 6C.2E COMPLETE")


# In[59]:


# ==========================================================================================
# SECTION 6C.2F — LOAD FULL FEATURE-BUILDER DEPENDENCY CLOSURE
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2F — LOAD FULL FEATURE-BUILDER DEPENDENCY CLOSURE")
print("=" * 100)

import ast

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "54_tournament_strength_optimization_clean.py"
)

TARGET_FUNCTION = "build_legality_aware_feature_row"

assert SCRIPT_PATH.exists()

source_text = SCRIPT_PATH.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source_text)

# ------------------------------------------------------------------------------------------
# Collect all top-level function definitions
# ------------------------------------------------------------------------------------------

function_nodes = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert TARGET_FUNCTION in function_nodes, (
    f"{TARGET_FUNCTION} not found."
)

# ------------------------------------------------------------------------------------------
# Recursively find functions called by the target function
# ------------------------------------------------------------------------------------------

def get_called_function_names(function_node):
    called = set()

    for node in ast.walk(function_node):

        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Name):
                called.add(node.func.id)

    return called


dependency_names = set()
queue = [TARGET_FUNCTION]

while queue:

    current_name = queue.pop()

    current_node = function_nodes.get(
        current_name
    )

    if current_node is None:
        continue

    called_names = get_called_function_names(
        current_node
    )

    for called_name in called_names:

        if (
            called_name in function_nodes
            and called_name not in dependency_names
            and called_name != TARGET_FUNCTION
        ):
            dependency_names.add(
                called_name
            )

            queue.append(
                called_name
            )

print("\nDEPENDENCIES DISCOVERED")
print("-" * 100)

for name in sorted(
    dependency_names
):
    print(name)

# ------------------------------------------------------------------------------------------
# Load dependencies first
# ------------------------------------------------------------------------------------------

for name in sorted(
    dependency_names
):

    node = function_nodes[name]

    function_source = ast.get_source_segment(
        source_text,
        node,
    )

    safe_source = (
        "from __future__ import annotations\n\n"
        + function_source
    )

    exec(
        safe_source,
        globals(),
    )

# ------------------------------------------------------------------------------------------
# Reload target builder last
# ------------------------------------------------------------------------------------------

target_node = function_nodes[
    TARGET_FUNCTION
]

target_source = ast.get_source_segment(
    source_text,
    target_node,
)

safe_target_source = (
    "from __future__ import annotations\n\n"
    + target_source
)

exec(
    safe_target_source,
    globals(),
)

# ------------------------------------------------------------------------------------------
# Validate
# ------------------------------------------------------------------------------------------

missing_dependencies = [
    name
    for name in dependency_names
    if (
        name not in globals()
        or not callable(
            globals()[name]
        )
    )
]

assert not missing_dependencies, (
    f"Dependencies still missing: "
    f"{missing_dependencies}"
)

assert TARGET_FUNCTION in globals()
assert callable(
    build_legality_aware_feature_row
)

print("\n✅ FULL FEATURE-BUILDER DEPENDENCY CLOSURE LOADED")
print(
    "✅ build_legality_aware_feature_row() RELOADED"
)
print(
    "✅ normalize_action_name AND RELATED HELPERS AVAILABLE"
)
print("✅ SECTION 6C.2F COMPLETE")


# In[61]:


# ==========================================================================================
# SECTION 6C.2G — LOAD STANDARD LIBRARY DEPENDENCY
# ==========================================================================================

import re

print("✅ re loaded")
print("✅ SECTION 6C.2G COMPLETE")


# In[64]:


# ==========================================================================================
# SECTION 6C.2H — RESTORE POLICY_ACTION_CLASSES FROM APPROVED R3 MODEL
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2H — RESTORE POLICY_ACTION_CLASSES FROM APPROVED R3 MODEL")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# The approved R3 classifier already contains the exact trained action classes.
# Use that frozen contract rather than depending on Notebook 54 runtime variables.
# ------------------------------------------------------------------------------------------

assert r3_live_model is not None
assert hasattr(
    r3_live_model,
    "classes_",
)

POLICY_ACTION_CLASSES = [
    str(action)
    for action in r3_live_model.classes_
]

assert len(
    POLICY_ACTION_CLASSES
) == 6, (
    f"Expected 6 policy action classes, "
    f"found {len(POLICY_ACTION_CLASSES)}."
)

assert len(
    POLICY_ACTION_CLASSES
) == len(
    set(POLICY_ACTION_CLASSES)
), (
    "Duplicate action classes detected."
)

print("\nPOLICY_ACTION_CLASSES")
print("-" * 100)

for index, action_name in enumerate(
    POLICY_ACTION_CLASSES
):
    print(
        f"{index}: {action_name}"
    )

print(
    "\nClass count:",
    len(POLICY_ACTION_CLASSES)
)

print(
    "\nSource: approved "
    "R3_STATE_CENTRIC_POLICY.classes_"
)

print("\n✅ POLICY_ACTION_CLASSES RESTORED")
print("✅ CLASS ORDER PRESERVED FROM FROZEN R3 MODEL")
print("✅ SECTION 6C.2H COMPLETE")


# In[69]:


# ==========================================================================================
# SECTION 6C.2I — LOAD REMAINING BUILDER GLOBAL DEPENDENCIES
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2I — LOAD REMAINING BUILDER GLOBAL DEPENDENCIES")
print("=" * 100)

import ast

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "54_tournament_strength_optimization_clean.py"
)

TARGET_FUNCTION = "build_legality_aware_feature_row"

assert SCRIPT_PATH.exists()

source_text = SCRIPT_PATH.read_text(
    encoding="utf-8",
    errors="ignore",
)

tree = ast.parse(source_text)

# ------------------------------------------------------------------------------------------
# Collect all top-level function definitions
# ------------------------------------------------------------------------------------------

function_nodes = {
    node.name: node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

assert TARGET_FUNCTION in function_nodes

# ------------------------------------------------------------------------------------------
# Find every bare-name reference used inside the target function
# ------------------------------------------------------------------------------------------

target_node = function_nodes[TARGET_FUNCTION]

referenced_names = set()

for node in ast.walk(target_node):

    if isinstance(node, ast.Name):
        referenced_names.add(
            node.id
        )

# ------------------------------------------------------------------------------------------
# Load any referenced top-level functions not already present
# ------------------------------------------------------------------------------------------

loaded_functions = []

for name in sorted(
    referenced_names
):

    if name not in function_nodes:
        continue

    if (
        name in globals()
        and callable(
            globals()[name]
        )
    ):
        continue

    function_source = ast.get_source_segment(
        source_text,
        function_nodes[name],
    )

    safe_source = (
        "from __future__ import annotations\n\n"
        + function_source
    )

    exec(
        safe_source,
        globals(),
    )

    loaded_functions.append(
        name
    )

# ------------------------------------------------------------------------------------------
# Recover simple referenced assignments/constants where possible
# ------------------------------------------------------------------------------------------

loaded_constants = []

for node in tree.body:

    assignment_name = None

    if isinstance(node, ast.Assign):

        simple_targets = [
            target.id
            for target in node.targets
            if isinstance(
                target,
                ast.Name,
            )
        ]

        if len(simple_targets) == 1:
            assignment_name = (
                simple_targets[0]
            )

    elif isinstance(
        node,
        ast.AnnAssign,
    ):

        if isinstance(
            node.target,
            ast.Name,
        ):
            assignment_name = (
                node.target.id
            )

    if assignment_name is None:
        continue

    if assignment_name not in referenced_names:
        continue

    if assignment_name in globals():
        continue

    # Only evaluate constants that can be safely literal-evaluated.
    value_node = getattr(
        node,
        "value",
        None,
    )

    if value_node is None:
        continue

    try:

        literal_value = ast.literal_eval(
            value_node
        )

    except Exception:
        continue

    globals()[
        assignment_name
    ] = literal_value

    loaded_constants.append(
        assignment_name
    )

# ------------------------------------------------------------------------------------------
# Keep POLICY_ACTION_CLASSES tied to the frozen R3 model
# ------------------------------------------------------------------------------------------

POLICY_ACTION_CLASSES = [
    str(value)
    for value in r3_live_model.classes_
]

# ------------------------------------------------------------------------------------------
# Reload the target function last
# ------------------------------------------------------------------------------------------

target_source = ast.get_source_segment(
    source_text,
    target_node,
)

safe_target_source = (
    "from __future__ import annotations\n\n"
    + target_source
)

exec(
    safe_target_source,
    globals(),
)

# ------------------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------------------

print("\nADDITIONAL FUNCTIONS LOADED")
print("-" * 100)

if loaded_functions:

    for name in loaded_functions:
        print(name)

else:
    print("None")

print("\nSIMPLE CONSTANTS LOADED")
print("-" * 100)

if loaded_constants:

    for name in loaded_constants:
        print(name)

else:
    print("None")

print("\nKEY DEPENDENCY CHECK")
print("-" * 100)

key_dependencies = [
    "get_state_value",
    "get_card_name",
    "get_attached_energy",
    "get_pokemon_damage",
    "extract_move_name",
    "normalize_action_name",
    "action_feature_name",
    "build_legality_aware_feature_row",
]

for name in key_dependencies:

    print(
        f"{name}:",
        name in globals(),
    )

assert (
    "action_feature_name"
    in globals()
), (
    "action_feature_name is still missing."
)

assert callable(
    action_feature_name
)

assert callable(
    build_legality_aware_feature_row
)

print(
    "\n✅ action_feature_name AVAILABLE"
)

print(
    "✅ REMAINING FUNCTION DEPENDENCIES LOADED"
)

print(
    "✅ BUILDER RELOADED"
)

print(
    "✅ SECTION 6C.2I COMPLETE"
)


# In[67]:


# ==========================================================================================
# SECTION 6C.2J — LOCATE action_feature_name EXACTLY
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2J — LOCATE action_feature_name EXACTLY")
print("=" * 100)

from pathlib import Path
import ast

SEARCH_NAME = "action_feature_name"

search_roots = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "src",
]

matches = []

for search_root in search_roots:

    if not search_root.exists():
        continue

    for file_path in search_root.rglob("*.py"):

        if "__pycache__" in file_path.parts:
            continue

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if SEARCH_NAME not in text:
            continue

        lines = text.splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            if SEARCH_NAME in line:

                start = max(
                    0,
                    line_number - 4,
                )

                end = min(
                    len(lines),
                    line_number + 5,
                )

                matches.append(
                    {
                        "file": str(file_path),
                        "line_number": line_number,
                        "line": line.strip(),
                        "context": "\n".join(
                            lines[start:end]
                        ),
                    }
                )

print("\nMATCHES")
print("-" * 100)

assert matches, (
    "action_feature_name was not found "
    "under scripts/ or src/."
)

for index, match in enumerate(
    matches,
    start=1,
):

    print()
    print("=" * 90)
    print(
        f"MATCH {index}"
    )
    print(
        match["file"]
    )
    print(
        f"Line {match['line_number']}"
    )
    print("=" * 90)

    print(
        match["context"]
    )

print()
print("✅ action_feature_name LOCATED")
print("✅ SECTION 6C.2J COMPLETE")


# In[68]:


# ==========================================================================================
# SECTION 6C.2K — LOAD EXACT action_feature_name FROM NOTEBOOK 53
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2K — LOAD EXACT action_feature_name FROM NOTEBOOK 53")
print("=" * 100)

from pathlib import Path
import ast


# ------------------------------------------------------------------------------------------
# 1. Exact verified source discovered in Section 6C.2J
# ------------------------------------------------------------------------------------------

ACTION_FEATURE_SOURCE_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "53_legality_aware_policy_optimization.py"
)

assert ACTION_FEATURE_SOURCE_FILE.exists(), (
    f"Notebook 53 script not found:\n"
    f"{ACTION_FEATURE_SOURCE_FILE}"
)

print("\nSource:")
print(ACTION_FEATURE_SOURCE_FILE)


# ------------------------------------------------------------------------------------------
# 2. Parse Notebook 53 source
# ------------------------------------------------------------------------------------------

source_text = ACTION_FEATURE_SOURCE_FILE.read_text(
    encoding="utf-8",
    errors="ignore",
)

parsed_source = ast.parse(
    source_text,
    filename=str(
        ACTION_FEATURE_SOURCE_FILE
    ),
)


# ------------------------------------------------------------------------------------------
# 3. Locate exact top-level action_feature_name definition
# ------------------------------------------------------------------------------------------

target_node = None

for node in parsed_source.body:

    if (
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and node.name == "action_feature_name"
    ):

        target_node = node
        break


assert target_node is not None, (
    "Exact top-level action_feature_name() "
    "definition was not found in Notebook 53."
)


# ------------------------------------------------------------------------------------------
# 4. Extract exact function source
# ------------------------------------------------------------------------------------------

exact_function_source = ast.get_source_segment(
    source_text,
    target_node,
)

if exact_function_source is None:

    exact_function_source = ast.unparse(
        target_node
    )


print("\nEXACT FUNCTION RECOVERED")
print("-" * 100)
print(exact_function_source)


# ------------------------------------------------------------------------------------------
# 5. Execute exact function definition in current Notebook 59 namespace
# ------------------------------------------------------------------------------------------

compiled_function = compile(
    ast.Module(
        body=[
            target_node
        ],
        type_ignores=[],
    ),
    filename=str(
        ACTION_FEATURE_SOURCE_FILE
    ),
    mode="exec",
)

exec(
    compiled_function,
    globals(),
    globals(),
)


# ------------------------------------------------------------------------------------------
# 6. Validate binding
# ------------------------------------------------------------------------------------------

assert (
    "action_feature_name"
    in globals()
), (
    "action_feature_name was not bound "
    "into Notebook 59."
)

assert callable(
    action_feature_name
), (
    "action_feature_name exists but "
    "is not callable."
)


# ------------------------------------------------------------------------------------------
# 7. Smoke test against known policy actions if available
# ------------------------------------------------------------------------------------------

print("\nFUNCTION TEST")
print("-" * 100)

if (
    "POLICY_ACTION_CLASSES"
    in globals()
):

    test_actions = list(
        POLICY_ACTION_CLASSES
    )

else:

    test_actions = [
        "Ascension",
        "Bind Down",
        "Live Coal",
        "Pass",
        "Quick Attack",
    ]


test_results = []

for action_name in test_actions:

    suffix = action_feature_name(
        action_name
    )

    test_results.append(
        {
            "action_name":
                action_name,

            "feature_suffix":
                suffix,

            "is_string":
                isinstance(
                    suffix,
                    str,
                ),

            "non_empty":
                bool(
                    str(
                        suffix
                    ).strip()
                ),
        }
    )


section6c2k_validation_df = pd.DataFrame(
    test_results
)

display(
    section6c2k_validation_df
)


# ------------------------------------------------------------------------------------------
# 8. Final assertions
# ------------------------------------------------------------------------------------------

assert (
    section6c2k_validation_df[
        "is_string"
    ].all()
)

assert (
    section6c2k_validation_df[
        "non_empty"
    ].all()
)


print()
print("✅ EXACT NOTEBOOK 53 action_feature_name() LOADED")
print("✅ FUNCTION IS CALLABLE")
print("✅ ACTION FEATURE SUFFIX TEST PASSED")
print("✅ NO FALLBACK IMPLEMENTATION USED")
print("✅ SECTION 6C.2K COMPLETE")


# In[71]:


# ==========================================================================================
# SECTION 6C.2L — RECOVER LEGALITY FEATURE SCHEMA CONSTANTS
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2L — RECOVER LEGALITY FEATURE SCHEMA CONSTANTS")
print("=" * 100)

from pathlib import Path
import ast


# ------------------------------------------------------------------------------------------
# 1. Use authoritative Notebook 53 source
# ------------------------------------------------------------------------------------------

SOURCE_FILE_6C2L = (
    PROJECT_ROOT
    / "scripts"
    / "53_legality_aware_policy_optimization.py"
)

assert SOURCE_FILE_6C2L.exists(), (
    f"Notebook 53 source not found:\n"
    f"{SOURCE_FILE_6C2L}"
)

source_text_6c2l = SOURCE_FILE_6C2L.read_text(
    encoding="utf-8",
    errors="ignore",
)

source_tree_6c2l = ast.parse(
    source_text_6c2l,
    filename=str(SOURCE_FILE_6C2L),
)


# ------------------------------------------------------------------------------------------
# 2. Constants required by the legality-aware feature builder
# ------------------------------------------------------------------------------------------

TARGET_NAMES_6C2L = [
    "legality_training_numeric_features",
    "legality_training_categorical_features",
    "PREPROCESS_NUMERIC_FEATURES",
    "PREPROCESS_CATEGORICAL_FEATURES",
    "PREPROCESS_RAW_FEATURES",
]


# ------------------------------------------------------------------------------------------
# 3. Locate assignments in Notebook 53
# ------------------------------------------------------------------------------------------

assignment_nodes_6c2l = {}

for node in source_tree_6c2l.body:

    assigned_names = []

    if isinstance(node, ast.Assign):

        for target in node.targets:

            if isinstance(target, ast.Name):
                assigned_names.append(
                    target.id
                )

    elif isinstance(node, ast.AnnAssign):

        if isinstance(node.target, ast.Name):
            assigned_names.append(
                node.target.id
            )

    for assigned_name in assigned_names:

        if (
            assigned_name
            in TARGET_NAMES_6C2L
            and assigned_name
            not in assignment_nodes_6c2l
        ):
            assignment_nodes_6c2l[
                assigned_name
            ] = node


# ------------------------------------------------------------------------------------------
# 4. Report what was found
# ------------------------------------------------------------------------------------------

print("\nSCHEMA CONSTANT SEARCH")
print("-" * 100)

for target_name in TARGET_NAMES_6C2L:

    print(
        f"{target_name}:",
        target_name in assignment_nodes_6c2l,
    )


assert (
    "legality_training_numeric_features"
    in assignment_nodes_6c2l
), (
    "Could not locate legality_training_numeric_features "
    "in Notebook 53."
)


# ------------------------------------------------------------------------------------------
# 5. Dependency-aware execution
#
# Some constants may depend on constants defined immediately before them.
# We therefore make several passes over Notebook 53's top-level assignments,
# executing only safe/simple assignments.
# ------------------------------------------------------------------------------------------

loaded_names_6c2l = set()

SAFE_NODE_TYPES_6C2L = (
    ast.Assign,
    ast.AnnAssign,
)


for pass_number in range(10):

    progress = False

    for node in source_tree_6c2l.body:

        if not isinstance(
            node,
            SAFE_NODE_TYPES_6C2L,
        ):
            continue

        assigned_names = []

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(target, ast.Name):
                    assigned_names.append(
                        target.id
                    )

        else:

            if isinstance(node.target, ast.Name):
                assigned_names.append(
                    node.target.id
                )

        if not assigned_names:
            continue

        # Only execute assignments relevant to preprocessing /
        # legality feature schemas.
        relevant = any(
            (
                name in TARGET_NAMES_6C2L
                or
                "feature" in name.lower()
                or
                "preprocess" in name.lower()
            )
            for name in assigned_names
        )

        if not relevant:
            continue

        try:

            compiled_assignment = compile(
                ast.Module(
                    body=[node],
                    type_ignores=[],
                ),
                filename=str(
                    SOURCE_FILE_6C2L
                ),
                mode="exec",
            )

            exec(
                compiled_assignment,
                globals(),
                globals(),
            )

            for name in assigned_names:

                if name in globals():

                    if name not in loaded_names_6c2l:
                        loaded_names_6c2l.add(
                            name
                        )
                        progress = True

        except (
            NameError,
            AttributeError,
            TypeError,
            KeyError,
        ):

            # Dependency not available yet.
            # Another pass may resolve it.
            continue

    if not progress:
        break


# ------------------------------------------------------------------------------------------
# 6. Verify required schemas
# ------------------------------------------------------------------------------------------

print("\nREQUIRED SCHEMA CHECK")
print("-" * 100)

schema_check_rows_6c2l = []

for target_name in TARGET_NAMES_6C2L:

    exists = (
        target_name
        in globals()
    )

    value = (
        globals().get(
            target_name
        )
    )

    if exists:

        try:
            length = len(value)
        except Exception:
            length = None

    else:
        length = None

    schema_check_rows_6c2l.append(
        {
            "name":
                target_name,

            "loaded":
                exists,

            "length":
                length,

            "type":
                (
                    type(value).__name__
                    if exists
                    else None
                ),
        }
    )


section6c2l_schema_df = pd.DataFrame(
    schema_check_rows_6c2l
)

display(
    section6c2l_schema_df
)


# ------------------------------------------------------------------------------------------
# 7. Hard requirements for current builder
# ------------------------------------------------------------------------------------------

assert (
    "legality_training_numeric_features"
    in globals()
), (
    "legality_training_numeric_features "
    "was not recovered."
)

assert isinstance(
    legality_training_numeric_features,
    (
        list,
        tuple,
    ),
), (
    "legality_training_numeric_features "
    "has unexpected type."
)


print("\nNUMERIC LEGALITY FEATURES")
print("-" * 100)

for index, feature_name in enumerate(
    legality_training_numeric_features,
    start=1,
):

    print(
        f"{index:02d}. {feature_name}"
    )


# ------------------------------------------------------------------------------------------
# 8. Confirm critical previously recovered dependencies remain intact
# ------------------------------------------------------------------------------------------

critical_dependencies_6c2l = [
    "build_legality_aware_feature_row",
    "get_state_value",
    "get_card_name",
    "get_attached_energy",
    "get_pokemon_damage",
    "extract_move_name",
    "normalize_action_name",
    "action_feature_name",
    "POLICY_ACTION_CLASSES",
    "training_active_card_action_map",
]

print("\nBUILDER DEPENDENCY STATUS")
print("-" * 100)

for name in critical_dependencies_6c2l:

    print(
        f"{name}:",
        name in globals(),
    )


assert callable(
    build_legality_aware_feature_row
)

assert callable(
    action_feature_name
)

assert (
    "POLICY_ACTION_CLASSES"
    in globals()
)

assert (
    "training_active_card_action_map"
    in globals()
)


print()
print("✅ legality_training_numeric_features RECOVERED")
print("✅ LEGALITY FEATURE SCHEMA RECOVERY EXECUTED")
print("✅ EXISTING BUILDER DEPENDENCIES PRESERVED")
print("✅ NOTEBOOK 53 SOURCE REMAINS AUTHORITATIVE")
print("✅ SECTION 6C.2L COMPLETE")


# In[76]:


# ==========================================================================================
# SECTION 6C.2M — RESTORE EXACT 41-COLUMN NOTEBOOK 54 RAW FEATURE SCHEMA
# ==========================================================================================

print("=" * 100)
print("SECTION 6C.2M — RESTORE EXACT 41-COLUMN NOTEBOOK 54 RAW FEATURE SCHEMA")
print("=" * 100)

# ------------------------------------------------------------------------------------------
# 1. Validate authoritative dependencies already recovered
# ------------------------------------------------------------------------------------------

assert "POLICY_ACTION_CLASSES" in globals()
assert len(POLICY_ACTION_CLASSES) == 6

assert "action_feature_name" in globals()
assert callable(action_feature_name)

assert "legality_training_numeric_features" in globals()
assert "legality_training_categorical_features" in globals()

print("\nCURRENT BASE FEATURE COUNTS")
print("-" * 100)

print(
    "Base numeric features:",
    len(legality_training_numeric_features),
)

print(
    "Base categorical features:",
    len(legality_training_categorical_features),
)

# Notebook 54 contract:
# 19 base numeric + 12 action indicators = 31 numeric
# 10 categorical
# 31 + 10 = 41 raw features

assert len(
    legality_training_numeric_features
) == 19, (
    "Expected 19 base numeric legality features."
)

assert len(
    legality_training_categorical_features
) == 10, (
    "Expected 10 categorical legality features."
)

# ==========================================================================================
# 2. Reconstruct the exact 12 action-indicator features
# ==========================================================================================

section4vb_action_indicator_features = []

for action_name in POLICY_ACTION_CLASSES:

    suffix = action_feature_name(
        action_name
    )

    section4vb_action_indicator_features.extend(
        [
            "is_legal__" + suffix,
            "active_card_can_use__" + suffix,
        ]
    )

print("\nACTION INDICATOR FEATURES")
print("-" * 100)

for index, feature_name in enumerate(
    section4vb_action_indicator_features,
    start=1,
):
    print(
        f"{index:02d}. {feature_name}"
    )

assert len(
    section4vb_action_indicator_features
) == 12, (
    "Expected exactly 12 action-indicator features."
)

assert len(
    section4vb_action_indicator_features
) == len(
    set(section4vb_action_indicator_features)
), (
    "Duplicate action-indicator features detected."
)

# ==========================================================================================
# 3. Reconstruct Notebook 54 preprocessing schemas EXACTLY
# ==========================================================================================

PREPROCESS_NUMERIC_FEATURES = (
    list(
        legality_training_numeric_features
    )
    +
    list(
        section4vb_action_indicator_features
    )
)

PREPROCESS_CATEGORICAL_FEATURES = list(
    legality_training_categorical_features
)

PREPROCESS_RAW_FEATURES = (
    PREPROCESS_NUMERIC_FEATURES
    +
    PREPROCESS_CATEGORICAL_FEATURES
)

# ==========================================================================================
# 4. Hard validation against Notebook 54 contract
# ==========================================================================================

print("\nRECONSTRUCTED FEATURE COUNTS")
print("-" * 100)

print(
    "Base numeric features       :",
    len(legality_training_numeric_features),
)

print(
    "Action indicator features   :",
    len(section4vb_action_indicator_features),
)

print(
    "Preprocess numeric features :",
    len(PREPROCESS_NUMERIC_FEATURES),
)

print(
    "Categorical features        :",
    len(PREPROCESS_CATEGORICAL_FEATURES),
)

print(
    "Total raw features          :",
    len(PREPROCESS_RAW_FEATURES),
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

assert len(
    PREPROCESS_RAW_FEATURES
) == len(
    set(PREPROCESS_RAW_FEATURES)
)

# ==========================================================================================
# 5. Ensure builder sees repaired globals
# ==========================================================================================

build_legality_aware_feature_row.__globals__.update(
    {
        "POLICY_ACTION_CLASSES":
            POLICY_ACTION_CLASSES,

        "action_feature_name":
            action_feature_name,

        "training_active_card_action_map":
            training_active_card_action_map,

        "legality_training_numeric_features":
            legality_training_numeric_features,

        "legality_training_categorical_features":
            legality_training_categorical_features,

        "section4vb_action_indicator_features":
            section4vb_action_indicator_features,

        "PREPROCESS_NUMERIC_FEATURES":
            PREPROCESS_NUMERIC_FEATURES,

        "PREPROCESS_CATEGORICAL_FEATURES":
            PREPROCESS_CATEGORICAL_FEATURES,

        "PREPROCESS_RAW_FEATURES":
            PREPROCESS_RAW_FEATURES,
    }
)

# ==========================================================================================
# 6. Final schema report
# ==========================================================================================

section6c2m_schema_validation_df = pd.DataFrame(
    [
        {
            "schema_component":
                "base_numeric",

            "count":
                len(
                    legality_training_numeric_features
                ),

            "expected":
                19,

            "passed":
                len(
                    legality_training_numeric_features
                ) == 19,
        },

        {
            "schema_component":
                "action_indicators",

            "count":
                len(
                    section4vb_action_indicator_features
                ),

            "expected":
                12,

            "passed":
                len(
                    section4vb_action_indicator_features
                ) == 12,
        },

        {
            "schema_component":
                "preprocess_numeric",

            "count":
                len(
                    PREPROCESS_NUMERIC_FEATURES
                ),

            "expected":
                31,

            "passed":
                len(
                    PREPROCESS_NUMERIC_FEATURES
                ) == 31,
        },

        {
            "schema_component":
                "preprocess_categorical",

            "count":
                len(
                    PREPROCESS_CATEGORICAL_FEATURES
                ),

            "expected":
                10,

            "passed":
                len(
                    PREPROCESS_CATEGORICAL_FEATURES
                ) == 10,
        },

        {
            "schema_component":
                "raw_total",

            "count":
                len(
                    PREPROCESS_RAW_FEATURES
                ),

            "expected":
                41,

            "passed":
                len(
                    PREPROCESS_RAW_FEATURES
                ) == 41,
        },
    ]
)

print("\nRAW FEATURE SCHEMA VALIDATION")
print("-" * 100)

display(
    section6c2m_schema_validation_df
)

assert section6c2m_schema_validation_df[
    "passed"
].all()

print()
print("✅ 12 ACTION-INDICATOR FEATURES RESTORED")
print("✅ 31 NUMERIC FEATURES RESTORED")
print("✅ 10 CATEGORICAL FEATURES VERIFIED")
print("✅ EXACT 41-COLUMN RAW FEATURE SCHEMA RESTORED")
print("✅ BUILDER GLOBAL SCHEMA UPDATED")
print("✅ SECTION 6C.2M COMPLETE")


# In[81]:


# ==========================================================================================
# SECTION 6D — BUILD R3 LIVE POLICY ENGINE
# ==========================================================================================

print("=" * 100)
print("SECTION 6D — BUILD R3 LIVE POLICY ENGINE")
print("=" * 100)

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd


# ==========================================================================================
# 1. Validate dependencies
# ==========================================================================================

required_dependencies_6d = [
    "build_legality_aware_feature_row",
    "encode_r3_feature_row",
    "r3_live_model",
    "r3_live_feature_names",
    "extract_move_name",
    "normalize_action_name",
]

missing_dependencies_6d = [
    name
    for name in required_dependencies_6d
    if name not in globals()
]

assert not missing_dependencies_6d, (
    "Missing Section 6D dependencies:\n"
    f"{missing_dependencies_6d}"
)

assert callable(
    build_legality_aware_feature_row
)

assert callable(
    encode_r3_feature_row
)

assert callable(
    extract_move_name
)

assert callable(
    normalize_action_name
)

assert hasattr(
    r3_live_model,
    "predict_proba",
)

assert len(
    r3_live_feature_names
) == 46


# ==========================================================================================
# 2. R3 live policy engine
# ==========================================================================================

class R3LivePolicyEngine:
    """
    Live legality-aware wrapper around the approved
    R3_STATE_CENTRIC_POLICY RandomForestClassifier.

    Contract intentionally mirrors the existing
    CompetitivePolicyEngine.score_moves() interface.
    """

    def __init__(
        self,
        model,
        retained_feature_names,
        *,
        name: str = "R3_STATE_CENTRIC_POLICY",
    ) -> None:

        self.model = model

        self.retained_feature_names = list(
            retained_feature_names
        )

        self.name = str(
            name
        )

        self.checkpoint_epoch = 0

        self.classes_ = [
            str(value)
            for value in self.model.classes_
        ]

        assert len(
            self.retained_feature_names
        ) == 46

        assert int(
            self.model.n_features_in_
        ) == 46

        assert len(
            self.classes_
        ) > 0

        # --------------------------------------------------------------
        # Normalized class-name lookup
        # --------------------------------------------------------------

        self._normalized_class_lookup = {
            normalize_action_name(
                class_name
            ):
            class_name

            for class_name
            in self.classes_
        }


    # ======================================================================================
    # Build model-ready 46-feature matrix
    # ======================================================================================

    def build_model_matrix(
        self,
        battle_state: Any,
        legal_moves: Sequence[Any],
    ) -> pd.DataFrame:

        legal_moves = list(
            legal_moves
        )

        if not legal_moves:

            raise ValueError(
                "At least one legal move is required."
            )

        # --------------------------------------------------------------
        # 41-column production raw feature row
        # --------------------------------------------------------------

        raw_feature_row = (
            build_legality_aware_feature_row(
                battle_state,
                legal_moves,
            )
        )

        assert isinstance(
            raw_feature_row,
            pd.DataFrame,
        )

        assert raw_feature_row.shape == (
            1,
            41,
        ), (
            "Unexpected raw feature shape: "
            f"{raw_feature_row.shape}; "
            "expected (1, 41)."
        )

        # --------------------------------------------------------------
        # Exact 46-column R3 matrix
        # --------------------------------------------------------------

        encoded_feature_row = (
            encode_r3_feature_row(
                raw_feature_row
            )
        )

        assert isinstance(
            encoded_feature_row,
            pd.DataFrame,
        )

        assert encoded_feature_row.shape == (
            1,
            46,
        ), (
            "Unexpected R3 feature shape: "
            f"{encoded_feature_row.shape}; "
            "expected (1, 46)."
        )

        assert list(
            encoded_feature_row.columns
        ) == self.retained_feature_names, (
            "R3 encoded feature order does not match "
            "the frozen retained-feature contract."
        )

        assert np.isfinite(
            encoded_feature_row.to_numpy(
                dtype=float
            )
        ).all(), (
            "R3 feature matrix contains non-finite values."
        )

        return encoded_feature_row


    # ======================================================================================
    # Score legal moves
    # ======================================================================================

    def score_moves(
        self,
        battle_state: Any,
        legal_moves: Sequence[Any],
    ) -> dict[str, Any]:

        legal_moves = list(
            legal_moves
        )

        if not legal_moves:

            raise ValueError(
                "At least one legal move is required."
            )

        # --------------------------------------------------------------
        # Build R3 feature matrix
        # --------------------------------------------------------------

        model_matrix = self.build_model_matrix(
            battle_state=battle_state,
            legal_moves=legal_moves,
        )

        # --------------------------------------------------------------
        # Full six-class RF probability distribution
        # --------------------------------------------------------------

        full_probability_matrix = (
            self.model.predict_proba(
                model_matrix
            )
        )

        assert full_probability_matrix.shape == (
            1,
            len(self.classes_),
        )

        full_probabilities = np.asarray(
            full_probability_matrix[0],
            dtype=float,
        )

        assert np.isfinite(
            full_probabilities
        ).all()

        assert np.isclose(
            full_probabilities.sum(),
            1.0,
            atol=1e-9,
        )

        # --------------------------------------------------------------
        # Class probability lookup
        # --------------------------------------------------------------

        class_probability_lookup = {}

        for class_name, probability in zip(
            self.classes_,
            full_probabilities,
        ):

            normalized_name = (
                normalize_action_name(
                    class_name
                )
            )

            class_probability_lookup[
                normalized_name
            ] = float(
                probability
            )

        # --------------------------------------------------------------
        # Resolve actual legal move names
        # --------------------------------------------------------------

        legal_move_names = [
            str(
                extract_move_name(
                    move
                )
            )
            for move in legal_moves
        ]

        normalized_legal_names = [
            normalize_action_name(
                move_name
            )
            for move_name in legal_move_names
        ]

        # --------------------------------------------------------------
        # Mask model probabilities to legal actions only
        # --------------------------------------------------------------

        legal_probability_values = np.asarray(
            [
                class_probability_lookup.get(
                    normalized_name,
                    0.0,
                )
                for normalized_name
                in normalized_legal_names
            ],
            dtype=float,
        )

        legal_probability_mass = float(
            legal_probability_values.sum()
        )

        fallback = False

        if legal_probability_mass > 0.0:

            legal_probabilities = (
                legal_probability_values
                /
                legal_probability_mass
            )

        else:

            # ----------------------------------------------------------
            # Defensive fallback:
            # model classes could not be mapped to available moves.
            #
            # Preserve legality by selecting first legal move.
            # ----------------------------------------------------------

            fallback = True

            legal_probabilities = np.zeros(
                len(
                    legal_moves
                ),
                dtype=float,
            )

            legal_probabilities[0] = 1.0

        # --------------------------------------------------------------
        # Select best LEGAL move
        # --------------------------------------------------------------

        selected_index = int(
            np.argmax(
                legal_probabilities
            )
        )

        selected_move = legal_moves[
            selected_index
        ]

        selected_move_name = (
            legal_move_names[
                selected_index
            ]
        )

        confidence = float(
            legal_probabilities[
                selected_index
            ]
        )

        # --------------------------------------------------------------
        # Compatibility logits
        #
        # RandomForestClassifier does not expose neural-network logits.
        # Log-probabilities provide a deterministic score vector with
        # the same ordering as the legal probabilities.
        # --------------------------------------------------------------

        epsilon = 1e-12

        legal_logits = np.log(
            np.clip(
                legal_probabilities,
                epsilon,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Return existing CompetitivePolicyEngine-compatible contract
        # --------------------------------------------------------------

        return {
            "selected_index":
                selected_index,

            "selected_move":
                selected_move,

            "selected_move_name":
                selected_move_name,

            "confidence":
                confidence,

            "probabilities":
                [
                    float(value)
                    for value
                    in legal_probabilities
                ],

            "logits":
                [
                    float(value)
                    for value
                    in legal_logits
                ],

            "move_names":
                legal_move_names,

            "checkpoint_epoch":
                self.checkpoint_epoch,

            "fallback":
                fallback,

            # Additional diagnostics — harmless to existing callers
            "policy_name":
                self.name,

            "full_class_names":
                list(
                    self.classes_
                ),

            "full_class_probabilities":
                [
                    float(value)
                    for value
                    in full_probabilities
                ],

            "legal_probability_mass":
                legal_probability_mass,
        }


    # ======================================================================================
    # Convenience interface matching existing engine
    # ======================================================================================

    def choose_move(
        self,
        battle_state: Any,
        legal_moves: Sequence[Any],
    ) -> Any:

        result = self.score_moves(
            battle_state=battle_state,
            legal_moves=legal_moves,
        )

        return result[
            "selected_move"
        ]


# ==========================================================================================
# 3. Instantiate approved live R3 engine
# ==========================================================================================

r3_live_policy_engine = (
    R3LivePolicyEngine(
        model=r3_live_model,
        retained_feature_names=
            r3_live_feature_names,
    )
)

print("\nR3 LIVE POLICY ENGINE")
print("-" * 100)

print(
    "Type           :",
    type(
        r3_live_policy_engine
    ).__name__,
)

print(
    "Policy         :",
    r3_live_policy_engine.name,
)

print(
    "Feature count  :",
    len(
        r3_live_policy_engine
        .retained_feature_names
    ),
)

print(
    "Class count    :",
    len(
        r3_live_policy_engine
        .classes_
    ),
)


# ==========================================================================================
# 4. Smoke-test score_moves() using proven Section 6C test state
# ==========================================================================================

section6d_result = (
    r3_live_policy_engine.score_moves(
        battle_state=
            section6c2_test_state,

        legal_moves=
            section6c2_legal_moves,
    )
)


# ==========================================================================================
# 5. Inspect score_moves contract
# ==========================================================================================

expected_contract_keys_6d = [
    "selected_index",
    "selected_move",
    "selected_move_name",
    "confidence",
    "probabilities",
    "logits",
    "move_names",
    "checkpoint_epoch",
    "fallback",
]

missing_contract_keys_6d = [
    key
    for key in expected_contract_keys_6d
    if key not in section6d_result
]

assert not missing_contract_keys_6d, (
    "R3 engine contract missing keys:\n"
    f"{missing_contract_keys_6d}"
)

print("\nR3 SCORE_MOVES RESULT")
print("-" * 100)

for key in expected_contract_keys_6d:

    print(
        f"{key}:",
        section6d_result[
            key
        ],
    )


# ==========================================================================================
# 6. Legal move validation
# ==========================================================================================

section6d_move_names = (
    section6d_result[
        "move_names"
    ]
)

section6d_probabilities = np.asarray(
    section6d_result[
        "probabilities"
    ],
    dtype=float,
)

section6d_logits = np.asarray(
    section6d_result[
        "logits"
    ],
    dtype=float,
)

assert len(
    section6d_move_names
) == len(
    section6c2_legal_moves
)

assert len(
    section6d_probabilities
) == len(
    section6c2_legal_moves
)

assert len(
    section6d_logits
) == len(
    section6c2_legal_moves
)

assert np.isclose(
    section6d_probabilities.sum(),
    1.0,
    atol=1e-9,
)

assert (
    section6d_result[
        "selected_move_name"
    ]
    in
    section6d_move_names
)

assert (
    0
    <=
    section6d_result[
        "selected_index"
    ]
    <
    len(
        section6c2_legal_moves
    )
)


# ==========================================================================================
# 7. Validation report
# ==========================================================================================

section6d_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "engine_instantiated",

            "passed":
                isinstance(
                    r3_live_policy_engine,
                    R3LivePolicyEngine,
                ),
        },

        {
            "validation":
                "score_moves_contract_complete",

            "passed":
                len(
                    missing_contract_keys_6d
                ) == 0,
        },

        {
            "validation":
                "legal_probability_count",

            "passed":
                len(
                    section6d_probabilities
                )
                ==
                len(
                    section6c2_legal_moves
                ),
        },

        {
            "validation":
                "legal_probabilities_sum_to_one",

            "passed":
                np.isclose(
                    section6d_probabilities.sum(),
                    1.0,
                    atol=1e-9,
                ),
        },

        {
            "validation":
                "selected_move_is_legal",

            "passed":
                (
                    section6d_result[
                        "selected_move_name"
                    ]
                    in
                    section6d_move_names
                ),
        },

        {
            "validation":
                "fallback_not_required",

            "passed":
                not bool(
                    section6d_result[
                        "fallback"
                    ]
                ),
        },
    ]
)

print("\nR3 LIVE POLICY ENGINE VALIDATION")
print("-" * 100)

display(
    section6d_validation_df
)

assert section6d_validation_df[
    "passed"
].all()


# ==========================================================================================
# 8. Final result
# ==========================================================================================

print("\n" + "=" * 100)

print("✅ R3 LIVE POLICY ENGINE CREATED")
print("✅ EXACT 41 → 46 FEATURE PIPELINE CONNECTED")
print("✅ SIX-CLASS R3 PROBABILITIES GENERATED")
print("✅ PROBABILITIES MASKED TO ACTUAL LEGAL MOVES")
print("✅ LEGAL PROBABILITIES RENORMALIZED")
print("✅ SELECTED MOVE IS LEGAL")
print("✅ EXISTING score_moves() CONTRACT PRESERVED")
print("✅ SECTION 6D COMPLETE")

print("=" * 100)


# In[84]:


# ==========================================================================================
# SECTION 7A — INSPECT LIVE SIMULATOR ENTRY CONTRACT
# PACKAGE-AWARE VERSION
# ==========================================================================================

print("=" * 100)
print("SECTION 7A — INSPECT LIVE SIMULATOR ENTRY CONTRACT")
print("=" * 100)

import inspect
import sys
from pathlib import Path


# ------------------------------------------------------------------------------------------
# 1. Resolve project and src directories
# ------------------------------------------------------------------------------------------

SRC_ROOT_7A = (
    PROJECT_ROOT
    / "src"
)

assert SRC_ROOT_7A.exists(), (
    f"src directory not found:\n"
    f"{SRC_ROOT_7A}"
)

assert (
    SRC_ROOT_7A
    / "__init__.py"
).exists(), (
    "src/__init__.py is missing, so src is not currently a package."
)


print("\nSRC ROOT")
print("-" * 100)
print(SRC_ROOT_7A)


# ------------------------------------------------------------------------------------------
# 2. Add PROJECT ROOT to sys.path
#
# IMPORTANT:
# We add PROJECT_ROOT, not src itself.
# That allows:
#
#     import src.battle_simulation
#
# and preserves relative imports inside battle_simulation.py.
# ------------------------------------------------------------------------------------------

PROJECT_ROOT_STRING_7A = str(
    PROJECT_ROOT
)

if PROJECT_ROOT_STRING_7A not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT_STRING_7A,
    )


print("\nPROJECT ROOT ON SYS.PATH")
print("-" * 100)

print(
    PROJECT_ROOT_STRING_7A
    in sys.path
)


# ------------------------------------------------------------------------------------------
# 3. Import src package first
# ------------------------------------------------------------------------------------------

import src

print("\nSRC PACKAGE")
print("-" * 100)

print(
    "src package loaded:",
    src is not None,
)

print(
    "src package file:",
    getattr(
        src,
        "__file__",
        None,
    ),
)


# ------------------------------------------------------------------------------------------
# 4. Import simulator module normally
# ------------------------------------------------------------------------------------------

from src.battle_simulation import (
    simulate_ai_battle,
    determine_battle_winner,
    create_battle_transcript,
    BattleTurnRecord,
    BattleSimulationResult,
)


# ------------------------------------------------------------------------------------------
# 5. Validate imported objects
# ------------------------------------------------------------------------------------------

assert callable(
    simulate_ai_battle
)

assert callable(
    determine_battle_winner
)

assert callable(
    create_battle_transcript
)

assert BattleTurnRecord is not None
assert BattleSimulationResult is not None


# ==========================================================================================
# 6. Inspect simulator contract
# ==========================================================================================

print("\nSIMULATE_AI_BATTLE SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        simulate_ai_battle
    )
)


print("\nSIMULATE_AI_BATTLE SOURCE")
print("-" * 100)

print(
    inspect.getsource(
        simulate_ai_battle
    )
)


print("\nDETERMINE_BATTLE_WINNER SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        determine_battle_winner
    )
)


print("\nCREATE_BATTLE_TRANSCRIPT SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        create_battle_transcript
    )
)


# ==========================================================================================
# 7. Result types
# ==========================================================================================

print("\nLIVE BATTLE RESULT TYPES")
print("-" * 100)

print(
    "BattleTurnRecord:",
    BattleTurnRecord,
)

print(
    "BattleSimulationResult:",
    BattleSimulationResult,
)


# ==========================================================================================
# 8. Final validation
# ==========================================================================================

section7a_validation_df = pd.DataFrame(
    [
        {
            "component":
                "src package",

            "available":
                src is not None,
        },

        {
            "component":
                "simulate_ai_battle",

            "available":
                callable(
                    simulate_ai_battle
                ),
        },

        {
            "component":
                "determine_battle_winner",

            "available":
                callable(
                    determine_battle_winner
                ),
        },

        {
            "component":
                "create_battle_transcript",

            "available":
                callable(
                    create_battle_transcript
                ),
        },

        {
            "component":
                "BattleTurnRecord",

            "available":
                BattleTurnRecord
                is not None,
        },

        {
            "component":
                "BattleSimulationResult",

            "available":
                BattleSimulationResult
                is not None,
        },
    ]
)

print("\nSIMULATOR CONTRACT VALIDATION")
print("-" * 100)

display(
    section7a_validation_df
)

assert section7a_validation_df[
    "available"
].all()


print()
print("✅ src PACKAGE IMPORTED")
print("✅ RELATIVE IMPORT CONTEXT PRESERVED")
print("✅ simulate_ai_battle() RECOVERED")
print("✅ LIVE SIMULATOR CONTRACT INSPECTED")
print("✅ SECTION 7A COMPLETE")


# In[85]:


# ==========================================================================================
# SECTION 7B — EXTRACT SIMULATOR / AGENT HANDSHAKE
# ==========================================================================================

print("=" * 100)
print("SECTION 7B — EXTRACT SIMULATOR / AGENT HANDSHAKE")
print("=" * 100)

import inspect

# ------------------------------------------------------------------------------------------
# 1. Exact signature
# ------------------------------------------------------------------------------------------

simulation_signature_7b = inspect.signature(
    simulate_ai_battle
)

print("\nSIMULATE_AI_BATTLE SIGNATURE")
print("-" * 100)

print(
    simulation_signature_7b
)

# ------------------------------------------------------------------------------------------
# 2. Read source and show only lines relevant to agent execution
# ------------------------------------------------------------------------------------------

simulation_source_7b = inspect.getsource(
    simulate_ai_battle
)

source_lines_7b = (
    simulation_source_7b
    .splitlines()
)

important_terms_7b = [
    "agent.",
    "choose_move",
    "select_move",
    "legal",
    "apply_move",
    "current_state",
    "next_state",
]

important_lines_7b = []

for line_number, line in enumerate(
    source_lines_7b,
    start=1,
):

    if any(
        term in line
        for term in important_terms_7b
    ):

        important_lines_7b.append(
            {
                "line_number": line_number,
                "source": line,
            }
        )

section7b_handshake_df = pd.DataFrame(
    important_lines_7b
)

print("\nSIMULATOR / AGENT HANDSHAKE LINES")
print("-" * 100)

display(
    section7b_handshake_df
)

# ------------------------------------------------------------------------------------------
# 3. Inspect existing PokemonBattleAgent interface for comparison
# ------------------------------------------------------------------------------------------

from src.battle_agent import (
    PokemonBattleAgent,
)

print("\nEXISTING PokemonBattleAgent.choose_move SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        PokemonBattleAgent.choose_move
    )
)

print("\nEXISTING PokemonBattleAgent.choose_move SOURCE")
print("-" * 100)

print(
    inspect.getsource(
        PokemonBattleAgent.choose_move
    )
)

# ------------------------------------------------------------------------------------------
# 4. Inspect current R3 engine method
# ------------------------------------------------------------------------------------------

print("\nCURRENT R3 choose_move SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        r3_live_policy_engine.choose_move
    )
)

# ------------------------------------------------------------------------------------------
# 5. Determine whether a bridge is required
# ------------------------------------------------------------------------------------------

existing_agent_signature_7b = str(
    inspect.signature(
        PokemonBattleAgent.choose_move
    )
)

r3_engine_signature_7b = str(
    inspect.signature(
        r3_live_policy_engine.choose_move
    )
)

bridge_required_7b = (
    existing_agent_signature_7b
    !=
    r3_engine_signature_7b
)

print("\nADAPTER REQUIREMENT")
print("-" * 100)

print(
    "Existing agent signature :",
    existing_agent_signature_7b,
)

print(
    "R3 engine signature      :",
    r3_engine_signature_7b,
)

print(
    "Bridge required          :",
    bridge_required_7b,
)

print()
print("✅ SIMULATOR → AGENT HANDSHAKE EXTRACTED")
print("✅ EXISTING AGENT INTERFACE CONFIRMED")
print("✅ R3 INTERFACE COMPARED")
print("✅ SECTION 7B COMPLETE")


# In[87]:


# ==========================================================================================
# SECTION 7C — BUILD R3 LIVE BATTLE-AGENT BRIDGE
# PYTORCH-INDEPENDENT VERSION
# ==========================================================================================

print("=" * 100)
print("SECTION 7C — BUILD R3 LIVE BATTLE-AGENT BRIDGE")
print("=" * 100)

from __future__ import annotations

import inspect
from typing import Any

from src.agent_decision import AgentDecision

# IMPORTANT:
# Import the legal-move resolver DIRECTLY.
# Do NOT import src.competitive because that package loads PyTorch.
from src.legal_moves import (
    get_current_legal_moves,
)


# ==========================================================================================
# 1. Validate simulator output contract
# ==========================================================================================

print("\nAgentDecision")
print("-" * 100)

print(
    inspect.signature(
        AgentDecision
    )
)

assert AgentDecision is not None


# ==========================================================================================
# 2. Validate project legal-move resolver
# ==========================================================================================

assert callable(
    get_current_legal_moves
)

print("\nLEGAL MOVE RESOLVER")
print("-" * 100)

print(
    "Function:",
    get_current_legal_moves.__name__,
)

print(
    "Signature:",
    inspect.signature(
        get_current_legal_moves
    ),
)


# ==========================================================================================
# 3. Build simulator-compatible R3 live battle agent
# ==========================================================================================

class R3LiveBattleAgent:
    """
    Simulator bridge for the approved
    R3_STATE_CENTRIC_POLICY.

    Simulator expects:
        choose_move(state, depth=6) -> AgentDecision

    R3 engine expects:
        score_moves(battle_state, legal_moves) -> dict

    This bridge intentionally does NOT depend on PyTorch.
    """

    def __init__(
        self,
        policy_engine,
        name: str = "Team Jesus R3 Live Agent",
    ) -> None:

        self.policy_engine = policy_engine
        self.name = str(name)

        self.last_decision = None
        self.last_policy_result = None


    def choose_move(
        self,
        state: Any,
        depth: int = 6,
    ) -> AgentDecision:

        # ------------------------------------------------------------------
        # 1. Resolve actual legal moves from current simulator state
        # ------------------------------------------------------------------

        legal_moves = list(
            get_current_legal_moves(
                state
            )
        )

        if not legal_moves:

            raise RuntimeError(
                "R3LiveBattleAgent received a state "
                "with no legal moves."
            )

        # ------------------------------------------------------------------
        # 2. Score legal moves using approved R3 Random Forest engine
        # ------------------------------------------------------------------

        policy_result = (
            self.policy_engine.score_moves(
                battle_state=state,
                legal_moves=legal_moves,
            )
        )

        selected_index = int(
            policy_result[
                "selected_index"
            ]
        )

        assert (
            0
            <= selected_index
            < len(legal_moves)
        ), (
            "R3 returned an invalid legal-move index."
        )

        selected_move = legal_moves[
            selected_index
        ]

        # ------------------------------------------------------------------
        # 3. Preserve diagnostics
        # ------------------------------------------------------------------

        self.last_policy_result = (
            policy_result
        )

        # ------------------------------------------------------------------
        # 4. Convert R3 policy output to simulator AgentDecision
        # ------------------------------------------------------------------

        decision = AgentDecision(
            move=selected_move,

            score=float(
                policy_result[
                    "confidence"
                ]
            ),

            search_depth=int(
                depth
            ),

            nodes=0,

            principal_variation=[
                selected_move
            ],
        )

        self.last_decision = decision

        return decision


    def get_last_decision(
        self,
    ):

        return self.last_decision


    def get_last_policy_result(
        self,
    ):

        return self.last_policy_result


# ==========================================================================================
# 4. Instantiate live R3 agent
# ==========================================================================================

r3_live_battle_agent = (
    R3LiveBattleAgent(
        policy_engine=
            r3_live_policy_engine,

        name=
            "Team Jesus R3 Live Agent",
    )
)


# ==========================================================================================
# 5. Verify simulator-compatible signature
# ==========================================================================================

r3_agent_signature_7c = (
    inspect.signature(
        R3LiveBattleAgent.choose_move
    )
)

print("\nR3 LIVE BATTLE AGENT SIGNATURE")
print("-" * 100)

print(
    r3_agent_signature_7c
)


# ==========================================================================================
# 6. Structural validation
# ==========================================================================================

section7c_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "R3 battle agent instantiated",

            "passed":
                isinstance(
                    r3_live_battle_agent,
                    R3LiveBattleAgent,
                ),
        },

        {
            "validation":
                "choose_move callable",

            "passed":
                callable(
                    r3_live_battle_agent.choose_move
                ),
        },

        {
            "validation":
                "legal move resolver callable",

            "passed":
                callable(
                    get_current_legal_moves
                ),
        },

        {
            "validation":
                "R3 policy engine connected",

            "passed":
                (
                    r3_live_battle_agent.policy_engine
                    is
                    r3_live_policy_engine
                ),
        },

        {
            "validation":
                "AgentDecision available",

            "passed":
                AgentDecision is not None,
        },

        {
            "validation":
                "state argument supported",

            "passed":
                (
                    "state"
                    in
                    r3_agent_signature_7c.parameters
                ),
        },

        {
            "validation":
                "depth argument supported",

            "passed":
                (
                    "depth"
                    in
                    r3_agent_signature_7c.parameters
                ),
        },
    ]
)


print("\nR3 BATTLE-AGENT BRIDGE VALIDATION")
print("-" * 100)

display(
    section7c_validation_df
)

assert (
    section7c_validation_df[
        "passed"
    ].all()
)


# ==========================================================================================
# 7. Architecture summary
# ==========================================================================================

print("\nLIVE R3 BATTLE PATH")
print("-" * 100)

print(
    """
simulate_ai_battle()
        ↓
R3LiveBattleAgent.choose_move(state, depth)
        ↓
src.legal_moves.get_current_legal_moves(state)
        ↓
R3LivePolicyEngine.score_moves()
        ↓
41-column raw features
        ↓
46-column R3 state-centric matrix
        ↓
RandomForest.predict_proba()
        ↓
legal-action mask + renormalization
        ↓
AgentDecision
        ↓
apply_move()
"""
)


print("=" * 100)

print("✅ R3 LIVE BATTLE AGENT CREATED")
print("✅ DIRECT LEGAL-MOVE RESOLVER CONNECTED")
print("✅ PYTORCH COMPETITIVE PACKAGE BYPASSED")
print("✅ R3 RANDOM FOREST POLICY ENGINE CONNECTED")
print("✅ SIMULATOR choose_move(state, depth) CONTRACT PRESERVED")
print("✅ AgentDecision OUTPUT CONTRACT PRESERVED")
print("✅ NO SIMULATOR SOURCE MODIFICATION REQUIRED")
print("✅ SECTION 7C COMPLETE")

print("=" * 100)


# In[89]:


# ==========================================================================================
# 5. Inspect battle_state module public callables
# ==========================================================================================

import src.battle_state as battle_state_module_7d


battle_state_callables_7d = []

for name, obj in inspect.getmembers(
    battle_state_module_7d
):

    if (
        not name.startswith("_")
        and callable(obj)
    ):

        try:

            signature = str(
                inspect.signature(
                    obj
                )
            )

        except Exception:

            signature = (
                "<signature unavailable>"
            )

        battle_state_callables_7d.append(
            {
                "name": name,
                "signature": signature,
            }
        )


section7d_battle_state_callables_df = (
    pd.DataFrame(
        battle_state_callables_7d
    )
)


print("\nPUBLIC CALLABLES IN src.battle_state")
print("-" * 100)

display(
    section7d_battle_state_callables_df
)


# In[90]:


# ==========================================================================================
# SECTION 7D.1 — INSPECT EXACT BATTLE STATE DATA CONTRACT
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.1 — INSPECT EXACT BATTLE STATE DATA CONTRACT")
print("=" * 100)

import inspect

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

# ------------------------------------------------------------------------------------------
# Exact constructor signatures
# ------------------------------------------------------------------------------------------

print("\nPokemonState SIGNATURE")
print("-" * 100)
print(
    inspect.signature(
        PokemonState
    )
)

print("\nPlayerState SIGNATURE")
print("-" * 100)
print(
    inspect.signature(
        PlayerState
    )
)

print("\nBattleState SIGNATURE")
print("-" * 100)
print(
    inspect.signature(
        BattleState
    )
)

# ------------------------------------------------------------------------------------------
# Exact source definitions
# ------------------------------------------------------------------------------------------

print("\nPokemonState SOURCE")
print("-" * 100)
print(
    inspect.getsource(
        PokemonState
    )
)

print("\nPlayerState SOURCE")
print("-" * 100)
print(
    inspect.getsource(
        PlayerState
    )
)

print("\nBattleState SOURCE")
print("-" * 100)
print(
    inspect.getsource(
        BattleState
    )
)

# ------------------------------------------------------------------------------------------
# Dataclass fields
# ------------------------------------------------------------------------------------------

def show_dataclass_fields(cls):

    print(
        f"\n{cls.__name__} FIELDS"
    )
    print("-" * 100)

    rows = []

    for field_name, field_obj in (
        cls.__dataclass_fields__.items()
    ):

        rows.append(
            {
                "field":
                    field_name,

                "type":
                    str(
                        field_obj.type
                    ),

                "default":
                    (
                        field_obj.default
                        if str(
                            field_obj.default
                        )
                        !=
                        "<dataclasses._MISSING_TYPE object"
                        else None
                    ),
            }
        )

    display(
        pd.DataFrame(
            rows
        )
    )


show_dataclass_fields(
    PokemonState
)

show_dataclass_fields(
    PlayerState
)

show_dataclass_fields(
    BattleState
)

print("\n✅ PokemonState CONTRACT INSPECTED")
print("✅ PlayerState CONTRACT INSPECTED")
print("✅ BattleState CONTRACT INSPECTED")
print("✅ SECTION 7D.1 COMPLETE")


# In[91]:


# ==========================================================================================
# SECTION 7D.2 — RECOVER REAL CARD / BATTLESTATE EXAMPLES
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.2 — RECOVER REAL CARD / BATTLESTATE EXAMPLES")
print("=" * 100)

from pathlib import Path
import ast
import pandas as pd


# ==========================================================================================
# 1. Search locations
# ==========================================================================================

SEARCH_ROOTS_7D2 = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
]

SEARCH_TERMS_7D2 = [
    "PokemonState(",
    "PlayerState(",
    "BattleState(",
    "simulate_ai_battle(",
]


# ==========================================================================================
# 2. Search project source for real construction examples
# ==========================================================================================

example_records_7d2 = []

for search_root in SEARCH_ROOTS_7D2:

    if not search_root.exists():
        continue

    for py_file in search_root.rglob("*.py"):

        if "__pycache__" in py_file.parts:
            continue

        try:
            source_text = py_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        source_lines = source_text.splitlines()

        for line_index, line in enumerate(
            source_lines
        ):

            matched_terms = [
                term
                for term in SEARCH_TERMS_7D2
                if term in line
            ]

            if not matched_terms:
                continue

            start = max(
                0,
                line_index - 12,
            )

            end = min(
                len(source_lines),
                line_index + 22,
            )

            context = "\n".join(
                source_lines[
                    start:end
                ]
            )

            example_records_7d2.append(
                {
                    "file":
                        str(
                            py_file.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "line":
                        line_index + 1,

                    "matched_terms":
                        matched_terms,

                    "context":
                        context,
                }
            )


# ==========================================================================================
# 3. Summary
# ==========================================================================================

section7d2_example_summary_df = pd.DataFrame(
    [
        {
            "file":
                record["file"],

            "line":
                record["line"],

            "matched_terms":
                ", ".join(
                    record[
                        "matched_terms"
                    ]
                ),
        }
        for record in example_records_7d2
    ]
)


print("\nREAL BATTLE-STATE / CARD EXAMPLE LOCATIONS")
print("-" * 100)

if section7d2_example_summary_df.empty:

    print(
        "No explicit BattleState construction "
        "examples were found."
    )

else:

    display(
        section7d2_example_summary_df
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


# ==========================================================================================
# 4. Print useful contexts
# ==========================================================================================

print("\nDETAILED CONSTRUCTION EXAMPLES")
print("-" * 100)

for record_number, record in enumerate(
    example_records_7d2[:30],
    start=1,
):

    print("\n" + "=" * 100)

    print(
        f"EXAMPLE {record_number}"
    )

    print(
        "FILE:",
        record["file"],
    )

    print(
        "LINE:",
        record["line"],
    )

    print(
        "MATCH:",
        record["matched_terms"],
    )

    print("-" * 100)

    print(
        record["context"]
    )


# ==========================================================================================
# 5. Inspect move-generation expectations directly
# ==========================================================================================

from src.legal_moves import (
    get_current_legal_moves,
)

import inspect


print("\nget_current_legal_moves() SOURCE")
print("-" * 100)

print(
    inspect.getsource(
        get_current_legal_moves
    )
)


# ==========================================================================================
# 6. Inspect apply_move expectations
# ==========================================================================================

from src.simulator import (
    apply_move,
)


print("\napply_move() SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        apply_move
    )
)


print("\napply_move() SOURCE")
print("-" * 100)

print(
    inspect.getsource(
        apply_move
    )
)


# ==========================================================================================
# 7. Completion
# ==========================================================================================

print("\n" + "=" * 100)

print("✅ REAL BATTLE-STATE CONSTRUCTION EXAMPLES SEARCHED")
print("✅ CARD-DICTIONARY USAGE SEARCHED")
print("✅ LEGAL-MOVE REQUIREMENTS INSPECTED")
print("✅ apply_move() REQUIREMENTS INSPECTED")
print("✅ NO CARD STRUCTURE FABRICATED")
print("✅ SECTION 7D.2 COMPLETE")

print("=" * 100)


# In[96]:


# ==========================================================================================
# SECTION 7D.3A — RECOVER EXACT INTEGRATION CARDS FROM NOTEBOOK 11 SCRIPT
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.3A — RECOVER EXACT INTEGRATION CARDS FROM NOTEBOOK 11 SCRIPT")
print("=" * 100)

from pathlib import Path
import ast
import pandas as pd

SOURCE_FILE_7D3A = (
    PROJECT_ROOT
    / "scripts"
    / "11_advanced_search_engine.py"
)

assert SOURCE_FILE_7D3A.exists(), (
    f"Source file not found:\n{SOURCE_FILE_7D3A}"
)

# IMPORTANT:
# utf-8-sig removes a possible BOM (U+FEFF) at the beginning of the file.
source_text_7d3a = SOURCE_FILE_7D3A.read_text(
    encoding="utf-8-sig",
    errors="ignore",
)

tree_7d3a = ast.parse(
    source_text_7d3a
)

print("✅ Notebook 11 source parsed successfully")


# In[97]:


# ==========================================================================================
# SECTION 7D.3A — CONTINUE: RECOVER EXACT INTEGRATION CARDS
# ==========================================================================================

TARGET_CARD_NAMES_7D3A = {
    "integration_player_card",
    "integration_opponent_card",
}

recovered_card_assignments_7d3a = {}

for node in tree_7d3a.body:

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue

    assigned_names = []

    if isinstance(node, ast.Assign):

        for target in node.targets:

            if isinstance(
                target,
                ast.Name,
            ):
                assigned_names.append(
                    target.id
                )

    else:

        if isinstance(
            node.target,
            ast.Name,
        ):
            assigned_names.append(
                node.target.id
            )

    matching_names = [
        name
        for name in assigned_names
        if name in TARGET_CARD_NAMES_7D3A
    ]

    if not matching_names:
        continue

    value_node = getattr(
        node,
        "value",
        None,
    )

    if value_node is None:
        continue

    try:

        literal_value = ast.literal_eval(
            value_node
        )

    except Exception as exc:

        print(
            f"Could not literal-evaluate "
            f"{matching_names}: "
            f"{type(exc).__name__}: {exc}"
        )

        continue

    for name in matching_names:

        recovered_card_assignments_7d3a[
            name
        ] = literal_value


# ==========================================================================================
# VALIDATE RECOVERY
# ==========================================================================================

print("\nRECOVERED CARD VARIABLES")
print("-" * 100)

for name, card in (
    recovered_card_assignments_7d3a.items()
):

    print(
        name,
        "->",
        card.get(
            "name",
            "<no name>",
        ),
    )


assert (
    "integration_player_card"
    in recovered_card_assignments_7d3a
), (
    "integration_player_card was not recovered."
)

assert (
    "integration_opponent_card"
    in recovered_card_assignments_7d3a
), (
    "integration_opponent_card was not recovered."
)


integration_player_card_7d3a = (
    recovered_card_assignments_7d3a[
        "integration_player_card"
    ]
)

integration_opponent_card_7d3a = (
    recovered_card_assignments_7d3a[
        "integration_opponent_card"
    ]
)


# ==========================================================================================
# DISPLAY CARD CONTRACTS
# ==========================================================================================

print("\nPLAYER CARD")
print("-" * 100)

display(
    pd.DataFrame(
        [
            {
                "name":
                    integration_player_card_7d3a.get(
                        "name"
                    ),

                "hp":
                    integration_player_card_7d3a.get(
                        "hp"
                    ),

                "keys":
                    list(
                        integration_player_card_7d3a.keys()
                    ),
            }
        ]
    )
)


print("\nOPPONENT CARD")
print("-" * 100)

display(
    pd.DataFrame(
        [
            {
                "name":
                    integration_opponent_card_7d3a.get(
                        "name"
                    ),

                "hp":
                    integration_opponent_card_7d3a.get(
                        "hp"
                    ),

                "keys":
                    list(
                        integration_opponent_card_7d3a.keys()
                    ),
            }
        ]
    )
)


# ==========================================================================================
# HARD VALIDATION
# ==========================================================================================

assert isinstance(
    integration_player_card_7d3a,
    dict,
)

assert isinstance(
    integration_opponent_card_7d3a,
    dict,
)

assert (
    "name"
    in integration_player_card_7d3a
)

assert (
    "hp"
    in integration_player_card_7d3a
)

assert (
    "name"
    in integration_opponent_card_7d3a
)

assert (
    "hp"
    in integration_opponent_card_7d3a
)


print()
print("✅ EXACT NOTEBOOK 11 PLAYER CARD RECOVERED")
print("✅ EXACT NOTEBOOK 11 OPPONENT CARD RECOVERED")
print("✅ REAL CARD DICTIONARIES AVAILABLE")
print("✅ SECTION 7D.3A COMPLETE")


# In[98]:


# ==========================================================================================
# SECTION 7D.3B — BUILD AND VALIDATE ONE REAL R3 SIMULATOR TURN
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.3B — BUILD AND VALIDATE ONE REAL R3 SIMULATOR TURN")
print("=" * 100)

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)

from src.legal_moves import (
    get_current_legal_moves,
)

from src.simulator import (
    apply_move,
)


# ------------------------------------------------------------------------------------------
# Build real Pokémon states from recovered Notebook 11 cards
# ------------------------------------------------------------------------------------------

player_pokemon_7d3b = PokemonState(
    card=integration_player_card_7d3a,
    current_hp=float(
        integration_player_card_7d3a["hp"]
    ),
    attached_energy=2,
    status=None,
    damage=0.0,
    is_active=True,
)

opponent_pokemon_7d3b = PokemonState(
    card=integration_opponent_card_7d3a,
    current_hp=float(
        integration_opponent_card_7d3a["hp"]
    ),
    attached_energy=1,
    status=None,
    damage=0.0,
    is_active=True,
)


# ------------------------------------------------------------------------------------------
# Build player states
# ------------------------------------------------------------------------------------------

player_state_7d3b = PlayerState(
    active=player_pokemon_7d3b,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)

opponent_state_7d3b = PlayerState(
    active=opponent_pokemon_7d3b,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


# ------------------------------------------------------------------------------------------
# Build real BattleState
# ------------------------------------------------------------------------------------------

r3_live_initial_state = BattleState(
    player=player_state_7d3b,
    opponent=opponent_state_7d3b,
    turn_number=1,
    current_player="Player",
)


print("\nLIVE INITIAL STATE")
print("-" * 100)

print(
    "Player:",
    r3_live_initial_state.player.active.card["name"],
)

print(
    "Player HP:",
    r3_live_initial_state.player.active.current_hp,
)

print(
    "Player Energy:",
    r3_live_initial_state.player.active.attached_energy,
)

print()

print(
    "Opponent:",
    r3_live_initial_state.opponent.active.card["name"],
)

print(
    "Opponent HP:",
    r3_live_initial_state.opponent.active.current_hp,
)

print(
    "Opponent Energy:",
    r3_live_initial_state.opponent.active.attached_energy,
)

print()

print(
    "Turn:",
    r3_live_initial_state.turn_number,
)

print(
    "Current side:",
    r3_live_initial_state.current_player,
)


# ------------------------------------------------------------------------------------------
# Generate actual simulator legal moves
# ------------------------------------------------------------------------------------------

r3_live_initial_legal_moves = (
    get_current_legal_moves(
        r3_live_initial_state
    )
)

print("\nLEGAL MOVES")
print("-" * 100)

for index, move in enumerate(
    r3_live_initial_legal_moves,
    start=1,
):
    print(
        f"{index:02d}.",
        move,
    )

assert len(
    r3_live_initial_legal_moves
) >= 1


# ------------------------------------------------------------------------------------------
# R3 chooses move
# ------------------------------------------------------------------------------------------

r3_live_initial_decision = (
    r3_live_battle_agent.choose_move(
        state=r3_live_initial_state,
        depth=6,
    )
)

print("\nR3 DECISION")
print("-" * 100)

print(
    "Move:",
    r3_live_initial_decision.move,
)

print(
    "Score:",
    r3_live_initial_decision.score,
)

print(
    "Depth:",
    r3_live_initial_decision.search_depth,
)


assert (
    r3_live_initial_decision.move
    in r3_live_initial_legal_moves
)


# ------------------------------------------------------------------------------------------
# Apply one real simulator move
# ------------------------------------------------------------------------------------------

r3_live_next_state = apply_move(
    r3_live_initial_state,
    r3_live_initial_decision.move,
)


print("\nSTATE TRANSITION")
print("-" * 100)

print(
    "Turn:",
    r3_live_initial_state.turn_number,
    "→",
    r3_live_next_state.turn_number,
)

print(
    "Side:",
    r3_live_initial_state.current_player,
    "→",
    r3_live_next_state.current_player,
)

print(
    "Player HP:",
    r3_live_next_state.player.active.current_hp,
)

print(
    "Opponent HP:",
    r3_live_next_state.opponent.active.current_hp,
)

print(
    "Player prizes remaining:",
    r3_live_next_state.player.prize_cards_remaining,
)

print(
    "Opponent prizes remaining:",
    r3_live_next_state.opponent.prize_cards_remaining,
)


# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------

section7d3b_validation_df = pd.DataFrame(
    [
        {
            "validation": "BattleState created",
            "passed": isinstance(
                r3_live_initial_state,
                BattleState,
            ),
        },
        {
            "validation": "legal moves generated",
            "passed": len(
                r3_live_initial_legal_moves
            ) >= 1,
        },
        {
            "validation": "R3 decision returned",
            "passed": hasattr(
                r3_live_initial_decision,
                "move",
            ),
        },
        {
            "validation": "selected move legal",
            "passed": (
                r3_live_initial_decision.move
                in r3_live_initial_legal_moves
            ),
        },
        {
            "validation": "turn advanced",
            "passed": (
                r3_live_next_state.turn_number
                ==
                r3_live_initial_state.turn_number + 1
            ),
        },
        {
            "validation": "side switched",
            "passed": (
                r3_live_next_state.current_player
                !=
                r3_live_initial_state.current_player
            ),
        },
    ]
)

print("\nONE-TURN LIVE VALIDATION")
print("-" * 100)

display(
    section7d3b_validation_df
)

assert (
    section7d3b_validation_df[
        "passed"
    ].all()
)


print("\n" + "=" * 100)

print("✅ REAL NOTEBOOK 11 CARDS USED")
print("✅ GENUINE BattleState CREATED")
print("✅ LEGAL MOVES GENERATED")
print("✅ R3 AGENT CHOSE A LEGAL MOVE")
print("✅ apply_move() EXECUTED")
print("✅ STATE TRANSITION VERIFIED")
print("✅ SECTION 7D.3B COMPLETE")

print("=" * 100)


# In[99]:


# ==========================================================================================
# SECTION 7D.4 — VERIFY R3 DECISION WAS MODEL-DRIVEN
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.4 — VERIFY R3 DECISION WAS MODEL-DRIVEN")
print("=" * 100)

last_policy_result_7d4 = (
    r3_live_battle_agent
    .get_last_policy_result()
)

assert last_policy_result_7d4 is not None

print("\nLAST R3 POLICY RESULT")
print("-" * 100)

print(
    "Selected move:",
    last_policy_result_7d4[
        "selected_move_name"
    ],
)

print(
    "Fallback:",
    last_policy_result_7d4[
        "fallback"
    ],
)

print(
    "Legal probability mass:",
    last_policy_result_7d4[
        "legal_probability_mass"
    ],
)

print(
    "Legal moves:",
    last_policy_result_7d4[
        "move_names"
    ],
)

print(
    "R3 classes:",
    last_policy_result_7d4[
        "full_class_names"
    ],
)

print(
    "Full probabilities:",
    last_policy_result_7d4[
        "full_class_probabilities"
    ],
)

model_driven_7d4 = (
    not bool(
        last_policy_result_7d4[
            "fallback"
        ]
    )
    and
    float(
        last_policy_result_7d4[
            "legal_probability_mass"
        ]
    ) > 0.0
)

print("\nMODEL-DRIVEN DECISION")
print("-" * 100)

print(
    "Model driven:",
    model_driven_7d4
)

if model_driven_7d4:

    print(
        "\n✅ R3 DECISION WAS MODEL-DRIVEN"
    )

else:

    print(
        "\n⚠️ R3 DECISION USED LEGALITY FALLBACK"
    )

print("✅ SECTION 7D.4 COMPLETE")


# In[101]:


# ==========================================================================================
# SECTION 7D.5 — LOCATE R3-COMPATIBLE REAL PROJECT CARDS / ACTIONS
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.5 — LOCATE R3-COMPATIBLE REAL PROJECT CARDS / ACTIONS")
print("=" * 100)

from pathlib import Path
import pandas as pd


# ==========================================================================================
# 1. R3 trained action vocabulary
# ==========================================================================================

R3_ACTION_CLASSES_7D5 = [
    str(value)
    for value in r3_live_model.classes_
]

print("\nR3 ACTION VOCABULARY")
print("-" * 100)

for action_name in R3_ACTION_CLASSES_7D5:
    print(action_name)


# ==========================================================================================
# 2. Search project source for actual references
# ==========================================================================================

SEARCH_ROOTS_7D5 = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
]

match_records_7d5 = []


for search_root in SEARCH_ROOTS_7D5:

    if not search_root.exists():
        continue

    for py_file in search_root.rglob("*.py"):

        if "__pycache__" in py_file.parts:
            continue

        try:
            source_text = py_file.read_text(
                encoding="utf-8-sig",
                errors="ignore",
            )
        except Exception:
            continue

        source_lines = source_text.splitlines()

        for action_name in R3_ACTION_CLASSES_7D5:

            # Pass is simulator-generated and does not identify a card.
            if action_name.lower() == "pass":
                continue

            for line_index, line in enumerate(source_lines):

                if action_name.lower() not in line.lower():
                    continue

                start = max(
                    0,
                    line_index - 15,
                )

                end = min(
                    len(source_lines),
                    line_index + 18,
                )

                context = "\n".join(
                    source_lines[start:end]
                )

                match_records_7d5.append(
                    {
                        "action":
                            action_name,

                        "file":
                            str(
                                py_file.relative_to(
                                    PROJECT_ROOT
                                )
                            ),

                        "line":
                            line_index + 1,

                        "context":
                            context,
                    }
                )


# ==========================================================================================
# 3. Summary table
# ==========================================================================================

summary_rows_7d5 = [
    {
        "action":
            record["action"],

        "file":
            record["file"],

        "line":
            record["line"],
    }
    for record in match_records_7d5
]

section7d5_summary_df = pd.DataFrame(
    summary_rows_7d5
)

print("\nR3-COMPATIBLE ACTION REFERENCES")
print("-" * 100)

if section7d5_summary_df.empty:

    print("No R3 action references found.")

else:

    section7d5_summary_df = (
        section7d5_summary_df
        .drop_duplicates()
        .sort_values(
            [
                "action",
                "file",
                "line",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    display(
        section7d5_summary_df
    )


# ==========================================================================================
# 4. Detailed contexts
# ==========================================================================================

print("\nDETAILED R3 ACTION CONTEXT")
print("-" * 100)

for index, record in enumerate(
    match_records_7d5[:50],
    start=1,
):

    print("\n" + "=" * 100)
    print(f"MATCH {index}")
    print("Action:", record["action"])
    print("File:", record["file"])
    print("Line:", record["line"])
    print("-" * 100)
    print(record["context"])


# ==========================================================================================
# 5. Basic validation
# ==========================================================================================

assert len(
    match_records_7d5
) > 0, (
    "No project references to the R3 action vocabulary were found."
)

found_actions_7d5 = sorted(
    {
        record["action"]
        for record in match_records_7d5
    }
)

print("\nFOUND R3 ACTIONS")
print("-" * 100)

for action_name in found_actions_7d5:
    print(action_name)


print("\n" + "=" * 100)
print("✅ R3 ACTION VOCABULARY SEARCHED")
print("✅ REAL PROJECT ACTION REFERENCES LOCATED")
print("✅ R3-COMPATIBLE CARD CONTEXTS IDENTIFIED")
print("✅ SECTION 7D.5 COMPLETE")
print("=" * 100)


# In[102]:


# ==========================================================================================
# SECTION 7D.6 — PROVE A GENUINE MODEL-DRIVEN R3 LIVE DECISION
# ==========================================================================================

print("=" * 100)
print("SECTION 7D.6 — PROVE A GENUINE MODEL-DRIVEN R3 LIVE DECISION")
print("=" * 100)

import pandas as pd


# ==========================================================================================
# 1. Recover exact R3-compatible real cards
#
# These records come from the project's established integration/tournament card library.
# Eevee is especially useful because BOTH of its moves belong to the trained R3 classes:
#
#   Ascension
#   Quick Attack
#
# ==========================================================================================

section7d6_eevee_card = {
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
}


section7d6_bulbasaur_card = {
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
                "During your opponent's next turn, "
                "the Defending Pokémon can't retreat.",
        }
    ],
}


# ==========================================================================================
# 2. Build genuine simulator state
#
# Eevee acts with 3 Energy.
#
# Therefore BOTH:
#   Ascension    cost 1
#   Quick Attack cost 3
#
# are affordable and legal.
# ==========================================================================================

section7d6_player_pokemon = PokemonState(
    card=section7d6_eevee_card,
    current_hp=50.0,
    attached_energy=3,
    damage=0.0,
    is_active=True,
)

section7d6_opponent_pokemon = PokemonState(
    card=section7d6_bulbasaur_card,
    current_hp=80.0,
    attached_energy=1,
    damage=0.0,
    is_active=True,
)


section7d6_player_state = PlayerState(
    active=section7d6_player_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)

section7d6_opponent_state = PlayerState(
    active=section7d6_opponent_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


section7d6_battle_state = BattleState(
    player=section7d6_player_state,
    opponent=section7d6_opponent_state,
    turn_number=3,
    current_player="Player",
)


# ==========================================================================================
# 3. Ask the REAL legal-move resolver
# ==========================================================================================

section7d6_legal_moves = get_current_legal_moves(
    section7d6_battle_state
)

print("\nLIVE R3 TEST STATE")
print("-" * 100)

print(
    "Player card   :",
    section7d6_battle_state.player.active.card.get(
        "Card Name",
        section7d6_battle_state.player.active.card.get("name"),
    ),
)

print(
    "Player energy :",
    section7d6_battle_state.player.active.attached_energy,
)

print(
    "Opponent card :",
    section7d6_battle_state.opponent.active.card.get(
        "Card Name",
        section7d6_battle_state.opponent.active.card.get("name"),
    ),
)


print("\nLEGAL MOVES FROM PRODUCTION RESOLVER")
print("-" * 100)

for index, move in enumerate(
    section7d6_legal_moves,
    start=1,
):
    print(
        f"{index:02d}.",
        move,
    )


# ==========================================================================================
# 4. Require both learned Eevee actions
# ==========================================================================================

def section7d6_move_name(move):
    if isinstance(move, dict):
        return str(
            move.get(
                "name",
                move.get(
                    "Move Name",
                    "",
                ),
            )
        )

    return str(
        getattr(
            move,
            "name",
            move,
        )
    )


section7d6_legal_names = [
    section7d6_move_name(move)
    for move in section7d6_legal_moves
]


print("\nLEGAL MOVE NAMES")
print("-" * 100)

print(section7d6_legal_names)


assert "Ascension" in section7d6_legal_names, (
    "Ascension was not generated as a legal move."
)

assert "Quick Attack" in section7d6_legal_names, (
    "Quick Attack was not generated as a legal move."
)


# ==========================================================================================
# 5. Run R3 directly
# ==========================================================================================

section7d6_policy_result = (
    r3_live_policy_engine.score_moves(
        section7d6_battle_state,
        section7d6_legal_moves,
    )
)


print("\nR3 POLICY RESULT")
print("-" * 100)

for key, value in section7d6_policy_result.items():
    print(
        f"{key}: {value}"
    )


# ==========================================================================================
# 6. Validate genuine model-driven decision
# ==========================================================================================

section7d6_selected_move_name = (
    section7d6_policy_result.get(
        "selected_move_name"
    )
)

section7d6_fallback = bool(
    section7d6_policy_result.get(
        "fallback",
        section7d6_policy_result.get(
            "fallback_used",
            False,
        ),
    )
)


section7d6_probabilities = (
    section7d6_policy_result.get(
        "probabilities",
        []
    )
)


section7d6_probability_sum = float(
    sum(section7d6_probabilities)
)


section7d6_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "legal_moves_generated",

            "observed":
                len(section7d6_legal_moves),

            "expected":
                ">= 2",

            "passed":
                len(section7d6_legal_moves) >= 2,
        },

        {
            "validation":
                "Ascension_is_legal",

            "observed":
                "Ascension" in section7d6_legal_names,

            "expected":
                True,

            "passed":
                "Ascension" in section7d6_legal_names,
        },

        {
            "validation":
                "Quick_Attack_is_legal",

            "observed":
                "Quick Attack" in section7d6_legal_names,

            "expected":
                True,

            "passed":
                "Quick Attack" in section7d6_legal_names,
        },

        {
            "validation":
                "selected_move_is_legal",

            "observed":
                section7d6_selected_move_name,

            "expected":
                section7d6_legal_names,

            "passed":
                section7d6_selected_move_name
                in section7d6_legal_names,
        },

        {
            "validation":
                "fallback_not_used",

            "observed":
                section7d6_fallback,

            "expected":
                False,

            "passed":
                section7d6_fallback is False,
        },

        {
            "validation":
                "legal_probabilities_sum_to_one",

            "observed":
                round(
                    section7d6_probability_sum,
                    12,
                ),

            "expected":
                1.0,

            "passed":
                abs(
                    section7d6_probability_sum - 1.0
                ) < 1e-9,
        },
    ]
)


print("\nMODEL-DRIVEN R3 VALIDATION")
print("-" * 100)

display(
    section7d6_validation_df
)


# ==========================================================================================
# 7. Hard requirements
# ==========================================================================================

assert section7d6_validation_df[
    "passed"
].all(), (
    "At least one model-driven R3 validation failed."
)

assert section7d6_fallback is False, (
    "R3 still used fallback. "
    "Do not proceed to full battle simulation."
)

assert (
    section7d6_selected_move_name
    in {
        "Ascension",
        "Quick Attack",
    }
), (
    "R3 selected an unexpected action."
)


print("\n" + "=" * 100)
print("✅ REAL R3-COMPATIBLE CARD STATE CREATED")
print("✅ PRODUCTION LEGAL-MOVE RESOLVER USED")
print("✅ ASCENSION IS GENUINELY LEGAL")
print("✅ QUICK ATTACK IS GENUINELY LEGAL")
print("✅ R3 MODEL SCORED THE LIVE STATE")
print("✅ SELECTED MOVE CAME FROM R3 ACTION SPACE")
print("✅ NO LEGALITY FALLBACK USED")
print("✅ GENUINE MODEL-DRIVEN R3 DECISION CONFIRMED")
print("✅ SECTION 7D.6 COMPLETE")
print("=" * 100)


# In[103]:


# ==========================================================================================
# SECTION 7E.1 — RUN FIRST CONTROLLED MULTI-TURN R3 LIVE SIMULATOR BATTLE
# ==========================================================================================

print("=" * 100)
print("SECTION 7E.1 — RUN FIRST CONTROLLED MULTI-TURN R3 LIVE SIMULATOR BATTLE")
print("=" * 100)

from copy import deepcopy
import pandas as pd


# ==========================================================================================
# 1. Pre-flight checks
# ==========================================================================================

required_7e1_objects = {
    "simulate_ai_battle": simulate_ai_battle,
    "determine_battle_winner": determine_battle_winner,
    "create_battle_transcript": create_battle_transcript,
    "r3_live_battle_agent": r3_live_battle_agent,
    "section7d6_battle_state": section7d6_battle_state,
}

print("\nPRE-FLIGHT COMPONENT CHECK")
print("-" * 100)

for name, obj in required_7e1_objects.items():
    print(
        f"{name:<30}:",
        "LOADED" if obj is not None else "MISSING",
    )

assert all(
    obj is not None
    for obj in required_7e1_objects.values()
)


# ==========================================================================================
# 2. Create a clean battle copy
# ==========================================================================================

section7e1_initial_state = deepcopy(
    section7d6_battle_state
)

print("\nINITIAL BATTLE STATE")
print("-" * 100)

print(
    "Player card        :",
    section7e1_initial_state.player.active.card.get(
        "Card Name",
        section7e1_initial_state.player.active.card.get("name"),
    ),
)

print(
    "Player HP          :",
    section7e1_initial_state.player.active.current_hp,
)

print(
    "Player energy      :",
    section7e1_initial_state.player.active.attached_energy,
)

print(
    "Opponent card      :",
    section7e1_initial_state.opponent.active.card.get(
        "Card Name",
        section7e1_initial_state.opponent.active.card.get("name"),
    ),
)

print(
    "Opponent HP        :",
    section7e1_initial_state.opponent.active.current_hp,
)

print(
    "Opponent energy    :",
    section7e1_initial_state.opponent.active.attached_energy,
)

print(
    "Starting turn      :",
    section7e1_initial_state.turn_number,
)

print(
    "Starting side      :",
    section7e1_initial_state.current_player,
)


# ==========================================================================================
# 3. Reset bridge counters/history if supported
# ==========================================================================================

if hasattr(
    r3_live_battle_agent,
    "reset",
):
    r3_live_battle_agent.reset()


# ==========================================================================================
# 4. Run the real simulator
#
# Keep this deliberately short first.
# We are validating integration stability before running long battles.
# ==========================================================================================

print("\nRUNNING CONTROLLED LIVE BATTLE")
print("-" * 100)

section7e1_result = simulate_ai_battle(
    initial_state=section7e1_initial_state,
    agent=r3_live_battle_agent,
    search_depth=6,
    max_turns=6,
    verbose=False,
)


# ==========================================================================================
# 5. Inspect result contract
# ==========================================================================================

print("\nSIMULATION RESULT")
print("-" * 100)

print(
    "Result type       :",
    type(section7e1_result).__name__,
)

print(
    "Winner            :",
    section7e1_result.winner,
)

print(
    "Stop reason       :",
    section7e1_result.stop_reason,
)

print(
    "Turn records      :",
    len(section7e1_result.turns),
)

print(
    "Final turn number :",
    section7e1_result.final_state.turn_number,
)

print(
    "Final side        :",
    section7e1_result.final_state.current_player,
)


# ==========================================================================================
# 6. Inspect every turn
# ==========================================================================================

print("\nTURN-BY-TURN R3 DECISIONS")
print("-" * 100)

section7e1_turn_rows = []

for index, record in enumerate(
    section7e1_result.turns,
    start=1,
):

    move = getattr(
        record,
        "move",
        None,
    )

    if isinstance(move, dict):
        move_name = str(
            move.get(
                "name",
                move.get(
                    "Move Name",
                    move,
                ),
            )
        )
    else:
        move_name = str(
            getattr(
                move,
                "name",
                move,
            )
        )

    row = {
        "record":
            index,

        "turn_number":
            getattr(
                record,
                "turn_number",
                None,
            ),

        "acting_side":
            getattr(
                record,
                "acting_side",
                None,
            ),

        "move":
            move_name,

        "score":
            getattr(
                record,
                "score",
                None,
            ),

        "next_side":
            getattr(
                record,
                "next_side",
                None,
            ),
    }

    section7e1_turn_rows.append(
        row
    )


section7e1_turn_df = pd.DataFrame(
    section7e1_turn_rows
)

display(
    section7e1_turn_df
)


# ==========================================================================================
# 7. Create transcript using the project's own transcript function
# ==========================================================================================

print("\nBATTLE TRANSCRIPT")
print("-" * 100)

section7e1_transcript = (
    create_battle_transcript(
        section7e1_result
    )
)

print(
    section7e1_transcript
)


# ==========================================================================================
# 8. Validate the multi-turn simulator path
# ==========================================================================================

section7e1_final_state = (
    section7e1_result.final_state
)

section7e1_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "simulation_result_returned",

            "passed":
                section7e1_result is not None,
        },

        {
            "validation":
                "final_state_is_BattleState",

            "passed":
                isinstance(
                    section7e1_final_state,
                    BattleState,
                ),
        },

        {
            "validation":
                "at_least_one_turn_executed",

            "passed":
                len(
                    section7e1_result.turns
                ) >= 1,
        },

        {
            "validation":
                "turn_number_advanced",

            "passed":
                (
                    section7e1_final_state.turn_number
                    >
                    section7e1_initial_state.turn_number
                ),
        },

        {
            "validation":
                "simulator_returned_stop_reason",

            "passed":
                bool(
                    section7e1_result.stop_reason
                ),
        },

        {
            "validation":
                "transcript_created",

            "passed":
                isinstance(
                    section7e1_transcript,
                    str,
                )
                and len(
                    section7e1_transcript
                ) > 0,
        },
    ]
)


print("\nMULTI-TURN INTEGRATION VALIDATION")
print("-" * 100)

display(
    section7e1_validation_df
)


# ==========================================================================================
# 9. Hard assertions
# ==========================================================================================

assert section7e1_validation_df[
    "passed"
].all(), (
    "One or more multi-turn R3 simulator validations failed."
)


# ==========================================================================================
# 10. Completion
# ==========================================================================================

print("\n" + "=" * 100)

print("✅ REAL R3 BATTLE STATE USED")
print("✅ R3 LIVE BATTLE AGENT PASSED INTO simulate_ai_battle()")
print("✅ MULTI-TURN SIMULATION EXECUTED")
print("✅ REAL apply_move() PATH EXECUTED")
print("✅ TURN NUMBER ADVANCED")
print("✅ SIMULATOR RESULT CONTRACT PRESERVED")
print("✅ BATTLE TRANSCRIPT CREATED")
print("✅ SECTION 7E.1 COMPLETE")

print("=" * 100)


# In[104]:


# ==========================================================================================
# SECTION 7E.2 — VERIFY EVERY LIVE BATTLE TURN IS MODEL-DRIVEN
# ==========================================================================================

print("=" * 100)
print("SECTION 7E.2 — VERIFY EVERY LIVE BATTLE TURN IS MODEL-DRIVEN")
print("=" * 100)

from copy import deepcopy
import inspect
import pandas as pd


# ==========================================================================================
# 1. Inspect BattleTurnRecord contract
# ==========================================================================================

print("\nBattleTurnRecord SIGNATURE")
print("-" * 100)

print(
    inspect.signature(
        BattleTurnRecord
    )
)


print("\nBattleTurnRecord SOURCE")
print("-" * 100)

print(
    inspect.getsource(
        BattleTurnRecord
    )
)


# ==========================================================================================
# 2. Build a logging version of the already validated R3 battle agent
# ==========================================================================================

class R3LoggingBattleAgent(
    R3LiveBattleAgent
):
    """
    R3 live battle agent with per-turn policy diagnostics.
    """

    def __init__(
        self,
        policy_engine,
        name="R3 Logging Battle Agent",
    ):

        super().__init__(
            policy_engine=policy_engine,
            name=name,
        )

        self.policy_history = []


    def choose_move(
        self,
        state,
        depth=6,
    ):

        decision = super().choose_move(
            state=state,
            depth=depth,
        )

        result = dict(
            self.last_policy_result
        )

        result["turn_number"] = int(
            state.turn_number
        )

        result["acting_side"] = str(
            state.current_player
        )

        self.policy_history.append(
            result
        )

        return decision


    def reset(
        self,
    ):

        self.last_decision = None
        self.last_policy_result = None
        self.policy_history = []


# ==========================================================================================
# 3. Instantiate clean logging agent
# ==========================================================================================

section7e2_agent = (
    R3LoggingBattleAgent(
        policy_engine=
            r3_live_policy_engine,

        name=
            "Team Jesus R3 Logging Agent",
    )
)

section7e2_agent.reset()


# ==========================================================================================
# 4. Run another controlled six-turn battle
# ==========================================================================================

section7e2_initial_state = deepcopy(
    section7d6_battle_state
)


print("\nRUNNING LOGGED R3 LIVE BATTLE")
print("-" * 100)

section7e2_result = simulate_ai_battle(
    initial_state=
        section7e2_initial_state,

    agent=
        section7e2_agent,

    search_depth=6,

    max_turns=6,

    verbose=False,
)


# ==========================================================================================
# 5. Inspect raw BattleTurnRecord dictionaries
# ==========================================================================================

print("\nRAW TURN RECORDS")
print("-" * 100)

for index, record in enumerate(
    section7e2_result.turns,
    start=1,
):

    print(
        f"\nTURN RECORD {index}"
    )

    if hasattr(
        record,
        "as_dict",
    ):

        print(
            record.as_dict()
        )

    elif hasattr(
        record,
        "__dict__",
    ):

        print(
            record.__dict__
        )

    else:

        print(
            record
        )


# ==========================================================================================
# 6. Build policy-history table
# ==========================================================================================

section7e2_policy_rows = []


for index, policy_result in enumerate(
    section7e2_agent.policy_history,
    start=1,
):

    section7e2_policy_rows.append(
        {
            "record":
                index,

            "turn_number":
                policy_result.get(
                    "turn_number"
                ),

            "acting_side":
                policy_result.get(
                    "acting_side"
                ),

            "selected_move":
                policy_result.get(
                    "selected_move_name"
                ),

            "confidence":
                policy_result.get(
                    "confidence"
                ),

            "fallback":
                policy_result.get(
                    "fallback"
                ),

            "legal_probability_mass":
                policy_result.get(
                    "legal_probability_mass"
                ),

            "legal_moves":
                policy_result.get(
                    "move_names"
                ),
        }
    )


section7e2_policy_df = pd.DataFrame(
    section7e2_policy_rows
)


print("\nPER-TURN R3 POLICY HISTORY")
print("-" * 100)

display(
    section7e2_policy_df
)


# ==========================================================================================
# 7. Compute model-driven status per turn
# ==========================================================================================

section7e2_policy_df[
    "model_driven"
] = (
    (
        section7e2_policy_df[
            "fallback"
        ] == False
    )
    &
    (
        section7e2_policy_df[
            "legal_probability_mass"
        ].astype(float)
        > 0.0
    )
)


print("\nMODEL-DRIVEN TURN VALIDATION")
print("-" * 100)

display(
    section7e2_policy_df[
        [
            "record",
            "turn_number",
            "acting_side",
            "selected_move",
            "confidence",
            "fallback",
            "legal_probability_mass",
            "model_driven",
        ]
    ]
)


# ==========================================================================================
# 8. Summary statistics
# ==========================================================================================

section7e2_total_turns = len(
    section7e2_policy_df
)

section7e2_model_driven_turns = int(
    section7e2_policy_df[
        "model_driven"
    ].sum()
)

section7e2_fallback_turns = int(
    section7e2_policy_df[
        "fallback"
    ].sum()
)


print("\nR3 LIVE BATTLE DECISION SUMMARY")
print("-" * 100)

print(
    "Total decisions       :",
    section7e2_total_turns,
)

print(
    "Model-driven decisions:",
    section7e2_model_driven_turns,
)

print(
    "Fallback decisions    :",
    section7e2_fallback_turns,
)

print(
    "Model-driven rate     :",
    (
        section7e2_model_driven_turns
        /
        section7e2_total_turns
        if section7e2_total_turns
        else 0.0
    ),
)


# ==========================================================================================
# 9. Validate simulator and policy history alignment
# ==========================================================================================

section7e2_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "six_turns_recorded",

            "observed":
                len(
                    section7e2_result.turns
                ),

            "expected":
                6,

            "passed":
                len(
                    section7e2_result.turns
                ) == 6,
        },

        {
            "validation":
                "six_policy_decisions_logged",

            "observed":
                len(
                    section7e2_policy_df
                ),

            "expected":
                6,

            "passed":
                len(
                    section7e2_policy_df
                ) == 6,
        },

        {
            "validation":
                "turn_and_policy_counts_match",

            "observed":
                len(
                    section7e2_policy_df
                ),

            "expected":
                len(
                    section7e2_result.turns
                ),

            "passed":
                (
                    len(
                        section7e2_policy_df
                    )
                    ==
                    len(
                        section7e2_result.turns
                    )
                ),
        },

        {
            "validation":
                "all_selected_moves_present",

            "observed":
                section7e2_policy_df[
                    "selected_move"
                ].notna().all(),

            "expected":
                True,

            "passed":
                section7e2_policy_df[
                    "selected_move"
                ].notna().all(),
        },

        {
            "validation":
                "all_selected_moves_legal",

            "observed":
                all(
                    row["selected_move"]
                    in row["legal_moves"]

                    for _, row
                    in section7e2_policy_df.iterrows()
                ),

            "expected":
                True,

            "passed":
                all(
                    row["selected_move"]
                    in row["legal_moves"]

                    for _, row
                    in section7e2_policy_df.iterrows()
                ),
        },
    ]
)


print("\nSECTION 7E.2 VALIDATION")
print("-" * 100)

display(
    section7e2_validation_df
)


assert section7e2_validation_df[
    "passed"
].all()


# ==========================================================================================
# 10. Interpret model-driven coverage
# ==========================================================================================

print("\nMODEL-DRIVEN COVERAGE INTERPRETATION")
print("-" * 100)

if section7e2_fallback_turns == 0:

    print(
        "✅ All six simulator decisions were model-driven."
    )

elif section7e2_model_driven_turns > 0:

    print(
        "⚠️ Mixed model-driven and fallback decisions detected."
    )

else:

    print(
        "❌ All decisions used fallback."
    )


print("\n" + "=" * 100)

print("✅ PER-TURN R3 POLICY HISTORY CAPTURED")
print("✅ LIVE MOVE NAMES RECOVERED")
print("✅ LEGALITY VERIFIED FOR EVERY TURN")
print("✅ FALLBACK STATUS VERIFIED FOR EVERY TURN")
print("✅ MODEL-DRIVEN COVERAGE MEASURED")
print("✅ SECTION 7E.2 COMPLETE")

print("=" * 100)


# In[105]:


# ==========================================================================================
# SECTION 8A — CONTROLLED R3 COMPETITIVE EVALUATION
# ==========================================================================================

print("=" * 100)
print("SECTION 8A — CONTROLLED R3 COMPETITIVE EVALUATION")
print("=" * 100)

from copy import deepcopy
from collections import Counter

import pandas as pd
import numpy as np


# ==========================================================================================
# 1. Production-compatible R3 card library
# ==========================================================================================

R3_CARD_LIBRARY_8A = {
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
                    "During your opponent's next turn, "
                    "the Defending Pokémon can't retreat.",
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


# ==========================================================================================
# 2. Helper — construct one real BattleState
# ==========================================================================================

def build_r3_battle_state_8a(
    player_card_name,
    opponent_card_name,
    *,
    player_energy=3,
    opponent_energy=3,
    starting_side="Player",
    turn_number=1,
):

    player_card = deepcopy(
        R3_CARD_LIBRARY_8A[
            player_card_name
        ]
    )

    opponent_card = deepcopy(
        R3_CARD_LIBRARY_8A[
            opponent_card_name
        ]
    )

    player_pokemon = PokemonState(
        card=player_card,
        current_hp=float(
            player_card["hp"]
        ),
        attached_energy=int(
            player_energy
        ),
        status=None,
        damage=0.0,
        is_active=True,
    )

    opponent_pokemon = PokemonState(
        card=opponent_card,
        current_hp=float(
            opponent_card["hp"]
        ),
        attached_energy=int(
            opponent_energy
        ),
        status=None,
        damage=0.0,
        is_active=True,
    )

    return BattleState(
        player=PlayerState(
            active=player_pokemon,
            bench=[],
            prize_cards_remaining=6,
            hand_size=7,
        ),
        opponent=PlayerState(
            active=opponent_pokemon,
            bench=[],
            prize_cards_remaining=6,
            hand_size=7,
        ),
        turn_number=int(
            turn_number
        ),
        current_player=str(
            starting_side
        ),
    )


# ==========================================================================================
# 3. Helper — run one logged R3 battle
# ==========================================================================================

def run_r3_battle_8a(
    player_card,
    opponent_card,
    starting_side,
    *,
    max_turns=20,
):

    agent = R3LoggingBattleAgent(
        policy_engine=
            r3_live_policy_engine,

        name=
            "R3 Controlled Evaluation Agent",
    )

    agent.reset()

    initial_state = (
        build_r3_battle_state_8a(
            player_card,
            opponent_card,
            player_energy=3,
            opponent_energy=3,
            starting_side=
                starting_side,
            turn_number=1,
        )
    )

    result = simulate_ai_battle(
        initial_state=
            initial_state,

        agent=
            agent,

        search_depth=6,

        max_turns=
            max_turns,

        verbose=False,
    )

    history = list(
        agent.policy_history
    )

    total_decisions = len(
        history
    )

    fallback_count = sum(
        bool(
            row.get(
                "fallback",
                False,
            )
        )
        for row in history
    )

    model_driven_count = (
        total_decisions
        -
        fallback_count
    )

    move_counter = Counter(
        row.get(
            "selected_move_name"
        )
        for row in history
    )

    return {
        "player_card":
            player_card,

        "opponent_card":
            opponent_card,

        "starting_side":
            starting_side,

        "winner":
            result.winner,

        "stop_reason":
            result.stop_reason,

        "turns":
            len(
                result.turns
            ),

        "total_decisions":
            total_decisions,

        "model_driven_decisions":
            model_driven_count,

        "fallback_decisions":
            fallback_count,

        "model_driven_rate":
            (
                model_driven_count
                /
                total_decisions
                if total_decisions
                else 0.0
            ),

        "final_player_hp":
            float(
                result.final_state
                .player
                .active
                .current_hp
            ),

        "final_opponent_hp":
            float(
                result.final_state
                .opponent
                .active
                .current_hp
            ),

        "action_usage":
            dict(
                move_counter
            ),
    }


# ==========================================================================================
# 4. Controlled matchup matrix
#
# Each pairing is evaluated from BOTH starting sides.
# ==========================================================================================

matchups_8a = [
    ("Eevee", "Bulbasaur"),
    ("Eevee", "Charmander"),
    ("Eevee", "Meowth"),
    ("Bulbasaur", "Charmander"),
    ("Bulbasaur", "Meowth"),
    ("Charmander", "Meowth"),
]


evaluation_records_8a = []


print("\nRUNNING CONTROLLED MATCHUPS")
print("-" * 100)


for player_card, opponent_card in matchups_8a:

    for starting_side in [
        "Player",
        "Opponent",
    ]:

        print(
            f"{player_card:<12} vs "
            f"{opponent_card:<12} | "
            f"Start: {starting_side}"
        )

        evaluation_records_8a.append(
            run_r3_battle_8a(
                player_card=
                    player_card,

                opponent_card=
                    opponent_card,

                starting_side=
                    starting_side,

                max_turns=20,
            )
        )


# ==========================================================================================
# 5. Evaluation table
# ==========================================================================================

section8a_results_df = pd.DataFrame(
    evaluation_records_8a
)


print("\nCONTROLLED R3 COMPETITIVE RESULTS")
print("-" * 100)

display(
    section8a_results_df
)


# ==========================================================================================
# 6. Aggregate action usage
# ==========================================================================================

aggregate_action_usage_8a = Counter()

for usage in section8a_results_df[
    "action_usage"
]:

    aggregate_action_usage_8a.update(
        usage
    )


section8a_action_usage_df = (
    pd.DataFrame(
        [
            {
                "action":
                    action,

                "count":
                    count,
            }
            for action, count
            in aggregate_action_usage_8a.items()
        ]
    )
    .sort_values(
        "count",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print("\nR3 ACTION USAGE")
print("-" * 100)

display(
    section8a_action_usage_df
)


# ==========================================================================================
# 7. Summary metrics
# ==========================================================================================

section8a_total_battles = len(
    section8a_results_df
)

section8a_total_decisions = int(
    section8a_results_df[
        "total_decisions"
    ].sum()
)

section8a_total_fallbacks = int(
    section8a_results_df[
        "fallback_decisions"
    ].sum()
)

section8a_total_model_driven = int(
    section8a_results_df[
        "model_driven_decisions"
    ].sum()
)

section8a_model_driven_rate = (
    section8a_total_model_driven
    /
    section8a_total_decisions
    if section8a_total_decisions
    else 0.0
)

section8a_average_turns = float(
    section8a_results_df[
        "turns"
    ].mean()
)


print("\nCONTROLLED EVALUATION SUMMARY")
print("-" * 100)

print(
    "Battles evaluated       :",
    section8a_total_battles,
)

print(
    "Total R3 decisions      :",
    section8a_total_decisions,
)

print(
    "Model-driven decisions  :",
    section8a_total_model_driven,
)

print(
    "Fallback decisions      :",
    section8a_total_fallbacks,
)

print(
    "Model-driven rate       :",
    round(
        section8a_model_driven_rate,
        6,
    ),
)

print(
    "Average battle turns    :",
    round(
        section8a_average_turns,
        3,
    ),
)


# ==========================================================================================
# 8. Starting-side summary
# ==========================================================================================

section8a_side_summary_df = (
    section8a_results_df
    .groupby(
        "starting_side",
        as_index=False,
    )
    .agg(
        battles=(
            "player_card",
            "count",
        ),

        average_turns=(
            "turns",
            "mean",
        ),

        total_decisions=(
            "total_decisions",
            "sum",
        ),

        fallback_decisions=(
            "fallback_decisions",
            "sum",
        ),
    )
)


section8a_side_summary_df[
    "model_driven_rate"
] = (
    (
        section8a_side_summary_df[
            "total_decisions"
        ]
        -
        section8a_side_summary_df[
            "fallback_decisions"
        ]
    )
    /
    section8a_side_summary_df[
        "total_decisions"
    ]
)


print("\nSTARTING-SIDE SUMMARY")
print("-" * 100)

display(
    section8a_side_summary_df
)


# ==========================================================================================
# 9. Validation gates
# ==========================================================================================

section8a_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "twelve_battles_completed",

            "observed":
                section8a_total_battles,

            "expected":
                12,

            "passed":
                section8a_total_battles
                == 12,
        },

        {
            "validation":
                "all_battles_produced_decisions",

            "observed":
                bool(
                    (
                        section8a_results_df[
                            "total_decisions"
                        ]
                        > 0
                    ).all()
                ),

            "expected":
                True,

            "passed":
                bool(
                    (
                        section8a_results_df[
                            "total_decisions"
                        ]
                        > 0
                    ).all()
                ),
        },

        {
            "validation":
                "all_decisions_model_driven",

            "observed":
                section8a_total_fallbacks,

            "expected":
                0,

            "passed":
                section8a_total_fallbacks
                == 0,
        },

        {
            "validation":
                "model_driven_rate_100_percent",

            "observed":
                section8a_model_driven_rate,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    section8a_model_driven_rate,
                    1.0,
                ),
        },

        {
            "validation":
                "both_starting_sides_tested",

            "observed":
                sorted(
                    section8a_results_df[
                        "starting_side"
                    ].unique()
                ),

            "expected":
                [
                    "Opponent",
                    "Player",
                ],

            "passed":
                set(
                    section8a_results_df[
                        "starting_side"
                    ].unique()
                )
                ==
                {
                    "Player",
                    "Opponent",
                },
        },
    ]
)


print("\nSECTION 8A VALIDATION")
print("-" * 100)

display(
    section8a_validation_df
)


assert (
    section8a_validation_df[
        "passed"
    ].all()
)


# ==========================================================================================
# 10. Completion
# ==========================================================================================

print("\n" + "=" * 100)

print("✅ 12 CONTROLLED LIVE BATTLES COMPLETED")
print("✅ FOUR R3-COMPATIBLE POKÉMON EVALUATED")
print("✅ BOTH STARTING SIDES EVALUATED")
print("✅ MULTI-TURN LIVE POLICY DECISIONS RECORDED")
print("✅ ACTION USAGE RECORDED")
print("✅ ZERO FALLBACK REQUIRED")
print("✅ CONTROLLED COMPETITIVE EVALUATION PASSED")
print("✅ SECTION 8A COMPLETE")

print("=" * 100)


# In[106]:


# ==========================================================================================
# SECTION 8B — INTERPRET CONTROLLED COMPETITIVE PERFORMANCE
# ==========================================================================================

print("=" * 100)
print("SECTION 8B — INTERPRET CONTROLLED COMPETITIVE PERFORMANCE")
print("=" * 100)

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


# ==========================================================================================
# 1. Build winner-by-card summaries
# ==========================================================================================

card_names_8b = sorted(
    set(
        section8a_results_df["player_card"]
    )
    |
    set(
        section8a_results_df["opponent_card"]
    )
)

card_stats_8b = {
    card_name: {
        "battles": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    for card_name in card_names_8b
}


for _, row in section8a_results_df.iterrows():

    player_card = row["player_card"]
    opponent_card = row["opponent_card"]
    winner = row["winner"]

    card_stats_8b[player_card]["battles"] += 1
    card_stats_8b[opponent_card]["battles"] += 1

    if winner == "Player":

        card_stats_8b[player_card]["wins"] += 1
        card_stats_8b[opponent_card]["losses"] += 1

    elif winner == "Opponent":

        card_stats_8b[opponent_card]["wins"] += 1
        card_stats_8b[player_card]["losses"] += 1

    else:

        card_stats_8b[player_card]["draws"] += 1
        card_stats_8b[opponent_card]["draws"] += 1


card_summary_rows_8b = []

for card_name, stats in card_stats_8b.items():

    battles = stats["battles"]
    wins = stats["wins"]

    card_summary_rows_8b.append(
        {
            "card":
                card_name,

            "battles":
                battles,

            "wins":
                wins,

            "losses":
                stats["losses"],

            "draws":
                stats["draws"],

            "win_rate":
                (
                    wins / battles
                    if battles
                    else 0.0
                ),
        }
    )


section8b_card_summary_df = pd.DataFrame(
    card_summary_rows_8b
).sort_values(
    [
        "win_rate",
        "wins",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


print("\nCARD-LEVEL CONTROLLED RESULTS")
print("-" * 100)

display(
    section8b_card_summary_df
)


# ==========================================================================================
# 2. Pairwise matchup summary
# ==========================================================================================

pairwise_rows_8b = []


for (
    player_card,
    opponent_card
), group in section8a_results_df.groupby(
    [
        "player_card",
        "opponent_card",
    ]
):

    player_wins = int(
        (
            group["winner"]
            == "Player"
        ).sum()
    )

    opponent_wins = int(
        (
            group["winner"]
            == "Opponent"
        ).sum()
    )

    draws = int(
        group["winner"].isna().sum()
    )

    side_flip_consistent = (
        len(
            group["winner"].dropna().unique()
        )
        == 1
    )

    pairwise_rows_8b.append(
        {
            "player_card":
                player_card,

            "opponent_card":
                opponent_card,

            "battles":
                len(group),

            "player_wins":
                player_wins,

            "opponent_wins":
                opponent_wins,

            "draws":
                draws,

            "winner_consistent_across_start_side":
                side_flip_consistent,

            "average_turns":
                float(
                    group["turns"].mean()
                ),
        }
    )


section8b_pairwise_df = pd.DataFrame(
    pairwise_rows_8b
)


print("\nPAIRWISE MATCHUP SUMMARY")
print("-" * 100)

display(
    section8b_pairwise_df
)


# ==========================================================================================
# 3. Starting-side effect
# ==========================================================================================

player_start_df_8b = (
    section8a_results_df[
        section8a_results_df[
            "starting_side"
        ] == "Player"
    ]
)

opponent_start_df_8b = (
    section8a_results_df[
        section8a_results_df[
            "starting_side"
        ] == "Opponent"
    ]
)


player_start_wins_8b = int(
    (
        player_start_df_8b[
            "winner"
        ]
        ==
        "Player"
    ).sum()
)

opponent_start_wins_8b = int(
    (
        opponent_start_df_8b[
            "winner"
        ]
        ==
        "Opponent"
    ).sum()
)


player_start_rate_8b = (
    player_start_wins_8b
    /
    len(player_start_df_8b)
)

opponent_start_rate_8b = (
    opponent_start_wins_8b
    /
    len(opponent_start_df_8b)
)


section8b_side_effect_df = pd.DataFrame(
    [
        {
            "starting_side":
                "Player",

            "battles":
                len(
                    player_start_df_8b
                ),

            "starter_wins":
                player_start_wins_8b,

            "starter_win_rate":
                player_start_rate_8b,

            "average_turns":
                float(
                    player_start_df_8b[
                        "turns"
                    ].mean()
                ),
        },

        {
            "starting_side":
                "Opponent",

            "battles":
                len(
                    opponent_start_df_8b
                ),

            "starter_wins":
                opponent_start_wins_8b,

            "starter_win_rate":
                opponent_start_rate_8b,

            "average_turns":
                float(
                    opponent_start_df_8b[
                        "turns"
                    ].mean()
                ),
        },
    ]
)


print("\nSTARTING-SIDE EFFECT")
print("-" * 100)

display(
    section8b_side_effect_df
)


# ==========================================================================================
# 4. Action usage share
# ==========================================================================================

section8b_action_usage_df = (
    section8a_action_usage_df.copy()
)

section8b_total_actions = int(
    section8b_action_usage_df[
        "count"
    ].sum()
)

section8b_action_usage_df[
    "usage_rate"
] = (
    section8b_action_usage_df[
        "count"
    ]
    /
    section8b_total_actions
)


print("\nACTION USAGE DISTRIBUTION")
print("-" * 100)

display(
    section8b_action_usage_df
)


# ==========================================================================================
# 5. Controlled ranking
# ==========================================================================================

section8b_ranking_df = (
    section8b_card_summary_df[
        [
            "card",
            "battles",
            "wins",
            "losses",
            "draws",
            "win_rate",
        ]
    ]
    .copy()
)

section8b_ranking_df.insert(
    0,
    "rank",
    range(
        1,
        len(
            section8b_ranking_df
        ) + 1,
    ),
)


print("\nCONTROLLED MATCHUP RANKING")
print("-" * 100)

display(
    section8b_ranking_df
)


# ==========================================================================================
# 6. Core metrics
# ==========================================================================================

side_consistency_rate_8b = float(
    section8b_pairwise_df[
        "winner_consistent_across_start_side"
    ].mean()
)

terminal_rate_8b = float(
    (
        section8a_results_df[
            "stop_reason"
        ]
        ==
        "Battle reached a terminal state."
    ).mean()
)

fallback_rate_8b = (
    section8a_total_fallbacks
    /
    section8a_total_decisions
)

print("\nCONTROLLED EVALUATION DIAGNOSTICS")
print("-" * 100)

print(
    "Terminal battle rate             :",
    round(
        terminal_rate_8b,
        6,
    ),
)

print(
    "Model-driven decision rate       :",
    round(
        section8a_model_driven_rate,
        6,
    ),
)

print(
    "Fallback rate                    :",
    round(
        fallback_rate_8b,
        6,
    ),
)

print(
    "Winner consistency across sides  :",
    round(
        side_consistency_rate_8b,
        6,
    ),
)

print(
    "Average battle length            :",
    round(
        section8a_average_turns,
        3,
    ),
)


# ==========================================================================================
# 7. Interpretation
# ==========================================================================================

print("\nINTERPRETATION")
print("-" * 100)

best_card_8b = (
    section8b_ranking_df.iloc[0][
        "card"
    ]
)

worst_card_8b = (
    section8b_ranking_df.iloc[-1][
        "card"
    ]
)

print(
    f"Highest controlled win rate : "
    f"{best_card_8b}"
)

print(
    f"Lowest controlled win rate  : "
    f"{worst_card_8b}"
)

print(
    "All controlled battles reached a terminal state."
)

print(
    "All R3 decisions were model-driven; "
    "no legality fallback was required."
)

print(
    "Pairwise winners remained stable when "
    "the starting side was reversed."
)

print(
    "This indicates that, in this small controlled card set, "
    "matchup/card strength dominates starting-side advantage."
)

print(
    "These results validate the live evaluation harness, "
    "but they are not yet a Kaggle/general win-rate estimate."
)


# ==========================================================================================
# 8. Validation gates
# ==========================================================================================

section8b_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "all_battles_terminal",

            "observed":
                terminal_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    terminal_rate_8b,
                    1.0,
                ),
        },

        {
            "validation":
                "zero_fallback_rate",

            "observed":
                fallback_rate_8b,

            "expected":
                0.0,

            "passed":
                np.isclose(
                    fallback_rate_8b,
                    0.0,
                ),
        },

        {
            "validation":
                "model_driven_rate_100_percent",

            "observed":
                section8a_model_driven_rate,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    section8a_model_driven_rate,
                    1.0,
                ),
        },

        {
            "validation":
                "all_pairwise_winners_side_stable",

            "observed":
                side_consistency_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    side_consistency_rate_8b,
                    1.0,
                ),
        },

        {
            "validation":
                "four_cards_ranked",

            "observed":
                len(
                    section8b_ranking_df
                ),

            "expected":
                4,

            "passed":
                len(
                    section8b_ranking_df
                ) == 4,
        },
    ]
)


print("\nSECTION 8B VALIDATION")
print("-" * 100)

display(
    section8b_validation_df
)

assert (
    section8b_validation_df[
        "passed"
    ].all()
)


print("\n" + "=" * 100)

print("✅ CARD-LEVEL WIN RATES COMPUTED")
print("✅ PAIRWISE MATCHUPS SUMMARIZED")
print("✅ STARTING-SIDE EFFECT MEASURED")
print("✅ ACTION USAGE DISTRIBUTION COMPUTED")
print("✅ CONTROLLED CARD RANKING CREATED")
print("✅ ZERO-FALLBACK LIVE EVALUATION CONFIRMED")
print("✅ NOTEBOOK 60 BENCHMARK HANDOFF READY")
print("✅ SECTION 8B COMPLETE")

print("=" * 100)


# In[107]:


# ==========================================================================================
# SECTION 8B — INTERPRET CONTROLLED COMPETITIVE PERFORMANCE
# ==========================================================================================

print("=" * 100)
print("SECTION 8B — INTERPRET CONTROLLED COMPETITIVE PERFORMANCE")
print("=" * 100)

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


# ==========================================================================================
# 1. Build winner-by-card summaries
# ==========================================================================================

card_names_8b = sorted(
    set(
        section8a_results_df["player_card"]
    )
    |
    set(
        section8a_results_df["opponent_card"]
    )
)

card_stats_8b = {
    card_name: {
        "battles": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    for card_name in card_names_8b
}


for _, row in section8a_results_df.iterrows():

    player_card = row["player_card"]
    opponent_card = row["opponent_card"]
    winner = row["winner"]

    card_stats_8b[player_card]["battles"] += 1
    card_stats_8b[opponent_card]["battles"] += 1

    if winner == "Player":

        card_stats_8b[player_card]["wins"] += 1
        card_stats_8b[opponent_card]["losses"] += 1

    elif winner == "Opponent":

        card_stats_8b[opponent_card]["wins"] += 1
        card_stats_8b[player_card]["losses"] += 1

    else:

        card_stats_8b[player_card]["draws"] += 1
        card_stats_8b[opponent_card]["draws"] += 1


card_summary_rows_8b = []

for card_name, stats in card_stats_8b.items():

    battles = stats["battles"]
    wins = stats["wins"]

    card_summary_rows_8b.append(
        {
            "card":
                card_name,

            "battles":
                battles,

            "wins":
                wins,

            "losses":
                stats["losses"],

            "draws":
                stats["draws"],

            "win_rate":
                (
                    wins / battles
                    if battles
                    else 0.0
                ),
        }
    )


section8b_card_summary_df = pd.DataFrame(
    card_summary_rows_8b
).sort_values(
    [
        "win_rate",
        "wins",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(
    drop=True
)


print("\nCARD-LEVEL CONTROLLED RESULTS")
print("-" * 100)

display(
    section8b_card_summary_df
)


# ==========================================================================================
# 2. Pairwise matchup summary
# ==========================================================================================

pairwise_rows_8b = []


for (
    player_card,
    opponent_card
), group in section8a_results_df.groupby(
    [
        "player_card",
        "opponent_card",
    ]
):

    player_wins = int(
        (
            group["winner"]
            == "Player"
        ).sum()
    )

    opponent_wins = int(
        (
            group["winner"]
            == "Opponent"
        ).sum()
    )

    draws = int(
        group["winner"].isna().sum()
    )

    side_flip_consistent = (
        len(
            group["winner"].dropna().unique()
        )
        == 1
    )

    pairwise_rows_8b.append(
        {
            "player_card":
                player_card,

            "opponent_card":
                opponent_card,

            "battles":
                len(group),

            "player_wins":
                player_wins,

            "opponent_wins":
                opponent_wins,

            "draws":
                draws,

            "winner_consistent_across_start_side":
                side_flip_consistent,

            "average_turns":
                float(
                    group["turns"].mean()
                ),
        }
    )


section8b_pairwise_df = pd.DataFrame(
    pairwise_rows_8b
)


print("\nPAIRWISE MATCHUP SUMMARY")
print("-" * 100)

display(
    section8b_pairwise_df
)


# ==========================================================================================
# 3. Starting-side effect
# ==========================================================================================

player_start_df_8b = (
    section8a_results_df[
        section8a_results_df[
            "starting_side"
        ] == "Player"
    ]
)

opponent_start_df_8b = (
    section8a_results_df[
        section8a_results_df[
            "starting_side"
        ] == "Opponent"
    ]
)


player_start_wins_8b = int(
    (
        player_start_df_8b[
            "winner"
        ]
        ==
        "Player"
    ).sum()
)

opponent_start_wins_8b = int(
    (
        opponent_start_df_8b[
            "winner"
        ]
        ==
        "Opponent"
    ).sum()
)


player_start_rate_8b = (
    player_start_wins_8b
    /
    len(player_start_df_8b)
)

opponent_start_rate_8b = (
    opponent_start_wins_8b
    /
    len(opponent_start_df_8b)
)


section8b_side_effect_df = pd.DataFrame(
    [
        {
            "starting_side":
                "Player",

            "battles":
                len(
                    player_start_df_8b
                ),

            "starter_wins":
                player_start_wins_8b,

            "starter_win_rate":
                player_start_rate_8b,

            "average_turns":
                float(
                    player_start_df_8b[
                        "turns"
                    ].mean()
                ),
        },

        {
            "starting_side":
                "Opponent",

            "battles":
                len(
                    opponent_start_df_8b
                ),

            "starter_wins":
                opponent_start_wins_8b,

            "starter_win_rate":
                opponent_start_rate_8b,

            "average_turns":
                float(
                    opponent_start_df_8b[
                        "turns"
                    ].mean()
                ),
        },
    ]
)


print("\nSTARTING-SIDE EFFECT")
print("-" * 100)

display(
    section8b_side_effect_df
)


# ==========================================================================================
# 4. Action usage share
# ==========================================================================================

section8b_action_usage_df = (
    section8a_action_usage_df.copy()
)

section8b_total_actions = int(
    section8b_action_usage_df[
        "count"
    ].sum()
)

section8b_action_usage_df[
    "usage_rate"
] = (
    section8b_action_usage_df[
        "count"
    ]
    /
    section8b_total_actions
)


print("\nACTION USAGE DISTRIBUTION")
print("-" * 100)

display(
    section8b_action_usage_df
)


# ==========================================================================================
# 5. Controlled ranking
# ==========================================================================================

section8b_ranking_df = (
    section8b_card_summary_df[
        [
            "card",
            "battles",
            "wins",
            "losses",
            "draws",
            "win_rate",
        ]
    ]
    .copy()
)

section8b_ranking_df.insert(
    0,
    "rank",
    range(
        1,
        len(
            section8b_ranking_df
        ) + 1,
    ),
)


print("\nCONTROLLED MATCHUP RANKING")
print("-" * 100)

display(
    section8b_ranking_df
)


# ==========================================================================================
# 6. Core metrics
# ==========================================================================================

side_consistency_rate_8b = float(
    section8b_pairwise_df[
        "winner_consistent_across_start_side"
    ].mean()
)

terminal_rate_8b = float(
    (
        section8a_results_df[
            "stop_reason"
        ]
        ==
        "Battle reached a terminal state."
    ).mean()
)

fallback_rate_8b = (
    section8a_total_fallbacks
    /
    section8a_total_decisions
)

print("\nCONTROLLED EVALUATION DIAGNOSTICS")
print("-" * 100)

print(
    "Terminal battle rate             :",
    round(
        terminal_rate_8b,
        6,
    ),
)

print(
    "Model-driven decision rate       :",
    round(
        section8a_model_driven_rate,
        6,
    ),
)

print(
    "Fallback rate                    :",
    round(
        fallback_rate_8b,
        6,
    ),
)

print(
    "Winner consistency across sides  :",
    round(
        side_consistency_rate_8b,
        6,
    ),
)

print(
    "Average battle length            :",
    round(
        section8a_average_turns,
        3,
    ),
)


# ==========================================================================================
# 7. Interpretation
# ==========================================================================================

print("\nINTERPRETATION")
print("-" * 100)

best_card_8b = (
    section8b_ranking_df.iloc[0][
        "card"
    ]
)

worst_card_8b = (
    section8b_ranking_df.iloc[-1][
        "card"
    ]
)

print(
    f"Highest controlled win rate : "
    f"{best_card_8b}"
)

print(
    f"Lowest controlled win rate  : "
    f"{worst_card_8b}"
)

print(
    "All controlled battles reached a terminal state."
)

print(
    "All R3 decisions were model-driven; "
    "no legality fallback was required."
)

print(
    "Pairwise winners remained stable when "
    "the starting side was reversed."
)

print(
    "This indicates that, in this small controlled card set, "
    "matchup/card strength dominates starting-side advantage."
)

print(
    "These results validate the live evaluation harness, "
    "but they are not yet a Kaggle/general win-rate estimate."
)


# ==========================================================================================
# 8. Validation gates
# ==========================================================================================

section8b_validation_df = pd.DataFrame(
    [
        {
            "validation":
                "all_battles_terminal",

            "observed":
                terminal_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    terminal_rate_8b,
                    1.0,
                ),
        },

        {
            "validation":
                "zero_fallback_rate",

            "observed":
                fallback_rate_8b,

            "expected":
                0.0,

            "passed":
                np.isclose(
                    fallback_rate_8b,
                    0.0,
                ),
        },

        {
            "validation":
                "model_driven_rate_100_percent",

            "observed":
                section8a_model_driven_rate,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    section8a_model_driven_rate,
                    1.0,
                ),
        },

        {
            "validation":
                "all_pairwise_winners_side_stable",

            "observed":
                side_consistency_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    side_consistency_rate_8b,
                    1.0,
                ),
        },

        {
            "validation":
                "four_cards_ranked",

            "observed":
                len(
                    section8b_ranking_df
                ),

            "expected":
                4,

            "passed":
                len(
                    section8b_ranking_df
                ) == 4,
        },
    ]
)


print("\nSECTION 8B VALIDATION")
print("-" * 100)

display(
    section8b_validation_df
)

assert (
    section8b_validation_df[
        "passed"
    ].all()
)


print("\n" + "=" * 100)

print("✅ CARD-LEVEL WIN RATES COMPUTED")
print("✅ PAIRWISE MATCHUPS SUMMARIZED")
print("✅ STARTING-SIDE EFFECT MEASURED")
print("✅ ACTION USAGE DISTRIBUTION COMPUTED")
print("✅ CONTROLLED CARD RANKING CREATED")
print("✅ ZERO-FALLBACK LIVE EVALUATION CONFIRMED")
print("✅ NOTEBOOK 60 BENCHMARK HANDOFF READY")
print("✅ SECTION 8B COMPLETE")

print("=" * 100)


# In[108]:


# ==========================================================================================
# SECTION 9A — NOTEBOOK 59 FINAL VALIDATION AND HANDOFF EXPORT
# ==========================================================================================

print("=" * 100)
print("SECTION 9A — NOTEBOOK 59 FINAL VALIDATION AND HANDOFF EXPORT")
print("=" * 100)

from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


# ==========================================================================================
# 1. Create Notebook 59 output directory
# ==========================================================================================

NOTEBOOK59_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "notebook59"
)

NOTEBOOK59_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nOUTPUT DIRECTORY")
print("-" * 100)
print(NOTEBOOK59_OUTPUT_DIR)


# ==========================================================================================
# 2. Define output paths
# ==========================================================================================

RESULTS_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "r3_controlled_battle_results.csv"
)

CARD_RANKING_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "r3_controlled_card_ranking.csv"
)

PAIRWISE_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "r3_pairwise_matchup_summary.csv"
)

SIDE_EFFECT_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "r3_starting_side_summary.csv"
)

ACTION_USAGE_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "r3_action_usage.csv"
)

SUMMARY_PATH_59 = (
    NOTEBOOK59_OUTPUT_DIR
    / "notebook59_r3_live_integration_summary.json"
)


# ==========================================================================================
# 3. Export evaluation tables
# ==========================================================================================

section8a_results_df.to_csv(
    RESULTS_PATH_59,
    index=False,
)

section8b_ranking_df.to_csv(
    CARD_RANKING_PATH_59,
    index=False,
)

section8b_pairwise_df.to_csv(
    PAIRWISE_PATH_59,
    index=False,
)

section8b_side_effect_df.to_csv(
    SIDE_EFFECT_PATH_59,
    index=False,
)

section8b_action_usage_df.to_csv(
    ACTION_USAGE_PATH_59,
    index=False,
)


# ==========================================================================================
# 4. Final Notebook 59 metrics
# ==========================================================================================

final_metrics_59 = {
    "notebook":
        59,

    "purpose":
        "R3 live simulator integration and controlled competitive validation",

    "production_strategy":
        "R3_STATE_CENTRIC_POLICY",

    "production_model_type":
        type(
            r3_live_model
        ).__name__,

    "retained_feature_count":
        int(
            len(
                r3_live_feature_names
            )
        ),

    "model_class_count":
        int(
            len(
                r3_live_model.classes_
            )
        ),

    "model_classes":
        [
            str(value)
            for value
            in r3_live_model.classes_
        ],

    "controlled_battles":
        int(
            section8a_total_battles
        ),

    "total_live_decisions":
        int(
            section8a_total_decisions
        ),

    "model_driven_decisions":
        int(
            section8a_total_model_driven
        ),

    "fallback_decisions":
        int(
            section8a_total_fallbacks
        ),

    "model_driven_rate":
        float(
            section8a_model_driven_rate
        ),

    "fallback_rate":
        float(
            fallback_rate_8b
        ),

    "terminal_battle_rate":
        float(
            terminal_rate_8b
        ),

    "average_battle_turns":
        float(
            section8a_average_turns
        ),

    "pairwise_winner_side_consistency":
        float(
            side_consistency_rate_8b
        ),

    "controlled_ranking":
        (
            section8b_ranking_df[
                [
                    "rank",
                    "card",
                    "wins",
                    "losses",
                    "win_rate",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),

    "integration_status":
        "PASSED",

    "benchmark_status":
        "READY_FOR_NOTEBOOK_60",

    "created_at":
        datetime.now().isoformat(
            timespec="seconds"
        ),
}


with open(
    SUMMARY_PATH_59,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_metrics_59,
        file,
        indent=2,
    )


# ==========================================================================================
# 5. Final validation matrix
# ==========================================================================================

final_validation_59 = pd.DataFrame(
    [
        {
            "requirement":
                "R3 production model loaded",

            "observed":
                type(
                    r3_live_model
                ).__name__,

            "expected":
                "RandomForestClassifier",

            "passed":
                type(
                    r3_live_model
                ).__name__
                ==
                "RandomForestClassifier",
        },

        {
            "requirement":
                "46-feature live contract",

            "observed":
                len(
                    r3_live_feature_names
                ),

            "expected":
                46,

            "passed":
                len(
                    r3_live_feature_names
                )
                == 46,
        },

        {
            "requirement":
                "Controlled battles completed",

            "observed":
                section8a_total_battles,

            "expected":
                12,

            "passed":
                section8a_total_battles
                == 12,
        },

        {
            "requirement":
                "All battles terminal",

            "observed":
                terminal_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    terminal_rate_8b,
                    1.0,
                ),
        },

        {
            "requirement":
                "All decisions model-driven",

            "observed":
                section8a_model_driven_rate,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    section8a_model_driven_rate,
                    1.0,
                ),
        },

        {
            "requirement":
                "Zero fallback decisions",

            "observed":
                section8a_total_fallbacks,

            "expected":
                0,

            "passed":
                section8a_total_fallbacks
                == 0,
        },

        {
            "requirement":
                "Pairwise side consistency",

            "observed":
                side_consistency_rate_8b,

            "expected":
                1.0,

            "passed":
                np.isclose(
                    side_consistency_rate_8b,
                    1.0,
                ),
        },
    ]
)


print("\nNOTEBOOK 59 FINAL VALIDATION")
print("-" * 100)

display(
    final_validation_59
)


assert (
    final_validation_59[
        "passed"
    ].all()
), (
    "Notebook 59 final validation failed."
)


# ==========================================================================================
# 6. Verify exported files
# ==========================================================================================

exported_files_59 = [
    RESULTS_PATH_59,
    CARD_RANKING_PATH_59,
    PAIRWISE_PATH_59,
    SIDE_EFFECT_PATH_59,
    ACTION_USAGE_PATH_59,
    SUMMARY_PATH_59,
]

export_validation_rows_59 = []

for path in exported_files_59:

    export_validation_rows_59.append(
        {
            "file":
                path.name,

            "exists":
                path.exists(),

            "size_bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else 0
                ),
        }
    )


export_validation_df_59 = pd.DataFrame(
    export_validation_rows_59
)


print("\nEXPORTED HANDOFF ARTIFACTS")
print("-" * 100)

display(
    export_validation_df_59
)


assert (
    export_validation_df_59[
        "exists"
    ].all()
)

assert (
    export_validation_df_59[
        "size_bytes"
    ].gt(0).all()
)


# ==========================================================================================
# 7. Final summary
# ==========================================================================================

print("\nNOTEBOOK 59 FINAL METRICS")
print("-" * 100)

print(
    "Production model       :",
    type(
        r3_live_model
    ).__name__,
)

print(
    "Feature contract       :",
    len(
        r3_live_feature_names
    ),
)

print(
    "Controlled battles     :",
    section8a_total_battles,
)

print(
    "Live decisions         :",
    section8a_total_decisions,
)

print(
    "Model-driven decisions :",
    section8a_total_model_driven,
)

print(
    "Fallback decisions     :",
    section8a_total_fallbacks,
)

print(
    "Model-driven rate      :",
    round(
        section8a_model_driven_rate,
        6,
    ),
)

print(
    "Terminal battle rate   :",
    round(
        terminal_rate_8b,
        6,
    ),
)

print(
    "Average battle turns   :",
    round(
        section8a_average_turns,
        3,
    ),
)


print("\nCONTROLLED CARD RANKING")
print("-" * 100)

display(
    section8b_ranking_df
)


# ==========================================================================================
# 8. Notebook 60 handoff
# ==========================================================================================

print("\nNOTEBOOK 60 HANDOFF")
print("-" * 100)

print(
    "Notebook 59 has validated the live R3 production path."
)

print(
    "Notebook 60 should expand evaluation breadth rather "
    "than modify or retrain the approved R3 model."
)

print(
    "Primary Notebook 60 objective:"
)

print(
    "Large-scale side-neutral competitive benchmarking, "
    "matchup coverage, robustness analysis, and comparison "
    "against established project baselines."
)


print("\n" + "=" * 100)

print("✅ R3 PRODUCTION MODEL LIVE-INTEGRATED")
print("✅ 46-FEATURE CONTRACT VERIFIED")
print("✅ REAL SIMULATOR HANDSHAKE VERIFIED")
print("✅ MULTI-TURN MODEL-DRIVEN PLAY VERIFIED")
print("✅ 12 CONTROLLED BATTLES COMPLETED")
print("✅ 86 / 86 DECISIONS MODEL-DRIVEN")
print("✅ ZERO FALLBACK DECISIONS")
print("✅ CONTROLLED PERFORMANCE ARTIFACTS EXPORTED")
print("✅ NOTEBOOK 60 HANDOFF PACKAGE CREATED")
print("✅ NOTEBOOK 59 COMPLETE")

print("=" * 100)


# In[ ]:




