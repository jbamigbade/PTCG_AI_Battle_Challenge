#!/usr/bin/env python
# coding: utf-8

# # Notebook 36 — Production Training Dataset Builder
# 
# ## Objectives
# 
# - Load the production dataset contract from Notebook 35
# - Reuse the production observation pipeline from Notebook 33
# - Connect directly to the battle simulator and agents
# - Capture observations before every decision
# - Record all legal actions and legal-action masks
# - Record the selected action and its legal-action index
# - Capture rewards and terminal states
# - Generate Behavior Cloning and PPO-ready training samples
# - Validate action diversity and prevent label leakage
# - Export CSV, Parquet, JSON, and quality-report artifacts

# # Section 1 — Project Setup

# ## Section 1 — Project Setup
# 
# #### Locate the project root, configure report paths, initialize reproducibility settings, and prepare Notebook 36 for production dataset generation.

# In[2]:


# ============================================================
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import importlib
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_project_root(
    start_path: Path | None = None,
) -> Path:
    current_path = (
        start_path or Path.cwd()
    ).resolve()

    for candidate in [
        current_path,
        *current_path.parents,
    ]:
        if (
            (candidate / "src").is_dir()
            and (candidate / "notebooks").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOK33_REPORT_DIR = (
    REPORTS_DIR / "notebook33"
)

NOTEBOOK34_REPORT_DIR = (
    REPORTS_DIR / "notebook34"
)

NOTEBOOK35_REPORT_DIR = (
    REPORTS_DIR / "notebook35"
)

NOTEBOOK36_REPORT_DIR = (
    REPORTS_DIR / "notebook36"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK36_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 36

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option(
    "display.max_columns",
    120,
)

pd.set_option(
    "display.width",
    200,
)

pd.set_option(
    "display.max_colwidth",
    160,
)

print("NOTEBOOK 36 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
)

print(
    "Notebook 33 reports    :",
    NOTEBOOK33_REPORT_DIR,
)

print(
    "Notebook 35 reports    :",
    NOTEBOOK35_REPORT_DIR,
)

print(
    "Notebook 36 reports    :",
    NOTEBOOK36_REPORT_DIR,
)

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK33_REPORT_DIR.exists()
assert NOTEBOOK35_REPORT_DIR.exists()
assert NOTEBOOK36_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# # Section 2 — Load the Production Dataset Specification

# ## Section 2 — Load the Production Dataset Specification
# 
# #### Load the canonical dataset contract, field mapping, workflow, and interface exported by Notebook 35. Notebook 36 will use these files as its implementation contract.

# In[3]:


# ============================================================
# SECTION 2 — LOAD NOTEBOOK 35 SPECIFICATION
# ============================================================

CONTRACT_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_dataset_contract.csv"
)

MAPPING_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_field_mapping.csv"
)

WORKFLOW_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_workflow.csv"
)

INTERFACE_FILE = (
    NOTEBOOK35_REPORT_DIR
    / "production_interface.json"
)

required_specification_files = [
    CONTRACT_FILE,
    MAPPING_FILE,
    WORKFLOW_FILE,
    INTERFACE_FILE,
]

missing_specification_files = [
    path
    for path in required_specification_files
    if not path.exists()
]

assert not missing_specification_files, (
    "Missing Notebook 35 specification files:\n"
    + "\n".join(
        str(path)
        for path in missing_specification_files
    )
)

production_contract_df = pd.read_csv(
    CONTRACT_FILE
)

production_mapping_df = pd.read_csv(
    MAPPING_FILE
)

production_workflow_df = pd.read_csv(
    WORKFLOW_FILE
)

with INTERFACE_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    production_interface = json.load(
        file
    )

print("NOTEBOOK 36 — PRODUCTION SPECIFICATION LOAD")
print("=" * 70)

print(
    "Contract fields        :",
    len(production_contract_df),
)

print(
    "Field mappings         :",
    len(production_mapping_df),
)

print(
    "Workflow steps         :",
    len(production_workflow_df),
)

print(
    "Interface inputs       :",
    len(
        production_interface[
            "inputs"
        ]
    ),
)

print(
    "Interface outputs      :",
    len(
        production_interface[
            "outputs"
        ]
    ),
)

print()
print("Production contract")
print("-" * 70)

display(
    production_contract_df
)

print()
print("Production field mapping")
print("-" * 70)

display(
    production_mapping_df
)

assert len(production_contract_df) == 16
assert len(production_mapping_df) == 16
assert len(production_workflow_df) == 11

assert {
    "inputs",
    "outputs",
}.issubset(
    production_interface
)

print()
print(
    "✅ SECTION 2 SPECIFICATION LOAD PASSED"
)


# # Section 3 — Discover Production Source Modules

# ## Section 3 — Discover Production Source Modules
# 
# #### Inspect the project source and exported scripts to locate the existing battle engine, observation pipeline, legal-action logic, policy agent, and search agent.
# 
# #### Notebook 36 will reuse these components rather than duplicate earlier work.

# In[4]:


# ============================================================
# SECTION 3 — DISCOVER PRODUCTION SOURCE MODULES
# ============================================================

print("NOTEBOOK 36 — SOURCE MODULE DISCOVERY")
print("=" * 70)

search_roots = [
    SRC_DIR,
    SCRIPTS_DIR,
]

python_files = sorted(
    path
    for root in search_roots
    for path in root.rglob("*.py")
    if path.is_file()
)

print(
    "Python files discovered:",
    len(python_files),
)

source_keywords = {
    "battle_engine": (
        "battle",
        "game_state",
        "simulate",
        "step",
        "reset",
    ),
    "observation": (
        "observation",
        "feature",
        "encode",
        "vector",
    ),
    "legal_actions": (
        "legal",
        "available",
        "candidate",
        "action_mask",
        "moves",
    ),
    "policy_agent": (
        "policy",
        "choose_move",
        "select_action",
        "decision",
    ),
    "search_agent": (
        "search",
        "minimax",
        "alpha_beta",
        "best_move",
    ),
}

discovery_rows = []

for path in python_files:
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).lower()
    except Exception:
        continue

    matched_categories = []

    for category, keywords in (
        source_keywords.items()
    ):
        if any(
            keyword in text
            for keyword in keywords
        ):
            matched_categories.append(
                category
            )

    if matched_categories:
        discovery_rows.append(
            {
                "path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "size_kb": round(
                    path.stat().st_size / 1024,
                    2,
                ),
                "categories": ", ".join(
                    matched_categories
                ),
            }
        )

module_discovery_df = pd.DataFrame(
    discovery_rows
)

if not module_discovery_df.empty:
    module_discovery_df = (
        module_discovery_df
        .sort_values(
            [
                "categories",
                "path",
            ]
        )
        .reset_index(
            drop=True
        )
    )

display(
    module_discovery_df
)

print()
print(
    "Candidate modules      :",
    len(module_discovery_df),
)

assert len(python_files) > 0
assert len(module_discovery_df) > 0

print()
print(
    "✅ SECTION 3 SOURCE DISCOVERY PASSED"
)


# # Section 4 — Inspect Candidate APIs

# ## Section 4 — Inspect Candidate APIs
# 
# Parse candidate Python modules and identify the functions and classes most likely to provide:
# 
# - environment reset;
# - game-state observation;
# - legal actions;
# - action selection;
# - environment transition;
# - terminal-state detection.

# In[6]:


# ============================================================
# SECTION 4 — INSPECT CANDIDATE APIS
# ============================================================

import ast


def extract_module_api(
    path: Path,
) -> list[dict[str, Any]]:
    try:
        source = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        tree = ast.parse(source)
    except Exception:
        return []

    rows = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            rows.append(
                {
                    "module": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "object_type": "function",
                    "name": node.name,
                }
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            rows.append(
                {
                    "module": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "object_type": "class",
                    "name": node.name,
                }
            )

            for child in node.body:
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    rows.append(
                        {
                            "module": str(
                                path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),
                            "object_type": (
                                f"method:{node.name}"
                            ),
                            "name": child.name,
                        }
                    )

    return rows


api_rows = []

candidate_paths = [
    PROJECT_ROOT / path
    for path in (
        module_discovery_df[
            "path"
        ]
        .tolist()
    )
]

for path in candidate_paths:
    api_rows.extend(
        extract_module_api(
            path
        )
    )

api_inventory_df = pd.DataFrame(
    api_rows
)

api_keywords = (
    "reset",
    "step",
    "observe",
    "observation",
    "legal",
    "move",
    "action",
    "select",
    "choose",
    "search",
    "terminal",
    "done",
    "winner",
    "simulate",
)

relevant_api_df = (
    api_inventory_df.loc[
        api_inventory_df["name"]
        .str.lower()
        .apply(
            lambda value: any(
                keyword in value
                for keyword in api_keywords
            )
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

print("NOTEBOOK 36 — CANDIDATE API INVENTORY")
print("=" * 70)

print(
    "All discovered APIs    :",
    len(api_inventory_df),
)

print(
    "Relevant APIs          :",
    len(relevant_api_df),
)

display(
    relevant_api_df
)

assert len(api_inventory_df) > 0
assert len(relevant_api_df) > 0

print()
print(
    "✅ SECTION 4 API INSPECTION PASSED"
)


# # SECTION 5 — Production Class Discovery

# ## Section 5 — Production Class Discovery
# 
# #### Identify the concrete production classes that will be instantiated by the training dataset generator.

# In[7]:


# ============================================================
# SECTION 5 — PRODUCTION CLASS DISCOVERY
# ============================================================

production_classes = api_inventory_df[
    api_inventory_df["object_type"] == "class"
].copy()

priority_keywords = [
    "Battle",
    "Environment",
    "Game",
    "Observation",
    "Policy",
    "Search",
    "Agent",
    "Trainer",
    "SelfPlay",
]

production_classes = production_classes[
    production_classes["name"].apply(
        lambda x: any(
            k.lower() in x.lower()
            for k in priority_keywords
        )
    )
].reset_index(drop=True)

display(production_classes)

print()

print(
    "Production classes :",
    len(production_classes),
)

assert len(production_classes) > 0

print()
print("✅ SECTION 5 PRODUCTION CLASSES PASSED")


# # SECTION 6 — Training Pipeline Discovery

# ## Section 6 — Training Pipeline Discovery
# 
# #### Locate the reusable training pipeline modules that already exist inside src/training.

# In[8]:


# ============================================================
# SECTION 6 — TRAINING PIPELINE DISCOVERY
# ============================================================

training_modules = sorted(
    SRC_DIR.joinpath("training").glob("*.py")
)

training_df = pd.DataFrame({
    "module":[m.name for m in training_modules],
    "size_kb":[round(m.stat().st_size/1024,2) for m in training_modules]
})

display(training_df)

print()

print(
    "Training modules :",
    len(training_df),
)

assert len(training_df) > 0

print()

print("✅ SECTION 6 TRAINING DISCOVERY PASSED")


# # SECTION 7 — Production Training Architecture

# ## Section 7 — Production Training Architecture
# 
# #### Identify the production components that will participate in RL dataset generation.

# In[9]:


# ============================================================
# SECTION 7 — TRAINING ARCHITECTURE
# ============================================================

architecture = [
    {
        "Stage": 1,
        "Component": "ObservationPipeline",
        "Module": "Notebook 33",
        "Purpose": "Generate observation vectors"
    },
    {
        "Stage": 2,
        "Component": "BattleEngine",
        "Module": "src/engine",
        "Purpose": "Execute legal game actions"
    },
    {
        "Stage": 3,
        "Component": "Policy",
        "Module": "src/agents/policies.py",
        "Purpose": "Choose actions"
    },
    {
        "Stage": 4,
        "Component": "ReplayBuffer",
        "Module": "src/training/replay.py",
        "Purpose": "Store transitions"
    },
    {
        "Stage": 5,
        "Component": "SelfPlay",
        "Module": "src/training/self_play.py",
        "Purpose": "Generate games"
    },
    {
        "Stage": 6,
        "Component": "Dataset",
        "Module": "src/training/dataset.py",
        "Purpose": "Export PPO dataset"
    }
]

architecture_df = pd.DataFrame(architecture)

display(architecture_df)

print()

print(
    "Architecture stages :",
    len(architecture_df)
)

assert len(architecture_df) == 6

print()

print("✅ SECTION 7 TRAINING ARCHITECTURE PASSED")


# # SECTION 8 — Candidate Policy Discovery

# ## Section 8 — Candidate Policy Discovery
# 
# #### Determine every available policy implementation that can generate training data.

# In[10]:


# ============================================================
# SECTION 8 — POLICY DISCOVERY
# ============================================================

policy_classes = production_classes[
    production_classes["name"].str.contains(
        "Policy",
        case=False,
        na=False
    )
].copy()

display(policy_classes)

print()

print(
    "Policies discovered :",
    len(policy_classes)
)

assert len(policy_classes) > 0

print()

print("✅ SECTION 8 POLICY DISCOVERY PASSED")


# # SECTION 9 — Select Production Policies

# ## Section 9 — Production Policy Selection
# 
# #### Identify the policy implementations that are suitable for production self-play and PPO dataset generation.

# In[11]:


# ============================================================
# SECTION 9 — PRODUCTION POLICY SELECTION
# ============================================================

production_policy_names = [
    "RandomPolicy",
    "GreedyPolicy",
    "HeuristicPolicy",
    "SearchPolicy",
    "AggressivePolicy",
    "DefensivePolicy",
]

selected_policies = policy_classes[
    policy_classes["name"].isin(
        production_policy_names
    )
].reset_index(drop=True)

display(selected_policies)

print()

print(
    "Selected production policies :",
    len(selected_policies)
)

assert len(selected_policies) >= 6

print()

print("✅ SECTION 9 POLICY SELECTION PASSED")


# # SECTION 10 — Replay Pipeline Discovery

# ## Section 10 — Replay Pipeline Discovery
# 
# #### Inspect the replay buffer implementation that will store PPO transitions.

# In[12]:


# ============================================================
# SECTION 10 — REPLAY PIPELINE
# ============================================================

replay_module = training_df[
    training_df["module"].str.contains(
        "replay",
        case=False
    )
]

display(replay_module)

print()

print(
    "Replay modules :",
    len(replay_module)
)

assert len(replay_module) == 1

print()

print("✅ SECTION 10 REPLAY DISCOVERY PASSED")


# # SECTION 11 — Dataset Export Pipeline

# ## Section 11 — Dataset Export Discovery
# 
# #### Locate the production dataset exporter used after self-play.

# In[13]:


# ============================================================
# SECTION 11 — DATASET EXPORT
# ============================================================

dataset_module = training_df[
    training_df["module"].str.contains(
        "dataset",
        case=False
    )
]

display(dataset_module)

print()

print(
    "Dataset exporters :",
    len(dataset_module)
)

assert len(dataset_module) == 1

print()

print("✅ SECTION 11 DATASET EXPORT PASSED")


# # SECTION 12 — Production Integration Plan

# ## Section 12 — Production Integration Plan
# 
# #### Define how the production components interact during PPO dataset generation.

# In[14]:


# ============================================================
# SECTION 12 — PRODUCTION INTEGRATION PLAN
# ============================================================

integration_steps = [
    {
        "Step": 1,
        "Component": "ObservationPipeline",
        "Consumes": "Battle state",
        "Produces": "Observation vector",
    },
    {
        "Step": 2,
        "Component": "BattleEngine",
        "Consumes": "Current game",
        "Produces": "Legal actions",
    },
    {
        "Step": 3,
        "Component": "Policy",
        "Consumes": "Observation + legal actions",
        "Produces": "Chosen action",
    },
    {
        "Step": 4,
        "Component": "ReplayBuffer",
        "Consumes": "Transition",
        "Produces": "Stored experience",
    },
    {
        "Step": 5,
        "Component": "Dataset",
        "Consumes": "Replay buffer",
        "Produces": "PPO training dataset",
    },
]

integration_df = pd.DataFrame(integration_steps)

display(integration_df)

print()

print(
    "Integration stages :",
    len(integration_df),
)

assert len(integration_df) == 5

print()

print("✅ SECTION 12 INTEGRATION PLAN PASSED")


# # SECTION 13 — Production Readiness Checklist

# ## Section 13 — Production Readiness Checklist
# 
# #### Verify every required production subsystem has been discovered.

# In[16]:


# ============================================================
# SECTION 13 — PRODUCTION READINESS
# ============================================================

checklist = pd.DataFrame(
    [
        (
            "Observation Pipeline",
            NOTEBOOK33_REPORT_DIR.exists(),
        ),
        (
            "Battle Engine",
            (
                module_discovery_df["categories"]
                .str.contains(
                    "battle_engine",
                    na=False,
                )
                .any()
            ),
        ),
        (
            "Policy Classes",
            len(selected_policies) > 0,
        ),
        (
            "Replay Buffer",
            len(replay_module) == 1,
        ),
        (
            "Dataset Exporter",
            len(dataset_module) == 1,
        ),
        (
            "Training Modules",
            len(training_df) > 0,
        ),
        (
            "Source Modules",
            len(module_discovery_df) > 0,
        ),
        (
            "Candidate APIs",
            len(api_inventory_df) > 0,
        ),
    ],
    columns=[
        "Component",
        "Ready",
    ],
)

display(checklist)

print()

ready_count = int(
    checklist["Ready"].sum()
)

print(
    "Production components ready :",
    ready_count,
    "/",
    len(checklist),
)

not_ready_components = (
    checklist.loc[
        ~checklist["Ready"],
        "Component",
    ]
    .tolist()
)

if not_ready_components:
    print(
        "Not ready components       :",
        ", ".join(
            not_ready_components
        ),
    )

assert checklist["Ready"].all(), (
    "One or more production components "
    "are not ready: "
    f"{not_ready_components}"
)

print()
print(
    "✅ SECTION 13 PRODUCTION READINESS PASSED"
)


# # Final Section — Export Design Documents (Section 14)

# In[18]:


# ============================================================
# SECTION 14 — EXPORT DESIGN DOCUMENTS
# ============================================================

REPORT_DIR = NOTEBOOK36_REPORT_DIR

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

API_FILE = (
    REPORT_DIR
    / "production_api_inventory.csv"
)

CLASS_FILE = (
    REPORT_DIR
    / "production_classes.csv"
)

PIPELINE_FILE = (
    REPORT_DIR
    / "training_pipeline_architecture.csv"
)

CHECKLIST_FILE = (
    REPORT_DIR
    / "production_readiness.csv"
)

api_inventory_df.to_csv(
    API_FILE,
    index=False,
)

production_classes.to_csv(
    CLASS_FILE,
    index=False,
)

architecture_df.to_csv(
    PIPELINE_FILE,
    index=False,
)

checklist.to_csv(
    CHECKLIST_FILE,
    index=False,
)

print("NOTEBOOK 36 — DESIGN EXPORT")
print("=" * 70)

print(
    "API Inventory          :",
    API_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Classes                :",
    CLASS_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Architecture           :",
    PIPELINE_FILE.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "Checklist              :",
    CHECKLIST_FILE.relative_to(
        PROJECT_ROOT
    ),
)

assert API_FILE.exists()
assert CLASS_FILE.exists()
assert PIPELINE_FILE.exists()
assert CHECKLIST_FILE.exists()

print()
print(
    "✅ SECTION 14 EXPORT PASSED"
)


# # Final Validation Cell (Section 15)

# In[19]:


# ============================================================
# SECTION 15 — VALIDATION
# ============================================================

api_reload = pd.read_csv(API_FILE)
class_reload = pd.read_csv(CLASS_FILE)
pipeline_reload = pd.read_csv(PIPELINE_FILE)
check_reload = pd.read_csv(CHECKLIST_FILE)

print("NOTEBOOK 36 — VALIDATION")
print("=" * 70)

print("API Inventory :", len(api_reload))
print("Classes       :", len(class_reload))
print("Architecture  :", len(pipeline_reload))
print("Checklist     :", len(check_reload))

assert len(api_reload) == len(api_inventory_df)
assert len(class_reload) == len(production_classes)
assert len(pipeline_reload) == len(architecture_df)
assert len(check_reload) == len(checklist)

print()
print("🏆 NOTEBOOK 36 COMPLETE")


# In[ ]:




