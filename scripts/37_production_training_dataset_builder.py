#!/usr/bin/env python
# coding: utf-8

# # Notebook 37 — Production Training Dataset Builder
# 
# ## Objectives
# 
# - Load the production specifications from Notebook 35
# - Load the integration and readiness artifacts from Notebook 36
# - Reuse the observation pipeline and training modules already present in `src`
# - Build structured production training samples
# - Record observations, legal moves, masks, chosen actions, rewards, and terminal flags
# - Validate schema completeness and action diversity
# - Export CSV, Parquet, JSON, and quality reports

# ## Section 1 — Project Setup
# 
# #### Locate the project root, configure input and output paths, initialize reproducibility settings, and prepare Notebook 37 for production dataset generation.

# In[1]:


# ============================================================
# SECTION 1 — PROJECT SETUP
# ============================================================

from __future__ import annotations

import ast
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

NOTEBOOK35_REPORT_DIR = (
    REPORTS_DIR / "notebook35"
)

NOTEBOOK36_REPORT_DIR = (
    REPORTS_DIR / "notebook36"
)

NOTEBOOK37_REPORT_DIR = (
    REPORTS_DIR / "notebook37"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

NOTEBOOK37_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 37

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option(
    "display.max_columns",
    140,
)

pd.set_option(
    "display.width",
    220,
)

pd.set_option(
    "display.max_colwidth",
    180,
)

print("NOTEBOOK 37 — PROJECT SETUP")
print("=" * 70)

print(
    "Project root           :",
    PROJECT_ROOT,
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
    "Notebook 37 reports    :",
    NOTEBOOK37_REPORT_DIR,
)

print(
    "Random seed            :",
    RANDOM_SEED,
)

assert SRC_DIR.exists()
assert SCRIPTS_DIR.exists()
assert NOTEBOOKS_DIR.exists()
assert REPORTS_DIR.exists()
assert NOTEBOOK35_REPORT_DIR.exists()
assert NOTEBOOK36_REPORT_DIR.exists()
assert NOTEBOOK37_REPORT_DIR.exists()

print()
print(
    "✅ SECTION 1 SETUP PASSED"
)


# ## Section 2 — Load Production Specifications
# 
# #### Load the canonical dataset contract from Notebook 35 and the production integration artifacts from Notebook 36.

# In[2]:


# ============================================================
# SECTION 2 — LOAD PRODUCTION SPECIFICATIONS
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

API_INVENTORY_FILE = (
    NOTEBOOK36_REPORT_DIR
    / "production_api_inventory.csv"
)

PRODUCTION_CLASSES_FILE = (
    NOTEBOOK36_REPORT_DIR
    / "production_classes.csv"
)

ARCHITECTURE_FILE = (
    NOTEBOOK36_REPORT_DIR
    / "training_pipeline_architecture.csv"
)

READINESS_FILE = (
    NOTEBOOK36_REPORT_DIR
    / "production_readiness.csv"
)

required_files = [
    CONTRACT_FILE,
    MAPPING_FILE,
    WORKFLOW_FILE,
    INTERFACE_FILE,
    API_INVENTORY_FILE,
    PRODUCTION_CLASSES_FILE,
    ARCHITECTURE_FILE,
    READINESS_FILE,
]

missing_files = [
    path
    for path in required_files
    if not path.exists()
]

assert not missing_files, (
    "Missing production specification files:\n"
    + "\n".join(
        str(path)
        for path in missing_files
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

api_inventory_df = pd.read_csv(
    API_INVENTORY_FILE
)

production_classes_df = pd.read_csv(
    PRODUCTION_CLASSES_FILE
)

architecture_df = pd.read_csv(
    ARCHITECTURE_FILE
)

readiness_df = pd.read_csv(
    READINESS_FILE
)

print("NOTEBOOK 37 — SPECIFICATION LOAD")
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
    "API inventory rows     :",
    len(api_inventory_df),
)

print(
    "Production classes     :",
    len(production_classes_df),
)

print(
    "Architecture stages    :",
    len(architecture_df),
)

print(
    "Readiness checks       :",
    len(readiness_df),
)

print()
print("Production contract")
print("-" * 70)

display(
    production_contract_df
)

assert len(production_contract_df) == 16
assert len(production_mapping_df) == 16
assert len(production_workflow_df) == 11
assert readiness_df["Ready"].all()

print()
print(
    "✅ SECTION 2 SPECIFICATION LOAD PASSED"
)


# ## Section 3 — Inspect Training Module APIs
# 
# #### Inspect the concrete APIs exposed by `src/training/dataset.py`, `src/training/replay.py`, and `src/training/self_play.py` before importing or instantiating them.

# In[4]:


# ============================================================
# SECTION 3 — INSPECT TRAINING MODULE APIS
# ============================================================

training_paths = {
    "dataset": (
        SRC_DIR / "training" / "dataset.py"
    ),
    "replay": (
        SRC_DIR / "training" / "replay.py"
    ),
    "self_play": (
        SRC_DIR / "training" / "self_play.py"
    ),
}

for name, path in training_paths.items():
    assert path.exists(), (
        f"Missing training module: {path}"
    )


def extract_api_rows(
    module_name: str,
    path: Path,
) -> list[dict[str, Any]]:
    # utf-8-sig automatically removes a leading BOM.
    source = path.read_text(
        encoding="utf-8-sig",
        errors="ignore",
    )

    # Extra protection in case a BOM appears elsewhere.
    source = source.replace(
        "\ufeff",
        "",
    )

    try:
        tree = ast.parse(
            source,
            filename=str(path),
        )
    except SyntaxError as exc:
        print(
            f"[WARNING] Could not parse {path.name}: "
            f"{exc}"
        )
        return []

    rows = []

    for node in tree.body:
        if isinstance(
            node,
            ast.ClassDef,
        ):
            rows.append(
                {
                    "module": module_name,
                    "object_type": "class",
                    "class_name": node.name,
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
                            "module": module_name,
                            "object_type": "method",
                            "class_name": node.name,
                            "name": child.name,
                        }
                    )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            rows.append(
                {
                    "module": module_name,
                    "object_type": "function",
                    "class_name": None,
                    "name": node.name,
                }
            )

    return rows


training_api_rows = []

for module_name, path in (
    training_paths.items()
):
    training_api_rows.extend(
        extract_api_rows(
            module_name,
            path,
        )
    )

training_api_df = pd.DataFrame(
    training_api_rows
)

print("NOTEBOOK 37 — TRAINING API INSPECTION")
print("=" * 70)

print(
    "Training APIs found    :",
    len(training_api_df),
)

display(
    training_api_df
)

assert len(training_api_df) > 0, (
    "No training APIs were discovered."
)

print()
print(
    "✅ SECTION 3 TRAINING API INSPECTION PASSED"
)


# ## Section 4 — Inspect Training API Signatures
# 
# ### Inspect the constructor, function, and method signatures of the existing training components before using them.
# 
# #### This prevents Notebook 37 from assuming incompatible parameters.

# In[5]:


# ============================================================
# SECTION 4 — INSPECT TRAINING API SIGNATURES
# ============================================================

import ast


def format_arguments(
    arguments: ast.arguments,
) -> str:
    positional_args = [
        argument.arg
        for argument in (
            arguments.posonlyargs
            + arguments.args
        )
    ]

    defaults = [
        None
    ] * (
        len(positional_args)
        - len(arguments.defaults)
    ) + list(arguments.defaults)

    formatted = []

    for name, default in zip(
        positional_args,
        defaults,
    ):
        if default is None:
            formatted.append(name)
        else:
            try:
                default_text = ast.unparse(
                    default
                )
            except Exception:
                default_text = "..."

            formatted.append(
                f"{name}={default_text}"
            )

    if arguments.vararg is not None:
        formatted.append(
            f"*{arguments.vararg.arg}"
        )

    for argument, default in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
    ):
        if default is None:
            formatted.append(
                argument.arg
            )
        else:
            try:
                default_text = ast.unparse(
                    default
                )
            except Exception:
                default_text = "..."

            formatted.append(
                f"{argument.arg}={default_text}"
            )

    if arguments.kwarg is not None:
        formatted.append(
            f"**{arguments.kwarg.arg}"
        )

    return ", ".join(
        formatted
    )


def extract_signatures(
    module_name: str,
    path: Path,
) -> list[dict[str, Any]]:
    source = path.read_text(
        encoding="utf-8-sig",
        errors="ignore",
    ).replace(
        "\ufeff",
        "",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in tree.body:
        if isinstance(
            node,
            ast.ClassDef,
        ):
            rows.append(
                {
                    "module": module_name,
                    "object_type": "class",
                    "class_name": node.name,
                    "name": node.name,
                    "signature": "",
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
                            "module": module_name,
                            "object_type": "method",
                            "class_name": node.name,
                            "name": child.name,
                            "signature": (
                                f"{child.name}("
                                f"{format_arguments(child.args)}"
                                ")"
                            ),
                        }
                    )

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            rows.append(
                {
                    "module": module_name,
                    "object_type": "function",
                    "class_name": None,
                    "name": node.name,
                    "signature": (
                        f"{node.name}("
                        f"{format_arguments(node.args)}"
                        ")"
                    ),
                }
            )

    return rows


signature_rows = []

for module_name, path in (
    training_paths.items()
):
    signature_rows.extend(
        extract_signatures(
            module_name,
            path,
        )
    )

training_signatures_df = pd.DataFrame(
    signature_rows
)

relevant_signature_names = {
    "ReplayStep",
    "ReplayGame",
    "ReplayRecorder",
    "__init__",
    "add_step",
    "add_game",
    "iter_steps",
    "run_self_play_session",
    "replay_to_dataframe",
    "export_csv",
}

relevant_signatures_df = (
    training_signatures_df.loc[
        training_signatures_df[
            "name"
        ].isin(
            relevant_signature_names
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

print("NOTEBOOK 37 — TRAINING SIGNATURE INSPECTION")
print("=" * 70)

print(
    "Relevant signatures    :",
    len(relevant_signatures_df),
)

display(
    relevant_signatures_df
)

assert len(relevant_signatures_df) > 0

print()
print(
    "✅ SECTION 4 SIGNATURE INSPECTION PASSED"
)


# ## Section 5 — Inspect Replay Data Fields
# 
# #### Inspect the fields defined on `ReplayStep`, `ReplayGame`, and `SelfPlayRunSummary`.
# 
# #### These structures determine whether the current replay system already stores observations, legal actions, rewards, and terminal information.

# In[6]:


# ============================================================
# SECTION 5 — INSPECT REPLAY DATA FIELDS
# ============================================================

target_classes = {
    "ReplayStep",
    "ReplayGame",
    "SelfPlayRunSummary",
}


def extract_class_fields(
    module_name: str,
    path: Path,
) -> list[dict[str, Any]]:
    source = path.read_text(
        encoding="utf-8-sig",
        errors="ignore",
    ).replace(
        "\ufeff",
        "",
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in tree.body:
        if not isinstance(
            node,
            ast.ClassDef,
        ):
            continue

        if node.name not in target_classes:
            continue

        decorator_names = []

        for decorator in node.decorator_list:
            try:
                decorator_names.append(
                    ast.unparse(
                        decorator
                    )
                )
            except Exception:
                decorator_names.append(
                    "unknown"
                )

        for child in node.body:
            if isinstance(
                child,
                ast.AnnAssign,
            ) and isinstance(
                child.target,
                ast.Name,
            ):
                try:
                    annotation = ast.unparse(
                        child.annotation
                    )
                except Exception:
                    annotation = "unknown"

                if child.value is None:
                    default_value = None
                else:
                    try:
                        default_value = ast.unparse(
                            child.value
                        )
                    except Exception:
                        default_value = "..."

                rows.append(
                    {
                        "module": module_name,
                        "class_name": node.name,
                        "decorators": ", ".join(
                            decorator_names
                        ),
                        "field": child.target.id,
                        "annotation": annotation,
                        "default": default_value,
                    }
                )

    return rows


replay_field_rows = []

for module_name, path in (
    training_paths.items()
):
    replay_field_rows.extend(
        extract_class_fields(
            module_name,
            path,
        )
    )

replay_fields_df = pd.DataFrame(
    replay_field_rows
)

print("NOTEBOOK 37 — REPLAY FIELD INSPECTION")
print("=" * 70)

print(
    "Replay fields found    :",
    len(replay_fields_df),
)

display(
    replay_fields_df
)

if not replay_fields_df.empty:
    stored_fields = set(
        replay_fields_df["field"]
    )
else:
    stored_fields = set()

required_transition_fields = {
    "observation_vector",
    "legal_moves",
    "legal_move_mask",
    "chosen_move_index",
    "reward",
    "done",
}

transition_field_audit = pd.DataFrame(
    [
        {
            "required_field": field,
            "already_stored": (
                field in stored_fields
            ),
        }
        for field in sorted(
            required_transition_fields
        )
    ]
)

print()
print("Production transition-field audit")
print("-" * 70)

display(
    transition_field_audit
)

assert len(replay_fields_df) > 0

print()
print(
    "✅ SECTION 5 REPLAY FIELD INSPECTION PASSED"
)


# In[ ]:




