#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# NOTEBOOK 63 — CELL 1
# FINAL STRATEGY SIMULATION CAMPAIGN
# CERTIFIED NB62 HANDOFF + READ-ONLY ENVIRONMENT INITIALIZATION
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — FINAL STRATEGY SIMULATION CAMPAIGN")
print("CELL 1 — CERTIFIED NB62 HANDOFF + ENVIRONMENT INITIALIZATION")
print("=" * 92)


# =============================================================================
# 1. PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\02_AI_and_Data\Kaggle-AI-Agents\PTCG_AI_Battle_Challenge"
).resolve()

SUBMISSION_DIR = PROJECT_ROOT / "submission"

FINAL_ARCHIVE = (
    SUBMISSION_DIR
    / "ptcg_final_agent.zip"
)

FINAL_MAIN = (
    SUBMISSION_DIR
    / "main.py"
)

CERTIFIED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "notebook60"
    / "notebook60_certified_61_feature_random_forest.joblib"
)

NB63_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "notebook63"
)

NB63_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "notebook63"
)

NB63_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook63"
)


assert PROJECT_ROOT.is_dir()
assert SUBMISSION_DIR.is_dir()
assert FINAL_ARCHIVE.is_file()
assert FINAL_MAIN.is_file()
assert CERTIFIED_MODEL.is_file()


print()
print("Project root     :", PROJECT_ROOT)
print("Submission       :", SUBMISSION_DIR)
print("Final archive    :", FINAL_ARCHIVE)
print("Authoritative main:", FINAL_MAIN)
print("Certified model  :", CERTIFIED_MODEL)


# =============================================================================
# 2. HASH HELPER
# =============================================================================

def sha256_file_nb63(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# 3. CERTIFIED HASH FREEZE
# =============================================================================

EXPECTED_MAIN_SHA256 = (
    "e328508c39b6c48411f46ad43f5ec1ffe34f54f740aed24389f88d8166dc97cd"
)

EXPECTED_MODEL_SHA256 = (
    "615baf3d5725ccca0c1c94e37f62fb24a68613248e167bb566bef4a6d50f2581"
)

EXPECTED_MODEL_SIZE = 22396321


main_hash_nb63 = sha256_file_nb63(
    FINAL_MAIN
)

model_hash_nb63 = sha256_file_nb63(
    CERTIFIED_MODEL
)

model_size_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


print()
print("=" * 92)
print("CERTIFIED HANDOFF FREEZE")
print("=" * 92)

print("main.py SHA256 :", main_hash_nb63)
print("Model SHA256   :", model_hash_nb63)
print("Model size     :", model_size_nb63)


assert (
    main_hash_nb63
    ==
    EXPECTED_MAIN_SHA256
), (
    "Authoritative NB62 main.py has changed."
)

assert (
    model_hash_nb63
    ==
    EXPECTED_MODEL_SHA256
), (
    "Certified NB60 model hash has changed."
)

assert (
    model_size_nb63
    ==
    EXPECTED_MODEL_SIZE
), (
    "Certified NB60 model size has changed."
)


print()
print("[OK] Authoritative main.py preserved.")
print("[OK] Certified model preserved.")


# =============================================================================
# 4. CREATE NOTEBOOK 63 ANALYSIS DIRECTORIES
# =============================================================================

for directory in [
    NB63_ARTIFACT_DIR,
    NB63_OUTPUT_DIR,
    NB63_REPORT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


assert (
    NB63_ARTIFACT_DIR.resolve()
    !=
    SUBMISSION_DIR.resolve()
)

assert (
    NB63_OUTPUT_DIR.resolve()
    !=
    SUBMISSION_DIR.resolve()
)

assert (
    NB63_REPORT_DIR.resolve()
    !=
    SUBMISSION_DIR.resolve()
)


print()
print("NB63 artifacts :", NB63_ARTIFACT_DIR)
print("NB63 outputs   :", NB63_OUTPUT_DIR)
print("NB63 reports   :", NB63_REPORT_DIR)

print()
print("[OK] Notebook 63 evidence directories isolated from submission.")


# =============================================================================
# 5. CAMPAIGN POLICY
# =============================================================================

NB63_CAMPAIGN_POLICY = {

    "purpose":
        "FINAL_STRATEGY_SIMULATION_EVIDENCE",

    "certified_submission_frozen":
        True,

    "certified_model_frozen":
        True,

    "submission_writes_allowed":
        False,

    "model_training_allowed":
        False,

    "model_retraining_allowed":
        False,

    "simulation_allowed":
        True,

    "analysis_artifacts_allowed":
        True,

    "artifact_directory":
        str(NB63_ARTIFACT_DIR),

    "output_directory":
        str(NB63_OUTPUT_DIR),

    "report_directory":
        str(NB63_REPORT_DIR),
}


print()
print("=" * 92)
print("NOTEBOOK 63 CAMPAIGN POLICY")
print("=" * 92)

print(
    json.dumps(
        NB63_CAMPAIGN_POLICY,
        indent=2,
    )
)


# =============================================================================
# 6. FINAL STATUS
# =============================================================================

CELL1_NB63_STATUS = {

    "project_root_resolved":
        True,

    "nb62_main_certified":
        True,

    "certified_model_located":
        True,

    "certified_model_integrity_preserved":
        True,

    "analysis_directories_isolated":
        True,

    "submission_write_prohibited":
        True,

    "retraining_prohibited":
        True,

    "campaign_purpose":
        "FINAL_STRATEGY_SIMULATION_EVIDENCE",

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 1 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL1_NB63_STATUS,
        indent=2,
    )
)

print()
print(
    "Certified NB62 runtime and certified NB60 model successfully handed off."
)

print(
    "Notebook 63 may now begin controlled simulation evidence generation."
)


# In[2]:


# =============================================================================
# NOTEBOOK 63 — CELL 2
# SIMULATOR + RUNTIME ASSET DISCOVERY
# READ-ONLY INVENTORY
# =============================================================================

from __future__ import annotations

import ast
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 2")
print("SIMULATOR + RUNTIME ASSET DISCOVERY")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITE
# =============================================================================

assert (
    CELL1_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 1 must PASS before Cell 2."

print()
print("[OK] Cell 1 PASS confirmed.")
print("[OK] Discovery mode is read-only.")


# =============================================================================
# 1. CANDIDATE SIMULATION FILE DISCOVERY
# =============================================================================

simulation_patterns_nb63 = [
    "*simulat*.py",
    "*battle*.py",
    "*tournament*.py",
    "*agent*.py",
    "*policy*.py",
]

simulation_candidates_nb63 = set()


for pattern in simulation_patterns_nb63:

    for path in PROJECT_ROOT.rglob(pattern):

        if not path.is_file():
            continue

        # Ignore virtual environment / caches.
        path_text = str(path).lower()

        if "\\.venv\\" in path_text:
            continue

        if "\\__pycache__\\" in path_text:
            continue

        simulation_candidates_nb63.add(
            path.resolve()
        )


simulation_candidates_nb63 = sorted(
    simulation_candidates_nb63,
    key=lambda p: str(p).lower(),
)


print()
print("=" * 92)
print("1. SIMULATION / AGENT / POLICY CANDIDATES")
print("=" * 92)

print(
    "Candidate Python files:",
    len(simulation_candidates_nb63),
)


for path in simulation_candidates_nb63:

    try:
        relative = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative = path

    print(" ", relative)


assert simulation_candidates_nb63, (
    "No simulation-related Python files were discovered."
)


# =============================================================================
# 2. KNOWN DEPLOYMENT RUNTIME FILES
# =============================================================================

known_runtime_candidates_nb63 = [
    SUBMISSION_DIR / "final_agent" / "src" / "battle_simulation.py",
    SUBMISSION_DIR / "final_agent" / "src" / "battle_state.py",
    SUBMISSION_DIR / "final_agent" / "src" / "battle_agent.py",
    SUBMISSION_DIR / "final_agent" / "src" / "simulator.py",
    SUBMISSION_DIR / "final_agent" / "src" / "legal_moves.py",
    SUBMISSION_DIR / "final_agent" / "src" / "agent_decision.py",
    SUBMISSION_DIR / "final_agent" / "src" / "agents" / "final_ppo_agent.py",
    SUBMISSION_DIR / "final_agent" / "src" / "ppo" / "ppo_policy_engine.py",
]


runtime_inventory_nb63 = []


for path in known_runtime_candidates_nb63:

    exists = path.is_file()

    record = {
        "path": str(path),
        "exists": exists,
        "bytes": (
            path.stat().st_size
            if exists
            else None
        ),
    }

    runtime_inventory_nb63.append(
        record
    )


print()
print("=" * 92)
print("2. KNOWN DEPLOYMENT RUNTIME INVENTORY")
print("=" * 92)


for record in runtime_inventory_nb63:

    marker = (
        "[OK]"
        if record["exists"]
        else "[MISSING]"
    )

    print(
        marker,
        Path(record["path"]).relative_to(
            PROJECT_ROOT
        ),
        "|",
        record["bytes"],
        "bytes",
    )


existing_runtime_count_nb63 = sum(
    record["exists"]
    for record in runtime_inventory_nb63
)


print()
print(
    "Known runtime files present:",
    existing_runtime_count_nb63,
    "/",
    len(runtime_inventory_nb63),
)


# =============================================================================
# 3. STATIC FUNCTION / CLASS INVENTORY
# =============================================================================

runtime_symbol_inventory_nb63 = {}


for record in runtime_inventory_nb63:

    if not record["exists"]:
        continue

    path = Path(
        record["path"]
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
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
            classes.append(
                node.name
            )


    runtime_symbol_inventory_nb63[
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    ] = {
        "functions": functions,
        "classes": classes,
    }


print()
print("=" * 92)
print("3. STATIC RUNTIME SYMBOL INVENTORY")
print("=" * 92)


for relative_path, symbols in runtime_symbol_inventory_nb63.items():

    print()
    print(relative_path)

    print(
        "  Functions:",
        symbols["functions"],
    )

    print(
        "  Classes  :",
        symbols["classes"],
    )


# =============================================================================
# 4. SEARCH FOR SIMULATION ENTRY-POINT SIGNALS
# =============================================================================

simulation_signal_names_nb63 = {
    "simulate",
    "simulate_battle",
    "run_battle",
    "run_game",
    "play_game",
    "battle",
    "step",
    "reset",
    "run_match",
    "run_matches",
    "run_tournament",
}


simulation_entry_signals_nb63 = []


for relative_path, symbols in runtime_symbol_inventory_nb63.items():

    for function_name in symbols["functions"]:

        lowered = function_name.lower()

        if (
            lowered in simulation_signal_names_nb63
            or
            "simulat" in lowered
            or
            "battle" in lowered
            or
            "match" in lowered
            or
            "tournament" in lowered
        ):

            simulation_entry_signals_nb63.append(
                {
                    "file": relative_path,
                    "type": "function",
                    "name": function_name,
                }
            )


    for class_name in symbols["classes"]:

        lowered = class_name.lower()

        if (
            "simulat" in lowered
            or
            "battle" in lowered
            or
            "game" in lowered
            or
            "environment" in lowered
        ):

            simulation_entry_signals_nb63.append(
                {
                    "file": relative_path,
                    "type": "class",
                    "name": class_name,
                }
            )


print()
print("=" * 92)
print("4. SIMULATION ENTRY-POINT SIGNALS")
print("=" * 92)


if simulation_entry_signals_nb63:

    for signal in simulation_entry_signals_nb63:

        print(
            f"[FOUND] {signal['type']}: "
            f"{signal['name']} "
            f"({signal['file']})"
        )

else:

    print(
        "[INFO] No obvious simulation entry point "
        "identified by static naming."
    )


# =============================================================================
# 5. EXISTING TOURNAMENT / SIMULATION ARTIFACT DISCOVERY
# =============================================================================

evidence_extensions_nb63 = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".pkl",
    ".joblib",
}


evidence_keywords_nb63 = (
    "tournament",
    "simulation",
    "battle",
    "benchmark",
    "match",
)


existing_evidence_nb63 = []


search_roots_nb63 = [
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "models",
]


for root in search_roots_nb63:

    if not root.exists():
        continue

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in evidence_extensions_nb63:
            continue

        name_lower = path.name.lower()

        if not any(
            keyword in name_lower
            for keyword in evidence_keywords_nb63
        ):
            continue

        existing_evidence_nb63.append(
            path.resolve()
        )


existing_evidence_nb63 = sorted(
    set(existing_evidence_nb63),
    key=lambda p: str(p).lower(),
)


print()
print("=" * 92)
print("5. EXISTING SIMULATION / TOURNAMENT EVIDENCE")
print("=" * 92)

print(
    "Evidence files discovered:",
    len(existing_evidence_nb63),
)


for path in existing_evidence_nb63[:100]:

    try:
        relative = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative = path

    print(
        " ",
        relative,
        "|",
        path.stat().st_size,
        "bytes",
    )


if len(existing_evidence_nb63) > 100:

    print(
        f"... {len(existing_evidence_nb63) - 100} "
        "additional files not printed."
    )


# =============================================================================
# 6. PROVE CERTIFIED ASSETS REMAIN UNCHANGED
# =============================================================================

main_hash_after_cell2_nb63 = sha256_file_nb63(
    FINAL_MAIN
)

model_hash_after_cell2_nb63 = sha256_file_nb63(
    CERTIFIED_MODEL
)


assert (
    main_hash_after_cell2_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    model_hash_after_cell2_nb63
    ==
    EXPECTED_MODEL_SHA256
)


print()
print("=" * 92)
print("6. CERTIFIED ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py remains unchanged."
)

print(
    "[OK] Certified model remains unchanged."
)

print(
    "[OK] No training or simulation was executed."
)


# =============================================================================
# 7. CELL STATUS
# =============================================================================

CELL2_NB63_STATUS = {

    "cell1_pass":
        True,

    "discovery_read_only":
        True,

    "simulation_candidate_count":
        len(simulation_candidates_nb63),

    "known_runtime_files_expected":
        len(runtime_inventory_nb63),

    "known_runtime_files_present":
        existing_runtime_count_nb63,

    "simulation_entry_signal_count":
        len(simulation_entry_signals_nb63),

    "existing_evidence_file_count":
        len(existing_evidence_nb63),

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 2 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL2_NB63_STATUS,
        indent=2,
    )
)

print()
print(
    "Runtime discovery complete."
)

print(
    "NEXT STEP: Cell 3 — determine the authoritative simulation "
    "engine and its callable interface."
)


# In[3]:


# =============================================================================
# NOTEBOOK 63 — CELL 3
# AUTHORITATIVE SIMULATION ENGINE + CALLABLE INTERFACE AUDIT
# READ-ONLY / STATIC ANALYSIS ONLY
# =============================================================================

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 3")
print("AUTHORITATIVE SIMULATION ENGINE + CALLABLE INTERFACE AUDIT")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITE
# =============================================================================

assert (
    CELL2_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 2 must PASS before Cell 3."

assert (
    CELL2_NB63_STATUS["simulation_executed"]
    is False
)

assert (
    CELL2_NB63_STATUS["training_executed"]
    is False
)

print()
print("[OK] Cell 2 PASS confirmed.")
print("[OK] No simulation has been executed.")
print("[OK] No training has been executed.")
print("[OK] Cell 3 remains read-only.")


# =============================================================================
# 1. AUTHORITATIVE DEPLOYMENT SIMULATOR CANDIDATE
# =============================================================================

battle_simulation_path_nb63 = (
    SUBMISSION_DIR
    / "final_agent"
    / "src"
    / "battle_simulation.py"
)

assert battle_simulation_path_nb63.is_file(), (
    f"Missing deployment battle simulator: "
    f"{battle_simulation_path_nb63}"
)


battle_simulation_source_nb63 = (
    battle_simulation_path_nb63.read_text(
        encoding="utf-8"
    )
)

battle_simulation_tree_nb63 = ast.parse(
    battle_simulation_source_nb63,
    filename=str(
        battle_simulation_path_nb63
    ),
)


print()
print("=" * 92)
print("1. DEPLOYMENT SIMULATOR CANDIDATE")
print("=" * 92)

print(
    "Candidate:",
    battle_simulation_path_nb63,
)

print(
    "Bytes:",
    battle_simulation_path_nb63.stat().st_size,
)

print()
print(
    "[OK] Deployment battle_simulation.py located."
)


# =============================================================================
# 2. FUNCTION SIGNATURE EXTRACTION
# =============================================================================

def ast_argument_name_nb63(arg):
    return arg.arg


def ast_default_repr_nb63(node):

    if node is None:
        return None

    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


function_contracts_nb63 = {}


for node in battle_simulation_tree_nb63.body:

    if not isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        continue

    positional_args = [
        ast_argument_name_nb63(arg)
        for arg in (
            node.args.posonlyargs
            +
            node.args.args
        )
    ]

    keyword_only_args = [
        ast_argument_name_nb63(arg)
        for arg in node.args.kwonlyargs
    ]

    defaults = [
        ast_default_repr_nb63(default)
        for default in node.args.defaults
    ]

    kw_defaults = [
        ast_default_repr_nb63(default)
        for default in node.args.kw_defaults
    ]

    return_annotation = (
        ast.unparse(node.returns)
        if node.returns is not None
        else None
    )

    function_contracts_nb63[
        node.name
    ] = {

        "positional_args":
            positional_args,

        "keyword_only_args":
            keyword_only_args,

        "defaults":
            defaults,

        "keyword_defaults":
            kw_defaults,

        "vararg":
            (
                node.args.vararg.arg
                if node.args.vararg
                else None
            ),

        "kwarg":
            (
                node.args.kwarg.arg
                if node.args.kwarg
                else None
            ),

        "return_annotation":
            return_annotation,

        "line":
            node.lineno,
    }


print()
print("=" * 92)
print("2. FUNCTION CONTRACTS")
print("=" * 92)


for name, contract in function_contracts_nb63.items():

    print()
    print(name)

    print(
        "  positional args :",
        contract["positional_args"],
    )

    print(
        "  keyword-only    :",
        contract["keyword_only_args"],
    )

    print(
        "  defaults        :",
        contract["defaults"],
    )

    print(
        "  return          :",
        contract["return_annotation"],
    )

    print(
        "  source line     :",
        contract["line"],
    )


assert (
    "simulate_ai_battle"
    in
    function_contracts_nb63
), (
    "simulate_ai_battle() not found in deployment "
    "battle_simulation.py."
)


# =============================================================================
# 3. simulate_ai_battle STATIC BODY AUDIT
# =============================================================================

simulate_node_nb63 = next(
    node
    for node in battle_simulation_tree_nb63.body
    if (
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and
        node.name
        ==
        "simulate_ai_battle"
    )
)


simulate_calls_nb63 = []
simulate_names_nb63 = set()
simulate_attributes_nb63 = set()


for node in ast.walk(
    simulate_node_nb63
):

    if isinstance(
        node,
        ast.Call,
    ):

        try:
            call_name = ast.unparse(
                node.func
            )
        except Exception:
            call_name = ast.dump(
                node.func
            )

        simulate_calls_nb63.append(
            call_name
        )

    elif isinstance(
        node,
        ast.Name,
    ):

        simulate_names_nb63.add(
            node.id
        )

    elif isinstance(
        node,
        ast.Attribute,
    ):

        try:
            attribute_name = ast.unparse(
                node
            )
        except Exception:
            attribute_name = node.attr

        simulate_attributes_nb63.add(
            attribute_name
        )


simulate_calls_nb63 = sorted(
    set(
        simulate_calls_nb63
    )
)

simulate_names_nb63 = sorted(
    simulate_names_nb63
)

simulate_attributes_nb63 = sorted(
    simulate_attributes_nb63
)


print()
print("=" * 92)
print("3. simulate_ai_battle() STATIC BODY AUDIT")
print("=" * 92)

print()
print("Calls:")

for name in simulate_calls_nb63:
    print(" ", name)

print()
print("Referenced attributes:")

for name in simulate_attributes_nb63:
    print(" ", name)


# =============================================================================
# 4. MODULE IMPORT DEPENDENCY AUDIT
# =============================================================================

module_imports_nb63 = []


for node in battle_simulation_tree_nb63.body:

    if isinstance(
        node,
        ast.Import,
    ):

        for alias in node.names:

            module_imports_nb63.append(
                {
                    "type": "import",
                    "module": alias.name,
                    "name": None,
                    "alias": alias.asname,
                    "line": node.lineno,
                }
            )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        for alias in node.names:

            module_imports_nb63.append(
                {
                    "type": "from",
                    "module": node.module,
                    "name": alias.name,
                    "alias": alias.asname,
                    "level": node.level,
                    "line": node.lineno,
                }
            )


print()
print("=" * 92)
print("4. MODULE IMPORT DEPENDENCIES")
print("=" * 92)


for record in module_imports_nb63:

    if record["type"] == "import":

        print(
            f"line {record['line']}: "
            f"import {record['module']}"
        )

    else:

        prefix = "." * record.get(
            "level",
            0,
        )

        print(
            f"line {record['line']}: "
            f"from {prefix}"
            f"{record['module'] or ''} "
            f"import {record['name']}"
        )


# =============================================================================
# 5. RELATED DEPLOYMENT MODULE AVAILABILITY
# =============================================================================

deployment_src_nb63 = (
    SUBMISSION_DIR
    / "final_agent"
    / "src"
)


related_runtime_files_nb63 = {
    "battle_simulation":
        deployment_src_nb63
        / "battle_simulation.py",

    "battle_state":
        deployment_src_nb63
        / "battle_state.py",

    "battle_agent":
        deployment_src_nb63
        / "battle_agent.py",

    "simulator":
        deployment_src_nb63
        / "simulator.py",

    "legal_moves":
        deployment_src_nb63
        / "legal_moves.py",

    "agent_decision":
        deployment_src_nb63
        / "agent_decision.py",

    "final_ppo_agent":
        deployment_src_nb63
        / "agents"
        / "final_ppo_agent.py",

    "ppo_policy_engine":
        deployment_src_nb63
        / "ppo"
        / "ppo_policy_engine.py",
}


related_runtime_presence_nb63 = {}


print()
print("=" * 92)
print("5. RELATED DEPLOYMENT MODULE AVAILABILITY")
print("=" * 92)


for name, path in related_runtime_files_nb63.items():

    exists = path.is_file()

    related_runtime_presence_nb63[
        name
    ] = exists

    print(
        "[OK]" if exists else "[MISSING]",
        name,
        "->",
        path.relative_to(
            PROJECT_ROOT
        ),
    )


assert all(
    related_runtime_presence_nb63.values()
), (
    "One or more required deployment runtime files "
    "are missing."
)


# =============================================================================
# 6. BATTLE STATE DATA CONTRACT
# =============================================================================

battle_state_path_nb63 = (
    deployment_src_nb63
    / "battle_state.py"
)

battle_state_source_nb63 = (
    battle_state_path_nb63.read_text(
        encoding="utf-8"
    )
)

battle_state_tree_nb63 = ast.parse(
    battle_state_source_nb63,
    filename=str(
        battle_state_path_nb63
    ),
)


battle_state_classes_nb63 = {}


for node in battle_state_tree_nb63.body:

    if not isinstance(
        node,
        ast.ClassDef,
    ):
        continue

    fields = []

    for child in node.body:

        if isinstance(
            child,
            ast.AnnAssign,
        ):

            if isinstance(
                child.target,
                ast.Name,
            ):

                fields.append(
                    {
                        "name":
                            child.target.id,

                        "annotation":
                            (
                                ast.unparse(
                                    child.annotation
                                )
                                if child.annotation
                                else None
                            ),

                        "default":
                            (
                                ast.unparse(
                                    child.value
                                )
                                if child.value
                                else None
                            ),
                    }
                )

    battle_state_classes_nb63[
        node.name
    ] = fields


print()
print("=" * 92)
print("6. BATTLE-STATE DATA CONTRACT")
print("=" * 92)


for class_name, fields in battle_state_classes_nb63.items():

    print()
    print(class_name)

    if not fields:
        print("  [no annotated class fields found]")

    for field in fields:

        print(
            " ",
            field["name"],
            ":",
            field["annotation"],
            "=",
            field["default"],
        )


assert (
    "BattleState"
    in
    battle_state_classes_nb63
), (
    "BattleState class was not found."
)


# =============================================================================
# 7. AGENT INTERFACE STATIC AUDIT
# =============================================================================

agent_interface_records_nb63 = {}


for agent_file_name in [
    "battle_agent.py",
    "agents/final_ppo_agent.py",
]:

    path = (
        deployment_src_nb63
        / agent_file_name
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    class_records = {}


    for node in tree.body:

        if not isinstance(
            node,
            ast.ClassDef,
        ):
            continue

        methods = {}


        for child in node.body:

            if not isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            args = [
                arg.arg
                for arg in (
                    child.args.posonlyargs
                    +
                    child.args.args
                )
            ]

            methods[
                child.name
            ] = args


        class_records[
            node.name
        ] = methods


    agent_interface_records_nb63[
        agent_file_name
    ] = class_records


print()
print("=" * 92)
print("7. AGENT INTERFACE")
print("=" * 92)


for file_name, classes in agent_interface_records_nb63.items():

    print()
    print(file_name)

    for class_name, methods in classes.items():

        print(
            " ",
            class_name,
        )

        for method_name, args in methods.items():

            print(
                "    ",
                method_name,
                args,
            )


# =============================================================================
# 8. SIMULATION ENGINE CLASSIFICATION
# =============================================================================

simulate_contract_nb63 = (
    function_contracts_nb63[
        "simulate_ai_battle"
    ]
)


simulation_engine_classification_nb63 = {

    "engine_file":
        str(
            battle_simulation_path_nb63.relative_to(
                PROJECT_ROOT
            )
        ),

    "entry_point":
        "simulate_ai_battle",

    "entry_point_line":
        simulate_contract_nb63[
            "line"
        ],

    "entry_point_positional_args":
        simulate_contract_nb63[
            "positional_args"
        ],

    "entry_point_keyword_only_args":
        simulate_contract_nb63[
            "keyword_only_args"
        ],

    "entry_point_defaults":
        simulate_contract_nb63[
            "defaults"
        ],

    "return_annotation":
        simulate_contract_nb63[
            "return_annotation"
        ],

    "dependency_count":
        len(
            module_imports_nb63
        ),

    "runtime_files_complete":
        all(
            related_runtime_presence_nb63.values()
        ),

    "classification":
        "DEPLOYMENT_SIMULATION_ENGINE_CANDIDATE",
}


print()
print("=" * 92)
print("8. SIMULATION ENGINE CLASSIFICATION")
print("=" * 92)

print(
    json.dumps(
        simulation_engine_classification_nb63,
        indent=2,
    )
)


# =============================================================================
# 9. CERTIFIED ASSET INTEGRITY
# =============================================================================

main_hash_after_cell3_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

model_hash_after_cell3_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)


assert (
    main_hash_after_cell3_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    model_hash_after_cell3_nb63
    ==
    EXPECTED_MODEL_SHA256
)


print()
print("=" * 92)
print("9. CERTIFIED ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py remains unchanged."
)

print(
    "[OK] Certified model remains unchanged."
)

print(
    "[OK] No simulation executed."
)

print(
    "[OK] No training executed."
)


# =============================================================================
# 10. CELL STATUS
# =============================================================================

CELL3_NB63_STATUS = {

    "cell2_pass":
        True,

    "analysis_mode":
        "STATIC_READ_ONLY",

    "deployment_simulator_located":
        True,

    "simulate_ai_battle_found":
        True,

    "simulate_ai_battle_args":
        simulate_contract_nb63[
            "positional_args"
        ],

    "simulation_dependency_count":
        len(
            module_imports_nb63
        ),

    "related_runtime_files_complete":
        all(
            related_runtime_presence_nb63.values()
        ),

    "battle_state_contract_found":
        (
            "BattleState"
            in
            battle_state_classes_nb63
        ),

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 3 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL3_NB63_STATUS,
        indent=2,
    )
)

print()

print(
    "Deployment simulator interface audit complete."
)

print(
    "NEXT STEP: Cell 4 — construct a controlled "
    "simulation harness from the proven interface."
)


# In[4]:


# =============================================================================
# NOTEBOOK 63 — CELL 4
# CONTROLLED SIMULATION HARNESS CONTRACT ASSEMBLY
# MOVE SCHEMA + STATE CONSTRUCTION + AGENT RETURN AUDIT
# NO BATTLE EXECUTION
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 4")
print("CONTROLLED SIMULATION HARNESS CONTRACT ASSEMBLY")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL3_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 3 must PASS before Cell 4."

assert (
    CELL3_NB63_STATUS["simulation_executed"]
    is False
)

assert (
    CELL3_NB63_STATUS["training_executed"]
    is False
)


print()
print("[OK] Cell 3 PASS confirmed.")
print("[OK] No battle execution authorized yet.")
print("[OK] Cell 4 is contract analysis only.")


# =============================================================================
# 1. DEPLOYMENT RUNTIME PATHS
# =============================================================================

DEPLOYMENT_SRC_NB63 = (
    SUBMISSION_DIR
    / "final_agent"
    / "src"
)

BATTLE_STATE_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "battle_state.py"
)

BATTLE_AGENT_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "battle_agent.py"
)

LEGAL_MOVES_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "legal_moves.py"
)

SIMULATOR_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "simulator.py"
)

AGENT_DECISION_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "agent_decision.py"
)

FINAL_PPO_AGENT_PATH_NB63 = (
    DEPLOYMENT_SRC_NB63
    / "agents"
    / "final_ppo_agent.py"
)


required_runtime_paths_nb63 = [
    BATTLE_STATE_PATH_NB63,
    BATTLE_AGENT_PATH_NB63,
    LEGAL_MOVES_PATH_NB63,
    SIMULATOR_PATH_NB63,
    AGENT_DECISION_PATH_NB63,
    FINAL_PPO_AGENT_PATH_NB63,
]


for path in required_runtime_paths_nb63:

    assert path.is_file(), (
        f"Required deployment runtime file missing: {path}"
    )


print()
print("=" * 92)
print("1. RUNTIME CONTRACT FILES")
print("=" * 92)


for path in required_runtime_paths_nb63:

    print(
        "[OK]",
        path.relative_to(
            PROJECT_ROOT
        ),
        "|",
        path.stat().st_size,
        "bytes",
    )


# =============================================================================
# 2. GENERIC AST HELPERS
# =============================================================================

def source_tree_nb63(
    path: Path,
):

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    return source, tree


def function_signature_from_ast_nb63(
    node,
):

    positional = (
        node.args.posonlyargs
        +
        node.args.args
    )

    positional_names = [
        argument.arg
        for argument in positional
    ]


    default_count = len(
        node.args.defaults
    )


    required_count = (
        len(positional_names)
        -
        default_count
    )


    positional_records = []


    for index, name in enumerate(
        positional_names
    ):

        record = {
            "name":
                name,

            "required":
                index
                <
                required_count,

            "default":
                None,
        }


        if index >= required_count:

            default_node = (
                node.args.defaults[
                    index
                    -
                    required_count
                ]
            )

            try:

                record["default"] = (
                    ast.unparse(
                        default_node
                    )
                )

            except Exception:

                record["default"] = (
                    ast.dump(
                        default_node
                    )
                )


        positional_records.append(
            record
        )


    return {
        "positional":
            positional_records,

        "keyword_only": [
            argument.arg
            for argument
            in node.args.kwonlyargs
        ],

        "vararg":
            (
                node.args.vararg.arg
                if node.args.vararg
                else None
            ),

        "kwarg":
            (
                node.args.kwarg.arg
                if node.args.kwarg
                else None
            ),

        "return":
            (
                ast.unparse(
                    node.returns
                )
                if node.returns
                is not None
                else None
            ),
    }


def callable_name_nb63(
    node,
):

    try:

        return ast.unparse(
            node
        )

    except Exception:

        return ast.dump(
            node
        )


# =============================================================================
# 3. LEGAL-MOVE FUNCTION CONTRACTS
# =============================================================================

legal_source_nb63, legal_tree_nb63 = (
    source_tree_nb63(
        LEGAL_MOVES_PATH_NB63
    )
)


legal_function_contracts_nb63 = {}


for node in legal_tree_nb63.body:

    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):

        legal_function_contracts_nb63[
            node.name
        ] = {
            "line":
                node.lineno,

            "signature":
                function_signature_from_ast_nb63(
                    node
                ),
        }


print()
print("=" * 92)
print("2. LEGAL-MOVE FUNCTION CONTRACTS")
print("=" * 92)


for name, record in (
    legal_function_contracts_nb63.items()
):

    print()
    print(
        f"{name} — line {record['line']}"
    )

    print(
        json.dumps(
            record["signature"],
            indent=2,
        )
    )


assert (
    "get_legal_moves"
    in
    legal_function_contracts_nb63
), (
    "get_legal_moves() not found."
)


# =============================================================================
# 4. LEGAL-MOVE RETURN-SHAPE AUDIT
# =============================================================================

get_legal_moves_node_nb63 = next(
    node
    for node in legal_tree_nb63.body
    if (
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and
        node.name
        ==
        "get_legal_moves"
    )
)


legal_return_expressions_nb63 = []


for node in ast.walk(
    get_legal_moves_node_nb63
):

    if isinstance(
        node,
        ast.Return,
    ):

        if node.value is None:

            expression = None

        else:

            try:

                expression = ast.unparse(
                    node.value
                )

            except Exception:

                expression = ast.dump(
                    node.value
                )


        legal_return_expressions_nb63.append(
            expression
        )


legal_dict_literals_nb63 = []


for node in ast.walk(
    get_legal_moves_node_nb63
):

    if not isinstance(
        node,
        ast.Dict,
    ):
        continue


    keys = []


    for key in node.keys:

        if key is None:

            keys.append(
                "**"
            )

        elif isinstance(
            key,
            ast.Constant,
        ):

            keys.append(
                key.value
            )

        else:

            try:

                keys.append(
                    ast.unparse(
                        key
                    )
                )

            except Exception:

                keys.append(
                    ast.dump(
                        key
                    )
                )


    legal_dict_literals_nb63.append(
        keys
    )


print()
print("=" * 92)
print("3. LEGAL-MOVE RETURN SHAPE")
print("=" * 92)

print()
print(
    "Return expressions:"
)


for expression in (
    legal_return_expressions_nb63
):

    print(
        " ",
        expression,
    )


print()
print(
    "Dictionary key patterns discovered:"
)


for keys in legal_dict_literals_nb63:

    print(
        " ",
        keys,
    )


# =============================================================================
# 5. apply_move() CONTRACT
# =============================================================================

simulator_source_nb63, simulator_tree_nb63 = (
    source_tree_nb63(
        SIMULATOR_PATH_NB63
    )
)


apply_move_node_nb63 = next(
    node
    for node in simulator_tree_nb63.body
    if (
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        and
        node.name
        ==
        "apply_move"
    )
)


apply_move_signature_nb63 = (
    function_signature_from_ast_nb63(
        apply_move_node_nb63
    )
)


apply_move_calls_nb63 = []
apply_move_move_accesses_nb63 = []
apply_move_dict_keys_nb63 = []


for node in ast.walk(
    apply_move_node_nb63
):

    if isinstance(
        node,
        ast.Call,
    ):

        apply_move_calls_nb63.append(
            callable_name_nb63(
                node.func
            )
        )


    elif isinstance(
        node,
        ast.Subscript,
    ):

        try:

            subscript_text = (
                ast.unparse(
                    node
                )
            )

        except Exception:

            subscript_text = (
                ast.dump(
                    node
                )
            )


        if "move" in subscript_text:

            apply_move_move_accesses_nb63.append(
                subscript_text
            )


    elif isinstance(
        node,
        ast.Constant,
    ):

        if isinstance(
            node.value,
            str,
        ):

            apply_move_dict_keys_nb63.append(
                node.value
            )


apply_move_calls_nb63 = sorted(
    set(
        apply_move_calls_nb63
    )
)

apply_move_move_accesses_nb63 = sorted(
    set(
        apply_move_move_accesses_nb63
    )
)

apply_move_dict_keys_nb63 = sorted(
    set(
        apply_move_dict_keys_nb63
    )
)


print()
print("=" * 92)
print("4. apply_move() CONTRACT")
print("=" * 92)

print(
    json.dumps(
        apply_move_signature_nb63,
        indent=2,
    )
)

print()
print("Function calls:")

for value in apply_move_calls_nb63:

    print(
        " ",
        value,
    )


print()
print("Move-related expressions:")

for value in apply_move_move_accesses_nb63:

    print(
        " ",
        value,
    )


print()
print("String literals / possible move keys:")

for value in apply_move_dict_keys_nb63:

    print(
        " ",
        repr(
            value
        ),
    )


# =============================================================================
# 6. AgentDecision DATA CONTRACT
# =============================================================================

decision_source_nb63, decision_tree_nb63 = (
    source_tree_nb63(
        AGENT_DECISION_PATH_NB63
    )
)


decision_classes_nb63 = {}


for node in decision_tree_nb63.body:

    if not isinstance(
        node,
        ast.ClassDef,
    ):
        continue


    fields = []


    for child in node.body:

        if isinstance(
            child,
            ast.AnnAssign,
        ):

            if not isinstance(
                child.target,
                ast.Name,
            ):
                continue


            fields.append(
                {
                    "name":
                        child.target.id,

                    "annotation":
                        (
                            ast.unparse(
                                child.annotation
                            )
                            if child.annotation
                            is not None
                            else None
                        ),

                    "default":
                        (
                            ast.unparse(
                                child.value
                            )
                            if child.value
                            is not None
                            else None
                        ),
                }
            )


    decision_classes_nb63[
        node.name
    ] = fields


print()
print("=" * 92)
print("5. AGENT DECISION CONTRACT")
print("=" * 92)


for class_name, fields in (
    decision_classes_nb63.items()
):

    print()
    print(
        class_name
    )

    for field in fields:

        print(
            " ",
            field["name"],
            ":",
            field["annotation"],
            "=",
            field["default"],
        )


assert (
    "AgentDecision"
    in
    decision_classes_nb63
), (
    "AgentDecision class not found."
)


# =============================================================================
# 7. choose_move() RETURN AUDIT
# =============================================================================

agent_files_to_audit_nb63 = {
    "PokemonBattleAgent":
        BATTLE_AGENT_PATH_NB63,

    "FinalPPOBattleAgent":
        FINAL_PPO_AGENT_PATH_NB63,
}


choose_move_return_audit_nb63 = {}


for label, path in (
    agent_files_to_audit_nb63.items()
):

    source, tree = (
        source_tree_nb63(
            path
        )
    )


    choose_nodes = []


    for node in ast.walk(
        tree
    ):

        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and
            node.name
            ==
            "choose_move"
        ):

            choose_nodes.append(
                node
            )


    return_values = []
    calls = []


    for choose_node in choose_nodes:

        for node in ast.walk(
            choose_node
        ):

            if isinstance(
                node,
                ast.Return,
            ):

                if node.value is None:

                    value = None

                else:

                    try:

                        value = (
                            ast.unparse(
                                node.value
                            )
                        )

                    except Exception:

                        value = (
                            ast.dump(
                                node.value
                            )
                        )


                return_values.append(
                    value
                )


            elif isinstance(
                node,
                ast.Call,
            ):

                calls.append(
                    callable_name_nb63(
                        node.func
                    )
                )


    choose_move_return_audit_nb63[
        label
    ] = {
        "file":
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "definition_count":
            len(
                choose_nodes
            ),

        "returns":
            return_values,

        "calls":
            sorted(
                set(
                    calls
                )
            ),
    }


print()
print("=" * 92)
print("6. choose_move() RETURN AUDIT")
print("=" * 92)


for label, audit in (
    choose_move_return_audit_nb63.items()
):

    print()
    print(label)

    print(
        "  file:",
        audit["file"],
    )

    print(
        "  definitions:",
        audit["definition_count"],
    )

    print(
        "  returns:"
    )

    for value in audit["returns"]:

        print(
            "   ",
            value,
        )

    print(
        "  calls:"
    )

    for value in audit["calls"]:

        print(
            "   ",
            value,
        )


# =============================================================================
# 8. STATE CONSTRUCTION CONTRACT
# =============================================================================

battle_source_nb63, battle_tree_nb63 = (
    source_tree_nb63(
        BATTLE_STATE_PATH_NB63
    )
)


state_construction_contract_nb63 = {}


for node in battle_tree_nb63.body:

    if not isinstance(
        node,
        ast.ClassDef,
    ):
        continue


    annotated_fields = []


    for child in node.body:

        if not isinstance(
            child,
            ast.AnnAssign,
        ):
            continue

        if not isinstance(
            child.target,
            ast.Name,
        ):
            continue


        annotated_fields.append(
            {
                "name":
                    child.target.id,

                "annotation":
                    (
                        ast.unparse(
                            child.annotation
                        )
                        if child.annotation
                        else None
                    ),

                "default":
                    (
                        ast.unparse(
                            child.value
                        )
                        if child.value
                        else None
                    ),
            }
        )


    state_construction_contract_nb63[
        node.name
    ] = (
        annotated_fields
    )


print()
print("=" * 92)
print("7. STATE CONSTRUCTION CONTRACT")
print("=" * 92)


for class_name, fields in (
    state_construction_contract_nb63.items()
):

    print()
    print(
        class_name
    )

    for field in fields:

        print(
            " ",
            field["name"],
            ":",
            field["annotation"],
            "=",
            field["default"],
        )


# =============================================================================
# 9. SEARCH FOR PRIOR WORKING simulate_ai_battle() INVOCATIONS
# =============================================================================

prior_invocation_records_nb63 = []


search_roots_for_calls_nb63 = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "src",
]


for root in search_roots_for_calls_nb63:

    if not root.exists():

        continue


    for path in root.rglob(
        "*.py"
    ):

        path_text = str(
            path
        ).lower()


        if (
            "__pycache__"
            in
            path_text
        ):

            continue


        try:

            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )

        except Exception:

            continue


        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):

                continue


            function_text = (
                callable_name_nb63(
                    node.func
                )
            )


            if not (
                function_text.endswith(
                    "simulate_ai_battle"
                )
                or
                function_text
                ==
                "simulate_ai_battle"
            ):

                continue


            try:

                call_text = (
                    ast.unparse(
                        node
                    )
                )

            except Exception:

                call_text = (
                    ast.dump(
                        node
                    )
                )


            prior_invocation_records_nb63.append(
                {
                    "file":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "line":
                        node.lineno,

                    "call":
                        call_text,
                }
            )


print()
print("=" * 92)
print("8. PRIOR WORKING SIMULATOR INVOCATIONS")
print("=" * 92)


print(
    "simulate_ai_battle() call sites found:",
    len(
        prior_invocation_records_nb63
    ),
)


for record in (
    prior_invocation_records_nb63[
        :50
    ]
):

    print()
    print(
        record["file"],
        "line",
        record["line"],
    )

    print(
        " ",
        record["call"],
    )


# =============================================================================
# 10. HARNESS READINESS CLASSIFICATION
# =============================================================================

agent_decision_fields_nb63 = [
    field["name"]
    for field in (
        decision_classes_nb63[
            "AgentDecision"
        ]
    )
]


required_decision_fields_nb63 = {
    "move",
}


decision_move_contract_present_nb63 = (
    required_decision_fields_nb63
    .issubset(
        set(
            agent_decision_fields_nb63
        )
    )
)


harness_readiness_nb63 = {

    "simulate_ai_battle_contract_known":
        True,

    "get_legal_moves_contract_known":
        True,

    "apply_move_contract_known":
        True,

    "agent_decision_contract_known":
        True,

    "decision_move_field_present":
        decision_move_contract_present_nb63,

    "battle_state_contract_known":
        True,

    "prior_simulator_call_sites_found":
        len(
            prior_invocation_records_nb63
        ),

    "safe_for_smoke_harness_construction":
        decision_move_contract_present_nb63,

}


print()
print("=" * 92)
print("9. HARNESS READINESS CLASSIFICATION")
print("=" * 92)


print(
    json.dumps(
        harness_readiness_nb63,
        indent=2,
    )
)


assert (
    decision_move_contract_present_nb63
    is True
), (
    "AgentDecision.move contract not established."
)


# =============================================================================
# 11. CERTIFIED-ASSET IMMUTABILITY
# =============================================================================

main_hash_after_cell4_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)


model_hash_after_cell4_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)


assert (
    main_hash_after_cell4_nb63
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    model_hash_after_cell4_nb63
    ==
    EXPECTED_MODEL_SHA256
)


print()
print("=" * 92)
print("10. CERTIFIED ASSET INTEGRITY")
print("=" * 92)


print(
    "[OK] main.py remains unchanged."
)

print(
    "[OK] Certified model remains unchanged."
)

print(
    "[OK] No simulation executed."
)

print(
    "[OK] No training executed."
)


# =============================================================================
# 12. CELL STATUS
# =============================================================================

CELL4_NB63_STATUS = {

    "cell3_pass":
        True,

    "analysis_mode":
        "STATIC_HARNESS_CONTRACT_AUDIT",

    "simulate_ai_battle_contract_known":
        True,

    "legal_move_contract_known":
        True,

    "apply_move_contract_known":
        True,

    "agent_decision_contract_known":
        True,

    "decision_move_field_present":
        decision_move_contract_present_nb63,

    "state_construction_contract_known":
        True,

    "prior_simulator_call_site_count":
        len(
            prior_invocation_records_nb63
        ),

    "smoke_harness_ready":
        True,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "simulation_executed":
        False,

    "training_executed":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 4 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL4_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Controlled harness contract assembly complete."
)

print(
    "No battle has been executed."
)

print(
    "NEXT STEP: Cell 5 — isolated one-battle smoke test "
    "using the proven move/state/agent contracts."
)


# In[5]:


# =============================================================================
# NOTEBOOK 63 — CELL 5
# ISOLATED ONE-BATTLE DEPLOYMENT SIMULATOR SMOKE TEST
# TEMPORARY RUNTIME COPY — NO CERTIFIED SUBMISSION MUTATION
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 5")
print("ISOLATED ONE-BATTLE DEPLOYMENT SIMULATOR SMOKE TEST")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL4_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 4 must PASS before Cell 5."

assert (
    CELL4_NB63_STATUS["smoke_harness_ready"]
    is True
)

assert (
    CELL4_NB63_STATUS["simulation_executed"]
    is False
)

assert (
    CELL4_NB63_STATUS["training_executed"]
    is False
)


print()
print("[OK] Cell 4 PASS confirmed.")
print("[OK] Harness contract established.")
print("[OK] First isolated simulation authorized.")
print("[OK] Certified submission will remain read-only.")


# =============================================================================
# 1. HELPERS
# =============================================================================

def sha256_file_cell5_nb63(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def source_snapshot_cell5_nb63(
    root: Path,
) -> dict:

    snapshot = {}

    for path in sorted(
        root.rglob("*")
    ):

        if not path.is_file():
            continue

        relative = (
            path.relative_to(root)
            .as_posix()
        )

        # Python cache files are not authoritative source.
        if "__pycache__" in path.parts:
            continue

        if path.suffix.lower() in {
            ".pyc",
            ".pyo",
        }:
            continue

        snapshot[relative] = {
            "size":
                path.stat().st_size,

            "sha256":
                sha256_file_cell5_nb63(
                    path
                ),
        }

    return snapshot


# =============================================================================
# 2. CERTIFIED ASSET PRE-SIMULATION FREEZE
# =============================================================================

print()
print("=" * 92)
print("1. CERTIFIED ASSET PRE-SIMULATION FREEZE")
print("=" * 92)


main_hash_before_cell5_nb63 = (
    sha256_file_cell5_nb63(
        FINAL_MAIN
    )
)

model_hash_before_cell5_nb63 = (
    sha256_file_cell5_nb63(
        CERTIFIED_MODEL
    )
)

model_size_before_cell5_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


assert (
    main_hash_before_cell5_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    model_hash_before_cell5_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    model_size_before_cell5_nb63
    ==
    EXPECTED_MODEL_SIZE
)


print(
    "main.py SHA256:",
    main_hash_before_cell5_nb63,
)

print(
    "Model SHA256  :",
    model_hash_before_cell5_nb63,
)

print(
    "Model size    :",
    model_size_before_cell5_nb63,
)


print()
print("[OK] Certified assets frozen before smoke test.")


# =============================================================================
# 3. REAL DEPLOYMENT RUNTIME SOURCE SNAPSHOT
# =============================================================================

REAL_DEPLOYMENT_SRC_CELL5_NB63 = (
    SUBMISSION_DIR
    / "final_agent"
    / "src"
)


assert REAL_DEPLOYMENT_SRC_CELL5_NB63.is_dir()


runtime_snapshot_before_cell5_nb63 = (
    source_snapshot_cell5_nb63(
        REAL_DEPLOYMENT_SRC_CELL5_NB63
    )
)


print()
print("=" * 92)
print("2. REAL DEPLOYMENT RUNTIME SNAPSHOT")
print("=" * 92)

print(
    "Runtime source/data files:",
    len(
        runtime_snapshot_before_cell5_nb63
    ),
)

print()
print(
    "[OK] Real deployment runtime snapshot captured."
)


# =============================================================================
# 4. CREATE TEMPORARY RUNTIME PACKAGE
# =============================================================================

print()
print("=" * 92)
print("3. TEMPORARY RUNTIME ASSEMBLY")
print("=" * 92)


temp_root_cell5_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell5_"
    )
).resolve()


runtime_package_cell5_nb63 = (
    temp_root_cell5_nb63
    /
    "runtime_pkg"
)


shutil.copytree(
    REAL_DEPLOYMENT_SRC_CELL5_NB63,
    runtime_package_cell5_nb63,
)


# -------------------------------------------------------------------------
# Guarantee package marker exists ONLY in temporary copy.
# This is required so relative imports such as:
#
#     from .battle_agent import ...
#
# resolve correctly.
# -------------------------------------------------------------------------

runtime_init_cell5_nb63 = (
    runtime_package_cell5_nb63
    /
    "__init__.py"
)


if not runtime_init_cell5_nb63.exists():

    runtime_init_cell5_nb63.write_text(
        "# Temporary NB63 simulation package.\n",
        encoding="utf-8",
    )


print(
    "Temporary root:",
    temp_root_cell5_nb63,
)

print(
    "Temporary runtime:",
    runtime_package_cell5_nb63,
)


print()
print(
    "[OK] Deployment runtime copied to isolated temporary package."
)


# =============================================================================
# 5. BUILD ISOLATED SMOKE-TEST DRIVER
# =============================================================================

driver_path_cell5_nb63 = (
    temp_root_cell5_nb63
    /
    "cell5_smoke_driver.py"
)


driver_source_cell5_nb63 = r'''
from __future__ import annotations

import json
import traceback


RESULT = {
    "imports_pass": False,
    "state_constructed": False,
    "agent_constructed": False,
    "simulation_called": False,
    "simulation_completed": False,
    "winner": None,
    "turn_count": None,
    "record_count": None,
    "final_state_present": False,
    "exception_type": None,
    "exception_message": None,
    "traceback": None,
}


try:

    from runtime_pkg.battle_state import (
        PokemonState,
        PlayerState,
        BattleState,
    )

    from runtime_pkg.agent_decision import (
        AgentDecision,
    )

    from runtime_pkg.battle_simulation import (
        simulate_ai_battle,
        create_battle_transcript,
    )

    RESULT["imports_pass"] = True


    # =========================================================================
    # Deterministic smoke-test agent
    #
    # This is intentionally NOT the PPO policy.
    #
    # The purpose of Cell 5 is to prove that:
    #
    # BattleState
    #     -> agent.choose_move()
    #     -> AgentDecision.move
    #     -> apply_move()
    #     -> BattleSimulationResult
    #
    # works in the frozen deployment simulator.
    # =========================================================================

    class DeterministicSmokeAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            move = {
                "name":
                    "NB63_SMOKE_ATTACK",

                "damage":
                    10.0,

                "energy_cost":
                    0,

                "effect":
                    None,
            }

            return AgentDecision(
                move=move,
                score=10.0,
                search_depth=int(depth),
                nodes=1,
                principal_variation=[
                    move,
                ],
            )


    # =========================================================================
    # Minimal valid battle state
    # =========================================================================

    player_pokemon = PokemonState(
        card={
            "name":
                "NB63_Player_Mon",
        },
        current_hp=30.0,
        attached_energy=0,
        status=None,
        damage=0.0,
        is_active=True,
    )


    opponent_pokemon = PokemonState(
        card={
            "name":
                "NB63_Opponent_Mon",
        },
        current_hp=30.0,
        attached_energy=0,
        status=None,
        damage=0.0,
        is_active=True,
    )


    player = PlayerState(
        active=player_pokemon,
        bench=[],
        prize_cards_remaining=6,
        hand_size=7,
    )


    opponent = PlayerState(
        active=opponent_pokemon,
        bench=[],
        prize_cards_remaining=6,
        hand_size=7,
    )


    initial_state = BattleState(
        player=player,
        opponent=opponent,
        turn_number=1,
        current_player="Player",
    )


    RESULT["state_constructed"] = True


    agent = DeterministicSmokeAgent()

    RESULT["agent_constructed"] = True


    # =========================================================================
    # Execute exactly one controlled simulated battle
    # =========================================================================

    RESULT["simulation_called"] = True


    simulation = simulate_ai_battle(
        initial_state=initial_state,
        agent=agent,
        search_depth=1,
        max_turns=10,
        verbose=False,
    )


    RESULT["simulation_completed"] = True


    # =========================================================================
    # Extract result using defensive attribute access
    # =========================================================================

    RESULT["winner"] = getattr(
        simulation,
        "winner",
        None,
    )


    turn_records = getattr(
        simulation,
        "turn_records",
        None,
    )


    if turn_records is None:

        turn_records = getattr(
            simulation,
            "turns",
            None,
        )


    if turn_records is not None:

        try:

            RESULT["record_count"] = len(
                turn_records
            )

        except Exception:

            RESULT["record_count"] = None


    RESULT["turn_count"] = getattr(
        simulation,
        "turn_count",
        None,
    )


    final_state = getattr(
        simulation,
        "final_state",
        None,
    )


    RESULT["final_state_present"] = (
        final_state is not None
    )


    # -------------------------------------------------------------------------
    # Additional result introspection
    # -------------------------------------------------------------------------

    RESULT["simulation_type"] = (
        type(simulation).__name__
    )


    RESULT["simulation_attributes"] = sorted(
        [
            name
            for name
            in dir(simulation)
            if not name.startswith("_")
        ]
    )


    try:

        transcript = create_battle_transcript(
            simulation
        )

        RESULT["transcript_created"] = True

        RESULT["transcript_preview"] = (
            str(transcript)[:1000]
        )

    except Exception as transcript_exc:

        RESULT["transcript_created"] = False

        RESULT["transcript_error"] = (
            f"{type(transcript_exc).__name__}: "
            f"{transcript_exc}"
        )


except Exception as exc:

    RESULT["exception_type"] = (
        type(exc).__name__
    )

    RESULT["exception_message"] = (
        str(exc)
    )

    RESULT["traceback"] = (
        traceback.format_exc()
    )


print(
    "NB63_CELL5_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell5_nb63.write_text(
    textwrap.dedent(
        driver_source_cell5_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell5_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell5_nb63
    ),
    "exec",
)


print()
print(
    "[OK] Temporary smoke-test driver created."
)

print(
    "[OK] Smoke-test driver compiles."
)


# =============================================================================
# 6. EXECUTE ONE ISOLATED BATTLE
# =============================================================================

print()
print("=" * 92)
print("4. ONE-BATTLE ISOLATED EXECUTION")
print("=" * 92)


environment_cell5_nb63 = os.environ.copy()


# Prevent Python from creating pyc files even in the temporary package.
environment_cell5_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell5_nb63 = subprocess.run(
    [
        sys.executable,
        str(
            driver_path_cell5_nb63
        ),
    ],
    cwd=str(
        temp_root_cell5_nb63
    ),
    env=environment_cell5_nb63,
    capture_output=True,
    text=True,
    timeout=120,
)


print(
    "Return code:",
    process_cell5_nb63.returncode,
)

print()
print("--- STDOUT ---")

print(
    process_cell5_nb63.stdout
)


print()
print("--- STDERR ---")

print(
    process_cell5_nb63.stderr
)


assert (
    process_cell5_nb63.returncode
    ==
    0
), (
    "Smoke-test subprocess failed.\n"
    f"STDOUT:\n{process_cell5_nb63.stdout}\n"
    f"STDERR:\n{process_cell5_nb63.stderr}"
)


# =============================================================================
# 7. PARSE SMOKE RESULT
# =============================================================================

result_prefix_cell5_nb63 = (
    "NB63_CELL5_RESULT_JSON="
)


result_line_cell5_nb63 = None


for line in (
    process_cell5_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell5_nb63
    ):

        result_line_cell5_nb63 = line


assert result_line_cell5_nb63 is not None, (
    "Could not locate Cell 5 smoke-result JSON."
)


smoke_result_cell5_nb63 = json.loads(
    result_line_cell5_nb63[
        len(
            result_prefix_cell5_nb63
        ):
    ]
)


print()
print("=" * 92)
print("5. SMOKE-TEST RESULT")
print("=" * 92)


print(
    json.dumps(
        smoke_result_cell5_nb63,
        indent=2,
    )
)


# =============================================================================
# 8. REQUIRED SMOKE-TEST ASSERTIONS
# =============================================================================

assert (
    smoke_result_cell5_nb63[
        "imports_pass"
    ]
    is True
), (
    "Temporary deployment runtime failed to import."
)


assert (
    smoke_result_cell5_nb63[
        "state_constructed"
    ]
    is True
), (
    "BattleState construction failed."
)


assert (
    smoke_result_cell5_nb63[
        "agent_constructed"
    ]
    is True
), (
    "Smoke agent construction failed."
)


assert (
    smoke_result_cell5_nb63[
        "simulation_called"
    ]
    is True
), (
    "simulate_ai_battle() was not reached."
)


assert (
    smoke_result_cell5_nb63[
        "simulation_completed"
    ]
    is True
), (
    "simulate_ai_battle() did not complete.\n"
    f"{smoke_result_cell5_nb63.get('traceback')}"
)


assert (
    smoke_result_cell5_nb63[
        "exception_type"
    ]
    is None
), (
    "Smoke battle raised an exception.\n"
    f"{smoke_result_cell5_nb63.get('traceback')}"
)


print()
print(
    "[OK] Deployment runtime imported."
)

print(
    "[OK] Minimal BattleState constructed."
)

print(
    "[OK] AgentDecision.move contract executed."
)

print(
    "[OK] apply_move() path executed through simulator."
)

print(
    "[OK] One controlled simulated battle completed."
)


# =============================================================================
# 9. SAVE SMOKE RESULT AS NOTEBOOK 63 EVIDENCE
# =============================================================================

smoke_output_path_cell5_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell5_isolated_simulator_smoke_result.json"
)


smoke_artifact_path_cell5_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell5_isolated_simulator_smoke_result.json"
)


smoke_evidence_cell5_nb63 = {

    "notebook":
        63,

    "cell":
        5,

    "purpose":
        "ISOLATED_DEPLOYMENT_SIMULATOR_SMOKE_TEST",

    "runtime_source":
        str(
            REAL_DEPLOYMENT_SRC_CELL5_NB63
        ),

    "execution_scope":
        "TEMPORARY_COPY_ONLY",

    "policy_type":
        "DETERMINISTIC_SMOKE_AGENT",

    "certified_policy_invoked":
        False,

    "search_depth":
        1,

    "max_turns":
        10,

    "result":
        smoke_result_cell5_nb63,
}


for output_path in [
    smoke_output_path_cell5_nb63,
    smoke_artifact_path_cell5_nb63,
]:

    output_path.write_text(
        json.dumps(
            smoke_evidence_cell5_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("6. EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    smoke_output_path_cell5_nb63,
)

print(
    "[OK]",
    smoke_artifact_path_cell5_nb63,
)


# =============================================================================
# 10. REAL DEPLOYMENT SOURCE IMMUTABILITY
# =============================================================================

runtime_snapshot_after_cell5_nb63 = (
    source_snapshot_cell5_nb63(
        REAL_DEPLOYMENT_SRC_CELL5_NB63
    )
)


assert (
    runtime_snapshot_after_cell5_nb63
    ==
    runtime_snapshot_before_cell5_nb63
), (
    "Real deployment runtime source changed during smoke test."
)


print()
print("=" * 92)
print("7. REAL DEPLOYMENT SOURCE IMMUTABILITY")
print("=" * 92)


print(
    "[OK] Real deployment runtime files remained unchanged."
)


# =============================================================================
# 11. CERTIFIED ASSET POST-SIMULATION INTEGRITY
# =============================================================================

main_hash_after_cell5_nb63 = (
    sha256_file_cell5_nb63(
        FINAL_MAIN
    )
)

model_hash_after_cell5_nb63 = (
    sha256_file_cell5_nb63(
        CERTIFIED_MODEL
    )
)

model_size_after_cell5_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


assert (
    main_hash_after_cell5_nb63
    ==
    main_hash_before_cell5_nb63
)


assert (
    model_hash_after_cell5_nb63
    ==
    model_hash_before_cell5_nb63
)


assert (
    model_size_after_cell5_nb63
    ==
    model_size_before_cell5_nb63
)


print()
print("=" * 92)
print("8. CERTIFIED ASSET POST-SIMULATION INTEGRITY")
print("=" * 92)


print(
    "main.py SHA256:",
    main_hash_after_cell5_nb63,
)

print(
    "Model SHA256  :",
    model_hash_after_cell5_nb63,
)

print(
    "Model size    :",
    model_size_after_cell5_nb63,
)


print()
print(
    "[OK] Authoritative main.py remains unchanged."
)

print(
    "[OK] Certified model remains unchanged."
)

print(
    "[OK] No training or retraining occurred."
)


# =============================================================================
# 12. TEMPORARY RUNTIME CLEANUP
# =============================================================================

print()
print("=" * 92)
print("9. TEMPORARY RUNTIME CLEANUP")
print("=" * 92)


shutil.rmtree(
    temp_root_cell5_nb63,
    ignore_errors=True,
)


temporary_runtime_deleted_cell5_nb63 = (
    not temp_root_cell5_nb63.exists()
)


assert (
    temporary_runtime_deleted_cell5_nb63
    is True
)


print(
    "[OK] Temporary runtime deleted."
)


# =============================================================================
# 13. FINAL CELL STATUS
# =============================================================================

CELL5_NB63_STATUS = {

    "cell4_pass":
        True,

    "simulation_scope":
        "ISOLATED_TEMPORARY_DEPLOYMENT_RUNTIME",

    "simulation_executed":
        True,

    "battle_count":
        1,

    "smoke_agent_type":
        "DETERMINISTIC_SMOKE_AGENT",

    "certified_policy_invoked":
        False,

    "runtime_import_pass":
        smoke_result_cell5_nb63[
            "imports_pass"
        ],

    "battle_state_construction_pass":
        smoke_result_cell5_nb63[
            "state_constructed"
        ],

    "simulation_completed":
        smoke_result_cell5_nb63[
            "simulation_completed"
        ],

    "winner":
        smoke_result_cell5_nb63.get(
            "winner"
        ),

    "record_count":
        smoke_result_cell5_nb63.get(
            "record_count"
        ),

    "exception_type":
        smoke_result_cell5_nb63.get(
            "exception_type"
        ),

    "real_runtime_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "temporary_runtime_deleted":
        temporary_runtime_deleted_cell5_nb63,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 5 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL5_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "First isolated deployment-simulator battle completed successfully."
)

print(
    "Certified submission and certified model remained untouched."
)

print()
print(
    "NEXT STEP: Cell 6 — certified-policy construction and "
    "single-battle policy smoke test."
)


# In[6]:


# =============================================================================
# NOTEBOOK 63 — CELL 6
# CORRECTED DEPLOYMENT POLICY CONSTRUCTION + ONE-BATTLE POLICY SMOKE TEST
# FAITHFUL TEMPORARY final_agent/src PACKAGE LAYOUT
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 6")
print("CORRECTED DEPLOYMENT POLICY CONSTRUCTION + ONE-BATTLE POLICY SMOKE TEST")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL5_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 5 must PASS before Cell 6."

assert (
    CELL5_NB63_STATUS["simulation_completed"]
    is True
)

assert (
    CELL5_NB63_STATUS["real_runtime_modified"]
    is False
)

assert (
    CELL5_NB63_STATUS[
        "certified_model_integrity_preserved"
    ]
    is True
)


print()
print("[OK] Cell 5 PASS confirmed.")
print("[OK] Simulator execution path already proven.")
print("[OK] Corrected Cell 6 preserves the real src namespace.")
print("[OK] Certified submission remains frozen.")
print("[OK] No training or retraining is authorized.")


# =============================================================================
# 1. HELPERS
# =============================================================================

def sha256_file_cell6_nb63(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def snapshot_noncache_cell6_nb63(
    root: Path,
) -> dict:

    snapshot = {}

    for path in sorted(
        root.rglob("*")
    ):

        if not path.is_file():
            continue

        if "__pycache__" in path.parts:
            continue

        if path.suffix.lower() in {
            ".pyc",
            ".pyo",
        }:
            continue

        relative = (
            path.relative_to(root)
            .as_posix()
        )

        snapshot[relative] = {
            "size":
                path.stat().st_size,

            "sha256":
                sha256_file_cell6_nb63(
                    path
                ),
        }

    return snapshot


# =============================================================================
# 2. CERTIFIED ASSET PRE-TEST FREEZE
# =============================================================================

print()
print("=" * 92)
print("1. CERTIFIED ASSET PRE-TEST FREEZE")
print("=" * 92)


main_hash_before_cell6_nb63 = (
    sha256_file_cell6_nb63(
        FINAL_MAIN
    )
)


rf_hash_before_cell6_nb63 = (
    sha256_file_cell6_nb63(
        CERTIFIED_MODEL
    )
)


rf_size_before_cell6_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


assert (
    main_hash_before_cell6_nb63
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    rf_hash_before_cell6_nb63
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    rf_size_before_cell6_nb63
    ==
    EXPECTED_MODEL_SIZE
)


print(
    "main.py SHA256 :",
    main_hash_before_cell6_nb63,
)

print(
    "RF model SHA256:",
    rf_hash_before_cell6_nb63,
)

print(
    "RF model size  :",
    rf_size_before_cell6_nb63,
)


print()
print("[OK] Certified assets frozen before policy test.")


# =============================================================================
# 3. AUTHORITATIVE DEPLOYMENT AGENT DIRECTORY
# =============================================================================

REAL_FINAL_AGENT_CELL6_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
)


REAL_FINAL_AGENT_SRC_CELL6_NB63 = (
    REAL_FINAL_AGENT_CELL6_NB63
    /
    "src"
)


REAL_FINAL_AGENT_MODELS_CELL6_NB63 = (
    REAL_FINAL_AGENT_CELL6_NB63
    /
    "models"
)


assert REAL_FINAL_AGENT_CELL6_NB63.is_dir()
assert REAL_FINAL_AGENT_SRC_CELL6_NB63.is_dir()
assert REAL_FINAL_AGENT_MODELS_CELL6_NB63.is_dir()


final_ppo_agent_path_cell6_nb63 = (
    REAL_FINAL_AGENT_SRC_CELL6_NB63
    /
    "agents"
    /
    "final_ppo_agent.py"
)


ppo_engine_path_cell6_nb63 = (
    REAL_FINAL_AGENT_SRC_CELL6_NB63
    /
    "ppo"
    /
    "ppo_policy_engine.py"
)


assert final_ppo_agent_path_cell6_nb63.is_file()
assert ppo_engine_path_cell6_nb63.is_file()


print()
print("=" * 92)
print("2. AUTHORITATIVE DEPLOYMENT POLICY FILES")
print("=" * 92)


print(
    "[OK]",
    final_ppo_agent_path_cell6_nb63.relative_to(
        PROJECT_ROOT
    ),
)

print(
    "[OK]",
    ppo_engine_path_cell6_nb63.relative_to(
        PROJECT_ROOT
    ),
)


# =============================================================================
# 4. DEPLOYMENT CHECKPOINT DISCOVERY
# =============================================================================

print()
print("=" * 92)
print("3. DEPLOYMENT CHECKPOINT DISCOVERY")
print("=" * 92)


checkpoint_extensions_cell6_nb63 = {
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".joblib",
}


deployment_checkpoint_candidates_cell6_nb63 = sorted(
    [
        path.resolve()
        for path
        in REAL_FINAL_AGENT_MODELS_CELL6_NB63.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in checkpoint_extensions_cell6_nb63
        )
    ],
    key=lambda p: str(p).lower(),
)


print(
    "Deployment checkpoint candidates:",
    len(
        deployment_checkpoint_candidates_cell6_nb63
    ),
)


for index, path in enumerate(
    deployment_checkpoint_candidates_cell6_nb63
):

    print()
    print(
        f"[{index}]",
        path,
    )

    print(
        "    size  :",
        path.stat().st_size,
    )

    print(
        "    SHA256:",
        sha256_file_cell6_nb63(
            path
        ),
    )


assert (
    deployment_checkpoint_candidates_cell6_nb63
), (
    "No checkpoint exists under "
    "submission/final_agent/models."
)


# =============================================================================
# 5. REAL DEPLOYMENT SNAPSHOT
# =============================================================================

real_final_agent_snapshot_before_cell6_nb63 = (
    snapshot_noncache_cell6_nb63(
        REAL_FINAL_AGENT_CELL6_NB63
    )
)


print()
print("=" * 92)
print("4. REAL DEPLOYMENT SNAPSHOT")
print("=" * 92)


print(
    "Deployment files:",
    len(
        real_final_agent_snapshot_before_cell6_nb63
    ),
)


print()
print(
    "[OK] Real final_agent directory frozen for comparison."
)


# =============================================================================
# 6. BUILD FAITHFUL TEMPORARY DEPLOYMENT COPY
# =============================================================================

print()
print("=" * 92)
print("5. FAITHFUL TEMPORARY DEPLOYMENT COPY")
print("=" * 92)


temp_root_cell6_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell6_"
    )
).resolve()


temp_final_agent_cell6_nb63 = (
    temp_root_cell6_nb63
    /
    "final_agent"
)


shutil.copytree(
    REAL_FINAL_AGENT_CELL6_NB63,
    temp_final_agent_cell6_nb63,
)


temp_src_cell6_nb63 = (
    temp_final_agent_cell6_nb63
    /
    "src"
)


temp_models_cell6_nb63 = (
    temp_final_agent_cell6_nb63
    /
    "models"
)


assert temp_src_cell6_nb63.is_dir()
assert temp_models_cell6_nb63.is_dir()


# -------------------------------------------------------------------------
# Add package markers ONLY to temporary copy if absent.
#
# This keeps the package name `src`, instead of renaming it to runtime_pkg.
# -------------------------------------------------------------------------

temporary_package_markers_cell6_nb63 = []


for package_dir in [
    temp_src_cell6_nb63,
    temp_src_cell6_nb63 / "agents",
]:

    assert package_dir.is_dir()

    init_path = (
        package_dir
        /
        "__init__.py"
    )

    if not init_path.exists():

        init_path.write_text(
            "# Temporary NB63 package marker.\n",
            encoding="utf-8",
        )

        temporary_package_markers_cell6_nb63.append(
            str(
                init_path.relative_to(
                    temp_root_cell6_nb63
                )
            )
        )


print(
    "Temporary root:",
    temp_root_cell6_nb63,
)

print(
    "Temporary final_agent:",
    temp_final_agent_cell6_nb63,
)

print(
    "Temporary package markers added:",
    temporary_package_markers_cell6_nb63,
)


print()
print(
    "[OK] Original final_agent/src namespace preserved."
)


# =============================================================================
# 7. BUILD POLICY TEST DRIVER
# =============================================================================

driver_path_cell6_nb63 = (
    temp_root_cell6_nb63
    /
    "cell6_policy_driver.py"
)


driver_source_cell6_nb63 = r'''
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


RESULT = {

    "sys_path_root":
        None,

    "imports_pass":
        False,

    "import_error":
        None,

    "checkpoint_candidates":
        [],

    "checkpoint_attempts":
        [],

    "selected_checkpoint":
        None,

    "selected_checkpoint_name":
        None,

    "agent_constructed":
        False,

    "state_constructed":
        False,

    "legal_move_count":
        None,

    "legal_moves":
        None,

    "choose_move_probe_pass":
        False,

    "choose_move_probe":
        None,

    "simulation_called":
        False,

    "simulation_completed":
        False,

    "winner":
        None,

    "turn_count":
        None,

    "record_count":
        None,

    "stop_reason":
        None,

    "transcript_created":
        False,

    "transcript_preview":
        None,

    "exception_type":
        None,

    "exception_message":
        None,

    "traceback":
        None,
}


try:

    # =========================================================================
    # CRITICAL:
    #
    # Add final_agent directory itself to sys.path so Python resolves:
    #
    #     src.battle_state
    #     src.agents.final_ppo_agent
    #     src.ppo.ppo_policy_engine
    #
    # exactly under the real deployment namespace.
    # =========================================================================

    final_agent_root = Path(
        sys.argv[1]
    ).resolve()


    final_agent_text = str(
        final_agent_root
    )


    sys.path.insert(
        0,
        final_agent_text,
    )


    RESULT[
        "sys_path_root"
    ] = final_agent_text


    try:

        from src.battle_state import (
            PokemonState,
            PlayerState,
            BattleState,
        )

        from src.legal_moves import (
            get_current_legal_moves,
        )

        from src.agents.final_ppo_agent import (
            FinalPPOBattleAgent,
        )

        from src.battle_simulation import (
            simulate_ai_battle,
            create_battle_transcript,
        )

        RESULT[
            "imports_pass"
        ] = True


    except Exception as import_exc:

        RESULT[
            "import_error"
        ] = (
            f"{type(import_exc).__name__}: "
            f"{import_exc}"
        )

        raise


    # =========================================================================
    # Deterministic synthetic legal battle state
    # =========================================================================

    player_pokemon = PokemonState(
        card={
            "name":
                "NB63_POLICY_PLAYER",

            "attacks": [
                {
                    "name":
                        "Policy Strike",

                    "damage":
                        20.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Policy Heavy Strike",

                    "damage":
                        30.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },
        current_hp=90.0,
        attached_energy=1,
        status=None,
        damage=0.0,
        is_active=True,
    )


    opponent_pokemon = PokemonState(
        card={
            "name":
                "NB63_POLICY_OPPONENT",

            "attacks": [
                {
                    "name":
                        "Opponent Strike",

                    "damage":
                        15.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                }
            ],
        },
        current_hp=90.0,
        attached_energy=1,
        status=None,
        damage=0.0,
        is_active=True,
    )


    player = PlayerState(
        active=player_pokemon,
        bench=[],
        prize_cards_remaining=6,
        hand_size=7,
    )


    opponent = PlayerState(
        active=opponent_pokemon,
        bench=[],
        prize_cards_remaining=6,
        hand_size=7,
    )


    initial_state = BattleState(
        player=player,
        opponent=opponent,
        turn_number=1,
        current_player="Player",
    )


    RESULT[
        "state_constructed"
    ] = True


    legal_moves = (
        get_current_legal_moves(
            initial_state
        )
    )


    RESULT[
        "legal_move_count"
    ] = len(
        legal_moves
    )


    RESULT[
        "legal_moves"
    ] = legal_moves


    # =========================================================================
    # Deployment checkpoints — ONLY final_agent/models
    # =========================================================================

    models_dir = (
        final_agent_root
        /
        "models"
    )


    checkpoint_paths = sorted(
        [
            path
            for path
            in models_dir.rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".pkl",
                    ".pickle",
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".joblib",
                }
            )
        ]
    )


    RESULT[
        "checkpoint_candidates"
    ] = [
        str(path)
        for path
        in checkpoint_paths
    ]


    if not checkpoint_paths:

        raise RuntimeError(
            "No deployment checkpoint found "
            "under final_agent/models."
        )


    selected_agent = None
    selected_checkpoint = None


    for checkpoint_path in checkpoint_paths:

        attempt = {

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "name":
                checkpoint_path.name,

            "constructed":
                False,

            "choose_move_pass":
                False,

            "error":
                None,
        }


        try:

            agent = FinalPPOBattleAgent(
                checkpoint_path=str(
                    checkpoint_path
                ),
                device="cpu",
                deterministic=True,
            )


            attempt[
                "constructed"
            ] = True


            decision = agent.choose_move(
                initial_state,
                depth=1,
            )


            attempt[
                "choose_move_pass"
            ] = True


            attempt[
                "decision_type"
            ] = type(
                decision
            ).__name__


            attempt[
                "move"
            ] = getattr(
                decision,
                "move",
                None,
            )


            attempt[
                "score"
            ] = getattr(
                decision,
                "score",
                None,
            )


            selected_agent = agent

            selected_checkpoint = (
                checkpoint_path
            )


            RESULT[
                "choose_move_probe"
            ] = {

                "decision_type":
                    type(
                        decision
                    ).__name__,

                "move":
                    getattr(
                        decision,
                        "move",
                        None,
                    ),

                "score":
                    getattr(
                        decision,
                        "score",
                        None,
                    ),

                "search_depth":
                    getattr(
                        decision,
                        "search_depth",
                        None,
                    ),

                "nodes":
                    getattr(
                        decision,
                        "nodes",
                        None,
                    ),
            }


        except Exception as checkpoint_exc:

            attempt[
                "error"
            ] = (
                f"{type(checkpoint_exc).__name__}: "
                f"{checkpoint_exc}"
            )


        RESULT[
            "checkpoint_attempts"
        ].append(
            attempt
        )


        if selected_agent is not None:

            break


    if selected_agent is None:

        raise RuntimeError(
            "No deployment checkpoint successfully "
            "constructed FinalPPOBattleAgent and "
            "completed choose_move()."
        )


    RESULT[
        "selected_checkpoint"
    ] = str(
        selected_checkpoint
    )


    RESULT[
        "selected_checkpoint_name"
    ] = (
        selected_checkpoint.name
    )


    RESULT[
        "agent_constructed"
    ] = True


    RESULT[
        "choose_move_probe_pass"
    ] = True


    # =========================================================================
    # Exactly one deployment-policy battle
    # =========================================================================

    RESULT[
        "simulation_called"
    ] = True


    simulation = simulate_ai_battle(
        initial_state=initial_state,
        agent=selected_agent,
        search_depth=1,
        max_turns=12,
        verbose=False,
    )


    RESULT[
        "simulation_completed"
    ] = True


    RESULT[
        "winner"
    ] = getattr(
        simulation,
        "winner",
        None,
    )


    RESULT[
        "turn_count"
    ] = getattr(
        simulation,
        "turn_count",
        None,
    )


    RESULT[
        "stop_reason"
    ] = getattr(
        simulation,
        "stop_reason",
        None,
    )


    turns = getattr(
        simulation,
        "turns",
        None,
    )


    if turns is not None:

        RESULT[
            "record_count"
        ] = len(
            turns
        )


    try:

        transcript = (
            create_battle_transcript(
                simulation
            )
        )


        RESULT[
            "transcript_created"
        ] = True


        RESULT[
            "transcript_preview"
        ] = (
            str(
                transcript
            )[:1500]
        )


    except Exception as transcript_exc:

        RESULT[
            "transcript_created"
        ] = False


        RESULT[
            "transcript_error"
        ] = (
            f"{type(transcript_exc).__name__}: "
            f"{transcript_exc}"
        )


except Exception as exc:

    RESULT[
        "exception_type"
    ] = type(
        exc
    ).__name__


    RESULT[
        "exception_message"
    ] = str(
        exc
    )


    RESULT[
        "traceback"
    ] = traceback.format_exc()


print(
    "NB63_CELL6_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell6_nb63.write_text(
    textwrap.dedent(
        driver_source_cell6_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell6_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell6_nb63
    ),
    "exec",
)


print()
print(
    "[OK] Corrected policy-test driver created."
)

print(
    "[OK] Driver preserves `src` deployment namespace."
)


# =============================================================================
# 8. EXECUTE CORRECTED POLICY TEST
# =============================================================================

print()
print("=" * 92)
print("6. CORRECTED POLICY IMPORT / CONSTRUCTION / BATTLE TEST")
print("=" * 92)


environment_cell6_nb63 = (
    os.environ.copy()
)


environment_cell6_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell6_nb63 = subprocess.run(
    [
        sys.executable,

        str(
            driver_path_cell6_nb63
        ),

        str(
            temp_final_agent_cell6_nb63
        ),
    ],
    cwd=str(
        temp_root_cell6_nb63
    ),
    env=environment_cell6_nb63,
    capture_output=True,
    text=True,
    timeout=180,
)


print(
    "Return code:",
    process_cell6_nb63.returncode,
)


print()
print("--- STDOUT ---")

print(
    process_cell6_nb63.stdout
)


print()
print("--- STDERR ---")

print(
    process_cell6_nb63.stderr
)


assert (
    process_cell6_nb63.returncode
    ==
    0
), (
    "Corrected Cell 6 subprocess failed.\n"
    f"STDOUT:\n"
    f"{process_cell6_nb63.stdout}\n"
    f"STDERR:\n"
    f"{process_cell6_nb63.stderr}"
)


# =============================================================================
# 9. PARSE RESULT
# =============================================================================

result_prefix_cell6_nb63 = (
    "NB63_CELL6_RESULT_JSON="
)


result_line_cell6_nb63 = None


for line in (
    process_cell6_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell6_nb63
    ):

        result_line_cell6_nb63 = (
            line
        )


assert (
    result_line_cell6_nb63
    is not None
), (
    "Cell 6 result JSON not found."
)


policy_result_cell6_nb63 = json.loads(
    result_line_cell6_nb63[
        len(
            result_prefix_cell6_nb63
        ):
    ]
)


print()
print("=" * 92)
print("7. CORRECTED POLICY-TEST RESULT")
print("=" * 92)


print(
    json.dumps(
        policy_result_cell6_nb63,
        indent=2,
    )
)


# =============================================================================
# 10. IMPORT ASSERTION — WITH USEFUL ERROR MESSAGE
# =============================================================================

assert (
    policy_result_cell6_nb63[
        "imports_pass"
    ]
    is True
), (
    "Deployment-policy imports still failed.\n\n"
    "Import error:\n"
    f"{policy_result_cell6_nb63.get('import_error')}\n\n"
    "Traceback:\n"
    f"{policy_result_cell6_nb63.get('traceback')}"
)


print()
print(
    "[OK] Deployment src namespace imported successfully."
)


# =============================================================================
# 11. STATE / LEGAL MOVE ASSERTIONS
# =============================================================================

assert (
    policy_result_cell6_nb63[
        "state_constructed"
    ]
    is True
)


assert (
    policy_result_cell6_nb63[
        "legal_move_count"
    ]
    is not None
)


assert (
    policy_result_cell6_nb63[
        "legal_move_count"
    ]
    >
    0
), (
    "Synthetic policy state produced no legal moves."
)


print(
    "[OK] Policy smoke state constructed."
)

print(
    "[OK] Legal moves available:",
    policy_result_cell6_nb63[
        "legal_move_count"
    ],
)


# =============================================================================
# 12. POLICY CONSTRUCTION ASSERTIONS
# =============================================================================

assert (
    policy_result_cell6_nb63[
        "agent_constructed"
    ]
    is True
), (
    "Deployment policy checkpoint could not be loaded.\n\n"
    "Checkpoint attempts:\n"
    f"{json.dumps(policy_result_cell6_nb63['checkpoint_attempts'], indent=2)}\n\n"
    "Traceback:\n"
    f"{policy_result_cell6_nb63.get('traceback')}"
)


assert (
    policy_result_cell6_nb63[
        "choose_move_probe_pass"
    ]
    is True
), (
    "FinalPPOBattleAgent did not complete choose_move()."
)


print()
print(
    "[OK] FinalPPOBattleAgent constructed."
)

print(
    "[OK] choose_move() produced a decision."
)


# =============================================================================
# 13. BATTLE ASSERTIONS
# =============================================================================

assert (
    policy_result_cell6_nb63[
        "simulation_called"
    ]
    is True
)


assert (
    policy_result_cell6_nb63[
        "simulation_completed"
    ]
    is True
), (
    "Policy battle did not complete.\n\n"
    f"{policy_result_cell6_nb63.get('traceback')}"
)


assert (
    policy_result_cell6_nb63[
        "exception_type"
    ]
    is None
), (
    "Policy battle raised an exception.\n\n"
    f"{policy_result_cell6_nb63.get('traceback')}"
)


print()
print(
    "[OK] One deployment-policy battle completed."
)


# =============================================================================
# 14. MAP SELECTED TEMPORARY CHECKPOINT BACK TO REAL SOURCE
# =============================================================================

selected_checkpoint_name_cell6_nb63 = (
    policy_result_cell6_nb63[
        "selected_checkpoint_name"
    ]
)


selected_checkpoint_source_cell6_nb63 = None


for source_path in (
    deployment_checkpoint_candidates_cell6_nb63
):

    if (
        source_path.name
        ==
        selected_checkpoint_name_cell6_nb63
    ):

        selected_checkpoint_source_cell6_nb63 = (
            source_path
        )

        break


assert (
    selected_checkpoint_source_cell6_nb63
    is not None
), (
    "Selected checkpoint could not be mapped "
    "back to submission/final_agent/models."
)


selected_checkpoint_hash_cell6_nb63 = (
    sha256_file_cell6_nb63(
        selected_checkpoint_source_cell6_nb63
    )
)


print()
print("=" * 92)
print("8. SELECTED DEPLOYMENT CHECKPOINT")
print("=" * 92)


print(
    "Source:",
    selected_checkpoint_source_cell6_nb63,
)

print(
    "Size:",
    selected_checkpoint_source_cell6_nb63.stat().st_size,
)

print(
    "SHA256:",
    selected_checkpoint_hash_cell6_nb63,
)


# =============================================================================
# 15. PERSIST EVIDENCE
# =============================================================================

cell6_evidence_nb63 = {

    "notebook":
        63,

    "cell":
        6,

    "purpose":
        "CORRECTED_DEPLOYMENT_POLICY_SINGLE_BATTLE_SMOKE_TEST",

    "deployment_namespace":
        "src",

    "deployment_agent_class":
        "FinalPPOBattleAgent",

    "selected_checkpoint":
        str(
            selected_checkpoint_source_cell6_nb63
        ),

    "selected_checkpoint_sha256":
        selected_checkpoint_hash_cell6_nb63,

    "selected_checkpoint_size":
        selected_checkpoint_source_cell6_nb63.stat().st_size,

    "certified_rf_model":
        str(
            CERTIFIED_MODEL
        ),

    "certified_rf_model_sha256":
        rf_hash_before_cell6_nb63,

    "certified_rf_used_as_ppo_checkpoint":
        (
            selected_checkpoint_source_cell6_nb63.resolve()
            ==
            CERTIFIED_MODEL.resolve()
        ),

    "temporary_execution_only":
        True,

    "search_depth":
        1,

    "max_turns":
        12,

    "result":
        policy_result_cell6_nb63,
}


cell6_output_path_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell6_deployment_policy_smoke_result.json"
)


cell6_artifact_path_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell6_deployment_policy_smoke_result.json"
)


for path in [
    cell6_output_path_nb63,
    cell6_artifact_path_nb63,
]:

    path.write_text(
        json.dumps(
            cell6_evidence_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("9. EVIDENCE PERSISTENCE")
print("=" * 92)


print(
    "[OK]",
    cell6_output_path_nb63,
)

print(
    "[OK]",
    cell6_artifact_path_nb63,
)


# =============================================================================
# 16. REAL DEPLOYMENT IMMUTABILITY
# =============================================================================

real_final_agent_snapshot_after_cell6_nb63 = (
    snapshot_noncache_cell6_nb63(
        REAL_FINAL_AGENT_CELL6_NB63
    )
)


assert (
    real_final_agent_snapshot_after_cell6_nb63
    ==
    real_final_agent_snapshot_before_cell6_nb63
), (
    "Real final_agent directory changed during Cell 6."
)


print()
print("=" * 92)
print("10. REAL DEPLOYMENT IMMUTABILITY")
print("=" * 92)


print(
    "[OK] Real final_agent deployment files remained unchanged."
)


# =============================================================================
# 17. CERTIFIED ASSET POST-TEST INTEGRITY
# =============================================================================

main_hash_after_cell6_nb63 = (
    sha256_file_cell6_nb63(
        FINAL_MAIN
    )
)


rf_hash_after_cell6_nb63 = (
    sha256_file_cell6_nb63(
        CERTIFIED_MODEL
    )
)


rf_size_after_cell6_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


assert (
    main_hash_after_cell6_nb63
    ==
    main_hash_before_cell6_nb63
)


assert (
    rf_hash_after_cell6_nb63
    ==
    rf_hash_before_cell6_nb63
)


assert (
    rf_size_after_cell6_nb63
    ==
    rf_size_before_cell6_nb63
)


print()
print("=" * 92)
print("11. CERTIFIED ASSET POST-TEST INTEGRITY")
print("=" * 92)


print(
    "main.py SHA256 :",
    main_hash_after_cell6_nb63,
)

print(
    "RF model SHA256:",
    rf_hash_after_cell6_nb63,
)

print(
    "RF model size  :",
    rf_size_after_cell6_nb63,
)


print()
print(
    "[OK] Authoritative main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] No training or retraining occurred."
)


# =============================================================================
# 18. TEMPORARY CLEANUP
# =============================================================================

print()
print("=" * 92)
print("12. TEMPORARY POLICY TEST CLEANUP")
print("=" * 92)


shutil.rmtree(
    temp_root_cell6_nb63,
    ignore_errors=True,
)


temporary_deleted_cell6_nb63 = (
    not temp_root_cell6_nb63.exists()
)


assert (
    temporary_deleted_cell6_nb63
    is True
)


print(
    "[OK] Temporary deployment copy deleted."
)


# =============================================================================
# 19. FINAL STATUS
# =============================================================================

CELL6_NB63_STATUS = {

    "cell5_pass":
        True,

    "corrected_package_layout":
        True,

    "deployment_namespace":
        "src",

    "deployment_policy_class":
        "FinalPPOBattleAgent",

    "checkpoint_candidates_discovered":
        len(
            deployment_checkpoint_candidates_cell6_nb63
        ),

    "selected_checkpoint":
        str(
            selected_checkpoint_source_cell6_nb63
        ),

    "selected_checkpoint_sha256":
        selected_checkpoint_hash_cell6_nb63,

    "selected_checkpoint_size":
        selected_checkpoint_source_cell6_nb63.stat().st_size,

    "imports_pass":
        True,

    "policy_agent_constructed":
        True,

    "choose_move_probe_pass":
        True,

    "legal_move_count":
        policy_result_cell6_nb63[
            "legal_move_count"
        ],

    "simulation_executed":
        True,

    "battle_count":
        1,

    "simulation_completed":
        True,

    "winner":
        policy_result_cell6_nb63.get(
            "winner"
        ),

    "turn_count":
        policy_result_cell6_nb63.get(
            "turn_count"
        ),

    "record_count":
        policy_result_cell6_nb63.get(
            "record_count"
        ),

    "stop_reason":
        policy_result_cell6_nb63.get(
            "stop_reason"
        ),

    "exception_type":
        None,

    "certified_rf_used_as_ppo_checkpoint":
        (
            selected_checkpoint_source_cell6_nb63.resolve()
            ==
            CERTIFIED_MODEL.resolve()
        ),

    "real_runtime_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "temporary_runtime_deleted":
        temporary_deleted_cell6_nb63,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 6 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL6_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Corrected deployment policy smoke test completed."
)

print(
    "Actual src package namespace preserved."
)

print(
    "Certified assets remained frozen."
)

print()
print(
    "NEXT STEP: Cell 7 — reproducible multi-battle "
    "campaign design and baseline matrix."
)


# In[7]:


# =============================================================================
# NOTEBOOK 63 — CELL 7
# REPRODUCIBLE MULTI-BATTLE CAMPAIGN DESIGN + BASELINE MATRIX
# DESIGN / VALIDATION ONLY — NO LARGE BATCH EXECUTION
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 7")
print("REPRODUCIBLE MULTI-BATTLE CAMPAIGN DESIGN + BASELINE MATRIX")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL6_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 6 must PASS before Cell 7."

assert (
    CELL6_NB63_STATUS["policy_agent_constructed"]
    is True
)

assert (
    CELL6_NB63_STATUS["choose_move_probe_pass"]
    is True
)

assert (
    CELL6_NB63_STATUS["simulation_completed"]
    is True
)

assert (
    CELL6_NB63_STATUS["real_runtime_modified"]
    is False
)

assert (
    CELL6_NB63_STATUS["certified_model_integrity_preserved"]
    is True
)


print()
print("[OK] Cell 6 PASS confirmed.")
print("[OK] Deployment policy execution path proven.")
print("[OK] Cell 7 is campaign design only.")
print("[OK] No large simulation batch will execute in this cell.")


# =============================================================================
# 1. FREEZE DEPLOYMENT POLICY IDENTITY
# =============================================================================

DEPLOYMENT_POLICY_CHECKPOINT_NB63 = Path(
    CELL6_NB63_STATUS[
        "selected_checkpoint"
    ]
).resolve()


DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63 = (
    CELL6_NB63_STATUS[
        "selected_checkpoint_sha256"
    ]
)


DEPLOYMENT_POLICY_CHECKPOINT_SIZE_NB63 = (
    CELL6_NB63_STATUS[
        "selected_checkpoint_size"
    ]
)


assert (
    DEPLOYMENT_POLICY_CHECKPOINT_NB63.is_file()
)


assert (
    DEPLOYMENT_POLICY_CHECKPOINT_NB63.stat().st_size
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SIZE_NB63
)


actual_checkpoint_hash_cell7_nb63 = (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
)


assert (
    actual_checkpoint_hash_cell7_nb63
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("1. DEPLOYMENT POLICY FREEZE")
print("=" * 92)

print(
    "Checkpoint:",
    DEPLOYMENT_POLICY_CHECKPOINT_NB63,
)

print(
    "SHA256   :",
    actual_checkpoint_hash_cell7_nb63,
)

print(
    "Bytes    :",
    DEPLOYMENT_POLICY_CHECKPOINT_SIZE_NB63,
)

print()
print("[OK] Deployment policy checkpoint frozen.")


# =============================================================================
# 2. CERTIFIED ASSET REVALIDATION
# =============================================================================

main_hash_cell7_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

rf_hash_cell7_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)

rf_size_cell7_nb63 = (
    CERTIFIED_MODEL.stat().st_size
)


assert (
    main_hash_cell7_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    rf_hash_cell7_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    rf_size_cell7_nb63
    ==
    EXPECTED_MODEL_SIZE
)


print()
print("=" * 92)
print("2. CERTIFIED ASSET REVALIDATION")
print("=" * 92)

print(
    "main.py SHA256 :",
    main_hash_cell7_nb63,
)

print(
    "RF model SHA256:",
    rf_hash_cell7_nb63,
)

print(
    "RF model size  :",
    rf_size_cell7_nb63,
)

print()
print("[OK] Certified assets remain frozen.")


# =============================================================================
# 3. CAMPAIGN PRINCIPLES
# =============================================================================

NB63_CAMPAIGN_PRINCIPLES = {

    "purpose":
        "FINAL_STRATEGY_SIMULATION_EVIDENCE",

    "evaluation_only":
        True,

    "training_allowed":
        False,

    "retraining_allowed":
        False,

    "submission_modification_allowed":
        False,

    "certified_model_modification_allowed":
        False,

    "deterministic_seed_control":
        True,

    "side_swap_required":
        True,

    "timeout_is_not_win":
        True,

    "unknown_move_tracking_required":
        True,

    "zero_damage_tracking_required":
        True,

    "illegal_move_tracking_required":
        True,

    "runtime_exception_tracking_required":
        True,

    "checkpoint_identity_tracking_required":
        True,

    "evidence_persistence_required":
        True,
}


print()
print("=" * 92)
print("3. CAMPAIGN PRINCIPLES")
print("=" * 92)

print(
    json.dumps(
        NB63_CAMPAIGN_PRINCIPLES,
        indent=2,
    )
)


# =============================================================================
# 4. DETERMINISTIC SEED SET
# =============================================================================

NB63_CAMPAIGN_SEEDS = [
    63001,
    63002,
    63003,
    63004,
    63005,
    63006,
    63007,
    63008,
    63009,
    63010,
]


assert (
    len(
        NB63_CAMPAIGN_SEEDS
    )
    ==
    len(
        set(
            NB63_CAMPAIGN_SEEDS
        )
    )
)


print()
print("=" * 92)
print("4. DETERMINISTIC SEEDS")
print("=" * 92)

print(
    NB63_CAMPAIGN_SEEDS
)

print()
print(
    "[OK] 10 unique deterministic seeds defined."
)


# =============================================================================
# 5. CONTROLLED BATTLE SCENARIOS
# =============================================================================

NB63_SCENARIOS = [

    {
        "scenario_id":
            "S1_BALANCED_LOW_HP",

        "description":
            "Balanced low-HP battle with zero-cost and one-cost attacks.",

        "player": {
            "name":
                "S1_Player",

            "hp":
                60.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Quick Strike",

                    "damage":
                        10.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Power Strike",

                    "damage":
                        20.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },

        "opponent": {
            "name":
                "S1_Opponent",

            "hp":
                60.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Quick Jab",

                    "damage":
                        10.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Heavy Jab",

                    "damage":
                        20.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },
    },


    {
        "scenario_id":
            "S2_BALANCED_HIGH_HP",

        "description":
            "Balanced higher-HP battle to test sustained policy behavior.",

        "player": {
            "name":
                "S2_Player",

            "hp":
                120.0,

            "energy":
                2,

            "attacks": [
                {
                    "name":
                        "Steady Hit",

                    "damage":
                        15.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Heavy Hit",

                    "damage":
                        30.0,

                    "energy_cost":
                        2,

                    "effect":
                        None,
                },
            ],
        },

        "opponent": {
            "name":
                "S2_Opponent",

            "hp":
                120.0,

            "energy":
                2,

            "attacks": [
                {
                    "name":
                        "Steady Blow",

                    "damage":
                        15.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Heavy Blow",

                    "damage":
                        30.0,

                    "energy_cost":
                        2,

                    "effect":
                        None,
                },
            ],
        },
    },


    {
        "scenario_id":
            "S3_PLAYER_DAMAGE_ADVANTAGE",

        "description":
            "Player has stronger attacks; tests whether policy exploits advantage.",

        "player": {
            "name":
                "S3_Player",

            "hp":
                90.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Fast Hit",

                    "damage":
                        20.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Finisher",

                    "damage":
                        35.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },

        "opponent": {
            "name":
                "S3_Opponent",

            "hp":
                90.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Weak Hit",

                    "damage":
                        10.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Moderate Hit",

                    "damage":
                        20.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },
    },


    {
        "scenario_id":
            "S4_OPPONENT_DAMAGE_ADVANTAGE",

        "description":
            "Opponent has stronger attacks; tests policy behavior under disadvantage.",

        "player": {
            "name":
                "S4_Player",

            "hp":
                90.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Weak Hit",

                    "damage":
                        10.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Moderate Hit",

                    "damage":
                        20.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },

        "opponent": {
            "name":
                "S4_Opponent",

            "hp":
                90.0,

            "energy":
                1,

            "attacks": [
                {
                    "name":
                        "Fast Hit",

                    "damage":
                        20.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                },

                {
                    "name":
                        "Finisher",

                    "damage":
                        35.0,

                    "energy_cost":
                        1,

                    "effect":
                        None,
                },
            ],
        },
    },
]


scenario_ids_cell7_nb63 = [
    scenario[
        "scenario_id"
    ]
    for scenario
    in NB63_SCENARIOS
]


assert (
    len(
        scenario_ids_cell7_nb63
    )
    ==
    len(
        set(
            scenario_ids_cell7_nb63
        )
    )
)


print()
print("=" * 92)
print("5. CONTROLLED SCENARIOS")
print("=" * 92)

for scenario in NB63_SCENARIOS:

    print()
    print(
        scenario[
            "scenario_id"
        ],
        "—",
        scenario[
            "description"
        ],
    )

    print(
        "  Player HP:",
        scenario[
            "player"
        ][
            "hp"
        ],
        "| Energy:",
        scenario[
            "player"
        ][
            "energy"
        ],
    )

    print(
        "  Opponent HP:",
        scenario[
            "opponent"
        ][
            "hp"
        ],
        "| Energy:",
        scenario[
            "opponent"
        ][
            "energy"
        ],
    )


print()
print(
    "[OK]",
    len(
        NB63_SCENARIOS
    ),
    "controlled scenarios defined."
)


# =============================================================================
# 6. SIDE-SWAP MATRIX
# =============================================================================

NB63_SIDE_MODES = [
    {
        "side_mode":
            "NORMAL",

        "swap_sides":
            False,
    },

    {
        "side_mode":
            "SWAPPED",

        "swap_sides":
            True,
    },
]


print()
print("=" * 92)
print("6. SIDE-SWAP MATRIX")
print("=" * 92)

print(
    json.dumps(
        NB63_SIDE_MODES,
        indent=2,
    )
)

print()
print(
    "[OK] Normal and swapped-side evaluation required."
)


# =============================================================================
# 7. BASELINE DEFINITIONS
# =============================================================================

NB63_BASELINES = [

    {
        "baseline_id":
            "DEPLOYMENT_POLICY",

        "description":
            "FinalPPOBattleAgent using preserved deployment checkpoint.",

        "agent_type":
            "FinalPPOBattleAgent",

        "checkpoint":
            str(
                DEPLOYMENT_POLICY_CHECKPOINT_NB63
            ),

        "checkpoint_sha256":
            DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,
    },


    {
        "baseline_id":
            "FIRST_LEGAL_MOVE",

        "description":
            "Deterministic baseline selecting the first legal move.",

        "agent_type":
            "DeterministicFirstLegalMoveAgent",

        "checkpoint":
            None,

        "checkpoint_sha256":
            None,
    },


    {
        "baseline_id":
            "MAX_DAMAGE_LEGAL_MOVE",

        "description":
            "Deterministic baseline choosing the legal move with highest damage.",

        "agent_type":
            "DeterministicMaxDamageAgent",

        "checkpoint":
            None,

        "checkpoint_sha256":
            None,
    },
]


print()
print("=" * 92)
print("7. BASELINE MATRIX")
print("=" * 92)

for baseline in NB63_BASELINES:

    print()
    print(
        baseline[
            "baseline_id"
        ]
    )

    print(
        " ",
        baseline[
            "description"
        ]
    )


print()
print(
    "[OK]",
    len(
        NB63_BASELINES
    ),
    "agent/baseline definitions established."
)


# =============================================================================
# 8. EXECUTION SETTINGS
# =============================================================================

NB63_EXECUTION_SETTINGS = {

    "search_depth":
        1,

    "max_turns":
        40,

    "verbose":
        False,

    "seeds_per_scenario":
        len(
            NB63_CAMPAIGN_SEEDS
        ),

    "side_modes":
        len(
            NB63_SIDE_MODES
        ),

    "scenario_count":
        len(
            NB63_SCENARIOS
        ),

    "agent_variants":
        len(
            NB63_BASELINES
        ),
}


planned_battles_per_agent_cell7_nb63 = (
    NB63_EXECUTION_SETTINGS[
        "scenario_count"
    ]
    *
    NB63_EXECUTION_SETTINGS[
        "side_modes"
    ]
    *
    NB63_EXECUTION_SETTINGS[
        "seeds_per_scenario"
    ]
)


planned_total_battles_cell7_nb63 = (
    planned_battles_per_agent_cell7_nb63
    *
    NB63_EXECUTION_SETTINGS[
        "agent_variants"
    ]
)


NB63_EXECUTION_SETTINGS[
    "planned_battles_per_agent"
] = (
    planned_battles_per_agent_cell7_nb63
)


NB63_EXECUTION_SETTINGS[
    "planned_total_battles"
] = (
    planned_total_battles_cell7_nb63
)


print()
print("=" * 92)
print("8. EXECUTION SETTINGS")
print("=" * 92)

print(
    json.dumps(
        NB63_EXECUTION_SETTINGS,
        indent=2,
    )
)


print()
print(
    "[OK] Planned battles per agent:",
    planned_battles_per_agent_cell7_nb63,
)

print(
    "[OK] Planned total battles:",
    planned_total_battles_cell7_nb63,
)


# =============================================================================
# 9. REQUIRED PER-BATTLE METRICS
# =============================================================================

NB63_PER_BATTLE_FIELDS = [

    "battle_id",
    "agent_variant",
    "scenario_id",
    "side_mode",
    "seed",

    "winner",
    "normalized_outcome",

    "turn_count",
    "stop_reason",

    "player_final_hp",
    "opponent_final_hp",

    "total_moves",
    "player_moves",
    "opponent_moves",

    "unknown_move_count",
    "unknown_move_rate",

    "zero_damage_move_count",
    "zero_damage_move_rate",

    "positive_damage_move_count",
    "positive_damage_move_rate",

    "total_damage_dealt",

    "policy_score_mean",
    "policy_score_min",
    "policy_score_max",

    "runtime_exception",
    "runtime_exception_type",

    "checkpoint_sha256",
]


print()
print("=" * 92)
print("9. REQUIRED PER-BATTLE METRICS")
print("=" * 92)

for field in NB63_PER_BATTLE_FIELDS:

    print(
        " ",
        field
    )


print()
print(
    "[OK]",
    len(
        NB63_PER_BATTLE_FIELDS
    ),
    "per-battle metrics required."
)


# =============================================================================
# 10. NORMALIZED OUTCOME RULES
# =============================================================================

NB63_OUTCOME_RULES = {

    "Player":
        "PLAYER_WIN",

    "Opponent":
        "OPPONENT_WIN",

    "None + maximum turn":
        "TIMEOUT_DRAW",

    "None + other normal stop":
        "NO_WINNER",

    "exception":
        "RUNTIME_FAILURE",
}


print()
print("=" * 92)
print("10. OUTCOME NORMALIZATION")
print("=" * 92)

print(
    json.dumps(
        NB63_OUTCOME_RULES,
        indent=2,
    )
)


# =============================================================================
# 11. STRATEGIC QUALITY SIGNALS
# =============================================================================

NB63_STRATEGIC_QUALITY_SIGNALS = {

    "unknown_move_rate":
        (
            "Measures semantic move-mapping quality. "
            "Lower is better."
        ),

    "zero_damage_move_rate":
        (
            "Measures frequency of decisions that produce "
            "no battle damage. Lower is generally better "
            "for these controlled attack scenarios."
        ),

    "positive_damage_move_rate":
        (
            "Measures how often selected moves produce "
            "meaningful offensive progress."
        ),

    "timeout_rate":
        (
            "Measures how often a battle fails to resolve "
            "within the controlled turn budget."
        ),

    "win_rate":
        (
            "Primary outcome metric where the simulator "
            "produces a winner."
        ),

    "side_swap_delta":
        (
            "Difference in performance between NORMAL "
            "and SWAPPED conditions."
        ),

    "baseline_delta":
        (
            "Difference between deployment policy and "
            "deterministic baselines."
        ),
}


print()
print("=" * 92)
print("11. STRATEGIC QUALITY SIGNALS")
print("=" * 92)

print(
    json.dumps(
        NB63_STRATEGIC_QUALITY_SIGNALS,
        indent=2,
    )
)


# =============================================================================
# 12. CAMPAIGN MATRIX
# =============================================================================

NB63_CAMPAIGN_MATRIX = []


battle_counter_cell7_nb63 = 0


for baseline in NB63_BASELINES:

    for scenario in NB63_SCENARIOS:

        for side_mode in NB63_SIDE_MODES:

            for seed in NB63_CAMPAIGN_SEEDS:

                battle_counter_cell7_nb63 += 1


                battle_id = (
                    f"NB63_"
                    f"{battle_counter_cell7_nb63:04d}"
                )


                NB63_CAMPAIGN_MATRIX.append(
                    {
                        "battle_id":
                            battle_id,

                        "agent_variant":
                            baseline[
                                "baseline_id"
                            ],

                        "agent_type":
                            baseline[
                                "agent_type"
                            ],

                        "scenario_id":
                            scenario[
                                "scenario_id"
                            ],

                        "side_mode":
                            side_mode[
                                "side_mode"
                            ],

                        "swap_sides":
                            side_mode[
                                "swap_sides"
                            ],

                        "seed":
                            seed,

                        "search_depth":
                            NB63_EXECUTION_SETTINGS[
                                "search_depth"
                            ],

                        "max_turns":
                            NB63_EXECUTION_SETTINGS[
                                "max_turns"
                            ],

                        "checkpoint_sha256":
                            baseline[
                                "checkpoint_sha256"
                            ],
                    }
                )


assert (
    len(
        NB63_CAMPAIGN_MATRIX
    )
    ==
    planned_total_battles_cell7_nb63
)


battle_ids_cell7_nb63 = [
    record[
        "battle_id"
    ]
    for record
    in NB63_CAMPAIGN_MATRIX
]


assert (
    len(
        battle_ids_cell7_nb63
    )
    ==
    len(
        set(
            battle_ids_cell7_nb63
        )
    )
)


print()
print("=" * 92)
print("12. CAMPAIGN MATRIX")
print("=" * 92)

print(
    "Campaign rows:",
    len(
        NB63_CAMPAIGN_MATRIX
    )
)

print()
print("First 10 planned battles:")


for record in NB63_CAMPAIGN_MATRIX[:10]:

    print(
        json.dumps(
            record
        )
    )


print()
print(
    "[OK] Campaign matrix is complete and uniquely indexed."
)


# =============================================================================
# 13. CAMPAIGN CONFIGURATION HASH
# =============================================================================

NB63_CAMPAIGN_CONFIGURATION = {

    "policy_checkpoint":
        str(
            DEPLOYMENT_POLICY_CHECKPOINT_NB63
        ),

    "policy_checkpoint_sha256":
        DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,

    "certified_rf_sha256":
        EXPECTED_MODEL_SHA256,

    "main_sha256":
        EXPECTED_MAIN_SHA256,

    "principles":
        NB63_CAMPAIGN_PRINCIPLES,

    "seeds":
        NB63_CAMPAIGN_SEEDS,

    "scenarios":
        NB63_SCENARIOS,

    "side_modes":
        NB63_SIDE_MODES,

    "baselines":
        NB63_BASELINES,

    "execution_settings":
        NB63_EXECUTION_SETTINGS,

    "required_fields":
        NB63_PER_BATTLE_FIELDS,

    "quality_signals":
        NB63_STRATEGIC_QUALITY_SIGNALS,
}


campaign_config_json_cell7_nb63 = (
    json.dumps(
        NB63_CAMPAIGN_CONFIGURATION,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )
)


NB63_CAMPAIGN_CONFIG_SHA256 = (
    hashlib.sha256(
        campaign_config_json_cell7_nb63.encode(
            "utf-8"
        )
    ).hexdigest()
)


print()
print("=" * 92)
print("13. CAMPAIGN CONFIGURATION HASH")
print("=" * 92)

print(
    "SHA256:",
    NB63_CAMPAIGN_CONFIG_SHA256
)

print()
print(
    "[OK] Reproducible campaign configuration fingerprint created."
)


# =============================================================================
# 14. PERSIST CAMPAIGN DESIGN
# =============================================================================

campaign_design_artifact_cell7_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell7_campaign_design.json"
)


campaign_design_output_cell7_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell7_campaign_design.json"
)


campaign_matrix_artifact_cell7_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell7_campaign_matrix.json"
)


campaign_matrix_output_cell7_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell7_campaign_matrix.json"
)


campaign_design_payload_cell7_nb63 = {

    "notebook":
        63,

    "cell":
        7,

    "purpose":
        "REPRODUCIBLE_MULTI_BATTLE_CAMPAIGN_DESIGN",

    "configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "configuration":
        NB63_CAMPAIGN_CONFIGURATION,

    "planned_total_battles":
        planned_total_battles_cell7_nb63,

    "simulation_executed":
        False,

    "training_executed":
        False,
}


for path in [
    campaign_design_artifact_cell7_nb63,
    campaign_design_output_cell7_nb63,
]:

    path.write_text(
        json.dumps(
            campaign_design_payload_cell7_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


for path in [
    campaign_matrix_artifact_cell7_nb63,
    campaign_matrix_output_cell7_nb63,
]:

    path.write_text(
        json.dumps(
            NB63_CAMPAIGN_MATRIX,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("14. CAMPAIGN DESIGN PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    campaign_design_artifact_cell7_nb63,
)

print(
    "[OK]",
    campaign_design_output_cell7_nb63,
)

print(
    "[OK]",
    campaign_matrix_artifact_cell7_nb63,
)

print(
    "[OK]",
    campaign_matrix_output_cell7_nb63,
)


# =============================================================================
# 15. POST-DESIGN INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("15. POST-DESIGN ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py remains unchanged."
)

print(
    "[OK] Certified RF model remains unchanged."
)

print(
    "[OK] Deployment PPO checkpoint remains unchanged."
)

print(
    "[OK] No simulation batch executed."
)

print(
    "[OK] No training or retraining executed."
)


# =============================================================================
# 16. FINAL STATUS
# =============================================================================

CELL7_NB63_STATUS = {

    "cell6_pass":
        True,

    "campaign_design_complete":
        True,

    "configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "scenario_count":
        len(
            NB63_SCENARIOS
        ),

    "seed_count":
        len(
            NB63_CAMPAIGN_SEEDS
        ),

    "side_mode_count":
        len(
            NB63_SIDE_MODES
        ),

    "agent_variant_count":
        len(
            NB63_BASELINES
        ),

    "planned_battles_per_agent":
        planned_battles_per_agent_cell7_nb63,

    "planned_total_battles":
        planned_total_battles_cell7_nb63,

    "unknown_move_tracking_required":
        True,

    "zero_damage_tracking_required":
        True,

    "side_swap_required":
        True,

    "timeout_normalization_defined":
        True,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "simulation_executed":
        False,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 7 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL7_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Reproducible simulation campaign design is frozen."
)

print(
    f"Planned campaign size: "
    f"{planned_total_battles_cell7_nb63} battles."
)

print(
    "No campaign battles were executed in Cell 7."
)

print()
print(
    "NEXT STEP: Cell 8 — execute the first controlled campaign batch "
    "and capture battle-level evidence."
)


# In[8]:


# =============================================================================
# NOTEBOOK 63 — CELL 8
# CONTROLLED CAMPAIGN PILOT EXECUTION
# 24 BATTLES: 4 SCENARIOS × 2 SIDE MODES × 3 AGENT VARIANTS × 1 SEED
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 8")
print("CONTROLLED CAMPAIGN PILOT EXECUTION")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL7_NB63_STATUS["validation_status"]
    ==
    "PASS"
), "Cell 7 must PASS before Cell 8."

assert (
    CELL7_NB63_STATUS["campaign_design_complete"]
    is True
)

assert (
    CELL7_NB63_STATUS["configuration_sha256"]
    ==
    NB63_CAMPAIGN_CONFIG_SHA256
)

assert (
    CELL7_NB63_STATUS["simulation_executed"]
    is False
)


print()
print("[OK] Cell 7 PASS confirmed.")
print("[OK] Frozen campaign configuration confirmed.")
print("[OK] Pilot execution only — full 240-battle campaign not yet authorized.")


# =============================================================================
# 1. PILOT BATCH DEFINITION
# =============================================================================

PILOT_SEED_CELL8_NB63 = 63001


pilot_matrix_cell8_nb63 = [
    record
    for record in NB63_CAMPAIGN_MATRIX
    if (
        record["seed"]
        ==
        PILOT_SEED_CELL8_NB63
    )
]


expected_pilot_battles_cell8_nb63 = (
    len(NB63_SCENARIOS)
    *
    len(NB63_SIDE_MODES)
    *
    len(NB63_BASELINES)
)


assert (
    len(pilot_matrix_cell8_nb63)
    ==
    expected_pilot_battles_cell8_nb63
)


print()
print("=" * 92)
print("1. PILOT BATCH")
print("=" * 92)

print(
    "Pilot seed:",
    PILOT_SEED_CELL8_NB63,
)

print(
    "Pilot battles:",
    len(
        pilot_matrix_cell8_nb63
    ),
)

print(
    "Scenarios:",
    len(
        NB63_SCENARIOS
    ),
)

print(
    "Side modes:",
    len(
        NB63_SIDE_MODES
    ),
)

print(
    "Agent variants:",
    len(
        NB63_BASELINES
    ),
)


assert (
    len(
        pilot_matrix_cell8_nb63
    )
    ==
    24
)


print()
print("[OK] 24-battle representative pilot established.")


# =============================================================================
# 2. CERTIFIED-ASSET FREEZE
# =============================================================================

main_hash_before_cell8_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

rf_hash_before_cell8_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)

checkpoint_hash_before_cell8_nb63 = (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
)


assert (
    main_hash_before_cell8_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    rf_hash_before_cell8_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    checkpoint_hash_before_cell8_nb63
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("2. PRE-PILOT ASSET FREEZE")
print("=" * 92)

print(
    "main.py:",
    main_hash_before_cell8_nb63,
)

print(
    "RF model:",
    rf_hash_before_cell8_nb63,
)

print(
    "PPO checkpoint:",
    checkpoint_hash_before_cell8_nb63,
)

print()
print("[OK] All authoritative assets frozen.")


# =============================================================================
# 3. BUILD PILOT PAYLOAD
# =============================================================================

scenario_lookup_cell8_nb63 = {
    scenario["scenario_id"]:
        scenario

    for scenario
    in NB63_SCENARIOS
}


pilot_payload_cell8_nb63 = {

    "campaign_configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "checkpoint_sha256":
        DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,

    "search_depth":
        NB63_EXECUTION_SETTINGS[
            "search_depth"
        ],

    "max_turns":
        NB63_EXECUTION_SETTINGS[
            "max_turns"
        ],

    "scenarios":
        NB63_SCENARIOS,

    "pilot_matrix":
        pilot_matrix_cell8_nb63,
}


# =============================================================================
# 4. TEMPORARY DEPLOYMENT COPY
# =============================================================================

REAL_FINAL_AGENT_CELL8_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
)


assert (
    REAL_FINAL_AGENT_CELL8_NB63.is_dir()
)


temp_root_cell8_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell8_"
    )
).resolve()


temp_final_agent_cell8_nb63 = (
    temp_root_cell8_nb63
    /
    "final_agent"
)


shutil.copytree(
    REAL_FINAL_AGENT_CELL8_NB63,
    temp_final_agent_cell8_nb63,
)


# Add package markers only in temporary copy if needed.

for package_dir in [
    temp_final_agent_cell8_nb63 / "src",
    temp_final_agent_cell8_nb63 / "src" / "agents",
]:

    assert package_dir.is_dir()

    init_path = (
        package_dir
        /
        "__init__.py"
    )

    if not init_path.exists():

        init_path.write_text(
            "# Temporary NB63 pilot package marker.\n",
            encoding="utf-8",
        )


payload_path_cell8_nb63 = (
    temp_root_cell8_nb63
    /
    "pilot_payload.json"
)


payload_path_cell8_nb63.write_text(
    json.dumps(
        pilot_payload_cell8_nb63,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("3. TEMPORARY PILOT RUNTIME")
print("=" * 92)

print(
    "Temporary root:",
    temp_root_cell8_nb63,
)

print(
    "Temporary final_agent:",
    temp_final_agent_cell8_nb63,
)

print()
print(
    "[OK] Pilot will execute entirely against temporary deployment copy."
)


# =============================================================================
# 5. PILOT DRIVER
# =============================================================================

driver_path_cell8_nb63 = (
    temp_root_cell8_nb63
    /
    "cell8_pilot_driver.py"
)


driver_source_cell8_nb63 = r'''
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import traceback
from pathlib import Path


RESULT = {
    "imports_pass":
        False,

    "policy_constructed":
        False,

    "requested_battles":
        0,

    "completed_battles":
        0,

    "runtime_failures":
        0,

    "records":
        [],

    "fatal_exception_type":
        None,

    "fatal_exception_message":
        None,

    "fatal_traceback":
        None,
}


try:

    final_agent_root = Path(
        sys.argv[1]
    ).resolve()

    payload_path = Path(
        sys.argv[2]
    ).resolve()


    sys.path.insert(
        0,
        str(
            final_agent_root
        ),
    )


    from src.battle_state import (
        PokemonState,
        PlayerState,
        BattleState,
    )

    from src.agent_decision import (
        AgentDecision,
    )

    from src.legal_moves import (
        get_current_legal_moves,
    )

    from src.agents.final_ppo_agent import (
        FinalPPOBattleAgent,
    )

    from src.battle_simulation import (
        simulate_ai_battle,
    )


    RESULT[
        "imports_pass"
    ] = True


    payload = json.loads(
        payload_path.read_text(
            encoding="utf-8"
        )
    )


    matrix = payload[
        "pilot_matrix"
    ]

    scenarios = {
        record[
            "scenario_id"
        ]:
            record

        for record
        in payload[
            "scenarios"
        ]
    }


    RESULT[
        "requested_battles"
    ] = len(
        matrix
    )


    # =========================================================================
    # Reproducibility helper
    # =========================================================================

    def set_seed(seed):

        random.seed(
            seed
        )

        try:

            import numpy as np

            np.random.seed(
                seed
                %
                (2 ** 32 - 1)
            )

        except Exception:

            pass


        try:

            import torch

            torch.manual_seed(
                seed
            )

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(
                    seed
                )

        except Exception:

            pass


    # =========================================================================
    # State builder
    # =========================================================================

    def make_pokemon(record):

        return PokemonState(
            card={
                "name":
                    record[
                        "name"
                    ],

                "attacks":
                    record[
                        "attacks"
                    ],
            },

            current_hp=float(
                record[
                    "hp"
                ]
            ),

            attached_energy=int(
                record[
                    "energy"
                ]
            ),

            status=None,

            damage=0.0,

            is_active=True,
        )


    def build_state(
        scenario,
        swap_sides,
    ):

        player_record = (
            scenario[
                "player"
            ]
        )

        opponent_record = (
            scenario[
                "opponent"
            ]
        )


        if swap_sides:

            player_record, opponent_record = (
                opponent_record,
                player_record,
            )


        player = PlayerState(
            active=make_pokemon(
                player_record
            ),
            bench=[],
            prize_cards_remaining=6,
            hand_size=7,
        )


        opponent = PlayerState(
            active=make_pokemon(
                opponent_record
            ),
            bench=[],
            prize_cards_remaining=6,
            hand_size=7,
        )


        return BattleState(
            player=player,
            opponent=opponent,
            turn_number=1,
            current_player="Player",
        )


    # =========================================================================
    # Deterministic baseline agents
    # =========================================================================

    class FirstLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            if not legal_moves:

                move = {
                    "name":
                        "Unknown Move",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                }

            else:

                move = (
                    legal_moves[
                        0
                    ]
                )


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=1,

                principal_variation=[
                    move
                ],
            )


    class MaxDamageLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            if not legal_moves:

                move = {
                    "name":
                        "Unknown Move",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        None,
                }

            else:

                move = max(
                    legal_moves,
                    key=lambda item: float(
                        item.get(
                            "damage",
                            0.0,
                        )
                        or
                        0.0
                    ),
                )


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=len(
                    legal_moves
                ),

                principal_variation=[
                    move
                ],
            )


    # =========================================================================
    # Construct deployment policy once
    # =========================================================================

    checkpoint_candidates = sorted(
        [
            path
            for path
            in (
                final_agent_root
                /
                "models"
            ).rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".pkl",
                    ".pickle",
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".joblib",
                }
            )
        ]
    )


    if not checkpoint_candidates:

        raise RuntimeError(
            "No deployment policy checkpoint found."
        )


    deployment_agent = (
        FinalPPOBattleAgent(
            checkpoint_path=str(
                checkpoint_candidates[
                    0
                ]
            ),

            device="cpu",

            deterministic=True,
        )
    )


    RESULT[
        "policy_constructed"
    ] = True


    baseline_agents = {
        "DEPLOYMENT_POLICY":
            deployment_agent,

        "FIRST_LEGAL_MOVE":
            FirstLegalMoveAgent(),

        "MAX_DAMAGE_LEGAL_MOVE":
            MaxDamageLegalMoveAgent(),
    }


    # =========================================================================
    # Execute pilot battles
    # =========================================================================

    for battle_spec in matrix:

        record = {

            "battle_id":
                battle_spec[
                    "battle_id"
                ],

            "agent_variant":
                battle_spec[
                    "agent_variant"
                ],

            "scenario_id":
                battle_spec[
                    "scenario_id"
                ],

            "side_mode":
                battle_spec[
                    "side_mode"
                ],

            "seed":
                battle_spec[
                    "seed"
                ],

            "winner":
                None,

            "normalized_outcome":
                None,

            "turn_count":
                None,

            "stop_reason":
                None,

            "player_final_hp":
                None,

            "opponent_final_hp":
                None,

            "total_moves":
                0,

            "player_moves":
                0,

            "opponent_moves":
                0,

            "unknown_move_count":
                0,

            "unknown_move_rate":
                0.0,

            "zero_damage_move_count":
                0,

            "zero_damage_move_rate":
                0.0,

            "positive_damage_move_count":
                0,

            "positive_damage_move_rate":
                0.0,

            "total_damage_dealt":
                0.0,

            "policy_score_mean":
                None,

            "policy_score_min":
                None,

            "policy_score_max":
                None,

            "runtime_exception":
                False,

            "runtime_exception_type":
                None,

            "runtime_exception_message":
                None,

            "checkpoint_sha256":
                battle_spec.get(
                    "checkpoint_sha256"
                ),
        }


        try:

            seed = int(
                battle_spec[
                    "seed"
                ]
            )


            set_seed(
                seed
            )


            scenario = scenarios[
                battle_spec[
                    "scenario_id"
                ]
            ]


            initial_state = build_state(
                scenario=scenario,

                swap_sides=bool(
                    battle_spec[
                        "swap_sides"
                    ]
                ),
            )


            agent = baseline_agents[
                battle_spec[
                    "agent_variant"
                ]
            ]


            simulation = (
                simulate_ai_battle(
                    initial_state=initial_state,

                    agent=agent,

                    search_depth=int(
                        battle_spec[
                            "search_depth"
                        ]
                    ),

                    max_turns=int(
                        battle_spec[
                            "max_turns"
                        ]
                    ),

                    verbose=False,
                )
            )


            winner = getattr(
                simulation,
                "winner",
                None,
            )


            stop_reason = getattr(
                simulation,
                "stop_reason",
                None,
            )


            turn_count = getattr(
                simulation,
                "turn_count",
                None,
            )


            turns = list(
                getattr(
                    simulation,
                    "turns",
                    []
                )
                or
                []
            )


            final_state = getattr(
                simulation,
                "final_state",
                None,
            )


            record[
                "winner"
            ] = winner


            record[
                "turn_count"
            ] = turn_count


            record[
                "stop_reason"
            ] = stop_reason


            # -------------------------------------------------------------
            # Outcome normalization
            # -------------------------------------------------------------

            if winner == "Player":

                normalized_outcome = (
                    "PLAYER_WIN"
                )


            elif winner == "Opponent":

                normalized_outcome = (
                    "OPPONENT_WIN"
                )


            elif (
                turn_count
                is not None
                and
                int(
                    turn_count
                )
                >=
                int(
                    battle_spec[
                        "max_turns"
                    ]
                )
            ):

                normalized_outcome = (
                    "TIMEOUT_DRAW"
                )


            else:

                normalized_outcome = (
                    "NO_WINNER"
                )


            record[
                "normalized_outcome"
            ] = normalized_outcome


            # -------------------------------------------------------------
            # Final HP
            # -------------------------------------------------------------

            if final_state is not None:

                try:

                    record[
                        "player_final_hp"
                    ] = float(
                        final_state
                        .player
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


                try:

                    record[
                        "opponent_final_hp"
                    ] = float(
                        final_state
                        .opponent
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


            # -------------------------------------------------------------
            # Turn-level metrics
            # -------------------------------------------------------------

            record[
                "total_moves"
            ] = len(
                turns
            )


            move_names = []

            damages = []

            scores = []


            for turn in turns:

                acting_side = getattr(
                    turn,
                    "acting_side",
                    None,
                )


                if acting_side == "Player":

                    record[
                        "player_moves"
                    ] += 1


                elif acting_side == "Opponent":

                    record[
                        "opponent_moves"
                    ] += 1


                move_name = str(
                    getattr(
                        turn,
                        "move_name",
                        "",
                    )
                    or
                    ""
                )


                move_names.append(
                    move_name
                )


                if (
                    move_name.strip().lower()
                    ==
                    "unknown move"
                ):

                    record[
                        "unknown_move_count"
                    ] += 1


                raw_damage = getattr(
                    turn,
                    "damage",
                    0.0,
                )


                try:

                    damage = float(
                        raw_damage
                        or
                        0.0
                    )

                except Exception:

                    damage = 0.0


                damages.append(
                    damage
                )


                if damage <= 0:

                    record[
                        "zero_damage_move_count"
                    ] += 1


                else:

                    record[
                        "positive_damage_move_count"
                    ] += 1


                raw_score = getattr(
                    turn,
                    "search_score",
                    None,
                )


                if raw_score is not None:

                    try:

                        scores.append(
                            float(
                                raw_score
                            )
                        )

                    except Exception:

                        pass


            total_moves = record[
                "total_moves"
            ]


            if total_moves > 0:

                record[
                    "unknown_move_rate"
                ] = (
                    record[
                        "unknown_move_count"
                    ]
                    /
                    total_moves
                )


                record[
                    "zero_damage_move_rate"
                ] = (
                    record[
                        "zero_damage_move_count"
                    ]
                    /
                    total_moves
                )


                record[
                    "positive_damage_move_rate"
                ] = (
                    record[
                        "positive_damage_move_count"
                    ]
                    /
                    total_moves
                )


            record[
                "total_damage_dealt"
            ] = float(
                sum(
                    damages
                )
            )


            if scores:

                record[
                    "policy_score_mean"
                ] = float(
                    statistics.mean(
                        scores
                    )
                )


                record[
                    "policy_score_min"
                ] = float(
                    min(
                        scores
                    )
                )


                record[
                    "policy_score_max"
                ] = float(
                    max(
                        scores
                    )
                )


            RESULT[
                "completed_battles"
            ] += 1


        except Exception as battle_exc:

            record[
                "runtime_exception"
            ] = True


            record[
                "runtime_exception_type"
            ] = type(
                battle_exc
            ).__name__


            record[
                "runtime_exception_message"
            ] = str(
                battle_exc
            )


            record[
                "normalized_outcome"
            ] = "RUNTIME_FAILURE"


            RESULT[
                "runtime_failures"
            ] += 1


            record[
                "traceback"
            ] = traceback.format_exc()


        RESULT[
            "records"
        ].append(
            record
        )


except Exception as fatal_exc:

    RESULT[
        "fatal_exception_type"
    ] = type(
        fatal_exc
    ).__name__


    RESULT[
        "fatal_exception_message"
    ] = str(
        fatal_exc
    )


    RESULT[
        "fatal_traceback"
    ] = traceback.format_exc()


print(
    "NB63_CELL8_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell8_nb63.write_text(
    textwrap.dedent(
        driver_source_cell8_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell8_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell8_nb63
    ),
    "exec",
)


print()
print(
    "[OK] Pilot driver created and compiled."
)


# =============================================================================
# 6. EXECUTE PILOT
# =============================================================================

print()
print("=" * 92)
print("4. EXECUTE 24-BATTLE PILOT")
print("=" * 92)


environment_cell8_nb63 = (
    os.environ.copy()
)


environment_cell8_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell8_nb63 = subprocess.run(
    [
        sys.executable,

        str(
            driver_path_cell8_nb63
        ),

        str(
            temp_final_agent_cell8_nb63
        ),

        str(
            payload_path_cell8_nb63
        ),
    ],

    cwd=str(
        temp_root_cell8_nb63
    ),

    env=environment_cell8_nb63,

    capture_output=True,

    text=True,

    timeout=600,
)


print(
    "Return code:",
    process_cell8_nb63.returncode,
)


print()
print("--- STDERR ---")

print(
    process_cell8_nb63.stderr
)


assert (
    process_cell8_nb63.returncode
    ==
    0
), (
    "Pilot subprocess failed.\n"
    f"STDOUT:\n{process_cell8_nb63.stdout}\n"
    f"STDERR:\n{process_cell8_nb63.stderr}"
)


# =============================================================================
# 7. PARSE RESULT
# =============================================================================

result_prefix_cell8_nb63 = (
    "NB63_CELL8_RESULT_JSON="
)


result_line_cell8_nb63 = None


for line in (
    process_cell8_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell8_nb63
    ):

        result_line_cell8_nb63 = (
            line
        )


assert (
    result_line_cell8_nb63
    is not None
), (
    "Cell 8 pilot result JSON not found."
)


pilot_result_cell8_nb63 = json.loads(
    result_line_cell8_nb63[
        len(
            result_prefix_cell8_nb63
        ):
    ]
)


print()
print("=" * 92)
print("5. PILOT EXECUTION SUMMARY")
print("=" * 92)


summary_preview_cell8_nb63 = {

    "imports_pass":
        pilot_result_cell8_nb63[
            "imports_pass"
        ],

    "policy_constructed":
        pilot_result_cell8_nb63[
            "policy_constructed"
        ],

    "requested_battles":
        pilot_result_cell8_nb63[
            "requested_battles"
        ],

    "completed_battles":
        pilot_result_cell8_nb63[
            "completed_battles"
        ],

    "runtime_failures":
        pilot_result_cell8_nb63[
            "runtime_failures"
        ],

    "fatal_exception_type":
        pilot_result_cell8_nb63[
            "fatal_exception_type"
        ],
}


print(
    json.dumps(
        summary_preview_cell8_nb63,
        indent=2,
    )
)


# =============================================================================
# 8. BASIC PILOT VALIDATION
# =============================================================================

assert (
    pilot_result_cell8_nb63[
        "imports_pass"
    ]
    is True
)


assert (
    pilot_result_cell8_nb63[
        "policy_constructed"
    ]
    is True
)


assert (
    pilot_result_cell8_nb63[
        "requested_battles"
    ]
    ==
    24
)


assert (
    len(
        pilot_result_cell8_nb63[
            "records"
        ]
    )
    ==
    24
)


assert (
    pilot_result_cell8_nb63[
        "fatal_exception_type"
    ]
    is None
), (
    pilot_result_cell8_nb63.get(
        "fatal_traceback"
    )
)


print()
print(
    "[OK] Pilot infrastructure completed all 24 requested records."
)


# =============================================================================
# 9. AGGREGATE PILOT METRICS
# =============================================================================

records_cell8_nb63 = (
    pilot_result_cell8_nb63[
        "records"
    ]
)


agent_summary_cell8_nb63 = {}


for baseline in NB63_BASELINES:

    agent_id = (
        baseline[
            "baseline_id"
        ]
    )


    rows = [
        row
        for row in records_cell8_nb63
        if (
            row[
                "agent_variant"
            ]
            ==
            agent_id
        )
    ]


    completed_rows = [
        row
        for row in rows
        if not row[
            "runtime_exception"
        ]
    ]


    outcomes = {}


    for row in rows:

        outcome = (
            row[
                "normalized_outcome"
            ]
        )

        outcomes[
            outcome
        ] = (
            outcomes.get(
                outcome,
                0,
            )
            +
            1
        )


    total_moves = sum(
        int(
            row.get(
                "total_moves",
                0,
            )
            or
            0
        )
        for row
        in completed_rows
    )


    unknown_moves = sum(
        int(
            row.get(
                "unknown_move_count",
                0,
            )
            or
            0
        )
        for row
        in completed_rows
    )


    zero_damage_moves = sum(
        int(
            row.get(
                "zero_damage_move_count",
                0,
            )
            or
            0
        )
        for row
        in completed_rows
    )


    positive_damage_moves = sum(
        int(
            row.get(
                "positive_damage_move_count",
                0,
            )
            or
            0
        )
        for row
        in completed_rows
    )


    total_damage = sum(
        float(
            row.get(
                "total_damage_dealt",
                0.0,
            )
            or
            0.0
        )
        for row
        in completed_rows
    )


    agent_summary_cell8_nb63[
        agent_id
    ] = {

        "battles":
            len(
                rows
            ),

        "runtime_failures":
            sum(
                1
                for row
                in rows
                if row[
                    "runtime_exception"
                ]
            ),

        "outcomes":
            outcomes,

        "total_moves":
            total_moves,

        "unknown_move_count":
            unknown_moves,

        "unknown_move_rate":
            (
                unknown_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "zero_damage_move_count":
            zero_damage_moves,

        "zero_damage_move_rate":
            (
                zero_damage_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "positive_damage_move_count":
            positive_damage_moves,

        "positive_damage_move_rate":
            (
                positive_damage_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "total_damage":
            total_damage,
    }


print()
print("=" * 92)
print("6. PILOT AGENT SUMMARY")
print("=" * 92)


print(
    json.dumps(
        agent_summary_cell8_nb63,
        indent=2,
    )
)


# =============================================================================
# 10. PILOT QUALITY CLASSIFICATION
# =============================================================================

deployment_summary_cell8_nb63 = (
    agent_summary_cell8_nb63[
        "DEPLOYMENT_POLICY"
    ]
)


deployment_unknown_rate_cell8_nb63 = (
    deployment_summary_cell8_nb63[
        "unknown_move_rate"
    ]
)


deployment_zero_damage_rate_cell8_nb63 = (
    deployment_summary_cell8_nb63[
        "zero_damage_move_rate"
    ]
)


deployment_runtime_failures_cell8_nb63 = (
    deployment_summary_cell8_nb63[
        "runtime_failures"
    ]
)


pilot_infrastructure_pass_cell8_nb63 = (
    pilot_result_cell8_nb63[
        "runtime_failures"
    ]
    ==
    0
)


deployment_semantic_quality_cell8_nb63 = (

    "HEALTHY"

    if (
        deployment_unknown_rate_cell8_nb63
        ==
        0.0

        and

        deployment_zero_damage_rate_cell8_nb63
        <
        0.50
    )

    else

    "SEMANTIC_MAPPING_CONCERN"
)


print()
print("=" * 92)
print("7. PILOT QUALITY CLASSIFICATION")
print("=" * 92)

print(
    "Infrastructure pass:",
    pilot_infrastructure_pass_cell8_nb63,
)

print(
    "Deployment unknown-move rate:",
    deployment_unknown_rate_cell8_nb63,
)

print(
    "Deployment zero-damage rate:",
    deployment_zero_damage_rate_cell8_nb63,
)

print(
    "Deployment semantic quality:",
    deployment_semantic_quality_cell8_nb63,
)


# =============================================================================
# 11. PERSIST BATTLE-LEVEL EVIDENCE
# =============================================================================

pilot_json_output_cell8_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell8_pilot_battle_records.json"
)


pilot_json_artifact_cell8_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell8_pilot_battle_records.json"
)


pilot_csv_output_cell8_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell8_pilot_battle_records.csv"
)


pilot_summary_output_cell8_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell8_pilot_summary.json"
)


pilot_summary_artifact_cell8_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell8_pilot_summary.json"
)


for path in [
    pilot_json_output_cell8_nb63,
    pilot_json_artifact_cell8_nb63,
]:

    path.write_text(
        json.dumps(
            records_cell8_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


csv_fieldnames_cell8_nb63 = sorted(
    {
        key
        for row
        in records_cell8_nb63
        for key
        in row.keys()
        if key
        !=
        "traceback"
    }
)


with pilot_csv_output_cell8_nb63.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=csv_fieldnames_cell8_nb63,
    )

    writer.writeheader()


    for row in records_cell8_nb63:

        clean_row = {
            key:
                row.get(
                    key
                )

            for key
            in csv_fieldnames_cell8_nb63
        }

        writer.writerow(
            clean_row
        )


pilot_summary_payload_cell8_nb63 = {

    "notebook":
        63,

    "cell":
        8,

    "purpose":
        "CONTROLLED_CAMPAIGN_PILOT",

    "campaign_configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "seed":
        PILOT_SEED_CELL8_NB63,

    "requested_battles":
        24,

    "runtime_failures":
        pilot_result_cell8_nb63[
            "runtime_failures"
        ],

    "agent_summary":
        agent_summary_cell8_nb63,

    "deployment_semantic_quality":
        deployment_semantic_quality_cell8_nb63,
}


for path in [
    pilot_summary_output_cell8_nb63,
    pilot_summary_artifact_cell8_nb63,
]:

    path.write_text(
        json.dumps(
            pilot_summary_payload_cell8_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("8. PILOT EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    pilot_json_output_cell8_nb63,
)

print(
    "[OK]",
    pilot_csv_output_cell8_nb63,
)

print(
    "[OK]",
    pilot_summary_output_cell8_nb63,
)

print(
    "[OK]",
    pilot_json_artifact_cell8_nb63,
)

print(
    "[OK]",
    pilot_summary_artifact_cell8_nb63,
)


# =============================================================================
# 12. TEMPORARY CLEANUP
# =============================================================================

shutil.rmtree(
    temp_root_cell8_nb63,
    ignore_errors=True,
)


temporary_deleted_cell8_nb63 = (
    not temp_root_cell8_nb63.exists()
)


assert (
    temporary_deleted_cell8_nb63
    is True
)


print()
print(
    "[OK] Temporary pilot runtime deleted."
)


# =============================================================================
# 13. POST-PILOT ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    main_hash_before_cell8_nb63
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    rf_hash_before_cell8_nb63
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    checkpoint_hash_before_cell8_nb63
)


print()
print("=" * 92)
print("9. POST-PILOT ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training/retraining occurred."
)


# =============================================================================
# 14. FINAL STATUS
# =============================================================================

CELL8_NB63_STATUS = {

    "cell7_pass":
        True,

    "campaign_configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "pilot_seed":
        PILOT_SEED_CELL8_NB63,

    "pilot_battles_requested":
        24,

    "pilot_battle_records":
        len(
            records_cell8_nb63
        ),

    "runtime_failure_count":
        pilot_result_cell8_nb63[
            "runtime_failures"
        ],

    "infrastructure_pass":
        pilot_infrastructure_pass_cell8_nb63,

    "deployment_unknown_move_rate":
        deployment_unknown_rate_cell8_nb63,

    "deployment_zero_damage_rate":
        deployment_zero_damage_rate_cell8_nb63,

    "deployment_semantic_quality":
        deployment_semantic_quality_cell8_nb63,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "temporary_runtime_deleted":
        True,

    "validation_status":
        (
            "PASS"
            if pilot_infrastructure_pass_cell8_nb63
            else
            "FAIL"
        ),
}


print()
print("=" * 92)

print(
    "NOTEBOOK 63 — CELL 8 STATUS:",
    CELL8_NB63_STATUS[
        "validation_status"
    ],
)

print("=" * 92)


print(
    json.dumps(
        CELL8_NB63_STATUS,
        indent=2,
    )
)


print()

if (
    CELL8_NB63_STATUS[
        "validation_status"
    ]
    ==
    "PASS"
):

    print(
        "24-battle pilot infrastructure passed."
    )

    print(
        "Pilot evidence has been persisted."
    )

    print(
        "NEXT STEP: inspect strategic-quality results before "
        "authorizing the remaining campaign battles."
    )

else:

    print(
        "Pilot infrastructure failed."
    )

    print(
        "Do NOT launch the remaining campaign."
    )


# In[9]:


# =============================================================================
# NOTEBOOK 63 — CELL 9
# LEGAL-MOVE SEMANTIC CONTRACT FORENSIC AUDIT
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 9")
print("LEGAL-MOVE SEMANTIC CONTRACT FORENSIC AUDIT")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert CELL8_NB63_STATUS["validation_status"] == "PASS"
assert CELL8_NB63_STATUS["infrastructure_pass"] is True

assert CELL8_NB63_STATUS["deployment_unknown_move_rate"] == 1.0
assert CELL8_NB63_STATUS["deployment_zero_damage_rate"] == 1.0

print()
print("[OK] Cell 8 infrastructure PASS confirmed.")
print("[OK] 100% unknown-move condition confirmed.")
print("[OK] 100% zero-damage condition confirmed.")
print("[OK] Full campaign remains BLOCKED pending semantic diagnosis.")


# =============================================================================
# 1. CLASSIFY CELL 8 FAILURE
# =============================================================================

records_cell9_nb63 = records_cell8_nb63

agent_failure_summary_cell9_nb63 = {}

for agent_id in [
    "DEPLOYMENT_POLICY",
    "FIRST_LEGAL_MOVE",
    "MAX_DAMAGE_LEGAL_MOVE",
]:

    rows = [
        row
        for row in records_cell9_nb63
        if row["agent_variant"] == agent_id
    ]

    agent_failure_summary_cell9_nb63[agent_id] = {
        "battle_count": len(rows),

        "all_timeout": all(
            row["normalized_outcome"] == "TIMEOUT_DRAW"
            for row in rows
        ),

        "all_unknown_moves": all(
            float(row["unknown_move_rate"]) == 1.0
            for row in rows
        ),

        "all_zero_damage": all(
            float(row["zero_damage_move_rate"]) == 1.0
            for row in rows
        ),

        "runtime_failures": sum(
            1
            for row in rows
            if row["runtime_exception"]
        ),
    }


print()
print("=" * 92)
print("1. CROSS-AGENT FAILURE PATTERN")
print("=" * 92)

print(
    json.dumps(
        agent_failure_summary_cell9_nb63,
        indent=2,
    )
)


cross_agent_failure_cell9_nb63 = all(
    item["all_unknown_moves"]
    and item["all_zero_damage"]
    and item["runtime_failures"] == 0

    for item
    in agent_failure_summary_cell9_nb63.values()
)


assert cross_agent_failure_cell9_nb63 is True


print()
print(
    "[CONFIRMED] Failure is shared by deployment policy and both deterministic baselines."
)

print(
    "[CONFIRMED] Cell 8 cannot be interpreted as a PPO-policy performance failure."
)


# =============================================================================
# 2. RUNTIME SOURCE RESOLUTION
# =============================================================================

runtime_src_cell9_nb63 = (
    SUBMISSION_DIR
    /
    "final_agent"
    /
    "src"
)

legal_moves_path_cell9_nb63 = (
    runtime_src_cell9_nb63
    /
    "legal_moves.py"
)

battle_state_path_cell9_nb63 = (
    runtime_src_cell9_nb63
    /
    "battle_state.py"
)

simulator_path_cell9_nb63 = (
    runtime_src_cell9_nb63
    /
    "simulator.py"
)

battle_agent_path_cell9_nb63 = (
    runtime_src_cell9_nb63
    /
    "battle_agent.py"
)


required_files_cell9_nb63 = [
    legal_moves_path_cell9_nb63,
    battle_state_path_cell9_nb63,
    simulator_path_cell9_nb63,
    battle_agent_path_cell9_nb63,
]


print()
print("=" * 92)
print("2. AUTHORITATIVE RUNTIME FILES")
print("=" * 92)

for path in required_files_cell9_nb63:

    assert path.is_file(), f"Missing runtime file: {path}"

    print(
        f"[OK] {path.relative_to(SUBMISSION_DIR)} "
        f"| {path.stat().st_size} bytes "
        f"| {sha256_file_nb63(path)[:16]}..."
    )


# =============================================================================
# 3. SOURCE LOADING
# =============================================================================

legal_moves_source_cell9_nb63 = (
    legal_moves_path_cell9_nb63.read_text(
        encoding="utf-8"
    )
)

battle_state_source_cell9_nb63 = (
    battle_state_path_cell9_nb63.read_text(
        encoding="utf-8"
    )
)

simulator_source_cell9_nb63 = (
    simulator_path_cell9_nb63.read_text(
        encoding="utf-8"
    )
)


# =============================================================================
# 4. LEGAL-MOVE FUNCTION INVENTORY
# =============================================================================

legal_tree_cell9_nb63 = ast.parse(
    legal_moves_source_cell9_nb63
)


legal_function_inventory_cell9_nb63 = []


for node in ast.walk(legal_tree_cell9_nb63):

    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    ):

        legal_function_inventory_cell9_nb63.append({
            "name": node.name,

            "line": node.lineno,

            "args": [
                arg.arg
                for arg in node.args.args
            ],
        })


print()
print("=" * 92)
print("3. legal_moves.py FUNCTION INVENTORY")
print("=" * 92)

for item in sorted(
    legal_function_inventory_cell9_nb63,
    key=lambda x: x["line"],
):

    print(
        f'{item["name"]} '
        f'line={item["line"]} '
        f'args={item["args"]}'
    )


# =============================================================================
# 5. FIND get_current_legal_moves CONTRACT
# =============================================================================

target_function_cell9_nb63 = None


for node in legal_tree_cell9_nb63.body:

    if (
        isinstance(node, ast.FunctionDef)
        and
        node.name == "get_current_legal_moves"
    ):

        target_function_cell9_nb63 = node

        break


assert (
    target_function_cell9_nb63 is not None
), "get_current_legal_moves() not found."


target_source_cell9_nb63 = ast.get_source_segment(
    legal_moves_source_cell9_nb63,
    target_function_cell9_nb63,
)


print()
print("=" * 92)
print("4. get_current_legal_moves() AUTHORITATIVE SOURCE")
print("=" * 92)

print(
    target_source_cell9_nb63
)


# =============================================================================
# 6. ATTRIBUTE / SUBSCRIPT CONTRACT EXTRACTION
# =============================================================================

attribute_refs_cell9_nb63 = sorted(
    {
        ast.unparse(node)

        for node
        in ast.walk(target_function_cell9_nb63)

        if isinstance(node, ast.Attribute)
    }
)


subscript_refs_cell9_nb63 = sorted(
    {
        ast.unparse(node)

        for node
        in ast.walk(target_function_cell9_nb63)

        if isinstance(node, ast.Subscript)
    }
)


string_constants_cell9_nb63 = sorted(
    {
        node.value

        for node
        in ast.walk(target_function_cell9_nb63)

        if (
            isinstance(node, ast.Constant)
            and
            isinstance(node.value, str)
        )
    }
)


print()
print("=" * 92)
print("5. LEGAL-MOVE DATA CONTRACT REFERENCES")
print("=" * 92)

print()
print("Attributes:")

for item in attribute_refs_cell9_nb63:
    print(" ", item)


print()
print("Subscripts:")

for item in subscript_refs_cell9_nb63:
    print(" ", item)


print()
print("String keys/constants:")

for item in string_constants_cell9_nb63:
    print(" ", repr(item))


# =============================================================================
# 7. SYNTHETIC SCENARIO CARD SCHEMA
# =============================================================================

print()
print("=" * 92)
print("6. CELL 7 SYNTHETIC CARD SCHEMA")
print("=" * 92)


synthetic_schema_examples_cell9_nb63 = []


for scenario in NB63_SCENARIOS:

    for side in [
        "player",
        "opponent",
    ]:

        record = scenario[side]

        synthetic_schema_examples_cell9_nb63.append({
            "scenario_id":
                scenario["scenario_id"],

            "side":
                side,

            "pokemon_keys":
                sorted(record.keys()),

            "attack_count":
                len(
                    record.get(
                        "attacks",
                        [],
                    )
                ),

            "attack_examples":
                record.get(
                    "attacks",
                    [],
                ),
        })


for example in synthetic_schema_examples_cell9_nb63[:4]:

    print()
    print(
        json.dumps(
            example,
            indent=2,
        )
    )


# =============================================================================
# 8. STATIC KEY-COMPATIBILITY AUDIT
# =============================================================================

synthetic_attack_keys_cell9_nb63 = set()


for scenario in NB63_SCENARIOS:

    for side in [
        "player",
        "opponent",
    ]:

        for attack in (
            scenario[side]
            .get(
                "attacks",
                []
            )
        ):

            if isinstance(
                attack,
                dict,
            ):

                synthetic_attack_keys_cell9_nb63.update(
                    attack.keys()
                )


print()
print("=" * 92)
print("7. SYNTHETIC ATTACK KEYS")
print("=" * 92)

print(
    sorted(
        synthetic_attack_keys_cell9_nb63
    )
)


potential_contract_keys_cell9_nb63 = set(
    string_constants_cell9_nb63
)


missing_candidate_keys_cell9_nb63 = sorted(
    potential_contract_keys_cell9_nb63
    -
    synthetic_attack_keys_cell9_nb63
)


print()
print(
    "Legal-move string constants not present "
    "in synthetic attack dictionaries:"
)

for item in missing_candidate_keys_cell9_nb63:

    print(
        " ",
        repr(item),
    )


# =============================================================================
# 9. simulator.py MOVE CONTRACT
# =============================================================================

simulator_tree_cell9_nb63 = ast.parse(
    simulator_source_cell9_nb63
)


simulator_functions_cell9_nb63 = []


for node in ast.walk(
    simulator_tree_cell9_nb63
):

    if isinstance(
        node,
        ast.FunctionDef,
    ):

        simulator_functions_cell9_nb63.append({
            "name":
                node.name,

            "line":
                node.lineno,

            "args":
                [
                    arg.arg
                    for arg
                    in node.args.args
                ],
        })


print()
print("=" * 92)
print("8. simulator.py FUNCTION INVENTORY")
print("=" * 92)


for item in sorted(
    simulator_functions_cell9_nb63,
    key=lambda x: x["line"],
):

    print(
        f'{item["name"]} '
        f'line={item["line"]} '
        f'args={item["args"]}'
    )


apply_move_node_cell9_nb63 = None


for node in simulator_tree_cell9_nb63.body:

    if (
        isinstance(node, ast.FunctionDef)
        and
        node.name == "apply_move"
    ):

        apply_move_node_cell9_nb63 = node

        break


assert (
    apply_move_node_cell9_nb63
    is not None
)


apply_move_source_cell9_nb63 = (
    ast.get_source_segment(
        simulator_source_cell9_nb63,
        apply_move_node_cell9_nb63,
    )
)


print()
print("=" * 92)
print("9. apply_move() AUTHORITATIVE SOURCE")
print("=" * 92)

print(
    apply_move_source_cell9_nb63
)


# =============================================================================
# 10. ROOT-CAUSE CLASSIFICATION
# =============================================================================

root_cause_cell9_nb63 = (
    "CAMPAIGN_SYNTHETIC_STATE_DOES_NOT_SATISFY_"
    "DEPLOYMENT_LEGAL_MOVE_CONTRACT"
)


classification_cell9_nb63 = {

    "cell8_runtime_infrastructure":
        "PASS",

    "deployment_policy_only_failure":
        False,

    "cross_agent_failure":
        True,

    "first_legal_baseline_failed":
        True,

    "max_damage_baseline_failed":
        True,

    "all_agents_unknown_move_rate":
        1.0,

    "all_agents_zero_damage_rate":
        1.0,

    "full_campaign_authorized":
        False,

    "root_cause_classification":
        root_cause_cell9_nb63,

    "required_next_action":
        (
            "BUILD_SCHEMA_CORRECT_SCENARIOS_FROM_"
            "AUTHORITATIVE_LEGAL_MOVE_CONTRACT"
        ),
}


print()
print("=" * 92)
print("10. ROOT-CAUSE CLASSIFICATION")
print("=" * 92)

print(
    json.dumps(
        classification_cell9_nb63,
        indent=2,
    )
)


# =============================================================================
# 11. EVIDENCE PERSISTENCE
# =============================================================================

cell9_evidence_nb63 = {

    "classification":
        classification_cell9_nb63,

    "legal_move_function_source":
        target_source_cell9_nb63,

    "legal_move_attribute_refs":
        attribute_refs_cell9_nb63,

    "legal_move_subscript_refs":
        subscript_refs_cell9_nb63,

    "legal_move_string_constants":
        string_constants_cell9_nb63,

    "synthetic_attack_keys":
        sorted(
            synthetic_attack_keys_cell9_nb63
        ),

    "potential_missing_keys":
        missing_candidate_keys_cell9_nb63,

    "apply_move_source":
        apply_move_source_cell9_nb63,

    "agent_failure_summary":
        agent_failure_summary_cell9_nb63,
}


cell9_output_path_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell9_legal_move_contract_audit.json"
)


cell9_artifact_path_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell9_legal_move_contract_audit.json"
)


for path in [
    cell9_output_path_nb63,
    cell9_artifact_path_nb63,
]:

    path.write_text(
        json.dumps(
            cell9_evidence_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("11. EVIDENCE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    cell9_output_path_nb63,
)

print(
    "[OK]",
    cell9_artifact_path_nb63,
)


# =============================================================================
# 12. ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("12. CERTIFIED ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training or simulation executed in Cell 9."
)


# =============================================================================
# 13. FINAL STATUS
# =============================================================================

CELL9_NB63_STATUS = {

    "cell8_pass":
        True,

    "forensic_audit_complete":
        True,

    "cross_agent_failure_confirmed":
        True,

    "deployment_policy_failure_proven":
        False,

    "campaign_harness_semantic_issue":
        True,

    "root_cause_classification":
        root_cause_cell9_nb63,

    "full_campaign_authorized":
        False,

    "simulation_executed":
        False,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print(
    "NOTEBOOK 63 — CELL 9 STATUS:",
    CELL9_NB63_STATUS[
        "validation_status"
    ],
)
print("=" * 92)


print(
    json.dumps(
        CELL9_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Cell 8 infrastructure was valid, but its synthetic "
    "battle-state semantics were not."
)

print(
    "The deployment PPO policy has NOT been established "
    "as the cause of the failed battles."
)

print(
    "Full 240-battle campaign remains blocked."
)

print()
print(
    "NEXT STEP: Cell 10 — construct schema-correct controlled "
    "battle states directly from the authoritative legal-move contract."
)


# In[10]:


# =============================================================================
# NOTEBOOK 63 — CELL 10
# SCHEMA-CORRECT CONTROLLED BATTLE-STATE REPAIR
# AUTHORITATIVE legal_moves.py CONTRACT
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 10")
print("SCHEMA-CORRECT CONTROLLED BATTLE-STATE REPAIR")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL9_NB63_STATUS["validation_status"]
    ==
    "PASS"
)

assert (
    CELL9_NB63_STATUS[
        "campaign_harness_semantic_issue"
    ]
    is True
)

assert (
    CELL9_NB63_STATUS[
        "full_campaign_authorized"
    ]
    is False
)


print()
print("[OK] Cell 9 forensic audit confirmed.")
print("[OK] Cell 8 failure classified as harness-semantic, not PPO failure.")
print("[OK] Real deployment runtime will NOT be modified.")
print("[OK] Full campaign remains blocked until repaired scenarios validate.")


# =============================================================================
# 1. AUTHORITATIVE legal_moves.py CONTRACT
# =============================================================================

LEGAL_MOVES_PATH_CELL10_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
    /
    "src"
    /
    "legal_moves.py"
)


assert LEGAL_MOVES_PATH_CELL10_NB63.is_file()


legal_source_cell10_nb63 = (
    LEGAL_MOVES_PATH_CELL10_NB63.read_text(
        encoding="utf-8"
    )
)


required_contract_tokens_cell10_nb63 = [
    'pokemon_state.card.get("attacks", [])',
    '"Move Name"',
    '"damage_numeric"',
    '"energy_cost"',
    '"Effect Explanation"',
]


missing_contract_tokens_cell10_nb63 = [
    token
    for token
    in required_contract_tokens_cell10_nb63
    if token not in legal_source_cell10_nb63
]


assert (
    not missing_contract_tokens_cell10_nb63
), (
    "Authoritative legal-move contract changed unexpectedly: "
    f"{missing_contract_tokens_cell10_nb63}"
)


print()
print("=" * 92)
print("1. AUTHORITATIVE ATTACK SCHEMA")
print("=" * 92)


AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63 = [
    "Move Name",
    "damage_numeric",
    "energy_cost",
    "Effect Explanation",
]


for key in AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63:

    print(
        "[REQUIRED]",
        key,
    )


print()
print(
    "[OK] Authoritative legal_moves.py contract confirmed."
)


# =============================================================================
# 2. DOCUMENT CELL 7 SCHEMA MISMATCH
# =============================================================================

CELL7_ATTACK_KEYS_CELL10_NB63 = sorted(
    {
        key
        for scenario
        in NB63_SCENARIOS
        for side
        in [
            "player",
            "opponent",
        ]
        for attack
        in scenario[
            side
        ][
            "attacks"
        ]
        for key
        in attack.keys()
    }
)


print()
print("=" * 92)
print("2. CELL 7 VS AUTHORITATIVE SCHEMA")
print("=" * 92)


print(
    "Cell 7 attack keys:",
    CELL7_ATTACK_KEYS_CELL10_NB63,
)


print(
    "Authoritative keys :",
    AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63,
)


assert (
    "name"
    in
    CELL7_ATTACK_KEYS_CELL10_NB63
)


assert (
    "damage"
    in
    CELL7_ATTACK_KEYS_CELL10_NB63
)


assert (
    "Move Name"
    not in
    CELL7_ATTACK_KEYS_CELL10_NB63
)


assert (
    "damage_numeric"
    not in
    CELL7_ATTACK_KEYS_CELL10_NB63
)


print()
print(
    "[CONFIRMED] Cell 7 used normalized output-move keys "
    "instead of saved-card attack keys."
)

print(
    "[CONFIRMED] This explains Cell 8's 100% Unknown Move / "
    "zero-damage pattern."
)


# =============================================================================
# 3. CONVERT ONE ATTACK TO AUTHORITATIVE CARD SCHEMA
# =============================================================================

def convert_attack_to_authoritative_cell10_nb63(
    attack: dict,
) -> dict:

    return {

        "Move Name":
            str(
                attack[
                    "name"
                ]
            ),

        "damage_numeric":
            float(
                attack[
                    "damage"
                ]
            ),

        "energy_cost":
            int(
                attack[
                    "energy_cost"
                ]
            ),

        "Effect Explanation":
            attack.get(
                "effect"
            ),
    }


# =============================================================================
# 4. BUILD SCHEMA-CORRECT SCENARIO SET
# =============================================================================

NB63_SCENARIOS_SCHEMA_CORRECT = []


for original_scenario in NB63_SCENARIOS:

    repaired_scenario = {

        "scenario_id":
            original_scenario[
                "scenario_id"
            ],

        "description":
            original_scenario[
                "description"
            ],

        "schema_version":
            "AUTHORITATIVE_LEGAL_MOVES_V1",

        "player": {

            "name":
                original_scenario[
                    "player"
                ][
                    "name"
                ],

            "hp":
                float(
                    original_scenario[
                        "player"
                    ][
                        "hp"
                    ]
                ),

            "energy":
                int(
                    original_scenario[
                        "player"
                    ][
                        "energy"
                    ]
                ),

            "attacks": [
                convert_attack_to_authoritative_cell10_nb63(
                    attack
                )

                for attack
                in original_scenario[
                    "player"
                ][
                    "attacks"
                ]
            ],
        },

        "opponent": {

            "name":
                original_scenario[
                    "opponent"
                ][
                    "name"
                ],

            "hp":
                float(
                    original_scenario[
                        "opponent"
                    ][
                        "hp"
                    ]
                ),

            "energy":
                int(
                    original_scenario[
                        "opponent"
                    ][
                        "energy"
                    ]
                ),

            "attacks": [
                convert_attack_to_authoritative_cell10_nb63(
                    attack
                )

                for attack
                in original_scenario[
                    "opponent"
                ][
                    "attacks"
                ]
            ],
        },
    }


    NB63_SCENARIOS_SCHEMA_CORRECT.append(
        repaired_scenario
    )


assert (
    len(
        NB63_SCENARIOS_SCHEMA_CORRECT
    )
    ==
    len(
        NB63_SCENARIOS
    )
)


print()
print("=" * 92)
print("3. SCHEMA-CORRECT SCENARIOS")
print("=" * 92)


for scenario in NB63_SCENARIOS_SCHEMA_CORRECT:

    print()
    print(
        scenario[
            "scenario_id"
        ]
    )


    print(
        "  Player attacks:"
    )


    for attack in scenario[
        "player"
    ][
        "attacks"
    ]:

        print(
            "   ",
            attack,
        )


    print(
        "  Opponent attacks:"
    )


    for attack in scenario[
        "opponent"
    ][
        "attacks"
    ]:

        print(
            "   ",
            attack,
        )


# =============================================================================
# 5. STATIC SCHEMA VALIDATION
# =============================================================================

schema_errors_cell10_nb63 = []


for scenario in NB63_SCENARIOS_SCHEMA_CORRECT:

    for side in [
        "player",
        "opponent",
    ]:

        pokemon_record = scenario[
            side
        ]


        for attack_index, attack in enumerate(
            pokemon_record[
                "attacks"
            ]
        ):

            missing_keys = [
                key
                for key
                in AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63
                if key not in attack
            ]


            if missing_keys:

                schema_errors_cell10_nb63.append(
                    {
                        "scenario_id":
                            scenario[
                                "scenario_id"
                            ],

                        "side":
                            side,

                        "attack_index":
                            attack_index,

                        "missing_keys":
                            missing_keys,
                    }
                )


assert (
    not schema_errors_cell10_nb63
), schema_errors_cell10_nb63


print()
print(
    "[OK] Every repaired attack contains all authoritative keys."
)


# =============================================================================
# 6. BUILD TEMPORARY RUNTIME FOR TRUE get_legal_moves() PROBE
# =============================================================================

REAL_FINAL_AGENT_CELL10_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
)


temp_root_cell10_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell10_"
    )
).resolve()


temp_final_agent_cell10_nb63 = (
    temp_root_cell10_nb63
    /
    "final_agent"
)


shutil.copytree(
    REAL_FINAL_AGENT_CELL10_NB63,
    temp_final_agent_cell10_nb63,
)


for package_dir in [
    temp_final_agent_cell10_nb63
    /
    "src",

    temp_final_agent_cell10_nb63
    /
    "src"
    /
    "agents",
]:

    if package_dir.is_dir():

        init_path = (
            package_dir
            /
            "__init__.py"
        )


        if not init_path.exists():

            init_path.write_text(
                "# Temporary NB63 Cell 10 package marker.\n",
                encoding="utf-8",
            )


scenario_payload_path_cell10_nb63 = (
    temp_root_cell10_nb63
    /
    "schema_correct_scenarios.json"
)


scenario_payload_path_cell10_nb63.write_text(
    json.dumps(
        NB63_SCENARIOS_SCHEMA_CORRECT,
        indent=2,
    ),
    encoding="utf-8",
)


# =============================================================================
# 7. AUTHORITATIVE LEGAL-MOVE PROBE DRIVER
# =============================================================================

driver_path_cell10_nb63 = (
    temp_root_cell10_nb63
    /
    "cell10_legal_move_probe.py"
)


driver_source_cell10_nb63 = r'''
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


RESULT = {

    "imports_pass":
        False,

    "probe_count":
        0,

    "probe_pass_count":
        0,

    "probe_failure_count":
        0,

    "unknown_move_count":
        0,

    "zero_damage_only_probe_count":
        0,

    "records":
        [],

    "fatal_exception_type":
        None,

    "fatal_exception_message":
        None,

    "fatal_traceback":
        None,
}


try:

    final_agent_root = Path(
        sys.argv[1]
    ).resolve()


    scenarios_path = Path(
        sys.argv[2]
    ).resolve()


    sys.path.insert(
        0,
        str(
            final_agent_root
        ),
    )


    from src.battle_state import (
        PokemonState,
        PlayerState,
        BattleState,
    )


    from src.legal_moves import (
        get_current_legal_moves,
    )


    RESULT[
        "imports_pass"
    ] = True


    scenarios = json.loads(
        scenarios_path.read_text(
            encoding="utf-8"
        )
    )


    def make_pokemon(
        record,
    ):

        return PokemonState(

            card={
                "name":
                    record[
                        "name"
                    ],

                "attacks":
                    record[
                        "attacks"
                    ],
            },

            current_hp=float(
                record[
                    "hp"
                ]
            ),

            attached_energy=int(
                record[
                    "energy"
                ]
            ),

            status=None,

            damage=0.0,

            is_active=True,
        )


    for scenario in scenarios:

        for swap_sides in [
            False,
            True,
        ]:

            if swap_sides:

                player_record = (
                    scenario[
                        "opponent"
                    ]
                )

                opponent_record = (
                    scenario[
                        "player"
                    ]
                )

                side_mode = (
                    "SWAPPED"
                )

            else:

                player_record = (
                    scenario[
                        "player"
                    ]
                )

                opponent_record = (
                    scenario[
                        "opponent"
                    ]
                )

                side_mode = (
                    "NORMAL"
                )


            for current_player in [
                "Player",
                "Opponent",
            ]:

                RESULT[
                    "probe_count"
                ] += 1


                probe_record = {

                    "scenario_id":
                        scenario[
                            "scenario_id"
                        ],

                    "side_mode":
                        side_mode,

                    "current_player":
                        current_player,

                    "legal_moves":
                        None,

                    "legal_move_count":
                        0,

                    "unknown_move_count":
                        0,

                    "positive_damage_move_count":
                        0,

                    "zero_damage_move_count":
                        0,

                    "probe_pass":
                        False,

                    "error":
                        None,
                }


                try:

                    state = BattleState(

                        player=PlayerState(
                            active=make_pokemon(
                                player_record
                            ),
                            bench=[],
                            prize_cards_remaining=6,
                            hand_size=7,
                        ),

                        opponent=PlayerState(
                            active=make_pokemon(
                                opponent_record
                            ),
                            bench=[],
                            prize_cards_remaining=6,
                            hand_size=7,
                        ),

                        turn_number=1,

                        current_player=current_player,
                    )


                    legal_moves = (
                        get_current_legal_moves(
                            state
                        )
                    )


                    probe_record[
                        "legal_moves"
                    ] = legal_moves


                    probe_record[
                        "legal_move_count"
                    ] = len(
                        legal_moves
                    )


                    for move in legal_moves:

                        move_name = str(
                            move.get(
                                "name",
                                ""
                            )
                            or
                            ""
                        )


                        try:

                            damage = float(
                                move.get(
                                    "damage",
                                    0.0,
                                )
                                or
                                0.0
                            )

                        except Exception:

                            damage = 0.0


                        if (
                            move_name.strip().lower()
                            ==
                            "unknown move"
                        ):

                            probe_record[
                                "unknown_move_count"
                            ] += 1


                        if damage > 0:

                            probe_record[
                                "positive_damage_move_count"
                            ] += 1

                        else:

                            probe_record[
                                "zero_damage_move_count"
                            ] += 1


                    probe_record[
                        "probe_pass"
                    ] = (

                        len(
                            legal_moves
                        )
                        >
                        0

                        and

                        probe_record[
                            "unknown_move_count"
                        ]
                        ==
                        0

                        and

                        probe_record[
                            "positive_damage_move_count"
                        ]
                        >
                        0
                    )


                    RESULT[
                        "unknown_move_count"
                    ] += probe_record[
                        "unknown_move_count"
                    ]


                    if (
                        probe_record[
                            "positive_damage_move_count"
                        ]
                        ==
                        0
                    ):

                        RESULT[
                            "zero_damage_only_probe_count"
                        ] += 1


                    if probe_record[
                        "probe_pass"
                    ]:

                        RESULT[
                            "probe_pass_count"
                        ] += 1

                    else:

                        RESULT[
                            "probe_failure_count"
                        ] += 1


                except Exception as probe_exc:

                    probe_record[
                        "error"
                    ] = (
                        f"{type(probe_exc).__name__}: "
                        f"{probe_exc}"
                    )


                    RESULT[
                        "probe_failure_count"
                    ] += 1


                RESULT[
                    "records"
                ].append(
                    probe_record
                )


except Exception as fatal_exc:

    RESULT[
        "fatal_exception_type"
    ] = type(
        fatal_exc
    ).__name__


    RESULT[
        "fatal_exception_message"
    ] = str(
        fatal_exc
    )


    RESULT[
        "fatal_traceback"
    ] = traceback.format_exc()


print(
    "NB63_CELL10_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell10_nb63.write_text(
    textwrap.dedent(
        driver_source_cell10_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell10_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell10_nb63
    ),
    "exec",
)


# =============================================================================
# 8. EXECUTE AUTHORITATIVE LEGAL-MOVE PROBES
# =============================================================================

print()
print("=" * 92)
print("4. AUTHORITATIVE get_current_legal_moves() PROBES")
print("=" * 92)


environment_cell10_nb63 = (
    os.environ.copy()
)


environment_cell10_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell10_nb63 = subprocess.run(
    [
        sys.executable,

        str(
            driver_path_cell10_nb63
        ),

        str(
            temp_final_agent_cell10_nb63
        ),

        str(
            scenario_payload_path_cell10_nb63
        ),
    ],

    cwd=str(
        temp_root_cell10_nb63
    ),

    env=environment_cell10_nb63,

    capture_output=True,

    text=True,

    timeout=180,
)


print(
    "Return code:",
    process_cell10_nb63.returncode,
)


print()
print("--- STDERR ---")

print(
    process_cell10_nb63.stderr
)


assert (
    process_cell10_nb63.returncode
    ==
    0
), (
    "Cell 10 legal-move probe subprocess failed.\n"
    f"{process_cell10_nb63.stderr}"
)


result_prefix_cell10_nb63 = (
    "NB63_CELL10_RESULT_JSON="
)


result_line_cell10_nb63 = None


for line in (
    process_cell10_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell10_nb63
    ):

        result_line_cell10_nb63 = (
            line
        )


assert (
    result_line_cell10_nb63
    is not None
)


schema_probe_result_cell10_nb63 = json.loads(
    result_line_cell10_nb63[
        len(
            result_prefix_cell10_nb63
        ):
    ]
)


print(
    json.dumps(
        schema_probe_result_cell10_nb63,
        indent=2,
    )
)


# =============================================================================
# 9. REQUIRED REPAIR VALIDATION
# =============================================================================

assert (
    schema_probe_result_cell10_nb63[
        "imports_pass"
    ]
    is True
)


assert (
    schema_probe_result_cell10_nb63[
        "fatal_exception_type"
    ]
    is None
), (
    schema_probe_result_cell10_nb63.get(
        "fatal_traceback"
    )
)


expected_probe_count_cell10_nb63 = (
    len(
        NB63_SCENARIOS_SCHEMA_CORRECT
    )
    *
    2
    *
    2
)


assert (
    schema_probe_result_cell10_nb63[
        "probe_count"
    ]
    ==
    expected_probe_count_cell10_nb63
)


assert (
    schema_probe_result_cell10_nb63[
        "probe_failure_count"
    ]
    ==
    0
), (
    "One or more repaired scenarios still fail "
    "the authoritative legal-move contract."
)


assert (
    schema_probe_result_cell10_nb63[
        "unknown_move_count"
    ]
    ==
    0
), (
    "Unknown Move remains after schema repair."
)


assert (
    schema_probe_result_cell10_nb63[
        "zero_damage_only_probe_count"
    ]
    ==
    0
), (
    "At least one repaired state still has no "
    "positive-damage legal move."
)


print()
print(
    "[OK] All",
    expected_probe_count_cell10_nb63,
    "scenario/side/current-player probes passed."
)

print(
    "[OK] Unknown Move count is zero."
)

print(
    "[OK] Every probe exposes at least one positive-damage legal move."
)


# =============================================================================
# 10. BUILD CORRECTED CAMPAIGN MATRIX
# =============================================================================

NB63_CORRECTED_CAMPAIGN_MATRIX = []


corrected_battle_counter_cell10_nb63 = 0


for baseline in NB63_BASELINES:

    for scenario in (
        NB63_SCENARIOS_SCHEMA_CORRECT
    ):

        for side_mode in NB63_SIDE_MODES:

            for seed in NB63_CAMPAIGN_SEEDS:

                corrected_battle_counter_cell10_nb63 += 1


                NB63_CORRECTED_CAMPAIGN_MATRIX.append(
                    {

                        "battle_id":
                            (
                                "NB63C_"
                                f"{corrected_battle_counter_cell10_nb63:04d}"
                            ),

                        "agent_variant":
                            baseline[
                                "baseline_id"
                            ],

                        "agent_type":
                            baseline[
                                "agent_type"
                            ],

                        "scenario_id":
                            scenario[
                                "scenario_id"
                            ],

                        "side_mode":
                            side_mode[
                                "side_mode"
                            ],

                        "swap_sides":
                            side_mode[
                                "swap_sides"
                            ],

                        "seed":
                            seed,

                        "search_depth":
                            NB63_EXECUTION_SETTINGS[
                                "search_depth"
                            ],

                        "max_turns":
                            NB63_EXECUTION_SETTINGS[
                                "max_turns"
                            ],

                        "checkpoint_sha256":
                            baseline[
                                "checkpoint_sha256"
                            ],
                    }
                )


assert (
    len(
        NB63_CORRECTED_CAMPAIGN_MATRIX
    )
    ==
    240
)


# =============================================================================
# 11. CORRECTED CAMPAIGN CONFIGURATION HASH
# =============================================================================

NB63_CORRECTED_CAMPAIGN_CONFIGURATION = {

    "supersedes_configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "repair_reason":
        (
            "CELL7_ATTACK_SCHEMA_USED_NORMALIZED_MOVE_KEYS_"
            "INSTEAD_OF_AUTHORITATIVE_CARD_ATTACK_KEYS"
        ),

    "authoritative_attack_keys":
        AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63,

    "scenarios":
        NB63_SCENARIOS_SCHEMA_CORRECT,

    "seeds":
        NB63_CAMPAIGN_SEEDS,

    "side_modes":
        NB63_SIDE_MODES,

    "baselines":
        NB63_BASELINES,

    "execution_settings":
        NB63_EXECUTION_SETTINGS,
}


corrected_config_json_cell10_nb63 = (
    json.dumps(
        NB63_CORRECTED_CAMPAIGN_CONFIGURATION,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )
)


NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256 = (
    hashlib.sha256(
        corrected_config_json_cell10_nb63.encode(
            "utf-8"
        )
    ).hexdigest()
)


print()
print("=" * 92)
print("5. CORRECTED CAMPAIGN CONFIGURATION")
print("=" * 92)


print(
    "Original config SHA256 :",
    NB63_CAMPAIGN_CONFIG_SHA256,
)


print(
    "Corrected config SHA256:",
    NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,
)


print(
    "Corrected campaign rows:",
    len(
        NB63_CORRECTED_CAMPAIGN_MATRIX
    ),
)


# =============================================================================
# 12. PERSIST CORRECTED SCENARIOS / CONFIGURATION
# =============================================================================

corrected_scenarios_artifact_cell10_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell10_schema_correct_scenarios.json"
)


corrected_scenarios_output_cell10_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell10_schema_correct_scenarios.json"
)


corrected_matrix_artifact_cell10_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell10_corrected_campaign_matrix.json"
)


corrected_matrix_output_cell10_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell10_corrected_campaign_matrix.json"
)


corrected_audit_artifact_cell10_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell10_schema_repair_audit.json"
)


corrected_audit_output_cell10_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell10_schema_repair_audit.json"
)


for path in [
    corrected_scenarios_artifact_cell10_nb63,
    corrected_scenarios_output_cell10_nb63,
]:

    path.write_text(
        json.dumps(
            NB63_SCENARIOS_SCHEMA_CORRECT,
            indent=2,
        ),
        encoding="utf-8",
    )


for path in [
    corrected_matrix_artifact_cell10_nb63,
    corrected_matrix_output_cell10_nb63,
]:

    path.write_text(
        json.dumps(
            NB63_CORRECTED_CAMPAIGN_MATRIX,
            indent=2,
        ),
        encoding="utf-8",
    )


repair_audit_payload_cell10_nb63 = {

    "notebook":
        63,

    "cell":
        10,

    "root_cause":
        (
            "SYNTHETIC_ATTACK_FIELD_NAMES_DID_NOT_MATCH_"
            "AUTHORITATIVE_LEGAL_MOVE_CARD_SCHEMA"
        ),

    "incorrect_keys":
        CELL7_ATTACK_KEYS_CELL10_NB63,

    "authoritative_keys":
        AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63,

    "original_configuration_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "corrected_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "probe_result":
        schema_probe_result_cell10_nb63,

    "full_campaign_executed":
        False,
}


for path in [
    corrected_audit_artifact_cell10_nb63,
    corrected_audit_output_cell10_nb63,
]:

    path.write_text(
        json.dumps(
            repair_audit_payload_cell10_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("6. EVIDENCE PERSISTENCE")
print("=" * 92)


print(
    "[OK]",
    corrected_scenarios_artifact_cell10_nb63,
)

print(
    "[OK]",
    corrected_matrix_artifact_cell10_nb63,
)

print(
    "[OK]",
    corrected_audit_artifact_cell10_nb63,
)


# =============================================================================
# 13. TEMPORARY CLEANUP
# =============================================================================

shutil.rmtree(
    temp_root_cell10_nb63,
    ignore_errors=True,
)


temporary_deleted_cell10_nb63 = (
    not temp_root_cell10_nb63.exists()
)


assert (
    temporary_deleted_cell10_nb63
    is True
)


print()
print(
    "[OK] Temporary schema-validation runtime deleted."
)


# =============================================================================
# 14. CERTIFIED ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("7. CERTIFIED ASSET INTEGRITY")
print("=" * 92)


print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training/retraining performed."
)

print(
    "[OK] No real submission/runtime source modified."
)


# =============================================================================
# 15. FINAL STATUS
# =============================================================================

CELL10_NB63_STATUS = {

    "cell9_pass":
        True,

    "root_cause_confirmed":
        (
            "INCORRECT_SYNTHETIC_ATTACK_FIELD_NAMES"
        ),

    "incorrect_attack_keys":
        CELL7_ATTACK_KEYS_CELL10_NB63,

    "authoritative_attack_keys":
        AUTHORITATIVE_ATTACK_KEYS_CELL10_NB63,

    "schema_correct_scenarios_created":
        True,

    "scenario_count":
        len(
            NB63_SCENARIOS_SCHEMA_CORRECT
        ),

    "legal_move_probe_count":
        schema_probe_result_cell10_nb63[
            "probe_count"
        ],

    "legal_move_probe_pass_count":
        schema_probe_result_cell10_nb63[
            "probe_pass_count"
        ],

    "legal_move_probe_failure_count":
        schema_probe_result_cell10_nb63[
            "probe_failure_count"
        ],

    "unknown_move_count":
        schema_probe_result_cell10_nb63[
            "unknown_move_count"
        ],

    "zero_damage_only_probe_count":
        schema_probe_result_cell10_nb63[
            "zero_damage_only_probe_count"
        ],

    "original_campaign_config_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "corrected_campaign_config_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "corrected_campaign_rows":
        len(
            NB63_CORRECTED_CAMPAIGN_MATRIX
        ),

    "corrected_pilot_authorized":
        True,

    "full_240_battle_campaign_authorized":
        False,

    "simulation_executed":
        False,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 10 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL10_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Schema-correct campaign scenarios validated against "
    "the authoritative deployment legal-move generator."
)

print(
    "Cell 8 remains diagnostic history and is superseded "
    "for performance evaluation."
)

print(
    "Corrected 24-battle pilot is now authorized."
)

print(
    "Full 240-battle campaign remains blocked until "
    "the corrected pilot passes."
)

print()
print(
    "NEXT STEP: Cell 11 — rerun the 24-battle pilot "
    "using the schema-correct campaign."
)


# In[11]:


# =============================================================================
# NOTEBOOK 63 — CELL 11
# CORRECTED 24-BATTLE PILOT
# SCHEMA-CORRECT SCENARIOS + THREE AGENT VARIANTS
# =============================================================================

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 11")
print("CORRECTED 24-BATTLE PILOT")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL10_NB63_STATUS["validation_status"]
    ==
    "PASS"
)

assert (
    CELL10_NB63_STATUS[
        "corrected_pilot_authorized"
    ]
    is True
)

assert (
    CELL10_NB63_STATUS[
        "full_240_battle_campaign_authorized"
    ]
    is False
)

assert (
    CELL10_NB63_STATUS[
        "unknown_move_count"
    ]
    ==
    0
)

assert (
    CELL10_NB63_STATUS[
        "zero_damage_only_probe_count"
    ]
    ==
    0
)


print()
print("[OK] Cell 10 PASS confirmed.")
print("[OK] Schema-correct scenarios validated.")
print("[OK] Corrected 24-battle pilot authorized.")
print("[OK] Full campaign still blocked pending this pilot.")


# =============================================================================
# 1. CORRECTED PILOT MATRIX
# =============================================================================

CORRECTED_PILOT_SEED_CELL11_NB63 = 63001


corrected_pilot_matrix_cell11_nb63 = [
    record
    for record
    in NB63_CORRECTED_CAMPAIGN_MATRIX
    if (
        record["seed"]
        ==
        CORRECTED_PILOT_SEED_CELL11_NB63
    )
]


assert (
    len(
        corrected_pilot_matrix_cell11_nb63
    )
    ==
    24
)


print()
print("=" * 92)
print("1. CORRECTED PILOT MATRIX")
print("=" * 92)

print(
    "Pilot seed:",
    CORRECTED_PILOT_SEED_CELL11_NB63,
)

print(
    "Pilot battles:",
    len(
        corrected_pilot_matrix_cell11_nb63
    ),
)

print(
    "Corrected config SHA256:",
    NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,
)


# =============================================================================
# 2. CERTIFIED ASSET FREEZE
# =============================================================================

main_hash_before_cell11_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

rf_hash_before_cell11_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)

checkpoint_hash_before_cell11_nb63 = (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
)


assert (
    main_hash_before_cell11_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    rf_hash_before_cell11_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    checkpoint_hash_before_cell11_nb63
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("2. PRE-PILOT ASSET FREEZE")
print("=" * 92)

print(
    "main.py:",
    main_hash_before_cell11_nb63,
)

print(
    "RF model:",
    rf_hash_before_cell11_nb63,
)

print(
    "PPO checkpoint:",
    checkpoint_hash_before_cell11_nb63,
)


print()
print("[OK] Authoritative assets frozen.")


# =============================================================================
# 3. BUILD CORRECTED PILOT PAYLOAD
# =============================================================================

corrected_pilot_payload_cell11_nb63 = {

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "checkpoint_sha256":
        DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,

    "scenarios":
        NB63_SCENARIOS_SCHEMA_CORRECT,

    "pilot_matrix":
        corrected_pilot_matrix_cell11_nb63,
}


# =============================================================================
# 4. TEMPORARY DEPLOYMENT COPY
# =============================================================================

REAL_FINAL_AGENT_CELL11_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
)


temp_root_cell11_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell11_"
    )
).resolve()


temp_final_agent_cell11_nb63 = (
    temp_root_cell11_nb63
    /
    "final_agent"
)


shutil.copytree(
    REAL_FINAL_AGENT_CELL11_NB63,
    temp_final_agent_cell11_nb63,
)


for package_dir in [
    temp_final_agent_cell11_nb63
    /
    "src",

    temp_final_agent_cell11_nb63
    /
    "src"
    /
    "agents",
]:

    if package_dir.is_dir():

        init_path = (
            package_dir
            /
            "__init__.py"
        )

        if not init_path.exists():

            init_path.write_text(
                "# Temporary NB63 Cell 11 package marker.\n",
                encoding="utf-8",
            )


payload_path_cell11_nb63 = (
    temp_root_cell11_nb63
    /
    "corrected_pilot_payload.json"
)


payload_path_cell11_nb63.write_text(
    json.dumps(
        corrected_pilot_payload_cell11_nb63,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("3. TEMPORARY CORRECTED PILOT RUNTIME")
print("=" * 92)

print(
    "Temporary root:",
    temp_root_cell11_nb63,
)

print()
print(
    "[OK] Pilot will run only against temporary deployment copy."
)


# =============================================================================
# 5. CORRECTED PILOT DRIVER
# =============================================================================

driver_path_cell11_nb63 = (
    temp_root_cell11_nb63
    /
    "cell11_corrected_pilot_driver.py"
)


driver_source_cell11_nb63 = r'''
from __future__ import annotations

import json
import random
import statistics
import sys
import traceback
from pathlib import Path


RESULT = {

    "imports_pass":
        False,

    "policy_constructed":
        False,

    "requested_battles":
        0,

    "completed_battles":
        0,

    "runtime_failures":
        0,

    "records":
        [],

    "fatal_exception_type":
        None,

    "fatal_exception_message":
        None,

    "fatal_traceback":
        None,
}


try:

    final_agent_root = Path(
        sys.argv[1]
    ).resolve()


    payload_path = Path(
        sys.argv[2]
    ).resolve()


    sys.path.insert(
        0,
        str(
            final_agent_root
        ),
    )


    from src.battle_state import (
        PokemonState,
        PlayerState,
        BattleState,
    )


    from src.agent_decision import (
        AgentDecision,
    )


    from src.legal_moves import (
        get_current_legal_moves,
    )


    from src.agents.final_ppo_agent import (
        FinalPPOBattleAgent,
    )


    from src.battle_simulation import (
        simulate_ai_battle,
    )


    RESULT[
        "imports_pass"
    ] = True


    payload = json.loads(
        payload_path.read_text(
            encoding="utf-8"
        )
    )


    matrix = payload[
        "pilot_matrix"
    ]


    scenarios = {
        scenario[
            "scenario_id"
        ]:
            scenario

        for scenario
        in payload[
            "scenarios"
        ]
    }


    RESULT[
        "requested_battles"
    ] = len(
        matrix
    )


    # =========================================================================
    # Reproducibility
    # =========================================================================

    def set_seed(
        seed,
    ):

        random.seed(
            seed
        )


        try:

            import numpy as np

            np.random.seed(
                int(seed)
                %
                (2 ** 32 - 1)
            )

        except Exception:

            pass


        try:

            import torch

            torch.manual_seed(
                int(seed)
            )

        except Exception:

            pass


    # =========================================================================
    # State builder
    # =========================================================================

    def make_pokemon(
        record,
    ):

        return PokemonState(

            card={
                "name":
                    record[
                        "name"
                    ],

                "attacks":
                    record[
                        "attacks"
                    ],
            },

            current_hp=float(
                record[
                    "hp"
                ]
            ),

            attached_energy=int(
                record[
                    "energy"
                ]
            ),

            status=None,

            damage=0.0,

            is_active=True,
        )


    def build_state(
        scenario,
        swap_sides,
    ):

        player_record = (
            scenario[
                "player"
            ]
        )

        opponent_record = (
            scenario[
                "opponent"
            ]
        )


        if swap_sides:

            player_record, opponent_record = (
                opponent_record,
                player_record,
            )


        return BattleState(

            player=PlayerState(
                active=make_pokemon(
                    player_record
                ),
                bench=[],
                prize_cards_remaining=6,
                hand_size=7,
            ),

            opponent=PlayerState(
                active=make_pokemon(
                    opponent_record
                ),
                bench=[],
                prize_cards_remaining=6,
                hand_size=7,
            ),

            turn_number=1,

            current_player="Player",
        )


    # =========================================================================
    # Baselines
    # =========================================================================

    class FirstLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            move = (
                legal_moves[0]
                if legal_moves
                else {
                    "name":
                        "Pass",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        "No legal move.",
                }
            )


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=max(
                    1,
                    len(
                        legal_moves
                    ),
                ),

                principal_variation=[
                    move
                ],
            )


    class MaxDamageLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            if legal_moves:

                move = max(
                    legal_moves,
                    key=lambda item: float(
                        item.get(
                            "damage",
                            0.0,
                        )
                        or
                        0.0
                    ),
                )

            else:

                move = {
                    "name":
                        "Pass",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        "No legal move.",
                }


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=max(
                    1,
                    len(
                        legal_moves
                    ),
                ),

                principal_variation=[
                    move
                ],
            )


    # =========================================================================
    # Deployment PPO
    # =========================================================================

    checkpoint_paths = sorted(
        [
            path
            for path
            in (
                final_agent_root
                /
                "models"
            ).rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".pkl",
                    ".pickle",
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".joblib",
                }
            )
        ]
    )


    if not checkpoint_paths:

        raise RuntimeError(
            "Deployment checkpoint missing."
        )


    deployment_agent = (
        FinalPPOBattleAgent(
            checkpoint_path=str(
                checkpoint_paths[
                    0
                ]
            ),

            device="cpu",

            deterministic=True,
        )
    )


    RESULT[
        "policy_constructed"
    ] = True


    agents = {

        "DEPLOYMENT_POLICY":
            deployment_agent,

        "FIRST_LEGAL_MOVE":
            FirstLegalMoveAgent(),

        "MAX_DAMAGE_LEGAL_MOVE":
            MaxDamageLegalMoveAgent(),
    }


    # =========================================================================
    # Battle execution
    # =========================================================================

    for spec in matrix:

        row = {

            "battle_id":
                spec[
                    "battle_id"
                ],

            "agent_variant":
                spec[
                    "agent_variant"
                ],

            "scenario_id":
                spec[
                    "scenario_id"
                ],

            "side_mode":
                spec[
                    "side_mode"
                ],

            "seed":
                spec[
                    "seed"
                ],

            "winner":
                None,

            "normalized_outcome":
                None,

            "turn_count":
                None,

            "stop_reason":
                None,

            "player_final_hp":
                None,

            "opponent_final_hp":
                None,

            "total_moves":
                0,

            "unknown_move_count":
                0,

            "unknown_move_rate":
                0.0,

            "zero_damage_move_count":
                0,

            "zero_damage_move_rate":
                0.0,

            "positive_damage_move_count":
                0,

            "positive_damage_move_rate":
                0.0,

            "total_damage":
                0.0,

            "mean_damage_per_move":
                0.0,

            "policy_score_mean":
                None,

            "policy_score_min":
                None,

            "policy_score_max":
                None,

            "runtime_exception":
                False,

            "runtime_exception_type":
                None,

            "runtime_exception_message":
                None,

            "checkpoint_sha256":
                spec.get(
                    "checkpoint_sha256"
                ),
        }


        try:

            set_seed(
                int(
                    spec[
                        "seed"
                    ]
                )
            )


            state = build_state(

                scenarios[
                    spec[
                        "scenario_id"
                    ]
                ],

                bool(
                    spec[
                        "swap_sides"
                    ]
                ),
            )


            agent = agents[
                spec[
                    "agent_variant"
                ]
            ]


            simulation = (
                simulate_ai_battle(

                    initial_state=state,

                    agent=agent,

                    search_depth=int(
                        spec[
                            "search_depth"
                        ]
                    ),

                    max_turns=int(
                        spec[
                            "max_turns"
                        ]
                    ),

                    verbose=False,
                )
            )


            winner = getattr(
                simulation,
                "winner",
                None,
            )


            turn_count = getattr(
                simulation,
                "turn_count",
                None,
            )


            stop_reason = getattr(
                simulation,
                "stop_reason",
                None,
            )


            turns = list(
                getattr(
                    simulation,
                    "turns",
                    []
                )
                or
                []
            )


            final_state = getattr(
                simulation,
                "final_state",
                None,
            )


            row[
                "winner"
            ] = winner


            row[
                "turn_count"
            ] = turn_count


            row[
                "stop_reason"
            ] = stop_reason


            # -------------------------------------------------------------
            # Outcome normalization
            # -------------------------------------------------------------

            if winner == "Player":

                row[
                    "normalized_outcome"
                ] = "PLAYER_WIN"


            elif winner == "Opponent":

                row[
                    "normalized_outcome"
                ] = "OPPONENT_WIN"


            elif (
                turn_count
                is not None
                and
                int(
                    turn_count
                )
                >=
                int(
                    spec[
                        "max_turns"
                    ]
                )
            ):

                row[
                    "normalized_outcome"
                ] = "TIMEOUT_DRAW"


            else:

                row[
                    "normalized_outcome"
                ] = "NO_WINNER"


            # -------------------------------------------------------------
            # Final HP
            # -------------------------------------------------------------

            if final_state is not None:

                try:

                    row[
                        "player_final_hp"
                    ] = float(
                        final_state
                        .player
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


                try:

                    row[
                        "opponent_final_hp"
                    ] = float(
                        final_state
                        .opponent
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


            # -------------------------------------------------------------
            # Turn metrics
            # -------------------------------------------------------------

            row[
                "total_moves"
            ] = len(
                turns
            )


            damages = []
            scores = []


            for turn in turns:

                move_name = str(
                    getattr(
                        turn,
                        "move_name",
                        "",
                    )
                    or
                    ""
                )


                if (
                    move_name.strip().lower()
                    ==
                    "unknown move"
                ):

                    row[
                        "unknown_move_count"
                    ] += 1


                try:

                    damage = float(
                        getattr(
                            turn,
                            "damage",
                            0.0,
                        )
                        or
                        0.0
                    )

                except Exception:

                    damage = 0.0


                damages.append(
                    damage
                )


                if damage > 0:

                    row[
                        "positive_damage_move_count"
                    ] += 1

                else:

                    row[
                        "zero_damage_move_count"
                    ] += 1


                raw_score = getattr(
                    turn,
                    "search_score",
                    None,
                )


                if raw_score is not None:

                    try:

                        scores.append(
                            float(
                                raw_score
                            )
                        )

                    except Exception:

                        pass


            total_moves = row[
                "total_moves"
            ]


            if total_moves > 0:

                row[
                    "unknown_move_rate"
                ] = (
                    row[
                        "unknown_move_count"
                    ]
                    /
                    total_moves
                )


                row[
                    "zero_damage_move_rate"
                ] = (
                    row[
                        "zero_damage_move_count"
                    ]
                    /
                    total_moves
                )


                row[
                    "positive_damage_move_rate"
                ] = (
                    row[
                        "positive_damage_move_count"
                    ]
                    /
                    total_moves
                )


            row[
                "total_damage"
            ] = float(
                sum(
                    damages
                )
            )


            if total_moves > 0:

                row[
                    "mean_damage_per_move"
                ] = (
                    row[
                        "total_damage"
                    ]
                    /
                    total_moves
                )


            if scores:

                row[
                    "policy_score_mean"
                ] = float(
                    statistics.mean(
                        scores
                    )
                )


                row[
                    "policy_score_min"
                ] = float(
                    min(
                        scores
                    )
                )


                row[
                    "policy_score_max"
                ] = float(
                    max(
                        scores
                    )
                )


            RESULT[
                "completed_battles"
            ] += 1


        except Exception as battle_exc:

            row[
                "runtime_exception"
            ] = True


            row[
                "runtime_exception_type"
            ] = type(
                battle_exc
            ).__name__


            row[
                "runtime_exception_message"
            ] = str(
                battle_exc
            )


            row[
                "normalized_outcome"
            ] = "RUNTIME_FAILURE"


            row[
                "traceback"
            ] = traceback.format_exc()


            RESULT[
                "runtime_failures"
            ] += 1


        RESULT[
            "records"
        ].append(
            row
        )


except Exception as fatal_exc:

    RESULT[
        "fatal_exception_type"
    ] = type(
        fatal_exc
    ).__name__


    RESULT[
        "fatal_exception_message"
    ] = str(
        fatal_exc
    )


    RESULT[
        "fatal_traceback"
    ] = traceback.format_exc()


print(
    "NB63_CELL11_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell11_nb63.write_text(
    textwrap.dedent(
        driver_source_cell11_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell11_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell11_nb63
    ),
    "exec",
)


print()
print(
    "[OK] Corrected pilot driver created and compiled."
)


# =============================================================================
# 6. EXECUTE CORRECTED PILOT
# =============================================================================

print()
print("=" * 92)
print("4. EXECUTE CORRECTED 24-BATTLE PILOT")
print("=" * 92)


environment_cell11_nb63 = (
    os.environ.copy()
)


environment_cell11_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell11_nb63 = subprocess.run(
    [
        sys.executable,

        str(
            driver_path_cell11_nb63
        ),

        str(
            temp_final_agent_cell11_nb63
        ),

        str(
            payload_path_cell11_nb63
        ),
    ],

    cwd=str(
        temp_root_cell11_nb63
    ),

    env=environment_cell11_nb63,

    capture_output=True,

    text=True,

    timeout=600,
)


print(
    "Return code:",
    process_cell11_nb63.returncode,
)


print()
print("--- STDERR ---")

print(
    process_cell11_nb63.stderr
)


assert (
    process_cell11_nb63.returncode
    ==
    0
), (
    "Corrected pilot subprocess failed.\n"
    f"{process_cell11_nb63.stderr}"
)


# =============================================================================
# 7. PARSE RESULT
# =============================================================================

result_prefix_cell11_nb63 = (
    "NB63_CELL11_RESULT_JSON="
)


result_line_cell11_nb63 = None


for line in (
    process_cell11_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell11_nb63
    ):

        result_line_cell11_nb63 = line


assert (
    result_line_cell11_nb63
    is not None
)


corrected_pilot_result_cell11_nb63 = (
    json.loads(
        result_line_cell11_nb63[
            len(
                result_prefix_cell11_nb63
            ):
        ]
    )
)


print()
print("=" * 92)
print("5. CORRECTED PILOT EXECUTION SUMMARY")
print("=" * 92)


execution_summary_cell11_nb63 = {

    "imports_pass":
        corrected_pilot_result_cell11_nb63[
            "imports_pass"
        ],

    "policy_constructed":
        corrected_pilot_result_cell11_nb63[
            "policy_constructed"
        ],

    "requested_battles":
        corrected_pilot_result_cell11_nb63[
            "requested_battles"
        ],

    "completed_battles":
        corrected_pilot_result_cell11_nb63[
            "completed_battles"
        ],

    "runtime_failures":
        corrected_pilot_result_cell11_nb63[
            "runtime_failures"
        ],

    "fatal_exception_type":
        corrected_pilot_result_cell11_nb63[
            "fatal_exception_type"
        ],
}


print(
    json.dumps(
        execution_summary_cell11_nb63,
        indent=2,
    )
)


# =============================================================================
# 8. INFRASTRUCTURE ASSERTIONS
# =============================================================================

assert (
    corrected_pilot_result_cell11_nb63[
        "imports_pass"
    ]
    is True
)


assert (
    corrected_pilot_result_cell11_nb63[
        "policy_constructed"
    ]
    is True
)


assert (
    corrected_pilot_result_cell11_nb63[
        "requested_battles"
    ]
    ==
    24
)


assert (
    len(
        corrected_pilot_result_cell11_nb63[
            "records"
        ]
    )
    ==
    24
)


assert (
    corrected_pilot_result_cell11_nb63[
        "runtime_failures"
    ]
    ==
    0
)


assert (
    corrected_pilot_result_cell11_nb63[
        "fatal_exception_type"
    ]
    is None
), (
    corrected_pilot_result_cell11_nb63.get(
        "fatal_traceback"
    )
)


print()
print(
    "[OK] Corrected pilot infrastructure passed 24/24 battles."
)


# =============================================================================
# 9. AGGREGATE RESULTS BY AGENT
# =============================================================================

records_cell11_nb63 = (
    corrected_pilot_result_cell11_nb63[
        "records"
    ]
)


agent_summary_cell11_nb63 = {}


for baseline in NB63_BASELINES:

    agent_id = (
        baseline[
            "baseline_id"
        ]
    )


    rows = [
        row
        for row
        in records_cell11_nb63
        if (
            row[
                "agent_variant"
            ]
            ==
            agent_id
        )
    ]


    outcomes = {}


    for row in rows:

        outcome = (
            row[
                "normalized_outcome"
            ]
        )


        outcomes[
            outcome
        ] = (
            outcomes.get(
                outcome,
                0,
            )
            +
            1
        )


    total_moves = sum(
        int(
            row[
                "total_moves"
            ]
            or
            0
        )
        for row
        in rows
    )


    unknown_moves = sum(
        int(
            row[
                "unknown_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    zero_damage_moves = sum(
        int(
            row[
                "zero_damage_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    positive_damage_moves = sum(
        int(
            row[
                "positive_damage_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    total_damage = sum(
        float(
            row[
                "total_damage"
            ]
            or
            0.0
        )
        for row
        in rows
    )


    wins = outcomes.get(
        "PLAYER_WIN",
        0,
    )


    opponent_wins = outcomes.get(
        "OPPONENT_WIN",
        0,
    )


    timeouts = outcomes.get(
        "TIMEOUT_DRAW",
        0,
    )


    runtime_failures = outcomes.get(
        "RUNTIME_FAILURE",
        0,
    )


    decisive_games = (
        wins
        +
        opponent_wins
    )


    agent_summary_cell11_nb63[
        agent_id
    ] = {

        "battles":
            len(
                rows
            ),

        "outcomes":
            outcomes,

        "player_wins":
            wins,

        "opponent_wins":
            opponent_wins,

        "timeouts":
            timeouts,

        "runtime_failures":
            runtime_failures,

        "decisive_games":
            decisive_games,

        "player_win_rate_all":
            (
                wins
                /
                len(
                    rows
                )
                if rows
                else
                0.0
            ),

        "player_win_rate_decisive":
            (
                wins
                /
                decisive_games
                if decisive_games
                else
                None
            ),

        "timeout_rate":
            (
                timeouts
                /
                len(
                    rows
                )
                if rows
                else
                0.0
            ),

        "total_moves":
            total_moves,

        "unknown_move_count":
            unknown_moves,

        "unknown_move_rate":
            (
                unknown_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "zero_damage_move_count":
            zero_damage_moves,

        "zero_damage_move_rate":
            (
                zero_damage_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "positive_damage_move_count":
            positive_damage_moves,

        "positive_damage_move_rate":
            (
                positive_damage_moves
                /
                total_moves
                if total_moves
                else
                0.0
            ),

        "total_damage":
            total_damage,

        "mean_damage_per_move":
            (
                total_damage
                /
                total_moves
                if total_moves
                else
                0.0
            ),
    }


print()
print("=" * 92)
print("6. CORRECTED PILOT AGENT SUMMARY")
print("=" * 92)


print(
    json.dumps(
        agent_summary_cell11_nb63,
        indent=2,
    )
)


# =============================================================================
# 10. SEMANTIC VALIDITY CHECK
# =============================================================================

semantic_validity_cell11_nb63 = {}


for agent_id, summary in (
    agent_summary_cell11_nb63.items()
):

    semantic_validity_cell11_nb63[
        agent_id
    ] = {

        "unknown_move_rate_zero":
            (
                summary[
                    "unknown_move_rate"
                ]
                ==
                0.0
            ),

        "positive_damage_present":
            (
                summary[
                    "positive_damage_move_count"
                ]
                >
                0
            ),

        "total_damage_positive":
            (
                summary[
                    "total_damage"
                ]
                >
                0.0
            ),
    }


print()
print("=" * 92)
print("7. SEMANTIC VALIDITY")
print("=" * 92)


print(
    json.dumps(
        semantic_validity_cell11_nb63,
        indent=2,
    )
)


all_agents_semantically_valid_cell11_nb63 = all(

    record[
        "unknown_move_rate_zero"
    ]

    and

    record[
        "positive_damage_present"
    ]

    and

    record[
        "total_damage_positive"
    ]

    for record
    in semantic_validity_cell11_nb63.values()
)


assert (
    all_agents_semantically_valid_cell11_nb63
    is True
), (
    "Corrected pilot still has a semantic-mapping problem."
)


print()
print(
    "[OK] All three agent variants now execute real named attacks."
)

print(
    "[OK] All three agent variants produce positive damage."
)

print(
    "[OK] Unknown Move rate is zero across the corrected pilot."
)


# =============================================================================
# 11. FULL-CAMPAIGN AUTHORIZATION RULE
# =============================================================================

corrected_pilot_infrastructure_pass_cell11_nb63 = (
    corrected_pilot_result_cell11_nb63[
        "runtime_failures"
    ]
    ==
    0
)


full_campaign_authorized_cell11_nb63 = (

    corrected_pilot_infrastructure_pass_cell11_nb63

    and

    all_agents_semantically_valid_cell11_nb63
)


print()
print("=" * 92)
print("8. FULL-CAMPAIGN AUTHORIZATION")
print("=" * 92)


print(
    "Infrastructure pass:",
    corrected_pilot_infrastructure_pass_cell11_nb63,
)

print(
    "Semantic validity pass:",
    all_agents_semantically_valid_cell11_nb63,
)

print(
    "Full 240-battle campaign authorized:",
    full_campaign_authorized_cell11_nb63,
)


assert (
    full_campaign_authorized_cell11_nb63
    is True
)


# =============================================================================
# 12. PERSIST CORRECTED PILOT EVIDENCE
# =============================================================================

corrected_records_output_cell11_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell11_corrected_pilot_records.json"
)


corrected_records_artifact_cell11_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell11_corrected_pilot_records.json"
)


corrected_csv_output_cell11_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell11_corrected_pilot_records.csv"
)


corrected_summary_output_cell11_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell11_corrected_pilot_summary.json"
)


corrected_summary_artifact_cell11_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell11_corrected_pilot_summary.json"
)


for path in [
    corrected_records_output_cell11_nb63,
    corrected_records_artifact_cell11_nb63,
]:

    path.write_text(
        json.dumps(
            records_cell11_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


csv_fields_cell11_nb63 = sorted(
    {
        key
        for row
        in records_cell11_nb63
        for key
        in row.keys()
        if key
        !=
        "traceback"
    }
)


with corrected_csv_output_cell11_nb63.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=csv_fields_cell11_nb63,
    )

    writer.writeheader()


    for row in records_cell11_nb63:

        writer.writerow(
            {
                key:
                    row.get(
                        key
                    )

                for key
                in csv_fields_cell11_nb63
            }
        )


corrected_summary_payload_cell11_nb63 = {

    "notebook":
        63,

    "cell":
        11,

    "purpose":
        "SCHEMA_CORRECT_24_BATTLE_PILOT",

    "corrected_campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "pilot_seed":
        CORRECTED_PILOT_SEED_CELL11_NB63,

    "battle_count":
        24,

    "agent_summary":
        agent_summary_cell11_nb63,

    "semantic_validity":
        semantic_validity_cell11_nb63,

    "full_campaign_authorized":
        full_campaign_authorized_cell11_nb63,
}


for path in [
    corrected_summary_output_cell11_nb63,
    corrected_summary_artifact_cell11_nb63,
]:

    path.write_text(
        json.dumps(
            corrected_summary_payload_cell11_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("9. CORRECTED PILOT EVIDENCE PERSISTENCE")
print("=" * 92)


print(
    "[OK]",
    corrected_records_output_cell11_nb63,
)

print(
    "[OK]",
    corrected_csv_output_cell11_nb63,
)

print(
    "[OK]",
    corrected_summary_output_cell11_nb63,
)


# =============================================================================
# 13. TEMPORARY CLEANUP
# =============================================================================

shutil.rmtree(
    temp_root_cell11_nb63,
    ignore_errors=True,
)


temporary_deleted_cell11_nb63 = (
    not temp_root_cell11_nb63.exists()
)


assert (
    temporary_deleted_cell11_nb63
    is True
)


print()
print(
    "[OK] Temporary corrected-pilot runtime deleted."
)


# =============================================================================
# 14. POST-PILOT ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    main_hash_before_cell11_nb63
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    rf_hash_before_cell11_nb63
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    checkpoint_hash_before_cell11_nb63
)


print()
print("=" * 92)
print("10. POST-PILOT ASSET INTEGRITY")
print("=" * 92)


print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training or retraining occurred."
)


# =============================================================================
# 15. FINAL STATUS
# =============================================================================

CELL11_NB63_STATUS = {

    "cell10_pass":
        True,

    "corrected_campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "pilot_seed":
        CORRECTED_PILOT_SEED_CELL11_NB63,

    "pilot_battles":
        24,

    "runtime_failure_count":
        corrected_pilot_result_cell11_nb63[
            "runtime_failures"
        ],

    "infrastructure_pass":
        corrected_pilot_infrastructure_pass_cell11_nb63,

    "semantic_validity_pass":
        all_agents_semantically_valid_cell11_nb63,

    "agent_summary":
        agent_summary_cell11_nb63,

    "deployment_policy_unknown_move_rate":
        agent_summary_cell11_nb63[
            "DEPLOYMENT_POLICY"
        ][
            "unknown_move_rate"
        ],

    "deployment_policy_positive_damage_rate":
        agent_summary_cell11_nb63[
            "DEPLOYMENT_POLICY"
        ][
            "positive_damage_move_rate"
        ],

    "first_legal_unknown_move_rate":
        agent_summary_cell11_nb63[
            "FIRST_LEGAL_MOVE"
        ][
            "unknown_move_rate"
        ],

    "max_damage_unknown_move_rate":
        agent_summary_cell11_nb63[
            "MAX_DAMAGE_LEGAL_MOVE"
        ][
            "unknown_move_rate"
        ],

    "full_240_battle_campaign_authorized":
        full_campaign_authorized_cell11_nb63,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "temporary_runtime_deleted":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 11 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL11_NB63_STATUS,
        indent=2,
    )
)


print()

print(
    "Corrected 24-battle pilot completed."
)

print(
    "Schema repair successfully eliminated the Cell 8 harness artifact."
)

print(
    "Full 240-battle campaign is now authorized."
)

print()
print(
    "NEXT STEP: Cell 12 — execute the complete corrected "
    "240-battle simulation campaign."
)


# In[12]:


# =============================================================================
# NOTEBOOK 63 — CELL 12
# FULL CORRECTED 240-BATTLE SIMULATION CAMPAIGN
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 12")
print("FULL CORRECTED 240-BATTLE SIMULATION CAMPAIGN")
print("=" * 92)


# =============================================================================
# 0. AUTHORIZATION GATE
# =============================================================================

assert (
    CELL11_NB63_STATUS[
        "validation_status"
    ]
    ==
    "PASS"
)

assert (
    CELL11_NB63_STATUS[
        "infrastructure_pass"
    ]
    is True
)

assert (
    CELL11_NB63_STATUS[
        "semantic_validity_pass"
    ]
    is True
)

assert (
    CELL11_NB63_STATUS[
        "full_240_battle_campaign_authorized"
    ]
    is True
)

assert (
    CELL11_NB63_STATUS[
        "runtime_failure_count"
    ]
    ==
    0
)

assert (
    CELL11_NB63_STATUS[
        "deployment_policy_unknown_move_rate"
    ]
    ==
    0.0
)

assert (
    CELL11_NB63_STATUS[
        "deployment_policy_positive_damage_rate"
    ]
    >
    0.0
)


print()
print("[OK] Cell 11 PASS confirmed.")
print("[OK] Corrected pilot infrastructure validated.")
print("[OK] Corrected pilot semantics validated.")
print("[OK] Full 240-battle campaign authorized.")


# =============================================================================
# 1. FREEZE FULL CAMPAIGN
# =============================================================================

full_campaign_matrix_cell12_nb63 = list(
    NB63_CORRECTED_CAMPAIGN_MATRIX
)


assert (
    len(
        full_campaign_matrix_cell12_nb63
    )
    ==
    240
), (
    "Corrected campaign must contain exactly 240 rows."
)


campaign_agent_counts_cell12_nb63 = {}

campaign_scenario_counts_cell12_nb63 = {}

campaign_side_counts_cell12_nb63 = {}

campaign_seed_counts_cell12_nb63 = {}


for row in full_campaign_matrix_cell12_nb63:

    agent_id = row[
        "agent_variant"
    ]

    scenario_id = row[
        "scenario_id"
    ]

    side_mode = row[
        "side_mode"
    ]

    seed = row[
        "seed"
    ]


    campaign_agent_counts_cell12_nb63[
        agent_id
    ] = (
        campaign_agent_counts_cell12_nb63.get(
            agent_id,
            0,
        )
        +
        1
    )


    campaign_scenario_counts_cell12_nb63[
        scenario_id
    ] = (
        campaign_scenario_counts_cell12_nb63.get(
            scenario_id,
            0,
        )
        +
        1
    )


    campaign_side_counts_cell12_nb63[
        side_mode
    ] = (
        campaign_side_counts_cell12_nb63.get(
            side_mode,
            0,
        )
        +
        1
    )


    campaign_seed_counts_cell12_nb63[
        seed
    ] = (
        campaign_seed_counts_cell12_nb63.get(
            seed,
            0,
        )
        +
        1
    )


print()
print("=" * 92)
print("1. FULL CAMPAIGN MATRIX")
print("=" * 92)

print(
    "Campaign rows:",
    len(
        full_campaign_matrix_cell12_nb63
    ),
)

print(
    "Configuration SHA256:",
    NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,
)

print()
print(
    "Agent distribution:",
    json.dumps(
        campaign_agent_counts_cell12_nb63,
        indent=2,
    ),
)

print()
print(
    "Scenario distribution:",
    json.dumps(
        campaign_scenario_counts_cell12_nb63,
        indent=2,
    ),
)

print()
print(
    "Side distribution:",
    json.dumps(
        campaign_side_counts_cell12_nb63,
        indent=2,
    ),
)

print()
print(
    "Unique seeds:",
    len(
        campaign_seed_counts_cell12_nb63
    ),
)


assert (
    set(
        campaign_agent_counts_cell12_nb63
    )
    ==
    {
        "DEPLOYMENT_POLICY",
        "FIRST_LEGAL_MOVE",
        "MAX_DAMAGE_LEGAL_MOVE",
    }
)


assert (
    set(
        campaign_scenario_counts_cell12_nb63
    )
    ==
    {
        "S1_BALANCED_LOW_HP",
        "S2_BALANCED_HIGH_HP",
        "S3_PLAYER_DAMAGE_ADVANTAGE",
        "S4_OPPONENT_DAMAGE_ADVANTAGE",
    }
)


assert (
    set(
        campaign_side_counts_cell12_nb63
    )
    ==
    {
        "NORMAL",
        "SWAPPED",
    }
)


print()
print(
    "[OK] Full corrected campaign matrix frozen."
)


# =============================================================================
# 2. ASSET INTEGRITY FREEZE
# =============================================================================

main_hash_before_cell12_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

rf_hash_before_cell12_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)

ppo_hash_before_cell12_nb63 = (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
)


assert (
    main_hash_before_cell12_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    rf_hash_before_cell12_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    ppo_hash_before_cell12_nb63
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("2. PRE-CAMPAIGN CERTIFIED ASSET FREEZE")
print("=" * 92)

print(
    "main.py:",
    main_hash_before_cell12_nb63,
)

print(
    "Certified RF:",
    rf_hash_before_cell12_nb63,
)

print(
    "PPO checkpoint:",
    ppo_hash_before_cell12_nb63,
)


print()
print("[OK] Certified assets frozen.")
print("[OK] No training or model mutation authorized.")


# =============================================================================
# 3. CREATE FULL-CAMPAIGN PAYLOAD
# =============================================================================

full_campaign_payload_cell12_nb63 = {

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "checkpoint_sha256":
        DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,

    "scenarios":
        NB63_SCENARIOS_SCHEMA_CORRECT,

    "campaign_matrix":
        full_campaign_matrix_cell12_nb63,
}


full_campaign_payload_path_cell12_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell12_full_campaign_payload.json"
)


full_campaign_payload_path_cell12_nb63.write_text(
    json.dumps(
        full_campaign_payload_cell12_nb63,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("3. FULL CAMPAIGN PAYLOAD")
print("=" * 92)

print(
    "[OK]",
    full_campaign_payload_path_cell12_nb63,
)


# =============================================================================
# 4. CAMPAIGN EXECUTION CONTRACT
# =============================================================================

CELL12_NB63_EXECUTION_CONTRACT = {

    "campaign_rows":
        240,

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "scenario_count":
        len(
            campaign_scenario_counts_cell12_nb63
        ),

    "agent_variant_count":
        len(
            campaign_agent_counts_cell12_nb63
        ),

    "side_mode_count":
        len(
            campaign_side_counts_cell12_nb63
        ),

    "simulation_engine":
        "submission/final_agent/src/battle_simulation.py",

    "deployment_policy":
        "FinalPPOBattleAgent",

    "baseline_agents": [
        "FIRST_LEGAL_MOVE",
        "MAX_DAMAGE_LEGAL_MOVE",
    ],

    "execution_environment":
        "ISOLATED_TEMPORARY_DEPLOYMENT_COPY",

    "real_submission_writable":
        False,

    "training_allowed":
        False,

    "retraining_allowed":
        False,

    "model_mutation_allowed":
        False,
}


print()
print("=" * 92)
print("4. FULL CAMPAIGN EXECUTION CONTRACT")
print("=" * 92)

print(
    json.dumps(
        CELL12_NB63_EXECUTION_CONTRACT,
        indent=2,
    )
)


# =============================================================================
# 5. IMPORTANT EXECUTION HANDOFF
# =============================================================================
#
# Cell 11 has already proven the complete battle driver.
#
# For the full campaign, use the EXACT Cell 11 driver body with ONLY:
#
#     payload["pilot_matrix"]
#
# replaced by:
#
#     payload["campaign_matrix"]
#
# and execute all 240 records.
#
# Do NOT alter:
#
#     - build_state()
#     - FirstLegalMoveAgent
#     - MaxDamageLegalMoveAgent
#     - FinalPPOBattleAgent construction
#     - simulate_ai_battle()
#     - seed handling
#     - damage accounting
#     - outcome normalization
#
# This preserves experimental comparability with Cell 11.
#
# =============================================================================


print()
print("=" * 92)
print("5. EXECUTION HANDOFF")
print("=" * 92)

print(
    "[READY] Full 240-row payload frozen."
)

print(
    "[READY] Cell 11 execution harness is the validated execution engine."
)

print(
    "[REQUIRED] Execute the same validated harness over campaign_matrix."
)

print(
    "[BLOCKED] Do not interpret performance until all 240 records complete."
)


# =============================================================================
# 6. PRE-EXECUTION STATUS
# =============================================================================

CELL12_NB63_STATUS = {

    "cell11_pass":
        True,

    "full_campaign_authorized":
        True,

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "campaign_rows":
        240,

    "campaign_matrix_frozen":
        True,

    "campaign_payload_persisted":
        True,

    "campaign_payload_path":
        str(
            full_campaign_payload_path_cell12_nb63
        ),

    "agent_distribution":
        campaign_agent_counts_cell12_nb63,

    "scenario_distribution":
        campaign_scenario_counts_cell12_nb63,

    "side_distribution":
        campaign_side_counts_cell12_nb63,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "simulation_executed":
        False,

    "validation_status":
        "READY_FOR_FULL_EXECUTION",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 12 STATUS: READY FOR FULL EXECUTION")
print("=" * 92)

print(
    json.dumps(
        CELL12_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "The corrected 240-battle campaign is frozen and ready."
)

print(
    "NEXT STEP: Cell 12B — execute all 240 battles using "
    "the validated Cell 11 driver without changing battle semantics."
)


# In[13]:


# =============================================================================
# NOTEBOOK 63 — CELL 12B
# EXECUTE FULL CORRECTED 240-BATTLE CAMPAIGN
# REUSES VALIDATED CELL 11 BATTLE SEMANTICS
# =============================================================================

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 12B")
print("EXECUTE FULL CORRECTED 240-BATTLE CAMPAIGN")
print("=" * 92)


# =============================================================================
# 0. AUTHORIZATION GATE
# =============================================================================

assert (
    CELL12_NB63_STATUS[
        "validation_status"
    ]
    ==
    "READY_FOR_FULL_EXECUTION"
)

assert (
    CELL12_NB63_STATUS[
        "full_campaign_authorized"
    ]
    is True
)

assert (
    CELL12_NB63_STATUS[
        "campaign_rows"
    ]
    ==
    240
)

assert (
    CELL12_NB63_STATUS[
        "campaign_matrix_frozen"
    ]
    is True
)

assert (
    CELL12_NB63_STATUS[
        "main_integrity_preserved"
    ]
    is True
)

assert (
    CELL12_NB63_STATUS[
        "certified_model_integrity_preserved"
    ]
    is True
)

assert (
    CELL12_NB63_STATUS[
        "deployment_checkpoint_integrity_preserved"
    ]
    is True
)


print()
print("[OK] Cell 12 READY_FOR_FULL_EXECUTION confirmed.")
print("[OK] 240-row corrected campaign authorized.")
print("[OK] Battle semantics remain identical to Cell 11.")
print("[OK] No training or model modification permitted.")


# =============================================================================
# 1. PRE-EXECUTION ASSET FREEZE
# =============================================================================

main_hash_before_cell12b_nb63 = (
    sha256_file_nb63(
        FINAL_MAIN
    )
)

rf_hash_before_cell12b_nb63 = (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
)

ppo_hash_before_cell12b_nb63 = (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
)


assert (
    main_hash_before_cell12b_nb63
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    rf_hash_before_cell12b_nb63
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    ppo_hash_before_cell12b_nb63
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("1. PRE-EXECUTION ASSET FREEZE")
print("=" * 92)

print(
    "main.py:",
    main_hash_before_cell12b_nb63,
)

print(
    "Certified RF:",
    rf_hash_before_cell12b_nb63,
)

print(
    "PPO checkpoint:",
    ppo_hash_before_cell12b_nb63,
)

print()
print("[OK] Authoritative assets frozen before full campaign.")


# =============================================================================
# 2. TEMPORARY DEPLOYMENT COPY
# =============================================================================

REAL_FINAL_AGENT_CELL12B_NB63 = (
    SUBMISSION_DIR
    /
    "final_agent"
)


assert (
    REAL_FINAL_AGENT_CELL12B_NB63.is_dir()
)


temp_root_cell12b_nb63 = Path(
    tempfile.mkdtemp(
        prefix="ptcg_nb63_cell12b_"
    )
).resolve()


temp_final_agent_cell12b_nb63 = (
    temp_root_cell12b_nb63
    /
    "final_agent"
)


shutil.copytree(
    REAL_FINAL_AGENT_CELL12B_NB63,
    temp_final_agent_cell12b_nb63,
)


for package_dir in [
    temp_final_agent_cell12b_nb63
    /
    "src",

    temp_final_agent_cell12b_nb63
    /
    "src"
    /
    "agents",
]:

    if package_dir.is_dir():

        init_path = (
            package_dir
            /
            "__init__.py"
        )

        if not init_path.exists():

            init_path.write_text(
                "# Temporary NB63 Cell 12B package marker.\n",
                encoding="utf-8",
            )


payload_path_cell12b_nb63 = (
    temp_root_cell12b_nb63
    /
    "full_campaign_payload.json"
)


payload_path_cell12b_nb63.write_text(
    json.dumps(
        full_campaign_payload_cell12_nb63,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("2. TEMPORARY FULL-CAMPAIGN RUNTIME")
print("=" * 92)

print(
    "Temporary root:",
    temp_root_cell12b_nb63,
)

print()
print(
    "[OK] Full campaign will execute only against temporary deployment copy."
)


# =============================================================================
# 3. FULL CAMPAIGN DRIVER
# =============================================================================

driver_path_cell12b_nb63 = (
    temp_root_cell12b_nb63
    /
    "cell12b_full_campaign_driver.py"
)


driver_source_cell12b_nb63 = r'''
from __future__ import annotations

import json
import random
import statistics
import sys
import traceback
from pathlib import Path


RESULT = {

    "imports_pass":
        False,

    "policy_constructed":
        False,

    "requested_battles":
        0,

    "completed_battles":
        0,

    "runtime_failures":
        0,

    "records":
        [],

    "fatal_exception_type":
        None,

    "fatal_exception_message":
        None,

    "fatal_traceback":
        None,
}


try:

    final_agent_root = Path(
        sys.argv[1]
    ).resolve()


    payload_path = Path(
        sys.argv[2]
    ).resolve()


    sys.path.insert(
        0,
        str(
            final_agent_root
        ),
    )


    from src.battle_state import (
        PokemonState,
        PlayerState,
        BattleState,
    )

    from src.agent_decision import (
        AgentDecision,
    )

    from src.legal_moves import (
        get_current_legal_moves,
    )

    from src.agents.final_ppo_agent import (
        FinalPPOBattleAgent,
    )

    from src.battle_simulation import (
        simulate_ai_battle,
    )


    RESULT[
        "imports_pass"
    ] = True


    payload = json.loads(
        payload_path.read_text(
            encoding="utf-8"
        )
    )


    matrix = payload[
        "campaign_matrix"
    ]


    scenarios = {
        scenario[
            "scenario_id"
        ]:
            scenario

        for scenario
        in payload[
            "scenarios"
        ]
    }


    RESULT[
        "requested_battles"
    ] = len(
        matrix
    )


    # =========================================================================
    # Reproducibility
    # =========================================================================

    def set_seed(seed):

        random.seed(
            int(seed)
        )


        try:

            import numpy as np

            np.random.seed(
                int(seed)
                %
                (2 ** 32 - 1)
            )

        except Exception:

            pass


        try:

            import torch

            torch.manual_seed(
                int(seed)
            )

        except Exception:

            pass


    # =========================================================================
    # State builder
    # =========================================================================

    def make_pokemon(
        record,
    ):

        return PokemonState(

            card={
                "name":
                    record[
                        "name"
                    ],

                "attacks":
                    record[
                        "attacks"
                    ],
            },

            current_hp=float(
                record[
                    "hp"
                ]
            ),

            attached_energy=int(
                record[
                    "energy"
                ]
            ),

            status=None,

            damage=0.0,

            is_active=True,
        )


    def build_state(
        scenario,
        swap_sides,
    ):

        player_record = (
            scenario[
                "player"
            ]
        )

        opponent_record = (
            scenario[
                "opponent"
            ]
        )


        if swap_sides:

            player_record, opponent_record = (
                opponent_record,
                player_record,
            )


        return BattleState(

            player=PlayerState(
                active=make_pokemon(
                    player_record
                ),
                bench=[],
                prize_cards_remaining=6,
                hand_size=7,
            ),

            opponent=PlayerState(
                active=make_pokemon(
                    opponent_record
                ),
                bench=[],
                prize_cards_remaining=6,
                hand_size=7,
            ),

            turn_number=1,

            current_player="Player",
        )


    # =========================================================================
    # Baselines
    # =========================================================================

    class FirstLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            move = (
                legal_moves[0]
                if legal_moves
                else {
                    "name":
                        "Pass",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        "No legal move.",
                }
            )


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=max(
                    1,
                    len(
                        legal_moves
                    ),
                ),

                principal_variation=[
                    move
                ],
            )


    class MaxDamageLegalMoveAgent:

        def choose_move(
            self,
            state,
            depth,
        ):

            legal_moves = (
                get_current_legal_moves(
                    state
                )
            )


            if legal_moves:

                move = max(
                    legal_moves,
                    key=lambda item: float(
                        item.get(
                            "damage",
                            0.0,
                        )
                        or
                        0.0
                    ),
                )

            else:

                move = {
                    "name":
                        "Pass",

                    "damage":
                        0.0,

                    "energy_cost":
                        0,

                    "effect":
                        "No legal move.",
                }


            return AgentDecision(
                move=move,

                score=float(
                    move.get(
                        "damage",
                        0.0,
                    )
                    or
                    0.0
                ),

                search_depth=int(
                    depth
                ),

                nodes=max(
                    1,
                    len(
                        legal_moves
                    ),
                ),

                principal_variation=[
                    move
                ],
            )


    # =========================================================================
    # Deployment PPO
    # =========================================================================

    checkpoint_paths = sorted(
        [
            path
            for path
            in (
                final_agent_root
                /
                "models"
            ).rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".pkl",
                    ".pickle",
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".joblib",
                }
            )
        ]
    )


    if not checkpoint_paths:

        raise RuntimeError(
            "Deployment checkpoint missing."
        )


    deployment_agent = (
        FinalPPOBattleAgent(
            checkpoint_path=str(
                checkpoint_paths[
                    0
                ]
            ),

            device="cpu",

            deterministic=True,
        )
    )


    RESULT[
        "policy_constructed"
    ] = True


    agents = {

        "DEPLOYMENT_POLICY":
            deployment_agent,

        "FIRST_LEGAL_MOVE":
            FirstLegalMoveAgent(),

        "MAX_DAMAGE_LEGAL_MOVE":
            MaxDamageLegalMoveAgent(),
    }


    # =========================================================================
    # Full 240-battle execution
    # =========================================================================

    for spec in matrix:

        row = {

            "battle_id":
                spec[
                    "battle_id"
                ],

            "agent_variant":
                spec[
                    "agent_variant"
                ],

            "scenario_id":
                spec[
                    "scenario_id"
                ],

            "side_mode":
                spec[
                    "side_mode"
                ],

            "seed":
                spec[
                    "seed"
                ],

            "winner":
                None,

            "normalized_outcome":
                None,

            "turn_count":
                None,

            "stop_reason":
                None,

            "player_final_hp":
                None,

            "opponent_final_hp":
                None,

            "total_moves":
                0,

            "unknown_move_count":
                0,

            "unknown_move_rate":
                0.0,

            "zero_damage_move_count":
                0,

            "zero_damage_move_rate":
                0.0,

            "positive_damage_move_count":
                0,

            "positive_damage_move_rate":
                0.0,

            "total_damage":
                0.0,

            "mean_damage_per_move":
                0.0,

            "policy_score_mean":
                None,

            "policy_score_min":
                None,

            "policy_score_max":
                None,

            "runtime_exception":
                False,

            "runtime_exception_type":
                None,

            "runtime_exception_message":
                None,

            "checkpoint_sha256":
                spec.get(
                    "checkpoint_sha256"
                ),
        }


        try:

            set_seed(
                int(
                    spec[
                        "seed"
                    ]
                )
            )


            state = build_state(

                scenarios[
                    spec[
                        "scenario_id"
                    ]
                ],

                bool(
                    spec[
                        "swap_sides"
                    ]
                ),
            )


            agent = agents[
                spec[
                    "agent_variant"
                ]
            ]


            simulation = (
                simulate_ai_battle(

                    initial_state=state,

                    agent=agent,

                    search_depth=int(
                        spec[
                            "search_depth"
                        ]
                    ),

                    max_turns=int(
                        spec[
                            "max_turns"
                        ]
                    ),

                    verbose=False,
                )
            )


            winner = getattr(
                simulation,
                "winner",
                None,
            )


            turn_count = getattr(
                simulation,
                "turn_count",
                None,
            )


            stop_reason = getattr(
                simulation,
                "stop_reason",
                None,
            )


            turns = list(
                getattr(
                    simulation,
                    "turns",
                    []
                )
                or
                []
            )


            final_state = getattr(
                simulation,
                "final_state",
                None,
            )


            row[
                "winner"
            ] = winner


            row[
                "turn_count"
            ] = turn_count


            row[
                "stop_reason"
            ] = stop_reason


            # -------------------------------------------------------------
            # Outcome normalization
            # -------------------------------------------------------------

            if winner == "Player":

                row[
                    "normalized_outcome"
                ] = "PLAYER_WIN"


            elif winner == "Opponent":

                row[
                    "normalized_outcome"
                ] = "OPPONENT_WIN"


            elif (
                turn_count
                is not None
                and
                int(
                    turn_count
                )
                >=
                int(
                    spec[
                        "max_turns"
                    ]
                )
            ):

                row[
                    "normalized_outcome"
                ] = "TIMEOUT_DRAW"


            else:

                row[
                    "normalized_outcome"
                ] = "NO_WINNER"


            # -------------------------------------------------------------
            # Final HP
            # -------------------------------------------------------------

            if final_state is not None:

                try:

                    row[
                        "player_final_hp"
                    ] = float(
                        final_state
                        .player
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


                try:

                    row[
                        "opponent_final_hp"
                    ] = float(
                        final_state
                        .opponent
                        .active
                        .current_hp
                    )

                except Exception:

                    pass


            # -------------------------------------------------------------
            # Turn metrics
            # -------------------------------------------------------------

            row[
                "total_moves"
            ] = len(
                turns
            )


            damages = []
            scores = []


            for turn in turns:

                move_name = str(
                    getattr(
                        turn,
                        "move_name",
                        "",
                    )
                    or
                    ""
                )


                if (
                    move_name.strip().lower()
                    ==
                    "unknown move"
                ):

                    row[
                        "unknown_move_count"
                    ] += 1


                try:

                    damage = float(
                        getattr(
                            turn,
                            "damage",
                            0.0,
                        )
                        or
                        0.0
                    )

                except Exception:

                    damage = 0.0


                damages.append(
                    damage
                )


                if damage > 0:

                    row[
                        "positive_damage_move_count"
                    ] += 1

                else:

                    row[
                        "zero_damage_move_count"
                    ] += 1


                raw_score = getattr(
                    turn,
                    "search_score",
                    None,
                )


                if raw_score is not None:

                    try:

                        scores.append(
                            float(
                                raw_score
                            )
                        )

                    except Exception:

                        pass


            total_moves = row[
                "total_moves"
            ]


            if total_moves > 0:

                row[
                    "unknown_move_rate"
                ] = (
                    row[
                        "unknown_move_count"
                    ]
                    /
                    total_moves
                )


                row[
                    "zero_damage_move_rate"
                ] = (
                    row[
                        "zero_damage_move_count"
                    ]
                    /
                    total_moves
                )


                row[
                    "positive_damage_move_rate"
                ] = (
                    row[
                        "positive_damage_move_count"
                    ]
                    /
                    total_moves
                )


            row[
                "total_damage"
            ] = float(
                sum(
                    damages
                )
            )


            if total_moves > 0:

                row[
                    "mean_damage_per_move"
                ] = (
                    row[
                        "total_damage"
                    ]
                    /
                    total_moves
                )


            if scores:

                row[
                    "policy_score_mean"
                ] = float(
                    statistics.mean(
                        scores
                    )
                )


                row[
                    "policy_score_min"
                ] = float(
                    min(
                        scores
                    )
                )


                row[
                    "policy_score_max"
                ] = float(
                    max(
                        scores
                    )
                )


            RESULT[
                "completed_battles"
            ] += 1


        except Exception as battle_exc:

            row[
                "runtime_exception"
            ] = True


            row[
                "runtime_exception_type"
            ] = type(
                battle_exc
            ).__name__


            row[
                "runtime_exception_message"
            ] = str(
                battle_exc
            )


            row[
                "normalized_outcome"
            ] = "RUNTIME_FAILURE"


            row[
                "traceback"
            ] = traceback.format_exc()


            RESULT[
                "runtime_failures"
            ] += 1


        RESULT[
            "records"
        ].append(
            row
        )


except Exception as fatal_exc:

    RESULT[
        "fatal_exception_type"
    ] = type(
        fatal_exc
    ).__name__


    RESULT[
        "fatal_exception_message"
    ] = str(
        fatal_exc
    )


    RESULT[
        "fatal_traceback"
    ] = traceback.format_exc()


print(
    "NB63_CELL12B_RESULT_JSON="
    +
    json.dumps(
        RESULT,
        default=str,
    )
)
'''


driver_path_cell12b_nb63.write_text(
    textwrap.dedent(
        driver_source_cell12b_nb63
    ),
    encoding="utf-8",
)


compile(
    driver_path_cell12b_nb63.read_text(
        encoding="utf-8"
    ),
    str(
        driver_path_cell12b_nb63
    ),
    "exec",
)


print()
print(
    "[OK] Full-campaign driver created and compiled."
)


# =============================================================================
# 4. EXECUTE ALL 240 BATTLES
# =============================================================================

print()
print("=" * 92)
print("3. EXECUTE 240 BATTLES")
print("=" * 92)


environment_cell12b_nb63 = (
    os.environ.copy()
)


environment_cell12b_nb63[
    "PYTHONDONTWRITEBYTECODE"
] = "1"


process_cell12b_nb63 = subprocess.run(
    [
        sys.executable,

        str(
            driver_path_cell12b_nb63
        ),

        str(
            temp_final_agent_cell12b_nb63
        ),

        str(
            payload_path_cell12b_nb63
        ),
    ],

    cwd=str(
        temp_root_cell12b_nb63
    ),

    env=environment_cell12b_nb63,

    capture_output=True,

    text=True,

    timeout=1800,
)


print(
    "Return code:",
    process_cell12b_nb63.returncode,
)


print()
print("--- STDERR ---")

print(
    process_cell12b_nb63.stderr
)


assert (
    process_cell12b_nb63.returncode
    ==
    0
), (
    "Full campaign subprocess failed.\n"
    f"{process_cell12b_nb63.stderr}"
)


# =============================================================================
# 5. PARSE FULL RESULTS
# =============================================================================

result_prefix_cell12b_nb63 = (
    "NB63_CELL12B_RESULT_JSON="
)


result_line_cell12b_nb63 = None


for line in (
    process_cell12b_nb63.stdout.splitlines()
):

    if line.startswith(
        result_prefix_cell12b_nb63
    ):

        result_line_cell12b_nb63 = line


assert (
    result_line_cell12b_nb63
    is not None
)


full_campaign_result_cell12b_nb63 = json.loads(
    result_line_cell12b_nb63[
        len(
            result_prefix_cell12b_nb63
        ):
    ]
)


print()
print("=" * 92)
print("4. FULL CAMPAIGN EXECUTION SUMMARY")
print("=" * 92)


execution_summary_cell12b_nb63 = {

    "imports_pass":
        full_campaign_result_cell12b_nb63[
            "imports_pass"
        ],

    "policy_constructed":
        full_campaign_result_cell12b_nb63[
            "policy_constructed"
        ],

    "requested_battles":
        full_campaign_result_cell12b_nb63[
            "requested_battles"
        ],

    "completed_battles":
        full_campaign_result_cell12b_nb63[
            "completed_battles"
        ],

    "runtime_failures":
        full_campaign_result_cell12b_nb63[
            "runtime_failures"
        ],

    "fatal_exception_type":
        full_campaign_result_cell12b_nb63[
            "fatal_exception_type"
        ],
}


print(
    json.dumps(
        execution_summary_cell12b_nb63,
        indent=2,
    )
)


# =============================================================================
# 6. REQUIRED EXECUTION ASSERTIONS
# =============================================================================

assert (
    full_campaign_result_cell12b_nb63[
        "imports_pass"
    ]
    is True
)


assert (
    full_campaign_result_cell12b_nb63[
        "policy_constructed"
    ]
    is True
)


assert (
    full_campaign_result_cell12b_nb63[
        "requested_battles"
    ]
    ==
    240
)


assert (
    full_campaign_result_cell12b_nb63[
        "completed_battles"
    ]
    ==
    240
)


assert (
    len(
        full_campaign_result_cell12b_nb63[
            "records"
        ]
    )
    ==
    240
)


assert (
    full_campaign_result_cell12b_nb63[
        "runtime_failures"
    ]
    ==
    0
)


assert (
    full_campaign_result_cell12b_nb63[
        "fatal_exception_type"
    ]
    is None
), (
    full_campaign_result_cell12b_nb63.get(
        "fatal_traceback"
    )
)


print()
print(
    "[OK] All 240 campaign battles completed."
)

print(
    "[OK] Runtime failures: 0."
)


# =============================================================================
# 7. PERSIST FULL BATTLE RECORDS
# =============================================================================

records_cell12b_nb63 = (
    full_campaign_result_cell12b_nb63[
        "records"
    ]
)


full_records_json_output_cell12b_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell12b_full_campaign_records.json"
)


full_records_json_artifact_cell12b_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell12b_full_campaign_records.json"
)


full_records_csv_output_cell12b_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell12b_full_campaign_records.csv"
)


for path in [
    full_records_json_output_cell12b_nb63,
    full_records_json_artifact_cell12b_nb63,
]:

    path.write_text(
        json.dumps(
            records_cell12b_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


csv_fields_cell12b_nb63 = sorted(
    {
        key
        for row
        in records_cell12b_nb63
        for key
        in row.keys()
        if key
        !=
        "traceback"
    }
)


with full_records_csv_output_cell12b_nb63.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=csv_fields_cell12b_nb63,
    )

    writer.writeheader()


    for row in records_cell12b_nb63:

        writer.writerow(
            {
                key:
                    row.get(
                        key
                    )

                for key
                in csv_fields_cell12b_nb63
            }
        )


print()
print("=" * 92)
print("5. FULL CAMPAIGN EVIDENCE PERSISTENCE")
print("=" * 92)


print(
    "[OK]",
    full_records_json_output_cell12b_nb63,
)

print(
    "[OK]",
    full_records_csv_output_cell12b_nb63,
)

print(
    "[OK]",
    full_records_json_artifact_cell12b_nb63,
)


# =============================================================================
# 8. TEMPORARY CLEANUP
# =============================================================================

shutil.rmtree(
    temp_root_cell12b_nb63,
    ignore_errors=True,
)


temporary_deleted_cell12b_nb63 = (
    not temp_root_cell12b_nb63.exists()
)


assert (
    temporary_deleted_cell12b_nb63
    is True
)


print()
print(
    "[OK] Temporary full-campaign runtime deleted."
)


# =============================================================================
# 9. POST-CAMPAIGN ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    main_hash_before_cell12b_nb63
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    rf_hash_before_cell12b_nb63
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    ppo_hash_before_cell12b_nb63
)


print()
print("=" * 92)
print("6. POST-CAMPAIGN ASSET INTEGRITY")
print("=" * 92)


print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training or retraining occurred."
)


# =============================================================================
# 10. FINAL STATUS
# =============================================================================

CELL12B_NB63_STATUS = {

    "cell12_ready":
        True,

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "requested_battles":
        240,

    "completed_battles":
        full_campaign_result_cell12b_nb63[
            "completed_battles"
        ],

    "runtime_failure_count":
        full_campaign_result_cell12b_nb63[
            "runtime_failures"
        ],

    "records_persisted":
        True,

    "records_json":
        str(
            full_records_json_output_cell12b_nb63
        ),

    "records_csv":
        str(
            full_records_csv_output_cell12b_nb63
        ),

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "temporary_runtime_deleted":
        True,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 12B STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL12B_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Full corrected 240-battle campaign completed."
)

print(
    "All battle-level records persisted."
)

print(
    "Certified assets remained frozen."
)

print()
print(
    "NEXT STEP: Cell 13 — statistical and strategic analysis "
    "of the complete 240-battle campaign."
)


# In[14]:


# =============================================================================
# NOTEBOOK 63 — CELL 13
# STATISTICAL + STRATEGIC ANALYSIS OF FULL 240-BATTLE CAMPAIGN
# =============================================================================

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 13")
print("STATISTICAL + STRATEGIC ANALYSIS OF FULL 240-BATTLE CAMPAIGN")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert (
    CELL12B_NB63_STATUS["validation_status"]
    ==
    "PASS"
)

assert (
    CELL12B_NB63_STATUS["completed_battles"]
    ==
    240
)

assert (
    CELL12B_NB63_STATUS["runtime_failure_count"]
    ==
    0
)

assert (
    len(records_cell12b_nb63)
    ==
    240
)


print()
print("[OK] Cell 12B PASS confirmed.")
print("[OK] 240 battle records available.")
print("[OK] Runtime failures = 0.")
print("[OK] Beginning analysis only; no additional simulations will run.")


# =============================================================================
# 1. HELPERS
# =============================================================================

def safe_mean_cell13(values):

    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        statistics.mean(values)
    )


def safe_median_cell13(values):

    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        statistics.median(values)
    )


def proportion_ci_95_cell13(
    successes: int,
    total: int,
):

    if total <= 0:
        return {
            "rate": None,
            "lower": None,
            "upper": None,
        }

    p = (
        successes
        /
        total
    )

    z = 1.96

    se = math.sqrt(
        p
        *
        (1.0 - p)
        /
        total
    )

    lower = max(
        0.0,
        p - z * se,
    )

    upper = min(
        1.0,
        p + z * se,
    )

    return {
        "rate":
            float(p),

        "lower":
            float(lower),

        "upper":
            float(upper),
    }


# =============================================================================
# 2. OVERALL AGENT SUMMARY
# =============================================================================

agent_ids_cell13 = [
    "DEPLOYMENT_POLICY",
    "FIRST_LEGAL_MOVE",
    "MAX_DAMAGE_LEGAL_MOVE",
]


overall_summary_cell13 = {}


for agent_id in agent_ids_cell13:

    rows = [
        row
        for row in records_cell12b_nb63
        if (
            row[
                "agent_variant"
            ]
            ==
            agent_id
        )
    ]


    assert (
        len(rows)
        ==
        80
    )


    player_wins = sum(
        1
        for row in rows
        if (
            row[
                "normalized_outcome"
            ]
            ==
            "PLAYER_WIN"
        )
    )


    opponent_wins = sum(
        1
        for row in rows
        if (
            row[
                "normalized_outcome"
            ]
            ==
            "OPPONENT_WIN"
        )
    )


    timeouts = sum(
        1
        for row in rows
        if (
            row[
                "normalized_outcome"
            ]
            ==
            "TIMEOUT_DRAW"
        )
    )


    runtime_failures = sum(
        1
        for row in rows
        if row[
            "runtime_exception"
        ]
    )


    total_moves = sum(
        int(
            row[
                "total_moves"
            ]
            or
            0
        )
        for row
        in rows
    )


    unknown_moves = sum(
        int(
            row[
                "unknown_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    zero_damage_moves = sum(
        int(
            row[
                "zero_damage_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    positive_damage_moves = sum(
        int(
            row[
                "positive_damage_move_count"
            ]
            or
            0
        )
        for row
        in rows
    )


    total_damage = sum(
        float(
            row[
                "total_damage"
            ]
            or
            0.0
        )
        for row
        in rows
    )


    decisive_games = (
        player_wins
        +
        opponent_wins
    )


    win_ci = proportion_ci_95_cell13(
        player_wins,
        len(rows),
    )


    decisive_win_ci = proportion_ci_95_cell13(
        player_wins,
        decisive_games,
    )


    overall_summary_cell13[
        agent_id
    ] = {

        "battles":
            len(rows),

        "player_wins":
            player_wins,

        "opponent_wins":
            opponent_wins,

        "timeouts":
            timeouts,

        "runtime_failures":
            runtime_failures,

        "decisive_games":
            decisive_games,

        "player_win_rate_all":
            (
                player_wins
                /
                len(rows)
            ),

        "player_win_rate_all_ci95":
            win_ci,

        "player_win_rate_decisive":
            (
                player_wins
                /
                decisive_games
                if decisive_games
                else None
            ),

        "player_win_rate_decisive_ci95":
            decisive_win_ci,

        "timeout_rate":
            (
                timeouts
                /
                len(rows)
            ),

        "mean_turn_count":
            safe_mean_cell13(
                [
                    row[
                        "turn_count"
                    ]
                    for row
                    in rows
                ]
            ),

        "median_turn_count":
            safe_median_cell13(
                [
                    row[
                        "turn_count"
                    ]
                    for row
                    in rows
                ]
            ),

        "total_moves":
            total_moves,

        "mean_moves_per_battle":
            (
                total_moves
                /
                len(rows)
            ),

        "unknown_move_count":
            unknown_moves,

        "unknown_move_rate":
            (
                unknown_moves
                /
                total_moves
                if total_moves
                else 0.0
            ),

        "zero_damage_move_count":
            zero_damage_moves,

        "zero_damage_move_rate":
            (
                zero_damage_moves
                /
                total_moves
                if total_moves
                else 0.0
            ),

        "positive_damage_move_count":
            positive_damage_moves,

        "positive_damage_move_rate":
            (
                positive_damage_moves
                /
                total_moves
                if total_moves
                else 0.0
            ),

        "total_damage":
            total_damage,

        "mean_damage_per_move":
            (
                total_damage
                /
                total_moves
                if total_moves
                else 0.0
            ),

        "mean_damage_per_battle":
            (
                total_damage
                /
                len(rows)
            ),
    }


print()
print("=" * 92)
print("1. OVERALL AGENT SUMMARY")
print("=" * 92)

print(
    json.dumps(
        overall_summary_cell13,
        indent=2,
    )
)


# =============================================================================
# 3. SUMMARY BY SCENARIO
# =============================================================================

scenario_summary_cell13 = {}


scenario_ids_cell13 = sorted(
    {
        row[
            "scenario_id"
        ]
        for row
        in records_cell12b_nb63
    }
)


for scenario_id in scenario_ids_cell13:

    scenario_summary_cell13[
        scenario_id
    ] = {}


    for agent_id in agent_ids_cell13:

        rows = [
            row
            for row in records_cell12b_nb63
            if (
                row[
                    "scenario_id"
                ]
                ==
                scenario_id

                and

                row[
                    "agent_variant"
                ]
                ==
                agent_id
            )
        ]


        player_wins = sum(
            1
            for row in rows
            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "PLAYER_WIN"
            )
        )


        opponent_wins = sum(
            1
            for row in rows
            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "OPPONENT_WIN"
            )
        )


        timeouts = sum(
            1
            for row in rows
            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "TIMEOUT_DRAW"
            )
        )


        total_moves = sum(
            int(
                row[
                    "total_moves"
                ]
                or
                0
            )
            for row
            in rows
        )


        total_damage = sum(
            float(
                row[
                    "total_damage"
                ]
                or
                0.0
            )
            for row
            in rows
        )


        scenario_summary_cell13[
            scenario_id
        ][
            agent_id
        ] = {

            "battles":
                len(rows),

            "player_wins":
                player_wins,

            "opponent_wins":
                opponent_wins,

            "timeouts":
                timeouts,

            "player_win_rate":
                (
                    player_wins
                    /
                    len(rows)
                    if rows
                    else None
                ),

            "mean_turn_count":
                safe_mean_cell13(
                    [
                        row[
                            "turn_count"
                        ]
                        for row
                        in rows
                    ]
                ),

            "mean_damage_per_move":
                (
                    total_damage
                    /
                    total_moves
                    if total_moves
                    else 0.0
                ),

            "mean_damage_per_battle":
                (
                    total_damage
                    /
                    len(rows)
                    if rows
                    else 0.0
                ),
        }


print()
print("=" * 92)
print("2. SUMMARY BY SCENARIO")
print("=" * 92)

print(
    json.dumps(
        scenario_summary_cell13,
        indent=2,
    )
)


# =============================================================================
# 4. SUMMARY BY SIDE ORIENTATION
# =============================================================================

side_summary_cell13 = {}


for side_mode in [
    "NORMAL",
    "SWAPPED",
]:

    side_summary_cell13[
        side_mode
    ] = {}


    for agent_id in agent_ids_cell13:

        rows = [
            row
            for row in records_cell12b_nb63
            if (
                row[
                    "side_mode"
                ]
                ==
                side_mode

                and

                row[
                    "agent_variant"
                ]
                ==
                agent_id
            )
        ]


        player_wins = sum(
            1
            for row in rows
            if (
                row[
                    "normalized_outcome"
                ]
                ==
                "PLAYER_WIN"
            )
        )


        side_summary_cell13[
            side_mode
        ][
            agent_id
        ] = {

            "battles":
                len(rows),

            "player_wins":
                player_wins,

            "player_win_rate":
                (
                    player_wins
                    /
                    len(rows)
                    if rows
                    else None
                ),

            "mean_turn_count":
                safe_mean_cell13(
                    [
                        row[
                            "turn_count"
                        ]
                        for row
                        in rows
                    ]
                ),
        }


print()
print("=" * 92)
print("3. SIDE-ORIENTATION SUMMARY")
print("=" * 92)

print(
    json.dumps(
        side_summary_cell13,
        indent=2,
    )
)


# =============================================================================
# 5. DEPLOYMENT VS BASELINES
# =============================================================================

deployment_cell13 = (
    overall_summary_cell13[
        "DEPLOYMENT_POLICY"
    ]
)


first_cell13 = (
    overall_summary_cell13[
        "FIRST_LEGAL_MOVE"
    ]
)


max_damage_cell13 = (
    overall_summary_cell13[
        "MAX_DAMAGE_LEGAL_MOVE"
    ]
)


deployment_vs_baselines_cell13 = {

    "deployment_minus_first_legal_win_rate":
        (
            deployment_cell13[
                "player_win_rate_all"
            ]
            -
            first_cell13[
                "player_win_rate_all"
            ]
        ),

    "deployment_minus_max_damage_win_rate":
        (
            deployment_cell13[
                "player_win_rate_all"
            ]
            -
            max_damage_cell13[
                "player_win_rate_all"
            ]
        ),

    "deployment_minus_first_legal_damage_per_move":
        (
            deployment_cell13[
                "mean_damage_per_move"
            ]
            -
            first_cell13[
                "mean_damage_per_move"
            ]
        ),

    "deployment_minus_max_damage_damage_per_move":
        (
            deployment_cell13[
                "mean_damage_per_move"
            ]
            -
            max_damage_cell13[
                "mean_damage_per_move"
            ]
        ),

    "deployment_minus_first_legal_mean_turns":
        (
            deployment_cell13[
                "mean_turn_count"
            ]
            -
            first_cell13[
                "mean_turn_count"
            ]
        ),

    "deployment_minus_max_damage_mean_turns":
        (
            deployment_cell13[
                "mean_turn_count"
            ]
            -
            max_damage_cell13[
                "mean_turn_count"
            ]
        ),
}


print()
print("=" * 92)
print("4. DEPLOYMENT VS BASELINES")
print("=" * 92)

print(
    json.dumps(
        deployment_vs_baselines_cell13,
        indent=2,
    )
)


# =============================================================================
# 6. SIDE-SWAP DELTAS
# =============================================================================

side_swap_delta_cell13 = {}


for agent_id in agent_ids_cell13:

    normal_rate = (
        side_summary_cell13[
            "NORMAL"
        ][
            agent_id
        ][
            "player_win_rate"
        ]
    )


    swapped_rate = (
        side_summary_cell13[
            "SWAPPED"
        ][
            agent_id
        ][
            "player_win_rate"
        ]
    )


    side_swap_delta_cell13[
        agent_id
    ] = {

        "normal_win_rate":
            normal_rate,

        "swapped_win_rate":
            swapped_rate,

        "normal_minus_swapped":
            (
                normal_rate
                -
                swapped_rate
            ),
    }


print()
print("=" * 92)
print("5. SIDE-SWAP DELTAS")
print("=" * 92)

print(
    json.dumps(
        side_swap_delta_cell13,
        indent=2,
    )
)


# =============================================================================
# 7. EXACT ACTION-PATTERN COMPARISON
# =============================================================================
#
# We cannot reconstruct every turn-level move from the campaign record alone,
# but battle-level equality across available metrics can still identify whether
# deployment and FIRST_LEGAL_MOVE behave indistinguishably under this test.
# =============================================================================

comparison_keys_cell13 = [
    "scenario_id",
    "side_mode",
    "seed",
]


def build_lookup_cell13(
    agent_id,
):

    return {

        (
            row[
                "scenario_id"
            ],
            row[
                "side_mode"
            ],
            row[
                "seed"
            ],
        ):
            row

        for row
        in records_cell12b_nb63
        if (
            row[
                "agent_variant"
            ]
            ==
            agent_id
        )
    }


deployment_lookup_cell13 = (
    build_lookup_cell13(
        "DEPLOYMENT_POLICY"
    )
)


first_lookup_cell13 = (
    build_lookup_cell13(
        "FIRST_LEGAL_MOVE"
    )
)


max_lookup_cell13 = (
    build_lookup_cell13(
        "MAX_DAMAGE_LEGAL_MOVE"
    )
)


shared_keys_cell13 = sorted(
    set(
        deployment_lookup_cell13
    )
    &
    set(
        first_lookup_cell13
    )
    &
    set(
        max_lookup_cell13
    )
)


assert (
    len(
        shared_keys_cell13
    )
    ==
    80
)


deployment_first_match_count_cell13 = 0

deployment_max_match_count_cell13 = 0


metric_fields_for_equivalence_cell13 = [
    "winner",
    "turn_count",
    "player_final_hp",
    "opponent_final_hp",
    "total_moves",
    "total_damage",
]


for key in shared_keys_cell13:

    deployment_row = (
        deployment_lookup_cell13[
            key
        ]
    )


    first_row = (
        first_lookup_cell13[
            key
        ]
    )


    max_row = (
        max_lookup_cell13[
            key
        ]
    )


    if all(
        deployment_row.get(field)
        ==
        first_row.get(field)

        for field
        in metric_fields_for_equivalence_cell13
    ):

        deployment_first_match_count_cell13 += 1


    if all(
        deployment_row.get(field)
        ==
        max_row.get(field)

        for field
        in metric_fields_for_equivalence_cell13
    ):

        deployment_max_match_count_cell13 += 1


equivalence_summary_cell13 = {

    "paired_battles":
        len(
            shared_keys_cell13
        ),

    "deployment_matches_first_legal":
        deployment_first_match_count_cell13,

    "deployment_matches_first_legal_rate":
        (
            deployment_first_match_count_cell13
            /
            len(
                shared_keys_cell13
            )
        ),

    "deployment_matches_max_damage":
        deployment_max_match_count_cell13,

    "deployment_matches_max_damage_rate":
        (
            deployment_max_match_count_cell13
            /
            len(
                shared_keys_cell13
            )
        ),
}


print()
print("=" * 92)
print("6. BATTLE-LEVEL BEHAVIORAL EQUIVALENCE")
print("=" * 92)

print(
    json.dumps(
        equivalence_summary_cell13,
        indent=2,
    )
)


# =============================================================================
# 8. STRATEGIC INTERPRETATION FLAGS
# =============================================================================

strategic_flags_cell13 = {

    "deployment_runtime_stable":
        (
            deployment_cell13[
                "runtime_failures"
            ]
            ==
            0
        ),

    "deployment_unknown_move_free":
        (
            deployment_cell13[
                "unknown_move_rate"
            ]
            ==
            0.0
        ),

    "deployment_all_moves_positive_damage":
        (
            deployment_cell13[
                "positive_damage_move_rate"
            ]
            ==
            1.0
        ),

    "deployment_has_no_timeouts":
        (
            deployment_cell13[
                "timeout_rate"
            ]
            ==
            0.0
        ),

    "deployment_outperforms_first_legal_on_win_rate":
        (
            deployment_cell13[
                "player_win_rate_all"
            ]
            >
            first_cell13[
                "player_win_rate_all"
            ]
        ),

    "deployment_outperforms_max_damage_on_win_rate":
        (
            deployment_cell13[
                "player_win_rate_all"
            ]
            >
            max_damage_cell13[
                "player_win_rate_all"
            ]
        ),

    "deployment_identical_to_first_legal_all_paired_battles":
        (
            deployment_first_match_count_cell13
            ==
            len(
                shared_keys_cell13
            )
        ),

    "max_damage_more_damage_efficient_than_deployment":
        (
            max_damage_cell13[
                "mean_damage_per_move"
            ]
            >
            deployment_cell13[
                "mean_damage_per_move"
            ]
        ),
}


print()
print("=" * 92)
print("7. STRATEGIC INTERPRETATION FLAGS")
print("=" * 92)

print(
    json.dumps(
        strategic_flags_cell13,
        indent=2,
    )
)


# =============================================================================
# 9. SUMMARY CLASSIFICATION
# =============================================================================

if (
    strategic_flags_cell13[
        "deployment_identical_to_first_legal_all_paired_battles"
    ]
):

    deployment_behavior_classification_cell13 = (
        "DEPLOYMENT_POLICY_BEHAVIORALLY_EQUIVALENT_TO_FIRST_LEGAL_"
        "UNDER_CONTROLLED_ATTACK_ONLY_CAMPAIGN"
    )


elif (
    strategic_flags_cell13[
        "deployment_outperforms_first_legal_on_win_rate"
    ]
):

    deployment_behavior_classification_cell13 = (
        "DEPLOYMENT_POLICY_SHOWS_ADVANTAGE_OVER_FIRST_LEGAL_BASELINE"
    )


else:

    deployment_behavior_classification_cell13 = (
        "DEPLOYMENT_POLICY_DISTINCT_BUT_NO_WIN_RATE_ADVANTAGE_"
        "OVER_FIRST_LEGAL_BASELINE"
    )


analysis_classification_cell13 = {

    "campaign_execution":
        "VALID",

    "campaign_semantics":
        "VALID",

    "deployment_runtime":
        "STABLE",

    "deployment_behavior_classification":
        deployment_behavior_classification_cell13,

    "interpretation_scope":
        (
            "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES; "
            "NOT FULL OFFICIAL PTCG GAMEPLAY"
        ),
}


print()
print("=" * 92)
print("8. ANALYSIS CLASSIFICATION")
print("=" * 92)

print(
    json.dumps(
        analysis_classification_cell13,
        indent=2,
    )
)


# =============================================================================
# 10. PERSIST ANALYSIS
# =============================================================================

cell13_analysis_payload_nb63 = {

    "notebook":
        63,

    "cell":
        13,

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "battle_count":
        240,

    "overall_summary":
        overall_summary_cell13,

    "scenario_summary":
        scenario_summary_cell13,

    "side_summary":
        side_summary_cell13,

    "deployment_vs_baselines":
        deployment_vs_baselines_cell13,

    "side_swap_delta":
        side_swap_delta_cell13,

    "behavioral_equivalence":
        equivalence_summary_cell13,

    "strategic_flags":
        strategic_flags_cell13,

    "analysis_classification":
        analysis_classification_cell13,
}


cell13_output_path_nb63 = (
    NB63_OUTPUT_DIR
    /
    "cell13_full_campaign_analysis.json"
)


cell13_artifact_path_nb63 = (
    NB63_ARTIFACT_DIR
    /
    "cell13_full_campaign_analysis.json"
)


cell13_report_path_nb63 = (
    NB63_REPORT_DIR
    /
    "notebook63_full_campaign_analysis.json"
)


for path in [
    cell13_output_path_nb63,
    cell13_artifact_path_nb63,
    cell13_report_path_nb63,
]:

    path.write_text(
        json.dumps(
            cell13_analysis_payload_nb63,
            indent=2,
        ),
        encoding="utf-8",
    )


print()
print("=" * 92)
print("9. ANALYSIS PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    cell13_output_path_nb63,
)

print(
    "[OK]",
    cell13_artifact_path_nb63,
)

print(
    "[OK]",
    cell13_report_path_nb63,
)


# =============================================================================
# 11. CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)

assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)

assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("10. CERTIFIED ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training/retraining performed."
)


# =============================================================================
# 12. FINAL STATUS
# =============================================================================

CELL13_NB63_STATUS = {

    "cell12b_pass":
        True,

    "battle_count":
        240,

    "analysis_complete":
        True,

    "deployment_runtime_stable":
        strategic_flags_cell13[
            "deployment_runtime_stable"
        ],

    "deployment_unknown_move_rate":
        deployment_cell13[
            "unknown_move_rate"
        ],

    "deployment_positive_damage_rate":
        deployment_cell13[
            "positive_damage_move_rate"
        ],

    "deployment_win_rate":
        deployment_cell13[
            "player_win_rate_all"
        ],

    "first_legal_win_rate":
        first_cell13[
            "player_win_rate_all"
        ],

    "max_damage_win_rate":
        max_damage_cell13[
            "player_win_rate_all"
        ],

    "deployment_first_legal_equivalence_rate":
        equivalence_summary_cell13[
            "deployment_matches_first_legal_rate"
        ],

    "deployment_max_damage_equivalence_rate":
        equivalence_summary_cell13[
            "deployment_matches_max_damage_rate"
        ],

    "deployment_behavior_classification":
        deployment_behavior_classification_cell13,

    "analysis_scope":
        "CONTROLLED_ATTACK_ONLY_SYNTHETIC_BATTLE_STATES",

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 13 STATUS: PASS")
print("=" * 92)


print(
    json.dumps(
        CELL13_NB63_STATUS,
        indent=2,
    )
)


print()
print(
    "Full 240-battle campaign analysis complete."
)

print(
    "NEXT STEP: Cell 14 — final Notebook 63 evidence package, "
    "limitations, and handoff to Notebook 64."
)


# In[15]:


# =============================================================================
# NOTEBOOK 63 — CELL 14
# FINAL EVIDENCE PACKAGE + LIMITATIONS + NOTEBOOK 64 HANDOFF
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path


print("=" * 92)
print("NOTEBOOK 63 — CELL 14")
print("FINAL EVIDENCE PACKAGE + LIMITATIONS + NOTEBOOK 64 HANDOFF")
print("=" * 92)


# =============================================================================
# 0. PREREQUISITES
# =============================================================================

assert CELL13_NB63_STATUS["validation_status"] == "PASS"
assert CELL13_NB63_STATUS["analysis_complete"] is True
assert CELL13_NB63_STATUS["battle_count"] == 240
assert CELL13_NB63_STATUS["deployment_runtime_stable"] is True

print()
print("[OK] Cell 13 PASS confirmed.")
print("[OK] Full 240-battle analysis confirmed.")
print("[OK] Notebook 63 ready for final consolidation.")


# =============================================================================
# 1. FINAL FINDINGS
# =============================================================================

NB63_FINAL_FINDINGS = {

    "campaign_battles":
        240,

    "runtime_failures":
        0,

    "deployment_runtime_stable":
        True,

    "deployment_unknown_move_rate":
        CELL13_NB63_STATUS[
            "deployment_unknown_move_rate"
        ],

    "deployment_positive_damage_rate":
        CELL13_NB63_STATUS[
            "deployment_positive_damage_rate"
        ],

    "deployment_win_rate":
        CELL13_NB63_STATUS[
            "deployment_win_rate"
        ],

    "first_legal_win_rate":
        CELL13_NB63_STATUS[
            "first_legal_win_rate"
        ],

    "max_damage_win_rate":
        CELL13_NB63_STATUS[
            "max_damage_win_rate"
        ],

    "deployment_first_legal_equivalence_rate":
        CELL13_NB63_STATUS[
            "deployment_first_legal_equivalence_rate"
        ],

    "deployment_max_damage_equivalence_rate":
        CELL13_NB63_STATUS[
            "deployment_max_damage_equivalence_rate"
        ],

    "behavior_classification":
        CELL13_NB63_STATUS[
            "deployment_behavior_classification"
        ],

    "analysis_scope":
        CELL13_NB63_STATUS[
            "analysis_scope"
        ],
}


print()
print("=" * 92)
print("1. FINAL NOTEBOOK 63 FINDINGS")
print("=" * 92)

print(
    json.dumps(
        NB63_FINAL_FINDINGS,
        indent=2,
    )
)


# =============================================================================
# 2. SCIENTIFIC INTERPRETATION
# =============================================================================

NB63_VALIDATED_CLAIMS = [

    "The corrected simulation campaign completed all 240 battles.",

    "No runtime failures occurred during the full campaign.",

    "The deployment policy produced zero Unknown Move actions.",

    "The deployment policy produced positive damage on every "
    "recorded campaign move.",

    "No full-campaign battle ended in timeout.",

    "Normal and swapped-side win rates were identical.",

    "The deployment runtime remained stable throughout evaluation.",
]


NB63_LIMITATIONS = [

    "The Notebook 63 simulator is a controlled attack-only environment.",

    "The campaign does not reproduce complete official Pokémon TCG gameplay.",

    "Synthetic battle states were used instead of full tournament deck states.",

    "The deployment policy did not outperform either deterministic baseline "
    "on campaign win rate.",

    "The deployment policy was behaviorally equivalent to FIRST_LEGAL_MOVE "
    "across all 80 paired controlled battles.",

    "MAX_DAMAGE_LEGAL_MOVE achieved substantially greater damage efficiency "
    "and completed battles in fewer turns.",

    "Therefore Notebook 63 supports runtime robustness, semantic correctness, "
    "side neutrality, and controlled behavior claims — not policy superiority.",
]


print()
print("=" * 92)
print("2. VALIDATED CLAIMS")
print("=" * 92)

for index, claim in enumerate(
    NB63_VALIDATED_CLAIMS,
    start=1,
):
    print(
        f"{index}. {claim}"
    )


print()
print("=" * 92)
print("3. LIMITATIONS")
print("=" * 92)

for index, limitation in enumerate(
    NB63_LIMITATIONS,
    start=1,
):
    print(
        f"{index}. {limitation}"
    )


# =============================================================================
# 3. EVIDENCE INVENTORY
# =============================================================================

NB63_EXPECTED_EVIDENCE = [

    NB63_ARTIFACT_DIR
    / "cell5_isolated_simulator_smoke_result.json",

    NB63_ARTIFACT_DIR
    / "cell6_deployment_policy_smoke_result.json",

    NB63_ARTIFACT_DIR
    / "cell7_campaign_design.json",

    NB63_ARTIFACT_DIR
    / "cell8_pilot_summary.json",

    NB63_ARTIFACT_DIR
    / "cell9_legal_move_contract_audit.json",

    NB63_ARTIFACT_DIR
    / "cell10_schema_correct_scenarios.json",

    NB63_ARTIFACT_DIR
    / "cell10_corrected_campaign_matrix.json",

    NB63_ARTIFACT_DIR
    / "cell10_schema_repair_audit.json",

    NB63_ARTIFACT_DIR
    / "cell11_corrected_pilot_records.json",

    NB63_ARTIFACT_DIR
    / "cell11_corrected_pilot_summary.json",

    NB63_ARTIFACT_DIR
    / "cell12b_full_campaign_records.json",

    NB63_ARTIFACT_DIR
    / "cell13_full_campaign_analysis.json",
]


NB63_EVIDENCE_INVENTORY = []


for path in NB63_EXPECTED_EVIDENCE:

    exists = path.is_file()

    NB63_EVIDENCE_INVENTORY.append({

        "path":
            str(path),

        "exists":
            exists,

        "bytes":
            (
                path.stat().st_size
                if exists
                else None
            ),

        "sha256":
            (
                sha256_file_nb63(
                    path
                )
                if exists
                else None
            ),
    })


missing_nb63_evidence = [
    record
    for record in NB63_EVIDENCE_INVENTORY
    if not record["exists"]
]


print()
print("=" * 92)
print("4. EVIDENCE INVENTORY")
print("=" * 92)


for record in NB63_EVIDENCE_INVENTORY:

    print(
        "[OK]"
        if record["exists"]
        else "[MISSING]",
        record["path"],
    )


assert not missing_nb63_evidence, (
    "Notebook 63 evidence package is incomplete."
)


print()
print(
    "[OK] All expected Notebook 63 evidence is present."
)


# =============================================================================
# 4. AUTHORITATIVE IDENTIFIERS
# =============================================================================

NB63_FINAL_IDENTIFIERS = {

    "original_campaign_sha256":
        NB63_CAMPAIGN_CONFIG_SHA256,

    "corrected_campaign_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "main_sha256":
        EXPECTED_MAIN_SHA256,

    "certified_rf_sha256":
        EXPECTED_MODEL_SHA256,

    "deployment_ppo_checkpoint_sha256":
        DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63,
}


print()
print("=" * 92)
print("5. AUTHORITATIVE IDENTIFIERS")
print("=" * 92)

print(
    json.dumps(
        NB63_FINAL_IDENTIFIERS,
        indent=2,
    )
)


# =============================================================================
# 5. TRACK 1 CONCLUSION
# =============================================================================

NB63_TRACK1_CONCLUSION = {

    "track":
        "SIMULATION_AND_EVIDENCE",

    "status":
        "COMPLETE",

    "runtime_validation":
        "PASS",

    "semantic_validation":
        "PASS",

    "side_neutrality_validation":
        "PASS",

    "full_campaign_execution":
        "PASS",

    "battle_count":
        240,

    "runtime_failures":
        0,

    "policy_superiority_proven":
        False,

    "deployment_first_legal_equivalence":
        True,

    "recommended_interpretation":
        (
            "Notebook 63 provides controlled runtime, semantic, "
            "side-neutrality, and behavioral evidence. "
            "It must not be used to claim deployment-policy superiority."
        ),
}


print()
print("=" * 92)
print("6. TRACK 1 CONCLUSION")
print("=" * 92)

print(
    json.dumps(
        NB63_TRACK1_CONCLUSION,
        indent=2,
    )
)


# =============================================================================
# 6. NOTEBOOK 64 HANDOFF
# =============================================================================

NB64_HANDOFF_CONTRACT = {

    "source_notebook":
        63,

    "target_notebook":
        64,

    "target_title":
        "MATCHUP_DECISION_STRATEGY_INTERPRETATION",

    "campaign_configuration_sha256":
        NB63_CORRECTED_CAMPAIGN_CONFIG_SHA256,

    "authoritative_campaign_analysis":
        str(
            NB63_REPORT_DIR
            / "notebook63_full_campaign_analysis.json"
        ),

    "authoritative_campaign_records":
        str(
            NB63_OUTPUT_DIR
            / "cell12b_full_campaign_records.json"
        ),

    "authoritative_corrected_scenarios":
        str(
            NB63_ARTIFACT_DIR
            / "cell10_schema_correct_scenarios.json"
        ),

    "key_runtime_result":
        (
            "240/240 controlled battles completed with zero runtime failures."
        ),

    "key_semantic_result":
        (
            "Unknown Move rate = 0.0 and positive-damage move rate = 1.0."
        ),

    "key_behavioral_result":
        (
            "Deployment policy behaviorally matched FIRST_LEGAL_MOVE "
            "in all 80 paired controlled battles."
        ),

    "important_limitation":
        (
            "Controlled attack-only synthetic environment; "
            "not full official Pokémon TCG gameplay."
        ),

    "notebook64_required_focus": [

        "matchup interpretation",

        "scenario-level decision behavior",

        "damage-efficiency comparison",

        "side-neutrality evidence",

        "deployment-policy limitations",

        "integration with NB57-NB62 certified evidence",

        "strategy-report evidence extraction",
    ],
}


print()
print("=" * 92)
print("7. NOTEBOOK 64 HANDOFF")
print("=" * 92)

print(
    json.dumps(
        NB64_HANDOFF_CONTRACT,
        indent=2,
    )
)


# =============================================================================
# 7. BUILD FINAL PACKAGE
# =============================================================================

NB63_FINAL_PACKAGE = {

    "notebook":
        63,

    "title":
        "FINAL_STRATEGY_SIMULATION_CAMPAIGN",

    "status":
        "COMPLETE",

    "final_findings":
        NB63_FINAL_FINDINGS,

    "validated_claims":
        NB63_VALIDATED_CLAIMS,

    "limitations":
        NB63_LIMITATIONS,

    "track1_conclusion":
        NB63_TRACK1_CONCLUSION,

    "identifiers":
        NB63_FINAL_IDENTIFIERS,

    "evidence_inventory":
        NB63_EVIDENCE_INVENTORY,

    "notebook64_handoff":
        NB64_HANDOFF_CONTRACT,
}


package_canonical_json_nb63 = json.dumps(
    NB63_FINAL_PACKAGE,
    sort_keys=True,
    separators=(
        ",",
        ":",
    ),
)


NB63_FINAL_PACKAGE_SHA256 = hashlib.sha256(
    package_canonical_json_nb63.encode(
        "utf-8"
    )
).hexdigest()


NB63_FINAL_PACKAGE[
    "final_package_sha256"
] = NB63_FINAL_PACKAGE_SHA256


print()
print("=" * 92)
print("8. FINAL PACKAGE SHA256")
print("=" * 92)

print(
    NB63_FINAL_PACKAGE_SHA256
)


# =============================================================================
# 8. PERSIST FINAL PACKAGE
# =============================================================================

NB63_FINAL_ARTIFACT = (
    NB63_ARTIFACT_DIR
    / "notebook63_final_evidence_package.json"
)


NB63_FINAL_OUTPUT = (
    NB63_OUTPUT_DIR
    / "notebook63_final_evidence_package.json"
)


NB63_FINAL_REPORT = (
    NB63_REPORT_DIR
    / "notebook63_final_evidence_package.json"
)


NB63_TO_NB64_HANDOFF = (
    NB63_ARTIFACT_DIR
    / "notebook63_to_notebook64_handoff.json"
)


for path in [
    NB63_FINAL_ARTIFACT,
    NB63_FINAL_OUTPUT,
    NB63_FINAL_REPORT,
]:

    path.write_text(
        json.dumps(
            NB63_FINAL_PACKAGE,
            indent=2,
        ),
        encoding="utf-8",
    )


NB63_TO_NB64_HANDOFF.write_text(
    json.dumps(
        NB64_HANDOFF_CONTRACT,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 92)
print("9. FINAL PACKAGE PERSISTENCE")
print("=" * 92)

print(
    "[OK]",
    NB63_FINAL_ARTIFACT,
)

print(
    "[OK]",
    NB63_FINAL_OUTPUT,
)

print(
    "[OK]",
    NB63_FINAL_REPORT,
)

print(
    "[OK]",
    NB63_TO_NB64_HANDOFF,
)


# =============================================================================
# 9. FINAL CERTIFIED-ASSET INTEGRITY
# =============================================================================

assert (
    sha256_file_nb63(
        FINAL_MAIN
    )
    ==
    EXPECTED_MAIN_SHA256
)


assert (
    sha256_file_nb63(
        CERTIFIED_MODEL
    )
    ==
    EXPECTED_MODEL_SHA256
)


assert (
    sha256_file_nb63(
        DEPLOYMENT_POLICY_CHECKPOINT_NB63
    )
    ==
    DEPLOYMENT_POLICY_CHECKPOINT_SHA256_NB63
)


print()
print("=" * 92)
print("10. FINAL CERTIFIED-ASSET INTEGRITY")
print("=" * 92)

print(
    "[OK] main.py unchanged."
)

print(
    "[OK] Certified RF model unchanged."
)

print(
    "[OK] Deployment PPO checkpoint unchanged."
)

print(
    "[OK] No training or retraining occurred."
)


# =============================================================================
# 10. FINAL STATUS
# =============================================================================

CELL14_NB63_STATUS = {

    "cell13_pass":
        True,

    "notebook63_complete":
        True,

    "track1_complete":
        True,

    "campaign_battle_count":
        240,

    "runtime_failure_count":
        0,

    "campaign_execution_valid":
        True,

    "campaign_semantics_valid":
        True,

    "deployment_runtime_stable":
        True,

    "policy_superiority_proven":
        False,

    "deployment_first_legal_equivalence_rate":
        CELL13_NB63_STATUS[
            "deployment_first_legal_equivalence_rate"
        ],

    "evidence_package_complete":
        True,

    "evidence_file_count":
        len(
            NB63_EVIDENCE_INVENTORY
        ),

    "final_package_sha256":
        NB63_FINAL_PACKAGE_SHA256,

    "notebook64_handoff_ready":
        True,

    "main_integrity_preserved":
        True,

    "certified_model_integrity_preserved":
        True,

    "deployment_checkpoint_integrity_preserved":
        True,

    "fit_calls_executed":
        0,

    "retraining_performed":
        False,

    "model_modified":
        False,

    "validation_status":
        "PASS",
}


print()
print("=" * 92)
print("NOTEBOOK 63 — CELL 14 STATUS: PASS")
print("=" * 92)

print(
    json.dumps(
        CELL14_NB63_STATUS,
        indent=2,
    )
)


print()
print("=" * 92)
print("NOTEBOOK 63 STATUS: COMPLETE")
print("TRACK 1 — SIMULATION / EVIDENCE: COMPLETE")
print("=" * 92)

print()
print(
    "Notebook 63 final evidence package created."
)

print(
    "Certified assets remain unchanged."
)

print(
    "Notebook 64 handoff is ready."
)


# In[ ]:




