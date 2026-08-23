#!/usr/bin/env python
# coding: utf-8

# # Notebook 22 — End-to-End Agent Integration and Evaluation Harness
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 21 produced the Kaggle-facing battle agent.
# 
# Notebook 22 validates that the entire production chain can be loaded, exercised,
# measured, and monitored as one integrated system.
# 
# ## Integrated pipeline
# 
# ```text
# Official observation or adapted snapshot
#                 ↓
# KaggleBattleAgent
#                 ↓
# BattlePolicy
#                 ↓
# Observation adapter
#                 ↓
# BattleSnapshot
#                 ↓
# Battle and player feature extraction
#                 ↓
# Legal-action ranking
#                 ↓
# Official option indices

# ## Notebook Objectives
# Load the Notebook 21 production agent.
# Verify its dependencies and public interface.
# Reuse the validated 60-card deck.
# Test deck-request handling.
# Test action-selection handling.
# Test safe fallback behavior.
# Test repeated inference.
# Benchmark latency and throughput.
# Capture decision and error statistics.
# Build an evaluation harness for later replay and self-play experiments.
# Export reusable integration code.
# Save Notebook 22 through PowerShell.

# ## Cell 2 — Imports

# In[1]:


from __future__ import annotations

import sys
import time
import types
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

print("Python:", sys.version)
print("Working directory:", Path.cwd())


# ## Cell 3 — Locate the project

# In[2]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    required_markers = {
        "notebooks",
        "scripts",
        "src",
        "data",
    }

    for candidate in [current, *current.parents]:
        found = {
            marker
            for marker in required_markers
            if (candidate / marker).exists()
        }

        if len(found) >= 3:
            return candidate

    if current.name.casefold() == "notebooks":
        return current.parent

    return current


PROJECT_ROOT = find_project_root()

NOTEBOOK21_EXPORT = (
    PROJECT_ROOT
    / "scripts"
    / "21_kaggle_battle_agent.py"
)

NOTEBOOK22_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook22"
)

NOTEBOOK22_PACKAGE_DIR = (
    PROJECT_ROOT
    / "src"
    / "evaluation_harness"
)

for directory in [
    NOTEBOOK22_REPORT_DIR,
    NOTEBOOK22_PACKAGE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Notebook 21 export:", NOTEBOOK21_EXPORT)
print("Notebook 22 reports:", NOTEBOOK22_REPORT_DIR)
print("Evaluation package:", NOTEBOOK22_PACKAGE_DIR)


# ## Cell 4 — Confirm Notebook 21 export exists

# In[3]:


if not NOTEBOOK21_EXPORT.is_file():
    raise FileNotFoundError(
        "Notebook 21 export was not found:\n"
        f"{NOTEBOOK21_EXPORT}"
    )

print("Notebook 21 export located.")
print("Size:", NOTEBOOK21_EXPORT.stat().st_size, "bytes")


# ## Cell 5 — Load Notebook 21 safely

# In[4]:


source_21 = NOTEBOOK21_EXPORT.read_text(
    encoding="utf-8-sig"
)

source_21_lines = source_21.splitlines()

# Remove exported future imports and restore exactly one.
cleaned_21_lines = [
    line
    for line in source_21_lines
    if line.strip() != "from __future__ import annotations"
]

cleaned_21_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_21_lines)
)

# Patch dynamic modules created from specs so Python 3.13
# dataclass processing can resolve module namespaces.
module_variable_names = [
    "notebook18",
    "notebook19",
    "notebook20",
]

for variable_name in module_variable_names:
    spec_name = {
        "notebook18": "spec18",
        "notebook19": "spec19",
        "notebook20": "spec20",
    }[variable_name]

    original = (
        f"{variable_name} = "
        f"importlib.util.module_from_spec({spec_name})\n"
        f"{spec_name}.loader.exec_module({variable_name})"
    )

    replacement = (
        f"{variable_name} = "
        f"importlib.util.module_from_spec({spec_name})\n"
        f"sys.modules[{spec_name}.name] = {variable_name}\n"
        f"{spec_name}.loader.exec_module({variable_name})"
    )

    cleaned_21_source = cleaned_21_source.replace(
        original,
        replacement,
    )

module_name_21 = (
    "notebook21_kaggle_agent_"
    + uuid.uuid4().hex
)

notebook21 = types.ModuleType(module_name_21)
notebook21.__file__ = str(NOTEBOOK21_EXPORT)
notebook21.__package__ = ""

sys.modules[module_name_21] = notebook21

compiled_21 = compile(
    cleaned_21_source,
    str(NOTEBOOK21_EXPORT),
    "exec",
)

exec(
    compiled_21,
    notebook21.__dict__,
)

print("\nNotebook 21 loaded successfully.")


# ## Cell 6 — Inspect required production objects

# In[5]:


REQUIRED_NOTEBOOK21_OBJECTS = [
    "AgentStats",
    "KaggleBattleAgent",
    "kaggle_agent",
    "production_agent",
    "policy",
    "deck_ids",
    "sample_snapshot",
]

missing_objects = [
    name
    for name in REQUIRED_NOTEBOOK21_OBJECTS
    if not hasattr(notebook21, name)
]

for name in REQUIRED_NOTEBOOK21_OBJECTS:
    print(
        f"{'[OK]' if hasattr(notebook21, name) else '[MISSING]'} "
        f"{name}"
    )

if missing_objects:
    raise AttributeError(
        "Notebook 21 export is missing:\n"
        + "\n".join(missing_objects)
    )

print("\nNotebook 21 production interface verified.")


# ## Cell 7 — Retrieve the production objects

# In[6]:


AgentStats = notebook21.AgentStats
KaggleBattleAgent = notebook21.KaggleBattleAgent

kaggle_agent = notebook21.kaggle_agent
policy = notebook21.policy

production_agent = notebook21.production_agent

sample_snapshot = notebook21.sample_snapshot
deck_ids = notebook21.deck_ids

print("Production objects loaded.")

print("Deck size:", len(deck_ids))
print("Agent type:", type(kaggle_agent).__name__)
print("Policy type:", type(policy).__name__)

assert len(deck_ids) == 60

print("\nNotebook 21 objects imported successfully.")


# ## Cell 8 — Verify the production agent interface

# In[7]:


required_methods = [
    "choose",
    "stats",
]

missing = [
    method
    for method in required_methods
    if not hasattr(kaggle_agent, method)
]

for method in required_methods:
    print(
        "[OK]" if hasattr(kaggle_agent, method)
        else "[MISSING]",
        method,
    )

assert not missing

print("\nProduction agent interface verified.")


# ## Cell 9 — Test deck submission again

# In[8]:


class DeckRequest:
    select = None

deck_request = DeckRequest()

returned_deck = kaggle_agent.choose(deck_request)

print("Returned cards:", len(returned_deck))

assert isinstance(returned_deck, list)
assert len(returned_deck) == 60
assert returned_deck == list(deck_ids)

print("\nDeck submission verified.")


# ## Cell 10 — Test action selection again

# In[9]:


chosen = kaggle_agent.choose(sample_snapshot)

print("Chosen:", chosen)

assert chosen == [0]

print("\nAction selection verified.")


# # Cell 11 — Stress test the production agent

# In[10]:


results = []

for i in range(25):
    result = kaggle_agent.choose(sample_snapshot)

    print(
        f"Run {i+1:02d}:",
        result,
    )

    results.append(result)

assert all(r == [0] for r in results)

print()
print("Repeated inference passed.")
print("Calls:", kaggle_agent.calls)
print("Errors:", kaggle_agent.errors)
print("Fallbacks:", kaggle_agent.fallbacks)


# # Cell 12 — Validate AgentStats

# In[11]:


stats = kaggle_agent.stats()

print(stats)
print("Calls:", stats.calls)
print("Errors:", stats.errors)
print("Fallbacks:", stats.fallbacks)
print("Total seconds:", stats.total_seconds)
print("Average seconds:", stats.average_seconds)

assert stats.calls >= 29
assert stats.errors == 0
assert stats.fallbacks == 0
assert stats.total_seconds >= 0.0
assert stats.average_seconds >= 0.0

print("\nAgentStats validation passed.")


# # Cell 13 — Build the evaluation harness

# In[12]:


@dataclass(frozen=True)
class EvaluationRecord:
    run_index: int
    result: tuple[int, ...]
    elapsed_seconds: float
    valid: bool


@dataclass
class AgentEvaluationHarness:
    agent: KaggleBattleAgent

    def evaluate(
        self,
        observation: Any,
        runs: int = 100,
        expected: Sequence[int] | None = None,
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []

        for run_index in range(1, runs + 1):
            started = time.perf_counter()

            result = self.agent.choose(observation)

            elapsed = time.perf_counter() - started

            valid = (
                True
                if expected is None
                else list(result) == list(expected)
            )

            records.append(
                EvaluationRecord(
                    run_index=run_index,
                    result=tuple(result),
                    elapsed_seconds=elapsed,
                    valid=valid,
                )
            )

        return records


# ## Cell 14 — Run the harness

# In[13]:


kaggle_agent.debug = False

harness = AgentEvaluationHarness(
    agent=kaggle_agent
)

evaluation_records = harness.evaluate(
    sample_snapshot,
    runs=100,
    expected=[0],
)

valid_count = sum(
    record.valid
    for record in evaluation_records
)

elapsed_values = [
    record.elapsed_seconds
    for record in evaluation_records
]

print("Runs:", len(evaluation_records))
print("Valid:", valid_count)
print("Invalid:", len(evaluation_records) - valid_count)
print("Minimum seconds:", min(elapsed_values))
print("Maximum seconds:", max(elapsed_values))
print(
    "Average seconds:",
    sum(elapsed_values) / len(elapsed_values),
)

assert valid_count == 100

print("\nEvaluation harness passed.")


# ### Above my Benchmark result is interpreted as: 
# 
# | Metric  | Result      | Status       |
# | ------- | ----------- | -----------  |
# | Runs    | 100         | ✅           |
# | Valid   | 100         | ✅           |
# | Invalid | 0           | ✅           |
# | Minimum | 39.2 μs     | ✅           |
# | Maximum | 114.3 μs    | ✅           |
# | Average | **42.8 μs** | ⭐ Excellent |
# 
# ##### An average inference time of ~43 microseconds for a policy decision is extremely fast and means your wrapper is adding virtually no overhead.
# 

# ## Cell 15 — Validate the complete Notebook 22

# In[14]:


notebook22_checks = {
    "production_interface": True,
    "stress_test": len(results) == 25,
    "stress_errors": kaggle_agent.errors == 0,
    "stress_fallbacks": kaggle_agent.fallbacks == 0,
    "agent_stats": stats.calls >= 29,
    "evaluation_runs": len(evaluation_records) == 100,
    "evaluation_valid": valid_count == 100,
    "evaluation_invalid": len(evaluation_records) - valid_count == 0,
}

for name, passed in notebook22_checks.items():
    print(
        f"{'[OK]' if passed else '[FAIL]'} {name}"
    )

assert all(notebook22_checks.values())

print("\nNotebook 22 validation passed.")


# ## Cell 16 — Final Summary

# In[16]:


print("=" * 72)
print("Notebook 22 — Production Evaluation")
print("=" * 72)

repository = notebook21.repository

official_card_data_by_id = (
    notebook21.official_card_data_by_id
)

print("Repository cards:", len(repository))
print(
    "Official cards:",
    len(official_card_data_by_id),
)
print("Deck size:", len(deck_ids))

print()

print("Policy calls:", kaggle_agent.calls)
print("Policy errors:", kaggle_agent.errors)
print("Policy fallbacks:", kaggle_agent.fallbacks)

print()

print("Stress test:", len(results))
print(
    "Evaluation runs:",
    len(evaluation_records),
)

average_inference = (
    sum(elapsed_values)
    / len(elapsed_values)
)

print("Average inference:", average_inference)

print()

print("NOTEBOOK 22 COMPLETED SUCCESSFULLY")
print("Ready for PowerShell export.")


# ## Above Final Validation Summary 
# 
# | Check                               | Status                         |
# | ----------------------------------- | ------------------------------ |
# | Notebook 21 imported successfully   | ✅                              |
# | Repository loaded (1267 cards)      | ✅                              |
# | Official lookup loaded (1267 cards) | ✅                              |
# | KaggleBattleAgent instantiated      | ✅                              |
# | Deck submission                     | ✅                              |
# | Action selection                    | ✅                              |
# | Stress test (25/25)                 | ✅                              |
# | Evaluation harness (100/100)        | ✅                              |
# | Policy errors                       | **0** ✅                        |
# | Policy fallbacks                    | **0** ✅                        |
# | Average inference                   | **4.28 × 10⁻⁵ s (≈42.8 μs)** ✅ |
# | Notebook 22 completed               | ✅                              |
# 
# 
# ##### An average inference time of about 43 microseconds indicates the evaluation wrapper is extremely lightweight.

# In[ ]:




