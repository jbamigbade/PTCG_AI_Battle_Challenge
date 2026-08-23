#!/usr/bin/env python
# coding: utf-8

# # Notebook 26 — Tournament Simulation and Policy Benchmarking
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 25 completed the final submission audit.
# 
# Notebook 26 begins the competitive-testing phase by repeatedly evaluating the production agent across battle scenarios.
# 
# ## Objectives
# 
# 1. Load the verified production agent from Notebook 21.
# 2. Confirm the production interface.
# 3. Create a tournament-result data model.
# 4. Run repeated policy decisions.
# 5. Measure action consistency.
# 6. Measure inference latency.
# 7. Track errors and fallbacks.
# 8. Validate that every returned option is legal.
# 9. Produce scenario-level tournament statistics.
# 10. Save benchmark results for future comparison.
#     

# # Cell 2 — Imports

# In[2]:


from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
import time
import types
import uuid

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

print("Python:", sys.version)
print("Working directory:", Path.cwd())


# # Cell 3 — Locate project files

# In[4]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = {
        "notebooks",
        "scripts",
        "src",
        "reports",
        "submission_work",
    }

    for candidate in [current, *current.parents]:
        found = {
            marker
            for marker in markers
            if (candidate / marker).exists()
        }

        if len(found) >= 4:
            return candidate

    return current


PROJECT_ROOT = find_project_root()

AGENT_EXPORT = (
    PROJECT_ROOT
    / "src"
    / "kaggle_agent"
    / "notebook21_export.py"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook26"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Agent export:", AGENT_EXPORT)
print("Report directory:", REPORT_DIR)

assert AGENT_EXPORT.is_file()


# # Cell 4 — Load the production agent safely

# In[6]:


source_21 = AGENT_EXPORT.read_text(
    encoding="utf-8-sig"
)

cleaned_lines = [
    line
    for line in source_21.splitlines()
    if line.strip() != "from __future__ import annotations"
]

cleaned_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_lines)
)

module_name = (
    "notebook26_agent_"
    + uuid.uuid4().hex
)

notebook21 = types.ModuleType(module_name)

notebook21.__file__ = str(AGENT_EXPORT)
notebook21.__package__ = ""

sys.modules[module_name] = notebook21

compiled_agent = compile(
    cleaned_source,
    str(AGENT_EXPORT),
    "exec",
)

exec(
    compiled_agent,
    notebook21.__dict__,
)

print("Notebook 21 production agent loaded.")


# # Cell 5 — Retrieve and verify production objects

# In[7]:


REQUIRED_AGENT_OBJECTS = [
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
    for name in REQUIRED_AGENT_OBJECTS
    if not hasattr(notebook21, name)
]

for name in REQUIRED_AGENT_OBJECTS:
    print(
        f"{'[OK]' if hasattr(notebook21, name) else '[MISSING]'} "
        f"{name}"
    )

if missing_objects:
    raise AttributeError(
        "Notebook 21 export is missing:\n"
        + "\n".join(missing_objects)
    )

KaggleBattleAgent = notebook21.KaggleBattleAgent
kaggle_agent = notebook21.kaggle_agent
production_agent = notebook21.production_agent
policy = notebook21.policy
deck_ids = notebook21.deck_ids
sample_snapshot = notebook21.sample_snapshot

assert len(deck_ids) == 60
assert callable(production_agent)
assert callable(kaggle_agent.choose)

print("\nProduction agent interface verified.")


# # Cell 6 — Define the tournament result model

# In[8]:


@dataclass(frozen=True)
class TournamentRecord:
    run_number: int
    returned_action: tuple[int, ...]
    expected_action: tuple[int, ...]
    valid: bool
    elapsed_seconds: float
    agent_calls: int
    agent_errors: int
    agent_fallbacks: int


# # Cell 7 — Reset agent statistics

# In[9]:


kaggle_agent.calls = 0
kaggle_agent.errors = 0
kaggle_agent.fallbacks = 0

print("Agent statistics reset.")
print("Calls:", kaggle_agent.calls)
print("Errors:", kaggle_agent.errors)
print("Fallbacks:", kaggle_agent.fallbacks)


# # Cell 8 — Run the first 100-decision tournament benchmark

# In[11]:


EXPECTED_ACTION = (0,)
TOURNAMENT_RUNS = 100

tournament_records = []

for run_number in range(1, TOURNAMENT_RUNS + 1):
    started = time.perf_counter()

    returned = kaggle_agent.choose(
        sample_snapshot
    )

    elapsed = time.perf_counter() - started

    returned_action = tuple(returned)
    valid = returned_action == EXPECTED_ACTION

    tournament_records.append(
        TournamentRecord(
            run_number=run_number,
            returned_action=returned_action,
            expected_action=EXPECTED_ACTION,
            valid=valid,
            elapsed_seconds=elapsed,
            agent_calls=kaggle_agent.calls,
            agent_errors=kaggle_agent.errors,
            agent_fallbacks=kaggle_agent.fallbacks,
        )
    )

print("Tournament runs:", len(tournament_records))
print(
    "Valid decisions:",
    sum(record.valid for record in tournament_records),
)
print(
    "Invalid decisions:",
    sum(not record.valid for record in tournament_records),
)
print("Agent calls:", kaggle_agent.calls)
print("Agent errors:", kaggle_agent.errors)
print("Agent fallbacks:", kaggle_agent.fallbacks)

assert len(tournament_records) == TOURNAMENT_RUNS
assert all(record.valid for record in tournament_records)
assert kaggle_agent.errors == 0
assert kaggle_agent.fallbacks == 0

print("\nFirst tournament benchmark passed.")


# # Cell 9 — Disable debug and calculate latency statistics

# In[12]:


kaggle_agent.debug = False

elapsed_values = [
    record.elapsed_seconds
    for record in tournament_records
]

latency_summary = {
    "runs": len(elapsed_values),
    "minimum_seconds": min(elapsed_values),
    "maximum_seconds": max(elapsed_values),
    "average_seconds": statistics.mean(elapsed_values),
    "median_seconds": statistics.median(elapsed_values),
    "stdev_seconds": (
        statistics.stdev(elapsed_values)
        if len(elapsed_values) > 1
        else 0.0
    ),
}

for name, value in latency_summary.items():
    print(f"{name}: {value}")

assert latency_summary["runs"] == 100
assert latency_summary["average_seconds"] > 0

print("\nLatency statistics calculated.")


# # Cell 10 — Build the benchmark summary

# In[13]:


valid_count = sum(
    record.valid
    for record in tournament_records
)

benchmark_summary = {
    "project": "PTCG AI Battle Challenge",
    "team": "Team Jesus",
    "scenario": "Synthetic Mega Lucario ex attack scenario",
    "runs": len(tournament_records),
    "valid_decisions": valid_count,
    "invalid_decisions": (
        len(tournament_records) - valid_count
    ),
    "accuracy": (
        valid_count / len(tournament_records)
    ),
    "agent_calls": kaggle_agent.calls,
    "agent_errors": kaggle_agent.errors,
    "agent_fallbacks": kaggle_agent.fallbacks,
    "latency": latency_summary,
}

print(json.dumps(benchmark_summary, indent=4))

assert benchmark_summary["accuracy"] == 1.0
assert benchmark_summary["agent_errors"] == 0
assert benchmark_summary["agent_fallbacks"] == 0

print("\nBenchmark summary validated.")


# # Cell 11 — Save detailed tournament results

# In[14]:


TOURNAMENT_JSON = (
    REPORT_DIR
    / "tournament_benchmark.json"
)

records_payload = [
    asdict(record)
    for record in tournament_records
]

output_payload = {
    "summary": benchmark_summary,
    "records": records_payload,
}

TOURNAMENT_JSON.write_text(
    json.dumps(
        output_payload,
        indent=4,
    ),
    encoding="utf-8",
)

print("Tournament report:", TOURNAMENT_JSON)

assert TOURNAMENT_JSON.is_file()

print("\nTournament results saved.")


# # Cell 12 — Create a leaderboard DataFrame

# In[16]:


import pandas as pd

leaderboard = pd.DataFrame([
    {
        "Run": r.run_number,
        "Action": r.returned_action,
        "Correct": r.valid,
        "Latency (ms)": r.elapsed_seconds * 1000,
    }
    for r in tournament_records
])

display(leaderboard.head())

print()
print(leaderboard.describe(include="all"))


# # Cell 13 — Stress Test (1000 games)

# In[17]:


STRESS_RUNS = 1000

stress_results = []

for _ in range(STRESS_RUNS):
    action = kaggle_agent.choose(sample_snapshot)
    stress_results.append(tuple(action))

success_rate = (
    sum(a == (0,) for a in stress_results)
    / STRESS_RUNS
)

print("Stress Runs:", STRESS_RUNS)
print("Success Rate:", success_rate)

assert success_rate == 1.0

print("\n1000-game stress test passed.")


# # Cell 14 — Final Notebook Summary

# In[18]:


print("=" * 72)
print("Notebook 26 — Tournament Benchmark")
print("=" * 72)

print()

print("Project:", benchmark_summary["project"])
print("Team:", benchmark_summary["team"])

print()

print("Tournament Runs:", benchmark_summary["runs"])
print("Accuracy:", benchmark_summary["accuracy"])
print("Errors:", benchmark_summary["agent_errors"])
print("Fallbacks:", benchmark_summary["agent_fallbacks"])

print()

print("Average Latency:",
      latency_summary["average_seconds"] * 1000,
      "ms")

print()

print("Tournament Report:",
      TOURNAMENT_JSON.name)

print()

print("NOTEBOOK 26 COMPLETED SUCCESSFULLY")


# In[ ]:




