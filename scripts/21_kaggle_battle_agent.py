#!/usr/bin/env python
# coding: utf-8

# # Notebook 21 — Kaggle Battle Agent
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook wraps the completed policy engine in a production-facing agent
# that can be called by the official Kaggle simulator.
# 
# ## Existing pipeline
# 
# ```text
# Official observation dictionary
#         ↓
# to_observation_class()
#         ↓
# Notebook 18 observation adapter
#         ↓
# BattleSnapshot
#         ↓
# Notebook 19 feature and scoring engine
#         ↓
# Notebook 20 BattlePolicy
#         ↓
# Best official option index

# ## Notebook 21 objectives
# Load the official cg runtime.
# Load the completed Notebook 20 policy engine.
# Build a Kaggle-compatible agent wrapper.
# Handle initial deck requests.
# Handle normal legal-action requests.
# Return lists of official option indices.
# Enforce minCount and maxCount.
# Prevent duplicate option selections.
# Add deterministic fallback behavior.
# Track timing, decisions, errors, and fallback usage.
# Expose a production agent(obs_dict) entry point.
# Export reusable code into src/kaggle_agent/.

# ## Target structure
# 
# src/kaggle_agent/
# ├── __init__.py
# ├── models.py
# ├── runtime.py
# ├── agent.py
# ├── validation.py
# └── notebook21_export.py

# ## Cell 2 — Imports

# In[1]:


from __future__ import annotations

import sys
import time
import types

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

print("Python:", sys.version)
print("Current directory:", Path.cwd())


# ## Cell 3 — Locate project paths

# In[2]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = [
        "src",
        "notebooks",
        "scripts",
        "data",
    ]

    for candidate in [current, *current.parents]:
        marker_count = sum(
            (candidate / marker).exists()
            for marker in markers
        )

        if marker_count >= 3:
            return candidate

    if current.name.lower() == "notebooks":
        return current.parent

    return current


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

NOTEBOOK20_EXPORT = (
    SCRIPTS_DIR
    / "20_policy_engine.py"
)

KAGGLE_AGENT_DIR = (
    SRC_DIR
    / "kaggle_agent"
)

NOTEBOOK21_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook21"
)

for directory in [
    KAGGLE_AGENT_DIR,
    NOTEBOOK21_REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Notebook 20 export:", NOTEBOOK20_EXPORT)
print("Kaggle agent package:", KAGGLE_AGENT_DIR)
print("Notebook 21 reports:", NOTEBOOK21_REPORT_DIR)


# ## Cell 4 — Load Notebook 20 safely
# ### Notebook 20 is an exported notebook script, use the same safe loading method that worked previously

# In[3]:


# Cell 4 — Load Notebook 20 and safely register nested notebook modules

import sys
import types
import uuid


if not NOTEBOOK20_EXPORT.is_file():
    raise FileNotFoundError(
        f"Notebook 20 export not found:\n{NOTEBOOK20_EXPORT}"
    )

source_20 = NOTEBOOK20_EXPORT.read_text(
    encoding="utf-8-sig"
)

source_20_lines = source_20.splitlines()

# Normalize repeated future imports.
cleaned_20_lines = [
    line
    for line in source_20_lines
    if line.strip() != "from __future__ import annotations"
]

cleaned_20_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_20_lines)
)

# Patch Notebook 20's nested importlib loaders.
# Dataclasses in Python 3.13 require the module to be registered
# in sys.modules before the module code executes.
cleaned_20_source = cleaned_20_source.replace(
    "notebook18 = importlib.util.module_from_spec(spec18)\n"
    "spec18.loader.exec_module(notebook18)",
    "notebook18 = importlib.util.module_from_spec(spec18)\n"
    "sys.modules[spec18.name] = notebook18\n"
    "spec18.loader.exec_module(notebook18)",
)

cleaned_20_source = cleaned_20_source.replace(
    "notebook19 = importlib.util.module_from_spec(spec19)\n"
    "spec19.loader.exec_module(notebook19)",
    "notebook19 = importlib.util.module_from_spec(spec19)\n"
    "sys.modules[spec19.name] = notebook19\n"
    "spec19.loader.exec_module(notebook19)",
)

module_name_20 = (
    "notebook20_policy_engine_"
    + uuid.uuid4().hex
)

notebook20 = types.ModuleType(module_name_20)
notebook20.__file__ = str(NOTEBOOK20_EXPORT)
notebook20.__package__ = ""

# Register Notebook 20 itself before execution.
sys.modules[module_name_20] = notebook20

compiled_20 = compile(
    cleaned_20_source,
    str(NOTEBOOK20_EXPORT),
    "exec",
)

exec(
    compiled_20,
    notebook20.__dict__,
)

print("\nNotebook 20 loaded from:")
print(notebook20.__file__)
print()
print(
    "PolicyDecision available:",
    hasattr(notebook20, "PolicyDecision"),
)
print(
    "BattlePolicy available:",
    hasattr(notebook20, "BattlePolicy"),
)
print(
    "Repository available:",
    hasattr(notebook20, "repository"),
)
print(
    "Official lookup available:",
    hasattr(
        notebook20,
        "official_card_data_by_id",
    ),
)


# ## Cell 5 — Retrieve Notebook 20 objects

# In[4]:


required_names = [
    "PolicyDecision",
    "BattlePolicy",
    "repository",
    "official_card_data_by_id",
]

missing_names = [
    name
    for name in required_names
    if not hasattr(notebook20, name)
]

if missing_names:
    raise AttributeError(
        "Notebook 20 is missing:\n"
        + "\n".join(missing_names)
    )

PolicyDecision = notebook20.PolicyDecision
BattlePolicy = notebook20.BattlePolicy
repository = notebook20.repository

official_card_data_by_id = (
    notebook20.official_card_data_by_id
)

print("PolicyDecision:", PolicyDecision)
print("BattlePolicy:", BattlePolicy)
print("Repository size:", len(repository))
print(
    "Official lookup size:",
    len(official_card_data_by_id),
)

assert len(repository) == 1267
assert len(official_card_data_by_id) == 1267

print("\nNotebook 20 production objects loaded successfully.")


# ## Cell 6 — Verify the final policy class

# In[5]:


required_policy_methods = [
    "decide",
    "decide_snapshot",
    "choose_action",
    "choose_snapshot_action",
    "print_decision",
]

missing_methods = [
    name
    for name in required_policy_methods
    if not hasattr(BattlePolicy, name)
]

for name in required_policy_methods:
    status = (
        "[OK]"
        if hasattr(BattlePolicy, name)
        else "[MISSING]"
    )
    print(status, name)

if missing_methods:
    raise AttributeError(
        "BattlePolicy is missing methods:\n"
        + "\n".join(missing_methods)
    )

print("\nFinal BattlePolicy version verified.")


# ## Cell 7 — Instantiate the Notebook 21 policy

# In[6]:


policy = BattlePolicy(
    repository=repository,
    official_lookup=official_card_data_by_id,
    debug=False,
    fallback_index=0,
)

assert callable(policy.decide)
assert callable(policy.choose_action)

print("Notebook 21 policy instance created successfully.")


# ## Cell 8 — Create the Kaggle Agent wrapper

# In[7]:


from typing import Any


def agent(observation: Any) -> int:
    """
    Kaggle entry point.

    Receives an official Kaggle Observation object and
    returns the selected legal action index.
    """

    return policy.choose_action(observation)


print("Kaggle agent created.")


# ## Cell 9 — Verify the wrapper

# In[8]:


assert callable(agent)

print("Agent wrapper verified.")


# ## Cell 10 — Test using our synthetic snapshot
# 
# ### Unlike Notebook 20, Notebook 21 uses the production wrapper.

# In[9]:


# Cell 10 — Safely retrieve and test the synthetic snapshot

if not hasattr(notebook20, "sample_snapshot"):
    raise AttributeError(
        "Notebook 20 does not expose sample_snapshot."
    )

sample_snapshot = notebook20.sample_snapshot

if sample_snapshot.selection is None:
    raise ValueError(
        "Synthetic snapshot has no selection data."
    )

best = policy.choose_snapshot_action(
    sample_snapshot
)

print("Turn:", sample_snapshot.turn)
print(
    "Legal options:",
    len(sample_snapshot.selection.options),
)
print("Best option:", best)

assert best == 0

print("\nSnapshot inference passed.")


# ## Cell 11 — Build a production agent result model

# In[10]:


from dataclasses import dataclass


@dataclass(frozen=True)
class AgentStats:
    calls: int = 0
    errors: int = 0
    fallbacks: int = 0
    total_seconds: float = 0.0

    @property
    def average_seconds(self) -> float:
        if self.calls == 0:
            return 0.0

        return self.total_seconds / self.calls


# ## Cell 12 — Build the KaggleBattleAgent wrapper

# In[11]:


@dataclass
class KaggleBattleAgent:
    policy: BattlePolicy
    deck: tuple[int, ...]
    debug: bool = False

    calls: int = 0
    errors: int = 0
    fallbacks: int = 0
    total_seconds: float = 0.0

    def choose(self, observation: Any) -> list[int]:
        """
        Return a simulator-compatible list.

        Supports:
        1. An adapted BattleSnapshot for testing.
        2. An official initial deck request.
        3. A normal official Observation.
        """

        started = time.perf_counter()
        self.calls += 1

        try:
            # Adapted BattleSnapshot test path.
            if hasattr(observation, "selection"):
                decision = self.policy.decide_snapshot(
                    observation
                )

            # Official initial deck request.
            elif getattr(observation, "select", None) is None:
                return list(self.deck)

            # Normal official Observation.
            else:
                decision = self.policy.decide(
                    observation
                )

            if decision.used_fallback:
                self.fallbacks += 1

            result = [
                int(decision.option_index)
            ]

            if self.debug:
                print("Agent result:", result)
                print("Reason:", decision.reason)

            return result

        except Exception as exc:
            self.errors += 1
            self.fallbacks += 1

            if self.debug:
                print(
                    "Agent error:",
                    f"{type(exc).__name__}: {exc}",
                )

            return [0]

        finally:
            self.total_seconds += (
                time.perf_counter() - started
            )

    def stats(self) -> AgentStats:
        return AgentStats(
            calls=self.calls,
            errors=self.errors,
            fallbacks=self.fallbacks,
            total_seconds=self.total_seconds,
        )


# ## Cell 13 — Load the 60-card deck

# In[12]:


from pathlib import Path


DECK_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle_sample_submission"
    / "deck.csv"
)

if not DECK_FILE.is_file():
    DECK_FILE = (
        PROJECT_ROOT
        / "submission_work"
        / "team_jesus_baseline"
        / "deck.csv"
    )

deck_ids = tuple(
    int(line.strip())
    for line in DECK_FILE.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
)

print("Deck file:", DECK_FILE)
print("Deck size:", len(deck_ids))

assert len(deck_ids) == 60

print("Deck loaded successfully.")


# ## Cell 14 — Instantiate the production agent

# In[13]:


kaggle_agent = KaggleBattleAgent(
    policy=policy,
    deck=deck_ids,
    debug=True,
)

print("KaggleBattleAgent created.")
print("Deck size:", len(kaggle_agent.deck))


# ## Cell 15 — Test deck submission mode

# In[14]:


class DeckRequest:
    select = None

deck_request = DeckRequest()

deck_response = kaggle_agent.choose(deck_request)

print("Returned deck size:", len(deck_response))

assert len(deck_response) == 60
assert deck_response == list(deck_ids)

print("Deck submission passed.")


# ## Cell 16 — Test action-selection mode
# 
# ### Now verify that the production wrapper chooses the same action as the policy.

# In[15]:


chosen = kaggle_agent.choose(sample_snapshot)

print("Returned action:", chosen)

assert chosen == [0]

print("Action selection passed.")


# ## Cell 17 — Check runtime statistics

# In[16]:


stats = kaggle_agent.stats()

print(stats)
print("Calls:", stats.calls)
print("Errors:", stats.errors)
print("Fallbacks:", stats.fallbacks)
print("Total seconds:", stats.total_seconds)
print("Average seconds:", stats.average_seconds)

assert stats.calls >= 2
assert stats.errors == 0
assert stats.fallbacks == 0
assert stats.average_seconds >= 0.0

print("\nAgent statistics passed.")


# ## Cell 18 — Validate the production interface

# In[17]:


notebook21_checks = {
    "policy_loaded": callable(policy.choose_action),
    "wrapper_callable": callable(agent),
    "production_agent_created": isinstance(
        kaggle_agent,
        KaggleBattleAgent,
    ),
    "deck_size": len(kaggle_agent.deck) == 60,
    "deck_submission": deck_response == list(deck_ids),
    "action_selection": chosen == [0],
    "no_runtime_errors": kaggle_agent.errors == 0,
}

for check, passed in notebook21_checks.items():
    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{check}"
    )

assert all(notebook21_checks.values())

print("\nNotebook 21 validation passed.")


# ## Cell 19 — Create the final callable entry point
# 
# #### The competition sample expects a list of indices, not a single integer.

# In[18]:


def production_agent(observation: Any) -> list[int]:
    """
    Final Kaggle-facing agent entry point.
    """

    return kaggle_agent.choose(observation)


assert callable(production_agent)

print("Production agent entry point created.")


# ## Cell 20 — Final summary

# In[19]:


print("=" * 72)
print("Notebook 21 — Kaggle Battle Agent")
print("=" * 72)
print("Repository cards:", len(repository))
print("Official cards:", len(official_card_data_by_id))
print("Deck size:", len(deck_ids))
print("Agent calls:", kaggle_agent.calls)
print("Agent errors:", kaggle_agent.errors)
print("Agent fallbacks:", kaggle_agent.fallbacks)
print("Deck path validated:", len(deck_response) == 60)
print("Action path validated:", chosen == [0])
print()
print("NOTEBOOK 21 COMPLETED SUCCESSFULLY")
print("Ready for PowerShell export and packaging.")


# In[ ]:




