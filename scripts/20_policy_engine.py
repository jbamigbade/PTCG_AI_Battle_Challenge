#!/usr/bin/env python
# coding: utf-8

# # Notebook 20 — Policy Engine & Official Game Integration
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook builds the production decision engine that converts an official
# Kaggle `Observation` into the best legal official action index.
# 
# ## Pipeline
# 
# ```text
# Observation dictionary
#         ↓
# to_observation_class()
#         ↓
# BattleSnapshot
#         ↓
# Battle features
#         ↓
# Legal-action ranking
#         ↓
# Policy safety checks
#         ↓
# Official option index

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import importlib.util
import sys
import types

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

print("Python:", sys.version)
print("Current directory:", Path.cwd())


# # Cell 3 — Locate the project

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

NOTEBOOK18_EXPORT = (
    SCRIPTS_DIR
    / "18_kaggle_observation_adapter.py"
)

NOTEBOOK19_EXPORT = (
    SCRIPTS_DIR
    / "19_battle_feature_extraction.py"
)

POLICY_ENGINE_DIR = (
    SRC_DIR
    / "policy_engine"
)

NOTEBOOK20_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook20"
)

for directory in [
    POLICY_ENGINE_DIR,
    NOTEBOOK20_REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Notebook 18 export:", NOTEBOOK18_EXPORT)
print("Notebook 19 export:", NOTEBOOK19_EXPORT)
print("Policy engine:", POLICY_ENGINE_DIR)
print("Notebook 20 reports:", NOTEBOOK20_REPORT_DIR)


# # Cell 4 — Load Notebook 18

# In[4]:


# Cell 4 — Load Notebook 18 safely

import sys
import types


if not NOTEBOOK18_EXPORT.is_file():
    raise FileNotFoundError(
        f"Notebook 18 export not found:\n{NOTEBOOK18_EXPORT}"
    )

module_name_18 = "notebook18_adapter"

source_18 = NOTEBOOK18_EXPORT.read_text(
    encoding="utf-8-sig"
)

source_18_lines = source_18.splitlines()

cleaned_18_lines = [
    line
    for line in source_18_lines
    if line.strip() != "from __future__ import annotations"
]

cleaned_18_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_18_lines)
)

notebook18 = types.ModuleType(module_name_18)
notebook18.__file__ = str(NOTEBOOK18_EXPORT)
notebook18.__package__ = ""

sys.modules[module_name_18] = notebook18

compiled_18 = compile(
    cleaned_18_source,
    str(NOTEBOOK18_EXPORT),
    "exec",
)

exec(
    compiled_18,
    notebook18.__dict__,
)

print("Notebook 18 loaded successfully.")
print(
    "Repeated future imports removed:",
    len(source_18_lines) - len(cleaned_18_lines),
)


# # Cell 5 — Load Notebook 19

# In[8]:


spec19 = importlib.util.spec_from_file_location(
    "notebook19_features",
    NOTEBOOK19_EXPORT,
)

notebook19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(notebook19)

print("Notebook 19 loaded.")


# # Cell 6 — Import the production functions

# In[9]:


adapt_observation = notebook18.adapt_observation

extract_player_features = notebook19.extract_player_features
extract_battle_features = notebook19.extract_battle_features

rank_legal_actions = notebook19.rank_legal_actions
choose_best_option_index = notebook19.choose_best_option_index

print("Production functions imported.")


# # Cell 7 — Verify everything

# In[10]:


print(callable(adapt_observation))
print(callable(extract_player_features))
print(callable(extract_battle_features))
print(callable(rank_legal_actions))
print(callable(choose_best_option_index))

assert callable(adapt_observation)
assert callable(rank_legal_actions)
assert callable(choose_best_option_index)

print("\nNotebook dependencies verified.")


# # Cell 8 — Build the BattlePolicy class

# In[11]:


from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BattlePolicy:
    repository: Any
    official_lookup: dict[int, Any]
    debug: bool = False

    def choose_action(self, observation):
        """
        Convert an official Kaggle Observation into the
        best legal official option index.
        """

        snapshot = adapt_observation(
            observation,
            repository=self.repository,
            official_lookup=self.official_lookup,
        )

        ranked = rank_legal_actions(
            snapshot,
            repository=self.repository,
        )

        if self.debug:

            print("=" * 70)
            print("Ranked legal actions")
            print("=" * 70)

            for action in ranked:
                print(action)

        return choose_best_option_index(
            snapshot,
            repository=self.repository,
        )


# # Cell 9A — Retrieve shared dependencies:

# In[13]:


from src.card_database import load_repository

repository = load_repository()

official_card_data_by_id = getattr(
    notebook18,
    "official_card_data_by_id",
    None,
)

if official_card_data_by_id is None:
    official_cards = notebook18.all_card_data()

    official_card_data_by_id = {
        int(card.cardId): card
        for card in official_cards
    }

print("Repository size:", len(repository))
print(
    "Official CardData lookup size:",
    len(official_card_data_by_id),
)

assert len(repository) == 1267
assert len(official_card_data_by_id) == 1267

print("\nPolicy dependencies loaded.")


# # Cell 9B — Instantiate the policy

# In[14]:


policy = BattlePolicy(
    repository=repository,
    official_lookup=official_card_data_by_id,
    debug=True,
)

print("BattlePolicy created.")
print("Debug mode:", policy.debug)


# ## Cell 10 

# In[15]:


assert callable(policy.choose_action)
assert len(policy.repository) == 1267
assert len(policy.official_lookup) == 1267

print("BattlePolicy created successfully.")


# # Cell 11 — Add policy result model

# In[16]:


@dataclass(frozen=True)
class PolicyDecision:
    """
    Complete policy result for debugging and evaluation.
    """

    option_index: int
    used_fallback: bool
    reason: str
    ranked_actions: tuple[Any, ...]


# # Cell 12 — Upgrade BattlePolicy with safe fallback

# In[17]:


@dataclass(slots=True)
class BattlePolicy:
    repository: Any
    official_lookup: dict[int, Any]
    debug: bool = False
    fallback_index: int = 0

    def decide(self, observation: Any) -> PolicyDecision:
        """
        Convert an official Observation into a safe policy decision.
        """

        try:
            snapshot = adapt_observation(
                observation,
                repository=self.repository,
                official_lookup=self.official_lookup,
            )
        except Exception as exc:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason=(
                    "Observation adaptation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                ranked_actions=(),
            )

        if snapshot is None:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason="Observation contains no current game state.",
                ranked_actions=(),
            )

        try:
            ranked = rank_legal_actions(
                snapshot,
                repository=self.repository,
            )
        except Exception as exc:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason=(
                    "Action ranking failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                ranked_actions=(),
            )

        if not ranked:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason="No legal ranked actions were available.",
                ranked_actions=(),
            )

        decision = PolicyDecision(
            option_index=int(ranked[0].option_index),
            used_fallback=False,
            reason=(
                "Selected highest-scoring legal action: "
                f"{ranked[0].semantic_label}"
            ),
            ranked_actions=tuple(ranked),
        )

        if self.debug:
            self.print_decision(decision)

        return decision

    def choose_action(self, observation: Any) -> int:
        """
        Return only the official option index expected by Kaggle.
        """

        return self.decide(observation).option_index

    def print_decision(
        self,
        decision: PolicyDecision,
    ) -> None:
        print("=" * 72)
        print("BattlePolicy Decision")
        print("=" * 72)
        print("Chosen option:", decision.option_index)
        print("Fallback used:", decision.used_fallback)
        print("Reason:", decision.reason)

        if not decision.ranked_actions:
            print("Ranked actions: none")
            return

        print("\nRanked actions:")

        for rank, action in enumerate(
            decision.ranked_actions,
            start=1,
        ):
            print(
                f"{rank:>2}. "
                f"index={action.option_index:<3} "
                f"type={action.option_type:<10} "
                f"score={action.score:>9.3f} "
                f"{action.semantic_label}"
            )


# # Cell 13 — Recreate and validate the upgraded policy

# In[18]:


policy = BattlePolicy(
    repository=repository,
    official_lookup=official_card_data_by_id,
    debug=True,
    fallback_index=0,
)

assert callable(policy.decide)
assert callable(policy.choose_action)
assert policy.fallback_index == 0

print("Safe BattlePolicy created successfully.")


# # Cell 14 — Test fallback behavior

# In[19]:


fallback_decision = policy.decide(None)

print("Option index:", fallback_decision.option_index)
print("Fallback used:", fallback_decision.used_fallback)
print("Reason:", fallback_decision.reason)

assert fallback_decision.option_index == 0
assert fallback_decision.used_fallback is True

print("\nFallback behavior passed.")


# # Cell 15 — Reuse the synthetic snapshot

# In[20]:


sample_snapshot = notebook19.sample_snapshot

print("Synthetic snapshot loaded.")
print("Turn:", sample_snapshot.turn)
print(
    "Legal options:",
    len(sample_snapshot.selection.options),
)

assert sample_snapshot.selection is not None
assert len(sample_snapshot.selection.options) == 2

print("\nSynthetic snapshot validation passed.")


# # Cell 16 — Add direct snapshot decision support

# In[21]:


@dataclass(slots=True)
class BattlePolicy:
    repository: Any
    official_lookup: dict[int, Any]
    debug: bool = False
    fallback_index: int = 0

    def decide_snapshot(
        self,
        snapshot: Any,
    ) -> PolicyDecision:
        """
        Rank legal actions from an already adapted BattleSnapshot.
        """

        if snapshot is None:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason="BattleSnapshot was None.",
                ranked_actions=(),
            )

        try:
            ranked = rank_legal_actions(
                snapshot,
                repository=self.repository,
            )
        except Exception as exc:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason=(
                    "Action ranking failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                ranked_actions=(),
            )

        if not ranked:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason="No legal ranked actions were available.",
                ranked_actions=(),
            )

        decision = PolicyDecision(
            option_index=int(ranked[0].option_index),
            used_fallback=False,
            reason=(
                "Selected highest-scoring legal action: "
                f"{ranked[0].semantic_label}"
            ),
            ranked_actions=tuple(ranked),
        )

        if self.debug:
            self.print_decision(decision)

        return decision

    def decide(self, observation: Any) -> PolicyDecision:
        """
        Convert an official Observation into a safe policy decision.
        """

        try:
            snapshot = adapt_observation(
                observation,
                repository=self.repository,
                official_lookup=self.official_lookup,
            )
        except Exception as exc:
            return PolicyDecision(
                option_index=int(self.fallback_index),
                used_fallback=True,
                reason=(
                    "Observation adaptation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                ranked_actions=(),
            )

        return self.decide_snapshot(snapshot)

    def choose_action(self, observation: Any) -> int:
        return self.decide(observation).option_index

    def choose_snapshot_action(
        self,
        snapshot: Any,
    ) -> int:
        return self.decide_snapshot(snapshot).option_index

    def print_decision(
        self,
        decision: PolicyDecision,
    ) -> None:
        print("=" * 72)
        print("BattlePolicy Decision")
        print("=" * 72)
        print("Chosen option:", decision.option_index)
        print("Fallback used:", decision.used_fallback)
        print("Reason:", decision.reason)

        if not decision.ranked_actions:
            print("Ranked actions: none")
            return

        print("\nRanked actions:")

        for rank, action in enumerate(
            decision.ranked_actions,
            start=1,
        ):
            print(
                f"{rank:>2}. "
                f"index={action.option_index:<3} "
                f"type={action.option_type:<10} "
                f"score={action.score:>9.3f} "
                f"{action.semantic_label}"
            )


# # Cell 17 — Test successful policy decision

# In[22]:


policy = BattlePolicy(
    repository=repository,
    official_lookup=official_card_data_by_id,
    debug=True,
    fallback_index=0,
)

synthetic_decision = policy.decide_snapshot(
    sample_snapshot
)

assert synthetic_decision.used_fallback is False
assert synthetic_decision.option_index == 0
assert synthetic_decision.ranked_actions[0].option_type == "ATTACK"

print("\nSynthetic policy decision passed.")


# In[ ]:




