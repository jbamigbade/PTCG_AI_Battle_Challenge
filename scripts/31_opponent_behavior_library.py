#!/usr/bin/env python
# coding: utf-8

# # Notebook 31 — Opponent Behavior Library
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 29 identified low replay diversity.
# 
# Notebook 30 created a policy-based agent framework with reusable factories and policies.
# 
# Notebook 31 builds a library of distinct opponent behaviors that can be used in tournaments and mixed-opponent self-play.
# 
# ## Objectives
# 
# 1. Load and validate the `src.agents` framework.
# 2. Define reusable opponent profiles.
# 3. Implement aggressive and defensive policies.
# 4. Create random, greedy, heuristic, search, aggressive, and defensive opponents.
# 5. Validate that each opponent produces a legal decision.
# 6. Compare opponent behavior on the same sample state.
# 7. Create a balanced opponent pool.
# 8. Export reusable opponent modules into `src/agents`.
# 9. Prepare mixed-opponent self-play for Notebook 32.

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import json
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

print("Python:", sys.version)
print("Notebook 31 initialized.")


# # Cell 3 — Locate Project and Update Python Path

# In[2]:


def find_project_root(
    start: Path | None = None,
) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = {
        "notebooks",
        "scripts",
        "src",
        "reports",
    }

    for candidate in [
        current,
        *current.parents,
    ]:
        if all(
            (candidate / marker).exists()
            for marker in markers
        ):
            return candidate

    raise FileNotFoundError(
        "Project root not found."
    )


PROJECT_ROOT = find_project_root()

project_root_str = str(PROJECT_ROOT)

if project_root_str not in sys.path:
    sys.path.insert(
        0,
        project_root_str,
    )

AGENTS_DIR = (
    PROJECT_ROOT
    / "src"
    / "agents"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook31"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Agents directory:", AGENTS_DIR)
print("Report directory:", REPORT_DIR)

assert AGENTS_DIR.exists()


# # Cell 4 — Import the Notebook 30 Framework

# In[3]:


from src.agents import (
    AgentDecision,
    AgentFactory,
    BaseAgent,
    BasePolicy,
    GreedyPolicy,
    HeuristicPolicy,
    PolicyConfig,
    PolicyFactory,
    RandomPolicy,
    SearchPolicy,
)

print("Imported agent framework successfully.")

print()
print("Available policies:")
print(PolicyFactory.available_policies())

assert {
    "random",
    "greedy",
    "heuristic",
    "search",
}.issubset(
    set(
        PolicyFactory.available_policies()
    )
)


# # Cell 5 — Opponent Profile

# In[4]:


@dataclass(frozen=True, slots=True)
class OpponentProfile:
    name: str
    policy_type: str
    description: str

    seed: int | None = None
    search_depth: int = 0

    aggression_weight: float = 1.0
    defense_weight: float = 1.0
    immediate_value_weight: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


sample_profile = OpponentProfile(
    name="Random Opponent",
    policy_type="random",
    description=(
        "Chooses uniformly from legal moves."
    ),
    seed=42,
)

print(sample_profile)

assert sample_profile.policy_type == "random"


# # Cell 6 — Aggressive Policy

# In[6]:


class AggressivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(
            legal_moves
        )

        damage_scores = state.get(
            "damage_scores",
            {},
        )

        knockout_scores = state.get(
            "knockout_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.aggression_weight
                * float(
                    damage_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    knockout_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason=(
                "Highest damage and knockout value"
            ),
            metadata={
                "policy": self.name,
            },
        )


# # Cell 7 — Validate Aggressive Policy

# In[7]:


aggressive_config = PolicyConfig(
    name="AggressivePolicy",
    policy_type="aggressive",
    aggression_weight=1.5,
)

aggressive_policy = AggressivePolicy(
    aggressive_config
)

aggressive_state = {
    "damage_scores": {
        "Evolution Burst": 60,
        "Quick Attack": 30,
        "Retreat": 0,
    },
    "knockout_scores": {
        "Evolution Burst": 100,
        "Quick Attack": 0,
        "Retreat": 0,
    },
}

aggressive_decision = (
    aggressive_policy.choose_move(
        state=aggressive_state,
        legal_moves=[
            "Evolution Burst",
            "Quick Attack",
            "Retreat",
        ],
    )
)

print(aggressive_decision)

assert (
    aggressive_decision.move
    == "Evolution Burst"
)

print()
print("AggressivePolicy validated.")


# # Cell 8 — Defensive Policy

# In[8]:


class DefensivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(
            legal_moves
        )

        defense_scores = state.get(
            "defense_scores",
            {},
        )

        healing_scores = state.get(
            "healing_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.defense_weight
                * float(
                    defense_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    healing_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason="Highest defensive value",
            metadata={
                "policy": self.name,
            },
        )


# # Cell 9 — Validate DefensivePolicy

# In[9]:


defensive_config = PolicyConfig(
    name="DefensivePolicy",
    policy_type="defensive",
    defense_weight=1.5,
)

defensive_policy = DefensivePolicy(
    defensive_config
)

state = {
    "defense_scores": {
        "Heal": 80,
        "Retreat": 60,
        "Attack": 10,
    },
    "healing_scores": {
        "Heal": 40,
        "Retreat": 0,
        "Attack": 0,
    },
}

decision = defensive_policy.choose_move(
    state=state,
    legal_moves=[
        "Heal",
        "Retreat",
        "Attack",
    ],
)

print(decision)

assert decision.move == "Heal"

print()
print("DefensivePolicy validated.")


# # Cell 10 — Opponent Library

# In[14]:


opponents = [
    OpponentProfile(
        name="Random",
        policy_type="random",
        description="Random legal moves",
    ),
    OpponentProfile(
        name="Greedy",
        policy_type="greedy",
        description="Immediate reward",
    ),
    OpponentProfile(
        name="Heuristic",
        policy_type="heuristic",
        description="Rule-based play",
    ),
    OpponentProfile(
        name="Search",
        policy_type="search",
        description="Tree search",
        search_depth=6,
    ),
    OpponentProfile(
        name="Aggressive",
        policy_type="aggressive",
        description="Maximum damage",
        aggression_weight=1.5,
    ),
    OpponentProfile(
        name="Defensive",
        policy_type="defensive",
        description="Preserve resources",
        defense_weight=1.5,
    ),
]

from dataclasses import asdict


library_df = pd.DataFrame(
    [
        asdict(opponent)
        for opponent in opponents
    ]
)


print()
print("Opponent library validated.")

display(library_df)

assert len(opponents) == 6


# # Cell 11 — Register Aggressive and Defensive Policies

# In[15]:


PolicyFactory._registry["aggressive"] = AggressivePolicy
PolicyFactory._registry["defensive"] = DefensivePolicy

available_policies = (
    PolicyFactory.available_policies()
)

print("Available policies:")
print(available_policies)

assert "aggressive" in available_policies
assert "defensive" in available_policies

print()
print(
    "Aggressive and defensive policies registered."
)


# # Cell 12 — Create Opponents from Profiles

# In[16]:


def create_opponent(
    profile: OpponentProfile,
) -> BaseAgent:

    return AgentFactory.create(
        name=profile.name,
        policy_type=profile.policy_type,
        seed=profile.seed,
        search_depth=profile.search_depth,
        aggression_weight=(
            profile.aggression_weight
        ),
        defense_weight=(
            profile.defense_weight
        ),
        immediate_value_weight=(
            profile.immediate_value_weight
        ),
        metadata=profile.metadata,
    )


opponent_agents = [
    create_opponent(profile)
    for profile in opponents
]

for agent in opponent_agents:
    print(agent)

assert len(opponent_agents) == 6

print()
print("All opponent agents created.")


# # Cell 13 — Shared Behavior Test State

# In[17]:


test_legal_moves = [
    "Evolution Burst",
    "Quick Attack",
    "Retreat",
]

test_state = {
    "move_scores": {
        "Evolution Burst": 80,
        "Quick Attack": 40,
        "Retreat": 10,
    },
    "heuristic_scores": {
        "Evolution Burst": 70,
        "Quick Attack": 65,
        "Retreat": 20,
    },
    "search_scores": {
        "Evolution Burst": 90,
        "Quick Attack": 85,
        "Retreat": 15,
    },
    "damage_scores": {
        "Evolution Burst": 60,
        "Quick Attack": 30,
        "Retreat": 0,
    },
    "knockout_scores": {
        "Evolution Burst": 100,
        "Quick Attack": 0,
        "Retreat": 0,
    },
    "defense_scores": {
        "Evolution Burst": 10,
        "Quick Attack": 20,
        "Retreat": 90,
    },
    "healing_scores": {
        "Evolution Burst": 0,
        "Quick Attack": 0,
        "Retreat": 20,
    },
}

print("Shared behavior test state created.")


# # Cell 14 — Compare Opponent Decisions

# In[18]:


behavior_rows = []

for agent in opponent_agents:
    decision = agent.choose_move(
        state=test_state,
        legal_moves=test_legal_moves,
    )

    behavior_rows.append(
        {
            "opponent": agent.name,
            "policy_type": (
                agent.policy.policy_type
            ),
            "chosen_move": decision.move,
            "score": decision.score,
            "reason": decision.reason,
        }
    )

behavior_df = pd.DataFrame(
    behavior_rows
)

display(behavior_df)

assert len(behavior_df) == 6
assert behavior_df["chosen_move"].notna().all()

print()
print("Opponent behavior comparison validated.")


# # Cell 15 — Opponent Diversity Statistics

# In[19]:


print("=" * 70)
print("OPPONENT DIVERSITY")
print("=" * 70)

decision_counts = (
    behavior_df["chosen_move"]
    .value_counts()
    .rename_axis("move")
    .reset_index(name="count")
)

display(decision_counts)

unique_moves = behavior_df["chosen_move"].nunique()

print()
print("Unique chosen moves :", unique_moves)
print("Total opponents     :", len(behavior_df))
print(
    "Decision diversity  :",
    f"{unique_moves / len(behavior_df):.2f}",
)

assert unique_moves >= 2

print()
print("Opponent diversity validated.")


# # Cell 16 — Build the Tournament Opponent Pool

# In[20]:


opponent_pool = {
    profile.name: create_opponent(profile)
    for profile in opponents
}

print("Opponent pool:")

for name in opponent_pool:
    print("-", name)

assert len(opponent_pool) == 6

print()
print("Tournament opponent pool created.")


# # Cell 17 — Create an Opponent Selection Strategy

# In[21]:


selection_plan = pd.DataFrame(
    [
        {
            "strategy": "Round Robin",
            "description": "Cycle through every opponent equally",
            "recommended": True,
        },
        {
            "strategy": "Random Uniform",
            "description": "Randomly sample all opponents equally",
            "recommended": True,
        },
        {
            "strategy": "Weighted Random",
            "description": "Favor stronger opponents later",
            "recommended": False,
        },
        {
            "strategy": "Curriculum",
            "description": "Easy opponents before difficult ones",
            "recommended": False,
        },
    ]
)

display(selection_plan)

assert len(selection_plan) == 4

print()
print("Opponent selection strategies documented.")


# # Cell 18 — Select the Recommended Strategy

# In[22]:


training_strategy = {
    "name": "Round Robin",
    "description": (
        "Each opponent is used the same number of times "
        "during replay generation."
    ),
    "opponents": list(opponent_pool.keys()),
}

print(training_strategy)

assert training_strategy["name"] == "Round Robin"

print()
print("Training strategy selected.")


# # Cell 19 — Build the Round-Robin Scheduler

# In[23]:


from itertools import cycle

round_robin_scheduler = cycle(opponent_pool.keys())

scheduled_opponents = [
    next(round_robin_scheduler)
    for _ in range(18)
]

schedule_df = pd.DataFrame({
    "Game": range(1, 19),
    "Opponent": scheduled_opponents,
})

display(schedule_df)

assert len(schedule_df) == 18

print()
print("Round-robin scheduler validated.")


# # Cell 20 — Validate Equal Distribution

# In[24]:


distribution = (
    schedule_df["Opponent"]
    .value_counts()
    .sort_index()
)

distribution_df = distribution.rename("Games").reset_index()
distribution_df.columns = ["Opponent", "Games"]

display(distribution_df)

assert distribution.nunique() == 1

print()
print("Balanced opponent distribution validated.")


# # Cell 21 — Scale the Scheduler to 2,000 Games

# In[25]:


TOTAL_TRAINING_GAMES = 2_000

opponent_names = list(opponent_pool.keys())

full_schedule = [
    opponent_names[i % len(opponent_names)]
    for i in range(TOTAL_TRAINING_GAMES)
]

schedule_counts = (
    pd.Series(full_schedule)
      .value_counts()
      .sort_index()
)

schedule_summary = (
    schedule_counts
    .rename("Games")
    .reset_index()
)

schedule_summary.columns = [
    "Opponent",
    "Games",
]

display(schedule_summary)

print()
print("Total games:", schedule_counts.sum())

assert schedule_counts.sum() == TOTAL_TRAINING_GAMES
assert schedule_counts.max() - schedule_counts.min() <= 1

print()
print("2,000-game scheduler validated.")


# # Cell 22 — Visualize the Training Allocation

# In[26]:


allocation_df = schedule_summary.copy()

allocation_df["Percent"] = (
    allocation_df["Games"]
    / TOTAL_TRAINING_GAMES
    * 100
)

display(allocation_df)

assert abs(
    allocation_df["Percent"].sum() - 100
) < 1e-8

print()
print("Training allocation validated.")


# # Cell 23 — Export Opponent Profiles and Scheduler

# In[28]:


opponents_module = '''from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Mapping

from .base_agent import BaseAgent
from .factories import AgentFactory


@dataclass(frozen=True, slots=True)
class OpponentProfile:
    name: str
    policy_type: str
    description: str

    seed: int | None = None
    search_depth: int = 0

    aggression_weight: float = 1.0
    defense_weight: float = 1.0
    immediate_value_weight: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


def create_opponent(
    profile: OpponentProfile,
) -> BaseAgent:
    return AgentFactory.create(
        name=profile.name,
        policy_type=profile.policy_type,
        seed=profile.seed,
        search_depth=profile.search_depth,
        aggression_weight=profile.aggression_weight,
        defense_weight=profile.defense_weight,
        immediate_value_weight=(
            profile.immediate_value_weight
        ),
        metadata=profile.metadata,
    )


def build_round_robin_schedule(
    opponent_names: list[str],
    num_games: int,
) -> list[str]:
    if not opponent_names:
        raise ValueError(
            "opponent_names cannot be empty."
        )

    if num_games < 1:
        raise ValueError(
            "num_games must be at least 1."
        )

    scheduler = cycle(opponent_names)

    return [
        next(scheduler)
        for _ in range(num_games)
    ]
'''

opponents_file = (
    AGENTS_DIR
    / "opponents.py"
)

opponents_file.write_text(
    opponents_module,
    encoding="utf-8",
)

print("[OK]", opponents_file)


# # Cell 24 — Add Aggressive and Defensive Policies to policies.py

# In[33]:


policies_module = '''from __future__ import annotations

import random

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .configs import PolicyConfig
from .decisions import AgentDecision


class BasePolicy(ABC):

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        self.config = config
        self.name = config.name
        self.policy_type = config.policy_type
        self._rng = random.Random(config.seed)

    @abstractmethod
    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:
        ...

    def validate_legal_moves(
        self,
        legal_moves: Sequence[Any],
    ) -> None:
        if not legal_moves:
            raise ValueError(
                f"{self.name} received no legal moves."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"policy_type={self.policy_type!r})"
        )


class RandomPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        move = self._rng.choice(
            list(legal_moves)
        )

        return AgentDecision(
            move=move,
            score=0.0,
            reason="Random legal move",
            metadata={"policy": self.name},
        )


class GreedyPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "move_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                0.0,
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Highest immediate value",
            metadata={"policy": self.name},
        )


class HeuristicPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "heuristic_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                float("-inf"),
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Highest heuristic evaluation",
            metadata={"policy": self.name},
        )


class SearchPolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        scores = state.get(
            "search_scores",
            {},
        )

        best_move = max(
            legal_moves,
            key=lambda move: scores.get(
                move,
                float("-inf"),
            ),
        )

        return AgentDecision(
            move=best_move,
            score=float(
                scores.get(best_move, 0.0)
            ),
            reason="Search engine recommendation",
            metadata={
                "policy": self.name,
                "search_depth": (
                    self.config.search_depth
                ),
            },
        )


class AggressivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        damage_scores = state.get(
            "damage_scores",
            {},
        )

        knockout_scores = state.get(
            "knockout_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.aggression_weight
                * float(
                    damage_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    knockout_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason=(
                "Highest damage and knockout value"
            ),
            metadata={"policy": self.name},
        )


class DefensivePolicy(BasePolicy):

    def choose_move(
        self,
        state: Mapping[str, Any],
        legal_moves: Sequence[Any],
    ) -> AgentDecision:

        self.validate_legal_moves(legal_moves)

        defense_scores = state.get(
            "defense_scores",
            {},
        )

        healing_scores = state.get(
            "healing_scores",
            {},
        )

        total_scores = {
            move: (
                self.config.defense_weight
                * float(
                    defense_scores.get(
                        move,
                        0.0,
                    )
                )
                + float(
                    healing_scores.get(
                        move,
                        0.0,
                    )
                )
            )
            for move in legal_moves
        }

        best_move = max(
            legal_moves,
            key=lambda move: total_scores[move],
        )

        return AgentDecision(
            move=best_move,
            score=total_scores[best_move],
            reason="Highest defensive value",
            metadata={"policy": self.name},
        )
'''

policies_file = AGENTS_DIR / "policies.py"

policies_file.write_text(
    policies_module,
    encoding="utf-8",
)

print("[OK]", policies_file)
print("AggressivePolicy present:", "class AggressivePolicy" in policies_module)
print("DefensivePolicy present:", "class DefensivePolicy" in policies_module)


# # Cell 25 — Update Factories and Package Exports

# In[34]:


factories_module = '''from __future__ import annotations

from .base_agent import BaseAgent
from .configs import PolicyConfig
from .policies import (
    AggressivePolicy,
    BasePolicy,
    DefensivePolicy,
    GreedyPolicy,
    HeuristicPolicy,
    RandomPolicy,
    SearchPolicy,
)


class PolicyFactory:

    _registry = {
        "random": RandomPolicy,
        "greedy": GreedyPolicy,
        "heuristic": HeuristicPolicy,
        "search": SearchPolicy,
        "aggressive": AggressivePolicy,
        "defensive": DefensivePolicy,
    }

    @classmethod
    def create(
        cls,
        config: PolicyConfig,
    ) -> BasePolicy:

        try:
            policy_class = cls._registry[
                config.policy_type
            ]
        except KeyError as error:
            raise ValueError(
                f"Unknown policy "
                f"{config.policy_type!r}"
            ) from error

        return policy_class(config)

    @classmethod
    def available_policies(
        cls,
    ) -> list[str]:

        return sorted(cls._registry)


class AgentFactory:

    @staticmethod
    def create(
        name: str,
        policy_type: str,
        **kwargs,
    ) -> BaseAgent:

        config = PolicyConfig(
            name=f"{policy_type.title()}Policy",
            policy_type=policy_type,
            **kwargs,
        )

        return BaseAgent(
            name=name,
            policy=PolicyFactory.create(config),
        )
'''

factories_file = AGENTS_DIR / "factories.py"

factories_file.write_text(
    factories_module,
    encoding="utf-8",
)

init_module = '''from .base_agent import BaseAgent
from .configs import PolicyConfig
from .decisions import AgentDecision
from .factories import AgentFactory, PolicyFactory
from .opponents import (
    OpponentProfile,
    build_round_robin_schedule,
    create_opponent,
)
from .policies import (
    AggressivePolicy,
    BasePolicy,
    DefensivePolicy,
    GreedyPolicy,
    HeuristicPolicy,
    RandomPolicy,
    SearchPolicy,
)

__all__ = [
    "AgentDecision",
    "PolicyConfig",
    "BasePolicy",
    "RandomPolicy",
    "GreedyPolicy",
    "HeuristicPolicy",
    "SearchPolicy",
    "AggressivePolicy",
    "DefensivePolicy",
    "BaseAgent",
    "PolicyFactory",
    "AgentFactory",
    "OpponentProfile",
    "create_opponent",
    "build_round_robin_schedule",
]
'''

init_file = AGENTS_DIR / "__init__.py"

init_file.write_text(
    init_module,
    encoding="utf-8",
)

print("[OK]", factories_file)
print("[OK]", init_file)


# # Cell 26 — Export Notebook 31 Reports

# In[31]:


library_file = (
    REPORT_DIR
    / "opponent_library.csv"
)

behavior_file = (
    REPORT_DIR
    / "opponent_behavior_comparison.csv"
)

allocation_file = (
    REPORT_DIR
    / "training_allocation_2000.csv"
)

strategy_file = (
    REPORT_DIR
    / "training_strategy.json"
)

library_df.to_csv(
    library_file,
    index=False,
)

behavior_df.to_csv(
    behavior_file,
    index=False,
)

allocation_df.to_csv(
    allocation_file,
    index=False,
)

with open(
    strategy_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        training_strategy,
        file,
        indent=2,
    )

print("Notebook 31 reports exported.")


# # Cell 27 — Final Validation

# In[35]:


import importlib
import sys


cached_agent_modules = [
    name
    for name in list(sys.modules)
    if (
        name == "src.agents"
        or name.startswith("src.agents.")
    )
]

for module_name in cached_agent_modules:
    del sys.modules[module_name]

importlib.invalidate_caches()

from src.agents import (
    AgentFactory,
    AggressivePolicy,
    DefensivePolicy,
    OpponentProfile,
    build_round_robin_schedule,
    create_opponent,
)

available_policies = (
    AgentFactory.create(
        name="Aggressive Test",
        policy_type="aggressive",
    )
)

assert isinstance(
    available_policies.policy,
    AggressivePolicy,
)

defensive_test = AgentFactory.create(
    name="Defensive Test",
    policy_type="defensive",
)

assert isinstance(
    defensive_test.policy,
    DefensivePolicy,
)

test_schedule = build_round_robin_schedule(
    [
        "Random",
        "Greedy",
        "Defensive",
    ],
    9,
)

assert test_schedule == [
    "Random",
    "Greedy",
    "Defensive",
    "Random",
    "Greedy",
    "Defensive",
    "Random",
    "Greedy",
    "Defensive",
]

expected_files = [
    opponents_file,
    policies_file,
    factories_file,
    init_file,
    library_file,
    behavior_file,
    allocation_file,
    strategy_file,
]

for path in expected_files:
    assert path.exists(), path

    print(
        f"[OK] {path.name:<38} "
        f"{path.stat().st_size:>8,} bytes"
    )

print()
print(
    "Notebook 31 implementation and exports validated."
)


# In[ ]:




