#!/usr/bin/env python
# coding: utf-8

# # Notebook 32 — Mixed-Opponent Self-Play and Scale Benchmarks
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 28 created the original self-play pipeline.
# 
# Notebook 29 identified low replay diversity and documented 100-, 1,000-, and 2,000-game benchmark plans.
# 
# Notebook 30 created the policy-based agent framework.
# 
# Notebook 31 created six opponent behaviors and a balanced round-robin scheduler.
# 
# Notebook 32 integrates those components and measures how the self-play system scales across mixed opponents.
# 
# ## Objectives
# 
# 1. Import the training, tournament, and opponent frameworks.
# 2. Recreate the validated battle factories from Notebook 28.
# 3. Build mixed-opponent match factories.
# 4. Run a 100-game mixed-opponent smoke test.
# 5. Run a 1,000-game scaling benchmark.
# 6. Run a 2,000-game full benchmark.
# 7. Compare runtime, throughput, winners, turns, states, and moves.
# 8. Measure whether opponent diversity improves replay diversity.
# 9. Export benchmark datasets and summaries.
# 10. Prepare the richer replay data for reward shaping and feature engineering.

# # Cell 2 — Imports

# In[2]:


from __future__ import annotations

import json
import sys
import time

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

print("Python:", sys.version)
print("Notebook 32 initialized.")


# # Cell 3 — Locate Project and Update Python Path

# In[3]:


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook32"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Report directory:", REPORT_DIR)

assert REPORT_DIR.exists()


# # Cell 4 — Import Reusable Frameworks

# In[4]:


from src.agents import (
    AgentFactory,
    OpponentProfile,
    build_round_robin_schedule,
    create_opponent,
)

from src.training.self_play import (
    SelfPlayRunSummary,
    run_self_play_session,
)

from src.tournament.match import (
    TournamentMatch,
)

print("Reusable frameworks imported.")

print()
print(
    "Available policies:",
    AgentFactory.create(
        name="Search Test",
        policy_type="search",
        search_depth=6,
    ).policy.policy_type,
)


# # Cell 5 — Benchmark Configuration

# In[5]:


BENCHMARK_MATCH_COUNTS = [
    100,
    1_000,
    2_000,
]

OPPONENT_PROFILES = [
    OpponentProfile(
        name="Random",
        policy_type="random",
        description="Random legal moves",
        seed=42,
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

OPPONENT_POOL = {
    profile.name: create_opponent(profile)
    for profile in OPPONENT_PROFILES
}

print("Benchmark counts:", BENCHMARK_MATCH_COUNTS)
print("Opponent pool:", list(OPPONENT_POOL))

assert len(OPPONENT_POOL) == 6


# # Cell 6 — Inspect TournamentMatch
# 
# ### We need the exact constructor before rebuilding the match factory.

# In[6]:


import inspect


print("TournamentMatch signature:")
print(
    inspect.signature(
        TournamentMatch
    )
)

print()
print("TournamentMatch fields:")

for field_name, field_info in (
    TournamentMatch.__dataclass_fields__.items()
):
    print(
        f" - {field_name}: "
        f"{field_info.type}"
    )


# # Cell 7 — Inspect Prior Battle State and Match Helpers

# In[7]:


candidate_modules = [
    "src.tournament.runner",
    "src.tournament.match",
    "src.engine",
    "src.full_game_simulation",
]

for module_name in candidate_modules:
    try:
        module = __import__(
            module_name,
            fromlist=["*"],
        )

        public_names = [
            name
            for name in dir(module)
            if not name.startswith("_")
        ]

        print()
        print("=" * 70)
        print(module_name)
        print("=" * 70)

        for name in public_names:
            if any(
                token in name.lower()
                for token in [
                    "state",
                    "match",
                    "agent",
                    "battle",
                    "factory",
                ]
            ):
                print(" -", name)

    except Exception as error:
        print(
            f"[SKIP] {module_name}: {error}"
        )


# # Cell 8 — Inspect PokemonBattleAgent

# In[9]:


from src.tournament.match import PokemonBattleAgent


print("PokemonBattleAgent signature:")
print(inspect.signature(PokemonBattleAgent))

print()
print("PokemonBattleAgent is dataclass:")
print(hasattr(PokemonBattleAgent, "__dataclass_fields__"))

print()
print("Public methods and attributes:")

public_names = [
    name
    for name in dir(PokemonBattleAgent)
    if not name.startswith("_")
]

for name in public_names:
    value = getattr(
        PokemonBattleAgent,
        name,
        None,
    )

    if callable(value):
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            signature = "<signature unavailable>"

        print(f" - {name}{signature}")
    else:
        print(f" - {name}")


# In[10]:


print("=" * 70)
print("PokemonBattleAgent source")
print("=" * 70)

print(
    inspect.getsource(
        PokemonBattleAgent
    )
)


# # Cell 9 — Inspect BattleState

# In[11]:


from src.tournament.runner import BattleState


print("BattleState signature:")
print(inspect.signature(BattleState))

print()
print("BattleState is dataclass:")
print(hasattr(BattleState, "__dataclass_fields__"))

if hasattr(
    BattleState,
    "__dataclass_fields__",
):
    print()
    print("BattleState fields:")

    for field_name, field_info in (
        BattleState.__dataclass_fields__.items()
    ):
        print(
            f" - {field_name}: "
            f"{field_info.type}"
        )
else:
    print()
    print("BattleState source:")
    print(inspect.getsource(BattleState))


# # Cell 10 — Inspect Relevant Source Code

# In[12]:


print("=" * 70)
print("PokemonBattleAgent source")
print("=" * 70)

print(
    inspect.getsource(
        PokemonBattleAgent
    )
)

print()
print("=" * 70)
print("TournamentMatch source")
print("=" * 70)

print(
    inspect.getsource(
        TournamentMatch
    )
)


# # Cell 11 — Inspect Match Runner Requirements

# In[13]:


from src.tournament.runner import (
    run_two_agent_match,
)


print("run_two_agent_match signature:")
print(
    inspect.signature(
        run_two_agent_match
    )
)

print()
print("=" * 70)
print("run_two_agent_match source")
print("=" * 70)

print(
    inspect.getsource(
        run_two_agent_match
    )
)


# # Cell 12 — Inspect Legal-Move Utilities

# In[14]:


import src.legal_moves as legal_moves_module


print("=" * 70)
print("LEGAL MOVE UTILITIES")
print("=" * 70)

legal_move_names = [
    name
    for name in dir(legal_moves_module)
    if not name.startswith("_")
]

for name in legal_move_names:
    value = getattr(
        legal_moves_module,
        name,
    )

    if callable(value):
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            signature = "<signature unavailable>"

        print(f" - {name}{signature}")
    else:
        print(f" - {name}")


# # Cell 13 — Inspect Tournament AgentDecision

# In[16]:


import inspect
import src.tournament.match as match_module


print("=" * 70)
print("AGENT DECISION DISCOVERY")
print("=" * 70)

decision_candidates = []

for name, value in vars(match_module).items():
    if "decision" in name.lower():
        decision_candidates.append(
            (name, value)
        )

if not decision_candidates:
    print(
        "No decision-related symbol is exposed "
        "by src.tournament.match."
    )
else:
    for name, value in decision_candidates:
        print()
        print("Name:", name)
        print("Value:", value)
        print("Module:", getattr(value, "__module__", None))

        if inspect.isclass(value):
            try:
                print(
                    "Signature:",
                    inspect.signature(value),
                )
            except (TypeError, ValueError):
                print("Signature unavailable")

            print(
                "Is dataclass:",
                hasattr(
                    value,
                    "__dataclass_fields__",
                ),
            )


# In[17]:


MATCH_FILE = (
    PROJECT_ROOT
    / "src"
    / "tournament"
    / "match.py"
)

match_source = MATCH_FILE.read_text(
    encoding="utf-8-sig",
)

print(match_source[:4000])


# # Cell 14 — Inspect Search Result Type

# In[18]:


search_result_candidates = []

candidate_modules = [
    "src.engine",
    "src.engine.search",
    "src.engine.advanced_search",
    "src.tournament.match",
]

for module_name in candidate_modules:
    try:
        module = __import__(
            module_name,
            fromlist=["*"],
        )
    except Exception as error:
        print(
            f"[SKIP] {module_name}: {error}"
        )
        continue

    for name in dir(module):
        if (
            "result" in name.lower()
            or "search" in name.lower()
        ):
            value = getattr(
                module,
                name,
            )

            if inspect.isclass(value):
                search_result_candidates.append(
                    (
                        module_name,
                        name,
                        value,
                    )
                )

for module_name, name, value in search_result_candidates:
    print()
    print(f"{module_name}.{name}")

    try:
        print(
            "Signature:",
            inspect.signature(value),
        )
    except (TypeError, ValueError):
        print("Signature unavailable")


# # Cell 15 — Inspect BattleState

# In[19]:


print("=" * 70)
print("BattleState source")
print("=" * 70)

print(
    inspect.getsource(
        BattleState
    )
)


# # Cell 16 — Inspect PlayerState

# In[22]:


import inspect
import sys

from typing import get_type_hints


battle_state_hints = get_type_hints(
    BattleState,
    globalns=vars(
        sys.modules[BattleState.__module__]
    ),
)

PlayerState = battle_state_hints["player"]

print("Resolved PlayerState:", PlayerState)
print("Defined in module:", PlayerState.__module__)

print()
print("PlayerState signature:")
print(inspect.signature(PlayerState))

print()
print("PlayerState is dataclass:")
print(
    hasattr(
        PlayerState,
        "__dataclass_fields__",
    )
)

print()
print("=" * 70)
print("PlayerState source")
print("=" * 70)

print(
    inspect.getsource(
        PlayerState
    )
)


# In[23]:


print("BattleState module:", BattleState.__module__)
print("PlayerState module:", PlayerState.__module__)

if hasattr(
    PlayerState,
    "__dataclass_fields__",
):
    print()
    print("PlayerState fields:")

    for field_name, field_info in (
        PlayerState.__dataclass_fields__.items()
    ):
        print(
            f" - {field_name}: "
            f"{field_info.type}"
        )


# # Cell 17 — Inspect Legal Move Functions

# In[25]:


import inspect
import src.legal_moves as legal_moves_module


print("=" * 70)
print("LEGAL MOVE FUNCTIONS")
print("=" * 70)

for name in dir(legal_moves_module):
    if name.startswith("_"):
        continue

    value = getattr(
        legal_moves_module,
        name,
    )

    if callable(value):
        try:
            signature = inspect.signature(
                value
            )
        except (TypeError, ValueError):
            signature = (
                "<signature unavailable>"
            )

        print(f"{name}{signature}")


# # Cell 18 — Inspect Move and PokemonState

# In[28]:


from src.legal_moves import Move, PokemonState


print("Move signature:")
print(inspect.signature(Move))

print()
print("Move is dataclass:")
print(hasattr(Move, "__dataclass_fields__"))

if hasattr(Move, "__dataclass_fields__"):
    print()
    print("Move fields:")

    for field_name, field_info in (
        Move.__dataclass_fields__.items()
    ):
        print(
            f" - {field_name}: "
            f"{field_info.type}"
        )

print()
print("PokemonState signature:")
print(inspect.signature(PokemonState))

print()
print("PokemonState fields:")

for field_name, field_info in (
    PokemonState.__dataclass_fields__.items()
):
    print(
        f" - {field_name}: "
        f"{field_info.type}"
    )


# # Cell 19 — Inspect Move Source

# In[31]:


print("=" * 70)
print("MOVE INSPECTION")
print("=" * 70)

print("Move object:", Move)
print("Move type:", type(Move))
print("Move module:", getattr(Move, "__module__", None))
print("Move name:", getattr(Move, "__name__", None))

print()
print("Move signature:")
print(inspect.signature(Move))

print()
print(
    "Move is dataclass:",
    hasattr(Move, "__dataclass_fields__"),
)

if hasattr(Move, "__dataclass_fields__"):
    print()
    print("Move fields:")

    for field_name, field_info in (
        Move.__dataclass_fields__.items()
    ):
        print(
            f" - {field_name}: "
            f"{field_info.type}"
        )
else:
    print()
    print("Public attributes:")

    for name in dir(Move):
        if not name.startswith("_"):
            print(" -", name)

print()
print("Move inspection completed.")


# # Cell 20 — Build Policy Features from a Battle State

# In[32]:


from src.legal_moves import (
    get_current_legal_moves,
)


def move_label(
    move: Any,
) -> str:
    for attribute in [
        "name",
        "move_name",
        "action",
        "attack_name",
    ]:
        value = getattr(
            move,
            attribute,
            None,
        )

        if value:
            return str(value)

    return str(move)


def build_policy_state(
    battle_state: BattleState,
    legal_moves: list[Any],
) -> dict[str, Any]:

    current_side = (
        battle_state.player
        if battle_state.current_player == "Player"
        else battle_state.opponent
    )

    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    current_hp = float(
        current_side.active.current_hp
    )

    opponent_hp = float(
        opposing_side.active.current_hp
    )

    labels = [
        move_label(move)
        for move in legal_moves
    ]

    state = {
        "turn_number": battle_state.turn_number,
        "current_player": battle_state.current_player,
        "current_hp": current_hp,
        "opponent_hp": opponent_hp,
        "current_hand_size": current_side.hand_size,
        "opponent_hand_size": opposing_side.hand_size,
        "current_prizes": (
            current_side.prize_cards_remaining
        ),
        "opponent_prizes": (
            opposing_side.prize_cards_remaining
        ),
        "move_objects": {
            label: move
            for label, move in zip(
                labels,
                legal_moves,
            )
        },
    }

    return state


# In[33]:


print(
    "initial_state_factory exists:",
    "initial_state_factory" in globals(),
)

print(
    "initial_state exists:",
    "initial_state" in globals(),
)

NOTEBOOK28_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "28_self_play_training_pipeline.py"
)

print()
print("Notebook 28 script exists:", NOTEBOOK28_SCRIPT.exists())

if NOTEBOOK28_SCRIPT.exists():
    notebook28_text = NOTEBOOK28_SCRIPT.read_text(
        encoding="utf-8-sig",
    )

    search_terms = [
        "def initial_state_factory",
        "initial_state_factory =",
        "BattleState(",
        "PlayerState(",
        "PokemonState(",
    ]

    for term in search_terms:
        position = notebook28_text.find(term)

        print()
        print(f"{term!r}: {position}")

        if position >= 0:
            start = max(0, position - 800)
            end = min(
                len(notebook28_text),
                position + 2500,
            )

            print(
                notebook28_text[start:end]
            )


# # Cell 21 — Validate Legal-Move Extraction

# In[34]:


print(
    "initial_state_factory exists:",
    "initial_state_factory" in globals(),
)

print(
    "sample initial state exists:",
    "initial_state" in globals(),
)


# In[35]:


NOTEBOOK28_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "28_self_play_training_pipeline.py"
)

print(
    NOTEBOOK28_SCRIPT.read_text(
        encoding="utf-8-sig",
    )[:12000]
)


# # Cell 22 — Locate Notebook 27 Helpers

# In[37]:


from pathlib import Path
import importlib.util
import inspect


HELPER_MATCHES = []

for path in PROJECT_ROOT.rglob("*.py"):
    try:
        text = path.read_text(
            encoding="utf-8-sig",
        )
    except (UnicodeDecodeError, OSError):
        continue

    if (
        "def create_battle_agent" in text
        and "def create_test_battle" in text
    ):
        HELPER_MATCHES.append(path)

print("Helper matches:")

for path in HELPER_MATCHES:
    print(" -", path)

assert HELPER_MATCHES, (
    "No Python file containing both helper "
    "functions was found."
)


# # Cell 23

# In[38]:


HELPER_FILE = HELPER_MATCHES[0]

spec = importlib.util.spec_from_file_location(
    "notebook27_helpers",
    HELPER_FILE,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load helper file: {HELPER_FILE}"
    )

notebook27_helpers = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    notebook27_helpers
)

print("Loaded helper file:", HELPER_FILE)

print()
print(
    "create_battle_agent:",
    inspect.signature(
        notebook27_helpers.create_battle_agent
    ),
)

print(
    "create_test_battle:",
    inspect.signature(
        notebook27_helpers.create_test_battle
    ),
)


# # Cell 24 

# In[39]:


BASELINE_SEED = 42

baseline_player_agent = (
    notebook27_helpers.create_battle_agent(
        seed=BASELINE_SEED,
    )
)

baseline_opponent_agent = (
    notebook27_helpers.create_battle_agent(
        seed=BASELINE_SEED + 1,
    )
)


def baseline_match_factory(
    game_index: int,
) -> TournamentMatch:

    return TournamentMatch(
        player_agent=baseline_player_agent,
        opponent_agent=baseline_opponent_agent,
        player_name="TrainingAgent",
        opponent_name="TrainingOpponent",
    )


def initial_state_factory(
    game_index: int,
) -> BattleState:

    starting_player = (
        "Player"
        if game_index % 2 == 0
        else "Opponent"
    )

    return notebook27_helpers.create_test_battle(
        starting_player=starting_player,
    )


test_match = baseline_match_factory(0)
test_state = initial_state_factory(0)

print(test_match)
print(test_state)

assert isinstance(test_match, TournamentMatch)
assert isinstance(test_state, BattleState)

print()
print("Baseline factories validated.")


# # Cell 25 — Validate Real Legal Moves

# In[40]:


real_legal_moves = get_current_legal_moves(
    test_state
)

print("Legal move count:", len(real_legal_moves))

for move in real_legal_moves:
    print(
        " -",
        move_label(move),
        move,
    )

assert real_legal_moves

print()
print("Real legal-move extraction validated.")


# # Cell 26 — Build Policy Features

# In[43]:


def move_label(
    move: Any,
) -> str:
    if isinstance(move, dict):
        return str(
            move.get("name")
            or move.get("Move Name")
            or move
        )

    for attribute in [
        "name",
        "move_name",
        "action",
        "attack_name",
    ]:
        value = getattr(
            move,
            attribute,
            None,
        )

        if value:
            return str(value)

    return str(move)


def numeric_move_value(
    move: Any,
    *candidate_names: str,
    default: float = 0.0,
) -> float:
    if isinstance(move, dict):
        for name in candidate_names:
            value = move.get(name)

            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return default

    for name in candidate_names:
        value = getattr(
            move,
            name,
            None,
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return default


def build_policy_state(
    battle_state: BattleState,
    legal_moves: list[Any],
) -> dict[str, Any]:

    current_side = (
        battle_state.player
        if battle_state.current_player == "Player"
        else battle_state.opponent
    )

    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    current_hp = float(
        current_side.active.current_hp
    )

    opponent_hp = float(
        opposing_side.active.current_hp
    )

    move_scores = {}
    heuristic_scores = {}
    search_scores = {}
    damage_scores = {}
    knockout_scores = {}
    defense_scores = {}
    healing_scores = {}
    move_lookup = {}

    policy_legal_moves = []

    for index, move in enumerate(legal_moves):
        base_label = move_label(move)

        label = base_label

        if label in move_lookup:
            label = f"{base_label} #{index + 1}"

        move_lookup[label] = move
        policy_legal_moves.append(label)

        damage = numeric_move_value(
            move,
            "damage",
            "damage_numeric",
            default=0.0,
        )

        knockout_value = (
            100.0
            if damage >= opponent_hp
            else 0.0
        )

        defensive_value = (
            50.0
            if "retreat" in label.lower()
            else 0.0
        )

        healing_value = (
            40.0
            if "heal" in label.lower()
            else 0.0
        )

        damage_scores[label] = damage
        knockout_scores[label] = knockout_value
        defense_scores[label] = defensive_value
        healing_scores[label] = healing_value

        move_scores[label] = damage

        heuristic_scores[label] = (
            damage
            + knockout_value
            + defensive_value * 0.25
            + healing_value * 0.25
        )

        search_scores[label] = (
            heuristic_scores[label]
        )

    return {
        "turn_number": battle_state.turn_number,
        "current_player": battle_state.current_player,
        "current_hp": current_hp,
        "opponent_hp": opponent_hp,
        "current_hand_size": current_side.hand_size,
        "opponent_hand_size": opposing_side.hand_size,
        "current_prizes": (
            current_side.prize_cards_remaining
        ),
        "opponent_prizes": (
            opposing_side.prize_cards_remaining
        ),
        "policy_legal_moves": policy_legal_moves,
        "move_lookup": move_lookup,
        "move_scores": move_scores,
        "heuristic_scores": heuristic_scores,
        "search_scores": search_scores,
        "damage_scores": damage_scores,
        "knockout_scores": knockout_scores,
        "defense_scores": defense_scores,
        "healing_scores": healing_scores,
    }


# # Cell 27 — Validate Policy Features

# In[44]:


policy_state = build_policy_state(
    test_state,
    real_legal_moves,
)

print("Policy-state keys:")

for key in policy_state:
    print(" -", key)

print()
print(
    "Policy legal moves:",
    policy_state["policy_legal_moves"],
)

assert policy_state["move_scores"]
assert policy_state["damage_scores"]
assert policy_state["search_scores"]
assert policy_state["move_lookup"]
assert len(
    policy_state["policy_legal_moves"]
) == len(real_legal_moves)

print()
print("Policy feature extraction validated.")


# # Cell 28 — Policy Search Engine Adapter

# In[46]:


from src.engine import (
    SearchResult,
    SearchStats,
)


class PolicySearchEngineAdapter:

    def __init__(
        self,
        policy_agent,
    ) -> None:
        self.policy_agent = policy_agent

    def search_depth(
        self,
        *,
        state: BattleState,
        depth: int,
        clear_table: bool = True,
    ) -> SearchResult:

        legal_moves = get_current_legal_moves(
            state
        )

        if not legal_moves:
            return SearchResult(
                best_move=None,
                score=0.0,
                completed_depth=depth,
                principal_variation=[],
                stats=SearchStats(),
            )

        policy_state = build_policy_state(
            state,
            legal_moves,
        )

        decision = self.policy_agent.choose_move(
            state=policy_state,
            legal_moves=(
                policy_state["policy_legal_moves"]
            ),
        )

        selected_move = policy_state[
            "move_lookup"
        ][decision.move]

        return SearchResult(
            best_move=selected_move,
            score=float(decision.score),
            completed_depth=depth,
            principal_variation=[
                selected_move
            ],
            stats=SearchStats(
                nodes=1,
            ),
        )


print("PolicySearchEngineAdapter created.")


# # Cell 29 — Validate One Adapted Opponent

# In[47]:


from src.tournament.match import (
    PokemonBattleAgent,
)


random_policy_agent = OPPONENT_POOL["Random"]

random_engine = PolicySearchEngineAdapter(
    random_policy_agent
)

random_battle_agent = PokemonBattleAgent(
    random_engine
)

adapted_decision = (
    random_battle_agent.choose_move(
        test_state,
        depth=6,
    )
)

print(adapted_decision)

assert adapted_decision.move in real_legal_moves
assert adapted_decision.search_depth == 6

print()
print("Policy adapter validated.")


# # Cell 30 — Build Adapted Opponent Agents

# In[48]:


def build_policy_battle_agent(
    opponent_name: str,
) -> PokemonBattleAgent:

    policy_agent = OPPONENT_POOL[
        opponent_name
    ]

    engine = PolicySearchEngineAdapter(
        policy_agent
    )

    return PokemonBattleAgent(
        engine
    )


ADAPTED_OPPONENTS = {
    name: build_policy_battle_agent(name)
    for name in OPPONENT_POOL
}

for name, agent in ADAPTED_OPPONENTS.items():
    print(
        f"{name:<12}",
        type(agent).__name__,
    )

assert len(ADAPTED_OPPONENTS) == 6

print()
print("All policy opponents adapted.")


# # Cell 31 — Build the Mixed-Opponent Match Factory

# In[49]:


OPPONENT_NAMES = list(
    ADAPTED_OPPONENTS
)

training_player_agent = (
    notebook27_helpers.create_battle_agent(
        seed=BASELINE_SEED,
    )
)


def mixed_match_factory(
    game_index: int,
) -> TournamentMatch:

    opponent_name = OPPONENT_NAMES[
        game_index % len(OPPONENT_NAMES)
    ]

    return TournamentMatch(
        player_agent=training_player_agent,
        opponent_agent=(
            ADAPTED_OPPONENTS[
                opponent_name
            ]
        ),
        player_name="TrainingAgent",
        opponent_name=opponent_name,
        search_depth=6,
        max_turns=100,
    )


sample_matches = [
    mixed_match_factory(i)
    for i in range(6)
]

for index, match in enumerate(
    sample_matches,
    start=1,
):
    print(
        f"Game {index}: "
        f"{match.player_name} vs "
        f"{match.opponent_name}"
    )

assert [
    match.opponent_name
    for match in sample_matches
] == OPPONENT_NAMES

print()
print("Mixed-opponent match factory validated.")


# # Cell 32 — Run One Match Against Every Opponent

# In[50]:


from src.tournament.runner import (
    run_two_agent_match,
)


validation_rows = []

for game_index, opponent_name in enumerate(
    OPPONENT_NAMES
):
    match = mixed_match_factory(
        game_index
    )

    state = initial_state_factory(
        game_index
    )

    result = run_two_agent_match(
        match,
        state,
        verbose=False,
    )

    validation_rows.append(
        {
            "game_index": game_index,
            "opponent": opponent_name,
            "winner": result.winner,
            "turns": result.turns,
            "final_score": result.final_score,
        }
    )

validation_df = pd.DataFrame(
    validation_rows
)

display(validation_df)

assert len(validation_df) == 6
assert validation_df["winner"].notna().all()
assert validation_df["turns"].gt(0).all()

print()
print(
    "One-match-per-opponent validation passed."
)


# # Cell 33 — Benchmark Runner

# In[51]:


def run_mixed_benchmark(
    num_games: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    rows = []

    start_time = time.perf_counter()

    for game_index in range(num_games):
        match = mixed_match_factory(
            game_index
        )

        state = initial_state_factory(
            game_index
        )

        result = run_two_agent_match(
            match,
            state,
            verbose=False,
        )

        rows.append(
            {
                "game_index": game_index,
                "opponent": match.opponent_name,
                "winner": result.winner,
                "turns": result.turns,
                "final_score": result.final_score,
                "transcript": result.transcript,
            }
        )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    results_df = pd.DataFrame(rows)

    summary = {
        "requested_games": num_games,
        "completed_games": int(
            len(results_df)
        ),
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": (
            num_games / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
        "average_turns": float(
            results_df["turns"].mean()
        ),
        "average_score": float(
            results_df["final_score"].mean()
        ),
        "unique_winners": int(
            results_df["winner"].nunique()
        ),
        "winner_counts": {
            str(key): int(value)
            for key, value in (
                results_df["winner"]
                .value_counts()
                .items()
            )
        },
        "opponent_counts": {
            str(key): int(value)
            for key, value in (
                results_df["opponent"]
                .value_counts()
                .items()
            )
        },
    }

    return results_df, summary


# # Cell 34 — Run the 100-Game Mixed Benchmark

# In[53]:


mixed_100_df, mixed_100_summary = (
    run_mixed_benchmark(
        100
    )
)

display(
    mixed_100_df.head()
)

print()
for key, value in mixed_100_summary.items():
    print(f"{key:22}: {value}")

assert len(mixed_100_df) == 100
assert sum(
    mixed_100_summary[
        "opponent_counts"
    ].values()
) == 100

print()
print("100-game mixed benchmark completed.")


# # Cell 35 — Validate Opponent Distribution

# In[55]:


opponent_distribution_100 = (
    mixed_100_df["opponent"]
    .value_counts()
    .sort_index()
    .rename("games")
    .reset_index()
)

opponent_distribution_100.columns = [
    "opponent",
    "games",
]

display(opponent_distribution_100)

distribution_gap = (
    opponent_distribution_100["games"].max()
    - opponent_distribution_100["games"].min()
)

assert distribution_gap <= 1

print()
print(
    "100-game opponent distribution validated."
)


# # Cell 36 — Extract Move Diversity from Transcripts

# In[58]:


import re


def extract_move_names(
    transcript: str,
) -> list[str]:

    move_names = []

    patterns = [
        r"used\s+(.+?)(?:\n|$)",
        r"Move Name:\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            transcript,
            flags=re.IGNORECASE,
        )

        move_names.extend(
            match.strip()
            for match in matches
            if match.strip()
        )

    return move_names


all_moves_100 = []

for transcript in mixed_100_df["transcript"]:
    all_moves_100.extend(
        extract_move_names(
            transcript
        )
    )

move_counts_100 = (
    pd.Series(
        all_moves_100,
        dtype="object",
    )
    .value_counts()
    .rename("count")
    .reset_index()
)

move_counts_100.columns = [
    "move",
    "count",
]

display(move_counts_100)

print()
print(
    "Unique moves:",
    len(move_counts_100),
)

assert len(all_moves_100) > 0

print()
print("Move extraction validated.")


# # Cell 37 — Compare Baseline and Mixed 100-Game Results

# In[59]:


baseline_vs_mixed = pd.DataFrame(
    [
        {
            "benchmark": "Notebook 28 baseline",
            "games": 100,
            "opponent_profiles": 1,
            "unique_winners": 1,
            "unique_moves": 2,
            "average_turns": 3.50,
            "average_score": 155.00,
            "games_per_second": 299.85,
        },
        {
            "benchmark": "Notebook 32 mixed",
            "games": int(
                mixed_100_summary[
                    "completed_games"
                ]
            ),
            "opponent_profiles": int(
                mixed_100_df[
                    "opponent"
                ].nunique()
            ),
            "unique_winners": int(
                mixed_100_df[
                    "winner"
                ].nunique()
            ),
            "unique_moves": int(
                len(move_counts_100)
            ),
            "average_turns": float(
                mixed_100_summary[
                    "average_turns"
                ]
            ),
            "average_score": float(
                mixed_100_summary[
                    "average_score"
                ]
            ),
            "games_per_second": float(
                mixed_100_summary[
                    "games_per_second"
                ]
            ),
        },
    ]
)

display(baseline_vs_mixed)

assert (
    baseline_vs_mixed.loc[
        1,
        "opponent_profiles",
    ]
    == 6
)

print()
print(
    "Opponent diversity increased:",
    "1 profile → 6 profiles",
)

print(
    "Move diversity changed:",
    "2 moves →",
    len(move_counts_100),
    "moves",
)

print()
print(
    "The current toy battle limits actual "
    "state and move diversity."
)


# # Cell 38 — Run the 1,000-Game Scaling Benchmark

# In[60]:


mixed_1000_df, mixed_1000_summary = (
    run_mixed_benchmark(
        1_000
    )
)

print("=" * 70)
print("1,000-GAME MIXED BENCHMARK")
print("=" * 70)

for key, value in (
    mixed_1000_summary.items()
):
    print(f"{key:22}: {value}")

assert len(mixed_1000_df) == 1_000

assert sum(
    mixed_1000_summary[
        "opponent_counts"
    ].values()
) == 1_000

print()
print(
    "1,000-game mixed benchmark completed."
)


# # Cell 39 — Validate the 1,000-Game Distribution

# In[61]:


distribution_1000 = (
    mixed_1000_df["opponent"]
    .value_counts()
    .sort_index()
    .rename("games")
    .reset_index()
)

distribution_1000.columns = [
    "opponent",
    "games",
]

display(distribution_1000)

distribution_gap_1000 = (
    distribution_1000["games"].max()
    - distribution_1000["games"].min()
)

assert distribution_gap_1000 <= 1
assert distribution_1000["games"].sum() == 1_000

print()
print(
    "1,000-game opponent distribution validated."
)


# #  Cell 40 — Extract 1,000-Game Move Statistics

# In[62]:


all_moves_1000 = []

for transcript in mixed_1000_df[
    "transcript"
]:
    all_moves_1000.extend(
        extract_move_names(
            transcript
        )
    )

move_counts_1000 = (
    pd.Series(
        all_moves_1000,
        dtype="object",
    )
    .value_counts()
    .rename("count")
    .reset_index()
)

move_counts_1000.columns = [
    "move",
    "count",
]

display(move_counts_1000)

print()
print(
    "Recorded move decisions:",
    len(all_moves_1000),
)

print(
    "Unique moves:",
    len(move_counts_1000),
)

assert len(all_moves_1000) > 0

print()
print(
    "1,000-game move statistics validated."
)


# # Cell 41 — Scaling Summary

# In[63]:


scaling_df = pd.DataFrame(
    [
        {
            "games": 100,
            "elapsed_seconds": mixed_100_summary["elapsed_seconds"],
            "games_per_second": mixed_100_summary["games_per_second"],
        },
        {
            "games": 1000,
            "elapsed_seconds": mixed_1000_summary["elapsed_seconds"],
            "games_per_second": mixed_1000_summary["games_per_second"],
        },
    ]
)

display(scaling_df)

print()

speedup = (
    scaling_df.iloc[1]["games_per_second"]
    /
    scaling_df.iloc[0]["games_per_second"]
)

print(f"Scaling ratio: {speedup:.2f}x")

assert scaling_df.iloc[1]["games"] == 1000

print()
print("Scaling summary validated.")


# # Cell 42 — Opponent Win Rates

# In[64]:


winner_table = (
    mixed_1000_df
    .groupby("opponent")["winner"]
    .value_counts()
    .unstack(fill_value=0)
)

display(winner_table)

print()

for opponent in winner_table.index:
    total = winner_table.loc[opponent].sum()

    for winner in winner_table.columns:

        pct = (
            100
            * winner_table.loc[opponent, winner]
            / total
        )

        print(
            f"{opponent:12}  "
            f"{winner:15} "
            f"{pct:6.2f}%"
        )

print()
print("Opponent win table validated.")


# # Cell 43 — Benchmark Summary Report

# In[65]:


summary_report = {
    "games": len(mixed_1000_df),
    "average_turns": mixed_1000_summary["average_turns"],
    "average_score": mixed_1000_summary["average_score"],
    "games_per_second": mixed_1000_summary["games_per_second"],
    "unique_opponents": mixed_1000_df["opponent"].nunique(),
    "unique_winners": mixed_1000_df["winner"].nunique(),
    "unique_moves": len(move_counts_1000),
}

for k, v in summary_report.items():
    print(f"{k:20}: {v}")

assert summary_report["games"] == 1000

print()
print("Benchmark summary validated.")


# # Cell 44 — Export Reports

# In[66]:


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

mixed_1000_df.to_csv(
    REPORT_DIR /
    "notebook32_mixed_benchmark.csv",
    index=False,
)

move_counts_1000.to_csv(
    REPORT_DIR /
    "notebook32_move_counts.csv",
    index=False,
)

distribution_1000.to_csv(
    REPORT_DIR /
    "notebook32_opponent_distribution.csv",
    index=False,
)

with open(
    REPORT_DIR /
    "notebook32_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_report,
        f,
        indent=2,
    )

print("Notebook 32 reports exported.")


# # Cell 45 — Final Validation

# In[67]:


print("=" * 70)
print("NOTEBOOK 32 FINAL VALIDATION")
print("=" * 70)

print()

print("Games:", len(mixed_1000_df))

print(
    "Opponent Profiles:",
    mixed_1000_df["opponent"].nunique(),
)

print(
    "Unique Winners:",
    mixed_1000_df["winner"].nunique(),
)

print(
    "Unique Moves:",
    len(move_counts_1000),
)

print(
    "Games/sec:",
    round(
        mixed_1000_summary["games_per_second"],
        2,
    ),
)

print(
    "Average Turns:",
    mixed_1000_summary["average_turns"],
)

print(
    "Average Score:",
    mixed_1000_summary["average_score"],
)

print()

print("Reports saved to:")

print(REPORT_DIR)

assert len(mixed_1000_df) == 1000
assert mixed_1000_df["opponent"].nunique() == 6
assert len(move_counts_1000) >= 2

print()
print("✅ NOTEBOOK 32 PASSED")


# # Cell 46 — Create battle_state_adapter.py

# In[69]:


OBSERVATION_ADAPTER_DIR = (
    PROJECT_ROOT
    / "src"
    / "observation_adapter"
)

OBSERVATION_ADAPTER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

battle_state_adapter_module = '''from __future__ import annotations

from typing import Any

from src.battle_state import BattleState


def move_label(
    move: Any,
) -> str:
    if isinstance(move, dict):
        return str(
            move.get("name")
            or move.get("Move Name")
            or move
        )

    for attribute in [
        "name",
        "move_name",
        "action",
        "attack_name",
    ]:
        value = getattr(
            move,
            attribute,
            None,
        )

        if value:
            return str(value)

    return str(move)


def numeric_move_value(
    move: Any,
    *candidate_names: str,
    default: float = 0.0,
) -> float:
    if isinstance(move, dict):
        for name in candidate_names:
            value = move.get(name)

            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return default

    for name in candidate_names:
        value = getattr(
            move,
            name,
            None,
        )

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return default


def build_policy_state(
    battle_state: BattleState,
    legal_moves: list[Any],
) -> dict[str, Any]:
    current_side = (
        battle_state.player
        if battle_state.current_player == "Player"
        else battle_state.opponent
    )

    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    current_hp = float(
        current_side.active.current_hp
    )

    opponent_hp = float(
        opposing_side.active.current_hp
    )

    move_scores: dict[str, float] = {}
    heuristic_scores: dict[str, float] = {}
    search_scores: dict[str, float] = {}
    damage_scores: dict[str, float] = {}
    knockout_scores: dict[str, float] = {}
    defense_scores: dict[str, float] = {}
    healing_scores: dict[str, float] = {}
    move_lookup: dict[str, Any] = {}
    policy_legal_moves: list[str] = []

    for index, move in enumerate(legal_moves):
        base_label = move_label(move)
        label = base_label

        if label in move_lookup:
            label = f"{base_label} #{index + 1}"

        move_lookup[label] = move
        policy_legal_moves.append(label)

        damage = numeric_move_value(
            move,
            "damage",
            "damage_numeric",
            default=0.0,
        )

        knockout_value = (
            100.0
            if damage >= opponent_hp
            else 0.0
        )

        defensive_value = (
            50.0
            if "retreat" in label.lower()
            else 0.0
        )

        healing_value = (
            40.0
            if "heal" in label.lower()
            else 0.0
        )

        damage_scores[label] = damage
        knockout_scores[label] = knockout_value
        defense_scores[label] = defensive_value
        healing_scores[label] = healing_value
        move_scores[label] = damage

        heuristic_scores[label] = (
            damage
            + knockout_value
            + defensive_value * 0.25
            + healing_value * 0.25
        )

        search_scores[label] = (
            heuristic_scores[label]
        )

    return {
        "turn_number": battle_state.turn_number,
        "current_player": battle_state.current_player,
        "current_hp": current_hp,
        "opponent_hp": opponent_hp,
        "current_hand_size": current_side.hand_size,
        "opponent_hand_size": opposing_side.hand_size,
        "current_prizes": (
            current_side.prize_cards_remaining
        ),
        "opponent_prizes": (
            opposing_side.prize_cards_remaining
        ),
        "policy_legal_moves": policy_legal_moves,
        "move_lookup": move_lookup,
        "move_scores": move_scores,
        "heuristic_scores": heuristic_scores,
        "search_scores": search_scores,
        "damage_scores": damage_scores,
        "knockout_scores": knockout_scores,
        "defense_scores": defense_scores,
        "healing_scores": healing_scores,
    }


__all__ = [
    "build_policy_state",
    "move_label",
    "numeric_move_value",
]
'''

battle_state_adapter_file = (
    OBSERVATION_ADAPTER_DIR
    / "battle_state_adapter.py"
)

battle_state_adapter_file.write_text(
    battle_state_adapter_module,
    encoding="utf-8",
)

print("[OK]", battle_state_adapter_file)


# # Cell 47 — Create policy_search_adapter.py

# In[70]:


policy_search_adapter_module = '''from __future__ import annotations

from typing import Any

from src.battle_state import BattleState
from src.engine import SearchResult, SearchStats
from src.legal_moves import get_current_legal_moves

from .battle_state_adapter import build_policy_state


class PolicySearchEngineAdapter:
    def __init__(
        self,
        policy_agent: Any,
    ) -> None:
        self.policy_agent = policy_agent

    def search_depth(
        self,
        *,
        state: BattleState,
        depth: int,
        clear_table: bool = True,
    ) -> SearchResult:
        legal_moves = get_current_legal_moves(
            state
        )

        if not legal_moves:
            return SearchResult(
                best_move=None,
                score=0.0,
                completed_depth=depth,
                principal_variation=[],
                stats=SearchStats(),
            )

        policy_state = build_policy_state(
            state,
            legal_moves,
        )

        decision = self.policy_agent.choose_move(
            state=policy_state,
            legal_moves=(
                policy_state["policy_legal_moves"]
            ),
        )

        selected_move = policy_state[
            "move_lookup"
        ][decision.move]

        return SearchResult(
            best_move=selected_move,
            score=float(decision.score),
            completed_depth=depth,
            principal_variation=[
                selected_move
            ],
            stats=SearchStats(
                nodes=1,
            ),
        )


__all__ = [
    "PolicySearchEngineAdapter",
]
'''

policy_search_adapter_file = (
    OBSERVATION_ADAPTER_DIR
    / "policy_search_adapter.py"
)

policy_search_adapter_file.write_text(
    policy_search_adapter_module,
    encoding="utf-8",
)

print("[OK]", policy_search_adapter_file)


# # Cell 48 — Create __init__.py

# In[71]:


observation_init_module = '''from .battle_state_adapter import (
    build_policy_state,
    move_label,
    numeric_move_value,
)

from .policy_search_adapter import (
    PolicySearchEngineAdapter,
)

__all__ = [
    "PolicySearchEngineAdapter",
    "build_policy_state",
    "move_label",
    "numeric_move_value",
]
'''

observation_init_file = (
    OBSERVATION_ADAPTER_DIR
    / "__init__.py"
)

observation_init_file.write_text(
    observation_init_module,
    encoding="utf-8",
)

print("[OK]", observation_init_file)


# # Cell 49 — Validate Production Imports

# In[72]:


import importlib
import sys


cached_modules = [
    name
    for name in list(sys.modules)
    if (
        name == "src.observation_adapter"
        or name.startswith(
            "src.observation_adapter."
        )
    )
]

for module_name in cached_modules:
    del sys.modules[module_name]

importlib.invalidate_caches()

from src.observation_adapter import (
    PolicySearchEngineAdapter as ProductionPolicyAdapter,
    build_policy_state as production_build_policy_state,
    move_label as production_move_label,
)

production_policy_state = (
    production_build_policy_state(
        test_state,
        real_legal_moves,
    )
)

assert (
    production_policy_state[
        "policy_legal_moves"
    ]
    == [
        "Tera",
        "Evolution Burst",
    ]
)

production_random_engine = (
    ProductionPolicyAdapter(
        OPPONENT_POOL["Random"]
    )
)

production_random_agent = (
    PokemonBattleAgent(
        production_random_engine
    )
)

production_decision = (
    production_random_agent.choose_move(
        test_state,
        depth=6,
    )
)

assert (
    production_decision.move
    in real_legal_moves
)

print(
    "Production observation adapter validated."
)


# # Cell 50 — Validate Files

# In[74]:


expected_adapter_files = [
    battle_state_adapter_file,
    policy_search_adapter_file,
    observation_init_file,
]

for path in expected_adapter_files:
    assert path.exists(), path

    print(
        f"[OK] {path.name:<32} "
        f"{path.stat().st_size:>8,} bytes"
    )

print()
print(
    "Observation adapter package completed."
)


# In[ ]:




