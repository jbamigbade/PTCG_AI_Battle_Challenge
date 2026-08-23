#!/usr/bin/env python
# coding: utf-8

# # Notebook 33 - Rich Observation Adapter & Feature Engineering
# 
# ## Objectives
# 
# - Build a production-quality feature engineering pipeline
# - Generate normalized observation vectors
# - Extract battle, player, opponent, and move features
# - Validate feature consistency
# - Prepare observations for future machine-learning training

# ## 1. Project Setup
# 
# This section:
# 
# - locates the project root
# - configures reusable project paths
# - imports the required Python libraries
# - validates the production observation adapter
# - creates the Notebook 33 report directory

# # Imports

# In[1]:


from __future__ import annotations

import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Locate project root
# ---------------------------------------------------------

def find_project_root(
    start_path: Path | None = None,
) -> Path:
    """
    Locate the project root by searching upward for both
    the 'src' and 'notebooks' directories.
    """
    current_path = (
        start_path or Path.cwd()
    ).resolve()

    candidate_paths = [
        current_path,
        *current_path.parents,
    ]

    for candidate in candidate_paths:
        if (
            (candidate / "src").is_dir()
            and (candidate / "notebooks").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Expected a parent directory containing "
        "'src' and 'notebooks'."
    )


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK33_REPORT_DIR = (
    REPORTS_DIR / "notebook33"
)
OBSERVATION_ADAPTER_DIR = (
    SRC_DIR / "observation_adapter"
)


# ---------------------------------------------------------
# Make project imports available
# ---------------------------------------------------------

project_root_string = str(PROJECT_ROOT)

if project_root_string not in sys.path:
    sys.path.insert(
        0,
        project_root_string,
    )


# ---------------------------------------------------------
# Create output directories
# ---------------------------------------------------------

NOTEBOOK33_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_SEED = 33

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# Display configuration
# ---------------------------------------------------------

pd.set_option(
    "display.max_columns",
    100,
)

pd.set_option(
    "display.width",
    160,
)

pd.set_option(
    "display.max_colwidth",
    120,
)


# ---------------------------------------------------------
# Setup validation
# ---------------------------------------------------------

required_paths = {
    "Project root": PROJECT_ROOT,
    "Source directory": SRC_DIR,
    "Notebook directory": NOTEBOOKS_DIR,
    "Observation adapter": OBSERVATION_ADAPTER_DIR,
    "Notebook 33 reports": NOTEBOOK33_REPORT_DIR,
}

print("NOTEBOOK 33 — PROJECT SETUP")
print("=" * 60)

for label, path in required_paths.items():
    status = (
        "OK"
        if path.exists()
        else "MISSING"
    )

    print(
        f"{label:<24}: "
        f"{status:<8} {path}"
    )

print("-" * 60)
print(
    f"Random seed             : {RANDOM_SEED}"
)
print(
    f"Python version          : "
    f"{sys.version.split()[0]}"
)
print(
    f"NumPy version           : "
    f"{np.__version__}"
)
print(
    f"Pandas version          : "
    f"{pd.__version__}"
)


# ## Validate Production Observation Adapter

# In[2]:


# ---------------------------------------------------------
# Validate production observation-adapter files
# ---------------------------------------------------------

expected_adapter_files = [
    OBSERVATION_ADAPTER_DIR
    / "__init__.py",

    OBSERVATION_ADAPTER_DIR
    / "battle_state_adapter.py",

    OBSERVATION_ADAPTER_DIR
    / "policy_search_adapter.py",
]

adapter_file_records = []

for file_path in expected_adapter_files:
    adapter_file_records.append(
        {
            "file": file_path.name,
            "exists": file_path.exists(),
            "size_bytes": (
                file_path.stat().st_size
                if file_path.exists()
                else 0
            ),
            "path": str(file_path),
        }
    )

adapter_files_df = pd.DataFrame(
    adapter_file_records
)

display(adapter_files_df)

assert adapter_files_df[
    "exists"
].all(), (
    "One or more observation-adapter "
    "files are missing."
)

print(
    "\nProduction observation-adapter "
    "files validated."
)


# # Validate production imports

# In[3]:


# ---------------------------------------------------------
# Validate production imports
# ---------------------------------------------------------

importlib.invalidate_caches()

from src.observation_adapter import (
    PolicySearchEngineAdapter,
    build_policy_state,
    move_label,
    numeric_move_value,
)

validated_imports = {
    "PolicySearchEngineAdapter": (
        PolicySearchEngineAdapter
    ),
    "build_policy_state": (
        build_policy_state
    ),
    "move_label": (
        move_label
    ),
    "numeric_move_value": (
        numeric_move_value
    ),
}

for name, imported_object in (
    validated_imports.items()
):
    print(
        f"[OK] {name:<28} "
        f"{imported_object}"
    )

assert all(
    callable(imported_object)
    for imported_object in [
        PolicySearchEngineAdapter,
        build_policy_state,
        move_label,
        numeric_move_value,
    ]
)

print()
print(
    "✅ SECTION 1 PASSED — "
    "Notebook 33 setup is complete."
)


# # 2. Inspect the Current Battle-State Structure
# 
# Before creating richer features, we need to inspect the actual production objects used by the simulator.
# 
# This section will:
# 
# - inspect the `BattleState` class
# - inspect the player-side and active Pokémon structures
# - inspect legal move dictionaries
# - identify which attributes are currently available
# - prevent the feature pipeline from assuming fields that do not exist

# ###  Cell 1 - Inspect production classes and function signatures

# In[4]:


# ---------------------------------------------------------
# Inspect production classes and function signatures
# ---------------------------------------------------------

import inspect

from src.battle_state import BattleState
from src.legal_moves import get_current_legal_moves


objects_to_inspect = {
    "BattleState": BattleState,
    "get_current_legal_moves": get_current_legal_moves,
}

for object_name, production_object in (
    objects_to_inspect.items()
):
    print("=" * 80)
    print(object_name)
    print("=" * 80)

    try:
        print(
            inspect.getsource(
                production_object
            )
        )
    except (
        OSError,
        TypeError,
    ):
        print(
            f"Source code could not be displayed for "
            f"{object_name}."
        )

    print()


# ###  Cell 2 - Locate an existing battle-state factory or test state

# In[5]:


# ---------------------------------------------------------
# Locate an existing battle-state factory or test state
# ---------------------------------------------------------

candidate_module_names = [
    "src.tournament",
    "src.battle",
    "src.battle_engine",
    "src.simulation",
    "src.game_setup",
]

discovered_factories = []

for module_name in candidate_module_names:
    try:
        module = importlib.import_module(
            module_name
        )
    except ModuleNotFoundError:
        continue

    for attribute_name in dir(module):
        lowercase_name = (
            attribute_name.lower()
        )

        if any(
            keyword in lowercase_name
            for keyword in [
                "state",
                "battle",
                "match",
                "setup",
                "factory",
            ]
        ):
            attribute = getattr(
                module,
                attribute_name,
            )

            if callable(attribute):
                discovered_factories.append(
                    {
                        "module": module_name,
                        "name": attribute_name,
                        "object": attribute,
                    }
                )

factory_summary_df = pd.DataFrame(
    [
        {
            "module": record["module"],
            "name": record["name"],
        }
        for record in discovered_factories
    ]
)

display(factory_summary_df)

print(
    f"\nDiscovered callable candidates: "
    f"{len(discovered_factories)}"
)


# ###  Cell 3 - Locate BattleState usage throughout the project

# In[6]:


# ---------------------------------------------------------
# Locate BattleState usage throughout the project
# ---------------------------------------------------------

battle_state_usage = []

for python_file in SRC_DIR.rglob("*.py"):

    try:
        source = python_file.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        continue

    if "BattleState" in source:

        battle_state_usage.append(
            {
                "file": str(
                    python_file.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "mentions": source.count(
                    "BattleState"
                ),
            }
        )

battle_state_usage_df = (
    pd.DataFrame(
        battle_state_usage
    )
    .sort_values(
        "mentions",
        ascending=False,
    )
)

display(
    battle_state_usage_df
)

print(
    f"\nFiles mentioning BattleState: "
    f"{len(battle_state_usage_df)}"
)


# ##  Section 3. Feature Engineering Architecture
# 
# Rather than generating a single large feature vector immediately,
# Notebook 33 extracts features in logical groups.
# 
# The groups are:
# 
# 1. Player features
# 2. Opponent features
# 3. Battle features
# 4. Move features
# 
# These groups will later be combined into a normalized observation vector suitable for heuristic agents and machine-learning models.

# ## Cell 1 - Feature Categories

# In[7]:


# ---------------------------------------------------------
# Notebook 33 Feature Categories
# ---------------------------------------------------------

FEATURE_GROUPS = {
    "player": [
        "current_hp",
        "current_hp_ratio",
        "hand_size",
        "prize_cards_remaining",
    ],

    "opponent": [
        "opponent_hp",
        "opponent_hp_ratio",
        "opponent_hand_size",
        "opponent_prize_cards_remaining",
    ],

    "battle": [
        "turn_number",
        "current_player",
    ],

    "moves": [
        "damage_scores",
        "knockout_scores",
        "defense_scores",
        "healing_scores",
        "heuristic_scores",
    ],
}

feature_summary = []

for group_name, features in FEATURE_GROUPS.items():

    feature_summary.append(
        {
            "Group": group_name,
            "Feature Count": len(features),
            "Features": ", ".join(features),
        }
    )

feature_groups_df = pd.DataFrame(
    feature_summary
)

display(feature_groups_df)

print()
print(
    f"Total Feature Groups : {len(FEATURE_GROUPS)}"
)

print(
    f"Total Planned Features : "
    f"{sum(len(v) for v in FEATURE_GROUPS.values())}"
)


# # Section 4. Player Feature Engineering
# 
# This section extracts numerical features describing the current player.
# 
# These features become reusable building blocks for heuristic evaluation,
# machine-learning datasets, and neural-network observations.
# 
# The feature extractor is intentionally independent of any specific AI
# algorithm so that every downstream component can reuse the same
# representation.

# # Player Feature Extraction

# In[8]:


# ---------------------------------------------------------
# Player Feature Extraction
# ---------------------------------------------------------

def extract_player_features(
    policy_state: dict[str, Any],
) -> dict[str, float]:

    current_hp = float(
        policy_state["current_hp"]
    )

    hand_size = float(
        policy_state["current_hand_size"]
    )

    prizes = float(
        policy_state["current_prizes"]
    )

    opponent_hp = max(
        float(
            policy_state["opponent_hp"]
        ),
        1.0,
    )

    hp_ratio = (
        current_hp
        / max(
            current_hp,
            opponent_hp,
        )
    )

    return {
        "current_hp": current_hp,
        "current_hp_ratio": hp_ratio,
        "hand_size": hand_size,
        "prize_cards_remaining": prizes,
    }


print(
    "Player feature extractor created."
)


# ## Inspect the real BattleState dataclass constructors

# In[9]:


# ---------------------------------------------------------
# Inspect the real BattleState dataclass constructors
# ---------------------------------------------------------

import inspect
from dataclasses import fields, is_dataclass

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)


classes_to_inspect = [
    PokemonState,
    PlayerState,
    BattleState,
]

for class_object in classes_to_inspect:
    print("=" * 70)
    print(class_object.__name__)
    print("=" * 70)

    print(
        "Signature:",
        inspect.signature(class_object),
    )

    if is_dataclass(class_object):
        print("\nDataclass fields:")

        for field in fields(class_object):
            print(
                f"  {field.name:<28} "
                f"type={field.type!s:<25} "
                f"default={field.default!r}"
            )

    print()


# #  Display PokemonState and PlayerState source

# In[10]:


# ---------------------------------------------------------
# Display PokemonState and PlayerState source
# ---------------------------------------------------------

for class_object in [
    PokemonState,
    PlayerState,
]:
    print("=" * 80)
    print(class_object.__name__)
    print("=" * 80)

    try:
        print(
            inspect.getsource(
                class_object
            )
        )
    except OSError:
        print(
            "Source code could not be displayed."
        )

    print()


# ## Build a valid production BattleState for Notebook 33

# In[11]:


# ---------------------------------------------------------
# Build a valid production BattleState for Notebook 33
# ---------------------------------------------------------

from src.battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)
from src.legal_moves import get_current_legal_moves


eevee_card = {
    "name": "Eevee ex",
    "hp": 200,
    "attacks": [
        {
            "name": "Tera",
            "damage": 0.0,
            "energy_cost": 0,
            "effect": "Tera ability.",
        },
        {
            "name": "Evolution Burst",
            "damage": 60.0,
            "energy_cost": 2,
            "effect": "Deals 60 damage.",
        },
    ],
}

electrike_card = {
    "name": "Electrike",
    "hp": 70,
    "attacks": [
        {
            "name": "Thunder Jolt",
            "damage": 30.0,
            "energy_cost": 1,
            "effect": "Deals 30 damage.",
        },
    ],
}


test_state = BattleState(
    player=PlayerState(
        active=PokemonState(
            card=eevee_card,
            current_hp=200.0,
            attached_energy=2,
            status=None,
            damage=0.0,
            is_active=True,
        ),
        bench=[],
        prize_cards_remaining=6,
        hand_size=5,
    ),
    opponent=PlayerState(
        active=PokemonState(
            card=electrike_card,
            current_hp=70.0,
            attached_energy=1,
            status=None,
            damage=0.0,
            is_active=True,
        ),
        bench=[],
        prize_cards_remaining=6,
        hand_size=5,
    ),
    turn_number=1,
    current_player="Player",
)


real_legal_moves = get_current_legal_moves(
    test_state
)

production_policy_state = build_policy_state(
    test_state,
    real_legal_moves,
)

print("Live production battle state created.")
print(
    "Player active:",
    test_state.player.active.card["name"],
)
print(
    "Opponent active:",
    test_state.opponent.active.card["name"],
)
print(
    "Legal moves:",
    production_policy_state[
        "policy_legal_moves"
    ],
)


# ## Rich player feature extraction

# In[12]:


# ---------------------------------------------------------
# Rich player feature extraction
# ---------------------------------------------------------

def card_numeric_value(
    card: dict[str, Any],
    *candidate_keys: str,
    default: float = 0.0,
) -> float:
    """
    Safely extract a numerical value from a card dictionary.
    """
    for key in candidate_keys:
        value = card.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return float(default)


def extract_player_features(
    battle_state: BattleState,
) -> dict[str, float]:
    """
    Extract features for whichever side currently has the turn.
    """
    current_side = (
        battle_state.player
        if battle_state.current_player == "Player"
        else battle_state.opponent
    )

    active = current_side.active
    card = active.card

    current_hp = float(
        active.current_hp
    )

    max_hp = card_numeric_value(
        card,
        "hp",
        "HP",
        "max_hp",
        default=current_hp,
    )

    safe_max_hp = max(
        max_hp,
        1.0,
    )

    hp_ratio = max(
        0.0,
        min(
            current_hp / safe_max_hp,
            1.0,
        ),
    )

    status_present = float(
        active.status is not None
    )

    return {
        "current_hp": current_hp,
        "maximum_hp": max_hp,
        "current_hp_ratio": hp_ratio,
        "attached_energy": float(
            active.attached_energy
        ),
        "damage_taken": float(
            active.damage
        ),
        "bench_size": float(
            len(current_side.bench)
        ),
        "hand_size": float(
            current_side.hand_size
        ),
        "prize_cards_remaining": float(
            current_side.prize_cards_remaining
        ),
        "status_present": status_present,
    }


print(
    "Rich player feature extractor created."
)


# ## Validate the real extractor

# In[13]:


# ---------------------------------------------------------
# Validate rich player features
# ---------------------------------------------------------

player_features = extract_player_features(
    test_state
)

player_features_df = pd.DataFrame(
    player_features.items(),
    columns=[
        "Feature",
        "Value",
    ],
)

display(
    player_features_df
)

expected_player_features = {
    "current_hp": 200.0,
    "maximum_hp": 200.0,
    "current_hp_ratio": 1.0,
    "attached_energy": 2.0,
    "damage_taken": 0.0,
    "bench_size": 0.0,
    "hand_size": 5.0,
    "prize_cards_remaining": 6.0,
    "status_present": 0.0,
}

for feature_name, expected_value in (
    expected_player_features.items()
):
    actual_value = player_features[
        feature_name
    ]

    assert math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ), (
        f"{feature_name}: expected "
        f"{expected_value}, got {actual_value}"
    )

assert all(
    math.isfinite(value)
    for value in player_features.values()
)

assert (
    0.0
    <= player_features[
        "current_hp_ratio"
    ]
    <= 1.0
)

print()
print(
    f"Player features extracted: "
    f"{len(player_features)}"
)
print(
    "✅ PLAYER FEATURE EXTRACTION PASSED"
)


# # Section 5 - Opponent Feature Engineering
# 
# This section extracts numerical features describing the opposing side.
# 
# The opponent representation mirrors the player representation so that the
# AI can compare both sides consistently and calculate relative advantages.

# ## Opponent feature extraction

# In[14]:


# ---------------------------------------------------------
# Opponent feature extraction
# ---------------------------------------------------------

def extract_opponent_features(
    battle_state: BattleState,
) -> dict[str, float]:
    """
    Extract features for the side that does not currently have the turn.
    """
    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    active = opposing_side.active
    card = active.card

    opponent_hp = float(
        active.current_hp
    )

    opponent_maximum_hp = card_numeric_value(
        card,
        "hp",
        "HP",
        "max_hp",
        default=opponent_hp,
    )

    safe_maximum_hp = max(
        opponent_maximum_hp,
        1.0,
    )

    opponent_hp_ratio = max(
        0.0,
        min(
            opponent_hp / safe_maximum_hp,
            1.0,
        ),
    )

    opponent_status_present = float(
        active.status is not None
    )

    return {
        "opponent_hp": opponent_hp,
        "opponent_maximum_hp": (
            opponent_maximum_hp
        ),
        "opponent_hp_ratio": (
            opponent_hp_ratio
        ),
        "opponent_attached_energy": float(
            active.attached_energy
        ),
        "opponent_damage_taken": float(
            active.damage
        ),
        "opponent_bench_size": float(
            len(opposing_side.bench)
        ),
        "opponent_hand_size": float(
            opposing_side.hand_size
        ),
        "opponent_prize_cards_remaining": float(
            opposing_side.prize_cards_remaining
        ),
        "opponent_status_present": (
            opponent_status_present
        ),
    }


print(
    "Rich opponent feature extractor created."
)


# ## Validate the opponent extractor

# In[15]:


# ---------------------------------------------------------
# Validate rich opponent features
# ---------------------------------------------------------

opponent_features = (
    extract_opponent_features(
        test_state
    )
)

opponent_features_df = pd.DataFrame(
    opponent_features.items(),
    columns=[
        "Feature",
        "Value",
    ],
)

display(
    opponent_features_df
)

expected_opponent_features = {
    "opponent_hp": 70.0,
    "opponent_maximum_hp": 70.0,
    "opponent_hp_ratio": 1.0,
    "opponent_attached_energy": 1.0,
    "opponent_damage_taken": 0.0,
    "opponent_bench_size": 0.0,
    "opponent_hand_size": 5.0,
    "opponent_prize_cards_remaining": 6.0,
    "opponent_status_present": 0.0,
}

for feature_name, expected_value in (
    expected_opponent_features.items()
):
    actual_value = opponent_features[
        feature_name
    ]

    assert math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ), (
        f"{feature_name}: expected "
        f"{expected_value}, got {actual_value}"
    )

assert all(
    math.isfinite(value)
    for value in opponent_features.values()
)

assert (
    0.0
    <= opponent_features[
        "opponent_hp_ratio"
    ]
    <= 1.0
)

print()
print(
    f"Opponent features extracted: "
    f"{len(opponent_features)}"
)
print(
    "✅ OPPONENT FEATURE EXTRACTION PASSED"
)


# # Section 6. Battle Feature Engineering
# 
# This section extracts features that describe the overall game state rather than either player individually.
# 
# These features capture:
# 
# - turn progression
# - whose turn it is
# - HP advantage
# - Energy advantage
# - Bench advantage
# - Hand advantage
# - Prize advantage
# - current tempo

# # 1. Battle-level feature extraction

# In[21]:


# ---------------------------------------------------------
# Battle-level feature extraction
# ---------------------------------------------------------

def extract_battle_features(
    battle_state: BattleState,
) -> dict[str, float]:
    """
    Extract features describing the overall battle position.
    """
    player_features = extract_player_features(
        battle_state
    )

    opponent_features = (
        extract_opponent_features(
            battle_state
        )
    )

    turn_number = float(
        battle_state.turn_number
    )

    current_player_is_player = float(
        battle_state.current_player
        == "Player"
    )

    hp_advantage = (
        player_features[
            "current_hp_ratio"
        ]
        - opponent_features[
            "opponent_hp_ratio"
        ]
    )

    energy_advantage = (
        player_features[
            "attached_energy"
        ]
        - opponent_features[
            "opponent_attached_energy"
        ]
    )

    bench_advantage = (
        player_features[
            "bench_size"
        ]
        - opponent_features[
            "opponent_bench_size"
        ]
    )

    hand_advantage = (
        player_features[
            "hand_size"
        ]
        - opponent_features[
            "opponent_hand_size"
        ]
    )

    # Fewer remaining prizes means the side is closer to winning.
    prize_advantage = (
        opponent_features[
            "opponent_prize_cards_remaining"
        ]
        - player_features[
            "prize_cards_remaining"
        ]
    )

    tempo_score = (
        hp_advantage
        + 0.10 * energy_advantage
        + 0.05 * bench_advantage
        + 0.02 * hand_advantage
        + 0.15 * prize_advantage
    )

    return {
        "turn_number": turn_number,
        "current_player_is_player": (
            current_player_is_player
        ),
        "hp_advantage": hp_advantage,
        "energy_advantage": (
            energy_advantage
        ),
        "bench_advantage": (
            bench_advantage
        ),
        "hand_advantage": (
            hand_advantage
        ),
        "prize_advantage": (
            prize_advantage
        ),
        "tempo_score": tempo_score,
    }


print(
    "Battle feature extractor created."
)


# # 2. Validate the battle extractor

# In[22]:


# ---------------------------------------------------------
# Validate battle-level features
# ---------------------------------------------------------

battle_features = extract_battle_features(
    test_state
)

battle_features_df = pd.DataFrame(
    battle_features.items(),
    columns=[
        "Feature",
        "Value",
    ],
)

display(
    battle_features_df
)

expected_battle_features = {
    "turn_number": 1.0,
    "current_player_is_player": 1.0,
    "hp_advantage": 0.0,
    "energy_advantage": 1.0,
    "bench_advantage": 0.0,
    "hand_advantage": 0.0,
    "prize_advantage": 0.0,
    "tempo_score": 0.1,
}

for feature_name, expected_value in (
    expected_battle_features.items()
):
    actual_value = battle_features[
        feature_name
    ]

    assert math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ), (
        f"{feature_name}: expected "
        f"{expected_value}, got "
        f"{actual_value}"
    )

assert all(
    math.isfinite(value)
    for value in battle_features.values()
)

print()
print(
    f"Battle features extracted: "
    f"{len(battle_features)}"
)
print(
    "✅ BATTLE FEATURE EXTRACTION PASSED"
)


# # 7. Inspect Current Legal Move Structure
# 
# ### Before engineering move features, inspect the production legal-move objects returned by the simulator.
# 
# ### This inspection verifies the current structure, available fields, and placeholder values so that the move feature extractor is built against the real production format instead of assumptions.

# In[24]:


# ---------------------------------------------------------
# Inspect current legal move structure
# ---------------------------------------------------------

# legal moves from the current production test state.
real_legal_moves = get_current_legal_moves(
    test_state
)

print("LEGAL MOVE INSPECTION")
print("=" * 70)

legal_move_records = []

for index, move in enumerate(
    real_legal_moves,
    start=1,
):
    print()
    print(f"Move {index}")
    print("-" * 70)
    print(
        "Type:",
        type(move).__name__,
    )
    print(
        "Raw value:",
        move,
    )

    if isinstance(move, dict):
        print(
            "Keys:",
            sorted(move.keys()),
        )

        for key, value in move.items():
            print(
                f"  {key:<20}: {value!r}"
            )

        move_name = str(
            move.get(
                "name",
                "Unknown Move",
            )
            or "Unknown Move"
        )

        try:
            move_damage = float(
                move.get(
                    "damage",
                    0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            move_damage = 0.0

        try:
            move_energy_cost = float(
                move.get(
                    "energy_cost",
                    0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            move_energy_cost = 0.0

        move_effect = move.get(
            "effect"
        )

    else:
        move_name = move_label(
            move
        )

        move_damage = numeric_move_value(
            move,
            "damage",
            "damage_numeric",
            default=0.0,
        )

        move_energy_cost = numeric_move_value(
            move,
            "energy_cost",
            "cost",
            default=0.0,
        )

        move_effect = getattr(
            move,
            "effect",
            None,
        )

    legal_move_records.append(
        {
            "move_number": index,
            "name": move_name,
            "damage": move_damage,
            "energy_cost": move_energy_cost,
            "effect": move_effect,
        }
    )

print()
print(
    f"Total legal moves: "
    f"{len(real_legal_moves)}"
)

legal_move_structure_df = pd.DataFrame(
    legal_move_records
)

display(
    legal_move_structure_df
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(
    legal_move_structure_df
) == len(
    real_legal_moves
), (
    "The inspection table does not contain "
    "one row for every legal move."
)

assert not legal_move_structure_df.empty, (
    "No legal moves were returned."
)

assert legal_move_structure_df[
    "name"
].notna().all(), (
    "One or more legal moves has a missing name."
)

assert np.isfinite(
    legal_move_structure_df[
        [
            "damage",
            "energy_cost",
        ]
    ].to_numpy(
        dtype=float
    )
).all(), (
    "One or more numeric move values is not finite."
)

assert (
    legal_move_structure_df[
        "damage"
    ]
    >= 0
).all(), (
    "Move damage must not be negative."
)

assert (
    legal_move_structure_df[
        "energy_cost"
    ]
    >= 0
).all(), (
    "Move energy cost must not be negative."
)

print()
print(
    "✅ LEGAL MOVE STRUCTURE INSPECTION PASSED"
)


# # Section 8. Move Feature Engineering
# 
# This section extracts numerical features describing every legal move
# available from the current battle state.
# 
# Unlike the previous sections, move features are generated for **each
# available action**, allowing heuristic agents and future machine-learning
# models to compare candidate moves using a consistent feature space.
# 
# Each move is represented by:
# 
# - damage
# - energy cost
# - knockout potential
# - damage efficiency
# - heuristic evaluation

# ##  Inspect real legal move dictionaries

# In[19]:


# ---------------------------------------------------------
# Inspect real legal move dictionaries
# ---------------------------------------------------------

print("LEGAL MOVE INSPECTION")
print("=" * 70)

for index, move in enumerate(
    real_legal_moves,
    start=1,
):
    print(
        f"\nMove {index}"
    )
    print("-" * 70)
    print(
        "Type:",
        type(move).__name__,
    )
    print(
        "Raw value:",
        move,
    )

    if isinstance(move, dict):
        print(
            "Keys:",
            sorted(move.keys()),
        )

        for key, value in move.items():
            print(
                f"  {key:<20}: "
                f"{value!r}"
            )

print()
print(
    f"Total legal moves: "
    f"{len(real_legal_moves)}"
)


# ## Cell 2 — Build the Move Feature Extractor

# In[25]:


# ---------------------------------------------------------
# Build move feature extractor
# ---------------------------------------------------------

def extract_move_features(
    move: Any,
    battle_state: BattleState,
    move_index: int,
) -> dict[str, float | str]:
    """
    Convert one legal move into a consistent feature record.

    The function is intentionally robust to incomplete move
    metadata such as 'Unknown Move', missing effects, or zero damage.
    """
    move_name = move_label(
        move
    )

    damage = numeric_move_value(
        move,
        "damage",
        "damage_numeric",
        default=0.0,
    )

    energy_cost = numeric_move_value(
        move,
        "energy_cost",
        "cost",
        default=0.0,
    )

    opposing_side = (
        battle_state.opponent
        if battle_state.current_player == "Player"
        else battle_state.player
    )

    opponent_hp = max(
        float(
            opposing_side.active.current_hp
        ),
        0.0,
    )

    safe_energy_cost = max(
        float(energy_cost),
        1.0,
    )

    damage_per_energy = (
        float(damage)
        / safe_energy_cost
    )

    knockout_possible = float(
        opponent_hp > 0
        and float(damage) >= opponent_hp
    )

    estimated_hp_after_attack = max(
        opponent_hp - float(damage),
        0.0,
    )

    overkill_damage = max(
        float(damage) - opponent_hp,
        0.0,
    )

    effect_text = ""

    if isinstance(move, dict):
        raw_effect = move.get(
            "effect"
        )

        effect_text = (
            ""
            if raw_effect is None
            else str(raw_effect).lower()
        )
    else:
        raw_effect = getattr(
            move,
            "effect",
            None,
        )

        effect_text = (
            ""
            if raw_effect is None
            else str(raw_effect).lower()
        )

    is_healing_move = float(
        "heal" in move_name.lower()
        or "heal" in effect_text
    )

    is_retreat_move = float(
        "retreat" in move_name.lower()
        or "retreat" in effect_text
    )

    is_status_move = float(
        any(
            keyword in effect_text
            for keyword in [
                "poison",
                "burn",
                "asleep",
                "sleep",
                "paralyze",
                "paralyzed",
                "confuse",
                "confused",
                "status",
            ]
        )
    )

    is_pass_move = float(
        move_name.lower() == "pass"
    )

    heuristic_score = (
        float(damage)
        + 100.0 * knockout_possible
        + 0.25 * damage_per_energy
        + 5.0 * is_healing_move
        + 5.0 * is_retreat_move
        + 3.0 * is_status_move
        - 2.0 * float(energy_cost)
        - 10.0 * is_pass_move
    )

    return {
        "move_index": float(
            move_index
        ),
        "move_name": move_name,
        "damage": float(
            damage
        ),
        "energy_cost": float(
            energy_cost
        ),
        "damage_per_energy": float(
            damage_per_energy
        ),
        "opponent_hp": float(
            opponent_hp
        ),
        "knockout_possible": (
            knockout_possible
        ),
        "estimated_hp_after_attack": float(
            estimated_hp_after_attack
        ),
        "overkill_damage": float(
            overkill_damage
        ),
        "is_healing_move": (
            is_healing_move
        ),
        "is_retreat_move": (
            is_retreat_move
        ),
        "is_status_move": (
            is_status_move
        ),
        "is_pass_move": (
            is_pass_move
        ),
        "heuristic_score": float(
            heuristic_score
        ),
    }


print(
    "Move feature extractor created."
)


# ## Cell 3 — Validate Move Feature Extraction

# In[26]:


# ---------------------------------------------------------
# Validate move feature extraction
# ---------------------------------------------------------

move_feature_records = [
    extract_move_features(
        move=move,
        battle_state=test_state,
        move_index=index,
    )
    for index, move in enumerate(
        real_legal_moves,
        start=1,
    )
]

move_features_df = pd.DataFrame(
    move_feature_records
)

display(
    move_features_df
)

# ---------------------------------------------------------
# Structural validation
# ---------------------------------------------------------

assert len(
    move_features_df
) == len(
    real_legal_moves
), (
    "The move feature table must contain "
    "one row for each legal move."
)

assert not move_features_df.empty, (
    "The move feature table is empty."
)

required_move_columns = [
    "move_index",
    "move_name",
    "damage",
    "energy_cost",
    "damage_per_energy",
    "opponent_hp",
    "knockout_possible",
    "estimated_hp_after_attack",
    "overkill_damage",
    "is_healing_move",
    "is_retreat_move",
    "is_status_move",
    "is_pass_move",
    "heuristic_score",
]

missing_move_columns = [
    column
    for column in required_move_columns
    if column not in move_features_df.columns
]

assert not missing_move_columns, (
    f"Missing move feature columns: "
    f"{missing_move_columns}"
)

assert move_features_df[
    "move_name"
].notna().all(), (
    "One or more move names is missing."
)

# ---------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------

numeric_move_columns = [
    column
    for column in required_move_columns
    if column != "move_name"
]

assert np.isfinite(
    move_features_df[
        numeric_move_columns
    ].to_numpy(
        dtype=float
    )
).all(), (
    "One or more move features is not finite."
)

assert (
    move_features_df[
        "damage"
    ]
    >= 0
).all(), (
    "Move damage must not be negative."
)

assert (
    move_features_df[
        "energy_cost"
    ]
    >= 0
).all(), (
    "Move energy cost must not be negative."
)

assert (
    move_features_df[
        "damage_per_energy"
    ]
    >= 0
).all(), (
    "Damage efficiency must not be negative."
)

assert (
    move_features_df[
        "estimated_hp_after_attack"
    ]
    >= 0
).all(), (
    "Estimated remaining HP must not be negative."
)

binary_move_columns = [
    "knockout_possible",
    "is_healing_move",
    "is_retreat_move",
    "is_status_move",
    "is_pass_move",
]

for column in binary_move_columns:
    assert set(
        move_features_df[
            column
        ].unique()
    ).issubset(
        {
            0.0,
            1.0,
        }
    ), (
        f"{column} must contain only 0 or 1."
    )

print()
print(
    f"Legal moves represented : "
    f"{len(move_features_df)}"
)
print(
    f"Features per move        : "
    f"{len(move_features_df.columns) - 1}"
)
print(
    "✅ MOVE FEATURE EXTRACTION PASSED"
)


# # Section 9 — Complete Observation Vector
# 
# This section combines player, opponent, battle, and move features into one stable numerical observation.
# 
# Each legal move receives its own observation row so future agents and machine-learning models can compare candidate actions in the same battle state.

# ### Build one complete observation for one legal move

# In[27]:


# ---------------------------------------------------------
# Build one complete observation for one legal move
# ---------------------------------------------------------

def build_observation_record(
    battle_state: BattleState,
    move: Any,
    move_index: int,
) -> dict[str, float | str]:
    """
    Combine state and action features into one observation row.
    """
    player = extract_player_features(
        battle_state
    )

    opponent = extract_opponent_features(
        battle_state
    )

    battle = extract_battle_features(
        battle_state
    )

    move_features = extract_move_features(
        move=move,
        battle_state=battle_state,
        move_index=move_index,
    )

    observation = {
        **{
            f"player_{key}": value
            for key, value in player.items()
        },
        **opponent,
        **{
            f"battle_{key}": value
            for key, value in battle.items()
        },
        **move_features,
    }

    return observation


print(
    "Complete observation builder created."
)


# ### Build complete observations for all legal moves

# In[28]:


# ---------------------------------------------------------
# Build complete observations for all legal moves
# ---------------------------------------------------------

observation_records = [
    build_observation_record(
        battle_state=test_state,
        move=move,
        move_index=index,
    )
    for index, move in enumerate(
        real_legal_moves,
        start=1,
    )
]

observations_df = pd.DataFrame(
    observation_records
)

display(
    observations_df
)

print()
print(
    f"Observation rows    : "
    f"{len(observations_df)}"
)
print(
    f"Observation columns : "
    f"{len(observations_df.columns)}"
)


# ### Validate Complete Observation Vector

# In[30]:


# ---------------------------------------------------------
# Validate complete observation vector
# ---------------------------------------------------------

assert len(observations_df) == len(real_legal_moves), (
    "One observation should exist for every legal move."
)

assert not observations_df.empty, (
    "Observation DataFrame is empty."
)

assert observations_df.columns.is_unique, (
    "Duplicate observation columns detected."
)

assert observations_df.notna().all().all(), (
    "Observation contains missing values."
)

numeric_columns = [
    column
    for column in observations_df.columns
    if column != "move_name"
]

assert np.isfinite(
    observations_df[
        numeric_columns
    ].to_numpy(dtype=float)
).all(), (
    "Observation contains non-finite values."
)

assert (
    observations_df["move_index"]
    .is_unique
), (
    "Move indices should be unique."
)

print()
print("OBSERVATION SUMMARY")
print("=" * 60)
print(f"Rows              : {len(observations_df)}")
print(f"Columns           : {len(observations_df.columns)}")
print(f"Numeric Features  : {len(numeric_columns)}")
print(f"Moves             : {len(real_legal_moves)}")
print()
print("✅ COMPLETE OBSERVATION VALIDATION PASSED")


# ### Create Stable Feature Ordering

# In[31]:


# ---------------------------------------------------------
# Stable observation feature ordering
# ---------------------------------------------------------

OBSERVATION_COLUMNS = list(
    observations_df.columns
)

schema_df = pd.DataFrame(
    {
        "Position": range(
            len(OBSERVATION_COLUMNS)
        ),
        "Feature": OBSERVATION_COLUMNS,
        "Type": [
            str(
                observations_df[col].dtype
            )
            for col in OBSERVATION_COLUMNS
        ],
    }
)

display(schema_df)

print()
print(
    f"Stable feature schema created with "
    f"{len(OBSERVATION_COLUMNS)} columns."
)

print(
    "✅ FEATURE SCHEMA CREATED"
)


# ### Convert to ML Matrix

# In[32]:


# ---------------------------------------------------------
# Build ML observation matrix
# ---------------------------------------------------------

MODEL_COLUMNS = [
    column
    for column in OBSERVATION_COLUMNS
    if column != "move_name"
]

observation_matrix = observations_df[
    MODEL_COLUMNS
].to_numpy(
    dtype=np.float32
)

assert observation_matrix.shape == (
    len(real_legal_moves),
    len(MODEL_COLUMNS),
)

assert np.isfinite(
    observation_matrix
).all()

print()
print("Observation Matrix")
print("=" * 60)
print(
    "Shape :",
    observation_matrix.shape,
)
print(
    "Type  :",
    observation_matrix.dtype,
)
print()
print("✅ OBSERVATION MATRIX CREATED")


# ### Notebook Summary

# In[33]:


# ---------------------------------------------------------
# Notebook 33 Final Summary
# ---------------------------------------------------------

print("=" * 70)
print("NOTEBOOK 33 SUMMARY")
print("=" * 70)

print()
print("Player Features        :", len(player_features))
print("Opponent Features      :", len(opponent_features))
print("Battle Features        :", len(battle_features))
print("Move Features          :", len(move_features_df.columns) - 1)

print()
print("Legal Moves            :", len(real_legal_moves))
print("Observation Columns    :", len(observations_df.columns))
print("ML Features            :", observation_matrix.shape[1])

print()
print("Observation Matrix Shape :", observation_matrix.shape)
print("Observation Matrix Type  :", observation_matrix.dtype)

print()
print("Feature Schema Columns :", len(OBSERVATION_COLUMNS))

print()
print("✅ NOTEBOOK 33 COMPLETED SUCCESSFULLY")


# ### Save the Feature Schema

# In[34]:


# ---------------------------------------------------------
# Save feature schema
# ---------------------------------------------------------

from pathlib import Path

report_dir = Path("reports/notebook33")
report_dir.mkdir(parents=True, exist_ok=True)

schema_df.to_csv(
    report_dir / "feature_schema.csv",
    index=False,
)

observations_df.to_csv(
    report_dir / "sample_observations.csv",
    index=False,
)

move_features_df.to_csv(
    report_dir / "move_features.csv",
    index=False,
)

print("Reports saved to:")
print(report_dir.resolve())

print()
print("✅ REPORTS EXPORTED")


# ### Final Validation

# In[35]:


# ---------------------------------------------------------
# Final notebook validation
# ---------------------------------------------------------

assert observation_matrix.shape[0] == len(real_legal_moves)
assert observation_matrix.shape[1] == len(MODEL_COLUMNS)

assert len(schema_df) == len(OBSERVATION_COLUMNS)

assert observations_df.notna().all().all()

assert np.isfinite(observation_matrix).all()

print("=" * 70)
print("NOTEBOOK 33 FINAL VALIDATION")
print("=" * 70)

print()
print("Observation Columns :", len(OBSERVATION_COLUMNS))
print("ML Columns          :", len(MODEL_COLUMNS))
print("Legal Moves         :", len(real_legal_moves))
print("Observation Rows    :", len(observations_df))

print()
print("Reports")
print("-------")
print("feature_schema.csv")
print("sample_observations.csv")
print("move_features.csv")

print()
print("🎉 NOTEBOOK 33 PASSED")


# In[ ]:




