#!/usr/bin/env python
# coding: utf-8

# # Cell 1 — Notebook heading
# 

# # Notebook 19 — Battle Feature Extraction and Action Scoring
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook converts the stable `BattleSnapshot` representation from
# Notebook 18 into numerical features that can support heuristic decision-making,
# search, behavior cloning, and self-play.
# 
# ## Project context
# 
# Notebook 17 created the official card repository.
# 
# Notebook 18 created the Kaggle observation adapter:
# 
# ```text
# obs_dict
#     ↓
# cg.api.Observation
#     ↓
# BattleSnapshot
# ```
# 
# Notebook 19 adds:
# 
# ```text
# BattleSnapshot
#     ↓
# State features
#     ↓
# Legal-action features
#     ↓
# Action scores
#     ↓
# Selected option index
# ```
# 
# ## Objectives
# 
# 1. Import the production card database.
# 2. load the exported Notebook 18 adapter safely.
# 3. Define state and action feature models.
# 4. Extract player and opponent board features.
# 5. Preserve Card IDs and semantic action labels.
# 6. Calculate prize, HP, Energy, bench, and tempo features.
# 7. Create a deterministic baseline action scorer.
# 8. Rank only official legal options.
# 9. Return the correct Kaggle option indices.
# 10. Export reusable modules into `src/decision_features/`.
# 
# ## Target production structure
# 
# ```text
# src/decision_features/
# ├── __init__.py
# ├── models.py
# ├── state_features.py
# ├── action_features.py
# ├── scoring.py
# ├── policy.py
# └── validation.py
# ```

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import importlib.util
import json
import math
import sys

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

print("Python:", sys.version)
print("Current directory:", Path.cwd())


# # Cell 3 — Locate project paths

# In[2]:


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the PTCG AI Battle Challenge project root.
    """

    current = (start or Path.cwd()).resolve()

    markers = [
        "src",
        "notebooks",
        "data",
        "scripts",
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

DECISION_FEATURE_DIR = (
    SRC_DIR
    / "decision_features"
)

NOTEBOOK19_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook19"
)

for directory in [
    DECISION_FEATURE_DIR,
    NOTEBOOK19_REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Notebook 18 export:", NOTEBOOK18_EXPORT)
print("Decision-feature package:", DECISION_FEATURE_DIR)
print("Notebook 19 reports:", NOTEBOOK19_REPORT_DIR)


# # Cell 4 — Verify Notebook 17 card database

# In[3]:


from src.card_database import (
    CardRecord,
    CardRepository,
    get_card,
    load_repository,
    search_cards,
)

repository = load_repository()

print("Repository size:", len(repository))

mega_lucario = repository.get(678)

print(
    "Test card:",
    mega_lucario.card_id,
    mega_lucario.name,
)

print(
    "Move count:",
    len(mega_lucario.moves),
)

assert len(repository) == 1267
assert mega_lucario.name == "Mega Lucario ex"

print("\nCard database integration passed.")


# # Cell 5 — Load Notebook 18 adapter definitions
# 
# ### For now, Notebook 18 exists as an exported script. This cell loads its classes and adapter functions without copying them.

# In[5]:


import sys
import types


if not NOTEBOOK18_EXPORT.is_file():
    raise FileNotFoundError(
        f"Notebook 18 export not found:\n"
        f"{NOTEBOOK18_EXPORT}"
    )

module_name = "notebook18_adapter"

source = NOTEBOOK18_EXPORT.read_text(
    encoding="utf-8-sig"
)

source_lines = source.splitlines()

cleaned_lines = [
    line
    for line in source_lines
    if line.strip()
    != "from __future__ import annotations"
]

cleaned_source = (
    "from __future__ import annotations\n"
    + "\n".join(cleaned_lines)
)

notebook18_adapter = types.ModuleType(
    module_name
)

notebook18_adapter.__file__ = str(
    NOTEBOOK18_EXPORT
)

notebook18_adapter.__package__ = ""

# Dataclasses need the module registered while classes are created.
sys.modules[module_name] = notebook18_adapter

compiled_code = compile(
    cleaned_source,
    str(NOTEBOOK18_EXPORT),
    "exec",
)

exec(
    compiled_code,
    notebook18_adapter.__dict__,
)

print("Notebook 18 export loaded successfully.")
print(
    "Repeated future imports removed:",
    len(source_lines) - len(cleaned_lines),
)


# # Cell 6 — Retrieve the required Notebook 18 objects

# In[6]:


REQUIRED_ADAPTER_OBJECTS = [
    "CardInstance",
    "PokemonInstance",
    "PlayerSnapshot",
    "LegalOption",
    "SelectionSnapshot",
    "BattleSnapshot",
    "adapt_observation",
]

missing_adapter_objects = [
    name
    for name in REQUIRED_ADAPTER_OBJECTS
    if not hasattr(notebook18_adapter, name)
]

if missing_adapter_objects:
    raise AttributeError(
        "Notebook 18 export is missing:\n"
        + "\n".join(missing_adapter_objects)
    )

CardInstance = notebook18_adapter.CardInstance
PokemonInstance = notebook18_adapter.PokemonInstance
PlayerSnapshot = notebook18_adapter.PlayerSnapshot
LegalOption = notebook18_adapter.LegalOption
SelectionSnapshot = notebook18_adapter.SelectionSnapshot
BattleSnapshot = notebook18_adapter.BattleSnapshot
adapt_observation = notebook18_adapter.adapt_observation

print("Notebook 18 objects imported:")

for name in REQUIRED_ADAPTER_OBJECTS:
    print("-", name)

print("\nObservation-adapter integration passed.")


# # Cell 7 — Define the state-feature model

# In[7]:


@dataclass(frozen=True)
class PlayerFeatures:
    """
    Numerical description of one player's visible state.
    """

    player_index: int

    active_count: int
    bench_count: int
    available_bench_slots: int

    total_active_hp: int
    total_active_max_hp: int
    total_active_damage: int

    total_bench_hp: int
    total_bench_max_hp: int
    total_bench_damage: int

    total_energy: int
    active_energy: int
    bench_energy: int

    total_tools: int
    evolved_pokemon: int
    damaged_pokemon: int

    deck_count: int
    discard_count: int
    prize_count: int
    hand_count: int

    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass(frozen=True)
class BattleFeatures:
    """
    Numerical and categorical summary of a complete BattleSnapshot.
    """

    turn: int
    turn_action_count: int
    your_index: int
    opponent_index: int
    first_player: int

    supporter_available: bool
    stadium_available: bool
    energy_attachment_available: bool
    retreat_available: bool

    your: PlayerFeatures
    opponent: PlayerFeatures

    prize_advantage: int
    hand_advantage: int
    board_hp_advantage: int
    energy_advantage: int
    bench_advantage: int
    deck_advantage: int

    legal_option_count: int
    legal_option_types: tuple[str, ...]

    terminal_result: int


# # Cell 8 — Safe snapshot helpers

# In[8]:


def pokemon_energy_count(
    pokemon: PokemonInstance | None,
) -> int:
    """
    Count attached Energy while avoiding double counting.

    The official runtime may expose both energy types and energy-card objects.
    """

    if pokemon is None:
        return 0

    return max(
        len(pokemon.energy_types),
        len(pokemon.energy_cards),
    )


def pokemon_tool_count(
    pokemon: PokemonInstance | None,
) -> int:
    if pokemon is None:
        return 0

    return len(pokemon.tools)


def pokemon_damage(
    pokemon: PokemonInstance | None,
) -> int:
    if pokemon is None:
        return 0

    return max(
        0,
        int(pokemon.max_hp) - int(pokemon.hp),
    )


def visible_pokemon(
    player: PlayerSnapshot,
) -> tuple[PokemonInstance, ...]:
    active = tuple(
        pokemon
        for pokemon in player.active
        if pokemon is not None
    )

    return active + tuple(player.bench)


# # Cell 9 — Extract player features

# In[10]:


def extract_player_features(
    player: PlayerSnapshot,
) -> PlayerFeatures:
    """
    Extract numerical features from one PlayerSnapshot.
    """

    active = tuple(
        pokemon
        for pokemon in player.active
        if pokemon is not None
    )

    bench = tuple(player.bench)
    all_pokemon = active + bench

    total_active_hp = sum(
        int(pokemon.hp)
        for pokemon in active
    )

    total_active_max_hp = sum(
        int(pokemon.max_hp)
        for pokemon in active
    )

    total_bench_hp = sum(
        int(pokemon.hp)
        for pokemon in bench
    )

    total_bench_max_hp = sum(
        int(pokemon.max_hp)
        for pokemon in bench
    )

    active_energy = sum(
        pokemon_energy_count(pokemon)
        for pokemon in active
    )

    bench_energy = sum(
        pokemon_energy_count(pokemon)
        for pokemon in bench
    )

    return PlayerFeatures(
        player_index=int(player.player_index),

        active_count=len(active),
        bench_count=len(bench),
        available_bench_slots=max(
            0,
            int(player.bench_max) - len(bench),
        ),

        total_active_hp=total_active_hp,
        total_active_max_hp=total_active_max_hp,
        total_active_damage=sum(
            pokemon_damage(pokemon)
            for pokemon in active
        ),

        total_bench_hp=total_bench_hp,
        total_bench_max_hp=total_bench_max_hp,
        total_bench_damage=sum(
            pokemon_damage(pokemon)
            for pokemon in bench
        ),

        total_energy=active_energy + bench_energy,
        active_energy=active_energy,
        bench_energy=bench_energy,

        total_tools=sum(
            pokemon_tool_count(pokemon)
            for pokemon in all_pokemon
        ),

        evolved_pokemon=sum(
            bool(pokemon.pre_evolution)
            for pokemon in all_pokemon
        ),

        damaged_pokemon=sum(
            pokemon_damage(pokemon) > 0
            for pokemon in all_pokemon
        ),

        deck_count=int(player.deck_count),
        discard_count=len(player.discard),

        prize_count=sum(
            card is not None
            for card in player.prize
        ),

        hand_count=int(player.hand_count),

        poisoned=bool(player.poisoned),
        burned=bool(player.burned),
        asleep=bool(player.asleep),
        paralyzed=bool(player.paralyzed),
        confused=bool(player.confused),
    )


# # Cell 10 — Test player feature extraction synthetically

# In[11]:


sample_player = PlayerSnapshot(
    player_index=0,
    active=(
        PokemonInstance(
            card_id=678,
            serial=1001,
            player_index=0,
            name="Mega Lucario ex",
            hp=300,
            max_hp=340,
            damage=40,
            appeared_this_turn=False,
            energy_types=(
                "FIGHTING",
                "FIGHTING",
            ),
            energy_cards=(),
            tools=(),
            pre_evolution=(),
        ),
    ),
    bench=(),
    bench_max=5,
    deck_count=40,
    discard=(),
    prize=(None, None, None, None, None, None),
    hand_count=7,
    hand=None,
    poisoned=False,
    burned=False,
    asleep=False,
    paralyzed=False,
    confused=False,
)

sample_player_features = extract_player_features(
    sample_player
)

print(sample_player_features)

assert sample_player_features.active_count == 1
assert sample_player_features.total_active_hp == 300
assert sample_player_features.total_active_damage == 40
assert sample_player_features.total_energy == 2
assert sample_player_features.hand_count == 7

print("\nPlayer feature extraction passed.")


# # Cell 11 — Extract full battle features

# In[12]:


def extract_battle_features(
    snapshot: BattleSnapshot,
) -> BattleFeatures:
    """
    Extract numerical features from a complete BattleSnapshot.
    """

    if len(snapshot.players) < 2:
        raise ValueError(
            "BattleSnapshot must contain at least two players."
        )

    your_index = int(snapshot.your_index)

    opponent_candidates = [
        index
        for index in range(len(snapshot.players))
        if index != your_index
    ]

    if not opponent_candidates:
        raise ValueError(
            "Could not determine opponent player index."
        )

    opponent_index = opponent_candidates[0]

    your_features = extract_player_features(
        snapshot.players[your_index]
    )

    opponent_features = extract_player_features(
        snapshot.players[opponent_index]
    )

    selection = snapshot.selection

    legal_option_types = (
        tuple(
            option.option_type
            for option in selection.options
        )
        if selection is not None
        else ()
    )

    return BattleFeatures(
        turn=int(snapshot.turn),
        turn_action_count=int(
            snapshot.turn_action_count
        ),
        your_index=your_index,
        opponent_index=opponent_index,
        first_player=int(snapshot.first_player),

        supporter_available=not bool(
            snapshot.supporter_played
        ),
        stadium_available=not bool(
            snapshot.stadium_played
        ),
        energy_attachment_available=not bool(
            snapshot.energy_attached
        ),
        retreat_available=not bool(
            snapshot.retreated
        ),

        your=your_features,
        opponent=opponent_features,

        prize_advantage=(
            opponent_features.prize_count
            - your_features.prize_count
        ),

        hand_advantage=(
            your_features.hand_count
            - opponent_features.hand_count
        ),

        board_hp_advantage=(
            (
                your_features.total_active_hp
                + your_features.total_bench_hp
            )
            - (
                opponent_features.total_active_hp
                + opponent_features.total_bench_hp
            )
        ),

        energy_advantage=(
            your_features.total_energy
            - opponent_features.total_energy
        ),

        bench_advantage=(
            your_features.bench_count
            - opponent_features.bench_count
        ),

        deck_advantage=(
            your_features.deck_count
            - opponent_features.deck_count
        ),

        legal_option_count=len(
            selection.options
        )
        if selection is not None
        else 0,

        legal_option_types=legal_option_types,

        terminal_result=int(
            snapshot.result
        ),
    )


# # Cell 12 — Create a synthetic opponent and battle snapshot

# In[14]:


sample_opponent = PlayerSnapshot(
    player_index=1,
    active=(
        PokemonInstance(
            card_id=23,
            serial=2001,
            player_index=1,
            name="Hippowdon",
            hp=180,
            max_hp=180,
            damage=0,
            appeared_this_turn=False,
            energy_types=("FIGHTING",),
            energy_cards=(),
            tools=(),
            pre_evolution=(),
        ),
    ),
    bench=(),
    bench_max=5,
    deck_count=42,
    discard=(),
    prize=(None, None, None, None, None, None),
    hand_count=5,
    hand=None,
    poisoned=False,
    burned=False,
    asleep=False,
    paralyzed=False,
    confused=False,
)

sample_snapshot = BattleSnapshot(
    turn=3,
    turn_action_count=2,
    your_index=0,
    first_player=0,

    supporter_played=False,
    stadium_played=False,
    energy_attached=False,
    retreated=False,

    result=0,

    stadium=(),
    looking=None,

    players=(
        sample_player,
        sample_opponent,
    ),

    selection=None,

    log_count=12,
    search_begin_input=None,
)

sample_battle_features = extract_battle_features(
    sample_snapshot
)

print(sample_battle_features)


# # Cell 13 — Validate battle feature extraction 

# In[15]:


assert sample_battle_features.your_index == 0
assert sample_battle_features.opponent_index == 1
assert sample_battle_features.hand_advantage == 2
assert sample_battle_features.energy_advantage == 1
assert sample_battle_features.board_hp_advantage == 120
assert sample_battle_features.legal_option_count == 0

print("Battle feature extraction passed.")


# # Cell 14 — Define action feature model

# In[17]:


@dataclass(frozen=True)
class ActionFeatures:
    """
    Numerical and semantic features for one legal option.
    """

    option_index: int
    option_type: str
    semantic_label: str

    card_id: int | None
    card_name: str | None
    attack_id: int | None

    is_play: bool
    is_attach: bool
    is_evolve: bool
    is_ability: bool
    is_retreat: bool
    is_attack: bool
    is_end: bool
    is_discard: bool

    card_is_pokemon: bool
    card_is_trainer: bool
    card_is_energy: bool

    card_battle_score: float
    card_attack_score: float
    card_tank_score: float
    card_mobility_score: float

    base_priority: float


# # Cell 15 — Define option-type priorities

# In[18]:


OPTION_BASE_PRIORITY = {
    "ATTACK": 100.0,
    "EVOLVE": 80.0,
    "ABILITY": 70.0,
    "ATTACH": 60.0,
    "PLAY": 50.0,
    "RETREAT": 40.0,
    "DISCARD": 20.0,
    "END": 0.0,
}

print("Configured option priorities:")

for option_type, priority in OPTION_BASE_PRIORITY.items():
    print(f"- {option_type}: {priority}")


# ###  Above are baseline priorities. Later notebooks can replace them with learned scores.

# # Cell 16 — Extract action features

# In[19]:


def safe_float(
    value: float | int | None,
) -> float:
    if value is None:
        return 0.0

    return float(value)


def extract_action_features(
    option: LegalOption,
    *,
    repository: CardRepository,
) -> ActionFeatures:
    """
    Convert one legal option into numerical and semantic features.
    """

    option_type = str(option.option_type).upper()

    record = (
        repository.get_optional(option.card_id)
        if option.card_id is not None
        else None
    )

    return ActionFeatures(
        option_index=int(option.option_index),
        option_type=option_type,
        semantic_label=(
            option.semantic_label
            or option_type
        ),

        card_id=option.card_id,
        card_name=option.card_name,
        attack_id=option.attack_id,

        is_play=option_type == "PLAY",
        is_attach=option_type == "ATTACH",
        is_evolve=option_type == "EVOLVE",
        is_ability=option_type == "ABILITY",
        is_retreat=option_type == "RETREAT",
        is_attack=option_type == "ATTACK",
        is_end=option_type == "END",
        is_discard=option_type == "DISCARD",

        card_is_pokemon=bool(
            record.is_pokemon
            if record is not None
            else False
        ),

        card_is_trainer=bool(
            record.is_trainer
            if record is not None
            else False
        ),

        card_is_energy=bool(
            record.is_energy
            if record is not None
            else False
        ),

        card_battle_score=safe_float(
            record.overall_battle_score
            if record is not None
            else None
        ),

        card_attack_score=safe_float(
            record.attack_power_score
            if record is not None
            else None
        ),

        card_tank_score=safe_float(
            record.tank_score
            if record is not None
            else None
        ),

        card_mobility_score=safe_float(
            record.mobility_score
            if record is not None
            else None
        ),

        base_priority=OPTION_BASE_PRIORITY.get(
            option_type,
            10.0,
        ),
    )


# # Cell 17 — Define scored-action model

# In[20]:


@dataclass(frozen=True)
class ScoredAction:
    """
    One legal action together with its deterministic score.
    """

    option_index: int
    option_type: str
    semantic_label: str
    score: float
    reasons: tuple[str, ...]


# # Cell 18 — Score one legal action

# In[22]:


def score_action(
    action: ActionFeatures,
    battle: BattleFeatures,
) -> ScoredAction:
    """
    Produce a deterministic baseline score for one legal action.
    """

    score = float(action.base_priority)
    reasons: list[str] = [
        f"base={action.base_priority:.1f}"
    ]

    if action.is_attack:
        attack_bonus = (
            20.0
            + max(
                0.0,
                action.card_attack_score,
            )
        )

        score += attack_bonus
        reasons.append(
            f"attack_bonus={attack_bonus:.1f}"
        )

        if battle.your.active_energy > 0:
            score += 5.0
            reasons.append("active_energy=5.0")

    if action.is_evolve:
        score += 15.0
        reasons.append("evolution_tempo=15.0")

    if action.is_attach:
        if battle.energy_attachment_available:
            score += 12.0
            reasons.append("attachment_available=12.0")
        else:
            score -= 50.0
            reasons.append("attachment_unavailable=-50.0")

    if action.is_ability:
        score += 10.0
        reasons.append("ability_value=10.0")

    if action.is_play and action.card_is_trainer:
        score += 8.0
        reasons.append("trainer_play=8.0")

    if action.is_play and action.card_is_pokemon:
        if battle.your.available_bench_slots > 0:
            score += 7.0
            reasons.append("bench_development=7.0")
        else:
            score -= 20.0
            reasons.append("bench_full=-20.0")

    if action.is_retreat:
        status_pressure = any(
            [
                battle.your.poisoned,
                battle.your.burned,
                battle.your.asleep,
                battle.your.paralyzed,
                battle.your.confused,
            ]
        )

        if status_pressure:
            score += 35.0
            reasons.append("status_escape=35.0")
        else:
            score -= 5.0
            reasons.append("retreat_cost=-5.0")

    if action.is_discard:
        score -= 10.0
        reasons.append("discard_penalty=-10.0")

    if action.is_end:
        score -= 25.0
        reasons.append("end_turn_penalty=-25.0")

    score += 0.02 * action.card_battle_score

    if action.card_battle_score:
        reasons.append(
            "card_battle_score="
            f"{0.02 * action.card_battle_score:.2f}"
        )

    return ScoredAction(
        option_index=action.option_index,
        option_type=action.option_type,
        semantic_label=action.semantic_label,
        score=score,
        reasons=tuple(reasons),
    )


# # Cell 19 — Rank legal options

# In[23]:


def rank_legal_actions(
    snapshot: BattleSnapshot,
    *,
    repository: CardRepository,
) -> list[ScoredAction]:
    """
    Score and rank only the legal options exposed by Kaggle.
    """

    if snapshot.selection is None:
        return []

    battle_features = extract_battle_features(
        snapshot
    )

    scored_actions = []

    for option in snapshot.selection.options:
        action_features = extract_action_features(
            option,
            repository=repository,
        )

        scored_actions.append(
            score_action(
                action_features,
                battle_features,
            )
        )

    return sorted(
        scored_actions,
        key=lambda action: (
            -action.score,
            action.option_index,
        ),
    )


# # Cell 20 — Select the best legal option

# In[24]:


def choose_best_option_index(
    snapshot: BattleSnapshot,
    *,
    repository: CardRepository,
) -> int:
    """
    Return the official Kaggle option index with the highest score.
    """

    ranked = rank_legal_actions(
        snapshot,
        repository=repository,
    )

    if not ranked:
        raise ValueError(
            "No legal options are available."
        )

    return int(ranked[0].option_index)


# # Cell 21 — Create synthetic legal actions

# In[26]:


from dataclasses import replace


sample_attack = LegalOption(
    option_index=0,
    option_type="ATTACK",
    semantic_label="Attack with Mega Lucario ex",
    card_id=678,
    card_name="Mega Lucario ex",
    attack_id=1,
)

sample_end = LegalOption(
    option_index=1,
    option_type="END",
    semantic_label="End Turn",
    card_id=None,
    card_name=None,
    attack_id=None,
)

sample_selection = SelectionSnapshot(
    select_type="MAIN",
    context="MAIN",
    min_count=1,
    max_count=1,
    remain_damage_counter=0,
    remain_energy_cost=0,
    options=(
        sample_attack,
        sample_end,
    ),
    deck=None,
    context_card=None,
    effect_card=None,
)

sample_snapshot = replace(
    sample_snapshot,
    selection=sample_selection,
)

print("Synthetic selection created successfully.")
print("Legal options:", len(sample_snapshot.selection.options))


# # Cell 22 — Rank the legal actions

# In[27]:


ranked = rank_legal_actions(
    sample_snapshot,
    repository=repository,
)

for action in ranked:
    print(action)


# # Cell 23 — Validate the ranking

# In[28]:


assert ranked[0].option_type == "ATTACK"
assert ranked[0].score > ranked[1].score

print("Legal action ranking passed.")


# # Cell 24 — Choose the best action

# In[29]:


best = choose_best_option_index(
    sample_snapshot,
    repository=repository,
)

print("Best option index:", best)

assert best == 0

print("Action selection passed.")


# # Cell 25 — Display the ranking table

# In[31]:


ranking_rows = [
    {
        "rank": rank,
        "option_index": action.option_index,
        "option_type": action.option_type,
        "semantic_label": action.semantic_label,
        "score": round(action.score, 3),
        "reasons": " | ".join(action.reasons),
    }
    for rank, action in enumerate(
        ranked,
        start=1,
    )
]

ranking_df = pd.DataFrame(ranking_rows)

display(ranking_df)


# # Cell 26 — Notebook 19 validation summary

# In[32]:


notebook19_checks = {
    "player_feature_extraction": True,
    "battle_feature_extraction": True,
    "action_feature_extraction": True,
    "legal_action_ranking": ranked[0].option_type == "ATTACK",
    "best_option_index": best == 0,
}

for check, passed in notebook19_checks.items():
    print(
        f"{'[OK]' if passed else '[FAIL]'} "
        f"{check}"
    )

assert all(notebook19_checks.values())

print("\nNotebook 19 validation passed.")


# In[ ]:




