from __future__ import annotations

#!/usr/bin/env python
# coding: utf-8

# # Notebook 18 â€” Kaggle Observation Adapter
# 
# ## The PokÃ©mon Company â€” PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook converts the official Kaggle observation structure into stable,
# card-aware Team Jesus battle-state objects.
# 
# ## Project context
# 
# Notebook 17 produced the reusable card database package:
# 
# ```python
# from src.card_database import get_card, load_repository
# ```
# 
# Notebook 18 will use that repository to preserve card identity throughout the
# gameplay pipeline.
# 
# ## Objectives
# 
# 1. Inspect the official `cg.api.Observation` model.
# 2. Identify visible player, opponent, board, and selection fields.
# 3. Create immutable card-zone and player-state models.
# 4. Preserve Card IDs rather than learning from hand positions.
# 5. Convert Kaggle cards and PokÃ©mon into Team Jesus objects.
# 6. Convert legal options into card-aware action candidates.
# 7. Respect hidden information.
# 8. Export reusable code to `src/observation_adapter/`.
# 9. Validate the adapter against official sample structures.
# 
# ## Target production structure
# 
# ```text
# src/observation_adapter/
# â”œâ”€â”€ __init__.py
# â”œâ”€â”€ models.py
# â”œâ”€â”€ card_adapter.py
# â”œâ”€â”€ player_adapter.py
# â”œâ”€â”€ action_adapter.py
# â”œâ”€â”€ observation_adapter.py
# â””â”€â”€ validation.py
# ```

# # Cell 2 â€” Imports and environment

# In[1]:



import inspect
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

print("Python:", sys.version)
print("Current directory:", Path.cwd())


# # Cell 3 â€” Locate the project root

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
        "README.md",
    ]

    for candidate in [current, *current.parents]:
        marker_count = sum(
            (candidate / marker).exists()
            for marker in markers
        )

        if marker_count >= 2:
            return candidate

    if current.name.lower() == "notebooks":
        return current.parent

    return current


PROJECT_ROOT = find_project_root()

SRC_DIR = PROJECT_ROOT / "src"
CARD_DATABASE_DIR = SRC_DIR / "card_database"
OBSERVATION_ADAPTER_DIR = SRC_DIR / "observation_adapter"

REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK18_REPORT_DIR = REPORTS_DIR / "notebook18"

for directory in [
    OBSERVATION_ADAPTER_DIR,
    NOTEBOOK18_REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Card database:", CARD_DATABASE_DIR)
print("Observation adapter:", OBSERVATION_ADAPTER_DIR)
print("Notebook 18 reports:", NOTEBOOK18_REPORT_DIR)


# # Cell 4 â€” Verify the Notebook 17 package

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

print("Test card:", mega_lucario.card_id, mega_lucario.name)
print("Move count:", len(mega_lucario.moves))

assert len(repository) == 1267
assert mega_lucario.name == "Mega Lucario ex"

print("\nNotebook 17 package integration passed.")


# # Cell 5 â€” Locate the official cg runtime
# 
# ### Notebook 16 extracted the official runtime. This cell searches the known locations.

# In[4]:


CG_CANDIDATES = [
    PROJECT_ROOT
    / "submission_work"
    / "official_sample_extracted"
    / "cg",

    PROJECT_ROOT
    / "submission_work"
    / "team_jesus_baseline"
    / "cg",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle_sample_submission"
    / "cg",
]

CG_DIR = next(
    (
        path.resolve()
        for path in CG_CANDIDATES
        if path.is_dir()
    ),
    None,
)

print("Candidate cg directories:")

for path in CG_CANDIDATES:
    print(
        "FOUND  " if path.is_dir() else "MISSING",
        path,
    )

if CG_DIR is None:
    raise FileNotFoundError(
        "The official cg runtime directory was not found."
    )

CG_PARENT = CG_DIR.parent

if str(CG_PARENT) not in sys.path:
    sys.path.insert(0, str(CG_PARENT))

print("\nSelected cg directory:")
print(CG_DIR)


# # Cell 6 â€” Import and inspect the official API

# In[5]:


from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)

print("Official cg API imported successfully.")
print()
print("Observation:", Observation)
print("Card:", Card)
print("Pokemon:", Pokemon)
print("AreaType:", list(AreaType))
print("OptionType:", list(OptionType))


# # Cell 7 â€” Inspect official class fields safely

# In[6]:


OFFICIAL_CLASSES = [
    Observation,
    Card,
    Pokemon,
    SelectContext,
]

for cls in OFFICIAL_CLASSES:
    print("=" * 80)
    print(cls.__name__)
    print("=" * 80)

    annotations = getattr(cls, "__annotations__", {})

    if annotations:
        print("Annotations:")

        for name, annotation in annotations.items():
            print(f"- {name}: {annotation}")
    else:
        print("No class annotations found.")

    print("\nPublic attributes:")

    public_names = [
        name
        for name in dir(cls)
        if not name.startswith("_")
    ]

    for name in public_names[:80]:
        print("-", name)

    print()


# # Cell 8 â€” Inspect all official API classes

# In[7]:


import cg.api as cg_api


def inspect_api_class(class_name: str) -> None:
    """
    Print annotations and public attributes for one cg.api class.
    """

    cls = getattr(cg_api, class_name, None)

    print("=" * 90)
    print(class_name)
    print("=" * 90)

    if cls is None:
        print("Class not found.\n")
        return

    print("Class:", cls)

    annotations = getattr(cls, "__annotations__", {})

    if annotations:
        print("\nAnnotations:")

        for name, annotation in annotations.items():
            print(f"- {name}: {annotation}")
    else:
        print("\nNo annotations found.")

    public_names = [
        name
        for name in dir(cls)
        if not name.startswith("_")
    ]

    if public_names:
        print("\nPublic attributes:")

        for name in public_names[:100]:
            print("-", name)

    print()


API_CLASS_NAMES = [
    "State",
    "PlayerState",
    "SelectData",
    "Option",
    "Log",
    "Observation",
    "Card",
    "Pokemon",
]

for class_name in API_CLASS_NAMES:
    inspect_api_class(class_name)


# # Cell 9 â€” Discover every class defined in cg.api

# In[8]:


api_classes = []

for name, value in vars(cg_api).items():
    if inspect.isclass(value):
        api_classes.append(
            {
                "name": name,
                "module": getattr(value, "__module__", None),
                "annotations": list(
                    getattr(
                        value,
                        "__annotations__",
                        {},
                    ).keys()
                ),
            }
        )

api_classes_df = pd.DataFrame(api_classes)

display(
    api_classes_df.sort_values("name")
)


# # Cell 10 â€” Inspect the most important discovered classes

# In[9]:


IMPORTANT_NAME_PARTS = [
    "state",
    "player",
    "select",
    "option",
    "log",
    "card",
    "pokemon",
]

important_class_names = sorted(
    row["name"]
    for row in api_classes
    if any(
        part in row["name"].casefold()
        for part in IMPORTANT_NAME_PARTS
    )
)

print("Important official classes:\n")

for name in important_class_names:
    print("-", name)

print()

for name in important_class_names:
    inspect_api_class(name)


# # Cell 11 â€” Inspect enum values in a structured format

# In[10]:


def enum_report(enum_type: type) -> pd.DataFrame:
    """
    Convert an enum into a readable dataframe.
    """

    return pd.DataFrame(
        [
            {
                "name": member.name,
                "value": member.value,
            }
            for member in enum_type
        ]
    )


print("AreaType")
display(enum_report(AreaType))

print("OptionType")
display(enum_report(OptionType))

print("SelectContext")
display(enum_report(SelectContext))


# # Cell 12 â€” Load official card metadata from cg
# 
# ### This verifies that the official runtime Card IDs align with Notebook 17.

# In[11]:


official_cards = all_card_data()

print("Official cg card objects:", len(official_cards))
print("Notebook 17 repository:", len(repository))

official_card_ids = {
    int(card.cardId)
    for card in official_cards
}

repository_card_ids = {
    card.card_id
    for card in repository.all_cards()
}

missing_from_repository = sorted(
    official_card_ids - repository_card_ids
)

extra_in_repository = sorted(
    repository_card_ids - official_card_ids
)

print()
print("Official unique Card IDs:", len(official_card_ids))
print("Repository unique Card IDs:", len(repository_card_ids))
print("Missing from repository:", len(missing_from_repository))
print("Extra in repository:", len(extra_in_repository))

if missing_from_repository:
    print(
        "\nFirst missing IDs:",
        missing_from_repository[:20],
    )

if extra_in_repository:
    print(
        "\nFirst extra IDs:",
        extra_in_repository[:20],
    )


# # Cell 13 â€” Inspect official card-object fields

# In[12]:


sample_official_cards = official_cards[:5]

for index, card in enumerate(
    sample_official_cards,
    start=1,
):
    print("=" * 80)
    print(f"Official card sample {index}")
    print("=" * 80)

    print("Type:", type(card))

    card_attributes = {
        name: getattr(card, name)
        for name in dir(card)
        if not name.startswith("_")
        and not callable(getattr(card, name))
    }

    for name, value in card_attributes.items():
        print(f"{name}: {value}")

    print()


# # Cell 14 â€” Notebook findings

# ## Official API Findings
# 
# ### Observation
# 
# The official observation contains:
# 
# - `current`: current visible game state
# - `select`: current legal selection request
# - `logs`: game-event history
# - `search_begin_input`: optional search-related input
# 
# ### Card instances
# 
# A game-state `Card` preserves:
# 
# - Card ID
# - unique serial number
# - owning player index
# 
# This allows the adapter to distinguish multiple copies of the same Card ID.
# 
# ### PokÃ©mon instances
# 
# A game-state `Pokemon` preserves:
# 
# - Card ID
# - unique serial number
# - current HP
# - maximum HP
# - whether it appeared this turn
# - attached Energy types
# - attached Energy cards
# - attached Tools
# - pre-evolution cards
# 
# ### Legal actions
# 
# The simulator exposes legal action types through `OptionType`, including:
# 
# - play
# - attach
# - evolve
# - ability
# - discard
# - retreat
# - attack
# - end turn
# 
# The Team Jesus adapter should preserve both:
# 
# 1. The legal option index required by Kaggle.
# 2. The card-aware semantic meaning required by the AI.

# # Cell 14A â€” Build official CardData lookup

# In[14]:


official_card_data_by_id = {
    int(card.cardId): card
    for card in official_cards
}

print(
    "Official CardData lookup size:",
    len(official_card_data_by_id),
)

sample = official_card_data_by_id[678]

print("\nCard 678")
print("Name:", sample.name)
print("Card type:", sample.cardType)
print("HP:", sample.hp)
print("Attacks:", len(sample.attacks))
print("Skills:", len(sample.skills))
print("Mega ex:", sample.megaEx)


# # Cell 14B â€” Compare official and processed names

# In[15]:


name_mismatches = []

for card_id, official_card in official_card_data_by_id.items():
    repository_card = repository.get_optional(card_id)

    if repository_card is None:
        continue

    if official_card.name.strip() != repository_card.name.strip():
        name_mismatches.append(
            {
                "card_id": card_id,
                "official_name": official_card.name,
                "repository_name": repository_card.name,
            }
        )

print("Name mismatches:", len(name_mismatches))

if name_mismatches:
    display(
        pd.DataFrame(name_mismatches).head(20)
    )


# # Cell 15 â€” Define Team Jesus observation models

# In[16]:



from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CardInstance:
    """
    One visible card instance in the current game.

    card_id identifies the card definition.
    serial identifies this physical copy in the match.
    """

    card_id: int
    serial: int
    player_index: int

    name: str | None = None
    category: str | None = None
    stage_or_type: str | None = None

    official_card_type: str | None = None
    energy_type: str | None = None

    is_ex: bool = False
    is_mega_ex: bool = False
    is_tera: bool = False
    is_ace_spec: bool = False


@dataclass(frozen=True)
class PokemonInstance:
    """
    One PokÃ©mon currently in play.
    """

    card_id: int
    serial: int
    player_index: int

    name: str | None
    hp: int
    max_hp: int
    damage: int
    appeared_this_turn: bool

    energy_types: tuple[str, ...] = ()
    energy_cards: tuple[CardInstance, ...] = ()
    tools: tuple[CardInstance, ...] = ()
    pre_evolution: tuple[CardInstance, ...] = ()


@dataclass(frozen=True)
class PlayerSnapshot:
    """
    Visible information for one player.
    """

    player_index: int

    active: tuple[PokemonInstance | None, ...]
    bench: tuple[PokemonInstance, ...]
    bench_max: int

    deck_count: int
    discard: tuple[CardInstance, ...]
    prize: tuple[CardInstance | None, ...]

    hand_count: int
    hand: tuple[CardInstance, ...] | None

    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass(frozen=True)
class LegalOption:
    """
    One legal option exposed by Kaggle.

    option_index is the integer that must be returned to the simulator.
    """

    option_index: int
    option_type: str

    number: int | None = None
    area: str | None = None
    index: int | None = None
    player_index: int | None = None
    tool_index: int | None = None
    energy_index: int | None = None
    count: int | None = None

    in_play_area: str | None = None
    in_play_index: int | None = None

    attack_id: int | None = None
    card_id: int | None = None
    serial: int | None = None

    special_condition: str | None = None

    card_name: str | None = None
    semantic_label: str | None = None


@dataclass(frozen=True)
class SelectionSnapshot:
    """
    Current Kaggle selection request.
    """

    select_type: str
    context: str

    min_count: int
    max_count: int

    remain_damage_counter: int
    remain_energy_cost: int

    options: tuple[LegalOption, ...]

    deck: tuple[CardInstance, ...] | None = None
    context_card: CardInstance | None = None
    effect_card: CardInstance | None = None


@dataclass(frozen=True)
class BattleSnapshot:
    """
    Stable Team Jesus representation of one observation.
    """

    turn: int
    turn_action_count: int
    your_index: int
    first_player: int

    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool

    result: int

    stadium: tuple[CardInstance, ...]
    looking: tuple[CardInstance | None, ...] | None

    players: tuple[PlayerSnapshot, ...]

    selection: SelectionSnapshot | None

    log_count: int
    search_begin_input: str | None


# In[ ]:





# In[17]:


def enum_name(value: Any) -> str | None:
    """
    Convert enum-like values into stable text.
    """

    if value is None:
        return None

    name = getattr(value, "name", None)

    if name is not None:
        return str(name)

    return str(value)


def card_metadata_name(
    card_id: int,
    repository: CardRepository,
) -> str | None:
    """
    Return the official card name when available.
    """

    record = repository.get_optional(card_id)

    if record is None:
        return None

    return record.name


def card_metadata_category(
    card_id: int,
    repository: CardRepository,
) -> str | None:
    record = repository.get_optional(card_id)

    if record is None:
        return None

    return record.category or record.stage_or_type


# # Cell 16 â€” Add safe enum conversion helpers

# In[19]:


def enum_name(value: Any) -> str | None:
    """
    Convert enum-like values into stable text.
    """

    if value is None:
        return None

    name = getattr(value, "name", None)

    if name is not None:
        return str(name)

    return str(value)


def card_metadata_name(
    card_id: int,
    repository: CardRepository,
) -> str | None:
    """
    Return the official card name when available.
    """

    record = repository.get_optional(card_id)

    if record is None:
        return None

    return record.name


def card_metadata_category(
    card_id: int,
    repository: CardRepository,
) -> str | None:
    record = repository.get_optional(card_id)

    if record is None:
        return None

    return record.category or record.stage_or_type


# # Cell 17 â€” Convert official cards into CardInstance

# In[20]:


def adapt_card(
    card: Card | None,
    *,
    repository: CardRepository,
) -> CardInstance | None:
    """
    Convert one official cg.api.Card into a stable CardInstance.
    """

    if card is None:
        return None

    card_id = int(card.id)

    record = repository.get_optional(card_id)

    return CardInstance(
        card_id=card_id,
        serial=int(card.serial),
        player_index=int(card.playerIndex),
        name=record.name if record else None,
        category=(
            record.card_category
            if record is not None
            and hasattr(record, "card_category")
            else (
                record.category
                if record is not None
                else None
            )
        ),
        stage_or_type=(
            record.stage_or_type
            if record is not None
            else None
        ),
    )


# # Cell 18 â€” Convert official PokÃ©mon into PokemonInstance

# In[21]:


def adapt_pokemon(
    pokemon: Pokemon | None,
    *,
    repository: CardRepository,
    player_index: int,
) -> PokemonInstance | None:
    """
    Convert one official cg.api.Pokemon into a stable PokemonInstance.
    """

    if pokemon is None:
        return None

    card_id = int(pokemon.id)
    record = repository.get_optional(card_id)

    hp = int(pokemon.hp)
    max_hp = int(pokemon.maxHp)

    energy_types = tuple(
        enum_name(energy) or "UNKNOWN"
        for energy in pokemon.energies
    )

    energy_cards = tuple(
        card
        for card in (
            adapt_card(
                energy_card,
                repository=repository,
            )
            for energy_card in pokemon.energyCards
        )
        if card is not None
    )

    tools = tuple(
        card
        for card in (
            adapt_card(
                tool_card,
                repository=repository,
            )
            for tool_card in pokemon.tools
        )
        if card is not None
    )

    pre_evolution = tuple(
        card
        for card in (
            adapt_card(
                previous_card,
                repository=repository,
            )
            for previous_card in pokemon.preEvolution
        )
        if card is not None
    )

    return PokemonInstance(
        card_id=card_id,
        serial=int(pokemon.serial),
        player_index=int(player_index),
        name=record.name if record else None,
        hp=hp,
        max_hp=max_hp,
        damage=max(0, max_hp - hp),
        appeared_this_turn=bool(
            pokemon.appearThisTurn
        ),
        energy_types=energy_types,
        energy_cards=energy_cards,
        tools=tools,
        pre_evolution=pre_evolution,
    )


# # Cell 19 â€” Validate the models with synthetic objects

# In[22]:


sample_card_instance = CardInstance(
    card_id=678,
    serial=1001,
    player_index=0,
    name="Mega Lucario ex",
    category="PokÃ©mon",
    stage_or_type="Stage 1",
)

sample_pokemon_instance = PokemonInstance(
    card_id=678,
    serial=1001,
    player_index=0,
    name="Mega Lucario ex",
    hp=250,
    max_hp=280,
    damage=30,
    appeared_this_turn=False,
    energy_types=("FIGHTING", "FIGHTING"),
    energy_cards=(),
    tools=(),
    pre_evolution=(),
)

print(sample_card_instance)
print()
print(sample_pokemon_instance)

assert sample_card_instance.card_id == 678
assert sample_pokemon_instance.damage == 30
assert sample_pokemon_instance.name == "Mega Lucario ex"

print("\nObservation model validation passed.")


# # Next: Cell 20 â€” Adapt a player state

# In[23]:


def adapt_player_state(
    player_state: cg_api.PlayerState,
    *,
    player_index: int,
    repository: CardRepository,
    official_lookup: dict[int, Any],
) -> PlayerSnapshot:
    """
    Convert one official PlayerState into a Team Jesus PlayerSnapshot.
    """

    active = tuple(
        adapt_pokemon(
            pokemon,
            repository=repository,
            player_index=player_index,
        )
        if pokemon is not None
        else None
        for pokemon in player_state.active
    )

    bench = tuple(
        pokemon
        for pokemon in (
            adapt_pokemon(
                item,
                repository=repository,
                player_index=player_index,
            )
            for item in player_state.bench
        )
        if pokemon is not None
    )

    discard = tuple(
        card
        for card in (
            adapt_card(
                item,
                repository=repository,
                official_lookup=official_lookup,
            )
            for item in player_state.discard
        )
        if card is not None
    )

    prize = tuple(
        adapt_card(
            item,
            repository=repository,
            official_lookup=official_lookup,
        )
        if item is not None
        else None
        for item in player_state.prize
    )

    hand = (
        tuple(
            card
            for card in (
                adapt_card(
                    item,
                    repository=repository,
                    official_lookup=official_lookup,
                )
                for item in player_state.hand
            )
            if card is not None
        )
        if player_state.hand is not None
        else None
    )

    return PlayerSnapshot(
        player_index=int(player_index),
        active=active,
        bench=bench,
        bench_max=int(player_state.benchMax),
        deck_count=int(player_state.deckCount),
        discard=discard,
        prize=prize,
        hand_count=int(player_state.handCount),
        hand=hand,
        poisoned=bool(player_state.poisoned),
        burned=bool(player_state.burned),
        asleep=bool(player_state.asleep),
        paralyzed=bool(player_state.paralyzed),
        confused=bool(player_state.confused),
    )


# # Cell 21 â€” Adapt legal options

# In[24]:


def build_semantic_label(
    option: cg_api.Option,
    *,
    card_name: str | None,
) -> str:
    """
    Build a human-readable semantic label for one legal option.
    """

    option_type = enum_name(option.type) or "UNKNOWN"

    parts = [option_type]

    if card_name:
        parts.append(card_name)

    if option.attackId is not None:
        parts.append(f"attack={option.attackId}")

    if option.area is not None:
        parts.append(f"area={enum_name(option.area)}")

    if option.index is not None:
        parts.append(f"index={option.index}")

    if option.playerIndex is not None:
        parts.append(f"player={option.playerIndex}")

    return " | ".join(parts)


def adapt_option(
    option: cg_api.Option,
    *,
    option_index: int,
    repository: CardRepository,
) -> LegalOption:
    """
    Convert one official legal option into a Team Jesus LegalOption.
    """

    card_id = (
        int(option.cardId)
        if option.cardId is not None
        else None
    )

    card_name = None

    if card_id is not None:
        record = repository.get_optional(card_id)

        if record is not None:
            card_name = record.name

    semantic_label = build_semantic_label(
        option,
        card_name=card_name,
    )

    return LegalOption(
        option_index=int(option_index),
        option_type=enum_name(option.type) or "UNKNOWN",
        number=option.number,
        area=enum_name(option.area),
        index=option.index,
        player_index=option.playerIndex,
        tool_index=option.toolIndex,
        energy_index=option.energyIndex,
        count=option.count,
        in_play_area=enum_name(option.inPlayArea),
        in_play_index=option.inPlayIndex,
        attack_id=option.attackId,
        card_id=card_id,
        serial=option.serial,
        special_condition=enum_name(
            option.specialConditionType
        ),
        card_name=card_name,
        semantic_label=semantic_label,
    )


# # Cell 22 â€” Adapt the selection request

# In[26]:


def adapt_selection(
    selection: cg_api.SelectData | None,
    *,
    repository: CardRepository,
    official_lookup: dict[int, Any],
) -> SelectionSnapshot | None:
    """
    Convert the current Kaggle selection request.
    """

    if selection is None:
        return None

    options = tuple(
        adapt_option(
            option,
            option_index=index,
            repository=repository,
        )
        for index, option in enumerate(selection.option)
    )

    deck = (
        tuple(
            card
            for card in (
                adapt_card(
                    item,
                    repository=repository,
                    official_lookup=official_lookup,
                )
                for item in selection.deck
            )
            if card is not None
        )
        if selection.deck is not None
        else None
    )

    return SelectionSnapshot(
        select_type=enum_name(selection.type) or "UNKNOWN",
        context=enum_name(selection.context) or "UNKNOWN",
        min_count=int(selection.minCount),
        max_count=int(selection.maxCount),
        remain_damage_counter=int(
            selection.remainDamageCounter
        ),
        remain_energy_cost=int(
            selection.remainEnergyCost
        ),
        options=options,
        deck=deck,
        context_card=adapt_card(
            selection.contextCard,
            repository=repository,
            official_lookup=official_lookup,
        ),
        effect_card=adapt_card(
            selection.effect,
            repository=repository,
            official_lookup=official_lookup,
        ),
    )


# # Cell 23 â€” Confirm adapter functions are defined

# In[27]:


adapter_functions = [
    adapt_player_state,
    build_semantic_label,
    adapt_option,
    adapt_selection,
]

print("Adapter functions defined:")

for function in adapter_functions:
    print(f"- {function.__name__}")

assert callable(adapt_player_state)
assert callable(adapt_option)
assert callable(adapt_selection)

print("\nCells 20â€“22 loaded successfully.")


# # Cell 24 â€” Adapt the complete observation

# In[28]:


def adapt_observation(
    observation: Observation,
    *,
    repository: CardRepository,
    official_lookup: dict[int, Any],
) -> BattleSnapshot | None:
    """
    Convert one official Observation into a Team Jesus BattleSnapshot.
    """

    state = observation.current

    if state is None:
        return None

    players = tuple(
        adapt_player_state(
            player_state,
            player_index=index,
            repository=repository,
            official_lookup=official_lookup,
        )
        for index, player_state in enumerate(state.players)
    )

    stadium = tuple(
        card
        for card in (
            adapt_card(
                item,
                repository=repository,
                official_lookup=official_lookup,
            )
            for item in state.stadium
        )
        if card is not None
    )

    looking = (
        tuple(
            adapt_card(
                item,
                repository=repository,
                official_lookup=official_lookup,
            )
            if item is not None
            else None
            for item in state.looking
        )
        if state.looking is not None
        else None
    )

    selection = adapt_selection(
        observation.select,
        repository=repository,
        official_lookup=official_lookup,
    )

    return BattleSnapshot(
        turn=int(state.turn),
        turn_action_count=int(state.turnActionCount),
        your_index=int(state.yourIndex),
        first_player=int(state.firstPlayer),
        supporter_played=bool(state.supporterPlayed),
        stadium_played=bool(state.stadiumPlayed),
        energy_attached=bool(state.energyAttached),
        retreated=bool(state.retreated),
        result=int(state.result),
        stadium=stadium,
        looking=looking,
        players=players,
        selection=selection,
        log_count=len(observation.logs),
        search_begin_input=observation.search_begin_input,
    )


# # Cell 25 â€” Validate the full adapter definition

# In[29]:


print("Full observation adapter defined:", callable(adapt_observation))

assert callable(adapt_observation)

print("Cell 24 loaded successfully.")


# # Cell 26 â€” Create an Observation Inspector

# In[30]:


def summarize_observation(
    observation: Observation,
):
    """
    Print a compact summary of one Observation.
    """

    print("=" * 80)
    print("Observation Summary")
    print("=" * 80)

    if observation.current is None:
        print("No current state.")
        return

    state = observation.current

    print("Turn:", state.turn)
    print("Turn Action:", state.turnActionCount)
    print("Your Player:", state.yourIndex)
    print("First Player:", state.firstPlayer)

    print()

    print("Supporter Played:", state.supporterPlayed)
    print("Energy Attached:", state.energyAttached)
    print("Retreated:", state.retreated)

    print()

    print("Players:", len(state.players))
    print("Logs:", len(observation.logs))

    if observation.select is None:
        print("No active selection.")
    else:
        print("Selection Context:", observation.select.context)
        print("Legal Options:", len(observation.select.option))


# In[31]:


# Cell 27 â€” Is there a live Observation yet?


# In[32]:


print(type(Observation))


# In[33]:


import inspect

print(inspect.signature(to_observation_class))


# In[34]:


help(to_observation_class)


# # What we just learned
# 
# ### The official runtime does not create an Observation directly.
# 
# Instead it does:
# 
# Raw Simulator
#       â”‚
#       â–¼
# Python Dictionary
#       â”‚
#       â–¼
# to_observation_class(obs)
#       â”‚
#       â–¼
# Observation
#       â”‚
#       â–¼
# Your Adapter
#       â”‚
#       â–¼
# BattleSnapshot
#       â”‚
#       â–¼
# Decision Engine

# ### Every game turn probably does something like
# 
# obs = simulator.get_observation()
# 
# obs = to_observation_class(obs)
# 
# which becomes
# 
# dict
#  â†“
# Observation
#  â†“
# adapt_observation()
#  â†“
# BattleSnapshot
#  â†“
# Agent

# In[ ]:




