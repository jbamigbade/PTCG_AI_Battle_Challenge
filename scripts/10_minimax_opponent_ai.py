#!/usr/bin/env python
# coding: utf-8

# # Notebook 10 – Minimax Search and Opponent AI
# 
# This notebook extends the strategic planner by adding an intelligent opponent.
# 
# Instead of assuming that only the player's actions matter, the search now alternates between:
# 
# - the player's best move,
# - the opponent's best response,
# - and future counter-moves.
# 
# The notebook will implement:
# 
# - alternating player and opponent turns,
# - legal move generation for both sides,
# - game-state evaluation,
# - minimax search,
# - alpha-beta pruning,
# - best-move selection,
# - and explainable opponent-aware strategy.

# ## Step 1 – Import Libraries and Locate Project Artifacts
# 
# Notebook 10 loads the same saved project artifacts used in the earlier notebooks.
# 
# These files provide the card knowledge base, generated decks, and baseline battle results required for opponent-aware search.

# In[1]:


from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import pickle
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_project_paths():
    """
    Locate the project root, notebooks directory,
    and processed-data directory.
    """

    project_root = Path.cwd().resolve()

    if project_root.name == "notebooks":
        notebooks_dir = project_root
        project_root = project_root.parent
    else:
        notebooks_dir = project_root / "notebooks"

    processed_dir = notebooks_dir / "data" / "processed"

    return {
        "project_root": project_root,
        "notebooks_dir": notebooks_dir,
        "processed_dir": processed_dir,
    }


paths = get_project_paths()

PROJECT_ROOT = paths["project_root"]
NOTEBOOKS_DIR = paths["notebooks_dir"]
PROCESSED_DIR = paths["processed_dir"]

KNOWLEDGE_BASE_FILE = (
    PROCESSED_DIR / "card_knowledge_base.pkl"
)

STARTER_DECK_FILE = (
    PROCESSED_DIR / "starter_deck.pkl"
)

LIGHTNING_DECK_FILE = (
    PROCESSED_DIR / "lightning_deck.pkl"
)

BASELINE_RESULTS_FILE = (
    PROCESSED_DIR / "prototype_battle_results.csv"
)


print("Project Root:", PROJECT_ROOT)
print("Notebooks Directory:", NOTEBOOKS_DIR)
print("Processed Directory:", PROCESSED_DIR)

print("\nAvailable artifacts:")
print(
    "Knowledge Base:",
    KNOWLEDGE_BASE_FILE.exists(),
)
print(
    "Starter Deck:",
    STARTER_DECK_FILE.exists(),
)
print(
    "Lightning Deck:",
    LIGHTNING_DECK_FILE.exists(),
)
print(
    "Baseline Results:",
    BASELINE_RESULTS_FILE.exists(),
)


# ## Step 2 – Load Saved Project Artifacts
# 
# The minimax search uses the same card and deck artifacts created in the earlier notebooks.

# In[2]:


def load_pickle_file(file_path):
    """
    Load and return a Python object from a pickle file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required artifact was not found: {file_path}"
        )

    with open(file_path, "rb") as file:
        return pickle.load(file)


card_knowledge_base = load_pickle_file(
    KNOWLEDGE_BASE_FILE
)

starter_deck = load_pickle_file(
    STARTER_DECK_FILE
)

lightning_deck = load_pickle_file(
    LIGHTNING_DECK_FILE
)

baseline_results = pd.read_csv(
    BASELINE_RESULTS_FILE
)


print("Artifacts loaded successfully.\n")

print(
    "Knowledge Base Entries:",
    len(card_knowledge_base),
)

print(
    "Starter Deck Cards:",
    len(starter_deck),
)

print(
    "Lightning Deck Cards:",
    len(lightning_deck),
)

print(
    "Baseline Battles:",
    len(baseline_results),
)


# ## Step 3 – Recreate Battle State Classes
# 
# Notebook 10 uses lightweight battle-state objects so that minimax can copy and evaluate hypothetical future positions safely.

# In[3]:


@dataclass
class PokemonState:
    """Represents one Pokémon currently in play."""

    card: dict
    current_hp: float
    attached_energy: int = 0
    status: Optional[str] = None
    damage: float = 0
    is_active: bool = False


@dataclass
class PlayerState:
    """Represents one player's current battlefield."""

    active: PokemonState
    bench: List[PokemonState]
    prize_cards_remaining: int = 6
    hand_size: int = 7


@dataclass
class BattleState:
    """Complete state used by minimax search."""

    player: PlayerState
    opponent: PlayerState
    turn_number: int
    current_player: str


# ## Step 4 – Build the Minimax Test Battle
# 
# A controlled battle state is created with both players having an Active Pokémon and a Bench Pokémon.
# 
# This state will be used to test alternating player and opponent decisions.

# In[4]:


def get_pokemon_cards(deck, count):
    """
    Return the requested number of Pokémon cards.
    """

    pokemon_cards = [
        card
        for card in deck
        if card.get("category") == "Pokémon"
    ]

    if len(pokemon_cards) < count:
        raise ValueError(
            f"Only {len(pokemon_cards)} Pokémon cards were found."
        )

    return pokemon_cards[:count]


starter_pokemon = get_pokemon_cards(
    starter_deck,
    count=3,
)

lightning_pokemon = get_pokemon_cards(
    lightning_deck,
    count=3,
)


player_active = PokemonState(
    card=starter_pokemon[0],
    current_hp=float(
        starter_pokemon[0]["hp"]
    ),
    attached_energy=1,
    is_active=True,
)

player_bench = [
    PokemonState(
        card=starter_pokemon[1],
        current_hp=float(
            starter_pokemon[1]["hp"]
        ),
    ),
    PokemonState(
        card=starter_pokemon[2],
        current_hp=float(
            starter_pokemon[2]["hp"]
        ),
    ),
]

player_state = PlayerState(
    active=player_active,
    bench=player_bench,
)


opponent_active = PokemonState(
    card=lightning_pokemon[0],
    current_hp=float(
        lightning_pokemon[0]["hp"]
    ),
    attached_energy=1,
    is_active=True,
)

opponent_bench = [
    PokemonState(
        card=lightning_pokemon[1],
        current_hp=float(
            lightning_pokemon[1]["hp"]
        ),
    ),
    PokemonState(
        card=lightning_pokemon[2],
        current_hp=float(
            lightning_pokemon[2]["hp"]
        ),
    ),
]

opponent_state = PlayerState(
    active=opponent_active,
    bench=opponent_bench,
)


minimax_battle = BattleState(
    player=player_state,
    opponent=opponent_state,
    turn_number=1,
    current_player="Player",
)


print("Minimax battle state created successfully.\n")

print(
    "Player Active:",
    minimax_battle.player.active.card["name"],
)

print(
    "Player Energy:",
    minimax_battle.player.active.attached_energy,
)

print(
    "Player Bench:",
    [
        pokemon.card["name"]
        for pokemon in minimax_battle.player.bench
    ],
)

print()

print(
    "Opponent Active:",
    minimax_battle.opponent.active.card["name"],
)

print(
    "Opponent Energy:",
    minimax_battle.opponent.active.attached_energy,
)

print(
    "Opponent Bench:",
    [
        pokemon.card["name"]
        for pokemon in minimax_battle.opponent.bench
    ],
)

print()

print(
    "Current Player:",
    minimax_battle.current_player,
)


# ## Step 5 – Generate Legal Moves
# 
# Minimax alternates between Player and Opponent turns.
# 
# This function reads the attack structure used by the saved card knowledge base and returns only the attacks that are currently legal based on attached Energy.

# In[9]:


def get_legal_moves(pokemon_state: PokemonState):
    """
    Return every attack currently usable by the selected Pokémon.

    The saved attack dictionaries use:
    - Move Name
    - damage_numeric
    - energy_cost
    - Effect Explanation
    """

    legal_moves = []

    attacks = pokemon_state.card.get("attacks", [])

    for attack in attacks:
        move_name = attack.get(
            "Move Name",
            "Unknown Move",
        )

        damage = float(
            attack.get("damage_numeric", 0) or 0
        )

        required_energy = int(
            attack.get("energy_cost", 0) or 0
        )

        usable = (
            pokemon_state.attached_energy
            >= required_energy
        )

        if usable:
            legal_moves.append({
                "name": move_name,
                "damage": damage,
                "energy_cost": required_energy,
                "effect": attack.get(
                    "Effect Explanation"
                ),
            })

    return legal_moves


# ## Step 6 – Test Player Legal Moves
# 
# Display all attacks that the Player's Active Pokémon can legally use with its currently attached Energy.

# In[10]:


player_moves = get_legal_moves(
    minimax_battle.player.active
)

print("=" * 70)
print("PLAYER LEGAL MOVES")
print("=" * 70)

print(
    "Player:",
    minimax_battle.player.active.card["name"],
)

print(
    "Attached Energy:",
    minimax_battle.player.active.attached_energy,
)

print(
    "Legal Move Count:",
    len(player_moves),
)

print()

if not player_moves:
    print("No legal moves available.")

else:
    for move in player_moves:
        print("Move:", move["name"])
        print("Damage:", move["damage"])
        print(
            "Energy Cost:",
            move["energy_cost"],
        )

        effect = move.get("effect")

        if isinstance(effect, str) and effect.strip():
            print("Effect:", effect)

        print("-" * 45)


# ## Step 7 – Test Opponent Legal Moves
# 
# Display all attacks that the Opponent's Active Pokémon can legally use with its currently attached Energy.

# In[11]:


opponent_moves = get_legal_moves(
    minimax_battle.opponent.active
)

print("=" * 70)
print("OPPONENT LEGAL MOVES")
print("=" * 70)

print(
    "Opponent:",
    minimax_battle.opponent.active.card["name"],
)

print(
    "Attached Energy:",
    minimax_battle.opponent.active.attached_energy,
)

print(
    "Legal Move Count:",
    len(opponent_moves),
)

print()

if not opponent_moves:
    print("No legal moves available.")

else:
    for move in opponent_moves:
        print("Move:", move["name"])
        print("Damage:", move["damage"])
        print(
            "Energy Cost:",
            move["energy_cost"],
        )

        effect = move.get("effect")

        if isinstance(effect, str) and effect.strip():
            print("Effect:", effect)

        print("-" * 45)


# ## Step 8 – Create the Position Evaluation Function
# 
# Minimax assigns a numeric value to every future battle position.
# 
# Positive scores favor the Player, while negative scores favor the Opponent.
# 
# The evaluator considers:
# 
# - remaining Active Pokémon HP,
# - Knock Outs,
# - attached Energy,
# - Bench strength,
# - and Prize Card progress.

# In[12]:


def safe_max_hp(pokemon_state: PokemonState):
    """
    Return a safe maximum HP value for a Pokémon.
    """

    max_hp = float(
        pokemon_state.card.get("hp", 0) or 0
    )

    return max_hp


def total_bench_hp(bench):
    """
    Return the total remaining HP of healthy Bench Pokémon.
    """

    return sum(
        max(0.0, float(pokemon.current_hp))
        for pokemon in bench
    )


def evaluate_position(
    battle_state: BattleState,
):
    """
    Evaluate a battle position from the Player's perspective.

    Positive score:
        position favors the Player.

    Negative score:
        position favors the Opponent.
    """

    player_active = battle_state.player.active
    opponent_active = battle_state.opponent.active

    player_max_hp = safe_max_hp(
        player_active
    )

    opponent_max_hp = safe_max_hp(
        opponent_active
    )

    player_hp_ratio = (
        player_active.current_hp / player_max_hp
        if player_max_hp > 0
        else 0.0
    )

    opponent_hp_ratio = (
        opponent_active.current_hp / opponent_max_hp
        if opponent_max_hp > 0
        else 0.0
    )

    score = 0.0

    # Preserve the Player's Active Pokémon.
    score += player_hp_ratio * 100

    # Reward damage already dealt to the Opponent.
    score += (1 - opponent_hp_ratio) * 100

    # Penalize damage already taken by the Player.
    score -= (1 - player_hp_ratio) * 100

    # Energy advantage.
    energy_difference = (
        player_active.attached_energy
        - opponent_active.attached_energy
    )

    score += energy_difference * 15

    # Bench health advantage.
    player_bench_hp = total_bench_hp(
        battle_state.player.bench
    )

    opponent_bench_hp = total_bench_hp(
        battle_state.opponent.bench
    )

    score += (
        player_bench_hp
        - opponent_bench_hp
    ) * 0.10

    # Prize progress.
    player_prizes_taken = (
        6
        - battle_state.player.prize_cards_remaining
    )

    opponent_prizes_taken = (
        6
        - battle_state.opponent.prize_cards_remaining
    )

    score += (
        player_prizes_taken
        - opponent_prizes_taken
    ) * 150

    # Terminal Knock Out bonuses.
    if opponent_active.current_hp <= 0:
        score += 500

    if player_active.current_hp <= 0:
        score -= 500

    return round(score, 2)


# ## Step 9 – Test the Position Evaluator
# 
# Evaluate the current Minimax battle state and then verify that the score changes when either side takes damage.

# In[13]:


initial_score = evaluate_position(
    minimax_battle
)

print("=" * 70)
print("INITIAL POSITION EVALUATION")
print("=" * 70)

print(
    "Player Active:",
    minimax_battle.player.active.card["name"],
)

print(
    "Player HP:",
    minimax_battle.player.active.current_hp,
)

print(
    "Opponent Active:",
    minimax_battle.opponent.active.card["name"],
)

print(
    "Opponent HP:",
    minimax_battle.opponent.active.current_hp,
)

print(
    "Evaluation Score:",
    initial_score,
)


# In[14]:


player_advantage_test = deepcopy(
    minimax_battle
)

player_advantage_test.opponent.active.current_hp = 20

player_advantage_score = evaluate_position(
    player_advantage_test
)


opponent_advantage_test = deepcopy(
    minimax_battle
)

opponent_advantage_test.player.active.current_hp = 20

opponent_advantage_score = evaluate_position(
    opponent_advantage_test
)


print("=" * 70)
print("POSITION EVALUATION TESTS")
print("=" * 70)

print(
    "Original Position Score:",
    initial_score,
)

print(
    "Opponent Reduced to 20 HP:",
    player_advantage_score,
)

print(
    "Player Reduced to 20 HP:",
    opponent_advantage_score,
)


# ## Step 10 – Simulate One Legal Move
# 
# Each Minimax branch must create an independent future battle state.
# 
# The simulator:
# 
# - copies the current battle,
# - applies attack damage,
# - updates Prize Card progress after a Knock Out,
# - advances the turn,
# - and passes control to the other side.

# In[15]:


def apply_move(
    battle_state: BattleState,
    move: dict,
):
    """
    Apply one legal move for the side identified by
    battle_state.current_player.

    Returns a new BattleState and leaves the original unchanged.
    """

    future_state = deepcopy(battle_state)

    if future_state.current_player == "Player":
        acting_player = future_state.player
        defending_player = future_state.opponent
        next_player = "Opponent"

    elif future_state.current_player == "Opponent":
        acting_player = future_state.opponent
        defending_player = future_state.player
        next_player = "Player"

    else:
        raise ValueError(
            "current_player must be 'Player' or 'Opponent'."
        )

    damage = float(
        move.get("damage", 0) or 0
    )

    defending_player.active.current_hp = max(
        0.0,
        defending_player.active.current_hp - damage,
    )

    knockout_occurred = (
        defending_player.active.current_hp <= 0
    )

    if knockout_occurred:
        acting_player.prize_cards_remaining = max(
            0,
            acting_player.prize_cards_remaining - 1,
        )

    future_state.turn_number += 1
    future_state.current_player = next_player

    return future_state


# ## Step 11 – Test Alternating Move Simulation
# 
# The same move simulator is tested from both perspectives.
# 
# A Player action should damage the Opponent, while an Opponent action should damage the Player.

# In[16]:


def show_move_result(
    original_state: BattleState,
    future_state: BattleState,
    move: dict,
):
    """Print a readable summary of one simulated move."""

    print("Move:", move["name"])
    print("Damage:", move["damage"])

    print(
        "Original Player HP:",
        original_state.player.active.current_hp,
    )

    print(
        "Future Player HP:",
        future_state.player.active.current_hp,
    )

    print(
        "Original Opponent HP:",
        original_state.opponent.active.current_hp,
    )

    print(
        "Future Opponent HP:",
        future_state.opponent.active.current_hp,
    )

    print(
        "Next Player:",
        future_state.current_player,
    )

    print(
        "Future Turn Number:",
        future_state.turn_number,
    )


print("=" * 70)
print("PLAYER MOVE SIMULATION")
print("=" * 70)

player_test_moves = get_legal_moves(
    minimax_battle.player.active
)

if not player_test_moves:
    print("Player has no legal moves.")
else:
    player_test_move = player_test_moves[0]

    player_future_state = apply_move(
        minimax_battle,
        player_test_move,
    )

    show_move_result(
        minimax_battle,
        player_future_state,
        player_test_move,
    )


print("\n" + "=" * 70)
print("OPPONENT MOVE SIMULATION")
print("=" * 70)

opponent_turn_state = deepcopy(
    minimax_battle
)

opponent_turn_state.current_player = "Opponent"

opponent_test_moves = get_legal_moves(
    opponent_turn_state.opponent.active
)

if not opponent_test_moves:
    print("Opponent has no legal moves.")
else:
    opponent_test_move = opponent_test_moves[0]

    opponent_future_state = apply_move(
        opponent_turn_state,
        opponent_test_move,
    )

    show_move_result(
        opponent_turn_state,
        opponent_future_state,
        opponent_test_move,
    )


# ## Step 12 – Generate Actions for the Current Side
# 
# Minimax must automatically inspect whose turn it is and generate moves for that side.
# 
# When no attack is available, a Pass action prevents the search tree from ending incorrectly.

# In[17]:


def get_current_side_pokemon(
    battle_state: BattleState,
):
    """Return the Active Pokémon belonging to the current side."""

    if battle_state.current_player == "Player":
        return battle_state.player.active

    if battle_state.current_player == "Opponent":
        return battle_state.opponent.active

    raise ValueError(
        "current_player must be 'Player' or 'Opponent'."
    )


def get_current_legal_moves(
    battle_state: BattleState,
):
    """
    Return legal moves for whichever side currently has the turn.
    """

    active_pokemon = get_current_side_pokemon(
        battle_state
    )

    legal_moves = get_legal_moves(
        active_pokemon
    )

    if not legal_moves:
        legal_moves = [{
            "name": "Pass",
            "damage": 0.0,
            "energy_cost": 0,
            "effect": "No legal attack was available.",
        }]

    return legal_moves


# In[18]:


print("=" * 70)
print("CURRENT-SIDE MOVE GENERATION")
print("=" * 70)

player_turn_moves = get_current_legal_moves(
    minimax_battle
)

print("Current Side:", minimax_battle.current_player)
print(
    "Moves:",
    [move["name"] for move in player_turn_moves],
)


opponent_move_test = deepcopy(
    minimax_battle
)

opponent_move_test.current_player = "Opponent"

opponent_turn_moves = get_current_legal_moves(
    opponent_move_test
)

print()
print("Current Side:", opponent_move_test.current_player)
print(
    "Moves:",
    [move["name"] for move in opponent_turn_moves],
)


# ## Step 13 – Detect Terminal Battle States
# 
# Minimax stops searching when:
# 
# - either Active Pokémon is Knocked Out,
# - either side has taken all six Prize Cards,
# - or the requested search depth reaches zero.

# In[19]:


def is_terminal_state(
    battle_state: BattleState,
):
    """Return True when the simplified battle has ended."""

    player_knocked_out = (
        battle_state.player.active.current_hp <= 0
    )

    opponent_knocked_out = (
        battle_state.opponent.active.current_hp <= 0
    )

    player_won_all_prizes = (
        battle_state.player.prize_cards_remaining <= 0
    )

    opponent_won_all_prizes = (
        battle_state.opponent.prize_cards_remaining <= 0
    )

    return (
        player_knocked_out
        or opponent_knocked_out
        or player_won_all_prizes
        or opponent_won_all_prizes
    )


# In[20]:


normal_terminal_test = is_terminal_state(
    minimax_battle
)

knockout_terminal_state = deepcopy(
    minimax_battle
)

knockout_terminal_state.opponent.active.current_hp = 0

knockout_terminal_test = is_terminal_state(
    knockout_terminal_state
)

print("=" * 70)
print("TERMINAL STATE TEST")
print("=" * 70)

print(
    "Normal Battle Is Terminal:",
    normal_terminal_test,
)

print(
    "Opponent at 0 HP Is Terminal:",
    knockout_terminal_test,
)


# ## Step 14 – Implement Recursive Minimax Search
# 
# Minimax alternates between two objectives:
# 
# - the Player maximizes the evaluation score,
# - the Opponent minimizes the evaluation score.
# 
# The recursion continues until the battle ends or the search depth reaches zero.

# In[21]:


def minimax(
    battle_state: BattleState,
    depth: int,
):
    """
    Recursively evaluate a battle tree.

    Player turn:
        maximize the score.

    Opponent turn:
        minimize the score.
    """

    if (
        depth == 0
        or is_terminal_state(battle_state)
    ):
        return evaluate_position(
            battle_state
        )

    legal_moves = get_current_legal_moves(
        battle_state
    )

    if battle_state.current_player == "Player":
        best_score = -math.inf

        for move in legal_moves:
            future_state = apply_move(
                battle_state,
                move,
            )

            score = minimax(
                future_state,
                depth - 1,
            )

            best_score = max(
                best_score,
                score,
            )

        return best_score

    best_score = math.inf

    for move in legal_moves:
        future_state = apply_move(
            battle_state,
            move,
        )

        score = minimax(
            future_state,
            depth - 1,
        )

        best_score = min(
            best_score,
            score,
        )

    return best_score


# ## Step 15 – Test Minimax Search Depth
# 
# The initial battle is evaluated at several search depths.
# 
# Increasing depth allows the AI to consider more alternating Player and Opponent actions.

# In[22]:


print("=" * 70)
print("MINIMAX DEPTH TEST")
print("=" * 70)

for search_depth in range(1, 5):
    minimax_score = minimax(
        minimax_battle,
        depth=search_depth,
    )

    print(
        f"Depth {search_depth}: "
        f"Minimax Score = {minimax_score}"
    )


# ## Step 16 – Select the Best Minimax Move
# 
# The existing Minimax function returns only the value of a battle position.
# 
# This step evaluates every legal Player move and returns both:
# 
# - the selected move,
# - and the Minimax score produced by the opponent's best response.

# In[23]:


def choose_best_minimax_move(
    battle_state: BattleState,
    depth: int,
):
    """
    Choose the Player move with the highest Minimax score.

    Parameters
    ----------
    battle_state:
        Current battle position. The current player must be "Player".

    depth:
        Number of future plies to search after applying each candidate move.

    Returns
    -------
    dict or None
        A dictionary containing the best move, its score,
        and the resulting battle state.
    """

    if battle_state.current_player != "Player":
        raise ValueError(
            "choose_best_minimax_move requires "
            "current_player='Player'."
        )

    legal_moves = get_current_legal_moves(
        battle_state
    )

    if not legal_moves:
        return None

    best_result = None
    best_score = -math.inf

    for move in legal_moves:
        future_state = apply_move(
            battle_state,
            move,
        )

        score = minimax(
            future_state,
            depth=max(depth - 1, 0),
        )

        if score > best_score:
            best_score = score

            best_result = {
                "move": move,
                "score": round(score, 2),
                "future_state": future_state,
            }

    return best_result


# ## Step 17 – Test Minimax Best-Move Selection
# 
# The AI now evaluates every legal Player move and reports the strongest move at several search depths.

# In[24]:


print("=" * 70)
print("MINIMAX BEST-MOVE TEST")
print("=" * 70)

for search_depth in range(1, 5):
    best_result = choose_best_minimax_move(
        minimax_battle,
        depth=search_depth,
    )

    print(f"\nSearch Depth: {search_depth}")
    print("-" * 50)

    if best_result is None:
        print("No legal move was found.")
        continue

    print(
        "Chosen Move:",
        best_result["move"]["name"],
    )

    print(
        "Move Damage:",
        best_result["move"]["damage"],
    )

    print(
        "Minimax Score:",
        best_result["score"],
    )

    print(
        "Opponent HP After Move:",
        best_result[
            "future_state"
        ].opponent.active.current_hp,
    )

    print(
        "Next Player:",
        best_result[
            "future_state"
        ].current_player,
    )


# ## Step 18 – Compare Root Move Scores
# 
# To make the search transparent, display the Minimax score assigned to every legal move from the current position.

# In[25]:


def evaluate_root_moves(
    battle_state: BattleState,
    depth: int,
):
    """
    Evaluate every legal move from the root battle state.
    """

    if battle_state.current_player != "Player":
        raise ValueError(
            "Root move evaluation requires "
            "current_player='Player'."
        )

    results = []

    legal_moves = get_current_legal_moves(
        battle_state
    )

    for move in legal_moves:
        future_state = apply_move(
            battle_state,
            move,
        )

        score = minimax(
            future_state,
            depth=max(depth - 1, 0),
        )

        results.append({
            "move": move,
            "score": round(score, 2),
            "future_state": future_state,
        })

    return sorted(
        results,
        key=lambda result: (
            result["score"],
            result["move"]["damage"],
            -result["move"]["energy_cost"],
        ),
        reverse=True,
    )


root_move_results = evaluate_root_moves(
    minimax_battle,
    depth=4,
)

print("=" * 70)
print("ROOT MOVE COMPARISON")
print("=" * 70)

for rank, result in enumerate(
    root_move_results,
    start=1,
):
    print(f"Rank {rank}")
    print(
        "Move:",
        result["move"]["name"],
    )
    print(
        "Damage:",
        result["move"]["damage"],
    )
    print(
        "Energy Cost:",
        result["move"]["energy_cost"],
    )
    print(
        "Minimax Score:",
        result["score"],
    )
    print("-" * 50)


# ## Step 19 – Track Minimax Search Statistics
# 
# The search engine now records:
# 
# - nodes visited,
# - terminal positions evaluated,
# - and the deepest level reached.
# 
# These measurements will later be compared with alpha-beta pruning.

# In[26]:


@dataclass
class SearchStatistics:
    """Performance measurements for one search."""

    nodes_visited: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    maximum_depth_reached: int = 0


def minimax_with_statistics(
    battle_state: BattleState,
    depth: int,
    statistics: SearchStatistics,
    starting_depth: Optional[int] = None,
):
    """
    Run Minimax while recording search statistics.
    """

    if starting_depth is None:
        starting_depth = depth

    statistics.nodes_visited += 1

    depth_reached = (
        starting_depth - depth
    )

    statistics.maximum_depth_reached = max(
        statistics.maximum_depth_reached,
        depth_reached,
    )

    terminal = is_terminal_state(
        battle_state
    )

    if depth == 0 or terminal:
        statistics.leaf_nodes += 1

        if terminal:
            statistics.terminal_nodes += 1

        return evaluate_position(
            battle_state
        )

    legal_moves = get_current_legal_moves(
        battle_state
    )

    if battle_state.current_player == "Player":
        best_score = -math.inf

        for move in legal_moves:
            future_state = apply_move(
                battle_state,
                move,
            )

            score = minimax_with_statistics(
                future_state,
                depth - 1,
                statistics,
                starting_depth,
            )

            best_score = max(
                best_score,
                score,
            )

        return best_score

    best_score = math.inf

    for move in legal_moves:
        future_state = apply_move(
            battle_state,
            move,
        )

        score = minimax_with_statistics(
            future_state,
            depth - 1,
            statistics,
            starting_depth,
        )

        best_score = min(
            best_score,
            score,
        )

    return best_score


# ## Step 20 – Measure Standard Minimax Search
# 
# Run standard Minimax at several depths and record how the number of explored nodes grows.

# In[27]:


minimax_statistics_rows = []

print("=" * 70)
print("STANDARD MINIMAX SEARCH STATISTICS")
print("=" * 70)

for search_depth in range(1, 7):
    statistics = SearchStatistics()

    score = minimax_with_statistics(
        minimax_battle,
        depth=search_depth,
        statistics=statistics,
    )

    row = {
        "depth": search_depth,
        "score": round(score, 2),
        "nodes_visited": statistics.nodes_visited,
        "leaf_nodes": statistics.leaf_nodes,
        "terminal_nodes": statistics.terminal_nodes,
        "maximum_depth_reached": (
            statistics.maximum_depth_reached
        ),
    }

    minimax_statistics_rows.append(row)

    print(f"\nDepth {search_depth}")
    print("Score:", row["score"])
    print(
        "Nodes Visited:",
        row["nodes_visited"],
    )
    print(
        "Leaf Nodes:",
        row["leaf_nodes"],
    )
    print(
        "Terminal Nodes:",
        row["terminal_nodes"],
    )
    print(
        "Maximum Depth Reached:",
        row["maximum_depth_reached"],
    )


# ## Step 21 – Implement Alpha-Beta Pruning
# 
# Alpha-beta pruning produces the same Minimax result while avoiding branches that cannot change the final decision.
# 
# - Alpha stores the best score already guaranteed to the maximizing Player.
# - Beta stores the best score already guaranteed to the minimizing Opponent.
# - A branch is pruned when further exploration cannot improve the outcome.

# In[28]:


@dataclass
class AlphaBetaStatistics:
    """Performance measurements for alpha-beta search."""

    nodes_visited: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    branches_pruned: int = 0
    maximum_depth_reached: int = 0


def alpha_beta(
    battle_state: BattleState,
    depth: int,
    alpha: float,
    beta: float,
    statistics: AlphaBetaStatistics,
    starting_depth: Optional[int] = None,
):
    """
    Run recursive Minimax search with alpha-beta pruning.
    """

    if starting_depth is None:
        starting_depth = depth

    statistics.nodes_visited += 1

    depth_reached = (
        starting_depth - depth
    )

    statistics.maximum_depth_reached = max(
        statistics.maximum_depth_reached,
        depth_reached,
    )

    terminal = is_terminal_state(
        battle_state
    )

    if depth == 0 or terminal:
        statistics.leaf_nodes += 1

        if terminal:
            statistics.terminal_nodes += 1

        return evaluate_position(
            battle_state
        )

    legal_moves = get_current_legal_moves(
        battle_state
    )

    if battle_state.current_player == "Player":
        value = -math.inf

        for move_index, move in enumerate(
            legal_moves
        ):
            future_state = apply_move(
                battle_state,
                move,
            )

            child_score = alpha_beta(
                future_state,
                depth - 1,
                alpha,
                beta,
                statistics,
                starting_depth,
            )

            value = max(
                value,
                child_score,
            )

            alpha = max(
                alpha,
                value,
            )

            if alpha >= beta:
                statistics.branches_pruned += (
                    len(legal_moves)
                    - move_index
                    - 1
                )
                break

        return value

    value = math.inf

    for move_index, move in enumerate(
        legal_moves
    ):
        future_state = apply_move(
            battle_state,
            move,
        )

        child_score = alpha_beta(
            future_state,
            depth - 1,
            alpha,
            beta,
            statistics,
            starting_depth,
        )

        value = min(
            value,
            child_score,
        )

        beta = min(
            beta,
            value,
        )

        if alpha >= beta:
            statistics.branches_pruned += (
                len(legal_moves)
                - move_index
                - 1
            )
            break

    return value


# ## Step 22 – Test Alpha-Beta Search
# 
# Confirm that alpha-beta pruning returns the same scores as standard Minimax while recording the number of branches removed.

# In[29]:


alpha_beta_rows = []

print("=" * 70)
print("ALPHA-BETA SEARCH TEST")
print("=" * 70)

for search_depth in range(1, 7):
    statistics = AlphaBetaStatistics()

    score = alpha_beta(
        minimax_battle,
        depth=search_depth,
        alpha=-math.inf,
        beta=math.inf,
        statistics=statistics,
    )

    row = {
        "depth": search_depth,
        "score": round(score, 2),
        "nodes_visited": statistics.nodes_visited,
        "leaf_nodes": statistics.leaf_nodes,
        "terminal_nodes": statistics.terminal_nodes,
        "branches_pruned": statistics.branches_pruned,
        "maximum_depth_reached": (
            statistics.maximum_depth_reached
        ),
    }

    alpha_beta_rows.append(row)

    print(f"\nDepth {search_depth}")
    print("Score:", row["score"])
    print(
        "Nodes Visited:",
        row["nodes_visited"],
    )
    print(
        "Leaf Nodes:",
        row["leaf_nodes"],
    )
    print(
        "Branches Pruned:",
        row["branches_pruned"],
    )
    print(
        "Maximum Depth Reached:",
        row["maximum_depth_reached"],
    )


# ## Step 23 – Compare Search Performance
# 
# Standard Minimax and alpha-beta pruning are compared at each depth.
# 
# Both methods should return the same score, but alpha-beta may inspect fewer nodes.

# In[30]:


minimax_statistics_df = pd.DataFrame(
    minimax_statistics_rows
)

alpha_beta_df = pd.DataFrame(
    alpha_beta_rows
)

search_comparison = (
    minimax_statistics_df[
        [
            "depth",
            "score",
            "nodes_visited",
            "leaf_nodes",
        ]
    ]
    .rename(
        columns={
            "score": "minimax_score",
            "nodes_visited": "minimax_nodes",
            "leaf_nodes": "minimax_leaf_nodes",
        }
    )
    .merge(
        alpha_beta_df[
            [
                "depth",
                "score",
                "nodes_visited",
                "leaf_nodes",
                "branches_pruned",
            ]
        ].rename(
            columns={
                "score": "alpha_beta_score",
                "nodes_visited": "alpha_beta_nodes",
                "leaf_nodes": "alpha_beta_leaf_nodes",
            }
        ),
        on="depth",
        how="inner",
    )
)

search_comparison["scores_match"] = (
    search_comparison["minimax_score"]
    == search_comparison["alpha_beta_score"]
)

search_comparison["nodes_saved"] = (
    search_comparison["minimax_nodes"]
    - search_comparison["alpha_beta_nodes"]
)

search_comparison["reduction_percent"] = np.where(
    search_comparison["minimax_nodes"] > 0,
    (
        search_comparison["nodes_saved"]
        / search_comparison["minimax_nodes"]
        * 100
    ),
    0,
).round(2)

display(search_comparison)


# In[ ]:




