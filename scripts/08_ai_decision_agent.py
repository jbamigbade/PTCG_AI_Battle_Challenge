#!/usr/bin/env python
# coding: utf-8

# # Notebook 08 – AI Decision Agent
# 
# ## Objective
# 
# Notebook 07 created a functional prototype Battle Engine capable of running complete automated matches.
# 
# However, the current engine uses simple scripted behavior:
# 
# - Attach the first available Energy card.
# - Select the first usable attack.
# - Promote the first available Benched Pokémon.
# - Ignore strategic alternatives.
# 
# In this notebook, we create an AI Decision Agent that evaluates available actions and selects actions based on battle value.
# 
# The AI agent will initially use a transparent rule-based and heuristic strategy.
# 
# By the end of this notebook, the agent will be able to:
# 
# - Evaluate the current battle state.
# - Score available attacks.
# - Select the best usable attack.
# - Choose where to attach Energy.
# - Compare Active and Benched Pokémon.
# - Make simple retreat decisions.
# - Record explanations for its decisions.
# - Replace scripted choices in the Battle Engine.
# - Compare AI-guided performance against the baseline policy.
# 
# This notebook focuses on explainable decision-making rather than reinforcement learning.
# 
# More advanced search and learning methods can be added after the heuristic agent is validated.

# ## Step 1 – Import Libraries and Locate Project Artifacts
# 
# The AI Decision Agent uses the saved Card Knowledge Base, generated decks, and battle simulation results from previous notebooks.
# 
# The Battle Engine functions from Notebook 07 will initially be recreated or moved into reusable Python modules later.

# In[7]:


from pathlib import Path
from copy import deepcopy
import pickle
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Notebook 08 is currently running with the project folder as the working directory.


def get_project_paths():
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

KNOWLEDGE_BASE_FILE = PROCESSED_DIR / "card_knowledge_base.pkl"
STARTER_DECK_FILE = PROCESSED_DIR / "starter_deck.pkl"
LIGHTNING_DECK_FILE = PROCESSED_DIR / "lightning_deck.pkl"
BASELINE_RESULTS_FILE = PROCESSED_DIR / "prototype_battle_results.csv"

print("Project Root:", PROJECT_ROOT)
print("Notebooks Directory:", NOTEBOOKS_DIR)
print("Processed Directory:", PROCESSED_DIR)

print("\nAvailable artifacts:")
print("Knowledge Base:", KNOWLEDGE_BASE_FILE.exists())
print("Starter Deck:", STARTER_DECK_FILE.exists())
print("Lightning Deck:", LIGHTNING_DECK_FILE.exists())
print("Baseline Results:", BASELINE_RESULTS_FILE.exists())


# ## Step 2 – Load Saved Project Artifacts
# 
# The AI Decision Agent requires the saved Card Knowledge Base, generated decks, and baseline battle results from the earlier notebooks.
# 
# This step loads those artifacts and verifies their Python object types before the decision logic is developed.

# In[8]:


def load_pickle_file(file_path):
    """Load and return a Python object stored in a pickle file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file was not found: {file_path}")

    with open(file_path, "rb") as file:
        return pickle.load(file)


card_knowledge_base = load_pickle_file(KNOWLEDGE_BASE_FILE)
starter_deck = load_pickle_file(STARTER_DECK_FILE)
lightning_deck = load_pickle_file(LIGHTNING_DECK_FILE)

baseline_results = pd.read_csv(BASELINE_RESULTS_FILE)


print("Artifacts loaded successfully.\n")

print("Card Knowledge Base Type:", type(card_knowledge_base).__name__)
print("Starter Deck Type:", type(starter_deck).__name__)
print("Lightning Deck Type:", type(lightning_deck).__name__)
print("Baseline Results Type:", type(baseline_results).__name__)

print("\nArtifact sizes:")

try:
    print("Card Knowledge Base Entries:", len(card_knowledge_base))
except TypeError:
    print("Card Knowledge Base Entries: length unavailable")

try:
    print("Starter Deck Entries:", len(starter_deck))
except TypeError:
    print("Starter Deck Entries: length unavailable")

try:
    print("Lightning Deck Entries:", len(lightning_deck))
except TypeError:
    print("Lightning Deck Entries: length unavailable")

print("Baseline Result Rows:", len(baseline_results))
print("Baseline Result Columns:", len(baseline_results.columns))


# ## Step 3 – Inspect Loaded Data Structures
# 
# Before creating AI decision rules, the internal structure of the saved knowledge base and decks must be inspected.
# 
# This prevents the agent from making assumptions about field names or object formats.

# In[9]:


def preview_object(name, obj, max_items=3):
    """Display a short structural preview of a loaded object."""
    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Python Type:", type(obj).__name__)

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("Number of Keys:", len(keys))
        print("First Keys:", keys[:max_items])

        for key in keys[:max_items]:
            print(f"\nKey: {key}")
            print("Value Type:", type(obj[key]).__name__)
            print("Value Preview:", repr(obj[key])[:500])

    elif isinstance(obj, list):
        print("Number of Items:", len(obj))

        for index, item in enumerate(obj[:max_items]):
            print(f"\nItem {index}")
            print("Item Type:", type(item).__name__)
            print("Item Preview:", repr(item)[:500])

    elif isinstance(obj, pd.DataFrame):
        print("Shape:", obj.shape)
        print("Columns:", obj.columns.tolist())
        display(obj.head(max_items))

    else:
        print("Preview:", repr(obj)[:1000])


preview_object("CARD KNOWLEDGE BASE", card_knowledge_base)
preview_object("STARTER DECK", starter_deck)
preview_object("LIGHTNING DECK", lightning_deck)
preview_object("BASELINE RESULTS", baseline_results)


# # Step 4 – Create Battle State Objects
# 
# Before building the AI decision system, we define lightweight classes that describe the current battle.
# 
# These classes keep the decision engine independent from the simulator itself and make later reinforcement-learning upgrades much easier.

# In[11]:


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PokemonState:
    """Represents a Pokémon currently in play."""

    card: dict
    current_hp: float
    attached_energy: int = 0
    status: Optional[str] = None
    damage: float = 0
    is_active: bool = False


@dataclass
class PlayerState:
    """Represents one player's battlefield."""

    active: PokemonState
    bench: List[PokemonState]
    prize_cards_remaining: int = 6
    hand_size: int = 7


@dataclass
class BattleState:
    """Complete battle state used by the AI."""

    player: PlayerState
    opponent: PlayerState
    turn_number: int
    current_player: str


# ## Step 5 – Build a Sample Battle State
# 
# Before connecting the AI to the full simulator, we create a small battle state using cards from the generated decks.
# 
# This gives us a controlled example for testing attack scoring, Energy attachment, retreat logic, and AI explanations.

# In[12]:


def first_pokemon_card(deck):
    """Return the first Pokémon card found in a deck."""
    for card in deck:
        if card.get("category") == "Pokémon":
            return card

    raise ValueError("No Pokémon card was found in the deck.")


def next_pokemon_cards(deck, count=2):
    """Return the requested number of Pokémon cards from a deck."""
    pokemon_cards = [
        card for card in deck
        if card.get("category") == "Pokémon"
    ]

    if len(pokemon_cards) < count:
        raise ValueError(
            f"Only {len(pokemon_cards)} Pokémon cards were found."
        )

    return pokemon_cards[:count]


starter_pokemon = next_pokemon_cards(starter_deck, count=3)
lightning_pokemon = next_pokemon_cards(lightning_deck, count=1)


player_active = PokemonState(
    card=starter_pokemon[0],
    current_hp=float(starter_pokemon[0]["hp"]),
    attached_energy=1,
    is_active=True,
)

player_bench = [
    PokemonState(
        card=starter_pokemon[1],
        current_hp=float(starter_pokemon[1]["hp"]),
    ),
    PokemonState(
        card=starter_pokemon[2],
        current_hp=float(starter_pokemon[2]["hp"]),
    ),
]

player = PlayerState(
    active=player_active,
    bench=player_bench,
)


opponent_active = PokemonState(
    card=lightning_pokemon[0],
    current_hp=float(lightning_pokemon[0]["hp"]),
    attached_energy=1,
    is_active=True,
)

opponent = PlayerState(
    active=opponent_active,
    bench=[],
)


battle = BattleState(
    player=player,
    opponent=opponent,
    turn_number=1,
    current_player="Player 1",
)


print("Battle state successfully created.\n")
print("Player Active:", battle.player.active.card["name"])
print("Player Active HP:", battle.player.active.current_hp)
print("Player Bench:", [pokemon.card["name"] for pokemon in battle.player.bench])

print("\nOpponent Active:", battle.opponent.active.card["name"])
print("Opponent Active HP:", battle.opponent.active.current_hp)

print("\nTurn Number:", battle.turn_number)
print("Current Player:", battle.current_player)


# ## Step 6 – Evaluate Pokémon Strength
# 
# The AI first estimates how strong each Pokémon is in the current battle.
# 
# Rather than choosing attacks immediately, the agent computes a numerical score based on the Pokémon's current state.
# 
# Later notebooks can expand this into reinforcement learning or search algorithms.

# In[13]:


def evaluate_pokemon(pokemon: PokemonState):
    """
    Computes a heuristic score for a Pokémon.
    Higher scores indicate a stronger battle position.
    """

    score = 0.0

    card = pokemon.card

    # --------------------------
    # Remaining HP
    # --------------------------
    max_hp = float(card.get("hp", 0) or 0)

    if max_hp > 0:
        hp_ratio = pokemon.current_hp / max_hp
        score += hp_ratio * 40

    # --------------------------
    # Overall card quality
    # --------------------------
    score += float(card.get("overall_score", 0))

    # --------------------------
    # Attached Energy
    # --------------------------
    score += pokemon.attached_energy * 10

    # --------------------------
    # Active Bonus
    # --------------------------
    if pokemon.is_active:
        score += 5

    # --------------------------
    # Status Penalty
    # --------------------------
    if pokemon.status is not None:
        score -= 15

    return round(score, 2)


# ## Step 7 – Test Pokémon Evaluation
# 
# Evaluate both Active Pokémon using the heuristic scoring function.

# In[14]:


player_score = evaluate_pokemon(
    battle.player.active
)

opponent_score = evaluate_pokemon(
    battle.opponent.active
)

print("Player Pokémon :", battle.player.active.card["name"])
print("Score :", player_score)

print()

print("Opponent Pokémon :", battle.opponent.active.card["name"])
print("Score :", opponent_score)


# ## Step 8 – Evaluate Available Attacks
# 
# The AI now scores every attack the Active Pokémon can use.
# 
# Each attack receives a heuristic score based on:
# 
# - Base damage
# - Energy requirement
# - Whether the attack is currently usable
# - Additional attack effects (placeholder for future improvements)
# 
# The highest-scoring attack will later be selected automatically.

# In[17]:


def evaluate_attack(pokemon: PokemonState):
    """
    Evaluate every attack available to a Pokémon.
    Returns a list of scored attacks.
    """

    attacks = pokemon.card.get("attacks", [])

    attack_scores = []

    for attack in attacks:

        attack_name = attack.get("Move Name")

        damage = float(attack.get("damage_numeric", 0) or 0)

        energy_required = int(attack.get("energy_cost", 0) or 0)

        usable = pokemon.attached_energy >= energy_required

        # Attack cannot currently be used
        if not usable:
            score = -1000

        else:
            score = 0

            # Base damage
            score += damage

            # Reward usable attacks
            score += 25

            # Slight penalty for higher energy costs
            score -= energy_required * 2

            # Small bonus for attack effects
            effect = attack.get("Effect Explanation")

            if isinstance(effect, str) and effect.strip():
                score += 10

        attack_scores.append({
            "name": attack_name,
            "damage": damage,
            "energy_required": energy_required,
            "usable": usable,
            "score": round(score, 2),
        })

    return attack_scores


# ## Step 9 – Test Attack Evaluation
# 
# Display the heuristic score for every attack available to the player's Active Pokémon.

# In[18]:


attack_scores = evaluate_attack(
    battle.player.active
)

print(f"Active Pokémon: {battle.player.active.card['name']}\n")

for attack in attack_scores:

    print(f"Attack: {attack['name']}")
    print(f"Damage: {attack['damage']}")
    print(f"Energy Required: {attack['energy_required']}")
    print(f"Usable: {attack['usable']}")
    print(f"Score: {attack['score']}")
    print("-" * 50)


# ## Step 10 – Select the Best Available Attack
# 
# The AI compares all attack scores and chooses the highest-scoring attack that is currently usable.
# 
# This is the first autonomous battle decision made by the AI.

# In[19]:


def choose_best_attack(pokemon: PokemonState):
    """Return the highest-scoring usable attack."""

    attacks = evaluate_attack(pokemon)

    usable_attacks = [
        attack for attack in attacks
        if attack["usable"]
    ]

    if not usable_attacks:
        return None

    best_attack = max(
        usable_attacks,
        key=lambda attack: attack["score"]
    )

    return best_attack


# ## Step 11 – Test the AI's First Decision
# 
# The AI evaluates all legal attacks and chooses the highest-scoring option.

# In[20]:


best_attack = choose_best_attack(
    battle.player.active
)

print("=" * 60)
print("AI DECISION")
print("=" * 60)

if best_attack is None:

    print("No legal attacks available.")

else:

    print(f"Chosen Attack : {best_attack['name']}")
    print(f"Damage        : {best_attack['damage']}")
    print(f"Energy Needed : {best_attack['energy_required']}")
    print(f"AI Score      : {best_attack['score']}")


# ## Step 12 – Create the AI Decision Agent Class
# 
# The previously tested evaluation functions are now organized into a reusable AI agent class.
# 
# The class will eventually manage attack selection, Energy attachment, retreat decisions, and explanations.

# In[21]:


class AIDecisionAgent:
    """Explainable heuristic decision agent for Pokémon battles."""

    def evaluate_pokemon(self, pokemon: PokemonState):
        """Compute a heuristic score for a Pokémon."""

        score = 0.0
        card = pokemon.card

        max_hp = float(card.get("hp", 0) or 0)

        if max_hp > 0:
            hp_ratio = pokemon.current_hp / max_hp
            score += hp_ratio * 40

        score += float(card.get("overall_score", 0) or 0)
        score += pokemon.attached_energy * 10

        if pokemon.is_active:
            score += 5

        if pokemon.status is not None:
            score -= 15

        return round(score, 2)

    def evaluate_attacks(self, pokemon: PokemonState):
        """Score every listed move for the selected Pokémon."""

        attacks = pokemon.card.get("attacks", [])
        attack_scores = []

        for attack in attacks:
            attack_name = attack.get("Move Name")
            damage = float(attack.get("damage_numeric", 0) or 0)
            energy_required = int(attack.get("energy_cost", 0) or 0)
            usable = pokemon.attached_energy >= energy_required

            if not usable:
                score = -1000.0
            else:
                score = damage + 25
                score -= energy_required * 2

                effect = attack.get("Effect Explanation")

                if isinstance(effect, str) and effect.strip():
                    score += 10

            attack_scores.append({
                "name": attack_name,
                "damage": damage,
                "energy_required": energy_required,
                "usable": usable,
                "score": round(score, 2),
                "effect": attack.get("Effect Explanation"),
            })

        return attack_scores

    def choose_best_attack(self, pokemon: PokemonState):
        """Choose the highest-scoring legal attack."""

        attack_scores = self.evaluate_attacks(pokemon)

        usable_attacks = [
            attack
            for attack in attack_scores
            if attack["usable"]
        ]

        if not usable_attacks:
            return None

        return max(
            usable_attacks,
            key=lambda attack: (
                attack["score"],
                attack["damage"],
                -attack["energy_required"],
            )
        )


# ## Step 13 – Test the AI Decision Agent
# 
# Instantiate the reusable agent and confirm that its evaluations match the earlier standalone functions.

# In[22]:


agent = AIDecisionAgent()

player_pokemon_score = agent.evaluate_pokemon(
    battle.player.active
)

opponent_pokemon_score = agent.evaluate_pokemon(
    battle.opponent.active
)

agent_attack_scores = agent.evaluate_attacks(
    battle.player.active
)

agent_best_attack = agent.choose_best_attack(
    battle.player.active
)


print("=" * 60)
print("AI AGENT TEST")
print("=" * 60)

print(
    f"Player Pokémon Score   : "
    f"{battle.player.active.card['name']} = {player_pokemon_score}"
)

print(
    f"Opponent Pokémon Score : "
    f"{battle.opponent.active.card['name']} = {opponent_pokemon_score}"
)

print("\nAttack Evaluations:")

for attack in agent_attack_scores:
    print(
        f"- {attack['name']}: "
        f"score={attack['score']}, "
        f"usable={attack['usable']}"
    )

print("\nChosen Attack:")

if agent_best_attack is None:
    print("No legal attack is available.")
else:
    print(agent_best_attack["name"])
    print("Damage:", agent_best_attack["damage"])
    print("Energy Required:", agent_best_attack["energy_required"])
    print("Score:", agent_best_attack["score"])


# ## Step 14 – Explain the Attack Decision
# 
# The agent should not only choose an action; it should also explain the main reasons behind the decision.
# 
# This makes the heuristic policy transparent and easier to debug.

# In[23]:


def explain_attack_decision(agent, pokemon):
    """Return a readable explanation of the selected attack."""

    attack_scores = agent.evaluate_attacks(pokemon)
    best_attack = agent.choose_best_attack(pokemon)

    pokemon_name = pokemon.card.get("name", "Unknown Pokémon")

    if best_attack is None:
        unavailable = [
            attack["name"]
            for attack in attack_scores
            if not attack["usable"]
        ]

        return {
            "pokemon": pokemon_name,
            "chosen_attack": None,
            "explanation": (
                f"{pokemon_name} has no currently usable attacks. "
                f"Unavailable moves: {unavailable}"
            ),
        }

    explanation_parts = [
        f"{pokemon_name} selected {best_attack['name']}.",
        f"The attack is usable with {pokemon.attached_energy} attached Energy.",
        f"It requires {best_attack['energy_required']} Energy.",
        f"It has an AI score of {best_attack['score']}.",
        f"Its recorded damage is {best_attack['damage']}.",
    ]

    tied_attacks = [
        attack["name"]
        for attack in attack_scores
        if attack["usable"]
        and attack["score"] == best_attack["score"]
        and attack["name"] != best_attack["name"]
    ]

    if tied_attacks:
        explanation_parts.append(
            "The following legal attacks had the same score: "
            + ", ".join(tied_attacks)
            + ". The tie was resolved using damage and Energy cost."
        )

    return {
        "pokemon": pokemon_name,
        "chosen_attack": best_attack,
        "explanation": " ".join(explanation_parts),
    }


decision_report = explain_attack_decision(
    agent,
    battle.player.active
)

print("=" * 60)
print("AI DECISION EXPLANATION")
print("=" * 60)
print(decision_report["explanation"])


# ## Step 15 – Predict Knockouts
# 
# The AI now checks whether a legal attack can Knock Out the opponent's Active Pokémon.
# 
# Attacks that can immediately Knock Out the opponent receive a large strategic bonus.

# In[25]:


def evaluate_attacks_with_knockout(
    agent,
    attacker: PokemonState,
    defender: PokemonState,
):
    """
    Score attacks while considering whether they can Knock Out
    the opponent's Active Pokémon.
    """

    attack_scores = agent.evaluate_attacks(attacker)
    defender_hp = float(defender.current_hp)

    for attack in attack_scores:
        attack["can_knock_out"] = (
            attack["usable"]
            and attack["damage"] >= defender_hp
        )

        if attack["can_knock_out"]:
            attack["score"] += 500

        attack["score"] = round(attack["score"], 2)

    return attack_scores


# ## Step 16 – Test Knockout Prediction
# 
# Evaluate the player's attacks against the opponent's current HP and identify any immediate Knock Out opportunities.

# In[26]:


knockout_attack_scores = evaluate_attacks_with_knockout(
    agent,
    battle.player.active,
    battle.opponent.active,
)

print("=" * 60)
print("KNOCKOUT EVALUATION")
print("=" * 60)

print(
    "Opponent:",
    battle.opponent.active.card["name"],
)

print(
    "Opponent Current HP:",
    battle.opponent.active.current_hp,
)

print()

for attack in knockout_attack_scores:
    print("Attack:", attack["name"])
    print("Damage:", attack["damage"])
    print("Usable:", attack["usable"])
    print("Can Knock Out:", attack["can_knock_out"])
    print("Strategic Score:", attack["score"])
    print("-" * 50)


# ## Step 17 – Strategic Attack Selection
# 
# The AI selects the highest-scoring legal attack after incorporating Knock Out prediction into its evaluation.

# In[28]:


def choose_strategic_attack(agent, attacker, defender):
    """
    Choose the strongest legal attack after considering
    knockout opportunities.
    """

    attack_scores = evaluate_attacks_with_knockout(
        agent,
        attacker,
        defender
    )

    legal_attacks = [
        attack
        for attack in attack_scores
        if attack["usable"]
    ]

    if not legal_attacks:
        return None

    return max(
        legal_attacks,
        key=lambda attack: (
            attack["score"],
            attack["can_knock_out"],
            attack["damage"],
            -attack["energy_required"],
        ),
    )


# ## Step 18 – Demonstrate Strategic Decision Making
# 
# The AI first evaluates the current battle, then a hypothetical situation where Eevee ex has enough Energy to use every attack.
# 
# This demonstrates that the AI immediately switches to the knockout attack when it becomes legal.

# In[29]:


print("=" * 70)
print("CURRENT BATTLE")
print("=" * 70)

current_choice = choose_strategic_attack(
    agent,
    battle.player.active,
    battle.opponent.active,
)

print("Attached Energy:", battle.player.active.attached_energy)
print("Chosen Attack:", current_choice["name"])
print("Damage:", current_choice["damage"])
print("Knock Out:", current_choice["can_knock_out"])
print("Score:", current_choice["score"])


print("\n")

test_attacker = deepcopy(battle.player.active)
test_attacker.attached_energy = 3

future_choice = choose_strategic_attack(
    agent,
    test_attacker,
    battle.opponent.active,
)

print("=" * 70)
print("HYPOTHETICAL FUTURE TURN")
print("=" * 70)

print("Attached Energy:", test_attacker.attached_energy)
print("Chosen Attack:", future_choice["name"])
print("Damage:", future_choice["damage"])
print("Knock Out:", future_choice["can_knock_out"])
print("Score:", future_choice["score"])


# ## Step 19 – Choose Where to Attach Energy
# 
# The AI evaluates the Active Pokémon and all Benched Pokémon to decide where the next Energy should be attached.
# 
# The decision favors Pokémon that are close to unlocking a strong attack, while also considering overall card strength and current HP.

# In[30]:


def minimum_energy_needed_for_next_attack(pokemon: PokemonState):
    """
    Return the smallest number of additional Energy needed
    to unlock one of the Pokémon's attacks.
    """

    attacks = pokemon.card.get("attacks", [])
    energy_gaps = []

    for attack in attacks:
        energy_required = int(attack.get("energy_cost", 0) or 0)

        if energy_required > pokemon.attached_energy:
            energy_gaps.append(
                energy_required - pokemon.attached_energy
            )

    if not energy_gaps:
        return 0

    return min(energy_gaps)


def choose_energy_attachment(
    agent,
    player_state: PlayerState,
):
    """
    Choose the best Pokémon to receive one Energy.
    """

    candidates = [
        player_state.active,
        *player_state.bench,
    ]

    scored_candidates = []

    for pokemon in candidates:
        base_score = agent.evaluate_pokemon(pokemon)

        energy_gap = minimum_energy_needed_for_next_attack(
            pokemon
        )

        score = base_score

        # Strongly reward a Pokémon that is one Energy
        # away from unlocking another attack.
        if energy_gap == 1:
            score += 80

        # Smaller reward when two Energy are still needed.
        elif energy_gap == 2:
            score += 30

        # No immediate attack to unlock.
        elif energy_gap == 0:
            score += 5

        # Slight preference for the Active Pokémon.
        if pokemon.is_active:
            score += 10

        scored_candidates.append({
            "pokemon": pokemon,
            "name": pokemon.card.get(
                "name",
                "Unknown Pokémon",
            ),
            "energy_gap": energy_gap,
            "score": round(score, 2),
        })

    best_candidate = max(
        scored_candidates,
        key=lambda candidate: (
            candidate["score"],
            -candidate["energy_gap"],
            candidate["pokemon"].current_hp,
        ),
    )

    return best_candidate, scored_candidates


# ## Step 20 – Test the Energy Attachment Decision
# 
# The AI compares the Active Pokémon and the Bench, then recommends the best target for the next Energy attachment.

# In[31]:


energy_choice, energy_candidates = choose_energy_attachment(
    agent,
    battle.player,
)

print("=" * 60)
print("ENERGY ATTACHMENT EVALUATION")
print("=" * 60)

for candidate in energy_candidates:
    role = (
        "Active"
        if candidate["pokemon"].is_active
        else "Bench"
    )

    print(
        f"{candidate['name']} ({role})"
    )
    print(
        "Current Energy:",
        candidate["pokemon"].attached_energy,
    )
    print(
        "Energy Needed for Next Attack:",
        candidate["energy_gap"],
    )
    print(
        "Attachment Score:",
        candidate["score"],
    )
    print("-" * 50)

print("\nRecommended Energy Target:")
print(energy_choice["name"])
print("Score:", energy_choice["score"])


# ## Step 21 – Apply the Recommended Energy Attachment
# 
# A copied battle state is used so that the original test battle remains unchanged.

# In[32]:


energy_test_battle = deepcopy(battle)

energy_choice, _ = choose_energy_attachment(
    agent,
    energy_test_battle.player,
)

chosen_pokemon = energy_choice["pokemon"]
energy_before = chosen_pokemon.attached_energy

chosen_pokemon.attached_energy += 1

energy_after = chosen_pokemon.attached_energy

print("=" * 60)
print("ENERGY ATTACHMENT APPLIED")
print("=" * 60)

print("Pokémon:", energy_choice["name"])
print("Energy Before:", energy_before)
print("Energy After:", energy_after)


# ## Step 22 – Evaluate Whether the Active Pokémon Should Retreat
# 
# The AI considers retreating when the Active Pokémon is badly damaged and a healthier or stronger Pokémon is available on the Bench.
# 
# The first retreat policy is intentionally transparent and rule-based.

# In[33]:


def choose_best_bench_pokemon(
    agent,
    bench,
):
    """Return the highest-scoring usable Bench Pokémon."""

    if not bench:
        return None

    available_bench = [
        pokemon
        for pokemon in bench
        if pokemon.current_hp > 0
    ]

    if not available_bench:
        return None

    return max(
        available_bench,
        key=agent.evaluate_pokemon,
    )


def evaluate_retreat_decision(
    agent,
    player_state: PlayerState,
):
    """
    Decide whether the Active Pokémon should retreat.
    """

    active = player_state.active
    best_bench = choose_best_bench_pokemon(
        agent,
        player_state.bench,
    )

    max_hp = float(
        active.card.get("hp", 0) or 0
    )

    if max_hp <= 0:
        hp_ratio = 0
    else:
        hp_ratio = active.current_hp / max_hp

    active_score = agent.evaluate_pokemon(active)

    if best_bench is None:
        return {
            "should_retreat": False,
            "target": None,
            "reason": (
                "No healthy Bench Pokémon is available."
            ),
            "active_hp_ratio": round(hp_ratio, 3),
            "active_score": active_score,
            "bench_score": None,
        }

    bench_score = agent.evaluate_pokemon(best_bench)

    badly_damaged = hp_ratio <= 0.30
    bench_is_stronger = bench_score > active_score + 15

    should_retreat = (
        badly_damaged
        and bench_is_stronger
    )

    if should_retreat:
        reason = (
            f"{active.card['name']} has only "
            f"{hp_ratio:.0%} of its HP remaining, "
            f"while {best_bench.card['name']} has a "
            f"higher evaluation score."
        )
    else:
        reason = (
            f"Retreat is not recommended. "
            f"The Active Pokémon has {hp_ratio:.0%} "
            f"of its HP remaining, or the best Bench "
            f"option is not sufficiently stronger."
        )

    return {
        "should_retreat": should_retreat,
        "target": best_bench,
        "reason": reason,
        "active_hp_ratio": round(hp_ratio, 3),
        "active_score": active_score,
        "bench_score": bench_score,
    }


# ## Step 23 – Test the Retreat Policy
# 
# The policy is first tested on the current battle, then on a hypothetical low-HP scenario.

# In[34]:


current_retreat = evaluate_retreat_decision(
    agent,
    battle.player,
)

print("=" * 60)
print("CURRENT RETREAT DECISION")
print("=" * 60)

print(
    "Should Retreat:",
    current_retreat["should_retreat"],
)

print(
    "Reason:",
    current_retreat["reason"],
)

if current_retreat["target"] is not None:
    print(
        "Best Bench Target:",
        current_retreat["target"].card["name"],
    )


# ## Step 24 – A Single Turn Decision

# In[40]:


def take_turn(self, battle_state: BattleState):
    """
    Complete one AI decision cycle.

    The agent:
    1. Evaluates whether to retreat.
    2. Chooses the best Energy attachment target.
    3. Chooses the best strategic attack.
    """

    retreat_decision = evaluate_retreat_decision(
        self,
        battle_state.player,
    )

    energy_choice, energy_candidates = choose_energy_attachment(
        self,
        battle_state.player,
    )

    strategic_attack = choose_strategic_attack(
        self,
        battle_state.player.active,
        battle_state.opponent.active,
    )

    return {
        "retreat": retreat_decision,
        "energy_target": energy_choice,
        "energy_candidates": energy_candidates,
        "attack": strategic_attack,
    }


# Add the method to the existing AIDecisionAgent class.
AIDecisionAgent.take_turn = take_turn


# Recreate the agent so the notebook clearly uses the updated class.
agent = AIDecisionAgent()


print("Step 24 completed successfully.")
print("Agent has take_turn method:", hasattr(agent, "take_turn"))


# ## Step 25 – Test the Complete AI Turn
# 
# The AI now performs one complete decision cycle by evaluating retreat, Energy attachment, and attack selection.

# In[41]:


decision = agent.take_turn(battle)

print("=" * 70)
print("FULL AI TURN")
print("=" * 70)

print("\n1. RETREAT DECISION")
print("-" * 70)
print("Should Retreat:", decision["retreat"]["should_retreat"])
print("Reason:", decision["retreat"]["reason"])

if decision["retreat"]["target"] is not None:
    print(
        "Best Bench Target:",
        decision["retreat"]["target"].card["name"],
    )
else:
    print("Best Bench Target: None")


print("\n2. ENERGY ATTACHMENT DECISION")
print("-" * 70)
print(
    "Attach Energy To:",
    decision["energy_target"]["name"],
)
print(
    "Current Energy:",
    decision["energy_target"]["pokemon"].attached_energy,
)
print(
    "Energy Needed for Next Attack:",
    decision["energy_target"]["energy_gap"],
)
print(
    "Attachment Score:",
    decision["energy_target"]["score"],
)


print("\n3. ATTACK DECISION")
print("-" * 70)

if decision["attack"] is None:
    print("No legal attack is available.")
else:
    print("Chosen Attack:", decision["attack"]["name"])
    print("Damage:", decision["attack"]["damage"])
    print(
        "Energy Required:",
        decision["attack"]["energy_required"],
    )
    print(
        "Can Knock Out:",
        decision["attack"]["can_knock_out"],
    )
    print(
        "Strategic Score:",
        decision["attack"]["score"],
    )


print("\n" + "=" * 70)
print("AI TURN SUMMARY")
print("=" * 70)

retreat_text = (
    "Retreat"
    if decision["retreat"]["should_retreat"]
    else "Stay Active"
)

energy_text = decision["energy_target"]["name"]

attack_text = (
    decision["attack"]["name"]
    if decision["attack"] is not None
    else "No legal attack"
)

print("Retreat Action:", retreat_text)
print("Energy Target:", energy_text)
print("Attack Action:", attack_text)


# In[ ]:




