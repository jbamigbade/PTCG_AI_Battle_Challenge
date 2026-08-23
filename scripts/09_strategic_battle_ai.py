#!/usr/bin/env python
# coding: utf-8

# # Notebook 09 – Strategic Battle AI
# 
# This notebook extends the tactical AI from Notebook 08 into a multi-turn strategic planner.
# 
# The strategic agent will:
# 
# - project future Energy attachments,
# - estimate when stronger attacks become available,
# - compare immediate damage with future payoff,
# - identify likely Knock Out sequences,
# - evaluate short planning horizons,
# - and explain why a multi-turn plan was selected.
# 
# The goal is to move from one-turn decision-making toward limited look-ahead planning.

# ## Step 1 – Import Libraries and Locate Project Artifacts
# 
# Notebook 09 loads the same project artifacts used in Notebook 08.
# 
# The existing Card Knowledge Base, generated decks, and baseline battle results provide the foundation for strategic planning.

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
    Locate the project, notebooks, and processed-data directories.
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
# The strategic planner uses the same saved card, deck, and simulation artifacts created in the previous notebooks.

# In[2]:


def load_pickle_file(file_path):
    """Load and return an object stored in a pickle file."""

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


# ## Step 3 – Recreate the Battle State Classes
# 
# Notebook 09 recreates the lightweight battle-state objects used by the AI.
# 
# These objects make it possible to copy and project hypothetical future game states without changing the real current state.

# In[3]:


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
    """Complete battle state used for strategic planning."""

    player: PlayerState
    opponent: PlayerState
    turn_number: int
    current_player: str


# ## Step 4 – Build the Starting Strategic Test State
# 
# A controlled test battle is created using Pokémon from the generated decks.
# 
# This state will be copied repeatedly so that the planner can explore future turns without altering the original battle.

# In[4]:


def get_pokemon_cards(deck, count):
    """Return the requested number of Pokémon cards."""

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
    count=2,
)


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

player_state = PlayerState(
    active=player_active,
    bench=player_bench,
)


opponent_active = PokemonState(
    card=lightning_pokemon[0],
    current_hp=float(lightning_pokemon[0]["hp"]),
    attached_energy=1,
    is_active=True,
)

opponent_bench = [
    PokemonState(
        card=lightning_pokemon[1],
        current_hp=float(lightning_pokemon[1]["hp"]),
    )
]

opponent_state = PlayerState(
    active=opponent_active,
    bench=opponent_bench,
)


strategic_battle = BattleState(
    player=player_state,
    opponent=opponent_state,
    turn_number=1,
    current_player="Player 1",
)


print("Strategic battle state created successfully.\n")

print(
    "Player Active:",
    strategic_battle.player.active.card["name"],
)

print(
    "Player Energy:",
    strategic_battle.player.active.attached_energy,
)

print(
    "Player Bench:",
    [
        pokemon.card["name"]
        for pokemon in strategic_battle.player.bench
    ],
)

print()

print(
    "Opponent Active:",
    strategic_battle.opponent.active.card["name"],
)

print(
    "Opponent HP:",
    strategic_battle.opponent.active.current_hp,
)

print(
    "Opponent Bench:",
    [
        pokemon.card["name"]
        for pokemon in strategic_battle.opponent.bench
    ],
)


# ## Step 5 – Create the Strategic Planning Agent
# 
# The Strategic AI extends the heuristic decision engine by projecting several turns into the future.
# 
# Instead of evaluating only the current turn, it estimates:
# 
# - future Energy attachments,
# - future attack availability,
# - future knockouts,
# - and long-term board value.

# In[5]:


class StrategicBattlePlanner:
    """
    Multi-turn strategic planner.

    This planner evaluates how the current battle may evolve over the
    next few turns instead of considering only the current move.
    """

    def __init__(self, planning_horizon=3):

        self.planning_horizon = planning_horizon

    def simulate_future_energy(
        self,
        pokemon: PokemonState,
        turns=3,
    ):
        """
        Simulate future Energy attachments.
        """

        future = []

        current_energy = pokemon.attached_energy

        for turn in range(1, turns + 1):

            current_energy += 1

            future.append(
                {
                    "turn": turn,
                    "energy": current_energy,
                }
            )

        return future


# ## Step 6 – Project Future Energy Growth
# 
# Estimate how much Energy will be attached over the next several turns.

# In[6]:


planner = StrategicBattlePlanner(
    planning_horizon=3
)

projection = planner.simulate_future_energy(
    strategic_battle.player.active,
    turns=3,
)

print("=" * 60)
print("FUTURE ENERGY PROJECTION")
print("=" * 60)

print()

print(
    "Pokemon:",
    strategic_battle.player.active.card["name"],
)

print(
    "Current Energy:",
    strategic_battle.player.active.attached_energy,
)

print()

for state in projection:

    print(
        f"Turn +{state['turn']}: "
        f"{state['energy']} Energy"
    )


# ## Step 7 – Forecast Future Attack Availability
# 
# The planner now estimates when each attack becomes usable as Energy is attached over future turns.
# 
# This helps the AI compare immediate actions with delayed, higher-damage options.

# In[7]:


def forecast_attack_availability(
    planner: StrategicBattlePlanner,
    pokemon: PokemonState,
):
    """
    Forecast the first future turn on which each attack becomes usable.
    """

    attacks = pokemon.card.get("attacks", [])
    current_energy = pokemon.attached_energy

    forecast = []

    for attack in attacks:
        attack_name = attack.get("Move Name")
        energy_required = int(
            attack.get("energy_cost", 0) or 0
        )
        damage = float(
            attack.get("damage_numeric", 0) or 0
        )

        if current_energy >= energy_required:
            first_usable_turn = 0
        else:
            energy_gap = energy_required - current_energy

            if energy_gap <= planner.planning_horizon:
                first_usable_turn = energy_gap
            else:
                first_usable_turn = None

        forecast.append({
            "name": attack_name,
            "damage": damage,
            "energy_required": energy_required,
            "current_energy": current_energy,
            "first_usable_turn": first_usable_turn,
            "usable_now": first_usable_turn == 0,
            "reachable_within_horizon": (
                first_usable_turn is not None
            ),
        })

    return forecast


# ## Step 8 – Test the Future Attack Forecast
# 
# Display when each attack becomes available within the planner's current look-ahead horizon.

# In[8]:


attack_forecast = forecast_attack_availability(
    planner,
    strategic_battle.player.active,
)

print("=" * 70)
print("FUTURE ATTACK AVAILABILITY")
print("=" * 70)

print(
    "Pokémon:",
    strategic_battle.player.active.card["name"],
)

print(
    "Current Energy:",
    strategic_battle.player.active.attached_energy,
)

print(
    "Planning Horizon:",
    planner.planning_horizon,
    "turns",
)

print()

for attack in attack_forecast:
    print("Attack:", attack["name"])
    print("Damage:", attack["damage"])
    print(
        "Energy Required:",
        attack["energy_required"],
    )

    if attack["usable_now"]:
        availability = "Usable now"
    elif attack["first_usable_turn"] is not None:
        availability = (
            f"Usable in {attack['first_usable_turn']} "
            f"future turn(s)"
        )
    else:
        availability = (
            "Not reachable within planning horizon"
        )

    print("Availability:", availability)
    print("-" * 55)


# ## Step 9 – Estimate Future Knockout Timing
# 
# The planner now combines attack availability with the opponent's current HP.
# 
# This identifies the earliest future turn on which the Active Pokémon could produce a Knock Out.

# In[10]:


def forecast_knockout_opportunities(
    planner: StrategicBattlePlanner,
    attacker: PokemonState,
    defender: PokemonState,
):
    """
    Forecast the earliest turn on which each attack could
    Knock Out the defender.
    """

    attack_forecast = forecast_attack_availability(
        planner,
        attacker,
    )

    defender_hp = float(defender.current_hp)

    knockout_forecast = []

    for attack in attack_forecast:
        can_knock_out = (
            attack["damage"] >= defender_hp
        )

        if (
            can_knock_out
            and attack["first_usable_turn"] is not None
        ):
            knockout_turn = attack["first_usable_turn"]
        else:
            knockout_turn = None

        knockout_forecast.append({
            **attack,
            "defender_hp": defender_hp,
            "can_knock_out": can_knock_out,
            "knockout_turn": knockout_turn,
        })

    return knockout_forecast


# ## Step 10 – Test Future Knockout Forecasting
# 
# The planner identifies whether a legal or future attack can Knock Out the opponent and how many turns of preparation are required.

# In[11]:


knockout_forecast = forecast_knockout_opportunities(
    planner,
    strategic_battle.player.active,
    strategic_battle.opponent.active,
)

print("=" * 70)
print("FUTURE KNOCKOUT FORECAST")
print("=" * 70)

print(
    "Attacker:",
    strategic_battle.player.active.card["name"],
)

print(
    "Defender:",
    strategic_battle.opponent.active.card["name"],
)

print(
    "Defender HP:",
    strategic_battle.opponent.active.current_hp,
)

print()

for attack in knockout_forecast:
    print("Attack:", attack["name"])
    print("Damage:", attack["damage"])
    print(
        "Can Knock Out:",
        attack["can_knock_out"],
    )

    if attack["knockout_turn"] == 0:
        timing = "Knock Out available now"
    elif attack["knockout_turn"] is not None:
        timing = (
            f"Knock Out available in "
            f"{attack['knockout_turn']} future turn(s)"
        )
    else:
        timing = (
            "No projected Knock Out within horizon"
        )

    print("Timing:", timing)
    print("-" * 55)


# ## Step 11 – Evaluate Future Prize Rewards
# 
# Future attacks are now scored according to the Prize Cards they could earn.
# 
# This allows the planner to favor attacks that immediately win prizes over attacks that only deal damage.

# In[12]:


def evaluate_future_prize_value(
    planner: StrategicBattlePlanner,
    attacker: PokemonState,
    defender: PokemonState,
):
    """
    Estimate the strategic reward of future knockouts.
    """

    forecast = forecast_knockout_opportunities(
        planner,
        attacker,
        defender,
    )

    prize_values = []

    for attack in forecast:

        prize_reward = 0

        if attack["can_knock_out"]:

            prize_reward = 300

            if attack["knockout_turn"] is not None:
                prize_reward -= attack["knockout_turn"] * 20

        prize_values.append({

            **attack,

            "prize_reward": prize_reward,

            "strategic_value":
                attack["damage"] + prize_reward

        })

    return prize_values



# ## Step 12 – Test Future Prize Forecast
# 
# Display how valuable each attack becomes after considering Prize Cards.

# In[13]:


future_prizes = evaluate_future_prize_value(
    planner,
    strategic_battle.player.active,
    strategic_battle.opponent.active,
)

print("=" * 70)
print("FUTURE PRIZE FORECAST")
print("=" * 70)

for attack in future_prizes:

    print("Attack:", attack["name"])

    print(
        "Can Knock Out:",
        attack["can_knock_out"]
    )

    print(
        "Prize Reward:",
        attack["prize_reward"]
    )

    print(
        "Strategic Value:",
        attack["strategic_value"]
    )

    print("-" * 55)


# ## Step 13 – Score Multi-Turn Attack Plans
# 
# The planner now evaluates complete attack plans rather than isolated attacks.
# 
# Each plan considers:
# 
# - damage,
# - Prize Card value,
# - number of turns required,
# - Energy preparation,
# - and delayed-action penalties.
# 
# Higher scores represent stronger long-term plans.

# In[14]:


def score_multi_turn_attack_plans(
    planner: StrategicBattlePlanner,
    attacker: PokemonState,
    defender: PokemonState,
):
    """
    Score every attack as a multi-turn strategic plan.
    """

    future_values = evaluate_future_prize_value(
        planner,
        attacker,
        defender,
    )

    plans = []

    for attack in future_values:
        usable_turn = attack["first_usable_turn"]

        if usable_turn is None:
            delay_penalty = 1000
            reachable = False
        else:
            delay_penalty = usable_turn * 25
            reachable = True

        energy_preparation = max(
            attack["energy_required"]
            - attacker.attached_energy,
            0,
        )

        preparation_penalty = energy_preparation * 10

        plan_score = (
            attack["strategic_value"]
            - delay_penalty
            - preparation_penalty
        )

        plans.append({
            **attack,
            "reachable": reachable,
            "energy_preparation": energy_preparation,
            "delay_penalty": delay_penalty,
            "preparation_penalty": preparation_penalty,
            "plan_score": round(plan_score, 2),
        })

    return plans


# ## Step 14 – Test Multi-Turn Plan Scoring
# 
# Display the complete strategic score for each projected attack plan.

# In[15]:


multi_turn_plans = score_multi_turn_attack_plans(
    planner,
    strategic_battle.player.active,
    strategic_battle.opponent.active,
)

print("=" * 70)
print("MULTI-TURN ATTACK PLAN SCORES")
print("=" * 70)

for plan in multi_turn_plans:
    print("Attack:", plan["name"])
    print("Damage:", plan["damage"])
    print("Reachable:", plan["reachable"])
    print(
        "First Usable Turn:",
        plan["first_usable_turn"],
    )
    print(
        "Energy Preparation Needed:",
        plan["energy_preparation"],
    )
    print(
        "Prize Reward:",
        plan["prize_reward"],
    )
    print(
        "Delay Penalty:",
        plan["delay_penalty"],
    )
    print(
        "Preparation Penalty:",
        plan["preparation_penalty"],
    )
    print(
        "Final Plan Score:",
        plan["plan_score"],
    )
    print("-" * 55)


# ## Step 15 – Choose the Best Long-Term Attack Plan
# 
# The planner compares all reachable plans and selects the strongest long-term objective.

# In[16]:


def choose_best_multi_turn_plan(
    planner: StrategicBattlePlanner,
    attacker: PokemonState,
    defender: PokemonState,
):
    """
    Return the highest-scoring reachable multi-turn plan.
    """

    plans = score_multi_turn_attack_plans(
        planner,
        attacker,
        defender,
    )

    reachable_plans = [
        plan
        for plan in plans
        if plan["reachable"]
    ]

    if not reachable_plans:
        return None

    return max(
        reachable_plans,
        key=lambda plan: (
            plan["plan_score"],
            plan["can_knock_out"],
            plan["damage"],
            -plan["first_usable_turn"],
        ),
    )


# ## Step 16 – Test the Best Long-Term Plan
# 
# The planner now identifies the strongest strategic objective over the current planning horizon.

# In[17]:


best_long_term_plan = choose_best_multi_turn_plan(
    planner,
    strategic_battle.player.active,
    strategic_battle.opponent.active,
)

print("=" * 70)
print("BEST LONG-TERM PLAN")
print("=" * 70)

if best_long_term_plan is None:
    print("No reachable strategic plan was found.")
else:
    print(
        "Pokémon:",
        strategic_battle.player.active.card["name"],
    )
    print(
        "Target:",
        strategic_battle.opponent.active.card["name"],
    )
    print(
        "Planned Attack:",
        best_long_term_plan["name"],
    )
    print(
        "Damage:",
        best_long_term_plan["damage"],
    )
    print(
        "Energy Required:",
        best_long_term_plan["energy_required"],
    )
    print(
        "Additional Energy Needed:",
        best_long_term_plan["energy_preparation"],
    )
    print(
        "First Usable Turn:",
        best_long_term_plan["first_usable_turn"],
    )
    print(
        "Can Knock Out:",
        best_long_term_plan["can_knock_out"],
    )
    print(
        "Prize Reward:",
        best_long_term_plan["prize_reward"],
    )
    print(
        "Final Plan Score:",
        best_long_term_plan["plan_score"],
    )


# ## Step 17 – Explain the Multi-Turn Strategy
# 
# The planner produces a readable explanation showing why the long-term plan was selected.

# In[18]:


def explain_multi_turn_plan(
    attacker: PokemonState,
    defender: PokemonState,
    plan,
):
    """
    Create a readable explanation of a strategic plan.
    """

    attacker_name = attacker.card.get(
        "name",
        "Unknown Pokémon",
    )

    defender_name = defender.card.get(
        "name",
        "Unknown Pokémon",
    )

    if plan is None:
        return (
            f"No reachable strategic attack plan was found "
            f"for {attacker_name}."
        )

    explanation = [
        f"{attacker_name} should prepare to use "
        f"{plan['name']} against {defender_name}.",
        f"The attack deals {plan['damage']} damage.",
        f"It requires {plan['energy_required']} Energy.",
        f"{attacker_name} currently has "
        f"{attacker.attached_energy} Energy attached.",
        f"The plan therefore requires "
        f"{plan['energy_preparation']} additional "
        f"Energy attachment(s).",
    ]

    if plan["first_usable_turn"] == 0:
        explanation.append(
            "The attack is available immediately."
        )
    else:
        explanation.append(
            f"The attack is projected to become usable "
            f"in {plan['first_usable_turn']} future turn(s)."
        )

    if plan["can_knock_out"]:
        explanation.append(
            f"It can Knock Out {defender_name} and "
            f"earn the projected Prize reward."
        )
    else:
        explanation.append(
            f"It does not currently guarantee a Knock Out."
        )

    explanation.append(
        f"The final strategic plan score is "
        f"{plan['plan_score']}."
    )

    return " ".join(explanation)


strategic_explanation = explain_multi_turn_plan(
    strategic_battle.player.active,
    strategic_battle.opponent.active,
    best_long_term_plan,
)

print("=" * 70)
print("STRATEGIC PLAN EXPLANATION")
print("=" * 70)
print(strategic_explanation)


# ## Step 18 – Evaluate Bench Pokémon Strategies
# 
# The planner now compares the Active Pokémon with each Benched Pokémon.
# 
# This allows the AI to determine whether continuing with the current Active Pokémon is better than developing or switching to a Bench Pokémon.

# In[19]:


def evaluate_pokemon_long_term_plan(
    planner: StrategicBattlePlanner,
    pokemon: PokemonState,
    defender: PokemonState,
):
    """
    Evaluate the best long-term attack plan for one Pokémon.
    """

    best_plan = choose_best_multi_turn_plan(
        planner,
        pokemon,
        defender,
    )

    if best_plan is None:
        return {
            "pokemon": pokemon,
            "name": pokemon.card.get("name", "Unknown Pokémon"),
            "best_plan": None,
            "plan_score": -1000.0,
            "first_usable_turn": None,
            "can_knock_out": False,
        }

    return {
        "pokemon": pokemon,
        "name": pokemon.card.get("name", "Unknown Pokémon"),
        "best_plan": best_plan,
        "plan_score": best_plan["plan_score"],
        "first_usable_turn": best_plan["first_usable_turn"],
        "can_knock_out": best_plan["can_knock_out"],
    }


def evaluate_all_player_pokemon(
    planner: StrategicBattlePlanner,
    player_state: PlayerState,
    defender: PokemonState,
):
    """
    Evaluate the Active Pokémon and every Benched Pokémon.
    """

    candidates = [
        player_state.active,
        *player_state.bench,
    ]

    evaluations = []

    for pokemon in candidates:
        evaluation = evaluate_pokemon_long_term_plan(
            planner,
            pokemon,
            defender,
        )

        evaluation["role"] = (
            "Active"
            if pokemon.is_active
            else "Bench"
        )

        evaluations.append(evaluation)

    return evaluations


# ## Step 19 – Compare Active and Bench Strategies
# 
# Display the best projected attack plan for each Pokémon currently in play.

# In[20]:


pokemon_strategy_evaluations = evaluate_all_player_pokemon(
    planner,
    strategic_battle.player,
    strategic_battle.opponent.active,
)

print("=" * 70)
print("ACTIVE AND BENCH STRATEGY COMPARISON")
print("=" * 70)

for evaluation in pokemon_strategy_evaluations:
    print(
        f"{evaluation['name']} "
        f"({evaluation['role']})"
    )

    if evaluation["best_plan"] is None:
        print("Best Plan: None")
        print("Plan Score:", evaluation["plan_score"])
    else:
        print(
            "Best Planned Attack:",
            evaluation["best_plan"]["name"],
        )
        print(
            "First Usable Turn:",
            evaluation["first_usable_turn"],
        )
        print(
            "Can Knock Out:",
            evaluation["can_knock_out"],
        )
        print(
            "Plan Score:",
            evaluation["plan_score"],
        )

    print("-" * 55)


# ## Step 20 – Choose the Best Strategic Pokémon
# 
# The planner selects the Pokémon with the strongest projected multi-turn plan.
# 
# A small switching penalty is applied to Bench Pokémon because moving them into the Active position may require retreat resources or a future turn.

# In[21]:


def choose_best_strategic_pokemon(
    planner: StrategicBattlePlanner,
    player_state: PlayerState,
    defender: PokemonState,
    switching_penalty=40,
):
    """
    Choose the Pokémon with the strongest long-term plan.
    """

    evaluations = evaluate_all_player_pokemon(
        planner,
        player_state,
        defender,
    )

    adjusted_evaluations = []

    for evaluation in evaluations:
        adjusted_score = evaluation["plan_score"]

        if evaluation["role"] == "Bench":
            adjusted_score -= switching_penalty

        adjusted_evaluation = {
            **evaluation,
            "switching_penalty": (
                switching_penalty
                if evaluation["role"] == "Bench"
                else 0
            ),
            "adjusted_score": round(
                adjusted_score,
                2,
            ),
        }

        adjusted_evaluations.append(
            adjusted_evaluation
        )

    best_choice = max(
        adjusted_evaluations,
        key=lambda evaluation: (
            evaluation["adjusted_score"],
            evaluation["can_knock_out"],
            -(
                evaluation["first_usable_turn"]
                if evaluation["first_usable_turn"]
                is not None
                else 999
            ),
        ),
    )

    return best_choice, adjusted_evaluations


# ## Step 21 – Test Strategic Pokémon Selection
# 
# The planner compares all Pokémon after accounting for the cost of switching from the Active position.

# In[22]:


best_strategic_pokemon, strategic_candidates = (
    choose_best_strategic_pokemon(
        planner,
        strategic_battle.player,
        strategic_battle.opponent.active,
        switching_penalty=40,
    )
)

print("=" * 70)
print("STRATEGIC POKÉMON SELECTION")
print("=" * 70)

for candidate in strategic_candidates:
    print(
        f"{candidate['name']} "
        f"({candidate['role']})"
    )
    print(
        "Raw Plan Score:",
        candidate["plan_score"],
    )
    print(
        "Switching Penalty:",
        candidate["switching_penalty"],
    )
    print(
        "Adjusted Score:",
        candidate["adjusted_score"],
    )

    if candidate["best_plan"] is not None:
        print(
            "Planned Attack:",
            candidate["best_plan"]["name"],
        )

    print("-" * 55)

print("\nRecommended Strategic Pokémon:")
print(best_strategic_pokemon["name"])
print("Role:", best_strategic_pokemon["role"])
print(
    "Adjusted Score:",
    best_strategic_pokemon["adjusted_score"],
)


# ## Step 22 – Decide Whether to Switch Pokémon
# 
# The planner recommends switching only when a Benched Pokémon offers a meaningfully stronger long-term plan than the current Active Pokémon.

# In[23]:


def evaluate_strategic_switch(
    planner: StrategicBattlePlanner,
    player_state: PlayerState,
    defender: PokemonState,
    switching_penalty=40,
    minimum_improvement=25,
):
    """
    Decide whether switching to a Bench Pokémon is strategically justified.
    """

    best_choice, candidates = choose_best_strategic_pokemon(
        planner,
        player_state,
        defender,
        switching_penalty=switching_penalty,
    )

    active_evaluation = next(
        candidate
        for candidate in candidates
        if candidate["role"] == "Active"
    )

    improvement = (
        best_choice["adjusted_score"]
        - active_evaluation["adjusted_score"]
    )

    should_switch = (
        best_choice["role"] == "Bench"
        and improvement >= minimum_improvement
    )

    if should_switch:
        reason = (
            f"Switch to {best_choice['name']} because its "
            f"adjusted strategic score is "
            f"{improvement:.2f} points higher than the "
            f"current Active Pokémon."
        )
    else:
        reason = (
            f"Keep {active_evaluation['name']} Active. "
            f"No Bench Pokémon improves the strategic "
            f"outlook by at least {minimum_improvement} points."
        )

    return {
        "should_switch": should_switch,
        "target": (
            best_choice["pokemon"]
            if should_switch
            else None
        ),
        "best_choice": best_choice,
        "active_evaluation": active_evaluation,
        "improvement": round(improvement, 2),
        "reason": reason,
        "candidates": candidates,
    }


# ## Step 23 – Test the Strategic Switch Decision
# 
# The planner now decides whether the long-term benefit of a Bench Pokémon is large enough to justify switching.

# In[24]:


strategic_switch = evaluate_strategic_switch(
    planner,
    strategic_battle.player,
    strategic_battle.opponent.active,
    switching_penalty=40,
    minimum_improvement=25,
)

print("=" * 70)
print("STRATEGIC SWITCH DECISION")
print("=" * 70)

print(
    "Should Switch:",
    strategic_switch["should_switch"],
)

print(
    "Improvement:",
    strategic_switch["improvement"],
)

print(
    "Reason:",
    strategic_switch["reason"],
)

if strategic_switch["target"] is not None:
    print(
        "Recommended Switch Target:",
        strategic_switch["target"].card["name"],
    )
else:
    print("Recommended Switch Target: None")


# # Step 24 – Create a Battle Tree Search
# 
# Instead of evaluating only one attack, the AI now explores
# multiple future turns.
# 
# Each possible future becomes a branch in the battle tree.

# In[25]:


from dataclasses import dataclass

@dataclass
class BattleNode:
    battle_state: BattleState
    depth: int
    cumulative_score: float
    action_taken: str


# # Step 25 – Simulate Future Battle States
# 
# Each action creates a brand-new Battle State.
# 
# The original battle remains untouched.

# In[26]:


def simulate_future_state(
    battle_state,
    damage,
):

    future = deepcopy(battle_state)

    opponent = future.opponent.active

    opponent.current_hp = max(
        0,
        opponent.current_hp - damage,
    )

    return future


# # Step 26 – Evaluate Future Battle States

# In[31]:


def evaluate_future_state(future_battle: BattleState):
    """
    Score a future battle position from the player's perspective.

    Positive scores favor the player.
    Negative scores favor the opponent.
    """

    player_active = future_battle.player.active
    opponent_active = future_battle.opponent.active

    player_max_hp = float(
        player_active.card.get("hp", 0) or 0
    )

    opponent_max_hp = float(
        opponent_active.card.get("hp", 0) or 0
    )

    player_hp_ratio = (
        player_active.current_hp / player_max_hp
        if player_max_hp > 0
        else 0
    )

    opponent_hp_ratio = (
        opponent_active.current_hp / opponent_max_hp
        if opponent_max_hp > 0
        else 0
    )

    score = 0.0

    # Reward preserving the player's Active Pokémon.
    score += player_hp_ratio * 100

    # Reward damaging the opponent.
    score += (1 - opponent_hp_ratio) * 100

    # Large reward for a Knock Out.
    if opponent_active.current_hp <= 0:
        score += 500

    # Large penalty if the player's Active Pokémon is Knocked Out.
    if player_active.current_hp <= 0:
        score -= 500

    return round(score, 2)


# # Step 27 – Build the First Search Layer

# In[32]:


def get_current_legal_attacks(
    pokemon: PokemonState,
):
    """
    Return attacks that are usable with the Pokémon's
    currently attached Energy.
    """

    legal_attacks = []

    for attack in pokemon.card.get("attacks", []):
        attack_name = attack.get(
            "Move Name",
            "Unknown Move",
        )

        damage = float(
            attack.get("damage_numeric", 0) or 0
        )

        energy_required = int(
            attack.get("energy_cost", 0) or 0
        )

        usable = (
            pokemon.attached_energy
            >= energy_required
        )

        if usable:
            legal_attacks.append({
                "name": attack_name,
                "damage": damage,
                "energy_required": energy_required,
            })

    return legal_attacks


def simulate_future_state(
    battle_state: BattleState,
    damage: float,
):
    """
    Create a copied battle state and apply attack damage
    to the opponent's Active Pokémon.
    """

    future_state = deepcopy(battle_state)

    future_opponent = (
        future_state.opponent.active
    )

    future_opponent.current_hp = max(
        0.0,
        future_opponent.current_hp - damage,
    )

    future_state.turn_number += 1

    return future_state


def generate_child_nodes(
    battle_state: BattleState,
):
    """
    Generate one child node for every currently legal attack.
    """

    legal_attacks = get_current_legal_attacks(
        battle_state.player.active
    )

    children = []

    for attack in legal_attacks:
        future_state = simulate_future_state(
            battle_state,
            attack["damage"],
        )

        future_score = evaluate_future_state(
            future_state
        )

        child_node = BattleNode(
            battle_state=future_state,
            depth=1,
            cumulative_score=future_score,
            action_taken=attack["name"],
        )

        children.append(child_node)

    return children


# # Step 28 – Test Battle Tree Generation

# In[33]:


battle_tree = generate_child_nodes(
    strategic_battle
)

print("=" * 70)
print("FIRST BATTLE TREE")
print("=" * 70)

print(
    "Active Pokémon:",
    strategic_battle.player.active.card["name"],
)

print(
    "Attached Energy:",
    strategic_battle.player.active.attached_energy,
)

print(
    "Legal Branches Generated:",
    len(battle_tree),
)

print()

if not battle_tree:
    print("No legal attack branches were generated.")

else:
    for node in battle_tree:
        print("Action:", node.action_taken)
        print("Tree Depth:", node.depth)
        print(
            "Future Position Score:",
            node.cumulative_score,
        )
        print(
            "Opponent Remaining HP:",
            node.battle_state.opponent.active.current_hp,
        )
        print(
            "Future Turn Number:",
            node.battle_state.turn_number,
        )
        print("-" * 55)


# ## Optional legality test

# In[34]:


three_energy_battle = deepcopy(
    strategic_battle
)

three_energy_battle.player.active.attached_energy = 3

three_energy_tree = generate_child_nodes(
    three_energy_battle
)

print("=" * 70)
print("THREE-ENERGY BATTLE TREE")
print("=" * 70)

for node in three_energy_tree:
    print("Action:", node.action_taken)
    print(
        "Opponent Remaining HP:",
        node.battle_state.opponent.active.current_hp,
    )
    print(
        "Future Position Score:",
        node.cumulative_score,
    )
    print("-" * 55)


# # Step 29 – Expand the Battle Tree to Depth Two
# 
# Each first-layer child now generates its own children.
# 
# The AI begins evaluating move sequences instead of single moves.

# In[35]:


def expand_tree_depth_two(root_children):
    """
    Expand every first-layer node into a second layer.
    """

    expanded_tree = []

    for child in root_children:

        second_layer = generate_child_nodes(
            child.battle_state
        )

        expanded_tree.append({
            "parent": child,
            "children": second_layer
        })

    return expanded_tree


# # Step 30 – Test the Depth-Two Battle Tree
# 
# Expand every first-level branch into another layer and
# display the resulting search tree.

# In[36]:


depth_two_tree = expand_tree_depth_two(
    battle_tree
)

print("=" * 70)
print("DEPTH-TWO BATTLE TREE")
print("=" * 70)

print()

for branch in depth_two_tree:

    parent = branch["parent"]

    print(f"FIRST MOVE: {parent.action_taken}")
    print("-" * 55)

    children = branch["children"]

    if len(children) == 0:
        print("No legal second moves.")
        print()
        continue

    for child in children:

        print("Second Move:", child.action_taken)
        print("Depth:", child.depth)
        print("Score:", child.cumulative_score)

        print(
            "Opponent HP:",
            child.battle_state.opponent.active.current_hp
        )

        print()

    print("=" * 70)


# # Step 31 – Score Complete Move Sequences
# 
# Instead of evaluating isolated future positions, the AI now
# scores the entire sequence of planned actions.
# 
# The cumulative score becomes the total value of a strategic line.

# In[37]:


def score_move_sequence(parent_node, child_node):
    """
    Score an entire two-turn sequence.
    """

    return (
        parent_node.cumulative_score
        + child_node.cumulative_score
    )


# # Step 32 – Evaluate Every Strategic Line

# In[38]:


sequence_scores = []

for branch in depth_two_tree:

    parent = branch["parent"]

    for child in branch["children"]:

        total = score_move_sequence(
            parent,
            child,
        )

        sequence_scores.append({

            "first_move": parent.action_taken,

            "second_move": child.action_taken,

            "score": total,

            "final_hp":
                child.battle_state
                .opponent
                .active
                .current_hp

        })


# # Step 33 – Display the Best Two-Turn Lines

# In[39]:


print("=" * 70)
print("TWO-TURN STRATEGIC LINES")
print("=" * 70)

for line in sequence_scores:

    print(
        "Sequence:",
        f"{line['first_move']} -> {line['second_move']}"
    )

    print(
        "Combined Score:",
        line["score"]
    )

    print(
        "Opponent HP:",
        line["final_hp"]
    )

    print("-" * 55)


# # Step 34 – Rank the Two-Turn Strategic Lines
# 
# The planner now sorts all two-turn action sequences from strongest to weakest.
# 
# This makes it possible to identify the best projected line instead of only listing every available sequence.

# In[40]:


ranked_sequence_scores = sorted(
    sequence_scores,
    key=lambda line: (
        line["score"],
        -line["final_hp"],
    ),
    reverse=True,
)

print("=" * 70)
print("RANKED TWO-TURN STRATEGIC LINES")
print("=" * 70)

for rank, line in enumerate(
    ranked_sequence_scores,
    start=1,
):
    print(f"Rank {rank}")
    print(
        "Sequence:",
        f"{line['first_move']} -> {line['second_move']}",
    )
    print("Combined Score:", line["score"])
    print("Opponent Remaining HP:", line["final_hp"])
    print("-" * 55)


# # Step 35 – Select the Best Two-Turn Strategic Line
# 
# The planner selects the highest-ranked action sequence as its preferred two-turn plan.

# In[41]:


def choose_best_two_turn_line(sequence_scores):
    """
    Return the strongest two-turn action sequence.
    """

    if not sequence_scores:
        return None

    return max(
        sequence_scores,
        key=lambda line: (
            line["score"],
            -line["final_hp"],
        ),
    )


best_two_turn_line = choose_best_two_turn_line(
    sequence_scores
)

print("=" * 70)
print("BEST TWO-TURN STRATEGIC LINE")
print("=" * 70)

if best_two_turn_line is None:
    print("No two-turn strategic line was available.")
else:
    print(
        "First Move:",
        best_two_turn_line["first_move"],
    )
    print(
        "Second Move:",
        best_two_turn_line["second_move"],
    )
    print(
        "Combined Score:",
        best_two_turn_line["score"],
    )
    print(
        "Opponent Remaining HP:",
        best_two_turn_line["final_hp"],
    )


# # Step 36 – Simulate Energy Growth Between Turns
# 
# A future Pokémon turn normally includes one Energy attachment.
# 
# Before generating the next layer of the search tree, the copied battle state receives one additional Energy on the player's Active Pokémon.
# 
# This allows stronger attacks to become available in later branches.

# In[42]:


def prepare_next_player_turn(
    battle_state: BattleState,
):
    """
    Create a copied state for the player's next turn
    and attach one Energy to the Active Pokémon.
    """

    future_state = deepcopy(battle_state)

    future_state.player.active.attached_energy += 1
    future_state.turn_number += 1

    return future_state


# # Step 37 – Build an Energy-Aware Depth-Two Tree
# 
# The second search layer now represents a later turn after another Energy attachment.
# 
# This lets the planner discover attacks that become legal over time.

# In[43]:


def expand_energy_aware_depth_two(
    root_children,
):
    """
    Expand first-layer nodes after simulating one future
    Energy attachment.
    """

    expanded_tree = []

    for parent in root_children:
        prepared_state = prepare_next_player_turn(
            parent.battle_state
        )

        second_layer = generate_child_nodes(
            prepared_state
        )

        # Correct the displayed depth.
        for child in second_layer:
            child.depth = 2

        expanded_tree.append({
            "parent": parent,
            "prepared_state": prepared_state,
            "children": second_layer,
        })

    return expanded_tree


energy_aware_tree = expand_energy_aware_depth_two(
    battle_tree
)

print("=" * 70)
print("ENERGY-AWARE DEPTH-TWO TREE")
print("=" * 70)

for branch in energy_aware_tree:
    parent = branch["parent"]
    prepared_state = branch["prepared_state"]

    print("First Move:", parent.action_taken)
    print(
        "Energy Before Second Move:",
        prepared_state.player.active.attached_energy,
    )

    if not branch["children"]:
        print("No legal second moves.")
    else:
        for child in branch["children"]:
            print("Second Move:", child.action_taken)
            print("Depth:", child.depth)
            print(
                "Opponent Remaining HP:",
                child.battle_state.opponent.active.current_hp,
            )
            print(
                "Future Position Score:",
                child.cumulative_score,
            )
            print("-" * 45)

    print("=" * 70)


# # Step 38 – Build a Three-Turn Energy-Aware Search
# 
# The planner now advances far enough for Eevee ex to reach three attached Energy.
# 
# At the third player turn, Coruscating Quartz should become a legal branch.

# In[44]:


def build_three_turn_sequences(
    battle_state: BattleState,
):
    """
    Generate all legal three-turn player action sequences
    with one Energy attached before each later turn.
    """

    sequences = []

    first_layer = generate_child_nodes(
        battle_state
    )

    for first_node in first_layer:
        second_turn_state = prepare_next_player_turn(
            first_node.battle_state
        )

        second_layer = generate_child_nodes(
            second_turn_state
        )

        for second_node in second_layer:
            third_turn_state = prepare_next_player_turn(
                second_node.battle_state
            )

            third_layer = generate_child_nodes(
                third_turn_state
            )

            for third_node in third_layer:
                total_score = (
                    first_node.cumulative_score
                    + second_node.cumulative_score
                    + third_node.cumulative_score
                )

                sequences.append({
                    "first_move": first_node.action_taken,
                    "second_move": second_node.action_taken,
                    "third_move": third_node.action_taken,
                    "total_score": round(total_score, 2),
                    "final_opponent_hp": (
                        third_node
                        .battle_state
                        .opponent
                        .active
                        .current_hp
                    ),
                    "final_energy": (
                        third_node
                        .battle_state
                        .player
                        .active
                        .attached_energy
                    ),
                })

    return sequences


three_turn_sequences = build_three_turn_sequences(
    strategic_battle
)

print("=" * 70)
print("THREE-TURN ENERGY-AWARE SEQUENCES")
print("=" * 70)

print(
    "Number of Sequences:",
    len(three_turn_sequences),
)

print()

for sequence in three_turn_sequences:
    print(
        "Sequence:",
        f"{sequence['first_move']} -> "
        f"{sequence['second_move']} -> "
        f"{sequence['third_move']}",
    )
    print("Total Score:", sequence["total_score"])
    print(
        "Final Opponent HP:",
        sequence["final_opponent_hp"],
    )
    print("Final Energy:", sequence["final_energy"])
    print("-" * 55)


# In[ ]:




