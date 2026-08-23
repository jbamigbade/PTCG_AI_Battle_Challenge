#!/usr/bin/env python
# coding: utf-8

# # Notebook 12 — Integrated Pokémon AI Battle Agent
# 
# This notebook connects the Pokémon battle-state modules with the advanced
# search engine created in Notebook 11.
# 
# The completed system will:
# 
# - inspect the current battle state,
# - generate legal moves,
# - search future battle positions,
# - select the strongest available action,
# - explain its decision,
# - apply the selected move,
# - and simulate complete AI-versus-AI battles.

# ## Step 2 — Import the Project Modules

# In[1]:


from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import (
    BattleState,
    PlayerState,
    PokemonState,
    apply_move,
    current_player_adapter,
    evaluate_position,
    evaluate_state_adapter,
    generate_moves_adapter,
    get_current_legal_moves,
    get_legal_moves,
    pokemon_state_features,
    terminal_state_adapter,
)

from src.engine import (
    AdvancedSearchEngine,
    SearchResult,
    SearchStats,
    SearchTimeLimit,
    ZobristHasher,
)


print("✅ Notebook 12 imports completed successfully.")
print("Project root:", PROJECT_ROOT)


# ## Step 3 — Confirm the Required Components

# In[2]:


required_components = {
    "BattleState": BattleState,
    "PlayerState": PlayerState,
    "PokemonState": PokemonState,
    "AdvancedSearchEngine": AdvancedSearchEngine,
    "ZobristHasher": ZobristHasher,
    "apply_move": apply_move,
    "generate_moves_adapter": generate_moves_adapter,
    "evaluate_state_adapter": evaluate_state_adapter,
    "terminal_state_adapter": terminal_state_adapter,
    "current_player_adapter": current_player_adapter,
    "pokemon_state_features": pokemon_state_features,
}

for name, component in required_components.items():
    print(
        f"{name:<28}",
        "available" if component is not None else "MISSING",
    )


# ## Step 4 — Create the Agent Result Object

# In[3]:


###############################################################################
# Step 4 — Agent Decision
###############################################################################

@dataclass
class AgentDecision:
    """
    Represents one AI decision.
    """

    move: dict
    score: float
    search_depth: int
    nodes: int
    principal_variation: list = field(default_factory=list)

    def summary(self):

        return {
            "move": self.move["name"],
            "score": self.score,
            "depth": self.search_depth,
            "nodes": self.nodes,
            "pv": [
                m["name"]
                for m in self.principal_variation
            ],
        }


print("✅ AgentDecision created.")


# ## Step 5 — Build the PokémonBattleAgent

# In[4]:


###############################################################################
# Step 5 — Pokémon Battle Agent
###############################################################################

class PokemonBattleAgent:

    def __init__(self, engine):
        self.engine = engine

    def choose_move(
        self,
        state,
        depth=6,
    ):
        result = self.engine.search_depth(
            state=state,
            depth=depth,
            clear_table=True,
        )

        if result.best_move is None:
            raise RuntimeError(
                "The search engine did not return a legal move."
            )

        return AgentDecision(
            move=result.best_move,
            score=result.score,
            search_depth=result.completed_depth,
            nodes=result.stats.nodes,
            principal_variation=result.principal_variation,
        )


# ## Step 6 — Create the Fully Optimized Search Engine

# In[5]:


###############################################################################
# Step 6 — Create the Fully Optimized Search Engine
###############################################################################

def pokemon_move_key(move: dict):
    """
    Return a stable, hashable key for a Pokémon move dictionary.
    """

    return (
        move.get("name", "Unknown Move"),
        float(move.get("damage", 0) or 0),
        int(move.get("energy_cost", 0) or 0),
    )


pokemon_zobrist = ZobristHasher(
    state_features=pokemon_state_features,
    seed=20260714,
)


battle_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],

    state_key=pokemon_zobrist,
    move_key=pokemon_move_key,

    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=True,
    use_history_heuristic=True,
)


battle_agent = PokemonBattleAgent(
    battle_engine
)

print("✅ Battle agent ready.")
print(
    "Move key test:",
    pokemon_move_key(
        {
            "name": "Evolution Burst",
            "damage": 60.0,
            "energy_cost": 2,
        }
    ),
)


# ## Step 7 — Create a Self-Contained Test Battle

# In[6]:


###############################################################################
# Step 7 — Create a Self-Contained Test Battle
###############################################################################

eevee_ex_card = {
    "name": "Eevee ex",
    "hp": 200,
    "attacks": [
        {
            "Move Name": "Tera",
            "damage_numeric": 0,
            "energy_cost": 0,
            "Effect Explanation": (
                "A zero-damage utility action used for integration testing."
            ),
        },
        {
            "Move Name": "Evolution Burst",
            "damage_numeric": 60,
            "energy_cost": 2,
            "Effect Explanation": (
                "Deals 60 damage to the opposing Active Pokémon."
            ),
        },
    ],
}

electrike_card = {
    "name": "Electrike",
    "hp": 70,
    "attacks": [
        {
            "Move Name": "Thunder Jolt",
            "damage_numeric": 30,
            "energy_cost": 1,
            "Effect Explanation": (
                "This Pokémon also does 10 damage to itself."
            ),
        },
    ],
}


player_active = PokemonState(
    card=eevee_ex_card,
    current_hp=200.0,
    attached_energy=2,
    status=None,
    damage=0.0,
    is_active=True,
)

opponent_active = PokemonState(
    card=electrike_card,
    current_hp=70.0,
    attached_energy=1,
    status=None,
    damage=0.0,
    is_active=True,
)


player_state = PlayerState(
    active=player_active,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)

opponent_state = PlayerState(
    active=opponent_active,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


battle_state = BattleState(
    player=player_state,
    opponent=opponent_state,
    turn_number=1,
    current_player="Player",
)

print("✅ Notebook 12 test battle created.")
print("Player:", battle_state.player.active.card["name"])
print("Player HP:", battle_state.player.active.current_hp)
print("Player Energy:", battle_state.player.active.attached_energy)
print("Opponent:", battle_state.opponent.active.card["name"])
print("Opponent HP:", battle_state.opponent.active.current_hp)
print("Opponent Energy:", battle_state.opponent.active.attached_energy)
print("Current side:", battle_state.current_player)


# ## Step 8 — Validate Legal Moves

# In[7]:


###############################################################################
# Step 8 — Validate Legal Moves
###############################################################################

legal_moves = generate_moves_adapter(
    battle_state
)

print("Current side:", battle_state.current_player)
print("Legal move count:", len(legal_moves))
print()

for move in legal_moves:
    print(
        f"{move['name']:<20} "
        f"Damage: {move['damage']:<6} "
        f"Energy: {move['energy_cost']}"
    )


# In[8]:


assert battle_state.current_player == "Player"
assert len(legal_moves) == 2

assert {
    move["name"]
    for move in legal_moves
} == {
    "Tera",
    "Evolution Burst",
}

print("✅ Step 8 legal-move validation passed.")


# ## Step 9 — Ask the Agent to Choose a Move

# In[9]:


###############################################################################
# Step 9 — Ask the Agent to Choose a Move
###############################################################################

decision = battle_agent.choose_move(
    state=battle_state,
    depth=10,
)

decision.summary()


# ## Step 10 — Validate the Agent Decision

# In[10]:


###############################################################################
# Step 10 — Validate the Agent Decision
###############################################################################

decision_summary = decision.summary()

assert decision.move is not None
assert decision.move["name"] == "Evolution Burst"
assert decision.search_depth == 10
assert decision.nodes > 0
assert len(decision.principal_variation) > 0

assert (
    decision.move
    == decision.principal_variation[0]
)

print("✅ Agent decision validation passed.")
print()
print("Chosen move:", decision.move["name"])
print("Score:", decision.score)
print("Completed depth:", decision.search_depth)
print("Nodes:", decision.nodes)
print(
    "Principal variation:",
    " → ".join(
        move["name"]
        for move in decision.principal_variation
    ),
)


# # Part 3 — Apply the Agent’s Decision
# ## Step 11 — Apply the Chosen Move

# In[11]:


###############################################################################
# Step 11 — Apply the Chosen Move
###############################################################################

future_battle_state = apply_move(
    battle_state,
    decision.move,
)

print("✅ Chosen move applied.")
print()
print("Move:", decision.move["name"])
print("Original turn:", battle_state.turn_number)
print("Future turn:", future_battle_state.turn_number)
print("Original current side:", battle_state.current_player)
print("Future current side:", future_battle_state.current_player)
print("Original opponent HP:", battle_state.opponent.active.current_hp)
print("Future opponent HP:", future_battle_state.opponent.active.current_hp)


# # Step 12 — Validate State Independence

# In[12]:


###############################################################################
# Step 12 — Validate State Independence
###############################################################################

assert future_battle_state is not battle_state

assert battle_state.turn_number == 1
assert future_battle_state.turn_number == 2

assert battle_state.current_player == "Player"
assert future_battle_state.current_player == "Opponent"

assert battle_state.opponent.active.current_hp == 70.0
assert future_battle_state.opponent.active.current_hp == 10.0

assert battle_state.player.active.current_hp == 200.0
assert future_battle_state.player.active.current_hp == 200.0

print("✅ State-independence validation passed.")


# ## Step 13 — Ask the Opponent Agent to Respond

# In[13]:


###############################################################################
# Step 13 — Ask the Opponent Agent to Respond
###############################################################################

opponent_decision = battle_agent.choose_move(
    state=future_battle_state,
    depth=10,
)

opponent_decision.summary()


# ## Step 14 — Apply the Opponent Move

# In[14]:


###############################################################################
# Step 14 — Apply the Opponent Move
###############################################################################

second_future_state = apply_move(
    future_battle_state,
    opponent_decision.move,
)

print("✅ Opponent move applied.")
print()
print("Move:", opponent_decision.move["name"])
print("Turn number:", second_future_state.turn_number)
print("Current side:", second_future_state.current_player)
print("Player HP:", second_future_state.player.active.current_hp)
print("Opponent HP:", second_future_state.opponent.active.current_hp)


# # Part 4 — Run a Complete AI-vs-AI Battle
# ## Step 15 — Create a Battle Turn Record

# In[15]:


###############################################################################
# Step 15 — Battle Turn Record
###############################################################################

@dataclass
class BattleTurnRecord:
    turn_number: int
    acting_side: str
    pokemon_name: str
    move_name: str
    damage: float
    score: float
    search_depth: int
    nodes: int
    player_hp_after: float
    opponent_hp_after: float
    next_side: str

    def as_dict(self):
        return {
            "turn_number": self.turn_number,
            "acting_side": self.acting_side,
            "pokemon_name": self.pokemon_name,
            "move_name": self.move_name,
            "damage": self.damage,
            "score": self.score,
            "search_depth": self.search_depth,
            "nodes": self.nodes,
            "player_hp_after": self.player_hp_after,
            "opponent_hp_after": self.opponent_hp_after,
            "next_side": self.next_side,
        }


print("✅ BattleTurnRecord created.")


# # Step 16 — Create a Battle Simulation Result

# In[16]:


###############################################################################
# Step 16 — Battle Simulation Result
###############################################################################

@dataclass
class BattleSimulationResult:
    final_state: BattleState
    turns: list[BattleTurnRecord] = field(default_factory=list)
    winner: Optional[str] = None
    stop_reason: str = ""

    @property
    def turn_count(self):
        return len(self.turns)

    def as_dict(self):
        return {
            "winner": self.winner,
            "stop_reason": self.stop_reason,
            "turn_count": self.turn_count,
            "player_hp": self.final_state.player.active.current_hp,
            "opponent_hp": self.final_state.opponent.active.current_hp,
            "turns": [
                turn.as_dict()
                for turn in self.turns
            ],
        }


print("✅ BattleSimulationResult created.")


# # Step 17 — Create the Winner Detector

# In[17]:


###############################################################################
# Step 17 — Winner Detector
###############################################################################

def determine_battle_winner(state: BattleState):

    player_hp = state.player.active.current_hp
    opponent_hp = state.opponent.active.current_hp

    if player_hp <= 0 and opponent_hp <= 0:
        return "Draw"

    if opponent_hp <= 0:
        return "Player"

    if player_hp <= 0:
        return "Opponent"

    if state.player.prize_cards_remaining <= 0:
        return "Player"

    if state.opponent.prize_cards_remaining <= 0:
        return "Opponent"

    return None


print("Winner before knockout:", determine_battle_winner(battle_state))


# # Step 18 — Build the Full AI-vs-AI Simulation Function

# In[18]:


###############################################################################
# Step 18 — Full AI-vs-AI Simulation
###############################################################################

def simulate_ai_battle(
    initial_state: BattleState,
    agent: PokemonBattleAgent,
    search_depth: int = 6,
    max_turns: int = 20,
    verbose: bool = True,
) -> BattleSimulationResult:

    current_state = deepcopy(initial_state)
    turn_records = []

    for _ in range(max_turns):

        winner = determine_battle_winner(current_state)

        if winner is not None:
            return BattleSimulationResult(
                final_state=current_state,
                turns=turn_records,
                winner=winner,
                stop_reason="Battle reached a terminal state.",
            )

        acting_side = current_state.current_player

        if acting_side == "Player":
            active_pokemon = current_state.player.active
        elif acting_side == "Opponent":
            active_pokemon = current_state.opponent.active
        else:
            raise ValueError(
                "current_player must be 'Player' or 'Opponent'."
            )

        decision = agent.choose_move(
            state=current_state,
            depth=search_depth,
        )

        next_state = apply_move(
            current_state,
            decision.move,
        )

        record = BattleTurnRecord(
            turn_number=current_state.turn_number,
            acting_side=acting_side,
            pokemon_name=active_pokemon.card["name"],
            move_name=decision.move["name"],
            damage=float(
                decision.move.get("damage", 0) or 0
            ),
            score=decision.score,
            search_depth=decision.search_depth,
            nodes=decision.nodes,
            player_hp_after=next_state.player.active.current_hp,
            opponent_hp_after=next_state.opponent.active.current_hp,
            next_side=next_state.current_player,
        )

        turn_records.append(record)

        if verbose:
            print(
                f"Turn {record.turn_number}: "
                f"{record.acting_side} — "
                f"{record.pokemon_name} used "
                f"{record.move_name}"
            )
            print(
                f"  Player HP: {record.player_hp_after} | "
                f"Opponent HP: {record.opponent_hp_after}"
            )

        current_state = next_state

    winner = determine_battle_winner(current_state)

    return BattleSimulationResult(
        final_state=current_state,
        turns=turn_records,
        winner=winner,
        stop_reason=(
            "Maximum turn limit reached."
            if winner is None
            else "Battle reached a terminal state."
        ),
    )


print("✅ AI battle simulator created.")


# # Step 19 — Run the First Complete AI-vs-AI Battle

# In[19]:


###############################################################################
# Step 19 — Run the First Complete AI-vs-AI Battle
###############################################################################

simulation_result = simulate_ai_battle(
    initial_state=battle_state,
    agent=battle_agent,
    search_depth=6,
    max_turns=10,
    verbose=True,
)


# # Step 20 — Inspect the Battle Result

# In[20]:


###############################################################################
# Step 20 — Inspect the Battle Result
###############################################################################

simulation_result.as_dict()


# # Step 21 — Validate the Complete Battle

# In[21]:


###############################################################################
# Step 21 — Validate the Complete Battle
###############################################################################

assert simulation_result.winner == "Player"

assert (
    simulation_result.stop_reason
    == "Battle reached a terminal state."
)

assert simulation_result.turn_count == 3

assert (
    simulation_result.final_state.player.active.current_hp
    == 170.0
)

assert (
    simulation_result.final_state.opponent.active.current_hp
    == 0.0
)

assert [
    turn.move_name
    for turn in simulation_result.turns
] == [
    "Evolution Burst",
    "Thunder Jolt",
    "Evolution Burst",
]

assert [
    turn.acting_side
    for turn in simulation_result.turns
] == [
    "Player",
    "Opponent",
    "Player",
]

print("✅ Complete AI battle validation passed.")
print()
print("Winner:", simulation_result.winner)
print("Turns played:", simulation_result.turn_count)
print(
    "Final Player HP:",
    simulation_result.final_state.player.active.current_hp,
)
print(
    "Final Opponent HP:",
    simulation_result.final_state.opponent.active.current_hp,
)


# # Part 5 — Record and Analyze Battle History

# In[22]:


###############################################################################
# Step 22 — Convert Battle History to a DataFrame
###############################################################################

import pandas as pd


battle_history_rows = [
    turn.as_dict()
    for turn in simulation_result.turns
]

battle_history_df = pd.DataFrame(
    battle_history_rows
)

battle_history_df


# # Step 23 — Validate the Battle History Table

# In[23]:


###############################################################################
# Step 23 — Validate the Battle History Table
###############################################################################

expected_columns = [
    "turn_number",
    "acting_side",
    "pokemon_name",
    "move_name",
    "damage",
    "score",
    "search_depth",
    "nodes",
    "player_hp_after",
    "opponent_hp_after",
    "next_side",
]

assert list(battle_history_df.columns) == expected_columns
assert len(battle_history_df) == 3

assert battle_history_df["move_name"].tolist() == [
    "Evolution Burst",
    "Thunder Jolt",
    "Evolution Burst",
]

assert battle_history_df["acting_side"].tolist() == [
    "Player",
    "Opponent",
    "Player",
]

assert battle_history_df["player_hp_after"].tolist() == [
    200.0,
    170.0,
    170.0,
]

assert battle_history_df["opponent_hp_after"].tolist() == [
    10.0,
    10.0,
    0.0,
]

print("✅ Battle-history table validation passed.")


# # Step 24 — Create a Readable Battle Transcript

# In[24]:


###############################################################################
# Step 24 — Create a Readable Battle Transcript
###############################################################################

def create_battle_transcript(
    simulation: BattleSimulationResult,
) -> str:

    lines = []

    lines.append("=" * 70)
    lines.append("POKÉMON AI BATTLE TRANSCRIPT")
    lines.append("=" * 70)

    for turn in simulation.turns:

        lines.append(
            f"Turn {turn.turn_number}: "
            f"{turn.acting_side} — "
            f"{turn.pokemon_name} used "
            f"{turn.move_name}"
        )

        lines.append(
            f"  Damage: {turn.damage:.1f}"
        )

        lines.append(
            f"  Search score: {turn.score:.2f}"
        )

        lines.append(
            f"  Search depth: {turn.search_depth}"
        )

        lines.append(
            f"  Nodes searched: {turn.nodes}"
        )

        lines.append(
            f"  HP after move — "
            f"Player: {turn.player_hp_after:.1f}, "
            f"Opponent: {turn.opponent_hp_after:.1f}"
        )

        lines.append(
            f"  Next side: {turn.next_side}"
        )

        lines.append("-" * 70)

    lines.append(
        f"Winner: {simulation.winner}"
    )

    lines.append(
        f"Stop reason: {simulation.stop_reason}"
    )

    return "\n".join(lines)


battle_transcript = create_battle_transcript(
    simulation_result
)

print(battle_transcript)


# ## Step 25 — Validate the Transcript

# In[25]:


###############################################################################
# Step 25 — Validate the Transcript
###############################################################################

assert "POKÉMON AI BATTLE TRANSCRIPT" in battle_transcript
assert "Turn 1: Player" in battle_transcript
assert "Evolution Burst" in battle_transcript
assert "Thunder Jolt" in battle_transcript
assert "Winner: Player" in battle_transcript

print("✅ Battle transcript validation passed.")


# # Step 26 — Save the Battle History

# In[26]:


###############################################################################
# Recovery Cell — Rebuild Battle Output Variables
###############################################################################

import pandas as pd

battle_history_rows = [
    turn.as_dict()
    for turn in simulation_result.turns
]

battle_history_df = pd.DataFrame(
    battle_history_rows
)

battle_transcript = create_battle_transcript(
    simulation_result
)

print("✅ Battle output variables rebuilt.")
print("History rows:", len(battle_history_df))
print("Transcript characters:", len(battle_transcript))


# In[28]:


###############################################################################
# Step 26B — Save the Rebuilt Battle History
###############################################################################

from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

history_csv_path = (
    OUTPUT_DIR
    / "notebook12_battle_history.csv"
)

transcript_path = (
    OUTPUT_DIR
    / "notebook12_battle_transcript.txt"
)

battle_history_df.to_csv(
    history_csv_path,
    index=False,
)

transcript_path.write_text(
    battle_transcript,
    encoding="utf-8",
)

print("✅ Battle output files saved.")
print("History CSV:", history_csv_path)
print("Transcript:", transcript_path)
print("History size:", history_csv_path.stat().st_size, "bytes")
print("Transcript size:", transcript_path.stat().st_size, "bytes")


# ## Step 27 — Verify the Saved Files

# In[29]:


###############################################################################
# Step 27 — Verify Saved Files
###############################################################################

assert history_csv_path.exists()
assert transcript_path.exists()

assert history_csv_path.stat().st_size > 0
assert transcript_path.stat().st_size > 0

saved_history_df = pd.read_csv(
    history_csv_path
)

saved_transcript = transcript_path.read_text(
    encoding="utf-8"
)

assert len(saved_history_df) == 3
assert "Winner: Player" in saved_transcript

print("✅ Saved battle files validated.")
print("Saved history rows:", len(saved_history_df))
print("Transcript characters:", len(saved_transcript))


# In[ ]:




