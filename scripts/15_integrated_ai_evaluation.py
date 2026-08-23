#!/usr/bin/env python
# coding: utf-8

# # Step 1 — Notebook Introduction
# 
# # Notebook 15 — Integrated AI Evaluation and Benchmarking
# 
# This notebook combines the production components created in earlier notebooks
# into a unified evaluation workflow.
# 
# The notebook will:
# 
# - import the production battle engine,
# - create multiple AI configurations,
# - run controlled AI-vs-AI experiments,
# - compare search optimizations,
# - measure wins, turns, scores, nodes, and runtime,
# - generate benchmark tables,
# - save evaluation reports,
# - prepare the project for the final competition agent.

# # Step 2 — Imports and Project Root

# In[1]:


###############################################################################
# Step 2 — Imports and Project Root
###############################################################################

from __future__ import annotations

import sys
import time

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src import (
    BattleState,
    PlayerState,
    PokemonBattleAgent,
    PokemonState,
    TournamentEntrant,
    TournamentManager,
    TournamentMatch,
    TournamentResult,
    apply_move,
    build_leaderboard_dataframe,
    create_round_robin_schedule,
    current_player_adapter,
    evaluate_state_adapter,
    generate_moves_adapter,
    pokemon_state_features,
    run_two_agent_match,
    terminal_state_adapter,
)

from src.engine.advanced_search import (
    AdvancedSearchEngine,
    ZobristHasher,
)


print("✅ Notebook 15 imports completed.")
print("Project root:", PROJECT_ROOT)


# # Step 3 — Confirm Production Components

# In[2]:


###############################################################################
# Step 3 — Confirm Production Components
###############################################################################

required_components = {
    "BattleState": BattleState,
    "PokemonBattleAgent": PokemonBattleAgent,
    "TournamentEntrant": TournamentEntrant,
    "TournamentManager": TournamentManager,
    "AdvancedSearchEngine": AdvancedSearchEngine,
    "ZobristHasher": ZobristHasher,
    "run_two_agent_match": run_two_agent_match,
    "build_leaderboard_dataframe": build_leaderboard_dataframe,
}

for name, component in required_components.items():
    print(
        f"{name:<32}",
        "available"
        if component is not None
        else "MISSING",
    )

assert TournamentManager.__module__ == "src.tournament.manager"
assert TournamentEntrant.__module__ == "src.tournament.manager"
assert run_two_agent_match.__module__ == "src.tournament.runner"

print()
print("✅ Notebook 15 production components verified.")


# # Step 4 — Create Benchmark Test Cards

# In[3]:


###############################################################################
# Step 4 — Create Benchmark Test Cards
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
                "Utility action used for integration testing."
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
                "Deals 30 damage."
            ),
        },
    ],
}

print("✅ Benchmark test cards created.")


# # Step 4 — Create Benchmark Test Cards

# In[4]:


###############################################################################
# Step 4 — Create Benchmark Test Cards
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
                "Utility action used for integration testing."
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
                "Deals 30 damage."
            ),
        },
    ],
}

print("✅ Benchmark test cards created.")


# # Step 5 — Create a Fresh Benchmark Battle Factory

# In[5]:


###############################################################################
# Step 5 — Create Fresh Benchmark Battle
###############################################################################

def create_benchmark_battle(
    starting_player: str = "Player",
) -> BattleState:
    """Create a fresh battle state for benchmark experiments."""

    if starting_player not in {
        "Player",
        "Opponent",
    }:
        raise ValueError(
            "starting_player must be "
            "'Player' or 'Opponent'."
        )

    player_active = PokemonState(
        card=deepcopy(eevee_ex_card),
        current_hp=200.0,
        attached_energy=2,
        status=None,
        damage=0.0,
        is_active=True,
    )

    opponent_active = PokemonState(
        card=deepcopy(electrike_card),
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

    return BattleState(
        player=player_state,
        opponent=opponent_state,
        turn_number=1,
        current_player=starting_player,
    )


benchmark_state = create_benchmark_battle()

print("✅ Fresh benchmark battle created.")
print(
    "Player:",
    benchmark_state.player.active.card["name"],
)
print(
    "Opponent:",
    benchmark_state.opponent.active.card["name"],
)
print(
    "Starting side:",
    benchmark_state.current_player,
)


# # Step 6 — Create the Stable Move Key

# In[6]:


###############################################################################
# Step 6 — Create Stable Move Key
###############################################################################

def pokemon_move_key(
    move: dict[str, Any],
) -> tuple[Any, ...]:
    """Convert a move dictionary into a stable hashable key."""

    return (
        move.get(
            "name",
            "Unknown Move",
        ),
        float(
            move.get(
                "damage",
                0,
            )
            or 0
        ),
        int(
            move.get(
                "energy_cost",
                0,
            )
            or 0
        ),
    )


test_move_key = pokemon_move_key(
    {
        "name": "Evolution Burst",
        "damage": 60.0,
        "energy_cost": 2,
    }
)

assert test_move_key == (
    "Evolution Burst",
    60.0,
    2,
)

print("✅ Benchmark move key validated:", test_move_key)


# # Step 7 — Create a Configurable Benchmark Agent Factory

# In[7]:


###############################################################################
# Step 7 — Configurable Benchmark Agent Factory
###############################################################################

def create_benchmark_agent(
    *,
    use_transposition_table: bool,
    use_move_ordering: bool,
    use_killer_moves: bool,
    use_history_heuristic: bool,
    seed: int,
) -> PokemonBattleAgent:
    """Create one independently configured benchmark agent."""

    zobrist_hasher = ZobristHasher(
        state_features=pokemon_state_features,
        seed=seed,
    )

    engine = AdvancedSearchEngine(
        generate_moves=generate_moves_adapter,
        apply_move=apply_move,
        evaluate_state=evaluate_state_adapter,
        is_terminal=terminal_state_adapter,
        current_player=current_player_adapter,
        move_to_string=lambda move: move["name"],
        state_key=zobrist_hasher,
        move_key=pokemon_move_key,
        use_transposition_table=use_transposition_table,
        use_move_ordering=use_move_ordering,
        use_killer_moves=use_killer_moves,
        use_history_heuristic=use_history_heuristic,
    )

    return PokemonBattleAgent(engine)


print("✅ Benchmark agent factory created.")


# # Step 8 — Define Benchmark Configurations

# In[9]:


###############################################################################
# Step 8 — Define Benchmark Configurations
###############################################################################

benchmark_configurations = [
    {
        "name": "Baseline",
        "use_transposition_table": False,
        "use_move_ordering": False,
        "use_killer_moves": False,
        "use_history_heuristic": False,
        "seed": 101,
    },
    {
        "name": "Transposition Only",
        "use_transposition_table": True,
        "use_move_ordering": False,
        "use_killer_moves": False,
        "use_history_heuristic": False,
        "seed": 202,
    },
    {
        "name": "Move Ordering Only",
        "use_transposition_table": False,
        "use_move_ordering": True,
        "use_killer_moves": False,
        "use_history_heuristic": False,
        "seed": 303,
    },
    {
        "name": "Fully Optimized",
        "use_transposition_table": True,
        "use_move_ordering": True,
        "use_killer_moves": True,
        "use_history_heuristic": True,
        "seed": 404,
    },
]

print("✅ Benchmark configurations created.")
print("Configurations:", len(benchmark_configurations))

for configuration in benchmark_configurations:
    print("-", configuration["name"])


# # Step 9 — Create the Benchmark Agents

# In[11]:


###############################################################################
# Step 9 — Create Benchmark Agents
###############################################################################

benchmark_agents = {}

for configuration in benchmark_configurations:
    agent = create_benchmark_agent(
        use_transposition_table=configuration[
            "use_transposition_table"
        ],
        use_move_ordering=configuration[
            "use_move_ordering"
        ],
        use_killer_moves=configuration[
            "use_killer_moves"
        ],
        use_history_heuristic=configuration[
            "use_history_heuristic"
        ],
        seed=configuration["seed"],
    )

    benchmark_agents[
        configuration["name"]
    ] = agent

print("✅ Benchmark agents created.")
print("Agents:", len(benchmark_agents))

for name, agent in benchmark_agents.items():
    print(
        f"- {name}: "
        f"{type(agent).__name__} / "
        f"{type(agent.engine).__name__}"
    )


# # Step 10 — Validate Independent Engines

# In[12]:


###############################################################################
# Step 10 — Validate Independent Benchmark Engines
###############################################################################

agent_ids = {
    id(agent)
    for agent in benchmark_agents.values()
}

engine_ids = {
    id(agent.engine)
    for agent in benchmark_agents.values()
}

assert len(agent_ids) == len(benchmark_agents)
assert len(engine_ids) == len(benchmark_agents)

print("✅ Independent benchmark agents validated.")
print("Unique agents:", len(agent_ids))
print("Unique engines:", len(engine_ids))


# # Step 11 — Create a Single Benchmark Match Helper

# In[13]:


###############################################################################
# Step 11 — Single Benchmark Match Helper
###############################################################################

def run_benchmark_match(
    *,
    player_name: str,
    opponent_name: str,
    search_depth: int,
    starting_player: str = "Player",
) -> dict[str, Any]:
    """Run one benchmark match and collect timing information."""

    if player_name not in benchmark_agents:
        raise KeyError(
            f"Unknown benchmark agent: {player_name}"
        )

    if opponent_name not in benchmark_agents:
        raise KeyError(
            f"Unknown benchmark agent: {opponent_name}"
        )

    match = TournamentMatch(
        player_agent=benchmark_agents[player_name],
        opponent_agent=benchmark_agents[opponent_name],
        player_name=player_name,
        opponent_name=opponent_name,
        search_depth=search_depth,
        max_turns=20,
    )

    start_time = time.perf_counter()

    result = run_two_agent_match(
        match=match,
        initial_state=create_benchmark_battle(
            starting_player=starting_player
        ),
        verbose=False,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return {
        "player": player_name,
        "opponent": opponent_name,
        "starting_player": starting_player,
        "search_depth": search_depth,
        "winner": result.winner,
        "turns": result.turns,
        "final_score": result.final_score,
        "elapsed_seconds": elapsed_seconds,
        "transcript_characters": len(result.transcript),
    }


print("✅ Single benchmark match helper created.")


# # Step 12 — Run the First Benchmark Match

# In[14]:


###############################################################################
# Step 12 — Run the First Benchmark Match
###############################################################################

first_benchmark = run_benchmark_match(
    player_name="Fully Optimized",
    opponent_name="Baseline",
    search_depth=6,
    starting_player="Player",
)

first_benchmark


# # Step 13 — Validate the First Benchmark Match

# In[15]:


###############################################################################
# Step 13 — Validate the First Benchmark Match
###############################################################################

assert first_benchmark["player"] == "Fully Optimized"
assert first_benchmark["opponent"] == "Baseline"
assert first_benchmark["search_depth"] == 6

assert first_benchmark["turns"] > 0
assert first_benchmark["elapsed_seconds"] >= 0
assert first_benchmark["transcript_characters"] > 0

assert first_benchmark["winner"] in {
    "Fully Optimized",
    "Baseline",
    "Draw",
    "No winner",
}

print("✅ First benchmark match validated.")


# # Step 14 — Define the Benchmark Matrix

# In[16]:


###############################################################################
# Step 14 — Define the Benchmark Matrix
###############################################################################

benchmark_depths = [
    2,
    4,
    6,
    8,
]

benchmark_pairings = [
    (
        "Fully Optimized",
        "Baseline",
    ),
    (
        "Baseline",
        "Fully Optimized",
    ),
    (
        "Transposition Only",
        "Move Ordering Only",
    ),
    (
        "Move Ordering Only",
        "Transposition Only",
    ),
]

print("✅ Benchmark matrix created.")
print("Depths:", benchmark_depths)
print("Pairings:", len(benchmark_pairings))


# # Step 15 — Run the Benchmark Matrix

# In[17]:


###############################################################################
# Step 15 — Run the Benchmark Matrix
###############################################################################

benchmark_rows = []

benchmark_number = 1

for search_depth in benchmark_depths:
    for player_name, opponent_name in benchmark_pairings:
        benchmark_record = run_benchmark_match(
            player_name=player_name,
            opponent_name=opponent_name,
            search_depth=search_depth,
            starting_player="Player",
        )

        benchmark_record[
            "benchmark_number"
        ] = benchmark_number

        benchmark_rows.append(
            benchmark_record
        )

        print(
            f"Benchmark {benchmark_number:>2} | "
            f"Depth {search_depth:>2} | "
            f"{player_name} vs {opponent_name} | "
            f"Winner: {benchmark_record['winner']} | "
            f"Turns: {benchmark_record['turns']} | "
            f"Time: "
            f"{benchmark_record['elapsed_seconds']:.6f}s"
        )

        benchmark_number += 1

print()
print("✅ Benchmark matrix completed.")
print("Benchmark runs:", len(benchmark_rows))


# # Step 16 — Create the Benchmark DataFrame

# In[19]:


###############################################################################
# Step 16 — Create the Benchmark DataFrame
###############################################################################

benchmark_df = pd.DataFrame(
    benchmark_rows
)

benchmark_df = benchmark_df[
    [
        "benchmark_number",
        "player",
        "opponent",
        "starting_player",
        "search_depth",
        "winner",
        "turns",
        "final_score",
        "elapsed_seconds",
        "transcript_characters",
    ]
]

benchmark_df


# # Step 17 — Validate the Benchmark DataFrame

# In[20]:


###############################################################################
# Step 17 — Validate the Benchmark DataFrame
###############################################################################

assert len(benchmark_df) == 16

assert set(
    benchmark_df["search_depth"]
) == {
    2,
    4,
    6,
    8,
}

assert benchmark_df["benchmark_number"].is_unique

assert all(
    benchmark_df["turns"] > 0
)

assert all(
    benchmark_df["elapsed_seconds"] >= 0
)

assert all(
    benchmark_df["transcript_characters"] > 0
)

assert all(
    benchmark_df["winner"].isin(
        {
            "Fully Optimized",
            "Baseline",
            "Transposition Only",
            "Move Ordering Only",
            "Draw",
            "No winner",
        }
    )
)

print("✅ Benchmark DataFrame validated.")
print("Rows:", len(benchmark_df))


# # Step 18 — Build Depth-Level Performance Summary

# In[21]:


###############################################################################
# Step 18 — Build Depth-Level Performance Summary
###############################################################################

depth_summary_df = (
    benchmark_df
    .groupby(
        "search_depth",
        as_index=False,
    )
    .agg(
        matches=(
            "benchmark_number",
            "count",
        ),
        average_turns=(
            "turns",
            "mean",
        ),
        average_score=(
            "final_score",
            "mean",
        ),
        average_elapsed_seconds=(
            "elapsed_seconds",
            "mean",
        ),
        minimum_elapsed_seconds=(
            "elapsed_seconds",
            "min",
        ),
        maximum_elapsed_seconds=(
            "elapsed_seconds",
            "max",
        ),
    )
)

depth_summary_df


# # Step 19 — Validate the Depth Summary

# In[22]:


###############################################################################
# Step 19 — Validate the Depth Summary
###############################################################################

assert len(depth_summary_df) == 4

assert depth_summary_df[
    "matches"
].tolist() == [
    4,
    4,
    4,
    4,
]

assert set(
    depth_summary_df["search_depth"]
) == {
    2,
    4,
    6,
    8,
}

assert all(
    depth_summary_df[
        "average_elapsed_seconds"
    ] >= 0
)

print("✅ Depth-level performance summary validated.")


# # Step 20 — Display a Compact Benchmark Report

# In[24]:


###############################################################################
# Step 20 — Display the Benchmark Report
###############################################################################

print("=" * 90)
print("INTEGRATED AI BENCHMARK REPORT")
print("=" * 90)

print()

print(
    depth_summary_df.to_string(
        index=False,
    )
)

print()

print("=" * 90)
print("INDIVIDUAL MATCH RESULTS")
print("=" * 90)

print(
    benchmark_df[
        [
            "benchmark_number",
            "player",
            "opponent",
            "search_depth",
            "winner",
            "turns",
            "final_score",
            "elapsed_seconds",
        ]
    ].to_string(
        index=False,
    )
)


# # Step 21 — Build Overall Benchmark Statistics

# In[25]:


###############################################################################
# Step 21 — Build Overall Benchmark Statistics
###############################################################################

overall_summary = {
    "benchmark_runs": len(benchmark_df),
    "search_depths": sorted(
        benchmark_df["search_depth"].unique().tolist()
    ),
    "average_turns": benchmark_df["turns"].mean(),
    "average_score": benchmark_df["final_score"].mean(),
    "average_runtime": benchmark_df["elapsed_seconds"].mean(),
    "minimum_runtime": benchmark_df["elapsed_seconds"].min(),
    "maximum_runtime": benchmark_df["elapsed_seconds"].max(),
}

overall_summary


# # Step 22 — Validate Overall Statistics

# In[26]:


###############################################################################
# Step 22 — Validate Overall Statistics
###############################################################################

assert overall_summary["benchmark_runs"] == 16

assert overall_summary["search_depths"] == [
    2,
    4,
    6,
    8,
]

assert overall_summary["average_turns"] > 0
assert overall_summary["average_score"] > 0
assert overall_summary["average_runtime"] >= 0

print("✅ Overall benchmark statistics validated.")


# # Step 23 — Save Benchmark CSV

# In[27]:


###############################################################################
# Step 23 — Save Benchmark CSV
###############################################################################

OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

benchmark_csv_path = (
    OUTPUT_DIR
    / "notebook15_benchmark.csv"
)

benchmark_df.to_csv(
    benchmark_csv_path,
    index=False,
)

print("✅ Benchmark CSV saved.")
print("CSV:", benchmark_csv_path)
print(
    "Rows:",
    len(benchmark_df),
)


# # Step 24 — Save Benchmark Summary (JSON)

# In[28]:


###############################################################################
# Step 24 — Save Benchmark Summary
###############################################################################

import json

benchmark_summary_path = (
    OUTPUT_DIR
    / "notebook15_summary.json"
)

benchmark_summary_path.write_text(
    json.dumps(
        overall_summary,
        indent=4,
    ),
    encoding="utf-8",
)

print("✅ Benchmark summary saved.")
print("JSON:", benchmark_summary_path)


# # Step 25 — Save the Benchmark Report (TXT)

# In[29]:


###############################################################################
# Step 25 — Save Benchmark Report
###############################################################################

benchmark_report_path = (
    OUTPUT_DIR
    / "notebook15_report.txt"
)

report_lines = [
    "=" * 90,
    "INTEGRATED AI BENCHMARK REPORT",
    "=" * 90,
    "",
    f"Benchmark runs: {overall_summary['benchmark_runs']}",
    f"Average turns: {overall_summary['average_turns']:.2f}",
    f"Average score: {overall_summary['average_score']:.2f}",
    (
        "Average runtime: "
        f"{overall_summary['average_runtime']:.6f}"
    ),
    "",
    "DEPTH SUMMARY",
    "",
    depth_summary_df.to_string(index=False),
    "",
    "=" * 90,
    "",
    "INDIVIDUAL MATCH RESULTS",
    "",
    benchmark_df.to_string(index=False),
]

benchmark_report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("✅ Benchmark report saved.")
print("TXT:", benchmark_report_path)
print(
    "Size:",
    benchmark_report_path.stat().st_size,
    "bytes",
)


# # Step 26 — Verify Saved Reports

# In[30]:


###############################################################################
# Step 26 — Verify Notebook 15 Reports
###############################################################################

assert benchmark_csv_path.exists()
assert benchmark_summary_path.exists()
assert benchmark_report_path.exists()

assert benchmark_csv_path.stat().st_size > 0
assert benchmark_summary_path.stat().st_size > 0
assert benchmark_report_path.stat().st_size > 0

saved_benchmark_df = pd.read_csv(
    benchmark_csv_path
)

saved_summary = json.loads(
    benchmark_summary_path.read_text(
        encoding="utf-8"
    )
)

saved_report = benchmark_report_path.read_text(
    encoding="utf-8"
)

assert len(saved_benchmark_df) == 16
assert saved_summary["benchmark_runs"] == 16

assert (
    "INTEGRATED AI BENCHMARK REPORT"
    in saved_report
)

print("✅ Notebook 15 reports validated.")
print(
    "Benchmark rows:",
    len(saved_benchmark_df),
)
print(
    "Benchmark runs:",
    saved_summary["benchmark_runs"],
)
print(
    "Report characters:",
    len(saved_report),
)


# # Step 27 — Final Notebook 15 Validation

# In[32]:


###############################################################################
# Step 27 — Final Notebook 15 Validation
###############################################################################

assert len(benchmark_df) == 16
assert len(depth_summary_df) == 4

assert overall_summary["benchmark_runs"] == 16

assert benchmark_csv_path.exists()
assert benchmark_summary_path.exists()
assert benchmark_report_path.exists()

assert all(
    benchmark_df["elapsed_seconds"] >= 0
)

assert all(
    benchmark_df["turns"] > 0
)

print("=" * 90)
print("✅ NOTEBOOK 15 FINAL VALIDATION PASSED")
print("=" * 90)

print("Benchmark runs:", len(benchmark_df))
print("Depths:", benchmark_depths)
print("Reports:", OUTPUT_DIR)


# In[ ]:




