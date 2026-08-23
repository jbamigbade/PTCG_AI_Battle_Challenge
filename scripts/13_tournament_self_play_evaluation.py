#!/usr/bin/env python
# coding: utf-8

# # Notebook 13 — Tournament Framework and Self-Play Evaluation
# 
# This notebook builds a reusable evaluation framework for repeatedly running
# AI-versus-AI Pokémon battles.
# 
# The framework will:
# 
# - run complete matches,
# - repeat matches across multiple configurations,
# - compare agents and search depths,
# - calculate win rates and performance metrics,
# - store structured match records,
# - export tournament reports for later analysis.

# ## Part 1 — Setup
# Step 2 — Imports and Project Path

# In[1]:


###############################################################################
# Step 2 — Imports and Project Path
###############################################################################

from __future__ import annotations

import sys
import time

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


###############################################################################
# Notebook 13 — Imports from Production Source Modules
###############################################################################

from src import (
    BattleSimulationResult,
    BattleState,
    BattleTurnRecord,
    PlayerState,
    PokemonBattleAgent,
    PokemonState,
    TournamentMatch,
    TournamentResult,
    TournamentStatistics,
    apply_move,
    create_battle_transcript,
    current_player_adapter,
    determine_battle_winner,
    evaluate_state_adapter,
    generate_moves_adapter,
    pokemon_state_features,
    run_two_agent_match,
    simulate_ai_battle,
    terminal_state_adapter,
    update_statistics,
)

from src.engine.advanced_search import (
    AdvancedSearchEngine,
    ZobristHasher,
)

print("✅ Notebook 13 imports completed.")
print("Project root:", PROJECT_ROOT)


# ## Step 3 — Confirm Required Components

# In[2]:


###############################################################################
# Step 3 — Confirm Required Components
###############################################################################

required_components = {
    "AdvancedSearchEngine": AdvancedSearchEngine,
    "PokemonBattleAgent": PokemonBattleAgent,
    "BattleSimulationResult": BattleSimulationResult,
    "BattleState": BattleState,
    "PokemonState": PokemonState,
    "PlayerState": PlayerState,
    "ZobristHasher": ZobristHasher,
    "simulate_ai_battle": simulate_ai_battle,
    "generate_moves_adapter": generate_moves_adapter,
    "evaluate_state_adapter": evaluate_state_adapter,
    "terminal_state_adapter": terminal_state_adapter,
    "current_player_adapter": current_player_adapter,
    "pokemon_state_features": pokemon_state_features,
    "apply_move": apply_move,
}

for name, component in required_components.items():
    print(
        f"{name:<28}",
        "available" if component is not None else "MISSING",
    )


# # Part 2 — Rebuild the Test Matchup
# ## Step 4 — Create the Test Cards

# In[3]:


###############################################################################
# Step 4 — Create the Test Cards
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
                "A utility action used for integration testing."
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

print("✅ Tournament test cards created.")


# ## Step 5 — Create a Reusable Initial Battle Factory

# In[4]:


###############################################################################
# Step 5 — Reusable Initial Battle Factory
###############################################################################

def create_test_battle(
    starting_player: str = "Player",
) -> BattleState:
    """Create a fresh Eevee-ex versus Electrike battle."""

    if starting_player not in {"Player", "Opponent"}:
        raise ValueError(
            "starting_player must be 'Player' or 'Opponent'."
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


test_battle = create_test_battle()

print("✅ Fresh tournament test battle created.")
print("Player:", test_battle.player.active.card["name"])
print("Opponent:", test_battle.opponent.active.card["name"])
print("Starting side:", test_battle.current_player)


# ## Step 6 — Create the Move-Key Function

# In[5]:


###############################################################################
# Step 6 — Stable Move Key
###############################################################################

def pokemon_move_key(move: dict[str, Any]) -> tuple[Any, ...]:
    """Return a hashable key for a move dictionary."""

    return (
        move.get("name", "Unknown Move"),
        float(move.get("damage", 0) or 0),
        int(move.get("energy_cost", 0) or 0),
    )


test_move_key = pokemon_move_key(
    {
        "name": "Evolution Burst",
        "damage": 60.0,
        "energy_cost": 2,
    }
)

assert isinstance(test_move_key, tuple)
assert test_move_key[0] == "Evolution Burst"

print("✅ Pokémon move key ready:", test_move_key)


# ## Step 7 — Create a Reusable Agent Factory

# In[6]:


###############################################################################
# Step 7 — Create a Reusable Agent Factory
###############################################################################

def create_battle_agent(
    *,
    use_transposition_table: bool = True,
    use_move_ordering: bool = True,
    use_killer_moves: bool = True,
    use_history_heuristic: bool = True,
    seed: int = 20260714,
) -> PokemonBattleAgent:
    """Create a configured Pokémon battle agent."""

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


# These lines must be OUTSIDE the function.
tournament_agent = create_battle_agent()

print("✅ Tournament battle agent created.")
print(type(tournament_agent))
print(type(tournament_agent.engine))


# ## Step 8 — Create Two Tournament Agents

# In[7]:


###############################################################################
# Step 8 — Create Both Tournament Players
###############################################################################

player_agent = create_battle_agent(
    seed=1
)

opponent_agent = create_battle_agent(
    seed=2
)

print("✅ Tournament agents created.")
print()

print("Player agent:")
print(player_agent)

print()

print("Opponent agent:")
print(opponent_agent)


# ## Step 9 — Verify They Are Independent

# In[8]:


###############################################################################
# Step 9 — Verify Independent Engines
###############################################################################

assert player_agent is not opponent_agent
assert player_agent.engine is not opponent_agent.engine

print("✅ Independent tournament agents verified.")


# ## Step 10 — Build a Tournament Match Configuration

# In[9]:


###############################################################################
# Step 10 — Inspect Tournament Model Sources
###############################################################################

print("TournamentMatch module:", TournamentMatch.__module__)
print("TournamentResult module:", TournamentResult.__module__)
print("TournamentMatch:", TournamentMatch)
print("TournamentResult:", TournamentResult)


# ## Step 11 — Create the Match

# In[10]:


###############################################################################
# Step 11 — Create Tournament Match
###############################################################################

match = TournamentMatch(
    player_agent=player_agent,
    opponent_agent=opponent_agent,
    player_name="Tournament AI A",
    opponent_name="Tournament AI B",
    search_depth=6,
    max_turns=100,
)

print(match)


# ## Step 12 — Verify the Match

# In[11]:


###############################################################################
# Step 12 — Verify Tournament Match
###############################################################################

assert match.player_agent is player_agent
assert match.opponent_agent is opponent_agent

assert match.search_depth == 6
assert match.max_turns == 100

print("✅ Tournament match verified.")


# # Step 13 — Create the Tournament Result

# In[12]:


###############################################################################
# Step 13 — Confirm Production Tournament Result
###############################################################################

test_tournament_result = TournamentResult(
    winner="Tournament AI A",
    turns=3,
    final_score=170.0,
    transcript="Test transcript",
)

assert test_tournament_result.winner == "Tournament AI A"
assert test_tournament_result.turns == 3

print("✅ Production TournamentResult validated.")


# # Step 14 — Create a Match Runner

# In[13]:


###############################################################################
# Step 14 — Run a Tournament Match
###############################################################################

def run_match(
    match: TournamentMatch,
    initial_state: BattleState,
):

    simulation = simulate_ai_battle(
        initial_state=initial_state,
        agent=match.player_agent,
        search_depth=match.search_depth,
        max_turns=match.max_turns,
        verbose=False,
    )

    winner = determine_battle_winner(
        simulation.final_state
    )

    winner_name = (
        match.player_name
        if winner == "Player"
        else match.opponent_name
    )

    return TournamentResult(
        winner=winner_name,
        turns=len(simulation.turns),
        final_score=simulation.final_state.player.active.current_hp
        - simulation.final_state.opponent.active.current_hp,
        transcript=create_battle_transcript(simulation),
    )

print("✅ Match runner created.")


# # Step 15 — Execute the First Tournament Match

# In[14]:


###############################################################################
# Step 15 — Execute Tournament Match
###############################################################################

tournament_result = run_match(
    match=match,
    initial_state=create_test_battle(),
)

print("Winner:", tournament_result.winner)
print("Turns :", tournament_result.turns)
print("Score :", tournament_result.final_score)


# # Step 16 — Create a True Two-Agent Match Runner

# In[15]:


###############################################################################
# Step 16 — Confirm Production Two-Agent Match Runner
###############################################################################

from src.tournament.runner import (
    run_two_agent_match as production_run_two_agent_match,
)

run_two_agent_match = production_run_two_agent_match

assert run_two_agent_match.__module__ == "src.tournament.runner", (
    f"Unexpected runner source: "
    f"{run_two_agent_match.__module__}"
)

print("✅ Production two-agent match runner imported.")
print("Runner module:", run_two_agent_match.__module__)


# ## Step 17 — Run the First True Two-Agent Match

# In[16]:


###############################################################################
# Step 17 — Execute the First True Two-Agent Match
###############################################################################

two_agent_result = run_two_agent_match(
    match=match,
    initial_state=create_test_battle(),
    verbose=True,
)

print()
print("=" * 70)
print("TRUE TWO-AGENT MATCH RESULT")
print("=" * 70)
print("Winner:", two_agent_result.winner)
print("Turns:", two_agent_result.turns)
print("Final score:", two_agent_result.final_score)


# ## Step 18 — Validate the Two-Agent Match

# In[17]:


###############################################################################
# Step 18 — Validate the Two-Agent Match
###############################################################################

assert two_agent_result.winner == "Tournament AI A"
assert two_agent_result.turns == 3
assert two_agent_result.final_score == 170.0

assert "Evolution Burst" in two_agent_result.transcript
assert "Thunder Jolt" in two_agent_result.transcript
assert "Winner: Player" in two_agent_result.transcript

print("✅ True two-agent tournament match validated.")


# ## Step 19 — Create Tournament Statistics

# In[18]:


###############################################################################
# Step 19 — Confirm Production Statistics Model
###############################################################################

assert (
    TournamentStatistics.__module__
    == "src.tournament.statistics"
)

stats = TournamentStatistics()

print("✅ Production tournament statistics model imported.")
print("Statistics module:", TournamentStatistics.__module__)


# ## Step 20 — Statistics Updater

# In[19]:


###############################################################################
# Step 20 — Confirm Production Statistics Updater
###############################################################################

assert update_statistics.__module__ == "src.tournament.statistics"

print("✅ Production statistics updater imported.")
print("Updater module:", update_statistics.__module__)


# ## Step 21 — Create Tournament Statistics

# In[20]:


###############################################################################
# Step 21 — Compute Statistics with Production Module
###############################################################################

stats = TournamentStatistics()

stats = update_statistics(
    stats,
    two_agent_result,
    player_name=match.player_name,
    opponent_name=match.opponent_name,
)

print(stats)


# ## Step 22 — Validate Statistics

# In[21]:


###############################################################################
# Step 22 — Validate Statistics
###############################################################################

assert stats.matches_played == 1
assert stats.player_wins == 1
assert stats.opponent_wins == 0
assert stats.draws == 0

assert stats.average_turns == 3.0
assert stats.average_score == 170.0

print("✅ Tournament statistics validated.")


# ## Step 23 — Run a Match Series

# In[22]:


###############################################################################
# Step 23 — Run a Repeated Match Series
###############################################################################

def run_match_series(
    match: TournamentMatch,
    *,
    number_of_matches: int,
    alternate_starting_side: bool = True,
) -> tuple[list[TournamentResult], TournamentStatistics]:
    """Run repeated tournament matches and aggregate the statistics."""

    if number_of_matches < 1:
        raise ValueError(
            "number_of_matches must be at least 1."
        )

    results: list[TournamentResult] = []
    series_stats = TournamentStatistics()

    for match_index in range(number_of_matches):

        if alternate_starting_side:
            starting_player = (
                "Player"
                if match_index % 2 == 0
                else "Opponent"
            )
        else:
            starting_player = "Player"

        initial_state = create_test_battle(
            starting_player=starting_player
        )

        result = run_two_agent_match(
            match=match,
            initial_state=initial_state,
            verbose=False,
        )

        results.append(result)

        update_statistics(
            series_stats,
            result,
            player_name=match.player_name,
            opponent_name=match.opponent_name,
        )

        print(
            f"Match {match_index + 1:>2}: "
            f"start={starting_player:<8} | "
            f"winner={result.winner:<16} | "
            f"turns={result.turns}"
        )

    return results, series_stats


print("✅ Repeated match-series runner created.")


# ## Step 24 — Run a Best-of-5 Series

# In[24]:


###############################################################################
# Step 24 — Run a Best-of-5 Tournament Series
###############################################################################

series_results, series_stats = run_match_series(
    match=match,
    number_of_matches=5,
    alternate_starting_side=True,
)

print()
print("=" * 70)
print("BEST-OF-5 SERIES RESULTS")
print("=" * 70)
print("Matches played:", series_stats.matches_played)
print("Player wins:", series_stats.player_wins)
print("Opponent wins:", series_stats.opponent_wins)
print("Draws:", series_stats.draws)
print("Average turns:", series_stats.average_turns)
print("Average score:", series_stats.average_score)


# ## Step 25 — Validate the Series

# In[25]:


###############################################################################
# Step 25 — Validate the Match Series
###############################################################################

assert len(series_results) == 5
assert series_stats.matches_played == 5

assert (
    series_stats.player_wins
    + series_stats.opponent_wins
    + series_stats.draws
    == 5
)

assert all(
    result.turns > 0
    for result in series_results
)

assert all(
    result.transcript
    for result in series_results
)

print("✅ Best-of-5 tournament series validated.")


# ## Step 26 — Convert the Series Results to Rows

# In[26]:


###############################################################################
# Step 26 — Build Match-Series Records
###############################################################################

series_rows = []

for match_number, result in enumerate(
    series_results,
    start=1,
):
    series_rows.append(
        {
            "match_number": match_number,
            "winner": result.winner,
            "turns": result.turns,
            "final_score": result.final_score,
            "transcript_characters": len(result.transcript),
        }
    )

print("✅ Match-series records created.")
print("Rows:", len(series_rows))


# ## Step 27 — Create the Series DataFrame

# In[27]:


###############################################################################
# Step 27 — Create the Match-Series DataFrame
###############################################################################

series_df = pd.DataFrame(
    series_rows
)

series_df


# ## Step 28 — Add Summary Statistics

# In[28]:


###############################################################################
# Step 28 — Build the Series Summary
###############################################################################

series_summary = {
    "matches_played": series_stats.matches_played,
    "player_wins": series_stats.player_wins,
    "opponent_wins": series_stats.opponent_wins,
    "draws": series_stats.draws,
    "player_win_rate": series_stats.player_win_rate,
    "opponent_win_rate": series_stats.opponent_win_rate,
    "draw_rate": series_stats.draw_rate,
    "average_turns": series_stats.average_turns,
    "average_score": series_stats.average_score,
}

series_summary


# ## Step 29 — Validate the DataFrame and Summary

# In[29]:


###############################################################################
# Step 29 — Validate Series Data
###############################################################################

assert len(series_df) == 5

assert list(series_df.columns) == [
    "match_number",
    "winner",
    "turns",
    "final_score",
    "transcript_characters",
]

assert (
    series_summary["matches_played"]
    == len(series_df)
)

rate_total = (
    series_summary["player_win_rate"]
    + series_summary["opponent_win_rate"]
    + series_summary["draw_rate"]
)

assert abs(rate_total - 1.0) < 1e-9

assert series_df["turns"].min() > 0
assert series_df["transcript_characters"].min() > 0

print("✅ Match-series DataFrame and summary validated.")


# ## Step 30 — Display a Compact Tournament Report

# In[30]:


###############################################################################
# Step 30 — Display the Tournament Series Report
###############################################################################

print("=" * 70)
print("TOURNAMENT SERIES REPORT")
print("=" * 70)

print("Matches played:", series_stats.matches_played)
print(
    f"{match.player_name} wins:",
    series_stats.player_wins,
)
print(
    f"{match.opponent_name} wins:",
    series_stats.opponent_wins,
)
print("Draws:", series_stats.draws)

print()
print(
    f"{match.player_name} win rate:",
    f"{series_stats.player_win_rate:.1%}",
)
print(
    f"{match.opponent_name} win rate:",
    f"{series_stats.opponent_win_rate:.1%}",
)
print(
    "Draw rate:",
    f"{series_stats.draw_rate:.1%}",
)

print()
print(
    "Average turns:",
    f"{series_stats.average_turns:.2f}",
)
print(
    "Average score:",
    f"{series_stats.average_score:.2f}",
)

print()
print(series_df.to_string(index=False))


# ## Step 32 — Final Notebook 13 validation cell

# In[31]:


###############################################################################
# Step 32 — Notebook 13 Final Validation
###############################################################################

from pathlib import Path

notebook13_required_files = [
    PROJECT_ROOT / "src" / "tournament" / "__init__.py",
    PROJECT_ROOT / "src" / "tournament" / "match.py",
    PROJECT_ROOT / "src" / "tournament" / "runner.py",
    PROJECT_ROOT / "src" / "tournament" / "statistics.py",
]

for required_file in notebook13_required_files:
    assert required_file.exists(), (
        f"Missing production file: {required_file}"
    )

assert len(series_results) == 5
assert series_stats.matches_played == 5
assert len(series_df) == 5

assert (
    series_stats.player_wins
    + series_stats.opponent_wins
    + series_stats.draws
    == series_stats.matches_played
)

assert TournamentMatch.__module__ == "src.tournament.match"
assert TournamentResult.__module__ == "src.tournament.match"
assert TournamentStatistics.__module__ == "src.tournament.statistics"
assert run_two_agent_match.__module__ == "src.tournament.runner"
assert update_statistics.__module__ == "src.tournament.statistics"

print("=" * 70)
print("✅ NOTEBOOK 13 FINAL VALIDATION PASSED")
print("=" * 70)
print("Matches:", series_stats.matches_played)
print("Player wins:", series_stats.player_wins)
print("Opponent wins:", series_stats.opponent_wins)
print("Draws:", series_stats.draws)
print("Average turns:", series_stats.average_turns)
print("Average score:", series_stats.average_score)
print("Production package:", PROJECT_ROOT / "src" / "tournament")


# ## Step 33 — Save the tournament reports

# In[33]:


###############################################################################
# Step 33 — Save Notebook 13 Tournament Reports
###############################################################################

import json

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

series_csv_path = (
    REPORTS_DIR
    / "notebook13_tournament_series.csv"
)

summary_json_path = (
    REPORTS_DIR
    / "notebook13_tournament_summary.json"
)

report_txt_path = (
    REPORTS_DIR
    / "notebook13_tournament_report.txt"
)

series_df.to_csv(
    series_csv_path,
    index=False,
)

summary_json_path.write_text(
    json.dumps(
        series_summary,
        indent=2,
    ),
    encoding="utf-8",
)

report_lines = [
    "NOTEBOOK 13 TOURNAMENT SERIES REPORT",
    "=" * 70,
    f"Matches played: {series_stats.matches_played}",
    f"{match.player_name} wins: {series_stats.player_wins}",
    f"{match.opponent_name} wins: {series_stats.opponent_wins}",
    f"Draws: {series_stats.draws}",
    f"Player win rate: {series_stats.player_win_rate:.1%}",
    f"Opponent win rate: {series_stats.opponent_win_rate:.1%}",
    f"Draw rate: {series_stats.draw_rate:.1%}",
    f"Average turns: {series_stats.average_turns:.2f}",
    f"Average score: {series_stats.average_score:.2f}",
    "",
    series_df.to_string(index=False),
]

report_txt_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("✅ Notebook 13 reports saved.")
print("Series CSV:", series_csv_path)
print("Summary JSON:", summary_json_path)
print("Text report:", report_txt_path)


# ## Step 34 — Verify the saved reports

# In[34]:


###############################################################################
# Step 34 — Verify Notebook 13 Saved Reports
###############################################################################

assert series_csv_path.exists()
assert summary_json_path.exists()
assert report_txt_path.exists()

assert series_csv_path.stat().st_size > 0
assert summary_json_path.stat().st_size > 0
assert report_txt_path.stat().st_size > 0

saved_series_df = pd.read_csv(
    series_csv_path
)

saved_summary = json.loads(
    summary_json_path.read_text(
        encoding="utf-8"
    )
)

saved_report = report_txt_path.read_text(
    encoding="utf-8"
)

assert len(saved_series_df) == 5
assert saved_summary["matches_played"] == 5
assert "TOURNAMENT SERIES REPORT" in saved_report

print("✅ Notebook 13 saved reports validated.")
print("Saved series rows:", len(saved_series_df))
print("Saved summary matches:", saved_summary["matches_played"])
print("Saved report characters:", len(saved_report))


# In[ ]:




