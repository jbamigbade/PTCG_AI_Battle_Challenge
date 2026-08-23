#!/usr/bin/env python
# coding: utf-8

# # Part 1 — Setup and production imports
# ## Step 1 — Notebook introduction

# # Notebook 14 — Tournament Manager and Leaderboard
# 
# This notebook builds a reusable tournament-management layer on top of the
# production tournament package created in Notebook 13.
# 
# The system will:
# 
# - register multiple AI agents,
# - schedule repeated matches,
# - alternate starting sides,
# - collect individual match results,
# - calculate standings and win rates,
# - rank tournament entrants,
# - export match logs and leaderboard reports,
# - prepare the project for larger self-play experiments.

# # Step 2 — Imports and project root

# In[1]:


###############################################################################
# Step 2 — Imports and Project Root
###############################################################################

from __future__ import annotations

import sys

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

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
    TournamentStatistics,
    apply_move,
    build_leaderboard_dataframe,
    create_round_robin_schedule,
    current_player_adapter,
    evaluate_state_adapter,
    generate_moves_adapter,
    pokemon_state_features,
    run_two_agent_match,
    terminal_state_adapter,
    update_statistics,
)

from src.engine.advanced_search import (
    AdvancedSearchEngine,
    ZobristHasher,
)


print("✅ Notebook 14 imports completed.")
print("Project root:", PROJECT_ROOT)


# # Step 3 — Confirm production components

# In[2]:


###############################################################################
# Step 3 — Confirm Production Components
###############################################################################

required_components = {
    "TournamentMatch": TournamentMatch,
    "TournamentResult": TournamentResult,
    "TournamentStatistics": TournamentStatistics,
    "run_two_agent_match": run_two_agent_match,
    "update_statistics": update_statistics,
    "AdvancedSearchEngine": AdvancedSearchEngine,
    "PokemonBattleAgent": PokemonBattleAgent,
    "BattleState": BattleState,
}

for name, component in required_components.items():
    print(
        f"{name:<28}",
        "available"
        if component is not None
        else "MISSING",
    )

assert (
    TournamentMatch.__module__
    == "src.tournament.match"
)

assert (
    TournamentResult.__module__
    == "src.tournament.match"
)

assert (
    TournamentStatistics.__module__
    == "src.tournament.statistics"
)

assert (
    run_two_agent_match.__module__
    == "src.tournament.runner"
)

print()
print("✅ Production tournament components verified.")


# # Part 2 — Recreate the reusable battle setup
# ## Step 4 — Create test cards

# In[3]:


###############################################################################
# Step 4 — Tournament Test Cards
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

print("✅ Notebook 14 test cards created.")


# # Step 5 — Create the fresh battle-state factory

# In[4]:


###############################################################################
# Step 5 — Fresh Battle-State Factory
###############################################################################

def create_test_battle(
    starting_player: str = "Player",
) -> BattleState:
    """Create a fresh Eevee-ex versus Electrike battle."""

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


test_battle = create_test_battle()

print("✅ Fresh Notebook 14 battle created.")
print(
    "Player:",
    test_battle.player.active.card["name"],
)
print(
    "Opponent:",
    test_battle.opponent.active.card["name"],
)
print(
    "Starting side:",
    test_battle.current_player,
)


# ## Step 6 — Stable move-key function

# In[5]:


###############################################################################
# Step 6 — Stable Pokémon Move Key
###############################################################################

def pokemon_move_key(
    move: dict[str, Any],
) -> tuple[Any, ...]:
    """Convert a move dictionary into a hashable key."""

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


test_key = pokemon_move_key(
    {
        "name": "Evolution Burst",
        "damage": 60.0,
        "energy_cost": 2,
    }
)

assert test_key == (
    "Evolution Burst",
    60.0,
    2,
)

print("✅ Stable move key validated:", test_key)


# ## Step 7 — Reusable tournament-agent factory

# In[6]:


###############################################################################
# Step 7 — Reusable Tournament-Agent Factory
###############################################################################

def create_battle_agent(
    *,
    use_transposition_table: bool = True,
    use_move_ordering: bool = True,
    use_killer_moves: bool = True,
    use_history_heuristic: bool = True,
    seed: int = 20260715,
) -> PokemonBattleAgent:
    """Create one independently configured Pokémon battle agent."""

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


baseline_agent = create_battle_agent(
    use_transposition_table=False,
    use_move_ordering=False,
    use_killer_moves=False,
    use_history_heuristic=False,
    seed=101,
)

optimized_agent = create_battle_agent(
    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=True,
    use_history_heuristic=True,
    seed=202,
)

print("✅ Baseline and optimized agents created.")
print(
    "Independent agents:",
    baseline_agent is not optimized_agent,
)
print(
    "Independent engines:",
    baseline_agent.engine
    is not optimized_agent.engine,
)


# ## Step 8 — Validate One Production Match

# In[7]:


###############################################################################
# Step 8 — Validate One Production Match
###############################################################################

validation_match = TournamentMatch(
    player_agent=optimized_agent,
    opponent_agent=baseline_agent,
    player_name="Optimized AI",
    opponent_name="Baseline AI",
    search_depth=6,
    max_turns=20,
)

validation_result = run_two_agent_match(
    match=validation_match,
    initial_state=create_test_battle(),
    verbose=False,
)

print("Winner:", validation_result.winner)
print("Turns:", validation_result.turns)
print("Final score:", validation_result.final_score)

assert validation_result.turns > 0
assert validation_result.transcript

assert validation_result.winner in {
    "Optimized AI",
    "Baseline AI",
    "Draw",
    "No winner",
}

print("✅ Production tournament match validated.")


# ## Step 9 — Create a Tournament Entrant Model

# In[8]:


###############################################################################
# Step 9 — Confirm Production Tournament Entrant
###############################################################################

assert TournamentEntrant.__module__ == "src.tournament.manager"

print("✅ Production TournamentEntrant imported.")
print("Module:", TournamentEntrant.__module__)


# ## Step 10 — Register the Tournament Entrants

# In[9]:


###############################################################################
# Step 10 — Register Tournament Entrants
###############################################################################

baseline_entrant = TournamentEntrant(
    name="Baseline AI",
    agent=baseline_agent,
    description=(
        "Alpha-beta search without transposition tables, "
        "move ordering, killer moves, or history heuristic."
    ),
)

optimized_entrant = TournamentEntrant(
    name="Optimized AI",
    agent=optimized_agent,
    description=(
        "Alpha-beta search with transposition tables, "
        "move ordering, killer moves, and history heuristic."
    ),
)

entrants = [
    baseline_entrant,
    optimized_entrant,
]

print("✅ Tournament entrants registered.")
print("Entrants:", len(entrants))

for entrant in entrants:
    print(
        f"- {entrant.name}: "
        f"{entrant.description}"
    )


# # Step 11 — Tournament Manager

# In[10]:


###############################################################################
# Step 11 — Confirm Production Tournament Manager
###############################################################################

assert TournamentManager.__module__ == "src.tournament.manager"

print("✅ Production TournamentManager imported.")
print("Module:", TournamentManager.__module__)


# # Step 12 — Register Entrants

# In[11]:


###############################################################################
# Step 12 — Register Entrants
###############################################################################

manager = TournamentManager()

manager.register(
    baseline_entrant
)

manager.register(
    optimized_entrant
)

print(
    "Entrants:",
    manager.number_of_entrants
)

assert manager.number_of_entrants == 2

print("✅ Entrants registered.")


# # Step 13 — Inspect the Manager

# In[12]:


###############################################################################
# Step 13 — Inspect Tournament Manager
###############################################################################

print("=" * 70)
print("TOURNAMENT MANAGER")
print("=" * 70)

for entrant in manager.entrants:

    print()

    print(
        "Name:",
        entrant.name,
    )

    print(
        "Description:",
        entrant.description,
    )

    print(
        "Matches:",
        entrant.matches_played,
    )

    print(
        "Wins:",
        entrant.wins,
    )


# # Step 14 — Add Match Execution to the Tournament Manager

# In[13]:


###############################################################################
# Step 14 — Tournament Manager with Match Execution
###############################################################################

@dataclass
class TournamentManager:
    """Manage tournament entrants, matches, and standings."""

    entrants: list[TournamentEntrant] = field(
        default_factory=list
    )

    match_history: list[TournamentResult] = field(
        default_factory=list
    )

    statistics: TournamentStatistics = field(
        default_factory=TournamentStatistics
    )

    def register(
        self,
        entrant: TournamentEntrant,
    ) -> None:
        """Register a new entrant."""

        if any(
            existing.name == entrant.name
            for existing in self.entrants
        ):
            raise ValueError(
                f"An entrant named {entrant.name!r} "
                "is already registered."
            )

        self.entrants.append(
            entrant
        )

    @property
    def number_of_entrants(
        self,
    ) -> int:
        return len(
            self.entrants
        )

    @property
    def matches_played(
        self,
    ) -> int:
        return len(
            self.match_history
        )

    def get_entrant(
        self,
        name: str,
    ) -> TournamentEntrant:
        """Return a registered entrant by name."""

        for entrant in self.entrants:
            if entrant.name == name:
                return entrant

        raise KeyError(
            f"No entrant named {name!r} is registered."
        )

    def run_match(
        self,
        player_name: str,
        opponent_name: str,
        initial_state: BattleState,
        *,
        search_depth: int = 6,
        max_turns: int = 100,
        verbose: bool = False,
    ) -> TournamentResult:
        """Run one tournament match and update standings."""

        player_entrant = self.get_entrant(
            player_name
        )

        opponent_entrant = self.get_entrant(
            opponent_name
        )

        match = TournamentMatch(
            player_agent=player_entrant.agent,
            opponent_agent=opponent_entrant.agent,
            player_name=player_entrant.name,
            opponent_name=opponent_entrant.name,
            search_depth=search_depth,
            max_turns=max_turns,
        )

        result = run_two_agent_match(
            match=match,
            initial_state=initial_state,
            verbose=verbose,
        )

        self.match_history.append(
            result
        )

        self._update_entrant_records(
            player_entrant=player_entrant,
            opponent_entrant=opponent_entrant,
            result=result,
        )

        update_statistics(
            self.statistics,
            result,
            player_name=player_entrant.name,
            opponent_name=opponent_entrant.name,
        )

        return result

    def _update_entrant_records(
        self,
        *,
        player_entrant: TournamentEntrant,
        opponent_entrant: TournamentEntrant,
        result: TournamentResult,
    ) -> None:
        """Update both entrants after one match."""

        player_entrant.matches_played += 1
        opponent_entrant.matches_played += 1

        player_entrant.total_turns += (
            result.turns
        )

        opponent_entrant.total_turns += (
            result.turns
        )

        player_entrant.total_score += (
            result.final_score
        )

        opponent_entrant.total_score -= (
            result.final_score
        )

        if result.winner == player_entrant.name:
            player_entrant.wins += 1
            opponent_entrant.losses += 1

        elif result.winner == opponent_entrant.name:
            opponent_entrant.wins += 1
            player_entrant.losses += 1

        else:
            player_entrant.draws += 1
            opponent_entrant.draws += 1


print("✅ TournamentManager match execution added.")


# # Step 15 — Recreate and Register the Entrants

# In[14]:


###############################################################################
# Step 15 — Recreate Tournament Manager
###############################################################################

manager = TournamentManager()

manager.register(
    baseline_entrant
)

manager.register(
    optimized_entrant
)

print("✅ Tournament manager recreated.")
print("Entrants:", manager.number_of_entrants)
print("Matches played:", manager.matches_played)


# # Step 16 — Run the First Managed Match

# In[15]:


###############################################################################
# Step 16 — Run the First Managed Match
###############################################################################

managed_result = manager.run_match(
    player_name="Optimized AI",
    opponent_name="Baseline AI",
    initial_state=create_test_battle(
        starting_player="Player"
    ),
    search_depth=6,
    max_turns=20,
    verbose=True,
)

print()
print("=" * 70)
print("MANAGED MATCH RESULT")
print("=" * 70)

print("Winner:", managed_result.winner)
print("Turns:", managed_result.turns)
print("Final score:", managed_result.final_score)
print("Manager matches:", manager.matches_played)


# # Step 17 — Validate Updated Entrant Records

# In[16]:


###############################################################################
# Step 17 — Validate Updated Entrant Records
###############################################################################

optimized_record = manager.get_entrant(
    "Optimized AI"
)

baseline_record = manager.get_entrant(
    "Baseline AI"
)

assert manager.matches_played == 1

assert optimized_record.matches_played == 1
assert baseline_record.matches_played == 1

assert (
    optimized_record.wins
    + optimized_record.losses
    + optimized_record.draws
    == 1
)

assert (
    baseline_record.wins
    + baseline_record.losses
    + baseline_record.draws
    == 1
)

assert (
    optimized_record.wins
    == baseline_record.losses
)

assert (
    baseline_record.wins
    == optimized_record.losses
)

print("✅ Managed match records validated.")
print()
print(
    "Optimized AI:",
    {
        "wins": optimized_record.wins,
        "losses": optimized_record.losses,
        "draws": optimized_record.draws,
    },
)
print(
    "Baseline AI:",
    {
        "wins": baseline_record.wins,
        "losses": baseline_record.losses,
        "draws": baseline_record.draws,
    },
)


# # Step 18 — Build the Leaderboard

# In[17]:


###############################################################################
# Step 18 — Build Tournament Leaderboard
###############################################################################

leaderboard = sorted(
    manager.entrants,
    key=lambda entrant: (
        entrant.wins,
        entrant.win_rate,
        entrant.average_score,
    ),
    reverse=True,
)

print("=" * 70)
print("TOURNAMENT LEADERBOARD")
print("=" * 70)

for rank, entrant in enumerate(
    leaderboard,
    start=1,
):
    print(
        f"{rank}. {entrant.name}"
    )
    print(
        f"   Record : "
        f"{entrant.wins}-"
        f"{entrant.losses}-"
        f"{entrant.draws}"
    )
    print(
        f"   Win Rate : "
        f"{entrant.win_rate:.1%}"
    )
    print(
        f"   Avg Turns : "
        f"{entrant.average_turns:.2f}"
    )
    print(
        f"   Avg Score : "
        f"{entrant.average_score:.2f}"
    )
    print()


# # Step 19 — Validate the Leaderboard

# In[18]:


###############################################################################
# Step 19 — Validate Leaderboard
###############################################################################

assert len(leaderboard) == 2

assert leaderboard[0].name == "Optimized AI"

assert leaderboard[1].name == "Baseline AI"

assert leaderboard[0].wins == 1

assert leaderboard[1].losses == 1

print("✅ Leaderboard validated.")


# # Step 20 — Create a Round-Robin Schedule

# In[19]:


###############################################################################
# Step 20 — Confirm Production Round-Robin Scheduler
###############################################################################

assert (
    create_round_robin_schedule.__module__
    == "src.tournament.scheduler"
)

print("✅ Production round-robin scheduler imported.")
print("Module:", create_round_robin_schedule.__module__)


# # Step 21 — Validate the Schedule

# In[20]:


################################################################################
# Step 20 — Create Production Round-Robin Schedule
###############################################################################

assert (
    create_round_robin_schedule.__module__
    == "src.tournament.scheduler"
)

round_robin_schedule = create_round_robin_schedule(
    manager.entrants,
    games_per_pairing=2,
)

print("✅ Production round-robin scheduler used.")
print("Module:", create_round_robin_schedule.__module__)
print("Scheduled matches:", len(round_robin_schedule))

for scheduled_match in round_robin_schedule:
    print(scheduled_match)


# # Step 22 — Reset the Tournament Records

# In[21]:


###############################################################################
# Step 22 — Reset Tournament Records with Production Classes
###############################################################################

from src.tournament.manager import (
    TournamentEntrant as ProductionTournamentEntrant,
    TournamentManager as ProductionTournamentManager,
)

TournamentEntrant = ProductionTournamentEntrant
TournamentManager = ProductionTournamentManager


fresh_baseline_entrant = TournamentEntrant(
    name="Baseline AI",
    agent=baseline_agent,
    description=baseline_entrant.description,
)

fresh_optimized_entrant = TournamentEntrant(
    name="Optimized AI",
    agent=optimized_agent,
    description=optimized_entrant.description,
)

round_robin_manager = TournamentManager()

round_robin_manager.register(
    fresh_baseline_entrant
)

round_robin_manager.register(
    fresh_optimized_entrant
)

assert (
    TournamentEntrant.__module__
    == "src.tournament.manager"
)

assert (
    TournamentManager.__module__
    == "src.tournament.manager"
)

assert round_robin_manager.matches_played == 0
assert round_robin_manager.number_of_entrants == 2

assert all(
    entrant.matches_played == 0
    for entrant in round_robin_manager.entrants
)

print("✅ Fresh production round-robin manager created.")
print("Entrant module:", TournamentEntrant.__module__)
print("Manager module:", TournamentManager.__module__)
print("Entrants:", round_robin_manager.number_of_entrants)
print("Matches played:", round_robin_manager.matches_played)


# # Step 23 — Run the Round-Robin Tournament

# In[22]:


###############################################################################
# Step 23 — Run the Production Round-Robin Tournament
###############################################################################

round_robin_results: list[TournamentResult] = []

for scheduled_match in round_robin_schedule:
    result = round_robin_manager.run_match(
        player_name=scheduled_match["player_name"],
        opponent_name=scheduled_match["opponent_name"],
        initial_state=create_test_battle(
            starting_player=scheduled_match["starting_player"]
        ),
        search_depth=6,
        max_turns=20,
        verbose=False,
    )

    round_robin_results.append(result)

    print(
        f"Match {scheduled_match['match_number']}: "
        f"{scheduled_match['player_name']} vs "
        f"{scheduled_match['opponent_name']} | "
        f"Winner: {result.winner} | "
        f"Turns: {result.turns} | "
        f"Score: {result.final_score:.1f}"
    )

print()
print("✅ Production round-robin tournament completed.")
print("Matches played:", round_robin_manager.matches_played)


# # Step 24 — Validate the Round-Robin Results

# In[33]:


###############################################################################
# Step 24 — Validate the Production Round-Robin Tournament
###############################################################################

assert len(round_robin_results) == 2

assert round_robin_manager.matches_played == 2

assert all(
    result.turns > 0
    for result in round_robin_results
)

assert all(
    result.transcript
    for result in round_robin_results
)

total_matches = sum(
    entrant.matches_played
    for entrant in round_robin_manager.entrants
)

assert total_matches == 4

for entrant in round_robin_manager.entrants:
    assert entrant.matches_played == 2

    assert (
        entrant.wins
        + entrant.losses
        + entrant.draws
        == 2
    )

print("✅ Production round-robin validated.")


# # Step 25 — Build the Tournament Leaderboard DataFrame

# In[34]:


###############################################################################
# Step 25 — Build the Production Tournament Leaderboard
###############################################################################

assert (
    build_leaderboard_dataframe.__module__
    == "src.tournament.leaderboard"
)

leaderboard_df = build_leaderboard_dataframe(
    round_robin_manager.entrants
)

print("✅ Production tournament leaderboard created.")
print("Module:", build_leaderboard_dataframe.__module__)

leaderboard_df


# In[25]:


assert (
    build_leaderboard_dataframe.__module__
    == "src.tournament.leaderboard"
)

print("✅ Production leaderboard builder used.")


# # Step 26 — Validate the Leaderboard

# In[35]:


###############################################################################
# Step 26 — Validate the Production Leaderboard
###############################################################################

assert len(leaderboard_df) == 2

assert leaderboard_df["matches"].sum() == 4
assert leaderboard_df["wins"].sum() == 2
assert leaderboard_df["losses"].sum() == 2
assert leaderboard_df["draws"].sum() == 0

assert all(
    leaderboard_df["matches"] == 2
)

assert set(leaderboard_df["name"]) == {
    "Baseline AI",
    "Optimized AI",
}

assert all(
    leaderboard_df["win_rate"] == 0.5
)

assert all(
    leaderboard_df["average_turns"] == 3.0
)

print("✅ Production tournament leaderboard validated.")


# # Step 27 — Tournament Summary

# In[36]:


###############################################################################
# Step 27 — Display the Production Tournament Summary
###############################################################################

print("=" * 70)
print("PRODUCTION TOURNAMENT SUMMARY")
print("=" * 70)
print()

print(
    "Matches played:",
    round_robin_manager.matches_played,
)
print()

for entrant in leaderboard_df.itertuples():
    print(entrant.name)

    print(
        f"  Record: "
        f"{entrant.wins}-"
        f"{entrant.losses}-"
        f"{entrant.draws}"
    )

    print(
        f"  Win rate: "
        f"{entrant.win_rate:.1%}"
    )

    print(
        f"  Average turns: "
        f"{entrant.average_turns:.2f}"
    )

    print(
        f"  Average score: "
        f"{entrant.average_score:.2f}"
    )

    print()

print("=" * 70)

leaderboard_df


# # Step 28 — Save the Tournament Leaderboard

# In[37]:


###############################################################################
# Step 28 — Save the Production Leaderboard
###############################################################################

from pathlib import Path

OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

leaderboard_csv_path = (
    OUTPUT_DIR
    / "notebook14_leaderboard.csv"
)

leaderboard_df.to_csv(
    leaderboard_csv_path,
    index=False,
)

print("✅ Production leaderboard saved.")
print("CSV:", leaderboard_csv_path)
print(
    "Size:",
    leaderboard_csv_path.stat().st_size,
    "bytes",
)


# # Step 29 — Save Tournament Summary (JSON)

# In[38]:


###############################################################################
# Step 29 — Save the Production Tournament Summary
###############################################################################

import json

summary_json_path = (
    OUTPUT_DIR
    / "notebook14_summary.json"
)

summary = {
    "matches_played": round_robin_manager.matches_played,
    "leaderboard": leaderboard_df.to_dict(
        orient="records"
    ),
}

summary_json_path.write_text(
    json.dumps(
        summary,
        indent=4,
    ),
    encoding="utf-8",
)

print("✅ Production tournament summary saved.")
print("JSON:", summary_json_path)
print(
    "Size:",
    summary_json_path.stat().st_size,
    "bytes",
)


# # Step 30 — Save Tournament Report (TXT)

# In[39]:


###############################################################################
# Step 30 — Save the Production Tournament Report
###############################################################################

report_txt_path = (
    OUTPUT_DIR
    / "notebook14_report.txt"
)

report_lines = [
    "=" * 70,
    "PRODUCTION TOURNAMENT REPORT",
    "=" * 70,
    "",
    f"Matches played: {round_robin_manager.matches_played}",
    "",
]

for entrant in leaderboard_df.itertuples():
    report_lines.extend(
        [
            entrant.name,
            (
                f"  Record: "
                f"{entrant.wins}-"
                f"{entrant.losses}-"
                f"{entrant.draws}"
            ),
            f"  Win rate: {entrant.win_rate:.1%}",
            f"  Average turns: {entrant.average_turns:.2f}",
            f"  Average score: {entrant.average_score:.2f}",
            "",
        ]
    )

report_lines.extend(
    [
        "=" * 70,
        "",
        "LEADERBOARD",
        "",
        leaderboard_df.to_string(index=False),
    ]
)

report_txt_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("✅ Production tournament report saved.")
print("TXT:", report_txt_path)
print(
    "Size:",
    report_txt_path.stat().st_size,
    "bytes",
)


# # Step 31 — Verify the Saved Tournament Files

# In[40]:


###############################################################################
# Step 31 — Verify All Production Report Files
###############################################################################

assert leaderboard_csv_path.exists()
assert summary_json_path.exists()
assert report_txt_path.exists()

assert leaderboard_csv_path.stat().st_size > 0
assert summary_json_path.stat().st_size > 0
assert report_txt_path.stat().st_size > 0

saved_leaderboard = pd.read_csv(
    leaderboard_csv_path
)

saved_summary = json.loads(
    summary_json_path.read_text(
        encoding="utf-8"
    )
)

saved_report = report_txt_path.read_text(
    encoding="utf-8"
)

assert len(saved_leaderboard) == 2
assert saved_summary["matches_played"] == 2
assert len(saved_summary["leaderboard"]) == 2

assert (
    set(saved_leaderboard["name"])
    == {
        "Baseline AI",
        "Optimized AI",
    }
)

assert (
    "PRODUCTION TOURNAMENT REPORT"
    in saved_report
)

assert "Baseline AI" in saved_report
assert "Optimized AI" in saved_report

print("✅ Notebook 14 production reports validated.")
print("Leaderboard rows:", len(saved_leaderboard))
print("Summary matches:", saved_summary["matches_played"])
print("Summary entrants:", len(saved_summary["leaderboard"]))
print("Report characters:", len(saved_report))


# ## Step 32 — Final Notebook 14 Validation

# In[41]:


###############################################################################
# Step 32 — Final Notebook 14 Validation
###############################################################################

required_production_files = [
    PROJECT_ROOT / "src" / "tournament" / "__init__.py",
    PROJECT_ROOT / "src" / "tournament" / "leaderboard.py",
    PROJECT_ROOT / "src" / "tournament" / "manager.py",
    PROJECT_ROOT / "src" / "tournament" / "match.py",
    PROJECT_ROOT / "src" / "tournament" / "runner.py",
    PROJECT_ROOT / "src" / "tournament" / "scheduler.py",
    PROJECT_ROOT / "src" / "tournament" / "statistics.py",
]

for required_file in required_production_files:
    assert required_file.exists(), (
        f"Missing production file: {required_file}"
    )

assert TournamentEntrant.__module__ == "src.tournament.manager"
assert TournamentManager.__module__ == "src.tournament.manager"

assert (
    create_round_robin_schedule.__module__
    == "src.tournament.scheduler"
)

assert (
    build_leaderboard_dataframe.__module__
    == "src.tournament.leaderboard"
)

assert round_robin_manager.matches_played == 2
assert len(round_robin_results) == 2
assert len(leaderboard_df) == 2

assert leaderboard_csv_path.exists()
assert summary_json_path.exists()
assert report_txt_path.exists()

assert (
    round_robin_manager.statistics.matches_played
    == 2
)

assert all(
    entrant.matches_played == 2
    for entrant in round_robin_manager.entrants
)

print("=" * 70)
print("✅ NOTEBOOK 14 FINAL VALIDATION PASSED")
print("=" * 70)

print(
    "Production package:",
    PROJECT_ROOT / "src" / "tournament",
)

print(
    "Matches played:",
    round_robin_manager.matches_played,
)

print(
    "Leaderboard rows:",
    len(leaderboard_df),
)

print(
    "Reports directory:",
    OUTPUT_DIR,
)

print(
    "Production files:",
    len(required_production_files),
)


# In[ ]:




