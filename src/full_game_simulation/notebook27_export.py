#!/usr/bin/env python
# coding: utf-8

# # Notebook 27 — Full Game Simulation
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 26 verified production-agent consistency, latency, accuracy, and runtime stability.
# 
# Notebook 27 advances from repeated single-state decisions to full-game simulation.
# 
# ## Objectives
# 
# 1. Inspect the existing battle-engine and integrated-agent interfaces.
# 2. Reuse the project’s battle-state and action models.
# 3. Create a full-game simulation controller.
# 4. Alternate turns between two agents.
# 5. apply legal actions to the evolving game state.
# 6. Detect game-ending conditions.
# 7. Record winners, turns, actions, errors, and termination reasons.
# 8. Run repeatable simulated matches.
# 9. Generate full-game statistics and reports.
# 10. Prepare the simulator for self-play and strategy optimization.
# 
# ## Important distinction
# 
# A full game must include evolving states, alternating players, legal actions,
# turn progression, victory conditions, and a final winner or termination result.

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
import types
import uuid

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

print("Python:", sys.version)
print("Working directory:", Path.cwd())


# # Cell 3 — Locate the project and prior components

# In[4]:


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    required_markers = {
        "notebooks",
        "scripts",
        "src",
        "reports",
    }

    for candidate in [current, *current.parents]:
        existing = {
            marker
            for marker in required_markers
            if (candidate / marker).exists()
        }

        if existing == required_markers:
            return candidate

    raise FileNotFoundError(
        "Could not locate the PTCG project root."
    )


PROJECT_ROOT = find_project_root()

BATTLE_ENGINE_EXPORT = (
    PROJECT_ROOT
    / "scripts"
    / "07_battle_engine.py"
)

INTEGRATED_AGENT_EXPORT = (
    PROJECT_ROOT
    / "scripts"
    / "12_integrated_battle_agent.py"
)

SELF_PLAY_EXPORT = (
    PROJECT_ROOT
    / "scripts"
    / "13_tournament_self_play_evaluation.py"
)

PRODUCTION_AGENT_EXPORT = (
    PROJECT_ROOT
    / "src"
    / "kaggle_agent"
    / "notebook21_export.py"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook27"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_FILES = {
    "battle_engine": BATTLE_ENGINE_EXPORT,
    "integrated_agent": INTEGRATED_AGENT_EXPORT,
    "self_play": SELF_PLAY_EXPORT,
    "production_agent": PRODUCTION_AGENT_EXPORT,
}

print("Project root:", PROJECT_ROOT)
print()

for name, path in SOURCE_FILES.items():
    print(
        f"{'[FOUND]' if path.is_file() else '[MISSING]'} "
        f"{name}: {path}"
    )

print()
print("Report directory:", REPORT_DIR)

missing_sources = [
    str(path)
    for path in SOURCE_FILES.values()
    if not path.is_file()
]

if missing_sources:
    raise FileNotFoundError(
        "Notebook 27 requires these files:\n"
        + "\n".join(missing_sources)
    )

print("\nAll required source files located.")


# # Cell 4 — Safe export loader

# In[5]:


def load_export_module(
    path: Path,
    label: str,
) -> types.ModuleType:
    source = path.read_text(
        encoding="utf-8-sig"
    )

    source_lines = [
        line
        for line in source.splitlines()
        if line.strip()
        != "from __future__ import annotations"
    ]

    cleaned_source = (
        "from __future__ import annotations\n"
        + "\n".join(source_lines)
    )

    module_name = (
        f"notebook27_{label}_"
        + uuid.uuid4().hex
    )

    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""

    sys.modules[module_name] = module

    compiled = compile(
        cleaned_source,
        str(path),
        "exec",
    )

    exec(
        compiled,
        module.__dict__,
    )

    return module


print("Safe export loader created.")


# # Cell 5 — Inspect declarations

# In[7]:


import ast


def inspect_source_structure(
    path: Path,
) -> dict[str, list[str]]:
    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    classes = []
    functions = []
    imports = []
    assignments = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""

            for alias in node.names:
                imports.append(
                    f"{module_name}.{alias.name}"
                )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            targets = []

            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]

            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.append(
                        target.id
                    )

    return {
        "classes": sorted(set(classes)),
        "functions": sorted(set(functions)),
        "imports": sorted(set(imports)),
        "assignments": sorted(set(assignments)),
    }


source_declarations = {}

for label, path in SOURCE_FILES.items():
    declarations = inspect_source_structure(
        path
    )

    source_declarations[label] = declarations

    print("=" * 72)
    print(label)
    print("=" * 72)

    print("Classes:")
    if declarations["classes"]:
        for name in declarations["classes"]:
            print(" -", name)
    else:
        print(" - None found")

    print("\nFunctions:")
    if declarations["functions"]:
        for name in declarations["functions"]:
            print(" -", name)
    else:
        print(" - None found")

    print("\nImports:")
    for name in declarations["imports"][:30]:
        print(" -", name)

    if len(declarations["imports"]) > 30:
        print(
            " - ...",
            len(declarations["imports"]) - 30,
            "more imports",
        )

    print("\nAssigned names:")
    for name in declarations["assignments"][:40]:
        print(" -", name)

    if len(declarations["assignments"]) > 40:
        print(
            " - ...",
            len(declarations["assignments"]) - 40,
            "more assigned names",
        )

    print()


battle_structure = source_declarations[
    "battle_engine"
]

assert (
    battle_structure["classes"]
    or battle_structure["functions"]
    or battle_structure["imports"]
), (
    "The battle-engine export appears to contain "
    "no detectable executable interface."
)

print("Source-interface inspection completed.")


# # Cell 5A — Add project root to Python path

# In[10]:


# Cell 5A — Make the project package importable

PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

print("Project root added to sys.path:")
print(PROJECT_ROOT_STR)

print()
print("src exists:", (PROJECT_ROOT / "src").is_dir())
print("src __init__ exists:", (PROJECT_ROOT / "src" / "__init__.py").is_file())

assert (PROJECT_ROOT / "src").is_dir()


# In[11]:


import src

print("src imported successfully.")
print("src location:", src.__file__)


# # Cell 6 — Import the existing full-game components

# In[12]:


# Cell 6 — Import the existing full-game components

from src import (
    BattleSimulationResult,
    BattleState,
    BattleTurnRecord,
    PlayerState,
    PokemonBattleAgent,
    PokemonState,
    apply_move,
    create_battle_transcript,
    current_player_adapter,
    determine_battle_winner,
    evaluate_state_adapter,
    generate_moves_adapter,
    pokemon_state_features,
    simulate_ai_battle,
    terminal_state_adapter,
)

from src.engine.advanced_search import (
    AdvancedSearchEngine,
    ZobristHasher,
)

from src.tournament.runner import (
    run_two_agent_match,
)

print("Full-game components imported:")
print("- BattleState")
print("- PlayerState")
print("- PokemonState")
print("- PokemonBattleAgent")
print("- AdvancedSearchEngine")
print("- simulate_ai_battle")
print("- run_two_agent_match")
print("- determine_battle_winner")
print("- terminal_state_adapter")

print("\nExisting full-game interface loaded.")


# # Cell 7 — Inspect constructor and function signatures

# In[13]:


# Cell 7 — Inspect constructor and function signatures

FULL_GAME_OBJECTS = {
    "BattleState": BattleState,
    "PlayerState": PlayerState,
    "PokemonState": PokemonState,
    "PokemonBattleAgent": PokemonBattleAgent,
    "AdvancedSearchEngine": AdvancedSearchEngine,
    "ZobristHasher": ZobristHasher,
    "simulate_ai_battle": simulate_ai_battle,
    "run_two_agent_match": run_two_agent_match,
    "apply_move": apply_move,
    "determine_battle_winner": determine_battle_winner,
    "terminal_state_adapter": terminal_state_adapter,
    "generate_moves_adapter": generate_moves_adapter,
}

for name, obj in FULL_GAME_OBJECTS.items():
    print("=" * 72)
    print(name)
    print("=" * 72)

    try:
        print("Signature:", inspect.signature(obj))
    except (TypeError, ValueError):
        print("Signature: unavailable")

    doc = inspect.getdoc(obj)

    if doc:
        print("Documentation:")
        for line in doc.splitlines()[:8]:
            print(" ", line)
    else:
        print("Documentation: none")

    print()

print("Full-game signatures inspected.")


# # Cell 8 — Inspect state-model fields

# In[14]:


# Cell 8 — Inspect state-model fields

from dataclasses import MISSING, fields

STATE_MODELS = {
    "PokemonState": PokemonState,
    "PlayerState": PlayerState,
    "BattleState": BattleState,
}

for model_name, model_class in STATE_MODELS.items():
    print("=" * 72)
    print(model_name)
    print("=" * 72)

    if not is_dataclass(model_class):
        print("Not a dataclass.")
        print()
        continue

    for item in fields(model_class):
        if item.default is not MISSING:
            requirement = f"default={item.default!r}"
        elif item.default_factory is not MISSING:
            requirement = "default_factory"
        else:
            requirement = "required"

        print(
            f"- {item.name}: {item.type} "
            f"({requirement})"
        )

    print()

print("State-model inspection completed.")


# # Cell 8 — Inspect state-model fields

# In[15]:


# Cell 8 — Inspect state-model fields

from dataclasses import MISSING, fields

STATE_MODELS = {
    "PokemonState": PokemonState,
    "PlayerState": PlayerState,
    "BattleState": BattleState,
}

for model_name, model_class in STATE_MODELS.items():
    print("=" * 72)
    print(model_name)
    print("=" * 72)

    if not is_dataclass(model_class):
        print("Not a dataclass.")
        print()
        continue

    for item in fields(model_class):
        if item.default is not MISSING:
            requirement = f"default={item.default!r}"
        elif item.default_factory is not MISSING:
            requirement = "default_factory"
        else:
            requirement = "required"

        print(
            f"- {item.name}: {item.type} "
            f"({requirement})"
        )

    print()

print("State-model inspection completed.")


# # Cell 9 — Inspect tournament classes and match runner

# In[16]:


# Cell 9 — Inspect tournament interface

from src import (
    TournamentMatch,
    TournamentResult,
    TournamentStatistics,
)

TOURNAMENT_OBJECTS = {
    "TournamentMatch": TournamentMatch,
    "TournamentResult": TournamentResult,
    "TournamentStatistics": TournamentStatistics,
    "run_two_agent_match": run_two_agent_match,
}

for name, obj in TOURNAMENT_OBJECTS.items():
    print("=" * 72)
    print(name)
    print("=" * 72)

    try:
        print("Signature:", inspect.signature(obj))
    except (TypeError, ValueError):
        print("Signature: unavailable")

    if is_dataclass(obj):
        print("Fields:")

        for item in fields(obj):
            if item.default is not MISSING:
                requirement = f"default={item.default!r}"
            elif item.default_factory is not MISSING:
                requirement = "default_factory"
            else:
                requirement = "required"

            print(
                f"- {item.name}: {item.type} "
                f"({requirement})"
            )

    doc = inspect.getdoc(obj)

    if doc:
        print("Documentation:")
        for line in doc.splitlines()[:8]:
            print(" ", line)

    print()

print("Tournament interface inspected.")


# # Cell 10 — Inspect the existing test-battle builder

# In[17]:


# Cell 10 — Load the existing self-play helpers

self_play_module = load_export_module(
    SELF_PLAY_EXPORT,
    "self_play",
)

REQUIRED_SELF_PLAY_HELPERS = [
    "create_test_battle",
    "create_battle_agent",
    "run_match",
    "run_match_series",
]

missing_helpers = [
    name
    for name in REQUIRED_SELF_PLAY_HELPERS
    if not hasattr(self_play_module, name)
]

for name in REQUIRED_SELF_PLAY_HELPERS:
    print(
        f"{'[OK]' if hasattr(self_play_module, name) else '[MISSING]'} "
        f"{name}"
    )

if missing_helpers:
    raise AttributeError(
        "Self-play export is missing:\n"
        + "\n".join(missing_helpers)
    )

create_test_battle = self_play_module.create_test_battle
create_battle_agent = self_play_module.create_battle_agent
run_match = self_play_module.run_match
run_match_series = self_play_module.run_match_series

print()
print("create_test_battle signature:")
print(inspect.signature(create_test_battle))

print()
print("create_battle_agent signature:")
print(inspect.signature(create_battle_agent))

print()
print("run_match signature:")
print(inspect.signature(run_match))

print()
print("run_match_series signature:")
print(inspect.signature(run_match_series))

print("\nExisting self-play helpers loaded.")


# # Cell 11 — Create fresh agents and the first Notebook 27 battle

# In[18]:


# Cell 11 — Create fresh agents and a full-game battle

from src import TournamentMatch

player_agent = create_battle_agent(
    seed=20260727,
)

opponent_agent = create_battle_agent(
    seed=20260728,
)

initial_battle = create_test_battle(
    starting_player="Player",
)

full_game_match = TournamentMatch(
    player_agent=player_agent,
    opponent_agent=opponent_agent,
    player_name="Team Jesus AI",
    opponent_name="Baseline Opponent",
    search_depth=6,
    max_turns=100,
)

print("Player agent:", type(player_agent))
print("Opponent agent:", type(opponent_agent))
print("Starting side:", initial_battle.current_player)
print("Player active:", initial_battle.player.active.card.get("name"))
print("Opponent active:", initial_battle.opponent.active.card.get("name"))
print("Maximum turns:", full_game_match.max_turns)

assert player_agent is not opponent_agent
assert initial_battle.current_player == "Player"

print("\nFresh full-game match created.")


# # Cell 12 — Run the first true full-game match

# In[19]:


# Cell 12 — Run the first true full-game match

started = time.perf_counter()

first_match_result = run_two_agent_match(
    full_game_match,
    initial_battle,
    verbose=True,
)

first_match_seconds = (
    time.perf_counter() - started
)

print()
print("=" * 72)
print("NOTEBOOK 27 — FIRST FULL-GAME RESULT")
print("=" * 72)

print("Winner:", first_match_result.winner)
print("Turns:", first_match_result.turns)
print("Final score:", first_match_result.final_score)
print("Elapsed seconds:", first_match_seconds)

assert first_match_result.winner in {
    "Team Jesus AI",
    "Baseline Opponent",
    "Draw",
}

assert 1 <= first_match_result.turns <= 100

print("\nFirst full-game simulation passed.")


# # Cell 13 — Inspect and save the match transcript

# In[20]:


# Cell 13 — Inspect and save the match transcript

first_transcript = (
    first_match_result.transcript
    if hasattr(first_match_result, "transcript")
    else create_battle_transcript(first_match_result)
)

print(first_transcript)

TRANSCRIPT_FILE = (
    REPORT_DIR
    / "first_full_game_transcript.txt"
)

TRANSCRIPT_FILE.write_text(
    str(first_transcript),
    encoding="utf-8",
)

print()
print("Transcript saved:", TRANSCRIPT_FILE)

assert TRANSCRIPT_FILE.is_file()
assert TRANSCRIPT_FILE.stat().st_size > 0

print("\nFirst full-game transcript saved successfully.")


# # Cell 14 — Run an alternating-start match series

# In[21]:


# Cell 14 — Run a full-game match series

SERIES_MATCHES = 20

series_started = time.perf_counter()

series_results, series_statistics = run_match_series(
    full_game_match,
    number_of_matches=SERIES_MATCHES,
    alternate_starting_side=True,
)

series_elapsed_seconds = (
    time.perf_counter() - series_started
)

print("=" * 72)
print("NOTEBOOK 27 — FULL-GAME SERIES")
print("=" * 72)

print("Matches played:", series_statistics.matches_played)
print("Team Jesus AI wins:", series_statistics.player_wins)
print("Baseline Opponent wins:", series_statistics.opponent_wins)
print("Draws:", series_statistics.draws)
print("Average turns:", series_statistics.average_turns)
print("Average score:", series_statistics.average_score)
print("Elapsed seconds:", series_elapsed_seconds)

assert series_statistics.matches_played == SERIES_MATCHES

assert (
    series_statistics.player_wins
    + series_statistics.opponent_wins
    + series_statistics.draws
    == SERIES_MATCHES
)

print("\nFull-game series completed successfully.")


# # Cell 15 — Create the series results table

# In[23]:


# Cell 15 — Create the full-game series DataFrame

import pandas as pd

series_rows = []

for match_number, result in enumerate(
    series_results,
    start=1,
):
    starting_side = (
        "Player"
        if match_number % 2 == 1
        else "Opponent"
    )

    series_rows.append(
        {
            "match_number": match_number,
            "starting_side": starting_side,
            "winner": result.winner,
            "turns": result.turns,
            "final_score": result.final_score,
            "transcript_characters": len(
                str(result.transcript)
            ),
        }
    )

series_df = pd.DataFrame(series_rows)

display(series_df)

print()
print("Winner counts:")
print(series_df["winner"].value_counts())

print()
print("Turn statistics:")
print(series_df["turns"].describe())

assert len(series_df) == SERIES_MATCHES
assert series_df["turns"].between(1, 100).all()

print("\nFull-game series table validated.")


# # Cell 16 — Save the full-game series reports

# In[25]:


# Cell 16 — Save full-game simulation reports

SERIES_CSV = (
    REPORT_DIR
    / "full_game_series.csv"
)

SERIES_JSON = (
    REPORT_DIR
    / "full_game_summary.json"
)

series_df.to_csv(
    SERIES_CSV,
    index=False,
)

series_summary = {
    "project": "PTCG AI Battle Challenge",
    "team": "Team Jesus",
    "simulation_scope": (
        "Simplified full-game agent-versus-agent model"
    ),
    "matches_played": (
        series_statistics.matches_played
    ),
    "team_jesus_wins": (
        series_statistics.player_wins
    ),
    "baseline_wins": (
        series_statistics.opponent_wins
    ),
    "draws": series_statistics.draws,
    "average_turns": (
        series_statistics.average_turns
    ),
    "average_score": (
        series_statistics.average_score
    ),
    "elapsed_seconds": series_elapsed_seconds,
}

SERIES_JSON.write_text(
    json.dumps(
        series_summary,
        indent=4,
    ),
    encoding="utf-8",
)

print("Series CSV:", SERIES_CSV)
print("Summary JSON:", SERIES_JSON)

assert SERIES_CSV.is_file()
assert SERIES_JSON.is_file()
assert SERIES_CSV.stat().st_size > 0
assert SERIES_JSON.stat().st_size > 0

print("\nFull-game simulation reports saved.")


# # Cell 17 — Validate starting-side fairness

# In[26]:


# Cell 17 — Compare results by starting side

starting_side_summary = (
    series_df
    .groupby("starting_side")
    .agg(
        matches=("match_number", "count"),
        average_turns=("turns", "mean"),
        average_score=("final_score", "mean"),
        minimum_turns=("turns", "min"),
        maximum_turns=("turns", "max"),
    )
    .reset_index()
)

display(starting_side_summary)

winner_by_start = pd.crosstab(
    series_df["starting_side"],
    series_df["winner"],
)

print()
print("Winner distribution by starting side:")
display(winner_by_start)

assert starting_side_summary["matches"].sum() == SERIES_MATCHES
assert set(starting_side_summary["starting_side"]) == {
    "Player",
    "Opponent",
}

print("\nStarting-side comparison validated.")


# # Cell 18 — Final Notebook 27 summary

# In[28]:


# Cell 18 — Final Notebook 27 summary

matches_played = series_summary["matches_played"]

team_jesus_win_rate = series_summary.get(
    "team_jesus_win_rate",
    series_summary["team_jesus_wins"] / matches_played,
)

baseline_win_rate = series_summary.get(
    "baseline_win_rate",
    series_summary["baseline_wins"] / matches_played,
)

draw_rate = series_summary.get(
    "draw_rate",
    series_summary["draws"] / matches_played,
)

print("=" * 72)
print("Notebook 27 — Full Game Simulation")
print("=" * 72)
print()

print("Project:", series_summary["project"])
print("Team:", series_summary["team"])
print(
    "Simulation scope:",
    series_summary["simulation_scope"],
)

print()
print("First-match winner:", first_match_result.winner)
print("First-match turns:", first_match_result.turns)
print("First-match score:", first_match_result.final_score)

print()
print("Series matches:", matches_played)
print("Team Jesus AI wins:", series_summary["team_jesus_wins"])
print("Baseline wins:", series_summary["baseline_wins"])
print("Draws:", series_summary["draws"])
print("Team Jesus win rate:", f"{team_jesus_win_rate:.1%}")
print("Baseline win rate:", f"{baseline_win_rate:.1%}")
print("Draw rate:", f"{draw_rate:.1%}")
print("Average turns:", series_summary["average_turns"])
print("Average score:", series_summary["average_score"])

print()
print("Transcript:", TRANSCRIPT_FILE.name)
print("Series CSV:", SERIES_CSV.name)
print("Summary JSON:", SERIES_JSON.name)

assert matches_played == SERIES_MATCHES
assert (
    series_summary["team_jesus_wins"]
    + series_summary["baseline_wins"]
    + series_summary["draws"]
    == matches_played
)

print()
print("FULL-GAME SIMULATION VALIDATED")
print("NOTEBOOK 27 COMPLETED SUCCESSFULLY")


# In[ ]:




