#!/usr/bin/env python
# coding: utf-8

# # Notebook 29 — Replay Dataset Analysis and Opponent Diversity
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 28 generated 100 self-play games and 350 structured replay steps.
# 
# Notebook 29 analyzes that replay dataset, measures repetition and class imbalance, compares outcomes by starting side, and prepares more diverse opponent configurations for future self-play training.
# 
# ## Objectives
# 
# 1. Load Notebook 28 artifacts.
# 2. Validate replay dataset integrity.
# 3. Analyze winner imbalance.
# 4. Measure move repetition.
# 5. Compare game length and score distributions.
# 6. Examine starting-side effects.
# 7. Identify duplicated training patterns.
# 8. Define diverse opponent profiles.
# 9. Generate an opponent-diversity experiment plan.
# 10. Export analysis reports for Notebook 30.

# # Cell 2 — Imports

# In[1]:


from __future__ import annotations

import json
import sys

from pathlib import Path
from typing import Any

import pandas as pd

print("Python:", sys.version)
print("Notebook 29 initialized.")


# # Cell 3 — Locate the Project and Reports

# In[2]:


from pathlib import Path


def find_project_root(
    start: Path | None = None,
) -> Path:
    current = (start or Path.cwd()).resolve()

    markers = {
        "notebooks",
        "scripts",
        "src",
        "reports",
    }

    for candidate in [
        current,
        *current.parents,
    ]:
        if all(
            (candidate / marker).exists()
            for marker in markers
        ):
            return candidate

    raise FileNotFoundError(
        "Project root not found."
    )


PROJECT_ROOT = find_project_root()

NOTEBOOK28_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook28"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook29"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Project root:", PROJECT_ROOT)
print("Notebook 28 reports:", NOTEBOOK28_REPORT_DIR)
print("Notebook 29 reports:", REPORT_DIR)

assert NOTEBOOK28_REPORT_DIR.exists()


# # Cell 4 — Load Notebook 28 Artifacts

# In[3]:


DATASET_FILE = NOTEBOOK28_REPORT_DIR / "self_play_dataset.csv"
STATS_FILE = NOTEBOOK28_REPORT_DIR / "self_play_statistics.csv"
SUMMARY_FILE = NOTEBOOK28_REPORT_DIR / "self_play_summary.json"

assert DATASET_FILE.exists(), DATASET_FILE
assert STATS_FILE.exists(), STATS_FILE
assert SUMMARY_FILE.exists(), SUMMARY_FILE

dataset = pd.read_csv(DATASET_FILE)
statistics = pd.read_csv(STATS_FILE)

with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    summary = json.load(f)

print("=" * 60)
print("Notebook 28 Artifacts Loaded")
print("=" * 60)

print(f"Dataset rows : {len(dataset):,}")
print(f"Statistics rows : {len(statistics):,}")
print()

print("Summary")
for key, value in summary.items():
    print(f"{key:20} {value}")



# # Cell 5 — Dataset Overview

# In[4]:


print("=" * 60)
print("Dataset Overview")
print("=" * 60)

display(dataset.head())

print()

print(dataset.info())

print()

print(dataset.describe(include="all"))


# # Cell 6 — Data Quality Report

# In[5]:


print("=" * 70)
print("DATA QUALITY REPORT")
print("=" * 70)

print(f"Games                : {dataset.game_id.nunique():>6}")
print(f"Replay steps         : {len(dataset):>6}")
print(f"Unique winners       : {dataset.winner.nunique():>6}")
print(f"Unique players       : {dataset.player.nunique():>6}")
print(f"Unique moves         : {dataset.chosen_move.nunique():>6}")
print(f"Unique states        : {dataset.state.nunique():>6}")
print(f"Missing values       : {dataset.isna().sum().sum():>6}")

print()

print("Winner Distribution")
print(dataset["winner"].value_counts())

print()

print("Player Distribution")
print(dataset["player"].value_counts())

print()

print("Move Distribution")
print(dataset["chosen_move"].value_counts())


# # Cell 7 — Replay Diversity Metrics

# In[6]:


print("=" * 70)
print("Replay Diversity Metrics")
print("=" * 70)

games = dataset.game_id.nunique()

steps_per_game = len(dataset) / games

unique_states = dataset.state.nunique()

unique_moves = dataset.chosen_move.nunique()

print(f"Average steps/game : {steps_per_game:.2f}")
print(f"Unique states      : {unique_states}")
print(f"Unique moves       : {unique_moves}")

print()

state_diversity = unique_states / len(dataset)

move_diversity = unique_moves / len(dataset)

print(f"State diversity : {state_diversity:.3f}")
print(f"Move diversity  : {move_diversity:.3f}")

print()

if state_diversity < 0.10:
    print("⚠ Low state diversity")

if move_diversity < 0.05:
    print("⚠ Low move diversity")


# # Cell 8 — Analyze Game Length

# In[7]:


print("=" * 70)
print("Game Length Analysis")
print("=" * 70)

turns_per_game = (
    dataset.groupby("game_id")["turn"]
    .max()
    .sort_values()
)

display(turns_per_game.describe())

print()

print("Turn Frequency")
print(turns_per_game.value_counts().sort_index())

print()

print(f"Shortest game : {turns_per_game.min()} turns")
print(f"Longest game  : {turns_per_game.max()} turns")
print(f"Average length: {turns_per_game.mean():.2f} turns")


# # Cell 9 — Move Frequency Analysis

# In[8]:


print("=" * 70)
print("Move Frequency Analysis")
print("=" * 70)

move_counts = (
    dataset["chosen_move"]
    .value_counts()
    .rename_axis("Move")
    .reset_index(name="Count")
)

display(move_counts)

move_counts["Percent"] = (
    move_counts["Count"]
    / len(dataset)
    * 100
)

display(move_counts)


# # Cell 10 — Player Analysis

# In[9]:


print("=" * 70)
print("Player Analysis")
print("=" * 70)

player_summary = (
    dataset
    .groupby("player")
    .agg(
        Steps=("turn", "count"),
        AvgEvaluation=("evaluation", "mean"),
        AvgTurn=("turn", "mean"),
    )
)

display(player_summary)


# # Cell 11 — Duplicate State Analysis

# In[10]:


print("=" * 70)
print("Duplicate State Analysis")
print("=" * 70)

state_counts = dataset["state"].value_counts()

display(state_counts)

duplicate_states = (state_counts > 1).sum()

print()

print(f"Unique states      : {dataset.state.nunique()}")
print(f"Duplicate patterns : {duplicate_states}")
print(f"Most common state appears {state_counts.max()} times")


# # Cell 12 — Match-Count Experiment Plan

# In[11]:


experiment_plan = pd.DataFrame(
    [
        {
            "experiment": "smoke_test",
            "num_matches": 100,
            "purpose": "Validate pipeline and exports",
        },
        {
            "experiment": "analysis_run",
            "num_matches": 1_000,
            "purpose": "Measure stability and repetition",
        },
        {
            "experiment": "full_training_run",
            "num_matches": 2_000,
            "purpose": "Generate training data after opponent diversity",
        },
    ]
)

display(experiment_plan)


# # Cell 13 — Opponent Allocation Plan

# In[12]:


opponent_allocation = pd.DataFrame(
    [
        {
            "opponent_profile": "random",
            "num_matches": 400,
            "purpose": "Increase action and state variety",
        },
        {
            "opponent_profile": "aggressive",
            "num_matches": 400,
            "purpose": "Test damage-first strategies",
        },
        {
            "opponent_profile": "defensive",
            "num_matches": 400,
            "purpose": "Test survival and resource conservation",
        },
        {
            "opponent_profile": "greedy",
            "num_matches": 400,
            "purpose": "Test immediate-value decisions",
        },
        {
            "opponent_profile": "search_based",
            "num_matches": 400,
            "purpose": "Test stronger calculated play",
        },
    ]
)

assert opponent_allocation["num_matches"].sum() == 2_000

display(opponent_allocation)

print(
    "Total planned matches:",
    opponent_allocation["num_matches"].sum(),
)


# # Cell 14 — Create the Notebook 29 Analysis Summary

# In[13]:


analysis_summary = {
    "source_notebook": 28,
    "total_games": int(dataset["game_id"].nunique()),
    "total_steps": int(len(dataset)),
    "unique_winners": int(dataset["winner"].nunique()),
    "unique_players": int(dataset["player"].nunique()),
    "unique_moves": int(dataset["chosen_move"].nunique()),
    "unique_states": int(dataset["state"].nunique()),
    "state_diversity": float(state_diversity),
    "move_diversity": float(move_diversity),
    "average_steps_per_game": float(steps_per_game),
    "duplicate_state_patterns": int(duplicate_states),
    "most_common_state_frequency": int(state_counts.max()),
    "recommended_analysis_matches": 1_000,
    "recommended_training_matches": 2_000,
    "recommendation": (
        "Introduce opponent diversity before generating "
        "the 2,000-game training dataset."
    ),
}

for key, value in analysis_summary.items():
    print(f"{key:32}: {value}")


# # Cell 15 — Export Notebook 29 Reports

# In[14]:


analysis_summary_file = (
    REPORT_DIR
    / "replay_analysis_summary.json"
)

experiment_plan_file = (
    REPORT_DIR
    / "match_count_experiment_plan.csv"
)

opponent_allocation_file = (
    REPORT_DIR
    / "opponent_allocation_plan.csv"
)

move_frequency_file = (
    REPORT_DIR
    / "move_frequency.csv"
)

state_frequency_file = (
    REPORT_DIR
    / "state_frequency.csv"
)

with open(
    analysis_summary_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        analysis_summary,
        file,
        indent=2,
    )

experiment_plan.to_csv(
    experiment_plan_file,
    index=False,
)

opponent_allocation.to_csv(
    opponent_allocation_file,
    index=False,
)

move_counts.to_csv(
    move_frequency_file,
    index=False,
)

state_counts.rename("count").to_csv(
    state_frequency_file,
)

print("Exported:")
print(analysis_summary_file)
print(experiment_plan_file)
print(opponent_allocation_file)
print(move_frequency_file)
print(state_frequency_file)


# # Cell 16 — Validate Exports

# In[15]:


expected_files = [
    analysis_summary_file,
    experiment_plan_file,
    opponent_allocation_file,
    move_frequency_file,
    state_frequency_file,
]

missing_files = [
    path
    for path in expected_files
    if not path.exists()
]

assert not missing_files, missing_files

for path in expected_files:
    print(
        f"[OK] {path.name:<36} "
        f"{path.stat().st_size:>8,} bytes"
    )

print()
print(
    "All Notebook 29 reports exported successfully."
)


# # Cell 17 — Configurable Benchmark Configuration

# In[16]:


BENCHMARK_MATCH_COUNTS = [
    100,
    1_000,
    2_000,
]

print("=" * 70)
print("SELF-PLAY SCALE BENCHMARK")
print("=" * 70)

for match_count in BENCHMARK_MATCH_COUNTS:
    print(f"Planned benchmark: {match_count:,} matches")

assert BENCHMARK_MATCH_COUNTS == [100, 1_000, 2_000]


# ## Cell 18 — Inspect the Existing Self-Play API
# 
# #### Before running thousands of games, let’s confirm the exact reusable functions and configuration classes exported by Notebook 28.

# In[17]:


import inspect
import sys

from pathlib import Path


def find_project_root(
    start: Path | None = None,
) -> Path:
    current = (start or Path.cwd()).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        if (
            (candidate / "src").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Project root could not be located."
    )


PROJECT_ROOT = find_project_root()

project_root_str = str(PROJECT_ROOT)

if project_root_str not in sys.path:
    sys.path.insert(
        0,
        project_root_str,
    )

print("Project root:", PROJECT_ROOT)
print("Python path updated:", project_root_str in sys.path)

import src.training.self_play as self_play_module


print("=" * 70)
print("SELF-PLAY MODULE INSPECTION")
print("=" * 70)

public_names = [
    name
    for name in dir(self_play_module)
    if not name.startswith("_")
]

print("Public names:")
for name in public_names:
    print(f" - {name}")

print()

if hasattr(
    self_play_module,
    "run_self_play_session",
):
    run_self_play_session = (
        self_play_module.run_self_play_session
    )

    print("run_self_play_session signature:")
    print(
        inspect.signature(
            run_self_play_session
        )
    )

    print()
    print("Function located successfully.")
else:
    raise ImportError(
        "run_self_play_session was not found "
        "in src.training.self_play."
    )


# # Cell 19 — Inspect Configuration Classes

# In[18]:


possible_config_names = [
    "SelfPlayConfig",
    "TrainingConfig",
    "SelfPlaySessionConfig",
]

available_config_classes = {}

for name in possible_config_names:
    value = getattr(
        self_play_module,
        name,
        None,
    )

    if value is not None:
        available_config_classes[name] = value

print("=" * 70)
print("AVAILABLE CONFIGURATION CLASSES")
print("=" * 70)

if available_config_classes:
    for name, config_class in available_config_classes.items():
        print(f"{name}:")
        print(inspect.signature(config_class))
        print()
else:
    print(
        "No expected configuration class name "
        "was found."
    )

    print()
    print(
        "The benchmark runner will be adapted "
        "to the actual function signature above."
    )


# # Cell 20 — Benchmark Runner

# In[19]:


import inspect

import src.training.self_play as self_play_module


print("=" * 70)
print("SELF-PLAY FACTORY DISCOVERY")
print("=" * 70)

candidate_modules = [
    self_play_module,
]

candidate_names = [
    "match_factory",
    "initial_state_factory",
    "build_match_factory",
    "build_initial_state",
    "create_match",
    "create_initial_state",
]

for module in candidate_modules:
    print(f"\nModule: {module.__name__}")

    for name in candidate_names:
        value = getattr(module, name, None)

        if value is not None:
            print(f" - {name}: {value}")

            if callable(value):
                try:
                    print(
                        "   signature:",
                        inspect.signature(value),
                    )
                except (TypeError, ValueError):
                    pass


# # Cell 21

# In[20]:


from pathlib import Path

SELF_PLAY_FILE = (
    PROJECT_ROOT
    / "src"
    / "training"
    / "self_play.py"
)

print("Self-play file:", SELF_PLAY_FILE)
print("Exists:", SELF_PLAY_FILE.exists())

print()
print(SELF_PLAY_FILE.read_text(encoding="utf-8"))


# ## Cell - 22

# In[21]:


import pandas as pd


benchmark_plan = pd.DataFrame(
    [
        {
            "phase": "Baseline validation",
            "games": 100,
            "status": "Completed in Notebook 28",
        },
        {
            "phase": "Scaling benchmark",
            "games": 1_000,
            "status": "Planned after opponent diversity",
        },
        {
            "phase": "Full training benchmark",
            "games": 2_000,
            "status": "Planned after opponent diversity",
        },
    ]
)

display(benchmark_plan)


# ## Cell 23

# In[22]:


benchmark_plan_file = (
    REPORT_DIR
    / "benchmark_execution_plan.csv"
)

benchmark_plan.to_csv(
    benchmark_plan_file,
    index=False,
)

print("[OK]", benchmark_plan_file)


# ## Cell 24

# In[23]:


expected_report_names = [
    "replay_analysis_summary.json",
    "match_count_experiment_plan.csv",
    "opponent_allocation_plan.csv",
    "move_frequency.csv",
    "state_frequency.csv",
    "benchmark_execution_plan.csv",
]

missing_reports = [
    name
    for name in expected_report_names
    if not (REPORT_DIR / name).exists()
]

assert not missing_reports, missing_reports

for name in expected_report_names:
    path = REPORT_DIR / name

    print(
        f"[OK] {name:<38} "
        f"{path.stat().st_size:>8,} bytes"
    )

print()
print("All six Notebook 29 reports validated.")


# In[ ]:




