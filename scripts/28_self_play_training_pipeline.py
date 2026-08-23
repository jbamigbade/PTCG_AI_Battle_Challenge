from __future__ import annotations

#!/usr/bin/env python
# coding: utf-8

# # Cell 1 - Overview 

# # Notebook 28 â€” Self-Play Training Pipeline
# 
# ## The PokÃ©mon Company â€” PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# Notebook 27 validated that two production agents can complete full PokÃ©mon battles.
# 
# Notebook 28 begins the learning pipeline by generating self-play data that can later be used for policy improvement, evaluation, reinforcement learning, and behavior cloning.
# 
# ## Objectives
# 
# 1. Create two independent production agents.
# 2. Execute repeated self-play matches.
# 3. Record every important decision.
# 4. Capture replay trajectories.
# 5. Build a reusable training dataset.
# 6. Generate self-play statistics.
# 7. Export replay data for future learning.
# 8. Validate deterministic execution.
# 9. Measure long-run stability.
# 10. Prepare the policy-improvement stage.
# 
# ## Expected Outputs
# 
# - self_play_dataset.csv
# - self_play_summary.json
# - self_play_statistics.csv
# - training_examples.pkl

# # Cell 2

# In[1]:



import json
import pickle
import random
import sys
import time
import uuid

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

print("Python:", sys.version)
print("Notebook 28 initialized.")


# # Cell 3 - Locate the project

# In[2]:


from pathlib import Path

def find_project_root(
    start: Path | None = None,
):
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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "notebook28"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(PROJECT_ROOT)


# # Cell 4 - Import the production tournament engine

# In[3]:


PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT_STR,
    )

from src import (
    TournamentMatch,
    TournamentStatistics,
)

from src.tournament.runner import (
    run_two_agent_match,
)

print("Production tournament engine imported.")


# # Cell 5 â€” Self-Play Configuration
# 
# ### Building a configuration object that controls the experiment from one place.

# In[4]:



from dataclasses import dataclass


@dataclass(slots=True)
class SelfPlayConfig:
    """
    Configuration for Notebook 28 self-play experiments.
    """

    num_matches: int = 100

    random_seed: int = 42

    alternate_starting_player: bool = True

    save_replays: bool = True

    save_statistics: bool = True

    export_dataset: bool = True

    verbose: bool = False


CONFIG = SelfPlayConfig()

random.seed(CONFIG.random_seed)

print(CONFIG)


# # Instead of hunting through the notebook for values like:
# 
# #### num_matches = 100
# 
# #### or
# 
# #### seed = 42
# 
# #### Above everything lives in one configuration object. Later, if you want to run 1,000, 5,000, or 10,000 self-play games, I'll only need to change:
# 
# #### CONFIG.num_matches

# ## Cell 6 â€” Replay Data Structures
# 
# #### This cell defines the data model that will capture every decision made during self-play. These records become the foundation for future policy improvement and training.
# 
# #### Each ReplayStep stores:
# 
# 1. The game state
# 2. The legal moves available
# 3. The move the AI selected
# 4. The search evaluation
# 5. The search depth

# In[5]:


# ============================================================
# Cell 6 - Replay Data Structures
# ============================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReplayStep:
    """
    One decision made during a battle.
    """

    game_id: str

    turn: int

    player: str

    state: dict[str, Any]

    legal_moves: list[str]

    chosen_move: str

    evaluation: float

    search_depth: int


@dataclass(slots=True)
class ReplayGame:
    """
    Complete replay of one self-play game.
    """

    game_id: str

    winner: str = ""

    total_turns: int = 0

    final_score: float = 0.0

    steps: list[ReplayStep] = field(default_factory=list)


print("Replay data structures created.")


# ## Cell 7 â€” Replay Recorder
# 
# #### Notebook 28, every self-play game will flow through this recorder:
# 
# Battle Engine
#       â”‚
#       â–¼
# ReplayGame
#       â”‚
#       â–¼
# ReplayRecorder
#       â”‚
#       â–¼
# Pandas DataFrame
#       â”‚
#       â–¼
# CSV / JSON / PKL
# 
# #### That means the self-play loop (Cell 8) only has one job: play games. The recorder handles collecting and organizing the data.

# In[6]:


# ============================================================
# Cell 7 - Replay Recorder
# ============================================================

from dataclasses import asdict
from typing import Iterable


class ReplayRecorder:
    """
    Records self-play games and converts them into
    training-ready datasets.
    """

    def __init__(self):
        self.games: list[ReplayGame] = []

    def add_game(self, game: ReplayGame) -> None:
        self.games.append(game)

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def total_steps(self) -> int:
        return sum(len(game.steps) for game in self.games)

    def iter_steps(self) -> Iterable[ReplayStep]:
        for game in self.games:
            yield from game.steps

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert replay steps into a pandas DataFrame.
        """

        rows = []

        for step in self.iter_steps():

            rows.append(
                {
                    "game_id": step.game_id,
                    "turn": step.turn,
                    "player": step.player,
                    "chosen_move": step.chosen_move,
                    "evaluation": step.evaluation,
                    "search_depth": step.search_depth,
                    "num_legal_moves": len(step.legal_moves),
                    "state": step.state,
                }
            )

        return pd.DataFrame(rows)


recorder = ReplayRecorder()

print("Replay recorder initialized.")
print(f"Games: {recorder.total_games}")
print(f"Steps: {recorder.total_steps}")


# ## Cell 8 â€” Inspect Available Tournament Functions
# 
# #### Proposed architecture
# 
# Notebook 27
#     â”‚
#     â–¼
# run_two_agent_match(...)
#     â”‚
#     â–¼
# Battle Result
#     â”‚
#     â–¼
# Notebook 28
# ReplayRecorder
#     â”‚
#     â–¼
# Training Dataset
# 
# #### Notebook 28 shouldn't know how to play a game, it should only know how to record the results.

# In[7]:


# ============================================================
# Cell 8 - Inspect Tournament API
# ============================================================

import inspect

print("=" * 70)
print("run_two_agent_match")
print("=" * 70)

print(inspect.signature(run_two_agent_match))

print("\n")

print("=" * 70)
print("TournamentMatch")
print("=" * 70)

print(inspect.signature(TournamentMatch))

print("\n")

print("=" * 70)
print("TournamentStatistics")
print("=" * 70)

print(inspect.signature(TournamentStatistics))


# # Cell 9 â€” Inspect TournamentResult

# In[8]:


# ============================================================
# Cell 9 - Inspect TournamentResult
# ============================================================

import inspect

from dataclasses import fields, is_dataclass
from typing import get_type_hints


# Resolve string annotations into actual Python classes.
type_hints = get_type_hints(run_two_agent_match)

result_type = type_hints["return"]


print("=" * 70)
print("TournamentResult Type")
print("=" * 70)
print(result_type)


print("\n" + "=" * 70)
print("TournamentResult Constructor")
print("=" * 70)
print(inspect.signature(result_type))


print("\n" + "=" * 70)
print("TournamentResult Fields")
print("=" * 70)

if is_dataclass(result_type):
    for result_field in fields(result_type):
        print(
            f"{result_field.name:20} : "
            f"{result_field.type}"
        )
else:
    print("TournamentResult is not a dataclass.")


print("\nTournamentResult inspection completed.")


# ##  Cell 10 â€“ Self-Play Session Manager
# 
# #### This class will manage many self-play games and store them with the recorder.

# In[9]:


# ============================================================
# Cell 10 - Self-Play Session
# ============================================================

import uuid
from dataclasses import dataclass, field


@dataclass(slots=True)
class SelfPlaySession:
    """
    Executes and tracks multiple self-play games.
    """

    config: SelfPlayConfig

    recorder: ReplayRecorder

    statistics: TournamentStatistics = field(
        default_factory=TournamentStatistics
    )

    session_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    completed_games: int = 0


session = SelfPlaySession(
    config=CONFIG,
    recorder=recorder,
)

print("Session ID:", session.session_id)
print("Completed games:", session.completed_games)
print("Target games:", session.config.num_matches)


# # Cell 11 â€” Validate Training Replay Module:

# In[10]:


# ============================================================
# Cell 11 - Validate Training Replay Module
# ============================================================

from src.training import (
    ReplayGame as ProductionReplayGame,
    ReplayRecorder as ProductionReplayRecorder,
    ReplayStep as ProductionReplayStep,
)

production_recorder = ProductionReplayRecorder()

print("Training replay module imported.")
print("Games:", production_recorder.total_games)
print("Steps:", production_recorder.total_steps)


# # Cell 12 

# In[11]:


# ============================================================
# Cell 12 - Validate Dataset Module
# ============================================================

from src.training import replay_to_dataframe

empty_df = replay_to_dataframe(production_recorder)

print(empty_df)

print()

print("Rows:", len(empty_df))
print("Columns:", len(empty_df.columns))


# # Notebook Cell 13 â€” Validate Statistics Module

# In[12]:


# ============================================================
# Cell 13 - Validate Statistics Module
# ============================================================

from src.training import calculate_statistics

empty_statistics = calculate_statistics(
    production_recorder
)

print(empty_statistics)
print()
print(empty_statistics.to_dict())


# # Cell 14 â€” Validate Self-Play Module

# In[13]:


# ============================================================
# Cell 14 - Validate Self-Play Module
# ============================================================

from src.training.self_play import (
    SelfPlayRunSummary,
    run_self_play_session,
)

print("Self-play module imported.")
print("Summary type:", SelfPlayRunSummary)
print("Runner:", run_self_play_session.__name__)


# # Cell 15 â€” Inspect Notebook 27 Helpers

# In[14]:


# ============================================================
# Cell 15 - Inspect Existing Battle Setup Helpers
# ============================================================

import importlib.util
import inspect

notebook27_script = (
    PROJECT_ROOT
    / "scripts"
    / "27_full_game_simulation.py"
)

spec = importlib.util.spec_from_file_location(
    "notebook27_helpers",
    notebook27_script,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Unable to load: {notebook27_script}"
    )

notebook27_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notebook27_helpers)

helper_names = [
    name
    for name in dir(notebook27_helpers)
    if (
        name.startswith("create_")
        or name.startswith("build_")
        or name.startswith("make_")
        or name.startswith("run_")
    )
]

print("Reusable Notebook 27 helpers:\n")

for name in helper_names:
    obj = getattr(notebook27_helpers, name)

    if callable(obj):
        try:
            print(f"- {name}{inspect.signature(obj)}")
        except (TypeError, ValueError):
            print(f"- {name}")


# # Cell 16 â€” Build Factory Functions

# In[15]:


# ============================================================
# Cell 16 - Self-Play Factory Functions
# ============================================================

player_agent = notebook27_helpers.create_battle_agent(
    seed=CONFIG.random_seed,
)

opponent_agent = notebook27_helpers.create_battle_agent(
    seed=CONFIG.random_seed + 1,
)


def match_factory(game_index: int):

    return TournamentMatch(
        player_agent=player_agent,
        opponent_agent=opponent_agent,
        player_name="TrainingAgent",
        opponent_name="TrainingOpponent",
    )


def initial_state_factory(game_index: int):

    starting_player = (
        "Player"
        if (
            not CONFIG.alternate_starting_player
            or game_index % 2 == 0
        )
        else "Opponent"
    )

    return notebook27_helpers.create_test_battle(
        starting_player=starting_player,
    )


print("Factory functions created.")


# # Cell 17 â€” Run First Self-Play Session

# In[16]:


# ============================================================
# Cell 17 - Run One Validation Game
# ============================================================

summary = run_self_play_session(
    match_factory=match_factory,
    initial_state_factory=initial_state_factory,
    num_games=1,
    verbose=False,
)

print(summary)

print()

print("Games recorded:",
      summary.recorder.total_games)

print("Steps recorded:",
      summary.recorder.total_steps)


# # Cell 18 â€” Inspect the Recorded Game

# In[17]:


# ============================================================
# Cell 18 - Inspect Recorded Self-Play Game
# ============================================================

if "summary" not in globals():
    raise RuntimeError(
        "Run Cell 17 first to create the self-play summary."
    )

if summary.recorder.total_games != 1:
    raise RuntimeError(
        "Expected exactly one recorded validation game."
    )

recorded_game = next(
    summary.recorder.iter_games()
)

print("=" * 70)
print("Recorded Self-Play Game")
print("=" * 70)

print("Game ID:", recorded_game.game_id)
print("Winner:", recorded_game.winner)
print("Total turns:", recorded_game.total_turns)
print("Final score:", recorded_game.final_score)
print("Recorded steps:", len(recorded_game.steps))
print("Transcript characters:", len(recorded_game.transcript))

print("\nTranscript preview:")
print("-" * 70)

transcript_preview = recorded_game.transcript[:1000]

print(
    transcript_preview
    if transcript_preview
    else "[No transcript recorded]"
)

assert recorded_game.game_id
assert recorded_game.winner
assert recorded_game.total_turns > 0
assert recorded_game.transcript.strip()

print("\nValidation passed: game-level replay data is complete.")


# ## Cell 19 â€” Parse Transcript into Replay Steps
# 
# #### This converts each turn in the battle transcript into a structured ReplayStep.

# In[18]:


# ============================================================
# Cell 19 - Parse Transcript into Replay Steps
# ============================================================

import re

from src.training.replay import ReplayStep


TURN_PATTERN = re.compile(
    r"""
    Turn\s+(?P<turn>\d+):\s*
    (?P<player>Player|Opponent)\s*
    [â€”â€“-]\s*
    (?P<pokemon>.+?)\s+used\s+
    (?P<move>.+?)\s*\n

    \s*Damage:\s*(?P<damage>-?\d+(?:\.\d+)?)\s*\n
    \s*Search\s+score:\s*(?P<evaluation>-?\d+(?:\.\d+)?)\s*\n
    \s*Search\s+depth:\s*(?P<search_depth>\d+)\s*\n
    \s*Nodes\s+searched:\s*(?P<nodes_searched>\d+)\s*\n

    \s*HP\s+after\s+move\s*
    [â€”â€“-]\s*
    Player:\s*(?P<player_hp>-?\d+(?:\.\d+)?),\s*
    Opponent:\s*(?P<opponent_hp>-?\d+(?:\.\d+)?)\s*\n

    \s*Next\s+side:\s*(?P<next_side>Player|Opponent)
    """,
    flags=re.VERBOSE | re.IGNORECASE,
)


def parse_transcript_to_steps(
    *,
    game_id: str,
    transcript: str,
) -> list[ReplayStep]:
    """
    Parse the tournament transcript into structured replay steps.

    The transcript currently reports the selected move, but not every
    legal move that was available. Therefore legal_moves remains empty
    rather than inserting inaccurate information.
    """

    if not game_id:
        raise ValueError("game_id cannot be empty.")

    if not transcript.strip():
        return []

    replay_steps: list[ReplayStep] = []

    for match in TURN_PATTERN.finditer(transcript):
        values = match.groupdict()

        state = {
            "pokemon": values["pokemon"].strip(),
            "damage": float(values["damage"]),
            "nodes_searched": int(values["nodes_searched"]),
            "player_hp_after": float(values["player_hp"]),
            "opponent_hp_after": float(values["opponent_hp"]),
            "next_side": values["next_side"].title(),
        }

        replay_steps.append(
            ReplayStep(
                game_id=game_id,
                turn=int(values["turn"]),
                player=values["player"].title(),
                state=state,
                legal_moves=[],
                chosen_move=values["move"].strip(),
                evaluation=float(values["evaluation"]),
                search_depth=int(values["search_depth"]),
            )
        )

    return replay_steps


parsed_steps = parse_transcript_to_steps(
    game_id=recorded_game.game_id,
    transcript=recorded_game.transcript,
)

print("Transcript turns:", recorded_game.total_turns)
print("Parsed replay steps:", len(parsed_steps))

for step in parsed_steps:
    print(
        f"Turn {step.turn}: "
        f"{step.player} chose {step.chosen_move} | "
        f"score={step.evaluation:.2f} | "
        f"depth={step.search_depth}"
    )

assert len(parsed_steps) == recorded_game.total_turns

print("\nTranscript parser validation passed.")


# # Cell 20 â€” Attach Parsed Steps to the Validation Game

# In[19]:


# ============================================================
# Cell 20 - Attach Parsed Steps to Validation Game
# ============================================================

# Prevent duplicates if this cell is rerun.
recorded_game.steps.clear()

for step in parsed_steps:
    recorded_game.add_step(step)

print("Game ID:", recorded_game.game_id)
print("Expected turns:", recorded_game.total_turns)
print("Attached replay steps:", len(recorded_game.steps))
print("Recorder total steps:", summary.recorder.total_steps)

assert len(recorded_game.steps) == recorded_game.total_turns
assert summary.recorder.total_steps == recorded_game.total_turns

for step in recorded_game.steps:
    assert step.game_id == recorded_game.game_id
    assert step.turn > 0
    assert step.chosen_move
    assert step.search_depth > 0

print("\nReplay steps attached successfully.")


# # Cell 21 â€” Build and Validate the Training DataFrame

# In[20]:


# ============================================================
# Cell 21 - Build Validation Training DataFrame
# ============================================================

from src.training.dataset import replay_to_dataframe

validation_df = replay_to_dataframe(
    summary.recorder
)

print("Dataset shape:", validation_df.shape)
print()
display(validation_df)

expected_columns = {
    "game_id",
    "winner",
    "turn",
    "player",
    "chosen_move",
    "evaluation",
    "search_depth",
    "num_legal_moves",
    "state",
}

assert len(validation_df) == recorded_game.total_turns
assert expected_columns.issubset(validation_df.columns)
assert validation_df["game_id"].nunique() == 1
assert validation_df["chosen_move"].notna().all()
assert validation_df["search_depth"].gt(0).all()

print("\nValidation training DataFrame created successfully.")


# # Cell 22 â€” Run the Full Self-Play Session

# In[21]:


# ============================================================
# Cell 22 - Run Full Self-Play Session
# ============================================================

import time

session_start = time.perf_counter()

full_summary = run_self_play_session(
    match_factory=match_factory,
    initial_state_factory=initial_state_factory,
    num_games=CONFIG.num_matches,
    verbose=CONFIG.verbose,
)

# Parse every game transcript and attach its turn-level steps.
parse_failures = []

for game in full_summary.recorder.iter_games():
    game.steps.clear()

    game_steps = parse_transcript_to_steps(
        game_id=game.game_id,
        transcript=game.transcript,
    )

    if len(game_steps) != game.total_turns:
        parse_failures.append(
            {
                "game_id": game.game_id,
                "expected_turns": game.total_turns,
                "parsed_steps": len(game_steps),
            }
        )

    for step in game_steps:
        game.add_step(step)

session_elapsed = time.perf_counter() - session_start

print("=" * 70)
print("Full Self-Play Session")
print("=" * 70)
print("Requested games:", full_summary.requested_games)
print("Completed games:", full_summary.completed_games)
print("Recorded games:", full_summary.recorder.total_games)
print("Recorded steps:", full_summary.recorder.total_steps)
print("Parse failures:", len(parse_failures))
print(f"Elapsed time: {session_elapsed:.3f} seconds")
print(
    "Games per second:",
    round(
        full_summary.completed_games / session_elapsed,
        2,
    )
    if session_elapsed > 0
    else 0.0,
)

assert full_summary.completed_games == CONFIG.num_matches
assert full_summary.recorder.total_games == CONFIG.num_matches
assert not parse_failures
assert full_summary.recorder.total_steps > 0

print("\nFull self-play execution passed.")


# # Cell 23 â€” Generate Self-Play Statistics

# In[22]:


# ============================================================
# Cell 23 - Generate Self-Play Statistics
# ============================================================

from src.training.statistics import calculate_statistics

full_statistics = calculate_statistics(
    full_summary.recorder
)

game_rows = []

for game_number, game in enumerate(
    full_summary.recorder.iter_games(),
    start=1,
):
    game_rows.append(
        {
            "game_number": game_number,
            "game_id": game.game_id,
            "winner": game.winner,
            "total_turns": game.total_turns,
            "final_score": game.final_score,
            "replay_steps": len(game.steps),
            "transcript_characters": len(game.transcript),
        }
    )

game_statistics_df = pd.DataFrame(game_rows)

print("=" * 70)
print("Self-Play Statistics")
print("=" * 70)
print("Total games:", full_statistics.total_games)
print("Total replay steps:", full_statistics.total_steps)
print(f"Average turns: {full_statistics.average_turns:.2f}")
print(f"Average score: {full_statistics.average_score:.2f}")
print("Winner counts:", full_statistics.winner_counts)

print("\nGame-level statistics preview:")
display(game_statistics_df.head(10))

assert full_statistics.total_games == CONFIG.num_matches
assert full_statistics.total_steps == full_summary.recorder.total_steps
assert len(game_statistics_df) == CONFIG.num_matches
assert (
    game_statistics_df["total_turns"]
    == game_statistics_df["replay_steps"]
).all()

print("\nSelf-play statistics validated.")


# # Cell 24 â€” Export All Notebook 28 Artifacts

# In[23]:


# ============================================================
# Cell 24 - Export Notebook 28 Artifacts
# ============================================================

import json
import pickle

from dataclasses import asdict


step_dataset_df = replay_to_dataframe(
    full_summary.recorder
)

dataset_csv_path = (
    REPORT_DIR
    / "self_play_dataset.csv"
)

statistics_csv_path = (
    REPORT_DIR
    / "self_play_statistics.csv"
)

summary_json_path = (
    REPORT_DIR
    / "self_play_summary.json"
)

training_pickle_path = (
    REPORT_DIR
    / "training_examples.pkl"
)

transcripts_path = (
    REPORT_DIR
    / "self_play_transcripts.txt"
)


# 1. Turn-level training dataset
step_dataset_df.to_csv(
    dataset_csv_path,
    index=False,
)


# 2. Game-level statistics
game_statistics_df.to_csv(
    statistics_csv_path,
    index=False,
)


# 3. JSON summary
summary_payload = {
    "notebook": 28,
    "project": "PTCG AI Battle Challenge",
    "team": "Team Jesus",
    "session_id": session.session_id,
    "random_seed": CONFIG.random_seed,
    "requested_games": full_summary.requested_games,
    "completed_games": full_summary.completed_games,
    "total_replay_steps": full_summary.recorder.total_steps,
    "average_turns": full_statistics.average_turns,
    "average_score": full_statistics.average_score,
    "winner_counts": full_statistics.winner_counts,
    "parse_failures": parse_failures,
    "elapsed_seconds": session_elapsed,
}

with summary_json_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary_payload,
        file,
        indent=2,
    )


# 4. Pickled training examples
training_examples = [
    asdict(step)
    for step in full_summary.recorder.iter_steps()
]

with training_pickle_path.open("wb") as file:
    pickle.dump(
        training_examples,
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


# 5. Human-readable transcripts
with transcripts_path.open(
    "w",
    encoding="utf-8",
) as file:

    for game_number, game in enumerate(
        full_summary.recorder.iter_games(),
        start=1,
    ):
        file.write("=" * 80 + "\n")
        file.write(f"GAME {game_number}\n")
        file.write(f"Game ID: {game.game_id}\n")
        file.write(f"Winner: {game.winner}\n")
        file.write(f"Turns: {game.total_turns}\n")
        file.write(f"Final score: {game.final_score}\n")
        file.write("=" * 80 + "\n")
        file.write(game.transcript.strip())
        file.write("\n\n")


artifact_paths = [
    dataset_csv_path,
    statistics_csv_path,
    summary_json_path,
    training_pickle_path,
    transcripts_path,
]

print("=" * 70)
print("Notebook 28 Artifacts")
print("=" * 70)

for artifact_path in artifact_paths:
    print(
        f"{artifact_path.name}: "
        f"{artifact_path.stat().st_size:,} bytes"
    )

assert all(path.exists() for path in artifact_paths)
assert all(path.stat().st_size > 0 for path in artifact_paths)
assert len(step_dataset_df) == full_summary.recorder.total_steps
assert len(training_examples) == full_summary.recorder.total_steps

print("\nAll Notebook 28 artifacts exported successfully.")


# # Cell 25 â€” Reload and Validate Exported Artifacts

# In[24]:


# ============================================================
# Cell 25 - Reload and Validate Exported Artifacts
# ============================================================

import json
import pickle

reloaded_dataset_df = pd.read_csv(
    dataset_csv_path
)

reloaded_statistics_df = pd.read_csv(
    statistics_csv_path
)

with summary_json_path.open(
    "r",
    encoding="utf-8",
) as file:
    reloaded_summary = json.load(file)

with training_pickle_path.open("rb") as file:
    reloaded_training_examples = pickle.load(file)

transcript_text = transcripts_path.read_text(
    encoding="utf-8"
)

print("=" * 70)
print("Artifact Reload Validation")
print("=" * 70)
print("Dataset rows:", len(reloaded_dataset_df))
print("Statistics rows:", len(reloaded_statistics_df))
print("Pickled examples:", len(reloaded_training_examples))
print("Completed games:", reloaded_summary["completed_games"])
print("Transcript characters:", len(transcript_text))

assert len(reloaded_dataset_df) == full_summary.recorder.total_steps
assert len(reloaded_statistics_df) == CONFIG.num_matches
assert len(reloaded_training_examples) == full_summary.recorder.total_steps
assert reloaded_summary["completed_games"] == CONFIG.num_matches
assert reloaded_summary["parse_failures"] == []
assert transcript_text.strip()

print("\nAll exported artifacts reloaded successfully.")


# # Cell 26 â€” Final Pipeline Validation

# In[25]:


# ============================================================
# Cell 26 - Final Pipeline Validation
# ============================================================

validation_checks = {
    "requested_games_completed": (
        full_summary.completed_games
        == CONFIG.num_matches
    ),
    "all_games_recorded": (
        full_summary.recorder.total_games
        == CONFIG.num_matches
    ),
    "replay_steps_created": (
        full_summary.recorder.total_steps > 0
    ),
    "all_transcripts_parsed": (
        len(parse_failures) == 0
    ),
    "dataset_matches_steps": (
        len(step_dataset_df)
        == full_summary.recorder.total_steps
    ),
    "statistics_match_games": (
        len(game_statistics_df)
        == CONFIG.num_matches
    ),
    "turns_match_replay_steps": (
        game_statistics_df["total_turns"]
        == game_statistics_df["replay_steps"]
    ).all(),
    "all_artifacts_exist": all(
        path.exists()
        for path in artifact_paths
    ),
    "all_artifacts_nonempty": all(
        path.stat().st_size > 0
        for path in artifact_paths
    ),
}

validation_df = pd.DataFrame(
    [
        {
            "check": check_name,
            "passed": bool(passed),
        }
        for check_name, passed
        in validation_checks.items()
    ]
)

display(validation_df)

failed_checks = validation_df.loc[
    ~validation_df["passed"],
    "check",
].tolist()

if failed_checks:
    raise RuntimeError(
        "Notebook 28 validation failed: "
        + ", ".join(failed_checks)
    )

print(
    f"\nAll {len(validation_checks)} "
    "Notebook 28 validation checks passed."
)


# # Cell 27 â€” Final Summary

# In[26]:


# ============================================================
# Cell 27 - Notebook 28 Final Summary
# ============================================================

print("=" * 72)
print("Notebook 28 â€” Self-Play Training Pipeline")
print("=" * 72)

print("\nProject: PTCG AI Battle Challenge")
print("Team: Team Jesus")

print("\nExecution")
print("-" * 72)
print("Games completed:", full_summary.completed_games)
print("Replay steps generated:", full_summary.recorder.total_steps)
print("Parse failures:", len(parse_failures))
print(f"Elapsed time: {session_elapsed:.3f} seconds")

print("\nPerformance")
print("-" * 72)
print(f"Average turns: {full_statistics.average_turns:.2f}")
print(f"Average score: {full_statistics.average_score:.2f}")
print("Winner counts:", full_statistics.winner_counts)

print("\nArtifacts")
print("-" * 72)

for artifact_path in artifact_paths:
    print(
        f"{artifact_path.name}: "
        f"{artifact_path.stat().st_size:,} bytes"
    )

print("\nCapabilities validated")
print("-" * 72)
print("âœ“ Repeated production self-play")
print("âœ“ Alternating starting side")
print("âœ“ Game-level replay capture")
print("âœ“ Transcript-to-step parsing")
print("âœ“ Turn-level training dataset")
print("âœ“ Self-play statistics")
print("âœ“ CSV, JSON, PKL, and transcript exports")
print("âœ“ Artifact reload validation")

print("\nKnown limitation")
print("-" * 72)
print(
    "The current simulator uses a simplified PokÃ©mon battle model, "
    "and complete legal-move lists are not yet stored in transcripts."
)

print("\nNOTEBOOK 28 COMPLETED SUCCESSFULLY")


# In[ ]:




