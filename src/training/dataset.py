from __future__ import annotations

from pathlib import Path

import pandas as pd

from .replay import ReplayRecorder


DATASET_COLUMNS = [
    "game_id",
    "winner",
    "turn",
    "player",
    "chosen_move",
    "evaluation",
    "search_depth",
    "num_legal_moves",
    "state",
]


def replay_to_dataframe(
    recorder: ReplayRecorder,
) -> pd.DataFrame:
    """
    Convert all replay steps into a pandas DataFrame.
    """

    rows = []

    for game in recorder.iter_games():
        for step in game.steps:
            rows.append(
                {
                    "game_id": game.game_id,
                    "winner": game.winner,
                    "turn": step.turn,
                    "player": step.player,
                    "chosen_move": step.chosen_move,
                    "evaluation": step.evaluation,
                    "search_depth": step.search_depth,
                    "num_legal_moves": len(step.legal_moves),
                    "state": step.state,
                }
            )

    return pd.DataFrame(
        rows,
        columns=DATASET_COLUMNS,
    )


def export_csv(
    recorder: ReplayRecorder,
    filename: str | Path,
) -> Path:
    """
    Export replay steps to CSV and return the output path.
    """

    output_path = Path(filename)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_to_dataframe(recorder).to_csv(
        output_path,
        index=False,
    )

    return output_path


__all__ = [
    "DATASET_COLUMNS",
    "replay_to_dataframe",
    "export_csv",
]
