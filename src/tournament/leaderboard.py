"""Tournament leaderboard construction utilities."""

from __future__ import annotations

import pandas as pd

from .manager import TournamentEntrant


LEADERBOARD_COLUMNS = [
    "name",
    "matches",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "average_turns",
    "average_score",
]


def build_leaderboard_dataframe(
    entrants: list[TournamentEntrant],
) -> pd.DataFrame:
    """Build and rank a tournament leaderboard."""

    rows = [
        {
            "name": entrant.name,
            "matches": entrant.matches_played,
            "wins": entrant.wins,
            "losses": entrant.losses,
            "draws": entrant.draws,
            "win_rate": entrant.win_rate,
            "average_turns": entrant.average_turns,
            "average_score": entrant.average_score,
        }
        for entrant in entrants
    ]

    if not rows:
        return pd.DataFrame(
            columns=LEADERBOARD_COLUMNS
        )

    return (
        pd.DataFrame(
            rows,
            columns=LEADERBOARD_COLUMNS,
        )
        .sort_values(
            by=[
                "wins",
                "win_rate",
                "average_score",
                "name",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )
