"""Tournament scheduling utilities."""

from __future__ import annotations

from typing import Any

from .manager import TournamentEntrant


def create_round_robin_schedule(
    entrants: list[TournamentEntrant],
    *,
    games_per_pairing: int = 2,
) -> list[dict[str, Any]]:
    """Create every unique pairing with alternating player positions."""

    if len(entrants) < 2:
        raise ValueError(
            "At least two entrants are required."
        )

    if games_per_pairing < 1:
        raise ValueError(
            "games_per_pairing must be at least 1."
        )

    names = [
        entrant.name
        for entrant in entrants
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "Entrant names must be unique."
        )

    schedule: list[dict[str, Any]] = []
    match_number = 1

    for first_index in range(len(entrants)):
        for second_index in range(
            first_index + 1,
            len(entrants),
        ):
            first_entrant = entrants[first_index]
            second_entrant = entrants[second_index]

            for game_index in range(
                games_per_pairing
            ):
                if game_index % 2 == 0:
                    player_name = first_entrant.name
                    opponent_name = second_entrant.name
                else:
                    player_name = second_entrant.name
                    opponent_name = first_entrant.name

                schedule.append(
                    {
                        "match_number": match_number,
                        "player_name": player_name,
                        "opponent_name": opponent_name,
                        "starting_player": "Player",
                    }
                )

                match_number += 1

    return schedule
