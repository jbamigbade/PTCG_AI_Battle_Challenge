"""Tournament entrant registration, match execution, and record tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..battle_agent import PokemonBattleAgent
from ..battle_state import BattleState
from .match import TournamentMatch, TournamentResult
from .runner import run_two_agent_match
from .statistics import TournamentStatistics, update_statistics


@dataclass
class TournamentEntrant:
    """One registered AI competitor."""

    name: str
    agent: PokemonBattleAgent
    description: str = ""

    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    total_turns: int = 0
    total_score: float = 0.0

    @property
    def win_rate(self) -> float:
        """Return the entrant's fraction of matches won."""

        if self.matches_played == 0:
            return 0.0

        return self.wins / self.matches_played

    @property
    def average_turns(self) -> float:
        """Return the average match length."""

        if self.matches_played == 0:
            return 0.0

        return self.total_turns / self.matches_played

    @property
    def average_score(self) -> float:
        """Return the entrant's average score."""

        if self.matches_played == 0:
            return 0.0

        return self.total_score / self.matches_played


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
        """Register an entrant with a unique name."""

        if any(
            existing.name == entrant.name
            for existing in self.entrants
        ):
            raise ValueError(
                f"An entrant named {entrant.name!r} "
                "is already registered."
            )

        self.entrants.append(entrant)

    @property
    def number_of_entrants(self) -> int:
        """Return the number of registered entrants."""

        return len(self.entrants)

    @property
    def matches_played(self) -> int:
        """Return the number of managed matches."""

        return len(self.match_history)

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
        """Run one match and update both entrant records."""

        if player_name == opponent_name:
            raise ValueError(
                "An entrant cannot play against itself."
            )

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

        self.match_history.append(result)

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

    @staticmethod
    def _update_entrant_records(
        *,
        player_entrant: TournamentEntrant,
        opponent_entrant: TournamentEntrant,
        result: TournamentResult,
    ) -> None:
        """Update both entrants after a completed match."""

        player_entrant.matches_played += 1
        opponent_entrant.matches_played += 1

        player_entrant.total_turns += result.turns
        opponent_entrant.total_turns += result.turns

        player_entrant.total_score += result.final_score
        opponent_entrant.total_score -= result.final_score

        if result.winner == player_entrant.name:
            player_entrant.wins += 1
            opponent_entrant.losses += 1

        elif result.winner == opponent_entrant.name:
            opponent_entrant.wins += 1
            player_entrant.losses += 1

        else:
            player_entrant.draws += 1
            opponent_entrant.draws += 1
