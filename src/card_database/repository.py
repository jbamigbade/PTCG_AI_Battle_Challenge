"""Fast Card ID and card-name repository."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .constants import PROCESSED_CARD_FILE
from .loader import load_card_records
from .models import CardRecord
from .normalization import normalize_card_name


class CardRepository:
    """Read-only repository for official card records."""

    def __init__(
        self,
        records: dict[int, CardRecord],
    ) -> None:
        self._records = dict(records)
        self._name_index: dict[
            str,
            list[CardRecord],
        ] = defaultdict(list)

        for card in self._records.values():
            key = normalize_card_name(card.name)
            self._name_index[key].append(card)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, card_id: object) -> bool:
        try:
            numeric_id = int(card_id)
        except (TypeError, ValueError):
            return False

        return numeric_id in self._records

    def contains(self, card_id: int) -> bool:
        """Return whether the Card ID exists."""

        return int(card_id) in self._records

    def get(self, card_id: int) -> CardRecord:
        """Return one card or raise KeyError."""

        numeric_id = int(card_id)

        try:
            return self._records[numeric_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Card ID: {numeric_id}"
            ) from exc

    def get_optional(
        self,
        card_id: int,
    ) -> CardRecord | None:
        """Return one card or None."""

        return self._records.get(int(card_id))

    def find_by_name(
        self,
        name: str,
    ) -> list[CardRecord]:
        """Return exact normalized-name matches."""

        key = normalize_card_name(name)

        return list(self._name_index.get(key, []))

    def search_names(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[CardRecord]:
        """Return partial normalized-name matches."""

        if limit < 1:
            return []

        normalized_query = normalize_card_name(query)

        if not normalized_query:
            return []

        matches = [
            card
            for card in self._records.values()
            if normalized_query
            in normalize_card_name(card.name)
        ]

        return sorted(
            matches,
            key=lambda card: (
                normalize_card_name(card.name),
                card.card_id,
            ),
        )[:limit]

    def all_cards(self) -> list[CardRecord]:
        """Return all cards sorted by Card ID."""

        return [
            self._records[card_id]
            for card_id in sorted(self._records)
        ]

    def pokemon(self) -> list[CardRecord]:
        """Return every Pokémon card."""

        return [
            card
            for card in self.all_cards()
            if card.is_pokemon
        ]

    def trainers(self) -> list[CardRecord]:
        """Return every Trainer card."""

        return [
            card
            for card in self.all_cards()
            if card.is_trainer
        ]

    def energies(self) -> list[CardRecord]:
        """Return every Energy card."""

        return [
            card
            for card in self.all_cards()
            if card.is_energy
        ]


def load_repository(
    path: Path | str = PROCESSED_CARD_FILE,
) -> CardRepository:
    """Load the canonical card repository."""

    return CardRepository(
        load_card_records(path)
    )
