"""Convenience search helpers for the official card database."""

from __future__ import annotations

from functools import lru_cache

from .repository import CardRepository, load_repository


@lru_cache(maxsize=1)
def get_repository() -> CardRepository:
    """Load and cache the canonical repository."""
    return load_repository()


def get_card(card_id: int):
    """Return a card by Card ID."""
    return get_repository().get(card_id)


def find_cards(name: str):
    """Return exact normalized-name matches."""
    return get_repository().find_by_name(name)


def search_cards(
    query: str,
    limit: int = 10,
):
    """Return partial card-name matches."""
    return get_repository().search_names(
        query,
        limit=limit,
    )


def pokemon():
    """Return every Pokémon card."""
    return get_repository().pokemon()


def trainers():
    """Return every Trainer card."""
    return get_repository().trainers()


def energies():
    """Return every Energy card."""
    return get_repository().energies()
