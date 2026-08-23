"""Official card database package for the PTCG AI Battle Challenge."""

from .constants import (
    PROCESSED_CARD_FILE,
    RAW_CARD_FILE,
)
from .loader import (
    build_card_records,
    load_card_records,
    load_processed_dataframe,
)
from .models import CardMove, CardRecord
from .normalization import (
    clean_optional_float,
    clean_optional_int,
    clean_optional_text,
    normalize_card_name,
)
from .repository import (
    CardRepository,
    load_repository,
)
from .search import (
    energies,
    find_cards,
    get_card,
    get_repository,
    pokemon,
    search_cards,
    trainers,
)
from .validation import (
    assert_valid_card_database,
    build_validation_report,
    validate_deck_ids,
    validate_repository,
)

__all__ = [
    "CardMove",
    "CardRecord",
    "CardRepository",
    "PROCESSED_CARD_FILE",
    "RAW_CARD_FILE",
    "assert_valid_card_database",
    "build_card_records",
    "build_validation_report",
    "clean_optional_float",
    "clean_optional_int",
    "clean_optional_text",
    "energies",
    "find_cards",
    "get_card",
    "get_repository",
    "load_card_records",
    "load_processed_dataframe",
    "load_repository",
    "normalize_card_name",
    "pokemon",
    "search_cards",
    "trainers",
    "validate_deck_ids",
    "validate_repository",
]
