"""Validation helpers for card datasets and repositories."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .constants import (
    CLASSIFICATION_COLUMNS,
    ENGINEERED_COLUMNS,
    PROCESSED_CARD_FILE,
    RAW_COLUMNS,
)
from .repository import CardRepository, load_repository


def validate_dataframe_schema(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return schema-validation errors for a processed card dataframe."""

    errors: list[str] = []

    missing_raw = [
        column
        for column in RAW_COLUMNS
        if column not in dataframe.columns
    ]

    missing_engineered = [
        column
        for column in ENGINEERED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_raw:
        errors.append(
            "Missing raw columns: "
            + ", ".join(missing_raw)
        )

    if missing_engineered:
        errors.append(
            "Missing engineered columns: "
            + ", ".join(missing_engineered)
        )

    if "Card ID" in dataframe.columns:
        missing_ids = int(
            dataframe["Card ID"].isna().sum()
        )

        if missing_ids:
            errors.append(
                f"Card ID contains {missing_ids} missing values."
            )

    return errors


def validate_classification_flags(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Validate Pokémon, Trainer, and Energy classification flags."""

    errors: list[str] = []

    missing_columns = [
        column
        for column in CLASSIFICATION_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        return [
            "Missing classification columns: "
            + ", ".join(missing_columns)
        ]

    numeric_flags = dataframe[
        CLASSIFICATION_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in CLASSIFICATION_COLUMNS:
        invalid_values = sorted(
            set(
                numeric_flags[column]
                .dropna()
                .unique()
            ) - {0, 1}
        )

        if invalid_values:
            errors.append(
                f"{column} contains invalid values: "
                f"{invalid_values}"
            )

    row_totals = (
        numeric_flags
        .fillna(0)
        .astype(int)
        .sum(axis=1)
    )

    unclassified = int(
        (row_totals == 0).sum()
    )

    multiply_classified = int(
        (row_totals > 1).sum()
    )

    if unclassified:
        errors.append(
            f"{unclassified} rows are unclassified."
        )

    if multiply_classified:
        errors.append(
            f"{multiply_classified} rows have multiple classifications."
        )

    return errors


def validate_repository(
    repository: CardRepository,
) -> list[str]:
    """Validate grouped CardRecord objects."""

    errors: list[str] = []

    if len(repository) == 0:
        errors.append("Repository is empty.")
        return errors

    missing_names = [
        card.card_id
        for card in repository.all_cards()
        if not card.name.strip()
    ]

    if missing_names:
        errors.append(
            f"{len(missing_names)} cards have missing names."
        )

    invalid_ids = [
        card.card_id
        for card in repository.all_cards()
        if card.card_id < 0
    ]

    if invalid_ids:
        errors.append(
            f"{len(invalid_ids)} cards have negative Card IDs."
        )

    classification_errors = [
        card.card_id
        for card in repository.all_cards()
        if sum(
            [
                card.is_pokemon,
                card.is_trainer,
                card.is_energy,
            ]
        ) != 1
    ]

    if classification_errors:
        errors.append(
            f"{len(classification_errors)} cards have "
            "invalid top-level classifications."
        )

    return errors


def validate_deck_ids(
    repository: CardRepository,
    deck_ids: list[int],
    *,
    expected_size: int | None = 60,
) -> list[str]:
    """Validate that every deck Card ID exists in the repository."""

    errors: list[str] = []

    if expected_size is not None:
        if len(deck_ids) != expected_size:
            errors.append(
                f"Deck contains {len(deck_ids)} cards "
                f"instead of {expected_size}."
            )

    unknown_ids = sorted(
        {
            int(card_id)
            for card_id in deck_ids
            if not repository.contains(int(card_id))
        }
    )

    if unknown_ids:
        errors.append(
            "Unknown Card IDs: "
            + ", ".join(map(str, unknown_ids))
        )

    return errors


def build_validation_report(
    path: Path | str = PROCESSED_CARD_FILE,
) -> dict[str, object]:
    """Run full validation and return a structured report."""

    dataset_path = Path(path).resolve()

    dataframe = pd.read_csv(
        dataset_path,
        encoding="utf-8-sig",
    )

    repository = load_repository(dataset_path)

    errors: list[str] = []
    errors.extend(validate_dataframe_schema(dataframe))
    errors.extend(validate_classification_flags(dataframe))
    errors.extend(validate_repository(repository))

    card_id_counts = Counter(
        int(value)
        for value in dataframe["Card ID"]
    )

    return {
        "valid": not errors,
        "errors": errors,
        "dataset_path": str(dataset_path),
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "unique_card_ids": len(card_id_counts),
        "repeated_id_rows": (
            len(dataframe) - len(card_id_counts)
        ),
        "repository_size": len(repository),
        "pokemon_count": len(repository.pokemon()),
        "trainer_count": len(repository.trainers()),
        "energy_count": len(repository.energies()),
    }


def assert_valid_card_database(
    path: Path | str = PROCESSED_CARD_FILE,
) -> dict[str, object]:
    """Raise ValueError when full card-database validation fails."""

    report = build_validation_report(path)

    if report["errors"]:
        raise ValueError(
            "Card database validation failed:\n"
            + "\n".join(
                str(error)
                for error in report["errors"]
            )
        )

    return report
