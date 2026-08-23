"""Load and convert the processed card dataset into CardRecord objects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import PROCESSED_CARD_FILE
from .models import CardMove, CardRecord
from .normalization import (
    clean_optional_float,
    clean_optional_int,
    clean_optional_text,
)


def load_processed_dataframe(
    path: Path | str = PROCESSED_CARD_FILE,
) -> pd.DataFrame:
    """Load the canonical processed card dataset."""

    dataset_path = Path(path).resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed card dataset not found: {dataset_path}"
        )

    dataframe = pd.read_csv(
        dataset_path,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            f"Processed card dataset is empty: {dataset_path}"
        )

    return dataframe


def row_to_move(row: pd.Series) -> CardMove | None:
    """
    Convert one dataset row into a CardMove.

    Rows without move, ability, or effect information return None.
    """

    name = clean_optional_text(row.get("Move Name"))
    cost = clean_optional_text(row.get("Cost"))
    damage = clean_optional_text(row.get("Damage"))
    effect = clean_optional_text(row.get("Effect Explanation"))

    if not any([name, cost, damage, effect]):
        return None

    return CardMove(
        name=name,
        cost=cost,
        damage=damage,
        effect=effect,
        damage_numeric=clean_optional_int(
            row.get("damage_numeric")
        ),
        energy_cost=clean_optional_int(
            row.get("energy_cost")
        ),
    )


def group_to_card_record(
    card_id: int,
    group: pd.DataFrame,
) -> CardRecord:
    """Combine all rows for one Card ID into one CardRecord."""

    if group.empty:
        raise ValueError(
            f"Cannot build CardRecord from empty group: {card_id}"
        )

    first = group.iloc[0]
    moves: list[CardMove] = []

    for _, row in group.iterrows():
        move = row_to_move(row)

        if move is not None and move not in moves:
            moves.append(move)

    card_name = clean_optional_text(first.get("Card Name"))

    if card_name is None:
        raise ValueError(
            f"Card ID {card_id} has no valid card name."
        )

    return CardRecord(
        card_id=int(card_id),
        name=card_name,
        expansion=clean_optional_text(first.get("Expansion")),
        collection_number=clean_optional_text(
            first.get("Collection No.")
        ),
        stage_or_type=clean_optional_text(
            first.get(
                "Stage (Pokémon)/Type (Energy and Trainer)"
            )
        ),
        rule=clean_optional_text(first.get("Rule")),
        category=clean_optional_text(first.get("Category")),
        previous_stage=clean_optional_text(
            first.get("Previous stage")
        ),
        hp=clean_optional_int(first.get("hp_numeric")),
        card_type=clean_optional_text(
            first.get("primary_type")
        ),
        weakness=clean_optional_text(
            first.get("weakness_type")
        ),
        resistance=clean_optional_text(
            first.get("resistance_type")
        ),
        retreat=clean_optional_int(
            first.get("retreat_numeric")
        ),
        is_pokemon=bool(first.get("is_pokemon", 0)),
        is_trainer=bool(first.get("is_trainer", 0)),
        is_energy=bool(first.get("is_energy", 0)),
        is_basic=bool(first.get("is_basic", 0)),
        is_stage1=bool(first.get("is_stage1", 0)),
        is_stage2=bool(first.get("is_stage2", 0)),
        has_rule_box=bool(first.get("has_rule_box", 0)),
        has_attack=bool(first.get("has_attack", 0)),
        has_ability=bool(first.get("has_ability", 0)),
        attack_power_score=clean_optional_float(
            first.get("attack_power_score")
        ),
        tank_score=clean_optional_float(
            first.get("tank_score")
        ),
        mobility_score=clean_optional_float(
            first.get("mobility_score")
        ),
        evolution_score=clean_optional_float(
            first.get("evolution_score")
        ),
        overall_battle_score=clean_optional_float(
            first.get("overall_battle_score")
        ),
        moves=moves,
    )


def build_card_records(
    dataframe: pd.DataFrame,
) -> dict[int, CardRecord]:
    """Build a Card ID to CardRecord dictionary."""

    if "Card ID" not in dataframe.columns:
        raise ValueError(
            "Processed dataframe is missing the 'Card ID' column."
        )

    records: dict[int, CardRecord] = {}

    for card_id, group in dataframe.groupby(
        "Card ID",
        sort=True,
    ):
        numeric_id = int(card_id)

        records[numeric_id] = group_to_card_record(
            numeric_id,
            group,
        )

    return records


def load_card_records(
    path: Path | str = PROCESSED_CARD_FILE,
) -> dict[int, CardRecord]:
    """Load the processed dataset and return grouped card records."""

    dataframe = load_processed_dataframe(path)

    return build_card_records(dataframe)
