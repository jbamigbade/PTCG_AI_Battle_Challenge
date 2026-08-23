#!/usr/bin/env python
# coding: utf-8

# # Notebook 17 — Official Card Knowledge Base
# 
# Now we will connect the official Pokémon card metadata to the Team Jesus project.
# 
# This notebook will:
# 
# locate the official card CSV,
# inspect every column,
# validate Card IDs,
# clean and normalize the data,
# create fast lookup tables,
# export reusable production code into src/card_database/.

# # Cell 1 — Notebook heading

# # Notebook 17 — Official Card Database Integration
# 
# ## The Pokémon Company — PTCG AI Battle Challenge
# 
# ### Team Jesus
# 
# This notebook validates the official English card metadata and packages the
# previously cleaned and engineered card dataset for production use.
# 
# ## Important project context
# 
# The official card data was previously:
# 
# - explored,
# - compared with the Japanese dataset,
# - cleaned,
# - normalized,
# - feature engineered,
# - and exported as `pokemon_cards_processed.csv`.
# 
# Therefore, this notebook does not repeat that work.
# 
# ## Objectives
# 
# 1. Validate the official raw English card dataset.
# 2. Locate and validate the previously processed dataset.
# 3. Confirm raw-to-processed Card ID consistency.
# 4. Inspect engineered feature coverage and data quality.
# 5. Move or copy the canonical processed dataset into `data/processed/`.
# 6. Build fast Card ID and card-name lookups.
# 7. Create reusable production modules in `src/card_database/`.
# 8. Test the resulting card repository.
# 9. Produce a validation report for later Kaggle integration.
# 
# ## Expected production structure
# 
# ```text
# data/
# ├── raw/
# │   └── EN_Card_Data.csv
# └── processed/
#     └── pokemon_cards_processed.csv
# 
# src/card_database/
# ├── __init__.py
# ├── loader.py
# ├── models.py
# ├── repository.py
# └── validation.py
# ```

# # Cell 2 — Imports

# In[2]:



from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("Current directory:", Path.cwd())

# # Cell 3 — Locate the project root

# In[3]:


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the PTCG project root using common project markers.
    """

    current = (start or Path.cwd()).resolve()

    markers = [
        "src",
        "notebooks",
        "data",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
    ]

    for candidate in [current, *current.parents]:
        marker_count = sum(
            (candidate / marker).exists()
            for marker in markers
        )

        if marker_count >= 2:
            return candidate

    if current.name.lower() == "notebooks":
        return current.parent

    return current


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = PROJECT_ROOT / "src"
CARD_DATABASE_DIR = SRC_DIR / "card_database"

REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK17_REPORT_DIR = REPORTS_DIR / "notebook17"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CARD_DATABASE_DIR,
    NOTEBOOK17_REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw data:", RAW_DATA_DIR)
print("Processed data:", PROCESSED_DATA_DIR)
print("Card database package:", CARD_DATABASE_DIR)
print("Notebook 17 reports:", NOTEBOOK17_REPORT_DIR)

# # Cell 4 — Search for possible official card-data files

# In[4]:


CARD_FILE_PATTERNS = [
    "*.csv",
    "*.CSV",
]

candidate_csv_files: list[Path] = []

for pattern in CARD_FILE_PATTERNS:
    candidate_csv_files.extend(
        path
        for path in PROJECT_ROOT.rglob(pattern)
        if ".git" not in path.parts
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "submission_work" not in path.parts
    )

candidate_csv_files = sorted(
    set(path.resolve() for path in candidate_csv_files)
)

print(
    f"Found {len(candidate_csv_files)} CSV file(s) "
    "inside the project.\n"
)

for index, path in enumerate(candidate_csv_files, start=1):
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path

    print(
        f"{index:2}. {relative} "
        f"({path.stat().st_size:,} bytes)"
    )

# # Cell 5 — Rank likely official card files

# In[5]:


def card_file_score(path: Path) -> int:
    """
    Give likely official card datasets a higher score.
    """

    name = path.name.lower()
    full_path = str(path).lower()

    score = 0

    keyword_weights = {
        "card": 10,
        "cards": 10,
        "en": 4,
        "english": 8,
        "pokemon": 6,
        "metadata": 5,
        "data": 3,
    }

    for keyword, weight in keyword_weights.items():
        if keyword in name:
            score += weight

    if "raw" in full_path:
        score += 3

    if "processed" in full_path:
        score -= 2

    if "report" in full_path:
        score -= 5

    if "battle_history" in name:
        score -= 20

    if "tournament" in name:
        score -= 20

    if "benchmark" in name:
        score -= 20

    return score


ranked_card_candidates = sorted(
    candidate_csv_files,
    key=lambda path: (
        card_file_score(path),
        path.stat().st_size,
    ),
    reverse=True,
)

print("Most likely card datasets:\n")

for index, path in enumerate(
    ranked_card_candidates[:15],
    start=1,
):
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path

    print(
        f"{index:2}. score={card_file_score(path):2} | "
        f"{relative} | "
        f"{path.stat().st_size:,} bytes"
    )

# # Cell 6 — Preview candidate CSV structures

# In[7]:


def preview_csv(
    path: Path,
    *,
    rows: int = 2,
) -> None:
    """
    Preview a CSV file without loading the full dataset.
    """

    print("=" * 100)
    print(path)
    print("=" * 100)

    try:
        preview = pd.read_csv(
            path,
            nrows=rows,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        preview = pd.read_csv(
            path,
            nrows=rows,
            encoding="latin-1",
        )
    except Exception as exc:
        print(
            f"Could not read file: "
            f"{type(exc).__name__}: {exc}\n"
        )
        return

    print("Columns:")
    for index, column in enumerate(
        preview.columns,
        start=1,
    ):
        print(f"{index:2}. {column}")

    print("\nPreview:")
    display(preview)

    print()


for candidate in ranked_card_candidates[:5]:
    preview_csv(candidate)

# # Cell 7 — Load the official English card database

# In[9]:


OFFICIAL_CARD_FILE = RAW_DATA_DIR / "EN_Card_Data.csv"

if not OFFICIAL_CARD_FILE.exists():
    raise FileNotFoundError(
        f"Official card database not found:\n{OFFICIAL_CARD_FILE}"
    )

cards_df = pd.read_csv(
    OFFICIAL_CARD_FILE,
    encoding="utf-8-sig",
)

print("Loaded successfully.")
print()

print("Rows :", len(cards_df))
print("Columns :", len(cards_df.columns))
print()

display(cards_df.head())

# # Cell 8 — Locate the processed dataset

# In[10]:


PROCESSED_FILE_NAMES = {
    "pokemon_cards_processed.csv",
}

processed_candidates = sorted(
    path.resolve()
    for path in PROJECT_ROOT.rglob("pokemon_cards_processed.csv")
    if ".git" not in path.parts
    and ".venv" not in path.parts
    and "__pycache__" not in path.parts
)

print(
    f"Found {len(processed_candidates)} processed "
    "card dataset candidate(s).\n"
)

for index, path in enumerate(
    processed_candidates,
    start=1,
):
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = path

    print(
        f"{index}. {relative} "
        f"({path.stat().st_size:,} bytes)"
    )

if not processed_candidates:
    raise FileNotFoundError(
        "pokemon_cards_processed.csv was not found."
    )

# # Cell 9 — Select the existing processed dataset

# In[11]:


def choose_processed_card_file(
    candidates: list[Path],
) -> Path:
    """
    Select the most appropriate processed card dataset.
    """

    preferred_paths = [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pokemon_cards_processed.csv",

        PROJECT_ROOT
        / "notebooks"
        / "data"
        / "processed"
        / "pokemon_cards_processed.csv",
    ]

    for preferred in preferred_paths:
        preferred = preferred.resolve()

        if preferred in candidates:
            return preferred

    return max(
        candidates,
        key=lambda path: path.stat().st_size,
    )


EXISTING_PROCESSED_FILE = choose_processed_card_file(
    processed_candidates
)

CANONICAL_PROCESSED_FILE = (
    PROCESSED_DATA_DIR
    / "pokemon_cards_processed.csv"
)

print("Existing processed file:")
print(EXISTING_PROCESSED_FILE)

print("\nCanonical processed location:")
print(CANONICAL_PROCESSED_FILE)

# # Cell 10 — Load the processed dataset

# In[12]:


processed_df = pd.read_csv(
    EXISTING_PROCESSED_FILE,
    encoding="utf-8-sig",
)

print("Processed dataset loaded successfully.")
print()
print("Rows:", len(processed_df))
print("Columns:", len(processed_df.columns))

display(processed_df.head())

# # Cell 11 — Validate original and engineered columns

# In[13]:


RAW_COLUMNS = [
    "Card ID",
    "Card Name",
    "Expansion",
    "Collection No.",
    "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule",
    "Category",
    "Previous stage",
    "HP",
    "Type",
    "Weakness",
    "Resistance (Type)",
    "Retreat",
    "Move Name",
    "Cost",
    "Damage",
    "Effect Explanation",
]

ENGINEERED_COLUMNS = [
    "stage_numeric",
    "is_basic",
    "is_stage1",
    "is_stage2",
    "hp_numeric",
    "retreat_numeric",
    "damage_numeric",
    "energy_cost",
    "primary_type",
    "type_numeric",
    "weakness_type",
    "weakness_numeric",
    "resistance_type",
    "resistance_numeric",
    "attack_power_score",
    "tank_score",
    "mobility_score",
    "evolution_score",
    "has_rule_box",
    "overall_battle_score",
    "is_pokemon",
    "is_trainer",
    "is_energy",
    "card_category",
    "has_attack",
    "has_ability",
]

missing_raw_columns = [
    column
    for column in RAW_COLUMNS
    if column not in processed_df.columns
]

missing_engineered_columns = [
    column
    for column in ENGINEERED_COLUMNS
    if column not in processed_df.columns
]

print("Missing original columns:", missing_raw_columns)
print("Missing engineered columns:", missing_engineered_columns)

if missing_raw_columns or missing_engineered_columns:
    raise ValueError(
        "The processed dataset is missing required columns."
    )

print("\nProcessed dataset schema validation passed.")

# # Cell 12 — Compare raw and processed Card IDs

# In[14]:


raw_card_ids = set(
    pd.to_numeric(
        cards_df["Card ID"],
        errors="raise",
    ).astype(int)
)

processed_card_ids = set(
    pd.to_numeric(
        processed_df["Card ID"],
        errors="raise",
    ).astype(int)
)

missing_from_processed = sorted(
    raw_card_ids - processed_card_ids
)

extra_in_processed = sorted(
    processed_card_ids - raw_card_ids
)

print("Raw Card IDs:", len(raw_card_ids))
print("Processed Card IDs:", len(processed_card_ids))
print("Missing from processed:", len(missing_from_processed))
print("Extra in processed:", len(extra_in_processed))

if missing_from_processed:
    print(
        "\nFirst missing IDs:",
        missing_from_processed[:20],
    )

if extra_in_processed:
    print(
        "\nFirst extra IDs:",
        extra_in_processed[:20],
    )

if missing_from_processed or extra_in_processed:
    raise ValueError(
        "Raw and processed Card ID sets do not match."
    )

print("\nRaw-to-processed Card ID validation passed.")

# # Cell 13 — Validate uniqueness and row alignment

# In[17]:


validation_errors: list[str] = []

raw_rows = len(cards_df)
processed_rows = len(processed_df)

raw_unique_ids = cards_df["Card ID"].nunique(dropna=True)
processed_unique_ids = processed_df["Card ID"].nunique(dropna=True)

raw_duplicate_rows = int(
    cards_df["Card ID"].duplicated().sum()
)

processed_duplicate_rows = int(
    processed_df["Card ID"].duplicated().sum()
)

raw_id_counts = (
    cards_df["Card ID"]
    .value_counts(dropna=False)
    .sort_index()
)

processed_id_counts = (
    processed_df["Card ID"]
    .value_counts(dropna=False)
    .sort_index()
)

if raw_rows != processed_rows:
    validation_errors.append(
        f"Row-count mismatch: raw={raw_rows}, "
        f"processed={processed_rows}."
    )

if raw_unique_ids != processed_unique_ids:
    validation_errors.append(
        f"Unique Card ID mismatch: "
        f"raw={raw_unique_ids}, "
        f"processed={processed_unique_ids}."
    )

if not raw_id_counts.equals(processed_id_counts):
    validation_errors.append(
        "Card ID row-frequency patterns do not match."
    )

print("Raw rows:", raw_rows)
print("Processed rows:", processed_rows)
print()

print("Raw unique Card IDs:", raw_unique_ids)
print("Processed unique Card IDs:", processed_unique_ids)
print()

print("Raw repeated-ID rows:", raw_duplicate_rows)
print(
    "Processed repeated-ID rows:",
    processed_duplicate_rows,
)

print(
    "\nRepeated Card IDs are expected because "
    "cards with multiple attacks may occupy multiple rows."
)

if validation_errors:
    print("\nValidation errors:")

    for error in validation_errors:
        print("-", error)

    raise ValueError(
        "\n".join(validation_errors)
    )

print("\nDataset alignment validation passed.")
print(
    "Raw and processed datasets have matching "
    "row counts and Card ID frequencies."
)

# # Cell 14 — Validate important numeric features

# In[18]:


NUMERIC_FEATURE_COLUMNS = [
    "stage_numeric",
    "hp_numeric",
    "retreat_numeric",
    "damage_numeric",
    "energy_cost",
    "type_numeric",
    "weakness_numeric",
    "resistance_numeric",
    "attack_power_score",
    "tank_score",
    "mobility_score",
    "evolution_score",
    "overall_battle_score",
]

numeric_report: list[dict[str, object]] = []

for column in NUMERIC_FEATURE_COLUMNS:
    numeric_values = pd.to_numeric(
        processed_df[column],
        errors="coerce",
    )

    numeric_report.append(
        {
            "column": column,
            "non_null": int(numeric_values.notna().sum()),
            "missing": int(numeric_values.isna().sum()),
            "minimum": (
                float(numeric_values.min())
                if numeric_values.notna().any()
                else None
            ),
            "maximum": (
                float(numeric_values.max())
                if numeric_values.notna().any()
                else None
            ),
        }
    )

numeric_report_df = pd.DataFrame(numeric_report)

display(numeric_report_df)

# # Cell 15 — Validate classification flags

# In[19]:


CLASSIFICATION_COLUMNS = [
    "is_pokemon",
    "is_trainer",
    "is_energy",
]

for column in CLASSIFICATION_COLUMNS:
    values = pd.to_numeric(
        processed_df[column],
        errors="coerce",
    )

    invalid_values = sorted(
        set(values.dropna().unique()) - {0, 1}
    )

    print(
        f"{column}: "
        f"counts={values.value_counts(dropna=False).to_dict()}"
    )

    if invalid_values:
        raise ValueError(
            f"{column} contains invalid values: "
            f"{invalid_values}"
        )

classification_total = (
    processed_df[CLASSIFICATION_COLUMNS]
    .fillna(0)
    .astype(int)
    .sum(axis=1)
)

unclassified_count = int(
    (classification_total == 0).sum()
)

multiply_classified_count = int(
    (classification_total > 1).sum()
)

print("\nUnclassified rows:", unclassified_count)
print("Multiply classified rows:", multiply_classified_count)

if unclassified_count or multiply_classified_count:
    raise ValueError(
        "Card classification flags are inconsistent."
    )

print("\nCard classification validation passed.")

# # Cell 16 — Copy the processed dataset to its canonical location

# In[20]:


import shutil

source_processed_file = (
    EXISTING_PROCESSED_FILE.resolve()
)

target_processed_file = (
    CANONICAL_PROCESSED_FILE.resolve()
)

target_processed_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if source_processed_file != target_processed_file:
    shutil.copy2(
        source_processed_file,
        target_processed_file,
    )

    print("Processed dataset copied to canonical location.")
else:
    print(
        "Processed dataset is already in "
        "the canonical location."
    )

print(target_processed_file)

# # Cell 17 — Reload the canonical dataset

# In[21]:


canonical_cards_df = pd.read_csv(
    CANONICAL_PROCESSED_FILE,
    encoding="utf-8-sig",
)

print("Canonical processed dataset loaded.")
print("Rows:", len(canonical_cards_df))
print("Columns:", len(canonical_cards_df.columns))

assert len(canonical_cards_df) == len(processed_df)
assert set(canonical_cards_df["Card ID"]) == set(
    processed_df["Card ID"]
)

print("\nCanonical dataset verification passed.")

# # Cell 18 — Define card and move models

# In[22]:


from dataclasses import dataclass, field
from typing import Any


def clean_optional_text(value: Any) -> str | None:
    """
    Convert null-like values into None and trim text.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    return text or None


def clean_optional_int(value: Any) -> int | None:
    """
    Convert a numeric value into int when possible.
    """

    if pd.isna(value):
        return None

    numeric = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric):
        return None

    return int(numeric)


def clean_optional_float(value: Any) -> float | None:
    """
    Convert a numeric value into float when possible.
    """

    if pd.isna(value):
        return None

    numeric = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric):
        return None

    return float(numeric)


@dataclass(frozen=True)
class CardMove:
    name: str | None
    cost: str | None
    damage: str | None
    effect: str | None
    damage_numeric: int | None
    energy_cost: int | None


@dataclass
class CardRecord:
    card_id: int
    name: str
    expansion: str | None
    collection_number: str | None
    stage_or_type: str | None
    rule: str | None
    category: str | None
    previous_stage: str | None
    hp: int | None
    card_type: str | None
    weakness: str | None
    resistance: str | None
    retreat: int | None

    is_pokemon: bool
    is_trainer: bool
    is_energy: bool
    is_basic: bool
    is_stage1: bool
    is_stage2: bool
    has_rule_box: bool
    has_attack: bool
    has_ability: bool

    attack_power_score: float | None
    tank_score: float | None
    mobility_score: float | None
    evolution_score: float | None
    overall_battle_score: float | None

    moves: list[CardMove] = field(default_factory=list)

# # Cell 19 — Build grouped card records

# In[23]:


def row_to_move(row: pd.Series) -> CardMove | None:
    """
    Convert one attack row into a CardMove.
    Return None when the row has no move information.
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
    """
    Combine all rows for one Card ID into one CardRecord.
    """

    first = group.iloc[0]

    moves: list[CardMove] = []

    for _, row in group.iterrows():
        move = row_to_move(row)

        if move is not None and move not in moves:
            moves.append(move)

    return CardRecord(
        card_id=int(card_id),
        name=str(first["Card Name"]).strip(),
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
        card_type=clean_optional_text(first.get("primary_type")),
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


card_records: dict[int, CardRecord] = {}

for card_id, group in canonical_cards_df.groupby(
    "Card ID",
    sort=True,
):
    card_records[int(card_id)] = group_to_card_record(
        int(card_id),
        group,
    )

print("Grouped card records:", len(card_records))
print("Original rows:", len(canonical_cards_df))
print()

sample_ids = list(card_records)[:5]

for card_id in sample_ids:
    card = card_records[card_id]

    print(
        f"{card.card_id}: {card.name} | "
        f"moves={len(card.moves)}"
    )

# # Cell 20 — Validate grouped records

# In[24]:


grouped_validation_errors: list[str] = []

expected_unique_cards = int(
    canonical_cards_df["Card ID"].nunique()
)

if len(card_records) != expected_unique_cards:
    grouped_validation_errors.append(
        "Grouped record count mismatch: "
        f"expected={expected_unique_cards}, "
        f"actual={len(card_records)}."
    )

missing_names = [
    card.card_id
    for card in card_records.values()
    if not card.name.strip()
]

if missing_names:
    grouped_validation_errors.append(
        f"{len(missing_names)} cards have missing names."
    )

mismatched_ids = [
    key
    for key, card in card_records.items()
    if key != card.card_id
]

if mismatched_ids:
    grouped_validation_errors.append(
        f"{len(mismatched_ids)} dictionary keys "
        "do not match Card IDs."
    )

print("Expected unique cards:", expected_unique_cards)
print("Created card records:", len(card_records))
print("Missing names:", len(missing_names))
print("Mismatched IDs:", len(mismatched_ids))

if grouped_validation_errors:
    raise ValueError(
        "\n".join(grouped_validation_errors)
    )

print("\nGrouped card-record validation passed.")

# # Cell 21 — Build normalized name lookup

# In[26]:


import re
from collections import defaultdict


def normalize_card_name(name: str) -> str:
    """
    Normalize card names for reliable lookup.
    """

    normalized = name.casefold().strip()
    normalized = re.sub(r"[’']", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


cards_by_name: dict[str, list[CardRecord]] = defaultdict(list)

for card in card_records.values():
    normalized_name = normalize_card_name(card.name)
    cards_by_name[normalized_name].append(card)

print("Normalized name keys:", len(cards_by_name))

sample_names = list(cards_by_name)[:10]

for name in sample_names:
    print(
        f"{name!r}: "
        f"{len(cards_by_name[name])} card(s)"
    )

# # Cell 22 — Create repository class

# In[27]:


class CardRepository:
    """
    Fast access to official card records by Card ID and name.
    """

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

    def contains(self, card_id: int) -> bool:
        return int(card_id) in self._records

    def get(self, card_id: int) -> CardRecord:
        try:
            return self._records[int(card_id)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Card ID: {card_id}"
            ) from exc

    def get_optional(
        self,
        card_id: int,
    ) -> CardRecord | None:
        return self._records.get(int(card_id))

    def find_by_name(
        self,
        name: str,
    ) -> list[CardRecord]:
        key = normalize_card_name(name)
        return list(self._name_index.get(key, []))

    def search_names(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[CardRecord]:
        normalized_query = normalize_card_name(query)

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

    def pokemon(self) -> list[CardRecord]:
        return [
            card
            for card in self._records.values()
            if card.is_pokemon
        ]

    def trainers(self) -> list[CardRecord]:
        return [
            card
            for card in self._records.values()
            if card.is_trainer
        ]

    def energies(self) -> list[CardRecord]:
        return [
            card
            for card in self._records.values()
            if card.is_energy
        ]

    def all_cards(self) -> list[CardRecord]:
        return list(self._records.values())

# # Cell 23 — Instantiate and test repository

# In[28]:


repository = CardRepository(card_records)

print("Repository size:", len(repository))
print("Expected size:", canonical_cards_df["Card ID"].nunique())

assert len(repository) == canonical_cards_df["Card ID"].nunique()

print("\nRepository created successfully.")

# # Cell 24 — Test known competition cards
# 
# ## These Card IDs came from the valid Mega Lucario baseline deck.

# In[29]:


TEST_CARD_IDS = [
    6,
    673,
    674,
    675,
    676,
    677,
    678,
    1102,
    1123,
    1141,
    1142,
    1152,
    1159,
    1182,
    1192,
    1227,
    1252,
]

missing_test_cards: list[int] = []

for card_id in TEST_CARD_IDS:
    card = repository.get_optional(card_id)

    if card is None:
        missing_test_cards.append(card_id)
        print(f"MISSING | Card ID {card_id}")
        continue

    print(
        f"{card.card_id:4} | "
        f"{card.name:30} | "
        f"moves={len(card.moves):2} | "
        f"pokemon={card.is_pokemon} | "
        f"trainer={card.is_trainer} | "
        f"energy={card.is_energy}"
    )

if missing_test_cards:
    raise ValueError(
        "Competition deck contains unknown Card IDs: "
        + ", ".join(map(str, missing_test_cards))
    )

print("\nAll baseline deck Card IDs were found.")

# # Cell 25 — Inspect a multi-move Pokémon

# In[30]:


multi_move_cards = [
    card
    for card in repository.pokemon()
    if len(card.moves) > 1
]

print(
    "Pokémon with multiple move/effect rows:",
    len(multi_move_cards),
)

for card in multi_move_cards[:5]:
    print()
    print(f"{card.card_id} — {card.name}")

    for index, move in enumerate(
        card.moves,
        start=1,
    ):
        print(
            f"  Move {index}: "
            f"name={move.name!r}, "
            f"cost={move.cost!r}, "
            f"damage={move.damage!r}, "
            f"energy_cost={move.energy_cost}"
        )

# # Cell 26 — Test name searches

# In[31]:


SEARCH_QUERIES = [
    "Mega Lucario ex",
    "Lillie's Determination",
    "Boss's Orders",
    "Energy",
]

for query in SEARCH_QUERIES:
    print("=" * 70)
    print("Query:", query)

    exact_matches = repository.find_by_name(query)

    if exact_matches:
        print("Exact matches:")

        for card in exact_matches[:10]:
            print(
                f"- {card.card_id}: "
                f"{card.name} "
                f"({card.expansion})"
            )
    else:
        print("No exact match.")

    partial_matches = repository.search_names(
        query,
        limit=5,
    )

    print("Partial matches:")

    for card in partial_matches:
        print(
            f"- {card.card_id}: "
            f"{card.name}"
        )

    print()

# # Cell 27 — Create Trainer subtype helpers

# In[32]:


def normalized_stage_or_type(
    card: CardRecord,
) -> str:
    return (
        card.stage_or_type or ""
    ).casefold()


def is_supporter(card: CardRecord) -> bool:
    return (
        card.is_trainer
        and "supporter" in normalized_stage_or_type(card)
    )


def is_item(card: CardRecord) -> bool:
    return (
        card.is_trainer
        and "item" in normalized_stage_or_type(card)
    )


def is_stadium(card: CardRecord) -> bool:
    return (
        card.is_trainer
        and "stadium" in normalized_stage_or_type(card)
    )


def is_tool(card: CardRecord) -> bool:
    return (
        card.is_trainer
        and "tool" in normalized_stage_or_type(card)
    )


trainer_type_report = {
    "supporters": sum(
        is_supporter(card)
        for card in repository.trainers()
    ),
    "items": sum(
        is_item(card)
        for card in repository.trainers()
    ),
    "stadiums": sum(
        is_stadium(card)
        for card in repository.trainers()
    ),
    "tools": sum(
        is_tool(card)
        for card in repository.trainers()
    ),
}

trainer_type_report

# In[ ]:



