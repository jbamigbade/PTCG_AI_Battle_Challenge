"""Paths and schema constants for the card database package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_CARD_FILE = RAW_DATA_DIR / "EN_Card_Data.csv"
PROCESSED_CARD_FILE = (
    PROCESSED_DATA_DIR / "pokemon_cards_processed.csv"
)

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

CLASSIFICATION_COLUMNS = [
    "is_pokemon",
    "is_trainer",
    "is_energy",
]
