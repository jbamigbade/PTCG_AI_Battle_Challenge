"""Normalization and safe-conversion helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_optional_text(value: Any) -> str | None:
    """Convert null-like values into trimmed strings or None."""

    if pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def clean_optional_int(value: Any) -> int | None:
    """Convert a value into an integer when possible."""

    if pd.isna(value):
        return None

    numeric = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric):
        return None

    return int(numeric)


def clean_optional_float(value: Any) -> float | None:
    """Convert a value into a float when possible."""

    if pd.isna(value):
        return None

    numeric = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric):
        return None

    return float(numeric)


def normalize_card_name(name: str) -> str:
    """Normalize a card name for stable exact and partial searching."""

    normalized = str(name).casefold().strip()
    normalized = re.sub(r"[’']", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()
