"""Benchmark DataFrame construction and statistical summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


BENCHMARK_COLUMNS = [
    "benchmark_number",
    "player",
    "opponent",
    "starting_player",
    "search_depth",
    "winner",
    "turns",
    "final_score",
    "elapsed_seconds",
    "transcript_characters",
]


DEPTH_SUMMARY_COLUMNS = [
    "search_depth",
    "matches",
    "average_turns",
    "average_score",
    "average_elapsed_seconds",
    "minimum_elapsed_seconds",
    "maximum_elapsed_seconds",
]


def build_benchmark_dataframe(
    benchmark_rows: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Build a normalized benchmark-result DataFrame."""

    dataframe = pd.DataFrame(
        benchmark_rows
    )

    if dataframe.empty:
        return pd.DataFrame(
            columns=BENCHMARK_COLUMNS
        )

    missing_columns = [
        column
        for column in BENCHMARK_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Benchmark records are missing columns: "
            + ", ".join(missing_columns)
        )

    dataframe = dataframe[
        BENCHMARK_COLUMNS
    ].copy()

    dataframe = dataframe.sort_values(
        by="benchmark_number"
    ).reset_index(
        drop=True
    )

    return dataframe


def build_depth_summary(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate benchmark performance by search depth."""

    if benchmark_df.empty:
        return pd.DataFrame(
            columns=DEPTH_SUMMARY_COLUMNS
        )

    required_columns = {
        "benchmark_number",
        "search_depth",
        "turns",
        "final_score",
        "elapsed_seconds",
    }

    missing_columns = (
        required_columns
        - set(benchmark_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Benchmark DataFrame is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return (
        benchmark_df
        .groupby(
            "search_depth",
            as_index=False,
        )
        .agg(
            matches=(
                "benchmark_number",
                "count",
            ),
            average_turns=(
                "turns",
                "mean",
            ),
            average_score=(
                "final_score",
                "mean",
            ),
            average_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            minimum_elapsed_seconds=(
                "elapsed_seconds",
                "min",
            ),
            maximum_elapsed_seconds=(
                "elapsed_seconds",
                "max",
            ),
        )
        .sort_values(
            by="search_depth"
        )
        .reset_index(
            drop=True
        )
    )


def build_overall_summary(
    benchmark_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build overall benchmark summary statistics."""

    if benchmark_df.empty:
        return {
            "benchmark_runs": 0,
            "search_depths": [],
            "average_turns": 0.0,
            "average_score": 0.0,
            "average_runtime": 0.0,
            "minimum_runtime": 0.0,
            "maximum_runtime": 0.0,
        }

    required_columns = {
        "search_depth",
        "turns",
        "final_score",
        "elapsed_seconds",
    }

    missing_columns = (
        required_columns
        - set(benchmark_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Benchmark DataFrame is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return {
        "benchmark_runs": int(
            len(benchmark_df)
        ),
        "search_depths": sorted(
            int(value)
            for value in benchmark_df[
                "search_depth"
            ].unique().tolist()
        ),
        "average_turns": float(
            benchmark_df["turns"].mean()
        ),
        "average_score": float(
            benchmark_df[
                "final_score"
            ].mean()
        ),
        "average_runtime": float(
            benchmark_df[
                "elapsed_seconds"
            ].mean()
        ),
        "minimum_runtime": float(
            benchmark_df[
                "elapsed_seconds"
            ].min()
        ),
        "maximum_runtime": float(
            benchmark_df[
                "elapsed_seconds"
            ].max()
        ),
    }
