"""Benchmark CSV, JSON, and text-report utilities."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pandas as pd


def build_benchmark_report(
    *,
    benchmark_df: pd.DataFrame,
    depth_summary_df: pd.DataFrame,
    overall_summary: dict[str, Any],
) -> str:
    """Create a human-readable benchmark report."""

    lines = [
        "=" * 90,
        "INTEGRATED AI BENCHMARK REPORT",
        "=" * 90,
        "",
        (
            "Benchmark runs: "
            f"{overall_summary['benchmark_runs']}"
        ),
        (
            "Average turns: "
            f"{overall_summary['average_turns']:.2f}"
        ),
        (
            "Average score: "
            f"{overall_summary['average_score']:.2f}"
        ),
        (
            "Average runtime: "
            f"{overall_summary['average_runtime']:.6f}"
        ),
        (
            "Minimum runtime: "
            f"{overall_summary['minimum_runtime']:.6f}"
        ),
        (
            "Maximum runtime: "
            f"{overall_summary['maximum_runtime']:.6f}"
        ),
        "",
        "DEPTH SUMMARY",
        "",
        depth_summary_df.to_string(
            index=False
        ),
        "",
        "=" * 90,
        "",
        "INDIVIDUAL MATCH RESULTS",
        "",
        benchmark_df.to_string(
            index=False
        ),
    ]

    return "\n".join(lines)


def save_benchmark_reports(
    *,
    output_dir: Path,
    benchmark_df: pd.DataFrame,
    depth_summary_df: pd.DataFrame,
    overall_summary: dict[str, Any],
    file_prefix: str = "notebook15",
) -> dict[str, Path]:
    """Save CSV, JSON, and text benchmark reports."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_csv_path = (
        output_dir
        / f"{file_prefix}_benchmark.csv"
    )

    summary_json_path = (
        output_dir
        / f"{file_prefix}_summary.json"
    )

    report_txt_path = (
        output_dir
        / f"{file_prefix}_report.txt"
    )

    benchmark_df.to_csv(
        benchmark_csv_path,
        index=False,
    )

    summary_json_path.write_text(
        json.dumps(
            overall_summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    report_text = build_benchmark_report(
        benchmark_df=benchmark_df,
        depth_summary_df=depth_summary_df,
        overall_summary=overall_summary,
    )

    report_txt_path.write_text(
        report_text,
        encoding="utf-8",
    )

    return {
        "benchmark_csv": benchmark_csv_path,
        "summary_json": summary_json_path,
        "report_txt": report_txt_path,
    }


def validate_benchmark_reports(
    *,
    benchmark_csv_path: Path,
    summary_json_path: Path,
    report_txt_path: Path,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Reload and validate saved benchmark reports."""

    paths = [
        Path(benchmark_csv_path),
        Path(summary_json_path),
        Path(report_txt_path),
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark report: {path}"
            )

        if path.stat().st_size <= 0:
            raise ValueError(
                f"Benchmark report is empty: {path}"
            )

    saved_benchmark_df = pd.read_csv(
        benchmark_csv_path
    )

    saved_summary = json.loads(
        Path(
            summary_json_path
        ).read_text(
            encoding="utf-8"
        )
    )

    saved_report = Path(
        report_txt_path
    ).read_text(
        encoding="utf-8"
    )

    if expected_rows is not None:
        if len(saved_benchmark_df) != expected_rows:
            raise AssertionError(
                "Unexpected saved benchmark row count: "
                f"{len(saved_benchmark_df)}"
            )

    if (
        saved_summary.get("benchmark_runs")
        != len(saved_benchmark_df)
    ):
        raise AssertionError(
            "Saved summary run count does not match "
            "the saved CSV row count."
        )

    if (
        "INTEGRATED AI BENCHMARK REPORT"
        not in saved_report
    ):
        raise AssertionError(
            "Saved benchmark report heading is missing."
        )

    return {
        "benchmark_rows": len(
            saved_benchmark_df
        ),
        "benchmark_runs": saved_summary[
            "benchmark_runs"
        ],
        "report_characters": len(
            saved_report
        ),
    }
