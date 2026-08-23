"""Benchmark helpers for the advanced search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EngineBenchmarkResult:
    """Normalized result from one completed engine benchmark."""

    engine_name: str
    best_move: str
    score: float
    completed_depth: int

    nodes: int
    leaf_nodes: int
    terminal_nodes: int
    cutoffs: int

    transposition_hits: int
    killer_hits: int
    history_hits: int

    elapsed_seconds: float
    nodes_per_second: float

    principal_variation: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a display-friendly dictionary."""

        return {
            "engine": self.engine_name,
            "best_move": self.best_move,
            "score": self.score,
            "depth": self.completed_depth,
            "nodes": self.nodes,
            "leaf_nodes": self.leaf_nodes,
            "terminal_nodes": self.terminal_nodes,
            "cutoffs": self.cutoffs,
            "tt_hits": self.transposition_hits,
            "killer_hits": self.killer_hits,
            "history_hits": self.history_hits,
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "nodes_per_second": round(
                self.nodes_per_second,
                2,
            ),
            "principal_variation": " → ".join(
                self.principal_variation
            ),
        }


def benchmark_engine(
    engine_name: str,
    engine: Any,
    state: Any,
    depth: int,
) -> EngineBenchmarkResult:
    """Run one fixed-depth engine benchmark."""

    result = engine.search_depth(
        state=state,
        depth=depth,
        clear_table=True,
        clear_killers=True,
        clear_history=True,
    )

    best_move_name = (
        engine.move_to_string(result.best_move)
        if result.best_move is not None
        else "None"
    )

    pv_names = [
        engine.move_to_string(move)
        for move in result.principal_variation
    ]

    return EngineBenchmarkResult(
        engine_name=engine_name,
        best_move=best_move_name,
        score=float(result.score),
        completed_depth=result.completed_depth,

        nodes=result.stats.nodes,
        leaf_nodes=result.stats.leaf_nodes,
        terminal_nodes=result.stats.terminal_nodes,
        cutoffs=result.stats.cutoffs,

        transposition_hits=(
            result.stats.transposition_hits
        ),
        killer_hits=result.stats.killer_hits,
        history_hits=result.stats.history_hits,

        elapsed_seconds=result.stats.elapsed_seconds,
        nodes_per_second=result.stats.nodes_per_second,

        principal_variation=pv_names,
    )
