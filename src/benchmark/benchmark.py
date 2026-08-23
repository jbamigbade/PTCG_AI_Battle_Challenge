"""Benchmark configuration models and default experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """One configurable advanced-search benchmark entrant."""

    name: str
    use_transposition_table: bool
    use_move_ordering: bool
    use_killer_moves: bool
    use_history_heuristic: bool
    seed: int

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "use_transposition_table": self.use_transposition_table,
            "use_move_ordering": self.use_move_ordering,
            "use_killer_moves": self.use_killer_moves,
            "use_history_heuristic": self.use_history_heuristic,
            "seed": self.seed,
        }


DEFAULT_BENCHMARK_CONFIGURATIONS = [
    BenchmarkConfiguration(
        name="Baseline",
        use_transposition_table=False,
        use_move_ordering=False,
        use_killer_moves=False,
        use_history_heuristic=False,
        seed=101,
    ),
    BenchmarkConfiguration(
        name="Transposition Only",
        use_transposition_table=True,
        use_move_ordering=False,
        use_killer_moves=False,
        use_history_heuristic=False,
        seed=202,
    ),
    BenchmarkConfiguration(
        name="Move Ordering Only",
        use_transposition_table=False,
        use_move_ordering=True,
        use_killer_moves=False,
        use_history_heuristic=False,
        seed=303,
    ),
    BenchmarkConfiguration(
        name="Fully Optimized",
        use_transposition_table=True,
        use_move_ordering=True,
        use_killer_moves=True,
        use_history_heuristic=True,
        seed=404,
    ),
]


DEFAULT_BENCHMARK_DEPTHS = [
    2,
    4,
    6,
    8,
]


DEFAULT_BENCHMARK_PAIRINGS = [
    (
        "Fully Optimized",
        "Baseline",
    ),
    (
        "Baseline",
        "Fully Optimized",
    ),
    (
        "Transposition Only",
        "Move Ordering Only",
    ),
    (
        "Move Ordering Only",
        "Transposition Only",
    ),
]
