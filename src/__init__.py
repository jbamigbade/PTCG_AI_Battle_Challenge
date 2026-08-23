"""Reusable components for the PTCG AI Battle Challenge."""

from .battle_state import (
    BattleState,
    PlayerState,
    PokemonState,
)
from .legal_moves import (
    get_current_legal_moves,
    get_current_side_pokemon,
    get_legal_moves,
)
from .evaluation import evaluate_position
from .simulator import apply_move
from .adapters import (
    current_player_adapter,
    evaluate_state_adapter,
    generate_moves_adapter,
    pokemon_state_features,
    terminal_state_adapter,
)

__all__ = [
    "BattleState",
    "PlayerState",
    "PokemonState",
    "get_legal_moves",
    "get_current_side_pokemon",
    "get_current_legal_moves",
    "evaluate_position",
    "apply_move",
    "generate_moves_adapter",
    "evaluate_state_adapter",
    "current_player_adapter",
    "terminal_state_adapter",
    "pokemon_state_features",
]

from .agent_decision import AgentDecision
from .battle_agent import PokemonBattleAgent
from .battle_simulation import (
    BattleSimulationResult,
    BattleTurnRecord,
    create_battle_transcript,
    determine_battle_winner,
    simulate_ai_battle,
)

from .engine.advanced_search import (
    AdvancedSearchEngine,
    BoundType,
    SearchResult,
    SearchStats,
    TranspositionEntry,
    ZobristHasher,
)

from .tournament import (
    TournamentMatch,
    TournamentResult,
    TournamentStatistics,
    run_two_agent_match,
    update_statistics,
)

from .tournament import (
    LEADERBOARD_COLUMNS,
    TournamentEntrant,
    TournamentManager,
    build_leaderboard_dataframe,
    create_round_robin_schedule,
)

from .benchmark import (
    BENCHMARK_COLUMNS,
    DEFAULT_BENCHMARK_CONFIGURATIONS,
    DEFAULT_BENCHMARK_DEPTHS,
    DEFAULT_BENCHMARK_PAIRINGS,
    DEPTH_SUMMARY_COLUMNS,
    BenchmarkConfiguration,
    build_benchmark_dataframe,
    build_benchmark_report,
    build_depth_summary,
    build_overall_summary,
    run_benchmark_match,
    run_benchmark_matrix,
    save_benchmark_reports,
    validate_benchmark_reports,
)
