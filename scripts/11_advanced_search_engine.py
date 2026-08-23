from __future__ import annotations

#!/usr/bin/env python
# coding: utf-8

# # Notebook 11 â€” Advanced Search Engine
# 
# ## PokÃ©mon TCG AI Battle Challenge Simulation
# 
# This notebook upgrades the search engine with:
# 
# 1. Iterative Deepening Search
# 2. Transposition Tables
# 3. Zobrist Hashing
# 4. Move Ordering
# 5. Killer Move Heuristic
# 6. History Heuristic
# 7. Principal Variation Extraction
# 8. Time-Limited Search
# 9. Engine Benchmarking

# ## Step 1 â€” Create Notebook 11

# In[1]:


import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    BattleState,
    PlayerState,
    PokemonState,
    get_legal_moves,
    get_current_legal_moves,
    evaluate_position,
    apply_move,
    generate_moves_adapter,
    evaluate_state_adapter,
    current_player_adapter,
    terminal_state_adapter,
    pokemon_state_features,
)

print("âœ… PokÃ©mon battle modules imported successfully.")


# ## Step 2 â€” Imports

# In[14]:



from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import math
import time


# ## Step 3 â€” Search Type Aliases

# In[15]:


# Generic types used by the advanced search engine.
#
# We intentionally use Any for State and Move because Notebook 11
# should work with the PokÃ©mon TCG classes created in earlier notebooks.

State = Any
Move = Any
Player = Hashable

GenerateMovesFn = Callable[[State], Sequence[Move]]
ApplyMoveFn = Callable[[State, Move], State]
EvaluateFn = Callable[[State, Player], float]
IsTerminalFn = Callable[[State], bool]
CurrentPlayerFn = Callable[[State], Player]
MoveToStringFn = Callable[[Move], str]


# ## Step 4 â€” Search Statistics

# In[16]:


@dataclass
class SearchStats:
    """
    Statistics collected during one complete search.
    """

    nodes: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    cutoffs: int = 0

    completed_depth: int = 0
    elapsed_seconds: float = 0.0

    depth_nodes: Dict[int, int] = field(default_factory=dict)
    depth_times: Dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all statistics."""

        self.nodes = 0
        self.leaf_nodes = 0
        self.terminal_nodes = 0
        self.cutoffs = 0

        self.completed_depth = 0
        self.elapsed_seconds = 0.0

        self.depth_nodes.clear()
        self.depth_times.clear()

    @property
    def nodes_per_second(self) -> float:
        """Return the average number of searched nodes per second."""

        if self.elapsed_seconds <= 0:
            return 0.0

        return self.nodes / self.elapsed_seconds

    def as_dict(self) -> Dict[str, Any]:
        """Return statistics in dictionary form."""

        return {
            "nodes": self.nodes,
            "leaf_nodes": self.leaf_nodes,
            "terminal_nodes": self.terminal_nodes,
            "cutoffs": self.cutoffs,
            "completed_depth": self.completed_depth,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "nodes_per_second": round(self.nodes_per_second, 2),
            "depth_nodes": dict(self.depth_nodes),
            "depth_times": {
                depth: round(seconds, 6)
                for depth, seconds in self.depth_times.items()
            },
        }


# ## Step 5 â€” Search Result Object

# In[17]:


@dataclass
class SearchResult:
    """
    Final result returned by the search engine.
    """

    best_move: Optional[Move]
    score: float
    completed_depth: int

    principal_variation: List[Move] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)

    def as_dict(
        self,
        move_to_string: Optional[MoveToStringFn] = None,
    ) -> Dict[str, Any]:
        """
        Convert the result into a display-friendly dictionary.
        """

        formatter = move_to_string or str

        return {
            "best_move": (
                formatter(self.best_move)
                if self.best_move is not None
                else None
            ),
            "score": self.score,
            "completed_depth": self.completed_depth,
            "principal_variation": [
                formatter(move)
                for move in self.principal_variation
            ],
            "stats": self.stats.as_dict(),
        }


# ## Step 6 â€” Advanced Search Engine Skeleton
# 
# This class contains every Notebook 11's optimization.
# 
# For now, we are implementing:
# 
# Alpha-beta search
# Iterative deepening
# Search statistics
# Basic principal-variation tracking

# In[18]:


class AdvancedSearchEngine:
    """
    Advanced adversarial search engine for the PokÃ©mon TCG simulator.

    The engine communicates with the game through callback functions.
    Therefore, it does not depend on one specific GameState class.

    Notebook 11 will progressively add:

    - Iterative deepening
    - Transposition tables
    - Zobrist hashing
    - Move ordering
    - Killer moves
    - History heuristic
    - Principal variation extraction
    - Time limits
    """

    def __init__(
        self,
        generate_moves: GenerateMovesFn,
        apply_move: ApplyMoveFn,
        evaluate_state: EvaluateFn,
        is_terminal: IsTerminalFn,
        current_player: CurrentPlayerFn,
        move_to_string: Optional[MoveToStringFn] = None,
    ) -> None:
        """
        Initialize the search engine.

        Parameters
        ----------
        generate_moves:
            Returns all legal moves for a state.

        apply_move:
            Returns the successor state produced by a move.

            Important:
            This function should not permanently modify the original state.
            It should either return a copied state or use the safe state-copy
            behavior established in the earlier notebooks.

        evaluate_state:
            Evaluates the state from the root player's perspective.

            Positive score:
                Favors the root player.

            Negative score:
                Favors the opponent.

        is_terminal:
            Returns True when the game is over.

        current_player:
            Returns the player whose turn it is.

        move_to_string:
            Optional function for displaying moves.
        """

        self.generate_moves = generate_moves
        self.apply_move = apply_move
        self.evaluate_state = evaluate_state
        self.is_terminal = is_terminal
        self.current_player = current_player
        self.move_to_string = move_to_string or str

        self.stats = SearchStats()

        # Stores the best line found during the current depth search.
        self._principal_variation: List[Move] = []

    def _validate_depth(self, depth: int) -> None:
        """Validate a requested search depth."""

        if not isinstance(depth, int):
            raise TypeError(
                f"Search depth must be an integer, received {type(depth).__name__}."
            )

        if depth < 1:
            raise ValueError(
                f"Search depth must be at least 1, received {depth}."
            )

    def _safe_moves(self, state: State) -> List[Move]:
        """
        Generate legal moves and convert them to a normal list.
        """

        moves = self.generate_moves(state)

        if moves is None:
            return []

        return list(moves)

    def _alpha_beta(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
    ) -> Tuple[float, List[Move]]:
        """
        Search one position using alpha-beta pruning.

        Returns
        -------
        score:
            Evaluation of the position.

        line:
            Best move sequence discovered from this position.
        """

        self.stats.nodes += 1

        if self.is_terminal(state):
            self.stats.terminal_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        if depth <= 0:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        legal_moves = self._safe_moves(state)

        # A position with no legal moves is treated like a leaf.
        if not legal_moves:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        player_to_move = self.current_player(state)
        maximizing = player_to_move == root_player

        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []

            for move in legal_moves:
                child_state = self.apply_move(state, move)

                child_score, child_line = self._alpha_beta(
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                )

                if child_score > best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                alpha = max(alpha, best_score)

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break

            return best_score, best_line

        best_score = math.inf
        best_line = []

        for move in legal_moves:
            child_state = self.apply_move(state, move)

            child_score, child_line = self._alpha_beta(
                state=child_state,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
            )

            if child_score < best_score:
                best_score = child_score
                best_line = [move] + child_line

            beta = min(beta, best_score)

            if alpha >= beta:
                self.stats.cutoffs += 1
                break

        return best_score, best_line

    def search_depth(
        self,
        state: State,
        depth: int,
        root_player: Optional[Player] = None,
    ) -> SearchResult:
        """
        Perform one fixed-depth alpha-beta search.
        """

        self._validate_depth(depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        start_time = time.perf_counter()

        score, principal_variation = self._alpha_beta(
            state=state,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            root_player=root_player,
        )

        elapsed = time.perf_counter() - start_time

        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed
        self.stats.depth_nodes[depth] = self.stats.nodes
        self.stats.depth_times[depth] = elapsed

        best_move = (
            principal_variation[0]
            if principal_variation
            else None
        )

        self._principal_variation = principal_variation

        return SearchResult(
            best_move=best_move,
            score=score,
            completed_depth=depth,
            principal_variation=principal_variation,
            stats=self.stats,
        )

    def iterative_deepening(
        self,
        state: State,
        max_depth: int,
        root_player: Optional[Player] = None,
        verbose: bool = True,
    ) -> SearchResult:
        """
        Search successively from depth 1 through max_depth.

        The result from the deepest completed search is returned.
        """

        self._validate_depth(max_depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        overall_start = time.perf_counter()

        best_move: Optional[Move] = None
        best_score = float(
            self.evaluate_state(state, root_player)
        )
        best_line: List[Move] = []

        total_nodes = 0
        total_leaf_nodes = 0
        total_terminal_nodes = 0
        total_cutoffs = 0

        depth_nodes: Dict[int, int] = {}
        depth_times: Dict[int, float] = {}

        completed_depth = 0

        for depth in range(1, max_depth + 1):
            # Reset counters for this individual iteration.
            self.stats.nodes = 0
            self.stats.leaf_nodes = 0
            self.stats.terminal_nodes = 0
            self.stats.cutoffs = 0

            depth_start = time.perf_counter()

            score, principal_variation = self._alpha_beta(
                state=state,
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
            )

            depth_elapsed = time.perf_counter() - depth_start

            iteration_nodes = self.stats.nodes

            depth_nodes[depth] = iteration_nodes
            depth_times[depth] = depth_elapsed

            total_nodes += self.stats.nodes
            total_leaf_nodes += self.stats.leaf_nodes
            total_terminal_nodes += self.stats.terminal_nodes
            total_cutoffs += self.stats.cutoffs

            completed_depth = depth
            best_score = score
            best_line = principal_variation

            if principal_variation:
                best_move = principal_variation[0]

            if verbose:
                move_name = (
                    self.move_to_string(best_move)
                    if best_move is not None
                    else "None"
                )

                pv_text = " -> ".join(
                    self.move_to_string(move)
                    for move in principal_variation
                )

                print(
                    f"Depth {depth:>2} | "
                    f"Score {score:>10.2f} | "
                    f"Nodes {iteration_nodes:>8,} | "
                    f"Time {depth_elapsed:>8.4f}s | "
                    f"Best move: {move_name}"
                )

                if pv_text:
                    print(f"         PV: {pv_text}")

        overall_elapsed = time.perf_counter() - overall_start

        final_stats = SearchStats(
            nodes=total_nodes,
            leaf_nodes=total_leaf_nodes,
            terminal_nodes=total_terminal_nodes,
            cutoffs=total_cutoffs,
            completed_depth=completed_depth,
            elapsed_seconds=overall_elapsed,
            depth_nodes=depth_nodes,
            depth_times=depth_times,
        )

        self.stats = final_stats
        self._principal_variation = best_line

        return SearchResult(
            best_move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=best_line,
            stats=final_stats,
        )

    def get_principal_variation(self) -> List[Move]:
        """
        Return a copy of the best line from the most recent search.
        """

        return list(self._principal_variation)


# ## Step 7 â€” Create a Small Validation Game
# 
# Before connecting the engine to the full PokÃ©mon TCG environment, we need to prove that iterative deepening works correctly.
# 
# This test game lets two players subtract either 1, 2, or 3 from a pile. The player who reaches zero wins.

# In[19]:


@dataclass(frozen=True)
class SubtractionState:
    """
    Small deterministic game used to validate the search engine.
    """

    remaining: int
    player_to_move: int


def subtraction_generate_moves(
    state: SubtractionState,
) -> List[int]:
    """
    A move subtracts 1, 2, or 3 from the remaining total.
    """

    return [
        amount
        for amount in (1, 2, 3)
        if amount <= state.remaining
    ]


def subtraction_apply_move(
    state: SubtractionState,
    move: int,
) -> SubtractionState:
    """
    Return the next immutable game state.
    """

    if move not in subtraction_generate_moves(state):
        raise ValueError(
            f"Illegal move {move} for state {state}."
        )

    return SubtractionState(
        remaining=state.remaining - move,
        player_to_move=1 - state.player_to_move,
    )


def subtraction_is_terminal(
    state: SubtractionState,
) -> bool:
    return state.remaining == 0


def subtraction_current_player(
    state: SubtractionState,
) -> int:
    return state.player_to_move


def subtraction_evaluate(
    state: SubtractionState,
    root_player: int,
) -> float:
    """
    Evaluate the state from the root player's perspective.

    When remaining reaches zero, the player whose turn would be next
    has lost because the previous player removed the final token.
    """

    if state.remaining == 0:
        winner = 1 - state.player_to_move

        return 10_000.0 if winner == root_player else -10_000.0

    # Simple non-terminal heuristic.
    #
    # Multiples of 4 are generally losing positions in this game.
    if state.remaining % 4 == 0:
        return -10.0 if state.player_to_move == root_player else 10.0

    return 5.0 if state.player_to_move == root_player else -5.0


def subtraction_move_to_string(move: int) -> str:
    return f"Subtract {move}"


# ## Step 8 â€” Create the Test Engine

# In[20]:


test_engine = AdvancedSearchEngine(
    generate_moves=subtraction_generate_moves,
    apply_move=subtraction_apply_move,
    evaluate_state=subtraction_evaluate,
    is_terminal=subtraction_is_terminal,
    current_player=subtraction_current_player,
    move_to_string=subtraction_move_to_string,
)

test_state = SubtractionState(
    remaining=10,
    player_to_move=0,
)

print("Test state:", test_state)
print(
    "Legal moves:",
    subtraction_generate_moves(test_state),
)


# ## Step 9 â€” Test Fixed-Depth Search

# In[21]:


fixed_result = test_engine.search_depth(
    state=test_state,
    depth=6,
)

fixed_result.as_dict(
    move_to_string=subtraction_move_to_string
)


# ## Step 10 â€” Test Iterative Deepening

# In[22]:


iterative_result = test_engine.iterative_deepening(
    state=test_state,
    max_depth=8,
    verbose=True,
)


# In[23]:


iterative_result.as_dict(
    move_to_string=subtraction_move_to_string
)


# ### Step 11 â€” Validate the Result Automatically

# In[24]:


assert iterative_result.completed_depth == 8
assert iterative_result.best_move in (1, 2, 3)
assert iterative_result.stats.nodes > 0
assert iterative_result.stats.elapsed_seconds >= 0
assert len(iterative_result.stats.depth_nodes) == 8
assert len(iterative_result.stats.depth_times) == 8

if iterative_result.principal_variation:
    assert (
        iterative_result.best_move
        == iterative_result.principal_variation[0]
    )

print("âœ… Iterative deepening validation passed.")
print(
    "Best move:",
    subtraction_move_to_string(
        iterative_result.best_move
    ),
)
print(
    "Completed depth:",
    iterative_result.completed_depth,
)
print(
    "Total nodes:",
    f"{iterative_result.stats.nodes:,}",
)
print(
    "Nodes per second:",
    f"{iterative_result.stats.nodes_per_second:,.2f}",
)


# ## Step 12 â€” PokÃ©mon Engine Integration Deferred
# 
# The PokÃ©mon TCG engine connection is intentionally deferred until after:
# 
# - Part 2 â€” Transposition Tables
# - Part 3 â€” Zobrist Hashing
# 
# This ensures that the actual PokÃ©mon search engine is created with the completed caching and position-hashing system.
# 
# The final integration appears after Part 3 and before Part 4.

# In[26]:


# pokemon_search_engine = AdvancedSearchEngine(
#     generate_moves=generate_legal_actions,
#     apply_move=apply_action_to_copy,
#     evaluate_state=evaluate_game_state,
#     is_terminal=is_game_over,
#     current_player=get_current_player,
#     move_to_string=lambda move: str(move),
# )


# # Part 2 â€” Transposition Tables
# 
# A transposition table stores positions that the engine has already searched.
# 
# Different move sequences can sometimes reach the same game state. Without
# caching, the engine searches that position repeatedly. A transposition table
# allows the engine to reuse the earlier result.
# 
# Each stored entry will contain:
# 
# - The searched depth
# - The evaluated score
# - The bound type
# - The best move
# - The principal variation

# ## Step 13 â€” Add Transposition Table Types

# In[27]:


from enum import Enum, auto


# In[28]:


class BoundType(Enum):
    """
    Describes how a transposition-table score should be interpreted.
    """

    EXACT = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()


@dataclass
class TranspositionEntry:
    """
    One cached search result.
    """

    depth: int
    score: float
    bound_type: BoundType

    best_move: Optional[Move] = None
    principal_variation: List[Move] = field(default_factory=list)


# ## Step 14 â€” Extend Search Statistics

# In[29]:


@dataclass
class SearchStats:
    """
    Statistics collected during one complete search.
    """

    nodes: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    cutoffs: int = 0

    transposition_hits: int = 0
    transposition_stores: int = 0
    transposition_cutoffs: int = 0

    completed_depth: int = 0
    elapsed_seconds: float = 0.0

    depth_nodes: Dict[int, int] = field(default_factory=dict)
    depth_times: Dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all statistics."""

        self.nodes = 0
        self.leaf_nodes = 0
        self.terminal_nodes = 0
        self.cutoffs = 0

        self.transposition_hits = 0
        self.transposition_stores = 0
        self.transposition_cutoffs = 0

        self.completed_depth = 0
        self.elapsed_seconds = 0.0

        self.depth_nodes.clear()
        self.depth_times.clear()

    @property
    def nodes_per_second(self) -> float:
        """Return the average searched nodes per second."""

        if self.elapsed_seconds <= 0:
            return 0.0

        return self.nodes / self.elapsed_seconds

    @property
    def transposition_hit_rate(self) -> float:
        """
        Return cache hits as a fraction of searched nodes.
        """

        if self.nodes <= 0:
            return 0.0

        return self.transposition_hits / self.nodes

    def as_dict(self) -> Dict[str, Any]:
        """Return statistics in dictionary form."""

        return {
            "nodes": self.nodes,
            "leaf_nodes": self.leaf_nodes,
            "terminal_nodes": self.terminal_nodes,
            "cutoffs": self.cutoffs,
            "transposition_hits": self.transposition_hits,
            "transposition_stores": self.transposition_stores,
            "transposition_cutoffs": self.transposition_cutoffs,
            "transposition_hit_rate": round(
                self.transposition_hit_rate,
                6,
            ),
            "completed_depth": self.completed_depth,
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "nodes_per_second": round(
                self.nodes_per_second,
                2,
            ),
            "depth_nodes": dict(self.depth_nodes),
            "depth_times": {
                depth: round(seconds, 6)
                for depth, seconds in self.depth_times.items()
            },
        }


# ## Step 15 â€” Replace the Advanced Search Engine

# In[30]:


class AdvancedSearchEngine:
    """
    Advanced adversarial search engine with iterative deepening
    and transposition-table support.
    """

    def __init__(
        self,
        generate_moves: GenerateMovesFn,
        apply_move: ApplyMoveFn,
        evaluate_state: EvaluateFn,
        is_terminal: IsTerminalFn,
        current_player: CurrentPlayerFn,
        move_to_string: Optional[MoveToStringFn] = None,
        state_key: Optional[Callable[[State], Hashable]] = None,
        use_transposition_table: bool = True,
    ) -> None:
        """
        Initialize the search engine.

        state_key
        ---------
        Converts a state into a stable hashable key.

        When state_key is omitted, the engine attempts to use the state
        object itself as the dictionary key. This works for immutable,
        hashable states such as frozen dataclasses.
        """

        self.generate_moves = generate_moves
        self.apply_move = apply_move
        self.evaluate_state = evaluate_state
        self.is_terminal = is_terminal
        self.current_player = current_player
        self.move_to_string = move_to_string or str

        self.state_key = state_key or self._default_state_key
        self.use_transposition_table = use_transposition_table

        self.stats = SearchStats()

        self.transposition_table: Dict[
            Hashable,
            TranspositionEntry,
        ] = {}

        self._principal_variation: List[Move] = []

    @staticmethod
    def _default_state_key(state: State) -> Hashable:
        """
        Use the state itself as its cache key.

        Raises a clear error when the state is not hashable.
        """

        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "The game state is not hashable. Supply a state_key "
                "function when creating AdvancedSearchEngine."
            ) from exc

        return state

    def _validate_depth(self, depth: int) -> None:
        """Validate a requested search depth."""

        if not isinstance(depth, int):
            raise TypeError(
                "Search depth must be an integer, "
                f"received {type(depth).__name__}."
            )

        if depth < 1:
            raise ValueError(
                f"Search depth must be at least 1, received {depth}."
            )

    def _safe_moves(self, state: State) -> List[Move]:
        """
        Generate legal moves and convert them to a list.
        """

        moves = self.generate_moves(state)

        if moves is None:
            return []

        return list(moves)

    def clear_transposition_table(self) -> None:
        """
        Remove every cached position.
        """

        self.transposition_table.clear()

    def transposition_table_size(self) -> int:
        """
        Return the number of cached positions.
        """

        return len(self.transposition_table)

    def _probe_transposition_table(
        self,
        key: Hashable,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[float, List[Move]]]:
        """
        Attempt to reuse a cached search result.
        """

        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        if entry.depth < depth:
            return None

        self.stats.transposition_hits += 1

        if entry.bound_type is BoundType.EXACT:
            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type is BoundType.LOWER_BOUND
            and entry.score >= beta
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type is BoundType.UPPER_BOUND
            and entry.score <= alpha
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        return None

    def _store_transposition_entry(
        self,
        key: Hashable,
        depth: int,
        score: float,
        alpha_original: float,
        beta_original: float,
        best_line: List[Move],
    ) -> None:
        """
        Store a completed node search in the cache.
        """

        if not self.use_transposition_table:
            return

        if score <= alpha_original:
            bound_type = BoundType.UPPER_BOUND
        elif score >= beta_original:
            bound_type = BoundType.LOWER_BOUND
        else:
            bound_type = BoundType.EXACT

        existing = self.transposition_table.get(key)

        # Prefer deeper search results over shallower ones.
        if existing is not None and existing.depth > depth:
            return

        best_move = best_line[0] if best_line else None

        self.transposition_table[key] = TranspositionEntry(
            depth=depth,
            score=score,
            bound_type=bound_type,
            best_move=best_move,
            principal_variation=list(best_line),
        )

        self.stats.transposition_stores += 1

    def _alpha_beta(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
    ) -> Tuple[float, List[Move]]:
        """
        Search one position using alpha-beta pruning and caching.
        """

        self.stats.nodes += 1

        if self.is_terminal(state):
            self.stats.terminal_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        if depth <= 0:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        key = self.state_key(state)

        cached_result = self._probe_transposition_table(
            key=key,
            depth=depth,
            alpha=alpha,
            beta=beta,
        )

        if cached_result is not None:
            return cached_result

        legal_moves = self._safe_moves(state)

        if not legal_moves:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(state, root_player)
            )

            return score, []

        alpha_original = alpha
        beta_original = beta

        player_to_move = self.current_player(state)
        maximizing = player_to_move == root_player

        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []

            for move in legal_moves:
                child_state = self.apply_move(state, move)

                child_score, child_line = self._alpha_beta(
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                )

                if child_score > best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                alpha = max(alpha, best_score)

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break

        else:
            best_score = math.inf
            best_line = []

            for move in legal_moves:
                child_state = self.apply_move(state, move)

                child_score, child_line = self._alpha_beta(
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                )

                if child_score < best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                beta = min(beta, best_score)

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break

        self._store_transposition_entry(
            key=key,
            depth=depth,
            score=best_score,
            alpha_original=alpha_original,
            beta_original=beta_original,
            best_line=best_line,
        )

        return best_score, best_line

    def search_depth(
        self,
        state: State,
        depth: int,
        root_player: Optional[Player] = None,
        clear_table: bool = True,
    ) -> SearchResult:
        """
        Perform one fixed-depth alpha-beta search.
        """

        self._validate_depth(depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        start_time = time.perf_counter()

        score, principal_variation = self._alpha_beta(
            state=state,
            depth=depth,
            alpha=-math.inf,
            beta=math.inf,
            root_player=root_player,
        )

        elapsed = time.perf_counter() - start_time

        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed
        self.stats.depth_nodes[depth] = self.stats.nodes
        self.stats.depth_times[depth] = elapsed

        best_move = (
            principal_variation[0]
            if principal_variation
            else None
        )

        self._principal_variation = principal_variation

        return SearchResult(
            best_move=best_move,
            score=score,
            completed_depth=depth,
            principal_variation=principal_variation,
            stats=self.stats,
        )

    def iterative_deepening(
        self,
        state: State,
        max_depth: int,
        root_player: Optional[Player] = None,
        verbose: bool = True,
        clear_table: bool = True,
    ) -> SearchResult:
        """
        Search successively from depth 1 through max_depth.

        The transposition table is retained between depth iterations so
        earlier searches can help deeper searches.
        """

        self._validate_depth(max_depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        overall_start = time.perf_counter()

        best_move: Optional[Move] = None
        best_score = float(
            self.evaluate_state(state, root_player)
        )
        best_line: List[Move] = []

        total_nodes = 0
        total_leaf_nodes = 0
        total_terminal_nodes = 0
        total_cutoffs = 0

        total_tt_hits = 0
        total_tt_stores = 0
        total_tt_cutoffs = 0

        depth_nodes: Dict[int, int] = {}
        depth_times: Dict[int, float] = {}

        completed_depth = 0

        for depth in range(1, max_depth + 1):
            self.stats.nodes = 0
            self.stats.leaf_nodes = 0
            self.stats.terminal_nodes = 0
            self.stats.cutoffs = 0

            self.stats.transposition_hits = 0
            self.stats.transposition_stores = 0
            self.stats.transposition_cutoffs = 0

            depth_start = time.perf_counter()

            score, principal_variation = self._alpha_beta(
                state=state,
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
            )

            depth_elapsed = (
                time.perf_counter() - depth_start
            )

            iteration_nodes = self.stats.nodes

            depth_nodes[depth] = iteration_nodes
            depth_times[depth] = depth_elapsed

            total_nodes += self.stats.nodes
            total_leaf_nodes += self.stats.leaf_nodes
            total_terminal_nodes += self.stats.terminal_nodes
            total_cutoffs += self.stats.cutoffs

            total_tt_hits += self.stats.transposition_hits
            total_tt_stores += self.stats.transposition_stores
            total_tt_cutoffs += (
                self.stats.transposition_cutoffs
            )

            completed_depth = depth
            best_score = score
            best_line = principal_variation

            if principal_variation:
                best_move = principal_variation[0]

            if verbose:
                move_name = (
                    self.move_to_string(best_move)
                    if best_move is not None
                    else "None"
                )

                pv_text = " -> ".join(
                    self.move_to_string(move)
                    for move in principal_variation
                )

                print(
                    f"Depth {depth:>2} | "
                    f"Score {score:>10.2f} | "
                    f"Nodes {iteration_nodes:>8,} | "
                    f"TT hits "
                    f"{self.stats.transposition_hits:>6,} | "
                    f"Time {depth_elapsed:>8.4f}s | "
                    f"Best move: {move_name}"
                )

                if pv_text:
                    print(f"         PV: {pv_text}")

        overall_elapsed = (
            time.perf_counter() - overall_start
        )

        final_stats = SearchStats(
            nodes=total_nodes,
            leaf_nodes=total_leaf_nodes,
            terminal_nodes=total_terminal_nodes,
            cutoffs=total_cutoffs,
            transposition_hits=total_tt_hits,
            transposition_stores=total_tt_stores,
            transposition_cutoffs=total_tt_cutoffs,
            completed_depth=completed_depth,
            elapsed_seconds=overall_elapsed,
            depth_nodes=depth_nodes,
            depth_times=depth_times,
        )

        self.stats = final_stats
        self._principal_variation = best_line

        return SearchResult(
            best_move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=best_line,
            stats=final_stats,
        )

    def get_principal_variation(self) -> List[Move]:
        """
        Return a copy of the best line from the latest search.
        """

        return list(self._principal_variation)


# ## Step 16 â€” Recreate the Test Engine

# In[31]:


test_engine = AdvancedSearchEngine(
    generate_moves=subtraction_generate_moves,
    apply_move=subtraction_apply_move,
    evaluate_state=subtraction_evaluate,
    is_terminal=subtraction_is_terminal,
    current_player=subtraction_current_player,
    move_to_string=subtraction_move_to_string,
    state_key=lambda state: (
        state.remaining,
        state.player_to_move,
    ),
    use_transposition_table=True,
)

test_state = SubtractionState(
    remaining=10,
    player_to_move=0,
)

print("Transposition table enabled.")
print("Initial table size:", test_engine.transposition_table_size())


# ## Step 17 â€” Test the Cached Search

# In[32]:


cached_result = test_engine.iterative_deepening(
    state=test_state,
    max_depth=8,
    verbose=True,
)

cached_result.as_dict(
    move_to_string=subtraction_move_to_string
)


# ## Step 18 â€” Inspect the Cache

# In[33]:


print(
    "Cached positions:",
    test_engine.transposition_table_size(),
)

print(
    "Transposition hits:",
    cached_result.stats.transposition_hits,
)

print(
    "Transposition stores:",
    cached_result.stats.transposition_stores,
)

print(
    "Transposition cutoffs:",
    cached_result.stats.transposition_cutoffs,
)

print(
    "Hit rate:",
    f"{cached_result.stats.transposition_hit_rate:.2%}",
)


# ## Step 19 â€” Validate the Transposition Table

# In[34]:


assert cached_result.best_move == 2
assert cached_result.score == 10_000.0
assert cached_result.completed_depth == 8

assert test_engine.transposition_table_size() > 0
assert cached_result.stats.transposition_hits > 0
assert cached_result.stats.transposition_stores > 0

assert (
    cached_result.best_move
    == cached_result.principal_variation[0]
)

print("âœ… Transposition-table validation passed.")
print(
    "Cached positions:",
    test_engine.transposition_table_size(),
)
print(
    "Cache hits:",
    cached_result.stats.transposition_hits,
)


# ## Step 20 â€” Add the Zobrist Hashing Section

# # Part 3 â€” Zobrist Hashing
# 
# Zobrist hashing creates a compact integer key for a game position.
# 
# Each position feature receives a deterministic random 64-bit number.
# The numbers for all active features are combined using the XOR operation.
# 
# Benefits:
# 
# - Fast position-key generation
# - Compact transposition-table keys
# - Efficient incremental updates
# - Suitable for complex game states
# - Deterministic results when a fixed seed is used

# In[35]:


import random


# In[36]:


StateFeaturesFn = Callable[[State], Sequence[Hashable]]


# In[ ]:





# ## Step 22 â€” Create the Generic Zobrist Hasher

# In[37]:


class ZobristHasher:
    """
    Deterministic 64-bit Zobrist hashing system.

    Each distinct feature receives one pseudo-random 64-bit value.
    A state's final hash is the XOR combination of all feature values.

    The fixed seed ensures that identical features receive identical
    values during the current program run and after a hasher reset.
    """

    MASK_64 = (1 << 64) - 1

    def __init__(
        self,
        state_features: StateFeaturesFn,
        seed: int = 20260712,
    ) -> None:
        """
        Initialize the hasher.

        Parameters
        ----------
        state_features:
            Function that converts a game state into a sequence of
            hashable position features.

        seed:
            Fixed random seed used to produce deterministic values.
        """

        if not callable(state_features):
            raise TypeError(
                "state_features must be a callable function."
            )

        if not isinstance(seed, int):
            raise TypeError(
                "Zobrist seed must be an integer."
            )

        self.state_features = state_features
        self.seed = seed

        self._random = random.Random(seed)

        self._feature_values: Dict[
            Hashable,
            int,
        ] = {}

    def clear(self) -> None:
        """
        Clear generated feature values and restore the original seed.

        After clearing, hashing the same features in the same order
        produces the same values again.
        """

        self._feature_values.clear()
        self._random = random.Random(self.seed)

    def feature_count(self) -> int:
        """
        Return the number of distinct registered features.
        """

        return len(self._feature_values)

    def _value_for_feature(
        self,
        feature: Hashable,
    ) -> int:
        """
        Return the 64-bit random number assigned to one feature.
        """

        try:
            hash(feature)
        except TypeError as exc:
            raise TypeError(
                "Every Zobrist feature must be hashable. "
                f"Received feature: {feature!r}"
            ) from exc

        if feature not in self._feature_values:
            value = self._random.getrandbits(64)

            # Avoid assigning zero because XOR with zero has no effect.
            while value == 0:
                value = self._random.getrandbits(64)

            self._feature_values[feature] = value

        return self._feature_values[feature]

    def hash_features(
        self,
        features: Sequence[Hashable],
    ) -> int:
        """
        Hash a supplied sequence of position features.
        """

        zobrist_hash = 0

        for feature in features:
            zobrist_hash ^= self._value_for_feature(feature)

        return zobrist_hash & self.MASK_64

    def hash_state(
        self,
        state: State,
    ) -> int:
        """
        Convert a state into features and return its 64-bit hash.
        """

        features = self.state_features(state)

        if features is None:
            raise ValueError(
                "state_features returned None. It must return "
                "a sequence of hashable features."
            )

        return self.hash_features(list(features))

    def __call__(
        self,
        state: State,
    ) -> int:
        """
        Allow the hasher itself to be used as a state_key function.
        """

        return self.hash_state(state)


# In[38]:


pokemon_zobrist = ZobristHasher(
    state_features=pokemon_state_features,
    seed=20260712,
)

print("âœ… PokÃ©mon Zobrist hasher ready.")


# ## Step 23 â€” Define Features for the Test Game

# In[39]:


def subtraction_state_features(
    state: SubtractionState,
) -> Tuple[Hashable, ...]:
    """
    Convert a subtraction-game position into Zobrist features.

    Each changing state property receives a labeled feature.
    """

    if not isinstance(state, SubtractionState):
        raise TypeError(
            "Expected SubtractionState, "
            f"received {type(state).__name__}."
        )

    return (
        ("game", "subtraction"),
        ("remaining", state.remaining),
        ("player_to_move", state.player_to_move),
    )


# In[40]:


subtraction_zobrist = ZobristHasher(
    state_features=subtraction_state_features,
    seed=20260712,
)

print("Zobrist hasher created.")
print(
    "Registered features:",
    subtraction_zobrist.feature_count(),
)


# ## Step 24 â€” Inspect Several Position Hashes

# In[41]:


zobrist_test_states = [
    SubtractionState(
        remaining=10,
        player_to_move=0,
    ),
    SubtractionState(
        remaining=9,
        player_to_move=1,
    ),
    SubtractionState(
        remaining=8,
        player_to_move=0,
    ),
    SubtractionState(
        remaining=10,
        player_to_move=1,
    ),
]

for state in zobrist_test_states:
    state_hash = subtraction_zobrist.hash_state(state)

    print(
        f"{state} -> "
        f"{state_hash} "
        f"(0x{state_hash:016x})"
    )

print()
print(
    "Registered features:",
    subtraction_zobrist.feature_count(),
)


# ## Step 25 â€” Validate Determinism and Position Separation

# In[42]:


state_a = SubtractionState(
    remaining=10,
    player_to_move=0,
)

state_a_copy = SubtractionState(
    remaining=10,
    player_to_move=0,
)

state_b = SubtractionState(
    remaining=10,
    player_to_move=1,
)

state_c = SubtractionState(
    remaining=9,
    player_to_move=0,
)

hash_a_1 = subtraction_zobrist.hash_state(state_a)
hash_a_2 = subtraction_zobrist.hash_state(state_a)
hash_a_copy = subtraction_zobrist.hash_state(
    state_a_copy
)
hash_b = subtraction_zobrist.hash_state(state_b)
hash_c = subtraction_zobrist.hash_state(state_c)

assert isinstance(hash_a_1, int)
assert 0 <= hash_a_1 <= ZobristHasher.MASK_64

# The same state must always produce the same hash.
assert hash_a_1 == hash_a_2

# Equivalent state objects must produce the same hash.
assert hash_a_1 == hash_a_copy

# Changing the player must change the hash.
assert hash_a_1 != hash_b

# Changing the remaining total must change the hash.
assert hash_a_1 != hash_c

print("âœ… Zobrist determinism validation passed.")
print("State A hash:", hash_a_1)
print("State B hash:", hash_b)
print("State C hash:", hash_c)


# ## Step 26 â€” Recreate the Engine with Zobrist Hashing
# 
# We do not need to replace AdvancedSearchEngine.
# 
# The existing state_key parameter accepts our Zobrist hasher directly because the hasher is callable.

# In[43]:


zobrist_engine = AdvancedSearchEngine(
    generate_moves=subtraction_generate_moves,
    apply_move=subtraction_apply_move,
    evaluate_state=subtraction_evaluate,
    is_terminal=subtraction_is_terminal,
    current_player=subtraction_current_player,
    move_to_string=subtraction_move_to_string,

    # The ZobristHasher replaces the earlier tuple state key.
    state_key=subtraction_zobrist,

    use_transposition_table=True,
)

print("Zobrist-powered search engine created.")
print(
    "Initial transposition-table size:",
    zobrist_engine.transposition_table_size(),
)


# ## Step 27 â€” Run Iterative Deepening with Zobrist Keys

# In[44]:


zobrist_result = zobrist_engine.iterative_deepening(
    state=test_state,
    max_depth=8,
    verbose=True,
)

zobrist_result.as_dict(
    move_to_string=subtraction_move_to_string
)


# ## Step 28 â€” Inspect the Zobrist Transposition Table

# In[45]:


print(
    "Cached Zobrist positions:",
    zobrist_engine.transposition_table_size(),
)

print(
    "Transposition hits:",
    zobrist_result.stats.transposition_hits,
)

print(
    "Transposition stores:",
    zobrist_result.stats.transposition_stores,
)

print(
    "Transposition cutoffs:",
    zobrist_result.stats.transposition_cutoffs,
)

print(
    "Zobrist features registered:",
    subtraction_zobrist.feature_count(),
)

root_hash = subtraction_zobrist.hash_state(test_state)

print(
    "Root state hash:",
    root_hash,
)

print(
    "Root state hexadecimal hash:",
    f"0x{root_hash:016x}",
)

print(
    "Root state stored in table:",
    root_hash in zobrist_engine.transposition_table,
)


# ## Step 29 â€” Compare Tuple Keys and Zobrist Keys

# In[46]:


comparison_rows = [
    {
        "engine": "Tuple-key transposition table",
        "best_move": cached_result.best_move,
        "score": cached_result.score,
        "depth": cached_result.completed_depth,
        "nodes": cached_result.stats.nodes,
        "tt_hits": cached_result.stats.transposition_hits,
        "cached_positions": (
            test_engine.transposition_table_size()
        ),
    },
    {
        "engine": "Zobrist transposition table",
        "best_move": zobrist_result.best_move,
        "score": zobrist_result.score,
        "depth": zobrist_result.completed_depth,
        "nodes": zobrist_result.stats.nodes,
        "tt_hits": (
            zobrist_result.stats.transposition_hits
        ),
        "cached_positions": (
            zobrist_engine.transposition_table_size()
        ),
    },
]

for row in comparison_rows:
    print(
        f"{row['engine']}\n"
        f"  Best move: {row['best_move']}\n"
        f"  Score: {row['score']}\n"
        f"  Depth: {row['depth']}\n"
        f"  Nodes: {row['nodes']:,}\n"
        f"  TT hits: {row['tt_hits']:,}\n"
        f"  Cached positions: "
        f"{row['cached_positions']:,}\n"
    )


# ## Step 30 â€” Final Zobrist Validation

# In[47]:


assert zobrist_result.best_move == 2
assert zobrist_result.score == 10_000.0
assert zobrist_result.completed_depth == 8

assert zobrist_result.principal_variation
assert (
    zobrist_result.principal_variation[0]
    == zobrist_result.best_move
)

assert zobrist_engine.transposition_table_size() > 0
assert zobrist_result.stats.transposition_hits > 0
assert zobrist_result.stats.transposition_stores > 0

assert (
    zobrist_result.best_move
    == cached_result.best_move
)

assert (
    zobrist_result.score
    == cached_result.score
)

assert (
    zobrist_result.completed_depth
    == cached_result.completed_depth
)

assert (
    root_hash
    in zobrist_engine.transposition_table
)

print("âœ… Zobrist hashing validation passed.")
print(
    "Best move:",
    subtraction_move_to_string(
        zobrist_result.best_move
    ),
)
print(
    "Root hash:",
    f"0x{root_hash:016x}",
)
print(
    "Cached positions:",
    zobrist_engine.transposition_table_size(),
)
print(
    "Cache hits:",
    zobrist_result.stats.transposition_hits,
)


# In[48]:


pokemon_zobrist = ZobristHasher(
    state_features=pokemon_state_features,
    seed=20260712,
)

print("âœ… PokÃ©mon Zobrist hasher ready.")


# ## Step 31 â€” Create a PokÃ©mon Integration Test Battle

# # PokÃ©mon TCG Engine Integration
# 
# The reusable PokÃ©mon battle components are now stored in the `src` package.
# 
# This section connects the completed advanced search engine to an actual
# PokÃ©mon battle state before proceeding to move ordering.

# In[50]:


# Simple cards used only to validate the Notebook 11 integration.
#
# These dictionaries follow the same attack structure used by
# get_legal_moves() in src/legal_moves.py.

integration_player_card = {
    "id": "integration-eevee-ex",
    "name": "Eevee ex",
    "hp": 200,
    "attacks": [
        {
            "Move Name": "Tera",
            "damage_numeric": 0,
            "energy_cost": 0,
            "Effect Explanation": (
                "This attack is used as a zero-damage validation move."
            ),
        },
        {
            "Move Name": "Evolution Burst",
            "damage_numeric": 60,
            "energy_cost": 2,
            "Effect Explanation": (
                "A stronger attack used by the integration test."
            ),
        },
    ],
}


integration_opponent_card = {
    "id": "integration-electrike",
    "name": "Electrike",
    "hp": 70,
    "attacks": [
        {
            "Move Name": "Thunder Jolt",
            "damage_numeric": 30,
            "energy_cost": 1,
            "Effect Explanation": (
                "This PokÃ©mon also does 10 damage to itself."
            ),
        }
    ],
}


integration_player_pokemon = PokemonState(
    card=integration_player_card,
    current_hp=200.0,
    attached_energy=2,
    status=None,
    damage=0.0,
    is_active=True,
)


integration_opponent_pokemon = PokemonState(
    card=integration_opponent_card,
    current_hp=70.0,
    attached_energy=1,
    status=None,
    damage=0.0,
    is_active=True,
)


integration_player = PlayerState(
    active=integration_player_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


integration_opponent = PlayerState(
    active=integration_opponent_pokemon,
    bench=[],
    prize_cards_remaining=6,
    hand_size=7,
)


pokemon_test_battle = BattleState(
    player=integration_player,
    opponent=integration_opponent,
    turn_number=1,
    current_player="Player",
)


print("âœ… PokÃ©mon integration test battle created.")
print(
    "Player:",
    pokemon_test_battle.player.active.card["name"],
)
print(
    "Player HP:",
    pokemon_test_battle.player.active.current_hp,
)
print(
    "Player Energy:",
    pokemon_test_battle.player.active.attached_energy,
)
print(
    "Opponent:",
    pokemon_test_battle.opponent.active.card["name"],
)
print(
    "Opponent HP:",
    pokemon_test_battle.opponent.active.current_hp,
)
print(
    "Opponent Energy:",
    pokemon_test_battle.opponent.active.attached_energy,
)
print(
    "Current player:",
    pokemon_test_battle.current_player,
)


# In[51]:


player_integration_moves = generate_moves_adapter(
    pokemon_test_battle
)

assert isinstance(pokemon_test_battle, BattleState)
assert pokemon_test_battle.current_player == "Player"
assert pokemon_test_battle.player.active.current_hp == 200.0
assert pokemon_test_battle.opponent.active.current_hp == 70.0

assert len(player_integration_moves) == 2

assert {
    move["name"]
    for move in player_integration_moves
} == {
    "Tera",
    "Evolution Burst",
}

print("âœ… Step 31 validation passed.")
print(
    "Legal moves:",
    [
        move["name"]
        for move in player_integration_moves
    ],
)


# In[54]:


from copy import deepcopy


# ## Step 32 â€” Create and Validate the PokÃ©mon Zobrist Hasher
# 
# The PokÃ©mon battle state is converted into labeled position features and
# hashed into a deterministic 64-bit key for use by the transposition table.

# In[55]:


pokemon_zobrist = ZobristHasher(
    state_features=pokemon_state_features,
    seed=20260712,
)

print("âœ… PokÃ©mon Zobrist hasher created.")
print(
    "Initial registered features:",
    pokemon_zobrist.feature_count(),
)


# In[56]:


pokemon_root_hash_1 = pokemon_zobrist.hash_state(
    pokemon_test_battle
)

pokemon_root_hash_2 = pokemon_zobrist.hash_state(
    pokemon_test_battle
)

pokemon_test_battle_copy = deepcopy(
    pokemon_test_battle
)

pokemon_copy_hash = pokemon_zobrist.hash_state(
    pokemon_test_battle_copy
)

pokemon_changed_state = deepcopy(
    pokemon_test_battle
)

pokemon_changed_state.opponent.active.current_hp = 40.0

pokemon_changed_hash = pokemon_zobrist.hash_state(
    pokemon_changed_state
)

assert isinstance(pokemon_root_hash_1, int)

assert (
    0
    <= pokemon_root_hash_1
    <= ZobristHasher.MASK_64
)

# Same state, same hash.
assert pokemon_root_hash_1 == pokemon_root_hash_2

# Equivalent copied state, same hash.
assert pokemon_root_hash_1 == pokemon_copy_hash

# Changed HP, different hash.
assert pokemon_root_hash_1 != pokemon_changed_hash

print("âœ… Step 32 validation passed.")
print(
    "Root hash:",
    pokemon_root_hash_1,
)
print(
    "Root hexadecimal hash:",
    f"0x{pokemon_root_hash_1:016x}",
)
print(
    "Changed-state hash:",
    pokemon_changed_hash,
)
print(
    "Registered features:",
    pokemon_zobrist.feature_count(),
)


# ## Step 33 â€” Build the PokÃ©mon Search Engine
# 
# The generic advanced search engine is now connected to the reusable PokÃ©mon
# battle modules, the PokÃ©mon adapters, the transposition table, and Zobrist
# hashing.

# In[57]:


pokemon_search_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
)

print("âœ… PokÃ©mon search engine created.")
print(
    "Transposition table enabled:",
    pokemon_search_engine.use_transposition_table,
)
print(
    "Initial cached positions:",
    pokemon_search_engine.transposition_table_size(),
)


# In[58]:


assert isinstance(
    pokemon_search_engine,
    AdvancedSearchEngine,
)

assert pokemon_search_engine.use_transposition_table is True

assert (
    pokemon_search_engine.transposition_table_size()
    == 0
)

assert callable(
    pokemon_search_engine.generate_moves
)

assert callable(
    pokemon_search_engine.apply_move
)

assert callable(
    pokemon_search_engine.evaluate_state
)

assert callable(
    pokemon_search_engine.is_terminal
)

assert callable(
    pokemon_search_engine.current_player
)

print("âœ… Step 33 validation passed.")


# ## Step 34 â€” First PokÃ©mon Search
# 
# Run the search engine for the first time and verify that it chooses one of the
# currently legal PokÃ©mon moves.

# In[60]:


print("=" * 70)
print("FIRST POKÃ‰MON SEARCH")
print("=" * 70)

search_result = pokemon_search_engine.search_depth(
    state=pokemon_test_battle,
    depth=2,
)

print("Search completed.")
print()

print("Evaluation Score:")
print(search_result.score)

print()

print("Best Move:")
print(
    search_result.best_move
)

print()

print("Best Move Name:")
print(
    search_result.best_move["name"]
    if search_result.best_move is not None
    else None
)

print()

print("Principal Variation:")
print(
    [
        move["name"]
        for move in search_result.principal_variation
    ]
)

print()

print("Completed Depth:")
print(search_result.completed_depth)

print()

print("Nodes Searched:")
print(search_result.stats.nodes)

print()

print("Transposition Hits:")
print(search_result.stats.transposition_hits)

print()

print("Cached Positions:")
print(
    pokemon_search_engine.transposition_table_size()
)


# In[61]:


legal_move_names = {
    move["name"]
    for move in generate_moves_adapter(
        pokemon_test_battle
    )
}

assert search_result.best_move is not None

assert (
    search_result.best_move["name"]
    in legal_move_names
)

assert isinstance(
    search_result.score,
    (int, float),
)

assert isinstance(
    search_result.principal_variation,
    list,
)

assert search_result.completed_depth == 2

assert search_result.stats.nodes > 0

assert (
    search_result.best_move
    == search_result.principal_variation[0]
)

print("âœ… Step 34 validation passed.")

print(
    "Chosen move:",
    search_result.best_move["name"],
)

print(
    "Search score:",
    search_result.score,
)

print(
    "Principal variation:",
    [
        move["name"]
        for move in search_result.principal_variation
    ],
)

print(
    "Nodes:",
    search_result.stats.nodes,
)

print(
    "Transposition hits:",
    search_result.stats.transposition_hits,
)

print(
    "Cached positions:",
    pokemon_search_engine.transposition_table_size(),
)


# ## Step 35 â€” PokÃ©mon Iterative Deepening Validation
# 
# The integrated PokÃ©mon engine now searches the same battle repeatedly from
# depth 1 through a selected maximum depth.
# 
# This confirms that iterative deepening, principal-variation tracking,
# transposition tables, Zobrist hashing, and the PokÃ©mon battle adapters work
# together end to end.

# In[62]:


pokemon_iterative_result = (
    pokemon_search_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=4,
        verbose=True,
        clear_table=True,
    )
)


# In[63]:


pokemon_iterative_result.as_dict(
    move_to_string=lambda move: move["name"]
)


# In[65]:


pokemon_legal_move_names = {
    move["name"]
    for move in generate_moves_adapter(
        pokemon_test_battle
    )
}

assert pokemon_iterative_result.completed_depth == 4

assert pokemon_iterative_result.best_move is not None

assert (
    pokemon_iterative_result.best_move["name"]
    in pokemon_legal_move_names
)

assert pokemon_iterative_result.principal_variation

assert (
    pokemon_iterative_result.best_move
    == pokemon_iterative_result.principal_variation[0]
)

assert pokemon_iterative_result.stats.nodes > 0

assert len(
    pokemon_iterative_result.stats.depth_nodes
) == 4

assert len(
    pokemon_iterative_result.stats.depth_times
) == 4

assert (
    pokemon_search_engine.transposition_table_size()
    > 0
)

print("âœ… Step 35 validation passed.")

print(
    "Best move:",
    pokemon_iterative_result.best_move["name"],
)

print(
    "Final score:",
    pokemon_iterative_result.score,
)

print(
    "Completed depth:",
    pokemon_iterative_result.completed_depth,
)

print(
    "Principal variation:",
    [
        move["name"]
        for move
        in pokemon_iterative_result.principal_variation
    ],
)

print(
    "Total nodes:",
    pokemon_iterative_result.stats.nodes,
)

print(
    "Transposition hits:",
    pokemon_iterative_result.stats.transposition_hits,
)

print(
    "Cached positions:",
    pokemon_search_engine.transposition_table_size(),
)

print(
    "Nodes by depth:",
    pokemon_iterative_result.stats.depth_nodes,
)


# ## Step 36 â€” Add the Move-Ordering Section

# # Part 4 â€” Move Ordering
# 
# Move ordering searches promising actions before weaker actions.
# 
# Good move ordering allows alpha-beta pruning to reach useful bounds earlier,
# reducing the number of positions that must be explored.
# 
# The initial PokÃ©mon move-ordering priorities are:
# 
# 1. Transposition-table best move
# 2. Previous principal-variation move
# 3. Immediate Knock Out
# 4. Higher attack damage
# 5. Lower Energy cost
# 6. Pass actions last

# ## Step 37 â€” Add the Move-Ordering Type Alias

# In[67]:


MoveOrderScoreFn = Callable[
    [State, Move, Player],
    float,
]


# ## Step 38 â€” Create PokÃ©mon Move-Ordering Scores

# In[68]:


def pokemon_move_order_score(
    state: BattleState,
    move: dict,
    root_player: str,
) -> float:
    """
    Return a move-ordering priority for one PokÃ©mon action.

    This score affects only search order. It does not replace the
    position evaluation function.
    """

    move_name = str(
        move.get("name", "Unknown Move")
    )

    try:
        damage = float(
            move.get("damage", 0) or 0
        )
    except (TypeError, ValueError):
        damage = 0.0

    try:
        energy_cost = int(
            move.get("energy_cost", 0) or 0
        )
    except (TypeError, ValueError):
        energy_cost = 0

    if state.current_player == "Player":
        defending_hp = (
            state.opponent.active.current_hp
        )
    elif state.current_player == "Opponent":
        defending_hp = (
            state.player.active.current_hp
        )
    else:
        raise ValueError(
            "current_player must be "
            "'Player' or 'Opponent'."
        )

    score = 0.0

    # Pass should normally be searched last.
    if move_name.lower() == "pass":
        score -= 1_000_000.0

    # Immediate Knock Outs receive very high priority.
    if damage > 0 and damage >= defending_hp:
        score += 1_000_000.0

    # Prefer attacks that deal more damage.
    score += damage * 1_000.0

    # Among otherwise similar attacks, prefer lower Energy cost.
    score -= energy_cost * 10.0

    # Small deterministic tie breaker.
    score += len(move_name) * 0.001

    return score


# ## Step 39 â€” Inspect PokÃ©mon Move Scores

# In[69]:


pokemon_ordering_preview = []

for move in generate_moves_adapter(
    pokemon_test_battle
):
    priority = pokemon_move_order_score(
        state=pokemon_test_battle,
        move=move,
        root_player="Player",
    )

    pokemon_ordering_preview.append(
        {
            "name": move["name"],
            "damage": move["damage"],
            "energy_cost": move["energy_cost"],
            "priority": priority,
        }
    )

pokemon_ordering_preview.sort(
    key=lambda row: row["priority"],
    reverse=True,
)

for row in pokemon_ordering_preview:
    print(
        f"{row['name']:<20} | "
        f"Damage {row['damage']:>6.1f} | "
        f"Energy {row['energy_cost']:>2} | "
        f"Priority {row['priority']:>12.3f}"
    )


# ## Step 40 â€” Validate the PokÃ©mon Ordering Function

# In[70]:


ordering_moves = generate_moves_adapter(
    pokemon_test_battle
)

ordering_scores = {
    move["name"]: pokemon_move_order_score(
        state=pokemon_test_battle,
        move=move,
        root_player="Player",
    )
    for move in ordering_moves
}

assert "Evolution Burst" in ordering_scores
assert "Tera" in ordering_scores

assert (
    ordering_scores["Evolution Burst"]
    > ordering_scores["Tera"]
)

print("âœ… PokÃ©mon move-order scoring validation passed.")
print(
    "Evolution Burst priority:",
    ordering_scores["Evolution Burst"],
)
print(
    "Tera priority:",
    ordering_scores["Tera"],
)


# ## Step 41 â€” Replace AdvancedSearchEngine
# 
# Replace the entire existing AdvancedSearchEngine class with this updated version.

# In[72]:


class AdvancedSearchEngine:
    """
    Advanced adversarial search engine with:

    - Alpha-beta pruning
    - Iterative deepening
    - Transposition tables
    - Move ordering
    - Principal-variation tracking
    """

    def __init__(
        self,
        generate_moves: GenerateMovesFn,
        apply_move: ApplyMoveFn,
        evaluate_state: EvaluateFn,
        is_terminal: IsTerminalFn,
        current_player: CurrentPlayerFn,
        move_to_string: Optional[MoveToStringFn] = None,
        state_key: Optional[Callable[[State], Hashable]] = None,
        use_transposition_table: bool = True,
        move_order_score: Optional[MoveOrderScoreFn] = None,
        use_move_ordering: bool = True,
    ) -> None:
        """
        Initialize the search engine.

        move_order_score
        ----------------
        Optional callback that returns a numeric move-ordering priority.

        Higher scores are searched first.

        use_move_ordering
        -----------------
        Enables or disables all move ordering without removing the
        configured callback.
        """

        self.generate_moves = generate_moves
        self.apply_move = apply_move
        self.evaluate_state = evaluate_state
        self.is_terminal = is_terminal
        self.current_player = current_player
        self.move_to_string = move_to_string or str

        self.state_key = (
            state_key
            or self._default_state_key
        )

        self.use_transposition_table = (
            use_transposition_table
        )

        self.move_order_score = move_order_score
        self.use_move_ordering = use_move_ordering

        self.stats = SearchStats()

        self.transposition_table: Dict[
            Hashable,
            TranspositionEntry,
        ] = {}

        self._principal_variation: List[Move] = []

        # Previous completed iterative-deepening line.
        self._previous_iteration_pv: List[Move] = []

    @staticmethod
    def _default_state_key(
        state: State,
    ) -> Hashable:
        """
        Use the state itself as its cache key.
        """

        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "The game state is not hashable. "
                "Supply a state_key function when creating "
                "AdvancedSearchEngine."
            ) from exc

        return state

    def _validate_depth(
        self,
        depth: int,
    ) -> None:
        """Validate a requested search depth."""

        if not isinstance(depth, int):
            raise TypeError(
                "Search depth must be an integer, "
                f"received {type(depth).__name__}."
            )

        if depth < 1:
            raise ValueError(
                "Search depth must be at least 1, "
                f"received {depth}."
            )

    def _safe_moves(
        self,
        state: State,
    ) -> List[Move]:
        """
        Generate legal moves and convert them to a list.
        """

        moves = self.generate_moves(state)

        if moves is None:
            return []

        return list(moves)

    def clear_transposition_table(
        self,
    ) -> None:
        """Remove every cached position."""

        self.transposition_table.clear()

    def transposition_table_size(
        self,
    ) -> int:
        """Return the number of cached positions."""

        return len(self.transposition_table)

    def _find_transposition_move(
        self,
        key: Hashable,
    ) -> Optional[Move]:
        """
        Return the cached best move for a position, when available.
        """

        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        return entry.best_move

    def _moves_equal(
        self,
        first: Optional[Move],
        second: Optional[Move],
    ) -> bool:
        """
        Safely compare two moves.

        This supports dictionary moves as well as immutable moves.
        """

        if first is None or second is None:
            return False

        try:
            return bool(first == second)
        except Exception:
            return False

    def _order_moves(
        self,
        state: State,
        moves: Sequence[Move],
        root_player: Player,
        ply: int,
        transposition_move: Optional[Move] = None,
    ) -> List[Move]:
        """
        Order legal moves from most promising to least promising.

        Priority:

        1. Transposition-table best move
        2. Previous-iteration principal-variation move
        3. Custom move-order score
        4. Original generation order
        """

        ordered_moves = list(moves)

        if (
            not self.use_move_ordering
            or len(ordered_moves) <= 1
        ):
            return ordered_moves

        pv_move: Optional[Move] = None

        if ply < len(self._previous_iteration_pv):
            pv_move = self._previous_iteration_pv[ply]

        scored_moves = []

        for original_index, move in enumerate(
            ordered_moves
        ):
            priority = 0.0

            # The transposition-table move receives the
            # strongest ordering bonus.
            if self._moves_equal(
                move,
                transposition_move,
            ):
                priority += 10_000_000_000.0

            # The previous principal-variation move receives
            # the next strongest bonus.
            if self._moves_equal(
                move,
                pv_move,
            ):
                priority += 1_000_000_000.0

            if self.move_order_score is not None:
                custom_score = self.move_order_score(
                    state,
                    move,
                    root_player,
                )

                priority += float(custom_score)

            scored_moves.append(
                (
                    priority,
                    -original_index,
                    move,
                )
            )

        scored_moves.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [
            move
            for _, _, move in scored_moves
        ]

    def preview_move_order(
        self,
        state: State,
        root_player: Optional[Player] = None,
        ply: int = 0,
    ) -> List[Move]:
        """
        Return the move order that the engine would search.

        This is useful for debugging and validation.
        """

        if root_player is None:
            root_player = self.current_player(state)

        legal_moves = self._safe_moves(state)

        key = self.state_key(state)

        transposition_move = (
            self._find_transposition_move(key)
        )

        return self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

    def _probe_transposition_table(
        self,
        key: Hashable,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[float, List[Move]]]:
        """
        Attempt to reuse a cached search result.
        """

        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        if entry.depth < depth:
            return None

        self.stats.transposition_hits += 1

        if entry.bound_type is BoundType.EXACT:
            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.LOWER_BOUND
            and entry.score >= beta
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.UPPER_BOUND
            and entry.score <= alpha
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        return None

    def _store_transposition_entry(
        self,
        key: Hashable,
        depth: int,
        score: float,
        alpha_original: float,
        beta_original: float,
        best_line: List[Move],
    ) -> None:
        """
        Store a completed node search in the cache.
        """

        if not self.use_transposition_table:
            return

        if score <= alpha_original:
            bound_type = BoundType.UPPER_BOUND

        elif score >= beta_original:
            bound_type = BoundType.LOWER_BOUND

        else:
            bound_type = BoundType.EXACT

        existing = self.transposition_table.get(key)

        if (
            existing is not None
            and existing.depth > depth
        ):
            return

        best_move = (
            best_line[0]
            if best_line
            else None
        )

        self.transposition_table[key] = (
            TranspositionEntry(
                depth=depth,
                score=score,
                bound_type=bound_type,
                best_move=best_move,
                principal_variation=list(
                    best_line
                ),
            )
        )

        self.stats.transposition_stores += 1

    def _alpha_beta(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
        ply: int = 0,
    ) -> Tuple[float, List[Move]]:
        """
        Search one position with alpha-beta pruning,
        transposition tables, and move ordering.
        """

        self.stats.nodes += 1

        if self.is_terminal(state):
            self.stats.terminal_nodes += 1

            score = float(
                self.evaluate_state(
                    state,
                    root_player,
                )
            )

            return score, []

        if depth <= 0:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(
                    state,
                    root_player,
                )
            )

            return score, []

        key = self.state_key(state)

        cached_result = (
            self._probe_transposition_table(
                key=key,
                depth=depth,
                alpha=alpha,
                beta=beta,
            )
        )

        if cached_result is not None:
            return cached_result

        legal_moves = self._safe_moves(state)

        if not legal_moves:
            self.stats.leaf_nodes += 1

            score = float(
                self.evaluate_state(
                    state,
                    root_player,
                )
            )

            return score, []

        transposition_move = (
            self._find_transposition_move(key)
        )

        legal_moves = self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

        alpha_original = alpha
        beta_original = beta

        player_to_move = self.current_player(
            state
        )

        maximizing = (
            player_to_move == root_player
        )

        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score > best_score:
                    best_score = child_score
                    best_line = (
                        [move] + child_line
                    )

                alpha = max(
                    alpha,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break

        else:
            best_score = math.inf
            best_line = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score < best_score:
                    best_score = child_score
                    best_line = (
                        [move] + child_line
                    )

                beta = min(
                    beta,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    break

        self._store_transposition_entry(
            key=key,
            depth=depth,
            score=best_score,
            alpha_original=alpha_original,
            beta_original=beta_original,
            best_line=best_line,
        )

        return best_score, best_line

    def search_depth(
        self,
        state: State,
        depth: int,
        root_player: Optional[Player] = None,
        clear_table: bool = True,
    ) -> SearchResult:
        """
        Perform one fixed-depth search.
        """

        self._validate_depth(depth)

        if root_player is None:
            root_player = self.current_player(
                state
            )

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        self._previous_iteration_pv = []

        start_time = time.perf_counter()

        score, principal_variation = (
            self._alpha_beta(
                state=state,
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
                ply=0,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed
        self.stats.depth_nodes[depth] = (
            self.stats.nodes
        )
        self.stats.depth_times[depth] = elapsed

        best_move = (
            principal_variation[0]
            if principal_variation
            else None
        )

        self._principal_variation = (
            principal_variation
        )

        return SearchResult(
            best_move=best_move,
            score=score,
            completed_depth=depth,
            principal_variation=(
                principal_variation
            ),
            stats=self.stats,
        )

    def iterative_deepening(
        self,
        state: State,
        max_depth: int,
        root_player: Optional[Player] = None,
        verbose: bool = True,
        clear_table: bool = True,
    ) -> SearchResult:
        """
        Search successively from depth 1 through max_depth.
        """

        self._validate_depth(max_depth)

        if root_player is None:
            root_player = self.current_player(
                state
            )

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        self._previous_iteration_pv = []

        overall_start = time.perf_counter()

        best_move: Optional[Move] = None

        best_score = float(
            self.evaluate_state(
                state,
                root_player,
            )
        )

        best_line: List[Move] = []

        total_nodes = 0
        total_leaf_nodes = 0
        total_terminal_nodes = 0
        total_cutoffs = 0

        total_tt_hits = 0
        total_tt_stores = 0
        total_tt_cutoffs = 0

        depth_nodes: Dict[int, int] = {}
        depth_times: Dict[int, float] = {}

        completed_depth = 0

        for depth in range(
            1,
            max_depth + 1,
        ):
            self.stats.nodes = 0
            self.stats.leaf_nodes = 0
            self.stats.terminal_nodes = 0
            self.stats.cutoffs = 0

            self.stats.transposition_hits = 0
            self.stats.transposition_stores = 0
            self.stats.transposition_cutoffs = 0

            depth_start = time.perf_counter()

            score, principal_variation = (
                self._alpha_beta(
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    root_player=root_player,
                    ply=0,
                )
            )

            depth_elapsed = (
                time.perf_counter()
                - depth_start
            )

            iteration_nodes = (
                self.stats.nodes
            )

            depth_nodes[depth] = (
                iteration_nodes
            )
            depth_times[depth] = (
                depth_elapsed
            )

            total_nodes += self.stats.nodes
            total_leaf_nodes += (
                self.stats.leaf_nodes
            )
            total_terminal_nodes += (
                self.stats.terminal_nodes
            )
            total_cutoffs += (
                self.stats.cutoffs
            )

            total_tt_hits += (
                self.stats.transposition_hits
            )
            total_tt_stores += (
                self.stats.transposition_stores
            )
            total_tt_cutoffs += (
                self.stats.transposition_cutoffs
            )

            completed_depth = depth
            best_score = score
            best_line = principal_variation

            if principal_variation:
                best_move = (
                    principal_variation[0]
                )

            # The completed line becomes the ordering guide
            # for the next iteration.
            self._previous_iteration_pv = list(
                principal_variation
            )

            if verbose:
                move_name = (
                    self.move_to_string(
                        best_move
                    )
                    if best_move is not None
                    else "None"
                )

                pv_text = " -> ".join(
                    self.move_to_string(move)
                    for move
                    in principal_variation
                )

                print(
                    f"Depth {depth:>2} | "
                    f"Score {score:>10.2f} | "
                    f"Nodes {iteration_nodes:>8,} | "
                    f"TT hits "
                    f"{self.stats.transposition_hits:>6,} | "
                    f"Time {depth_elapsed:>8.4f}s | "
                    f"Best move: {move_name}"
                )

                if pv_text:
                    print(
                        f"         PV: {pv_text}"
                    )

        overall_elapsed = (
            time.perf_counter()
            - overall_start
        )

        final_stats = SearchStats(
            nodes=total_nodes,
            leaf_nodes=total_leaf_nodes,
            terminal_nodes=total_terminal_nodes,
            cutoffs=total_cutoffs,
            transposition_hits=total_tt_hits,
            transposition_stores=total_tt_stores,
            transposition_cutoffs=(
                total_tt_cutoffs
            ),
            completed_depth=completed_depth,
            elapsed_seconds=overall_elapsed,
            depth_nodes=depth_nodes,
            depth_times=depth_times,
        )

        self.stats = final_stats
        self._principal_variation = (
            best_line
        )

        return SearchResult(
            best_move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=best_line,
            stats=final_stats,
        )

    def get_principal_variation(
        self,
    ) -> List[Move]:
        """
        Return a copy of the best line from the latest search.
        """

        return list(
            self._principal_variation
        )


# ## Step 42 â€” Recreate the PokÃ©mon Search Engine
# 
# Because the class definition changed, recreate the engine:

# In[73]:


pokemon_search_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
    move_order_score=pokemon_move_order_score,
    use_move_ordering=True,
)

print("âœ… Move-ordering PokÃ©mon engine created.")


# ## Step 43 â€” Preview the Actual Search Order

# In[76]:


ordered_root_moves = (
    pokemon_search_engine.preview_move_order(
        state=pokemon_test_battle,
        root_player="Player",
    )
)

print("PokÃ©mon root move order:")

for index, move in enumerate(
    ordered_root_moves,
    start=1,
):
    print(
        f"{index}. {move['name']} "
        f"(damage={move['damage']}, "
        f"energy={move['energy_cost']})"
    )


# ## Step 44 â€” Validate Move Ordering

# In[78]:


assert len(ordered_root_moves) == 2

assert (
    ordered_root_moves[0]["name"]
    == "Evolution Burst"
)

assert (
    ordered_root_moves[1]["name"]
    == "Tera"
)

print("âœ… Step 44 move-order validation passed.")


# ## Step 45 â€” Benchmark Ordered vs Unordered Search
# 
# Two otherwise identical engines are compared:
# 
# - one with move ordering disabled
# - one with move ordering enabled
# 
# Both engines must return the same best move and score. The ordered engine should
# search the same number of nodes or fewer.

# In[79]:


unordered_pokemon_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
    move_order_score=pokemon_move_order_score,
    use_move_ordering=False,
)

ordered_pokemon_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
    move_order_score=pokemon_move_order_score,
    use_move_ordering=True,
)

print("âœ… Benchmark engines created.")


# ## Step 46 â€” Run the Comparison

# In[80]:


unordered_result = (
    unordered_pokemon_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=6,
        verbose=False,
        clear_table=True,
    )
)

ordered_result = (
    ordered_pokemon_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=6,
        verbose=False,
        clear_table=True,
    )
)

print("=" * 70)
print("MOVE ORDERING BENCHMARK")
print("=" * 70)

print("Without move ordering")
print(
    "  Best move:",
    unordered_result.best_move["name"],
)
print(
    "  Score:",
    unordered_result.score,
)
print(
    "  Nodes:",
    unordered_result.stats.nodes,
)
print(
    "  Cutoffs:",
    unordered_result.stats.cutoffs,
)
print(
    "  TT hits:",
    unordered_result.stats.transposition_hits,
)
print(
    "  Time:",
    f"{unordered_result.stats.elapsed_seconds:.6f}s",
)

print()

print("With move ordering")
print(
    "  Best move:",
    ordered_result.best_move["name"],
)
print(
    "  Score:",
    ordered_result.score,
)
print(
    "  Nodes:",
    ordered_result.stats.nodes,
)
print(
    "  Cutoffs:",
    ordered_result.stats.cutoffs,
)
print(
    "  TT hits:",
    ordered_result.stats.transposition_hits,
)
print(
    "  Time:",
    f"{ordered_result.stats.elapsed_seconds:.6f}s",
)


# ## Step 47 â€” Calculate the Improvement

# In[81]:


unordered_nodes = unordered_result.stats.nodes
ordered_nodes = ordered_result.stats.nodes

nodes_saved = unordered_nodes - ordered_nodes

if unordered_nodes > 0:
    node_reduction_percent = (
        nodes_saved / unordered_nodes
    ) * 100.0
else:
    node_reduction_percent = 0.0

if ordered_nodes > 0:
    node_speedup_ratio = (
        unordered_nodes / ordered_nodes
    )
else:
    node_speedup_ratio = math.inf

print(
    "Nodes saved:",
    nodes_saved,
)

print(
    "Node reduction:",
    f"{node_reduction_percent:.2f}%",
)

print(
    "Node-search ratio:",
    f"{node_speedup_ratio:.2f}x",
)


# ## Step 48 â€” Validate the Benchmark

# In[82]:


assert (
    unordered_result.best_move["name"]
    == ordered_result.best_move["name"]
)

assert (
    unordered_result.score
    == ordered_result.score
)

assert (
    unordered_result.completed_depth
    == ordered_result.completed_depth
    == 6
)

assert ordered_result.best_move is not None

assert ordered_result.stats.nodes > 0
assert unordered_result.stats.nodes > 0

assert (
    ordered_result.best_move
    == ordered_result.principal_variation[0]
)

print("âœ… Step 48 benchmark validation passed.")

print(
    "Best move:",
    ordered_result.best_move["name"],
)

print(
    "Depth:",
    ordered_result.completed_depth,
)

print(
    "Unordered nodes:",
    unordered_result.stats.nodes,
)

print(
    "Ordered nodes:",
    ordered_result.stats.nodes,
)


# ## Step 49 â€” Add the Killer Move Section

# # Part 5 â€” Killer Move Heuristic
# 
# The killer move heuristic remembers moves that caused alpha-beta cutoffs.
# 
# A move that caused a cutoff in one branch may also cause a cutoff in another
# branch at the same search depth.
# 
# The engine stores up to two killer moves per ply and gives them a strong
# move-ordering bonus.

# ## Step 50 â€” Extend Search Statistics
# 
# Replace the current SearchStats class with this updated version:

# In[84]:


@dataclass
class SearchStats:
    """
    Statistics collected during one complete search.
    """

    nodes: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    cutoffs: int = 0

    transposition_hits: int = 0
    transposition_stores: int = 0
    transposition_cutoffs: int = 0

    killer_hits: int = 0
    killer_stores: int = 0

    completed_depth: int = 0
    elapsed_seconds: float = 0.0

    depth_nodes: Dict[int, int] = field(default_factory=dict)
    depth_times: Dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all statistics."""

        self.nodes = 0
        self.leaf_nodes = 0
        self.terminal_nodes = 0
        self.cutoffs = 0

        self.transposition_hits = 0
        self.transposition_stores = 0
        self.transposition_cutoffs = 0

        self.killer_hits = 0
        self.killer_stores = 0

        self.completed_depth = 0
        self.elapsed_seconds = 0.0

        self.depth_nodes.clear()
        self.depth_times.clear()

    @property
    def nodes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return self.nodes / self.elapsed_seconds

    @property
    def transposition_hit_rate(self) -> float:
        if self.nodes <= 0:
            return 0.0

        return self.transposition_hits / self.nodes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "leaf_nodes": self.leaf_nodes,
            "terminal_nodes": self.terminal_nodes,
            "cutoffs": self.cutoffs,
            "transposition_hits": self.transposition_hits,
            "transposition_stores": self.transposition_stores,
            "transposition_cutoffs": self.transposition_cutoffs,
            "transposition_hit_rate": round(
                self.transposition_hit_rate,
                6,
            ),
            "killer_hits": self.killer_hits,
            "killer_stores": self.killer_stores,
            "completed_depth": self.completed_depth,
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "nodes_per_second": round(
                self.nodes_per_second,
                2,
            ),
            "depth_nodes": dict(self.depth_nodes),
            "depth_times": {
                depth: round(seconds, 6)
                for depth, seconds in self.depth_times.items()
            },
        }


# ## Step 51 â€” Add a Move-Key Function

# In[85]:


MoveKeyFn = Callable[[Move], Hashable]


# In[86]:


def pokemon_move_key(
    move: dict,
) -> Tuple[Hashable, ...]:
    """
    Return a stable, hashable identifier for one PokÃ©mon move.
    """

    return (
        str(move.get("name", "Unknown Move")),
        float(move.get("damage", 0) or 0),
        int(move.get("energy_cost", 0) or 0),
        str(move.get("effect", "") or ""),
    )


# ## Step 52 â€” Replace AdvancedSearchEngine
# 
# Replace the entire current AdvancedSearchEngine class with the updated killer-move version below:

# In[87]:


class AdvancedSearchEngine:
    """
    Advanced adversarial search engine with:

    - Alpha-beta pruning
    - Iterative deepening
    - Transposition tables
    - Move ordering
    - Killer move heuristic
    - Principal-variation tracking
    """

    def __init__(
        self,
        generate_moves: GenerateMovesFn,
        apply_move: ApplyMoveFn,
        evaluate_state: EvaluateFn,
        is_terminal: IsTerminalFn,
        current_player: CurrentPlayerFn,
        move_to_string: Optional[MoveToStringFn] = None,
        state_key: Optional[Callable[[State], Hashable]] = None,
        use_transposition_table: bool = True,
        move_order_score: Optional[MoveOrderScoreFn] = None,
        use_move_ordering: bool = True,
        move_key: Optional[MoveKeyFn] = None,
        use_killer_moves: bool = True,
        killer_slots: int = 2,
    ) -> None:
        self.generate_moves = generate_moves
        self.apply_move = apply_move
        self.evaluate_state = evaluate_state
        self.is_terminal = is_terminal
        self.current_player = current_player
        self.move_to_string = move_to_string or str

        self.state_key = state_key or self._default_state_key

        self.use_transposition_table = use_transposition_table
        self.move_order_score = move_order_score
        self.use_move_ordering = use_move_ordering

        self.move_key = move_key or self._default_move_key
        self.use_killer_moves = use_killer_moves

        if killer_slots < 1:
            raise ValueError(
                "killer_slots must be at least 1."
            )

        self.killer_slots = killer_slots

        self.stats = SearchStats()

        self.transposition_table: Dict[
            Hashable,
            TranspositionEntry,
        ] = {}

        self.killer_moves: Dict[
            int,
            List[Move],
        ] = {}

        self._principal_variation: List[Move] = []
        self._previous_iteration_pv: List[Move] = []

    @staticmethod
    def _default_state_key(
        state: State,
    ) -> Hashable:
        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "The game state is not hashable. "
                "Supply a state_key function."
            ) from exc

        return state

    @staticmethod
    def _default_move_key(
        move: Move,
    ) -> Hashable:
        try:
            hash(move)
        except TypeError as exc:
            raise TypeError(
                "The move is not hashable. "
                "Supply a move_key function."
            ) from exc

        return move

    def _validate_depth(
        self,
        depth: int,
    ) -> None:
        if not isinstance(depth, int):
            raise TypeError(
                "Search depth must be an integer."
            )

        if depth < 1:
            raise ValueError(
                "Search depth must be at least 1."
            )

    def _safe_moves(
        self,
        state: State,
    ) -> List[Move]:
        moves = self.generate_moves(state)

        if moves is None:
            return []

        return list(moves)

    def clear_transposition_table(
        self,
    ) -> None:
        self.transposition_table.clear()

    def clear_killer_moves(
        self,
    ) -> None:
        self.killer_moves.clear()

    def transposition_table_size(
        self,
    ) -> int:
        return len(self.transposition_table)

    def killer_move_count(
        self,
    ) -> int:
        return sum(
            len(moves)
            for moves in self.killer_moves.values()
        )

    def _find_transposition_move(
        self,
        key: Hashable,
    ) -> Optional[Move]:
        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        return entry.best_move

    def _moves_equal(
        self,
        first: Optional[Move],
        second: Optional[Move],
    ) -> bool:
        if first is None or second is None:
            return False

        try:
            return (
                self.move_key(first)
                == self.move_key(second)
            )
        except Exception:
            try:
                return bool(first == second)
            except Exception:
                return False

    def _is_killer_move(
        self,
        move: Move,
        ply: int,
    ) -> bool:
        if not self.use_killer_moves:
            return False

        killers = self.killer_moves.get(
            ply,
            [],
        )

        return any(
            self._moves_equal(move, killer)
            for killer in killers
        )

    def _store_killer_move(
        self,
        move: Move,
        ply: int,
    ) -> None:
        if not self.use_killer_moves:
            return

        killers = self.killer_moves.setdefault(
            ply,
            [],
        )

        for existing in killers:
            if self._moves_equal(
                move,
                existing,
            ):
                return

        killers.insert(
            0,
            move,
        )

        del killers[
            self.killer_slots:
        ]

        self.stats.killer_stores += 1

    def _order_moves(
        self,
        state: State,
        moves: Sequence[Move],
        root_player: Player,
        ply: int,
        transposition_move: Optional[Move] = None,
    ) -> List[Move]:
        ordered_moves = list(moves)

        if (
            not self.use_move_ordering
            or len(ordered_moves) <= 1
        ):
            return ordered_moves

        pv_move: Optional[Move] = None

        if ply < len(self._previous_iteration_pv):
            pv_move = self._previous_iteration_pv[ply]

        scored_moves = []

        for original_index, move in enumerate(
            ordered_moves
        ):
            priority = 0.0

            if self._moves_equal(
                move,
                transposition_move,
            ):
                priority += 10_000_000_000.0

            if self._moves_equal(
                move,
                pv_move,
            ):
                priority += 1_000_000_000.0

            if self._is_killer_move(
                move,
                ply,
            ):
                priority += 100_000_000.0
                self.stats.killer_hits += 1

            if self.move_order_score is not None:
                priority += float(
                    self.move_order_score(
                        state,
                        move,
                        root_player,
                    )
                )

            scored_moves.append(
                (
                    priority,
                    -original_index,
                    move,
                )
            )

        scored_moves.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [
            move
            for _, _, move in scored_moves
        ]

    def preview_move_order(
        self,
        state: State,
        root_player: Optional[Player] = None,
        ply: int = 0,
    ) -> List[Move]:
        if root_player is None:
            root_player = self.current_player(state)

        legal_moves = self._safe_moves(state)
        key = self.state_key(state)

        transposition_move = (
            self._find_transposition_move(key)
        )

        return self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

    def _probe_transposition_table(
        self,
        key: Hashable,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[float, List[Move]]]:
        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        if entry.depth < depth:
            return None

        self.stats.transposition_hits += 1

        if entry.bound_type is BoundType.EXACT:
            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.LOWER_BOUND
            and entry.score >= beta
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.UPPER_BOUND
            and entry.score <= alpha
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        return None

    def _store_transposition_entry(
        self,
        key: Hashable,
        depth: int,
        score: float,
        alpha_original: float,
        beta_original: float,
        best_line: List[Move],
    ) -> None:
        if not self.use_transposition_table:
            return

        if score <= alpha_original:
            bound_type = BoundType.UPPER_BOUND
        elif score >= beta_original:
            bound_type = BoundType.LOWER_BOUND
        else:
            bound_type = BoundType.EXACT

        existing = self.transposition_table.get(key)

        if (
            existing is not None
            and existing.depth > depth
        ):
            return

        best_move = (
            best_line[0]
            if best_line
            else None
        )

        self.transposition_table[key] = (
            TranspositionEntry(
                depth=depth,
                score=score,
                bound_type=bound_type,
                best_move=best_move,
                principal_variation=list(best_line),
            )
        )

        self.stats.transposition_stores += 1

    def _alpha_beta(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
        ply: int = 0,
    ) -> Tuple[float, List[Move]]:
        self.stats.nodes += 1

        if self.is_terminal(state):
            self.stats.terminal_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        if depth <= 0:
            self.stats.leaf_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        key = self.state_key(state)

        cached_result = (
            self._probe_transposition_table(
                key=key,
                depth=depth,
                alpha=alpha,
                beta=beta,
            )
        )

        if cached_result is not None:
            return cached_result

        legal_moves = self._safe_moves(state)

        if not legal_moves:
            self.stats.leaf_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        transposition_move = (
            self._find_transposition_move(key)
        )

        legal_moves = self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

        alpha_original = alpha
        beta_original = beta

        maximizing = (
            self.current_player(state)
            == root_player
        )

        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score > best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                alpha = max(
                    alpha,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    self._store_killer_move(
                        move,
                        ply,
                    )
                    break

        else:
            best_score = math.inf
            best_line = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score < best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                beta = min(
                    beta,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1
                    self._store_killer_move(
                        move,
                        ply,
                    )
                    break

        self._store_transposition_entry(
            key=key,
            depth=depth,
            score=best_score,
            alpha_original=alpha_original,
            beta_original=beta_original,
            best_line=best_line,
        )

        return best_score, best_line

    def search_depth(
        self,
        state: State,
        depth: int,
        root_player: Optional[Player] = None,
        clear_table: bool = True,
        clear_killers: bool = True,
    ) -> SearchResult:
        self._validate_depth(depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        if clear_killers:
            self.clear_killer_moves()

        self._previous_iteration_pv = []

        start_time = time.perf_counter()

        score, principal_variation = (
            self._alpha_beta(
                state=state,
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
                ply=0,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed
        self.stats.depth_nodes[depth] = self.stats.nodes
        self.stats.depth_times[depth] = elapsed

        best_move = (
            principal_variation[0]
            if principal_variation
            else None
        )

        self._principal_variation = principal_variation

        return SearchResult(
            best_move=best_move,
            score=score,
            completed_depth=depth,
            principal_variation=principal_variation,
            stats=self.stats,
        )

    def iterative_deepening(
        self,
        state: State,
        max_depth: int,
        root_player: Optional[Player] = None,
        verbose: bool = True,
        clear_table: bool = True,
        clear_killers: bool = True,
    ) -> SearchResult:
        self._validate_depth(max_depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        if clear_killers:
            self.clear_killer_moves()

        self._previous_iteration_pv = []

        overall_start = time.perf_counter()

        best_move: Optional[Move] = None
        best_score = float(
            self.evaluate_state(
                state,
                root_player,
            )
        )
        best_line: List[Move] = []

        total_nodes = 0
        total_leaf_nodes = 0
        total_terminal_nodes = 0
        total_cutoffs = 0

        total_tt_hits = 0
        total_tt_stores = 0
        total_tt_cutoffs = 0

        total_killer_hits = 0
        total_killer_stores = 0

        depth_nodes: Dict[int, int] = {}
        depth_times: Dict[int, float] = {}

        completed_depth = 0

        for depth in range(1, max_depth + 1):
            self.stats.nodes = 0
            self.stats.leaf_nodes = 0
            self.stats.terminal_nodes = 0
            self.stats.cutoffs = 0

            self.stats.transposition_hits = 0
            self.stats.transposition_stores = 0
            self.stats.transposition_cutoffs = 0

            self.stats.killer_hits = 0
            self.stats.killer_stores = 0

            depth_start = time.perf_counter()

            score, principal_variation = (
                self._alpha_beta(
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    root_player=root_player,
                    ply=0,
                )
            )

            depth_elapsed = (
                time.perf_counter()
                - depth_start
            )

            depth_nodes[depth] = self.stats.nodes
            depth_times[depth] = depth_elapsed

            total_nodes += self.stats.nodes
            total_leaf_nodes += self.stats.leaf_nodes
            total_terminal_nodes += self.stats.terminal_nodes
            total_cutoffs += self.stats.cutoffs

            total_tt_hits += self.stats.transposition_hits
            total_tt_stores += self.stats.transposition_stores
            total_tt_cutoffs += (
                self.stats.transposition_cutoffs
            )

            total_killer_hits += self.stats.killer_hits
            total_killer_stores += self.stats.killer_stores

            completed_depth = depth
            best_score = score
            best_line = principal_variation

            if principal_variation:
                best_move = principal_variation[0]

            self._previous_iteration_pv = list(
                principal_variation
            )

            if verbose:
                move_name = (
                    self.move_to_string(best_move)
                    if best_move is not None
                    else "None"
                )

                pv_text = " -> ".join(
                    self.move_to_string(move)
                    for move in principal_variation
                )

                print(
                    f"Depth {depth:>2} | "
                    f"Score {score:>10.2f} | "
                    f"Nodes {self.stats.nodes:>8,} | "
                    f"TT hits "
                    f"{self.stats.transposition_hits:>6,} | "
                    f"Killer hits "
                    f"{self.stats.killer_hits:>5,} | "
                    f"Time {depth_elapsed:>8.4f}s | "
                    f"Best move: {move_name}"
                )

                if pv_text:
                    print(
                        f"         PV: {pv_text}"
                    )

        overall_elapsed = (
            time.perf_counter()
            - overall_start
        )

        final_stats = SearchStats(
            nodes=total_nodes,
            leaf_nodes=total_leaf_nodes,
            terminal_nodes=total_terminal_nodes,
            cutoffs=total_cutoffs,
            transposition_hits=total_tt_hits,
            transposition_stores=total_tt_stores,
            transposition_cutoffs=total_tt_cutoffs,
            killer_hits=total_killer_hits,
            killer_stores=total_killer_stores,
            completed_depth=completed_depth,
            elapsed_seconds=overall_elapsed,
            depth_nodes=depth_nodes,
            depth_times=depth_times,
        )

        self.stats = final_stats
        self._principal_variation = best_line

        return SearchResult(
            best_move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=best_line,
            stats=final_stats,
        )

    def get_principal_variation(
        self,
    ) -> List[Move]:
        return list(
            self._principal_variation
        )


# ## Step 53 â€” Recreate the PokÃ©mon Engine

# In[88]:


pokemon_search_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
    move_order_score=pokemon_move_order_score,
    use_move_ordering=True,
    move_key=pokemon_move_key,
    use_killer_moves=True,
    killer_slots=2,
)

print("âœ… Killer-move PokÃ©mon engine created.")


# ## Step 54 â€” Run the Killer-Move Search

# In[89]:


killer_result = (
    pokemon_search_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=8,
        verbose=True,
        clear_table=True,
        clear_killers=True,
    )
)


# In[90]:


print(
    "Best move:",
    killer_result.best_move["name"],
)

print(
    "Score:",
    killer_result.score,
)

print(
    "Nodes:",
    killer_result.stats.nodes,
)

print(
    "Cutoffs:",
    killer_result.stats.cutoffs,
)

print(
    "Killer hits:",
    killer_result.stats.killer_hits,
)

print(
    "Killer stores:",
    killer_result.stats.killer_stores,
)

print(
    "Stored killer moves:",
    pokemon_search_engine.killer_move_count(),
)

print()

for ply, moves in sorted(
    pokemon_search_engine.killer_moves.items()
):
    print(
        f"Ply {ply}:",
        [
            move["name"]
            for move in moves
        ],
    )


# ## Step 55 â€” Validate the Killer Heuristic

# In[91]:


assert killer_result.best_move is not None

assert (
    killer_result.best_move["name"]
    == "Evolution Burst"
)

assert killer_result.completed_depth == 8

assert killer_result.stats.nodes > 0

assert killer_result.stats.cutoffs >= 0

assert killer_result.stats.killer_stores >= 0

assert (
    killer_result.best_move
    == killer_result.principal_variation[0]
)

print("âœ… Step 55 killer-move validation passed.")
print(
    "Killer moves stored:",
    pokemon_search_engine.killer_move_count(),
)


# ## Part 6 â€” History Heuristic
# ## Step 56 â€” Add the History Heuristic Section

# # Part 6 â€” History Heuristic
# 
# The history heuristic records moves that repeatedly cause alpha-beta cutoffs.
# 
# Each successful cutoff increases the move's history score by:
# 
# depthÂ²
# 
# Moves with stronger history scores are searched earlier in future branches.
# 
# The history heuristic complements:
# 
# - Transposition-table move ordering
# - Principal-variation ordering
# - Killer moves
# - PokÃ©mon-specific tactical move scoring

# ## Step 57 â€” Replace SearchStats

# In[92]:


@dataclass
class SearchStats:
    """
    Statistics collected during one complete search.
    """

    nodes: int = 0
    leaf_nodes: int = 0
    terminal_nodes: int = 0
    cutoffs: int = 0

    transposition_hits: int = 0
    transposition_stores: int = 0
    transposition_cutoffs: int = 0

    killer_hits: int = 0
    killer_stores: int = 0

    history_hits: int = 0
    history_updates: int = 0

    completed_depth: int = 0
    elapsed_seconds: float = 0.0

    depth_nodes: Dict[int, int] = field(default_factory=dict)
    depth_times: Dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset every search statistic."""

        self.nodes = 0
        self.leaf_nodes = 0
        self.terminal_nodes = 0
        self.cutoffs = 0

        self.transposition_hits = 0
        self.transposition_stores = 0
        self.transposition_cutoffs = 0

        self.killer_hits = 0
        self.killer_stores = 0

        self.history_hits = 0
        self.history_updates = 0

        self.completed_depth = 0
        self.elapsed_seconds = 0.0

        self.depth_nodes.clear()
        self.depth_times.clear()

    @property
    def nodes_per_second(self) -> float:
        """Return average nodes searched per second."""

        if self.elapsed_seconds <= 0:
            return 0.0

        return self.nodes / self.elapsed_seconds

    @property
    def transposition_hit_rate(self) -> float:
        """Return transposition hits divided by searched nodes."""

        if self.nodes <= 0:
            return 0.0

        return self.transposition_hits / self.nodes

    def as_dict(self) -> Dict[str, Any]:
        """Return statistics as a display-friendly dictionary."""

        return {
            "nodes": self.nodes,
            "leaf_nodes": self.leaf_nodes,
            "terminal_nodes": self.terminal_nodes,
            "cutoffs": self.cutoffs,

            "transposition_hits": self.transposition_hits,
            "transposition_stores": self.transposition_stores,
            "transposition_cutoffs": self.transposition_cutoffs,
            "transposition_hit_rate": round(
                self.transposition_hit_rate,
                6,
            ),

            "killer_hits": self.killer_hits,
            "killer_stores": self.killer_stores,

            "history_hits": self.history_hits,
            "history_updates": self.history_updates,

            "completed_depth": self.completed_depth,
            "elapsed_seconds": round(
                self.elapsed_seconds,
                6,
            ),
            "nodes_per_second": round(
                self.nodes_per_second,
                2,
            ),

            "depth_nodes": dict(self.depth_nodes),
            "depth_times": {
                depth: round(seconds, 6)
                for depth, seconds in self.depth_times.items()
            },
        }


# ## Step 58 â€” Replace AdvancedSearchEngine
# 
# Replace the entire current AdvancedSearchEngine class with complete history-enabled version:

# In[93]:


class AdvancedSearchEngine:
    """
    Advanced adversarial search engine with:

    - Alpha-beta pruning
    - Iterative deepening
    - Transposition tables
    - Tactical move ordering
    - Killer move heuristic
    - History heuristic
    - Principal-variation tracking
    """

    def __init__(
        self,
        generate_moves: GenerateMovesFn,
        apply_move: ApplyMoveFn,
        evaluate_state: EvaluateFn,
        is_terminal: IsTerminalFn,
        current_player: CurrentPlayerFn,
        move_to_string: Optional[MoveToStringFn] = None,
        state_key: Optional[Callable[[State], Hashable]] = None,
        use_transposition_table: bool = True,
        move_order_score: Optional[MoveOrderScoreFn] = None,
        use_move_ordering: bool = True,
        move_key: Optional[MoveKeyFn] = None,
        use_killer_moves: bool = True,
        killer_slots: int = 2,
        use_history_heuristic: bool = True,
        history_multiplier: float = 1_000.0,
    ) -> None:
        self.generate_moves = generate_moves
        self.apply_move = apply_move
        self.evaluate_state = evaluate_state
        self.is_terminal = is_terminal
        self.current_player = current_player
        self.move_to_string = move_to_string or str

        self.state_key = (
            state_key
            or self._default_state_key
        )

        self.use_transposition_table = (
            use_transposition_table
        )

        self.move_order_score = move_order_score
        self.use_move_ordering = use_move_ordering

        self.move_key = (
            move_key
            or self._default_move_key
        )

        self.use_killer_moves = use_killer_moves

        if killer_slots < 1:
            raise ValueError(
                "killer_slots must be at least 1."
            )

        self.killer_slots = killer_slots

        self.use_history_heuristic = (
            use_history_heuristic
        )

        if history_multiplier < 0:
            raise ValueError(
                "history_multiplier cannot be negative."
            )

        self.history_multiplier = float(
            history_multiplier
        )

        self.stats = SearchStats()

        self.transposition_table: Dict[
            Hashable,
            TranspositionEntry,
        ] = {}

        self.killer_moves: Dict[
            int,
            List[Move],
        ] = {}

        self.history_table: Dict[
            Tuple[Player, Hashable],
            int,
        ] = {}

        self._principal_variation: List[Move] = []
        self._previous_iteration_pv: List[Move] = []

    @staticmethod
    def _default_state_key(
        state: State,
    ) -> Hashable:
        """Use a hashable game state directly as its key."""

        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "The game state is not hashable. "
                "Supply a state_key function."
            ) from exc

        return state

    @staticmethod
    def _default_move_key(
        move: Move,
    ) -> Hashable:
        """Use a hashable move directly as its move key."""

        try:
            hash(move)
        except TypeError as exc:
            raise TypeError(
                "The move is not hashable. "
                "Supply a move_key function."
            ) from exc

        return move

    def _validate_depth(
        self,
        depth: int,
    ) -> None:
        """Validate search depth."""

        if not isinstance(depth, int):
            raise TypeError(
                "Search depth must be an integer."
            )

        if depth < 1:
            raise ValueError(
                "Search depth must be at least 1."
            )

    def _safe_moves(
        self,
        state: State,
    ) -> List[Move]:
        """Return generated legal moves as a list."""

        moves = self.generate_moves(state)

        if moves is None:
            return []

        return list(moves)

    def clear_transposition_table(
        self,
    ) -> None:
        """Remove every cached game position."""

        self.transposition_table.clear()

    def clear_killer_moves(
        self,
    ) -> None:
        """Remove all stored killer moves."""

        self.killer_moves.clear()

    def clear_history_table(
        self,
    ) -> None:
        """Remove all accumulated history scores."""

        self.history_table.clear()

    def transposition_table_size(
        self,
    ) -> int:
        """Return the number of cached positions."""

        return len(self.transposition_table)

    def killer_move_count(
        self,
    ) -> int:
        """Return the total number of stored killer moves."""

        return sum(
            len(moves)
            for moves in self.killer_moves.values()
        )

    def history_table_size(
        self,
    ) -> int:
        """Return the number of scored player-move pairs."""

        return len(self.history_table)

    def _find_transposition_move(
        self,
        key: Hashable,
    ) -> Optional[Move]:
        """Return a cached best move, when available."""

        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        return entry.best_move

    def _moves_equal(
        self,
        first: Optional[Move],
        second: Optional[Move],
    ) -> bool:
        """Compare moves using their stable move keys."""

        if first is None or second is None:
            return False

        try:
            return (
                self.move_key(first)
                == self.move_key(second)
            )
        except Exception:
            try:
                return bool(first == second)
            except Exception:
                return False

    def _is_killer_move(
        self,
        move: Move,
        ply: int,
    ) -> bool:
        """Return True when a move is stored as a killer at this ply."""

        if not self.use_killer_moves:
            return False

        killers = self.killer_moves.get(
            ply,
            [],
        )

        return any(
            self._moves_equal(move, killer)
            for killer in killers
        )

    def _store_killer_move(
        self,
        move: Move,
        ply: int,
    ) -> None:
        """Store a cutoff-producing move at one ply."""

        if not self.use_killer_moves:
            return

        killers = self.killer_moves.setdefault(
            ply,
            [],
        )

        for existing in killers:
            if self._moves_equal(
                move,
                existing,
            ):
                return

        killers.insert(
            0,
            move,
        )

        del killers[
            self.killer_slots:
        ]

        self.stats.killer_stores += 1

    def _history_key(
        self,
        player: Player,
        move: Move,
    ) -> Tuple[Player, Hashable]:
        """Create a player-specific history-table key."""

        return (
            player,
            self.move_key(move),
        )

    def _get_history_score(
        self,
        player: Player,
        move: Move,
    ) -> int:
        """Return the accumulated history score for a move."""

        if not self.use_history_heuristic:
            return 0

        key = self._history_key(
            player,
            move,
        )

        return self.history_table.get(
            key,
            0,
        )

    def _update_history(
        self,
        player: Player,
        move: Move,
        depth: int,
    ) -> None:
        """
        Reward a move that caused an alpha-beta cutoff.

        Standard history bonus:

        depthÂ²
        """

        if not self.use_history_heuristic:
            return

        safe_depth = max(
            1,
            int(depth),
        )

        bonus = safe_depth * safe_depth

        key = self._history_key(
            player,
            move,
        )

        self.history_table[key] = (
            self.history_table.get(key, 0)
            + bonus
        )

        self.stats.history_updates += 1

    def get_history_score(
        self,
        player: Player,
        move: Move,
    ) -> int:
        """
        Public helper for inspecting one move's history score.
        """

        return self._get_history_score(
            player,
            move,
        )

    def get_history_entries(
        self,
    ) -> List[
        Tuple[Player, Hashable, int]
    ]:
        """
        Return history entries sorted from strongest to weakest.
        """

        rows = [
            (
                player,
                move_key,
                score,
            )
            for (
                player,
                move_key,
            ), score in self.history_table.items()
        ]

        rows.sort(
            key=lambda row: row[2],
            reverse=True,
        )

        return rows

    def _order_moves(
        self,
        state: State,
        moves: Sequence[Move],
        root_player: Player,
        ply: int,
        transposition_move: Optional[Move] = None,
    ) -> List[Move]:
        """
        Order moves using all enabled heuristics.

        Priority:

        1. Transposition-table move
        2. Previous principal-variation move
        3. Killer move
        4. History heuristic
        5. PokÃ©mon-specific tactical score
        """

        ordered_moves = list(moves)

        if (
            not self.use_move_ordering
            or len(ordered_moves) <= 1
        ):
            return ordered_moves

        pv_move: Optional[Move] = None

        if ply < len(self._previous_iteration_pv):
            pv_move = self._previous_iteration_pv[ply]

        player_to_move = self.current_player(
            state
        )

        scored_moves = []

        for original_index, move in enumerate(
            ordered_moves
        ):
            priority = 0.0

            if self._moves_equal(
                move,
                transposition_move,
            ):
                priority += 10_000_000_000.0

            if self._moves_equal(
                move,
                pv_move,
            ):
                priority += 1_000_000_000.0

            if self._is_killer_move(
                move,
                ply,
            ):
                priority += 100_000_000.0
                self.stats.killer_hits += 1

            history_score = self._get_history_score(
                player_to_move,
                move,
            )

            if history_score > 0:
                priority += (
                    history_score
                    * self.history_multiplier
                )

                self.stats.history_hits += 1

            if self.move_order_score is not None:
                priority += float(
                    self.move_order_score(
                        state,
                        move,
                        root_player,
                    )
                )

            scored_moves.append(
                (
                    priority,
                    -original_index,
                    move,
                )
            )

        scored_moves.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [
            move
            for _, _, move in scored_moves
        ]

    def preview_move_order(
        self,
        state: State,
        root_player: Optional[Player] = None,
        ply: int = 0,
    ) -> List[Move]:
        """Return the order the engine would search."""

        if root_player is None:
            root_player = self.current_player(state)

        legal_moves = self._safe_moves(state)

        key = self.state_key(state)

        transposition_move = (
            self._find_transposition_move(key)
        )

        return self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

    def _probe_transposition_table(
        self,
        key: Hashable,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Optional[Tuple[float, List[Move]]]:
        """Attempt to reuse a cached search result."""

        if not self.use_transposition_table:
            return None

        entry = self.transposition_table.get(key)

        if entry is None:
            return None

        if entry.depth < depth:
            return None

        self.stats.transposition_hits += 1

        if entry.bound_type is BoundType.EXACT:
            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.LOWER_BOUND
            and entry.score >= beta
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        if (
            entry.bound_type
            is BoundType.UPPER_BOUND
            and entry.score <= alpha
        ):
            self.stats.transposition_cutoffs += 1

            return (
                entry.score,
                list(entry.principal_variation),
            )

        return None

    def _store_transposition_entry(
        self,
        key: Hashable,
        depth: int,
        score: float,
        alpha_original: float,
        beta_original: float,
        best_line: List[Move],
    ) -> None:
        """Store a completed node result."""

        if not self.use_transposition_table:
            return

        if score <= alpha_original:
            bound_type = BoundType.UPPER_BOUND

        elif score >= beta_original:
            bound_type = BoundType.LOWER_BOUND

        else:
            bound_type = BoundType.EXACT

        existing = self.transposition_table.get(key)

        if (
            existing is not None
            and existing.depth > depth
        ):
            return

        best_move = (
            best_line[0]
            if best_line
            else None
        )

        self.transposition_table[key] = (
            TranspositionEntry(
                depth=depth,
                score=score,
                bound_type=bound_type,
                best_move=best_move,
                principal_variation=list(best_line),
            )
        )

        self.stats.transposition_stores += 1

    def _alpha_beta(
        self,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
        ply: int = 0,
    ) -> Tuple[float, List[Move]]:
        """
        Search one position with all enabled optimizations.
        """

        self.stats.nodes += 1

        if self.is_terminal(state):
            self.stats.terminal_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        if depth <= 0:
            self.stats.leaf_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        key = self.state_key(state)

        cached_result = (
            self._probe_transposition_table(
                key=key,
                depth=depth,
                alpha=alpha,
                beta=beta,
            )
        )

        if cached_result is not None:
            return cached_result

        legal_moves = self._safe_moves(state)

        if not legal_moves:
            self.stats.leaf_nodes += 1

            return (
                float(
                    self.evaluate_state(
                        state,
                        root_player,
                    )
                ),
                [],
            )

        transposition_move = (
            self._find_transposition_move(key)
        )

        legal_moves = self._order_moves(
            state=state,
            moves=legal_moves,
            root_player=root_player,
            ply=ply,
            transposition_move=transposition_move,
        )

        alpha_original = alpha
        beta_original = beta

        player_to_move = self.current_player(
            state
        )

        maximizing = (
            player_to_move == root_player
        )

        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score > best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                alpha = max(
                    alpha,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1

                    self._store_killer_move(
                        move,
                        ply,
                    )

                    self._update_history(
                        player=player_to_move,
                        move=move,
                        depth=depth,
                    )

                    break

        else:
            best_score = math.inf
            best_line = []

            for move in legal_moves:
                child_state = self.apply_move(
                    state,
                    move,
                )

                child_score, child_line = (
                    self._alpha_beta(
                        state=child_state,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                        ply=ply + 1,
                    )
                )

                if child_score < best_score:
                    best_score = child_score
                    best_line = [move] + child_line

                beta = min(
                    beta,
                    best_score,
                )

                if alpha >= beta:
                    self.stats.cutoffs += 1

                    self._store_killer_move(
                        move,
                        ply,
                    )

                    self._update_history(
                        player=player_to_move,
                        move=move,
                        depth=depth,
                    )

                    break

        self._store_transposition_entry(
            key=key,
            depth=depth,
            score=best_score,
            alpha_original=alpha_original,
            beta_original=beta_original,
            best_line=best_line,
        )

        return best_score, best_line

    def search_depth(
        self,
        state: State,
        depth: int,
        root_player: Optional[Player] = None,
        clear_table: bool = True,
        clear_killers: bool = True,
        clear_history: bool = True,
    ) -> SearchResult:
        """Perform one fixed-depth search."""

        self._validate_depth(depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        if clear_killers:
            self.clear_killer_moves()

        if clear_history:
            self.clear_history_table()

        self._previous_iteration_pv = []

        start_time = time.perf_counter()

        score, principal_variation = (
            self._alpha_beta(
                state=state,
                depth=depth,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
                ply=0,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed

        self.stats.depth_nodes[depth] = (
            self.stats.nodes
        )

        self.stats.depth_times[depth] = elapsed

        best_move = (
            principal_variation[0]
            if principal_variation
            else None
        )

        self._principal_variation = (
            principal_variation
        )

        return SearchResult(
            best_move=best_move,
            score=score,
            completed_depth=depth,
            principal_variation=principal_variation,
            stats=self.stats,
        )

    def iterative_deepening(
        self,
        state: State,
        max_depth: int,
        root_player: Optional[Player] = None,
        verbose: bool = True,
        clear_table: bool = True,
        clear_killers: bool = True,
        clear_history: bool = True,
    ) -> SearchResult:
        """Search depth 1 through max_depth."""

        self._validate_depth(max_depth)

        if root_player is None:
            root_player = self.current_player(state)

        self.stats.reset()

        if clear_table:
            self.clear_transposition_table()

        if clear_killers:
            self.clear_killer_moves()

        if clear_history:
            self.clear_history_table()

        self._previous_iteration_pv = []

        overall_start = time.perf_counter()

        best_move: Optional[Move] = None

        best_score = float(
            self.evaluate_state(
                state,
                root_player,
            )
        )

        best_line: List[Move] = []

        total_nodes = 0
        total_leaf_nodes = 0
        total_terminal_nodes = 0
        total_cutoffs = 0

        total_tt_hits = 0
        total_tt_stores = 0
        total_tt_cutoffs = 0

        total_killer_hits = 0
        total_killer_stores = 0

        total_history_hits = 0
        total_history_updates = 0

        depth_nodes: Dict[int, int] = {}
        depth_times: Dict[int, float] = {}

        completed_depth = 0

        for depth in range(
            1,
            max_depth + 1,
        ):
            self.stats.nodes = 0
            self.stats.leaf_nodes = 0
            self.stats.terminal_nodes = 0
            self.stats.cutoffs = 0

            self.stats.transposition_hits = 0
            self.stats.transposition_stores = 0
            self.stats.transposition_cutoffs = 0

            self.stats.killer_hits = 0
            self.stats.killer_stores = 0

            self.stats.history_hits = 0
            self.stats.history_updates = 0

            depth_start = time.perf_counter()

            score, principal_variation = (
                self._alpha_beta(
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    root_player=root_player,
                    ply=0,
                )
            )

            depth_elapsed = (
                time.perf_counter()
                - depth_start
            )

            depth_nodes[depth] = (
                self.stats.nodes
            )

            depth_times[depth] = (
                depth_elapsed
            )

            total_nodes += self.stats.nodes
            total_leaf_nodes += self.stats.leaf_nodes
            total_terminal_nodes += self.stats.terminal_nodes
            total_cutoffs += self.stats.cutoffs

            total_tt_hits += (
                self.stats.transposition_hits
            )

            total_tt_stores += (
                self.stats.transposition_stores
            )

            total_tt_cutoffs += (
                self.stats.transposition_cutoffs
            )

            total_killer_hits += (
                self.stats.killer_hits
            )

            total_killer_stores += (
                self.stats.killer_stores
            )

            total_history_hits += (
                self.stats.history_hits
            )

            total_history_updates += (
                self.stats.history_updates
            )

            completed_depth = depth
            best_score = score
            best_line = principal_variation

            if principal_variation:
                best_move = principal_variation[0]

            self._previous_iteration_pv = list(
                principal_variation
            )

            if verbose:
                move_name = (
                    self.move_to_string(best_move)
                    if best_move is not None
                    else "None"
                )

                pv_text = " -> ".join(
                    self.move_to_string(move)
                    for move in principal_variation
                )

                print(
                    f"Depth {depth:>2} | "
                    f"Score {score:>10.2f} | "
                    f"Nodes {self.stats.nodes:>8,} | "
                    f"TT {self.stats.transposition_hits:>4,} | "
                    f"Killer {self.stats.killer_hits:>4,} | "
                    f"History {self.stats.history_hits:>4,} | "
                    f"Time {depth_elapsed:>8.4f}s | "
                    f"Best: {move_name}"
                )

                if pv_text:
                    print(
                        f"         PV: {pv_text}"
                    )

        overall_elapsed = (
            time.perf_counter()
            - overall_start
        )

        final_stats = SearchStats(
            nodes=total_nodes,
            leaf_nodes=total_leaf_nodes,
            terminal_nodes=total_terminal_nodes,
            cutoffs=total_cutoffs,

            transposition_hits=total_tt_hits,
            transposition_stores=total_tt_stores,
            transposition_cutoffs=total_tt_cutoffs,

            killer_hits=total_killer_hits,
            killer_stores=total_killer_stores,

            history_hits=total_history_hits,
            history_updates=total_history_updates,

            completed_depth=completed_depth,
            elapsed_seconds=overall_elapsed,
            depth_nodes=depth_nodes,
            depth_times=depth_times,
        )

        self.stats = final_stats
        self._principal_variation = best_line

        return SearchResult(
            best_move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            principal_variation=best_line,
            stats=final_stats,
        )

    def get_principal_variation(
        self,
    ) -> List[Move]:
        """Return a copy of the latest principal variation."""

        return list(
            self._principal_variation
        )


# # Step 59 â€” Recreate the PokÃ©mon Engine

# In[94]:


pokemon_search_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,

    use_transposition_table=True,

    move_order_score=pokemon_move_order_score,
    use_move_ordering=True,

    move_key=pokemon_move_key,

    use_killer_moves=True,
    killer_slots=2,

    use_history_heuristic=True,
    history_multiplier=1_000.0,
)

print("âœ… History-enabled PokÃ©mon engine created.")


# # Step 60 â€” Run the History-Heuristic Search

# In[95]:


history_result = (
    pokemon_search_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=10,
        verbose=True,
        clear_table=True,
        clear_killers=True,
        clear_history=True,
    )
)


# In[96]:


print("=" * 70)
print("HISTORY HEURISTIC RESULTS")
print("=" * 70)

print(
    "Best move:",
    history_result.best_move["name"],
)

print(
    "Score:",
    history_result.score,
)

print(
    "Completed depth:",
    history_result.completed_depth,
)

print(
    "Total nodes:",
    history_result.stats.nodes,
)

print(
    "Cutoffs:",
    history_result.stats.cutoffs,
)

print(
    "Killer hits:",
    history_result.stats.killer_hits,
)

print(
    "History hits:",
    history_result.stats.history_hits,
)

print(
    "History updates:",
    history_result.stats.history_updates,
)

print(
    "History-table size:",
    pokemon_search_engine.history_table_size(),
)


# ## Step 61 â€” Inspect Learned History Scores

# In[97]:


print("Learned PokÃ©mon history scores:")
print()

for (
    player,
    stored_move_key,
    score,
) in pokemon_search_engine.get_history_entries():
    move_name = stored_move_key[0]

    print(
        f"{player:<10} | "
        f"{move_name:<20} | "
        f"History score: {score}"
    )


# ## Step 62 â€” Validate the History Heuristic

# In[98]:


assert history_result.best_move is not None

assert (
    history_result.best_move["name"]
    == "Evolution Burst"
)

assert history_result.completed_depth == 10

assert history_result.stats.nodes > 0

assert history_result.stats.history_updates > 0

assert pokemon_search_engine.history_table_size() > 0

assert (
    history_result.best_move
    == history_result.principal_variation[0]
)

history_entries = (
    pokemon_search_engine.get_history_entries()
)

assert len(history_entries) > 0

assert all(
    entry[2] > 0
    for entry in history_entries
)

print("âœ… Step 62 history-heuristic validation passed.")

print(
    "History entries:",
    pokemon_search_engine.history_table_size(),
)

print(
    "History hits:",
    history_result.stats.history_hits,
)

print(
    "History updates:",
    history_result.stats.history_updates,
)


# ## Step 63
# 
# # Part 7 â€” Principal Variation Extraction
# 
# The principal variation (PV) is the strongest line discovered during search.
# 
# Instead of returning only the best move, the engine returns
# the complete sequence of moves it expects both players to play.
# 
# Every completed search updates the stored PV.
# 
# The PV is used for:
# 
# - analysis
# - debugging
# - tournament display
# - move ordering
# - future searches

# ## Step 64 â€” Display the Current Principal Variation

# In[99]:


print("=" * 70)
print("CURRENT PRINCIPAL VARIATION")
print("=" * 70)

pv = pokemon_search_engine.get_principal_variation()

print(
    "Length:",
    len(pv),
)

print()

for index, move in enumerate(
    pv,
    start=1,
):
    print(
        f"{index}. {move['name']}"
    )


# ## Step 65 â€” Validate the Stored PV

# In[100]:


pv = pokemon_search_engine.get_principal_variation()

assert isinstance(
    pv,
    list,
)

assert len(pv) > 0

assert (
    pv[0]["name"]
    == "Evolution Burst"
)

assert (
    pv
    == history_result.principal_variation
)

assert (
    pokemon_search_engine.get_principal_variation()
    == pv
)

print("âœ… Step 65 principal variation validation passed.")

print()

print(
    "PV length:",
    len(pv),
)

print(
    "First move:",
    pv[0]["name"],
)

print(
    "Stored PV:",
)

for index, move in enumerate(
    pv,
    start=1,
):
    print(
        f"  {index}. {move['name']}"
    )


# ## Step 66 â€” Verify SearchResult Matches Engine PV

# In[101]:


engine_pv = (
    pokemon_search_engine.get_principal_variation()
)

search_pv = (
    history_result.principal_variation
)

assert (
    engine_pv == search_pv
)

assert (
    history_result.best_move
    == engine_pv[0]
)

assert (
    history_result.best_move["name"]
    == "Evolution Burst"
)

print("âœ… Step 66 PV consistency validation passed.")

print()

print(
    "Engine PV:",
    [move["name"] for move in engine_pv],
)

print(
    "SearchResult PV:",
    [move["name"] for move in search_pv],
)


# ## Step 67 â€” Display the PV as a Readable Line

# In[103]:


pv = pokemon_search_engine.get_principal_variation()

pv_string = " â†’ ".join(
    move["name"]
    for move in pv
)

print("=" * 70)
print("PRINCIPAL VARIATION")
print("=" * 70)

print(pv_string)

print()

print(
    "Evaluation:",
    history_result.score,
)

print(
    "Completed depth:",
    history_result.completed_depth,
)


# ## Step 68 â€” Import Timing Utilities

# In[104]:


############################################################
# Step 68 â€” Import Timing Utilities
############################################################

import time
from dataclasses import dataclass

print("âœ… Timing utilities imported.")


# ## Step 69 â€” Create SearchTimeLimit

# In[105]:


############################################################
# Step 69 â€” Search Time Limit
############################################################

@dataclass
class SearchTimeLimit:
    max_seconds: float

    def start(self):
        self.start_time = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start_time

    def remaining(self):
        return max(
            0.0,
            self.max_seconds - self.elapsed(),
        )

    def expired(self):
        return self.elapsed() >= self.max_seconds


# ## Step 70 â€” Validate Timer

# In[106]:


############################################################
# Step 70 â€” Validate Timer
############################################################

pokemon_timer = SearchTimeLimit(
    max_seconds=0.5
)

pokemon_timer.start()

time.sleep(0.15)

elapsed = pokemon_timer.elapsed()

assert elapsed > 0.14
assert elapsed < 0.30

print("âœ… Step 70 timer validation passed.")

print("Elapsed:")
print(f"{elapsed:.3f} seconds")

print("Remaining:")
print(f"{pokemon_timer.remaining():.3f} seconds")

print("Expired:")
print(pokemon_timer.expired())


# # Part 8 â€” Step 71
# ## Add Time-Controlled Iterative Deepening
# 
# ## Now we're going to teach the search engine to stop because of time, not because of depth.

# In[110]:


############################################################
# Step 71 â€” Time-Limited Search
############################################################

def search_for_time(
    self,
    state,
    seconds=1.0,
    verbose=False,
    max_depth=100,
    clear_table=True,
    clear_killers=True,
    clear_history=True,
):
    """
    Run iterative deepening until the time budget expires.

    The deepest fully completed search result is returned.
    Time is checked between completed depths.
    """

    if seconds <= 0:
        raise ValueError(
            "seconds must be greater than 0."
        )

    if max_depth < 1:
        raise ValueError(
            "max_depth must be at least 1."
        )

    timer = SearchTimeLimit(
        max_seconds=float(seconds)
    )
    timer.start()

    if clear_table:
        self.clear_transposition_table()

    if clear_killers:
        self.clear_killer_moves()

    if clear_history:
        self.clear_history_table()

    best_result = None

    for depth in range(1, max_depth + 1):
        if timer.expired():
            break

        depth_start = time.perf_counter()

        result = self.search_depth(
            state=state,
            depth=depth,
            clear_table=False,
            clear_killers=False,
            clear_history=False,
        )

        depth_elapsed = (
            time.perf_counter() - depth_start
        )

        best_result = result

        if verbose:
            best_move_name = (
                self.move_to_string(
                    result.best_move
                )
                if result.best_move is not None
                else "None"
            )

            print(
                f"Depth {depth:>2} | "
                f"Score {result.score:>8.2f} | "
                f"Nodes {result.stats.nodes:>6,} | "
                f"Depth time {depth_elapsed:.4f}s | "
                f"Elapsed {timer.elapsed():.4f}s | "
                f"Remaining {timer.remaining():.4f}s | "
                f"Best: {best_move_name}"
            )

        if timer.expired():
            break

    # Guarantee a result even if the time budget was extremely small.
    if best_result is None:
        best_result = self.search_depth(
            state=state,
            depth=1,
            clear_table=False,
            clear_killers=False,
            clear_history=False,
        )

    self._principal_variation = list(
        best_result.principal_variation
    )

    return best_result


# Attach the method to the active class.
AdvancedSearchEngine.search_for_time = search_for_time

print(
    "âœ… search_for_time attached:",
    hasattr(
        AdvancedSearchEngine,
        "search_for_time",
    ),
)

print(
    "âœ… Existing engine has method:",
    hasattr(
        pokemon_search_engine,
        "search_for_time",
    ),
)


# # Step 72 â€” Recreate the Engine
# ## Because we've added a new method, recreate the engine:

# In[108]:


pokemon_search_engine = AdvancedSearchEngine(
    generate_moves=generate_moves_adapter,
    apply_move=apply_move,
    evaluate_state=evaluate_state_adapter,
    is_terminal=terminal_state_adapter,
    current_player=current_player_adapter,
    move_to_string=lambda move: move["name"],
    state_key=pokemon_zobrist,
    use_transposition_table=True,
    move_order_score=pokemon_move_order_score,
    use_move_ordering=True,
    move_key=pokemon_move_key,
    use_killer_moves=True,
    killer_slots=2,
    use_history_heuristic=True,
)

print("âœ… Time-aware PokÃ©mon engine created.")


# # Step 73 â€” Run a Timed Search

# In[111]:


############################################################
# Step 73 â€” Run Timed Search
############################################################

timed_result = (
    pokemon_search_engine.search_for_time(
        state=pokemon_test_battle,
        seconds=0.20,
        verbose=True,
        max_depth=100,
        clear_table=True,
        clear_killers=True,
        clear_history=True,
    )
)

print()
print("=" * 70)
print("TIMED SEARCH COMPLETE")
print("=" * 70)

print(
    "Best move:",
    timed_result.best_move["name"],
)

print(
    "Score:",
    timed_result.score,
)

print(
    "Completed depth:",
    timed_result.completed_depth,
)

print(
    "Principal variation:",
    [
        move["name"]
        for move
        in timed_result.principal_variation
    ],
)


# ## Step 74 â€” Add the Benchmarking Section

# # Part 9 â€” Engine Benchmarking
# 
# Benchmarking measures the correctness and efficiency of each search-engine
# configuration.
# 
# Every engine must return the same best move and evaluation score.
# 
# We will compare:
# 
# 1. Basic alpha-beta
# 2. Transposition tables
# 3. Move ordering
# 4. Killer moves
# 5. History heuristic
# 6. Fully optimized search
# 
# The benchmark records:
# 
# - Best move
# - Evaluation score
# - Completed depth
# - Nodes searched
# - Alpha-beta cutoffs
# - Transposition-table hits
# - Killer-move hits
# - History-heuristic hits
# - Elapsed time
# - Nodes per second

# # Step 75 â€” Create a Benchmark Result Record
# 

# In[112]:


@dataclass
class EngineBenchmarkResult:
    """One completed search-engine benchmark."""

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

    principal_variation: List[str]

    def as_dict(self) -> Dict[str, Any]:
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
            "principal_variation": (
                " â†’ ".join(self.principal_variation)
            ),
        }


print("âœ… EngineBenchmarkResult created.")


# # Step 76 â€” Create the Benchmark Helper

# In[113]:


def benchmark_engine(
    engine_name: str,
    engine: AdvancedSearchEngine,
    state: BattleState,
    depth: int,
) -> EngineBenchmarkResult:
    """
    Run one fixed-depth benchmark and return a normalized result.
    """

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
        nodes_per_second=(
            result.stats.nodes_per_second
        ),

        principal_variation=pv_names,
    )


print("âœ… Benchmark helper created.")


# # Step 77 â€” Create the Benchmark Engines

# In[114]:


common_engine_arguments = {
    "generate_moves": generate_moves_adapter,
    "apply_move": apply_move,
    "evaluate_state": evaluate_state_adapter,
    "is_terminal": terminal_state_adapter,
    "current_player": current_player_adapter,
    "move_to_string": lambda move: move["name"],
    "state_key": pokemon_zobrist,
    "move_order_score": pokemon_move_order_score,
    "move_key": pokemon_move_key,
}


basic_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=False,
    use_move_ordering=False,
    use_killer_moves=False,
    use_history_heuristic=False,
)


transposition_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=True,
    use_move_ordering=False,
    use_killer_moves=False,
    use_history_heuristic=False,
)


move_ordering_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=False,
    use_history_heuristic=False,
)


killer_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=True,
    killer_slots=2,
    use_history_heuristic=False,
)


history_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=False,
    use_history_heuristic=True,
    history_multiplier=1_000.0,
)


fully_optimized_engine = AdvancedSearchEngine(
    **common_engine_arguments,
    use_transposition_table=True,
    use_move_ordering=True,
    use_killer_moves=True,
    killer_slots=2,
    use_history_heuristic=True,
    history_multiplier=1_000.0,
)


benchmark_engines = [
    ("Basic alpha-beta", basic_engine),
    ("Transposition table", transposition_engine),
    ("Move ordering", move_ordering_engine),
    ("Killer moves", killer_engine),
    ("History heuristic", history_engine),
    ("Fully optimized", fully_optimized_engine),
]


print("âœ… Six benchmark engines created.")


# # Step 78 â€” Run the Fixed-Depth Benchmark

# In[115]:


BENCHMARK_DEPTH = 10

fixed_depth_benchmarks = []

for engine_name, engine in benchmark_engines:
    benchmark_result = benchmark_engine(
        engine_name=engine_name,
        engine=engine,
        state=pokemon_test_battle,
        depth=BENCHMARK_DEPTH,
    )

    fixed_depth_benchmarks.append(
        benchmark_result
    )

    print(
        f"{engine_name:<22} | "
        f"Move {benchmark_result.best_move:<18} | "
        f"Score {benchmark_result.score:>8.2f} | "
        f"Nodes {benchmark_result.nodes:>5,} | "
        f"Cutoffs {benchmark_result.cutoffs:>4,} | "
        f"Time "
        f"{benchmark_result.elapsed_seconds:.6f}s"
    )


# ## Step 79 â€” Display the Benchmark Table

# In[116]:


benchmark_rows = [
    result.as_dict()
    for result in fixed_depth_benchmarks
]

benchmark_rows


# In[117]:


import pandas as pd

benchmark_df = pd.DataFrame(
    benchmark_rows
)

benchmark_df


# # Step 80 â€” Validate Search Correctness Across Engines

# In[118]:


benchmark_best_moves = {
    result.best_move
    for result in fixed_depth_benchmarks
}

benchmark_scores = {
    result.score
    for result in fixed_depth_benchmarks
}

benchmark_depths = {
    result.completed_depth
    for result in fixed_depth_benchmarks
}


assert benchmark_best_moves == {
    "Evolution Burst"
}

assert len(benchmark_scores) == 1

assert benchmark_depths == {
    BENCHMARK_DEPTH
}

assert all(
    result.nodes > 0
    for result in fixed_depth_benchmarks
)

assert all(
    result.principal_variation
    for result in fixed_depth_benchmarks
)

assert all(
    result.best_move
    == result.principal_variation[0]
    for result in fixed_depth_benchmarks
)


print("âœ… Step 80 benchmark correctness validation passed.")

print(
    "Shared best move:",
    next(iter(benchmark_best_moves)),
)

print(
    "Shared score:",
    next(iter(benchmark_scores)),
)

print(
    "Completed depth:",
    BENCHMARK_DEPTH,
)


# # Step 81 â€” Compare Nodes and Speed

# In[119]:


basic_benchmark = fixed_depth_benchmarks[0]
optimized_benchmark = fixed_depth_benchmarks[-1]

nodes_saved = (
    basic_benchmark.nodes
    - optimized_benchmark.nodes
)

if basic_benchmark.nodes > 0:
    node_reduction_percent = (
        nodes_saved
        / basic_benchmark.nodes
        * 100.0
    )
else:
    node_reduction_percent = 0.0

if optimized_benchmark.nodes > 0:
    node_ratio = (
        basic_benchmark.nodes
        / optimized_benchmark.nodes
    )
else:
    node_ratio = math.inf


print("=" * 70)
print("BASIC VS FULLY OPTIMIZED")
print("=" * 70)

print(
    "Basic nodes:",
    basic_benchmark.nodes,
)

print(
    "Optimized nodes:",
    optimized_benchmark.nodes,
)

print(
    "Nodes saved:",
    nodes_saved,
)

print(
    "Node reduction:",
    f"{node_reduction_percent:.2f}%",
)

print(
    "Node-search ratio:",
    f"{node_ratio:.2f}x",
)

print(
    "Basic time:",
    f"{basic_benchmark.elapsed_seconds:.6f}s",
)

print(
    "Optimized time:",
    f"{optimized_benchmark.elapsed_seconds:.6f}s",
)


# # Step 82 â€” Benchmark the Timed Search

# In[120]:


timed_benchmark_budgets = [
    0.01,
    0.05,
    0.10,
    0.20,
]

timed_benchmark_rows = []

for seconds in timed_benchmark_budgets:
    timed_engine = AdvancedSearchEngine(
        **common_engine_arguments,
        use_transposition_table=True,
        use_move_ordering=True,
        use_killer_moves=True,
        killer_slots=2,
        use_history_heuristic=True,
        history_multiplier=1_000.0,
    )

    timed_start = time.perf_counter()

    result = timed_engine.search_for_time(
        state=pokemon_test_battle,
        seconds=seconds,
        verbose=False,
        max_depth=500,
        clear_table=True,
        clear_killers=True,
        clear_history=True,
    )

    actual_elapsed = (
        time.perf_counter() - timed_start
    )

    timed_benchmark_rows.append(
        {
            "time_budget_seconds": seconds,
            "actual_elapsed_seconds": round(
                actual_elapsed,
                6,
            ),
            "completed_depth": (
                result.completed_depth
            ),
            "best_move": (
                result.best_move["name"]
            ),
            "score": result.score,
            "nodes_last_depth": (
                result.stats.nodes
            ),
            "principal_variation": " â†’ ".join(
                move["name"]
                for move
                in result.principal_variation
            ),
        }
    )

    print(
        f"Budget {seconds:>5.2f}s | "
        f"Actual {actual_elapsed:>7.4f}s | "
        f"Depth {result.completed_depth:>3} | "
        f"Best {result.best_move['name']}"
    )


# # Step 83 â€” Display the Timed Benchmark Table

# In[121]:


timed_benchmark_df = pd.DataFrame(
    timed_benchmark_rows
)

timed_benchmark_df


# # Step 84 â€” Validate the Timed Benchmarks

# In[122]:


assert len(timed_benchmark_rows) == 4

assert all(
    row["best_move"] == "Evolution Burst"
    for row in timed_benchmark_rows
)

assert all(
    row["completed_depth"] >= 1
    for row in timed_benchmark_rows
)

assert all(
    row["actual_elapsed_seconds"] > 0
    for row in timed_benchmark_rows
)

assert all(
    row["principal_variation"]
    for row in timed_benchmark_rows
)


print("âœ… Step 84 timed benchmark validation passed.")

for row in timed_benchmark_rows:
    print(
        f"{row['time_budget_seconds']:.2f}s "
        f"budget â†’ depth "
        f"{row['completed_depth']}"
    )


# # Step 85 â€” Final Notebook 11 Validation

# ## Final Notebook 11 Validation
# 
# This final validation confirms that the completed engine supports:
# 
# - Fixed-depth alpha-beta search
# - Iterative deepening
# - Transposition tables
# - Zobrist hashing
# - Move ordering
# - Killer moves
# - History heuristic
# - Principal-variation extraction
# - Time-limited search
# - Engine benchmarking

# In[123]:


final_engine_result = (
    fully_optimized_engine.iterative_deepening(
        state=pokemon_test_battle,
        max_depth=10,
        verbose=False,
        clear_table=True,
        clear_killers=True,
        clear_history=True,
    )
)


assert final_engine_result.best_move is not None

assert (
    final_engine_result.best_move["name"]
    == "Evolution Burst"
)

assert final_engine_result.score == 633.0

assert final_engine_result.completed_depth == 10

assert final_engine_result.principal_variation

assert (
    final_engine_result.best_move
    == final_engine_result.principal_variation[0]
)

assert fully_optimized_engine.transposition_table_size() > 0

assert fully_optimized_engine.killer_move_count() > 0

assert fully_optimized_engine.history_table_size() > 0


print("=" * 70)
print("âœ… NOTEBOOK 11 FINAL VALIDATION PASSED")
print("=" * 70)

print(
    "Best move:",
    final_engine_result.best_move["name"],
)

print(
    "Score:",
    final_engine_result.score,
)

print(
    "Completed depth:",
    final_engine_result.completed_depth,
)

print(
    "Principal variation:",
    [
        move["name"]
        for move
        in final_engine_result.principal_variation
    ],
)

print(
    "Nodes:",
    final_engine_result.stats.nodes,
)

print(
    "Cached positions:",
    fully_optimized_engine.transposition_table_size(),
)

print(
    "Killer moves:",
    fully_optimized_engine.killer_move_count(),
)

print(
    "History entries:",
    fully_optimized_engine.history_table_size(),
)


# In[ ]:




