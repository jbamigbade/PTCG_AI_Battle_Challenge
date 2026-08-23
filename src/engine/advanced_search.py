"""Production advanced search engine from Notebook 11."""

from __future__ import annotations

import math
import random
import time

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
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

from .timer import SearchTimeLimit


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


class ZobristHasher:
    """
    Deterministic 64-bit Zobrist hashing system.

    Each distinct feature receives one pseudo-random 64-bit value.
    A state's final hash is the XOR combination of all feature values.

    The fixed seed ensures that identical features receive identical
    values during the current program run and after a hasher reset.
    """
    MASK_64 = (1 << 64) - 1

    def __init__(self, state_features: StateFeaturesFn, seed: int=20260712) -> None:
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
            raise TypeError('state_features must be a callable function.')
        if not isinstance(seed, int):
            raise TypeError('Zobrist seed must be an integer.')
        self.state_features = state_features
        self.seed = seed
        self._random = random.Random(seed)
        self._feature_values: Dict[Hashable, int] = {}

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

    def _value_for_feature(self, feature: Hashable) -> int:
        """
        Return the 64-bit random number assigned to one feature.
        """
        try:
            hash(feature)
        except TypeError as exc:
            raise TypeError(f'Every Zobrist feature must be hashable. Received feature: {feature!r}') from exc
        if feature not in self._feature_values:
            value = self._random.getrandbits(64)
            while value == 0:
                value = self._random.getrandbits(64)
            self._feature_values[feature] = value
        return self._feature_values[feature]

    def hash_features(self, features: Sequence[Hashable]) -> int:
        """
        Hash a supplied sequence of position features.
        """
        zobrist_hash = 0
        for feature in features:
            zobrist_hash ^= self._value_for_feature(feature)
        return zobrist_hash & self.MASK_64

    def hash_state(self, state: State) -> int:
        """
        Convert a state into features and return its 64-bit hash.
        """
        features = self.state_features(state)
        if features is None:
            raise ValueError('state_features returned None. It must return a sequence of hashable features.')
        return self.hash_features(list(features))

    def __call__(self, state: State) -> int:
        """
        Allow the hasher itself to be used as a state_key function.
        """
        return self.hash_state(state)


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
        return {'nodes': self.nodes, 'leaf_nodes': self.leaf_nodes, 'terminal_nodes': self.terminal_nodes, 'cutoffs': self.cutoffs, 'transposition_hits': self.transposition_hits, 'transposition_stores': self.transposition_stores, 'transposition_cutoffs': self.transposition_cutoffs, 'transposition_hit_rate': round(self.transposition_hit_rate, 6), 'killer_hits': self.killer_hits, 'killer_stores': self.killer_stores, 'history_hits': self.history_hits, 'history_updates': self.history_updates, 'completed_depth': self.completed_depth, 'elapsed_seconds': round(self.elapsed_seconds, 6), 'nodes_per_second': round(self.nodes_per_second, 2), 'depth_nodes': dict(self.depth_nodes), 'depth_times': {depth: round(seconds, 6) for depth, seconds in self.depth_times.items()}}


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

    def as_dict(self, move_to_string: Optional[MoveToStringFn]=None) -> Dict[str, Any]:
        """
        Convert the result into a display-friendly dictionary.
        """
        formatter = move_to_string or str
        return {'best_move': formatter(self.best_move) if self.best_move is not None else None, 'score': self.score, 'completed_depth': self.completed_depth, 'principal_variation': [formatter(move) for move in self.principal_variation], 'stats': self.stats.as_dict()}


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

    def __init__(self, generate_moves: GenerateMovesFn, apply_move: ApplyMoveFn, evaluate_state: EvaluateFn, is_terminal: IsTerminalFn, current_player: CurrentPlayerFn, move_to_string: Optional[MoveToStringFn]=None, state_key: Optional[Callable[[State], Hashable]]=None, use_transposition_table: bool=True, move_order_score: Optional[MoveOrderScoreFn]=None, use_move_ordering: bool=True, move_key: Optional[MoveKeyFn]=None, use_killer_moves: bool=True, killer_slots: int=2, use_history_heuristic: bool=True, history_multiplier: float=1000.0) -> None:
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
            raise ValueError('killer_slots must be at least 1.')
        self.killer_slots = killer_slots
        self.use_history_heuristic = use_history_heuristic
        if history_multiplier < 0:
            raise ValueError('history_multiplier cannot be negative.')
        self.history_multiplier = float(history_multiplier)
        self.stats = SearchStats()
        self.transposition_table: Dict[Hashable, TranspositionEntry] = {}
        self.killer_moves: Dict[int, List[Move]] = {}
        self.history_table: Dict[Tuple[Player, Hashable], int] = {}
        self._principal_variation: List[Move] = []
        self._previous_iteration_pv: List[Move] = []

    @staticmethod
    def _default_state_key(state: State) -> Hashable:
        """Use a hashable game state directly as its key."""
        try:
            hash(state)
        except TypeError as exc:
            raise TypeError('The game state is not hashable. Supply a state_key function.') from exc
        return state

    @staticmethod
    def _default_move_key(move: Move) -> Hashable:
        """Use a hashable move directly as its move key."""
        try:
            hash(move)
        except TypeError as exc:
            raise TypeError('The move is not hashable. Supply a move_key function.') from exc
        return move

    def _validate_depth(self, depth: int) -> None:
        """Validate search depth."""
        if not isinstance(depth, int):
            raise TypeError('Search depth must be an integer.')
        if depth < 1:
            raise ValueError('Search depth must be at least 1.')

    def _safe_moves(self, state: State) -> List[Move]:
        """Return generated legal moves as a list."""
        moves = self.generate_moves(state)
        if moves is None:
            return []
        return list(moves)

    def clear_transposition_table(self) -> None:
        """Remove every cached game position."""
        self.transposition_table.clear()

    def clear_killer_moves(self) -> None:
        """Remove all stored killer moves."""
        self.killer_moves.clear()

    def clear_history_table(self) -> None:
        """Remove all accumulated history scores."""
        self.history_table.clear()

    def transposition_table_size(self) -> int:
        """Return the number of cached positions."""
        return len(self.transposition_table)

    def killer_move_count(self) -> int:
        """Return the total number of stored killer moves."""
        return sum((len(moves) for moves in self.killer_moves.values()))

    def history_table_size(self) -> int:
        """Return the number of scored player-move pairs."""
        return len(self.history_table)

    def _find_transposition_move(self, key: Hashable) -> Optional[Move]:
        """Return a cached best move, when available."""
        if not self.use_transposition_table:
            return None
        entry = self.transposition_table.get(key)
        if entry is None:
            return None
        return entry.best_move

    def _moves_equal(self, first: Optional[Move], second: Optional[Move]) -> bool:
        """Compare moves using their stable move keys."""
        if first is None or second is None:
            return False
        try:
            return self.move_key(first) == self.move_key(second)
        except Exception:
            try:
                return bool(first == second)
            except Exception:
                return False

    def _is_killer_move(self, move: Move, ply: int) -> bool:
        """Return True when a move is stored as a killer at this ply."""
        if not self.use_killer_moves:
            return False
        killers = self.killer_moves.get(ply, [])
        return any((self._moves_equal(move, killer) for killer in killers))

    def _store_killer_move(self, move: Move, ply: int) -> None:
        """Store a cutoff-producing move at one ply."""
        if not self.use_killer_moves:
            return
        killers = self.killer_moves.setdefault(ply, [])
        for existing in killers:
            if self._moves_equal(move, existing):
                return
        killers.insert(0, move)
        del killers[self.killer_slots:]
        self.stats.killer_stores += 1

    def _history_key(self, player: Player, move: Move) -> Tuple[Player, Hashable]:
        """Create a player-specific history-table key."""
        return (player, self.move_key(move))

    def _get_history_score(self, player: Player, move: Move) -> int:
        """Return the accumulated history score for a move."""
        if not self.use_history_heuristic:
            return 0
        key = self._history_key(player, move)
        return self.history_table.get(key, 0)

    def _update_history(self, player: Player, move: Move, depth: int) -> None:
        """
        Reward a move that caused an alpha-beta cutoff.

        Standard history bonus:

        depthÂ²
        """
        if not self.use_history_heuristic:
            return
        safe_depth = max(1, int(depth))
        bonus = safe_depth * safe_depth
        key = self._history_key(player, move)
        self.history_table[key] = self.history_table.get(key, 0) + bonus
        self.stats.history_updates += 1

    def get_history_score(self, player: Player, move: Move) -> int:
        """
        Public helper for inspecting one move's history score.
        """
        return self._get_history_score(player, move)

    def get_history_entries(self) -> List[Tuple[Player, Hashable, int]]:
        """
        Return history entries sorted from strongest to weakest.
        """
        rows = [(player, move_key, score) for (player, move_key), score in self.history_table.items()]
        rows.sort(key=lambda row: row[2], reverse=True)
        return rows

    def _order_moves(self, state: State, moves: Sequence[Move], root_player: Player, ply: int, transposition_move: Optional[Move]=None) -> List[Move]:
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
        if not self.use_move_ordering or len(ordered_moves) <= 1:
            return ordered_moves
        pv_move: Optional[Move] = None
        if ply < len(self._previous_iteration_pv):
            pv_move = self._previous_iteration_pv[ply]
        player_to_move = self.current_player(state)
        scored_moves = []
        for original_index, move in enumerate(ordered_moves):
            priority = 0.0
            if self._moves_equal(move, transposition_move):
                priority += 10000000000.0
            if self._moves_equal(move, pv_move):
                priority += 1000000000.0
            if self._is_killer_move(move, ply):
                priority += 100000000.0
                self.stats.killer_hits += 1
            history_score = self._get_history_score(player_to_move, move)
            if history_score > 0:
                priority += history_score * self.history_multiplier
                self.stats.history_hits += 1
            if self.move_order_score is not None:
                priority += float(self.move_order_score(state, move, root_player))
            scored_moves.append((priority, -original_index, move))
        scored_moves.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [move for _, _, move in scored_moves]

    def preview_move_order(self, state: State, root_player: Optional[Player]=None, ply: int=0) -> List[Move]:
        """Return the order the engine would search."""
        if root_player is None:
            root_player = self.current_player(state)
        legal_moves = self._safe_moves(state)
        key = self.state_key(state)
        transposition_move = self._find_transposition_move(key)
        return self._order_moves(state=state, moves=legal_moves, root_player=root_player, ply=ply, transposition_move=transposition_move)

    def _probe_transposition_table(self, key: Hashable, depth: int, alpha: float, beta: float) -> Optional[Tuple[float, List[Move]]]:
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
            return (entry.score, list(entry.principal_variation))
        if entry.bound_type is BoundType.LOWER_BOUND and entry.score >= beta:
            self.stats.transposition_cutoffs += 1
            return (entry.score, list(entry.principal_variation))
        if entry.bound_type is BoundType.UPPER_BOUND and entry.score <= alpha:
            self.stats.transposition_cutoffs += 1
            return (entry.score, list(entry.principal_variation))
        return None

    def _store_transposition_entry(self, key: Hashable, depth: int, score: float, alpha_original: float, beta_original: float, best_line: List[Move]) -> None:
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
        if existing is not None and existing.depth > depth:
            return
        best_move = best_line[0] if best_line else None
        self.transposition_table[key] = TranspositionEntry(depth=depth, score=score, bound_type=bound_type, best_move=best_move, principal_variation=list(best_line))
        self.stats.transposition_stores += 1

    def _alpha_beta(self, state: State, depth: int, alpha: float, beta: float, root_player: Player, ply: int=0) -> Tuple[float, List[Move]]:
        """
        Search one position with all enabled optimizations.
        """
        self.stats.nodes += 1
        if self.is_terminal(state):
            self.stats.terminal_nodes += 1
            return (float(self.evaluate_state(state, root_player)), [])
        if depth <= 0:
            self.stats.leaf_nodes += 1
            return (float(self.evaluate_state(state, root_player)), [])
        key = self.state_key(state)
        cached_result = self._probe_transposition_table(key=key, depth=depth, alpha=alpha, beta=beta)
        if cached_result is not None:
            return cached_result
        legal_moves = self._safe_moves(state)
        if not legal_moves:
            self.stats.leaf_nodes += 1
            return (float(self.evaluate_state(state, root_player)), [])
        transposition_move = self._find_transposition_move(key)
        legal_moves = self._order_moves(state=state, moves=legal_moves, root_player=root_player, ply=ply, transposition_move=transposition_move)
        alpha_original = alpha
        beta_original = beta
        player_to_move = self.current_player(state)
        maximizing = player_to_move == root_player
        if maximizing:
            best_score = -math.inf
            best_line: List[Move] = []
            for move in legal_moves:
                child_state = self.apply_move(state, move)
                child_score, child_line = self._alpha_beta(state=child_state, depth=depth - 1, alpha=alpha, beta=beta, root_player=root_player, ply=ply + 1)
                if child_score > best_score:
                    best_score = child_score
                    best_line = [move] + child_line
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    self.stats.cutoffs += 1
                    self._store_killer_move(move, ply)
                    self._update_history(player=player_to_move, move=move, depth=depth)
                    break
        else:
            best_score = math.inf
            best_line = []
            for move in legal_moves:
                child_state = self.apply_move(state, move)
                child_score, child_line = self._alpha_beta(state=child_state, depth=depth - 1, alpha=alpha, beta=beta, root_player=root_player, ply=ply + 1)
                if child_score < best_score:
                    best_score = child_score
                    best_line = [move] + child_line
                beta = min(beta, best_score)
                if alpha >= beta:
                    self.stats.cutoffs += 1
                    self._store_killer_move(move, ply)
                    self._update_history(player=player_to_move, move=move, depth=depth)
                    break
        self._store_transposition_entry(key=key, depth=depth, score=best_score, alpha_original=alpha_original, beta_original=beta_original, best_line=best_line)
        return (best_score, best_line)

    def search_depth(self, state: State, depth: int, root_player: Optional[Player]=None, clear_table: bool=True, clear_killers: bool=True, clear_history: bool=True) -> SearchResult:
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
        score, principal_variation = self._alpha_beta(state=state, depth=depth, alpha=-math.inf, beta=math.inf, root_player=root_player, ply=0)
        elapsed = time.perf_counter() - start_time
        self.stats.completed_depth = depth
        self.stats.elapsed_seconds = elapsed
        self.stats.depth_nodes[depth] = self.stats.nodes
        self.stats.depth_times[depth] = elapsed
        best_move = principal_variation[0] if principal_variation else None
        self._principal_variation = principal_variation
        return SearchResult(best_move=best_move, score=score, completed_depth=depth, principal_variation=principal_variation, stats=self.stats)

    def iterative_deepening(self, state: State, max_depth: int, root_player: Optional[Player]=None, verbose: bool=True, clear_table: bool=True, clear_killers: bool=True, clear_history: bool=True) -> SearchResult:
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
        best_score = float(self.evaluate_state(state, root_player))
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
            self.stats.history_hits = 0
            self.stats.history_updates = 0
            depth_start = time.perf_counter()
            score, principal_variation = self._alpha_beta(state=state, depth=depth, alpha=-math.inf, beta=math.inf, root_player=root_player, ply=0)
            depth_elapsed = time.perf_counter() - depth_start
            depth_nodes[depth] = self.stats.nodes
            depth_times[depth] = depth_elapsed
            total_nodes += self.stats.nodes
            total_leaf_nodes += self.stats.leaf_nodes
            total_terminal_nodes += self.stats.terminal_nodes
            total_cutoffs += self.stats.cutoffs
            total_tt_hits += self.stats.transposition_hits
            total_tt_stores += self.stats.transposition_stores
            total_tt_cutoffs += self.stats.transposition_cutoffs
            total_killer_hits += self.stats.killer_hits
            total_killer_stores += self.stats.killer_stores
            total_history_hits += self.stats.history_hits
            total_history_updates += self.stats.history_updates
            completed_depth = depth
            best_score = score
            best_line = principal_variation
            if principal_variation:
                best_move = principal_variation[0]
            self._previous_iteration_pv = list(principal_variation)
            if verbose:
                move_name = self.move_to_string(best_move) if best_move is not None else 'None'
                pv_text = ' -> '.join((self.move_to_string(move) for move in principal_variation))
                print(f'Depth {depth:>2} | Score {score:>10.2f} | Nodes {self.stats.nodes:>8,} | TT {self.stats.transposition_hits:>4,} | Killer {self.stats.killer_hits:>4,} | History {self.stats.history_hits:>4,} | Time {depth_elapsed:>8.4f}s | Best: {move_name}')
                if pv_text:
                    print(f'         PV: {pv_text}')
        overall_elapsed = time.perf_counter() - overall_start
        final_stats = SearchStats(nodes=total_nodes, leaf_nodes=total_leaf_nodes, terminal_nodes=total_terminal_nodes, cutoffs=total_cutoffs, transposition_hits=total_tt_hits, transposition_stores=total_tt_stores, transposition_cutoffs=total_tt_cutoffs, killer_hits=total_killer_hits, killer_stores=total_killer_stores, history_hits=total_history_hits, history_updates=total_history_updates, completed_depth=completed_depth, elapsed_seconds=overall_elapsed, depth_nodes=depth_nodes, depth_times=depth_times)
        self.stats = final_stats
        self._principal_variation = best_line
        return SearchResult(best_move=best_move, score=best_score, completed_depth=completed_depth, principal_variation=best_line, stats=final_stats)

    def get_principal_variation(self) -> List[Move]:
        """Return a copy of the latest principal variation."""
        return list(self._principal_variation)


def search_for_time(self, state, seconds=1.0, verbose=False, max_depth=100, clear_table=True, clear_killers=True, clear_history=True):
    """
    Run iterative deepening until the time budget expires.

    The deepest fully completed search result is returned.
    Time is checked between completed depths.
    """
    if seconds <= 0:
        raise ValueError('seconds must be greater than 0.')
    if max_depth < 1:
        raise ValueError('max_depth must be at least 1.')
    timer = SearchTimeLimit(max_seconds=float(seconds))
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
        result = self.search_depth(state=state, depth=depth, clear_table=False, clear_killers=False, clear_history=False)
        depth_elapsed = time.perf_counter() - depth_start
        best_result = result
        if verbose:
            best_move_name = self.move_to_string(result.best_move) if result.best_move is not None else 'None'
            print(f'Depth {depth:>2} | Score {result.score:>8.2f} | Nodes {result.stats.nodes:>6,} | Depth time {depth_elapsed:.4f}s | Elapsed {timer.elapsed():.4f}s | Remaining {timer.remaining():.4f}s | Best: {best_move_name}')
        if timer.expired():
            break
    if best_result is None:
        best_result = self.search_depth(state=state, depth=1, clear_table=False, clear_killers=False, clear_history=False)
    self._principal_variation = list(best_result.principal_variation)
    return best_result


# Attach the timed-search function to the engine class.
AdvancedSearchEngine.search_for_time = search_for_time

