from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Any
import cg.api as cg_api
from cg.api import Observation, all_card_data, to_observation_class
from src.card_database import load_repository
OPTION_BASE_PRIORITY = {'ATTACK': 100.0, 'EVOLVE': 80.0, 'ABILITY': 70.0, 'ATTACH': 60.0, 'PLAY': 50.0, 'RETREAT': 40.0, 'DISCARD': 20.0, 'END': 0.0}
repository = load_repository()
OFFICIAL_LOOKUP = {int(card.cardId): card for card in all_card_data()}

@dataclass(frozen=True)
class CardInstance:
    """
    One visible card instance in the current game.

    card_id identifies the card definition.
    serial identifies this physical copy in the match.
    """
    card_id: int
    serial: int
    player_index: int
    name: str | None = None
    category: str | None = None
    stage_or_type: str | None = None
    official_card_type: str | None = None
    energy_type: str | None = None
    is_ex: bool = False
    is_mega_ex: bool = False
    is_tera: bool = False
    is_ace_spec: bool = False

@dataclass(frozen=True)
class PokemonInstance:
    """
    One PokÃ©mon currently in play.
    """
    card_id: int
    serial: int
    player_index: int
    name: str | None
    hp: int
    max_hp: int
    damage: int
    appeared_this_turn: bool
    energy_types: tuple[str, ...] = ()
    energy_cards: tuple[CardInstance, ...] = ()
    tools: tuple[CardInstance, ...] = ()
    pre_evolution: tuple[CardInstance, ...] = ()

@dataclass(frozen=True)
class PlayerSnapshot:
    """
    Visible information for one player.
    """
    player_index: int
    active: tuple[PokemonInstance | None, ...]
    bench: tuple[PokemonInstance, ...]
    bench_max: int
    deck_count: int
    discard: tuple[CardInstance, ...]
    prize: tuple[CardInstance | None, ...]
    hand_count: int
    hand: tuple[CardInstance, ...] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool

@dataclass(frozen=True)
class LegalOption:
    """
    One legal option exposed by Kaggle.

    option_index is the integer that must be returned to the simulator.
    """
    option_index: int
    option_type: str
    number: int | None = None
    area: str | None = None
    index: int | None = None
    player_index: int | None = None
    tool_index: int | None = None
    energy_index: int | None = None
    count: int | None = None
    in_play_area: str | None = None
    in_play_index: int | None = None
    attack_id: int | None = None
    card_id: int | None = None
    serial: int | None = None
    special_condition: str | None = None
    card_name: str | None = None
    semantic_label: str | None = None

@dataclass(frozen=True)
class SelectionSnapshot:
    """
    Current Kaggle selection request.
    """
    select_type: str
    context: str
    min_count: int
    max_count: int
    remain_damage_counter: int
    remain_energy_cost: int
    options: tuple[LegalOption, ...]
    deck: tuple[CardInstance, ...] | None = None
    context_card: CardInstance | None = None
    effect_card: CardInstance | None = None

@dataclass(frozen=True)
class BattleSnapshot:
    """
    Stable Team Jesus representation of one observation.
    """
    turn: int
    turn_action_count: int
    your_index: int
    first_player: int
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    result: int
    stadium: tuple[CardInstance, ...]
    looking: tuple[CardInstance | None, ...] | None
    players: tuple[PlayerSnapshot, ...]
    selection: SelectionSnapshot | None
    log_count: int
    search_begin_input: str | None

def enum_name(value: Any) -> str | None:
    """
    Convert enum-like values into stable text.
    """
    if value is None:
        return None
    name = getattr(value, 'name', None)
    if name is not None:
        return str(name)
    return str(value)

def enum_name(value: Any) -> str | None:
    """
    Convert enum-like values into stable text.
    """
    if value is None:
        return None
    name = getattr(value, 'name', None)
    if name is not None:
        return str(name)
    return str(value)

def adapt_card(card: Card | None, *, repository: CardRepository) -> CardInstance | None:
    """
    Convert one official cg.api.Card into a stable CardInstance.
    """
    if card is None:
        return None
    card_id = int(card.id)
    record = repository.get_optional(card_id)
    return CardInstance(card_id=card_id, serial=int(card.serial), player_index=int(card.playerIndex), name=record.name if record else None, category=record.card_category if record is not None and hasattr(record, 'card_category') else record.category if record is not None else None, stage_or_type=record.stage_or_type if record is not None else None)

def adapt_pokemon(pokemon: Pokemon | None, *, repository: CardRepository, player_index: int) -> PokemonInstance | None:
    """
    Convert one official cg.api.Pokemon into a stable PokemonInstance.
    """
    if pokemon is None:
        return None
    card_id = int(pokemon.id)
    record = repository.get_optional(card_id)
    hp = int(pokemon.hp)
    max_hp = int(pokemon.maxHp)
    energy_types = tuple((enum_name(energy) or 'UNKNOWN' for energy in pokemon.energies))
    energy_cards = tuple((card for card in (adapt_card(energy_card, repository=repository) for energy_card in pokemon.energyCards) if card is not None))
    tools = tuple((card for card in (adapt_card(tool_card, repository=repository) for tool_card in pokemon.tools) if card is not None))
    pre_evolution = tuple((card for card in (adapt_card(previous_card, repository=repository) for previous_card in pokemon.preEvolution) if card is not None))
    return PokemonInstance(card_id=card_id, serial=int(pokemon.serial), player_index=int(player_index), name=record.name if record else None, hp=hp, max_hp=max_hp, damage=max(0, max_hp - hp), appeared_this_turn=bool(pokemon.appearThisTurn), energy_types=energy_types, energy_cards=energy_cards, tools=tools, pre_evolution=pre_evolution)

def adapt_player_state(player_state: cg_api.PlayerState, *, player_index: int, repository: CardRepository, official_lookup: dict[int, Any]) -> PlayerSnapshot:
    """
    Convert one official PlayerState into a Team Jesus PlayerSnapshot.
    """
    active = tuple((adapt_pokemon(pokemon, repository=repository, player_index=player_index) if pokemon is not None else None for pokemon in player_state.active))
    bench = tuple((pokemon for pokemon in (adapt_pokemon(item, repository=repository, player_index=player_index) for item in player_state.bench) if pokemon is not None))
    discard = tuple((card for card in (adapt_card(item, repository=repository) for item in player_state.discard) if card is not None))
    prize = tuple((adapt_card(item, repository=repository) if item is not None else None for item in player_state.prize))
    hand = tuple((card for card in (adapt_card(item, repository=repository) for item in player_state.hand) if card is not None)) if player_state.hand is not None else None
    return PlayerSnapshot(player_index=int(player_index), active=active, bench=bench, bench_max=int(player_state.benchMax), deck_count=int(player_state.deckCount), discard=discard, prize=prize, hand_count=int(player_state.handCount), hand=hand, poisoned=bool(player_state.poisoned), burned=bool(player_state.burned), asleep=bool(player_state.asleep), paralyzed=bool(player_state.paralyzed), confused=bool(player_state.confused))

def build_semantic_label(option: cg_api.Option, *, card_name: str | None) -> str:
    """
    Build a human-readable semantic label for one legal option.
    """
    option_type = enum_name(option.type) or 'UNKNOWN'
    parts = [option_type]
    if card_name:
        parts.append(card_name)
    if option.attackId is not None:
        parts.append(f'attack={option.attackId}')
    if option.area is not None:
        parts.append(f'area={enum_name(option.area)}')
    if option.index is not None:
        parts.append(f'index={option.index}')
    if option.playerIndex is not None:
        parts.append(f'player={option.playerIndex}')
    return ' | '.join(parts)


def official_option_type_name(value: Any) -> str | None:
    """
    Normalize official cg.api.Option.type semantics.

    The official schema defines Option.type as cg.api.OptionType, but the
    official observation converter currently leaves the field as a raw int.

    This helper is deliberately restricted to Option.type semantics and
    therefore uses the official cg_api.OptionType enum for normalization.
    """
    if value is None:
        return None

    name = getattr(value, "name", None)

    if name is not None:
        return str(name)

    try:
        return str(
            cg_api.OptionType(
                int(value)
            ).name
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(value)

def adapt_option(option: cg_api.Option, *, option_index: int, repository: CardRepository) -> LegalOption:
    """
    Convert one official legal option into a Team Jesus LegalOption.
    """
    card_id = int(option.cardId) if option.cardId is not None else None
    card_name = None
    if card_id is not None:
        record = repository.get_optional(card_id)
        if record is not None:
            card_name = record.name
    semantic_label = build_semantic_label(option, card_name=card_name)
    return LegalOption(option_index=int(option_index), option_type=official_option_type_name(option.type) or 'UNKNOWN', number=option.number, area=enum_name(option.area), index=option.index, player_index=option.playerIndex, tool_index=option.toolIndex, energy_index=option.energyIndex, count=option.count, in_play_area=enum_name(option.inPlayArea), in_play_index=option.inPlayIndex, attack_id=option.attackId, card_id=card_id, serial=option.serial, special_condition=enum_name(option.specialConditionType), card_name=card_name, semantic_label=semantic_label)

def adapt_selection(selection: cg_api.SelectData | None, *, repository: CardRepository, official_lookup: dict[int, Any]) -> SelectionSnapshot | None:
    """
    Convert the current Kaggle selection request.
    """
    if selection is None:
        return None
    options = tuple((adapt_option(option, option_index=index, repository=repository) for index, option in enumerate(selection.option)))
    deck = tuple((card for card in (adapt_card(item, repository=repository) for item in selection.deck) if card is not None)) if selection.deck is not None else None
    return SelectionSnapshot(select_type=enum_name(selection.type) or 'UNKNOWN', context=enum_name(selection.context) or 'UNKNOWN', min_count=int(selection.minCount), max_count=int(selection.maxCount), remain_damage_counter=int(selection.remainDamageCounter), remain_energy_cost=int(selection.remainEnergyCost), options=options, deck=deck, context_card=adapt_card(selection.contextCard, repository=repository), effect_card=adapt_card(selection.effect, repository=repository))

def adapt_observation(observation: Observation, *, repository: CardRepository, official_lookup: dict[int, Any]) -> BattleSnapshot | None:
    """
    Convert one official Observation into a Team Jesus BattleSnapshot.
    """
    state = observation.current
    if state is None:
        return None
    players = tuple((adapt_player_state(player_state, player_index=index, repository=repository, official_lookup=official_lookup) for index, player_state in enumerate(state.players)))
    stadium = tuple((card for card in (adapt_card(item, repository=repository) for item in state.stadium) if card is not None))
    looking = tuple((adapt_card(item, repository=repository) if item is not None else None for item in state.looking)) if state.looking is not None else None
    selection = adapt_selection(observation.select, repository=repository, official_lookup=official_lookup)
    return BattleSnapshot(turn=int(state.turn), turn_action_count=int(state.turnActionCount), your_index=int(state.yourIndex), first_player=int(state.firstPlayer), supporter_played=bool(state.supporterPlayed), stadium_played=bool(state.stadiumPlayed), energy_attached=bool(state.energyAttached), retreated=bool(state.retreated), result=int(state.result), stadium=stadium, looking=looking, players=players, selection=selection, log_count=len(observation.logs), search_begin_input=observation.search_begin_input)

@dataclass(frozen=True)
class PlayerFeatures:
    """
    Numerical description of one player's visible state.
    """
    player_index: int
    active_count: int
    bench_count: int
    available_bench_slots: int
    total_active_hp: int
    total_active_max_hp: int
    total_active_damage: int
    total_bench_hp: int
    total_bench_max_hp: int
    total_bench_damage: int
    total_energy: int
    active_energy: int
    bench_energy: int
    total_tools: int
    evolved_pokemon: int
    damaged_pokemon: int
    deck_count: int
    discard_count: int
    prize_count: int
    hand_count: int
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool

@dataclass(frozen=True)
class BattleFeatures:
    """
    Numerical and categorical summary of a complete BattleSnapshot.
    """
    turn: int
    turn_action_count: int
    your_index: int
    opponent_index: int
    first_player: int
    supporter_available: bool
    stadium_available: bool
    energy_attachment_available: bool
    retreat_available: bool
    your: PlayerFeatures
    opponent: PlayerFeatures
    prize_advantage: int
    hand_advantage: int
    board_hp_advantage: int
    energy_advantage: int
    bench_advantage: int
    deck_advantage: int
    legal_option_count: int
    legal_option_types: tuple[str, ...]
    terminal_result: int

def pokemon_energy_count(pokemon: PokemonInstance | None) -> int:
    """
    Count attached Energy while avoiding double counting.

    The official runtime may expose both energy types and energy-card objects.
    """
    if pokemon is None:
        return 0
    return max(len(pokemon.energy_types), len(pokemon.energy_cards))

def pokemon_tool_count(pokemon: PokemonInstance | None) -> int:
    if pokemon is None:
        return 0
    return len(pokemon.tools)

def pokemon_damage(pokemon: PokemonInstance | None) -> int:
    if pokemon is None:
        return 0
    return max(0, int(pokemon.max_hp) - int(pokemon.hp))

def extract_player_features(player: PlayerSnapshot) -> PlayerFeatures:
    """
    Extract numerical features from one PlayerSnapshot.
    """
    active = tuple((pokemon for pokemon in player.active if pokemon is not None))
    bench = tuple(player.bench)
    all_pokemon = active + bench
    total_active_hp = sum((int(pokemon.hp) for pokemon in active))
    total_active_max_hp = sum((int(pokemon.max_hp) for pokemon in active))
    total_bench_hp = sum((int(pokemon.hp) for pokemon in bench))
    total_bench_max_hp = sum((int(pokemon.max_hp) for pokemon in bench))
    active_energy = sum((pokemon_energy_count(pokemon) for pokemon in active))
    bench_energy = sum((pokemon_energy_count(pokemon) for pokemon in bench))
    return PlayerFeatures(player_index=int(player.player_index), active_count=len(active), bench_count=len(bench), available_bench_slots=max(0, int(player.bench_max) - len(bench)), total_active_hp=total_active_hp, total_active_max_hp=total_active_max_hp, total_active_damage=sum((pokemon_damage(pokemon) for pokemon in active)), total_bench_hp=total_bench_hp, total_bench_max_hp=total_bench_max_hp, total_bench_damage=sum((pokemon_damage(pokemon) for pokemon in bench)), total_energy=active_energy + bench_energy, active_energy=active_energy, bench_energy=bench_energy, total_tools=sum((pokemon_tool_count(pokemon) for pokemon in all_pokemon)), evolved_pokemon=sum((bool(pokemon.pre_evolution) for pokemon in all_pokemon)), damaged_pokemon=sum((pokemon_damage(pokemon) > 0 for pokemon in all_pokemon)), deck_count=int(player.deck_count), discard_count=len(player.discard), prize_count=sum((card is not None for card in player.prize)), hand_count=int(player.hand_count), poisoned=bool(player.poisoned), burned=bool(player.burned), asleep=bool(player.asleep), paralyzed=bool(player.paralyzed), confused=bool(player.confused))

def extract_battle_features(snapshot: BattleSnapshot) -> BattleFeatures:
    """
    Extract numerical features from a complete BattleSnapshot.
    """
    if len(snapshot.players) < 2:
        raise ValueError('BattleSnapshot must contain at least two players.')
    your_index = int(snapshot.your_index)
    opponent_candidates = [index for index in range(len(snapshot.players)) if index != your_index]
    if not opponent_candidates:
        raise ValueError('Could not determine opponent player index.')
    opponent_index = opponent_candidates[0]
    your_features = extract_player_features(snapshot.players[your_index])
    opponent_features = extract_player_features(snapshot.players[opponent_index])
    selection = snapshot.selection
    legal_option_types = tuple((option.option_type for option in selection.options)) if selection is not None else ()
    return BattleFeatures(turn=int(snapshot.turn), turn_action_count=int(snapshot.turn_action_count), your_index=your_index, opponent_index=opponent_index, first_player=int(snapshot.first_player), supporter_available=not bool(snapshot.supporter_played), stadium_available=not bool(snapshot.stadium_played), energy_attachment_available=not bool(snapshot.energy_attached), retreat_available=not bool(snapshot.retreated), your=your_features, opponent=opponent_features, prize_advantage=opponent_features.prize_count - your_features.prize_count, hand_advantage=your_features.hand_count - opponent_features.hand_count, board_hp_advantage=your_features.total_active_hp + your_features.total_bench_hp - (opponent_features.total_active_hp + opponent_features.total_bench_hp), energy_advantage=your_features.total_energy - opponent_features.total_energy, bench_advantage=your_features.bench_count - opponent_features.bench_count, deck_advantage=your_features.deck_count - opponent_features.deck_count, legal_option_count=len(selection.options) if selection is not None else 0, legal_option_types=legal_option_types, terminal_result=int(snapshot.result))

@dataclass(frozen=True)
class ActionFeatures:
    """
    Numerical and semantic features for one legal option.
    """
    option_index: int
    option_type: str
    semantic_label: str
    card_id: int | None
    card_name: str | None
    attack_id: int | None
    is_play: bool
    is_attach: bool
    is_evolve: bool
    is_ability: bool
    is_retreat: bool
    is_attack: bool
    is_end: bool
    is_discard: bool
    card_is_pokemon: bool
    card_is_trainer: bool
    card_is_energy: bool
    card_battle_score: float
    card_attack_score: float
    card_tank_score: float
    card_mobility_score: float
    base_priority: float

def safe_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)

def extract_action_features(option: LegalOption, *, repository: CardRepository) -> ActionFeatures:
    """
    Convert one legal option into numerical and semantic features.
    """
    option_type = str(option.option_type).upper()
    record = repository.get_optional(option.card_id) if option.card_id is not None else None
    return ActionFeatures(option_index=int(option.option_index), option_type=option_type, semantic_label=option.semantic_label or option_type, card_id=option.card_id, card_name=option.card_name, attack_id=option.attack_id, is_play=option_type == 'PLAY', is_attach=option_type == 'ATTACH', is_evolve=option_type == 'EVOLVE', is_ability=option_type == 'ABILITY', is_retreat=option_type == 'RETREAT', is_attack=option_type == 'ATTACK', is_end=option_type == 'END', is_discard=option_type == 'DISCARD', card_is_pokemon=bool(record.is_pokemon if record is not None else False), card_is_trainer=bool(record.is_trainer if record is not None else False), card_is_energy=bool(record.is_energy if record is not None else False), card_battle_score=safe_float(record.overall_battle_score if record is not None else None), card_attack_score=safe_float(record.attack_power_score if record is not None else None), card_tank_score=safe_float(record.tank_score if record is not None else None), card_mobility_score=safe_float(record.mobility_score if record is not None else None), base_priority=OPTION_BASE_PRIORITY.get(option_type, 10.0))

@dataclass(frozen=True)
class ScoredAction:
    """
    One legal action together with its deterministic score.
    """
    option_index: int
    option_type: str
    semantic_label: str
    score: float
    reasons: tuple[str, ...]

def score_action(action: ActionFeatures, battle: BattleFeatures) -> ScoredAction:
    """
    Produce a deterministic baseline score for one legal action.
    """
    score = float(action.base_priority)
    reasons: list[str] = [f'base={action.base_priority:.1f}']
    if action.is_attack:
        attack_bonus = 20.0 + max(0.0, action.card_attack_score)
        score += attack_bonus
        reasons.append(f'attack_bonus={attack_bonus:.1f}')
        if battle.your.active_energy > 0:
            score += 5.0
            reasons.append('active_energy=5.0')
    if action.is_evolve:
        score += 15.0
        reasons.append('evolution_tempo=15.0')
    if action.is_attach:
        if battle.energy_attachment_available:
            score += 12.0
            reasons.append('attachment_available=12.0')
        else:
            score -= 50.0
            reasons.append('attachment_unavailable=-50.0')
    if action.is_ability:
        score += 10.0
        reasons.append('ability_value=10.0')
    if action.is_play and action.card_is_trainer:
        score += 8.0
        reasons.append('trainer_play=8.0')
    if action.is_play and action.card_is_pokemon:
        if battle.your.available_bench_slots > 0:
            score += 7.0
            reasons.append('bench_development=7.0')
        else:
            score -= 20.0
            reasons.append('bench_full=-20.0')
    if action.is_retreat:
        status_pressure = any([battle.your.poisoned, battle.your.burned, battle.your.asleep, battle.your.paralyzed, battle.your.confused])
        if status_pressure:
            score += 35.0
            reasons.append('status_escape=35.0')
        else:
            score -= 5.0
            reasons.append('retreat_cost=-5.0')
    if action.is_discard:
        score -= 10.0
        reasons.append('discard_penalty=-10.0')
    if action.is_end:
        score -= 25.0
        reasons.append('end_turn_penalty=-25.0')
    score += 0.02 * action.card_battle_score
    if action.card_battle_score:
        reasons.append(f'card_battle_score={0.02 * action.card_battle_score:.2f}')
    return ScoredAction(option_index=action.option_index, option_type=action.option_type, semantic_label=action.semantic_label, score=score, reasons=tuple(reasons))

def rank_legal_actions(snapshot: BattleSnapshot, *, repository: CardRepository) -> list[ScoredAction]:
    """
    Score and rank only the legal options exposed by Kaggle.
    """
    if snapshot.selection is None:
        return []
    battle_features = extract_battle_features(snapshot)
    scored_actions = []
    for option in snapshot.selection.options:
        action_features = extract_action_features(option, repository=repository)
        scored_actions.append(score_action(action_features, battle_features))
    return sorted(scored_actions, key=lambda action: (-action.score, action.option_index))

def choose_best_option_index(snapshot: BattleSnapshot, *, repository: CardRepository) -> int:
    """
    Return the official Kaggle option index with the highest score.
    """
    ranked = rank_legal_actions(snapshot, repository=repository)
    if not ranked:
        raise ValueError('No legal options are available.')
    return int(ranked[0].option_index)

@dataclass(frozen=True)
class PolicyDecision:
    """
    Complete policy result for debugging and evaluation.
    """
    option_index: int
    used_fallback: bool
    reason: str
    ranked_actions: tuple[Any, ...]

@dataclass(slots=True)
class BattlePolicy:
    repository: Any
    official_lookup: dict[int, Any]
    debug: bool = False
    fallback_index: int = 0

    def decide_snapshot(self, snapshot: Any) -> PolicyDecision:
        """
        Rank legal actions from an already adapted BattleSnapshot.
        """
        if snapshot is None:
            return PolicyDecision(option_index=int(self.fallback_index), used_fallback=True, reason='BattleSnapshot was None.', ranked_actions=())
        try:
            ranked = rank_legal_actions(snapshot, repository=self.repository)
        except Exception as exc:
            return PolicyDecision(option_index=int(self.fallback_index), used_fallback=True, reason=f'Action ranking failed: {type(exc).__name__}: {exc}', ranked_actions=())
        if not ranked:
            return PolicyDecision(option_index=int(self.fallback_index), used_fallback=True, reason='No legal ranked actions were available.', ranked_actions=())
        decision = PolicyDecision(option_index=int(ranked[0].option_index), used_fallback=False, reason=f'Selected highest-scoring legal action: {ranked[0].semantic_label}', ranked_actions=tuple(ranked))
        if self.debug:
            self.print_decision(decision)
        return decision

    def decide(self, observation: Any) -> PolicyDecision:
        """
        Convert an official Observation into a safe policy decision.
        """
        try:
            snapshot = adapt_observation(observation, repository=self.repository, official_lookup=self.official_lookup)
        except Exception as exc:
            return PolicyDecision(option_index=int(self.fallback_index), used_fallback=True, reason=f'Observation adaptation failed: {type(exc).__name__}: {exc}', ranked_actions=())
        return self.decide_snapshot(snapshot)

    def choose_action(self, observation: Any) -> int:
        return self.decide(observation).option_index

    def choose_snapshot_action(self, snapshot: Any) -> int:
        return self.decide_snapshot(snapshot).option_index

    def print_decision(self, decision: PolicyDecision) -> None:
        print('=' * 72)
        print('BattlePolicy Decision')
        print('=' * 72)
        print('Chosen option:', decision.option_index)
        print('Fallback used:', decision.used_fallback)
        print('Reason:', decision.reason)
        if not decision.ranked_actions:
            print('Ranked actions: none')
            return
        print('\nRanked actions:')
        for rank, action in enumerate(decision.ranked_actions, start=1):
            print(f'{rank:>2}. index={action.option_index:<3} type={action.option_type:<10} score={action.score:>9.3f} {action.semantic_label}')

def enforce_official_cardinality(ranked_indices, legal_indices, min_count, max_count):
    legal = []
    legal_seen = set()
    for value in legal_indices:
        idx = int(value)
        if idx not in legal_seen:
            legal.append(idx)
            legal_seen.add(idx)
    minimum = max(0, int(min_count))
    maximum = max(0, int(max_count))
    if minimum > maximum:
        raise ValueError(f'Invalid official cardinality: minCount={minimum}, maxCount={maximum}')
    if maximum == 0:
        return []
    ranked_legal = []
    ranked_seen = set()
    for value in ranked_indices:
        idx = int(value)
        if idx in legal_seen and idx not in ranked_seen:
            ranked_legal.append(idx)
            ranked_seen.add(idx)
    result = ranked_legal[:maximum]
    if len(result) < minimum:
        for idx in legal:
            if idx not in result:
                result.append(idx)
            if len(result) >= minimum:
                break
    if len(legal) >= minimum and len(result) < minimum:
        raise RuntimeError('Unable to satisfy official minCount despite sufficient legal options.')
    return result
POLICY = BattlePolicy(repository=repository, official_lookup=OFFICIAL_LOOKUP)

def _load_deck():
    deck_path = 'deck.csv'
    if not os.path.exists(deck_path):
        deck_path = '/kaggle_simulations/agent/deck.csv'
    with open(deck_path, 'r', encoding='utf-8') as file:
        deck = [int(line.strip()) for line in file if line.strip()]
    if len(deck) != 60:
        raise RuntimeError(f'Expected 60-card deck; got {len(deck)}.')
    return deck
MY_DECK = _load_deck()

def agent(obs_dict: dict) -> list[int]:
    observation = to_observation_class(obs_dict)
    if getattr(observation, 'select', None) is None:
        return list(MY_DECK)
    selection = observation.select
    options = list(getattr(selection, 'option', []) or [])
    legal_indices = list(range(len(options)))
    min_count = int(getattr(selection, 'minCount', 0))
    max_count = int(getattr(selection, 'maxCount', 0))
    if max_count == 0:
        return []
    if not legal_indices and min_count > 0:
        raise RuntimeError(f'Official selection requires at least {min_count} choice(s), but selection.option contains no legal options.')
    decision = POLICY.decide(observation)
    primary_index = int(decision.option_index)
    ranked_indices = [int(action.option_index) for action in decision.ranked_actions or ()]
    ranked_indices = [primary_index] + ranked_indices
    return enforce_official_cardinality(ranked_indices=ranked_indices, legal_indices=legal_indices, min_count=min_count, max_count=max_count)