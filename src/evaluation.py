"""Position evaluation for Pokémon TCG search."""

from __future__ import annotations

from .battle_state import BattleState, PokemonState


def _maximum_hp(pokemon: PokemonState) -> float:
    """
    Read the printed maximum HP from a card dictionary.

    Several possible field names are supported because earlier
    notebooks may use slightly different schemas.
    """

    candidate_keys = (
        "hp",
        "HP",
        "hp_numeric",
        "HP Numeric",
    )

    for key in candidate_keys:
        value = pokemon.card.get(key)

        if value is None:
            continue

        try:
            maximum_hp = float(value)

            if maximum_hp > 0:
                return maximum_hp

        except (TypeError, ValueError):
            continue

    # Safe fallback when printed HP is unavailable.
    return max(
        1.0,
        float(pokemon.current_hp)
        + float(pokemon.damage),
    )


def _hp_ratio(pokemon: PokemonState) -> float:
    maximum_hp = _maximum_hp(pokemon)

    return max(
        0.0,
        min(
            1.0,
            float(pokemon.current_hp) / maximum_hp,
        ),
    )


def _bench_strength(
    bench: list[PokemonState],
) -> float:
    """Return total surviving-HP ratio for a bench."""

    return sum(
        _hp_ratio(pokemon)
        for pokemon in bench
    )


def evaluate_position(
    battle_state: BattleState,
) -> float:
    """
    Evaluate a position from the Player's perspective.

    Positive values favor Player.
    Negative values favor Opponent.
    """

    player = battle_state.player
    opponent = battle_state.opponent

    # Strong terminal values.
    if player.prize_cards_remaining <= 0:
        return 10_000.0

    if opponent.prize_cards_remaining <= 0:
        return -10_000.0

    score = 0.0

    # Active Pokémon survival.
    score += 100.0 * (
        _hp_ratio(player.active)
        - _hp_ratio(opponent.active)
    )

    # Prize-card progress. Fewer remaining prizes is better.
    score += 40.0 * (
        opponent.prize_cards_remaining
        - player.prize_cards_remaining
    )

    # Attached Energy represents attack readiness.
    score += 8.0 * (
        player.active.attached_energy
        - opponent.active.attached_energy
    )

    # Bench depth and health.
    score += 10.0 * (
        _bench_strength(player.bench)
        - _bench_strength(opponent.bench)
    )

    # Small hand-size advantage.
    score += 1.5 * (
        player.hand_size
        - opponent.hand_size
    )

    # Knocked-Out Active Pokémon penalty.
    if player.active.current_hp <= 0:
        score -= 500.0

    if opponent.active.current_hp <= 0:
        score += 500.0

    return round(score, 2)
