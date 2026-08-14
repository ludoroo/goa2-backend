import logging

from goa2.domain.models import Card, GamePhase
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.steps import (
    FinishedExpiringEffectStep,
    GameStep,
    ResolveTieBreakerStep,
)

logger = logging.getLogger(__name__)


def hero_can_play_two_cards(hero) -> bool:
    """True when the hero's active ultimate allows committing two cards
    during Planning (Emmitt's Alternative Timelines)."""
    from goa2.engine.effects import CardEffectRegistry

    ult = hero.ultimate_card
    if ult is None or hero.level < 8 or not ult.effect_id:
        return False
    effect = CardEffectRegistry.get(ult.effect_id)
    return bool(effect and effect.plays_two_cards)


def commit_card(state: GameState, hero_id: HeroID, card: Card):
    """
    Called when a player selects a card during the Planning Phase.
    Validates that the card is in the player's hand.
    """
    if state.phase != GamePhase.PLANNING:
        logger.warning("Cannot commit card. Game is in %s", state.phase)
        return

    hero = state.get_hero(hero_id)
    if not hero:
        logger.warning("Hero %s not found.", hero_id)
        return

    if hero_id in state.pending_inputs:
        first = state.pending_inputs[hero_id]
        if first is None or not hero_can_play_two_cards(hero):
            raise ValueError(f"{hero_id} has already committed a card this turn")
        if hero_id in state.pending_second_cards:
            raise ValueError(f"{hero_id} has already committed two cards this turn")
        if hero_id in state.planning_done:
            raise ValueError(f"{hero_id} has already finished planning this turn")

        if card not in hero.hand:
            logger.warning(
                "%s tried to play card %s which is not in hand.",
                hero_id,
                card.id,
            )
            return

        try:
            hero.play_card(card)
        except ValueError as e:
            logger.warning("Error playing card: %s", e)
            return
        # play_card points current_turn_card at the latest commit; revelation
        # reassigns it from the buffers, so the transient overwrite is harmless.
        state.pending_second_cards[hero_id] = card
        logger.info("%s committed a second card (Alternative Timelines).", hero_id)

        _check_phase_transition(state)
        return

    # Check if card is in hand
    if card not in hero.hand:
        logger.warning(
            "%s tried to play card %s which is not in hand.",
            hero_id,
            card.id,
        )
        return

    # Move card from hand to pending buffer (Facedown on board)
    # Using helper to ensure state consistency
    try:
        hero.play_card(card)
    except ValueError as e:
        logger.warning("Error playing card: %s", e)
        return

    state.pending_inputs[hero_id] = card
    logger.info("%s committed a card.", hero_id)

    _check_phase_transition(state)


def pass_turn(state: GameState, hero_id: HeroID):
    """
    Called when a player has no cards and must Pass.
    """
    if state.phase != GamePhase.PLANNING:
        raise ValueError(f"Cannot pass in {state.phase} phase")

    hero = state.get_hero(hero_id)
    if not hero:
        raise ValueError(f"Hero {hero_id} not found")

    if hero_id in state.pending_inputs:
        raise ValueError(f"{hero_id} has already completed planning this turn")

    # Rule Check: You must play a card if able.
    if len(hero.hand) > 0:
        raise ValueError(f"{hero_id} cannot pass while holding {len(hero.hand)} card(s)")

    state.pending_inputs[hero_id] = None
    logger.info("%s passed.", hero_id)

    _check_phase_transition(state)


def uncommit_card(state: GameState, hero_id: HeroID) -> Card:
    """
    Take a committed card back into hand during Planning (board-game
    take-back). Allowed until the last player commits — that final commit
    runs revelation synchronously, after which this raises.

    LIFO for a two-card hero (Emmitt's Alternative Timelines): a committed
    second card comes back first; the first commit only after that. Any
    planning-done signal is cleared so the hero can fully rethink.
    """
    if state.phase != GamePhase.PLANNING:
        raise ValueError(f"Cannot uncommit card in {state.phase} phase")

    hero = state.get_hero(hero_id)
    if not hero:
        raise ValueError(f"Hero {hero_id} not found")

    if hero_id not in state.pending_inputs:
        raise ValueError(f"{hero_id} has no committed card to take back")

    first = state.pending_inputs[hero_id]
    if first is None:
        raise ValueError(f"{hero_id} passed (empty hand); nothing to take back")

    second = state.pending_second_cards.get(hero_id)
    if second is not None:
        card = second
        del state.pending_second_cards[hero_id]
        hero.unplay_card(card)
        # play_card pointed current_turn_card at the second commit; the
        # first committed card becomes the pending one again.
        hero.current_turn_card = first
    else:
        card = first
        del state.pending_inputs[hero_id]
        hero.unplay_card(card)
        hero.current_turn_card = None

    if hero_id in state.planning_done:
        state.planning_done.remove(hero_id)

    logger.info("%s took back %s.", hero_id, card.name)
    return card


def finish_planning(state: GameState, hero_id: HeroID):
    """
    Explicit done-signal for a two-card-capable hero (Emmitt's ultimate) who
    chooses to play only one card this turn. No-op for heroes that already
    committed a second card.
    """
    if state.phase != GamePhase.PLANNING:
        logger.warning("Cannot finish planning. Game is in %s", state.phase)
        return

    if hero_id not in state.pending_inputs:
        raise ValueError(f"{hero_id} must commit a card before finishing planning")

    if hero_id not in state.planning_done:
        state.planning_done.append(hero_id)
    logger.info("%s finished planning.", hero_id)

    _check_phase_transition(state)


def planning_open_for_second_card(state: GameState, hero_id: HeroID) -> bool:
    """A two-card-capable hero keeps planning open after their first commit
    until they commit a second card, signal done, or run out of cards.

    Also drives the client's ``can_commit_second_card`` hero-view flag: while
    this is True the hero may POST a second card or call planning-done."""
    card = state.pending_inputs.get(hero_id)
    if card is None:  # passed (or not committed — caller checks count)
        return False
    hero = state.get_hero(hero_id)
    if not hero or not hero_can_play_two_cards(hero):
        return False
    return (
        hero_id not in state.pending_second_cards
        and hero_id not in state.planning_done
        and len(hero.hand) > 0
    )


def _check_phase_transition(state: GameState):
    # Check if all heroes have committed (Card or Pass)
    total_heroes = sum(len(team.heroes) for team in state.teams.values())
    if len(state.pending_inputs) < total_heroes:
        return
    # Two-card-capable heroes must close planning explicitly (second commit,
    # done-signal, or empty hand)
    if any(planning_open_for_second_card(state, h_id) for h_id in state.pending_inputs):
        return
    start_revelation_phase(state)


def start_revelation_phase(state: GameState):
    """
    Reveals all cards and sets up the unresolved pool.
    """
    state.phase = GamePhase.REVELATION
    logger.info("Revelation phase started.")

    state.unresolved_hero_ids = []

    # Assign cards to heroes and populate the unresolved list
    for h_id, card in state.pending_inputs.items():
        # If card is None, the player Passed. They do not enter the resolution pool.
        if card is None:
            continue

        hero = state.get_hero(h_id)
        if hero:
            logger.info(
                "%s reveals %s (initiative: %s)",
                h_id,
                card.name,
                card.initiative,
            )
            card.is_facedown = False
            state.record_public_revealed_card(h_id, str(card.id))
            # card.state is already UNRESOLVED from play_card

            hero.current_turn_card = card

            second = state.pending_second_cards.get(h_id)
            if second is not None:
                logger.info("%s also reveals %s (Alternative Timelines).", h_id, second.name)
                second.is_facedown = False
                state.record_public_revealed_card(h_id, str(second.id))
                hero.extra_turn_card = second

            state.unresolved_hero_ids.append(h_id)
        else:
            logger.warning("Hero %s not found during revelation.", h_id)

    state.pending_inputs = {}  # Clear buffers
    state.pending_second_cards = {}
    state.planning_done = []

    # Transition to Resolution
    start_resolution_phase(state)


def start_resolution_phase(state: GameState):
    state.phase = GamePhase.RESOLUTION
    logger.info("Resolution phase started.")

    # Heroes that revealed two cards must retrieve one before anyone resolves
    # (Emmitt's ultimate). The choice step pauses for input; FindNextActorStep
    # then starts normal initiative resolution.
    dual_hero_ids = [
        h_id
        for h_id in state.unresolved_hero_ids
        if (hero := state.get_hero(h_id)) and hero.extra_turn_card is not None
    ]

    # "Next turn, after playing cards:" payloads (NebKher's Imbue Doubt
    # family) fire here — after reveal (and after any two-card retrieval
    # settles hands), before the first actor. Fires after retrieval so a
    # retrieved card is back in hand when e.g. a forced discard resolves.
    trigger_steps = _collect_after_cards_played_steps(state)

    if dual_hero_ids or trigger_steps:
        from goa2.engine.steps import FindNextActorStep, RetrieveUnresolvedCardStep

        steps: list[GameStep] = [
            RetrieveUnresolvedCardStep(hero_id=str(h_id)) for h_id in dual_hero_ids
        ]
        steps.extend(trigger_steps)
        steps.append(FindNextActorStep())
        push_steps(state, steps)
        return

    resolve_next_action(state)


def _collect_after_cards_played_steps(state: GameState) -> list[GameStep]:
    """Pop due AFTER_CARDS_PLAYED_TRIGGER effects and build their payload
    steps.

    Due = scheduled last turn in the SAME round (created_at_turn + 1 ==
    current turn). Anything else — a cross-round leftover or a stale copy —
    is dropped without firing (NEXT_TURN never crosses rounds). Each payload
    runs with the scheduling hero as current actor so prompts and relational
    filters resolve from the scheduler's perspective; FindNextActorStep
    re-establishes normal initiative afterwards.
    """
    from goa2.domain.models.effect import EffectType

    triggers = [
        e for e in state.active_effects if e.effect_type == EffectType.AFTER_CARDS_PLAYED_TRIGGER
    ]
    if not triggers:
        return []

    state.active_effects = [
        e for e in state.active_effects if e.effect_type != EffectType.AFTER_CARDS_PLAYED_TRIGGER
    ]

    from goa2.engine.steps import SetActorStep

    steps: list[GameStep] = []
    for effect in triggers:
        due = state.round == effect.created_at_round and state.turn == effect.created_at_turn + 1
        if not due:
            logger.info(
                "After-cards-played trigger from %s fizzles (round boundary).",
                effect.source_id,
            )
            continue
        if not effect.finishing_steps:
            continue
        steps.append(SetActorStep(actor_id=str(effect.source_id)))
        steps.extend(effect.finishing_steps)
        steps.append(FinishedExpiringEffectStep())
    return steps


def resolve_next_action(state: GameState):
    """
    Dynamically identifies the next actor based on current initiatives.
    Follows Rule: "After each action... re-identify the player with Highest Initiative".
    """
    if not state.unresolved_hero_ids:
        logger.info("All cards resolved. Turn complete.")
        end_turn(state)
        return

    # 1. Calculate current initiatives for all candidates
    from goa2.domain.models import StatType
    from goa2.engine.stats import get_computed_stat

    candidates: list[tuple[HeroID, int]] = []
    for h_id in state.unresolved_hero_ids:
        hero = state.get_hero(h_id)
        if hero and hero.current_turn_card:
            # Safety Check: Cards must be revealed to have effective initiative > 0
            if hero.current_turn_card.is_facedown:
                logger.warning(
                    "Initiative calculated for facedown card of %s.",
                    h_id,
                )

            # Use Computed Stat (Card Base + Items + Modifiers)
            base_init = hero.current_turn_card.get_base_stat_value(StatType.INITIATIVE)
            total_init = get_computed_stat(state, h_id, StatType.INITIATIVE, base_init)

            candidates.append((h_id, total_init))

    if not candidates:
        return

    # 2. Sort Descending — unless a live REVERSED_INITIATIVE effect (Emmitt's
    # Reverse Time, NEXT_TURN duration) inverts the order: lower computed
    # initiative acts first. Global rule; ignores immunity. Ties unchanged.
    from goa2.domain.models.effect import EffectType

    reversed_order = any(
        e.effect_type == EffectType.REVERSED_INITIATIVE
        and e.is_active
        and state.round == e.created_at_round
        and state.turn == e.created_at_turn + 1
        for e in state.active_effects
    )
    candidates.sort(key=lambda x: x[1], reverse=not reversed_order)

    # 3. Identify Tied Group
    highest_init = candidates[0][1]
    tied_hero_ids = [c[0] for c in candidates if c[1] == highest_init]

    # 4. If no tie -> Resolve immediately
    if len(tied_hero_ids) == 1:
        hero_id = tied_hero_ids[0]
        state.current_actor_id = hero_id
        state.resolution_owner_id = hero_id

        # Remove from pool immediately (Acting/Resolved)
        if hero_id in state.unresolved_hero_ids:
            state.unresolved_hero_ids.remove(hero_id)

        logger.info("Next actor: %s (initiative: %s)", hero_id, highest_init)

        # Convert Card to Steps
        from goa2.engine.steps import (
            ConfirmResolutionStep,
            FinalizeHeroTurnStep,
            ResolveCardStep,
            RespawnHeroStep,
        )

        steps: list[GameStep] = []
        if not state.has_board_presence(hero_id):
            steps.append(RespawnHeroStep(hero_id=hero_id))
        steps.extend(
            [
                ResolveCardStep(hero_id=hero_id),
                ConfirmResolutionStep(hero_id=hero_id),
                FinalizeHeroTurnStep(hero_id=hero_id),
            ]
        )
        push_steps(state, steps)
        return

    # 5. If tie -> Push Tie Breaker Step
    logger.info(
        "Tie detected at initiative %s between %s",
        highest_init,
        tied_hero_ids,
    )

    # We DO NOT remove them from unresolved_hero_ids yet.
    state.execution_stack.append(ResolveTieBreakerStep(tied_hero_ids=tied_hero_ids))


def record_position_snapshot(state: GameState):
    """
    Snapshot every entity's position at the turn boundary.

    Call this wherever the phase becomes PLANNING (turn advance, round reset,
    game creation). Planning moves nothing, so this single snapshot answers
    both "where was that unit at the start of this turn" and "has this unit
    remained in the same space since the last turn" (Emmitt).
    """
    state.last_turn_positions = dict(state.entity_locations)


def end_turn(state: GameState):
    """
    Called when all players have acted in the Resolution Phase.
    Expires THIS_TURN and active NEXT_TURN effects. If any have finishing
    steps, those are pushed onto the stack followed by AdvanceTurnStep
    (deferred advancement). Otherwise, advances synchronously.
    """
    logger.info("End of turn %s.", state.turn)

    # The turn is over: "discarded this turn" resets. (Discards made by
    # finishing steps at this boundary land in the next turn's log.)
    state.turn_discard_log = {}

    from goa2.engine.effect_manager import EffectManager

    finishing = EffectManager.expire_active_turn_effects(state)

    if finishing:
        from goa2.engine.steps import AdvanceTurnStep, SetActorStep

        finish_steps: list[GameStep] = []
        for source_id, steps in finishing:
            finish_steps.append(SetActorStep(actor_id=source_id))
            finish_steps.extend(steps)
            finish_steps.append(FinishedExpiringEffectStep())

        finish_steps.append(AdvanceTurnStep())
        push_steps(state, finish_steps)
        return

    # No finishing steps — advance synchronously (existing behavior)
    if state.turn < 4:
        state.turn += 1
        record_position_snapshot(state)
        state.phase = GamePhase.PLANNING
        logger.info("Start of turn %s. Phase: planning.", state.turn)
        # Auto-pass heroes with no cards in hand
        auto_passed = False
        for team in state.teams.values():
            for hero in team.heroes:
                if len(hero.hand) == 0:
                    state.pending_inputs[hero.id] = None
                    logger.info("%s auto-passed (empty hand).", hero.id)
                    auto_passed = True
        if auto_passed:
            _check_phase_transition(state)
    else:
        start_end_phase(state)


def start_end_phase(state: GameState):
    state.phase = GamePhase.CLEANUP
    logger.info("End phase started.")

    from goa2.engine.steps import EndPhaseStep

    push_steps(state, [EndPhaseStep()])
