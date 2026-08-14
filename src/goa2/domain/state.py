from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from goa2.domain.board import DEFAULT_LANE_ID, Board
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import (
    Card,
    GamePhase,
    Hero,
    ResolutionStep,
    Team,
    TeamColor,
    Token,
    TokenType,
    Unit,
)
from goa2.domain.models.effect import ActiveEffect
from goa2.domain.models.marker import Marker, MarkerType
from goa2.domain.time_control import GameClockState, TimeControlConfig
from goa2.domain.types import BoardEntityID, HeroID, UnitID

logger = logging.getLogger(__name__)


class PublicRevealedCard(BaseModel):
    """A permanently public card identity learned during revelation."""

    model_config = ConfigDict(frozen=True)

    hero_id: str
    card_id: str


class GameState(BaseModel):
    """
    The Mutable State of the World.
    Contains everything needed to serialize/save/restore the game.
    """

    board: Board
    teams: dict[TeamColor, Team]

    # Current Battle Zone per lane (lane_id -> zone_id). Single-lane games have
    # one entry under DEFAULT_LANE_ID; use the `active_zone_id` property for
    # legacy single-lane access.
    battle_zones: dict[str, str] = Field(default_factory=dict)

    phase: GamePhase = GamePhase.SETUP

    resolution_step: ResolutionStep = ResolutionStep.NONE
    round: int = 1
    turn: int = 1
    # Remaining Wave counters per lane (lane_id -> count). Use the
    # `wave_counter` property for legacy single-lane access.
    wave_counters: dict[str, int] = Field(default_factory=lambda: {DEFAULT_LANE_ID: 5})
    cheats_enabled: bool = False
    rng_seed: int | None = None

    current_actor_id: HeroID | None = None  # ID of the Hero currently acting (Resolution Phase)
    # Owner of the unresolved card being resolved. Unlike current_actor_id,
    # defenses and reactions do not temporarily replace this value.
    resolution_owner_id: HeroID | None = None

    # Optional server-authoritative match clock. Missing values preserve
    # untimed behavior for existing requests and legacy saves.
    time_control: TimeControlConfig | None = None
    clock: GameClockState | None = None

    # The team that currently wins ties (Red or Blue)
    # Flips every time a different-team tie is resolved.
    tie_breaker_team: TeamColor = TeamColor.RED

    winner: TeamColor | None = None
    individual_winner_id: HeroID | None = None
    victory_condition: str | None = None

    input_stack: list[InputRequest] = Field(
        default_factory=list
    )  # The top of the stack is the active request waiting for input.
    # Logic:
    # 1. Action pushes Request.
    # 2. State pauses.
    # 3. Client responds to Request[0].
    # 4. Engine pops Request.
    execution_stack: list[Any] = Field(
        default_factory=list
    )  # Stores instances of GameStep (from goa2.engine.steps)
    # Typed as List[Any] to avoid circular imports with steps.py

    execution_context: dict[str, Any] = Field(
        default_factory=dict
    )  # Stores transient data like "selected_target_id" between steps

    pending_inputs: dict[HeroID, Card | None] = Field(
        default_factory=dict
    )  # Planning Phase Buffer: HeroID -> Card

    pending_second_cards: dict[HeroID, Card] = Field(
        default_factory=dict
    )  # Planning Phase Buffer for heroes that may play two cards (Emmitt's ultimate)

    # Append-only public knowledge. The default keeps saves made before this
    # tracker was introduced valid.
    public_revealed_cards: tuple[PublicRevealedCard, ...] = ()

    planning_done: list[HeroID] = Field(
        default_factory=list
    )  # Two-card-capable heroes that explicitly closed planning with one card

    turn_discard_log: dict[HeroID, list[str]] = Field(
        default_factory=dict
    )  # Card IDs each hero discarded THIS TURN (any source); cleared at end_turn.
    # Read by "retrieve all cards discarded this turn" effects (Emmitt).

    card_guess: dict[str, Any] | None = Field(
        default=None
    )  # Public table state for a card-color guess: the facedown card placed for
    # each attempt and, once flipped, its result. Deliberately NOT in
    # execution_context: the reveal, the discard and FinalizeHeroTurnStep all
    # drain in one process_stack pass, so context is already cleared when the
    # post-mutation view is built. Cleared when the next hero turn begins.

    card_reveal: dict[str, Any] | None = Field(
        default=None
    )  # Public table state for an intentionally revealed hand card (Cordelia).
    # Kept outside execution_context so the direct faceup reveal survives the
    # resolving turn's finalization and reconnects. Cleared after another hero
    # finishes a turn, matching card_guess's tabletop presentation lifetime.

    last_turn_positions: dict[BoardEntityID, Hex] = Field(
        default_factory=dict
    )  # Entity positions at the turn boundary, recorded wherever the phase
    # becomes PLANNING. Nothing moves during Planning, so this is both "where
    # the unit was at the start of this turn" and "where it was last turn".
    # Read by Emmitt's Back to the Future / Time Walk / Fast Forward. Units
    # that arrived after the boundary (spawns) simply have no entry.

    pending_upgrades: dict[HeroID, int] = Field(
        default_factory=dict
    )  # Level Up Phase Buffer: HeroID -> Number of upgrades pending

    unresolved_hero_ids: list[HeroID] = Field(
        default_factory=list
    )  # Resolution Phase Tracker: Set of HeroIDs who have not yet acted this turn.
    # We dynamically re-sort this set every step to determine the next actor.
    # Using List for JSON stability, acts as Set.

    # Registry for active non-Unit entities (Tokens, Traps, etc.)
    # Units are stored in self.teams, everything else goes here.
    misc_entities: dict[BoardEntityID, Any] = Field(default_factory=dict)
    token_pool: dict[TokenType, list[Token]] = Field(default_factory=dict)

    # Master Record of positions for ALL entities (Units + Tokens)
    entity_locations: dict[BoardEntityID, Hex] = Field(default_factory=dict)

    next_entity_id: int = 1

    # Multi-piece heroes (Razzle): the piece performing the current action.
    # Bound by ChooseActingPieceStep, cleared by FinalizeHeroTurnStep.
    acting_piece_id: BoardEntityID | None = None

    active_effects: list[ActiveEffect] = Field(default_factory=list)

    # Singleton markers - each MarkerType has exactly one Marker instance
    markers: dict[MarkerType, Marker] = Field(default_factory=dict)

    # Track heroes defeated in the current round (for coin-granting effects)
    heroes_defeated_this_round: list[HeroID] = Field(default_factory=list)

    # Private field for cached validator (not serialized)
    _validator: Any | None = None

    def record_public_revealed_card(self, hero_id: HeroID | str, card_id: str) -> None:
        """Authoritatively and idempotently record a faceup committed card."""
        record = PublicRevealedCard(hero_id=str(hero_id), card_id=str(card_id))
        if record not in self.public_revealed_cards:
            self.public_revealed_cards = (*self.public_revealed_cards, record)

    @property
    def coin_face(self) -> str:
        """The Tie Breaker coin's currently-showing face ("BLUE" or "ORANGE").

        The coin is the same bit as ``tie_breaker_team``: a BLUE-favored coin
        shows its blue face, a RED-favored coin shows its orange face. Ignatia's
        deck branches on this face.
        """
        return "BLUE" if self.tie_breaker_team == TeamColor.BLUE else "ORANGE"

    def add_effect(self, effect: ActiveEffect):
        """Adds a spatial/behavioral effect to the active list."""
        self.active_effects.append(effect)

    def get_marker(self, marker_type: MarkerType) -> Marker:
        """
        Get or create a marker of the given type.
        Markers are singletons - this creates one if it doesn't exist.
        """
        if marker_type not in self.markers:
            self.markers[marker_type] = Marker(type=marker_type)
        return self.markers[marker_type]

    def place_marker(
        self, marker_type: MarkerType, target_id: str, value: int, source_id: str
    ) -> Marker:
        """
        Place a marker on a target hero or hero piece.
        A marker placed on a HeroPiece records that piece as the target, but
        markers are a shared hero inventory: the penalty applies to every
        piece's attack/defense and counts once at the owner level for
        initiative (stats.py).
        If marker was on another hero, it automatically leaves them (singleton).
        """
        marker = self.get_marker(marker_type)
        marker.place(target_id=target_id, value=value, source_id=source_id)
        return marker

    def remove_marker(self, marker_type: MarkerType) -> Marker | None:
        """
        Return a marker to supply (remove from its current target).
        Returns the marker if it existed, None otherwise.
        """
        if marker_type in self.markers:
            marker = self.markers[marker_type]
            marker.remove()
            return marker
        return None

    def get_markers_on_hero(self, hero_id: str) -> list[Marker]:
        """Get all markers currently placed on a specific hero."""
        return [m for m in self.markers.values() if m.target_id == hero_id]

    def return_all_markers(self) -> None:
        """Return all markers to supply. Called at end of round."""
        for marker in self.markers.values():
            marker.remove()

    def return_markers_from_hero(self, hero_id: str) -> list[Marker]:
        """
        Return all markers from a specific hero (e.g., on defeat).
        Markers attached to any of the hero's pieces are returned too.
        Returns list of markers that were removed.
        """
        removed = []
        for marker in self.markers.values():
            if marker.target_id and self.marker_target_belongs_to_hero(marker.target_id, hero_id):
                marker.remove()
                removed.append(marker)
        return removed

    def hero_owner_id(self, entity_id: str) -> str:
        """Player-level identity for an entity ID.

        Hero-piece IDs normalize to their owning hero; hero IDs and anything
        else (minions, tokens, "system") pass through unchanged. Use this for
        "is this you?" comparisons — a multi-piece hero and her pieces are the
        same player identity.
        """
        owner = self.get_hero(HeroID(str(entity_id)))
        return str(owner.id) if owner else str(entity_id)

    def marker_target_belongs_to_hero(self, marker_target: str, hero_id: str) -> bool:
        """True if a marker target (hero ID or piece ID) belongs to hero_id."""
        return self.hero_owner_id(marker_target) == str(hero_id)

    # NOTE: there is deliberately no `return_markers_by_source`. A marker does
    # not leave play because the hero who placed it was defeated — it returns to
    # supply at end of round, or when the hero it is ON is defeated
    # (`return_markers_from_hero`). Bain's Bounty keeps imposing its extra life
    # loss after Bain is gone.

    @property
    def validator(self) -> Any:
        """Lazy-loaded validation service."""
        if self._validator is None:
            from goa2.engine.validation import ValidationService

            self._validator = ValidationService()
        return self._validator

    def create_entity_id(self, prefix: str) -> str:
        """
        Generates a guaranteed unique ID for a new entity.
        Format: {prefix}_{counter}
        """
        new_id = f"{prefix}_{self.next_entity_id}"
        self.next_entity_id += 1
        return new_id

    def register_entity(self, entity: Any, collection_type: str = "token"):
        """
        Registers a new entity into the state (misc_entities or team rosters).
        Enforces Global Uniqueness check.
        """
        if self.get_entity(entity.id) is not None:
            raise ValueError(f"ID Collision: Entity with ID {entity.id} already exists!")

        if collection_type in {"token", "misc", "board_entity"}:
            self.misc_entities[entity.id] = entity
        elif collection_type == "minion":
            if not hasattr(entity, "team") or entity.team not in self.teams:
                raise ValueError(f"Cannot register minion {entity.id}: Invalid or missing team.")
            self.teams[entity.team].minions.append(entity)
            logger.debug("Registered minion %s to team %s", entity.id, entity.team)
        elif collection_type == "hero":
            if not hasattr(entity, "team") or entity.team not in self.teams:
                raise ValueError(f"Cannot register hero {entity.id}: Invalid or missing team.")
            self.teams[entity.team].heroes.append(entity)
            logger.debug("Registered hero %s to team %s", entity.id, entity.team)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_lane_fields(cls, data: Any) -> Any:
        """Accept the legacy single-lane shape from old saves and old call
        sites: `active_zone_id: str` -> `battle_zones` and
        `wave_counter: int` -> `wave_counters`."""
        if isinstance(data, dict):
            legacy_zone = data.pop("active_zone_id", None)
            if legacy_zone and not data.get("battle_zones"):
                data["battle_zones"] = {DEFAULT_LANE_ID: legacy_zone}
            legacy_waves = data.pop("wave_counter", None)
            if legacy_waves is not None and "wave_counters" not in data:
                data["wave_counters"] = {DEFAULT_LANE_ID: legacy_waves}
        return data

    # --- Lane helpers -----------------------------------------------------
    # Engine/effect code should use these (or `battle_zones` directly) rather
    # than the legacy single-lane properties below.

    def battle_zone_ids(self) -> set[str]:
        """Zone ids of all current Battle Zones (one per lane)."""
        return set(self.battle_zones.values())

    def battle_zone_for_lane(self, lane_id: str) -> str | None:
        """Current Battle Zone of the given lane, or None."""
        return self.battle_zones.get(lane_id)

    def lane_of_zone(self, zone_id: str) -> str | None:
        """Lane whose current Battle Zone (or lane sequence) contains zone_id."""
        for lane_id, active in self.battle_zones.items():
            if active == zone_id:
                return lane_id
        return self.board.lane_of_zone(zone_id)

    @property
    def active_zone_id(self) -> str | None:
        """Legacy single-lane accessor for the current Battle Zone.

        Raises on multi-lane games so a missed call site fails loudly instead
        of silently operating on the wrong lane.
        """
        if len(self.battle_zones) > 1:
            raise RuntimeError(
                "GameState.active_zone_id is single-lane only; this game has multiple "
                "lanes. Use battle_zones / battle_zone_for_lane() instead."
            )
        return next(iter(self.battle_zones.values()), None)

    @active_zone_id.setter
    def active_zone_id(self, value: str | None) -> None:
        if len(self.battle_zones) > 1:
            raise RuntimeError(
                "GameState.active_zone_id is single-lane only; this game has multiple "
                "lanes. Write battle_zones[lane_id] instead."
            )
        self.battle_zones = {DEFAULT_LANE_ID: value} if value else {}

    @property
    def wave_counter(self) -> int:
        """Legacy single-lane accessor for the Wave counter (raises on multi-lane)."""
        if len(self.wave_counters) > 1:
            raise RuntimeError(
                "GameState.wave_counter is single-lane only; this game has multiple "
                "lanes. Use wave_counters[lane_id] instead."
            )
        return next(iter(self.wave_counters.values()), 0)

    @wave_counter.setter
    def wave_counter(self, value: int) -> None:
        if len(self.wave_counters) > 1:
            raise RuntimeError(
                "GameState.wave_counter is single-lane only; this game has multiple "
                "lanes. Write wave_counters[lane_id] instead."
            )
        self.wave_counters = {DEFAULT_LANE_ID: value}

    @model_validator(mode="after")
    def unify_token_references(self) -> GameState:
        """
        Re-link `misc_entities` token entries to the same object held in
        `token_pool`.

        At setup a Token is stored in BOTH `misc_entities` (the lookup registry
        used by `get_entity`) and `token_pool` (the supply). In a live session
        these are the same object, so mutations made during placement (owner_id,
        is_immune_to_enemy_actions, ...) are visible through `get_entity`.

        After a JSON reload they deserialize into two SEPARATE objects, and
        placement only mutates the `token_pool` copy. That left `get_entity`
        returning a stale token — e.g. an enemy-immune Totem appearing clearable
        to enemies. Unify them here so `token_pool` (where placement writes)
        stays authoritative.
        """
        for tokens in self.token_pool.values():
            for token in tokens:
                self.misc_entities[token.id] = token
        return self

    @model_validator(mode="after")
    def unify_card_references(self) -> GameState:
        """Restore the shared card identity used by the live lifecycle.

        ``Hero.deck`` is the master card collection, while hand/current/played/
        discard and the planning buffers hold references into it. JSON cannot
        preserve those aliases, so a reload otherwise creates independent Card
        objects whose state changes drift apart. An active lifecycle copy is
        authoritative because it contains the newest state when loading an
        already-split save produced by an older engine version.
        """
        deck_cards_by_hero: dict[str, dict[str, Card]] = {}

        for team in self.teams.values():
            for hero in team.heroes:
                deck_ids = {card.id for card in hero.deck}
                active_cards = [
                    hero.current_turn_card,
                    hero.extra_turn_card,
                    *hero.hand,
                    *hero.played_cards,
                    *hero.discard_pile,
                ]
                active_by_id: dict[str, Card] = {}
                for card in active_cards:
                    if card is not None and card.id in deck_ids:
                        active_by_id.setdefault(card.id, card)

                canonical = {card.id: active_by_id.get(card.id, card) for card in hero.deck}
                hero.deck = [canonical[card.id] for card in hero.deck]
                hero.hand = [canonical.get(card.id, card) for card in hero.hand]
                hero.played_cards = [
                    canonical.get(card.id, card) if card is not None else None
                    for card in hero.played_cards
                ]
                hero.discard_pile = [canonical.get(card.id, card) for card in hero.discard_pile]
                if hero.current_turn_card is not None:
                    hero.current_turn_card = canonical.get(
                        hero.current_turn_card.id, hero.current_turn_card
                    )
                if hero.extra_turn_card is not None:
                    hero.extra_turn_card = canonical.get(
                        hero.extra_turn_card.id, hero.extra_turn_card
                    )
                deck_cards_by_hero[str(hero.id)] = canonical

        for hero_id, card in list(self.pending_inputs.items()):
            if card is not None:
                self.pending_inputs[hero_id] = deck_cards_by_hero.get(str(hero_id), {}).get(
                    card.id, card
                )
        for hero_id, card in list(self.pending_second_cards.items()):
            self.pending_second_cards[hero_id] = deck_cards_by_hero.get(str(hero_id), {}).get(
                card.id, card
            )
        return self

    @model_validator(mode="after")
    def rebuild_occupancy_cache(self) -> GameState:
        """
        Synchronizes board.tiles.occupant_id based on entity_locations.
        This ensures that entity_locations is the Single Source of Truth for persistence.
        """
        # 1. Clear existing occupancy on board tiles
        for tile in self.board.tiles.values():
            tile.occupant_id = None

        # 2. Re-populate based on entity_locations
        from goa2.domain.types import BoardEntityID

        for uid, location in self.entity_locations.items():
            if location in self.board.tiles:
                self.board.tiles[location].occupant_id = BoardEntityID(str(uid))

        return self

    @property
    def is_game_over(self) -> bool:
        """Returns True if the game has ended."""
        return self.phase == GamePhase.GAME_OVER

    @property
    def unit_locations(self) -> dict[UnitID, Hex]:
        """Legacy accessor for backwards compatibility during refactor."""
        # Convert keys to UnitID
        return {UnitID(str(k)): v for k, v in self.entity_locations.items()}

    @property
    def awaiting_input_type(self) -> InputRequestType:
        """
        Helper to get the current expected input type from the top of the stack.
        Returns NONE if stack is empty.
        """
        if not self.input_stack:
            return InputRequestType.NONE
        return self.input_stack[-1].request_type

    def get_hero(self, hero_id: HeroID) -> Hero | None:
        """Finds a Hero by ID. A HeroPiece ID resolves to its owning Hero."""
        for team in self.teams.values():
            for hero in team.heroes:
                if hero.id == hero_id:
                    return hero
        # Piece IDs resolve to the player-level Hero that owns them.
        from goa2.domain.models.unit import HeroPiece

        entity = self.misc_entities.get(BoardEntityID(str(hero_id)))
        if isinstance(entity, HeroPiece):
            return self.get_hero(HeroID(entity.owner_hero_id))
        return None

    def get_entity(self, entity_id: BoardEntityID) -> Any | None:
        """
        Unified lookup for ANY entity on the board (Unit or Token).
        """
        # 1. Check Misc Entities
        if entity_id in self.misc_entities:
            return self.misc_entities[entity_id]

        # 2. Check Units
        return self.get_unit(UnitID(str(entity_id)))

    def get_unit(self, unit_id: UnitID) -> Unit | None:
        """
        Finds a Unit (Hero, Minion, or HeroPiece) by ID.
        O(N) search across all teams, then misc-entity units (hero pieces).
        """
        for team in self.teams.values():
            for hero in team.heroes:
                if str(hero.id) == str(unit_id):
                    return hero
            for minion in team.minions:
                if str(minion.id) == str(unit_id):
                    return minion
        entity = self.misc_entities.get(BoardEntityID(str(unit_id)))
        if isinstance(entity, Unit):
            return entity
        return None

    def get_card_by_id(self, card_id: str) -> Card | None:
        """
        Finds a Card by ID across all heroes.
        Searches normal card zones, spell cards, and ultimate cards.
        """
        for team in self.teams.values():
            for hero in team.heroes:
                if hero.current_turn_card and hero.current_turn_card.id == card_id:
                    return hero.current_turn_card
                for card in hero.played_cards:
                    if card and card.id == card_id:
                        return card
                for card in hero.hand:
                    if card.id == card_id:
                        return card
                for card in hero.deck:
                    if card.id == card_id:
                        return card
                for card in hero.discard_pile:
                    if card.id == card_id:
                        return card
                for spell in hero.spells:
                    if spell.id == card_id:
                        return spell
                if hero.ultimate_card and hero.ultimate_card.id == card_id:
                    return hero.ultimate_card
        return None

    def get_card_for_hero(self, hero_id: str, card_id: str) -> Card | None:
        """Find a normal or spell card belonging to one hero."""
        hero = self.get_hero(HeroID(hero_id))
        if hero is None:
            return None
        candidates = [
            hero.current_turn_card,
            hero.extra_turn_card,
            *hero.played_cards,
            *hero.hand,
            *hero.deck,
            *hero.discard_pile,
            *hero.spells,
            hero.ultimate_card,
        ]
        return next(
            (card for card in candidates if card is not None and card.id == card_id),
            None,
        )

    def get_performing_card(self, hero_id: str | None = None) -> Card | None:
        """Resolve the nested action's source card, then the ordinary turn card."""
        card_id = self.execution_context.get("performing_card_id")
        owner_id = self.execution_context.get("performing_card_owner_id")
        if card_id:
            if owner_id:
                owned = self.get_card_for_hero(str(owner_id), str(card_id))
                if owned is not None:
                    return owned
            matched = self.get_card_by_id(str(card_id))
            if matched is not None:
                return matched
        resolved_hero_id = hero_id or (
            str(self.current_actor_id) if self.current_actor_id is not None else None
        )
        if not resolved_hero_id:
            return None
        hero = self.get_hero(HeroID(resolved_hero_id))
        return hero.current_turn_card if hero is not None else None

    def get_spellbook_owner(self) -> Hero | None:
        """Return the game's unique hero with spell cards, if one exists."""
        owners = [hero for team in self.teams.values() for hero in team.heroes if hero.spells]
        if len(owners) > 1:
            raise ValueError(f"Expected exactly one spellbook owner; found {len(owners)}.")
        return owners[0] if owners else None

    def get_units_and_tokens(self) -> list[BoardEntityID]:
        """
        Returns IDs of all Units and Tokens currently on the board.

        This explicitly filters for Unit and Token types only, excluding
        any future board entity types (e.g., Structures, Hazards).

        Used by SelectStep for UNIT_OR_TOKEN target type.
        """
        from goa2.domain.models import Token, Unit

        result = []
        for eid in self.entity_locations:
            entity = self.get_entity(eid)
            if entity and isinstance(entity, (Unit, Token)):
                result.append(eid)
        return result

    def _multi_piece_hero(self, entity_id: str) -> Hero | None:
        """Return the Hero if entity_id names a multi-piece hero, else None."""
        for team in self.teams.values():
            for hero in team.heroes:
                if str(hero.id) == str(entity_id) and hero.is_multi_piece:
                    return hero
        return None

    def get_piece_ids(self, hero_id: str) -> list[str]:
        """On-board piece IDs for a hero. Normal on-board hero → [hero_id]."""
        from goa2.engine.hero_pieces import piece_id as _piece_id

        hero = self._multi_piece_hero(hero_id)
        if hero is None:
            if BoardEntityID(str(hero_id)) in self.entity_locations:
                return [str(hero_id)]
            return []
        return [
            _piece_id(str(hero.id), i)
            for i in range(1, hero.piece_supply + 1)
            if BoardEntityID(_piece_id(str(hero.id), i)) in self.entity_locations
        ]

    def get_positions(self, entity_id: str) -> list[Hex]:
        """All board positions for an entity. Multi-piece hero → all piece hexes."""
        if self._multi_piece_hero(entity_id) is not None:
            return [
                self.entity_locations[BoardEntityID(pid)] for pid in self.get_piece_ids(entity_id)
            ]
        loc = self.entity_locations.get(BoardEntityID(str(entity_id)))
        return [loc] if loc else []

    def get_position(self, entity_id: str) -> Hex | None:
        """Single position — bound contexts only.

        Multi-piece hero IDs resolve through acting_piece_id (None if unbound).
        Everything else is a direct entity_locations lookup.
        """
        direct = self.entity_locations.get(BoardEntityID(str(entity_id)))
        if direct is not None:
            return direct
        hero = self._multi_piece_hero(entity_id)
        if hero is not None and self.acting_piece_id:
            piece = self.misc_entities.get(self.acting_piece_id)
            if piece is not None and getattr(piece, "owner_hero_id", None) == str(hero.id):
                return self.entity_locations.get(self.acting_piece_id)
        return None

    def has_board_presence(self, hero_id: str) -> bool:
        """True if the hero (or any of its pieces) is on the board."""
        return bool(self.get_positions(hero_id))

    def resolve_board_actor(self, unit_id: str) -> str:
        """Board entity that physically performs an action for unit_id.

        Multi-piece hero with a bound acting piece → the piece ID; else identity.
        """
        if self._multi_piece_hero(unit_id) is not None and self.acting_piece_id:
            piece = self.misc_entities.get(self.acting_piece_id)
            if piece is not None and getattr(piece, "owner_hero_id", None) == str(unit_id):
                return str(self.acting_piece_id)
        return str(unit_id)

    def place_entity(self, entity_id: BoardEntityID, target_hex: Hex):
        """
        Primary Primitive for putting something on the map.
        Updates Location Record AND Tile Cache.
        Overwrites any existing position.
        """
        # Ad-hoc boards (map_id="") intentionally support virtual coordinates
        # for isolated rules/effect scenarios. Boards loaded from a map file
        # must never accept a virtual tile returned by Board.get_tile().
        has_loaded_map = bool(self.board.map_id)
        if has_loaded_map and not self.board.is_on_map(target_hex):
            raise ValueError(
                f"Cannot place entity {entity_id} at {target_hex}: Hex is not on the board"
            )

        target_tile = (
            self.board.tiles[target_hex] if has_loaded_map else self.board.get_tile(target_hex)
        )
        if target_tile.occupant_id and str(target_tile.occupant_id) != str(entity_id):
            raise ValueError(
                f"Cannot place entity {entity_id} at {target_hex}: "
                f"Tile is occupied by {target_tile.occupant_id}"
            )

        # Validate the destination before mutating either source of truth.
        old_hex = self.entity_locations.get(entity_id)
        if old_hex:
            old_tile = self.board.get_tile(old_hex)
            if old_tile and old_tile.occupant_id and str(old_tile.occupant_id) == str(entity_id):
                old_tile.occupant_id = None

        self.entity_locations[entity_id] = target_hex
        target_tile.occupant_id = entity_id

    def remove_entity(self, entity_id: BoardEntityID):
        """
        Removes an entity from the board (locations and tiles).
        Does NOT remove it from Teams or Misc Registry.
        """
        if entity_id in self.entity_locations:
            loc = self.entity_locations[entity_id]
            del self.entity_locations[entity_id]

            tile = self.board.get_tile(loc)
            if tile and tile.occupant_id and str(tile.occupant_id) == str(entity_id):
                tile.occupant_id = None

    def move_unit(self, unit_id: UnitID, target_hex: Hex):
        """Wrapper for place_entity."""
        from goa2.domain.types import BoardEntityID

        self.place_entity(BoardEntityID(str(unit_id)), target_hex)

    def remove_unit(self, unit_id: UnitID):
        """Wrapper for remove_entity."""
        from goa2.domain.types import BoardEntityID

        self.remove_entity(BoardEntityID(str(unit_id)))

    model_config = ConfigDict(arbitrary_types_allowed=True)


GameState.model_rebuild()
