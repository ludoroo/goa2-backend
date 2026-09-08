import hashlib
import logging
import random

from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.factory import EntityFactory
from goa2.domain.models import (
    TOKEN_SUPPLY,
    CardState,
    CardTier,
    GamePhase,
    GameType,
    Team,
    TeamColor,
    Token,
    TokenType,
)
from goa2.domain.models.spawn import SpawnType
from goa2.domain.state import GameState
from goa2.domain.time_control import TimeControlConfig, create_game_clock
from goa2.domain.types import BoardEntityID
from goa2.engine.map_loader import load_map

logger = logging.getLogger(__name__)


def _normalize_game_type(game_type: str | GameType) -> GameType:
    if isinstance(game_type, GameType):
        return game_type
    try:
        return GameType(game_type)
    except ValueError as exc:
        raise ValueError(f"Invalid game_type '{game_type}'. Must be QUICK or LONG.") from exc


class GameSetup:
    """
    Orchestrates the initialization of a new game.
    """

    @staticmethod
    def get_game_config(
        game_type: str | GameType, total_players: int, lane_count: int = 1
    ) -> tuple[int, int]:
        """
        Returns (wave_counters_per_lane, life_counters) for the given game
        type, player count, and lane count. Uneven player counts use the next
        configured player-count bracket.

        Rules:
          Single lane:
            Quick Game: 3 waves, 4 LC (4p) / 4 LC (5p) / 5 LC (6p)
            Long Game:  5 waves, 6 LC (4-5p) / 8 LC (6p)
          Two lanes (no QUICK/LONG split):
            2 x 7 waves, 6 LC (6-8p) / 7 LC (9-10p)
        """
        gt = _normalize_game_type(game_type)

        if lane_count == 2:
            lc_lookup = {8: 6, 10: 7}
            waves = 7
        elif lane_count != 1:
            raise ValueError(f"Unsupported lane count {lane_count}. Supported: 1 or 2.")
        elif gt == GameType.QUICK:
            lc_lookup = {2: 3, 4: 4, 5: 4, 6: 5}
            waves = 3
        else:
            lc_lookup = {2: 6, 4: 6, 5: 6, 6: 8}
            waves = 5

        life_bracket = next(
            (player_count for player_count in sorted(lc_lookup) if player_count >= total_players),
            None,
        )
        if life_bracket is None:
            raise ValueError(
                f"Unsupported player count {total_players} for {gt.value} game. "
                f"Supported: {sorted(lc_lookup.keys())}."
            )
        life_counters = lc_lookup[life_bracket]

        return waves, life_counters

    @staticmethod
    def create_game(
        map_path: str,
        red_heroes: list[str],
        blue_heroes: list[str],
        cheats_enabled: bool = False,
        game_type: str = "LONG",
        seed: int | None = None,
        time_control: TimeControlConfig | None = None,
        tie_breaker_team: TeamColor | None = None,
    ) -> GameState:
        """
        Initializes a game with the specified map and heroes.
        :param map_path: Path to the JSON map file.
        :param red_heroes: List of Hero Names for Red Team.
        :param blue_heroes: List of Hero Names for Blue Team.
        :param cheats_enabled: Whether cheats are enabled for this game.
        :param game_type: "QUICK" or "LONG" (default: "LONG").
        :param seed: Optional deterministic RNG seed. If omitted, a stable seed is
            derived from the setup inputs.
        :param tie_breaker_team: Result of a coin flip already held elsewhere (a
            draft lobby's, say). If omitted, the coin is flipped here from the seed.
        """

        normalized_game_type = _normalize_game_type(game_type)

        # 1. Load Map
        board = load_map(map_path)

        # 2. Calculate Wave & Life Counters
        total_players = len(red_heroes) + len(blue_heroes)
        waves_per_lane, life_counters = GameSetup.get_game_config(
            normalized_game_type, total_players, lane_count=len(board.lanes)
        )

        # 3. Initialize State
        state = GameState(
            board=board,
            teams={
                TeamColor.RED: Team(
                    color=TeamColor.RED,
                    heroes=[],
                    minions=[],
                    life_counters=life_counters,
                ),
                TeamColor.BLUE: Team(
                    color=TeamColor.BLUE,
                    heroes=[],
                    minions=[],
                    life_counters=life_counters,
                ),
            },
            phase=GamePhase.SETUP,
            game_type=normalized_game_type,
            cheats_enabled=cheats_enabled,
            time_control=time_control,
        )

        # Derive the deterministic game seed before creating hidden supplies.
        # Mine subtype assignment uses a separate seed-derived stream so it is
        # replayable without affecting the public tie-breaker coin flip.
        if seed is None:
            seed_material = "|".join(
                [
                    map_path,
                    ",".join(red_heroes),
                    ",".join(blue_heroes),
                    str(cheats_enabled),
                    game_type,
                ]
            )
            seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
        state.rng_seed = seed

        # 4. Determine Battle Zone & Wave counters per lane
        if not board.lanes:
            raise ValueError("Map does not have a defined lane!")

        state.battle_zones = {}
        state.wave_counters = {}
        for lane_id, lane in board.lanes.items():
            # Map-configured starting Battle Zone wins; otherwise the lane center.
            configured = board.starting_battle_zones.get(lane_id)
            if configured and configured in lane:
                state.battle_zones[lane_id] = configured
            else:
                mid_index = len(lane) // 2
                state.battle_zones[lane_id] = lane[mid_index]
            state.wave_counters[lane_id] = waves_per_lane

        # 5. Register Heroes & Place them
        GameSetup._setup_team(state, TeamColor.RED, red_heroes)
        GameSetup._setup_team(state, TeamColor.BLUE, blue_heroes)

        # 6. Build token pool
        GameSetup._initialize_token_pool(state)

        # 7. Spawn Initial Minions (In each lane's Battle Zone)
        for lane_id, zone_id in state.battle_zones.items():
            GameSetup._spawn_initial_minions(state, zone_id, lane_id)

        # 8. Finalize Setup
        # Flip Coin
        rng = random.Random(seed)
        state.tie_breaker_team = tie_breaker_team or rng.choice([TeamColor.RED, TeamColor.BLUE])

        # Transition to Planning. Initial setup counts as the first position
        # snapshot, so turn-1 effects have a defined "last turn" to read.
        from goa2.engine.phases import record_position_snapshot

        record_position_snapshot(state)
        state.phase = GamePhase.PLANNING
        if time_control is not None:
            hero_ids = [str(hero.id) for team in state.teams.values() for hero in team.heroes]
            state.clock = create_game_clock(
                time_control,
                hero_ids,
                round_number=state.round,
                turn_number=state.turn,
            )

        logger.info(
            "Game created (%s). Map: %s tiles. Players: %s. Waves: %s. Life: %s.",
            game_type,
            len(board.tiles),
            total_players,
            waves_per_lane,
            life_counters,
        )
        logger.info(
            "Battle zones: %s. Coin favors: %s",
            state.battle_zones,
            state.tie_breaker_team.name,
        )

        return state

    @staticmethod
    def _initialize_token_pool(state: GameState):
        state.token_pool = {token_type: [] for token_type in TokenType}

        for token_type in TokenType:
            if token_type == TokenType.MINE_DUD:
                # Both mine subtypes are allocated together below so their
                # generic IDs do not reveal a stable blast/dud boundary.
                continue

            if token_type == TokenType.MINE_BLAST:
                mine_types = [
                    *([TokenType.MINE_BLAST] * TOKEN_SUPPLY[TokenType.MINE_BLAST]),
                    *([TokenType.MINE_DUD] * TOKEN_SUPPLY[TokenType.MINE_DUD]),
                ]
                mine_rng = random.Random(f"goa2:mine-supply:{state.rng_seed}")
                mine_rng.shuffle(mine_types)
                for mine_type in mine_types:
                    token_id = state.create_entity_id("mine")
                    token = Token(
                        id=BoardEntityID(token_id),
                        name=mine_type.value.replace("_", " ").title(),
                        token_type=mine_type,
                        is_passable=True,
                        is_facedown=True,
                    )
                    state.register_entity(token, "token")
                    state.token_pool[mine_type].append(token)
                continue

            supply = TOKEN_SUPPLY.get(token_type, 0)

            persists_end_of_round = token_type in (
                TokenType.ZOMBIE,
                TokenType.PYRO,
                TokenType.TREE,
            )
            for _ in range(supply):
                token_id = state.create_entity_id(token_type.value)
                token = Token(
                    id=BoardEntityID(token_id),
                    name=token_type.value.replace("_", " ").title(),
                    token_type=token_type,
                    persists_end_of_round=persists_end_of_round,
                )
                state.register_entity(token, "token")
                state.token_pool[token_type].append(token)

    @staticmethod
    def _setup_team(state: GameState, team_color: TeamColor, hero_names: list[str]):
        """
        Instantiates heroes, sets up their hand, and places them at spawn points.
        """
        available_spawns = [
            sp
            for sp in state.board.spawn_points
            if sp.type == SpawnType.HERO and sp.team == team_color
        ]

        for name in hero_names:
            # A. Instantiate
            hero = HeroRegistry.get(name)
            if not hero:
                raise ValueError(f"Hero '{name}' not found in registry.")

            # Ensure unique ID if duplicates allowed (usually not, but safe practice)
            # HeroRegistry returns fixed ID. If duplicate heroes allowed, we'd need dynamic IDs.
            # For now, standard GoA2 assumes unique heroes.
            hero.team = team_color

            # B. Setup Hand (Tier I + Untiered)
            hero.hand = []

            for card in hero.deck:
                # `starts_in_deck` cards (Takahide's Sting/Strike) are Tier I but
                # begin the game in the deck rather than the opening hand.
                if card.tier in [CardTier.I, CardTier.UNTIERED] and not card.starts_in_deck:
                    hero.return_card_to_hand(card)
                else:
                    card.state = CardState.DECK

            # Register to Team and State
            state.register_entity(hero, "hero")

            # C. Place on Board
            # Find an empty spawn point
            spawn_loc = None

            for sp in available_spawns:
                tile = state.board.get_tile(sp.location)
                if tile and not tile.is_occupied:
                    spawn_loc = sp.location
                    break

            if spawn_loc is None:
                # Rulebook (8-10 players): extra heroes are placed in an empty
                # space adjacent to one of their team's occupied hero spawn
                # points. Spawn-point and neighbor order are fixed, so the
                # placement is deterministic.
                spawn_loc = next(
                    (
                        adj
                        for sp in available_spawns
                        if state.board.get_tile(sp.location).is_occupied
                        for adj in sp.location.neighbors()
                        if state.board.is_on_map(adj) and not state.board.get_tile(adj).is_obstacle
                    ),
                    None,
                )

            if spawn_loc:
                if hero.is_multi_piece:
                    # Multi-piece heroes (Razzle) never occupy the board
                    # themselves: register the piece supply and place piece 1.
                    from goa2.engine.hero_pieces import create_hero_pieces, piece_id

                    create_hero_pieces(state, hero)
                    state.place_entity(BoardEntityID(piece_id(str(hero.id), 1)), spawn_loc)
                else:
                    state.place_entity(hero.id, spawn_loc)
            else:
                raise ValueError(
                    f"No hero spawn point available for {hero.name}: the map has "
                    f"too few {team_color.value} hero spawn points for this team size."
                )

    @staticmethod
    def _spawn_initial_minions(state: GameState, zone_id: str, lane_id: str | None = None):
        """
        Spawns minions at all minion spawn points in the target zone,
        binding them to the given lane for respawn purposes.
        """
        zone = state.board.zones.get(zone_id)
        if not zone:
            return

        for h in zone.hexes:
            sp = state.board.get_spawn_point(h)
            if sp and sp.type == SpawnType.MINION and sp.minion_type:
                # Create Minion (bound to its lane)
                minion = EntityFactory.create_minion(state, sp.team, sp.minion_type, lane_id)
                state.register_entity(minion, "minion")

                # Place
                state.place_entity(minion.id, h)
