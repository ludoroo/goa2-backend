"""REST integration tests using FastAPI TestClient."""

import os

import pytest
from fastapi.testclient import TestClient

from goa2.server.app import create_app


@pytest.fixture
def client(tmp_path):
    os.environ["GOA2_SAVE_DIR"] = str(tmp_path)
    app = create_app()
    with TestClient(app) as c:
        yield c
    os.environ.pop("GOA2_SAVE_DIR", None)


@pytest.fixture
def game_data(client):
    """Create a game and return (response_json, client)."""
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def _token_for(game_data: dict, hero_id: str) -> str:
    for pt in game_data["player_tokens"]:
        if pt["hero_id"] == hero_id:
            return pt["token"]
    raise ValueError(f"No token for {hero_id}")


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---- /heroes ----


def test_list_heroes(client):
    resp = client.get("/heroes")
    assert resp.status_code == 200
    heroes = resp.json()
    assert isinstance(heroes, list)
    assert "Arien" in heroes
    assert "Cordelia" not in heroes


def test_list_heroes_can_include_playtest_heroes(client):
    resp = client.get("/heroes", params={"include_playtest": True})

    assert resp.status_code == 200
    assert "Cordelia" in resp.json()


def test_list_hero_metadata_includes_difficulty_stars(client):
    resp = client.get("/heroes/metadata")
    assert resp.status_code == 200
    heroes = {hero["id"]: hero["difficulty_stars"] for hero in resp.json()}

    assert heroes == {
        "Arien": 1,
        "Xargatha": 1,
        "Wasp": 1,
        "Brogan": 1,
        "Tigerclaw": 1,
        "Sabina": 1,
        "Dodger": 1,
        "Bain": 2,
        "Whisper": 2,
        "Rowenna": 2,
        "Ursafar": 2,
        "Min": 2,
        "Misa": 2,
        "Garrus": 2,
        "Silverarrow": 2,
        "Mortimer": 3,
        "Widget": 3,
        "Trinkets": 3,
        "Tali": 3,
        "Wuk": 3,
        "Swift": 3,
        "Mrak": 3,
        "Cutter": 3,
        "Hanu": 3,
        "Brynn": 3,
        "Razzle": 4,
        "Ignatia": 4,
        "Emmitt": 4,
        "Gydion": 4,
        "NebKher": 4,
        "Snorri": 4,
        "Takahide": 4,
    }


def test_list_hero_metadata_can_include_playtest_heroes(client):
    resp = client.get("/heroes/metadata", params={"include_playtest": True})

    assert resp.status_code == 200
    heroes = {hero["id"]: hero["difficulty_stars"] for hero in resp.json()}
    assert heroes["Cordelia"] == 2


# ---- POST /games ----


def test_create_game(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "game_id" in data
    assert len(data["player_tokens"]) == 2
    assert data["spectator_token"]


def test_create_quick_game(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "game_type": "QUICK",
        },
    )
    assert resp.status_code == 201
    data = resp.json()

    token = _token_for(data, "hero_arien")
    view_resp = client.get(f"/games/{data['game_id']}", headers=_auth(token))
    assert view_resp.status_code == 200
    view = view_resp.json()["view"]

    assert view["teams"]["RED"]["life_counters"] == 3
    assert view["teams"]["BLUE"]["life_counters"] == 3


def test_create_uneven_player_game_uses_upper_life_count(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien", "Min"],
            "blue_heroes": ["Wasp"],
            "game_type": "QUICK",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["player_tokens"]) == 3

    token = _token_for(data, "hero_arien")
    view_resp = client.get(f"/games/{data['game_id']}", headers=_auth(token))
    assert view_resp.status_code == 200
    view = view_resp.json()["view"]

    assert view["teams"]["RED"]["life_counters"] == 4
    assert view["teams"]["BLUE"]["life_counters"] == 4


def test_create_game_invalid_game_type(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "game_type": "BLITZ",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid game_type 'BLITZ'. Must be QUICK or LONG."


def test_create_game_bad_map(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "nonexistent_map",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 404


# ---- POST /games with bots ----


_NO_BOTS_REQUEST = {
    "map_name": "forgotten_island",
    "red_heroes": ["Arien"],
    "blue_heroes": ["Wasp"],
}


def _create_game_response_shape(data: dict) -> set[str]:
    return set(data.keys())


def test_create_game_without_bots_shape_unchanged(client):
    """A creation request without a ``bots`` field must produce the same
    top-level response shape as an untouched request (game_id, player_tokens,
    spectator_token) — no extra fields leak in."""
    resp = client.post("/games", json=_NO_BOTS_REQUEST)
    assert resp.status_code == 201
    data = resp.json()
    assert _create_game_response_shape(data) == {
        "game_id",
        "player_tokens",
        "spectator_token",
    }
    # Every hero, bot or not, still gets a player token.
    hero_ids = {pt["hero_id"] for pt in data["player_tokens"]}
    assert hero_ids == {"hero_arien", "hero_wasp"}
    assert data["spectator_token"]


def test_create_game_without_bots_field_registry_has_no_specs(client):
    """No ``bots`` in the request means ``ManagedGame.bot_specs`` stays empty
    — a plain humans-only game must not become bot-driven by omission."""
    resp = client.post("/games", json=_NO_BOTS_REQUEST)
    assert resp.status_code == 201
    game_id = resp.json()["game_id"]
    game = client.app.state.registry.get(game_id)
    assert game.bot_specs == {}


def test_create_game_with_random_bot_response_shape_unchanged(client):
    """Adding a Random bot MUST NOT change response shape or hero token
    semantics: every hero (bot or not) still gets a player token; the
    spectator token still exists; no extra top-level fields appear."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "random"}},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert _create_game_response_shape(data) == {
        "game_id",
        "player_tokens",
        "spectator_token",
    }
    # Bot heroes still receive tokens (see plan §Scope And Invariants).
    hero_ids = {pt["hero_id"] for pt in data["player_tokens"]}
    assert hero_ids == {"hero_arien", "hero_wasp"}


def test_create_game_with_random_bot_persists_spec(client):
    """A Random ``BotSpec`` from the request must land on
    ``ManagedGame.bot_specs`` verbatim."""
    from goa2.server.bot_models import BotSpec

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "random"}},
        },
    )
    assert resp.status_code == 201
    game = client.app.state.registry.get(resp.json()["game_id"])
    assert game.bot_specs == {"hero_wasp": BotSpec(kind="random")}


def test_create_game_with_heuristic_bot_persists_spec(client):
    """Heuristic is a supported production bot kind (plan §Release Gates)."""
    from goa2.server.bot_models import BotSpec

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "heuristic"}},
        },
    )
    assert resp.status_code == 201
    game = client.app.state.registry.get(resp.json()["game_id"])
    assert game.bot_specs == {"hero_wasp": BotSpec(kind="heuristic")}


def test_create_game_with_random_bot_follows_human_commit(client):
    """End-to-end: a Random bot created via the public API should follow a
    human commit and make engine progress (matches the pattern already
    validated in test_server_bots for direct registry injection)."""
    import time

    from goa2.domain.types import HeroID

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "random"}},
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")
    game = client.app.state.registry.get(game_id)

    view = client.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
    arien_hand = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_hand = hero["hand"]
    assert arien_hand
    card_id = arien_hand[0]["id"]
    resp2 = client.post(
        f"/games/{game_id}/cards",
        json={"card_id": card_id},
        headers=_auth(arien_token),
    )
    assert resp2.status_code == 200

    async def _pump_and_check() -> bool:
        import asyncio

        for _ in range(50):
            await asyncio.sleep(0)
        wasp = next(
            h
            for team in game.session.state.teams.values()
            for h in team.heroes
            if h.id == "hero_wasp"
        )
        return (
            HeroID("hero_wasp") in game.session.state.pending_inputs
            or wasp.current_turn_card is not None
            or game.session.state.phase.value != "PLANNING"
        )

    deadline = time.monotonic() + 5.0
    bot_done = False
    while time.monotonic() < deadline:
        if client.portal.call(_pump_and_check):
            bot_done = True
            break
        time.sleep(0.05)
    assert bot_done, "random bot from public API must progress after human commit"


def test_create_game_accepts_ismcts_bot(client):
    """:class:`CreateBotSpec` accepts ``kind='ismcts'``.
    The public schema advertises ``ismcts`` as a supported kind, and
    the route persists the spec on ``ManagedGame.bot_specs``. Runtime
    bounds (semaphore, queue timeout, search timeout, heuristic fallback)
    live in the coordinator — the request boundary only validates the
    declared settings."""
    from goa2.server.bot_models import BotSpec, SearchSettings

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "ismcts"}},
        },
    )
    assert resp.status_code == 201, resp.text
    game_id = resp.json()["game_id"]
    game = client.app.state.registry.get(game_id)
    # No ``search`` supplied → the spec carries ``None`` and the
    # coordinator applies the SearchSettings defaults at agent-build time.
    assert game.bot_specs == {"hero_wasp": BotSpec(kind="ismcts")}
    # Sanity: the persisted defaults are within the production bounds.
    _sanity = SearchSettings()
    assert _sanity.iterations > 0


def test_create_game_accepts_ismcts_bot_with_search_settings(client):
    """A well-formed ISMCTS spec with bounded search settings persists
    verbatim. Kind + search field are the whole public contract."""
    from goa2.server.bot_models import BotSpec, SearchSettings

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "ismcts",
                    "search": {"iterations": 50, "decision_timeout_seconds": 0.5},
                }
            },
        },
    )
    assert resp.status_code == 201, resp.text
    game_id = resp.json()["game_id"]
    game = client.app.state.registry.get(game_id)
    assert game.bot_specs == {
        "hero_wasp": BotSpec(
            kind="ismcts",
            search=SearchSettings(iterations=50, decision_timeout_seconds=0.5),
        )
    }


def test_create_game_rejects_ismcts_iterations_over_max(client):
    """``iterations`` is bounded at :data:`PROD_MAX_ITERATIONS`. A
    request that exceeds the cap must fail with a 422 — this is the
    upper-bound validation exercised at the request boundary."""
    from automata.search.config import PROD_MAX_ITERATIONS

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "ismcts",
                    "search": {
                        "iterations": PROD_MAX_ITERATIONS + 1,
                        "decision_timeout_seconds": 1.0,
                    },
                }
            },
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_game_rejects_ismcts_decision_timeout_over_max(client):
    """``decision_timeout_seconds`` is bounded at
    :data:`PROD_MAX_DECISION_TIMEOUT_SECONDS`."""
    from automata.search.config import PROD_MAX_DECISION_TIMEOUT_SECONDS

    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "ismcts",
                    "search": {
                        "iterations": 10,
                        "decision_timeout_seconds": PROD_MAX_DECISION_TIMEOUT_SECONDS
                        + 1.0,
                    },
                }
            },
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_game_rejects_ismcts_iterations_below_min(client):
    """Zero and negative iteration budgets are rejected as the lower-bound
    check the plan requires."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "ismcts",
                    "search": {"iterations": 0, "decision_timeout_seconds": 1.0},
                }
            },
        },
    )
    assert resp.status_code == 422, resp.text


def test_create_game_rejects_search_settings_for_non_ismcts(client):
    """``search`` is only valid on ``kind='ismcts'``. Supplying it
    on Random or Heuristic bot specs is a 422 (the model_validator on
    :class:`CreateBotSpec` enforces this before persistence)."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "random",
                    "search": {"iterations": 10, "decision_timeout_seconds": 0.5},
                }
            },
        },
    )
    assert resp.status_code == 422

    resp2 = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {
                "hero_wasp": {
                    "kind": "heuristic",
                    "search": {"iterations": 10, "decision_timeout_seconds": 0.5},
                }
            },
        },
    )
    assert resp2.status_code == 422


def test_create_game_rejects_unknown_bot_kind(client):
    """Unknown agent kinds are rejected by BotSpec's Literal — 422 from
    Pydantic before any registry work."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "definitely_not_a_kind"}},
        },
    )
    assert resp.status_code == 422


def test_create_game_rejects_extra_fields_on_bot_spec(client):
    """BotSpec has ``extra='forbid'``; unknown fields → 422."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_wasp": {"kind": "random", "epsilon": 0.1}},
        },
    )
    assert resp.status_code == 422


def test_create_game_rejects_hero_not_in_roster(client):
    """A bot spec for a hero not on either team must be rejected — the
    registry cannot persist a spec that no session hero controls."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            # Bain is a real hero, but not on this game's teams.
            "bots": {"hero_bain": {"kind": "random"}},
        },
    )
    assert 400 <= resp.status_code < 500
    body = resp.json()
    detail = body.get("detail")
    text = detail if isinstance(detail, str) else str(body)
    assert "hero_bain" in text or "roster" in text.lower()


def test_create_game_rejects_completely_unknown_hero_id(client):
    """A totally fabricated hero id must be rejected — not silently ignored."""
    resp = client.post(
        "/games",
        json={
            **_NO_BOTS_REQUEST,
            "bots": {"hero_nonexistent_qqq": {"kind": "random"}},
        },
    )
    assert 400 <= resp.status_code < 500


def test_create_game_with_bots_still_yields_player_and_spectator_tokens(client):
    """Every roster hero (bot or human) still gets a player token and a
    spectator token is issued. This is the core token-parity contract the
    bots feature must preserve."""
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien", "Min"],
            "blue_heroes": ["Wasp", "Brogan"],
            "bots": {
                "hero_wasp": {"kind": "random"},
                "hero_brogan": {"kind": "heuristic"},
            },
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    hero_ids = {pt["hero_id"] for pt in data["player_tokens"]}
    assert hero_ids == {"hero_arien", "hero_min", "hero_wasp", "hero_brogan"}
    for pt in data["player_tokens"]:
        assert pt["token"], f"missing token for {pt['hero_id']}"
    assert data["spectator_token"]


def test_create_game_with_empty_bots_mapping(client):
    """An empty ``bots`` mapping is equivalent to omitting the field — the
    game is fully human-controlled and no bot state is set up."""
    resp = client.post(
        "/games",
        json={**_NO_BOTS_REQUEST, "bots": {}},
    )
    assert resp.status_code == 201
    game_id = resp.json()["game_id"]
    game = client.app.state.registry.get(game_id)
    assert game.bot_specs == {}


def test_create_draft_rejects_top_level_bots_key(client):
    """Draft-created games do not support bot configuration.
    ``POST /drafts`` must **reject** a top-level ``bots`` key with a clear
    4xx — silent ``extra='ignore'`` was a footgun (client thinks bots
    landed; they did not). The rejection is targeted: unrelated unknown
    fields (forward-compat additions) must continue to be ignored per the
    existing draft policy."""
    resp = client.post(
        "/drafts",
        json={
            "host_name": "Alice",
            "map_name": "forgotten_island",
            "bots": {"hero_wasp": {"kind": "random"}},
        },
    )
    # One rejection outcome (422 from the model_validator).
    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail = body.get("detail")
    # FastAPI serializes model_validator failures into detail list.
    text = detail if isinstance(detail, str) else str(body)
    assert "bots" in text.lower() or "not supported" in text.lower(), body


def test_update_draft_settings_rejects_top_level_bots_key(client):
    """The same targeted rejection applies to draft settings updates —
    a host cannot add bot configuration to an existing lobby either."""
    # Create a lobby first.
    create = client.post(
        "/drafts",
        json={"host_name": "Alice", "map_name": "forgotten_island"},
    )
    assert create.status_code == 201, create.text
    data = create.json()
    draft_id = data["draft_id"]
    token = data["player_token"]
    resp = client.patch(
        f"/drafts/{draft_id}/settings",
        json={"bots": {"hero_wasp": {"kind": "random"}}},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    text = str(body)
    assert "bots" in text.lower() or "not supported" in text.lower(), body


def test_create_draft_still_ignores_unrelated_unknown_fields(client):
    """Rejection of ``bots`` must not turn on strict ``extra='forbid'`` for
    the whole draft body. Existing draft policy is ``extra='ignore'`` —
    that must be preserved so a schema-additive client change (say a new
    ``client_hint`` field) does not break older servers."""
    resp = client.post(
        "/drafts",
        json={
            "host_name": "Alice",
            "map_name": "forgotten_island",
            # Deliberately-unknown forward-compat-style field:
            "client_hint": "future-only",
        },
    )
    assert resp.status_code == 201, resp.text


# ---- OpenAPI schema honesty ----


def test_openapi_bot_schema_advertises_public_kinds_and_search(client):
    """The OpenAPI schema advertises ``ismcts`` as a public
    ``kind`` alongside ``random`` and ``heuristic``, and exposes the
    optional ``search`` field on the public ``CreateBotSpec``. Clients
    that read the schema for autocomplete or type-generation should see
    exactly what they may send today."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    components = schema.get("components", {}).get("schemas", {})
    assert "CreateBotSpec" in components, sorted(components.keys())
    bot_schema = components["CreateBotSpec"]
    props = bot_schema.get("properties", {})
    # ``search`` is now part of the public schema (nullable / optional).
    assert "search" in props, props
    # ``kind`` advertises exactly the three supported kinds.
    kind_field = props.get("kind", {})
    enum = kind_field.get("enum")
    if enum is None:
        ref = kind_field.get("$ref")
        assert ref, kind_field
        ref_name = ref.rsplit("/", 1)[-1]
        enum = components.get(ref_name, {}).get("enum")
    assert enum is not None, bot_schema
    assert set(enum) == {"random", "heuristic", "ismcts"}, enum
    # SearchSettings is exposed as a referenced schema so clients can
    # type-generate the bounded fields.
    assert "SearchSettings" in components, sorted(components.keys())
    ss_props = components["SearchSettings"].get("properties", {})
    assert "iterations" in ss_props
    assert "decision_timeout_seconds" in ss_props


def test_create_draft_does_not_accept_bots(client):
    """Backwards-compatible name for the targeted-rejection test; retained
    so an audit script grepping for the old spec name still finds the
    coverage. Delegates to the primary tight test."""
    test_create_draft_rejects_top_level_bots_key(client)


# ---- GET /games/{game_id} ----


def test_get_game_view(client, game_data):
    token = _token_for(game_data, "hero_arien")
    resp = client.get(f"/games/{game_data['game_id']}", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert "view" in body
    assert body["view"]["phase"] == "PLANNING"


def test_get_game_view_spectator(client, game_data):
    resp = client.get(
        f"/games/{game_data['game_id']}",
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 200


def test_get_game_view_no_auth(client, game_data):
    resp = client.get(f"/games/{game_data['game_id']}")
    assert resp.status_code == 401


def test_get_game_view_wrong_game(client, game_data):
    token = _token_for(game_data, "hero_arien")
    resp = client.get("/games/wrong_id", headers=_auth(token))
    assert resp.status_code == 403


# ---- POST /games/{game_id}/cards ----


def test_commit_card(client, game_data):
    """Commit a card during PLANNING."""
    token = _token_for(game_data, "hero_arien")
    game_id = game_data["game_id"]

    # Get view to find a card
    view = client.get(f"/games/{game_id}", headers=_auth(token)).json()
    arien_cards = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_cards = hero["hand"]
                break
    assert arien_cards and len(arien_cards) > 0

    card_id = arien_cards[0]["id"]
    resp = client.post(
        f"/games/{game_id}/cards",
        json={"card_id": card_id},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_type"] in ("ACTION_COMPLETE", "PHASE_CHANGED", "INPUT_NEEDED")


def test_commit_card_bad_id(client, game_data):
    token = _token_for(game_data, "hero_arien")
    resp = client.post(
        f"/games/{game_data['game_id']}/cards",
        json={"card_id": "nonexistent_card"},
        headers=_auth(token),
    )
    assert resp.status_code == 404


def test_commit_card_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/cards",
        json={"card_id": "some_card"},
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


# ---- POST /games/{game_id}/pass ----


def test_pass_turn_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/pass",
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


# ---- POST /games/{game_id}/input ----


def test_submit_input_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/input",
        json={"request_id": "spectator-request", "selection": "x"},
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


# ---- POST /games/{game_id}/advance ----


def test_advance_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/advance",
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


# ---- Full flow: commit both cards -> phase transition ----


def test_full_planning_flow(client, game_data):
    """Both players commit cards, triggering phase transition."""
    game_id = game_data["game_id"]
    arien_token = _token_for(game_data, "hero_arien")
    wasp_token = _token_for(game_data, "hero_wasp")

    # Get Arien's cards
    view = client.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
    arien_hand = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_hand = hero["hand"]
    assert arien_hand

    # Get Wasp's cards
    view = client.get(f"/games/{game_id}", headers=_auth(wasp_token)).json()
    wasp_hand = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_wasp":
                wasp_hand = hero["hand"]
    assert wasp_hand

    # Commit Arien's card
    resp1 = client.post(
        f"/games/{game_id}/cards",
        json={"card_id": arien_hand[0]["id"]},
        headers=_auth(arien_token),
    )
    assert resp1.status_code == 200
    assert resp1.json()["result_type"] == "ACTION_COMPLETE"

    # Commit Wasp's card -> triggers phase transition
    resp2 = client.post(
        f"/games/{game_id}/cards",
        json={"card_id": wasp_hand[0]["id"]},
        headers=_auth(wasp_token),
    )
    assert resp2.status_code == 200
    # After both commit, phase should change
    body = resp2.json()
    assert body["current_phase"] != "PLANNING"


# ---- Cheats ----


def test_create_game_with_cheats_enabled(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()

    # Verify cheats_enabled is in the view
    arien_token = _token_for(data, "hero_arien")
    view_resp = client.get(f"/games/{data['game_id']}", headers=_auth(arien_token))
    assert view_resp.status_code == 200
    view = view_resp.json()
    assert view["view"]["cheats_enabled"] is True


def test_create_game_without_cheats(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()

    # Verify cheats_enabled defaults to False
    arien_token = _token_for(data, "hero_arien")
    view_resp = client.get(f"/games/{data['game_id']}", headers=_auth(arien_token))
    assert view_resp.status_code == 200
    view = view_resp.json()
    assert view["view"]["cheats_enabled"] is False


def test_give_gold_cheat_success(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    # Give gold to Arien
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_arien", "amount": 5},
        headers=_auth(arien_token),
    )
    assert cheat_resp.status_code == 200
    result = cheat_resp.json()
    assert result["result_type"] == "ACTION_COMPLETE"
    assert result["events"]
    assert result["events"][0]["event_type"] == "GOLD_GAINED"
    assert result["events"][0]["metadata"]["amount"] == 5
    assert result["events"][0]["metadata"]["reason"] == "cheat"

    # Verify gold was added
    view_resp = client.get(f"/games/{game_id}", headers=_auth(arien_token))
    view = view_resp.json()
    arien_gold = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien":
                arien_gold = hero["gold"]
    assert arien_gold == 5


def test_give_gold_cheat_disabled(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": False,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    # Try to give gold when cheats are disabled
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_arien", "amount": 5},
        headers=_auth(arien_token),
    )
    assert cheat_resp.status_code == 403
    assert "Cheats are not enabled" in cheat_resp.json()["detail"]


def test_give_gold_cheat_wrong_phase(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")
    wasp_token = _token_for(data, "hero_wasp")

    # Get Arien's hand
    view_resp = client.get(f"/games/{game_id}", headers=_auth(arien_token))
    view = view_resp.json()
    arien_card_id = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_arien" and hero["hand"]:
                arien_card_id = hero["hand"][0]["id"]

    # Get Wasp's hand
    view_resp = client.get(f"/games/{game_id}", headers=_auth(wasp_token))
    view = view_resp.json()
    wasp_card_id = None
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == "hero_wasp" and hero["hand"]:
                wasp_card_id = hero["hand"][0]["id"]

    # Both players commit cards to move to REVELATION phase
    client.post(
        f"/games/{game_id}/cards",
        json={"card_id": arien_card_id},
        headers=_auth(arien_token),
    )
    client.post(
        f"/games/{game_id}/cards",
        json={"card_id": wasp_card_id},
        headers=_auth(wasp_token),
    )

    # Try to give gold in non-PLANNING phase
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_arien", "amount": 5},
        headers=_auth(arien_token),
    )
    assert cheat_resp.status_code == 409
    assert "Expected phase PLANNING" in cheat_resp.json()["detail"]


def test_give_gold_cheat_invalid_hero(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    # Try to give gold to non-existent hero
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_does_not_exist", "amount": 5},
        headers=_auth(arien_token),
    )
    assert cheat_resp.status_code == 404


def test_give_gold_cheat_negative_amount(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    arien_token = _token_for(data, "hero_arien")

    # Try to give negative gold
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_arien", "amount": -5},
        headers=_auth(arien_token),
    )
    assert cheat_resp.status_code == 400
    assert "Amount must be a positive integer" in cheat_resp.json()["detail"]


# ---- POST /games/{game_id}/rollback ----


def _advance_to_resolution(client, game_data):
    """Commit cards for both players to transition to RESOLUTION, return tokens."""
    game_id = game_data["game_id"]
    arien_token = _token_for(game_data, "hero_arien")
    wasp_token = _token_for(game_data, "hero_wasp")

    # Get hands
    view = client.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
    arien_card = None
    for td in view["view"]["teams"].values():
        for h in td["heroes"]:
            if h["id"] == "hero_arien" and h["hand"]:
                arien_card = h["hand"][0]["id"]

    view = client.get(f"/games/{game_id}", headers=_auth(wasp_token)).json()
    wasp_card = None
    for td in view["view"]["teams"].values():
        for h in td["heroes"]:
            if h["id"] == "hero_wasp" and h["hand"]:
                wasp_card = h["hand"][0]["id"]

    # Commit both
    client.post(f"/games/{game_id}/cards", json={"card_id": arien_card}, headers=_auth(arien_token))
    client.post(f"/games/{game_id}/cards", json={"card_id": wasp_card}, headers=_auth(wasp_token))
    return arien_token, wasp_token


def test_rollback_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/rollback",
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


def test_rollback_no_active_resolution(client, game_data):
    """Rollback fails when there's no active resolution."""
    token = _token_for(game_data, "hero_arien")
    resp = client.post(
        f"/games/{game_data['game_id']}/rollback",
        headers=_auth(token),
    )
    assert resp.status_code == 400


def test_rollback_not_current_actor(client, game_data):
    """Only the current actor can rollback."""
    game_id = game_data["game_id"]
    arien_token, wasp_token = _advance_to_resolution(client, game_data)

    # Check who the current actor is from the input_request
    view = client.get(f"/games/{game_id}", headers=_auth(arien_token)).json()
    ir = view.get("input_request")
    if ir:
        current_actor = ir["player_id"]
        non_actor_token = wasp_token if current_actor == "hero_arien" else arien_token
        resp = client.post(
            f"/games/{game_id}/rollback",
            headers=_auth(non_actor_token),
        )
        assert resp.status_code == 403


def test_rollback_authorizes_controller_during_action_control(client, game_data):
    """Under Hanu's ultimate the action's inputs are remapped to the controller,
    who owns the confirm/rollback — not the controlled actor."""
    game_id = game_data["game_id"]
    _advance_to_resolution(client, game_data)

    registry = client.app.state.registry
    game = registry.get(game_id)
    session = game.session
    ir = game.last_result.input_request if game.last_result else None
    if ir is None or session.state.current_actor_id is None:
        pytest.skip("no active resolution input to remap")

    actor = str(session.state.current_actor_id)
    controller = "hero_wasp" if actor == "hero_arien" else "hero_arien"

    # Simulate the handler's control remap: the pending input is addressed to
    # the controller, the original actor is preserved in context, and a snapshot
    # was taken for the actor's action.
    ir.player_id = controller
    ir.context["controlled_hero_id"] = actor
    session._rollback_snapshot = session._make_snapshot()
    session._rollback_actor_id = actor

    # The controlled actor may NOT rollback the controlled action.
    resp_actor = client.post(
        f"/games/{game_id}/rollback", headers=_auth(_token_for(game_data, actor))
    )
    assert resp_actor.status_code == 403

    # The controller MAY rollback.
    resp_ctrl = client.post(
        f"/games/{game_id}/rollback", headers=_auth(_token_for(game_data, controller))
    )
    assert resp_ctrl.status_code == 200


def test_give_gold_cheat_spectator_blocked(client):
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Arien"],
            "blue_heroes": ["Wasp"],
            "cheats_enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game_id = data["game_id"]
    spectator_token = data["spectator_token"]

    # Try to use cheats as spectator
    cheat_resp = client.post(
        f"/games/{game_id}/cheats/gold",
        json={"hero_id": "hero_arien", "amount": 5},
        headers=_auth(spectator_token),
    )
    assert cheat_resp.status_code == 403
    assert "Spectators cannot use cheats" in cheat_resp.json()["detail"]


# ---- Alternative Timelines (Emmitt ultimate) planning flow ----


@pytest.fixture
def emmitt_game(client):
    """Game with Emmitt (RED, level 8 → ultimate active) vs Wasp (BLUE)."""
    resp = client.post(
        "/games",
        json={
            "map_name": "forgotten_island",
            "red_heroes": ["Emmitt"],
            "blue_heroes": ["Wasp"],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    game = client.app.state.registry.get(data["game_id"])
    game.session.state.get_hero("hero_emmitt").level = 8
    return data


def _first_hand_card_id(client, game_data, hero_id):
    token = _token_for(game_data, hero_id)
    view = client.get(f"/games/{game_data['game_id']}", headers=_auth(token)).json()
    for team in view["view"]["teams"].values():
        for hero in team["heroes"]:
            if hero["id"] == hero_id:
                return hero["hand"][0]["id"]
    raise ValueError(f"No hand for {hero_id}")


def test_emmitt_two_card_commit_and_retrieve(client, emmitt_game):
    """Second commit accepted; after all commit, the retrieve prompt goes to
    Emmitt; answering it returns the card to hand and starts resolution."""
    game_id = emmitt_game["game_id"]
    em_token = _token_for(emmitt_game, "hero_emmitt")
    wa_token = _token_for(emmitt_game, "hero_wasp")

    r1 = client.post(
        f"/games/{game_id}/cards", json={"card_id": "reverse_time"}, headers=_auth(em_token)
    )
    assert r1.status_code == 200
    r2 = client.post(
        f"/games/{game_id}/cards", json={"card_id": "unstable_timeline"}, headers=_auth(em_token)
    )
    assert r2.status_code == 200

    wasp_card = _first_hand_card_id(client, emmitt_game, "hero_wasp")
    r3 = client.post(
        f"/games/{game_id}/cards", json={"card_id": wasp_card}, headers=_auth(wa_token)
    )
    assert r3.status_code == 200
    body = r3.json()
    assert body["current_phase"] == "RESOLUTION"
    # The final commit starts an Emmitt-only choice, which must not ride back
    # on Wasp's action response or appear in public/opponent state reads.
    assert body["input_request"] is None
    assert client.get(f"/games/{game_id}", headers=_auth(wa_token)).json()["input_request"] is None
    assert (
        client.get(f"/games/{game_id}", headers=_auth(emmitt_game["spectator_token"])).json()[
            "input_request"
        ]
        is None
    )

    owner_request = client.get(f"/games/{game_id}", headers=_auth(em_token)).json()["input_request"]
    assert owner_request["player_id"] == "hero_emmitt"
    assert set(owner_request["valid_options"]) == {
        "reverse_time",
        "unstable_timeline",
    }

    stale = client.post(
        f"/games/{game_id}/input",
        json={"request_id": "stale-request", "selection": "unstable_timeline"},
        headers=_auth(em_token),
    )
    assert stale.status_code == 400

    r4 = client.post(
        f"/games/{game_id}/input",
        json={
            "request_id": owner_request["request_id"],
            "selection": "unstable_timeline",
        },
        headers=_auth(em_token),
    )
    assert r4.status_code == 200

    view = client.get(f"/games/{game_id}", headers=_auth(em_token)).json()["view"]
    emmitt_view = next(
        h for t in view["teams"].values() for h in t["heroes"] if h["id"] == "hero_emmitt"
    )
    assert "unstable_timeline" in [c["id"] for c in emmitt_view["hand"]]
    assert emmitt_view["current_turn_card"]["id"] == "reverse_time"
    assert emmitt_view["extra_turn_card"] is None


def test_emmitt_planning_done_endpoint(client, emmitt_game):
    """Commit one + planning-done closes Emmitt's planning without a second card."""
    game_id = emmitt_game["game_id"]
    em_token = _token_for(emmitt_game, "hero_emmitt")
    wa_token = _token_for(emmitt_game, "hero_wasp")

    client.post(
        f"/games/{game_id}/cards", json={"card_id": "reverse_time"}, headers=_auth(em_token)
    )
    wasp_card = _first_hand_card_id(client, emmitt_game, "hero_wasp")
    r = client.post(f"/games/{game_id}/cards", json={"card_id": wasp_card}, headers=_auth(wa_token))
    assert r.json()["current_phase"] == "PLANNING"  # waits for Emmitt

    done = client.post(f"/games/{game_id}/planning-done", headers=_auth(em_token))
    assert done.status_code == 200
    assert done.json()["current_phase"] == "RESOLUTION"


def test_planning_done_before_commit_rejected(client, emmitt_game):
    game_id = emmitt_game["game_id"]
    em_token = _token_for(emmitt_game, "hero_emmitt")
    resp = client.post(f"/games/{game_id}/planning-done", headers=_auth(em_token))
    assert resp.status_code == 400


def _hero_view(client, game_id, token, hero_id):
    view = client.get(f"/games/{game_id}", headers=_auth(token)).json()["view"]
    return next(h for t in view["teams"].values() for h in t["heroes"] if h["id"] == hero_id)


def test_can_commit_second_card_flag(client, emmitt_game):
    """The own-hero view exposes can_commit_second_card across the two-card window."""
    game_id = emmitt_game["game_id"]
    em_token = _token_for(emmitt_game, "hero_emmitt")
    wa_token = _token_for(emmitt_game, "hero_wasp")

    # Before committing: not yet open.
    assert _hero_view(client, game_id, em_token, "hero_emmitt")["can_commit_second_card"] is False

    # After the first commit: open in Emmitt's own view...
    client.post(
        f"/games/{game_id}/cards", json={"card_id": "reverse_time"}, headers=_auth(em_token)
    )
    assert _hero_view(client, game_id, em_token, "hero_emmitt")["can_commit_second_card"] is True
    # ...but never leaked to the opponent's view of Emmitt.
    assert _hero_view(client, game_id, wa_token, "hero_emmitt")["can_commit_second_card"] is False

    # After the second commit: closed again.
    client.post(
        f"/games/{game_id}/cards", json={"card_id": "unstable_timeline"}, headers=_auth(em_token)
    )
    assert _hero_view(client, game_id, em_token, "hero_emmitt")["can_commit_second_card"] is False


def test_two_committed_cards_are_both_exposed_during_planning(client, emmitt_game):
    """The planning view uses extra_turn_card for the buffered first commit."""
    game_id = emmitt_game["game_id"]
    em_token = _token_for(emmitt_game, "hero_emmitt")
    wa_token = _token_for(emmitt_game, "hero_wasp")

    client.post(
        f"/games/{game_id}/cards", json={"card_id": "reverse_time"}, headers=_auth(em_token)
    )
    after_first = _hero_view(client, game_id, em_token, "hero_emmitt")
    assert after_first["current_turn_card"]["id"] == "reverse_time"
    assert after_first["extra_turn_card"] is None

    client.post(
        f"/games/{game_id}/cards",
        json={"card_id": "unstable_timeline"},
        headers=_auth(em_token),
    )

    owner_view = _hero_view(client, game_id, em_token, "hero_emmitt")
    assert owner_view["current_turn_card"]["id"] == "unstable_timeline"
    assert owner_view["extra_turn_card"]["id"] == "reverse_time"

    opponent_view = _hero_view(client, game_id, wa_token, "hero_emmitt")
    assert opponent_view["current_turn_card"]["is_facedown"] is True
    assert opponent_view["extra_turn_card"]["is_facedown"] is True
    assert "id" not in opponent_view["extra_turn_card"]


def test_can_commit_second_card_false_for_normal_hero(client, game_data):
    """A hero without the two-card ultimate never sees the flag set."""
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")
    assert _hero_view(client, game_id, token, "hero_arien")["can_commit_second_card"] is False
    card = _first_hand_card_id(client, game_data, "hero_arien")
    client.post(f"/games/{game_id}/cards", json={"card_id": card}, headers=_auth(token))
    assert _hero_view(client, game_id, token, "hero_arien")["can_commit_second_card"] is False


def test_second_commit_without_ultimate_still_409(client, game_data):
    """Regression: a normal hero's second commit is still rejected."""
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")
    card = _first_hand_card_id(client, game_data, "hero_arien")
    assert (
        client.post(
            f"/games/{game_id}/cards", json={"card_id": card}, headers=_auth(token)
        ).status_code
        == 200
    )
    view = client.get(f"/games/{game_id}", headers=_auth(token)).json()["view"]
    arien = next(h for t in view["teams"].values() for h in t["heroes"] if h["id"] == "hero_arien")
    second = arien["hand"][0]["id"]
    resp = client.post(f"/games/{game_id}/cards", json={"card_id": second}, headers=_auth(token))
    assert resp.status_code == 409


# ---- POST /games/{game_id}/uncommit ----


def _hero_view_of(client, game_id: str, token: str, hero_id: str) -> dict:
    view = client.get(f"/games/{game_id}", headers=_auth(token)).json()
    for team_data in view["view"]["teams"].values():
        for hero in team_data["heroes"]:
            if hero["id"] == hero_id:
                return hero
    raise ValueError(f"Hero {hero_id} not in view")


def test_uncommit_returns_card_to_hand(client, game_data):
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")

    hand = _hero_view_of(client, game_id, token, "hero_arien")["hand"]
    card_id = hand[0]["id"]
    hand_size = len(hand)

    resp = client.post(f"/games/{game_id}/cards", json={"card_id": card_id}, headers=_auth(token))
    assert resp.status_code == 200

    resp = client.post(f"/games/{game_id}/uncommit", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["current_phase"] == "PLANNING"

    hero = _hero_view_of(client, game_id, token, "hero_arien")
    assert hero["current_turn_card"] is None
    assert len(hero["hand"]) == hand_size
    assert any(c["id"] == card_id for c in hero["hand"])


def test_uncommit_then_commit_other_card(client, game_data):
    game_id = game_data["game_id"]
    token = _token_for(game_data, "hero_arien")

    hand = _hero_view_of(client, game_id, token, "hero_arien")["hand"]
    card0, card1 = hand[0]["id"], hand[1]["id"]

    client.post(f"/games/{game_id}/cards", json={"card_id": card0}, headers=_auth(token))
    client.post(f"/games/{game_id}/uncommit", headers=_auth(token))
    resp = client.post(f"/games/{game_id}/cards", json={"card_id": card1}, headers=_auth(token))
    assert resp.status_code == 200

    hero = _hero_view_of(client, game_id, token, "hero_arien")
    assert hero["current_turn_card"]["id"] == card1


def test_uncommit_nothing_committed_is_400(client, game_data):
    token = _token_for(game_data, "hero_arien")
    resp = client.post(f"/games/{game_data['game_id']}/uncommit", headers=_auth(token))
    assert resp.status_code == 400
    assert "no committed card" in resp.json()["detail"]


def test_uncommit_spectator_forbidden(client, game_data):
    resp = client.post(
        f"/games/{game_data['game_id']}/uncommit",
        headers=_auth(game_data["spectator_token"]),
    )
    assert resp.status_code == 403


def test_uncommit_after_lock_in_is_409(client, game_data):
    """Once the last commit fires revelation, a take-back gets a phase error."""
    game_id = game_data["game_id"]
    a_token = _token_for(game_data, "hero_arien")
    w_token = _token_for(game_data, "hero_wasp")

    a_card = _hero_view_of(client, game_id, a_token, "hero_arien")["hand"][0]["id"]
    w_card = _hero_view_of(client, game_id, w_token, "hero_wasp")["hand"][0]["id"]

    client.post(f"/games/{game_id}/cards", json={"card_id": a_card}, headers=_auth(a_token))
    client.post(f"/games/{game_id}/cards", json={"card_id": w_card}, headers=_auth(w_token))

    resp = client.post(f"/games/{game_id}/uncommit", headers=_auth(a_token))
    assert resp.status_code == 409
