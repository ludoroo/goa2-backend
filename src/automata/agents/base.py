"""Agent interface.

An Agent makes two kinds of decisions, matching the engine's two decision
points:

1. PLANNING: pick which card to commit from a hero's hand (or pass).
2. RESOLUTION: answer an `InputRequest` by choosing one of its enumerated
   `options` (or SKIP), returning the raw selection value the engine expects.

Keeping this contract engine-driven (we never reimplement legality) means the
same interface serves random, heuristic, and search (ISMCTS) agents.

Planning is expressed at two levels:

- ``Agent.choose_card(state, hero) -> Card | None`` is the *policy* boundary
  every agent implements. Returning ``None`` means "no card to commit" — the
  hero has an empty hand, or the policy declined to play one.
- :class:`PlanningDecision` is the *driver* vocabulary. It distinguishes
  ``COMMIT`` (play a specific card), ``FINISH`` (end a multi-card planning
  sequence — Emmitt), and ``PASS`` (skip the turn), so callers can act on the
  intent explicitly rather than overloading ``None``. The runtime driver
  consumes it directly; ``plan_from_card_choice()`` converts the raw policy
  output for callers that still speak in ``Card | None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from goa2.domain.input import InputRequest, selection_value
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState

__all__ = [
    "Agent",
    "Cube",
    "HexLike",
    "PlanningDecision",
    "PlanningKind",
    "hex_distance",
    "plan_from_card_choice",
    "selection_value",
    "to_cube",
]


class Agent(Protocol):
    """A decision-maker for one or more heroes."""

    def choose_card(self, state: GameState, hero: Hero) -> Card | None:
        """Return a card from ``hero.hand`` to commit, or None to pass."""
        ...

    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        """Return the raw `selection` value answering ``request``.

        Typically the raw value behind a chosen ``InputOption`` (a unit id, a
        {q,r,s} hex dict, an int, or an option id), or the ``SKIP`` sentinel.

        ``owned_hero_ids`` lets runtime drivers pass the bot's controlled
        hero set uniformly to every agent implementation:

        - Search-backed agents (:class:`~automata.search.agent.ISMCTSAgent`)
          require it non-empty to anchor the root and reject wrong-team /
          non-owned calls at the public boundary.
        - Simpler agents (Random / Heuristic) accept it for signature
          compatibility and ignore it — they read only ``state`` and
          ``request``.

        The default ``None`` keeps ordinary existing callers (headless
        harness, tests that don't route through the driver) unchanged.
        """
        ...


# --------------------------------------------------------------------------- #
# Planning decisions
#
# ``choose_card`` returning ``None`` conflates three intents at the driver
# level: the hand is empty, the agent chose not to play, or (for multi-card
# heroes like Emmitt) the agent finished a legal sequence. ``PlanningDecision``
# separates those cases so the driver can call
# ``session.commit_card()`` / ``session.finish_planning()`` / ``session.pass_turn()``
# unambiguously.
# --------------------------------------------------------------------------- #


class PlanningKind(StrEnum):
    """The three driver-level planning intents.

    A ``StrEnum`` so equality with the historical string values (``"COMMIT"``,
    ``"FINISH"``, ``"PASS"``) keeps working and the enum serializes cleanly.
    Instances of :class:`PlanningDecision` reject any other value at
    construction time — no silent acceptance of a stray string kind.
    """

    COMMIT = "COMMIT"
    FINISH = "FINISH"
    PASS = "PASS"


@dataclass(frozen=True)
class PlanningDecision:
    """A driver-level planning intent for a single hero.

    Instances are constructed via :meth:`commit`, :meth:`finish`, or
    :meth:`pass_`. The invariants (``COMMIT`` iff ``card is not None``;
    ``FINISH``/``PASS`` iff ``card is None``; ``kind`` is a valid
    :class:`PlanningKind`) are enforced in ``__post_init__`` and applied to
    values coerced from raw strings so the runtime never accepts an unknown
    kind.
    """

    kind: PlanningKind
    card: Card | None = None

    def __post_init__(self) -> None:
        # Coerce raw strings so callers (or naive deserializers) get a clear
        # error instead of silently constructing a decision with an unknown
        # kind. ``PlanningKind(<invalid>)`` raises ``ValueError`` with the
        # offending value, which is what we want here.
        if not isinstance(self.kind, PlanningKind):
            coerced = PlanningKind(self.kind)  # raises on unknown values
            object.__setattr__(self, "kind", coerced)

        if self.kind is PlanningKind.COMMIT:
            if self.card is None:
                raise ValueError("COMMIT PlanningDecision requires a card")
        else:
            if self.card is not None:
                raise ValueError(
                    f"{self.kind.value} PlanningDecision must not carry a card"
                )

    @classmethod
    def commit(cls, card: Card) -> PlanningDecision:
        """A concrete card to commit for this hero this turn."""
        return cls(kind=PlanningKind.COMMIT, card=card)

    @classmethod
    def finish(cls) -> PlanningDecision:
        """End a legal multi-card planning sequence (e.g. Emmitt)."""
        return cls(kind=PlanningKind.FINISH, card=None)

    @classmethod
    def pass_(cls) -> PlanningDecision:
        """Pass this hero's turn (empty hand or the agent chose not to play)."""
        return cls(kind=PlanningKind.PASS, card=None)


def plan_from_card_choice(hero: Hero, card: Card | None) -> PlanningDecision:
    """Map an ``Agent.choose_card`` result to a :class:`PlanningDecision`.

    Contract:

    - ``card is None`` → ``PASS``. The agent declined to play, or the hero has
      no legal cards; the driver will call ``session.pass_turn()``.
    - ``card is not None`` and ``card in hero.hand`` → ``COMMIT``. The driver
      will call ``session.commit_card(hero, card)``.
    - ``card is not None`` and ``card not in hero.hand`` → :class:`ValueError`.
      This includes the empty-hand case: an agent may only commit a card the
      hero actually holds. We fail loudly rather than silently downgrading to
      ``PASS`` so a bug in an agent's ``choose_card`` is caught at the driver
      boundary, not swallowed.

    ``FINISH`` is not produced here: multi-card heroes need a driver-level
    signal, which this helper is deliberately unaware of.
    """
    if card is None:
        return PlanningDecision.pass_()
    if card not in hero.hand:
        raise ValueError(
            f"choose_card returned card {card.id!r} that is not in hero "
            f"{hero.id!r}'s hand ({[c.id for c in hero.hand]!r})"
        )
    return PlanningDecision.commit(card)


# --------------------------------------------------------------------------- #
# Shared hex geometry. Agents receive locations in loose forms — {q,r,s} dicts
# (option metadata, entity_locations), engine Hex objects, or anything with
# .q/.r/.s — so these tolerate all of them, unlike the engine's typed
# Hex.distance. Shared here so every agent (and feature extractor) reuses one
# implementation.
# --------------------------------------------------------------------------- #

Cube = tuple[int, int, int]


def to_cube(loc: Any) -> Cube | None:
    """Coerce a loose location into cube coords ``(q, r, s)``, or None.

    Accepts a ``{q, r, s}`` dict, an engine ``Hex``, or any object exposing
    ``.q``/``.r`` (``.s`` derived if absent). Returns None for None / unparseable
    inputs.
    """
    if loc is None:
        return None
    if isinstance(loc, dict):
        dq, dr = int(loc.get("q", 0)), int(loc.get("r", 0))
        return (dq, dr, int(loc.get("s", -dq - dr)))
    aq, ar = getattr(loc, "q", None), getattr(loc, "r", None)
    if aq is None or ar is None:
        return None
    s = getattr(loc, "s", None)
    return (int(aq), int(ar), int(s if s is not None else -aq - ar))


def hex_distance(a: Any, b: Any) -> int:
    """Cube hex distance between two loose locations.

    Returns a large sentinel (99) when either side is unparseable, so callers
    can treat "unknown" as "very far" without special-casing None.
    """
    ca, cb = to_cube(a), to_cube(b)
    if ca is None or cb is None:
        return 99
    return (abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) + abs(ca[2] - cb[2])) // 2


class HexLike:
    """Wrap a ``{q, r, s}`` dict so ``x in zone.hexes`` works (Hex equality by
    coords). Lets agents membership-test loose hex dicts against engine zones."""

    __slots__ = ("q", "r", "s")

    def __init__(self, d: dict[str, Any]) -> None:
        self.q = d.get("q", 0)
        self.r = d.get("r", 0)
        self.s = d.get("s", d.get("q", 0) * -1 - d.get("r", 0))

    def __eq__(self, other: object) -> bool:
        return (
            getattr(other, "q", None) == self.q
            and getattr(other, "r", None) == self.r
            and getattr(other, "s", None) == self.s
        )

    def __hash__(self) -> int:
        return hash((self.q, self.r, self.s))
