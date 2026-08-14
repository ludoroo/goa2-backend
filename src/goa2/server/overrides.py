"""Consensus-override proposal lifecycle.

A proposal is coordination, not game state: it lives on ManagedGame, is never
saved or broadcast in views, and dies on server restart. Only the outcome is
recorded (as a replay decision, by the ws apply path).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from goa2.engine.overrides import OverrideRejectedError, get_op, summarize_op

if TYPE_CHECKING:
    from goa2.server.registry import ManagedGame


def override_proposal_timeout_seconds() -> int:
    try:
        return int(os.environ.get("GOA2_OVERRIDE_TIMEOUT_SECONDS", 120))
    except ValueError:
        return 120


@dataclass
class OverrideProposal:
    id: str
    proposer_hero_id: str
    family: str  # "patch" | "unstick" | "rewind"
    op: str | None  # None for rewind
    args: dict[str, Any]
    to: int | None  # rewind target decision index
    summary: str
    eligible_voters: list[str]  # snapshotted at proposal time
    votes: dict[str, bool]
    created_at: float
    expires_at: float

    def threshold(self) -> int:
        """Strictly more than half of the snapshotted voters."""
        return len(self.eligible_voters) // 2 + 1

    def tally(self) -> dict[str, list[str]]:
        return {
            "yes": sorted(h for h, v in self.votes.items() if v),
            "no": sorted(h for h, v in self.votes.items() if not v),
        }

    def outcome(self) -> str | None:
        """'applied' | 'rejected' when decided, None while still open."""
        yes = sum(1 for v in self.votes.values() if v)
        no = sum(1 for v in self.votes.values() if not v)
        if yes >= self.threshold():
            return "applied"
        if yes + (len(self.eligible_voters) - yes - no) < self.threshold():
            return "rejected"  # threshold unreachable
        return None


def connected_hero_ids(game: ManagedGame) -> list[str]:
    """Heroes with a live player websocket right now. Spectators never count."""
    return sorted(
        {game.player_tokens[token] for token in game.ws_connections if token in game.player_tokens}
    )


def create_proposal(
    game: ManagedGame, proposer_hero_id: str, data: dict[str, Any]
) -> OverrideProposal:
    if game.pending_override is not None:
        raise ValueError("Another override proposal is already open")

    family = data.get("family")
    if family not in ("patch", "unstick", "rewind"):
        raise ValueError("family must be patch, unstick, or rewind")

    op_name: str | None = None
    args: dict[str, Any] = {}
    to: int | None = None
    if family == "rewind":
        raw_to = data.get("to")
        if not isinstance(raw_to, int) or isinstance(raw_to, bool) or raw_to < 0:
            raise ValueError("rewind requires a non-negative integer 'to'")
        to = raw_to
        summary = f"Rewind the game to decision {to}"
    else:
        op_name = str(data.get("op", ""))
        try:
            op = get_op(op_name)
        except OverrideRejectedError as exc:
            raise ValueError(exc.message) from exc
        if op.family != family:
            raise ValueError(f"op {op_name!r} belongs to family {op.family!r}")
        args = data.get("args") or {}
        try:
            op.args_model.model_validate(args)
        except Exception as exc:
            raise ValueError(f"Invalid args for {op_name}: {exc}") from exc
        summary = summarize_op(op_name, args)

    eligible = connected_hero_ids(game)
    if proposer_hero_id not in eligible:
        raise ValueError("Proposer must be a connected player")

    now = time.time()
    return OverrideProposal(
        id=uuid.uuid4().hex[:12],
        proposer_hero_id=proposer_hero_id,
        family=family,
        op=op_name,
        args=args,
        to=to,
        summary=summary,
        eligible_voters=eligible,
        votes={proposer_hero_id: True},
        created_at=now,
        expires_at=now + override_proposal_timeout_seconds(),
    )


def register_vote(proposal: OverrideProposal, hero_id: str, approve: bool) -> None:
    if hero_id not in proposal.eligible_voters:
        raise ValueError("Only players connected at proposal time may vote")
    proposal.votes[hero_id] = approve


# ---- WS payload builders --------------------------------------------------


def proposed_msg(proposal: OverrideProposal) -> dict[str, Any]:
    return {
        "type": "OVERRIDE_PROPOSED",
        "proposal_id": proposal.id,
        "proposer_hero_id": proposal.proposer_hero_id,
        "family": proposal.family,
        "op": proposal.op,
        "args": proposal.args,
        "to": proposal.to,
        "summary": proposal.summary,
        "eligible_voters": proposal.eligible_voters,
        "threshold": proposal.threshold(),
        "tally": proposal.tally(),
        "expires_at": proposal.expires_at,
    }


def updated_msg(proposal: OverrideProposal) -> dict[str, Any]:
    return {
        "type": "OVERRIDE_UPDATED",
        "proposal_id": proposal.id,
        "tally": proposal.tally(),
    }


def resolved_msg(
    proposal: OverrideProposal,
    outcome: str,
    reason: dict[str, str] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "OVERRIDE_RESOLVED",
        "proposal_id": proposal.id,
        "outcome": outcome,  # applied | rejected | expired | cancelled
        "tally": proposal.tally(),
    }
    if reason is not None:
        msg["reason"] = reason  # {"code": ..., "message": ...}
    return msg
