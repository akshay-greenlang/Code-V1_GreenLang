# -*- coding: utf-8 -*-
"""Phase 3 source-safety helper — gate ingestion runs by environment.

Authority: CTO Phase 3 brief 2026-04-28, Block 7 / Gate 4.
Owner    : GL-Factors Engineering (Wave 3.0).

This module provides the runtime guard ``assert_source_safe_for_env`` that
the Phase 3 :class:`IngestionPipelineRunner` MUST call before running or
publishing against the ``production`` environment. The guard refuses to run
when the source registry entry has ``status`` in
``{pending_legal_review, blocked}`` or when its ``release_milestone`` is
later than ``v0.1``.

Hard rules:
    * ``env in {dev, test, staging}`` -> always allow (the gate is no-op).
    * ``env == 'production'`` -> enforce status + release_milestone gates.
    * Anything else -> reject conservatively.

The helper is deliberately stdlib-only — no ``yaml`` import here. The caller
(IngestionPipelineRunner / CLI) is expected to pass the *parsed* registry
dict for the source under test.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from greenlang.factors.ingestion.exceptions import IngestionError


__all__ = [
    "SourceNotApprovedForEnvError",
    "is_source_safe_for_env",
    "assert_source_safe_for_env",
    "parse_release_milestone",
    "BLOCKING_STATUSES",
    "ALPHA_RELEASE_MILESTONE",
]


# Statuses that block production ingestion outright. Any other status (e.g.
# ``alpha_v0_1``, ``approved``) is allowed pending the release-milestone
# check below.
BLOCKING_STATUSES: frozenset = frozenset({
    "pending_legal_review",
    "blocked",
})

# Only sources milestoned at v0.1 (or earlier — there is no earlier today)
# may be ingested into production at this stage of the program. Wave 4+
# will widen this as later milestones earn legal sign-off.
ALPHA_RELEASE_MILESTONE: tuple = (0, 1)


class SourceNotApprovedForEnvError(IngestionError):
    """A production ingestion targeted a source that has not cleared rights/legal.

    The exception is intentionally an :class:`IngestionError` subclass so the
    runner's blanket ``except IngestionError`` clauses transition the run to
    ``failed`` with a structured ``error_json`` payload identifying the
    source + the failing gate.
    """

    def __init__(
        self,
        message: str,
        *,
        source_id: Optional[str] = None,
        env: Optional[str] = None,
        status: Optional[str] = None,
        release_milestone: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            stage="source_safety",
            details={
                "source_id": source_id,
                "env": env,
                "status": status,
                "release_milestone": release_milestone,
                "reason": reason,
            },
        )


def parse_release_milestone(value: Any) -> Optional[tuple]:
    """Parse a ``release_milestone`` string like ``"v0.1"`` or ``"v2.5"``.

    Returns a ``(major, minor)`` tuple suitable for tuple comparison, or
    ``None`` if the value is missing / malformed (callers treat ``None`` as
    "no milestone declared" -> conservative deny in production).
    """
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s.startswith("v"):
        return None
    rest = s[1:]
    parts = rest.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (TypeError, ValueError):
        return None
    return (major, minor)


def _normalise_env(env: str) -> str:
    return (env or "").strip().lower()


def is_source_safe_for_env(
    source_entry: Mapping[str, Any],
    env: str,
) -> bool:
    """Return True iff ``source_entry`` may be ingested in ``env``.

    Non-production environments always return True (the helper is a no-op
    outside of production deploys). Production runs apply the
    BLOCKING_STATUSES + ALPHA_RELEASE_MILESTONE gate.
    """
    env_norm = _normalise_env(env)
    if env_norm != "production":
        return True

    status = (source_entry.get("status") or "").strip().lower()
    if status in BLOCKING_STATUSES:
        return False

    milestone_value = source_entry.get("release_milestone")
    parsed = parse_release_milestone(milestone_value)
    if parsed is None:
        return False
    if parsed > ALPHA_RELEASE_MILESTONE:
        return False
    return True


def assert_source_safe_for_env(
    source_entry: Mapping[str, Any],
    env: str,
) -> None:
    """Raise :class:`SourceNotApprovedForEnvError` if the source cannot run in ``env``.

    Parameters
    ----------
    source_entry
        The parsed ``source_registry.yaml`` entry for the source under test.
        Must carry ``source_id``, ``status``, and ``release_milestone`` keys.
    env
        Environment label: ``dev``, ``test``, ``staging``, ``production``.

    Raises
    ------
    SourceNotApprovedForEnvError
        When ``env == 'production'`` AND the source is unapproved.
    """
    if not isinstance(source_entry, Mapping):
        raise SourceNotApprovedForEnvError(
            "source_entry must be a mapping",
            env=env,
            reason="source_entry_not_mapping",
        )

    env_norm = _normalise_env(env)
    source_id = source_entry.get("source_id") or "<unknown>"
    status_raw = source_entry.get("status")
    status = (status_raw or "").strip().lower()
    milestone_value = source_entry.get("release_milestone")

    if env_norm not in {"dev", "test", "staging", "production"}:
        raise SourceNotApprovedForEnvError(
            "unknown env %r; allowed: dev|test|staging|production" % env,
            source_id=source_id,
            env=env,
            status=status_raw,
            release_milestone=milestone_value,
            reason="unknown_env",
        )

    if env_norm != "production":
        return

    if status in BLOCKING_STATUSES:
        raise SourceNotApprovedForEnvError(
            "source %s cannot run in production: status=%s"
            % (source_id, status),
            source_id=source_id,
            env=env,
            status=status_raw,
            release_milestone=milestone_value,
            reason="status_blocked",
        )

    parsed = parse_release_milestone(milestone_value)
    if parsed is None:
        raise SourceNotApprovedForEnvError(
            "source %s cannot run in production: missing/invalid release_milestone"
            % source_id,
            source_id=source_id,
            env=env,
            status=status_raw,
            release_milestone=milestone_value,
            reason="release_milestone_missing",
        )
    if parsed > ALPHA_RELEASE_MILESTONE:
        raise SourceNotApprovedForEnvError(
            "source %s cannot run in production: release_milestone=%s "
            "exceeds approved alpha milestone v0.1"
            % (source_id, milestone_value),
            source_id=source_id,
            env=env,
            status=status_raw,
            release_milestone=milestone_value,
            reason="release_milestone_too_late",
        )
