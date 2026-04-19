# -*- coding: utf-8 -*-
"""
Hotfix / rollback edition override (U6).

Provides safe edition rollback for the factor catalog. Validates that the
target edition exists, contains factors, and is not already the active
default before performing the switch.

The ``GL_FACTORS_FORCE_EDITION`` environment variable mechanism (see
``service.resolve_edition_id``) is a separate override path for emergency
pinning without modifying the database. This module handles the permanent
database-level rollback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.service import resolve_edition_id

logger = logging.getLogger(__name__)


@dataclass
class RollbackResult:
    """Result of an edition rollback operation."""

    success: bool
    target_edition_id: str
    previous_edition_id: Optional[str] = None
    reason: str = ""
    timestamp: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "success": self.success,
            "target_edition_id": self.target_edition_id,
            "previous_edition_id": self.previous_edition_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "warnings": self.warnings,
        }


def resolve_edition_with_rollback_override(
    repo: FactorCatalogRepository,
    header_edition: Optional[str],
    query_edition: Optional[str],
) -> Tuple[str, str]:
    """Delegate to resolve_edition_id (env override lives there).

    Retained for backward compatibility with callers that import from
    this module.
    """
    return resolve_edition_id(repo, header_edition, query_edition)


def _validate_rollback_target(
    repo: FactorCatalogRepository,
    target_edition_id: str,
) -> Tuple[bool, str]:
    """
    Validate that the target edition is a safe rollback destination.

    Checks:
        1. The edition exists in the repository.
        2. The edition contains at least one factor.
        3. The edition is not already the active default.

    Returns:
        (is_valid, error_message) where error_message is empty on success.
    """
    # Check edition exists
    try:
        repo.resolve_edition(target_edition_id)
    except ValueError:
        return False, f"Target edition {target_edition_id!r} does not exist"

    # Check edition has factors (refuse rollback to empty edition)
    summaries = repo.list_factor_summaries(target_edition_id)
    if len(summaries) == 0:
        return False, (
            f"Target edition {target_edition_id!r} contains 0 factors; "
            "rollback to an empty edition is not allowed"
        )

    # Check not already the active default
    try:
        current_default = repo.get_default_edition_id()
    except Exception:
        current_default = None

    if current_default == target_edition_id:
        return False, (
            f"Target edition {target_edition_id!r} is already the active default; "
            "no rollback needed"
        )

    return True, ""


def rollback_to_edition(
    repo: FactorCatalogRepository,
    target_edition_id: str,
    *,
    reason: str = "",
    operator: str = "system",
) -> RollbackResult:
    """
    Roll back the default edition pointer to a previous edition.

    Performs validation, updates the edition status, and logs the rollback
    event.  This function does NOT delete any data; it only changes which
    edition is considered the default/stable one.

    For SQLite-backed repositories, the function updates the ``status``
    column of the editions table. For memory repositories, this is a
    no-op with a warning (memory repos are single-edition).

    Args:
        repo: Factor catalog repository.
        target_edition_id: The edition to roll back to.
        reason: Human-readable reason for the rollback (audit trail).
        operator: Who initiated the rollback (audit trail).

    Returns:
        RollbackResult with success/failure and details.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = RollbackResult(
        success=False,
        target_edition_id=target_edition_id,
        timestamp=timestamp,
    )

    # Step 1: Validate the target
    is_valid, error_msg = _validate_rollback_target(repo, target_edition_id)
    if not is_valid:
        result.reason = error_msg
        logger.warning(
            "Rollback rejected: target=%s reason=%s",
            target_edition_id,
            error_msg,
        )
        return result

    # Step 2: Capture the current default for the audit trail
    try:
        previous_edition = repo.get_default_edition_id()
    except Exception:
        previous_edition = None
    result.previous_edition_id = previous_edition

    # Step 3: Perform the rollback
    #   For SQLite repos we update edition statuses directly.
    #   For memory repos we warn that rollback is not persistent.
    from greenlang.factors.catalog_repository import (
        MemoryFactorCatalogRepository,
        SqliteFactorCatalogRepository,
    )

    if isinstance(repo, SqliteFactorCatalogRepository):
        _rollback_sqlite(repo, target_edition_id, previous_edition)
        result.success = True
        result.reason = reason or "Rollback completed successfully"
        logger.info(
            "Edition rollback completed: %s -> %s by %s reason=%s",
            previous_edition,
            target_edition_id,
            operator,
            reason,
        )
    elif isinstance(repo, MemoryFactorCatalogRepository):
        result.success = False
        result.reason = (
            "Memory-backed repository does not support persistent rollback; "
            "use GL_FACTORS_FORCE_EDITION environment variable instead"
        )
        result.warnings.append("In-memory repos are single-edition")
        logger.warning(
            "Rollback skipped for memory repo: target=%s",
            target_edition_id,
        )
    else:
        # Unknown repo type -- attempt generic approach
        result.success = False
        result.reason = (
            f"Rollback not implemented for repository type {type(repo).__name__}; "
            "use GL_FACTORS_FORCE_EDITION as a workaround"
        )
        logger.warning(
            "Rollback not supported for repo type %s",
            type(repo).__name__,
        )

    return result


def _rollback_sqlite(
    repo: "SqliteFactorCatalogRepository",
    target_edition_id: str,
    previous_edition_id: Optional[str],
) -> None:
    """
    Perform the SQLite-level edition status swap.

    Demotes the current stable edition to ``rollback_demoted`` and promotes
    the target edition to ``stable``.
    """
    conn = repo._conn()
    try:
        # Demote the current stable edition(s)
        conn.execute(
            """
            UPDATE editions
               SET status = 'rollback_demoted'
             WHERE status = 'stable'
               AND edition_id != ?
            """,
            (target_edition_id,),
        )

        # Promote the target edition to stable
        conn.execute(
            """
            UPDATE editions
               SET status = 'stable'
             WHERE edition_id = ?
            """,
            (target_edition_id,),
        )

        conn.commit()
        logger.info(
            "SQLite rollback: demoted previous stable, promoted %s to stable",
            target_edition_id,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
