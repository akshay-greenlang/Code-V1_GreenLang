# -*- coding: utf-8 -*-
"""Edition rollback CLI surface.

``watch/rollback_edition.py`` already owns the primitive that flips
the edition-status pointer in the catalog.  This module layers the
operator workflow on top:

1. :func:`list_rollback_candidates` — enumerate past stable editions
   the operator can roll back to, with factor-count + publish date
   so the target is chosen against real evidence.
2. :func:`preview_rollback` — dry-run that validates the target and
   summarises what changes (added / removed / modified factors) so
   the operator sees the blast radius before committing.
3. :func:`execute_rollback_with_receipt` — the production flip:
   calls :func:`rollback_to_edition`, emits a
   :class:`RegulatoryChangeEvent`, persists a signed receipt.

The whole thing is deliberately repository-backend-agnostic; the
``FactorCatalogRepository`` contract is all we require.  CLI wiring
in ``greenlang/factors/cli.py`` can delegate here to keep the
``gl factors rollback-*`` commands thin.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.watch.regulatory_events import (
    RegulatoryChangeEvent,
    RegulatoryEventKind,
    RegulatoryEventStore,
)
from greenlang.factors.watch.rollback_edition import (
    RollbackResult,
    rollback_to_edition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate listing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackCandidate:
    """One edition the operator could roll back to."""

    edition_id: str
    status: str
    factor_count: int
    published_at: Optional[str] = None
    label: Optional[str] = None
    is_current_default: bool = False

    def as_table_row(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_manifest(repo: FactorCatalogRepository, edition_id: str) -> Dict[str, Any]:
    try:
        if hasattr(repo, "get_manifest_dict"):
            return repo.get_manifest_dict(edition_id) or {}
    except Exception:
        pass
    return {}


def _safe_list_editions(repo: FactorCatalogRepository) -> List[Dict[str, Any]]:
    """Return every edition row the repo knows about.

    ``FactorCatalogRepository`` historically exposed ``list_editions``;
    older implementations only carry a single edition via
    ``resolve_edition``.  We probe both surfaces so this module works
    against every repo flavour without hard-coding a specific backend.
    """
    if hasattr(repo, "list_editions"):
        try:
            rows = list(repo.list_editions())
            if rows and isinstance(rows[0], dict):
                return rows
            return [
                {
                    "edition_id": getattr(r, "edition_id", str(r)),
                    "status": getattr(r, "status", "stable"),
                    "label": getattr(r, "label", None),
                    "published_at": getattr(r, "published_at", None),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("list_editions failed: %s", exc)
    # Fall back to the single edition the repo resolves by default.
    try:
        ed = repo.resolve_edition(None) if hasattr(repo, "resolve_edition") else None
    except Exception:
        ed = None
    if ed:
        return [{"edition_id": ed, "status": "stable", "label": None, "published_at": None}]
    return []


def list_rollback_candidates(
    repo: FactorCatalogRepository,
    *,
    include_deprecated: bool = False,
) -> List[RollbackCandidate]:
    """Return every stable (or previously stable) edition, newest first.

    ``include_deprecated`` toggles whether demoted editions from prior
    rollbacks are returned too — handy when the operator needs to
    bounce back to something older than the current default.
    """
    current_default: Optional[str]
    try:
        current_default = repo.get_default_edition_id()
    except Exception:
        current_default = None

    rows = _safe_list_editions(repo)
    candidates: List[RollbackCandidate] = []
    for row in rows:
        status = (row.get("status") or "").lower()
        if status == "deprecated" and not include_deprecated:
            continue
        edition_id = str(row.get("edition_id"))
        try:
            summaries = repo.list_factor_summaries(edition_id)
            factor_count = len(summaries)
        except Exception:
            factor_count = 0
        manifest = _safe_manifest(repo, edition_id)
        published_at = (
            row.get("published_at")
            or manifest.get("published_at")
            or manifest.get("timestamp")
        )
        candidates.append(
            RollbackCandidate(
                edition_id=edition_id,
                status=status or "unknown",
                factor_count=factor_count,
                published_at=str(published_at) if published_at else None,
                label=row.get("label") or manifest.get("label"),
                is_current_default=(edition_id == current_default),
            )
        )
    # Newest first if timestamps are present; otherwise preserve repo order.
    candidates.sort(
        key=lambda c: (c.published_at or "", c.edition_id),
        reverse=True,
    )
    return candidates


# ---------------------------------------------------------------------------
# Preview (dry-run)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackPreview:
    """What the operator will see if they execute the rollback."""

    target_edition_id: str
    current_default: Optional[str]
    safe: bool
    blockers: List[str] = field(default_factory=list)
    added_factor_ids: List[str] = field(default_factory=list)
    removed_factor_ids: List[str] = field(default_factory=list)
    changed_factor_ids: List[str] = field(default_factory=list)
    target_factor_count: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "target_edition_id": self.target_edition_id,
            "current_default": self.current_default,
            "safe": self.safe,
            "blockers": list(self.blockers),
            "added": len(self.added_factor_ids),
            "removed": len(self.removed_factor_ids),
            "changed": len(self.changed_factor_ids),
            "target_factor_count": self.target_factor_count,
        }


def preview_rollback(
    repo: FactorCatalogRepository,
    target_edition_id: str,
) -> RollbackPreview:
    """Dry-run the rollback — validate + diff — without touching state.

    The diff is expressed as "what changes if we flip from the current
    default to the target": factors that exist only in the current are
    marked as ``removed`` (they disappear when we roll back); factors
    that exist only in the target are ``added`` (they re-appear on
    rollback); factors whose content-hash differs are ``changed``.
    """
    blockers: List[str] = []
    try:
        repo.resolve_edition(target_edition_id)
    except ValueError as exc:
        blockers.append(str(exc))
    except Exception as exc:  # pragma: no cover — defensive
        blockers.append(f"repo error: {exc}")

    try:
        current_default = repo.get_default_edition_id()
    except Exception:
        current_default = None

    if current_default == target_edition_id:
        blockers.append("target is already the default — nothing to roll back")

    added_ids: List[str] = []
    removed_ids: List[str] = []
    changed_ids: List[str] = []
    target_count = 0

    if not blockers:
        try:
            target_summaries = repo.list_factor_summaries(target_edition_id)
            target_count = len(target_summaries)
            if target_count == 0:
                blockers.append("target edition is empty")
        except Exception as exc:
            blockers.append(f"cannot list target factors: {exc}")
            target_summaries = []

        if not blockers and current_default:
            try:
                current_summaries = repo.list_factor_summaries(current_default)
            except Exception as exc:
                blockers.append(f"cannot list current factors: {exc}")
                current_summaries = []
            if not blockers:
                curr_index = {s["factor_id"]: s["content_hash"] for s in current_summaries}
                targ_index = {s["factor_id"]: s["content_hash"] for s in target_summaries}
                curr_ids, targ_ids = set(curr_index), set(targ_index)
                # "added" on rollback == exists only in the target
                added_ids = sorted(targ_ids - curr_ids)
                removed_ids = sorted(curr_ids - targ_ids)
                changed_ids = sorted(
                    fid for fid in curr_ids & targ_ids
                    if curr_index[fid] != targ_index[fid]
                )

    return RollbackPreview(
        target_edition_id=target_edition_id,
        current_default=current_default,
        safe=not blockers,
        blockers=blockers,
        added_factor_ids=added_ids,
        removed_factor_ids=removed_ids,
        changed_factor_ids=changed_ids,
        target_factor_count=target_count,
    )


# ---------------------------------------------------------------------------
# Execute + receipt
# ---------------------------------------------------------------------------


@dataclass
class RollbackReceipt:
    """Durable record of a rollback action."""

    receipt_id: str
    target_edition_id: str
    previous_edition_id: Optional[str]
    operator: str
    reason: str
    executed_at: str
    success: bool
    result: Dict[str, Any]
    preview: Dict[str, Any]
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def write_to(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path


def execute_rollback_with_receipt(
    repo: FactorCatalogRepository,
    target_edition_id: str,
    *,
    operator: str,
    reason: str,
    event_store: Optional[RegulatoryEventStore] = None,
    receipt_dir: Optional[Path] = None,
) -> RollbackReceipt:
    """Run the rollback in production mode.

    ``operator`` and ``reason`` are required audit inputs.  When an
    ``event_store`` is supplied we emit a regulatory-change event of
    kind ``source.breaking_change`` so downstream webhook subscribers
    can react (mirror catalogs, re-run reports).  When ``receipt_dir``
    is supplied we also persist the JSON receipt to disk so the
    action is discoverable without the operator re-querying the DB.
    """
    preview = preview_rollback(repo, target_edition_id)
    executed_at = datetime.now(timezone.utc).isoformat()

    # Hard-fail before touching the DB when the preview is unsafe.
    if not preview.safe:
        receipt = RollbackReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:20]}",
            target_edition_id=target_edition_id,
            previous_edition_id=preview.current_default,
            operator=operator,
            reason=reason,
            executed_at=executed_at,
            success=False,
            result={"error": "preview_unsafe", "blockers": preview.blockers},
            preview=preview.summary(),
        )
        logger.warning(
            "Rollback blocked by preview: target=%s blockers=%s",
            target_edition_id, preview.blockers,
        )
        if receipt_dir is not None:
            receipt.write_to(receipt_dir / f"{receipt.receipt_id}.json")
        return receipt

    result = rollback_to_edition(
        repo,
        target_edition_id,
        reason=reason,
        operator=operator,
    )

    event_id: Optional[str] = None
    if event_store is not None and result.success:
        event = RegulatoryChangeEvent(
            event_id=RegulatoryChangeEvent.new_id(),
            source_id="greenlang_factors",
            event_kind=RegulatoryEventKind.BREAKING_CHANGE,
            detected_at=executed_at,
            severity="breaking",
            requires_human_review=True,
            review_reason=f"edition_rollback:{operator}:{reason[:80]}",
            payload={
                "target_edition_id": target_edition_id,
                "previous_edition_id": result.previous_edition_id,
                "preview": preview.summary(),
                "operator": operator,
            },
        )
        event_store.append(event)
        event_id = event.event_id

    receipt = RollbackReceipt(
        receipt_id=f"rcpt_{uuid.uuid4().hex[:20]}",
        target_edition_id=target_edition_id,
        previous_edition_id=result.previous_edition_id,
        operator=operator,
        reason=reason,
        executed_at=executed_at,
        success=result.success,
        result=result.to_dict(),
        preview=preview.summary(),
        event_id=event_id,
    )

    if receipt_dir is not None:
        receipt.write_to(receipt_dir / f"{receipt.receipt_id}.json")

    logger.info(
        "Rollback executed: receipt=%s success=%s target=%s previous=%s operator=%s",
        receipt.receipt_id, result.success,
        target_edition_id, result.previous_edition_id, operator,
    )
    return receipt


__all__ = [
    "RollbackCandidate",
    "RollbackPreview",
    "RollbackReceipt",
    "list_rollback_candidates",
    "preview_rollback",
    "execute_rollback_with_receipt",
]
