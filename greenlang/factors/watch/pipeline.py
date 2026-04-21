# -*- coding: utf-8 -*-
"""Integrated regulatory-watch pipeline.

Ties the existing building blocks together into a single callable that
can be scheduled (cron, Celery beat, K8s CronJob):

    scheduler.run_watch                     # poll every source
        -> for each source that changed:
             change_detector.detect_source_change   # per-factor diff
             change_classification.classify_change  # per-factor severity
             build RegulatoryChangeEvent + persist  # event stream
             dispatch customer webhooks             # factor.updated, etc.
             feed approval_gate + release pipeline  # certified re-review

The pipeline is deliberately composed of pure functions so each leg can
be unit-tested in isolation — the orchestrator below wires optional
hooks (``factor_fetcher``, ``parser_fetcher``) so callers can inject
their own parsing strategy for each source without modifying this
module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from greenlang.factors.source_registry import (
    SourceRegistryEntry,
    registry_by_id,
)
from greenlang.factors.watch.change_detector import (
    ChangeReport,
    detect_source_change,
)
from greenlang.factors.watch.regulatory_events import (
    RegulatoryChangeEvent,
    RegulatoryEventKind,
    RegulatoryEventStore,
    build_artifact_change_event,
    build_factor_event,
    build_source_unavailable_event,
)
from greenlang.factors.watch.scheduler import WatchResult, run_watch
from greenlang.factors.webhooks import (
    WebhookEvent,
    WebhookEventType,
    WebhookRegistry,
    dispatch_event,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetcher callback types
# ---------------------------------------------------------------------------


FactorFetcher = Callable[[str], List[Dict[str, Any]]]
"""Return the list of factor dicts currently stored for ``source_id``."""

ParserFetcher = Callable[[SourceRegistryEntry, WatchResult], List[Dict[str, Any]]]
"""Download and parse a new artifact into factor dicts."""


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineSourceResult:
    """What happened for one source in a single pipeline cycle."""

    source_id: str
    watch_result: WatchResult
    change_report: Optional[ChangeReport] = None
    events: List[RegulatoryChangeEvent] = field(default_factory=list)
    webhook_receipts: List[Dict[str, Any]] = field(default_factory=list)
    approval_gate_triggered: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineCycleResult:
    """Aggregate result of one pipeline invocation."""

    started_at: str
    finished_at: str
    sources_checked: int = 0
    sources_changed: int = 0
    sources_errored: int = 0
    events_emitted: int = 0
    webhooks_delivered: int = 0
    webhooks_failed: int = 0
    per_source: List[PipelineSourceResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sources_checked": self.sources_checked,
            "sources_changed": self.sources_changed,
            "sources_errored": self.sources_errored,
            "events_emitted": self.events_emitted,
            "webhooks_delivered": self.webhooks_delivered,
            "webhooks_failed": self.webhooks_failed,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_FACTOR_EVENT_FOR_KIND = {
    "added": RegulatoryEventKind.FACTOR_ADDED,
    "removed": RegulatoryEventKind.FACTOR_REMOVED,
    "modified": RegulatoryEventKind.FACTOR_UPDATED,
}

_WEBHOOK_FOR_EVENT = {
    RegulatoryEventKind.SOURCE_ARTIFACT_CHANGED: WebhookEventType.SOURCE_ARTIFACT_CHANGED,
    RegulatoryEventKind.SOURCE_UNAVAILABLE: WebhookEventType.SOURCE_UNAVAILABLE,
    RegulatoryEventKind.FACTOR_ADDED: WebhookEventType.FACTOR_ADDED,
    RegulatoryEventKind.FACTOR_UPDATED: WebhookEventType.FACTOR_UPDATED,
    RegulatoryEventKind.FACTOR_REMOVED: WebhookEventType.FACTOR_REMOVED,
    RegulatoryEventKind.FACTOR_DEPRECATED: WebhookEventType.FACTOR_DEPRECATED,
    RegulatoryEventKind.LICENSE_CHANGED: WebhookEventType.LICENSE_CHANGED,
    RegulatoryEventKind.METHODOLOGY_CHANGED: WebhookEventType.METHODOLOGY_CHANGED,
    RegulatoryEventKind.BREAKING_CHANGE: WebhookEventType.SOURCE_BREAKING_CHANGE,
}


def _events_from_change_report(
    source_id: str,
    report: ChangeReport,
) -> List[RegulatoryChangeEvent]:
    """Build per-factor regulatory events from a ``ChangeReport``."""
    out: List[RegulatoryChangeEvent] = []
    for bucket in (report.added, report.removed, report.modified):
        for entry in bucket:
            kind = _FACTOR_EVENT_FOR_KIND.get(entry.change_kind)
            if not kind:
                continue
            severity = "breaking" if (
                entry.change_kind == "removed"
                or entry.field_changes.get("unit_changed")
            ) else "info"
            out.append(
                build_factor_event(
                    source_id=source_id,
                    kind=kind,
                    factor_id=entry.factor_id,
                    old_value=entry.old_value,
                    new_value=entry.new_value,
                    unit=entry.field_changes.get("unit", {}).get("new") if isinstance(entry.field_changes.get("unit"), dict) else None,
                    severity=severity,
                    payload={"field_changes": entry.field_changes} if entry.field_changes else {},
                )
            )
    if report.has_breaking_changes:
        out.append(
            RegulatoryChangeEvent(
                event_id=RegulatoryChangeEvent.new_id(),
                source_id=source_id,
                event_kind=RegulatoryEventKind.BREAKING_CHANGE,
                detected_at=datetime.now(timezone.utc).isoformat(),
                severity="breaking",
                requires_human_review=True,
                review_reason=report.review_reason or "breaking_change_detected",
                payload=report.to_dict(),
            )
        )
    return out


def _dispatch_events_as_webhooks(
    events: Sequence[RegulatoryChangeEvent],
    webhook_registry: Optional[WebhookRegistry],
    *,
    delivery_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if webhook_registry is None or not events:
        return []
    receipts: List[Dict[str, Any]] = []
    for ev in events:
        wh_type = _WEBHOOK_FOR_EVENT.get(ev.event_kind)
        if not wh_type or wh_type not in WebhookEventType.ALL:
            continue
        wh_event = WebhookEvent(event_type=wh_type, payload=ev.to_webhook_payload())
        receipts.extend(
            dispatch_event(
                webhook_registry,
                wh_event,
                **(delivery_kwargs or {}),
            )
        )
    return receipts


def _maybe_trigger_approval_gate(
    entry: SourceRegistryEntry,
    events: Sequence[RegulatoryChangeEvent],
) -> bool:
    """Return True when we ought to re-run certified promotion for this source.

    We only inform the gate that a review is needed — the gate itself
    enforces the decision when the next edition is prepared.  This keeps
    the pipeline side-effect-light and avoids racing with concurrent
    release preparation.
    """
    if not entry.approval_required_for_certified:
        return False
    breaking = any(
        ev.severity == "breaking"
        or ev.event_kind in (
            RegulatoryEventKind.BREAKING_CHANGE,
            RegulatoryEventKind.LICENSE_CHANGED,
            RegulatoryEventKind.METHODOLOGY_CHANGED,
            RegulatoryEventKind.FACTOR_REMOVED,
        )
        for ev in events
    )
    if breaking:
        logger.warning(
            "Approval-gate review triggered for source=%s due to breaking/license change",
            entry.source_id,
        )
    return breaking


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_regulatory_watch_cycle(
    *,
    registry_path: Optional[Path] = None,
    watch_db_path: Optional[Path] = None,
    event_store: Optional[RegulatoryEventStore] = None,
    webhook_registry: Optional[WebhookRegistry] = None,
    factor_fetcher: Optional[FactorFetcher] = None,
    parser_fetcher: Optional[ParserFetcher] = None,
    notify: Optional[Callable[[str, WatchResult], None]] = None,
    delivery_kwargs: Optional[Dict[str, Any]] = None,
) -> PipelineCycleResult:
    """Run one end-to-end watch cycle.

    Parameters
    ----------
    registry_path :
        Override path to ``source_registry.yaml``.
    watch_db_path :
        SQLite path where the scheduler stores watch_results rows.  The
        scheduler uses this file to compute the previous hash.
    event_store :
        Regulatory-event sink.  If omitted, per-factor events are built
        in memory and returned but not persisted.
    webhook_registry :
        Customer webhook subscriptions.  When provided, each event is
        fanned out to matching subscribers.
    factor_fetcher / parser_fetcher :
        Inject how to read the "current" factor snapshot and parse the
        fresh artifact.  When either is ``None`` the pipeline records an
        artifact-level event but skips per-factor diffing.
    notify :
        Optional callback forwarded to ``scheduler.run_watch`` so Slack /
        email notifications keep working unchanged.
    delivery_kwargs :
        Forwarded to ``dispatch_event`` (useful for tests that stub
        ``transport`` / ``sleep``).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    reg = registry_by_id(registry_path)

    # Step 1: poll every source (this already persists watch_results rows
    # and computes content-hash deltas).
    watch_results = run_watch(
        registry_path=registry_path,
        db_path=watch_db_path,
        notify=notify,
        store=watch_db_path is not None,
    )

    cycle = PipelineCycleResult(
        started_at=started_at,
        finished_at=started_at,  # will be overwritten at the end
        sources_checked=len(watch_results),
    )

    for wr in watch_results:
        entry = reg.get(wr.source_id)
        per = PipelineSourceResult(source_id=wr.source_id, watch_result=wr)

        # Unreachable / 4xx-5xx: emit a source-unavailable event.
        if wr.error_message or (wr.http_status is not None and wr.http_status >= 400):
            cycle.sources_errored += 1
            ev = build_source_unavailable_event(
                source_id=wr.source_id,
                url=wr.url,
                error=wr.error_message or f"http_{wr.http_status}",
            )
            per.events.append(ev)

        # Content changed: emit artifact-level event + optional per-factor diff.
        if wr.change_detected and wr.change_type == "content_changed":
            cycle.sources_changed += 1
            artifact_event = build_artifact_change_event(
                source_id=wr.source_id,
                artifact_hash_old=wr.previous_hash,
                artifact_hash_new=wr.file_hash,
                url=wr.url,
            )
            per.events.append(artifact_event)

            # Optional: per-factor diffing when callers wire it in.
            if factor_fetcher is not None and parser_fetcher is not None and entry is not None:
                try:
                    old_factors = factor_fetcher(entry.source_id)
                    new_factors = parser_fetcher(entry, wr)
                    report = detect_source_change(
                        entry.source_id,
                        old_factors,
                        new_factors,
                        artifact_hash_old=wr.previous_hash,
                        artifact_hash_new=wr.file_hash,
                    )
                    per.change_report = report
                    per.events.extend(_events_from_change_report(entry.source_id, report))
                    if report.requires_human_review:
                        artifact_event.requires_human_review = True
                        artifact_event.review_reason = report.review_reason
                        artifact_event.severity = "warning"
                except Exception as exc:  # parser failure should not kill the cycle
                    logger.exception(
                        "Per-factor diff failed for source=%s", entry.source_id
                    )
                    per.errors.append(f"diff_failed:{exc}")

        # Persist events in the durable stream.
        if event_store is not None and per.events:
            event_store.append_many(per.events)

        # Fan out customer webhooks (retrying with exponential backoff).
        receipts = _dispatch_events_as_webhooks(
            per.events,
            webhook_registry,
            delivery_kwargs=delivery_kwargs,
        )
        per.webhook_receipts = receipts
        for rec in receipts:
            if rec.get("status") == "ok":
                cycle.webhooks_delivered += 1
            else:
                cycle.webhooks_failed += 1

        # Tell the approval gate a re-review is warranted.
        if entry is not None:
            per.approval_gate_triggered = _maybe_trigger_approval_gate(entry, per.events)

        cycle.events_emitted += len(per.events)
        cycle.per_source.append(per)

    cycle.finished_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Regulatory watch cycle complete: %s",
        cycle.summary(),
    )
    return cycle


__all__ = [
    "FactorFetcher",
    "ParserFetcher",
    "PipelineCycleResult",
    "PipelineSourceResult",
    "run_regulatory_watch_cycle",
]
