# -*- coding: utf-8 -*-
"""
Data-retention policy engine for the Factors API (SEC-5).

Business rules (owner: GreenLang Legal + Security, last reviewed 2026-04-23):

    =============== =========== ============= ============= ============
    Tier            logs (d)    signed_rcpt   explain_hist  customer_priv
    =============== =========== ============= ============= ============
    community       30          90            30            none
    pro             90          365           90            indefinite*
    platform        365         1095          365           indefinite*
    enterprise      2555 (7y)   2555          unlimited     indefinite*
    =============== =========== ============= ============= ============

    *indefinite* for customer_private_factors means we honor the tenant's
    retention setting — we never purge private factors without an explicit
    tenant instruction or an open data-subject-deletion request.

    Raw artifacts (upstream PDFs, ingested CSVs, watch receipts) have a
    hard 10-year compliance baseline that applies across all tiers.

Cron entry-point: ``factors_retention_purge`` — scheduled via
``deployment/k8s/factors/base/cronjob-retention.yaml``. A single run is
idempotent: scanning a resource twice in the same window produces the same
set of purges; items already purged are skipped.

Manual purge: Enterprise customers can invoke :func:`purge_tenant_data`
via the admin API to satisfy GDPR art. 17 / CCPA "right to delete"
requests. The manual endpoint writes a tamper-evident audit record
through the existing :mod:`greenlang.factors.security.audit` pipeline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier + resource enums
# ---------------------------------------------------------------------------

class RetentionTier(str, Enum):
    COMMUNITY = "community"
    PRO = "pro"
    PLATFORM = "platform"
    ENTERPRISE = "enterprise"


class RetentionResource(str, Enum):
    LOGS = "logs"
    SIGNED_RECEIPTS = "signed_receipts"
    EXPLAIN_HISTORY = "explain_history"
    CUSTOMER_PRIVATE = "customer_private_factors"
    RAW_ARTIFACTS = "raw_artifacts"


class PurgeReason(str, Enum):
    RETENTION = "retention"           # automatic cron purge
    TENANT_DELETION = "tenant_deletion"
    GDPR_ART17 = "gdpr_art17"         # right to erasure
    CCPA_DELETE = "ccpa_delete"
    MANUAL_ADMIN = "manual_admin"


# ---------------------------------------------------------------------------
# Policy table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetentionRule:
    resource: RetentionResource
    days: Optional[int]  # None = indefinite

    def is_indefinite(self) -> bool:
        return self.days is None


# Per-tier retention days. ``None`` means indefinite.
TIER_POLICY: Dict[RetentionTier, Dict[RetentionResource, Optional[int]]] = {
    RetentionTier.COMMUNITY: {
        RetentionResource.LOGS: 30,
        RetentionResource.SIGNED_RECEIPTS: 90,
        RetentionResource.EXPLAIN_HISTORY: 30,
        RetentionResource.CUSTOMER_PRIVATE: None,
        RetentionResource.RAW_ARTIFACTS: 3650,
    },
    RetentionTier.PRO: {
        RetentionResource.LOGS: 90,
        RetentionResource.SIGNED_RECEIPTS: 365,
        RetentionResource.EXPLAIN_HISTORY: 90,
        RetentionResource.CUSTOMER_PRIVATE: None,
        RetentionResource.RAW_ARTIFACTS: 3650,
    },
    RetentionTier.PLATFORM: {
        RetentionResource.LOGS: 365,
        RetentionResource.SIGNED_RECEIPTS: 1095,
        RetentionResource.EXPLAIN_HISTORY: 365,
        RetentionResource.CUSTOMER_PRIVATE: None,
        RetentionResource.RAW_ARTIFACTS: 3650,
    },
    RetentionTier.ENTERPRISE: {
        RetentionResource.LOGS: 2555,
        RetentionResource.SIGNED_RECEIPTS: 2555,
        RetentionResource.EXPLAIN_HISTORY: None,
        RetentionResource.CUSTOMER_PRIVATE: None,
        RetentionResource.RAW_ARTIFACTS: 3650,
    },
}


def get_rule(tier: str, resource: RetentionResource) -> RetentionRule:
    try:
        t = RetentionTier(tier)
    except ValueError:
        t = RetentionTier.COMMUNITY
    days = TIER_POLICY[t].get(resource)
    return RetentionRule(resource=resource, days=days)


# ---------------------------------------------------------------------------
# Resource adapters
# ---------------------------------------------------------------------------


class ResourceAdapter:
    """Interface that the purge engine uses to read + delete records.

    Callers plug in one adapter per logical resource. Adapters wrap the
    actual storage backend (Postgres, S3, Loki, etc.) and expose only the
    two operations the engine needs: count + purge older than cutoff.

    ``purge_older_than`` MUST be idempotent — re-running it over the same
    cutoff produces no additional state change after the first call. This
    is what makes the cron safe to re-run.
    """

    resource: RetentionResource = RetentionResource.LOGS  # overridden

    def purge_older_than(
        self,
        tenant_id: Optional[str],
        cutoff: datetime,
        reason: PurgeReason,
    ) -> int:
        """Return the number of rows purged (0 if nothing to do)."""
        raise NotImplementedError

    def purge_all_for_tenant(self, tenant_id: str, reason: PurgeReason) -> int:
        """Full cascade. Used for tenant-deletion / GDPR art 17."""
        raise NotImplementedError


# In-memory reference adapter for tests. Production wires SQL/S3 adapters.
class InMemoryResourceAdapter(ResourceAdapter):
    def __init__(self, resource: RetentionResource) -> None:
        self.resource = resource
        self.records: List[Dict[str, Any]] = []

    def add(self, tenant_id: str, created_at: datetime, record_id: Optional[str] = None) -> None:
        self.records.append(
            {"tenant_id": tenant_id, "created_at": created_at, "id": record_id}
        )

    def purge_older_than(self, tenant_id, cutoff, reason):
        before = len(self.records)
        self.records = [
            r
            for r in self.records
            if (tenant_id is not None and r["tenant_id"] != tenant_id)
            or r["created_at"] >= cutoff
        ]
        return before - len(self.records)

    def purge_all_for_tenant(self, tenant_id, reason):
        before = len(self.records)
        self.records = [r for r in self.records if r["tenant_id"] != tenant_id]
        return before - len(self.records)


# ---------------------------------------------------------------------------
# Metric hooks (bind Prometheus at call site)
# ---------------------------------------------------------------------------

MetricHook = Callable[[RetentionResource, str, PurgeReason, int], None]


def _noop_metric(*_args, **_kwargs) -> None:
    return None


_METRIC_HOOK: MetricHook = _noop_metric


def set_metric_hook(hook: MetricHook) -> None:
    """Install a metric emitter. Called by the Prometheus exporter at boot."""
    global _METRIC_HOOK
    _METRIC_HOOK = hook


# ---------------------------------------------------------------------------
# Audit hooks
# ---------------------------------------------------------------------------

AuditHook = Callable[[Dict[str, Any]], None]


def _noop_audit(_evt: Dict[str, Any]) -> None:
    return None


_AUDIT_HOOK: AuditHook = _noop_audit


def set_audit_hook(hook: AuditHook) -> None:
    global _AUDIT_HOOK
    _AUDIT_HOOK = hook


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class TenantTierResolver:
    """Minimal per-tenant tier lookup. Swap with a DB-backed version."""
    static: Dict[str, str] = field(default_factory=dict)

    def tier(self, tenant_id: str) -> str:
        return (self.static.get(tenant_id) or "community").lower()


@dataclass
class PurgeReport:
    resource: RetentionResource
    tenant_id: Optional[str]
    tier: str
    reason: PurgeReason
    cutoff: datetime
    purged_rows: int
    duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource.value,
            "tenant_id": self.tenant_id,
            "tier": self.tier,
            "reason": self.reason.value,
            "cutoff": self.cutoff.isoformat(),
            "purged_rows": self.purged_rows,
            "duration_ms": self.duration_ms,
        }


class RetentionEngine:
    """Runs scheduled purges and honors manual deletion requests."""

    def __init__(
        self,
        adapters: Dict[RetentionResource, ResourceAdapter],
        resolver: TenantTierResolver,
    ) -> None:
        self._adapters = adapters
        self._resolver = resolver

    # ------------------------------------------------------------------
    def run_scheduled_purge(
        self,
        tenants: Iterable[str],
        now: Optional[datetime] = None,
    ) -> List[PurgeReport]:
        """Iterate every (tenant x resource) pair and purge stale rows.

        Idempotent. Emits one :class:`PurgeReport` per (tenant, resource)
        pair, even when ``purged_rows == 0`` — downstream dashboards rely
        on the "zero-delete" signal to confirm the cron ran to completion.
        """
        now = now or datetime.now(timezone.utc)
        reports: List[PurgeReport] = []
        for tenant_id in tenants:
            tier = self._resolver.tier(tenant_id)
            for resource, adapter in self._adapters.items():
                rule = get_rule(tier, resource)
                if rule.is_indefinite():
                    _METRIC_HOOK(resource, tier, PurgeReason.RETENTION, 0)
                    continue
                cutoff = now - timedelta(days=rule.days)  # type: ignore[arg-type]
                start = time.monotonic()
                purged = adapter.purge_older_than(
                    tenant_id=tenant_id,
                    cutoff=cutoff,
                    reason=PurgeReason.RETENTION,
                )
                dur_ms = int((time.monotonic() - start) * 1000)
                report = PurgeReport(
                    resource=resource,
                    tenant_id=tenant_id,
                    tier=tier,
                    reason=PurgeReason.RETENTION,
                    cutoff=cutoff,
                    purged_rows=purged,
                    duration_ms=dur_ms,
                )
                reports.append(report)
                _METRIC_HOOK(resource, tier, PurgeReason.RETENTION, purged)
                _AUDIT_HOOK(
                    {
                        "event": "retention.cron.purge",
                        "timestamp": now.isoformat(),
                        **report.to_dict(),
                    }
                )
        return reports

    # ------------------------------------------------------------------
    def purge_tenant_data(
        self,
        tenant_id: str,
        reason: PurgeReason = PurgeReason.TENANT_DELETION,
        resources: Optional[Iterable[RetentionResource]] = None,
    ) -> List[PurgeReport]:
        """Cascade-delete every resource we hold for ``tenant_id``.

        Used for tenant offboarding, GDPR art. 17, and CCPA deletes.
        Writes a signed audit record so Legal can produce evidence that
        the request was honored.
        """
        now = datetime.now(timezone.utc)
        tier = self._resolver.tier(tenant_id)
        targets = list(resources) if resources else list(self._adapters.keys())
        reports: List[PurgeReport] = []
        for resource in targets:
            adapter = self._adapters.get(resource)
            if adapter is None:
                continue
            start = time.monotonic()
            purged = adapter.purge_all_for_tenant(tenant_id=tenant_id, reason=reason)
            dur_ms = int((time.monotonic() - start) * 1000)
            report = PurgeReport(
                resource=resource,
                tenant_id=tenant_id,
                tier=tier,
                reason=reason,
                cutoff=now,
                purged_rows=purged,
                duration_ms=dur_ms,
            )
            reports.append(report)
            _METRIC_HOOK(resource, tier, reason, purged)
            _AUDIT_HOOK(
                {
                    "event": "retention.manual.purge",
                    "timestamp": now.isoformat(),
                    **report.to_dict(),
                }
            )
        logger.info(
            "manual purge tenant=%s reason=%s resources=%s rows=%d",
            tenant_id,
            reason.value,
            [r.resource.value for r in reports],
            sum(r.purged_rows for r in reports),
        )
        return reports

    # ------------------------------------------------------------------
    def resurrect_if_within_retention(
        self,
        resource: RetentionResource,
        tenant_id: str,
        record_age_days: float,
    ) -> bool:
        """Return True if a record ``record_age_days`` old is still within
        retention (callers use this to decide whether to keep a soft-deleted
        row accessible during the grace window)."""
        tier = self._resolver.tier(tenant_id)
        rule = get_rule(tier, resource)
        if rule.is_indefinite():
            return True
        return record_age_days <= rule.days  # type: ignore[operator]


# ---------------------------------------------------------------------------
# CLI-style entry point for the CronJob
# ---------------------------------------------------------------------------


def factors_retention_purge(
    engine: RetentionEngine,
    tenants: Iterable[str],
) -> Dict[str, Any]:
    """Cron entry point. Returns a JSON-serialisable summary.

    Example (scheduled from a Kubernetes CronJob running daily at 04:00 UTC)::

        python -m greenlang.factors.middleware.retention \
            --tenants-file /etc/factors/tenants.txt
    """
    start = datetime.now(timezone.utc)
    reports = engine.run_scheduled_purge(tenants=tenants, now=start)
    end = datetime.now(timezone.utc)
    return {
        "job": "factors_retention_purge",
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "total_purged": sum(r.purged_rows for r in reports),
        "reports": [r.to_dict() for r in reports],
    }


# ---------------------------------------------------------------------------
# __main__ so the CronJob can run with:
#   python -m greenlang.factors.middleware.retention
# ---------------------------------------------------------------------------


def _build_engine_from_env() -> RetentionEngine:
    """Wire a default engine from environment config.

    Heavy lifting (SQL adapters, S3 adapters) is deferred to integration
    modules; the env stub falls back to in-memory adapters so the CronJob
    image boots cleanly even without live backends. A warning is emitted
    when this fallback is taken in staging/production.
    """
    env = (os.getenv("APP_ENV") or os.getenv("GL_ENV") or "").lower()
    try:
        from greenlang.factors.middleware._retention_adapters import (  # type: ignore
            build_sql_adapters,
            build_tier_resolver,
        )
        adapters = build_sql_adapters()
        resolver = build_tier_resolver()
        return RetentionEngine(adapters=adapters, resolver=resolver)
    except Exception as exc:  # noqa: BLE001
        if env in {"staging", "production", "prod"}:
            logger.warning(
                "retention engine falling back to in-memory adapters in %s: %s",
                env,
                exc,
            )
        adapters = {
            r: InMemoryResourceAdapter(r)
            for r in RetentionResource
        }
        return RetentionEngine(adapters=adapters, resolver=TenantTierResolver())


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json as _json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--tenants-file", required=False)
    args = parser.parse_args()

    tenants: List[str] = []
    if args.tenants_file and os.path.exists(args.tenants_file):
        with open(args.tenants_file, "r", encoding="utf-8") as fh:
            tenants = [line.strip() for line in fh if line.strip()]

    summary = factors_retention_purge(_build_engine_from_env(), tenants=tenants)
    _json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


__all__ = [
    "RetentionTier",
    "RetentionResource",
    "PurgeReason",
    "RetentionRule",
    "TIER_POLICY",
    "get_rule",
    "ResourceAdapter",
    "InMemoryResourceAdapter",
    "TenantTierResolver",
    "PurgeReport",
    "RetentionEngine",
    "factors_retention_purge",
    "set_metric_hook",
    "set_audit_hook",
]
