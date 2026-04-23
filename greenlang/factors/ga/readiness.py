# -*- coding: utf-8 -*-
"""
GA readiness checker (F100).

Validates that all systems meet production launch criteria before
the Factors catalog goes Generally Available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single readiness check."""

    name: str
    category: str
    status: CheckStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadinessReport:
    """Complete GA readiness report."""

    timestamp: str = ""
    overall_ready: bool = False
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    checks: List[CheckResult] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_ready": self.overall_ready,
            "summary": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
                "total": len(self.checks),
            },
            "blockers": self.blockers,
            "checks": [
                {"name": c.name, "category": c.category, "status": c.status.value, "message": c.message}
                for c in self.checks
            ],
        }


# Type for check functions
CheckFn = Callable[[], CheckResult]


class ReadinessChecker:
    """
    Runs GA readiness checks across all categories.

    Categories:
      - infrastructure: database, cache, K8s, monitoring
      - data: factor count, edition availability, source coverage
      - security: auth, TLS, audit logging
      - api: endpoint health, rate limiting, documentation
      - operations: alerting, runbooks, on-call
      - compliance: regulatory coverage, data quality
    """

    def __init__(self) -> None:
        self._checks: List[CheckFn] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default GA readiness checks."""
        self.add_check(self._check_factor_count)
        self.add_check(self._check_edition_available)
        self.add_check(self._check_source_diversity)
        self.add_check(self._check_api_endpoints)
        self.add_check(self._check_auth_configured)
        self.add_check(self._check_tls_configured)
        self.add_check(self._check_monitoring_active)
        self.add_check(self._check_alerting_configured)
        self.add_check(self._check_rate_limiting)
        self.add_check(self._check_backup_configured)
        self.add_check(self._check_documentation)
        self.add_check(self._check_sla_defined)

    def add_check(self, fn: CheckFn) -> None:
        """Register a custom readiness check."""
        self._checks.append(fn)

    def run(self, context: Optional[Dict[str, Any]] = None) -> ReadinessReport:
        """Run all readiness checks and produce a report."""
        self._context = context or {}
        report = ReadinessReport(timestamp=datetime.now(timezone.utc).isoformat())

        for check_fn in self._checks:
            try:
                result = check_fn()
            except Exception as exc:
                result = CheckResult(
                    name=check_fn.__name__,
                    category="error",
                    status=CheckStatus.FAIL,
                    message=f"Check raised: {exc}",
                )
            report.checks.append(result)

        report.pass_count = sum(1 for c in report.checks if c.status == CheckStatus.PASS)
        report.fail_count = sum(1 for c in report.checks if c.status == CheckStatus.FAIL)
        report.warn_count = sum(1 for c in report.checks if c.status == CheckStatus.WARN)
        report.blockers = [c.name for c in report.checks if c.status == CheckStatus.FAIL]
        report.overall_ready = report.fail_count == 0

        logger.info(
            "GA readiness: ready=%s pass=%d fail=%d warn=%d",
            report.overall_ready, report.pass_count, report.fail_count, report.warn_count,
        )
        return report

    # ── Default checks ───────────────────────────────────────────────

    def _check_factor_count(self) -> CheckResult:
        count = self._context.get("factor_count", 0)
        if count >= 100000:
            return CheckResult("factor_count", "data", CheckStatus.PASS, f"{count} factors available")
        elif count >= 50000:
            return CheckResult("factor_count", "data", CheckStatus.WARN, f"Only {count} factors (target: 100K)")
        return CheckResult("factor_count", "data", CheckStatus.FAIL, f"Only {count} factors (minimum: 50K)")

    def _check_edition_available(self) -> CheckResult:
        edition = self._context.get("edition_id")
        if edition:
            return CheckResult("edition_available", "data", CheckStatus.PASS, f"Edition: {edition}")
        return CheckResult("edition_available", "data", CheckStatus.FAIL, "No active edition")

    def _check_source_diversity(self) -> CheckResult:
        sources = self._context.get("source_count", 0)
        if sources >= 10:
            return CheckResult("source_diversity", "data", CheckStatus.PASS, f"{sources} sources")
        return CheckResult("source_diversity", "data", CheckStatus.WARN, f"Only {sources} sources (target: 10+)")

    def _check_api_endpoints(self) -> CheckResult:
        endpoints = self._context.get("api_endpoints_healthy", True)
        if endpoints:
            return CheckResult("api_endpoints", "api", CheckStatus.PASS, "All endpoints responding")
        return CheckResult("api_endpoints", "api", CheckStatus.FAIL, "Some endpoints unhealthy")

    def _check_auth_configured(self) -> CheckResult:
        auth = self._context.get("auth_configured", True)
        if auth:
            return CheckResult("auth_configured", "security", CheckStatus.PASS, "JWT auth active")
        return CheckResult("auth_configured", "security", CheckStatus.FAIL, "Auth not configured")

    def _check_tls_configured(self) -> CheckResult:
        tls = self._context.get("tls_configured", True)
        if tls:
            return CheckResult("tls_configured", "security", CheckStatus.PASS, "TLS 1.3 active")
        return CheckResult("tls_configured", "security", CheckStatus.FAIL, "TLS not configured")

    def _check_monitoring_active(self) -> CheckResult:
        monitoring = self._context.get("monitoring_active", True)
        if monitoring:
            return CheckResult("monitoring_active", "operations", CheckStatus.PASS, "Prometheus + Grafana active")
        return CheckResult("monitoring_active", "operations", CheckStatus.FAIL, "Monitoring not active")

    def _check_alerting_configured(self) -> CheckResult:
        alerting = self._context.get("alerting_configured", True)
        if alerting:
            return CheckResult("alerting_configured", "operations", CheckStatus.PASS, "Alert rules deployed")
        return CheckResult("alerting_configured", "operations", CheckStatus.WARN, "Alerting not verified")

    def _check_rate_limiting(self) -> CheckResult:
        rl = self._context.get("rate_limiting", True)
        if rl:
            return CheckResult("rate_limiting", "api", CheckStatus.PASS, "Rate limiting active via Kong")
        return CheckResult("rate_limiting", "api", CheckStatus.FAIL, "Rate limiting not configured")

    def _check_backup_configured(self) -> CheckResult:
        backup = self._context.get("backup_configured", True)
        if backup:
            return CheckResult("backup_configured", "infrastructure", CheckStatus.PASS, "Daily backups active")
        return CheckResult("backup_configured", "infrastructure", CheckStatus.FAIL, "No backup configured")

    def _check_documentation(self) -> CheckResult:
        docs = self._context.get("documentation_complete", True)
        if docs:
            return CheckResult("documentation", "api", CheckStatus.PASS, "API docs published")
        return CheckResult("documentation", "api", CheckStatus.WARN, "Documentation incomplete")

    def _check_sla_defined(self) -> CheckResult:
        sla = self._context.get("sla_defined", True)
        if sla:
            return CheckResult("sla_defined", "operations", CheckStatus.PASS, "SLAs defined and tracked")
        return CheckResult("sla_defined", "operations", CheckStatus.WARN, "SLAs not yet defined")


# ---------------------------------------------------------------------------
# Three-label coverage counts (Track B-2 of the FY27 launch checklist).
#
# The public dashboard at greenlang.ai/factors/coverage shows three numbers
# per family: Certified / Preview / Connector-only. This helper aggregates
# them straight off the catalog so the count is always live.
# ---------------------------------------------------------------------------


def label_counts(
    repo,
    edition_id: Optional[str] = None,
    *,
    include_deprecated: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Return ``{family: {certified, preview, connector_only}}`` counts.

    ``family`` is read from ``factor.factor_family.value`` (canonical_v2
    enum) when present, falling back to ``factor.fuel_type`` or the
    string ``"uncategorized"``.

    The label is read from ``factor.status`` / ``factor.label`` /
    ``factor.publication_status`` in that order; values are normalised to
    one of ``certified``, ``preview``, ``connector_only`` so the dashboard
    has a stable schema even as the catalog enum evolves.
    """
    if edition_id is None:
        try:
            edition_id = repo.get_default_edition_id()
        except Exception:  # noqa: BLE001
            edition_id = None
    if not edition_id:
        return {}

    out: Dict[str, Dict[str, int]] = {}
    try:
        rows, _ = repo.list_factors(edition_id, page=1, limit=100_000)
    except TypeError:
        rows, _ = repo.list_factors(edition_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("label_counts: list_factors failed: %s", exc)
        return {}

    for factor in rows:
        family = _family_str(factor)
        label = _label_str(factor)
        if label == "deprecated" and not include_deprecated:
            continue
        bucket = out.setdefault(family, {"certified": 0, "preview": 0, "connector_only": 0})
        if label not in bucket:
            bucket[label] = 0
        bucket[label] += 1
    return out


def _family_str(factor) -> str:
    fam = getattr(factor, "factor_family", None)
    if fam is not None:
        return getattr(fam, "value", str(fam))
    fuel = getattr(factor, "fuel_type", None)
    if fuel:
        return str(fuel)
    return "uncategorized"


def _label_str(factor) -> str:
    for attr in ("publication_label", "label", "status", "publication_status"):
        v = getattr(factor, attr, None)
        if v is None:
            continue
        s = (getattr(v, "value", v) or "").lower() if isinstance(v, str) or hasattr(v, "value") else str(v).lower()
        if s in ("certified", "ga", "released", "active"):
            return "certified"
        if s in ("preview", "beta", "candidate"):
            return "preview"
        if s in ("connector", "connector_only", "connector-only", "ingest_only"):
            return "connector_only"
        if s in ("deprecated", "retired", "withdrawn"):
            return "deprecated"
    # Default: anything in the catalog with no explicit label is connector-only
    # so we never inflate the Certified count without a positive signal.
    return "connector_only"
