# -*- coding: utf-8 -*-
"""
Partner integration health checker for the Factors API.

Runs a series of checks to verify that a partner's integration is
functioning correctly: API connectivity, authentication, search, and
factor retrieval.

Example:
    >>> from greenlang.factors.onboarding.health_check import run_partner_health_check
    >>> report = run_partner_health_check(partner_config)
    >>> print(report.overall_status)
    pass
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Status of an individual health check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report from all partner checks."""

    partner_id: str
    timestamp: str = ""
    overall_status: str = "pass"
    checks: List[HealthCheckResult] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    warn_count: int = 0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a dictionary."""
        return {
            "partner_id": self.partner_id,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "summary": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
                "total": len(self.checks),
            },
            "total_latency_ms": round(self.total_latency_ms, 1),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": round(c.latency_ms, 1),
                }
                for c in self.checks
            ],
        }

    @property
    def is_healthy(self) -> bool:
        """Return True if no checks failed."""
        return self.fail_count == 0


def _check_api_connectivity(
    base_url: str,
    api_key: str,
    timeout_seconds: float = 10.0,
) -> HealthCheckResult:
    """Verify basic API connectivity by hitting the health endpoint.

    Args:
        base_url: API base URL.
        api_key: Partner API key.
        timeout_seconds: Request timeout.

    Returns:
        HealthCheckResult for the connectivity check.
    """
    start = time.monotonic()
    try:
        import urllib.request
        import urllib.error

        url = "%s/health" % base_url.rstrip("/")
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            latency = (time.monotonic() - start) * 1000
            status_code = resp.status
            if status_code == 200:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=CheckStatus.PASS,
                    message="API reachable at %s (HTTP %d)" % (base_url, status_code),
                    latency_ms=latency,
                )
            return HealthCheckResult(
                name="api_connectivity",
                status=CheckStatus.WARN,
                message="Unexpected status %d from health endpoint" % status_code,
                latency_ms=latency,
            )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthCheckResult(
            name="api_connectivity",
            status=CheckStatus.FAIL,
            message="Cannot reach API at %s: %s" % (base_url, exc),
            latency_ms=latency,
        )


def _check_auth_working(
    base_url: str,
    api_key: str,
    timeout_seconds: float = 10.0,
) -> HealthCheckResult:
    """Verify that authentication works with the provided API key.

    Args:
        base_url: API base URL.
        api_key: Partner API key.
        timeout_seconds: Request timeout.

    Returns:
        HealthCheckResult for the auth check.
    """
    start = time.monotonic()
    try:
        import urllib.request
        import urllib.error
        import json as json_mod

        url = "%s/v2/factors?limit=1" % base_url.rstrip("/")
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            latency = (time.monotonic() - start) * 1000
            if resp.status == 200:
                return HealthCheckResult(
                    name="auth_working",
                    status=CheckStatus.PASS,
                    message="Authentication successful",
                    latency_ms=latency,
                )
            return HealthCheckResult(
                name="auth_working",
                status=CheckStatus.FAIL,
                message="Auth returned HTTP %d" % resp.status,
                latency_ms=latency,
            )
    except urllib.error.HTTPError as exc:
        latency = (time.monotonic() - start) * 1000
        if exc.code in (401, 403):
            return HealthCheckResult(
                name="auth_working",
                status=CheckStatus.FAIL,
                message="Authentication failed (HTTP %d): check API key" % exc.code,
                latency_ms=latency,
            )
        return HealthCheckResult(
            name="auth_working",
            status=CheckStatus.WARN,
            message="Auth check got HTTP %d" % exc.code,
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthCheckResult(
            name="auth_working",
            status=CheckStatus.FAIL,
            message="Auth check failed: %s" % exc,
            latency_ms=latency,
        )


def _check_search_returns_results(
    base_url: str,
    api_key: str,
    timeout_seconds: float = 10.0,
) -> HealthCheckResult:
    """Verify that search returns non-empty results.

    Args:
        base_url: API base URL.
        api_key: Partner API key.
        timeout_seconds: Request timeout.

    Returns:
        HealthCheckResult for the search check.
    """
    start = time.monotonic()
    try:
        import urllib.request
        import urllib.parse
        import json as json_mod

        query = urllib.parse.urlencode({"q": "electricity", "limit": "5"})
        url = "%s/v2/factors/search?%s" % (base_url.rstrip("/"), query)
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            latency = (time.monotonic() - start) * 1000
            body = resp.read().decode("utf-8")
            data = json_mod.loads(body)
            factors = data.get("factors", [])
            count = len(factors)
            if count > 0:
                return HealthCheckResult(
                    name="search_returns_results",
                    status=CheckStatus.PASS,
                    message="Search returned %d factors" % count,
                    latency_ms=latency,
                    details={"result_count": count},
                )
            return HealthCheckResult(
                name="search_returns_results",
                status=CheckStatus.WARN,
                message="Search returned 0 factors (catalog may be empty)",
                latency_ms=latency,
            )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthCheckResult(
            name="search_returns_results",
            status=CheckStatus.FAIL,
            message="Search check failed: %s" % exc,
            latency_ms=latency,
        )


def _check_factor_retrieval(
    base_url: str,
    api_key: str,
    timeout_seconds: float = 10.0,
) -> HealthCheckResult:
    """Verify that individual factor retrieval works.

    First searches for a factor, then retrieves it by ID.

    Args:
        base_url: API base URL.
        api_key: Partner API key.
        timeout_seconds: Request timeout.

    Returns:
        HealthCheckResult for the retrieval check.
    """
    start = time.monotonic()
    try:
        import urllib.request
        import urllib.parse
        import json as json_mod

        # Step 1: Get a factor ID via search
        query = urllib.parse.urlencode({"q": "diesel", "limit": "1"})
        search_url = "%s/v2/factors/search?%s" % (base_url.rstrip("/"), query)
        req = urllib.request.Request(
            search_url,
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            data = json_mod.loads(body)
            factors = data.get("factors", [])
            if not factors:
                latency = (time.monotonic() - start) * 1000
                return HealthCheckResult(
                    name="factor_retrieval",
                    status=CheckStatus.SKIP,
                    message="No factors found to test retrieval",
                    latency_ms=latency,
                )
            factor_id = factors[0].get("factor_id")

        # Step 2: Retrieve the specific factor
        detail_url = "%s/v2/factors/%s" % (
            base_url.rstrip("/"),
            urllib.parse.quote(factor_id, safe=""),
        )
        req2 = urllib.request.Request(
            detail_url,
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req2, timeout=timeout_seconds) as resp2:
            latency = (time.monotonic() - start) * 1000
            body2 = resp2.read().decode("utf-8")
            factor_data = json_mod.loads(body2)
            returned_id = factor_data.get("factor_id")
            if returned_id == factor_id:
                return HealthCheckResult(
                    name="factor_retrieval",
                    status=CheckStatus.PASS,
                    message="Factor %s retrieved successfully" % factor_id,
                    latency_ms=latency,
                    details={"factor_id": factor_id},
                )
            return HealthCheckResult(
                name="factor_retrieval",
                status=CheckStatus.WARN,
                message="Factor ID mismatch: expected=%s got=%s" % (factor_id, returned_id),
                latency_ms=latency,
            )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthCheckResult(
            name="factor_retrieval",
            status=CheckStatus.FAIL,
            message="Factor retrieval failed: %s" % exc,
            latency_ms=latency,
        )


def run_partner_health_check(
    partner_config: Any,
    *,
    timeout_seconds: float = 10.0,
) -> HealthReport:
    """Run a complete health check for a partner's integration.

    Verifies:
      1. API connectivity (health endpoint)
      2. Authentication working (API key valid)
      3. Search returns results
      4. Factor retrieval works (search then get by ID)

    Args:
        partner_config: A PartnerConfig or dict with 'base_url', 'api_key',
            and 'partner_id' fields.
        timeout_seconds: Timeout for each HTTP request.

    Returns:
        HealthReport with pass/fail for each check.
    """
    # Extract config fields
    if isinstance(partner_config, dict):
        base_url = partner_config.get("base_url", "http://localhost:8000")
        api_key = partner_config.get("api_key", "")
        partner_id = partner_config.get("partner_id", "unknown")
    else:
        base_url = getattr(partner_config, "base_url", "http://localhost:8000")
        api_key = getattr(partner_config, "api_key", "")
        partner_id = getattr(partner_config, "partner_id", "unknown")

    logger.info("Running health check for partner=%s base_url=%s", partner_id, base_url)

    report = HealthReport(
        partner_id=partner_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Run checks in order
    checks = [
        _check_api_connectivity(base_url, api_key, timeout_seconds),
        _check_auth_working(base_url, api_key, timeout_seconds),
        _check_search_returns_results(base_url, api_key, timeout_seconds),
        _check_factor_retrieval(base_url, api_key, timeout_seconds),
    ]

    for check in checks:
        report.checks.append(check)
        report.total_latency_ms += check.latency_ms

    # Aggregate
    report.pass_count = sum(1 for c in report.checks if c.status == CheckStatus.PASS)
    report.fail_count = sum(1 for c in report.checks if c.status == CheckStatus.FAIL)
    report.warn_count = sum(1 for c in report.checks if c.status == CheckStatus.WARN)

    if report.fail_count > 0:
        report.overall_status = "fail"
    elif report.warn_count > 0:
        report.overall_status = "warn"
    else:
        report.overall_status = "pass"

    logger.info(
        "Health check complete: partner=%s status=%s pass=%d fail=%d warn=%d",
        partner_id, report.overall_status,
        report.pass_count, report.fail_count, report.warn_count,
    )
    return report
