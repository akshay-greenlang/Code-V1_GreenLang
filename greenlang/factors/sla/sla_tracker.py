# -*- coding: utf-8 -*-
"""
Per-tenant SLA tracker (DEP9).

Computes, for a ``(tenant_id, [from, to])`` window:

    * Uptime %              (1 - error-rate fraction)
    * p95 / p99 latency     (histogram_quantile on
                             factors_http_request_duration_seconds_bucket)
    * Error rate            (5xx / total)
    * Error-budget burn     (fast + slow burn windows per SRE workbook)
    * Throughput            (requests / sec averaged)

All values come from Prometheus via the injectable :class:`PromQLClient`;
the class is deliberately dumb so tests swap in a :class:`FakePromClient`
fixture.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus client (real + fake)
# ---------------------------------------------------------------------------


class PromQLClient:
    """Tiny HTTP client against the Prometheus HTTP API.

    Only the ``/api/v1/query`` endpoint is used. No service discovery, no
    range queries: the tracker slices windows client-side for clarity.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("GL_FACTORS_PROMETHEUS_URL")
            or "http://prometheus-server.observability.svc.cluster.local:9090"
        ).rstrip("/")
        self.timeout = timeout

    def query(self, promql: str, at: Optional[float] = None) -> float:
        params = {"query": promql}
        if at is not None:
            params["time"] = str(at)
        url = f"{self.base_url}/api/v1/query?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") != "success":
            raise RuntimeError(f"prometheus query failed: {data}")
        result = data["data"].get("result") or []
        if not result:
            return float("nan")
        # PromQL scalar/instant-vector: return first value.
        val = result[0]["value"][1]
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")


class FakePromClient:
    """In-memory Prometheus stub used by tests."""

    def __init__(self, values: Optional[Dict[str, float]] = None) -> None:
        self.values = values or {}
        self.calls: List[Tuple[str, Optional[float]]] = []

    def query(self, promql: str, at: Optional[float] = None) -> float:
        self.calls.append((promql, at))
        for key, val in self.values.items():
            if key in promql:
                return val
        return 0.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BurnRate:
    window_minutes: int
    observed: float   # fraction of budget consumed per hour
    threshold: float  # alert threshold
    burning: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_minutes": self.window_minutes,
            "observed": self.observed,
            "threshold": self.threshold,
            "burning": self.burning,
        }


@dataclass
class SLAReport:
    tenant_id: str
    window_from: str
    window_to: str
    uptime_percent: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_rps: float
    error_budget_remaining: float
    burn_rates: List[BurnRate] = field(default_factory=list)
    slo_targets: Dict[str, float] = field(default_factory=dict)
    met: Dict[str, bool] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "window_from": self.window_from,
            "window_to": self.window_to,
            "uptime_percent": self.uptime_percent,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "error_rate": self.error_rate,
            "throughput_rps": self.throughput_rps,
            "error_budget_remaining": self.error_budget_remaining,
            "burn_rates": [b.to_dict() for b in self.burn_rates],
            "slo_targets": dict(self.slo_targets),
            "met": dict(self.met),
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


DEFAULT_SLO_TARGETS: Dict[str, float] = {
    "availability": 0.999,       # 99.9% monthly
    "p95_latency_ms": 500.0,
    "p99_latency_ms": 1500.0,
    "error_rate": 0.001,         # 0.1%
}


class SLATracker:
    """Per-tenant SLA report generator."""

    def __init__(
        self,
        prom: PromQLClient,
        *,
        targets: Optional[Dict[str, float]] = None,
        tier_overrides: Optional[Callable[[str], Dict[str, float]]] = None,
    ) -> None:
        self._prom = prom
        self._targets = targets or DEFAULT_SLO_TARGETS
        self._tier_overrides = tier_overrides or (lambda _t: {})

    # ------------------------------------------------------------------
    def _promql(self, tenant_id: str, window: str) -> Dict[str, str]:
        sel = f'tenant_id="{tenant_id}"'
        return {
            "uptime": (
                f'1 - (sum(rate(factors_http_requests_total{{code=~"5..",{sel}}}[{window}]))'
                f' / sum(rate(factors_http_requests_total{{{sel}}}[{window}])))'
            ),
            "p95": (
                f'histogram_quantile(0.95, sum by (le) ('
                f'rate(factors_http_request_duration_seconds_bucket{{{sel}}}[{window}])))'
            ),
            "p99": (
                f'histogram_quantile(0.99, sum by (le) ('
                f'rate(factors_http_request_duration_seconds_bucket{{{sel}}}[{window}])))'
            ),
            "err": (
                f'sum(rate(factors_http_requests_total{{code=~"5..",{sel}}}[{window}]))'
                f' / sum(rate(factors_http_requests_total{{{sel}}}[{window}]))'
            ),
            "rps": f'sum(rate(factors_http_requests_total{{{sel}}}[{window}]))',
        }

    # ------------------------------------------------------------------
    def report(
        self,
        tenant_id: str,
        window_from: datetime,
        window_to: datetime,
        tier: Optional[str] = None,
    ) -> SLAReport:
        if window_to <= window_from:
            raise ValueError("window_to must be after window_from")
        window_minutes = int((window_to - window_from).total_seconds() // 60)
        window = f"{max(window_minutes, 1)}m"

        q = self._promql(tenant_id, window)
        uptime = self._prom.query(q["uptime"])
        p95 = self._prom.query(q["p95"]) * 1000.0
        p99 = self._prom.query(q["p99"]) * 1000.0
        err = self._prom.query(q["err"])
        rps = self._prom.query(q["rps"])

        targets = dict(self._targets)
        if tier:
            targets.update(self._tier_overrides(tier) or {})

        met = {
            "availability": uptime >= targets["availability"],
            "p95_latency_ms": p95 <= targets["p95_latency_ms"] if not _is_nan(p95) else False,
            "p99_latency_ms": p99 <= targets["p99_latency_ms"] if not _is_nan(p99) else False,
            "error_rate": err <= targets["error_rate"],
        }
        budget_total = 1.0 - targets["availability"]
        budget_remaining = max(0.0, 1.0 - (1.0 - uptime) / max(budget_total, 1e-9))

        burn = [
            self._burn(tenant_id, window_minutes=5, threshold=14.4),
            self._burn(tenant_id, window_minutes=60, threshold=6.0),
            self._burn(tenant_id, window_minutes=24 * 60, threshold=1.0),
        ]

        return SLAReport(
            tenant_id=tenant_id,
            window_from=window_from.isoformat(),
            window_to=window_to.isoformat(),
            uptime_percent=round(uptime * 100.0, 4) if not _is_nan(uptime) else 0.0,
            p95_latency_ms=round(p95, 2) if not _is_nan(p95) else 0.0,
            p99_latency_ms=round(p99, 2) if not _is_nan(p99) else 0.0,
            error_rate=round(err, 6) if not _is_nan(err) else 0.0,
            throughput_rps=round(rps, 2) if not _is_nan(rps) else 0.0,
            error_budget_remaining=round(budget_remaining, 4),
            burn_rates=burn,
            slo_targets=targets,
            met=met,
        )

    # ------------------------------------------------------------------
    def _burn(self, tenant_id: str, window_minutes: int, threshold: float) -> BurnRate:
        q = self._promql(tenant_id, f"{window_minutes}m")["err"]
        observed = self._prom.query(q) or 0.0
        target_err = self._targets["error_rate"]
        burn = observed / max(target_err, 1e-9)
        return BurnRate(
            window_minutes=window_minutes,
            observed=round(burn, 3),
            threshold=threshold,
            burning=burn > threshold,
        )


def _is_nan(x: float) -> bool:
    return x != x  # NaN != NaN


# ---------------------------------------------------------------------------
# FastAPI installer
# ---------------------------------------------------------------------------


def install_sla_routes(
    app,
    *,
    prefix: str = "/v1/sla",
    tracker: Optional[SLATracker] = None,
) -> None:
    """Mount per-tenant SLA endpoints.

    Endpoints:

      * ``GET /v1/sla/report?tenant_id=...&from=...&to=...`` -> :class:`SLAReport`.
      * ``GET /v1/sla/burn?tenant_id=...`` -> fast/slow burn-rate triples for dashboards.
    """
    from fastapi import HTTPException, Query

    tr = tracker or SLATracker(PromQLClient())

    def _parse_window(
        window_from: Optional[str],
        window_to: Optional[str],
    ) -> Tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        to_dt = datetime.fromisoformat(window_to) if window_to else now
        # Default window: last 30 days (the monthly SLA window).
        from_dt = (
            datetime.fromisoformat(window_from)
            if window_from
            else to_dt - timedelta(days=30)
        )
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
        return from_dt, to_dt

    @app.get(prefix + "/report")
    def sla_report(
        tenant_id: str = Query(..., min_length=1),
        window_from: Optional[str] = Query(default=None, alias="from"),
        window_to: Optional[str] = Query(default=None, alias="to"),
    ):
        try:
            from_dt, to_dt = _parse_window(window_from, window_to)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date: {exc}")
        return tr.report(tenant_id=tenant_id, window_from=from_dt, window_to=to_dt).to_dict()

    @app.get(prefix + "/burn")
    def sla_burn(tenant_id: str = Query(..., min_length=1)):
        from_dt, to_dt = _parse_window(None, None)
        rep = tr.report(tenant_id=tenant_id, window_from=from_dt, window_to=to_dt)
        return {"tenant_id": tenant_id, "burn_rates": [b.to_dict() for b in rep.burn_rates]}

    logger.info("SLA routes installed at %s", prefix)


__all__ = [
    "BurnRate",
    "DEFAULT_SLO_TARGETS",
    "FakePromClient",
    "PromQLClient",
    "SLAReport",
    "SLATracker",
    "install_sla_routes",
]
