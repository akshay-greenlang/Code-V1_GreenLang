# -*- coding: utf-8 -*-
"""
SLI Calculator - OBS-005: SLO/SLI Definitions & Error Budget Management

Builds PromQL queries for SLI ratio calculations and evaluates them
against a Prometheus endpoint.  Supports all SLI types (availability,
latency, correctness, throughput, freshness) and generates Prometheus
recording rules for pre-computed SLI values.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.debug("httpx not installed; SLI calculator operates in offline mode")

from greenlang.infrastructure.slo_service.models import SLI, SLO, SLIType


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def build_sli_ratio_query(sli: SLI, window: str = "30d") -> str:
    """Build a PromQL ratio query for the given SLI.

    Args:
        sli: SLI definition containing good and total queries.
        window: PromQL duration string (e.g. ``5m``, ``30d``).

    Returns:
        PromQL expression computing good/total ratio.
    """
    if sli.sli_type == SLIType.AVAILABILITY:
        return _build_availability_query(sli, window)
    elif sli.sli_type == SLIType.LATENCY:
        return _build_latency_query(sli, window)
    elif sli.sli_type == SLIType.CORRECTNESS:
        return _build_correctness_query(sli, window)
    elif sli.sli_type == SLIType.THROUGHPUT:
        return _build_throughput_query(sli, window)
    elif sli.sli_type == SLIType.FRESHNESS:
        return _build_freshness_query(sli, window)
    else:
        return f"({sli.good_query}) / ({sli.total_query})"


def _build_availability_query(sli: SLI, window: str) -> str:
    """Build availability ratio query (good requests / total requests)."""
    return (
        f"sum(increase({sli.good_query}[{window}]))"
        f" / "
        f"sum(increase({sli.total_query}[{window}]))"
    )


def _build_latency_query(sli: SLI, window: str) -> str:
    """Build latency ratio query (requests under threshold / total)."""
    return (
        f"sum(increase({sli.good_query}[{window}]))"
        f" / "
        f"sum(increase({sli.total_query}[{window}]))"
    )


def _build_correctness_query(sli: SLI, window: str) -> str:
    """Build correctness ratio query (correct results / total results)."""
    return (
        f"sum(increase({sli.good_query}[{window}]))"
        f" / "
        f"sum(increase({sli.total_query}[{window}]))"
    )


def _build_throughput_query(sli: SLI, window: str) -> str:
    """Build throughput ratio query (actual / expected throughput)."""
    return (
        f"sum(rate({sli.good_query}[{window}]))"
        f" / "
        f"sum(rate({sli.total_query}[{window}]))"
    )


def _build_freshness_query(sli: SLI, window: str) -> str:
    """Build freshness ratio query (fresh data / total data)."""
    return (
        f"sum(increase({sli.good_query}[{window}]))"
        f" / "
        f"sum(increase({sli.total_query}[{window}]))"
    )


def build_error_rate_query(sli: SLI, window: str = "30d") -> str:
    """Build a PromQL error rate query (1 - SLI ratio).

    Args:
        sli: SLI definition.
        window: PromQL duration string.

    Returns:
        PromQL expression for the error rate.
    """
    ratio = build_sli_ratio_query(sli, window)
    return f"1 - ({ratio})"


# ---------------------------------------------------------------------------
# Recording rule generation
# ---------------------------------------------------------------------------


def generate_recording_rule(slo: SLO) -> Dict[str, Any]:
    """Generate a Prometheus recording rule for an SLO's SLI.

    The recording rule pre-computes the SLI ratio so dashboards and
    alerts can reference a named metric instead of a raw PromQL query.

    Args:
        slo: SLO definition.

    Returns:
        Dictionary representing a single Prometheus recording rule.
    """
    metric_name = f"slo:{slo.safe_name}:sli_ratio"
    expr = build_sli_ratio_query(slo.sli, slo.window.prometheus_duration)

    rule: Dict[str, Any] = {
        "record": metric_name,
        "expr": expr,
        "labels": {
            "slo_id": slo.slo_id,
            "slo_name": slo.name,
            "service": slo.service,
            "sli_type": slo.sli.sli_type.value,
        },
    }
    if slo.team:
        rule["labels"]["team"] = slo.team

    return rule


def generate_recording_rules_yaml(slos: List[SLO]) -> Dict[str, Any]:
    """Generate a complete Prometheus recording rules file for all SLOs.

    Args:
        slos: List of SLO definitions.

    Returns:
        Dictionary representing the YAML structure for recording rules.
    """
    rules = [generate_recording_rule(slo) for slo in slos if slo.enabled]

    return {
        "groups": [
            {
                "name": "slo_sli_recording_rules",
                "interval": "60s",
                "rules": rules,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Prometheus query execution
# ---------------------------------------------------------------------------


async def query_prometheus(
    prometheus_url: str,
    query: str,
    timeout_seconds: int = 30,
) -> Optional[float]:
    """Execute a PromQL instant query and return the scalar result.

    Args:
        prometheus_url: Base URL of the Prometheus server.
        query: PromQL expression.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Scalar float result, or None on error / empty result.

    Raises:
        ConnectionError: When Prometheus is unreachable.
        TimeoutError: When the request exceeds the timeout.
    """
    if not HTTPX_AVAILABLE:
        logger.warning("httpx not installed; cannot query Prometheus")
        return None

    url = f"{prometheus_url.rstrip('/')}/api/v1/query"
    params = {"query": query}

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "success":
            logger.error("Prometheus query failed: %s", data)
            return None

        result = data.get("data", {}).get("result", [])
        if not result:
            return None

        value = float(result[0].get("value", [0, "0"])[1])
        return max(0.0, min(1.0, value))

    except Exception as exc:
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            raise TimeoutError(f"Prometheus query timed out: {exc}") from exc
        if "connect" in str(exc).lower():
            raise ConnectionError(f"Cannot connect to Prometheus: {exc}") from exc
        logger.error("Prometheus query error: %s", exc)
        return None


async def calculate_sli(
    slo: SLO,
    prometheus_url: str,
    window: Optional[str] = None,
    timeout_seconds: int = 30,
) -> Optional[float]:
    """Calculate the current SLI value for an SLO.

    Args:
        slo: SLO definition.
        prometheus_url: Prometheus URL.
        window: Override window (defaults to SLO's window).
        timeout_seconds: Request timeout.

    Returns:
        SLI ratio as a float between 0.0 and 1.0, or None on error.
    """
    w = window or slo.window.prometheus_duration
    query = build_sli_ratio_query(slo.sli, w)
    result = await query_prometheus(prometheus_url, query, timeout_seconds)

    if result is not None:
        return max(0.0, min(1.0, result))
    return None
