# -*- coding: utf-8 -*-
"""
Burn Rate Calculator - OBS-005: SLO/SLI Definitions & Error Budget Management

Implements multi-window burn rate calculation following the Google SRE
Book methodology.  Generates PromQL-based burn rate alert rules and
determines when alerts should fire based on both long and short window
thresholds.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from greenlang.infrastructure.slo_service.models import (
    BurnRateAlert,
    BurnRateWindow,
    SLO,
)


# ---------------------------------------------------------------------------
# Burn rate calculation
# ---------------------------------------------------------------------------


def calculate_burn_rate(
    error_rate: float,
    error_budget_fraction: float,
) -> float:
    """Calculate the burn rate: how fast the error budget is being consumed.

    Burn rate = actual error rate / allowed error rate
    A burn rate of 1.0 means the budget is consumed at the expected rate.
    A burn rate of 14.4 exhausts a 30-day budget in ~2 hours.

    Args:
        error_rate: Actual error rate (0.0 to 1.0).
        error_budget_fraction: Allowed error fraction (e.g. 0.001 for 99.9%).

    Returns:
        Burn rate multiplier.
    """
    if error_budget_fraction <= 0:
        return float("inf") if error_rate > 0 else 0.0
    return error_rate / error_budget_fraction


def should_alert(
    burn_rate_long: float,
    burn_rate_short: float,
    threshold: float,
) -> bool:
    """Determine whether a burn rate alert should fire.

    Both the long and short window burn rates must exceed the threshold
    for the alert to fire.  This prevents spurious alerts from brief
    spikes (short only) or stale data (long only).

    Args:
        burn_rate_long: Burn rate over the long observation window.
        burn_rate_short: Burn rate over the short confirmation window.
        threshold: Burn rate threshold for this window tier.

    Returns:
        True if the alert should fire.
    """
    return burn_rate_long > threshold and burn_rate_short > threshold


def evaluate_burn_rate_windows(
    slo: SLO,
    burn_rates: Dict[str, Dict[str, float]],
) -> List[BurnRateAlert]:
    """Evaluate all burn rate windows for an SLO and return firing alerts.

    Args:
        slo: SLO definition.
        burn_rates: Mapping of window name to ``{"long": float, "short": float}``.

    Returns:
        List of BurnRateAlert instances for firing alerts.
    """
    alerts: List[BurnRateAlert] = []

    for window_name in ["fast", "medium", "slow"]:
        window = BurnRateWindow(window_name)
        rates = burn_rates.get(window_name, {})
        long_rate = rates.get("long", 0.0)
        short_rate = rates.get("short", 0.0)

        if should_alert(long_rate, short_rate, window.threshold):
            alert = BurnRateAlert(
                slo_id=slo.slo_id,
                slo_name=slo.name,
                burn_window=window_name,
                burn_rate_long=long_rate,
                burn_rate_short=short_rate,
                threshold=window.threshold,
                severity=window.severity,
                service=slo.service,
                message=(
                    f"SLO '{slo.name}' burn rate alert ({window_name}): "
                    f"long={long_rate:.2f}x, short={short_rate:.2f}x, "
                    f"threshold={window.threshold}x"
                ),
            )
            alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# Exhaustion time estimates
# ---------------------------------------------------------------------------


def time_to_exhaustion_hours(burn_rate: float, window_days: int = 30) -> float:
    """Estimate hours until error budget exhaustion at the given burn rate.

    For a 30-day window with burn rate of 1.0, exhaustion takes 30 days
    (720 hours).  With burn rate 14.4, exhaustion takes ~2 hours.

    Args:
        burn_rate: Current burn rate multiplier.
        window_days: SLO window in days.

    Returns:
        Hours until exhaustion.
    """
    if burn_rate <= 0:
        return float("inf")
    total_hours = window_days * 24
    return total_hours / burn_rate


# ---------------------------------------------------------------------------
# PromQL rule generation
# ---------------------------------------------------------------------------


def build_burn_rate_promql(slo: SLO, window: str) -> str:
    """Build a PromQL expression for burn rate over a given window.

    Args:
        slo: SLO definition.
        window: PromQL duration string (e.g. ``1h``, ``5m``).

    Returns:
        PromQL expression computing the burn rate.
    """
    error_budget = slo.error_budget_fraction
    sli = slo.sli

    return (
        f"("
        f"1 - (sum(increase({sli.good_query}[{window}])) "
        f"/ sum(increase({sli.total_query}[{window}])))"
        f") / {error_budget}"
    )


def generate_burn_rate_alert_rule(
    slo: SLO,
    window: BurnRateWindow,
) -> Dict[str, Any]:
    """Generate a Prometheus alert rule for a burn rate window.

    Args:
        slo: SLO definition.
        window: Burn rate window tier.

    Returns:
        Dictionary representing a Prometheus alert rule.
    """
    long_expr = build_burn_rate_promql(slo, window.long_window)
    short_expr = build_burn_rate_promql(slo, window.short_window)

    alert_name = (
        f"SLOBurnRate{window.value.capitalize()}"
        f"_{slo.safe_name}"
    )

    for_duration = {"fast": "2m", "medium": "5m", "slow": "30m"}

    return {
        "alert": alert_name,
        "expr": f"({long_expr}) > {window.threshold} and ({short_expr}) > {window.threshold}",
        "for": for_duration.get(window.value, "5m"),
        "labels": {
            "severity": window.severity,
            "slo_id": slo.slo_id,
            "slo_name": slo.name,
            "service": slo.service,
            "burn_window": window.value,
        },
        "annotations": {
            "summary": (
                f"SLO '{slo.name}' is burning error budget too fast "
                f"({window.value} burn rate)"
            ),
            "description": (
                f"The error budget for SLO '{slo.name}' (target {slo.target}%) "
                f"is being consumed at more than {window.threshold}x the "
                f"sustainable rate over the {window.long_window} window."
            ),
            "runbook_url": (
                f"https://runbooks.greenlang.io/slo/{slo.slo_id}/burn-rate"
            ),
        },
    }


def generate_all_burn_rate_rules(slos: List[SLO]) -> Dict[str, Any]:
    """Generate all burn rate alert rules for a list of SLOs.

    Args:
        slos: List of SLO definitions.

    Returns:
        Dictionary representing the YAML structure.
    """
    rules = []
    for slo in slos:
        if not slo.enabled:
            continue
        for window_name in ["fast", "medium", "slow"]:
            window = BurnRateWindow(window_name)
            rules.append(generate_burn_rate_alert_rule(slo, window))

    return {
        "groups": [
            {
                "name": "slo_burn_rate_alerts",
                "rules": rules,
            }
        ]
    }
