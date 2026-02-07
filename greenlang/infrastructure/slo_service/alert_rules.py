# -*- coding: utf-8 -*-
"""
Alert Rules Generator - OBS-005: SLO/SLI Definitions & Error Budget Management

Generates Prometheus alert rules for burn rate alerts (fast, medium, slow),
budget exhaustion/critical/warning alerts, and self-monitoring alerts.
Writes rules to YAML files for Prometheus to load.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

from greenlang.infrastructure.slo_service.models import SLO, BurnRateWindow
from greenlang.infrastructure.slo_service.burn_rate import (
    generate_burn_rate_alert_rule,
)


# ---------------------------------------------------------------------------
# Budget threshold alerts
# ---------------------------------------------------------------------------


def generate_budget_exhausted_alert(slo: SLO) -> Dict[str, Any]:
    """Generate an alert for when the error budget is fully exhausted.

    Args:
        slo: SLO definition.

    Returns:
        Prometheus alert rule dictionary.
    """
    return {
        "alert": f"SLOBudgetExhausted_{slo.safe_name}",
        "expr": f'slo:{slo.safe_name}:error_budget_remaining <= 0',
        "for": "5m",
        "labels": {
            "severity": "critical",
            "slo_id": slo.slo_id,
            "slo_name": slo.name,
            "service": slo.service,
        },
        "annotations": {
            "summary": f"Error budget exhausted for SLO '{slo.name}'",
            "description": (
                f"The error budget for SLO '{slo.name}' (target {slo.target}%) "
                f"has been completely consumed."
            ),
            "runbook_url": f"https://runbooks.greenlang.io/slo/{slo.slo_id}/budget-exhausted",
        },
    }


def generate_budget_critical_alert(slo: SLO) -> Dict[str, Any]:
    """Generate an alert for when the error budget is critically low.

    Args:
        slo: SLO definition.

    Returns:
        Prometheus alert rule dictionary.
    """
    return {
        "alert": f"SLOBudgetCritical_{slo.safe_name}",
        "expr": f'slo:{slo.safe_name}:error_budget_remaining < 50',
        "for": "15m",
        "labels": {
            "severity": "warning",
            "slo_id": slo.slo_id,
            "slo_name": slo.name,
            "service": slo.service,
        },
        "annotations": {
            "summary": f"Error budget critically low for SLO '{slo.name}'",
            "description": (
                f"Less than 50% of the error budget remains for SLO "
                f"'{slo.name}' (target {slo.target}%)."
            ),
            "runbook_url": f"https://runbooks.greenlang.io/slo/{slo.slo_id}/budget-critical",
        },
    }


def generate_budget_warning_alert(slo: SLO) -> Dict[str, Any]:
    """Generate an alert for when the error budget is getting low.

    Args:
        slo: SLO definition.

    Returns:
        Prometheus alert rule dictionary.
    """
    return {
        "alert": f"SLOBudgetWarning_{slo.safe_name}",
        "expr": f'slo:{slo.safe_name}:error_budget_remaining < 80',
        "for": "30m",
        "labels": {
            "severity": "info",
            "slo_id": slo.slo_id,
            "slo_name": slo.name,
            "service": slo.service,
        },
        "annotations": {
            "summary": f"Error budget getting low for SLO '{slo.name}'",
            "description": (
                f"Less than 80% of the error budget remains for SLO "
                f"'{slo.name}' (target {slo.target}%)."
            ),
            "runbook_url": f"https://runbooks.greenlang.io/slo/{slo.slo_id}/budget-warning",
        },
    }


# ---------------------------------------------------------------------------
# Self-monitoring alerts
# ---------------------------------------------------------------------------


def generate_self_monitoring_alerts() -> List[Dict[str, Any]]:
    """Generate alerts for the SLO service itself.

    Returns:
        List of self-monitoring alert rule dictionaries.
    """
    return [
        {
            "alert": "SLOServiceEvaluationStale",
            "expr": (
                'time() - gl_slo_last_evaluation_timestamp_seconds > 300'
            ),
            "for": "5m",
            "labels": {"severity": "warning", "service": "slo-service"},
            "annotations": {
                "summary": "SLO service evaluation is stale",
                "description": "SLO evaluations have not run in over 5 minutes.",
                "runbook_url": "https://runbooks.greenlang.io/slo/service-stale",
            },
        },
        {
            "alert": "SLOServiceEvaluationErrors",
            "expr": (
                'rate(gl_slo_evaluation_errors_total[5m]) > 0'
            ),
            "for": "10m",
            "labels": {"severity": "warning", "service": "slo-service"},
            "annotations": {
                "summary": "SLO service evaluation errors",
                "description": "SLO evaluations are producing errors.",
                "runbook_url": "https://runbooks.greenlang.io/slo/evaluation-errors",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Aggregate generation
# ---------------------------------------------------------------------------


def generate_all_alert_rules(slos: List[SLO]) -> Dict[str, Any]:
    """Generate all alert rules for a list of SLOs.

    Includes:
    - Burn rate alerts (fast, medium, slow) per SLO
    - Budget threshold alerts (exhausted, critical, warning) per SLO
    - Self-monitoring alerts

    Args:
        slos: List of SLO definitions.

    Returns:
        Dictionary representing the alert rules YAML.
    """
    burn_rate_rules = []
    budget_rules = []

    for slo in slos:
        if not slo.enabled:
            continue

        for window_name in ["fast", "medium", "slow"]:
            window = BurnRateWindow(window_name)
            burn_rate_rules.append(generate_burn_rate_alert_rule(slo, window))

        budget_rules.append(generate_budget_exhausted_alert(slo))
        budget_rules.append(generate_budget_critical_alert(slo))
        budget_rules.append(generate_budget_warning_alert(slo))

    return {
        "groups": [
            {
                "name": "slo_burn_rate_alerts",
                "rules": burn_rate_rules,
            },
            {
                "name": "slo_budget_alerts",
                "rules": budget_rules,
            },
            {
                "name": "slo_self_monitoring",
                "rules": generate_self_monitoring_alerts(),
            },
        ]
    }


def write_alert_rules_file(
    slos: List[SLO],
    output_path: str,
) -> str:
    """Generate and write alert rules to a YAML file.

    Args:
        slos: List of SLO definitions.
        output_path: Filesystem path for the output file.

    Returns:
        Absolute path to the written file.
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for writing alert rules")

    rules = generate_all_alert_rules(slos)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

    logger.info("Alert rules written to %s", path)
    return str(path.resolve())
