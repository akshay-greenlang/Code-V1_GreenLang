# -*- coding: utf-8 -*-
"""
Recording Rules Generator - OBS-005: SLO/SLI Definitions & Error Budget Management

Generates Prometheus recording rules for SLI ratios, error budget
remaining, and burn rate metrics.  Writes rules to YAML files for
Prometheus to load.

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
from greenlang.infrastructure.slo_service.sli_calculator import (
    build_sli_ratio_query,
    build_error_rate_query,
)
from greenlang.infrastructure.slo_service.burn_rate import build_burn_rate_promql


# ---------------------------------------------------------------------------
# Recording rule generators
# ---------------------------------------------------------------------------


def generate_sli_recording_rule(slo: SLO) -> Dict[str, Any]:
    """Generate a recording rule for the SLI ratio.

    Args:
        slo: SLO definition.

    Returns:
        Recording rule dictionary.
    """
    return {
        "record": f"slo:{slo.safe_name}:sli_ratio",
        "expr": build_sli_ratio_query(slo.sli, slo.window.prometheus_duration),
        "labels": {
            "slo_id": slo.slo_id,
            "service": slo.service,
            "sli_type": slo.sli.sli_type.value,
        },
    }


def generate_error_budget_recording_rule(slo: SLO) -> Dict[str, Any]:
    """Generate a recording rule for remaining error budget percentage.

    Args:
        slo: SLO definition.

    Returns:
        Recording rule dictionary.
    """
    error_rate_expr = build_error_rate_query(
        slo.sli, slo.window.prometheus_duration
    )
    budget_fraction = slo.error_budget_fraction

    expr = (
        f"clamp_min(1 - (({error_rate_expr}) / {budget_fraction}), 0) * 100"
    )

    return {
        "record": f"slo:{slo.safe_name}:error_budget_remaining",
        "expr": expr,
        "labels": {
            "slo_id": slo.slo_id,
            "service": slo.service,
        },
    }


def generate_burn_rate_recording_rule(
    slo: SLO,
    window: BurnRateWindow,
) -> Dict[str, Any]:
    """Generate a recording rule for burn rate over a specific window.

    Args:
        slo: SLO definition.
        window: Burn rate window tier.

    Returns:
        Recording rule dictionary.
    """
    return {
        "record": f"slo:{slo.safe_name}:burn_rate_{window.value}",
        "expr": build_burn_rate_promql(slo, window.long_window),
        "labels": {
            "slo_id": slo.slo_id,
            "service": slo.service,
            "burn_window": window.value,
        },
    }


def generate_all_recording_rules(slos: List[SLO]) -> Dict[str, Any]:
    """Generate all recording rules for a list of SLOs.

    Produces three groups:
    - SLI ratio rules
    - Error budget remaining rules
    - Burn rate rules (fast, medium, slow)

    Args:
        slos: List of SLO definitions.

    Returns:
        Dictionary representing the complete recording rules YAML.
    """
    sli_rules = []
    budget_rules = []
    burn_rate_rules = []

    for slo in slos:
        if not slo.enabled:
            continue

        sli_rules.append(generate_sli_recording_rule(slo))
        budget_rules.append(generate_error_budget_recording_rule(slo))

        for window_name in ["fast", "medium", "slow"]:
            window = BurnRateWindow(window_name)
            burn_rate_rules.append(
                generate_burn_rate_recording_rule(slo, window)
            )

    return {
        "groups": [
            {
                "name": "slo_sli_ratio_rules",
                "interval": "60s",
                "rules": sli_rules,
            },
            {
                "name": "slo_error_budget_rules",
                "interval": "60s",
                "rules": budget_rules,
            },
            {
                "name": "slo_burn_rate_rules",
                "interval": "30s",
                "rules": burn_rate_rules,
            },
        ]
    }


def write_recording_rules_file(
    slos: List[SLO],
    output_path: str,
) -> str:
    """Generate and write recording rules to a YAML file.

    Args:
        slos: List of SLO definitions.
        output_path: Filesystem path for the output file.

    Returns:
        Absolute path to the written file.
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for writing recording rules")

    rules = generate_all_recording_rules(slos)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

    logger.info("Recording rules written to %s", path)
    return str(path.resolve())
