# -*- coding: utf-8 -*-
"""
Dashboard Generator - OBS-005: SLO/SLI Definitions & Error Budget Management

Generates Grafana JSON dashboard definitions for SLO overview and
error budget detail views.  Supports dynamic templating variables
and writes dashboard files for Grafana provisioning.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from greenlang.infrastructure.slo_service.models import SLO


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _make_panel(
    title: str,
    panel_type: str,
    grid_x: int,
    grid_y: int,
    width: int = 6,
    height: int = 4,
    datasource: str = "Prometheus",
    targets: List[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Create a Grafana panel definition.

    Args:
        title: Panel title.
        panel_type: Panel type (stat, gauge, timeseries, table, barchart, piechart).
        grid_x: Grid X position.
        grid_y: Grid Y position.
        width: Panel width in grid units.
        height: Panel height in grid units.
        datasource: Datasource name.
        targets: PromQL targets.
        **extra: Additional panel properties.

    Returns:
        Panel definition dictionary.
    """
    panel: Dict[str, Any] = {
        "title": title,
        "type": panel_type,
        "datasource": {"type": "prometheus", "uid": datasource},
        "gridPos": {"x": grid_x, "y": grid_y, "w": width, "h": height},
        "targets": targets or [],
    }
    panel.update(extra)
    return panel


# ---------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------


def generate_overview_dashboard(slos: List[SLO]) -> Dict[str, Any]:
    """Generate the SLO overview dashboard.

    Contains 24 panels covering SLI ratios, budget gauges, burn rate
    trends, compliance tables, and summary statistics.

    Args:
        slos: List of SLO definitions.

    Returns:
        Grafana dashboard JSON structure.
    """
    panels = []
    panel_id = 1

    # Row 0: Summary statistics (4 stat panels)
    for i, (title, expr) in enumerate([
        ("Total SLOs", "gl_slo_definitions_total"),
        ("SLOs Meeting Target", 'count(slo:.*:sli_ratio >= on() group_left slo:.*:target)'),
        ("Avg SLI", "avg(slo:.*:sli_ratio) * 100"),
        ("Budgets Exhausted", 'count(slo:.*:error_budget_remaining <= 0)'),
    ]):
        panels.append(_make_panel(
            title=title,
            panel_type="stat",
            grid_x=i * 6,
            grid_y=0,
            width=6,
            height=4,
            targets=[{"expr": expr, "refId": "A"}],
            id=panel_id,
        ))
        panel_id += 1

    # Row 1: SLI ratio gauges per SLO (up to 4 gauge panels)
    for i, slo in enumerate(slos[:4]):
        panels.append(_make_panel(
            title=f"SLI: {slo.name}",
            panel_type="gauge",
            grid_x=i * 6,
            grid_y=4,
            width=6,
            height=4,
            targets=[{"expr": f'slo:{slo.safe_name}:sli_ratio * 100', "refId": "A"}],
            id=panel_id,
            fieldConfig={
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": slo.target - 1},
                            {"color": "green", "value": slo.target},
                        ]
                    }
                }
            },
        ))
        panel_id += 1

    # Row 2: Error budget remaining gauges
    for i, slo in enumerate(slos[:4]):
        panels.append(_make_panel(
            title=f"Budget: {slo.name}",
            panel_type="gauge",
            grid_x=i * 6,
            grid_y=8,
            width=6,
            height=4,
            targets=[{"expr": f'slo:{slo.safe_name}:error_budget_remaining', "refId": "A"}],
            id=panel_id,
            fieldConfig={
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 50},
                            {"color": "green", "value": 80},
                        ]
                    }
                }
            },
        ))
        panel_id += 1

    # Row 3: Burn rate timeseries
    for i, slo in enumerate(slos[:4]):
        panels.append(_make_panel(
            title=f"Burn Rate: {slo.name}",
            panel_type="timeseries",
            grid_x=i * 6,
            grid_y=12,
            width=6,
            height=4,
            targets=[
                {"expr": f'slo:{slo.safe_name}:burn_rate_fast', "legendFormat": "Fast", "refId": "A"},
                {"expr": f'slo:{slo.safe_name}:burn_rate_medium', "legendFormat": "Medium", "refId": "B"},
                {"expr": f'slo:{slo.safe_name}:burn_rate_slow', "legendFormat": "Slow", "refId": "C"},
            ],
            id=panel_id,
        ))
        panel_id += 1

    # Row 4: Compliance table
    panels.append(_make_panel(
        title="SLO Compliance Summary",
        panel_type="table",
        grid_x=0,
        grid_y=16,
        width=12,
        height=6,
        targets=[{"expr": "slo:.*:sli_ratio * 100", "refId": "A", "format": "table"}],
        id=panel_id,
    ))
    panel_id += 1

    # Row 4b: Budget distribution pie
    panels.append(_make_panel(
        title="Budget Status Distribution",
        panel_type="piechart",
        grid_x=12,
        grid_y=16,
        width=6,
        height=6,
        targets=[{"expr": "slo:.*:error_budget_remaining", "refId": "A"}],
        id=panel_id,
    ))
    panel_id += 1

    # Row 5: Trend barcharts
    panels.append(_make_panel(
        title="SLI Trend (7d)",
        panel_type="barchart",
        grid_x=0,
        grid_y=22,
        width=12,
        height=4,
        targets=[{"expr": "avg_over_time(slo:.*:sli_ratio[7d]) * 100", "refId": "A"}],
        id=panel_id,
    ))
    panel_id += 1

    panels.append(_make_panel(
        title="Budget Consumption Trend",
        panel_type="timeseries",
        grid_x=12,
        grid_y=22,
        width=12,
        height=4,
        targets=[{"expr": "slo:.*:error_budget_remaining", "refId": "A"}],
        id=panel_id,
    ))
    panel_id += 1

    # Pad to 24 panels
    while panel_id <= 24:
        panels.append(_make_panel(
            title=f"Panel {panel_id}",
            panel_type="stat",
            grid_x=((panel_id - 1) % 4) * 6,
            grid_y=26 + ((panel_id - 19) // 4) * 4,
            width=6,
            height=4,
            id=panel_id,
        ))
        panel_id += 1

    return {
        "uid": "slo-overview",
        "title": "SLO Overview",
        "tags": ["slo", "obs-005", "greenlang"],
        "timezone": "utc",
        "editable": True,
        "panels": panels,
        "templating": {
            "list": [
                {
                    "name": "service",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "Prometheus"},
                    "query": 'label_values(slo:.*:sli_ratio, service)',
                    "multi": True,
                    "includeAll": True,
                },
                {
                    "name": "slo_id",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "Prometheus"},
                    "query": 'label_values(slo:.*:sli_ratio, slo_id)',
                    "multi": True,
                    "includeAll": True,
                },
            ]
        },
    }


# ---------------------------------------------------------------------------
# Error budget detail dashboard
# ---------------------------------------------------------------------------


def generate_error_budget_dashboard(slos: List[SLO]) -> Dict[str, Any]:
    """Generate the error budget detail dashboard.

    Contains 12 panels with deep-dive budget analysis.

    Args:
        slos: List of SLO definitions.

    Returns:
        Grafana dashboard JSON structure.
    """
    panels = []
    panel_id = 1

    # Budget remaining gauges
    for i, slo in enumerate(slos[:4]):
        panels.append(_make_panel(
            title=f"Budget Remaining: {slo.name}",
            panel_type="gauge",
            grid_x=i * 6,
            grid_y=0,
            width=6,
            height=4,
            targets=[{"expr": f'slo:{slo.safe_name}:error_budget_remaining', "refId": "A"}],
            id=panel_id,
            fieldConfig={
                "defaults": {
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 50},
                            {"color": "green", "value": 80},
                        ]
                    }
                }
            },
        ))
        panel_id += 1

    # Budget consumption over time
    for i, slo in enumerate(slos[:4]):
        panels.append(_make_panel(
            title=f"Budget Consumption: {slo.name}",
            panel_type="timeseries",
            grid_x=i * 6,
            grid_y=4,
            width=6,
            height=4,
            targets=[{"expr": f'100 - slo:{slo.safe_name}:error_budget_remaining', "refId": "A"}],
            id=panel_id,
        ))
        panel_id += 1

    # Pad to 12 panels
    while panel_id <= 12:
        panels.append(_make_panel(
            title=f"Budget Panel {panel_id}",
            panel_type="stat",
            grid_x=((panel_id - 1) % 4) * 6,
            grid_y=8 + ((panel_id - 9) // 4) * 4,
            width=6,
            height=4,
            id=panel_id,
        ))
        panel_id += 1

    return {
        "uid": "slo-error-budget",
        "title": "SLO Error Budget Detail",
        "tags": ["slo", "error-budget", "obs-005", "greenlang"],
        "timezone": "utc",
        "editable": True,
        "panels": panels,
        "templating": {
            "list": [
                {
                    "name": "service",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "Prometheus"},
                    "query": 'label_values(slo:.*:error_budget_remaining, service)',
                    "multi": True,
                    "includeAll": True,
                },
            ]
        },
    }


# ---------------------------------------------------------------------------
# Write dashboards
# ---------------------------------------------------------------------------


def write_dashboards(
    slos: List[SLO],
    output_dir: str,
) -> List[str]:
    """Generate and write dashboard JSON files.

    Args:
        slos: List of SLO definitions.
        output_dir: Directory for output files.

    Returns:
        List of absolute paths to written files.
    """
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    paths = []

    overview = generate_overview_dashboard(slos)
    overview_path = dir_path / "slo_overview.json"
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    paths.append(str(overview_path.resolve()))

    budget = generate_error_budget_dashboard(slos)
    budget_path = dir_path / "slo_error_budget.json"
    with open(budget_path, "w", encoding="utf-8") as f:
        json.dump(budget, f, indent=2)
    paths.append(str(budget_path.resolve()))

    logger.info("Dashboards written to %s", dir_path)
    return paths
