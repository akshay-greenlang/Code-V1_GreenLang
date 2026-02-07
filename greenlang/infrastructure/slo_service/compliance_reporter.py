# -*- coding: utf-8 -*-
"""
Compliance Reporter - OBS-005: SLO/SLI Definitions & Error Budget Management

Generates periodic compliance reports (weekly, monthly, quarterly)
summarizing SLO performance, budget consumption, trends, and
violation counts.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from greenlang.infrastructure.slo_service.models import (
    ErrorBudget,
    SLO,
    SLOReport,
    SLOReportEntry,
)


# ---------------------------------------------------------------------------
# Trend calculation
# ---------------------------------------------------------------------------


def calculate_trend(
    current_sli: float,
    previous_sli: float,
    threshold: float = 0.001,
) -> str:
    """Determine the SLI performance trend.

    Args:
        current_sli: Current SLI value (0-1).
        previous_sli: Previous period SLI value (0-1).
        threshold: Minimum difference for trend detection.

    Returns:
        One of ``improving``, ``stable``, or ``degrading``.
    """
    diff = current_sli - previous_sli
    if diff > threshold:
        return "improving"
    elif diff < -threshold:
        return "degrading"
    else:
        return "stable"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    report_type: str,
    slos: List[SLO],
    budgets: Dict[str, ErrorBudget],
    previous_slis: Optional[Dict[str, float]] = None,
    period_start: Optional[datetime] = None,
    period_end: Optional[datetime] = None,
) -> SLOReport:
    """Generate a compliance report for the given SLOs.

    Args:
        report_type: Type of report (weekly, monthly, quarterly).
        slos: List of SLO definitions.
        budgets: Mapping of slo_id to ErrorBudget.
        previous_slis: Previous period SLI values for trend calculation.
        period_start: Start of reporting period.
        period_end: End of reporting period.

    Returns:
        SLOReport instance.
    """
    now = datetime.now(timezone.utc)

    if period_end is None:
        period_end = now

    if period_start is None:
        if report_type == "weekly":
            period_start = period_end - timedelta(days=7)
        elif report_type == "monthly":
            period_start = period_end - timedelta(days=30)
        elif report_type == "quarterly":
            period_start = period_end - timedelta(days=90)
        else:
            period_start = period_end - timedelta(days=7)

    prev = previous_slis or {}
    entries = []
    slos_met = 0
    slos_not_met = 0

    for slo in slos:
        if slo.deleted or not slo.enabled:
            continue

        budget = budgets.get(slo.slo_id)
        if budget is None:
            continue

        current_sli_ratio = budget.sli_value / 100.0
        met = budget.sli_value >= slo.target
        if met:
            slos_met += 1
        else:
            slos_not_met += 1

        prev_sli = prev.get(slo.slo_id, current_sli_ratio)
        trend = calculate_trend(current_sli_ratio, prev_sli)

        entry = SLOReportEntry(
            slo_id=slo.slo_id,
            slo_name=slo.name,
            service=slo.service,
            target=slo.target,
            current_sli=budget.sli_value,
            met=met,
            budget_remaining_percent=budget.remaining_percent,
            budget_status=budget.status.value,
            trend=trend,
        )
        entries.append(entry)

    total = slos_met + slos_not_met
    compliance_pct = (slos_met / total * 100.0) if total > 0 else 0.0

    return SLOReport(
        report_type=report_type,
        period_start=period_start,
        period_end=period_end,
        entries=entries,
        overall_compliance_percent=compliance_pct,
        total_slos=total,
        slos_met=slos_met,
        slos_not_met=slos_not_met,
    )


def store_report(report: SLOReport, output_dir: str) -> str:
    """Persist a compliance report to disk.

    Args:
        report: SLOReport to store.
        output_dir: Directory for report files.

    Returns:
        Absolute path to the stored report file.
    """
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"slo_report_{report.report_type}_{report.report_id}.json"
    file_path = dir_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info("Report stored: %s", file_path)
    return str(file_path.resolve())
