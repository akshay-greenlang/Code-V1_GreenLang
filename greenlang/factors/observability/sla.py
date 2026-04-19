# -*- coding: utf-8 -*-
"""
Factors API SLO/SLI definitions and error budget helpers (F071).

Defines the Service Level Objectives and Indicators for the Factors API.
Each SLI has a target threshold and a function to evaluate the current
value against that target. Error budget helpers compute remaining budget
as a fraction of the compliance window.

SLI definitions:
  - Request latency p99 < 500 ms
  - Error rate (5xx) < 0.1%
  - Search latency p95 < 200 ms
  - Match confidence median > 0.5
  - Availability >= 99.9%

Usage::

    from greenlang.factors.observability.sla import (
        FACTORS_SLOS,
        check_all_slis,
        calculate_error_budget,
    )

    results = check_all_slis(current_values)
    for r in results:
        print(f"{r['name']}: {'PASS' if r['met'] else 'FAIL'}")

    budget = calculate_error_budget(
        target_availability=99.9,
        actual_availability=99.85,
        window_minutes=30 * 24 * 60,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SLICategory(str, Enum):
    """Categories of Service Level Indicators."""

    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    QUALITY = "quality"


class ComplianceStatus(str, Enum):
    """Whether an SLI is currently meeting its target."""

    MET = "met"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class BudgetHealth(str, Enum):
    """Error budget health classification."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------------
# SLI Definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLIDefinition:
    """A single Service Level Indicator definition.

    Attributes:
        name: Human-readable SLI name.
        sli_id: Machine-friendly identifier.
        category: Classification of this SLI.
        description: What this SLI measures.
        target: Target threshold value.
        unit: Unit of measurement (e.g. ``ms``, ``percent``, ``ratio``).
        comparison: How to compare actual vs target (``lt``, ``gt``, ``lte``, ``gte``).
        promql: PromQL expression to compute the current SLI value.
    """

    name: str
    sli_id: str
    category: SLICategory
    description: str
    target: float
    unit: str
    comparison: str  # "lt" | "gt" | "lte" | "gte"
    promql: str

    def evaluate(self, actual: float) -> bool:
        """Evaluate whether the actual value meets this SLI target.

        Args:
            actual: The current measured value.

        Returns:
            True if the SLI target is met, False otherwise.
        """
        if self.comparison == "lt":
            return actual < self.target
        if self.comparison == "lte":
            return actual <= self.target
        if self.comparison == "gt":
            return actual > self.target
        if self.comparison == "gte":
            return actual >= self.target
        logger.warning("Unknown comparison operator %r for SLI %s", self.comparison, self.sli_id)
        return False


# ---------------------------------------------------------------------------
# SLO Definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLODefinition:
    """A Service Level Objective grouping one or more SLIs.

    Attributes:
        name: Human-readable name.
        slo_id: Machine-friendly identifier.
        service: Service this SLO applies to.
        description: What this SLO represents.
        target_availability: Target availability percentage (e.g. 99.9).
        window_days: Compliance window in days.
        slis: List of SLI definitions belonging to this SLO.
    """

    name: str
    slo_id: str
    service: str
    description: str
    target_availability: float
    window_days: int
    slis: List[SLIDefinition] = field(default_factory=list)

    @property
    def window_minutes(self) -> int:
        """Total minutes in the compliance window."""
        return self.window_days * 24 * 60

    @property
    def error_budget_fraction(self) -> float:
        """Fraction of time allowed to be in violation (e.g. 0.001 for 99.9%)."""
        return 1.0 - (self.target_availability / 100.0)

    @property
    def error_budget_minutes(self) -> float:
        """Total error budget in minutes for the compliance window."""
        return self.window_minutes * self.error_budget_fraction


# ---------------------------------------------------------------------------
# SLI check result
# ---------------------------------------------------------------------------


@dataclass
class SLICheckResult:
    """Result of evaluating a single SLI.

    Attributes:
        sli_id: Identifier of the SLI.
        name: Human-readable name.
        target: Target threshold.
        actual: Measured value.
        unit: Unit of measurement.
        met: Whether the target was met.
        status: Compliance status enum.
        checked_at: Timestamp of the check.
    """

    sli_id: str
    name: str
    target: float
    actual: float
    unit: str
    met: bool
    status: ComplianceStatus
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "sli_id": self.sli_id,
            "name": self.name,
            "target": self.target,
            "actual": self.actual,
            "unit": self.unit,
            "met": self.met,
            "status": self.status.value,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# Error budget result
# ---------------------------------------------------------------------------


@dataclass
class ErrorBudgetResult:
    """Error budget calculation result.

    Attributes:
        slo_id: Identifier of the SLO.
        target_availability: Target availability percentage.
        actual_availability: Measured availability percentage.
        window_days: Compliance window in days.
        total_budget_minutes: Total error budget in minutes.
        consumed_minutes: Minutes of budget consumed.
        remaining_minutes: Minutes of budget remaining.
        consumed_percent: Percentage of budget consumed.
        remaining_percent: Percentage of budget remaining.
        health: Budget health classification.
        exhaustion_forecast_hours: Estimated hours until budget exhaustion
            at the current burn rate, or None if not burning.
    """

    slo_id: str
    target_availability: float
    actual_availability: float
    window_days: int
    total_budget_minutes: float
    consumed_minutes: float
    remaining_minutes: float
    consumed_percent: float
    remaining_percent: float
    health: BudgetHealth
    exhaustion_forecast_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "slo_id": self.slo_id,
            "target_availability": self.target_availability,
            "actual_availability": self.actual_availability,
            "window_days": self.window_days,
            "total_budget_minutes": round(self.total_budget_minutes, 2),
            "consumed_minutes": round(self.consumed_minutes, 2),
            "remaining_minutes": round(self.remaining_minutes, 2),
            "consumed_percent": round(self.consumed_percent, 2),
            "remaining_percent": round(self.remaining_percent, 2),
            "health": self.health.value,
            "exhaustion_forecast_hours": (
                round(self.exhaustion_forecast_hours, 1)
                if self.exhaustion_forecast_hours is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Factors SLI definitions
# ---------------------------------------------------------------------------

SLI_REQUEST_LATENCY_P99 = SLIDefinition(
    name="Request Latency p99",
    sli_id="factors.latency.p99",
    category=SLICategory.LATENCY,
    description="99th percentile request latency must be under 500ms",
    target=0.5,  # 500ms in seconds
    unit="seconds",
    comparison="lt",
    promql=(
        'histogram_quantile(0.99, rate(greenlang_factors_api_latency_seconds_bucket[5m]))'
    ),
)

SLI_ERROR_RATE = SLIDefinition(
    name="Error Rate (5xx)",
    sli_id="factors.errors.5xx",
    category=SLICategory.ERROR_RATE,
    description="5xx error rate must be below 0.1%",
    target=0.001,  # 0.1% as a ratio
    unit="ratio",
    comparison="lt",
    promql=(
        'sum(rate(greenlang_factors_api_requests_total{status=~"5.."}[5m]))'
        ' / sum(rate(greenlang_factors_api_requests_total[5m]))'
    ),
)

SLI_SEARCH_LATENCY_P95 = SLIDefinition(
    name="Search Latency p95",
    sli_id="factors.search.latency.p95",
    category=SLICategory.LATENCY,
    description="95th percentile search latency must be under 200ms",
    target=0.2,  # 200ms in seconds
    unit="seconds",
    comparison="lt",
    promql=(
        'histogram_quantile(0.95, rate('
        'greenlang_factors_api_latency_seconds_bucket{path=~"/api/v1/factors/search.*"}[5m]))'
    ),
)

SLI_MATCH_CONFIDENCE_MEDIAN = SLIDefinition(
    name="Match Confidence Median",
    sli_id="factors.match.confidence.p50",
    category=SLICategory.QUALITY,
    description="Median top-1 match confidence score must exceed 0.5",
    target=0.5,
    unit="ratio",
    comparison="gt",
    promql=(
        'histogram_quantile(0.50, rate(greenlang_factors_match_score_top1_bucket[5m]))'
    ),
)

SLI_AVAILABILITY = SLIDefinition(
    name="Service Availability",
    sli_id="factors.availability",
    category=SLICategory.AVAILABILITY,
    description="Factors API must maintain 99.9% availability",
    target=99.9,
    unit="percent",
    comparison="gte",
    promql=(
        '100 * (1 - sum(rate(greenlang_factors_api_requests_total{status=~"5.."}[30d]))'
        ' / sum(rate(greenlang_factors_api_requests_total[30d])))'
    ),
)


# ---------------------------------------------------------------------------
# Factors SLO definition
# ---------------------------------------------------------------------------

FACTORS_SLO = SLODefinition(
    name="Factors API SLO",
    slo_id="factors-api-slo-99.9",
    service="factors",
    description=(
        "The Factors API must maintain 99.9% availability with sub-500ms p99 "
        "latency, sub-200ms p95 search latency, below 0.1% error rate, and "
        "median match confidence above 0.5."
    ),
    target_availability=99.9,
    window_days=30,
    slis=[
        SLI_REQUEST_LATENCY_P99,
        SLI_ERROR_RATE,
        SLI_SEARCH_LATENCY_P95,
        SLI_MATCH_CONFIDENCE_MEDIAN,
        SLI_AVAILABILITY,
    ],
)

# Convenience list of all defined SLOs
FACTORS_SLOS: List[SLODefinition] = [FACTORS_SLO]


# ---------------------------------------------------------------------------
# SLI evaluation
# ---------------------------------------------------------------------------


def check_sli(sli: SLIDefinition, actual_value: float) -> SLICheckResult:
    """Evaluate a single SLI against its target.

    Args:
        sli: The SLI definition to check.
        actual_value: The current measured value.

    Returns:
        An SLICheckResult indicating whether the target is met.
    """
    met = sli.evaluate(actual_value)
    return SLICheckResult(
        sli_id=sli.sli_id,
        name=sli.name,
        target=sli.target,
        actual=actual_value,
        unit=sli.unit,
        met=met,
        status=ComplianceStatus.MET if met else ComplianceStatus.VIOLATED,
    )


def check_all_slis(
    current_values: Dict[str, float],
    slo: Optional[SLODefinition] = None,
) -> List[SLICheckResult]:
    """Evaluate all SLIs for an SLO against current measured values.

    Args:
        current_values: Mapping of ``sli_id`` to current measured value.
            SLIs not present in the mapping are reported as ``UNKNOWN``.
        slo: The SLO whose SLIs to check. Defaults to ``FACTORS_SLO``.

    Returns:
        List of SLICheckResult objects, one per SLI.

    Example::

        results = check_all_slis({
            "factors.latency.p99": 0.35,
            "factors.errors.5xx": 0.0005,
            "factors.search.latency.p95": 0.12,
            "factors.match.confidence.p50": 0.72,
            "factors.availability": 99.95,
        })
    """
    target_slo = slo or FACTORS_SLO
    results: List[SLICheckResult] = []

    for sli in target_slo.slis:
        if sli.sli_id in current_values:
            results.append(check_sli(sli, current_values[sli.sli_id]))
        else:
            results.append(
                SLICheckResult(
                    sli_id=sli.sli_id,
                    name=sli.name,
                    target=sli.target,
                    actual=0.0,
                    unit=sli.unit,
                    met=False,
                    status=ComplianceStatus.UNKNOWN,
                )
            )
            logger.warning("No current value for SLI %s; marked as UNKNOWN", sli.sli_id)

    met_count = sum(1 for r in results if r.met)
    total_count = len(results)
    logger.info(
        "SLI check complete: %d/%d SLIs met for SLO %s",
        met_count,
        total_count,
        target_slo.slo_id,
    )

    return results


# ---------------------------------------------------------------------------
# Error budget calculation
# ---------------------------------------------------------------------------


def calculate_error_budget(
    target_availability: float = 99.9,
    actual_availability: float = 100.0,
    window_minutes: Optional[int] = None,
    window_days: int = 30,
    slo_id: str = "factors-api-slo-99.9",
    elapsed_minutes: Optional[float] = None,
) -> ErrorBudgetResult:
    """Calculate error budget consumption for the Factors API.

    Args:
        target_availability: Target availability percentage (e.g. 99.9).
        actual_availability: Current measured availability percentage.
        window_minutes: Total window in minutes. Overrides ``window_days``.
        window_days: Compliance window in days (default 30).
        slo_id: Identifier of the SLO.
        elapsed_minutes: Minutes elapsed in the current window. Used for
            burn rate forecasting. If None, assumed to be half the window.

    Returns:
        ErrorBudgetResult with consumption details and health classification.

    Example::

        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.85,
        )
        print(budget.health)  # BudgetHealth.WARNING
    """
    total_minutes = window_minutes if window_minutes is not None else window_days * 24 * 60

    target_ratio = target_availability / 100.0
    actual_ratio = actual_availability / 100.0

    error_budget_fraction = 1.0 - target_ratio
    total_budget_minutes = total_minutes * error_budget_fraction

    actual_error_rate = 1.0 - actual_ratio
    allowed_error_rate = error_budget_fraction

    if allowed_error_rate <= 0:
        consumed_percent = 100.0 if actual_error_rate > 0 else 0.0
    else:
        consumed_percent = min((actual_error_rate / allowed_error_rate) * 100.0, 100.0)

    consumed_minutes = total_budget_minutes * (consumed_percent / 100.0)
    remaining_minutes = max(total_budget_minutes - consumed_minutes, 0.0)
    remaining_percent = max(100.0 - consumed_percent, 0.0)

    health = _classify_budget_health(consumed_percent)

    # Forecast exhaustion
    exhaustion_forecast_hours: Optional[float] = None
    if consumed_minutes > 0 and remaining_minutes > 0:
        effective_elapsed = elapsed_minutes if elapsed_minutes is not None else total_minutes / 2.0
        if effective_elapsed > 0:
            burn_rate_per_minute = consumed_minutes / effective_elapsed
            exhaustion_forecast_hours = (remaining_minutes / burn_rate_per_minute) / 60.0

    logger.info(
        "Error budget for %s: consumed=%.1f%% remaining=%.1f min health=%s",
        slo_id,
        consumed_percent,
        remaining_minutes,
        health.value,
    )

    return ErrorBudgetResult(
        slo_id=slo_id,
        target_availability=target_availability,
        actual_availability=actual_availability,
        window_days=window_days,
        total_budget_minutes=total_budget_minutes,
        consumed_minutes=consumed_minutes,
        remaining_minutes=remaining_minutes,
        consumed_percent=consumed_percent,
        remaining_percent=remaining_percent,
        health=health,
        exhaustion_forecast_hours=exhaustion_forecast_hours,
    )


def _classify_budget_health(consumed_percent: float) -> BudgetHealth:
    """Classify error budget health from consumed percentage.

    Thresholds:
      - < 50%  -> HEALTHY
      - < 80%  -> WARNING
      - < 100% -> CRITICAL
      - >= 100% -> EXHAUSTED

    Args:
        consumed_percent: Budget consumed as a percentage (0-100).

    Returns:
        BudgetHealth classification.
    """
    if consumed_percent >= 100.0:
        return BudgetHealth.EXHAUSTED
    if consumed_percent >= 80.0:
        return BudgetHealth.CRITICAL
    if consumed_percent >= 50.0:
        return BudgetHealth.WARNING
    return BudgetHealth.HEALTHY


# ---------------------------------------------------------------------------
# Convenience: compliance summary
# ---------------------------------------------------------------------------


def compliance_summary(
    current_values: Dict[str, float],
    actual_availability: float = 100.0,
    slo: Optional[SLODefinition] = None,
) -> Dict[str, Any]:
    """Generate a combined SLI + error budget compliance summary.

    Args:
        current_values: Mapping of ``sli_id`` to current measured value.
        actual_availability: Current availability percentage.
        slo: The SLO to evaluate. Defaults to ``FACTORS_SLO``.

    Returns:
        Dictionary with ``sli_results``, ``error_budget``, and ``overall_status``.
    """
    target_slo = slo or FACTORS_SLO

    sli_results = check_all_slis(current_values, target_slo)
    budget = calculate_error_budget(
        target_availability=target_slo.target_availability,
        actual_availability=actual_availability,
        window_days=target_slo.window_days,
        slo_id=target_slo.slo_id,
    )

    all_met = all(r.met for r in sli_results if r.status != ComplianceStatus.UNKNOWN)
    budget_ok = budget.health in (BudgetHealth.HEALTHY, BudgetHealth.WARNING)

    if all_met and budget_ok:
        overall = "compliant"
    elif not all_met and budget.health == BudgetHealth.EXHAUSTED:
        overall = "critical"
    else:
        overall = "at_risk"

    return {
        "slo_id": target_slo.slo_id,
        "service": target_slo.service,
        "overall_status": overall,
        "sli_results": [r.to_dict() for r in sli_results],
        "error_budget": budget.to_dict(),
        "checked_at": sli_results[0].checked_at if sli_results else None,
    }


__all__ = [
    "SLICategory",
    "ComplianceStatus",
    "BudgetHealth",
    "SLIDefinition",
    "SLODefinition",
    "SLICheckResult",
    "ErrorBudgetResult",
    "SLI_REQUEST_LATENCY_P99",
    "SLI_ERROR_RATE",
    "SLI_SEARCH_LATENCY_P95",
    "SLI_MATCH_CONFIDENCE_MEDIAN",
    "SLI_AVAILABILITY",
    "FACTORS_SLO",
    "FACTORS_SLOS",
    "check_sli",
    "check_all_slis",
    "calculate_error_budget",
    "compliance_summary",
]
