# -*- coding: utf-8 -*-
"""
Quarterly Monitoring Workflow
====================================

4-phase DAG workflow for quarterly emissions monitoring and early-warning
alerting within PACK-029 Interim Targets Pack.  The workflow collects
quarterly actual emissions, compares against quarterly milestone targets,
generates RAG-scored alerts, and triggers corrective actions for red/amber
status.

Phases:
    1. CollectQuarterly    -- Collect quarterly actual emissions data
                              (annualized run-rate calculations)
    2. CompareMilestone    -- Compare actual vs. quarterly milestone
                              using ProgressTrackerEngine
    3. GenerateAlerts      -- Generate RAG alerts with severity scoring
                              (red/amber/green traffic-light system)
    4. TriggerCorrective   -- Trigger corrective action workflow if
                              status is red or amber

Regulatory references:
    - SBTi Target Tracking Protocol (quarterly cadence)
    - GHG Protocol Corporate Standard (interim reporting)
    - CDP Climate Disclosure (C4 progress tracking)
    - TCFD Implementation Monitoring

Zero-hallucination: all calculations use deterministic formulas.
No LLM calls in computation path.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


class Quarter(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"
    INFO = "info"


class AlertCategory(str, Enum):
    ABSOLUTE_GAP = "absolute_gap"
    INTENSITY_GAP = "intensity_gap"
    TREND_REVERSAL = "trend_reversal"
    BUDGET_DEPLETION = "budget_depletion"
    DATA_QUALITY = "data_quality"
    SCOPE_SPECIFIC = "scope_specific"
    RUN_RATE_DEVIATION = "run_rate_deviation"


class CorrectiveActionPriority(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    MONITOR = "monitor"


# =============================================================================
# QUARTERLY ALERT RULES (Zero-Hallucination: Deterministic Thresholds)
# =============================================================================


QUARTERLY_ALERT_RULES: Dict[str, Dict[str, Any]] = {
    "QAR-001": {
        "name": "Quarterly Absolute Gap > 15%",
        "condition": "quarterly_gap_pct > 15",
        "severity": "critical",
        "category": "absolute_gap",
        "description": "Quarterly emissions exceed target by more than 15%.",
        "action": "Immediate escalation to sustainability team; identify root cause within 5 business days.",
        "rag": "red",
    },
    "QAR-002": {
        "name": "Quarterly Absolute Gap 5-15%",
        "condition": "5 < quarterly_gap_pct <= 15",
        "severity": "high",
        "category": "absolute_gap",
        "description": "Quarterly emissions exceed target by 5-15%.",
        "action": "Review operations for quick-win reductions; report to management within 2 weeks.",
        "rag": "amber",
    },
    "QAR-003": {
        "name": "Annualized Run-Rate Exceeds Annual Target",
        "condition": "annualized_run_rate > annual_target * 1.05",
        "severity": "high",
        "category": "run_rate_deviation",
        "description": "Annualized run-rate based on YTD emissions exceeds full-year target.",
        "action": "Project corrective measures needed for remaining quarters.",
        "rag": "amber",
    },
    "QAR-004": {
        "name": "Quarter-over-Quarter Increase > 5%",
        "condition": "qoq_change_pct > 5",
        "severity": "warning",
        "category": "trend_reversal",
        "description": "Emissions increased by more than 5% compared to previous quarter.",
        "action": "Investigate driver of increase; assess if seasonal or systemic.",
        "rag": "amber",
    },
    "QAR-005": {
        "name": "Carbon Budget > 80% Consumed",
        "condition": "budget_consumed_pct > 80",
        "severity": "critical",
        "category": "budget_depletion",
        "description": "More than 80% of carbon budget consumed before proportional time elapsed.",
        "action": "Emergency budget review; implement immediate reduction measures.",
        "rag": "red",
    },
    "QAR-006": {
        "name": "Carbon Budget 60-80% Consumed",
        "condition": "60 < budget_consumed_pct <= 80",
        "severity": "warning",
        "category": "budget_depletion",
        "description": "Carbon budget consumption is ahead of schedule.",
        "action": "Tighten quarterly targets; review upcoming reduction initiatives.",
        "rag": "amber",
    },
    "QAR-007": {
        "name": "Scope 1 Spike > 20%",
        "condition": "scope1_qoq_change > 20",
        "severity": "critical",
        "category": "scope_specific",
        "description": "Scope 1 emissions spiked by more than 20% quarter-over-quarter.",
        "action": "Investigate process/fugitive emissions; check for equipment failures.",
        "rag": "red",
    },
    "QAR-008": {
        "name": "Data Quality Degradation",
        "condition": "data_quality_score < 3.0",
        "severity": "warning",
        "category": "data_quality",
        "description": "Quarterly data quality score below acceptable threshold.",
        "action": "Review data collection processes; engage data stewards.",
        "rag": "amber",
    },
    "QAR-009": {
        "name": "Intensity Increase Despite Revenue Growth",
        "condition": "intensity_qoq_change > 0 and revenue_qoq_change > 0",
        "severity": "high",
        "category": "intensity_gap",
        "description": "Emissions intensity increased even though revenue grew.",
        "action": "Decouple emissions from growth; review efficiency programs.",
        "rag": "amber",
    },
    "QAR-010": {
        "name": "On Track - Continue Monitoring",
        "condition": "quarterly_gap_pct <= 0",
        "severity": "info",
        "category": "absolute_gap",
        "description": "Quarterly emissions are at or below target.",
        "action": "Maintain current trajectory; document successful practices.",
        "rag": "green",
    },
}

# Quarterly seasonality adjustment factors (typical patterns)
SEASONALITY_FACTORS: Dict[str, Dict[str, float]] = {
    "standard": {"Q1": 1.05, "Q2": 0.95, "Q3": 0.92, "Q4": 1.08},
    "heating_dominant": {"Q1": 1.20, "Q2": 0.85, "Q3": 0.75, "Q4": 1.20},
    "cooling_dominant": {"Q1": 0.85, "Q2": 1.05, "Q3": 1.25, "Q4": 0.85},
    "manufacturing": {"Q1": 1.00, "Q2": 1.05, "Q3": 0.95, "Q4": 1.00},
    "flat": {"Q1": 1.00, "Q2": 1.00, "Q3": 1.00, "Q4": 1.00},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class QuarterlyEmissions(BaseModel):
    """Quarterly emissions data."""
    year: int = Field(default=2025)
    quarter: Quarter = Field(default=Quarter.Q1)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_musd: float = Field(default=0.0, ge=0.0)
    intensity: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=3.0, ge=0.0, le=5.0)
    provenance_hash: str = Field(default="")


class QuarterlyMilestone(BaseModel):
    """Quarterly milestone target."""
    year: int = Field(default=2025)
    quarter: Quarter = Field(default=Quarter.Q1)
    target_tco2e: float = Field(default=0.0, ge=0.0)
    seasonality_factor: float = Field(default=1.0)
    adjusted_target_tco2e: float = Field(default=0.0, ge=0.0)
    cumulative_ytd_target_tco2e: float = Field(default=0.0, ge=0.0)


class QuarterlyComparison(BaseModel):
    """Comparison of quarterly actual vs milestone."""
    quarter: Quarter = Field(default=Quarter.Q1)
    year: int = Field(default=2025)
    target_tco2e: float = Field(default=0.0)
    actual_tco2e: float = Field(default=0.0)
    gap_tco2e: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    ytd_target_tco2e: float = Field(default=0.0)
    ytd_actual_tco2e: float = Field(default=0.0)
    ytd_gap_pct: float = Field(default=0.0)
    annualized_run_rate_tco2e: float = Field(default=0.0)
    annual_target_tco2e: float = Field(default=0.0)
    run_rate_vs_target_pct: float = Field(default=0.0)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    qoq_change_pct: float = Field(default=0.0)
    scope1_qoq_change_pct: float = Field(default=0.0)


class QuarterlyAlert(BaseModel):
    """A single quarterly monitoring alert."""
    alert_id: str = Field(default="")
    rule_id: str = Field(default="")
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    category: AlertCategory = Field(default=AlertCategory.ABSOLUTE_GAP)
    title: str = Field(default="")
    description: str = Field(default="")
    metric_value: str = Field(default="")
    threshold_value: str = Field(default="")
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    recommended_action: str = Field(default="")
    escalation_required: bool = Field(default=False)


class CorrectiveActionTrigger(BaseModel):
    """Trigger for corrective action workflow."""
    trigger_id: str = Field(default="")
    triggered: bool = Field(default=False)
    trigger_reason: str = Field(default="")
    priority: CorrectiveActionPriority = Field(default=CorrectiveActionPriority.MONITOR)
    gap_to_close_tco2e: float = Field(default=0.0)
    required_quarterly_reduction_tco2e: float = Field(default=0.0)
    remaining_quarters_in_year: int = Field(default=0)
    recommended_initiatives: List[str] = Field(default_factory=list)
    escalation_contacts: List[str] = Field(default_factory=list)


class QuarterlyMonitoringReport(BaseModel):
    """Complete quarterly monitoring report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    year: int = Field(default=2025)
    quarter: Quarter = Field(default=Quarter.Q1)
    company_name: str = Field(default="")
    actuals: QuarterlyEmissions = Field(default_factory=QuarterlyEmissions)
    comparison: QuarterlyComparison = Field(default_factory=QuarterlyComparison)
    alerts: List[QuarterlyAlert] = Field(default_factory=list)
    corrective_trigger: CorrectiveActionTrigger = Field(default_factory=CorrectiveActionTrigger)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    executive_summary: str = Field(default="")
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class QuarterlyMonitoringConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    year: int = Field(default=2025, ge=2020, le=2060)
    quarter: Quarter = Field(default=Quarter.Q1)
    annual_target_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    carbon_budget_total_tco2e: float = Field(default=0.0, ge=0.0)
    carbon_budget_consumed_tco2e: float = Field(default=0.0, ge=0.0)
    seasonality_profile: str = Field(default="standard")
    red_threshold_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    amber_threshold_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    auto_trigger_corrective: bool = Field(default=True)


class QuarterlyMonitoringInput(BaseModel):
    config: QuarterlyMonitoringConfig = Field(default_factory=QuarterlyMonitoringConfig)
    current_quarter: QuarterlyEmissions = Field(default_factory=QuarterlyEmissions)
    previous_quarters: List[QuarterlyEmissions] = Field(
        default_factory=list,
        description="Previous quarters in the same year (for YTD calculation)",
    )
    previous_year_same_quarter: Optional[QuarterlyEmissions] = Field(default=None)


class QuarterlyMonitoringResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="quarterly_monitoring")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    actuals: QuarterlyEmissions = Field(default_factory=QuarterlyEmissions)
    comparison: QuarterlyComparison = Field(default_factory=QuarterlyComparison)
    alerts: List[QuarterlyAlert] = Field(default_factory=list)
    corrective_trigger: CorrectiveActionTrigger = Field(default_factory=CorrectiveActionTrigger)
    report: QuarterlyMonitoringReport = Field(default_factory=QuarterlyMonitoringReport)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    key_findings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class QuarterlyMonitoringWorkflow:
    """
    4-phase DAG workflow for quarterly emissions monitoring.

    Phase 1: CollectQuarterly   -- Collect and validate quarterly emissions.
    Phase 2: CompareMilestone   -- Compare actual vs. quarterly milestone; RAG.
    Phase 3: GenerateAlerts     -- Generate alerts (red/amber/green scoring).
    Phase 4: TriggerCorrective  -- Trigger corrective action if red/amber.

    DAG Dependencies:
        Phase 1 -> Phase 2 -> Phase 3 -> Phase 4

    Example:
        >>> wf = QuarterlyMonitoringWorkflow()
        >>> inp = QuarterlyMonitoringInput(
        ...     config=QuarterlyMonitoringConfig(
        ...         company_name="Acme Corp",
        ...         annual_target_tco2e=80000,
        ...     ),
        ...     current_quarter=QuarterlyEmissions(
        ...         year=2025, quarter=Quarter.Q1, total_tco2e=22000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[QuarterlyMonitoringConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or QuarterlyMonitoringConfig()
        self._phase_results: List[PhaseResult] = []
        self._actuals: QuarterlyEmissions = QuarterlyEmissions()
        self._comparison: QuarterlyComparison = QuarterlyComparison()
        self._alerts: List[QuarterlyAlert] = []
        self._corrective: CorrectiveActionTrigger = CorrectiveActionTrigger()
        self._report: QuarterlyMonitoringReport = QuarterlyMonitoringReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: QuarterlyMonitoringInput) -> QuarterlyMonitoringResult:
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting quarterly monitoring workflow %s, %s %d, company=%s",
            self.workflow_id, self.config.quarter.value, self.config.year,
            self.config.company_name,
        )

        try:
            phase1 = await self._phase_collect_quarterly(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_compare_milestone(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_generate_alerts(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_trigger_corrective(input_data)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Quarterly monitoring failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        overall_rag = self._comparison.rag_status

        # Build report
        exec_parts = [
            f"Quarterly Monitoring Report: {self.config.quarter.value} {self.config.year} - {self.config.company_name or 'Company'}.",
            f"Quarterly emissions: {self._actuals.total_tco2e:,.0f} tCO2e.",
            f"Target: {self._comparison.target_tco2e:,.0f} tCO2e.",
            f"Gap: {self._comparison.gap_tco2e:,.0f} tCO2e ({self._comparison.gap_pct:+.1f}%).",
            f"Annualized run-rate: {self._comparison.annualized_run_rate_tco2e:,.0f} tCO2e.",
            f"Status: {overall_rag.value.upper()}.",
        ]
        if self._corrective.triggered:
            exec_parts.append(
                f"Corrective action triggered: {self._corrective.trigger_reason}.",
            )

        self._report = QuarterlyMonitoringReport(
            report_id=f"QMR-{self.workflow_id[:8]}",
            report_date=_utcnow().strftime("%Y-%m-%d"),
            year=self.config.year,
            quarter=self.config.quarter,
            company_name=self.config.company_name,
            actuals=self._actuals,
            comparison=self._comparison,
            alerts=self._alerts,
            corrective_trigger=self._corrective,
            overall_rag=overall_rag,
            executive_summary=" ".join(exec_parts),
            key_metrics={
                "quarterly_emissions_tco2e": self._actuals.total_tco2e,
                "quarterly_target_tco2e": self._comparison.target_tco2e,
                "gap_pct": self._comparison.gap_pct,
                "annualized_run_rate": self._comparison.annualized_run_rate_tco2e,
                "rag_status": overall_rag.value,
                "alerts_count": len(self._alerts),
                "corrective_triggered": self._corrective.triggered,
            },
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        result = QuarterlyMonitoringResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            actuals=self._actuals,
            comparison=self._comparison,
            alerts=self._alerts,
            corrective_trigger=self._corrective,
            report=self._report,
            overall_rag=overall_rag,
            key_findings=self._generate_findings(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Collect Quarterly Data
    # -------------------------------------------------------------------------

    async def _phase_collect_quarterly(self, input_data: QuarterlyMonitoringInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        q = input_data.current_quarter
        if q.total_tco2e <= 0:
            q.total_tco2e = q.scope1_tco2e + q.scope2_tco2e + q.scope3_tco2e
            if q.total_tco2e <= 0:
                annual = self.config.annual_target_tco2e or self.config.base_year_emissions_tco2e
                q.total_tco2e = annual * 0.25 * 1.05  # Slightly above quarterly target
                q.scope1_tco2e = q.total_tco2e * 0.45
                q.scope2_tco2e = q.total_tco2e * 0.20
                q.scope3_tco2e = q.total_tco2e * 0.35
                warnings.append("Quarterly emissions estimated from annual target.")

        q.year = self.config.year
        q.quarter = self.config.quarter

        if q.revenue_musd > 0:
            q.intensity = round(q.total_tco2e / q.revenue_musd, 4)

        q.provenance_hash = _compute_hash(q.model_dump_json(exclude={"provenance_hash"}))
        self._actuals = q

        outputs["year"] = q.year
        outputs["quarter"] = q.quarter.value
        outputs["total_tco2e"] = q.total_tco2e
        outputs["scope1_tco2e"] = q.scope1_tco2e
        outputs["scope2_tco2e"] = q.scope2_tco2e
        outputs["scope3_tco2e"] = q.scope3_tco2e
        outputs["data_quality_score"] = q.data_quality_score

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="collect_quarterly", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_collect_quarterly",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Compare Milestone
    # -------------------------------------------------------------------------

    async def _phase_compare_milestone(self, input_data: QuarterlyMonitoringInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        annual_target = self.config.annual_target_tco2e
        if annual_target <= 0:
            annual_target = self.config.base_year_emissions_tco2e * 0.95
            warnings.append("Annual target estimated from base year emissions.")

        # Apply seasonality adjustment
        season_profile = SEASONALITY_FACTORS.get(
            self.config.seasonality_profile, SEASONALITY_FACTORS["standard"],
        )
        season_factor = season_profile.get(self.config.quarter.value, 1.0)

        quarterly_target = (annual_target / 4.0) * season_factor

        # Gap calculation
        actual = self._actuals.total_tco2e
        gap = actual - quarterly_target
        gap_pct = (gap / max(quarterly_target, 1e-10)) * 100

        # YTD calculation
        ytd_actual = actual + sum(pq.total_tco2e for pq in input_data.previous_quarters)
        quarters_elapsed = len(input_data.previous_quarters) + 1
        ytd_target = 0.0
        quarter_order = [Quarter.Q1, Quarter.Q2, Quarter.Q3, Quarter.Q4]
        for i in range(quarters_elapsed):
            q_key = quarter_order[i].value
            sf = season_profile.get(q_key, 1.0)
            ytd_target += (annual_target / 4.0) * sf
        ytd_gap_pct = ((ytd_actual - ytd_target) / max(ytd_target, 1e-10)) * 100

        # Annualized run-rate
        if quarters_elapsed > 0:
            run_rate = (ytd_actual / quarters_elapsed) * 4
        else:
            run_rate = actual * 4
        run_rate_vs_target = ((run_rate - annual_target) / max(annual_target, 1e-10)) * 100

        # QoQ change
        prev_quarters = input_data.previous_quarters
        qoq_change = 0.0
        scope1_qoq = 0.0
        if prev_quarters:
            last_q = prev_quarters[-1]
            if last_q.total_tco2e > 0:
                qoq_change = ((actual - last_q.total_tco2e) / last_q.total_tco2e) * 100
            if last_q.scope1_tco2e > 0:
                scope1_qoq = ((self._actuals.scope1_tco2e - last_q.scope1_tco2e) / last_q.scope1_tco2e) * 100

        # RAG status
        if gap_pct > self.config.red_threshold_pct:
            rag = RAGStatus.RED
        elif gap_pct > self.config.amber_threshold_pct:
            rag = RAGStatus.AMBER
        else:
            rag = RAGStatus.GREEN

        self._comparison = QuarterlyComparison(
            quarter=self.config.quarter,
            year=self.config.year,
            target_tco2e=round(quarterly_target, 2),
            actual_tco2e=round(actual, 2),
            gap_tco2e=round(gap, 2),
            gap_pct=round(gap_pct, 2),
            ytd_target_tco2e=round(ytd_target, 2),
            ytd_actual_tco2e=round(ytd_actual, 2),
            ytd_gap_pct=round(ytd_gap_pct, 2),
            annualized_run_rate_tco2e=round(run_rate, 2),
            annual_target_tco2e=round(annual_target, 2),
            run_rate_vs_target_pct=round(run_rate_vs_target, 2),
            rag_status=rag,
            qoq_change_pct=round(qoq_change, 2),
            scope1_qoq_change_pct=round(scope1_qoq, 2),
        )

        outputs["quarterly_target_tco2e"] = round(quarterly_target, 2)
        outputs["gap_tco2e"] = round(gap, 2)
        outputs["gap_pct"] = round(gap_pct, 2)
        outputs["ytd_gap_pct"] = round(ytd_gap_pct, 2)
        outputs["annualized_run_rate"] = round(run_rate, 2)
        outputs["run_rate_vs_target_pct"] = round(run_rate_vs_target, 2)
        outputs["rag_status"] = rag.value
        outputs["qoq_change_pct"] = round(qoq_change, 2)
        outputs["seasonality_factor"] = season_factor

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="compare_milestone", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_compare_milestone",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Generate Alerts
    # -------------------------------------------------------------------------

    async def _phase_generate_alerts(self, input_data: QuarterlyMonitoringInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        alerts: List[QuarterlyAlert] = []
        comp = self._comparison

        # QAR-001: Critical absolute gap
        if comp.gap_pct > self.config.red_threshold_pct:
            rule = QUARTERLY_ALERT_RULES["QAR-001"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-001",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ABSOLUTE_GAP,
                title=rule["name"],
                description=rule["description"],
                metric_value=f"{comp.gap_pct:.1f}%",
                threshold_value=f">{self.config.red_threshold_pct}%",
                rag_status=RAGStatus.RED,
                recommended_action=rule["action"],
                escalation_required=True,
            ))
        # QAR-002: Amber absolute gap
        elif comp.gap_pct > self.config.amber_threshold_pct:
            rule = QUARTERLY_ALERT_RULES["QAR-002"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-002",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.ABSOLUTE_GAP,
                title=rule["name"],
                description=rule["description"],
                metric_value=f"{comp.gap_pct:.1f}%",
                threshold_value=f"{self.config.amber_threshold_pct}-{self.config.red_threshold_pct}%",
                rag_status=RAGStatus.AMBER,
                recommended_action=rule["action"],
            ))

        # QAR-003: Run-rate exceeds annual target
        if comp.run_rate_vs_target_pct > 5:
            rule = QUARTERLY_ALERT_RULES["QAR-003"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-003",
                severity=AlertSeverity.HIGH,
                category=AlertCategory.RUN_RATE_DEVIATION,
                title=rule["name"],
                description=f"Annualized run-rate ({comp.annualized_run_rate_tco2e:,.0f}) exceeds annual target ({comp.annual_target_tco2e:,.0f}) by {comp.run_rate_vs_target_pct:.1f}%.",
                metric_value=f"{comp.annualized_run_rate_tco2e:,.0f} tCO2e",
                threshold_value=f"{comp.annual_target_tco2e:,.0f} tCO2e",
                rag_status=RAGStatus.AMBER,
                recommended_action=rule["action"],
            ))

        # QAR-004: QoQ increase
        if comp.qoq_change_pct > 5:
            rule = QUARTERLY_ALERT_RULES["QAR-004"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-004",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TREND_REVERSAL,
                title=rule["name"],
                description=f"Emissions increased {comp.qoq_change_pct:.1f}% vs. previous quarter.",
                rag_status=RAGStatus.AMBER,
                recommended_action=rule["action"],
            ))

        # QAR-005/006: Budget depletion
        if self.config.carbon_budget_total_tco2e > 0:
            consumed = self.config.carbon_budget_consumed_tco2e + self._actuals.total_tco2e
            consumed_pct = (consumed / self.config.carbon_budget_total_tco2e) * 100
            if consumed_pct > 80:
                rule = QUARTERLY_ALERT_RULES["QAR-005"]
                alerts.append(QuarterlyAlert(
                    alert_id=f"QA-{_new_uuid()[:6]}",
                    rule_id="QAR-005",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.BUDGET_DEPLETION,
                    title=rule["name"],
                    description=f"Carbon budget {consumed_pct:.0f}% consumed.",
                    rag_status=RAGStatus.RED,
                    recommended_action=rule["action"],
                    escalation_required=True,
                ))
            elif consumed_pct > 60:
                rule = QUARTERLY_ALERT_RULES["QAR-006"]
                alerts.append(QuarterlyAlert(
                    alert_id=f"QA-{_new_uuid()[:6]}",
                    rule_id="QAR-006",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.BUDGET_DEPLETION,
                    title=rule["name"],
                    description=f"Carbon budget {consumed_pct:.0f}% consumed.",
                    rag_status=RAGStatus.AMBER,
                    recommended_action=rule["action"],
                ))

        # QAR-007: Scope 1 spike
        if comp.scope1_qoq_change_pct > 20:
            rule = QUARTERLY_ALERT_RULES["QAR-007"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-007",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SCOPE_SPECIFIC,
                title=rule["name"],
                description=f"Scope 1 increased {comp.scope1_qoq_change_pct:.1f}% QoQ.",
                rag_status=RAGStatus.RED,
                recommended_action=rule["action"],
                escalation_required=True,
            ))

        # QAR-008: Data quality
        if self._actuals.data_quality_score < 3.0:
            rule = QUARTERLY_ALERT_RULES["QAR-008"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-008",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.DATA_QUALITY,
                title=rule["name"],
                description=f"Data quality score: {self._actuals.data_quality_score:.1f}.",
                rag_status=RAGStatus.AMBER,
                recommended_action=rule["action"],
            ))

        # QAR-010: On track
        if comp.gap_pct <= 0:
            rule = QUARTERLY_ALERT_RULES["QAR-010"]
            alerts.append(QuarterlyAlert(
                alert_id=f"QA-{_new_uuid()[:6]}",
                rule_id="QAR-010",
                severity=AlertSeverity.INFO,
                category=AlertCategory.ABSOLUTE_GAP,
                title=rule["name"],
                description=f"Quarterly emissions ({self._actuals.total_tco2e:,.0f}) at/below target ({comp.target_tco2e:,.0f}).",
                rag_status=RAGStatus.GREEN,
                recommended_action=rule["action"],
            ))

        self._alerts = alerts

        outputs["alerts_count"] = len(alerts)
        outputs["critical_count"] = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        outputs["high_count"] = sum(1 for a in alerts if a.severity == AlertSeverity.HIGH)
        outputs["warning_count"] = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
        outputs["info_count"] = sum(1 for a in alerts if a.severity == AlertSeverity.INFO)
        outputs["escalation_required"] = any(a.escalation_required for a in alerts)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="generate_alerts", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_generate_alerts",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Trigger Corrective Action
    # -------------------------------------------------------------------------

    async def _phase_trigger_corrective(self, input_data: QuarterlyMonitoringInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        rag = self._comparison.rag_status
        should_trigger = (
            self.config.auto_trigger_corrective
            and rag in (RAGStatus.RED, RAGStatus.AMBER)
        )

        # Remaining quarters
        q_idx = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        current_q_num = q_idx.get(self.config.quarter.value, 1)
        remaining_q = 4 - current_q_num

        # Gap to close
        ytd_gap = self._comparison.ytd_actual_tco2e - self._comparison.ytd_target_tco2e
        annual_gap = self._comparison.annualized_run_rate_tco2e - self._comparison.annual_target_tco2e

        required_per_q = 0.0
        if remaining_q > 0 and annual_gap > 0:
            required_per_q = annual_gap / remaining_q

        # Priority
        if rag == RAGStatus.RED:
            priority = CorrectiveActionPriority.IMMEDIATE
        elif rag == RAGStatus.AMBER:
            priority = CorrectiveActionPriority.SHORT_TERM
        else:
            priority = CorrectiveActionPriority.MONITOR

        # Recommended initiatives
        initiatives: List[str] = []
        if rag == RAGStatus.RED:
            initiatives.extend([
                "Emergency operations review for quick-win emission reductions.",
                "Accelerate planned technology deployments by 1-2 quarters.",
                "Engage top 5 suppliers for immediate Scope 3 reductions.",
                "Implement temporary operational measures (e.g., load shifting, demand response).",
                "Board-level escalation with corrective action plan within 10 business days.",
            ])
        elif rag == RAGStatus.AMBER:
            initiatives.extend([
                "Review and prioritize planned abatement initiatives.",
                "Identify operational efficiency improvements for next quarter.",
                "Engage energy procurement team on renewable energy options.",
                "Tighten monitoring cadence to monthly for critical sources.",
            ])

        trigger_reason = ""
        if should_trigger:
            trigger_reason = (
                f"Quarterly emissions gap: {self._comparison.gap_pct:+.1f}% "
                f"({self._comparison.gap_tco2e:,.0f} tCO2e above target). "
                f"RAG status: {rag.value.upper()}."
            )

        self._corrective = CorrectiveActionTrigger(
            trigger_id=f"CAT-{self.workflow_id[:8]}",
            triggered=should_trigger,
            trigger_reason=trigger_reason,
            priority=priority,
            gap_to_close_tco2e=round(max(annual_gap, 0), 2),
            required_quarterly_reduction_tco2e=round(max(required_per_q, 0), 2),
            remaining_quarters_in_year=remaining_q,
            recommended_initiatives=initiatives,
            escalation_contacts=[
                "Sustainability Director",
                "Chief Operations Officer",
                "Board Sustainability Committee",
            ] if rag == RAGStatus.RED else [
                "Sustainability Manager",
                "Operations Manager",
            ],
        )

        outputs["triggered"] = should_trigger
        outputs["priority"] = priority.value
        outputs["gap_to_close_tco2e"] = round(max(annual_gap, 0), 2)
        outputs["required_per_quarter_tco2e"] = round(max(required_per_q, 0), 2)
        outputs["remaining_quarters"] = remaining_q
        outputs["initiatives_count"] = len(initiatives)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="trigger_corrective", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_trigger_corrective",
        )

    # -------------------------------------------------------------------------
    # Report Generator
    # -------------------------------------------------------------------------

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        c = self._comparison
        findings.append(
            f"{c.quarter.value} {c.year}: {self._actuals.total_tco2e:,.0f} tCO2e "
            f"vs. target {c.target_tco2e:,.0f} tCO2e "
            f"(gap: {c.gap_tco2e:+,.0f} tCO2e, {c.gap_pct:+.1f}%).",
        )
        findings.append(f"RAG status: {c.rag_status.value.upper()}.")
        findings.append(
            f"Annualized run-rate: {c.annualized_run_rate_tco2e:,.0f} tCO2e "
            f"vs. annual target {c.annual_target_tco2e:,.0f} tCO2e "
            f"({c.run_rate_vs_target_pct:+.1f}%).",
        )
        if c.qoq_change_pct != 0:
            findings.append(f"Quarter-over-quarter change: {c.qoq_change_pct:+.1f}%.")
        findings.append(f"Alerts generated: {len(self._alerts)}.")
        if self._corrective.triggered:
            findings.append(
                f"Corrective action triggered: {self._corrective.priority.value} priority.",
            )
        return findings
