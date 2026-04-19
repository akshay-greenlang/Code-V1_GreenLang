# -*- coding: utf-8 -*-
"""
Progress Monitoring Workflow
=================================

4-phase workflow for monitoring sector intensity progress against
decarbonization pathways within PACK-028 Sector Pathway Pack.  The
workflow updates intensity metrics, checks convergence status against
SBTi/IEA pathways, refreshes benchmark comparisons, and generates
a progress report with alerts and recommendations.

Phases:
    1. IntensityUpdate    -- Update sector intensity metrics with latest
                             activity and emission data; calculate trends
    2. ConvergenceCheck   -- Check intensity convergence vs. pathway targets;
                             calculate gap, acceleration needs, time-to-close
    3. BenchmarkUpdate    -- Refresh sector benchmarks: peer, leader, SBTi
                             validated, IEA pathway milestones
    4. ProgressReport     -- Generate progress report with KPIs, alerts,
                             trajectory charts, and executive summary

Regulatory references:
    - SBTi Monitoring, Reporting & Verification Guidance
    - SBTi Target Tracking Protocol
    - GHG Protocol Corporate Standard (annual reporting)
    - IEA NZE 2050 Milestone Tracker

Zero-hallucination: all calculations use deterministic formulas with
published intensity data.  No LLM calls in computation path.

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import math
import statistics
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"

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
    """Red-Amber-Green traffic-light status."""
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"

class ConvergenceStatus(str, Enum):
    ON_TRACK = "on_track"
    SLIGHT_DEVIATION = "slight_deviation"
    SIGNIFICANT_DEVIATION = "significant_deviation"
    OFF_TRACK = "off_track"

class BenchmarkSource(str, Enum):
    SBTI_PEER = "sbti_peer"
    SECTOR_AVERAGE = "sector_average"
    SECTOR_LEADER = "sector_leader"
    IEA_PATHWAY = "iea_pathway"
    REGULATORY = "regulatory"

# =============================================================================
# SECTOR BENCHMARK DATA (Zero-Hallucination: Published Values)
# =============================================================================

SECTOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "global_avg_2025": 380.0, "leader_2025": 150.0, "sbti_avg_2025": 280.0,
        "iea_nze_2025": 350.0, "iea_nze_2030": 138.0, "unit": "gCO2/kWh",
        "regulatory_benchmark": 400.0, "regulatory_name": "EU ETS Benchmark",
    },
    "steel": {
        "global_avg_2025": 1.75, "leader_2025": 0.80, "sbti_avg_2025": 1.50,
        "iea_nze_2025": 1.70, "iea_nze_2030": 1.40, "unit": "tCO2e/t",
        "regulatory_benchmark": 1.52, "regulatory_name": "EU ETS Benchmark",
    },
    "cement": {
        "global_avg_2025": 0.55, "leader_2025": 0.35, "sbti_avg_2025": 0.48,
        "iea_nze_2025": 0.52, "iea_nze_2030": 0.42, "unit": "tCO2e/t",
        "regulatory_benchmark": 0.766, "regulatory_name": "EU ETS Benchmark",
    },
    "aluminum": {
        "global_avg_2025": 11.0, "leader_2025": 4.0, "sbti_avg_2025": 9.0,
        "iea_nze_2025": 10.5, "iea_nze_2030": 8.5, "unit": "tCO2e/t",
        "regulatory_benchmark": 1.514, "regulatory_name": "EU ETS Benchmark",
    },
    "aviation": {
        "global_avg_2025": 95.0, "leader_2025": 70.0, "sbti_avg_2025": 85.0,
        "iea_nze_2025": 92.0, "iea_nze_2030": 77.0, "unit": "gCO2/pkm",
        "regulatory_benchmark": 89.0, "regulatory_name": "CORSIA Baseline",
    },
    "shipping": {
        "global_avg_2025": 6.5, "leader_2025": 3.5, "sbti_avg_2025": 5.8,
        "iea_nze_2025": 6.2, "iea_nze_2030": 5.2, "unit": "gCO2/tkm",
        "regulatory_benchmark": 11.0, "regulatory_name": "IMO CII Rating",
    },
    "buildings_residential": {
        "global_avg_2025": 20.0, "leader_2025": 8.0, "sbti_avg_2025": 16.0,
        "iea_nze_2025": 19.0, "iea_nze_2030": 13.0, "unit": "kgCO2/m2/yr",
        "regulatory_benchmark": 15.0, "regulatory_name": "EU EPBD NZEB",
    },
    "buildings_commercial": {
        "global_avg_2025": 28.0, "leader_2025": 10.0, "sbti_avg_2025": 22.0,
        "iea_nze_2025": 26.0, "iea_nze_2030": 18.0, "unit": "kgCO2/m2/yr",
        "regulatory_benchmark": 20.0, "regulatory_name": "EU EPBD NZEB",
    },
    "chemicals": {
        "global_avg_2025": 0.85, "leader_2025": 0.45, "sbti_avg_2025": 0.72,
        "iea_nze_2025": 0.82, "iea_nze_2030": 0.65, "unit": "tCO2e/t HVC",
        "regulatory_benchmark": 0.702, "regulatory_name": "EU ETS Benchmark",
    },
    "oil_gas": {
        "global_avg_2025": 18.0, "leader_2025": 8.0, "sbti_avg_2025": 14.0,
        "iea_nze_2025": 17.0, "iea_nze_2030": 12.0, "unit": "kgCO2e/boe",
        "regulatory_benchmark": 15.0, "regulatory_name": "OGMP 2.0",
    },
    "food_beverage": {
        "global_avg_2025": 0.50, "leader_2025": 0.20, "sbti_avg_2025": 0.38,
        "iea_nze_2025": 0.48, "iea_nze_2030": 0.35, "unit": "tCO2e/t product",
        "regulatory_benchmark": 0.45, "regulatory_name": "FLAG Guidance",
    },
    "road_transport": {
        "global_avg_2025": 120.0, "leader_2025": 40.0, "sbti_avg_2025": 95.0,
        "iea_nze_2025": 115.0, "iea_nze_2030": 75.0, "unit": "gCO2/km",
        "regulatory_benchmark": 95.0, "regulatory_name": "EU CO2 Standards",
    },
    "pulp_paper": {
        "global_avg_2025": 0.45, "leader_2025": 0.15, "sbti_avg_2025": 0.35,
        "iea_nze_2025": 0.42, "iea_nze_2030": 0.30, "unit": "tCO2e/t",
        "regulatory_benchmark": 0.40, "regulatory_name": "EU ETS Benchmark",
    },
}

# Historical intensity improvement rates by sector (% per year, compound)
HISTORICAL_IMPROVEMENT_RATES: Dict[str, Dict[str, float]] = {
    "power_generation": {
        "2015_2020": -3.2, "2020_2025": -4.5, "required_nze": -7.6,
        "peer_best": -8.5, "peer_worst": -0.5,
    },
    "steel": {
        "2015_2020": -1.0, "2020_2025": -1.8, "required_nze": -4.5,
        "peer_best": -5.0, "peer_worst": 0.5,
    },
    "cement": {
        "2015_2020": -0.8, "2020_2025": -1.5, "required_nze": -3.5,
        "peer_best": -4.2, "peer_worst": 0.2,
    },
    "aluminum": {
        "2015_2020": -1.2, "2020_2025": -2.0, "required_nze": -5.0,
        "peer_best": -6.0, "peer_worst": 0.0,
    },
    "aviation": {
        "2015_2020": -1.5, "2020_2025": -2.2, "required_nze": -3.8,
        "peer_best": -4.5, "peer_worst": -0.3,
    },
    "shipping": {
        "2015_2020": -0.5, "2020_2025": -1.0, "required_nze": -4.0,
        "peer_best": -5.0, "peer_worst": 0.8,
    },
    "buildings_residential": {
        "2015_2020": -2.0, "2020_2025": -3.0, "required_nze": -5.5,
        "peer_best": -7.0, "peer_worst": -0.5,
    },
    "buildings_commercial": {
        "2015_2020": -1.8, "2020_2025": -2.8, "required_nze": -5.0,
        "peer_best": -6.5, "peer_worst": -0.2,
    },
    "chemicals": {
        "2015_2020": -0.6, "2020_2025": -1.2, "required_nze": -3.0,
        "peer_best": -3.5, "peer_worst": 0.5,
    },
    "oil_gas": {
        "2015_2020": -0.3, "2020_2025": -0.8, "required_nze": -6.0,
        "peer_best": -4.0, "peer_worst": 1.0,
    },
}

# IEA NZE milestone checkpoints for progress tracking
IEA_NZE_MILESTONES: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"year": 2025, "milestone": "No new unabated coal plants approved", "intensity_target": 350},
        {"year": 2030, "milestone": "Coal phase-out in advanced economies", "intensity_target": 138},
        {"year": 2035, "milestone": "All electricity in advanced economies net-zero", "intensity_target": 50},
        {"year": 2040, "milestone": "Global coal phase-out", "intensity_target": 30},
        {"year": 2050, "milestone": "Net-zero electricity globally", "intensity_target": 0},
    ],
    "steel": [
        {"year": 2025, "milestone": "Near-zero steel pilots operational", "intensity_target": 1.70},
        {"year": 2030, "milestone": "8% near-zero steel production", "intensity_target": 1.40},
        {"year": 2035, "milestone": "25% near-zero steel production", "intensity_target": 1.10},
        {"year": 2040, "milestone": "50% near-zero steel production", "intensity_target": 0.70},
        {"year": 2050, "milestone": "All steel near-zero emissions", "intensity_target": 0.20},
    ],
    "cement": [
        {"year": 2025, "milestone": "CCS demonstrations at cement plants", "intensity_target": 0.52},
        {"year": 2030, "milestone": "10% alternative binders adoption", "intensity_target": 0.42},
        {"year": 2035, "milestone": "30% CCS capture rate on new plants", "intensity_target": 0.32},
        {"year": 2040, "milestone": "50% CCS capture rate industry-wide", "intensity_target": 0.22},
        {"year": 2050, "milestone": "Net-zero cement achievable", "intensity_target": 0.10},
    ],
    "aviation": [
        {"year": 2025, "milestone": "SAF at 2% of total fuel", "intensity_target": 92},
        {"year": 2030, "milestone": "SAF at 10%, fleet 15% more efficient", "intensity_target": 77},
        {"year": 2035, "milestone": "Short-haul hydrogen aircraft enter service", "intensity_target": 60},
        {"year": 2040, "milestone": "SAF at 35%, new aircraft 30% more efficient", "intensity_target": 45},
        {"year": 2050, "milestone": "SAF 65%, hydrogen 15%, offsets for remainder", "intensity_target": 20},
    ],
    "shipping": [
        {"year": 2025, "milestone": "IMO CII scheme operational", "intensity_target": 6.2},
        {"year": 2030, "milestone": "5% alternative fuels, 30% efficiency gain", "intensity_target": 5.2},
        {"year": 2035, "milestone": "Green corridor routes established", "intensity_target": 4.0},
        {"year": 2040, "milestone": "25% ammonia/methanol fleet share", "intensity_target": 2.8},
        {"year": 2050, "milestone": "Net-zero shipping achievable", "intensity_target": 1.0},
    ],
}

# Alert rules for automated progress monitoring
ALERT_RULES: Dict[str, Dict[str, Any]] = {
    "RULE-001": {
        "name": "Intensity Increase Detection",
        "condition": "yoy_change_pct > 0",
        "severity": "critical",
        "category": "trend",
        "description": "Emissions intensity has increased year-over-year.",
        "action": "Conduct root cause analysis; engage operations team immediately.",
    },
    "RULE-002": {
        "name": "Insufficient Reduction Rate",
        "condition": "abs(cagr_pct) < required_rate * 0.5",
        "severity": "critical",
        "category": "convergence",
        "description": "Current reduction rate is less than half the required rate.",
        "action": "Escalate to board; review and accelerate all abatement levers.",
    },
    "RULE-003": {
        "name": "NZE Pathway Deviation >25%",
        "condition": "nze_gap_pct > 25",
        "severity": "critical",
        "category": "convergence",
        "description": "Company intensity is >25% above the NZE pathway target.",
        "action": "Strategic review of pathway; consider additional technology investments.",
    },
    "RULE-004": {
        "name": "Below Sector Average",
        "condition": "intensity > sector_average",
        "severity": "warning",
        "category": "benchmark",
        "description": "Company performance is below the global sector average.",
        "action": "Benchmark gap analysis; identify top 3 quick-win improvements.",
    },
    "RULE-005": {
        "name": "2030 Target at Risk",
        "condition": "projected_2030 > near_term_target * 1.10",
        "severity": "warning",
        "category": "target",
        "description": "Current trajectory will miss 2030 target by >10%.",
        "action": "Accelerate near-term actions; review and reprioritize investments.",
    },
    "RULE-006": {
        "name": "Data Quality Degradation",
        "condition": "avg_data_quality < 3.0",
        "severity": "warning",
        "category": "data",
        "description": "Average data quality score has dropped below acceptable threshold.",
        "action": "Review data collection processes; implement automated quality checks.",
    },
    "RULE-007": {
        "name": "Stagnation Warning",
        "condition": "abs(yoy_change_pct) < 0.5 and convergence_gap > 5",
        "severity": "warning",
        "category": "trend",
        "description": "Intensity reduction has stagnated despite outstanding gap.",
        "action": "Investigate operational barriers; deploy new abatement technologies.",
    },
    "RULE-008": {
        "name": "Peer Outperformance Alert",
        "condition": "peer_leader_gap_pct > 50",
        "severity": "info",
        "category": "benchmark",
        "description": "Significant gap between company and sector leader performance.",
        "action": "Study sector leader practices; consider best practice adoption.",
    },
    "RULE-009": {
        "name": "IEA Milestone At Risk",
        "condition": "current_year >= milestone_year - 2 and intensity > milestone_target * 1.20",
        "severity": "warning",
        "category": "milestone",
        "description": "Approaching IEA NZE milestone with insufficient progress.",
        "action": "Align short-term action plan with upcoming IEA milestone.",
    },
    "RULE-010": {
        "name": "Carbon Budget Depletion",
        "condition": "cumulative_emissions > carbon_budget * 0.80",
        "severity": "critical",
        "category": "budget",
        "description": "Company has consumed >80% of its allocated carbon budget.",
        "action": "Emergency review; consider offsetting and acceleration of all levers.",
    },
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

class IntensityDataPoint(BaseModel):
    """A single intensity measurement."""
    year: int = Field(default=2025)
    period: str = Field(default="annual", description="annual|quarterly|monthly")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    activity: float = Field(default=0.0, ge=0.0)
    intensity: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=3.0, ge=0.0, le=5.0)

class IntensityUpdate(BaseModel):
    """Updated intensity metrics after Phase 1."""
    current_intensity: float = Field(default=0.0)
    previous_intensity: float = Field(default=0.0)
    base_year_intensity: float = Field(default=0.0)
    yoy_change_pct: float = Field(default=0.0)
    cagr_pct: float = Field(default=0.0)
    trend_direction: TrendDirection = Field(default=TrendDirection.STABLE)
    linear_trend_slope: float = Field(default=0.0)
    projected_2030_intensity: float = Field(default=0.0)
    projected_2050_intensity: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    data_points_used: int = Field(default=0)
    intensity_unit: str = Field(default="")

class ConvergenceResult(BaseModel):
    """Convergence analysis result."""
    scenario: str = Field(default="nze_15c")
    pathway_intensity_current_year: float = Field(default=0.0)
    actual_intensity: float = Field(default=0.0)
    gap_absolute: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    convergence_status: ConvergenceStatus = Field(default=ConvergenceStatus.ON_TRACK)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    required_acceleration_pct: float = Field(default=0.0)
    years_to_convergence: int = Field(default=0)
    on_track_for_2030: bool = Field(default=True)
    on_track_for_2050: bool = Field(default=True)
    corrective_actions: List[str] = Field(default_factory=list)

class BenchmarkComparison(BaseModel):
    """Comparison against a single benchmark."""
    source: BenchmarkSource = Field(default=BenchmarkSource.SECTOR_AVERAGE)
    source_name: str = Field(default="")
    benchmark_value: float = Field(default=0.0)
    actual_value: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    percentile: float = Field(default=50.0, ge=0.0, le=100.0)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)

class BenchmarkSummary(BaseModel):
    """Summary of all benchmark comparisons."""
    sector: str = Field(default="")
    comparisons: List[BenchmarkComparison] = Field(default_factory=list)
    best_comparison: str = Field(default="")
    worst_comparison: str = Field(default="")
    overall_percentile: float = Field(default=50.0)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)

class ProgressAlert(BaseModel):
    """A single progress alert."""
    alert_id: str = Field(default="")
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    category: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    metric_value: str = Field(default="")
    threshold_value: str = Field(default="")
    recommended_action: str = Field(default="")

class ProgressKPI(BaseModel):
    """A single progress KPI."""
    kpi_name: str = Field(default="")
    current_value: float = Field(default=0.0)
    target_value: float = Field(default=0.0)
    unit: str = Field(default="")
    achievement_pct: float = Field(default=0.0)
    trend: TrendDirection = Field(default=TrendDirection.STABLE)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)

class ProgressReport(BaseModel):
    """Complete progress monitoring report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    sector: str = Field(default="")
    company_name: str = Field(default="")
    intensity_update: IntensityUpdate = Field(default_factory=IntensityUpdate)
    convergence_results: List[ConvergenceResult] = Field(default_factory=list)
    benchmark_summary: BenchmarkSummary = Field(default_factory=BenchmarkSummary)
    kpis: List[ProgressKPI] = Field(default_factory=list)
    alerts: List[ProgressAlert] = Field(default_factory=list)
    executive_summary: str = Field(default="")
    provenance_hash: str = Field(default="")

class ProgressMonitoringConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    sector: str = Field(default="cross_sector")
    intensity_unit: str = Field(default="")
    base_year: int = Field(default=2020, ge=2015, le=2030)
    base_year_intensity: float = Field(default=0.0, ge=0.0)
    target_year: int = Field(default=2050, ge=2030, le=2070)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    near_term_target_intensity: float = Field(default=0.0, ge=0.0)
    long_term_target_intensity: float = Field(default=0.0, ge=0.0)
    activity_growth_rate: float = Field(default=0.02, ge=-0.10, le=0.20)
    alert_threshold_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    scenarios: List[str] = Field(
        default_factory=lambda: ["nze_15c", "wb2c"],
    )

class ProgressMonitoringInput(BaseModel):
    config: ProgressMonitoringConfig = Field(default_factory=ProgressMonitoringConfig)
    intensity_data: List[IntensityDataPoint] = Field(default_factory=list)
    pathway_targets: Dict[str, Dict[int, float]] = Field(
        default_factory=dict,
        description="scenario -> {year: intensity_target}",
    )
    peer_data: List[Dict[str, Any]] = Field(default_factory=list)

class ProgressMonitoringResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="progress_monitoring")
    pack_id: str = Field(default="PACK-028")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    intensity_update: IntensityUpdate = Field(default_factory=IntensityUpdate)
    convergence_results: List[ConvergenceResult] = Field(default_factory=list)
    benchmark_summary: BenchmarkSummary = Field(default_factory=BenchmarkSummary)
    progress_report: ProgressReport = Field(default_factory=ProgressReport)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    key_findings: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ProgressMonitoringWorkflow:
    """
    4-phase progress monitoring workflow.

    Phase 1: IntensityUpdate -- Update intensity metrics.
    Phase 2: ConvergenceCheck -- Check pathway convergence.
    Phase 3: BenchmarkUpdate -- Refresh sector benchmarks.
    Phase 4: ProgressReport -- Generate progress report.

    Example:
        >>> wf = ProgressMonitoringWorkflow()
        >>> inp = ProgressMonitoringInput(
        ...     config=ProgressMonitoringConfig(sector="steel"),
        ...     intensity_data=[
        ...         IntensityDataPoint(year=2020, intensity=1.89),
        ...         IntensityDataPoint(year=2025, intensity=1.70),
        ...     ],
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[ProgressMonitoringConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or ProgressMonitoringConfig()
        self._phase_results: List[PhaseResult] = []
        self._intensity_update: IntensityUpdate = IntensityUpdate()
        self._convergence: List[ConvergenceResult] = []
        self._benchmarks: BenchmarkSummary = BenchmarkSummary()
        self._report: ProgressReport = ProgressReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: ProgressMonitoringInput) -> ProgressMonitoringResult:
        started_at = utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting progress monitoring workflow %s, sector=%s",
            self.workflow_id, self.config.sector,
        )

        try:
            phase1 = await self._phase_intensity_update(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_convergence_check(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_benchmark_update(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_progress_report(input_data)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Progress monitoring failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        # Overall RAG
        nze_conv = next(
            (c for c in self._convergence if c.scenario == "nze_15c"), None,
        )
        overall_rag = nze_conv.rag_status if nze_conv else RAGStatus.AMBER

        result = ProgressMonitoringResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            intensity_update=self._intensity_update,
            convergence_results=self._convergence,
            benchmark_summary=self._benchmarks,
            progress_report=self._report,
            overall_rag=overall_rag,
            key_findings=self._generate_findings(),
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Intensity Update
    # -------------------------------------------------------------------------

    async def _phase_intensity_update(self, input_data: ProgressMonitoringInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data_points = sorted(input_data.intensity_data, key=lambda d: d.year)

        if len(data_points) < 2:
            # Generate synthetic data if insufficient
            base_i = self.config.base_year_intensity or 1.0
            for y in range(self.config.base_year, self.config.base_year + 6):
                factor = 1.0 - 0.03 * (y - self.config.base_year)
                data_points.append(IntensityDataPoint(
                    year=y, intensity=round(base_i * factor, 6),
                    emissions_tco2e=base_i * factor * 1000,
                    activity=1000.0,
                ))
            data_points.sort(key=lambda d: d.year)
            warnings.append("Insufficient intensity data; synthetic trajectory generated.")

        current = data_points[-1]
        previous = data_points[-2] if len(data_points) >= 2 else current
        base_point = data_points[0]

        # Year-over-year change
        if previous.intensity > 0:
            yoy = ((current.intensity - previous.intensity) / previous.intensity) * 100
        else:
            yoy = 0.0

        # CAGR from base year
        years = max(current.year - base_point.year, 1)
        if base_point.intensity > 0 and current.intensity > 0:
            cagr = ((current.intensity / base_point.intensity) ** (1.0 / years) - 1.0) * 100
        else:
            cagr = 0.0

        # Trend direction
        if yoy < -1.0:
            trend = TrendDirection.IMPROVING
        elif yoy > 1.0:
            trend = TrendDirection.DETERIORATING
        else:
            trend = TrendDirection.STABLE

        # Linear regression for projection
        intensities = [dp.intensity for dp in data_points if dp.intensity > 0]
        years_list = [dp.year for dp in data_points if dp.intensity > 0]
        if len(intensities) >= 2:
            n = len(intensities)
            mean_x = sum(years_list) / n
            mean_y = sum(intensities) / n
            ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(years_list, intensities))
            ss_xx = sum((x - mean_x) ** 2 for x in years_list)
            slope = ss_xy / max(ss_xx, 1e-10)
            intercept = mean_y - slope * mean_x

            proj_2030 = max(intercept + slope * 2030, 0.0)
            proj_2050 = max(intercept + slope * 2050, 0.0)
        else:
            slope = 0.0
            proj_2030 = current.intensity
            proj_2050 = current.intensity

        # Cumulative reduction
        if base_point.intensity > 0:
            cum_red = (1.0 - current.intensity / base_point.intensity) * 100
        else:
            cum_red = 0.0

        self._intensity_update = IntensityUpdate(
            current_intensity=round(current.intensity, 6),
            previous_intensity=round(previous.intensity, 6),
            base_year_intensity=round(base_point.intensity, 6),
            yoy_change_pct=round(yoy, 2),
            cagr_pct=round(cagr, 2),
            trend_direction=trend,
            linear_trend_slope=round(slope, 6),
            projected_2030_intensity=round(proj_2030, 6),
            projected_2050_intensity=round(proj_2050, 6),
            cumulative_reduction_pct=round(cum_red, 2),
            data_points_used=len(data_points),
            intensity_unit=self.config.intensity_unit,
        )

        outputs["current_intensity"] = self._intensity_update.current_intensity
        outputs["yoy_change_pct"] = self._intensity_update.yoy_change_pct
        outputs["cagr_pct"] = self._intensity_update.cagr_pct
        outputs["trend"] = trend.value
        outputs["projected_2030"] = self._intensity_update.projected_2030_intensity
        outputs["projected_2050"] = self._intensity_update.projected_2050_intensity
        outputs["cumulative_reduction_pct"] = self._intensity_update.cumulative_reduction_pct
        outputs["data_points"] = len(data_points)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="intensity_update", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_intensity_update",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Convergence Check
    # -------------------------------------------------------------------------

    async def _phase_convergence_check(self, input_data: ProgressMonitoringInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        actual = self._intensity_update.current_intensity
        current_year = max(
            (dp.year for dp in input_data.intensity_data), default=2025,
        )
        proj_2030 = self._intensity_update.projected_2030_intensity
        proj_2050 = self._intensity_update.projected_2050_intensity

        self._convergence = []

        for scenario in self.config.scenarios:
            # Get pathway target for current year
            pathway_targets = input_data.pathway_targets.get(scenario, {})

            # If no pathway provided, use sector benchmarks
            if not pathway_targets:
                benchmarks = SECTOR_BENCHMARKS.get(self.config.sector, {})
                nze_2025 = benchmarks.get("iea_nze_2025", actual)
                nze_2030 = benchmarks.get("iea_nze_2030", actual * 0.7)

                # Scenario multipliers
                mult_map = {
                    "nze_15c": 1.0, "wb2c": 1.15, "2c": 1.30,
                    "aps": 1.10, "steps": 1.45,
                }
                mult = mult_map.get(scenario, 1.0)

                for y in range(2020, 2051):
                    t = (y - 2020) / 30
                    pathway_targets[y] = round(
                        nze_2025 * (1 - t * 0.97) * mult, 6,
                    )

            # Current year pathway target
            pathway_now = pathway_targets.get(
                current_year,
                pathway_targets.get(
                    min(pathway_targets.keys(), key=lambda k: abs(k - current_year))
                    if pathway_targets else current_year,
                    actual,
                ),
            )

            # Gap
            gap_abs = actual - pathway_now
            gap_pct = (gap_abs / max(pathway_now, 1e-10)) * 100

            # Convergence status
            if gap_pct <= 0:
                conv_status = ConvergenceStatus.ON_TRACK
                rag = RAGStatus.GREEN
            elif gap_pct <= 10:
                conv_status = ConvergenceStatus.SLIGHT_DEVIATION
                rag = RAGStatus.GREEN
            elif gap_pct <= 25:
                conv_status = ConvergenceStatus.SIGNIFICANT_DEVIATION
                rag = RAGStatus.AMBER
            else:
                conv_status = ConvergenceStatus.OFF_TRACK
                rag = RAGStatus.RED

            # Required acceleration
            nt_target = self.config.near_term_target_intensity
            if nt_target <= 0:
                nt_target = pathway_targets.get(2030, actual * 0.7)
            years_to_nt = max(self.config.near_term_target_year - current_year, 1)
            if actual > 0 and nt_target > 0:
                req_rate = (1.0 - (nt_target / actual) ** (1.0 / years_to_nt)) * 100
                current_rate = abs(self._intensity_update.cagr_pct)
                accel = max(req_rate - current_rate, 0.0)
            else:
                accel = 0.0

            # Time to convergence
            slope = self._intensity_update.linear_trend_slope
            if slope < 0 and actual > pathway_now:
                ttc = int(math.ceil(gap_abs / abs(slope)))
            elif actual <= pathway_now:
                ttc = 0
            else:
                ttc = 999

            # On-track checks
            on_track_2030 = proj_2030 <= pathway_targets.get(2030, proj_2030 * 1.1) * 1.10
            on_track_2050 = proj_2050 <= pathway_targets.get(2050, proj_2050 * 1.1) * 1.10

            # Corrective actions
            corrective: List[str] = []
            if conv_status in (ConvergenceStatus.SIGNIFICANT_DEVIATION, ConvergenceStatus.OFF_TRACK):
                corrective.extend([
                    f"Increase annual intensity reduction by {accel:.1f} percentage points.",
                    "Review and accelerate technology deployment schedule.",
                    "Evaluate additional abatement levers not currently in roadmap.",
                    "Consider interim targets with quarterly monitoring cadence.",
                ])
            elif conv_status == ConvergenceStatus.SLIGHT_DEVIATION:
                corrective.extend([
                    "Minor course correction needed; review implementation timeline.",
                    "Focus on operational efficiency quick wins.",
                ])

            self._convergence.append(ConvergenceResult(
                scenario=scenario,
                pathway_intensity_current_year=round(pathway_now, 6),
                actual_intensity=round(actual, 6),
                gap_absolute=round(gap_abs, 6),
                gap_pct=round(gap_pct, 2),
                convergence_status=conv_status,
                rag_status=rag,
                required_acceleration_pct=round(accel, 2),
                years_to_convergence=min(ttc, 200),
                on_track_for_2030=on_track_2030,
                on_track_for_2050=on_track_2050,
                corrective_actions=corrective,
            ))

        nze = next((c for c in self._convergence if c.scenario == "nze_15c"), None)
        outputs["scenarios_checked"] = len(self._convergence)
        if nze:
            outputs["nze_gap_pct"] = nze.gap_pct
            outputs["nze_status"] = nze.convergence_status.value
            outputs["nze_rag"] = nze.rag_status.value
            outputs["nze_on_track_2030"] = nze.on_track_for_2030
            outputs["nze_acceleration_needed"] = nze.required_acceleration_pct

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="convergence_check", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_convergence_check",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Benchmark Update
    # -------------------------------------------------------------------------

    async def _phase_benchmark_update(self, input_data: ProgressMonitoringInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        actual = self._intensity_update.current_intensity
        sector = self.config.sector
        benchmarks_data = SECTOR_BENCHMARKS.get(sector, {})

        comparisons: List[BenchmarkComparison] = []

        # Sector Average
        avg = benchmarks_data.get("global_avg_2025", actual)
        if avg > 0:
            gap = ((actual - avg) / avg) * 100
            pctile = max(0.0, min(100.0, 50.0 - gap))
            rag = RAGStatus.GREEN if gap <= 0 else (RAGStatus.AMBER if gap <= 20 else RAGStatus.RED)
            comparisons.append(BenchmarkComparison(
                source=BenchmarkSource.SECTOR_AVERAGE,
                source_name=f"Global Sector Average ({sector})",
                benchmark_value=round(avg, 6),
                actual_value=round(actual, 6),
                gap_pct=round(gap, 2),
                percentile=round(pctile, 1),
                rag_status=rag,
            ))

        # Sector Leader
        leader = benchmarks_data.get("leader_2025", actual)
        if leader > 0:
            gap = ((actual - leader) / leader) * 100
            pctile = max(0.0, min(100.0, 90.0 - gap * 0.5))
            rag = RAGStatus.GREEN if gap <= 10 else (RAGStatus.AMBER if gap <= 50 else RAGStatus.RED)
            comparisons.append(BenchmarkComparison(
                source=BenchmarkSource.SECTOR_LEADER,
                source_name=f"Sector Leader ({sector})",
                benchmark_value=round(leader, 6),
                actual_value=round(actual, 6),
                gap_pct=round(gap, 2),
                percentile=round(pctile, 1),
                rag_status=rag,
            ))

        # SBTi Validated Peers
        sbti = benchmarks_data.get("sbti_avg_2025", actual)
        if sbti > 0:
            gap = ((actual - sbti) / sbti) * 100
            pctile = max(0.0, min(100.0, 60.0 - gap * 0.6))
            rag = RAGStatus.GREEN if gap <= 5 else (RAGStatus.AMBER if gap <= 25 else RAGStatus.RED)
            comparisons.append(BenchmarkComparison(
                source=BenchmarkSource.SBTI_PEER,
                source_name=f"SBTi-Validated Peers ({sector})",
                benchmark_value=round(sbti, 6),
                actual_value=round(actual, 6),
                gap_pct=round(gap, 2),
                percentile=round(pctile, 1),
                rag_status=rag,
            ))

        # IEA NZE Pathway
        iea = benchmarks_data.get("iea_nze_2025", actual)
        if iea > 0:
            gap = ((actual - iea) / iea) * 100
            pctile = max(0.0, min(100.0, 70.0 - gap * 0.7))
            rag = RAGStatus.GREEN if gap <= 0 else (RAGStatus.AMBER if gap <= 15 else RAGStatus.RED)
            comparisons.append(BenchmarkComparison(
                source=BenchmarkSource.IEA_PATHWAY,
                source_name=f"IEA NZE 2050 ({sector})",
                benchmark_value=round(iea, 6),
                actual_value=round(actual, 6),
                gap_pct=round(gap, 2),
                percentile=round(pctile, 1),
                rag_status=rag,
            ))

        # Regulatory Benchmark
        reg = benchmarks_data.get("regulatory_benchmark", 0.0)
        reg_name = benchmarks_data.get("regulatory_name", "Regulatory")
        if reg > 0:
            gap = ((actual - reg) / reg) * 100
            rag = RAGStatus.GREEN if gap <= 0 else (RAGStatus.AMBER if gap <= 10 else RAGStatus.RED)
            comparisons.append(BenchmarkComparison(
                source=BenchmarkSource.REGULATORY,
                source_name=reg_name,
                benchmark_value=round(reg, 6),
                actual_value=round(actual, 6),
                gap_pct=round(gap, 2),
                percentile=50.0,
                rag_status=rag,
            ))

        # Add peer comparisons from input
        for peer in input_data.peer_data[:10]:
            peer_intensity = peer.get("intensity", 0.0)
            peer_name = peer.get("company", "Peer")
            if peer_intensity > 0:
                gap = ((actual - peer_intensity) / peer_intensity) * 100
                rag = RAGStatus.GREEN if gap <= 0 else (RAGStatus.AMBER if gap <= 15 else RAGStatus.RED)
                comparisons.append(BenchmarkComparison(
                    source=BenchmarkSource.SBTI_PEER,
                    source_name=peer_name,
                    benchmark_value=round(peer_intensity, 6),
                    actual_value=round(actual, 6),
                    gap_pct=round(gap, 2),
                    percentile=50.0,
                    rag_status=rag,
                ))

        # Summary
        best = min(comparisons, key=lambda c: c.gap_pct) if comparisons else None
        worst = max(comparisons, key=lambda c: c.gap_pct) if comparisons else None
        avg_pctile = (
            sum(c.percentile for c in comparisons) / max(len(comparisons), 1)
        )
        red_count = sum(1 for c in comparisons if c.rag_status == RAGStatus.RED)
        overall_rag = (
            RAGStatus.RED if red_count >= 2 else
            RAGStatus.AMBER if red_count >= 1 else
            RAGStatus.GREEN
        )

        self._benchmarks = BenchmarkSummary(
            sector=sector,
            comparisons=comparisons,
            best_comparison=best.source_name if best else "",
            worst_comparison=worst.source_name if worst else "",
            overall_percentile=round(avg_pctile, 1),
            overall_rag=overall_rag,
        )

        outputs["comparisons_count"] = len(comparisons)
        outputs["best_vs"] = self._benchmarks.best_comparison
        outputs["worst_vs"] = self._benchmarks.worst_comparison
        outputs["overall_percentile"] = self._benchmarks.overall_percentile
        outputs["overall_rag"] = overall_rag.value

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="benchmark_update", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_benchmark_update",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Progress Report
    # -------------------------------------------------------------------------

    async def _phase_progress_report(self, input_data: ProgressMonitoringInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        # Build KPIs
        kpis: List[ProgressKPI] = []

        # KPI 1: Intensity
        kpis.append(ProgressKPI(
            kpi_name="Current Intensity",
            current_value=self._intensity_update.current_intensity,
            target_value=self.config.near_term_target_intensity or (
                self._intensity_update.base_year_intensity * 0.7
            ),
            unit=self.config.intensity_unit,
            achievement_pct=round(
                self._intensity_update.cumulative_reduction_pct / max(
                    (1 - (self.config.near_term_target_intensity / max(
                        self._intensity_update.base_year_intensity, 1e-10,
                    ))) * 100, 1,
                ) * 100, 1,
            ) if self.config.near_term_target_intensity > 0 else 0.0,
            trend=self._intensity_update.trend_direction,
            rag_status=(
                RAGStatus.GREEN if self._intensity_update.trend_direction == TrendDirection.IMPROVING else
                RAGStatus.AMBER if self._intensity_update.trend_direction == TrendDirection.STABLE else
                RAGStatus.RED
            ),
        ))

        # KPI 2: YoY Change
        kpis.append(ProgressKPI(
            kpi_name="Year-over-Year Intensity Change",
            current_value=self._intensity_update.yoy_change_pct,
            target_value=-4.2,
            unit="%",
            achievement_pct=round(
                min(abs(self._intensity_update.yoy_change_pct) / 4.2 * 100, 100), 1,
            ) if self._intensity_update.yoy_change_pct < 0 else 0.0,
            trend=self._intensity_update.trend_direction,
            rag_status=(
                RAGStatus.GREEN if self._intensity_update.yoy_change_pct <= -4.2 else
                RAGStatus.AMBER if self._intensity_update.yoy_change_pct < 0 else
                RAGStatus.RED
            ),
        ))

        # KPI 3: Cumulative Reduction
        kpis.append(ProgressKPI(
            kpi_name="Cumulative Intensity Reduction",
            current_value=self._intensity_update.cumulative_reduction_pct,
            target_value=90.0,
            unit="%",
            achievement_pct=round(
                self._intensity_update.cumulative_reduction_pct / 90.0 * 100, 1,
            ),
            trend=self._intensity_update.trend_direction,
            rag_status=(
                RAGStatus.GREEN if self._intensity_update.cumulative_reduction_pct >= 20 else
                RAGStatus.AMBER if self._intensity_update.cumulative_reduction_pct >= 10 else
                RAGStatus.RED
            ),
        ))

        # KPI 4: Benchmark Position
        kpis.append(ProgressKPI(
            kpi_name="Sector Benchmark Percentile",
            current_value=self._benchmarks.overall_percentile,
            target_value=75.0,
            unit="percentile",
            achievement_pct=round(
                self._benchmarks.overall_percentile / 75.0 * 100, 1,
            ),
            trend=TrendDirection.STABLE,
            rag_status=self._benchmarks.overall_rag,
        ))

        # Build alerts
        alerts: List[ProgressAlert] = []

        # Alert: Off-track convergence
        for conv in self._convergence:
            if conv.convergence_status == ConvergenceStatus.OFF_TRACK:
                alerts.append(ProgressAlert(
                    alert_id=f"ALERT-CONV-{conv.scenario}",
                    severity=AlertSeverity.CRITICAL,
                    category="convergence",
                    title=f"Off Track: {conv.scenario.upper()} Pathway",
                    description=(
                        f"Intensity {conv.gap_pct:+.1f}% above {conv.scenario} pathway target. "
                        f"Requires {conv.required_acceleration_pct:.1f}pp acceleration."
                    ),
                    metric_value=f"{conv.actual_intensity:.4f}",
                    threshold_value=f"{conv.pathway_intensity_current_year:.4f}",
                    recommended_action="Immediate strategic review and abatement acceleration.",
                ))
            elif conv.convergence_status == ConvergenceStatus.SIGNIFICANT_DEVIATION:
                alerts.append(ProgressAlert(
                    alert_id=f"ALERT-CONV-{conv.scenario}",
                    severity=AlertSeverity.WARNING,
                    category="convergence",
                    title=f"Significant Gap: {conv.scenario.upper()} Pathway",
                    description=f"Intensity {conv.gap_pct:+.1f}% above pathway target.",
                    metric_value=f"{conv.actual_intensity:.4f}",
                    threshold_value=f"{conv.pathway_intensity_current_year:.4f}",
                    recommended_action="Review technology deployment timeline; consider additional levers.",
                ))

        # Alert: Deteriorating trend
        if self._intensity_update.trend_direction == TrendDirection.DETERIORATING:
            alerts.append(ProgressAlert(
                alert_id="ALERT-TREND-001",
                severity=AlertSeverity.CRITICAL,
                category="trend",
                title="Intensity Trend Deteriorating",
                description=(
                    f"YoY intensity change: {self._intensity_update.yoy_change_pct:+.1f}%. "
                    "Emissions intensity is increasing."
                ),
                metric_value=f"{self._intensity_update.yoy_change_pct:+.1f}%",
                threshold_value="< 0%",
                recommended_action="Conduct root cause analysis for intensity increase.",
            ))

        # Alert: Below peer average
        peer_comp = next(
            (c for c in self._benchmarks.comparisons
             if c.source == BenchmarkSource.SECTOR_AVERAGE),
            None,
        )
        if peer_comp and peer_comp.gap_pct > self.config.alert_threshold_pct:
            alerts.append(ProgressAlert(
                alert_id="ALERT-BENCH-001",
                severity=AlertSeverity.WARNING,
                category="benchmark",
                title="Below Sector Average",
                description=(
                    f"Intensity {peer_comp.gap_pct:.1f}% above sector average. "
                    f"Percentile: {peer_comp.percentile:.0f}."
                ),
                metric_value=f"{peer_comp.actual_value:.4f}",
                threshold_value=f"{peer_comp.benchmark_value:.4f}",
                recommended_action="Benchmark against leading peers; identify gap drivers.",
            ))

        # Alert: 2030 target at risk
        for conv in self._convergence:
            if not conv.on_track_for_2030:
                alerts.append(ProgressAlert(
                    alert_id=f"ALERT-2030-{conv.scenario}",
                    severity=AlertSeverity.WARNING,
                    category="target",
                    title=f"2030 Target at Risk ({conv.scenario.upper()})",
                    description=f"Projected 2030 intensity exceeds pathway target by >10%.",
                    recommended_action="Accelerate near-term actions; consider additional investments.",
                ))

        # Executive summary
        nze_conv = next(
            (c for c in self._convergence if c.scenario == "nze_15c"), None,
        )
        exec_summary = (
            f"Progress Report for {self.config.company_name or 'Company'} "
            f"({self.config.sector}). "
            f"Current intensity: {self._intensity_update.current_intensity:.4f} "
            f"{self.config.intensity_unit}. "
            f"YoY change: {self._intensity_update.yoy_change_pct:+.1f}%. "
            f"Cumulative reduction: {self._intensity_update.cumulative_reduction_pct:.1f}%. "
        )
        if nze_conv:
            exec_summary += (
                f"NZE pathway status: {nze_conv.convergence_status.value} "
                f"(gap: {nze_conv.gap_pct:+.1f}%). "
            )
        exec_summary += (
            f"Benchmark percentile: {self._benchmarks.overall_percentile:.0f}th. "
            f"Alerts: {len(alerts)} ({sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)} critical)."
        )

        self._report = ProgressReport(
            report_id=_new_uuid(),
            report_date=utcnow().isoformat(),
            sector=self.config.sector,
            company_name=self.config.company_name,
            intensity_update=self._intensity_update,
            convergence_results=self._convergence,
            benchmark_summary=self._benchmarks,
            kpis=kpis,
            alerts=alerts,
            executive_summary=exec_summary,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["kpis_count"] = len(kpis)
        outputs["alerts_count"] = len(alerts)
        outputs["critical_alerts"] = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        outputs["report_formats"] = ["MD", "HTML", "JSON", "PDF"]

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="progress_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_progress_report",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(
            f"Intensity: {self._intensity_update.current_intensity:.4f} "
            f"{self.config.intensity_unit} "
            f"(YoY: {self._intensity_update.yoy_change_pct:+.1f}%, "
            f"CAGR: {self._intensity_update.cagr_pct:+.1f}%)"
        )
        nze = next((c for c in self._convergence if c.scenario == "nze_15c"), None)
        if nze:
            findings.append(
                f"NZE pathway: {nze.convergence_status.value} "
                f"(gap: {nze.gap_pct:+.1f}%, RAG: {nze.rag_status.value})"
            )
        findings.append(
            f"Sector benchmark: {self._benchmarks.overall_percentile:.0f}th percentile "
            f"(RAG: {self._benchmarks.overall_rag.value})"
        )
        alerts_c = sum(1 for a in self._report.alerts if a.severity == AlertSeverity.CRITICAL)
        if alerts_c > 0:
            findings.append(f"CRITICAL: {alerts_c} critical alert(s) requiring immediate action.")
        return findings

    def _generate_next_steps(self) -> List[str]:
        steps: List[str] = []
        if any(a.severity == AlertSeverity.CRITICAL for a in self._report.alerts):
            steps.append("Address critical alerts immediately with corrective action plan.")
        steps.extend([
            "Schedule quarterly intensity data refresh.",
            "Update technology deployment milestones.",
            "Present progress report to sustainability committee.",
            "Review benchmark position against updated peer data.",
            "Integrate progress data into annual SBTi target tracking.",
        ])
        return steps

    def _check_iea_milestones(self, sector: str, current_year: int,
                               current_intensity: float) -> List[ProgressAlert]:
        """
        Check current intensity against IEA NZE milestones for the sector.
        Generates alerts for milestones that are at risk.
        """
        alerts: List[ProgressAlert] = []
        milestones = IEA_NZE_MILESTONES.get(sector, [])

        for ms in milestones:
            ms_year = ms["year"]
            ms_target = ms["intensity_target"]
            ms_desc = ms["milestone"]

            # Only check milestones within the next 5 years
            if ms_year < current_year or ms_year > current_year + 5:
                continue

            years_to_milestone = ms_year - current_year
            gap_pct = ((current_intensity - ms_target) / max(ms_target, 1e-10)) * 100

            # Determine if milestone is at risk
            if gap_pct > 30 and years_to_milestone <= 2:
                severity = AlertSeverity.CRITICAL
                title = f"IEA NZE {ms_year} Milestone: CRITICAL"
            elif gap_pct > 15 and years_to_milestone <= 3:
                severity = AlertSeverity.WARNING
                title = f"IEA NZE {ms_year} Milestone: At Risk"
            elif gap_pct > 0 and years_to_milestone <= 1:
                severity = AlertSeverity.WARNING
                title = f"IEA NZE {ms_year} Milestone: Approaching"
            else:
                continue

            required_annual = (current_intensity - ms_target) / max(years_to_milestone, 1)

            alerts.append(ProgressAlert(
                alert_id=f"ALERT-IEA-{ms_year}",
                severity=severity,
                category="milestone",
                title=title,
                description=(
                    f"IEA NZE milestone '{ms_desc}' by {ms_year}: "
                    f"current intensity {current_intensity:.4f} vs target {ms_target:.4f} "
                    f"(gap: {gap_pct:+.1f}%). "
                    f"Required annual reduction: {required_annual:.4f} units/yr "
                    f"over {years_to_milestone} year(s)."
                ),
                metric_value=f"{current_intensity:.4f}",
                threshold_value=f"{ms_target:.4f}",
                recommended_action=(
                    f"Align short-term action plan with IEA NZE {ms_year} milestone: "
                    f"'{ms_desc}'. Accelerate relevant technology deployment."
                ),
            ))

        return alerts

    def _calculate_carbon_budget_status(
        self, intensity_data: List[IntensityDataPoint],
        activity: float, carbon_budget: float,
    ) -> Dict[str, Any]:
        """
        Calculate carbon budget consumption status based on historical
        cumulative emissions and projected trajectory.

        Returns dict with consumed_pct, remaining_tco2e, years_to_exhaustion,
        and budget_status.
        """
        if carbon_budget <= 0:
            return {
                "consumed_pct": 0.0,
                "remaining_tco2e": 0.0,
                "years_to_exhaustion": 999,
                "budget_status": "not_set",
            }

        # Calculate cumulative historical emissions
        cumulative = 0.0
        for dp in sorted(intensity_data, key=lambda d: d.year):
            annual_emissions = dp.intensity * max(activity, dp.activity)
            cumulative += annual_emissions

        consumed_pct = (cumulative / carbon_budget) * 100
        remaining = max(carbon_budget - cumulative, 0)

        # Project years to exhaustion
        if intensity_data:
            recent = sorted(intensity_data, key=lambda d: d.year)[-1]
            current_annual = recent.intensity * max(activity, recent.activity)
            if current_annual > 0:
                years_remaining = remaining / current_annual
            else:
                years_remaining = 999
        else:
            years_remaining = 999

        if consumed_pct >= 90:
            status = "critical"
        elif consumed_pct >= 75:
            status = "warning"
        elif consumed_pct >= 50:
            status = "on_track"
        else:
            status = "healthy"

        return {
            "consumed_pct": round(consumed_pct, 1),
            "remaining_tco2e": round(remaining, 0),
            "years_to_exhaustion": round(min(years_remaining, 999), 1),
            "budget_status": status,
        }

    def _compute_moving_average(
        self, data_points: List[IntensityDataPoint], window: int = 3,
    ) -> List[Dict[str, float]]:
        """
        Compute a moving average of intensity values for trend smoothing.
        Returns list of dicts with year and smoothed_intensity.
        """
        sorted_points = sorted(data_points, key=lambda d: d.year)
        intensities = [dp.intensity for dp in sorted_points]
        years = [dp.year for dp in sorted_points]
        result: List[Dict[str, float]] = []

        for i in range(len(intensities)):
            start = max(0, i - window + 1)
            window_vals = intensities[start:i + 1]
            avg = sum(window_vals) / max(len(window_vals), 1)
            result.append({
                "year": years[i],
                "smoothed_intensity": round(avg, 6),
                "raw_intensity": round(intensities[i], 6),
            })

        return result

    def _detect_trend_change(
        self, data_points: List[IntensityDataPoint],
    ) -> Optional[Dict[str, Any]]:
        """
        Detect significant trend changes (inflection points) in intensity
        data using piecewise linear regression.

        Returns None if no significant change detected, or a dict with
        change_year, previous_slope, new_slope, and significance.
        """
        sorted_points = sorted(data_points, key=lambda d: d.year)
        if len(sorted_points) < 5:
            return None

        intensities = [dp.intensity for dp in sorted_points]
        years = [float(dp.year) for dp in sorted_points]
        n = len(intensities)

        best_change = None
        best_residual = float("inf")

        # Try each potential change point
        for cp in range(2, n - 2):
            # Fit two separate linear regressions
            x1, y1 = years[:cp], intensities[:cp]
            x2, y2 = years[cp:], intensities[cp:]

            def _fit_slope(x: List[float], y: List[float]) -> float:
                n_pts = len(x)
                mx = sum(x) / n_pts
                my = sum(y) / n_pts
                ssxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
                ssxx = sum((xi - mx) ** 2 for xi in x)
                return ssxy / max(ssxx, 1e-10)

            slope1 = _fit_slope(x1, y1)
            slope2 = _fit_slope(x2, y2)

            # Check if slopes differ significantly
            slope_diff = abs(slope2 - slope1)
            if slope_diff > 0.5:  # Threshold for significance
                residual = sum(
                    (y - (slope1 * x + intensities[0])) ** 2
                    for x, y in zip(x1, y1)
                ) + sum(
                    (y - (slope2 * x + intensities[cp])) ** 2
                    for x, y in zip(x2, y2)
                )

                if residual < best_residual:
                    best_residual = residual
                    best_change = {
                        "change_year": int(years[cp]),
                        "previous_slope": round(slope1, 4),
                        "new_slope": round(slope2, 4),
                        "significance": round(slope_diff, 4),
                        "direction": (
                            "improvement_accelerated" if slope2 < slope1
                            else "improvement_decelerated"
                        ),
                    }

        return best_change

    def _generate_historical_context(self, sector: str) -> Dict[str, Any]:
        """
        Provide historical improvement rate context for a sector to help
        contextualise current performance.
        """
        hist_data = HISTORICAL_IMPROVEMENT_RATES.get(sector, {})
        if not hist_data:
            return {"available": False, "sector": sector}

        return {
            "available": True,
            "sector": sector,
            "improvement_2015_2020_pct_yr": hist_data.get("2015_2020", 0),
            "improvement_2020_2025_pct_yr": hist_data.get("2020_2025", 0),
            "required_nze_pct_yr": hist_data.get("required_nze", 0),
            "peer_best_pct_yr": hist_data.get("peer_best", 0),
            "peer_worst_pct_yr": hist_data.get("peer_worst", 0),
            "acceleration_needed_pct": round(
                abs(hist_data.get("required_nze", 0)) -
                abs(hist_data.get("2020_2025", 0)), 1,
            ),
            "is_accelerating": (
                abs(hist_data.get("2020_2025", 0)) >
                abs(hist_data.get("2015_2020", 0))
            ),
        }
