# -*- coding: utf-8 -*-
"""
Advanced Progress Tracking Workflow
=========================================

5-phase workflow for advanced year-over-year emission progress analysis
within PACK-022 Net-Zero Acceleration Pack.  The workflow ingests annual
emissions data, decomposes changes using LMDI-I method, attributes
drivers to business units, forecasts future trajectories, and generates
alerts on deviation from target pathways.

Phases:
    1. DataIngestion    -- Ingest new year's emissions data, validate structure
    2. Decomposition    -- Decompose YoY change using LMDI-I (activity/intensity/structural)
    3. Attribution      -- Attribute changes to specific drivers (BUs, facilities)
    4. Forecasting      -- Generate rolling 1-3 year forecast based on trends
    5. AlertGeneration  -- Compare forecast to target pathway, generate alerts

Regulatory references:
    - GHG Protocol Corporate Standard
    - SBTi Monitoring, Reporting and Verification guidance
    - Ang (2004) - Decomposition analysis for policymaking in energy (LMDI-I)
    - ISO 14064-1 - GHG quantification at organisation level

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class TrendDirection(str, Enum):
    """Emission trend direction."""

    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"

# =============================================================================
# ALERT THRESHOLDS (Zero-Hallucination)
# =============================================================================

DEFAULT_ALERT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "info": {"deviation_pct": 5.0, "description": "Slight deviation from target pathway"},
    "warning": {"deviation_pct": 10.0, "description": "Significant deviation requiring attention"},
    "critical": {"deviation_pct": 20.0, "description": "Major deviation requiring urgent action"},
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AnnualEmissionRecord(BaseModel):
    """Emissions data for a single year."""

    year: int = Field(default=2024, ge=2000, le=2060)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_usd: float = Field(default=0.0, ge=0.0, description="For intensity metric")
    activity_level: float = Field(default=0.0, ge=0.0, description="Activity metric value")
    headcount: int = Field(default=0, ge=0)

class BusinessUnitEmission(BaseModel):
    """Emissions data for a single business unit in a year."""

    bu_id: str = Field(default="")
    bu_name: str = Field(default="")
    year: int = Field(default=2024)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    activity_level: float = Field(default=0.0, ge=0.0)
    share_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class TargetPathwayPoint(BaseModel):
    """A point on the target emission pathway."""

    year: int = Field(default=2025)
    target_tco2e: float = Field(default=0.0, ge=0.0)

class LMDIDecomposition(BaseModel):
    """LMDI-I decomposition result for a year-over-year change."""

    period: str = Field(default="", description="e.g. '2023-2024'")
    total_change_tco2e: float = Field(default=0.0)
    total_change_pct: float = Field(default=0.0)
    activity_effect_tco2e: float = Field(default=0.0, description="Change due to activity level")
    intensity_effect_tco2e: float = Field(default=0.0, description="Change due to emission intensity")
    structural_effect_tco2e: float = Field(default=0.0, description="Change due to structural shift")
    activity_effect_pct: float = Field(default=0.0)
    intensity_effect_pct: float = Field(default=0.0)
    structural_effect_pct: float = Field(default=0.0)
    residual_tco2e: float = Field(default=0.0, description="Unexplained residual")

class DriverAttribution(BaseModel):
    """Attribution of emission change to a specific driver."""

    driver_id: str = Field(default="")
    driver_name: str = Field(default="")
    driver_type: str = Field(default="", description="business_unit, facility, product")
    change_tco2e: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)
    contribution_to_total_change_pct: float = Field(default=0.0)
    direction: TrendDirection = Field(default=TrendDirection.STABLE)
    explanation: str = Field(default="")

class ForecastPoint(BaseModel):
    """A single forecast point."""

    year: int = Field(default=2025)
    forecast_tco2e: float = Field(default=0.0, ge=0.0)
    lower_bound_tco2e: float = Field(default=0.0, ge=0.0)
    upper_bound_tco2e: float = Field(default=0.0, ge=0.0)
    target_tco2e: float = Field(default=0.0, ge=0.0)
    deviation_pct: float = Field(default=0.0)
    confidence_level: float = Field(default=0.80)

class ProgressAlert(BaseModel):
    """Alert generated from progress tracking."""

    alert_id: str = Field(default="")
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    year: int = Field(default=2025)
    message: str = Field(default="")
    deviation_pct: float = Field(default=0.0)
    forecast_tco2e: float = Field(default=0.0)
    target_tco2e: float = Field(default=0.0)
    recommended_action: str = Field(default="")

class AdvancedProgressConfig(BaseModel):
    """Configuration for the advanced progress workflow."""

    annual_data: List[AnnualEmissionRecord] = Field(default_factory=list)
    business_units: List[BusinessUnitEmission] = Field(default_factory=list)
    target_pathway: List[TargetPathwayPoint] = Field(default_factory=list)
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"info": 5.0, "warning": 10.0, "critical": 20.0}
    )
    forecast_years: int = Field(default=3, ge=1, le=10)
    base_year: int = Field(default=2020, ge=2000, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class AdvancedProgressResult(BaseModel):
    """Complete result from the advanced progress workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="advanced_progress")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    validated_records: List[AnnualEmissionRecord] = Field(default_factory=list)
    decompositions: List[LMDIDecomposition] = Field(default_factory=list)
    attributions: List[DriverAttribution] = Field(default_factory=list)
    forecasts: List[ForecastPoint] = Field(default_factory=list)
    alerts: List[ProgressAlert] = Field(default_factory=list)
    overall_trend: TrendDirection = Field(default=TrendDirection.STABLE)
    cumulative_reduction_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class AdvancedProgressWorkflow:
    """
    5-phase advanced progress tracking workflow.

    Ingests annual emissions data, decomposes changes using the LMDI-I
    method, attributes drivers, forecasts trajectories, and generates
    alerts on deviation from target pathways.

    Zero-hallucination: all LMDI-I decomposition, linear regression
    forecasting, and deviation calculations use deterministic formulas.
    No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = AdvancedProgressWorkflow()
        >>> config = AdvancedProgressConfig(annual_data=[...], target_pathway=[...])
        >>> result = await wf.execute(config)
        >>> assert len(result.alerts) >= 0
    """

    def __init__(self) -> None:
        """Initialise AdvancedProgressWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._records: List[AnnualEmissionRecord] = []
        self._decompositions: List[LMDIDecomposition] = []
        self._attributions: List[DriverAttribution] = []
        self._forecasts: List[ForecastPoint] = []
        self._alerts: List[ProgressAlert] = []
        self._trend: TrendDirection = TrendDirection.STABLE
        self._cumulative_reduction: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: AdvancedProgressConfig) -> AdvancedProgressResult:
        """
        Execute the 5-phase advanced progress workflow.

        Args:
            config: Progress configuration with annual data, business unit
                breakdowns, target pathway, and alert thresholds.

        Returns:
            AdvancedProgressResult with decompositions, attributions,
            forecasts, and alerts.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting advanced progress workflow %s, years=%d",
            self.workflow_id, len(config.annual_data),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_ingestion(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Data ingestion failed; cannot proceed")

            phase2 = await self._phase_decomposition(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_attribution(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_forecasting(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_alert_generation(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Advanced progress workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = AdvancedProgressResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            validated_records=self._records,
            decompositions=self._decompositions,
            attributions=self._attributions,
            forecasts=self._forecasts,
            alerts=self._alerts,
            overall_trend=self._trend,
            cumulative_reduction_pct=round(self._cumulative_reduction, 2),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Advanced progress workflow %s completed in %.2fs, alerts=%d",
            self.workflow_id, elapsed, len(self._alerts),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion
    # -------------------------------------------------------------------------

    async def _phase_data_ingestion(self, config: AdvancedProgressConfig) -> PhaseResult:
        """Ingest and validate annual emissions data."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        records = config.annual_data
        if not records:
            records = self._generate_sample_data(config)
            warnings.append(f"No annual data provided; generated {len(records)} sample records")

        # Sort by year
        records = sorted(records, key=lambda r: r.year)

        # Validate totals
        for rec in records:
            computed_total = rec.scope1_tco2e + rec.scope2_tco2e + rec.scope3_tco2e
            if rec.total_tco2e > 0 and abs(rec.total_tco2e - computed_total) > 0.01:
                warnings.append(
                    f"Year {rec.year}: total_tco2e ({rec.total_tco2e:.2f}) differs from "
                    f"sum of scopes ({computed_total:.2f}); using sum"
                )
                rec.total_tco2e = computed_total
            elif rec.total_tco2e <= 0:
                rec.total_tco2e = computed_total

        # Need at least 2 years for decomposition
        if len(records) < 2:
            errors.append("At least 2 years of data required for decomposition analysis")

        self._records = records

        # Calculate cumulative reduction from base year
        if len(records) >= 2:
            base_emissions = records[0].total_tco2e
            latest_emissions = records[-1].total_tco2e
            if base_emissions > 0:
                self._cumulative_reduction = (
                    (base_emissions - latest_emissions) / base_emissions * 100.0
                )

        outputs["records_count"] = len(records)
        outputs["year_range"] = f"{records[0].year}-{records[-1].year}" if records else "none"
        outputs["base_year_tco2e"] = round(records[0].total_tco2e, 2) if records else 0.0
        outputs["latest_year_tco2e"] = round(records[-1].total_tco2e, 2) if records else 0.0
        outputs["cumulative_reduction_pct"] = round(self._cumulative_reduction, 2)

        elapsed = (utcnow() - started).total_seconds()
        status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        self.logger.info("Data ingestion: %d records, cumulative reduction=%.1f%%",
                         len(records), self._cumulative_reduction)
        return PhaseResult(
            phase_name="data_ingestion",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_sample_data(self, config: AdvancedProgressConfig) -> List[AnnualEmissionRecord]:
        """Generate sample annual emission records."""
        base_year = config.base_year
        base_total = 100000.0
        records: List[AnnualEmissionRecord] = []
        for i in range(5):
            year = base_year + i
            # Simulate 3-5% annual reduction with some noise
            factor = (1.0 - 0.04) ** i
            total = base_total * factor
            records.append(AnnualEmissionRecord(
                year=year,
                total_tco2e=round(total, 2),
                scope1_tco2e=round(total * 0.30, 2),
                scope2_tco2e=round(total * 0.20, 2),
                scope3_tco2e=round(total * 0.50, 2),
                revenue_usd=round(50_000_000 * (1.03 ** i), 2),
                activity_level=round(10000 * (1.02 ** i), 2),
                headcount=1000 + i * 20,
            ))
        return records

    # -------------------------------------------------------------------------
    # Phase 2: LMDI-I Decomposition
    # -------------------------------------------------------------------------

    async def _phase_decomposition(self, config: AdvancedProgressConfig) -> PhaseResult:
        """Decompose YoY changes using LMDI-I method."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._decompositions = []
        records = self._records

        for i in range(1, len(records)):
            prev = records[i - 1]
            curr = records[i]
            decomp = self._lmdi_decompose(prev, curr)
            self._decompositions.append(decomp)

        # Determine overall trend
        if len(self._decompositions) >= 2:
            recent_changes = [d.total_change_pct for d in self._decompositions[-3:]]
            avg_change = sum(recent_changes) / len(recent_changes)
            if avg_change < -1.0:
                self._trend = TrendDirection.DECREASING
            elif avg_change > 1.0:
                self._trend = TrendDirection.INCREASING
            else:
                self._trend = TrendDirection.STABLE

        outputs["decomposition_count"] = len(self._decompositions)
        outputs["overall_trend"] = self._trend.value
        if self._decompositions:
            latest = self._decompositions[-1]
            outputs["latest_total_change_pct"] = round(latest.total_change_pct, 2)
            outputs["latest_activity_effect_pct"] = round(latest.activity_effect_pct, 2)
            outputs["latest_intensity_effect_pct"] = round(latest.intensity_effect_pct, 2)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Decomposition: %d periods, trend=%s",
                         len(self._decompositions), self._trend.value)
        return PhaseResult(
            phase_name="decomposition",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _lmdi_decompose(
        self, prev: AnnualEmissionRecord, curr: AnnualEmissionRecord
    ) -> LMDIDecomposition:
        """
        Perform LMDI-I (Log Mean Divisia Index) decomposition.

        Decomposes emission change into:
        - Activity effect: change due to overall activity level
        - Intensity effect: change due to emission intensity
        - Structural effect: change due to scope mix shifts

        Based on Ang (2004) LMDI-I method.
        """
        e0 = prev.total_tco2e
        e1 = curr.total_tco2e
        total_change = e1 - e0
        total_change_pct = ((e1 - e0) / e0 * 100.0) if e0 > 0 else 0.0

        # Activity effect (using activity_level or revenue as proxy)
        a0 = prev.activity_level if prev.activity_level > 0 else prev.revenue_usd
        a1 = curr.activity_level if curr.activity_level > 0 else curr.revenue_usd

        if a0 > 0 and a1 > 0 and e0 > 0 and e1 > 0:
            # LMDI weight function: L(e1, e0) = (e1 - e0) / (ln(e1) - ln(e0))
            log_weight = self._lmdi_weight(e1, e0)

            # Activity effect
            activity_effect = log_weight * math.log(a1 / a0) if a0 > 0 else 0.0

            # Intensity (emissions per activity)
            i0 = e0 / a0
            i1 = e1 / a1
            intensity_effect = log_weight * math.log(i1 / i0) if i0 > 0 and i1 > 0 else 0.0

            # Structural effect: changes in scope composition
            # Using scope shares as structural components
            structural_effect = total_change - activity_effect - intensity_effect
        else:
            # Fallback: simple decomposition
            activity_effect = 0.0
            intensity_effect = total_change
            structural_effect = 0.0

        # Calculate percentage effects
        activity_pct = (activity_effect / e0 * 100.0) if e0 > 0 else 0.0
        intensity_pct = (intensity_effect / e0 * 100.0) if e0 > 0 else 0.0
        structural_pct = (structural_effect / e0 * 100.0) if e0 > 0 else 0.0

        residual = total_change - activity_effect - intensity_effect - structural_effect

        return LMDIDecomposition(
            period=f"{prev.year}-{curr.year}",
            total_change_tco2e=round(total_change, 4),
            total_change_pct=round(total_change_pct, 2),
            activity_effect_tco2e=round(activity_effect, 4),
            intensity_effect_tco2e=round(intensity_effect, 4),
            structural_effect_tco2e=round(structural_effect, 4),
            activity_effect_pct=round(activity_pct, 2),
            intensity_effect_pct=round(intensity_pct, 2),
            structural_effect_pct=round(structural_pct, 2),
            residual_tco2e=round(residual, 4),
        )

    def _lmdi_weight(self, v1: float, v0: float) -> float:
        """Calculate LMDI logarithmic mean weight: L(v1, v0)."""
        if v1 == v0:
            return v1
        if v1 <= 0 or v0 <= 0:
            return 0.0
        ln_ratio = math.log(v1) - math.log(v0)
        if abs(ln_ratio) < 1e-12:
            return v1
        return (v1 - v0) / ln_ratio

    # -------------------------------------------------------------------------
    # Phase 3: Attribution
    # -------------------------------------------------------------------------

    async def _phase_attribution(self, config: AdvancedProgressConfig) -> PhaseResult:
        """Attribute emission changes to specific drivers."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bu_data = config.business_units
        if not bu_data and len(self._records) >= 2:
            bu_data = self._generate_sample_bus(self._records)
            warnings.append("No business unit data provided; generated sample attributions")

        self._attributions = []
        if len(self._records) >= 2 and bu_data:
            prev_year = self._records[-2].year
            curr_year = self._records[-1].year
            total_change = self._records[-1].total_tco2e - self._records[-2].total_tco2e

            # Group BU data by year
            prev_bus = {b.bu_id: b for b in bu_data if b.year == prev_year}
            curr_bus = {b.bu_id: b for b in bu_data if b.year == curr_year}

            all_bus = set(prev_bus.keys()) | set(curr_bus.keys())

            for bu_id in sorted(all_bus):
                prev_bu = prev_bus.get(bu_id)
                curr_bu = curr_bus.get(bu_id)
                prev_em = prev_bu.emissions_tco2e if prev_bu else 0.0
                curr_em = curr_bu.emissions_tco2e if curr_bu else 0.0
                bu_change = curr_em - prev_em
                bu_name = (curr_bu.bu_name if curr_bu else prev_bu.bu_name) if (curr_bu or prev_bu) else bu_id

                change_pct = (bu_change / prev_em * 100.0) if prev_em > 0 else 0.0
                contrib_pct = (bu_change / total_change * 100.0) if total_change != 0 else 0.0

                if bu_change < -0.01:
                    direction = TrendDirection.DECREASING
                elif bu_change > 0.01:
                    direction = TrendDirection.INCREASING
                else:
                    direction = TrendDirection.STABLE

                explanation = self._generate_attribution_explanation(
                    bu_name, bu_change, change_pct, direction
                )

                self._attributions.append(DriverAttribution(
                    driver_id=bu_id,
                    driver_name=bu_name,
                    driver_type="business_unit",
                    change_tco2e=round(bu_change, 4),
                    change_pct=round(change_pct, 2),
                    contribution_to_total_change_pct=round(contrib_pct, 2),
                    direction=direction,
                    explanation=explanation,
                ))

            # Sort by absolute change descending
            self._attributions.sort(key=lambda a: abs(a.change_tco2e), reverse=True)

        outputs["attribution_count"] = len(self._attributions)
        outputs["increasing_drivers"] = sum(
            1 for a in self._attributions if a.direction == TrendDirection.INCREASING
        )
        outputs["decreasing_drivers"] = sum(
            1 for a in self._attributions if a.direction == TrendDirection.DECREASING
        )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Attribution: %d drivers identified", len(self._attributions))
        return PhaseResult(
            phase_name="attribution",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_sample_bus(
        self, records: List[AnnualEmissionRecord]
    ) -> List[BusinessUnitEmission]:
        """Generate sample business unit breakdowns from total records."""
        bu_shares = [
            ("BU-001", "Manufacturing", 0.40),
            ("BU-002", "Logistics", 0.25),
            ("BU-003", "Office Operations", 0.15),
            ("BU-004", "Data Centers", 0.12),
            ("BU-005", "Other", 0.08),
        ]
        results: List[BusinessUnitEmission] = []
        for rec in records:
            for bu_id, bu_name, share in bu_shares:
                # Add slight variation per year
                variation = 1.0 + (hash(f"{bu_id}_{rec.year}") % 10 - 5) / 100.0
                emissions = rec.total_tco2e * share * variation
                results.append(BusinessUnitEmission(
                    bu_id=bu_id,
                    bu_name=bu_name,
                    year=rec.year,
                    emissions_tco2e=round(emissions, 2),
                    activity_level=round(rec.activity_level * share, 2) if rec.activity_level > 0 else 0.0,
                    share_pct=round(share * 100, 2),
                ))
        return results

    def _generate_attribution_explanation(
        self, name: str, change: float, pct: float, direction: TrendDirection
    ) -> str:
        """Generate a human-readable explanation for a driver attribution."""
        if direction == TrendDirection.DECREASING:
            return (
                f"{name} reduced emissions by {abs(change):.0f} tCO2e ({abs(pct):.1f}%). "
                "Contributing to overall decarbonisation progress."
            )
        elif direction == TrendDirection.INCREASING:
            return (
                f"{name} increased emissions by {change:.0f} tCO2e ({pct:.1f}%). "
                "Investigation and corrective action recommended."
            )
        else:
            return f"{name} emissions remained stable (change: {change:.0f} tCO2e)."

    # -------------------------------------------------------------------------
    # Phase 4: Forecasting
    # -------------------------------------------------------------------------

    async def _phase_forecasting(self, config: AdvancedProgressConfig) -> PhaseResult:
        """Generate rolling forecast based on trends and planned actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        target_map: Dict[int, float] = {}
        for tp in config.target_pathway:
            target_map[tp.year] = tp.target_tco2e

        self._forecasts = []
        if len(self._records) >= 2:
            # Linear regression on historical data
            years = [r.year for r in self._records]
            emissions = [r.total_tco2e for r in self._records]
            slope, intercept = self._linear_regression(years, emissions)

            # Calculate forecast standard error
            residuals = [e - (slope * y + intercept) for y, e in zip(years, emissions)]
            se = (sum(r ** 2 for r in residuals) / max(len(residuals) - 2, 1)) ** 0.5

            latest_year = self._records[-1].year
            for fyr in range(latest_year + 1, latest_year + config.forecast_years + 1):
                forecast = slope * fyr + intercept
                forecast = max(forecast, 0.0)
                lower = max(forecast - 1.645 * se, 0.0)  # 90% CI
                upper = forecast + 1.645 * se

                target = target_map.get(fyr, 0.0)
                if target <= 0 and target_map:
                    target = self._interpolate_target(target_map, fyr)

                deviation = 0.0
                if target > 0:
                    deviation = ((forecast - target) / target) * 100.0

                self._forecasts.append(ForecastPoint(
                    year=fyr,
                    forecast_tco2e=round(forecast, 2),
                    lower_bound_tco2e=round(lower, 2),
                    upper_bound_tco2e=round(upper, 2),
                    target_tco2e=round(target, 2),
                    deviation_pct=round(deviation, 2),
                    confidence_level=0.90,
                ))

        outputs["forecast_count"] = len(self._forecasts)
        outputs["trend_slope_tco2e_yr"] = round(slope, 2) if len(self._records) >= 2 else 0.0
        if self._forecasts:
            outputs["next_year_forecast"] = self._forecasts[0].forecast_tco2e
            outputs["next_year_deviation_pct"] = self._forecasts[0].deviation_pct

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Forecasting: %d years forecasted", len(self._forecasts))
        return PhaseResult(
            phase_name="forecasting",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _linear_regression(
        self, x_vals: List[float], y_vals: List[float]
    ) -> Tuple[float, float]:
        """Simple linear regression returning (slope, intercept)."""
        n = len(x_vals)
        if n < 2:
            return (0.0, y_vals[0] if y_vals else 0.0)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        if abs(denominator) < 1e-12:
            return (0.0, y_mean)
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return (slope, intercept)

    def _interpolate_target(self, target_map: Dict[int, float], year: int) -> float:
        """Linearly interpolate a target for a given year."""
        years = sorted(target_map.keys())
        if not years:
            return 0.0
        if year <= years[0]:
            return target_map[years[0]]
        if year >= years[-1]:
            return target_map[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                y0, y1 = years[i], years[i + 1]
                v0, v1 = target_map[y0], target_map[y1]
                frac = (year - y0) / (y1 - y0)
                return v0 + frac * (v1 - v0)
        return target_map[years[-1]]

    # -------------------------------------------------------------------------
    # Phase 5: Alert Generation
    # -------------------------------------------------------------------------

    async def _phase_alert_generation(self, config: AdvancedProgressConfig) -> PhaseResult:
        """Compare forecast to target pathway and generate alerts."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        thresholds = config.alert_thresholds
        info_threshold = thresholds.get("info", 5.0)
        warning_threshold = thresholds.get("warning", 10.0)
        critical_threshold = thresholds.get("critical", 20.0)

        self._alerts = []
        for forecast in self._forecasts:
            dev = abs(forecast.deviation_pct)
            if forecast.target_tco2e <= 0:
                continue

            if dev >= critical_threshold:
                severity = AlertSeverity.CRITICAL
                action = (
                    "Urgent action required: emissions trajectory significantly exceeds target. "
                    "Convene emergency review and implement accelerated reduction measures."
                )
            elif dev >= warning_threshold:
                severity = AlertSeverity.WARNING
                action = (
                    "Attention needed: emissions trajectory deviating from target pathway. "
                    "Review reduction plan and consider additional abatement actions."
                )
            elif dev >= info_threshold:
                severity = AlertSeverity.INFO
                action = (
                    "Minor deviation detected. Monitor closely and ensure planned "
                    "reduction actions are on schedule."
                )
            else:
                continue  # No alert needed

            direction = "above" if forecast.deviation_pct > 0 else "below"
            message = (
                f"Year {forecast.year}: Forecast ({forecast.forecast_tco2e:.0f} tCO2e) is "
                f"{dev:.1f}% {direction} target ({forecast.target_tco2e:.0f} tCO2e)."
            )

            self._alerts.append(ProgressAlert(
                alert_id=f"ALERT-{forecast.year}-{severity.value}",
                severity=severity,
                year=forecast.year,
                message=message,
                deviation_pct=round(forecast.deviation_pct, 2),
                forecast_tco2e=forecast.forecast_tco2e,
                target_tco2e=forecast.target_tco2e,
                recommended_action=action,
            ))

        info_count = sum(1 for a in self._alerts if a.severity == AlertSeverity.INFO)
        warn_count = sum(1 for a in self._alerts if a.severity == AlertSeverity.WARNING)
        crit_count = sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL)

        outputs["total_alerts"] = len(self._alerts)
        outputs["info_alerts"] = info_count
        outputs["warning_alerts"] = warn_count
        outputs["critical_alerts"] = crit_count

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Alerts: %d total (info=%d, warn=%d, crit=%d)",
                         len(self._alerts), info_count, warn_count, crit_count)
        return PhaseResult(
            phase_name="alert_generation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
