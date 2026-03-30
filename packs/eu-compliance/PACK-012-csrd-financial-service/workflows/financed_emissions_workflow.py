# -*- coding: utf-8 -*-
"""
Financed Emissions Workflow
==============================

Five-phase workflow for calculating and reporting financed emissions (Scope 3
Category 15) for financial institutions under CSRD/ESRS E1 and PCAF Standard.

Phases:
    1. DataCollection - Ingest counterparty financials, emissions, exposure data
    2. AttributionCalculation - Compute per-asset attribution factors using PCAF
    3. QualityAssessment - Score data quality (1-5) per PCAF, flag gaps
    4. Aggregation - Aggregate by asset class, sector, geography; compute WACI
    5. Reporting - Generate PCAF-compliant disclosure with YoY comparison

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

# ---------------------------------------------------------------------------
#  Shared data models
# ---------------------------------------------------------------------------

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class PCAFAssetClass(str, Enum):
    """PCAF asset classes for financed emissions."""
    LISTED_EQUITY = "LISTED_EQUITY"
    CORPORATE_BONDS = "CORPORATE_BONDS"
    BUSINESS_LOANS = "BUSINESS_LOANS"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    COMMERCIAL_REAL_ESTATE = "COMMERCIAL_REAL_ESTATE"
    MORTGAGES = "MORTGAGES"
    MOTOR_VEHICLE_LOANS = "MOTOR_VEHICLE_LOANS"
    SOVEREIGN_DEBT = "SOVEREIGN_DEBT"

# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class CounterpartyExposure(BaseModel):
    """Single counterparty exposure record."""
    counterparty_id: str = Field(..., description="Unique counterparty ID")
    counterparty_name: str = Field(default="", description="Counterparty name")
    asset_class: str = Field(..., description="PCAF asset class")
    outstanding_amount: float = Field(..., ge=0.0, description="Outstanding exposure EUR")
    sector: str = Field(default="", description="NACE/GICS sector code")
    country: str = Field(default="", description="ISO country code")
    scope1_emissions: Optional[float] = Field(None, ge=0.0)
    scope2_emissions: Optional[float] = Field(None, ge=0.0)
    scope3_emissions: Optional[float] = Field(None, ge=0.0)
    total_equity_plus_debt: Optional[float] = Field(None, ge=0.0, description="EVIC EUR")
    revenue: Optional[float] = Field(None, ge=0.0)
    data_quality_score: int = Field(default=5, ge=1, le=5)

class FinancedEmissionsInput(BaseModel):
    """Input for the financed emissions workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_period: str = Field(..., description="Reporting period YYYY")
    base_currency: str = Field(default="EUR")
    exposures: List[CounterpartyExposure] = Field(default_factory=list)
    include_scope3: bool = Field(default=False)
    pcaf_version: str = Field(default="2022")
    prior_period_total_tco2e: Optional[float] = Field(None)
    prior_period_waci: Optional[float] = Field(None)
    target_data_quality: float = Field(default=3.0, le=5.0, ge=1.0)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        if len(v) == 4:
            int(v)
        else:
            datetime.strptime(v, "%Y-%m-%d")
        return v

class FinancedEmissionsResult(WorkflowResult):
    """Result from the financed emissions workflow."""
    total_financed_emissions_tco2e: float = Field(default=0.0)
    scope1_financed: float = Field(default=0.0)
    scope2_financed: float = Field(default=0.0)
    scope3_financed: float = Field(default=0.0)
    weighted_data_quality_score: float = Field(default=5.0)
    waci: float = Field(default=0.0)
    portfolio_value_eur: float = Field(default=0.0)
    counterparties_covered: int = Field(default=0)
    asset_classes_covered: int = Field(default=0)
    yoy_change_pct: Optional[float] = Field(None)

# ---------------------------------------------------------------------------
#  Phase 1: Data Collection
# ---------------------------------------------------------------------------

class DataCollectionPhase:
    """Ingest and validate counterparty financial and emissions data."""

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            exposures = config.get("exposures", [])
            outputs["total_exposures"] = len(exposures)

            valid_exposures = []
            missing_emissions = 0
            missing_evic = 0

            for exp in exposures:
                has_em = (exp.get("scope1_emissions") is not None
                          or exp.get("scope2_emissions") is not None)
                has_evic = exp.get("total_equity_plus_debt") is not None
                if not has_em:
                    missing_emissions += 1
                if not has_evic:
                    missing_evic += 1
                valid_exposures.append({**exp, "has_emissions": has_em, "has_evic": has_evic})

            outputs["valid_exposures"] = valid_exposures
            outputs["missing_emissions_count"] = missing_emissions
            outputs["missing_evic_count"] = missing_evic
            outputs["total_outstanding_eur"] = sum(
                e.get("outstanding_amount", 0.0) for e in exposures
            )
            asset_classes = set(e.get("asset_class", "") for e in exposures)
            outputs["asset_classes"] = list(asset_classes)
            outputs["asset_classes_count"] = len(asset_classes)
            sectors = set(e.get("sector", "") for e in exposures if e.get("sector"))
            outputs["sectors"] = list(sectors)

            if missing_emissions > 0:
                pct = missing_emissions / max(len(exposures), 1) * 100
                warnings.append(f"{missing_emissions} exposures ({pct:.1f}%) missing emissions")

            status = PhaseStatus.COMPLETED
            records = len(exposures)

        except Exception as exc:
            logger.error("DataCollection failed: %s", exc, exc_info=True)
            errors.append(f"Data collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records,
        )

# ---------------------------------------------------------------------------
#  Phase 2: Attribution Calculation
# ---------------------------------------------------------------------------

class AttributionCalculationPhase:
    """Compute per-asset attribution factors and financed emissions via PCAF."""

    PHASE_NAME = "attribution_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            collection = context.get_phase_output("data_collection")
            exposures = collection.get("valid_exposures", [])
            include_s3 = config.get("include_scope3", False)

            attributed = []
            total_s1 = total_s2 = total_s3 = 0.0

            for exp in exposures:
                outstanding = exp.get("outstanding_amount", 0.0)
                evic = exp.get("total_equity_plus_debt")
                revenue = exp.get("revenue")
                s1 = exp.get("scope1_emissions", 0.0) or 0.0
                s2 = exp.get("scope2_emissions", 0.0) or 0.0
                s3 = exp.get("scope3_emissions", 0.0) or 0.0

                if evic and evic > 0:
                    af = outstanding / evic
                    method = "evic"
                elif revenue and revenue > 0:
                    af = outstanding / revenue
                    method = "revenue"
                else:
                    af = 1.0
                    method = "full_attribution"
                    warnings.append(f"Counterparty {exp.get('counterparty_id','?')}: full attribution")

                fs1 = s1 * af
                fs2 = s2 * af
                fs3 = s3 * af if include_s3 else 0.0

                total_s1 += fs1
                total_s2 += fs2
                total_s3 += fs3

                attributed.append({
                    "counterparty_id": exp.get("counterparty_id", ""),
                    "counterparty_name": exp.get("counterparty_name", ""),
                    "asset_class": exp.get("asset_class", ""),
                    "sector": exp.get("sector", ""),
                    "country": exp.get("country", ""),
                    "outstanding_amount": outstanding,
                    "attribution_factor": round(af, 6),
                    "attribution_method": method,
                    "financed_s1": round(fs1, 4),
                    "financed_s2": round(fs2, 4),
                    "financed_s3": round(fs3, 4),
                    "financed_total": round(fs1 + fs2 + fs3, 4),
                    "data_quality_score": exp.get("data_quality_score", 5),
                })

            outputs["attributed_exposures"] = attributed
            outputs["total_s1_financed"] = round(total_s1, 4)
            outputs["total_s2_financed"] = round(total_s2, 4)
            outputs["total_s3_financed"] = round(total_s3, 4)
            outputs["total_financed_emissions"] = round(total_s1 + total_s2 + total_s3, 4)
            outputs["counterparties_calculated"] = len(attributed)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Attribution failed: %s", exc, exc_info=True)
            errors.append(f"Attribution calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 3: Quality Assessment
# ---------------------------------------------------------------------------

class QualityAssessmentPhase:
    """Score data quality (1-5) per PCAF, compute weighted average."""

    PHASE_NAME = "quality_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            attribution = context.get_phase_output("attribution_calculation")
            attributed = attribution.get("attributed_exposures", [])
            target_dq = config.get("target_data_quality", 3.0)

            total_out = sum(a.get("outstanding_amount", 0.0) for a in attributed)
            wdq_sum = 0.0
            dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            low_quality = []

            for a in attributed:
                dq = a.get("data_quality_score", 5)
                out = a.get("outstanding_amount", 0.0)
                wdq_sum += dq * (out / max(total_out, 1.0))
                dist[dq] = dist.get(dq, 0) + 1
                if dq >= 4:
                    low_quality.append({
                        "counterparty_id": a.get("counterparty_id", ""),
                        "counterparty_name": a.get("counterparty_name", ""),
                        "data_quality_score": dq,
                        "outstanding_amount": out,
                    })

            wdq = round(wdq_sum, 2)
            outputs["weighted_data_quality_score"] = wdq
            outputs["quality_distribution"] = dist
            outputs["low_quality_counterparties"] = low_quality
            outputs["low_quality_count"] = len(low_quality)
            outputs["meets_target"] = wdq <= target_dq
            outputs["target_data_quality"] = target_dq

            ac_quality: Dict[str, List[int]] = {}
            for a in attributed:
                ac_quality.setdefault(a.get("asset_class", ""), []).append(
                    a.get("data_quality_score", 5)
                )
            outputs["asset_class_quality"] = {
                ac: round(sum(s) / max(len(s), 1), 2) for ac, s in ac_quality.items()
            }

            if wdq > target_dq:
                warnings.append(f"Weighted DQ ({wdq:.2f}) exceeds target ({target_dq:.1f})")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("QualityAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Quality assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 4: Aggregation
# ---------------------------------------------------------------------------

class AggregationPhase:
    """Aggregate financed emissions by asset class, sector, geography; compute WACI."""

    PHASE_NAME = "aggregation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            attribution = context.get_phase_output("attribution_calculation")
            attributed = attribution.get("attributed_exposures", [])
            total_out = sum(a.get("outstanding_amount", 0.0) for a in attributed)

            by_ac: Dict[str, Dict[str, Any]] = {}
            for a in attributed:
                ac = a.get("asset_class", "OTHER")
                if ac not in by_ac:
                    by_ac[ac] = {"asset_class": ac, "count": 0,
                                 "outstanding_total": 0.0, "financed_emissions": 0.0}
                by_ac[ac]["count"] += 1
                by_ac[ac]["outstanding_total"] += a.get("outstanding_amount", 0.0)
                by_ac[ac]["financed_emissions"] += a.get("financed_total", 0.0)

            outputs["by_asset_class"] = list(by_ac.values())

            by_sector: Dict[str, Dict[str, Any]] = {}
            for a in attributed:
                sec = a.get("sector", "UNKNOWN") or "UNKNOWN"
                if sec not in by_sector:
                    by_sector[sec] = {"sector": sec, "count": 0,
                                      "outstanding_total": 0.0, "financed_emissions": 0.0}
                by_sector[sec]["count"] += 1
                by_sector[sec]["outstanding_total"] += a.get("outstanding_amount", 0.0)
                by_sector[sec]["financed_emissions"] += a.get("financed_total", 0.0)

            outputs["by_sector"] = sorted(
                by_sector.values(), key=lambda x: x["financed_emissions"], reverse=True
            )

            by_country: Dict[str, Dict[str, Any]] = {}
            for a in attributed:
                ctry = a.get("country", "UNKNOWN") or "UNKNOWN"
                if ctry not in by_country:
                    by_country[ctry] = {"country": ctry, "count": 0,
                                        "outstanding_total": 0.0, "financed_emissions": 0.0}
                by_country[ctry]["count"] += 1
                by_country[ctry]["outstanding_total"] += a.get("outstanding_amount", 0.0)
                by_country[ctry]["financed_emissions"] += a.get("financed_total", 0.0)

            outputs["by_country"] = list(by_country.values())

            # WACI = Sum(weight_i * intensity_i)
            waci = 0.0
            for a in attributed:
                weight = a.get("outstanding_amount", 0.0) / max(total_out, 1.0)
                rev = a.get("revenue") or a.get("outstanding_amount", 1.0) or 1.0
                intensity = a.get("financed_total", 0.0) / max(rev / 1_000_000, 0.001)
                waci += weight * intensity

            outputs["waci"] = round(waci, 4)
            outputs["portfolio_value_eur"] = total_out
            outputs["total_financed_emissions"] = attribution.get("total_financed_emissions", 0.0)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Aggregation failed: %s", exc, exc_info=True)
            errors.append(f"Aggregation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 5: Reporting
# ---------------------------------------------------------------------------

class ReportingPhase:
    """Generate PCAF-compliant disclosure with YoY comparison."""

    PHASE_NAME = "reporting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            attribution = context.get_phase_output("attribution_calculation")
            quality = context.get_phase_output("quality_assessment")
            aggregation = context.get_phase_output("aggregation")

            total = aggregation.get("total_financed_emissions", 0.0)
            pv = aggregation.get("portfolio_value_eur", 0.0)
            waci = aggregation.get("waci", 0.0)
            wdq = quality.get("weighted_data_quality_score", 5.0)

            prior_total = config.get("prior_period_total_tco2e")
            prior_waci = config.get("prior_period_waci")
            yoy_pct = None
            yoy_waci = None
            if prior_total and prior_total > 0:
                yoy_pct = round((total - prior_total) / prior_total * 100, 2)
            if prior_waci and prior_waci > 0:
                yoy_waci = round((waci - prior_waci) / prior_waci * 100, 2)

            outputs["disclosure"] = {
                "reporting_period": config.get("reporting_period", ""),
                "pcaf_version": config.get("pcaf_version", "2022"),
                "total_financed_emissions_tco2e": round(total, 2),
                "scope1_financed": attribution.get("total_s1_financed", 0.0),
                "scope2_financed": attribution.get("total_s2_financed", 0.0),
                "scope3_financed": attribution.get("total_s3_financed", 0.0),
                "portfolio_value_eur": pv,
                "waci": waci,
                "weighted_data_quality_score": wdq,
                "counterparties_covered": attribution.get("counterparties_calculated", 0),
                "asset_classes_covered": len(aggregation.get("by_asset_class", [])),
            }
            outputs["yoy_comparison"] = {
                "prior_total": prior_total, "current_total": round(total, 2),
                "yoy_change_pct": yoy_pct,
                "prior_waci": prior_waci, "current_waci": waci,
                "yoy_waci_change_pct": yoy_waci,
            }
            outputs["asset_class_breakdown"] = aggregation.get("by_asset_class", [])
            outputs["sector_breakdown"] = aggregation.get("by_sector", [])
            outputs["quality_summary"] = {
                "weighted_score": wdq,
                "distribution": quality.get("quality_distribution", {}),
                "meets_target": quality.get("meets_target", False),
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Reporting failed: %s", exc, exc_info=True)
            errors.append(f"Reporting failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class FinancedEmissionsWorkflow:
    """
    Five-phase financed emissions workflow for CSRD financial institutions.

    Orchestrates PCAF-aligned financed emissions calculation from data
    collection through attribution, quality assessment, aggregation, and
    PCAF reporting. Supports checkpoint/resume and phase skipping.
    """

    WORKFLOW_NAME = "financed_emissions"

    PHASE_ORDER = [
        "data_collection", "attribution_calculation",
        "quality_assessment", "aggregation", "reporting",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_collection": DataCollectionPhase(),
            "attribution_calculation": AttributionCalculationPhase(),
            "quality_assessment": QualityAssessmentPhase(),
            "aggregation": AggregationPhase(),
            "reporting": ReportingPhase(),
        }

    async def run(self, input_data: FinancedEmissionsInput) -> FinancedEmissionsResult:
        """Execute the complete 5-phase financed emissions workflow."""
        started_at = utcnow()
        logger.info(
            "Starting financed emissions workflow %s org=%s period=%s",
            self.workflow_id, input_data.organization_id, input_data.reporting_period,
        )
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue
            if context.is_phase_completed(phase_name):
                continue

            self._notify_progress(phase_name, f"Starting: {phase_name}",
                                  idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "data_collection":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return FinancedEmissionsResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            total_financed_emissions_tco2e=summary.get("total_financed_emissions_tco2e", 0.0),
            scope1_financed=summary.get("scope1_financed", 0.0),
            scope2_financed=summary.get("scope2_financed", 0.0),
            scope3_financed=summary.get("scope3_financed", 0.0),
            weighted_data_quality_score=summary.get("weighted_data_quality_score", 5.0),
            waci=summary.get("waci", 0.0),
            portfolio_value_eur=summary.get("portfolio_value_eur", 0.0),
            counterparties_covered=summary.get("counterparties_covered", 0),
            asset_classes_covered=summary.get("asset_classes_covered", 0),
            yoy_change_pct=summary.get("yoy_change_pct"),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        attr = context.get_phase_output("attribution_calculation")
        qual = context.get_phase_output("quality_assessment")
        agg = context.get_phase_output("aggregation")
        rep = context.get_phase_output("reporting")
        disc = rep.get("disclosure", {})
        yoy = rep.get("yoy_comparison", {})
        return {
            "total_financed_emissions_tco2e": disc.get("total_financed_emissions_tco2e", 0.0),
            "scope1_financed": attr.get("total_s1_financed", 0.0),
            "scope2_financed": attr.get("total_s2_financed", 0.0),
            "scope3_financed": attr.get("total_s3_financed", 0.0),
            "weighted_data_quality_score": qual.get("weighted_data_quality_score", 5.0),
            "waci": agg.get("waci", 0.0),
            "portfolio_value_eur": agg.get("portfolio_value_eur", 0.0),
            "counterparties_covered": attr.get("counterparties_calculated", 0),
            "asset_classes_covered": len(agg.get("by_asset_class", [])),
            "yoy_change_pct": yoy.get("yoy_change_pct"),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
