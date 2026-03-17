# -*- coding: utf-8 -*-
"""
Insurance Emissions Workflow
================================

Four-phase workflow for calculating insurance-associated emissions for
(re)insurance undertakings under CSRD/ESRS E1 and PCAF Insurance-Associated
Emissions Standard.

Phases:
    1. PolicyDataIngestion - Ingest policy-level premium and claims data
    2. EmissionAttribution - Attribute emissions to underwriting portfolio
    3. ReinsuranceAdjustment - Adjust for ceded reinsurance
    4. ReportGeneration - Generate insurance emissions disclosure

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

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
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
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")



class InsuranceLineOfBusiness(str, Enum):
    """Insurance lines of business."""
    MOTOR = "MOTOR"
    PROPERTY = "PROPERTY"
    LIABILITY = "LIABILITY"
    MARINE = "MARINE"
    AVIATION = "AVIATION"
    ENERGY = "ENERGY"
    LIFE = "LIFE"
    HEALTH = "HEALTH"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class InsurancePolicy(BaseModel):
    """Single insurance policy record."""
    policy_id: str = Field(..., description="Policy identifier")
    policyholder_name: str = Field(default="")
    line_of_business: str = Field(default="OTHER")
    gross_written_premium: float = Field(..., ge=0.0, description="GWP in EUR")
    net_earned_premium: Optional[float] = Field(None, ge=0.0)
    claims_incurred: Optional[float] = Field(None, ge=0.0)
    policyholder_sector: str = Field(default="", description="NACE sector code")
    policyholder_country: str = Field(default="")
    scope1_emissions: Optional[float] = Field(None, ge=0.0, description="Policyholder Scope 1 tCO2e")
    scope2_emissions: Optional[float] = Field(None, ge=0.0)
    policyholder_revenue: Optional[float] = Field(None, ge=0.0)
    data_quality_score: int = Field(default=5, ge=1, le=5)
    ceded_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Reinsurance ceded %")


class InsuranceEmissionsInput(BaseModel):
    """Input for the insurance emissions workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_period: str = Field(..., description="Reporting period YYYY")
    policies: List[InsurancePolicy] = Field(default_factory=list)
    total_market_premium: Optional[float] = Field(None, ge=0.0)
    include_scope2: bool = Field(default=True)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        if len(v) == 4:
            int(v)
        return v


class InsuranceEmissionsResult(WorkflowResult):
    """Result from the insurance emissions workflow."""
    gross_attributed_emissions_tco2e: float = Field(default=0.0)
    net_attributed_emissions_tco2e: float = Field(default=0.0)
    reinsurance_adjustment_tco2e: float = Field(default=0.0)
    total_gwp_eur: float = Field(default=0.0)
    policies_covered: int = Field(default=0)
    emission_intensity_per_meur_gwp: float = Field(default=0.0)
    weighted_data_quality_score: float = Field(default=5.0)
    lines_of_business_covered: int = Field(default=0)


# ---------------------------------------------------------------------------
#  Phase 1: Policy Data Ingestion
# ---------------------------------------------------------------------------

class PolicyDataIngestionPhase:
    """Ingest and validate policy-level premium, claims, and emissions data."""

    PHASE_NAME = "policy_data_ingestion"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            policies = config.get("policies", [])
            outputs["total_policies"] = len(policies)

            total_gwp = sum(p.get("gross_written_premium", 0.0) for p in policies)
            outputs["total_gwp"] = total_gwp

            missing_emissions = sum(
                1 for p in policies
                if p.get("scope1_emissions") is None and p.get("scope2_emissions") is None
            )
            outputs["missing_emissions_count"] = missing_emissions

            lob_set = set(p.get("line_of_business", "OTHER") for p in policies)
            outputs["lines_of_business"] = list(lob_set)
            outputs["lines_of_business_count"] = len(lob_set)

            outputs["validated_policies"] = policies

            if missing_emissions > 0:
                pct = missing_emissions / max(len(policies), 1) * 100
                warnings.append(f"{missing_emissions} policies ({pct:.1f}%) missing emissions data")

            status = PhaseStatus.COMPLETED
            records = len(policies)

        except Exception as exc:
            logger.error("PolicyDataIngestion failed: %s", exc, exc_info=True)
            errors.append(f"Policy data ingestion failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records,
        )


# ---------------------------------------------------------------------------
#  Phase 2: Emission Attribution
# ---------------------------------------------------------------------------

class EmissionAttributionPhase:
    """Attribute emissions to underwriting portfolio using premium-based approach."""

    PHASE_NAME = "emission_attribution"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            ingestion = context.get_phase_output("policy_data_ingestion")
            policies = ingestion.get("validated_policies", [])
            include_s2 = config.get("include_scope2", True)

            attributed = []
            total_gross = 0.0

            for policy in policies:
                gwp = policy.get("gross_written_premium", 0.0)
                s1 = policy.get("scope1_emissions", 0.0) or 0.0
                s2 = (policy.get("scope2_emissions", 0.0) or 0.0) if include_s2 else 0.0
                revenue = policy.get("policyholder_revenue")

                # Attribution factor: GWP / policyholder revenue
                if revenue and revenue > 0:
                    af = gwp / revenue
                    method = "revenue_based"
                else:
                    af = 1.0
                    method = "full_attribution"

                gross_attr = (s1 + s2) * af
                total_gross += gross_attr

                attributed.append({
                    "policy_id": policy.get("policy_id", ""),
                    "policyholder_name": policy.get("policyholder_name", ""),
                    "line_of_business": policy.get("line_of_business", "OTHER"),
                    "gwp": gwp,
                    "attribution_factor": round(af, 6),
                    "attribution_method": method,
                    "gross_attributed_emissions": round(gross_attr, 4),
                    "ceded_pct": policy.get("ceded_pct", 0.0),
                    "data_quality_score": policy.get("data_quality_score", 5),
                    "sector": policy.get("policyholder_sector", ""),
                    "country": policy.get("policyholder_country", ""),
                })

            outputs["attributed_policies"] = attributed
            outputs["total_gross_attributed"] = round(total_gross, 4)
            outputs["policies_attributed"] = len(attributed)

            # Weighted data quality
            total_gwp = sum(a.get("gwp", 0.0) for a in attributed)
            wdq = sum(
                a.get("data_quality_score", 5) * (a.get("gwp", 0.0) / max(total_gwp, 1.0))
                for a in attributed
            )
            outputs["weighted_data_quality_score"] = round(wdq, 2)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("EmissionAttribution failed: %s", exc, exc_info=True)
            errors.append(f"Emission attribution failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 3: Reinsurance Adjustment
# ---------------------------------------------------------------------------

class ReinsuranceAdjustmentPhase:
    """Adjust attributed emissions for ceded reinsurance."""

    PHASE_NAME = "reinsurance_adjustment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            attribution = context.get_phase_output("emission_attribution")
            attributed = attribution.get("attributed_policies", [])
            total_gross = attribution.get("total_gross_attributed", 0.0)

            total_ceded = 0.0
            adjusted = []

            for policy in attributed:
                gross = policy.get("gross_attributed_emissions", 0.0)
                ceded_pct = policy.get("ceded_pct", 0.0)
                ceded_amount = gross * ceded_pct / 100.0
                net = gross - ceded_amount
                total_ceded += ceded_amount

                adjusted.append({
                    **policy,
                    "ceded_emissions": round(ceded_amount, 4),
                    "net_attributed_emissions": round(net, 4),
                })

            outputs["adjusted_policies"] = adjusted
            outputs["total_gross_attributed"] = round(total_gross, 4)
            outputs["total_ceded_emissions"] = round(total_ceded, 4)
            outputs["total_net_attributed"] = round(total_gross - total_ceded, 4)

            # By line of business
            by_lob: Dict[str, Dict[str, float]] = {}
            for a in adjusted:
                lob = a.get("line_of_business", "OTHER")
                if lob not in by_lob:
                    by_lob[lob] = {"gross": 0.0, "net": 0.0, "gwp": 0.0}
                by_lob[lob]["gross"] += a.get("gross_attributed_emissions", 0.0)
                by_lob[lob]["net"] += a.get("net_attributed_emissions", 0.0)
                by_lob[lob]["gwp"] += a.get("gwp", 0.0)

            outputs["by_line_of_business"] = {
                k: {kk: round(vv, 2) for kk, vv in v.items()}
                for k, v in by_lob.items()
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ReinsuranceAdjustment failed: %s", exc, exc_info=True)
            errors.append(f"Reinsurance adjustment failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 4: Report Generation
# ---------------------------------------------------------------------------

class ReportGenerationPhase:
    """Generate insurance emissions disclosure report."""

    PHASE_NAME = "report_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            ingestion = context.get_phase_output("policy_data_ingestion")
            attribution = context.get_phase_output("emission_attribution")
            reinsurance = context.get_phase_output("reinsurance_adjustment")

            total_gwp = ingestion.get("total_gwp", 0.0)
            gross = reinsurance.get("total_gross_attributed", 0.0)
            net = reinsurance.get("total_net_attributed", 0.0)
            ceded = reinsurance.get("total_ceded_emissions", 0.0)

            outputs["disclosure"] = {
                "reporting_period": config.get("reporting_period", ""),
                "gross_attributed_emissions_tco2e": round(gross, 2),
                "net_attributed_emissions_tco2e": round(net, 2),
                "reinsurance_adjustment_tco2e": round(ceded, 2),
                "total_gwp_eur": round(total_gwp, 2),
                "policies_covered": ingestion.get("total_policies", 0),
                "emission_intensity_per_meur_gwp": round(
                    net / max(total_gwp / 1_000_000, 0.001), 2
                ),
                "weighted_data_quality_score": attribution.get("weighted_data_quality_score", 5.0),
                "lines_of_business_covered": ingestion.get("lines_of_business_count", 0),
            }
            outputs["by_line_of_business"] = reinsurance.get("by_line_of_business", {})

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ReportGeneration failed: %s", exc, exc_info=True)
            errors.append(f"Report generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
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

class InsuranceEmissionsWorkflow:
    """Four-phase insurance emissions workflow for CSRD (re)insurance undertakings."""

    WORKFLOW_NAME = "insurance_emissions"
    PHASE_ORDER = ["policy_data_ingestion", "emission_attribution",
                    "reinsurance_adjustment", "report_generation"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "policy_data_ingestion": PolicyDataIngestionPhase(),
            "emission_attribution": EmissionAttributionPhase(),
            "reinsurance_adjustment": ReinsuranceAdjustmentPhase(),
            "report_generation": ReportGenerationPhase(),
        }

    async def run(self, input_data: InsuranceEmissionsInput) -> InsuranceEmissionsResult:
        """Execute the complete 4-phase insurance emissions workflow."""
        started_at = _utcnow()
        logger.info("Starting insurance emissions workflow %s org=%s",
                     self.workflow_id, input_data.organization_id)
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
            self._notify_progress(phase_name, f"Starting: {phase_name}", idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "policy_data_ingestion":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=_utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = _utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return InsuranceEmissionsResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            gross_attributed_emissions_tco2e=summary.get("gross_attributed_emissions_tco2e", 0.0),
            net_attributed_emissions_tco2e=summary.get("net_attributed_emissions_tco2e", 0.0),
            reinsurance_adjustment_tco2e=summary.get("reinsurance_adjustment_tco2e", 0.0),
            total_gwp_eur=summary.get("total_gwp_eur", 0.0),
            policies_covered=summary.get("policies_covered", 0),
            emission_intensity_per_meur_gwp=summary.get("emission_intensity_per_meur_gwp", 0.0),
            weighted_data_quality_score=summary.get("weighted_data_quality_score", 5.0),
            lines_of_business_covered=summary.get("lines_of_business_covered", 0),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        rep = context.get_phase_output("report_generation")
        disc = rep.get("disclosure", {})
        return {
            "gross_attributed_emissions_tco2e": disc.get("gross_attributed_emissions_tco2e", 0.0),
            "net_attributed_emissions_tco2e": disc.get("net_attributed_emissions_tco2e", 0.0),
            "reinsurance_adjustment_tco2e": disc.get("reinsurance_adjustment_tco2e", 0.0),
            "total_gwp_eur": disc.get("total_gwp_eur", 0.0),
            "policies_covered": disc.get("policies_covered", 0),
            "emission_intensity_per_meur_gwp": disc.get("emission_intensity_per_meur_gwp", 0.0),
            "weighted_data_quality_score": disc.get("weighted_data_quality_score", 5.0),
            "lines_of_business_covered": disc.get("lines_of_business_covered", 0),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
