# -*- coding: utf-8 -*-
"""
Circular Economy Workflow
=========================

Four-phase workflow for assessing circular economy readiness including
material flow analysis, waste stream management, circularity metrics
(MCI), and Extended Producer Responsibility (EPR) compliance.

Phases:
    1. MaterialFlowMapping - Map material inputs, outputs, and loops
    2. WasteAnalysis - Classify waste streams and diversion rates
    3. CircularityMetrics - Compute MCI score and recycled content
    4. EPRCompliance - Assess EPR scheme participation and obligations

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

from pydantic import BaseModel, Field

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


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
#  Input / Result
# ---------------------------------------------------------------------------

class MaterialFlow(BaseModel):
    """Single material flow record."""
    material_id: str = Field(...)
    material_name: str = Field(default="")
    material_type: str = Field(default="virgin", description="virgin, recycled, bio-based")
    mass_kg: float = Field(default=0.0, ge=0.0)
    flow_direction: str = Field(default="input", description="input or output")
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    is_critical_raw_material: bool = Field(default=False)


class WasteStream(BaseModel):
    """Single waste stream record."""
    waste_id: str = Field(...)
    waste_type: str = Field(default="", description="Waste type classification")
    mass_kg: float = Field(default=0.0, ge=0.0)
    treatment_method: str = Field(default="landfill", description="recycle, reuse, recovery, landfill, incineration")
    hazardous: bool = Field(default=False)
    ewc_code: str = Field(default="", description="European Waste Catalogue code")


class EPRScheme(BaseModel):
    """Extended Producer Responsibility scheme."""
    scheme_id: str = Field(...)
    scheme_name: str = Field(default="")
    product_category: str = Field(default="")
    jurisdiction: str = Field(default="")
    registered: bool = Field(default=False)
    fees_paid: bool = Field(default=False)
    reporting_complete: bool = Field(default=False)


class CircularEconomyInput(BaseModel):
    """Input for circular economy workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2000, le=2100)
    material_flows: List[MaterialFlow] = Field(default_factory=list)
    waste_streams: List[WasteStream] = Field(default_factory=list)
    products: List[Dict[str, Any]] = Field(default_factory=list)
    epr_schemes: List[EPRScheme] = Field(default_factory=list)
    skip_phases: List[str] = Field(default_factory=list)


class CircularEconomyResult(WorkflowResult):
    """Result from the circular economy workflow."""
    mci_score: float = Field(default=0.0, description="Material Circularity Indicator 0-1")
    recycled_content: float = Field(default=0.0, description="Weighted recycled content %")
    waste_diversion: float = Field(default=0.0, description="Waste diversion rate %")
    epr_status: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------

class MaterialFlowMappingPhase:
    """Map material inputs, outputs, and circular loops."""

    PHASE_NAME = "material_flow_mapping"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Analyze material flows and classify by type."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            flows = config.get("material_flows", [])

            total_input = 0.0
            total_output = 0.0
            virgin_input = 0.0
            recycled_input = 0.0
            crm_mass = 0.0
            by_type: Dict[str, float] = {}

            for f in flows:
                mass = f.get("mass_kg", 0.0)
                direction = f.get("flow_direction", "input")
                mtype = f.get("material_type", "virgin")
                recycled_pct = f.get("recycled_content_pct", 0.0)
                is_crm = f.get("is_critical_raw_material", False)

                if direction == "input":
                    total_input += mass
                    recycled_portion = mass * (recycled_pct / 100.0)
                    recycled_input += recycled_portion
                    virgin_input += mass - recycled_portion
                    if is_crm:
                        crm_mass += mass
                else:
                    total_output += mass

                by_type[mtype] = by_type.get(mtype, 0.0) + mass

            outputs["total_input_kg"] = round(total_input, 2)
            outputs["total_output_kg"] = round(total_output, 2)
            outputs["virgin_input_kg"] = round(virgin_input, 2)
            outputs["recycled_input_kg"] = round(recycled_input, 2)
            outputs["crm_mass_kg"] = round(crm_mass, 2)
            outputs["by_material_type"] = {k: round(v, 2) for k, v in by_type.items()}
            outputs["recycled_content_pct"] = round(
                recycled_input / max(total_input, 0.001) * 100, 2
            )

            if not flows:
                warnings.append("No material flow data provided")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("MaterialFlowMapping failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class WasteAnalysisPhase:
    """Classify waste streams and compute diversion rates."""

    PHASE_NAME = "waste_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Analyze waste treatment methods and diversion metrics."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            waste = config.get("waste_streams", [])

            total_waste = 0.0
            diverted = 0.0
            hazardous_total = 0.0
            by_method: Dict[str, float] = {}

            diversion_methods = {"recycle", "reuse", "recovery"}

            for w in waste:
                mass = w.get("mass_kg", 0.0)
                method = w.get("treatment_method", "landfill")
                haz = w.get("hazardous", False)

                total_waste += mass
                if method in diversion_methods:
                    diverted += mass
                if haz:
                    hazardous_total += mass

                by_method[method] = by_method.get(method, 0.0) + mass

            diversion_rate = round(diverted / max(total_waste, 0.001) * 100, 2)

            outputs["total_waste_kg"] = round(total_waste, 2)
            outputs["diverted_kg"] = round(diverted, 2)
            outputs["landfill_kg"] = round(total_waste - diverted, 2)
            outputs["hazardous_kg"] = round(hazardous_total, 2)
            outputs["diversion_rate_pct"] = diversion_rate
            outputs["by_treatment_method"] = {k: round(v, 2) for k, v in by_method.items()}

            # Waste hierarchy compliance
            outputs["waste_hierarchy_score"] = self._hierarchy_score(by_method, total_waste)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("WasteAnalysis failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _hierarchy_score(self, by_method: Dict[str, float], total: float) -> float:
        """Compute waste hierarchy score (0-100)."""
        if total <= 0:
            return 0.0
        weights = {"reuse": 1.0, "recycle": 0.8, "recovery": 0.6, "incineration": 0.3, "landfill": 0.0}
        score = sum(
            (mass / total) * weights.get(method, 0.0)
            for method, mass in by_method.items()
        )
        return round(score * 100, 2)


class CircularityMetricsPhase:
    """Compute Material Circularity Indicator (MCI) and related metrics."""

    PHASE_NAME = "circularity_metrics"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Calculate MCI score per Ellen MacArthur Foundation methodology."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            flow_out = context.get_phase_output("material_flow_mapping")
            waste_out = context.get_phase_output("waste_analysis")

            total_input = flow_out.get("total_input_kg", 0.0)
            recycled_input = flow_out.get("recycled_input_kg", 0.0)
            total_waste = waste_out.get("total_waste_kg", 0.0)
            diverted = waste_out.get("diverted_kg", 0.0)

            # Simplified MCI: MCI = 1 - LFI * F(X)
            # LFI = linear flow index
            # F(X) = utility factor (simplified to 1.0)
            virgin_fraction = 1.0 - (recycled_input / max(total_input, 0.001))
            waste_fraction = 1.0 - (diverted / max(total_waste, 0.001)) if total_waste > 0 else 1.0
            lfi = (virgin_fraction + waste_fraction) / 2.0
            mci = max(0.0, min(1.0, 1.0 - lfi))

            outputs["mci_score"] = round(mci, 4)
            outputs["linear_flow_index"] = round(lfi, 4)
            outputs["virgin_fraction"] = round(virgin_fraction, 4)
            outputs["waste_fraction"] = round(waste_fraction, 4)
            outputs["recycled_content_pct"] = flow_out.get("recycled_content_pct", 0.0)
            outputs["waste_diversion_pct"] = waste_out.get("diversion_rate_pct", 0.0)
            outputs["waste_hierarchy_score"] = waste_out.get("waste_hierarchy_score", 0.0)
            outputs["crm_mass_kg"] = flow_out.get("crm_mass_kg", 0.0)

            # Rating
            if mci >= 0.8:
                outputs["circularity_rating"] = "EXCELLENT"
            elif mci >= 0.6:
                outputs["circularity_rating"] = "GOOD"
            elif mci >= 0.4:
                outputs["circularity_rating"] = "MODERATE"
            else:
                outputs["circularity_rating"] = "LOW"

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("CircularityMetrics failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class EPRCompliancePhase:
    """Assess Extended Producer Responsibility scheme compliance."""

    PHASE_NAME = "epr_compliance"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Evaluate EPR registration, fees, and reporting status."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            schemes = config.get("epr_schemes", [])

            total_schemes = len(schemes)
            registered = sum(1 for s in schemes if s.get("registered", False))
            fees_paid = sum(1 for s in schemes if s.get("fees_paid", False))
            reporting_done = sum(1 for s in schemes if s.get("reporting_complete", False))

            compliant = sum(
                1 for s in schemes
                if s.get("registered") and s.get("fees_paid") and s.get("reporting_complete")
            )

            non_compliant = []
            for s in schemes:
                issues = []
                if not s.get("registered"):
                    issues.append("not_registered")
                if not s.get("fees_paid"):
                    issues.append("fees_unpaid")
                if not s.get("reporting_complete"):
                    issues.append("reporting_incomplete")
                if issues:
                    non_compliant.append({
                        "scheme_id": s.get("scheme_id", ""),
                        "scheme_name": s.get("scheme_name", ""),
                        "jurisdiction": s.get("jurisdiction", ""),
                        "issues": issues,
                    })

            compliance_rate = round(
                compliant / max(total_schemes, 1) * 100, 2
            )

            outputs["total_schemes"] = total_schemes
            outputs["compliant_schemes"] = compliant
            outputs["non_compliant_schemes"] = non_compliant
            outputs["compliance_rate_pct"] = compliance_rate
            outputs["registration_rate"] = round(registered / max(total_schemes, 1) * 100, 2)
            outputs["fees_paid_rate"] = round(fees_paid / max(total_schemes, 1) * 100, 2)
            outputs["reporting_rate"] = round(reporting_done / max(total_schemes, 1) * 100, 2)

            if non_compliant:
                warnings.append(
                    f"{len(non_compliant)} EPR scheme(s) have compliance gaps"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("EPRCompliance failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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

class CircularEconomyWorkflow:
    """
    Four-phase circular economy readiness assessment workflow.

    Evaluates material circularity, waste management, MCI scoring,
    and EPR compliance for manufacturing organizations under ESRS E5.
    """

    WORKFLOW_NAME = "circular_economy"

    PHASE_ORDER = [
        "material_flow_mapping", "waste_analysis",
        "circularity_metrics", "epr_compliance",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize CircularEconomyWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "material_flow_mapping": MaterialFlowMappingPhase(),
            "waste_analysis": WasteAnalysisPhase(),
            "circularity_metrics": CircularityMetricsPhase(),
            "epr_compliance": EPRCompliancePhase(),
        }

    async def run(self, input_data: CircularEconomyInput) -> CircularEconomyResult:
        """Execute the complete 4-phase circular economy workflow."""
        started_at = _utcnow()
        logger.info("Starting circular economy workflow %s", self.workflow_id)
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
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = _utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return CircularEconomyResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            mci_score=summary.get("mci_score", 0.0),
            recycled_content=summary.get("recycled_content_pct", 0.0),
            waste_diversion=summary.get("waste_diversion_pct", 0.0),
            epr_status=summary.get("epr_status", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        circ = context.get_phase_output("circularity_metrics")
        epr = context.get_phase_output("epr_compliance")
        return {
            "mci_score": circ.get("mci_score", 0.0),
            "recycled_content_pct": circ.get("recycled_content_pct", 0.0),
            "waste_diversion_pct": circ.get("waste_diversion_pct", 0.0),
            "circularity_rating": circ.get("circularity_rating", ""),
            "epr_status": {
                "compliance_rate": epr.get("compliance_rate_pct", 0.0),
                "total_schemes": epr.get("total_schemes", 0),
                "compliant": epr.get("compliant_schemes", 0),
            },
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
