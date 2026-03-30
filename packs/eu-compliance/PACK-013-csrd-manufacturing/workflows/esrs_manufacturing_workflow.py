# -*- coding: utf-8 -*-
"""
ESRS Manufacturing Disclosure Workflow
=======================================

Four-phase workflow for generating ESRS-compliant disclosures tailored
to manufacturing organizations, covering E1-E5, S1, and G1 topical
standards with double materiality assessment.

Phases:
    1. MaterialityAssessment - Double materiality for manufacturing context
    2. DataPointCollection - Map and validate required ESRS data points
    3. DisclosureGeneration - Generate ESRS metric values per topical standard
    4. AuditPreparation - Prepare documentation for limited assurance

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

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

class ESRSManufacturingInput(BaseModel):
    """Input for ESRS manufacturing disclosure workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2000, le=2100)
    company_name: str = Field(default="")
    nace_codes: List[str] = Field(default_factory=list)
    # E1 - Climate
    emissions_data: Dict[str, Any] = Field(default_factory=dict)
    # E2 - Pollution
    energy_data: Dict[str, Any] = Field(default_factory=dict)
    # E3 - Water
    water_data: Dict[str, Any] = Field(default_factory=dict)
    # E5 - Circular economy
    waste_data: Dict[str, Any] = Field(default_factory=dict)
    # S1 - Own workforce
    social_data: Dict[str, Any] = Field(default_factory=dict)
    # G1 - Governance
    governance_data: Dict[str, Any] = Field(default_factory=dict)
    # E4 - Biodiversity (optional for manufacturing)
    biodiversity_data: Dict[str, Any] = Field(default_factory=dict)
    materiality_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"impact": 3.0, "financial": 3.0},
    )
    skip_phases: List[str] = Field(default_factory=list)

class ESRSManufacturingResult(WorkflowResult):
    """Result from the ESRS manufacturing workflow."""
    e1_metrics: Dict[str, Any] = Field(default_factory=dict)
    e2_metrics: Dict[str, Any] = Field(default_factory=dict)
    e3_metrics: Dict[str, Any] = Field(default_factory=dict)
    e4_metrics: Dict[str, Any] = Field(default_factory=dict)
    e5_metrics: Dict[str, Any] = Field(default_factory=dict)
    s1_metrics: Dict[str, Any] = Field(default_factory=dict)
    g1_metrics: Dict[str, Any] = Field(default_factory=dict)
    materiality_assessment: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
#  Phase 1: Materiality Assessment
# ---------------------------------------------------------------------------

class MaterialityAssessmentPhase:
    """Double materiality assessment for manufacturing context."""

    PHASE_NAME = "materiality_assessment"

    # Default manufacturing materiality scores (1-5)
    MANUFACTURING_DEFAULTS = {
        "E1": {"impact": 5.0, "financial": 4.5, "label": "Climate Change"},
        "E2": {"impact": 4.0, "financial": 3.5, "label": "Pollution"},
        "E3": {"impact": 3.5, "financial": 3.0, "label": "Water & Marine Resources"},
        "E4": {"impact": 2.5, "financial": 2.0, "label": "Biodiversity & Ecosystems"},
        "E5": {"impact": 4.0, "financial": 3.5, "label": "Resource Use & Circular Economy"},
        "S1": {"impact": 4.0, "financial": 3.0, "label": "Own Workforce"},
        "G1": {"impact": 3.5, "financial": 3.5, "label": "Business Conduct"},
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Assess double materiality for each ESRS topical standard."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            thresholds = config.get("materiality_thresholds", {"impact": 3.0, "financial": 3.0})
            impact_threshold = thresholds.get("impact", 3.0)
            financial_threshold = thresholds.get("financial", 3.0)

            material_topics: List[Dict[str, Any]] = []
            non_material: List[str] = []

            for topic_id, defaults in self.MANUFACTURING_DEFAULTS.items():
                impact = defaults["impact"]
                financial = defaults["financial"]
                is_material = impact >= impact_threshold or financial >= financial_threshold

                entry = {
                    "topic_id": topic_id,
                    "label": defaults["label"],
                    "impact_score": impact,
                    "financial_score": financial,
                    "is_material": is_material,
                }

                if is_material:
                    material_topics.append(entry)
                else:
                    non_material.append(topic_id)

            outputs["material_topics"] = material_topics
            outputs["non_material_topics"] = non_material
            outputs["material_count"] = len(material_topics)
            outputs["total_topics_assessed"] = len(self.MANUFACTURING_DEFAULTS)
            outputs["thresholds"] = thresholds

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("MaterialityAssessment failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 2: Data Point Collection
# ---------------------------------------------------------------------------

class DataPointCollectionPhase:
    """Map and validate required ESRS data points."""

    PHASE_NAME = "data_point_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Collect data points for all material ESRS topics."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            mat = context.get_phase_output("materiality_assessment")
            material_topics = mat.get("material_topics", [])
            material_ids = {t["topic_id"] for t in material_topics}

            data_map = {
                "E1": config.get("emissions_data", {}),
                "E2": config.get("energy_data", {}),
                "E3": config.get("water_data", {}),
                "E4": config.get("biodiversity_data", {}),
                "E5": config.get("waste_data", {}),
                "S1": config.get("social_data", {}),
                "G1": config.get("governance_data", {}),
            }

            collected: Dict[str, Dict[str, Any]] = {}
            coverage_stats: Dict[str, float] = {}

            for topic_id in material_ids:
                data = data_map.get(topic_id, {})
                if data:
                    collected[topic_id] = data
                    # Simple coverage: count non-empty fields
                    fields = len(data)
                    coverage_stats[topic_id] = min(round(fields / max(fields, 1) * 100, 1), 100.0)
                else:
                    warnings.append(f"No data provided for material topic {topic_id}")
                    coverage_stats[topic_id] = 0.0

            outputs["collected_topics"] = list(collected.keys())
            outputs["data_by_topic"] = collected
            outputs["coverage_stats"] = coverage_stats
            outputs["overall_coverage_pct"] = round(
                sum(coverage_stats.values()) / max(len(coverage_stats), 1), 1
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("DataPointCollection failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 3: Disclosure Generation
# ---------------------------------------------------------------------------

class DisclosureGenerationPhase:
    """Generate ESRS metric values per topical standard."""

    PHASE_NAME = "disclosure_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Compute ESRS disclosure metrics for each material topic."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            dp = context.get_phase_output("data_point_collection")
            data_by_topic = dp.get("data_by_topic", {})

            # E1 - Climate
            e1_data = data_by_topic.get("E1", {})
            e1_metrics = {
                "scope1_tco2e": e1_data.get("scope1_total", 0.0),
                "scope2_tco2e": e1_data.get("scope2_total", 0.0),
                "scope3_tco2e": e1_data.get("scope3_total", 0.0),
                "total_tco2e": (
                    e1_data.get("scope1_total", 0.0)
                    + e1_data.get("scope2_total", 0.0)
                    + e1_data.get("scope3_total", 0.0)
                ),
                "intensity_tco2e_per_revenue": e1_data.get("intensity", 0.0),
                "reduction_target_pct": e1_data.get("reduction_target_pct", 0.0),
                "transition_plan": e1_data.get("has_transition_plan", False),
            }
            outputs["e1_metrics"] = e1_metrics

            # E2 - Pollution
            e2_data = data_by_topic.get("E2", {})
            outputs["e2_metrics"] = {
                "pollutant_emissions_tonnes": e2_data.get("total_pollutant_emissions", 0.0),
                "nox_tonnes": e2_data.get("nox", 0.0),
                "sox_tonnes": e2_data.get("sox", 0.0),
                "pm_tonnes": e2_data.get("pm", 0.0),
                "voc_tonnes": e2_data.get("voc", 0.0),
                "svhc_substances": e2_data.get("svhc_count", 0),
            }

            # E3 - Water
            e3_data = data_by_topic.get("E3", {})
            outputs["e3_metrics"] = {
                "total_water_withdrawal_m3": e3_data.get("withdrawal_m3", 0.0),
                "total_water_discharge_m3": e3_data.get("discharge_m3", 0.0),
                "water_consumption_m3": e3_data.get("consumption_m3", 0.0),
                "water_recycled_pct": e3_data.get("recycled_pct", 0.0),
                "water_stress_areas": e3_data.get("in_water_stress_area", False),
            }

            # E4 - Biodiversity
            e4_data = data_by_topic.get("E4", {})
            outputs["e4_metrics"] = {
                "sites_near_protected_areas": e4_data.get("sites_near_protected", 0),
                "land_use_change_ha": e4_data.get("land_use_change_ha", 0.0),
                "biodiversity_action_plan": e4_data.get("has_action_plan", False),
            }

            # E5 - Circular Economy
            e5_data = data_by_topic.get("E5", {})
            outputs["e5_metrics"] = {
                "total_waste_tonnes": e5_data.get("total_waste_tonnes", 0.0),
                "hazardous_waste_tonnes": e5_data.get("hazardous_waste_tonnes", 0.0),
                "waste_diversion_pct": e5_data.get("diversion_rate_pct", 0.0),
                "recycled_content_pct": e5_data.get("recycled_content_pct", 0.0),
                "mci_score": e5_data.get("mci_score", 0.0),
            }

            # S1 - Own Workforce
            s1_data = data_by_topic.get("S1", {})
            outputs["s1_metrics"] = {
                "total_employees": s1_data.get("total_employees", 0),
                "gender_pay_gap_pct": s1_data.get("gender_pay_gap_pct", 0.0),
                "training_hours_per_employee": s1_data.get("training_hours", 0.0),
                "health_safety_incidents": s1_data.get("incidents", 0),
                "fatalities": s1_data.get("fatalities", 0),
                "ltir": s1_data.get("ltir", 0.0),
            }

            # G1 - Governance
            g1_data = data_by_topic.get("G1", {})
            outputs["g1_metrics"] = {
                "anti_corruption_training_pct": g1_data.get("anti_corruption_training_pct", 0.0),
                "whistleblower_cases": g1_data.get("whistleblower_cases", 0),
                "political_contributions_eur": g1_data.get("political_contributions", 0.0),
                "supplier_code_of_conduct": g1_data.get("has_supplier_code", False),
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("DisclosureGeneration failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 4: Audit Preparation
# ---------------------------------------------------------------------------

class AuditPreparationPhase:
    """Prepare documentation for limited assurance readiness."""

    PHASE_NAME = "audit_preparation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Assess audit readiness and generate evidence checklist."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            disc = context.get_phase_output("disclosure_generation")
            dp = context.get_phase_output("data_point_collection")
            mat = context.get_phase_output("materiality_assessment")

            material_topics = mat.get("material_topics", [])
            coverage = dp.get("coverage_stats", {})

            checklist: List[Dict[str, Any]] = []
            ready_count = 0

            for topic in material_topics:
                tid = topic["topic_id"]
                cov = coverage.get(tid, 0.0)
                has_data = cov > 0
                metrics_key = f"{tid.lower()}_metrics"
                has_metrics = bool(disc.get(metrics_key, {}))

                ready = has_data and has_metrics
                if ready:
                    ready_count += 1

                checklist.append({
                    "topic_id": tid,
                    "label": topic["label"],
                    "data_available": has_data,
                    "metrics_computed": has_metrics,
                    "audit_ready": ready,
                    "evidence_needed": [] if ready else [
                        "source_documentation",
                        "calculation_methodology",
                        "internal_controls",
                    ],
                })

            total_material = len(material_topics)
            readiness_pct = round(ready_count / max(total_material, 1) * 100, 1)

            outputs["audit_checklist"] = checklist
            outputs["readiness_pct"] = readiness_pct
            outputs["topics_ready"] = ready_count
            outputs["topics_total"] = total_material
            outputs["assurance_level"] = "limited"
            outputs["recommended_actions"] = []

            if readiness_pct < 100:
                not_ready = [c["topic_id"] for c in checklist if not c["audit_ready"]]
                outputs["recommended_actions"].append(
                    f"Complete data collection for: {', '.join(not_ready)}"
                )
                warnings.append(f"Audit readiness at {readiness_pct}%; gaps in {', '.join(not_ready)}")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("AuditPreparation failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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

class ESRSManufacturingWorkflow:
    """
    Four-phase ESRS manufacturing disclosure workflow.

    Generates ESRS-compliant disclosures for manufacturing organizations
    covering E1-E5, S1, and G1 topical standards with double materiality
    assessment and audit preparation.
    """

    WORKFLOW_NAME = "esrs_manufacturing"

    PHASE_ORDER = [
        "materiality_assessment", "data_point_collection",
        "disclosure_generation", "audit_preparation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize ESRSManufacturingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "materiality_assessment": MaterialityAssessmentPhase(),
            "data_point_collection": DataPointCollectionPhase(),
            "disclosure_generation": DisclosureGenerationPhase(),
            "audit_preparation": AuditPreparationPhase(),
        }

    async def run(self, input_data: ESRSManufacturingInput) -> ESRSManufacturingResult:
        """Execute the complete 4-phase ESRS manufacturing workflow."""
        started_at = utcnow()
        logger.info("Starting ESRS manufacturing workflow %s", self.workflow_id)
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
                    started_at=utcnow(), errors=[str(exc)],
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

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return ESRSManufacturingResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            e1_metrics=summary.get("e1_metrics", {}),
            e2_metrics=summary.get("e2_metrics", {}),
            e3_metrics=summary.get("e3_metrics", {}),
            e4_metrics=summary.get("e4_metrics", {}),
            e5_metrics=summary.get("e5_metrics", {}),
            s1_metrics=summary.get("s1_metrics", {}),
            g1_metrics=summary.get("g1_metrics", {}),
            materiality_assessment=summary.get("materiality_assessment", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        disc = context.get_phase_output("disclosure_generation")
        mat = context.get_phase_output("materiality_assessment")
        return {
            "e1_metrics": disc.get("e1_metrics", {}),
            "e2_metrics": disc.get("e2_metrics", {}),
            "e3_metrics": disc.get("e3_metrics", {}),
            "e4_metrics": disc.get("e4_metrics", {}),
            "e5_metrics": disc.get("e5_metrics", {}),
            "s1_metrics": disc.get("s1_metrics", {}),
            "g1_metrics": disc.get("g1_metrics", {}),
            "materiality_assessment": {
                "material_topics": mat.get("material_topics", []),
                "material_count": mat.get("material_count", 0),
            },
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
