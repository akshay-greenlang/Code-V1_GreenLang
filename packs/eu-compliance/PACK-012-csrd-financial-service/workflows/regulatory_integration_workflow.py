# -*- coding: utf-8 -*-
"""
Regulatory Integration Workflow
===================================

Three-phase workflow for mapping and aligning cross-regulatory requirements
for financial institutions: CSRD, Pillar 3 ESG, SFDR, EU Taxonomy, TCFD,
ECB Climate Guide, EBA GL on ESG risks.

Phases:
    1. RequirementMapping - Map all applicable regulatory requirements
    2. CrossReferenceAlignment - Identify overlaps and cross-references
    3. GapAnalysis - Identify coverage gaps and remediation priorities

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



# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class RegulatoryRequirement(BaseModel):
    """A single regulatory requirement."""
    requirement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    regulation: str = Field(..., description="CSRD, Pillar3, SFDR, Taxonomy, TCFD, etc.")
    reference: str = Field(default="", description="Article/paragraph reference")
    description: str = Field(default="")
    topic: str = Field(default="", description="E1, S1, G1, etc.")
    applicable: bool = Field(default=True)
    covered: bool = Field(default=False)
    data_source: str = Field(default="")


class RegulatoryIntegrationInput(BaseModel):
    """Input for the regulatory integration workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_period: str = Field(default="2025")
    institution_type: str = Field(default="credit_institution")
    applicable_regulations: List[str] = Field(
        default_factory=lambda: ["CSRD", "Pillar3_ESG", "SFDR", "EU_Taxonomy", "TCFD"]
    )
    requirements: List[RegulatoryRequirement] = Field(default_factory=list)
    existing_coverage: Dict[str, bool] = Field(default_factory=dict)
    skip_phases: List[str] = Field(default_factory=list)


class RegulatoryIntegrationResult(WorkflowResult):
    """Result from the regulatory integration workflow."""
    total_requirements: int = Field(default=0)
    covered_requirements: int = Field(default=0)
    gap_count: int = Field(default=0)
    coverage_pct: float = Field(default=0.0)
    cross_references_found: int = Field(default=0)
    regulations_mapped: int = Field(default=0)
    high_priority_gaps: int = Field(default=0)


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------

class RequirementMappingPhase:
    """Map all applicable regulatory requirements."""
    PHASE_NAME = "requirement_mapping"

    DEFAULT_REQUIREMENTS = {
        "CSRD": [
            {"reference": "ESRS E1", "description": "Climate change (financed emissions)", "topic": "E1"},
            {"reference": "ESRS E1-6", "description": "Scope 3 Category 15", "topic": "E1"},
            {"reference": "ESRS E1-9", "description": "Transition plan", "topic": "E1"},
            {"reference": "ESRS S1", "description": "Own workforce", "topic": "S1"},
            {"reference": "ESRS S4", "description": "Consumers and end-users", "topic": "S4"},
            {"reference": "ESRS G1", "description": "Business conduct", "topic": "G1"},
        ],
        "Pillar3_ESG": [
            {"reference": "Template 1", "description": "Banking book climate transition risk", "topic": "E1"},
            {"reference": "Template 2", "description": "Banking book physical risk", "topic": "E1"},
            {"reference": "Template 4", "description": "Alignment metrics (GAR/BTAR)", "topic": "E1"},
            {"reference": "Template 5", "description": "Top 20 carbon-intensive exposures", "topic": "E1"},
            {"reference": "Template 8", "description": "Qualitative ESG disclosures", "topic": "G1"},
        ],
        "SFDR": [
            {"reference": "Art 4", "description": "PAI statement", "topic": "E1"},
            {"reference": "Art 7", "description": "Product-level PAI consideration", "topic": "E1"},
            {"reference": "Annex I", "description": "PAI indicators", "topic": "E1"},
        ],
        "EU_Taxonomy": [
            {"reference": "Art 8 DA", "description": "Green Asset Ratio disclosure", "topic": "E1"},
            {"reference": "Art 8 DA", "description": "BTAR disclosure", "topic": "E1"},
        ],
        "TCFD": [
            {"reference": "Governance", "description": "Board climate oversight", "topic": "G1"},
            {"reference": "Strategy", "description": "Climate scenario analysis", "topic": "E1"},
            {"reference": "Risk Mgmt", "description": "Climate risk integration", "topic": "E1"},
            {"reference": "Metrics", "description": "GHG emissions and targets", "topic": "E1"},
        ],
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            custom_reqs = config.get("requirements", [])
            applicable = config.get("applicable_regulations", [])
            coverage = config.get("existing_coverage", {})

            all_reqs = []
            if custom_reqs:
                all_reqs = custom_reqs
            else:
                for reg in applicable:
                    defaults = self.DEFAULT_REQUIREMENTS.get(reg, [])
                    for d in defaults:
                        all_reqs.append({
                            "regulation": reg, **d,
                            "applicable": True,
                            "covered": coverage.get(f"{reg}_{d.get('reference', '')}", False),
                        })

            outputs["requirements"] = all_reqs
            outputs["total_requirements"] = len(all_reqs)
            outputs["regulations_mapped"] = len(set(r.get("regulation", "") for r in all_reqs))
            outputs["covered_count"] = sum(1 for r in all_reqs if r.get("covered"))

            by_reg: Dict[str, int] = {}
            for r in all_reqs:
                reg = r.get("regulation", "OTHER")
                by_reg[reg] = by_reg.get(reg, 0) + 1
            outputs["requirements_by_regulation"] = by_reg

            status = PhaseStatus.COMPLETED
            records = len(all_reqs)
        except Exception as exc:
            logger.error("RequirementMapping failed: %s", exc, exc_info=True)
            errors.append(f"Requirement mapping failed: {str(exc)}")
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


class CrossReferenceAlignmentPhase:
    """Identify overlaps and cross-references between regulations."""
    PHASE_NAME = "cross_reference_alignment"

    KNOWN_CROSS_REFS = [
        {"from_reg": "CSRD", "from_ref": "ESRS E1-6", "to_reg": "Pillar3_ESG",
         "to_ref": "Template 4", "overlap": "Financed emissions / GAR alignment"},
        {"from_reg": "CSRD", "from_ref": "ESRS E1-9", "to_reg": "TCFD",
         "to_ref": "Strategy", "overlap": "Transition plan / scenario analysis"},
        {"from_reg": "Pillar3_ESG", "from_ref": "Template 4", "to_reg": "EU_Taxonomy",
         "to_ref": "Art 8 DA", "overlap": "GAR/BTAR computation"},
        {"from_reg": "CSRD", "from_ref": "ESRS E1", "to_reg": "SFDR",
         "to_ref": "Art 4", "overlap": "PAI / financed emissions"},
        {"from_reg": "CSRD", "from_ref": "ESRS G1", "to_reg": "TCFD",
         "to_ref": "Governance", "overlap": "Board ESG governance"},
        {"from_reg": "SFDR", "from_ref": "Annex I", "to_reg": "CSRD",
         "to_ref": "ESRS E1", "overlap": "PAI indicators / climate metrics"},
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            mapping = context.get_phase_output("requirement_mapping")
            requirements = mapping.get("requirements", [])
            applicable_regs = set(r.get("regulation", "") for r in requirements)

            relevant_xrefs = [
                xr for xr in self.KNOWN_CROSS_REFS
                if xr["from_reg"] in applicable_regs and xr["to_reg"] in applicable_regs
            ]

            # Group by topic
            by_topic: Dict[str, List[Dict[str, Any]]] = {}
            for r in requirements:
                topic = r.get("topic", "OTHER")
                by_topic.setdefault(topic, []).append(r)

            topic_overlaps = {}
            for topic, reqs in by_topic.items():
                regs_in_topic = set(r.get("regulation", "") for r in reqs)
                if len(regs_in_topic) > 1:
                    topic_overlaps[topic] = list(regs_in_topic)

            outputs["cross_references"] = relevant_xrefs
            outputs["cross_references_count"] = len(relevant_xrefs)
            outputs["topic_overlaps"] = topic_overlaps
            outputs["topics_with_multi_reg"] = len(topic_overlaps)

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("CrossReferenceAlignment failed: %s", exc, exc_info=True)
            errors.append(f"Cross-reference alignment failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class GapAnalysisPhase:
    """Identify coverage gaps and remediation priorities."""
    PHASE_NAME = "gap_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            mapping = context.get_phase_output("requirement_mapping")
            xref = context.get_phase_output("cross_reference_alignment")
            requirements = mapping.get("requirements", [])
            total = len(requirements)
            covered = sum(1 for r in requirements if r.get("covered"))

            gaps = [r for r in requirements if not r.get("covered")]
            coverage_pct = round(covered / max(total, 1) * 100, 2)

            # Prioritize gaps
            high_priority = []
            medium_priority = []
            low_priority = []

            for g in gaps:
                reg = g.get("regulation", "")
                if reg in ("CSRD", "Pillar3_ESG"):
                    high_priority.append(g)
                elif reg in ("EU_Taxonomy", "SFDR"):
                    medium_priority.append(g)
                else:
                    low_priority.append(g)

            outputs["gaps"] = gaps
            outputs["gap_count"] = len(gaps)
            outputs["coverage_pct"] = coverage_pct
            outputs["covered_count"] = covered
            outputs["total_requirements"] = total
            outputs["high_priority_gaps"] = len(high_priority)
            outputs["medium_priority_gaps"] = len(medium_priority)
            outputs["low_priority_gaps"] = len(low_priority)
            outputs["high_priority_list"] = high_priority
            outputs["remediation_roadmap"] = {
                "immediate": [g.get("reference", "") for g in high_priority[:5]],
                "short_term": [g.get("reference", "") for g in medium_priority[:5]],
                "medium_term": [g.get("reference", "") for g in low_priority[:5]],
            }

            if coverage_pct < 50:
                warnings.append(f"Regulatory coverage below 50% ({coverage_pct}%)")

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("GapAnalysis failed: %s", exc, exc_info=True)
            errors.append(f"Gap analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class RegulatoryIntegrationWorkflow:
    """Three-phase regulatory integration workflow for FI compliance mapping."""

    WORKFLOW_NAME = "regulatory_integration"
    PHASE_ORDER = ["requirement_mapping", "cross_reference_alignment", "gap_analysis"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "requirement_mapping": RequirementMappingPhase(),
            "cross_reference_alignment": CrossReferenceAlignmentPhase(),
            "gap_analysis": GapAnalysisPhase(),
        }

    async def run(self, input_data: RegulatoryIntegrationInput) -> RegulatoryIntegrationResult:
        """Execute the workflow."""
        started_at = _utcnow()
        logger.info("Starting %s workflow %s org=%s", self.WORKFLOW_NAME,
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
                    if phase_name == self.PHASE_ORDER[0]:
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

        return RegulatoryIntegrationResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            **{k: summary.get(k, v) for k, v in self._result_defaults().items()}
        )

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        mapping = context.get_phase_output("requirement_mapping")
        xref = context.get_phase_output("cross_reference_alignment")
        gaps = context.get_phase_output("gap_analysis")
        return {
            "total_requirements": mapping.get("total_requirements", 0),
            "covered_requirements": gaps.get("covered_count", 0),
            "gap_count": gaps.get("gap_count", 0),
            "coverage_pct": gaps.get("coverage_pct", 0.0),
            "cross_references_found": xref.get("cross_references_count", 0),
            "regulations_mapped": mapping.get("regulations_mapped", 0),
            "high_priority_gaps": gaps.get("high_priority_gaps", 0),
        }

    @staticmethod
    def _result_defaults() -> Dict[str, Any]:
        return {
            "total_requirements": 0, "covered_requirements": 0,
            "gap_count": 0, "coverage_pct": 0.0,
            "cross_references_found": 0, "regulations_mapped": 0,
            "high_priority_gaps": 0,
        }
