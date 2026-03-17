# -*- coding: utf-8 -*-
"""
Financial Services Materiality Workflow
==========================================

Four-phase workflow for conducting double materiality assessment specific to
financial institutions under CSRD/ESRS. Addresses FI-specific impacts including
financed emissions, financial inclusion, responsible lending, and systemic risk.

Phases:
    1. StakeholderEngagement - Identify and engage FI-specific stakeholders
    2. ImpactAssessment - Assess impact materiality for FI topics
    3. FinancialAssessment - Assess financial materiality (risks/opportunities)
    4. MatrixGeneration - Generate double materiality matrix and thresholds

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

class StakeholderGroup(BaseModel):
    """Stakeholder group for materiality assessment."""
    group_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Stakeholder group name")
    category: str = Field(default="external", description="internal, external, regulatory")
    relevance_score: float = Field(default=5.0, ge=0.0, le=10.0)
    topics_raised: List[str] = Field(default_factory=list)


class MaterialityTopic(BaseModel):
    """A sustainability topic to assess for materiality."""
    topic_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Topic name")
    esrs_reference: str = Field(default="", description="ESRS reference e.g. E1, S1")
    fi_specific: bool = Field(default=False, description="Financial institution specific")
    impact_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    financial_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    stakeholder_priority: float = Field(default=5.0, ge=0.0, le=10.0)


class FSMaterialityInput(BaseModel):
    """Input for the FS materiality workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_period: str = Field(..., description="Reporting period YYYY")
    institution_type: str = Field(default="credit_institution",
                                   description="credit_institution, insurance, asset_manager")
    stakeholder_groups: List[StakeholderGroup] = Field(default_factory=list)
    topics: List[MaterialityTopic] = Field(default_factory=list)
    impact_threshold: float = Field(default=5.0, ge=0.0, le=10.0)
    financial_threshold: float = Field(default=5.0, ge=0.0, le=10.0)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        int(v)
        return v


class FSMaterialityResult(WorkflowResult):
    """Result from the FS materiality workflow."""
    material_topics_count: int = Field(default=0)
    impact_material_count: int = Field(default=0)
    financial_material_count: int = Field(default=0)
    double_material_count: int = Field(default=0)
    fi_specific_material_count: int = Field(default=0)
    stakeholder_groups_engaged: int = Field(default=0)
    topics_assessed: int = Field(default=0)


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------

class StakeholderEngagementPhase:
    """Identify and engage FI-specific stakeholders (depositors, borrowers, regulators)."""
    PHASE_NAME = "stakeholder_engagement"

    FI_DEFAULT_STAKEHOLDERS = [
        {"name": "Depositors/Account Holders", "category": "external", "relevance_score": 8.0},
        {"name": "Borrowers/Counterparties", "category": "external", "relevance_score": 8.0},
        {"name": "Regulators (ECB/NCA)", "category": "regulatory", "relevance_score": 9.5},
        {"name": "Investors/Shareholders", "category": "external", "relevance_score": 9.0},
        {"name": "Employees", "category": "internal", "relevance_score": 7.0},
        {"name": "Civil Society/NGOs", "category": "external", "relevance_score": 6.0},
        {"name": "Portfolio Companies", "category": "external", "relevance_score": 7.5},
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            groups = config.get("stakeholder_groups", [])
            if not groups:
                groups = self.FI_DEFAULT_STAKEHOLDERS
                warnings.append("Using default FI stakeholder groups (no custom groups provided)")

            outputs["stakeholder_groups"] = groups
            outputs["stakeholder_count"] = len(groups)
            outputs["categories"] = list(set(g.get("category", "") for g in groups))
            avg_relevance = sum(g.get("relevance_score", 5.0) for g in groups) / max(len(groups), 1)
            outputs["avg_relevance_score"] = round(avg_relevance, 2)
            all_topics = []
            for g in groups:
                all_topics.extend(g.get("topics_raised", []))
            outputs["unique_topics_raised"] = list(set(all_topics))
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("StakeholderEngagement failed: %s", exc, exc_info=True)
            errors.append(f"Stakeholder engagement failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class ImpactAssessmentPhase:
    """Assess impact materiality for FI-specific sustainability topics."""
    PHASE_NAME = "impact_assessment"

    FI_DEFAULT_TOPICS = [
        {"name": "Financed GHG Emissions", "esrs_reference": "E1", "fi_specific": True, "impact_score": 9.0},
        {"name": "Climate Risk (Physical)", "esrs_reference": "E1", "fi_specific": True, "impact_score": 8.0},
        {"name": "Climate Risk (Transition)", "esrs_reference": "E1", "fi_specific": True, "impact_score": 8.5},
        {"name": "Biodiversity via Financing", "esrs_reference": "E4", "fi_specific": True, "impact_score": 6.5},
        {"name": "Financial Inclusion", "esrs_reference": "S1", "fi_specific": True, "impact_score": 7.0},
        {"name": "Responsible Lending", "esrs_reference": "S4", "fi_specific": True, "impact_score": 7.5},
        {"name": "Data Privacy", "esrs_reference": "S4", "fi_specific": False, "impact_score": 6.0},
        {"name": "Anti-Money Laundering", "esrs_reference": "G1", "fi_specific": True, "impact_score": 8.0},
        {"name": "Board ESG Governance", "esrs_reference": "G1", "fi_specific": False, "impact_score": 7.0},
        {"name": "Remuneration ESG Linkage", "esrs_reference": "G1", "fi_specific": True, "impact_score": 6.5},
        {"name": "Water via Financing", "esrs_reference": "E3", "fi_specific": True, "impact_score": 5.5},
        {"name": "Circular Economy via Financing", "esrs_reference": "E5", "fi_specific": True, "impact_score": 5.0},
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            topics = config.get("topics", [])
            if not topics:
                topics = self.FI_DEFAULT_TOPICS
                warnings.append("Using default FI materiality topics")

            threshold = config.get("impact_threshold", 5.0)
            assessed = []
            for t in topics:
                score = t.get("impact_score", 5.0) if t.get("impact_score") is not None else 5.0
                is_material = score >= threshold
                assessed.append({
                    **t,
                    "impact_score": score,
                    "impact_material": is_material,
                })

            outputs["impact_assessed"] = assessed
            outputs["impact_material_count"] = sum(1 for a in assessed if a["impact_material"])
            outputs["total_assessed"] = len(assessed)
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("ImpactAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Impact assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class FinancialAssessmentPhase:
    """Assess financial materiality (risks and opportunities) for FI topics."""
    PHASE_NAME = "financial_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            impact = context.get_phase_output("impact_assessment")
            assessed = impact.get("impact_assessed", [])
            threshold = config.get("financial_threshold", 5.0)

            financial_assessed = []
            for t in assessed:
                fin_score = t.get("financial_score")
                if fin_score is None:
                    # Estimate: FI-specific topics tend to have higher financial materiality
                    base = t.get("impact_score", 5.0) * 0.8
                    if t.get("fi_specific"):
                        base += 1.5
                    fin_score = min(base, 10.0)
                fin_material = fin_score >= threshold
                financial_assessed.append({
                    **t,
                    "financial_score": round(fin_score, 2),
                    "financial_material": fin_material,
                })

            outputs["financial_assessed"] = financial_assessed
            outputs["financial_material_count"] = sum(1 for a in financial_assessed if a["financial_material"])
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("FinancialAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Financial assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class MatrixGenerationPhase:
    """Generate double materiality matrix with material/non-material classification."""
    PHASE_NAME = "matrix_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            financial = context.get_phase_output("financial_assessment")
            topics = financial.get("financial_assessed", [])

            matrix = []
            double_material = 0
            impact_only = 0
            financial_only = 0
            not_material = 0
            fi_specific_material = 0

            for t in topics:
                im = t.get("impact_material", False)
                fm = t.get("financial_material", False)
                if im and fm:
                    classification = "DOUBLE_MATERIAL"
                    double_material += 1
                elif im:
                    classification = "IMPACT_MATERIAL_ONLY"
                    impact_only += 1
                elif fm:
                    classification = "FINANCIAL_MATERIAL_ONLY"
                    financial_only += 1
                else:
                    classification = "NOT_MATERIAL"
                    not_material += 1

                is_material = im or fm
                if is_material and t.get("fi_specific"):
                    fi_specific_material += 1

                matrix.append({
                    "name": t.get("name", ""),
                    "esrs_reference": t.get("esrs_reference", ""),
                    "fi_specific": t.get("fi_specific", False),
                    "impact_score": t.get("impact_score", 0.0),
                    "financial_score": t.get("financial_score", 0.0),
                    "impact_material": im,
                    "financial_material": fm,
                    "classification": classification,
                    "is_material": is_material,
                })

            outputs["materiality_matrix"] = matrix
            outputs["double_material_count"] = double_material
            outputs["impact_only_count"] = impact_only
            outputs["financial_only_count"] = financial_only
            outputs["not_material_count"] = not_material
            outputs["total_material"] = double_material + impact_only + financial_only
            outputs["fi_specific_material_count"] = fi_specific_material
            outputs["topics_assessed"] = len(topics)
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("MatrixGeneration failed: %s", exc, exc_info=True)
            errors.append(f"Matrix generation failed: {str(exc)}")
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

class FSMaterialityWorkflow:
    """Four-phase FS double materiality workflow for CSRD financial institutions."""

    WORKFLOW_NAME = "fs_materiality"
    PHASE_ORDER = ["stakeholder_engagement", "impact_assessment",
                    "financial_assessment", "matrix_generation"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "stakeholder_engagement": StakeholderEngagementPhase(),
            "impact_assessment": ImpactAssessmentPhase(),
            "financial_assessment": FinancialAssessmentPhase(),
            "matrix_generation": MatrixGenerationPhase(),
        }

    async def run(self, input_data: FSMaterialityInput) -> FSMaterialityResult:
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

        return FSMaterialityResult(
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
        engagement = context.get_phase_output("stakeholder_engagement")
        matrix = context.get_phase_output("matrix_generation")
        impact = context.get_phase_output("impact_assessment")
        financial = context.get_phase_output("financial_assessment")
        return {
            "material_topics_count": matrix.get("total_material", 0),
            "impact_material_count": impact.get("impact_material_count", 0),
            "financial_material_count": financial.get("financial_material_count", 0),
            "double_material_count": matrix.get("double_material_count", 0),
            "fi_specific_material_count": matrix.get("fi_specific_material_count", 0),
            "stakeholder_groups_engaged": engagement.get("stakeholder_count", 0),
            "topics_assessed": matrix.get("topics_assessed", 0),
        }

    @staticmethod
    def _result_defaults() -> Dict[str, Any]:
        return {
            "material_topics_count": 0, "impact_material_count": 0,
            "financial_material_count": 0, "double_material_count": 0,
            "fi_specific_material_count": 0, "stakeholder_groups_engaged": 0,
            "topics_assessed": 0,
        }
