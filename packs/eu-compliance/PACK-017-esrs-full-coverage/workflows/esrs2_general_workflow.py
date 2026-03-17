# -*- coding: utf-8 -*-
"""
ESRS 2 General Disclosures Workflow
====================================

5-phase workflow for ESRS 2 General Disclosures covering governance, strategy,
impact/risk/opportunity identification, minimum disclosure requirement validation,
and report generation with full provenance tracking.

Phases:
    1. GovernanceAssessment   -- Evaluate GOV-1 through GOV-5
    2. StrategyAnalysis       -- Assess SBM-1, SBM-2, SBM-3
    3. IROIdentification      -- Process IRO-1, IRO-2
    4. MDRValidation          -- Validate MDR-P, MDR-A, MDR-T, MDR-M
    5. ReportGeneration       -- Compile ESRS 2 disclosure

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the ESRS 2 general disclosures workflow."""
    GOVERNANCE_ASSESSMENT = "governance_assessment"
    STRATEGY_ANALYSIS = "strategy_analysis"
    IRO_IDENTIFICATION = "iro_identification"
    MDR_VALIDATION = "mdr_validation"
    REPORT_GENERATION = "report_generation"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GovernanceArea(str, Enum):
    """ESRS 2 GOV disclosure areas."""
    GOV_1 = "gov_1_board_role"
    GOV_2 = "gov_2_board_sustainability_info"
    GOV_3 = "gov_3_integration_performance"
    GOV_4 = "gov_4_due_diligence_statement"
    GOV_5 = "gov_5_risk_management"


class StrategyArea(str, Enum):
    """ESRS 2 SBM disclosure areas."""
    SBM_1 = "sbm_1_strategy_business_model"
    SBM_2 = "sbm_2_stakeholder_interests"
    SBM_3 = "sbm_3_material_impacts"


class IROArea(str, Enum):
    """ESRS 2 IRO disclosure areas."""
    IRO_1 = "iro_1_materiality_process"
    IRO_2 = "iro_2_esrs_disclosure_requirements"


class MDRType(str, Enum):
    """Minimum Disclosure Requirement types."""
    MDR_P = "mdr_p_policies"
    MDR_A = "mdr_a_actions"
    MDR_T = "mdr_t_targets"
    MDR_M = "mdr_m_metrics"


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


class GovernanceRecord(BaseModel):
    """Governance assessment data for GOV-1 to GOV-5."""
    record_id: str = Field(default_factory=lambda: f"gov-{_new_uuid()[:8]}")
    area: GovernanceArea = Field(..., description="GOV disclosure area")
    description: str = Field(default="", description="Assessment description")
    compliance_status: str = Field(default="not_assessed")
    evidence_references: List[str] = Field(default_factory=list)
    gaps_identified: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="Compliance score 0-100")


class StrategyRecord(BaseModel):
    """Strategy and business model assessment data."""
    record_id: str = Field(default_factory=lambda: f"sbm-{_new_uuid()[:8]}")
    area: StrategyArea = Field(..., description="SBM disclosure area")
    description: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    stakeholder_groups: List[str] = Field(default_factory=list)
    material_impacts: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class IRORecord(BaseModel):
    """Impact, risk, and opportunity identification record."""
    record_id: str = Field(default_factory=lambda: f"iro-{_new_uuid()[:8]}")
    area: IROArea = Field(..., description="IRO disclosure area")
    description: str = Field(default="")
    impacts_identified: int = Field(default=0, ge=0)
    risks_identified: int = Field(default=0, ge=0)
    opportunities_identified: int = Field(default=0, ge=0)
    materiality_topics: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class MDRRecord(BaseModel):
    """Minimum Disclosure Requirement validation record."""
    record_id: str = Field(default_factory=lambda: f"mdr-{_new_uuid()[:8]}")
    mdr_type: MDRType = Field(..., description="MDR category")
    standard_ref: str = Field(default="", description="ESRS standard reference")
    is_disclosed: bool = Field(default=False)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps: List[str] = Field(default_factory=list)


class MaterialityResult(BaseModel):
    """External materiality assessment result."""
    topic: str = Field(default="")
    is_material: bool = Field(default=False)
    impact_score: float = Field(default=0.0)
    financial_score: float = Field(default=0.0)


class ESRS2GeneralInput(BaseModel):
    """Input data model for ESRS2GeneralWorkflow."""
    governance_data: List[GovernanceRecord] = Field(
        default_factory=list, description="Governance assessment records for GOV-1 to GOV-5"
    )
    strategy_data: List[StrategyRecord] = Field(
        default_factory=list, description="Strategy and business model records"
    )
    stakeholder_data: Dict[str, Any] = Field(
        default_factory=dict, description="Stakeholder engagement data"
    )
    materiality_results: List[MaterialityResult] = Field(
        default_factory=list, description="Double materiality assessment results"
    )
    iro_records: List[IRORecord] = Field(
        default_factory=list, description="IRO identification records"
    )
    mdr_records: List[MDRRecord] = Field(
        default_factory=list, description="MDR validation records"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class ESRS2WorkflowResult(BaseModel):
    """Complete result from ESRS 2 general disclosures workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="esrs2_general_disclosures")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    governance_scores: Dict[str, float] = Field(default_factory=dict)
    strategy_scores: Dict[str, float] = Field(default_factory=dict)
    iro_summary: Dict[str, Any] = Field(default_factory=dict)
    mdr_compliance: Dict[str, float] = Field(default_factory=dict)
    overall_compliance_pct: float = Field(default=0.0)
    total_gaps: int = Field(default=0)
    disclosure_items_assessed: int = Field(default=0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ESRS2GeneralWorkflow:
    """
    5-phase ESRS 2 General Disclosures workflow.

    Implements end-to-end assessment of ESRS 2 general disclosure requirements
    including governance (GOV-1 to GOV-5), strategy and business model
    (SBM-1 to SBM-3), impact/risk/opportunity identification (IRO-1, IRO-2),
    and minimum disclosure requirement validation (MDR-P, MDR-A, MDR-T, MDR-M).

    Zero-hallucination: all compliance scoring uses deterministic rules.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = ESRS2GeneralWorkflow()
        >>> inp = ESRS2GeneralInput(governance_data=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_compliance_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ESRS2GeneralWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._governance_scores: Dict[str, float] = {}
        self._strategy_scores: Dict[str, float] = {}
        self._iro_summary: Dict[str, Any] = {}
        self._mdr_compliance: Dict[str, float] = {}
        self._total_gaps: int = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.GOVERNANCE_ASSESSMENT.value, "description": "Evaluate GOV-1 through GOV-5"},
            {"name": WorkflowPhase.STRATEGY_ANALYSIS.value, "description": "Assess SBM-1, SBM-2, SBM-3"},
            {"name": WorkflowPhase.IRO_IDENTIFICATION.value, "description": "Process IRO-1, IRO-2"},
            {"name": WorkflowPhase.MDR_VALIDATION.value, "description": "Validate MDR-P, MDR-A, MDR-T, MDR-M"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Compile ESRS 2 disclosure"},
        ]

    def validate_inputs(self, input_data: ESRS2GeneralInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.governance_data:
            issues.append("No governance data provided for GOV assessment")
        if not input_data.strategy_data:
            issues.append("No strategy data provided for SBM assessment")
        if not input_data.materiality_results:
            issues.append("No materiality results provided; IRO assessment may be incomplete")
        gov_areas = {r.area for r in input_data.governance_data}
        expected_gov = set(GovernanceArea)
        missing_gov = expected_gov - gov_areas
        if missing_gov:
            issues.append(f"Missing governance areas: {', '.join(a.value for a in missing_gov)}")
        return issues

    async def execute(
        self,
        input_data: Optional[ESRS2GeneralInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ESRS2WorkflowResult:
        """
        Execute the 5-phase ESRS 2 general disclosures workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            ESRS2WorkflowResult with compliance scores and gap analysis.
        """
        if input_data is None:
            input_data = ESRS2GeneralInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting ESRS 2 general disclosures workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_governance_assessment(input_data))
            phases_done += 1
            phase_results.append(await self._phase_strategy_analysis(input_data))
            phases_done += 1
            phase_results.append(await self._phase_iro_identification(input_data))
            phases_done += 1
            phase_results.append(await self._phase_mdr_validation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("ESRS 2 workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        # Calculate overall compliance
        all_scores: List[float] = []
        all_scores.extend(self._governance_scores.values())
        all_scores.extend(self._strategy_scores.values())
        all_scores.extend(self._mdr_compliance.values())
        overall_pct = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        items_assessed = len(self._governance_scores) + len(self._strategy_scores) + len(self._mdr_compliance)

        result = ESRS2WorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            governance_scores=self._governance_scores,
            strategy_scores=self._strategy_scores,
            iro_summary=self._iro_summary,
            mdr_compliance=self._mdr_compliance,
            overall_compliance_pct=overall_pct,
            total_gaps=self._total_gaps,
            disclosure_items_assessed=items_assessed,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "ESRS 2 workflow %s completed in %.2fs: %.1f%% compliance",
            self.workflow_id, elapsed, overall_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Governance Assessment
    # -------------------------------------------------------------------------

    async def _phase_governance_assessment(
        self, input_data: ESRS2GeneralInput,
    ) -> PhaseResult:
        """Evaluate governance disclosures GOV-1 through GOV-5."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._governance_scores = {}

        for area in GovernanceArea:
            matching = [r for r in input_data.governance_data if r.area == area]
            if matching:
                record = matching[0]
                self._governance_scores[area.value] = record.score
                for gap in record.gaps_identified:
                    self._total_gaps += 1
                    warnings.append(f"{area.value}: {gap}")
            else:
                self._governance_scores[area.value] = 0.0
                self._total_gaps += 1
                warnings.append(f"{area.value}: No data provided")

        avg_score = (
            round(sum(self._governance_scores.values()) / len(self._governance_scores), 2)
            if self._governance_scores else 0.0
        )
        outputs["governance_scores"] = self._governance_scores
        outputs["average_governance_score"] = avg_score
        outputs["areas_assessed"] = len(self._governance_scores)
        outputs["areas_with_gaps"] = sum(1 for s in self._governance_scores.values() if s < 100.0)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 GovernanceAssessment: %d areas, avg score %.1f%%",
            len(self._governance_scores), avg_score,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.GOVERNANCE_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Strategy Analysis
    # -------------------------------------------------------------------------

    async def _phase_strategy_analysis(
        self, input_data: ESRS2GeneralInput,
    ) -> PhaseResult:
        """Assess strategy and business model disclosures SBM-1 to SBM-3."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._strategy_scores = {}

        for area in StrategyArea:
            matching = [r for r in input_data.strategy_data if r.area == area]
            if matching:
                record = matching[0]
                self._strategy_scores[area.value] = record.score
                if not record.key_findings:
                    warnings.append(f"{area.value}: No key findings documented")
            else:
                self._strategy_scores[area.value] = 0.0
                self._total_gaps += 1
                warnings.append(f"{area.value}: No data provided")

        # Assess stakeholder coverage for SBM-2
        stakeholder_count = len(input_data.stakeholder_data.get("groups", []))
        if stakeholder_count == 0:
            warnings.append("SBM-2: No stakeholder groups identified")

        # Check material impacts for SBM-3
        material_topics = [m for m in input_data.materiality_results if m.is_material]
        if not material_topics:
            warnings.append("SBM-3: No material topics from DMA linked to strategy")

        avg_score = (
            round(sum(self._strategy_scores.values()) / len(self._strategy_scores), 2)
            if self._strategy_scores else 0.0
        )
        outputs["strategy_scores"] = self._strategy_scores
        outputs["average_strategy_score"] = avg_score
        outputs["stakeholder_groups_count"] = stakeholder_count
        outputs["material_topics_linked"] = len(material_topics)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 StrategyAnalysis: %d areas, avg score %.1f%%",
            len(self._strategy_scores), avg_score,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.STRATEGY_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: IRO Identification
    # -------------------------------------------------------------------------

    async def _phase_iro_identification(
        self, input_data: ESRS2GeneralInput,
    ) -> PhaseResult:
        """Process impact, risk, and opportunity identification (IRO-1, IRO-2)."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._iro_summary = {}

        total_impacts = 0
        total_risks = 0
        total_opportunities = 0
        all_topics: List[str] = []

        for area in IROArea:
            matching = [r for r in input_data.iro_records if r.area == area]
            if matching:
                record = matching[0]
                total_impacts += record.impacts_identified
                total_risks += record.risks_identified
                total_opportunities += record.opportunities_identified
                all_topics.extend(record.materiality_topics)
                self._iro_summary[area.value] = {
                    "score": record.score,
                    "impacts": record.impacts_identified,
                    "risks": record.risks_identified,
                    "opportunities": record.opportunities_identified,
                }
            else:
                self._total_gaps += 1
                warnings.append(f"{area.value}: No IRO data provided")
                self._iro_summary[area.value] = {
                    "score": 0.0, "impacts": 0, "risks": 0, "opportunities": 0,
                }

        # Cross-validate with materiality results
        material_topics = {m.topic for m in input_data.materiality_results if m.is_material}
        iro_topics = set(all_topics)
        uncovered = material_topics - iro_topics
        if uncovered:
            warnings.append(
                f"IRO-2: {len(uncovered)} material topics not covered in IRO assessment"
            )

        self._iro_summary["totals"] = {
            "impacts": total_impacts,
            "risks": total_risks,
            "opportunities": total_opportunities,
            "materiality_topics_covered": len(iro_topics & material_topics),
        }

        outputs["iro_summary"] = self._iro_summary
        outputs["total_impacts"] = total_impacts
        outputs["total_risks"] = total_risks
        outputs["total_opportunities"] = total_opportunities

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 IROIdentification: %d impacts, %d risks, %d opportunities",
            total_impacts, total_risks, total_opportunities,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.IRO_IDENTIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: MDR Validation
    # -------------------------------------------------------------------------

    async def _phase_mdr_validation(
        self, input_data: ESRS2GeneralInput,
    ) -> PhaseResult:
        """Validate Minimum Disclosure Requirements (MDR-P, MDR-A, MDR-T, MDR-M)."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._mdr_compliance = {}

        for mdr_type in MDRType:
            matching = [r for r in input_data.mdr_records if r.mdr_type == mdr_type]
            if matching:
                # Average completeness across all MDR records of this type
                avg_complete = round(
                    sum(r.completeness_pct for r in matching) / len(matching), 2
                )
                self._mdr_compliance[mdr_type.value] = avg_complete
                for r in matching:
                    for gap in r.gaps:
                        self._total_gaps += 1
                        warnings.append(f"{mdr_type.value} ({r.standard_ref}): {gap}")
                undisclosed = [r for r in matching if not r.is_disclosed]
                if undisclosed:
                    warnings.append(
                        f"{mdr_type.value}: {len(undisclosed)} items not yet disclosed"
                    )
            else:
                self._mdr_compliance[mdr_type.value] = 0.0
                self._total_gaps += 1
                warnings.append(f"{mdr_type.value}: No MDR records provided")

        avg_mdr = (
            round(sum(self._mdr_compliance.values()) / len(self._mdr_compliance), 2)
            if self._mdr_compliance else 0.0
        )
        outputs["mdr_compliance"] = self._mdr_compliance
        outputs["average_mdr_compliance"] = avg_mdr
        outputs["mdr_types_assessed"] = len(self._mdr_compliance)
        outputs["mdr_fully_compliant"] = sum(
            1 for v in self._mdr_compliance.values() if v >= 100.0
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 MDRValidation: avg MDR compliance %.1f%%", avg_mdr,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MDR_VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: ESRS2GeneralInput,
    ) -> PhaseResult:
        """Compile complete ESRS 2 disclosure report."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Compile governance disclosure summary
        gov_avg = (
            round(sum(self._governance_scores.values()) / len(self._governance_scores), 2)
            if self._governance_scores else 0.0
        )
        strat_avg = (
            round(sum(self._strategy_scores.values()) / len(self._strategy_scores), 2)
            if self._strategy_scores else 0.0
        )
        mdr_avg = (
            round(sum(self._mdr_compliance.values()) / len(self._mdr_compliance), 2)
            if self._mdr_compliance else 0.0
        )

        outputs["esrs2_disclosure"] = {
            "governance_section": {
                "areas_covered": list(self._governance_scores.keys()),
                "average_compliance": gov_avg,
                "scores": self._governance_scores,
            },
            "strategy_section": {
                "areas_covered": list(self._strategy_scores.keys()),
                "average_compliance": strat_avg,
                "scores": self._strategy_scores,
            },
            "iro_section": self._iro_summary,
            "mdr_section": {
                "types_covered": list(self._mdr_compliance.keys()),
                "average_compliance": mdr_avg,
                "compliance": self._mdr_compliance,
            },
            "reporting_year": input_data.reporting_year,
            "entity_name": input_data.entity_name,
            "total_gaps": self._total_gaps,
        }
        outputs["report_ready"] = True
        outputs["total_disclosure_items"] = (
            len(self._governance_scores) + len(self._strategy_scores)
            + len(self._iro_summary) + len(self._mdr_compliance)
        )

        if self._total_gaps > 10:
            warnings.append(
                f"High gap count ({self._total_gaps}): disclosure may require additional work"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReportGeneration: ESRS 2 disclosure ready, %d gaps",
            self._total_gaps,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ESRS2WorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
