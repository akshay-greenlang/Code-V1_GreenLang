# -*- coding: utf-8 -*-
"""
ESRS 2 General Disclosures Governance Workflow
===============================================

5-phase workflow for ESRS 2 General Disclosures covering governance bodies,
strategy review, stakeholder engagement, impacts/risks/opportunities (IRO)
identification, and report assembly with full provenance tracking.

Phases:
    1. GovernanceAssessment   -- Evaluate governance bodies and oversight (GOV-1 to GOV-5)
    2. StrategyReview         -- Assess strategy and business model (SBM-1 to SBM-3)
    3. StakeholderMapping     -- Map material stakeholders and engagement channels
    4. IROIdentification      -- Identify impacts, risks, and opportunities (IRO-1, IRO-2)
    5. ReportAssembly         -- Assemble complete ESRS 2 disclosure package

ESRS 2 Disclosure Requirements (10 DRs):
    GOV-1: Role of administrative, management and supervisory bodies
    GOV-2: Information provided to and sustainability matters addressed by bodies
    GOV-3: Integration of sustainability performance in incentive schemes
    GOV-4: Statement on due diligence
    GOV-5: Risk management and internal controls over sustainability reporting
    SBM-1: Strategy, business model and value chain
    SBM-2: Interests and views of stakeholders
    SBM-3: Material impacts, risks and opportunities and interaction with strategy
    IRO-1: Description of process to identify and assess material IROs
    IRO-2: Disclosure requirements in ESRS covered by the sustainability statement

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

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

class WorkflowPhase(str, Enum):
    """Phases of the ESRS 2 governance workflow."""
    GOVERNANCE_ASSESSMENT = "governance_assessment"
    STRATEGY_REVIEW = "strategy_review"
    STAKEHOLDER_MAPPING = "stakeholder_mapping"
    IRO_IDENTIFICATION = "iro_identification"
    REPORT_ASSEMBLY = "report_assembly"

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

class DisclosureStatus(str, Enum):
    """Status of an individual ESRS 2 disclosure requirement."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"

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

class GovernanceBody(BaseModel):
    """Governance body definition per GOV-1."""
    body_id: str = Field(default_factory=lambda: f"gov-{_new_uuid()[:8]}")
    name: str = Field(default="", description="Name of governance body")
    body_type: str = Field(default="board", description="board, committee, or management")
    member_count: int = Field(default=0, ge=0)
    sustainability_expertise: bool = Field(default=False)
    meeting_frequency: str = Field(default="quarterly")
    sustainability_oversight: bool = Field(default=False)

class StakeholderGroup(BaseModel):
    """Stakeholder group definition per SBM-2."""
    group_id: str = Field(default_factory=lambda: f"stk-{_new_uuid()[:8]}")
    name: str = Field(default="", description="Stakeholder group name")
    category: str = Field(default="", description="employees, investors, communities, etc.")
    engagement_channel: str = Field(default="", description="How engagement occurs")
    material_topics: List[str] = Field(default_factory=list)
    engagement_frequency: str = Field(default="annual")

class IROItem(BaseModel):
    """Impact, Risk, or Opportunity item per IRO-1/IRO-2."""
    iro_id: str = Field(default_factory=lambda: f"iro-{_new_uuid()[:8]}")
    iro_type: str = Field(default="impact", description="impact, risk, or opportunity")
    topic: str = Field(default="", description="ESRS topic (E1, S1, etc.)")
    description: str = Field(default="")
    is_material: bool = Field(default=False)
    severity_score: float = Field(default=0.0, ge=0.0, le=5.0)
    likelihood_score: float = Field(default=0.0, ge=0.0, le=5.0)
    time_horizon: str = Field(default="medium_term", description="short/medium/long_term")

class ESRS2DisclosureItem(BaseModel):
    """Status of an individual ESRS 2 disclosure requirement."""
    disclosure_id: str = Field(..., description="GOV-X, SBM-X, or IRO-X identifier")
    name: str = Field(default="")
    status: DisclosureStatus = Field(default=DisclosureStatus.MISSING)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_points_required: int = Field(default=0, ge=0)
    data_points_completed: int = Field(default=0, ge=0)
    warnings: List[str] = Field(default_factory=list)

class ESRS2GovernanceInput(BaseModel):
    """Input data model for ESRS2GovernanceWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    governance_bodies: List[GovernanceBody] = Field(
        default_factory=list, description="GOV-1 through GOV-5 governance bodies"
    )
    strategy_data: Dict[str, Any] = Field(
        default_factory=dict, description="SBM-1 strategy and business model data"
    )
    stakeholders: List[StakeholderGroup] = Field(
        default_factory=list, description="SBM-2 stakeholder groups"
    )
    iro_items: List[IROItem] = Field(
        default_factory=list, description="IRO-1/IRO-2 impacts, risks, opportunities"
    )
    due_diligence_statement: Dict[str, Any] = Field(
        default_factory=dict, description="GOV-4 due diligence statement"
    )
    incentive_schemes: Dict[str, Any] = Field(
        default_factory=dict, description="GOV-3 incentive scheme data"
    )
    risk_management_data: Dict[str, Any] = Field(
        default_factory=dict, description="GOV-5 risk management and internal controls"
    )
    value_chain_data: Dict[str, Any] = Field(
        default_factory=dict, description="SBM-1 value chain description"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class ESRS2GovernanceResult(BaseModel):
    """Complete result from ESRS 2 governance workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="esrs2_governance")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    disclosures: List[ESRS2DisclosureItem] = Field(default_factory=list)
    overall_completeness_pct: float = Field(default=0.0)
    disclosures_complete: int = Field(default=0)
    disclosures_partial: int = Field(default=0)
    disclosures_missing: int = Field(default=0)
    governance_bodies_count: int = Field(default=0)
    sustainability_oversight_exists: bool = Field(default=False)
    stakeholder_groups_count: int = Field(default=0)
    material_iro_count: int = Field(default=0)
    has_due_diligence_statement: bool = Field(default=False)
    has_incentive_schemes: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# ESRS 2 DISCLOSURE REQUIREMENTS
# =============================================================================

ESRS2_DISCLOSURES: List[Dict[str, Any]] = [
    {"id": "GOV-1", "name": "Role of administrative, management and supervisory bodies", "data_points": 8},
    {"id": "GOV-2", "name": "Information provided to and sustainability matters addressed", "data_points": 6},
    {"id": "GOV-3", "name": "Integration of sustainability in incentive schemes", "data_points": 4},
    {"id": "GOV-4", "name": "Statement on due diligence", "data_points": 5},
    {"id": "GOV-5", "name": "Risk management and internal controls", "data_points": 6},
    {"id": "SBM-1", "name": "Strategy, business model and value chain", "data_points": 10},
    {"id": "SBM-2", "name": "Interests and views of stakeholders", "data_points": 6},
    {"id": "SBM-3", "name": "Material impacts, risks and opportunities", "data_points": 8},
    {"id": "IRO-1", "name": "Process to identify and assess material IROs", "data_points": 6},
    {"id": "IRO-2", "name": "Disclosure requirements covered by sustainability statement", "data_points": 4},
]

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ESRS2GovernanceWorkflow:
    """
    5-phase ESRS 2 General Disclosures governance workflow.

    Orchestrates governance assessment, strategy review, stakeholder mapping,
    IRO identification, and report assembly for full ESRS 2 compliance
    covering GOV-1 through GOV-5, SBM-1 through SBM-3, and IRO-1/IRO-2.

    Zero-hallucination: completeness scoring uses deterministic data-point
    counting. No LLM in numeric assessment paths.

    Example:
        >>> wf = ESRS2GovernanceWorkflow()
        >>> inp = ESRS2GovernanceInput(governance_bodies=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_completeness_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ESRS2GovernanceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._disclosures: List[ESRS2DisclosureItem] = []
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.GOVERNANCE_ASSESSMENT.value, "description": "Evaluate governance bodies and oversight"},
            {"name": WorkflowPhase.STRATEGY_REVIEW.value, "description": "Assess strategy and business model"},
            {"name": WorkflowPhase.STAKEHOLDER_MAPPING.value, "description": "Map stakeholders and engagement"},
            {"name": WorkflowPhase.IRO_IDENTIFICATION.value, "description": "Identify impacts, risks, opportunities"},
            {"name": WorkflowPhase.REPORT_ASSEMBLY.value, "description": "Assemble ESRS 2 disclosure package"},
        ]

    def validate_inputs(self, input_data: ESRS2GovernanceInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.governance_bodies:
            issues.append("No governance bodies provided (GOV-1 required)")
        if not input_data.strategy_data:
            issues.append("No strategy data provided (SBM-1 required)")
        if not input_data.stakeholders:
            issues.append("No stakeholders mapped (SBM-2 required)")
        if not input_data.iro_items:
            issues.append("No IRO items provided (IRO-1 required)")
        return issues

    async def execute(
        self,
        input_data: Optional[ESRS2GovernanceInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ESRS2GovernanceResult:
        """
        Execute the 5-phase ESRS 2 governance workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            ESRS2GovernanceResult with disclosure completeness for all GOV/SBM/IRO items.
        """
        if input_data is None:
            input_data = ESRS2GovernanceInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting ESRS 2 governance workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_governance_assessment(input_data))
            phase_results.append(await self._phase_strategy_review(input_data))
            phase_results.append(await self._phase_stakeholder_mapping(input_data))
            phase_results.append(await self._phase_iro_identification(input_data))
            phase_results.append(await self._phase_report_assembly(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("ESRS 2 governance workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        complete = sum(1 for d in self._disclosures if d.status == DisclosureStatus.COMPLETE)
        partial = sum(1 for d in self._disclosures if d.status == DisclosureStatus.PARTIAL)
        missing = sum(1 for d in self._disclosures if d.status == DisclosureStatus.MISSING)
        overall_pct = round(
            sum(d.completeness_pct for d in self._disclosures) / len(self._disclosures)
            if self._disclosures else 0.0, 1
        )

        has_oversight = any(b.sustainability_oversight for b in input_data.governance_bodies)
        material_iros = sum(1 for iro in input_data.iro_items if iro.is_material)

        result = ESRS2GovernanceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            disclosures=self._disclosures,
            overall_completeness_pct=overall_pct,
            disclosures_complete=complete,
            disclosures_partial=partial,
            disclosures_missing=missing,
            governance_bodies_count=len(input_data.governance_bodies),
            sustainability_oversight_exists=has_oversight,
            stakeholder_groups_count=len(input_data.stakeholders),
            material_iro_count=material_iros,
            has_due_diligence_statement=bool(input_data.due_diligence_statement),
            has_incentive_schemes=bool(input_data.incentive_schemes),
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "ESRS 2 governance %s completed in %.2fs: %d/%d disclosures complete (%.1f%%)",
            self.workflow_id, elapsed, complete, len(self._disclosures), overall_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Governance Assessment (GOV-1 to GOV-5)
    # -------------------------------------------------------------------------

    async def _phase_governance_assessment(
        self, input_data: ESRS2GovernanceInput,
    ) -> PhaseResult:
        """Evaluate governance bodies and sustainability oversight."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bodies = input_data.governance_bodies
        outputs["bodies_count"] = len(bodies)
        outputs["has_sustainability_oversight"] = any(b.sustainability_oversight for b in bodies)
        outputs["sustainability_expertise_count"] = sum(1 for b in bodies if b.sustainability_expertise)
        outputs["has_due_diligence"] = bool(input_data.due_diligence_statement)
        outputs["has_incentive_schemes"] = bool(input_data.incentive_schemes)
        outputs["has_risk_management"] = bool(input_data.risk_management_data)

        if not bodies:
            warnings.append("No governance bodies defined; GOV-1 will be incomplete")
        if not any(b.sustainability_oversight for b in bodies):
            warnings.append("No body has explicit sustainability oversight responsibility")
        if not input_data.due_diligence_statement:
            warnings.append("GOV-4 due diligence statement is missing")
        if not input_data.incentive_schemes:
            warnings.append("GOV-3 incentive scheme data is missing")
        if not input_data.risk_management_data:
            warnings.append("GOV-5 risk management data is missing")

        self._sub_results["governance"] = outputs

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 GovernanceAssessment: %d bodies, oversight=%s",
            len(bodies), outputs["has_sustainability_oversight"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.GOVERNANCE_ASSESSMENT.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Strategy Review (SBM-1 to SBM-3)
    # -------------------------------------------------------------------------

    async def _phase_strategy_review(
        self, input_data: ESRS2GovernanceInput,
    ) -> PhaseResult:
        """Assess strategy, business model, and value chain disclosures."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.strategy_data
        outputs["has_strategy_description"] = bool(data.get("strategy_description"))
        outputs["has_business_model"] = bool(data.get("business_model"))
        outputs["has_value_chain"] = bool(input_data.value_chain_data)
        outputs["has_resilience_analysis"] = bool(data.get("resilience_analysis"))
        outputs["markets_count"] = len(data.get("markets", []))
        outputs["products_services_count"] = len(data.get("products_services", []))

        completed_points = sum([
            bool(data.get("strategy_description")),
            bool(data.get("business_model")),
            bool(input_data.value_chain_data),
            bool(data.get("markets")),
            bool(data.get("products_services")),
            bool(data.get("resilience_analysis")),
            bool(data.get("revenue_breakdown")),
            bool(data.get("geographic_areas")),
        ])
        outputs["strategy_data_points"] = completed_points

        if not data:
            warnings.append("No strategy data provided; SBM-1 will be empty")
        if not input_data.value_chain_data:
            warnings.append("Value chain description missing for SBM-1")

        self._sub_results["strategy"] = outputs

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 StrategyReview: %d data points", completed_points)
        return PhaseResult(
            phase_name=WorkflowPhase.STRATEGY_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Stakeholder Mapping (SBM-2)
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_mapping(
        self, input_data: ESRS2GovernanceInput,
    ) -> PhaseResult:
        """Map stakeholder groups and engagement channels."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        stakeholders = input_data.stakeholders
        outputs["stakeholder_groups_count"] = len(stakeholders)
        outputs["categories"] = list(set(s.category for s in stakeholders if s.category))
        outputs["engagement_channels"] = list(set(s.engagement_channel for s in stakeholders if s.engagement_channel))
        outputs["total_material_topics"] = len(set(
            topic for s in stakeholders for topic in s.material_topics
        ))

        if not stakeholders:
            warnings.append("No stakeholder groups mapped; SBM-2 will be incomplete")
        expected_categories = {"employees", "investors", "customers", "suppliers", "communities"}
        present = set(s.category for s in stakeholders)
        missing_cats = expected_categories - present
        if missing_cats:
            warnings.append(f"Missing expected stakeholder categories: {', '.join(sorted(missing_cats))}")

        self._sub_results["stakeholders"] = outputs

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 StakeholderMapping: %d groups, %d categories",
            len(stakeholders), len(outputs["categories"]),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.STAKEHOLDER_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: IRO Identification (IRO-1, IRO-2)
    # -------------------------------------------------------------------------

    async def _phase_iro_identification(
        self, input_data: ESRS2GovernanceInput,
    ) -> PhaseResult:
        """Identify and assess material impacts, risks, and opportunities."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        iros = input_data.iro_items
        material_iros = [iro for iro in iros if iro.is_material]
        impacts = [iro for iro in iros if iro.iro_type == "impact"]
        risks = [iro for iro in iros if iro.iro_type == "risk"]
        opportunities = [iro for iro in iros if iro.iro_type == "opportunity"]

        outputs["total_iro_count"] = len(iros)
        outputs["material_iro_count"] = len(material_iros)
        outputs["impacts_count"] = len(impacts)
        outputs["risks_count"] = len(risks)
        outputs["opportunities_count"] = len(opportunities)
        outputs["topics_covered"] = list(set(iro.topic for iro in iros if iro.topic))
        outputs["high_severity_count"] = sum(1 for iro in iros if iro.severity_score >= 4.0)

        if not iros:
            warnings.append("No IRO items defined; IRO-1 will be incomplete")
        if not material_iros:
            warnings.append("No material IROs identified; materiality assessment may be incomplete")
        if not impacts:
            warnings.append("No impacts identified in IRO assessment")
        if not risks:
            warnings.append("No risks identified in IRO assessment")

        self._sub_results["iro"] = outputs

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 IROIdentification: %d total, %d material",
            len(iros), len(material_iros),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.IRO_IDENTIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Assembly
    # -------------------------------------------------------------------------

    async def _phase_report_assembly(
        self, input_data: ESRS2GovernanceInput,
    ) -> PhaseResult:
        """Assemble complete ESRS 2 disclosure from all phase results."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._disclosures = []

        # Map disclosure IDs to data sources for completeness counting
        data_source_map: Dict[str, Dict[str, Any]] = {
            "GOV-1": {"data": input_data.governance_bodies, "count_method": "list"},
            "GOV-2": {"data": input_data.governance_bodies, "count_method": "list"},
            "GOV-3": {"data": input_data.incentive_schemes, "count_method": "dict"},
            "GOV-4": {"data": input_data.due_diligence_statement, "count_method": "dict"},
            "GOV-5": {"data": input_data.risk_management_data, "count_method": "dict"},
            "SBM-1": {"data": input_data.strategy_data, "count_method": "dict"},
            "SBM-2": {"data": input_data.stakeholders, "count_method": "list"},
            "SBM-3": {"data": input_data.iro_items, "count_method": "list"},
            "IRO-1": {"data": input_data.iro_items, "count_method": "list"},
            "IRO-2": {"data": input_data.iro_items, "count_method": "list"},
        }

        for disc_def in ESRS2_DISCLOSURES:
            disc_id = disc_def["id"]
            data_points = disc_def["data_points"]
            source = data_source_map.get(disc_id, {})
            source_data = source.get("data")
            count_method = source.get("count_method", "dict")

            if count_method == "list":
                completed = min(len(source_data) if source_data else 0, data_points)
            else:
                completed = sum(1 for v in source_data.values() if v) if isinstance(source_data, dict) else 0
                completed = min(completed, data_points)

            completeness = round((completed / data_points * 100) if data_points > 0 else 0.0, 1)

            if completeness >= 80:
                status = DisclosureStatus.COMPLETE
            elif completeness > 0:
                status = DisclosureStatus.PARTIAL
            else:
                status = DisclosureStatus.MISSING

            disc_warnings: List[str] = []
            if status == DisclosureStatus.MISSING:
                disc_warnings.append(f"{disc_id} has no data provided")

            self._disclosures.append(ESRS2DisclosureItem(
                disclosure_id=disc_id,
                name=disc_def["name"],
                status=status,
                completeness_pct=completeness,
                data_points_required=data_points,
                data_points_completed=completed,
                warnings=disc_warnings,
            ))

        complete = sum(1 for d in self._disclosures if d.status == DisclosureStatus.COMPLETE)
        partial = sum(1 for d in self._disclosures if d.status == DisclosureStatus.PARTIAL)
        missing = sum(1 for d in self._disclosures if d.status == DisclosureStatus.MISSING)

        outputs["disclosures_complete"] = complete
        outputs["disclosures_partial"] = partial
        outputs["disclosures_missing"] = missing
        outputs["overall_completeness_pct"] = round(
            sum(d.completeness_pct for d in self._disclosures) / len(self._disclosures)
            if self._disclosures else 0.0, 1
        )
        outputs["disclosure_ready"] = complete == len(ESRS2_DISCLOSURES)

        if missing > 0:
            warnings.append(f"{missing} ESRS 2 disclosures have no data")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReportAssembly: %d complete, %d partial, %d missing",
            complete, partial, missing,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_ASSEMBLY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ESRS2GovernanceResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
