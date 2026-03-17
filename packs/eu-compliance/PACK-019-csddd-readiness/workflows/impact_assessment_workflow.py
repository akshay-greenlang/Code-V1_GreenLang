# -*- coding: utf-8 -*-
"""
CSDDD Impact Assessment Workflow
===============================================

4-phase workflow for identifying and assessing adverse impacts on human rights
and the environment under the EU Corporate Sustainability Due Diligence
Directive (CSDDD / CS3D). Scores severity and likelihood, prioritizes impacts,
and validates through stakeholder engagement records.

Phases:
    1. ImpactScanning          -- Scan value chain for adverse impacts
    2. SeverityLikelihoodScoring -- Score each impact by severity and likelihood
    3. Prioritization          -- Rank impacts for DD action prioritization
    4. StakeholderValidation   -- Validate findings against stakeholder input

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 6: Identifying and assessing actual and potential adverse impacts
    - Art. 6(1): Mapping operations, subsidiaries, and business relationships
    - Art. 6(4): Severity and likelihood criteria
    - Art. 10: Meaningful engagement with stakeholders
    - Annex Part I: Human rights instruments
    - Annex Part II: Environmental conventions

Author: GreenLang Team
Version: 19.0.0
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

_MODULE_VERSION = "1.0.0"


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
    """Phases of the impact assessment workflow."""
    IMPACT_SCANNING = "impact_scanning"
    SEVERITY_LIKELIHOOD_SCORING = "severity_likelihood_scoring"
    PRIORITIZATION = "prioritization"
    STAKEHOLDER_VALIDATION = "stakeholder_validation"


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


class ImpactCategory(str, Enum):
    """Category of adverse impact per CSDDD Annexes."""
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENT = "environment"


class ImpactType(str, Enum):
    """Whether the impact is actual or potential."""
    ACTUAL = "actual"
    POTENTIAL = "potential"


class SeverityLevel(str, Enum):
    """Severity classification for adverse impacts per Art. 6(4)."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StakeholderValidationStatus(str, Enum):
    """Validation status from stakeholder engagement."""
    VALIDATED = "validated"
    PARTIALLY_VALIDATED = "partially_validated"
    NOT_VALIDATED = "not_validated"
    DISPUTED = "disputed"


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


class AdverseImpact(BaseModel):
    """Record of an actual or potential adverse impact."""
    impact_id: str = Field(default_factory=lambda: f"imp-{_new_uuid()[:8]}")
    impact_name: str = Field(default="", description="Short description of the impact")
    description: str = Field(default="", description="Detailed impact description")
    category: ImpactCategory = Field(default=ImpactCategory.HUMAN_RIGHTS)
    impact_type: ImpactType = Field(default=ImpactType.POTENTIAL)
    value_chain_stage: str = Field(default="", description="Where in the value chain")
    supplier_id: str = Field(default="", description="Related supplier if applicable")
    country_code: str = Field(default="", description="ISO 3166-1 alpha-2")
    sector: str = Field(default="", description="Sector of activity")
    affected_stakeholder_groups: List[str] = Field(default_factory=list)
    annex_reference: str = Field(default="", description="CSDDD Annex Part I/II reference")
    # Severity dimensions per Art. 6(4)
    scale: float = Field(default=0.0, ge=0.0, le=10.0, description="Scale of the impact (0-10)")
    scope: float = Field(default=0.0, ge=0.0, le=10.0, description="Scope / reach (0-10)")
    irremediability: float = Field(default=0.0, ge=0.0, le=10.0, description="Irremediable nature (0-10)")
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0, description="Likelihood 0-1")


class StakeholderEngagement(BaseModel):
    """Stakeholder engagement record for validation."""
    engagement_id: str = Field(default_factory=lambda: f"eng-{_new_uuid()[:8]}")
    stakeholder_group: str = Field(default="", description="Name of stakeholder group")
    engagement_method: str = Field(default="", description="Method: consultation, survey, etc.")
    date_conducted: str = Field(default="", description="ISO date of engagement")
    impact_ids_reviewed: List[str] = Field(
        default_factory=list, description="Impact IDs reviewed in this engagement"
    )
    findings: str = Field(default="", description="Key findings from engagement")
    validation_status: StakeholderValidationStatus = Field(
        default=StakeholderValidationStatus.NOT_VALIDATED
    )
    concerns_raised: List[str] = Field(default_factory=list)


class ImpactAssessmentInput(BaseModel):
    """Input data model for ImpactAssessmentWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    adverse_impacts: List[AdverseImpact] = Field(
        default_factory=list, description="Identified adverse impacts"
    )
    stakeholder_engagements: List[StakeholderEngagement] = Field(
        default_factory=list, description="Stakeholder engagement records"
    )
    value_chain_countries: List[str] = Field(
        default_factory=list, description="Countries in value chain"
    )
    value_chain_sectors: List[str] = Field(
        default_factory=list, description="Sectors in value chain"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class ScoredImpact(BaseModel):
    """Impact with computed severity and priority scores."""
    impact_id: str = Field(...)
    impact_name: str = Field(default="")
    category: str = Field(default="")
    impact_type: str = Field(default="")
    severity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    severity_level: SeverityLevel = Field(default=SeverityLevel.MEDIUM)
    likelihood: float = Field(default=0.5)
    priority_score: float = Field(default=0.0, ge=0.0, le=100.0)
    priority_rank: int = Field(default=0, ge=0)
    validation_status: str = Field(default="not_validated")
    recommended_action: str = Field(default="")


class ImpactAssessmentResult(BaseModel):
    """Complete result from impact assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="impact_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Impact summary
    total_impacts: int = Field(default=0, ge=0)
    actual_impacts: int = Field(default=0, ge=0)
    potential_impacts: int = Field(default=0, ge=0)
    human_rights_impacts: int = Field(default=0, ge=0)
    environmental_impacts: int = Field(default=0, ge=0)
    # Severity distribution
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)
    # Scored and ranked
    scored_impacts: List[ScoredImpact] = Field(default_factory=list)
    priority_list: List[Dict[str, Any]] = Field(default_factory=list)
    # Stakeholder
    stakeholder_validation_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    engagements_conducted: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ImpactAssessmentWorkflow:
    """
    4-phase CSDDD impact assessment workflow.

    Scans value chain data for adverse impacts, scores them by severity and
    likelihood per Art. 6(4) criteria, prioritizes for DD action, and validates
    against stakeholder engagement records.

    Zero-hallucination: all severity/priority scores use deterministic
    formulas. No LLM in numeric calculation paths.

    Example:
        >>> wf = ImpactAssessmentWorkflow()
        >>> inp = ImpactAssessmentInput(adverse_impacts=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_impacts >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ImpactAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._scored_impacts: List[ScoredImpact] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.IMPACT_SCANNING.value, "description": "Scan for adverse impacts"},
            {"name": WorkflowPhase.SEVERITY_LIKELIHOOD_SCORING.value, "description": "Score severity and likelihood"},
            {"name": WorkflowPhase.PRIORITIZATION.value, "description": "Prioritize impacts for action"},
            {"name": WorkflowPhase.STAKEHOLDER_VALIDATION.value, "description": "Validate with stakeholder input"},
        ]

    def validate_inputs(self, input_data: ImpactAssessmentInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.adverse_impacts:
            issues.append("No adverse impacts provided for assessment")
        for imp in input_data.adverse_impacts:
            if imp.scale == 0 and imp.scope == 0 and imp.irremediability == 0:
                issues.append(f"Impact {imp.impact_id} has no severity dimensions scored")
        return issues

    async def execute(
        self,
        input_data: Optional[ImpactAssessmentInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ImpactAssessmentResult:
        """
        Execute the 4-phase impact assessment workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            ImpactAssessmentResult with scored impacts and priority list.
        """
        if input_data is None:
            input_data = ImpactAssessmentInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting impact assessment workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_impact_scanning(input_data))
            phase_results.append(await self._phase_severity_likelihood_scoring(input_data))
            phase_results.append(await self._phase_prioritization(input_data))
            phase_results.append(await self._phase_stakeholder_validation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Impact assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        impacts = input_data.adverse_impacts
        severity_counts = self._count_severity_levels()
        validated_count = sum(
            1 for si in self._scored_impacts if si.validation_status == "validated"
        )
        validation_rate = (
            (validated_count / len(self._scored_impacts)) * 100
        ) if self._scored_impacts else 0.0

        result = ImpactAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            total_impacts=len(impacts),
            actual_impacts=sum(1 for i in impacts if i.impact_type == ImpactType.ACTUAL),
            potential_impacts=sum(1 for i in impacts if i.impact_type == ImpactType.POTENTIAL),
            human_rights_impacts=sum(1 for i in impacts if i.category == ImpactCategory.HUMAN_RIGHTS),
            environmental_impacts=sum(1 for i in impacts if i.category == ImpactCategory.ENVIRONMENT),
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            scored_impacts=self._scored_impacts,
            priority_list=[
                {
                    "rank": si.priority_rank,
                    "impact_id": si.impact_id,
                    "impact_name": si.impact_name,
                    "priority_score": si.priority_score,
                    "severity_level": si.severity_level.value,
                    "recommended_action": si.recommended_action,
                }
                for si in self._scored_impacts
            ],
            stakeholder_validation_rate=round(validation_rate, 1),
            engagements_conducted=len(input_data.stakeholder_engagements),
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Impact assessment %s completed in %.2fs: %d impacts scored",
            self.workflow_id, elapsed, len(self._scored_impacts),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Impact Scanning
    # -------------------------------------------------------------------------

    async def _phase_impact_scanning(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Scan value chain for adverse impacts and catalogue them."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        impacts = input_data.adverse_impacts

        # Categorize
        by_category: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_stage: Dict[str, int] = {}
        by_country: Dict[str, int] = {}

        for imp in impacts:
            by_category[imp.category.value] = by_category.get(imp.category.value, 0) + 1
            by_type[imp.impact_type.value] = by_type.get(imp.impact_type.value, 0) + 1
            if imp.value_chain_stage:
                by_stage[imp.value_chain_stage] = by_stage.get(imp.value_chain_stage, 0) + 1
            if imp.country_code:
                by_country[imp.country_code] = by_country.get(imp.country_code, 0) + 1

        outputs["total_impacts_scanned"] = len(impacts)
        outputs["by_category"] = by_category
        outputs["by_type"] = by_type
        outputs["by_value_chain_stage"] = by_stage
        outputs["by_country"] = by_country
        outputs["unique_countries"] = len(by_country)
        outputs["affected_stakeholder_groups"] = list(set(
            grp for imp in impacts for grp in imp.affected_stakeholder_groups
        ))

        if not impacts:
            warnings.append("No adverse impacts identified -- ensure Art. 6 scanning is complete")
        if by_category.get("human_rights", 0) == 0:
            warnings.append("No human rights impacts identified -- review Annex Part I coverage")
        if by_category.get("environment", 0) == 0:
            warnings.append("No environmental impacts identified -- review Annex Part II coverage")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ImpactScanning: %d impacts across %d countries",
            len(impacts), len(by_country),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.IMPACT_SCANNING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Severity / Likelihood Scoring
    # -------------------------------------------------------------------------

    async def _phase_severity_likelihood_scoring(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Score each impact by severity (scale, scope, irremediability) and likelihood."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._scored_impacts = []

        for imp in input_data.adverse_impacts:
            # Severity: weighted combination of scale (40%), scope (30%), irremediability (30%)
            severity_raw = (
                0.40 * imp.scale
                + 0.30 * imp.scope
                + 0.30 * imp.irremediability
            )
            # Normalize to 0-100
            severity_score = round((severity_raw / 10.0) * 100, 1)

            # Determine severity level
            if severity_score >= 75:
                severity_level = SeverityLevel.CRITICAL
            elif severity_score >= 50:
                severity_level = SeverityLevel.HIGH
            elif severity_score >= 25:
                severity_level = SeverityLevel.MEDIUM
            else:
                severity_level = SeverityLevel.LOW

            # Priority score: severity * likelihood weighting
            # For actual impacts, increase priority (they are already occurring)
            type_multiplier = 1.2 if imp.impact_type == ImpactType.ACTUAL else 1.0
            priority_score = round(
                severity_score * imp.likelihood * type_multiplier, 1
            )
            priority_score = min(100.0, priority_score)

            # Recommend action based on severity and type
            recommended = self._recommend_action(severity_level, imp.impact_type)

            self._scored_impacts.append(ScoredImpact(
                impact_id=imp.impact_id,
                impact_name=imp.impact_name,
                category=imp.category.value,
                impact_type=imp.impact_type.value,
                severity_score=severity_score,
                severity_level=severity_level,
                likelihood=imp.likelihood,
                priority_score=priority_score,
                priority_rank=0,  # Set in prioritization phase
                validation_status="not_validated",
                recommended_action=recommended,
            ))

        # Summary statistics
        severity_scores = [si.severity_score for si in self._scored_impacts]
        outputs["impacts_scored"] = len(self._scored_impacts)
        outputs["avg_severity_score"] = round(
            sum(severity_scores) / len(severity_scores), 1
        ) if severity_scores else 0.0
        outputs["max_severity_score"] = max(severity_scores) if severity_scores else 0.0
        outputs["severity_distribution"] = {
            level.value: sum(
                1 for si in self._scored_impacts if si.severity_level == level
            )
            for level in SeverityLevel
        }

        if any(si.severity_level == SeverityLevel.CRITICAL for si in self._scored_impacts):
            warnings.append("Critical severity impacts identified -- immediate action required per Art. 8")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 SeverityScoring: %d scored, avg=%.1f",
            len(self._scored_impacts), outputs["avg_severity_score"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SEVERITY_LIKELIHOOD_SCORING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Prioritization
    # -------------------------------------------------------------------------

    async def _phase_prioritization(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Rank impacts by priority score for DD action planning."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Sort by priority score descending
        self._scored_impacts.sort(key=lambda si: si.priority_score, reverse=True)

        # Assign ranks
        for rank, si in enumerate(self._scored_impacts, start=1):
            si.priority_rank = rank

        # Top-priority (critical + high with high likelihood)
        top_priority = [
            si for si in self._scored_impacts
            if si.severity_level in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
            and si.likelihood >= 0.5
        ]

        # Category split
        hr_priority = [si for si in top_priority if si.category == "human_rights"]
        env_priority = [si for si in top_priority if si.category == "environment"]

        outputs["total_ranked"] = len(self._scored_impacts)
        outputs["top_priority_count"] = len(top_priority)
        outputs["top_5_impacts"] = [
            {
                "rank": si.priority_rank,
                "impact_id": si.impact_id,
                "impact_name": si.impact_name,
                "priority_score": si.priority_score,
                "severity": si.severity_level.value,
                "category": si.category,
            }
            for si in self._scored_impacts[:5]
        ]
        outputs["human_rights_in_top_priority"] = len(hr_priority)
        outputs["environment_in_top_priority"] = len(env_priority)
        outputs["actual_in_top_priority"] = sum(
            1 for si in top_priority if si.impact_type == "actual"
        )

        if not top_priority:
            warnings.append("No impacts meet top-priority threshold -- verify scoring")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 Prioritization: %d ranked, %d top priority",
            len(self._scored_impacts), len(top_priority),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PRIORITIZATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Stakeholder Validation
    # -------------------------------------------------------------------------

    async def _phase_stakeholder_validation(
        self, input_data: ImpactAssessmentInput,
    ) -> PhaseResult:
        """Validate impact findings against stakeholder engagement data."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        engagements = input_data.stakeholder_engagements

        # Build map of impact_id -> validation status from engagements
        impact_validation: Dict[str, StakeholderValidationStatus] = {}
        impact_concerns: Dict[str, List[str]] = {}

        for eng in engagements:
            for imp_id in eng.impact_ids_reviewed:
                # Take the most positive validation status for each impact
                current = impact_validation.get(imp_id)
                if current is None or _validation_rank(eng.validation_status) > _validation_rank(current):
                    impact_validation[imp_id] = eng.validation_status
                if eng.concerns_raised:
                    impact_concerns.setdefault(imp_id, []).extend(eng.concerns_raised)

        # Update scored impacts with validation status
        for si in self._scored_impacts:
            vs = impact_validation.get(si.impact_id)
            if vs is not None:
                si.validation_status = vs.value
            else:
                si.validation_status = "not_validated"

        validated_count = sum(
            1 for si in self._scored_impacts
            if si.validation_status == "validated"
        )
        partially_validated = sum(
            1 for si in self._scored_impacts
            if si.validation_status == "partially_validated"
        )
        not_validated = sum(
            1 for si in self._scored_impacts
            if si.validation_status == "not_validated"
        )
        disputed = sum(
            1 for si in self._scored_impacts
            if si.validation_status == "disputed"
        )

        outputs["engagements_conducted"] = len(engagements)
        outputs["unique_stakeholder_groups"] = len(set(e.stakeholder_group for e in engagements))
        outputs["impacts_validated"] = validated_count
        outputs["impacts_partially_validated"] = partially_validated
        outputs["impacts_not_validated"] = not_validated
        outputs["impacts_disputed"] = disputed
        outputs["validation_rate_pct"] = round(
            (validated_count / len(self._scored_impacts)) * 100, 1
        ) if self._scored_impacts else 0.0
        outputs["total_concerns_raised"] = sum(len(c) for c in impact_concerns.values())

        if not engagements:
            warnings.append("No stakeholder engagements recorded -- Art. 10 compliance at risk")
        if not_validated > len(self._scored_impacts) * 0.5:
            warnings.append("Over 50% of impacts lack stakeholder validation")
        if disputed > 0:
            warnings.append(f"{disputed} impacts disputed by stakeholders -- review required")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 StakeholderValidation: %d validated, %d disputed, rate=%.1f%%",
            validated_count, disputed, outputs["validation_rate_pct"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.STAKEHOLDER_VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _recommend_action(severity: SeverityLevel, impact_type: ImpactType) -> str:
        """Recommend DD action based on severity and impact type."""
        if impact_type == ImpactType.ACTUAL:
            if severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
                return "Cease/suspend activity and provide remediation per Art. 8-9"
            return "Implement corrective action plan per Art. 8"
        else:
            if severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
                return "Implement prevention measures and contractual assurances per Art. 7"
            return "Monitor and develop preventive measures per Art. 7"

    def _count_severity_levels(self) -> Dict[str, int]:
        """Count scored impacts by severity level."""
        counts: Dict[str, int] = {}
        for si in self._scored_impacts:
            counts[si.severity_level.value] = counts.get(si.severity_level.value, 0) + 1
        return counts

    def _compute_provenance(self, result: ImpactAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)


def _validation_rank(status: StakeholderValidationStatus) -> int:
    """Rank validation statuses for comparison (higher = better)."""
    ranking = {
        StakeholderValidationStatus.VALIDATED: 3,
        StakeholderValidationStatus.PARTIALLY_VALIDATED: 2,
        StakeholderValidationStatus.NOT_VALIDATED: 1,
        StakeholderValidationStatus.DISPUTED: 0,
    }
    return ranking.get(status, 0)
