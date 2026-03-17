# -*- coding: utf-8 -*-
"""
CSDDDOrchestrator - Master Pipeline for PACK-019 CSDDD Readiness Assessment
===============================================================================

This module implements the master pipeline orchestrator for the CSDDD (Corporate
Sustainability Due Diligence Directive) Readiness Pack. It coordinates a 7-phase
assessment pipeline covering all mandatory CSDDD obligations from scope
determination through final scorecard generation.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Articles 5-16: Due diligence obligations
    - Article 22: Climate transition plan
    - Article 26: Civil liability
    - Annex Part I & II: Adverse impacts catalogue

Pipeline Phases (7 total):
    1. SCOPE_DETERMINATION       -- Determine if company falls within CSDDD scope
    2. IMPACT_ASSESSMENT         -- Identify and assess actual/potential adverse impacts
    3. PREVENTION_REVIEW         -- Review prevention and mitigation measures
    4. GRIEVANCE_ASSESSMENT      -- Assess grievance mechanisms and remediation
    5. CLIMATE_TRANSITION        -- Evaluate climate transition plan (Art 22)
    6. LIABILITY_ASSESSMENT      -- Assess civil liability exposure (Art 26)
    7. SCORECARD_GENERATION      -- Generate final CSDDD readiness scorecard

DAG Dependencies:
    SCOPE_DETERMINATION --> IMPACT_ASSESSMENT
    IMPACT_ASSESSMENT   --> PREVENTION_REVIEW
    IMPACT_ASSESSMENT   --> GRIEVANCE_ASSESSMENT  (parallel with PREVENTION_REVIEW)
    SCOPE_DETERMINATION --> CLIMATE_TRANSITION     (parallel with IMPACT_ASSESSMENT)
    PREVENTION_REVIEW + GRIEVANCE_ASSESSMENT --> LIABILITY_ASSESSMENT
    LIABILITY_ASSESSMENT + CLIMATE_TRANSITION --> SCORECARD_GENERATION

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
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

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking.

    Args:
        data: Data to hash. Supports Pydantic models, dicts, and strings.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CSDDDPhase(str, Enum):
    """The 7 phases of the CSDDD readiness assessment pipeline."""

    SCOPE_DETERMINATION = "scope_determination"
    IMPACT_ASSESSMENT = "impact_assessment"
    PREVENTION_REVIEW = "prevention_review"
    GRIEVANCE_ASSESSMENT = "grievance_assessment"
    CLIMATE_TRANSITION = "climate_transition"
    LIABILITY_ASSESSMENT = "liability_assessment"
    SCORECARD_GENERATION = "scorecard_generation"


class ExecutionStatus(str, Enum):
    """Pipeline execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class CompanySize(str, Enum):
    """CSDDD company size classification per Article 2."""

    GROUP_1 = "group_1"          # >1000 employees AND >EUR 450M turnover
    GROUP_2 = "group_2"          # >500 employees AND >EUR 150M turnover
    FRANCHISE_LICENSEE = "franchise_licensee"  # >EUR 80M turnover with royalties
    OUT_OF_SCOPE = "out_of_scope"


class ReadinessLevel(str, Enum):
    """CSDDD readiness maturity level."""

    NOT_READY = "not_ready"
    EARLY_STAGE = "early_stage"
    DEVELOPING = "developing"
    ADVANCED = "advanced"
    FULLY_READY = "fully_ready"


# ---------------------------------------------------------------------------
# CSDDD Article References
# ---------------------------------------------------------------------------

CSDDD_ARTICLES: Dict[str, str] = {
    "Art_2": "Scope - companies in scope",
    "Art_5": "Due diligence - integration into policies",
    "Art_6": "Identifying adverse impacts",
    "Art_7": "Prioritisation of identified impacts",
    "Art_8": "Preventing potential adverse impacts",
    "Art_9": "Bringing actual adverse impacts to an end",
    "Art_10": "Remediation of actual adverse impacts",
    "Art_11": "Meaningful engagement with stakeholders",
    "Art_12": "Notification mechanism (grievance)",
    "Art_13": "Monitoring effectiveness",
    "Art_14": "Public communication and reporting",
    "Art_15": "Delegated acts",
    "Art_16": "Model contractual clauses",
    "Art_22": "Climate transition plan",
    "Art_26": "Civil liability",
}

PHASE_TO_ARTICLES: Dict[CSDDDPhase, List[str]] = {
    CSDDDPhase.SCOPE_DETERMINATION: ["Art_2"],
    CSDDDPhase.IMPACT_ASSESSMENT: ["Art_6", "Art_7", "Art_11"],
    CSDDDPhase.PREVENTION_REVIEW: ["Art_5", "Art_8", "Art_9", "Art_13", "Art_16"],
    CSDDDPhase.GRIEVANCE_ASSESSMENT: ["Art_10", "Art_12"],
    CSDDDPhase.CLIMATE_TRANSITION: ["Art_22"],
    CSDDDPhase.LIABILITY_ASSESSMENT: ["Art_26"],
    CSDDDPhase.SCORECARD_GENERATION: ["Art_14"],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class OrchestratorConfig(BaseModel):
    """Configuration for the CSDDD Readiness Pipeline Orchestrator."""

    pack_id: str = Field(default="PACK-019")
    pack_version: str = Field(default="1.0.0")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    base_currency: str = Field(default="EUR")
    transition_deadline_group1: int = Field(
        default=2027, description="CSDDD compliance deadline for Group 1"
    )
    transition_deadline_group2: int = Field(
        default=2029, description="CSDDD compliance deadline for Group 2"
    )


class PhaseProvenance(BaseModel):
    """Provenance tracking for a single phase execution."""

    phase: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


class PhaseResult(BaseModel):
    """Result of a single phase execution."""

    phase: CSDDDPhase = Field(...)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[PhaseProvenance] = Field(None)
    articles_covered: List[str] = Field(default_factory=list)


class CompanyProfile(BaseModel):
    """Company profile input for scope determination."""

    company_name: str = Field(default="")
    employee_count: int = Field(default=0, ge=0)
    net_turnover_eur: float = Field(default=0.0, ge=0.0)
    is_eu_incorporated: bool = Field(default=True)
    has_eu_operations: bool = Field(default=True)
    is_franchise_or_licensee: bool = Field(default=False)
    royalty_revenue_eur: float = Field(default=0.0, ge=0.0)
    sector: str = Field(default="")
    nace_codes: List[str] = Field(default_factory=list)
    countries_of_operation: List[str] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    """Complete result of the CSDDD readiness assessment pipeline."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-019")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    company_size: CompanySize = Field(default=CompanySize.OUT_OF_SCOPE)
    readiness_level: ReadinessLevel = Field(default=ReadinessLevel.NOT_READY)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    articles_covered: List[str] = Field(default_factory=list)
    compliance_deadline: Optional[int] = Field(None)
    provenance_hash: str = Field(default="")


class AssessmentSummary(BaseModel):
    """Summary of a completed CSDDD readiness assessment."""

    execution_id: str = Field(default="")
    company_name: str = Field(default="")
    company_size: CompanySize = Field(default=CompanySize.OUT_OF_SCOPE)
    readiness_level: ReadinessLevel = Field(default=ReadinessLevel.NOT_READY)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    phase_scores: Dict[str, float] = Field(default_factory=dict)
    key_gaps: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)
    compliance_deadline: Optional[int] = Field(None)
    articles_covered: int = Field(default=0)
    articles_total: int = Field(default=15)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Phase Execution Order and Dependencies
# ---------------------------------------------------------------------------

PHASE_EXECUTION_ORDER: List[CSDDDPhase] = [
    CSDDDPhase.SCOPE_DETERMINATION,
    CSDDDPhase.IMPACT_ASSESSMENT,
    CSDDDPhase.PREVENTION_REVIEW,
    CSDDDPhase.GRIEVANCE_ASSESSMENT,
    CSDDDPhase.CLIMATE_TRANSITION,
    CSDDDPhase.LIABILITY_ASSESSMENT,
    CSDDDPhase.SCORECARD_GENERATION,
]

PHASE_DEPENDENCIES: Dict[CSDDDPhase, List[CSDDDPhase]] = {
    CSDDDPhase.SCOPE_DETERMINATION: [],
    CSDDDPhase.IMPACT_ASSESSMENT: [CSDDDPhase.SCOPE_DETERMINATION],
    CSDDDPhase.PREVENTION_REVIEW: [CSDDDPhase.IMPACT_ASSESSMENT],
    CSDDDPhase.GRIEVANCE_ASSESSMENT: [CSDDDPhase.IMPACT_ASSESSMENT],
    CSDDDPhase.CLIMATE_TRANSITION: [CSDDDPhase.SCOPE_DETERMINATION],
    CSDDDPhase.LIABILITY_ASSESSMENT: [
        CSDDDPhase.PREVENTION_REVIEW,
        CSDDDPhase.GRIEVANCE_ASSESSMENT,
    ],
    CSDDDPhase.SCORECARD_GENERATION: [
        CSDDDPhase.LIABILITY_ASSESSMENT,
        CSDDDPhase.CLIMATE_TRANSITION,
    ],
}


# ---------------------------------------------------------------------------
# CSDDDOrchestrator
# ---------------------------------------------------------------------------


class CSDDDOrchestrator:
    """Master pipeline orchestrator for PACK-019 CSDDD Readiness Assessment.

    Executes a 7-phase DAG pipeline covering all CSDDD obligations from scope
    determination through final scorecard generation. Each phase maps to
    specific CSDDD articles and produces a deterministic readiness score.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.

    Example:
        >>> config = OrchestratorConfig(reporting_year=2025)
        >>> orch = CSDDDOrchestrator(config)
        >>> profile = CompanyProfile(company_name="Acme", employee_count=1500,
        ...                          net_turnover_eur=500_000_000)
        >>> result = orch.run_full_assessment(profile=profile)
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        """Initialize CSDDDOrchestrator.

        Args:
            config: Orchestrator configuration. Defaults used if None.
        """
        self.config = config or OrchestratorConfig()
        self._results: Dict[str, AssessmentResult] = {}
        logger.info(
            "CSDDDOrchestrator initialized (pack=%s, year=%d)",
            self.config.pack_id,
            self.config.reporting_year,
        )

    def run_full_assessment(
        self,
        profile: Optional[CompanyProfile] = None,
        impacts: Optional[List[Dict[str, Any]]] = None,
        measures: Optional[List[Dict[str, Any]]] = None,
        cases: Optional[List[Dict[str, Any]]] = None,
        targets: Optional[List[Dict[str, Any]]] = None,
        engagements: Optional[List[Dict[str, Any]]] = None,
    ) -> AssessmentResult:
        """Execute the full 7-phase CSDDD readiness assessment pipeline.

        Args:
            profile: Company profile for scope determination.
            impacts: Identified adverse impacts data.
            measures: Prevention and mitigation measures data.
            cases: Grievance cases data.
            targets: Climate transition targets data.
            engagements: Stakeholder engagement records.

        Returns:
            AssessmentResult with status, phase results, and readiness score.
        """
        result = AssessmentResult(
            pack_id=self.config.pack_id,
            started_at=_utcnow(),
            status=ExecutionStatus.RUNNING,
        )
        self._results[result.execution_id] = result

        context: Dict[str, Any] = {
            "profile": profile.model_dump() if profile else {},
            "impacts": impacts or [],
            "measures": measures or [],
            "cases": cases or [],
            "targets": targets or [],
            "engagements": engagements or [],
            "reporting_year": self.config.reporting_year,
        }

        try:
            completed_phases: set = set()

            for phase in PHASE_EXECUTION_ORDER:
                deps = PHASE_DEPENDENCIES.get(phase, [])
                unmet = [d for d in deps if d.value not in completed_phases]

                if unmet:
                    logger.warning(
                        "Phase %s skipped: unmet dependencies %s",
                        phase.value,
                        [d.value for d in unmet],
                    )
                    result.phases_skipped.append(phase.value)
                    continue

                phase_result = self._execute_phase(phase, context)
                result.phase_results[phase.value] = phase_result

                if phase_result.status == ExecutionStatus.COMPLETED:
                    result.phases_completed.append(phase.value)
                    completed_phases.add(phase.value)
                    result.total_records_processed += phase_result.records_processed
                    result.articles_covered.extend(phase_result.articles_covered)
                    context[f"{phase.value}_result"] = phase_result.outputs
                else:
                    result.errors.append(f"Phase {phase.value} failed")
                    if phase == CSDDDPhase.SCOPE_DETERMINATION:
                        result.status = ExecutionStatus.FAILED
                        break

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            logger.error("Assessment failed: %s", str(exc), exc_info=True)
            result.status = ExecutionStatus.FAILED
            result.errors.append(str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000

        # Extract scope from context
        scope_result = context.get("scope_determination_result", {})
        result.company_size = CompanySize(
            scope_result.get("company_size", CompanySize.OUT_OF_SCOPE.value)
        )
        result.compliance_deadline = scope_result.get("compliance_deadline")

        # Calculate readiness
        result.readiness_score = self._compute_readiness_score(result)
        result.readiness_level = self._score_to_level(result.readiness_score)
        result.articles_covered = list(set(result.articles_covered))

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "Assessment %s: %s in %.1fms (readiness=%.1f%%, level=%s)",
            result.execution_id,
            result.status.value,
            result.total_duration_ms,
            result.readiness_score,
            result.readiness_level.value,
        )
        return result

    def run_quick_assessment(
        self,
        profile: CompanyProfile,
    ) -> AssessmentResult:
        """Run a quick scope-only assessment to determine CSDDD applicability.

        Args:
            profile: Company profile for scope determination.

        Returns:
            AssessmentResult with scope determination only.
        """
        result = AssessmentResult(
            pack_id=self.config.pack_id,
            started_at=_utcnow(),
            status=ExecutionStatus.RUNNING,
        )
        self._results[result.execution_id] = result

        context: Dict[str, Any] = {
            "profile": profile.model_dump(),
            "reporting_year": self.config.reporting_year,
        }

        try:
            phase_result = self._execute_phase(
                CSDDDPhase.SCOPE_DETERMINATION, context
            )
            result.phase_results[CSDDDPhase.SCOPE_DETERMINATION.value] = phase_result

            if phase_result.status == ExecutionStatus.COMPLETED:
                result.phases_completed.append(CSDDDPhase.SCOPE_DETERMINATION.value)
                result.company_size = CompanySize(
                    phase_result.outputs.get(
                        "company_size", CompanySize.OUT_OF_SCOPE.value
                    )
                )
                result.compliance_deadline = phase_result.outputs.get(
                    "compliance_deadline"
                )
            result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            result.status = ExecutionStatus.FAILED
            result.errors.append(str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_status(self, execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of a pipeline execution or overall orchestrator.

        Args:
            execution_id: Specific execution ID, or None for overall status.

        Returns:
            Dict with status information.
        """
        if execution_id:
            r = self._results.get(execution_id)
            if r is None:
                return {"status": "not_found", "execution_id": execution_id}
            return {
                "execution_id": r.execution_id,
                "status": r.status.value,
                "readiness_score": r.readiness_score,
                "phases_completed": r.phases_completed,
                "total_duration_ms": r.total_duration_ms,
            }

        return {
            "pack_id": self.config.pack_id,
            "total_assessments": len(self._results),
            "phase_count": len(PHASE_EXECUTION_ORDER),
            "articles_count": len(CSDDD_ARTICLES),
        }

    def get_assessment_summary(self, execution_id: str) -> AssessmentSummary:
        """Get a human-readable summary of a completed assessment.

        Args:
            execution_id: Execution ID to summarize.

        Returns:
            AssessmentSummary with readiness details and priority actions.
        """
        r = self._results.get(execution_id)
        if r is None:
            return AssessmentSummary(execution_id=execution_id)

        phase_scores: Dict[str, float] = {}
        for phase_name, phase_result in r.phase_results.items():
            score = phase_result.outputs.get("phase_score", 0.0)
            phase_scores[phase_name] = score

        key_gaps = self._identify_gaps(r)
        priority_actions = self._generate_priority_actions(r, key_gaps)

        profile_data = r.phase_results.get(
            CSDDDPhase.SCOPE_DETERMINATION.value
        )
        company_name = ""
        if profile_data:
            company_name = profile_data.outputs.get("company_name", "")

        summary = AssessmentSummary(
            execution_id=r.execution_id,
            company_name=company_name,
            company_size=r.company_size,
            readiness_level=r.readiness_level,
            readiness_score=r.readiness_score,
            phase_scores=phase_scores,
            key_gaps=key_gaps,
            priority_actions=priority_actions,
            compliance_deadline=r.compliance_deadline,
            articles_covered=len(set(r.articles_covered)),
            articles_total=len(CSDDD_ARTICLES),
        )
        summary.provenance_hash = _compute_hash(summary)
        return summary

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_phase(
        self,
        phase: CSDDDPhase,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """Execute a single assessment phase.

        Args:
            phase: Phase to execute.
            context: Shared pipeline context.

        Returns:
            PhaseResult with outputs and status.
        """
        phase_result = PhaseResult(
            phase=phase,
            started_at=_utcnow(),
            status=ExecutionStatus.RUNNING,
            articles_covered=PHASE_TO_ARTICLES.get(phase, []),
        )

        try:
            input_hash = _compute_hash(context) if self.config.enable_provenance else ""

            handlers = {
                CSDDDPhase.SCOPE_DETERMINATION: self._phase_scope_determination,
                CSDDDPhase.IMPACT_ASSESSMENT: self._phase_impact_assessment,
                CSDDDPhase.PREVENTION_REVIEW: self._phase_prevention_review,
                CSDDDPhase.GRIEVANCE_ASSESSMENT: self._phase_grievance_assessment,
                CSDDDPhase.CLIMATE_TRANSITION: self._phase_climate_transition,
                CSDDDPhase.LIABILITY_ASSESSMENT: self._phase_liability_assessment,
                CSDDDPhase.SCORECARD_GENERATION: self._phase_scorecard_generation,
            }

            handler = handlers.get(phase)
            if handler is None:
                raise ValueError(f"No handler for phase: {phase.value}")

            outputs = handler(context)
            phase_result.outputs = outputs
            phase_result.status = ExecutionStatus.COMPLETED
            phase_result.records_processed = outputs.get("records_processed", 0)

            if self.config.enable_provenance:
                output_hash = _compute_hash(outputs)
                phase_result.provenance = PhaseProvenance(
                    phase=phase.value,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    duration_ms=0.0,
                )

            logger.info("Phase %s completed", phase.value)

        except Exception as exc:
            phase_result.status = ExecutionStatus.FAILED
            phase_result.errors.append(str(exc))
            logger.error("Phase %s failed: %s", phase.value, str(exc))

        phase_result.completed_at = _utcnow()
        if phase_result.started_at:
            phase_result.duration_ms = (
                phase_result.completed_at - phase_result.started_at
            ).total_seconds() * 1000
        return phase_result

    # ------------------------------------------------------------------
    # Phase logic implementations (deterministic, zero-hallucination)
    # ------------------------------------------------------------------

    def _phase_scope_determination(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Determine CSDDD scope per Article 2."""
        profile = context.get("profile", {})
        employees = profile.get("employee_count", 0)
        turnover = profile.get("net_turnover_eur", 0.0)
        is_franchise = profile.get("is_franchise_or_licensee", False)
        royalty_revenue = profile.get("royalty_revenue_eur", 0.0)

        # Deterministic scope rules per Art 2
        if employees > 1000 and turnover > 450_000_000:
            company_size = CompanySize.GROUP_1.value
            deadline = self.config.transition_deadline_group1
        elif employees > 500 and turnover > 150_000_000:
            company_size = CompanySize.GROUP_2.value
            deadline = self.config.transition_deadline_group2
        elif is_franchise and royalty_revenue > 80_000_000:
            company_size = CompanySize.FRANCHISE_LICENSEE.value
            deadline = self.config.transition_deadline_group2
        else:
            company_size = CompanySize.OUT_OF_SCOPE.value
            deadline = None

        in_scope = company_size != CompanySize.OUT_OF_SCOPE.value

        return {
            "company_name": profile.get("company_name", ""),
            "company_size": company_size,
            "in_scope": in_scope,
            "compliance_deadline": deadline,
            "employee_count": employees,
            "net_turnover_eur": turnover,
            "phase_score": 100.0 if in_scope else 0.0,
            "records_processed": 1,
        }

    def _phase_impact_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Identify and assess adverse impacts (Art 6-7)."""
        impacts = context.get("impacts", [])
        engagements = context.get("engagements", [])

        human_rights_impacts = [
            i for i in impacts if i.get("category") == "human_rights"
        ]
        environmental_impacts = [
            i for i in impacts if i.get("category") == "environmental"
        ]
        actual_impacts = [i for i in impacts if i.get("type") == "actual"]
        potential_impacts = [i for i in impacts if i.get("type") == "potential"]

        has_identification = len(impacts) > 0
        has_prioritisation = any(i.get("priority") for i in impacts)
        has_engagement = len(engagements) > 0

        score_components = [
            30.0 if has_identification else 0.0,
            30.0 if has_prioritisation else 0.0,
            20.0 if has_engagement else 0.0,
            20.0 if len(human_rights_impacts) > 0 and len(environmental_impacts) > 0 else 10.0,
        ]

        return {
            "total_impacts": len(impacts),
            "human_rights_impacts": len(human_rights_impacts),
            "environmental_impacts": len(environmental_impacts),
            "actual_impacts": len(actual_impacts),
            "potential_impacts": len(potential_impacts),
            "stakeholder_engagements": len(engagements),
            "has_identification": has_identification,
            "has_prioritisation": has_prioritisation,
            "has_engagement": has_engagement,
            "phase_score": round(sum(score_components), 1),
            "records_processed": len(impacts) + len(engagements),
        }

    def _phase_prevention_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Review prevention and mitigation measures (Art 5, 8-9, 13)."""
        measures = context.get("measures", [])

        prevention_measures = [m for m in measures if m.get("type") == "prevention"]
        mitigation_measures = [m for m in measures if m.get("type") == "mitigation"]
        cessation_measures = [m for m in measures if m.get("type") == "cessation"]

        has_policy_integration = any(
            m.get("policy_integrated", False) for m in measures
        )
        has_monitoring = any(m.get("has_monitoring", False) for m in measures)
        has_contractual_clauses = any(
            m.get("contractual_clauses", False) for m in measures
        )

        score_components = [
            20.0 if len(prevention_measures) > 0 else 0.0,
            20.0 if len(mitigation_measures) > 0 else 0.0,
            15.0 if len(cessation_measures) > 0 else 0.0,
            15.0 if has_policy_integration else 0.0,
            15.0 if has_monitoring else 0.0,
            15.0 if has_contractual_clauses else 0.0,
        ]

        return {
            "total_measures": len(measures),
            "prevention_measures": len(prevention_measures),
            "mitigation_measures": len(mitigation_measures),
            "cessation_measures": len(cessation_measures),
            "has_policy_integration": has_policy_integration,
            "has_monitoring": has_monitoring,
            "has_contractual_clauses": has_contractual_clauses,
            "phase_score": round(sum(score_components), 1),
            "records_processed": len(measures),
        }

    def _phase_grievance_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Assess grievance mechanisms and remediation (Art 10, 12)."""
        cases = context.get("cases", [])

        resolved_cases = [c for c in cases if c.get("status") == "resolved"]
        pending_cases = [c for c in cases if c.get("status") == "pending"]
        remediated_cases = [c for c in cases if c.get("remediation_provided", False)]

        has_mechanism = len(cases) > 0 or context.get("has_grievance_mechanism", False)
        has_remediation_policy = any(
            c.get("remediation_policy", False) for c in cases
        ) or context.get("has_remediation_policy", False)

        resolution_rate = (
            round(len(resolved_cases) / len(cases) * 100, 1) if cases else 0.0
        )

        score_components = [
            30.0 if has_mechanism else 0.0,
            25.0 if has_remediation_policy else 0.0,
            25.0 if resolution_rate >= 75.0 else (15.0 if resolution_rate >= 50.0 else 0.0),
            20.0 if len(remediated_cases) > 0 else 0.0,
        ]

        return {
            "total_cases": len(cases),
            "resolved_cases": len(resolved_cases),
            "pending_cases": len(pending_cases),
            "remediated_cases": len(remediated_cases),
            "resolution_rate_pct": resolution_rate,
            "has_mechanism": has_mechanism,
            "has_remediation_policy": has_remediation_policy,
            "phase_score": round(sum(score_components), 1),
            "records_processed": len(cases),
        }

    def _phase_climate_transition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Evaluate climate transition plan (Art 22)."""
        targets = context.get("targets", [])

        has_paris_alignment = any(
            t.get("paris_aligned", False) for t in targets
        )
        has_interim_targets = any(
            t.get("is_interim", False) for t in targets
        )
        has_implementation_plan = any(
            t.get("has_implementation_actions", False) for t in targets
        )
        has_decarbonization_levers = any(
            t.get("has_decarbonization_levers", False) for t in targets
        )
        has_financial_plan = any(
            t.get("has_financial_plan", False) for t in targets
        )

        score_components = [
            25.0 if has_paris_alignment else 0.0,
            20.0 if has_interim_targets else 0.0,
            20.0 if has_implementation_plan else 0.0,
            20.0 if has_decarbonization_levers else 0.0,
            15.0 if has_financial_plan else 0.0,
        ]

        return {
            "total_targets": len(targets),
            "has_paris_alignment": has_paris_alignment,
            "has_interim_targets": has_interim_targets,
            "has_implementation_plan": has_implementation_plan,
            "has_decarbonization_levers": has_decarbonization_levers,
            "has_financial_plan": has_financial_plan,
            "phase_score": round(sum(score_components), 1),
            "records_processed": len(targets),
        }

    def _phase_liability_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Assess civil liability exposure (Art 26)."""
        prevention_result = context.get("prevention_review_result", {})
        grievance_result = context.get("grievance_assessment_result", {})

        prevention_score = prevention_result.get("phase_score", 0.0)
        grievance_score = grievance_result.get("phase_score", 0.0)

        # Deterministic liability risk scoring
        combined_score = (prevention_score + grievance_score) / 2.0
        if combined_score >= 80.0:
            liability_risk = "low"
        elif combined_score >= 50.0:
            liability_risk = "medium"
        else:
            liability_risk = "high"

        has_d_and_o_insurance = context.get("has_d_and_o_insurance", False)
        has_legal_review = context.get("has_legal_review", False)

        score_components = [
            40.0 if combined_score >= 60.0 else 20.0,
            20.0 if has_d_and_o_insurance else 0.0,
            20.0 if has_legal_review else 0.0,
            20.0 if liability_risk == "low" else (10.0 if liability_risk == "medium" else 0.0),
        ]

        return {
            "liability_risk": liability_risk,
            "prevention_adequacy_score": prevention_score,
            "grievance_adequacy_score": grievance_score,
            "combined_due_diligence_score": round(combined_score, 1),
            "has_d_and_o_insurance": has_d_and_o_insurance,
            "has_legal_review": has_legal_review,
            "phase_score": round(sum(score_components), 1),
            "records_processed": 1,
        }

    def _phase_scorecard_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 7: Generate final CSDDD readiness scorecard (Art 14)."""
        phase_scores: Dict[str, float] = {}
        for phase in PHASE_EXECUTION_ORDER:
            if phase == CSDDDPhase.SCORECARD_GENERATION:
                continue
            result_key = f"{phase.value}_result"
            phase_data = context.get(result_key, {})
            phase_scores[phase.value] = phase_data.get("phase_score", 0.0)

        # Weighted average: scope(5%), impact(20%), prevention(25%),
        # grievance(15%), climate(20%), liability(15%)
        weights = {
            CSDDDPhase.SCOPE_DETERMINATION.value: 0.05,
            CSDDDPhase.IMPACT_ASSESSMENT.value: 0.20,
            CSDDDPhase.PREVENTION_REVIEW.value: 0.25,
            CSDDDPhase.GRIEVANCE_ASSESSMENT.value: 0.15,
            CSDDDPhase.CLIMATE_TRANSITION.value: 0.20,
            CSDDDPhase.LIABILITY_ASSESSMENT.value: 0.15,
        }

        weighted_score = sum(
            phase_scores.get(phase, 0.0) * weight
            for phase, weight in weights.items()
        )

        return {
            "phase_scores": phase_scores,
            "weighted_readiness_score": round(weighted_score, 1),
            "phase_weights": weights,
            "articles_referenced": list(CSDDD_ARTICLES.keys()),
            "phase_score": round(weighted_score, 1),
            "records_processed": len(phase_scores),
        }

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _compute_readiness_score(self, result: AssessmentResult) -> float:
        """Compute overall readiness score from phase results."""
        scorecard = result.phase_results.get(CSDDDPhase.SCORECARD_GENERATION.value)
        if scorecard and scorecard.outputs:
            return scorecard.outputs.get("weighted_readiness_score", 0.0)

        total = len(PHASE_EXECUTION_ORDER) - 1  # exclude scorecard itself
        completed = len(result.phases_completed)
        if total == 0:
            return 0.0
        return round(completed / total * 100, 1)

    def _score_to_level(self, score: float) -> ReadinessLevel:
        """Map a numeric score to a readiness level."""
        if score >= 90.0:
            return ReadinessLevel.FULLY_READY
        if score >= 70.0:
            return ReadinessLevel.ADVANCED
        if score >= 50.0:
            return ReadinessLevel.DEVELOPING
        if score >= 25.0:
            return ReadinessLevel.EARLY_STAGE
        return ReadinessLevel.NOT_READY

    def _identify_gaps(self, result: AssessmentResult) -> List[str]:
        """Identify key gaps from phase results."""
        gaps: List[str] = []
        for phase_name, phase_result in result.phase_results.items():
            score = phase_result.outputs.get("phase_score", 0.0)
            if score < 50.0 and phase_name != CSDDDPhase.SCORECARD_GENERATION.value:
                gaps.append(
                    f"{phase_name}: score {score}% - needs improvement"
                )

        for phase in PHASE_EXECUTION_ORDER:
            if phase.value in result.phases_skipped:
                gaps.append(f"{phase.value}: not assessed")

        return gaps

    def _generate_priority_actions(
        self,
        result: AssessmentResult,
        gaps: List[str],
    ) -> List[str]:
        """Generate priority actions based on identified gaps."""
        actions: List[str] = []

        phase_map: Dict[str, str] = {
            CSDDDPhase.IMPACT_ASSESSMENT.value: (
                "Conduct comprehensive adverse impact identification "
                "covering human rights and environmental impacts (Art 6-7)"
            ),
            CSDDDPhase.PREVENTION_REVIEW.value: (
                "Develop and integrate prevention/mitigation measures into "
                "company policies with monitoring mechanisms (Art 5, 8-9, 13)"
            ),
            CSDDDPhase.GRIEVANCE_ASSESSMENT.value: (
                "Establish or strengthen grievance mechanism with "
                "clear remediation procedures (Art 10, 12)"
            ),
            CSDDDPhase.CLIMATE_TRANSITION.value: (
                "Develop Paris-aligned climate transition plan with "
                "interim targets and financial planning (Art 22)"
            ),
            CSDDDPhase.LIABILITY_ASSESSMENT.value: (
                "Review civil liability exposure and ensure "
                "adequate insurance and legal safeguards (Art 26)"
            ),
        }

        for gap in gaps:
            phase_name = gap.split(":")[0].strip()
            action = phase_map.get(phase_name)
            if action and action not in actions:
                actions.append(action)

        return actions
