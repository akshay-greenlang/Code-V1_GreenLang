# -*- coding: utf-8 -*-
"""
Claim Assessment Workflow - PACK-018 EU Green Claims Prep
==========================================================

5-phase workflow that evaluates environmental claims against the EU Green
Claims Directive requirements. The pipeline ingests raw claim data and
supporting evidence, classifies each claim by type and scope, runs a
substantiation assessment to verify whether the evidence base is
sufficient, assigns risk scores, and finally generates a structured
assessment report with provenance tracking.

Phases:
    1. DataIntake               -- Ingest and normalise claim/evidence data
    2. ClaimClassification      -- Classify claims by type, scope, specificity
    3. SubstantiationAssessment -- Verify evidence sufficiency per claim
    4. RiskScoring              -- Score residual greenwashing risk
    5. ReportGeneration         -- Produce the final assessment report

Reference:
    EU Green Claims Directive (COM/2023/166)
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ClaimType(str, Enum):
    """EU Green Claims Directive claim classification."""
    ENVIRONMENTAL = "environmental"
    CARBON = "carbon"
    BIODIVERSITY = "biodiversity"
    CIRCULAR = "circular"
    WATER = "water"
    GENERIC = "generic"


class RiskLevel(str, Enum):
    """Greenwashing risk tier."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# DATA MODELS
# =============================================================================


class WorkflowInput(BaseModel):
    """Input model for ClaimAssessmentWorkflow."""
    claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of environmental claim objects to assess",
    )
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting evidence items linked to claims",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    config: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class WorkflowResult(BaseModel):
    """Complete result from ClaimAssessmentWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="claim_assessment")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ClaimAssessmentWorkflow:
    """
    5-phase claim assessment workflow for the EU Green Claims Directive.

    Evaluates environmental claims against Directive requirements, classifies
    them by type and specificity, assesses evidence sufficiency, assigns risk
    scores, and generates a structured report.

    Zero-hallucination: all scoring and aggregation uses deterministic
    arithmetic. No LLM calls in numeric calculation paths.

    Example:
        >>> wf = ClaimAssessmentWorkflow()
        >>> result = wf.execute(
        ...     claims=[{"id": "C1", "text": "100% recyclable packaging"}],
        ...     evidence=[{"claim_id": "C1", "type": "lca_report"}],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "claim_assessment"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClaimAssessmentWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 5-phase claim assessment pipeline.

        Keyword Args:
            claims: List of claim dictionaries.
            evidence: List of evidence dictionaries.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            claims=kwargs.get("claims", []),
            evidence=kwargs.get("evidence", []),
            entity_name=kwargs.get("entity_name", ""),
            reporting_year=kwargs.get("reporting_year", 2025),
            config=kwargs.get("config", {}),
        )

        started_at = _utcnow()
        self.logger.info("Starting %s workflow %s", self.WORKFLOW_NAME, self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Data Intake
            phase_results.append(self._phase_data_intake(input_data))

            # Phase 2 -- Claim Classification
            phase_results.append(self._phase_claim_classification(input_data))

            # Phase 3 -- Substantiation Assessment
            classification_data = phase_results[1].result_data
            phase_results.append(
                self._phase_substantiation_assessment(input_data, classification_data)
            )

            # Phase 4 -- Risk Scoring
            substantiation_data = phase_results[2].result_data
            phase_results.append(
                self._phase_risk_scoring(input_data, substantiation_data)
            )

            # Phase 5 -- Report Generation
            risk_data = phase_results[3].result_data
            phase_results.append(
                self._phase_report_generation(input_data, phase_results)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=_utcnow(),
                completed_at=_utcnow(),
                error_message=str(exc),
            ))

        completed_at = _utcnow()

        # Build overall result summary
        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_claims": len(input_data.claims),
            "total_evidence": len(input_data.evidence),
            "phases_completed": len(completed_phases),
            "phases_total": 5,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Workflow %s %s in %.1fs -- %d claims assessed",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.claims),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Phase 1: Data Intake
    # ------------------------------------------------------------------

    def _phase_data_intake(self, input_data: WorkflowInput) -> PhaseResult:
        """Ingest and normalise claim and evidence data."""
        started = _utcnow()
        self.logger.info("Phase 1/5 DataIntake -- ingesting %d claims, %d evidence items",
                         len(input_data.claims), len(input_data.evidence))

        warnings: List[str] = []
        claim_ids = {c.get("id", f"auto-{i}") for i, c in enumerate(input_data.claims)}
        evidence_claim_ids = {e.get("claim_id") for e in input_data.evidence}

        orphaned_evidence = evidence_claim_ids - claim_ids
        uncovered_claims = claim_ids - evidence_claim_ids

        if orphaned_evidence:
            warnings.append(f"{len(orphaned_evidence)} evidence items reference unknown claims")
        if uncovered_claims:
            warnings.append(f"{len(uncovered_claims)} claims have no linked evidence")

        result_data: Dict[str, Any] = {
            "claims_ingested": len(input_data.claims),
            "evidence_ingested": len(input_data.evidence),
            "orphaned_evidence_count": len(orphaned_evidence),
            "uncovered_claims_count": len(uncovered_claims),
            "warnings": warnings,
        }

        return PhaseResult(
            phase_name="DataIntake",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Claim Classification
    # ------------------------------------------------------------------

    def _phase_claim_classification(self, input_data: WorkflowInput) -> PhaseResult:
        """Classify each claim by type, scope, and specificity."""
        started = _utcnow()
        self.logger.info("Phase 2/5 ClaimClassification -- classifying %d claims",
                         len(input_data.claims))

        classifications: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {t.value: 0 for t in ClaimType}

        for idx, claim in enumerate(input_data.claims):
            claim_id = claim.get("id", f"auto-{idx}")
            claim_text = claim.get("text", "").lower()
            claim_type = self._classify_claim_type(claim_text)
            is_specific = self._is_specific_claim(claim_text)

            classifications.append({
                "claim_id": claim_id,
                "claim_type": claim_type.value,
                "is_specific": is_specific,
                "scope": claim.get("scope", "product"),
                "text_length": len(claim.get("text", "")),
            })
            type_counts[claim_type.value] += 1

        result_data: Dict[str, Any] = {
            "classifications": classifications,
            "type_distribution": type_counts,
            "specific_claims_count": sum(1 for c in classifications if c["is_specific"]),
            "generic_claims_count": sum(1 for c in classifications if not c["is_specific"]),
        }

        return PhaseResult(
            phase_name="ClaimClassification",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Substantiation Assessment
    # ------------------------------------------------------------------

    def _phase_substantiation_assessment(
        self,
        input_data: WorkflowInput,
        classification_data: Dict[str, Any],
    ) -> PhaseResult:
        """Verify evidence sufficiency for each classified claim."""
        started = _utcnow()
        self.logger.info("Phase 3/5 SubstantiationAssessment -- evaluating evidence")

        evidence_by_claim: Dict[str, List[Dict[str, Any]]] = {}
        for ev in input_data.evidence:
            cid = ev.get("claim_id", "unknown")
            evidence_by_claim.setdefault(cid, []).append(ev)

        assessments: List[Dict[str, Any]] = []
        substantiated_count = 0

        for cls_item in classification_data.get("classifications", []):
            claim_id = cls_item["claim_id"]
            claim_evidence = evidence_by_claim.get(claim_id, [])
            evidence_types = {e.get("type", "unknown") for e in claim_evidence}

            has_lca = "lca_report" in evidence_types or "lifecycle_assessment" in evidence_types
            has_third_party = "third_party_verification" in evidence_types
            has_test_data = "test_report" in evidence_types or "lab_test" in evidence_types

            score = self._calculate_substantiation_score(
                evidence_count=len(claim_evidence),
                has_lca=has_lca,
                has_third_party=has_third_party,
                has_test_data=has_test_data,
                is_specific=cls_item.get("is_specific", False),
            )

            is_substantiated = score >= 60.0
            if is_substantiated:
                substantiated_count += 1

            assessments.append({
                "claim_id": claim_id,
                "evidence_count": len(claim_evidence),
                "evidence_types": sorted(evidence_types),
                "substantiation_score": score,
                "is_substantiated": is_substantiated,
                "has_lca": has_lca,
                "has_third_party": has_third_party,
            })

        total_claims = len(assessments)
        result_data: Dict[str, Any] = {
            "assessments": assessments,
            "substantiated_count": substantiated_count,
            "unsubstantiated_count": total_claims - substantiated_count,
            "substantiation_rate_pct": round(
                (substantiated_count / total_claims * 100) if total_claims else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name="SubstantiationAssessment",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Risk Scoring
    # ------------------------------------------------------------------

    def _phase_risk_scoring(
        self,
        input_data: WorkflowInput,
        substantiation_data: Dict[str, Any],
    ) -> PhaseResult:
        """Assign greenwashing risk scores based on substantiation gaps."""
        started = _utcnow()
        self.logger.info("Phase 4/5 RiskScoring -- computing risk levels")

        risk_items: List[Dict[str, Any]] = []
        risk_distribution: Dict[str, int] = {r.value: 0 for r in RiskLevel}

        for assessment in substantiation_data.get("assessments", []):
            score = assessment.get("substantiation_score", 0.0)
            risk_level = self._determine_risk_level(score)
            risk_distribution[risk_level.value] += 1

            risk_items.append({
                "claim_id": assessment["claim_id"],
                "substantiation_score": score,
                "risk_level": risk_level.value,
                "requires_action": risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL),
            })

        total = len(risk_items)
        high_risk_count = risk_distribution.get("high", 0) + risk_distribution.get("critical", 0)

        result_data: Dict[str, Any] = {
            "risk_items": risk_items,
            "risk_distribution": risk_distribution,
            "high_risk_count": high_risk_count,
            "overall_risk_level": self._determine_overall_risk(risk_distribution, total),
        }

        return PhaseResult(
            phase_name="RiskScoring",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 5: Report Generation
    # ------------------------------------------------------------------

    def _phase_report_generation(
        self,
        input_data: WorkflowInput,
        all_phases: List[PhaseResult],
    ) -> PhaseResult:
        """Produce the final assessment report from all phase outputs."""
        started = _utcnow()
        self.logger.info("Phase 5/5 ReportGeneration -- assembling report")

        phase_summaries: Dict[str, Any] = {}
        for phase in all_phases:
            phase_summaries[phase.phase_name] = {
                "status": phase.status.value,
                "started_at": phase.started_at.isoformat() if phase.started_at else None,
                "completed_at": phase.completed_at.isoformat() if phase.completed_at else None,
            }

        # Aggregate from earlier phases
        risk_data = {}
        subst_data = {}
        for p in all_phases:
            if p.phase_name == "RiskScoring":
                risk_data = p.result_data
            elif p.phase_name == "SubstantiationAssessment":
                subst_data = p.result_data

        result_data: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "entity_name": input_data.entity_name,
            "reporting_year": input_data.reporting_year,
            "total_claims_assessed": len(input_data.claims),
            "substantiation_rate_pct": subst_data.get("substantiation_rate_pct", 0.0),
            "overall_risk_level": risk_data.get("overall_risk_level", "unknown"),
            "high_risk_count": risk_data.get("high_risk_count", 0),
            "phase_summaries": phase_summaries,
            "recommendation": self._generate_recommendation(
                subst_data.get("substantiation_rate_pct", 0.0),
                risk_data.get("overall_risk_level", "unknown"),
            ),
        }

        return PhaseResult(
            phase_name="ReportGeneration",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_claim_type(self, text: str) -> ClaimType:
        """Classify claim type from text content using keyword matching."""
        carbon_keywords = {"carbon", "co2", "emission", "ghg", "climate", "net zero", "neutral"}
        biodiversity_keywords = {"biodiversity", "species", "habitat", "ecosystem", "nature"}
        circular_keywords = {"recyclable", "recycled", "circular", "reuse", "compostable"}
        water_keywords = {"water", "aquatic", "marine", "ocean", "freshwater"}

        if any(kw in text for kw in carbon_keywords):
            return ClaimType.CARBON
        if any(kw in text for kw in biodiversity_keywords):
            return ClaimType.BIODIVERSITY
        if any(kw in text for kw in circular_keywords):
            return ClaimType.CIRCULAR
        if any(kw in text for kw in water_keywords):
            return ClaimType.WATER
        if any(kw in text for kw in {"green", "eco", "sustainable", "friendly"}):
            return ClaimType.GENERIC
        return ClaimType.ENVIRONMENTAL

    def _is_specific_claim(self, text: str) -> bool:
        """Determine whether a claim is specific (quantified) vs. vague."""
        specificity_indicators = {"%", "percent", "kg", "tonnes", "litre", "kwh", "mwh"}
        return any(ind in text for ind in specificity_indicators)

    def _calculate_substantiation_score(
        self,
        evidence_count: int,
        has_lca: bool,
        has_third_party: bool,
        has_test_data: bool,
        is_specific: bool,
    ) -> float:
        """Compute deterministic substantiation score (0-100)."""
        score = 0.0
        # Evidence quantity contribution (max 30)
        score += min(evidence_count * 10.0, 30.0)
        # LCA presence (25 points)
        if has_lca:
            score += 25.0
        # Third-party verification (25 points)
        if has_third_party:
            score += 25.0
        # Test data (10 points)
        if has_test_data:
            score += 10.0
        # Specificity bonus (10 points)
        if is_specific:
            score += 10.0
        return min(round(score, 1), 100.0)

    def _determine_risk_level(self, substantiation_score: float) -> RiskLevel:
        """Map substantiation score to risk level."""
        if substantiation_score >= 80.0:
            return RiskLevel.LOW
        if substantiation_score >= 60.0:
            return RiskLevel.MEDIUM
        if substantiation_score >= 30.0:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def _determine_overall_risk(
        self, distribution: Dict[str, int], total: int,
    ) -> str:
        """Determine aggregate risk level from distribution."""
        if total == 0:
            return RiskLevel.LOW.value
        critical_pct = (distribution.get("critical", 0) / total) * 100
        high_pct = (distribution.get("high", 0) / total) * 100
        if critical_pct > 10 or (critical_pct + high_pct) > 30:
            return RiskLevel.CRITICAL.value
        if (critical_pct + high_pct) > 15:
            return RiskLevel.HIGH.value
        if (critical_pct + high_pct) > 5:
            return RiskLevel.MEDIUM.value
        return RiskLevel.LOW.value

    def _generate_recommendation(
        self, substantiation_rate: float, overall_risk: str,
    ) -> str:
        """Generate a deterministic recommendation string."""
        if overall_risk == RiskLevel.CRITICAL.value:
            return (
                "URGENT: Multiple claims lack substantiation. Suspend use of "
                "unsubstantiated claims and initiate evidence collection immediately."
            )
        if overall_risk == RiskLevel.HIGH.value:
            return (
                "HIGH PRIORITY: Several claims require additional evidence. "
                "Commission third-party verification and LCA studies."
            )
        if substantiation_rate < 80.0:
            return (
                "MODERATE: Most claims are partially substantiated. Fill evidence "
                "gaps before the EU Green Claims Directive enforcement date."
            )
        return (
            "GOOD STANDING: Claims are well-substantiated. Maintain evidence "
            "freshness and schedule periodic reviews."
        )
