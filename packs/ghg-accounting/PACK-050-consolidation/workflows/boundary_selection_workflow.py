# -*- coding: utf-8 -*-
"""
Boundary Selection Workflow
====================================

4-phase workflow for selecting the GHG consolidation approach per GHG Protocol
Corporate Standard Chapter 3 within PACK-050 GHG Consolidation Pack.

Phases:
    1. ApproachEvaluation    -- Evaluate suitability of each consolidation
                                approach (equity share, operational control,
                                financial control) against the organisation's
                                corporate structure.
    2. ImpactAnalysis        -- Calculate the impact of each approach on
                                consolidated emissions totals to understand
                                reporting implications.
    3. StakeholderApproval   -- Present approach options with impact analysis
                                to stakeholders for approval decision.
    4. BoundaryLock          -- Lock the selected approach for the reporting
                                period with SHA-256 provenance and audit trail.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3) -- Setting Organisational Boundaries
    ISO 14064-1:2018 (Cl. 5.1) -- Organisational boundaries
    CSRD / ESRS E1 -- Climate change disclosure
    IFRS S2 -- Climate-related disclosures (boundary considerations)

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundarySelectionPhase(str, Enum):
    APPROACH_EVALUATION = "approach_evaluation"
    IMPACT_ANALYSIS = "impact_analysis"
    STAKEHOLDER_APPROVAL = "stakeholder_approval"
    BOUNDARY_LOCK = "boundary_lock"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class ApproachSuitability(str, Enum):
    HIGHLY_SUITABLE = "highly_suitable"
    SUITABLE = "suitable"
    PARTIALLY_SUITABLE = "partially_suitable"
    NOT_SUITABLE = "not_suitable"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    CONDITIONAL = "conditional"


class BoundaryLockStatus(str, Enum):
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    PENDING_APPROVAL = "pending_approval"


# =============================================================================
# REFERENCE DATA
# =============================================================================

APPROACH_DESCRIPTIONS: Dict[str, str] = {
    "equity_share": (
        "Accounts for GHG emissions proportional to the equity share "
        "in each operation. Recommended when JVs are significant."
    ),
    "financial_control": (
        "Accounts for 100% of GHG emissions from operations where the "
        "company has financial control (ability to direct financial and "
        "operating policies). Aligned with financial reporting."
    ),
    "operational_control": (
        "Accounts for 100% of GHG emissions from operations where the "
        "company has operational control (authority to introduce operating "
        "policies). Most commonly used approach."
    ),
}

SUITABILITY_SCORES: Dict[str, Decimal] = {
    "highly_suitable": Decimal("1.00"),
    "suitable": Decimal("0.75"),
    "partially_suitable": Decimal("0.50"),
    "not_suitable": Decimal("0.25"),
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EntitySummary(BaseModel):
    """Summary of an entity for approach evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(...)
    entity_name: str = Field("")
    ownership_pct: Decimal = Field(Decimal("100.00"))
    has_operational_control: bool = Field(False)
    has_financial_control: bool = Field(False)
    estimated_emissions_tco2e: Decimal = Field(Decimal("0"))


class ApproachEvaluation(BaseModel):
    """Evaluation of a single consolidation approach."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    approach: ConsolidationApproach = Field(...)
    description: str = Field("")
    suitability: ApproachSuitability = Field(ApproachSuitability.SUITABLE)
    suitability_score: Decimal = Field(Decimal("0.75"))
    entities_included: int = Field(0)
    entities_excluded: int = Field(0)
    jv_treatment: str = Field("", description="How JVs are treated under this approach")
    regulatory_alignment: List[str] = Field(
        default_factory=list, description="Aligned regulatory frameworks"
    )
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    recommendation_notes: str = Field("")


class ImpactAnalysisResult(BaseModel):
    """Impact analysis for a single approach."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    approach: ConsolidationApproach = Field(...)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    entities_reporting: int = Field(0)
    coverage_pct: Decimal = Field(Decimal("0"), description="Coverage of total org emissions")
    variance_from_equity: Decimal = Field(Decimal("0"), description="Variance vs equity share approach")


class StakeholderVote(BaseModel):
    """A stakeholder approval vote record."""
    stakeholder_id: str = Field(default_factory=_new_uuid)
    stakeholder_name: str = Field("")
    role: str = Field("")
    decision: ApprovalDecision = Field(ApprovalDecision.APPROVED)
    selected_approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    comments: str = Field("")
    voted_at: str = Field(default_factory=lambda: _utcnow().isoformat())


class BoundaryLockRecord(BaseModel):
    """Final boundary lock record."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lock_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    selected_approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    approval_decision: ApprovalDecision = Field(ApprovalDecision.APPROVED)
    lock_status: BoundaryLockStatus = Field(BoundaryLockStatus.LOCKED)
    entities_in_boundary: int = Field(0)
    total_emissions_tco2e: Decimal = Field(Decimal("0"))
    locked_at: str = Field("")
    locked_by: str = Field("")
    provenance_hash: str = Field("")


class BoundarySelectionInput(BaseModel):
    """Input for the boundary selection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    organisation_name: str = Field("")
    reporting_year: int = Field(...)
    entity_summaries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entity data for approach evaluation"
    )
    entity_emissions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entity-level emissions for impact analysis"
    )
    stakeholder_votes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Stakeholder approval votes"
    )
    preferred_approach: Optional[str] = Field(
        None, description="Pre-selected preferred approach"
    )
    skip_phases: List[str] = Field(default_factory=list)


class BoundarySelectionResult(BaseModel):
    """Output from the boundary selection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    approach_evaluations: List[ApproachEvaluation] = Field(default_factory=list)
    impact_analyses: List[ImpactAnalysisResult] = Field(default_factory=list)
    selected_approach: Optional[ConsolidationApproach] = Field(None)
    boundary_lock: Optional[BoundaryLockRecord] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class BoundarySelectionWorkflow:
    """
    4-phase boundary selection workflow for GHG consolidation.

    Evaluates consolidation approaches, analyses their impact on emissions,
    collects stakeholder approval, and locks the selected approach with
    SHA-256 provenance.

    Example:
        >>> wf = BoundarySelectionWorkflow()
        >>> inp = BoundarySelectionInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     entity_summaries=[{"entity_id": "E1", "ownership_pct": "60"}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.selected_approach is not None
    """

    PHASE_ORDER: List[BoundarySelectionPhase] = [
        BoundarySelectionPhase.APPROACH_EVALUATION,
        BoundarySelectionPhase.IMPACT_ANALYSIS,
        BoundarySelectionPhase.STAKEHOLDER_APPROVAL,
        BoundarySelectionPhase.BOUNDARY_LOCK,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._entities: List[EntitySummary] = []
        self._evaluations: Dict[ConsolidationApproach, ApproachEvaluation] = {}
        self._impacts: Dict[ConsolidationApproach, ImpactAnalysisResult] = {}
        self._approved_approach: Optional[ConsolidationApproach] = None

    def execute(self, input_data: BoundarySelectionInput) -> BoundarySelectionResult:
        """Execute the full 4-phase boundary selection workflow."""
        start = _utcnow()
        result = BoundarySelectionResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            BoundarySelectionPhase.APPROACH_EVALUATION: self._phase_approach_evaluation,
            BoundarySelectionPhase.IMPACT_ANALYSIS: self._phase_impact_analysis,
            BoundarySelectionPhase.STAKEHOLDER_APPROVAL: self._phase_stakeholder_approval,
            BoundarySelectionPhase.BOUNDARY_LOCK: self._phase_boundary_lock,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=ph_hash,
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value} failed: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{result.selected_approach}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- APPROACH EVALUATION
    # -----------------------------------------------------------------

    def _phase_approach_evaluation(
        self, input_data: BoundarySelectionInput, result: BoundarySelectionResult,
    ) -> Dict[str, Any]:
        """
        Evaluate suitability of each consolidation approach against
        the organisation's corporate structure.
        """
        logger.info("Phase 1 -- Approach Evaluation: %d entities", len(input_data.entity_summaries))

        # Parse entity summaries
        entities: List[EntitySummary] = []
        for raw in input_data.entity_summaries:
            es = EntitySummary(
                entity_id=raw.get("entity_id", _new_uuid()),
                entity_name=raw.get("entity_name", ""),
                ownership_pct=self._dec(raw.get("ownership_pct", "100")),
                has_operational_control=bool(raw.get("has_operational_control", False)),
                has_financial_control=bool(raw.get("has_financial_control", False)),
                estimated_emissions_tco2e=self._dec(raw.get("estimated_emissions_tco2e", "0")),
            )
            entities.append(es)
        self._entities = entities

        jv_count = sum(1 for e in entities if e.ownership_pct < Decimal("100"))
        evaluations: Dict[ConsolidationApproach, ApproachEvaluation] = {}

        for approach in ConsolidationApproach:
            eval_result = self._evaluate_single_approach(approach, entities, jv_count)
            evaluations[approach] = eval_result

        self._evaluations = evaluations
        result.approach_evaluations = list(evaluations.values())

        # Determine recommended approach
        best = max(evaluations.values(), key=lambda e: e.suitability_score)
        logger.info("Evaluation complete: recommended %s (score %.2f)",
                     best.approach.value, float(best.suitability_score))

        return {
            "approaches_evaluated": len(evaluations),
            "recommended_approach": best.approach.value,
            "recommended_score": float(best.suitability_score),
            "joint_ventures_count": jv_count,
        }

    def _evaluate_single_approach(
        self, approach: ConsolidationApproach,
        entities: List[EntitySummary], jv_count: int,
    ) -> ApproachEvaluation:
        """Evaluate a single consolidation approach deterministically."""
        included = 0
        excluded = 0
        pros: List[str] = []
        cons: List[str] = []

        for entity in entities:
            if approach == ConsolidationApproach.EQUITY_SHARE:
                if entity.ownership_pct > Decimal("0"):
                    included += 1
                else:
                    excluded += 1
            elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                if entity.has_operational_control:
                    included += 1
                else:
                    excluded += 1
            else:  # financial control
                if entity.has_financial_control:
                    included += 1
                else:
                    excluded += 1

        # Suitability scoring
        score = Decimal("0.50")
        if approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            score += Decimal("0.20")
            pros.append("Most commonly used approach; clearest for operational decisions")
            pros.append("Aligned with operational management responsibility")
            if jv_count > 0:
                cons.append("JV emissions excluded if no operational control")
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            score += Decimal("0.15")
            pros.append("Aligned with financial reporting consolidation")
            pros.append("Consistent with IFRS consolidation rules")
            if jv_count > 0:
                cons.append("JV treatment may differ from operational view")
        else:  # equity share
            score += Decimal("0.10")
            pros.append("Proportional representation of JV emissions")
            pros.append("Reflects economic interest in emissions")
            cons.append("More complex calculation for partial ownership")

        if jv_count > 2:
            if approach == ConsolidationApproach.EQUITY_SHARE:
                score += Decimal("0.20")
            else:
                score -= Decimal("0.10")

        if included > excluded:
            score += Decimal("0.10")

        score = min(max(score, Decimal("0.25")), Decimal("1.00"))

        suitability = ApproachSuitability.NOT_SUITABLE
        if score >= Decimal("0.85"):
            suitability = ApproachSuitability.HIGHLY_SUITABLE
        elif score >= Decimal("0.65"):
            suitability = ApproachSuitability.SUITABLE
        elif score >= Decimal("0.45"):
            suitability = ApproachSuitability.PARTIALLY_SUITABLE

        jv_treatment = {
            ConsolidationApproach.EQUITY_SHARE: "Report proportional to equity interest",
            ConsolidationApproach.OPERATIONAL_CONTROL: "Include 100% if operational control, else exclude",
            ConsolidationApproach.FINANCIAL_CONTROL: "Include 100% if financial control, else exclude",
        }

        return ApproachEvaluation(
            approach=approach,
            description=APPROACH_DESCRIPTIONS.get(approach.value, ""),
            suitability=suitability,
            suitability_score=score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            entities_included=included,
            entities_excluded=excluded,
            jv_treatment=jv_treatment.get(approach, ""),
            regulatory_alignment=self._get_regulatory_alignment(approach),
            pros=pros,
            cons=cons,
        )

    def _get_regulatory_alignment(self, approach: ConsolidationApproach) -> List[str]:
        """Return aligned regulatory frameworks for the approach."""
        base = ["GHG Protocol Corporate Standard", "ISO 14064-1:2018"]
        if approach == ConsolidationApproach.FINANCIAL_CONTROL:
            base.extend(["IFRS S2", "CSRD / ESRS E1"])
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            base.extend(["CSRD / ESRS E1", "CDP Climate Change"])
        else:
            base.extend(["GHG Protocol Scope 2 Guidance"])
        return base

    # -----------------------------------------------------------------
    # PHASE 2 -- IMPACT ANALYSIS
    # -----------------------------------------------------------------

    def _phase_impact_analysis(
        self, input_data: BoundarySelectionInput, result: BoundarySelectionResult,
    ) -> Dict[str, Any]:
        """
        Calculate the impact of each consolidation approach on
        consolidated emissions totals.
        """
        logger.info("Phase 2 -- Impact Analysis")

        # Parse entity-level emissions
        emissions_map: Dict[str, Dict[str, Decimal]] = {}
        for raw in input_data.entity_emissions:
            eid = raw.get("entity_id", "")
            if eid:
                emissions_map[eid] = {
                    "scope_1": self._dec(raw.get("scope_1_tco2e", "0")),
                    "scope_2_location": self._dec(raw.get("scope_2_location_tco2e", "0")),
                    "scope_2_market": self._dec(raw.get("scope_2_market_tco2e", "0")),
                    "scope_3": self._dec(raw.get("scope_3_tco2e", "0")),
                }

        # Calculate equity share baseline first
        equity_total = Decimal("0")
        for entity in self._entities:
            em = emissions_map.get(entity.entity_id, {})
            factor = entity.ownership_pct / Decimal("100")
            entity_total = sum(em.values()) * factor
            equity_total += entity_total

        impacts: Dict[ConsolidationApproach, ImpactAnalysisResult] = {}

        for approach in ConsolidationApproach:
            s1 = Decimal("0")
            s2l = Decimal("0")
            s2m = Decimal("0")
            s3 = Decimal("0")
            reporting_count = 0

            for entity in self._entities:
                em = emissions_map.get(entity.entity_id, {})
                factor = self._get_reporting_factor(approach, entity)

                if factor > Decimal("0"):
                    reporting_count += 1

                s1 += em.get("scope_1", Decimal("0")) * factor
                s2l += em.get("scope_2_location", Decimal("0")) * factor
                s2m += em.get("scope_2_market", Decimal("0")) * factor
                s3 += em.get("scope_3", Decimal("0")) * factor

            total_loc = (s1 + s2l + s3).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Coverage
            total_possible = sum(
                sum(emissions_map.get(e.entity_id, {}).values())
                for e in self._entities
            )
            coverage = Decimal("0")
            if total_possible > Decimal("0"):
                coverage = (total_loc / total_possible * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            # Variance from equity
            variance = Decimal("0")
            if equity_total > Decimal("0"):
                variance = ((total_loc - equity_total) / equity_total * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            impacts[approach] = ImpactAnalysisResult(
                approach=approach,
                scope_1_tco2e=s1.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                scope_2_location_tco2e=s2l.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                scope_2_market_tco2e=s2m.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                scope_3_tco2e=s3.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                total_location_tco2e=total_loc,
                entities_reporting=reporting_count,
                coverage_pct=coverage,
                variance_from_equity=variance,
            )

        self._impacts = impacts
        result.impact_analyses = list(impacts.values())

        logger.info("Impact analysis: %d approaches analysed", len(impacts))
        return {
            "approaches_analysed": len(impacts),
            "equity_share_total": float(impacts.get(
                ConsolidationApproach.EQUITY_SHARE,
                ImpactAnalysisResult(approach=ConsolidationApproach.EQUITY_SHARE)
            ).total_location_tco2e),
            "operational_control_total": float(impacts.get(
                ConsolidationApproach.OPERATIONAL_CONTROL,
                ImpactAnalysisResult(approach=ConsolidationApproach.OPERATIONAL_CONTROL)
            ).total_location_tco2e),
            "financial_control_total": float(impacts.get(
                ConsolidationApproach.FINANCIAL_CONTROL,
                ImpactAnalysisResult(approach=ConsolidationApproach.FINANCIAL_CONTROL)
            ).total_location_tco2e),
        }

    def _get_reporting_factor(
        self, approach: ConsolidationApproach, entity: EntitySummary
    ) -> Decimal:
        """Get the reporting factor for an entity under a given approach."""
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return (entity.ownership_pct / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            return Decimal("1.0") if entity.has_operational_control else Decimal("0")
        else:
            return Decimal("1.0") if entity.has_financial_control else Decimal("0")

    # -----------------------------------------------------------------
    # PHASE 3 -- STAKEHOLDER APPROVAL
    # -----------------------------------------------------------------

    def _phase_stakeholder_approval(
        self, input_data: BoundarySelectionInput, result: BoundarySelectionResult,
    ) -> Dict[str, Any]:
        """
        Present approach options to stakeholders and collect approval.
        """
        logger.info("Phase 3 -- Stakeholder Approval")

        votes: List[StakeholderVote] = []
        for raw in input_data.stakeholder_votes:
            try:
                decision = ApprovalDecision(raw.get("decision", "approved"))
            except ValueError:
                decision = ApprovalDecision.DEFERRED

            try:
                approach = ConsolidationApproach(raw.get("selected_approach", "operational_control"))
            except ValueError:
                approach = ConsolidationApproach.OPERATIONAL_CONTROL

            vote = StakeholderVote(
                stakeholder_name=raw.get("stakeholder_name", ""),
                role=raw.get("role", ""),
                decision=decision,
                selected_approach=approach,
                comments=raw.get("comments", ""),
            )
            votes.append(vote)

        # Tally votes
        approach_votes: Dict[str, int] = {}
        approved_count = 0
        rejected_count = 0

        for vote in votes:
            if vote.decision == ApprovalDecision.APPROVED:
                approved_count += 1
                key = vote.selected_approach.value
                approach_votes[key] = approach_votes.get(key, 0) + 1
            elif vote.decision == ApprovalDecision.REJECTED:
                rejected_count += 1

        # Determine winning approach
        if approach_votes:
            winning_approach_str = max(approach_votes, key=approach_votes.get)  # type: ignore[arg-type]
            self._approved_approach = ConsolidationApproach(winning_approach_str)
        elif input_data.preferred_approach:
            try:
                self._approved_approach = ConsolidationApproach(input_data.preferred_approach)
            except ValueError:
                self._approved_approach = ConsolidationApproach.OPERATIONAL_CONTROL
        else:
            # Default: highest suitability score
            if self._evaluations:
                best = max(self._evaluations.values(), key=lambda e: e.suitability_score)
                self._approved_approach = best.approach
            else:
                self._approved_approach = ConsolidationApproach.OPERATIONAL_CONTROL

        result.selected_approach = self._approved_approach

        if rejected_count > approved_count and votes:
            result.warnings.append(
                "More rejections than approvals -- selected approach may need review"
            )

        logger.info("Stakeholder approval: %s selected with %d votes",
                     self._approved_approach.value, approach_votes.get(self._approved_approach.value, 0))
        return {
            "total_votes": len(votes),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "selected_approach": self._approved_approach.value,
            "vote_distribution": approach_votes,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- BOUNDARY LOCK
    # -----------------------------------------------------------------

    def _phase_boundary_lock(
        self, input_data: BoundarySelectionInput, result: BoundarySelectionResult,
    ) -> Dict[str, Any]:
        """
        Lock the selected consolidation approach for the reporting period.
        """
        logger.info("Phase 4 -- Boundary Lock")
        now_iso = _utcnow().isoformat()

        if self._approved_approach is None:
            self._approved_approach = ConsolidationApproach.OPERATIONAL_CONTROL

        # Count entities in boundary
        entities_in = 0
        total_emissions = Decimal("0")
        impact = self._impacts.get(self._approved_approach)
        if impact:
            entities_in = impact.entities_reporting
            total_emissions = impact.total_location_tco2e

        prov_input = (
            f"{input_data.organisation_id}|{input_data.reporting_year}|"
            f"{self._approved_approach.value}|{entities_in}|"
            f"{float(total_emissions)}|{now_iso}"
        )
        prov_hash = _compute_hash(prov_input)

        lock = BoundaryLockRecord(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            selected_approach=self._approved_approach,
            approval_decision=ApprovalDecision.APPROVED,
            lock_status=BoundaryLockStatus.LOCKED,
            entities_in_boundary=entities_in,
            total_emissions_tco2e=total_emissions,
            locked_at=now_iso,
            locked_by=self.config.get("locked_by", "system"),
            provenance_hash=prov_hash,
        )
        result.boundary_lock = lock

        logger.info("Boundary locked: %s, %d entities, %.2f tCO2e",
                     self._approved_approach.value, entities_in, float(total_emissions))
        return {
            "selected_approach": self._approved_approach.value,
            "entities_in_boundary": entities_in,
            "total_emissions_tco2e": float(total_emissions),
            "lock_status": BoundaryLockStatus.LOCKED.value,
            "provenance_hash": prov_hash,
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BoundarySelectionWorkflow",
    "BoundarySelectionInput",
    "BoundarySelectionResult",
    "BoundarySelectionPhase",
    "ConsolidationApproach",
    "ApproachSuitability",
    "ApprovalDecision",
    "BoundaryLockStatus",
    "EntitySummary",
    "ApproachEvaluation",
    "ImpactAnalysisResult",
    "StakeholderVote",
    "BoundaryLockRecord",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
