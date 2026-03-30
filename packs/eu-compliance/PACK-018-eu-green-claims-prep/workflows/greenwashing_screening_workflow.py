# -*- coding: utf-8 -*-
"""
Greenwashing Screening Workflow - PACK-018 EU Green Claims Prep
================================================================

4-phase workflow that screens an organisation's environmental claims for
greenwashing risk. Extracts the complete claim universe across channels,
evaluates claims against TerraChoice Seven Sins and UCPD criteria, scores
each claim on a 0-100 risk scale, and generates a prioritised action plan
with concrete remediation steps.

Phases:
    1. ClaimUniverseExtraction -- Extract claims from all channels
    2. SevenSinsAnalysis       -- Evaluate TerraChoice + UCPD patterns
    3. RiskScoring             -- Score 0-100 per claim
    4. ActionPlanGeneration    -- Generate prioritised remediation plan

Reference:
    EU Green Claims Directive (COM/2023/166)
    Empowering Consumers Directive (EU) 2024/825
    TerraChoice Seven Sins of Greenwashing
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

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

class ScreeningPhase(str, Enum):
    """Greenwashing screening workflow phase identifiers."""
    CLAIM_UNIVERSE_EXTRACTION = "ClaimUniverseExtraction"
    SEVEN_SINS_ANALYSIS = "SevenSinsAnalysis"
    RISK_SCORING = "RiskScoring"
    ACTION_PLAN_GENERATION = "ActionPlanGeneration"

class SevenSin(str, Enum):
    """TerraChoice Seven Sins of Greenwashing + UCPD extensions."""
    HIDDEN_TRADEOFF = "hidden_tradeoff"
    NO_PROOF = "no_proof"
    VAGUENESS = "vagueness"
    IRRELEVANCE = "irrelevance"
    LESSER_OF_TWO_EVILS = "lesser_of_two_evils"
    FIBBING = "fibbing"
    FALSE_LABELS = "false_labels"
    # UCPD extensions
    GENERIC_CLAIM = "generic_claim"
    FUTURE_PROMISE = "future_promise"
    OFFSET_RELIANCE = "offset_reliance"

class ClaimChannel(str, Enum):
    """Communication channel where claims appear."""
    WEBSITE = "website"
    PACKAGING = "packaging"
    ADVERTISING = "advertising"
    SOCIAL_MEDIA = "social_media"
    ANNUAL_REPORT = "annual_report"
    PRESS_RELEASE = "press_release"
    POINT_OF_SALE = "point_of_sale"
    OTHER = "other"

class RiskTier(str, Enum):
    """Risk tier classification for remediation prioritisation."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

# =============================================================================
# DATA MODELS
# =============================================================================

class ScreeningConfig(BaseModel):
    """Configuration for GreenwashingScreeningWorkflow."""
    critical_threshold: float = Field(
        default=80.0, ge=0.0, le=100.0,
        description="Risk score threshold for critical classification",
    )
    high_threshold: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Risk score threshold for high classification",
    )
    medium_threshold: float = Field(
        default=40.0, ge=0.0, le=100.0,
        description="Risk score threshold for medium classification",
    )
    low_threshold: float = Field(
        default=20.0, ge=0.0, le=100.0,
        description="Risk score threshold for low classification",
    )

class ScreeningResult(BaseModel):
    """Per-claim screening result."""
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., description="Original claim text")
    channel: str = Field(default="unknown", description="Source channel")
    sins_detected: List[str] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_tier: str = Field(default="negligible")
    remediation_actions: List[str] = Field(default_factory=list)
    priority_rank: int = Field(default=0, ge=0)

class WorkflowInput(BaseModel):
    """Input model for GreenwashingScreeningWorkflow."""
    claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of claim objects with text and channel metadata",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
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
    """Complete result from GreenwashingScreeningWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="greenwashing_screening")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class GreenwashingScreeningWorkflow:
    """
    4-phase greenwashing screening workflow for EU Green Claims Directive.

    Extracts the complete claim universe from all communication channels,
    evaluates each claim against TerraChoice Seven Sins and UCPD criteria,
    assigns a 0-100 risk score, and generates a prioritised action plan.

    Zero-hallucination: all pattern detection, scoring, and prioritisation
    uses deterministic keyword matching and rule-based logic. No LLM calls
    in calculation paths.

    Example:
        >>> wf = GreenwashingScreeningWorkflow()
        >>> result = wf.execute(
        ...     claims=[{"text": "100% eco-friendly", "channel": "packaging"}],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "greenwashing_screening"

    # Keyword patterns for each sin
    SIN_PATTERNS: Dict[str, set] = {
        SevenSin.VAGUENESS.value: {
            "eco-friendly", "green", "sustainable", "natural", "conscious",
            "environmentally friendly", "earth-friendly", "responsible",
            "eco", "clean", "pure", "gentle",
        },
        SevenSin.NO_PROOF.value: {
            "carbon neutral", "climate neutral", "net zero", "co2 free",
            "zero emissions", "emission free", "carbon free",
        },
        SevenSin.FUTURE_PROMISE.value: {
            "by 2030", "by 2035", "by 2040", "by 2050",
            "will be", "commitment to", "pledge to", "aim to",
            "working towards", "on track to",
        },
        SevenSin.OFFSET_RELIANCE.value: {
            "offset", "carbon credit", "compensation",
            "neutralise", "neutralize", "compensate",
        },
        SevenSin.GENERIC_CLAIM.value: {
            "planet-friendly", "better for the environment",
            "good for the planet", "saving the earth",
        },
        SevenSin.FIBBING.value: {
            "certified organic", "officially approved",
        },
        SevenSin.FALSE_LABELS.value: {
            "green seal", "eco-certified",
        },
        SevenSin.HIDDEN_TRADEOFF.value: {
            "made from recycled", "plant-based",
        },
        SevenSin.IRRELEVANCE.value: {
            "cfc-free", "cfc free", "lead-free paint",
        },
        SevenSin.LESSER_OF_TWO_EVILS.value: {
            "lighter cigarette", "fuel-efficient suv", "cleaner coal",
        },
    }

    # Severity weight per sin (used in risk scoring)
    SIN_WEIGHTS: Dict[str, float] = {
        SevenSin.FIBBING.value: 30.0,
        SevenSin.FALSE_LABELS.value: 28.0,
        SevenSin.NO_PROOF.value: 22.0,
        SevenSin.HIDDEN_TRADEOFF.value: 18.0,
        SevenSin.VAGUENESS.value: 15.0,
        SevenSin.GENERIC_CLAIM.value: 14.0,
        SevenSin.OFFSET_RELIANCE.value: 12.0,
        SevenSin.FUTURE_PROMISE.value: 10.0,
        SevenSin.IRRELEVANCE.value: 8.0,
        SevenSin.LESSER_OF_TWO_EVILS.value: 6.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GreenwashingScreeningWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.screening_config = ScreeningConfig(**self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 4-phase greenwashing screening pipeline.

        Keyword Args:
            claims: List of claim dicts with 'text' and optional 'channel'.
            entity_name: Organisation name.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            claims=kwargs.get("claims", []),
            entity_name=kwargs.get("entity_name", ""),
            config=kwargs.get("config", {}),
        )

        started_at = utcnow()
        self.logger.info("Starting %s workflow %s -- %d claims",
                         self.WORKFLOW_NAME, self.workflow_id, len(input_data.claims))
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Claim Universe Extraction
            phase_results.append(self._run_claim_universe_extraction(input_data))

            # Phase 2 -- Seven Sins Analysis
            universe = phase_results[0].result_data
            phase_results.append(self._run_seven_sins_analysis(universe))

            # Phase 3 -- Risk Scoring
            sins_data = phase_results[1].result_data
            phase_results.append(self._run_risk_scoring(sins_data))

            # Phase 4 -- Action Plan Generation
            risk_data = phase_results[2].result_data
            phase_results.append(self._run_action_plan_generation(risk_data))

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=utcnow(),
                completed_at=utcnow(),
                error_message=str(exc),
            ))

        completed_at = utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_claims_screened": len(input_data.claims),
            "phases_completed": len(completed_phases),
            "phases_total": 4,
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
            "Workflow %s %s in %.1fs -- %d claims screened",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.claims),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # run_phase dispatcher
    # ------------------------------------------------------------------

    def run_phase(self, phase: ScreeningPhase, **kwargs: Any) -> PhaseResult:
        """
        Run a single named phase independently.

        Args:
            phase: The ScreeningPhase to execute.
            **kwargs: Phase-specific keyword arguments.

        Returns:
            PhaseResult for the executed phase.
        """
        dispatch: Dict[ScreeningPhase, Any] = {
            ScreeningPhase.CLAIM_UNIVERSE_EXTRACTION: lambda: self._run_claim_universe_extraction(
                WorkflowInput(claims=kwargs.get("claims", []))
            ),
            ScreeningPhase.SEVEN_SINS_ANALYSIS: lambda: self._run_seven_sins_analysis(
                kwargs.get("universe", {})
            ),
            ScreeningPhase.RISK_SCORING: lambda: self._run_risk_scoring(
                kwargs.get("sins_data", {})
            ),
            ScreeningPhase.ACTION_PLAN_GENERATION: lambda: self._run_action_plan_generation(
                kwargs.get("risk_data", {})
            ),
        }
        handler = dispatch.get(phase)
        if handler is None:
            return PhaseResult(
                phase_name=phase.value,
                status=PhaseStatus.FAILED,
                error_message=f"Unknown phase: {phase.value}",
            )
        return handler()

    # ------------------------------------------------------------------
    # Phase 1: Claim Universe Extraction
    # ------------------------------------------------------------------

    def _run_claim_universe_extraction(self, input_data: WorkflowInput) -> PhaseResult:
        """Extract and normalise claims from all communication channels."""
        started = utcnow()
        self.logger.info("Phase 1/4 ClaimUniverseExtraction -- processing %d claims",
                         len(input_data.claims))

        extracted: List[Dict[str, Any]] = []
        channel_counts: Dict[str, int] = {}

        for idx, claim in enumerate(input_data.claims):
            text = claim.get("text", "")
            channel = claim.get("channel", ClaimChannel.OTHER.value)
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

            extracted.append({
                "claim_id": claim.get("id", f"CLM-{idx:04d}"),
                "text": text,
                "text_normalised": text.lower().strip(),
                "channel": channel,
                "text_length": len(text),
                "has_quantification": self._has_quantification(text),
                "word_count": len(text.split()),
            })

        result_data: Dict[str, Any] = {
            "extracted_claims": extracted,
            "total_claims": len(extracted),
            "channel_distribution": channel_counts,
            "channels_covered": len(channel_counts),
            "quantified_claims": sum(1 for e in extracted if e["has_quantification"]),
        }

        return PhaseResult(
            phase_name=ScreeningPhase.CLAIM_UNIVERSE_EXTRACTION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Seven Sins Analysis
    # ------------------------------------------------------------------

    def _run_seven_sins_analysis(self, universe_data: Dict[str, Any]) -> PhaseResult:
        """Evaluate each claim against TerraChoice Seven Sins and UCPD criteria."""
        started = utcnow()
        self.logger.info("Phase 2/4 SevenSinsAnalysis -- evaluating patterns")

        analysed: List[Dict[str, Any]] = []
        sin_totals: Dict[str, int] = {s.value: 0 for s in SevenSin}

        for claim in universe_data.get("extracted_claims", []):
            text_lower = claim.get("text_normalised", "")
            detected_sins = self._detect_sins(text_lower)

            for sin in detected_sins:
                sin_totals[sin] += 1

            analysed.append({
                "claim_id": claim["claim_id"],
                "text": claim["text"],
                "channel": claim["channel"],
                "has_quantification": claim["has_quantification"],
                "sins_detected": detected_sins,
                "sin_count": len(detected_sins),
                "is_flagged": len(detected_sins) > 0,
            })

        flagged_count = sum(1 for a in analysed if a["is_flagged"])

        result_data: Dict[str, Any] = {
            "analysed_claims": analysed,
            "total_analysed": len(analysed),
            "flagged_count": flagged_count,
            "clean_count": len(analysed) - flagged_count,
            "sin_distribution": {k: v for k, v in sin_totals.items() if v > 0},
            "unique_sins_detected": sum(1 for v in sin_totals.values() if v > 0),
        }

        return PhaseResult(
            phase_name=ScreeningPhase.SEVEN_SINS_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Risk Scoring
    # ------------------------------------------------------------------

    def _run_risk_scoring(self, sins_data: Dict[str, Any]) -> PhaseResult:
        """Score each claim 0-100 based on detected sins and context."""
        started = utcnow()
        self.logger.info("Phase 3/4 RiskScoring -- computing scores")

        scored: List[Dict[str, Any]] = []
        tier_counts: Dict[str, int] = {t.value: 0 for t in RiskTier}

        for claim in sins_data.get("analysed_claims", []):
            risk_score = self._calculate_risk_score(claim)
            risk_tier = self._score_to_tier(risk_score)
            tier_counts[risk_tier.value] += 1

            scored.append({
                "claim_id": claim["claim_id"],
                "text": claim["text"],
                "channel": claim["channel"],
                "sins_detected": claim["sins_detected"],
                "risk_score": risk_score,
                "risk_tier": risk_tier.value,
            })

        # Sort by risk score descending
        scored.sort(key=lambda x: x["risk_score"], reverse=True)

        avg_score = round(
            sum(s["risk_score"] for s in scored) / len(scored) if scored else 0.0, 1
        )

        result_data: Dict[str, Any] = {
            "scored_claims": scored,
            "tier_distribution": tier_counts,
            "average_risk_score": avg_score,
            "critical_count": tier_counts.get(RiskTier.CRITICAL.value, 0),
            "high_count": tier_counts.get(RiskTier.HIGH.value, 0),
            "claims_requiring_action": (
                tier_counts.get(RiskTier.CRITICAL.value, 0)
                + tier_counts.get(RiskTier.HIGH.value, 0)
            ),
        }

        return PhaseResult(
            phase_name=ScreeningPhase.RISK_SCORING.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Action Plan Generation
    # ------------------------------------------------------------------

    def _run_action_plan_generation(self, risk_data: Dict[str, Any]) -> PhaseResult:
        """Generate a prioritised remediation action plan."""
        started = utcnow()
        self.logger.info("Phase 4/4 ActionPlanGeneration -- building action plan")

        action_items: List[Dict[str, Any]] = []
        total_effort_days = 0

        for rank, claim in enumerate(risk_data.get("scored_claims", []), start=1):
            if claim["risk_score"] < self.screening_config.low_threshold:
                continue

            actions = self._get_remediation_actions(claim["sins_detected"])
            effort = self._estimate_effort(claim["risk_tier"])
            total_effort_days += effort

            action_items.append({
                "priority_rank": rank,
                "claim_id": claim["claim_id"],
                "claim_text": claim["text"],
                "channel": claim["channel"],
                "risk_score": claim["risk_score"],
                "risk_tier": claim["risk_tier"],
                "sins_detected": claim["sins_detected"],
                "remediation_actions": actions,
                "estimated_effort_days": effort,
                "action_id": _new_uuid(),
            })

        result_data: Dict[str, Any] = {
            "action_plan": action_items,
            "total_action_items": len(action_items),
            "total_remediation_actions": sum(
                len(a["remediation_actions"]) for a in action_items
            ),
            "estimated_total_effort_days": total_effort_days,
            "screening_outcome": self._determine_outcome(risk_data),
        }

        return PhaseResult(
            phase_name=ScreeningPhase.ACTION_PLAN_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_quantification(self, text: str) -> bool:
        """Check if claim text contains quantified values."""
        indicators = {"%", "percent", "kg", "tonnes", "litre", "kwh", "mwh", "grams"}
        lower = text.lower()
        return any(ind in lower for ind in indicators)

    def _detect_sins(self, text_lower: str) -> List[str]:
        """Detect greenwashing sins from normalised text."""
        detected: List[str] = []
        for sin_name, keywords in self.SIN_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(sin_name)
        return detected

    def _calculate_risk_score(self, claim: Dict[str, Any]) -> float:
        """Calculate deterministic risk score 0-100 for a claim."""
        if not claim.get("sins_detected"):
            return 0.0

        # Sum weights for detected sins
        weight_sum = sum(
            self.SIN_WEIGHTS.get(sin, 10.0) for sin in claim["sins_detected"]
        )

        # Penalty for lacking quantification
        if not claim.get("has_quantification", False) and claim.get("sins_detected"):
            weight_sum += 8.0

        # Multiple sins compound the risk
        sin_count = len(claim["sins_detected"])
        if sin_count >= 3:
            weight_sum *= 1.3
        elif sin_count == 2:
            weight_sum *= 1.15

        return min(round(weight_sum, 1), 100.0)

    def _score_to_tier(self, score: float) -> RiskTier:
        """Map risk score to tier using configured thresholds."""
        if score >= self.screening_config.critical_threshold:
            return RiskTier.CRITICAL
        if score >= self.screening_config.high_threshold:
            return RiskTier.HIGH
        if score >= self.screening_config.medium_threshold:
            return RiskTier.MEDIUM
        if score >= self.screening_config.low_threshold:
            return RiskTier.LOW
        return RiskTier.NEGLIGIBLE

    def _get_remediation_actions(self, sins: List[str]) -> List[str]:
        """Map detected sins to concrete remediation actions."""
        actions_map: Dict[str, List[str]] = {
            SevenSin.VAGUENESS.value: [
                "Replace vague terms with specific, measurable claims",
                "Add quantified environmental metrics",
            ],
            SevenSin.NO_PROOF.value: [
                "Commission third-party verification or LCA study",
                "Compile and publish supporting evidence dossier",
            ],
            SevenSin.FUTURE_PROMISE.value: [
                "Publish detailed implementation plan with milestones",
                "Disclose current baseline and progress metrics",
            ],
            SevenSin.OFFSET_RELIANCE.value: [
                "Prioritise absolute emission reductions over offsets",
                "Disclose offset quality criteria and certification",
            ],
            SevenSin.HIDDEN_TRADEOFF.value: [
                "Disclose full lifecycle impacts including trade-offs",
                "Include complete environmental footprint analysis",
            ],
            SevenSin.FIBBING.value: [
                "Immediately withdraw the claim pending investigation",
                "Engage independent verifier to audit claim accuracy",
            ],
            SevenSin.FALSE_LABELS.value: [
                "Remove unapproved labels from all products",
                "Replace with EU-recognised certification schemes",
            ],
            SevenSin.GENERIC_CLAIM.value: [
                "Specify which environmental attribute is addressed",
                "Narrow claim scope to verifiable product features",
            ],
            SevenSin.IRRELEVANCE.value: [
                "Remove claims referencing legally mandated attributes",
            ],
            SevenSin.LESSER_OF_TWO_EVILS.value: [
                "Reframe claim within broader environmental context",
            ],
        }

        result: List[str] = []
        seen: set = set()
        for sin in sins:
            for action in actions_map.get(sin, ["Review claim against Directive requirements"]):
                if action not in seen:
                    result.append(action)
                    seen.add(action)
        return result

    def _estimate_effort(self, risk_tier: str) -> int:
        """Estimate remediation effort in working days."""
        effort_map: Dict[str, int] = {
            RiskTier.CRITICAL.value: 20,
            RiskTier.HIGH.value: 10,
            RiskTier.MEDIUM.value: 5,
            RiskTier.LOW.value: 2,
            RiskTier.NEGLIGIBLE.value: 1,
        }
        return effort_map.get(risk_tier, 5)

    def _determine_outcome(self, risk_data: Dict[str, Any]) -> str:
        """Determine overall screening outcome."""
        critical = risk_data.get("critical_count", 0)
        high = risk_data.get("high_count", 0)
        total = len(risk_data.get("scored_claims", []))

        if total == 0:
            return "NO_CLAIMS_SCREENED"
        if critical > 0:
            return "CRITICAL -- Immediate action required on high-risk claims"
        if high > 0:
            return "ATTENTION -- High-risk claims require prompt remediation"
        if risk_data.get("average_risk_score", 0) > 30.0:
            return "MODERATE -- Several claims need review before enforcement"
        return "CLEAR -- No significant greenwashing risks detected"
