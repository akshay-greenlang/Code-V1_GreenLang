# -*- coding: utf-8 -*-
"""
Risk Mitigation Coordinator - AGENT-EUDR-026

Phase 3 coordinator for EUDR Article 11 risk mitigation. Manages the
risk mitigation assessment workflow, determining whether mitigation is
required, what measures should be applied, and whether residual risk
has been reduced to acceptable levels.

Phase 3 uses primarily EUDR-025 (Risk Mitigation Advisor) to evaluate
mitigation measures against the composite risk profile computed in
Phase 2. The coordinator determines:

1. Whether mitigation is required (based on composite score vs threshold)
2. What level of mitigation (none, standard, enhanced)
3. Whether mitigation measures are adequate (residual risk <= target)
4. Whether measures are proportionate to identified risks
5. Final mitigation decision for QG-3 evaluation

Risk Mitigation Thresholds (configurable):
    - Negligible risk (< 20): No mitigation required
    - Standard risk (20-50): Standard mitigation measures
    - Enhanced risk (>= 50): Enhanced due diligence required
    - Residual risk target: <= 15 after mitigation

Features:
    - Evaluate mitigation necessity based on composite risk score
    - Classify mitigation level (none/standard/enhanced)
    - Assess mitigation adequacy and proportionality
    - Compute residual risk after mitigation measures
    - Generate mitigation decision with full provenance
    - Support Article 13 simplified (no mitigation path)
    - Track mitigation strategies and their effectiveness
    - Provide evidence for QG-3 quality gate evaluation

Zero-Hallucination:
    - All residual risk calculations are deterministic Decimal arithmetic
    - Mitigation level determined by threshold comparison only
    - No LLM involvement in numeric calculations

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    CompositeRiskProfile,
    MitigationDecision,
    WorkflowType,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mitigation strategy catalog
# ---------------------------------------------------------------------------

#: Available mitigation strategies mapped to risk dimensions and levels.
_MITIGATION_STRATEGIES: Dict[str, Dict[str, List[str]]] = {
    "country": {
        "standard": [
            "Enhanced geolocation verification frequency",
            "Additional satellite monitoring coverage",
        ],
        "enhanced": [
            "Third-party on-site audit in country of origin",
            "Independent verification of legal compliance",
            "Engagement with local stakeholders",
        ],
    },
    "supplier": {
        "standard": [
            "Supplier questionnaire and self-assessment",
            "Annual supplier audit schedule",
        ],
        "enhanced": [
            "Unannounced supplier site visits",
            "Independent third-party supplier audit",
            "Supply chain forensic investigation",
        ],
    },
    "commodity": {
        "standard": [
            "Commodity-specific traceability verification",
            "Origin certification verification",
        ],
        "enhanced": [
            "DNA/isotope origin testing",
            "Full chain of custody audit",
        ],
    },
    "corruption": {
        "standard": [
            "Enhanced due diligence on intermediaries",
            "Anti-bribery compliance check",
        ],
        "enhanced": [
            "Independent governance assessment",
            "Beneficial ownership verification",
        ],
    },
    "deforestation": {
        "standard": [
            "Increased satellite monitoring frequency",
            "Deforestation-free commitment verification",
        ],
        "enhanced": [
            "Real-time deforestation alert system",
            "Ground-truth verification visits",
            "Remediation plan for detected changes",
        ],
    },
    "indigenous": {
        "standard": [
            "FPIC process verification",
            "Community engagement assessment",
        ],
        "enhanced": [
            "Independent FPIC audit",
            "Grievance mechanism review",
        ],
    },
    "protected": {
        "standard": [
            "Protected area buffer zone verification",
            "Boundary compliance check",
        ],
        "enhanced": [
            "Environmental impact assessment",
            "Conservation authority consultation",
        ],
    },
    "legal": {
        "standard": [
            "Legal compliance document review",
            "Permit and license verification",
        ],
        "enhanced": [
            "Independent legal compliance audit",
            "Regulatory authority consultation",
        ],
    },
    "audit": {
        "standard": [
            "Schedule third-party verification",
            "Review existing certifications",
        ],
        "enhanced": [
            "Commission independent audit",
            "Multi-stakeholder verification process",
        ],
    },
    "mitigation": {
        "standard": [
            "Implement standard risk management procedures",
            "Document mitigation measures and timeline",
        ],
        "enhanced": [
            "Develop comprehensive risk mitigation plan",
            "Establish ongoing monitoring framework",
        ],
    },
}


# ---------------------------------------------------------------------------
# RiskMitigationCoordinator
# ---------------------------------------------------------------------------


class RiskMitigationCoordinator:
    """Phase 3 coordinator for EUDR Article 11 risk mitigation.

    Evaluates whether mitigation is required based on the composite risk
    score from Phase 2, determines appropriate mitigation measures, and
    assesses whether residual risk has been reduced to acceptable levels.

    All numeric calculations use deterministic Decimal arithmetic with
    no LLM involvement. Mitigation level classification is purely
    threshold-based per configured risk thresholds.

    Attributes:
        _config: Configuration with risk thresholds and targets.

    Example:
        >>> coordinator = RiskMitigationCoordinator()
        >>> decision = coordinator.evaluate_mitigation(
        ...     workflow_id="wf-001",
        ...     risk_profile=profile,
        ... )
        >>> assert decision.mitigation_required or decision.mitigation_level == "none"
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the RiskMitigationCoordinator.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()
        logger.info("RiskMitigationCoordinator initialized")

    # ------------------------------------------------------------------
    # Mitigation evaluation
    # ------------------------------------------------------------------

    def evaluate_mitigation(
        self,
        workflow_id: str,
        risk_profile: CompositeRiskProfile,
        mitigation_evidence: Optional[Dict[str, Any]] = None,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> MitigationDecision:
        """Evaluate whether mitigation is required and compute decision.

        Deterministic decision logic:
        1. If simplified workflow -> no mitigation required
        2. If composite_score < negligible_threshold -> no mitigation
        3. If composite_score < standard_threshold -> standard mitigation
        4. Otherwise -> enhanced mitigation required

        If mitigation evidence is provided (from EUDR-025), the post-
        mitigation residual risk is computed and adequacy is assessed.

        Args:
            workflow_id: Parent workflow identifier.
            risk_profile: Composite risk profile from Phase 2.
            mitigation_evidence: Optional evidence from EUDR-025 agent.
            workflow_type: Standard or simplified workflow.

        Returns:
            MitigationDecision with complete assessment.

        Example:
            >>> coordinator = RiskMitigationCoordinator()
            >>> decision = coordinator.evaluate_mitigation(
            ...     "wf-001", profile
            ... )
        """
        start_time = utcnow()

        composite_score = risk_profile.composite_score
        negligible = self._config.negligible_risk_threshold
        standard = self._config.standard_risk_threshold
        target = self._config.residual_risk_target

        # Simplified due diligence: no mitigation required
        if workflow_type == WorkflowType.SIMPLIFIED:
            return self._create_no_mitigation_decision(
                workflow_id, composite_score,
                bypass_justification="Simplified due diligence per Article 13"
            )

        # Determine mitigation requirement and level
        if composite_score < negligible:
            return self._create_no_mitigation_decision(
                workflow_id, composite_score,
                bypass_justification=(
                    f"Composite risk score {composite_score} below "
                    f"negligible threshold {negligible}"
                )
            )

        mitigation_level = "standard" if composite_score < standard else "enhanced"
        mitigation_required = True

        # Identify recommended strategies based on risk profile
        strategies = self._recommend_strategies(
            risk_profile, mitigation_level
        )

        # Compute post-mitigation score if evidence is available
        post_score: Optional[Decimal] = None
        adequacy_verified = False
        proportionality_verified = False

        if mitigation_evidence:
            post_score = self._compute_residual_risk(
                risk_profile, mitigation_evidence
            )
            adequacy_verified = (
                post_score is not None and post_score <= target
            )
            proportionality_verified = self._assess_proportionality(
                composite_score, mitigation_level, strategies
            )

        # Build evidence dictionary
        evidence: Dict[str, Any] = {
            "pre_mitigation_score": str(composite_score),
            "risk_level": risk_profile.risk_level,
            "mitigation_level": mitigation_level,
            "negligible_threshold": str(negligible),
            "standard_threshold": str(standard),
            "residual_risk_target": str(target),
            "strategies_recommended": len(strategies),
            "top_risk_dimensions": risk_profile.highest_risk_dimensions,
        }
        if mitigation_evidence:
            evidence["mitigation_evidence_provided"] = True
            evidence["post_mitigation_score"] = str(post_score)
        if post_score is not None:
            evidence["residual_risk_delta"] = str(composite_score - post_score)

        decision = MitigationDecision(
            decision_id=_new_uuid(),
            workflow_id=workflow_id,
            mitigation_required=mitigation_required,
            mitigation_level=mitigation_level,
            pre_mitigation_score=composite_score,
            post_mitigation_score=post_score,
            mitigation_strategies=strategies,
            adequacy_verified=adequacy_verified,
            proportionality_verified=proportionality_verified,
            evidence=evidence,
            decided_at=utcnow(),
            provenance_hash=self._hash_decision(
                workflow_id, composite_score, mitigation_level,
                post_score, strategies
            ),
        )

        duration_ms = (utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Mitigation decision for {workflow_id}: "
            f"required={mitigation_required}, level={mitigation_level}, "
            f"pre={composite_score}, post={post_score}, "
            f"adequate={adequacy_verified} in {duration_ms:.1f}ms"
        )

        return decision

    # ------------------------------------------------------------------
    # Residual risk assessment
    # ------------------------------------------------------------------

    def compute_residual_risk(
        self,
        risk_profile: CompositeRiskProfile,
        mitigation_evidence: Dict[str, Any],
    ) -> Decimal:
        """Compute residual risk score after applying mitigation measures.

        Zero-Hallucination: residual risk is computed as:
            residual = composite_score - sum(reduction_i)
        where reduction_i is the documented risk reduction from each
        mitigation measure, clamped to [0, 100].

        Args:
            risk_profile: Pre-mitigation risk profile.
            mitigation_evidence: Evidence of mitigation measures applied.

        Returns:
            Residual risk score (0-100).
        """
        return self._compute_residual_risk(risk_profile, mitigation_evidence)

    def assess_adequacy(
        self,
        residual_risk: Decimal,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> bool:
        """Assess whether residual risk meets the adequacy threshold.

        Args:
            residual_risk: Post-mitigation residual risk score.
            workflow_type: Standard or simplified workflow.

        Returns:
            True if residual risk is within acceptable limits.
        """
        if workflow_type == WorkflowType.SIMPLIFIED:
            threshold = self._config.qg3_simplified_threshold
        else:
            threshold = self._config.qg3_residual_risk_threshold

        return residual_risk <= threshold

    # ------------------------------------------------------------------
    # Strategy recommendation
    # ------------------------------------------------------------------

    def recommend_strategies(
        self,
        risk_profile: CompositeRiskProfile,
        mitigation_level: str = "standard",
    ) -> List[str]:
        """Recommend mitigation strategies based on risk profile.

        Selects strategies from the catalog based on the highest-risk
        dimensions and the required mitigation level.

        Args:
            risk_profile: Composite risk profile.
            mitigation_level: Required mitigation level.

        Returns:
            Ordered list of recommended strategy descriptions.
        """
        return self._recommend_strategies(risk_profile, mitigation_level)

    def get_available_strategies(
        self,
        dimension: str,
        level: str = "standard",
    ) -> List[str]:
        """Get available mitigation strategies for a risk dimension.

        Args:
            dimension: Risk dimension name (e.g., "country").
            level: Mitigation level ("standard" or "enhanced").

        Returns:
            List of strategy descriptions.
        """
        dim_strategies = _MITIGATION_STRATEGIES.get(dimension, {})
        return dim_strategies.get(level, [])

    # ------------------------------------------------------------------
    # QG-3 preparation
    # ------------------------------------------------------------------

    def prepare_qg3_data(
        self,
        decision: MitigationDecision,
    ) -> Dict[str, Any]:
        """Prepare data for QG-3 (Mitigation Adequacy) quality gate.

        Args:
            decision: Mitigation decision to evaluate.

        Returns:
            Dictionary with QG-3 evaluation data.
        """
        return {
            "mitigation_required": decision.mitigation_required,
            "mitigation_level": decision.mitigation_level,
            "pre_mitigation_score": str(decision.pre_mitigation_score),
            "post_mitigation_score": (
                str(decision.post_mitigation_score)
                if decision.post_mitigation_score is not None
                else None
            ),
            "residual_risk_target": str(
                self._config.qg3_residual_risk_threshold
            ),
            "adequacy_verified": decision.adequacy_verified,
            "proportionality_verified": decision.proportionality_verified,
            "strategies_count": len(decision.mitigation_strategies),
            "strategies": decision.mitigation_strategies,
            "provenance_hash": decision.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_no_mitigation_decision(
        self,
        workflow_id: str,
        composite_score: Decimal,
        bypass_justification: str,
    ) -> MitigationDecision:
        """Create a decision indicating no mitigation is required.

        Args:
            workflow_id: Workflow identifier.
            composite_score: Pre-mitigation composite score.
            bypass_justification: Reason mitigation is not required.

        Returns:
            MitigationDecision with mitigation_required=False.
        """
        return MitigationDecision(
            decision_id=_new_uuid(),
            workflow_id=workflow_id,
            mitigation_required=False,
            mitigation_level="none",
            pre_mitigation_score=composite_score,
            post_mitigation_score=composite_score,
            mitigation_strategies=[],
            adequacy_verified=True,
            proportionality_verified=True,
            evidence={
                "bypass_reason": bypass_justification,
                "pre_mitigation_score": str(composite_score),
            },
            bypass_justification=bypass_justification,
            decided_at=utcnow(),
            provenance_hash=self._hash_decision(
                workflow_id, composite_score, "none",
                composite_score, []
            ),
        )

    def _compute_residual_risk(
        self,
        risk_profile: CompositeRiskProfile,
        mitigation_evidence: Dict[str, Any],
    ) -> Decimal:
        """Compute residual risk from mitigation evidence.

        Deterministic calculation:
            residual = max(0, composite_score - total_reduction)

        Args:
            risk_profile: Pre-mitigation risk profile.
            mitigation_evidence: Evidence with reduction amounts.

        Returns:
            Residual risk score (0-100).
        """
        composite = risk_profile.composite_score

        # Extract reduction amounts from evidence
        total_reduction = Decimal("0")
        reductions = mitigation_evidence.get("risk_reductions", {})
        for dimension, reduction in reductions.items():
            try:
                reduction_val = Decimal(str(reduction))
                total_reduction += reduction_val
            except Exception:
                logger.warning(
                    f"Invalid reduction value for {dimension}: {reduction}"
                )

        residual = composite - total_reduction
        residual = max(Decimal("0"), min(residual, Decimal("100")))
        return residual.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _recommend_strategies(
        self,
        risk_profile: CompositeRiskProfile,
        mitigation_level: str,
    ) -> List[str]:
        """Select mitigation strategies based on risk dimensions.

        Prioritizes strategies for the highest-risk dimensions.

        Args:
            risk_profile: Composite risk profile.
            mitigation_level: Required level ("standard" or "enhanced").

        Returns:
            Ordered list of strategy descriptions.
        """
        strategies: List[str] = []

        # Get top risk dimensions from contributions
        sorted_contribs = sorted(
            risk_profile.contributions,
            key=lambda c: c.weighted_score,
            reverse=True,
        )

        for contrib in sorted_contribs:
            dimension_entry = {
                "EUDR-016": "country",
                "EUDR-017": "supplier",
                "EUDR-018": "commodity",
                "EUDR-019": "corruption",
                "EUDR-020": "deforestation",
                "EUDR-021": "indigenous",
                "EUDR-022": "protected",
                "EUDR-023": "legal",
                "EUDR-024": "audit",
                "EUDR-025": "mitigation",
            }
            dimension = dimension_entry.get(contrib.agent_id)
            if dimension:
                dim_strategies = _MITIGATION_STRATEGIES.get(dimension, {})
                level_strategies = dim_strategies.get(mitigation_level, [])
                strategies.extend(level_strategies)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for s in strategies:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique

    def _assess_proportionality(
        self,
        composite_score: Decimal,
        mitigation_level: str,
        strategies: List[str],
    ) -> bool:
        """Assess whether mitigation measures are proportionate.

        Proportionality requires that the number and intensity of
        mitigation measures match the risk level. Enhanced mitigation
        should have more strategies than standard.

        Args:
            composite_score: Pre-mitigation composite score.
            mitigation_level: Applied mitigation level.
            strategies: Applied strategies.

        Returns:
            True if proportionality requirements are met.
        """
        if not strategies:
            return composite_score < self._config.negligible_risk_threshold

        # Minimum strategies per level
        min_strategies = {"standard": 2, "enhanced": 5}
        required = min_strategies.get(mitigation_level, 1)

        return len(strategies) >= required

    def _hash_decision(
        self,
        workflow_id: str,
        pre_score: Decimal,
        level: str,
        post_score: Optional[Decimal],
        strategies: List[str],
    ) -> str:
        """Compute SHA-256 hash for a mitigation decision.

        Args:
            workflow_id: Workflow identifier.
            pre_score: Pre-mitigation score.
            level: Mitigation level.
            post_score: Post-mitigation score (may be None).
            strategies: Applied strategies.

        Returns:
            64-character hex SHA-256 hash.
        """
        data = {
            "workflow_id": workflow_id,
            "pre_mitigation_score": str(pre_score),
            "mitigation_level": level,
            "post_mitigation_score": str(post_score) if post_score else "null",
            "strategy_count": len(strategies),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
