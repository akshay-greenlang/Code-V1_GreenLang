# -*- coding: utf-8 -*-
"""
Risk Assessment Coordinator - AGENT-EUDR-026

Phase 2 coordinator for EUDR Article 10 risk assessment. Orchestrates
the 10 risk assessment agents (EUDR-016 through EUDR-025) to compute
a composite risk score across 10 weighted risk dimensions.

The composite risk score formula (zero-hallucination, deterministic):

    CompositeRiskScore = sum(weight_i * raw_score_i) for i in 1..10

    where:
        weight_country      = 0.15  (EUDR-016 Country Risk Evaluator)
        weight_supplier     = 0.12  (EUDR-017 Supplier Risk Scorer)
        weight_commodity    = 0.10  (EUDR-018 Commodity Risk Analyzer)
        weight_corruption   = 0.08  (EUDR-019 Corruption Index Monitor)
        weight_deforestation= 0.15  (EUDR-020 Deforestation Alert System)
        weight_indigenous   = 0.10  (EUDR-021 Indigenous Rights Checker)
        weight_protected    = 0.10  (EUDR-022 Protected Area Validator)
        weight_legal        = 0.10  (EUDR-023 Legal Compliance Verifier)
        weight_audit        = 0.05  (EUDR-024 Third-Party Audit Manager)
        weight_mitigation   = 0.05  (EUDR-025 Risk Mitigation Advisor)
        sum(weights)        = 1.00

Risk Level Classification:
    - Negligible: score < 20 (no mitigation required)
    - Low: 20 <= score < 35
    - Medium: 35 <= score < 50
    - High: 50 <= score < 75
    - Critical: score >= 75

Features:
    - Orchestrate all 10 risk assessment agents
    - Compute deterministic weighted composite risk score
    - Classify risk levels per EUDR Article 10 requirements
    - Track risk coverage percentage for QG-2 evaluation
    - Map risk factors to Article 10(2) criteria
    - Identify top risk dimensions for mitigation prioritization
    - Support simplified due diligence risk assessment (Article 13)
    - Provide confidence scores for each risk dimension
    - Complete provenance tracking for all risk calculations

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
from typing import Any, Dict, List, Optional, Set, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    PHASE_2_AGENTS,
    AgentExecutionStatus,
    CompositeRiskProfile,
    DueDiligencePhase,
    EUDRCommodity,
    RiskScoreContribution,
    WorkflowType,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-to-risk-dimension mapping
# ---------------------------------------------------------------------------

#: Maps each Phase 2 agent to its risk dimension name and weight config key.
_AGENT_RISK_DIMENSIONS: Dict[str, Tuple[str, str]] = {
    "EUDR-016": ("country", "w_country"),
    "EUDR-017": ("supplier", "w_supplier"),
    "EUDR-018": ("commodity", "w_commodity"),
    "EUDR-019": ("corruption", "w_corruption"),
    "EUDR-020": ("deforestation", "w_deforestation"),
    "EUDR-021": ("indigenous", "w_indigenous"),
    "EUDR-022": ("protected", "w_protected"),
    "EUDR-023": ("legal", "w_legal"),
    "EUDR-024": ("audit", "w_audit"),
    "EUDR-025": ("mitigation", "w_mitigation"),
}

#: Article 10(2) factor categories mapped to risk dimensions.
_ARTICLE_10_FACTORS: Dict[str, List[str]] = {
    "country": [
        "Art.10(2)(a) country/region risk level",
        "Art.10(2)(a) EU benchmarking classification",
    ],
    "supplier": [
        "Art.10(2)(b) operator/trader compliance history",
        "Art.10(2)(c) supply chain complexity",
    ],
    "commodity": [
        "Art.10(2)(d) commodity deforestation risk profile",
    ],
    "corruption": [
        "Art.10(2)(e) corruption perception index",
        "Art.10(2)(e) governance indicators",
    ],
    "deforestation": [
        "Art.10(2)(a) deforestation rate in sourcing region",
        "Art.10(2)(d) satellite-detected land use change",
    ],
    "indigenous": [
        "Art.10(2)(f) indigenous peoples and local communities",
        "Art.10(2)(f) FPIC compliance",
    ],
    "protected": [
        "Art.10(2)(a) proximity to protected areas",
        "Art.10(2)(a) UNESCO/Ramsar site overlap",
    ],
    "legal": [
        "Art.10(2)(g) applicable legislation compliance",
        "Art.10(2)(g) legality of production",
    ],
    "audit": [
        "Art.10(2)(h) third-party verification availability",
        "Art.10(2)(h) certification status",
    ],
    "mitigation": [
        "Art.10(2)(i) existing mitigation measures",
        "Art.10(2)(i) risk management effectiveness",
    ],
}

#: Risk level thresholds and labels.
_RISK_LEVELS: List[Tuple[Decimal, str]] = [
    (Decimal("20"), "negligible"),
    (Decimal("35"), "low"),
    (Decimal("50"), "medium"),
    (Decimal("75"), "high"),
    (Decimal("100"), "critical"),
]


# ---------------------------------------------------------------------------
# RiskAssessmentCoordinator
# ---------------------------------------------------------------------------


class RiskAssessmentCoordinator:
    """Phase 2 coordinator for EUDR Article 10 risk assessment.

    Orchestrates the 10 risk assessment agents to compute a deterministic
    weighted composite risk score. All calculations use Decimal arithmetic
    with no LLM involvement to ensure zero-hallucination compliance.

    The composite risk score drives the due diligence pathway:
    - Score < negligible_threshold (20): Skip mitigation (Art. 13)
    - Score < standard_threshold (50): Standard mitigation (Art. 11)
    - Score >= standard_threshold (50): Enhanced mitigation required

    Attributes:
        _config: Agent configuration with risk weights and thresholds.

    Example:
        >>> coordinator = RiskAssessmentCoordinator()
        >>> profile = coordinator.compute_composite_risk(
        ...     workflow_id="wf-001",
        ...     agent_scores={"EUDR-016": Decimal("45"), ...}
        ... )
        >>> assert profile.risk_level in ("negligible", "low", "medium", "high", "critical")
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the RiskAssessmentCoordinator.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()
        logger.info("RiskAssessmentCoordinator initialized")

    # ------------------------------------------------------------------
    # Phase 2 agents
    # ------------------------------------------------------------------

    def get_phase2_agents(self) -> List[str]:
        """Return the ordered list of Phase 2 agent IDs.

        Returns:
            List of 10 Phase 2 EUDR agent identifiers.
        """
        return list(PHASE_2_AGENTS)

    def get_risk_dimension(self, agent_id: str) -> Optional[str]:
        """Get the risk dimension name for a Phase 2 agent.

        Args:
            agent_id: EUDR agent identifier.

        Returns:
            Risk dimension name or None if not a Phase 2 agent.
        """
        entry = _AGENT_RISK_DIMENSIONS.get(agent_id)
        return entry[0] if entry else None

    # ------------------------------------------------------------------
    # Composite risk computation
    # ------------------------------------------------------------------

    def compute_composite_risk(
        self,
        workflow_id: str,
        agent_scores: Dict[str, Decimal],
        agent_confidences: Optional[Dict[str, Decimal]] = None,
        agent_risk_factors: Optional[Dict[str, List[str]]] = None,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> CompositeRiskProfile:
        """Compute the weighted composite risk score from all risk agents.

        Zero-Hallucination Implementation:
            CompositeRiskScore = sum(weight_i * raw_score_i)
            All arithmetic uses Python Decimal for exact precision.
            No LLM or ML model is involved in score calculation.

        Args:
            workflow_id: Parent workflow identifier.
            agent_scores: Raw risk scores (0-100) from each Phase 2 agent,
                keyed by agent_id.
            agent_confidences: Optional confidence scores (0-1) per agent.
            agent_risk_factors: Optional risk factor lists per agent.
            workflow_type: Standard or simplified workflow.

        Returns:
            CompositeRiskProfile with composite score and contributions.

        Raises:
            ValueError: If agent_scores contains invalid values.

        Example:
            >>> coordinator = RiskAssessmentCoordinator()
            >>> scores = {f"EUDR-{i:03d}": Decimal("50") for i in range(16, 26)}
            >>> profile = coordinator.compute_composite_risk("wf-001", scores)
            >>> assert profile.composite_score == Decimal("50.0000")
        """
        start_time = utcnow()
        weights = self._config.get_weight_dict()

        # Validate input scores
        self._validate_scores(agent_scores)

        # Build individual contributions
        contributions: List[RiskScoreContribution] = []
        composite = Decimal("0")
        scored_count = 0

        for agent_id in PHASE_2_AGENTS:
            dimension_entry = _AGENT_RISK_DIMENSIONS.get(agent_id)
            if dimension_entry is None:
                continue

            dimension_name, weight_key = dimension_entry
            weight = weights.get(dimension_name, Decimal("0"))

            if agent_id in agent_scores:
                raw_score = agent_scores[agent_id]
                weighted_score = (raw_score * weight).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                composite += weighted_score
                scored_count += 1

                confidence = Decimal("1.0")
                if agent_confidences and agent_id in agent_confidences:
                    confidence = agent_confidences[agent_id]

                risk_factors: List[str] = []
                if agent_risk_factors and agent_id in agent_risk_factors:
                    risk_factors = agent_risk_factors[agent_id]

                article_10_mapping = _ARTICLE_10_FACTORS.get(
                    dimension_name, []
                )

                contribution = RiskScoreContribution(
                    agent_id=agent_id,
                    agent_name=AGENT_NAMES.get(agent_id, agent_id),
                    raw_score=raw_score,
                    weight=weight,
                    weighted_score=weighted_score,
                    risk_factors=risk_factors,
                    article_10_mapping=article_10_mapping,
                    confidence=confidence,
                    provenance_hash=self._hash_contribution(
                        agent_id, raw_score, weight, weighted_score
                    ),
                )
                contributions.append(contribution)

        # Round composite score
        composite = composite.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Classify risk level
        risk_level = self._classify_risk_level(composite)

        # Identify top risk dimensions
        sorted_contribs = sorted(
            contributions,
            key=lambda c: c.weighted_score,
            reverse=True,
        )
        top_dimensions = [c.agent_name for c in sorted_contribs[:3]]

        # Coverage calculation
        total_dimensions = len(_AGENT_RISK_DIMENSIONS)
        coverage_pct = Decimal(str(
            scored_count / total_dimensions * 100
        )).quantize(Decimal("0.01"))

        # Build profile
        profile = CompositeRiskProfile(
            profile_id=_new_uuid(),
            workflow_id=workflow_id,
            contributions=contributions,
            composite_score=composite,
            risk_level=risk_level,
            highest_risk_dimensions=top_dimensions,
            all_dimensions_scored=(scored_count == total_dimensions),
            coverage_pct=coverage_pct,
            assessed_at=utcnow(),
            provenance_hash=self._hash_profile(
                workflow_id, composite, contributions
            ),
        )

        duration_ms = (utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Composite risk score for {workflow_id}: {composite} "
            f"({risk_level}) from {scored_count}/{total_dimensions} agents "
            f"in {duration_ms:.1f}ms"
        )

        return profile

    # ------------------------------------------------------------------
    # Risk coverage evaluation (QG-2 support)
    # ------------------------------------------------------------------

    def evaluate_coverage(
        self,
        agent_statuses: Dict[str, AgentExecutionStatus],
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> Tuple[Decimal, List[str]]:
        """Evaluate risk assessment coverage for QG-2 quality gate.

        Computes the percentage of required risk dimensions that have
        been successfully scored. QG-2 requires >= 95% coverage for
        standard workflows and >= 85% for simplified.

        Args:
            agent_statuses: Execution status of each Phase 2 agent.
            workflow_type: Standard or simplified workflow.

        Returns:
            Tuple of (coverage_percentage, missing_dimensions).

        Example:
            >>> coordinator = RiskAssessmentCoordinator()
            >>> cov, missing = coordinator.evaluate_coverage(
            ...     {"EUDR-016": AgentExecutionStatus.COMPLETED, ...}
            ... )
            >>> assert cov >= Decimal("95")
        """
        required_agents = self._get_required_risk_agents(workflow_type)
        completed = set()
        missing_dims: List[str] = []

        for agent_id in required_agents:
            status = agent_statuses.get(agent_id)
            if status == AgentExecutionStatus.COMPLETED:
                completed.add(agent_id)
            else:
                dim = self.get_risk_dimension(agent_id)
                if dim:
                    missing_dims.append(dim)

        if len(required_agents) == 0:
            return Decimal("100"), []

        coverage = Decimal(str(
            len(completed) / len(required_agents) * 100
        )).quantize(Decimal("0.01"))

        return coverage, missing_dims

    # ------------------------------------------------------------------
    # Risk level classification
    # ------------------------------------------------------------------

    def classify_risk_level(self, score: Decimal) -> str:
        """Classify a composite risk score into a risk level.

        Args:
            score: Composite risk score (0-100).

        Returns:
            Risk level string: negligible, low, medium, high, or critical.
        """
        return self._classify_risk_level(score)

    def is_mitigation_required(
        self,
        composite_score: Decimal,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> bool:
        """Determine if risk mitigation is required based on score.

        For standard workflows, mitigation is required when the
        composite score >= negligible_risk_threshold (20).
        For simplified workflows (Article 13), mitigation is
        typically not required.

        Args:
            composite_score: Computed composite risk score.
            workflow_type: Standard or simplified.

        Returns:
            True if mitigation measures must be applied.
        """
        if workflow_type == WorkflowType.SIMPLIFIED:
            return False

        return composite_score >= self._config.negligible_risk_threshold

    def get_mitigation_level(self, composite_score: Decimal) -> str:
        """Determine the required mitigation level based on score.

        Args:
            composite_score: Computed composite risk score.

        Returns:
            Mitigation level: "none", "standard", or "enhanced".
        """
        if composite_score < self._config.negligible_risk_threshold:
            return "none"
        elif composite_score < self._config.standard_risk_threshold:
            return "standard"
        return "enhanced"

    # ------------------------------------------------------------------
    # Input preparation for risk agents
    # ------------------------------------------------------------------

    def prepare_risk_agent_input(
        self,
        agent_id: str,
        workflow_context: Dict[str, Any],
        phase1_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare input data for a Phase 2 risk assessment agent.

        Assembles the input payload combining workflow context with
        relevant Phase 1 evidence for risk evaluation.

        Args:
            agent_id: Target risk agent identifier.
            workflow_context: Shared workflow context data.
            phase1_evidence: Collected evidence from Phase 1.

        Returns:
            Input data dictionary for the risk agent.
        """
        input_data: Dict[str, Any] = {
            "agent_id": agent_id,
            "phase": DueDiligencePhase.RISK_ASSESSMENT.value,
            "workflow_context": workflow_context,
        }

        # Map phase 1 evidence to risk agent inputs
        evidence_mapping = self._get_evidence_mapping(agent_id)
        for evidence_key, input_key in evidence_mapping.items():
            if evidence_key in phase1_evidence:
                input_data[input_key] = phase1_evidence[evidence_key]

        return input_data

    # ------------------------------------------------------------------
    # Article 10(2) factor reporting
    # ------------------------------------------------------------------

    def get_article_10_factors(
        self,
        dimension: str,
    ) -> List[str]:
        """Get Article 10(2) factor descriptions for a risk dimension.

        Args:
            dimension: Risk dimension name (e.g., "country").

        Returns:
            List of Article 10(2) factor descriptions.
        """
        return _ARTICLE_10_FACTORS.get(dimension, [])

    def get_all_article_10_factors(self) -> Dict[str, List[str]]:
        """Get all Article 10(2) factors mapped to risk dimensions.

        Returns:
            Complete mapping of dimensions to Article 10(2) factors.
        """
        return dict(_ARTICLE_10_FACTORS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_scores(self, agent_scores: Dict[str, Decimal]) -> None:
        """Validate that all scores are within the valid range.

        Args:
            agent_scores: Raw scores to validate.

        Raises:
            ValueError: If any score is outside 0-100.
        """
        for agent_id, score in agent_scores.items():
            if score < Decimal("0") or score > Decimal("100"):
                raise ValueError(
                    f"Invalid risk score for {agent_id}: {score}. "
                    f"Must be between 0 and 100."
                )

    def _classify_risk_level(self, score: Decimal) -> str:
        """Classify risk score into level using threshold boundaries.

        Args:
            score: Composite risk score (0-100).

        Returns:
            Risk level label string.
        """
        for threshold, level in _RISK_LEVELS:
            if score < threshold:
                return level
        return "critical"

    def _get_required_risk_agents(
        self,
        workflow_type: WorkflowType,
    ) -> List[str]:
        """Get the list of required risk agents for a workflow type.

        Args:
            workflow_type: Standard or simplified.

        Returns:
            List of required agent IDs.
        """
        if workflow_type == WorkflowType.SIMPLIFIED:
            return ["EUDR-016", "EUDR-018", "EUDR-023"]
        return list(PHASE_2_AGENTS)

    def _get_evidence_mapping(
        self,
        agent_id: str,
    ) -> Dict[str, str]:
        """Get Phase 1 evidence field mapping for a risk agent.

        Args:
            agent_id: Risk agent identifier.

        Returns:
            Dictionary: evidence_field -> input_field.
        """
        mappings: Dict[str, Dict[str, str]] = {
            "EUDR-016": {
                "country_codes": "countries",
                "verified_coordinates": "geolocations",
            },
            "EUDR-017": {
                "supplier_tiers": "supplier_data",
                "supply_chain_map": "supply_chain",
            },
            "EUDR-018": {
                "supply_chain_map": "commodity_data",
            },
            "EUDR-019": {
                "country_codes": "countries",
            },
            "EUDR-020": {
                "satellite_observations": "satellite_data",
                "forest_cover_analysis": "forest_data",
                "land_use_changes": "land_use_data",
            },
            "EUDR-021": {
                "verified_coordinates": "coordinates",
                "plot_boundaries": "boundaries",
            },
            "EUDR-022": {
                "verified_coordinates": "coordinates",
                "plot_boundaries": "boundaries",
            },
            "EUDR-023": {
                "country_codes": "jurisdictions",
                "document_authenticity": "documents",
            },
            "EUDR-024": {
                "document_authenticity": "audit_documents",
                "blockchain_records": "audit_trail",
            },
            "EUDR-025": {
                "supply_chain_map": "supply_chain",
                "custody_chain": "custody_data",
            },
        }
        return mappings.get(agent_id, {})

    def _hash_contribution(
        self,
        agent_id: str,
        raw_score: Decimal,
        weight: Decimal,
        weighted_score: Decimal,
    ) -> str:
        """Compute SHA-256 hash for a single risk contribution.

        Args:
            agent_id: Agent identifier.
            raw_score: Raw risk score.
            weight: Dimension weight.
            weighted_score: Computed weighted score.

        Returns:
            64-character hex SHA-256 hash.
        """
        data = {
            "agent_id": agent_id,
            "raw_score": str(raw_score),
            "weight": str(weight),
            "weighted_score": str(weighted_score),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _hash_profile(
        self,
        workflow_id: str,
        composite_score: Decimal,
        contributions: List[RiskScoreContribution],
    ) -> str:
        """Compute SHA-256 hash for the composite risk profile.

        Args:
            workflow_id: Workflow identifier.
            composite_score: Computed composite score.
            contributions: All risk contributions.

        Returns:
            64-character hex SHA-256 hash.
        """
        data = {
            "workflow_id": workflow_id,
            "composite_score": str(composite_score),
            "contribution_hashes": [
                c.provenance_hash or "" for c in contributions
            ],
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Test-compatible wrapper methods
    # ------------------------------------------------------------------

    def _is_phase2_agent(self, agent_id: str) -> bool:
        """Check if an agent belongs to Phase 2.

        Test-compatible wrapper method.

        Args:
            agent_id: EUDR agent identifier.

        Returns:
            True if the agent is a Phase 2 agent.
        """
        return agent_id in PHASE_2_AGENTS

    def calculate_composite_risk(
        self,
        agent_scores: Dict[str, Decimal],
    ):
        """Calculate composite risk score from agent scores.

        Test-compatible wrapper for compute_composite_risk.

        Args:
            agent_scores: Dict mapping agent_id to risk score.

        Returns:
            CompositeRiskProfile object.
        """
        return self.compute_composite_risk(
            workflow_id="test-workflow",
            agent_scores=agent_scores,
        )
