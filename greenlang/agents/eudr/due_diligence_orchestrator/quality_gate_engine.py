# -*- coding: utf-8 -*-
"""
Quality Gate Engine - AGENT-EUDR-026

Evaluates the three quality gates (QG-1, QG-2, QG-3) that enforce
transition requirements between due diligence phases per EUDR Articles
9, 10, 11, and 13.

Quality Gates:
    QG-1 (Information Gathering Completeness):
        - Phase transition: Art. 9 -> Art. 10
        - Standard threshold: >= 90% completeness
        - Simplified threshold: >= 80% completeness
        - Checks: supply chain mapping, geolocation coverage,
          satellite monitoring, document completeness

    QG-2 (Risk Assessment Coverage):
        - Phase transition: Art. 10 -> Art. 11
        - Standard threshold: >= 95% risk dimension coverage
        - Simplified threshold: >= 85% coverage
        - Checks: all 10 risk dimensions scored, composite score
          computed, Article 10(2) factors mapped

    QG-3 (Mitigation Adequacy):
        - Phase transition: Art. 11 -> Art. 12
        - Standard threshold: residual risk <= 15
        - Simplified threshold: residual risk <= 25
        - Checks: mitigation measures adequate, proportionality
          verified, evidence documented

Features:
    - Evaluate individual quality gate checks with weights and thresholds
    - Compute weighted aggregate gate scores
    - Support manual override with justification and audit trail
    - Generate remediation suggestions for failed checks
    - Track gate evaluation history for compliance audit
    - Deterministic scoring with Decimal arithmetic
    - Complete provenance tracking for all evaluations

Zero-Hallucination:
    - All gate scores are deterministic weighted averages
    - No LLM involvement in pass/fail determination
    - Thresholds come from validated configuration

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    DueDiligencePhase,
    QualityGateCheck,
    QualityGateEvaluation,
    QualityGateId,
    QualityGateResultEnum,
    WorkflowType,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate check definitions
# ---------------------------------------------------------------------------

#: QG-1 checks with weights (must sum to 1.0).
_QG1_CHECK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "Supply Chain Mapping Coverage",
        "description": "Percentage of supply chain nodes mapped with operator ID and coordinates",
        "weight": Decimal("0.25"),
        "source_agents": ["EUDR-001"],
    },
    {
        "name": "Geolocation Verification Coverage",
        "description": "Percentage of production plots with verified GPS coordinates",
        "weight": Decimal("0.20"),
        "source_agents": ["EUDR-002", "EUDR-006", "EUDR-007"],
    },
    {
        "name": "Satellite Monitoring Coverage",
        "description": "Percentage of production areas with satellite monitoring data",
        "weight": Decimal("0.15"),
        "source_agents": ["EUDR-003", "EUDR-004", "EUDR-005"],
    },
    {
        "name": "Chain of Custody Completeness",
        "description": "Percentage of supply chain links with documented custody",
        "weight": Decimal("0.15"),
        "source_agents": ["EUDR-009", "EUDR-010", "EUDR-011"],
    },
    {
        "name": "Multi-Tier Supplier Coverage",
        "description": "Percentage of supplier tiers mapped and verified",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-008"],
    },
    {
        "name": "Documentary Evidence Completeness",
        "description": "Percentage of required documents authenticated",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-012", "EUDR-013"],
    },
    {
        "name": "Traceability Code Coverage",
        "description": "Percentage of products with traceability codes generated",
        "weight": Decimal("0.05"),
        "source_agents": ["EUDR-014", "EUDR-015"],
    },
]

#: QG-2 checks with weights (must sum to 1.0).
_QG2_CHECK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "Country Risk Dimension",
        "description": "Country risk score from EUDR-016 available and valid",
        "weight": Decimal("0.15"),
        "source_agents": ["EUDR-016"],
    },
    {
        "name": "Supplier Risk Dimension",
        "description": "Supplier risk score from EUDR-017 available and valid",
        "weight": Decimal("0.12"),
        "source_agents": ["EUDR-017"],
    },
    {
        "name": "Commodity Risk Dimension",
        "description": "Commodity risk score from EUDR-018 available and valid",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-018"],
    },
    {
        "name": "Corruption Risk Dimension",
        "description": "Corruption risk score from EUDR-019 available and valid",
        "weight": Decimal("0.08"),
        "source_agents": ["EUDR-019"],
    },
    {
        "name": "Deforestation Risk Dimension",
        "description": "Deforestation risk score from EUDR-020 available and valid",
        "weight": Decimal("0.15"),
        "source_agents": ["EUDR-020"],
    },
    {
        "name": "Indigenous Rights Dimension",
        "description": "Indigenous rights risk score from EUDR-021 available and valid",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-021"],
    },
    {
        "name": "Protected Area Dimension",
        "description": "Protected area risk score from EUDR-022 available and valid",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-022"],
    },
    {
        "name": "Legal Compliance Dimension",
        "description": "Legal compliance risk score from EUDR-023 available and valid",
        "weight": Decimal("0.10"),
        "source_agents": ["EUDR-023"],
    },
    {
        "name": "Third-Party Audit Dimension",
        "description": "Audit risk score from EUDR-024 available and valid",
        "weight": Decimal("0.05"),
        "source_agents": ["EUDR-024"],
    },
    {
        "name": "Mitigation Readiness Dimension",
        "description": "Mitigation readiness score from EUDR-025 available and valid",
        "weight": Decimal("0.05"),
        "source_agents": ["EUDR-025"],
    },
]

#: QG-3 checks with weights (must sum to 1.0).
_QG3_CHECK_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "Mitigation Adequacy",
        "description": "Residual risk after mitigation meets target threshold",
        "weight": Decimal("0.40"),
        "source_agents": ["EUDR-025"],
    },
    {
        "name": "Mitigation Proportionality",
        "description": "Mitigation measures proportionate to identified risks",
        "weight": Decimal("0.25"),
        "source_agents": ["EUDR-025"],
    },
    {
        "name": "Evidence Documentation",
        "description": "Mitigation measures documented with supporting evidence",
        "weight": Decimal("0.20"),
        "source_agents": ["EUDR-025", "EUDR-012"],
    },
    {
        "name": "Stakeholder Engagement",
        "description": "Relevant stakeholders engaged in mitigation process",
        "weight": Decimal("0.15"),
        "source_agents": ["EUDR-025", "EUDR-024"],
    },
]


# ---------------------------------------------------------------------------
# QualityGateEngine
# ---------------------------------------------------------------------------


class QualityGateEngine:
    """Quality gate evaluation engine for due diligence phase transitions.

    Evaluates the three quality gates (QG-1, QG-2, QG-3) that enforce
    transition requirements between due diligence phases. Each gate
    consists of weighted checks that produce a composite score compared
    against configurable thresholds.

    All scoring uses deterministic Decimal arithmetic with no LLM
    involvement in pass/fail determination.

    Attributes:
        _config: Configuration with gate thresholds.

    Example:
        >>> engine = QualityGateEngine()
        >>> evaluation = engine.evaluate_gate(
        ...     workflow_id="wf-001",
        ...     gate_id=QualityGateId.QG1,
        ...     check_scores={"Supply Chain Mapping Coverage": Decimal("0.95")},
        ... )
        >>> assert evaluation.result in (QualityGateResultEnum.PASSED, ...)
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the QualityGateEngine.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()
        logger.info("QualityGateEngine initialized")

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    def evaluate_gate(
        self,
        workflow_id: str,
        gate_id: QualityGateId,
        check_scores: Dict[str, Decimal],
        workflow_type: WorkflowType = WorkflowType.STANDARD,
        override: bool = False,
        override_justification: Optional[str] = None,
        override_by: Optional[str] = None,
    ) -> QualityGateEvaluation:
        """Evaluate a quality gate with all its checks.

        Computes the weighted aggregate score across all checks and
        compares against the configured threshold for the gate and
        workflow type.

        Args:
            workflow_id: Parent workflow identifier.
            gate_id: Quality gate to evaluate (QG-1, QG-2, QG-3).
            check_scores: Per-check measured values (0-1 for QG-1/QG-2,
                0-100 for QG-3 residual risk).
            workflow_type: Standard or simplified workflow.
            override: Whether to override a failed gate.
            override_justification: Justification text for override.
            override_by: User requesting override.

        Returns:
            QualityGateEvaluation with result and all check details.

        Example:
            >>> engine = QualityGateEngine()
            >>> result = engine.evaluate_gate(
            ...     "wf-001", QualityGateId.QG1,
            ...     {"Supply Chain Mapping Coverage": Decimal("0.95")},
            ... )
        """
        start_time = _utcnow()

        # Get check definitions and threshold for this gate
        check_defs = self._get_check_definitions(gate_id)
        threshold = self._get_threshold(gate_id, workflow_type)
        phase_from, phase_to = self._get_phase_transition(gate_id)

        # Evaluate individual checks
        checks: List[QualityGateCheck] = []
        for check_def in check_defs:
            check_name = check_def["name"]
            check_weight = check_def["weight"]
            check_desc = check_def.get("description", "")
            source_agents = check_def.get("source_agents", [])

            measured = check_scores.get(check_name, Decimal("0"))

            # For QG-3, threshold is a maximum (residual risk)
            if gate_id == QualityGateId.QG3:
                check_passed = measured <= threshold
            else:
                check_passed = measured >= threshold

            remediation = None
            if not check_passed:
                remediation = self._get_remediation(
                    gate_id, check_name, measured, threshold
                )

            check = QualityGateCheck(
                check_id=_new_uuid(),
                name=check_name,
                description=check_desc,
                weight=check_weight,
                measured_value=measured,
                threshold=threshold,
                passed=check_passed,
                source_agents=source_agents,
                remediation=remediation,
                evidence={"measured": str(measured), "threshold": str(threshold)},
            )
            checks.append(check)

        # Compute weighted aggregate score
        weighted_score = self._compute_weighted_score(checks, gate_id)

        # Determine gate result
        if gate_id == QualityGateId.QG3:
            gate_passed = weighted_score <= threshold
        else:
            gate_passed = weighted_score >= threshold

        if gate_passed:
            result = QualityGateResultEnum.PASSED
        elif override:
            result = QualityGateResultEnum.OVERRIDDEN
            logger.warning(
                f"Quality gate {gate_id.value} overridden by {override_by}: "
                f"{override_justification}"
            )
        else:
            result = QualityGateResultEnum.FAILED

        # Build evaluation
        evaluation = QualityGateEvaluation(
            evaluation_id=_new_uuid(),
            workflow_id=workflow_id,
            gate_id=gate_id,
            phase_from=phase_from,
            phase_to=phase_to,
            result=result,
            weighted_score=weighted_score,
            threshold=threshold,
            checks=checks,
            override_justification=override_justification if override else None,
            override_by=override_by if override else None,
            evaluated_at=_utcnow(),
            provenance_hash=self._hash_evaluation(
                workflow_id, gate_id, weighted_score, result, checks
            ),
        )

        duration_ms = (_utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Quality gate {gate_id.value} evaluation for {workflow_id}: "
            f"score={weighted_score}, threshold={threshold}, "
            f"result={result.value} in {duration_ms:.1f}ms"
        )

        return evaluation

    # ------------------------------------------------------------------
    # Check definitions retrieval
    # ------------------------------------------------------------------

    def get_check_definitions(
        self,
        gate_id: QualityGateId,
    ) -> List[Dict[str, Any]]:
        """Get check definitions for a quality gate.

        Args:
            gate_id: Quality gate identifier.

        Returns:
            List of check definition dictionaries.
        """
        return self._get_check_definitions(gate_id)

    def get_threshold(
        self,
        gate_id: QualityGateId,
        workflow_type: WorkflowType = WorkflowType.STANDARD,
    ) -> Decimal:
        """Get the configured threshold for a quality gate.

        Args:
            gate_id: Quality gate identifier.
            workflow_type: Standard or simplified workflow.

        Returns:
            Threshold value (Decimal).
        """
        return self._get_threshold(gate_id, workflow_type)

    # ------------------------------------------------------------------
    # Gate check helpers
    # ------------------------------------------------------------------

    def get_failed_checks(
        self,
        evaluation: QualityGateEvaluation,
    ) -> List[QualityGateCheck]:
        """Get all failed checks from a gate evaluation.

        Args:
            evaluation: Completed gate evaluation.

        Returns:
            List of QualityGateCheck objects that did not pass.
        """
        return [c for c in evaluation.checks if not c.passed]

    def get_remediations(
        self,
        evaluation: QualityGateEvaluation,
    ) -> List[Dict[str, str]]:
        """Get remediation suggestions for all failed checks.

        Args:
            evaluation: Completed gate evaluation.

        Returns:
            List of dicts with check name, measured value, and remediation.
        """
        remediations: List[Dict[str, str]] = []
        for check in evaluation.checks:
            if not check.passed and check.remediation:
                remediations.append({
                    "check_name": check.name,
                    "measured_value": str(check.measured_value),
                    "threshold": str(check.threshold),
                    "remediation": check.remediation,
                })
        return remediations

    def is_gate_passed(
        self,
        evaluation: QualityGateEvaluation,
    ) -> bool:
        """Check whether a gate evaluation passed (including overrides).

        Args:
            evaluation: Completed gate evaluation.

        Returns:
            True if result is PASSED or OVERRIDDEN.
        """
        return evaluation.result in (
            QualityGateResultEnum.PASSED,
            QualityGateResultEnum.OVERRIDDEN,
        )

    # ------------------------------------------------------------------
    # Override management
    # ------------------------------------------------------------------

    def apply_override(
        self,
        evaluation: QualityGateEvaluation,
        justification: str,
        override_by: str,
    ) -> QualityGateEvaluation:
        """Apply a manual override to a failed gate evaluation.

        Creates a new evaluation record with OVERRIDDEN status.
        The original evaluation is preserved for audit trail.

        Args:
            evaluation: Failed gate evaluation to override.
            justification: Justification text (required).
            override_by: User applying the override (required).

        Returns:
            New QualityGateEvaluation with OVERRIDDEN result.

        Raises:
            ValueError: If evaluation already passed or justification empty.
        """
        if evaluation.result == QualityGateResultEnum.PASSED:
            raise ValueError("Cannot override an already-passed gate")

        if not justification or not justification.strip():
            raise ValueError("Override justification is required")

        if not override_by or not override_by.strip():
            raise ValueError("Override user identity is required")

        overridden = QualityGateEvaluation(
            evaluation_id=_new_uuid(),
            workflow_id=evaluation.workflow_id,
            gate_id=evaluation.gate_id,
            phase_from=evaluation.phase_from,
            phase_to=evaluation.phase_to,
            result=QualityGateResultEnum.OVERRIDDEN,
            weighted_score=evaluation.weighted_score,
            threshold=evaluation.threshold,
            checks=evaluation.checks,
            override_justification=justification.strip(),
            override_by=override_by.strip(),
            evaluated_at=_utcnow(),
            provenance_hash=self._hash_evaluation(
                evaluation.workflow_id,
                evaluation.gate_id,
                evaluation.weighted_score,
                QualityGateResultEnum.OVERRIDDEN,
                evaluation.checks,
            ),
        )

        logger.warning(
            f"Quality gate {evaluation.gate_id.value} overridden by "
            f"{override_by}: {justification[:100]}..."
        )

        return overridden

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_check_definitions(
        self,
        gate_id: QualityGateId,
    ) -> List[Dict[str, Any]]:
        """Get check definitions for a quality gate.

        Args:
            gate_id: Quality gate identifier.

        Returns:
            List of check definition dictionaries.
        """
        gate_checks: Dict[QualityGateId, List[Dict[str, Any]]] = {
            QualityGateId.QG1: _QG1_CHECK_DEFINITIONS,
            QualityGateId.QG2: _QG2_CHECK_DEFINITIONS,
            QualityGateId.QG3: _QG3_CHECK_DEFINITIONS,
        }
        return gate_checks.get(gate_id, [])

    def _get_threshold(
        self,
        gate_id: QualityGateId,
        workflow_type: WorkflowType,
    ) -> Decimal:
        """Get threshold for a gate and workflow type.

        Args:
            gate_id: Quality gate identifier.
            workflow_type: Standard or simplified.

        Returns:
            Threshold value.
        """
        is_simplified = (workflow_type == WorkflowType.SIMPLIFIED)

        thresholds: Dict[QualityGateId, Tuple[Decimal, Decimal]] = {
            QualityGateId.QG1: (
                self._config.qg1_completeness_threshold,
                self._config.qg1_simplified_threshold,
            ),
            QualityGateId.QG2: (
                self._config.qg2_coverage_threshold,
                self._config.qg2_simplified_threshold,
            ),
            QualityGateId.QG3: (
                self._config.qg3_residual_risk_threshold,
                self._config.qg3_simplified_threshold,
            ),
        }

        standard, simplified = thresholds.get(
            gate_id, (Decimal("0.90"), Decimal("0.80"))
        )
        return simplified if is_simplified else standard

    def _get_phase_transition(
        self,
        gate_id: QualityGateId,
    ) -> Tuple[DueDiligencePhase, DueDiligencePhase]:
        """Get the phase transition for a quality gate.

        Args:
            gate_id: Quality gate identifier.

        Returns:
            Tuple of (from_phase, to_phase).
        """
        transitions: Dict[
            QualityGateId,
            Tuple[DueDiligencePhase, DueDiligencePhase]
        ] = {
            QualityGateId.QG1: (
                DueDiligencePhase.INFORMATION_GATHERING,
                DueDiligencePhase.RISK_ASSESSMENT,
            ),
            QualityGateId.QG2: (
                DueDiligencePhase.RISK_ASSESSMENT,
                DueDiligencePhase.RISK_MITIGATION,
            ),
            QualityGateId.QG3: (
                DueDiligencePhase.RISK_MITIGATION,
                DueDiligencePhase.PACKAGE_GENERATION,
            ),
        }
        return transitions.get(
            gate_id,
            (DueDiligencePhase.INFORMATION_GATHERING,
             DueDiligencePhase.RISK_ASSESSMENT),
        )

    def _compute_weighted_score(
        self,
        checks: List[QualityGateCheck],
        gate_id: QualityGateId,
    ) -> Decimal:
        """Compute weighted aggregate score from individual checks.

        For QG-1 and QG-2: weighted average of measured values.
        For QG-3: weighted average of residual risk values (lower is better).

        Args:
            checks: List of evaluated checks.
            gate_id: Quality gate identifier.

        Returns:
            Weighted aggregate score (Decimal).
        """
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for check in checks:
            weighted_sum += check.weight * check.measured_value
            total_weight += check.weight

        if total_weight == Decimal("0"):
            return Decimal("0")

        score = (weighted_sum / total_weight).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        return score

    def _get_remediation(
        self,
        gate_id: QualityGateId,
        check_name: str,
        measured: Decimal,
        threshold: Decimal,
    ) -> str:
        """Generate remediation suggestion for a failed check.

        Args:
            gate_id: Quality gate identifier.
            check_name: Name of the failed check.
            measured: Measured value that failed.
            threshold: Required threshold.

        Returns:
            Remediation suggestion text.
        """
        if gate_id == QualityGateId.QG3:
            gap = measured - threshold
            return (
                f"Residual risk ({measured}) exceeds target ({threshold}) "
                f"by {gap}. Apply additional mitigation measures to reduce "
                f"risk in the '{check_name}' dimension."
            )

        gap = threshold - measured
        pct_gap = (gap * Decimal("100")).quantize(Decimal("0.1"))
        return (
            f"'{check_name}' measured at {measured} but requires "
            f"{threshold} (gap: {pct_gap}%). Re-run relevant agents "
            f"or supplement with additional data collection."
        )

    def _hash_evaluation(
        self,
        workflow_id: str,
        gate_id: QualityGateId,
        weighted_score: Decimal,
        result: QualityGateResultEnum,
        checks: List[QualityGateCheck],
    ) -> str:
        """Compute SHA-256 hash for a gate evaluation.

        Args:
            workflow_id: Workflow identifier.
            gate_id: Quality gate identifier.
            weighted_score: Computed weighted score.
            result: Gate result.
            checks: Individual check results.

        Returns:
            64-character hex SHA-256 hash.
        """
        check_data = [
            {
                "name": c.name,
                "weight": str(c.weight),
                "measured": str(c.measured_value),
                "threshold": str(c.threshold),
                "passed": c.passed,
            }
            for c in checks
        ]
        data = {
            "workflow_id": workflow_id,
            "gate_id": gate_id.value,
            "weighted_score": str(weighted_score),
            "result": result.value,
            "checks": check_data,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
