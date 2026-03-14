# -*- coding: utf-8 -*-
"""
QualityGateEngine - PACK-002 CSRD Professional Engine 3

Three-gate quality assurance engine ensuring CSRD reporting integrity
through progressive validation: Data Completeness (QG-1), Calculation
Integrity (QG-2), and Compliance Readiness (QG-3).

Quality Gates:
    QG-1 (Data Completeness):
        Verifies ESRS coverage, source freshness, quality scores,
        subsidiary completeness, and mandatory data point presence.

    QG-2 (Calculation Integrity):
        Validates scope completeness, dual reporting variance,
        cross-entity balance, intensity metrics, and base year consistency.

    QG-3 (Compliance Readiness):
        Confirms rule pass rate, XBRL validity, cross-framework
        consistency, auditor package completeness, and management assertions.

Features:
    - Weighted check scoring within each gate (weights sum to 1.0)
    - Configurable pass/fail thresholds per gate
    - Manual override with audit trail and justification
    - Remediation suggestions for failed checks
    - Gate dependency enforcement (QG-1 must pass before QG-2)
    - Decimal arithmetic for all scoring
    - SHA-256 provenance hashing on all evaluations

Zero-Hallucination:
    - All scores computed via deterministic weighted average
    - Pass/fail is threshold comparison only
    - No LLM involvement in scoring or pass/fail decisions
    - Override decisions are human-authored, engine only records them

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualityGateId(IntEnum):
    """Quality gate identifiers."""

    DATA_COMPLETENESS = 1
    CALCULATION_INTEGRITY = 2
    COMPLIANCE_READINESS = 3


class RemediationPriority(str, Enum):
    """Priority level for remediation suggestions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class GateCheckDefinition(BaseModel):
    """Definition of a single check within a quality gate."""

    check_id: str = Field(..., description="Unique check identifier")
    name: str = Field(..., description="Human-readable check name")
    description: str = Field("", description="What this check evaluates")
    weight: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Check weight (gate weights must sum to 1.0)",
    )
    evaluate_fn_name: str = Field(
        ..., description="Name of the evaluation function to call"
    )


class GateCheckResult(BaseModel):
    """Result of a single gate check evaluation."""

    check_id: str = Field(..., description="Check identifier")
    name: str = Field("", description="Check name")
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Score from 0 to 100",
    )
    passed: bool = Field(..., description="Whether this check passed")
    details: str = Field("", description="Human-readable explanation")
    evidence: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence data"
    )
    weight: Decimal = Field(Decimal("0"), description="Weight applied to this check")


class GateOverride(BaseModel):
    """Manual override of a quality gate result."""

    override_id: str = Field(default_factory=_new_uuid, description="Override ID")
    overrider: str = Field(..., description="Username of person overriding")
    justification: str = Field(
        ..., min_length=10, description="Justification for override"
    )
    timestamp: datetime = Field(default_factory=_utcnow, description="When overridden")
    original_score: Decimal = Field(..., description="Score before override")
    provenance_hash: str = Field("", description="SHA-256 hash")


class QualityGateResult(BaseModel):
    """Result of evaluating a single quality gate."""

    evaluation_id: str = Field(default_factory=_new_uuid, description="Evaluation ID")
    gate_id: QualityGateId = Field(..., description="Which gate was evaluated")
    gate_name: str = Field("", description="Human-readable gate name")
    checks: List[GateCheckResult] = Field(
        default_factory=list, description="Individual check results"
    )
    aggregate_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted aggregate score",
    )
    threshold: Decimal = Field(
        Decimal("80"), description="Pass threshold (0-100)"
    )
    passed: bool = Field(False, description="Whether the gate passed")
    override: Optional[GateOverride] = Field(
        None, description="Manual override if applied"
    )
    evaluated_at: datetime = Field(default_factory=_utcnow, description="Evaluation time")
    provenance_hash: str = Field("", description="SHA-256 hash")


class RemediationSuggestion(BaseModel):
    """Suggested remediation for a failed check."""

    suggestion_id: str = Field(default_factory=_new_uuid, description="Suggestion ID")
    check_id: str = Field(..., description="Failed check identifier")
    issue: str = Field(..., description="What is wrong")
    suggestion: str = Field(..., description="How to fix it")
    priority: RemediationPriority = Field(..., description="Fix priority")
    estimated_effort_hours: float = Field(
        0.0, ge=0.0, description="Estimated hours to remediate"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class QualityGateConfig(BaseModel):
    """Configuration for the quality gate engine."""

    qg1_threshold: Decimal = Field(
        Decimal("80"), description="QG-1 Data Completeness pass threshold"
    )
    qg2_threshold: Decimal = Field(
        Decimal("85"), description="QG-2 Calculation Integrity pass threshold"
    )
    qg3_threshold: Decimal = Field(
        Decimal("90"), description="QG-3 Compliance Readiness pass threshold"
    )
    enforce_gate_order: bool = Field(
        True, description="Require gates to pass in order (QG-1 before QG-2)"
    )
    allow_overrides: bool = Field(
        True, description="Allow manual overrides of gate results"
    )


# ---------------------------------------------------------------------------
# Gate Check Definitions
# ---------------------------------------------------------------------------

_QG1_CHECKS: List[GateCheckDefinition] = [
    GateCheckDefinition(
        check_id="qg1_esrs_coverage",
        name="ESRS Data Point Coverage",
        description="Percentage of mandatory ESRS data points with values",
        weight=Decimal("0.25"),
        evaluate_fn_name="_eval_esrs_coverage",
    ),
    GateCheckDefinition(
        check_id="qg1_source_freshness",
        name="Data Source Freshness",
        description="Percentage of data sources within reporting period",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_source_freshness",
    ),
    GateCheckDefinition(
        check_id="qg1_quality_score",
        name="Data Quality Score",
        description="Average quality score across all data points",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_quality_score",
    ),
    GateCheckDefinition(
        check_id="qg1_subsidiary_completeness",
        name="Subsidiary Data Completeness",
        description="Percentage of subsidiaries that have submitted data",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_subsidiary_completeness",
    ),
    GateCheckDefinition(
        check_id="qg1_mandatory_points",
        name="Mandatory Data Points Present",
        description="All Phase-1 mandatory ESRS data points are populated",
        weight=Decimal("0.15"),
        evaluate_fn_name="_eval_mandatory_points",
    ),
]

_QG2_CHECKS: List[GateCheckDefinition] = [
    GateCheckDefinition(
        check_id="qg2_scope_completeness",
        name="Scope 1/2/3 Completeness",
        description="All applicable emission scopes are calculated",
        weight=Decimal("0.25"),
        evaluate_fn_name="_eval_scope_completeness",
    ),
    GateCheckDefinition(
        check_id="qg2_dual_reporting_variance",
        name="Dual Reporting Variance",
        description="Variance between location-based and market-based Scope 2",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_dual_reporting_variance",
    ),
    GateCheckDefinition(
        check_id="qg2_cross_entity_balance",
        name="Cross-Entity Balance Check",
        description="Consolidated total matches sum of entity contributions",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_cross_entity_balance",
    ),
    GateCheckDefinition(
        check_id="qg2_intensity_metrics",
        name="Intensity Metrics Validity",
        description="Emission intensity metrics are correctly calculated",
        weight=Decimal("0.15"),
        evaluate_fn_name="_eval_intensity_metrics",
    ),
    GateCheckDefinition(
        check_id="qg2_base_year_consistency",
        name="Base Year Consistency",
        description="Base year recalculation policy is consistently applied",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_base_year_consistency",
    ),
]

_QG3_CHECKS: List[GateCheckDefinition] = [
    GateCheckDefinition(
        check_id="qg3_rule_pass_rate",
        name="Validation Rule Pass Rate",
        description="Percentage of ESRS validation rules passing",
        weight=Decimal("0.25"),
        evaluate_fn_name="_eval_rule_pass_rate",
    ),
    GateCheckDefinition(
        check_id="qg3_xbrl_validity",
        name="XBRL Taxonomy Validity",
        description="Report validates against EFRAG XBRL taxonomy",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_xbrl_validity",
    ),
    GateCheckDefinition(
        check_id="qg3_cross_framework",
        name="Cross-Framework Consistency",
        description="ESRS data is consistent with GRI, CDP, TCFD mappings",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_cross_framework",
    ),
    GateCheckDefinition(
        check_id="qg3_auditor_package",
        name="Auditor Package Completeness",
        description="All required documents for limited assurance are present",
        weight=Decimal("0.20"),
        evaluate_fn_name="_eval_auditor_package",
    ),
    GateCheckDefinition(
        check_id="qg3_management_assertions",
        name="Management Assertions",
        description="Management assertions are documented and signed",
        weight=Decimal("0.15"),
        evaluate_fn_name="_eval_management_assertions",
    ),
]

_GATE_DEFINITIONS: Dict[QualityGateId, Tuple[str, List[GateCheckDefinition]]] = {
    QualityGateId.DATA_COMPLETENESS: ("Data Completeness", _QG1_CHECKS),
    QualityGateId.CALCULATION_INTEGRITY: ("Calculation Integrity", _QG2_CHECKS),
    QualityGateId.COMPLIANCE_READINESS: ("Compliance Readiness", _QG3_CHECKS),
}

# Remediation templates per check_id
_REMEDIATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "qg1_esrs_coverage": {
        "issue": "Insufficient ESRS data point coverage",
        "suggestion": "Review ESRS disclosure requirements and populate missing mandatory data points using the data collection templates.",
        "priority": RemediationPriority.CRITICAL,
        "effort": 16.0,
    },
    "qg1_source_freshness": {
        "issue": "Data sources are outdated",
        "suggestion": "Update data sources to current reporting period. Refresh ERP extractions and supplier questionnaires.",
        "priority": RemediationPriority.HIGH,
        "effort": 8.0,
    },
    "qg1_quality_score": {
        "issue": "Data quality score below threshold",
        "suggestion": "Run data quality profiler and fix detected issues. Address outliers, missing values, and format inconsistencies.",
        "priority": RemediationPriority.HIGH,
        "effort": 12.0,
    },
    "qg1_subsidiary_completeness": {
        "issue": "Missing subsidiary data submissions",
        "suggestion": "Contact subsidiaries with outstanding data submissions. Provide data collection templates and set deadlines.",
        "priority": RemediationPriority.CRITICAL,
        "effort": 24.0,
    },
    "qg1_mandatory_points": {
        "issue": "Mandatory ESRS data points missing",
        "suggestion": "Identify Phase-1 mandatory data points from ESRS Set 1 and ensure all are populated with validated values.",
        "priority": RemediationPriority.CRITICAL,
        "effort": 8.0,
    },
    "qg2_scope_completeness": {
        "issue": "Incomplete emission scope coverage",
        "suggestion": "Ensure Scope 1, 2, and material Scope 3 categories are calculated. Run the scope completeness checker.",
        "priority": RemediationPriority.CRITICAL,
        "effort": 16.0,
    },
    "qg2_dual_reporting_variance": {
        "issue": "Excessive variance between location-based and market-based Scope 2",
        "suggestion": "Review Scope 2 calculations. Verify grid emission factors (location) and contractual instruments (market).",
        "priority": RemediationPriority.HIGH,
        "effort": 8.0,
    },
    "qg2_cross_entity_balance": {
        "issue": "Consolidated totals do not match entity sum",
        "suggestion": "Run consolidation reconciliation report. Check intercompany elimination entries and ownership percentages.",
        "priority": RemediationPriority.HIGH,
        "effort": 12.0,
    },
    "qg2_intensity_metrics": {
        "issue": "Intensity metrics incorrectly calculated",
        "suggestion": "Verify denominators (revenue, FTE, production units) match audited financial data. Recalculate intensity ratios.",
        "priority": RemediationPriority.MEDIUM,
        "effort": 4.0,
    },
    "qg2_base_year_consistency": {
        "issue": "Base year recalculation inconsistency",
        "suggestion": "Review base year policy triggers (M&A, methodology changes). Apply recalculation if significance threshold exceeded.",
        "priority": RemediationPriority.MEDIUM,
        "effort": 8.0,
    },
    "qg3_rule_pass_rate": {
        "issue": "Validation rules failing",
        "suggestion": "Run full ESRS validation suite. Fix reported errors starting with critical rules, then high and medium.",
        "priority": RemediationPriority.CRITICAL,
        "effort": 16.0,
    },
    "qg3_xbrl_validity": {
        "issue": "XBRL taxonomy validation errors",
        "suggestion": "Run EFRAG XBRL taxonomy validator. Fix tagging errors, missing elements, and calculation linkbase issues.",
        "priority": RemediationPriority.HIGH,
        "effort": 12.0,
    },
    "qg3_cross_framework": {
        "issue": "Cross-framework data inconsistencies",
        "suggestion": "Run cross-framework mapper to identify discrepancies between ESRS, GRI, CDP, and TCFD data points.",
        "priority": RemediationPriority.MEDIUM,
        "effort": 8.0,
    },
    "qg3_auditor_package": {
        "issue": "Auditor package incomplete",
        "suggestion": "Compile all required assurance documents: methodology notes, data trails, calculation evidence, and sign-off sheets.",
        "priority": RemediationPriority.HIGH,
        "effort": 16.0,
    },
    "qg3_management_assertions": {
        "issue": "Management assertions not documented",
        "suggestion": "Prepare management assertion letter covering completeness, accuracy, and compliance. Obtain required signatures.",
        "priority": RemediationPriority.HIGH,
        "effort": 4.0,
    },
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class QualityGateEngine:
    """Three-gate quality assurance engine for CSRD reporting.

    Evaluates data completeness, calculation integrity, and compliance
    readiness through weighted check scoring with configurable thresholds.

    Attributes:
        config: Engine configuration with gate thresholds.
        gate_results: History of gate evaluations.
        overrides: Applied manual overrides.

    Example:
        >>> engine = QualityGateEngine()
        >>> result = await engine.evaluate_gate(QualityGateId.DATA_COMPLETENESS, data)
        >>> if not result.passed:
        ...     suggestions = await engine.get_remediation_suggestions(result)
    """

    def __init__(self, config: Optional[QualityGateConfig] = None) -> None:
        """Initialize QualityGateEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or QualityGateConfig()
        self.gate_results: Dict[QualityGateId, QualityGateResult] = {}
        self.overrides: List[GateOverride] = []
        self._evaluation_history: List[QualityGateResult] = []
        logger.info(
            "QualityGateEngine initialized (thresholds: QG1=%s, QG2=%s, QG3=%s)",
            self.config.qg1_threshold,
            self.config.qg2_threshold,
            self.config.qg3_threshold,
        )

    # -- Gate Evaluation ----------------------------------------------------

    async def evaluate_gate(
        self, gate_id: QualityGateId, data: Dict[str, Any]
    ) -> QualityGateResult:
        """Evaluate a single quality gate.

        Runs all checks defined for the gate, computes weighted aggregate
        score, and determines pass/fail against the configured threshold.

        Args:
            gate_id: Which gate to evaluate.
            data: Input data containing metrics for check evaluation.

        Returns:
            QualityGateResult with check scores and pass/fail status.

        Raises:
            ValueError: If gate order is enforced and prerequisite gates
                have not passed.
        """
        if self.config.enforce_gate_order:
            self._enforce_gate_order(gate_id)

        gate_name, check_defs = _GATE_DEFINITIONS[gate_id]
        threshold = self._get_threshold(gate_id)

        logger.info("Evaluating gate %s (%s)", gate_id.name, gate_name)

        check_results: List[GateCheckResult] = []
        for check_def in check_defs:
            result = self._evaluate_check(check_def, data, threshold)
            check_results.append(result)

        # Compute weighted aggregate
        aggregate = self._compute_aggregate(check_results)
        passed = aggregate >= threshold

        gate_result = QualityGateResult(
            gate_id=gate_id,
            gate_name=gate_name,
            checks=check_results,
            aggregate_score=aggregate,
            threshold=threshold,
            passed=passed,
        )
        gate_result.provenance_hash = _compute_hash(gate_result)

        self.gate_results[gate_id] = gate_result
        self._evaluation_history.append(gate_result)

        logger.info(
            "Gate %s evaluation: score=%s, threshold=%s, passed=%s",
            gate_name,
            aggregate,
            threshold,
            passed,
        )
        return gate_result

    async def evaluate_all_gates(
        self, data: Dict[str, Any]
    ) -> List[QualityGateResult]:
        """Evaluate all three quality gates in order.

        Args:
            data: Input data for evaluation.

        Returns:
            List of QualityGateResult for all gates.
        """
        results: List[QualityGateResult] = []
        for gate_id in QualityGateId:
            try:
                result = await self.evaluate_gate(gate_id, data)
                results.append(result)
                if not result.passed and self.config.enforce_gate_order:
                    logger.warning(
                        "Gate %s failed, stopping sequential evaluation",
                        gate_id.name,
                    )
                    break
            except ValueError as exc:
                logger.warning("Gate %s skipped: %s", gate_id.name, exc)
                break
        return results

    # -- Override -----------------------------------------------------------

    async def override_gate(
        self, gate_id: QualityGateId, override: GateOverride
    ) -> QualityGateResult:
        """Manually override a gate result.

        Args:
            gate_id: Gate to override.
            override: Override details with justification.

        Returns:
            Updated QualityGateResult with override applied.

        Raises:
            ValueError: If overrides are disabled or gate not yet evaluated.
        """
        if not self.config.allow_overrides:
            raise ValueError("Manual overrides are disabled in configuration")

        current_result = self.gate_results.get(gate_id)
        if current_result is None:
            raise ValueError(f"Gate {gate_id.name} has not been evaluated yet")

        override.original_score = current_result.aggregate_score
        override.provenance_hash = _compute_hash(override)

        current_result.override = override
        current_result.passed = True  # Override forces pass
        current_result.provenance_hash = _compute_hash(current_result)

        self.overrides.append(override)
        self.gate_results[gate_id] = current_result

        logger.warning(
            "Gate %s manually overridden by %s (original_score=%s): %s",
            gate_id.name,
            override.overrider,
            override.original_score,
            override.justification,
        )
        return current_result

    # -- Remediation --------------------------------------------------------

    async def get_remediation_suggestions(
        self, result: QualityGateResult
    ) -> List[RemediationSuggestion]:
        """Generate remediation suggestions for failed checks.

        Args:
            result: Gate evaluation result.

        Returns:
            List of remediation suggestions for checks that did not pass.
        """
        suggestions: List[RemediationSuggestion] = []

        for check in result.checks:
            if check.passed:
                continue

            template = _REMEDIATION_TEMPLATES.get(check.check_id)
            if template:
                suggestion = RemediationSuggestion(
                    check_id=check.check_id,
                    issue=template["issue"],
                    suggestion=template["suggestion"],
                    priority=template["priority"],
                    estimated_effort_hours=template["effort"],
                )
            else:
                suggestion = RemediationSuggestion(
                    check_id=check.check_id,
                    issue=f"Check '{check.name}' scored {check.score}/100",
                    suggestion="Review the check details and improve the underlying data or process.",
                    priority=RemediationPriority.MEDIUM,
                    estimated_effort_hours=4.0,
                )
            suggestions.append(suggestion)

        # Sort by priority (critical first)
        priority_order = {
            RemediationPriority.CRITICAL: 0,
            RemediationPriority.HIGH: 1,
            RemediationPriority.MEDIUM: 2,
            RemediationPriority.LOW: 3,
        }
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))

        return suggestions

    # -- Progression Check --------------------------------------------------

    async def can_proceed(
        self, gate_results: Optional[List[QualityGateResult]] = None
    ) -> bool:
        """Check if all gates pass (or are overridden).

        Args:
            gate_results: Specific results to check. Uses stored results if None.

        Returns:
            True if all gates pass.
        """
        results = gate_results or list(self.gate_results.values())

        if len(results) < len(QualityGateId):
            logger.warning(
                "Not all gates have been evaluated (%d/%d)",
                len(results),
                len(QualityGateId),
            )
            return False

        return all(r.passed for r in results)

    # -- Internal Check Evaluation ------------------------------------------

    def _evaluate_check(
        self,
        check_def: GateCheckDefinition,
        data: Dict[str, Any],
        gate_threshold: Decimal,
    ) -> GateCheckResult:
        """Evaluate a single gate check.

        Args:
            check_def: Check definition.
            data: Input data.
            gate_threshold: Gate-level pass threshold.

        Returns:
            GateCheckResult with score and pass/fail.
        """
        eval_fn = getattr(self, check_def.evaluate_fn_name, None)
        if eval_fn is not None:
            score, details, evidence = eval_fn(data)
        else:
            score, details, evidence = self._eval_generic(
                check_def.check_id, data
            )

        score_decimal = _decimal(score).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        score_decimal = max(Decimal("0"), min(Decimal("100"), score_decimal))

        return GateCheckResult(
            check_id=check_def.check_id,
            name=check_def.name,
            score=score_decimal,
            passed=score_decimal >= gate_threshold,
            details=details,
            evidence=evidence,
            weight=check_def.weight,
        )

    def _compute_aggregate(
        self, check_results: List[GateCheckResult]
    ) -> Decimal:
        """Compute weighted aggregate score from check results.

        Args:
            check_results: List of individual check results.

        Returns:
            Weighted average score (0-100).
        """
        total_weight = sum(cr.weight for cr in check_results)
        if total_weight == Decimal("0"):
            return Decimal("0")

        weighted_sum = sum(cr.score * cr.weight for cr in check_results)
        aggregate = (weighted_sum / total_weight).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return max(Decimal("0"), min(Decimal("100"), aggregate))

    def _get_threshold(self, gate_id: QualityGateId) -> Decimal:
        """Get the pass threshold for a gate."""
        thresholds = {
            QualityGateId.DATA_COMPLETENESS: self.config.qg1_threshold,
            QualityGateId.CALCULATION_INTEGRITY: self.config.qg2_threshold,
            QualityGateId.COMPLIANCE_READINESS: self.config.qg3_threshold,
        }
        return thresholds.get(gate_id, Decimal("80"))

    def _enforce_gate_order(self, gate_id: QualityGateId) -> None:
        """Enforce that prerequisite gates have passed.

        Args:
            gate_id: Gate being evaluated.

        Raises:
            ValueError: If a prerequisite gate has not passed.
        """
        if gate_id == QualityGateId.DATA_COMPLETENESS:
            return  # First gate, no prerequisites

        if gate_id == QualityGateId.CALCULATION_INTEGRITY:
            qg1 = self.gate_results.get(QualityGateId.DATA_COMPLETENESS)
            if qg1 is None or not qg1.passed:
                raise ValueError(
                    "QG-1 (Data Completeness) must pass before evaluating QG-2"
                )

        if gate_id == QualityGateId.COMPLIANCE_READINESS:
            qg2 = self.gate_results.get(QualityGateId.CALCULATION_INTEGRITY)
            if qg2 is None or not qg2.passed:
                raise ValueError(
                    "QG-2 (Calculation Integrity) must pass before evaluating QG-3"
                )

    # -- Check Evaluation Functions -----------------------------------------

    def _eval_esrs_coverage(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate ESRS data point coverage."""
        total = int(data.get("esrs_total_points", 0))
        populated = int(data.get("esrs_populated_points", 0))

        if total == 0:
            return 0.0, "No ESRS data points defined", {"total": 0, "populated": 0}

        coverage = (populated / total) * 100
        return (
            coverage,
            f"{populated}/{total} ESRS data points populated ({coverage:.1f}%)",
            {"total": total, "populated": populated, "coverage_pct": round(coverage, 2)},
        )

    def _eval_source_freshness(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate data source freshness."""
        total_sources = int(data.get("total_data_sources", 0))
        fresh_sources = int(data.get("fresh_data_sources", 0))

        if total_sources == 0:
            return 0.0, "No data sources registered", {}

        freshness = (fresh_sources / total_sources) * 100
        return (
            freshness,
            f"{fresh_sources}/{total_sources} sources within reporting period ({freshness:.1f}%)",
            {"total": total_sources, "fresh": fresh_sources},
        )

    def _eval_quality_score(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate average data quality score."""
        score = float(data.get("average_quality_score", 0.0))
        return (
            score,
            f"Average data quality score: {score:.1f}/100",
            {"average_quality_score": score},
        )

    def _eval_subsidiary_completeness(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate subsidiary data submission completeness."""
        total_subs = int(data.get("total_subsidiaries", 0))
        submitted = int(data.get("subsidiaries_submitted", 0))

        if total_subs == 0:
            return 100.0, "No subsidiaries (standalone entity)", {}

        completeness = (submitted / total_subs) * 100
        return (
            completeness,
            f"{submitted}/{total_subs} subsidiaries submitted ({completeness:.1f}%)",
            {"total": total_subs, "submitted": submitted},
        )

    def _eval_mandatory_points(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate mandatory data point presence."""
        mandatory_total = int(data.get("mandatory_points_total", 0))
        mandatory_present = int(data.get("mandatory_points_present", 0))

        if mandatory_total == 0:
            return 100.0, "No mandatory points defined", {}

        coverage = (mandatory_present / mandatory_total) * 100
        return (
            coverage,
            f"{mandatory_present}/{mandatory_total} mandatory points present ({coverage:.1f}%)",
            {"total": mandatory_total, "present": mandatory_present},
        )

    def _eval_scope_completeness(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate emission scope calculation completeness."""
        scopes_required = int(data.get("scopes_required", 3))
        scopes_calculated = int(data.get("scopes_calculated", 0))

        if scopes_required == 0:
            return 100.0, "No scopes required", {}

        completeness = (scopes_calculated / scopes_required) * 100
        return (
            completeness,
            f"{scopes_calculated}/{scopes_required} emission scopes calculated",
            {"required": scopes_required, "calculated": scopes_calculated},
        )

    def _eval_dual_reporting_variance(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate variance between location and market-based Scope 2."""
        variance_pct = float(data.get("dual_reporting_variance_pct", 0.0))

        # Lower variance is better; map to a score
        if variance_pct <= 5.0:
            score = 100.0
        elif variance_pct <= 10.0:
            score = 90.0
        elif variance_pct <= 20.0:
            score = 75.0
        elif variance_pct <= 30.0:
            score = 60.0
        else:
            score = max(0.0, 100.0 - variance_pct * 2)

        return (
            score,
            f"Dual reporting variance: {variance_pct:.1f}%",
            {"variance_pct": variance_pct, "computed_score": score},
        )

    def _eval_cross_entity_balance(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate cross-entity balance check."""
        entity_sum = float(data.get("entity_sum_total", 0.0))
        consolidated_total = float(data.get("consolidated_total", 0.0))

        if consolidated_total == 0 and entity_sum == 0:
            return 100.0, "Both entity sum and consolidated total are zero", {}

        if consolidated_total == 0:
            return 0.0, "Consolidated total is zero but entity sum is not", {}

        variance = abs(entity_sum - consolidated_total) / abs(consolidated_total) * 100
        score = max(0.0, 100.0 - variance * 10)

        return (
            score,
            f"Entity sum vs consolidated variance: {variance:.2f}%",
            {"entity_sum": entity_sum, "consolidated": consolidated_total, "variance_pct": variance},
        )

    def _eval_intensity_metrics(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate intensity metrics validity."""
        metrics_defined = int(data.get("intensity_metrics_defined", 0))
        metrics_valid = int(data.get("intensity_metrics_valid", 0))

        if metrics_defined == 0:
            return 100.0, "No intensity metrics defined", {}

        validity = (metrics_valid / metrics_defined) * 100
        return (
            validity,
            f"{metrics_valid}/{metrics_defined} intensity metrics valid ({validity:.1f}%)",
            {"defined": metrics_defined, "valid": metrics_valid},
        )

    def _eval_base_year_consistency(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate base year recalculation consistency."""
        is_consistent = data.get("base_year_consistent", True)
        recalc_needed = data.get("base_year_recalc_needed", False)
        recalc_done = data.get("base_year_recalc_done", False)

        if not recalc_needed:
            score = 100.0 if is_consistent else 50.0
        elif recalc_done:
            score = 100.0
        else:
            score = 30.0

        return (
            score,
            f"Base year consistent={is_consistent}, recalc_needed={recalc_needed}, recalc_done={recalc_done}",
            {"consistent": is_consistent, "recalc_needed": recalc_needed, "recalc_done": recalc_done},
        )

    def _eval_rule_pass_rate(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate validation rule pass rate."""
        total_rules = int(data.get("total_validation_rules", 0))
        rules_passed = int(data.get("validation_rules_passed", 0))

        if total_rules == 0:
            return 100.0, "No validation rules defined", {}

        pass_rate = (rules_passed / total_rules) * 100
        return (
            pass_rate,
            f"{rules_passed}/{total_rules} validation rules pass ({pass_rate:.1f}%)",
            {"total": total_rules, "passed": rules_passed},
        )

    def _eval_xbrl_validity(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate XBRL taxonomy validity."""
        xbrl_errors = int(data.get("xbrl_validation_errors", 0))
        xbrl_warnings = int(data.get("xbrl_validation_warnings", 0))

        if xbrl_errors == 0 and xbrl_warnings == 0:
            score = 100.0
        elif xbrl_errors == 0:
            score = max(80.0, 100.0 - xbrl_warnings * 2)
        else:
            score = max(0.0, 100.0 - xbrl_errors * 10 - xbrl_warnings * 2)

        return (
            score,
            f"XBRL: {xbrl_errors} errors, {xbrl_warnings} warnings",
            {"errors": xbrl_errors, "warnings": xbrl_warnings},
        )

    def _eval_cross_framework(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate cross-framework consistency."""
        frameworks_mapped = int(data.get("frameworks_mapped", 0))
        frameworks_consistent = int(data.get("frameworks_consistent", 0))

        if frameworks_mapped == 0:
            return 100.0, "No cross-framework mappings", {}

        consistency = (frameworks_consistent / frameworks_mapped) * 100
        return (
            consistency,
            f"{frameworks_consistent}/{frameworks_mapped} frameworks consistent ({consistency:.1f}%)",
            {"mapped": frameworks_mapped, "consistent": frameworks_consistent},
        )

    def _eval_auditor_package(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate auditor package completeness."""
        docs_required = int(data.get("auditor_docs_required", 0))
        docs_present = int(data.get("auditor_docs_present", 0))

        if docs_required == 0:
            return 100.0, "No auditor documents required", {}

        completeness = (docs_present / docs_required) * 100
        return (
            completeness,
            f"{docs_present}/{docs_required} auditor documents present ({completeness:.1f}%)",
            {"required": docs_required, "present": docs_present},
        )

    def _eval_management_assertions(
        self, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Evaluate management assertions documentation."""
        assertions_required = int(data.get("assertions_required", 0))
        assertions_signed = int(data.get("assertions_signed", 0))

        if assertions_required == 0:
            return 100.0, "No management assertions required", {}

        completeness = (assertions_signed / assertions_required) * 100
        return (
            completeness,
            f"{assertions_signed}/{assertions_required} assertions signed ({completeness:.1f}%)",
            {"required": assertions_required, "signed": assertions_signed},
        )

    def _eval_generic(
        self, check_id: str, data: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Generic fallback evaluation using data keyed by check_id.

        Args:
            check_id: Check identifier to look up in data.
            data: Input data dictionary.

        Returns:
            Tuple of (score, details, evidence).
        """
        score = float(data.get(f"{check_id}_score", 0.0))
        return (
            score,
            f"Generic evaluation for {check_id}: {score:.1f}",
            {"source": "generic", "raw_score": score},
        )

    # -- History ------------------------------------------------------------

    def get_evaluation_history(self) -> List[QualityGateResult]:
        """Return all gate evaluations in chronological order.

        Returns:
            List of QualityGateResult objects.
        """
        return list(self._evaluation_history)

    def reset(self) -> None:
        """Reset engine state, clearing all results and overrides."""
        self.gate_results.clear()
        self.overrides.clear()
        self._evaluation_history.clear()
        logger.info("QualityGateEngine reset")
