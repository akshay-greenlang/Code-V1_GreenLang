# -*- coding: utf-8 -*-
"""
PolicyComplianceEngine - PACK-006 EUDR Starter Engine 7
=========================================================

EUDR compliance rule enforcement engine implementing 45 compliance rules
across 7 categories. Evaluates operator and trader compliance against
the full scope of Regulation (EU) 2023/1115 requirements.

Key Capabilities:
    - 45 compliance rules across 7 regulatory categories
    - Automated rule evaluation with severity classification
    - Compliance score calculation (0-100)
    - Simplified due diligence eligibility assessment
    - Penalty exposure estimation
    - Remediation plan generation
    - Complete audit trail for compliance evaluations

Rule Categories (45 rules total):
    1. GEOLOCATION (8 rules): precision, polygon, area, country, overlap, WGS84, batch, cutoff
    2. COMMODITY (6 rules): CN validity, Annex I, derived, multi-commodity, description, quantity
    3. SUPPLIER (7 rules): profile, EORI, tier depth, certification, engagement, quality, DD status
    4. RISK (6 rules): composite, benchmark, simplified DD, documentation, threshold, trend
    5. DDS (8 rules): Annex II, declaration, geolocation format, risk summary, mitigation,
       reference, deadline, amendment
    6. DOCUMENTATION (5 rules): authenticity, validity, chain, retention (5 years), versioning
    7. CUTOFF (5 rules): date verification, evidence, land use, declaration, exemption

Severity Levels:
    - CRITICAL: Immediate compliance failure, blocks market placement
    - HIGH: Significant gap, must be resolved before submission
    - MEDIUM: Notable issue, should be addressed
    - LOW: Minor concern, advisory only

Zero-Hallucination:
    - All rule evaluations use deterministic boolean logic
    - No LLM involvement in any compliance determination
    - SHA-256 provenance hashing on every output
    - Pydantic validation at all input/output boundaries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-006 EUDR Starter
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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


class RuleSeverity(str, Enum):
    """Compliance rule severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RuleCategory(str, Enum):
    """Compliance rule categories."""

    GEOLOCATION = "GEOLOCATION"
    COMMODITY = "COMMODITY"
    SUPPLIER = "SUPPLIER"
    RISK = "RISK"
    DDS = "DDS"
    DOCUMENTATION = "DOCUMENTATION"
    CUTOFF = "CUTOFF"


class RuleStatus(str, Enum):
    """Rule evaluation result status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


class RemediationPriority(str, Enum):
    """Remediation action priority."""

    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ComplianceRule(BaseModel):
    """Definition of a single compliance rule."""

    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(default="", description="Rule description")
    category: RuleCategory = Field(..., description="Rule category")
    severity: RuleSeverity = Field(..., description="Rule severity")
    article_reference: str = Field(default="", description="EUDR article reference")
    is_blocking: bool = Field(default=False, description="Whether failure blocks submission")


class RuleResult(BaseModel):
    """Result of evaluating a single compliance rule."""

    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(default="", description="Rule name")
    category: RuleCategory = Field(..., description="Rule category")
    severity: RuleSeverity = Field(..., description="Rule severity")
    status: RuleStatus = Field(..., description="Evaluation result")
    is_compliant: bool = Field(default=False, description="Whether rule passed")
    message: str = Field(default="", description="Result message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    article_reference: str = Field(default="", description="EUDR article reference")
    remediation_hint: Optional[str] = Field(None, description="Suggested remediation action")
    evaluated_at: datetime = Field(default_factory=_utcnow, description="Evaluation timestamp")


class ComplianceResult(BaseModel):
    """Result of evaluating all compliance rules."""

    evaluation_id: str = Field(default_factory=_new_uuid, description="Evaluation identifier")
    total_rules: int = Field(default=0, description="Total rules evaluated")
    passed_rules: int = Field(default=0, description="Rules that passed")
    failed_rules: int = Field(default=0, description="Rules that failed")
    warning_rules: int = Field(default=0, description="Rules with warnings")
    not_applicable_rules: int = Field(default=0, description="Rules not applicable")
    error_rules: int = Field(default=0, description="Rules that errored")
    compliance_score: float = Field(default=0.0, ge=0, le=100, description="Overall score 0-100")
    is_compliant: bool = Field(default=False, description="Whether overall compliant")
    has_blocking_failures: bool = Field(default=False, description="Whether any blocking rule failed")
    critical_failures: List[RuleResult] = Field(
        default_factory=list, description="Critical-severity failures"
    )
    results: List[RuleResult] = Field(default_factory=list, description="All rule results")
    results_by_category: Dict[str, List[RuleResult]] = Field(
        default_factory=dict, description="Results grouped by category"
    )
    evaluated_at: datetime = Field(default_factory=_utcnow, description="Evaluation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SimplifiedDDCheck(BaseModel):
    """Result of simplified due diligence eligibility check."""

    is_eligible: bool = Field(default=False, description="Whether simplified DD is eligible")
    country_benchmarks: Dict[str, str] = Field(
        default_factory=dict, description="Country benchmark map"
    )
    all_low_risk: bool = Field(default=False, description="Whether all countries are LOW risk")
    reasons: List[str] = Field(default_factory=list, description="Eligibility reasons")
    disqualifying_rules: List[str] = Field(
        default_factory=list, description="Disqualifying rule IDs"
    )
    article_reference: str = Field(default="Article 13", description="EUDR article reference")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class PenaltyExposure(BaseModel):
    """Estimated penalty exposure from compliance gaps."""

    total_gaps: int = Field(default=0, description="Total compliance gaps")
    critical_gaps: int = Field(default=0, description="Critical-severity gaps")
    high_gaps: int = Field(default=0, description="High-severity gaps")
    medium_gaps: int = Field(default=0, description="Medium-severity gaps")
    low_gaps: int = Field(default=0, description="Low-severity gaps")
    estimated_fine_min_eur: float = Field(default=0.0, description="Minimum fine estimate EUR")
    estimated_fine_max_eur: float = Field(default=0.0, description="Maximum fine estimate EUR")
    market_ban_risk: bool = Field(default=False, description="Whether market ban is possible")
    confiscation_risk: bool = Field(default=False, description="Whether confiscation is possible")
    penalty_article_reference: str = Field(default="Article 25", description="Penalty article")
    risk_narrative: str = Field(default="", description="Risk exposure narrative")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class RemediationAction(BaseModel):
    """A single remediation action item."""

    action_id: str = Field(default_factory=_new_uuid, description="Action identifier")
    rule_id: str = Field(..., description="Related rule ID")
    description: str = Field(..., description="Action description")
    priority: RemediationPriority = Field(..., description="Action priority")
    category: RuleCategory = Field(..., description="Related category")
    estimated_effort_days: int = Field(default=1, description="Estimated effort in days")
    responsible_role: str = Field(default="Compliance Officer", description="Responsible role")
    deadline_days: int = Field(default=30, description="Target completion days")


class RemediationPlan(BaseModel):
    """Complete remediation plan for compliance gaps."""

    plan_id: str = Field(default_factory=_new_uuid, description="Plan identifier")
    total_actions: int = Field(default=0, description="Total remediation actions")
    immediate_actions: int = Field(default=0, description="Immediate priority actions")
    short_term_actions: int = Field(default=0, description="Short-term actions")
    medium_term_actions: int = Field(default=0, description="Medium-term actions")
    long_term_actions: int = Field(default=0, description="Long-term actions")
    estimated_total_effort_days: int = Field(default=0, description="Total estimated effort")
    actions: List[RemediationAction] = Field(
        default_factory=list, description="Remediation actions"
    )
    actions_by_category: Dict[str, List[RemediationAction]] = Field(
        default_factory=dict, description="Actions grouped by category"
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Plan generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ComplianceAuditEntry(BaseModel):
    """A single audit trail entry for compliance evaluation."""

    entry_id: str = Field(default_factory=_new_uuid, description="Audit entry identifier")
    entity_id: str = Field(..., description="Entity being audited")
    entity_type: str = Field(default="", description="Entity type (DDS, supplier, etc.)")
    action: str = Field(..., description="Audit action performed")
    result: str = Field(default="", description="Action result")
    rule_id: Optional[str] = Field(None, description="Related rule ID")
    compliance_score: Optional[float] = Field(None, description="Score at time of audit")
    performed_by: str = Field(default="SYSTEM", description="Performer")
    performed_at: datetime = Field(default_factory=_utcnow, description="Action timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Rule Definitions
# ---------------------------------------------------------------------------

# All 45 rules defined as static configuration
COMPLIANCE_RULES: List[Dict[str, Any]] = [
    # GEOLOCATION (8 rules)
    {"rule_id": "GEO-001", "name": "Coordinate Precision", "category": "GEOLOCATION",
     "severity": "HIGH", "article_reference": "Article 9(1)(d)",
     "description": "Coordinates must have at least 6 decimal places precision",
     "is_blocking": True},
    {"rule_id": "GEO-002", "name": "Polygon Validity", "category": "GEOLOCATION",
     "severity": "HIGH", "article_reference": "Article 9(1)(d)",
     "description": "Polygons must be closed, non-self-intersecting, and have >= 4 vertices",
     "is_blocking": True},
    {"rule_id": "GEO-003", "name": "Area Calculation", "category": "GEOLOCATION",
     "severity": "MEDIUM", "article_reference": "Article 9(1)(d)",
     "description": "Plot area must be calculated and recorded in hectares",
     "is_blocking": False},
    {"rule_id": "GEO-004", "name": "Country Match", "category": "GEOLOCATION",
     "severity": "HIGH", "article_reference": "Article 9(1)(c)",
     "description": "Geolocation must be within declared country of production",
     "is_blocking": True},
    {"rule_id": "GEO-005", "name": "Overlap Detection", "category": "GEOLOCATION",
     "severity": "MEDIUM", "article_reference": "Article 9(1)(d)",
     "description": "Production plots must not overlap",
     "is_blocking": False},
    {"rule_id": "GEO-006", "name": "WGS84 Datum", "category": "GEOLOCATION",
     "severity": "CRITICAL", "article_reference": "Article 9(1)(d)",
     "description": "All coordinates must use WGS84 geodetic datum",
     "is_blocking": True},
    {"rule_id": "GEO-007", "name": "Batch Coordinate Validation", "category": "GEOLOCATION",
     "severity": "HIGH", "article_reference": "Article 9(1)(d)",
     "description": "All coordinates in batch must be individually valid",
     "is_blocking": True},
    {"rule_id": "GEO-008", "name": "Plot Size Rule", "category": "GEOLOCATION",
     "severity": "HIGH", "article_reference": "Article 9(1)(d)",
     "description": "Plots >= 4ha must provide polygon; < 4ha may use point",
     "is_blocking": True},

    # COMMODITY (6 rules)
    {"rule_id": "COM-001", "name": "CN Code Validity", "category": "COMMODITY",
     "severity": "CRITICAL", "article_reference": "Article 2(1)",
     "description": "CN codes must be valid 8-digit format",
     "is_blocking": True},
    {"rule_id": "COM-002", "name": "Annex I Coverage", "category": "COMMODITY",
     "severity": "CRITICAL", "article_reference": "Annex I",
     "description": "Products must be identified in EUDR Annex I",
     "is_blocking": True},
    {"rule_id": "COM-003", "name": "Derived Product Mapping", "category": "COMMODITY",
     "severity": "MEDIUM", "article_reference": "Article 2(4)",
     "description": "Derived products must be traced to base commodity",
     "is_blocking": False},
    {"rule_id": "COM-004", "name": "Multi-Commodity Declaration", "category": "COMMODITY",
     "severity": "MEDIUM", "article_reference": "Article 4(2)",
     "description": "Products containing multiple commodities must declare each",
     "is_blocking": False},
    {"rule_id": "COM-005", "name": "Product Description", "category": "COMMODITY",
     "severity": "LOW", "article_reference": "Annex II(2)",
     "description": "Product description must be meaningful and accurate",
     "is_blocking": False},
    {"rule_id": "COM-006", "name": "Quantity Declaration", "category": "COMMODITY",
     "severity": "HIGH", "article_reference": "Annex II(3)",
     "description": "Net mass and supplementary units must be declared",
     "is_blocking": True},

    # SUPPLIER (7 rules)
    {"rule_id": "SUP-001", "name": "Supplier Profile Completeness", "category": "SUPPLIER",
     "severity": "HIGH", "article_reference": "Annex II(6)",
     "description": "Supplier profile must have minimum required fields",
     "is_blocking": True},
    {"rule_id": "SUP-002", "name": "EORI Number", "category": "SUPPLIER",
     "severity": "MEDIUM", "article_reference": "Article 4(1)",
     "description": "EU operators must have valid EORI number",
     "is_blocking": False},
    {"rule_id": "SUP-003", "name": "Tier Depth Visibility", "category": "SUPPLIER",
     "severity": "MEDIUM", "article_reference": "Article 10(2)(e)",
     "description": "Supply chain tier depth should be documented",
     "is_blocking": False},
    {"rule_id": "SUP-004", "name": "Certification Status", "category": "SUPPLIER",
     "severity": "LOW", "article_reference": "Article 10(2)(d)",
     "description": "Supplier certifications should be valid and current",
     "is_blocking": False},
    {"rule_id": "SUP-005", "name": "Engagement History", "category": "SUPPLIER",
     "severity": "LOW", "article_reference": "Article 10(2)(e)",
     "description": "Supplier engagement history should be documented",
     "is_blocking": False},
    {"rule_id": "SUP-006", "name": "Data Quality", "category": "SUPPLIER",
     "severity": "HIGH", "article_reference": "Article 10(1)",
     "description": "Supplier data must meet minimum quality threshold",
     "is_blocking": True},
    {"rule_id": "SUP-007", "name": "DD Status", "category": "SUPPLIER",
     "severity": "CRITICAL", "article_reference": "Article 4(1)",
     "description": "Due diligence must be completed for all direct suppliers",
     "is_blocking": True},

    # RISK (6 rules)
    {"rule_id": "RSK-001", "name": "Composite Risk Validity", "category": "RISK",
     "severity": "HIGH", "article_reference": "Article 10(1)",
     "description": "Composite risk score must be calculated from all dimensions",
     "is_blocking": True},
    {"rule_id": "RSK-002", "name": "Country Benchmark Check", "category": "RISK",
     "severity": "HIGH", "article_reference": "Article 29",
     "description": "Country risk benchmark must be checked per Article 29",
     "is_blocking": True},
    {"rule_id": "RSK-003", "name": "Simplified DD Prerequisites", "category": "RISK",
     "severity": "CRITICAL", "article_reference": "Article 13",
     "description": "Simplified DD only for all-LOW-risk-country sourcing",
     "is_blocking": True},
    {"rule_id": "RSK-004", "name": "Risk Documentation", "category": "RISK",
     "severity": "HIGH", "article_reference": "Article 10(1)",
     "description": "Risk assessment must be documented with methodology",
     "is_blocking": True},
    {"rule_id": "RSK-005", "name": "Risk Threshold Action", "category": "RISK",
     "severity": "HIGH", "article_reference": "Article 11",
     "description": "High/critical risk must trigger enhanced due diligence",
     "is_blocking": True},
    {"rule_id": "RSK-006", "name": "Risk Trend Monitoring", "category": "RISK",
     "severity": "LOW", "article_reference": "Article 10(2)(h)",
     "description": "Risk trends should be monitored over time",
     "is_blocking": False},

    # DDS (8 rules)
    {"rule_id": "DDS-001", "name": "Annex II Completeness", "category": "DDS",
     "severity": "CRITICAL", "article_reference": "Annex II",
     "description": "All 8 Annex II required fields must be present",
     "is_blocking": True},
    {"rule_id": "DDS-002", "name": "Operator Declaration", "category": "DDS",
     "severity": "CRITICAL", "article_reference": "Article 4(2)",
     "description": "Operator/trader declaration must be signed",
     "is_blocking": True},
    {"rule_id": "DDS-003", "name": "Geolocation Format", "category": "DDS",
     "severity": "HIGH", "article_reference": "Article 9(1)(d)",
     "description": "Geolocation must be in correct format per plot size",
     "is_blocking": True},
    {"rule_id": "DDS-004", "name": "Risk Summary", "category": "DDS",
     "severity": "HIGH", "article_reference": "Annex II(7)",
     "description": "Risk assessment summary must be included in DDS",
     "is_blocking": True},
    {"rule_id": "DDS-005", "name": "Mitigation Measures", "category": "DDS",
     "severity": "HIGH", "article_reference": "Annex II(8)",
     "description": "Risk mitigation measures must be documented for STANDARD/HIGH risk",
     "is_blocking": True},
    {"rule_id": "DDS-006", "name": "Reference Number", "category": "DDS",
     "severity": "MEDIUM", "article_reference": "Article 4(2)",
     "description": "DDS must have a unique reference number",
     "is_blocking": False},
    {"rule_id": "DDS-007", "name": "Submission Deadline", "category": "DDS",
     "severity": "HIGH", "article_reference": "Article 4(1)",
     "description": "DDS must be submitted before market placement",
     "is_blocking": True},
    {"rule_id": "DDS-008", "name": "Amendment Tracking", "category": "DDS",
     "severity": "MEDIUM", "article_reference": "Article 4(8)",
     "description": "DDS amendments must be tracked with version history",
     "is_blocking": False},

    # DOCUMENTATION (5 rules)
    {"rule_id": "DOC-001", "name": "Document Authenticity", "category": "DOCUMENTATION",
     "severity": "HIGH", "article_reference": "Article 10(2)(c)",
     "description": "Supporting documents must be authentic and verifiable",
     "is_blocking": True},
    {"rule_id": "DOC-002", "name": "Document Validity", "category": "DOCUMENTATION",
     "severity": "HIGH", "article_reference": "Article 10(2)(c)",
     "description": "Documents must not be expired at time of submission",
     "is_blocking": True},
    {"rule_id": "DOC-003", "name": "Chain of Custody", "category": "DOCUMENTATION",
     "severity": "HIGH", "article_reference": "Article 10(2)(e)",
     "description": "Documents must establish chain of custody for commodities",
     "is_blocking": True},
    {"rule_id": "DOC-004", "name": "Retention Period", "category": "DOCUMENTATION",
     "severity": "MEDIUM", "article_reference": "Article 4(6)",
     "description": "DDS and supporting documents must be retained for 5 years",
     "is_blocking": False},
    {"rule_id": "DOC-005", "name": "Version Control", "category": "DOCUMENTATION",
     "severity": "LOW", "article_reference": "Article 4(8)",
     "description": "Document versions must be tracked and managed",
     "is_blocking": False},

    # CUTOFF (5 rules)
    {"rule_id": "CUT-001", "name": "Cutoff Date Verification", "category": "CUTOFF",
     "severity": "CRITICAL", "article_reference": "Article 2(8), 3(a)",
     "description": "Production must be verified deforestation-free after 31/12/2020",
     "is_blocking": True},
    {"rule_id": "CUT-002", "name": "Temporal Evidence", "category": "CUTOFF",
     "severity": "HIGH", "article_reference": "Article 10(2)(a)",
     "description": "Sufficient temporal evidence must support cutoff compliance",
     "is_blocking": True},
    {"rule_id": "CUT-003", "name": "Land Use History", "category": "CUTOFF",
     "severity": "HIGH", "article_reference": "Article 10(2)(b)",
     "description": "Land use history must demonstrate no post-cutoff deforestation",
     "is_blocking": True},
    {"rule_id": "CUT-004", "name": "Cutoff Declaration", "category": "CUTOFF",
     "severity": "HIGH", "article_reference": "Article 3(a)",
     "description": "Formal cutoff compliance declaration must be generated",
     "is_blocking": True},
    {"rule_id": "CUT-005", "name": "Exemption Validity", "category": "CUTOFF",
     "severity": "MEDIUM", "article_reference": "Article 38",
     "description": "Any claimed exemptions must be properly documented",
     "is_blocking": False},
]

# Severity weights for compliance scoring
SEVERITY_WEIGHTS: Dict[str, float] = {
    "CRITICAL": 4.0,
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
}

# Penalty range estimates per gap severity (EUR)
PENALTY_RANGES: Dict[str, Tuple[float, float]] = {
    "CRITICAL": (50000.0, 500000.0),
    "HIGH": (10000.0, 100000.0),
    "MEDIUM": (1000.0, 20000.0),
    "LOW": (0.0, 5000.0),
}

# Document retention period (years)
DOCUMENT_RETENTION_YEARS = 5


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PolicyComplianceEngine:
    """
    EUDR Compliance Rule Enforcement Engine.

    Evaluates operator and trader compliance against 45 rules across 7
    regulatory categories. Each rule maps to specific EUDR articles and
    has a defined severity level.

    All evaluations are deterministic boolean logic with complete
    provenance tracking. No LLM involvement in any compliance determination.

    Attributes:
        config: Optional engine configuration
        _rules: Loaded compliance rules
        _audit_trail: Compliance audit trail entries

    Example:
        >>> engine = PolicyComplianceEngine()
        >>> result = engine.evaluate_all_rules(dds_data)
        >>> assert result.compliance_score >= 0
        >>> if not result.is_compliant:
        ...     plan = engine.generate_remediation_plan(result.results)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PolicyComplianceEngine.

        Args:
            config: Optional configuration dictionary with keys:
                - strict_mode: Fail on any HIGH+ severity (default: True)
                - retention_years: Document retention period (default: 5)
                - min_compliance_score: Minimum passing score (default: 80.0)
        """
        self.config = config or {}
        self._strict_mode: bool = self.config.get("strict_mode", True)
        self._retention_years: int = self.config.get("retention_years", DOCUMENT_RETENTION_YEARS)
        self._min_score: float = self.config.get("min_compliance_score", 80.0)
        self._rules: List[ComplianceRule] = self._load_rules()
        self._audit_trail: List[ComplianceAuditEntry] = []
        self._evaluation_count: int = 0
        logger.info(
            "PolicyComplianceEngine initialized (version=%s, rules=%d)",
            _MODULE_VERSION, len(self._rules),
        )

    # -------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------

    def evaluate_all_rules(self, data: Dict[str, Any]) -> ComplianceResult:
        """Evaluate all 45 compliance rules against provided data.

        Args:
            data: Complete compliance data dictionary with keys organized
                by category: geolocation, commodity, supplier, risk, dds,
                documentation, cutoff.

        Returns:
            ComplianceResult with all rule evaluations and overall score.
        """
        logger.info("Evaluating all %d compliance rules", len(self._rules))
        results: List[RuleResult] = []

        for rule in self._rules:
            rule_result = self.evaluate_rule(rule.rule_id, data)
            results.append(rule_result)

        # Calculate aggregates
        passed = sum(1 for r in results if r.status == RuleStatus.PASS)
        failed = sum(1 for r in results if r.status == RuleStatus.FAIL)
        warnings = sum(1 for r in results if r.status == RuleStatus.WARNING)
        na = sum(1 for r in results if r.status == RuleStatus.NOT_APPLICABLE)
        errors = sum(1 for r in results if r.status == RuleStatus.ERROR)

        score = self.get_compliance_score(results)

        critical_failures = [
            r for r in results
            if r.status == RuleStatus.FAIL and r.severity == RuleSeverity.CRITICAL
        ]

        has_blocking = any(
            r.status == RuleStatus.FAIL
            for r in results
            for rule in self._rules
            if rule.rule_id == r.rule_id and rule.is_blocking
        )

        is_compliant = failed == 0 or (not self._strict_mode and score >= self._min_score)

        # Group by category
        results_by_category: Dict[str, List[RuleResult]] = {}
        for r in results:
            cat = r.category.value
            if cat not in results_by_category:
                results_by_category[cat] = []
            results_by_category[cat].append(r)

        result = ComplianceResult(
            total_rules=len(results),
            passed_rules=passed,
            failed_rules=failed,
            warning_rules=warnings,
            not_applicable_rules=na,
            error_rules=errors,
            compliance_score=score,
            is_compliant=is_compliant,
            has_blocking_failures=has_blocking,
            critical_failures=critical_failures,
            results=results,
            results_by_category=results_by_category,
        )
        result.provenance_hash = _compute_hash(result)

        # Record audit trail
        self._record_audit("FULL_EVALUATION", "compliance_evaluation", score, None)
        self._evaluation_count += 1

        logger.info(
            "Compliance evaluation complete: score=%.1f, passed=%d, failed=%d, "
            "compliant=%s",
            score, passed, failed, is_compliant,
        )
        return result

    def evaluate_rule(self, rule_id: str, data: Dict[str, Any]) -> RuleResult:
        """Evaluate a single compliance rule.

        Dispatches to the appropriate rule-specific evaluation method
        based on the rule ID prefix.

        Args:
            rule_id: Rule identifier (e.g., 'GEO-001', 'COM-002').
            data: Compliance data dictionary.

        Returns:
            RuleResult with evaluation status.
        """
        rule_def = self._get_rule(rule_id)
        if not rule_def:
            return RuleResult(
                rule_id=rule_id,
                category=RuleCategory.DDS,
                severity=RuleSeverity.LOW,
                status=RuleStatus.ERROR,
                message=f"Unknown rule ID: {rule_id}",
            )

        try:
            # Dispatch by category prefix
            prefix = rule_id.split("-")[0]
            evaluator = self._get_evaluator(prefix)
            result = evaluator(rule_id, rule_def, data)
            return result
        except Exception as exc:
            logger.warning("Rule %s evaluation failed: %s", rule_id, str(exc))
            return RuleResult(
                rule_id=rule_id,
                rule_name=rule_def.name,
                category=rule_def.category,
                severity=rule_def.severity,
                status=RuleStatus.ERROR,
                message=f"Evaluation error: {str(exc)}",
                article_reference=rule_def.article_reference,
            )

    def get_compliance_score(self, results: List[RuleResult]) -> float:
        """Calculate overall compliance score from rule results.

        Uses severity-weighted scoring where critical rules have more
        impact on the overall score.

        Args:
            results: List of rule evaluation results.

        Returns:
            Compliance score from 0 to 100.
        """
        if not results:
            return 0.0

        total_weight = 0.0
        achieved_weight = 0.0

        for r in results:
            if r.status == RuleStatus.NOT_APPLICABLE:
                continue
            weight = SEVERITY_WEIGHTS.get(r.severity.value, 1.0)
            total_weight += weight
            if r.status == RuleStatus.PASS:
                achieved_weight += weight
            elif r.status == RuleStatus.WARNING:
                achieved_weight += weight * 0.5

        if total_weight == 0:
            return 100.0

        return round((achieved_weight / total_weight) * 100.0, 2)

    def get_rules_by_category(self, category: str) -> List[ComplianceRule]:
        """Get all compliance rules for a specific category.

        Args:
            category: Rule category name (e.g., 'GEOLOCATION', 'DDS').

        Returns:
            List of ComplianceRule objects for the category.
        """
        cat_upper = category.upper()
        return [r for r in self._rules if r.category.value == cat_upper]

    def check_simplified_dd_eligibility(
        self, data: Dict[str, Any]
    ) -> SimplifiedDDCheck:
        """Check eligibility for simplified due diligence per Article 13.

        Args:
            data: Data dictionary with keys:
                - countries (list of ISO codes)
                - country_benchmarks (dict: code -> benchmark)
                - risk_score (float, optional)

        Returns:
            SimplifiedDDCheck with eligibility status.
        """
        countries = data.get("countries", [])
        benchmarks = data.get("country_benchmarks", {})
        reasons: List[str] = []
        disqualifying: List[str] = []

        all_low = True
        for country in countries:
            benchmark = benchmarks.get(country.upper(), "STANDARD")
            if benchmark != "LOW":
                all_low = False
                disqualifying.append(
                    f"Country {country} is {benchmark} risk (must be LOW)"
                )

        if all_low and countries:
            reasons.append("All source countries are benchmarked as LOW risk")
        elif not countries:
            all_low = False
            disqualifying.append("No source countries declared")

        risk_score = data.get("risk_score")
        if risk_score and risk_score > 25.0:
            all_low = False
            disqualifying.append(
                f"Risk score {risk_score} exceeds simplified DD threshold (25)"
            )

        is_eligible = all_low and not disqualifying

        result = SimplifiedDDCheck(
            is_eligible=is_eligible,
            country_benchmarks=benchmarks,
            all_low_risk=all_low,
            reasons=reasons,
            disqualifying_rules=disqualifying,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def estimate_penalty_exposure(
        self, gaps: List[RuleResult]
    ) -> PenaltyExposure:
        """Estimate penalty exposure from compliance gaps.

        Calculates estimated fine ranges based on the severity distribution
        of failed compliance rules, per EUDR Article 25 penalty provisions.

        Args:
            gaps: List of RuleResult instances that represent failures.

        Returns:
            PenaltyExposure with estimated fine ranges and risk narrative.
        """
        failures = [g for g in gaps if g.status == RuleStatus.FAIL]

        critical = sum(1 for f in failures if f.severity == RuleSeverity.CRITICAL)
        high = sum(1 for f in failures if f.severity == RuleSeverity.HIGH)
        medium = sum(1 for f in failures if f.severity == RuleSeverity.MEDIUM)
        low = sum(1 for f in failures if f.severity == RuleSeverity.LOW)

        total_min = 0.0
        total_max = 0.0
        for sev_name, count in [("CRITICAL", critical), ("HIGH", high),
                                 ("MEDIUM", medium), ("LOW", low)]:
            min_pen, max_pen = PENALTY_RANGES[sev_name]
            total_min += min_pen * count
            total_max += max_pen * count

        market_ban = critical > 0
        confiscation = critical >= 2 or (critical >= 1 and high >= 2)

        # Build narrative
        parts: List[str] = []
        if critical > 0:
            parts.append(f"{critical} critical violation(s) - potential market ban")
        if high > 0:
            parts.append(f"{high} high-severity gap(s)")
        if medium > 0:
            parts.append(f"{medium} medium-severity gap(s)")
        if low > 0:
            parts.append(f"{low} low-severity issue(s)")

        narrative = (
            f"Penalty exposure assessment: {len(failures)} total compliance gaps. "
            + "; ".join(parts) + ". "
            f"Estimated fine range: EUR {total_min:,.0f} - {total_max:,.0f}. "
            f"Market ban risk: {'YES' if market_ban else 'NO'}. "
            f"Confiscation risk: {'YES' if confiscation else 'NO'}."
        )

        result = PenaltyExposure(
            total_gaps=len(failures),
            critical_gaps=critical,
            high_gaps=high,
            medium_gaps=medium,
            low_gaps=low,
            estimated_fine_min_eur=total_min,
            estimated_fine_max_eur=total_max,
            market_ban_risk=market_ban,
            confiscation_risk=confiscation,
            risk_narrative=narrative,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def generate_remediation_plan(
        self, results: List[RuleResult]
    ) -> RemediationPlan:
        """Generate a remediation plan from compliance evaluation results.

        Creates actionable remediation items for each failed rule,
        prioritized by severity and regulatory importance.

        Args:
            results: List of RuleResult from compliance evaluation.

        Returns:
            RemediationPlan with prioritized actions.
        """
        failures = [r for r in results if r.status == RuleStatus.FAIL]
        actions: List[RemediationAction] = []

        # Priority and effort mapping by severity
        priority_map = {
            RuleSeverity.CRITICAL: (RemediationPriority.IMMEDIATE, 2, 7),
            RuleSeverity.HIGH: (RemediationPriority.SHORT_TERM, 5, 14),
            RuleSeverity.MEDIUM: (RemediationPriority.MEDIUM_TERM, 10, 30),
            RuleSeverity.LOW: (RemediationPriority.LONG_TERM, 5, 60),
        }

        for failure in failures:
            priority, effort, deadline = priority_map.get(
                failure.severity, (RemediationPriority.MEDIUM_TERM, 10, 30)
            )

            description = failure.remediation_hint or (
                f"Resolve compliance gap: {failure.rule_name} - {failure.message}"
            )

            role = "Compliance Officer"
            if failure.category == RuleCategory.GEOLOCATION:
                role = "GIS/Data Team"
            elif failure.category == RuleCategory.SUPPLIER:
                role = "Procurement/Supply Chain Team"
            elif failure.category == RuleCategory.DOCUMENTATION:
                role = "Document Control"

            action = RemediationAction(
                rule_id=failure.rule_id,
                description=description,
                priority=priority,
                category=failure.category,
                estimated_effort_days=effort,
                responsible_role=role,
                deadline_days=deadline,
            )
            actions.append(action)

        # Sort by priority
        priority_order = {
            RemediationPriority.IMMEDIATE: 0,
            RemediationPriority.SHORT_TERM: 1,
            RemediationPriority.MEDIUM_TERM: 2,
            RemediationPriority.LONG_TERM: 3,
        }
        actions.sort(key=lambda a: priority_order.get(a.priority, 99))

        # Count by priority
        immediate = sum(1 for a in actions if a.priority == RemediationPriority.IMMEDIATE)
        short_term = sum(1 for a in actions if a.priority == RemediationPriority.SHORT_TERM)
        medium_term = sum(1 for a in actions if a.priority == RemediationPriority.MEDIUM_TERM)
        long_term = sum(1 for a in actions if a.priority == RemediationPriority.LONG_TERM)
        total_effort = sum(a.estimated_effort_days for a in actions)

        # Group by category
        by_category: Dict[str, List[RemediationAction]] = {}
        for a in actions:
            cat = a.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(a)

        plan = RemediationPlan(
            total_actions=len(actions),
            immediate_actions=immediate,
            short_term_actions=short_term,
            medium_term_actions=medium_term,
            long_term_actions=long_term,
            estimated_total_effort_days=total_effort,
            actions=actions,
            actions_by_category=by_category,
        )
        plan.provenance_hash = _compute_hash(plan)
        return plan

    def get_audit_trail(self, entity_id: str) -> List[ComplianceAuditEntry]:
        """Get audit trail entries for a specific entity.

        Args:
            entity_id: Entity identifier to filter by.

        Returns:
            List of ComplianceAuditEntry for the entity.
        """
        return [
            entry for entry in self._audit_trail
            if entry.entity_id == entity_id
        ]

    # -------------------------------------------------------------------
    # Private: Rule Loading
    # -------------------------------------------------------------------

    def _load_rules(self) -> List[ComplianceRule]:
        """Load all compliance rules from the static definition."""
        rules: List[ComplianceRule] = []
        for rule_def in COMPLIANCE_RULES:
            rules.append(ComplianceRule(
                rule_id=rule_def["rule_id"],
                name=rule_def["name"],
                description=rule_def.get("description", ""),
                category=RuleCategory(rule_def["category"]),
                severity=RuleSeverity(rule_def["severity"]),
                article_reference=rule_def.get("article_reference", ""),
                is_blocking=rule_def.get("is_blocking", False),
            ))
        return rules

    def _get_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        """Get a rule by ID."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    # -------------------------------------------------------------------
    # Private: Evaluation Dispatchers
    # -------------------------------------------------------------------

    def _get_evaluator(self, prefix: str) -> Callable:
        """Get the evaluation function for a rule prefix.

        Args:
            prefix: Rule ID prefix (GEO, COM, SUP, RSK, DDS, DOC, CUT).

        Returns:
            Evaluation function for the category.
        """
        evaluators = {
            "GEO": self._evaluate_geolocation_rule,
            "COM": self._evaluate_commodity_rule,
            "SUP": self._evaluate_supplier_rule,
            "RSK": self._evaluate_risk_rule,
            "DDS": self._evaluate_dds_rule,
            "DOC": self._evaluate_documentation_rule,
            "CUT": self._evaluate_cutoff_rule,
        }
        return evaluators.get(prefix, self._evaluate_default_rule)

    def _evaluate_geolocation_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate geolocation category rules."""
        geo = data.get("geolocation", {})

        if rule_id == "GEO-001":
            precision = geo.get("precision_decimals", 0)
            is_ok = precision >= 6
            return self._make_result(
                rule, RuleStatus.PASS if is_ok else RuleStatus.FAIL,
                f"Coordinate precision: {precision} decimals (required: 6)" if not is_ok
                else "Coordinate precision meets requirement (>= 6 decimals)",
                "Ensure all coordinates have at least 6 decimal places" if not is_ok else None,
            )

        if rule_id == "GEO-002":
            polygons_valid = geo.get("polygons_valid", True)
            return self._make_result(
                rule, RuleStatus.PASS if polygons_valid else RuleStatus.FAIL,
                "All polygons are valid" if polygons_valid
                else "One or more polygons have invalid topology",
                "Fix polygon topology: ensure closed ring, no self-intersection" if not polygons_valid else None,
            )

        if rule_id == "GEO-003":
            has_area = geo.get("area_calculated", False)
            return self._make_result(
                rule, RuleStatus.PASS if has_area else RuleStatus.WARNING,
                "Area calculated for all plots" if has_area
                else "Area not calculated for all plots",
                "Calculate area in hectares for all production plots" if not has_area else None,
            )

        if rule_id == "GEO-004":
            country_match = geo.get("country_match", True)
            return self._make_result(
                rule, RuleStatus.PASS if country_match else RuleStatus.FAIL,
                "Geolocation matches declared country" if country_match
                else "Geolocation does not match declared country of production",
                "Verify country of production matches plot coordinates" if not country_match else None,
            )

        if rule_id == "GEO-005":
            no_overlaps = geo.get("no_overlaps", True)
            return self._make_result(
                rule, RuleStatus.PASS if no_overlaps else RuleStatus.WARNING,
                "No plot overlaps detected" if no_overlaps
                else "Overlapping plots detected",
                "Resolve overlapping plot boundaries" if not no_overlaps else None,
            )

        if rule_id == "GEO-006":
            datum = geo.get("datum", "")
            is_wgs84 = datum.upper() == "WGS84"
            return self._make_result(
                rule, RuleStatus.PASS if is_wgs84 else RuleStatus.FAIL,
                "WGS84 datum confirmed" if is_wgs84
                else f"Non-WGS84 datum detected: {datum}",
                "Convert all coordinates to WGS84 datum" if not is_wgs84 else None,
            )

        if rule_id == "GEO-007":
            all_valid = geo.get("all_coordinates_valid", True)
            return self._make_result(
                rule, RuleStatus.PASS if all_valid else RuleStatus.FAIL,
                "All coordinates validated" if all_valid
                else "Some coordinates failed validation",
                "Review and fix invalid coordinates" if not all_valid else None,
            )

        if rule_id == "GEO-008":
            plot_size_compliant = geo.get("plot_size_rule_compliant", True)
            return self._make_result(
                rule, RuleStatus.PASS if plot_size_compliant else RuleStatus.FAIL,
                "Plot size rule met (polygon for >=4ha)" if plot_size_compliant
                else "Plot size rule violation: plots >=4ha need polygon format",
                "Provide polygon boundaries for plots >= 4 hectares" if not plot_size_compliant else None,
            )

        return self._make_na_result(rule)

    def _evaluate_commodity_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate commodity category rules."""
        com = data.get("commodity", {})

        if rule_id == "COM-001":
            cn_valid = com.get("cn_codes_valid", False)
            return self._make_result(
                rule, RuleStatus.PASS if cn_valid else RuleStatus.FAIL,
                "All CN codes are valid 8-digit format" if cn_valid
                else "Invalid CN code format detected",
                "Ensure all CN codes are exactly 8 digits" if not cn_valid else None,
            )

        if rule_id == "COM-002":
            annex_covered = com.get("annex_i_covered", False)
            return self._make_result(
                rule, RuleStatus.PASS if annex_covered else RuleStatus.FAIL,
                "All products identified in Annex I" if annex_covered
                else "Product(s) not found in EUDR Annex I",
                "Verify all products are covered by EUDR Annex I" if not annex_covered else None,
            )

        if rule_id == "COM-003":
            derived_mapped = com.get("derived_products_mapped", True)
            return self._make_result(
                rule, RuleStatus.PASS if derived_mapped else RuleStatus.WARNING,
                "Derived products mapped to base commodity" if derived_mapped
                else "Some derived products not mapped to base commodity",
                "Map all derived products to their base EUDR commodity" if not derived_mapped else None,
            )

        if rule_id == "COM-004":
            multi_declared = com.get("multi_commodity_declared", True)
            return self._make_result(
                rule, RuleStatus.PASS if multi_declared else RuleStatus.WARNING,
                "Multi-commodity products properly declared" if multi_declared
                else "Multi-commodity products not fully declared",
                "Declare each commodity component separately" if not multi_declared else None,
            )

        if rule_id == "COM-005":
            desc_quality = com.get("descriptions_adequate", True)
            return self._make_result(
                rule, RuleStatus.PASS if desc_quality else RuleStatus.WARNING,
                "Product descriptions are adequate" if desc_quality
                else "Some product descriptions are too brief",
                "Provide meaningful product descriptions" if not desc_quality else None,
            )

        if rule_id == "COM-006":
            qty_declared = com.get("quantities_declared", False)
            return self._make_result(
                rule, RuleStatus.PASS if qty_declared else RuleStatus.FAIL,
                "Quantities properly declared" if qty_declared
                else "Net mass / supplementary units not declared",
                "Declare net mass in kg and supplementary units" if not qty_declared else None,
            )

        return self._make_na_result(rule)

    def _evaluate_supplier_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate supplier category rules."""
        sup = data.get("supplier", {})

        if rule_id == "SUP-001":
            profile_complete = sup.get("profile_complete", False)
            return self._make_result(
                rule, RuleStatus.PASS if profile_complete else RuleStatus.FAIL,
                "Supplier profile is complete" if profile_complete
                else "Supplier profile missing required fields",
                "Complete all required supplier profile fields" if not profile_complete else None,
            )

        if rule_id == "SUP-002":
            has_eori = sup.get("has_eori", True)
            is_eu = sup.get("is_eu_operator", False)
            if not is_eu:
                return self._make_na_result(rule)
            return self._make_result(
                rule, RuleStatus.PASS if has_eori else RuleStatus.WARNING,
                "EORI number provided" if has_eori
                else "EU operator missing EORI number",
                "Obtain and register EORI number" if not has_eori else None,
            )

        if rule_id == "SUP-003":
            tier_documented = sup.get("tier_documented", True)
            return self._make_result(
                rule, RuleStatus.PASS if tier_documented else RuleStatus.WARNING,
                "Supply chain tiers documented" if tier_documented
                else "Supply chain tier depth not documented",
                "Document supply chain tier depth for all suppliers" if not tier_documented else None,
            )

        if rule_id == "SUP-004":
            certs_valid = sup.get("certifications_valid", True)
            return self._make_result(
                rule, RuleStatus.PASS if certs_valid else RuleStatus.WARNING,
                "Certifications are valid and current" if certs_valid
                else "Some certifications expired or invalid",
                "Renew expired certifications" if not certs_valid else None,
            )

        if rule_id == "SUP-005":
            engagement_documented = sup.get("engagement_documented", True)
            return self._make_result(
                rule, RuleStatus.PASS if engagement_documented else RuleStatus.WARNING,
                "Engagement history documented" if engagement_documented
                else "Supplier engagement history not documented",
                "Document all supplier engagement events" if not engagement_documented else None,
            )

        if rule_id == "SUP-006":
            data_quality = sup.get("data_quality_score", 0)
            is_ok = data_quality >= 70
            return self._make_result(
                rule, RuleStatus.PASS if is_ok else RuleStatus.FAIL,
                f"Data quality score: {data_quality}% (threshold: 70%)" if not is_ok
                else f"Data quality score: {data_quality}% - meets threshold",
                "Improve supplier data quality to >= 70%" if not is_ok else None,
            )

        if rule_id == "SUP-007":
            dd_complete = sup.get("dd_status", "") in ("COMPLETE", "VERIFIED")
            return self._make_result(
                rule, RuleStatus.PASS if dd_complete else RuleStatus.FAIL,
                "Due diligence completed" if dd_complete
                else f"Due diligence status: {sup.get('dd_status', 'NOT_STARTED')}",
                "Complete due diligence for all direct suppliers" if not dd_complete else None,
            )

        return self._make_na_result(rule)

    def _evaluate_risk_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate risk category rules."""
        risk = data.get("risk", {})

        if rule_id == "RSK-001":
            has_composite = risk.get("composite_score") is not None
            return self._make_result(
                rule, RuleStatus.PASS if has_composite else RuleStatus.FAIL,
                "Composite risk score calculated" if has_composite
                else "Composite risk score not calculated",
                "Calculate composite risk from all dimensions" if not has_composite else None,
            )

        if rule_id == "RSK-002":
            benchmark_checked = risk.get("benchmark_checked", False)
            return self._make_result(
                rule, RuleStatus.PASS if benchmark_checked else RuleStatus.FAIL,
                "Country benchmark checked per Article 29" if benchmark_checked
                else "Country benchmark not checked",
                "Check country risk benchmark per Article 29" if not benchmark_checked else None,
            )

        if rule_id == "RSK-003":
            is_simplified = data.get("dds", {}).get("dds_type") == "SIMPLIFIED"
            if not is_simplified:
                return self._make_na_result(rule)
            all_low = risk.get("all_countries_low_risk", False)
            return self._make_result(
                rule, RuleStatus.PASS if all_low else RuleStatus.FAIL,
                "All countries LOW risk - simplified DD eligible" if all_low
                else "Not all countries LOW risk - simplified DD not eligible",
                "Cannot use simplified DD unless all countries are LOW risk" if not all_low else None,
            )

        if rule_id == "RSK-004":
            risk_documented = risk.get("methodology_documented", False)
            return self._make_result(
                rule, RuleStatus.PASS if risk_documented else RuleStatus.FAIL,
                "Risk assessment methodology documented" if risk_documented
                else "Risk assessment methodology not documented",
                "Document risk assessment methodology and data sources" if not risk_documented else None,
            )

        if rule_id == "RSK-005":
            score = risk.get("composite_score", 0)
            enhanced_dd = risk.get("enhanced_dd_performed", False)
            if score <= 50:
                return self._make_result(rule, RuleStatus.PASS, "Risk below enhanced DD threshold")
            return self._make_result(
                rule, RuleStatus.PASS if enhanced_dd else RuleStatus.FAIL,
                "Enhanced DD performed for high risk" if enhanced_dd
                else f"Risk score {score} requires enhanced DD - not performed",
                "Perform enhanced due diligence for high/critical risk" if not enhanced_dd else None,
            )

        if rule_id == "RSK-006":
            trend_monitored = risk.get("trend_monitored", False)
            return self._make_result(
                rule, RuleStatus.PASS if trend_monitored else RuleStatus.WARNING,
                "Risk trends monitored" if trend_monitored
                else "Risk trend monitoring not established",
                "Establish risk trend monitoring across periods" if not trend_monitored else None,
            )

        return self._make_na_result(rule)

    def _evaluate_dds_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate DDS category rules."""
        dds = data.get("dds", {})

        if rule_id == "DDS-001":
            annex_complete = dds.get("annex_ii_complete", False)
            return self._make_result(
                rule, RuleStatus.PASS if annex_complete else RuleStatus.FAIL,
                "All 8 Annex II fields present" if annex_complete
                else "Annex II fields incomplete",
                "Complete all 8 required Annex II fields" if not annex_complete else None,
            )

        if rule_id == "DDS-002":
            declaration_signed = dds.get("operator_declaration_signed", False)
            return self._make_result(
                rule, RuleStatus.PASS if declaration_signed else RuleStatus.FAIL,
                "Operator declaration signed" if declaration_signed
                else "Operator declaration not signed",
                "Obtain operator/trader declaration signature" if not declaration_signed else None,
            )

        if rule_id == "DDS-003":
            geo_format_ok = dds.get("geolocation_format_correct", True)
            return self._make_result(
                rule, RuleStatus.PASS if geo_format_ok else RuleStatus.FAIL,
                "Geolocation format correct per plot size" if geo_format_ok
                else "Geolocation format does not match plot size requirements",
                "Use polygon for plots >=4ha, point for <4ha" if not geo_format_ok else None,
            )

        if rule_id == "DDS-004":
            has_risk_summary = dds.get("risk_summary_present", False)
            return self._make_result(
                rule, RuleStatus.PASS if has_risk_summary else RuleStatus.FAIL,
                "Risk summary included in DDS" if has_risk_summary
                else "Risk assessment summary missing from DDS",
                "Include risk assessment summary in DDS (Annex II, field 7)" if not has_risk_summary else None,
            )

        if rule_id == "DDS-005":
            risk_level = dds.get("risk_level", "NEGLIGIBLE")
            needs_mitigation = risk_level in ("STANDARD", "HIGH")
            has_mitigation = dds.get("mitigation_documented", False)
            if not needs_mitigation:
                return self._make_result(
                    rule, RuleStatus.PASS,
                    f"Risk level {risk_level} - mitigation not required",
                )
            return self._make_result(
                rule, RuleStatus.PASS if has_mitigation else RuleStatus.FAIL,
                "Mitigation measures documented" if has_mitigation
                else f"Risk level {risk_level} requires mitigation documentation",
                "Document risk mitigation measures (Annex II, field 8)" if not has_mitigation else None,
            )

        if rule_id == "DDS-006":
            has_ref = bool(dds.get("reference_number", ""))
            return self._make_result(
                rule, RuleStatus.PASS if has_ref else RuleStatus.WARNING,
                "DDS reference number assigned" if has_ref
                else "DDS reference number not assigned",
                "Generate unique DDS reference number" if not has_ref else None,
            )

        if rule_id == "DDS-007":
            submitted_before_placement = dds.get("submitted_before_placement", True)
            return self._make_result(
                rule, RuleStatus.PASS if submitted_before_placement else RuleStatus.FAIL,
                "DDS submitted before market placement" if submitted_before_placement
                else "DDS not submitted before market placement",
                "Submit DDS before placing product on EU market" if not submitted_before_placement else None,
            )

        if rule_id == "DDS-008":
            amendments_tracked = dds.get("amendments_tracked", True)
            return self._make_result(
                rule, RuleStatus.PASS if amendments_tracked else RuleStatus.WARNING,
                "DDS amendments tracked with version history" if amendments_tracked
                else "DDS amendment tracking not established",
                "Implement DDS version tracking for amendments" if not amendments_tracked else None,
            )

        return self._make_na_result(rule)

    def _evaluate_documentation_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate documentation category rules."""
        doc = data.get("documentation", {})

        if rule_id == "DOC-001":
            all_authentic = doc.get("all_authentic", True)
            return self._make_result(
                rule, RuleStatus.PASS if all_authentic else RuleStatus.FAIL,
                "All documents verified as authentic" if all_authentic
                else "Document authenticity concerns identified",
                "Verify authenticity of all supporting documents" if not all_authentic else None,
            )

        if rule_id == "DOC-002":
            all_valid = doc.get("all_valid_dates", True)
            return self._make_result(
                rule, RuleStatus.PASS if all_valid else RuleStatus.FAIL,
                "All documents within validity period" if all_valid
                else "Expired documents detected",
                "Replace expired documents with current versions" if not all_valid else None,
            )

        if rule_id == "DOC-003":
            chain_established = doc.get("chain_of_custody", False)
            return self._make_result(
                rule, RuleStatus.PASS if chain_established else RuleStatus.FAIL,
                "Chain of custody established" if chain_established
                else "Chain of custody not established",
                "Establish complete chain of custody documentation" if not chain_established else None,
            )

        if rule_id == "DOC-004":
            retention_policy = doc.get("retention_policy", False)
            return self._make_result(
                rule, RuleStatus.PASS if retention_policy else RuleStatus.WARNING,
                f"Document retention policy in place ({self._retention_years} years)"
                if retention_policy
                else f"No document retention policy (required: {self._retention_years} years)",
                f"Implement {self._retention_years}-year document retention policy"
                if not retention_policy else None,
            )

        if rule_id == "DOC-005":
            version_controlled = doc.get("version_controlled", True)
            return self._make_result(
                rule, RuleStatus.PASS if version_controlled else RuleStatus.WARNING,
                "Document version control in place" if version_controlled
                else "Document version control not established",
                "Implement document version control" if not version_controlled else None,
            )

        return self._make_na_result(rule)

    def _evaluate_cutoff_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Evaluate cutoff date category rules."""
        cut = data.get("cutoff", {})

        if rule_id == "CUT-001":
            verified = cut.get("cutoff_verified", False)
            return self._make_result(
                rule, RuleStatus.PASS if verified else RuleStatus.FAIL,
                "Cutoff date compliance verified (deforestation-free after 31/12/2020)"
                if verified
                else "Cutoff date compliance not verified",
                "Verify deforestation-free status after 31 December 2020" if not verified else None,
            )

        if rule_id == "CUT-002":
            evidence_sufficient = cut.get("evidence_sufficient", False)
            return self._make_result(
                rule, RuleStatus.PASS if evidence_sufficient else RuleStatus.FAIL,
                "Sufficient temporal evidence available" if evidence_sufficient
                else "Insufficient temporal evidence for cutoff verification",
                "Collect additional temporal evidence (satellite, land registry)" if not evidence_sufficient else None,
            )

        if rule_id == "CUT-003":
            land_use_verified = cut.get("land_use_history_verified", False)
            return self._make_result(
                rule, RuleStatus.PASS if land_use_verified else RuleStatus.FAIL,
                "Land use history verified - no post-cutoff deforestation" if land_use_verified
                else "Land use history not verified",
                "Verify land use history demonstrates no post-cutoff deforestation" if not land_use_verified else None,
            )

        if rule_id == "CUT-004":
            declaration_generated = cut.get("declaration_generated", False)
            return self._make_result(
                rule, RuleStatus.PASS if declaration_generated else RuleStatus.FAIL,
                "Cutoff compliance declaration generated" if declaration_generated
                else "Cutoff compliance declaration not generated",
                "Generate formal cutoff date compliance declaration" if not declaration_generated else None,
            )

        if rule_id == "CUT-005":
            has_exemption = cut.get("has_exemption", False)
            if not has_exemption:
                return self._make_na_result(rule)
            exemption_documented = cut.get("exemption_documented", False)
            return self._make_result(
                rule, RuleStatus.PASS if exemption_documented else RuleStatus.WARNING,
                "Exemption properly documented" if exemption_documented
                else "Claimed exemption not properly documented",
                "Document exemption with supporting evidence" if not exemption_documented else None,
            )

        return self._make_na_result(rule)

    def _evaluate_default_rule(
        self, rule_id: str, rule: ComplianceRule, data: Dict[str, Any]
    ) -> RuleResult:
        """Default evaluator for unknown rule categories."""
        return self._make_na_result(rule)

    # -------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------

    def _make_result(
        self,
        rule: ComplianceRule,
        status: RuleStatus,
        message: str,
        remediation_hint: Optional[str] = None,
    ) -> RuleResult:
        """Create a RuleResult from a rule and evaluation outcome.

        Args:
            rule: The compliance rule definition.
            status: Evaluation status.
            message: Result message.
            remediation_hint: Optional remediation suggestion.

        Returns:
            RuleResult instance.
        """
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            status=status,
            is_compliant=status in (RuleStatus.PASS, RuleStatus.NOT_APPLICABLE),
            message=message,
            article_reference=rule.article_reference,
            remediation_hint=remediation_hint,
        )

    def _make_na_result(self, rule: ComplianceRule) -> RuleResult:
        """Create a NOT_APPLICABLE result for a rule.

        Args:
            rule: The compliance rule definition.

        Returns:
            RuleResult with NOT_APPLICABLE status.
        """
        return RuleResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            status=RuleStatus.NOT_APPLICABLE,
            is_compliant=True,
            message="Rule not applicable to current data context",
            article_reference=rule.article_reference,
        )

    def _record_audit(
        self,
        action: str,
        entity_type: str,
        score: Optional[float],
        rule_id: Optional[str],
    ) -> None:
        """Record an audit trail entry.

        Args:
            action: Audit action performed.
            entity_type: Type of entity audited.
            score: Compliance score at time of audit.
            rule_id: Related rule ID if applicable.
        """
        entry = ComplianceAuditEntry(
            entity_id=_new_uuid(),
            entity_type=entity_type,
            action=action,
            result=f"score={score}" if score is not None else "",
            rule_id=rule_id,
            compliance_score=score,
        )
        entry.provenance_hash = _compute_hash(entry)
        self._audit_trail.append(entry)
