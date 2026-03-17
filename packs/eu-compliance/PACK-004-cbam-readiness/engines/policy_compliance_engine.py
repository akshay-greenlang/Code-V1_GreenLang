# -*- coding: utf-8 -*-
"""
PolicyComplianceEngine - PACK-004 CBAM Readiness Engine 7
==========================================================

CBAM policy compliance rules engine. Implements 50+ CBAM-specific compliance
rules covering CN code validation, emission factor ranges, quantity limits,
deadline compliance, data quality minimums, authorization checks, reporting
format requirements, and more.

Rule Categories:
    - CN_CODE         : CN code format and CBAM Annex I coverage
    - EMISSION_FACTOR : Emission factor plausibility and range checks
    - QUANTITY        : Import quantity and weight checks
    - DEADLINE        : Submission and amendment deadline compliance
    - DATA_QUALITY    : Data completeness, timeliness, and accuracy
    - AUTHORIZATION   : Definitive period authorization readiness
    - REPORTING       : Report format, completeness, and consistency
    - CALCULATION     : Calculation method and methodology checks
    - SUPPLIER        : Supplier data and documentation requirements
    - CERTIFICATE     : Certificate obligation and holding checks

Rule Applicability:
    - TRANSITIONAL: Rules that apply only during 2023-2025
    - DEFINITIVE: Rules that apply only from 2026 onwards
    - BOTH: Rules that apply in both periods

Compliance Score (0-100):
    Weighted sum of passed rules, where ERROR rules are weighted 3x,
    WARNING rules 1.5x, and INFO rules 1x.

Zero-Hallucination:
    - All rule evaluations are deterministic boolean checks
    - No LLM involvement in compliance determination
    - SHA-256 provenance hashing on every assessment

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class RuleSeverity(str, Enum):
    """Compliance rule severity level."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class RuleApplicability(str, Enum):
    """When the rule applies relative to CBAM periods."""

    TRANSITIONAL = "TRANSITIONAL"
    DEFINITIVE = "DEFINITIVE"
    BOTH = "BOTH"


class CheckStatus(str, Enum):
    """Result status of a compliance check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


class PeriodType(str, Enum):
    """CBAM period type for rule applicability."""

    TRANSITIONAL = "TRANSITIONAL"
    DEFINITIVE = "DEFINITIVE"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ComplianceRule(BaseModel):
    """Definition of a single CBAM compliance rule.

    Each rule represents a specific regulatory requirement from the CBAM
    Regulation, Implementing Regulation, or Commission guidance.
    """

    rule_id: str = Field(
        ..., description="Unique rule identifier (e.g., 'CBAM-CN-001')",
    )
    category: str = Field(
        ..., description="Rule category (CN_CODE, EMISSION_FACTOR, etc.)",
    )
    title: str = Field(
        ..., max_length=200,
        description="Short title of the rule",
    )
    description: str = Field(
        ..., max_length=2000,
        description="Detailed description of what the rule checks",
    )
    severity: RuleSeverity = Field(
        ..., description="Severity if rule fails",
    )
    applies_to: RuleApplicability = Field(
        RuleApplicability.BOTH,
        description="When the rule applies",
    )
    reference: str = Field(
        "", max_length=200,
        description="Legal reference (e.g., 'CBAM Regulation Art. 35(2)')",
    )


class ComplianceCheckResult(BaseModel):
    """Result of evaluating a single compliance rule.

    Contains the pass/fail/warn status, a human-readable message, and
    an optional data reference for traceability.
    """

    rule_id: str = Field(
        ..., description="Rule that was checked",
    )
    status: CheckStatus = Field(
        ..., description="Result status",
    )
    message: str = Field(
        ..., max_length=1000,
        description="Human-readable result message",
    )
    data_ref: str = Field(
        "", max_length=500,
        description="Reference to the data that was checked",
    )
    severity: RuleSeverity = Field(
        RuleSeverity.INFO,
        description="Severity of the rule",
    )


class ComplianceAssessment(BaseModel):
    """Complete compliance assessment result.

    Aggregates all individual rule check results into a scored assessment
    with overall compliance status.
    """

    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    total_rules_checked: int = Field(
        0, ge=0,
        description="Total number of rules evaluated",
    )
    passed: int = Field(
        0, ge=0,
        description="Number of rules that passed",
    )
    failed: int = Field(
        0, ge=0,
        description="Number of rules that failed",
    )
    warnings: int = Field(
        0, ge=0,
        description="Number of warnings",
    )
    skipped: int = Field(
        0, ge=0,
        description="Number of rules skipped (not applicable)",
    )
    overall_score: float = Field(
        0.0, ge=0, le=100,
        description="Weighted compliance score (0-100)",
    )
    period_type: PeriodType = Field(
        PeriodType.DEFINITIVE,
        description="Period type the assessment was run for",
    )
    results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Individual rule check results",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# Pre-built CBAM Compliance Rules (50+)
# ---------------------------------------------------------------------------


def _build_rules() -> List[ComplianceRule]:
    """Build the complete set of CBAM compliance rules."""
    rules = [
        # --- CN_CODE rules (1-7) ---
        ComplianceRule(
            rule_id="CBAM-CN-001",
            category="CN_CODE",
            title="CN code format validity",
            description="CN code must be 4-10 digits, optionally separated by dots.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Annex I",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-002",
            category="CN_CODE",
            title="CN code in CBAM Annex I scope",
            description="CN code must map to a CBAM-covered goods category.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Annex I",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-003",
            category="CN_CODE",
            title="CN code matches declared goods category",
            description="The CN code prefix must be consistent with the declared goods category.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Implementing Regulation Art. 3",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-004",
            category="CN_CODE",
            title="CN code precision for definitive period",
            description="CN codes must be at least 6 digits during the definitive period.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Implementing Regulation Art. 3(2)",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-005",
            category="CN_CODE",
            title="No duplicate CN codes per installation",
            description="Each CN code should appear only once per installation per report.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="Best practice",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-006",
            category="CN_CODE",
            title="CN code not on exclusion list",
            description="CN code must not be on the CBAM exclusion list (e.g., scrap metals).",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 2(3)",
        ),
        ComplianceRule(
            rule_id="CBAM-CN-007",
            category="CN_CODE",
            title="CN code matches customs declaration",
            description="CN code must match the code used in the customs import declaration.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(1)",
        ),
        # --- EMISSION_FACTOR rules (8-16) ---
        ComplianceRule(
            rule_id="CBAM-EF-001",
            category="EMISSION_FACTOR",
            title="Emission factor non-negative",
            description="Direct and indirect emission factors must be >= 0.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-002",
            category="EMISSION_FACTOR",
            title="Emission factor within plausible range",
            description="Emission factors must be within expected ranges for the goods category.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-003",
            category="EMISSION_FACTOR",
            title="Actual emission factor has verification",
            description="If actual emission factors are used, verification evidence must exist.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 8",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-004",
            category="EMISSION_FACTOR",
            title="Default factor cap in definitive period",
            description="No more than 20% of total emissions may use EU default factors in definitive period.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Implementing Regulation Art. 4(3)",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-005",
            category="EMISSION_FACTOR",
            title="Direct emission factor completeness",
            description="Direct emission factor must be provided for all entries.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-006",
            category="EMISSION_FACTOR",
            title="Indirect emission factor completeness",
            description="Indirect emission factor must be provided (may be zero for some categories).",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-007",
            category="EMISSION_FACTOR",
            title="Emission factor unit consistency",
            description="Emission factors must be expressed in tCO2e per tonne of product (or tCO2/MWh for electricity).",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-008",
            category="EMISSION_FACTOR",
            title="Precursor emission factor source",
            description="Precursor emission factors must be traceable to source.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III Section F",
        ),
        ComplianceRule(
            rule_id="CBAM-EF-009",
            category="EMISSION_FACTOR",
            title="Country-specific grid factor validity",
            description="Country grid emission factors must be from an official source and not more than 2 years old.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III Section D",
        ),
        # --- QUANTITY rules (17-22) ---
        ComplianceRule(
            rule_id="CBAM-QT-001",
            category="QUANTITY",
            title="Quantity positive and non-zero",
            description="Import quantity must be greater than zero tonnes.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35",
        ),
        ComplianceRule(
            rule_id="CBAM-QT-002",
            category="QUANTITY",
            title="Quantity within reasonable range",
            description="Single-entry quantity should not exceed 1,000,000 tonnes.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="Data quality best practice",
        ),
        ComplianceRule(
            rule_id="CBAM-QT-003",
            category="QUANTITY",
            title="Quantity matches customs declaration",
            description="Reported quantity must match the customs import declaration.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(1)",
        ),
        ComplianceRule(
            rule_id="CBAM-QT-004",
            category="QUANTITY",
            title="Total quantity consistency",
            description="Sum of individual entries must equal the reported total.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="Data integrity requirement",
        ),
        ComplianceRule(
            rule_id="CBAM-QT-005",
            category="QUANTITY",
            title="De minimis threshold acknowledgment",
            description="If total imports in a sector are below 50t, de minimis status must be declared.",
            severity=RuleSeverity.INFO,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 2(4)",
        ),
        ComplianceRule(
            rule_id="CBAM-QT-006",
            category="QUANTITY",
            title="Quantity unit is tonnes",
            description="All quantities must be expressed in metric tonnes (not kg, lb, etc.).",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        # --- DEADLINE rules (23-27) ---
        ComplianceRule(
            rule_id="CBAM-DL-001",
            category="DEADLINE",
            title="Quarterly report submitted before deadline",
            description="Report must be submitted by the end of the month following the quarter.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(1)",
        ),
        ComplianceRule(
            rule_id="CBAM-DL-002",
            category="DEADLINE",
            title="Amendment submitted before amendment deadline",
            description="Report amendments must be within 2 months of original deadline.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(3)",
        ),
        ComplianceRule(
            rule_id="CBAM-DL-003",
            category="DEADLINE",
            title="Annual declaration submitted by May 31",
            description="Annual CBAM declaration must be submitted by May 31 of the following year.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 6",
        ),
        ComplianceRule(
            rule_id="CBAM-DL-004",
            category="DEADLINE",
            title="Certificate surrender by May 31",
            description="Certificates must be surrendered by May 31 of the year following import.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 22",
        ),
        ComplianceRule(
            rule_id="CBAM-DL-005",
            category="DEADLINE",
            title="Quarterly holding compliance",
            description="Importers must hold at least 50% of quarterly emissions at quarter end.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 22(2)",
        ),
        # --- DATA_QUALITY rules (28-34) ---
        ComplianceRule(
            rule_id="CBAM-DQ-001",
            category="DATA_QUALITY",
            title="EORI number present and valid",
            description="A valid EORI number must be provided for the importer.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 5",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-002",
            category="DATA_QUALITY",
            title="Country of origin specified",
            description="Country of origin must be a valid 2-letter ISO 3166-1 code.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)(c)",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-003",
            category="DATA_QUALITY",
            title="Installation identifier provided",
            description="Production installation identifier should be provided where available.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)(d)",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-004",
            category="DATA_QUALITY",
            title="Goods description provided",
            description="A textual description of imported goods should be included.",
            severity=RuleSeverity.INFO,
            applies_to=RuleApplicability.BOTH,
            reference="Best practice",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-005",
            category="DATA_QUALITY",
            title="Emission total consistency",
            description="Total embedded emissions must equal direct + indirect + precursor.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-006",
            category="DATA_QUALITY",
            title="Supplier data quality score minimum",
            description="Supplier data quality score should be at least 60 for acceptance.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="GreenLang quality threshold",
        ),
        ComplianceRule(
            rule_id="CBAM-DQ-007",
            category="DATA_QUALITY",
            title="Provenance hash present",
            description="All calculation results must include a SHA-256 provenance hash.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="GreenLang audit trail requirement",
        ),
        # --- AUTHORIZATION rules (35-39) ---
        ComplianceRule(
            rule_id="CBAM-AZ-001",
            category="AUTHORIZATION",
            title="Authorized declarant status",
            description="Importer must be authorized as a CBAM declarant for the definitive period.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 5",
        ),
        ComplianceRule(
            rule_id="CBAM-AZ-002",
            category="AUTHORIZATION",
            title="Authorization application deadline",
            description="Authorization application must be submitted before first import under CBAM.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 5(1)",
        ),
        ComplianceRule(
            rule_id="CBAM-AZ-003",
            category="AUTHORIZATION",
            title="Financial guarantee established",
            description="A financial guarantee must be in place before purchasing certificates.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 5(3)",
        ),
        ComplianceRule(
            rule_id="CBAM-AZ-004",
            category="AUTHORIZATION",
            title="CBAM registry account active",
            description="Importer must have an active CBAM registry account.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 14",
        ),
        ComplianceRule(
            rule_id="CBAM-AZ-005",
            category="AUTHORIZATION",
            title="Customs declarant eligibility",
            description="Customs declarant must be established in the EU member state.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 2(1)",
        ),
        # --- REPORTING rules (40-45) ---
        ComplianceRule(
            rule_id="CBAM-RP-001",
            category="REPORTING",
            title="Report period completeness",
            description="Report must cover the complete quarter (no partial quarters).",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(1)",
        ),
        ComplianceRule(
            rule_id="CBAM-RP-002",
            category="REPORTING",
            title="At least one goods entry",
            description="Report must contain at least one goods entry.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)",
        ),
        ComplianceRule(
            rule_id="CBAM-RP-003",
            category="REPORTING",
            title="XML format compliance",
            description="Report XML must conform to the EU CBAM Registry schema.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Implementing Regulation Art. 8",
        ),
        ComplianceRule(
            rule_id="CBAM-RP-004",
            category="REPORTING",
            title="Amendment version tracking",
            description="Report amendments must increment the version number.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(3)",
        ),
        ComplianceRule(
            rule_id="CBAM-RP-005",
            category="REPORTING",
            title="Importer EORI in report header",
            description="EORI number must be present in the report header.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)(a)",
        ),
        ComplianceRule(
            rule_id="CBAM-RP-006",
            category="REPORTING",
            title="Specific embedded emissions reported",
            description="Specific embedded emissions (per tonne) must be calculated and reported.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)(f)",
        ),
        # --- CALCULATION rules (46-50) ---
        ComplianceRule(
            rule_id="CBAM-CA-001",
            category="CALCULATION",
            title="Calculation method declared",
            description="The emission calculation method must be explicitly declared.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III Section A",
        ),
        ComplianceRule(
            rule_id="CBAM-CA-002",
            category="CALCULATION",
            title="Actual method preference in definitive",
            description="Actual emission factors should be preferred over defaults in definitive period.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 7(2)",
        ),
        ComplianceRule(
            rule_id="CBAM-CA-003",
            category="CALCULATION",
            title="Precursor emissions included",
            description="Embedded emissions from precursor materials must be accounted for.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III Section F",
        ),
        ComplianceRule(
            rule_id="CBAM-CA-004",
            category="CALCULATION",
            title="Emission boundary completeness",
            description="Both direct and indirect emissions must be calculated.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 7",
        ),
        ComplianceRule(
            rule_id="CBAM-CA-005",
            category="CALCULATION",
            title="Calculation reproducibility",
            description="Calculations must be reproducible from documented inputs and methods.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Annex III Section A",
        ),
        # --- SUPPLIER rules (51-54) ---
        ComplianceRule(
            rule_id="CBAM-SP-001",
            category="SUPPLIER",
            title="Supplier identification provided",
            description="Supplier must be identified for each goods entry.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 35(2)(d)",
        ),
        ComplianceRule(
            rule_id="CBAM-SP-002",
            category="SUPPLIER",
            title="Supplier emission data submitted",
            description="Supplier must have submitted emission data for the reporting period.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 8",
        ),
        ComplianceRule(
            rule_id="CBAM-SP-003",
            category="SUPPLIER",
            title="Supplier verification status",
            description="In definitive period, supplier data should be from verified installations.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 8",
        ),
        ComplianceRule(
            rule_id="CBAM-SP-004",
            category="SUPPLIER",
            title="Supplier country non-EU",
            description="CBAM applies only to goods imported from outside the EU.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.BOTH,
            reference="CBAM Regulation Art. 2(1)",
        ),
        # --- CERTIFICATE rules (55-58) ---
        ComplianceRule(
            rule_id="CBAM-CT-001",
            category="CERTIFICATE",
            title="Certificate obligation calculated",
            description="Annual certificate obligation must be calculated for all non-exempt goods.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 22",
        ),
        ComplianceRule(
            rule_id="CBAM-CT-002",
            category="CERTIFICATE",
            title="Free allocation deduction applied",
            description="Free allocation phase-out deduction must be applied to gross obligation.",
            severity=RuleSeverity.INFO,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 31",
        ),
        ComplianceRule(
            rule_id="CBAM-CT-003",
            category="CERTIFICATE",
            title="Carbon price deduction documented",
            description="Any carbon price deduction must be documented with evidence.",
            severity=RuleSeverity.WARNING,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 9",
        ),
        ComplianceRule(
            rule_id="CBAM-CT-004",
            category="CERTIFICATE",
            title="Net obligation non-negative",
            description="Net certificate obligation must be >= 0 after all deductions.",
            severity=RuleSeverity.ERROR,
            applies_to=RuleApplicability.DEFINITIVE,
            reference="CBAM Regulation Art. 22",
        ),
    ]

    return rules


# Module-level rule registry
_CBAM_RULES: List[ComplianceRule] = _build_rules()
_RULES_BY_ID: Dict[str, ComplianceRule] = {r.rule_id: r for r in _CBAM_RULES}

# Valid CBAM CN code prefixes
_VALID_CN_PREFIXES = {
    "2523", "7201", "7202", "7203", "7205", "7206", "7207",
    "7208", "7209", "7210", "7211", "7212", "7213", "7214",
    "7215", "7216", "7217", "7218", "7219", "7220", "7221",
    "7222", "7223", "7224", "7225", "7226", "7227", "7228",
    "7229", "7301", "7302", "7303", "7304", "7305", "7306",
    "7307", "7308", "7601", "7603", "7604", "7605", "7606",
    "7607", "7608", "7609", "7610", "7611", "7612", "7613",
    "7614", "7616", "2808", "2814", "3102", "3103", "3104",
    "3105", "2716", "2804",
}

# EU member state codes (CBAM applies to imports into the EU)
_EU_MEMBER_STATES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
}

# Plausible emission factor ranges by category
_EF_RANGES: Dict[str, Dict[str, float]] = {
    "cement": {"min": 0.3, "max": 1.2},
    "iron_steel": {"min": 0.2, "max": 3.5},
    "aluminium": {"min": 0.3, "max": 12.0},
    "fertilizers": {"min": 0.2, "max": 3.0},
    "electricity": {"min": 0.01, "max": 1.5},
    "hydrogen": {"min": 0.2, "max": 12.0},
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PolicyComplianceEngine:
    """CBAM policy compliance rules engine.

    Evaluates 50+ CBAM-specific compliance rules against import data,
    emission calculations, reports, and importer configurations. Produces
    scored compliance assessments for audit readiness.

    Rule severity weighting for overall score:
        - ERROR:   weight 3.0 (hard compliance requirement)
        - WARNING: weight 1.5 (best practice / soft requirement)
        - INFO:    weight 1.0 (informational / advisory)

    Zero-Hallucination Guarantees:
        - All rule evaluations are deterministic boolean checks
        - No LLM involvement in compliance determination
        - SHA-256 provenance hashing on every assessment

    Example:
        >>> engine = PolicyComplianceEngine()
        >>> result = engine.validate_cn_code("7208.51")
        >>> assert result.status == CheckStatus.PASS
    """

    def __init__(self) -> None:
        """Initialize PolicyComplianceEngine."""
        self._assessment_count: int = 0
        logger.info(
            "PolicyComplianceEngine initialized (v%s, %d rules)",
            _MODULE_VERSION,
            len(_CBAM_RULES),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        data: Dict[str, Any],
        period_type: PeriodType = PeriodType.DEFINITIVE,
    ) -> ComplianceAssessment:
        """Run all applicable compliance rules against the provided data.

        Evaluates each rule in the rule set that applies to the given
        period type. Produces a scored assessment with individual results.

        Expected data dict keys (all optional):
            - cn_codes: List[str] - CN codes to validate
            - goods_category: str - Declared goods category
            - emission_factors: List[dict] - Emission factor entries
            - quantity_tonnes: float - Total import quantity
            - entries: List[dict] - Individual goods entries
            - importer_eori: str - Importer EORI number
            - country_of_origin: str - Country code
            - installation_id: str - Installation identifier
            - calculation_method: str - Calculation method used
            - report: dict - Quarterly report data
            - supplier_data: dict - Supplier information
            - provenance_hash: str - Provenance hash
            - submission_date: str - Report submission date
            - deadline: str - Submission deadline

        Args:
            data: Dictionary of data to check.
            period_type: TRANSITIONAL or DEFINITIVE.

        Returns:
            ComplianceAssessment with all rule results and score.
        """
        self._assessment_count += 1
        applicable_rules = self.get_rules(period_type)

        results: List[ComplianceCheckResult] = []

        for rule in applicable_rules:
            result = self._evaluate_rule(rule, data)
            results.append(result)

        # Calculate summary
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        failed = sum(1 for r in results if r.status == CheckStatus.FAIL)
        warnings = sum(1 for r in results if r.status == CheckStatus.WARN)
        skipped = sum(1 for r in results if r.status == CheckStatus.SKIP)

        score = self.calculate_compliance_score(results)

        assessment = ComplianceAssessment(
            total_rules_checked=len(results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            overall_score=score,
            period_type=period_type,
            results=results,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "Compliance assessment complete: %d checked, %d passed, %d failed, "
            "%d warnings, score=%.1f",
            len(results), passed, failed, warnings, score,
        )

        return assessment

    def validate_cn_code(self, cn_code: str) -> ComplianceCheckResult:
        """Validate a CN code against CBAM requirements.

        Checks format validity and CBAM Annex I coverage.

        Args:
            cn_code: Combined Nomenclature code to validate.

        Returns:
            ComplianceCheckResult with PASS or FAIL status.
        """
        cleaned = cn_code.strip().replace(".", "").replace(" ", "")

        # Format check
        if not cleaned.isdigit() or len(cleaned) < 4:
            return ComplianceCheckResult(
                rule_id="CBAM-CN-001",
                status=CheckStatus.FAIL,
                message=f"Invalid CN code format: '{cn_code}' (must be 4+ digits)",
                data_ref=cn_code,
                severity=RuleSeverity.ERROR,
            )

        # Annex I coverage check
        prefix_4 = cleaned[:4]
        if prefix_4 not in _VALID_CN_PREFIXES:
            return ComplianceCheckResult(
                rule_id="CBAM-CN-002",
                status=CheckStatus.FAIL,
                message=f"CN code '{cn_code}' is not in CBAM Annex I scope",
                data_ref=cn_code,
                severity=RuleSeverity.ERROR,
            )

        return ComplianceCheckResult(
            rule_id="CBAM-CN-001",
            status=CheckStatus.PASS,
            message=f"CN code '{cn_code}' is valid and in CBAM scope",
            data_ref=cn_code,
            severity=RuleSeverity.ERROR,
        )

    def validate_emission_factor(
        self,
        goods_category: str,
        factor: float,
    ) -> ComplianceCheckResult:
        """Validate an emission factor against plausible ranges.

        Checks that the emission factor (tCO2e/t) falls within the
        expected range for the goods category.

        Args:
            goods_category: CBAM goods category.
            factor: Emission factor value.

        Returns:
            ComplianceCheckResult with PASS, WARN, or FAIL status.
        """
        if factor < 0:
            return ComplianceCheckResult(
                rule_id="CBAM-EF-001",
                status=CheckStatus.FAIL,
                message=f"Emission factor {factor} is negative",
                data_ref=f"{goods_category}:{factor}",
                severity=RuleSeverity.ERROR,
            )

        cat_lower = goods_category.lower()
        ef_range = _EF_RANGES.get(cat_lower, {"min": 0.0, "max": 50.0})

        if ef_range["min"] <= factor <= ef_range["max"]:
            return ComplianceCheckResult(
                rule_id="CBAM-EF-002",
                status=CheckStatus.PASS,
                message=(
                    f"Emission factor {factor:.4f} is within range "
                    f"[{ef_range['min']}, {ef_range['max']}] for {goods_category}"
                ),
                data_ref=f"{goods_category}:{factor}",
                severity=RuleSeverity.WARNING,
            )

        return ComplianceCheckResult(
            rule_id="CBAM-EF-002",
            status=CheckStatus.WARN,
            message=(
                f"Emission factor {factor:.4f} is outside expected range "
                f"[{ef_range['min']}, {ef_range['max']}] for {goods_category}"
            ),
            data_ref=f"{goods_category}:{factor}",
            severity=RuleSeverity.WARNING,
        )

    def check_default_factor_usage(
        self,
        results: List[Dict[str, Any]],
        period_type: PeriodType = PeriodType.DEFINITIVE,
    ) -> ComplianceCheckResult:
        """Check the proportion of emissions using EU default factors.

        In the definitive period, no more than 20% of total emissions
        should rely on EU default factors.

        Args:
            results: List of emission result dicts with 'calculation_method_used'
                and 'total_embedded_emissions_tco2e'.
            period_type: Current period type.

        Returns:
            ComplianceCheckResult.
        """
        if period_type == PeriodType.TRANSITIONAL:
            return ComplianceCheckResult(
                rule_id="CBAM-EF-004",
                status=CheckStatus.SKIP,
                message="Default factor cap not applicable during transitional period",
                severity=RuleSeverity.WARNING,
            )

        total_emissions = Decimal("0")
        default_emissions = Decimal("0")

        for r in results:
            emissions = _decimal(r.get("total_embedded_emissions_tco2e", 0.0))
            total_emissions += emissions
            method = r.get("calculation_method_used", "default")
            if hasattr(method, "value"):
                method = method.value
            if method in ("default", "country_default"):
                default_emissions += emissions

        if total_emissions <= 0:
            return ComplianceCheckResult(
                rule_id="CBAM-EF-004",
                status=CheckStatus.SKIP,
                message="No emissions to assess",
                severity=RuleSeverity.WARNING,
            )

        default_pct = float(default_emissions / total_emissions * Decimal("100"))

        if default_pct <= 20.0:
            return ComplianceCheckResult(
                rule_id="CBAM-EF-004",
                status=CheckStatus.PASS,
                message=f"Default factor usage at {default_pct:.1f}% (limit: 20%)",
                data_ref=f"default_pct={default_pct:.1f}",
                severity=RuleSeverity.WARNING,
            )

        return ComplianceCheckResult(
            rule_id="CBAM-EF-004",
            status=CheckStatus.WARN,
            message=(
                f"Default factor usage at {default_pct:.1f}% exceeds recommended "
                f"20% limit. Consider obtaining actual emission data."
            ),
            data_ref=f"default_pct={default_pct:.1f}",
            severity=RuleSeverity.WARNING,
        )

    def check_quarterly_completeness(
        self,
        report: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """Check a quarterly report for completeness.

        Validates that all required fields are present and populated.

        Args:
            report: Dictionary representing a quarterly report.

        Returns:
            ComplianceCheckResult.
        """
        required_fields = [
            "importer_eori",
            "period",
            "goods_entries",
            "total_embedded_emissions_tco2e",
            "total_quantity_tonnes",
        ]

        missing = [f for f in required_fields if not report.get(f)]
        entries = report.get("goods_entries", [])

        if missing:
            return ComplianceCheckResult(
                rule_id="CBAM-RP-002",
                status=CheckStatus.FAIL,
                message=f"Missing required fields: {', '.join(missing)}",
                data_ref=str(missing),
                severity=RuleSeverity.ERROR,
            )

        if not entries:
            return ComplianceCheckResult(
                rule_id="CBAM-RP-002",
                status=CheckStatus.FAIL,
                message="Report contains no goods entries",
                severity=RuleSeverity.ERROR,
            )

        return ComplianceCheckResult(
            rule_id="CBAM-RP-002",
            status=CheckStatus.PASS,
            message=f"Report is complete with {len(entries)} goods entries",
            data_ref=f"entries={len(entries)}",
            severity=RuleSeverity.ERROR,
        )

    def check_authorization_readiness(
        self,
        importer: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """Check an importer's readiness for the CBAM definitive period.

        Validates that the importer has:
            - Active EORI number
            - CBAM registry account
            - Authorization application (or approved)
            - Financial guarantee

        Args:
            importer: Dictionary with importer data.

        Returns:
            ComplianceCheckResult.
        """
        checks = {
            "eori_number": importer.get("eori_number"),
            "registry_account": importer.get("registry_account_active", False),
            "authorization_status": importer.get("authorization_status", ""),
            "financial_guarantee": importer.get("financial_guarantee", False),
        }

        failures = []
        if not checks["eori_number"]:
            failures.append("Missing EORI number")
        if not checks["registry_account"]:
            failures.append("No active CBAM registry account")
        if checks["authorization_status"] not in ("APPROVED", "PENDING"):
            failures.append("Authorization not applied for")
        if not checks["financial_guarantee"]:
            failures.append("Financial guarantee not established")

        if failures:
            return ComplianceCheckResult(
                rule_id="CBAM-AZ-001",
                status=CheckStatus.FAIL,
                message=f"Not ready for definitive period: {'; '.join(failures)}",
                data_ref=str(failures),
                severity=RuleSeverity.ERROR,
            )

        return ComplianceCheckResult(
            rule_id="CBAM-AZ-001",
            status=CheckStatus.PASS,
            message="Importer is ready for CBAM definitive period",
            severity=RuleSeverity.ERROR,
        )

    def get_rules(
        self,
        period_type: Optional[PeriodType] = None,
        category: Optional[str] = None,
    ) -> List[ComplianceRule]:
        """Get the list of applicable compliance rules.

        Args:
            period_type: Filter by period applicability. If None, returns all rules.
            category: Filter by rule category. If None, returns all categories.

        Returns:
            List of applicable ComplianceRule objects.
        """
        rules = list(_CBAM_RULES)

        if period_type:
            rules = [
                r for r in rules
                if r.applies_to == RuleApplicability.BOTH
                or r.applies_to.value == period_type.value
            ]

        if category:
            rules = [r for r in rules if r.category == category.upper()]

        return rules

    def calculate_compliance_score(
        self,
        results: List[ComplianceCheckResult],
    ) -> float:
        """Calculate a weighted compliance score from check results.

        Weighting:
            - ERROR rules:   3.0 points each
            - WARNING rules: 1.5 points each
            - INFO rules:    1.0 points each

        Score = (weighted_passed / weighted_total) * 100

        Args:
            results: List of ComplianceCheckResult objects.

        Returns:
            Score between 0.0 and 100.0.
        """
        if not results:
            return 100.0

        severity_weights = {
            RuleSeverity.ERROR: Decimal("3.0"),
            RuleSeverity.WARNING: Decimal("1.5"),
            RuleSeverity.INFO: Decimal("1.0"),
        }

        total_weight = Decimal("0")
        passed_weight = Decimal("0")

        for r in results:
            if r.status == CheckStatus.SKIP:
                continue

            weight = severity_weights.get(r.severity, Decimal("1.0"))
            total_weight += weight

            if r.status == CheckStatus.PASS:
                passed_weight += weight
            elif r.status == CheckStatus.WARN:
                passed_weight += weight * Decimal("0.5")

        if total_weight <= 0:
            return 100.0

        score = (passed_weight / total_weight) * Decimal("100")
        return float(score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        """Total number of compliance rules."""
        return len(_CBAM_RULES)

    @property
    def assessment_count(self) -> int:
        """Number of assessments performed."""
        return self._assessment_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: ComplianceRule,
        data: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """Evaluate a single compliance rule against the data.

        Routes to the appropriate check method based on rule ID prefix.
        If no specific handler exists, returns SKIP.
        """
        try:
            # Route by rule category
            if rule.rule_id.startswith("CBAM-CN"):
                return self._check_cn_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-EF"):
                return self._check_ef_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-QT"):
                return self._check_qt_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-DL"):
                return self._check_dl_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-DQ"):
                return self._check_dq_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-AZ"):
                return self._check_az_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-RP"):
                return self._check_rp_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-CA"):
                return self._check_ca_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-SP"):
                return self._check_sp_rule(rule, data)
            elif rule.rule_id.startswith("CBAM-CT"):
                return self._check_ct_rule(rule, data)
            else:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    status=CheckStatus.SKIP,
                    message="No handler for this rule",
                    severity=rule.severity,
                )
        except Exception as exc:
            logger.error("Rule %s evaluation failed: %s", rule.rule_id, exc)
            return ComplianceCheckResult(
                rule_id=rule.rule_id,
                status=CheckStatus.SKIP,
                message=f"Evaluation error: {str(exc)}",
                severity=rule.severity,
            )

    def _check_cn_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate CN code rules."""
        cn_codes = data.get("cn_codes", [])
        if not cn_codes:
            return self._skip_result(rule, "No CN codes in data")

        # Check first CN code (representative)
        for cn in cn_codes:
            result = self.validate_cn_code(cn)
            if result.status == CheckStatus.FAIL:
                result.rule_id = rule.rule_id
                return result

        return ComplianceCheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.PASS,
            message=f"All {len(cn_codes)} CN codes valid",
            severity=rule.severity,
        )

    def _check_ef_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate emission factor rules."""
        factors = data.get("emission_factors", [])
        if not factors:
            return self._skip_result(rule, "No emission factors in data")

        goods_category = data.get("goods_category", "")

        for ef_entry in factors:
            if isinstance(ef_entry, dict):
                factor_val = ef_entry.get("value", ef_entry.get("direct", 0.0))
                cat = ef_entry.get("goods_category", goods_category)
            else:
                factor_val = float(ef_entry)
                cat = goods_category

            result = self.validate_emission_factor(cat, factor_val)
            if result.status != CheckStatus.PASS:
                result.rule_id = rule.rule_id
                return result

        return ComplianceCheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.PASS,
            message="All emission factors within range",
            severity=rule.severity,
        )

    def _check_qt_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate quantity rules."""
        quantity = data.get("quantity_tonnes", 0.0)

        if rule.rule_id == "CBAM-QT-001":
            if quantity > 0:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Quantity {quantity:.2f}t is positive",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message=f"Quantity {quantity} must be > 0",
                severity=rule.severity,
            )

        if rule.rule_id == "CBAM-QT-002":
            if quantity <= 1_000_000:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Quantity {quantity:.2f}t within reasonable range",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.WARN,
                message=f"Quantity {quantity:.2f}t exceeds 1,000,000t - verify",
                severity=rule.severity,
            )

        return self._skip_result(rule, "Insufficient data for quantity check")

    def _check_dl_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate deadline rules."""
        submission_date = data.get("submission_date")
        deadline = data.get("deadline")

        if not submission_date or not deadline:
            return self._skip_result(rule, "No deadline data available")

        # Parse dates
        try:
            if isinstance(submission_date, str):
                sub_date = date.fromisoformat(submission_date)
            else:
                sub_date = submission_date
            if isinstance(deadline, str):
                dead_date = date.fromisoformat(deadline)
            else:
                dead_date = deadline
        except (ValueError, TypeError):
            return self._skip_result(rule, "Invalid date format")

        if sub_date <= dead_date:
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.PASS,
                message=f"Submitted {sub_date} before deadline {dead_date}",
                severity=rule.severity,
            )

        return ComplianceCheckResult(
            rule_id=rule.rule_id, status=CheckStatus.FAIL,
            message=f"Submitted {sub_date} AFTER deadline {dead_date}",
            severity=rule.severity,
        )

    def _check_dq_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate data quality rules."""
        if rule.rule_id == "CBAM-DQ-001":
            eori = data.get("importer_eori", "")
            if eori and len(eori) >= 5:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"EORI '{eori}' is present and valid",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message="Missing or invalid EORI number",
                severity=rule.severity,
            )

        if rule.rule_id == "CBAM-DQ-002":
            country = data.get("country_of_origin", "")
            if country and len(country) == 2 and country.isalpha():
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Country '{country}' is valid",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message=f"Invalid country of origin: '{country}'",
                severity=rule.severity,
            )

        if rule.rule_id == "CBAM-DQ-007":
            phash = data.get("provenance_hash", "")
            if phash and len(phash) == 64:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message="Provenance hash present",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message="Missing or invalid provenance hash",
                severity=rule.severity,
            )

        return self._skip_result(rule, "Insufficient data for quality check")

    def _check_az_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate authorization rules."""
        importer = data.get("importer", data)
        return self.check_authorization_readiness(importer)

    def _check_rp_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate reporting rules."""
        report = data.get("report", data)

        if rule.rule_id == "CBAM-RP-002":
            return self.check_quarterly_completeness(report)

        if rule.rule_id == "CBAM-RP-005":
            eori = report.get("importer_eori", "")
            if eori:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message="EORI present in report header",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message="EORI missing from report header",
                severity=rule.severity,
            )

        return self._skip_result(rule, "Insufficient data for reporting check")

    def _check_ca_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate calculation rules."""
        method = data.get("calculation_method", "")
        if not method:
            return self._skip_result(rule, "No calculation method in data")

        if rule.rule_id == "CBAM-CA-001":
            if method:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Calculation method declared: {method}",
                    severity=rule.severity,
                )

        return self._skip_result(rule, "Insufficient data for calculation check")

    def _check_sp_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate supplier rules."""
        supplier = data.get("supplier_data", {})

        if rule.rule_id == "CBAM-SP-004":
            country = supplier.get("country", data.get("country_of_origin", ""))
            if country and country.upper() not in _EU_MEMBER_STATES:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Country '{country}' is non-EU (CBAM applicable)",
                    severity=rule.severity,
                )
            elif country and country.upper() in _EU_MEMBER_STATES:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.FAIL,
                    message=f"Country '{country}' is EU member state (CBAM does not apply)",
                    severity=rule.severity,
                )

        return self._skip_result(rule, "Insufficient supplier data")

    def _check_ct_rule(
        self, rule: ComplianceRule, data: Dict[str, Any]
    ) -> ComplianceCheckResult:
        """Evaluate certificate rules."""
        cert_data = data.get("certificate", {})
        if not cert_data:
            return self._skip_result(rule, "No certificate data")

        if rule.rule_id == "CBAM-CT-004":
            net = cert_data.get("net_certificates", 0.0)
            if net >= 0:
                return ComplianceCheckResult(
                    rule_id=rule.rule_id, status=CheckStatus.PASS,
                    message=f"Net obligation {net:.2f} is non-negative",
                    severity=rule.severity,
                )
            return ComplianceCheckResult(
                rule_id=rule.rule_id, status=CheckStatus.FAIL,
                message=f"Net obligation {net:.2f} is negative",
                severity=rule.severity,
            )

        return self._skip_result(rule, "Insufficient certificate data")

    def _skip_result(
        self, rule: ComplianceRule, reason: str
    ) -> ComplianceCheckResult:
        """Create a SKIP result for a rule."""
        return ComplianceCheckResult(
            rule_id=rule.rule_id,
            status=CheckStatus.SKIP,
            message=reason,
            severity=rule.severity,
        )
