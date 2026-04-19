"""
PACK-048 GHG Assurance Prep Pack - Configuration Manager

Pydantic v2 configuration for GHG assurance preparation including evidence
consolidation, readiness assessment, calculation provenance verification,
internal control testing, verifier collaboration, materiality assessment,
sampling plan generation, regulatory requirement mapping, cost and timeline
estimation, and assurance reporting.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (engagement-specific defaults)
    3. Environment overrides (ASSURANCE_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    ISAE 3410 (IAASB) - Assurance Engagements on GHG Statements
    ISO 14064-3:2019 - Specification for validation and verification
    AA1000AS v3 (AccountAbility) - Assurance Standard
    ISAE 3000 (Revised) - Assurance Engagements Other than Audits
    SSAE 18 (AICPA) - Attestation Standards for US engagements
    EU CSRD (2022/2464) - Mandatory limited assurance from 2024
    US SEC Climate Disclosure Rules (2024) - Attestation requirements
    California SB 253 (2023) - Climate Corporate Data Accountability Act
    UK SECR (2019) - Streamlined Energy and Carbon Reporting
    GHG Protocol Corporate Standard - Verification guidance
    ISO 14064-1:2018 Clause 9 - Verification requirements
    PCAF Global GHG Accounting Standard v3 - Data quality verification

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent

# =============================================================================
# Helper Functions
# =============================================================================

def _new_uuid() -> str:
    """Return new UUID4 string (mockable for testing)."""
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string for provenance tracking."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# Enums (18 total)
# =============================================================================

class AssuranceStandard(str, Enum):
    """Assurance engagement standard governing the verification."""
    ISAE_3410 = "ISAE_3410"
    ISO_14064_3 = "ISO_14064_3"
    AA1000AS_V3 = "AA1000AS_V3"
    ISAE_3000 = "ISAE_3000"
    SSAE_18 = "SSAE_18"
    CUSTOM = "CUSTOM"

class AssuranceLevel(str, Enum):
    """Level of assurance to be obtained or provided."""
    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    EXAMINATION = "EXAMINATION"
    REVIEW = "REVIEW"

class EvidenceCategory(str, Enum):
    """Category of audit evidence for assurance preparation."""
    SOURCE_DATA = "SOURCE_DATA"
    EMISSION_FACTOR = "EMISSION_FACTOR"
    CALCULATION = "CALCULATION"
    ASSUMPTION = "ASSUMPTION"
    METHODOLOGY = "METHODOLOGY"
    BOUNDARY = "BOUNDARY"
    COMPLETENESS = "COMPLETENESS"
    CONTROL = "CONTROL"
    APPROVAL = "APPROVAL"
    EXTERNAL = "EXTERNAL"

class EvidenceQuality(str, Enum):
    """Quality rating for individual evidence items."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ADEQUATE = "ADEQUATE"
    MARGINAL = "MARGINAL"
    INSUFFICIENT = "INSUFFICIENT"

class ControlCategory(str, Enum):
    """Category of internal control over GHG reporting."""
    DATA_COLLECTION = "DATA_COLLECTION"
    CALCULATION = "CALCULATION"
    REVIEW = "REVIEW"
    REPORTING = "REPORTING"
    IT_GENERAL = "IT_GENERAL"

class ControlType(str, Enum):
    """Type of internal control."""
    PREVENTIVE = "PREVENTIVE"
    DETECTIVE = "DETECTIVE"
    CORRECTIVE = "CORRECTIVE"

class ControlEffectiveness(str, Enum):
    """Effectiveness rating for an internal control."""
    EFFECTIVE = "EFFECTIVE"
    PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
    INEFFECTIVE = "INEFFECTIVE"
    NOT_TESTED = "NOT_TESTED"

class ControlMaturity(str, Enum):
    """Maturity level for internal controls (CMMI-inspired)."""
    LEVEL_1_ADHOC = "LEVEL_1_ADHOC"
    LEVEL_2_REPEATABLE = "LEVEL_2_REPEATABLE"
    LEVEL_3_DEFINED = "LEVEL_3_DEFINED"
    LEVEL_4_MANAGED = "LEVEL_4_MANAGED"
    LEVEL_5_OPTIMISED = "LEVEL_5_OPTIMISED"

class FindingSeverity(str, Enum):
    """Severity of a verifier finding."""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    OBSERVATION = "OBSERVATION"

class FindingType(str, Enum):
    """Type of verifier finding."""
    NON_CONFORMITY = "NON_CONFORMITY"
    OBSERVATION = "OBSERVATION"
    OPPORTUNITY = "OPPORTUNITY"
    RECOMMENDATION = "RECOMMENDATION"
    GOOD_PRACTICE = "GOOD_PRACTICE"

class QueryPriority(str, Enum):
    """Priority of a verifier query."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class QueryStatus(str, Enum):
    """Status of a verifier query in the engagement lifecycle."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESPONDED = "RESPONDED"
    FOLLOW_UP = "FOLLOW_UP"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"

class MaterialityType(str, Enum):
    """Type of materiality threshold for assurance."""
    OVERALL = "OVERALL"
    PERFORMANCE = "PERFORMANCE"
    CLEARLY_TRIVIAL = "CLEARLY_TRIVIAL"
    SCOPE_SPECIFIC = "SCOPE_SPECIFIC"
    SPECIFIC_ITEM = "SPECIFIC_ITEM"

class SamplingMethod(str, Enum):
    """Statistical sampling method for assurance testing."""
    MUS = "MUS"
    RANDOM = "RANDOM"
    SYSTEMATIC = "SYSTEMATIC"
    STRATIFIED = "STRATIFIED"
    JUDGMENTAL = "JUDGMENTAL"

class Jurisdiction(str, Enum):
    """Regulatory jurisdiction with GHG assurance mandates."""
    EU_CSRD = "EU_CSRD"
    US_SEC = "US_SEC"
    CALIFORNIA_SB253 = "CALIFORNIA_SB253"
    UK_SECR = "UK_SECR"
    SINGAPORE_SGX = "SINGAPORE_SGX"
    JAPAN_SSBJ = "JAPAN_SSBJ"
    AUSTRALIA_ASRS = "AUSTRALIA_ASRS"
    SOUTH_KOREA_KSQF = "SOUTH_KOREA_KSQF"
    HONG_KONG_HKEX = "HONG_KONG_HKEX"
    BRAZIL_CVM = "BRAZIL_CVM"
    INDIA_BRSR = "INDIA_BRSR"
    CANADA_CSSB = "CANADA_CSSB"

class CompanySize(str, Enum):
    """Company size classification for cost modelling and thresholds."""
    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    LARGE_ACCELERATED_FILER = "LARGE_ACCELERATED_FILER"
    ACCELERATED_FILER = "ACCELERATED_FILER"
    NON_ACCELERATED_FILER = "NON_ACCELERATED_FILER"

class EngagementPhase(str, Enum):
    """Phase of the assurance engagement lifecycle."""
    PLANNING = "PLANNING"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    FIELDWORK = "FIELDWORK"
    REPORTING = "REPORTING"
    CLOSEOUT = "CLOSEOUT"

# =============================================================================
# Reference Data Constants
# =============================================================================

STANDARD_CONTROLS: Dict[str, Dict[str, Any]] = {
    # Data Collection controls (DC-01 to DC-05)
    "DC-01": {
        "name": "Source Data Completeness Check",
        "category": ControlCategory.DATA_COLLECTION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated completeness check ensures all required source data "
            "files are received for each reporting period and facility."
        ),
    },
    "DC-02": {
        "name": "Meter Reading Validation",
        "category": ControlCategory.DATA_COLLECTION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated range check on meter readings flags values outside "
            "historical +/- 2 standard deviation bounds for investigation."
        ),
    },
    "DC-03": {
        "name": "Data Entry Four-Eyes Review",
        "category": ControlCategory.DATA_COLLECTION.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "All manually entered activity data requires review and approval "
            "by a second person before inclusion in calculations."
        ),
    },
    "DC-04": {
        "name": "Supplier Data Reconciliation",
        "category": ControlCategory.DATA_COLLECTION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Quarterly reconciliation of supplier-provided activity data "
            "against invoices and purchase orders."
        ),
    },
    "DC-05": {
        "name": "Data Cut-Off Procedure",
        "category": ControlCategory.DATA_COLLECTION.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "Formal data cut-off procedure ensures all activity data is "
            "captured within the correct reporting period boundaries."
        ),
    },
    # Calculation controls (CA-01 to CA-05)
    "CA-01": {
        "name": "Emission Factor Version Control",
        "category": ControlCategory.CALCULATION.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "Emission factors are version-controlled with approval workflow. "
            "Only approved factors from authorised databases are used."
        ),
    },
    "CA-02": {
        "name": "Calculation Logic Unit Testing",
        "category": ControlCategory.CALCULATION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated unit tests validate all calculation formulas against "
            "known test vectors with expected results."
        ),
    },
    "CA-03": {
        "name": "GWP Consistency Check",
        "category": ControlCategory.CALCULATION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated check ensures consistent GWP values (AR4/AR5/AR6) "
            "are applied across all gas-to-CO2e conversions."
        ),
    },
    "CA-04": {
        "name": "Scope Boundary Validation",
        "category": ControlCategory.CALCULATION.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "Scope 1/2/3 boundary definitions are documented and validated "
            "against the organisational boundary before each calculation run."
        ),
    },
    "CA-05": {
        "name": "Deterministic Replay Verification",
        "category": ControlCategory.CALCULATION.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "SHA-256 hash chain verification ensures calculation outputs "
            "are reproducible from identical inputs (zero-hallucination)."
        ),
    },
    # Review controls (RV-01 to RV-05)
    "RV-01": {
        "name": "Management Review of GHG Statement",
        "category": ControlCategory.REVIEW.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Senior management reviews and signs off on the final GHG "
            "statement before submission to the verifier."
        ),
    },
    "RV-02": {
        "name": "Year-on-Year Variance Analysis",
        "category": ControlCategory.REVIEW.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated variance analysis flags any scope or category "
            "with >10% year-on-year change for management investigation."
        ),
    },
    "RV-03": {
        "name": "Cross-Scope Reconciliation",
        "category": ControlCategory.REVIEW.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Reconciliation between Scope 1, 2, and 3 totals and the "
            "consolidated GHG statement ensures no double counting."
        ),
    },
    "RV-04": {
        "name": "Materiality Threshold Review",
        "category": ControlCategory.REVIEW.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Review of all items above the materiality threshold by "
            "the GHG reporting manager before finalisation."
        ),
    },
    "RV-05": {
        "name": "Base Year Recalculation Review",
        "category": ControlCategory.REVIEW.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Any base year recalculation trigger events are reviewed "
            "and approved before base year adjustments are applied."
        ),
    },
    # Reporting controls (RE-01 to RE-05)
    "RE-01": {
        "name": "Disclosure Completeness Check",
        "category": ControlCategory.REPORTING.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated completeness check against the applicable disclosure "
            "framework requirements (ESRS E1, CDP, SEC, etc.)."
        ),
    },
    "RE-02": {
        "name": "Report Version Control",
        "category": ControlCategory.REPORTING.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "All report drafts are version-controlled with tracked changes "
            "and formal approval before each version release."
        ),
    },
    "RE-03": {
        "name": "XBRL Tag Validation",
        "category": ControlCategory.REPORTING.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated validation of XBRL tags against the applicable "
            "taxonomy (ESRS 2024) before digital filing."
        ),
    },
    "RE-04": {
        "name": "Third-Party Data Reconciliation",
        "category": ControlCategory.REPORTING.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Reconciliation of reported figures against third-party data "
            "sources (utility bills, invoices, ERP extracts)."
        ),
    },
    "RE-05": {
        "name": "Provenance Hash Publication",
        "category": ControlCategory.REPORTING.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "SHA-256 provenance hash of the final GHG statement is published "
            "to the audit trail for immutable record keeping."
        ),
    },
    # IT General controls (IT-01 to IT-05)
    "IT-01": {
        "name": "Access Control Review",
        "category": ControlCategory.IT_GENERAL.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "Quarterly review of user access rights to the GHG reporting "
            "system ensures principle of least privilege."
        ),
    },
    "IT-02": {
        "name": "Change Management Approval",
        "category": ControlCategory.IT_GENERAL.value,
        "type": ControlType.PREVENTIVE.value,
        "description": (
            "All changes to calculation logic, emission factors, or system "
            "configuration require formal change management approval."
        ),
    },
    "IT-03": {
        "name": "Backup and Recovery Testing",
        "category": ControlCategory.IT_GENERAL.value,
        "type": ControlType.CORRECTIVE.value,
        "description": (
            "Annual backup and recovery testing ensures GHG data can be "
            "restored within the defined RTO and RPO."
        ),
    },
    "IT-04": {
        "name": "Audit Log Integrity",
        "category": ControlCategory.IT_GENERAL.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated integrity check on audit logs ensures no tampering "
            "with the GHG data trail. SHA-256 hash chain on log entries."
        ),
    },
    "IT-05": {
        "name": "System Interface Validation",
        "category": ControlCategory.IT_GENERAL.value,
        "type": ControlType.DETECTIVE.value,
        "description": (
            "Automated reconciliation of data transferred between systems "
            "(ERP, meters, GHG platform) to detect interface errors."
        ),
    },
}

JURISDICTION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "EU_CSRD": {
        "jurisdiction_name": "European Union - CSRD",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2028-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "effective_date": "2024-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Large undertakings (>250 employees or >40M EUR revenue)",
        "notes": "Limited assurance from 2024; reasonable assurance from 2028 (phased)",
    },
    "US_SEC": {
        "jurisdiction_name": "United States - SEC Climate Disclosure",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2029-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2026-01-01",
        "standard": AssuranceStandard.SSAE_18.value,
        "company_threshold": "Large accelerated filers (>700M USD public float)",
        "notes": "Limited from 2026, reasonable from 2029 for LAFs; accelerated filers from 2027",
    },
    "CALIFORNIA_SB253": {
        "jurisdiction_name": "California - SB 253 Climate Corporate Data Accountability",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2030-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "effective_date": "2026-01-01",
        "standard": AssuranceStandard.ISO_14064_3.value,
        "company_threshold": "Entities with >1B USD annual revenue doing business in CA",
        "notes": "Limited for S1+S2 from 2026; S3 from 2027; reasonable from 2030",
    },
    "UK_SECR": {
        "jurisdiction_name": "United Kingdom - SECR",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": None,
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2019-04-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Quoted companies, large unquoted, and LLPs",
        "notes": "Voluntary assurance recommended; mandatory for premium listed companies",
    },
    "SINGAPORE_SGX": {
        "jurisdiction_name": "Singapore - SGX Climate Reporting",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2027-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2025-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "All issuers listed on SGX by industry group",
        "notes": "Phased by industry; financial sector from 2025, others from 2026",
    },
    "JAPAN_SSBJ": {
        "jurisdiction_name": "Japan - SSBJ Sustainability Standards",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2028-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "effective_date": "2027-04-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Prime Market listed companies (>3T JPY market cap initially)",
        "notes": "SSBJ standards aligned with ISSB; phased from FY2027",
    },
    "AUSTRALIA_ASRS": {
        "jurisdiction_name": "Australia - ASRS Climate-related Financial Disclosure",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2030-07-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "effective_date": "2025-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Group 1: >500M AUD revenue; Group 2: >200M; Group 3: >50M",
        "notes": "Three-group phase-in from 2025 (Group 1) through 2027 (Group 3)",
    },
    "SOUTH_KOREA_KSQF": {
        "jurisdiction_name": "South Korea - KSQF Sustainability Disclosure",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2028-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2026-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "KOSPI-listed companies with >2T KRW assets",
        "notes": "Phased from 2026; all KOSPI companies by 2028",
    },
    "HONG_KONG_HKEX": {
        "jurisdiction_name": "Hong Kong - HKEX Climate Disclosure",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2028-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2025-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Main Board listed issuers with >HKD 8B market cap",
        "notes": "Aligned with ISSB S2; phased from 2025",
    },
    "BRAZIL_CVM": {
        "jurisdiction_name": "Brazil - CVM Sustainability Reporting",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2029-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2026-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Publicly traded companies on B3",
        "notes": "CVM Resolution 193; limited assurance from 2026",
    },
    "INDIA_BRSR": {
        "jurisdiction_name": "India - BRSR Core / SEBI",
        "assurance_level": AssuranceLevel.REASONABLE.value,
        "reasonable_assurance_from": "2026-04-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "effective_date": "2023-04-01",
        "standard": AssuranceStandard.ISAE_3000.value,
        "company_threshold": "Top 1000 listed companies by market capitalisation",
        "notes": "BRSR Core mandatory with reasonable assurance for top 150 from FY2024-25",
    },
    "CANADA_CSSB": {
        "jurisdiction_name": "Canada - CSSB Sustainability Disclosure Standards",
        "assurance_level": AssuranceLevel.LIMITED.value,
        "reasonable_assurance_from": "2029-01-01",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "effective_date": "2027-01-01",
        "standard": AssuranceStandard.ISAE_3410.value,
        "company_threshold": "Reporting issuers (securities regulators)",
        "notes": "CSSB S1/S2 based on ISSB; phased introduction from 2027",
    },
}

ISAE_3410_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "engagement_acceptance": {
        "category_name": "Engagement Acceptance and Continuance",
        "weight": Decimal("0.10"),
        "item_count": 8,
        "description": "Preconditions, independence, competence, engagement terms",
    },
    "planning": {
        "category_name": "Planning the Engagement",
        "weight": Decimal("0.12"),
        "item_count": 12,
        "description": "Understanding the entity, materiality, risk assessment, audit plan",
    },
    "risk_assessment": {
        "category_name": "Risk Assessment Procedures",
        "weight": Decimal("0.12"),
        "item_count": 10,
        "description": "Identify and assess risks of material misstatement",
    },
    "responses_to_risk": {
        "category_name": "Responses to Assessed Risks",
        "weight": Decimal("0.12"),
        "item_count": 14,
        "description": "Design and perform procedures responsive to assessed risks",
    },
    "evidence": {
        "category_name": "Obtaining Evidence",
        "weight": Decimal("0.10"),
        "item_count": 10,
        "description": "Sufficiency and appropriateness of evidence obtained",
    },
    "using_work_of_others": {
        "category_name": "Using the Work of Others",
        "weight": Decimal("0.08"),
        "item_count": 6,
        "description": "Use of work of experts, internal auditors, other practitioners",
    },
    "evaluating_misstatements": {
        "category_name": "Evaluating Misstatements",
        "weight": Decimal("0.10"),
        "item_count": 8,
        "description": "Evaluate identified misstatements and unadjusted misstatements",
    },
    "forming_conclusion": {
        "category_name": "Forming the Assurance Conclusion",
        "weight": Decimal("0.10"),
        "item_count": 10,
        "description": "Evaluate sufficiency of evidence, form conclusion, modified conclusions",
    },
    "reporting": {
        "category_name": "Assurance Report Content",
        "weight": Decimal("0.08"),
        "item_count": 12,
        "description": "Report elements, conclusion wording, emphasis of matter, other matter",
    },
    "documentation": {
        "category_name": "Documentation",
        "weight": Decimal("0.08"),
        "item_count": 10,
        "description": "Engagement file, timely completion, retention, access controls",
    },
}

COST_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "base_costs_by_size": {
        "MICRO": {
            "limited_eur": Decimal("8000"),
            "reasonable_eur": Decimal("18000"),
            "description": "Micro entities (<10 employees, <2M EUR revenue)",
        },
        "SMALL": {
            "limited_eur": Decimal("15000"),
            "reasonable_eur": Decimal("35000"),
            "description": "Small entities (10-250 employees, 2-50M EUR revenue)",
        },
        "MEDIUM": {
            "limited_eur": Decimal("30000"),
            "reasonable_eur": Decimal("70000"),
            "description": "Medium entities (250-2500 employees, 50-500M EUR revenue)",
        },
        "LARGE": {
            "limited_eur": Decimal("60000"),
            "reasonable_eur": Decimal("150000"),
            "description": "Large entities (2500-25000 employees, 500M-5B EUR revenue)",
        },
        "LARGE_ACCELERATED_FILER": {
            "limited_eur": Decimal("100000"),
            "reasonable_eur": Decimal("250000"),
            "description": "Large accelerated filers (>700M USD public float)",
        },
        "ACCELERATED_FILER": {
            "limited_eur": Decimal("75000"),
            "reasonable_eur": Decimal("180000"),
            "description": "Accelerated filers (75-700M USD public float)",
        },
        "NON_ACCELERATED_FILER": {
            "limited_eur": Decimal("40000"),
            "reasonable_eur": Decimal("95000"),
            "description": "Non-accelerated filers (<75M USD public float)",
        },
    },
    "multipliers": {
        "scope_3_included": Decimal("1.40"),
        "multi_jurisdiction": Decimal("1.25"),
        "first_time_engagement": Decimal("1.30"),
        "reasonable_vs_limited": Decimal("2.20"),
        "complex_operations": Decimal("1.35"),
        "multiple_facilities": Decimal("1.20"),
        "multi_currency": Decimal("1.10"),
        "xbrl_tagging": Decimal("1.15"),
        "expedited_timeline": Decimal("1.25"),
    },
    "hourly_rates": {
        "partner_eur": Decimal("450"),
        "senior_manager_eur": Decimal("320"),
        "manager_eur": Decimal("250"),
        "senior_associate_eur": Decimal("180"),
        "associate_eur": Decimal("130"),
    },
}

MATERIALITY_DEFAULTS: Dict[str, Decimal] = {
    "overall_pct": Decimal("5.0"),
    "performance_pct": Decimal("65.0"),
    "trivial_pct": Decimal("5.0"),
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_general": (
        "General corporate assurance prep with ISAE 3410 limited assurance, "
        "standard control testing, and single-jurisdiction CSRD compliance"
    ),
    "csrd_limited": (
        "EU CSRD limited assurance configuration for large undertakings "
        "with ESRS E1 alignment, standard materiality, and evidence mapping"
    ),
    "csrd_reasonable": (
        "EU CSRD reasonable assurance configuration (2028+) with enhanced "
        "control testing, expanded sampling, and full evidence requirements"
    ),
    "sec_attestation": (
        "US SEC climate disclosure attestation for large accelerated filers "
        "with SSAE 18 alignment and PCAOB-registered verifier requirements"
    ),
    "california_sb253": (
        "California SB 253 compliance with Scope 1+2+3 assurance prep, "
        "ISO 14064-3 standard, and California-specific reporting thresholds"
    ),
    "multi_jurisdiction": (
        "Multi-jurisdiction assurance prep covering EU CSRD, US SEC, UK SECR, "
        "and 2+ additional jurisdictions with consolidated evidence packages"
    ),
    "financial_services": (
        "Financial services assurance prep with PCAF alignment, financed "
        "emissions evidence, portfolio-level materiality, and SFDR compliance"
    ),
    "first_time_assurance": (
        "First-time assurance engagement prep with extended readiness "
        "assessment, remediation planning, and verifier onboarding support"
    ),
}

# =============================================================================
# Sub-Config Models (15+ Pydantic v2 models)
# =============================================================================

class EvidenceConfig(BaseModel):
    """Configuration for evidence consolidation and management."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    categories: List[EvidenceCategory] = Field(
        default_factory=lambda: [
            EvidenceCategory.SOURCE_DATA,
            EvidenceCategory.EMISSION_FACTOR,
            EvidenceCategory.CALCULATION,
            EvidenceCategory.ASSUMPTION,
            EvidenceCategory.METHODOLOGY,
            EvidenceCategory.BOUNDARY,
            EvidenceCategory.COMPLETENESS,
            EvidenceCategory.CONTROL,
            EvidenceCategory.APPROVAL,
            EvidenceCategory.EXTERNAL,
        ],
        description="Evidence categories to collect and organise",
    )
    minimum_quality: EvidenceQuality = Field(
        EvidenceQuality.ADEQUATE,
        description="Minimum acceptable quality rating for evidence items",
    )
    require_provenance_hash: bool = Field(
        True,
        description="Require SHA-256 provenance hash on all evidence items",
    )
    retention_years: int = Field(
        7, ge=3, le=15,
        description="Number of years to retain evidence (ISAE 3410 minimum: 5)",
    )
    auto_extract_from_packs: bool = Field(
        True,
        description="Automatically extract evidence from PACK-041 through PACK-047",
    )
    evidence_format: str = Field(
        "STRUCTURED",
        description="Evidence packaging format (STRUCTURED, FLAT, CUSTOM)",
    )
    max_evidence_size_mb: int = Field(
        500, ge=50, le=5000,
        description="Maximum total evidence package size in megabytes",
    )

    @field_validator("evidence_format")
    @classmethod
    def validate_evidence_format(cls, v: str) -> str:
        """Validate evidence format value."""
        allowed = {"STRUCTURED", "FLAT", "CUSTOM"}
        if v.upper() not in allowed:
            raise ValueError(f"evidence_format must be one of {allowed}, got '{v}'")
        return v.upper()

class ReadinessConfig(BaseModel):
    """Configuration for ISAE 3410 readiness assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    standard: AssuranceStandard = Field(
        AssuranceStandard.ISAE_3410,
        description="Assurance standard for readiness assessment",
    )
    target_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Target assurance level to prepare for",
    )
    minimum_readiness_score_pct: Decimal = Field(
        Decimal("70.0"), ge=Decimal("0"), le=Decimal("100"),
        description="Minimum readiness score percentage to pass assessment",
    )
    include_gap_remediation: bool = Field(
        True,
        description="Generate remediation plan for identified gaps",
    )
    assessment_categories: List[str] = Field(
        default_factory=lambda: list(ISAE_3410_CATEGORIES.keys()),
        description="ISAE 3410 categories to assess",
    )
    auto_score_from_evidence: bool = Field(
        True,
        description="Automatically score readiness based on available evidence",
    )
    remediation_priority_threshold: Decimal = Field(
        Decimal("50.0"),
        description="Score threshold below which items are flagged as high priority",
    )

class ProvenanceConfig(BaseModel):
    """Configuration for calculation provenance verification."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hash_algorithm: str = Field(
        "SHA-256",
        description="Hash algorithm for provenance chain (SHA-256 required)",
    )
    verify_all_calculations: bool = Field(
        True,
        description="Verify provenance for all calculation outputs",
    )
    replay_tolerance_pct: Decimal = Field(
        Decimal("0.01"),
        description="Tolerance for deterministic replay (rounding differences)",
    )
    include_input_hashes: bool = Field(
        True,
        description="Include input data hashes in provenance chain",
    )
    include_intermediate_hashes: bool = Field(
        True,
        description="Include intermediate calculation step hashes",
    )
    chain_depth_limit: int = Field(
        100, ge=10, le=1000,
        description="Maximum chain depth for provenance verification",
    )
    generate_certificate: bool = Field(
        True,
        description="Generate signed provenance certificate for verifier",
    )

    @field_validator("hash_algorithm")
    @classmethod
    def validate_hash_algorithm(cls, v: str) -> str:
        """Validate hash algorithm is SHA-256."""
        if v.upper().replace("-", "") != "SHA256":
            raise ValueError("Only SHA-256 is supported for provenance hashing")
        return "SHA-256"

class ControlConfig(BaseModel):
    """Configuration for internal control testing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    control_categories: List[ControlCategory] = Field(
        default_factory=lambda: [
            ControlCategory.DATA_COLLECTION,
            ControlCategory.CALCULATION,
            ControlCategory.REVIEW,
            ControlCategory.REPORTING,
            ControlCategory.IT_GENERAL,
        ],
        description="Control categories to test",
    )
    minimum_effectiveness: ControlEffectiveness = Field(
        ControlEffectiveness.PARTIALLY_EFFECTIVE,
        description="Minimum acceptable control effectiveness",
    )
    target_maturity: ControlMaturity = Field(
        ControlMaturity.LEVEL_3_DEFINED,
        description="Target control maturity level",
    )
    test_sample_size: int = Field(
        25, ge=5, le=100,
        description="Number of samples to test per control",
    )
    include_it_general_controls: bool = Field(
        True,
        description="Include IT general controls in testing scope",
    )
    walkthroughs_required: bool = Field(
        True,
        description="Require control walkthroughs in addition to testing",
    )
    remediation_timeline_days: int = Field(
        90, ge=30, le=365,
        description="Maximum days allowed for control remediation",
    )

class VerifierConfig(BaseModel):
    """Configuration for verifier collaboration management."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_query_response_days: int = Field(
        5, ge=1, le=30,
        description="Maximum days to respond to verifier queries",
    )
    query_escalation_days: int = Field(
        3, ge=1, le=14,
        description="Days before a query is escalated to management",
    )
    finding_response_days: int = Field(
        10, ge=3, le=60,
        description="Days to respond to verifier findings",
    )
    enable_collaboration_portal: bool = Field(
        True,
        description="Enable online collaboration portal for verifier access",
    )
    auto_evidence_sharing: bool = Field(
        True,
        description="Automatically share relevant evidence when queries are raised",
    )
    track_verifier_hours: bool = Field(
        True,
        description="Track verifier hours for cost management",
    )
    engagement_phases: List[EngagementPhase] = Field(
        default_factory=lambda: [
            EngagementPhase.PLANNING,
            EngagementPhase.RISK_ASSESSMENT,
            EngagementPhase.FIELDWORK,
            EngagementPhase.REPORTING,
            EngagementPhase.CLOSEOUT,
        ],
        description="Engagement phases to track",
    )

class MaterialityConfig(BaseModel):
    """Configuration for materiality assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    overall_materiality_pct: Decimal = Field(
        Decimal("5.0"), ge=Decimal("1.0"), le=Decimal("10.0"),
        description="Overall materiality as percentage of total GHG emissions",
    )
    performance_materiality_pct: Decimal = Field(
        Decimal("65.0"), ge=Decimal("50.0"), le=Decimal("80.0"),
        description="Performance materiality as percentage of overall materiality",
    )
    clearly_trivial_pct: Decimal = Field(
        Decimal("5.0"), ge=Decimal("1.0"), le=Decimal("10.0"),
        description="Clearly trivial threshold as percentage of overall materiality",
    )
    scope_specific_enabled: bool = Field(
        True,
        description="Calculate scope-specific materiality thresholds",
    )
    scope_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "SCOPE_1": Decimal("0.35"),
            "SCOPE_2": Decimal("0.30"),
            "SCOPE_3": Decimal("0.35"),
        },
        description="Weights for scope-specific materiality calculation",
    )
    revision_frequency: str = Field(
        "QUARTERLY",
        description="How often to reassess materiality (ANNUAL, QUARTERLY, MONTHLY)",
    )

    @field_validator("revision_frequency")
    @classmethod
    def validate_revision_frequency(cls, v: str) -> str:
        """Validate revision frequency value."""
        allowed = {"ANNUAL", "QUARTERLY", "MONTHLY"}
        if v.upper() not in allowed:
            raise ValueError(f"revision_frequency must be one of {allowed}, got '{v}'")
        return v.upper()

class SamplingConfig(BaseModel):
    """Configuration for statistical sampling plan generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary_method: SamplingMethod = Field(
        SamplingMethod.MUS,
        description="Primary sampling method for assurance testing",
    )
    confidence_level_pct: Decimal = Field(
        Decimal("95.0"), ge=Decimal("80.0"), le=Decimal("99.9"),
        description="Statistical confidence level for sampling",
    )
    tolerable_misstatement_pct: Decimal = Field(
        Decimal("5.0"), ge=Decimal("1.0"), le=Decimal("15.0"),
        description="Tolerable misstatement as percentage of population",
    )
    expected_misstatement_pct: Decimal = Field(
        Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("10.0"),
        description="Expected misstatement rate for sample size calculation",
    )
    minimum_sample_size: int = Field(
        25, ge=10, le=200,
        description="Minimum sample size regardless of statistical calculation",
    )
    stratification_enabled: bool = Field(
        True,
        description="Enable stratified sampling by emission category",
    )
    stratification_criteria: List[str] = Field(
        default_factory=lambda: ["scope", "category", "facility"],
        description="Criteria for stratification of the population",
    )
    top_stratum_coverage_pct: Decimal = Field(
        Decimal("80.0"),
        description="Percentage coverage target for the top stratum",
    )

class RegulatoryConfig(BaseModel):
    """Configuration for multi-jurisdiction regulatory requirement mapping."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    jurisdictions: List[Jurisdiction] = Field(
        default_factory=lambda: [Jurisdiction.EU_CSRD],
        description="Regulatory jurisdictions to map requirements for",
    )
    company_size: CompanySize = Field(
        CompanySize.LARGE,
        description="Company size classification for threshold mapping",
    )
    reporting_year: int = Field(
        2026, ge=2024, le=2035,
        description="Current reporting year for effective date checks",
    )
    include_upcoming_requirements: bool = Field(
        True,
        description="Include requirements not yet effective but within 2 years",
    )
    track_effective_dates: bool = Field(
        True,
        description="Track and alert on approaching effective dates",
    )
    deadline_warning_days: int = Field(
        90, ge=30, le=365,
        description="Days before a regulatory deadline to raise warning",
    )

class CostTimelineConfig(BaseModel):
    """Configuration for engagement cost and timeline estimation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_size: CompanySize = Field(
        CompanySize.LARGE,
        description="Company size for base cost lookup",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Assurance level for cost estimation",
    )
    include_scope_3: bool = Field(
        False,
        description="Include Scope 3 in assurance scope (increases cost)",
    )
    multi_jurisdiction: bool = Field(
        False,
        description="Multi-jurisdiction engagement (increases cost)",
    )
    first_time_engagement: bool = Field(
        False,
        description="First-time assurance engagement (increases cost)",
    )
    complex_operations: bool = Field(
        False,
        description="Complex operations or multiple facilities (increases cost)",
    )
    expedited_timeline: bool = Field(
        False,
        description="Expedited timeline requested (increases cost)",
    )
    target_completion_weeks: int = Field(
        12, ge=4, le=52,
        description="Target engagement completion in weeks",
    )
    include_cost_breakdown: bool = Field(
        True,
        description="Include detailed cost breakdown by phase and role",
    )
    currency: str = Field(
        "EUR",
        description="Currency for cost estimates (ISO 4217)",
    )

class ReportingConfig(BaseModel):
    """Configuration for assurance report generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.HTML, ReportFormat.JSON],
        description="Output format(s) for generated reports",
    )
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "readiness_assessment",
            "evidence_summary",
            "control_testing",
            "materiality_sampling",
            "provenance_verification",
            "regulatory_compliance",
            "verifier_collaboration",
            "cost_timeline",
            "methodology",
        ],
        description="Report sections to include",
    )
    branding: Dict[str, str] = Field(
        default_factory=lambda: {
            "logo_url": "",
            "primary_colour": "#1B5E20",
            "company_name": "",
        },
        description="Report branding configuration",
    )
    language: str = Field("en", description="Report language (ISO 639-1)")
    include_charts: bool = Field(True, description="Include charts and visualisations")
    include_data_tables: bool = Field(True, description="Include detailed data tables")
    include_appendices: bool = Field(True, description="Include technical appendices")
    decimal_places_display: int = Field(
        2, ge=0, le=6,
        description="Decimal places for display in reports",
    )

class AlertConfig(BaseModel):
    """Configuration for assurance monitoring and alerting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_types: List[str] = Field(
        default_factory=lambda: [
            "READINESS_GAP",
            "CONTROL_FAILURE",
            "EVIDENCE_MISSING",
            "QUERY_OVERDUE",
            "DEADLINE_APPROACHING",
            "PROVENANCE_MISMATCH",
        ],
        description="Types of alerts to enable",
    )
    channels: List[str] = Field(
        default_factory=lambda: ["EMAIL"],
        description="Notification delivery channels (EMAIL, SLACK, TEAMS, WEBHOOK)",
    )
    readiness_drop_threshold_pct: Decimal = Field(
        Decimal("5.0"),
        description="Readiness score drop percentage that triggers alert",
    )
    query_overdue_days: int = Field(
        3, ge=1, le=14,
        description="Days before a verifier query is considered overdue",
    )
    evidence_gap_alert: bool = Field(
        True,
        description="Alert when evidence gaps are detected in required categories",
    )
    daily_digest: bool = Field(
        False,
        description="Send daily digest of all assurance prep alerts",
    )

class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_calculation_time_seconds: int = Field(
        300, ge=30, le=3600,
        description="Maximum allowed calculation time in seconds",
    )
    cache_results: bool = Field(
        True, description="Cache assurance assessment results",
    )
    parallel_control_testing: bool = Field(
        True, description="Test controls in parallel where possible",
    )
    batch_size: int = Field(
        500, ge=50, le=5000,
        description="Batch size for bulk evidence processing",
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, le=86400,
        description="Cache TTL in seconds",
    )
    lazy_load_evidence: bool = Field(
        True, description="Lazy-load evidence packages only when needed",
    )

class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rbac_enabled: bool = Field(True, description="Enable role-based access control")
    audit_trail_enabled: bool = Field(True, description="Enable audit trail for all operations")
    encryption_at_rest: bool = Field(True, description="Encrypt assurance data at rest (AES-256)")
    verifier_access_controls: bool = Field(
        True,
        description="Enable granular access controls for external verifiers",
    )
    roles: List[str] = Field(
        default_factory=lambda: [
            "assurance_lead", "assurance_manager", "evidence_collector",
            "control_tester", "reviewer", "approver", "verifier_liaison",
            "verifier_readonly", "data_admin", "viewer", "admin",
        ],
        description="Available RBAC roles for assurance management",
    )

class EngagementConfig(BaseModel):
    """Configuration for the overall assurance engagement."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engagement_name: str = Field(
        "",
        description="Name of the assurance engagement",
    )
    verifier_firm: str = Field(
        "",
        description="Name of the appointed verifier / assurance provider",
    )
    verifier_lead: str = Field(
        "",
        description="Name of the lead verifier / engagement partner",
    )
    engagement_start_date: Optional[str] = Field(
        None,
        description="Planned engagement start date (ISO 8601)",
    )
    engagement_end_date: Optional[str] = Field(
        None,
        description="Planned engagement end date (ISO 8601)",
    )
    fieldwork_start_date: Optional[str] = Field(
        None,
        description="Planned fieldwork start date (ISO 8601)",
    )
    fieldwork_end_date: Optional[str] = Field(
        None,
        description="Planned fieldwork end date (ISO 8601)",
    )
    report_issuance_date: Optional[str] = Field(
        None,
        description="Target date for assurance report issuance (ISO 8601)",
    )

# =============================================================================
# Main Configuration Model
# =============================================================================

class AssurancePackConfig(BaseModel):
    """
    Top-level configuration for PACK-048 GHG Assurance Prep.

    Combines all sub-configurations required for evidence consolidation,
    readiness assessment, calculation provenance, control testing, verifier
    collaboration, materiality assessment, sampling plan generation,
    regulatory requirement mapping, cost and timeline estimation, and
    assurance reporting.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str = Field("", description="Reporting company legal name")
    assurance_standard: AssuranceStandard = Field(
        AssuranceStandard.ISAE_3410,
        description="Primary assurance standard for the engagement",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Target assurance level",
    )
    company_size: CompanySize = Field(
        CompanySize.LARGE,
        description="Company size classification",
    )
    country: str = Field("DE", description="Primary country (ISO 3166-1 alpha-2)")
    reporting_year: int = Field(2026, ge=2020, le=2035, description="Current reporting year")
    base_year: int = Field(2020, ge=2015, le=2030, description="Base year for GHG inventory")
    total_emissions_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Total GHG emissions in tCO2e (for materiality calculation)",
    )
    scopes_in_scope: List[str] = Field(
        default_factory=lambda: ["SCOPE_1", "SCOPE_2"],
        description="Scopes included in assurance scope",
    )

    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    readiness: ReadinessConfig = Field(default_factory=ReadinessConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    controls: ControlConfig = Field(default_factory=ControlConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    materiality: MaterialityConfig = Field(default_factory=MaterialityConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    regulatory: RegulatoryConfig = Field(default_factory=RegulatoryConfig)
    cost_timeline: CostTimelineConfig = Field(default_factory=CostTimelineConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    engagement: EngagementConfig = Field(default_factory=EngagementConfig)

    @model_validator(mode="after")
    def validate_base_year_consistency(self) -> AssurancePackConfig:
        """Ensure base year is before reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_assurance_level_standard(self) -> AssurancePackConfig:
        """Validate assurance level is compatible with the standard."""
        if self.assurance_standard == AssuranceStandard.SSAE_18:
            if self.assurance_level not in (AssuranceLevel.EXAMINATION, AssuranceLevel.REVIEW):
                logger.warning(
                    "SSAE 18 typically uses EXAMINATION or REVIEW levels, "
                    "not %s. Consider adjusting.",
                    self.assurance_level.value,
                )
        return self

    @model_validator(mode="after")
    def validate_scope_3_consistency(self) -> AssurancePackConfig:
        """Warn if Scope 3 is in scope but cost model excludes it."""
        if "SCOPE_3" in self.scopes_in_scope:
            if not self.cost_timeline.include_scope_3:
                logger.warning(
                    "Scope 3 is in assurance scope but cost_timeline.include_scope_3 "
                    "is False. Cost estimate may be understated."
                )
        return self

    @model_validator(mode="after")
    def validate_materiality_total_emissions(self) -> AssurancePackConfig:
        """Warn if total emissions needed for materiality but not provided."""
        if self.total_emissions_tco2e is None:
            logger.warning(
                "total_emissions_tco2e is not set. Materiality thresholds "
                "will be calculated as percentages only (no absolute values)."
            )
        return self

    @model_validator(mode="after")
    def validate_jurisdiction_alignment(self) -> AssurancePackConfig:
        """Warn if regulatory jurisdictions imply multi-jurisdiction but flag is off."""
        if len(self.regulatory.jurisdictions) > 1:
            if not self.cost_timeline.multi_jurisdiction:
                logger.warning(
                    "Multiple jurisdictions configured (%d) but "
                    "cost_timeline.multi_jurisdiction is False. "
                    "Cost estimate may be understated.",
                    len(self.regulatory.jurisdictions),
                )
        return self

# =============================================================================
# Pack Configuration Wrapper
# =============================================================================

class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-048 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pack: AssurancePackConfig = Field(default_factory=AssurancePackConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-048-assurance-prep", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
        """
        Load configuration from a named engagement preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'corporate_general').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialised PackConfig.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)
        pack_config = AssurancePackConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PackConfig:
        """
        Load configuration from an arbitrary YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully initialised PackConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = AssurancePackConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: PackConfig, overrides: Dict[str, Any]) -> PackConfig:
        """
        Create a new PackConfig by merging overrides into a base config.

        Args:
            base: Existing PackConfig to use as the base.
            overrides: Dict of overrides (supports nested keys).

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = AssurancePackConfig(**merged)
        return cls(pack=pack_config, preset_name=base.preset_name, config_version=base.config_version)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with ASSURANCE_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: ASSURANCE_PACK_CONTROLS__TEST_SAMPLE_SIZE=50
        """
        overrides: Dict[str, Any] = {}
        prefix = "ASSURANCE_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """
        Compute SHA-256 hash of the full configuration.

        Returns:
            Hex-encoded SHA-256 hash string for provenance tracking.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_completeness(self) -> List[str]:
        """
        Run domain-specific validation checks on the configuration.

        Returns:
            List of warning messages (empty list means no issues).
        """
        return validate_config(self.pack)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full configuration to a plain dictionary.

        Returns:
            Dict representation of the entire PackConfig.
        """
        return self.model_dump()

# =============================================================================
# Utility Functions
# =============================================================================

def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """
    Convenience function to load a preset configuration.

    Args:
        preset_name: Key from AVAILABLE_PRESETS.
        overrides: Optional dict of overrides.

    Returns:
        Initialised PackConfig from the named preset.
    """
    return PackConfig.from_preset(preset_name, overrides)

def validate_config(config: AssurancePackConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The assurance pack configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    # Company name check
    if not config.company_name:
        warnings.append("No company_name configured.")

    # Total emissions for materiality
    if config.total_emissions_tco2e is None:
        warnings.append(
            "total_emissions_tco2e not set. Materiality cannot calculate "
            "absolute thresholds (only percentages)."
        )

    # Scope 3 in scope but limited assurance
    if "SCOPE_3" in config.scopes_in_scope:
        if config.assurance_level == AssuranceLevel.LIMITED:
            warnings.append(
                "Scope 3 is in assurance scope with limited assurance. "
                "Consider whether this meets stakeholder expectations."
            )

    # Readiness standard alignment
    if config.readiness.standard != config.assurance_standard:
        warnings.append(
            f"Readiness standard ({config.readiness.standard.value}) differs "
            f"from engagement standard ({config.assurance_standard.value}). "
            "Consider aligning for consistency."
        )

    # Readiness vs assurance level alignment
    if config.readiness.target_level != config.assurance_level:
        warnings.append(
            f"Readiness target level ({config.readiness.target_level.value}) "
            f"differs from assurance level ({config.assurance_level.value}). "
            "Consider aligning."
        )

    # Control maturity for reasonable assurance
    if config.assurance_level == AssuranceLevel.REASONABLE:
        if config.controls.target_maturity.value < ControlMaturity.LEVEL_3_DEFINED.value:
            warnings.append(
                "Reasonable assurance typically requires Level 3 (Defined) or higher "
                "control maturity. Current target: " + config.controls.target_maturity.value
            )

    # Multi-jurisdiction cost alignment
    if len(config.regulatory.jurisdictions) > 1:
        if not config.cost_timeline.multi_jurisdiction:
            warnings.append(
                f"Multiple jurisdictions ({len(config.regulatory.jurisdictions)}) "
                "but cost_timeline.multi_jurisdiction is False."
            )

    # First-time engagement indicators
    if config.cost_timeline.first_time_engagement:
        if config.readiness.minimum_readiness_score_pct > Decimal("80.0"):
            warnings.append(
                "First-time engagement with high readiness threshold (>80%). "
                "Consider lowering for initial year."
            )

    # Sampling confidence level
    if config.sampling.confidence_level_pct < Decimal("90.0"):
        warnings.append(
            f"Sampling confidence level ({config.sampling.confidence_level_pct}%) "
            "is below 90%. Standard practice is 90-95%."
        )

    # Evidence retention minimum
    if config.evidence.retention_years < 5:
        warnings.append(
            f"Evidence retention ({config.evidence.retention_years} years) "
            "is below ISAE 3410 recommended minimum of 5 years."
        )

    # Security configuration
    if config.security.audit_trail_enabled and not config.security.rbac_enabled:
        warnings.append(
            "Audit trail is enabled but RBAC is disabled. "
            "Consider enabling RBAC for proper identity tracking."
        )

    # Verifier access controls
    if not config.security.verifier_access_controls:
        warnings.append(
            "Verifier access controls are disabled. External verifiers "
            "may have unrestricted access to assurance data."
        )

    # Engagement details
    if not config.engagement.verifier_firm:
        warnings.append("No verifier_firm configured in engagement settings.")

    return warnings

def get_default_config(
    standard: AssuranceStandard = AssuranceStandard.ISAE_3410,
) -> AssurancePackConfig:
    """
    Create a default configuration for the given assurance standard.

    Args:
        standard: Assurance standard for the engagement.

    Returns:
        Default AssurancePackConfig for the standard.
    """
    return AssurancePackConfig(assurance_standard=standard)

def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()

def get_standard_controls() -> Dict[str, Dict[str, Any]]:
    """
    Return all 25 standard controls with metadata.

    Returns:
        Dict mapping control ID to control metadata.
    """
    return STANDARD_CONTROLS.copy()

def get_jurisdiction_requirements(
    jurisdiction: str,
) -> Optional[Dict[str, Any]]:
    """
    Return jurisdiction requirements for a given jurisdiction code.

    Args:
        jurisdiction: Jurisdiction code from JURISDICTION_REQUIREMENTS.

    Returns:
        Dict of jurisdiction data, or None if not found.
    """
    return JURISDICTION_REQUIREMENTS.get(jurisdiction)

def get_isae3410_categories() -> Dict[str, Dict[str, Any]]:
    """
    Return ISAE 3410 assessment categories with weights and item counts.

    Returns:
        Dict mapping category key to category metadata.
    """
    return ISAE_3410_CATEGORIES.copy()

def get_cost_estimate(
    company_size: str,
    assurance_level: str,
) -> Optional[Decimal]:
    """
    Return base cost estimate for a company size and assurance level.

    Args:
        company_size: CompanySize value string.
        assurance_level: 'limited' or 'reasonable'.

    Returns:
        Base cost in EUR, or None if size not found.
    """
    size_data = COST_MODEL_PARAMS["base_costs_by_size"].get(company_size)
    if size_data is None:
        return None
    level_key = f"{assurance_level.lower()}_eur"
    return size_data.get(level_key)

def get_materiality_defaults() -> Dict[str, Decimal]:
    """
    Return default materiality thresholds.

    Returns:
        Dict with overall_pct, performance_pct, and trivial_pct.
    """
    return MATERIALITY_DEFAULTS.copy()
