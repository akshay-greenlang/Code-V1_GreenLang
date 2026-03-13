# -*- coding: utf-8 -*-
"""
Fraud Detection Rules Reference Data - AGENT-EUDR-012 Document Authentication

Defines the 15 deterministic fraud detection rules (FRD-001 through FRD-015)
that the FraudPatternDetector engine evaluates against every document
submitted through the EUDR Document Authentication pipeline.

Each rule is a pure deterministic check -- no LLM, no ML model, no
probabilistic scoring.  Rules use arithmetic comparison, regex matching,
date arithmetic, hash lookup, statistical pattern detection, and cross-
document consistency checks to flag potential fraud indicators.

Rule Structure:
    - rule_id: Unique rule identifier (FRD-001 through FRD-015)
    - pattern_type: FraudPatternType enum value from models.py
    - severity: FraudSeverity enum value (low, medium, high, critical)
    - description: Human-readable rule description
    - detection_method: Technical description of how the rule detects fraud
    - thresholds: Configurable numeric thresholds for the rule
    - enabled: Whether the rule is active (default True)
    - applicable_document_types: Which document types this rule applies to

Required Documents by Commodity:
    Specifies the minimum document set required for each of the 7 EUDR
    commodities to pass completeness validation (FRD-014).

Lookup helpers:
    get_rule(rule_id) -> dict | None
    get_all_rules() -> list[dict]
    get_rules_for_document_type(doc_type) -> list[dict]
    get_required_documents(commodity) -> list[str]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012) - Appendix B
Agent ID: GL-EUDR-DAV-012
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 20 document types for rule applicability
# ---------------------------------------------------------------------------

_ALL_DOC_TYPES: List[str] = [
    "coo", "pc", "bol", "cde", "cdi",
    "rspo_cert", "fsc_cert", "iscc_cert", "ft_cert", "utz_cert",
    "ltr", "ltd", "fmp", "fc", "wqc",
    "dds_draft", "ssd", "ic", "tc", "wr",
]

_CERT_DOC_TYPES: List[str] = [
    "rspo_cert", "fsc_cert", "iscc_cert", "ft_cert", "utz_cert",
]

_GOV_DOC_TYPES: List[str] = [
    "coo", "pc", "cde", "cdi", "fc", "ltr", "ltd",
]

_TRADE_DOC_TYPES: List[str] = [
    "bol", "ic", "tc", "wr",
]

# ---------------------------------------------------------------------------
# Fraud detection rules (15 rules per PRD Appendix B)
# ---------------------------------------------------------------------------

FRAUD_RULES: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # FRD-001: Duplicate Hash Detection
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-001",
        "pattern_type": "duplicate_reuse",
        "severity": "high",
        "description": (
            "Detects exact duplicate documents by comparing the SHA-256 "
            "hash of the submitted document against the hash registry. "
            "An exact match indicates the document has been submitted "
            "before, which may indicate reuse of a legitimate document "
            "for multiple shipments or an attempted forgery."
        ),
        "detection_method": (
            "Compute SHA-256 hash of the document binary content. "
            "Query the hash registry for an exact match. If found, "
            "retrieve the original submission metadata (document_id, "
            "operator_id, submission_date) and flag as duplicate."
        ),
        "thresholds": {
            "hash_algorithm": "sha256",
            "secondary_hash_algorithm": "sha512",
            "require_dual_match": True,
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-002: Quantity Exceeds 105% of Reference
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-002",
        "pattern_type": "quantity_tampering",
        "severity": "high",
        "description": (
            "Flags documents where the stated quantity exceeds 105% of "
            "the expected quantity based on reference documents in the "
            "same shipment chain. Quantity inflation is a common "
            "indicator of document tampering or creation of phantom "
            "volumes for mass balance credit gaming."
        ),
        "detection_method": (
            "Extract quantity values from the submitted document. "
            "Compare against corresponding quantities in linked "
            "reference documents (e.g. BOL quantity vs COO quantity "
            "vs weighbridge receipt). Flag if ratio > 1.05 "
            "(quantity_tolerance_percent / 100 + 1.0)."
        ),
        "thresholds": {
            "quantity_tolerance_percent": 5.0,
            "min_quantity_kg": 0.001,
            "comparison_fields": [
                "gross_weight", "net_weight", "quantity",
                "volume", "number_of_packages",
            ],
        },
        "enabled": True,
        "applicable_document_types": [
            "coo", "bol", "ic", "wr", "dds_draft",
            "rspo_cert", "fsc_cert", "iscc_cert",
        ],
    },

    # ------------------------------------------------------------------
    # FRD-003: Date Discrepancy >30 Days
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-003",
        "pattern_type": "date_manipulation",
        "severity": "medium",
        "description": (
            "Detects documents where key dates (issue date, inspection "
            "date, validity dates) differ by more than 30 days from "
            "expected dates based on the shipment timeline. Large date "
            "discrepancies may indicate antedating, postdating, or "
            "recycling of expired documents."
        ),
        "detection_method": (
            "Extract all date fields from the document. Compare each "
            "against the expected date range based on linked documents "
            "and the shipment timeline. Flag if any date pair differs "
            "by more than date_tolerance_days."
        ),
        "thresholds": {
            "date_tolerance_days": 30,
            "date_fields": [
                "date_of_issue", "date_of_inspection",
                "valid_from", "valid_until",
                "date_of_shipment", "date_of_arrival",
            ],
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-004: Expired Certificate
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-004",
        "pattern_type": "expired_cert",
        "severity": "high",
        "description": (
            "Flags sustainability certificates (RSPO, FSC, ISCC, "
            "Fairtrade, UTZ/RA) where the certificate validity period "
            "has expired at the time of document submission. Expired "
            "certificates invalidate the deforestation-free claim."
        ),
        "detection_method": (
            "Extract valid_from and valid_until fields from the "
            "certificate. Compare valid_until against the current "
            "date (UTC). If valid_until < current_date, flag as "
            "expired. Also check if valid_from > current_date "
            "(not-yet-valid)."
        ),
        "thresholds": {
            "grace_period_days": 0,
            "check_not_yet_valid": True,
        },
        "enabled": True,
        "applicable_document_types": list(_CERT_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-005: Serial Number Format Anomaly
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-005",
        "pattern_type": "serial_anomaly",
        "severity": "medium",
        "description": (
            "Detects documents with serial numbers that do not match "
            "the expected format for the document type and issuing "
            "country. Format anomalies suggest the document may have "
            "been created outside the legitimate issuance channel."
        ),
        "detection_method": (
            "Look up the expected serial number regex pattern from "
            "the document_templates reference data based on document "
            "type and country. Apply the regex to the extracted serial "
            "number. Flag if the serial number does not match the "
            "expected pattern."
        ),
        "thresholds": {
            "case_sensitive": False,
            "allow_prefix_whitespace": True,
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-006: Unauthorized Issuing Authority
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-006",
        "pattern_type": "issuer_mismatch",
        "severity": "high",
        "description": (
            "Flags documents where the signing certificate issuer or "
            "the stated issuing authority does not match the expected "
            "authorized issuers for the document type. For sustainability "
            "certificates, the signing CA must match the pinned issuers "
            "list. For government documents, the issuer must be a "
            "recognized national authority."
        ),
        "detection_method": (
            "Extract the issuer name from the signing certificate "
            "chain or from the document metadata. Compare against the "
            "pinned_issuers list (from trusted_cas reference data) "
            "and the issuing_authority_patterns (from document_templates). "
            "Flag if no match is found."
        ),
        "thresholds": {
            "fuzzy_match_threshold": 0.85,
            "check_signing_cert": True,
            "check_stated_authority": True,
        },
        "enabled": True,
        "applicable_document_types": (
            list(_CERT_DOC_TYPES) + list(_GOV_DOC_TYPES)
        ),
    },

    # ------------------------------------------------------------------
    # FRD-007: Template Deviation
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-007",
        "pattern_type": "template_forgery",
        "severity": "medium",
        "description": (
            "Detects documents that deviate from the expected template "
            "structure for their document type and country. Deviations "
            "in required fields, header patterns, or layout indicate "
            "potential forgery or unauthorized modification."
        ),
        "detection_method": (
            "Load the expected template from document_templates "
            "reference data. Compare the extracted document structure "
            "against the template: check for missing required_fields, "
            "absent key_indicators, and structural anomalies. Flag if "
            "the deviation score exceeds the threshold."
        ),
        "thresholds": {
            "max_missing_fields_pct": 20.0,
            "min_keyword_match_pct": 60.0,
            "structural_deviation_threshold": 0.30,
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-008: Cross-Document Quantity Inconsistency >5%
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-008",
        "pattern_type": "cross_doc_inconsistency",
        "severity": "high",
        "description": (
            "Flags shipment document sets where quantities across "
            "linked documents (COO, BOL, invoice, weighbridge receipt) "
            "differ by more than 5%. Cross-document inconsistency "
            "suggests selective tampering of individual documents."
        ),
        "detection_method": (
            "Group documents by shipment_id or dds_reference. "
            "Extract quantity fields from each document. For each "
            "quantity field, compute the coefficient of variation "
            "across the document set. Flag if any field's max-to-min "
            "ratio exceeds (1 + cross_doc_tolerance_percent / 100)."
        ),
        "thresholds": {
            "cross_doc_tolerance_percent": 5.0,
            "min_documents_for_comparison": 2,
            "quantity_fields": [
                "gross_weight", "net_weight", "quantity",
            ],
        },
        "enabled": True,
        "applicable_document_types": [
            "coo", "bol", "ic", "wr", "dds_draft",
        ],
    },

    # ------------------------------------------------------------------
    # FRD-009: Geographic Impossibility
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-009",
        "pattern_type": "geo_impossibility",
        "severity": "critical",
        "description": (
            "Detects documents where the stated country of origin or "
            "production location is geographically impossible given "
            "the commodity type, the shipping route, or the issuing "
            "authority's jurisdiction. Geographic impossibility is a "
            "strong indicator of document fraud or commodity laundering."
        ),
        "detection_method": (
            "Extract country_of_origin, port_of_loading, production "
            "coordinates, and commodity type. Cross-reference against "
            "known production regions for the commodity. Check that "
            "the issuing authority's country matches the stated origin. "
            "Verify shipping route plausibility. Flag if any geographic "
            "constraint is violated."
        ),
        "thresholds": {
            "check_commodity_origin": True,
            "check_issuer_jurisdiction": True,
            "check_shipping_route": True,
            "max_distance_km_port_to_origin": 5000,
        },
        "enabled": True,
        "applicable_document_types": [
            "coo", "pc", "bol", "dds_draft", "ssd",
        ],
    },

    # ------------------------------------------------------------------
    # FRD-010: Document Velocity >10 per Day
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-010",
        "pattern_type": "velocity_anomaly",
        "severity": "medium",
        "description": (
            "Flags issuers or operators that submit more than 10 "
            "documents per day, which exceeds normal operational "
            "velocity. High submission velocity may indicate automated "
            "document generation or bulk fraudulent submissions."
        ),
        "detection_method": (
            "Count the number of documents submitted by the same "
            "operator_id or issuer within a rolling 24-hour window. "
            "Flag if the count exceeds velocity_threshold_per_day. "
            "The time window is computed as [now - 24h, now]."
        ),
        "thresholds": {
            "velocity_threshold_per_day": 10,
            "window_hours": 24,
            "group_by": ["operator_id", "issuer_name"],
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-011: Modification Date After Issue Date
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-011",
        "pattern_type": "modification_anomaly",
        "severity": "medium",
        "description": (
            "Detects documents where the PDF metadata modification "
            "date is later than the stated issue date. This indicates "
            "post-issuance tampering of the document file, which "
            "compromises the document's integrity."
        ),
        "detection_method": (
            "Extract the metadata modification_date from the document "
            "file properties. Extract the stated date_of_issue from "
            "the document content. Compare: flag if modification_date "
            "> date_of_issue + tolerance_days."
        ),
        "thresholds": {
            "tolerance_days": 1,
            "check_creation_date": True,
            "check_modification_date": True,
        },
        "enabled": True,
        "applicable_document_types": list(_ALL_DOC_TYPES),
    },

    # ------------------------------------------------------------------
    # FRD-012: Round Number Bias >80%
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-012",
        "pattern_type": "round_number_bias",
        "severity": "low",
        "description": (
            "Flags documents where more than 80% of numeric values "
            "are round numbers (ending in 0 or 00). In genuine trade "
            "documents, quantities and weights rarely cluster at round "
            "numbers. A high proportion of round values suggests the "
            "document was manually fabricated rather than generated "
            "from actual measurements."
        ),
        "detection_method": (
            "Extract all numeric values from the document. Count "
            "values that are exact multiples of 10 (single-round) "
            "or 100 (double-round). Compute the percentage of round "
            "values. Flag if the percentage exceeds "
            "round_number_threshold_percent."
        ),
        "thresholds": {
            "round_number_threshold_percent": 80.0,
            "min_numeric_values": 3,
            "round_divisors": [10, 100, 1000],
        },
        "enabled": True,
        "applicable_document_types": [
            "coo", "bol", "ic", "wr", "dds_draft",
        ],
    },

    # ------------------------------------------------------------------
    # FRD-013: Identical Text Blocks (Copy-Paste Detection)
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-013",
        "pattern_type": "copy_paste",
        "severity": "medium",
        "description": (
            "Detects documents that share unusually large identical "
            "text blocks with other documents from different shipments "
            "or operators. Copy-paste patterns suggest template-based "
            "fraud where a legitimate document is duplicated and only "
            "key fields are modified."
        ),
        "detection_method": (
            "Extract text content from the document. Compute rolling "
            "hash (Rabin fingerprint or similar) over text blocks of "
            "configurable size. Compare block hashes against the "
            "registry of known document text blocks. Flag if the "
            "similarity ratio exceeds the threshold and the documents "
            "belong to different operators or shipments."
        ),
        "thresholds": {
            "min_block_size_chars": 200,
            "similarity_threshold": 0.85,
            "exclude_standard_boilerplate": True,
            "min_matching_blocks": 3,
        },
        "enabled": True,
        "applicable_document_types": [
            "coo", "dds_draft", "ssd", "pc",
        ],
    },

    # ------------------------------------------------------------------
    # FRD-014: Missing Required Documents
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-014",
        "pattern_type": "missing_required",
        "severity": "high",
        "description": (
            "Flags DDS submissions where the required supporting "
            "document set for the specified commodity is incomplete. "
            "Each EUDR commodity has a minimum set of required "
            "documents that must accompany the due diligence statement."
        ),
        "detection_method": (
            "Look up the required document types for the commodity "
            "from the REQUIRED_DOCUMENTS_BY_COMMODITY mapping. "
            "Compare against the actually submitted document types "
            "in the DDS package. Flag any missing document types."
        ),
        "thresholds": {
            "allow_substitutions": True,
            "substitution_pairs": {
                "coo": ["ssd"],
                "pc": [],
                "ltr": ["ltd"],
            },
        },
        "enabled": True,
        "applicable_document_types": ["dds_draft"],
    },

    # ------------------------------------------------------------------
    # FRD-015: Certification Scope Mismatch
    # ------------------------------------------------------------------
    {
        "rule_id": "FRD-015",
        "pattern_type": "scope_mismatch",
        "severity": "high",
        "description": (
            "Detects sustainability certificates where the certified "
            "scope (commodity type, product category, supply chain "
            "model) does not match the commodity or product being "
            "declared in the DDS. A scope mismatch means the "
            "certificate does not cover the declared goods."
        ),
        "detection_method": (
            "Extract the certificate scope fields: commodity_type, "
            "product_category, supply_chain_model, and geographic "
            "scope. Compare each against the corresponding fields "
            "in the DDS submission. Flag if any scope field does not "
            "match or if the certificate scope is narrower than the "
            "declared scope."
        ),
        "thresholds": {
            "check_commodity_match": True,
            "check_product_category": True,
            "check_supply_chain_model": True,
            "check_geographic_scope": True,
        },
        "enabled": True,
        "applicable_document_types": list(_CERT_DOC_TYPES),
    },
]

# ---------------------------------------------------------------------------
# Computed totals
# ---------------------------------------------------------------------------

TOTAL_FRAUD_RULES: int = len(FRAUD_RULES)

# ---------------------------------------------------------------------------
# Rule index: {rule_id: rule_dict}
# ---------------------------------------------------------------------------

FRAUD_RULE_INDEX: Dict[str, Dict[str, Any]] = {
    rule["rule_id"]: rule for rule in FRAUD_RULES
}

# ---------------------------------------------------------------------------
# Severity index: {severity: [rule_dict, ...]}
# ---------------------------------------------------------------------------

FRAUD_RULES_BY_SEVERITY: Dict[str, List[Dict[str, Any]]] = {}
for _rule in FRAUD_RULES:
    _sev = _rule["severity"]
    if _sev not in FRAUD_RULES_BY_SEVERITY:
        FRAUD_RULES_BY_SEVERITY[_sev] = []
    FRAUD_RULES_BY_SEVERITY[_sev].append(_rule)

# ---------------------------------------------------------------------------
# Pattern type index: {pattern_type: [rule_dict, ...]}
# ---------------------------------------------------------------------------

FRAUD_RULES_BY_PATTERN: Dict[str, List[Dict[str, Any]]] = {}
for _rule in FRAUD_RULES:
    _pat = _rule["pattern_type"]
    if _pat not in FRAUD_RULES_BY_PATTERN:
        FRAUD_RULES_BY_PATTERN[_pat] = []
    FRAUD_RULES_BY_PATTERN[_pat].append(_rule)

# ---------------------------------------------------------------------------
# Required documents by EUDR commodity type (for FRD-014)
# ---------------------------------------------------------------------------

REQUIRED_DOCUMENTS_BY_COMMODITY: Dict[str, List[str]] = {
    "cattle": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate (for hides)
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "cocoa": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "wr",               # Weighbridge Receipt
        "ltr",              # Laboratory Test Report (quality/grading)
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "coffee": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "wr",               # Weighbridge Receipt
        "ltr",              # Laboratory Test Report (quality/grading)
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "oil_palm": [
        "coo",              # Certificate of Origin
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "wr",               # Weighbridge Receipt
        "rspo_cert",        # RSPO Sustainability Certificate
        "iscc_cert",        # ISCC Certificate (if applicable)
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "rubber": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "wr",               # Weighbridge Receipt
        "ltr",              # Laboratory Test Report
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "soya": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "wr",               # Weighbridge Receipt
        "ltr",              # Laboratory Test Report
        "iscc_cert",        # ISCC Certificate (if applicable)
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
    "wood": [
        "coo",              # Certificate of Origin
        "pc",               # Phytosanitary Certificate
        "bol",              # Bill of Lading
        "ic",               # Commercial Invoice
        "fsc_cert",         # FSC Chain of Custody Certificate
        "fmp",              # Forest Management Plan
        "fc",               # Felling Certificate
        "wqc",              # Wood Quality Certificate
        "dds_draft",        # Due Diligence Statement
        "ssd",              # Supplier Self-Declaration
    ],
}

TOTAL_COMMODITIES: int = len(REQUIRED_DOCUMENTS_BY_COMMODITY)

# ---------------------------------------------------------------------------
# Document type to applicable rules index
# ---------------------------------------------------------------------------

DOC_TYPE_RULE_INDEX: Dict[str, List[str]] = {}
for _rule in FRAUD_RULES:
    for _doc_type in _rule["applicable_document_types"]:
        if _doc_type not in DOC_TYPE_RULE_INDEX:
            DOC_TYPE_RULE_INDEX[_doc_type] = []
        DOC_TYPE_RULE_INDEX[_doc_type].append(_rule["rule_id"])

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_rule(rule_id: str) -> Optional[Dict[str, Any]]:
    """Return a fraud rule by its identifier.

    Args:
        rule_id: Rule identifier (e.g. "FRD-001").

    Returns:
        Fraud rule dictionary, or None if not found.

    Example:
        >>> rule = get_rule("FRD-001")
        >>> rule["severity"]
        'high'
    """
    rule_upper = rule_id.upper().strip()
    return FRAUD_RULE_INDEX.get(rule_upper)


def get_all_rules() -> List[Dict[str, Any]]:
    """Return all 15 fraud detection rules.

    Returns:
        List of all fraud rule dictionaries.

    Example:
        >>> rules = get_all_rules()
        >>> len(rules)
        15
    """
    return list(FRAUD_RULES)


def get_enabled_rules() -> List[Dict[str, Any]]:
    """Return only enabled fraud detection rules.

    Returns:
        List of enabled fraud rule dictionaries.

    Example:
        >>> enabled = get_enabled_rules()
        >>> all(r["enabled"] for r in enabled)
        True
    """
    return [r for r in FRAUD_RULES if r["enabled"]]


def get_rules_by_severity(severity: str) -> List[Dict[str, Any]]:
    """Return fraud rules filtered by severity level.

    Args:
        severity: Severity level (low, medium, high, critical).

    Returns:
        List of fraud rules matching the severity.
        Returns empty list if no rules match.

    Example:
        >>> critical = get_rules_by_severity("critical")
        >>> all(r["severity"] == "critical" for r in critical)
        True
    """
    sev_lower = severity.lower().strip()
    return list(FRAUD_RULES_BY_SEVERITY.get(sev_lower, []))


def get_rules_by_pattern(pattern_type: str) -> List[Dict[str, Any]]:
    """Return fraud rules filtered by pattern type.

    Args:
        pattern_type: Pattern type identifier (e.g. "duplicate_reuse").

    Returns:
        List of fraud rules matching the pattern type.
        Returns empty list if no rules match.

    Example:
        >>> rules = get_rules_by_pattern("duplicate_reuse")
        >>> len(rules) >= 1
        True
    """
    pat_lower = pattern_type.lower().strip()
    return list(FRAUD_RULES_BY_PATTERN.get(pat_lower, []))


def get_rules_for_document_type(
    document_type: str,
) -> List[Dict[str, Any]]:
    """Return fraud rules applicable to a given document type.

    Args:
        document_type: Document type identifier (e.g. "coo", "fsc_cert").

    Returns:
        List of fraud rules that apply to the document type.
        Returns empty list if no rules apply.

    Example:
        >>> rules = get_rules_for_document_type("coo")
        >>> len(rules) >= 5
        True
    """
    doc_lower = document_type.lower().strip()
    rule_ids = DOC_TYPE_RULE_INDEX.get(doc_lower, [])
    return [
        FRAUD_RULE_INDEX[rid]
        for rid in rule_ids
        if rid in FRAUD_RULE_INDEX
    ]


def get_required_documents(commodity: str) -> List[str]:
    """Return required document types for an EUDR commodity.

    Args:
        commodity: EUDR commodity identifier (cattle, cocoa, coffee,
            oil_palm, rubber, soya, wood).

    Returns:
        List of required document type identifiers.
        Returns empty list if the commodity is not found.

    Example:
        >>> docs = get_required_documents("wood")
        >>> "fsc_cert" in docs
        True
        >>> "fmp" in docs
        True
    """
    com_lower = commodity.lower().strip()
    return list(REQUIRED_DOCUMENTS_BY_COMMODITY.get(com_lower, []))


def get_all_commodities() -> List[str]:
    """Return all EUDR commodity identifiers with required document definitions.

    Returns:
        Sorted list of commodity identifier strings.

    Example:
        >>> commodities = get_all_commodities()
        >>> "cocoa" in commodities
        True
    """
    return sorted(REQUIRED_DOCUMENTS_BY_COMMODITY.keys())


def get_rule_ids() -> List[str]:
    """Return all fraud rule identifiers in order.

    Returns:
        List of rule ID strings (FRD-001 through FRD-015).

    Example:
        >>> ids = get_rule_ids()
        >>> ids[0]
        'FRD-001'
        >>> ids[-1]
        'FRD-015'
    """
    return [r["rule_id"] for r in FRAUD_RULES]


def get_severity_distribution() -> Dict[str, int]:
    """Return the count of rules per severity level.

    Returns:
        Dictionary mapping severity levels to rule counts.

    Example:
        >>> dist = get_severity_distribution()
        >>> dist["high"] >= 5
        True
    """
    return {
        sev: len(rules)
        for sev, rules in FRAUD_RULES_BY_SEVERITY.items()
    }


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Fraud rules reference data loaded: "
    "%d rules (%s), %d commodities with required document definitions",
    TOTAL_FRAUD_RULES,
    ", ".join(
        f"{sev}={len(rules)}"
        for sev, rules in sorted(FRAUD_RULES_BY_SEVERITY.items())
    ),
    TOTAL_COMMODITIES,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "FRAUD_RULES",
    "FRAUD_RULE_INDEX",
    "FRAUD_RULES_BY_SEVERITY",
    "FRAUD_RULES_BY_PATTERN",
    "REQUIRED_DOCUMENTS_BY_COMMODITY",
    "DOC_TYPE_RULE_INDEX",
    # Counts
    "TOTAL_FRAUD_RULES",
    "TOTAL_COMMODITIES",
    # Lookup helpers
    "get_rule",
    "get_all_rules",
    "get_enabled_rules",
    "get_rules_by_severity",
    "get_rules_by_pattern",
    "get_rules_for_document_type",
    "get_required_documents",
    "get_all_commodities",
    "get_rule_ids",
    "get_severity_distribution",
]
