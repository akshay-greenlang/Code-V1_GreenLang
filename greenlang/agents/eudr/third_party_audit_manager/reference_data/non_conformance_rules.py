# -*- coding: utf-8 -*-
"""
Non-Conformance Classification Rules - AGENT-EUDR-024

Static reference data for deterministic rule-based NC severity
classification. Rules are organized by severity level (critical,
major, minor) with each rule specifying trigger indicators, EUDR
article mapping, Article 2(40) legislation category, and risk
impact score.

Rule Structure:
    Each rule contains:
    - rule_id: Unique rule identifier
    - indicator: Indicator key that triggers the rule
    - description: Human-readable rule description
    - eudr_article: Mapped EUDR article
    - article_2_40: Article 2(40) legislation category
    - risk_impact: Base risk impact score (0-100)
    - requires_evidence: Whether objective evidence is required
    - auto_car: Whether CAR should be automatically issued

Total Rules:
    - Critical: 7 rules (CRIT-001 through CRIT-007)
    - Major: 8 rules (MAJ-001 through MAJ-008)
    - Minor: 5 rules (MIN-001 through MIN-005)
    - Total: 20 classification rules

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Critical Severity Rules (7)
# ---------------------------------------------------------------------------

_CRITICAL_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "CRIT-001",
        "severity": "critical",
        "indicator": "fraud_or_falsification",
        "description": "Evidence of intentional fraud or document falsification",
        "eudr_article": "Art. 3",
        "article_2_40": "criminal_fraud",
        "risk_impact": Decimal("100"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": True,
        "examples": [
            "Forged supplier declarations",
            "Fabricated geolocation data",
            "Manipulated satellite imagery",
            "Falsified certification certificates",
        ],
    },
    {
        "rule_id": "CRIT-002",
        "severity": "critical",
        "indicator": "systematic_dds_failure",
        "description": "Systematic failure of due diligence system",
        "eudr_article": "Art. 4",
        "article_2_40": "regulatory_non_compliance",
        "risk_impact": Decimal("95"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": True,
        "examples": [
            "No DDS implemented despite obligation",
            "DDS exists on paper only with no implementation",
            "Complete absence of risk assessment process",
        ],
    },
    {
        "rule_id": "CRIT-003",
        "severity": "critical",
        "indicator": "active_deforestation_post_cutoff",
        "description": "Active deforestation after 31 December 2020 cutoff date",
        "eudr_article": "Art. 3(a)",
        "article_2_40": "environmental_crime",
        "risk_impact": Decimal("100"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": True,
        "examples": [
            "Satellite imagery showing forest loss after cutoff",
            "Land clearing permits issued after December 2020",
            "Supply chain linking to recently deforested areas",
        ],
    },
    {
        "rule_id": "CRIT-004",
        "severity": "critical",
        "indicator": "missing_all_geolocation",
        "description": "Missing geolocation data for all production plots",
        "eudr_article": "Art. 9(1)(d)",
        "article_2_40": "data_integrity_failure",
        "risk_impact": Decimal("90"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "No GPS coordinates recorded for any production plot",
            "Geolocation system completely non-functional",
            "All coordinates point to single default location",
        ],
    },
    {
        "rule_id": "CRIT-005",
        "severity": "critical",
        "indicator": "authority_order_non_compliance",
        "description": "Non-compliance with competent authority corrective action order",
        "eudr_article": "Art. 18",
        "article_2_40": "regulatory_defiance",
        "risk_impact": Decimal("95"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": True,
        "examples": [
            "Failure to implement authority-ordered corrective actions",
            "Continued market placement despite interim measure",
            "Ignoring definitive enforcement measures",
        ],
    },
    {
        "rule_id": "CRIT-006",
        "severity": "critical",
        "indicator": "certificate_falsification",
        "description": "Falsification of certification scheme certificate",
        "eudr_article": "Art. 10(2)",
        "article_2_40": "document_fraud",
        "risk_impact": Decimal("100"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": True,
        "examples": [
            "Counterfeit FSC/PEFC/RSPO/RA/ISCC certificate",
            "Altered certificate scope or validity dates",
            "Using terminated certificate as valid",
        ],
    },
    {
        "rule_id": "CRIT-007",
        "severity": "critical",
        "indicator": "concurrent_major_ncs_exceeded",
        "description": "More than 3 concurrent major NCs detected in same audit",
        "eudr_article": "Art. 4",
        "article_2_40": "systemic_failure",
        "risk_impact": Decimal("85"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Multiple major failures across different EUDR articles",
            "Pattern indicating systemic management breakdown",
        ],
    },
]

# ---------------------------------------------------------------------------
# Major Severity Rules (8)
# ---------------------------------------------------------------------------

_MAJOR_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "MAJ-001",
        "severity": "major",
        "indicator": "incomplete_risk_assessment",
        "description": "Risk assessment does not cover all required criteria",
        "eudr_article": "Art. 10(1)",
        "article_2_40": "risk_assessment_gap",
        "risk_impact": Decimal("70"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Risk assessment missing deforestation risk evaluation",
            "Country risk benchmarking not performed",
            "Supply chain complexity not assessed",
        ],
    },
    {
        "rule_id": "MAJ-002",
        "severity": "major",
        "indicator": "missing_supplier_information",
        "description": "Missing required supplier information per Article 9",
        "eudr_article": "Art. 9(1)(f)",
        "article_2_40": "data_completeness_failure",
        "risk_impact": Decimal("65"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Supplier name or address missing from records",
            "No contact details for key suppliers",
            "Supplier identification incomplete for >20% of volume",
        ],
    },
    {
        "rule_id": "MAJ-003",
        "severity": "major",
        "indicator": "inadequate_risk_mitigation",
        "description": "Risk mitigation measures inadequate for identified risks",
        "eudr_article": "Art. 11(1)",
        "article_2_40": "mitigation_inadequacy",
        "risk_impact": Decimal("70"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "High-risk suppliers without enhanced due diligence",
            "No mitigation plan for identified deforestation risks",
            "Mitigation measures not proportional to risk level",
        ],
    },
    {
        "rule_id": "MAJ-004",
        "severity": "major",
        "indicator": "expired_certification",
        "description": "Expired or suspended certification scheme certificate",
        "eudr_article": "Art. 10(2)",
        "article_2_40": "certification_lapse",
        "risk_impact": Decimal("60"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Certificate expired more than 30 days ago",
            "Certificate suspended by certification body",
            "Claiming certification without valid certificate",
        ],
    },
    {
        "rule_id": "MAJ-005",
        "severity": "major",
        "indicator": "traceability_gap_above_10pct",
        "description": "Traceability gap exceeding 10% of total volume",
        "eudr_article": "Art. 9(1)",
        "article_2_40": "traceability_failure",
        "risk_impact": Decimal("65"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Unable to trace origin for >10% of purchased volume",
            "Supply chain breaks in traceability documentation",
            "Mixed certified and uncertified without mass balance",
        ],
    },
    {
        "rule_id": "MAJ-006",
        "severity": "major",
        "indicator": "records_missing_above_5pct",
        "description": "Records missing for more than 5% of transactions",
        "eudr_article": "Art. 29",
        "article_2_40": "record_keeping_failure",
        "risk_impact": Decimal("55"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Purchase records missing for >5% of transactions",
            "Delivery notes not maintained for significant portion",
            "Audit trail gaps in record management system",
        ],
    },
    {
        "rule_id": "MAJ-007",
        "severity": "major",
        "indicator": "partial_geolocation_missing",
        "description": "Partial geolocation data missing for production plots",
        "eudr_article": "Art. 9(1)(d)",
        "article_2_40": "data_completeness_failure",
        "risk_impact": Decimal("60"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "Geolocation missing for 10-50% of production plots",
            "Coordinates recorded but insufficient precision",
            "Polygon data missing for some farm boundaries",
        ],
    },
    {
        "rule_id": "MAJ-008",
        "severity": "major",
        "indicator": "country_risk_not_assessed",
        "description": "Country risk benchmarking not incorporated in risk assessment",
        "eudr_article": "Art. 10(2)",
        "article_2_40": "risk_assessment_gap",
        "risk_impact": Decimal("55"),
        "requires_evidence": True,
        "auto_car": True,
        "authority_notification": False,
        "examples": [
            "EU country benchmarking classification not used",
            "Risk assessment ignores country-level risk factors",
            "No differentiation between high/standard/low risk countries",
        ],
    },
]

# ---------------------------------------------------------------------------
# Minor Severity Rules (5)
# ---------------------------------------------------------------------------

_MINOR_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "MIN-001",
        "severity": "minor",
        "indicator": "isolated_documentation_gap",
        "description": "Isolated documentation gap not undermining system integrity",
        "eudr_article": "Art. 29",
        "article_2_40": "documentation_gap",
        "risk_impact": Decimal("25"),
        "requires_evidence": False,
        "auto_car": False,
        "authority_notification": False,
        "examples": [
            "Single missing document in otherwise complete records",
            "Documentation formatting inconsistency",
            "Missing signature on one document",
        ],
    },
    {
        "rule_id": "MIN-002",
        "severity": "minor",
        "indicator": "minor_record_keeping",
        "description": "Minor record-keeping issue with limited scope",
        "eudr_article": "Art. 29",
        "article_2_40": "record_keeping_minor",
        "risk_impact": Decimal("20"),
        "requires_evidence": False,
        "auto_car": False,
        "authority_notification": False,
        "examples": [
            "Records filed in wrong category",
            "Minor data entry errors in records",
            "Incomplete metadata on some records",
        ],
    },
    {
        "rule_id": "MIN-003",
        "severity": "minor",
        "indicator": "procedural_deviation",
        "description": "Procedural deviation from documented process",
        "eudr_article": "Art. 4",
        "article_2_40": "procedural_non_compliance",
        "risk_impact": Decimal("15"),
        "requires_evidence": False,
        "auto_car": False,
        "authority_notification": False,
        "examples": [
            "Process step performed out of documented order",
            "Approval obtained verbally instead of in writing",
            "Minor deviation from standard operating procedure",
        ],
    },
    {
        "rule_id": "MIN-004",
        "severity": "minor",
        "indicator": "delayed_data_update",
        "description": "Data update delayed by less than 30 days",
        "eudr_article": "Art. 9",
        "article_2_40": "timeliness_issue",
        "risk_impact": Decimal("10"),
        "requires_evidence": False,
        "auto_car": False,
        "authority_notification": False,
        "examples": [
            "Supplier information update delayed by 2 weeks",
            "Geolocation data refresh overdue by <30 days",
            "Risk assessment review slightly past scheduled date",
        ],
    },
    {
        "rule_id": "MIN-005",
        "severity": "minor",
        "indicator": "formatting_non_compliance",
        "description": "Report or record formatting non-compliance",
        "eudr_article": "Art. 29",
        "article_2_40": "format_deviation",
        "risk_impact": Decimal("10"),
        "requires_evidence": False,
        "auto_car": False,
        "authority_notification": False,
        "examples": [
            "Report not using required template",
            "Date format inconsistency across records",
            "Missing report headers or footers",
        ],
    },
]

# ---------------------------------------------------------------------------
# Combined Rule Set
# ---------------------------------------------------------------------------

NC_CLASSIFICATION_RULES: Dict[str, List[Dict[str, Any]]] = {
    "critical": _CRITICAL_RULES,
    "major": _MAJOR_RULES,
    "minor": _MINOR_RULES,
}


def get_rules_by_severity(severity: str) -> List[Dict[str, Any]]:
    """Get classification rules by severity level.

    Args:
        severity: Severity level (critical, major, minor).

    Returns:
        List of rule definitions.

    Raises:
        ValueError: If severity level is not supported.
    """
    rules = NC_CLASSIFICATION_RULES.get(severity.lower())
    if rules is None:
        raise ValueError(
            f"Unsupported severity level: {severity}. "
            f"Must be one of {list(NC_CLASSIFICATION_RULES.keys())}"
        )
    return rules


def get_rule_by_id(rule_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific rule by its identifier.

    Args:
        rule_id: Rule identifier (e.g. CRIT-001, MAJ-003, MIN-002).

    Returns:
        Rule definition dictionary or None if not found.
    """
    for rules in NC_CLASSIFICATION_RULES.values():
        for rule in rules:
            if rule["rule_id"] == rule_id:
                return rule
    return None


def get_all_rules_flat() -> List[Dict[str, Any]]:
    """Get all classification rules as a flat list.

    Returns:
        List of all rule definitions across all severity levels.
    """
    all_rules: List[Dict[str, Any]] = []
    for rules in NC_CLASSIFICATION_RULES.values():
        all_rules.extend(rules)
    return all_rules
