# -*- coding: utf-8 -*-
"""
Quality Gate Rules - AGENT-EUDR-026

Reference data for the three quality gates (QG-1, QG-2, QG-3) including
check definitions, weight assignments, threshold configurations, and
remediation guidance for each check.

Quality Gate Summary:
    QG-1 (Information Gathering Completeness):
        Threshold: 90% standard / 80% simplified
        7 checks covering Phase 1 agents (EUDR-001 to EUDR-015)

    QG-2 (Risk Assessment Coverage):
        Threshold: 95% standard / 85% simplified
        10 checks covering Phase 2 agents (EUDR-016 to EUDR-025)

    QG-3 (Mitigation Adequacy):
        Threshold: residual risk <= 15 standard / <= 25 simplified
        4 checks covering mitigation measures and adequacy

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    QualityGateId,
)


# ---------------------------------------------------------------------------
# QG-1 rules: Information Gathering Completeness
# ---------------------------------------------------------------------------

QG1_RULES: Dict[str, Any] = {
    "gate_id": "QG-1",
    "name": "Information Gathering Completeness",
    "article_reference": "Article 9 -> Article 10 transition",
    "description": (
        "Validates that sufficient information has been gathered per "
        "Article 9 requirements before proceeding to risk assessment. "
        "Checks coverage of supply chain mapping, geolocation verification, "
        "satellite monitoring, chain of custody, and documentary evidence."
    ),
    "standard_threshold": Decimal("0.90"),
    "simplified_threshold": Decimal("0.80"),
    "checks": [
        {
            "name": "Supply Chain Mapping Coverage",
            "weight": Decimal("0.25"),
            "description": "Percentage of supply chain nodes mapped with operator ID and coordinates",
            "source_agents": ["EUDR-001"],
            "remediation": (
                "Re-run EUDR-001 Supply Chain Mapping Master with expanded "
                "scope. Ensure all operator IDs and coordinates are captured."
            ),
        },
        {
            "name": "Geolocation Verification Coverage",
            "weight": Decimal("0.20"),
            "description": "Percentage of production plots with verified GPS coordinates",
            "source_agents": ["EUDR-002", "EUDR-006", "EUDR-007"],
            "remediation": (
                "Run EUDR-002 Geolocation Verification and EUDR-007 GPS "
                "Validation on remaining unverified plots."
            ),
        },
        {
            "name": "Satellite Monitoring Coverage",
            "weight": Decimal("0.15"),
            "description": "Percentage of production areas with satellite monitoring data",
            "source_agents": ["EUDR-003", "EUDR-004", "EUDR-005"],
            "remediation": (
                "Expand satellite monitoring coverage to include all "
                "production areas using EUDR-003/004/005 agents."
            ),
        },
        {
            "name": "Chain of Custody Completeness",
            "weight": Decimal("0.15"),
            "description": "Percentage of supply chain links with documented custody",
            "source_agents": ["EUDR-009", "EUDR-010", "EUDR-011"],
            "remediation": (
                "Supplement chain of custody data with additional documentation "
                "using EUDR-009/010/011 agents."
            ),
        },
        {
            "name": "Multi-Tier Supplier Coverage",
            "weight": Decimal("0.10"),
            "description": "Percentage of supplier tiers mapped and verified",
            "source_agents": ["EUDR-008"],
            "remediation": (
                "Deepen supplier tier mapping with EUDR-008 Multi-Tier "
                "Supplier Tracker for uncovered tiers."
            ),
        },
        {
            "name": "Documentary Evidence Completeness",
            "weight": Decimal("0.10"),
            "description": "Percentage of required documents authenticated",
            "source_agents": ["EUDR-012", "EUDR-013"],
            "remediation": (
                "Collect and authenticate remaining required documents "
                "using EUDR-012 Document Authentication agent."
            ),
        },
        {
            "name": "Traceability Code Coverage",
            "weight": Decimal("0.05"),
            "description": "Percentage of products with traceability codes generated",
            "source_agents": ["EUDR-014", "EUDR-015"],
            "remediation": (
                "Generate traceability codes for uncovered products "
                "using EUDR-014 QR Code Generator."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# QG-2 rules: Risk Assessment Coverage
# ---------------------------------------------------------------------------

QG2_RULES: Dict[str, Any] = {
    "gate_id": "QG-2",
    "name": "Risk Assessment Coverage",
    "article_reference": "Article 10 -> Article 11 transition",
    "description": (
        "Validates that all required risk dimensions have been assessed "
        "per Article 10 requirements before proceeding to risk mitigation. "
        "Checks all 10 risk dimension scores and composite score computation."
    ),
    "standard_threshold": Decimal("0.95"),
    "simplified_threshold": Decimal("0.85"),
    "checks": [
        {
            "name": "Country Risk Dimension",
            "weight": Decimal("0.15"),
            "description": "Country risk score from EUDR-016 available and valid",
            "source_agents": ["EUDR-016"],
            "remediation": "Re-run EUDR-016 Country Risk Evaluator.",
        },
        {
            "name": "Supplier Risk Dimension",
            "weight": Decimal("0.12"),
            "description": "Supplier risk score from EUDR-017 available and valid",
            "source_agents": ["EUDR-017"],
            "remediation": "Re-run EUDR-017 Supplier Risk Scorer.",
        },
        {
            "name": "Commodity Risk Dimension",
            "weight": Decimal("0.10"),
            "description": "Commodity risk score from EUDR-018 available and valid",
            "source_agents": ["EUDR-018"],
            "remediation": "Re-run EUDR-018 Commodity Risk Analyzer.",
        },
        {
            "name": "Corruption Risk Dimension",
            "weight": Decimal("0.08"),
            "description": "Corruption risk score from EUDR-019 available and valid",
            "source_agents": ["EUDR-019"],
            "remediation": "Re-run EUDR-019 Corruption Index Monitor.",
        },
        {
            "name": "Deforestation Risk Dimension",
            "weight": Decimal("0.15"),
            "description": "Deforestation risk score from EUDR-020 available and valid",
            "source_agents": ["EUDR-020"],
            "remediation": "Re-run EUDR-020 Deforestation Alert System.",
        },
        {
            "name": "Indigenous Rights Dimension",
            "weight": Decimal("0.10"),
            "description": "Indigenous rights risk score from EUDR-021 available and valid",
            "source_agents": ["EUDR-021"],
            "remediation": "Re-run EUDR-021 Indigenous Rights Checker.",
        },
        {
            "name": "Protected Area Dimension",
            "weight": Decimal("0.10"),
            "description": "Protected area risk score from EUDR-022 available and valid",
            "source_agents": ["EUDR-022"],
            "remediation": "Re-run EUDR-022 Protected Area Validator.",
        },
        {
            "name": "Legal Compliance Dimension",
            "weight": Decimal("0.10"),
            "description": "Legal compliance risk score from EUDR-023 available and valid",
            "source_agents": ["EUDR-023"],
            "remediation": "Re-run EUDR-023 Legal Compliance Verifier.",
        },
        {
            "name": "Third-Party Audit Dimension",
            "weight": Decimal("0.05"),
            "description": "Audit risk score from EUDR-024 available and valid",
            "source_agents": ["EUDR-024"],
            "remediation": "Re-run EUDR-024 Third-Party Audit Manager.",
        },
        {
            "name": "Mitigation Readiness Dimension",
            "weight": Decimal("0.05"),
            "description": "Mitigation readiness score from EUDR-025 available and valid",
            "source_agents": ["EUDR-025"],
            "remediation": "Re-run EUDR-025 Risk Mitigation Advisor.",
        },
    ],
}


# ---------------------------------------------------------------------------
# QG-3 rules: Mitigation Adequacy
# ---------------------------------------------------------------------------

QG3_RULES: Dict[str, Any] = {
    "gate_id": "QG-3",
    "name": "Mitigation Adequacy",
    "article_reference": "Article 11 -> Article 12 transition",
    "description": (
        "Validates that risk mitigation measures are adequate and "
        "proportionate per Article 11 requirements before proceeding "
        "to package generation. Checks residual risk level, "
        "proportionality, evidence documentation, and stakeholder engagement."
    ),
    "standard_threshold": Decimal("15"),
    "simplified_threshold": Decimal("25"),
    "checks": [
        {
            "name": "Mitigation Adequacy",
            "weight": Decimal("0.40"),
            "description": "Residual risk after mitigation meets target threshold",
            "source_agents": ["EUDR-025"],
            "remediation": (
                "Apply additional mitigation measures to reduce residual "
                "risk below the target threshold. Consider enhanced due "
                "diligence measures per Article 11."
            ),
        },
        {
            "name": "Mitigation Proportionality",
            "weight": Decimal("0.25"),
            "description": "Mitigation measures proportionate to identified risks",
            "source_agents": ["EUDR-025"],
            "remediation": (
                "Ensure mitigation measures are proportionate to the "
                "risk level. Enhanced risks require more comprehensive "
                "measures. Document proportionality justification."
            ),
        },
        {
            "name": "Evidence Documentation",
            "weight": Decimal("0.20"),
            "description": "Mitigation measures documented with supporting evidence",
            "source_agents": ["EUDR-025", "EUDR-012"],
            "remediation": (
                "Document all mitigation measures with supporting evidence. "
                "Use EUDR-012 to authenticate evidence documents."
            ),
        },
        {
            "name": "Stakeholder Engagement",
            "weight": Decimal("0.15"),
            "description": "Relevant stakeholders engaged in mitigation process",
            "source_agents": ["EUDR-025", "EUDR-024"],
            "remediation": (
                "Engage relevant stakeholders (suppliers, communities, "
                "certification bodies) in the mitigation process. "
                "Document engagement activities."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_gate_rules(gate_id: QualityGateId) -> Dict[str, Any]:
    """Get the complete rule definition for a quality gate.

    Args:
        gate_id: Quality gate identifier.

    Returns:
        Gate rule dictionary with checks, thresholds, and remediations.

    Example:
        >>> rules = get_gate_rules(QualityGateId.QG1)
        >>> assert rules["standard_threshold"] == Decimal("0.90")
    """
    gate_map: Dict[QualityGateId, Dict[str, Any]] = {
        QualityGateId.QG1: QG1_RULES,
        QualityGateId.QG2: QG2_RULES,
        QualityGateId.QG3: QG3_RULES,
    }
    return gate_map.get(gate_id, {})


def get_all_gate_rules() -> Dict[str, Dict[str, Any]]:
    """Get all quality gate rule definitions.

    Returns:
        Dictionary mapping gate_id string to rule definitions.
    """
    return {
        "QG-1": QG1_RULES,
        "QG-2": QG2_RULES,
        "QG-3": QG3_RULES,
    }


def get_check_names(gate_id: QualityGateId) -> List[str]:
    """Get the list of check names for a quality gate.

    Args:
        gate_id: Quality gate identifier.

    Returns:
        List of check name strings.
    """
    rules = get_gate_rules(gate_id)
    return [c["name"] for c in rules.get("checks", [])]


def get_remediation(gate_id: QualityGateId, check_name: str) -> str:
    """Get remediation guidance for a specific check.

    Args:
        gate_id: Quality gate identifier.
        check_name: Name of the check.

    Returns:
        Remediation guidance text, or empty string if not found.
    """
    rules = get_gate_rules(gate_id)
    for check in rules.get("checks", []):
        if check["name"] == check_name:
            return check.get("remediation", "")
    return ""
