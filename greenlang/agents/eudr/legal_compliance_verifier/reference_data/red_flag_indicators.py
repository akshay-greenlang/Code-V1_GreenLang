# -*- coding: utf-8 -*-
"""
Red Flag Indicator Reference Data - AGENT-EUDR-023

Defines 40 red flag indicators across 6 categories for detection of
illegal activity, corruption, and non-compliance in EUDR commodity
supply chains. Each indicator has a deterministic base weight, severity
classification, triggering conditions, and data source attribution.

Zero-Hallucination: All red flag indicators use deterministic threshold
comparisons and pattern matching. No LLM is used for red flag detection.
Scoring formula per Architecture Spec Appendix D:
    flag_score = base_weight * country_multiplier * commodity_multiplier
    aggregate = (SUM(flag_scores) / max_possible) * 100

Red Flag Categories (6):
    1. Corruption & Bribery (RF-001 to RF-008): 8 indicators
    2. Illegal Logging (RF-009 to RF-015): 7 indicators
    3. Land Rights Violations (RF-016 to RF-021): 6 indicators
    4. Labour Violations (RF-022 to RF-027): 6 indicators
    5. Tax Evasion (RF-028 to RF-032): 5 indicators
    6. Document Fraud (RF-033 to RF-040): 8 indicators

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 40 Red flag indicator definitions
# ---------------------------------------------------------------------------

RED_FLAG_INDICATORS: Dict[str, Dict[str, Any]] = {
    # === Corruption & Bribery (RF-001 to RF-008) ===
    "RF-001": {
        "code": "RF-001",
        "category": "corruption_bribery",
        "description": "Supplier located in country with CPI < 30",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "country_cpi_score < 30",
        "data_source": "transparency_international_cpi",
    },
    "RF-002": {
        "code": "RF-002",
        "category": "corruption_bribery",
        "description": "Concession awarded without competitive tender",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "concession_tender_type == 'direct_award'",
        "data_source": "procurement_records",
    },
    "RF-003": {
        "code": "RF-003",
        "category": "corruption_bribery",
        "description": "Permit issued in < 50% of standard processing time",
        "base_weight": 0.15,
        "severity": "moderate",
        "trigger_condition": "permit_processing_days < (country_avg_days * 0.5)",
        "data_source": "permit_processing_times",
    },
    "RF-004": {
        "code": "RF-004",
        "category": "corruption_bribery",
        "description": "Multiple permits from same official in short timeframe",
        "base_weight": 0.15,
        "severity": "moderate",
        "trigger_condition": "permits_from_same_official_30d > 3",
        "data_source": "permit_issuance_records",
    },
    "RF-005": {
        "code": "RF-005",
        "category": "corruption_bribery",
        "description": "Politically exposed person (PEP) in ownership chain",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "pep_match_in_ownership == True",
        "data_source": "pep_database",
    },
    "RF-006": {
        "code": "RF-006",
        "category": "corruption_bribery",
        "description": "Supplier on OFAC/EU/UN sanctions list",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "sanctions_list_match == True",
        "data_source": "ofac_eu_un_sanctions",
    },
    "RF-007": {
        "code": "RF-007",
        "category": "corruption_bribery",
        "description": "Beneficial ownership obscured through shell companies",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "shell_company_layers > 2",
        "data_source": "corporate_registry",
    },
    "RF-008": {
        "code": "RF-008",
        "category": "corruption_bribery",
        "description": "Facilitation payment patterns in transaction records",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "facilitation_payment_pattern_detected == True",
        "data_source": "transaction_analysis",
    },
    # === Illegal Logging (RF-009 to RF-015) ===
    "RF-009": {
        "code": "RF-009",
        "category": "illegal_logging",
        "description": "Harvest volume exceeds concession permit limits",
        "base_weight": 0.30,
        "severity": "critical",
        "trigger_condition": "harvest_volume > permit_volume_limit",
        "data_source": "harvesting_records_vs_permits",
    },
    "RF-010": {
        "code": "RF-010",
        "category": "illegal_logging",
        "description": "Species harvested not listed in concession permit",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "species_not_in_permit_list == True",
        "data_source": "species_permit_cross_reference",
    },
    "RF-011": {
        "code": "RF-011",
        "category": "illegal_logging",
        "description": "Harvest location outside permitted boundaries",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "harvest_coords_outside_concession == True",
        "data_source": "gis_boundary_analysis",
    },
    "RF-012": {
        "code": "RF-012",
        "category": "illegal_logging",
        "description": "Transport documents inconsistent with harvest records",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "transport_volume_mismatch > 0.10",
        "data_source": "transport_vs_harvest_reconciliation",
    },
    "RF-013": {
        "code": "RF-013",
        "category": "illegal_logging",
        "description": "CITES-listed species without CITES permit",
        "base_weight": 0.30,
        "severity": "critical",
        "trigger_condition": "cites_species_no_permit == True",
        "data_source": "cites_species_database",
    },
    "RF-014": {
        "code": "RF-014",
        "category": "illegal_logging",
        "description": "Night-time satellite activity in forest concession",
        "base_weight": 0.15,
        "severity": "moderate",
        "trigger_condition": "nighttime_activity_detected == True",
        "data_source": "satellite_monitoring",
    },
    "RF-015": {
        "code": "RF-015",
        "category": "illegal_logging",
        "description": "Road construction in previously unlogged forest",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "new_road_in_unlogged_forest == True",
        "data_source": "satellite_change_detection",
    },
    # === Land Rights Violations (RF-016 to RF-021) ===
    "RF-016": {
        "code": "RF-016",
        "category": "land_rights_violation",
        "description": "Production on disputed land (overlapping claims)",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "overlapping_land_claims == True",
        "data_source": "land_registry_analysis",
    },
    "RF-017": {
        "code": "RF-017",
        "category": "land_rights_violation",
        "description": "No FPIC documentation for indigenous territory overlap",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "indigenous_overlap_no_fpic == True",
        "data_source": "indigenous_territory_map",
    },
    "RF-018": {
        "code": "RF-018",
        "category": "land_rights_violation",
        "description": "Active land rights litigation involving supplier",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "active_litigation == True",
        "data_source": "court_records",
    },
    "RF-019": {
        "code": "RF-019",
        "category": "land_rights_violation",
        "description": "Forced displacement reported in sourcing area",
        "base_weight": 0.20,
        "severity": "critical",
        "trigger_condition": "forced_displacement_reports > 0",
        "data_source": "human_rights_reports",
    },
    "RF-020": {
        "code": "RF-020",
        "category": "land_rights_violation",
        "description": "Customary tenure not recognized despite evidence",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "customary_tenure_unrecognized == True",
        "data_source": "community_rights_assessment",
    },
    "RF-021": {
        "code": "RF-021",
        "category": "land_rights_violation",
        "description": "Community grievance filed against supplier",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "community_grievance_count > 0",
        "data_source": "grievance_mechanism_records",
    },
    # === Labour Violations (RF-022 to RF-027) ===
    "RF-022": {
        "code": "RF-022",
        "category": "labour_violation",
        "description": "ILO core convention violation reports for supplier",
        "base_weight": 0.20,
        "severity": "critical",
        "trigger_condition": "ilo_violation_reports > 0",
        "data_source": "ilo_natlex_ceacr",
    },
    "RF-023": {
        "code": "RF-023",
        "category": "labour_violation",
        "description": "Child labour indicators (education enrollment gaps)",
        "base_weight": 0.20,
        "severity": "critical",
        "trigger_condition": "child_labour_indicator_detected == True",
        "data_source": "dol_ilab_list",
    },
    "RF-024": {
        "code": "RF-024",
        "category": "labour_violation",
        "description": "Forced labour indicators (debt bondage, withheld documents)",
        "base_weight": 0.20,
        "severity": "critical",
        "trigger_condition": "forced_labour_indicator_detected == True",
        "data_source": "ilo_forced_labour_indicators",
    },
    "RF-025": {
        "code": "RF-025",
        "category": "labour_violation",
        "description": "OSH violation citations from labour inspectorate",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "osh_violation_count > 0",
        "data_source": "labour_inspectorate_records",
    },
    "RF-026": {
        "code": "RF-026",
        "category": "labour_violation",
        "description": "Below minimum wage payment patterns",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "below_minimum_wage_detected == True",
        "data_source": "wage_analysis",
    },
    "RF-027": {
        "code": "RF-027",
        "category": "labour_violation",
        "description": "Excessive working hours (> 60h/week patterns)",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "avg_weekly_hours > 60",
        "data_source": "working_hours_records",
    },
    # === Tax Evasion (RF-028 to RF-032) ===
    "RF-028": {
        "code": "RF-028",
        "category": "tax_evasion",
        "description": "Transfer pricing anomalies (price < 70% market rate)",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "transfer_price < (market_price * 0.70)",
        "data_source": "price_analysis",
    },
    "RF-029": {
        "code": "RF-029",
        "category": "tax_evasion",
        "description": "Royalty underpayment patterns",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "royalty_paid < (expected_royalty * 0.80)",
        "data_source": "royalty_payment_analysis",
    },
    "RF-030": {
        "code": "RF-030",
        "category": "tax_evasion",
        "description": "Tax clearance certificate expired or missing",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "tax_clearance_valid == False",
        "data_source": "document_verification",
    },
    "RF-031": {
        "code": "RF-031",
        "category": "tax_evasion",
        "description": "Revenue inconsistent with declared production volume",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "revenue_production_ratio_outlier == True",
        "data_source": "financial_analysis",
    },
    "RF-032": {
        "code": "RF-032",
        "category": "tax_evasion",
        "description": "Export value significantly below import declaration",
        "base_weight": 0.05,
        "severity": "low",
        "trigger_condition": "export_import_value_gap > 0.30",
        "data_source": "trade_data_reconciliation",
    },
    # === Document Fraud (RF-033 to RF-040) ===
    "RF-033": {
        "code": "RF-033",
        "category": "document_fraud",
        "description": "Document dates inconsistent (issue after expiry)",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "issue_date > expiry_date",
        "data_source": "document_date_analysis",
    },
    "RF-034": {
        "code": "RF-034",
        "category": "document_fraud",
        "description": "Issuing authority not in registered authority database",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "authority_not_registered == True",
        "data_source": "authority_registry",
    },
    "RF-035": {
        "code": "RF-035",
        "category": "document_fraud",
        "description": "Certificate number fails format validation",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "cert_number_format_invalid == True",
        "data_source": "format_validation_engine",
    },
    "RF-036": {
        "code": "RF-036",
        "category": "document_fraud",
        "description": "Multiple documents with sequential serial numbers from same batch",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "sequential_serial_count > 3",
        "data_source": "serial_number_analysis",
    },
    "RF-037": {
        "code": "RF-037",
        "category": "document_fraud",
        "description": "Digital signature verification failure",
        "base_weight": 0.25,
        "severity": "critical",
        "trigger_condition": "digital_signature_valid == False",
        "data_source": "signature_verification",
    },
    "RF-038": {
        "code": "RF-038",
        "category": "document_fraud",
        "description": "Document metadata inconsistent with content",
        "base_weight": 0.10,
        "severity": "moderate",
        "trigger_condition": "metadata_content_mismatch == True",
        "data_source": "metadata_analysis",
    },
    "RF-039": {
        "code": "RF-039",
        "category": "document_fraud",
        "description": "Duplicate document submitted for different shipments",
        "base_weight": 0.20,
        "severity": "high",
        "trigger_condition": "duplicate_document_different_shipment == True",
        "data_source": "document_deduplication",
    },
    "RF-040": {
        "code": "RF-040",
        "category": "document_fraud",
        "description": "Document template mismatch with known issuing authority format",
        "base_weight": 0.15,
        "severity": "high",
        "trigger_condition": "template_mismatch_detected == True",
        "data_source": "template_matching",
    },
}


def get_red_flag_definition(flag_code: str) -> Optional[Dict[str, Any]]:
    """Get the definition for a specific red flag indicator.

    Args:
        flag_code: Red flag code (e.g. "RF-001").

    Returns:
        Red flag definition dict or None if not found.

    Example:
        >>> flag = get_red_flag_definition("RF-001")
        >>> assert flag["category"] == "corruption_bribery"
        >>> assert flag["base_weight"] == 0.20
    """
    return RED_FLAG_INDICATORS.get(flag_code)


def get_flags_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all red flag indicators for a specific category.

    Args:
        category: Red flag category (e.g. "corruption_bribery").

    Returns:
        List of red flag definition dicts matching the category.

    Example:
        >>> flags = get_flags_by_category("illegal_logging")
        >>> assert len(flags) == 7
    """
    return [
        flag for flag in RED_FLAG_INDICATORS.values()
        if flag["category"] == category
    ]


def get_max_possible_score() -> float:
    """Calculate the maximum possible aggregate red flag score.

    This is used as the denominator in the normalized scoring formula.
    max_possible = sum of all base weights (assuming all flags triggered
    with maximum multipliers of 2.0 country and 1.5 commodity).

    Returns:
        Maximum possible aggregate score.

    Example:
        >>> max_score = get_max_possible_score()
        >>> assert max_score > 0
    """
    return sum(
        flag["base_weight"] * 2.0 * 1.5
        for flag in RED_FLAG_INDICATORS.values()
    )
