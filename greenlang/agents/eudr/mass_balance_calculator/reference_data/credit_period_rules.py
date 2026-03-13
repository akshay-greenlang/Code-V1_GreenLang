# -*- coding: utf-8 -*-
"""
Credit Period Rules Reference Data - AGENT-EUDR-011 Mass Balance Calculator

Standard-specific credit period duration, carry-forward, overdraft,
grace period, and expiry rules for mass balance chain of custody.
These rules determine how long mass balance credits remain valid,
under what conditions they may be carried forward to the next period,
and what happens when a facility draws more certified output than it
has certified input credits.

Datasets:
    CREDIT_PERIOD_RULES:
        Per-certification-standard period rules.  Keyed by standard
        identifier (e.g. "RSPO-SCC", "FSC-STD-40-004").  Each entry
        contains period_duration_months, carry_forward settings,
        grace_period settings, overdraft settings, and reconciliation
        requirements.

    CARRY_FORWARD_RULES:
        Detailed carry-forward behaviour per standard including
        maximum CF percentage, expiry behaviour, partial utilization,
        and audit requirements.

    OVERDRAFT_POLICIES:
        Overdraft tolerance policies per standard including tolerance
        mode, tolerance value, resolution deadline, and escalation
        rules.

    RECONCILIATION_REQUIREMENTS:
        Period-end reconciliation procedures per standard including
        variance thresholds, sign-off requirements, and audit trail
        mandates.

Lookup helpers:
    get_period_rules(standard) -> dict | None
    get_all_rules() -> list[dict]
    get_carry_forward_rules(standard) -> dict | None
    get_grace_period(standard) -> int
    get_overdraft_tolerance(standard) -> dict | None

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011 (Mass Balance Calculator) - Appendix C
Agent ID: GL-EUDR-MBC-011
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14
Standard: ISO 22095:2020 Chain of Custody - Mass Balance
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credit period rules by certification standard
# ---------------------------------------------------------------------------

CREDIT_PERIOD_RULES: Dict[str, Dict[str, Any]] = {
    # ---- RSPO Supply Chain Certification (SCC) ----
    "RSPO-SCC": {
        "name": "RSPO Supply Chain Certification Standard",
        "version": "2020",
        "effective_date": "2020-11-01",
        "applicable_commodities": ["oil_palm"],
        "period_duration_months": 3,
        "period_duration_days": 90,
        "period_alignment": "calendar_quarter",
        "period_start_rule": "first_day_of_quarter",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "end_of_receiving_period",
            "expiry_description": (
                "Credits carried forward expire at the end of the "
                "receiving (next) period.  They cannot be carried "
                "forward again."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 5,
            "calendar_days": 7,
            "purpose": "reconciliation_and_correction",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 48,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_compliance_officer"},
                {"hours": 72, "action": "suspend_output_claims"},
            ],
            "exemption_process": "formal_request_with_justification",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "end_of_period",
            "variance_acceptable_pct": 2.0,
            "variance_warning_pct": 5.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "quality_manager"],
            "audit_trail_retention_years": 5,
            "third_party_verification": False,
        },
        "source": "RSPO",
        "reference_document": "RSPO SCC Standard 2020, Section 3.2",
        "notes": (
            "RSPO uses quarterly periods with strict zero-overdraft policy. "
            "Carry-forward is allowed but credits expire at end of next quarter."
        ),
    },

    # ---- FSC Chain of Custody (STD-40-004 v3.0) ----
    "FSC-STD-40-004": {
        "name": "FSC Chain of Custody Standard",
        "version": "3.0",
        "effective_date": "2024-01-01",
        "applicable_commodities": ["wood", "rubber"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "certification_anniversary",
        "period_start_rule": "certification_start_date",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "no_expiry_within_active_period",
            "expiry_description": (
                "Credits carried forward do not expire while the "
                "certification period remains active.  They are consumed "
                "on a FIFO basis."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 10,
            "calendar_days": 14,
            "purpose": "annual_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
                "carry_forward_calculation",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 120,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 48, "action": "alert_facility_manager"},
                {"hours": 96, "action": "alert_certification_body"},
                {"hours": 120, "action": "suspend_fsc_claims"},
            ],
            "exemption_process": "written_request_to_cb",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 3.0,
            "variance_warning_pct": 7.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "fsc_coordinator"],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "FSC",
        "reference_document": "FSC-STD-40-004 V3-0, Clause 7",
        "notes": (
            "FSC uses annual periods aligned with certification anniversary. "
            "Credits do not expire within the active period."
        ),
    },

    # ---- ISCC 202 v4.0 ----
    "ISCC-202": {
        "name": "ISCC Chain of Custody Requirements",
        "version": "4.0",
        "effective_date": "2024-03-01",
        "applicable_commodities": ["oil_palm", "soya", "wood", "rubber"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "calendar_year",
        "period_start_rule": "january_first",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "end_of_receiving_period",
            "expiry_description": (
                "Credits carried forward expire at the end of the "
                "receiving period.  Unused credits are voided."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 5,
            "calendar_days": 7,
            "purpose": "period_end_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "percentage",
            "tolerance_percentage": 5.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 72,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_sustainability_manager"},
                {"hours": 72, "action": "restrict_output_claims"},
            ],
            "exemption_process": "documented_justification_to_cb",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 5.0,
            "variance_warning_pct": 10.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "iscc_coordinator"],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "ISCC",
        "reference_document": "ISCC 202 V4.0, Section 4.3",
        "notes": (
            "ISCC allows 5% overdraft tolerance within a period. "
            "Annual periods aligned to calendar year."
        ),
    },

    # ---- UTZ / Rainforest Alliance CoC ----
    "UTZ-RA-CoC": {
        "name": "UTZ/Rainforest Alliance Chain of Custody Standard",
        "version": "2.0",
        "effective_date": "2022-07-01",
        "applicable_commodities": ["cocoa", "coffee"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "harvest_year",
        "period_start_rule": "harvest_start_date",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 50.0,
            "expiry_rule": "end_of_receiving_period",
            "expiry_description": (
                "Maximum 50% of closing balance may be carried forward. "
                "Credits expire at end of receiving period."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 5,
            "calendar_days": 7,
            "purpose": "period_end_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 48,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_ra_coordinator"},
                {"hours": 72, "action": "suspend_ra_claims"},
            ],
            "exemption_process": "formal_request_with_evidence",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 3.0,
            "variance_warning_pct": 7.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "ra_coordinator"],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "Rainforest Alliance",
        "reference_document": "RA CoC Standard 2.0, Chapter 4",
        "notes": (
            "UTZ/RA limits carry-forward to 50% of closing balance. "
            "Period aligned to harvest year."
        ),
    },

    # ---- Fairtrade SOP ----
    "Fairtrade": {
        "name": "Fairtrade Standard Operating Procedures for Mass Balance",
        "version": "1.0",
        "effective_date": "2023-01-01",
        "applicable_commodities": ["cocoa", "coffee", "soya"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "calendar_year",
        "period_start_rule": "january_first",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 25.0,
            "expiry_rule": "end_of_receiving_period",
            "expiry_description": (
                "Maximum 25% of closing balance may be carried forward. "
                "Fairtrade imposes the strictest CF limit."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 10,
            "calendar_days": 14,
            "purpose": "annual_reconciliation_and_reporting",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
                "fairtrade_reporting",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 72,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_fairtrade_coordinator"},
                {"hours": 72, "action": "suspend_fairtrade_claims"},
            ],
            "exemption_process": "written_request_to_flocert",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 2.0,
            "variance_warning_pct": 5.0,
            "sign_off_required": True,
            "sign_off_roles": [
                "facility_manager",
                "fairtrade_officer",
            ],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "Fairtrade International",
        "reference_document": "Fairtrade CoC SOP v1.0, Section 5",
        "notes": (
            "Fairtrade limits carry-forward to 25%. Strictest CF limit "
            "among all standards. Zero overdraft tolerance."
        ),
    },

    # ---- EUDR Default (no specific certification overlay) ----
    "EUDR-DEFAULT": {
        "name": "EUDR Default Mass Balance Rules",
        "version": "1.0",
        "effective_date": "2025-12-30",
        "applicable_commodities": [
            "cocoa", "coffee", "oil_palm", "soya",
            "rubber", "wood", "cattle",
        ],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "calendar_year",
        "period_start_rule": "january_first",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "configurable",
            "expiry_description": (
                "EUDR default allows carry-forward with operator-configurable "
                "expiry.  Recommended: expire at end of receiving period."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 5,
            "calendar_days": 7,
            "purpose": "period_end_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 48,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_compliance_officer"},
                {"hours": 72, "action": "block_due_diligence_statements"},
            ],
            "exemption_process": "documented_justification",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 2.0,
            "variance_warning_pct": 5.0,
            "sign_off_required": True,
            "sign_off_roles": [
                "facility_manager",
                "eudr_compliance_officer",
            ],
            "audit_trail_retention_years": 5,
            "third_party_verification": False,
        },
        "source": "EU",
        "reference_document": "EU 2023/1115, Article 10(2)(f)",
        "notes": (
            "EUDR default rules when no specific certification scheme overlay "
            "is selected.  Provides baseline compliance."
        ),
    },

    # ---- PEFC Chain of Custody ----
    "PEFC-CoC": {
        "name": "PEFC Chain of Custody Standard",
        "version": "2020",
        "effective_date": "2020-02-14",
        "applicable_commodities": ["wood"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "certification_anniversary",
        "period_start_rule": "certification_start_date",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "no_expiry_within_active_period",
            "expiry_description": (
                "Credits do not expire during the active certification "
                "period.  Similar to FSC approach."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 10,
            "calendar_days": 14,
            "purpose": "annual_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "zero_tolerance",
            "tolerance_percentage": 0.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 120,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 48, "action": "alert_facility_manager"},
                {"hours": 96, "action": "alert_certification_body"},
                {"hours": 120, "action": "suspend_pefc_claims"},
            ],
            "exemption_process": "written_request_to_notified_body",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 3.0,
            "variance_warning_pct": 7.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "pefc_coordinator"],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "PEFC",
        "reference_document": "PEFC ST 2002:2020, Annex 3",
        "notes": (
            "PEFC follows similar rules to FSC with annual periods "
            "and no in-period credit expiry."
        ),
    },

    # ---- Bonsucro (sugarcane, for future EUDR commodity extension) ----
    "Bonsucro": {
        "name": "Bonsucro Mass Balance Chain of Custody",
        "version": "5.2",
        "effective_date": "2022-01-01",
        "applicable_commodities": ["soya"],
        "period_duration_months": 12,
        "period_duration_days": 365,
        "period_alignment": "calendar_year",
        "period_start_rule": "january_first",
        "carry_forward": {
            "allowed": True,
            "max_percentage": 100.0,
            "expiry_rule": "end_of_receiving_period",
            "expiry_description": (
                "Credits carried forward expire at end of receiving period."
            ),
            "partial_utilization": True,
            "fifo_required": True,
            "audit_trail_required": True,
            "separate_line_item": True,
        },
        "grace_period": {
            "business_days": 5,
            "calendar_days": 7,
            "purpose": "period_end_reconciliation",
            "activities_allowed": [
                "late_entry_recording",
                "correction_of_errors",
                "reconciliation_finalization",
            ],
            "new_transactions_allowed": False,
        },
        "overdraft": {
            "tolerance_mode": "percentage",
            "tolerance_percentage": 2.0,
            "tolerance_absolute_kg": 0.0,
            "resolution_deadline_hours": 72,
            "auto_escalation": True,
            "escalation_levels": [
                {"hours": 24, "action": "alert_facility_manager"},
                {"hours": 48, "action": "alert_sustainability_manager"},
                {"hours": 72, "action": "restrict_claims"},
            ],
            "exemption_process": "formal_request_to_bonsucro",
        },
        "reconciliation": {
            "mandatory": True,
            "frequency": "annual",
            "variance_acceptable_pct": 3.0,
            "variance_warning_pct": 7.0,
            "sign_off_required": True,
            "sign_off_roles": ["facility_manager", "bonsucro_coordinator"],
            "audit_trail_retention_years": 5,
            "third_party_verification": True,
        },
        "source": "Bonsucro",
        "reference_document": "Bonsucro CoC Standard 5.2, Section 6",
        "notes": (
            "Bonsucro allows 2% overdraft tolerance. Applicable to soya "
            "under EUDR commodity scope."
        ),
    },
}

# ---------------------------------------------------------------------------
# Total counts
# ---------------------------------------------------------------------------

TOTAL_STANDARDS: int = len(CREDIT_PERIOD_RULES)

# ---------------------------------------------------------------------------
# Derived lookup structures
# ---------------------------------------------------------------------------

CARRY_FORWARD_RULES: Dict[str, Dict[str, Any]] = {
    standard_id: rule_set["carry_forward"]
    for standard_id, rule_set in CREDIT_PERIOD_RULES.items()
}

OVERDRAFT_POLICIES: Dict[str, Dict[str, Any]] = {
    standard_id: rule_set["overdraft"]
    for standard_id, rule_set in CREDIT_PERIOD_RULES.items()
}

RECONCILIATION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    standard_id: rule_set["reconciliation"]
    for standard_id, rule_set in CREDIT_PERIOD_RULES.items()
}

GRACE_PERIOD_RULES: Dict[str, Dict[str, Any]] = {
    standard_id: rule_set["grace_period"]
    for standard_id, rule_set in CREDIT_PERIOD_RULES.items()
}

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_period_rules(standard: str) -> Optional[Dict[str, Any]]:
    """Return complete credit period rules for a certification standard.

    Args:
        standard: Certification standard identifier (e.g. "RSPO-SCC",
            "FSC-STD-40-004", "ISCC-202", "EUDR-DEFAULT").

    Returns:
        Dictionary with full period rules including duration,
        carry_forward, grace_period, overdraft, and reconciliation.
        Returns None if standard is not found.

    Example:
        >>> rules = get_period_rules("RSPO-SCC")
        >>> rules["period_duration_months"]
        3
    """
    return CREDIT_PERIOD_RULES.get(standard)


def get_all_rules() -> List[Dict[str, Any]]:
    """Return all credit period rule sets with their standard identifiers.

    Returns:
        List of dictionaries, each containing "standard_id" plus all
        rule fields.

    Example:
        >>> rules = get_all_rules()
        >>> len(rules) >= 6
        True
    """
    result: List[Dict[str, Any]] = []
    for standard_id, rule_set in CREDIT_PERIOD_RULES.items():
        entry = {"standard_id": standard_id}
        entry.update(rule_set)
        result.append(entry)
    return result


def get_carry_forward_rules(standard: str) -> Optional[Dict[str, Any]]:
    """Return carry-forward rules for a specific standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Dictionary with allowed, max_percentage, expiry_rule,
        partial_utilization, fifo_required, etc.
        Returns None if standard is not found.

    Example:
        >>> cf = get_carry_forward_rules("UTZ-RA-CoC")
        >>> cf["max_percentage"]
        50.0
    """
    return CARRY_FORWARD_RULES.get(standard)


def get_grace_period(standard: str) -> int:
    """Return the grace period in business days for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Grace period in business days.  Defaults to 5 if standard
        is not found.

    Example:
        >>> get_grace_period("FSC-STD-40-004")
        10
    """
    rules = CREDIT_PERIOD_RULES.get(standard)
    if rules is None:
        return 5
    return int(rules.get("grace_period", {}).get("business_days", 5))


def get_grace_period_calendar_days(standard: str) -> int:
    """Return the grace period in calendar days for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Grace period in calendar days.  Defaults to 7 if standard
        is not found.

    Example:
        >>> get_grace_period_calendar_days("Fairtrade")
        14
    """
    rules = CREDIT_PERIOD_RULES.get(standard)
    if rules is None:
        return 7
    return int(rules.get("grace_period", {}).get("calendar_days", 7))


def get_overdraft_tolerance(standard: str) -> Optional[Dict[str, Any]]:
    """Return overdraft tolerance policy for a specific standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Dictionary with tolerance_mode, tolerance_percentage,
        resolution_deadline_hours, and escalation_levels.
        Returns None if standard is not found.

    Example:
        >>> od = get_overdraft_tolerance("ISCC-202")
        >>> od["tolerance_percentage"]
        5.0
    """
    return OVERDRAFT_POLICIES.get(standard)


def get_period_duration_months(standard: str) -> int:
    """Return the credit period duration in months for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Period duration in months.  Defaults to 12 if not found.

    Example:
        >>> get_period_duration_months("RSPO-SCC")
        3
    """
    rules = CREDIT_PERIOD_RULES.get(standard)
    if rules is None:
        return 12
    return int(rules.get("period_duration_months", 12))


def get_period_duration_days(standard: str) -> int:
    """Return the credit period duration in days for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Period duration in days.  Defaults to 365 if not found.

    Example:
        >>> get_period_duration_days("RSPO-SCC")
        90
    """
    rules = CREDIT_PERIOD_RULES.get(standard)
    if rules is None:
        return 365
    return int(rules.get("period_duration_days", 365))


def get_reconciliation_requirements(
    standard: str,
) -> Optional[Dict[str, Any]]:
    """Return reconciliation requirements for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Dictionary with variance thresholds, sign-off requirements,
        and audit trail mandates.  Returns None if not found.

    Example:
        >>> recon = get_reconciliation_requirements("RSPO-SCC")
        >>> recon["variance_acceptable_pct"]
        2.0
    """
    return RECONCILIATION_REQUIREMENTS.get(standard)


def get_max_carry_forward_pct(standard: str) -> float:
    """Return the maximum carry-forward percentage for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        Maximum carry-forward percentage (0-100).  Defaults to 100.0
        if not found.

    Example:
        >>> get_max_carry_forward_pct("Fairtrade")
        25.0
    """
    cf = get_carry_forward_rules(standard)
    if cf is None:
        return 100.0
    return float(cf.get("max_percentage", 100.0))


def is_carry_forward_allowed(standard: str) -> bool:
    """Check whether carry-forward is allowed for a standard.

    Args:
        standard: Certification standard identifier.

    Returns:
        True if carry-forward is allowed, False otherwise.
        Defaults to True if standard is not found.

    Example:
        >>> is_carry_forward_allowed("RSPO-SCC")
        True
    """
    cf = get_carry_forward_rules(standard)
    if cf is None:
        return True
    return bool(cf.get("allowed", True))


def is_zero_overdraft(standard: str) -> bool:
    """Check whether a standard enforces zero overdraft tolerance.

    Args:
        standard: Certification standard identifier.

    Returns:
        True if zero_tolerance mode, False otherwise.
        Defaults to True (fail-safe) if standard is not found.

    Example:
        >>> is_zero_overdraft("RSPO-SCC")
        True
        >>> is_zero_overdraft("ISCC-202")
        False
    """
    od = get_overdraft_tolerance(standard)
    if od is None:
        return True
    return od.get("tolerance_mode") == "zero_tolerance"


def get_all_standard_ids() -> List[str]:
    """Return all supported standard identifiers.

    Returns:
        Sorted list of standard identifier strings.

    Example:
        >>> ids = get_all_standard_ids()
        >>> "EUDR-DEFAULT" in ids
        True
    """
    return sorted(CREDIT_PERIOD_RULES.keys())


def get_standards_for_commodity(commodity: str) -> List[str]:
    """Return all standard IDs applicable to a given commodity.

    Args:
        commodity: EUDR commodity identifier.

    Returns:
        List of standard IDs whose applicable_commodities include
        the given commodity.

    Example:
        >>> stds = get_standards_for_commodity("wood")
        >>> "FSC-STD-40-004" in stds
        True
    """
    return [
        standard_id
        for standard_id, rule_set in CREDIT_PERIOD_RULES.items()
        if commodity in rule_set.get("applicable_commodities", [])
    ]


def get_resolution_deadline_hours(standard: str) -> int:
    """Return the overdraft resolution deadline in hours.

    Args:
        standard: Certification standard identifier.

    Returns:
        Resolution deadline in hours.  Defaults to 48 if not found.

    Example:
        >>> get_resolution_deadline_hours("FSC-STD-40-004")
        120
    """
    od = get_overdraft_tolerance(standard)
    if od is None:
        return 48
    return int(od.get("resolution_deadline_hours", 48))


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Credit period rules reference data loaded: %d standards",
    TOTAL_STANDARDS,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "CREDIT_PERIOD_RULES",
    "CARRY_FORWARD_RULES",
    "OVERDRAFT_POLICIES",
    "RECONCILIATION_REQUIREMENTS",
    "GRACE_PERIOD_RULES",
    # Counts
    "TOTAL_STANDARDS",
    # Lookup helpers
    "get_period_rules",
    "get_all_rules",
    "get_carry_forward_rules",
    "get_grace_period",
    "get_grace_period_calendar_days",
    "get_overdraft_tolerance",
    "get_period_duration_months",
    "get_period_duration_days",
    "get_reconciliation_requirements",
    "get_max_carry_forward_pct",
    "is_carry_forward_allowed",
    "is_zero_overdraft",
    "get_all_standard_ids",
    "get_standards_for_commodity",
    "get_resolution_deadline_hours",
]
