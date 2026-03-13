# -*- coding: utf-8 -*-
"""
Chain of Custody Model Rules - AGENT-EUDR-009 Chain of Custody Agent

Provides validation rules, certification applicability, credit period
definitions, model hierarchy, and cross-model handoff rules for the four
ISO 22095 Chain of Custody models used in EUDR compliance:

    - IP (Identity Preserved): Strictest; no mixing, single origin.
    - SG (Segregated): Compliant-only; pooled from multiple compliant origins.
    - MB (Mass Balance): Accounting-based; physical mixing with credit tracking.
    - CB (Controlled Blend): Percentage-based blending of compliant material.

These deterministic rules drive the CoCModelEnforcer engine, ensuring
zero-hallucination model compliance validation, downgrade detection, and
cross-facility handoff handling.

Data Sources:
    ISO 22095:2020 Chain of Custody - General Terminology and Models
    FSC-STD-40-004 V3-1 (Chain of Custody Certification)
    RSPO Supply Chain Certification Standard 2020
    ISCC EU 204:2023 (Mass Balance Requirements)
    UTZ/RA Chain of Custody Standard 2022

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Chain of Custody)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data version for provenance tracking
# ---------------------------------------------------------------------------

DATA_VERSION = "1.0.0"
DATA_SOURCE = "GreenLang Reference Data v1.0.0 (2026-03)"

# ---------------------------------------------------------------------------
# CoC Model Definitions and Rules
# ---------------------------------------------------------------------------
# Four models per ISO 22095:
#   IP  = Identity Preserved (strictest)
#   SG  = Segregated
#   MB  = Mass Balance
#   CB  = Controlled Blend (least strict)
#
# Each model entry contains:
#   name, description, iso_reference, no_mixing, single_origin,
#   physical_separation, compliant_only, pool_mixing, accounting_based,
#   physical_mixing, max_blend_ratio, traceability_pct,
#   default_credit_period_months, applicable_certifications,
#   eudr_acceptance, key_requirements

COC_MODEL_RULES: Dict[str, Dict[str, Any]] = {
    "IP": {
        "name": "Identity Preserved",
        "model_code": "IP",
        "description": (
            "The most stringent CoC model. Material from a single "
            "certified source is kept physically separate throughout "
            "the entire supply chain. No mixing with material from "
            "any other source is permitted at any point."
        ),
        "iso_reference": "ISO 22095:2020 Section 6.2",
        # Model characteristics
        "no_mixing": True,
        "single_origin": True,
        "physical_separation": True,
        "compliant_only": True,
        "pool_mixing": False,
        "accounting_based": False,
        "physical_mixing": False,
        "max_blend_ratio": None,
        "traceability_pct": 100.0,
        "default_credit_period_months": 0,
        # Applicable certifications
        "applicable_certifications": [
            "FSC_FM", "FSC_COC",
            "RSPO_IP",
        ],
        # EUDR alignment
        "eudr_acceptance": "full",
        "eudr_acceptance_notes": (
            "IP model provides the highest level of traceability "
            "required by EUDR. Material can be traced to a single "
            "production plot, fully meeting Article 9(1)(d)."
        ),
        # Key requirements for compliance
        "key_requirements": [
            "Material must originate from a single certified source",
            "Physical separation from all other material at all times",
            "No blending, mixing, or substitution permitted",
            "Full traceability to production plot",
            "Separate storage, transport, and processing",
            "Unique lot/batch identification maintained throughout",
            "Volume reconciliation at each transfer point",
            "Document chain linking output to specific input source",
        ],
        # Validation rules (machine-readable)
        "validation_rules": {
            "max_sources_per_batch": 1,
            "mixing_allowed": False,
            "substitution_allowed": False,
            "credit_trading_allowed": False,
            "mass_balance_required": True,
            "physical_separation_required": True,
            "lot_tracking_required": True,
            "yield_verification_required": True,
            "overdraft_tolerance_pct": 0.0,
        },
    },

    "SG": {
        "name": "Segregated",
        "model_code": "SG",
        "description": (
            "Material from multiple certified/compliant sources may be "
            "mixed together, but must remain physically separate from "
            "non-certified/non-compliant material. All material in a "
            "segregated batch is compliant."
        ),
        "iso_reference": "ISO 22095:2020 Section 6.3",
        # Model characteristics
        "no_mixing": False,
        "single_origin": False,
        "physical_separation": True,
        "compliant_only": True,
        "pool_mixing": True,
        "accounting_based": False,
        "physical_mixing": False,
        "max_blend_ratio": None,
        "traceability_pct": 100.0,
        "default_credit_period_months": 0,
        # Applicable certifications
        "applicable_certifications": [
            "FSC_FM", "FSC_COC",
            "RSPO_SG",
            "ISCC_EU",
        ],
        # EUDR alignment
        "eudr_acceptance": "full",
        "eudr_acceptance_notes": (
            "SG model is accepted for EUDR compliance when all "
            "contributing sources have verified geolocation data "
            "and deforestation-free status."
        ),
        # Key requirements
        "key_requirements": [
            "All material must be from certified/compliant sources",
            "Physical separation from non-compliant material",
            "Multiple compliant sources may be pooled",
            "Each source must have verified geolocation data",
            "Volume reconciliation across all sources",
            "Batch must track contributing source identifiers",
            "No non-compliant material may enter the segregated pool",
            "Processing must maintain segregation from conventional",
        ],
        # Validation rules
        "validation_rules": {
            "max_sources_per_batch": None,
            "mixing_allowed": True,
            "substitution_allowed": False,
            "credit_trading_allowed": False,
            "mass_balance_required": True,
            "physical_separation_required": True,
            "lot_tracking_required": True,
            "yield_verification_required": True,
            "overdraft_tolerance_pct": 0.0,
            "all_sources_must_be_compliant": True,
        },
    },

    "MB": {
        "name": "Mass Balance",
        "model_code": "MB",
        "description": (
            "Accounting-based model where certified and non-certified "
            "material may be physically mixed. A credit system tracks "
            "the volume of certified material entering the system and "
            "allows an equivalent volume to be sold as certified on "
            "output. The physical output may not contain any specific "
            "certified material."
        ),
        "iso_reference": "ISO 22095:2020 Section 6.4",
        # Model characteristics
        "no_mixing": False,
        "single_origin": False,
        "physical_separation": False,
        "compliant_only": False,
        "pool_mixing": True,
        "accounting_based": True,
        "physical_mixing": True,
        "max_blend_ratio": None,
        "traceability_pct": 0.0,
        "default_credit_period_months": 12,
        # Applicable certifications
        "applicable_certifications": [
            "RSPO_MB",
            "ISCC_EU",
            "ISCC_PLUS",
            "UTZ",
            "RAINFOREST_ALLIANCE",
        ],
        # EUDR alignment
        "eudr_acceptance": "conditional",
        "eudr_acceptance_notes": (
            "MB model is conditionally accepted for EUDR. Operators "
            "must demonstrate that the mass balance system is audited, "
            "credits are not over-sold, and geolocation data covers "
            "100% of certified input volumes. Enhanced scrutiny applies "
            "for high-risk origin countries."
        ),
        # Key requirements
        "key_requirements": [
            "Credit-based tracking of certified input volumes",
            "Output claims limited by credit balance (no overdraft)",
            "Credit period must not exceed certification scheme limit",
            "Regular reconciliation of input vs output volumes",
            "Conversion/yield factors applied to credits",
            "Credits expire at end of credit period",
            "Third-party audit of mass balance accounts",
            "Geolocation data required for all certified inputs",
        ],
        # Validation rules
        "validation_rules": {
            "max_sources_per_batch": None,
            "mixing_allowed": True,
            "substitution_allowed": False,
            "credit_trading_allowed": True,
            "mass_balance_required": True,
            "physical_separation_required": False,
            "lot_tracking_required": False,
            "yield_verification_required": True,
            "overdraft_tolerance_pct": 0.0,
            "credit_period_enforced": True,
            "credit_expiry_enforced": True,
            "conversion_factor_required": True,
        },
    },

    "CB": {
        "name": "Controlled Blend",
        "model_code": "CB",
        "description": (
            "Percentage-based model where a known proportion of "
            "certified/compliant material is blended with conventional "
            "material. The output is labeled with the certified "
            "percentage (e.g., 'contains 40% RSPO certified palm oil')."
        ),
        "iso_reference": "ISO 22095:2020 Section 6.5 (Book and Claim variant)",
        # Model characteristics
        "no_mixing": False,
        "single_origin": False,
        "physical_separation": False,
        "compliant_only": False,
        "pool_mixing": True,
        "accounting_based": True,
        "physical_mixing": True,
        "max_blend_ratio": 1.0,
        "traceability_pct": 0.0,
        "default_credit_period_months": 12,
        # Applicable certifications
        "applicable_certifications": [
            "RSPO_BC",
        ],
        # EUDR alignment
        "eudr_acceptance": "limited",
        "eudr_acceptance_notes": (
            "CB model has limited acceptance for EUDR purposes. "
            "Only the certified portion may count toward EUDR "
            "compliance. The non-certified portion must be treated "
            "as requiring separate due diligence. Not recommended "
            "for high-risk origin countries."
        ),
        # Key requirements
        "key_requirements": [
            "Declared blend percentage must be verified",
            "Certified portion must have full traceability",
            "Non-certified portion requires separate due diligence",
            "Blend ratio must be accurately communicated to buyer",
            "Regular verification of blend percentages",
            "Minimum certified content may apply per scheme",
            "Clear labeling of certified content percentage",
            "Annual third-party audit of blend calculations",
        ],
        # Validation rules
        "validation_rules": {
            "max_sources_per_batch": None,
            "mixing_allowed": True,
            "substitution_allowed": False,
            "credit_trading_allowed": True,
            "mass_balance_required": True,
            "physical_separation_required": False,
            "lot_tracking_required": False,
            "yield_verification_required": True,
            "overdraft_tolerance_pct": 0.0,
            "blend_ratio_required": True,
            "minimum_certified_pct": 0.0,
        },
    },
}

TOTAL_MODELS = len(COC_MODEL_RULES)

# ---------------------------------------------------------------------------
# Credit Period by Certification Scheme
# ---------------------------------------------------------------------------
# Different certification schemes impose different credit period limits
# for mass balance accounting. Credits must be used (sold) within the
# credit period or they expire.

CREDIT_PERIOD_BY_CERTIFICATION: Dict[str, Dict[str, Any]] = {
    "FSC_FM": {
        "scheme": "FSC",
        "credit_period_months": 12,
        "description": "FSC allows 12-month credit period for CoC claims",
        "source": "FSC-STD-40-004 V3-1 Section 8.4",
        "rollover_allowed": False,
        "notes": (
            "Credits not used within 12 months expire. No rollover "
            "to the next period. Accounts reconciled annually."
        ),
    },
    "FSC_COC": {
        "scheme": "FSC",
        "credit_period_months": 12,
        "description": "FSC Chain of Custody credit period",
        "source": "FSC-STD-40-004 V3-1 Section 8.4",
        "rollover_allowed": False,
        "notes": "Same as FSC FM; 12-month period.",
    },
    "RSPO_IP": {
        "scheme": "RSPO",
        "credit_period_months": 3,
        "description": "RSPO Identity Preserved - 3-month credit window",
        "source": "RSPO SCC Standard 2020 Section 4.3.4",
        "rollover_allowed": False,
        "notes": (
            "RSPO uses quarterly (3-month) reconciliation periods. "
            "Shorter than FSC to prevent excessive credit accumulation."
        ),
    },
    "RSPO_SG": {
        "scheme": "RSPO",
        "credit_period_months": 3,
        "description": "RSPO Segregated - 3-month credit window",
        "source": "RSPO SCC Standard 2020 Section 4.3.4",
        "rollover_allowed": False,
        "notes": "Same quarterly period as RSPO IP.",
    },
    "RSPO_MB": {
        "scheme": "RSPO",
        "credit_period_months": 3,
        "description": "RSPO Mass Balance - 3-month credit window",
        "source": "RSPO SCC Standard 2020 Section 4.3.4",
        "rollover_allowed": False,
        "notes": (
            "RSPO MB credits must be used within the same quarter. "
            "This is stricter than FSC and ISCC."
        ),
    },
    "RSPO_BC": {
        "scheme": "RSPO",
        "credit_period_months": 3,
        "description": "RSPO Book & Claim (Controlled Blend variant)",
        "source": "RSPO SCC Standard 2020 Section 4.3.4",
        "rollover_allowed": False,
        "notes": "RSPO BC credits follow the same quarterly cycle.",
    },
    "ISCC_EU": {
        "scheme": "ISCC",
        "credit_period_months": 12,
        "description": "ISCC EU mass balance credit period",
        "source": "ISCC EU 204:2023 Section 3.2",
        "rollover_allowed": False,
        "notes": (
            "ISCC allows 12-month (annual) credit periods. Credits "
            "cannot be transferred between sites."
        ),
    },
    "ISCC_PLUS": {
        "scheme": "ISCC",
        "credit_period_months": 12,
        "description": "ISCC PLUS mass balance credit period",
        "source": "ISCC PLUS 204:2023 Section 3.2",
        "rollover_allowed": False,
        "notes": "Same as ISCC EU; 12-month period.",
    },
    "UTZ": {
        "scheme": "UTZ/Rainforest Alliance",
        "credit_period_months": 12,
        "description": "UTZ (now RA) mass balance credit period",
        "source": "Rainforest Alliance CoC Standard 2022 Section 5.3",
        "rollover_allowed": False,
        "notes": (
            "UTZ/RA uses annual credit periods aligned with the "
            "certification cycle. Credits tied to specific origin."
        ),
    },
    "RAINFOREST_ALLIANCE": {
        "scheme": "Rainforest Alliance",
        "credit_period_months": 12,
        "description": "Rainforest Alliance CoC credit period",
        "source": "Rainforest Alliance CoC Standard 2022 Section 5.3",
        "rollover_allowed": False,
        "notes": "Same as UTZ; 12-month annual period.",
    },
}

TOTAL_CREDIT_PERIODS = len(CREDIT_PERIOD_BY_CERTIFICATION)

# ---------------------------------------------------------------------------
# Model Hierarchy
# ---------------------------------------------------------------------------
# IP > SG > MB > CB in terms of traceability stringency.
# Material may be downgraded (IP -> SG -> MB -> CB) but not upgraded.
# Hierarchy level: 1 = most stringent, 4 = least stringent.

MODEL_HIERARCHY: Dict[str, Dict[str, Any]] = {
    "IP": {
        "level": 1,
        "name": "Identity Preserved",
        "can_downgrade_to": ["SG", "MB", "CB"],
        "can_upgrade_from": [],
        "notes": (
            "IP material can be downgraded to any less-strict model "
            "but cannot be recovered once mixed."
        ),
    },
    "SG": {
        "level": 2,
        "name": "Segregated",
        "can_downgrade_to": ["MB", "CB"],
        "can_upgrade_from": [],
        "notes": (
            "SG material can be downgraded to MB or CB. Cannot be "
            "upgraded to IP since origin uniqueness is lost."
        ),
    },
    "MB": {
        "level": 3,
        "name": "Mass Balance",
        "can_downgrade_to": ["CB"],
        "can_upgrade_from": [],
        "notes": (
            "MB material can only be downgraded to CB. Physical "
            "mixing makes upgrade impossible."
        ),
    },
    "CB": {
        "level": 4,
        "name": "Controlled Blend",
        "can_downgrade_to": [],
        "can_upgrade_from": [],
        "notes": (
            "CB is the least-strict model. No further downgrade "
            "possible. Cannot be upgraded."
        ),
    },
}

# ---------------------------------------------------------------------------
# Cross-Model Handoff Rules
# ---------------------------------------------------------------------------
# When material moves between facilities operating under different CoC
# models, specific rules apply. These rules prevent unintended model
# upgrades and ensure audit trail integrity.

CROSS_MODEL_HANDOFF_RULES: Dict[str, Dict[str, Any]] = {
    "IP_to_IP": {
        "from_model": "IP",
        "to_model": "IP",
        "allowed": True,
        "result_model": "IP",
        "conditions": [
            "Single-origin identity must be maintained",
            "No mixing with other material at receiving facility",
            "Physical separation verified at receiving dock",
        ],
        "documentation_required": [
            "Transfer certificate with single-origin attestation",
            "Lot-level traceability linkage",
        ],
        "risk_level": "low",
    },
    "IP_to_SG": {
        "from_model": "IP",
        "to_model": "SG",
        "allowed": True,
        "result_model": "SG",
        "conditions": [
            "IP material downgrades to SG upon pooling",
            "Receiving facility must hold SG or higher certification",
            "Geolocation data from IP source retained",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from IP to SG",
            "Source geolocation data package",
        ],
        "risk_level": "low",
    },
    "IP_to_MB": {
        "from_model": "IP",
        "to_model": "MB",
        "allowed": True,
        "result_model": "MB",
        "conditions": [
            "IP material downgrades to MB upon physical mixing",
            "Credit balance updated with input volume",
            "Identity is irrecoverably lost",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from IP to MB",
            "Mass balance ledger entry",
        ],
        "risk_level": "medium",
    },
    "IP_to_CB": {
        "from_model": "IP",
        "to_model": "CB",
        "allowed": True,
        "result_model": "CB",
        "conditions": [
            "IP material downgrades to CB",
            "Certified percentage must be tracked",
            "Maximum downgrade in single step",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from IP to CB",
            "Blend ratio calculation record",
        ],
        "risk_level": "medium",
    },
    "SG_to_IP": {
        "from_model": "SG",
        "to_model": "IP",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: SG cannot be upgraded to IP",
            "Single-origin identity cannot be recovered once pooled",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "SG_to_SG": {
        "from_model": "SG",
        "to_model": "SG",
        "allowed": True,
        "result_model": "SG",
        "conditions": [
            "Both facilities must hold SG certification",
            "All material in pool must be compliant",
            "Physical separation from non-compliant maintained",
        ],
        "documentation_required": [
            "Transfer certificate confirming SG status",
            "Source compliance verification",
        ],
        "risk_level": "low",
    },
    "SG_to_MB": {
        "from_model": "SG",
        "to_model": "MB",
        "allowed": True,
        "result_model": "MB",
        "conditions": [
            "SG material downgrades to MB upon physical mixing",
            "Credit balance updated",
            "Segregation identity lost",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from SG to MB",
            "Mass balance ledger entry",
        ],
        "risk_level": "medium",
    },
    "SG_to_CB": {
        "from_model": "SG",
        "to_model": "CB",
        "allowed": True,
        "result_model": "CB",
        "conditions": [
            "SG material downgrades to CB",
            "Certified percentage tracked",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from SG to CB",
            "Blend ratio calculation",
        ],
        "risk_level": "medium",
    },
    "MB_to_IP": {
        "from_model": "MB",
        "to_model": "IP",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: MB cannot be upgraded to IP",
            "Physical mixing has destroyed identity",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "MB_to_SG": {
        "from_model": "MB",
        "to_model": "SG",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: MB cannot be upgraded to SG",
            "Physical mixing has destroyed segregation",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "MB_to_MB": {
        "from_model": "MB",
        "to_model": "MB",
        "allowed": True,
        "result_model": "MB",
        "conditions": [
            "Both facilities must operate mass balance systems",
            "Credit transfer between sites not allowed (per most schemes)",
            "Receiving site creates new credit from input volume",
        ],
        "documentation_required": [
            "Transfer certificate with mass balance credit details",
            "Mass balance ledger entries at both sites",
        ],
        "risk_level": "low",
    },
    "MB_to_CB": {
        "from_model": "MB",
        "to_model": "CB",
        "allowed": True,
        "result_model": "CB",
        "conditions": [
            "MB material downgrades to CB",
            "Only the credit-backed portion counts as certified",
        ],
        "documentation_required": [
            "Transfer certificate noting downgrade from MB to CB",
            "Credit-backed volume documentation",
        ],
        "risk_level": "medium",
    },
    "CB_to_IP": {
        "from_model": "CB",
        "to_model": "IP",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: CB cannot be upgraded to IP",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "CB_to_SG": {
        "from_model": "CB",
        "to_model": "SG",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: CB cannot be upgraded to SG",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "CB_to_MB": {
        "from_model": "CB",
        "to_model": "MB",
        "allowed": False,
        "result_model": None,
        "conditions": [
            "UPGRADE NOT ALLOWED: CB cannot be upgraded to MB",
        ],
        "documentation_required": [],
        "risk_level": "critical",
    },
    "CB_to_CB": {
        "from_model": "CB",
        "to_model": "CB",
        "allowed": True,
        "result_model": "CB",
        "conditions": [
            "Blend percentages must be recalculated",
            "Receiving facility tracks certified percentage",
        ],
        "documentation_required": [
            "Transfer certificate with blend ratio",
            "Updated blend calculation at receiving site",
        ],
        "risk_level": "low",
    },
}

TOTAL_HANDOFF_RULES = len(CROSS_MODEL_HANDOFF_RULES)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------


def get_model_rules(model_code: str) -> Optional[Dict[str, Any]]:
    """Return the full rule set for a CoC model.

    Args:
        model_code: Model code ('IP', 'SG', 'MB', or 'CB').

    Returns:
        Dictionary of model rules, or None if not found.

    Example:
        >>> rules = get_model_rules("IP")
        >>> rules["no_mixing"]
        True
    """
    return COC_MODEL_RULES.get(model_code.upper())


def get_credit_period(certification: str) -> Optional[int]:
    """Return the credit period in months for a certification scheme.

    Args:
        certification: Certification code (e.g., 'RSPO_MB', 'FSC_COC').

    Returns:
        Credit period in months, or None if not found.

    Example:
        >>> get_credit_period("RSPO_MB")
        3
        >>> get_credit_period("FSC_COC")
        12
    """
    entry = CREDIT_PERIOD_BY_CERTIFICATION.get(certification)
    if entry is None:
        return None
    return entry["credit_period_months"]


def validate_handoff(
    from_model: str,
    to_model: str,
) -> Dict[str, Any]:
    """Validate whether a cross-model handoff is allowed.

    Args:
        from_model: Source facility CoC model code.
        to_model: Destination facility CoC model code.

    Returns:
        Dictionary with: allowed (bool), result_model, conditions,
        documentation_required, risk_level.

    Example:
        >>> result = validate_handoff("IP", "SG")
        >>> result["allowed"]
        True
        >>> result["result_model"]
        'SG'
        >>> result = validate_handoff("MB", "IP")
        >>> result["allowed"]
        False
    """
    key = f"{from_model.upper()}_to_{to_model.upper()}"
    rule = CROSS_MODEL_HANDOFF_RULES.get(key)

    if rule is None:
        return {
            "allowed": False,
            "from_model": from_model.upper(),
            "to_model": to_model.upper(),
            "result_model": None,
            "conditions": [f"No handoff rule defined for {key}"],
            "documentation_required": [],
            "risk_level": "unknown",
            "message": f"Unknown model combination: {key}",
        }

    return {
        "allowed": rule["allowed"],
        "from_model": rule["from_model"],
        "to_model": rule["to_model"],
        "result_model": rule["result_model"],
        "conditions": rule["conditions"],
        "documentation_required": rule["documentation_required"],
        "risk_level": rule["risk_level"],
        "message": (
            f"Handoff {from_model.upper()} -> {to_model.upper()} is "
            f"{'allowed' if rule['allowed'] else 'NOT allowed'}. "
            f"Result model: {rule['result_model'] or 'N/A'}."
        ),
    }


def get_applicable_certifications(model_code: str) -> List[str]:
    """Return certification schemes applicable to a CoC model.

    Args:
        model_code: Model code ('IP', 'SG', 'MB', or 'CB').

    Returns:
        List of certification code strings. Empty list if model not found.

    Example:
        >>> certs = get_applicable_certifications("MB")
        >>> "RSPO_MB" in certs
        True
    """
    rules = get_model_rules(model_code)
    if rules is None:
        return []
    return list(rules.get("applicable_certifications", []))


def get_model_hierarchy_level(model_code: str) -> Optional[int]:
    """Return the hierarchy level of a CoC model (1=strictest, 4=least).

    Args:
        model_code: Model code ('IP', 'SG', 'MB', or 'CB').

    Returns:
        Hierarchy level integer, or None if model not found.

    Example:
        >>> get_model_hierarchy_level("IP")
        1
        >>> get_model_hierarchy_level("CB")
        4
    """
    entry = MODEL_HIERARCHY.get(model_code.upper())
    if entry is None:
        return None
    return entry["level"]


def can_upgrade_model(
    current_model: str,
    target_model: str,
) -> bool:
    """Check whether a model upgrade is possible.

    Model upgrades are NEVER allowed in ISO 22095. Material can only
    be downgraded (IP->SG->MB->CB) or maintained at the same level.

    Args:
        current_model: Current CoC model code.
        target_model: Desired CoC model code.

    Returns:
        True if the 'upgrade' is actually at same level or downgrade
        (allowed), False if it would be a true upgrade (not allowed).

    Example:
        >>> can_upgrade_model("IP", "SG")  # downgrade, allowed
        True
        >>> can_upgrade_model("MB", "IP")  # upgrade, NOT allowed
        False
        >>> can_upgrade_model("SG", "SG")  # same level, allowed
        True
    """
    current_level = get_model_hierarchy_level(current_model)
    target_level = get_model_hierarchy_level(target_model)

    if current_level is None or target_level is None:
        return False

    # Same level or downgrade (higher number = less strict)
    return target_level >= current_level


def get_model_validation_rules(model_code: str) -> Optional[Dict[str, Any]]:
    """Return the machine-readable validation rules for a CoC model.

    Args:
        model_code: Model code ('IP', 'SG', 'MB', or 'CB').

    Returns:
        Dictionary of validation rules, or None if model not found.

    Example:
        >>> rules = get_model_validation_rules("IP")
        >>> rules["mixing_allowed"]
        False
    """
    model = get_model_rules(model_code)
    if model is None:
        return None
    return model.get("validation_rules")


def get_all_models() -> List[str]:
    """Return all CoC model codes in hierarchy order (strictest first).

    Returns:
        List of model codes: ['IP', 'SG', 'MB', 'CB'].

    Example:
        >>> get_all_models()
        ['IP', 'SG', 'MB', 'CB']
    """
    return sorted(
        MODEL_HIERARCHY.keys(),
        key=lambda m: MODEL_HIERARCHY[m]["level"],
    )


def get_downgrade_options(model_code: str) -> List[str]:
    """Return models to which the given model can be downgraded.

    Args:
        model_code: Current CoC model code.

    Returns:
        List of model codes that accept downgraded material.
        Empty list if model not found or no downgrades available.

    Example:
        >>> get_downgrade_options("IP")
        ['SG', 'MB', 'CB']
        >>> get_downgrade_options("CB")
        []
    """
    entry = MODEL_HIERARCHY.get(model_code.upper())
    if entry is None:
        return []
    return list(entry.get("can_downgrade_to", []))


def validate_model_compliance(
    model_code: str,
    batch_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate batch data against CoC model rules.

    Performs a comprehensive compliance check of batch attributes
    against the requirements of the specified CoC model.

    Args:
        model_code: CoC model code ('IP', 'SG', 'MB', 'CB').
        batch_data: Dictionary with batch attributes including:
            - source_count: Number of contributing sources
            - all_sources_compliant: Whether all sources are compliant
            - physically_separated: Whether batch is physically separated
            - has_credit_balance: Whether credit balance is available (MB)
            - blend_ratio: Certified blend percentage (CB)

    Returns:
        Dictionary with: compliant (bool), model_code, violations (list),
        warnings (list).

    Example:
        >>> result = validate_model_compliance("IP", {
        ...     "source_count": 1,
        ...     "all_sources_compliant": True,
        ...     "physically_separated": True,
        ... })
        >>> result["compliant"]
        True
    """
    rules = get_model_rules(model_code)
    if rules is None:
        return {
            "compliant": False,
            "model_code": model_code,
            "violations": [f"Unknown model code: {model_code}"],
            "warnings": [],
        }

    validation_rules = rules.get("validation_rules", {})
    violations: List[str] = []
    warnings: List[str] = []

    source_count = batch_data.get("source_count", 0)
    all_compliant = batch_data.get("all_sources_compliant", False)
    physically_separated = batch_data.get("physically_separated", False)
    has_credit = batch_data.get("has_credit_balance", False)
    blend_ratio = batch_data.get("blend_ratio")

    # Check max sources
    max_sources = validation_rules.get("max_sources_per_batch")
    if max_sources is not None and source_count > max_sources:
        violations.append(
            f"Source count {source_count} exceeds maximum "
            f"{max_sources} for {model_code} model"
        )

    # Check mixing
    if not validation_rules.get("mixing_allowed", True) and source_count > 1:
        violations.append(
            f"{model_code} model does not allow mixing; "
            f"found {source_count} sources"
        )

    # Check physical separation
    if validation_rules.get("physical_separation_required", False):
        if not physically_separated:
            violations.append(
                f"{model_code} model requires physical separation"
            )

    # Check all sources compliant (SG requirement)
    if validation_rules.get("all_sources_must_be_compliant", False):
        if not all_compliant:
            violations.append(
                f"{model_code} model requires all sources to be compliant"
            )

    # Check compliant-only flag
    if rules.get("compliant_only", False) and not all_compliant:
        violations.append(
            f"{model_code} model permits only compliant material"
        )

    # Check credit balance for MB
    if validation_rules.get("credit_period_enforced", False):
        if not has_credit:
            warnings.append(
                "Mass balance credit balance not confirmed; "
                "verify before output claims"
            )

    # Check blend ratio for CB
    if validation_rules.get("blend_ratio_required", False):
        if blend_ratio is None:
            violations.append(
                f"{model_code} model requires a declared blend ratio"
            )
        elif blend_ratio < 0 or blend_ratio > 1.0:
            violations.append(
                f"Blend ratio {blend_ratio} is outside valid range [0, 1]"
            )

    compliant = len(violations) == 0

    return {
        "compliant": compliant,
        "model_code": model_code,
        "violations": violations,
        "warnings": warnings,
        "message": (
            f"Batch is compliant with {model_code} model rules."
            if compliant
            else f"Batch has {len(violations)} violation(s) against "
                 f"{model_code} model rules."
        ),
    }
