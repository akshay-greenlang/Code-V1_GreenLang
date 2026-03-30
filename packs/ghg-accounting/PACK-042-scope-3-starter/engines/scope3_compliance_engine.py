# -*- coding: utf-8 -*-
"""
Scope3ComplianceEngine - PACK-042 Scope 3 Starter Pack Engine 9
=================================================================

Maps a Scope 3 GHG inventory to disclosure requirements across 6
regulatory and voluntary frameworks, identifying compliance gaps,
scoring readiness, and generating remediation action plans.

Supported Frameworks:
    1. GHG Protocol Scope 3 Standard (2011) -- 15 requirements
    2. ESRS E1 (Delegated Act 2023/2772), para 44-46 -- 10 requirements
    3. CDP Climate Change (2024), C6.5/C6.7/C6.10 -- 8 requirements
    4. SBTi Corporate Manual (2023), Scope 3 screening -- 7 requirements
    5. SEC Climate Disclosure Rule (2024), Scope 3 materiality -- 5 requirements
    6. California SB 253 (2026), Scope 3 from 2027 -- 5 requirements

Compliance Scoring:
    Per-framework:
        score = (met * 1.0 + partially_met * 0.5) / total_applicable * 100

    Classification:
        >= 90%  -> "Compliant"
        >= 70%  -> "Substantially Compliant"
        >= 50%  -> "Partially Compliant"
        <  50%  -> "Non-Compliant"

    Overall readiness = weighted average of framework scores.

Requirement Database:
    50 requirements across all 6 frameworks, each with:
        requirement_id, framework, description, data_field,
        mandatory, validation_rule, reference

ESRS E1 Phase-In Schedule:
    2025: Categories 1-3 mandatory (large undertakings > 750 employees)
    2026: Categories 1-3 mandatory (all CSRD-scope undertakings)
    2027: All significant categories mandatory
    2029: All 15 categories mandatory (no phase-in exemption)

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Scope 3 Calculation Guidance (2013)
    - ESRS E1 (Delegated Act 2023/2772), Disclosure E1-6
    - CDP Climate Change Scoring Methodology (2024)
    - SBTi Corporate Manual (2023), Scope 3 requirements
    - SEC Final Rule 33-11275 (Climate-Related Disclosures)
    - California SB 253 (Climate Corporate Data Accountability Act)

Zero-Hallucination:
    - All requirements sourced from published regulatory text
    - Validation is deterministic field-presence and range checking
    - No LLM involvement in any compliance assessment path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
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

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FrameworkType(str, Enum):
    """Supported Scope 3 compliance frameworks.

    GHG_PROTOCOL_S3: GHG Protocol Corporate Value Chain Standard (2011).
    ESRS_E1:         ESRS E1 Climate Change (Scope 3 disclosures).
    CDP:             CDP Climate Change Questionnaire.
    SBTI:            Science Based Targets initiative.
    SEC:             US SEC Climate Disclosure Rule.
    SB_253:          California SB 253 (Scope 3 from 2027).
    """
    GHG_PROTOCOL_S3 = "ghg_protocol_scope3"
    ESRS_E1 = "esrs_e1"
    CDP = "cdp"
    SBTI = "sbti"
    SEC = "sec"
    SB_253 = "sb_253"

class RequirementStatus(str, Enum):
    """Compliance status for an individual requirement."""
    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"

class ComplianceClassification(str, Enum):
    """Overall compliance classification."""
    COMPLIANT = "compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"

class GapPriority(str, Enum):
    """Priority level for compliance gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ---------------------------------------------------------------------------
# Constants -- Framework Weights
# ---------------------------------------------------------------------------

FRAMEWORK_WEIGHTS: Dict[str, float] = {
    FrameworkType.GHG_PROTOCOL_S3: 1.0,
    FrameworkType.ESRS_E1: 0.95,
    FrameworkType.CDP: 0.85,
    FrameworkType.SBTI: 0.80,
    FrameworkType.SEC: 0.90,
    FrameworkType.SB_253: 0.85,
}
"""Importance weights for overall readiness scoring."""

FRAMEWORK_DISPLAY_NAMES: Dict[str, str] = {
    FrameworkType.GHG_PROTOCOL_S3: "GHG Protocol Scope 3 Standard",
    FrameworkType.ESRS_E1: "ESRS E1 (Scope 3 Disclosures)",
    FrameworkType.CDP: "CDP Climate Change (Scope 3)",
    FrameworkType.SBTI: "SBTi (Scope 3 Requirements)",
    FrameworkType.SEC: "SEC Climate Disclosure (Scope 3)",
    FrameworkType.SB_253: "California SB 253 (Scope 3)",
}
"""Human-readable framework names."""

# ---------------------------------------------------------------------------
# Constants -- Requirement Database (50 requirements)
# ---------------------------------------------------------------------------

REQUIREMENT_DATABASE: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------
    # GHG Protocol Scope 3 Standard (15 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "GHG-S3-001", "framework": "ghg_protocol_scope3",
     "description": "Scope 3 screening: all 15 categories evaluated for relevance",
     "data_field": "categories_screened", "mandatory": True,
     "validation_rule": "field_gte_15",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 5"},
    {"requirement_id": "GHG-S3-002", "framework": "ghg_protocol_scope3",
     "description": "Material categories identified and reported with quantified emissions",
     "data_field": "categories_reported", "mandatory": True,
     "validation_rule": "field_gte_1",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 5"},
    {"requirement_id": "GHG-S3-003", "framework": "ghg_protocol_scope3",
     "description": "Total Scope 3 emissions reported (tCO2e)",
     "data_field": "scope3_total_tco2e", "mandatory": True,
     "validation_rule": "field_positive_decimal",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 6"},
    {"requirement_id": "GHG-S3-004", "framework": "ghg_protocol_scope3",
     "description": "Per-category emissions reported for all material categories",
     "data_field": "per_category_emissions", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 6"},
    {"requirement_id": "GHG-S3-005", "framework": "ghg_protocol_scope3",
     "description": "Methodology description per category (spend-based, average-data, etc.)",
     "data_field": "per_category_methodology", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 7"},
    {"requirement_id": "GHG-S3-006", "framework": "ghg_protocol_scope3",
     "description": "Emission factor sources documented per category",
     "data_field": "emission_factor_sources", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 7"},
    {"requirement_id": "GHG-S3-007", "framework": "ghg_protocol_scope3",
     "description": "Data quality assessment performed (DQI indicators scored)",
     "data_field": "data_quality_assessment", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 7"},
    {"requirement_id": "GHG-S3-008", "framework": "ghg_protocol_scope3",
     "description": "Uncertainty assessment performed",
     "data_field": "uncertainty_assessment", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 7"},
    {"requirement_id": "GHG-S3-009", "framework": "ghg_protocol_scope3",
     "description": "Base year established and documented",
     "data_field": "base_year", "mandatory": True,
     "validation_rule": "field_positive_int",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 5"},
    {"requirement_id": "GHG-S3-010", "framework": "ghg_protocol_scope3",
     "description": "Exclusions documented with justification and estimated magnitude",
     "data_field": "exclusions_documented", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 5"},
    {"requirement_id": "GHG-S3-011", "framework": "ghg_protocol_scope3",
     "description": "Biogenic CO2 reported separately",
     "data_field": "biogenic_emissions", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 9"},
    {"requirement_id": "GHG-S3-012", "framework": "ghg_protocol_scope3",
     "description": "Reporting period specified (at least 12 months)",
     "data_field": "reporting_period", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 9"},
    {"requirement_id": "GHG-S3-013", "framework": "ghg_protocol_scope3",
     "description": "GWP values specified (AR4, AR5, or AR6)",
     "data_field": "gwp_source", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 8"},
    {"requirement_id": "GHG-S3-014", "framework": "ghg_protocol_scope3",
     "description": "Supplier engagement activities documented",
     "data_field": "supplier_engagement", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 7"},
    {"requirement_id": "GHG-S3-015", "framework": "ghg_protocol_scope3",
     "description": "Reduction targets for Scope 3 documented",
     "data_field": "scope3_reduction_targets", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "GHG Protocol Scope 3 Standard, Chapter 10"},
    # -------------------------------------------------------------------
    # ESRS E1 Scope 3 (10 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "ESRS-S3-001", "framework": "esrs_e1",
     "description": "E1-6 para 44: Gross Scope 3 GHG emissions (tCO2e)",
     "data_field": "scope3_total_tco2e", "mandatory": True,
     "validation_rule": "field_positive_decimal",
     "reference": "ESRS E1, Disclosure E1-6, para 44"},
    {"requirement_id": "ESRS-S3-002", "framework": "esrs_e1",
     "description": "E1-6 para 44: Scope 3 emissions by significant category",
     "data_field": "per_category_emissions", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "ESRS E1, Disclosure E1-6, para 44"},
    {"requirement_id": "ESRS-S3-003", "framework": "esrs_e1",
     "description": "E1-6 para 45: Categories 1-3 mandatory for phase-in (2025)",
     "data_field": "categories_1_3_reported", "mandatory": True,
     "validation_rule": "field_bool_true",
     "reference": "ESRS E1, Disclosure E1-6, para 45"},
    {"requirement_id": "ESRS-S3-004", "framework": "esrs_e1",
     "description": "E1-6 para 46: List of excluded categories with justification",
     "data_field": "exclusions_documented", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "ESRS E1, Disclosure E1-6, para 46"},
    {"requirement_id": "ESRS-S3-005", "framework": "esrs_e1",
     "description": "E1-6: Methodology description (calculation approach per category)",
     "data_field": "per_category_methodology", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "ESRS E1, Disclosure E1-6"},
    {"requirement_id": "ESRS-S3-006", "framework": "esrs_e1",
     "description": "E1-6: Percentage of Scope 3 from measured vs estimated data",
     "data_field": "measured_vs_estimated_pct", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "ESRS E1, Disclosure E1-6"},
    {"requirement_id": "ESRS-S3-007", "framework": "esrs_e1",
     "description": "E1-4: Scope 3 reduction targets (if material)",
     "data_field": "scope3_reduction_targets", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "ESRS E1, Disclosure E1-4"},
    {"requirement_id": "ESRS-S3-008", "framework": "esrs_e1",
     "description": "E1-6: XBRL tagging of Scope 3 data points",
     "data_field": "xbrl_tagging", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "ESRS E1, XBRL taxonomy requirements"},
    {"requirement_id": "ESRS-S3-009", "framework": "esrs_e1",
     "description": "E1-6: GHG intensity ratio (Scope 3 per unit of revenue)",
     "data_field": "scope3_intensity_per_revenue", "mandatory": True,
     "validation_rule": "field_positive_decimal",
     "reference": "ESRS E1, Disclosure E1-6"},
    {"requirement_id": "ESRS-S3-010", "framework": "esrs_e1",
     "description": "E1-6: Transition plan alignment for value chain emissions",
     "data_field": "transition_plan", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "ESRS E1, Disclosure E1-1"},
    # -------------------------------------------------------------------
    # CDP Climate Change -- Scope 3 (8 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "CDP-S3-001", "framework": "cdp",
     "description": "C6.5: Scope 3 emissions reported per category (15 categories)",
     "data_field": "per_category_emissions", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "CDP Climate Change 2024, C6.5"},
    {"requirement_id": "CDP-S3-002", "framework": "cdp",
     "description": "C6.5: Evaluation status per category (relevant, not relevant, N/A)",
     "data_field": "category_evaluation_status", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "CDP Climate Change 2024, C6.5"},
    {"requirement_id": "CDP-S3-003", "framework": "cdp",
     "description": "C6.5: Methodology description per relevant category",
     "data_field": "per_category_methodology", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "CDP Climate Change 2024, C6.5"},
    {"requirement_id": "CDP-S3-004", "framework": "cdp",
     "description": "C6.7: Total Scope 3 emissions (tCO2e)",
     "data_field": "scope3_total_tco2e", "mandatory": True,
     "validation_rule": "field_positive_decimal",
     "reference": "CDP Climate Change 2024, C6.7"},
    {"requirement_id": "CDP-S3-005", "framework": "cdp",
     "description": "C6.10: Scope 3 data quality description",
     "data_field": "data_quality_assessment", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "CDP Climate Change 2024, C6.10"},
    {"requirement_id": "CDP-S3-006", "framework": "cdp",
     "description": "C6.5: Percentage of emissions calculated using primary data",
     "data_field": "primary_data_pct", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "CDP Climate Change 2024, C6.5"},
    {"requirement_id": "CDP-S3-007", "framework": "cdp",
     "description": "C12.1a: Scope 3 supplier engagement activities",
     "data_field": "supplier_engagement", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "CDP Climate Change 2024, C12.1a"},
    {"requirement_id": "CDP-S3-008", "framework": "cdp",
     "description": "C4.1: Scope 3 reduction target (absolute or intensity)",
     "data_field": "scope3_reduction_targets", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "CDP Climate Change 2024, C4.1"},
    # -------------------------------------------------------------------
    # SBTi Scope 3 (7 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "SBTI-S3-001", "framework": "sbti",
     "description": "Scope 3 screening completed (all 15 categories evaluated)",
     "data_field": "categories_screened", "mandatory": True,
     "validation_rule": "field_gte_15",
     "reference": "SBTi Corporate Manual (2023), Scope 3 Step 1"},
    {"requirement_id": "SBTI-S3-002", "framework": "sbti",
     "description": "Scope 3 >= 40% of total S1+S2+S3 emissions => target required",
     "data_field": "scope3_materiality_pct", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "SBTi Corporate Manual (2023), Criteria 15"},
    {"requirement_id": "SBTI-S3-003", "framework": "sbti",
     "description": "Scope 3 target covers at least 67% of Scope 3 emissions",
     "data_field": "scope3_target_coverage_pct", "mandatory": True,
     "validation_rule": "field_gte_67",
     "reference": "SBTi Corporate Manual (2023), Criteria 16"},
    {"requirement_id": "SBTI-S3-004", "framework": "sbti",
     "description": "Near-term Scope 3 target: >= 2.5% annual reduction (WB2C)",
     "data_field": "scope3_annual_reduction_pct", "mandatory": True,
     "validation_rule": "field_gte_2_5",
     "reference": "SBTi Corporate Manual (2023), Criteria 17"},
    {"requirement_id": "SBTI-S3-005", "framework": "sbti",
     "description": "FLAG emissions assessed for land-intensive companies",
     "data_field": "flag_assessment", "mandatory": False,
     "validation_rule": "field_present",
     "reference": "SBTi FLAG Guidance (2022)"},
    {"requirement_id": "SBTI-S3-006", "framework": "sbti",
     "description": "Base year for Scope 3 established (no earlier than 2015)",
     "data_field": "base_year", "mandatory": True,
     "validation_rule": "field_gte_2015",
     "reference": "SBTi Corporate Manual (2023), Criteria 3"},
    {"requirement_id": "SBTI-S3-007", "framework": "sbti",
     "description": "Scope 3 per-category methodology documented",
     "data_field": "per_category_methodology", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "SBTi Corporate Manual (2023), Appendix B"},
    # -------------------------------------------------------------------
    # SEC Climate Disclosure -- Scope 3 (5 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "SEC-S3-001", "framework": "sec",
     "description": "Item 1504(b): Scope 3 disclosed if material or target-setting includes Scope 3",
     "data_field": "scope3_total_tco2e", "mandatory": False,
     "validation_rule": "field_positive_decimal",
     "reference": "SEC Final Rule 33-11275, Item 1504(b)"},
    {"requirement_id": "SEC-S3-002", "framework": "sec",
     "description": "Item 1502(d)(2): Scope 3 materiality assessment performed",
     "data_field": "scope3_materiality_assessment", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "SEC Final Rule 33-11275, Item 1502(d)(2)"},
    {"requirement_id": "SEC-S3-003", "framework": "sec",
     "description": "Safe harbor: methodology and assumptions described",
     "data_field": "per_category_methodology", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "SEC Final Rule 33-11275, Safe Harbor"},
    {"requirement_id": "SEC-S3-004", "framework": "sec",
     "description": "Data sources and third-party data provenance documented",
     "data_field": "emission_factor_sources", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "SEC Final Rule 33-11275"},
    {"requirement_id": "SEC-S3-005", "framework": "sec",
     "description": "Scope 3 intensity per unit revenue disclosed (if material)",
     "data_field": "scope3_intensity_per_revenue", "mandatory": False,
     "validation_rule": "field_positive_decimal",
     "reference": "SEC Final Rule 33-11275"},
    # -------------------------------------------------------------------
    # California SB 253 -- Scope 3 (5 requirements)
    # -------------------------------------------------------------------
    {"requirement_id": "SB253-S3-001", "framework": "sb_253",
     "description": "Scope 3 emissions reported annually (from 2027 for >$1B revenue)",
     "data_field": "scope3_total_tco2e", "mandatory": True,
     "validation_rule": "field_positive_decimal",
     "reference": "SB 253, Section 38532(a)(3)"},
    {"requirement_id": "SB253-S3-002", "framework": "sb_253",
     "description": "Per-category Scope 3 emissions reported",
     "data_field": "per_category_emissions", "mandatory": True,
     "validation_rule": "field_dict_not_empty",
     "reference": "SB 253, Section 38532(a)(3)"},
    {"requirement_id": "SB253-S3-003", "framework": "sb_253",
     "description": "Methodology consistent with GHG Protocol Scope 3 Standard",
     "data_field": "methodology_basis", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "SB 253, Section 38532(b)"},
    {"requirement_id": "SB253-S3-004", "framework": "sb_253",
     "description": "Third-party assurance obtained (limited assurance from 2027)",
     "data_field": "third_party_assurance", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "SB 253, Section 38532(c)"},
    {"requirement_id": "SB253-S3-005", "framework": "sb_253",
     "description": "Emissions reported to emissions reporting organisation",
     "data_field": "reporting_organisation", "mandatory": True,
     "validation_rule": "field_present",
     "reference": "SB 253, Section 38532(d)"},
]
"""Master requirement database for Scope 3 compliance mapping (50 requirements)."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class Scope3InventoryData(BaseModel):
    """Scope 3 inventory data for compliance assessment.

    Contains all fields that requirements may reference.

    Attributes:
        scope3_total_tco2e: Total Scope 3 emissions (tCO2e).
        per_category_emissions: Emissions per category (dict).
        per_category_methodology: Methodology per category (dict).
        categories_screened: Number of categories screened (0-15).
        categories_reported: Number of categories reported.
        categories_1_3_reported: Whether categories 1-3 are reported.
        emission_factor_sources: EF sources per category.
        data_quality_assessment: Whether DQI assessment done.
        uncertainty_assessment: Whether uncertainty assessment done.
        base_year: Base year for Scope 3.
        exclusions_documented: Whether exclusions documented.
        reporting_period: Reporting period description.
        gwp_source: GWP values source (AR4/AR5/AR6).
        supplier_engagement: Whether supplier engagement done.
        scope3_reduction_targets: Scope 3 reduction targets.
        biogenic_emissions: Biogenic CO2 reported separately.
        category_evaluation_status: CDP per-category status.
        primary_data_pct: Percentage using primary data.
        measured_vs_estimated_pct: Measured vs estimated split.
        scope3_materiality_pct: Scope 3 as % of total S1+S2+S3.
        scope3_target_coverage_pct: Coverage of Scope 3 by target (%).
        scope3_annual_reduction_pct: Annual Scope 3 reduction rate.
        flag_assessment: FLAG emissions assessment.
        scope3_materiality_assessment: SEC materiality assessment.
        scope3_intensity_per_revenue: Scope 3 per revenue intensity.
        xbrl_tagging: Whether XBRL tags are applied.
        transition_plan: Whether transition plan exists.
        methodology_basis: Methodology basis description.
        third_party_assurance: Whether third-party assured.
        reporting_organisation: Reporting organisation name.
        additional_fields: Any additional fields.
    """
    scope3_total_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    per_category_emissions: Dict[str, float] = Field(default_factory=dict)
    per_category_methodology: Dict[str, str] = Field(default_factory=dict)
    categories_screened: Optional[int] = Field(default=None, ge=0, le=15)
    categories_reported: Optional[int] = Field(default=None, ge=0, le=15)
    categories_1_3_reported: Optional[bool] = Field(default=None)
    emission_factor_sources: Dict[str, str] = Field(default_factory=dict)
    data_quality_assessment: Optional[str] = Field(default=None)
    uncertainty_assessment: Optional[str] = Field(default=None)
    base_year: Optional[int] = Field(default=None, ge=1990)
    exclusions_documented: Optional[str] = Field(default=None)
    reporting_period: Optional[str] = Field(default=None)
    gwp_source: Optional[str] = Field(default=None)
    supplier_engagement: Optional[str] = Field(default=None)
    scope3_reduction_targets: Optional[str] = Field(default=None)
    biogenic_emissions: Optional[str] = Field(default=None)
    category_evaluation_status: Dict[str, str] = Field(default_factory=dict)
    primary_data_pct: Optional[Decimal] = Field(default=None, ge=0, le=100)
    measured_vs_estimated_pct: Optional[str] = Field(default=None)
    scope3_materiality_pct: Optional[Decimal] = Field(default=None, ge=0, le=100)
    scope3_target_coverage_pct: Optional[Decimal] = Field(default=None, ge=0, le=100)
    scope3_annual_reduction_pct: Optional[Decimal] = Field(default=None, ge=0)
    flag_assessment: Optional[str] = Field(default=None)
    scope3_materiality_assessment: Optional[str] = Field(default=None)
    scope3_intensity_per_revenue: Optional[Decimal] = Field(default=None, ge=0)
    xbrl_tagging: Optional[str] = Field(default=None)
    transition_plan: Optional[str] = Field(default=None)
    methodology_basis: Optional[str] = Field(default=None)
    third_party_assurance: Optional[str] = Field(default=None)
    reporting_organisation: Optional[str] = Field(default=None)
    additional_fields: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class RequirementCheck(BaseModel):
    """Result of evaluating a single compliance requirement.

    Attributes:
        requirement_id: Unique requirement identifier.
        framework: Source framework.
        description: What is required.
        data_field: Inventory field that satisfies.
        mandatory: Whether mandatory.
        status: Compliance status.
        current_value: Current value found.
        gap_description: Gap description (if any).
        remediation_action: Recommended fix.
        priority: Gap priority.
        reference: Regulatory reference.
    """
    requirement_id: str = Field(default="", description="Requirement ID")
    framework: str = Field(default="", description="Framework")
    description: str = Field(default="", description="Description")
    data_field: str = Field(default="", description="Data field")
    mandatory: bool = Field(default=True, description="Mandatory")
    status: str = Field(default=RequirementStatus.NOT_MET, description="Status")
    current_value: str = Field(default="", description="Current value")
    gap_description: str = Field(default="", description="Gap description")
    remediation_action: str = Field(default="", description="Remediation")
    priority: str = Field(default=GapPriority.MEDIUM, description="Priority")
    reference: str = Field(default="", description="Regulatory reference")

class FrameworkResult(BaseModel):
    """Compliance result for a single framework.

    Attributes:
        framework: Framework identifier.
        framework_name: Human-readable name.
        score: Compliance score (0-100).
        total_requirements: Total evaluated.
        met: Count met.
        partially_met: Count partially met.
        not_met: Count not met.
        not_applicable: Count N/A.
        classification: Compliance classification.
        gaps: Requirement results that are not met.
        all_results: All requirement results.
    """
    framework: str = Field(default="", description="Framework")
    framework_name: str = Field(default="", description="Framework name")
    score: float = Field(default=0.0, ge=0, le=100, description="Score")
    total_requirements: int = Field(default=0, description="Total")
    met: int = Field(default=0, description="Met")
    partially_met: int = Field(default=0, description="Partially met")
    not_met: int = Field(default=0, description="Not met")
    not_applicable: int = Field(default=0, description="N/A")
    classification: str = Field(default="", description="Classification")
    gaps: List[RequirementCheck] = Field(default_factory=list, description="Gaps")
    all_results: List[RequirementCheck] = Field(
        default_factory=list, description="All results"
    )

class ComplianceGap(BaseModel):
    """A compliance gap requiring action.

    Attributes:
        framework: Affected framework.
        requirement_id: Requirement ID.
        description: Gap description.
        remediation_action: Recommended fix.
        priority: Priority level.
        estimated_effort: Effort estimate.
    """
    framework: str = Field(default="", description="Framework")
    requirement_id: str = Field(default="", description="Requirement ID")
    description: str = Field(default="", description="Description")
    remediation_action: str = Field(default="", description="Remediation")
    priority: str = Field(default=GapPriority.MEDIUM, description="Priority")
    estimated_effort: str = Field(default="", description="Effort")

class ActionItem(BaseModel):
    """Remediation action item.

    Attributes:
        action_id: Unique action identifier.
        description: Action description.
        frameworks_affected: Frameworks this action addresses.
        requirements_addressed: Requirement IDs addressed.
        priority: Priority level.
        effort_estimate: Effort estimate.
        expected_score_improvement: Expected per-framework score improvement.
    """
    action_id: str = Field(default_factory=_new_uuid, description="Action ID")
    description: str = Field(default="", description="Description")
    frameworks_affected: List[str] = Field(
        default_factory=list, description="Frameworks affected"
    )
    requirements_addressed: List[str] = Field(
        default_factory=list, description="Requirements addressed"
    )
    priority: str = Field(default=GapPriority.MEDIUM, description="Priority")
    effort_estimate: str = Field(default="", description="Effort")
    expected_score_improvement: Dict[str, float] = Field(
        default_factory=dict, description="Score improvement per framework"
    )

class ComplianceAssessment(BaseModel):
    """Complete Scope 3 compliance assessment with provenance.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        processing_time_ms: Processing time (ms).
        frameworks: Per-framework results.
        frameworks_evaluated: Count of frameworks.
        overall_readiness: Weighted readiness (0-100).
        overall_classification: Overall classification.
        critical_gaps: Critical gaps across all frameworks.
        action_plan: Consolidated action items.
        total_requirements: Total requirements evaluated.
        total_met: Total met.
        total_not_met: Total not met.
        methodology_notes: Notes.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    frameworks: List[FrameworkResult] = Field(
        default_factory=list, description="Framework results"
    )
    frameworks_evaluated: int = Field(default=0, description="Frameworks evaluated")
    overall_readiness: float = Field(default=0.0, ge=0, le=100, description="Readiness")
    overall_classification: str = Field(default="", description="Classification")
    critical_gaps: List[ComplianceGap] = Field(
        default_factory=list, description="Critical gaps"
    )
    action_plan: List[ActionItem] = Field(
        default_factory=list, description="Action plan"
    )
    total_requirements: int = Field(default=0, description="Total requirements")
    total_met: int = Field(default=0, description="Total met")
    total_not_met: int = Field(default=0, description="Total not met")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Scope3ComplianceEngine:
    """Scope 3 regulatory compliance mapping engine.

    Assesses a Scope 3 GHG inventory against 6 regulatory frameworks,
    identifying compliance gaps and generating remediation action plans.

    Guarantees:
        - Deterministic: same inputs produce identical compliance scores.
        - Traceable: SHA-256 provenance hash on every result.
        - Standards-based: requirements from published regulatory text.
        - No LLM: zero hallucination risk in compliance assessment.

    Usage::

        engine = Scope3ComplianceEngine()
        assessment = engine.assess_compliance(
            inventory, frameworks=[FrameworkType.GHG_PROTOCOL_S3, FrameworkType.ESRS_E1]
        )
        print(f"Overall readiness: {assessment.overall_readiness}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the compliance engine.

        Args:
            config: Optional configuration overrides.
        """
        self._config = config or {}
        self._requirements = list(REQUIREMENT_DATABASE)
        logger.info("Scope3ComplianceEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assess_compliance(
        self,
        inventory: Scope3InventoryData,
        frameworks: Optional[List[str]] = None,
    ) -> ComplianceAssessment:
        """Assess Scope 3 inventory against selected frameworks.

        Args:
            inventory: Scope 3 inventory data.
            frameworks: List of framework IDs to assess. None = all.

        Returns:
            ComplianceAssessment with per-framework results and action plan.
        """
        t0 = time.perf_counter()
        target_frameworks = frameworks or [fw.value for fw in FrameworkType]
        logger.info("Assessing compliance for %d frameworks.", len(target_frameworks))

        framework_results: List[FrameworkResult] = []
        all_gaps: List[ComplianceGap] = []

        assessors = {
            FrameworkType.GHG_PROTOCOL_S3: self._assess_ghg_protocol,
            FrameworkType.ESRS_E1: self._assess_esrs_e1,
            FrameworkType.CDP: self._assess_cdp,
            FrameworkType.SBTI: self._assess_sbti,
            FrameworkType.SEC: self._assess_sec,
            FrameworkType.SB_253: self._assess_sb253,
        }

        for fw_value in target_frameworks:
            fw_type = FrameworkType(fw_value) if isinstance(fw_value, str) else fw_value
            assessor = assessors.get(fw_type)
            if not assessor:
                logger.warning("No assessor for framework %s.", fw_value)
                continue

            fw_result = assessor(inventory)
            framework_results.append(fw_result)

            # Collect critical gaps.
            for gap_req in fw_result.gaps:
                if gap_req.mandatory and gap_req.status == RequirementStatus.NOT_MET:
                    all_gaps.append(ComplianceGap(
                        framework=fw_result.framework,
                        requirement_id=gap_req.requirement_id,
                        description=gap_req.gap_description or gap_req.description,
                        remediation_action=gap_req.remediation_action,
                        priority=gap_req.priority,
                        estimated_effort=self._estimate_effort(gap_req.data_field),
                    ))

        # Calculate overall readiness.
        overall_readiness = self._calculate_overall_readiness(framework_results)
        overall_class = self._classify(overall_readiness)

        # Generate action plan.
        action_plan = self.generate_action_plan(all_gaps)

        # Totals.
        total_reqs = sum(fr.total_requirements for fr in framework_results)
        total_met = sum(fr.met for fr in framework_results)
        total_not_met = sum(fr.not_met for fr in framework_results)

        elapsed = (time.perf_counter() - t0) * 1000

        result = ComplianceAssessment(
            frameworks=framework_results,
            frameworks_evaluated=len(framework_results),
            overall_readiness=overall_readiness,
            overall_classification=overall_class,
            critical_gaps=all_gaps,
            action_plan=action_plan,
            total_requirements=total_reqs,
            total_met=total_met,
            total_not_met=total_not_met,
            processing_time_ms=_round2(elapsed),
            methodology_notes=[
                "Compliance scoring: (met*1.0 + partial*0.5) / applicable * 100.",
                f"Frameworks assessed: {', '.join(target_frameworks)}.",
                f"Total requirements: {total_reqs}, met: {total_met}, not met: {total_not_met}.",
            ],
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Compliance assessment complete: readiness=%.1f%%, class=%s in %.1f ms.",
            overall_readiness, overall_class, elapsed,
        )
        return result

    def generate_gap_analysis(
        self,
        assessment: ComplianceAssessment,
    ) -> Dict[str, Any]:
        """Generate detailed gap analysis from assessment results.

        Args:
            assessment: Completed compliance assessment.

        Returns:
            Dict with gap analysis per framework.
        """
        t0 = time.perf_counter()
        gaps_by_framework: Dict[str, List[Dict[str, Any]]] = {}

        for fw_result in assessment.frameworks:
            fw_gaps: List[Dict[str, Any]] = []
            for req in fw_result.gaps:
                fw_gaps.append({
                    "requirement_id": req.requirement_id,
                    "description": req.description,
                    "status": req.status,
                    "mandatory": req.mandatory,
                    "gap": req.gap_description,
                    "remediation": req.remediation_action,
                    "priority": req.priority,
                })
            if fw_gaps:
                gaps_by_framework[fw_result.framework] = fw_gaps

        result = {
            "gap_count": len(assessment.critical_gaps),
            "gaps_by_framework": gaps_by_framework,
            "overall_readiness": assessment.overall_readiness,
            "provenance_hash": _compute_hash(gaps_by_framework),
        }

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Gap analysis generated in %.1f ms.", elapsed)
        return result

    def generate_action_plan(
        self,
        gaps: List[ComplianceGap],
    ) -> List[ActionItem]:
        """Generate consolidated action plan from gaps.

        Deduplicates remediation actions that address multiple frameworks.

        Args:
            gaps: List of compliance gaps.

        Returns:
            Consolidated action items sorted by priority.
        """
        t0 = time.perf_counter()

        # Group by remediation action (dedup).
        action_map: Dict[str, ActionItem] = {}
        for gap in gaps:
            key = gap.remediation_action or gap.description
            if key not in action_map:
                action_map[key] = ActionItem(
                    description=key,
                    priority=gap.priority,
                    effort_estimate=gap.estimated_effort,
                )
            item = action_map[key]
            if gap.framework not in item.frameworks_affected:
                item.frameworks_affected.append(gap.framework)
            if gap.requirement_id not in item.requirements_addressed:
                item.requirements_addressed.append(gap.requirement_id)
            # Upgrade priority if needed.
            item.priority = self._higher_priority(item.priority, gap.priority)

        actions = list(action_map.values())

        # Sort by priority.
        priority_order = {
            GapPriority.CRITICAL: 0,
            GapPriority.HIGH: 1,
            GapPriority.MEDIUM: 2,
            GapPriority.LOW: 3,
        }
        actions.sort(key=lambda a: priority_order.get(a.priority, 4))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Action plan: %d items in %.1f ms.", len(actions), elapsed)
        return actions

    def calculate_compliance_score(
        self,
        assessment: ComplianceAssessment,
    ) -> Dict[str, float]:
        """Extract per-framework compliance scores from assessment.

        Args:
            assessment: Completed compliance assessment.

        Returns:
            Dict of framework -> score (0-100).
        """
        return {
            fr.framework: fr.score for fr in assessment.frameworks
        }

    def _compute_provenance(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return _compute_hash(data)

    # -------------------------------------------------------------------
    # Private -- Framework Assessors
    # -------------------------------------------------------------------

    def _assess_ghg_protocol(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against GHG Protocol Scope 3 Standard.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for GHG Protocol Scope 3.
        """
        return self._assess_framework(
            inventory, FrameworkType.GHG_PROTOCOL_S3,
        )

    def _assess_esrs_e1(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against ESRS E1 Scope 3 requirements.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for ESRS E1.
        """
        return self._assess_framework(inventory, FrameworkType.ESRS_E1)

    def _assess_cdp(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against CDP Climate Change Scope 3 requirements.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for CDP.
        """
        return self._assess_framework(inventory, FrameworkType.CDP)

    def _assess_sbti(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against SBTi Scope 3 requirements.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for SBTi.
        """
        return self._assess_framework(inventory, FrameworkType.SBTI)

    def _assess_sec(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against SEC Climate Disclosure Scope 3 requirements.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for SEC.
        """
        return self._assess_framework(inventory, FrameworkType.SEC)

    def _assess_sb253(self, inventory: Scope3InventoryData) -> FrameworkResult:
        """Assess against California SB 253 Scope 3 requirements.

        Args:
            inventory: Scope 3 inventory data.

        Returns:
            FrameworkResult for SB 253.
        """
        return self._assess_framework(inventory, FrameworkType.SB_253)

    def _assess_framework(
        self,
        inventory: Scope3InventoryData,
        framework: FrameworkType,
    ) -> FrameworkResult:
        """Assess inventory against all requirements for a single framework.

        Args:
            inventory: Scope 3 inventory data.
            framework: Framework to assess.

        Returns:
            FrameworkResult with scores and gaps.
        """
        fw_reqs = [
            r for r in self._requirements if r["framework"] == framework.value
        ]
        results: List[RequirementCheck] = []
        gaps: List[RequirementCheck] = []

        for req in fw_reqs:
            check = self._evaluate_requirement(inventory, req)
            results.append(check)
            if check.status in (RequirementStatus.NOT_MET, RequirementStatus.PARTIALLY_MET):
                gaps.append(check)

        # Calculate score.
        total = len([r for r in results if r.status != RequirementStatus.NOT_APPLICABLE])
        met = len([r for r in results if r.status == RequirementStatus.MET])
        partial = len([r for r in results if r.status == RequirementStatus.PARTIALLY_MET])
        not_met = len([r for r in results if r.status == RequirementStatus.NOT_MET])
        na = len([r for r in results if r.status == RequirementStatus.NOT_APPLICABLE])

        score = _round2(
            _safe_divide(
                _decimal(met) + _decimal(partial) * Decimal("0.5"),
                _decimal(total),
            ) * Decimal("100")
        ) if total > 0 else 0.0

        classification = self._classify(score)

        return FrameworkResult(
            framework=framework.value,
            framework_name=FRAMEWORK_DISPLAY_NAMES.get(framework, framework.value),
            score=score,
            total_requirements=len(results),
            met=met,
            partially_met=partial,
            not_met=not_met,
            not_applicable=na,
            classification=classification,
            gaps=gaps,
            all_results=results,
        )

    # -------------------------------------------------------------------
    # Private -- Requirement Evaluation
    # -------------------------------------------------------------------

    def _evaluate_requirement(
        self,
        inventory: Scope3InventoryData,
        req: Dict[str, Any],
    ) -> RequirementCheck:
        """Evaluate a single requirement against inventory data.

        Args:
            inventory: Scope 3 inventory data.
            req: Requirement definition dict.

        Returns:
            RequirementCheck with status.
        """
        field_name = req["data_field"]
        rule = req["validation_rule"]
        mandatory = req.get("mandatory", True)

        # Get field value.
        value = getattr(inventory, field_name, None)
        if value is None:
            value = inventory.additional_fields.get(field_name)

        # Apply validation rule.
        status, current_str = self._apply_rule(rule, value)

        # Generate gap description and remediation.
        gap_desc = ""
        remediation = ""
        priority = GapPriority.MEDIUM
        if status != RequirementStatus.MET:
            gap_desc = f"Requirement not satisfied: {req['description']}"
            remediation = self._generate_remediation(field_name, rule, req["description"])
            priority = (
                GapPriority.CRITICAL if mandatory
                else GapPriority.LOW
            )

        return RequirementCheck(
            requirement_id=req["requirement_id"],
            framework=req["framework"],
            description=req["description"],
            data_field=field_name,
            mandatory=mandatory,
            status=status,
            current_value=current_str,
            gap_description=gap_desc,
            remediation_action=remediation,
            priority=priority,
            reference=req.get("reference", ""),
        )

    def _apply_rule(
        self, rule: str, value: Any,
    ) -> Tuple[str, str]:
        """Apply a validation rule to a value.

        Args:
            rule: Validation rule name.
            value: Field value from inventory.

        Returns:
            Tuple of (status, value_string).
        """
        if rule == "field_present":
            if value is not None and str(value).strip():
                return RequirementStatus.MET, str(value)[:100]
            return RequirementStatus.NOT_MET, "Not provided"

        if rule == "field_positive_decimal":
            if value is not None:
                try:
                    dec_val = _decimal(value)
                    if dec_val > 0:
                        return RequirementStatus.MET, str(_round2(dec_val))
                except Exception:
                    pass
            return RequirementStatus.NOT_MET, "Not provided or <= 0"

        if rule == "field_positive_int":
            if value is not None and isinstance(value, int) and value > 0:
                return RequirementStatus.MET, str(value)
            return RequirementStatus.NOT_MET, "Not provided or invalid"

        if rule == "field_dict_not_empty":
            if isinstance(value, dict) and len(value) > 0:
                return RequirementStatus.MET, f"{len(value)} entries"
            if isinstance(value, dict):
                return RequirementStatus.PARTIALLY_MET, "Empty dict"
            return RequirementStatus.NOT_MET, "Not provided"

        if rule == "field_bool_true":
            if value is True:
                return RequirementStatus.MET, "Yes"
            if value is False:
                return RequirementStatus.NOT_MET, "No"
            return RequirementStatus.NOT_MET, "Not provided"

        if rule == "field_gte_15":
            if value is not None and isinstance(value, (int, float)):
                if value >= 15:
                    return RequirementStatus.MET, str(value)
                if value >= 10:
                    return RequirementStatus.PARTIALLY_MET, f"{value} (< 15)"
            return RequirementStatus.NOT_MET, "Not provided or < 10"

        if rule == "field_gte_1":
            if value is not None and isinstance(value, (int, float)) and value >= 1:
                return RequirementStatus.MET, str(value)
            return RequirementStatus.NOT_MET, "Not provided or < 1"

        if rule == "field_gte_67":
            if value is not None:
                dec_val = _decimal(value)
                if dec_val >= Decimal("67"):
                    return RequirementStatus.MET, f"{_round2(dec_val)}%"
                if dec_val >= Decimal("50"):
                    return RequirementStatus.PARTIALLY_MET, f"{_round2(dec_val)}%"
            return RequirementStatus.NOT_MET, "Not provided or < 50%"

        if rule == "field_gte_2_5":
            if value is not None:
                dec_val = _decimal(value)
                if dec_val >= Decimal("2.5"):
                    return RequirementStatus.MET, f"{_round2(dec_val)}%"
                if dec_val > 0:
                    return RequirementStatus.PARTIALLY_MET, f"{_round2(dec_val)}%"
            return RequirementStatus.NOT_MET, "Not provided or 0"

        if rule == "field_gte_2015":
            if value is not None and isinstance(value, int) and value >= 2015:
                return RequirementStatus.MET, str(value)
            return RequirementStatus.NOT_MET, "Not provided or < 2015"

        # Fallback.
        if value is not None and str(value).strip():
            return RequirementStatus.MET, str(value)[:100]
        return RequirementStatus.NOT_MET, "Not provided"

    # -------------------------------------------------------------------
    # Private -- Scoring and Classification
    # -------------------------------------------------------------------

    def _classify(self, score: float) -> str:
        """Classify compliance level from score.

        Args:
            score: Compliance score (0-100).

        Returns:
            ComplianceClassification value.
        """
        if score >= 90:
            return ComplianceClassification.COMPLIANT
        if score >= 70:
            return ComplianceClassification.SUBSTANTIALLY_COMPLIANT
        if score >= 50:
            return ComplianceClassification.PARTIALLY_COMPLIANT
        return ComplianceClassification.NON_COMPLIANT

    def _calculate_overall_readiness(
        self, framework_results: List[FrameworkResult],
    ) -> float:
        """Calculate weighted overall readiness score.

        Args:
            framework_results: Per-framework results.

        Returns:
            Weighted readiness score (0-100).
        """
        total_weighted = Decimal("0")
        total_weight = Decimal("0")
        for fr in framework_results:
            weight = _decimal(FRAMEWORK_WEIGHTS.get(fr.framework, 0.5))
            total_weighted += _decimal(fr.score) * weight
            total_weight += weight
        return _round2(_safe_divide(total_weighted, total_weight))

    # -------------------------------------------------------------------
    # Private -- Remediation Helpers
    # -------------------------------------------------------------------

    def _generate_remediation(
        self, field_name: str, rule: str, description: str,
    ) -> str:
        """Generate a remediation action string.

        Args:
            field_name: Data field name.
            rule: Validation rule.
            description: Requirement description.

        Returns:
            Remediation action string.
        """
        remediations: Dict[str, str] = {
            "scope3_total_tco2e": "Calculate and report total Scope 3 emissions in tCO2e.",
            "per_category_emissions": "Quantify emissions per Scope 3 category.",
            "per_category_methodology": "Document methodology (spend, average, supplier) per category.",
            "categories_screened": "Screen all 15 Scope 3 categories for relevance.",
            "categories_reported": "Report emissions for all material categories.",
            "categories_1_3_reported": "Report Categories 1-3 (ESRS E1 phase-in requirement).",
            "emission_factor_sources": "Document emission factor sources per category.",
            "data_quality_assessment": "Perform DQI assessment per GHG Protocol guidance.",
            "uncertainty_assessment": "Perform uncertainty assessment (analytical or Monte Carlo).",
            "base_year": "Establish a Scope 3 base year (2015 or later for SBTi).",
            "exclusions_documented": "Document excluded categories with justification.",
            "reporting_period": "Specify reporting period (at least 12 months).",
            "gwp_source": "Specify GWP values used (IPCC AR4, AR5, or AR6).",
            "supplier_engagement": "Implement supplier carbon data engagement programme.",
            "scope3_reduction_targets": "Set Scope 3 reduction targets.",
            "scope3_materiality_pct": "Calculate Scope 3 as % of total S1+S2+S3.",
            "scope3_target_coverage_pct": "Ensure Scope 3 target covers >= 67% of emissions.",
            "scope3_annual_reduction_pct": "Set annual reduction rate >= 2.5% (SBTi WB2C).",
            "scope3_materiality_assessment": "Perform materiality assessment for Scope 3.",
            "scope3_intensity_per_revenue": "Calculate Scope 3 intensity per revenue.",
            "xbrl_tagging": "Apply ESRS E1 XBRL taxonomy tags to Scope 3 data.",
            "transition_plan": "Develop transition plan addressing value chain emissions.",
            "methodology_basis": "Document that GHG Protocol Scope 3 Standard is the basis.",
            "third_party_assurance": "Obtain third-party assurance for Scope 3 data.",
            "reporting_organisation": "Report to designated emissions reporting organisation.",
            "category_evaluation_status": "Provide evaluation status per category (CDP C6.5).",
            "measured_vs_estimated_pct": "Disclose percentage of measured vs estimated data.",
            "flag_assessment": "Assess FLAG (land-use) emissions if applicable.",
            "primary_data_pct": "Report percentage of Scope 3 from primary data.",
        }
        return remediations.get(field_name, f"Address: {description}")

    def _estimate_effort(self, field_name: str) -> str:
        """Estimate effort to close a gap.

        Args:
            field_name: Data field name.

        Returns:
            Effort estimate string.
        """
        high_effort = {
            "scope3_total_tco2e", "per_category_emissions",
            "supplier_engagement", "third_party_assurance",
            "scope3_reduction_targets",
        }
        medium_effort = {
            "per_category_methodology", "emission_factor_sources",
            "data_quality_assessment", "uncertainty_assessment",
            "scope3_intensity_per_revenue", "categories_screened",
        }
        if field_name in high_effort:
            return "High effort (3-6 months, dedicated resources)."
        if field_name in medium_effort:
            return "Medium effort (1-3 months, part-time resource)."
        return "Low effort (< 1 month, documentation task)."

    def _higher_priority(self, a: str, b: str) -> str:
        """Return the higher of two priorities.

        Args:
            a: First priority.
            b: Second priority.

        Returns:
            The higher priority.
        """
        order = {
            GapPriority.CRITICAL: 0,
            GapPriority.HIGH: 1,
            GapPriority.MEDIUM: 2,
            GapPriority.LOW: 3,
        }
        if order.get(a, 4) <= order.get(b, 4):
            return a
        return b

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

Scope3InventoryData.model_rebuild()
RequirementCheck.model_rebuild()
FrameworkResult.model_rebuild()
ComplianceGap.model_rebuild()
ActionItem.model_rebuild()
ComplianceAssessment.model_rebuild()
