# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Regulatory Compliance Validation (Engine 6 of 7)

AGENT-MRV-003: Mobile Combustion Agent

Validates mobile combustion emission calculations against six major
regulatory frameworks, providing data completeness checks, methodology
validation, reporting requirement verification, and actionable
recommendations for compliance gaps.

Supported Regulatory Frameworks (6):

    1. **GHG Protocol Corporate Standard** (GHG_PROTOCOL):
       Chapter 5 - Scope 1 mobile combustion. Requires fuel-based or
       distance-based method, separate biogenic CO2 reporting, per-gas
       reporting (CO2, CH4, N2O), GWP source disclosure, and base year
       recalculation for fleet changes >5%.

    2. **ISO 14064-1:2018** (ISO_14064):
       Category 1 direct emissions. Requires quantification methodology
       documentation, uncertainty statement, quality management
       procedures, and documentation of assumptions and limitations.

    3. **CSRD/ESRS E1** (CSRD_ESRS_E1):
       E1-6 Scope 1 by source category. Requires intensity metrics,
       target pathway disclosure, mitigation action reporting, and
       transport as a separate disclosure category.

    4. **EPA 40 CFR Part 98** (EPA_PART_98):
       Subpart C - General stationary/mobile combustion. Methodology:
       fuel consumption x HHV x EF x oxidation factor. Required:
       monthly reporting for large emitters (>25,000 tCO2e/yr),
       tier-specific calculations, equipment-specific emission factors.

    5. **UK SECR** (UK_SECR):
       Streamlined Energy and Carbon Reporting. Required: separate
       transport emissions category, intensity ratio (tCO2e/revenue,
       tCO2e/FTE), energy consumption in kWh, methodology disclosure.

    6. **EU ETS MRR** (EU_ETS_MRR):
       Monitoring and Reporting Regulation for combustion emissions.
       Tier approach, calibrated fuel measurement for Tier 3, annual
       emission factor analysis, verification requirements.

Compliance Check Architecture:
    Each framework check returns a ComplianceCheckResult containing:
    - status: COMPLIANT / NON_COMPLIANT / NEEDS_REVIEW
    - findings: List of individual findings with severity
    - recommendations: Actionable steps for remediation
    - provenance_hash: SHA-256 hash of the compliance assessment

Zero-Hallucination Guarantees:
    - All compliance rules are deterministic coded checks.
    - No LLM involvement in any compliance determination.
    - Every finding maps to a specific regulatory requirement.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.mobile_combustion.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_results={
    ...         "total_co2e_kg": 50000,
    ...         "method": "FUEL_BASED",
    ...         "tier": "TIER_2",
    ...         "gases": {"CO2": 49000, "CH4": 500, "N2O": 500},
    ...     },
    ...     framework="GHG_PROTOCOL",
    ... )
    >>> print(result.status, len(result.findings))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ComplianceCheckerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_compliance_check as _record_compliance_check,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_compliance_check = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ===========================================================================
# Enumerations
# ===========================================================================


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks for mobile combustion compliance.

    GHG_PROTOCOL: GHG Protocol Corporate Accounting and Reporting
        Standard - Chapter 5 Mobile Combustion (Scope 1).
    ISO_14064: ISO 14064-1:2018 Organization-level GHG quantification.
    CSRD_ESRS_E1: EU CSRD European Sustainability Reporting Standards
        ESRS E1 Climate Change - E1-6 Gross Scope 1 GHG emissions.
    EPA_PART_98: US EPA 40 CFR Part 98 Subpart C - General Stationary
        Fuel Combustion (also applicable to mobile sources).
    UK_SECR: UK Streamlined Energy and Carbon Reporting.
    EU_ETS_MRR: EU Emissions Trading System Monitoring and Reporting
        Regulation (mobile sources under installation boundary).
    """

    GHG_PROTOCOL = "GHG_PROTOCOL"
    ISO_14064 = "ISO_14064"
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    EPA_PART_98 = "EPA_PART_98"
    UK_SECR = "UK_SECR"
    EU_ETS_MRR = "EU_ETS_MRR"


class ComplianceStatus(str, Enum):
    """Compliance assessment status values.

    COMPLIANT: Fully meets all requirements of the framework.
    NON_COMPLIANT: Fails one or more mandatory requirements.
    NEEDS_REVIEW: Cannot determine compliance; additional data or
        review is needed.
    """

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class FindingSeverity(str, Enum):
    """Severity level for compliance findings.

    CRITICAL: Mandatory requirement not met; blocks compliance.
    MAJOR: Significant gap that must be addressed.
    MINOR: Minor issue; compliance not blocked but improvement needed.
    INFORMATIONAL: Observation for awareness; no action required.
    PASS: Requirement fully met.
    """

    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFORMATIONAL = "INFORMATIONAL"
    PASS = "PASS"


# ===========================================================================
# Framework Requirement Definitions
# ===========================================================================

# GHG Protocol Corporate Standard - Chapter 5 Mobile Combustion
_GHG_PROTOCOL_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "GHG-MC-001",
        "requirement": "Scope 1 boundary includes company-owned/controlled vehicles",
        "category": "boundary",
        "mandatory": True,
        "fields_required": ["organizational_boundary", "control_approach"],
    },
    {
        "id": "GHG-MC-002",
        "requirement": "Use fuel-based or distance-based calculation method",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["method"],
        "valid_methods": ["FUEL_BASED", "DISTANCE_BASED"],
    },
    {
        "id": "GHG-MC-003",
        "requirement": "Separate biogenic CO2 reporting from fossil CO2",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["biogenic_co2_separated"],
    },
    {
        "id": "GHG-MC-004",
        "requirement": "Report CO2, CH4, N2O emissions per gas",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["gases"],
        "required_gases": ["CO2", "CH4", "N2O"],
    },
    {
        "id": "GHG-MC-005",
        "requirement": "Disclose GWP source and assessment report version",
        "category": "transparency",
        "mandatory": True,
        "fields_required": ["gwp_source"],
    },
    {
        "id": "GHG-MC-006",
        "requirement": "Base year recalculation for fleet changes >5%",
        "category": "consistency",
        "mandatory": True,
        "fields_required": ["base_year", "base_year_emissions"],
    },
    {
        "id": "GHG-MC-007",
        "requirement": "Report total Scope 1 mobile combustion in tCO2e",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["total_co2e_kg"],
    },
    {
        "id": "GHG-MC-008",
        "requirement": "Document emission factors and their sources",
        "category": "transparency",
        "mandatory": True,
        "fields_required": ["emission_factor_source"],
    },
    {
        "id": "GHG-MC-009",
        "requirement": "Report fuel type and quantity per vehicle category",
        "category": "reporting",
        "mandatory": False,
        "fields_required": ["fuel_type", "fuel_quantity"],
    },
    {
        "id": "GHG-MC-010",
        "requirement": "Include uncertainty assessment (recommended)",
        "category": "quality",
        "mandatory": False,
        "fields_required": ["uncertainty_assessment"],
    },
]

# ISO 14064-1:2018 - Category 1 Direct Emissions
_ISO_14064_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "ISO-MC-001",
        "requirement": "Quantify direct Scope 1 mobile combustion emissions",
        "category": "quantification",
        "mandatory": True,
        "fields_required": ["total_co2e_kg"],
    },
    {
        "id": "ISO-MC-002",
        "requirement": "Document quantification methodology",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["method", "tier"],
    },
    {
        "id": "ISO-MC-003",
        "requirement": "Include uncertainty statement",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["uncertainty_assessment"],
    },
    {
        "id": "ISO-MC-004",
        "requirement": "Quality management procedures documented",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["quality_procedures"],
    },
    {
        "id": "ISO-MC-005",
        "requirement": "Document assumptions and limitations",
        "category": "transparency",
        "mandatory": True,
        "fields_required": ["assumptions_documented"],
    },
    {
        "id": "ISO-MC-006",
        "requirement": "Establish organizational boundaries (equity/control)",
        "category": "boundary",
        "mandatory": True,
        "fields_required": ["organizational_boundary", "control_approach"],
    },
    {
        "id": "ISO-MC-007",
        "requirement": "Maintain GHG information management system",
        "category": "system",
        "mandatory": True,
        "fields_required": ["data_management_system"],
    },
    {
        "id": "ISO-MC-008",
        "requirement": "Report per-gas emissions (CO2, CH4, N2O)",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["gases"],
        "required_gases": ["CO2", "CH4", "N2O"],
    },
    {
        "id": "ISO-MC-009",
        "requirement": "Subject to external verification",
        "category": "verification",
        "mandatory": False,
        "fields_required": ["verification_status"],
    },
    {
        "id": "ISO-MC-010",
        "requirement": "Base year and recalculation policy",
        "category": "consistency",
        "mandatory": True,
        "fields_required": ["base_year"],
    },
]

# CSRD/ESRS E1 - E1-6 Gross Scope 1 GHG Emissions
_CSRD_ESRS_E1_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "CSRD-MC-001",
        "requirement": "Disclose Scope 1 GHG emissions by source category",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["total_co2e_kg", "source_category"],
    },
    {
        "id": "CSRD-MC-002",
        "requirement": "Report transport as separate emission category",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["transport_category_separated"],
    },
    {
        "id": "CSRD-MC-003",
        "requirement": "Intensity metrics required (tCO2e/revenue or per unit)",
        "category": "metrics",
        "mandatory": True,
        "fields_required": ["intensity_metric"],
    },
    {
        "id": "CSRD-MC-004",
        "requirement": "Target pathway disclosure for emission reduction",
        "category": "target",
        "mandatory": True,
        "fields_required": ["reduction_target"],
    },
    {
        "id": "CSRD-MC-005",
        "requirement": "Mitigation actions disclosure",
        "category": "narrative",
        "mandatory": True,
        "fields_required": ["mitigation_actions"],
    },
    {
        "id": "CSRD-MC-006",
        "requirement": "Report gross and net Scope 1 emissions separately",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["gross_emissions", "net_emissions"],
    },
    {
        "id": "CSRD-MC-007",
        "requirement": "GHG reporting in accordance with GHG Protocol",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["method"],
    },
    {
        "id": "CSRD-MC-008",
        "requirement": "Double materiality assessment for climate impact",
        "category": "assessment",
        "mandatory": True,
        "fields_required": ["materiality_assessment"],
    },
    {
        "id": "CSRD-MC-009",
        "requirement": "Report energy consumption from mobile combustion",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["energy_consumption_kwh"],
    },
    {
        "id": "CSRD-MC-010",
        "requirement": "Describe methodology and significant assumptions",
        "category": "transparency",
        "mandatory": True,
        "fields_required": ["assumptions_documented"],
    },
]

# EPA 40 CFR Part 98 - Subpart C
_EPA_PART_98_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "EPA-MC-001",
        "requirement": "Calculate using fuel consumption x HHV x EF x oxidation factor",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["method", "fuel_quantity"],
    },
    {
        "id": "EPA-MC-002",
        "requirement": "Monthly reporting for facilities >25,000 tCO2e/yr",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["total_co2e_kg", "reporting_frequency"],
    },
    {
        "id": "EPA-MC-003",
        "requirement": "Tier-specific calculation methodology",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["tier"],
    },
    {
        "id": "EPA-MC-004",
        "requirement": "Equipment-specific emission factors",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["emission_factor_source", "equipment_type"],
    },
    {
        "id": "EPA-MC-005",
        "requirement": "Fuel analysis data (Tier 3 requirement)",
        "category": "methodology",
        "mandatory": False,
        "fields_required": ["fuel_analysis_data"],
    },
    {
        "id": "EPA-MC-006",
        "requirement": "Report CO2, CH4, N2O separately",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["gases"],
        "required_gases": ["CO2", "CH4", "N2O"],
    },
    {
        "id": "EPA-MC-007",
        "requirement": "Fuel type classification per EPA fuel categories",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["fuel_type"],
    },
    {
        "id": "EPA-MC-008",
        "requirement": "Documentation of measurement devices and calibration",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["measurement_documentation"],
    },
    {
        "id": "EPA-MC-009",
        "requirement": "Missing data substitution procedures (98.35)",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["missing_data_procedures"],
    },
    {
        "id": "EPA-MC-010",
        "requirement": "Annual report submission by March 31",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["reporting_year", "submission_date"],
    },
]

# UK SECR - Streamlined Energy and Carbon Reporting
_UK_SECR_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "SECR-MC-001",
        "requirement": "Separate transport emissions category",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["transport_category_separated"],
    },
    {
        "id": "SECR-MC-002",
        "requirement": "Intensity ratio: tCO2e per unit of revenue",
        "category": "metrics",
        "mandatory": True,
        "fields_required": ["intensity_per_revenue"],
    },
    {
        "id": "SECR-MC-003",
        "requirement": "Intensity ratio: tCO2e per FTE",
        "category": "metrics",
        "mandatory": True,
        "fields_required": ["intensity_per_fte"],
    },
    {
        "id": "SECR-MC-004",
        "requirement": "Energy consumption reported in kWh",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["energy_consumption_kwh"],
    },
    {
        "id": "SECR-MC-005",
        "requirement": "Methodology disclosure (DEFRA/BEIS guidance)",
        "category": "transparency",
        "mandatory": True,
        "fields_required": ["methodology_disclosed"],
    },
    {
        "id": "SECR-MC-006",
        "requirement": "Report total Scope 1 emissions in tCO2e",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["total_co2e_kg"],
    },
    {
        "id": "SECR-MC-007",
        "requirement": "UK and global emissions reported separately",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["uk_emissions", "global_emissions"],
    },
    {
        "id": "SECR-MC-008",
        "requirement": "Comparison with previous reporting period",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["previous_period_emissions"],
    },
    {
        "id": "SECR-MC-009",
        "requirement": "Description of energy efficiency actions taken",
        "category": "narrative",
        "mandatory": True,
        "fields_required": ["efficiency_actions"],
    },
    {
        "id": "SECR-MC-010",
        "requirement": "Disclosure of emission reduction measures",
        "category": "narrative",
        "mandatory": False,
        "fields_required": ["mitigation_actions"],
    },
]

# EU ETS MRR - Monitoring and Reporting Regulation
_EU_ETS_MRR_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "MRR-MC-001",
        "requirement": "Tier-based approach for combustion emissions",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["tier"],
    },
    {
        "id": "MRR-MC-002",
        "requirement": "Calibrated fuel measurement for Tier 3+",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["fuel_measurement_method"],
    },
    {
        "id": "MRR-MC-003",
        "requirement": "Annual emission factor analysis",
        "category": "methodology",
        "mandatory": True,
        "fields_required": ["ef_analysis_frequency"],
    },
    {
        "id": "MRR-MC-004",
        "requirement": "Verification by accredited verifier",
        "category": "verification",
        "mandatory": True,
        "fields_required": ["verification_status"],
    },
    {
        "id": "MRR-MC-005",
        "requirement": "Monitoring plan approved by competent authority",
        "category": "system",
        "mandatory": True,
        "fields_required": ["monitoring_plan_approved"],
    },
    {
        "id": "MRR-MC-006",
        "requirement": "Report CO2 emissions per source stream",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["total_co2e_kg", "source_streams"],
    },
    {
        "id": "MRR-MC-007",
        "requirement": "Biomass fraction determination and reporting",
        "category": "reporting",
        "mandatory": True,
        "fields_required": ["biomass_fraction"],
    },
    {
        "id": "MRR-MC-008",
        "requirement": "Uncertainty assessment per source stream",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["uncertainty_assessment"],
    },
    {
        "id": "MRR-MC-009",
        "requirement": "Data gap procedures documented",
        "category": "quality",
        "mandatory": True,
        "fields_required": ["data_gap_procedures"],
    },
    {
        "id": "MRR-MC-010",
        "requirement": "Annual improvement report",
        "category": "improvement",
        "mandatory": False,
        "fields_required": ["improvement_report"],
    },
]

# Master framework requirements map
_FRAMEWORK_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    RegulatoryFramework.GHG_PROTOCOL.value: _GHG_PROTOCOL_REQUIREMENTS,
    RegulatoryFramework.ISO_14064.value: _ISO_14064_REQUIREMENTS,
    RegulatoryFramework.CSRD_ESRS_E1.value: _CSRD_ESRS_E1_REQUIREMENTS,
    RegulatoryFramework.EPA_PART_98.value: _EPA_PART_98_REQUIREMENTS,
    RegulatoryFramework.UK_SECR.value: _UK_SECR_REQUIREMENTS,
    RegulatoryFramework.EU_ETS_MRR.value: _EU_ETS_MRR_REQUIREMENTS,
}


# ===========================================================================
# EPA Large Emitter Threshold
# ===========================================================================

_EPA_LARGE_EMITTER_THRESHOLD_TCO2E: Decimal = Decimal("25000")
_EPA_LARGE_EMITTER_THRESHOLD_KG: Decimal = _EPA_LARGE_EMITTER_THRESHOLD_TCO2E * Decimal("1000")

# Base year recalculation threshold (5% change)
_BASE_YEAR_RECALC_THRESHOLD: Decimal = Decimal("0.05")


# ===========================================================================
# Framework Methodology Mappings
# ===========================================================================

_VALID_METHODS_BY_FRAMEWORK: Dict[str, List[str]] = {
    RegulatoryFramework.GHG_PROTOCOL.value: [
        "FUEL_BASED", "DISTANCE_BASED",
    ],
    RegulatoryFramework.ISO_14064.value: [
        "FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED",
    ],
    RegulatoryFramework.CSRD_ESRS_E1.value: [
        "FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED",
    ],
    RegulatoryFramework.EPA_PART_98.value: [
        "FUEL_BASED",
    ],
    RegulatoryFramework.UK_SECR.value: [
        "FUEL_BASED", "DISTANCE_BASED",
    ],
    RegulatoryFramework.EU_ETS_MRR.value: [
        "FUEL_BASED",
    ],
}

_VALID_TIERS_BY_FRAMEWORK: Dict[str, List[str]] = {
    RegulatoryFramework.GHG_PROTOCOL.value: [
        "TIER_1", "TIER_2", "TIER_3",
    ],
    RegulatoryFramework.ISO_14064.value: [
        "TIER_1", "TIER_2", "TIER_3",
    ],
    RegulatoryFramework.CSRD_ESRS_E1.value: [
        "TIER_1", "TIER_2", "TIER_3",
    ],
    RegulatoryFramework.EPA_PART_98.value: [
        "TIER_1", "TIER_2", "TIER_3", "TIER_4",
    ],
    RegulatoryFramework.UK_SECR.value: [
        "TIER_1", "TIER_2", "TIER_3",
    ],
    RegulatoryFramework.EU_ETS_MRR.value: [
        "TIER_1", "TIER_2", "TIER_3",
    ],
}


# ===========================================================================
# Recommendation Templates
# ===========================================================================

_RECOMMENDATION_TEMPLATES: Dict[str, str] = {
    "missing_gases": (
        "Add per-gas emission reporting for {missing_gases}. "
        "Use IPCC emission factors to disaggregate total CO2e into "
        "individual gas components."
    ),
    "missing_gwp_source": (
        "Document the GWP source (e.g., IPCC AR5, AR6) used for "
        "CO2e conversion. This is required for transparent reporting."
    ),
    "missing_base_year": (
        "Establish a base year and document the base year emissions "
        "for mobile combustion. This enables trend tracking and "
        "recalculation when structural changes occur."
    ),
    "missing_uncertainty": (
        "Conduct an uncertainty assessment using Monte Carlo simulation "
        "or analytical error propagation. ISO 14064-1 and GHG Protocol "
        "both recommend (or require) uncertainty quantification."
    ),
    "missing_intensity": (
        "Calculate at least one intensity metric: tCO2e per unit of "
        "revenue, tCO2e per FTE, or tCO2e per vehicle-km. "
        "CSRD/ESRS E1 and UK SECR require intensity disclosure."
    ),
    "invalid_method": (
        "The calculation method '{method}' is not accepted by "
        "{framework}. Switch to one of: {valid_methods}."
    ),
    "upgrade_tier": (
        "Consider upgrading from {current_tier} to a higher tier for "
        "more accurate results and reduced uncertainty. {framework} "
        "recommends Tier 2 or Tier 3 where feasible."
    ),
    "missing_biogenic": (
        "Separate biogenic CO2 from fossil CO2 emissions. "
        "Biogenic CO2 from biofuels should be reported separately "
        "per GHG Protocol Chapter 5 guidance."
    ),
    "missing_energy_kwh": (
        "Report energy consumption in kWh. Convert fuel consumption "
        "using appropriate energy content factors (e.g., DEFRA, EIA)."
    ),
    "epa_monthly_reporting": (
        "Total emissions exceed {threshold} tCO2e/yr. "
        "EPA 40 CFR Part 98 requires monthly reporting for large emitters."
    ),
    "missing_transport_category": (
        "Separate transport emissions as a distinct category. "
        "Both CSRD/ESRS E1 and UK SECR require transport to be "
        "disclosed separately from other Scope 1 sources."
    ),
    "base_year_recalculation": (
        "Fleet change of {change_pct}% exceeds the 5% threshold. "
        "Recalculate base year emissions to maintain consistency "
        "per GHG Protocol guidance."
    ),
    "missing_verification": (
        "Obtain third-party verification of the emissions report. "
        "{framework} requires or strongly recommends independent "
        "verification."
    ),
    "missing_fuel_measurement": (
        "Implement calibrated fuel measurement for Tier 3 compliance. "
        "EU ETS MRR requires metered fuel data with documented "
        "calibration records."
    ),
}


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class ComplianceFinding:
    """Individual compliance finding with severity and detail.

    Attributes:
        finding_id: Unique identifier for this finding.
        requirement_id: The requirement ID from the framework.
        requirement_text: Human-readable requirement description.
        category: Requirement category (methodology, reporting, etc.).
        severity: Finding severity level.
        status: Whether this specific requirement is met.
        detail: Detailed explanation of the finding.
        data_present: Whether the required data was found.
    """

    finding_id: str
    requirement_id: str
    requirement_text: str
    category: str
    severity: str
    status: str
    detail: str
    data_present: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "finding_id": self.finding_id,
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "detail": self.detail,
            "data_present": self.data_present,
        }


@dataclass
class ComplianceCheckResult:
    """Complete compliance check result with provenance.

    Attributes:
        result_id: Unique identifier for this compliance check.
        framework: Regulatory framework assessed.
        status: Overall compliance status.
        findings: List of individual compliance findings.
        total_requirements: Total number of requirements checked.
        mandatory_requirements: Number of mandatory requirements.
        mandatory_met: Number of mandatory requirements met.
        optional_requirements: Number of optional requirements.
        optional_met: Number of optional requirements met.
        compliance_score_pct: Percentage of mandatory requirements met.
        recommendations: List of actionable recommendations.
        provenance_hash: SHA-256 hash of the compliance assessment.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    result_id: str
    framework: str
    status: str
    findings: List[ComplianceFinding]
    total_requirements: int
    mandatory_requirements: int
    mandatory_met: int
    optional_requirements: int
    optional_met: int
    compliance_score_pct: Decimal
    recommendations: List[str]
    provenance_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "framework": self.framework,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "total_requirements": self.total_requirements,
            "mandatory_requirements": self.mandatory_requirements,
            "mandatory_met": self.mandatory_met,
            "optional_requirements": self.optional_requirements,
            "optional_met": self.optional_met,
            "compliance_score_pct": str(self.compliance_score_pct),
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================


class ComplianceCheckerEngine:
    """Regulatory compliance validation engine for mobile combustion
    emission calculations across six regulatory frameworks.

    Provides deterministic, zero-hallucination compliance assessment
    against GHG Protocol, ISO 14064-1, CSRD/ESRS E1, EPA 40 CFR
    Part 98, UK SECR, and EU ETS MRR.

    The engine checks:
        - Data completeness against framework-specific field requirements
        - Methodology validity (accepted methods and tiers per framework)
        - Reporting requirements (per-gas, intensity, transport category)
        - Quality requirements (uncertainty, verification, documentation)
        - Threshold checks (EPA large emitter, base year recalculation)

    Thread Safety:
        All mutable state (_compliance_history) is protected by a
        reentrant lock.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(
        ...     calculation_results={"total_co2e_kg": 50000, "method": "FUEL_BASED"},
        ...     framework="GHG_PROTOCOL",
        ... )
        >>> print(result.status)
    """

    def __init__(self) -> None:
        """Initialize the ComplianceCheckerEngine."""
        self._compliance_history: List[ComplianceCheckResult] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "ComplianceCheckerEngine initialized: "
            "%d frameworks supported, "
            "%d total requirements",
            len(RegulatoryFramework),
            sum(len(reqs) for reqs in _FRAMEWORK_REQUIREMENTS.values()),
        )

    # ------------------------------------------------------------------
    # Public API: Single Framework Check
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        calculation_results: Dict[str, Any],
        framework: str,
    ) -> ComplianceCheckResult:
        """Check compliance against a specific regulatory framework.

        Evaluates the provided calculation results against all requirements
        of the specified framework, producing individual findings and
        an overall compliance status.

        Args:
            calculation_results: Dictionary of calculation results and
                metadata. Should include keys like total_co2e_kg, method,
                tier, gases, fuel_type, etc. The exact keys required
                depend on the framework.
            framework: Regulatory framework string matching a
                RegulatoryFramework enum value.

        Returns:
            ComplianceCheckResult with complete assessment details.

        Raises:
            ValueError: If framework is not recognized.
        """
        t_start = time.monotonic()

        self._validate_framework(framework)

        requirements = _FRAMEWORK_REQUIREMENTS[framework]
        findings: List[ComplianceFinding] = []
        recommendations: List[str] = []

        # Check data completeness
        completeness_findings = self.validate_data_completeness(
            calculation_results, framework
        )
        findings.extend(completeness_findings)

        # Check methodology
        methodology_findings = self.validate_methodology(
            calculation_results.get("method", ""),
            calculation_results.get("tier", "TIER_1"),
            framework,
        )
        findings.extend(methodology_findings)

        # Check reporting requirements
        reporting_findings = self.validate_reporting_requirements(
            calculation_results, framework
        )
        findings.extend(reporting_findings)

        # Check intensity metrics
        intensity_findings = self.validate_intensity_metrics(
            calculation_results
        )
        # Only add intensity findings for frameworks that require them
        if framework in {
            RegulatoryFramework.CSRD_ESRS_E1.value,
            RegulatoryFramework.UK_SECR.value,
        }:
            findings.extend(intensity_findings)

        # Framework-specific checks
        framework_specific = self._framework_specific_checks(
            calculation_results, framework
        )
        findings.extend(framework_specific)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            findings, framework, calculation_results
        )

        # Calculate compliance status
        mandatory_count = sum(
            1 for r in requirements if r.get("mandatory", True)
        )
        optional_count = sum(
            1 for r in requirements if not r.get("mandatory", True)
        )

        mandatory_failures = sum(
            1 for f in findings
            if f.severity in {FindingSeverity.CRITICAL.value, FindingSeverity.MAJOR.value}
            and f.status != ComplianceStatus.COMPLIANT.value
        )
        mandatory_met = mandatory_count - min(mandatory_failures, mandatory_count)

        optional_met = sum(
            1 for f in findings
            if f.severity in {FindingSeverity.MINOR.value, FindingSeverity.INFORMATIONAL.value}
            and f.status == ComplianceStatus.COMPLIANT.value
        )
        optional_met = min(optional_met, optional_count)

        # Overall status
        if mandatory_failures == 0:
            overall_status = ComplianceStatus.COMPLIANT.value
        elif mandatory_failures <= 2:
            overall_status = ComplianceStatus.NEEDS_REVIEW.value
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT.value

        # Compliance score
        compliance_score = Decimal("0")
        if mandatory_count > 0:
            compliance_score = (
                Decimal(str(mandatory_met)) / Decimal(str(mandatory_count))
                * Decimal("100")
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Provenance hash
        provenance_data = {
            "framework": framework,
            "status": overall_status,
            "mandatory_met": mandatory_met,
            "mandatory_count": mandatory_count,
            "findings_count": len(findings),
            "total_co2e_kg": str(
                calculation_results.get("total_co2e_kg", 0)
            ),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        result = ComplianceCheckResult(
            result_id=f"cc_{uuid4().hex[:12]}",
            framework=framework,
            status=overall_status,
            findings=findings,
            total_requirements=len(requirements),
            mandatory_requirements=mandatory_count,
            mandatory_met=mandatory_met,
            optional_requirements=optional_count,
            optional_met=optional_met,
            compliance_score_pct=compliance_score,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

        # Record in history
        with self._lock:
            self._compliance_history.append(result)

        # Provenance tracking
        self._record_provenance(
            "compliance_check", result.result_id, provenance_data
        )

        # Metrics
        elapsed = time.monotonic() - t_start
        self._record_metrics(framework, elapsed)

        logger.debug(
            "Compliance checked: framework=%s status=%s "
            "mandatory=%d/%d findings=%d in %.1fms",
            framework, overall_status,
            mandatory_met, mandatory_count,
            len(findings), elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: All Frameworks Check
    # ------------------------------------------------------------------

    def check_all_frameworks(
        self,
        calculation_results: Dict[str, Any],
    ) -> Dict[str, ComplianceCheckResult]:
        """Check compliance against all supported frameworks.

        Runs check_compliance for each of the six supported frameworks
        and returns all results.

        Args:
            calculation_results: Dictionary of calculation results.

        Returns:
            Dictionary mapping framework names to ComplianceCheckResult
            objects.
        """
        results: Dict[str, ComplianceCheckResult] = {}
        for fw in RegulatoryFramework:
            results[fw.value] = self.check_compliance(
                calculation_results, fw.value
            )
        return results

    # ------------------------------------------------------------------
    # Public API: Data Completeness Validation
    # ------------------------------------------------------------------

    def validate_data_completeness(
        self,
        data: Dict[str, Any],
        framework: str,
    ) -> List[ComplianceFinding]:
        """Validate data completeness against framework requirements.

        Checks whether all required fields specified by the framework
        are present and non-empty in the provided data dictionary.

        Args:
            data: Calculation results dictionary.
            framework: Regulatory framework string.

        Returns:
            List of ComplianceFinding objects for completeness checks.

        Raises:
            ValueError: If framework is not recognized.
        """
        self._validate_framework(framework)

        requirements = _FRAMEWORK_REQUIREMENTS[framework]
        findings: List[ComplianceFinding] = []

        for req in requirements:
            req_id = req["id"]
            req_text = req["requirement"]
            category = req.get("category", "general")
            mandatory = req.get("mandatory", True)
            fields = req.get("fields_required", [])

            # Check if all required fields are present and non-empty
            all_present = True
            missing_fields: List[str] = []

            for field_name in fields:
                value = data.get(field_name)
                if value is None or value == "" or value == {} or value == []:
                    all_present = False
                    missing_fields.append(field_name)

            if all_present:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=req_id,
                    requirement_text=req_text,
                    category=category,
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail=f"All required fields present: {', '.join(fields)}",
                    data_present=True,
                ))
            else:
                severity = (
                    FindingSeverity.CRITICAL.value if mandatory
                    else FindingSeverity.MINOR.value
                )
                status = (
                    ComplianceStatus.NON_COMPLIANT.value if mandatory
                    else ComplianceStatus.NEEDS_REVIEW.value
                )
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=req_id,
                    requirement_text=req_text,
                    category=category,
                    severity=severity,
                    status=status,
                    detail=(
                        f"Missing required fields: {', '.join(missing_fields)}"
                    ),
                    data_present=False,
                ))

        return findings

    # ------------------------------------------------------------------
    # Public API: Methodology Validation
    # ------------------------------------------------------------------

    def validate_methodology(
        self,
        method: str,
        tier: str,
        framework: str,
    ) -> List[ComplianceFinding]:
        """Validate calculation methodology against framework rules.

        Checks whether the specified method and tier are accepted
        by the framework and provides findings accordingly.

        Args:
            method: Calculation method (FUEL_BASED, DISTANCE_BASED, etc.).
            tier: Calculation tier (TIER_1, TIER_2, TIER_3).
            framework: Regulatory framework string.

        Returns:
            List of ComplianceFinding objects for methodology checks.

        Raises:
            ValueError: If framework is not recognized.
        """
        self._validate_framework(framework)

        findings: List[ComplianceFinding] = []

        # Check method validity
        valid_methods = _VALID_METHODS_BY_FRAMEWORK.get(framework, [])
        if method in valid_methods:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id=f"{framework}-METHOD",
                requirement_text=f"Calculation method accepted by {framework}",
                category="methodology",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Method '{method}' is accepted. Valid: {valid_methods}",
                data_present=True,
            ))
        elif method:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id=f"{framework}-METHOD",
                requirement_text=f"Calculation method accepted by {framework}",
                category="methodology",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail=(
                    f"Method '{method}' is NOT accepted by {framework}. "
                    f"Valid methods: {valid_methods}"
                ),
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id=f"{framework}-METHOD",
                requirement_text=f"Calculation method accepted by {framework}",
                category="methodology",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="No calculation method specified.",
                data_present=False,
            ))

        # Check tier validity
        valid_tiers = _VALID_TIERS_BY_FRAMEWORK.get(framework, [])
        if tier in valid_tiers:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id=f"{framework}-TIER",
                requirement_text=f"Calculation tier accepted by {framework}",
                category="methodology",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Tier '{tier}' is accepted. Valid: {valid_tiers}",
                data_present=True,
            ))

            # Recommend higher tier if using Tier 1
            if tier == "TIER_1" and framework in {
                RegulatoryFramework.EPA_PART_98.value,
                RegulatoryFramework.EU_ETS_MRR.value,
            }:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-TIER-UPGRADE",
                    requirement_text="Consider upgrading to higher tier",
                    category="methodology",
                    severity=FindingSeverity.INFORMATIONAL.value,
                    status=ComplianceStatus.NEEDS_REVIEW.value,
                    detail=(
                        f"{framework} recommends Tier 2 or Tier 3 for "
                        f"improved accuracy. Currently using {tier}."
                    ),
                    data_present=True,
                ))
        elif tier:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id=f"{framework}-TIER",
                requirement_text=f"Calculation tier accepted by {framework}",
                category="methodology",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail=(
                    f"Tier '{tier}' is not recognized by {framework}. "
                    f"Valid tiers: {valid_tiers}"
                ),
                data_present=True,
            ))

        return findings

    # ------------------------------------------------------------------
    # Public API: Reporting Requirements Validation
    # ------------------------------------------------------------------

    def validate_reporting_requirements(
        self,
        results: Dict[str, Any],
        framework: str,
    ) -> List[ComplianceFinding]:
        """Validate reporting requirements for a framework.

        Checks framework-specific reporting requirements such as
        per-gas disclosure, biogenic CO2 separation, energy
        consumption reporting, and transport category separation.

        Args:
            results: Calculation results dictionary.
            framework: Regulatory framework string.

        Returns:
            List of ComplianceFinding objects for reporting checks.

        Raises:
            ValueError: If framework is not recognized.
        """
        self._validate_framework(framework)

        findings: List[ComplianceFinding] = []

        # Per-gas reporting check (CO2, CH4, N2O)
        gases = results.get("gases", {})
        required_gases = ["CO2", "CH4", "N2O"]

        if framework in {
            RegulatoryFramework.GHG_PROTOCOL.value,
            RegulatoryFramework.ISO_14064.value,
            RegulatoryFramework.EPA_PART_98.value,
        }:
            missing_gases = [
                g for g in required_gases
                if g not in gases or gases[g] is None
            ]
            if not missing_gases:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-GASES",
                    requirement_text="Per-gas emission reporting (CO2, CH4, N2O)",
                    category="reporting",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail="All required gases reported: CO2, CH4, N2O",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-GASES",
                    requirement_text="Per-gas emission reporting (CO2, CH4, N2O)",
                    category="reporting",
                    severity=FindingSeverity.MAJOR.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail=f"Missing gas reporting: {', '.join(missing_gases)}",
                    data_present=False,
                ))

        # Biogenic CO2 check (GHG Protocol)
        if framework == RegulatoryFramework.GHG_PROTOCOL.value:
            biogenic = results.get("biogenic_co2_separated")
            if biogenic:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="GHG-MC-003",
                    requirement_text="Separate biogenic CO2 reporting",
                    category="reporting",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail="Biogenic CO2 is separated from fossil CO2.",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="GHG-MC-003",
                    requirement_text="Separate biogenic CO2 reporting",
                    category="reporting",
                    severity=FindingSeverity.MAJOR.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail=(
                        "Biogenic CO2 is not separated from fossil CO2. "
                        "GHG Protocol requires separate biogenic reporting."
                    ),
                    data_present=False,
                ))

        # GWP source disclosure (GHG Protocol)
        if framework == RegulatoryFramework.GHG_PROTOCOL.value:
            gwp_source = results.get("gwp_source")
            if gwp_source:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="GHG-MC-005",
                    requirement_text="GWP source disclosure",
                    category="transparency",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail=f"GWP source disclosed: {gwp_source}",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="GHG-MC-005",
                    requirement_text="GWP source disclosure",
                    category="transparency",
                    severity=FindingSeverity.MAJOR.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail="GWP source (e.g., AR5, AR6) not disclosed.",
                    data_present=False,
                ))

        # Energy consumption in kWh (UK SECR, CSRD)
        if framework in {
            RegulatoryFramework.UK_SECR.value,
            RegulatoryFramework.CSRD_ESRS_E1.value,
        }:
            energy_kwh = results.get("energy_consumption_kwh")
            if energy_kwh is not None and energy_kwh != "":
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-ENERGY",
                    requirement_text="Energy consumption reported in kWh",
                    category="reporting",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail=f"Energy consumption reported: {energy_kwh} kWh",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-ENERGY",
                    requirement_text="Energy consumption reported in kWh",
                    category="reporting",
                    severity=FindingSeverity.MAJOR.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail="Energy consumption in kWh not reported.",
                    data_present=False,
                ))

        # Transport category separation (CSRD, UK SECR)
        if framework in {
            RegulatoryFramework.CSRD_ESRS_E1.value,
            RegulatoryFramework.UK_SECR.value,
        }:
            transport_sep = results.get("transport_category_separated")
            if transport_sep:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-TRANSPORT",
                    requirement_text="Transport as separate emission category",
                    category="reporting",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail="Transport emissions reported as separate category.",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id=f"{framework}-TRANSPORT",
                    requirement_text="Transport as separate emission category",
                    category="reporting",
                    severity=FindingSeverity.MAJOR.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail="Transport not separated as a distinct category.",
                    data_present=False,
                ))

        return findings

    # ------------------------------------------------------------------
    # Public API: Intensity Metrics Validation
    # ------------------------------------------------------------------

    def validate_intensity_metrics(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Validate presence of intensity metrics.

        Checks whether required intensity metrics (tCO2e/revenue,
        tCO2e/FTE) are present in the results. Required by
        CSRD/ESRS E1 and UK SECR.

        Args:
            results: Calculation results dictionary.

        Returns:
            List of ComplianceFinding objects for intensity checks.
        """
        findings: List[ComplianceFinding] = []

        # Revenue intensity
        revenue_intensity = results.get("intensity_per_revenue")
        if revenue_intensity is not None:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="INTENSITY-REVENUE",
                requirement_text="Revenue intensity metric (tCO2e/revenue)",
                category="metrics",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Revenue intensity: {revenue_intensity}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="INTENSITY-REVENUE",
                requirement_text="Revenue intensity metric (tCO2e/revenue)",
                category="metrics",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Revenue intensity metric not provided.",
                data_present=False,
            ))

        # FTE intensity
        fte_intensity = results.get("intensity_per_fte")
        if fte_intensity is not None:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="INTENSITY-FTE",
                requirement_text="FTE intensity metric (tCO2e/FTE)",
                category="metrics",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"FTE intensity: {fte_intensity}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="INTENSITY-FTE",
                requirement_text="FTE intensity metric (tCO2e/FTE)",
                category="metrics",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="FTE intensity metric not provided.",
                data_present=False,
            ))

        return findings

    # ------------------------------------------------------------------
    # Public API: Recommendations Generation
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        findings: List[ComplianceFinding],
        framework: str,
        calculation_results: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate actionable recommendations from compliance findings.

        Analyses the findings to produce specific, actionable
        recommendations for addressing compliance gaps.

        Args:
            findings: List of ComplianceFinding objects.
            framework: Regulatory framework string.
            calculation_results: Optional calculation results for context.

        Returns:
            List of recommendation strings.
        """
        if calculation_results is None:
            calculation_results = {}

        recommendations: List[str] = []
        seen_rec_types: set = set()

        for finding in findings:
            if finding.status == ComplianceStatus.COMPLIANT.value:
                continue

            rec = self._finding_to_recommendation(
                finding, framework, calculation_results
            )
            if rec and rec not in recommendations:
                recommendations.append(rec)

        # Add general recommendations based on missing data
        if not calculation_results.get("uncertainty_assessment"):
            rec = _RECOMMENDATION_TEMPLATES["missing_uncertainty"]
            if rec not in recommendations:
                recommendations.append(rec)

        if not calculation_results.get("base_year"):
            rec = _RECOMMENDATION_TEMPLATES["missing_base_year"]
            if rec not in recommendations:
                recommendations.append(rec)

        return recommendations

    # ------------------------------------------------------------------
    # Public API: Base Year Threshold Check
    # ------------------------------------------------------------------

    def check_base_year_threshold(
        self,
        current_fleet_emissions: Decimal,
        base_fleet_emissions: Decimal,
    ) -> bool:
        """Check if fleet emissions change exceeds 5% threshold.

        Per GHG Protocol guidance, the base year should be recalculated
        when structural changes (acquisitions, divestitures, changes in
        methodology) result in a >5% change in emissions.

        Args:
            current_fleet_emissions: Current period total emissions.
            base_fleet_emissions: Base year total emissions.

        Returns:
            True if the change exceeds the 5% threshold (recalculation
            needed), False otherwise.

        Raises:
            ValueError: If either value is negative.
        """
        current = _to_decimal(current_fleet_emissions)
        base = _to_decimal(base_fleet_emissions)

        if current < Decimal("0"):
            raise ValueError(
                f"current_fleet_emissions must be >= 0, got {current}"
            )
        if base < Decimal("0"):
            raise ValueError(
                f"base_fleet_emissions must be >= 0, got {base}"
            )

        if base == Decimal("0"):
            return current > Decimal("0")

        change_pct = abs(current - base) / base

        exceeds = change_pct > _BASE_YEAR_RECALC_THRESHOLD

        logger.debug(
            "Base year check: current=%.1f base=%.1f change=%.2f%% "
            "threshold=%.1f%% exceeds=%s",
            current, base, change_pct * 100,
            float(_BASE_YEAR_RECALC_THRESHOLD) * 100, exceeds,
        )

        return exceeds

    # ------------------------------------------------------------------
    # Public API: History
    # ------------------------------------------------------------------

    def get_compliance_history(self) -> List[ComplianceCheckResult]:
        """Return a copy of the compliance check history.

        Returns:
            List of all ComplianceCheckResult objects produced.
        """
        with self._lock:
            return list(self._compliance_history)

    def clear_history(self) -> int:
        """Clear the compliance check history.

        Returns:
            Number of records cleared.
        """
        with self._lock:
            count = len(self._compliance_history)
            self._compliance_history.clear()
        logger.info("Compliance history cleared: %d records", count)
        return count

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Get the list of requirements for a framework.

        Args:
            framework: Regulatory framework string.

        Returns:
            List of requirement dictionaries.

        Raises:
            ValueError: If framework is not recognized.
        """
        self._validate_framework(framework)
        return list(_FRAMEWORK_REQUIREMENTS[framework])

    def list_frameworks(self) -> List[str]:
        """List all supported regulatory frameworks.

        Returns:
            Sorted list of framework identifiers.
        """
        return sorted(fw.value for fw in RegulatoryFramework)

    # ------------------------------------------------------------------
    # Internal: Framework-Specific Checks
    # ------------------------------------------------------------------

    def _framework_specific_checks(
        self,
        results: Dict[str, Any],
        framework: str,
    ) -> List[ComplianceFinding]:
        """Run framework-specific validation checks.

        Args:
            results: Calculation results dictionary.
            framework: Regulatory framework string.

        Returns:
            List of ComplianceFinding objects.
        """
        if framework == RegulatoryFramework.GHG_PROTOCOL.value:
            return self._ghg_protocol_specific(results)
        elif framework == RegulatoryFramework.ISO_14064.value:
            return self._iso_14064_specific(results)
        elif framework == RegulatoryFramework.CSRD_ESRS_E1.value:
            return self._csrd_specific(results)
        elif framework == RegulatoryFramework.EPA_PART_98.value:
            return self._epa_specific(results)
        elif framework == RegulatoryFramework.UK_SECR.value:
            return self._uk_secr_specific(results)
        elif framework == RegulatoryFramework.EU_ETS_MRR.value:
            return self._eu_ets_specific(results)
        return []

    def _ghg_protocol_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """GHG Protocol-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Base year recalculation check
        base_year_emissions = results.get("base_year_emissions")
        current_emissions = results.get("total_co2e_kg")
        if base_year_emissions is not None and current_emissions is not None:
            try:
                exceeds = self.check_base_year_threshold(
                    _to_decimal(current_emissions),
                    _to_decimal(base_year_emissions),
                )
                if exceeds:
                    base = _to_decimal(base_year_emissions)
                    current = _to_decimal(current_emissions)
                    change = abs(current - base) / base * Decimal("100") if base > 0 else Decimal("100")
                    findings.append(ComplianceFinding(
                        finding_id=f"f_{uuid4().hex[:8]}",
                        requirement_id="GHG-MC-006",
                        requirement_text="Base year recalculation check",
                        category="consistency",
                        severity=FindingSeverity.MAJOR.value,
                        status=ComplianceStatus.NEEDS_REVIEW.value,
                        detail=(
                            f"Fleet emissions changed by {change:.1f}%, "
                            f"exceeding the 5% recalculation threshold."
                        ),
                        data_present=True,
                    ))
                else:
                    findings.append(ComplianceFinding(
                        finding_id=f"f_{uuid4().hex[:8]}",
                        requirement_id="GHG-MC-006",
                        requirement_text="Base year recalculation check",
                        category="consistency",
                        severity=FindingSeverity.PASS.value,
                        status=ComplianceStatus.COMPLIANT.value,
                        detail="Fleet change within 5% threshold.",
                        data_present=True,
                    ))
            except (ValueError, InvalidOperation):
                pass

        # Emission factor source documentation
        ef_source = results.get("emission_factor_source")
        if ef_source:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="GHG-MC-008",
                requirement_text="Document emission factors and sources",
                category="transparency",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Emission factor source: {ef_source}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="GHG-MC-008",
                requirement_text="Document emission factors and sources",
                category="transparency",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Emission factor source not documented.",
                data_present=False,
            ))

        return findings

    def _iso_14064_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """ISO 14064-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Uncertainty statement
        uncertainty = results.get("uncertainty_assessment")
        if uncertainty:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-003",
                requirement_text="Uncertainty statement required",
                category="quality",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Uncertainty assessment provided.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-003",
                requirement_text="Uncertainty statement required",
                category="quality",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail=(
                    "ISO 14064-1 requires an uncertainty statement. "
                    "Run uncertainty quantification analysis."
                ),
                data_present=False,
            ))

        # Quality management procedures
        quality_procs = results.get("quality_procedures")
        if quality_procs:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-004",
                requirement_text="Quality management procedures",
                category="quality",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Quality management procedures documented.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-004",
                requirement_text="Quality management procedures",
                category="quality",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Quality management procedures not documented.",
                data_present=False,
            ))

        # Assumptions and limitations documentation
        assumptions = results.get("assumptions_documented")
        if assumptions:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-005",
                requirement_text="Document assumptions and limitations",
                category="transparency",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Assumptions and limitations documented.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="ISO-MC-005",
                requirement_text="Document assumptions and limitations",
                category="transparency",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Assumptions and limitations not documented.",
                data_present=False,
            ))

        return findings

    def _csrd_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """CSRD/ESRS E1-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Mitigation actions
        mitigation = results.get("mitigation_actions")
        if mitigation:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-005",
                requirement_text="Mitigation actions disclosure",
                category="narrative",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Mitigation actions documented.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-005",
                requirement_text="Mitigation actions disclosure",
                category="narrative",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Mitigation actions not disclosed.",
                data_present=False,
            ))

        # Target pathway
        target = results.get("reduction_target")
        if target:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-004",
                requirement_text="Target pathway disclosure",
                category="target",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Reduction target disclosed: {target}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-004",
                requirement_text="Target pathway disclosure",
                category="target",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Emission reduction target not disclosed.",
                data_present=False,
            ))

        # Materiality assessment
        materiality = results.get("materiality_assessment")
        if materiality:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-008",
                requirement_text="Double materiality assessment",
                category="assessment",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Materiality assessment completed.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="CSRD-MC-008",
                requirement_text="Double materiality assessment",
                category="assessment",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Double materiality assessment not completed.",
                data_present=False,
            ))

        return findings

    def _epa_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """EPA 40 CFR Part 98-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Large emitter threshold check
        total_co2e_kg = _to_decimal(results.get("total_co2e_kg", 0))
        total_co2e_tonne = total_co2e_kg / Decimal("1000")

        if total_co2e_tonne >= _EPA_LARGE_EMITTER_THRESHOLD_TCO2E:
            # Check monthly reporting
            freq = results.get("reporting_frequency", "").lower()
            if freq == "monthly":
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="EPA-MC-002",
                    requirement_text="Monthly reporting for large emitters",
                    category="reporting",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail=(
                        f"Emissions ({total_co2e_tonne:.0f} tCO2e) exceed "
                        f"25,000 tCO2e threshold. Monthly reporting is in place."
                    ),
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="EPA-MC-002",
                    requirement_text="Monthly reporting for large emitters",
                    category="reporting",
                    severity=FindingSeverity.CRITICAL.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail=(
                        f"Emissions ({total_co2e_tonne:.0f} tCO2e) exceed "
                        f"25,000 tCO2e threshold. Monthly reporting REQUIRED "
                        f"but current frequency is '{freq or 'not specified'}'."
                    ),
                    data_present=True,
                ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="EPA-MC-002",
                requirement_text="Large emitter threshold check",
                category="reporting",
                severity=FindingSeverity.INFORMATIONAL.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=(
                    f"Emissions ({total_co2e_tonne:.0f} tCO2e) below "
                    f"25,000 tCO2e threshold. Standard reporting frequency applies."
                ),
                data_present=True,
            ))

        # Tier 3 fuel analysis
        tier = results.get("tier", "TIER_1")
        if tier == "TIER_3":
            fuel_analysis = results.get("fuel_analysis_data")
            if fuel_analysis:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="EPA-MC-005",
                    requirement_text="Fuel analysis data for Tier 3",
                    category="methodology",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail="Fuel analysis data provided for Tier 3.",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="EPA-MC-005",
                    requirement_text="Fuel analysis data for Tier 3",
                    category="methodology",
                    severity=FindingSeverity.CRITICAL.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail=(
                        "Tier 3 requires fuel analysis data (carbon content, "
                        "heating value). Data not provided."
                    ),
                    data_present=False,
                ))

        # Missing data procedures
        missing_procs = results.get("missing_data_procedures")
        if missing_procs:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="EPA-MC-009",
                requirement_text="Missing data substitution procedures (98.35)",
                category="quality",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Missing data procedures documented per 98.35.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="EPA-MC-009",
                requirement_text="Missing data substitution procedures (98.35)",
                category="quality",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Missing data procedures not documented per 40 CFR 98.35.",
                data_present=False,
            ))

        return findings

    def _uk_secr_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """UK SECR-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Methodology disclosure
        methodology = results.get("methodology_disclosed")
        if methodology:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-005",
                requirement_text="Methodology disclosure (DEFRA guidance)",
                category="transparency",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Methodology disclosed per DEFRA guidance.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-005",
                requirement_text="Methodology disclosure (DEFRA guidance)",
                category="transparency",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Methodology not disclosed. DEFRA/BEIS guidance required.",
                data_present=False,
            ))

        # UK vs global emissions separation
        uk_emissions = results.get("uk_emissions")
        global_emissions = results.get("global_emissions")
        if uk_emissions is not None and global_emissions is not None:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-007",
                requirement_text="UK and global emissions reported separately",
                category="reporting",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="UK and global emissions reported separately.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-007",
                requirement_text="UK and global emissions reported separately",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="UK and global emissions not reported separately.",
                data_present=False,
            ))

        # Previous period comparison
        prev_period = results.get("previous_period_emissions")
        if prev_period is not None:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-008",
                requirement_text="Comparison with previous reporting period",
                category="reporting",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Previous period emissions provided for comparison.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="SECR-MC-008",
                requirement_text="Comparison with previous reporting period",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Previous period emissions not provided.",
                data_present=False,
            ))

        return findings

    def _eu_ets_specific(
        self,
        results: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """EU ETS MRR-specific compliance checks.

        Args:
            results: Calculation results.

        Returns:
            List of findings.
        """
        findings: List[ComplianceFinding] = []

        # Calibrated fuel measurement for Tier 3
        tier = results.get("tier", "TIER_1")
        if tier in {"TIER_3"}:
            fuel_meas = results.get("fuel_measurement_method", "")
            if fuel_meas.lower() in {"calibrated_meter", "metered", "calibrated"}:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="MRR-MC-002",
                    requirement_text="Calibrated fuel measurement for Tier 3",
                    category="methodology",
                    severity=FindingSeverity.PASS.value,
                    status=ComplianceStatus.COMPLIANT.value,
                    detail=f"Calibrated measurement: {fuel_meas}",
                    data_present=True,
                ))
            else:
                findings.append(ComplianceFinding(
                    finding_id=f"f_{uuid4().hex[:8]}",
                    requirement_id="MRR-MC-002",
                    requirement_text="Calibrated fuel measurement for Tier 3",
                    category="methodology",
                    severity=FindingSeverity.CRITICAL.value,
                    status=ComplianceStatus.NON_COMPLIANT.value,
                    detail=(
                        f"Tier 3 requires calibrated fuel measurement. "
                        f"Current method: '{fuel_meas or 'not specified'}'."
                    ),
                    data_present=bool(fuel_meas),
                ))

        # Annual EF analysis
        ef_freq = results.get("ef_analysis_frequency", "")
        if ef_freq.lower() in {"annual", "yearly", "annually"}:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-003",
                requirement_text="Annual emission factor analysis",
                category="methodology",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"EF analysis frequency: {ef_freq}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-003",
                requirement_text="Annual emission factor analysis",
                category="methodology",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail=(
                    f"Annual emission factor analysis required. "
                    f"Current: '{ef_freq or 'not specified'}'."
                ),
                data_present=bool(ef_freq),
            ))

        # Verification
        verification = results.get("verification_status", "")
        if verification.lower() in {"verified", "completed", "approved"}:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-004",
                requirement_text="Verification by accredited verifier",
                category="verification",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail=f"Verification status: {verification}",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-004",
                requirement_text="Verification by accredited verifier",
                category="verification",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail=(
                    f"Verification by accredited verifier required. "
                    f"Current status: '{verification or 'not provided'}'."
                ),
                data_present=bool(verification),
            ))

        # Monitoring plan
        monitoring = results.get("monitoring_plan_approved")
        if monitoring:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-005",
                requirement_text="Monitoring plan approved",
                category="system",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="Monitoring plan approved by competent authority.",
                data_present=True,
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"f_{uuid4().hex[:8]}",
                requirement_id="MRR-MC-005",
                requirement_text="Monitoring plan approved",
                category="system",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Monitoring plan not approved or not provided.",
                data_present=False,
            ))

        return findings

    # ------------------------------------------------------------------
    # Internal: Finding to Recommendation
    # ------------------------------------------------------------------

    def _finding_to_recommendation(
        self,
        finding: ComplianceFinding,
        framework: str,
        results: Dict[str, Any],
    ) -> Optional[str]:
        """Convert a non-compliant finding to a recommendation.

        Args:
            finding: The compliance finding.
            framework: Regulatory framework.
            results: Calculation results for context.

        Returns:
            Recommendation string, or None if no specific recommendation.
        """
        req_id = finding.requirement_id

        # Gas reporting
        if "GASES" in req_id:
            gases = results.get("gases", {})
            missing = [
                g for g in ["CO2", "CH4", "N2O"]
                if g not in gases or gases[g] is None
            ]
            if missing:
                return _RECOMMENDATION_TEMPLATES["missing_gases"].format(
                    missing_gases=", ".join(missing)
                )

        # GWP source
        if "GWP" in finding.requirement_text.upper() or "gwp" in req_id.lower():
            return _RECOMMENDATION_TEMPLATES["missing_gwp_source"]

        # Base year
        if "base year" in finding.requirement_text.lower():
            if "recalcul" in finding.detail.lower():
                change_pct = "5+"
                return _RECOMMENDATION_TEMPLATES["base_year_recalculation"].format(
                    change_pct=change_pct
                )
            return _RECOMMENDATION_TEMPLATES["missing_base_year"]

        # Uncertainty
        if "uncertainty" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_uncertainty"]

        # Intensity
        if "intensity" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_intensity"]

        # Method validity
        if "METHOD" in req_id:
            method = results.get("method", "")
            valid = _VALID_METHODS_BY_FRAMEWORK.get(framework, [])
            return _RECOMMENDATION_TEMPLATES["invalid_method"].format(
                method=method,
                framework=framework,
                valid_methods=", ".join(valid),
            )

        # Tier upgrade
        if "TIER" in req_id and "UPGRADE" in req_id:
            tier = results.get("tier", "TIER_1")
            return _RECOMMENDATION_TEMPLATES["upgrade_tier"].format(
                current_tier=tier,
                framework=framework,
            )

        # Biogenic CO2
        if "biogenic" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_biogenic"]

        # Energy kWh
        if "energy" in finding.requirement_text.lower() and "kwh" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_energy_kwh"]

        # Transport category
        if "transport" in finding.requirement_text.lower() and "category" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_transport_category"]

        # Verification
        if "verification" in finding.requirement_text.lower() or "verifier" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_verification"].format(
                framework=framework
            )

        # Calibrated measurement
        if "calibrated" in finding.requirement_text.lower():
            return _RECOMMENDATION_TEMPLATES["missing_fuel_measurement"]

        # EPA monthly reporting
        if "monthly" in finding.requirement_text.lower() and "EPA" in framework:
            return _RECOMMENDATION_TEMPLATES["epa_monthly_reporting"].format(
                threshold="25,000"
            )

        # Generic recommendation based on finding detail
        if finding.detail:
            return f"Address: {finding.requirement_text}. {finding.detail}"

        return None

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_framework(self, framework: str) -> None:
        """Validate that framework is recognized.

        Args:
            framework: Framework string to validate.

        Raises:
            ValueError: If not recognized.
        """
        valid = {fw.value for fw in RegulatoryFramework}
        if framework not in valid:
            raise ValueError(
                f"Unrecognized framework '{framework}'. "
                f"Supported: {sorted(valid)}"
            )

    # ------------------------------------------------------------------
    # Internal: Provenance and Metrics
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record provenance tracking event if available.

        Args:
            action: Action description.
            entity_id: Entity identifier.
            data: Provenance data dictionary.
        """
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="compliance",
                    action=action,
                    entity_id=entity_id,
                    data=data,
                    metadata={"engine": "ComplianceCheckerEngine"},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

    def _record_metrics(self, framework: str, elapsed: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            framework: Framework for labelling.
            elapsed: Elapsed time in seconds.
        """
        if _METRICS_AVAILABLE and _record_compliance_check is not None:
            try:
                _record_compliance_check(framework, "complete")
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
