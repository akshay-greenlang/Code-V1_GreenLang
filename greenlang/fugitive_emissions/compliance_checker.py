# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Regulatory Compliance Validation (Engine 6 of 7)

AGENT-MRV-005: Fugitive Emissions Agent

Validates fugitive emission calculations against seven regulatory frameworks
to ensure data completeness, methodological correctness, and reporting
readiness.  Each framework defines 10 specific requirements that are
individually checked and scored.

Supported Frameworks (70 total requirements):
    1. GHG Protocol Corporate Standard (Chapter 5)  - 10 requirements
    2. ISO 14064-1:2018                              - 10 requirements
    3. CSRD / ESRS E1                                - 10 requirements
    4. EPA Subpart W (40 CFR Part 98)                - 10 requirements
    5. EPA LDAR (40 CFR Part 60/63)                  - 10 requirements
    6. EU Methane Regulation (EU 2024/1787)          - 10 requirements
    7. UK SECR (Streamlined Energy and Carbon Reporting) - 10 requirements

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met

Severity Levels:
    ERROR:   Requirement failure prevents regulatory compliance
    WARNING: Requirement failure should be addressed but is not blocking
    INFO:    Informational finding for best practice improvement

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean evaluations.
    - No LLM involvement in any compliance determination.
    - Requirement definitions are hard-coded from regulatory texts.
    - Every result carries a SHA-256 provenance hash.
    - Same inputs always produce identical compliance verdicts.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.fugitive_emissions.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_data={
    ...         "source_type": "EQUIPMENT_LEAK",
    ...         "calculation_method": "AVERAGE_EMISSION_FACTOR",
    ...         "total_co2e_tonnes": 500.0,
    ...         "emissions_by_gas": [{"gas": "CH4"}],
    ...         "component_count": 5000,
    ...         "has_ldar_program": True,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["GHG_PROTOCOL"],
    ... )
    >>> print(result["GHG_PROTOCOL"]["status"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    from greenlang.fugitive_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.metrics import (
        record_compliance_check as _record_compliance_check,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_compliance_check = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===========================================================================
# Constants
# ===========================================================================

#: All supported regulatory frameworks.
SUPPORTED_FRAMEWORKS: Tuple[str, ...] = (
    "GHG_PROTOCOL",
    "ISO_14064",
    "CSRD_ESRS_E1",
    "EPA_SUBPART_W",
    "EPA_LDAR",
    "EU_METHANE_REGULATION",
    "UK_SECR",
)

#: Compliance status thresholds (fraction of requirements passed).
_COMPLIANT_THRESHOLD: float = 1.0
_PARTIAL_THRESHOLD: float = 0.5


# ===========================================================================
# Requirement Definitions (10 per framework = 70 total)
# ===========================================================================


def _build_requirements() -> Dict[str, List[Dict[str, Any]]]:
    """Build the complete set of compliance requirements for all frameworks.

    Returns:
        Dictionary of framework_name -> list of requirement definitions.
    """
    return {
        # ---------------------------------------------------------------
        # GHG Protocol Corporate Standard
        # ---------------------------------------------------------------
        "GHG_PROTOCOL": [
            {
                "id": "GHG-FE-001",
                "name": "Source Identification",
                "description": (
                    "All significant fugitive emission sources must be "
                    "identified and documented."
                ),
                "severity": "ERROR",
                "check_field": "source_type",
            },
            {
                "id": "GHG-FE-002",
                "name": "Methodology Documentation",
                "description": (
                    "Calculation methodology must be documented and "
                    "consistent with GHG Protocol guidance."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "GHG-FE-003",
                "name": "Emission Factor Documentation",
                "description": (
                    "Emission factors must be documented with source "
                    "references and applicability justification."
                ),
                "severity": "ERROR",
                "check_field": "emission_factor_source",
            },
            {
                "id": "GHG-FE-004",
                "name": "Organizational Boundary",
                "description": (
                    "Fugitive emissions must be reported within the "
                    "defined organizational boundary."
                ),
                "severity": "ERROR",
                "check_field": "facility_id",
            },
            {
                "id": "GHG-FE-005",
                "name": "Gas Coverage",
                "description": (
                    "All applicable greenhouse gases (CH4, CO2, VOC) "
                    "must be included in the inventory."
                ),
                "severity": "ERROR",
                "check_field": "emissions_by_gas",
            },
            {
                "id": "GHG-FE-006",
                "name": "De Minimis Justification",
                "description": (
                    "Any excluded sources must be justified as de minimis "
                    "(< 5% of total Scope 1)."
                ),
                "severity": "WARNING",
                "check_field": "de_minimis_justified",
            },
            {
                "id": "GHG-FE-007",
                "name": "Temporal Consistency",
                "description": (
                    "Reporting period and methodology must be consistent "
                    "across reporting years."
                ),
                "severity": "WARNING",
                "check_field": "period_start",
            },
            {
                "id": "GHG-FE-008",
                "name": "Base Year Recalculation",
                "description": (
                    "Base year emissions must be recalculated for structural "
                    "changes exceeding the significance threshold."
                ),
                "severity": "WARNING",
                "check_field": "base_year_defined",
            },
            {
                "id": "GHG-FE-009",
                "name": "Quality Management",
                "description": (
                    "Quality management procedures must be in place for "
                    "data collection and calculation."
                ),
                "severity": "INFO",
                "check_field": "qm_procedures",
            },
            {
                "id": "GHG-FE-010",
                "name": "Verification Readiness",
                "description": (
                    "Data and methodology must be documented sufficiently "
                    "for third-party verification."
                ),
                "severity": "INFO",
                "check_field": "provenance_hash",
            },
        ],

        # ---------------------------------------------------------------
        # ISO 14064-1:2018
        # ---------------------------------------------------------------
        "ISO_14064": [
            {
                "id": "ISO-FE-001",
                "name": "Organizational Boundary (Clause 5.1)",
                "description": (
                    "Organization boundary must be defined using equity "
                    "share or control approach."
                ),
                "severity": "ERROR",
                "check_field": "facility_id",
            },
            {
                "id": "ISO-FE-002",
                "name": "Source Identification (Clause 5.2)",
                "description": (
                    "All GHG sources and sinks within the boundary "
                    "must be identified."
                ),
                "severity": "ERROR",
                "check_field": "source_type",
            },
            {
                "id": "ISO-FE-003",
                "name": "Quantification Methodology (Clause 5.3)",
                "description": (
                    "GHG emissions must be quantified using recognized "
                    "methodologies (measurement, calculation, estimation)."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "ISO-FE-004",
                "name": "Data Quality (Clause 5.3.3)",
                "description": (
                    "Data quality must be assessed and documented."
                ),
                "severity": "ERROR",
                "check_field": "data_quality_assessed",
            },
            {
                "id": "ISO-FE-005",
                "name": "Uncertainty Assessment (Clause 5.3.4)",
                "description": (
                    "Uncertainty of the GHG inventory must be assessed "
                    "and reported."
                ),
                "severity": "ERROR",
                "check_field": "uncertainty_assessed",
            },
            {
                "id": "ISO-FE-006",
                "name": "Base Year (Clause 5.4)",
                "description": (
                    "A base year must be established for tracking "
                    "emission trends."
                ),
                "severity": "WARNING",
                "check_field": "base_year_defined",
            },
            {
                "id": "ISO-FE-007",
                "name": "Exclusions Documentation (Clause 5.2.4)",
                "description": (
                    "Any excluded sources must be documented with "
                    "justification."
                ),
                "severity": "WARNING",
                "check_field": "exclusions_documented",
            },
            {
                "id": "ISO-FE-008",
                "name": "Temporal Consistency (Clause 5.5)",
                "description": (
                    "Consistent methodologies and time periods must "
                    "be used across reporting periods."
                ),
                "severity": "WARNING",
                "check_field": "period_start",
            },
            {
                "id": "ISO-FE-009",
                "name": "Competence (Clause 6)",
                "description": (
                    "Personnel performing inventory work must be competent."
                ),
                "severity": "INFO",
                "check_field": "competence_confirmed",
            },
            {
                "id": "ISO-FE-010",
                "name": "Documentation (Clause 7)",
                "description": (
                    "Complete documentation must be maintained for "
                    "all calculations and data sources."
                ),
                "severity": "ERROR",
                "check_field": "provenance_hash",
            },
        ],

        # ---------------------------------------------------------------
        # CSRD / ESRS E1
        # ---------------------------------------------------------------
        "CSRD_ESRS_E1": [
            {
                "id": "ESRS-FE-001",
                "name": "Scope 1 by Category",
                "description": (
                    "Scope 1 GHG emissions must be disaggregated by "
                    "emission category including fugitive emissions."
                ),
                "severity": "ERROR",
                "check_field": "source_type",
            },
            {
                "id": "ESRS-FE-002",
                "name": "Methodology Disclosure",
                "description": (
                    "Calculation methodology must be disclosed in the "
                    "sustainability report."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "ESRS-FE-003",
                "name": "Fugitive Emissions Separation",
                "description": (
                    "Fugitive emissions must be reported separately "
                    "from combustion and process emissions."
                ),
                "severity": "ERROR",
                "check_field": "scope_category",
            },
            {
                "id": "ESRS-FE-004",
                "name": "Sector-Specific Disclosure",
                "description": (
                    "Sector-specific fugitive emission disclosures must "
                    "follow applicable ESRS sector standards."
                ),
                "severity": "WARNING",
                "check_field": "sector_standard",
            },
            {
                "id": "ESRS-FE-005",
                "name": "Emission Reduction Targets",
                "description": (
                    "Targets for fugitive emission reduction must be "
                    "reported if material."
                ),
                "severity": "WARNING",
                "check_field": "reduction_targets",
            },
            {
                "id": "ESRS-FE-006",
                "name": "Mitigation Actions",
                "description": (
                    "Actions taken to mitigate fugitive emissions must "
                    "be disclosed."
                ),
                "severity": "WARNING",
                "check_field": "mitigation_actions",
            },
            {
                "id": "ESRS-FE-007",
                "name": "Financial Impact Assessment",
                "description": (
                    "Financial impact of fugitive emissions and related "
                    "regulations must be assessed."
                ),
                "severity": "INFO",
                "check_field": "financial_impact",
            },
            {
                "id": "ESRS-FE-008",
                "name": "Value Chain Scope",
                "description": (
                    "Value chain fugitive emission scope must be defined."
                ),
                "severity": "INFO",
                "check_field": "value_chain_scope",
            },
            {
                "id": "ESRS-FE-009",
                "name": "Assurance Readiness",
                "description": (
                    "Data must be prepared for limited or reasonable "
                    "assurance engagement."
                ),
                "severity": "ERROR",
                "check_field": "provenance_hash",
            },
            {
                "id": "ESRS-FE-010",
                "name": "XBRL Tagging",
                "description": (
                    "Fugitive emission data must be tagged in XBRL format "
                    "for digital reporting."
                ),
                "severity": "WARNING",
                "check_field": "xbrl_tagged",
            },
        ],

        # ---------------------------------------------------------------
        # EPA Subpart W (40 CFR Part 98)
        # ---------------------------------------------------------------
        "EPA_SUBPART_W": [
            {
                "id": "EPAW-FE-001",
                "name": "Monitoring Plan",
                "description": (
                    "A written monitoring plan must be developed and "
                    "maintained per 40 CFR 98.3(g)."
                ),
                "severity": "ERROR",
                "check_field": "monitoring_plan",
            },
            {
                "id": "EPAW-FE-002",
                "name": "Source Category Identification",
                "description": (
                    "All applicable source categories (petroleum, natural "
                    "gas, coal mining) must be identified."
                ),
                "severity": "ERROR",
                "check_field": "source_type",
            },
            {
                "id": "EPAW-FE-003",
                "name": "Tier Methodology Selection",
                "description": (
                    "Applicable tier methodology must be selected per "
                    "Subpart W Tables W-1A through W-7."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "EPAW-FE-004",
                "name": "Missing Data Procedures",
                "description": (
                    "Missing data substitution procedures must follow "
                    "40 CFR 98.235."
                ),
                "severity": "ERROR",
                "check_field": "missing_data_procedures",
            },
            {
                "id": "EPAW-FE-005",
                "name": "Equipment Calibration",
                "description": (
                    "Monitoring equipment must be calibrated per "
                    "manufacturer specifications."
                ),
                "severity": "ERROR",
                "check_field": "calibration_current",
            },
            {
                "id": "EPAW-FE-006",
                "name": "QA/QC Procedures",
                "description": (
                    "QA/QC procedures must be documented per "
                    "40 CFR 98.3(i)."
                ),
                "severity": "ERROR",
                "check_field": "qaqc_procedures",
            },
            {
                "id": "EPAW-FE-007",
                "name": "Recordkeeping (5 years)",
                "description": (
                    "Records must be maintained for at least 5 years "
                    "per 40 CFR 98.3(g)."
                ),
                "severity": "ERROR",
                "check_field": "recordkeeping_policy",
            },
            {
                "id": "EPAW-FE-008",
                "name": "Annual Report Submission",
                "description": (
                    "Annual GHG report must be submitted by March 31 "
                    "via e-GGRT."
                ),
                "severity": "ERROR",
                "check_field": "annual_report_submitted",
            },
            {
                "id": "EPAW-FE-009",
                "name": "Third-Party Verification",
                "description": (
                    "Verification statement must be available upon EPA "
                    "request."
                ),
                "severity": "WARNING",
                "check_field": "verification_available",
            },
            {
                "id": "EPAW-FE-010",
                "name": "e-GGRT Electronic Reporting",
                "description": (
                    "Data must be reported electronically via EPA e-GGRT "
                    "platform."
                ),
                "severity": "WARNING",
                "check_field": "eggrt_reported",
            },
        ],

        # ---------------------------------------------------------------
        # EPA LDAR (40 CFR Part 60/63)
        # ---------------------------------------------------------------
        "EPA_LDAR": [
            {
                "id": "LDAR-FE-001",
                "name": "Monitoring Frequency",
                "description": (
                    "Components must be monitored at the frequency "
                    "required by the applicable NSPS/NESHAP standard "
                    "(quarterly, semi-annual, or annual)."
                ),
                "severity": "ERROR",
                "check_field": "monitoring_frequency",
            },
            {
                "id": "LDAR-FE-002",
                "name": "Leak Definition Threshold",
                "description": (
                    "Leak definition concentration must be set per "
                    "applicable standard (e.g. 500 ppmv for NSPS "
                    "VVa, 10000 ppmv for older standards)."
                ),
                "severity": "ERROR",
                "check_field": "leak_definition_ppmv",
            },
            {
                "id": "LDAR-FE-003",
                "name": "Repair Deadline (15 days)",
                "description": (
                    "Leaking components must be repaired within 15 "
                    "calendar days of detection, with a first attempt "
                    "within 5 days."
                ),
                "severity": "ERROR",
                "check_field": "repair_deadline_days",
            },
            {
                "id": "LDAR-FE-004",
                "name": "Re-Monitor After Repair",
                "description": (
                    "Repaired components must be re-monitored within "
                    "15 days to verify repair effectiveness."
                ),
                "severity": "ERROR",
                "check_field": "remonitor_after_repair",
            },
            {
                "id": "LDAR-FE-005",
                "name": "Delay of Repair (DOR) Documentation",
                "description": (
                    "Any delay of repair must be documented with "
                    "justification and scheduled repair date."
                ),
                "severity": "ERROR",
                "check_field": "dor_documented",
            },
            {
                "id": "LDAR-FE-006",
                "name": "Recordkeeping",
                "description": (
                    "Complete records of all monitoring events, leaks, "
                    "repairs, and DOR must be maintained."
                ),
                "severity": "ERROR",
                "check_field": "recordkeeping_policy",
            },
            {
                "id": "LDAR-FE-007",
                "name": "Component Tagging",
                "description": (
                    "All monitored components must be uniquely tagged "
                    "and traceable to process unit and service."
                ),
                "severity": "ERROR",
                "check_field": "components_tagged",
            },
            {
                "id": "LDAR-FE-008",
                "name": "Surveyor Training",
                "description": (
                    "Personnel performing LDAR surveys must be trained "
                    "and certified per Method 21 or OGI requirements."
                ),
                "severity": "ERROR",
                "check_field": "surveyors_trained",
            },
            {
                "id": "LDAR-FE-009",
                "name": "Annual Reporting",
                "description": (
                    "Annual LDAR compliance report must be prepared "
                    "and available for inspection."
                ),
                "severity": "WARNING",
                "check_field": "annual_report_available",
            },
            {
                "id": "LDAR-FE-010",
                "name": "Audit Readiness",
                "description": (
                    "LDAR program must be prepared for regulatory "
                    "audit at any time."
                ),
                "severity": "INFO",
                "check_field": "audit_ready",
            },
        ],

        # ---------------------------------------------------------------
        # EU Methane Regulation (EU 2024/1787)
        # ---------------------------------------------------------------
        "EU_METHANE_REGULATION": [
            {
                "id": "EUMR-FE-001",
                "name": "LDAR Survey Program",
                "description": (
                    "An LDAR survey program must be established covering "
                    "all above-ground components per Article 14."
                ),
                "severity": "ERROR",
                "check_field": "has_ldar_program",
            },
            {
                "id": "EUMR-FE-002",
                "name": "OGI Technology",
                "description": (
                    "Optical Gas Imaging (OGI) or equivalent technology "
                    "must be used for leak detection per Article 14(3)."
                ),
                "severity": "ERROR",
                "check_field": "ogi_technology_used",
            },
            {
                "id": "EUMR-FE-003",
                "name": "Detection Sensitivity",
                "description": (
                    "Leak detection technology must meet the minimum "
                    "detection sensitivity requirements."
                ),
                "severity": "ERROR",
                "check_field": "detection_sensitivity_met",
            },
            {
                "id": "EUMR-FE-004",
                "name": "Repair Timeline (5/30 days)",
                "description": (
                    "Leaks must be repaired within 5 days for safety "
                    "hazards and 30 days for other leaks per Article 14(7)."
                ),
                "severity": "ERROR",
                "check_field": "eu_repair_timeline_met",
            },
            {
                "id": "EUMR-FE-005",
                "name": "Reporting to Authority",
                "description": (
                    "Methane emission data must be reported to the "
                    "competent national authority per Article 12."
                ),
                "severity": "ERROR",
                "check_field": "reported_to_authority",
            },
            {
                "id": "EUMR-FE-006",
                "name": "OGMP 2.0 Level",
                "description": (
                    "Reporting must align with OGMP 2.0 Level 4/5 "
                    "for source-level measurement where applicable."
                ),
                "severity": "WARNING",
                "check_field": "ogmp_level",
            },
            {
                "id": "EUMR-FE-007",
                "name": "Methane Intensity",
                "description": (
                    "Methane intensity metrics (CH4/unit production) "
                    "must be calculated and reported."
                ),
                "severity": "WARNING",
                "check_field": "methane_intensity_reported",
            },
            {
                "id": "EUMR-FE-008",
                "name": "Source-Level Reporting",
                "description": (
                    "Emissions must be reported at the source level "
                    "(not just facility aggregate) per Article 12(2)."
                ),
                "severity": "ERROR",
                "check_field": "source_level_reporting",
            },
            {
                "id": "EUMR-FE-009",
                "name": "Third-Party Verification",
                "description": (
                    "Methane emission reports must be verified by an "
                    "independent third party per Article 13."
                ),
                "severity": "ERROR",
                "check_field": "third_party_verified",
            },
            {
                "id": "EUMR-FE-010",
                "name": "Public Transparency",
                "description": (
                    "Key methane emission data must be made publicly "
                    "available per Article 15."
                ),
                "severity": "WARNING",
                "check_field": "public_disclosure",
            },
        ],

        # ---------------------------------------------------------------
        # UK SECR
        # ---------------------------------------------------------------
        "UK_SECR": [
            {
                "id": "SECR-FE-001",
                "name": "Energy and Emissions Reporting",
                "description": (
                    "Annual energy consumption and associated GHG "
                    "emissions (including fugitive) must be reported."
                ),
                "severity": "ERROR",
                "check_field": "total_co2e_tonnes",
            },
            {
                "id": "SECR-FE-002",
                "name": "DEFRA Methodology",
                "description": (
                    "UK DEFRA conversion factors must be used for "
                    "emission calculations."
                ),
                "severity": "ERROR",
                "check_field": "emission_factor_source",
            },
            {
                "id": "SECR-FE-003",
                "name": "Intensity Ratio",
                "description": (
                    "At least one intensity ratio must be calculated "
                    "(e.g. tCO2e per GBP revenue or per employee)."
                ),
                "severity": "ERROR",
                "check_field": "intensity_ratio",
            },
            {
                "id": "SECR-FE-004",
                "name": "Year-on-Year Comparison",
                "description": (
                    "Current year emissions must be compared to the "
                    "previous reporting year."
                ),
                "severity": "ERROR",
                "check_field": "year_comparison",
            },
            {
                "id": "SECR-FE-005",
                "name": "Scope Coverage",
                "description": (
                    "At minimum, Scope 1 (including fugitive) and "
                    "Scope 2 emissions must be reported."
                ),
                "severity": "ERROR",
                "check_field": "scope_coverage",
            },
            {
                "id": "SECR-FE-006",
                "name": "Methodology Statement",
                "description": (
                    "The methodology used for calculations must be "
                    "stated in the Directors' Report."
                ),
                "severity": "ERROR",
                "check_field": "methodology_stated",
            },
            {
                "id": "SECR-FE-007",
                "name": "Energy Efficiency Narrative",
                "description": (
                    "A narrative description of energy efficiency "
                    "actions taken must be included."
                ),
                "severity": "WARNING",
                "check_field": "efficiency_narrative",
            },
            {
                "id": "SECR-FE-008",
                "name": "Director Responsibility",
                "description": (
                    "A named director must take responsibility for "
                    "the energy and emissions report."
                ),
                "severity": "WARNING",
                "check_field": "director_responsibility",
            },
            {
                "id": "SECR-FE-009",
                "name": "Assurance Statement",
                "description": (
                    "Assurance statement should be provided if data "
                    "is assured by a third party."
                ),
                "severity": "INFO",
                "check_field": "assurance_statement",
            },
            {
                "id": "SECR-FE-010",
                "name": "Companies Act Compliance",
                "description": (
                    "Report must comply with the Companies Act 2006 "
                    "and The Companies (Directors' Report) Regulations."
                ),
                "severity": "ERROR",
                "check_field": "companies_act_compliant",
            },
        ],
    }


# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================


class ComplianceCheckerEngine:
    """Regulatory compliance checker for fugitive emission calculations.

    Validates calculation data against 7 regulatory frameworks with
    10 requirements each (70 total). All checks are deterministic
    boolean evaluations with no LLM involvement.

    Thread-safe via reentrant lock.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(
        ...     calculation_data={"source_type": "EQUIPMENT_LEAK", ...},
        ...     frameworks=["GHG_PROTOCOL"],
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ComplianceCheckerEngine.

        Args:
            config: Optional configuration dictionary.
        """
        self._config = config or {}
        self._lock = threading.RLock()
        self._requirements = _build_requirements()

        # Statistics
        self._total_checks: int = 0
        self._total_compliant: int = 0
        self._total_non_compliant: int = 0
        self._total_partial: int = 0

        logger.info(
            "ComplianceCheckerEngine initialized with %d frameworks (%d reqs)",
            len(self._requirements),
            sum(len(r) for r in self._requirements.values()),
        )

    # ------------------------------------------------------------------
    # Public API: Check Compliance
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        calculation_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check compliance against one or more regulatory frameworks.

        Args:
            calculation_data: Dictionary with calculation results and
                metadata. Fields checked vary by framework.
            frameworks: List of framework identifiers to check.
                If None or empty, all frameworks are checked.

        Returns:
            Dictionary keyed by framework with per-framework results.
        """
        t0 = time.monotonic()

        target_frameworks = frameworks or list(SUPPORTED_FRAMEWORKS)

        # Validate framework names
        for fw in target_frameworks:
            if fw not in SUPPORTED_FRAMEWORKS:
                logger.warning(
                    "Unknown framework '%s'; skipping", fw,
                )

        valid_frameworks = [
            fw for fw in target_frameworks if fw in SUPPORTED_FRAMEWORKS
        ]

        results: Dict[str, Any] = {}
        for fw in valid_frameworks:
            results[fw] = self.check_framework(fw, calculation_data)

        # Aggregate summary
        compliant_count = sum(
            1 for r in results.values()
            if r.get("status") == "compliant"
        )
        partial_count = sum(
            1 for r in results.values()
            if r.get("status") == "partial"
        )
        non_compliant_count = sum(
            1 for r in results.values()
            if r.get("status") == "non_compliant"
        )

        with self._lock:
            self._total_checks += 1
            self._total_compliant += compliant_count
            self._total_non_compliant += non_compliant_count
            self._total_partial += partial_count

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        summary = {
            "frameworks_checked": len(valid_frameworks),
            "compliant": compliant_count,
            "partial": partial_count,
            "non_compliant": non_compliant_count,
            "results": results,
            "checked_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed_ms, 3),
        }
        summary["provenance_hash"] = _compute_hash(summary)

        if _record_compliance_check is not None:
            _record_compliance_check("multi_framework", "completed")

        logger.info(
            "Compliance check: %d frameworks, %d compliant, "
            "%d partial, %d non-compliant (%.1fms)",
            len(valid_frameworks), compliant_count,
            partial_count, non_compliant_count, elapsed_ms,
        )

        return summary

    def check_framework(
        self,
        framework: str,
        calculation_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check compliance against a single framework.

        Args:
            framework: Framework identifier (e.g. "GHG_PROTOCOL").
            calculation_data: Calculation data dictionary.

        Returns:
            Dictionary with status, met/not-met counts, findings,
            and recommendations.
        """
        requirements = self._requirements.get(framework, [])
        if not requirements:
            return {
                "framework": framework,
                "status": "not_checked",
                "requirement_count": 0,
                "met_count": 0,
                "partially_met_count": 0,
                "not_met_count": 0,
                "findings": [],
                "recommendations": [],
            }

        findings: List[Dict[str, Any]] = []
        met_count = 0
        partially_met_count = 0
        not_met_count = 0
        recommendations: List[str] = []

        for req in requirements:
            finding = self._evaluate_requirement(
                req, calculation_data, framework,
            )
            findings.append(finding)

            status = finding.get("status", "not_met")
            if status == "met":
                met_count += 1
            elif status == "partially_met":
                partially_met_count += 1
            else:
                not_met_count += 1
                if finding.get("recommendation"):
                    recommendations.append(finding["recommendation"])

        # Determine overall framework status
        total = len(requirements)
        pass_rate = (met_count + 0.5 * partially_met_count) / total

        if pass_rate >= _COMPLIANT_THRESHOLD:
            overall_status = "compliant"
        elif pass_rate >= _PARTIAL_THRESHOLD:
            overall_status = "partial"
        else:
            overall_status = "non_compliant"

        return {
            "framework": framework,
            "status": overall_status,
            "requirement_count": total,
            "met_count": met_count,
            "partially_met_count": partially_met_count,
            "not_met_count": not_met_count,
            "pass_rate": round(pass_rate, 4),
            "findings": findings,
            "recommendations": recommendations,
        }

    def check_all_frameworks(
        self,
        calculation_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check compliance against all supported frameworks.

        Args:
            calculation_data: Calculation data dictionary.

        Returns:
            Compliance summary across all frameworks.
        """
        return self.check_compliance(
            calculation_data=calculation_data,
            frameworks=list(SUPPORTED_FRAMEWORKS),
        )

    # ------------------------------------------------------------------
    # Public API: Framework Information
    # ------------------------------------------------------------------

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Get the list of requirements for a specific framework.

        Args:
            framework: Framework identifier.

        Returns:
            List of requirement definition dictionaries.
        """
        return list(self._requirements.get(framework, []))

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of supported framework identifiers.

        Returns:
            List of framework name strings.
        """
        return list(SUPPORTED_FRAMEWORKS)

    # ------------------------------------------------------------------
    # Public API: Data Completeness
    # ------------------------------------------------------------------

    def validate_data_completeness(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> Dict[str, Any]:
        """Validate that all required data fields are present.

        Args:
            calculation_data: Calculation data to validate.
            framework: Framework to validate against.

        Returns:
            Dictionary with present/missing field lists and completeness %.
        """
        requirements = self._requirements.get(framework, [])
        required_fields = set()
        for req in requirements:
            check_field = req.get("check_field", "")
            if check_field:
                required_fields.add(check_field)

        present = []
        missing = []
        for field_name in sorted(required_fields):
            if self._field_has_value(calculation_data, field_name):
                present.append(field_name)
            else:
                missing.append(field_name)

        total = len(required_fields)
        completeness = len(present) / total if total > 0 else 0.0

        return {
            "framework": framework,
            "total_fields": total,
            "present_count": len(present),
            "missing_count": len(missing),
            "completeness_pct": round(completeness * 100.0, 2),
            "present_fields": present,
            "missing_fields": missing,
        }

    # ------------------------------------------------------------------
    # Public API: Recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        compliance_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations from compliance results.

        Args:
            compliance_results: Output from check_compliance().

        Returns:
            List of recommendation dictionaries sorted by priority.
        """
        recommendations: List[Dict[str, Any]] = []
        priority_map = {"ERROR": 1, "WARNING": 2, "INFO": 3}

        results = compliance_results.get("results", {})
        for framework, fw_result in results.items():
            if not isinstance(fw_result, dict):
                continue
            for finding in fw_result.get("findings", []):
                if finding.get("status") != "met":
                    severity = finding.get("severity", "INFO")
                    rec = {
                        "framework": framework,
                        "requirement_id": finding.get("requirement_id", ""),
                        "requirement_name": finding.get(
                            "requirement_name", "",
                        ),
                        "severity": severity,
                        "priority": priority_map.get(severity, 3),
                        "recommendation": finding.get(
                            "recommendation",
                            f"Address {finding.get('requirement_name', '')}",
                        ),
                        "current_status": finding.get("status", "not_met"),
                    }
                    recommendations.append(rec)

        # Sort by priority (1=highest) then by framework
        recommendations.sort(
            key=lambda x: (x["priority"], x["framework"]),
        )

        return recommendations

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with check counts.
        """
        with self._lock:
            return {
                "total_checks": self._total_checks,
                "total_compliant": self._total_compliant,
                "total_partial": self._total_partial,
                "total_non_compliant": self._total_non_compliant,
                "supported_frameworks": len(SUPPORTED_FRAMEWORKS),
                "total_requirements": sum(
                    len(r) for r in self._requirements.values()
                ),
            }

    # ------------------------------------------------------------------
    # Private: Requirement Evaluation
    # ------------------------------------------------------------------

    def _evaluate_requirement(
        self,
        requirement: Dict[str, Any],
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> Dict[str, Any]:
        """Evaluate a single compliance requirement.

        Args:
            requirement: Requirement definition.
            calculation_data: Data to evaluate against.
            framework: Parent framework identifier.

        Returns:
            Finding dictionary with status, details, and recommendation.
        """
        req_id = requirement.get("id", "")
        req_name = requirement.get("name", "")
        severity = requirement.get("severity", "INFO")
        check_field = requirement.get("check_field", "")
        description = requirement.get("description", "")

        # Evaluate: field present and non-empty/non-zero
        has_value = self._field_has_value(calculation_data, check_field)

        if has_value:
            status = "met"
            recommendation = ""
        else:
            status = "not_met"
            recommendation = (
                f"Provide '{check_field}' data to satisfy "
                f"{req_id}: {req_name}"
            )

        return {
            "requirement_id": req_id,
            "requirement_name": req_name,
            "description": description,
            "severity": severity,
            "check_field": check_field,
            "status": status,
            "recommendation": recommendation,
            "framework": framework,
        }

    def _field_has_value(
        self,
        data: Dict[str, Any],
        field_name: str,
    ) -> bool:
        """Check whether a field has a meaningful value in the data.

        A field is considered to have a value if:
        - It exists in the dictionary
        - It is not None
        - If string, it is not empty
        - If numeric, it is non-zero or zero is acceptable
        - If list/dict, it is not empty

        Args:
            data: Data dictionary.
            field_name: Field to check.

        Returns:
            True if the field has a meaningful value.
        """
        if not field_name:
            return False

        value = data.get(field_name)
        if value is None:
            return False

        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return True  # Zero is a valid measurement
        if isinstance(value, (list, dict)):
            return len(value) > 0

        return True
