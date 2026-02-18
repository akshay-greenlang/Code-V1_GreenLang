# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Regulatory Compliance Validation (Engine 6 of 7)

AGENT-MRV-006: Flaring Agent

Validates flaring emission calculations against eight regulatory frameworks
to ensure data completeness, methodological correctness, and reporting
readiness.  Each framework defines a specific set of requirements that are
individually checked and scored.

Supported Frameworks (83 total requirements):
    1. GHG Protocol Corporate Standard (Chapter 5)   - 12 requirements
    2. ISO 14064-1:2018                               -  8 requirements
    3. CSRD / ESRS E1                                 -  8 requirements
    4. EPA 40 CFR Part 98 Subpart W (Sec. W.23)      - 15 requirements
    5. EU ETS MRR                                     - 10 requirements
    6. EU Methane Regulation 2024/1787 (Art. 14)      - 12 requirements
    7. World Bank Zero Routine Flaring by 2030        -  8 requirements
    8. OGMP 2.0                                       - 10 requirements

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met

Severity Levels:
    CRITICAL: Failure prevents any regulatory compliance; must fix immediately.
    ERROR:    Requirement failure prevents regulatory compliance.
    WARNING:  Requirement failure should be addressed but is not blocking.
    INFO:     Informational finding for best practice improvement.

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean evaluations.
    - No LLM involvement in any compliance determination.
    - Requirement definitions are hard-coded from regulatory texts.
    - Every result carries a SHA-256 provenance hash.
    - Same inputs always produce identical compliance verdicts.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.flaring.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_data={
    ...         "flare_type": "ELEVATED_STEAM_ASSISTED",
    ...         "calculation_method": "GAS_COMPOSITION",
    ...         "total_co2e_tonnes": 1500.0,
    ...         "combustion_efficiency": 0.98,
    ...         "gas_composition_analyzed": True,
    ...         "flow_measurement_method": "ULTRASONIC",
    ...         "event_categories_tracked": True,
    ...         "routine_volume_scf": 50000,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["EPA_SUBPART_W"],
    ... )
    >>> print(result["results"]["EPA_SUBPART_W"]["status"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
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
    from greenlang.flaring.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
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
    "EU_ETS_MRR",
    "EU_METHANE_REGULATION",
    "WORLD_BANK_ZRF",
    "OGMP_2_0",
)

#: Compliance status thresholds (fraction of requirements passed).
_COMPLIANT_THRESHOLD: float = 1.0
_PARTIAL_THRESHOLD: float = 0.5

#: Severity ordering for prioritization (lower = more severe).
_SEVERITY_PRIORITY: Dict[str, int] = {
    "CRITICAL": 0,
    "ERROR": 1,
    "WARNING": 2,
    "INFO": 3,
}


# ===========================================================================
# Requirement Definitions (83 total across 8 frameworks)
# ===========================================================================


def _build_requirements() -> Dict[str, List[Dict[str, Any]]]:
    """Build the complete set of compliance requirements for all frameworks.

    Returns:
        Dictionary of framework_name -> list of requirement definitions.
    """
    return {
        # ---------------------------------------------------------------
        # GHG Protocol Corporate Standard (12 requirements)
        # ---------------------------------------------------------------
        "GHG_PROTOCOL": [
            {
                "id": "GHG-FL-001",
                "name": "Scope 1 Classification",
                "description": (
                    "Flaring emissions must be classified as Scope 1 "
                    "direct emissions from owned or controlled sources."
                ),
                "severity": "ERROR",
                "check_field": "scope_classification",
                "check_fn": "_check_scope1_classification",
            },
            {
                "id": "GHG-FL-002",
                "name": "Completeness",
                "description": (
                    "All flare systems must be included in the inventory "
                    "with volumes and emission calculations."
                ),
                "severity": "ERROR",
                "check_field": "flare_systems_complete",
            },
            {
                "id": "GHG-FL-003",
                "name": "Methodology Documentation",
                "description": (
                    "Calculation methodology must be documented and "
                    "consistent with GHG Protocol guidance."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "GHG-FL-004",
                "name": "Emission Factor Source",
                "description": (
                    "Emission factors must be documented with source "
                    "references and applicability justification."
                ),
                "severity": "ERROR",
                "check_field": "emission_factor_source",
            },
            {
                "id": "GHG-FL-005",
                "name": "Gas Coverage",
                "description": (
                    "All applicable greenhouse gases (CO2, CH4, N2O) "
                    "must be included in the flaring inventory."
                ),
                "severity": "ERROR",
                "check_field": "emissions_by_gas",
            },
            {
                "id": "GHG-FL-006",
                "name": "Biogenic CO2 Separation",
                "description": (
                    "Biogenic CO2 from flaring of biogas must be "
                    "reported separately from fossil CO2."
                ),
                "severity": "WARNING",
                "check_field": "biogenic_co2_separated",
            },
            {
                "id": "GHG-FL-007",
                "name": "GWP Source Disclosure",
                "description": (
                    "The source of GWP values (AR4, AR5, AR6) must "
                    "be disclosed in reporting."
                ),
                "severity": "ERROR",
                "check_field": "gwp_source",
            },
            {
                "id": "GHG-FL-008",
                "name": "Base Year Recalculation Triggers",
                "description": (
                    "Significant changes in flare systems (addition, "
                    "removal, process change) must trigger base year "
                    "recalculation."
                ),
                "severity": "WARNING",
                "check_field": "base_year_recalculation_policy",
            },
            {
                "id": "GHG-FL-009",
                "name": "Uncertainty Reporting",
                "description": (
                    "Uncertainty of flaring emission estimates must "
                    "be assessed and reported."
                ),
                "severity": "WARNING",
                "check_field": "uncertainty_assessed",
            },
            {
                "id": "GHG-FL-010",
                "name": "Temporal Consistency",
                "description": (
                    "Reporting period and methodology must be consistent "
                    "across reporting years."
                ),
                "severity": "WARNING",
                "check_field": "period_start",
            },
            {
                "id": "GHG-FL-011",
                "name": "Quality Management",
                "description": (
                    "Quality management procedures must be in place "
                    "for flaring data collection and calculation."
                ),
                "severity": "INFO",
                "check_field": "qm_procedures",
            },
            {
                "id": "GHG-FL-012",
                "name": "Verification Readiness",
                "description": (
                    "Data and methodology must be documented "
                    "sufficiently for third-party verification."
                ),
                "severity": "INFO",
                "check_field": "provenance_hash",
            },
        ],

        # ---------------------------------------------------------------
        # ISO 14064-1:2018 (8 requirements)
        # ---------------------------------------------------------------
        "ISO_14064": [
            {
                "id": "ISO-FL-001",
                "name": "Category 1 Classification (Clause 5.2.1)",
                "description": (
                    "Flaring emissions must be classified under "
                    "Category 1: Direct GHG emissions."
                ),
                "severity": "ERROR",
                "check_field": "scope_classification",
            },
            {
                "id": "ISO-FL-002",
                "name": "Quantification Methodology (Clause 5.3)",
                "description": (
                    "Flaring emissions must be quantified using "
                    "recognized methodologies per ISO 14064-1."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "ISO-FL-003",
                "name": "Uncertainty Assessment (Clause 5.3.4)",
                "description": (
                    "Uncertainty of flaring emission estimates must "
                    "be assessed and documented."
                ),
                "severity": "ERROR",
                "check_field": "uncertainty_assessed",
            },
            {
                "id": "ISO-FL-004",
                "name": "Documentation Completeness (Clause 7)",
                "description": (
                    "Complete documentation must be maintained for "
                    "all flaring calculations and data sources."
                ),
                "severity": "ERROR",
                "check_field": "provenance_hash",
            },
            {
                "id": "ISO-FL-005",
                "name": "Data Quality Assessment (Clause 5.3.3)",
                "description": (
                    "Quality of flaring activity data must be assessed."
                ),
                "severity": "ERROR",
                "check_field": "data_quality_assessed",
            },
            {
                "id": "ISO-FL-006",
                "name": "Source Identification (Clause 5.2)",
                "description": (
                    "All flare systems must be identified as GHG "
                    "sources within the organizational boundary."
                ),
                "severity": "ERROR",
                "check_field": "flare_type",
            },
            {
                "id": "ISO-FL-007",
                "name": "Base Year (Clause 5.4)",
                "description": (
                    "A base year must be established for tracking "
                    "flaring emission trends."
                ),
                "severity": "WARNING",
                "check_field": "base_year_defined",
            },
            {
                "id": "ISO-FL-008",
                "name": "Exclusions Documentation (Clause 5.2.4)",
                "description": (
                    "Any excluded flare systems must be documented "
                    "with quantitative justification."
                ),
                "severity": "WARNING",
                "check_field": "exclusions_documented",
            },
        ],

        # ---------------------------------------------------------------
        # CSRD / ESRS E1 (8 requirements)
        # ---------------------------------------------------------------
        "CSRD_ESRS_E1": [
            {
                "id": "ESRS-FL-001",
                "name": "E1-6 Gross Scope 1 Reporting",
                "description": (
                    "Flaring emissions must be reported as part of "
                    "gross Scope 1 GHG emissions under E1-6."
                ),
                "severity": "ERROR",
                "check_field": "total_co2e_tonnes",
            },
            {
                "id": "ESRS-FL-002",
                "name": "Methodology Disclosure",
                "description": (
                    "Flaring calculation methodology must be disclosed "
                    "in the sustainability report."
                ),
                "severity": "ERROR",
                "check_field": "calculation_method",
            },
            {
                "id": "ESRS-FL-003",
                "name": "Activity Data Documentation",
                "description": (
                    "Flaring activity data (volumes, flow rates, gas "
                    "composition) must be documented."
                ),
                "severity": "ERROR",
                "check_field": "gas_volume_documented",
            },
            {
                "id": "ESRS-FL-004",
                "name": "Verification Readiness",
                "description": (
                    "Flaring data must be prepared for limited or "
                    "reasonable assurance engagement."
                ),
                "severity": "ERROR",
                "check_field": "provenance_hash",
            },
            {
                "id": "ESRS-FL-005",
                "name": "Disaggregation by Source",
                "description": (
                    "Flaring emissions must be disaggregated from "
                    "other Scope 1 categories (combustion, fugitive, process)."
                ),
                "severity": "ERROR",
                "check_field": "disaggregated_reporting",
            },
            {
                "id": "ESRS-FL-006",
                "name": "Reduction Targets",
                "description": (
                    "Targets for flaring emission reduction must be "
                    "reported if material."
                ),
                "severity": "WARNING",
                "check_field": "reduction_targets",
            },
            {
                "id": "ESRS-FL-007",
                "name": "Mitigation Actions",
                "description": (
                    "Actions taken to reduce flaring (gas recovery, "
                    "flare gas utilization) must be disclosed."
                ),
                "severity": "WARNING",
                "check_field": "mitigation_actions",
            },
            {
                "id": "ESRS-FL-008",
                "name": "XBRL Tagging",
                "description": (
                    "Flaring emission data must be tagged in XBRL "
                    "format for digital reporting."
                ),
                "severity": "WARNING",
                "check_field": "xbrl_tagged",
            },
        ],

        # ---------------------------------------------------------------
        # EPA 40 CFR Part 98 Subpart W Section W.23 (15 requirements)
        # ---------------------------------------------------------------
        "EPA_SUBPART_W": [
            {
                "id": "EPAW-FL-001",
                "name": "Flare Stack Calculation Methodology",
                "description": (
                    "Flare stack emissions must be calculated per "
                    "40 CFR 98.233(n) using Equation W-19 or W-20."
                ),
                "severity": "CRITICAL",
                "check_field": "calculation_method",
                "check_fn": "_check_epa_methodology",
            },
            {
                "id": "EPAW-FL-002",
                "name": "Gas Flow Measurement or Estimation",
                "description": (
                    "Gas flow to each flare must be measured using "
                    "continuous flow meters or estimated using "
                    "engineering calculations per W.23(a)."
                ),
                "severity": "CRITICAL",
                "check_field": "flow_measurement_method",
            },
            {
                "id": "EPAW-FL-003",
                "name": "Gas Composition Analysis (Annual Minimum)",
                "description": (
                    "Gas composition must be analyzed at least annually "
                    "using ASTM D1945 or equivalent per W.23(b)."
                ),
                "severity": "CRITICAL",
                "check_field": "gas_composition_analyzed",
            },
            {
                "id": "EPAW-FL-004",
                "name": "Combustion Efficiency Documentation",
                "description": (
                    "Combustion efficiency must be documented: 98% "
                    "default or measured per 40 CFR 63.11(b)."
                ),
                "severity": "ERROR",
                "check_field": "combustion_efficiency",
                "check_fn": "_check_combustion_efficiency",
            },
            {
                "id": "EPAW-FL-005",
                "name": "Continuous Pilot/Purge Gas Tracking",
                "description": (
                    "Continuous pilot flame and purge gas volumes "
                    "must be tracked and reported separately."
                ),
                "severity": "ERROR",
                "check_field": "pilot_purge_tracked",
            },
            {
                "id": "EPAW-FL-006",
                "name": "Event Categorization",
                "description": (
                    "Flaring events must be categorized as routine, "
                    "non-routine, or emergency per W.23(c)."
                ),
                "severity": "ERROR",
                "check_field": "event_categories_tracked",
            },
            {
                "id": "EPAW-FL-007",
                "name": "Data Retention (3 Years)",
                "description": (
                    "All flaring data, calculations, and supporting "
                    "documentation must be retained for at least 3 years."
                ),
                "severity": "ERROR",
                "check_field": "data_retention_years",
                "check_fn": "_check_data_retention_3yr",
            },
            {
                "id": "EPAW-FL-008",
                "name": "Missing Data Substitution",
                "description": (
                    "Missing data substitution procedures must follow "
                    "40 CFR 98.235 for flare gas flow and composition."
                ),
                "severity": "ERROR",
                "check_field": "missing_data_procedures",
            },
            {
                "id": "EPAW-FL-009",
                "name": "Monthly Reporting",
                "description": (
                    "Flaring emissions must be reported on a monthly "
                    "basis for annual aggregation."
                ),
                "severity": "ERROR",
                "check_field": "monthly_reporting",
            },
            {
                "id": "EPAW-FL-010",
                "name": "Annual Reporting via e-GGRT",
                "description": (
                    "Annual flaring emissions report must be submitted "
                    "via EPA e-GGRT by March 31."
                ),
                "severity": "ERROR",
                "check_field": "annual_report_submitted",
            },
            {
                "id": "EPAW-FL-011",
                "name": "Monitoring Plan",
                "description": (
                    "A written monitoring plan must be developed and "
                    "maintained per 40 CFR 98.3(g) covering all flare "
                    "systems."
                ),
                "severity": "ERROR",
                "check_field": "monitoring_plan",
            },
            {
                "id": "EPAW-FL-012",
                "name": "QA/QC Procedures",
                "description": (
                    "QA/QC procedures for flow meters and composition "
                    "analyzers must be documented per 40 CFR 98.3(i)."
                ),
                "severity": "ERROR",
                "check_field": "qaqc_procedures",
            },
            {
                "id": "EPAW-FL-013",
                "name": "Heating Value Calculation",
                "description": (
                    "Higher heating value must be calculated from "
                    "gas composition or measured per W.23(d)."
                ),
                "severity": "ERROR",
                "check_field": "heating_value_documented",
            },
            {
                "id": "EPAW-FL-014",
                "name": "Equipment Calibration",
                "description": (
                    "Flow meters and analytical equipment must be "
                    "calibrated per manufacturer specifications."
                ),
                "severity": "WARNING",
                "check_field": "calibration_current",
            },
            {
                "id": "EPAW-FL-015",
                "name": "Flare Design Documentation",
                "description": (
                    "Flare design parameters (tip diameter, height, "
                    "assist type) must be documented."
                ),
                "severity": "WARNING",
                "check_field": "flare_design_documented",
            },
        ],

        # ---------------------------------------------------------------
        # EU ETS MRR (10 requirements)
        # ---------------------------------------------------------------
        "EU_ETS_MRR": [
            {
                "id": "MRR-FL-001",
                "name": "Monitoring Plan Approval",
                "description": (
                    "A flaring monitoring plan must be approved by "
                    "the competent authority."
                ),
                "severity": "CRITICAL",
                "check_field": "monitoring_plan_approved",
            },
            {
                "id": "MRR-FL-002",
                "name": "Tier Approach Selection",
                "description": (
                    "Appropriate tier must be selected for flare "
                    "gas flow and composition monitoring."
                ),
                "severity": "ERROR",
                "check_field": "tier_approach",
            },
            {
                "id": "MRR-FL-003",
                "name": "Uncertainty Limits",
                "description": (
                    "Measurement uncertainty must not exceed tier-specific "
                    "limits (Tier 1: +/-7.5%, Tier 2: +/-5%, "
                    "Tier 3: +/-2.5%)."
                ),
                "severity": "ERROR",
                "check_field": "uncertainty_within_limits",
                "check_fn": "_check_mrr_uncertainty_limits",
            },
            {
                "id": "MRR-FL-004",
                "name": "Calibration Requirements",
                "description": (
                    "All measurement instruments must be calibrated "
                    "at intervals specified in the monitoring plan."
                ),
                "severity": "ERROR",
                "check_field": "calibration_current",
            },
            {
                "id": "MRR-FL-005",
                "name": "Data Gaps Handling",
                "description": (
                    "Procedures for handling data gaps must be defined "
                    "and applied consistently."
                ),
                "severity": "ERROR",
                "check_field": "data_gaps_procedures",
            },
            {
                "id": "MRR-FL-006",
                "name": "Verification Report",
                "description": (
                    "Annual emissions report must be verified by "
                    "an accredited verifier."
                ),
                "severity": "ERROR",
                "check_field": "verification_report",
            },
            {
                "id": "MRR-FL-007",
                "name": "Activity Data Records",
                "description": (
                    "Activity data (flow rates, volumes) must be "
                    "recorded and retained."
                ),
                "severity": "ERROR",
                "check_field": "activity_data_recorded",
            },
            {
                "id": "MRR-FL-008",
                "name": "Emission Factor Source",
                "description": (
                    "Emission factors must be derived from gas "
                    "composition analysis or approved defaults."
                ),
                "severity": "ERROR",
                "check_field": "emission_factor_source",
            },
            {
                "id": "MRR-FL-009",
                "name": "Oxidation Factor",
                "description": (
                    "Oxidation (combustion) efficiency must be "
                    "documented and justified."
                ),
                "severity": "WARNING",
                "check_field": "combustion_efficiency",
            },
            {
                "id": "MRR-FL-010",
                "name": "Improvement Reporting",
                "description": (
                    "Planned improvements to monitoring methodology "
                    "must be reported in the improvement plan."
                ),
                "severity": "INFO",
                "check_field": "improvement_plan",
            },
        ],

        # ---------------------------------------------------------------
        # EU Methane Regulation 2024/1787 (12 requirements)
        # ---------------------------------------------------------------
        "EU_METHANE_REGULATION": [
            {
                "id": "EUMR-FL-001",
                "name": "Article 14: Flaring Restrictions",
                "description": (
                    "Routine flaring is prohibited in oil production "
                    "facilities except under specific derogations "
                    "per Article 14(1)."
                ),
                "severity": "CRITICAL",
                "check_field": "routine_flaring_justified",
                "check_fn": "_check_routine_flaring_justification",
            },
            {
                "id": "EUMR-FL-002",
                "name": "Routine Flaring Prohibition Compliance",
                "description": (
                    "Routine flaring volume must be zero or have "
                    "an approved derogation with timeline for elimination."
                ),
                "severity": "CRITICAL",
                "check_field": "routine_volume_scf",
                "check_fn": "_check_routine_flaring_volume",
            },
            {
                "id": "EUMR-FL-003",
                "name": "Non-Routine Flaring Notification",
                "description": (
                    "Non-routine flaring events exceeding threshold "
                    "must be notified to the competent authority "
                    "within 48 hours."
                ),
                "severity": "ERROR",
                "check_field": "non_routine_notifications",
            },
            {
                "id": "EUMR-FL-004",
                "name": "LDAR Integration for Flare Gas Recovery",
                "description": (
                    "Facilities must assess flare gas recovery "
                    "opportunities and integrate with LDAR programs."
                ),
                "severity": "ERROR",
                "check_field": "flare_gas_recovery_assessed",
            },
            {
                "id": "EUMR-FL-005",
                "name": "Emission Measurement Requirements",
                "description": (
                    "Methane emissions from flaring (including "
                    "uncombusted CH4 slip) must be measured or "
                    "estimated using approved methods."
                ),
                "severity": "ERROR",
                "check_field": "ch4_emissions_quantified",
            },
            {
                "id": "EUMR-FL-006",
                "name": "Reporting to Competent Authority",
                "description": (
                    "Flaring volumes and emissions must be reported "
                    "to the competent national authority per Article 12."
                ),
                "severity": "ERROR",
                "check_field": "reported_to_authority",
            },
            {
                "id": "EUMR-FL-007",
                "name": "Flare Gas Recovery Assessment",
                "description": (
                    "An economic and technical assessment of flare "
                    "gas recovery must be performed per Article 14(4)."
                ),
                "severity": "ERROR",
                "check_field": "gas_recovery_assessment",
            },
            {
                "id": "EUMR-FL-008",
                "name": "Source-Level Reporting",
                "description": (
                    "Flaring emissions must be reported at the "
                    "individual flare stack level per Article 12(2)."
                ),
                "severity": "ERROR",
                "check_field": "source_level_reporting",
            },
            {
                "id": "EUMR-FL-009",
                "name": "Methane Intensity Calculation",
                "description": (
                    "Methane intensity from flaring must be calculated "
                    "and reported (CH4 per unit production)."
                ),
                "severity": "WARNING",
                "check_field": "methane_intensity_reported",
            },
            {
                "id": "EUMR-FL-010",
                "name": "Third-Party Verification",
                "description": (
                    "Flaring emission reports must be verified by "
                    "an independent verifier per Article 13."
                ),
                "severity": "ERROR",
                "check_field": "third_party_verified",
            },
            {
                "id": "EUMR-FL-011",
                "name": "Public Transparency",
                "description": (
                    "Key flaring emission data must be made publicly "
                    "available per Article 15."
                ),
                "severity": "WARNING",
                "check_field": "public_disclosure",
            },
            {
                "id": "EUMR-FL-012",
                "name": "Derogation Documentation",
                "description": (
                    "Any derogation from the routine flaring prohibition "
                    "must be fully documented with justification and "
                    "a phase-out plan."
                ),
                "severity": "WARNING",
                "check_field": "derogation_documented",
            },
        ],

        # ---------------------------------------------------------------
        # World Bank Zero Routine Flaring by 2030 (8 requirements)
        # ---------------------------------------------------------------
        "WORLD_BANK_ZRF": [
            {
                "id": "ZRF-FL-001",
                "name": "Routine Flaring Volume Reporting",
                "description": (
                    "Routine flaring volumes must be reported "
                    "accurately with clear distinction from non-routine."
                ),
                "severity": "ERROR",
                "check_field": "routine_volume_scf",
            },
            {
                "id": "ZRF-FL-002",
                "name": "Year-over-Year Reduction Tracking",
                "description": (
                    "Year-over-year change in routine flaring volume "
                    "must be tracked and reported."
                ),
                "severity": "ERROR",
                "check_field": "yoy_reduction_tracked",
            },
            {
                "id": "ZRF-FL-003",
                "name": "Flare Gas Recovery Assessment",
                "description": (
                    "Technical and economic feasibility of flare gas "
                    "recovery must be assessed for all routine flaring."
                ),
                "severity": "ERROR",
                "check_field": "gas_recovery_assessment",
            },
            {
                "id": "ZRF-FL-004",
                "name": "Alternative Utilization Analysis",
                "description": (
                    "Alternatives to flaring (gas-to-power, reinjection, "
                    "LPG extraction) must be evaluated."
                ),
                "severity": "ERROR",
                "check_field": "alternative_utilization_analyzed",
            },
            {
                "id": "ZRF-FL-005",
                "name": "2030 Zero Target Progress",
                "description": (
                    "Progress toward the 2030 zero routine flaring "
                    "target must be reported with a pathway plan."
                ),
                "severity": "ERROR",
                "check_field": "zrf_2030_progress",
            },
            {
                "id": "ZRF-FL-006",
                "name": "Non-Routine Flaring Justification",
                "description": (
                    "Non-routine flaring events must be justified "
                    "with root cause documentation."
                ),
                "severity": "WARNING",
                "check_field": "non_routine_justified",
            },
            {
                "id": "ZRF-FL-007",
                "name": "Emergency Flaring Protocols",
                "description": (
                    "Emergency flaring response protocols must be "
                    "documented and tested."
                ),
                "severity": "WARNING",
                "check_field": "emergency_protocols",
            },
            {
                "id": "ZRF-FL-008",
                "name": "Stakeholder Reporting",
                "description": (
                    "Flaring performance must be reported to "
                    "relevant stakeholders including the World Bank."
                ),
                "severity": "INFO",
                "check_field": "stakeholder_reporting",
            },
        ],

        # ---------------------------------------------------------------
        # OGMP 2.0 (10 requirements)
        # ---------------------------------------------------------------
        "OGMP_2_0": [
            {
                "id": "OGMP-FL-001",
                "name": "Reporting Level Assessment",
                "description": (
                    "Current OGMP 2.0 reporting level (1-5) for "
                    "flaring must be assessed and documented."
                ),
                "severity": "ERROR",
                "check_field": "ogmp_level",
                "check_fn": "_check_ogmp_level",
            },
            {
                "id": "OGMP-FL-002",
                "name": "Source-Level Quantification",
                "description": (
                    "Flaring emissions must be quantified at the "
                    "individual flare stack level."
                ),
                "severity": "ERROR",
                "check_field": "source_level_reporting",
            },
            {
                "id": "OGMP-FL-003",
                "name": "Measurement-Based Reconciliation",
                "description": (
                    "For Level 4+, site-level measurements must be "
                    "reconciled with source-level estimates."
                ),
                "severity": "ERROR",
                "check_field": "measurement_reconciliation",
                "check_fn": "_check_measurement_reconciliation",
            },
            {
                "id": "OGMP-FL-004",
                "name": "Improvement Pathway Documentation",
                "description": (
                    "A pathway to achieve higher reporting levels "
                    "must be documented with timelines."
                ),
                "severity": "ERROR",
                "check_field": "improvement_pathway",
            },
            {
                "id": "OGMP-FL-005",
                "name": "CH4 Emission Factor Specificity",
                "description": (
                    "CH4 emission factors must be at least site-specific "
                    "(Level 3) or measurement-based (Level 4/5)."
                ),
                "severity": "ERROR",
                "check_field": "ch4_ef_specificity",
            },
            {
                "id": "OGMP-FL-006",
                "name": "Combustion Efficiency Quantification",
                "description": (
                    "Combustion efficiency must be quantified per "
                    "flare system, not just assumed as 98%."
                ),
                "severity": "WARNING",
                "check_field": "ce_quantified_per_flare",
            },
            {
                "id": "OGMP-FL-007",
                "name": "Uncombusted Methane Tracking",
                "description": (
                    "Uncombusted CH4 slip from flaring must be "
                    "explicitly calculated and reported."
                ),
                "severity": "ERROR",
                "check_field": "ch4_slip_tracked",
            },
            {
                "id": "OGMP-FL-008",
                "name": "Annual Reporting",
                "description": (
                    "Annual flaring emission report must be submitted "
                    "to OGMP 2.0 framework."
                ),
                "severity": "ERROR",
                "check_field": "annual_report_submitted",
            },
            {
                "id": "OGMP-FL-009",
                "name": "Data Transparency",
                "description": (
                    "Methodology, emission factors, and activity data "
                    "must be transparently reported."
                ),
                "severity": "WARNING",
                "check_field": "data_transparent",
            },
            {
                "id": "OGMP-FL-010",
                "name": "Continuous Improvement",
                "description": (
                    "Commitment to continuous improvement in flaring "
                    "monitoring and reduction must be demonstrated."
                ),
                "severity": "INFO",
                "check_field": "continuous_improvement",
            },
        ],
    }


# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================


class ComplianceCheckerEngine:
    """Regulatory compliance checker for flaring emission calculations.

    Validates calculation data against 8 regulatory frameworks with
    83 total requirements. All checks are deterministic boolean evaluations
    with no LLM involvement.

    Thread-safe via reentrant lock.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(
        ...     calculation_data={"flare_type": "ELEVATED_STEAM_ASSISTED", ...},
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

        # Check history
        self._check_history: List[Dict[str, Any]] = []
        self._max_history = self._config.get("max_history", 10000)

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
            Dictionary with per-framework results and overall summary.
        """
        t0 = time.monotonic()

        target_frameworks = frameworks or list(SUPPORTED_FRAMEWORKS)

        # Validate framework names
        for fw in target_frameworks:
            if fw not in SUPPORTED_FRAMEWORKS:
                logger.warning("Unknown framework '%s'; skipping", fw)

        valid_frameworks = [
            fw for fw in target_frameworks if fw in SUPPORTED_FRAMEWORKS
        ]

        results: Dict[str, Any] = {}
        for fw in valid_frameworks:
            results[fw] = self.check_framework(fw, calculation_data)

        # Aggregate summary
        compliant_count = sum(
            1 for r in results.values()
            if r.get("status") == "COMPLIANT"
        )
        partial_count = sum(
            1 for r in results.values()
            if r.get("status") == "PARTIAL"
        )
        non_compliant_count = sum(
            1 for r in results.values()
            if r.get("status") == "NON_COMPLIANT"
        )

        with self._lock:
            self._total_checks += 1
            self._total_compliant += compliant_count
            self._total_non_compliant += non_compliant_count
            self._total_partial += partial_count

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        summary: Dict[str, Any] = {
            "frameworks_checked": len(valid_frameworks),
            "compliant": compliant_count,
            "partial": partial_count,
            "non_compliant": non_compliant_count,
            "results": results,
            "checked_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed_ms, 3),
        }
        summary["provenance_hash"] = _compute_hash(summary)

        # Store in history
        self._store_check_result(summary)

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
            framework: Framework identifier (e.g. "EPA_SUBPART_W").
            calculation_data: Calculation data dictionary.

        Returns:
            Dictionary with status, met/not-met counts, findings,
            and recommendations.
        """
        requirements = self._requirements.get(framework, [])
        if not requirements:
            return {
                "framework": framework,
                "status": "NOT_CHECKED",
                "requirements_checked": 0,
                "requirements_met": 0,
                "requirements_not_met": 0,
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

            status = finding.get("status", "NOT_MET")
            if status == "MET":
                met_count += 1
            elif status == "PARTIALLY_MET":
                partially_met_count += 1
            else:
                not_met_count += 1
                if finding.get("recommendation"):
                    recommendations.append(finding["recommendation"])

        # Determine overall framework status
        total = len(requirements)
        pass_rate = (met_count + 0.5 * partially_met_count) / total

        if pass_rate >= _COMPLIANT_THRESHOLD:
            overall_status = "COMPLIANT"
        elif pass_rate >= _PARTIAL_THRESHOLD:
            overall_status = "PARTIAL"
        else:
            overall_status = "NON_COMPLIANT"

        return {
            "framework": framework,
            "status": overall_status,
            "requirements_checked": total,
            "requirements_met": met_count,
            "requirements_partially_met": partially_met_count,
            "requirements_not_met": not_met_count,
            "pass_rate": round(pass_rate, 4),
            "findings": findings,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Public API: Get Requirements
    # ------------------------------------------------------------------

    def get_requirements(
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

    def get_all_requirements(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all requirements across all frameworks.

        Returns:
            Dictionary of framework -> list of requirements.
        """
        return {
            fw: list(reqs) for fw, reqs in self._requirements.items()
        }

    def get_supported_frameworks(self) -> List[str]:
        """Get the list of supported framework identifiers.

        Returns:
            List of framework name strings.
        """
        return list(SUPPORTED_FRAMEWORKS)

    # ------------------------------------------------------------------
    # Public API: Validate Calculation
    # ------------------------------------------------------------------

    def validate_calculation(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> Dict[str, Any]:
        """Validate that all required data fields are present.

        Checks field presence and returns completeness information
        without performing full compliance evaluation.

        Args:
            calculation_data: Calculation data to validate.
            framework: Framework to validate against.

        Returns:
            Dictionary with present/missing field lists and completeness %.
        """
        requirements = self._requirements.get(framework, [])
        required_fields: set = set()
        for req in requirements:
            check_field = req.get("check_field", "")
            if check_field:
                required_fields.add(check_field)

        present: List[str] = []
        missing: List[str] = []
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
    # Public API: Generate Compliance Report
    # ------------------------------------------------------------------

    def generate_compliance_report(
        self,
        compliance_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a structured compliance report from check results.

        Produces a prioritized report with executive summary,
        per-framework details, and actionable recommendations.

        Args:
            compliance_results: Output from check_compliance().

        Returns:
            Dictionary with structured report sections.
        """
        results = compliance_results.get("results", {})

        # Executive summary
        total_reqs = 0
        total_met = 0
        total_not_met = 0
        critical_findings: List[Dict[str, Any]] = []

        for framework, fw_result in results.items():
            if not isinstance(fw_result, dict):
                continue
            total_reqs += fw_result.get("requirements_checked", 0)
            total_met += fw_result.get("requirements_met", 0)
            total_not_met += fw_result.get("requirements_not_met", 0)

            for finding in fw_result.get("findings", []):
                if finding.get("severity") in ("CRITICAL", "ERROR"):
                    if finding.get("status") != "MET":
                        critical_findings.append({
                            "framework": framework,
                            "requirement_id": finding.get("requirement_id", ""),
                            "requirement_name": finding.get("requirement_name", ""),
                            "severity": finding.get("severity", ""),
                            "description": finding.get("description", ""),
                            "recommendation": finding.get("recommendation", ""),
                        })

        # Sort critical findings by severity
        critical_findings.sort(
            key=lambda x: _SEVERITY_PRIORITY.get(x.get("severity", "INFO"), 3)
        )

        # Recommendations (prioritized)
        recommendations = self._generate_prioritized_recommendations(results)

        overall_pct = (
            (total_met / total_reqs * 100) if total_reqs > 0 else 0.0
        )

        report = {
            "report_id": f"fl_cr_{uuid4().hex[:12]}",
            "generated_at": _utcnow().isoformat(),
            "executive_summary": {
                "total_frameworks": len(results),
                "total_requirements": total_reqs,
                "requirements_met": total_met,
                "requirements_not_met": total_not_met,
                "overall_compliance_pct": round(overall_pct, 1),
                "critical_finding_count": len(critical_findings),
            },
            "critical_findings": critical_findings,
            "framework_summaries": {
                fw: {
                    "status": r.get("status", "NOT_CHECKED"),
                    "pass_rate": r.get("pass_rate", 0.0),
                    "requirements_met": r.get("requirements_met", 0),
                    "requirements_checked": r.get("requirements_checked", 0),
                }
                for fw, r in results.items()
                if isinstance(r, dict)
            },
            "recommendations": recommendations,
        }
        report["provenance_hash"] = _compute_hash(report)

        return report

    # ------------------------------------------------------------------
    # Public API: Statistics
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
                "check_history_count": len(self._check_history),
            }

    def get_check_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent compliance check history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of recent check summaries.
        """
        with self._lock:
            return list(reversed(self._check_history[-limit:]))

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

        Supports both simple field-presence checks and custom check
        functions for complex validation logic.

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
        check_fn_name = requirement.get("check_fn")

        # Use custom check function if specified
        if check_fn_name and hasattr(self, check_fn_name):
            check_fn = getattr(self, check_fn_name)
            try:
                check_result = check_fn(calculation_data, requirement)
                status = check_result.get("status", "NOT_MET")
                recommendation = check_result.get("recommendation", "")
                details = check_result.get("details", "")
            except Exception as exc:
                logger.warning(
                    "Custom check %s failed: %s", check_fn_name, exc,
                )
                status = "NOT_MET"
                recommendation = (
                    f"Fix custom check error for {req_id}: {exc}"
                )
                details = str(exc)
        else:
            # Default: field presence check
            has_value = self._field_has_value(calculation_data, check_field)
            if has_value:
                status = "MET"
                recommendation = ""
                details = f"Field '{check_field}' is present and valid"
            else:
                status = "NOT_MET"
                recommendation = (
                    f"Provide '{check_field}' data to satisfy "
                    f"{req_id}: {req_name}"
                )
                details = f"Field '{check_field}' is missing or empty"

        return {
            "requirement_id": req_id,
            "requirement_name": req_name,
            "description": description,
            "severity": severity,
            "check_field": check_field,
            "status": status,
            "details": details,
            "recommendation": recommendation,
            "framework": framework,
        }

    # ------------------------------------------------------------------
    # Private: Custom Check Functions
    # ------------------------------------------------------------------

    def _check_scope1_classification(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check that emissions are classified as Scope 1.

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        scope = str(data.get("scope_classification", "")).upper().strip()
        if scope in ("SCOPE_1", "SCOPE1", "1", "DIRECT"):
            return {
                "status": "MET",
                "details": f"Scope classification: {scope}",
                "recommendation": "",
            }

        # If scope field missing but flare_type present, partially met
        if data.get("flare_type"):
            return {
                "status": "PARTIALLY_MET",
                "details": "Flare type documented but Scope 1 not explicit",
                "recommendation": (
                    "Explicitly classify flaring emissions as Scope 1 "
                    "direct emissions."
                ),
            }

        return {
            "status": "NOT_MET",
            "details": "No scope classification provided",
            "recommendation": (
                "Add scope_classification='SCOPE_1' to classify flaring "
                "emissions as direct Scope 1."
            ),
        }

    def _check_epa_methodology(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check EPA Subpart W methodology compliance.

        Flare stack emissions must use Equation W-19 (gas composition)
        or W-20 (default emission factor).

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        method = str(data.get("calculation_method", "")).upper().strip()
        valid_methods = {
            "GAS_COMPOSITION", "DEFAULT_EMISSION_FACTOR",
            "EQUATION_W_19", "EQUATION_W_20",
            "W_19", "W_20",
        }

        if method in valid_methods:
            return {
                "status": "MET",
                "details": f"EPA-approved methodology: {method}",
                "recommendation": "",
            }

        if method:
            return {
                "status": "PARTIALLY_MET",
                "details": f"Method '{method}' may need EPA mapping",
                "recommendation": (
                    "Map calculation method to EPA Subpart W Equation "
                    "W-19 (gas composition) or W-20 (default EF)."
                ),
            }

        return {
            "status": "NOT_MET",
            "details": "No calculation method specified",
            "recommendation": (
                "Specify calculation_method per EPA 40 CFR 98.233(n)."
            ),
        }

    def _check_combustion_efficiency(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check combustion efficiency documentation.

        Must be 98% default (EPA) or measured per 40 CFR 63.11(b).

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        ce = data.get("combustion_efficiency")
        if ce is None:
            return {
                "status": "NOT_MET",
                "details": "Combustion efficiency not provided",
                "recommendation": (
                    "Specify combustion_efficiency (0.98 default or "
                    "measured value per 40 CFR 63.11(b))."
                ),
            }

        try:
            ce_val = float(ce)
        except (ValueError, TypeError):
            return {
                "status": "NOT_MET",
                "details": f"Invalid combustion efficiency: {ce}",
                "recommendation": "Provide numeric combustion_efficiency [0-1].",
            }

        if ce_val < 0 or ce_val > 1:
            return {
                "status": "NOT_MET",
                "details": f"CE {ce_val} outside valid range [0, 1]",
                "recommendation": (
                    "Combustion efficiency must be between 0 and 1."
                ),
            }

        if ce_val == 0.98:
            return {
                "status": "MET",
                "details": "Using EPA default CE of 98%",
                "recommendation": "",
            }

        if 0.90 <= ce_val <= 1.0:
            # Measured CE is acceptable if documented
            ce_source = data.get("combustion_efficiency_source", "")
            if ce_source:
                return {
                    "status": "MET",
                    "details": (
                        f"Measured CE={ce_val} from {ce_source}"
                    ),
                    "recommendation": "",
                }
            return {
                "status": "PARTIALLY_MET",
                "details": f"CE={ce_val} but source not documented",
                "recommendation": (
                    "Document combustion_efficiency_source to justify "
                    "non-default CE value."
                ),
            }

        return {
            "status": "NOT_MET",
            "details": f"CE={ce_val} below acceptable range",
            "recommendation": (
                "Review combustion efficiency; values below 0.90 require "
                "engineering justification."
            ),
        }

    def _check_data_retention_3yr(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check that data retention meets 3-year minimum.

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        retention = data.get("data_retention_years")
        if retention is None:
            return {
                "status": "NOT_MET",
                "details": "Data retention policy not specified",
                "recommendation": (
                    "Define data_retention_years >= 3 per EPA 40 CFR 98."
                ),
            }

        try:
            years = int(retention)
        except (ValueError, TypeError):
            return {
                "status": "NOT_MET",
                "details": f"Invalid retention value: {retention}",
                "recommendation": "Provide integer data_retention_years >= 3.",
            }

        if years >= 3:
            return {
                "status": "MET",
                "details": f"Data retention: {years} years (>= 3 required)",
                "recommendation": "",
            }

        return {
            "status": "NOT_MET",
            "details": f"Data retention {years} years < 3 year minimum",
            "recommendation": (
                "Increase data_retention_years to at least 3 per EPA regs."
            ),
        }

    def _check_mrr_uncertainty_limits(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check EU ETS MRR tier-specific uncertainty limits.

        Tier 1: +/-7.5%, Tier 2: +/-5%, Tier 3: +/-2.5%

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        tier = str(data.get("tier_approach", "")).upper().strip()
        unc = data.get("measurement_uncertainty_pct")

        if not tier:
            return {
                "status": "NOT_MET",
                "details": "Tier approach not specified",
                "recommendation": "Specify tier_approach for MRR compliance.",
            }

        tier_limits = {
            "TIER_1": 7.5,
            "TIER1": 7.5,
            "1": 7.5,
            "TIER_2": 5.0,
            "TIER2": 5.0,
            "2": 5.0,
            "TIER_3": 2.5,
            "TIER3": 2.5,
            "3": 2.5,
        }
        limit = tier_limits.get(tier)
        if limit is None:
            return {
                "status": "PARTIALLY_MET",
                "details": f"Unrecognized tier: {tier}",
                "recommendation": (
                    "Use TIER_1, TIER_2, or TIER_3 for tier_approach."
                ),
            }

        if unc is None:
            return {
                "status": "NOT_MET",
                "details": f"Uncertainty not reported for {tier}",
                "recommendation": (
                    f"Provide measurement_uncertainty_pct (limit: {limit}% "
                    f"for {tier})."
                ),
            }

        try:
            unc_val = float(unc)
        except (ValueError, TypeError):
            return {
                "status": "NOT_MET",
                "details": f"Invalid uncertainty: {unc}",
                "recommendation": "Provide numeric measurement_uncertainty_pct.",
            }

        if unc_val <= limit:
            return {
                "status": "MET",
                "details": (
                    f"Uncertainty {unc_val}% within {tier} limit of {limit}%"
                ),
                "recommendation": "",
            }

        return {
            "status": "NOT_MET",
            "details": (
                f"Uncertainty {unc_val}% exceeds {tier} limit of {limit}%"
            ),
            "recommendation": (
                f"Reduce measurement uncertainty to <= {limit}% for {tier} "
                f"or apply for lower tier."
            ),
        }

    def _check_routine_flaring_justification(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check EU Methane Regulation Article 14 routine flaring justification.

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        routine_vol = data.get("routine_volume_scf")
        justified = data.get("routine_flaring_justified")
        derogation = data.get("derogation_documented")

        if routine_vol is not None:
            try:
                vol = float(routine_vol)
            except (ValueError, TypeError):
                vol = -1
        else:
            vol = 0

        if vol == 0:
            return {
                "status": "MET",
                "details": "No routine flaring volume reported",
                "recommendation": "",
            }

        if vol > 0 and justified:
            if derogation:
                return {
                    "status": "PARTIALLY_MET",
                    "details": (
                        f"Routine flaring ({vol} scf) justified with derogation"
                    ),
                    "recommendation": (
                        "Develop timeline for eliminating routine flaring "
                        "per Article 14."
                    ),
                }
            return {
                "status": "PARTIALLY_MET",
                "details": (
                    f"Routine flaring ({vol} scf) marked as justified but "
                    "derogation not documented"
                ),
                "recommendation": (
                    "Document derogation per Article 14(1) for any "
                    "routine flaring."
                ),
            }

        if vol > 0:
            return {
                "status": "NOT_MET",
                "details": (
                    f"Routine flaring ({vol} scf) without justification"
                ),
                "recommendation": (
                    "Routine flaring is prohibited under EU Methane Regulation "
                    "Article 14. Provide justification or eliminate."
                ),
            }

        return {
            "status": "NOT_MET",
            "details": "Routine flaring status unclear",
            "recommendation": (
                "Specify routine_volume_scf and routine_flaring_justified."
            ),
        }

    def _check_routine_flaring_volume(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check that routine flaring volume is zero or has derogation.

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        routine_vol = data.get("routine_volume_scf")

        if routine_vol is None:
            return {
                "status": "NOT_MET",
                "details": "routine_volume_scf not reported",
                "recommendation": (
                    "Report routine_volume_scf (0 for zero routine flaring)."
                ),
            }

        try:
            vol = float(routine_vol)
        except (ValueError, TypeError):
            return {
                "status": "NOT_MET",
                "details": f"Invalid routine volume: {routine_vol}",
                "recommendation": "Provide numeric routine_volume_scf.",
            }

        if vol == 0:
            return {
                "status": "MET",
                "details": "Zero routine flaring volume",
                "recommendation": "",
            }

        derogation = data.get("derogation_documented")
        if derogation:
            return {
                "status": "PARTIALLY_MET",
                "details": (
                    f"Routine flaring volume {vol} scf with derogation"
                ),
                "recommendation": (
                    "Continue working toward zero routine flaring per "
                    "derogation timeline."
                ),
            }

        return {
            "status": "NOT_MET",
            "details": f"Routine flaring volume {vol} scf without derogation",
            "recommendation": (
                "Eliminate routine flaring or obtain approved derogation."
            ),
        }

    def _check_ogmp_level(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check OGMP 2.0 reporting level assessment.

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        level = data.get("ogmp_level")
        if level is None:
            return {
                "status": "NOT_MET",
                "details": "OGMP 2.0 reporting level not assessed",
                "recommendation": (
                    "Assess and document ogmp_level (1-5) for flaring "
                    "emission reporting."
                ),
            }

        try:
            level_int = int(level)
        except (ValueError, TypeError):
            return {
                "status": "NOT_MET",
                "details": f"Invalid OGMP level: {level}",
                "recommendation": "Provide integer ogmp_level (1-5).",
            }

        if level_int < 1 or level_int > 5:
            return {
                "status": "NOT_MET",
                "details": f"OGMP level {level_int} outside range [1, 5]",
                "recommendation": "OGMP level must be between 1 and 5.",
            }

        if level_int >= 3:
            return {
                "status": "MET",
                "details": f"OGMP 2.0 Level {level_int} (site-specific or better)",
                "recommendation": "",
            }

        return {
            "status": "PARTIALLY_MET",
            "details": (
                f"OGMP 2.0 Level {level_int} (generic factors; target Level 3+)"
            ),
            "recommendation": (
                f"Upgrade from Level {level_int} to Level 3+ for "
                "site-specific quantification."
            ),
        }

    def _check_measurement_reconciliation(
        self,
        data: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check OGMP 2.0 measurement-based reconciliation (Level 4+).

        Args:
            data: Calculation data.
            requirement: Requirement definition.

        Returns:
            Check result dictionary.
        """
        level = data.get("ogmp_level")
        reconciled = data.get("measurement_reconciliation")

        if level is not None:
            try:
                level_int = int(level)
            except (ValueError, TypeError):
                level_int = 0
        else:
            level_int = 0

        # Only required for Level 4+
        if level_int < 4:
            if reconciled:
                return {
                    "status": "MET",
                    "details": (
                        f"Reconciliation available (Level {level_int}; "
                        "exceeds requirement)"
                    ),
                    "recommendation": "",
                }
            return {
                "status": "MET",
                "details": (
                    f"Reconciliation not required at Level {level_int}"
                ),
                "recommendation": "",
            }

        if reconciled:
            return {
                "status": "MET",
                "details": (
                    f"Measurement reconciliation documented at Level {level_int}"
                ),
                "recommendation": "",
            }

        return {
            "status": "NOT_MET",
            "details": (
                f"Level {level_int} requires measurement reconciliation"
            ),
            "recommendation": (
                "Perform site-level measurement reconciliation with "
                "source-level estimates per OGMP 2.0 Level 4+ requirements."
            ),
        }

    # ------------------------------------------------------------------
    # Private: Field Value Check
    # ------------------------------------------------------------------

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
        - If numeric, it is present (zero is valid)
        - If list/dict, it is not empty
        - If bool, it is True

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

    # ------------------------------------------------------------------
    # Private: Recommendations
    # ------------------------------------------------------------------

    def _generate_prioritized_recommendations(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations from framework results.

        Args:
            results: Per-framework results dictionary.

        Returns:
            List of recommendation dictionaries sorted by priority.
        """
        recommendations: List[Dict[str, Any]] = []

        for framework, fw_result in results.items():
            if not isinstance(fw_result, dict):
                continue
            for finding in fw_result.get("findings", []):
                if finding.get("status") != "MET":
                    severity = finding.get("severity", "INFO")
                    rec = {
                        "framework": framework,
                        "requirement_id": finding.get("requirement_id", ""),
                        "requirement_name": finding.get("requirement_name", ""),
                        "severity": severity,
                        "priority": _SEVERITY_PRIORITY.get(severity, 3),
                        "recommendation": finding.get(
                            "recommendation",
                            f"Address {finding.get('requirement_name', '')}",
                        ),
                        "current_status": finding.get("status", "NOT_MET"),
                    }
                    recommendations.append(rec)

        # Sort by priority (0=highest/CRITICAL) then by framework
        recommendations.sort(
            key=lambda x: (x["priority"], x["framework"]),
        )

        return recommendations

    # ------------------------------------------------------------------
    # Private: History Storage
    # ------------------------------------------------------------------

    def _store_check_result(self, result: Dict[str, Any]) -> None:
        """Store a compliance check result in history.

        Enforces maximum history size by evicting oldest entries.

        Args:
            result: Compliance check summary to store.
        """
        with self._lock:
            self._check_history.append({
                "checked_at": result.get("checked_at", ""),
                "frameworks_checked": result.get("frameworks_checked", 0),
                "compliant": result.get("compliant", 0),
                "partial": result.get("partial", 0),
                "non_compliant": result.get("non_compliant", 0),
                "provenance_hash": result.get("provenance_hash", ""),
            })
            while len(self._check_history) > self._max_history:
                self._check_history.pop(0)
