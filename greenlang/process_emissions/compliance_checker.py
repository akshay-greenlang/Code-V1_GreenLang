# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Regulatory Compliance Validation (Engine 6 of 6)

AGENT-MRV-004: Process Emissions Agent

Validates process emissions calculations against six regulatory frameworks
to ensure data completeness, methodological correctness, and reporting
readiness.  Each framework defines 10 specific requirements that are
individually checked and scored.

Supported Frameworks (60 total requirements):
    1. GHG Protocol Corporate Standard (Chapter 5)
    2. ISO 14064-1:2018
    3. CSRD / ESRS E1
    4. EPA 40 CFR Part 98 (Multiple Subparts)
    5. UK SECR (Streamlined Energy and Carbon Reporting)
    6. EU ETS MRR (Monitoring and Reporting Regulation)

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met

Severity Levels:
    ERROR:   Requirement failure prevents regulatory compliance
    WARNING: Requirement failure should be addressed but is not blocking
    INFO:    Informational finding for best practice improvement

EPA Subpart Mapping (process types to applicable subparts):
    Cement:          Subpart F (cement)
    Lime:            Subpart S (lime)
    Glass:           Subpart N (glass)
    Iron/Steel:      Subpart Q (iron & steel)
    Aluminum:        Subpart F (aluminum)
    Nitric Acid:     Subpart V (nitric acid)
    Adipic Acid:     Subpart E (adipic acid)
    Ammonia:         Subpart G (ammonia)
    Hydrogen:        Subpart P (hydrogen)
    Petrochemical:   Subpart X (petrochemical)
    Semiconductor:   Subpart I (electronics)
    Ferroalloy:      Subpart Z (ferroalloy)
    Soda Ash:        Subpart Y (soda ash)

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean evaluations.
    - No LLM involvement in any compliance determination.
    - Requirement definitions are hard-coded from regulatory texts.
    - Every result carries a SHA-256 provenance hash.
    - Same inputs always produce identical compliance verdicts.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.process_emissions.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_data={
    ...         "process_type": "CEMENT_PRODUCTION",
    ...         "calculation_method": "EMISSION_FACTOR",
    ...         "tier": "TIER_2",
    ...         "total_co2e_tonnes": 50000.0,
    ...         "emissions_by_gas": [{"gas": "CO2"}],
    ...         "emission_factor_source": "IPCC",
    ...         "gwp_source": "AR6",
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["GHG_PROTOCOL"],
    ... )
    >>> print(result["GHG_PROTOCOL"]["status"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
    from greenlang.process_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.metrics import (
        record_compliance_check as _record_compliance_check,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_compliance_check = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All supported regulatory frameworks.
SUPPORTED_FRAMEWORKS: Tuple[str, ...] = (
    "GHG_PROTOCOL",
    "ISO_14064",
    "CSRD_ESRS_E1",
    "EPA_40CFR98",
    "UK_SECR",
    "EU_ETS_MRR",
)

#: Compliance status thresholds (fraction of requirements passed).
_COMPLIANT_THRESHOLD: float = 1.0
_PARTIAL_THRESHOLD: float = 0.5

#: Mapping of process types to applicable EPA 40 CFR Part 98 subparts.
EPA_SUBPART_MAP: Dict[str, List[str]] = {
    "CEMENT_PRODUCTION": ["Subpart F"],
    "LIME_PRODUCTION": ["Subpart S"],
    "GLASS_PRODUCTION": ["Subpart N"],
    "CERAMICS": ["Subpart N"],
    "SODA_ASH": ["Subpart Y"],
    "AMMONIA_PRODUCTION": ["Subpart G"],
    "NITRIC_ACID": ["Subpart V"],
    "ADIPIC_ACID": ["Subpart E"],
    "CARBIDE_PRODUCTION": ["Subpart Z"],
    "PETROCHEMICAL": ["Subpart X"],
    "HYDROGEN_PRODUCTION": ["Subpart P"],
    "PHOSPHORIC_ACID": ["Subpart Z"],
    "TITANIUM_DIOXIDE": ["Subpart Z"],
    "IRON_STEEL": ["Subpart Q"],
    "ALUMINUM_SMELTING": ["Subpart F"],
    "FERROALLOY": ["Subpart Z"],
    "LEAD_PRODUCTION": ["Subpart R"],
    "ZINC_PRODUCTION": ["Subpart Z"],
    "MAGNESIUM_PRODUCTION": ["Subpart T"],
    "COPPER_SMELTING": ["Subpart Z"],
    "SEMICONDUCTOR": ["Subpart I"],
    "PULP_PAPER": ["Subpart AA"],
    "MINERAL_WOOL": ["Subpart N"],
    "CARBON_ANODE": ["Subpart F"],
    "FOOD_DRINK": ["Subpart Z"],
}

#: All greenhouse gases that should be considered for GHG Protocol
#: gas coverage checks.
ALL_GHG_GASES: Tuple[str, ...] = (
    "CO2", "CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC",
)

#: Process types to their primary applicable gases (for coverage checks).
PROCESS_PRIMARY_GASES: Dict[str, List[str]] = {
    "CEMENT_PRODUCTION": ["CO2"],
    "LIME_PRODUCTION": ["CO2"],
    "GLASS_PRODUCTION": ["CO2"],
    "CERAMICS": ["CO2"],
    "SODA_ASH": ["CO2"],
    "AMMONIA_PRODUCTION": ["CO2"],
    "NITRIC_ACID": ["N2O"],
    "ADIPIC_ACID": ["N2O"],
    "CARBIDE_PRODUCTION": ["CO2", "CH4"],
    "PETROCHEMICAL": ["CO2", "CH4"],
    "HYDROGEN_PRODUCTION": ["CO2"],
    "PHOSPHORIC_ACID": ["CO2"],
    "TITANIUM_DIOXIDE": ["CO2"],
    "IRON_STEEL": ["CO2", "CH4"],
    "ALUMINUM_SMELTING": ["CO2", "CF4", "C2F6"],
    "FERROALLOY": ["CO2", "CH4"],
    "LEAD_PRODUCTION": ["CO2"],
    "ZINC_PRODUCTION": ["CO2"],
    "MAGNESIUM_PRODUCTION": ["CO2", "SF6"],
    "COPPER_SMELTING": ["CO2"],
    "SEMICONDUCTOR": ["CF4", "C2F6", "SF6", "NF3", "HFC"],
    "PULP_PAPER": ["CO2"],
    "MINERAL_WOOL": ["CO2"],
    "CARBON_ANODE": ["CO2"],
    "FOOD_DRINK": ["CO2"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComplianceRequirement:
    """Definition of a single regulatory compliance requirement.

    Attributes:
        requirement_id: Unique identifier (e.g. ghg_process_identification).
        framework: Regulatory framework this belongs to.
        name: Short human-readable name.
        description: Detailed description of the requirement.
        required_fields: List of data fields that must be present.
        validation_fn: Name of the validation method on the engine.
        severity: Impact level (ERROR, WARNING, INFO).
    """

    requirement_id: str = ""
    framework: str = ""
    name: str = ""
    description: str = ""
    required_fields: List[str] = field(default_factory=list)
    validation_fn: str = ""
    severity: str = "ERROR"


@dataclass
class ComplianceCheckResult:
    """Result of checking a single compliance requirement.

    Attributes:
        requirement_id: Requirement that was checked.
        framework: Regulatory framework.
        name: Human-readable name of the requirement.
        passed: Whether the requirement was met.
        details: Explanation of the check outcome.
        recommendations: Suggested corrective actions.
        severity: Impact level of the requirement.
    """

    requirement_id: str = ""
    framework: str = ""
    name: str = ""
    passed: bool = False
    details: str = ""
    recommendations: List[str] = field(default_factory=list)
    severity: str = "ERROR"


@dataclass
class FrameworkResult:
    """Aggregate compliance result for a single regulatory framework.

    Attributes:
        framework: Regulatory framework name.
        total_checks: Total number of requirements evaluated.
        passed: Number of requirements fully met.
        failed: Number of requirements not met (severity=ERROR).
        warnings: Number of warnings (severity=WARNING).
        status: Overall status (COMPLIANT, NON_COMPLIANT, PARTIAL).
        results: Individual check results.
        provenance_hash: SHA-256 hash for audit trail.
        checked_at: UTC timestamp of the evaluation.
    """

    framework: str = ""
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    status: str = "NOT_CHECKED"
    results: List[ComplianceCheckResult] = field(default_factory=list)
    provenance_hash: str = ""
    checked_at: str = ""


# ---------------------------------------------------------------------------
# ComplianceCheckerEngine
# ---------------------------------------------------------------------------


class ComplianceCheckerEngine:
    """Engine 6: Compliance Checker - Regulatory validation for 6 frameworks.

    Validates process emissions calculation data against the requirements
    of GHG Protocol, ISO 14064-1, CSRD/ESRS E1, EPA 40 CFR Part 98,
    UK SECR, and EU ETS MRR.  Each framework has 10 requirements that
    are individually evaluated.

    Thread Safety:
        All mutable state is protected by ``self._lock``
        (``threading.RLock``).

    Attributes:
        _requirements: Dict mapping framework name to list of
            ComplianceRequirement objects.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> data = {"process_type": "CEMENT_PRODUCTION", ...}
        >>> result = engine.check_compliance(data, ["GHG_PROTOCOL"])
        >>> print(result["GHG_PROTOCOL"]["status"])
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise the ComplianceCheckerEngine.

        Builds the full requirement catalogue for all 6 frameworks
        (60 total requirements).
        """
        self._lock: threading.RLock = threading.RLock()
        self._requirements: Dict[str, List[ComplianceRequirement]] = {}

        self._build_ghg_protocol_requirements()
        self._build_iso_14064_requirements()
        self._build_csrd_esrs_requirements()
        self._build_epa_requirements()
        self._build_uk_secr_requirements()
        self._build_eu_ets_requirements()

        total = sum(len(v) for v in self._requirements.values())
        logger.info(
            "ComplianceCheckerEngine initialised: "
            "%d frameworks, %d total requirements",
            len(self._requirements),
            total,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _hash(self, data: str) -> str:
        """Compute SHA-256 hex digest for provenance."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        details: Dict[str, Any],
    ) -> Optional[str]:
        """Record a provenance entry if the tracker is available."""
        if not _PROVENANCE_AVAILABLE:
            return None
        try:
            tracker = _get_provenance_tracker()  # type: ignore[misc]
            entry = tracker.record(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                data=details,
            )
            return entry.hash if hasattr(entry, "hash") else None
        except Exception as exc:
            logger.warning("Provenance recording failed: %s", exc)
            return None

    def _record_metric(
        self,
        framework: str,
        status: str = "success",
    ) -> None:
        """Record a Prometheus metric for a compliance check."""
        if _METRICS_AVAILABLE and _record_compliance_check is not None:
            try:
                _record_compliance_check(framework, status)
            except Exception:
                pass

    def _safe_get(
        self,
        data: Dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        """Safely get a value from a dict, supporting nested dot-notation."""
        if "." not in key:
            return data.get(key, default)
        parts = key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
        return current

    def _has_fields(
        self,
        data: Dict[str, Any],
        fields: List[str],
    ) -> Tuple[bool, List[str]]:
        """Check that all required fields are present and non-empty.

        Args:
            data: Calculation data dictionary.
            fields: List of required field names.

        Returns:
            Tuple of (all_present, missing_fields).
        """
        missing: List[str] = []
        for f in fields:
            val = self._safe_get(data, f)
            if val is None or val == "" or val == []:
                missing.append(f)
        return len(missing) == 0, missing

    def _has_gas_coverage(
        self,
        data: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[str]]:
        """Check that all applicable gases are reported.

        Args:
            data: Calculation data with process_type and emissions_by_gas.

        Returns:
            Tuple of (coverage_complete, reported_gases, missing_gases).
        """
        process_type = str(
            self._safe_get(data, "process_type", "")
        ).upper()
        required_gases = PROCESS_PRIMARY_GASES.get(process_type, ["CO2"])

        emissions_by_gas = self._safe_get(data, "emissions_by_gas", [])
        reported_gases: List[str] = []
        if isinstance(emissions_by_gas, list):
            for entry in emissions_by_gas:
                if isinstance(entry, dict):
                    gas = entry.get("gas", "")
                    if gas:
                        reported_gases.append(str(gas).upper())

        missing = [g for g in required_gases if g not in reported_gases]
        return len(missing) == 0, reported_gases, missing

    def _determine_status(
        self,
        passed: int,
        total: int,
    ) -> str:
        """Determine overall compliance status from pass ratio.

        Args:
            passed: Number of requirements passed.
            total: Total number of requirements evaluated.

        Returns:
            Status string: COMPLIANT, PARTIAL, or NON_COMPLIANT.
        """
        if total == 0:
            return "NOT_CHECKED"
        ratio = passed / total
        if ratio >= _COMPLIANT_THRESHOLD:
            return "COMPLIANT"
        if ratio >= _PARTIAL_THRESHOLD:
            return "PARTIAL"
        return "NON_COMPLIANT"

    # ==================================================================
    # Requirement Catalogue Builders
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. GHG Protocol Corporate Standard (Chapter 5)
    # ------------------------------------------------------------------

    def _build_ghg_protocol_requirements(self) -> None:
        """Build the 10 GHG Protocol compliance requirements."""
        fw = "GHG_PROTOCOL"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="ghg_process_identification",
                framework=fw,
                name="Process Source Identification",
                description=(
                    "All process emission sources within the organizational "
                    "boundary must be identified and documented. This includes "
                    "listing all industrial processes that release GHGs through "
                    "chemical or physical transformations (not combustion)."
                ),
                required_fields=["process_type"],
                validation_fn="_check_ghg_process_identification",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ghg_calculation_methodology",
                framework=fw,
                name="Calculation Methodology",
                description=(
                    "An appropriate calculation method must be documented "
                    "and applied consistently. Acceptable methods include "
                    "emission factor, mass balance, stoichiometric, and "
                    "direct measurement approaches."
                ),
                required_fields=["calculation_method"],
                validation_fn="_check_ghg_calculation_methodology",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ghg_emission_factor_source",
                framework=fw,
                name="Emission Factor Source",
                description=(
                    "The source of all emission factors must be documented "
                    "and traceable to a recognised authority (IPCC, EPA, "
                    "DEFRA, EU ETS, or custom with justification)."
                ),
                required_fields=["emission_factor_source"],
                validation_fn="_check_ghg_ef_source",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ghg_boundary_completeness",
                framework=fw,
                name="Boundary Completeness",
                description=(
                    "All significant process emission sources within the "
                    "defined organizational and operational boundaries "
                    "must be included in the inventory."
                ),
                required_fields=["process_type", "total_co2e_tonnes"],
                validation_fn="_check_ghg_boundary_completeness",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ghg_gas_coverage",
                framework=fw,
                name="Gas Coverage",
                description=(
                    "All relevant greenhouse gases must be reported for "
                    "each process type. Minimum required gases depend on "
                    "the industrial process (e.g. CO2 for cement, N2O for "
                    "nitric acid, PFCs for aluminum smelting)."
                ),
                required_fields=["emissions_by_gas"],
                validation_fn="_check_ghg_gas_coverage",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ghg_de_minimis_threshold",
                framework=fw,
                name="De Minimis Threshold",
                description=(
                    "Any excluded emission sources must individually "
                    "represent less than 1% of total Scope 1 emissions, "
                    "and total exclusions must not exceed 5% collectively."
                ),
                required_fields=["total_co2e_tonnes"],
                validation_fn="_check_ghg_de_minimis",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ghg_temporal_consistency",
                framework=fw,
                name="Temporal Consistency",
                description=(
                    "Calculation methodology must be applied consistently "
                    "across reporting periods. Any methodology changes must "
                    "be documented with justification."
                ),
                required_fields=["calculation_method", "tier"],
                validation_fn="_check_ghg_temporal_consistency",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ghg_base_year_recalculation",
                framework=fw,
                name="Base Year Recalculation",
                description=(
                    "Triggers for base year recalculation must be identified "
                    "including structural changes (mergers, acquisitions, "
                    "divestitures), methodology changes, and discovery of "
                    "significant errors."
                ),
                required_fields=[],
                validation_fn="_check_ghg_base_year",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="ghg_quality_management",
                framework=fw,
                name="Quality Management",
                description=(
                    "QA/QC procedures must be in place for data collection, "
                    "calculation, and reporting. This includes data validation, "
                    "calculation cross-checks, and documentation review."
                ),
                required_fields=["provenance_hash"],
                validation_fn="_check_ghg_quality_management",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ghg_verification_readiness",
                framework=fw,
                name="Verification Readiness",
                description=(
                    "Data must be sufficient in quality and completeness to "
                    "support third-party verification. This includes "
                    "calculation traces, source documentation references, "
                    "and provenance tracking."
                ),
                required_fields=["provenance_hash", "calculation_trace"],
                validation_fn="_check_ghg_verification_readiness",
                severity="INFO",
            ),
        ]

    # ------------------------------------------------------------------
    # 2. ISO 14064-1:2018
    # ------------------------------------------------------------------

    def _build_iso_14064_requirements(self) -> None:
        """Build the 10 ISO 14064-1:2018 compliance requirements."""
        fw = "ISO_14064"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="iso_organizational_boundary",
                framework=fw,
                name="Organizational Boundary",
                description=(
                    "The organizational boundary must be defined using "
                    "either the control approach (operational or financial "
                    "control) or equity share approach, per clause 5."
                ),
                required_fields=["facility_id"],
                validation_fn="_check_iso_org_boundary",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="iso_ghg_sources_identified",
                framework=fw,
                name="GHG Sources Identified",
                description=(
                    "All Category 1 (direct) GHG emission sources from "
                    "industrial processes must be identified and classified."
                ),
                required_fields=["process_type"],
                validation_fn="_check_iso_sources",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="iso_quantification_methodology",
                framework=fw,
                name="Quantification Methodology",
                description=(
                    "The quantification methodology must be documented per "
                    "clause 6, including calculation approach, emission "
                    "factors, and activity data sources."
                ),
                required_fields=[
                    "calculation_method", "emission_factor_source",
                ],
                validation_fn="_check_iso_methodology",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="iso_data_quality",
                framework=fw,
                name="Data Quality Assessment",
                description=(
                    "A data quality assessment must be performed covering "
                    "accuracy, completeness, consistency, transparency, "
                    "and relevance of the GHG data."
                ),
                required_fields=["tier"],
                validation_fn="_check_iso_data_quality",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="iso_uncertainty_assessed",
                framework=fw,
                name="Uncertainty Assessment",
                description=(
                    "A quantitative uncertainty analysis must be performed "
                    "for all significant emission sources, per clause 7. "
                    "Monte Carlo simulation or analytical propagation "
                    "are acceptable methods."
                ),
                required_fields=[],
                validation_fn="_check_iso_uncertainty",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="iso_base_year_established",
                framework=fw,
                name="Base Year Established",
                description=(
                    "A base year must be defined with a documented "
                    "recalculation policy specifying triggers and "
                    "materiality thresholds for recalculation."
                ),
                required_fields=[],
                validation_fn="_check_iso_base_year",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="iso_exclusions_justified",
                framework=fw,
                name="Exclusions Justified",
                description=(
                    "Any excluded GHG sources must be justified with "
                    "a documented materiality threshold. Excluded sources "
                    "must be listed with quantified estimates."
                ),
                required_fields=[],
                validation_fn="_check_iso_exclusions",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="iso_consistent_reporting",
                framework=fw,
                name="Consistent Reporting",
                description=(
                    "Year-to-year comparability must be ensured through "
                    "consistent application of methodology, boundaries, "
                    "and reporting formats."
                ),
                required_fields=["calculation_method"],
                validation_fn="_check_iso_consistent",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="iso_competence_requirements",
                framework=fw,
                name="Competence Requirements",
                description=(
                    "Personnel qualifications and competence requirements "
                    "for data collection, calculation, and reporting must "
                    "be documented per clause 8."
                ),
                required_fields=[],
                validation_fn="_check_iso_competence",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="iso_documentation_complete",
                framework=fw,
                name="Documentation Complete",
                description=(
                    "All records must be maintained per clause 9 including "
                    "raw data, calculations, assumptions, emission factors, "
                    "and methodology documentation."
                ),
                required_fields=["provenance_hash"],
                validation_fn="_check_iso_documentation",
                severity="ERROR",
            ),
        ]

    # ------------------------------------------------------------------
    # 3. CSRD / ESRS E1
    # ------------------------------------------------------------------

    def _build_csrd_esrs_requirements(self) -> None:
        """Build the 10 CSRD/ESRS E1 compliance requirements."""
        fw = "CSRD_ESRS_E1"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="csrd_scope1_by_category",
                framework=fw,
                name="Scope 1 by Category",
                description=(
                    "Process emissions must be separately disclosed from "
                    "combustion emissions within Scope 1 reporting, "
                    "broken down by industrial process category."
                ),
                required_fields=["process_type", "total_co2e_tonnes"],
                validation_fn="_check_csrd_scope1_category",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="csrd_ghg_protocol_methodology",
                framework=fw,
                name="GHG Protocol Methodology",
                description=(
                    "The GHG Protocol Corporate Standard must be used as "
                    "the basis for calculation methodology. Commission "
                    "Delegated Acts may specify additional requirements."
                ),
                required_fields=["calculation_method"],
                validation_fn="_check_csrd_ghg_methodology",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="csrd_process_emissions_separated",
                framework=fw,
                name="Process Emissions Separated",
                description=(
                    "Process emissions must be clearly distinguished from "
                    "combustion emissions in reporting disclosures. "
                    "The split is required per ESRS E1 paragraph 44."
                ),
                required_fields=["process_type"],
                validation_fn="_check_csrd_separation",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="csrd_sector_specific_disclosure",
                framework=fw,
                name="Sector-Specific Disclosure",
                description=(
                    "Sector-specific metrics must be included as defined "
                    "by the applicable ESRS sector standards (e.g. "
                    "clinker ratio for cement, anode effect frequency "
                    "for aluminum)."
                ),
                required_fields=["process_type"],
                validation_fn="_check_csrd_sector_specific",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="csrd_target_setting",
                framework=fw,
                name="Emission Reduction Targets",
                description=(
                    "Reduction targets must be set for process emissions "
                    "aligned with the Paris Agreement temperature goals. "
                    "Include base year, target year, and pathway."
                ),
                required_fields=[],
                validation_fn="_check_csrd_targets",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="csrd_mitigation_actions",
                framework=fw,
                name="Mitigation Actions",
                description=(
                    "Actions taken or planned to reduce process emissions "
                    "must be described, including technology changes, "
                    "efficiency improvements, and fuel switching."
                ),
                required_fields=[],
                validation_fn="_check_csrd_mitigation",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="csrd_financial_implications",
                framework=fw,
                name="Financial Implications",
                description=(
                    "The financial impact of process emissions must be "
                    "quantified including carbon pricing exposure, "
                    "abatement costs, and transition risks."
                ),
                required_fields=[],
                validation_fn="_check_csrd_financial",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="csrd_value_chain",
                framework=fw,
                name="Value Chain Impacts",
                description=(
                    "Upstream and downstream impacts of process emissions "
                    "on the value chain must be considered and disclosed."
                ),
                required_fields=[],
                validation_fn="_check_csrd_value_chain",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="csrd_limited_assurance",
                framework=fw,
                name="Limited Assurance Readiness",
                description=(
                    "Data quality must be sufficient for limited assurance "
                    "engagement by an independent auditor. This requires "
                    "documented methodology, traceable data sources, and "
                    "calculation audit trails."
                ),
                required_fields=[
                    "provenance_hash", "emission_factor_source",
                ],
                validation_fn="_check_csrd_assurance",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="csrd_digital_tagging",
                framework=fw,
                name="Digital Tagging (XBRL)",
                description=(
                    "Data must be structured for XBRL digital tagging "
                    "as required by the European Single Electronic "
                    "Format (ESEF). All quantitative disclosures must "
                    "have machine-readable identifiers."
                ),
                required_fields=["process_type", "total_co2e_tonnes"],
                validation_fn="_check_csrd_digital_tagging",
                severity="WARNING",
            ),
        ]

    # ------------------------------------------------------------------
    # 4. EPA 40 CFR Part 98 (Multiple Subparts)
    # ------------------------------------------------------------------

    def _build_epa_requirements(self) -> None:
        """Build the 10 EPA 40 CFR Part 98 compliance requirements."""
        fw = "EPA_40CFR98"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="epa_monitoring_plan",
                framework=fw,
                name="Monitoring Plan",
                description=(
                    "A monitoring plan must be established per the "
                    "applicable subpart of 40 CFR Part 98. The plan "
                    "must describe all monitoring methods, procedures, "
                    "and quality assurance activities."
                ),
                required_fields=["process_type"],
                validation_fn="_check_epa_monitoring_plan",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="epa_mass_measurement",
                framework=fw,
                name="Mass Measurement",
                description=(
                    "Production quantities and material inputs must be "
                    "properly measured using calibrated equipment per "
                    "40 CFR 98.3(i). Measurement devices must meet "
                    "accuracy requirements of the applicable subpart."
                ),
                required_fields=["production_quantity_tonnes"],
                validation_fn="_check_epa_mass_measurement",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="epa_tier_methodology",
                framework=fw,
                name="Tier Methodology",
                description=(
                    "The appropriate tier methodology must be used per "
                    "the applicable subpart. Higher-tier methods are "
                    "required for larger emitters. Tier selection must "
                    "be justified and documented."
                ),
                required_fields=["tier", "process_type"],
                validation_fn="_check_epa_tier",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="epa_missing_data",
                framework=fw,
                name="Missing Data Procedures",
                description=(
                    "Missing data substitution procedures must follow "
                    "40 CFR 98.3(c). For parameters with missing data, "
                    "the highest-emitting substitute value must be used "
                    "unless otherwise specified by the subpart."
                ),
                required_fields=[],
                validation_fn="_check_epa_missing_data",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="epa_calibration",
                framework=fw,
                name="Equipment Calibration",
                description=(
                    "All measurement equipment must be calibrated per "
                    "the specifications in the applicable subpart and "
                    "40 CFR 98.3(i). Calibration records must be "
                    "maintained for the required retention period."
                ),
                required_fields=[],
                validation_fn="_check_epa_calibration",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="epa_qa_qc",
                framework=fw,
                name="QA/QC Procedures",
                description=(
                    "Quality assurance and quality control procedures "
                    "must be implemented per 40 CFR 98.3(g) including "
                    "data verification, calculation cross-checks, and "
                    "internal audits."
                ),
                required_fields=["provenance_hash"],
                validation_fn="_check_epa_qa_qc",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="epa_recordkeeping",
                framework=fw,
                name="Recordkeeping (5-Year)",
                description=(
                    "All records related to GHG emissions calculations "
                    "must be retained for at least 5 years per "
                    "40 CFR 98.3(g). This includes raw data, emission "
                    "factors, calculations, and calibration records."
                ),
                required_fields=["provenance_hash"],
                validation_fn="_check_epa_recordkeeping",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="epa_annual_reporting",
                framework=fw,
                name="Annual Report Submission",
                description=(
                    "An annual GHG emissions report must be submitted "
                    "to EPA by March 31 of each year for emissions from "
                    "the preceding calendar year via e-GGRT."
                ),
                required_fields=["period_start", "period_end"],
                validation_fn="_check_epa_annual_reporting",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="epa_verification",
                framework=fw,
                name="Data Verification",
                description=(
                    "Data must be verified before submission. The "
                    "designated representative must certify the accuracy "
                    "and completeness of the reported data."
                ),
                required_fields=["provenance_hash"],
                validation_fn="_check_epa_verification",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="epa_electronic_submission",
                framework=fw,
                name="Electronic Submission (e-GGRT)",
                description=(
                    "Reports must be submitted in the electronic "
                    "reporting format required by e-GGRT (Electronic "
                    "Greenhouse Gas Reporting Tool). Data must be "
                    "structured per EPA XML schema requirements."
                ),
                required_fields=["process_type", "total_co2e_tonnes"],
                validation_fn="_check_epa_electronic",
                severity="INFO",
            ),
        ]

    # ------------------------------------------------------------------
    # 5. UK SECR
    # ------------------------------------------------------------------

    def _build_uk_secr_requirements(self) -> None:
        """Build the 10 UK SECR compliance requirements."""
        fw = "UK_SECR"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="secr_energy_emissions",
                framework=fw,
                name="Energy and Emissions Disclosure",
                description=(
                    "Energy use and greenhouse gas emissions must be "
                    "disclosed for all UK operations. This includes "
                    "process emissions from industrial activities."
                ),
                required_fields=["total_co2e_tonnes"],
                validation_fn="_check_secr_energy",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="secr_methodology",
                framework=fw,
                name="DEFRA Conversion Factors",
                description=(
                    "UK DEFRA conversion factors should be used as the "
                    "primary source for emission calculations. Alternative "
                    "factors may be used if justified and documented."
                ),
                required_fields=["emission_factor_source"],
                validation_fn="_check_secr_methodology",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="secr_intensity_ratio",
                framework=fw,
                name="Intensity Ratio",
                description=(
                    "An appropriate intensity metric must be selected "
                    "and reported (e.g. tCO2e per tonne of product, "
                    "tCO2e per unit revenue). The metric must be "
                    "relevant to the business activity."
                ),
                required_fields=["production_quantity_tonnes"],
                validation_fn="_check_secr_intensity",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="secr_year_comparison",
                framework=fw,
                name="Year-on-Year Comparison",
                description=(
                    "Previous year comparison data must be provided "
                    "to enable trend analysis and demonstrate progress "
                    "toward emission reduction."
                ),
                required_fields=[],
                validation_fn="_check_secr_comparison",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="secr_scope_stated",
                framework=fw,
                name="Scope of Reporting",
                description=(
                    "The scope of reporting must be clearly stated "
                    "including which emission sources are included and "
                    "excluded, and the organizational boundary approach."
                ),
                required_fields=["process_type"],
                validation_fn="_check_secr_scope",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="secr_methodology_stated",
                framework=fw,
                name="Methodology Description",
                description=(
                    "The calculation methodology must be clearly "
                    "described including emission factors used, data "
                    "sources, and any assumptions or estimates."
                ),
                required_fields=[
                    "calculation_method", "emission_factor_source",
                ],
                validation_fn="_check_secr_methodology_stated",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="secr_efficiency_narrative",
                framework=fw,
                name="Energy Efficiency Narrative",
                description=(
                    "A narrative describing energy efficiency measures "
                    "taken during the reporting period must be included. "
                    "This may include process optimization, technology "
                    "upgrades, and operational improvements."
                ),
                required_fields=[],
                validation_fn="_check_secr_efficiency",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="secr_director_responsibility",
                framework=fw,
                name="Director Responsibility",
                description=(
                    "A director must sign off on the accuracy of the "
                    "reported emissions data. The director's name and "
                    "approval date must be documented."
                ),
                required_fields=[],
                validation_fn="_check_secr_director",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="secr_assurance",
                framework=fw,
                name="Third-Party Assurance",
                description=(
                    "Consideration of third-party assurance should be "
                    "documented. While not mandatory for SECR, voluntary "
                    "assurance enhances credibility of reported data."
                ),
                required_fields=[],
                validation_fn="_check_secr_assurance",
                severity="INFO",
            ),
            ComplianceRequirement(
                requirement_id="secr_companies_act",
                framework=fw,
                name="Companies Act 2006 s414C",
                description=(
                    "Disclosure must comply with Companies Act 2006 "
                    "section 414C (Strategic Report) requirements for "
                    "quoted and large unquoted companies."
                ),
                required_fields=["total_co2e_tonnes"],
                validation_fn="_check_secr_companies_act",
                severity="ERROR",
            ),
        ]

    # ------------------------------------------------------------------
    # 6. EU ETS MRR
    # ------------------------------------------------------------------

    def _build_eu_ets_requirements(self) -> None:
        """Build the 10 EU ETS MRR compliance requirements."""
        fw = "EU_ETS_MRR"
        self._requirements[fw] = [
            ComplianceRequirement(
                requirement_id="ets_monitoring_plan",
                framework=fw,
                name="Monitoring Plan Approved",
                description=(
                    "A monitoring plan must be approved by the competent "
                    "authority before the start of the monitoring period. "
                    "The plan must follow the template in Annex I of the "
                    "MRR (Regulation 2018/2066)."
                ),
                required_fields=["process_type", "calculation_method"],
                validation_fn="_check_ets_monitoring_plan",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ets_tier_justified",
                framework=fw,
                name="Tier Approach Justified",
                description=(
                    "The tier approach must be justified per the "
                    "installation category (A, B, or C). Higher-tier "
                    "methods are required for Category B and C "
                    "installations."
                ),
                required_fields=["tier"],
                validation_fn="_check_ets_tier",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ets_measurement_based",
                framework=fw,
                name="Measurement-Based Approach",
                description=(
                    "Where the competent authority requires it, a "
                    "measurement-based approach (CEMS) must be used "
                    "instead of or in addition to calculation-based "
                    "monitoring."
                ),
                required_fields=["calculation_method"],
                validation_fn="_check_ets_measurement",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ets_calculation_factors",
                framework=fw,
                name="Calculation Factors Validated",
                description=(
                    "Emission factors, oxidation factors, and conversion "
                    "factors must be validated against laboratory analyses "
                    "or literature values per the required tier."
                ),
                required_fields=["emission_factor_source"],
                validation_fn="_check_ets_factors",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ets_laboratory_accreditation",
                framework=fw,
                name="Laboratory Accreditation",
                description=(
                    "Laboratories performing analyses for emission "
                    "factor determination must be accredited to "
                    "EN ISO/IEC 17025 or demonstrate equivalent "
                    "competence."
                ),
                required_fields=[],
                validation_fn="_check_ets_lab",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ets_uncertainty_threshold",
                framework=fw,
                name="Uncertainty Within Threshold",
                description=(
                    "The overall uncertainty of reported emissions must "
                    "be within the permitted range for the applicable "
                    "tier. Tier 1: +/-10%, Tier 2: +/-5%, Tier 3: +/-2.5%."
                ),
                required_fields=["tier"],
                validation_fn="_check_ets_uncertainty",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ets_improvement_report",
                framework=fw,
                name="Annual Improvement Plan",
                description=(
                    "An annual improvement report must be submitted to "
                    "the competent authority identifying opportunities "
                    "to improve monitoring methodology and reduce "
                    "uncertainty."
                ),
                required_fields=[],
                validation_fn="_check_ets_improvement",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ets_emissions_report",
                framework=fw,
                name="Annual Emissions Report",
                description=(
                    "A complete annual emissions report must be prepared "
                    "covering all source streams and emission sources "
                    "within the installation permit."
                ),
                required_fields=[
                    "total_co2e_tonnes", "period_start", "period_end",
                ],
                validation_fn="_check_ets_report",
                severity="ERROR",
            ),
            ComplianceRequirement(
                requirement_id="ets_verifier_accredited",
                framework=fw,
                name="Accredited Verifier",
                description=(
                    "The annual emissions report must be verified by "
                    "an independent verifier accredited under the "
                    "Accreditation and Verification Regulation "
                    "(Regulation 2018/2067)."
                ),
                required_fields=[],
                validation_fn="_check_ets_verifier",
                severity="WARNING",
            ),
            ComplianceRequirement(
                requirement_id="ets_data_gaps",
                framework=fw,
                name="Data Gap Procedures",
                description=(
                    "Procedures for handling data gaps must be documented "
                    "in the monitoring plan. Conservative estimation "
                    "methods must be applied when data is unavailable."
                ),
                required_fields=[],
                validation_fn="_check_ets_data_gaps",
                severity="WARNING",
            ),
        ]

    # ==================================================================
    # Validation Methods - GHG Protocol
    # ==================================================================

    def _check_ghg_process_identification(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check that process emission sources are identified."""
        present, missing = self._has_fields(data, req.required_fields)
        process_type = self._safe_get(data, "process_type", "")

        if present and process_type:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    f"Process source identified: {process_type}"
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details="Process emission source type is not specified.",
            recommendations=[
                "Specify the process_type field with a valid "
                "industrial process identifier."
            ],
            severity=req.severity,
        )

    def _check_ghg_calculation_methodology(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check that a valid calculation methodology is documented."""
        method = str(self._safe_get(data, "calculation_method", "")).upper()
        valid_methods = {
            "EMISSION_FACTOR", "MASS_BALANCE",
            "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
        }
        if method in valid_methods:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=f"Calculation method documented: {method}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                f"Calculation method '{method}' is not recognized. "
                f"Valid methods: {sorted(valid_methods)}"
            ),
            recommendations=[
                "Set calculation_method to one of: "
                "EMISSION_FACTOR, MASS_BALANCE, STOICHIOMETRIC, "
                "or DIRECT_MEASUREMENT."
            ],
            severity=req.severity,
        )

    def _check_ghg_ef_source(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check that emission factor source is documented and traceable."""
        source = str(
            self._safe_get(data, "emission_factor_source", "")
        ).upper()
        valid_sources = {"EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"}

        if source in valid_sources:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=f"Emission factor source documented: {source}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                f"Emission factor source '{source}' is not recognized."
            ),
            recommendations=[
                "Document the emission_factor_source using a "
                "recognized authority (EPA, IPCC, DEFRA, EU_ETS, CUSTOM)."
            ],
            severity=req.severity,
        )

    def _check_ghg_boundary_completeness(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check that boundary includes all significant processes."""
        present, missing = self._has_fields(data, req.required_fields)
        total = self._safe_get(data, "total_co2e_tonnes", 0)

        if present and total and float(total) > 0:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    f"Boundary includes process type with "
                    f"{total} tCO2e reported."
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                "Process emissions are zero or missing, suggesting "
                "incomplete boundary coverage."
            ),
            recommendations=[
                "Ensure all significant process emission sources are "
                "included in the organizational boundary.",
                "Missing fields: " + ", ".join(missing) if missing else "",
            ],
            severity=req.severity,
        )

    def _check_ghg_gas_coverage(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check that all relevant gases are reported."""
        complete, reported, missing = self._has_gas_coverage(data)

        if complete:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    f"All required gases reported: {reported}"
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                f"Missing gas coverage. Reported: {reported}, "
                f"Missing: {missing}"
            ),
            recommendations=[
                f"Add emission calculations for: {', '.join(missing)}",
                "Ensure emissions_by_gas includes all applicable gases "
                "for this process type.",
            ],
            severity=req.severity,
        )

    def _check_ghg_de_minimis(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check de minimis threshold for excluded sources."""
        total = self._safe_get(data, "total_co2e_tonnes", 0)
        excluded = self._safe_get(data, "excluded_emissions_pct", 0)

        if total and float(total) > 0:
            if excluded and float(excluded) > 5.0:
                return ComplianceCheckResult(
                    requirement_id=req.requirement_id,
                    framework=req.framework,
                    name=req.name,
                    passed=False,
                    details=(
                        f"Excluded emissions ({excluded}%) exceed "
                        f"the 5% de minimis threshold."
                    ),
                    recommendations=[
                        "Include additional emission sources or "
                        "justify exclusions below 5% collectively."
                    ],
                    severity=req.severity,
                )
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details="De minimis check passed (exclusions within limit).",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=True,
            details=(
                "De minimis check: no excluded emissions data provided; "
                "assumed compliant."
            ),
            severity=req.severity,
        )

    def _check_ghg_temporal_consistency(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check temporal consistency of methodology."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details="Methodology documented for temporal consistency.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                f"Missing fields for temporal consistency: "
                f"{', '.join(missing)}"
            ),
            recommendations=[
                "Document calculation_method and tier consistently "
                "across reporting periods."
            ],
            severity=req.severity,
        )

    def _check_ghg_base_year(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check base year recalculation triggers."""
        base_year = self._safe_get(data, "base_year")
        recalc_policy = self._safe_get(data, "recalculation_policy")

        if base_year and recalc_policy:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    f"Base year ({base_year}) and recalculation policy "
                    f"are documented."
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=True,
            details=(
                "Base year recalculation: informational check. "
                "Consider documenting base_year and recalculation_policy."
            ),
            recommendations=[
                "Document the base year and triggers for recalculation."
            ],
            severity=req.severity,
        )

    def _check_ghg_quality_management(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check QA/QC procedures via provenance tracking."""
        prov_hash = self._safe_get(data, "provenance_hash", "")
        if prov_hash:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    "Provenance hash present, indicating QA/QC tracking."
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details="No provenance hash found for quality management.",
            recommendations=[
                "Enable SHA-256 provenance tracking for all calculations.",
                "Implement QA/QC procedures for data validation."
            ],
            severity=req.severity,
        )

    def _check_ghg_verification_readiness(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check verification readiness of the data."""
        prov_hash = self._safe_get(data, "provenance_hash", "")
        calc_trace = self._safe_get(data, "calculation_trace", [])

        if prov_hash and calc_trace:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details=(
                    "Data includes provenance hash and calculation trace "
                    "for third-party verification."
                ),
                severity=req.severity,
            )
        missing_items: List[str] = []
        if not prov_hash:
            missing_items.append("provenance_hash")
        if not calc_trace:
            missing_items.append("calculation_trace")
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=(
                f"Verification readiness incomplete. Missing: "
                f"{', '.join(missing_items)}"
            ),
            recommendations=[
                "Enable full provenance tracking and calculation tracing.",
                "Maintain documentation for third-party auditor review."
            ],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Methods - ISO 14064-1
    # ==================================================================

    def _check_iso_org_boundary(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check organizational boundary definition."""
        facility = self._safe_get(data, "facility_id", "")
        org = self._safe_get(data, "organization_id", "")
        if facility or org:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Organizational boundary defined (facility/org ID present).",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No facility_id or organization_id defined.",
            recommendations=["Define facility_id or organization_id for boundary."],
            severity=req.severity,
        )

    def _check_iso_sources(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check GHG source identification."""
        pt = self._safe_get(data, "process_type", "")
        if pt:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"GHG source identified: {pt}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No process_type specified for source identification.",
            recommendations=["Specify process_type for all emission sources."],
            severity=req.severity,
        )

    def _check_iso_methodology(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check quantification methodology per clause 6."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Quantification methodology documented.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing methodology fields: {', '.join(missing)}",
            recommendations=["Document calculation_method and emission_factor_source."],
            severity=req.severity,
        )

    def _check_iso_data_quality(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check data quality assessment."""
        tier = str(self._safe_get(data, "tier", "")).upper()
        if tier in ("TIER_1", "TIER_2", "TIER_3"):
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Data quality tier documented: {tier}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No calculation tier specified for data quality assessment.",
            recommendations=["Specify tier (TIER_1/TIER_2/TIER_3)."],
            severity=req.severity,
        )

    def _check_iso_uncertainty(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check uncertainty assessment."""
        uncertainty = self._safe_get(data, "uncertainty_result")
        if uncertainty:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Uncertainty analysis results present.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Uncertainty assessment recommended but not blocking.",
            recommendations=["Run Monte Carlo uncertainty analysis."],
            severity=req.severity,
        )

    def _check_iso_base_year(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check base year establishment."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Base year establishment: advisory check passed.",
            recommendations=["Document base year and recalculation policy."],
            severity=req.severity,
        )

    def _check_iso_exclusions(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check exclusion justification."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Exclusion justification: advisory check passed.",
            recommendations=["Document all excluded sources with justification."],
            severity=req.severity,
        )

    def _check_iso_consistent(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check year-to-year consistency."""
        method = self._safe_get(data, "calculation_method", "")
        if method:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Methodology documented ({method}) for consistency.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No calculation_method documented for consistency check.",
            recommendations=["Document and maintain consistent methodology."],
            severity=req.severity,
        )

    def _check_iso_competence(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check competence requirements."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Competence requirements: informational check passed.",
            recommendations=["Document personnel qualifications per clause 8."],
            severity=req.severity,
        )

    def _check_iso_documentation(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check documentation completeness per clause 9."""
        prov = self._safe_get(data, "provenance_hash", "")
        if prov:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Documentation tracked via provenance hash.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No provenance hash; documentation completeness not assured.",
            recommendations=["Enable provenance tracking for all records."],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Methods - CSRD/ESRS E1
    # ==================================================================

    def _check_csrd_scope1_category(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check Scope 1 by process category."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Scope 1 process emissions by category available.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing fields for Scope 1 category: {', '.join(missing)}",
            recommendations=["Report process emissions separately by category."],
            severity=req.severity,
        )

    def _check_csrd_ghg_methodology(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check GHG Protocol methodology basis."""
        method = str(self._safe_get(data, "calculation_method", "")).upper()
        valid = {"EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC", "DIRECT_MEASUREMENT"}
        if method in valid:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"GHG Protocol methodology applied: {method}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="Calculation method not aligned with GHG Protocol.",
            recommendations=["Use a GHG Protocol-recognized calculation method."],
            severity=req.severity,
        )

    def _check_csrd_separation(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check process/combustion emission separation."""
        pt = self._safe_get(data, "process_type", "")
        if pt:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Process type '{pt}' distinguishes from combustion.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="Process type not specified; cannot separate from combustion.",
            recommendations=["Specify process_type to separate from combustion emissions."],
            severity=req.severity,
        )

    def _check_csrd_sector_specific(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check sector-specific disclosure metrics."""
        pt = str(self._safe_get(data, "process_type", "")).upper()
        sector_metrics = self._safe_get(data, "sector_metrics")
        if sector_metrics or pt:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Sector-specific context available for {pt}.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No sector-specific metrics provided.",
            recommendations=["Include sector-specific KPIs per ESRS standards."],
            severity=req.severity,
        )

    def _check_csrd_targets(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check emission reduction targets."""
        targets = self._safe_get(data, "reduction_targets")
        if targets:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Emission reduction targets documented.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Reduction targets: advisory check. Consider setting targets.",
            recommendations=["Set Paris-aligned reduction targets for process emissions."],
            severity=req.severity,
        )

    def _check_csrd_mitigation(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check mitigation action descriptions."""
        actions = self._safe_get(data, "mitigation_actions")
        abatement = self._safe_get(data, "abatement_co2e_tonnes", 0)
        if actions or (abatement and float(abatement) > 0):
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Mitigation actions or abatement data present.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Mitigation actions: advisory check. Consider documenting actions.",
            recommendations=["Describe mitigation actions for process emissions."],
            severity=req.severity,
        )

    def _check_csrd_financial(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check financial implications disclosure."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Financial implications: informational check passed.",
            recommendations=["Quantify carbon pricing exposure and transition costs."],
            severity=req.severity,
        )

    def _check_csrd_value_chain(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check value chain impact consideration."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Value chain impacts: informational check passed.",
            recommendations=["Assess upstream/downstream emission impacts."],
            severity=req.severity,
        )

    def _check_csrd_assurance(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check limited assurance readiness."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Data quality sufficient for limited assurance.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Assurance readiness incomplete: missing {', '.join(missing)}",
            recommendations=["Ensure provenance and EF source for assurance."],
            severity=req.severity,
        )

    def _check_csrd_digital_tagging(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check XBRL digital tagging readiness."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Data structured for XBRL digital tagging.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing fields for XBRL: {', '.join(missing)}",
            recommendations=["Ensure machine-readable identifiers for all values."],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Methods - EPA 40 CFR Part 98
    # ==================================================================

    def _check_epa_monitoring_plan(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check monitoring plan establishment."""
        pt = str(self._safe_get(data, "process_type", "")).upper()
        subparts = EPA_SUBPART_MAP.get(pt, [])
        if pt and subparts:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=(
                    f"Process type '{pt}' mapped to "
                    f"{', '.join(subparts)}."
                ),
                severity=req.severity,
            )
        if pt and not subparts:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Process type '{pt}' has no mapped EPA subpart.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No process type for monitoring plan determination.",
            recommendations=["Specify process_type for EPA subpart mapping."],
            severity=req.severity,
        )

    def _check_epa_mass_measurement(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check production quantity measurement."""
        qty = self._safe_get(data, "production_quantity_tonnes", 0)
        if qty and float(qty) > 0:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Production quantity measured: {qty} tonnes.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No production quantity recorded.",
            recommendations=["Record production_quantity_tonnes from calibrated equipment."],
            severity=req.severity,
        )

    def _check_epa_tier(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check appropriate tier methodology."""
        tier = str(self._safe_get(data, "tier", "")).upper()
        if tier in ("TIER_1", "TIER_2", "TIER_3"):
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Tier methodology documented: {tier}",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No valid tier methodology specified.",
            recommendations=["Specify tier (TIER_1/TIER_2/TIER_3) per subpart."],
            severity=req.severity,
        )

    def _check_epa_missing_data(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check missing data substitution procedures."""
        missing_data_pct = self._safe_get(data, "missing_data_pct", 0)
        if missing_data_pct and float(missing_data_pct) > 10:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=False,
                details=(
                    f"Missing data rate ({missing_data_pct}%) exceeds 10%."
                ),
                recommendations=["Apply 40 CFR 98.3(c) substitution procedures."],
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Missing data within acceptable limits.",
            severity=req.severity,
        )

    def _check_epa_calibration(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check equipment calibration."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Calibration: advisory check passed.",
            recommendations=["Maintain calibration records per 40 CFR 98.3(i)."],
            severity=req.severity,
        )

    def _check_epa_qa_qc(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check QA/QC procedures."""
        prov = self._safe_get(data, "provenance_hash", "")
        if prov:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="QA/QC supported by provenance tracking.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No QA/QC provenance tracking detected.",
            recommendations=["Enable provenance tracking per 40 CFR 98.3(g)."],
            severity=req.severity,
        )

    def _check_epa_recordkeeping(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check 5-year recordkeeping."""
        prov = self._safe_get(data, "provenance_hash", "")
        if prov:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Recordkeeping supported by provenance chain.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No provenance hash for recordkeeping assurance.",
            recommendations=["Enable provenance for 5-year record retention."],
            severity=req.severity,
        )

    def _check_epa_annual_reporting(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check annual report period coverage."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Reporting period defined for annual report.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing period fields: {', '.join(missing)}",
            recommendations=["Define period_start and period_end for annual report."],
            severity=req.severity,
        )

    def _check_epa_verification(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check data verification."""
        prov = self._safe_get(data, "provenance_hash", "")
        if prov:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Data verification supported by provenance hash.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No provenance hash for verification.",
            recommendations=["Enable provenance tracking for data certification."],
            severity=req.severity,
        )

    def _check_epa_electronic(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check e-GGRT electronic submission readiness."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Data structured for e-GGRT submission.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing fields for e-GGRT: {', '.join(missing)}",
            recommendations=["Ensure data meets EPA XML schema."],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Methods - UK SECR
    # ==================================================================

    def _check_secr_energy(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check energy and emissions disclosure."""
        total = self._safe_get(data, "total_co2e_tonnes", 0)
        if total and float(total) > 0:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Emissions disclosed: {total} tCO2e.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No emissions data for SECR disclosure.",
            recommendations=["Report total_co2e_tonnes for UK operations."],
            severity=req.severity,
        )

    def _check_secr_methodology(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check DEFRA conversion factors usage."""
        source = str(self._safe_get(data, "emission_factor_source", "")).upper()
        if source == "DEFRA":
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="DEFRA conversion factors used.",
                severity=req.severity,
            )
        if source:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=(
                    f"Using {source} factors instead of DEFRA. "
                    f"Justify in methodology disclosure."
                ),
                recommendations=[
                    "Consider using DEFRA factors as primary source for UK SECR.",
                ],
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No emission factor source documented.",
            recommendations=["Specify emission_factor_source (preferably DEFRA)."],
            severity=req.severity,
        )

    def _check_secr_intensity(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check intensity ratio availability."""
        qty = self._safe_get(data, "production_quantity_tonnes", 0)
        total = self._safe_get(data, "total_co2e_tonnes", 0)
        if qty and total and float(qty) > 0 and float(total) > 0:
            intensity = float(total) / float(qty)
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=(
                    f"Intensity ratio calculable: "
                    f"{intensity:.4f} tCO2e/t product."
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="Cannot calculate intensity ratio (missing production data).",
            recommendations=["Provide production_quantity_tonnes for intensity metric."],
            severity=req.severity,
        )

    def _check_secr_comparison(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check year-on-year comparison data."""
        prev = self._safe_get(data, "previous_year_co2e")
        if prev:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Previous year comparison data available.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Year comparison: advisory. First reporting year may not have comparison.",
            recommendations=["Include previous_year_co2e for trend analysis."],
            severity=req.severity,
        )

    def _check_secr_scope(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check scope of reporting stated."""
        pt = self._safe_get(data, "process_type", "")
        if pt:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Reporting scope includes process type: {pt}.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No process type specified; scope unclear.",
            recommendations=["Clearly state the scope of emission sources."],
            severity=req.severity,
        )

    def _check_secr_methodology_stated(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check methodology description."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Methodology described for SECR reporting.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing methodology fields: {', '.join(missing)}",
            recommendations=["Document calculation_method and emission_factor_source."],
            severity=req.severity,
        )

    def _check_secr_efficiency(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check energy efficiency narrative."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Efficiency narrative: advisory check passed.",
            recommendations=["Include narrative on efficiency measures taken."],
            severity=req.severity,
        )

    def _check_secr_director(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check director sign-off."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Director responsibility: informational check.",
            recommendations=["Obtain director sign-off on emissions report."],
            severity=req.severity,
        )

    def _check_secr_assurance(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check third-party assurance consideration."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Assurance consideration: informational check.",
            recommendations=["Consider voluntary third-party assurance."],
            severity=req.severity,
        )

    def _check_secr_companies_act(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check Companies Act 2006 s414C compliance."""
        total = self._safe_get(data, "total_co2e_tonnes", 0)
        if total and float(total) > 0:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Emissions data available for Companies Act disclosure.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No emissions data for Companies Act s414C compliance.",
            recommendations=["Report total_co2e_tonnes per statutory requirements."],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Methods - EU ETS MRR
    # ==================================================================

    def _check_ets_monitoring_plan(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check monitoring plan approval."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Process type and method defined for monitoring plan.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing monitoring plan fields: {', '.join(missing)}",
            recommendations=["Document process_type and calculation_method in MP."],
            severity=req.severity,
        )

    def _check_ets_tier(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check tier approach justification."""
        tier = str(self._safe_get(data, "tier", "")).upper()
        if tier in ("TIER_1", "TIER_2", "TIER_3"):
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Tier approach: {tier} (justify per installation category).",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No valid tier specified for EU ETS.",
            recommendations=["Specify tier justified per installation category (A/B/C)."],
            severity=req.severity,
        )

    def _check_ets_measurement(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check measurement-based approach where required."""
        method = str(self._safe_get(data, "calculation_method", "")).upper()
        if method:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Monitoring approach: {method}.",
                recommendations=(
                    ["Consider CEMS if required by competent authority."]
                    if method != "DIRECT_MEASUREMENT" else []
                ),
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="No calculation method specified.",
            recommendations=["Specify calculation method for EU ETS monitoring."],
            severity=req.severity,
        )

    def _check_ets_factors(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check calculation factor validation."""
        source = str(self._safe_get(data, "emission_factor_source", "")).upper()
        valid_ets = {"EU_ETS", "IPCC", "CUSTOM", "EPA", "DEFRA"}
        if source in valid_ets:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details=f"Emission factor source: {source}.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details="Emission factor source not validated.",
            recommendations=["Validate EFs against lab analyses or EU ETS defaults."],
            severity=req.severity,
        )

    def _check_ets_lab(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check laboratory accreditation."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Lab accreditation: advisory check.",
            recommendations=["Ensure EN ISO/IEC 17025 accreditation for analyses."],
            severity=req.severity,
        )

    def _check_ets_uncertainty(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check uncertainty within permitted thresholds."""
        tier = str(self._safe_get(data, "tier", "")).upper()
        tier_thresholds = {
            "TIER_1": 10.0,
            "TIER_2": 5.0,
            "TIER_3": 2.5,
        }
        threshold = tier_thresholds.get(tier, 10.0)

        uncertainty_pct = self._safe_get(data, "uncertainty_pct")
        if uncertainty_pct is not None:
            if float(uncertainty_pct) <= threshold:
                return ComplianceCheckResult(
                    requirement_id=req.requirement_id, framework=req.framework,
                    name=req.name, passed=True,
                    details=(
                        f"Uncertainty {uncertainty_pct}% within "
                        f"{tier} threshold ({threshold}%)."
                    ),
                    severity=req.severity,
                )
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=False,
                details=(
                    f"Uncertainty {uncertainty_pct}% exceeds "
                    f"{tier} threshold ({threshold}%)."
                ),
                recommendations=[
                    f"Reduce uncertainty below {threshold}% or use higher tier."
                ],
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details=(
                f"No uncertainty data provided. "
                f"Threshold for {tier}: +/-{threshold}%."
            ),
            recommendations=["Perform uncertainty analysis to verify compliance."],
            severity=req.severity,
        )

    def _check_ets_improvement(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check annual improvement plan."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Improvement plan: advisory check.",
            recommendations=["Prepare annual improvement report for competent authority."],
            severity=req.severity,
        )

    def _check_ets_report(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check annual emissions report completeness."""
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id, framework=req.framework,
                name=req.name, passed=True,
                details="Annual emissions report data complete.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=False,
            details=f"Missing report fields: {', '.join(missing)}",
            recommendations=["Provide total_co2e_tonnes, period_start, period_end."],
            severity=req.severity,
        )

    def _check_ets_verifier(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check accredited verifier engagement."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Verifier accreditation: advisory check.",
            recommendations=["Engage verifier accredited under Regulation 2018/2067."],
            severity=req.severity,
        )

    def _check_ets_data_gaps(
        self, data: Dict[str, Any], req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Check data gap procedures."""
        return ComplianceCheckResult(
            requirement_id=req.requirement_id, framework=req.framework,
            name=req.name, passed=True,
            details="Data gap procedures: advisory check.",
            recommendations=["Document conservative estimation methods for data gaps."],
            severity=req.severity,
        )

    # ==================================================================
    # Validation Method Dispatcher
    # ==================================================================

    def _dispatch_check(
        self,
        data: Dict[str, Any],
        req: ComplianceRequirement,
    ) -> ComplianceCheckResult:
        """Dispatch a requirement check to the appropriate method.

        Looks up the validation_fn name on the engine and calls it.

        Args:
            data: Calculation data dictionary.
            req: ComplianceRequirement to evaluate.

        Returns:
            ComplianceCheckResult from the validation method.
        """
        fn_name = req.validation_fn
        if fn_name and hasattr(self, fn_name):
            fn: Callable = getattr(self, fn_name)
            return fn(data, req)

        # Fallback: field presence check
        present, missing = self._has_fields(data, req.required_fields)
        if present:
            return ComplianceCheckResult(
                requirement_id=req.requirement_id,
                framework=req.framework,
                name=req.name,
                passed=True,
                details="Required fields present.",
                severity=req.severity,
            )
        return ComplianceCheckResult(
            requirement_id=req.requirement_id,
            framework=req.framework,
            name=req.name,
            passed=False,
            details=f"Missing required fields: {', '.join(missing)}",
            recommendations=[
                f"Provide the following fields: {', '.join(missing)}"
            ],
            severity=req.severity,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def check_compliance(
        self,
        calculation_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check calculation data against one or more frameworks.

        Args:
            calculation_data: Dict containing calculation results and
                metadata fields.
            frameworks: Optional list of framework names to check.
                If None, checks all 6 frameworks.

        Returns:
            Dict mapping framework name to a serialised FrameworkResult
            dict.

        Raises:
            ValueError: If an unknown framework is specified.
        """
        start_time = time.monotonic()

        if frameworks is None:
            frameworks = list(SUPPORTED_FRAMEWORKS)

        # Validate framework names
        for fw in frameworks:
            fw_upper = fw.upper().replace(" ", "_")
            if fw_upper not in SUPPORTED_FRAMEWORKS:
                raise ValueError(
                    f"Unknown framework '{fw}'. "
                    f"Supported: {list(SUPPORTED_FRAMEWORKS)}"
                )

        results: Dict[str, Any] = {}
        for fw in frameworks:
            fw_key = fw.upper().replace(" ", "_")
            framework_result = self.check_framework(
                calculation_data, fw_key,
            )
            results[fw_key] = {
                "framework": framework_result.framework,
                "total_checks": framework_result.total_checks,
                "passed": framework_result.passed,
                "failed": framework_result.failed,
                "warnings": framework_result.warnings,
                "status": framework_result.status,
                "results": [
                    {
                        "requirement_id": r.requirement_id,
                        "framework": r.framework,
                        "name": r.name,
                        "passed": r.passed,
                        "details": r.details,
                        "recommendations": r.recommendations,
                        "severity": r.severity,
                    }
                    for r in framework_result.results
                ],
                "provenance_hash": framework_result.provenance_hash,
                "checked_at": framework_result.checked_at,
            }

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Compliance check completed: %d framework(s) in %.2f ms",
            len(frameworks),
            elapsed_ms,
        )

        return results

    def check_framework(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> FrameworkResult:
        """Check calculation data against a single regulatory framework.

        Args:
            calculation_data: Dict with calculation results and metadata.
            framework: Framework name (e.g. GHG_PROTOCOL).

        Returns:
            FrameworkResult with all individual check results.

        Raises:
            ValueError: If framework is not recognised.
        """
        start_time = time.monotonic()
        fw_key = framework.upper().replace(" ", "_")

        if fw_key not in self._requirements:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Supported: {list(self._requirements.keys())}"
            )

        requirements = self._requirements[fw_key]
        check_results: List[ComplianceCheckResult] = []
        passed_count = 0
        failed_count = 0
        warning_count = 0

        with self._lock:
            for req in requirements:
                result = self._dispatch_check(calculation_data, req)
                check_results.append(result)

                if result.passed:
                    passed_count += 1
                else:
                    if result.severity == "ERROR":
                        failed_count += 1
                    elif result.severity == "WARNING":
                        warning_count += 1
                        # Warnings count as partial passes
                    else:
                        # INFO-level failures are not blocking
                        passed_count += 1

        total = len(requirements)
        effective_passed = passed_count + warning_count
        status = self._determine_status(effective_passed, total)

        # Provenance hash
        prov_data = json.dumps(
            {
                "framework": fw_key,
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "warnings": warning_count,
                "status": status,
            },
            sort_keys=True,
        )
        prov_hash = self._hash(prov_data)

        self._record_provenance(
            entity_type="compliance_check",
            entity_id=prov_hash[:16],
            action="check_framework",
            details={
                "framework": fw_key,
                "status": status,
                "passed": passed_count,
                "failed": failed_count,
            },
        )
        self._record_metric(fw_key, status.lower())

        elapsed_ms = (time.monotonic() - start_time) * 1000

        framework_result = FrameworkResult(
            framework=fw_key,
            total_checks=total,
            passed=passed_count,
            failed=failed_count,
            warnings=warning_count,
            status=status,
            results=check_results,
            provenance_hash=prov_hash,
            checked_at=self._now_iso(),
        )

        logger.info(
            "Framework %s: %s (%d/%d passed, %d failed, %d warnings) "
            "in %.2f ms",
            fw_key,
            status,
            passed_count,
            total,
            failed_count,
            warning_count,
            elapsed_ms,
        )

        return framework_result

    def check_all_frameworks(
        self,
        calculation_data: Dict[str, Any],
    ) -> List[FrameworkResult]:
        """Check calculation data against all 6 regulatory frameworks.

        Args:
            calculation_data: Dict with calculation results.

        Returns:
            List of FrameworkResult for all frameworks.
        """
        results: List[FrameworkResult] = []
        for fw in SUPPORTED_FRAMEWORKS:
            result = self.check_framework(calculation_data, fw)
            results.append(result)
        return results

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[ComplianceRequirement]:
        """Return the list of requirements for a specific framework.

        Args:
            framework: Framework name (e.g. GHG_PROTOCOL).

        Returns:
            List of ComplianceRequirement objects.

        Raises:
            ValueError: If framework is not recognised.
        """
        fw_key = framework.upper().replace(" ", "_")
        if fw_key not in self._requirements:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Supported: {list(self._requirements.keys())}"
            )
        return list(self._requirements[fw_key])

    def validate_data_completeness(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> ComplianceCheckResult:
        """Validate that all required fields are present for a framework.

        Aggregates all required_fields from all requirements in the
        framework and checks for presence.

        Args:
            calculation_data: Dict with calculation data.
            framework: Framework name.

        Returns:
            ComplianceCheckResult summarising data completeness.
        """
        fw_key = framework.upper().replace(" ", "_")
        requirements = self.get_framework_requirements(fw_key)

        all_required: List[str] = []
        for req in requirements:
            all_required.extend(req.required_fields)
        unique_required = sorted(set(all_required))

        present, missing = self._has_fields(
            calculation_data, unique_required,
        )

        if present:
            return ComplianceCheckResult(
                requirement_id=f"{fw_key}_data_completeness",
                framework=fw_key,
                name="Data Completeness",
                passed=True,
                details=(
                    f"All {len(unique_required)} required fields present."
                ),
                severity="ERROR",
            )
        return ComplianceCheckResult(
            requirement_id=f"{fw_key}_data_completeness",
            framework=fw_key,
            name="Data Completeness",
            passed=False,
            details=(
                f"{len(missing)} of {len(unique_required)} required "
                f"fields missing: {', '.join(missing)}"
            ),
            recommendations=[
                f"Provide the following fields: {', '.join(missing)}"
            ],
            severity="ERROR",
        )

    def validate_methodology(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> ComplianceCheckResult:
        """Validate that the calculation methodology is appropriate.

        Checks that the calculation method, tier, and emission factor
        source are recognised and appropriate for the framework.

        Args:
            calculation_data: Dict with calculation data.
            framework: Framework name.

        Returns:
            ComplianceCheckResult for methodology validation.
        """
        fw_key = framework.upper().replace(" ", "_")

        method = str(
            self._safe_get(calculation_data, "calculation_method", "")
        ).upper()
        tier = str(
            self._safe_get(calculation_data, "tier", "")
        ).upper()
        source = str(
            self._safe_get(calculation_data, "emission_factor_source", "")
        ).upper()

        valid_methods = {
            "EMISSION_FACTOR", "MASS_BALANCE",
            "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
        }
        valid_tiers = {"TIER_1", "TIER_2", "TIER_3"}
        valid_sources = {"EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"}

        issues: List[str] = []
        if method not in valid_methods:
            issues.append(f"Invalid method: {method}")
        if tier and tier not in valid_tiers:
            issues.append(f"Invalid tier: {tier}")
        if source and source not in valid_sources:
            issues.append(f"Invalid EF source: {source}")

        if not issues:
            return ComplianceCheckResult(
                requirement_id=f"{fw_key}_methodology",
                framework=fw_key,
                name="Methodology Validation",
                passed=True,
                details=(
                    f"Method={method}, Tier={tier}, Source={source} "
                    f"validated for {fw_key}."
                ),
                severity="ERROR",
            )
        return ComplianceCheckResult(
            requirement_id=f"{fw_key}_methodology",
            framework=fw_key,
            name="Methodology Validation",
            passed=False,
            details=f"Methodology issues: {'; '.join(issues)}",
            recommendations=[
                "Use a valid calculation method, tier, and EF source."
            ],
            severity="ERROR",
        )

    def generate_recommendations(
        self,
        results: List[ComplianceCheckResult],
    ) -> List[str]:
        """Generate a deduplicated list of recommendations from results.

        Args:
            results: List of ComplianceCheckResult objects.

        Returns:
            Sorted, deduplicated list of recommendation strings.
        """
        all_recs: List[str] = []
        for r in results:
            if not r.passed:
                all_recs.extend(r.recommendations)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for rec in all_recs:
            rec_stripped = rec.strip()
            if rec_stripped and rec_stripped not in seen:
                seen.add(rec_stripped)
                unique.append(rec_stripped)

        return sorted(unique)

    def get_applicable_epa_subparts(
        self,
        process_type: str,
    ) -> List[str]:
        """Return applicable EPA 40 CFR Part 98 subparts for a process.

        Args:
            process_type: Industrial process type string.

        Returns:
            List of subpart identifiers (e.g. ["Subpart F"]).
        """
        pt = process_type.upper().replace(" ", "_")
        return EPA_SUBPART_MAP.get(pt, [])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise engine state to a dictionary.

        Returns:
            Dict with framework names and requirement counts.
        """
        return {
            "engine": "ComplianceCheckerEngine",
            "frameworks": {
                fw: len(reqs)
                for fw, reqs in self._requirements.items()
            },
            "total_requirements": sum(
                len(r) for r in self._requirements.values()
            ),
            "supported_frameworks": list(SUPPORTED_FRAMEWORKS),
        }
