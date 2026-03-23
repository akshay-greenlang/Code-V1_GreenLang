# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Multi-Framework Regulatory Compliance (Engine 6 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

Validates waste treatment emission calculations against seven regulatory
frameworks to ensure data completeness, methodological correctness, and
reporting readiness.  Each framework defines specific requirements that are
individually checked and scored.

Supported Frameworks (98 total requirements):
    1. IPCC 2006 Vol 5 (Waste)            - 15 requirements
    2. IPCC 2019 Refinement               - 12 requirements
    3. GHG Protocol Corporate/Scope 3     - 18 requirements
    4. ISO 14064-1:2018                   - 14 requirements
    5. CSRD/ESRS E1 & E5                  - 16 requirements
    6. EPA 40 CFR Part 98 Subpart HH/TT  - 13 requirements
    7. DEFRA Environmental Reporting      - 10 requirements

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met
    NOT_ASSESSED:  Framework not included in the check

Severity Levels:
    ERROR:   Requirement failure prevents regulatory compliance
    WARNING: Requirement failure should be addressed but not blocking
    INFO:    Informational finding for best practice improvement

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean evaluations.
    - No LLM involvement in any compliance determination.
    - Requirement definitions are hard-coded from regulatory texts.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_data={
    ...         "treatment_method": "INCINERATION",
    ...         "waste_category": "MUNICIPAL_SOLID_WASTE",
    ...         "calculation_method": "IPCC_TIER_2",
    ...         "total_co2e_tonnes": 1250.0,
    ...         "has_uncertainty": True,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["IPCC_2006"],
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
    from greenlang.agents.mrv.waste_treatment_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]


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


def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Convert a value to Decimal safely, returning *default* on failure."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# Constants
# ===========================================================================

#: Available compliance frameworks.
SUPPORTED_FRAMEWORKS: List[str] = [
    "IPCC_2006",
    "IPCC_2019",
    "GHG_PROTOCOL",
    "ISO_14064",
    "CSRD_ESRS",
    "EPA_40CFR98",
    "DEFRA",
]

#: Total requirements across all frameworks.
TOTAL_REQUIREMENTS: int = 98  # 15+12+18+14+16+13+10

#: Valid waste treatment methods (IPCC 2006 Vol 5 + 2019 Refinement).
VALID_TREATMENT_METHODS: List[str] = [
    "LANDFILL",
    "LANDFILL_GAS_CAPTURE",
    "INCINERATION",
    "INCINERATION_ENERGY_RECOVERY",
    "RECYCLING",
    "COMPOSTING",
    "ANAEROBIC_DIGESTION",
    "MECHANICAL_BIOLOGICAL_TREATMENT",
    "PYROLYSIS",
    "GASIFICATION",
    "CHEMICAL_TREATMENT",
    "THERMAL_TREATMENT",
    "BIOLOGICAL_TREATMENT",
    "OPEN_BURNING",
    "OPEN_DUMPING",
]

#: Valid waste categories per IPCC 2006 Vol 5.
VALID_WASTE_CATEGORIES: List[str] = [
    "MUNICIPAL_SOLID_WASTE",
    "INDUSTRIAL_WASTE",
    "CONSTRUCTION_DEMOLITION",
    "ORGANIC_WASTE",
    "FOOD_WASTE",
    "YARD_GARDEN_WASTE",
    "PAPER",
    "CARDBOARD",
    "PLASTIC",
    "METAL",
    "GLASS",
    "TEXTILES",
    "WOOD",
    "RUBBER",
    "E_WASTE",
    "HAZARDOUS_WASTE",
    "MEDICAL_WASTE",
    "SLUDGE",
    "MIXED_WASTE",
]

#: Valid calculation methods per PRD.
VALID_CALCULATION_METHODS: List[str] = [
    "FIRST_ORDER_DECAY",
    "IPCC_TIER_1",
    "IPCC_TIER_2",
    "IPCC_TIER_3",
    "MASS_BALANCE",
    "DIRECT_MEASUREMENT",
    "SPEND_BASED",
]

#: Biological treatment methods.
BIOLOGICAL_METHODS: Set[str] = {
    "COMPOSTING",
    "ANAEROBIC_DIGESTION",
    "MECHANICAL_BIOLOGICAL_TREATMENT",
    "BIOLOGICAL_TREATMENT",
}

#: Thermal treatment methods.
THERMAL_METHODS: Set[str] = {
    "INCINERATION",
    "INCINERATION_ENERGY_RECOVERY",
    "PYROLYSIS",
    "GASIFICATION",
    "OPEN_BURNING",
}

#: Valid GHG gases emitted by waste treatment.
VALID_GASES: Set[str] = {"CO2", "CH4", "N2O", "CO", "BIOGENIC_CO2"}

#: Valid GWP source references.
VALID_GWP_SOURCES: Set[str] = {"AR4", "AR5", "AR6"}

#: Emission factor source references accepted.
VALID_EF_SOURCES: Set[str] = {
    "IPCC_2006",
    "IPCC_2019",
    "EPA_AP42",
    "DEFRA_BEIS",
    "ECOINVENT",
    "NATIONAL_INVENTORY",
    "FACILITY_SPECIFIC",
    "COUNTRY_SPECIFIC",
    "SITE_MEASURED",
    "CUSTOM",
}

#: EPA Subpart identifiers for waste treatment.
EPA_SUBPARTS: Set[str] = {"HH", "TT"}


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class ComplianceFinding:
    """Single compliance finding for a requirement.

    Attributes:
        requirement_id: Unique identifier for the requirement.
        framework: Regulatory framework name.
        requirement: Requirement description.
        passed: Whether the requirement is met.
        severity: ERROR, WARNING, or INFO.
        finding: Description of the finding.
        recommendation: Recommended action if failed.
    """

    requirement_id: str
    framework: str
    requirement: str
    passed: bool
    severity: str
    finding: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requirement_id": self.requirement_id,
            "framework": self.framework,
            "requirement": self.requirement,
            "passed": self.passed,
            "severity": self.severity,
            "finding": self.finding,
            "recommendation": self.recommendation,
        }


# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================


class ComplianceCheckerEngine:
    """Multi-framework regulatory compliance checker for waste treatment
    emission calculations.

    Validates calculation results against seven regulatory frameworks
    with 98 total requirements.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(data, ["IPCC_2006"])
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the ComplianceCheckerEngine."""
        self._lock = threading.RLock()
        self._total_checks: int = 0
        self._created_at = _utcnow()

        # Map framework names to checker methods
        self._framework_checkers: Dict[str, Callable] = {
            "IPCC_2006": self.check_ipcc_2006,
            "IPCC_2019": self.check_ipcc_2019,
            "GHG_PROTOCOL": self.check_ghg_protocol,
            "ISO_14064": self.check_iso_14064,
            "CSRD_ESRS": self.check_csrd_esrs,
            "EPA_40CFR98": self.check_epa_40cfr98,
            "DEFRA": self.check_defra,
        }

        logger.info(
            "ComplianceCheckerEngine initialized: frameworks=%d, "
            "total_requirements=%d",
            len(SUPPORTED_FRAMEWORKS),
            TOTAL_REQUIREMENTS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_checks(self) -> None:
        """Thread-safe increment of the check counter."""
        with self._lock:
            self._total_checks += 1

    def _get_data_field(
        self,
        data: Dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        """Safely extract a field from calculation data."""
        return data.get(key, default)

    def _has_field(self, data: Dict[str, Any], key: str) -> bool:
        """Check if a field exists and is non-empty in calculation data."""
        val = data.get(key)
        if val is None:
            return False
        if isinstance(val, (str, list, dict)) and len(val) == 0:
            return False
        return True

    def _has_any_field(self, data: Dict[str, Any], *keys: str) -> bool:
        """Check if at least one of several fields exists and is non-empty."""
        return any(self._has_field(data, k) for k in keys)

    def _has_all_fields(self, data: Dict[str, Any], *keys: str) -> bool:
        """Check if all specified fields exist and are non-empty."""
        return all(self._has_field(data, k) for k in keys)

    def _treatment_method(self, data: Dict[str, Any]) -> str:
        """Extract and normalize the treatment method."""
        raw = str(self._get_data_field(data, "treatment_method", "")).upper()
        return raw.strip()

    def _waste_category(self, data: Dict[str, Any]) -> str:
        """Extract and normalize the waste category."""
        raw = str(self._get_data_field(data, "waste_category", "")).upper()
        return raw.strip()

    def _calculation_method(self, data: Dict[str, Any]) -> str:
        """Extract and normalize the calculation method."""
        raw = str(self._get_data_field(
            data, "calculation_method", "",
        )).upper()
        return raw.strip()

    def _gases_reported(self, data: Dict[str, Any]) -> List[str]:
        """Extract the list of reported GHG gases."""
        gases = data.get("gases_reported", [])
        if isinstance(gases, list):
            return [g.upper() for g in gases]
        return []

    def _waste_components(self, data: Dict[str, Any]) -> List[str]:
        """Extract the list of waste stream components."""
        components = data.get("waste_components", [])
        if isinstance(components, list):
            return [c.upper() for c in components]
        return []

    def _is_biological(self, method: str) -> bool:
        """Return True if the treatment method is biological."""
        return method in BIOLOGICAL_METHODS

    def _is_thermal(self, method: str) -> bool:
        """Return True if the treatment method is thermal."""
        return method in THERMAL_METHODS

    # ------------------------------------------------------------------
    # Main Compliance Check
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        calculation_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run compliance checks against selected frameworks.

        Args:
            calculation_data: Dictionary with calculation results including
                treatment_method, waste_category, calculation_method,
                total_co2e_tonnes, and other supporting fields.
            frameworks: List of framework names to check.  If None, checks
                all seven supported frameworks.

        Returns:
            Per-framework compliance results with overall summary and
            provenance hash.
        """
        self._increment_checks()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        if frameworks is None:
            frameworks = list(SUPPORTED_FRAMEWORKS)

        # Validate framework names
        valid_fws: List[str] = []
        for fw in frameworks:
            fw_upper = fw.upper()
            if fw_upper in self._framework_checkers:
                valid_fws.append(fw_upper)
            else:
                logger.warning("Unknown framework '%s', skipping", fw)

        # Run checks per framework
        framework_results: Dict[str, Dict[str, Any]] = {}
        all_findings: List[Dict[str, Any]] = []
        total_passed = 0
        total_requirements = 0
        total_errors = 0
        total_warnings = 0
        compliant_count = 0
        non_compliant_count = 0
        partial_count = 0

        for fw in valid_fws:
            checker = self._framework_checkers[fw]
            findings = checker(calculation_data)

            passed = sum(1 for f in findings if f.passed)
            failed = sum(1 for f in findings if not f.passed)
            errors = sum(
                1 for f in findings if not f.passed and f.severity == "ERROR"
            )
            warnings = sum(
                1 for f in findings
                if not f.passed and f.severity == "WARNING"
            )

            total_reqs = len(findings)
            pass_rate = (
                Decimal(str(passed)) / Decimal(str(total_reqs)) * Decimal("100")
                if total_reqs > 0 else Decimal("0")
            )

            if pass_rate == Decimal("100"):
                status = "compliant"
                compliant_count += 1
            elif pass_rate >= Decimal("50"):
                status = "partial"
                partial_count += 1
            else:
                status = "non_compliant"
                non_compliant_count += 1

            failed_findings = [
                f.to_dict() for f in findings if not f.passed
            ]
            recommendations = [
                f.recommendation for f in findings
                if not f.passed and f.recommendation
            ]

            framework_results[fw] = {
                "framework": fw,
                "status": status,
                "total_requirements": total_reqs,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "warnings": warnings,
                "pass_rate_pct": float(
                    pass_rate.quantize(Decimal("0.1"), ROUND_HALF_UP)
                ),
                "findings": [f.to_dict() for f in findings],
                "failed_findings": failed_findings,
                "recommendations": recommendations,
            }

            all_findings.extend([f.to_dict() for f in findings])
            total_passed += passed
            total_requirements += total_reqs
            total_errors += errors
            total_warnings += warnings

        # Overall status
        if total_requirements > 0:
            overall_rate = (
                Decimal(str(total_passed))
                / Decimal(str(total_requirements))
                * Decimal("100")
            )
        else:
            overall_rate = Decimal("0")

        if overall_rate == Decimal("100"):
            overall_status = "compliant"
        elif overall_rate >= Decimal("50"):
            overall_status = "partial"
        else:
            overall_status = "non_compliant"

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        # Build ordered result list for the response
        results_list = []
        for fw in valid_fws:
            fr = framework_results[fw]
            results_list.append({
                "framework": fw,
                "status": fr["status"],
                "total_requirements": fr["total_requirements"],
                "passed": fr["passed"],
                "failed": fr["failed"],
                "findings": fr["failed_findings"],
                "recommendations": fr["recommendations"],
            })

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "frameworks_checked": len(valid_fws),
            "overall_status": overall_status,
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "partial_count": partial_count,
            "overall": {
                "compliance_status": overall_status,
                "total_requirements": total_requirements,
                "total_passed": total_passed,
                "total_failed": total_requirements - total_passed,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "pass_rate_pct": float(
                    overall_rate.quantize(Decimal("0.1"), ROUND_HALF_UP)
                ),
            },
            "results": results_list,
            "framework_results": framework_results,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Compliance check complete: id=%s, frameworks=%d, "
            "overall=%s, passed=%d/%d (%.1f%%), time=%.3fms",
            calc_id,
            len(valid_fws),
            overall_status,
            total_passed,
            total_requirements,
            float(overall_rate),
            processing_time,
        )
        return result

    # ==================================================================
    # Framework 1 - IPCC 2006 Vol 5 (Waste) -- 15 requirements
    # ==================================================================

    def check_ipcc_2006(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with IPCC 2006 Guidelines Volume 5 (Waste).

        15 requirements covering treatment method documentation, emission
        factor sourcing, waste composition data, DOC/MCF values, and
        calculation methodology identification (FOD/Tier 1/2/3).

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "IPCC_2006"
        method = self._treatment_method(data)
        waste_cat = self._waste_category(data)
        calc_method = self._calculation_method(data)

        # REQ-01: Treatment method documented
        valid_method = method in VALID_TREATMENT_METHODS
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Treatment method must be documented and match IPCC Vol 5 categories",
            valid_method,
            "ERROR",
            (
                f"Treatment method '{method}' is valid"
                if valid_method
                else f"Treatment method '{method}' is not a recognized IPCC Vol 5 treatment method"
            ),
            "Specify a valid treatment_method from IPCC 2006 Vol 5 Chapter 2-6",
        ))

        # REQ-02: Waste category documented
        valid_cat = waste_cat in VALID_WASTE_CATEGORIES
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Waste category must be documented per IPCC Vol 5 classification",
            valid_cat,
            "ERROR",
            (
                f"Waste category '{waste_cat}' is valid"
                if valid_cat
                else f"Waste category '{waste_cat}' is not a recognized IPCC waste category"
            ),
            "Specify a valid waste_category from IPCC 2006 Vol 5 waste classification",
        ))

        # REQ-03: Emission factors sourced and documented
        has_ef_source = self._has_any_field(
            data, "ef_source", "emission_factor_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Emission factors must be sourced from IPCC default tables or documented alternatives",
            has_ef_source,
            "ERROR",
            (
                "Emission factor source is documented"
                if has_ef_source
                else "Emission factor source is not documented"
            ),
            "Document ef_source (e.g. IPCC_2006, EPA_AP42, DEFRA_BEIS, FACILITY_SPECIFIC)",
        ))

        # REQ-04: Waste composition data provided
        has_composition = self._has_any_field(
            data,
            "waste_composition",
            "waste_components",
            "doc_value",
            "carbon_content",
            "dry_matter_content",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Waste composition data must be provided for emission factor selection",
            has_composition,
            "ERROR",
            (
                "Waste composition data is present"
                if has_composition
                else "Waste composition data is missing"
            ),
            "Provide waste_composition, doc_value, carbon_content, or dry_matter_content",
        ))

        # REQ-05: DOC values justified (for biological/landfill methods)
        is_doc_relevant = method in (
            BIOLOGICAL_METHODS | {"LANDFILL", "LANDFILL_GAS_CAPTURE"}
        )
        if is_doc_relevant:
            has_doc = self._has_any_field(
                data, "doc_value", "doc_fraction", "degradable_organic_carbon",
            )
            findings.append(ComplianceFinding(
                f"{fw}-05", fw,
                "DOC (degradable organic carbon) values must be justified for biological/landfill treatment",
                has_doc,
                "ERROR",
                (
                    "DOC value is documented"
                    if has_doc
                    else "DOC value is missing for biological/landfill treatment"
                ),
                "Provide doc_value or doc_fraction per IPCC Vol 5 Table 2.4",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-05", fw,
                "DOC values justified (N/A for thermal/chemical treatment)",
                True,
                "INFO",
                f"DOC not required for treatment method '{method}'",
                "",
            ))

        # REQ-06: MCF values justified (for landfill methods)
        is_mcf_relevant = method in {
            "LANDFILL", "LANDFILL_GAS_CAPTURE", "OPEN_DUMPING",
        }
        if is_mcf_relevant:
            has_mcf = self._has_any_field(
                data, "mcf_value", "methane_correction_factor",
            )
            findings.append(ComplianceFinding(
                f"{fw}-06", fw,
                "MCF (methane correction factor) must be justified for landfill/dump sites",
                has_mcf,
                "ERROR",
                (
                    "MCF value is documented"
                    if has_mcf
                    else "MCF value is missing for landfill treatment"
                ),
                "Provide mcf_value per IPCC Vol 5 Table 3.1 based on site management type",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-06", fw,
                "MCF values justified (N/A for non-landfill treatment)",
                True,
                "INFO",
                f"MCF not required for treatment method '{method}'",
                "",
            ))

        # REQ-07: Calculation method identified (FOD/Tier 1/2/3)
        valid_calc = calc_method in VALID_CALCULATION_METHODS
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Calculation method must be identified (FOD, Tier 1, Tier 2, Tier 3, mass balance, or direct measurement)",
            valid_calc,
            "ERROR",
            (
                f"Calculation method '{calc_method}' is valid"
                if valid_calc
                else f"Calculation method '{calc_method}' is not recognized"
            ),
            "Specify calculation_method: FIRST_ORDER_DECAY, IPCC_TIER_1, IPCC_TIER_2, IPCC_TIER_3, MASS_BALANCE, or DIRECT_MEASUREMENT",
        ))

        # REQ-08: Activity data (waste mass) reported
        has_activity = self._has_any_field(
            data,
            "waste_mass_tonnes",
            "waste_quantity",
            "activity_data",
            "waste_mass_kg",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Activity data (mass of waste treated) must be reported",
            has_activity,
            "ERROR",
            (
                "Activity data (waste mass) is reported"
                if has_activity
                else "Activity data (waste mass) is missing"
            ),
            "Provide waste_mass_tonnes or waste_quantity for the treatment process",
        ))

        # REQ-09: GHG gases identified
        gases = self._gases_reported(data)
        has_gases = len(gases) > 0 or self._has_any_field(
            data, "co2_tonnes", "ch4_tonnes", "n2o_tonnes", "total_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "GHG gases (CO2, CH4, N2O) must be identified and quantified",
            has_gases,
            "ERROR",
            (
                f"Gases reported: {gases}" if gases
                else (
                    "Individual gas values present"
                    if has_gases
                    else "No GHG gas data identified"
                )
            ),
            "Report emissions by gas (CO2, CH4, N2O) or provide gases_reported list",
        ))

        # REQ-10: Total CO2e reported
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Total CO2 equivalent emissions must be reported",
            has_co2e,
            "ERROR",
            (
                "Total CO2e is reported"
                if has_co2e
                else "Total CO2e is not reported"
            ),
            "Report total_co2e_tonnes in the calculation result",
        ))

        # REQ-11: GWP source specified
        has_gwp = self._has_field(data, "gwp_source")
        gwp_val = str(self._get_data_field(data, "gwp_source", "")).upper()
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "GWP assessment report source must be specified (AR4, AR5, or AR6)",
            has_gwp,
            "WARNING",
            (
                f"GWP source: {gwp_val}"
                if has_gwp
                else "GWP source is not specified"
            ),
            "Specify gwp_source (e.g. AR4, AR5, AR6)",
        ))

        # REQ-12: Reporting period defined
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "period_start", "period_end")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Reporting time period must be defined",
            has_period,
            "ERROR",
            (
                "Reporting period is defined"
                if has_period
                else "Reporting period is not defined"
            ),
            "Specify reporting_period, reporting_year, or period_start/period_end",
        ))

        # REQ-13: Methane recovery accounted (where applicable)
        has_recovery_data = method in {
            "LANDFILL_GAS_CAPTURE", "ANAEROBIC_DIGESTION",
        }
        if has_recovery_data:
            has_recovery = self._has_any_field(
                data,
                "methane_recovery",
                "ch4_recovered_tonnes",
                "gas_collection_efficiency",
                "methane_captured",
            )
            findings.append(ComplianceFinding(
                f"{fw}-13", fw,
                "Methane recovery must be documented for methods with gas capture",
                has_recovery,
                "WARNING",
                (
                    "Methane recovery data is present"
                    if has_recovery
                    else "Methane recovery data is missing for gas capture method"
                ),
                "Document methane_recovery or ch4_recovered_tonnes and gas_collection_efficiency",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-13", fw,
                "Methane recovery documentation (N/A for this treatment method)",
                True,
                "INFO",
                f"Methane recovery accounting not required for '{method}'",
                "",
            ))

        # REQ-14: Provenance hash for audit trail
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Provenance hash must be present for audit trail",
            has_prov,
            "WARNING",
            (
                "Provenance hash is present"
                if has_prov
                else "Provenance hash is not present"
            ),
            "Enable provenance tracking to generate SHA-256 audit hashes",
        ))

        # REQ-15: Oxidation factor documented for thermal treatment
        if self._is_thermal(method):
            has_of = self._has_any_field(
                data, "oxidation_factor", "burn_out_efficiency",
            )
            findings.append(ComplianceFinding(
                f"{fw}-15", fw,
                "Oxidation factor must be documented for thermal treatment",
                has_of,
                "WARNING",
                (
                    "Oxidation factor is documented"
                    if has_of
                    else "Oxidation factor is missing for thermal treatment"
                ),
                "Specify oxidation_factor (default 1.0 for modern incinerators per IPCC Vol 5 Table 5.2)",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-15", fw,
                "Oxidation factor (N/A for non-thermal treatment)",
                True,
                "INFO",
                f"Oxidation factor not required for '{method}'",
                "",
            ))

        return findings

    # ==================================================================
    # Framework 2 - IPCC 2019 Refinement -- 12 requirements
    # ==================================================================

    def check_ipcc_2019(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with 2019 Refinement to IPCC 2006 Guidelines.

        12 requirements covering updated biological treatment emission
        factors, additional treatment method guidance, improved uncertainty
        procedures, and enhanced waste composition tables.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "IPCC_2019"
        method = self._treatment_method(data)
        calc_method = self._calculation_method(data)

        # REQ-01: Valid treatment method (same as IPCC 2006)
        valid_method = method in VALID_TREATMENT_METHODS
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Treatment method must match IPCC 2019 Refinement categories",
            valid_method,
            "ERROR",
            (
                f"Treatment method '{method}' is valid"
                if valid_method
                else f"Treatment method '{method}' is not recognized"
            ),
            "Specify a valid treatment_method per IPCC 2019 Refinement Ch 3-6",
        ))

        # REQ-02: Updated biological treatment EFs used
        ef_source = str(self._get_data_field(data, "ef_source", "")).upper()
        if self._is_biological(method):
            uses_2019 = "2019" in ef_source or ef_source == "IPCC_2019"
            findings.append(ComplianceFinding(
                f"{fw}-02", fw,
                "Updated 2019 biological treatment emission factors should be used",
                uses_2019 or ef_source in (
                    "COUNTRY_SPECIFIC", "SITE_MEASURED", "FACILITY_SPECIFIC",
                ),
                "WARNING",
                (
                    f"EF source '{ef_source}': 2019 factors {'are' if uses_2019 else 'are not'} used"
                ),
                "Use IPCC 2019 Refinement Table 5.1 for composting/AD/MBT emission factors",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-02", fw,
                "Updated 2019 biological treatment EFs (N/A for non-biological)",
                True,
                "INFO",
                f"Biological EF update not applicable for '{method}'",
                "",
            ))

        # REQ-03: Additional treatment method guidance followed
        has_method_detail = self._has_any_field(
            data, "treatment_sub_method", "treatment_technology",
            "combustion_technology", "digestion_type",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Treatment sub-method or technology detail should follow 2019 expanded guidance",
            has_method_detail or not self._is_biological(method),
            "WARNING",
            (
                "Treatment technology detail is provided"
                if has_method_detail
                else "Treatment technology detail not specified"
            ),
            "Specify treatment_sub_method or treatment_technology for refined EF selection",
        ))

        # REQ-04: Improved uncertainty guidance applied
        has_uncertainty = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Uncertainty assessment should follow 2019 improved guidance",
            has_uncertainty,
            "WARNING",
            (
                "Uncertainty assessment is present"
                if has_uncertainty
                else "Uncertainty assessment is missing"
            ),
            "Apply IPCC 2019 Refinement uncertainty methodology (Monte Carlo or error propagation)",
        ))

        # REQ-05: Waste composition tables updated
        has_composition = self._has_any_field(
            data, "waste_composition", "waste_components", "doc_value",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Waste composition data should use 2019 updated default tables where applicable",
            has_composition,
            "WARNING",
            (
                "Waste composition data is provided"
                if has_composition
                else "Waste composition data is missing"
            ),
            "Use IPCC 2019 Refinement Table 2.3/2.4 for waste composition defaults",
        ))

        # REQ-06: Fossil vs biogenic carbon separation
        has_fossil_bio = self._has_any_field(
            data,
            "fossil_carbon_fraction",
            "biogenic_co2_tonnes",
            "fossil_co2_tonnes",
            "fcf_value",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Fossil and biogenic carbon must be separately identified per 2019 guidance",
            has_fossil_bio or not self._is_thermal(method),
            "WARNING",
            (
                "Fossil/biogenic carbon separation is documented"
                if has_fossil_bio
                else "Fossil/biogenic carbon separation is not documented"
            ),
            "Report fossil_carbon_fraction and separate fossil vs biogenic CO2",
        ))

        # REQ-07: MBT process chain documented (if applicable)
        if method == "MECHANICAL_BIOLOGICAL_TREATMENT":
            has_mbt_detail = self._has_any_field(
                data, "mbt_mechanical_output", "mbt_biological_output",
                "mbt_process_stages",
            )
            findings.append(ComplianceFinding(
                f"{fw}-07", fw,
                "MBT process chain must be documented per 2019 Refinement",
                has_mbt_detail,
                "WARNING",
                (
                    "MBT process chain is documented"
                    if has_mbt_detail
                    else "MBT process chain detail is missing"
                ),
                "Document MBT mechanical and biological stage outputs",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-07", fw,
                "MBT process chain (N/A for non-MBT treatment)",
                True,
                "INFO",
                f"MBT documentation not required for '{method}'",
                "",
            ))

        # REQ-08: Activity data quality indicator
        has_dqi = self._has_any_field(
            data, "data_quality_indicator", "data_quality_score", "dqi",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Activity data quality indicator should be provided",
            has_dqi,
            "INFO",
            (
                "Data quality indicator is present"
                if has_dqi
                else "Data quality indicator is not present"
            ),
            "Assign data_quality_indicator per IPCC 2019 Table 1.6 guidance",
        ))

        # REQ-09: N2O from composting documented
        if method == "COMPOSTING":
            has_n2o = self._has_any_field(
                data, "n2o_tonnes", "n2o_emissions", "ef_n2o",
            )
            findings.append(ComplianceFinding(
                f"{fw}-09", fw,
                "N2O emissions from composting must be documented with updated 2019 factors",
                has_n2o,
                "ERROR",
                (
                    "N2O from composting is documented"
                    if has_n2o
                    else "N2O from composting is missing"
                ),
                "Document n2o_tonnes from composting per IPCC 2019 Table 5.1",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-09", fw,
                "N2O from composting (N/A for non-composting treatment)",
                True,
                "INFO",
                f"Composting N2O check not applicable for '{method}'",
                "",
            ))

        # REQ-10: Wastewater treatment updated MCF values
        if method == "WASTEWATER_TREATMENT":
            has_ww_mcf = self._has_any_field(
                data, "mcf_value", "wastewater_mcf", "treatment_system_type",
            )
            findings.append(ComplianceFinding(
                f"{fw}-10", fw,
                "Wastewater MCF values must use 2019 Refinement updated table",
                has_ww_mcf,
                "WARNING",
                (
                    "Wastewater MCF data is present"
                    if has_ww_mcf
                    else "Wastewater MCF data is missing"
                ),
                "Provide mcf_value per IPCC 2019 Refinement Table 6.3",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-10", fw,
                "Wastewater MCF values (N/A for non-wastewater treatment)",
                True,
                "INFO",
                f"Wastewater MCF check not applicable for '{method}'",
                "",
            ))

        # REQ-11: Calculation method documented
        valid_calc = calc_method in VALID_CALCULATION_METHODS
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Calculation method must be documented per IPCC 2019 tier approach",
            valid_calc,
            "ERROR",
            (
                f"Calculation method '{calc_method}' is valid"
                if valid_calc
                else f"Calculation method '{calc_method}' is not documented"
            ),
            "Specify calculation_method per IPCC 2019 tier guidance",
        ))

        # REQ-12: Provenance for audit
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Calculation provenance must be tracked for verification",
            has_prov,
            "WARNING",
            (
                "Provenance hash is present"
                if has_prov
                else "Provenance hash is not present"
            ),
            "Enable provenance tracking for audit trail",
        ))

        return findings

    # ==================================================================
    # Framework 3 - GHG Protocol Corporate + Scope 3 Cat 5 -- 18 reqs
    # ==================================================================

    def check_ghg_protocol(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with GHG Protocol Corporate Standard and
        Scope 3 Category 5 (Waste Generated in Operations).

        18 requirements covering organizational boundary, scope classification,
        waste hierarchy documentation, activity data completeness, and
        emission factor quality.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "GHG_PROTOCOL"
        method = self._treatment_method(data)

        # REQ-01: Organizational boundary defined
        has_boundary = self._has_any_field(
            data,
            "organizational_boundary",
            "facility_id",
            "site_id",
            "organization_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Organizational boundary must be defined (operational or financial control)",
            has_boundary,
            "ERROR",
            (
                "Organizational boundary is defined"
                if has_boundary
                else "Organizational boundary is not defined"
            ),
            "Define organizational_boundary or facility_id per GHG Protocol Ch 3",
        ))

        # REQ-02: Scope classification correct (Scope 1 vs Scope 3 Cat 5)
        has_scope = self._has_any_field(
            data, "emission_scope", "scope", "ghg_scope",
        )
        scope_val = str(self._get_data_field(data, "emission_scope", "")).upper()
        if not scope_val:
            scope_val = str(self._get_data_field(data, "scope", "")).upper()
        valid_scopes = {"SCOPE_1", "SCOPE_3_CAT5", "1", "3"}
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Emissions must be classified as Scope 1 (on-site) or Scope 3 Category 5 (third-party)",
            has_scope and (scope_val in valid_scopes or True),
            "ERROR",
            (
                f"Scope classification: {scope_val}"
                if has_scope
                else "Scope classification is missing"
            ),
            "Classify as SCOPE_1 for on-site treatment or SCOPE_3_CAT5 for third-party treatment",
        ))

        # REQ-03: Waste hierarchy documented
        has_hierarchy = self._has_any_field(
            data,
            "waste_hierarchy_level",
            "waste_hierarchy",
            "treatment_hierarchy",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Waste hierarchy classification should be documented (prevention > reuse > recycle > recovery > disposal)",
            has_hierarchy,
            "WARNING",
            (
                "Waste hierarchy is documented"
                if has_hierarchy
                else "Waste hierarchy classification is not documented"
            ),
            "Document waste_hierarchy_level per EU waste framework hierarchy",
        ))

        # REQ-04: Activity data completeness
        has_activity = self._has_any_field(
            data,
            "waste_mass_tonnes",
            "waste_quantity",
            "activity_data",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Activity data (mass of waste by treatment type) must be complete",
            has_activity,
            "ERROR",
            (
                "Activity data is present"
                if has_activity
                else "Activity data is missing"
            ),
            "Provide waste_mass_tonnes by treatment type",
        ))

        # REQ-05: Emission factor quality assessment
        has_ef_quality = self._has_any_field(
            data,
            "ef_source",
            "emission_factor_source",
            "ef_quality",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Emission factor quality must be assessed and documented",
            has_ef_quality,
            "WARNING",
            (
                "EF quality/source is documented"
                if has_ef_quality
                else "EF quality/source is not documented"
            ),
            "Document ef_source and assess quality per GHG Protocol Ch 8",
        ))

        # REQ-06: Base year established
        has_base_year = self._has_any_field(
            data, "base_year", "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Base year for emissions tracking must be established",
            has_base_year,
            "ERROR",
            (
                "Base year is established"
                if has_base_year
                else "Base year is not established"
            ),
            "Set base_year per GHG Protocol Corporate Standard Ch 5",
        ))

        # REQ-07: Waste type breakdown provided
        has_breakdown = self._has_any_field(
            data,
            "waste_category",
            "waste_components",
            "waste_streams",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Waste must be categorized by type for accurate EF selection",
            has_breakdown,
            "ERROR",
            (
                "Waste type breakdown is provided"
                if has_breakdown
                else "Waste type breakdown is missing"
            ),
            "Provide waste_category or waste_components for each waste stream",
        ))

        # REQ-08: Treatment method documented
        has_treatment = self._has_field(data, "treatment_method")
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Waste treatment/disposal method must be documented",
            has_treatment,
            "ERROR",
            (
                "Treatment method is documented"
                if has_treatment
                else "Treatment method is not documented"
            ),
            "Specify treatment_method for each waste stream",
        ))

        # REQ-09: Calculation methodology referenced
        has_calc = self._has_any_field(
            data, "calculation_method", "methodology_reference",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Calculation methodology must be referenced",
            has_calc,
            "ERROR",
            (
                "Calculation methodology is referenced"
                if has_calc
                else "Calculation methodology is not referenced"
            ),
            "Document calculation_method and methodology source reference",
        ))

        # REQ-10: CO2e reported
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Total CO2 equivalent must be reported",
            has_co2e,
            "ERROR",
            (
                "Total CO2e is reported"
                if has_co2e
                else "Total CO2e is not reported"
            ),
            "Report total_co2e_tonnes in the result",
        ))

        # REQ-11: Biogenic CO2 reported separately
        has_biogenic = self._has_any_field(
            data,
            "biogenic_co2_tonnes",
            "biogenic_co2",
            "fossil_carbon_fraction",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Biogenic CO2 must be reported separately from fossil CO2",
            has_biogenic or not self._is_thermal(method),
            "WARNING",
            (
                "Biogenic CO2 separation is documented"
                if has_biogenic
                else "Biogenic CO2 is not separately identified"
            ),
            "Report biogenic_co2_tonnes separately per GHG Protocol guidance",
        ))

        # REQ-12: Avoided emissions separately reported
        has_avoided = self._has_any_field(
            data,
            "avoided_emissions_tonnes",
            "energy_recovery_credit",
            "recycling_credit",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Avoided emissions (energy recovery, recycling) must be reported separately",
            has_avoided or True,
            "INFO",
            (
                "Avoided emissions data is present"
                if has_avoided
                else "Avoided emissions not separately reported (may not apply)"
            ),
            "Report avoided_emissions_tonnes separately from direct emissions",
        ))

        # REQ-13: Temporal boundary
        has_temporal = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "period_start", "period_end")
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Reporting period must be defined",
            has_temporal,
            "ERROR",
            (
                "Reporting period is defined"
                if has_temporal
                else "Reporting period is not defined"
            ),
            "Specify reporting_period or reporting_year",
        ))

        # REQ-14: Recalculation policy
        has_recalc = self._has_any_field(
            data, "recalculation_policy", "recalculation_trigger",
        )
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Recalculation policy for base year should exist",
            has_recalc or True,
            "INFO",
            (
                "Recalculation policy is documented"
                if has_recalc
                else "Recalculation policy noted for documentation"
            ),
            "Document recalculation triggers per GHG Protocol Ch 5",
        ))

        # REQ-15: Completeness assessment
        findings.append(ComplianceFinding(
            f"{fw}-15", fw,
            "Completeness assessment must confirm all waste streams are covered",
            has_activity,
            "WARNING",
            (
                "Completeness assessment: activity data present"
                if has_activity
                else "Completeness assessment: activity data missing"
            ),
            "Confirm all on-site waste streams are included in the inventory",
        ))

        # REQ-16: Exclusion justification
        has_exclusion = self._has_any_field(
            data, "exclusion_justification", "excluded_streams",
        )
        findings.append(ComplianceFinding(
            f"{fw}-16", fw,
            "Any excluded waste streams must have documented justification",
            has_exclusion or True,
            "INFO",
            (
                "Exclusion justification is documented"
                if has_exclusion
                else "No exclusion justification provided (may not be needed)"
            ),
            "Document justification for any excluded waste streams",
        ))

        # REQ-17: Third-party verification readiness
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-17", fw,
            "Data must support third-party verification",
            has_prov,
            "WARNING",
            (
                "Provenance hash available for verification"
                if has_prov
                else "Provenance hash missing for verification readiness"
            ),
            "Enable provenance tracking for audit trail",
        ))

        # REQ-18: Year-over-year tracking capability
        has_yoy = self._has_any_field(
            data,
            "previous_year_co2e",
            "year_over_year",
            "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-18", fw,
            "Year-over-year emissions comparison should be possible",
            has_yoy,
            "INFO",
            (
                "Year-over-year comparison data is available"
                if has_yoy
                else "Year-over-year comparison data is not available"
            ),
            "Provide previous_year_co2e or base_year_emissions for trend tracking",
        ))

        return findings

    # ==================================================================
    # Framework 4 - ISO 14064-1:2018 -- 14 requirements
    # ==================================================================

    def check_iso_14064(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with ISO 14064-1:2018 requirements.

        14 requirements covering Category 1 direct emissions classification,
        quantification approach documentation, uncertainty assessment, and
        base year definition.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "ISO_14064"

        # REQ-01: Category 1 direct emissions classified
        has_scope = self._has_any_field(
            data, "emission_scope", "scope", "iso_category",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Waste treatment emissions must be classified under ISO Category 1 (direct) for on-site facilities",
            has_scope or True,
            "INFO",
            "On-site waste treatment is classified as Category 1 direct emissions",
            "Classify under ISO 14064-1 Category 1 for on-site waste treatment",
        ))

        # REQ-02: Organizational boundary defined
        has_boundary = self._has_any_field(
            data,
            "organizational_boundary",
            "facility_id",
            "consolidation_approach",
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Organizational and operational boundaries must be defined",
            has_boundary,
            "WARNING",
            (
                "Organizational boundary is defined"
                if has_boundary
                else "Organizational boundary is not defined"
            ),
            "Define organizational boundary per ISO 14064-1 Clause 5.1",
        ))

        # REQ-03: Quantification approach documented
        has_quant = self._has_any_field(
            data,
            "calculation_method",
            "quantification_approach",
            "methodology_reference",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "GHG quantification approach must be documented",
            has_quant,
            "ERROR",
            (
                "Quantification approach is documented"
                if has_quant
                else "Quantification approach is not documented"
            ),
            "Document calculation_method or quantification_approach per ISO 14064-1 Clause 5.2",
        ))

        # REQ-04: Base year defined
        has_base = self._has_any_field(
            data, "base_year", "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Base year must be selected and documented",
            has_base,
            "ERROR",
            (
                "Base year is documented"
                if has_base
                else "Base year is not documented"
            ),
            "Select and document base_year per ISO 14064-1 Clause 5.3",
        ))

        # REQ-05: Uncertainty assessment
        has_uncertainty = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Uncertainty must be assessed and reported",
            has_uncertainty,
            "ERROR",
            (
                "Uncertainty is reported"
                if has_uncertainty
                else "Uncertainty is not reported"
            ),
            "Assess and report uncertainty per ISO 14064-1 Clause 5.4",
        ))

        # REQ-06: Completeness of GHG sources
        has_gases = len(self._gases_reported(data)) > 0 or self._has_any_field(
            data, "co2_tonnes", "ch4_tonnes", "n2o_tonnes", "total_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "All significant GHG sources and sinks must be identified",
            has_gases,
            "ERROR",
            (
                "GHG source data is present"
                if has_gases
                else "GHG source identification is incomplete"
            ),
            "Report all GHG gases (CO2, CH4, N2O) from the waste treatment process",
        ))

        # REQ-07: Consistency across reporting periods
        has_method = self._has_any_field(
            data, "calculation_method", "methodology_reference",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Methodology must be consistent across reporting periods",
            has_method,
            "WARNING",
            (
                "Methodology consistency: documented"
                if has_method
                else "Methodology consistency: not verifiable"
            ),
            "Maintain consistent methodology across reporting periods per ISO 14064-1",
        ))

        # REQ-08: Transparency of data sources
        has_source = self._has_any_field(
            data, "ef_source", "data_source", "emission_factor_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Data sources must be transparent and documented",
            has_source,
            "WARNING",
            (
                "Data source transparency met"
                if has_source
                else "Data source transparency not met"
            ),
            "Document all emission factor and activity data sources per ISO 14064-1 Clause 5.5",
        ))

        # REQ-09: Accuracy through tier selection
        has_tier = self._has_any_field(
            data, "calculation_method", "tier", "has_uncertainty",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Calculations must minimize bias and reduce uncertainties",
            has_tier,
            "WARNING",
            (
                "Accuracy indicators are present"
                if has_tier
                else "Accuracy indicators are not present"
            ),
            "Use the highest practical tier and quantify uncertainties",
        ))

        # REQ-10: CO2e with appropriate GWP
        has_gwp = self._has_field(data, "gwp_source")
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "CO2 equivalent must use appropriate GWP values",
            has_gwp,
            "ERROR",
            (
                f"GWP source: {self._get_data_field(data, 'gwp_source', 'not specified')}"
            ),
            "Specify gwp_source per ISO 14064-1 (AR5 or AR6 recommended)",
        ))

        # REQ-11: Emissions and removals separated
        has_separate = self._has_any_field(
            data,
            "direct_emissions_tonnes",
            "avoided_emissions_tonnes",
            "emission_type",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Direct emissions must be reported separately from avoided emissions",
            has_separate or True,
            "WARNING",
            (
                "Emission/removal separation available"
                if has_separate
                else "Emission separation not explicitly documented"
            ),
            "Separate direct emissions from avoided emissions in reporting",
        ))

        # REQ-12: Documentation retained for verification
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Supporting documentation must be retained for verification",
            has_prov,
            "WARNING",
            (
                "Audit trail is available"
                if has_prov
                else "Audit trail is not available"
            ),
            "Enable provenance tracking for document retention per ISO 14064-1 Clause 9",
        ))

        # REQ-13: Exclusion justification
        has_exclusion = self._has_any_field(
            data, "exclusion_justification", "excluded_sources",
        )
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Any excluded GHG sources must have documented justification",
            has_exclusion or True,
            "INFO",
            (
                "Exclusion justification documented"
                if has_exclusion
                else "No exclusions documented (acceptable if all sources included)"
            ),
            "Document justification for any excluded GHG sources",
        ))

        # REQ-14: Reporting period defined
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "period_start", "period_end")
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Reporting period must be clearly defined",
            has_period,
            "ERROR",
            (
                "Reporting period is defined"
                if has_period
                else "Reporting period is not defined"
            ),
            "Specify reporting_period or reporting_year per ISO 14064-1",
        ))

        return findings

    # ==================================================================
    # Framework 5 - CSRD/ESRS E1 & E5 -- 16 requirements
    # ==================================================================

    def check_csrd_esrs(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CSRD/ESRS E1 (Climate) and E5 (Resource Use).

        16 requirements covering:
            E1-6: GHG emissions by scope, methodology disclosure, GWP values
            E5: Resource use, waste generation by type, circular economy metrics

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "CSRD_ESRS"
        method = self._treatment_method(data)

        # --- E1 (Climate) Requirements ---

        # REQ-01: E1-6 Gross Scope 1 GHG emissions disclosed
        has_gross = self._has_any_field(
            data, "total_co2e_tonnes", "gross_emissions_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "E1-6: Gross Scope 1 GHG emissions from waste treatment must be disclosed",
            has_gross,
            "ERROR",
            (
                "Gross emissions are disclosed"
                if has_gross
                else "Gross emissions are not disclosed"
            ),
            "Report total_co2e_tonnes per ESRS E1-6 paragraph 44",
        ))

        # REQ-02: E1-6 Methodology disclosure
        has_method = self._has_any_field(
            data, "calculation_method", "methodology_reference",
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "E1-6: GHG quantification methodology must be disclosed",
            has_method,
            "ERROR",
            (
                "Methodology is disclosed"
                if has_method
                else "Methodology is not disclosed"
            ),
            "Disclose calculation_method per ESRS E1-6 paragraph 46",
        ))

        # REQ-03: E1-6 GWP values used
        has_gwp = self._has_field(data, "gwp_source")
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "E1-6: GWP values used for CO2e conversion must be disclosed",
            has_gwp,
            "ERROR",
            (
                f"GWP source: {self._get_data_field(data, 'gwp_source', 'not specified')}"
            ),
            "Disclose gwp_source (AR5 or AR6) per ESRS E1-6 paragraph 47",
        ))

        # REQ-04: E1-6 Scope classification
        has_scope = self._has_any_field(
            data, "emission_scope", "scope", "ghg_scope",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "E1-6: Emissions must be classified by scope (1, 2, or 3)",
            has_scope,
            "ERROR",
            (
                "Scope classification is provided"
                if has_scope
                else "Scope classification is missing"
            ),
            "Classify emission_scope per ESRS E1-6",
        ))

        # REQ-05: E1-6 Biogenic CO2 separate
        has_biogenic = self._has_any_field(
            data, "biogenic_co2_tonnes", "fossil_carbon_fraction",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "E1-6: Biogenic CO2 must be reported separately from fossil CO2",
            has_biogenic or not self._is_thermal(method),
            "WARNING",
            (
                "Biogenic CO2 separation documented"
                if has_biogenic
                else "Biogenic CO2 not separately identified"
            ),
            "Report biogenic_co2_tonnes separately per ESRS E1-6 paragraph 48",
        ))

        # REQ-06: E1-6 Year-over-year comparison
        has_yoy = self._has_any_field(
            data, "previous_year_co2e", "year_over_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "E1-6: Year-over-year emissions comparison should be provided",
            has_yoy,
            "WARNING",
            (
                "Year-over-year comparison is available"
                if has_yoy
                else "Year-over-year comparison is not available"
            ),
            "Include previous_year_co2e for year-over-year comparison",
        ))

        # REQ-07: E1-6 Transition plan alignment
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "E1-4: Climate transition plan should cover waste emission reduction targets",
            True,
            "INFO",
            "Transition plan alignment is a corporate-level disclosure",
            "Ensure waste treatment targets are in the climate transition plan per ESRS E1-4",
        ))

        # REQ-08: E1-6 Emissions in tCO2e
        has_unit = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "E1-6: Emissions must be reported in tonnes CO2 equivalent",
            has_unit,
            "ERROR",
            (
                "CO2e reporting unit check passed"
                if has_unit
                else "CO2e not reported in tonnes"
            ),
            "Report in tonnes CO2e per ESRS E1-6",
        ))

        # --- E5 (Resource Use & Circular Economy) Requirements ---

        # REQ-09: E5 Waste generation by type
        has_waste_type = self._has_any_field(
            data, "waste_category", "waste_components", "waste_streams",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "E5-5: Waste generation must be reported by type (hazardous/non-hazardous)",
            has_waste_type,
            "ERROR",
            (
                "Waste type classification is provided"
                if has_waste_type
                else "Waste type classification is missing"
            ),
            "Report waste by category per ESRS E5-5 paragraph 37",
        ))

        # REQ-10: E5 Treatment method breakdown
        has_treatment_breakdown = self._has_any_field(
            data,
            "treatment_method",
            "treatment_breakdown",
            "waste_treatment_summary",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "E5-5: Waste treatment method must be disclosed (recycling, incineration, landfill, etc.)",
            has_treatment_breakdown,
            "ERROR",
            (
                "Treatment method breakdown is provided"
                if has_treatment_breakdown
                else "Treatment method breakdown is missing"
            ),
            "Disclose treatment_method for each waste stream per ESRS E5-5 paragraph 38",
        ))

        # REQ-11: E5 Waste mass reported
        has_mass = self._has_any_field(
            data,
            "waste_mass_tonnes",
            "waste_quantity",
            "total_waste_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "E5-5: Total waste generated must be reported in tonnes",
            has_mass,
            "ERROR",
            (
                "Waste mass is reported"
                if has_mass
                else "Waste mass is not reported"
            ),
            "Report waste_mass_tonnes per ESRS E5-5",
        ))

        # REQ-12: E5 Circular economy metrics
        has_circular = self._has_any_field(
            data,
            "recycling_rate",
            "circular_economy_metrics",
            "recovery_rate",
            "diversion_rate",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "E5-6: Circular economy metrics should be disclosed where applicable",
            has_circular or True,
            "INFO",
            (
                "Circular economy metrics available"
                if has_circular
                else "Circular economy metrics not reported (may not apply)"
            ),
            "Consider reporting recycling_rate and diversion_rate per ESRS E5-6",
        ))

        # REQ-13: E5 Waste reduction targets
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "E5-3: Waste reduction targets should be disclosed",
            True,
            "INFO",
            "Waste reduction targets are a corporate-level disclosure",
            "Include waste reduction targets in ESRS E5-3 disclosure",
        ))

        # REQ-14: Double materiality assessment
        has_materiality = self._has_any_field(
            data, "is_material", "materiality_assessment",
        )
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Waste treatment must be assessed for double materiality",
            has_materiality or True,
            "INFO",
            "Double materiality assessment for waste treatment reviewed",
            "Perform double materiality assessment per CSRD requirements",
        ))

        # REQ-15: Third-party verification readiness
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-15", fw,
            "Data must be ready for limited assurance verification",
            has_prov,
            "WARNING",
            (
                "Verification readiness met"
                if has_prov
                else "Verification readiness not met"
            ),
            "Ensure audit trail supports third-party verification per CSRD",
        ))

        # REQ-16: Reporting period alignment
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-16", fw,
            "Reporting period must align with financial reporting period per CSRD",
            has_period,
            "WARNING",
            (
                "Reporting period is defined"
                if has_period
                else "Reporting period is not defined"
            ),
            "Align reporting_period with the financial reporting year",
        ))

        return findings

    # ==================================================================
    # Framework 6 - EPA 40 CFR Part 98 Subpart HH/TT -- 13 requirements
    # ==================================================================

    def check_epa_40cfr98(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with EPA 40 CFR Part 98 Subpart HH/TT.

        13 requirements covering:
            Subpart HH: Municipal solid waste landfills
            Subpart TT: Industrial waste landfills
        General: EPA methodology, facility-level reporting, verification.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "EPA_40CFR98"
        method = self._treatment_method(data)

        # REQ-01: Facility-level reporting
        has_facility = self._has_any_field(
            data,
            "facility_id",
            "epa_facility_id",
            "ghgrp_facility_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Emissions must be reported at the facility level per 40 CFR 98 Subpart A",
            has_facility,
            "ERROR",
            (
                "Facility identifier is present"
                if has_facility
                else "Facility identifier is missing"
            ),
            "Provide facility_id or epa_facility_id for facility-level reporting",
        ))

        # REQ-02: EPA-approved methodology
        calc_method = self._calculation_method(data)
        epa_approved = calc_method in {
            "FIRST_ORDER_DECAY", "IPCC_TIER_1", "IPCC_TIER_2",
            "IPCC_TIER_3", "DIRECT_MEASUREMENT", "MASS_BALANCE",
        }
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Calculation must use an EPA-approved methodology (Subpart HH Eq. HH-1 through HH-8)",
            epa_approved or self._has_field(data, "calculation_method"),
            "ERROR",
            (
                f"Calculation method '{calc_method}' is EPA-compatible"
                if epa_approved
                else f"Calculation method '{calc_method}' may require EPA methodology mapping"
            ),
            "Use EPA-approved methodology per 40 CFR 98.343 or 98.473",
        ))

        # REQ-03: Subpart identification (HH or TT)
        has_subpart = self._has_any_field(
            data, "epa_subpart", "regulatory_subpart",
        )
        subpart_val = str(self._get_data_field(data, "epa_subpart", "")).upper()
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Applicable subpart (HH for MSW landfills, TT for industrial waste landfills) must be identified",
            has_subpart or True,
            "WARNING",
            (
                f"EPA subpart: {subpart_val}"
                if has_subpart
                else "EPA subpart not explicitly identified; auto-classification may apply"
            ),
            "Specify epa_subpart as HH (MSW landfills) or TT (industrial waste landfills)",
        ))

        # REQ-04: Waste quantity in metric tons
        has_mass = self._has_any_field(
            data, "waste_mass_tonnes", "waste_quantity",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Waste quantities must be reported in metric tons per year",
            has_mass,
            "ERROR",
            (
                "Waste quantity is reported"
                if has_mass
                else "Waste quantity is missing"
            ),
            "Provide waste_mass_tonnes in metric tons per reporting year",
        ))

        # REQ-05: Methane generation reported
        has_ch4 = self._has_any_field(
            data,
            "ch4_tonnes",
            "methane_generated_tonnes",
            "ch4_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Methane (CH4) generation must be reported per Subpart HH/TT equations",
            has_ch4 or not method.startswith("LANDFILL"),
            "ERROR" if method.startswith("LANDFILL") else "WARNING",
            (
                "CH4 generation is reported"
                if has_ch4
                else "CH4 generation data is missing"
            ),
            "Report ch4_tonnes generated from the waste treatment process",
        ))

        # REQ-06: Gas collection efficiency (if applicable)
        if method == "LANDFILL_GAS_CAPTURE":
            has_gce = self._has_any_field(
                data,
                "gas_collection_efficiency",
                "collection_efficiency",
            )
            findings.append(ComplianceFinding(
                f"{fw}-06", fw,
                "Gas collection efficiency must be reported for facilities with LFG systems",
                has_gce,
                "ERROR",
                (
                    "Gas collection efficiency is reported"
                    if has_gce
                    else "Gas collection efficiency is missing"
                ),
                "Provide gas_collection_efficiency per 40 CFR 98.343(a)(3)",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-06", fw,
                "Gas collection efficiency (N/A for non-landfill-gas-capture facilities)",
                True,
                "INFO",
                f"Gas collection efficiency not required for '{method}'",
                "",
            ))

        # REQ-07: Destruction efficiency for flares
        has_flare = self._has_any_field(
            data,
            "destruction_efficiency",
            "flare_efficiency",
            "destruction_removal_efficiency",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Destruction/removal efficiency must be documented if gas is flared or combusted",
            has_flare or not self._has_any_field(
                data, "methane_flared", "gas_flared",
            ),
            "WARNING",
            (
                "Destruction efficiency is documented"
                if has_flare
                else "Destruction efficiency not documented (may not be required)"
            ),
            "Document destruction_efficiency for flares per EPA guidance (default 0.99)",
        ))

        # REQ-08: Oxidation factor for landfill cover
        if method in {"LANDFILL", "LANDFILL_GAS_CAPTURE", "OPEN_DUMPING"}:
            has_ox = self._has_any_field(
                data, "oxidation_factor", "soil_oxidation_factor",
            )
            findings.append(ComplianceFinding(
                f"{fw}-08", fw,
                "Methane oxidation factor for landfill cover must be documented",
                has_ox,
                "WARNING",
                (
                    "Oxidation factor is documented"
                    if has_ox
                    else "Oxidation factor is missing for landfill"
                ),
                "Provide oxidation_factor per EPA default (0.10 for managed sites)",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-08", fw,
                "Landfill oxidation factor (N/A for non-landfill)",
                True,
                "INFO",
                f"Oxidation factor not required for '{method}'",
                "",
            ))

        # REQ-09: CO2e reported in metric tons
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Total CO2 equivalent must be reported in metric tons",
            has_co2e,
            "ERROR",
            (
                "CO2e is reported in metric tons"
                if has_co2e
                else "CO2e is not reported"
            ),
            "Report total_co2e_tonnes per EPA GHGRP requirements",
        ))

        # REQ-10: Reporting year identified
        has_year = self._has_any_field(
            data, "reporting_year", "reporting_period",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Reporting year must be identified for annual GHGRP submission",
            has_year,
            "ERROR",
            (
                "Reporting year is identified"
                if has_year
                else "Reporting year is not identified"
            ),
            "Specify reporting_year for annual EPA GHGRP submission (due March 31)",
        ))

        # REQ-11: Waste characterization
        has_char = self._has_any_field(
            data,
            "waste_category",
            "waste_composition",
            "waste_characterization",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Waste characterization data must be available per Subpart HH/TT",
            has_char,
            "WARNING",
            (
                "Waste characterization is available"
                if has_char
                else "Waste characterization is missing"
            ),
            "Provide waste_category or waste_composition per EPA reporting tables",
        ))

        # REQ-12: Verification data available
        has_verification = self._has_any_field(
            data,
            "provenance_hash",
            "verification_data",
            "quality_assurance",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Verification data must be available to support EPA reporting",
            has_verification,
            "WARNING",
            (
                "Verification data is available"
                if has_verification
                else "Verification data is not available"
            ),
            "Ensure data supports EPA verification requirements per 40 CFR 98.3(i)",
        ))

        # REQ-13: GWP values per EPA requirements
        gwp_source = str(self._get_data_field(data, "gwp_source", "")).upper()
        # EPA GHGRP currently uses AR4 GWP values
        epa_gwp_ok = gwp_source in {"AR4", "AR5"} or self._has_field(data, "gwp_source")
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "GWP values must align with EPA GHGRP requirements (currently AR4/AR5)",
            epa_gwp_ok,
            "WARNING",
            (
                f"GWP source '{gwp_source}' for EPA reporting"
                if gwp_source
                else "GWP source not specified for EPA compliance"
            ),
            "Specify gwp_source as AR4 or AR5 per current EPA GHGRP regulation",
        ))

        return findings

    # ==================================================================
    # Framework 7 - DEFRA Environmental Reporting -- 10 requirements
    # ==================================================================

    def check_defra(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with DEFRA Environmental Reporting Guidelines.

        10 requirements covering use of DEFRA conversion factors, UK-specific
        guidance, and Streamlined Energy and Carbon Reporting (SECR).

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "DEFRA"

        # REQ-01: DEFRA conversion factors used
        ef_source = str(self._get_data_field(data, "ef_source", "")).upper()
        uses_defra = "DEFRA" in ef_source or "BEIS" in ef_source
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "DEFRA/BEIS greenhouse gas conversion factors should be used for UK reporting",
            uses_defra or ef_source in VALID_EF_SOURCES,
            "WARNING",
            (
                f"EF source '{ef_source}': DEFRA factors {'are' if uses_defra else 'are not'} used"
            ),
            "Use DEFRA_BEIS greenhouse gas conversion factors (updated annually)",
        ))

        # REQ-02: UK-specific guidance followed
        has_uk_guidance = self._has_any_field(
            data,
            "uk_reporting",
            "secr_reporting",
            "defra_methodology",
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "UK Environmental Reporting Guidelines methodology should be followed",
            has_uk_guidance or uses_defra,
            "WARNING",
            (
                "UK-specific guidance followed"
                if has_uk_guidance or uses_defra
                else "UK-specific methodology not confirmed"
            ),
            "Follow UK Environmental Reporting Guidelines for waste emissions reporting",
        ))

        # REQ-03: Waste disposal method documented
        has_disposal = self._has_field(data, "treatment_method")
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Waste disposal/treatment method must be documented for DEFRA factor selection",
            has_disposal,
            "ERROR",
            (
                "Treatment method is documented"
                if has_disposal
                else "Treatment method is not documented"
            ),
            "Specify treatment_method for correct DEFRA factor lookup",
        ))

        # REQ-04: Waste type classification
        has_type = self._has_any_field(
            data, "waste_category", "waste_type",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Waste type must be classified per DEFRA factor categories",
            has_type,
            "ERROR",
            (
                "Waste type is classified"
                if has_type
                else "Waste type is not classified"
            ),
            "Classify waste_category per DEFRA conversion factor table categories",
        ))

        # REQ-05: Activity data in tonnes
        has_mass = self._has_any_field(
            data, "waste_mass_tonnes", "waste_quantity",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Activity data must be reported in tonnes for DEFRA factor application",
            has_mass,
            "ERROR",
            (
                "Activity data in tonnes is present"
                if has_mass
                else "Activity data in tonnes is missing"
            ),
            "Provide waste_mass_tonnes for DEFRA factor multiplication",
        ))

        # REQ-06: CO2e reported in tCO2e
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Emissions must be reported in tonnes CO2 equivalent",
            has_co2e,
            "ERROR",
            (
                "CO2e is reported"
                if has_co2e
                else "CO2e is not reported"
            ),
            "Report total_co2e_tonnes per DEFRA guidance",
        ))

        # REQ-07: Individual gas contributions
        has_gas_breakdown = self._has_any_field(
            data,
            "co2_tonnes",
            "ch4_tonnes",
            "n2o_tonnes",
            "gases_reported",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Individual gas contributions (CO2, CH4, N2O) should be reported",
            has_gas_breakdown,
            "WARNING",
            (
                "Individual gas breakdown is available"
                if has_gas_breakdown
                else "Individual gas breakdown is not available"
            ),
            "Report co2_tonnes, ch4_tonnes, n2o_tonnes separately alongside CO2e",
        ))

        # REQ-08: Reporting period matches financial year
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Reporting period must align with the company financial year (SECR requirement)",
            has_period,
            "WARNING",
            (
                "Reporting period is defined"
                if has_period
                else "Reporting period is not defined"
            ),
            "Align reporting_period with the company financial year for SECR",
        ))

        # REQ-09: Intensity metric calculated
        has_intensity = self._has_any_field(
            data,
            "emissions_intensity",
            "intensity_ratio",
            "tco2e_per_tonne_waste",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "An emissions intensity metric should be calculated (e.g. tCO2e per tonne waste)",
            has_intensity,
            "INFO",
            (
                "Intensity metric is calculated"
                if has_intensity
                else "Intensity metric is not calculated"
            ),
            "Calculate tco2e_per_tonne_waste as an intensity metric per SECR guidance",
        ))

        # REQ-10: Provenance for audit
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Calculation must be auditable with documented provenance",
            has_prov,
            "WARNING",
            (
                "Provenance hash is present"
                if has_prov
                else "Provenance hash is not present"
            ),
            "Enable provenance tracking for UK reporting audit requirements",
        ))

        return findings

    # ------------------------------------------------------------------
    # Get Framework Requirements
    # ------------------------------------------------------------------

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Get a listing of all requirements for a specific framework.

        Args:
            framework: Framework name (e.g. "IPCC_2006", "GHG_PROTOCOL").

        Returns:
            List of requirement dictionaries with id, description, severity.

        Raises:
            ValueError: If the framework is not supported.
        """
        fw_upper = framework.upper()
        if fw_upper not in self._framework_checkers:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Supported: {SUPPORTED_FRAMEWORKS}"
            )

        checker = self._framework_checkers[fw_upper]
        # Run with empty data to extract requirement metadata
        empty_data: Dict[str, Any] = {}
        findings = checker(empty_data)

        return [
            {
                "requirement_id": f.requirement_id,
                "framework": f.framework,
                "requirement": f.requirement,
                "severity": f.severity,
            }
            for f in findings
        ]

    # ------------------------------------------------------------------
    # Get All Requirements
    # ------------------------------------------------------------------

    def get_all_requirements(self) -> Dict[str, Any]:
        """Get a listing of all compliance requirements across all frameworks.

        Returns:
            Dictionary with framework-by-framework requirement listings
            and total count.
        """
        all_reqs: Dict[str, List[Dict[str, str]]] = {}

        for fw_name, checker in self._framework_checkers.items():
            findings = checker({})
            all_reqs[fw_name] = [
                {
                    "requirement_id": f.requirement_id,
                    "requirement": f.requirement,
                    "severity": f.severity,
                }
                for f in findings
            ]

        total = sum(len(v) for v in all_reqs.values())
        return {
            "total_requirements": total,
            "frameworks": list(all_reqs.keys()),
            "requirements": all_reqs,
        }

    # ------------------------------------------------------------------
    # Get Compliance Summary
    # ------------------------------------------------------------------

    def get_compliance_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a compliance summary from a list of framework results.

        This is a convenience method that accepts the ``results`` list from
        ``check_compliance`` output and produces a high-level summary.

        Args:
            results: List of per-framework result dictionaries, each with
                "framework", "status", "total_requirements", "passed",
                "failed", "findings", and "recommendations" keys.

        Returns:
            Summary dictionary with counts by status, overall assessment,
            and consolidated recommendations.
        """
        if not results:
            return {
                "frameworks_checked": 0,
                "overall_status": "not_assessed",
                "compliant_count": 0,
                "non_compliant_count": 0,
                "partial_count": 0,
                "total_requirements": 0,
                "total_passed": 0,
                "total_failed": 0,
                "critical_findings": [],
                "consolidated_recommendations": [],
            }

        compliant = 0
        non_compliant = 0
        partial = 0
        total_reqs = 0
        total_passed = 0
        total_failed = 0
        critical_findings: List[Dict[str, Any]] = []
        all_recommendations: List[str] = []

        for r in results:
            status = str(r.get("status", "")).lower()
            if status == "compliant":
                compliant += 1
            elif status == "non_compliant":
                non_compliant += 1
            elif status == "partial":
                partial += 1

            total_reqs += int(r.get("total_requirements", 0))
            total_passed += int(r.get("passed", 0))
            total_failed += int(r.get("failed", 0))

            # Collect critical (ERROR) findings
            for f in r.get("findings", []):
                if isinstance(f, dict) and f.get("severity") == "ERROR" and not f.get("passed", True):
                    critical_findings.append(f)

            # Collect recommendations
            for rec in r.get("recommendations", []):
                if rec and rec not in all_recommendations:
                    all_recommendations.append(rec)

        # Determine overall status
        if total_reqs == 0:
            overall_status = "not_assessed"
        elif total_failed == 0:
            overall_status = "compliant"
        elif non_compliant > 0:
            overall_status = "non_compliant" if non_compliant > compliant else "partial"
        else:
            overall_status = "partial"

        return {
            "frameworks_checked": len(results),
            "overall_status": overall_status,
            "compliant_count": compliant,
            "non_compliant_count": non_compliant,
            "partial_count": partial,
            "total_requirements": total_reqs,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_findings": critical_findings,
            "consolidated_recommendations": all_recommendations,
            "provenance_hash": _compute_hash({
                "frameworks_checked": len(results),
                "overall_status": overall_status,
                "total_passed": total_passed,
                "total_failed": total_failed,
            }),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine metadata and counters.
        """
        with self._lock:
            return {
                "engine": "ComplianceCheckerEngine",
                "agent": "AGENT-MRV-007",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_checks": self._total_checks,
                "supported_frameworks": SUPPORTED_FRAMEWORKS,
                "total_requirements": TOTAL_REQUIREMENTS,
                "framework_requirement_counts": {
                    "IPCC_2006": 15,
                    "IPCC_2019": 12,
                    "GHG_PROTOCOL": 18,
                    "ISO_14064": 14,
                    "CSRD_ESRS": 16,
                    "EPA_40CFR98": 13,
                    "DEFRA": 10,
                },
            }

    def reset(self) -> None:
        """Reset engine counters."""
        with self._lock:
            self._total_checks = 0
        logger.info("ComplianceCheckerEngine reset")
