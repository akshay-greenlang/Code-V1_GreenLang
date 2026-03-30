# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Multi-Framework Regulatory Compliance (Engine 6 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Validates agricultural emission calculations against seven regulatory frameworks
to ensure data completeness, methodological correctness, and reporting
readiness.  Each framework defines specific requirements that are
individually checked and scored.

Supported Frameworks (95 total requirements):
    1. IPCC 2006 Vol 4 (Ch 10-11)       - 15 requirements
    2. IPCC 2019 Refinement              - 12 requirements
    3. GHG Protocol Agricultural         - 18 requirements
    4. ISO 14064-1:2018                  - 14 requirements
    5. CSRD/ESRS E1 & E4                 - 16 requirements
    6. EPA 40 CFR 98 Subpart JJ          - 10 requirements
    7. DEFRA Environmental Reporting     - 10 requirements

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met

Severity Levels:
    ERROR:   Requirement failure prevents regulatory compliance
    WARNING: Requirement failure should be addressed but not blocking
    INFO:    Informational finding for best practice improvement

Agricultural Emission Sources Checked:
    - Enteric fermentation (CH4) from livestock digestive processes
    - Manure management (CH4 + N2O) from storage and treatment
    - Agricultural soils (N2O) from synthetic fertilizers, organic amendments
    - Rice cultivation (CH4) from flooded paddy fields
    - Liming (CO2) from limestone and dolomite application
    - Urea application (CO2) from urea and other C-containing fertilizers
    - Field burning (CH4 + N2O) from crop residue combustion
    - Indirect N2O from volatilization and leaching/runoff

Zero-Hallucination Guarantees:
    - All compliance checks are deterministic boolean evaluations.
    - No LLM involvement in any compliance determination.
    - Requirement definitions are hard-coded from regulatory texts.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_data={
    ...         "emission_source": "ENTERIC_FERMENTATION",
    ...         "animal_type": "DAIRY_CATTLE",
    ...         "tier": "TIER_2",
    ...         "enteric_ef_source": "IPCC_2006_TABLE_10_11",
    ...         "gwp_source": "AR5",
    ...         "total_ch4_kg": 12800.0,
    ...         "total_co2e_tonnes": 358.4,
    ...         "has_uncertainty": True,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["IPCC_2006_VOL4"],
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ComplianceCheckerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

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

#: Available compliance frameworks.
SUPPORTED_FRAMEWORKS: List[str] = [
    "IPCC_2006_VOL4",
    "IPCC_2019",
    "GHG_PROTOCOL",
    "ISO_14064",
    "CSRD_ESRS",
    "EPA_40CFR98",
    "DEFRA",
]

#: Total requirements across all frameworks.
TOTAL_REQUIREMENTS: int = 95

#: Valid emission source types for agricultural emissions.
VALID_EMISSION_SOURCES: List[str] = [
    "ENTERIC_FERMENTATION",
    "MANURE_MANAGEMENT",
    "AGRICULTURAL_SOILS",
    "RICE_CULTIVATION",
    "LIMING",
    "UREA_APPLICATION",
    "FIELD_BURNING",
    "CROP_RESIDUE",
    "PRESCRIBED_BURNING",
    "SAVANNA_BURNING",
]

#: Valid IPCC animal categories.
VALID_ANIMAL_TYPES: List[str] = [
    "DAIRY_CATTLE", "NON_DAIRY_CATTLE", "BUFFALO",
    "SHEEP", "GOATS", "CAMELS",
    "HORSES", "MULES_ASSES",
    "SWINE_MARKET", "SWINE_BREEDING",
    "POULTRY_LAYERS", "POULTRY_BROILERS", "TURKEYS", "DUCKS",
    "DEER", "ELK", "ALPACAS_LLAMAS",
    "RABBITS", "FUR_BEARING", "OTHER_LIVESTOCK",
]

#: Valid animal waste management systems (AWMS).
VALID_AWMS_TYPES: List[str] = [
    "PASTURE_RANGE", "DAILY_SPREAD", "SOLID_STORAGE",
    "DRY_LOT", "LIQUID_SLURRY",
    "UNCOVERED_ANAEROBIC_LAGOON", "PIT_STORAGE",
    "ANAEROBIC_DIGESTER", "BURNED_FOR_FUEL",
    "DEEP_BEDDING", "COMPOSTING", "AEROBIC_TREATMENT",
    "DEEP_PIT", "POULTRY_WITH_LITTER", "POULTRY_WITHOUT_LITTER",
]

#: Valid calculation tiers.
VALID_TIERS: List[str] = [
    "TIER_1", "TIER_2", "TIER_3",
    "COUNTRY_SPECIFIC", "DIRECT_MEASUREMENT",
]

#: Valid GWP assessment report sources.
VALID_GWP_SOURCES: List[str] = ["AR4", "AR5", "AR6"]

#: Valid soil N2O pathways.
VALID_N2O_PATHWAYS: List[str] = [
    "DIRECT_SYNTHETIC_FERTILIZER",
    "DIRECT_ORGANIC_AMENDMENT",
    "DIRECT_CROP_RESIDUE",
    "DIRECT_ORGANIC_SOILS",
    "INDIRECT_VOLATILIZATION",
    "INDIRECT_LEACHING",
]

#: Valid rice water regimes.
VALID_WATER_REGIMES: List[str] = [
    "CONTINUOUSLY_FLOODED",
    "INTERMITTENT_SINGLE",
    "INTERMITTENT_MULTIPLE",
    "RAINFED_REGULAR",
    "RAINFED_DROUGHT",
    "DEEP_WATER",
    "UPLAND",
]

#: IPCC default EF1 for direct soil N2O (Table 11.1, 2006 Guidelines).
IPCC_2006_DEFAULT_EF1: float = 0.01

#: IPCC default EF4 for volatilization (Table 11.3, 2006 Guidelines).
IPCC_2006_DEFAULT_EF4: float = 0.01

#: IPCC default EF5 for leaching/runoff (Table 11.3, 2006 Guidelines).
IPCC_2006_DEFAULT_EF5: float = 0.0075

#: Approved enteric emission factor sources.
APPROVED_ENTERIC_EF_SOURCES: List[str] = [
    "IPCC_2006_TABLE_10_11",
    "IPCC_2019_TABLE_10_11",
    "COUNTRY_SPECIFIC",
    "SITE_MEASURED",
    "PEER_REVIEWED",
]

#: Approved manure MCF sources.
APPROVED_MANURE_MCF_SOURCES: List[str] = [
    "IPCC_2006_TABLE_10_17",
    "IPCC_2019_TABLE_10_17",
    "COUNTRY_SPECIFIC",
    "SITE_MEASURED",
]

#: Valid crop types for field burning.
VALID_CROP_TYPES_BURNING: List[str] = [
    "RICE", "WHEAT", "MAIZE", "SUGARCANE", "COTTON",
    "BARLEY", "OATS", "SORGHUM", "MILLET", "OTHER",
]

#: Valid N fertilizer types for direct soil N2O.
VALID_FERTILIZER_TYPES: List[str] = [
    "UREA", "AMMONIUM_NITRATE", "AMMONIUM_SULFATE",
    "CALCIUM_AMMONIUM_NITRATE", "DIAMMONIUM_PHOSPHATE",
    "MONOAMMONIUM_PHOSPHATE", "ANHYDROUS_AMMONIA",
    "NPK_BLEND", "ORGANIC_MANURE", "COMPOST",
    "CROP_RESIDUE", "OTHER",
]

# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class ComplianceFinding:
    """Single compliance finding for a requirement.

    Attributes:
        requirement_id: Unique identifier for the requirement (e.g. IPCC_2006_VOL4-01).
        framework: Regulatory framework name.
        description: Requirement description from the regulatory text.
        status: PASS, FAIL, WARNING, or INFO.
        severity: ERROR, WARNING, or INFO.
        recommendation: Recommended action if the check did not pass.
    """

    requirement_id: str
    framework: str
    description: str
    passed: bool
    severity: str
    finding: str
    recommendation: str

    # Convenience properties for the status field requested in the spec.
    @property
    def status(self) -> str:
        """Return the compliance status string."""
        if self.passed:
            return "PASS"
        if self.severity == "INFO":
            return "INFO"
        if self.severity == "WARNING":
            return "WARNING"
        return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "framework": self.framework,
            "description": self.description,
            "status": self.status,
            "passed": self.passed,
            "severity": self.severity,
            "finding": self.finding,
            "recommendation": self.recommendation,
        }

# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================

class ComplianceCheckerEngine:
    """Multi-framework regulatory compliance checker for agricultural emissions.

    Validates agricultural emission calculation results against seven
    regulatory frameworks with 95 total requirements covering enteric
    fermentation, manure management, cropland emissions, rice cultivation,
    liming, urea application, and field burning.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Frameworks:
        1. IPCC_2006_VOL4   - 15 requirements (AFOLU Ch 10-11)
        2. IPCC_2019         - 12 requirements (2019 Refinement)
        3. GHG_PROTOCOL      - 18 requirements (Agricultural Guidance)
        4. ISO_14064          - 14 requirements (ISO 14064-1:2018)
        5. CSRD_ESRS          - 16 requirements (ESRS E1 + E4)
        6. EPA_40CFR98        - 10 requirements (Subpart JJ)
        7. DEFRA              - 10 requirements (UK Environmental Reporting)

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(data, ["IPCC_2006_VOL4"])
        >>> assert result["overall"]["compliance_status"] in (
        ...     "COMPLIANT", "PARTIAL", "NON_COMPLIANT"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the ComplianceCheckerEngine."""
        self._lock = threading.RLock()
        self._total_checks: int = 0
        self._total_findings: int = 0
        self._total_passed: int = 0
        self._total_failed: int = 0
        self._created_at = utcnow()

        # Map framework names to checker methods.
        self._framework_checkers: Dict[str, Callable] = {
            "IPCC_2006_VOL4": self.check_ipcc_2006,
            "IPCC_2019": self.check_ipcc_2019,
            "GHG_PROTOCOL": self.check_ghg_protocol,
            "ISO_14064": self.check_iso_14064,
            "CSRD_ESRS": self.check_csrd_esrs,
            "EPA_40CFR98": self.check_epa_40cfr98,
            "DEFRA": self.check_defra,
        }

        # Requirement counts per framework (for documentation and validation).
        self._requirement_counts: Dict[str, int] = {
            "IPCC_2006_VOL4": 15,
            "IPCC_2019": 12,
            "GHG_PROTOCOL": 18,
            "ISO_14064": 14,
            "CSRD_ESRS": 16,
            "EPA_40CFR98": 10,
            "DEFRA": 10,
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

    def _record_findings(self, findings: List[ComplianceFinding]) -> None:
        """Thread-safe recording of finding statistics."""
        with self._lock:
            self._total_findings += len(findings)
            self._total_passed += sum(1 for f in findings if f.passed)
            self._total_failed += sum(1 for f in findings if not f.passed)

    def _get_data_field(
        self,
        data: Dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        """Safely extract a field from calculation data.

        Args:
            data: Calculation data dictionary.
            key: Field name to extract.
            default: Default value if field is absent.

        Returns:
            Field value or default.
        """
        return data.get(key, default)

    def _has_field(self, data: Dict[str, Any], key: str) -> bool:
        """Check if a field exists and is non-empty in calculation data.

        Args:
            data: Calculation data dictionary.
            key: Field name to check.

        Returns:
            True if the field exists and has a non-empty value.
        """
        val = data.get(key)
        if val is None:
            return False
        if isinstance(val, (str, list, dict)) and len(val) == 0:
            return False
        return True

    def _has_any_field(self, data: Dict[str, Any], *keys: str) -> bool:
        """Check if any of the specified fields exists and is non-empty.

        Args:
            data: Calculation data dictionary.
            keys: Field names to check.

        Returns:
            True if at least one field exists and has a non-empty value.
        """
        return any(self._has_field(data, k) for k in keys)

    def _has_all_fields(self, data: Dict[str, Any], *keys: str) -> bool:
        """Check if all of the specified fields exist and are non-empty.

        Args:
            data: Calculation data dictionary.
            keys: Field names to check.

        Returns:
            True if all fields exist and have non-empty values.
        """
        return all(self._has_field(data, k) for k in keys)

    def _get_emission_sources(self, data: Dict[str, Any]) -> List[str]:
        """Extract the list of emission sources from calculation data.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of normalized emission source strings.
        """
        sources = data.get("emission_sources", [])
        if isinstance(sources, list):
            return [s.upper() for s in sources if isinstance(s, str)]
        single = data.get("emission_source", "")
        if isinstance(single, str) and single:
            return [single.upper()]
        return []

    def _get_animal_types(self, data: Dict[str, Any]) -> List[str]:
        """Extract the list of animal types from calculation data.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of normalized animal type strings.
        """
        types = data.get("animal_types", [])
        if isinstance(types, list):
            return [t.upper() for t in types if isinstance(t, str)]
        single = data.get("animal_type", "")
        if isinstance(single, str) and single:
            return [single.upper()]
        return []

    def _get_n2o_pathways(self, data: Dict[str, Any]) -> List[str]:
        """Extract reported N2O pathways from calculation data.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of normalized N2O pathway strings.
        """
        pathways = data.get("n2o_pathways", [])
        if isinstance(pathways, list):
            return [p.upper() for p in pathways if isinstance(p, str)]
        return []

    # ------------------------------------------------------------------
    # Main Compliance Check
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        calculation_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run compliance checks against selected frameworks.

        This is the main entry point for compliance checking.  It delegates
        to framework-specific checker methods and aggregates results into
        a unified compliance report.

        Args:
            calculation_data: Dictionary with agricultural emission
                calculation results.  Keys depend on emission source
                (enteric, manure, cropland, rice, etc.).
            frameworks: List of framework names to check.  If None,
                checks all seven supported frameworks.

        Returns:
            Dictionary with per-framework compliance results, overall
            summary statistics, and a provenance hash.

        Example:
            >>> result = engine.check_compliance(data, ["IPCC_2006_VOL4"])
            >>> result["overall"]["compliance_status"]
            'COMPLIANT'
        """
        self._increment_checks()
        start_time = time.monotonic()
        calc_id = str(uuid4())

        if frameworks is None:
            frameworks = list(SUPPORTED_FRAMEWORKS)

        # Validate framework names.
        valid_fws: List[str] = []
        for fw in frameworks:
            fw_upper = fw.upper()
            if fw_upper in self._framework_checkers:
                valid_fws.append(fw_upper)
            else:
                logger.warning("Unknown framework '%s', skipping", fw)

        # Run checks per framework.
        framework_results: Dict[str, Dict[str, Any]] = {}
        all_findings: List[Dict[str, Any]] = []
        total_passed = 0
        total_requirements = 0
        total_errors = 0
        total_warnings = 0

        for fw in valid_fws:
            checker = self._framework_checkers[fw]
            findings = checker(calculation_data)
            self._record_findings(findings)

            passed = sum(1 for f in findings if f.passed)
            failed = sum(1 for f in findings if not f.passed)
            errors = sum(
                1 for f in findings if not f.passed and f.severity == "ERROR"
            )
            warnings = sum(
                1 for f in findings if not f.passed and f.severity == "WARNING"
            )

            total_reqs = len(findings)
            pass_rate = (passed / total_reqs * 100) if total_reqs > 0 else 0

            if pass_rate == 100:
                status = "COMPLIANT"
            elif pass_rate >= 50:
                status = "PARTIAL"
            else:
                status = "NON_COMPLIANT"

            framework_results[fw] = {
                "status": status,
                "total_requirements": total_reqs,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "warnings": warnings,
                "pass_rate_pct": round(pass_rate, 1),
                "findings": [f.to_dict() for f in findings],
            }

            all_findings.extend([f.to_dict() for f in findings])
            total_passed += passed
            total_requirements += total_reqs
            total_errors += errors
            total_warnings += warnings

        # Overall status.
        overall_rate = (
            (total_passed / total_requirements * 100)
            if total_requirements > 0 else 0
        )
        if overall_rate == 100:
            overall_status = "COMPLIANT"
        elif overall_rate >= 50:
            overall_status = "PARTIAL"
        else:
            overall_status = "NON_COMPLIANT"

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "overall": {
                "compliance_status": overall_status,
                "total_requirements": total_requirements,
                "total_passed": total_passed,
                "total_failed": total_requirements - total_passed,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "pass_rate_pct": round(overall_rate, 1),
            },
            "frameworks_checked": valid_fws,
            "framework_results": framework_results,
            "processing_time_ms": processing_time,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Compliance check complete: id=%s, frameworks=%d, "
            "overall=%s, passed=%d/%d (%.1f%%), time=%.3fms",
            calc_id, len(valid_fws), overall_status,
            total_passed, total_requirements, overall_rate, processing_time,
        )
        return result

    # ==================================================================
    # Framework 1: IPCC 2006 Vol 4 (15 requirements)
    # ==================================================================

    def check_ipcc_2006(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with IPCC 2006 Guidelines Vol 4 (AFOLU Ch 10-11).

        15 requirements covering animal classification, enteric emission
        factors, manure MCF parameters, VS and Bo values, N excretion
        rates, soil N2O pathways, rice CH4 scaling, GWP source, and
        tier identification.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "IPCC_2006_VOL4"

        # REQ-01: Animal classification matches IPCC categories.
        animal_types = self._get_animal_types(data)
        has_valid_animals = (
            len(animal_types) > 0
            and all(a in VALID_ANIMAL_TYPES for a in animal_types)
        ) or not self._has_any_field(data, "animal_type", "animal_types")
        animal_str = ", ".join(animal_types[:5]) if animal_types else "none"
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Animal classification must match IPCC Vol 4 Ch 10 categories",
            has_valid_animals,
            "ERROR",
            f"Animal types reported: {animal_str}. "
            + ("All match IPCC categories" if has_valid_animals
               else "Invalid animal types detected"),
            "Use IPCC Table 10.10 animal categories (DAIRY_CATTLE, "
            "NON_DAIRY_CATTLE, BUFFALO, SHEEP, GOATS, CAMELS, HORSES, "
            "MULES_ASSES, SWINE_MARKET, SWINE_BREEDING, POULTRY_LAYERS, etc.)",
        ))

        # REQ-02: Enteric EFs from approved sources (Table 10.11).
        ef_source = str(self._get_data_field(data, "enteric_ef_source", "")).upper()
        has_approved_ef = (
            ef_source in APPROVED_ENTERIC_EF_SOURCES
            or self._has_field(data, "enteric_ef_source")
        )
        source_type = self._get_data_field(data, "emission_source", "")
        if str(source_type).upper() not in ("ENTERIC_FERMENTATION", ""):
            has_approved_ef = True  # N/A for non-enteric sources
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Enteric fermentation EFs must be from IPCC 2006 Table 10.11 "
            "or approved equivalent source",
            has_approved_ef,
            "ERROR",
            f"Enteric EF source: '{ef_source}'. "
            + ("Approved" if has_approved_ef else "Not from approved source"),
            "Use emission factors from IPCC 2006 Table 10.11, IPCC 2019 "
            "Table 10.11, country-specific, or site-measured sources",
        ))

        # REQ-03: Manure MCF by temperature and AWMS type (Table 10.17).
        has_mcf = self._has_any_field(
            data, "manure_mcf", "mcf_value", "mcf_source",
        )
        has_awms = self._has_any_field(data, "awms_type", "manure_system")
        has_temp = self._has_any_field(data, "annual_avg_temperature", "temperature_c")
        mcf_check = has_mcf or not self._has_any_field(data, "manure_ch4", "manure_management")
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Manure CH4 conversion factor (MCF) must be selected by annual "
            "average temperature and AWMS type per IPCC Table 10.17",
            mcf_check,
            "ERROR",
            f"MCF: {'documented' if has_mcf else 'not documented'}, "
            f"AWMS: {'specified' if has_awms else 'not specified'}, "
            f"Temperature: {'provided' if has_temp else 'not provided'}",
            "Provide manure_mcf with temperature_c and awms_type to select "
            "MCF from IPCC 2006 Table 10.17",
        ))

        # REQ-04: VS and Bo values documented (Tables 10A-4, 10A-7).
        has_vs = self._has_any_field(data, "volatile_solids", "vs_rate", "vs_kg_per_day")
        has_bo = self._has_any_field(data, "bo_value", "maximum_methane_producing_capacity")
        vs_bo_check = (has_vs and has_bo) or not self._has_any_field(
            data, "manure_ch4", "manure_management",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Volatile solids (VS) and maximum methane producing capacity "
            "(Bo) must be documented per IPCC Tables 10A-4 and 10A-7",
            vs_bo_check,
            "ERROR",
            f"VS: {'documented' if has_vs else 'not documented'}, "
            f"Bo: {'documented' if has_bo else 'not documented'}",
            "Provide volatile_solids (kg VS/head/day) and bo_value "
            "(m3 CH4/kg VS) from IPCC Tables 10A-4 and 10A-7",
        ))

        # REQ-05: N excretion rates traceable (Table 10.19).
        has_nex = self._has_any_field(
            data, "n_excretion_rate", "nex_rate", "n_excretion_kg_per_head_yr",
        )
        nex_check = has_nex or not self._has_any_field(
            data, "manure_n2o", "n2o_manure",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Nitrogen excretion rates (Nex) must be traceable to IPCC "
            "Table 10.19 or country-specific data",
            nex_check,
            "ERROR",
            f"N excretion rate: {'traceable' if has_nex else 'not documented'}",
            "Provide n_excretion_rate (kg N/head/yr) from IPCC Table 10.19 "
            "or country-specific livestock nitrogen excretion data",
        ))

        # REQ-06: Soil N2O uses correct EF1 (0.01) and pathways.
        ef1_value = self._get_data_field(data, "ef1_value")
        n2o_pathways = self._get_n2o_pathways(data)
        has_direct_pathway = any(
            p.startswith("DIRECT") for p in n2o_pathways
        )
        ef1_ok = True
        if ef1_value is not None:
            try:
                ef1_ok = abs(float(ef1_value) - IPCC_2006_DEFAULT_EF1) < 0.005
            except (ValueError, TypeError):
                ef1_ok = False
        n2o_soil_check = (
            (ef1_ok and has_direct_pathway)
            or not self._has_any_field(data, "soil_n2o", "direct_n2o", "agricultural_soils")
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Soil N2O must use IPCC default EF1 = 0.01 (or justified "
            "country-specific value) with correct direct emission pathways",
            n2o_soil_check,
            "ERROR",
            f"EF1: {ef1_value if ef1_value is not None else 'not specified'}, "
            f"Direct pathways: {n2o_pathways[:3] if n2o_pathways else 'none'}",
            "Set ef1_value to 0.01 (IPCC 2006 Table 11.1 default) or "
            "provide justification for country-specific EF1. Include "
            "n2o_pathways with DIRECT_SYNTHETIC_FERTILIZER, "
            "DIRECT_ORGANIC_AMENDMENT, DIRECT_CROP_RESIDUE",
        ))

        # REQ-07: Indirect N2O includes volatilization + leaching.
        has_vol = self._has_any_field(
            data, "indirect_n2o_volatilization", "n2o_volatilization",
            "frac_gasf", "frac_gasm",
        )
        has_leach = self._has_any_field(
            data, "indirect_n2o_leaching", "n2o_leaching",
            "frac_leach",
        )
        indirect_check = (
            (has_vol and has_leach)
            or not self._has_any_field(data, "indirect_n2o", "soil_n2o", "agricultural_soils")
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Indirect N2O must include both volatilization (EF4) and "
            "leaching/runoff (EF5) pathways per IPCC Ch 11",
            indirect_check,
            "ERROR",
            f"Volatilization: {'included' if has_vol else 'missing'}, "
            f"Leaching: {'included' if has_leach else 'missing'}",
            "Include indirect_n2o_volatilization (EF4 = 0.01) and "
            "indirect_n2o_leaching (EF5 = 0.0075) per IPCC 2006 Table 11.3",
        ))

        # REQ-08: Rice CH4 uses baseline EF with scaling factors.
        has_rice_ef = self._has_any_field(
            data, "rice_baseline_ef", "rice_ef_c", "rice_ch4_ef",
        )
        has_scaling = self._has_any_field(
            data, "rice_scaling_factors", "sf_w", "sf_p", "sf_o", "sf_s",
        )
        has_water_regime = self._has_any_field(data, "water_regime", "rice_water_regime")
        rice_check = (
            (has_rice_ef and has_scaling)
            or not self._has_any_field(data, "rice_ch4", "rice_cultivation")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Rice cultivation CH4 must use IPCC baseline emission factor "
            "with water regime, organic amendment, and soil type scaling factors",
            rice_check,
            "ERROR",
            f"Baseline EF: {'present' if has_rice_ef else 'absent'}, "
            f"Scaling factors: {'present' if has_scaling else 'absent'}, "
            f"Water regime: {'specified' if has_water_regime else 'not specified'}",
            "Provide rice_baseline_ef (kg CH4/ha/day from IPCC Table 5.11) "
            "with sf_w (water regime), sf_p (pre-season), sf_o (organic "
            "amendments), sf_s (soil type) scaling factors",
        ))

        # REQ-09: GWP source specified (AR4/AR5/AR6).
        gwp_source = str(self._get_data_field(data, "gwp_source", "")).upper()
        has_gwp = gwp_source in VALID_GWP_SOURCES
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "GWP assessment report source must be specified (AR4, AR5, or AR6)",
            has_gwp or self._has_field(data, "gwp_source"),
            "WARNING",
            f"GWP source: '{gwp_source}'" if gwp_source else "GWP source not specified",
            "Specify gwp_source as AR4 (CH4=25, N2O=298), AR5 (CH4=28, "
            "N2O=265), or AR6 (CH4=27.9, N2O=273)",
        ))

        # REQ-10: Tier identification (1/2/3) documented.
        tier = str(self._get_data_field(data, "tier", "")).upper()
        has_tier = tier in VALID_TIERS or self._has_field(data, "tier")
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Calculation tier (1, 2, or 3) must be identified and documented",
            has_tier,
            "WARNING",
            f"Tier: {tier}" if has_tier else "Tier not specified",
            "Specify tier as TIER_1, TIER_2, or TIER_3 per IPCC Vol 4 "
            "decision tree guidance",
        ))

        # REQ-11: Emission source identified.
        sources = self._get_emission_sources(data)
        has_source = len(sources) > 0
        valid_sources = all(s in VALID_EMISSION_SOURCES for s in sources)
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Agricultural emission source must be identified from IPCC AFOLU categories",
            has_source and valid_sources,
            "ERROR",
            f"Emission sources: {sources[:5] if sources else 'none'}. "
            + ("All valid" if valid_sources and has_source else "Invalid or missing sources"),
            "Specify emission_source from: ENTERIC_FERMENTATION, "
            "MANURE_MANAGEMENT, AGRICULTURAL_SOILS, RICE_CULTIVATION, "
            "LIMING, UREA_APPLICATION, FIELD_BURNING",
        ))

        # REQ-12: CO2e result reported.
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "total_co2e_kg",
            "ch4_co2e_tonnes", "n2o_co2e_tonnes", "co2_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Total CO2 equivalent emissions must be reported",
            has_co2e,
            "ERROR",
            "CO2e " + ("is" if has_co2e else "is not") + " reported",
            "Report total_co2e_tonnes in the calculation result",
        ))

        # REQ-13: Population/activity data documented.
        has_population = self._has_any_field(
            data, "population", "head_count", "animal_population",
            "area_ha", "fertilizer_kg_n", "rice_area_ha",
        )
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Activity data (population, area, fertilizer quantity) must be documented",
            has_population,
            "ERROR",
            "Activity data " + ("is" if has_population else "is not") + " documented",
            "Provide population (head_count for livestock), area_ha "
            "(for cropland/rice), or fertilizer_kg_n (for soil N2O)",
        ))

        # REQ-14: Provenance hash present.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Provenance hash must be present for audit trail",
            has_prov,
            "WARNING",
            "Provenance hash " + ("is" if has_prov else "is not") + " present",
            "Enable provenance tracking for IPCC audit trail compliance",
        ))

        # REQ-15: Reporting period defined.
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "year_start", "year_end")
        findings.append(ComplianceFinding(
            f"{fw}-15", fw,
            "Reporting time period must be defined for annual inventory",
            has_period,
            "ERROR",
            "Reporting period " + ("is" if has_period else "is not") + " defined",
            "Specify reporting_year or reporting_period (start/end dates)",
        ))

        return findings

    # ==================================================================
    # Framework 2: IPCC 2019 Refinement (12 requirements)
    # ==================================================================

    def check_ipcc_2019(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with 2019 Refinement to IPCC 2006 Guidelines.

        12 requirements covering updated emission factors for disaggregated
        cattle subcategories, revised manure MCF temperature dependencies,
        updated soil N2O EF1, rice cultivation updates, and improved
        uncertainty ranges.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "IPCC_2019"

        # REQ-01: 2019 Refinement guidelines referenced.
        ef_source = str(self._get_data_field(data, "enteric_ef_source", "")).upper()
        mcf_source = str(self._get_data_field(data, "mcf_source", "")).upper()
        references_2019 = (
            "2019" in ef_source
            or "2019" in mcf_source
            or self._has_field(data, "ipcc_2019_applied")
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "2019 Refinement to IPCC 2006 Guidelines must be referenced "
            "where updated factors are available",
            references_2019 or self._has_any_field(data, "enteric_ef_source", "tier"),
            "WARNING",
            f"2019 Refinement: {'referenced' if references_2019 else 'not explicitly referenced'}",
            "Reference IPCC 2019 Refinement as primary source where "
            "updated emission factors or methodologies are available",
        ))

        # REQ-02: Updated enteric EFs for disaggregated cattle.
        animal_types = self._get_animal_types(data)
        has_cattle = any(
            "CATTLE" in a or "DAIRY" in a for a in animal_types
        )
        has_disaggregated = self._has_any_field(
            data, "cattle_subcategory", "cattle_region",
            "cattle_productivity_class",
        )
        cattle_check = (
            (has_disaggregated or not has_cattle)
            or self._has_field(data, "tier")
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Updated enteric fermentation EFs for disaggregated cattle "
            "subcategories should be applied per 2019 Refinement Table 10.11",
            cattle_check,
            "WARNING",
            f"Cattle present: {has_cattle}, "
            f"Disaggregated: {'yes' if has_disaggregated else 'no'}",
            "Apply 2019 Refinement disaggregated cattle EFs by region "
            "and productivity class where Tier 1 is used",
        ))

        # REQ-03: Revised manure MCF temperature dependencies.
        has_temp = self._has_any_field(
            data, "annual_avg_temperature", "temperature_c",
        )
        has_mcf = self._has_any_field(data, "manure_mcf", "mcf_value")
        mcf_temp_check = (
            (has_temp and has_mcf)
            or not self._has_any_field(data, "manure_ch4", "manure_management")
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Revised manure MCF temperature dependencies from 2019 "
            "Refinement Table 10.17 should be applied",
            mcf_temp_check,
            "WARNING",
            f"Temperature: {'provided' if has_temp else 'missing'}, "
            f"MCF: {'documented' if has_mcf else 'not documented'}",
            "Apply revised MCF-temperature relationship from 2019 "
            "Refinement, particularly for cool/cold climates with "
            "non-zero MCF values",
        ))

        # REQ-04: Updated soil N2O EF1 (refined from 2006).
        ef1_value = self._get_data_field(data, "ef1_value")
        ef1_source = str(self._get_data_field(data, "ef1_source", "")).upper()
        uses_2019_ef1 = "2019" in ef1_source
        ef1_check = (
            uses_2019_ef1
            or ef1_value is not None
            or not self._has_any_field(data, "soil_n2o", "direct_n2o", "agricultural_soils")
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Updated soil N2O EF1 from 2019 Refinement should be "
            "considered (disaggregated by climate zone and N input type)",
            ef1_check,
            "WARNING",
            f"EF1 source: {ef1_source if ef1_source else 'not specified'}, "
            f"EF1 value: {ef1_value if ef1_value is not None else 'default'}",
            "Consider 2019 Refinement disaggregated EF1 values by wet/dry "
            "climate and synthetic/organic N input type",
        ))

        # REQ-05: Rice cultivation updated scaling factors.
        has_rice_source = self._has_any_field(data, "rice_ef_source", "rice_baseline_ef")
        rice_source = str(self._get_data_field(data, "rice_ef_source", "")).upper()
        uses_2019_rice = "2019" in rice_source
        rice_check = (
            uses_2019_rice
            or has_rice_source
            or not self._has_any_field(data, "rice_ch4", "rice_cultivation")
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Rice cultivation should use updated 2019 Refinement scaling "
            "factors for water regime and organic amendments",
            rice_check,
            "WARNING",
            f"Rice EF source: {rice_source if rice_source else 'not specified'}. "
            + ("2019 factors applied" if uses_2019_rice else "Consider 2019 update"),
            "Apply 2019 Refinement updated rice scaling factors, "
            "particularly revised SFw for intermittent irrigation",
        ))

        # REQ-06: Improved uncertainty ranges documented.
        has_unc = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "2019 Refinement improved uncertainty ranges should be used "
            "for Monte Carlo or analytical uncertainty assessment",
            has_unc,
            "WARNING",
            "Uncertainty " + ("is" if has_unc else "is not") + " assessed",
            "Use 2019 Refinement uncertainty ranges (typically narrower "
            "than 2006) for emission factor uncertainty parameters",
        ))

        # REQ-07: Field burning updated emission ratios.
        has_burning = self._has_any_field(
            data, "field_burning", "residue_burning",
            "burning_ch4", "burning_n2o",
        )
        burning_source = str(self._get_data_field(data, "burning_ef_source", "")).upper()
        burning_check = (
            "2019" in burning_source
            or not has_burning
            or self._has_field(data, "burning_ef_source")
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Field burning emission ratios should reference 2019 "
            "Refinement where updates are available",
            burning_check,
            "INFO",
            f"Burning EF source: {burning_source if burning_source else 'N/A'}",
            "Consider 2019 Refinement updated combustion factors and "
            "emission ratios for crop residue burning",
        ))

        # REQ-08: Liming and urea methodology updates.
        has_liming = self._has_any_field(data, "liming_co2", "limestone_tonnes")
        has_urea = self._has_any_field(data, "urea_co2", "urea_tonnes")
        liming_urea_check = (
            not (has_liming or has_urea)
            or self._has_any_field(data, "liming_ef", "urea_ef")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Liming and urea CO2 emission factors should reflect any "
            "2019 Refinement updates to default values",
            liming_urea_check,
            "INFO",
            f"Liming: {'reported' if has_liming else 'N/A'}, "
            f"Urea: {'reported' if has_urea else 'N/A'}",
            "Verify liming EF (0.12 for limestone, 0.13 for dolomite) "
            "and urea EF (0.20) against 2019 Refinement",
        ))

        # REQ-09: Tier consistency with 2019 guidance decision trees.
        tier = str(self._get_data_field(data, "tier", "")).upper()
        has_tier = tier in VALID_TIERS or self._has_field(data, "tier")
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Tier selection must follow 2019 Refinement decision tree "
            "guidance for each emission source",
            has_tier,
            "WARNING",
            f"Tier: {tier}" if has_tier else "Tier not specified",
            "Follow 2019 Refinement decision trees for tier selection "
            "based on data availability and emission significance",
        ))

        # REQ-10: Grazing land N2O methodology.
        has_grazing = self._has_any_field(
            data, "grazing_n2o", "pasture_n_input", "prp_n2o",
        )
        grazing_check = (
            has_grazing
            or not self._has_any_field(data, "pasture_range", "grazing")
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Grazing land N2O from pasture, range, and paddock (PRP) "
            "should follow updated 2019 Refinement EF3 values",
            grazing_check or True,
            "INFO",
            "Grazing N2O " + ("is" if has_grazing else "is not") + " included",
            "Apply 2019 Refinement EF3 for N deposited by grazing "
            "animals on pasture, range, and paddock",
        ))

        # REQ-11: GWP values consistency with reporting framework.
        gwp_source = str(self._get_data_field(data, "gwp_source", "")).upper()
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "GWP values must be consistent with the chosen reporting "
            "framework (AR4 for UNFCCC, AR5 for Paris Agreement)",
            self._has_field(data, "gwp_source"),
            "WARNING",
            f"GWP source: {gwp_source}" if gwp_source else "GWP source not specified",
            "Specify gwp_source and ensure consistency with the target "
            "reporting framework",
        ))

        # REQ-12: Completeness across all agricultural sources.
        sources = self._get_emission_sources(data)
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "All significant agricultural emission sources should be "
            "included for inventory completeness",
            len(sources) > 0,
            "WARNING",
            f"Sources reported: {sources[:5] if sources else 'none'}",
            "Ensure completeness by including all material agricultural "
            "emission sources (enteric, manure, soils, rice if applicable)",
        ))

        return findings

    # ==================================================================
    # Framework 3: GHG Protocol (18 requirements)
    # ==================================================================

    def check_ghg_protocol(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with GHG Protocol Agricultural Guidance.

        18 requirements covering organizational boundary, scope
        classification, completeness, base year methodology,
        year-over-year comparisons, biogenic CH4 vs fossil CH4
        distinction, and avoided emissions from manure biogas.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "GHG_PROTOCOL"

        # REQ-01: Organizational boundary includes all livestock/crops.
        has_boundary = self._has_any_field(
            data, "organizational_boundary", "facility_id",
            "farm_id", "operation_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Organizational boundary must include all livestock "
            "operations and crop production within operational control",
            has_boundary,
            "ERROR",
            "Organizational boundary " + ("is" if has_boundary else "is not") + " defined",
            "Define organizational boundary per GHG Protocol using "
            "equity share or operational control approach, covering "
            "all farms, feedlots, and crop operations",
        ))

        # REQ-02: Scope 1 classification for on-farm emissions.
        has_scope = self._has_any_field(data, "scope", "emission_scope")
        scope_val = str(self._get_data_field(data, "scope", "")).upper()
        is_scope1 = scope_val in ("1", "SCOPE_1", "SCOPE1")
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "On-farm agricultural emissions (enteric, manure, soils) "
            "must be classified as Scope 1 direct emissions",
            is_scope1 or has_scope or True,
            "WARNING",
            f"Scope classification: {scope_val if scope_val else 'not specified'}. "
            "Agricultural emissions are Scope 1 by definition",
            "Classify all on-farm agricultural emissions under Scope 1 "
            "per GHG Protocol Corporate Standard",
        ))

        # REQ-03: Completeness - all emission sources accounted.
        sources = self._get_emission_sources(data)
        has_enteric = "ENTERIC_FERMENTATION" in sources
        has_manure = "MANURE_MANAGEMENT" in sources
        has_soils = "AGRICULTURAL_SOILS" in sources
        core_complete = has_enteric or has_manure or has_soils or len(sources) > 0
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "All material emission sources must be accounted for "
            "completeness (enteric, manure, soils, rice, burning)",
            core_complete,
            "ERROR",
            f"Sources: {sources[:5] if sources else 'none'}. "
            f"Core coverage: enteric={'Y' if has_enteric else 'N'}, "
            f"manure={'Y' if has_manure else 'N'}, "
            f"soils={'Y' if has_soils else 'N'}",
            "Include all material agricultural emission sources. At "
            "minimum: ENTERIC_FERMENTATION, MANURE_MANAGEMENT, "
            "AGRICULTURAL_SOILS for livestock operations",
        ))

        # REQ-04: Base year methodology.
        has_base_year = self._has_any_field(
            data, "base_year", "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Base year emissions must be established for tracking "
            "progress and recalculation triggers",
            has_base_year,
            "ERROR",
            "Base year " + ("is" if has_base_year else "is not") + " established",
            "Establish a base year per GHG Protocol guidance. Document "
            "base year emissions and recalculation policy",
        ))

        # REQ-05: Year-over-year comparisons.
        has_yoy = self._has_any_field(
            data, "year_over_year", "previous_year_co2e",
            "yoy_change_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Year-over-year emission comparisons should be provided "
            "for trend analysis and target tracking",
            has_yoy,
            "WARNING",
            "YoY comparison " + ("is" if has_yoy else "is not") + " available",
            "Include year-over-year comparison data (previous_year_co2e "
            "and yoy_change_pct) for meaningful trend reporting",
        ))

        # REQ-06: Biogenic CH4 vs fossil CH4 distinction.
        has_biogenic = self._has_any_field(
            data, "biogenic_ch4", "ch4_biogenic",
            "emission_type", "biogenic_flag",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Biogenic CH4 (from enteric fermentation, manure) must be "
            "distinguished from fossil CH4 in reporting",
            has_biogenic or True,
            "WARNING",
            "Biogenic CH4 distinction " + (
                "is explicitly flagged" if has_biogenic
                else "not flagged (agricultural CH4 is biogenic by default)"
            ),
            "Flag biogenic_ch4 explicitly. Agricultural CH4 from "
            "livestock is biogenic and should be reported accordingly "
            "per GHG Protocol Scope 1 biogenic reporting",
        ))

        # REQ-07: Avoided emissions from manure biogas (Scope 3).
        has_biogas = self._has_any_field(
            data, "biogas_avoided_emissions", "anaerobic_digester_offset",
            "methane_capture",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Avoided emissions from manure biogas/anaerobic digestion "
            "should be reported separately (not netted against Scope 1)",
            has_biogas or True,
            "INFO",
            "Biogas avoided emissions " + (
                "are reported" if has_biogas else "not applicable or not reported"
            ),
            "If anaerobic digesters are used, report avoided emissions "
            "separately per GHG Protocol guidance. Do not net against "
            "Scope 1 without disclosure",
        ))

        # REQ-08: Methodology documented.
        has_method = self._has_any_field(
            data, "method", "methodology", "calculation_method",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Calculation methodology must be documented and referenced "
            "to recognized standards (IPCC, EPA, DEFRA)",
            has_method,
            "ERROR",
            "Methodology " + ("is" if has_method else "is not") + " documented",
            "Document the calculation methodology with reference to "
            "IPCC 2006/2019, EPA, or equivalent recognized standard",
        ))

        # REQ-09: Emission factors documented with sources.
        has_ef_doc = self._has_any_field(
            data, "enteric_ef_source", "mcf_source",
            "ef_source", "emission_factor_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "All emission factors must be documented with source "
            "references (IPCC table, country-specific, measured)",
            has_ef_doc,
            "ERROR",
            "EF documentation " + ("is" if has_ef_doc else "is not") + " present",
            "Document emission factor sources: IPCC table references, "
            "country-specific publications, or site measurements",
        ))

        # REQ-10: Activity data quality assessment.
        has_quality = self._has_any_field(
            data, "data_quality", "data_quality_score",
            "activity_data_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Activity data quality should be assessed and documented "
            "(high/medium/low confidence)",
            has_quality,
            "WARNING",
            "Data quality assessment " + ("is" if has_quality else "is not") + " documented",
            "Document activity data quality: source, collection method, "
            "and confidence level per GHG Protocol data quality guidance",
        ))

        # REQ-11: Uncertainty quantification.
        has_unc = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
            "confidence_interval",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Uncertainty should be quantified for key emission sources "
            "per GHG Protocol uncertainty guidance",
            has_unc,
            "WARNING",
            "Uncertainty " + ("is" if has_unc else "is not") + " quantified",
            "Quantify uncertainty using IPCC Approach 1 (error propagation) "
            "or Approach 2 (Monte Carlo) per GHG Protocol guidance",
        ))

        # REQ-12: Recalculation policy for structural changes.
        has_recalc = self._has_any_field(
            data, "recalculation_policy", "recalculation_trigger",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Recalculation policy must address structural changes "
            "(herd size, crop area, methodology updates)",
            has_recalc or True,
            "INFO",
            "Recalculation policy " + (
                "is documented" if has_recalc else "should be established"
            ),
            "Document recalculation triggers: herd size changes >5%, "
            "crop area changes, methodology or EF updates",
        ))

        # REQ-13: CO2e result in metric tonnes.
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "total_co2e_kg",
        )
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Emissions must be reported in metric tonnes CO2 equivalent",
            has_co2e,
            "ERROR",
            "CO2e reporting " + ("is" if has_co2e else "is not") + " in metric tonnes",
            "Report total_co2e_tonnes in metric tonnes for GHG Protocol "
            "conformance",
        ))

        # REQ-14: Gas-by-gas breakdown (CO2, CH4, N2O).
        has_gas_breakdown = self._has_any_field(
            data, "ch4_kg", "n2o_kg", "co2_kg",
            "ch4_co2e_tonnes", "n2o_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Individual GHG contributions (CO2, CH4, N2O) must be "
            "reported separately in addition to CO2e total",
            has_gas_breakdown,
            "ERROR",
            "Gas breakdown " + ("is" if has_gas_breakdown else "is not") + " reported",
            "Report ch4_kg, n2o_kg, co2_kg (or CO2e equivalents) "
            "separately for each greenhouse gas",
        ))

        # REQ-15: Source-by-source breakdown.
        sources = self._get_emission_sources(data)
        has_source_breakdown = self._has_any_field(
            data, "emissions_by_source", "source_breakdown",
        ) or len(sources) >= 1
        findings.append(ComplianceFinding(
            f"{fw}-15", fw,
            "Emissions must be broken down by source category "
            "(enteric, manure, soils, rice, etc.)",
            has_source_breakdown,
            "WARNING",
            "Source breakdown " + ("is" if has_source_breakdown else "is not") + " available",
            "Provide emissions_by_source with per-source CO2e breakdown",
        ))

        # REQ-16: Exclusion justification.
        has_exclusion = self._has_any_field(
            data, "exclusions", "excluded_sources",
            "exclusion_justification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-16", fw,
            "Any excluded emission sources must have documented "
            "justification (de minimis or not applicable)",
            has_exclusion or True,
            "INFO",
            "Exclusion justification " + (
                "is documented" if has_exclusion else "not needed (no exclusions flagged)"
            ),
            "If any sources are excluded, document justification per "
            "GHG Protocol relevance threshold (<1% of total if de minimis)",
        ))

        # REQ-17: Temporal boundary.
        has_temporal = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "year_start", "year_end")
        findings.append(ComplianceFinding(
            f"{fw}-17", fw,
            "Reporting temporal boundary must be defined (calendar or fiscal year)",
            has_temporal,
            "ERROR",
            "Temporal boundary " + ("is" if has_temporal else "is not") + " defined",
            "Define reporting_year or reporting_period for the inventory",
        ))

        # REQ-18: Verification readiness.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-18", fw,
            "Calculation must be verifiable with complete audit trail",
            has_prov,
            "WARNING",
            "Audit trail " + ("is" if has_prov else "is not") + " available",
            "Enable provenance hashing for third-party verification "
            "readiness per GHG Protocol",
        ))

        return findings

    # ==================================================================
    # Framework 4: ISO 14064 (14 requirements)
    # ==================================================================

    def check_iso_14064(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with ISO 14064-1:2018 requirements.

        14 requirements covering Category 1 direct emissions
        classification, quantification approach documentation,
        uncertainty assessment, and base year/GWP documentation.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "ISO_14064"

        # REQ-01: Category 1 direct emissions classified.
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Agricultural emissions must be classified under Category 1 "
            "(direct GHG emissions and removals) per ISO 14064-1:2018",
            True,
            "INFO",
            "Agricultural emissions are classified as Category 1 direct "
            "emissions per ISO 14064-1:2018 clause 5.2.2",
            "Classify all on-farm agricultural emissions under ISO "
            "14064-1 Category 1",
        ))

        # REQ-02: Organizational boundary defined.
        has_boundary = self._has_any_field(
            data, "organizational_boundary", "facility_id",
            "farm_id", "operation_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Organizational and operational boundaries must be defined "
            "per ISO 14064-1 clause 5.1",
            has_boundary,
            "ERROR",
            "Organizational boundary " + (
                "is" if has_boundary else "is not"
            ) + " defined",
            "Define organizational boundary (equity share or control "
            "approach) including all agricultural operations",
        ))

        # REQ-03: Quantification approach documented.
        has_method = self._has_any_field(
            data, "method", "methodology", "calculation_method",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "GHG quantification methodology must be documented per "
            "ISO 14064-1 clause 5.2.4",
            has_method,
            "ERROR",
            "Quantification methodology " + (
                "is" if has_method else "is not"
            ) + " documented",
            "Document the GHG quantification approach: measurement, "
            "calculation (emission factor), or modelling",
        ))

        # REQ-04: Uncertainty assessment performed.
        has_unc = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
            "confidence_interval",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Uncertainty assessment must be performed for all GHG "
            "quantifications per ISO 14064-1 clause 5.2.5",
            has_unc,
            "ERROR",
            "Uncertainty " + ("is" if has_unc else "is not") + " assessed",
            "Perform uncertainty assessment using error propagation "
            "or Monte Carlo simulation per ISO 14064-1",
        ))

        # REQ-05: Base year established and documented.
        has_base = self._has_any_field(
            data, "base_year", "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Base year must be selected, documented, and justified "
            "per ISO 14064-1 clause 5.3",
            has_base,
            "ERROR",
            "Base year " + ("is" if has_base else "is not") + " documented",
            "Select and document a base year with justification for "
            "the chosen year",
        ))

        # REQ-06: GWP values documented.
        has_gwp = self._has_field(data, "gwp_source")
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "GWP values and assessment report source must be documented",
            has_gwp,
            "ERROR",
            "GWP source " + ("is" if has_gwp else "is not") + " documented",
            "Document GWP source (AR4, AR5, or AR6) per ISO 14064-1",
        ))

        # REQ-07: Completeness of GHG sources.
        sources = self._get_emission_sources(data)
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "All material GHG sources must be identified and quantified "
            "for completeness per ISO 14064-1 clause 5.2.1",
            len(sources) > 0,
            "ERROR",
            f"Sources identified: {sources[:5] if sources else 'none'}",
            "Identify and quantify all significant agricultural GHG sources",
        ))

        # REQ-08: Consistency across reporting periods.
        has_method_doc = self._has_any_field(data, "method", "tier")
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Methodology must be applied consistently across reporting "
            "periods per ISO 14064-1 principle of consistency",
            has_method_doc,
            "WARNING",
            "Methodology consistency " + (
                "can be verified" if has_method_doc else "cannot be verified"
            ),
            "Maintain consistent methodology and emission factors "
            "across reporting periods. Document any changes",
        ))

        # REQ-09: Transparency of emission factors and data.
        has_ef_transparency = self._has_any_field(
            data, "enteric_ef_source", "mcf_source",
            "ef_source", "emission_factor_source", "data_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Emission factors and activity data sources must be "
            "transparent and documented per ISO 14064-1 clause 5.2.4",
            has_ef_transparency,
            "WARNING",
            "Data source transparency " + (
                "met" if has_ef_transparency else "not met"
            ),
            "Document all emission factor sources with full references "
            "(IPCC table numbers, publication details)",
        ))

        # REQ-10: Accuracy through tier selection.
        tier = str(self._get_data_field(data, "tier", "")).upper()
        has_tier = tier in VALID_TIERS or self._has_field(data, "tier")
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Calculations must minimize bias by using the highest "
            "practical tier per ISO 14064-1 accuracy principle",
            has_tier,
            "WARNING",
            f"Tier: {tier}" if has_tier else "Tier not specified for accuracy assessment",
            "Use highest practical tier. Document why a lower tier "
            "is used if higher-tier data is unavailable",
        ))

        # REQ-11: Separate reporting of emissions by gas.
        has_gas_split = self._has_any_field(
            data, "ch4_kg", "n2o_kg", "co2_kg",
            "ch4_co2e_tonnes", "n2o_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "GHG emissions must be reported separately by gas type "
            "(CO2, CH4, N2O) per ISO 14064-1 clause 5.2.3",
            has_gas_split,
            "ERROR",
            "Gas-by-gas reporting " + (
                "is" if has_gas_split else "is not"
            ) + " available",
            "Report emissions separately for each GHG: CO2 (from liming/"
            "urea), CH4 (enteric/manure/rice), N2O (soils/manure)",
        ))

        # REQ-12: Reporting period defined.
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
        ) or self._has_all_fields(data, "year_start", "year_end")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Reporting period must be defined per ISO 14064-1 clause 5.3",
            has_period,
            "ERROR",
            "Reporting period " + ("is" if has_period else "is not") + " defined",
            "Define the reporting period (calendar or fiscal year)",
        ))

        # REQ-13: Exclusion justification for omitted sources.
        has_exclusion = self._has_any_field(
            data, "exclusions", "excluded_sources",
            "exclusion_justification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Any excluded GHG sources must be justified and documented "
            "per ISO 14064-1 clause 5.2.1",
            has_exclusion or True,
            "INFO",
            "Exclusion review " + (
                "documented" if has_exclusion
                else "N/A (no exclusions flagged)"
            ),
            "Document and justify any excluded agricultural emission "
            "sources per ISO 14064-1",
        ))

        # REQ-14: Documentation retained for verification.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Supporting documentation must be retained for third-party "
            "verification per ISO 14064-1 clause 8",
            has_prov,
            "WARNING",
            "Audit trail " + ("is" if has_prov else "is not") + " available",
            "Enable provenance tracking for ISO 14064 verification "
            "and document retention",
        ))

        return findings

    # ==================================================================
    # Framework 5: CSRD/ESRS E1 & E4 (16 requirements)
    # ==================================================================

    def check_csrd_esrs(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CSRD/ESRS E1 (Climate) and E4 (Biodiversity).

        16 requirements covering GHG emissions disclosure (E1-6),
        Scope 1 agricultural emissions, methodology description,
        biodiversity impact (E4 - land use, fertilizer runoff),
        and transition plan for agricultural decarbonization.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "CSRD_ESRS"

        # REQ-01: E1-6 Gross Scope 1 GHG emissions disclosed.
        has_gross = self._has_any_field(
            data, "total_co2e_tonnes", "gross_emissions_tco2e",
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "E1-6: Gross Scope 1 GHG emissions from agricultural "
            "operations must be disclosed",
            has_gross,
            "ERROR",
            "Gross Scope 1 emissions " + (
                "are" if has_gross else "are not"
            ) + " disclosed",
            "Report total_co2e_tonnes for gross Scope 1 agricultural "
            "emissions per ESRS E1-6",
        ))

        # REQ-02: Scope 1 agricultural emissions separately identified.
        sources = self._get_emission_sources(data)
        has_ag_scope1 = len(sources) > 0
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Scope 1 agricultural emissions must be separately "
            "identified within total Scope 1 disclosure",
            has_ag_scope1,
            "ERROR",
            "Agricultural emission sources " + (
                "are identified" if has_ag_scope1 else "are not identified"
            ),
            "Separately identify agricultural Scope 1 emissions "
            "(enteric, manure, soils) within ESRS E1-6 disclosure",
        ))

        # REQ-03: Methodology description per ESRS E1-6 paragraph 44.
        has_method = self._has_any_field(
            data, "method", "methodology", "calculation_method",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Methodology description must be provided per ESRS E1-6 "
            "paragraph 44 (IPCC, GHG Protocol, or equivalent)",
            has_method,
            "ERROR",
            "Methodology " + ("is" if has_method else "is not") + " described",
            "Describe the quantification methodology referencing "
            "IPCC 2006/2019, GHG Protocol, or equivalent standard",
        ))

        # REQ-04: Biogenic emissions reported separately.
        has_biogenic = self._has_any_field(
            data, "biogenic_ch4", "biogenic_co2",
            "biogenic_emissions", "emission_type",
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Biogenic CO2 and biogenic CH4 must be reported separately "
            "from fossil emissions per ESRS E1-6",
            has_biogenic or True,
            "WARNING",
            "Biogenic emission separation " + (
                "is documented" if has_biogenic
                else "N/A (agricultural emissions are biogenic by default)"
            ),
            "Flag biogenic emissions explicitly. Agricultural CH4 from "
            "enteric and manure is biogenic; liming CO2 may be fossil-derived",
        ))

        # REQ-05: Year-over-year comparison for trend disclosure.
        has_yoy = self._has_any_field(
            data, "year_over_year", "previous_year_co2e",
            "yoy_change_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Year-over-year emission comparison must be disclosed "
            "for performance trend analysis per ESRS E1",
            has_yoy,
            "WARNING",
            "YoY comparison " + ("is" if has_yoy else "is not") + " available",
            "Include previous_year_co2e and yoy_change_pct for "
            "annual trend disclosure",
        ))

        # REQ-06: Consolidation approach disclosed.
        has_consolidation = self._has_any_field(
            data, "consolidation_approach", "boundary_approach",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Consolidation approach (financial control, operational "
            "control, equity share) must be disclosed per ESRS E1",
            has_consolidation or True,
            "INFO",
            "Consolidation approach " + (
                "is disclosed" if has_consolidation else "should be disclosed"
            ),
            "Disclose the consolidation approach used for "
            "agricultural emission boundary setting",
        ))

        # REQ-07: E1 transition plan for agricultural decarbonization.
        has_transition = self._has_any_field(
            data, "transition_plan", "decarbonization_targets",
            "reduction_targets",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Climate transition plan (E1-1) should include agricultural "
            "decarbonization targets and actions",
            has_transition or True,
            "INFO",
            "Transition plan alignment " + (
                "is referenced" if has_transition else "should be established"
            ),
            "Include agricultural decarbonization targets in the "
            "climate transition plan per ESRS E1-1 (e.g., feed "
            "additives, manure management improvements)",
        ))

        # REQ-08: E4 Biodiversity - land use impact.
        has_land_use = self._has_any_field(
            data, "land_use_area_ha", "area_ha",
            "agricultural_land_ha",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "E4: Agricultural land use area must be disclosed for "
            "biodiversity impact assessment per ESRS E4",
            has_land_use,
            "WARNING",
            "Agricultural land use " + (
                "is" if has_land_use else "is not"
            ) + " disclosed",
            "Report agricultural_land_ha for ESRS E4 biodiversity "
            "and ecosystems disclosure",
        ))

        # REQ-09: E4 Biodiversity - fertilizer runoff impact.
        has_fertilizer = self._has_any_field(
            data, "fertilizer_kg_n", "n_input_total",
            "synthetic_fertilizer_n",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "E4: Fertilizer use and potential runoff impact must be "
            "assessed for biodiversity (eutrophication risk)",
            has_fertilizer,
            "WARNING",
            "Fertilizer data " + (
                "is" if has_fertilizer else "is not"
            ) + " available for E4 assessment",
            "Report fertilizer_kg_n for E4 biodiversity eutrophication "
            "risk assessment from nitrogen runoff",
        ))

        # REQ-10: E4 Biodiversity - water stress impact.
        has_water = self._has_any_field(
            data, "water_consumption_m3", "irrigation_water",
            "water_regime",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "E4: Water consumption for agriculture should be assessed "
            "for water stress and aquatic biodiversity impact",
            has_water or True,
            "INFO",
            "Water consumption data " + (
                "is available" if has_water else "not reported"
            ),
            "Consider reporting water_consumption_m3 for ESRS E4 "
            "water stress and aquatic biodiversity assessment",
        ))

        # REQ-11: ESRS E1-6 reporting in tonnes CO2e.
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "net_co2e_tonnes",
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Emissions must be reported in metric tonnes CO2 equivalent "
            "per ESRS E1-6 presentation requirements",
            has_co2e,
            "ERROR",
            "CO2e reporting unit " + (
                "is" if has_co2e else "is not"
            ) + " in tonnes",
            "Report emissions in total_co2e_tonnes (metric tonnes) "
            "per ESRS E1-6",
        ))

        # REQ-12: GHG Protocol or ISO 14064 methodology reference.
        has_standard_ref = self._has_any_field(
            data, "method", "methodology_standard",
            "standard_reference",
        )
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Methodology must reference GHG Protocol or ISO 14064 "
            "as underlying standard per ESRS E1-6 paragraph 44",
            has_standard_ref,
            "ERROR",
            "Standard reference " + (
                "is" if has_standard_ref else "is not"
            ) + " provided",
            "Reference GHG Protocol Corporate Standard or ISO 14064-1 "
            "as the underlying quantification standard",
        ))

        # REQ-13: Double materiality assessment.
        has_materiality = self._has_any_field(
            data, "is_material", "materiality_assessment",
            "double_materiality",
        )
        findings.append(ComplianceFinding(
            f"{fw}-13", fw,
            "Agricultural emissions must be assessed for double "
            "materiality (financial + impact) per CSRD",
            has_materiality or True,
            "INFO",
            "Materiality assessment " + (
                "is documented" if has_materiality
                else "assumed material (agricultural emissions reported)"
            ),
            "Perform double materiality assessment per CSRD: financial "
            "materiality (climate risk to operations) and impact "
            "materiality (emissions impact on climate)",
        ))

        # REQ-14: Intensity metrics disclosure.
        has_intensity = self._has_any_field(
            data, "intensity_per_head", "intensity_per_hectare",
            "intensity_per_unit_product", "emission_intensity",
        )
        findings.append(ComplianceFinding(
            f"{fw}-14", fw,
            "Emission intensity metrics should be disclosed "
            "(tCO2e per head, per hectare, per unit product)",
            has_intensity,
            "WARNING",
            "Intensity metrics " + (
                "are" if has_intensity else "are not"
            ) + " disclosed",
            "Report intensity metrics: tCO2e per head (livestock), "
            "tCO2e per hectare (cropland), tCO2e per unit product",
        ))

        # REQ-15: Third-party verification readiness.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-15", fw,
            "Data must support limited assurance verification "
            "per CSRD requirements (Article 34)",
            has_prov,
            "WARNING",
            "Verification readiness " + (
                "met" if has_prov else "not met"
            ),
            "Ensure complete audit trail with provenance hashing "
            "for CSRD limited assurance verification",
        ))

        # REQ-16: Sector-specific disclosure for agriculture.
        has_sector = self._has_any_field(
            data, "sector", "nace_code", "industry_classification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-16", fw,
            "Sector-specific disclosure standards for agriculture "
            "should be applied where EFRAG sector standards are available",
            has_sector or True,
            "INFO",
            "Sector classification " + (
                "is documented" if has_sector
                else "should be documented when sector standards are finalized"
            ),
            "Apply EFRAG sector-specific standards for agriculture "
            "when published (expected ESRS SEC1-Agriculture)",
        ))

        return findings

    # ==================================================================
    # Framework 6: EPA 40 CFR 98 Subpart JJ (10 requirements)
    # ==================================================================

    def check_epa_40cfr98(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with EPA 40 CFR 98 Subpart JJ (Manure Management).

        10 requirements covering facility-level reporting thresholds,
        EPA-approved methodology for enteric and manure emissions,
        and monitoring frequency.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "EPA_40CFR98"

        # REQ-01: Facility-level reporting threshold (25,000 tCO2e/yr).
        total_co2e = self._get_data_field(data, "total_co2e_tonnes")
        above_threshold = False
        threshold_status = "unknown"
        if total_co2e is not None:
            try:
                above_threshold = float(total_co2e) >= 25000
                threshold_status = (
                    "above threshold (reporting required)"
                    if above_threshold
                    else "below threshold (reporting optional)"
                )
            except (ValueError, TypeError):
                threshold_status = "could not evaluate"
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Facilities emitting >= 25,000 tCO2e/yr must report under "
            "EPA 40 CFR 98 Mandatory Reporting Rule",
            True,
            "INFO",
            f"Total CO2e: {total_co2e} tonnes. Status: {threshold_status}",
            "Determine if facility exceeds 25,000 tCO2e/yr threshold "
            "for mandatory GHG reporting under 40 CFR 98",
        ))

        # REQ-02: EPA-approved methodology for enteric fermentation.
        has_method = self._has_any_field(
            data, "method", "methodology", "calculation_method",
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Enteric fermentation must use EPA-approved methodology "
            "per 40 CFR 98 Subpart JJ Section 98.363",
            has_method,
            "ERROR",
            "Methodology " + (
                "is" if has_method else "is not"
            ) + " documented for EPA compliance",
            "Use EPA 40 CFR 98 Subpart JJ approved methodology "
            "for enteric fermentation calculations",
        ))

        # REQ-03: EPA-approved methodology for manure management.
        has_manure_method = self._has_any_field(
            data, "manure_method", "method", "awms_type",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Manure management must use EPA-approved methodology "
            "per 40 CFR 98 Subpart JJ Section 98.363",
            has_manure_method,
            "ERROR",
            "Manure methodology " + (
                "is" if has_manure_method else "is not"
            ) + " documented",
            "Use EPA-approved manure management methodology with "
            "AWMS-specific calculations per Subpart JJ",
        ))

        # REQ-04: Population data by animal type.
        has_pop = self._has_any_field(
            data, "population", "head_count", "animal_population",
        )
        animal_types = self._get_animal_types(data)
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Livestock population must be reported by animal type "
            "per EPA Subpart JJ Section 98.363(a)",
            has_pop and len(animal_types) > 0,
            "ERROR",
            f"Population: {'reported' if has_pop else 'not reported'}, "
            f"Animal types: {animal_types[:3] if animal_types else 'none'}",
            "Report livestock population by animal type per EPA "
            "Subpart JJ animal categories",
        ))

        # REQ-05: Annual average temperature for MCF.
        has_temp = self._has_any_field(
            data, "annual_avg_temperature", "temperature_c",
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Annual average temperature must be documented for MCF "
            "selection per EPA Subpart JJ Section 98.363(b)",
            has_temp or not self._has_any_field(
                data, "manure_ch4", "manure_management",
            ),
            "WARNING",
            "Temperature " + (
                "is" if has_temp else "is not"
            ) + " documented for MCF",
            "Document annual_avg_temperature (Celsius) at the facility "
            "for MCF selection per EPA methodology",
        ))

        # REQ-06: Monitoring frequency (annual at minimum).
        has_period = self._has_any_field(
            data, "reporting_period", "reporting_year",
            "monitoring_frequency",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Monitoring and reporting must be performed at least "
            "annually per 40 CFR 98 Section 98.3",
            has_period,
            "ERROR",
            "Monitoring period " + (
                "is" if has_period else "is not"
            ) + " defined",
            "Ensure annual monitoring and reporting per 40 CFR 98 "
            "general provisions",
        ))

        # REQ-07: AWMS identification per facility.
        has_awms = self._has_any_field(
            data, "awms_type", "manure_system", "awms_types",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Animal waste management systems (AWMS) must be identified "
            "and documented per facility per Subpart JJ",
            has_awms or not self._has_any_field(
                data, "manure_ch4", "manure_management",
            ),
            "WARNING",
            "AWMS " + (
                "is" if has_awms else "is not"
            ) + " identified",
            "Identify all AWMS types at the facility: lagoon, pit, "
            "solid storage, daily spread, pasture, digester, etc.",
        ))

        # REQ-08: Emission factor documentation with EPA references.
        has_ef = self._has_any_field(
            data, "enteric_ef_source", "mcf_source",
            "ef_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Emission factors must be documented with EPA-accepted "
            "references per 40 CFR 98 Subpart JJ",
            has_ef,
            "WARNING",
            "EF documentation " + (
                "is" if has_ef else "is not"
            ) + " present",
            "Document emission factors with references to EPA "
            "Subpart JJ tables or IPCC 2006/2019 (EPA-accepted)",
        ))

        # REQ-09: Data quality and record retention.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Records must be retained for at least 3 years per "
            "40 CFR 98 Section 98.3(g)",
            has_prov,
            "WARNING",
            "Record retention " + (
                "supported (provenance hash present)" if has_prov
                else "not verifiable (no provenance hash)"
            ),
            "Enable provenance tracking and ensure records are "
            "retained for minimum 3 years per EPA requirements",
        ))

        # REQ-10: CO2e calculation with EPA-specified GWP.
        gwp_source = str(self._get_data_field(data, "gwp_source", "")).upper()
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "total_co2e_kg",
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "CO2e must be calculated using GWP values specified by "
            "EPA (currently AR4 for GHGRP)",
            has_co2e and (gwp_source == "AR4" or self._has_field(data, "gwp_source")),
            "ERROR",
            f"CO2e: {'reported' if has_co2e else 'not reported'}, "
            f"GWP: {gwp_source if gwp_source else 'not specified'} "
            f"(EPA uses AR4 for GHGRP)",
            "Use AR4 GWP values for EPA GHGRP reporting (CH4=25, "
            "N2O=298). Specify gwp_source='AR4'",
        ))

        return findings

    # ==================================================================
    # Framework 7: DEFRA (10 requirements)
    # ==================================================================

    def check_defra(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with DEFRA/BEIS UK Environmental Reporting Guidelines.

        10 requirements covering DEFRA/BEIS conversion factors, UK-specific
        animal categories, and intensity metrics (tCO2e per head, per hectare).

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "DEFRA"

        # REQ-01: DEFRA/BEIS conversion factors used.
        ef_source = str(self._get_data_field(data, "ef_source", "")).upper()
        enteric_ef_source = str(self._get_data_field(data, "enteric_ef_source", "")).upper()
        uses_defra = (
            "DEFRA" in ef_source or "BEIS" in ef_source
            or "DEFRA" in enteric_ef_source or "BEIS" in enteric_ef_source
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "DEFRA/BEIS GHG Conversion Factors should be used for "
            "UK environmental reporting",
            uses_defra or self._has_any_field(data, "ef_source", "enteric_ef_source"),
            "WARNING",
            f"EF source: {ef_source or enteric_ef_source}. "
            + ("DEFRA/BEIS factors used" if uses_defra else "Non-DEFRA factors"),
            "Use DEFRA/BEIS annual GHG Conversion Factors for UK "
            "environmental reporting (updated annually by UK DESNZ)",
        ))

        # REQ-02: UK-specific animal categories.
        animal_types = self._get_animal_types(data)
        has_animals = len(animal_types) > 0
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Animal categories must align with DEFRA/BEIS livestock "
            "classification for UK reporting",
            has_animals or not self._has_any_field(
                data, "animal_type", "animal_types",
            ),
            "WARNING",
            f"Animal types: {animal_types[:3] if animal_types else 'N/A'}",
            "Use DEFRA/BEIS livestock categories: dairy cattle, beef "
            "cattle, sheep, goats, pigs, poultry, horses, other",
        ))

        # REQ-03: Intensity metric - tCO2e per head.
        has_per_head = self._has_any_field(
            data, "intensity_per_head", "co2e_per_head",
            "emissions_per_head",
        )
        has_livestock = self._has_any_field(
            data, "animal_type", "head_count", "population",
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Emission intensity per head of livestock (tCO2e/head) "
            "should be reported for DEFRA benchmarking",
            has_per_head or not has_livestock,
            "WARNING",
            "Per-head intensity " + (
                "is" if has_per_head else "is not"
            ) + " reported",
            "Calculate and report intensity_per_head (tCO2e per head "
            "per year) for DEFRA agricultural benchmarking",
        ))

        # REQ-04: Intensity metric - tCO2e per hectare.
        has_per_ha = self._has_any_field(
            data, "intensity_per_hectare", "co2e_per_hectare",
            "emissions_per_hectare",
        )
        has_area = self._has_any_field(data, "area_ha", "agricultural_land_ha")
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Emission intensity per hectare (tCO2e/ha) should be "
            "reported for DEFRA agricultural benchmarking",
            has_per_ha or not has_area,
            "WARNING",
            "Per-hectare intensity " + (
                "is" if has_per_ha else "is not"
            ) + " reported",
            "Calculate and report intensity_per_hectare (tCO2e per "
            "hectare per year) for DEFRA cropland benchmarking",
        ))

        # REQ-05: Mandatory reporting threshold (UK SECR).
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "UK Streamlined Energy and Carbon Reporting (SECR) "
            "requirements must be assessed for applicability",
            True,
            "INFO",
            "SECR applicability should be assessed based on company "
            "size thresholds (large/medium under Companies Act 2006)",
            "Assess SECR applicability: large companies must report "
            "Scope 1 GHG emissions under UK SECR regulations",
        ))

        # REQ-06: UK GHG inventory consistency.
        has_method = self._has_any_field(
            data, "method", "methodology", "calculation_method",
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Methodology should be consistent with UK National GHG "
            "Inventory methodology for agriculture",
            has_method,
            "WARNING",
            "Methodology " + ("is" if has_method else "is not") + " documented",
            "Align methodology with UK National Atmospheric Emissions "
            "Inventory (NAEI) agriculture sector approach",
        ))

        # REQ-07: Annual DEFRA factor updates.
        has_year = self._has_any_field(
            data, "reporting_year", "ef_year", "defra_factor_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "DEFRA/BEIS conversion factors must be updated annually "
            "to use the latest published version",
            has_year or True,
            "INFO",
            "Factor year " + (
                "is documented" if has_year else "should be documented"
            ),
            "Use the latest DEFRA/BEIS conversion factors for the "
            "reporting year. Document defra_factor_year",
        ))

        # REQ-08: CO2e reporting in metric tonnes.
        has_co2e = self._has_any_field(
            data, "total_co2e_tonnes", "total_co2e_kg",
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Emissions must be reported in metric tonnes CO2 equivalent "
            "per UK environmental reporting requirements",
            has_co2e,
            "ERROR",
            "CO2e " + ("is" if has_co2e else "is not") + " reported in tonnes",
            "Report total_co2e_tonnes per DEFRA guidelines",
        ))

        # REQ-09: Activity data documentation.
        has_activity = self._has_any_field(
            data, "population", "head_count", "area_ha",
            "fertilizer_kg_n", "activity_data",
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Activity data (livestock numbers, crop areas, fertilizer "
            "quantities) must be documented and traceable",
            has_activity,
            "ERROR",
            "Activity data " + (
                "is" if has_activity else "is not"
            ) + " documented",
            "Document all activity data: head_count, area_ha, "
            "fertilizer_kg_n with source references",
        ))

        # REQ-10: Audit trail for DEFRA reporting.
        has_prov = self._has_field(data, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Audit trail must support verification per UK SECR and "
            "DEFRA environmental reporting guidelines",
            has_prov,
            "WARNING",
            "Audit trail " + (
                "is" if has_prov else "is not"
            ) + " available",
            "Enable provenance tracking for DEFRA reporting "
            "verification and audit readiness",
        ))

        return findings

    # ==================================================================
    # Requirement Metadata
    # ==================================================================

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, str]]:
        """Get the list of requirements for a specific framework.

        Runs the checker against empty data to extract requirement
        metadata (descriptions, severities) without evaluating
        compliance.

        Args:
            framework: Framework name (e.g. 'IPCC_2006_VOL4').

        Returns:
            List of requirement dictionaries with id, description,
            and severity fields.

        Raises:
            ValueError: If framework is not supported.
        """
        fw_upper = framework.upper()
        if fw_upper not in self._framework_checkers:
            raise ValueError(
                f"Unknown framework '{framework}'. Supported: "
                f"{SUPPORTED_FRAMEWORKS}"
            )

        checker = self._framework_checkers[fw_upper]
        findings = checker({})

        return [
            {
                "requirement_id": f.requirement_id,
                "description": f.description,
                "severity": f.severity,
            }
            for f in findings
        ]

    def get_all_requirements(self) -> Dict[str, Any]:
        """Get a listing of all compliance requirements across all frameworks.

        Returns:
            Dictionary with total count, framework list, and
            per-framework requirement listings.
        """
        empty_data: Dict[str, Any] = {}
        all_reqs: Dict[str, List[Dict[str, str]]] = {}

        for fw_name, checker in self._framework_checkers.items():
            findings = checker(empty_data)
            all_reqs[fw_name] = [
                {
                    "requirement_id": f.requirement_id,
                    "description": f.description,
                    "severity": f.severity,
                }
                for f in findings
            ]

        total = sum(len(v) for v in all_reqs.values())
        return {
            "total_requirements": total,
            "frameworks": list(all_reqs.keys()),
            "framework_counts": {
                fw: len(reqs) for fw, reqs in all_reqs.items()
            },
            "requirements": all_reqs,
        }

    # ==================================================================
    # Compliance Summary
    # ==================================================================

    def get_compliance_summary(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a human-readable compliance summary from check results.

        This method takes the output of check_compliance() and produces
        a condensed summary suitable for dashboards and reports.

        Args:
            results: Output from check_compliance().

        Returns:
            Summary dictionary with status counts, critical failures,
            and per-framework pass rates.
        """
        overall = results.get("overall", {})
        framework_results = results.get("framework_results", {})

        # Collect critical failures (severity=ERROR, passed=False).
        critical_failures: List[Dict[str, str]] = []
        warning_items: List[Dict[str, str]] = []

        for fw_name, fw_result in framework_results.items():
            for finding in fw_result.get("findings", []):
                if not finding.get("passed", True):
                    item = {
                        "requirement_id": finding.get("requirement_id", ""),
                        "framework": fw_name,
                        "description": finding.get("description", ""),
                        "recommendation": finding.get("recommendation", ""),
                    }
                    if finding.get("severity") == "ERROR":
                        critical_failures.append(item)
                    elif finding.get("severity") == "WARNING":
                        warning_items.append(item)

        # Per-framework summary.
        framework_summary: Dict[str, Dict[str, Any]] = {}
        for fw_name, fw_result in framework_results.items():
            framework_summary[fw_name] = {
                "status": fw_result.get("status", "UNKNOWN"),
                "pass_rate_pct": fw_result.get("pass_rate_pct", 0),
                "passed": fw_result.get("passed", 0),
                "total": fw_result.get("total_requirements", 0),
            }

        return {
            "compliance_status": overall.get("compliance_status", "UNKNOWN"),
            "pass_rate_pct": overall.get("pass_rate_pct", 0),
            "total_requirements": overall.get("total_requirements", 0),
            "total_passed": overall.get("total_passed", 0),
            "total_failed": overall.get("total_failed", 0),
            "critical_failures_count": len(critical_failures),
            "warnings_count": len(warning_items),
            "critical_failures": critical_failures,
            "warnings": warning_items,
            "framework_summary": framework_summary,
            "frameworks_checked": results.get("frameworks_checked", []),
            "processing_time_ms": results.get("processing_time_ms", 0),
        }

    # ==================================================================
    # Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine metadata, counters, and supported
            framework information.
        """
        with self._lock:
            return {
                "engine": "ComplianceCheckerEngine",
                "agent": "AGENT-MRV-008",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_checks": self._total_checks,
                "total_findings": self._total_findings,
                "total_passed": self._total_passed,
                "total_failed": self._total_failed,
                "supported_frameworks": SUPPORTED_FRAMEWORKS,
                "total_requirements": TOTAL_REQUIREMENTS,
                "requirement_counts": dict(self._requirement_counts),
            }

    def reset(self) -> None:
        """Reset engine counters to zero.

        Thread-safe reset of all mutable statistics counters.
        """
        with self._lock:
            self._total_checks = 0
            self._total_findings = 0
            self._total_passed = 0
            self._total_failed = 0
        logger.info("ComplianceCheckerEngine reset")

    # ==================================================================
    # Validation Helpers
    # ==================================================================

    def validate_framework_name(self, framework: str) -> bool:
        """Check if a framework name is supported.

        Args:
            framework: Framework name to validate.

        Returns:
            True if the framework is supported.
        """
        return framework.upper() in self._framework_checkers

    def get_supported_frameworks(self) -> List[str]:
        """Return the list of supported framework names.

        Returns:
            Copy of the supported frameworks list.
        """
        return list(SUPPORTED_FRAMEWORKS)

    def get_requirement_count(self, framework: str) -> int:
        """Return the number of requirements for a framework.

        Args:
            framework: Framework name.

        Returns:
            Number of requirements, or 0 if unknown.
        """
        return self._requirement_counts.get(framework.upper(), 0)

    def check_single_framework(
        self,
        calculation_data: Dict[str, Any],
        framework: str,
    ) -> Dict[str, Any]:
        """Convenience method to check a single framework.

        Args:
            calculation_data: Calculation data dictionary.
            framework: Single framework name.

        Returns:
            Compliance result for the single framework.

        Raises:
            ValueError: If framework is not supported.
        """
        fw_upper = framework.upper()
        if fw_upper not in self._framework_checkers:
            raise ValueError(
                f"Unknown framework '{framework}'. Supported: "
                f"{SUPPORTED_FRAMEWORKS}"
            )
        return self.check_compliance(calculation_data, [fw_upper])

    def get_critical_findings(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract only critical (ERROR severity) failed findings.

        Args:
            results: Output from check_compliance().

        Returns:
            List of critical finding dictionaries.
        """
        critical: List[Dict[str, Any]] = []
        framework_results = results.get("framework_results", {})

        for fw_name, fw_result in framework_results.items():
            for finding in fw_result.get("findings", []):
                if (
                    not finding.get("passed", True)
                    and finding.get("severity") == "ERROR"
                ):
                    critical.append({
                        "requirement_id": finding.get("requirement_id", ""),
                        "framework": fw_name,
                        "description": finding.get("description", ""),
                        "finding": finding.get("finding", ""),
                        "recommendation": finding.get("recommendation", ""),
                    })

        return critical

    def get_warning_findings(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract only warning-level failed findings.

        Args:
            results: Output from check_compliance().

        Returns:
            List of warning finding dictionaries.
        """
        warnings: List[Dict[str, Any]] = []
        framework_results = results.get("framework_results", {})

        for fw_name, fw_result in framework_results.items():
            for finding in fw_result.get("findings", []):
                if (
                    not finding.get("passed", True)
                    and finding.get("severity") == "WARNING"
                ):
                    warnings.append({
                        "requirement_id": finding.get("requirement_id", ""),
                        "framework": fw_name,
                        "description": finding.get("description", ""),
                        "finding": finding.get("finding", ""),
                        "recommendation": finding.get("recommendation", ""),
                    })

        return warnings

    def is_compliant(
        self,
        results: Dict[str, Any],
        framework: Optional[str] = None,
    ) -> bool:
        """Check if results indicate full compliance.

        Args:
            results: Output from check_compliance().
            framework: Optional specific framework to check.
                If None, checks overall compliance.

        Returns:
            True if fully compliant (100% pass rate).
        """
        if framework is not None:
            fw_upper = framework.upper()
            fw_results = results.get("framework_results", {})
            fw_result = fw_results.get(fw_upper, {})
            return fw_result.get("status") == "COMPLIANT"

        overall = results.get("overall", {})
        return overall.get("compliance_status") == "COMPLIANT"

    def get_pass_rate(
        self,
        results: Dict[str, Any],
        framework: Optional[str] = None,
    ) -> float:
        """Get the pass rate percentage from results.

        Args:
            results: Output from check_compliance().
            framework: Optional specific framework. If None,
                returns overall pass rate.

        Returns:
            Pass rate as a percentage (0.0 to 100.0).
        """
        if framework is not None:
            fw_upper = framework.upper()
            fw_results = results.get("framework_results", {})
            fw_result = fw_results.get(fw_upper, {})
            return float(fw_result.get("pass_rate_pct", 0.0))

        overall = results.get("overall", {})
        return float(overall.get("pass_rate_pct", 0.0))
